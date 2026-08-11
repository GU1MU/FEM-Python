"""Analysis run lifecycle and result provenance."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Mapping

from fem.core.result import ModelResult

from .results.data import (
    ResultMaterializationPatch,
    ResultMaterializationSnapshot,
)
from .results.execution import (
    OutputExecutionStatus,
    ResultExecutionReport,
)
from .results._ownership import (
    deep_owned_materialization,
    deep_owned_result,
)
from .results.provider import ResultProvider, restore_result_provider


B31_BEAM_FORMULATION = "abaqus-b31-linear-timoshenko-v1"
B31_RESULT_POSITION = "INTEGRATION_POINT"
B31_RECOVERY_CONTRACT = 4


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class RunStatus(str, Enum):
    """Lifecycle states for an in-session analysis run."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True)
class AnalysisRun:
    """One run tied to an exact model artifact and analysis step."""

    run_id: str
    name: str
    step_name: str
    artifact_id: str
    model_revision: int
    status: RunStatus = RunStatus.PENDING
    created_at: datetime = field(default_factory=utc_now)
    started_at: datetime | None = None
    finished_at: datetime | None = None
    source_run_id: str | None = None
    result_id: str | None = None
    error: str | None = None
    cancellation_requested: bool = False
    messages: tuple[str, ...] = ()
    timings: Mapping[str, float] = field(default_factory=dict)

    @property
    def has_result(self) -> bool:
        return self.status is RunStatus.SUCCEEDED and self.result_id is not None

    @property
    def elapsed_seconds(self) -> float | None:
        if self.started_at is None:
            return None
        end = utc_now() if self.finished_at is None else self.finished_at
        return max(0.0, (end - self.started_at).total_seconds())


@dataclass(frozen=True, slots=True)
class ResultProvenance:
    """Identity of the session inputs that produced a result."""

    session_id: str
    artifact_id: str
    model_revision: int
    step_name: str
    run_id: str
    beam_formulation: str | None = None
    beam_result_position: str | None = None
    beam_recovery_contract: int | None = None

    def __post_init__(self) -> None:
        formulation_values = (
            self.beam_formulation,
            self.beam_result_position,
            self.beam_recovery_contract,
        )
        if all(value is None for value in formulation_values):
            return
        if any(value is None for value in formulation_values):
            raise ValueError("Beam result provenance must be complete")
        if type(self.beam_formulation) is not str or not self.beam_formulation.strip():
            raise ValueError("beam_formulation must be a nonblank string")
        if (
            type(self.beam_result_position) is not str
            or not self.beam_result_position.strip()
        ):
            raise ValueError("beam_result_position must be a nonblank string")
        if (
            type(self.beam_recovery_contract) is not int
            or self.beam_recovery_contract < 1
        ):
            raise ValueError("beam_recovery_contract must be a positive integer")


def b31_result_provenance(model: object) -> dict[str, object]:
    """Return the formulation identity for a result containing Beam2."""

    mesh = getattr(model, "mesh", None)
    elements = getattr(mesh, "elements", ())
    if not any(str(getattr(element, "type", "")).casefold() == "beam2" for element in elements):
        return {}
    return {
        "beam_formulation": B31_BEAM_FORMULATION,
        "beam_result_position": B31_RESULT_POSITION,
        "beam_recovery_contract": B31_RECOVERY_CONTRACT,
    }


def archived_b31_result_provenance(
    provenance: Mapping[str, object],
) -> dict[str, object]:
    """Keep complete B31 archive provenance without relabelling legacy data."""

    keys = (
        "beam_formulation",
        "beam_result_position",
        "beam_recovery_contract",
    )
    if not all(key in provenance for key in keys):
        return {}
    return {key: provenance[key] for key in keys}


@dataclass(frozen=True, slots=True, eq=False)
class ResultRecord:
    """One solver result plus its complete provenance."""

    result_id: str
    provenance: ResultProvenance
    result: ModelResult | None
    output_report: ResultExecutionReport
    materialization: ResultMaterializationSnapshot
    created_at: datetime = field(default_factory=utc_now)
    _provider: ResultProvider | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if type(self.result_id) is not str or not self.result_id.strip():
            raise ValueError("result_id must be a nonblank string")
        if type(self.provenance) is not ResultProvenance:
            raise TypeError("provenance must be ResultProvenance")
        if self.result is not None and type(self.result) is not ModelResult:
            raise TypeError("result must be exactly ModelResult or None")
        if type(self.output_report) is not ResultExecutionReport:
            raise TypeError("output_report must be ResultExecutionReport")
        if (
            type(self.materialization)
            is not ResultMaterializationSnapshot
        ):
            raise TypeError(
                "materialization must be ResultMaterializationSnapshot"
            )

        source = self.output_report.source
        if self.materialization.source != source:
            raise ValueError(
                "output report and materialization sources must match"
            )
        expected_identity = (
            self.result_id,
            self.provenance.session_id,
            self.provenance.artifact_id,
            self.provenance.model_revision,
            self.provenance.step_name,
            self.provenance.run_id,
        )
        actual_identity = (
            source.result_id,
            source.session_id,
            source.artifact_id,
            source.model_revision,
            source.step_name,
            source.run_id,
        )
        if actual_identity != expected_identity:
            raise ValueError(
                "result source must match record identity and provenance"
            )
        materialized_keys = {
            field_data.key for field_data in self.materialization.fields
        }
        executed_keys = {
            key
            for request in self.output_report.requests
            for variable in request.variables
            if variable.status is OutputExecutionStatus.EXECUTED
            for key in variable.field_keys
        }
        if not executed_keys.issubset(materialized_keys):
            raise ValueError(
                "every executed output field must be materialized"
            )

        provider = self._provider
        if provider is None:
            if self.result is None:
                raise TypeError(
                    "result is required when no provider is supplied"
                )
            provider = restore_result_provider(
                self.result,
                self.materialization,
                published_keys=executed_keys,
            )
        else:
            if type(provider) is not ResultProvider:
                raise TypeError("_provider must be exactly ResultProvider")
            if provider.is_archived:
                if self.result is not None:
                    raise ValueError(
                        "archived provider records cannot carry ModelResult"
                    )
            elif provider._owned_result is not self.result:
                raise ValueError(
                    "_provider must own the exact record result"
                )
            if provider.snapshot is not self.materialization:
                raise ValueError(
                    "_provider must expose the exact record materialization"
                )
        object.__setattr__(self, "result", provider._owned_result)
        object.__setattr__(
            self,
            "output_report",
            deepcopy(self.output_report),
        )
        object.__setattr__(
            self,
            "materialization",
            provider.snapshot,
        )
        object.__setattr__(self, "_provider", provider)


def detached_result_record(record: ResultRecord) -> ResultRecord:
    """Return a detached record whose public result vectors stay readonly."""

    if type(record) is not ResultRecord:
        raise TypeError("record must be exactly ResultRecord")
    provider = result_record_provider(record)
    if provider.is_archived:
        return ResultRecord(
            result_id=record.result_id,
            provenance=deepcopy(record.provenance),
            result=None,
            output_report=record.output_report,
            materialization=record.materialization,
            created_at=record.created_at,
            _provider=provider,
        )
    if record.result is None:
        raise RuntimeError("live result record has no ModelResult")
    owned_result = deep_owned_result(record.result)
    owned_materialization = deep_owned_materialization(
        record.materialization
    )
    detached_provider = ResultProvider(
        _owned_result=owned_result,
        _profile=provider.profile,
        _catalog=provider.catalog(),
        _snapshot=owned_materialization,
    )
    return ResultRecord(
        result_id=record.result_id,
        provenance=deepcopy(record.provenance),
        result=owned_result,
        output_report=record.output_report,
        materialization=owned_materialization,
        created_at=record.created_at,
        _provider=detached_provider,
    )


def result_record_provider(record: ResultRecord) -> ResultProvider:
    """Return the immutable provider owned by one already accepted record."""

    if type(record) is not ResultRecord:
        raise TypeError("record must be exactly ResultRecord")
    provider = record._provider
    if type(provider) is not ResultProvider:
        raise RuntimeError("accepted result record has no provider")
    return provider


def advance_result_record(
    record: ResultRecord,
    patch: ResultMaterializationPatch,
) -> ResultRecord:
    """Advance one accepted record without restoring its owned result graph."""

    provider = result_record_provider(record).advance(patch)
    return ResultRecord(
        result_id=record.result_id,
        provenance=record.provenance,
        result=provider._owned_result,
        output_report=record.output_report,
        materialization=provider.snapshot,
        created_at=record.created_at,
        _provider=provider,
    )

