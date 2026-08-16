"""A6 deterministic preflight identity and native solve proposals."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Protocol

from fem.application import (
    PreflightReport,
    PreflightSeverity,
    UnitContext,
    ValidationRecord,
)

from .authoring import (
    AgentProposal,
    ModelOperation,
    OperationKind,
    ProposalKind,
)
from .naming import NamePolicy


class SolveAuthoringError(ValueError):
    """The current local model cannot produce an executable A6 proposal."""


class _SolveSnapshot(Protocol):
    session_id: str
    session_revision: int
    source_kind: str | None
    model_name: str | None
    model_revision: int
    artifact: object | None
    materials: Sequence[object]
    sections: Sequence[object]
    steps: Sequence[object]
    runs: Sequence[object]
    unit_context: UnitContext | None
    model_current: bool
    mesh_current: bool

    def validation_for(self, step_name: str) -> ValidationRecord | None: ...


@dataclass(frozen=True, slots=True)
class SolveValidationStamp:
    """Exact accepted preflight identity used by an A6 solve proposal."""

    session_id: str
    artifact_id: str
    model_revision: int
    step_name: str
    report_hash: str
    stamp_hash: str

    def __post_init__(self) -> None:
        for field_name in ("session_id", "artifact_id", "step_name"):
            _require_nonblank(getattr(self, field_name), field_name)
        if type(self.model_revision) is not int or self.model_revision < 0:
            raise TypeError("model_revision must be a non-negative exact integer")
        for field_name in ("report_hash", "stamp_hash"):
            _require_sha256(getattr(self, field_name), field_name)
        if self.stamp_hash != _hash(self._core_dict()):
            raise SolveAuthoringError("validation stamp hash does not match")

    @classmethod
    def from_record(cls, record: ValidationRecord) -> SolveValidationStamp:
        if type(record) is not ValidationRecord:
            raise TypeError("record must be exactly ValidationRecord")
        stamp = record.stamp
        report_hash = _hash(_report_hash_payload(record.report))
        core = {
            "session_id": stamp.session_id,
            "artifact_id": stamp.artifact_id,
            "model_revision": stamp.model_revision,
            "step_name": stamp.step_name,
            "report_hash": report_hash,
        }
        return cls(**core, stamp_hash=_hash(core))

    def _core_dict(self) -> dict[str, object]:
        return {
            "session_id": self.session_id,
            "artifact_id": self.artifact_id,
            "model_revision": self.model_revision,
            "step_name": self.step_name,
            "report_hash": self.report_hash,
        }

    def to_dict(self) -> dict[str, object]:
        return {**self._core_dict(), "stamp_hash": self.stamp_hash}

    @classmethod
    def from_dict(cls, value: object) -> SolveValidationStamp:
        if type(value) is not dict:
            raise TypeError("validation_stamp must be an exact object")
        expected = {
            "session_id",
            "artifact_id",
            "model_revision",
            "step_name",
            "report_hash",
            "stamp_hash",
        }
        if set(value) != expected:
            raise SolveAuthoringError(
                "validation_stamp fields do not match the A6 schema"
            )
        return cls(
            session_id=value["session_id"],  # type: ignore[arg-type]
            artifact_id=value["artifact_id"],  # type: ignore[arg-type]
            model_revision=value["model_revision"],  # type: ignore[arg-type]
            step_name=value["step_name"],  # type: ignore[arg-type]
            report_hash=value["report_hash"],  # type: ignore[arg-type]
            stamp_hash=value["stamp_hash"],  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class SolveSummary:
    """Provider-safe, bounded facts shown on the local solve card."""

    model_name: str
    step_name: str
    job_name: str
    artifact_id: str
    model_revision: int
    node_count: int
    element_count: int
    dof_count: int
    element_types: tuple[tuple[str, int], ...]
    material_names: tuple[str, ...]
    section_names: tuple[str, ...]
    constraint_names: tuple[str, ...]
    load_names: tuple[str, ...]
    output_names: tuple[str, ...]
    unit_context: tuple[tuple[str, str | None], ...]
    warning_codes: tuple[str, ...]
    blocking_diagnostic_count: int
    collections_truncated: bool

    def __post_init__(self) -> None:
        for field_name in (
            "model_name",
            "step_name",
            "job_name",
            "artifact_id",
        ):
            _require_nonblank(getattr(self, field_name), field_name)
        for field_name in (
            "model_revision",
            "node_count",
            "element_count",
            "dof_count",
            "blocking_diagnostic_count",
        ):
            value = getattr(self, field_name)
            if type(value) is not int or value < 0:
                raise TypeError(f"{field_name} must be a non-negative exact integer")
        if type(self.collections_truncated) is not bool:
            raise TypeError("collections_truncated must be an exact bool")

    def to_dict(self) -> dict[str, object]:
        return {
            "model_name": self.model_name,
            "step_name": self.step_name,
            "job_name": self.job_name,
            "artifact_id": self.artifact_id,
            "model_revision": self.model_revision,
            "mesh": {
                "node_count": self.node_count,
                "element_count": self.element_count,
                "dof_count": self.dof_count,
                "element_types": dict(self.element_types),
            },
            "materials": list(self.material_names),
            "sections": list(self.section_names),
            "constraints": list(self.constraint_names),
            "loads": list(self.load_names),
            "outputs": list(self.output_names),
            "unit_context": dict(self.unit_context),
            "warning_codes": list(self.warning_codes),
            "blocking_diagnostic_count": self.blocking_diagnostic_count,
            "collections_truncated": self.collections_truncated,
        }


def validation_stamp_for_snapshot(
    snapshot: _SolveSnapshot,
    step_name: str,
) -> SolveValidationStamp:
    """Return the exact current passing validation stamp or fail closed."""

    clean_step = _require_nonblank(step_name, "step_name")
    record = snapshot.validation_for(clean_step)
    if record is None:
        raise SolveAuthoringError("the selected step has no current validation")
    if not record.passed:
        raise SolveAuthoringError("blocking preflight diagnostics prevent solving")
    stamp = SolveValidationStamp.from_record(record)
    artifact = snapshot.artifact
    if (
        snapshot.source_kind != "native"
        or not snapshot.model_current
        or not snapshot.mesh_current
        or artifact is None
        or stamp.session_id != snapshot.session_id
        or stamp.artifact_id != getattr(artifact, "artifact_id", None)
        or stamp.model_revision != snapshot.model_revision
        or stamp.step_name != clean_step
    ):
        raise SolveAuthoringError(
            "validation stamp does not match the current native model"
        )
    return stamp


def build_solve_summary(
    snapshot: _SolveSnapshot,
    step_name: str,
    job_name: str,
    *,
    max_collection_items: int = 32,
) -> SolveSummary:
    """Build one bounded local summary from a current passing validation."""

    if type(max_collection_items) is not int or max_collection_items <= 0:
        raise TypeError("max_collection_items must be a positive exact integer")
    clean_step = _require_nonblank(step_name, "step_name")
    clean_job = NamePolicy().validate(job_name)
    if not clean_job.startswith("作业-"):
        raise SolveAuthoringError("job_name must use the controlled 作业 type")
    units = snapshot.unit_context
    if type(units) is not UnitContext:
        raise SolveAuthoringError(
            "a confirmed UnitContext is required before native solving"
        )
    stamp = validation_stamp_for_snapshot(snapshot, clean_step)
    record = snapshot.validation_for(clean_step)
    if record is None:
        raise SolveAuthoringError("the selected step has no current validation")
    step = _one_step(snapshot.steps, clean_step)
    artifact = snapshot.artifact
    model = None if artifact is None else getattr(artifact, "model", None)
    mesh = None if model is None else getattr(model, "mesh", None)
    elements = tuple(getattr(mesh, "elements", ()))
    element_types = tuple(
        sorted(Counter(str(getattr(item, "type", "unknown")) for item in elements).items())
    )
    collections: list[tuple[tuple[str, ...], bool]] = [
        _bounded_names(snapshot.materials, max_collection_items),
        _bounded_names(snapshot.sections, max_collection_items),
        _bounded_names(getattr(step, "boundaries", ()), max_collection_items),
        _bounded_names(_step_loads(step), max_collection_items),
        _bounded_names(getattr(step, "outputs", ()), max_collection_items),
    ]
    warnings = tuple(
        item.code
        for item in record.report.diagnostics
        if item.severity is PreflightSeverity.WARNING
    )
    warning_codes = warnings[:max_collection_items]
    element_types_bounded = element_types[:max_collection_items]
    truncated = (
        any(item[1] for item in collections)
        or len(warnings) > max_collection_items
        or len(element_types) > max_collection_items
    )
    facts = record.report.facts
    return SolveSummary(
        model_name=_bounded_text(
            facts.model_name or snapshot.model_name or "模型-未命名"
        ),
        step_name=clean_step,
        job_name=clean_job,
        artifact_id=stamp.artifact_id,
        model_revision=stamp.model_revision,
        node_count=facts.node_count,
        element_count=facts.element_count,
        dof_count=facts.dof_count,
        element_types=element_types_bounded,
        material_names=collections[0][0],
        section_names=collections[1][0],
        constraint_names=collections[2][0],
        load_names=collections[3][0],
        output_names=collections[4][0],
        unit_context=tuple(units.to_dict().items()),
        warning_codes=warning_codes,
        blocking_diagnostic_count=len(record.report.errors),
        collections_truncated=truncated,
    )


def create_solve_proposal(
    *,
    proposal_id: str,
    agent_session_id: str,
    turn_id: str,
    source_tool_call_ids: Sequence[str],
    snapshot: _SolveSnapshot,
    draft_revision: int,
    step_name: str,
    job_name: str,
    target_document_id: str | None = None,
) -> AgentProposal:
    """Create one executable A6 proposal from an accepted preflight only."""

    summary = build_solve_summary(snapshot, step_name, job_name)
    if summary.blocking_diagnostic_count:
        raise SolveAuthoringError("blocking diagnostics prevent a solve proposal")
    if any(
        str(getattr(run, "name", "")).casefold() == job_name.casefold()
        for run in snapshot.runs
    ):
        raise SolveAuthoringError("job_name is already used in this session")
    stamp = validation_stamp_for_snapshot(snapshot, step_name)
    operation = ModelOperation(
        OperationKind.REQUEST_SOLVE,
        {
            "step_name": summary.step_name,
            "job_name": summary.job_name,
            "artifact_id": summary.artifact_id,
            "model_revision": summary.model_revision,
            "validation_stamp": stamp.to_dict(),
        },
    )
    return AgentProposal.create(
        proposal_id=proposal_id,
        proposal_kind=ProposalKind.SOLVE,
        agent_session_id=agent_session_id,
        turn_id=turn_id,
        source_tool_call_ids=source_tool_call_ids,
        # Multi-document workspaces bind stable integer document identities;
        # the caller passes the bound document_id so the GUI acceptance
        # matcher recognizes the proposal target.  The synthesized format is
        # kept only as a fallback for unbound single-document callers.
        target_document_id=(
            str(target_document_id)
            if target_document_id is not None
            else f"document:{snapshot.session_id}"
        ),
        target_session_id=snapshot.session_id,
        base_session_revision=snapshot.session_revision,
        draft_revision=draft_revision,
        operations=(operation,),
        preconditions={
            "authoring_phase": "A6",
            "source_kind": "native",
            "mesh_current": True,
            "validation_stamp": stamp.to_dict(),
            "blocking_diagnostic_count": 0,
        },
        expected_changes={
            "run_count": 1,
            "result_bound_to_current_artifact": True,
        },
        invalidation_impact={
            "model": False,
            "mesh": False,
            "definitions": False,
            "results": False,
        },
        display_summary={
            "title": "提交线性静力求解",
            "confirm_label": "开始求解",
            "summary": summary.to_dict(),
            "validation_stamp": stamp.stamp_hash,
        },
    )


def solve_operation_identity(
    operation: ModelOperation,
) -> tuple[str, str, str, int, SolveValidationStamp]:
    """Strictly decode the one operation accepted by the GUI solve port."""

    if type(operation) is not ModelOperation or operation.kind is not OperationKind.REQUEST_SOLVE:
        raise SolveAuthoringError("A6 requires one REQUEST_SOLVE operation")
    parameters = operation.parameters
    if set(parameters) != {
        "step_name",
        "job_name",
        "artifact_id",
        "model_revision",
        "validation_stamp",
    }:
        raise SolveAuthoringError(
            "A6 REQUEST_SOLVE fields do not match the exact schema"
        )
    step_name = _require_nonblank(parameters["step_name"], "step_name")
    job_name = NamePolicy().validate(parameters["job_name"])  # type: ignore[arg-type]
    if not job_name.startswith("作业-"):
        raise SolveAuthoringError("job_name must use the controlled 作业 type")
    artifact_id = _require_nonblank(parameters["artifact_id"], "artifact_id")
    model_revision = parameters["model_revision"]
    if type(model_revision) is not int or model_revision < 0:
        raise TypeError("model_revision must be a non-negative exact integer")
    stamp = SolveValidationStamp.from_dict(parameters["validation_stamp"])
    if (
        stamp.step_name != step_name
        or stamp.artifact_id != artifact_id
        or stamp.model_revision != model_revision
    ):
        raise SolveAuthoringError(
            "solve operation identity does not match its validation stamp"
        )
    return step_name, job_name, artifact_id, model_revision, stamp


def _report_hash_payload(report: PreflightReport) -> dict[str, object]:
    if type(report) is not PreflightReport:
        raise TypeError("report must be exactly PreflightReport")
    return {
        "step_name": report.step_name,
        "session_id": report.session_id,
        "artifact_id": report.artifact_id,
        "model_revision": report.model_revision,
        "numerical_stability_checked": report.numerical_stability_checked,
        "facts": asdict(report.facts),
        "diagnostics": [
            {
                "code": item.code,
                "severity": item.severity.value,
                "stage": item.stage.value,
                "message": item.message,
                "path": list(item.path),
                "remediation": item.remediation,
            }
            for item in report.diagnostics
        ],
    }


def _one_step(steps: Sequence[object], step_name: str) -> object:
    selected = [
        item for item in steps if str(getattr(item, "name", "")) == step_name
    ]
    if len(selected) != 1:
        raise SolveAuthoringError(
            "the current project must contain exactly one matching analysis step"
        )
    return selected[0]


def _step_loads(step: object) -> tuple[object, ...]:
    result: list[object] = []
    for field_name in (
        "cloads",
        "edge_loads",
        "surface_loads",
        "line_loads",
        "body_loads",
        "gravity_loads",
    ):
        result.extend(tuple(getattr(step, field_name, ())))
    return tuple(result)


def _bounded_names(
    values: Sequence[object],
    limit: int,
) -> tuple[tuple[str, ...], bool]:
    names = tuple(
        _bounded_text(getattr(value, "name", None) or type(value).__name__)
        for value in values
    )
    return names[:limit], len(names) > limit


def _bounded_text(value: object, limit: int = 160) -> str:
    text = str(value).replace("\x00", "").strip()
    return (text or "—")[:limit]


def _require_nonblank(value: object, name: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{name} must be an exact string")
    if not value or value != value.strip():
        raise SolveAuthoringError(
            f"{name} must be nonblank without surrounding whitespace"
        )
    return value


def _require_sha256(value: object, name: str) -> str:
    text = _require_nonblank(value, name)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise SolveAuthoringError(f"{name} must be a lowercase SHA-256 value")
    return text


def _hash(value: Mapping[str, object]) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "SolveAuthoringError",
    "SolveSummary",
    "SolveValidationStamp",
    "build_solve_summary",
    "create_solve_proposal",
    "solve_operation_identity",
    "validation_stamp_for_snapshot",
]
