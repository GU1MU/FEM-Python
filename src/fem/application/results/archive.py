"""Immutable, solver-neutral contracts for ``.femres`` archives.

The archive layer deliberately owns only data which is already accepted by
the result provider.  It does not retain a :class:`ModelResult`, a mesh model,
or any application/session object.  This keeps the object suitable for a
background codec and for a result-only document in a later phase.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..revisions import TaskToken

from .data import (
    FieldData,
    FieldState,
    ResultCatalog,
    ResultMaterializationSnapshot,
    ResultTopologyProjection,
    _freeze_json_mapping,
)
from .execution import OutputExecutionStatus, ResultExecutionReport
from .fields import ResultSourceKey
from .registry import ElementResultProfile
from fem.application.units import UnitContext
from fem.post.fields import ResultRegionKey, encode_result_region_key


def _required_text(value: object, *, label: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{label} must be a string")
    if not value.strip():
        raise ValueError(f"{label} must not be blank")
    return value


def _optional_text(value: object, *, label: str) -> str | None:
    if value is None:
        return None
    return _required_text(value, label=label)


def _checked_datetime(value: object, *, label: str) -> datetime:
    if not isinstance(value, datetime):
        raise TypeError(f"{label} must be a datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{label} must be timezone-aware")
    return value.astimezone(timezone.utc)


def _freeze_mapping(value: Mapping[str, Any], *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    return _freeze_json_mapping(value, path=label, ancestors=set())


def _freeze_index_mapping(
    value: Mapping[str, Any],
    *,
    label: str,
) -> Mapping[str, tuple[int, ...]]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    checked: dict[str, tuple[int, ...]] = {}
    for name in sorted(value):
        if type(name) is not str or not name.strip():
            raise TypeError(f"{label} keys must be nonblank strings")
        ids = value[name]
        if type(ids) not in {tuple, list}:
            raise TypeError(f"{label}.{name} must be a sequence of integers")
        checked_ids = tuple(ids)
        if any(type(item) is not int or item <= 0 for item in checked_ids):
            raise ValueError(f"{label}.{name} must contain positive integers")
        if len(set(checked_ids)) != len(checked_ids):
            raise ValueError(f"{label}.{name} must contain unique integers")
        checked[name] = checked_ids
    return _freeze_json_mapping(
        checked,
        path=label,
        ancestors=set(),
    )


def _topology_semantically_equal(
    left: ResultTopologyProjection,
    right: ResultTopologyProjection,
) -> bool:
    if left is right:
        return True
    return (
        left.source == right.source
        and left.node_ids == right.node_ids
        and bool((left._node_coordinates == right._node_coordinates).all())
        and bool(
            (left._nodal_displacements == right._nodal_displacements).all()
        )
        and left.element_ids == right.element_ids
        and left.element_types == right.element_types
        and left.connectivity == right.connectivity
        and left.element_region_keys == right.element_region_keys
    )


def result_model_fingerprint(
    topology: ResultTopologyProjection,
    profile: ElementResultProfile,
    *,
    step_name: str,
    unit_context: UnitContext | None,
) -> str:
    """Return a deterministic fingerprint for archive model semantics."""

    coordinates = topology._node_coordinates
    coordinate_digest = hashlib.sha256()
    coordinate_digest.update(memoryview(coordinates).cast("B"))

    payload = {
        "node_ids": list(topology.node_ids),
        "node_coordinates": {
            "dtype": str(coordinates.dtype),
            "shape": list(coordinates.shape),
            "sha256": coordinate_digest.hexdigest(),
        },
        "element_ids": list(topology.element_ids),
        "element_types": list(topology.element_types),
        "connectivity": [list(row) for row in topology.connectivity],
        "element_region_keys": [
            encode_result_region_key(item)
            for item in topology.element_region_keys
        ],
        "step_name": step_name,
        "profile": {
            "family": profile.family.value,
            "canonical_element_types": list(profile.canonical_element_types),
            "element_families": list(profile.element_families),
            "dofs_per_node": profile.dofs_per_node,
            "dof_labels": list(profile.dof_labels),
            "force_labels": list(profile.force_labels),
            "primary_compatible": profile.primary_compatible,
            "stress_compatible": profile.stress_compatible,
        },
        "unit_context": (
            None if unit_context is None else unit_context.to_dict()
        ),
    }
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _result_model_fingerprint(
    topology: ResultTopologyProjection,
    profile: ElementResultProfile,
    *,
    step_name: str,
    unit_context: UnitContext | None,
) -> str:
    """Backward-compatible private alias for the public fingerprint helper."""

    return result_model_fingerprint(
        topology,
        profile,
        step_name=step_name,
        unit_context=unit_context,
    )


@dataclass(frozen=True, slots=True)
class ResultArchiveOrigin:
    """Shareable provenance, separated from local session identity."""

    model_name: str
    source_basename: str | None = None
    model_fingerprint: str | None = None
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _required_text(self.model_name, label="model_name")
        basename = _optional_text(self.source_basename, label="source_basename")
        if basename is not None:
            # A basename is intentionally not allowed to smuggle an absolute
            # path into a shareable archive.
            if any(separator in basename for separator in ("/", "\\", ":")):
                raise ValueError("source_basename must be a basename")
            if basename in {".", ".."}:
                raise ValueError("source_basename must not be a path marker")
        fingerprint = _optional_text(
            self.model_fingerprint,
            label="model_fingerprint",
        )
        if fingerprint is None:
            raise ValueError("model_fingerprint is required for an archive origin")
        if not re.fullmatch(r"[0-9a-f]{64}", fingerprint):
            raise ValueError("model_fingerprint must be a lowercase SHA-256 hex digest")
        object.__setattr__(self, "source_basename", basename)
        object.__setattr__(self, "model_fingerprint", fingerprint)
        provenance = self.provenance
        if isinstance(provenance, ResultSourceKey):
            provenance = {
                "result_id": provenance.result_id,
                "session_id": provenance.session_id,
                "artifact_id": provenance.artifact_id,
                "model_revision": provenance.model_revision,
                "step_name": provenance.step_name,
                "run_id": provenance.run_id,
            }
        elif not isinstance(provenance, Mapping) and all(
            hasattr(provenance, name)
            for name in (
                "session_id",
                "artifact_id",
                "model_revision",
                "step_name",
                "run_id",
            )
        ):
            provenance = {
                name: getattr(provenance, name)
                for name in (
                    "session_id",
                    "artifact_id",
                    "model_revision",
                    "step_name",
                    "run_id",
                )
            }
        object.__setattr__(self, "provenance", _freeze_mapping(provenance, label="provenance"))

    @property
    def model_display_name(self) -> str:
        return self.model_name

    @property
    def source_file_basename(self) -> str | None:
        return self.source_basename

    @property
    def fingerprint(self) -> str | None:
        return self.model_fingerprint


@dataclass(frozen=True, slots=True)
class ResultArchiveRun:
    """Detached metadata for the one successful run in an archive."""

    name: str
    step_name: str
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    timings: Mapping[str, float] = field(default_factory=dict)
    messages: tuple[str, ...] = ()
    output_report: ResultExecutionReport | None = None

    def __post_init__(self) -> None:
        _required_text(self.name, label="run.name")
        _required_text(self.step_name, label="run.step_name")
        object.__setattr__(self, "created_at", _checked_datetime(self.created_at, label="run.created_at"))
        if self.started_at is not None:
            object.__setattr__(
                self,
                "started_at",
                _checked_datetime(self.started_at, label="run.started_at"),
            )
        if self.finished_at is not None:
            object.__setattr__(
                self,
                "finished_at",
                _checked_datetime(self.finished_at, label="run.finished_at"),
            )
        if self.started_at is not None and self.started_at < self.created_at:
            raise ValueError("run.started_at cannot precede run.created_at")
        if self.finished_at is not None and self.started_at is not None and self.finished_at < self.started_at:
            raise ValueError("run.finished_at cannot precede run.started_at")
        if not isinstance(self.timings, Mapping):
            raise TypeError("run.timings must be a mapping")
        timings: dict[str, float] = {}
        for key in sorted(self.timings):
            if type(key) is not str or not key.strip():
                raise TypeError("run.timings keys must be nonblank strings")
            value = self.timings[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError("run.timings values must be real numbers")
            result = float(value)
            if result != result or result in {float("inf"), float("-inf")}:
                raise ValueError("run.timings values must be finite")
            if result < 0.0:
                raise ValueError("run.timings values must be non-negative")
            timings[key] = result
        object.__setattr__(
            self,
            "timings",
            _freeze_json_mapping(timings, path="run.timings", ancestors=set()),
        )
        if type(self.messages) is not tuple:
            raise TypeError("run.messages must be a tuple")
        if any(type(message) is not str for message in self.messages):
            raise TypeError("run.messages must contain strings")
        object.__setattr__(self, "messages", tuple(self.messages))
        if self.output_report is not None and type(self.output_report) is not ResultExecutionReport:
            raise TypeError("run.output_report must be ResultExecutionReport or None")

    @property
    def job_name(self) -> str:
        return self.name

    @property
    def analysis_step(self) -> str:
        return self.step_name


@dataclass(frozen=True, slots=True)
class ResultArchiveModelProjection:
    """Read-only model facts needed by result consumers."""

    topology: ResultTopologyProjection
    unit_context: UnitContext | None = None
    named_region_node_ids: Mapping[str, tuple[int, ...]] = field(default_factory=dict)
    named_region_element_ids: Mapping[str, tuple[int, ...]] = field(default_factory=dict)
    summaries: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if type(self.topology) is not ResultTopologyProjection:
            raise TypeError("model projection topology must be ResultTopologyProjection")
        if self.unit_context is not None and type(self.unit_context) is not UnitContext:
            raise TypeError("unit_context must be UnitContext or None")
        object.__setattr__(
            self,
            "named_region_node_ids",
            _freeze_index_mapping(self.named_region_node_ids, label="named_region_node_ids"),
        )
        object.__setattr__(
            self,
            "named_region_element_ids",
            _freeze_index_mapping(self.named_region_element_ids, label="named_region_element_ids"),
        )
        node_ids = frozenset(self.topology.node_ids)
        element_ids = frozenset(self.topology.element_ids)
        for name, values in self.named_region_node_ids.items():
            if not frozenset(values).issubset(node_ids):
                raise ValueError(f"named_region_node_ids.{name} references an unknown node")
        for name, values in self.named_region_element_ids.items():
            if not frozenset(values).issubset(element_ids):
                raise ValueError(f"named_region_element_ids.{name} references an unknown element")
        object.__setattr__(self, "summaries", _freeze_mapping(self.summaries, label="summaries"))


@dataclass(frozen=True, slots=True)
class ResultArchiveSnapshot:
    """Complete immutable payload for one schema-v1 result archive."""

    archive_id: str
    created_at: datetime
    producer_version: str
    origin: ResultArchiveOrigin
    run: ResultArchiveRun
    profile: ElementResultProfile
    catalog: ResultCatalog
    materialization: ResultMaterializationSnapshot
    model_projection: ResultArchiveModelProjection
    unit_context: UnitContext | None = None

    def __post_init__(self) -> None:
        _required_text(self.archive_id, label="archive_id")
        object.__setattr__(self, "created_at", _checked_datetime(self.created_at, label="created_at"))
        _required_text(self.producer_version, label="producer_version")
        if type(self.origin) is not ResultArchiveOrigin:
            raise TypeError("origin must be ResultArchiveOrigin")
        if type(self.run) is not ResultArchiveRun:
            raise TypeError("run must be ResultArchiveRun")
        if type(self.profile) is not ElementResultProfile:
            raise TypeError("profile must be ElementResultProfile")
        if type(self.catalog) is not ResultCatalog:
            raise TypeError("catalog must be ResultCatalog")
        if type(self.materialization) is not ResultMaterializationSnapshot:
            raise TypeError("materialization must be ResultMaterializationSnapshot")
        if type(self.model_projection) is not ResultArchiveModelProjection:
            raise TypeError("model_projection must be ResultArchiveModelProjection")
        if self.unit_context is not None and type(self.unit_context) is not UnitContext:
            raise TypeError("unit_context must be UnitContext or None")
        source = self.materialization.source
        if self.run.output_report is None:
            raise ValueError("archive runs require an output report")
        if self.catalog.source != source:
            raise ValueError("catalog source must match materialization source")
        if self.model_projection.topology.source != source:
            raise ValueError("model projection topology source must match source")
        if not _topology_semantically_equal(
            self.model_projection.topology,
            self.materialization.topology,
        ):
            raise ValueError(
                "model projection topology must match materialization topology"
            )
        if self.run.step_name != source.step_name:
            raise ValueError("run step_name must match result source")
        if self.run.output_report is not None:
            if self.run.output_report.source != source:
                raise ValueError("run output report source must match source")
            executed_keys = {
                key
                for request in self.run.output_report.requests
                for variable in request.variables
                if variable.status is OutputExecutionStatus.EXECUTED
                for key in variable.field_keys
            }
            materialized_keys = {field_data.key for field_data in self.materialization.fields}
            if not executed_keys.issubset(materialized_keys):
                raise ValueError("run output report references an unmaterialized field")
        if self.unit_context is None and self.model_projection.unit_context is not None:
            object.__setattr__(self, "unit_context", self.model_projection.unit_context)
        elif (
            self.unit_context is not None
            and self.model_projection.unit_context is not None
            and self.unit_context != self.model_projection.unit_context
        ):
            raise ValueError(
                "top-level and model projection unit contexts must match"
            )
        ready_keys = {
            item.key
            for item in self.catalog.fields
            if item.state is FieldState.READY
        }
        materialized_keys = {field_data.key for field_data in self.materialization.fields}
        if ready_keys != materialized_keys:
            raise ValueError(
                "catalog READY keys must exactly match materialized field keys"
            )

    @property
    def source(self) -> ResultSourceKey:
        return self.materialization.source

    @property
    def topology(self) -> ResultTopologyProjection:
        return self.materialization.topology

    @property
    def fields(self) -> tuple[Any, ...]:
        return self.materialization.fields

    @property
    def output_report(self) -> ResultExecutionReport | None:
        return self.run.output_report

    @property
    def model(self) -> ResultArchiveModelProjection:
        """Compatibility alias for consumers calling the projection ``model``."""

        return self.model_projection

    @property
    def materialization_snapshot(self) -> ResultMaterializationSnapshot:
        return self.materialization

    def rebind_source(self, source: ResultSourceKey) -> "ResultArchiveSnapshot":
        """Return the same archive facts under a new local result identity.

        Rebinding only replaces scalar/source references.  Numeric arrays and
        field locations remain shared with the decoded immutable archive so a
        result-only document can be installed without a second full-matrix
        copy.
        """

        return rebind_result_archive_snapshot(self, source)

    @classmethod
    def from_result_record(
        cls,
        record: object,
        *,
        model_name: str | None = None,
        source_basename: str | None = None,
        unit_context: UnitContext | None = None,
        model_fingerprint: str | None = None,
        run: ResultArchiveRun | None = None,
        archive_id: str | None = None,
        producer_version: str = "fem-python",
        named_region_node_ids: Mapping[str, tuple[int, ...]] | None = None,
        named_region_element_ids: Mapping[str, tuple[int, ...]] | None = None,
        summaries: Mapping[str, Any] | None = None,
    ) -> "ResultArchiveSnapshot":
        """Build an archive snapshot from an accepted ``ResultRecord``.

        The import is intentionally local so the archive contract remains
        independent of the Session/run lifecycle modules.
        """

        from fem.application.runs import ResultRecord, result_record_provider

        if type(record) is not ResultRecord:
            raise TypeError("record must be exactly ResultRecord")
        provider = result_record_provider(record)
        result_model = record.result.model
        resolved_model_name = model_name or getattr(result_model, "name", None) or "model"
        source = record.materialization.source
        topology = record.materialization.topology
        effective_unit_context = unit_context
        fingerprint = (
            model_fingerprint
            if model_fingerprint is not None
            else result_model_fingerprint(
                topology,
                provider.profile,
                step_name=source.step_name,
                unit_context=effective_unit_context,
            )
        )
        provenance = {
            "result_id": source.result_id,
            "session_id": source.session_id,
            "artifact_id": source.artifact_id,
            "model_revision": source.model_revision,
            "step_name": source.step_name,
            "run_id": source.run_id,
        }
        for name in (
            "beam_formulation",
            "beam_result_position",
            "beam_recovery_contract",
        ):
            value = getattr(record.provenance, name)
            if value is not None:
                provenance[name] = value
        resolved_run = run or ResultArchiveRun(
            name=getattr(record.result, "name", None) or source.run_id,
            step_name=source.step_name,
            created_at=record.created_at,
            output_report=record.output_report,
        )
        origin = ResultArchiveOrigin(
            model_name=resolved_model_name,
            source_basename=source_basename,
            model_fingerprint=fingerprint,
            provenance=provenance,
        )
        projection = ResultArchiveModelProjection(
            topology=topology,
            unit_context=unit_context,
            named_region_node_ids=named_region_node_ids or {},
            named_region_element_ids=named_region_element_ids or {},
            summaries=summaries or {},
        )
        return cls(
            archive_id=archive_id or source.result_id,
            created_at=record.created_at,
            producer_version=producer_version,
            origin=origin,
            run=resolved_run,
            profile=provider.profile,
            catalog=(
                provider.publish_fields(
                    tuple(field_data.key for field_data in record.materialization.fields)
                ).catalog()
                if {
                    item.key
                    for item in provider.catalog().fields
                    if item.state is FieldState.READY
                }
                != {field_data.key for field_data in record.materialization.fields}
                else provider.catalog()
            ),
            materialization=record.materialization,
            model_projection=projection,
            unit_context=unit_context,
        )


@dataclass(frozen=True, slots=True)
class LoadedResultArchive:
    """Decoded archive with path metadata, detached from open file handles."""

    snapshot: ResultArchiveSnapshot
    path: Path | None
    source_schema: int
    notices: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if type(self.snapshot) is not ResultArchiveSnapshot:
            raise TypeError("snapshot must be ResultArchiveSnapshot")
        if self.path is not None and not isinstance(self.path, Path):
            raise TypeError("path must be a pathlib.Path or None")
        if type(self.source_schema) is not int or self.source_schema <= 0:
            raise TypeError("source_schema must be a positive integer")
        if type(self.notices) is not tuple or any(type(item) is not str for item in self.notices):
            raise TypeError("notices must be a tuple of strings")


@dataclass(frozen=True, slots=True)
class ResultFileState:
    """The last accepted on-disk generation for one in-session result.

    A file state deliberately lives outside :class:`AnalysisRun`: the run
    remains a scientific/lifecycle record while this value only describes a
    user's optional ``.femres`` save.  ``saved_generation`` is the
    materialization generation represented by ``path``; a later generation is
    therefore visibly unsaved while retaining the old path for UI display.
    """

    path: Path
    saved_generation: int
    run_id: str
    result_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.path, Path):
            raise TypeError("result file path must be a pathlib.Path")
        if type(self.saved_generation) is not int:
            raise TypeError("saved_generation must be an integer")
        if self.saved_generation < 0:
            raise ValueError("saved_generation must be non-negative")
        if type(self.run_id) is not str or not self.run_id.strip():
            raise ValueError("run_id must be a nonblank string")
        if type(self.result_id) is not str or not self.result_id.strip():
            raise ValueError("result_id must be a nonblank string")

    @property
    def generation(self) -> int:
        """Compatibility alias for consumers that call this saved generation."""

        return self.saved_generation

    @property
    def file_path(self) -> Path:
        return self.path


@dataclass(frozen=True, slots=True)
class ResultArchiveSaveSnapshot:
    """Worker-safe, generation-bound payload for one result archive save.

    The payload contains only immutable archive DTOs and scalar identity.  It
    intentionally does not retain a Session, ResultRecord, ModelResult,
    provider materializer or file handle.  Numeric arrays are shared through
    the read-only ownership established by ``ResultMaterializationSnapshot``;
    no public array-copy property or ``deepcopy`` is needed to prepare this
    object.
    """

    token: "TaskToken"
    archive: ResultArchiveSnapshot
    source: ResultSourceKey
    materialization_generation: int
    run_id: str
    result_id: str

    def __post_init__(self) -> None:
        # Importing TaskToken here avoids making the archive DTO depend on the
        # Session module during module initialisation while still enforcing an
        # exact token contract at runtime.
        from ..revisions import TaskToken

        if type(self.token) is not TaskToken:
            raise TypeError("token must be exactly TaskToken")
        if type(self.archive) is not ResultArchiveSnapshot:
            raise TypeError("archive must be exactly ResultArchiveSnapshot")
        if type(self.source) is not ResultSourceKey:
            raise TypeError("source must be exactly ResultSourceKey")
        if type(self.materialization_generation) is not int:
            raise TypeError("materialization_generation must be an integer")
        if self.materialization_generation < 0:
            raise ValueError("materialization_generation must be non-negative")
        for value, label in (
            (self.run_id, "run_id"),
            (self.result_id, "result_id"),
        ):
            if type(value) is not str or not value.strip():
                raise ValueError(f"{label} must be a nonblank string")
        if self.archive.source != self.source:
            raise ValueError("archive source must match save source")
        if self.archive.materialization.generation != self.materialization_generation:
            raise ValueError(
                "archive generation must match save materialization generation"
            )
        if self.source.run_id != self.run_id:
            raise ValueError("save run_id must match source")
        if self.source.result_id != self.result_id:
            raise ValueError("save result_id must match source")
        if self.token.task_kind != "result_archive_save":
            raise ValueError("save token must be a result_archive_save token")
        if (
            self.token.session_id != self.source.session_id
            or self.token.artifact_id != self.source.artifact_id
            or self.token.step_name != self.source.step_name
            or self.token.run_id != self.source.run_id
            or self.token.result_id != self.source.result_id
        ):
            raise ValueError("save token identity must match result source")
        dependencies = dict(self.token.dependency_revisions)
        if dependencies.get("model_revision") != self.source.model_revision:
            raise ValueError("save token model revision must match result source")
        if dependencies.get("materialization_generation") != self.materialization_generation:
            raise ValueError(
                "save token generation must match materialization generation"
            )

    @property
    def snapshot(self) -> ResultArchiveSnapshot:
        """Alias used by workers that call the archive payload ``snapshot``."""

        return self.archive

    @property
    def result_archive(self) -> ResultArchiveSnapshot:
        return self.archive

    @property
    def task_token(self) -> "TaskToken":
        return self.token

    @property
    def result_source(self) -> ResultSourceKey:
        return self.source

    @property
    def generation(self) -> int:
        return self.materialization_generation


def archive_region_dictionary(snapshot: ResultArchiveSnapshot) -> tuple[ResultRegionKey, ...]:
    """Return deterministic region identities used by topology and fields."""

    if type(snapshot) is not ResultArchiveSnapshot:
        raise TypeError("snapshot must be ResultArchiveSnapshot")
    regions = set(snapshot.topology.element_region_keys)
    for field_data in snapshot.fields:
        regions.update(
            location.region_key
            for location in field_data.locations
            if location.region_key is not None
        )
    return tuple(sorted(regions, key=encode_result_region_key))


def build_result_archive_snapshot(
    record: object,
    **kwargs: object,
) -> ResultArchiveSnapshot:
    """Functional factory kept alongside the classmethod for callers that
    prefer the result-layer factory style used by the live provider APIs.
    """

    return ResultArchiveSnapshot.from_result_record(record, **kwargs)


def _rebind_topology_source(
    topology: ResultTopologyProjection,
    source: ResultSourceKey,
) -> ResultTopologyProjection:
    if topology.source == source:
        return topology
    rebound = object.__new__(ResultTopologyProjection)
    object.__setattr__(rebound, "source", source)
    object.__setattr__(rebound, "node_ids", topology.node_ids)
    object.__setattr__(rebound, "_node_coordinates", topology._node_coordinates)
    object.__setattr__(rebound, "_nodal_displacements", topology._nodal_displacements)
    object.__setattr__(rebound, "element_ids", topology.element_ids)
    object.__setattr__(rebound, "element_types", topology.element_types)
    object.__setattr__(rebound, "connectivity", topology.connectivity)
    object.__setattr__(rebound, "element_region_keys", topology.element_region_keys)
    return rebound


def _rebind_field_source(
    field_data: FieldData,
    source: ResultSourceKey,
) -> FieldData:
    if field_data.source == source:
        return field_data
    rebound = object.__new__(FieldData)
    object.__setattr__(rebound, "descriptor", field_data.descriptor)
    object.__setattr__(rebound, "source", source)
    object.__setattr__(rebound, "key", field_data.key)
    object.__setattr__(rebound, "locations", field_data.locations)
    object.__setattr__(rebound, "_values", field_data._values)
    return rebound


def rebind_result_archive_snapshot(
    snapshot: ResultArchiveSnapshot,
    source: ResultSourceKey,
) -> ResultArchiveSnapshot:
    """Rebind every archive source key without copying numeric arrays."""

    if type(snapshot) is not ResultArchiveSnapshot:
        raise TypeError("snapshot must be exactly ResultArchiveSnapshot")
    if type(source) is not ResultSourceKey:
        raise TypeError("source must be exactly ResultSourceKey")
    topology = _rebind_topology_source(snapshot.topology, source)
    materialization = replace(
        snapshot.materialization,
        source=source,
        topology=topology,
        fields=tuple(
            _rebind_field_source(field_data, source)
            for field_data in snapshot.materialization.fields
        ),
    )
    catalog = replace(snapshot.catalog, source=source)
    report = (
        None
        if snapshot.output_report is None
        else replace(snapshot.output_report, source=source)
    )
    run = replace(snapshot.run, output_report=report)
    projection = replace(snapshot.model_projection, topology=topology)
    return ResultArchiveSnapshot(
        archive_id=snapshot.archive_id,
        created_at=snapshot.created_at,
        producer_version=snapshot.producer_version,
        origin=snapshot.origin,
        run=run,
        profile=snapshot.profile,
        catalog=catalog,
        materialization=materialization,
        model_projection=projection,
        unit_context=snapshot.unit_context,
    )


__all__ = [
    "LoadedResultArchive",
    "ResultArchiveSaveSnapshot",
    "ResultArchiveModelProjection",
    "ResultArchiveOrigin",
    "ResultArchiveRun",
    "ResultArchiveSnapshot",
    "ResultFileState",
    "archive_region_dictionary",
    "build_result_archive_snapshot",
    "rebind_result_archive_snapshot",
    "result_model_fingerprint",
]
