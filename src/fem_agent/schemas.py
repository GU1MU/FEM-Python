"""Versioned, JSON-compatible contracts used by the FEM Agent."""

from __future__ import annotations

import json
import math
import unicodedata
from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum
from typing import Any, ClassVar, Mapping, Self


SCHEMA_VERSION = 1


class SchemaValidationError(ValueError):
    """Raised when persisted or provider-supplied data violates a contract."""


class DiagnosticSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class ExportFormat(str, Enum):
    CSV = "csv"
    VTK = "vtk"


class ResultQueryKind(str, Enum):
    DISPLACEMENT_COMPONENT = "displacement_component"
    DISPLACEMENT_MAGNITUDE = "displacement_magnitude"
    MAX_DISPLACEMENT_COMPONENT = "max_displacement_component"
    MAX_DISPLACEMENT_MAGNITUDE = "max_displacement_magnitude"
    REACTION_COMPONENT = "reaction_component"
    REACTION_SUM = "reaction_sum"
    STRESS_EXTREMA = "stress_extrema"


class RunStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class SessionPhase(str, Enum):
    EMPTY = "empty"
    INSPECTED = "inspected"
    DRAFT_READY = "draft_ready"
    AWAITING_CONFIRMATION = "awaiting_confirmation"
    CONFIRMED = "confirmed"
    RUNNING = "running"
    SOLVED = "solved"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JsonContract:
    """Small serialization mixin with deterministic UTF-8 JSON output."""

    schema_version: ClassVar[int] = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return _json_value(self)

    def to_json(self, *, indent: int | None = None) -> str:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            indent=indent,
            sort_keys=True,
            separators=None if indent is not None else (",", ":"),
        )


@dataclass(frozen=True)
class UnitContext(JsonContract):
    """Declared unit convention; values are recorded and never converted."""

    length: str
    force: str
    stress: str
    density: str
    acceleration: str
    convention: str | None = None

    def __post_init__(self) -> None:
        for name in ("length", "force", "stress", "density", "acceleration"):
            _validate_label(getattr(self, name), name)
        if self.convention is not None:
            _validate_label(self.convention, "convention")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Self:
        data = _strict_mapping(
            value,
            required={"length", "force", "stress", "density", "acceleration"},
            optional={"convention"},
            context="UnitContext",
        )
        return cls(
            length=_string(data["length"], "length"),
            force=_string(data["force"], "force"),
            stress=_string(data["stress"], "stress"),
            density=_string(data["density"], "density"),
            acceleration=_string(data["acceleration"], "acceleration"),
            convention=_optional_string(data.get("convention"), "convention"),
        )


@dataclass(frozen=True)
class ResourceLimits(JsonContract):
    """Deterministic execution limits attached to a revision."""

    max_input_bytes: int = 50 * 1024 * 1024
    max_nodes: int = 250_000
    max_elements: int = 250_000
    max_dofs: int = 2_000_000
    worker_timeout_seconds: float = 300.0
    max_output_files: int = 32
    max_output_bytes: int = 1024 * 1024 * 1024

    def __post_init__(self) -> None:
        for name in (
            "max_input_bytes",
            "max_nodes",
            "max_elements",
            "max_dofs",
            "max_output_files",
            "max_output_bytes",
        ):
            _positive_int(getattr(self, name), name)
        object.__setattr__(
            self,
            "worker_timeout_seconds",
            _positive_number(
                self.worker_timeout_seconds,
                "worker_timeout_seconds",
            ),
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Self:
        names = {field.name for field in fields(cls)}
        data = _strict_mapping(
            value,
            required=set(),
            optional=names,
            context="ResourceLimits",
        )
        defaults = cls()
        return cls(
            max_input_bytes=_integer(
                data.get("max_input_bytes", defaults.max_input_bytes),
                "max_input_bytes",
            ),
            max_nodes=_integer(data.get("max_nodes", defaults.max_nodes), "max_nodes"),
            max_elements=_integer(
                data.get("max_elements", defaults.max_elements),
                "max_elements",
            ),
            max_dofs=_integer(data.get("max_dofs", defaults.max_dofs), "max_dofs"),
            worker_timeout_seconds=_number(
                data.get(
                    "worker_timeout_seconds",
                    defaults.worker_timeout_seconds,
                ),
                "worker_timeout_seconds",
            ),
            max_output_files=_integer(
                data.get("max_output_files", defaults.max_output_files),
                "max_output_files",
            ),
            max_output_bytes=_integer(
                data.get("max_output_bytes", defaults.max_output_bytes),
                "max_output_bytes",
            ),
        )


@dataclass(frozen=True)
class ResultQuery(JsonContract):
    """One bounded, read-only result request."""

    kind: ResultQueryKind
    component: int | None = None
    node_id: int | None = None
    node_set: str | None = None
    element_set: str | None = None
    measure: str | None = None
    edge: str | None = field(
        default=None,
        metadata={"json_omit_none": True},
    )
    surface: str | None = field(
        default=None,
        metadata={"json_omit_none": True},
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", _enum(self.kind, ResultQueryKind, "kind"))
        if self.component is not None:
            _positive_int(self.component, "component")
        if self.node_id is not None:
            _positive_int(self.node_id, "node_id")
        for name in ("node_set", "edge", "surface", "element_set", "measure"):
            value = getattr(self, name)
            if value is not None:
                _validate_label(value, name)
        self._validate_shape()

    def _validate_shape(self) -> None:
        node_kinds = {
            ResultQueryKind.DISPLACEMENT_COMPONENT,
            ResultQueryKind.DISPLACEMENT_MAGNITUDE,
            ResultQueryKind.REACTION_COMPONENT,
        }
        component_kinds = {
            ResultQueryKind.DISPLACEMENT_COMPONENT,
            ResultQueryKind.MAX_DISPLACEMENT_COMPONENT,
            ResultQueryKind.REACTION_COMPONENT,
            ResultQueryKind.REACTION_SUM,
        }
        aggregate_node_kinds = {
            ResultQueryKind.MAX_DISPLACEMENT_COMPONENT,
            ResultQueryKind.MAX_DISPLACEMENT_MAGNITUDE,
            ResultQueryKind.REACTION_SUM,
        }
        selected_regions = tuple(
            name
            for name in ("node_set", "edge", "surface")
            if getattr(self, name) is not None
        )
        if self.kind in node_kinds and self.node_id is None:
            raise SchemaValidationError(f"{self.kind.value} requires node_id")
        if self.kind in component_kinds and self.component is None:
            raise SchemaValidationError(f"{self.kind.value} requires component")
        if len(selected_regions) > 1:
            raise SchemaValidationError(
                "node_set, edge, and surface are mutually exclusive"
            )
        if (
            (self.edge is not None or self.surface is not None)
            and self.kind not in aggregate_node_kinds
        ):
            raise SchemaValidationError(
                "edge and surface are valid only for aggregate nodal queries"
            )
        if (
            self.kind == ResultQueryKind.REACTION_SUM
            and not selected_regions
        ):
            raise SchemaValidationError(
                "reaction_sum requires node_set, edge, or surface"
            )
        if self.kind == ResultQueryKind.STRESS_EXTREMA:
            if self.measure is None:
                object.__setattr__(self, "measure", "von_mises")
        elif self.element_set is not None or self.measure is not None:
            raise SchemaValidationError(
                "element_set and measure are valid only for stress_extrema"
            )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Self:
        data = _strict_mapping(
            value,
            required={"kind"},
            optional={
                "component",
                "node_id",
                "node_set",
                "edge",
                "surface",
                "element_set",
                "measure",
            },
            context="ResultQuery",
        )
        return cls(
            kind=_enum(data["kind"], ResultQueryKind, "kind"),
            component=_optional_integer(data.get("component"), "component"),
            node_id=_optional_integer(data.get("node_id"), "node_id"),
            node_set=_optional_string(data.get("node_set"), "node_set"),
            edge=_optional_string(data.get("edge"), "edge"),
            surface=_optional_string(data.get("surface"), "surface"),
            element_set=_optional_string(data.get("element_set"), "element_set"),
            measure=_optional_string(data.get("measure"), "measure"),
        )


@dataclass(frozen=True)
class ImportAnalysisSpec(JsonContract):
    """Immutable, serializable source of truth for one analysis revision."""

    session_id: str
    revision: int
    source_artifact_id: str
    source_sha256: str
    unit_context: UnitContext | None
    analysis_step: str | None
    requested_queries: tuple[ResultQuery, ...]
    export_formats: tuple[ExportFormat, ...]
    resource_limits: ResourceLimits = field(default_factory=ResourceLimits)
    assumptions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _validate_identifier(self.session_id, "session_id")
        _positive_int(self.revision, "revision")
        _validate_identifier(self.source_artifact_id, "source_artifact_id")
        _validate_sha256(self.source_sha256)
        if self.analysis_step is not None:
            _validate_label(self.analysis_step, "analysis_step")
        object.__setattr__(self, "requested_queries", tuple(self.requested_queries))
        object.__setattr__(
            self,
            "export_formats",
            tuple(_enum(item, ExportFormat, "export_formats") for item in self.export_formats),
        )
        if len(set(self.export_formats)) != len(self.export_formats):
            raise SchemaValidationError("export_formats must not contain duplicates")
        object.__setattr__(self, "assumptions", tuple(self.assumptions))
        for assumption in self.assumptions:
            _validate_label(assumption, "assumption", max_length=512)

    @property
    def ready_for_confirmation(self) -> bool:
        return (
            self.unit_context is not None
            and self.analysis_step is not None
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Self:
        data = _versioned_mapping(
            value,
            required={
                "session_id",
                "revision",
                "source_artifact_id",
                "source_sha256",
                "unit_context",
                "analysis_step",
                "requested_queries",
                "export_formats",
                "resource_limits",
                "assumptions",
            },
            context="ImportAnalysisSpec",
        )
        return cls(
            session_id=_string(data["session_id"], "session_id"),
            revision=_integer(data["revision"], "revision"),
            source_artifact_id=_string(
                data["source_artifact_id"],
                "source_artifact_id",
            ),
            source_sha256=_string(data["source_sha256"], "source_sha256"),
            unit_context=(
                None
                if data["unit_context"] is None
                else UnitContext.from_dict(_mapping(data["unit_context"], "unit_context"))
            ),
            analysis_step=_optional_string(data["analysis_step"], "analysis_step"),
            requested_queries=tuple(
                ResultQuery.from_dict(_mapping(item, "requested_queries item"))
                for item in _array(data["requested_queries"], "requested_queries")
            ),
            export_formats=tuple(
                _enum(item, ExportFormat, "export_formats")
                for item in _array(data["export_formats"], "export_formats")
            ),
            resource_limits=ResourceLimits.from_dict(
                _mapping(data["resource_limits"], "resource_limits")
            ),
            assumptions=tuple(
                _string(item, "assumption")
                for item in _array(data["assumptions"], "assumptions")
            ),
        )


@dataclass(frozen=True)
class Diagnostic(JsonContract):
    """Machine-readable failure or warning safe for bounded display."""

    code: str
    severity: DiagnosticSeverity
    message: str
    source: str
    entity: str | None = None
    step: str | None = None
    remediation: str | None = None

    def __post_init__(self) -> None:
        _validate_identifier(self.code, "code", allow_upper=True)
        object.__setattr__(
            self,
            "severity",
            _enum(self.severity, DiagnosticSeverity, "severity"),
        )
        _validate_label(self.message, "message", max_length=2048)
        _validate_label(self.source, "source", max_length=128)
        for name in ("entity", "step", "remediation"):
            value = getattr(self, name)
            if value is not None:
                _validate_label(value, name, max_length=1024)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Self:
        data = _strict_mapping(
            value,
            required={"code", "severity", "message", "source"},
            optional={"entity", "step", "remediation"},
            context="Diagnostic",
        )
        return cls(
            code=_string(data["code"], "code"),
            severity=_enum(data["severity"], DiagnosticSeverity, "severity"),
            message=_string(data["message"], "message"),
            source=_string(data["source"], "source"),
            entity=_optional_string(data.get("entity"), "entity"),
            step=_optional_string(data.get("step"), "step"),
            remediation=_optional_string(data.get("remediation"), "remediation"),
        )


@dataclass(frozen=True)
class AnalysisSummary(JsonContract):
    """Bounded deterministic model summary shown before confirmation."""

    revision: int
    revision_hash: str
    source_artifact_id: str
    source_sha256: str
    model_name: str
    node_count: int
    element_count: int
    dofs_per_node: int
    total_dofs: int
    element_types: Mapping[str, int]
    node_sets: tuple[Mapping[str, Any], ...]
    element_sets: tuple[Mapping[str, Any], ...]
    edges: tuple[Mapping[str, Any], ...]
    surfaces: tuple[Mapping[str, Any], ...]
    materials: tuple[Mapping[str, Any], ...]
    sections: tuple[Mapping[str, Any], ...]
    analysis_step: Mapping[str, Any] | None
    constraints: tuple[Mapping[str, Any], ...]
    loads: tuple[Mapping[str, Any], ...]
    unit_context: UnitContext | None
    requested_queries: tuple[ResultQuery, ...]
    export_formats: tuple[ExportFormat, ...]
    keyword_inventory: tuple[Mapping[str, Any], ...]
    diagnostics: tuple[Diagnostic, ...]
    resource_class: str
    collections_truncated: bool = False

    def __post_init__(self) -> None:
        _positive_int(self.revision, "revision")
        _validate_sha256(self.revision_hash, "revision_hash")
        _validate_identifier(self.source_artifact_id, "source_artifact_id")
        _validate_sha256(self.source_sha256)
        _validate_label(self.model_name, "model_name")
        for name in ("node_count", "element_count", "dofs_per_node", "total_dofs"):
            _nonnegative_int(getattr(self, name), name)
        _validate_label(self.resource_class, "resource_class")
        object.__setattr__(
            self,
            "element_types",
            {
                _string(key, "element type"): _nonnegative_int(value, "element count")
                for key, value in self.element_types.items()
            },
        )
        for name in (
            "node_sets",
            "element_sets",
            "edges",
            "surfaces",
            "materials",
            "sections",
            "constraints",
            "loads",
            "keyword_inventory",
        ):
            value = tuple(_json_mapping(item, name) for item in getattr(self, name))
            object.__setattr__(self, name, value)
        if self.analysis_step is not None:
            object.__setattr__(
                self,
                "analysis_step",
                _json_mapping(self.analysis_step, "analysis_step"),
            )
        object.__setattr__(self, "requested_queries", tuple(self.requested_queries))
        object.__setattr__(
            self,
            "export_formats",
            tuple(_enum(item, ExportFormat, "export_formats") for item in self.export_formats),
        )
        object.__setattr__(self, "diagnostics", tuple(self.diagnostics))

    @property
    def has_blocking_diagnostics(self) -> bool:
        return any(item.severity == DiagnosticSeverity.ERROR for item in self.diagnostics)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Self:
        names = {field.name for field in fields(cls)}
        data = _versioned_mapping(
            value,
            required=names,
            context="AnalysisSummary",
        )
        mapping_tuple_names = (
            "node_sets",
            "element_sets",
            "edges",
            "surfaces",
            "materials",
            "sections",
            "constraints",
            "loads",
            "keyword_inventory",
        )
        mapping_tuples = {
            name: tuple(
                _json_mapping(item, f"{name} item")
                for item in _array(data[name], name)
            )
            for name in mapping_tuple_names
        }
        return cls(
            revision=_integer(data["revision"], "revision"),
            revision_hash=_string(data["revision_hash"], "revision_hash"),
            source_artifact_id=_string(
                data["source_artifact_id"],
                "source_artifact_id",
            ),
            source_sha256=_string(data["source_sha256"], "source_sha256"),
            model_name=_string(data["model_name"], "model_name"),
            node_count=_integer(data["node_count"], "node_count"),
            element_count=_integer(data["element_count"], "element_count"),
            dofs_per_node=_integer(data["dofs_per_node"], "dofs_per_node"),
            total_dofs=_integer(data["total_dofs"], "total_dofs"),
            element_types={
                _string(key, "element type"): _integer(count, "element count")
                for key, count in _mapping(
                    data["element_types"],
                    "element_types",
                ).items()
            },
            analysis_step=(
                None
                if data["analysis_step"] is None
                else _json_mapping(data["analysis_step"], "analysis_step")
            ),
            unit_context=(
                None
                if data["unit_context"] is None
                else UnitContext.from_dict(_mapping(data["unit_context"], "unit_context"))
            ),
            requested_queries=tuple(
                ResultQuery.from_dict(_mapping(item, "requested query"))
                for item in _array(data["requested_queries"], "requested_queries")
            ),
            export_formats=tuple(
                _enum(item, ExportFormat, "export_formats")
                for item in _array(data["export_formats"], "export_formats")
            ),
            diagnostics=tuple(
                Diagnostic.from_dict(_mapping(item, "diagnostic"))
                for item in _array(data["diagnostics"], "diagnostics")
            ),
            resource_class=_string(data["resource_class"], "resource_class"),
            collections_truncated=_boolean(
                data["collections_truncated"],
                "collections_truncated",
            ),
            **mapping_tuples,
        )


@dataclass(frozen=True)
class ArtifactRecord(JsonContract):
    """Opaque artifact metadata safe to expose outside the local store."""

    artifact_id: str
    kind: str
    display_path: str
    sha256: str
    size_bytes: int

    def __post_init__(self) -> None:
        _validate_identifier(self.artifact_id, "artifact_id")
        _validate_label(self.kind, "kind")
        _validate_relative_display_path(self.display_path)
        _validate_sha256(self.sha256)
        _nonnegative_int(self.size_bytes, "size_bytes")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Self:
        data = _strict_mapping(
            value,
            required={"artifact_id", "kind", "display_path", "sha256", "size_bytes"},
            context="ArtifactRecord",
        )
        return cls(
            artifact_id=_string(data["artifact_id"], "artifact_id"),
            kind=_string(data["kind"], "kind"),
            display_path=_string(data["display_path"], "display_path"),
            sha256=_string(data["sha256"], "sha256"),
            size_bytes=_integer(data["size_bytes"], "size_bytes"),
        )


@dataclass(frozen=True)
class ScalarResult(JsonContract):
    """One bounded numerical result with provenance and declared units."""

    query_kind: ResultQueryKind
    value: float
    unit: str
    measure: str
    run_id: str
    step: str
    node_id: int | None = None
    element_id: int | None = None
    region: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "query_kind",
            _enum(self.query_kind, ResultQueryKind, "query_kind"),
        )
        _finite_number(self.value, "value")
        for name in ("unit", "measure", "step"):
            _validate_label(getattr(self, name), name)
        _validate_identifier(self.run_id, "run_id")
        for name in ("node_id", "element_id"):
            value = getattr(self, name)
            if value is not None:
                _positive_int(value, name)
        if self.region is not None:
            _validate_label(self.region, "region")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Self:
        data = _strict_mapping(
            value,
            required={"query_kind", "value", "unit", "measure", "run_id", "step"},
            optional={"node_id", "element_id", "region"},
            context="ScalarResult",
        )
        return cls(
            query_kind=_enum(data["query_kind"], ResultQueryKind, "query_kind"),
            value=_number(data["value"], "value"),
            unit=_string(data["unit"], "unit"),
            measure=_string(data["measure"], "measure"),
            run_id=_string(data["run_id"], "run_id"),
            step=_string(data["step"], "step"),
            node_id=_optional_integer(data.get("node_id"), "node_id"),
            element_id=_optional_integer(data.get("element_id"), "element_id"),
            region=_optional_string(data.get("region"), "region"),
        )


@dataclass(frozen=True)
class ResultSummary(JsonContract):
    """Bounded results returned from a completed local worker run."""

    run_id: str
    step: str
    finite_vectors: bool
    scalars: tuple[ScalarResult, ...]
    diagnostics: tuple[Diagnostic, ...] = ()

    def __post_init__(self) -> None:
        _validate_identifier(self.run_id, "run_id")
        _validate_label(self.step, "step")
        if not isinstance(self.finite_vectors, bool):
            raise SchemaValidationError("finite_vectors must be a boolean")
        object.__setattr__(self, "scalars", tuple(self.scalars))
        object.__setattr__(self, "diagnostics", tuple(self.diagnostics))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Self:
        data = _versioned_mapping(
            value,
            required={"run_id", "step", "finite_vectors", "scalars", "diagnostics"},
            context="ResultSummary",
        )
        return cls(
            run_id=_string(data["run_id"], "run_id"),
            step=_string(data["step"], "step"),
            finite_vectors=_boolean(data["finite_vectors"], "finite_vectors"),
            scalars=tuple(
                ScalarResult.from_dict(_mapping(item, "scalar"))
                for item in _array(data["scalars"], "scalars")
            ),
            diagnostics=tuple(
                Diagnostic.from_dict(_mapping(item, "diagnostic"))
                for item in _array(data["diagnostics"], "diagnostics")
            ),
        )


@dataclass(frozen=True)
class ToolResult(JsonContract):
    """Normalized result returned by every model-callable tool."""

    ok: bool
    session_id: str
    input_revision: int
    idempotency_key: str
    summary: str
    output_revision: int | None = None
    data: Mapping[str, Any] | None = None
    diagnostics: tuple[Diagnostic, ...] = ()
    artifacts: tuple[ArtifactRecord, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.ok, bool):
            raise SchemaValidationError("ok must be a boolean")
        _validate_identifier(self.session_id, "session_id")
        _nonnegative_int(self.input_revision, "input_revision")
        if self.output_revision is not None:
            _positive_int(self.output_revision, "output_revision")
        _validate_identifier(self.idempotency_key, "idempotency_key")
        _validate_label(self.summary, "summary", max_length=2048)
        if self.data is not None:
            object.__setattr__(self, "data", _json_mapping(self.data, "data"))
        object.__setattr__(self, "diagnostics", tuple(self.diagnostics))
        object.__setattr__(self, "artifacts", tuple(self.artifacts))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Self:
        data = _versioned_mapping(
            value,
            required={
                "ok",
                "session_id",
                "input_revision",
                "idempotency_key",
                "summary",
                "output_revision",
                "data",
                "diagnostics",
                "artifacts",
            },
            context="ToolResult",
        )
        return cls(
            ok=_boolean(data["ok"], "ok"),
            session_id=_string(data["session_id"], "session_id"),
            input_revision=_integer(data["input_revision"], "input_revision"),
            idempotency_key=_string(data["idempotency_key"], "idempotency_key"),
            summary=_string(data["summary"], "summary"),
            output_revision=_optional_integer(
                data["output_revision"],
                "output_revision",
            ),
            data=(
                None
                if data["data"] is None
                else _json_mapping(data["data"], "data")
            ),
            diagnostics=tuple(
                Diagnostic.from_dict(_mapping(item, "diagnostic"))
                for item in _array(data["diagnostics"], "diagnostics")
            ),
            artifacts=tuple(
                ArtifactRecord.from_dict(_mapping(item, "artifact"))
                for item in _array(data["artifacts"], "artifacts")
            ),
        )


@dataclass(frozen=True)
class RunManifest(JsonContract):
    """Auditable numerical run record stored separately from conversation text."""

    session_id: str
    revision: int
    revision_hash: str
    run_id: str
    status: RunStatus
    source_sha256: str
    repository_commit: str | None
    runtime_versions: Mapping[str, str]
    unit_context: UnitContext
    analysis_step: str
    tool_parameters: Mapping[str, Any]
    validation_diagnostics: tuple[Diagnostic, ...]
    result_summary_artifact_id: str | None
    artifacts: tuple[ArtifactRecord, ...]
    timestamps: Mapping[str, str]
    durations_seconds: Mapping[str, float]
    diagnostics: tuple[Diagnostic, ...] = ()

    def __post_init__(self) -> None:
        _validate_identifier(self.session_id, "session_id")
        _positive_int(self.revision, "revision")
        _validate_sha256(self.revision_hash, "revision_hash")
        _validate_identifier(self.run_id, "run_id")
        object.__setattr__(self, "status", _enum(self.status, RunStatus, "status"))
        _validate_sha256(self.source_sha256)
        if self.repository_commit is not None:
            _validate_label(self.repository_commit, "repository_commit", max_length=128)
        _validate_label(self.analysis_step, "analysis_step")
        if self.result_summary_artifact_id is not None:
            _validate_identifier(
                self.result_summary_artifact_id,
                "result_summary_artifact_id",
            )
        object.__setattr__(
            self,
            "runtime_versions",
            {
                _string(key, "runtime version key"): _string(
                    item,
                    "runtime version",
                )
                for key, item in self.runtime_versions.items()
            },
        )
        object.__setattr__(
            self,
            "tool_parameters",
            _json_mapping(self.tool_parameters, "tool_parameters"),
        )
        object.__setattr__(
            self,
            "timestamps",
            {
                _string(key, "timestamp key"): _string(value, "timestamp")
                for key, value in self.timestamps.items()
            },
        )
        object.__setattr__(
            self,
            "durations_seconds",
            {
                _string(key, "duration key"): _nonnegative_number(
                    value,
                    "duration",
                )
                for key, value in self.durations_seconds.items()
            },
        )
        object.__setattr__(
            self,
            "validation_diagnostics",
            tuple(self.validation_diagnostics),
        )
        object.__setattr__(self, "artifacts", tuple(self.artifacts))
        object.__setattr__(self, "diagnostics", tuple(self.diagnostics))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Self:
        names = {field.name for field in fields(cls)}
        data = _versioned_mapping(value, required=names, context="RunManifest")
        return cls(
            session_id=_string(data["session_id"], "session_id"),
            revision=_integer(data["revision"], "revision"),
            revision_hash=_string(data["revision_hash"], "revision_hash"),
            run_id=_string(data["run_id"], "run_id"),
            status=_enum(data["status"], RunStatus, "status"),
            source_sha256=_string(data["source_sha256"], "source_sha256"),
            repository_commit=_optional_string(
                data["repository_commit"],
                "repository_commit",
            ),
            runtime_versions={
                _string(key, "runtime version key"): _string(item, "runtime version")
                for key, item in _mapping(
                    data["runtime_versions"],
                    "runtime_versions",
                ).items()
            },
            unit_context=UnitContext.from_dict(
                _mapping(data["unit_context"], "unit_context")
            ),
            analysis_step=_string(data["analysis_step"], "analysis_step"),
            tool_parameters=_json_mapping(
                data["tool_parameters"],
                "tool_parameters",
            ),
            validation_diagnostics=tuple(
                Diagnostic.from_dict(_mapping(item, "validation diagnostic"))
                for item in _array(
                    data["validation_diagnostics"],
                    "validation_diagnostics",
                )
            ),
            result_summary_artifact_id=_optional_string(
                data["result_summary_artifact_id"],
                "result_summary_artifact_id",
            ),
            artifacts=tuple(
                ArtifactRecord.from_dict(_mapping(item, "artifact"))
                for item in _array(data["artifacts"], "artifacts")
            ),
            timestamps={
                _string(key, "timestamp key"): _string(item, "timestamp")
                for key, item in _mapping(data["timestamps"], "timestamps").items()
            },
            durations_seconds={
                _string(key, "duration key"): _number(item, "duration")
                for key, item in _mapping(
                    data["durations_seconds"],
                    "durations_seconds",
                ).items()
            },
            diagnostics=tuple(
                Diagnostic.from_dict(_mapping(item, "diagnostic"))
                for item in _array(data["diagnostics"], "diagnostics")
            ),
        )


def load_contract_json(contract: type[JsonContract], payload: str) -> JsonContract:
    """Parse one JSON object and delegate to a strict ``from_dict`` method."""

    try:
        value = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise SchemaValidationError(f"invalid JSON: {exc.msg}") from exc
    if not isinstance(value, dict):
        raise SchemaValidationError("contract JSON must contain an object")
    factory = getattr(contract, "from_dict", None)
    if factory is None:
        raise TypeError(f"{contract.__name__} does not define from_dict")
    return factory(value)


def _json_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        result = {}
        for item in fields(value):
            item_value = getattr(value, item.name)
            if (
                item_value is None
                and item.metadata.get("json_omit_none", False)
            ):
                continue
            result[item.name] = _json_value(item_value)
        if isinstance(value, JsonContract) and "schema_version" not in result:
            result = {"schema_version": value.schema_version, **result}
        return result
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and not math.isfinite(value):
            raise SchemaValidationError("JSON numbers must be finite")
        return value
    raise SchemaValidationError(
        f"value of type {type(value).__name__} is not JSON-compatible"
    )


def _strict_mapping(
    value: Mapping[str, Any],
    *,
    required: set[str],
    optional: set[str] | None = None,
    context: str,
) -> dict[str, Any]:
    data = dict(_mapping(value, context))
    optional = optional or set()
    if (
        "schema_version" in data
        and "schema_version" not in required
        and "schema_version" not in optional
    ):
        version = _integer(data.pop("schema_version"), "schema_version")
        if version != SCHEMA_VERSION:
            raise SchemaValidationError(
                f"unsupported schema_version {version}; expected {SCHEMA_VERSION}"
            )
    missing = required - data.keys()
    unknown = data.keys() - required - optional
    if missing:
        raise SchemaValidationError(
            f"{context} is missing fields: {', '.join(sorted(missing))}"
        )
    if unknown:
        raise SchemaValidationError(
            f"{context} has unknown fields: {', '.join(sorted(unknown))}"
        )
    return data


def _versioned_mapping(
    value: Mapping[str, Any],
    *,
    required: set[str],
    context: str,
) -> dict[str, Any]:
    data = _strict_mapping(
        value,
        required=required | {"schema_version"},
        context=context,
    )
    version = _integer(data.pop("schema_version"), "schema_version")
    if version != SCHEMA_VERSION:
        raise SchemaValidationError(
            f"unsupported schema_version {version}; expected {SCHEMA_VERSION}"
        )
    return data


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SchemaValidationError(f"{name} must be an object")
    if not all(isinstance(key, str) for key in value):
        raise SchemaValidationError(f"{name} keys must be strings")
    return value


def _json_mapping(value: Any, name: str) -> dict[str, Any]:
    mapping = dict(_mapping(value, name))
    normalized = _json_value(mapping)
    if not isinstance(normalized, dict):
        raise SchemaValidationError(f"{name} must be an object")
    return normalized


def _array(value: Any, name: str) -> list[Any] | tuple[Any, ...]:
    if not isinstance(value, (list, tuple)):
        raise SchemaValidationError(f"{name} must be an array")
    return value


def _string(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise SchemaValidationError(f"{name} must be a string")
    return value


def _optional_string(value: Any, name: str) -> str | None:
    return None if value is None else _string(value, name)


def _integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SchemaValidationError(f"{name} must be an integer")
    return value


def _optional_integer(value: Any, name: str) -> int | None:
    return None if value is None else _integer(value, name)


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SchemaValidationError(f"{name} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise SchemaValidationError(f"{name} must be finite")
    return result


def _boolean(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise SchemaValidationError(f"{name} must be a boolean")
    return value


def _enum(value: Any, enum_type: type[Enum], name: str) -> Any:
    if isinstance(value, enum_type):
        return value
    try:
        return enum_type(value)
    except (TypeError, ValueError) as exc:
        allowed = ", ".join(repr(item.value) for item in enum_type)
        raise SchemaValidationError(f"{name} must be one of {allowed}") from exc


def _validate_label(value: Any, name: str, *, max_length: int = 128) -> str:
    text = _string(value, name)
    if not text.strip():
        raise SchemaValidationError(f"{name} must not be blank")
    if len(text) > max_length:
        raise SchemaValidationError(
            f"{name} must contain at most {max_length} characters"
        )
    if any(
        unicodedata.category(character) in {"Cc", "Cf", "Cs", "Zl", "Zp"}
        for character in text
    ):
        raise SchemaValidationError(
            f"{name} must be a single-line label without control characters"
        )
    return text


def _validate_identifier(
    value: Any,
    name: str,
    *,
    allow_upper: bool = False,
) -> str:
    text = _validate_label(value, name)
    allowed = set("abcdefghijklmnopqrstuvwxyz0123456789_-")
    if allow_upper:
        allowed.update("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    if any(character not in allowed for character in text):
        raise SchemaValidationError(
            f"{name} may contain only letters, digits, underscores, and hyphens"
        )
    return text


def _validate_sha256(value: Any, name: str = "source_sha256") -> str:
    text = _string(value, name)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise SchemaValidationError(f"{name} must be a lowercase SHA-256 digest")
    return text


def _validate_relative_display_path(value: Any) -> str:
    text = _validate_label(value, "display_path", max_length=1024)
    normalized = text.replace("\\", "/")
    if normalized.startswith("/") or ":" in normalized.split("/", 1)[0]:
        raise SchemaValidationError("display_path must be relative")
    if any(part in {"", ".", ".."} for part in normalized.split("/")):
        raise SchemaValidationError("display_path contains an invalid segment")
    return normalized


def _positive_int(value: Any, name: str) -> int:
    result = _integer(value, name)
    if result <= 0:
        raise SchemaValidationError(f"{name} must be greater than zero")
    return result


def _nonnegative_int(value: Any, name: str) -> int:
    result = _integer(value, name)
    if result < 0:
        raise SchemaValidationError(f"{name} must be nonnegative")
    return result


def _finite_number(value: Any, name: str) -> float:
    return _number(value, name)


def _positive_number(value: Any, name: str) -> float:
    result = _number(value, name)
    if result <= 0:
        raise SchemaValidationError(f"{name} must be greater than zero")
    return result


def _nonnegative_number(value: Any, name: str) -> float:
    result = _number(value, name)
    if result < 0:
        raise SchemaValidationError(f"{name} must be nonnegative")
    return result


__all__ = [
    "SCHEMA_VERSION",
    "AnalysisSummary",
    "ArtifactRecord",
    "Diagnostic",
    "DiagnosticSeverity",
    "ExportFormat",
    "ImportAnalysisSpec",
    "JsonContract",
    "ResourceLimits",
    "ResultQuery",
    "ResultQueryKind",
    "ResultSummary",
    "RunManifest",
    "RunStatus",
    "ScalarResult",
    "SchemaValidationError",
    "SessionPhase",
    "ToolResult",
    "UnitContext",
    "load_contract_json",
]
