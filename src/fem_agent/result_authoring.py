"""Strict A7 contracts for bounded queries over accepted native results."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
import math
from numbers import Real
from typing import Mapping, Protocol


RESULT_QUERY_SCHEMA_VERSION = "1.0"
RESULT_QUERY_TOOL_NAME = "query_accepted_result"
RESULT_CATALOG_TOOL_NAME = "read_accepted_result_catalog"
ANALYSIS_RUN_CATALOG_TOOL_NAME = "read_analysis_run_catalog"
ANALYSIS_RUN_CATALOG_MAX_LIMIT = 20


class ResultAuthoringError(ValueError):
    """Fail-closed A7 result-query contract error."""


class AgentResultVariable(str, Enum):
    DISPLACEMENT = "U"
    REACTION_FORCE = "RF"
    STRESS = "S"


@dataclass(frozen=True, slots=True)
class AnalysisRunCatalogEntry:
    """Provider-safe identity and lifecycle metadata for one analysis run."""

    run_id: str
    name: str
    step_name: str
    status: str
    artifact_id: str
    model_revision: int
    source_run_id: str | None
    result_id: str | None
    materialization_generation: int | None

    def __post_init__(self) -> None:
        for field_name in ("run_id", "artifact_id"):
            _bounded_text(getattr(self, field_name), field_name, maximum=128)
        _bounded_text(self.name, "name", maximum=160)
        _bounded_text(self.step_name, "step_name", maximum=256)
        _bounded_text(self.status, "status", maximum=16)
        if self.status not in {
            "pending",
            "running",
            "succeeded",
            "failed",
            "cancelled",
        }:
            raise ResultAuthoringError("run status is unsupported")
        _nonnegative_integer(self.model_revision, "model_revision")
        for field_name in ("source_run_id", "result_id"):
            value = getattr(self, field_name)
            if value is not None:
                _bounded_text(value, field_name, maximum=128)
        if self.materialization_generation is not None:
            _nonnegative_integer(
                self.materialization_generation,
                "materialization_generation",
            )
        if (self.result_id is None) != (
            self.materialization_generation is None
        ):
            raise ResultAuthoringError(
                "result_id and materialization_generation must appear together"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "name": self.name,
            "step_name": self.step_name,
            "status": self.status,
            "artifact_id": self.artifact_id,
            "model_revision": self.model_revision,
            "source_run_id": self.source_run_id,
            "result_id": self.result_id,
            "materialization_generation": self.materialization_generation,
        }


@dataclass(frozen=True, slots=True)
class AnalysisRunCatalog:
    """One bounded offset page over accepted local analysis-run metadata."""

    document_id: str
    session_id: str
    session_revision: int
    selected_run_id: str | None
    displayed_result_run_id: str | None
    cursor: int
    limit: int
    runs: tuple[AnalysisRunCatalogEntry, ...]
    next_cursor: int | None
    total_count: int

    def __post_init__(self) -> None:
        for field_name in ("document_id", "session_id"):
            _bounded_text(getattr(self, field_name), field_name, maximum=128)
        _nonnegative_integer(self.session_revision, "session_revision")
        for field_name in ("selected_run_id", "displayed_result_run_id"):
            value = getattr(self, field_name)
            if value is not None:
                _bounded_text(value, field_name, maximum=256)
        _nonnegative_integer(self.cursor, "cursor")
        if (
            type(self.limit) is not int
            or self.limit < 1
            or self.limit > ANALYSIS_RUN_CATALOG_MAX_LIMIT
        ):
            raise ResultAuthoringError("run catalog limit is outside its bound")
        if type(self.runs) is not tuple or any(
            type(item) is not AnalysisRunCatalogEntry for item in self.runs
        ):
            raise TypeError("runs must be a tuple of AnalysisRunCatalogEntry")
        if len(self.runs) > self.limit:
            raise ResultAuthoringError("run catalog page exceeds its limit")
        _nonnegative_integer(self.total_count, "total_count")
        if self.cursor > self.total_count:
            raise ResultAuthoringError("run catalog cursor exceeds total_count")
        if self.next_cursor is not None:
            _nonnegative_integer(self.next_cursor, "next_cursor")
            if self.next_cursor <= self.cursor or self.next_cursor > self.total_count:
                raise ResultAuthoringError("next_cursor is invalid")

    @property
    def truncated(self) -> bool:
        return self.next_cursor is not None

    @property
    def omitted_run_count(self) -> int:
        return max(0, self.total_count - self.cursor - len(self.runs))

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": RESULT_QUERY_SCHEMA_VERSION,
            "document_id": self.document_id,
            "session_id": self.session_id,
            "session_revision": self.session_revision,
            "selected_run_id": self.selected_run_id,
            "displayed_result_run_id": self.displayed_result_run_id,
            "cursor": self.cursor,
            "limit": self.limit,
            "runs": [item.to_dict() for item in self.runs],
            "next_cursor": self.next_cursor,
            "total_count": self.total_count,
            "truncated": self.truncated,
            "omitted_run_count": self.omitted_run_count,
        }


class AgentResultAggregation(str, Enum):
    MAXIMUM = "maximum"
    MINIMUM = "minimum"
    ABSOLUTE_EXTREME = "absolute_extreme"
    SUM = "sum"


@dataclass(frozen=True, slots=True)
class AgentResultField:
    """One bounded current-catalog field exposed to the Agent Provider."""

    variable: AgentResultVariable
    position: str
    components: tuple[str, ...]
    unit: str

    def __post_init__(self) -> None:
        if type(self.variable) is not AgentResultVariable:
            raise TypeError("variable must be AgentResultVariable")
        _bounded_text(self.position, "position", maximum=128)
        _bounded_text(self.unit, "unit", maximum=128)
        if type(self.components) is not tuple:
            raise TypeError("components must be a tuple")
        if not self.components or len(self.components) > 32:
            raise ResultAuthoringError(
                "components must contain from 1 through 32 values"
            )
        for component in self.components:
            _bounded_text(component, "component", maximum=128)
        if len(set(self.components)) != len(self.components):
            raise ResultAuthoringError("components must be unique")

    def to_dict(self) -> dict[str, object]:
        return {
            "variable": self.variable.value,
            "position": self.position,
            "components": list(self.components),
            "unit": self.unit,
        }


@dataclass(frozen=True, slots=True)
class AcceptedResultSource:
    """Provider-safe identity of one accepted native solve result."""

    result_id: str
    session_id: str
    artifact_id: str
    model_revision: int
    step_name: str
    run_id: str

    def __post_init__(self) -> None:
        for field_name in (
            "result_id",
            "session_id",
            "artifact_id",
            "step_name",
            "run_id",
        ):
            _bounded_text(getattr(self, field_name), field_name, maximum=256)
        _nonnegative_integer(self.model_revision, "model_revision")

    def to_dict(self) -> dict[str, object]:
        return {
            "result_id": self.result_id,
            "session_id": self.session_id,
            "artifact_id": self.artifact_id,
            "model_revision": self.model_revision,
            "step_name": self.step_name,
            "run_id": self.run_id,
        }

    @classmethod
    def from_dict(cls, value: object) -> AcceptedResultSource:
        data = _strict_mapping(
            value,
            {
                "result_id",
                "session_id",
                "artifact_id",
                "model_revision",
                "step_name",
                "run_id",
            },
            "accepted result source",
        )
        return cls(
            result_id=_exact_string(data["result_id"], "result_id"),
            session_id=_exact_string(data["session_id"], "session_id"),
            artifact_id=_exact_string(data["artifact_id"], "artifact_id"),
            model_revision=_exact_integer(
                data["model_revision"],
                "model_revision",
            ),
            step_name=_exact_string(data["step_name"], "step_name"),
            run_id=_exact_string(data["run_id"], "run_id"),
        )


@dataclass(frozen=True, slots=True)
class AgentResultQuery:
    """One complete, unambiguous query bound to an exact accepted result."""

    variable: AgentResultVariable
    component: str
    position: str
    region: str
    aggregation: AgentResultAggregation
    expected_source: AcceptedResultSource
    expected_materialization_generation: int

    def __post_init__(self) -> None:
        if type(self.variable) is not AgentResultVariable:
            raise TypeError("variable must be AgentResultVariable")
        if type(self.aggregation) is not AgentResultAggregation:
            raise TypeError("aggregation must be AgentResultAggregation")
        for field_name in ("component", "position", "region"):
            _bounded_text(getattr(self, field_name), field_name, maximum=256)
        if type(self.expected_source) is not AcceptedResultSource:
            raise TypeError("expected_source must be AcceptedResultSource")
        _nonnegative_integer(
            self.expected_materialization_generation,
            "expected_materialization_generation",
        )
        if (
            self.aggregation is AgentResultAggregation.SUM
            and self.variable is not AgentResultVariable.REACTION_FORCE
        ):
            raise ResultAuthoringError(
                "sum is supported only for reaction force RF"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": RESULT_QUERY_SCHEMA_VERSION,
            "variable": self.variable.value,
            "component": self.component,
            "position": self.position,
            "region": self.region,
            "aggregation": self.aggregation.value,
            "expected_source": self.expected_source.to_dict(),
            "expected_materialization_generation": (
                self.expected_materialization_generation
            ),
        }

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    @classmethod
    def from_dict(cls, value: object) -> AgentResultQuery:
        data = _strict_mapping(
            value,
            {
                "schema_version",
                "variable",
                "component",
                "position",
                "region",
                "aggregation",
                "expected_source",
                "expected_materialization_generation",
            },
            "Agent result query",
        )
        if data["schema_version"] != RESULT_QUERY_SCHEMA_VERSION:
            raise ResultAuthoringError(
                "Agent result query schema version is unsupported"
            )
        return cls(
            variable=_exact_enum(
                data["variable"],
                AgentResultVariable,
                "variable",
            ),
            component=_exact_string(data["component"], "component"),
            position=_exact_string(data["position"], "position"),
            region=_exact_string(data["region"], "region"),
            aggregation=_exact_enum(
                data["aggregation"],
                AgentResultAggregation,
                "aggregation",
            ),
            expected_source=AcceptedResultSource.from_dict(
                data["expected_source"]
            ),
            expected_materialization_generation=_exact_integer(
                data["expected_materialization_generation"],
                "expected_materialization_generation",
            ),
        )


@dataclass(frozen=True, slots=True)
class AgentResultCatalog:
    """Bounded fields and named regions for one exact accepted generation."""

    source: AcceptedResultSource
    materialization_generation: int
    fields: tuple[AgentResultField, ...]
    nodal_regions: tuple[str, ...]
    element_regions: tuple[str, ...]

    def __post_init__(self) -> None:
        if type(self.source) is not AcceptedResultSource:
            raise TypeError("source must be AcceptedResultSource")
        _nonnegative_integer(
            self.materialization_generation,
            "materialization_generation",
        )
        if type(self.fields) is not tuple or any(
            type(item) is not AgentResultField for item in self.fields
        ):
            raise TypeError("fields must be a tuple of AgentResultField")
        if not self.fields or len(self.fields) > 32:
            raise ResultAuthoringError(
                "catalog fields must contain from 1 through 32 values"
            )
        for field_name in ("nodal_regions", "element_regions"):
            values = getattr(self, field_name)
            if type(values) is not tuple:
                raise TypeError(f"{field_name} must be a tuple")
            if len(values) > 128 or len(set(values)) != len(values):
                raise ResultAuthoringError(
                    f"{field_name} exceeds its bound or contains duplicates"
                )
            for value in values:
                _bounded_text(value, field_name, maximum=256)

    def to_dict(self) -> dict[str, object]:
        return {
            "source": self.source.to_dict(),
            "materialization_generation": self.materialization_generation,
            "fields": [item.to_dict() for item in self.fields],
            "nodal_regions": list(self.nodal_regions),
            "element_regions": list(self.element_regions),
        }


@dataclass(frozen=True, slots=True)
class AgentResultLocation:
    """Bounded FEM identity for one aggregate location; coordinates stay local."""

    association: str
    node_id: int | None = None
    element_id: int | None = None
    integration_point: int | None = None
    local_node: int | None = None

    def __post_init__(self) -> None:
        _bounded_text(self.association, "association", maximum=64)
        for field_name in (
            "node_id",
            "element_id",
            "integration_point",
            "local_node",
        ):
            value = getattr(self, field_name)
            if value is not None:
                _positive_integer(value, field_name)
        if self.node_id is None and self.element_id is None:
            raise ResultAuthoringError(
                "a result location requires a node or element identity"
            )

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {"association": self.association}
        for field_name in (
            "node_id",
            "element_id",
            "integration_point",
            "local_node",
        ):
            value = getattr(self, field_name)
            if value is not None:
                result[field_name] = value
        return result


@dataclass(frozen=True, slots=True)
class AgentResultScalar:
    """One finite engineering scalar with complete accepted-result provenance."""

    variable: AgentResultVariable
    component: str
    position: str
    region: str
    aggregation: AgentResultAggregation
    value: float
    unit: str
    source: AcceptedResultSource
    materialization_generation: int
    location: AgentResultLocation | None = None

    def __post_init__(self) -> None:
        if type(self.variable) is not AgentResultVariable:
            raise TypeError("variable must be AgentResultVariable")
        if type(self.aggregation) is not AgentResultAggregation:
            raise TypeError("aggregation must be AgentResultAggregation")
        for field_name in ("component", "position", "region", "unit"):
            _bounded_text(getattr(self, field_name), field_name, maximum=256)
        if isinstance(self.value, bool) or not isinstance(self.value, Real):
            raise TypeError("value must be a real number")
        numeric = float(self.value)
        if not math.isfinite(numeric):
            raise ResultAuthoringError("value must be finite")
        object.__setattr__(self, "value", numeric)
        if type(self.source) is not AcceptedResultSource:
            raise TypeError("source must be AcceptedResultSource")
        _nonnegative_integer(
            self.materialization_generation,
            "materialization_generation",
        )
        if (
            self.location is not None
            and type(self.location) is not AgentResultLocation
        ):
            raise TypeError("location must be AgentResultLocation or None")
        if (
            self.aggregation is AgentResultAggregation.SUM
            and self.location is not None
        ):
            raise ResultAuthoringError(
                "an aggregate sum is identified by its region, not one location"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "variable": self.variable.value,
            "component": self.component,
            "position": self.position,
            "region": self.region,
            "aggregation": self.aggregation.value,
            "value": self.value,
            "unit": self.unit,
            "source": self.source.to_dict(),
            "materialization_generation": self.materialization_generation,
            "location": (
                None if self.location is None else self.location.to_dict()
            ),
        }


@dataclass(frozen=True, slots=True)
class AgentResultDiagnostic:
    """Bounded, stable failure returned without any engineering scalar."""

    code: str
    message: str
    retryable: bool
    clarification_required: bool
    phase: str = "A7"

    def __post_init__(self) -> None:
        _bounded_text(self.code, "code", maximum=128)
        _bounded_text(self.message, "message", maximum=1024)
        _bounded_text(self.phase, "phase", maximum=16)
        if type(self.retryable) is not bool:
            raise TypeError("retryable must be boolean")
        if type(self.clarification_required) is not bool:
            raise TypeError("clarification_required must be boolean")

    def to_dict(self) -> dict[str, object]:
        return {
            "code": self.code,
            "message": self.message,
            "retryable": self.retryable,
            "clarification_required": self.clarification_required,
            "phase": self.phase,
        }


@dataclass(frozen=True, slots=True)
class AgentResultQueryResponse:
    """Exactly one scalar or one-or-more bounded diagnostics."""

    scalar: AgentResultScalar | None = None
    diagnostics: tuple[AgentResultDiagnostic, ...] = ()

    def __post_init__(self) -> None:
        if self.scalar is not None and type(self.scalar) is not AgentResultScalar:
            raise TypeError("scalar must be AgentResultScalar or None")
        if type(self.diagnostics) is not tuple or any(
            type(item) is not AgentResultDiagnostic
            for item in self.diagnostics
        ):
            raise TypeError(
                "diagnostics must be a tuple of AgentResultDiagnostic"
            )
        if (self.scalar is None) == (not self.diagnostics):
            raise ResultAuthoringError(
                "response requires exactly one scalar or diagnostics"
            )
        if len(self.diagnostics) > 8:
            raise ResultAuthoringError("response diagnostics exceed the A7 bound")

    @property
    def ok(self) -> bool:
        return self.scalar is not None

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": RESULT_QUERY_SCHEMA_VERSION,
            "ok": self.ok,
            "scalar": None if self.scalar is None else self.scalar.to_dict(),
            "diagnostics": [item.to_dict() for item in self.diagnostics],
        }

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    @classmethod
    def success(cls, scalar: AgentResultScalar) -> AgentResultQueryResponse:
        return cls(scalar=scalar)

    @classmethod
    def failure(
        cls,
        code: str,
        message: str,
        *,
        retryable: bool,
        clarification_required: bool,
    ) -> AgentResultQueryResponse:
        return cls(
            diagnostics=(
                AgentResultDiagnostic(
                    code=code,
                    message=message,
                    retryable=retryable,
                    clarification_required=clarification_required,
                ),
            )
        )


@dataclass(frozen=True, slots=True)
class AgentResultCatalogResponse:
    """Exactly one bounded catalog or one-or-more bounded diagnostics."""

    catalog: AgentResultCatalog | None = None
    diagnostics: tuple[AgentResultDiagnostic, ...] = ()

    def __post_init__(self) -> None:
        if self.catalog is not None and type(self.catalog) is not AgentResultCatalog:
            raise TypeError("catalog must be AgentResultCatalog or None")
        if type(self.diagnostics) is not tuple or any(
            type(item) is not AgentResultDiagnostic
            for item in self.diagnostics
        ):
            raise TypeError(
                "diagnostics must be a tuple of AgentResultDiagnostic"
            )
        if (self.catalog is None) == (not self.diagnostics):
            raise ResultAuthoringError(
                "catalog response requires exactly one catalog or diagnostics"
            )
        if len(self.diagnostics) > 8:
            raise ResultAuthoringError("catalog diagnostics exceed the A7 bound")

    @property
    def ok(self) -> bool:
        return self.catalog is not None

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": RESULT_QUERY_SCHEMA_VERSION,
            "ok": self.ok,
            "catalog": (
                None if self.catalog is None else self.catalog.to_dict()
            ),
            "diagnostics": [item.to_dict() for item in self.diagnostics],
        }

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    @classmethod
    def success(
        cls,
        catalog: AgentResultCatalog,
    ) -> AgentResultCatalogResponse:
        return cls(catalog=catalog)

    @classmethod
    def failure(
        cls,
        code: str,
        message: str,
        *,
        retryable: bool,
        clarification_required: bool,
    ) -> AgentResultCatalogResponse:
        return cls(
            diagnostics=(
                AgentResultDiagnostic(
                    code=code,
                    message=message,
                    retryable=retryable,
                    clarification_required=clarification_required,
                ),
            )
        )


class AgentResultQueryPort(Protocol):
    """Cross-layer read-only protocol; implementations return bounded DTOs."""

    def catalog(self, run_id: str | None = None) -> AgentResultCatalogResponse: ...

    def query(self, request: AgentResultQuery) -> AgentResultQueryResponse: ...


class FakeAgentResultQueryPort:
    """Deterministic Fake Provider boundary for A7 tool and explanation tests."""

    def __init__(
        self,
        responses: Mapping[
            AgentResultQuery,
            AgentResultQueryResponse,
        ] | None = None,
        *,
        catalog_response: AgentResultCatalogResponse | None = None,
    ) -> None:
        self._responses = dict(responses or {})
        self.calls: list[AgentResultQuery] = []
        self.catalog_calls = 0
        self.catalog_run_ids: list[str | None] = []
        self._catalog_response = catalog_response

    def register(
        self,
        request: AgentResultQuery,
        response: AgentResultQueryResponse,
    ) -> None:
        if type(request) is not AgentResultQuery:
            raise TypeError("request must be AgentResultQuery")
        if type(response) is not AgentResultQueryResponse:
            raise TypeError("response must be AgentResultQueryResponse")
        self._responses[request] = response

    def query(self, request: AgentResultQuery) -> AgentResultQueryResponse:
        if type(request) is not AgentResultQuery:
            raise TypeError("request must be AgentResultQuery")
        self.calls.append(request)
        try:
            return self._responses[request]
        except KeyError:
            return AgentResultQueryResponse.failure(
                "result.query.not_configured",
                "The requested accepted-result scalar is not configured.",
                retryable=False,
                clarification_required=True,
            )

    def catalog(self, run_id: str | None = None) -> AgentResultCatalogResponse:
        self.catalog_calls += 1
        self.catalog_run_ids.append(run_id)
        if self._catalog_response is not None:
            return self._catalog_response
        return AgentResultCatalogResponse.failure(
            "result.catalog.not_configured",
            "The accepted-result catalog is not configured.",
            retryable=False,
            clarification_required=True,
        )


class AgentResultQueryBridge:
    """Strict model-callable A7 query boundary over an injected local port."""

    def __init__(self, port: AgentResultQueryPort) -> None:
        if not all(
            callable(getattr(port, name, None))
            for name in ("catalog", "query")
        ):
            raise TypeError("port must implement the Agent result query protocol")
        self._port = port

    @property
    def port(self) -> AgentResultQueryPort:
        return self._port

    def query(
        self,
        request: AgentResultQuery | Mapping[str, object],
    ) -> AgentResultQueryResponse:
        normalized = (
            request
            if type(request) is AgentResultQuery
            else AgentResultQuery.from_dict(request)
        )
        response = self._port.query(normalized)
        if type(response) is not AgentResultQueryResponse:
            raise TypeError(
                "result query port must return AgentResultQueryResponse"
        )
        return response

    def catalog(self, run_id: str | None = None) -> AgentResultCatalogResponse:
        if run_id is not None:
            run_id = _exact_string(run_id, "run_id")
            _bounded_text(run_id, "run_id", maximum=128)
        response = (
            self._port.catalog()
            if run_id is None
            else self._port.catalog(run_id)
        )
        if type(response) is not AgentResultCatalogResponse:
            raise TypeError(
                "result query port must return AgentResultCatalogResponse"
            )
        return response


def result_catalog_tool_schema() -> dict[str, object]:
    """Return the closed schema for displayed or explicitly selected results."""

    return {
        "name": RESULT_CATALOG_TOOL_NAME,
        "description": (
            "Read bounded field and named-region identities for the selected "
            "result or for one exact accepted run_id."
        ),
        "input_schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "run_id": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 128,
                }
            },
        },
    }


def analysis_run_catalog_tool_schema() -> dict[str, object]:
    """Return the closed, hard-bounded run-catalog pagination schema."""

    return {
        "name": ANALYSIS_RUN_CATALOG_TOOL_NAME,
        "description": "Read one bounded page of local analysis-run metadata.",
        "input_schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "cursor": {"type": "integer", "minimum": 0},
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": ANALYSIS_RUN_CATALOG_MAX_LIMIT,
                },
            },
        },
    }


read_analysis_run_catalog_tool_schema = analysis_run_catalog_tool_schema


def result_query_tool_schema() -> dict[str, object]:
    """Return the closed model-callable schema; no display or confirm tools."""

    source_properties = {
        "result_id": {"type": "string"},
        "session_id": {"type": "string"},
        "artifact_id": {"type": "string"},
        "model_revision": {"type": "integer", "minimum": 0},
        "step_name": {"type": "string"},
        "run_id": {"type": "string"},
    }
    properties: dict[str, object] = {
        "schema_version": {
            "type": "string",
            "const": RESULT_QUERY_SCHEMA_VERSION,
        },
        "variable": {
            "type": "string",
            "enum": [item.value for item in AgentResultVariable],
        },
        "component": {"type": "string"},
        "position": {"type": "string"},
        "region": {"type": "string"},
        "aggregation": {
            "type": "string",
            "enum": [item.value for item in AgentResultAggregation],
        },
        "expected_source": {
            "type": "object",
            "additionalProperties": False,
            "required": list(source_properties),
            "properties": source_properties,
        },
        "expected_materialization_generation": {
            "type": "integer",
            "minimum": 0,
        },
    }
    return {
        "name": RESULT_QUERY_TOOL_NAME,
        "description": (
            "Read one bounded scalar from the exact current accepted FEM result."
        ),
        "input_schema": {
            "type": "object",
            "additionalProperties": False,
            "required": list(properties),
            "properties": properties,
        },
    }


def explain_result_response(response: AgentResultQueryResponse) -> str:
    """Format only the local scalar or local diagnostic supplied by the port."""

    if type(response) is not AgentResultQueryResponse:
        raise TypeError("response must be AgentResultQueryResponse")
    if response.scalar is None:
        return response.diagnostics[0].message
    scalar = response.scalar
    location = _location_text(scalar.location, scalar.region)
    return (
        f"{scalar.variable.value} {scalar.component} 的"
        f"{_aggregation_text(scalar.aggregation)}为 "
        f"{scalar.value:.12g} {scalar.unit}，{location}；"
        f"run {scalar.source.run_id}，step {scalar.source.step_name}。"
    )


def _aggregation_text(value: AgentResultAggregation) -> str:
    return {
        AgentResultAggregation.MAXIMUM: "最大值",
        AgentResultAggregation.MINIMUM: "最小值",
        AgentResultAggregation.ABSOLUTE_EXTREME: "绝对值极值",
        AgentResultAggregation.SUM: "合计",
    }[value]


def _location_text(
    location: AgentResultLocation | None,
    region: str,
) -> str:
    if location is None:
        return f"区域 {region}"
    identities = []
    if location.node_id is not None:
        identities.append(f"节点 {location.node_id}")
    if location.element_id is not None:
        identities.append(f"单元 {location.element_id}")
    if location.integration_point is not None:
        identities.append(f"积分点 {location.integration_point}")
    if location.local_node is not None:
        identities.append(f"局部节点 {location.local_node}")
    return f"区域 {region}，位置 " + " / ".join(identities)


def _strict_mapping(
    value: object,
    expected: set[str],
    label: str,
) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be an object")
    if any(type(key) is not str for key in value):
        raise TypeError(f"{label} keys must be strings")
    if set(value) != expected:
        raise ResultAuthoringError(f"{label} fields do not match the schema")
    return value


def _exact_string(value: object, label: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{label} must be a string")
    return value


def _exact_integer(value: object, label: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{label} must be an integer")
    return value


def _exact_enum(
    value: object,
    enum_type: type[Enum],
    label: str,
):
    if type(value) is not str:
        raise TypeError(f"{label} must be a string")
    try:
        return enum_type(value)
    except ValueError as error:
        raise ResultAuthoringError(f"{label} is unsupported") from error


def _bounded_text(value: object, label: str, *, maximum: int) -> str:
    if type(value) is not str:
        raise TypeError(f"{label} must be a string")
    if value != value.strip() or not value:
        raise ResultAuthoringError(
            f"{label} must be nonblank without surrounding whitespace"
        )
    if len(value) > maximum:
        raise ResultAuthoringError(f"{label} exceeds the A7 bound")
    if "\x00" in value:
        raise ResultAuthoringError(f"{label} contains a null character")
    return value


def _nonnegative_integer(value: object, label: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{label} must be an integer")
    if value < 0:
        raise ResultAuthoringError(f"{label} must be non-negative")
    return value


def _positive_integer(value: object, label: str) -> int:
    _nonnegative_integer(value, label)
    if value <= 0:
        raise ResultAuthoringError(f"{label} must be positive")
    return value


__all__ = [
    "ANALYSIS_RUN_CATALOG_MAX_LIMIT",
    "ANALYSIS_RUN_CATALOG_TOOL_NAME",
    "AnalysisRunCatalog",
    "AnalysisRunCatalogEntry",
    "RESULT_CATALOG_TOOL_NAME",
    "RESULT_QUERY_SCHEMA_VERSION",
    "RESULT_QUERY_TOOL_NAME",
    "AcceptedResultSource",
    "AgentResultAggregation",
    "AgentResultCatalog",
    "AgentResultCatalogResponse",
    "AgentResultDiagnostic",
    "AgentResultField",
    "AgentResultLocation",
    "AgentResultQuery",
    "AgentResultQueryBridge",
    "AgentResultQueryPort",
    "AgentResultQueryResponse",
    "AgentResultScalar",
    "AgentResultVariable",
    "FakeAgentResultQueryPort",
    "ResultAuthoringError",
    "analysis_run_catalog_tool_schema",
    "explain_result_response",
    "result_catalog_tool_schema",
    "read_analysis_run_catalog_tool_schema",
    "result_query_tool_schema",
]
