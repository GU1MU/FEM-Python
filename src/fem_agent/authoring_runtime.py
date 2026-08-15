"""A8 dynamic authoring workflow and provider-safe tool boundary."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from enum import Enum
import hashlib
import json
import math
import re
import threading
from typing import TYPE_CHECKING, Any
import uuid

from fem.geometry.construction_ir import (
    MAX_BOOLEAN_OPERANDS,
    MAX_CANONICAL_PAYLOAD_BYTES,
    MAX_DAG_DEPTH,
    MAX_NAME_LENGTH,
    MAX_NODES,
    MAX_NODE_ID_LENGTH,
    MAX_PATH_POINTS,
    MAX_PATTERN_INSTANCES,
    MAX_POLYGON_VERTICES,
    SCHEMA_VERSION as PLANAR_CONSTRUCTION_SCHEMA_VERSION,
)

from .authoring import (
    AuthoringContext,
    ProposalState,
    RequirementLedger,
    RequirementReview,
    RequirementReviewStatus,
    RequirementStatus,
)
from .diagnostics import DiagnosticCode, make_diagnostic
from .geometry_authoring import geometry_feature_catalog_tool_schema
from .providers.base import ToolDefinition
from .result_authoring import (
    ANALYSIS_RUN_CATALOG_TOOL_NAME,
    RESULT_CATALOG_TOOL_NAME,
    RESULT_COMPARISON_TOOL_NAME,
    RESULT_QUERY_TOOL_NAME,
    analysis_run_catalog_tool_schema,
    result_catalog_tool_schema,
    result_comparison_tool_schema,
    result_query_tool_schema,
)
from .workspace_catalog import workspace_documents_tool_schema
from .schemas import ToolResult

if TYPE_CHECKING:
    from .tools.registry import ToolExecutionContext


class AuthoringWorkflowStage(str, Enum):
    REQUIREMENTS = "requirements"
    REVIEW_PENDING = "review_pending"
    GEOMETRY_READY = "geometry_ready"
    GEOMETRY_PENDING = "geometry_pending"
    MESH_READY = "mesh_ready"
    MESH_PENDING = "mesh_pending"
    DEFINITIONS_READY = "definitions_ready"
    ANALYSIS_DEFINITIONS_READY = "analysis_definitions_ready"
    PREFLIGHT_READY = "preflight_ready"
    PREFLIGHT_PENDING = "preflight_pending"
    SOLVE_READY = "solve_ready"
    SOLVE_PENDING = "solve_pending"
    RESULTS_READY = "results_ready"
    PROJECT_SAVE_PENDING = "project_save_pending"
    DESTRUCTIVE_EDIT_PENDING = "destructive_edit_pending"
    STALE = "stale"
    CANCELLED = "cancelled"


# A turn snapshot is deliberately smaller than a tool result.  It is sent on
# every provider round, so keeping this budget independent from the general
# authoring payload prevents a large feature catalog from leaking into the
# conversation context.
AUTHORING_TURN_SNAPSHOT_MAX_BYTES = 8_192
AUTHORING_TURN_SNAPSHOT_MAX_ITEMS = 96
AUTHORING_TURN_SNAPSHOT_SCHEMA_VERSION = "1"
_SNAPSHOT_BUDGET_MARGIN_BYTES = 128
_SNAPSHOT_PREFERRED_NAMES = (
    "read_authoring_context",
    "read_geometry_feature_catalog",
    "read_geometry_edit_context",
    "prepare_geometry_edit",
    "edit_native_geometry",
)


@dataclass(frozen=True, slots=True)
class AuthoringToolOutcome:
    summary: str
    data: Mapping[str, object]
    ok: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.summary, str) or not self.summary.strip():
            raise ValueError("authoring tool summary must be non-blank")
        if not isinstance(self.data, Mapping):
            raise TypeError("authoring tool data must be an object")
        if not isinstance(self.ok, bool):
            raise TypeError("authoring tool ok must be boolean")
        object.__setattr__(
            self,
            "summary",
            _provider_safe_summary(self.summary),
        )
        object.__setattr__(self, "data", dict(self.data))


@dataclass(frozen=True, slots=True)
class AuthoringTurnSnapshot:
    """Minimal immutable authoring state projected at a provider turn.

    The snapshot is intentionally not an ``AuthoringContext`` replacement:
    canonical profile IDs and complete feature catalogs remain local tools.
    ``available`` is false until a typed context has been observed on the GUI
    owner thread; callers must not fill an unavailable snapshot from history.
    """

    available: bool = False
    source_kind: str | None = None
    workflow_stage: str | None = None
    document_id: str | None = None
    session_id: str | None = None
    session_revision: int | None = None
    active_part_id: str | None = None
    active_part_dimension: int | None = None
    active_part_recipe_kind: str | None = None
    active_part_suppressed: bool | None = None
    mesh_present: bool = False
    mesh_current: bool = False
    enabled_capabilities: tuple[str, ...] = ()
    published_tool_names: tuple[str, ...] = ()
    snapshot_generation: int = 0
    truncated: bool = False
    schema_version: str = AUTHORING_TURN_SNAPSHOT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if type(self.available) is not bool:
            raise TypeError("snapshot available must be boolean")
        if self.schema_version != AUTHORING_TURN_SNAPSHOT_SCHEMA_VERSION:
            raise ValueError("unsupported authoring turn snapshot schema")
        if (
            isinstance(self.snapshot_generation, bool)
            or not isinstance(self.snapshot_generation, int)
            or self.snapshot_generation < 0
        ):
            raise ValueError("snapshot_generation must be a non-negative integer")
        if self.session_revision is not None and (
            isinstance(self.session_revision, bool)
            or not isinstance(self.session_revision, int)
            or self.session_revision < 0
        ):
            raise ValueError("session_revision must be a non-negative integer")
        if (
            self.active_part_dimension is not None
            and self.active_part_dimension
            not in {
                1,
                2,
                3,
            }
        ):
            raise ValueError("active_part_dimension must be 1, 2, or 3")
        if type(self.mesh_present) is not bool or type(self.mesh_current) is not bool:
            raise TypeError("snapshot mesh flags must be boolean")
        if type(self.truncated) is not bool:
            raise TypeError("snapshot truncated must be boolean")

        for name in (
            "source_kind",
            "workflow_stage",
            "document_id",
            "session_id",
            "active_part_id",
            "active_part_recipe_kind",
        ):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _snapshot_text(value, name))
        if (
            self.active_part_suppressed is not None
            and type(self.active_part_suppressed) is not bool
        ):
            raise TypeError("active_part_suppressed must be boolean or null")
        object.__setattr__(
            self,
            "enabled_capabilities",
            _snapshot_names(self.enabled_capabilities),
        )
        object.__setattr__(
            self,
            "published_tool_names",
            _snapshot_names(self.published_tool_names),
        )

        # Enforce the byte budget after normalizing names.  A deterministic
        # tail trim is used only for non-preferred names; core read/prepare
        # capabilities are retained whenever they are present.
        payload = self._to_dict_unbounded()
        budget = AUTHORING_TURN_SNAPSHOT_MAX_BYTES - _SNAPSHOT_BUDGET_MARGIN_BYTES
        if _json_size_bytes(payload) > budget:
            payload["truncated"] = True
            capabilities = list(self.enabled_capabilities)
            tools = list(self.published_tool_names)
            while _json_size_bytes(payload) > budget:
                removed = False
                for values, field_name in (
                    (capabilities, "enabled_capabilities"),
                    (tools, "published_tool_names"),
                ):
                    preferred = {
                        item for item in _SNAPSHOT_PREFERRED_NAMES if item in values
                    }
                    candidate_index = next(
                        (
                            index
                            for index in range(len(values) - 1, -1, -1)
                            if values[index] not in preferred
                        ),
                        None,
                    )
                    if candidate_index is None:
                        continue
                    values.pop(candidate_index)
                    payload[field_name] = list(values)
                    removed = True
                    break
                if not removed:
                    break
            object.__setattr__(self, "enabled_capabilities", tuple(capabilities))
            object.__setattr__(self, "published_tool_names", tuple(tools))
            object.__setattr__(self, "truncated", True)

    @classmethod
    def unavailable(cls, *, generation: int = 0) -> "AuthoringTurnSnapshot":
        return cls(snapshot_generation=generation)

    @classmethod
    def from_context(
        cls,
        context: AuthoringContext | None,
        *,
        workflow_stage: AuthoringWorkflowStage | str | None,
        published_tool_names: Sequence[str] = (),
        generation: int = 0,
    ) -> "AuthoringTurnSnapshot":
        if context is None:
            return cls.unavailable(generation=generation)
        active = next(
            (item for item in context.parts if item.part_id == context.active_part_id),
            None,
        )
        callable_names = {
            str(name).strip() for name in published_tool_names if str(name).strip()
        }
        return cls(
            available=True,
            source_kind=context.binding.source_kind,
            workflow_stage=(
                None
                if workflow_stage is None
                else str(getattr(workflow_stage, "value", workflow_stage))
            ),
            document_id=context.binding.document_id,
            session_id=context.binding.session_id,
            session_revision=context.binding.session_revision,
            active_part_id=context.active_part_id,
            active_part_dimension=(None if active is None else active.dimension),
            active_part_recipe_kind=(None if active is None else active.recipe_kind),
            active_part_suppressed=(None if active is None else active.suppressed),
            mesh_present=context.mesh.present,
            mesh_current=context.mesh.current,
            enabled_capabilities=tuple(
                item.operation
                for item in context.capabilities
                if item.enabled and item.operation in callable_names
            ),
            published_tool_names=published_tool_names,
            snapshot_generation=generation,
        )

    @property
    def generation(self) -> int:
        """Alias used by adapters that call the freshness field generation."""

        return self.snapshot_generation

    @property
    def revision(self) -> int | None:
        return self.session_revision

    def _to_dict_unbounded(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "available": self.available,
            "source_kind": self.source_kind,
            "workflow_stage": self.workflow_stage,
            "document_id": self.document_id,
            "session_id": self.session_id,
            "session_revision": self.session_revision,
            "active_part_id": self.active_part_id,
            "active_part_dimension": self.active_part_dimension,
            "active_part_recipe_kind": self.active_part_recipe_kind,
            "active_part_suppressed": self.active_part_suppressed,
            "mesh_present": self.mesh_present,
            "mesh_current": self.mesh_current,
            "enabled_capabilities": list(self.enabled_capabilities),
            "published_tool_names": list(self.published_tool_names),
            "snapshot_generation": self.snapshot_generation,
            "truncated": self.truncated,
        }

    def to_provider_dict(self) -> dict[str, object]:
        """Return a detached bounded JSON projection for the provider."""

        return dict(self._to_dict_unbounded())

    def to_dict(self) -> dict[str, object]:
        return self.to_provider_dict()


@dataclass(frozen=True, slots=True)
class AuthoringTerminalRecord:
    operation: str
    state: str
    message: str


@dataclass(frozen=True, slots=True)
class PlanarConstructionAuditRecord:
    construction_digest: str
    stage: str
    diagnostic_code: str | None
    proposal_id: str | None
    terminal_state: str | None


@dataclass(frozen=True, slots=True)
class _PlanarConstructionRetryState:
    turn_id: str
    construction_digest: str
    recovery_digest: str
    diagnostic_code: str
    node_id: str | None
    allowed_fields: tuple[str, ...]
    attempt: int


@dataclass(frozen=True, slots=True)
class ProjectSaveProposalRecord:
    proposal_id: str
    proposal_hash: str
    target_document_id: str
    target_session_id: str
    base_session_revision: int
    resume_stage: AuthoringWorkflowStage
    state: ProposalState = ProposalState.PENDING_CONFIRMATION
    message: str = ""

    def __post_init__(self) -> None:
        identifier = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}")
        for field_name in (
            "proposal_id",
            "target_document_id",
            "target_session_id",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or identifier.fullmatch(value) is None:
                raise ValueError(f"{field_name} must be a bounded identifier")
        if (
            not isinstance(self.proposal_hash, str)
            or re.fullmatch(r"[0-9a-f]{64}", self.proposal_hash) is None
        ):
            raise ValueError("proposal_hash must be a SHA-256 hexadecimal value")
        if (
            type(self.base_session_revision) is not int
            or self.base_session_revision < 0
        ):
            raise ValueError("base_session_revision must be non-negative")
        object.__setattr__(
            self,
            "resume_stage",
            AuthoringWorkflowStage(self.resume_stage),
        )
        object.__setattr__(self, "state", ProposalState(self.state))
        object.__setattr__(self, "message", str(self.message).strip()[:512])


AuthoringToolHandler = Callable[
    [Mapping[str, object], "AuthoringWorkflowController"],
    AuthoringToolOutcome,
]
ContextReader = Callable[[], AuthoringContext | Mapping[str, object]]


_FORBIDDEN_CONFIRMATION_TOOL_NAMES = frozenset(
    {
        "accept_proposal",
        "confirm_mesh",
        "confirm_solve",
        "confirm_requirement_review",
        "reject_proposal",
        "cancel_operation",
    }
)
_NO_ARGUMENTS = {
    "type": "object",
    "properties": {},
    "required": [],
    "additionalProperties": False,
}
_ABSOLUTE_PATH = re.compile(
    r"""(?ix)
    (?:
        [a-z]:[\\/]
        |
        \\\\[^\\/\s]+[\\/][^\\/\s]+
        |
        (?<![a-z0-9])/(?:[^/\s]+/)+[^/\s]*
    )
    """
)
_FORBIDDEN_PAYLOAD_KEYS = frozenset(
    {
        "nodes",
        "node_ids",
        "node_coordinates",
        "coordinates",
        "elements",
        "element_ids",
        "connectivity",
        "records",
        "result_arrays",
        "result_provider",
        "model_session",
        "model_patch",
        "qt_object",
        "vtk",
        "gmsh_model",
        "raw_patch",
        "patch_before",
        "patch_after",
        "source_path",
        "absolute_path",
        "api_key",
        "credential",
    }
)
_MAX_PROVIDER_PAYLOAD_BYTES = 32_768
_MAX_PROVIDER_COLLECTION_ITEMS = 128
_MAX_PROVIDER_DEPTH = 12


def _snapshot_text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-blank string")
    normalized = value.strip()
    if len(normalized.encode("utf-8")) > 128:
        raise ValueError(f"{field_name} exceeds the snapshot identity budget")
    if any(ord(character) < 32 for character in normalized):
        raise ValueError(f"{field_name} contains control characters")
    return normalized


def _snapshot_names(values: Sequence[str]) -> tuple[str, ...]:
    if not isinstance(values, (tuple, list)):
        values = tuple(values)
    normalized = {
        _snapshot_text(value, "snapshot name")
        for value in values
        if isinstance(value, str) and value.strip()
    }
    ordered = sorted(normalized)
    preferred = [item for item in _SNAPSHOT_PREFERRED_NAMES if item in normalized]
    remainder = [item for item in ordered if item not in preferred]
    return tuple((preferred + remainder)[:AUTHORING_TURN_SNAPSHOT_MAX_ITEMS])


def _json_size_bytes(payload: Mapping[str, object]) -> int:
    return (
        len(
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        + 1
    )


_REQUIREMENT_SPECS: dict[str, dict[str, object]] = {
    "modeling_assumption": {
        "type": "string",
        "enum": ["plane_stress", "plane_strain"],
    },
    "length_unit": {"type": "string"},
    "force_unit": {"type": "string"},
    "stress_unit": {"type": "string"},
    "plate_thickness": {"type": "number", "exclusiveMinimum": 0},
    "young_modulus": {"type": "number", "exclusiveMinimum": 0},
    "poisson_ratio": {
        "type": "number",
        "exclusiveMinimum": -1,
        "exclusiveMaximum": 0.5,
    },
    "mesh_cell_shape": {
        "type": "string",
        "enum": [
            "line",
            "triangle",
            "quadrilateral",
            "tetrahedron",
            "hexahedron",
        ],
    },
    "mesh_order": {"type": "integer", "enum": [1, 2]},
    "mesh_global_size": {"type": "number", "exclusiveMinimum": 0},
    "line_element_type": {
        "type": "string",
        "enum": ["Truss2", "Beam2"],
    },
    "fixed_dofs": {
        "type": "array",
        "items": {"type": "integer", "minimum": 1, "maximum": 3},
        "minItems": 1,
        "maxItems": 3,
        "uniqueItems": True,
    },
    "load_type": {
        "type": "string",
        "enum": [
            "edge_traction",
            "edge_pressure",
            "surface_traction",
            "surface_pressure",
            "nodal",
        ],
    },
    "load_direction": {
        "type": "string",
        "enum": [
            "x",
            "y",
            "z",
            "inward_normal",
            "outward_normal",
        ],
    },
    "load_magnitude": {"type": "number"},
    "load_unit": {"type": "string"},
    "load_distribution": {"type": "string", "enum": ["uniform"]},
    "analysis_procedure": {"type": "string", "enum": ["static"]},
    "nlgeom": {"type": "boolean", "enum": [False]},
    "result_requests": {
        "type": "array",
        "items": {"type": "string", "enum": ["U", "S", "RF"]},
        "minItems": 1,
        "maxItems": 3,
        "uniqueItems": True,
    },
}
_REQUIRED_REQUIREMENTS = tuple(
    key for key in _REQUIREMENT_SPECS if key != "line_element_type"
)
_GEOMETRY_REQUIREMENTS = (
    "length_unit",
    "force_unit",
    "stress_unit",
)
_DEFAULT_GEOMETRY_REQUIREMENTS: dict[str, str] = {
    "length_unit": "mm",
    "force_unit": "N",
    "stress_unit": "MPa",
}
_DEFAULT_REQUIREMENT_SOURCE_TURN_ID = "agent-default"
_MESH_REQUIREMENTS = (
    "mesh_cell_shape",
    "mesh_order",
    "mesh_global_size",
)
_LINE_MESH_REQUIREMENTS = (
    "mesh_cell_shape",
    "mesh_order",
    "mesh_global_size",
    "line_element_type",
)
_DEFINITION_REQUIREMENTS = (
    "modeling_assumption",
    "plate_thickness",
    "young_modulus",
    "poisson_ratio",
)
_ANALYSIS_REQUIREMENTS = (
    "fixed_dofs",
    "load_type",
    "load_direction",
    "load_magnitude",
    "load_unit",
    "load_distribution",
    "analysis_procedure",
    "nlgeom",
    "result_requests",
)
_REQUIREMENT_GROUPS: dict[str, tuple[str, ...]] = {
    "geometry": _GEOMETRY_REQUIREMENTS,
    "mesh": _MESH_REQUIREMENTS,
    "definitions": _DEFINITION_REQUIREMENTS,
    "analysis": _ANALYSIS_REQUIREMENTS,
}
_HANDLER_REQUIREMENTS: dict[str, tuple[str, ...]] = {
    **_REQUIREMENT_GROUPS,
    "analysis": (
        "length_unit",
        "force_unit",
        "stress_unit",
        *_ANALYSIS_REQUIREMENTS,
    ),
}
_REQUIREMENT_GATE_BY_STAGE = {
    AuthoringWorkflowStage.REQUIREMENTS: "geometry",
    AuthoringWorkflowStage.GEOMETRY_READY: "geometry",
    AuthoringWorkflowStage.MESH_READY: "mesh",
    AuthoringWorkflowStage.DEFINITIONS_READY: "mesh",
    AuthoringWorkflowStage.ANALYSIS_DEFINITIONS_READY: "mesh",
    AuthoringWorkflowStage.PREFLIGHT_READY: "mesh",
    AuthoringWorkflowStage.SOLVE_READY: "mesh",
    AuthoringWorkflowStage.RESULTS_READY: "mesh",
}
_GATED_OPERATIONS_BY_STAGE = {
    AuthoringWorkflowStage.REQUIREMENTS: frozenset(
        {"prepare_geometry_proposal", "prepare_planar_construction_proposal"}
    ),
    AuthoringWorkflowStage.GEOMETRY_READY: frozenset(
        {"prepare_geometry_proposal", "prepare_planar_construction_proposal"}
    ),
    AuthoringWorkflowStage.MESH_READY: frozenset({"prepare_mesh_proposal"}),
    AuthoringWorkflowStage.DEFINITIONS_READY: frozenset({"prepare_mesh_proposal"}),
    AuthoringWorkflowStage.ANALYSIS_DEFINITIONS_READY: frozenset(
        {"prepare_mesh_proposal"}
    ),
    AuthoringWorkflowStage.PREFLIGHT_READY: frozenset({"prepare_mesh_proposal"}),
    AuthoringWorkflowStage.SOLVE_READY: frozenset({"prepare_mesh_proposal"}),
    AuthoringWorkflowStage.RESULTS_READY: frozenset({"prepare_mesh_proposal"}),
}


def _tool(
    name: str,
    description: str,
    parameters: Mapping[str, object],
) -> ToolDefinition:
    return ToolDefinition(name, description, parameters)


def _result_tool_definition(schema: Mapping[str, object]) -> ToolDefinition:
    return ToolDefinition(
        str(schema["name"]),
        str(schema["description"]),
        schema["input_schema"],  # type: ignore[arg-type]
    )


_READ_CONTEXT = _tool(
    "read_authoring_context",
    "Read the bounded current native authoring context and workflow stage.",
    _NO_ARGUMENTS,
)
_SET_REQUIREMENTS = _tool(
    "set_authoring_requirements",
    (
        "Record current-stage engineering values explicitly supplied by the "
        "user. Blank projects already provide mm-N-MPa geometry defaults; "
        "this tool overrides them only when the user requests other units. "
        "Complete geometry or mesh values may then be presented in the "
        "corresponding operation confirmation card."
    ),
    {
        "type": "object",
        "properties": {
            "turn_id": {"type": "string"},
            "requirements": {
                "type": "object",
                "properties": _REQUIREMENT_SPECS,
                "additionalProperties": False,
                "minProperties": 1,
            },
        },
        "required": ["turn_id", "requirements"],
        "additionalProperties": False,
    },
)
_REQUEST_REVIEW = _tool(
    "request_requirement_review",
    (
        "Create a local RequirementReview after every required value for the "
        "current stage has been explicitly supplied. This tool cannot confirm "
        "it."
    ),
    _NO_ARGUMENTS,
)

# Blank-project composite geometry uses the same bounded profile vocabulary as
# the existing planar proposal, then wraps that strict XY sketch in one native
# 3D recipe.  ``role``/``operation`` are optional annotations: topology
# containment remains the authority for deciding whether a contour is a hole.
_COMPOSITE_PROFILE_ITEM_SCHEMA = {
    "oneOf": [
        {
            "type": "object",
            "properties": {
                "kind": {"const": "rectangle"},
                "x": {"type": "number"},
                "y": {"type": "number"},
                "width": {"type": "number", "exclusiveMinimum": 0},
                "height": {"type": "number", "exclusiveMinimum": 0},
                "role": {"type": "string", "enum": ["material", "hole"]},
                "operation": {
                    "type": "string",
                    "enum": ["material", "cut"],
                },
            },
            "required": ["kind", "x", "y", "width", "height"],
            "additionalProperties": False,
        },
        {
            "type": "object",
            "properties": {
                "kind": {"const": "circle"},
                "center_x": {"type": "number"},
                "center_y": {"type": "number"},
                "radius": {"type": "number", "exclusiveMinimum": 0},
                "role": {"type": "string", "enum": ["material", "hole"]},
                "operation": {
                    "type": "string",
                    "enum": ["material", "cut"],
                },
            },
            "required": ["kind", "center_x", "center_y", "radius"],
            "additionalProperties": False,
        },
        {
            "type": "object",
            "properties": {
                "kind": {"const": "polygon"},
                "vertices": {
                    "type": "array",
                    "minItems": 3,
                    "maxItems": 64,
                    "items": {
                        "type": "object",
                        "properties": {
                            "x": {"type": "number"},
                            "y": {"type": "number"},
                        },
                        "required": ["x", "y"],
                        "additionalProperties": False,
                    },
                },
                "role": {"type": "string", "enum": ["material", "hole"]},
                "operation": {
                    "type": "string",
                    "enum": ["material", "cut"],
                },
            },
            "required": ["kind", "vertices"],
            "additionalProperties": False,
        },
    ]
}
_COMPOSITE_PROFILE_ARRAY_SCHEMA = {
    "type": "array",
    "minItems": 1,
    "maxItems": 32,
    "description": (
        "Closed boundaries of the final Part. The first planar boundary is "
        "the outer material profile; contained boundaries are cuts. Encode a "
        "non-convex cutout or any cutout whose centerline would branch as one "
        "ordered, closed, non-self-intersecting polygon with role=hole and "
        "operation=cut. Do not replace the requested boundary with a bounding "
        "rectangle plus a material island."
    ),
    "items": _COMPOSITE_PROFILE_ITEM_SCHEMA,
}
_COMPOSITE_PATH_SCHEMA = {
    "type": "object",
    "properties": {
        "points": {
            "type": "array",
            "minItems": 2,
            "maxItems": 64,
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "minLength": 1, "maxLength": 64},
                    "x": {"type": "number"},
                    "y": {"type": "number"},
                    "z": {"type": "number"},
                },
                "required": ["name", "x", "y", "z"],
                "additionalProperties": False,
            },
        },
        "members": {
            "type": "array",
            "minItems": 1,
            "maxItems": 63,
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "minLength": 1, "maxLength": 64},
                    "start": {"type": "string", "minLength": 1, "maxLength": 64},
                    "end": {"type": "string", "minLength": 1, "maxLength": 64},
                },
                "required": ["name", "start", "end"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["points", "members"],
    "additionalProperties": False,
}
_PREPARE_GEOMETRY = _tool(
    "prepare_geometry_proposal",
    (
        "Build and present a revision-bound geometry proposal from general "
        "planar profiles, a constant-width path slot cut through a plate, a "
        "named spatial wire, or a supported solid primitive. Use wire only for "
        "a final 1D beam, truss, or frame Part; never use it as a centerline "
        "placeholder for a planar or solid Part. Record the project "
        "unit context first. The geometry is not added until the local GUI "
        "control is clicked."
    ),
    {
        "type": "object",
        "properties": {
            "part_function": {
                "type": "string",
                "minLength": 1,
                "maxLength": 96,
            },
            "geometry": {
                "oneOf": [
                    {
                        "type": "object",
                        "properties": {
                            "kind": {"const": "wire"},
                            "points": {
                                "type": "array",
                                "minItems": 2,
                                "maxItems": 128,
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "name": {
                                            "type": "string",
                                            "minLength": 1,
                                            "maxLength": 96,
                                        },
                                        "x": {"type": "number"},
                                        "y": {"type": "number"},
                                        "z": {"type": "number"},
                                    },
                                    "required": ["name", "x", "y", "z"],
                                    "additionalProperties": False,
                                },
                            },
                            "members": {
                                "type": "array",
                                "minItems": 1,
                                "maxItems": 128,
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "name": {
                                            "type": "string",
                                            "minLength": 1,
                                            "maxLength": 96,
                                        },
                                        "start": {
                                            "type": "string",
                                            "minLength": 1,
                                            "maxLength": 96,
                                        },
                                        "end": {
                                            "type": "string",
                                            "minLength": 1,
                                            "maxLength": 96,
                                        },
                                    },
                                    "required": ["name", "start", "end"],
                                    "additionalProperties": False,
                                },
                            },
                        },
                        "required": ["kind", "points", "members"],
                        "additionalProperties": False,
                    },
                    {
                        "type": "object",
                        "properties": {
                            "kind": {"const": "planar_profiles"},
                            "profiles": _COMPOSITE_PROFILE_ARRAY_SCHEMA,
                        },
                        "required": ["kind", "profiles"],
                        "additionalProperties": False,
                    },
                    {
                        "type": "object",
                        "properties": {
                            "kind": {"const": "box"},
                            "width": {
                                "type": "number",
                                "exclusiveMinimum": 0,
                            },
                            "depth": {
                                "type": "number",
                                "exclusiveMinimum": 0,
                            },
                            "height": {
                                "type": "number",
                                "exclusiveMinimum": 0,
                            },
                        },
                        "required": ["kind", "width", "depth", "height"],
                        "additionalProperties": False,
                    },
                    {
                        "type": "object",
                        "properties": {
                            "kind": {"const": "cylinder"},
                            "radius": {
                                "type": "number",
                                "exclusiveMinimum": 0,
                            },
                            "height": {
                                "type": "number",
                                "exclusiveMinimum": 0,
                            },
                        },
                        "required": ["kind", "radius", "height"],
                        "additionalProperties": False,
                    },
                    {
                        "type": "object",
                        "properties": {
                            "kind": {"const": "extruded_profiles"},
                            "profiles": _COMPOSITE_PROFILE_ARRAY_SCHEMA,
                            "height": {
                                "type": "number",
                                "exclusiveMinimum": 0,
                            },
                            "provisional": {"type": "boolean"},
                        },
                        "required": ["kind", "profiles", "height"],
                        "additionalProperties": False,
                    },
                    {
                        "type": "object",
                        "properties": {
                            "kind": {"const": "extruded_path_slot_plate"},
                            "plate": {
                                "type": "object",
                                "properties": {
                                    "x": {"type": "number"},
                                    "y": {"type": "number"},
                                    "width": {
                                        "type": "number",
                                        "exclusiveMinimum": 0,
                                    },
                                    "height": {
                                        "type": "number",
                                        "exclusiveMinimum": 0,
                                    },
                                },
                                "required": ["x", "y", "width", "height"],
                                "additionalProperties": False,
                            },
                            "slot_path": {
                                "type": "array",
                                "minItems": 2,
                                "maxItems": 32,
                                "description": (
                                    "Ordered open XY centerline for a slot fully "
                                    "defined by one non-branching path and one "
                                    "constant width. This is a cutout, not a wire Part."
                                ),
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "x": {"type": "number"},
                                        "y": {"type": "number"},
                                    },
                                    "required": ["x", "y"],
                                    "additionalProperties": False,
                                },
                            },
                            "slot_width": {
                                "type": "number",
                                "exclusiveMinimum": 0,
                            },
                            "height": {
                                "type": "number",
                                "exclusiveMinimum": 0,
                            },
                            "provisional": {"type": "boolean"},
                        },
                        "required": [
                            "kind",
                            "plate",
                            "slot_path",
                            "slot_width",
                            "height",
                        ],
                        "additionalProperties": False,
                    },
                    {
                        "type": "object",
                        "properties": {
                            "kind": {"const": "path_swept_profile"},
                            "profiles": _COMPOSITE_PROFILE_ARRAY_SCHEMA,
                            "path": _COMPOSITE_PATH_SCHEMA,
                            "frame_strategy": {
                                "type": "string",
                                "enum": ["fixed", "transport"],
                            },
                            "provisional": {"type": "boolean"},
                        },
                        "required": [
                            "kind",
                            "profiles",
                            "path",
                            "frame_strategy",
                        ],
                        "additionalProperties": False,
                    },
                ]
            },
        },
        "required": ["part_function", "geometry"],
        "additionalProperties": False,
    },
)

# Keep the legacy composite decoder callable for one migration cycle while the
# provider sees one planar-construction path.  The remaining general branches
# are deliberately disjoint from Planar Construction IR.
_LEGACY_PREPARE_GEOMETRY = _PREPARE_GEOMETRY
_LEGACY_COMPOSITE_GEOMETRY_KINDS = frozenset(
    {
        "planar_profiles",
        "extruded_profiles",
        "extruded_path_slot_plate",
        "path_swept_profile",
    }
)
_prepare_geometry_parameters = dict(_PREPARE_GEOMETRY.parameters)
_prepare_geometry_properties = dict(_prepare_geometry_parameters["properties"])
_provider_geometry_schema = dict(_prepare_geometry_properties["geometry"])
_provider_geometry_schema["oneOf"] = [
    branch
    for branch in _provider_geometry_schema["oneOf"]
    if branch.get("properties", {}).get("kind", {}).get("const")
    not in _LEGACY_COMPOSITE_GEOMETRY_KINDS
]
_prepare_geometry_properties["geometry"] = _provider_geometry_schema
_prepare_geometry_parameters["properties"] = _prepare_geometry_properties
_PREPARE_GEOMETRY = ToolDefinition(
    _PREPARE_GEOMETRY.name,
    (
        "Prepare one revision-bound general geometry proposal for a named "
        "spatial wire or a supported simple solid primitive. Planar regions "
        "and any planar-derived 3D output use "
        "prepare_planar_construction_proposal. The model changes only after "
        "the local GUI control is clicked."
    ),
    _prepare_geometry_parameters,
)
_READ_GEOMETRY_EDIT_CONTEXT = _tool(
    "read_geometry_edit_context",
    (
        "Read a bounded editable projection of one existing native Part. "
        "Use this before changing an accepted geometry and again after success. "
        "For planar sketches it returns exact Profile/hole counts and the generic "
        "freeform closed-contour policy."
    ),
    {
        "type": "object",
        "properties": {
            "part_id": {"type": "string", "minLength": 1, "maxLength": 128},
        },
        "required": ["part_id"],
        "additionalProperties": False,
    },
)
_PREPARE_GEOMETRY_EDIT = _tool(
    "prepare_geometry_edit",
    (
        "Prepare an in-place, revision-bound edit of an existing native Part. "
        "Selected-Profile extrusion requires explicit canonical source_face_ids "
        "from read_geometry_feature_catalog and creates one independent Part per "
        "selected material Profile. Revolution and path sweep require one explicit "
        "canonical Profile; path sweep also requires an ordered open polyline and "
        "a fixed or transport frame. Exact Part/Body Boolean requires explicit "
        "target, tool, fuse/cut operation, result name, and the sole supported "
        "tool-consumption policy. A two-dimensional fuse or cut uses planar_boolean "
        "with one detached rectangle, circle, polygon, or path-stroke tool and is "
        "appended to native feature history. The edit runs only after the local "
        "GUI control is clicked."
    ),
    {
        "type": "object",
        "properties": {
            "part_id": {"type": "string", "minLength": 1, "maxLength": 128},
            "edit": {
                "oneOf": [
                    {
                        "type": "object",
                        "properties": {
                            "operation": {"const": "extrude_profiles"},
                            "source_face_ids": {
                                "type": "array",
                                "minItems": 1,
                                "maxItems": 32,
                                "uniqueItems": True,
                                "items": {
                                    "type": "string",
                                    "pattern": "^face:[^\\s]+$",
                                    "maxLength": 192,
                                },
                                "description": (
                                    "Explicit canonical material Profile IDs "
                                    "from read_geometry_feature_catalog."
                                ),
                            },
                            "height": {
                                "type": "number",
                                "exclusiveMinimum": 0,
                            },
                        },
                        "required": ["operation", "source_face_ids", "height"],
                        "additionalProperties": False,
                    },
                    {
                        "type": "object",
                        "properties": {
                            "operation": {"const": "revolve_profile"},
                            "source_face_id": {
                                "type": "string",
                                "pattern": "^face:[^\\s]+$",
                                "maxLength": 192,
                            },
                            "axis": {
                                "type": "string",
                                "enum": ["x", "y", "z"],
                            },
                            "angle_degrees": {
                                "type": "number",
                                "exclusiveMinimum": 0,
                                "maximum": 360,
                            },
                        },
                        "required": [
                            "operation",
                            "source_face_id",
                            "axis",
                            "angle_degrees",
                        ],
                        "additionalProperties": False,
                    },
                    {
                        "type": "object",
                        "properties": {
                            "operation": {"const": "path_sweep_profile"},
                            "source_face_id": {
                                "type": "string",
                                "pattern": "^face:[^\\s]+$",
                                "maxLength": 192,
                            },
                            "path": {
                                "type": "object",
                                "properties": {
                                    "points": {
                                        "type": "array",
                                        "minItems": 2,
                                        "maxItems": 64,
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "name": {
                                                    "type": "string",
                                                    "minLength": 1,
                                                    "maxLength": 64,
                                                },
                                                "x": {"type": "number"},
                                                "y": {"type": "number"},
                                                "z": {"type": "number"},
                                            },
                                            "required": ["name", "x", "y", "z"],
                                            "additionalProperties": False,
                                        },
                                    },
                                    "members": {
                                        "type": "array",
                                        "minItems": 1,
                                        "maxItems": 63,
                                        "description": (
                                            "Explicit traversal order; ordinary 1D graph order is never inferred."
                                        ),
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "name": {
                                                    "type": "string",
                                                    "minLength": 1,
                                                    "maxLength": 64,
                                                },
                                                "start": {
                                                    "type": "string",
                                                    "minLength": 1,
                                                    "maxLength": 64,
                                                },
                                                "end": {
                                                    "type": "string",
                                                    "minLength": 1,
                                                    "maxLength": 64,
                                                },
                                            },
                                            "required": ["name", "start", "end"],
                                            "additionalProperties": False,
                                        },
                                    },
                                },
                                "required": ["points", "members"],
                                "additionalProperties": False,
                            },
                            "frame_strategy": {
                                "type": "string",
                                "enum": ["fixed", "transport"],
                            },
                        },
                        "required": [
                            "operation",
                            "source_face_id",
                            "path",
                            "frame_strategy",
                        ],
                        "additionalProperties": False,
                    },
                    {
                        "type": "object",
                        "properties": {
                            "operation": {"const": "planar_boolean"},
                            "boolean_operation": {
                                "type": "string",
                                "enum": ["fuse", "cut"],
                            },
                            "tool": {
                                "oneOf": [
                                    {
                                        "type": "object",
                                        "properties": {
                                            "kind": {"const": "rectangle"},
                                            "x": {"type": "number"},
                                            "y": {"type": "number"},
                                            "width": {
                                                "type": "number",
                                                "exclusiveMinimum": 0,
                                            },
                                            "height": {
                                                "type": "number",
                                                "exclusiveMinimum": 0,
                                            },
                                        },
                                        "required": [
                                            "kind",
                                            "x",
                                            "y",
                                            "width",
                                            "height",
                                        ],
                                        "additionalProperties": False,
                                    },
                                    {
                                        "type": "object",
                                        "properties": {
                                            "kind": {"const": "circle"},
                                            "center_x": {"type": "number"},
                                            "center_y": {"type": "number"},
                                            "radius": {
                                                "type": "number",
                                                "exclusiveMinimum": 0,
                                            },
                                        },
                                        "required": [
                                            "kind",
                                            "center_x",
                                            "center_y",
                                            "radius",
                                        ],
                                        "additionalProperties": False,
                                    },
                                    {
                                        "type": "object",
                                        "properties": {
                                            "kind": {"const": "polygon"},
                                            "vertices": {
                                                "type": "array",
                                                "minItems": 3,
                                                "maxItems": 64,
                                                "items": {
                                                    "type": "object",
                                                    "properties": {
                                                        "x": {"type": "number"},
                                                        "y": {"type": "number"},
                                                    },
                                                    "required": ["x", "y"],
                                                    "additionalProperties": False,
                                                },
                                            },
                                        },
                                        "required": ["kind", "vertices"],
                                        "additionalProperties": False,
                                    },
                                    {
                                        "type": "object",
                                        "properties": {
                                            "kind": {"const": "path_stroke"},
                                            "points": {
                                                "type": "array",
                                                "minItems": 2,
                                                "maxItems": 64,
                                                "items": {
                                                    "type": "object",
                                                    "properties": {
                                                        "x": {"type": "number"},
                                                        "y": {"type": "number"},
                                                    },
                                                    "required": ["x", "y"],
                                                    "additionalProperties": False,
                                                },
                                            },
                                            "width": {
                                                "type": "number",
                                                "exclusiveMinimum": 0,
                                            },
                                            "cap": {
                                                "type": "string",
                                                "enum": ["butt", "square", "round"],
                                            },
                                            "join": {
                                                "type": "string",
                                                "enum": ["miter", "bevel", "round"],
                                            },
                                        },
                                        "required": [
                                            "kind",
                                            "points",
                                            "width",
                                            "cap",
                                            "join",
                                        ],
                                        "additionalProperties": False,
                                    },
                                ]
                            },
                        },
                        "required": ["operation", "boolean_operation", "tool"],
                        "additionalProperties": False,
                    },
                    {
                        "type": "object",
                        "properties": {
                            "operation": {"const": "part_boolean"},
                            "boolean_operation": {
                                "type": "string",
                                "enum": ["fuse", "cut"],
                            },
                            "tool_part_id": {
                                "type": "string",
                                "pattern": "^P[1-9][0-9]*$",
                            },
                            "result_name": {
                                "type": "string",
                                "minLength": 1,
                                "maxLength": 128,
                            },
                            "tool_handling": {
                                "const": "consume_tool_part",
                            },
                        },
                        "required": [
                            "operation",
                            "boolean_operation",
                            "tool_part_id",
                            "result_name",
                            "tool_handling",
                        ],
                        "additionalProperties": False,
                    },
                    {
                        "type": "object",
                        "properties": {
                            "operation": {"const": "body_boolean"},
                            "boolean_operation": {
                                "type": "string",
                                "enum": ["fuse", "cut"],
                            },
                            "target_body_id": {
                                "type": "string",
                                "pattern": "^B[1-9][0-9]*$",
                            },
                            "tool_body_id": {
                                "type": "string",
                                "pattern": "^B[1-9][0-9]*$",
                            },
                            "result_name": {
                                "type": "string",
                                "minLength": 1,
                                "maxLength": 128,
                            },
                            "tool_handling": {
                                "const": "consume_tool_body",
                            },
                        },
                        "required": [
                            "operation",
                            "boolean_operation",
                            "target_body_id",
                            "tool_body_id",
                            "result_name",
                            "tool_handling",
                        ],
                        "additionalProperties": False,
                    },
                    {
                        "type": "object",
                        "properties": {
                            "operation": {"const": "add_circle"},
                            "center_x": {"type": "number"},
                            "center_y": {"type": "number"},
                            "radius": {
                                "type": "number",
                                "exclusiveMinimum": 0,
                            },
                        },
                        "required": [
                            "operation",
                            "center_x",
                            "center_y",
                            "radius",
                        ],
                        "additionalProperties": False,
                    },
                    {
                        "type": "object",
                        "properties": {
                            "operation": {"const": "add_rectangle"},
                            "x": {"type": "number"},
                            "y": {"type": "number"},
                            "width": {
                                "type": "number",
                                "exclusiveMinimum": 0,
                            },
                            "height": {
                                "type": "number",
                                "exclusiveMinimum": 0,
                            },
                        },
                        "required": [
                            "operation",
                            "x",
                            "y",
                            "width",
                            "height",
                        ],
                        "additionalProperties": False,
                    },
                    {
                        "type": "object",
                        "properties": {
                            "operation": {"const": "add_polygon"},
                            "vertices": {
                                "type": "array",
                                "minItems": 3,
                                "maxItems": 64,
                                "description": (
                                    "One ordered, non-self-intersecting boundary; "
                                    "the last vertex closes back to the first. "
                                    "Prefer this for arbitrary freeform silhouettes."
                                ),
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "x": {"type": "number"},
                                        "y": {"type": "number"},
                                    },
                                    "required": ["x", "y"],
                                    "additionalProperties": False,
                                },
                            },
                        },
                        "required": ["operation", "vertices"],
                        "additionalProperties": False,
                    },
                    {
                        "type": "object",
                        "properties": {
                            "operation": {"const": "add_path_slot"},
                            "points": {
                                "type": "array",
                                "minItems": 2,
                                "maxItems": 64,
                                "description": (
                                    "Ordered open, non-branching centerline of one "
                                    "connected constant-width slot. Do not submit "
                                    "offset boundary vertices or disconnected rectangles."
                                ),
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "x": {"type": "number"},
                                        "y": {"type": "number"},
                                    },
                                    "required": ["x", "y"],
                                    "additionalProperties": False,
                                },
                            },
                            "width": {
                                "type": "number",
                                "exclusiveMinimum": 0,
                                "description": "Full slot width.",
                            },
                            "cap": {
                                "type": "string",
                                "enum": ["butt", "square", "round"],
                            },
                            "join": {
                                "type": "string",
                                "enum": ["miter", "bevel", "round"],
                            },
                        },
                        "required": ["operation", "points", "width", "cap", "join"],
                        "additionalProperties": False,
                    },
                    {
                        "type": "object",
                        "properties": {
                            "operation": {"const": "update_point"},
                            "point_id": {
                                "type": "string",
                                "minLength": 1,
                                "maxLength": 128,
                            },
                            "x": {"type": "number"},
                            "y": {"type": "number"},
                        },
                        "required": ["operation", "point_id"],
                        "minProperties": 3,
                        "additionalProperties": False,
                    },
                    {
                        "type": "object",
                        "properties": {
                            "operation": {"const": "update_circle"},
                            "circle_id": {
                                "type": "string",
                                "minLength": 1,
                                "maxLength": 128,
                            },
                            "center_x": {"type": "number"},
                            "center_y": {"type": "number"},
                            "radius": {
                                "type": "number",
                                "exclusiveMinimum": 0,
                            },
                        },
                        "required": ["operation", "circle_id"],
                        "minProperties": 3,
                        "additionalProperties": False,
                    },
                    {
                        "type": "object",
                        "properties": {
                            "operation": {"const": "translate"},
                            "dx": {"type": "number"},
                            "dy": {"type": "number"},
                            "dz": {"type": "number"},
                        },
                        "required": ["operation", "dx", "dy"],
                        "additionalProperties": False,
                    },
                    {
                        "type": "object",
                        "properties": {
                            "operation": {"const": "rotate"},
                            "axis": {
                                "type": "string",
                                "enum": ["x", "y", "z"],
                            },
                            "angle_degrees": {"type": "number"},
                        },
                        "required": ["operation", "axis", "angle_degrees"],
                        "additionalProperties": False,
                    },
                ]
            },
        },
        "required": ["part_id", "edit"],
        "additionalProperties": False,
    },
)
# Profile transforms retain a compatibility dispatcher in the GUI bridge, but
# the provider-facing general edit schema no longer advertises those three
# large union branches.  Dedicated tools above are the sole discovery seam.
_legacy_geometry_edit_parameters = dict(_PREPARE_GEOMETRY_EDIT.parameters)
_legacy_edit = dict(_legacy_geometry_edit_parameters["properties"]["edit"])
_PROFILE_TRANSFORM_OPERATION_CONSTS = frozenset(
    {"extrude_profiles", "revolve_profile", "path_sweep_profile"}
)
_legacy_edit["oneOf"] = [
    branch
    for branch in _legacy_edit["oneOf"]
    if branch.get("properties", {}).get("operation", {}).get("const")
    not in _PROFILE_TRANSFORM_OPERATION_CONSTS
]
_DELETE_CIRCLES_EDIT_SCHEMA = {
    "type": "object",
    "properties": {
        "operation": {"const": "delete_circles"},
        "circle_ids": {
            "type": "array",
            "minItems": 1,
            "maxItems": 32,
            "uniqueItems": True,
            "items": {"type": "string", "minLength": 1, "maxLength": 128},
        },
    },
    "required": ["operation", "circle_ids"],
    "additionalProperties": False,
}
_REPLACE_CIRCLE_PATTERN_EDIT_SCHEMA = {
    "type": "object",
    "properties": {
        "operation": {"const": "replace_circle_pattern"},
        "target_circle_ids": {
            "type": "array",
            "minItems": 1,
            "maxItems": 32,
            "uniqueItems": True,
            "items": {"type": "string", "minLength": 1, "maxLength": 128},
        },
        "count": {"type": "integer", "minimum": 1, "maximum": 32},
        "start_center_x": {"type": "number"},
        "start_center_y": {"type": "number"},
        "spacing_x": {"type": "number"},
        "spacing_y": {"type": "number"},
        "radius": {"type": "number", "exclusiveMinimum": 0},
    },
    "required": [
        "operation",
        "target_circle_ids",
        "count",
        "start_center_x",
        "start_center_y",
        "spacing_x",
        "spacing_y",
        "radius",
    ],
    "additionalProperties": False,
}
_SKETCH_ID_SCHEMA = {"type": "string", "minLength": 1, "maxLength": 128}
_PLANAR_POINT_REF_SCHEMA = {
    "oneOf": [
        {
            "type": "object",
            "properties": {"point_id": _SKETCH_ID_SCHEMA},
            "required": ["point_id"],
            "additionalProperties": False,
        },
        {
            "type": "object",
            "properties": {"x": {"type": "number"}, "y": {"type": "number"}},
            "required": ["x", "y"],
            "additionalProperties": False,
        },
    ]
}


def _constraint_edit_schema(
    kind: str,
    properties: dict[str, object],
    required: list[str],
) -> dict[str, object]:
    return {
        "type": "object",
        "properties": {
            "kind": {"const": kind},
            **properties,
            "enabled": {"type": "boolean"},
        },
        "required": ["kind", *required],
        "additionalProperties": False,
    }


_PLANAR_CONSTRAINT_SCHEMA = {
    "oneOf": [
        _constraint_edit_schema(
            "coincident",
            {"first_point_id": _SKETCH_ID_SCHEMA, "second_point_id": _SKETCH_ID_SCHEMA},
            ["first_point_id", "second_point_id"],
        ),
        _constraint_edit_schema(
            "point_on_curve",
            {"point_id": _SKETCH_ID_SCHEMA, "curve_id": _SKETCH_ID_SCHEMA},
            ["point_id", "curve_id"],
        ),
        _constraint_edit_schema(
            "horizontal", {"line_id": _SKETCH_ID_SCHEMA}, ["line_id"]
        ),
        _constraint_edit_schema(
            "vertical", {"line_id": _SKETCH_ID_SCHEMA}, ["line_id"]
        ),
        _constraint_edit_schema(
            "parallel",
            {"first_line_id": _SKETCH_ID_SCHEMA, "second_line_id": _SKETCH_ID_SCHEMA},
            ["first_line_id", "second_line_id"],
        ),
        _constraint_edit_schema(
            "perpendicular",
            {"first_line_id": _SKETCH_ID_SCHEMA, "second_line_id": _SKETCH_ID_SCHEMA},
            ["first_line_id", "second_line_id"],
        ),
        _constraint_edit_schema(
            "equal_length",
            {"first_line_id": _SKETCH_ID_SCHEMA, "second_line_id": _SKETCH_ID_SCHEMA},
            ["first_line_id", "second_line_id"],
        ),
        _constraint_edit_schema(
            "tangent",
            {
                "first_curve_id": _SKETCH_ID_SCHEMA,
                "second_curve_id": _SKETCH_ID_SCHEMA,
                "branch_hint": {"type": "integer", "enum": [-1, 0, 1]},
            },
            ["first_curve_id", "second_curve_id"],
        ),
        _constraint_edit_schema(
            "equal_radius",
            {"first_curve_id": _SKETCH_ID_SCHEMA, "second_curve_id": _SKETCH_ID_SCHEMA},
            ["first_curve_id", "second_curve_id"],
        ),
        _constraint_edit_schema(
            "concentric",
            {"first_curve_id": _SKETCH_ID_SCHEMA, "second_curve_id": _SKETCH_ID_SCHEMA},
            ["first_curve_id", "second_curve_id"],
        ),
        _constraint_edit_schema("fixed", {"point_id": _SKETCH_ID_SCHEMA}, ["point_id"]),
        _constraint_edit_schema(
            "distance",
            {
                "first_point_id": _SKETCH_ID_SCHEMA,
                "second_point_id": _SKETCH_ID_SCHEMA,
                "value": {"type": "number", "exclusiveMinimum": 0},
                "driving": {"type": "boolean"},
            },
            ["first_point_id", "second_point_id", "value"],
        ),
        _constraint_edit_schema(
            "radius",
            {
                "curve_id": _SKETCH_ID_SCHEMA,
                "value": {"type": "number", "exclusiveMinimum": 0},
                "driving": {"type": "boolean"},
            },
            ["curve_id", "value"],
        ),
        _constraint_edit_schema(
            "angle",
            {
                "first_line_id": _SKETCH_ID_SCHEMA,
                "second_line_id": _SKETCH_ID_SCHEMA,
                "angle_degrees": {"type": "number"},
                "driving": {"type": "boolean"},
            },
            ["first_line_id", "second_line_id", "angle_degrees"],
        ),
    ]
}
_ADD_LINE_EDIT_SCHEMA = {
    "type": "object",
    "properties": {
        "operation": {"const": "add_line"},
        "start": _PLANAR_POINT_REF_SCHEMA,
        "end": _PLANAR_POINT_REF_SCHEMA,
    },
    "required": ["operation", "start", "end"],
    "additionalProperties": False,
}
_ADD_ARC_EDIT_SCHEMA = {
    "type": "object",
    "properties": {
        "operation": {"const": "add_arc"},
        "start": _PLANAR_POINT_REF_SCHEMA,
        "center": _PLANAR_POINT_REF_SCHEMA,
        "end": _PLANAR_POINT_REF_SCHEMA,
        "orientation": {"type": "string", "enum": ["cw", "ccw"]},
    },
    "required": ["operation", "start", "center", "end", "orientation"],
    "additionalProperties": False,
}
_UPDATE_LINE_EDIT_SCHEMA = {
    "type": "object",
    "properties": {
        "operation": {"const": "update_line"},
        "line_id": _SKETCH_ID_SCHEMA,
        "start": _PLANAR_POINT_REF_SCHEMA,
        "end": _PLANAR_POINT_REF_SCHEMA,
    },
    "required": ["operation", "line_id"],
    "minProperties": 3,
    "additionalProperties": False,
}
_UPDATE_ARC_EDIT_SCHEMA = {
    "type": "object",
    "properties": {
        "operation": {"const": "update_arc"},
        "arc_id": _SKETCH_ID_SCHEMA,
        "start": _PLANAR_POINT_REF_SCHEMA,
        "center": _PLANAR_POINT_REF_SCHEMA,
        "end": _PLANAR_POINT_REF_SCHEMA,
        "orientation": {"type": "string", "enum": ["cw", "ccw"]},
    },
    "required": ["operation", "arc_id"],
    "minProperties": 3,
    "additionalProperties": False,
}
_DELETE_CURVES_EDIT_SCHEMA = {
    "type": "object",
    "properties": {
        "operation": {"const": "delete_curves"},
        "curve_ids": {
            "type": "array",
            "minItems": 1,
            "maxItems": 32,
            "uniqueItems": True,
            "items": _SKETCH_ID_SCHEMA,
        },
    },
    "required": ["operation", "curve_ids"],
    "additionalProperties": False,
}
_ADD_CONSTRAINT_EDIT_SCHEMA = {
    "type": "object",
    "properties": {
        "operation": {"const": "add_constraint"},
        "constraint": _PLANAR_CONSTRAINT_SCHEMA,
    },
    "required": ["operation", "constraint"],
    "additionalProperties": False,
}
_REPLACE_CONSTRAINT_EDIT_SCHEMA = {
    "type": "object",
    "properties": {
        "operation": {"const": "replace_constraint"},
        "constraint_id": _SKETCH_ID_SCHEMA,
        "constraint": _PLANAR_CONSTRAINT_SCHEMA,
    },
    "required": ["operation", "constraint_id", "constraint"],
    "additionalProperties": False,
}
_DELETE_CONSTRAINTS_EDIT_SCHEMA = {
    "type": "object",
    "properties": {
        "operation": {"const": "delete_constraints"},
        "constraint_ids": {
            "type": "array",
            "minItems": 1,
            "maxItems": 32,
            "uniqueItems": True,
            "items": _SKETCH_ID_SCHEMA,
        },
    },
    "required": ["operation", "constraint_ids"],
    "additionalProperties": False,
}
_legacy_edit["oneOf"].extend(
    [
        _DELETE_CIRCLES_EDIT_SCHEMA,
        _REPLACE_CIRCLE_PATTERN_EDIT_SCHEMA,
        _ADD_LINE_EDIT_SCHEMA,
        _ADD_ARC_EDIT_SCHEMA,
        _UPDATE_LINE_EDIT_SCHEMA,
        _UPDATE_ARC_EDIT_SCHEMA,
        _DELETE_CURVES_EDIT_SCHEMA,
        _ADD_CONSTRAINT_EDIT_SCHEMA,
        _REPLACE_CONSTRAINT_EDIT_SCHEMA,
        _DELETE_CONSTRAINTS_EDIT_SCHEMA,
    ]
)
_BATCH_PLANAR_OPERATION_CONSTS = frozenset(
    {
        "add_circle",
        "add_rectangle",
        "add_polygon",
        "update_point",
        "update_circle",
        "delete_circles",
        "replace_circle_pattern",
        "add_line",
        "add_arc",
        "update_line",
        "update_arc",
        "delete_curves",
        "add_constraint",
        "replace_constraint",
        "delete_constraints",
    }
)
_batch_planar_branches = [
    branch
    for branch in _legacy_edit["oneOf"]
    if branch.get("properties", {}).get("operation", {}).get("const")
    in _BATCH_PLANAR_OPERATION_CONSTS
]
_legacy_edit["oneOf"].append(
    {
        "type": "object",
        "properties": {
            "operation": {"const": "batch"},
            "edits": {
                "type": "array",
                "minItems": 1,
                "maxItems": 16,
                "items": {"oneOf": _batch_planar_branches},
            },
        },
        "required": ["operation", "edits"],
        "additionalProperties": False,
    }
)
_SPATIAL_RELATION_SCHEMA = {
    "type": "object",
    "properties": {
        "reference_feature_id": {
            "type": "string",
            "minLength": 1,
            "maxLength": 128,
        },
        "relation": {
            "type": "string",
            "enum": ["above", "below", "left_of", "right_of"],
        },
        "clearance": {"type": "number", "minimum": 0},
        "tolerance": {"type": "number", "exclusiveMinimum": 0},
    },
    "required": ["reference_feature_id", "relation", "clearance"],
    "additionalProperties": False,
}
_legacy_geometry_edit_parameters["properties"] = {
    **dict(_legacy_geometry_edit_parameters["properties"]),
    "edit": _legacy_edit,
    "spatial_relation": _SPATIAL_RELATION_SCHEMA,
}
_PREPARE_GEOMETRY_EDIT = ToolDefinition(
    _PREPARE_GEOMETRY_EDIT.name,
    (
        "Prepare a revision-bound edit of an existing native Part. Profile "
        "transforms use the dedicated prepare_profile_* tools; this compatibility "
        "tool retains sketch, rigid, and exact Boolean edits. A planar cutout is "
        "one contained closed inner Profile, so it does not require Part Boolean. "
        "Use add_path_slot only when one ordered open, non-branching centerline "
        "and one width fully define the slot. Branching centerlines and arbitrary "
        "silhouettes require one complete closed add_polygon boundary or one "
        "complete line/arc batch. Invalid Profiles return exact topology diagnostics "
        "and must be revised before presenting a confirmation. When the user "
        "specifies an exact clearance from an existing planar Boolean feature, "
        "include optional spatial_relation with that feature_id, direction, and "
        "clearance; "
        "the local preflight rejects a mismatched proposal."
    ),
    _legacy_geometry_edit_parameters,
)


_READ_MESH_REFINEMENT_CONTEXT = _tool(
    "read_mesh_refinement_context",
    (
        "Read the bounded selectable logical entities of the active Part. "
        "Use their stable logical_id values for optional local mesh refinement."
    ),
    _NO_ARGUMENTS,
)
_PREPARE_MESH = _tool(
    "prepare_mesh_proposal",
    (
        "Build and present a mesh proposal. Gmsh is not called until the GUI "
        "control is clicked."
    ),
    {
        "type": "object",
        "properties": {
            "local_refinements": {
                "type": "array",
                "maxItems": 32,
                "items": {
                    "type": "object",
                    "properties": {
                        "target": {
                            "type": "string",
                            "pattern": "^(point|edge|face):",
                            "minLength": 3,
                            "maxLength": 256,
                        },
                        "size": {
                            "type": "number",
                            "exclusiveMinimum": 0,
                        },
                        "falloff": {
                            "type": "object",
                            "properties": {
                                "reference": {
                                    "type": "string",
                                    "enum": [
                                        "global_size",
                                        "target_radius",
                                    ],
                                },
                                "start_factor": {
                                    "type": "number",
                                    "minimum": 0,
                                },
                                "end_factor": {
                                    "type": "number",
                                    "exclusiveMinimum": 0,
                                },
                            },
                            "required": [
                                "reference",
                                "start_factor",
                                "end_factor",
                            ],
                            "additionalProperties": False,
                        },
                    },
                    "required": ["target", "size", "falloff"],
                    "additionalProperties": False,
                },
            },
        },
        "required": [],
        "additionalProperties": False,
    },
)
_READ_MODEL_TOPOLOGY_CONTEXT = _tool(
    "read_model_topology_context",
    (
        "Read a bounded catalog of current exact logical geometry entities "
        "and their materializable mesh reference counts. Use the returned "
        "part_id, logical_id, mesh_kind, and matched_count unchanged when "
        "calling create_named_region."
    ),
    _NO_ARGUMENTS,
)


def _exact_schema(
    properties: Mapping[str, object],
    *,
    required: tuple[str, ...] | None = None,
) -> dict[str, object]:
    fields = dict(properties)
    return {
        "type": "object",
        "properties": fields,
        "required": list(fields if required is None else required),
        "additionalProperties": False,
    }


def _controlled_name_schema(*prefixes: str) -> dict[str, object]:
    alternatives = "|".join(re.escape(prefix) for prefix in prefixes)
    return {
        "type": "string",
        "pattern": f"^({alternatives})-.+$",
        "minLength": 3,
        "maxLength": 96,
    }


def _definition_action_schema(
    action: str,
    parameters: Mapping[str, object],
) -> dict[str, object]:
    return _exact_schema(
        {
            "action": {"type": "string", "const": action},
            "parameters": dict(parameters),
        }
    )


def _one_of_object_schema(
    schemas: list[dict[str, object]],
) -> dict[str, object]:
    return {"type": "object", "oneOf": schemas}


_UNIT_SCHEMA = {"type": "string", "minLength": 1, "maxLength": 64}
_CONFIRMED_SCHEMA = {"type": "boolean", "const": True}
_STEP_NAME_SCHEMA = _controlled_name_schema("分析步")
_LOGICAL_IDS_SCHEMA = {
    "type": "array",
    "items": {
        "type": "string",
        "pattern": "^(point|edge|face|body):.+$",
        "minLength": 3,
        "maxLength": 256,
    },
    "minItems": 1,
    "maxItems": 128,
    "uniqueItems": True,
}


def _named_region_parameters(
    mesh_kind: str,
    name_prefix: str,
) -> dict[str, object]:
    return _exact_schema(
        {
            "name": _controlled_name_schema(name_prefix),
            "part_id": {
                "type": "string",
                "pattern": "^P[1-9][0-9]*$",
            },
            "logical_ids": _LOGICAL_IDS_SCHEMA,
            "mesh_kind": {"type": "string", "const": mesh_kind},
            "expected_count": {"type": "integer", "minimum": 1},
        }
    )


def _boundary_parameters(
    target_kind: str,
    scope_prefix: str,
) -> dict[str, object]:
    return _exact_schema(
        {
            "name": _controlled_name_schema("位移"),
            "step_name": _STEP_NAME_SCHEMA,
            "target_scope": _controlled_name_schema(scope_prefix),
            "target_kind": {"type": "string", "const": target_kind},
            "first_component": {
                "type": "integer",
                "minimum": 1,
                "maximum": 6,
            },
            "last_component": {
                "type": "integer",
                "minimum": 1,
                "maximum": 6,
            },
            "value": {"type": "number"},
            "unit": _UNIT_SCHEMA,
            "distribution": {"type": "string", "const": "uniform"},
            "confirmed": _CONFIRMED_SCHEMA,
        }
    )


def _load_parameters(
    *,
    entity_type: str,
    load_type: str,
    scope_prefix: str,
    component: Mapping[str, object],
    vector: Mapping[str, object],
    magnitude: Mapping[str, object],
    direction: str,
    distribution: str,
) -> dict[str, object]:
    return _exact_schema(
        {
            "name": _controlled_name_schema("载荷"),
            "step_name": _STEP_NAME_SCHEMA,
            "target_scope": _controlled_name_schema(scope_prefix),
            "entity_type": {"type": "string", "const": entity_type},
            "load_type": {"type": "string", "const": load_type},
            "component": dict(component),
            "vector": dict(vector),
            "magnitude": dict(magnitude),
            "direction": {"type": "string", "const": direction},
            "unit": _UNIT_SCHEMA,
            "distribution": {"type": "string", "const": distribution},
            "confirmed": _CONFIRMED_SCHEMA,
        }
    )


_NULL_SCHEMA = {"type": "null"}
_NUMBER_SCHEMA = {"type": "number"}
_LOAD_PARAMETER_SCHEMAS = [
    _load_parameters(
        entity_type="node",
        load_type="nodal",
        scope_prefix="点",
        component={"type": "integer", "const": component},
        vector=_NULL_SCHEMA,
        magnitude=_NUMBER_SCHEMA,
        direction=direction,
        distribution="concentrated",
    )
    for component, direction in (
        (1, "global_x"),
        (2, "global_y"),
        (3, "global_z"),
        (4, "global_rx"),
        (5, "global_ry"),
        (6, "global_rz"),
    )
] + [
    _load_parameters(
        entity_type="edge",
        load_type="edge_traction",
        scope_prefix="边",
        component=_NULL_SCHEMA,
        vector={
            "type": "array",
            "items": {"type": "number"},
            "minItems": 2,
            "maxItems": 2,
        },
        magnitude=_NULL_SCHEMA,
        direction="global_xy",
        distribution="uniform",
    ),
    _load_parameters(
        entity_type="edge",
        load_type="edge_pressure",
        scope_prefix="边",
        component=_NULL_SCHEMA,
        vector=_NULL_SCHEMA,
        magnitude=_NUMBER_SCHEMA,
        direction="inward_normal",
        distribution="uniform",
    ),
    _load_parameters(
        entity_type="edge",
        load_type="edge_pressure",
        scope_prefix="边",
        component=_NULL_SCHEMA,
        vector=_NULL_SCHEMA,
        magnitude=_NUMBER_SCHEMA,
        direction="outward_normal",
        distribution="uniform",
    ),
    _load_parameters(
        entity_type="surface",
        load_type="surface_traction",
        scope_prefix="面",
        component=_NULL_SCHEMA,
        vector={
            "type": "array",
            "items": {"type": "number"},
            "minItems": 3,
            "maxItems": 3,
        },
        magnitude=_NULL_SCHEMA,
        direction="global_xyz",
        distribution="uniform",
    ),
    _load_parameters(
        entity_type="surface",
        load_type="surface_pressure",
        scope_prefix="面",
        component=_NULL_SCHEMA,
        vector=_NULL_SCHEMA,
        magnitude=_NUMBER_SCHEMA,
        direction="inward_normal",
        distribution="uniform",
    ),
    _load_parameters(
        entity_type="surface",
        load_type="surface_pressure",
        scope_prefix="面",
        component=_NULL_SCHEMA,
        vector=_NULL_SCHEMA,
        magnitude=_NUMBER_SCHEMA,
        direction="outward_normal",
        distribution="uniform",
    ),
    _exact_schema(
        {
            "name": _controlled_name_schema("载荷"),
            "step_name": _STEP_NAME_SCHEMA,
            "target_scope": _controlled_name_schema("域"),
            "entity_type": {"type": "string", "const": "element"},
            "load_type": {"type": "string", "const": "line"},
            "vector": {
                "type": "array",
                "items": {"type": "number"},
                "minItems": 3,
                "maxItems": 3,
            },
            "coordinate_system": {
                "type": "string",
                "enum": ["global", "local"],
            },
            "unit": _UNIT_SCHEMA,
            "distribution": {"type": "string", "const": "uniform"},
            "confirmed": _CONFIRMED_SCHEMA,
        }
    ),
    _exact_schema(
        {
            "name": _controlled_name_schema("载荷"),
            "step_name": _STEP_NAME_SCHEMA,
            "target_scope": _controlled_name_schema("域"),
            "entity_type": {"type": "string", "const": "element"},
            "load_type": {"type": "string", "const": "body"},
            "vector": {
                "type": "array",
                "items": {"type": "number"},
                "minItems": 2,
                "maxItems": 3,
            },
            "direction": {"type": "string", "const": "global"},
            "unit": _UNIT_SCHEMA,
            "distribution": {"type": "string", "const": "uniform"},
            "confirmed": _CONFIRMED_SCHEMA,
        }
    ),
    _exact_schema(
        {
            "name": _controlled_name_schema("载荷"),
            "step_name": _STEP_NAME_SCHEMA,
            "target_scope": {
                "oneOf": [
                    {"type": "null"},
                    _controlled_name_schema("域"),
                ]
            },
            "entity_type": {"type": "string", "const": "element"},
            "load_type": {"type": "string", "const": "gravity"},
            "acceleration": {
                "type": "array",
                "items": {"type": "number"},
                "minItems": 2,
                "maxItems": 3,
            },
            "direction": {"type": "string", "const": "global"},
            "unit": _UNIT_SCHEMA,
            "distribution": {"type": "string", "const": "uniform"},
            "confirmed": _CONFIRMED_SCHEMA,
        }
    ),
]


_APPLY_DEFINITION = _tool(
    "apply_model_definition",
    (
        "Immediately apply one supported scope, material, section, assignment, "
        "analysis-step, boundary-condition, nodal/edge/surface/line/body/gravity "
        "load, or result-request action and "
        "synchronize the GUI. Only an edit that invalidates accepted results "
        "creates a confirmation card."
    ),
    {
        "type": "object",
        "oneOf": [
            _definition_action_schema(
                "create_named_region",
                _one_of_object_schema(
                    [
                        _named_region_parameters("node", "点"),
                        _named_region_parameters("edge", "边"),
                        _named_region_parameters("face", "面"),
                        _named_region_parameters("element", "域"),
                    ]
                ),
            ),
            _definition_action_schema(
                "create_material",
                _exact_schema(
                    {
                        "name": _controlled_name_schema("材料"),
                        "properties": _exact_schema(
                            {
                                "E": {
                                    "type": "number",
                                    "exclusiveMinimum": 0,
                                },
                                "nu": {
                                    "type": "number",
                                    "exclusiveMinimum": -1,
                                    "exclusiveMaximum": 0.5,
                                },
                                "density": {
                                    "type": "number",
                                    "exclusiveMinimum": 0,
                                },
                            },
                            required=("E", "nu"),
                        ),
                    }
                ),
            ),
            _definition_action_schema(
                "create_section",
                _one_of_object_schema(
                    [
                        _exact_schema(
                            {
                                "name": _controlled_name_schema("截面"),
                                "material": _controlled_name_schema("材料"),
                                "plane_type": {
                                    "type": "string",
                                    "enum": ["stress", "strain"],
                                },
                                "thickness": {
                                    "type": "number",
                                    "exclusiveMinimum": 0,
                                },
                                "properties": _exact_schema({}),
                            }
                        ),
                        _exact_schema(
                            {
                                "name": _controlled_name_schema("截面"),
                                "material": _controlled_name_schema("材料"),
                                "section_type": {
                                    "type": "string",
                                    "const": "solid",
                                },
                                "properties": _exact_schema({}),
                            }
                        ),
                        _exact_schema(
                            {
                                "name": _controlled_name_schema("截面"),
                                "material": _controlled_name_schema("材料"),
                                "section_type": {
                                    "type": "string",
                                    "const": "truss",
                                },
                                "properties": _exact_schema(
                                    {
                                        "area": {
                                            "type": "number",
                                            "exclusiveMinimum": 0,
                                        }
                                    }
                                ),
                            }
                        ),
                        _exact_schema(
                            {
                                "name": _controlled_name_schema("截面"),
                                "material": _controlled_name_schema("材料"),
                                "section_type": {
                                    "type": "string",
                                    "const": "rectangle",
                                },
                                "properties": _exact_schema(
                                    {
                                        "height": {
                                            "type": "number",
                                            "exclusiveMinimum": 0,
                                            "description": (
                                                "矩形截面沿梁局部 z（Abaqus n2）"
                                                "方向的高度"
                                            ),
                                        },
                                        "width": {
                                            "type": "number",
                                            "exclusiveMinimum": 0,
                                            "description": (
                                                "矩形截面沿梁局部 y（Abaqus n1）"
                                                "方向的宽度"
                                            ),
                                        },
                                    }
                                ),
                            }
                        ),
                        _exact_schema(
                            {
                                "name": _controlled_name_schema("截面"),
                                "material": _controlled_name_schema("材料"),
                                "section_type": {
                                    "type": "string",
                                    "const": "solid_circle",
                                },
                                "properties": _exact_schema(
                                    {
                                        "radius": {
                                            "type": "number",
                                            "exclusiveMinimum": 0,
                                        }
                                    }
                                ),
                            }
                        ),
                        _exact_schema(
                            {
                                "name": _controlled_name_schema("截面"),
                                "material": _controlled_name_schema("材料"),
                                "section_type": {
                                    "type": "string",
                                    "const": "hollow_circle",
                                },
                                "properties": _exact_schema(
                                    {
                                        "outer_radius": {
                                            "type": "number",
                                            "exclusiveMinimum": 0,
                                        },
                                        "inner_radius": {
                                            "type": "number",
                                            "exclusiveMinimum": 0,
                                        },
                                    }
                                ),
                            }
                        ),
                    ]
                ),
            ),
            _definition_action_schema(
                "assign_section",
                _one_of_object_schema(
                    [
                        _exact_schema(
                            {
                                "section_name": _controlled_name_schema("截面"),
                                "region_name": _controlled_name_schema("域"),
                            }
                        ),
                        _exact_schema(
                            {
                                "section_name": _controlled_name_schema("截面"),
                                "region_name": _controlled_name_schema("域"),
                                "local_y_reference": {
                                    "type": "array",
                                    "items": {"type": "number"},
                                    "minItems": 3,
                                    "maxItems": 3,
                                },
                            }
                        ),
                    ]
                ),
            ),
            _definition_action_schema(
                "create_static_step",
                _exact_schema({"name": _STEP_NAME_SCHEMA}),
            ),
            _definition_action_schema(
                "create_boundary_condition",
                _one_of_object_schema(
                    [
                        _boundary_parameters("node_set", "点"),
                        _boundary_parameters("edge", "边"),
                        _boundary_parameters("surface", "面"),
                    ]
                ),
            ),
            _definition_action_schema(
                "create_load",
                _one_of_object_schema(_LOAD_PARAMETER_SCHEMAS),
            ),
            _definition_action_schema(
                "create_result_request",
                _one_of_object_schema(
                    [
                        _exact_schema(
                            {
                                "name": _controlled_name_schema("结果请求"),
                                "step_name": _STEP_NAME_SCHEMA,
                                "target": {
                                    "type": "string",
                                    "const": "node",
                                },
                                "variables": {
                                    "type": "array",
                                    "items": {
                                        "type": "string",
                                        "enum": ["U", "UR", "RF", "RM"],
                                    },
                                    "minItems": 1,
                                    "maxItems": 4,
                                    "uniqueItems": True,
                                },
                                "units": {
                                    "type": "array",
                                    "items": _UNIT_SCHEMA,
                                    "minItems": 1,
                                    "maxItems": 4,
                                },
                                "confirmed": _CONFIRMED_SCHEMA,
                            }
                        ),
                        _exact_schema(
                            {
                                "name": _controlled_name_schema("结果请求"),
                                "step_name": _STEP_NAME_SCHEMA,
                                "target": {
                                    "type": "string",
                                    "const": "element",
                                },
                                "variables": {
                                    "type": "array",
                                    "items": {
                                        "type": "string",
                                        "enum": ["SF", "SM", "LE", "S"],
                                    },
                                    "minItems": 1,
                                    "maxItems": 4,
                                    "uniqueItems": True,
                                },
                                "units": {
                                    "type": "array",
                                    "items": _UNIT_SCHEMA,
                                    "minItems": 1,
                                    "maxItems": 4,
                                },
                                "confirmed": _CONFIRMED_SCHEMA,
                            }
                        ),
                    ]
                ),
            ),
        ],
    },
)
_RUN_PREFLIGHT = _tool(
    "run_native_preflight",
    "Run the existing deterministic native preflight without solving.",
    {
        "type": "object",
        "properties": {
            "step_name": {
                "type": "string",
                "minLength": 1,
                "maxLength": 160,
            }
        },
        "additionalProperties": False,
    },
)
_PREPARE_SOLVE = _tool(
    "prepare_solve_proposal",
    (
        "Build and present a revision/stamp-bound solve proposal. The solver "
        "is not started until the GUI control is clicked."
    ),
    {
        "type": "object",
        "properties": {
            "step_name": {
                "type": "string",
                "minLength": 1,
                "maxLength": 160,
            }
        },
        "additionalProperties": False,
    },
)
_REQUEST_PROJECT_SAVE = _tool(
    "request_project_save",
    (
        "Present a revision-bound local project-save card. This tool has no "
        "path argument and cannot write a file; only its GUI button may start "
        "the existing native project save command."
    ),
    _NO_ARGUMENTS,
)

_PLANAR_NODE_ID_SCHEMA = {
    "type": "string",
    "minLength": 1,
    "maxLength": MAX_NODE_ID_LENGTH,
    "pattern": "^[A-Za-z][A-Za-z0-9_.-]*$",
}
_PLANAR_POINT_SCHEMA = {
    "type": "array",
    "items": {"type": "number"},
    "minItems": 2,
    "maxItems": 2,
}


def _planar_node_schema(
    kind: str,
    properties: Mapping[str, object],
    required: Sequence[str],
) -> dict[str, object]:
    return {
        "type": "object",
        "properties": {
            "id": _PLANAR_NODE_ID_SCHEMA,
            "kind": {"const": kind},
            **dict(properties),
        },
        "required": ["id", "kind", *required],
        "additionalProperties": False,
    }


def _planar_construction_context_capability() -> dict[str, object]:
    return {
        "schema_version": PLANAR_CONSTRUCTION_SCHEMA_VERSION,
        "plane": "XY",
        "coordinate_conventions": {
            "rectangle_anchor": "lower_left",
            "rectangle_extent": "x..x+width, y..y+height",
            "circle_position": "center_x, center_y",
            "circle_size": "radius; use diameter/2 when the request gives a diameter",
            "pattern_seed": "included_as_instance_zero",
        },
        "node_kinds": {
            "primitive": ["rectangle", "circle", "polygon", "path_stroke"],
            "boolean": ["union", "difference", "intersection"],
            "transform": ["translate", "rotate", "mirror"],
            "pattern": [
                "linear_pattern",
                "rectangular_pattern",
                "circular_pattern",
            ],
        },
        "budgets": {
            "max_node_count": MAX_NODES,
            "max_boolean_operands": MAX_BOOLEAN_OPERANDS,
            "max_polygon_vertices": MAX_POLYGON_VERTICES,
            "max_path_points": MAX_PATH_POINTS,
            "max_pattern_instances": MAX_PATTERN_INSTANCES,
            "max_dag_depth": MAX_DAG_DEPTH,
            "max_canonical_payload_bytes": MAX_CANONICAL_PAYLOAD_BYTES,
        },
        "output_kinds": ["planar", "extrusion", "revolution", "path_sweep"],
    }


_PLANAR_CONSTRUCTION_NODE_SCHEMAS = (
    _planar_node_schema(
        "rectangle",
        {
            "x": {
                "type": "number",
                "description": (
                    "X coordinate of the lower-left corner, not the center. "
                    "The rectangle spans x through x + width."
                ),
            },
            "y": {
                "type": "number",
                "description": (
                    "Y coordinate of the lower-left corner, not the center. "
                    "The rectangle spans y through y + height."
                ),
            },
            "width": {
                "type": "number",
                "exclusiveMinimum": 0,
                "description": (
                    "Positive X extent. To center width w at cx, set x = cx - w/2."
                ),
            },
            "height": {
                "type": "number",
                "exclusiveMinimum": 0,
                "description": (
                    "Positive Y extent. To center height h at cy, set y = cy - h/2."
                ),
            },
        },
        ("x", "y", "width", "height"),
    ),
    _planar_node_schema(
        "circle",
        {
            "center_x": {
                "type": "number",
                "description": "X coordinate of the circle center.",
            },
            "center_y": {
                "type": "number",
                "description": "Y coordinate of the circle center.",
            },
            "radius": {
                "type": "number",
                "exclusiveMinimum": 0,
                "description": (
                    "Circle radius, never diameter. If the request gives hole "
                    "diameter d (including Chinese 孔径/直径), use radius = d/2."
                ),
            },
        },
        ("center_x", "center_y", "radius"),
    ),
    _planar_node_schema(
        "polygon",
        {
            "vertices": {
                "type": "array",
                "minItems": 3,
                "maxItems": MAX_POLYGON_VERTICES,
                "items": _PLANAR_POINT_SCHEMA,
            }
        },
        ("vertices",),
    ),
    _planar_node_schema(
        "path_stroke",
        {
            "points": {
                "type": "array",
                "minItems": 2,
                "maxItems": MAX_PATH_POINTS,
                "items": _PLANAR_POINT_SCHEMA,
                "description": (
                    "One ordered, open, non-branching centerline. A slot with "
                    "a junction cannot be represented by one path_stroke; use "
                    "overlapping primitives or strokes and a union node."
                ),
            },
            "width": {"type": "number", "exclusiveMinimum": 0},
            "cap": {"type": "string", "enum": ["butt", "square", "round"]},
            "join": {"type": "string", "enum": ["miter", "bevel", "round"]},
        },
        ("points", "width", "cap", "join"),
    ),
    _planar_node_schema(
        "union",
        {
            "operands": {
                "type": "array",
                "minItems": 1,
                "maxItems": MAX_BOOLEAN_OPERANDS,
                "uniqueItems": True,
                "items": _PLANAR_NODE_ID_SCHEMA,
            }
        },
        ("operands",),
    ),
    _planar_node_schema(
        "difference",
        {
            "base": _PLANAR_NODE_ID_SCHEMA,
            "subtract": {
                "type": "array",
                "minItems": 1,
                "maxItems": MAX_BOOLEAN_OPERANDS,
                "uniqueItems": True,
                "items": _PLANAR_NODE_ID_SCHEMA,
            },
        },
        ("base", "subtract"),
    ),
    _planar_node_schema(
        "intersection",
        {
            "operands": {
                "type": "array",
                "minItems": 2,
                "maxItems": MAX_BOOLEAN_OPERANDS,
                "uniqueItems": True,
                "items": _PLANAR_NODE_ID_SCHEMA,
            }
        },
        ("operands",),
    ),
    _planar_node_schema(
        "translate",
        {
            "source": _PLANAR_NODE_ID_SCHEMA,
            "dx": {"type": "number"},
            "dy": {"type": "number"},
        },
        ("source", "dx", "dy"),
    ),
    _planar_node_schema(
        "rotate",
        {
            "source": _PLANAR_NODE_ID_SCHEMA,
            "center_x": {"type": "number"},
            "center_y": {"type": "number"},
            "angle_degrees": {"type": "number"},
        },
        ("source", "center_x", "center_y", "angle_degrees"),
    ),
    _planar_node_schema(
        "mirror",
        {
            "source": _PLANAR_NODE_ID_SCHEMA,
            "line_point_x": {"type": "number"},
            "line_point_y": {"type": "number"},
            "line_direction_x": {"type": "number"},
            "line_direction_y": {"type": "number"},
        },
        (
            "source",
            "line_point_x",
            "line_point_y",
            "line_direction_x",
            "line_direction_y",
        ),
    ),
    _planar_node_schema(
        "linear_pattern",
        {
            "seed": _PLANAR_NODE_ID_SCHEMA,
            "count": {"type": "integer", "minimum": 1},
            "step_x": {"type": "number"},
            "step_y": {"type": "number"},
        },
        ("seed", "count", "step_x", "step_y"),
    ),
    _planar_node_schema(
        "rectangular_pattern",
        {
            "seed": _PLANAR_NODE_ID_SCHEMA,
            "count_x": {"type": "integer", "minimum": 1},
            "count_y": {"type": "integer", "minimum": 1},
            "spacing_x": {"type": "number"},
            "spacing_y": {"type": "number"},
        },
        ("seed", "count_x", "count_y", "spacing_x", "spacing_y"),
    ),
    _planar_node_schema(
        "circular_pattern",
        {
            "seed": _PLANAR_NODE_ID_SCHEMA,
            "count": {"type": "integer", "minimum": 1},
            "center_x": {"type": "number"},
            "center_y": {"type": "number"},
            "total_angle_degrees": {"type": "number"},
        },
        ("seed", "count", "center_x", "center_y", "total_angle_degrees"),
    ),
)

_PLANAR_CONSTRUCTION_PROFILE_SELECTION_SCHEMA = {
    "oneOf": [
        {"const": "unique_material_profile"},
        {
            "type": "array",
            "minItems": 1,
            "maxItems": 32,
            "uniqueItems": True,
            "items": {
                "type": "string",
                "pattern": r"^face:[^\s]+$",
                "minLength": 6,
                "maxLength": 192,
            },
        },
        {
            "type": "object",
            "properties": {
                "source_face_ids": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 32,
                    "uniqueItems": True,
                    "items": {
                        "type": "string",
                        "pattern": r"^face:[^\s]+$",
                        "minLength": 6,
                        "maxLength": 192,
                    },
                },
            },
            "required": ["source_face_ids"],
            "additionalProperties": False,
        },
    ]
}
_PLANAR_CONSTRUCTION_OUTPUT_SCHEMA = {
    "oneOf": [
        {"const": "planar"},
        {
            "type": "object",
            "properties": {
                "kind": {"const": "planar"},
            },
            "required": ["kind"],
            "additionalProperties": False,
        },
        {
            "type": "object",
            "properties": {
                "kind": {"const": "extrusion"},
                "profile_selection": _PLANAR_CONSTRUCTION_PROFILE_SELECTION_SCHEMA,
                "height": {"type": "number", "exclusiveMinimum": 0},
            },
            "required": ["kind", "profile_selection", "height"],
            "additionalProperties": False,
        },
        {
            "type": "object",
            "properties": {
                "kind": {"const": "revolution"},
                "profile_selection": _PLANAR_CONSTRUCTION_PROFILE_SELECTION_SCHEMA,
                "axis": {"type": "string", "enum": ["x", "y", "z"]},
                "angle_degrees": {
                    "type": "number",
                    "exclusiveMinimum": 0,
                    "maximum": 360,
                },
            },
            "required": [
                "kind",
                "profile_selection",
                "axis",
                "angle_degrees",
            ],
            "additionalProperties": False,
        },
        {
            "type": "object",
            "properties": {
                "kind": {"const": "path_sweep"},
                "profile_selection": _PLANAR_CONSTRUCTION_PROFILE_SELECTION_SCHEMA,
                "path": _COMPOSITE_PATH_SCHEMA,
                "frame_strategy": {
                    "type": "string",
                    "enum": ["fixed", "transport"],
                },
            },
            "required": [
                "kind",
                "profile_selection",
                "path",
                "frame_strategy",
            ],
            "additionalProperties": False,
        },
    ]
}

_PREPARE_PLANAR_CONSTRUCTION = _tool(
    "prepare_planar_construction_proposal",
    (
        "Compile one bounded declarative Planar Construction IR v1 graph on "
        "the global XY plane, prove its exact Profiles and materialized strict "
        "sketch locally, then present one revision-bound planar or derived 3D "
        "Part proposal. "
        "Use general primitives, Boolean operations, transforms, and patterns; "
        "rectangle x/y are the lower-left corner, circle radius is not diameter, "
        "path_stroke is one non-branching open centerline and branching slots "
        "must unite multiple overlapping connected primitives or strokes, "
        "semantically distinct slots and hole patterns use separate subtraction "
        "operands so they remain separate native Cut features, "
        "and 2D output accepts either the literal string 'planar' or the "
        "equivalent object {'kind': 'planar'}; both normalize to planar. "
        "do not calculate final Boolean boundary vertices or submit CAD code. "
        "The model changes only after the local GUI control is clicked."
    ),
    {
        "type": "object",
        "properties": {
            "part_function": {
                "type": "string",
                "minLength": 1,
                "maxLength": 96,
            },
            "construction": {
                "type": "object",
                "properties": {
                    "schema_version": {"const": PLANAR_CONSTRUCTION_SCHEMA_VERSION},
                    "name": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": MAX_NAME_LENGTH,
                    },
                    "plane": {"const": "XY"},
                    "nodes": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": MAX_NODES,
                        "items": {"oneOf": list(_PLANAR_CONSTRUCTION_NODE_SCHEMAS)},
                    },
                    "result_node_id": _PLANAR_NODE_ID_SCHEMA,
                },
                "required": [
                    "schema_version",
                    "name",
                    "plane",
                    "nodes",
                    "result_node_id",
                ],
                "additionalProperties": False,
            },
            "output": _PLANAR_CONSTRUCTION_OUTPUT_SCHEMA,
        },
        "required": ["part_function", "construction", "output"],
        "additionalProperties": False,
    },
)
_CREATE_NATIVE_MODEL_DOCUMENT = _tool(
    "create_native_model_document",
    (
        "Create and activate an additional blank native model document while "
        "preserving every existing workspace document. Use this only when the "
        "user asks for another or separate model document, not for a new Part "
        "inside the current model. After success, read the new authoring context "
        "and prepare the requested geometry in the same turn."
    ),
    {
        "type": "object",
        "properties": {
            "model_name": {
                "type": "string",
                "minLength": 1,
                "maxLength": 160,
            }
        },
        "additionalProperties": False,
    },
)
_READ_DELETABLE_OBJECTS = _tool(
    "read_deletable_objects",
    (
        "Read a bounded catalog of current native model objects that can be "
        "selected for a GUI-confirmed deletion."
    ),
    _NO_ARGUMENTS,
)
_PREPARE_DELETE = _tool(
    "prepare_delete_proposal",
    (
        "Build and present one revision-bound destructive-edit card for an "
        "exact object returned by read_deletable_objects. This tool cannot "
        "perform the deletion; only its GUI button may do so."
    ),
    {
        "type": "object",
        "properties": {
            "object_type": {
                "type": "string",
                "enum": [
                    "part",
                    "feature",
                    "generated_mesh",
                    "named_region",
                    "analysis_step",
                    "boundary_condition",
                    "load",
                    "result_request",
                ],
            },
            "target_id": {
                "type": "string",
                "minLength": 1,
                "maxLength": 160,
            },
            "step_name": {
                "type": "string",
                "minLength": 1,
                "maxLength": 160,
            },
        },
        "required": ["object_type", "target_id"],
        "additionalProperties": False,
    },
)
_READ_EDITABLE_OBJECTS = _tool(
    "read_editable_model_objects",
    (
        "Read current native scopes, materials, sections, assignments, "
        "analysis steps, boundary conditions, loads, and result requests "
        "with bounded editable fields and stable identities."
    ),
    _NO_ARGUMENTS,
)
_EDIT_FLAT_VALUE_SCHEMA = {
    "anyOf": [
        {"type": "number"},
        {"type": "string", "minLength": 1, "maxLength": 160},
        {"type": "boolean"},
        {"type": "null"},
        {
            "type": "array",
            "minItems": 1,
            "maxItems": 16,
            "items": {
                "anyOf": [
                    {"type": "number"},
                    {"type": "string", "minLength": 1, "maxLength": 160},
                    {"type": "boolean"},
                    {"type": "null"},
                ]
            },
        },
    ]
}
_EDIT_BOUNDED_MAPPING_SCHEMA = {
    "type": "object",
    "description": "Partial key updates; omitted keys are retained.",
    "propertyNames": {"type": "string", "minLength": 1, "maxLength": 96},
    "maxProperties": 32,
    "additionalProperties": _EDIT_FLAT_VALUE_SCHEMA,
}
_EDIT_MODEL_OBJECT = _tool(
    "edit_model_object",
    (
        "Immediately edit one exact object returned by "
        "read_editable_model_objects and synchronize the GUI. Definition "
        "edits retain completed run/result history, reset the current "
        "preflight and result display, and require a new preflight."
    ),
    {
        "type": "object",
        "properties": {
            "object_type": {
                "type": "string",
                "enum": [
                    "named_region",
                    "material",
                    "section",
                    "section_assignment",
                    "analysis_step",
                    "boundary_condition",
                    "load",
                    "result_request",
                ],
            },
            "target_id": {
                "type": "string",
                "minLength": 1,
                "maxLength": 160,
            },
            "step_name": {
                "type": "string",
                "minLength": 1,
                "maxLength": 160,
            },
            "changes": {
                "type": "object",
                "properties": {
                    "new_name": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 160,
                    },
                    "part_id": {
                        "type": "string",
                        "pattern": "^P[1-9][0-9]*$",
                    },
                    "logical_ids": _LOGICAL_IDS_SCHEMA,
                    "mesh_kind": {
                        "type": "string",
                        "enum": ["node", "edge", "face", "element"],
                    },
                    "expected_count": {
                        "type": "integer",
                        "minimum": 1,
                    },
                    "reference_keys": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "minLength": 1,
                            "maxLength": 96,
                        },
                        "minItems": 1,
                        "maxItems": 128,
                        "uniqueItems": True,
                    },
                    "target_scope": {
                        "anyOf": [
                            {
                                "type": "string",
                                "minLength": 1,
                                "maxLength": 96,
                            },
                            {"type": "null"},
                        ]
                    },
                    "material": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 160,
                    },
                    "section_name": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 160,
                    },
                    "region_name": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 160,
                    },
                    "section_type": {
                        "type": "string",
                        "enum": [
                            "solid",
                            "truss",
                            "rectangle",
                            "solid_circle",
                            "hollow_circle",
                        ],
                    },
                    "properties": _EDIT_BOUNDED_MAPPING_SCHEMA,
                    "procedure": {"type": "string", "enum": ["static"]},
                    "metadata": _EDIT_BOUNDED_MAPPING_SCHEMA,
                    "output_kind": {"type": "string", "enum": ["field"]},
                    "kind": {"type": "string", "enum": ["field"]},
                    "target": {
                        "type": "string",
                        "enum": ["node", "element"],
                    },
                    "variables": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": ["U", "UR", "RF", "RM", "SF", "SM", "LE", "S"],
                        },
                        "minItems": 1,
                        "maxItems": 16,
                        "uniqueItems": True,
                    },
                    "units": {
                        "type": "array",
                        "items": _UNIT_SCHEMA,
                        "minItems": 1,
                        "maxItems": 16,
                    },
                    "first_component": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 3,
                    },
                    "last_component": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 3,
                    },
                    "component": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 3,
                    },
                    "value": {"type": "number"},
                    "vector": {
                        "type": ["array", "null"],
                        "items": {"type": "number"},
                        "minItems": 1,
                        "maxItems": 3,
                    },
                    "magnitude": {
                        "anyOf": [
                            {"type": "number"},
                            {"type": "null"},
                        ]
                    },
                    "load_type": {
                        "type": "string",
                        "enum": [
                            "nodal",
                            "edge_traction",
                            "edge_pressure",
                            "surface_traction",
                            "surface_pressure",
                        ],
                    },
                    "entity_type": {
                        "type": "string",
                        "enum": ["node", "edge", "surface"],
                    },
                    "direction": {
                        "type": "string",
                        "enum": [
                            "global_x",
                            "global_y",
                            "global_z",
                            "global_xy",
                            "global_xyz",
                            "global",
                            "inward_normal",
                            "outward_normal",
                        ],
                    },
                    "coordinate_system": {
                        "type": "string",
                        "enum": ["global", "local"],
                    },
                    "acceleration": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 1,
                        "maxItems": 3,
                    },
                    "unit": _UNIT_SCHEMA,
                    "distribution": {
                        "type": "string",
                        "enum": ["uniform", "concentrated"],
                    },
                    "confirmed": _CONFIRMED_SCHEMA,
                },
                "minProperties": 1,
                "additionalProperties": False,
            },
        },
        "required": ["object_type", "target_id", "changes"],
        "additionalProperties": False,
    },
)
_RESULT_CATALOG = _result_tool_definition(result_catalog_tool_schema())
_RESULT_QUERY = _result_tool_definition(result_query_tool_schema())
_RESULT_COMPARISON = _result_tool_definition(result_comparison_tool_schema())
_ANALYSIS_RUN_CATALOG = _result_tool_definition(analysis_run_catalog_tool_schema())
_WORKSPACE_DOCUMENTS = _result_tool_definition(workspace_documents_tool_schema())
_GEOMETRY_FEATURE_CATALOG = _result_tool_definition(
    geometry_feature_catalog_tool_schema()
)


_PROFILE_FACE_ID_SCHEMA = {
    "type": "string",
    "pattern": r"^face:[^\s]+$",
    "minLength": 6,
    "maxLength": 192,
}
_PROFILE_UNIQUE_SELECTION_SCHEMA = {
    "oneOf": [
        {
            "type": "string",
            "const": "unique_material_profile",
        },
        {
            "type": "object",
            "properties": {
                "mode": {"const": "unique_material_profile"},
            },
            "required": ["mode"],
            "additionalProperties": False,
        },
        {
            "type": "object",
            "properties": {
                "kind": {"const": "unique_material_profile"},
            },
            "required": ["kind"],
            "additionalProperties": False,
        },
    ]
}
_PROFILE_EXPLICIT_SELECTION_SCHEMA = {
    "oneOf": [
        {
            "type": "array",
            "minItems": 1,
            "maxItems": 32,
            "uniqueItems": True,
            "items": _PROFILE_FACE_ID_SCHEMA,
            "description": (
                "Explicit canonical material Profile IDs returned by "
                "read_profile_transform_context."
            ),
        },
        {
            "type": "object",
            "properties": {
                "source_face_ids": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 32,
                    "uniqueItems": True,
                    "items": _PROFILE_FACE_ID_SCHEMA,
                },
            },
            "required": ["source_face_ids"],
            "additionalProperties": False,
        },
    ]
}
_PROFILE_SELECTION_SCHEMA = {
    "oneOf": [
        _PROFILE_UNIQUE_SELECTION_SCHEMA["oneOf"][0],
        _PROFILE_EXPLICIT_SELECTION_SCHEMA["oneOf"][0],
        {
            "type": "object",
            "oneOf": [
                *_PROFILE_UNIQUE_SELECTION_SCHEMA["oneOf"][1:],
                _PROFILE_EXPLICIT_SELECTION_SCHEMA["oneOf"][1],
            ],
        },
    ]
}


_READ_PROFILE_TRANSFORM_CONTEXT = _tool(
    "read_profile_transform_context",
    (
        "Read the bounded canonical material Profile and transform capability "
        "context for one existing native Part. This read is required before a "
        "dedicated Profile extrusion, revolution, or path-sweep proposal."
    ),
    {
        "type": "object",
        "properties": {
            "part_id": {
                "type": "string",
                "minLength": 1,
                "maxLength": 128,
            },
        },
        "required": ["part_id"],
        "additionalProperties": False,
    },
)


def _profile_transform_prepare_schema(
    *,
    operation: str,
    extra_properties: Mapping[str, object],
    required_extra: tuple[str, ...],
    description: str,
) -> ToolDefinition:
    properties = {
        "part_id": {
            "type": "string",
            "minLength": 1,
            "maxLength": 128,
        },
        "context_revision": {
            "type": "integer",
            "minimum": 0,
            "description": (
                "Revision returned by read_profile_transform_context. It is "
                "required for explicit source_face_ids; unique_material_profile "
                "may omit it. A mismatch is rejected as stale."
            ),
        },
        "profile_selection": _PROFILE_SELECTION_SCHEMA,
        **dict(extra_properties),
    }
    return _tool(
        operation,
        description,
        {
            "type": "object",
            "properties": properties,
            "required": ["part_id", "profile_selection", *required_extra],
            "oneOf": [
                {
                    "properties": {
                        "profile_selection": _PROFILE_UNIQUE_SELECTION_SCHEMA,
                    },
                },
                {
                    "properties": {
                        "profile_selection": _PROFILE_EXPLICIT_SELECTION_SCHEMA,
                    },
                    "required": ["context_revision"],
                },
            ],
            "additionalProperties": False,
        },
    )


_PREPARE_PROFILE_EXTRUSION = _profile_transform_prepare_schema(
    operation="prepare_profile_extrusion",
    extra_properties={
        "height": {
            "type": "number",
            "exclusiveMinimum": 0,
        },
    },
    required_extra=("height",),
    description=(
        "Prepare a revision-bound positive extrusion proposal from a canonical "
        "material Profile. unique_material_profile is accepted only when the "
        "local read proves exactly one material Profile; explicit IDs may select "
        "multiple independent Profiles."
    ),
)
_PREPARE_PROFILE_REVOLUTION = _profile_transform_prepare_schema(
    operation="prepare_profile_revolution",
    extra_properties={
        "axis": {"type": "string", "enum": ["x", "y", "z"]},
        "angle_degrees": {
            "type": "number",
            "exclusiveMinimum": 0,
            "maximum": 360,
        },
    },
    required_extra=("axis", "angle_degrees"),
    description=(
        "Prepare a revision-bound revolution proposal from exactly one canonical "
        "material Profile."
    ),
)
_PREPARE_PROFILE_PATH_SWEEP = _profile_transform_prepare_schema(
    operation="prepare_profile_path_sweep",
    extra_properties={
        "path": {
            "type": "object",
            "properties": {
                "points": {
                    "type": "array",
                    "minItems": 2,
                    "maxItems": 64,
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string",
                                "minLength": 1,
                                "maxLength": 64,
                            },
                            "x": {"type": "number"},
                            "y": {"type": "number"},
                            "z": {"type": "number"},
                        },
                        "required": ["name", "x", "y", "z"],
                        "additionalProperties": False,
                    },
                },
                "members": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 63,
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string",
                                "minLength": 1,
                                "maxLength": 64,
                            },
                            "start": {
                                "type": "string",
                                "minLength": 1,
                                "maxLength": 64,
                            },
                            "end": {
                                "type": "string",
                                "minLength": 1,
                                "maxLength": 64,
                            },
                        },
                        "required": ["name", "start", "end"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["points", "members"],
            "additionalProperties": False,
        },
        "frame_strategy": {
            "type": "string",
            "enum": ["fixed", "transport"],
        },
    },
    required_extra=("path", "frame_strategy"),
    description=(
        "Prepare a revision-bound path-sweep proposal from exactly one canonical "
        "material Profile and an explicitly ordered open polyline."
    ),
)


_PROJECT_SAVE_READY_STAGES = frozenset(
    {
        AuthoringWorkflowStage.REQUIREMENTS,
        AuthoringWorkflowStage.GEOMETRY_READY,
        AuthoringWorkflowStage.MESH_READY,
        AuthoringWorkflowStage.DEFINITIONS_READY,
        AuthoringWorkflowStage.ANALYSIS_DEFINITIONS_READY,
        AuthoringWorkflowStage.PREFLIGHT_READY,
        AuthoringWorkflowStage.SOLVE_READY,
        AuthoringWorkflowStage.RESULTS_READY,
    }
)
_DESTRUCTIVE_EDIT_READY_STAGES = _PROJECT_SAVE_READY_STAGES
_DESTRUCTIVE_EDIT_TOOLS = frozenset(
    {_READ_DELETABLE_OBJECTS.name, _PREPARE_DELETE.name}
)
_MODEL_EDIT_TOOLS = frozenset({_READ_EDITABLE_OBJECTS.name, _EDIT_MODEL_OBJECT.name})
_GEOMETRY_EDIT_TOOLS = frozenset(
    {
        _READ_GEOMETRY_EDIT_CONTEXT.name,
        _PREPARE_GEOMETRY_EDIT.name,
    }
)
_GEOMETRY_CATALOG_TOOLS = frozenset({_GEOMETRY_FEATURE_CATALOG.name})
_PROFILE_TRANSFORM_TOOLS = frozenset(
    {
        _READ_PROFILE_TRANSFORM_CONTEXT.name,
        _PREPARE_PROFILE_EXTRUSION.name,
        _PREPARE_PROFILE_REVOLUTION.name,
        _PREPARE_PROFILE_PATH_SWEEP.name,
    }
)


_STAGE_TOOLS: dict[AuthoringWorkflowStage, tuple[ToolDefinition, ...]] = {
    AuthoringWorkflowStage.REQUIREMENTS: (
        _READ_CONTEXT,
        _GEOMETRY_FEATURE_CATALOG,
        _SET_REQUIREMENTS,
        _PREPARE_GEOMETRY,
        _PREPARE_PLANAR_CONSTRUCTION,
        _READ_GEOMETRY_EDIT_CONTEXT,
        _PREPARE_GEOMETRY_EDIT,
        _APPLY_DEFINITION,
        _REQUEST_PROJECT_SAVE,
        _READ_DELETABLE_OBJECTS,
        _PREPARE_DELETE,
        _READ_EDITABLE_OBJECTS,
        _EDIT_MODEL_OBJECT,
    ),
    AuthoringWorkflowStage.REVIEW_PENDING: (_READ_CONTEXT,),
    AuthoringWorkflowStage.GEOMETRY_READY: (
        _READ_CONTEXT,
        _GEOMETRY_FEATURE_CATALOG,
        _SET_REQUIREMENTS,
        _PREPARE_GEOMETRY,
        _PREPARE_PLANAR_CONSTRUCTION,
        _READ_GEOMETRY_EDIT_CONTEXT,
        _PREPARE_GEOMETRY_EDIT,
        _APPLY_DEFINITION,
        _REQUEST_PROJECT_SAVE,
        _READ_DELETABLE_OBJECTS,
        _PREPARE_DELETE,
        _READ_EDITABLE_OBJECTS,
        _EDIT_MODEL_OBJECT,
    ),
    AuthoringWorkflowStage.GEOMETRY_PENDING: (_READ_CONTEXT,),
    AuthoringWorkflowStage.MESH_READY: (
        _READ_CONTEXT,
        _GEOMETRY_FEATURE_CATALOG,
        _SET_REQUIREMENTS,
        _READ_MESH_REFINEMENT_CONTEXT,
        _PREPARE_MESH,
        _READ_GEOMETRY_EDIT_CONTEXT,
        _PREPARE_GEOMETRY_EDIT,
        _APPLY_DEFINITION,
        _REQUEST_PROJECT_SAVE,
        _READ_DELETABLE_OBJECTS,
        _PREPARE_DELETE,
        _READ_EDITABLE_OBJECTS,
        _EDIT_MODEL_OBJECT,
    ),
    AuthoringWorkflowStage.MESH_PENDING: (_READ_CONTEXT,),
    AuthoringWorkflowStage.DEFINITIONS_READY: (
        _READ_CONTEXT,
        _GEOMETRY_FEATURE_CATALOG,
        _SET_REQUIREMENTS,
        _READ_MESH_REFINEMENT_CONTEXT,
        _PREPARE_MESH,
        _READ_MODEL_TOPOLOGY_CONTEXT,
        _APPLY_DEFINITION,
        _RUN_PREFLIGHT,
        _READ_GEOMETRY_EDIT_CONTEXT,
        _PREPARE_GEOMETRY_EDIT,
        _REQUEST_PROJECT_SAVE,
        _READ_DELETABLE_OBJECTS,
        _PREPARE_DELETE,
        _READ_EDITABLE_OBJECTS,
        _EDIT_MODEL_OBJECT,
    ),
    AuthoringWorkflowStage.ANALYSIS_DEFINITIONS_READY: (
        _READ_CONTEXT,
        _GEOMETRY_FEATURE_CATALOG,
        _SET_REQUIREMENTS,
        _READ_MESH_REFINEMENT_CONTEXT,
        _PREPARE_MESH,
        _READ_MODEL_TOPOLOGY_CONTEXT,
        _APPLY_DEFINITION,
        _RUN_PREFLIGHT,
        _READ_GEOMETRY_EDIT_CONTEXT,
        _PREPARE_GEOMETRY_EDIT,
        _REQUEST_PROJECT_SAVE,
        _READ_DELETABLE_OBJECTS,
        _PREPARE_DELETE,
        _READ_EDITABLE_OBJECTS,
        _EDIT_MODEL_OBJECT,
    ),
    AuthoringWorkflowStage.PREFLIGHT_READY: (
        _READ_CONTEXT,
        _GEOMETRY_FEATURE_CATALOG,
        _SET_REQUIREMENTS,
        _READ_MESH_REFINEMENT_CONTEXT,
        _PREPARE_MESH,
        _READ_MODEL_TOPOLOGY_CONTEXT,
        _APPLY_DEFINITION,
        _RUN_PREFLIGHT,
        _READ_GEOMETRY_EDIT_CONTEXT,
        _PREPARE_GEOMETRY_EDIT,
        _REQUEST_PROJECT_SAVE,
        _READ_DELETABLE_OBJECTS,
        _PREPARE_DELETE,
        _READ_EDITABLE_OBJECTS,
        _EDIT_MODEL_OBJECT,
    ),
    AuthoringWorkflowStage.PREFLIGHT_PENDING: (_READ_CONTEXT,),
    AuthoringWorkflowStage.SOLVE_READY: (
        _READ_CONTEXT,
        _GEOMETRY_FEATURE_CATALOG,
        _SET_REQUIREMENTS,
        _READ_MESH_REFINEMENT_CONTEXT,
        _PREPARE_MESH,
        _READ_MODEL_TOPOLOGY_CONTEXT,
        _APPLY_DEFINITION,
        _RUN_PREFLIGHT,
        _PREPARE_SOLVE,
        _READ_GEOMETRY_EDIT_CONTEXT,
        _PREPARE_GEOMETRY_EDIT,
        _REQUEST_PROJECT_SAVE,
        _READ_DELETABLE_OBJECTS,
        _PREPARE_DELETE,
        _READ_EDITABLE_OBJECTS,
        _EDIT_MODEL_OBJECT,
    ),
    AuthoringWorkflowStage.SOLVE_PENDING: (_READ_CONTEXT,),
    AuthoringWorkflowStage.RESULTS_READY: (
        _READ_CONTEXT,
        _GEOMETRY_FEATURE_CATALOG,
        _SET_REQUIREMENTS,
        _READ_MESH_REFINEMENT_CONTEXT,
        _PREPARE_MESH,
        _READ_MODEL_TOPOLOGY_CONTEXT,
        _RUN_PREFLIGHT,
        _PREPARE_SOLVE,
        _RESULT_CATALOG,
        _RESULT_QUERY,
        _RESULT_COMPARISON,
        _READ_GEOMETRY_EDIT_CONTEXT,
        _PREPARE_GEOMETRY_EDIT,
        _APPLY_DEFINITION,
        _REQUEST_PROJECT_SAVE,
        _READ_DELETABLE_OBJECTS,
        _PREPARE_DELETE,
        _READ_EDITABLE_OBJECTS,
        _EDIT_MODEL_OBJECT,
    ),
    AuthoringWorkflowStage.PROJECT_SAVE_PENDING: (_READ_CONTEXT,),
    AuthoringWorkflowStage.DESTRUCTIVE_EDIT_PENDING: (_READ_CONTEXT,),
    AuthoringWorkflowStage.STALE: (_READ_CONTEXT,),
    AuthoringWorkflowStage.CANCELLED: (_READ_CONTEXT,),
}

# Keep the dedicated tools adjacent to the feature catalog in every ready
# native-authoring stage.  Pending/terminal stages intentionally publish only
# their read context and cannot create a proposal.
_PROFILE_TRANSFORM_STAGE_TOOLS = (
    _READ_PROFILE_TRANSFORM_CONTEXT,
    _PREPARE_PROFILE_EXTRUSION,
    _PREPARE_PROFILE_REVOLUTION,
    _PREPARE_PROFILE_PATH_SWEEP,
)
for _stage, _definitions in tuple(_STAGE_TOOLS.items()):
    if _stage not in _PROJECT_SAVE_READY_STAGES:
        continue
    expanded: list[ToolDefinition] = []
    for _definition in _definitions:
        expanded.append(_definition)
        if _definition.name == _GEOMETRY_FEATURE_CATALOG.name:
            expanded.extend(_PROFILE_TRANSFORM_STAGE_TOOLS)
    _STAGE_TOOLS[_stage] = tuple(expanded)


def _stage_requirement_tool(
    group: str,
    keys: tuple[str, ...],
    specs: Mapping[str, Mapping[str, object]],
) -> ToolDefinition:
    description = (
        "Override the blank-project mm-N-MPa defaults only with geometry "
        "units explicitly supplied by the user. Complete values may be used "
        "only to present the geometry operation card."
        if group == "geometry"
        else (
            f"Record only explicitly supplied {group} values for the current "
            "stage. Complete values may be used only to present the matching "
            "geometry or mesh operation card."
        )
    )
    return _tool(
        _SET_REQUIREMENTS.name,
        description,
        {
            "type": "object",
            "properties": {
                "turn_id": {"type": "string"},
                "requirements": {
                    "type": "object",
                    "properties": {key: specs[key] for key in keys},
                    "additionalProperties": False,
                    "minProperties": 1,
                },
            },
            "required": ["turn_id", "requirements"],
            "additionalProperties": False,
        },
    )


class AuthoringWorkflowController:
    """Strict A8 state machine over injected A1-A7 local handlers."""

    def __init__(
        self,
        context_reader: ContextReader,
        handlers: Mapping[str, AuthoringToolHandler],
        *,
        workspace_result_inventory: Callable[[], tuple[int, int]] | None = None,
    ) -> None:
        if not callable(context_reader):
            raise TypeError("context_reader must be callable")
        normalized = dict(handlers)
        if any(
            not isinstance(name, str) or not callable(handler)
            for name, handler in normalized.items()
        ):
            raise TypeError("authoring handlers must be named callables")
        forbidden = [
            name
            for name in normalized
            if name.casefold() in _FORBIDDEN_CONFIRMATION_TOOL_NAMES
        ]
        if forbidden:
            raise ValueError("confirmation handlers cannot be model-callable")
        self._context_reader = context_reader
        self._handlers = normalized
        if workspace_result_inventory is not None and not callable(
            workspace_result_inventory
        ):
            raise TypeError("workspace_result_inventory must be callable or None")
        self._workspace_result_inventory = workspace_result_inventory
        self._ledger = RequirementLedger()
        self._stage = AuthoringWorkflowStage.REQUIREMENTS
        self._pending_review: RequirementReview | None = None
        self._review_binding: tuple[str, str, int] | None = None
        self._review_source_stage: AuthoringWorkflowStage | None = None
        self._pending_operation: str | None = None
        self._project_save_record: ProjectSaveProposalRecord | None = None
        self._geometry_resume_stage: AuthoringWorkflowStage | None = None
        self._mesh_resume_stage: AuthoringWorkflowStage | None = None
        self._destructive_resume_stage: AuthoringWorkflowStage | None = None
        self._pending_destructive_object_type: str | None = None
        self._terminals: list[AuthoringTerminalRecord] = []
        self._planar_retry_state: _PlanarConstructionRetryState | None = None
        self._planar_audit: list[PlanarConstructionAuditRecord] = []
        self._pending_planar_proposal_id: str | None = None
        self._active_tool_context: ToolExecutionContext | None = None
        self._binding_identity: tuple[str, str, int] | None = None
        self._observed_context: AuthoringContext | None = None
        self._snapshot_generation = 0
        self._turn_snapshot = AuthoringTurnSnapshot.unavailable()
        self._lock = threading.RLock()

    @property
    def stage(self) -> AuthoringWorkflowStage:
        with self._lock:
            return self._stage

    @property
    def planar_construction_audit(self) -> tuple[PlanarConstructionAuditRecord, ...]:
        with self._lock:
            return tuple(self._planar_audit)

    def assess_planar_construction_failure(
        self,
        request: Mapping[str, object],
        *,
        code: str,
        node_id: str | None,
        retryable: bool,
        allowed_fields: Sequence[str],
    ) -> dict[str, object]:
        """Bound retries and require a changed failing node or top-level payload."""

        raw_construction = request.get("construction")
        construction = (
            raw_construction if isinstance(raw_construction, Mapping) else request
        )
        payload = json.dumps(
            construction,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=True,
        )
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        fields = tuple(str(field) for field in allowed_fields)
        node_payload = next(
            (
                item
                for item in construction.get("nodes", [])
                if isinstance(item, Mapping) and item.get("id") == node_id
            ),
            None,
        )
        recovery_payload = {
            field: (
                node_payload
                if field == "nodes" and node_payload is not None
                else construction.get("nodes")
                if field == "nodes"
                else node_payload.get(field)
                if node_payload is not None and field in node_payload
                else construction
                if field == "construction"
                else request.get(field)
                if field in request
                else construction.get(field)
            )
            for field in fields
        }
        recovery_digest = hashlib.sha256(
            json.dumps(
                recovery_payload,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=True,
            ).encode("utf-8")
        ).hexdigest()
        active = self._active_tool_context
        turn_id = (
            "local" if active is None else str(active.turn_id or active.idempotency_key)
        )
        with self._lock:
            prior = self._planar_retry_state
            if prior is not None and prior.turn_id != turn_id:
                prior = None
            attempt = 1 if prior is None else prior.attempt + 1
            changed = prior is None or recovery_digest != prior.recovery_digest
            may_retry = bool(retryable and changed and attempt < 3)
            blocker = None
            if prior is not None and not changed:
                blocker = "Retry must modify the failed node or allowed field."
            elif retryable and attempt >= 3:
                blocker = (
                    "Planar construction retry limit reached after three attempts."
                )
            self._planar_retry_state = _PlanarConstructionRetryState(
                turn_id,
                digest,
                recovery_digest,
                str(code),
                node_id,
                fields,
                attempt,
            )
            self._append_planar_audit(
                PlanarConstructionAuditRecord(
                    digest,
                    "validation"
                    if code
                    in {
                        "planar-ir.schema-invalid",
                        "planar-ir.budget-exceeded",
                        "planar-ir.duplicate-node-id",
                        "planar-ir.reference-missing",
                        "planar-ir.cycle-detected",
                        "planar-ir.unreachable-node",
                        "planar-ir.invalid-primitive",
                        "planar-ir.invalid-path-stroke",
                    }
                    else "compile",
                    str(code),
                    None,
                    "failed",
                )
            )
            return {
                "construction_digest": digest,
                "attempt": attempt,
                "limit": 3,
                "retryable": may_retry,
                "blocker": blocker,
            }

    def planar_construction_retry_blocker(self) -> dict[str, object] | None:
        """Reject further planar submissions after this provider turn is exhausted."""

        active = self._active_tool_context
        if active is None:
            return None
        turn_id = str(active.turn_id or active.idempotency_key)
        with self._lock:
            prior = self._planar_retry_state
            if (
                prior is None
                or prior.turn_id != turn_id
                or prior.attempt < 3
            ):
                return None
            return {
                "construction_digest": prior.construction_digest,
                "attempt": prior.attempt,
                "limit": 3,
                "retryable": False,
                "blocker": (
                    "Planar construction retry limit reached after three attempts."
                ),
            }

    def record_planar_construction_proposal(
        self,
        construction_digest: str,
        proposal_id: str,
    ) -> None:
        with self._lock:
            self._planar_retry_state = None
            self._pending_planar_proposal_id = str(proposal_id)
            self._append_planar_audit(
                PlanarConstructionAuditRecord(
                    str(construction_digest),
                    "proposal",
                    None,
                    str(proposal_id),
                    ProposalState.PENDING_CONFIRMATION.value,
                )
            )

    def _append_planar_audit(self, record: PlanarConstructionAuditRecord) -> None:
        self._planar_audit.append(record)
        del self._planar_audit[:-64]

    @property
    def ledger(self) -> RequirementLedger:
        return self._ledger

    @property
    def pending_review(self) -> RequirementReview | None:
        with self._lock:
            return self._pending_review

    @property
    def terminal_records(self) -> tuple[AuthoringTerminalRecord, ...]:
        with self._lock:
            return tuple(self._terminals)

    @property
    def binding_identity(self) -> tuple[str, str, int] | None:
        """Return the last owner-thread-observed model identity."""

        with self._lock:
            return self._binding_identity

    @property
    def turn_snapshot(self) -> AuthoringTurnSnapshot:
        """Return the last owner-thread projected immutable turn snapshot."""

        with self._lock:
            return self._turn_snapshot

    @property
    def authoring_turn_snapshot(self) -> AuthoringTurnSnapshot:
        return self.turn_snapshot

    @property
    def provider_snapshot(self) -> AuthoringTurnSnapshot:
        snapshot = self.turn_snapshot
        # A controller can be used directly by headless baseline tests.  Until
        # its owner adapter binds the actual published tool catalog, exposing
        # document details would claim a provider turn that never received an
        # advertisement.  The Qt adapter calls ``set_published_tool_names``
        # before each turn.
        if not snapshot.published_tool_names:
            return AuthoringTurnSnapshot.unavailable(
                generation=snapshot.snapshot_generation,
            )
        return snapshot

    def refresh_turn_snapshot(
        self,
        published_tool_names: Sequence[str] = (),
    ) -> AuthoringTurnSnapshot:
        """Observe current typed context and update the owner-thread cache.

        The Qt adapter calls this method on its owner thread immediately before
        a provider turn and after local tool/proposal transitions.  Provider
        threads only consume :attr:`turn_snapshot` and never call the context
        reader through this method.
        """

        with self._lock:
            raw = self._context_reader()
            if type(raw) is AuthoringContext:
                self.observe_binding(raw)
            else:
                # A malformed/unavailable reader result must not keep using a
                # previously observed document as if it were current.
                self._observed_context = None
                self._binding_identity = None
            self._snapshot_generation += 1
            self._turn_snapshot = AuthoringTurnSnapshot.from_context(
                self._observed_context,
                workflow_stage=self._stage if self._observed_context else None,
                published_tool_names=published_tool_names,
                generation=self._snapshot_generation,
            )
            return self._turn_snapshot

    def set_published_tool_names(
        self,
        published_tool_names: Sequence[str],
    ) -> AuthoringTurnSnapshot:
        """Bind the owner-thread tool advertisement to the latest snapshot."""

        with self._lock:
            self._snapshot_generation += 1
            self._turn_snapshot = AuthoringTurnSnapshot.from_context(
                self._observed_context,
                workflow_stage=(
                    self._stage if self._observed_context is not None else None
                ),
                published_tool_names=published_tool_names,
                generation=self._snapshot_generation,
            )
            return self._turn_snapshot

    def invalidate_turn_snapshot(self) -> AuthoringTurnSnapshot:
        """Invalidate provider projection without discarding state-machine context."""

        with self._lock:
            self._snapshot_generation += 1
            self._turn_snapshot = AuthoringTurnSnapshot.unavailable(
                generation=self._snapshot_generation,
            )
            return self._turn_snapshot

    def _refresh_turn_snapshot_locked(self) -> None:
        self._snapshot_generation += 1
        self._turn_snapshot = AuthoringTurnSnapshot.from_context(
            self._observed_context,
            workflow_stage=self._stage if self._observed_context else None,
            published_tool_names=self._turn_snapshot.published_tool_names,
            generation=self._snapshot_generation,
        )

    @property
    def project_save_record(self) -> ProjectSaveProposalRecord | None:
        with self._lock:
            return self._project_save_record

    @property
    def definitions(self) -> tuple[ToolDefinition, ...]:
        with self._lock:
            inventory = self._result_inventory()
            if self._observed_context is None and inventory is not None:
                run_count, result_count = inventory
                global_reads = (
                    *(
                        (_WORKSPACE_DOCUMENTS,)
                        if _WORKSPACE_DOCUMENTS.name in self._handlers
                        else ()
                    ),
                    *((_ANALYSIS_RUN_CATALOG,) if run_count > 0 else ()),
                    *((_RESULT_CATALOG, _RESULT_QUERY) if result_count > 0 else ()),
                    *((_RESULT_COMPARISON,) if result_count >= 2 else ()),
                )
                return tuple(
                    item for item in global_reads if item.name in self._handlers
                )
            definitions = list(_STAGE_TOOLS[self._stage])
            if (
                self._stage in _PROJECT_SAVE_READY_STAGES
                and _CREATE_NATIVE_MODEL_DOCUMENT.name in self._handlers
            ):
                definitions.insert(1, _CREATE_NATIVE_MODEL_DOCUMENT)
            if _WORKSPACE_DOCUMENTS.name in self._handlers:
                definitions.append(_WORKSPACE_DOCUMENTS)
            run_count = (
                self._observed_context.run_count
                if inventory is None and self._observed_context is not None
                else 0
                if inventory is None
                else inventory[0]
            )
            result_count = (
                self._observed_context.result_count
                if inventory is None and self._observed_context is not None
                else 0
                if inventory is None
                else inventory[1]
            )
            if self._stage in _PROJECT_SAVE_READY_STAGES and run_count > 0:
                historical_definitions = [_ANALYSIS_RUN_CATALOG]
                if result_count > 0:
                    historical_definitions.extend((_RESULT_CATALOG, _RESULT_QUERY))
                if result_count >= 2:
                    historical_definitions.append(_RESULT_COMPARISON)
                for result_definition in historical_definitions:
                    if all(item.name != result_definition.name for item in definitions):
                        definitions.append(result_definition)
            requirement_group = self._current_requirement_group()
            requirements_complete = (
                requirement_group is not None
                and self._requirement_group_complete(requirement_group)
            )
            gated_operations = _GATED_OPERATIONS_BY_STAGE.get(
                self._stage,
                frozenset(),
            )
            visible: list[ToolDefinition] = []
            for item in definitions:
                if item.name in gated_operations and not requirements_complete:
                    continue
                if (
                    item.name == _APPLY_DEFINITION.name
                    and self._stage
                    in {
                        AuthoringWorkflowStage.REQUIREMENTS,
                        AuthoringWorkflowStage.GEOMETRY_READY,
                        AuthoringWorkflowStage.MESH_READY,
                    }
                    and not self._current_mesh_available()
                ):
                    continue
                if (
                    item.name == _SET_REQUIREMENTS.name
                    and requirement_group is not None
                ):
                    keys = self._requirement_keys(requirement_group)
                    item = _stage_requirement_tool(
                        requirement_group,
                        keys,
                        {key: self._requirement_spec(key) for key in keys},
                    )
                visible.append(item)
            return tuple(
                item
                for item in visible
                if (
                    (
                        item.name
                        in {
                            _READ_CONTEXT.name,
                            _SET_REQUIREMENTS.name,
                        }
                        or item.name in self._handlers
                    )
                    and (item.name != _ANALYSIS_RUN_CATALOG.name or (run_count > 0))
                    and (
                        item.name
                        not in {
                            _RESULT_CATALOG.name,
                            _RESULT_QUERY.name,
                            _RESULT_COMPARISON.name,
                        }
                        or (
                            result_count
                            >= (2 if item.name == _RESULT_COMPARISON.name else 1)
                        )
                    )
                    and (
                        item.name != _REQUEST_PROJECT_SAVE.name
                        or self._project_save_available()
                    )
                    and (
                        item.name not in _DESTRUCTIVE_EDIT_TOOLS
                        or self._destructive_edit_available()
                    )
                    and (
                        item.name not in _MODEL_EDIT_TOOLS
                        or self._model_edit_available()
                    )
                    and (
                        item.name not in _GEOMETRY_EDIT_TOOLS
                        or self._geometry_edit_available()
                    )
                    and (
                        item.name not in _GEOMETRY_CATALOG_TOOLS
                        or self._geometry_catalog_available()
                    )
                    and (
                        item.name not in _PROFILE_TRANSFORM_TOOLS
                        or self._profile_transform_available()
                    )
                )
            )

    def _result_inventory(self) -> tuple[int, int] | None:
        reader = self._workspace_result_inventory
        if reader is None:
            return None
        run_count, result_count = reader()
        if (
            type(run_count) is not int
            or type(result_count) is not int
            or run_count < 0
            or result_count < 0
            or result_count > run_count
        ):
            raise ValueError("workspace result inventory is invalid")
        return run_count, result_count

    def dispatch(
        self,
        name: str,
        arguments: Mapping[str, Any],
        context: ToolExecutionContext,
    ) -> ToolResult:
        with self._lock:
            available = {item.name for item in self.definitions}
            if (
                name not in available
                and self._stage
                in {
                    AuthoringWorkflowStage.STALE,
                    AuthoringWorkflowStage.CANCELLED,
                }
                and name.startswith("read_")
                and (name == _READ_CONTEXT.name or name in self._handlers)
            ):
                try:
                    self._read_context()
                except Exception as error:
                    return self._failure(
                        context,
                        DiagnosticCode.INVALID_MODEL,
                        f"Authoring context resynchronization failed: {error}",
                    )
                available = {item.name for item in self.definitions}
            if name not in available:
                return self._failure(
                    context,
                    DiagnosticCode.UNKNOWN_TOOL,
                    f"Authoring tool {name!r} is unavailable in stage {self._stage.value}.",
                )
            try:
                if name == _READ_CONTEXT.name:
                    _require_exact_fields(arguments, set())
                    outcome = self._read_context()
                elif name == _SET_REQUIREMENTS.name:
                    outcome = self._set_requirements(arguments)
                elif name == _REQUEST_REVIEW.name:
                    _require_exact_fields(arguments, set())
                    outcome = self._request_review()
                else:
                    _require_exact_fields(arguments, set()) if name not in {
                        RESULT_QUERY_TOOL_NAME,
                        RESULT_CATALOG_TOOL_NAME,
                        RESULT_COMPARISON_TOOL_NAME,
                        ANALYSIS_RUN_CATALOG_TOOL_NAME,
                        _RUN_PREFLIGHT.name,
                        _PREPARE_SOLVE.name,
                        _PREPARE_DELETE.name,
                        _CREATE_NATIVE_MODEL_DOCUMENT.name,
                        _PREPARE_GEOMETRY.name,
                        _PREPARE_PLANAR_CONSTRUCTION.name,
                        _GEOMETRY_FEATURE_CATALOG.name,
                        _READ_GEOMETRY_EDIT_CONTEXT.name,
                        _PREPARE_GEOMETRY_EDIT.name,
                        *_PROFILE_TRANSFORM_TOOLS,
                        _PREPARE_MESH.name,
                        _EDIT_MODEL_OBJECT.name,
                        _APPLY_DEFINITION.name,
                    } else None
                    self._active_tool_context = context
                    try:
                        outcome = self._handlers[name](dict(arguments), self)
                    finally:
                        self._active_tool_context = None
                    if type(outcome) is not AuthoringToolOutcome:
                        raise TypeError(
                            "authoring handler must return AuthoringToolOutcome"
                        )
                    self._advance_after_success(name, outcome)
                safe_data = provider_safe_authoring_payload(outcome.data)
                return ToolResult(
                    ok=outcome.ok,
                    session_id=context.session_id,
                    input_revision=max(context.expected_revision, 0),
                    idempotency_key=context.idempotency_key,
                    summary=outcome.summary,
                    data=safe_data,
                    diagnostics=(
                        ()
                        if outcome.ok
                        else (
                            make_diagnostic(
                                DiagnosticCode.INVALID_INPUT,
                                outcome.summary,
                                source="agent.authoring_runtime",
                            ),
                        )
                    ),
                )
            except Exception as error:
                error_type = (
                    re.sub(
                        r"[^A-Za-z0-9_]",
                        "",
                        type(error).__name__,
                    )[:64]
                    or "LocalAuthoringError"
                )
                return ToolResult(
                    ok=False,
                    session_id=context.session_id,
                    input_revision=max(context.expected_revision, 0),
                    idempotency_key=context.idempotency_key,
                    summary="The local authoring tool rejected the request.",
                    diagnostics=(
                        make_diagnostic(
                            DiagnosticCode.INVALID_TOOL_ARGUMENTS,
                            (
                                f"{error_type}: the local authoring request "
                                "was rejected."
                            ),
                            source="agent.authoring_runtime",
                        ),
                    ),
                )

    def resolve_requirement_review(
        self,
        review: RequirementReview,
    ) -> None:
        """Advance only after the existing GUI bridge returned a review state."""

        if type(review) is not RequirementReview:
            raise TypeError("review must be RequirementReview")
        with self._lock:
            pending = self._pending_review
            source_stage = self._review_source_stage
            if (
                pending is None
                or source_stage is None
                or pending.review_id != review.review_id
                or pending.review_hash != review.review_hash
            ):
                raise ValueError("RequirementReview does not match the pending review")
            if review.status is RequirementReviewStatus.CONFIRMED:
                self._stage = (
                    AuthoringWorkflowStage.GEOMETRY_READY
                    if source_stage is AuthoringWorkflowStage.REQUIREMENTS
                    else source_stage
                )
            elif review.status in {
                RequirementReviewStatus.REJECTED,
                RequirementReviewStatus.STALE,
            }:
                self._stage = source_stage
            else:
                raise ValueError("RequirementReview is not terminal")
            self._pending_review = None
            self._review_binding = None
            self._review_source_stage = None
            self._record_terminal("requirement_review", review.status.value, "")
            self._refresh_turn_snapshot_locked()

    def stale_review_for_binding(
        self,
        context: AuthoringContext,
    ) -> bool:
        if type(context) is not AuthoringContext:
            raise TypeError("context must be AuthoringContext")
        current = (
            context.binding.document_id,
            context.binding.session_id,
            context.binding.session_revision,
        )
        with self._lock:
            if (
                self._stage is not AuthoringWorkflowStage.REVIEW_PENDING
                or self._review_binding is None
                or self._review_binding == current
            ):
                return False
            self._record_terminal(
                "requirement_review",
                "stale",
                "document, session, or revision changed",
            )
            self._pending_review = None
            self._review_binding = None
            self._review_source_stage = None
            self._binding_identity = current
            self._ledger = RequirementLedger()
            self._stage = AuthoringWorkflowStage.STALE
            self._observed_context = context
            self._refresh_turn_snapshot_locked()
            return True

    def observe_binding(
        self,
        context: AuthoringContext,
        *,
        proposal_staled: bool = False,
        saved_state_transition: bool = False,
        project_save_terminal_transition: bool = False,
    ) -> bool:
        """Track accepted GUI identity and reject cross-binding stage reuse."""

        if type(context) is not AuthoringContext:
            raise TypeError("context must be AuthoringContext")
        current = (
            context.binding.document_id,
            context.binding.session_id,
            context.binding.session_revision,
        )
        with self._lock:
            previous_context = self._observed_context
            self._observed_context = context
            prior = self._binding_identity
            if prior is None:
                self._binding_identity = current
                if self._stage is not AuthoringWorkflowStage.STALE:
                    self._stage = _restored_stage_for_context(context)
                self._seed_default_geometry_requirements(context)
                self._refresh_turn_snapshot_locked()
                return True
            if prior == current:
                self._seed_default_geometry_requirements(context)
                if self._pending_operation is None:
                    was_active_job = (
                        previous_context is not None
                        and _job_status_is_active(previous_context.job_status)
                    )
                    is_active_job = _job_status_is_active(context.job_status)
                    if is_active_job or (
                        self._stage is AuthoringWorkflowStage.SOLVE_PENDING
                        and was_active_job
                    ):
                        self._stage = _restored_stage_for_context(context)
                self._refresh_turn_snapshot_locked()
                return True
            same_session = prior[:2] == current[:2]
            revision_increased = current[2] > prior[2]
            first_native_project_transition = (
                self._pending_operation == "geometry"
                and prior[2] == 0
                and current[2] == 1
            )
            expected_local_transition = (
                not proposal_staled
                and revision_increased
                and (
                    (
                        same_session
                        and (
                            self._active_tool_context is not None
                            or self._pending_operation
                            in {
                                "geometry",
                                "mesh",
                                "preflight",
                                "solve",
                                "destructive_edit",
                            }
                        )
                    )
                    or first_native_project_transition
                    or (same_session and saved_state_transition)
                    or (
                        same_session
                        and project_save_terminal_transition
                        and self._pending_operation == "project_save"
                    )
                )
            )
            self._binding_identity = current
            if expected_local_transition:
                self._refresh_turn_snapshot_locked()
                return True
            operation = self._pending_operation or "binding"
            if self._pending_operation == "project_save":
                self._mark_project_save_terminal(
                    ProposalState.STALE,
                    (
                        "a pending project save was staled by the binding change"
                        if proposal_staled
                        else "document, session, or revision changed"
                    ),
                )
            self._record_terminal(
                operation,
                "stale",
                (
                    "a pending proposal was staled by the binding change"
                    if proposal_staled
                    else (
                        "document, session, or revision changed outside "
                        "the active workflow"
                    )
                ),
            )
            self._pending_operation = None
            self._clear_destructive_pending()
            self._pending_review = None
            self._review_binding = None
            self._review_source_stage = None
            self._ledger = RequirementLedger()
            self._stage = AuthoringWorkflowStage.STALE
            self._refresh_turn_snapshot_locked()
            return False

    def record_proposal_state(
        self,
        operation: str,
        state: ProposalState | str,
        message: str = "",
    ) -> None:
        """Consume a state returned by the existing GUI-controlled bridge."""

        normalized_operation = str(operation)
        normalized_state = ProposalState(state)
        with self._lock:
            if self._pending_operation != normalized_operation:
                raise ValueError("proposal state does not match the pending operation")
            if normalized_state in {
                ProposalState.PENDING_CONFIRMATION,
                ProposalState.ACCEPTED,
                ProposalState.RUNNING,
            }:
                return
            if normalized_operation == "geometry":
                self._stage = (
                    AuthoringWorkflowStage.MESH_READY
                    if normalized_state is ProposalState.SUCCEEDED
                    else (
                        self._geometry_resume_stage
                        or AuthoringWorkflowStage.GEOMETRY_READY
                    )
                )
                self._geometry_resume_stage = None
            elif normalized_operation == "mesh":
                self._stage = (
                    AuthoringWorkflowStage.DEFINITIONS_READY
                    if normalized_state is ProposalState.SUCCEEDED
                    else (self._mesh_resume_stage or AuthoringWorkflowStage.MESH_READY)
                )
                self._mesh_resume_stage = None
            elif normalized_operation == "solve":
                self._stage = (
                    AuthoringWorkflowStage.RESULTS_READY
                    if normalized_state is ProposalState.SUCCEEDED
                    else AuthoringWorkflowStage.SOLVE_READY
                )
            elif normalized_operation == "destructive_edit":
                resume_stage = self._destructive_resume_stage
                object_type = self._pending_destructive_object_type
                if resume_stage is None or object_type is None:
                    raise ValueError("destructive edit has no pending target")
                if normalized_state is ProposalState.SUCCEEDED:
                    if (
                        object_type in {"part", "feature", "generated_mesh"}
                        and self._observed_context is not None
                    ):
                        self._stage = _restored_stage_for_context(
                            self._observed_context
                        )
                        self._seed_default_geometry_requirements(self._observed_context)
                    else:
                        self._stage = {
                            "part": AuthoringWorkflowStage.GEOMETRY_READY,
                            "feature": AuthoringWorkflowStage.GEOMETRY_READY,
                            "generated_mesh": AuthoringWorkflowStage.MESH_READY,
                            "named_region": AuthoringWorkflowStage.DEFINITIONS_READY,
                            "analysis_step": (
                                AuthoringWorkflowStage.ANALYSIS_DEFINITIONS_READY
                            ),
                            "boundary_condition": (
                                AuthoringWorkflowStage.ANALYSIS_DEFINITIONS_READY
                            ),
                            "load": (AuthoringWorkflowStage.ANALYSIS_DEFINITIONS_READY),
                            "result_request": (
                                AuthoringWorkflowStage.ANALYSIS_DEFINITIONS_READY
                            ),
                        }[object_type]
                else:
                    self._stage = resume_stage
                self._clear_destructive_pending()
            else:
                raise ValueError("unknown pending proposal operation")
            if (
                normalized_operation == "geometry"
                and self._pending_planar_proposal_id is not None
            ):
                pending_id = self._pending_planar_proposal_id
                pending = next(
                    (
                        item
                        for item in reversed(self._planar_audit)
                        if item.proposal_id == pending_id
                    ),
                    None,
                )
                if pending is not None:
                    self._append_planar_audit(
                        replace(pending, terminal_state=normalized_state.value)
                    )
                self._pending_planar_proposal_id = None
            self._pending_operation = None
            self._record_terminal(
                normalized_operation,
                normalized_state.value,
                message,
            )
            self._refresh_turn_snapshot_locked()

    def invalidate_binding(self, reason: str) -> None:
        normalized = str(reason).strip()
        if not normalized:
            raise ValueError("stale reason must be non-blank")
        with self._lock:
            operation = self._pending_operation or "binding"
            if self._pending_operation == "project_save":
                self._mark_project_save_terminal(
                    ProposalState.STALE,
                    normalized,
                )
            self._record_terminal(operation, ProposalState.STALE.value, normalized)
            self._pending_operation = None
            self._clear_destructive_pending()
            self._pending_review = None
            self._review_binding = None
            self._review_source_stage = None
            self._ledger = RequirementLedger()
            # A document/session transition invalidates the last typed GUI
            # context.  Keep the workflow terminal record, but do not let
            # provider-facing projections or dynamic definitions reuse that
            # context until the owner thread observes a fresh one.
            self._observed_context = None
            self._binding_identity = None
            self._stage = AuthoringWorkflowStage.STALE
            self.invalidate_turn_snapshot()

    def record_preflight_state(
        self,
        state: str,
        message: str = "",
    ) -> None:
        normalized = str(state).strip().casefold()
        if normalized not in {
            "passed",
            "blocked",
            "failed",
            "cancelled",
            "stale",
        }:
            raise ValueError("unknown preflight terminal state")
        with self._lock:
            if (
                self._stage is not AuthoringWorkflowStage.PREFLIGHT_PENDING
                or self._pending_operation != "preflight"
            ):
                raise ValueError("there is no pending preflight")
            self._stage = (
                AuthoringWorkflowStage.SOLVE_READY
                if normalized == "passed"
                else AuthoringWorkflowStage.PREFLIGHT_READY
            )
            self._pending_operation = None
            self._record_terminal("preflight", normalized, message)
            self._refresh_turn_snapshot_locked()

    def cancel_turn(self, reason: str = "provider turn cancelled") -> None:
        normalized = str(reason).strip()
        with self._lock:
            self._record_terminal("provider_turn", "cancelled", normalized)
            if self._stage not in {
                AuthoringWorkflowStage.GEOMETRY_PENDING,
                AuthoringWorkflowStage.MESH_PENDING,
                AuthoringWorkflowStage.SOLVE_PENDING,
                AuthoringWorkflowStage.PROJECT_SAVE_PENDING,
                AuthoringWorkflowStage.DESTRUCTIVE_EDIT_PENDING,
            }:
                self._stage = AuthoringWorkflowStage.CANCELLED
            self._refresh_turn_snapshot_locked()

    def reset_for_binding(self) -> None:
        with self._lock:
            self._ledger = RequirementLedger()
            self._pending_review = None
            self._review_binding = None
            self._review_source_stage = None
            self._pending_operation = None
            self._project_save_record = None
            self._geometry_resume_stage = None
            self._mesh_resume_stage = None
            self._planar_retry_state = None
            self._pending_planar_proposal_id = None
            self._clear_destructive_pending()
            self._binding_identity = None
            self._observed_context = None
            self._stage = AuthoringWorkflowStage.REQUIREMENTS
            self._refresh_turn_snapshot_locked()

    def register_project_save_proposal(
        self,
        proposal_id: str,
        context: AuthoringContext,
    ) -> ProjectSaveProposalRecord:
        """Register one path-free save card during a model tool dispatch."""

        record = self.preview_project_save_proposal(proposal_id, context)
        with self._lock:
            self._project_save_record = record
            return record

    def preview_project_save_proposal(
        self,
        proposal_id: str,
        context: AuthoringContext,
    ) -> ProjectSaveProposalRecord:
        """Build the path-free save card record without registering it."""

        normalized_id = _nonblank_string(proposal_id, "proposal_id")
        if type(context) is not AuthoringContext:
            raise TypeError("context must be AuthoringContext")
        with self._lock:
            if not self._project_save_available(context):
                raise ValueError("current native project is not ready to save")
            active = self._project_save_record
            if active is not None and active.state in {
                ProposalState.PENDING_CONFIRMATION,
                ProposalState.RUNNING,
            }:
                raise ValueError("a project save proposal is already active")
            binding = context.binding
            identity = {
                "proposal_id": normalized_id,
                "target_document_id": binding.document_id,
                "target_session_id": binding.session_id,
                "base_session_revision": binding.session_revision,
                "operation": "project_save",
            }
            proposal_hash = hashlib.sha256(
                json.dumps(
                    identity,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            return ProjectSaveProposalRecord(
                normalized_id,
                proposal_hash,
                binding.document_id,
                binding.session_id,
                binding.session_revision,
                self._stage,
            )

    def can_accept_project_save_from_gui(
        self,
        proposal_id: str,
        proposal_hash: str,
        context: AuthoringContext,
    ) -> bool:
        if type(context) is not AuthoringContext:
            return False
        with self._lock:
            record = self._project_save_record
            binding = context.binding
            return bool(
                self._stage is AuthoringWorkflowStage.PROJECT_SAVE_PENDING
                and self._pending_operation == "project_save"
                and record is not None
                and record.state is ProposalState.PENDING_CONFIRMATION
                and record.proposal_id == proposal_id
                and record.proposal_hash == proposal_hash
                and binding.supported
                and binding.source_kind == "native"
                and binding.document_id == record.target_document_id
                and binding.session_id == record.target_session_id
                and binding.session_revision == record.base_session_revision
                and _capability_enabled(context, "request_project_save")
            )

    def begin_project_save_from_gui(
        self,
        proposal_id: str,
        proposal_hash: str,
        context: AuthoringContext,
    ) -> None:
        """Consume the sole GUI acceptance edge for a save proposal."""

        with self._lock:
            if not self.can_accept_project_save_from_gui(
                proposal_id,
                proposal_hash,
                context,
            ):
                raise ValueError("project save proposal is stale or already consumed")
            assert self._project_save_record is not None
            self._project_save_record = replace(
                self._project_save_record,
                state=ProposalState.RUNNING,
            )

    def record_project_save_state(
        self,
        proposal_id: str,
        proposal_hash: str,
        state: ProposalState | str,
        message: str = "",
    ) -> None:
        """Record exactly one local save terminal returned by MainWindow."""

        normalized_state = ProposalState(state)
        if normalized_state not in {
            ProposalState.SUCCEEDED,
            ProposalState.FAILED,
            ProposalState.CANCELLED,
            ProposalState.STALE,
            ProposalState.REJECTED,
        }:
            raise ValueError("project save state must be terminal")
        with self._lock:
            record = self._project_save_record
            if (
                record is None
                or record.proposal_id != proposal_id
                or record.proposal_hash != proposal_hash
                or record.state
                not in {
                    ProposalState.PENDING_CONFIRMATION,
                    ProposalState.RUNNING,
                }
            ):
                raise ValueError("project save state does not match an active proposal")
            self._mark_project_save_terminal(normalized_state, message)
            self._pending_operation = None
            self._stage = record.resume_stage
            self._record_terminal(
                "project_save",
                normalized_state.value,
                message,
            )
            self._refresh_turn_snapshot_locked()

    def confirmed_requirements(
        self,
        group: str = "all",
    ) -> dict[str, object]:
        if group == "all":
            keys = _REQUIRED_REQUIREMENTS
        else:
            keys = self._requirement_keys(group, handler=True)
        return {
            item.key: item.value
            for item in self._ledger.require_confirmed(
                group,
                keys,
            )
        }

    def collected_requirements(
        self,
        group: str,
    ) -> dict[str, object]:
        """Return a complete explicit geometry or mesh requirement group."""

        keys = self._requirement_keys(group, handler=True)
        values = {
            item.key: item.value
            for item in self._ledger.entries
            if item.status
            in {
                RequirementStatus.PROPOSED,
                RequirementStatus.CONFIRMED,
            }
        }
        missing = [key for key in keys if key not in values]
        if missing:
            raise ValueError("clarification_required: " + ", ".join(missing))
        return {key: values[key] for key in keys}

    def defaulted_requirement_keys(
        self,
        group: str,
    ) -> tuple[str, ...]:
        """Return active requirement keys supplied by local defaults."""

        keys = self._requirement_keys(group, handler=True)
        entries = {
            item.key: item
            for item in self._ledger.entries
            if item.status
            in {
                RequirementStatus.PROPOSED,
                RequirementStatus.CONFIRMED,
            }
        }
        return tuple(
            key
            for key in keys
            if key in entries
            and entries[key].source_turn_id == _DEFAULT_REQUIREMENT_SOURCE_TURN_ID
        )

    def _current_requirement_group(self) -> str | None:
        stage = (
            self._review_source_stage
            if self._stage is AuthoringWorkflowStage.REVIEW_PENDING
            else self._stage
        )
        if stage is None:
            return None
        return _REQUIREMENT_GATE_BY_STAGE.get(stage)

    def _active_part_dimension(self) -> int | None:
        context = self._observed_context
        if context is None or context.active_part_id is None:
            return None
        part = next(
            (
                item
                for item in context.parts
                if item.part_id == context.active_part_id and not item.suppressed
            ),
            None,
        )
        return None if part is None else part.dimension

    def _requirement_keys(
        self,
        group: str,
        *,
        handler: bool = False,
    ) -> tuple[str, ...]:
        source = _HANDLER_REQUIREMENTS if handler else _REQUIREMENT_GROUPS
        try:
            keys = source[group]
        except KeyError as exc:
            raise ValueError("unknown requirement group") from exc
        if group == "mesh" and self._active_part_dimension() == 1:
            return _LINE_MESH_REQUIREMENTS
        return keys

    def _requirement_spec(self, key: str) -> Mapping[str, object]:
        spec = _REQUIREMENT_SPECS[key]
        dimension = self._active_part_dimension()
        if key == "mesh_cell_shape":
            if dimension == 1:
                return {"type": "string", "enum": ["line"]}
            if dimension == 3:
                return {
                    "type": "string",
                    "enum": ["tetrahedron", "hexahedron"],
                }
            return {
                "type": "string",
                "enum": ["triangle", "quadrilateral"],
            }
        if key == "mesh_order" and dimension == 1:
            return {"type": "integer", "enum": [1]}
        if key == "fixed_dofs":
            maximum = 3 if dimension == 3 else 2
            return {
                "type": "array",
                "items": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": maximum,
                },
                "minItems": 1,
                "maxItems": maximum,
                "uniqueItems": True,
            }
        if key == "load_type":
            return {
                "type": "string",
                "enum": (
                    ["nodal", "surface_traction", "surface_pressure"]
                    if dimension == 3
                    else ["nodal", "edge_traction", "edge_pressure"]
                ),
            }
        if key == "load_direction":
            return {
                "type": "string",
                "enum": (
                    ["x", "y", "z", "inward_normal", "outward_normal"]
                    if dimension == 3
                    else ["x", "y", "inward_normal", "outward_normal"]
                ),
            }
        return spec

    def _requirement_group_complete(self, group: str) -> bool:
        required = set(self._requirement_keys(group))
        recorded = {
            item.key
            for item in self._ledger.entries
            if item.status
            in {
                RequirementStatus.PROPOSED,
                RequirementStatus.CONFIRMED,
            }
        }
        return required <= recorded

    def _seed_default_geometry_requirements(
        self,
        context: AuthoringContext,
    ) -> None:
        if context.binding.source_kind not in {"blank", "native"} or context.parts:
            return
        recorded = {
            item.key
            for item in self._ledger.entries
            if item.status
            in {
                RequirementStatus.PROPOSED,
                RequirementStatus.CONFIRMED,
            }
        }
        accepted_units = (
            {}
            if context.unit_context is None
            else {
                "length_unit": context.unit_context.length,
                "force_unit": context.unit_context.force,
                "stress_unit": context.unit_context.stress,
            }
        )
        values = accepted_units or _DEFAULT_GEOMETRY_REQUIREMENTS
        source_turn_id = (
            "accepted-unit-context"
            if accepted_units
            else _DEFAULT_REQUIREMENT_SOURCE_TURN_ID
        )
        for key, value in values.items():
            if value is None:
                continue
            if key in recorded:
                continue
            self._ledger.record(
                key,
                field_type=str(_REQUIREMENT_SPECS[key]["type"]),
                stage=_requirement_stage(key),
                value=value,
                source_turn_id=source_turn_id,
                status=RequirementStatus.PROPOSED,
            )

    def invocation_metadata(self, prefix: str) -> dict[str, object]:
        """Return bounded envelope identities for the active local handler."""

        normalized = re.sub(r"[^A-Za-z0-9_-]", "", str(prefix))[:24]
        if not normalized:
            raise ValueError("invocation prefix must contain an identifier")
        with self._lock:
            context = self._active_tool_context
            if context is None:
                raise RuntimeError(
                    "invocation metadata is available only during dispatch"
                )
            suffix = context.idempotency_key[:24]
            return {
                "agent_session_id": context.session_id,
                "turn_id": f"turn-{suffix}",
                "source_tool_call_ids": (f"call-{suffix}",),
                "draft_revision": self._ledger.revision,
                "identity_suffix": f"{normalized}-{suffix}",
            }

    def _read_context(self) -> AuthoringToolOutcome:
        raw = self._context_reader()
        if self._stage in {
            AuthoringWorkflowStage.STALE,
            AuthoringWorkflowStage.CANCELLED,
        }:
            self.reset_for_binding()
        if type(raw) is AuthoringContext:
            self.observe_binding(raw)
        data = (
            raw.to_provider_dict()
            if type(raw) is AuthoringContext
            else dict(raw)
            if isinstance(raw, Mapping)
            else None
        )
        if data is None:
            raise TypeError("context_reader must return AuthoringContext or an object")
        published_authoring_tools = tuple(item.name for item in self.definitions)
        published_set = set(published_authoring_tools)
        raw_capabilities = data.get("capabilities")
        if isinstance(raw_capabilities, list):
            data["capabilities"] = [
                dict(item)
                for item in raw_capabilities
                if isinstance(item, Mapping) and item.get("operation") in published_set
            ]
        data["published_authoring_tool_names"] = list(published_authoring_tools)
        if _PREPARE_PLANAR_CONSTRUCTION.name in published_set:
            data["planar_construction_ir"] = _planar_construction_context_capability()
        requirement_group = self._current_requirement_group()
        required = (
            ()
            if requirement_group is None
            else self._requirement_keys(requirement_group)
        )
        recorded = {
            item.key
            for item in self._ledger.entries
            if item.status
            in {
                RequirementStatus.PROPOSED,
                RequirementStatus.CONFIRMED,
            }
        }
        return AuthoringToolOutcome(
            "Bounded authoring context read locally.",
            {
                "workflow_stage": self._stage.value,
                "requirement_stage": requirement_group,
                "context": data,
                "missing_requirements": [
                    key for key in required if key not in recorded
                ],
                "defaulted_requirements": list(
                    self.defaulted_requirement_keys(requirement_group)
                    if requirement_group is not None
                    else ()
                ),
            },
        )

    def _set_requirements(
        self,
        arguments: Mapping[str, Any],
    ) -> AuthoringToolOutcome:
        data = _require_exact_fields(
            arguments,
            {"turn_id", "requirements"},
        )
        turn_id = _nonblank_string(data["turn_id"], "turn_id")
        raw_requirements = data["requirements"]
        if not isinstance(raw_requirements, Mapping) or not raw_requirements:
            raise ValueError("requirements must be a non-empty object")
        requirement_group = self._current_requirement_group()
        if requirement_group is None:
            raise ValueError("there is no active requirement stage")
        allowed = set(self._requirement_keys(requirement_group))
        unknown = set(raw_requirements) - set(_REQUIREMENT_SPECS)
        if unknown:
            raise ValueError(
                f"unknown requirement fields: {', '.join(sorted(unknown))}"
            )
        out_of_stage = set(raw_requirements) - allowed
        if out_of_stage:
            raise ValueError(
                "out-of-stage requirement fields: " + ", ".join(sorted(out_of_stage))
            )
        for key, value in raw_requirements.items():
            _validate_requirement_value(
                key,
                value,
                self._requirement_spec(key),
            )
        for key, value in raw_requirements.items():
            self._ledger.record(
                key,
                field_type=str(self._requirement_spec(key)["type"]),
                stage=_requirement_stage(key),
                value=value,
                source_turn_id=turn_id,
                status=RequirementStatus.PROPOSED,
            )
        recorded = {
            item.key
            for item in self._ledger.entries
            if item.status
            in {
                RequirementStatus.PROPOSED,
                RequirementStatus.CONFIRMED,
            }
        }
        missing = [
            key
            for key in self._requirement_keys(requirement_group)
            if key not in recorded
        ]
        return AuthoringToolOutcome(
            "Explicit current-operation requirements recorded.",
            {
                "ledger_revision": self._ledger.revision,
                "requirement_stage": requirement_group,
                "recorded": sorted(raw_requirements),
                "missing_requirements": missing,
                "operation_confirmation_required": not missing,
            },
        )

    def _request_review(self) -> AuthoringToolOutcome:
        requirement_group = self._current_requirement_group()
        if requirement_group is None:
            raise ValueError("there is no active requirement stage")
        required = self._requirement_keys(requirement_group)
        recorded = {
            item.key
            for item in self._ledger.entries
            if item.status
            in {
                RequirementStatus.PROPOSED,
                RequirementStatus.CONFIRMED,
            }
        }
        missing = [key for key in required if key not in recorded]
        if missing:
            raise ValueError("clarification_required: " + ", ".join(missing))
        values = {item.key: item.value for item in self._ledger.entries}
        if requirement_group == "analysis":
            _validate_supported_requirement_combination(
                values,
                dimension=self._active_part_dimension(),
            )
        review = self._ledger.create_review(
            f"review-{uuid.uuid4().hex}",
            required,
        )
        raw_context = self._context_reader()
        context_data = (
            raw_context.to_provider_dict()
            if type(raw_context) is AuthoringContext
            else dict(raw_context)
            if isinstance(raw_context, Mapping)
            else None
        )
        if context_data is None or not isinstance(
            context_data.get("binding"),
            Mapping,
        ):
            raise TypeError("context_reader returned no binding")
        binding = dict(context_data["binding"])
        self._pending_review = review
        self._review_source_stage = self._stage
        self._review_binding = (
            str(binding["document_id"]),
            str(binding["session_id"]),
            int(binding["session_revision"]),
        )
        self._stage = AuthoringWorkflowStage.REVIEW_PENDING
        self._refresh_turn_snapshot_locked()
        title, summary, impact = {
            "geometry": (
                "确认几何需求",
                f"请审阅 {len(review.fields)} 项几何与项目单位参数",
                "确认后这些值才可用于创建几何",
            ),
            "mesh": (
                "确认网格需求",
                f"请审阅 {len(review.fields)} 项网格参数",
                "确认后这些值才可用于划分网格",
            ),
            "definitions": (
                "确认材料与截面需求",
                f"请审阅 {len(review.fields)} 项材料与截面参数",
                "确认后这些值才可用于材料、截面和指派",
            ),
            "analysis": (
                "确认分析需求",
                f"请审阅 {len(review.fields)} 项边界条件、载荷与结果参数",
                "确认后这些值才可用于分析定义",
            ),
        }[requirement_group]
        return AuthoringToolOutcome(
            "RequirementReview is waiting for the local GUI control.",
            {
                "review_id": review.review_id,
                "review_hash": review.review_hash,
                "ledger_revision": review.ledger_revision,
                "status": review.status.value,
                "requirement_stage": requirement_group,
                "fields": [item.to_dict() for item in review.fields],
                "proposal_view": {
                    "proposal_id": review.review_id,
                    "proposal_hash": review.review_hash,
                    "proposal_kind": "requirement_review",
                    "title": title,
                    "summary": summary,
                    "impact": impact,
                    "confirm_label": "确认",
                    "target_document_id": binding["document_id"],
                    "target_session_id": binding["session_id"],
                    "base_session_revision": binding["session_revision"],
                },
            },
        )

    def _advance_after_success(
        self,
        name: str,
        outcome: AuthoringToolOutcome,
    ) -> None:
        if not outcome.ok:
            return
        if name == _PREPARE_GEOMETRY.name:
            self._stage = AuthoringWorkflowStage.GEOMETRY_PENDING
            self._pending_operation = "geometry"
        elif name == _PREPARE_PLANAR_CONSTRUCTION.name:
            self._stage = AuthoringWorkflowStage.GEOMETRY_PENDING
            self._pending_operation = "geometry"
        elif name == _PREPARE_GEOMETRY_EDIT.name:
            self._geometry_resume_stage = self._stage
            self._stage = AuthoringWorkflowStage.GEOMETRY_PENDING
            self._pending_operation = "geometry"
        elif name in {
            _PREPARE_PROFILE_EXTRUSION.name,
            _PREPARE_PROFILE_REVOLUTION.name,
            _PREPARE_PROFILE_PATH_SWEEP.name,
        }:
            self._geometry_resume_stage = self._stage
            self._stage = AuthoringWorkflowStage.GEOMETRY_PENDING
            self._pending_operation = "geometry"
        elif name == _PREPARE_MESH.name:
            self._mesh_resume_stage = self._stage
            self._stage = AuthoringWorkflowStage.MESH_PENDING
            self._pending_operation = "mesh"
        elif name == _APPLY_DEFINITION.name:
            object_type = outcome.data.get("definition_object_type")
            if object_type not in {"named_region", "analysis_step"}:
                raise ValueError(
                    "definition handler registered no exact resume object type"
                )
            if "proposal_id" in outcome.data:
                self._destructive_resume_stage = self._stage
                self._pending_destructive_object_type = object_type
                self._stage = AuthoringWorkflowStage.DESTRUCTIVE_EDIT_PENDING
                self._pending_operation = "destructive_edit"
            else:
                self._stage = (
                    _restored_stage_for_context(self._observed_context)
                    if self._observed_context is not None
                    else (
                        AuthoringWorkflowStage.ANALYSIS_DEFINITIONS_READY
                        if object_type == "analysis_step"
                        else AuthoringWorkflowStage.DEFINITIONS_READY
                    )
                )
                self._record_terminal(
                    "model_definition",
                    "succeeded",
                    outcome.summary,
                )
        elif name == _EDIT_MODEL_OBJECT.name:
            object_type = outcome.data.get("edit_object_type")
            if object_type not in {
                "named_region",
                "material",
                "section",
                "section_assignment",
                "analysis_step",
                "boundary_condition",
                "load",
                "result_request",
            }:
                raise ValueError("edit handler registered no exact edited object type")
            if "proposal_id" in outcome.data:
                self._destructive_resume_stage = self._stage
                self._pending_destructive_object_type = object_type
                self._stage = AuthoringWorkflowStage.DESTRUCTIVE_EDIT_PENDING
                self._pending_operation = "destructive_edit"
            else:
                self._stage = (
                    _restored_stage_for_context(self._observed_context)
                    if self._observed_context is not None
                    else AuthoringWorkflowStage.DEFINITIONS_READY
                )
                self._record_terminal(
                    "model_edit",
                    "succeeded",
                    outcome.summary,
                )
        elif name == _RUN_PREFLIGHT.name:
            if outcome.data.get("passed") is True:
                self._stage = AuthoringWorkflowStage.SOLVE_READY
                self._record_terminal("preflight", "passed", outcome.summary)
            elif outcome.data.get("state") == "running":
                self._stage = AuthoringWorkflowStage.PREFLIGHT_PENDING
                self._pending_operation = "preflight"
            else:
                raise ValueError("preflight did not pass or start")
        elif name == _PREPARE_SOLVE.name:
            self._stage = AuthoringWorkflowStage.SOLVE_PENDING
            self._pending_operation = "solve"
        elif name == _REQUEST_PROJECT_SAVE.name:
            if self._project_save_record is None:
                raise ValueError("project save handler registered no proposal")
            self._stage = AuthoringWorkflowStage.PROJECT_SAVE_PENDING
            self._pending_operation = "project_save"
        elif name == _PREPARE_DELETE.name:
            object_type = outcome.data.get("delete_object_type")
            if type(object_type) is not str or object_type not in {
                "part",
                "feature",
                "generated_mesh",
                "named_region",
                "analysis_step",
                "boundary_condition",
                "load",
                "result_request",
            }:
                raise ValueError("destructive handler registered no exact target type")
            self._destructive_resume_stage = self._stage
            self._pending_destructive_object_type = object_type
            self._stage = AuthoringWorkflowStage.DESTRUCTIVE_EDIT_PENDING
            self._pending_operation = "destructive_edit"
        self._refresh_turn_snapshot_locked()

    def _project_save_available(
        self,
        context: AuthoringContext | None = None,
    ) -> bool:
        if (
            self._stage not in _PROJECT_SAVE_READY_STAGES
            or _REQUEST_PROJECT_SAVE.name not in self._handlers
        ):
            return False
        if context is None:
            try:
                raw = self._context_reader()
            except Exception:
                return False
            if type(raw) is not AuthoringContext:
                return False
            context = raw
        binding = context.binding
        return bool(
            binding.supported
            and binding.source_kind == "native"
            and _capability_enabled(context, "request_project_save")
        )

    def _destructive_edit_available(
        self,
        context: AuthoringContext | None = None,
    ) -> bool:
        if (
            self._stage not in _DESTRUCTIVE_EDIT_READY_STAGES
            or not _DESTRUCTIVE_EDIT_TOOLS.issubset(self._handlers)
        ):
            return False
        if context is None:
            try:
                raw = self._context_reader()
            except Exception:
                return False
            if type(raw) is not AuthoringContext:
                return False
            context = raw
        binding = context.binding
        return bool(
            binding.supported
            and binding.source_kind == "native"
            and _capability_enabled(context, "delete_model_objects")
        )

    def _model_edit_available(
        self,
        context: AuthoringContext | None = None,
    ) -> bool:
        if (
            self._stage not in _DESTRUCTIVE_EDIT_READY_STAGES
            or not _MODEL_EDIT_TOOLS.issubset(self._handlers)
        ):
            return False
        if context is None:
            try:
                raw = self._context_reader()
            except Exception:
                return False
            if type(raw) is not AuthoringContext:
                return False
            context = raw
        binding = context.binding
        return bool(
            binding.supported
            and binding.source_kind == "native"
            and _capability_enabled(context, "edit_model_objects")
        )

    def _geometry_edit_available(
        self,
        context: AuthoringContext | None = None,
    ) -> bool:
        if (
            self._stage not in _PROJECT_SAVE_READY_STAGES
            or not _GEOMETRY_EDIT_TOOLS.issubset(self._handlers)
        ):
            return False
        if context is None:
            try:
                raw = self._context_reader()
            except Exception:
                return False
            if type(raw) is not AuthoringContext:
                return False
            context = raw
        binding = context.binding
        return bool(
            binding.supported
            and binding.source_kind == "native"
            and bool(context.parts)
            and _capability_enabled(context, "edit_native_geometry")
        )

    def _geometry_catalog_available(
        self,
        context: AuthoringContext | None = None,
    ) -> bool:
        if (
            self._stage not in _PROJECT_SAVE_READY_STAGES
            or not _GEOMETRY_CATALOG_TOOLS.issubset(self._handlers)
        ):
            return False
        if context is None:
            try:
                raw = self._context_reader()
            except Exception:
                return False
            if type(raw) is not AuthoringContext:
                return False
            context = raw
        binding = context.binding
        return bool(
            binding.supported
            and binding.source_kind == "native"
            and any(not part.suppressed for part in context.parts)
            and _capability_enabled(context, "read_geometry_feature_catalog")
        )

    def _profile_transform_available(
        self,
        context: AuthoringContext | None = None,
    ) -> bool:
        if (
            self._stage not in _PROJECT_SAVE_READY_STAGES
            or not _PROFILE_TRANSFORM_TOOLS.issubset(self._handlers)
        ):
            return False
        if context is None:
            try:
                raw = self._context_reader()
            except Exception:
                return False
            if type(raw) is not AuthoringContext:
                return False
            context = raw
        binding = context.binding
        return bool(
            binding.supported
            and binding.source_kind == "native"
            and any(not part.suppressed for part in context.parts)
            and _capability_enabled(context, "edit_native_geometry")
        )

    def _current_mesh_available(self) -> bool:
        context = self._observed_context
        return bool(
            type(context) is AuthoringContext
            and context.binding.supported
            and context.binding.source_kind == "native"
            and context.mesh.present
            and context.mesh.current
            and _APPLY_DEFINITION.name in self._handlers
        )

    def _clear_destructive_pending(self) -> None:
        self._destructive_resume_stage = None
        self._pending_destructive_object_type = None

    def _mark_project_save_terminal(
        self,
        state: ProposalState,
        message: str,
    ) -> None:
        record = self._project_save_record
        if record is not None:
            self._project_save_record = replace(
                record,
                state=state,
                message=str(message)[:512],
            )

    def _record_terminal(
        self,
        operation: str,
        state: str,
        message: str,
    ) -> None:
        self._terminals.append(
            AuthoringTerminalRecord(
                str(operation),
                str(state),
                str(message)[:512],
            )
        )
        self._terminals = self._terminals[-64:]

    @staticmethod
    def _failure(
        context: ToolExecutionContext,
        code: DiagnosticCode,
        message: str,
    ) -> ToolResult:
        return ToolResult(
            ok=False,
            session_id=context.session_id,
            input_revision=max(context.expected_revision, 0),
            idempotency_key=context.idempotency_key,
            summary=message,
            diagnostics=(
                make_diagnostic(
                    code,
                    message,
                    source="agent.authoring_runtime",
                ),
            ),
        )


def provider_safe_authoring_payload(
    value: Mapping[str, object],
) -> dict[str, object]:
    """Validate and copy one bounded JSON object before Provider exposure."""

    if not isinstance(value, Mapping):
        raise TypeError("authoring payload must be an object")
    normalized = _safe_json_value(dict(value), depth=0)
    assert isinstance(normalized, dict)
    encoded = json.dumps(
        normalized,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    if len(encoded) > _MAX_PROVIDER_PAYLOAD_BYTES:
        raise ValueError("authoring payload exceeds the provider-safe byte budget")
    return normalized


def _restored_stage_for_context(
    context: AuthoringContext,
) -> AuthoringWorkflowStage:
    """Resume one accepted native project from its authoritative local state."""

    binding = context.binding
    if not binding.supported:
        return AuthoringWorkflowStage.STALE
    if binding.source_kind != "native" or not context.parts:
        return AuthoringWorkflowStage.REQUIREMENTS
    if _job_status_is_active(context.job_status):
        return AuthoringWorkflowStage.SOLVE_PENDING
    if context.result_available:
        return AuthoringWorkflowStage.RESULTS_READY
    if (
        context.mesh.current
        and context.validation_status == "passed"
        and context.definitions.analysis_step_count > 0
    ):
        return AuthoringWorkflowStage.SOLVE_READY
    if context.mesh.current and context.definitions.analysis_step_count > 0:
        return AuthoringWorkflowStage.PREFLIGHT_READY
    if context.mesh.current:
        return AuthoringWorkflowStage.DEFINITIONS_READY
    return AuthoringWorkflowStage.MESH_READY


def _job_status_is_active(value: str) -> bool:
    return value.casefold() in {
        "running",
        "queued",
        "pending",
        "cancelling",
        "canceling",
    }


def _provider_safe_summary(value: str) -> str:
    normalized = value.strip()
    encoded = normalized.encode("utf-8")
    if len(encoded) > 2048:
        raise ValueError("authoring summary exceeds the provider-safe byte budget")
    if _ABSOLUTE_PATH.search(normalized):
        raise ValueError("absolute paths are local-only")
    lowered = normalized.casefold()
    if any(
        marker in lowered
        for marker in ("api_key", "apikey", "password", "secret", "credential")
    ):
        raise ValueError("credential material is local-only")
    return normalized


def _safe_json_value(value: object, *, depth: int) -> object:
    if depth > _MAX_PROVIDER_DEPTH:
        raise ValueError("authoring payload exceeds the nesting budget")
    if isinstance(value, Mapping):
        if len(value) > _MAX_PROVIDER_COLLECTION_ITEMS:
            raise ValueError("authoring payload object exceeds the item budget")
        result: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("authoring payload keys must be strings")
            if key.casefold() in _FORBIDDEN_PAYLOAD_KEYS:
                raise ValueError(f"authoring payload field {key!r} is local-only")
            result[key] = _safe_json_value(item, depth=depth + 1)
        return result
    if isinstance(value, (list, tuple)):
        if len(value) > _MAX_PROVIDER_COLLECTION_ITEMS:
            raise ValueError("authoring payload array exceeds the item budget")
        return [_safe_json_value(item, depth=depth + 1) for item in value]
    if isinstance(value, str):
        if len(value) > 4096:
            raise ValueError("authoring payload text exceeds the item budget")
        if _ABSOLUTE_PATH.search(value):
            raise ValueError("absolute paths are local-only")
        return value
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise TypeError("authoring payload must contain finite JSON values only")


def _require_exact_fields(
    value: Mapping[str, Any],
    required: set[str],
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("tool arguments must be an object")
    data = dict(value)
    if set(data) != required:
        raise ValueError("tool argument fields do not match the strict schema")
    return data


def _nonblank_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"{label} must be a non-blank string")
    return value.strip()


def _validate_requirement_value(
    key: str,
    value: object,
    spec: Mapping[str, object],
) -> None:
    expected = spec["type"]
    if expected == "string":
        normalized = _nonblank_string(value, key)
        if "enum" in spec and normalized not in spec["enum"]:
            raise ValueError(f"{key} is outside the supported values")
    elif expected == "number":
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise TypeError(f"{key} must be a finite number")
        numeric = float(value)
        if "exclusiveMinimum" in spec and numeric <= float(spec["exclusiveMinimum"]):
            raise ValueError(f"{key} is below its strict lower bound")
        if "exclusiveMaximum" in spec and numeric >= float(spec["exclusiveMaximum"]):
            raise ValueError(f"{key} is above its strict upper bound")
    elif expected == "integer":
        if isinstance(value, bool) or type(value) is not int:
            raise TypeError(f"{key} must be an integer")
        if "minimum" in spec and value < int(spec["minimum"]):
            raise ValueError(f"{key} is below its lower bound")
        if "maximum" in spec and value > int(spec["maximum"]):
            raise ValueError(f"{key} is above its upper bound")
        if "enum" in spec and value not in spec["enum"]:
            raise ValueError(f"{key} is outside the supported values")
    elif expected == "boolean":
        if type(value) is not bool:
            raise TypeError(f"{key} must be a boolean")
        if "enum" in spec and value not in spec["enum"]:
            raise ValueError(f"{key} is outside the supported values")
    elif expected == "array":
        if not isinstance(value, list):
            raise TypeError(f"{key} must be an array")
        if not int(spec["minItems"]) <= len(value) <= int(spec["maxItems"]):
            raise ValueError(f"{key} has an unsupported number of values")
        if len({json.dumps(item, sort_keys=True) for item in value}) != len(value):
            raise ValueError(f"{key} must contain unique values")
        item_spec = spec["items"]
        assert isinstance(item_spec, Mapping)
        for item in value:
            _validate_requirement_value(key, item, item_spec)
    else:
        raise RuntimeError("unsupported requirement schema")


def _validate_supported_requirement_combination(
    values: Mapping[str, object],
    *,
    dimension: int | None = None,
) -> None:
    active_dimension = 3 if dimension == 3 else 2
    fixed_dofs = tuple(values["fixed_dofs"])
    if fixed_dofs != tuple(range(min(fixed_dofs), max(fixed_dofs) + 1)):
        raise ValueError(
            "clarification_required: fixed_dofs must be one contiguous "
            f"{active_dimension}D translation range"
        )
    load_type = str(values["load_type"])
    direction = str(values["load_direction"])
    magnitude = float(values["load_magnitude"])
    if load_type == "nodal":
        if direction not in ({"x", "y", "z"} if active_dimension == 3 else {"x", "y"}):
            raise ValueError(
                "clarification_required: nodal force requires a global "
                "translation direction"
            )
        return
    if load_type.endswith("traction"):
        if direction not in ({"x", "y", "z"} if active_dimension == 3 else {"x", "y"}):
            raise ValueError(
                "clarification_required: traction requires an explicit global "
                "translation direction"
            )
        return
    if not load_type.endswith("pressure") or direction not in {
        "inward_normal",
        "outward_normal",
    }:
        raise ValueError("clarification_required: pressure requires a normal direction")
    if (direction == "inward_normal" and magnitude <= 0.0) or (
        direction == "outward_normal" and magnitude >= 0.0
    ):
        raise ValueError(
            "clarification_required: pressure sign must match its normal direction"
        )


def _requirement_stage(key: str) -> str:
    if key == "line_element_type":
        return "mesh"
    for stage, keys in _REQUIREMENT_GROUPS.items():
        if key in keys:
            return stage
    raise ValueError("unknown requirement field")


def _capability_enabled(
    context: AuthoringContext,
    operation: str,
) -> bool:
    return any(
        item.operation == operation and item.enabled for item in context.capabilities
    )


__all__ = [
    "AUTHORING_TURN_SNAPSHOT_MAX_BYTES",
    "AUTHORING_TURN_SNAPSHOT_SCHEMA_VERSION",
    "AuthoringTerminalRecord",
    "AuthoringTurnSnapshot",
    "AuthoringToolHandler",
    "AuthoringToolOutcome",
    "AuthoringWorkflowController",
    "AuthoringWorkflowStage",
    "ProjectSaveProposalRecord",
    "provider_safe_authoring_payload",
]
