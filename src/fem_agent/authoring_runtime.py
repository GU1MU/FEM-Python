"""A8 dynamic authoring workflow and provider-safe tool boundary."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from enum import Enum
import hashlib
import json
import math
import re
import threading
from typing import TYPE_CHECKING, Any
import uuid

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
    RESULT_QUERY_TOOL_NAME,
    result_catalog_tool_schema,
    result_query_tool_schema,
)
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
class AuthoringTerminalRecord:
    operation: str
    state: str
    message: str


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
        "enum": ["line", "triangle", "quadrilateral"],
    },
    "mesh_order": {"type": "integer", "enum": [1, 2]},
    "mesh_global_size": {"type": "number", "exclusiveMinimum": 0},
    "line_element_type": {
        "type": "string",
        "enum": ["Truss2", "Beam2"],
    },
    "fixed_dofs": {
        "type": "array",
        "items": {"type": "integer", "minimum": 1, "maximum": 2},
        "minItems": 1,
        "maxItems": 2,
        "uniqueItems": True,
    },
    "load_type": {
        "type": "string",
        "enum": ["edge_traction", "edge_pressure"],
    },
    "load_direction": {
        "type": "string",
        "enum": ["x", "y", "inward_normal", "outward_normal"],
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
_GATED_OPERATION_BY_STAGE = {
    AuthoringWorkflowStage.REQUIREMENTS: "prepare_geometry_proposal",
    AuthoringWorkflowStage.GEOMETRY_READY: "prepare_geometry_proposal",
    AuthoringWorkflowStage.MESH_READY: "prepare_mesh_proposal",
    AuthoringWorkflowStage.DEFINITIONS_READY: "prepare_mesh_proposal",
    AuthoringWorkflowStage.ANALYSIS_DEFINITIONS_READY: "prepare_mesh_proposal",
    AuthoringWorkflowStage.PREFLIGHT_READY: "prepare_mesh_proposal",
    AuthoringWorkflowStage.SOLVE_READY: "prepare_mesh_proposal",
    AuthoringWorkflowStage.RESULTS_READY: "prepare_mesh_proposal",
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
_PREPARE_GEOMETRY = _tool(
    "prepare_geometry_proposal",
    (
        "Build and present a revision-bound geometry proposal from general "
        "planar profiles, a named spatial wire, or a supported solid primitive. "
        "Record the project "
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
                            "profiles": {
                                "type": "array",
                                "minItems": 1,
                                "maxItems": 32,
                                "items": {
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
                                                            "x": {
                                                                "type": "number"
                                                            },
                                                            "y": {
                                                                "type": "number"
                                                            },
                                                        },
                                                        "required": ["x", "y"],
                                                        "additionalProperties": (
                                                            False
                                                        ),
                                                    },
                                                },
                                            },
                                            "required": ["kind", "vertices"],
                                            "additionalProperties": False,
                                        },
                                    ]
                                },
                            },
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
                ]
            },
        },
        "required": ["part_function", "geometry"],
        "additionalProperties": False,
    },
)
_READ_GEOMETRY_EDIT_CONTEXT = _tool(
    "read_geometry_edit_context",
    (
        "Read a bounded editable projection of one existing native Part. "
        "Use this before changing an accepted geometry."
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
        "selected material Profile. The edit runs only after the local GUI "
        "control is clicked."
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
                        "required": [
                            "operation", "source_face_ids", "height"
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
]


_APPLY_DEFINITION = _tool(
    "apply_model_definition",
    (
        "Immediately apply one supported scope, material, section, assignment, "
        "analysis-step, boundary-condition, load, or result-request action and "
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
                                        },
                                        "width": {
                                            "type": "number",
                                            "exclusiveMinimum": 0,
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
                                        "enum": ["U", "RF"],
                                    },
                                    "minItems": 1,
                                    "maxItems": 2,
                                    "uniqueItems": True,
                                },
                                "units": {
                                    "type": "array",
                                    "items": _UNIT_SCHEMA,
                                    "minItems": 1,
                                    "maxItems": 2,
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
                                        "const": "S",
                                    },
                                    "minItems": 1,
                                    "maxItems": 1,
                                },
                                "units": {
                                    "type": "array",
                                    "items": _UNIT_SCHEMA,
                                    "minItems": 1,
                                    "maxItems": 1,
                                },
                                "confirmed": _CONFIRMED_SCHEMA,
                            }
                        ),
                    ]
                ),
            ),
        ]
    },
)
_RUN_PREFLIGHT = _tool(
    "run_native_preflight",
    "Run the existing deterministic native preflight without solving.",
    _NO_ARGUMENTS,
)
_PREPARE_SOLVE = _tool(
    "prepare_solve_proposal",
    (
        "Build and present a revision/stamp-bound solve proposal. The solver "
        "is not started until the GUI control is clicked."
    ),
    _NO_ARGUMENTS,
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
        "Read current named scopes, boundary conditions, and loads with their "
        "bounded editable fields and stable identities."
    ),
    _NO_ARGUMENTS,
)
_EDIT_MODEL_OBJECT = _tool(
    "edit_model_object",
    (
        "Immediately edit one exact object returned by "
        "read_editable_model_objects and synchronize the GUI. An edit that "
        "invalidates accepted results creates a confirmation card."
    ),
    {
        "type": "object",
        "properties": {
            "object_type": {
                "type": "string",
                "enum": [
                    "named_region",
                    "boundary_condition",
                    "load",
                ],
            },
            "target_id": {
                "type": "string",
                "minLength": 1,
                "maxLength": 96,
            },
            "step_name": {
                "type": "string",
                "minLength": 1,
                "maxLength": 96,
            },
            "changes": {
                "type": "object",
                "properties": {
                    "new_name": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 96,
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
                            "inward_normal",
                            "outward_normal",
                        ],
                    },
                    "coordinate_system": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 64,
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
_GEOMETRY_FEATURE_CATALOG = _result_tool_definition(
    geometry_feature_catalog_tool_schema()
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
_MODEL_EDIT_TOOLS = frozenset(
    {_READ_EDITABLE_OBJECTS.name, _EDIT_MODEL_OBJECT.name}
)
_GEOMETRY_EDIT_TOOLS = frozenset(
    {
        _READ_GEOMETRY_EDIT_CONTEXT.name,
        _PREPARE_GEOMETRY_EDIT.name,
    }
)
_GEOMETRY_CATALOG_TOOLS = frozenset({_GEOMETRY_FEATURE_CATALOG.name})


_STAGE_TOOLS: dict[AuthoringWorkflowStage, tuple[ToolDefinition, ...]] = {
    AuthoringWorkflowStage.REQUIREMENTS: (
        _READ_CONTEXT,
        _GEOMETRY_FEATURE_CATALOG,
        _SET_REQUIREMENTS,
        _PREPARE_GEOMETRY,
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
        _RESULT_CATALOG,
        _RESULT_QUERY,
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
                    "properties": {
                        key: specs[key] for key in keys
                    },
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
        self._active_tool_context: ToolExecutionContext | None = None
        self._binding_identity: tuple[str, str, int] | None = None
        self._observed_context: AuthoringContext | None = None
        self._lock = threading.RLock()

    @property
    def stage(self) -> AuthoringWorkflowStage:
        with self._lock:
            return self._stage

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
    def project_save_record(self) -> ProjectSaveProposalRecord | None:
        with self._lock:
            return self._project_save_record

    @property
    def definitions(self) -> tuple[ToolDefinition, ...]:
        with self._lock:
            definitions = _STAGE_TOOLS[self._stage]
            requirement_group = self._current_requirement_group()
            requirements_complete = (
                requirement_group is not None
                and self._requirement_group_complete(requirement_group)
            )
            gated_operation = _GATED_OPERATION_BY_STAGE.get(self._stage)
            visible: list[ToolDefinition] = []
            for item in definitions:
                if (
                    gated_operation is not None
                    and item.name == gated_operation
                    and not requirements_complete
                ):
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
                        {
                            key: self._requirement_spec(key)
                            for key in keys
                        },
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
                )
            )

    def dispatch(
        self,
        name: str,
        arguments: Mapping[str, Any],
        context: ToolExecutionContext,
    ) -> ToolResult:
        with self._lock:
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
                        _PREPARE_DELETE.name,
                        _PREPARE_GEOMETRY.name,
                        _READ_GEOMETRY_EDIT_CONTEXT.name,
                        _PREPARE_GEOMETRY_EDIT.name,
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
                error_type = re.sub(
                    r"[^A-Za-z0-9_]",
                    "",
                    type(error).__name__,
                )[:64] or "LocalAuthoringError"
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
                self._stage = _restored_stage_for_context(context)
                self._seed_default_geometry_requirements(context)
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
                    or (
                        same_session
                        and saved_state_transition
                    )
                    or (
                        same_session
                        and project_save_terminal_transition
                        and self._pending_operation == "project_save"
                    )
                )
            )
            self._binding_identity = current
            if expected_local_transition:
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
                    else (
                        self._mesh_resume_stage
                        or AuthoringWorkflowStage.MESH_READY
                    )
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
                    self._stage = {
                        "part": AuthoringWorkflowStage.GEOMETRY_READY,
                        "generated_mesh": AuthoringWorkflowStage.MESH_READY,
                        "named_region": AuthoringWorkflowStage.DEFINITIONS_READY,
                        "analysis_step": (
                            AuthoringWorkflowStage.ANALYSIS_DEFINITIONS_READY
                        ),
                        "boundary_condition": (
                            AuthoringWorkflowStage.ANALYSIS_DEFINITIONS_READY
                        ),
                        "load": (
                            AuthoringWorkflowStage.ANALYSIS_DEFINITIONS_READY
                        ),
                        "result_request": (
                            AuthoringWorkflowStage.ANALYSIS_DEFINITIONS_READY
                        ),
                    }[object_type]
                else:
                    self._stage = resume_stage
                self._clear_destructive_pending()
            else:
                raise ValueError("unknown pending proposal operation")
            self._pending_operation = None
            self._record_terminal(
                normalized_operation,
                normalized_state.value,
                message,
            )

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
            self._stage = AuthoringWorkflowStage.STALE

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
            self._clear_destructive_pending()
            self._binding_identity = None
            self._observed_context = None
            self._stage = AuthoringWorkflowStage.REQUIREMENTS

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
            if (
                active is not None
                and active.state
                in {
                    ProposalState.PENDING_CONFIRMATION,
                    ProposalState.RUNNING,
                }
            ):
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
            raise ValueError(
                "clarification_required: " + ", ".join(missing)
            )
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
            and entries[key].source_turn_id
            == _DEFAULT_REQUIREMENT_SOURCE_TURN_ID
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
            return {
                "type": "string",
                "enum": ["triangle", "quadrilateral"],
            }
        if key == "mesh_order" and dimension == 1:
            return {"type": "integer", "enum": [1]}
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
        if (
            context.binding.source_kind != "blank"
            or context.unit_context is not None
            or context.parts
        ):
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
        for key, value in _DEFAULT_GEOMETRY_REQUIREMENTS.items():
            if key in recorded:
                continue
            self._ledger.record(
                key,
                field_type=str(_REQUIREMENT_SPECS[key]["type"]),
                stage=_requirement_stage(key),
                value=value,
                source_turn_id=_DEFAULT_REQUIREMENT_SOURCE_TURN_ID,
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
            raise TypeError(
                "context_reader must return AuthoringContext or an object"
            )
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
                    key
                    for key in required
                    if key not in recorded
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
                "out-of-stage requirement fields: "
                + ", ".join(sorted(out_of_stage))
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
        missing = [
            key
            for key in required
            if key not in recorded
        ]
        if missing:
            raise ValueError(
                "clarification_required: " + ", ".join(missing)
            )
        values = {item.key: item.value for item in self._ledger.entries}
        if requirement_group == "analysis":
            _validate_supported_requirement_combination(values)
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
        elif name == _PREPARE_GEOMETRY_EDIT.name:
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
                    AuthoringWorkflowStage.ANALYSIS_DEFINITIONS_READY
                    if object_type == "analysis_step"
                    else AuthoringWorkflowStage.DEFINITIONS_READY
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
                "boundary_condition",
                "load",
            }:
                raise ValueError(
                    "edit handler registered no exact edited object type"
                )
            if "proposal_id" in outcome.data:
                self._destructive_resume_stage = self._stage
                self._pending_destructive_object_type = object_type
                self._stage = AuthoringWorkflowStage.DESTRUCTIVE_EDIT_PENDING
                self._pending_operation = "destructive_edit"
            else:
                self._stage = AuthoringWorkflowStage.DEFINITIONS_READY
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
            if (
                type(object_type) is not str
                or object_type
                not in {
                    "part",
                    "generated_mesh",
                    "named_region",
                    "analysis_step",
                    "boundary_condition",
                    "load",
                    "result_request",
                }
            ):
                raise ValueError(
                    "destructive handler registered no exact target type"
                )
            self._destructive_resume_stage = self._stage
            self._pending_destructive_object_type = object_type
            self._stage = AuthoringWorkflowStage.DESTRUCTIVE_EDIT_PENDING
            self._pending_operation = "destructive_edit"

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
) -> None:
    fixed_dofs = tuple(values["fixed_dofs"])
    if fixed_dofs != tuple(range(min(fixed_dofs), max(fixed_dofs) + 1)):
        raise ValueError(
            "clarification_required: fixed_dofs must be one contiguous 2D range"
        )
    load_type = str(values["load_type"])
    direction = str(values["load_direction"])
    magnitude = float(values["load_magnitude"])
    if load_type == "edge_traction":
        if direction not in {"x", "y"}:
            raise ValueError(
                "clarification_required: edge traction direction must be x or y"
            )
        return
    if direction not in {"inward_normal", "outward_normal"}:
        raise ValueError(
            "clarification_required: edge pressure requires a normal direction"
        )
    if (
        direction == "inward_normal"
        and magnitude <= 0.0
    ) or (
        direction == "outward_normal"
        and magnitude >= 0.0
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
        item.operation == operation and item.enabled
        for item in context.capabilities
    )


__all__ = [
    "AuthoringTerminalRecord",
    "AuthoringToolHandler",
    "AuthoringToolOutcome",
    "AuthoringWorkflowController",
    "AuthoringWorkflowStage",
    "ProjectSaveProposalRecord",
    "provider_safe_authoring_payload",
]
