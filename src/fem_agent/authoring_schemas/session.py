"""Authoring context, engineering requirements and document tools."""

from __future__ import annotations

from collections.abc import Mapping

from ..providers.base import ToolDefinition
from .common import _NO_ARGUMENTS, _tool


_REQUIREMENT_SPECS: dict[str, dict[str, object]] = {
    "length_unit": {"type": "string"},
    "force_unit": {"type": "string"},
    "stress_unit": {"type": "string"},
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
}


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
        "Complete geometry or mesh requirements enable the corresponding "
        "operation confirmation proposal."
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


_REQUEST_PROJECT_SAVE = _tool(
    "request_project_save",
    (
        "Present a revision-bound local project-save card. This tool has no "
        "path argument and cannot write a file; only its GUI button may start "
        "the existing native project save command."
    ),
    _NO_ARGUMENTS,
)


_CREATE_NATIVE_MODEL_DOCUMENT = _tool(
    "create_native_model_document",
    (
        "Create and activate an additional blank native model document while "
        "preserving existing workspace documents. This creates a separate document "
        "rather than a Part in the current model."
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
