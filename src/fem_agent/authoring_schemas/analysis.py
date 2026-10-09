"""Mesh preparation, analysis preflight and solve tool contracts."""

from __future__ import annotations

from .common import _NO_ARGUMENTS, _tool


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


_RUN_PREFLIGHT = _tool(
    "run_native_preflight",
    (
        "Run the existing deterministic native preflight without solving. "
        "The call waits locally until the background GUI check reports its "
        "terminal state, returned in the state field; running is only "
        "returned when the local wait budget elapsed."
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
