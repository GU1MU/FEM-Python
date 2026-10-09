"""Geometry creation, scoped editing and profile-transform tool contracts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

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

from ..providers.base import ToolDefinition
from .common import _tool


_PATH_STROKE_OBJECT_POINT_DESCRIPTION = (
    'Ordered open centerline points as {"x": number, "y": number} objects. '
    "Multiple bends are supported; the first and last points differ."
)
_PATH_STROKE_ARRAY_POINT_DESCRIPTION = (
    "Ordered open centerline points as [x, y] pairs. Multiple bends are "
    "supported; the first and last points differ."
)


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
        "Prepare a revision-bound geometry proposal for a named spatial wire "
        "or a simple solid primitive. Planar regions and planar-derived 3D "
        "outputs use prepare_planar_construction_proposal. Applying the "
        "proposal requires local GUI confirmation."
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
        "Read the current editable native Part, supported operations, geometry "
        "IDs, and exact Profile/hole counts for planar sketches."
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
_GEOMETRY_EDIT_PARAMETERS = {
    "type": "object",
    "properties": {
        "part_id": {"type": "string", "minLength": 1, "maxLength": 128},
        "edit": {
            "oneOf": [
                {
                    "type": "object",
                    "properties": {
                        "operation": {"const": "planar_boolean"},
                        "boolean_operation": {
                            "type": "string",
                            "enum": ["fuse", "cut"],
                        },
                        "tool": {
                            "description": (
                                "Cut/fuse profile as a primitive region, "
                                "closed polygon boundary, or open non- "
                                "branching constant-width path_stroke."
                            ),
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
                                            "description": (
                                                "Ordered vertices of one closed "
                                                "perimeter. Concave boundaries "
                                                "are valid."
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
                                            "description": (
                                                "Open non-branching centerline "
                                                "for one constant-width slot. "
                                                + _PATH_STROKE_OBJECT_POINT_DESCRIPTION
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
                                            "description": (
                                                "Full constant slot width."
                                            ),
                                        },
                                        "cap": {
                                            "type": "string",
                                            "enum": ["butt", "square", "round"],
                                            "description": (
                                                "Boundary style at both path ends."
                                            ),
                                        },
                                        "join": {
                                            "type": "string",
                                            "enum": ["miter", "bevel", "round"],
                                            "description": (
                                                "Boundary style at path bends."
                                            ),
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
                                "Ordered non-self-intersecting perimeter. The "
                                "closing edge is implicit; concave boundaries "
                                "are valid."
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
                                "Open non-branching centerline for one "
                                "constant-width slot; the compiler generates "
                                "its offset boundary. "
                                + _PATH_STROKE_OBJECT_POINT_DESCRIPTION
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
}
_geometry_edit_schema = _GEOMETRY_EDIT_PARAMETERS["properties"]["edit"]
_PLANAR_BOOLEAN_EDIT_SCHEMA = next(
    branch
    for branch in _geometry_edit_schema["oneOf"]
    if branch.get("properties", {}).get("operation", {}).get("const")
    == "planar_boolean"
)
_REPLACE_PLANAR_BOOLEAN_FEATURE_EDIT_SCHEMA = {
    "type": "object",
    "properties": {
        "operation": {"const": "replace_planar_boolean_feature"},
        "feature_id": {
            "type": "string",
            "pattern": "^PB[1-9][0-9]*$",
            "maxLength": 128,
            "description": (
                "Existing planar Boolean feature_id from "
                "read_geometry_edit_context."
            ),
        },
        "tool": _PLANAR_BOOLEAN_EDIT_SCHEMA["properties"]["tool"],
    },
    "required": ["operation", "feature_id", "tool"],
    "additionalProperties": False,
}
_geometry_edit_schema["oneOf"].append(_REPLACE_PLANAR_BOOLEAN_FEATURE_EDIT_SCHEMA)
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
_geometry_edit_schema["oneOf"].extend(
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
    for branch in _geometry_edit_schema["oneOf"]
    if branch.get("properties", {}).get("operation", {}).get("const")
    in _BATCH_PLANAR_OPERATION_CONSTS
]
_geometry_edit_schema["oneOf"].append(
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
_GEOMETRY_EDIT_PARAMETERS["properties"]["spatial_relation"] = _SPATIAL_RELATION_SCHEMA
_PREPARE_GEOMETRY_EDIT = ToolDefinition(
    "prepare_geometry_edit",
    (
        "Prepare a revision-bound native Part edit using the current geometry IDs "
        "and narrowed operation schema. Dedicated prepare_profile_* tools handle "
        "Profile transforms. Feature tool recipes are read-only snapshots; "
        "replace_planar_boolean_feature replaces their boundary. A planar cutout "
        "in a strict sketch is a contained closed inner Profile. Invalid geometry "
        "returns topology diagnostics. Optional spatial_relation specifies a "
        "reference feature_id, direction, and exact clearance checked by local "
        "preflight."
    ),
    _GEOMETRY_EDIT_PARAMETERS,
)


def _geometry_edit_operation(branch: Mapping[str, object]) -> str | None:
    properties = branch.get("properties")
    if not isinstance(properties, Mapping):
        return None
    operation = properties.get("operation")
    if not isinstance(operation, Mapping):
        return None
    value = operation.get("const")
    return value if isinstance(value, str) else None


def _geometry_edit_tool_definition(
    part_id: str,
    recipe_kind: str,
    supported_operations: Sequence[str],
) -> ToolDefinition:
    """Return one Part-scoped definition containing only executable branches."""

    allowed = set(str(item) for item in supported_operations)
    edit_schema = _PREPARE_GEOMETRY_EDIT.parameters["properties"]["edit"]
    branches = [
        branch
        for branch in edit_schema["oneOf"]
        if _geometry_edit_operation(branch) in allowed
    ]
    if not branches:
        return _PREPARE_GEOMETRY_EDIT
    properties = dict(_PREPARE_GEOMETRY_EDIT.parameters["properties"])
    properties["part_id"] = {"const": str(part_id)}
    properties["edit"] = {"oneOf": branches}
    parameters = dict(_PREPARE_GEOMETRY_EDIT.parameters)
    parameters["properties"] = properties
    operations = ", ".join(
        operation
        for branch in branches
        if (operation := _geometry_edit_operation(branch)) is not None
    )
    return ToolDefinition(
        _PREPARE_GEOMETRY_EDIT.name,
        (
            f"Part-scoped geometry edit for {part_id} ({recipe_kind}). "
            f"Allowed operations after the latest local read: {operations}. "
            "Feature tool snapshots and their local point/curve IDs are read-only."
        ),
        parameters,
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
                "description": (
                    "Ordered perimeter vertices with an implicit closing edge. "
                    "Concave boundaries are valid; the first vertex is not "
                    "repeated at the end."
                ),
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
                    "Open non-branching centerline for one constant-width slot. "
                    "The compiler generates the offset boundary using cap and "
                    "join. "
                    + _PATH_STROKE_ARRAY_POINT_DESCRIPTION
                ),
            },
            "width": {
                "type": "number",
                "exclusiveMinimum": 0,
                "description": "Full constant width along the entire centerline.",
            },
            "cap": {
                "type": "string",
                "enum": ["butt", "square", "round"],
                "description": "Boundary style at the two open centerline ends.",
            },
            "join": {
                "type": "string",
                "enum": ["miter", "bevel", "round"],
                "description": "Boundary style at each centerline bend.",
            },
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
        "Compile a bounded Planar Construction IR v1 graph on the global XY "
        "plane, validate its Profiles and materialized sketch, and present a "
        "revision-bound planar or derived 3D Part proposal. Separate "
        "subtraction operands produce separate native Cut features. Planar "
        "output accepts 'planar' or {'kind': 'planar'}. Applying the proposal "
        "requires local GUI confirmation."
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
                        "description": "Construction nodes with unique IDs and explicit operand references.",
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
