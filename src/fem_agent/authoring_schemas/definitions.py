"""Model topology, engineering definitions and object-editing tool contracts."""

from __future__ import annotations

from collections.abc import Mapping
import re

from ..definition_rules import NODAL_LOAD_DIRECTIONS
from .common import _NO_ARGUMENTS, _tool


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
        "description": (
            "Use " + ", ".join(prefixes)
            + " prefixes for new names; preserve existing names."
        ),
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
_STEP_NAME_SCHEMA = _controlled_name_schema("Step")
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
        }
    )


def _boundary_parameters(
    target_kind: str,
    scope_prefix: str,
) -> dict[str, object]:
    return _exact_schema(
        {
            "name": _controlled_name_schema("Displacement"),
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
            "name": _controlled_name_schema("Load"),
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
        scope_prefix="Point",
        component={"type": "integer", "const": component},
        vector=_NULL_SCHEMA,
        magnitude=_NUMBER_SCHEMA,
        direction=direction,
        distribution="concentrated",
    )
    for component, direction in NODAL_LOAD_DIRECTIONS.items()
] + [
    _load_parameters(
        entity_type="edge",
        load_type="edge_traction",
        scope_prefix="Edge",
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
        scope_prefix="Edge",
        component=_NULL_SCHEMA,
        vector=_NULL_SCHEMA,
        magnitude=_NUMBER_SCHEMA,
        direction="inward_normal",
        distribution="uniform",
    ),
    _load_parameters(
        entity_type="edge",
        load_type="edge_pressure",
        scope_prefix="Edge",
        component=_NULL_SCHEMA,
        vector=_NULL_SCHEMA,
        magnitude=_NUMBER_SCHEMA,
        direction="outward_normal",
        distribution="uniform",
    ),
    _load_parameters(
        entity_type="surface",
        load_type="surface_traction",
        scope_prefix="Face",
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
        scope_prefix="Face",
        component=_NULL_SCHEMA,
        vector=_NULL_SCHEMA,
        magnitude=_NUMBER_SCHEMA,
        direction="inward_normal",
        distribution="uniform",
    ),
    _load_parameters(
        entity_type="surface",
        load_type="surface_pressure",
        scope_prefix="Face",
        component=_NULL_SCHEMA,
        vector=_NULL_SCHEMA,
        magnitude=_NUMBER_SCHEMA,
        direction="outward_normal",
        distribution="uniform",
    ),
    _exact_schema(
        {
            "name": _controlled_name_schema("Load"),
            "step_name": _STEP_NAME_SCHEMA,
            "target_scope": _controlled_name_schema("Domain"),
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
            "name": _controlled_name_schema("Load"),
            "step_name": _STEP_NAME_SCHEMA,
            "target_scope": _controlled_name_schema("Domain"),
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
            "name": _controlled_name_schema("Load"),
            "step_name": _STEP_NAME_SCHEMA,
            "target_scope": {
                "oneOf": [
                    {"type": "null"},
                    _controlled_name_schema("Domain"),
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
                        _named_region_parameters("node", "Point"),
                        _named_region_parameters("edge", "Edge"),
                        _named_region_parameters("face", "Face"),
                        _named_region_parameters("element", "Domain"),
                    ]
                ),
            ),
            _definition_action_schema(
                "create_material",
                _exact_schema(
                    {
                        "name": _controlled_name_schema("Material"),
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
                                "name": _controlled_name_schema("Section"),
                                "material": _controlled_name_schema("Material"),
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
                                "name": _controlled_name_schema("Section"),
                                "material": _controlled_name_schema("Material"),
                                "section_type": {
                                    "type": "string",
                                    "const": "solid",
                                },
                                "properties": _exact_schema({}),
                            }
                        ),
                        _exact_schema(
                            {
                                "name": _controlled_name_schema("Section"),
                                "material": _controlled_name_schema("Material"),
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
                                "name": _controlled_name_schema("Section"),
                                "material": _controlled_name_schema("Material"),
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
                                                "Rectangular section height along beam local z (Abaqus n2)"
                                            ),
                                        },
                                        "width": {
                                            "type": "number",
                                            "exclusiveMinimum": 0,
                                            "description": (
                                                "Rectangular section width along beam local y (Abaqus n1)"
                                            ),
                                        },
                                    }
                                ),
                            }
                        ),
                        _exact_schema(
                            {
                                "name": _controlled_name_schema("Section"),
                                "material": _controlled_name_schema("Material"),
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
                                "name": _controlled_name_schema("Section"),
                                "material": _controlled_name_schema("Material"),
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
                                "section_name": _controlled_name_schema("Section"),
                                "region_name": _controlled_name_schema("Domain"),
                            }
                        ),
                        _exact_schema(
                            {
                                "section_name": _controlled_name_schema("Section"),
                                "region_name": _controlled_name_schema("Domain"),
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
                        _boundary_parameters("node_set", "Point"),
                        _boundary_parameters("edge", "Edge"),
                        _boundary_parameters("surface", "Face"),
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
                                "name": _controlled_name_schema("Output"),
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
                                "name": _controlled_name_schema("Output"),
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
                        "maximum": 6,
                    },
                    "last_component": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 6,
                    },
                    "component": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 6,
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
                            *NODAL_LOAD_DIRECTIONS.values(),
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
