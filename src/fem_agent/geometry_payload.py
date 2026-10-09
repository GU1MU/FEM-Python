"""Bounded native geometry recipe payload encoding and decoding."""

from __future__ import annotations

import json
from typing import Mapping, Sequence

from fem.geometry import (
    BooleanBodyContext,
    BooleanGeometry,
    BooleanLineageEntity,
    BooleanLineageMapping,
    BoxGeometry,
    CylinderGeometry,
    DiskGeometry,
    ExtrudedGeometry,
    FaceSketchBooleanGeometry,
    MovedGeometry,
    MultiBodyGeometry,
    PlateWithHoleGeometry,
    PartBooleanContext,
    PathSweptGeometry,
    PlanarBooleanContext,
    RectangleGeometry,
    RevolvedGeometry,
    RotatedGeometry,
    SketchArc,
    SketchAngleDimension,
    SketchCircle,
    SketchCoincidentConstraint,
    SketchConcentricConstraint,
    SketchConstraint,
    SketchDistanceDimension,
    SketchEqualLengthConstraint,
    SketchEqualRadiusConstraint,
    SketchFixedConstraint,
    SketchGeometry,
    SketchHorizontalConstraint,
    SketchLine,
    SketchParallelConstraint,
    SketchPlane,
    SketchPoint,
    SketchPointOnCurveConstraint,
    SketchPerpendicularConstraint,
    SketchRadiusDimension,
    SketchTangentConstraint,
    SketchVerticalConstraint,
    SolidBody,
    WireGeometry,
    WireMember,
    WirePoint,
)

from .geometry_validation import _MAX_PREVIEW_POINTS


_RECIPE_SCHEMA_VERSION = 1
_MAX_RECIPE_BYTES = 524_288
_MAX_RECIPE_NODES = 128
_MAX_BOOLEAN_RECIPE_PAYLOAD_NODES = 4_096
_MAX_RECIPE_DEPTH = 16


def geometry_recipe_to_payload(recipe: object) -> dict[str, object]:
    payload = _geometry_recipe_to_payload(recipe)
    payload = {"schema_version": _RECIPE_SCHEMA_VERSION, **payload}
    _validate_recipe_payload_budget(payload)
    return payload


def _geometry_recipe_to_payload(recipe: object) -> dict[str, object]:
    if type(recipe) is WireGeometry:
        return {
            "kind": "wire",
            "name": recipe.name,
            "points": [
                {
                    "name": point.name,
                    "x": point.x,
                    "y": point.y,
                    "z": point.z,
                }
                for point in recipe.points
            ],
            "members": [
                {
                    "name": member.name,
                    "start": member.start,
                    "end": member.end,
                }
                for member in recipe.members
            ],
        }
    if type(recipe) is RectangleGeometry:
        return {
            "kind": "rectangle",
            "name": recipe.name,
            "width": recipe.width,
            "height": recipe.height,
        }
    if type(recipe) is DiskGeometry:
        return {"kind": "disk", "name": recipe.name, "radius": recipe.radius}
    if type(recipe) is PlateWithHoleGeometry:
        return {
            "kind": "plate_with_hole",
            "name": recipe.name,
            "width": recipe.width,
            "height": recipe.height,
            "hole_x": recipe.hole_x,
            "hole_y": recipe.hole_y,
            "hole_radius": recipe.hole_radius,
        }
    if type(recipe) is BoxGeometry:
        return {
            "kind": "box",
            "name": recipe.name,
            "width": recipe.width,
            "depth": recipe.depth,
            "height": recipe.height,
        }
    if type(recipe) is CylinderGeometry:
        return {
            "kind": "cylinder",
            "name": recipe.name,
            "radius": recipe.radius,
            "height": recipe.height,
        }
    if type(recipe) is MovedGeometry:
        return {
            "kind": "translated",
            "base": _geometry_recipe_to_payload(recipe.base),
            "dx": recipe.dx,
            "dy": recipe.dy,
            "dz": recipe.dz,
        }
    if type(recipe) is RotatedGeometry:
        return {
            "kind": "rotated",
            "base": _geometry_recipe_to_payload(recipe.base),
            "axis": recipe.axis,
            "angle_degrees": recipe.angle_degrees,
        }
    if type(recipe) is ExtrudedGeometry:
        return {
            "kind": "extruded",
            "base": _geometry_recipe_to_payload(recipe.base),
            "height": recipe.height,
            "source_face_ids": list(recipe.source_face_ids),
        }
    if type(recipe) is RevolvedGeometry:
        return {
            "kind": "revolved",
            "base": _geometry_recipe_to_payload(recipe.base),
            "axis": recipe.axis,
            "angle_degrees": recipe.angle_degrees,
            "source_face_ids": list(recipe.source_face_ids),
        }
    if type(recipe) is PathSweptGeometry:
        return {
            "kind": "path_swept",
            "base": _geometry_recipe_to_payload(recipe.base),
            "path": _geometry_recipe_to_payload(recipe.path),
            "source_face_ids": list(recipe.source_face_ids),
            "frame_strategy": recipe.frame_strategy,
        }
    if type(recipe) is BooleanGeometry:
        return {
            "kind": "boolean",
            "name": recipe.name,
            "operation": recipe.operation,
            "object": _geometry_recipe_to_payload(recipe.object_geometry),
            "tool": _geometry_recipe_to_payload(recipe.tool_geometry),
            "body_context": _boolean_context_to_payload(recipe.body_context),
            "planar_context": _boolean_context_to_payload(recipe.planar_context),
            "part_context": _boolean_context_to_payload(recipe.part_context),
        }
    if type(recipe) is FaceSketchBooleanGeometry:
        return {
            "kind": "face_sketch_boolean",
            "feature_id": recipe.feature_id,
            "name": recipe.name,
            "base": _geometry_recipe_to_payload(recipe.base),
            "support_face_id": recipe.support_face_id,
            "workplane_strategy": {
                "seed_axis": recipe.workplane_strategy.seed_axis,
                "sign": recipe.workplane_strategy.sign,
                "origin_rule": recipe.workplane_strategy.origin_rule,
            },
            "sketch": _geometry_recipe_to_payload(recipe.sketch),
            "operation": recipe.operation.value,
            "direction": recipe.direction.value,
            "distance": recipe.distance,
            "participating_profile_ids": list(recipe.participating_profile_ids),
            "external_reference_count": len(recipe.external_references),
            "step_proof_count": len(recipe.step_proofs),
            "capability": {
                "read": True,
                "create": False,
                "edit": False,
                "message": "Agent can only read face-sketch Extrude Boolean features; creation and editing are not yet supported.",
            },
        }
    if type(recipe) is MultiBodyGeometry:
        return {
            "kind": "multi_body",
            "name": recipe.name,
            "bodies": [
                {
                    "id": body.id,
                    "name": body.name,
                    "recipe": _geometry_recipe_to_payload(body.recipe),
                }
                for body in recipe.bodies
            ],
            "retired_body_ids": list(recipe.retired_body_ids),
            "retired_boolean_feature_ids": list(recipe.retired_boolean_feature_ids),
        }
    if type(recipe) is SketchGeometry and recipe.is_strict:
        if len(recipe.constraints) > 128:
            raise ValueError("planar sketch constraints exceed the bounded schema")
        return {
            "kind": "planar_sketch",
            "name": recipe.name,
            "plane": {
                "origin": list(recipe.plane.origin),
                "x_direction": list(recipe.plane.x_direction),
                "y_direction": list(recipe.plane.y_direction),
            },
            "points": [
                {"id": point.id, "u": point.u, "v": point.v} for point in recipe.points
            ],
            "curves": [_sketch_curve_to_payload(curve) for curve in recipe.curves],
            "constraints": [
                _sketch_constraint_to_payload(constraint)
                for constraint in recipe.constraints
            ],
        }
    raise TypeError("recipe is outside the native Agent geometry subset")


def geometry_recipe_from_payload(value: object) -> object:
    if not isinstance(value, Mapping):
        raise TypeError("geometry recipe payload must be an object")
    payload = dict(value)
    version = payload.pop("schema_version", None)
    if version != _RECIPE_SCHEMA_VERSION:
        raise ValueError("unknown geometry recipe schema_version")
    _validate_recipe_payload_budget(value)
    return _geometry_recipe_from_payload(payload)


def _geometry_recipe_from_payload(value: object) -> object:
    if not isinstance(value, Mapping):
        raise TypeError("geometry recipe payload must be an object")
    kind = value.get("kind")
    if kind == "face_sketch_boolean":
        raise ValueError("Agent can only read face-sketch Extrude Boolean features; creation and editing are not yet supported.")
    fields: dict[str, set[str]] = {
        "rectangle": {"kind", "name", "width", "height"},
        "disk": {"kind", "name", "radius"},
        "plate_with_hole": {
            "kind",
            "name",
            "width",
            "height",
            "hole_x",
            "hole_y",
            "hole_radius",
        },
        "box": {"kind", "name", "width", "depth", "height"},
        "cylinder": {"kind", "name", "radius", "height"},
        "translated": {"kind", "base", "dx", "dy", "dz"},
        "rotated": {"kind", "base", "axis", "angle_degrees"},
        "planar_sketch": {
            "kind",
            "name",
            "plane",
            "points",
            "curves",
            "constraints",
        },
        "wire": {"kind", "name", "points", "members"},
        "extruded": {"kind", "base", "height", "source_face_ids"},
        "revolved": {"kind", "base", "axis", "angle_degrees", "source_face_ids"},
        "path_swept": {"kind", "base", "path", "source_face_ids", "frame_strategy"},
        "boolean": {
            "kind",
            "name",
            "operation",
            "object",
            "tool",
            "body_context",
            "planar_context",
            "part_context",
        },
        "multi_body": {
            "kind",
            "name",
            "bodies",
            "retired_body_ids",
            "retired_boolean_feature_ids",
        },
    }
    if kind not in fields or set(value) != fields[kind]:  # type: ignore[index]
        raise ValueError("geometry recipe fields do not match the A2 schema")
    if kind == "wire":
        points = value["points"]
        members = value["members"]
        if (
            not isinstance(points, list)
            or not isinstance(members, list)
            or not 2 <= len(points) <= _MAX_PREVIEW_POINTS
            or not 1 <= len(members) <= _MAX_PREVIEW_POINTS
        ):
            raise ValueError("wire entities exceed the bounded schema")
        if any(
            not isinstance(item, Mapping) or set(item) != {"name", "x", "y", "z"}
            for item in points
        ):
            raise ValueError("wire point fields do not match")
        if any(
            not isinstance(item, Mapping) or set(item) != {"name", "start", "end"}
            for item in members
        ):
            raise ValueError("wire member fields do not match")
        return WireGeometry(
            value["name"],
            tuple(
                WirePoint(item["name"], item["x"], item["y"], item["z"])
                for item in points
            ),
            tuple(
                WireMember(item["name"], item["start"], item["end"]) for item in members
            ),
        )
    if kind == "rectangle":
        return RectangleGeometry(value["name"], value["width"], value["height"])
    if kind == "disk":
        return DiskGeometry(value["name"], value["radius"])
    if kind == "plate_with_hole":
        return PlateWithHoleGeometry(
            value["name"],
            value["width"],
            value["height"],
            value["hole_x"],
            value["hole_y"],
            value["hole_radius"],
        )
    if kind == "box":
        return BoxGeometry(
            value["name"],
            value["width"],
            value["depth"],
            value["height"],
        )
    if kind == "cylinder":
        return CylinderGeometry(
            value["name"],
            value["radius"],
            value["height"],
        )
    if kind == "extruded":
        return ExtrudedGeometry(
            _geometry_recipe_from_payload(value["base"]),
            value["height"],
            _logical_id_list(value["source_face_ids"], "source_face_ids"),
        )
    if kind == "revolved":
        return RevolvedGeometry(
            _geometry_recipe_from_payload(value["base"]),
            value["axis"],
            value["angle_degrees"],
            _logical_id_list(value["source_face_ids"], "source_face_ids"),
        )
    if kind == "path_swept":
        path = _geometry_recipe_from_payload(value["path"])
        if type(path) is not WireGeometry:
            raise ValueError("path_swept.path must decode to WireGeometry")
        return PathSweptGeometry(
            _geometry_recipe_from_payload(value["base"]),
            path,
            _logical_id_list(value["source_face_ids"], "source_face_ids"),
            value["frame_strategy"],
        )
    if kind == "boolean":
        return BooleanGeometry(
            value["name"],
            value["operation"],
            _geometry_recipe_from_payload(value["object"]),
            _geometry_recipe_from_payload(value["tool"]),
            _boolean_context_from_payload(value["body_context"], "body"),
            _boolean_context_from_payload(value["planar_context"], "planar"),
            _boolean_context_from_payload(value["part_context"], "part"),
        )
    if kind == "multi_body":
        bodies = value["bodies"]
        if not isinstance(bodies, list) or not 1 <= len(bodies) <= _MAX_RECIPE_NODES:
            raise ValueError("multi_body bodies exceed the bounded schema")
        if any(
            not isinstance(item, Mapping) or set(item) != {"id", "name", "recipe"}
            for item in bodies
        ):
            raise ValueError("multi_body Body fields do not match")
        return MultiBodyGeometry(
            value["name"],
            tuple(
                SolidBody(
                    item["id"],
                    item["name"],
                    _geometry_recipe_from_payload(item["recipe"]),
                )
                for item in bodies
            ),
            _string_list(value["retired_body_ids"], "retired_body_ids"),
            _string_list(
                value["retired_boolean_feature_ids"],
                "retired_boolean_feature_ids",
            ),
        )
    if kind == "translated":
        return MovedGeometry(
            _geometry_recipe_from_payload(value["base"]),
            value["dx"],
            value["dy"],
            value["dz"],
        )
    if kind == "planar_sketch":
        plane = value["plane"]
        if not isinstance(plane, Mapping) or set(plane) != {
            "origin",
            "x_direction",
            "y_direction",
        }:
            raise ValueError("planar sketch plane fields do not match")
        points = value["points"]
        curves = value["curves"]
        constraints = value["constraints"]
        if (
            not isinstance(points, list)
            or not isinstance(curves, list)
            or not isinstance(constraints, list)
            or not 1 <= len(points) <= 128
            or not 1 <= len(curves) <= 128
            or len(constraints) > 128
        ):
            raise ValueError("planar sketch entities exceed the bounded schema")
        if any(
            not isinstance(item, Mapping) or set(item) != {"id", "u", "v"}
            for item in points
        ):
            raise ValueError("planar sketch point fields do not match")
        return SketchGeometry(
            value["name"],
            SketchPlane(
                tuple(plane["origin"]),
                tuple(plane["x_direction"]),
                tuple(plane["y_direction"]),
            ),
            tuple(SketchPoint(item["id"], item["u"], item["v"]) for item in points),
            tuple(_sketch_curve_from_payload(item) for item in curves),
            tuple(_sketch_constraint_from_payload(item) for item in constraints),
        )
    return RotatedGeometry(
        _geometry_recipe_from_payload(value["base"]),
        value["axis"],
        value["angle_degrees"],
    )


def _validate_recipe_payload_budget(value: object) -> None:
    try:
        size = len(
            json.dumps(
                value,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            ).encode("utf-8")
        )
    except (TypeError, ValueError, RecursionError) as error:
        raise ValueError("geometry recipe payload must be finite JSON") from error
    if size > _MAX_RECIPE_BYTES:
        raise ValueError("geometry recipe payload exceeds the byte budget")
    nodes = 0

    def contains_feature_payload(item: object) -> bool:
        if isinstance(item, Mapping):
            if item.get("kind") in {"boolean", "multi_body"}:
                return True
            return any(contains_feature_payload(child) for child in item.values())
        if isinstance(item, list):
            return any(contains_feature_payload(child) for child in item)
        return False

    node_budget = (
        _MAX_BOOLEAN_RECIPE_PAYLOAD_NODES
        if contains_feature_payload(value)
        else _MAX_RECIPE_NODES
    )

    def visit(item: object, depth: int, *, planar_entity: bool = False) -> None:
        nonlocal nodes
        if depth > _MAX_RECIPE_DEPTH:
            raise ValueError("geometry recipe payload exceeds the depth budget")
        if isinstance(item, Mapping):
            if not planar_entity:
                nodes += 1
                if nodes > node_budget:
                    raise ValueError("geometry recipe payload exceeds the node budget")
            sketch_fields = (
                {"points", "curves", "constraints"}
                if item.get("kind") == "planar_sketch"
                else set()
            )
            for key, child in item.items():
                visit(
                    child,
                    depth + 1,
                    planar_entity=key in sketch_fields,
                )
        elif isinstance(item, list):
            for child in item:
                visit(child, depth + 1, planar_entity=planar_entity)

    visit(value, 0)


def _string_list(value: object, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or len(value) > _MAX_RECIPE_NODES:
        raise ValueError(f"{field_name} must be a bounded array")
    if any(type(item) is not str for item in value):
        raise ValueError(f"{field_name} must contain strings")
    return tuple(value)


def _logical_id_list(value: object, field_name: str) -> tuple[str, ...]:
    return _string_list(value, field_name)


def _lineage_to_payload(
    entities: Sequence[BooleanLineageEntity],
    mappings: Sequence[BooleanLineageMapping],
) -> dict[str, object]:
    return {
        "result_entities": [
            {
                "kind": item.kind,
                "logical_id": item.logical_id,
                "semantic_role": item.semantic_role,
                "topology_links": list(item.topology_links),
            }
            for item in entities
        ],
        "topology_mappings": [
            {
                "source": item.source,
                "source_logical_id": item.source_logical_id,
                "target_logical_id": item.target_logical_id,
                "relation": item.relation,
            }
            for item in mappings
        ],
    }


def _boolean_context_to_payload(value: object) -> dict[str, object] | None:
    if value is None:
        return None
    lineage = _lineage_to_payload(value.result_entities, value.topology_mappings)
    if type(value) is BooleanBodyContext:
        return {
            "feature_id": value.feature_id,
            "target_body_id": value.target_body_id,
            "tool_body_id": value.tool_body_id,
            "tool_body_name": value.tool_body_name,
            **lineage,
        }
    if type(value) is PlanarBooleanContext:
        return {
            "feature_id": value.feature_id,
            "target_face_id": value.target_face_id,
            "tool_face_ids": list(value.tool_face_ids),
            **lineage,
        }
    if type(value) is PartBooleanContext:
        return {
            "feature_id": value.feature_id,
            "target_part_id": value.target_part_id,
            "tool_part_id": value.tool_part_id,
            "result_part_id": value.result_part_id,
            **lineage,
        }
    raise TypeError("unknown Boolean context type")


def _boolean_context_from_payload(value: object, kind: str) -> object:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError(f"{kind}_context must be an object or null")
    common = {"feature_id", "result_entities", "topology_mappings"}
    specific = {
        "body": {"target_body_id", "tool_body_id", "tool_body_name"},
        "planar": {"target_face_id", "tool_face_ids"},
        "part": {"target_part_id", "tool_part_id", "result_part_id"},
    }[kind]
    if set(value) != common | specific:
        raise ValueError(f"{kind}_context fields do not match")
    raw_entities = value["result_entities"]
    raw_mappings = value["topology_mappings"]
    if not isinstance(raw_entities, list) or not isinstance(raw_mappings, list):
        raise ValueError("Boolean lineage must use arrays")
    if len(raw_entities) > 512 or len(raw_mappings) > 1024:
        raise ValueError("Boolean lineage exceeds the bounded schema")
    if any(
        not isinstance(item, Mapping)
        or set(item) != {"kind", "logical_id", "semantic_role", "topology_links"}
        for item in raw_entities
    ):
        raise ValueError("Boolean lineage entity fields do not match")
    if any(
        not isinstance(item, Mapping)
        or set(item) != {"source", "source_logical_id", "target_logical_id", "relation"}
        for item in raw_mappings
    ):
        raise ValueError("Boolean lineage mapping fields do not match")
    entities = tuple(
        BooleanLineageEntity(
            item["kind"],
            item["logical_id"],
            item["semantic_role"],
            _string_list(item["topology_links"], "topology_links"),
        )
        for item in raw_entities
    )
    mappings = tuple(
        BooleanLineageMapping(
            item["source"],
            item["source_logical_id"],
            item["target_logical_id"],
            item["relation"],
        )
        for item in raw_mappings
    )
    if kind == "body":
        return BooleanBodyContext(
            value["feature_id"],
            value["target_body_id"],
            value["tool_body_id"],
            value["tool_body_name"],
            entities,
            mappings,
        )
    if kind == "planar":
        return PlanarBooleanContext(
            value["feature_id"],
            value["target_face_id"],
            _string_list(value["tool_face_ids"], "tool_face_ids"),
            entities,
            mappings,
        )
    return PartBooleanContext(
        value["feature_id"],
        value["target_part_id"],
        value["tool_part_id"],
        value["result_part_id"],
        entities,
        mappings,
    )


def _sketch_curve_to_payload(
    curve: SketchLine | SketchArc | SketchCircle,
) -> dict[str, object]:
    if isinstance(curve, SketchLine):
        return {
            "kind": "line",
            "id": curve.id,
            "start_point_id": curve.start_point_id,
            "end_point_id": curve.end_point_id,
        }
    if isinstance(curve, SketchArc):
        return {
            "kind": "arc",
            "id": curve.id,
            "start_point_id": curve.start_point_id,
            "center_point_id": curve.center_point_id,
            "end_point_id": curve.end_point_id,
            "orientation": curve.orientation,
        }
    if not curve.is_curve:
        raise TypeError("planar sketch payload requires strict circles")
    return {
        "kind": "circle",
        "id": curve.id,
        "center_point_id": curve.center_point_id,
        "radius": curve.radius,
    }


def _sketch_constraint_to_payload(
    constraint: SketchConstraint,
) -> dict[str, object]:
    common: dict[str, object] = {
        "id": constraint.id,
        "source": constraint.source,
        "enabled": constraint.enabled,
    }
    cases: tuple[tuple[type, str, tuple[str, ...]], ...] = (
        (
            SketchCoincidentConstraint,
            "coincident",
            ("first_point_id", "second_point_id"),
        ),
        (SketchPointOnCurveConstraint, "point_on_curve", ("point_id", "curve_id")),
        (SketchHorizontalConstraint, "horizontal", ("line_id",)),
        (SketchVerticalConstraint, "vertical", ("line_id",)),
        (SketchParallelConstraint, "parallel", ("first_line_id", "second_line_id")),
        (
            SketchPerpendicularConstraint,
            "perpendicular",
            ("first_line_id", "second_line_id"),
        ),
        (
            SketchTangentConstraint,
            "tangent",
            ("first_curve_id", "second_curve_id", "branch_hint"),
        ),
        (
            SketchEqualLengthConstraint,
            "equal_length",
            ("first_line_id", "second_line_id"),
        ),
        (
            SketchEqualRadiusConstraint,
            "equal_radius",
            ("first_curve_id", "second_curve_id"),
        ),
        (
            SketchConcentricConstraint,
            "concentric",
            ("first_curve_id", "second_curve_id"),
        ),
        (SketchFixedConstraint, "fixed", ("point_id", "u", "v")),
        (
            SketchDistanceDimension,
            "distance",
            ("first_point_id", "second_point_id", "value", "driving"),
        ),
        (SketchRadiusDimension, "radius", ("curve_id", "value", "driving")),
        (
            SketchAngleDimension,
            "angle",
            ("first_line_id", "second_line_id", "value", "driving"),
        ),
    )
    for expected, kind, fields in cases:
        if type(constraint) is expected:
            return {
                "kind": kind,
                **common,
                **{field: getattr(constraint, field) for field in fields},
            }
    raise TypeError("unsupported sketch constraint type")


def _sketch_constraint_from_payload(value: object) -> SketchConstraint:
    if not isinstance(value, Mapping):
        raise TypeError("sketch constraint payload must be an object")
    data = dict(value)
    kind = data.get("kind")
    common = {"kind", "id", "source", "enabled"}
    fields: dict[str, tuple[type, tuple[str, ...]]] = {
        "coincident": (
            SketchCoincidentConstraint,
            ("first_point_id", "second_point_id"),
        ),
        "point_on_curve": (SketchPointOnCurveConstraint, ("point_id", "curve_id")),
        "horizontal": (SketchHorizontalConstraint, ("line_id",)),
        "vertical": (SketchVerticalConstraint, ("line_id",)),
        "parallel": (SketchParallelConstraint, ("first_line_id", "second_line_id")),
        "perpendicular": (
            SketchPerpendicularConstraint,
            ("first_line_id", "second_line_id"),
        ),
        "tangent": (
            SketchTangentConstraint,
            ("first_curve_id", "second_curve_id", "branch_hint"),
        ),
        "equal_length": (
            SketchEqualLengthConstraint,
            ("first_line_id", "second_line_id"),
        ),
        "equal_radius": (
            SketchEqualRadiusConstraint,
            ("first_curve_id", "second_curve_id"),
        ),
        "concentric": (
            SketchConcentricConstraint,
            ("first_curve_id", "second_curve_id"),
        ),
        "fixed": (SketchFixedConstraint, ("point_id", "u", "v")),
        "distance": (
            SketchDistanceDimension,
            ("first_point_id", "second_point_id", "value", "driving"),
        ),
        "radius": (SketchRadiusDimension, ("curve_id", "value", "driving")),
        "angle": (
            SketchAngleDimension,
            ("first_line_id", "second_line_id", "value", "driving"),
        ),
    }
    if kind not in fields:
        raise ValueError("sketch constraint kind is unsupported")
    constraint_type, specific = fields[str(kind)]
    if set(data) != common | set(specific):
        raise ValueError("sketch constraint fields do not match its kind")
    return constraint_type(
        data["id"],
        *(data[field] for field in specific),
        source=data["source"],
        enabled=data["enabled"],
    )


def _sketch_curve_from_payload(
    value: object,
) -> SketchLine | SketchArc | SketchCircle:
    if not isinstance(value, Mapping):
        raise TypeError("sketch curve payload must be an object")
    kind = value.get("kind")
    if kind == "line" and set(value) == {
        "kind",
        "id",
        "start_point_id",
        "end_point_id",
    }:
        return SketchLine(
            value["id"],
            value["start_point_id"],
            value["end_point_id"],
        )
    if kind == "arc" and set(value) == {
        "kind",
        "id",
        "start_point_id",
        "center_point_id",
        "end_point_id",
        "orientation",
    }:
        return SketchArc(
            value["id"],
            value["start_point_id"],
            value["center_point_id"],
            value["end_point_id"],
            value["orientation"],
        )
    if kind == "circle" and set(value) == {
        "kind",
        "id",
        "center_point_id",
        "radius",
    }:
        return SketchCircle(
            value["id"],
            value["center_point_id"],
            value["radius"],
        )
    raise ValueError("sketch curve fields do not match the bounded schema")


__all__ = [
    "geometry_recipe_to_payload",
    "geometry_recipe_from_payload",
]
