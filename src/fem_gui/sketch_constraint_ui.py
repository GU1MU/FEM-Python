"""Qt-free presentation and inference helpers for sketch constraints."""

from __future__ import annotations

from dataclasses import dataclass
import math

from fem.geometry.recipes import (
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
    sketch_constraint_entity_ids,
)
from fem.geometry.sketch_solver import SketchSolveResult


@dataclass(frozen=True, slots=True)
class SketchConstraintOverlay:
    """Stable, renderer-independent description of one viewport annotation."""

    constraint_id: str
    kind: str
    text: str
    position: tuple[float, float, float]
    entity_ids: tuple[str, ...]
    warning: bool = False


@dataclass(frozen=True, slots=True)
class SketchInferencePreview:
    """Transient inference result; this value is never part of draft history."""

    kinds: tuple[str, ...] = ()
    snapped_point_id: str | None = None
    intersection_curve_ids: tuple[str, ...] = ()


_KIND_TEXT = {
    SketchCoincidentConstraint: ("coincident", "重合"),
    SketchPointOnCurveConstraint: ("point_on_curve", "点在曲线上"),
    SketchHorizontalConstraint: ("horizontal", "水平"),
    SketchVerticalConstraint: ("vertical", "垂直"),
    SketchParallelConstraint: ("parallel", "平行"),
    SketchPerpendicularConstraint: ("perpendicular", "互相垂直"),
    SketchTangentConstraint: ("tangent", "相切"),
    SketchEqualLengthConstraint: ("equal_length", "等长"),
    SketchEqualRadiusConstraint: ("equal_radius", "等半径"),
    SketchConcentricConstraint: ("concentric", "同心"),
    SketchFixedConstraint: ("fixed", "固定"),
    SketchDistanceDimension: ("distance", "距离"),
    SketchRadiusDimension: ("radius", "半径"),
    SketchAngleDimension: ("angle", "角度"),
}


def constraint_text(constraint: SketchConstraint) -> str:
    kind, label = _KIND_TEXT[type(constraint)]
    del kind
    if isinstance(
        constraint,
        (SketchDistanceDimension, SketchRadiusDimension, SketchAngleDimension),
    ):
        mode = "驱动" if constraint.driving else "参考"
        return f"{label} {constraint.value:g}（{mode}）"
    return label


def constraints_for_entities(
    constraints: tuple[SketchConstraint, ...],
    entity_ids: tuple[str, ...],
) -> tuple[SketchConstraint, ...]:
    selected = frozenset(entity_ids)
    return tuple(
        constraint
        for constraint in constraints
        if selected.intersection(sketch_constraint_entity_ids(constraint))
    )


def measured_dimension_value(
    constraint: SketchConstraint,
    points: dict[str, SketchPoint],
    curves: dict[str, object],
) -> float | None:
    if isinstance(constraint, SketchDistanceDimension):
        first = points[constraint.first_point_id]
        second = points[constraint.second_point_id]
        return math.hypot(second.u - first.u, second.v - first.v)
    if isinstance(constraint, SketchRadiusDimension):
        curve = curves[constraint.curve_id]
        if isinstance(curve, SketchCircle):
            return curve.radius
        if isinstance(curve, SketchArc):
            center = points[curve.center_point_id]
            start = points[curve.start_point_id]
            return math.hypot(start.u - center.u, start.v - center.v)
    if isinstance(constraint, SketchAngleDimension):
        first = curves[constraint.first_line_id]
        second = curves[constraint.second_line_id]
        if isinstance(first, SketchLine) and isinstance(second, SketchLine):
            first_start = points[first.start_point_id]
            first_end = points[first.end_point_id]
            second_start = points[second.start_point_id]
            second_end = points[second.end_point_id]
            first_vector = (
                first_end.u - first_start.u, first_end.v - first_start.v
            )
            second_vector = (
                second_end.u - second_start.u, second_end.v - second_start.v
            )
            return math.atan2(
                first_vector[0] * second_vector[1]
                - first_vector[1] * second_vector[0],
                first_vector[0] * second_vector[0]
                + first_vector[1] * second_vector[1],
            )
    return None


def build_constraint_overlays(
    points: tuple[SketchPoint, ...],
    curves: tuple[object, ...],
    constraints: tuple[SketchConstraint, ...],
    plane: SketchPlane,
    *,
    warning_ids: tuple[str, ...] = (),
) -> tuple[SketchConstraintOverlay, ...]:
    """Project every Phase-2 constraint to deterministic viewport label data."""

    point_map = {point.id: point for point in points}
    curve_map = {curve.id: curve for curve in curves}
    warnings = frozenset(warning_ids)
    overlays: list[SketchConstraintOverlay] = []
    for constraint in constraints:
        entity_ids = sketch_constraint_entity_ids(constraint)
        anchors: list[SketchPoint] = [
            point_map[item] for item in entity_ids if item in point_map
        ]
        for entity_id in entity_ids:
            curve = curve_map.get(entity_id)
            if isinstance(curve, SketchLine):
                anchors.extend(
                    (point_map[curve.start_point_id], point_map[curve.end_point_id])
                )
            elif isinstance(curve, (SketchCircle, SketchArc)):
                anchors.append(point_map[curve.center_point_id])
        if not anchors:
            continue
        u = sum(point.u for point in anchors) / len(anchors)
        v = sum(point.v for point in anchors) / len(anchors)
        kind, _label = _KIND_TEXT[type(constraint)]
        value = measured_dimension_value(constraint, point_map, curve_map)
        text = constraint_text(constraint)
        if value is not None and not constraint.driving:
            text = text.replace(f"{constraint.value:g}", f"{value:g}")
        overlays.append(
            SketchConstraintOverlay(
                constraint.id,
                kind,
                text,
                plane.to_global(u, v),
                entity_ids,
                constraint.id in warnings,
            )
        )
    return tuple(overlays)


def infer_line_preview(
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    auto_constraints: bool,
    snap_kind: str | None = None,
    snapped_point_id: str | None = None,
    intersection_curve_ids: tuple[str, ...] = (),
    direction_tolerance: float = 0.02,
) -> SketchInferencePreview:
    """Infer only first-release relations and keep grid snaps coordinate-only."""

    if not auto_constraints:
        return SketchInferencePreview()
    du = end[0] - start[0]
    dv = end[1] - start[1]
    length = math.hypot(du, dv)
    kinds: list[str] = []
    if length > 0.0:
        if abs(dv) <= direction_tolerance * length:
            kinds.append("horizontal")
        elif abs(du) <= direction_tolerance * length:
            kinds.append("vertical")
    if snap_kind == "sketch_point" and snapped_point_id is not None:
        kinds.append("coincident")
    if snap_kind == "intersection" and len(intersection_curve_ids) == 2:
        kinds.extend(("point_on_curve", "point_on_curve"))
    return SketchInferencePreview(
        tuple(kinds),
        snapped_point_id if snap_kind == "sketch_point" else None,
        tuple(intersection_curve_ids) if snap_kind == "intersection" else (),
    )


def solve_status_text(result: SketchSolveResult) -> str:
    labels = {
        "under_constrained": "欠约束",
        "fully_constrained": "完全约束",
        "redundant": "冗余约束",
        "conflicting": "约束冲突",
        "failed": "约束求解失败",
    }
    text = f"{labels[result.status]}；剩余自由度：{result.remaining_dof}"
    if result.redundant_constraint_ids:
        text += "；冗余：" + "、".join(result.redundant_constraint_ids)
    if result.conflicting_constraint_ids:
        text += "；冲突候选：" + "、".join(result.conflicting_constraint_ids)
    return text


__all__ = [
    "SketchConstraintOverlay",
    "SketchInferencePreview",
    "build_constraint_overlays",
    "constraint_text",
    "constraints_for_entities",
    "infer_line_preview",
    "measured_dimension_value",
    "solve_status_text",
]
