"""Pure residual construction for the first sketch-constraint solver.

All coordinates in this module are plane-local U/V values.  Residuals are
dimensionless: geometric distances are divided by one characteristic length.
The module deliberately has no GUI or optimizer dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

from .recipes import (
    SketchArc,
    SketchCircle,
    SketchCoincidentConstraint,
    SketchConstraint,
    SketchCurve,
    SketchDistanceDimension,
    SketchFixedConstraint,
    SketchHorizontalConstraint,
    SketchLine,
    SketchPoint,
    SketchPointOnCurveConstraint,
    SketchRadiusDimension,
    SketchVerticalConstraint,
)


_INTERNAL_PREFIX = "__internal__:"


@dataclass(frozen=True, slots=True)
class SketchResidualBlock:
    """Residual values contributed by one constraint or intrinsic curve rule."""

    owner_id: str
    values: tuple[float, ...]
    internal: bool = False


def sketch_characteristic_length(
    points: Mapping[str, SketchPoint],
    curves: tuple[SketchCurve, ...],
    constraints: tuple[SketchConstraint, ...] = (),
) -> float:
    """Return a stable positive scale for normalized sketch equations."""

    coordinates = tuple(points.values())
    spans: list[float] = []
    if coordinates:
        spans.extend(
            (
                max(point.u for point in coordinates)
                - min(point.u for point in coordinates),
                max(point.v for point in coordinates)
                - min(point.v for point in coordinates),
            )
        )
    spans.extend(
        curve.radius
        for curve in curves
        if isinstance(curve, SketchCircle) and curve.is_curve
    )
    spans.extend(
        constraint.value
        for constraint in constraints
        if isinstance(constraint, (SketchDistanceDimension, SketchRadiusDimension))
    )
    characteristic = max(spans, default=1.0)
    return characteristic if characteristic > 1.0e-12 else 1.0


def constraint_signature(constraint: SketchConstraint) -> tuple[object, ...] | None:
    """Return a canonical signature used for deterministic duplicate detection."""

    if not constraint.enabled:
        return None
    if isinstance(constraint, SketchCoincidentConstraint):
        return (type(constraint), *sorted((constraint.first_point_id, constraint.second_point_id)))
    if isinstance(constraint, SketchPointOnCurveConstraint):
        return (type(constraint), constraint.point_id, constraint.curve_id)
    if isinstance(constraint, (SketchHorizontalConstraint, SketchVerticalConstraint)):
        return (type(constraint), constraint.line_id)
    if isinstance(constraint, SketchFixedConstraint):
        return (type(constraint), constraint.point_id, constraint.u, constraint.v)
    if isinstance(constraint, SketchDistanceDimension):
        if not constraint.driving:
            return None
        return (
            type(constraint),
            *sorted((constraint.first_point_id, constraint.second_point_id)),
            constraint.value,
        )
    if isinstance(constraint, SketchRadiusDimension):
        if not constraint.driving:
            return None
        return (type(constraint), constraint.curve_id, constraint.value)
    raise TypeError("unsupported sketch constraint")


def duplicate_constraint_ids(
    constraints: tuple[SketchConstraint, ...],
) -> tuple[str, ...]:
    """Return later IDs from semantically identical active constraints."""

    seen: set[tuple[object, ...]] = set()
    duplicates: list[str] = []
    for constraint in constraints:
        signature = constraint_signature(constraint)
        if signature is None:
            continue
        if signature in seen:
            duplicates.append(constraint.id)
        else:
            seen.add(signature)
    return tuple(duplicates)


def evaluate_sketch_residuals(
    points: Mapping[str, SketchPoint],
    curves: tuple[SketchCurve, ...],
    constraints: tuple[SketchConstraint, ...],
    *,
    characteristic_length: float | None = None,
) -> tuple[SketchResidualBlock, ...]:
    """Evaluate normalized residual blocks for all active driving constraints."""

    scale = (
        sketch_characteristic_length(points, curves, constraints)
        if characteristic_length is None
        else float(characteristic_length)
    )
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("characteristic_length must be finite and positive")
    curve_map = {curve.id: curve for curve in curves}
    blocks: list[SketchResidualBlock] = []

    for curve in curves:
        if isinstance(curve, SketchArc):
            center = points[curve.center_point_id]
            start = points[curve.start_point_id]
            end = points[curve.end_point_id]
            blocks.append(
                SketchResidualBlock(
                    f"{_INTERNAL_PREFIX}{curve.id}",
                    ((_distance(start, center) - _distance(end, center)) / scale,),
                    internal=True,
                )
            )

    for constraint in constraints:
        if not constraint.enabled:
            continue
        values: tuple[float, ...]
        if isinstance(constraint, SketchCoincidentConstraint):
            first = points[constraint.first_point_id]
            second = points[constraint.second_point_id]
            values = ((first.u - second.u) / scale, (first.v - second.v) / scale)
        elif isinstance(constraint, SketchPointOnCurveConstraint):
            point = points[constraint.point_id]
            curve = curve_map[constraint.curve_id]
            values = (_point_on_curve_residual(point, curve, points, scale),)
        elif isinstance(constraint, SketchHorizontalConstraint):
            line = curve_map[constraint.line_id]
            assert isinstance(line, SketchLine)
            values = (
                (points[line.end_point_id].v - points[line.start_point_id].v) / scale,
            )
        elif isinstance(constraint, SketchVerticalConstraint):
            line = curve_map[constraint.line_id]
            assert isinstance(line, SketchLine)
            values = (
                (points[line.end_point_id].u - points[line.start_point_id].u) / scale,
            )
        elif isinstance(constraint, SketchFixedConstraint):
            point = points[constraint.point_id]
            values = ((point.u - constraint.u) / scale, (point.v - constraint.v) / scale)
        elif isinstance(constraint, SketchDistanceDimension):
            if not constraint.driving:
                continue
            values = (
                (
                    _distance(
                        points[constraint.first_point_id],
                        points[constraint.second_point_id],
                    )
                    - constraint.value
                )
                / scale,
            )
        elif isinstance(constraint, SketchRadiusDimension):
            if not constraint.driving:
                continue
            curve = curve_map[constraint.curve_id]
            radius = _curve_radius(curve, points)
            values = ((radius - constraint.value) / scale,)
        else:
            raise TypeError("unsupported sketch constraint")
        blocks.append(SketchResidualBlock(constraint.id, values))
    return tuple(blocks)


def flatten_residual_blocks(blocks: tuple[SketchResidualBlock, ...]) -> tuple[float, ...]:
    return tuple(value for block in blocks for value in block.values)


def _point_on_curve_residual(
    point: SketchPoint,
    curve: SketchCurve,
    points: Mapping[str, SketchPoint],
    scale: float,
) -> float:
    if isinstance(curve, SketchLine):
        start = points[curve.start_point_id]
        end = points[curve.end_point_id]
        du = end.u - start.u
        dv = end.v - start.v
        length = math.hypot(du, dv)
        if length <= 1.0e-15 * scale:
            return 1.0
        parameter = (
            (point.u - start.u) * du + (point.v - start.v) * dv
        ) / (length * length)
        if parameter < 0.0:
            return _distance(point, start) / scale
        if parameter > 1.0:
            return _distance(point, end) / scale
        signed_distance = (du * (point.v - start.v) - dv * (point.u - start.u)) / length
        return signed_distance / scale
    center = points[curve.center_point_id]
    radial = (_distance(point, center) - _curve_radius(curve, points)) / scale
    if isinstance(curve, SketchCircle):
        return radial
    assert isinstance(curve, SketchArc)
    if _point_angle_on_arc(point, curve, points):
        return radial
    start = points[curve.start_point_id]
    end = points[curve.end_point_id]
    nearest_endpoint_distance = min(_distance(point, start), _distance(point, end))
    return math.copysign(nearest_endpoint_distance / scale, radial or 1.0)


def _curve_radius(
    curve: SketchCurve,
    points: Mapping[str, SketchPoint],
) -> float:
    if isinstance(curve, SketchCircle):
        return curve.radius
    if isinstance(curve, SketchArc):
        return _distance(points[curve.start_point_id], points[curve.center_point_id])
    raise TypeError("line curves do not have a radius")


def _distance(first: SketchPoint, second: SketchPoint) -> float:
    return math.hypot(first.u - second.u, first.v - second.v)


def _point_angle_on_arc(
    point: SketchPoint,
    arc: SketchArc,
    points: Mapping[str, SketchPoint],
) -> bool:
    center = points[arc.center_point_id]
    start = points[arc.start_point_id]
    end = points[arc.end_point_id]
    tau = 2.0 * math.pi
    start_angle = math.atan2(start.v - center.v, start.u - center.u) % tau
    end_angle = math.atan2(end.v - center.v, end.u - center.u) % tau
    point_angle = math.atan2(point.v - center.v, point.u - center.u) % tau
    if arc.orientation == "ccw":
        sweep = (end_angle - start_angle) % tau
        offset = (point_angle - start_angle) % tau
    else:
        sweep = (start_angle - end_angle) % tau
        offset = (start_angle - point_angle) % tau
    return offset <= sweep + 1.0e-12


__all__ = [
    "SketchResidualBlock",
    "constraint_signature",
    "duplicate_constraint_ids",
    "evaluate_sketch_residuals",
    "flatten_residual_blocks",
    "sketch_characteristic_length",
]
