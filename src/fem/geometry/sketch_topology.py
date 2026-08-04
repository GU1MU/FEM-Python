"""Exact, deterministic topology operations for strict planar sketches."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import math
from collections.abc import Iterable

from .recipes import (
    SketchAngleDimension,
    SketchArc,
    SketchCircle,
    SketchConcentricConstraint,
    SketchConstraint,
    SketchEqualLengthConstraint,
    SketchEqualRadiusConstraint,
    SketchGeometry,
    SketchHorizontalConstraint,
    SketchLine,
    SketchParallelConstraint,
    SketchPerpendicularConstraint,
    SketchPoint,
    SketchPointOnCurveConstraint,
    SketchRadiusDimension,
    SketchTangentConstraint,
    SketchVerticalConstraint,
    sketch_constraint_entity_ids,
)
from .sketch_intersections import SketchIntersection


@dataclass(frozen=True, slots=True)
class SketchCurveSplitResult:
    """Detached result of one exact curve split and its migration diagnostics."""

    sketch: SketchGeometry
    original_curve_id: str
    derived_curve_ids: tuple[str, ...]
    split_point_ids: tuple[str, ...]
    migrated_constraint_ids: tuple[str, ...]
    removed_constraint_ids: tuple[str, ...]
    diagnostics: tuple[str, ...] = ()

    @property
    def points(self) -> tuple[SketchPoint, ...]:
        return self.sketch.points

    @property
    def curves(self):
        return self.sketch.curves

    @property
    def constraints(self) -> tuple[SketchConstraint, ...]:
        return self.sketch.constraints


def split_curve_at(
    sketch: SketchGeometry,
    curve_id: str,
    parameter: float | SketchIntersection | Iterable[float] | None = None,
    *,
    parameters: Iterable[float] = (),
    tolerance: float = 1.0e-8,
) -> SketchCurveSplitResult:
    """Split one line, circle, or arc at exact normalized curve parameters.

    Line and arc parameters must be strictly internal.  A circle with one split
    location gains its deterministic antipode so the result remains expressible
    as two non-degenerate arcs.  Derived IDs depend only on source identity and
    normalized parameters.
    """

    if type(sketch) is not SketchGeometry or not sketch.is_strict:
        raise TypeError("sketch must be a strict SketchGeometry")
    clean_tolerance = float(tolerance)
    if not math.isfinite(clean_tolerance) or clean_tolerance <= 0.0:
        raise ValueError("tolerance must be finite and positive")
    curve = sketch.curve(str(curve_id))
    raw_parameters = list(parameters)
    if isinstance(parameter, SketchIntersection):
        if parameter.left_curve_id == curve.id:
            raw_parameters.append(parameter.left_parameter)
        elif parameter.right_curve_id == curve.id:
            raw_parameters.append(parameter.right_parameter)
        else:
            raise ValueError("intersection does not reference the target curve")
    elif parameter is not None and isinstance(parameter, Iterable) and not isinstance(
        parameter, (str, bytes)
    ):
        raw_parameters.extend(parameter)
    elif parameter is not None:
        raw_parameters.append(parameter)
    if not raw_parameters:
        raise ValueError("at least one split parameter is required")
    normalized = _normalize_parameters(curve, raw_parameters, clean_tolerance)
    point_map = {point.id: point for point in sketch.points}
    split_parameters = normalized
    if isinstance(curve, SketchCircle) and len(split_parameters) == 1:
        split_parameters = tuple(sorted((split_parameters[0], (split_parameters[0] + 0.5) % 1.0)))

    split_points: list[SketchPoint] = []
    parameter_point_ids: dict[float, str] = {}
    occupied_ids = {item.id.casefold() for item in (*sketch.points, *sketch.curves)}
    for value in split_parameters:
        u, v = _curve_coordinate(curve, value, point_map)
        existing = _point_at(sketch.points, u, v, clean_tolerance)
        if existing is None:
            point_id = _derived_point_id(u, v)
            if point_id.casefold() in occupied_ids:
                raise ValueError(f"derived split point id is already used: {point_id}")
            existing = SketchPoint(point_id, u, v)
            split_points.append(existing)
            occupied_ids.add(point_id.casefold())
        parameter_point_ids[value] = existing.id

    derived = _derived_curves(curve, split_parameters, parameter_point_ids)
    for item in derived:
        if item.id.casefold() in occupied_ids:
            raise ValueError(f"derived split curve id is already used: {item.id}")
        occupied_ids.add(item.id.casefold())
    all_point_map = dict(point_map)
    all_point_map.update((point.id, point) for point in split_points)
    migrated, removed, diagnostics = _migrate_constraints(
        sketch.constraints, curve, derived, all_point_map, clean_tolerance
    )
    curves = tuple(item for item in sketch.curves if item.id != curve.id) + derived
    result_sketch = SketchGeometry(
        sketch.name,
        sketch.plane,
        sketch.points + tuple(split_points),
        curves,
        migrated,
    )
    return SketchCurveSplitResult(
        result_sketch,
        curve.id,
        tuple(item.id for item in derived),
        tuple(parameter_point_ids[value] for value in split_parameters),
        tuple(
            item.id
            for item in migrated
            if any(entity_id in {curve.id for curve in derived}
                   for entity_id in sketch_constraint_entity_ids(item))
        ),
        removed,
        diagnostics,
    )


def _normalize_parameters(curve, values: Iterable[float], tolerance: float) -> tuple[float, ...]:
    result: list[float] = []
    for raw in values:
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise TypeError("split parameter must be a finite real number")
        value = float(raw)
        if not math.isfinite(value):
            raise ValueError("split parameter must be finite")
        if isinstance(curve, SketchCircle):
            value %= 1.0
        elif value <= tolerance or value >= 1.0 - tolerance:
            raise ValueError("split parameter must be strictly inside the curve")
        if any(abs(value - existing) <= tolerance for existing in result):
            continue
        result.append(value)
    result.sort()
    if not result:
        raise ValueError("split would create only degenerate fragments")
    if isinstance(curve, SketchCircle) and len(result) > 1:
        gaps = tuple(
            (result[(index + 1) % len(result)] - result[index]) % 1.0
            for index in range(len(result))
        )
        if min(gaps) <= tolerance:
            raise ValueError("split would create a degenerate circle arc")
    return tuple(result)


def _curve_coordinate(curve, parameter: float, points: dict[str, SketchPoint]) -> tuple[float, float]:
    if isinstance(curve, SketchLine):
        start = points[curve.start_point_id]
        end = points[curve.end_point_id]
        return (
            start.u + parameter * (end.u - start.u),
            start.v + parameter * (end.v - start.v),
        )
    center = points[curve.center_point_id]
    if isinstance(curve, SketchCircle):
        angle = math.tau * parameter
        radius = curve.radius
    else:
        start = points[curve.start_point_id]
        end = points[curve.end_point_id]
        start_angle = math.atan2(start.v - center.v, start.u - center.u)
        end_angle = math.atan2(end.v - center.v, end.u - center.u)
        sweep = (
            (end_angle - start_angle) % math.tau
            if curve.orientation == "ccw"
            else -((start_angle - end_angle) % math.tau)
        )
        angle = start_angle + parameter * sweep
        radius = math.hypot(start.u - center.u, start.v - center.v)
    return center.u + radius * math.cos(angle), center.v + radius * math.sin(angle)


def _derived_curves(curve, parameters: tuple[float, ...], point_ids: dict[float, str]):
    token = ",".join(_parameter_token(value) for value in parameters)
    if isinstance(curve, SketchLine):
        endpoints = (curve.start_point_id, *(point_ids[value] for value in parameters), curve.end_point_id)
        return tuple(
            SketchLine(_derived_curve_id(curve.id, token, index), start, end)
            for index, (start, end) in enumerate(zip(endpoints, endpoints[1:]))
        )
    if isinstance(curve, SketchArc):
        endpoints = (curve.start_point_id, *(point_ids[value] for value in parameters), curve.end_point_id)
        return tuple(
            SketchArc(
                _derived_curve_id(curve.id, token, index),
                start,
                curve.center_point_id,
                end,
                curve.orientation,
            )
            for index, (start, end) in enumerate(zip(endpoints, endpoints[1:]))
        )
    ordered = tuple(point_ids[value] for value in parameters)
    return tuple(
        SketchArc(
            _derived_curve_id(curve.id, token, index),
            ordered[index],
            curve.center_point_id,
            ordered[(index + 1) % len(ordered)],
            "ccw",
        )
        for index in range(len(ordered))
    )


def _migrate_constraints(
    constraints: tuple[SketchConstraint, ...],
    original,
    derived: tuple,
    points: dict[str, SketchPoint],
    tolerance: float,
) -> tuple[tuple[SketchConstraint, ...], tuple[str, ...], tuple[str, ...]]:
    migrated: list[SketchConstraint] = []
    removed: list[str] = []
    diagnostics: list[str] = []
    for constraint in constraints:
        if original.id not in sketch_constraint_entity_ids(constraint):
            migrated.append(constraint)
            continue
        replacements: list[SketchConstraint] = []
        if isinstance(constraint, SketchPointOnCurveConstraint):
            point = points[constraint.point_id]
            matches = [item for item in derived if _point_lies_on_curve(point, item, points, tolerance)]
            if matches:
                replacements = [replace(constraint, curve_id=matches[0].id)]
        elif isinstance(constraint, (SketchHorizontalConstraint, SketchVerticalConstraint)):
            replacements = [replace(constraint, line_id=item.id) for item in derived]
        elif isinstance(
            constraint,
            (SketchParallelConstraint, SketchPerpendicularConstraint,
             SketchEqualLengthConstraint, SketchAngleDimension),
        ):
            if isinstance(constraint, SketchEqualLengthConstraint):
                replacements = []
            else:
                replacements = [
                    replace(
                        constraint,
                        first_line_id=item.id if constraint.first_line_id == original.id else constraint.first_line_id,
                        second_line_id=item.id if constraint.second_line_id == original.id else constraint.second_line_id,
                    )
                    for item in derived
                ]
        elif isinstance(constraint, SketchRadiusDimension):
            replacements = [replace(constraint, curve_id=item.id) for item in derived]
        elif isinstance(constraint, (SketchEqualRadiusConstraint, SketchConcentricConstraint)):
            replacements = [
                replace(
                    constraint,
                    first_curve_id=item.id if constraint.first_curve_id == original.id else constraint.first_curve_id,
                    second_curve_id=item.id if constraint.second_curve_id == original.id else constraint.second_curve_id,
                )
                for item in derived
            ]
        elif isinstance(constraint, SketchTangentConstraint):
            replacements = []
        if not replacements:
            removed.append(constraint.id)
            diagnostics.append(f"约束 {constraint.id} 因曲线分割后含义歧义而移除")
            continue
        for index, replacement in enumerate(replacements):
            migrated.append(
                replacement if index == 0 else replace(
                    replacement,
                    id=_derived_constraint_id(constraint.id, replacement, index),
                )
            )
    return tuple(migrated), tuple(removed), tuple(diagnostics)


def _point_lies_on_curve(
    point: SketchPoint, curve, points: dict[str, SketchPoint], tolerance: float
) -> bool:
    extended = dict(points)
    if isinstance(curve, SketchLine):
        start, end = extended[curve.start_point_id], extended[curve.end_point_id]
        du, dv = end.u - start.u, end.v - start.v
        length_squared = du * du + dv * dv
        parameter = ((point.u - start.u) * du + (point.v - start.v) * dv) / length_squared
        cross = du * (point.v - start.v) - dv * (point.u - start.u)
        return -tolerance <= parameter <= 1.0 + tolerance and abs(cross) <= tolerance * math.sqrt(length_squared)
    center = extended[curve.center_point_id]
    start = extended[curve.start_point_id]
    end = extended[curve.end_point_id]
    radius = math.hypot(start.u - center.u, start.v - center.v)
    if abs(math.hypot(point.u - center.u, point.v - center.v) - radius) > tolerance:
        return False
    start_angle = math.atan2(start.v - center.v, start.u - center.u) % math.tau
    end_angle = math.atan2(end.v - center.v, end.u - center.u) % math.tau
    point_angle = math.atan2(point.v - center.v, point.u - center.u) % math.tau
    travelled = ((point_angle - start_angle) % math.tau) if curve.orientation == "ccw" else ((start_angle - point_angle) % math.tau)
    sweep = ((end_angle - start_angle) % math.tau) if curve.orientation == "ccw" else ((start_angle - end_angle) % math.tau)
    return travelled <= sweep + tolerance


def _point_at(points: tuple[SketchPoint, ...], u: float, v: float, tolerance: float) -> SketchPoint | None:
    return next((point for point in points if math.hypot(point.u - u, point.v - v) <= tolerance), None)


def _parameter_token(value: float) -> str:
    return format(value, ".15g")


def _derived_point_id(u: float, v: float) -> str:
    canonical_u = 0.0 if abs(u) <= 1.0e-12 else round(u, 12)
    canonical_v = 0.0 if abs(v) <= 1.0e-12 else round(v, 12)
    digest = hashlib.sha1(
        f"{canonical_u:.12g},{canonical_v:.12g}".encode("utf-8")
    ).hexdigest()[:16]
    return f"split-point/{digest}"


def _derived_curve_id(source_id: str, token: str, index: int) -> str:
    digest = hashlib.sha1(f"{source_id}|{token}|{index}".encode("utf-8")).hexdigest()[:16]
    return f"{source_id}/split/{digest}"


def _derived_constraint_id(source_id: str, constraint: SketchConstraint, index: int) -> str:
    entities = "|".join(sketch_constraint_entity_ids(constraint))
    digest = hashlib.sha1(f"{source_id}|{entities}|{index}".encode("utf-8")).hexdigest()[:16]
    return f"{source_id}/split/{digest}"


__all__ = ["SketchCurveSplitResult", "split_curve_at"]
