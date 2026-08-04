"""Pure two-dimensional analytic intersections for strict sketch curves."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal, Mapping

from .recipes import SketchArc, SketchCircle, SketchCurve, SketchLine, SketchPoint


IntersectionKind = Literal["crossing", "tangent"]
IntersectionDiagnosticKind = Literal["degenerate", "coincident", "overlap"]


@dataclass(frozen=True, slots=True)
class SketchIntersection:
    """One stable intersection in authoritative sketch U/V coordinates."""

    left_curve_id: str
    right_curve_id: str
    u: float
    v: float
    left_parameter: float
    right_parameter: float
    kind: IntersectionKind
    branch_hint: int


@dataclass(frozen=True, slots=True)
class SketchIntersectionDiagnostic:
    """A typed reason why a curve pair has no unique discrete intersection."""

    kind: IntersectionDiagnosticKind
    left_curve_id: str
    right_curve_id: str
    message: str


@dataclass(frozen=True, slots=True)
class SketchIntersectionResult:
    intersections: tuple[SketchIntersection, ...] = ()
    diagnostics: tuple[SketchIntersectionDiagnostic, ...] = ()


@dataclass(frozen=True, slots=True)
class _CurveGeometry:
    curve: SketchCurve
    start: tuple[float, float] | None = None
    end: tuple[float, float] | None = None
    center: tuple[float, float] | None = None
    radius: float | None = None
    start_angle: float | None = None
    sweep: float | None = None


def intersect_sketch_curves(
    left: SketchCurve,
    right: SketchCurve,
    points: Mapping[str, SketchPoint | tuple[float, float]],
    *,
    tolerance: float = 1.0e-9,
) -> SketchIntersectionResult:
    """Intersect two finite strict sketch curves without display tessellation."""

    clean_tolerance = _validate_tolerance(tolerance)
    point_map = _coordinate_map(points)
    left_geometry, left_diagnostic = _curve_geometry(left, point_map, clean_tolerance)
    right_geometry, right_diagnostic = _curve_geometry(right, point_map, clean_tolerance)
    diagnostics = tuple(
        SketchIntersectionDiagnostic(
            item.kind,
            str(left.id),
            str(right.id),
            item.message,
        )
        for item in (left_diagnostic, right_diagnostic)
        if item is not None
    )
    if diagnostics:
        return SketchIntersectionResult(diagnostics=diagnostics)

    assert left_geometry is not None and right_geometry is not None
    if isinstance(left, SketchLine) and isinstance(right, SketchLine):
        raw, diagnostic = _line_line(left_geometry, right_geometry, clean_tolerance)
    elif isinstance(left, SketchLine):
        raw, diagnostic = _line_round(left_geometry, right_geometry, clean_tolerance)
    elif isinstance(right, SketchLine):
        swapped, diagnostic = _line_round(right_geometry, left_geometry, clean_tolerance)
        raw = tuple(
            (point, right_parameter, left_parameter, kind)
            for point, left_parameter, right_parameter, kind in swapped
        )
    else:
        raw, diagnostic = _round_round(left_geometry, right_geometry, clean_tolerance)
    if diagnostic is not None:
        return SketchIntersectionResult(
            diagnostics=(
                SketchIntersectionDiagnostic(
                    diagnostic,
                    str(left.id),
                    str(right.id),
                    _diagnostic_message(diagnostic, str(left.id), str(right.id)),
                ),
            )
        )

    ordered = sorted(raw, key=lambda item: (item[1], item[2], item[0][0], item[0][1]))
    unique: list[tuple[tuple[float, float], float, float, IntersectionKind]] = []
    for item in ordered:
        if unique and math.hypot(
            item[0][0] - unique[-1][0][0], item[0][1] - unique[-1][0][1]
        ) <= clean_tolerance:
            if item[3] == "tangent" and unique[-1][3] != "tangent":
                unique[-1] = item
            continue
        unique.append(item)
    intersections = tuple(
        SketchIntersection(
            str(left.id),
            str(right.id),
            point[0],
            point[1],
            _stable_parameter(left_parameter),
            _stable_parameter(right_parameter),
            kind,
            index,
        )
        for index, (point, left_parameter, right_parameter, kind) in enumerate(unique)
    )
    return SketchIntersectionResult(intersections)


def sketch_intersections(
    curves: tuple[SketchCurve, ...],
    points: Mapping[str, SketchPoint | tuple[float, float]],
    *,
    tolerance: float = 1.0e-9,
) -> SketchIntersectionResult:
    """Return stable pairwise intersections and all non-unique diagnostics."""

    intersections: list[SketchIntersection] = []
    diagnostics: list[SketchIntersectionDiagnostic] = []
    for index, left in enumerate(curves):
        for right in curves[index + 1 :]:
            result = intersect_sketch_curves(left, right, points, tolerance=tolerance)
            intersections.extend(result.intersections)
            diagnostics.extend(result.diagnostics)
    intersections.sort(
        key=lambda item: (
            item.left_curve_id.casefold(),
            item.right_curve_id.casefold(),
            item.left_parameter,
            item.right_parameter,
        )
    )
    return SketchIntersectionResult(tuple(intersections), tuple(diagnostics))


def _coordinate_map(
    points: Mapping[str, SketchPoint | tuple[float, float]],
) -> dict[str, tuple[float, float]]:
    result: dict[str, tuple[float, float]] = {}
    for point_id, value in points.items():
        coordinates = (value.u, value.v) if isinstance(value, SketchPoint) else value
        if len(coordinates) != 2:
            raise ValueError("sketch point coordinates must contain U and V")
        u, v = float(coordinates[0]), float(coordinates[1])
        if not math.isfinite(u) or not math.isfinite(v):
            raise ValueError("sketch point coordinates must be finite")
        result[str(point_id)] = (u, v)
    return result


def _curve_geometry(
    curve: SketchCurve,
    points: Mapping[str, tuple[float, float]],
    tolerance: float,
) -> tuple[_CurveGeometry | None, SketchIntersectionDiagnostic | None]:
    curve_id = str(curve.id)
    try:
        if isinstance(curve, SketchLine):
            start = points[curve.start_point_id]
            end = points[curve.end_point_id]
            if _distance(start, end) <= tolerance:
                return None, _degenerate(curve_id, "line has zero length")
            return _CurveGeometry(curve, start=start, end=end), None
        if isinstance(curve, SketchCircle):
            if not curve.is_curve or curve.center_point_id is None:
                return None, _degenerate(curve_id, "circle is not a strict curve")
            center = points[curve.center_point_id]
            if curve.radius <= tolerance:
                return None, _degenerate(curve_id, "circle has zero radius")
            return _CurveGeometry(curve, center=center, radius=curve.radius), None
        if isinstance(curve, SketchArc):
            start = points[curve.start_point_id]
            center = points[curve.center_point_id]
            end = points[curve.end_point_id]
            start_radius = _distance(start, center)
            end_radius = _distance(end, center)
            scale = max(1.0, start_radius, end_radius)
            if min(start_radius, end_radius) <= tolerance:
                return None, _degenerate(curve_id, "arc has zero radius")
            if abs(start_radius - end_radius) > tolerance * scale:
                return None, _degenerate(curve_id, "arc endpoint radii differ")
            start_angle = _angle(start, center)
            end_angle = _angle(end, center)
            sweep = (
                (end_angle - start_angle) % math.tau
                if curve.orientation == "ccw"
                else -((start_angle - end_angle) % math.tau)
            )
            if abs(sweep) <= tolerance / start_radius:
                return None, _degenerate(curve_id, "arc has zero sweep")
            return _CurveGeometry(
                curve,
                start=start,
                end=end,
                center=center,
                radius=0.5 * (start_radius + end_radius),
                start_angle=start_angle,
                sweep=sweep,
            ), None
    except KeyError as error:
        return None, _degenerate(curve_id, f"curve references missing point {error.args[0]!r}")
    raise TypeError(f"unsupported sketch curve: {type(curve).__name__}")


def _line_line(
    left: _CurveGeometry,
    right: _CurveGeometry,
    tolerance: float,
) -> tuple[
    tuple[tuple[tuple[float, float], float, float, IntersectionKind], ...],
    IntersectionDiagnosticKind | None,
]:
    assert left.start is not None and left.end is not None
    assert right.start is not None and right.end is not None
    left_vector = _subtract(left.end, left.start)
    right_vector = _subtract(right.end, right.start)
    offset = _subtract(right.start, left.start)
    denominator = _cross(left_vector, right_vector)
    scale = max(1.0, _length(left_vector) * _length(right_vector))
    if abs(denominator) <= tolerance * scale:
        if abs(_cross(offset, left_vector)) > tolerance * max(1.0, _length(left_vector)):
            return (), None
        start_parameter = _projection_parameter(left.start, left.end, right.start)
        end_parameter = _projection_parameter(left.start, left.end, right.end)
        low = max(0.0, min(start_parameter, end_parameter))
        high = min(1.0, max(start_parameter, end_parameter))
        parameter_tolerance = tolerance / max(_distance(left.start, left.end), tolerance)
        if high < low - parameter_tolerance:
            return (), None
        if high - low > parameter_tolerance:
            return (), "overlap"
        parameter = min(1.0, max(0.0, 0.5 * (low + high)))
        point = _point_at(left.start, left.end, parameter)
        return (
            (
                (
                    point,
                    parameter,
                    _projection_parameter(right.start, right.end, point),
                    "tangent",
                ),
            ),
            None,
        )
    left_parameter = _cross(offset, right_vector) / denominator
    right_parameter = _cross(offset, left_vector) / denominator
    if not (_in_unit(left_parameter, tolerance) and _in_unit(right_parameter, tolerance)):
        return (), None
    point = _point_at(left.start, left.end, left_parameter)
    return ((point, left_parameter, right_parameter, "crossing"),), None


def _line_round(
    line: _CurveGeometry,
    round_curve: _CurveGeometry,
    tolerance: float,
) -> tuple[
    tuple[tuple[tuple[float, float], float, float, IntersectionKind], ...],
    IntersectionDiagnosticKind | None,
]:
    assert line.start is not None and line.end is not None
    assert round_curve.center is not None and round_curve.radius is not None
    direction = _subtract(line.end, line.start)
    relative = _subtract(line.start, round_curve.center)
    a = _dot(direction, direction)
    closest_parameter = -_dot(relative, direction) / a
    closest_vector = (
        relative[0] + closest_parameter * direction[0],
        relative[1] + closest_parameter * direction[1],
    )
    radial_delta = round_curve.radius**2 - _dot(closest_vector, closest_vector)
    radial_tolerance = tolerance * max(1.0, round_curve.radius)
    if radial_delta < -radial_tolerance:
        return (), None
    tangent = abs(radial_delta) <= radial_tolerance
    parameter_offset = 0.0 if tangent else math.sqrt(max(0.0, radial_delta) / a)
    roots = (closest_parameter,) if tangent else (
        closest_parameter - parameter_offset,
        closest_parameter + parameter_offset,
    )
    result = []
    for line_parameter in roots:
        if not _in_unit(line_parameter, tolerance):
            continue
        point = _point_at(line.start, line.end, line_parameter)
        round_parameter = _round_parameter(round_curve, point, tolerance)
        if round_parameter is None:
            continue
        result.append(
            (
                point,
                line_parameter,
                round_parameter,
                "tangent" if tangent else "crossing",
            )
        )
    return tuple(result), None


def _round_round(
    left: _CurveGeometry,
    right: _CurveGeometry,
    tolerance: float,
) -> tuple[
    tuple[tuple[tuple[float, float], float, float, IntersectionKind], ...],
    IntersectionDiagnosticKind | None,
]:
    assert left.center is not None and left.radius is not None
    assert right.center is not None and right.radius is not None
    center_distance = _distance(left.center, right.center)
    scale = max(1.0, left.radius, right.radius, center_distance)
    scaled_tolerance = tolerance * scale
    if center_distance <= scaled_tolerance and abs(left.radius - right.radius) <= scaled_tolerance:
        return _coincident_round_curves(left, right, tolerance)
    if center_distance <= scaled_tolerance:
        return (), None
    if center_distance > left.radius + right.radius + scaled_tolerance:
        return (), None
    if center_distance < abs(left.radius - right.radius) - scaled_tolerance:
        return (), None
    axis = _subtract(right.center, left.center)
    along = (
        left.radius**2 - right.radius**2 + center_distance**2
    ) / (2.0 * center_distance)
    height_squared = left.radius**2 - along**2
    tangent = (
        abs(center_distance - (left.radius + right.radius)) <= scaled_tolerance
        or abs(center_distance - abs(left.radius - right.radius)) <= scaled_tolerance
    )
    height = 0.0 if tangent else math.sqrt(max(0.0, height_squared))
    unit = (axis[0] / center_distance, axis[1] / center_distance)
    base = (left.center[0] + along * unit[0], left.center[1] + along * unit[1])
    offsets = ((0.0, 0.0),) if tangent else (
        (-height * unit[1], height * unit[0]),
        (height * unit[1], -height * unit[0]),
    )
    result = []
    for offset in offsets:
        point = (base[0] + offset[0], base[1] + offset[1])
        left_parameter = _round_parameter(left, point, tolerance)
        right_parameter = _round_parameter(right, point, tolerance)
        if left_parameter is None or right_parameter is None:
            continue
        result.append(
            (
                point,
                left_parameter,
                right_parameter,
                "tangent" if tangent else "crossing",
            )
        )
    return tuple(result), None


def _coincident_round_curves(
    left: _CurveGeometry,
    right: _CurveGeometry,
    tolerance: float,
) -> tuple[
    tuple[tuple[tuple[float, float], float, float, IntersectionKind], ...],
    IntersectionDiagnosticKind | None,
]:
    left_arc = isinstance(left.curve, SketchArc)
    right_arc = isinstance(right.curve, SketchArc)
    if not left_arc and not right_arc:
        return (), "coincident"
    if left_arc != right_arc:
        return (), "overlap"
    assert left.radius is not None
    angular_tolerance = tolerance / max(left.radius, tolerance)
    overlap = _arc_overlap_length(left, right)
    if overlap > angular_tolerance:
        return (), "overlap"
    candidates = (left.start, left.end, right.start, right.end)
    unique: list[tuple[float, float]] = []
    for point in candidates:
        assert point is not None
        left_parameter = _round_parameter(left, point, tolerance)
        right_parameter = _round_parameter(right, point, tolerance)
        if left_parameter is None or right_parameter is None:
            continue
        if not any(_distance(point, existing) <= tolerance for existing in unique):
            unique.append(point)
    return tuple(
        (
            point,
            _round_parameter(left, point, tolerance),
            _round_parameter(right, point, tolerance),
            "tangent",
        )
        for point in unique
    ), None


def _arc_overlap_length(left: _CurveGeometry, right: _CurveGeometry) -> float:
    def ranges(geometry: _CurveGeometry) -> tuple[tuple[float, float], ...]:
        assert geometry.start_angle is not None and geometry.sweep is not None
        start = (
            geometry.start_angle
            if geometry.sweep > 0.0
            else geometry.start_angle + geometry.sweep
        )
        start %= math.tau
        end = start + abs(geometry.sweep)
        if end <= math.tau:
            return ((start, end),)
        return ((start, math.tau), (0.0, end - math.tau))

    return sum(
        max(0.0, min(left_end, right_end) - max(left_start, right_start))
        for left_start, left_end in ranges(left)
        for right_start, right_end in ranges(right)
    )


def _round_parameter(
    geometry: _CurveGeometry,
    point: tuple[float, float],
    tolerance: float,
) -> float | None:
    assert geometry.center is not None and geometry.radius is not None
    angle = _angle(point, geometry.center)
    if isinstance(geometry.curve, SketchCircle):
        return angle / math.tau
    assert geometry.start_angle is not None and geometry.sweep is not None
    travelled = (
        (angle - geometry.start_angle) % math.tau
        if geometry.sweep > 0.0
        else (geometry.start_angle - angle) % math.tau
    )
    total = abs(geometry.sweep)
    angular_tolerance = tolerance / max(geometry.radius, tolerance)
    if travelled > total + angular_tolerance:
        return None
    if travelled <= angular_tolerance:
        return 0.0
    if abs(travelled - total) <= angular_tolerance:
        return 1.0
    return travelled / total


def _degenerate(curve_id: str, detail: str) -> SketchIntersectionDiagnostic:
    return SketchIntersectionDiagnostic("degenerate", curve_id, curve_id, detail)


def _diagnostic_message(kind: IntersectionDiagnosticKind, left_id: str, right_id: str) -> str:
    descriptions = {
        "coincident": "coincident curves have infinitely many intersections",
        "overlap": "overlapping curves have no unique intersection set",
        "degenerate": "degenerate curve cannot be intersected",
    }
    return f"{left_id} and {right_id}: {descriptions[kind]}"


def _validate_tolerance(value: float) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError("tolerance must be a positive finite value")
    return result


def _stable_parameter(value: float) -> float:
    if abs(value) <= 1.0e-15:
        return 0.0
    if abs(value - 1.0) <= 1.0e-15:
        return 1.0
    return float(value)


def _in_unit(value: float, tolerance: float) -> bool:
    return -tolerance <= value <= 1.0 + tolerance


def _subtract(left: tuple[float, float], right: tuple[float, float]) -> tuple[float, float]:
    return left[0] - right[0], left[1] - right[1]


def _dot(left: tuple[float, float], right: tuple[float, float]) -> float:
    return left[0] * right[0] + left[1] * right[1]


def _cross(left: tuple[float, float], right: tuple[float, float]) -> float:
    return left[0] * right[1] - left[1] * right[0]


def _length(vector: tuple[float, float]) -> float:
    return math.hypot(*vector)


def _distance(left: tuple[float, float], right: tuple[float, float]) -> float:
    return math.hypot(left[0] - right[0], left[1] - right[1])


def _angle(point: tuple[float, float], center: tuple[float, float]) -> float:
    return math.atan2(point[1] - center[1], point[0] - center[0]) % math.tau


def _point_at(
    start: tuple[float, float], end: tuple[float, float], parameter: float
) -> tuple[float, float]:
    return (
        start[0] + parameter * (end[0] - start[0]),
        start[1] + parameter * (end[1] - start[1]),
    )


def _projection_parameter(
    start: tuple[float, float], end: tuple[float, float], point: tuple[float, float]
) -> float:
    direction = _subtract(end, start)
    return _dot(_subtract(point, start), direction) / _dot(direction, direction)


__all__ = [
    "IntersectionDiagnosticKind",
    "IntersectionKind",
    "SketchIntersection",
    "SketchIntersectionDiagnostic",
    "SketchIntersectionResult",
    "intersect_sketch_curves",
    "sketch_intersections",
]
