"""Pure geometric predicates used by the Gmsh geometry implementation."""

from __future__ import annotations

import math

from .constants import (
    _OCC_BOUNDING_BOX_PADDING,
    _PLANAR_TOLERANCE,
)

_Point2D = tuple[float, float]
_Point3D = tuple[float, float, float]
_PlaneFrame = tuple[_Point3D, _Point3D, _Point3D, _Point3D, float]
_GeometrySignature = tuple[
    tuple[float, float, float, float, float, float],
    _Point3D,
    float,
]
_RigidShapeSignature = tuple[float, tuple[float, ...]]


def _matches_translated_signature(
    source: tuple[
        tuple[float, float, float, float, float, float],
        tuple[float, float, float],
        float,
    ],
    candidate: tuple[
        tuple[float, float, float, float, float, float],
        tuple[float, float, float],
        float,
    ],
    vector: tuple[float, float, float],
) -> bool:
    source_bounds, source_center, source_measure = source
    candidate_bounds, candidate_center, candidate_measure = candidate
    source_extent = max(
        source_bounds[axis + 3] - source_bounds[axis] for axis in range(3)
    )
    for axis in range(3):
        for bound_index in (axis, axis + 3):
            if not _matches_translated_coordinate(
                source_bounds[bound_index],
                candidate_bounds[bound_index],
                vector[axis],
                local_extent=source_extent,
                absolute_padding=_OCC_BOUNDING_BOX_PADDING,
            ):
                return False
        if not _matches_translated_coordinate(
            source_center[axis],
            candidate_center[axis],
            vector[axis],
            local_extent=source_extent,
        ):
            return False
    return abs(candidate_measure - source_measure) <= _PLANAR_TOLERANCE * max(
        1.0,
        abs(candidate_measure),
        abs(source_measure),
    )


def _matches_translated_coordinate(
    source: float,
    candidate: float,
    translation: float,
    *,
    local_extent: float,
    absolute_padding: float = 0.0,
) -> bool:
    # Scale modeling tolerance with local geometry only. Absolute world
    # coordinates contribute solely their unavoidable floating-point resolution.
    floating_resolution = 4.0 * max(
        math.ulp(source),
        math.ulp(candidate),
        math.ulp(translation),
    )
    tolerance = (
        absolute_padding
        + _PLANAR_TOLERANCE * max(1.0, local_extent)
        + floating_resolution
    )
    return abs((candidate - source) - translation) <= tolerance


def _point_axis_distance(
    point: _Point3D,
    axis_point: _Point3D,
    axis: _Point3D,
) -> float:
    relative = tuple(
        value - origin for value, origin in zip(point, axis_point, strict=True)
    )
    axis_norm = _vector_norm(axis)
    if axis_norm == 0.0:
        raise ValueError("axis must be nonzero")
    return _vector_norm(_vector_cross(relative, axis)) / axis_norm


def _rotate_point_about_axis(
    point: _Point3D,
    axis_point: _Point3D,
    axis: _Point3D,
    angle: float,
) -> _Point3D:
    axis_norm = _vector_norm(axis)
    if axis_norm == 0.0:
        raise ValueError("axis must be nonzero")
    unit_axis = tuple(component / axis_norm for component in axis)
    relative = tuple(
        value - origin for value, origin in zip(point, axis_point, strict=True)
    )
    cosine = math.cos(angle)
    sine = math.sin(angle)
    cross = _vector_cross(unit_axis, relative)
    projection = _vector_dot(unit_axis, relative)
    rotated_relative = tuple(
        relative[index] * cosine
        + cross[index] * sine
        + unit_axis[index] * projection * (1.0 - cosine)
        for index in range(3)
    )
    return tuple(
        axis_point[index] + rotated_relative[index] for index in range(3)
    )  # type: ignore[return-value]


def _matches_rotated_signature(
    source: _GeometrySignature,
    candidate: _GeometrySignature,
    axis_point: _Point3D,
    axis: _Point3D,
    angle: float,
) -> bool:
    source_bounds, source_center, source_measure = source
    candidate_bounds, candidate_center, candidate_measure = candidate
    expected_center = _rotate_point_about_axis(
        source_center,
        axis_point,
        axis,
        angle,
    )
    source_extent = max(
        source_bounds[index + 3] - source_bounds[index] for index in range(3)
    )
    candidate_extent = max(
        candidate_bounds[index + 3] - candidate_bounds[index]
        for index in range(3)
    )
    floating_resolution = 4.0 * max(
        *(math.ulp(value) for value in (*source_center, *candidate_center)),
        math.ulp(angle),
    )
    tolerance = (
        _OCC_BOUNDING_BOX_PADDING
        + _PLANAR_TOLERANCE * max(1.0, source_extent, candidate_extent)
        + floating_resolution
    )
    if _coordinate_distance(expected_center, candidate_center) > tolerance:
        return False
    return abs(candidate_measure - source_measure) <= _PLANAR_TOLERANCE * max(
        1.0,
        abs(candidate_measure),
        abs(source_measure),
    )


def _matches_rigid_shape_signature(
    source: _RigidShapeSignature,
    candidate: _RigidShapeSignature,
) -> bool:
    source_measure, source_boundaries = source
    candidate_measure, candidate_boundaries = candidate
    if len(source_boundaries) != len(candidate_boundaries):
        return False
    values = (
        (source_measure, candidate_measure),
        *zip(
            source_boundaries,
            candidate_boundaries,
            strict=True,
        ),
    )
    return all(
        abs(right - left)
        <= _PLANAR_TOLERANCE * max(1.0, abs(left), abs(right))
        for left, right in values
    )


def _coordinate_distance(
    first: tuple[float, float, float],
    second: tuple[float, float, float],
) -> float:
    return math.sqrt(
        sum((left - right) ** 2 for left, right in zip(first, second, strict=True))
    )


def _validate_elliptical_arc_geometry(
    coordinates: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ],
) -> None:
    start, center, major_axis_point, end = coordinates
    axis = _vector_difference(major_axis_point, center)
    start_vector = _vector_difference(start, center)
    end_vector = _vector_difference(end, center)
    axis_unit = _scale_vector(axis, 1.0 / _vector_norm(axis))

    start_perpendicular = _vector_difference(
        start_vector,
        _scale_vector(axis_unit, _vector_dot(start_vector, axis_unit)),
    )
    end_perpendicular = _vector_difference(
        end_vector,
        _scale_vector(axis_unit, _vector_dot(end_vector, axis_unit)),
    )
    angular_tolerance = 1.0e-6
    start_perpendicular_norm = _vector_norm(start_perpendicular)
    end_perpendicular_norm = _vector_norm(end_perpendicular)
    if start_perpendicular_norm > angular_tolerance * _vector_norm(start_vector):
        minor_unit = _scale_vector(
            start_perpendicular,
            1.0 / start_perpendicular_norm,
        )
    elif end_perpendicular_norm > angular_tolerance * _vector_norm(end_vector):
        minor_unit = _scale_vector(
            end_perpendicular,
            1.0 / end_perpendicular_norm,
        )
    else:
        raise ValueError(
            "elliptical_arc start and end must not both lie on the major axis"
        )

    normal_unit = _vector_cross(axis_unit, minor_unit)
    for label, vector in (("start", start_vector), ("end", end_vector)):
        if abs(_vector_dot(vector, normal_unit)) > (
            angular_tolerance * _vector_norm(vector)
        ):
            raise ValueError(
                f"elliptical_arc {label} must lie in the center/major-axis plane"
            )

    start_major_squared = _vector_dot(start_vector, axis_unit) ** 2
    start_minor_squared = _vector_dot(start_vector, minor_unit) ** 2
    end_major_squared = _vector_dot(end_vector, axis_unit) ** 2
    end_minor_squared = _vector_dot(end_vector, minor_unit) ** 2
    equality_tolerance = _PLANAR_TOLERANCE**2
    if math.isclose(
        start_major_squared,
        end_major_squared,
        rel_tol=1.0e-12,
        abs_tol=equality_tolerance,
    ) or math.isclose(
        start_minor_squared,
        end_minor_squared,
        rel_tol=1.0e-12,
        abs_tol=equality_tolerance,
    ):
        raise ValueError(
            "elliptical_arc endpoints must determine unique major and minor radii"
        )

    major_radius_squared = (
        start_minor_squared * end_major_squared
        - start_major_squared * end_minor_squared
    ) / (start_minor_squared - end_minor_squared)
    minor_radius_squared = (
        start_major_squared * end_minor_squared
        - start_minor_squared * end_major_squared
    ) / (start_major_squared - end_major_squared)
    if (
        not math.isfinite(major_radius_squared)
        or not math.isfinite(minor_radius_squared)
        or major_radius_squared <= 0.0
        or minor_radius_squared <= 0.0
    ):
        raise ValueError(
            "elliptical_arc endpoints must determine positive finite radii"
        )


def _vector_difference(
    left: tuple[float, float, float],
    right: tuple[float, float, float],
) -> tuple[float, float, float]:
    return tuple(
        left_value - right_value
        for left_value, right_value in zip(left, right, strict=True)
    )  # type: ignore[return-value]


def _scale_vector(
    vector: tuple[float, float, float],
    factor: float,
) -> tuple[float, float, float]:
    return tuple(value * factor for value in vector)  # type: ignore[return-value]


def _vector_dot(
    left: tuple[float, float, float],
    right: tuple[float, float, float],
) -> float:
    return sum(
        left_value * right_value
        for left_value, right_value in zip(left, right, strict=True)
    )


def _vector_cross(
    left: tuple[float, float, float],
    right: tuple[float, float, float],
) -> tuple[float, float, float]:
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def _vector_norm(vector: tuple[float, float, float]) -> float:
    return math.sqrt(_vector_dot(vector, vector))


def _plane_frame(
    points: tuple[_Point3D, ...],
    *,
    fixed_xy: bool,
    operation: str,
) -> _PlaneFrame:
    if len(points) < 4:
        raise ValueError(f"{operation} curve loops must enclose a nonzero area")
    origin = points[0]
    relative = tuple(_vector_difference(point, origin) for point in points[1:])
    scale = max(1.0, *(_vector_norm(vector) for vector in relative))
    tolerance = _PLANAR_TOLERANCE * scale
    if fixed_xy:
        return (
            origin,
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            tolerance,
        )

    first_axis = max(relative, key=_vector_norm)
    first_axis_norm = _vector_norm(first_axis)
    if first_axis_norm <= tolerance:
        raise ValueError(f"{operation} curve loops must enclose a nonzero area")
    first_unit = _scale_vector(first_axis, 1.0 / first_axis_norm)
    plane_vector = max(
        relative,
        key=lambda vector: _vector_norm(_vector_cross(first_unit, vector)),
    )
    normal = _vector_cross(first_unit, plane_vector)
    normal_norm = _vector_norm(normal)
    if normal_norm <= tolerance:
        raise ValueError(f"{operation} curve loops must enclose a nonzero area")
    normal_unit = _scale_vector(normal, 1.0 / normal_norm)
    second_unit = _vector_cross(normal_unit, first_unit)
    return origin, first_unit, second_unit, normal_unit, tolerance


def _project_plane_point(
    point: _Point3D,
    frame: _PlaneFrame,
    *,
    operation: str,
) -> _Point2D:
    origin, first_axis, second_axis, normal, tolerance = frame
    relative = _vector_difference(point, origin)
    if abs(_vector_dot(relative, normal)) > tolerance:
        raise ValueError(f"{operation} curve loops must be coplanar")
    return _vector_dot(relative, first_axis), _vector_dot(relative, second_axis)


def _project_plane_points(
    points: tuple[_Point3D, ...],
    frame: _PlaneFrame,
    *,
    operation: str,
) -> tuple[_Point2D, ...]:
    return tuple(
        _project_plane_point(point, frame, operation=operation) for point in points
    )


def _polyline_winding(
    points: tuple[_Point2D, ...],
    probe: _Point2D,
    tolerance: float,
) -> int | None:
    total_angle = 0.0
    for start, end in zip(points, points[1:]):
        if math.dist(start, end) <= tolerance:
            continue
        if _point_segment_distance_2d(probe, start, end) <= tolerance:
            return None
        start_vector = (start[0] - probe[0], start[1] - probe[1])
        end_vector = (end[0] - probe[0], end[1] - probe[1])
        if math.hypot(*start_vector) <= tolerance or math.hypot(*end_vector) <= tolerance:
            return None
        angle = math.atan2(
            start_vector[0] * end_vector[1] - start_vector[1] * end_vector[0],
            start_vector[0] * end_vector[0] + start_vector[1] * end_vector[1],
        )
        if abs(angle) >= math.pi / 2.0:
            return None
        total_angle += angle
    normalized = total_angle / (2.0 * math.pi)
    nearest = round(normalized)
    if abs(normalized - nearest) > 1.0e-8:
        return None
    return nearest


def _polyline_has_self_contact(
    points: tuple[_Point2D, ...],
    tolerance: float,
) -> bool:
    segment_count = len(points) - 1
    for left_index in range(segment_count):
        left_start = points[left_index]
        left_end = points[left_index + 1]
        if math.dist(left_start, left_end) <= tolerance:
            continue
        for right_index in range(left_index + 1, segment_count):
            if right_index == left_index + 1 or (
                left_index == 0 and right_index == segment_count - 1
            ):
                continue
            right_start = points[right_index]
            right_end = points[right_index + 1]
            if math.dist(right_start, right_end) <= tolerance:
                continue
            if _segments_contact_2d(
                left_start,
                left_end,
                right_start,
                right_end,
                tolerance,
            ):
                return True
    return False


def _segments_contact_2d(
    left_start: _Point2D,
    left_end: _Point2D,
    right_start: _Point2D,
    right_end: _Point2D,
    tolerance: float,
) -> bool:
    if (
        max(left_start[0], left_end[0]) + tolerance
        < min(right_start[0], right_end[0])
        or max(right_start[0], right_end[0]) + tolerance
        < min(left_start[0], left_end[0])
        or max(left_start[1], left_end[1]) + tolerance
        < min(right_start[1], right_end[1])
        or max(right_start[1], right_end[1]) + tolerance
        < min(left_start[1], left_end[1])
    ):
        return False
    left_first = _orientation_2d(left_start, left_end, right_start)
    left_second = _orientation_2d(left_start, left_end, right_end)
    right_first = _orientation_2d(right_start, right_end, left_start)
    right_second = _orientation_2d(right_start, right_end, left_end)
    if left_first * left_second < 0.0 and right_first * right_second < 0.0:
        return True
    return min(
        _point_segment_distance_2d(right_start, left_start, left_end),
        _point_segment_distance_2d(right_end, left_start, left_end),
        _point_segment_distance_2d(left_start, right_start, right_end),
        _point_segment_distance_2d(left_end, right_start, right_end),
    ) <= tolerance


def _orientation_2d(start: _Point2D, end: _Point2D, point: _Point2D) -> float:
    return (end[0] - start[0]) * (point[1] - start[1]) - (
        end[1] - start[1]
    ) * (point[0] - start[0])


def _point_segment_distance_2d(
    point: _Point2D,
    start: _Point2D,
    end: _Point2D,
) -> float:
    delta_x = end[0] - start[0]
    delta_y = end[1] - start[1]
    length_squared = delta_x * delta_x + delta_y * delta_y
    if length_squared == 0.0:
        return math.dist(point, start)
    parameter = (
        (point[0] - start[0]) * delta_x + (point[1] - start[1]) * delta_y
    ) / length_squared
    parameter = min(1.0, max(0.0, parameter))
    closest = (start[0] + parameter * delta_x, start[1] + parameter * delta_y)
    return math.dist(point, closest)
