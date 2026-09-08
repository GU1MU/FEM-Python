from __future__ import annotations

import math

import pytest

from fem.geometry._gmsh import predicates


def test_translated_coordinate_uses_local_scale_and_float_resolution() -> None:
    source = 1.0e12
    translation = 3.25

    assert predicates._matches_translated_coordinate(
        source,
        source + translation,
        translation,
        local_extent=2.0,
    )
    assert not predicates._matches_translated_coordinate(
        source,
        source + translation + 1.0e-2,
        translation,
        local_extent=2.0,
    )


def test_translated_signature_preserves_bounds_center_and_measure() -> None:
    source = (
        (1.0e9, -2.0e9, 3.0e9, 1.0e9 + 2.0, -2.0e9 + 3.0, 3.0e9 + 4.0),
        (1.0e9 + 1.0, -2.0e9 + 1.5, 3.0e9 + 2.0),
        24.0,
    )
    vector = (7.0, -11.0, 13.0)
    candidate = (
        tuple(
            value + vector[index % 3]
            for index, value in enumerate(source[0])
        ),
        tuple(
            value + vector[index] for index, value in enumerate(source[1])
        ),
        source[2],
    )

    assert predicates._matches_translated_signature(source, candidate, vector)
    assert not predicates._matches_translated_signature(
        source,
        (candidate[0], candidate[1], source[2] + 1.0e-4),
        vector,
    )


def test_axis_distance_and_rodrigues_rotation_support_displaced_axes() -> None:
    assert predicates._point_axis_distance(
        (4.0, 6.0, 10.0),
        (1.0, 2.0, 3.0),
        (0.0, 0.0, 2.0),
    ) == pytest.approx(5.0)
    assert predicates._rotate_point_about_axis(
        (2.0, 2.0, 3.0),
        (1.0, 2.0, 3.0),
        (0.0, 0.0, 2.0),
        math.pi / 2.0,
    ) == pytest.approx((1.0, 3.0, 3.0))


@pytest.mark.parametrize(
    "operation",
    (predicates._point_axis_distance, predicates._rotate_point_about_axis),
)
def test_axis_operations_reject_a_zero_axis(operation: object) -> None:
    arguments = ((1.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    if operation is predicates._rotate_point_about_axis:
        arguments = (*arguments, math.pi / 2.0)
    with pytest.raises(ValueError, match="^axis must be nonzero$"):
        operation(*arguments)  # type: ignore[operator]


def test_rotated_signature_matches_center_and_measure() -> None:
    source = ((0.0, 0.0, 0.0, 2.0, 4.0, 6.0), (1.0, 2.0, 3.0), 10.0)
    candidate = ((-4.0, 0.0, 0.0, 0.0, 2.0, 6.0), (-2.0, 1.0, 3.0), 10.0)
    axis_point = (0.0, 0.0, 0.0)
    axis = (0.0, 0.0, 1.0)
    angle = math.pi / 2.0

    assert predicates._matches_rotated_signature(
        source, candidate, axis_point, axis, angle
    )
    assert not predicates._matches_rotated_signature(
        source,
        (candidate[0], (-2.0, 1.01, 3.0), candidate[2]),
        axis_point,
        axis,
        angle,
    )
    assert not predicates._matches_rotated_signature(
        source,
        (candidate[0], candidate[1], candidate[2] + 1.0e-4),
        axis_point,
        axis,
        angle,
    )


def test_rigid_shape_signature_requires_matching_cardinality_order_and_measure() -> None:
    source = (10.0, (1.0, 2.0, 3.0))

    assert predicates._matches_rigid_shape_signature(
        source,
        (10.0 + 1.0e-11, (1.0, 2.0 + 1.0e-11, 3.0)),
    )
    assert not predicates._matches_rigid_shape_signature(source, (10.0, (1.0, 2.0)))
    assert not predicates._matches_rigid_shape_signature(
        source, (10.0, (1.0, 3.0, 2.0))
    )
    assert not predicates._matches_rigid_shape_signature(
        source, (10.001, (1.0, 2.0, 3.0))
    )


def test_vector_and_coordinate_primitives_preserve_standard_geometry() -> None:
    assert predicates._coordinate_distance((1.0, 2.0, 3.0), (4.0, 6.0, 3.0)) == 5.0
    assert predicates._vector_difference((4.0, 6.0, 8.0), (1.0, 2.0, 3.0)) == (
        3.0,
        4.0,
        5.0,
    )
    assert predicates._scale_vector((1.0, -2.0, 3.0), 2.5) == (
        2.5,
        -5.0,
        7.5,
    )
    assert predicates._vector_dot((1.0, 2.0, 3.0), (4.0, -5.0, 6.0)) == 12.0
    assert predicates._vector_cross((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)) == (
        0.0,
        0.0,
        1.0,
    )
    assert predicates._vector_norm((3.0, 4.0, 12.0)) == 13.0


def test_elliptical_arc_geometry_accepts_consistent_radii() -> None:
    predicates._validate_elliptical_arc_geometry(
        (
            (math.sqrt(3.0), 0.5, 0.0),
            (0.0, 0.0, 0.0),
            (2.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
        )
    )


@pytest.mark.parametrize(
    ("start", "end", "message"),
    (
        ((1.0, 0.0, 0.0), (-1.0, 0.0, 0.0), "both lie on the major axis"),
        (
            (math.sqrt(3.0), 0.5, 0.0),
            (0.0, 1.0, 0.1),
            "end must lie in the center/major-axis plane",
        ),
        (
            (math.sqrt(3.0), 0.5, 0.0),
            (math.sqrt(3.0), -0.5, 0.0),
            "endpoints must determine unique major and minor radii",
        ),
        (
            (3.0, 1.0, 0.0),
            (0.0, 0.5, 0.0),
            "endpoints must determine positive finite radii",
        ),
    ),
)
def test_elliptical_arc_geometry_preserves_degeneracy_messages(
    start: tuple[float, float, float],
    end: tuple[float, float, float],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        predicates._validate_elliptical_arc_geometry(
            (start, (0.0, 0.0, 0.0), (2.0, 0.0, 0.0), end)
        )


def test_plane_frames_support_fixed_xy_and_inclined_planes() -> None:
    xy_points = (
        (0.0, 0.0, 2.0),
        (1.0, 0.0, 2.0),
        (1.0, 1.0, 2.0),
        (0.0, 1.0, 2.0),
        (0.0, 0.0, 2.0),
    )
    xy_frame = predicates._plane_frame(
        xy_points, fixed_xy=True, operation="surface"
    )
    assert xy_frame[:4] == (
        xy_points[0],
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    )
    assert predicates._project_plane_points(
        xy_points, xy_frame, operation="surface"
    ) == (
        (0.0, 0.0),
        (1.0, 0.0),
        (1.0, 1.0),
        (0.0, 1.0),
        (0.0, 0.0),
    )

    inclined_points = (
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 1.0),
        (1.0, 1.0, 2.0),
        (0.0, 1.0, 1.0),
        (0.0, 0.0, 0.0),
    )
    inclined_frame = predicates._plane_frame(
        inclined_points, fixed_xy=False, operation="surface"
    )
    _, first_axis, second_axis, normal, _ = inclined_frame
    assert predicates._vector_norm(first_axis) == pytest.approx(1.0)
    assert predicates._vector_norm(second_axis) == pytest.approx(1.0)
    assert predicates._vector_norm(normal) == pytest.approx(1.0)
    assert predicates._vector_dot(first_axis, second_axis) == pytest.approx(0.0)
    assert predicates._vector_dot(first_axis, normal) == pytest.approx(0.0)
    predicates._project_plane_points(
        inclined_points, inclined_frame, operation="surface"
    )


@pytest.mark.parametrize(
    "points",
    (
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (2.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
        ),
    ),
)
def test_plane_frame_rejects_degenerate_point_sets(
    points: tuple[tuple[float, float, float], ...],
) -> None:
    with pytest.raises(ValueError, match="surface curve loops must enclose a nonzero area"):
        predicates._plane_frame(points, fixed_xy=False, operation="surface")


def test_plane_projection_rejects_non_coplanar_points() -> None:
    points = (
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (1.0, 1.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 0.0),
    )
    frame = predicates._plane_frame(points, fixed_xy=True, operation="surface")

    with pytest.raises(ValueError, match="surface curve loops must be coplanar"):
        predicates._project_plane_point(
            (0.5, 0.5, 1.0e-3), frame, operation="surface"
        )


def test_polyline_winding_distinguishes_orientation_boundary_and_ambiguity() -> None:
    diagonal = math.sqrt(0.5)
    polygon = (
        (1.0, 0.0),
        (diagonal, diagonal),
        (0.0, 1.0),
        (-diagonal, diagonal),
        (-1.0, 0.0),
        (-diagonal, -diagonal),
        (0.0, -1.0),
        (diagonal, -diagonal),
        (1.0, 0.0),
    )
    tolerance = 1.0e-12

    assert predicates._polyline_winding(polygon, (0.0, 0.0), tolerance) == 1
    assert predicates._polyline_winding(tuple(reversed(polygon)), (0.0, 0.0), tolerance) == -1
    assert predicates._polyline_winding(polygon, (3.0, 0.0), tolerance) == 0
    assert predicates._polyline_winding(polygon, polygon[0], tolerance) is None
    assert predicates._polyline_winding(
        ((1.0, 1.0), (-1.0, 1.0), (-1.0, -1.0), (1.0, -1.0), (1.0, 1.0)),
        (0.0, 0.0),
        tolerance,
    ) is None


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    (
        (((0.0, 0.0), (2.0, 2.0)), ((0.0, 2.0), (2.0, 0.0)), True),
        (((0.0, 0.0), (1.0, 0.0)), ((1.0, 0.0), (1.0, 1.0)), True),
        (((0.0, 0.0), (2.0, 0.0)), ((1.0, 0.0), (3.0, 0.0)), True),
        (((0.0, 0.0), (1.0, 0.0)), ((2.0, 0.0), (3.0, 0.0)), False),
    ),
)
def test_segment_contact_handles_crossing_tangent_collinear_and_separated_cases(
    left: tuple[tuple[float, float], tuple[float, float]],
    right: tuple[tuple[float, float], tuple[float, float]],
    expected: bool,
) -> None:
    assert predicates._segments_contact_2d(*left, *right, 1.0e-12) is expected


def test_polyline_self_contact_and_point_segment_distance() -> None:
    assert predicates._polyline_has_self_contact(
        ((0.0, 0.0), (2.0, 2.0), (0.0, 2.0), (2.0, 0.0), (0.0, 0.0)),
        1.0e-12,
    )
    assert not predicates._polyline_has_self_contact(
        ((0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0), (0.0, 0.0)),
        1.0e-12,
    )
    assert predicates._orientation_2d((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)) > 0.0
    assert predicates._point_segment_distance_2d(
        (3.0, 4.0), (0.0, 0.0), (0.0, 0.0)
    ) == 5.0
    assert predicates._point_segment_distance_2d(
        (1.0, 1.0), (0.0, 0.0), (2.0, 0.0)
    ) == 1.0
