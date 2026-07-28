from __future__ import annotations

import numpy as np
import pytest

from fem_gui.visualization.symbols import (
    arc_points,
    camera_facing_offset,
    constraint_outward_direction,
    constraint_rotation_axes,
    constraint_sample_indices,
    constraint_spatial_regions,
    constraint_symbol_dimensions,
    load_arrow_origins,
    load_symbol_length,
    region_sample_indices,
    rotation_lock_points,
    sample_face,
    sample_polyline,
    symbol_length,
)


def test_polyline_sampling_follows_actual_high_order_edge_path():
    points = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.0], [1.0, 0.0, 0.0]])

    samples = sample_polyline(points, "medium")

    assert samples.shape == (3, 3)
    assert samples[1] == pytest.approx([0.5, 0.5, 0.0])


def test_face_sampling_density_changes_real_symbol_count():
    points = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]]
    )

    assert sample_face(points, "low").shape[0] == 1
    assert sample_face(points, "medium").shape[0] == 5
    assert sample_face(points, "high").shape[0] == 5


def test_distributed_region_sampling_is_capped_and_not_linear_with_refinement():
    coarse = np.column_stack((np.linspace(0.0, 10.0, 200), np.zeros(200), np.zeros(200)))
    fine = np.column_stack((np.linspace(0.0, 10.0, 20_000), np.zeros(20_000), np.zeros(20_000)))

    coarse_count = len(region_sample_indices(coarse, "medium"))
    fine_count = len(region_sample_indices(fine, "medium"))

    assert coarse_count == 12
    assert fine_count == 12
    assert len(region_sample_indices(fine, "high")) == 24


def test_continuous_support_sampling_is_more_sparse_than_load_sampling():
    points = np.column_stack((np.linspace(0.0, 10.0, 200), np.zeros(200), np.zeros(200)))

    assert len(constraint_sample_indices(points, "low")) == 3
    assert len(constraint_sample_indices(points, "medium")) == 6
    assert len(constraint_sample_indices(points, "high")) == 12


def test_continuous_supports_are_sampled_along_their_boundary():
    x, y = np.meshgrid(np.linspace(-1.0, 1.0, 9), np.linspace(-1.0, 1.0, 9))
    points = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size)))

    selected = points[constraint_sample_indices(points, "medium")]

    assert len(selected) == 4
    assert np.all(np.isclose(np.abs(selected[:, 0]), 1.0) | np.isclose(np.abs(selected[:, 1]), 1.0))


def test_rectangular_support_keeps_all_four_corners_at_low_density():
    x, y = np.meshgrid(np.linspace(-1.0, 1.0, 9), np.linspace(-1.0, 1.0, 9))
    points = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size)))

    selected = points[constraint_sample_indices(points, "low")]

    assert len(selected) == 4
    assert {
        tuple(point) for point in selected
    } == {
        (-1.0, -1.0, 0.0), (-1.0, 1.0, 0.0),
        (1.0, -1.0, 0.0), (1.0, 1.0, 0.0),
    }


def test_constraint_depth_offset_follows_the_camera_ray():
    point = np.array((1.0, -2.0, 3.0))

    assert camera_facing_offset(point, np.array((1.0, -2.0, 8.0)), 0.5) == pytest.approx((0.0, 0.0, 0.5))
    assert camera_facing_offset(point, np.array((1.0, -2.0, -2.0)), 0.5) == pytest.approx((0.0, 0.0, -0.5))
    assert camera_facing_offset(point, None, 0.5) == pytest.approx((0.0, 0.0, 0.0))


def test_distant_end_supports_receive_independent_four_corner_regions():
    face = np.array((
        (0.0, -1.0, -1.0), (0.0, 1.0, -1.0),
        (0.0, 1.0, 1.0), (0.0, -1.0, 1.0),
        (0.0, 0.0, -1.0), (0.0, 0.0, 1.0),
    ))
    points = np.vstack((face, face + np.array((100.0, 0.0, 0.0))))
    model_points = np.array(((0.0, -1.0, -1.0), (100.0, 1.0, 1.0)))

    regions = constraint_spatial_regions(points, model_points)
    selected = [
        region[constraint_sample_indices(points[region], "medium")]
        for region in regions
    ]

    assert tuple(len(region) for region in regions) == (6, 6)
    assert tuple(len(indices) for indices in selected) == (4, 4)


def test_line_support_sampling_covers_both_ends_and_intermediate_positions():
    points = np.column_stack((np.linspace(0.0, 10.0, 101), np.zeros(101), np.zeros(101)))

    selected = points[constraint_sample_indices(points, "medium")]

    assert selected[0, 0] == pytest.approx(0.0)
    assert selected[-1, 0] == pytest.approx(10.0)
    assert np.all(np.diff(selected[:, 0]) > 0.0)


def test_symbol_length_uses_effective_sides_for_thin_models():
    thin_plate = np.array([[0.0, 0.0, 0.0], [100.0, 10.0, 0.001]])
    length = symbol_length(thin_plate)

    assert length == pytest.approx(0.4)
    assert symbol_length(thin_plate, 2.0) == pytest.approx(0.8)


def test_symbol_length_is_screen_limited_before_user_multiplier():
    thin_plate = np.array([[0.0, 0.0, 0.0], [100.0, 10.0, 0.001]])

    minimum_limited = symbol_length(thin_plate, world_per_pixel=0.1)
    maximum_limited = symbol_length(thin_plate, world_per_pixel=0.001)

    assert minimum_limited / 0.1 == pytest.approx(24.0)
    assert maximum_limited / 0.001 == pytest.approx(56.0)
    assert symbol_length(thin_plate, 2.0, world_per_pixel=0.1) == pytest.approx(4.8)


def test_symbol_grows_on_screen_as_the_camera_zooms_in():
    line = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])

    fitted = symbol_length(line, world_per_pixel=0.02) / 0.02
    zoomed = symbol_length(line, world_per_pixel=0.01) / 0.01
    close_up = symbol_length(line, world_per_pixel=0.005) / 0.005

    assert fitted == pytest.approx(24.0)
    assert zoomed == pytest.approx(40.0)
    assert close_up == pytest.approx(56.0)


def test_constraint_symbol_is_compact_and_has_a_slender_visible_width():
    length, radius = constraint_symbol_dimensions(20.0)

    assert length == pytest.approx(27.0)
    assert radius == pytest.approx(3.24)
    assert load_symbol_length(20.0) == pytest.approx(66.0)


def test_constraint_marker_uses_the_nearest_exterior_axis_side():
    center = np.array((5.0, 5.0, 0.0))

    assert constraint_outward_direction(
        np.array((0.0, 10.0, 0.0)), center, 0
    ) == pytest.approx((-1.0, 0.0, 0.0))
    assert constraint_outward_direction(
        np.array((0.0, 10.0, 0.0)), center, 1
    ) == pytest.approx((0.0, 1.0, 0.0))
    assert constraint_outward_direction(
        np.array((0.0, 0.0, 0.0)), center, 1
    ) == pytest.approx((0.0, -1.0, 0.0))


def test_edge_load_can_align_its_arrow_start_while_other_loads_align_the_tip():
    origins = load_arrow_origins(
        anchors=np.array(((10.0, 1.0, 0.0), (3.0, 2.0, 0.0))),
        directions=np.array(((1.0, 0.0, 0.0), (0.0, -1.0, 0.0))),
        lengths=np.array((2.0, 1.0)),
        start_aligned=np.array((True, False)),
    )

    assert origins == pytest.approx(np.array(((10.0, 1.0, 0.0), (3.0, 3.0, 0.0))))


def test_region_sampling_starts_near_the_region_center_and_is_mesh_order_independent():
    x, y = np.meshgrid(np.linspace(-1.0, 1.0, 9), np.linspace(-1.0, 1.0, 9))
    points = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size)))
    shuffled = points[np.random.default_rng(4).permutation(len(points))]

    selected = points[region_sample_indices(points, "medium")]
    shuffled_selected = shuffled[region_sample_indices(shuffled, "medium")]

    assert np.any(np.all(np.isclose(selected, [0.0, 0.0, 0.0]), axis=1))
    assert np.any(np.all(np.isclose(shuffled_selected, [0.0, 0.0, 0.0]), axis=1))
    assert np.ptp(selected[:, 0]) == pytest.approx(2.0)
    assert np.ptp(selected[:, 1]) == pytest.approx(2.0)


def test_moment_arc_carries_axis_and_rotation_direction():
    positive = arc_points(np.zeros(3), np.array([0.0, 0.0, 1.0]), 1.0)
    negative = arc_points(np.zeros(3), np.array([0.0, 0.0, -1.0]), 1.0)

    assert positive.shape == (19, 3)
    assert np.allclose(positive[:, 2], 0.0)
    assert not np.allclose(positive, negative)


def test_rotation_constraint_is_a_closed_crossed_ring_normal_to_its_axis():
    center = np.array((1.0, 2.0, 3.0))
    ring, bars = rotation_lock_points(center, np.array((1.0, 0.0, 0.0)), 2.0)

    assert ring.shape == (25, 3)
    assert bars.shape == (4, 3)
    assert ring[0] == pytest.approx(ring[-1])
    assert np.allclose(ring[:, 0], center[0])
    assert np.allclose(np.mean(bars.reshape((2, 2, 3)), axis=1), center)


def test_full_3d_rotation_lock_collapses_to_one_camera_facing_symbol():
    axes = constraint_rotation_axes(
        (0, 1, 2, 3, 4, 5),
        is_3d=True,
        point=np.zeros(3),
        camera_position=np.array((0.0, -4.0, 3.0)),
    )

    assert axes.shape == (1, 3)
    assert axes[0] == pytest.approx((0.0, -0.8, 0.6))


def test_partial_rotation_lock_keeps_its_physical_axes():
    axes = constraint_rotation_axes(
        (3, 5),
        is_3d=True,
        point=np.zeros(3),
        camera_position=None,
    )

    assert np.allclose(axes, ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0)))
