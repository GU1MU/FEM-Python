from __future__ import annotations

import numpy as np
import pytest

from fem_gui.visualization.symbols import (
    arc_points,
    camera_facing_offset,
    constraint_sample_indices,
    constraint_spatial_regions,
    constraint_symbol_dimensions,
    load_symbol_length,
    region_sample_indices,
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

    assert length == pytest.approx(0.15)
    assert symbol_length(thin_plate, 2.0) == pytest.approx(0.3)


def test_symbol_length_is_screen_limited_before_user_multiplier():
    thin_plate = np.array([[0.0, 0.0, 0.0], [100.0, 10.0, 0.001]])

    minimum_limited = symbol_length(thin_plate, world_per_pixel=0.1)
    maximum_limited = symbol_length(thin_plate, world_per_pixel=0.001)

    assert minimum_limited / 0.1 == pytest.approx(18.0)
    assert maximum_limited / 0.001 == pytest.approx(32.0)
    assert symbol_length(thin_plate, 2.0, world_per_pixel=0.1) == pytest.approx(3.6)


def test_constraint_symbol_is_larger_than_load_glyph_and_has_visible_width():
    length, radius = constraint_symbol_dimensions(20.0)

    assert length == pytest.approx(33.0)
    assert radius == pytest.approx(6.6)
    assert load_symbol_length(20.0) == pytest.approx(66.0)


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
