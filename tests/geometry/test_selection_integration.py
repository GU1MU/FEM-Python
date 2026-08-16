from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from fem import geometry
from fem.core import Mesh2D
from fem.io import gmsh as gmsh_io
from fem.mesh import gmsh as gmsh_meshing
from fem.selection import curves, points, surfaces, volumes


_PRIMARY_EDGES = {
    "Tri3": ((0, 1), (1, 2), (2, 0)),
    "Tri6": ((0, 1), (1, 2), (2, 0)),
    "Quad4": ((0, 1), (1, 2), (2, 3), (3, 0)),
    "Quad8": ((0, 1), (1, 2), (2, 3), (3, 0)),
}


def _element_geometry(mesh: Mesh2D) -> tuple[tuple[np.ndarray, float], ...]:
    coordinates = {
        node.id: np.asarray((node.x, node.y), dtype=float)
        for node in mesh.nodes
    }
    records: list[tuple[np.ndarray, float]] = []
    for element in mesh.elements:
        primary_edges = _PRIMARY_EDGES.get(element.type)
        assert primary_edges is not None, (
            f"unsupported geometry-selection integration element {element.type!r}"
        )
        primary_node_count = max(max(edge) for edge in primary_edges) + 1
        corners = np.asarray(
            [
                coordinates[node_id]
                for node_id in element.node_ids[:primary_node_count]
            ],
            dtype=float,
        )
        edge_lengths = np.asarray(
            [
                np.linalg.norm(corners[end] - corners[start])
                for start, end in primary_edges
            ],
            dtype=float,
        )
        assert np.all(np.isfinite(edge_lengths))
        assert np.all(edge_lengths > 0.0)
        records.append(
            (np.mean(corners, axis=0), float(np.median(edge_lengths)))
        )
    assert records
    return tuple(records)


def _regional_median(
    records: tuple[tuple[np.ndarray, float], ...],
    selector: Callable[[np.ndarray], bool],
    *,
    label: str,
    minimum_cells: int = 8,
) -> float:
    sizes = [size for centroid, size in records if selector(centroid)]
    assert len(sizes) >= minimum_cells, (
        f"{label} region contains only {len(sizes)} cells"
    )
    return float(np.median(np.asarray(sizes, dtype=float)))


def test_real_curve_center_selection_drives_circular_hole_refinement(
    real_gmsh: Any,
) -> None:
    center = np.asarray((1.2, 1.0), dtype=float)
    radius = 0.25
    with geometry.model("selection_circular_hole", dimension=2) as cad:
        plate = cad.rectangle(0.0, 0.0, 4.0, 2.0)
        disk = cad.disk(*center, radius)
        domain = cad.cut([plate], [disk]).of_dimension(2)
        assert len(domain) == 1

        boundary = cad.boundary(domain)
        hole_curves = curves.by_center(
            cad,
            boundary,
            x=float(center[0]),
            y=float(center[1]),
        )
        assert len(boundary) == 5
        assert len(hole_curves) == 1
        assert cad.length(hole_curves[0]) == pytest.approx(
            2.0 * np.pi * radius
        )

        mesher = gmsh_meshing.Mesher(cad)
        distance = mesher.distance_field(curves=hole_curves, sampling=100)
        threshold = mesher.threshold_field(
            distance,
            size_min=0.045,
            size_max=0.36,
            dist_min=0.08,
            dist_max=0.65,
        )
        mesher.background_field(threshold)
        native_mesh = mesher.generate(gmsh_meshing.MeshSpec(order=1))
        mesh = gmsh_io.read(native_mesh)

    records = _element_geometry(mesh)
    near = _regional_median(
        records,
        lambda centroid: np.linalg.norm(centroid - center) - radius < 0.16,
        label="circular-opening near field",
    )
    far = _regional_median(
        records,
        lambda centroid: centroid[0] > 3.0,
        label="circular-opening far field",
    )

    assert near < 0.5 * far


def test_real_fragment_adjacency_selects_union_and_shared_curve(
    real_gmsh: Any,
) -> None:
    with geometry.model("selection_fragment_adjacency", dimension=2) as cad:
        left = cad.rectangle(0.0, 0.0, 1.0, 1.0)
        right = cad.rectangle(1.0, 0.0, 1.0, 1.0)
        fragmented = cad.fragment([left], [right]).of_dimension(2)
        assert len(fragmented) == 2

        individual_boundaries = tuple(
            cad.boundary((surface,), combined=False)
            for surface in fragmented
        )
        expected_union = cad.boundary(fragmented, combined=False)
        expected_shared = tuple(
            curve
            for curve in expected_union
            if all(curve in boundary for boundary in individual_boundaries)
        )

        selected_union = curves.adjacent_to(
            cad,
            fragmented,
            mode="any",
        )
        selected_shared = curves.adjacent_to(
            cad,
            fragmented,
            mode="all",
        )

        assert len(expected_union) == 7
        assert selected_union == expected_union
        assert len(expected_shared) == 1
        assert selected_shared == expected_shared


def test_real_volume_predicates_and_surface_adjacency_select_disjoint_boxes(
    real_gmsh: Any,
) -> None:
    with geometry.model("selection_disjoint_boxes", dimension=3) as cad:
        first = cad.box(0.0, 0.0, 0.0, 1.0, 2.0, 3.0)
        second = cad.box(4.0, 5.0, 6.0, 2.0, 1.0, 1.0)
        cases = (
            (
                first,
                (0.5, 1.0, 1.5),
                (0.0, 1.0, 0.0, 2.0, 0.0, 3.0),
                6.0,
            ),
            (
                second,
                (5.0, 5.5, 6.5),
                (4.0, 6.0, 5.0, 6.0, 6.0, 7.0),
                2.0,
            ),
        )

        for expected, center, bounds, measure in cases:
            assert volumes.by_center(
                cad,
                x=center[0],
                y=center[1],
                z=center[2],
            ) == (expected,)
            assert volumes.in_box(
                cad,
                xmin=bounds[0],
                xmax=bounds[1],
                ymin=bounds[2],
                ymax=bounds[3],
                zmin=bounds[4],
                zmax=bounds[5],
            ) == (expected,)
            assert volumes.by_volume(cad, value=measure) == (expected,)

        first_surfaces = surfaces.adjacent_to(cad, (first,))
        second_surfaces = surfaces.adjacent_to(cad, (second,))
        assert len(first_surfaces) == 6
        assert first_surfaces == cad.boundary((first,), combined=False)
        assert len(second_surfaces) == 6
        assert second_surfaces == cad.boundary((second,), combined=False)
        assert set(first_surfaces).isdisjoint(second_surfaces)
        assert surfaces.adjacent_to(
            cad,
            (first, second),
            mode="any",
        ) == cad.boundary((first, second), combined=False)


def test_real_box_intersection_and_measure_ranges_cover_each_cad_dimension(
    real_gmsh: Any,
) -> None:
    del real_gmsh
    with geometry.model("selection_v12_spatial_ranges", dimension=3) as cad:
        near_point = cad.point(1.0, 0.0, 0.0)
        far_point = cad.point(4.0, 0.0, 0.0)
        near_curve = cad.line(
            cad.point(0.0, 1.0, 0.0),
            cad.point(2.0, 1.0, 0.0),
        )
        far_curve = cad.line(
            cad.point(3.0, 1.0, 0.0),
            cad.point(4.0, 1.0, 0.0),
        )
        near_surface = cad.rectangle(0.0, 2.0, 2.0, 1.0)
        far_surface = cad.rectangle(3.0, 2.0, 1.0, 1.0)
        near_volume = cad.box(0.0, 4.0, 0.0, 2.0, 1.0, 1.0)
        far_volume = cad.box(3.0, 4.0, 0.0, 1.0, 1.0, 1.0)
        x_bounds = {"xmin": 0.75, "xmax": 1.25}

        assert points.intersects_box(
            cad,
            (far_point, near_point),
            **x_bounds,
        ) == (near_point,)
        assert curves.intersects_box(
            cad,
            (far_curve, near_curve),
            **x_bounds,
        ) == (near_curve,)
        assert surfaces.intersects_box(
            cad,
            (far_surface, near_surface),
            **x_bounds,
        ) == (near_surface,)
        assert volumes.intersects_box(
            cad,
            (far_volume, near_volume),
            **x_bounds,
        ) == (near_volume,)

        assert curves.in_box(cad, (near_curve,), **x_bounds) == ()
        assert curves.by_length_range(
            cad,
            (far_curve, near_curve),
            minimum=2.0,
            maximum=2.0,
        ) == (near_curve,)
        assert surfaces.by_area_range(
            cad,
            (far_surface, near_surface),
            minimum=2.0,
            maximum=2.0,
        ) == (near_surface,)
        assert volumes.by_volume_range(
            cad,
            (far_volume, near_volume),
            minimum=2.0,
            maximum=2.0,
        ) == (near_volume,)


def test_real_entity_distance_semantics_and_selectors(
    real_gmsh: Any,
) -> None:
    del real_gmsh
    with geometry.model("selection_v12_distance_semantics", dimension=3) as cad:
        origin = cad.point(0.0, 0.0, 0.0)
        shared = cad.point(1.0, 0.0, 0.0)
        touching_curve = cad.line(origin, shared)
        other_touching_curve = cad.line(shared, cad.point(1.0, 1.0, 0.0))
        near_curve = cad.line(
            cad.point(2.0, 0.0, 0.0),
            cad.point(2.0, 1.0, 0.0),
        )
        far_curve = cad.line(
            cad.point(4.0, 0.0, 0.0),
            cad.point(4.0, 1.0, 0.0),
        )
        plane = cad.rectangle(-1.0, -1.0, 2.0, 2.0)
        crossing_curve = cad.line(
            cad.point(0.0, 0.0, -1.0),
            cad.point(0.0, 0.0, 1.0),
        )
        contained_point = cad.point(0.25, 0.25, 0.25)
        container = cad.box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0)

        assert cad.distance(origin, far_curve) == pytest.approx(4.0)
        assert cad.distance(far_curve, origin) == pytest.approx(
            cad.distance(origin, far_curve)
        )
        assert cad.distance(origin, origin) == pytest.approx(0.0)
        assert cad.distance(touching_curve, other_touching_curve) == pytest.approx(
            0.0
        )
        assert cad.distance(crossing_curve, plane) == pytest.approx(0.0)
        assert cad.distance(contained_point, container) == pytest.approx(0.0)

        candidates = (far_curve, near_curve, touching_curve)
        assert curves.nearest_to(cad, origin, candidates) == touching_curve
        assert curves.within_distance(
            cad,
            origin,
            candidates,
            max_distance=2.0,
            tolerance=0.0,
        ) == (near_curve, touching_curve)


def test_real_distance_supports_all_ordered_dimension_pairs(
    real_gmsh: Any,
) -> None:
    del real_gmsh
    with geometry.model("selection_v12_distance_pairs", dimension=3) as cad:
        left = (
            cad.point(0.5, 0.5, 1.0),
            cad.line(
                cad.point(0.0, 0.5, 1.0),
                cad.point(1.0, 0.5, 1.0),
            ),
            cad.rectangle(0.0, 0.0, 1.0, 1.0, z=1.0),
            cad.box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0),
        )
        right = (
            cad.point(0.5, 0.5, 5.0),
            cad.line(
                cad.point(0.0, 0.5, 5.0),
                cad.point(1.0, 0.5, 5.0),
            ),
            cad.rectangle(0.0, 0.0, 1.0, 1.0, z=5.0),
            cad.box(0.0, 0.0, 5.0, 1.0, 1.0, 1.0),
        )

        assert tuple(entity.dimension for entity in left) == (0, 1, 2, 3)
        assert tuple(entity.dimension for entity in right) == (0, 1, 2, 3)
        for anchor in left:
            distances = cad.distances_to(anchor, right)
            assert distances == pytest.approx((4.0, 4.0, 4.0, 4.0))
            assert all(distance > 0.0 for distance in distances)
