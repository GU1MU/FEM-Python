from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from fem import geometry
from fem.core import Mesh2D
from fem.io import gmsh as gmsh_io
from fem.mesh import gmsh as gmsh_meshing
from fem.selection import curves, surfaces, volumes


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
