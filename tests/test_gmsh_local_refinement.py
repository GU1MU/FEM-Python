from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from fem.core import Mesh2D, Mesh3D
from fem import geometry
from fem.io import gmsh as gmsh_io
from fem.mesh import gmsh as gmsh_meshing


_PRIMARY_EDGES = {
    "Tri3": ((0, 1), (1, 2), (2, 0)),
    "Tri6": ((0, 1), (1, 2), (2, 0)),
    "Quad4": ((0, 1), (1, 2), (2, 3), (3, 0)),
    "Tet4": ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)),
}


def _element_geometry(
    mesh: Mesh2D | Mesh3D,
) -> tuple[tuple[np.ndarray, float], ...]:
    coordinates = {
        node.id: np.asarray(
            (node.x, node.y, getattr(node, "z", 0.0)),
            dtype=float,
        )
        for node in mesh.nodes
    }
    records: list[tuple[np.ndarray, float]] = []
    for element in mesh.elements:
        primary_edges = _PRIMARY_EDGES.get(element.type)
        assert primary_edges is not None, (
            f"unsupported local-refinement element family {element.type!r}"
        )
        primary_node_count = max(max(edge) for edge in primary_edges) + 1
        corners = np.asarray(
            [coordinates[node_id] for node_id in element.node_ids[:primary_node_count]],
            dtype=float,
        )
        edge_lengths = np.asarray(
            [np.linalg.norm(corners[end] - corners[start]) for start, end in primary_edges],
            dtype=float,
        )
        assert np.all(np.isfinite(edge_lengths))
        assert np.all(edge_lengths > 0.0)
        records.append((np.mean(corners, axis=0), float(np.median(edge_lengths))))
    assert records
    return tuple(records)


def _regional_median(
    records: tuple[tuple[np.ndarray, float], ...],
    selector: Callable[[np.ndarray], bool],
    *,
    label: str,
    minimum_cells: int = 4,
) -> float:
    sizes = [size for centroid, size in records if selector(centroid)]
    assert len(sizes) >= minimum_cells, (
        f"{label} region contains only {len(sizes)} cells"
    )
    return float(np.median(np.asarray(sizes, dtype=float)))


def _assert_positive_primary_jacobians(mesh: Mesh2D | Mesh3D) -> None:
    coordinates = {
        node.id: np.asarray(
            (node.x, node.y, getattr(node, "z", 0.0)),
            dtype=float,
        )
        for node in mesh.nodes
    }
    determinants: list[float] = []
    gauss_coordinate = 1.0 / np.sqrt(3.0)
    for element in mesh.elements:
        if element.type in {"Tri3", "Tri6"}:
            points = np.asarray(
                [coordinates[node_id][:2] for node_id in element.node_ids[:3]],
                dtype=float,
            )
            jacobian = np.column_stack((points[1] - points[0], points[2] - points[0]))
            determinants.append(float(np.linalg.det(jacobian)))
        elif element.type == "Quad4":
            points = np.asarray(
                [coordinates[node_id][:2] for node_id in element.node_ids[:4]],
                dtype=float,
            )
            for xi in (-gauss_coordinate, gauss_coordinate):
                for eta in (-gauss_coordinate, gauss_coordinate):
                    derivatives = 0.25 * np.asarray(
                        (
                            (-(1.0 - eta), -(1.0 - xi)),
                            (1.0 - eta, -(1.0 + xi)),
                            (1.0 + eta, 1.0 + xi),
                            (-(1.0 + eta), 1.0 - xi),
                        ),
                        dtype=float,
                    )
                    determinants.append(float(np.linalg.det(points.T @ derivatives)))
        elif element.type == "Tet4":
            points = np.asarray(
                [coordinates[node_id] for node_id in element.node_ids[:4]],
                dtype=float,
            )
            jacobian = np.column_stack(
                (points[1] - points[0], points[2] - points[0], points[3] - points[0])
            )
            determinants.append(float(np.linalg.det(jacobian)))
        else:
            pytest.fail(
                f"unsupported Jacobian element family {element.type!r}"
            )

    determinant_array = np.asarray(determinants, dtype=float)
    assert determinant_array.size >= mesh.num_elements
    assert np.all(np.isfinite(determinant_array))
    assert np.all(determinant_array > 0.0)


def test_real_selected_point_mesh_size_refines_one_rectangle_corner(
    real_gmsh: Any,
) -> None:
    external_options = {
        "Mesh.ElementOrder": 2.0,
        "Mesh.SecondOrderIncomplete": 1.0,
        "Mesh.RecombineAll": 1.0,
        "Mesh.MeshSizeFromPoints": 0.0,
        "Mesh.MeshSizeFromCurvature": 1.0,
        "Mesh.MeshSizeExtendFromBoundary": 0.0,
        "Mesh.MeshSizeMin": 0.21,
        "Mesh.MeshSizeMax": 0.29,
        "Mesh.MeshSizeFactor": 1.7,
    }
    for name, value in external_options.items():
        real_gmsh.option.setNumber(name, value)

    with geometry.model("selected_point_refinement", dimension=2) as cad:
        domain = cad.rectangle(0.0, 0.0, 3.0, 1.5)
        boundary = cad.boundary([domain])
        points = cad.boundary(boundary, combined=False)
        refined_point = cad.select(points, x=0.0, y=0.0)
        assert len(points) == 4
        assert len(refined_point) == 1

        mesher = gmsh_meshing.Mesher(cad)
        mesher.mesh_size(points, size=0.45)
        mesher.mesh_size(refined_point, size=0.05)
        native_mesh = mesher.generate(gmsh_meshing.MeshSpec(order=2))
        mesh = gmsh_io.read(native_mesh)

        for name, value in external_options.items():
            assert real_gmsh.option.getNumber(name) == pytest.approx(value)

    records = _element_geometry(mesh)
    near = _regional_median(
        records,
        lambda centroid: centroid[0] < 0.32 and centroid[1] < 0.32,
        label="selected-point near field",
    )
    far = _regional_median(
        records,
        lambda centroid: centroid[0] > 2.0 and centroid[1] > 0.55,
        label="selected-point far field",
    )

    assert near < 0.55 * far
    assert {element.type for element in mesh.elements} == {"Tri6"}
    _assert_positive_primary_jacobians(mesh)


def test_real_curve_threshold_refines_a_circular_hole_with_valid_cells(
    real_gmsh: Any,
) -> None:
    center = np.asarray((1.2, 1.0), dtype=float)
    radius = 0.25
    with geometry.model("circular_hole_refinement", dimension=2) as cad:
        plate = cad.rectangle(0.0, 0.0, 4.0, 2.0)
        disk = cad.disk(*center, radius)
        domain = cad.cut([plate], [disk]).of_dimension(2)
        assert len(domain) == 1

        boundary = cad.boundary(domain)
        outer = tuple(
            dict.fromkeys(
                cad.select(boundary, x=0.0)
                + cad.select(boundary, x=4.0)
                + cad.select(boundary, y=0.0)
                + cad.select(boundary, y=2.0)
            )
        )
        hole = tuple(curve for curve in boundary if curve not in outer)
        assert len(boundary) == 5
        assert len(outer) == 4
        assert len(hole) == 1

        mesher = gmsh_meshing.Mesher(cad)
        distance = mesher.distance_field(curves=hole, sampling=100)
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
        lambda centroid: np.linalg.norm(centroid[:2] - center) - radius < 0.16,
        label="circular-hole near field",
        minimum_cells=8,
    )
    far = _regional_median(
        records,
        lambda centroid: centroid[0] > 3.0,
        label="circular-hole far field",
        minimum_cells=8,
    )

    assert near < 0.5 * far
    assert {element.type for element in mesh.elements} == {"Tri3"}
    _assert_positive_primary_jacobians(mesh)


def test_real_min_field_refines_two_regions_with_entity_recombination(
    real_gmsh: Any,
) -> None:
    width = 6.0
    with geometry.model("two_region_min_refinement", dimension=2) as cad:
        domain = cad.rectangle(0.0, 0.0, width, 2.0)
        boundary = cad.boundary([domain])
        left = cad.select(boundary, x=0.0)
        right = cad.select(boundary, x=width)
        assert len(left) == 1
        assert len(right) == 1

        mesher = gmsh_meshing.Mesher(cad)
        left_distance = mesher.distance_field(curves=left, sampling=40)
        left_threshold = mesher.threshold_field(
            left_distance,
            size_min=0.08,
            size_max=0.48,
            dist_min=0.12,
            dist_max=1.1,
        )
        right_distance = mesher.distance_field(curves=right, sampling=40)
        right_threshold = mesher.threshold_field(
            right_distance,
            size_min=0.08,
            size_max=0.48,
            dist_min=0.12,
            dist_max=1.1,
        )
        combined = mesher.min_field([left_threshold, right_threshold])
        mesher.background_field(combined)
        mesher.recombine(domain)
        native_mesh = mesher.generate(
            gmsh_meshing.MeshSpec(recombine=False)
        )
        mesh = gmsh_io.read(native_mesh)

    records = _element_geometry(mesh)
    near_left = _regional_median(
        records,
        lambda centroid: centroid[0] < 0.32,
        label="left Min-field neighborhood",
        minimum_cells=8,
    )
    near_right = _regional_median(
        records,
        lambda centroid: centroid[0] > width - 0.32,
        label="right Min-field neighborhood",
        minimum_cells=8,
    )
    far = _regional_median(
        records,
        lambda centroid: abs(centroid[0] - 0.5 * width) < 0.55,
        label="Min-field far field",
        minimum_cells=8,
    )

    assert near_left < 0.6 * far
    assert near_right < 0.6 * far
    assert {element.type for element in mesh.elements} <= {
        "Tri3",
        "Quad4",
    }
    assert "Quad4" in {element.type for element in mesh.elements}
    _assert_positive_primary_jacobians(mesh)


def test_real_surface_distance_refines_one_face_of_a_small_box(
    real_gmsh: Any,
) -> None:
    length = 2.0
    with geometry.model("surface_distance_refinement", dimension=3) as cad:
        volume = cad.box(0.0, 0.0, 0.0, length, 0.8, 0.8)
        faces = cad.boundary([volume])
        refined_face = cad.select(faces, x=0.0)
        assert len(refined_face) == 1

        mesher = gmsh_meshing.Mesher(cad)
        distance = mesher.distance_field(surfaces=refined_face, sampling=40)
        threshold = mesher.threshold_field(
            distance,
            size_min=0.13,
            size_max=0.5,
            dist_min=0.12,
            dist_max=0.75,
        )
        mesher.background_field(threshold)
        native_mesh = mesher.generate(gmsh_meshing.MeshSpec(order=1))
        mesh = gmsh_io.read(native_mesh)

    records = _element_geometry(mesh)
    near = _regional_median(
        records,
        lambda centroid: centroid[0] < 0.3,
        label="surface-distance near field",
        minimum_cells=8,
    )
    far = _regional_median(
        records,
        lambda centroid: centroid[0] > 1.35,
        label="surface-distance far field",
        minimum_cells=8,
    )

    assert near < 0.65 * far
    assert {element.type for element in mesh.elements} == {"Tet4"}
    _assert_positive_primary_jacobians(mesh)


def test_real_mesher_binding_seals_geometry_mutation_after_size_controls(
    real_gmsh: Any,
) -> None:
    with geometry.model("distance_topology_dependency_guard", dimension=2) as cad:
        plate = cad.rectangle(0.0, 0.0, 2.0, 1.0)
        disk = cad.disk(3.0, 0.5, 0.25)
        assert cad.translate([disk], 0.25, 0.0, 0.0) == (disk,)
        disk_curves = cad.boundary([disk])
        mesher = gmsh_meshing.Mesher(cad)
        distance = mesher.distance_field(curves=disk_curves, sampling=40)

        with pytest.raises(
            geometry.GeometryStateError,
            match="CONFIGURING_MESH",
        ):
            cad.translate([disk], 0.25, 0.0, 0.0)
        with pytest.raises(
            geometry.GeometryStateError,
            match="CONFIGURING_MESH",
        ):
            cad.cut([plate], [disk])

        threshold = mesher.threshold_field(
            distance,
            size_min=0.08,
            size_max=0.35,
            dist_min=0.1,
            dist_max=0.6,
        )
        mesher.background_field(threshold)
        native_mesh = mesher.generate(gmsh_meshing.MeshSpec())
        field_mesh = gmsh_io.read(native_mesh)

    assert {element.type for element in field_mesh.elements} == {"Tri3"}
    _assert_positive_primary_jacobians(field_mesh)

    with geometry.model("point_size_topology_dependency_guard", dimension=2) as cad:
        left = cad.rectangle(0.0, 0.0, 1.0, 1.0)
        right = cad.rectangle(2.0, 0.0, 1.0, 1.0)
        points = cad.boundary(cad.boundary([left, right]), combined=False)
        unrelated = cad.rectangle(4.0, 0.0, 1.0, 1.0)
        overlapping_tool = cad.rectangle(4.5, 0.0, 1.0, 1.0)
        assert cad.fuse([unrelated], [overlapping_tool]).outputs
        mesher = gmsh_meshing.Mesher(cad)
        mesher.mesh_size(points, size=0.15)

        with pytest.raises(
            geometry.GeometryStateError,
            match="CONFIGURING_MESH",
        ):
            cad.translate([left], 0.25, 0.0, 0.0)

        with pytest.raises(
            geometry.GeometryStateError,
            match="CONFIGURING_MESH",
        ):
            cad.fuse([left], [right])

        native_mesh = mesher.generate(gmsh_meshing.MeshSpec())
        point_mesh = gmsh_io.read(native_mesh)

    assert {element.type for element in point_mesh.elements} == {"Tri3"}
    _assert_positive_primary_jacobians(point_mesh)


def test_real_mesher_binding_seals_transform_after_recombine(
    real_gmsh: Any,
) -> None:
    with geometry.model("recombine_transform_dependency_guard", dimension=2) as cad:
        surface = cad.rectangle(0.0, 0.0, 1.0, 1.0)
        mesher = gmsh_meshing.Mesher(cad)
        mesher.recombine(surface)

        with pytest.raises(
            geometry.GeometryStateError,
            match="CONFIGURING_MESH",
        ):
            cad.rotate([surface], 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.25)

        native_mesh = mesher.generate(gmsh_meshing.MeshSpec(size=0.2))
        mesh = gmsh_io.read(native_mesh)

    assert {element.type for element in mesh.elements} == {"Quad4"}
    _assert_positive_primary_jacobians(mesh)


def test_real_consecutive_controlled_extrusions_allow_preserving_topology(
    real_gmsh: Any,
) -> None:
    with geometry.model("consecutive_controlled_extrusions", dimension=3) as cad:
        first_surface = cad.rectangle(0.0, 0.0, 1.0, 1.0)
        second_surface = cad.rectangle(2.0, 0.0, 1.0, 1.0)
        assert cad.translate([second_surface], 0.0, 0.25, 0.0) == (
            second_surface,
        )
        mesher = gmsh_meshing.Mesher(cad)
        first_extrusion = mesher.structured_extrude(
            [first_surface],
            0.0,
            0.0,
            1.0,
            num_elements=(2,),
            recombine=True,
        )
        first_volumes = first_extrusion.primary
        assert len(first_volumes) == 1

        second_extrusion = mesher.structured_extrude(
            [second_surface],
            0.0,
            0.0,
            1.0,
            num_elements=(3,),
            heights=(1.0,),
            recombine=True,
        )
        second_volumes = second_extrusion.primary
        assert len(second_volumes) == 1

        native_mesh = mesher.generate(
            gmsh_meshing.MeshSpec(size=0.5, order=1, recombine=True)
        )
        mesh = gmsh_io.read(native_mesh)

    assert isinstance(mesh, Mesh3D)
    assert mesh.num_elements > 0
    assert {element.type for element in mesh.elements} == {"Hex8"}


def test_real_controlled_extrusion_accepts_repeated_shared_output(
    real_gmsh: Any,
) -> None:
    with geometry.model("shared_controlled_extrusion_output", dimension=3) as cad:
        left = cad.rectangle(0.0, 0.0, 1.0, 1.0)
        right = cad.rectangle(1.0, 0.0, 1.0, 1.0)
        fragmented = cad.fragment([left], [right])
        surfaces = tuple(
            dict.fromkeys(
                entity for entity in fragmented.outputs if entity.dimension == 2
            )
        )

        mesher = gmsh_meshing.Mesher(cad)
        result = mesher.structured_extrude(
            surfaces,
            0.0,
            0.0,
            1.0,
            num_elements=(1,),
            recombine=True,
        )

    output_keys = tuple(
        (entity.dimension, entity.tag) for entity in result.outputs
    )
    assert len(output_keys) > len(set(output_keys))
    assert len({key for key in output_keys if key[0] == 3}) == 2
