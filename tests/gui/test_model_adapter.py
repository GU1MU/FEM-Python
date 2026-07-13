from __future__ import annotations

import pytest

from fem.core.mesh import Element2D, Element3D, HexMesh3D, Node2D, Node3D, PlaneMesh2D
from fem.core.model import FEMModel
from fem_gui.visualization.model_adapter import build_model_geometry


@pytest.mark.parametrize(
    ("element_type", "node_count", "vtk_type", "dimension"),
    [
        ("Truss2D", 2, 3, 2),
        ("Beam2D", 2, 3, 2),
        ("Tri3", 3, 5, 2),
        ("Tri6", 6, 22, 2),
        ("Quad4", 4, 9, 2),
        ("Quad8", 8, 23, 2),
        ("Tet4", 4, 10, 3),
        ("Tet10", 10, 24, 3),
        ("Hex8", 8, 12, 3),
        ("Hex20", 20, 25, 3),
    ],
)
def test_registered_element_topology_and_id_mapping(element_type, node_count, vtk_type, dimension):
    node_ids = [10 + 3 * index for index in range(node_count)]
    if dimension == 2:
        nodes = [Node2D(node_id, float(index), float(index % 2)) for index, node_id in enumerate(node_ids)]
        mesh = PlaneMesh2D(nodes, [Element2D(105, node_ids, element_type)])
    else:
        nodes = [Node3D(node_id, float(index), float(index % 2), float(index % 3)) for index, node_id in enumerate(node_ids)]
        mesh = HexMesh3D(nodes, [Element3D(105, node_ids, element_type)])

    geometry = build_model_geometry(FEMModel(mesh))

    assert geometry.points.shape == (node_count, 3)
    assert geometry.cells == (tuple(range(node_count)),)
    assert geometry.cell_types.tolist() == [vtk_type]
    assert geometry.node_id_to_point_index[node_ids[-1]] == node_count - 1
    assert geometry.point_index_to_node_id[0] == node_ids[0]
    assert geometry.element_id_to_cell_index == {105: 0}
    assert geometry.cell_index_to_element_id == {0: 105}


def test_mixed_mesh_keeps_connectivity_and_bidirectional_ids():
    nodes = [Node2D(node_id, float(index), float(index % 2)) for index, node_id in enumerate((11, 21, 31, 41, 51))]
    elements = [
        Element2D(70, [11, 21, 31], "Tri3"),
        Element2D(90, [21, 41, 51, 31], "Quad4"),
    ]

    geometry = build_model_geometry(FEMModel(PlaneMesh2D(nodes, elements)))

    assert geometry.cell_types.tolist() == [5, 9]
    assert geometry.cells == ((0, 1, 2), (1, 3, 4, 2))
    assert geometry.element_id_to_cell_index == {70: 0, 90: 1}
    assert geometry.cell_index_to_element_id == {0: 70, 1: 90}
