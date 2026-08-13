from __future__ import annotations

import numpy as np
import pytest

from fem.application.results import (
    ResultArchiveModelProjection,
    ResultSourceKey,
    ResultTopologyProjection,
)
from fem.core.mesh import Element2D, Element3D, Mesh2D, Mesh3D, Node2D, Node3D
from fem.core.model import FEMModel
from fem.post.fields import ResultRegionKey, make_result_region_signature
import fem_gui.visualization.model_adapter as model_adapter_module
from fem_gui.visualization.model_adapter import (
    build_model_geometry,
    build_result_archive_geometry,
)


@pytest.mark.parametrize(
    ("element_type", "node_count", "vtk_type", "dimension"),
    [
        ("Truss2", 2, 3, 3),
        ("Beam2", 2, 3, 3),
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
        mesh = Mesh2D(nodes, [Element2D(105, node_ids, element_type)])
    else:
        nodes = [Node3D(node_id, float(index), float(index % 2), float(index % 3)) for index, node_id in enumerate(node_ids)]
        mesh = Mesh3D(
            nodes,
            [Element3D(105, node_ids, element_type)],
            dofs_per_node=6 if element_type == "Beam2" else 3,
        )

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

    geometry = build_model_geometry(FEMModel(Mesh2D(nodes, elements)))

    assert geometry.cell_types.tolist() == [5, 9]
    assert geometry.cells == ((0, 1, 2), (1, 3, 4, 2))
    assert geometry.element_id_to_cell_index == {70: 0, 90: 1}
    assert geometry.cell_index_to_element_id == {0: 70, 1: 90}
    np.testing.assert_array_equal(geometry.node_ids, (11, 21, 31, 41, 51))
    np.testing.assert_array_equal(geometry.element_ids, (70, 90))


def test_geometry_builder_avoids_legacy_nested_cell_projection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mesh = Mesh2D(
        [Node2D(10, 0.0, 0.0), Node2D(20, 1.0, 0.0)],
        [Element2D(30, [10, 20], "Truss2")],
    )

    def fail_legacy_build(_mesh: object) -> object:
        raise AssertionError("geometry must build its flat VTK array directly")

    monkeypatch.setattr(model_adapter_module.vtk_cells, "build", fail_legacy_build)

    geometry = build_model_geometry(FEMModel(mesh))

    np.testing.assert_array_equal(geometry.cell_array, (2, 0, 1))


def test_archive_geometry_keeps_inverse_node_and_element_maps() -> None:
    source = ResultSourceKey(
        "result-1", "session-1", "artifact-1", 1, "Step-1", "run-1"
    )
    signature = make_result_region_signature({})
    projection = ResultArchiveModelProjection(
        ResultTopologyProjection(
            source=source,
            node_ids=(10, 20, 30),
            node_coordinates=np.asarray(
                ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
            ),
            nodal_displacements=np.zeros((3, 3)),
            element_ids=(99,),
            element_types=("Tri3",),
            connectivity=((10, 20, 30),),
            element_region_keys=(ResultRegionKey(signature, signature),),
        )
    )

    geometry = build_result_archive_geometry(projection)

    assert geometry.point_index_to_node_id == {0: 10, 1: 20, 2: 30}
    assert geometry.cell_index_to_element_id == {0: 99}
