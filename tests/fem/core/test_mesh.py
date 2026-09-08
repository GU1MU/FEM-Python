import pytest

from fem.core.dof import DofMap
from fem.core.mesh import Element2D, Element3D, Mesh2D, Mesh3D, MeshProtocol, Node2D, Node3D


def test_plane_mesh_numbers_nodes_by_id_and_preserves_element_connectivity_order():
    mesh = Mesh2D(
        [Node2D(30, 0., 1.), Node2D(10, 0., 0.), Node2D(20, 1., 0.)],
        [Element2D(1, [30, 10, 20], "Tri3")],
    )

    assert isinstance(mesh, MeshProtocol)
    assert mesh.dofs_per_node == 2
    assert mesh.num_dofs == 6
    assert mesh.node_ids == [10, 20, 30]
    assert [mesh.node_dofs(node) for node in (10, 20, 30)] == [[0, 1], [2, 3], [4, 5]]
    assert mesh.element_dofs(mesh.elements[0]) == [4, 5, 0, 1, 2, 3]
    assert mesh.global_dof(30, 1) == 5


@pytest.mark.parametrize(
    ("element_type", "layout", "node_blocks", "element_blocks", "last_component", "last_dof"),
    [
        ("Truss2", {}, [[0, 1, 2], [3, 4, 5], [6, 7, 8]],
         [[6, 7, 8, 0, 1, 2], [0, 1, 2, 3, 4, 5]], 2, 8),
        ("Beam2", {"dofs_per_node": 6},
         [[0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11], [12, 13, 14, 15, 16, 17]],
         [[12, 13, 14, 15, 16, 17, 0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]],
         5, 17),
    ],
    ids=["default-translations", "beam-translations-and-rotations"],
)
def test_line_meshes_share_global_node_blocks_across_elements(
    element_type, layout, node_blocks, element_blocks, last_component, last_dof
):
    mesh = Mesh3D(
        [Node3D(30, 0., 0., 0.), Node3D(10, 1., 0., 0.), Node3D(20, 2., 0., 0.)],
        [Element3D(1, [30, 10], element_type), Element3D(2, [10, 20], element_type)],
        **layout,
    )

    assert isinstance(mesh, MeshProtocol)
    assert mesh.node_ids == [10, 20, 30]
    assert mesh.dofs_per_node == last_component + 1
    assert mesh.num_dofs == last_dof + 1
    assert [mesh.node_dofs(node) for node in (10, 20, 30)] == node_blocks
    assert [mesh.element_dofs(element) for element in mesh.elements] == element_blocks
    assert mesh.global_dof(30, last_component) == last_dof


def test_dof_map_rejects_duplicate_node_ids():
    with pytest.raises(ValueError, match="node ids must be unique"):
        DofMap.from_nodes([Node2D(1, 0., 0.), Node2D(1, 1., 0.)], dofs_per_node=2)


@pytest.mark.parametrize("component", [-1, 2])
def test_dof_map_rejects_components_outside_node_block(component):
    dof_map = DofMap.from_nodes([Node2D(10, 0., 0.)], dofs_per_node=2)

    with pytest.raises(IndexError, match="component .* out of range"):
        dof_map.global_dof(10, component)


def test_mesh_requires_positive_dofs_per_node():
    with pytest.raises(ValueError, match="dofs_per_node must be positive"):
        Mesh3D([], [], dofs_per_node=0)


def test_element3d_requires_an_explicit_type():
    with pytest.raises(TypeError):
        Element3D(1, [1, 2])
