import pytest

from fem import selection
from fem.core.mesh import Element2D, Element3D, Mesh3D, Node2D, Node3D
from fem.core.model import Edge, ElementEdge, ElementFace, Surface
from tests.helpers.mesh_builders import (
    make_hex20_stiffness_mesh,
    make_selection_hex_mesh,
    make_selection_quad_mesh,
    make_tet10_stiffness_mesh,
    make_tet4_stiffness_mesh,
    make_tri6_load_mesh,
)


def test_faces_select_all_hex20_faces_with_quadratic_nodes():
    mesh = make_hex20_stiffness_mesh()

    selected = selection.faces.all(mesh)

    assert selected == [
        (1, 0, [1, 4, 3, 2, 12, 11, 10, 9]),
        (1, 1, [5, 6, 7, 8, 13, 14, 15, 16]),
        (1, 2, [1, 2, 6, 5, 9, 18, 13, 17]),
        (1, 3, [3, 4, 8, 7, 11, 20, 15, 19]),
        (1, 4, [1, 5, 8, 4, 17, 16, 20, 12]),
        (1, 5, [2, 3, 7, 6, 10, 19, 14, 18]),
    ]


def test_faces_do_not_select_reduced_integration_hex20():
    mesh = make_hex20_stiffness_mesh()
    mesh.elements[0].type = "C3D20R"

    assert selection.faces.all(mesh) == []


def test_boundary_faces_exclude_both_sides_of_a_shared_hex20_hex8_interface():
    hex20_mesh = make_hex20_stiffness_mesh()
    mesh = Mesh3D(
        nodes=[
            *hex20_mesh.nodes,
            Node3D(21, 0.0, 0.0, -1.0),
            Node3D(22, 1.0, 0.0, -1.0),
            Node3D(23, 1.0, 1.0, -1.0),
            Node3D(24, 0.0, 1.0, -1.0),
        ],
        elements=[
            hex20_mesh.elements[0],
            Element3D(2, [21, 22, 23, 24, 1, 2, 3, 4], type="Hex8"),
        ],
    )

    selected = selection.faces.boundary(mesh)

    assert len(selected) == 10
    assert not any(elem_id == 1 and local_face == 0 for elem_id, local_face, _ in selected)
    assert not any(elem_id == 2 and local_face == 1 for elem_id, local_face, _ in selected)


def test_quad_boundary_edges_exclude_the_shared_edge_and_build_named_load_edges():
    mesh = make_selection_quad_mesh()
    mesh.nodes.extend([Node2D(5, 2.0, 0.0), Node2D(6, 2.0, 1.0)])
    mesh.elements.append(Element2D(2, [2, 5, 6, 3], "Quad4"))
    mesh.rebuild_dof_map()

    assert selection.edges.boundary(mesh) == [
        (1, 0, [1, 2]), (1, 2, [3, 4]), (1, 3, [4, 1]),
        (2, 0, [2, 5]), (2, 1, [5, 6]), (2, 2, [6, 3]),
    ]
    assert selection.edges.by_x(mesh, 1.0) == []
    assert selection.edges.by_x(mesh, 1.0, boundary_only=False) == [
        (1, 1, [2, 3]), (2, 3, [3, 2]),
    ]
    assert selection.edges.by_x(mesh, 0.0) == [(1, 3, [4, 1])]
    assert selection.edges.by_y(mesh, 1.0) == [(1, 2, [3, 4]), (2, 2, [6, 3])]
    assert selection.edges.edge_by_y(mesh, "TOP", 1.0) == Edge(
        "TOP", [ElementEdge(1, 2, (3, 4)), ElementEdge(2, 2, (6, 3))],
    )


def test_tri6_edge_selection_preserves_the_midnode_between_endpoints():
    mesh = make_tri6_load_mesh()

    assert selection.edges.by_y(mesh, 0.0) == [(1, 0, [1, 4, 2])]
    assert selection.edges.edge_by_y(mesh, "BOTTOM", 0.0) == Edge(
        "BOTTOM", [ElementEdge(1, 0, (1, 4, 2))],
    )

    midpoint = next(node for node in mesh.nodes if node.id == 4)
    midpoint.y = 0.25
    assert selection.edges.by_y(mesh, 0.0) == []


@pytest.mark.parametrize(
    ("builder", "bottom_edges", "left_nodes"),
    [
        (make_tet4_stiffness_mesh,
         [(1, 0, [1, 2]), (1, 1, [2, 3]), (1, 2, [3, 1])], (3, 1)),
        (make_tet10_stiffness_mesh,
         [(1, 0, [1, 5, 2]), (1, 1, [2, 6, 3]), (1, 2, [3, 7, 1])], (3, 7, 1)),
    ],
    ids=["tet4", "tet10"],
)
def test_tetrahedron_edges_preserve_linear_and_quadratic_node_order(
    builder, bottom_edges, left_nodes,
):
    mesh = builder()

    assert selection.edges.by_z(mesh, 0.0) == bottom_edges
    assert selection.edges.edge_by_coord(mesh, "BOTTOM_LEFT", x=0.0, z=0.0) == Edge(
        "BOTTOM_LEFT", [ElementEdge(1, 2, left_nodes)],
    )


def test_hex_edges_select_coordinate_planes_and_their_intersection():
    mesh = make_selection_hex_mesh()
    left_edges = [
        (1, 3, [4, 1]), (1, 7, [8, 5]), (1, 8, [1, 5]), (1, 11, [4, 8]),
    ]

    assert selection.edges.by_x(mesh, 0.0) == left_edges
    assert selection.edges.edge_by_x(mesh, "LEFT", 0.0) == Edge("LEFT", [
        ElementEdge(1, 3, (4, 1)), ElementEdge(1, 7, (8, 5)),
        ElementEdge(1, 8, (1, 5)), ElementEdge(1, 11, (4, 8)),
    ])
    assert selection.edges.edge_by_z(mesh, "TOP", 4.0) == Edge("TOP", [
        ElementEdge(1, 4, (5, 6)), ElementEdge(1, 5, (6, 7)),
        ElementEdge(1, 6, (7, 8)), ElementEdge(1, 7, (8, 5)),
    ])
    assert selection.edges.edge_by_coord(mesh, "TOP_FRONT", y=0.0, z=4.0) == Edge(
        "TOP_FRONT", [ElementEdge(1, 4, (5, 6))],
    )


@pytest.mark.parametrize(
    ("axis", "value", "face_index", "nodes"),
    [("x", 2.0, 5, (2, 3, 7, 6)),
     ("y", 0.0, 2, (1, 2, 6, 5)),
     ("z", 4.0, 1, (5, 6, 7, 8))],
)
def test_hex_face_coordinate_queries_build_named_surfaces(axis, value, face_index, nodes):
    mesh = make_selection_hex_mesh()

    assert getattr(selection.faces, "by_" + axis)(mesh, value) == [
        (1, face_index, list(nodes)),
    ]
    assert getattr(selection.faces, "surface_by_" + axis)(mesh, "LOAD", value) == Surface(
        "LOAD", [ElementFace(1, face_index, nodes)],
    )


def test_2d_edges_are_not_selectable_as_model_surfaces():
    mesh = make_selection_quad_mesh()

    assert selection.faces.all(mesh) == []
    with pytest.raises(ValueError, match="2D meshes do not have model surfaces"):
        selection.faces.surface_by_x(mesh, "LEFT", 0.0)
