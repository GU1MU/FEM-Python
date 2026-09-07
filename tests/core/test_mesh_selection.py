from copy import deepcopy

import pytest

from fem import selection
from fem.core.mesh import Element3D, Mesh2D, Mesh3D, Node2D, Node3D
from fem.core.model import ElementSet, NodeSet
from tests.helpers.mesh_builders import make_mixed_tri3_quad4_mesh, make_selection_hex_mesh


def test_node_coordinates_match_all_requested_axes_with_inclusive_tolerance():
    mesh2d = Mesh2D(
        [Node2D(30, 1.125, 2.0), Node2D(10, 1.0, 0.0), Node2D(20, 1.25, 2.0)],
        [],
    )
    mesh3d = Mesh3D(
        [Node3D(1, 0.0, 0.0, 0.0), Node3D(2, 1.0, 0.0, 0.0), Node3D(3, 1.0, 2.0, 3.0)],
        [],
    )
    original_nodes = deepcopy(mesh2d.nodes)

    assert selection.nodes.by_x(mesh2d, 1.0, tol=0.125) == [30, 10]
    assert selection.nodes.by_coord(mesh2d, x=1.0, y=2.0, tol=0.125) == [30]
    assert selection.nodes.by_y(mesh2d, 2.0) == [30, 20]
    assert selection.nodes.by_z(mesh3d, 3.0) == [3]
    assert selection.nodes.by_coord(mesh3d, x=1.0, z=0.0) == [2]
    assert mesh2d.nodes == original_nodes


def test_node_coordinate_selection_builds_a_named_set():
    mesh = make_selection_hex_mesh()

    assert selection.nodes.set_by_x(mesh, "FIXED", 0.0) == NodeSet("FIXED", (1, 4, 5, 8))


def test_element_id_selection_intersects_requests_in_mesh_order():
    mesh = make_mixed_tri3_quad4_mesh()
    mesh.elements.reverse()
    original_elements = deepcopy(mesh.elements)

    assert selection.elements.all(mesh) == [2, 1]
    assert selection.elements.by_ids(mesh, [1, 99, 2, 1]) == [2, 1]
    assert selection.elements.by_type(mesh, "quad4") == [2]
    assert selection.elements.set_by_type(mesh, "QUADS", "quad4") == ElementSet("QUADS", (2,))
    assert mesh.elements == original_elements


@pytest.mark.parametrize(
    ("node_ids", "mode", "expected"),
    [
        ([1, 2, 2, 3, 4], "all", [30, 10]),
        ([2, 4], "any", [30, 10, 20]),
        ([], "all", []),
        ([], "any", []),
    ],
    ids=["all-connectivity-nodes", "any-connectivity-node", "empty-all", "empty-any"],
)
def test_node_membership_selection_is_independent_of_registered_element_types(node_ids, mode, expected):
    mesh = Mesh3D(
        nodes=[Node3D(node_id, float(node_id), 0.0, 0.0) for node_id in range(1, 9)],
        elements=[
            Element3D(30, [1, 2], "unregistered_line"),
            Element3D(10, [2, 3, 4], "unregistered_plane"),
            Element3D(20, [4, 5, 6, 7], "unregistered_solid"),
        ],
    )
    original_elements = deepcopy(mesh.elements)

    assert selection.elements.by_nodes(mesh, node_ids, mode=mode) == expected
    assert selection.elements.set_by_nodes(mesh, "REGION", node_ids, mode=mode) == ElementSet(
        "REGION", tuple(expected),
    )
    assert mesh.elements == original_elements


def test_node_membership_selection_rejects_unknown_mode():
    with pytest.raises(ValueError, match="mode must be 'all' or 'any'"):
        selection.elements.by_nodes(make_mixed_tri3_quad4_mesh(), [1], mode="some")


@pytest.mark.parametrize(
    ("requested", "expected"),
    [("Truss2", [1]), ("Beam2", [5]), ("Quad4", [4]), ("CPS4", [4]), ("Truss2Extended", [])],
    ids=["truss", "beam", "canonical-quad", "quad-alias", "unknown-prefix-extension"],
)
def test_element_type_selection_matches_aliases_but_not_prefix_extensions(requested, expected):
    mesh = Mesh3D(
        nodes=[],
        elements=[
            Element3D(1, [], "Truss2"),
            Element3D(2, [], "Truss2Extended"),
            Element3D(4, [], "CPS4"),
            Element3D(5, [], "Beam2"),
            Element3D(6, [], "Beam2Extended"),
        ],
    )

    assert selection.elements.by_type(mesh, requested) == expected


@pytest.fixture
def spatial_node_mesh():
    return Mesh3D(
        [
            Node3D(30, 0.0, 0.0, 5.0),
            Node3D(10, 1.0, 0.0, 0.0),
            Node3D(40, 0.0, 2.0, 1.0),
            Node3D(20, -1.0, 1.0, 2.0),
        ],
        [],
    )


def test_node_box_selection_includes_limits_and_leaves_none_limits_open(spatial_node_mesh):
    assert selection.nodes.in_box(
        spatial_node_mesh, xmin=0.0, xmax=1.0, ymin=0.0, ymax=2.0,
    ) == [30, 10, 40]
    assert selection.nodes.in_box(
        spatial_node_mesh, xmin=None, xmax=0.0, zmin=1.0, zmax=None,
    ) == [30, 40, 20]


def test_circle_uses_xy_distance_and_nearest_uses_z_only_when_supplied(spatial_node_mesh):
    assert selection.nodes.in_circle(spatial_node_mesh, x=0.0, y=0.0, r=1.0) == [30, 10]
    assert selection.nodes.nearest(spatial_node_mesh, x=0.0, y=0.0) == 30
    assert selection.nodes.nearest(spatial_node_mesh, x=0.0, y=0.0, z=0.0) == 10


def test_node_boundary_groups_coordinate_extrema(spatial_node_mesh):
    assert selection.nodes.boundary(spatial_node_mesh) == {
        "left": [20],
        "right": [10],
        "bottom": [30, 10],
        "top": [40],
        "back": [10],
        "front": [30],
    }
