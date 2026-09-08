import numpy as np
import pytest

from fem.boundary.condition import BoundaryCondition
from fem.boundary.loads import build_load_vector
from fem.elements import get_element_kernel
from tests.helpers.mesh_builders import (
    make_tri3_load_mesh, make_tri6_load_mesh,
    make_quad4_boundary_mesh, make_quad8_load_mesh,
    make_tet4_stiffness_mesh, make_tet10_stiffness_mesh,
    make_hex8_stiffness_mesh, make_hex20_stiffness_mesh,
)


@pytest.mark.parametrize(
    ("element_type", "expected_faces"),
    [
        ("Hex8", [
            [0, 3, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4],
            [2, 3, 7, 6], [0, 4, 7, 3], [1, 2, 6, 5],
        ]),
        ("Hex20", [
            [0, 3, 2, 1, 11, 10, 9, 8], [4, 5, 6, 7, 12, 13, 14, 15],
            [0, 1, 5, 4, 8, 17, 12, 16], [2, 3, 7, 6, 10, 19, 14, 18],
            [0, 4, 7, 3, 16, 15, 19, 11], [1, 2, 6, 5, 9, 18, 13, 17],
        ]),
        ("Tet4", [[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]]),
        ("Tet10", [
            [1, 2, 3, 5, 9, 8], [0, 2, 3, 6, 9, 7],
            [0, 1, 3, 4, 8, 7], [0, 1, 2, 4, 5, 6],
        ]),
    ],
)
def test_solid_face_topology_preserves_corner_and_midside_node_order(
    element_type, expected_faces,
):
    actual = get_element_kernel(element_type).face_node_indices
    assert tuple(tuple(face) for face in actual) == tuple(tuple(face) for face in expected_faces)


@pytest.mark.parametrize(
    ("builder", "body_weights", "edge_weights"),
    [
        (make_tri3_load_mesh, [2/3, 2/3, 2/3], [2, 2, 0]),
        (make_tri6_load_mesh, [0, 0, 0, 2/3, 2/3, 2/3], [2/3, 2/3, 0, 8/3, 0, 0]),
        (make_quad4_boundary_mesh, [1, 1, 1, 1], [2, 2, 0, 0]),
        (make_quad8_load_mesh, [-0.5]*4 + [2]*4, [0.5, 0.5, 0, 0, 2, 0, 0, 0]),
    ],
    ids=["tri3", "tri6", "quad4", "quad8"],
)
def test_plane_loads_have_consistent_nodal_weights_without_elastic_properties(
    builder, body_weights, edge_weights,
):
    mesh = builder()
    element = mesh.elements[0]
    element.props = {"thickness": element.props["thickness"]}
    kernel = get_element_kernel(element.type)
    body_vector = np.array((4.0, -5.0))
    edge_vector = np.array((7.0, -11.0))

    body = kernel.body_force(mesh, element, body_vector)
    edge = kernel.edge_traction(mesh, element, 0, edge_vector)

    # Shape-function integrals include the helper geometry's area/edge length and thickness.
    np.testing.assert_allclose(
        body.reshape(-1, 2), np.array(body_weights)[:, None] * body_vector, atol=1e-14,
    )
    np.testing.assert_allclose(
        edge.reshape(-1, 2), np.array(edge_weights)[:, None] * edge_vector, atol=1e-14,
    )


@pytest.mark.parametrize(
    ("builder", "body_weights", "face_weights"),
    [
        (make_tet4_stiffness_mesh, [1/24]*4, [1/6, 1/6, 1/6, 0]),
        (make_tet10_stiffness_mesh, [-1/120]*4 + [1/30]*6,
         [0, 0, 0, 0, 1/6, 1/6, 1/6, 0, 0, 0]),
    ],
    ids=["tet4", "tet10"],
)
def test_tetrahedron_body_and_face_loads_have_consistent_nodal_weights(
    builder, body_weights, face_weights,
):
    mesh = builder()
    element = mesh.elements[0]
    kernel = get_element_kernel(element.type)
    body_vector = np.array((2.0, -3.0, -6.0))
    face_vector = np.array((1.0, 2.0, -2.0))

    body = kernel.body_force(mesh, element, body_vector)
    face = kernel.face_traction(mesh, element, 3, face_vector)

    # Unit tetrahedron: volume 1/6, bottom face area 1/2.
    # Quadratic body weights are -1/120 at corners and 1/30 at midsides.
    np.testing.assert_allclose(
        body.reshape(-1, 3), np.array(body_weights)[:, None] * body_vector, atol=1e-14,
    )
    np.testing.assert_allclose(
        face.reshape(-1, 3), np.array(face_weights)[:, None] * face_vector, atol=1e-14,
    )


def test_hex8_assembled_loads_scale_with_volume_and_face_area():
    mesh = make_hex8_stiffness_mesh()
    boundary = BoundaryCondition()
    boundary.add_body_force_element(1, 0.0, 0.0, -2.0)
    boundary.add_surface_traction(1, 1, 0.0, 0.0, -5.0)

    loads = build_load_vector(mesh, boundary).reshape(-1, 3)

    # Box dimensions 2 by 3 by 4: volume 24 and top area 6.
    expected = np.zeros((8, 3))
    expected[:, 2] = -6.0
    expected[4:, 2] -= 7.5
    np.testing.assert_allclose(loads, expected, atol=1e-13)


def test_hex20_body_force_has_consistent_corner_and_midside_weights():
    mesh = make_hex20_stiffness_mesh()
    vector = np.array((2.0, -3.0, 4.0))

    loads = get_element_kernel("C3D20").body_force(mesh, mesh.elements[0], vector)

    # The unit cube's serendipity shape integrals sum to volume 1.
    weights = np.array([-1/8]*8 + [1/6]*12)
    np.testing.assert_allclose(loads.reshape(-1, 3), weights[:, None] * vector, atol=1e-14)


def test_hex20_quadratic_face_load_scales_with_actual_area():
    mesh = make_hex20_stiffness_mesh()
    for node in mesh.nodes:
        node.x *= 2.0
        node.y *= 3.0

    loads = get_element_kernel("Hex20").face_traction(
        mesh, mesh.elements[0], 1, (0.0, 0.0, -5.0),
    )

    # Area 6: corner weights -area/12, midside weights area/3.
    expected = np.zeros((20, 3))
    expected[4:8, 2] = 2.5
    expected[12:16, 2] = -10.0
    np.testing.assert_allclose(loads.reshape(-1, 3), expected, atol=1e-13)


def test_hex20_body_force_reports_element_context_for_degenerate_volume():
    mesh = make_hex20_stiffness_mesh()
    for node in mesh.nodes:
        node.z = 0.0

    with pytest.raises(ValueError, match="Hex20 element 1.*non-positive Jacobian"):
        get_element_kernel("Hex20").body_force(mesh, mesh.elements[0], (0.0, 0.0, -1.0))
