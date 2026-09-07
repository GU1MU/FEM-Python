import numpy as np
import pytest

from fem.elements import get_element_kernel
from tests.helpers.mesh_builders import (
    make_tri3_stiffness_mesh, make_tri6_stiffness_mesh,
    make_quad4_stiffness_mesh, make_quad8_stiffness_mesh,
    make_hex8_stiffness_mesh, make_hex20_stiffness_mesh,
    make_tet4_stiffness_mesh, make_tet10_stiffness_mesh,
)


@pytest.mark.parametrize(
    ("builder", "measure"),
    [
        (make_tri3_stiffness_mesh, 1.5), (make_tri6_stiffness_mesh, 1.5),
        (make_quad4_stiffness_mesh, 3.0), (make_quad8_stiffness_mesh, 6.0),
        (make_hex8_stiffness_mesh, 24.0), (make_hex20_stiffness_mesh, 1.0),
        (make_tet4_stiffness_mesh, 1.0 / 6.0), (make_tet10_stiffness_mesh, 1.0 / 6.0),
    ],
    ids=["tri3", "tri6", "quad4", "quad8", "hex8", "hex20", "tet4", "tet10"],
)
def test_continuum_stiffness_preserves_rigid_motion_and_affine_strain_energy(builder, measure):
    mesh = builder()
    element = mesh.elements[0]
    element.props.update(E=120.0, nu=0.25, thickness=1.5, plane_type="stress")
    nodes = {node.id: node for node in mesh.nodes}
    coordinates = np.array([
        [nodes[node_id].x, nodes[node_id].y]
        if mesh.dofs_per_node == 2 else
        [nodes[node_id].x, nodes[node_id].y, nodes[node_id].z]
        for node_id in element.node_ids
    ])
    kernel = get_element_kernel(element.type)
    default_stiffness = kernel.stiffness(mesh, element)
    explicit_stiffness = kernel.stiffness(mesh, element, nodes)
    # Whole-matrix equality verifies the two lookup entry points, not the physics oracle.
    np.testing.assert_allclose(default_stiffness, explicit_stiffness)
    for stiffness in (default_stiffness, explicit_stiffness):
        np.testing.assert_allclose(stiffness, stiffness.T, atol=1e-12)
        dimension = mesh.dofs_per_node
        for axis in np.eye(dimension):
            translation = np.tile(axis, len(coordinates))
            np.testing.assert_allclose(stiffness @ translation, 0.0, atol=1e-11)
        if dimension == 2:
            rotations = [np.column_stack((-coordinates[:, 1], coordinates[:, 0]))]
            displacement = np.column_stack((
                0.01 * coordinates[:, 0] + 0.04 * coordinates[:, 1],
                0.02 * coordinates[:, 1],
            ))
            # Plane stress: strains (0.01, 0.02, 0.04), engineering shear modulus 48.
            energy_density = 0.0768
        else:
            rotations = [np.cross(axis, coordinates) for axis in np.eye(3)]
            x, y, z = coordinates.T
            displacement = np.column_stack((0.01*x + 0.04*y + 0.06*z, 0.02*y + 0.05*z, 0.03*z))
            # 3D: normal strains (0.01, 0.02, 0.03), shears (0.04, 0.05, 0.06).
            # E=120, nu=.25 give both Lame's first parameter and shear modulus 48.
            energy_density = 0.3384
        for rotation in rotations:
            np.testing.assert_allclose(stiffness @ rotation.ravel(), 0.0, atol=1e-11)
        u = displacement.ravel()
        assert 0.5 * u @ stiffness @ u == pytest.approx(measure * energy_density)


def test_tri3_stiffness_matches_complete_independent_numeric_matrix():
    mesh = make_tri3_stiffness_mesh()
    element = mesh.elements[0]
    element.props.update(E=120.0, nu=0.25, thickness=1.0)

    stiffness = get_element_kernel("Tri3").stiffness(mesh, element)

    # Triangle (0,0), (2,0), (0,1), plane stress with unit thickness.
    np.testing.assert_allclose(stiffness, [
        [80, 40, -32, -24, -48, -16],
        [40, 140, -16, -12, -24, -128],
        [-32, -16, 32, 0, 0, 16],
        [-24, -12, 0, 12, 24, 0],
        [-48, -24, 0, 24, 48, 0],
        [-16, -128, 16, 0, 0, 128],
    ], atol=1e-12)


@pytest.mark.parametrize(
    ("builder", "expected_count"),
    [(make_hex8_stiffness_mesh, 8), (make_hex20_stiffness_mesh, 20),
     (make_tet4_stiffness_mesh, 4), (make_tet10_stiffness_mesh, 10)],
    ids=["hex8", "hex20", "tet4", "tet10"],
)
def test_solid_stiffness_reports_required_node_count_and_element_context(builder, expected_count):
    mesh = builder()
    element = mesh.elements[0]
    element.node_ids.pop()

    with pytest.raises(ValueError) as error:
        get_element_kernel(element.type).stiffness(mesh, element)

    assert f"{element.type} element {element.id}" in str(error.value)
    assert f"requires {expected_count} nodes" in str(error.value)
