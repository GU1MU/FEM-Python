from itertools import product

import numpy as np
import pytest

from fem.elements import get_element_kernel
from tests.helpers.mesh_builders import (
    make_tri3_stiffness_mesh, make_tri6_stiffness_mesh,
    make_quad4_stiffness_mesh, make_quad8_stiffness_mesh,
    make_hex8_stiffness_mesh, make_hex8_solid_stress_mesh, make_hex20_stiffness_mesh,
    make_tet4_stiffness_mesh, make_tet10_stiffness_mesh,
)


def test_hex8_bbar_stress_matches_abaqus_c3d8_reference():
    """固定 Abaqus C3D8 基准，防止退回未修正的常规 B 矩阵。"""
    mesh = make_hex8_solid_stress_mesh()
    coordinates = (
        (1.25, 10.0, 200.0), (1.25, 0.0, 200.0),
        (1.25, 0.0, 192.0), (1.25, 10.0, 192.0),
        (11.25, 10.0, 200.0), (11.25, 0.0, 200.0),
        (11.25, 0.0, 192.0), (11.25, 10.0, 192.0),
    )
    for node, (x, y, z) in zip(mesh.nodes, coordinates):
        node.x, node.y, node.z = x, y, z
    mesh.elements[0].props = {"E": 220000.0, "nu": 0.3}
    nodal_displacements = (
        (0.0, 0.0, 0.0), (0.0, 0.0, 0.0),
        (0.0029052102537830754, -0.011516803538312588, 0.014116381956906784),
        (-0.0029052102536912526, -0.011516803538289972, -0.014116381957019121),
        (0.0, 0.0, 0.0), (0.0, 0.0, 0.0),
        (-0.0029052102536925393, -0.011516803538290809, 0.014116381957018316),
        (0.002905210253783908, -0.011516803538313872, -0.014116381956907585),
    )
    displacement = np.asarray(nodal_displacements, dtype=float).ravel()
    kernel = get_element_kernel("Hex8")
    a = 1.0 / np.sqrt(3.0)

    integration_stress = kernel.stress_at(
        mesh, mesh.elements[0], displacement, -a, -a, -a
    )
    nodal_stress = kernel.nodal_stress(mesh, mesh.elements[0], displacement)

    assert integration_stress == pytest.approx(
        [-49.4706, -61.4677, 110.938, -5.99856, 71.3284, 10.2427],
        rel=2.0e-5,
    )
    assert nodal_stress[0] == pytest.approx(
        [-99.5386, -99.5386, 199.077, 0.0, 121.812, 30.7282],
        rel=2.0e-5,
        abs=1.0e-8,
    )


def test_tet10_nodal_stress_recovers_quadratic_displacement_analytically():
    mesh = make_tet10_stiffness_mesh()
    element = mesh.elements[0]
    element.props.update(E=120.0, nu=0.25)
    displacement = np.zeros(mesh.num_dofs)
    coordinates = {node.id: node.x for node in mesh.nodes}
    for node in mesh.nodes:
        displacement[mesh.global_dof(node.id, 0)] = node.x ** 2

    stress = get_element_kernel("Tet10").nodal_stress(mesh, element, displacement)

    # eps_xx=2x; E=120, nu=.25 give lambda=mu=48.
    expected = np.array([
        (288.0*x, 96.0*x, 96.0*x, 0.0, 0.0, 0.0)
        for x in (coordinates[node_id] for node_id in element.node_ids)
    ])
    np.testing.assert_allclose(stress, expected, atol=1e-11)


@pytest.mark.parametrize(
    ("builder", "alias", "formulation", "expected"),
    [
        (make_tri3_stiffness_mesh, "CPS3", "stress", [2.56, 5.44, -0.48]),
        (make_tri3_stiffness_mesh, "CPE3", "strain", [3.36, 6.24, -0.48]),
        (make_tri6_stiffness_mesh, "CPS6", "stress", [2.56, 5.44, -0.48]),
        (make_tri6_stiffness_mesh, "CPE6", "strain", [3.36, 6.24, -0.48]),
        (make_quad4_stiffness_mesh, "CPS4", "stress", [2.56, 5.44, -0.48]),
        (make_quad4_stiffness_mesh, "CPE4", "strain", [3.36, 6.24, -0.48]),
        (make_quad8_stiffness_mesh, "CPS8", "stress", [2.56, 5.44, -0.48]),
        (make_quad8_stiffness_mesh, "CPE8", "strain", [3.36, 6.24, -0.48]),
    ],
    ids=["tri3-stress", "tri3-strain", "tri6-stress", "tri6-strain",
         "quad4-stress", "quad4-strain", "quad8-stress", "quad8-strain"],
)
def test_plane_aliases_recover_analytic_affine_stress(builder, alias, formulation, expected):
    mesh = builder()
    element = mesh.elements[0]
    element.type = alias
    element.props.update(E=120.0, nu=0.25)
    element.props.pop("plane_type", None)
    displacement = np.zeros(mesh.num_dofs)
    for node in mesh.nodes:
        displacement[mesh.global_dof(node.id, 0)] = 0.3 + 0.01*node.x + 0.02*node.y
        displacement[mesh.global_dof(node.id, 1)] = -0.2 - 0.03*node.x + 0.04*node.y
    # Engineering strains are (.01,.04,-.01); the numeric stresses use E=120, nu=.25.
    kernel = get_element_kernel(alias)
    for lookup in (None, {node.id: node for node in mesh.nodes}):
        if lookup is not None and kernel.canonical_type in ("Quad4", "Quad8"):
            gauss_order = 2 if kernel.canonical_type == "Quad4" else 3
            stress, plane_type, nu = kernel.nodal_stress(
                mesh, element, displacement, lookup, gauss_order,
            )
        else:
            stress, plane_type, nu = kernel.nodal_stress(mesh, element, displacement, lookup)
        assert plane_type == formulation
        assert nu == pytest.approx(0.25)
        np.testing.assert_allclose(stress, np.tile(expected, (len(element.node_ids), 1)), atol=1e-11)


@pytest.mark.parametrize(
    "builder", [make_hex8_stiffness_mesh, make_hex20_stiffness_mesh,
                make_tet4_stiffness_mesh, make_tet10_stiffness_mesh],
    ids=["hex8", "hex20", "tet4", "tet10"],
)
def test_solid_affine_stress_is_exact_at_nodes_and_integration_points(builder):
    mesh = builder()
    element = mesh.elements[0]
    element.props.update(E=120.0, nu=0.25)
    displacement = np.zeros(mesh.num_dofs)
    for node in mesh.nodes:
        displacement[mesh.global_dof(node.id, 0)] = 0.3 + 0.01*node.x + 0.02*node.y + 0.03*node.z
        displacement[mesh.global_dof(node.id, 1)] = -0.2 - 0.04*node.x + 0.05*node.y + 0.06*node.z
        displacement[mesh.global_dof(node.id, 2)] = 0.1 + 0.07*node.x - 0.08*node.y + 0.09*node.z
    # Normal strains (.01,.05,.09), engineering shear strains (-.02,-.02,.10).
    expected = [8.16, 12.0, 15.84, -0.96, -0.96, 4.8]
    kernel = get_element_kernel(element.type)
    for lookup in (None, {node.id: node for node in mesh.nodes}):
        stress = kernel.nodal_stress(mesh, element, displacement, lookup)
        np.testing.assert_allclose(stress, np.tile(expected, (len(element.node_ids), 1)), atol=1e-10)
        natural, values = kernel.integration_point_stress(mesh, element, displacement, lookup)
        np.testing.assert_allclose(values, np.tile(expected, (len(natural), 1)), atol=1e-10)
        for point in natural:
            np.testing.assert_allclose(kernel.stress_at(mesh, element, displacement, *point, lookup), expected, atol=1e-10)
        if element.type.startswith("Hex"):
            a = 1.0 / np.sqrt(3.0) if element.type == "Hex8" else np.sqrt(3.0 / 5.0)
            axis = [-a, a] if element.type == "Hex8" else [-a, 0.0, a]
            np.testing.assert_allclose(sorted(map(tuple, natural)), sorted(product(axis, repeat=3)), atol=1e-14)


def _polynomials(points, powers):
    return np.column_stack([np.prod(points ** power, axis=1) for power in powers])


@pytest.mark.parametrize(
    "builder", [make_hex8_stiffness_mesh, make_hex20_stiffness_mesh, make_tet10_stiffness_mesh],
    ids=["trilinear-hex8", "serendipity-curved-hex20", "linear-curved-tet10"],
)
def test_nodal_recovery_projects_integration_stress_onto_element_polynomial_space(builder):
    mesh = builder()
    element = mesh.elements[0]
    coordinates = np.array([[node.x, node.y, node.z] for node in mesh.nodes])
    if element.type == "Tet10":
        natural_nodes = coordinates.copy()
        mesh.nodes[4].z += 0.08
        powers = [(0,0,0), (1,0,0), (0,1,0), (0,0,1)]
    else:
        natural_nodes = 2.0 * coordinates / coordinates.max(axis=0) - 1.0
        if element.type == "Hex20":
            mesh.nodes[8].z += 0.05
            # The 20-node serendipity space permits at most one squared coordinate.
            powers = [p for p in product(range(3), repeat=3) if sum(n for n in p if n > 1) <= 2]
        else:
            powers = list(product(range(2), repeat=3))
    displacement = np.linspace(0.01, 0.01*mesh.num_dofs, mesh.num_dofs)
    kernel = get_element_kernel(element.type)
    lookup = {node.id: node for node in mesh.nodes}
    natural, integration_stress = kernel.integration_point_stress(mesh, element, displacement, lookup)
    coefficients = np.linalg.lstsq(_polynomials(natural, powers), integration_stress, rcond=None)[0]
    expected = _polynomials(natural_nodes, powers) @ coefficients

    np.testing.assert_allclose(kernel.nodal_stress(mesh, element, displacement, lookup), expected, atol=1e-10)
