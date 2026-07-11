import importlib
import sys

import numpy as np
import pytest

from fem import boundary
from fem.core.mesh import Element2D, Node2D, TrussMesh2D
from fem.elements import get_element_kernel
from fem.elements.hexahedron import (
    HEX20_EXTRAPOLATION_MATRIX,
    HEX20_NATURAL_NODE_COORDS,
    Hex8Kernel,
    hex20_gauss_points,
    hex20_shape_funcs_grads,
    hex8_gauss_points,
    hex8_shape_funcs_grads,
)
from fem.elements.line import line2_geometry
from fem.elements.quadrilateral import (
    Quad4PlaneKernel,
    Quad8PlaneKernel,
    quad4_shape_grad_xi_eta,
    quad8_shape_funcs_grads,
)
from fem.elements.registry import register_element_kernel
from fem.elements.tetrahedron import (
    TET10_NATURAL_NODE_COORDS,
    Tet4Kernel,
    Tet10Kernel,
    tet10_gauss_points,
    tet10_shape_funcs_grads,
)
from fem.elements.triangle import (
    Tri3PlaneKernel,
    Tri6PlaneKernel,
    tri6_shape_funcs_grads,
)
from fem.materials import linear_elastic
from tests.helpers.mesh_builders import (
    make_beam_stiffness_mesh,
    make_hex20_stiffness_mesh,
    make_hex8_solid_stress_mesh,
    make_hex8_stiffness_mesh,
    make_quad4_boundary_mesh,
    make_quad4_stiffness_mesh,
    make_quad8_load_mesh,
    make_quad8_stiffness_mesh,
    make_tet4_stiffness_mesh,
    make_tet10_stiffness_mesh,
    make_tri3_load_mesh,
    make_tri3_stiffness_mesh,
    make_tri6_load_mesh,
    make_tri6_stiffness_mesh,
    make_truss_stiffness_mesh,
)


def _node_lookup(mesh):
    return {node.id: node for node in mesh.nodes}


def _assert_kernel_matches_explicit_node_lookup(mesh):
    elem = mesh.elements[0]
    kernel = get_element_kernel(elem.type)

    ke = kernel.stiffness(mesh, elem)
    expected = kernel.stiffness(mesh, elem, _node_lookup(mesh))

    assert np.allclose(ke, expected)


@pytest.mark.parametrize(
    "builder",
    [
        make_truss_stiffness_mesh,
        make_beam_stiffness_mesh,
        make_tri3_stiffness_mesh,
        make_tri6_stiffness_mesh,
        make_quad4_stiffness_mesh,
        make_quad8_stiffness_mesh,
        make_hex8_stiffness_mesh,
        make_tet4_stiffness_mesh,
        make_tet10_stiffness_mesh,
    ],
    ids=[
        "truss2d",
        "beam2d",
        "tri3",
        "tri6",
        "quad4",
        "quad8",
        "hex8",
        "tet4",
        "tet10",
    ],
)
def test_kernels_match_explicit_node_lookup(builder):
    _assert_kernel_matches_explicit_node_lookup(builder())


# Line element kernels


def test_truss_kernel_provides_element_stress():
    mesh = make_truss_stiffness_mesh()
    elem = mesh.elements[0]
    U = np.array([0.0, 0.0, 0.02, 0.0], dtype=float)

    axial_strain, axial_stress, mises = get_element_kernel("Truss2D").element_stress(
        mesh, elem, U
    )

    assert axial_strain == pytest.approx(0.01)
    assert axial_stress == pytest.approx(2.1)
    assert mises == pytest.approx(2.1)


def test_inclined_truss_matches_closed_form_axial_response():
    mesh = TrussMesh2D(
        nodes=[Node2D(1, 0.0, 0.0), Node2D(2, 3.0, 4.0)],
        elements=[
            Element2D(
                1,
                [1, 2],
                "Truss2D",
                {"E": 200.0, "area": 2.0},
            )
        ],
    )
    elem = mesh.elements[0]
    kernel = get_element_kernel(elem.type)
    U = np.array([0.0, 0.0, 0.6, 0.8])

    internal_force = kernel.stiffness(mesh, elem) @ U
    strain, stress, mises = kernel.element_stress(mesh, elem, U)

    assert np.allclose(internal_force, [-48.0, -64.0, 48.0, 64.0])
    assert strain == pytest.approx(0.2)
    assert stress == pytest.approx(40.0)
    assert mises == pytest.approx(40.0)


@pytest.mark.parametrize(
    "mode",
    ["translation_x", "translation_y", "rotation"],
)
def test_inclined_beam_rigid_body_mode_has_zero_internal_force(mode):
    mesh = make_beam_stiffness_mesh()
    elem = mesh.elements[0]
    kernel = get_element_kernel(elem.type)
    U = np.zeros(mesh.num_dofs, dtype=float)

    for node in mesh.nodes:
        if mode == "translation_x":
            U[mesh.global_dof(node.id, 0)] = 1.0
        elif mode == "translation_y":
            U[mesh.global_dof(node.id, 1)] = 1.0
        else:
            U[mesh.global_dof(node.id, 0)] = -node.y
            U[mesh.global_dof(node.id, 1)] = node.x
            U[mesh.global_dof(node.id, 2)] = 1.0

    Ke = kernel.stiffness(mesh, elem)
    Ue = U[mesh.element_dofs(elem)]

    assert np.allclose(Ke @ Ue, np.zeros(6), atol=1e-12)
    assert float(Ue @ Ke @ Ue) == pytest.approx(0.0, abs=1e-12)


def test_inclined_beam_cantilever_matches_euler_bernoulli_tip_response():
    mesh = make_beam_stiffness_mesh()
    elem = mesh.elements[0]
    kernel = get_element_kernel(elem.type)
    Ke = kernel.stiffness(mesh, elem)
    L, c, s = line2_geometry(mesh, elem)
    E = float(elem.props["E"])
    Izz = float(elem.props["Izz"])
    free = mesh.node_dofs(elem.node_ids[1])
    F = np.zeros(mesh.num_dofs, dtype=float)
    F[free[0]] = -s
    F[free[1]] = c

    U_tip = np.linalg.solve(Ke[np.ix_(free, free)], F[free])
    v_local = L**3 / (3.0 * E * Izz)
    expected = np.array([
        -s * v_local,
        c * v_local,
        L**2 / (2.0 * E * Izz),
    ])

    assert np.allclose(U_tip, expected)


@pytest.mark.parametrize(
    "builder",
    [make_truss_stiffness_mesh, make_beam_stiffness_mesh],
    ids=["truss2d", "beam2d"],
)
def test_line_element_body_force_preserves_global_resultant(builder):
    mesh = builder()
    elem = mesh.elements[0]
    kernel = get_element_kernel(elem.type)
    vector = np.array([4.0, -5.0])
    length, _, _ = line2_geometry(mesh, elem)
    area = float(elem.props["area"])

    fe = kernel.body_force(mesh, elem, tuple(vector))
    stride = mesh.dofs_per_node

    assert float(fe[0::stride].sum()) == pytest.approx(area * length * vector[0])
    assert float(fe[1::stride].sum()) == pytest.approx(area * length * vector[1])


def test_beam_body_force_preserves_global_moment():
    mesh = make_beam_stiffness_mesh()
    elem = mesh.elements[0]
    vector = np.array([4.0, -5.0])
    fe = get_element_kernel(elem.type).body_force(mesh, elem, tuple(vector))
    length, _, _ = line2_geometry(mesh, elem)
    area = float(elem.props["area"])
    node_lookup = _node_lookup(mesh)
    ni = node_lookup[elem.node_ids[0]]
    nj = node_lookup[elem.node_ids[1]]
    total_force = area * length * vector
    midpoint = np.array([(ni.x + nj.x) / 2.0, (ni.y + nj.y) / 2.0])

    assembled_moment = (
        ni.x * fe[1] - ni.y * fe[0] + fe[2]
        + nj.x * fe[4] - nj.y * fe[3] + fe[5]
    )
    expected_moment = midpoint[0] * total_force[1] - midpoint[1] * total_force[0]

    assert assembled_moment == pytest.approx(expected_moment)


def test_beam_stiffness_is_invariant_to_element_node_order():
    mesh = make_beam_stiffness_mesh()
    elem = mesh.elements[0]
    kernel = get_element_kernel(elem.type)
    K_forward = kernel.stiffness(mesh, elem)

    elem.node_ids = list(reversed(elem.node_ids))
    K_reverse = kernel.stiffness(mesh, elem)
    swap = np.array([3, 4, 5, 0, 1, 2])

    assert np.allclose(K_forward, K_reverse[np.ix_(swap, swap)])


@pytest.mark.parametrize(
    "builder",
    [make_truss_stiffness_mesh, make_beam_stiffness_mesh],
    ids=["truss2d", "beam2d"],
)
def test_line_element_rejects_zero_length(builder):
    mesh = builder()
    elem = mesh.elements[0]
    node_lookup = _node_lookup(mesh)
    ni = node_lookup[elem.node_ids[0]]
    nj = node_lookup[elem.node_ids[1]]
    nj.x = ni.x
    nj.y = ni.y

    with pytest.raises(ValueError, match="zero length"):
        get_element_kernel(elem.type).stiffness(mesh, elem)


@pytest.mark.parametrize(
    "builder",
    [make_truss_stiffness_mesh, make_beam_stiffness_mesh],
    ids=["truss2d", "beam2d"],
)
def test_line_element_reports_missing_node(builder):
    mesh = builder()
    elem = mesh.elements[0]
    elem.node_ids[1] = 999

    with pytest.raises(KeyError, match=r"Element 1 references missing node 999"):
        get_element_kernel(elem.type).stiffness(mesh, elem)


@pytest.mark.parametrize(
    ("builder", "missing_property"),
    [
        (make_truss_stiffness_mesh, "area"),
        (make_beam_stiffness_mesh, "Izz"),
    ],
    ids=["truss2d", "beam2d"],
)
def test_line_element_reports_missing_required_property(builder, missing_property):
    mesh = builder()
    elem = mesh.elements[0]
    elem.props.pop(missing_property)

    with pytest.raises(
        KeyError,
        match=rf"Element {elem.id} missing property {missing_property}",
    ):
        get_element_kernel(elem.type).stiffness(mesh, elem)


@pytest.mark.parametrize(
    ("builder", "expected_count"),
    (
        (make_hex8_stiffness_mesh, 8),
        (make_hex20_stiffness_mesh, 20),
        (make_tet4_stiffness_mesh, 4),
        (make_tet10_stiffness_mesh, 10),
    ),
    ids=["hex8", "hex20", "tet4", "tet10"],
)
def test_solid_stiffness_reports_invalid_node_count_with_context(
    builder,
    expected_count,
):
    mesh = builder()
    elem = mesh.elements[0]
    elem.node_ids = elem.node_ids[:-1]

    with pytest.raises(ValueError) as exc_info:
        get_element_kernel(elem.type).stiffness(mesh, elem)

    message = str(exc_info.value)
    assert f"{elem.type} element {elem.id}" in message
    assert f"requires {expected_count} nodes" in message
    assert f"got {expected_count - 1}" in message
    assert f"node_ids={elem.node_ids}" in message


# Plane element kernels


def test_tri6_shape_functions_interpolate_nodes():
    node_coords = [
        (0.0, 0.0),
        (1.0, 0.0),
        (0.0, 1.0),
        (0.5, 0.0),
        (0.5, 0.5),
        (0.0, 0.5),
    ]

    for i, (xi, eta) in enumerate(node_coords):
        N, dN_dxi, dN_deta = tri6_shape_funcs_grads(xi, eta)

        expected = np.zeros(6, dtype=float)
        expected[i] = 1.0
        assert np.allclose(N, expected)
        assert np.isclose(float(np.sum(N)), 1.0)
        assert np.isclose(float(np.sum(dN_dxi)), 0.0)
        assert np.isclose(float(np.sum(dN_deta)), 0.0)


def test_quad4_stiffness_builds_node_lookup_from_mesh_when_omitted():
    mesh = make_quad4_stiffness_mesh()

    ke = get_element_kernel("Quad4Plane").stiffness(mesh, mesh.elements[0])

    assert ke.shape == (8, 8)
    assert np.allclose(ke, ke.T)


@pytest.mark.parametrize(
    ("builder", "edge", "body_measure", "edge_measure"),
    [
        (make_tri3_load_mesh, 0, 2.0, 4.0),
        (make_tri6_load_mesh, 0, 2.0, 4.0),
        (make_quad4_boundary_mesh, 0, 4.0, 4.0),
        (make_quad8_load_mesh, 0, 6.0, 3.0),
    ],
    ids=["tri3", "tri6", "quad4", "quad8"],
)
def test_plane_kernels_provide_body_force_and_edge_traction(
    builder, edge, body_measure, edge_measure
):
    mesh = builder()
    elem = mesh.elements[0]
    kernel = get_element_kernel(elem.type)

    body = kernel.body_force(mesh, elem, (4.0, -5.0))
    edge_load = kernel.edge_traction(mesh, elem, edge, (7.0, -11.0))

    assert body.shape == (len(elem.node_ids) * 2,)
    assert edge_load.shape == (len(elem.node_ids) * 2,)
    assert float(body[0::2].sum()) == pytest.approx(4.0 * body_measure)
    assert float(body[1::2].sum()) == pytest.approx(-5.0 * body_measure)
    assert float(edge_load[0::2].sum()) == pytest.approx(7.0 * edge_measure)
    assert float(edge_load[1::2].sum()) == pytest.approx(-11.0 * edge_measure)


@pytest.mark.parametrize(
    "builder",
    [make_tet4_stiffness_mesh, make_tet10_stiffness_mesh],
    ids=["tet4", "tet10"],
)
def test_tet_kernels_provide_body_force_and_face_traction(builder):
    mesh = builder()
    elem = mesh.elements[0]
    kernel = get_element_kernel(elem.type)

    body = kernel.body_force(mesh, elem, (0.0, 0.0, -6.0))
    face = kernel.face_traction(mesh, elem, 3, (0.0, 0.0, -2.0))

    assert body.shape == (len(elem.node_ids) * 3,)
    assert face.shape == (len(elem.node_ids) * 3,)
    assert float(body[2::3].sum()) == pytest.approx(-1.0)
    assert float(face[2::3].sum()) == pytest.approx(-1.0)


@pytest.mark.parametrize(
    "builder",
    [make_quad4_boundary_mesh, make_quad8_load_mesh],
    ids=["quad4", "quad8"],
)
def test_plane_load_kernels_do_not_require_elastic_props(builder):
    mesh = builder()
    elem = mesh.elements[0]
    elem.props = {"thickness": elem.props["thickness"]}
    kernel = get_element_kernel(elem.type)

    body = kernel.body_force(mesh, elem, (4.0, -5.0))
    edge_load = kernel.edge_traction(mesh, elem, 0, (7.0, -11.0))

    assert body.shape == (len(elem.node_ids) * 2,)
    assert edge_load.shape == (len(elem.node_ids) * 2,)


@pytest.mark.parametrize(
    ("builder", "gauss_order"),
    [
        (make_tri3_stiffness_mesh, None),
        (make_tri6_stiffness_mesh, None),
        (make_quad4_stiffness_mesh, 2),
        (make_quad8_stiffness_mesh, 3),
    ],
    ids=["tri3", "tri6", "quad4", "quad8"],
)
def test_plane_kernels_provide_stress_interfaces_without_post_helpers(
    builder, gauss_order
):
    mesh = builder()
    elem = mesh.elements[0]
    U = np.linspace(0.01, 0.01 * mesh.num_dofs, mesh.num_dofs)
    kernel = get_element_kernel(elem.type)

    if "tri" in elem.type.lower():
        node_vals, plane_type, nu = kernel.nodal_stress(
            mesh, elem, U, _node_lookup(mesh)
        )
    else:
        node_vals, plane_type, nu = kernel.nodal_stress(
            mesh, elem, U, _node_lookup(mesh), gauss_order
        )

    assert node_vals.shape == (len(elem.node_ids), 3)
    assert plane_type == "stress"
    assert nu == elem.props["nu"]


# Solid element kernels


def test_hex20_kernel_stiffness_and_body_force_contract():
    mesh = make_hex20_stiffness_mesh()
    elem = mesh.elements[0]
    kernel = get_element_kernel("C3D20")

    Ke = kernel.stiffness(mesh, elem)
    fe = kernel.body_force(mesh, elem, (2.0, -3.0, 4.0))

    assert Ke.shape == (60, 60)
    assert fe.shape == (60,)
    assert np.all(np.isfinite(Ke))
    assert np.allclose(Ke, Ke.T)
    assert fe[0::3].sum() == pytest.approx(2.0)
    assert fe[1::3].sum() == pytest.approx(-3.0)
    assert fe[2::3].sum() == pytest.approx(4.0)


def test_hex20_body_force_reports_element_type_for_nonpositive_jacobian():
    mesh = make_hex20_stiffness_mesh()
    elem = mesh.elements[0]
    for node in mesh.nodes:
        node.z = 0.0
    kernel = get_element_kernel(elem.type)

    with pytest.raises(
        ValueError,
        match=r"Hex20 element 1 has non-positive Jacobian determinant.*expected > 0",
    ):
        kernel.body_force(mesh, elem, (0.0, 0.0, -1.0))


def test_hex20_affine_displacement_produces_constant_exact_stress():
    mesh = make_hex20_stiffness_mesh()
    elem = mesh.elements[0]
    kernel = get_element_kernel(elem.type)
    U = np.zeros(mesh.num_dofs)
    for node in mesh.nodes:
        U[mesh.global_dof(node.id, 0)] = 0.01 * node.x + 0.02 * node.y + 0.03 * node.z
        U[mesh.global_dof(node.id, 1)] = -0.02 * node.x + 0.04 * node.y + 0.01 * node.z
        U[mesh.global_dof(node.id, 2)] = 0.03 * node.x - 0.01 * node.y + 0.05 * node.z
    strain = np.array([0.01, 0.04, 0.05, 0.0, 0.0, 0.06])
    expected = linear_elastic.solid_3d_matrix(210.0, 0.3) @ strain

    for xi, eta, zeta, _ in hex20_gauss_points():
        assert np.allclose(kernel.stress_at(mesh, elem, U, xi, eta, zeta), expected)


def test_hex20_nodal_stress_recovers_gauss_point_values():
    mesh = make_hex20_stiffness_mesh(curved=True)
    elem = mesh.elements[0]
    kernel = get_element_kernel(elem.type)
    U = np.linspace(0.01, 0.01 * mesh.num_dofs, mesh.num_dofs)
    gp_vals = np.array([
        kernel.stress_at(mesh, elem, U, xi, eta, zeta)
        for xi, eta, zeta, _ in hex20_gauss_points()
    ])
    expected = HEX20_EXTRAPOLATION_MATRIX @ gp_vals

    assert np.allclose(kernel.nodal_stress(mesh, elem, U), expected)


def test_hex20_shape_functions_interpolate_all_nodes():
    for local_index, coords in enumerate(HEX20_NATURAL_NODE_COORDS):
        N, dN_dxi, dN_deta, dN_dzeta = hex20_shape_funcs_grads(*coords)
        expected = np.zeros(20)
        expected[local_index] = 1.0
        assert np.allclose(N, expected)
        assert np.isclose(dN_dxi.sum(), 0.0)
        assert np.isclose(dN_deta.sum(), 0.0)
        assert np.isclose(dN_dzeta.sum(), 0.0)


def test_hex20_partition_of_unity_and_full_integration_weights():
    for xi, eta, zeta in [(-0.3, 0.2, 0.4), (0.0, 0.0, 0.0), (0.7, -0.5, 0.1)]:
        N, dN_dxi, dN_deta, dN_dzeta = hex20_shape_funcs_grads(xi, eta, zeta)
        assert N.sum() == pytest.approx(1.0)
        assert dN_dxi.sum() == pytest.approx(0.0)
        assert dN_deta.sum() == pytest.approx(0.0)
        assert dN_dzeta.sum() == pytest.approx(0.0)
    assert sum(w for *_, w in hex20_gauss_points()) == pytest.approx(8.0)


def test_hex20_recovery_matrix_is_left_inverse_of_gauss_shape_matrix():
    n_gp = np.array([
        hex20_shape_funcs_grads(xi, eta, zeta)[0]
        for xi, eta, zeta, _ in hex20_gauss_points()
    ])
    assert n_gp.shape == (27, 20)
    assert np.allclose(HEX20_EXTRAPOLATION_MATRIX @ n_gp, np.eye(20))


def test_hex8_body_force_uses_actual_element_volume():
    mesh = make_hex8_stiffness_mesh()
    bc = boundary.condition.BoundaryCondition()
    bc.add_body_force_element(1, 0.0, 0.0, -2.0)

    loads = boundary.loads.build_load_vector(mesh, bc)

    assert float(loads[2::3].sum()) == pytest.approx(-48.0)
    assert np.allclose(loads[2::3], np.full(8, -6.0))


def test_hex8_face_traction_uses_actual_face_area():
    mesh = make_hex8_stiffness_mesh()
    bc = boundary.condition.BoundaryCondition()
    bc.add_surface_traction(1, 1, 0.0, 0.0, -5.0)

    loads = boundary.loads.build_load_vector(mesh, bc)

    assert float(loads[2::3].sum()) == pytest.approx(-30.0)
    assert np.allclose(loads[2::3][:4], np.zeros(4))
    assert np.allclose(loads[2::3][4:], np.full(4, -7.5))


def test_hex20_face_traction_uses_quadratic_face_and_actual_area():
    mesh = make_hex20_stiffness_mesh()
    elem = mesh.elements[0]
    for node in mesh.nodes:
        node.x *= 2.0
        node.y *= 3.0

    fe = get_element_kernel(elem.type).face_traction(
        mesh,
        elem,
        1,
        (0.0, 0.0, -5.0),
    )

    assert fe.shape == (60,)
    assert fe[0::3].sum() == pytest.approx(0.0)
    assert fe[1::3].sum() == pytest.approx(0.0)
    assert fe[2::3].sum() == pytest.approx(-30.0)

    expected_z = np.zeros(20)
    expected_z[[4, 5, 6, 7]] = 2.5
    expected_z[[12, 13, 14, 15]] = -10.0
    assert np.allclose(fe[0::3], np.zeros(20))
    assert np.allclose(fe[1::3], np.zeros(20))
    assert np.allclose(fe[2::3], expected_z)


@pytest.mark.parametrize(
    "builder",
    [
        make_hex8_solid_stress_mesh,
        make_tet4_stiffness_mesh,
        make_tet10_stiffness_mesh,
    ],
    ids=["hex8", "tet4", "tet10"],
)
def test_solid_kernels_provide_nodal_stress_matching_post_helpers(builder):
    mesh = builder()
    elem = mesh.elements[0]
    if elem.type.lower() == "tet10":
        # Keep the Tet10 expectation sensitive to the recovery rule, not only
        # to straight-sided geometry where direct node evaluation is equivalent.
        next(node for node in mesh.nodes if node.id == 5).z = 0.08
    U = np.linspace(0.01, 0.01 * mesh.num_dofs, mesh.num_dofs)
    node_lookup = _node_lookup(mesh)
    kernel = get_element_kernel(elem.type)

    if elem.type.lower() == "hex8":
        gauss_points = [(xi, eta, zeta) for xi, eta, zeta, _ in hex8_gauss_points()]
        gp_stresses = np.array(
            [
                kernel.stress_at(mesh, elem, U, xi, eta, zeta, node_lookup)
                for xi, eta, zeta in gauss_points
            ],
            dtype=float,
        )
        n_gp = np.array(
            [
                hex8_shape_funcs_grads(xi, eta, zeta)[0]
                for xi, eta, zeta in gauss_points
            ],
            dtype=float,
        )
        expected = np.linalg.solve(n_gp, gp_stresses)
        node_vals = kernel.nodal_stress(mesh, elem, U, node_lookup)
    elif elem.type.lower() == "tet4":
        stress = kernel.stress_at(mesh, elem, U, 0.25, 0.25, 0.25, node_lookup)
        expected = np.tile(stress, (4, 1))
        node_vals = kernel.nodal_stress(mesh, elem, U, node_lookup)
    else:
        gauss_coords = [(xi, eta, zeta) for xi, eta, zeta, _ in tet10_gauss_points()]
        gp_stresses = np.array(
            [
                kernel.stress_at(mesh, elem, U, xi, eta, zeta, node_lookup)
                for xi, eta, zeta in gauss_coords
            ],
            dtype=float,
        )
        a_gp = np.array([[1.0, xi, eta, zeta] for xi, eta, zeta in gauss_coords], dtype=float)
        coeffs = np.linalg.solve(a_gp, gp_stresses)
        a_nodes = np.array(
            [[1.0, xi, eta, zeta] for xi, eta, zeta in TET10_NATURAL_NODE_COORDS],
            dtype=float,
        )
        expected = a_nodes @ coeffs
        node_vals = kernel.nodal_stress(mesh, elem, U, node_lookup)

    assert np.allclose(node_vals, expected)


# Elements package


def test_elements_use_family_modules_and_registry_module():
    assert type(get_element_kernel("Quad4Plane")) is Quad4PlaneKernel
    assert type(get_element_kernel("Quad8Plane")) is Quad8PlaneKernel
    assert type(get_element_kernel("Tri3Plane")) is Tri3PlaneKernel
    assert type(get_element_kernel("Tri6Plane")) is Tri6PlaneKernel
    assert type(get_element_kernel("Hex8")) is Hex8Kernel
    assert type(get_element_kernel("Tet4")) is Tet4Kernel
    assert type(get_element_kernel("Tet10")) is Tet10Kernel
    assert callable(quad4_shape_grad_xi_eta)
    assert callable(quad8_shape_funcs_grads)
    assert callable(tri6_shape_funcs_grads)
    assert callable(hex8_shape_funcs_grads)
    assert callable(tet10_shape_funcs_grads)
    assert callable(register_element_kernel)

    for old_module in (
        "fem.elements.quad4",
        "fem.elements.quad8",
        "fem.elements.tri3",
        "fem.elements.tet",
        "fem.elements.hex8",
    ):
        sys.modules.pop(old_module, None)
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(old_module)


@pytest.mark.parametrize("element_type", ["C3D20R", "c3D20r"])
def test_registry_rejects_reduced_integration_hex20_alias(element_type):
    with pytest.raises(
        NotImplementedError,
        match=rf"Unsupported element type: {element_type}",
    ):
        get_element_kernel(element_type)


def test_registry_preserves_preexisting_variant_fallback():
    assert type(get_element_kernel("C3D8R")) is Hex8Kernel
