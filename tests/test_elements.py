import importlib
import sys

import numpy as np
import pytest

from fem import boundary
from fem.elements import get_element_kernel
from fem.elements.hexahedron import (
    Hex8Kernel,
    hex8_gauss_points,
    hex8_shape_funcs_grads,
)
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
from tests.helpers.mesh_builders import (
    make_beam_stiffness_mesh,
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
