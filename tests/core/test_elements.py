import numpy as np
import pytest

from fem.core.mesh import Element3D, Mesh3D, Node3D
from fem.elements import (
    get_element_kernel,
    resolve_beam_frame,
)
from fem.elements.beam_section import parse_beam2_section
from fem.elements.line import line3d_geometry
from tests.helpers.mesh_builders import (
    make_beam_stiffness_mesh,
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
    [make_truss_stiffness_mesh, make_beam_stiffness_mesh],
    ids=["truss2", "beam2"],
)
def test_line_kernels_support_default_and_explicit_node_lookup(builder):
    _assert_kernel_matches_explicit_node_lookup(builder())


# Line element kernels


def test_truss_kernel_provides_element_stress():
    mesh = make_truss_stiffness_mesh()
    elem = mesh.elements[0]
    U = np.array([0.0, 0.0, 0.0, 0.02, 0.0, 0.0], dtype=float)

    axial_strain, axial_stress, mises = get_element_kernel("Truss2").element_stress(
        mesh, elem, U
    )

    assert axial_strain == pytest.approx(0.01)
    assert axial_stress == pytest.approx(2.1)
    assert mises == pytest.approx(2.1)


def test_inclined_truss_matches_closed_form_axial_response():
    mesh = Mesh3D(
        nodes=[Node3D(1, 0.0, 0.0, 0.0), Node3D(2, 3.0, 4.0, 0.0)],
        elements=[
            Element3D(
                1,
                [1, 2],
                "Truss2",
                {"E": 200.0, "area": 2.0},
            )
        ],
    )
    elem = mesh.elements[0]
    kernel = get_element_kernel(elem.type)
    U = np.array([0.0, 0.0, 0.0, 0.6, 0.8, 0.0])

    internal_force = kernel.stiffness(mesh, elem) @ U
    strain, stress, mises = kernel.element_stress(mesh, elem, U)

    assert np.allclose(internal_force, [-48.0, -64.0, 0.0, 48.0, 64.0, 0.0])
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
            U[mesh.global_dof(node.id, 5)] = 1.0

    Ke = kernel.stiffness(mesh, elem)
    Ue = U[mesh.element_dofs(elem)]

    assert np.allclose(Ke @ Ue, np.zeros(12), atol=1e-12)
    assert float(Ue @ Ke @ Ue) == pytest.approx(0.0, abs=1e-12)


def test_inclined_beam_cantilever_matches_b31_one_point_tip_response():
    mesh = make_beam_stiffness_mesh()
    elem = mesh.elements[0]
    kernel = get_element_kernel(elem.type)
    Ke = kernel.stiffness(mesh, elem)
    frame = resolve_beam_frame(mesh, elem)
    L = frame.length
    rotation = frame.rotation
    E = float(elem.props["E"])
    nu = float(elem.props["nu"])
    section = parse_beam2_section(elem.props)
    Izz = section.Izz
    shear_modulus = E / (2.0 * (1.0 + nu))
    shear_y, _ = section.abaqus_b31_shear_rigidities(shear_modulus, nu, L)
    free = mesh.node_dofs(elem.node_ids[1])
    F = np.zeros(mesh.num_dofs, dtype=float)
    F[free[:3]] = rotation[1]

    U_tip = np.linalg.solve(Ke[np.ix_(free, free)], F[free])
    # A single B31 element has v = L * theta / 2 + L / shear_y under unit load.
    v_local = L**3 / (4.0 * E * Izz) + L / shear_y
    expected = np.concatenate([
        rotation[1] * v_local,
        rotation[2] * (L**2 / (2.0 * E * Izz)),
    ])

    assert np.allclose(U_tip, expected)


@pytest.mark.parametrize(
    "builder",
    [make_truss_stiffness_mesh, make_beam_stiffness_mesh],
    ids=["truss2", "beam2"],
)
def test_line_element_body_force_preserves_global_resultant(builder):
    mesh = builder()
    elem = mesh.elements[0]
    kernel = get_element_kernel(elem.type)
    vector = np.array([4.0, -5.0, 2.0])
    length, _ = line3d_geometry(mesh, elem)
    area = (
        parse_beam2_section(elem.props).area
        if str(elem.type).casefold() == "beam2"
        else float(elem.props["area"])
    )

    fe = kernel.body_force(mesh, elem, tuple(vector))
    stride = mesh.dofs_per_node

    assert float(fe[0::stride].sum()) == pytest.approx(area * length * vector[0])
    assert float(fe[1::stride].sum()) == pytest.approx(area * length * vector[1])
    assert float(fe[2::stride].sum()) == pytest.approx(area * length * vector[2])


def test_beam_body_force_preserves_global_moment():
    mesh = make_beam_stiffness_mesh()
    elem = mesh.elements[0]
    vector = np.array([4.0, -5.0, 2.0])
    fe = get_element_kernel(elem.type).body_force(mesh, elem, tuple(vector))
    length, _ = line3d_geometry(mesh, elem)
    area = parse_beam2_section(elem.props).area
    node_lookup = _node_lookup(mesh)
    ni = node_lookup[elem.node_ids[0]]
    nj = node_lookup[elem.node_ids[1]]
    total_force = area * length * vector
    midpoint = np.array([
        (ni.x + nj.x) / 2.0,
        (ni.y + nj.y) / 2.0,
        (ni.z + nj.z) / 2.0,
    ])

    assembled_moment = (
        np.cross([ni.x, ni.y, ni.z], fe[:3]) + fe[3:6]
        + np.cross([nj.x, nj.y, nj.z], fe[6:9]) + fe[9:12]
    )
    expected_moment = np.cross(midpoint, total_force)

    assert assembled_moment == pytest.approx(expected_moment)


def test_beam_stiffness_is_invariant_to_element_node_order():
    mesh = make_beam_stiffness_mesh()
    elem = mesh.elements[0]
    kernel = get_element_kernel(elem.type)
    K_forward = kernel.stiffness(mesh, elem)

    elem.node_ids = list(reversed(elem.node_ids))
    K_reverse = kernel.stiffness(mesh, elem)
    swap = np.array([6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5])

    assert np.allclose(K_forward, K_reverse[np.ix_(swap, swap)])


@pytest.mark.parametrize(
    "builder",
    [make_truss_stiffness_mesh, make_beam_stiffness_mesh],
    ids=["truss2", "beam2"],
)
def test_line_element_rejects_zero_length(builder):
    mesh = builder()
    elem = mesh.elements[0]
    node_lookup = _node_lookup(mesh)
    ni = node_lookup[elem.node_ids[0]]
    nj = node_lookup[elem.node_ids[1]]
    nj.x = ni.x
    nj.y = ni.y
    nj.z = ni.z

    with pytest.raises(ValueError, match="zero length"):
        get_element_kernel(elem.type).stiffness(mesh, elem)


@pytest.mark.parametrize(
    "builder",
    [make_truss_stiffness_mesh, make_beam_stiffness_mesh],
    ids=["truss2", "beam2"],
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
        (make_beam_stiffness_mesh, "width"),
    ],
    ids=["truss2", "beam2"],
)
def test_line_element_reports_missing_required_property(builder, missing_property):
    mesh = builder()
    elem = mesh.elements[0]
    elem.props.pop(missing_property)

    with pytest.raises(
        KeyError,
        match=rf"missing property {missing_property}",
    ):
        get_element_kernel(elem.type).stiffness(mesh, elem)
