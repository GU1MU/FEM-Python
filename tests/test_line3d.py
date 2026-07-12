import numpy as np
import pytest

from fem.boundary.condition import BoundaryCondition
from fem.boundary.loads import build_load_vector
from fem.core.mesh import BeamMesh3D, Element3D, Node3D, TrussMesh3D
from fem.elements import get_element_kernel
from fem.elements.line import beam3_geometry


def _truss_mesh(*, reversed_nodes=False, props=None):
    nodes = [Node3D(10, 1.0, -2.0, 0.5), Node3D(20, 3.0, 1.0, 6.5)]
    node_ids = [20, 10] if reversed_nodes else [10, 20]
    return TrussMesh3D(
        nodes=nodes,
        elements=[
            Element3D(
                1,
                node_ids,
                "Truss2",
                props or {"E": 210.0, "area": 2.5, "rho": 4.0},
            )
        ],
    )


def test_truss2_spatial_stiffness_matches_outer_product_contract():
    mesh = _truss_mesh()
    elem = mesh.elements[0]
    delta = np.array([2.0, 3.0, 6.0])
    length = np.linalg.norm(delta)
    direction = delta / length
    block = np.outer(direction, direction)
    expected = elem.props["E"] * elem.props["area"] / length * np.block(
        [[block, -block], [-block, block]]
    )

    stiffness = get_element_kernel("Truss2").stiffness(mesh, elem)

    assert stiffness == pytest.approx(expected)
    assert stiffness == pytest.approx(stiffness.T)
    assert np.linalg.matrix_rank(stiffness, tol=1e-10) == 1


def test_truss2_node_reversal_only_permutes_element_stiffness():
    forward = _truss_mesh()
    reversed_mesh = _truss_mesh(reversed_nodes=True)
    kernel = get_element_kernel("Truss2")
    permutation = np.eye(6)[[3, 4, 5, 0, 1, 2]]

    forward_stiffness = kernel.stiffness(forward, forward.elements[0])
    reversed_stiffness = kernel.stiffness(reversed_mesh, reversed_mesh.elements[0])

    assert reversed_stiffness == pytest.approx(
        permutation @ forward_stiffness @ permutation.T
    )


def test_truss2_rigid_translation_and_axial_extension_results():
    mesh = _truss_mesh()
    elem = mesh.elements[0]
    kernel = get_element_kernel("Truss2")
    direction = np.array([2.0, 3.0, 6.0]) / 7.0
    rigid = np.tile([0.4, -0.2, 0.7], 2)

    assert kernel.stiffness(mesh, elem) @ rigid == pytest.approx(np.zeros(6), abs=1e-12)
    assert kernel.element_stress(mesh, elem, rigid) == pytest.approx((0.0, 0.0, 0.0))

    extension = 0.14
    displacement = np.concatenate([np.zeros(3), extension * direction])
    strain, stress, mises = kernel.element_stress(mesh, elem, displacement)

    assert strain == pytest.approx(extension / 7.0)
    assert stress == pytest.approx(elem.props["E"] * extension / 7.0)
    assert mises == pytest.approx(abs(stress))


def test_truss2_body_force_and_gravity_preserve_total_force():
    mesh = _truss_mesh()
    elem = mesh.elements[0]
    kernel = get_element_kernel("Truss2")
    body_vector = np.array([1.5, -2.0, 0.25])
    expected_total = body_vector * elem.props["area"] * 7.0

    element_force = kernel.body_force(mesh, elem, tuple(body_vector))

    assert element_force[:3] == pytest.approx(expected_total / 2.0)
    assert element_force[3:] == pytest.approx(expected_total / 2.0)

    boundary = BoundaryCondition()
    boundary.set_gravity(0.0, 0.0, -9.81)
    gravity_force = build_load_vector(mesh, boundary)

    assert gravity_force.reshape(2, 3).sum(axis=0) == pytest.approx(
        [0.0, 0.0, -9.81 * elem.props["rho"] * elem.props["area"] * 7.0]
    )


@pytest.mark.parametrize("vector", [(1.0, 2.0), (1.0, 2.0, np.nan)])
def test_truss2_rejects_invalid_body_force_vectors(vector):
    mesh = _truss_mesh()

    with pytest.raises(ValueError, match="Truss2 body force"):
        get_element_kernel("Truss2").body_force(mesh, mesh.elements[0], vector)


def _beam_mesh(*, end=(4.0, 0.0, 0.0), local_y=(0.0, 1.0, 0.0), props=None):
    properties = {
        "E": 210.0,
        "nu": 0.25,
        "area": 3.0,
        "Iyy": 5.0,
        "Izz": 7.0,
        "J": 2.0,
        "local_y": local_y,
        "rho": 4.0,
    }
    if props:
        properties.update(props)
    return BeamMesh3D(
        nodes=[Node3D(1, 0.0, 0.0, 0.0), Node3D(2, *end)],
        elements=[Element3D(1, [1, 2], "Beam2", properties)],
    )


def test_beam2_local_axes_are_orthonormal_and_right_handed():
    mesh = _beam_mesh(end=(2.0, 3.0, 6.0), local_y=(1.0, 1.0, 0.0))

    length, rotation = beam3_geometry(mesh, mesh.elements[0])

    assert length == pytest.approx(7.0)
    assert rotation @ rotation.T == pytest.approx(np.eye(3), abs=1e-12)
    assert np.linalg.det(rotation) == pytest.approx(1.0)
    assert rotation[0] == pytest.approx(np.array([2.0, 3.0, 6.0]) / 7.0)


@pytest.mark.parametrize(
    ("properties", "message"),
    [
        ({"local_y": None}, "local_y"),
        ({"local_y": (0.0, 0.0, 0.0)}, "local_y"),
        ({"local_y": (1.0, np.nan, 0.0)}, "local_y"),
        ({"local_y": (2.0, 0.0, 0.0)}, "parallel"),
    ],
)
def test_beam2_rejects_invalid_local_y(properties, message):
    mesh = _beam_mesh(props=properties)

    with pytest.raises((KeyError, ValueError), match=message):
        get_element_kernel("Beam2").stiffness(mesh, mesh.elements[0])


@pytest.mark.parametrize("mode", range(6))
def test_beam2_six_rigid_body_modes_have_zero_internal_force(mode):
    mesh = _beam_mesh(end=(2.0, 3.0, 6.0), local_y=(1.0, 1.0, 0.0))
    elem = mesh.elements[0]
    stiffness = get_element_kernel("Beam2").stiffness(mesh, elem)
    displacement = np.zeros(12)

    if mode < 3:
        displacement[mode] = 1.0
        displacement[6 + mode] = 1.0
    else:
        omega = np.eye(3)[mode - 3]
        end_position = np.array([2.0, 3.0, 6.0])
        displacement[3:6] = omega
        displacement[6:9] = np.cross(omega, end_position)
        displacement[9:12] = omega

    assert stiffness @ displacement == pytest.approx(np.zeros(12), abs=1e-10)


@pytest.mark.parametrize(
    ("dof", "load", "expected"),
    [
        (0, 12.0, 12.0 * 4.0 / (210.0 * 3.0)),
        (1, 12.0, 12.0 * 4.0**3 / (3.0 * 210.0 * 7.0)),
        (2, 12.0, 12.0 * 4.0**3 / (3.0 * 210.0 * 5.0)),
        (3, 12.0, 12.0 * 4.0 / ((210.0 / 2.5) * 2.0)),
    ],
    ids=["axial", "bend-local-y", "bend-local-z", "torsion"],
)
def test_beam2_cantilever_matches_closed_form_tip_response(dof, load, expected):
    mesh = _beam_mesh()
    stiffness = get_element_kernel("Beam2").stiffness(mesh, mesh.elements[0])
    free_stiffness = stiffness[6:12, 6:12]
    force = np.zeros(6)
    force[dof] = load

    tip = np.linalg.solve(free_stiffness, force)

    assert tip[dof] == pytest.approx(expected)


def test_beam2_inclined_stiffness_is_symmetric_and_reversal_invariant():
    mesh = _beam_mesh(end=(2.0, 3.0, 6.0), local_y=(1.0, 1.0, 0.0))
    reversed_mesh = _beam_mesh(end=(2.0, 3.0, 6.0), local_y=(1.0, 1.0, 0.0))
    reversed_mesh.elements[0].node_ids = [2, 1]
    kernel = get_element_kernel("Beam2")
    permutation = np.eye(12)[[6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5]]

    stiffness = kernel.stiffness(mesh, mesh.elements[0])
    reversed_stiffness = kernel.stiffness(reversed_mesh, reversed_mesh.elements[0])

    assert stiffness == pytest.approx(stiffness.T, abs=1e-12)
    assert reversed_stiffness == pytest.approx(
        permutation @ stiffness @ permutation.T,
        abs=1e-10,
    )


def test_beam2_body_force_preserves_resultant_and_moment():
    mesh = _beam_mesh(end=(2.0, 3.0, 6.0), local_y=(1.0, 1.0, 0.0))
    elem = mesh.elements[0]
    body_vector = np.array([1.5, -2.0, 0.25])
    element_force = get_element_kernel("Beam2").body_force(
        mesh, elem, tuple(body_vector)
    )
    total_force = body_vector * elem.props["area"] * 7.0
    end = np.array([2.0, 3.0, 6.0])
    assembled_moment = (
        np.cross(np.zeros(3), element_force[:3])
        + element_force[3:6]
        + np.cross(end, element_force[6:9])
        + element_force[9:12]
    )

    assert element_force[:3] + element_force[6:9] == pytest.approx(total_force)
    assert assembled_moment == pytest.approx(np.cross(end / 2.0, total_force))
