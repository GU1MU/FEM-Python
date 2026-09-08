import numpy as np
import pytest

from fem.core.mesh import Element3D, Mesh3D, Node3D
from fem.elements import get_element_kernel
from fem.boundary.condition import BoundaryCondition
from fem.boundary.loads import build_load_vector
from tests.helpers.mesh_builders import make_truss_stiffness_mesh


def _truss_mesh(*, reversed_nodes=False, props=None):
    nodes = [Node3D(10, 1.0, -2.0, 0.5), Node3D(20, 3.0, 1.0, 6.5)]
    node_ids = [20, 10] if reversed_nodes else [10, 20]
    return Mesh3D(
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


def test_truss_rejects_zero_length():
    mesh = make_truss_stiffness_mesh()
    elem = mesh.elements[0]
    node_lookup = {node.id: node for node in mesh.nodes}
    ni = node_lookup[elem.node_ids[0]]
    nj = node_lookup[elem.node_ids[1]]
    nj.x = ni.x
    nj.y = ni.y
    nj.z = ni.z

    with pytest.raises(ValueError, match="zero length"):
        get_element_kernel(elem.type).stiffness(mesh, elem)


def test_truss_reports_missing_node():
    mesh = make_truss_stiffness_mesh()
    elem = mesh.elements[0]
    elem.node_ids[1] = 999

    with pytest.raises(KeyError, match=r"Element 1 references missing node 999"):
        get_element_kernel(elem.type).stiffness(mesh, elem)


def test_truss_reports_missing_required_property():
    missing_property = "area"
    mesh = make_truss_stiffness_mesh()
    elem = mesh.elements[0]
    elem.props.pop(missing_property)

    with pytest.raises(
        KeyError,
        match=rf"missing property {missing_property}",
    ):
        get_element_kernel(elem.type).stiffness(mesh, elem)


def test_truss_stiffness_supports_default_and_explicit_node_lookup():
    mesh = make_truss_stiffness_mesh()
    element = mesh.elements[0]
    kernel = get_element_kernel(element.type)
    expected = kernel.stiffness(mesh, element)
    actual = kernel.stiffness(mesh, element, {node.id: node for node in mesh.nodes})
    np.testing.assert_allclose(actual, expected)
