from __future__ import annotations

import numpy as np
import pytest

from fem.elements.quad4 import quad4_shape_grad_xi_eta
from fem.physics.mechanics import TotalLagrangianKinematics, get_recovery_service
from tests.helpers.mesh_builders import make_quad4_stiffness_mesh


def _reference_coordinates() -> np.ndarray:
    mesh = make_quad4_stiffness_mesh()
    return np.asarray(
        [[node.x, node.y] for node in mesh.nodes],
        dtype=float,
    )


def _at_center(reference: np.ndarray, displacement: np.ndarray):
    natural_gradients = quad4_shape_grad_xi_eta(0.0, 0.0)
    reference_jacobian = natural_gradients @ reference
    reference_det = float(np.linalg.det(reference_jacobian))
    reference_gradients = np.linalg.solve(reference_jacobian, natural_gradients)
    return TotalLagrangianKinematics().evaluate(
        reference,
        displacement,
        reference_gradients,
        reference_det,
    )


def test_quad4_total_lagrangian_zero_displacement_is_identity() -> None:
    reference = _reference_coordinates()
    kinematics = _at_center(reference, np.zeros_like(reference))

    np.testing.assert_allclose(kinematics.deformation_gradient, np.eye(2))
    np.testing.assert_allclose(kinematics.green_lagrange_strain, 0.0)
    assert kinematics.reference_jacobian_determinant > 0.0


def test_quad4_total_lagrangian_small_displacement_matches_linear_strain() -> None:
    mesh = make_quad4_stiffness_mesh()
    element = mesh.elements[0]
    reference = _reference_coordinates()
    displacement = np.array(
        [[0.0, 0.0], [1.0e-7, -2.0e-7], [2.0e-7, 1.0e-7], [0.0, 2.0e-7]],
    )
    local_displacement = displacement.reshape(-1)
    kinematics = _at_center(reference, displacement)
    linear_B = get_recovery_service(element.type)._B_matrix(
        mesh,
        element,
        0.0,
        0.0,
        None,
    )[0]
    engineering_strain = np.array(
        [
            kinematics.green_lagrange_strain[0, 0],
            kinematics.green_lagrange_strain[1, 1],
            2.0 * kinematics.green_lagrange_strain[0, 1],
        ]
    )

    np.testing.assert_allclose(
        engineering_strain,
        linear_B @ local_displacement,
        rtol=1.0e-6,
        atol=1.0e-13,
    )


def test_quad4_total_lagrangian_rigid_rotation_has_zero_green_strain() -> None:
    reference = _reference_coordinates()
    angle = 0.2
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]],
    )
    current = reference @ rotation.T
    kinematics = _at_center(reference, current - reference)

    np.testing.assert_allclose(kinematics.deformation_gradient, rotation)
    np.testing.assert_allclose(
        kinematics.green_lagrange_strain,
        0.0,
        atol=1.0e-14,
    )


def test_quad4_total_lagrangian_rejects_inverted_reference_element() -> None:
    reference = _reference_coordinates()[::-1]

    with pytest.raises(ValueError, match="jacobian_determinant"):
        _at_center(reference, np.zeros_like(reference))
