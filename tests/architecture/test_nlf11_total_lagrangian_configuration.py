from __future__ import annotations

import numpy as np
import pytest

from fem.elements.quad4 import quad4_shape_grad_xi_eta
from fem.physics.mechanics import TotalLagrangianKinematics


def _evaluate(reference: np.ndarray, displacement: np.ndarray):
    kinematics = TotalLagrangianKinematics()
    if (
        reference.ndim != 2
        or displacement.shape != reference.shape
        or not np.all(np.isfinite(reference))
    ):
        gradients = np.zeros((2, reference.shape[0]), dtype=float)
        return kinematics.evaluate(reference, displacement, gradients, 1.0)
    gradients_natural = quad4_shape_grad_xi_eta(0.0, 0.0)
    jacobian = gradients_natural @ reference
    determinant = float(np.linalg.det(jacobian))
    gradients_reference = np.linalg.solve(jacobian, gradients_natural)
    return kinematics.evaluate(
        reference,
        displacement,
        gradients_reference,
        determinant,
    )


def test_total_lagrangian_evaluation_keeps_reference_and_returns_trial_kinematics():
    reference = np.array(
        [[0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [0.0, 1.0]],
    )
    displacement = np.array(
        [[0.0, 0.0], [0.1, 0.0], [0.1, 0.05], [0.0, 0.05]],
    )
    reference_before = reference.copy()
    displacement_before = displacement.copy()

    point = _evaluate(reference, displacement)

    assert point.deformation_gradient.shape == (2, 2)
    assert point.green_lagrange_strain.shape == (2, 2)
    assert point.reference_jacobian_determinant > 0.0
    np.testing.assert_allclose(reference, reference_before)
    np.testing.assert_allclose(displacement, displacement_before)


def test_total_lagrangian_zero_displacement_is_identity():
    reference = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])

    point = _evaluate(reference, np.zeros_like(reference))

    np.testing.assert_allclose(point.deformation_gradient, np.eye(2))
    np.testing.assert_allclose(point.green_lagrange_strain, 0.0)


def test_kinematics_rejects_mismatched_or_nonfinite_arrays():
    reference = np.zeros((4, 2))
    with pytest.raises(ValueError, match="same shape"):
        _evaluate(reference, np.zeros((4, 3)))
    with pytest.raises(ValueError, match="finite"):
        _evaluate(np.array([[0.0, np.inf], [1.0, 0.0]]), np.zeros((2, 2)))
