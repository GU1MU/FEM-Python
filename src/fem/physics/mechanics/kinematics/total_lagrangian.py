"""Total Lagrangian motion description shared by compatible elements."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .contracts import KinematicPoint


@dataclass(frozen=True, slots=True)
class TotalLagrangianKinematics:
    """Compute ``F`` and Green--Lagrange ``E`` from reference data."""

    kinematic_measure = "deformation_gradient"

    def evaluate(
        self,
        reference_coordinates: np.ndarray,
        displacement: np.ndarray,
        reference_shape_gradients: np.ndarray,
        reference_jacobian_determinant: float,
    ) -> KinematicPoint:
        reference = np.asarray(reference_coordinates, dtype=float)
        trial_displacement = np.asarray(displacement, dtype=float)
        gradients = np.asarray(reference_shape_gradients, dtype=float)
        if reference.ndim != 2 or trial_displacement.shape != reference.shape:
            raise ValueError("reference and displacement coordinates must have the same shape")
        if gradients.shape != (reference.shape[1], reference.shape[0]):
            raise ValueError(
                "reference_shape_gradients must have shape (dimension, node_count)"
            )
        if not np.all(np.isfinite(reference)) or not np.all(np.isfinite(trial_displacement)):
            raise ValueError("reference coordinates and displacement must be finite")
        if not np.all(np.isfinite(gradients)):
            raise ValueError("reference_shape_gradients must be finite")
        current = reference + trial_displacement
        deformation_gradient = current.T @ gradients.T
        determinant = float(np.linalg.det(deformation_gradient))
        if determinant <= 0.0:
            raise ValueError("deformation gradient determinant must be > 0")
        green_lagrange_strain = 0.5 * (
            deformation_gradient.T @ deformation_gradient - np.eye(reference.shape[1])
        )
        return KinematicPoint(
            reference_shape_gradients=gradients,
            deformation_gradient=deformation_gradient,
            green_lagrange_strain=green_lagrange_strain,
            reference_jacobian_determinant=reference_jacobian_determinant,
        )


__all__ = ["TotalLagrangianKinematics"]
