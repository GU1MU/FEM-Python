"""Infinitesimal-strain kinematics for material-aware static analysis."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .contracts import KinematicPoint


@dataclass(frozen=True, slots=True)
class SmallStrainKinematics:
    """Compute the symmetric displacement gradient without updating geometry."""

    kinematic_measure = "small_strain"

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
            raise ValueError(
                "reference and displacement coordinates must have the same shape"
            )
        if gradients.shape != (reference.shape[1], reference.shape[0]):
            raise ValueError(
                "reference_shape_gradients must have shape (dimension, node_count)"
            )
        if not (
            np.all(np.isfinite(reference))
            and np.all(np.isfinite(trial_displacement))
            and np.all(np.isfinite(gradients))
        ):
            raise ValueError("small-strain kinematic inputs must be finite")
        displacement_gradient = trial_displacement.T @ gradients.T
        strain = 0.5 * (displacement_gradient + displacement_gradient.T)
        identity = np.eye(reference.shape[1], dtype=float)
        deformation_gradient = identity + displacement_gradient
        return KinematicPoint(
            reference_shape_gradients=gradients,
            deformation_gradient=deformation_gradient,
            green_lagrange_strain=strain,
            reference_jacobian_determinant=reference_jacobian_determinant,
            small_strain=strain,
        )


__all__ = ["SmallStrainKinematics"]
