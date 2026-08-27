"""Kinematics contracts independent of element topology and materials."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True, slots=True)
class KinematicPoint:
    """Motion state at one integration point."""

    reference_shape_gradients: np.ndarray
    deformation_gradient: np.ndarray
    green_lagrange_strain: np.ndarray
    reference_jacobian_determinant: float
    small_strain: np.ndarray | None = None

    def __post_init__(self) -> None:
        for name in (
            "reference_shape_gradients",
            "deformation_gradient",
            "green_lagrange_strain",
        ):
            value = np.asarray(getattr(self, name), dtype=float)
            if not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must contain finite values")
            owned = np.array(value, copy=True)
            owned.flags.writeable = False
            object.__setattr__(self, name, owned)
        if self.small_strain is not None:
            value = np.asarray(self.small_strain, dtype=float)
            if value.shape not in {(2, 2), (3, 3)}:
                raise ValueError("small_strain must be a 2x2 or 3x3 tensor")
            if not np.all(np.isfinite(value)):
                raise ValueError("small_strain must contain finite values")
            owned = np.array(value, copy=True)
            owned.flags.writeable = False
            object.__setattr__(self, "small_strain", owned)
        determinant = float(self.reference_jacobian_determinant)
        if not np.isfinite(determinant) or determinant <= 0.0:
            raise ValueError("reference_jacobian_determinant must be > 0")
        object.__setattr__(self, "reference_jacobian_determinant", determinant)


@runtime_checkable
class KinematicsModel(Protocol):
    """Convert local reference geometry and displacement to a kinematic point."""

    def evaluate(
        self,
        reference_coordinates: np.ndarray,
        displacement: np.ndarray,
        reference_shape_gradients: np.ndarray,
        reference_jacobian_determinant: float,
    ) -> KinematicPoint:
        """Evaluate one integration point."""


__all__ = ["KinematicPoint", "KinematicsModel"]
