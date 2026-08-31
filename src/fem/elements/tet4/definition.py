"""Geometric definition of the linear four-node tetrahedron."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def tet4_shape_functions(xi: float, eta: float, zeta: float) -> np.ndarray:
    """Return the barycentric shape functions on the reference tetrahedron."""

    return np.array(
        [1.0 - xi - eta - zeta, xi, eta, zeta],
        dtype=float,
    )


def tet4_shape_gradients(xi: float, eta: float, zeta: float) -> np.ndarray:
    """Return the constant ``dN/d(xi, eta, zeta)`` matrix."""

    del xi, eta, zeta
    return np.array(
        [
            [-1.0, 1.0, 0.0, 0.0],
            [-1.0, 0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


def tet4_shape_funcs_grads(
    xi: float,
    eta: float,
    zeta: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return ``N`` and the three parent-coordinate gradient rows."""

    gradients = tet4_shape_gradients(xi, eta, zeta)
    return (
        tet4_shape_functions(xi, eta, zeta),
        gradients[0],
        gradients[1],
        gradients[2],
    )


def tet4_gauss_points(gauss_order: int = 1) -> tuple[tuple[float, float, float, float], ...]:
    """Return the one-point full rule for the linear tetrahedron."""

    if gauss_order != 1:
        raise ValueError("Tet4 supports only the one-point full integration rule")
    return ((0.25, 0.25, 0.25, 1.0 / 6.0),)


@dataclass(frozen=True, slots=True)
class Tet4Definition:
    """Immutable spatial definition shared by Tet4 mechanics paths."""

    gauss_order: int = 1

    canonical_type = "Tet4"
    aliases = ("C3D4",)
    node_count = 4

    def __post_init__(self) -> None:
        if isinstance(self.gauss_order, bool) or self.gauss_order != 1:
            raise ValueError("Tet4 gauss_order must be 1")

    def gauss_points(self) -> tuple[tuple[float, float, float, float], ...]:
        return tet4_gauss_points(self.gauss_order)

    @staticmethod
    def shape_functions(xi: float, eta: float, zeta: float) -> np.ndarray:
        return tet4_shape_functions(xi, eta, zeta)

    @staticmethod
    def shape_gradients(xi: float, eta: float, zeta: float) -> np.ndarray:
        return tet4_shape_gradients(xi, eta, zeta)


__all__ = [
    "Tet4Definition",
    "tet4_gauss_points",
    "tet4_shape_functions",
    "tet4_shape_gradients",
    "tet4_shape_funcs_grads",
]
