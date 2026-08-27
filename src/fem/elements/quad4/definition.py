"""Geometric definition of the bilinear four-node quadrilateral."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def quad4_shape_functions(xi: float, eta: float) -> np.ndarray:
    """Return the four bilinear shape functions at ``(xi, eta)``."""

    return 0.25 * np.array(
        [
            (1.0 - xi) * (1.0 - eta),
            (1.0 + xi) * (1.0 - eta),
            (1.0 + xi) * (1.0 + eta),
            (1.0 - xi) * (1.0 + eta),
        ],
        dtype=float,
    )


def quad4_shape_grad_xi_eta(xi: float, eta: float) -> np.ndarray:
    """Return ``dN/dxi`` and ``dN/deta`` for bilinear Quad4."""

    dN_dxi = np.array(
        [-(1.0 - eta), (1.0 - eta), (1.0 + eta), -(1.0 + eta)],
        dtype=float,
    ) * 0.25
    dN_deta = np.array(
        [-(1.0 - xi), -(1.0 + xi), (1.0 + xi), (1.0 - xi)],
        dtype=float,
    ) * 0.25
    return np.vstack([dN_dxi, dN_deta])


def quad4_gauss_points(gauss_order: int) -> tuple[tuple[float, float, float], ...]:
    """Return the supported tensor-product Gauss rule for Quad4."""

    if gauss_order == 1:
        return ((0.0, 0.0, 4.0),)
    if gauss_order == 2:
        a = 1.0 / np.sqrt(3.0)
        return ((-a, -a, 1.0), (a, -a, 1.0), (a, a, 1.0), (-a, a, 1.0))
    raise ValueError("gauss_order must be 1 or 2")


@dataclass(frozen=True, slots=True)
class Quad4Definition:
    """Immutable spatial definition shared by linear and nonlinear Quad4 paths."""

    gauss_order: int = 2

    canonical_type = "Quad4"
    aliases = ("CPS4", "CPE4")
    node_count = 4

    def __post_init__(self) -> None:
        if isinstance(self.gauss_order, bool) or self.gauss_order not in (1, 2):
            raise ValueError("Quad4 gauss_order must be 1 or 2")

    def gauss_points(self) -> tuple[tuple[float, float, float], ...]:
        """Return this definition's integration rule."""

        return quad4_gauss_points(self.gauss_order)

    @staticmethod
    def shape_functions(xi: float, eta: float) -> np.ndarray:
        """Return shape functions at one natural coordinate."""

        return quad4_shape_functions(xi, eta)

    @staticmethod
    def shape_gradients(xi: float, eta: float) -> np.ndarray:
        """Return natural-coordinate shape gradients."""

        return quad4_shape_grad_xi_eta(xi, eta)


__all__ = [
    "Quad4Definition",
    "quad4_gauss_points",
    "quad4_shape_functions",
    "quad4_shape_grad_xi_eta",
]
