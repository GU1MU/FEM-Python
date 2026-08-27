"""Geometric definition of the eight-node serendipity quadrilateral."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def quad8_shape_functions(xi: float, eta: float) -> np.ndarray:
    """Return Quad8 shape functions in standard corner-edge order."""

    return np.array(
        [
            0.25 * (1.0 - xi) * (1.0 - eta) * (-xi - eta - 1.0),
            0.25 * (1.0 + xi) * (1.0 - eta) * (xi - eta - 1.0),
            0.25 * (1.0 + xi) * (1.0 + eta) * (xi + eta - 1.0),
            0.25 * (1.0 - xi) * (1.0 + eta) * (-xi + eta - 1.0),
            0.5 * (1.0 - xi * xi) * (1.0 - eta),
            0.5 * (1.0 + xi) * (1.0 - eta * eta),
            0.5 * (1.0 - xi * xi) * (1.0 + eta),
            0.5 * (1.0 - xi) * (1.0 - eta * eta),
        ],
        dtype=float,
    )


def quad8_shape_gradients(xi: float, eta: float) -> np.ndarray:
    """Return ``dN/d(xi, eta)`` in standard corner-edge order."""

    dN_dxi = np.array(
        [
            0.25 * (-(1.0 - eta) * (-xi - eta - 1.0)
                    - (1.0 - xi) * (1.0 - eta)),
            0.25 * ((1.0 - eta) * (xi - eta - 1.0)
                    + (1.0 + xi) * (1.0 - eta)),
            0.25 * ((1.0 + eta) * (xi + eta - 1.0)
                    + (1.0 + xi) * (1.0 + eta)),
            0.25 * (-(1.0 + eta) * (-xi + eta - 1.0)
                    - (1.0 - xi) * (1.0 + eta)),
            -xi * (1.0 - eta),
            0.5 * (1.0 - eta * eta),
            -xi * (1.0 + eta),
            -0.5 * (1.0 - eta * eta),
        ],
        dtype=float,
    )
    dN_deta = np.array(
        [
            0.25 * (-(1.0 - xi) * (-xi - eta - 1.0)
                    - (1.0 - xi) * (1.0 - eta)),
            0.25 * (-(1.0 + xi) * (xi - eta - 1.0)
                    - (1.0 + xi) * (1.0 - eta)),
            0.25 * ((1.0 + xi) * (xi + eta - 1.0)
                    + (1.0 + xi) * (1.0 + eta)),
            0.25 * ((1.0 - xi) * (-xi + eta - 1.0)
                    + (1.0 - xi) * (1.0 + eta)),
            -0.5 * (1.0 - xi * xi),
            -(1.0 + xi) * eta,
            0.5 * (1.0 - xi * xi),
            -(1.0 - xi) * eta,
        ],
        dtype=float,
    )
    return np.vstack((dN_dxi, dN_deta))


def quad8_shape_funcs_grads(
    xi: float,
    eta: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``N`` and the two parent-coordinate gradient rows."""

    gradients = quad8_shape_gradients(xi, eta)
    return quad8_shape_functions(xi, eta), gradients[0], gradients[1]


def quad8_gauss_points(
    gauss_order: int = 3,
) -> tuple[tuple[float, float, float], ...]:
    """Return the supported tensor-product Gauss rule for Quad8."""

    if gauss_order == 2:
        radius = 1.0 / np.sqrt(3.0)
        return (
            (-radius, -radius, 1.0),
            (radius, -radius, 1.0),
            (radius, radius, 1.0),
            (-radius, radius, 1.0),
        )
    if gauss_order != 3:
        raise ValueError("Quad8 supports only the 2x2 or 3x3 integration rule")
    radius = np.sqrt(3.0 / 5.0)
    one_dimensional = (
        (-radius, 5.0 / 9.0),
        (0.0, 8.0 / 9.0),
        (radius, 5.0 / 9.0),
    )
    return tuple(
        (xi, eta, weight_xi * weight_eta)
        for xi, weight_xi in one_dimensional
        for eta, weight_eta in one_dimensional
    )


@dataclass(frozen=True, slots=True)
class Quad8Definition:
    """Immutable spatial definition shared by Quad8 mechanics paths."""

    gauss_order: int = 3

    canonical_type = "Quad8"
    aliases = ("CPS8", "CPE8")
    node_count = 8

    def __post_init__(self) -> None:
        if isinstance(self.gauss_order, bool) or self.gauss_order not in (2, 3):
            raise ValueError("Quad8 gauss_order must be 2 or 3")

    def gauss_points(self) -> tuple[tuple[float, float, float], ...]:
        return quad8_gauss_points(self.gauss_order)

    @staticmethod
    def shape_functions(xi: float, eta: float) -> np.ndarray:
        return quad8_shape_functions(xi, eta)

    @staticmethod
    def shape_gradients(xi: float, eta: float) -> np.ndarray:
        return quad8_shape_gradients(xi, eta)


__all__ = [
    "Quad8Definition",
    "quad8_gauss_points",
    "quad8_shape_functions",
    "quad8_shape_gradients",
    "quad8_shape_funcs_grads",
]
