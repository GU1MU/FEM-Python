"""Geometric definition of the twenty-node serendipity hexahedron."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


HEX20_NATURAL_NODE_COORDS = np.array(
    [
        (-1.0, -1.0, -1.0),
        (1.0, -1.0, -1.0),
        (1.0, 1.0, -1.0),
        (-1.0, 1.0, -1.0),
        (-1.0, -1.0, 1.0),
        (1.0, -1.0, 1.0),
        (1.0, 1.0, 1.0),
        (-1.0, 1.0, 1.0),
        (0.0, -1.0, -1.0),
        (1.0, 0.0, -1.0),
        (0.0, 1.0, -1.0),
        (-1.0, 0.0, -1.0),
        (0.0, -1.0, 1.0),
        (1.0, 0.0, 1.0),
        (0.0, 1.0, 1.0),
        (-1.0, 0.0, 1.0),
        (-1.0, -1.0, 0.0),
        (1.0, -1.0, 0.0),
        (1.0, 1.0, 0.0),
        (-1.0, 1.0, 0.0),
    ],
    dtype=float,
)


def hex20_shape_functions(
    xi: float,
    eta: float,
    zeta: float,
) -> np.ndarray:
    """Return Hex20 serendipity shape functions in corner-edge order."""

    values = np.empty(20, dtype=float)
    for index, (a, b, c) in enumerate(HEX20_NATURAL_NODE_COORDS):
        if a != 0.0 and b != 0.0 and c != 0.0:
            values[index] = (
                (1.0 + a * xi)
                * (1.0 + b * eta)
                * (1.0 + c * zeta)
                * (a * xi + b * eta + c * zeta - 2.0)
                / 8.0
            )
        elif a == 0.0:
            values[index] = (
                (1.0 - xi * xi)
                * (1.0 + b * eta)
                * (1.0 + c * zeta)
                / 4.0
            )
        elif b == 0.0:
            values[index] = (
                (1.0 - eta * eta)
                * (1.0 + a * xi)
                * (1.0 + c * zeta)
                / 4.0
            )
        else:
            values[index] = (
                (1.0 - zeta * zeta)
                * (1.0 + a * xi)
                * (1.0 + b * eta)
                / 4.0
            )
    return values


def hex20_shape_gradients(
    xi: float,
    eta: float,
    zeta: float,
) -> np.ndarray:
    """Return ``dN/d(xi, eta, zeta)`` in corner-edge order."""

    gradients = np.empty((3, 20), dtype=float)
    for index, (a, b, c) in enumerate(HEX20_NATURAL_NODE_COORDS):
        if a != 0.0 and b != 0.0 and c != 0.0:
            gradients[0, index] = (
                a
                * (1.0 + b * eta)
                * (1.0 + c * zeta)
                * (2.0 * a * xi + b * eta + c * zeta - 1.0)
                / 8.0
            )
            gradients[1, index] = (
                b
                * (1.0 + a * xi)
                * (1.0 + c * zeta)
                * (a * xi + 2.0 * b * eta + c * zeta - 1.0)
                / 8.0
            )
            gradients[2, index] = (
                c
                * (1.0 + a * xi)
                * (1.0 + b * eta)
                * (a * xi + b * eta + 2.0 * c * zeta - 1.0)
                / 8.0
            )
        elif a == 0.0:
            gradients[0, index] = (
                -xi * (1.0 + b * eta) * (1.0 + c * zeta) / 2.0
            )
            gradients[1, index] = (
                b * (1.0 - xi * xi) * (1.0 + c * zeta) / 4.0
            )
            gradients[2, index] = (
                c * (1.0 - xi * xi) * (1.0 + b * eta) / 4.0
            )
        elif b == 0.0:
            gradients[0, index] = (
                a * (1.0 - eta * eta) * (1.0 + c * zeta) / 4.0
            )
            gradients[1, index] = (
                -eta * (1.0 + a * xi) * (1.0 + c * zeta) / 2.0
            )
            gradients[2, index] = (
                c * (1.0 - eta * eta) * (1.0 + a * xi) / 4.0
            )
        else:
            gradients[0, index] = (
                a * (1.0 - zeta * zeta) * (1.0 + b * eta) / 4.0
            )
            gradients[1, index] = (
                b * (1.0 - zeta * zeta) * (1.0 + a * xi) / 4.0
            )
            gradients[2, index] = (
                -zeta * (1.0 + a * xi) * (1.0 + b * eta) / 2.0
            )
    return gradients


def hex20_shape_funcs_grads(
    xi: float,
    eta: float,
    zeta: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return ``N`` and the three parent-coordinate gradient rows."""

    gradients = hex20_shape_gradients(xi, eta, zeta)
    return (
        hex20_shape_functions(xi, eta, zeta),
        gradients[0],
        gradients[1],
        gradients[2],
    )


HEX20_FACE_NODE_INDICES = (
    (0, 3, 2, 1, 11, 10, 9, 8),
    (4, 5, 6, 7, 12, 13, 14, 15),
    (0, 1, 5, 4, 8, 17, 12, 16),
    (2, 3, 7, 6, 10, 19, 14, 18),
    (0, 4, 7, 3, 16, 15, 19, 11),
    (1, 2, 6, 5, 9, 18, 13, 17),
)


def hex20_gauss_points(
    gauss_order: int = 3,
) -> tuple[tuple[float, float, float, float], ...]:
    """Return the 3×3×3 full-integration rule for Hex20."""

    if gauss_order != 3:
        raise ValueError("Hex20 supports only the 3x3x3 full integration rule")
    radius = np.sqrt(3.0 / 5.0)
    one_dimensional = (
        (-radius, 5.0 / 9.0),
        (0.0, 8.0 / 9.0),
        (radius, 5.0 / 9.0),
    )
    return tuple(
        (xi, eta, zeta, weight_xi * weight_eta * weight_zeta)
        for xi, weight_xi in one_dimensional
        for eta, weight_eta in one_dimensional
        for zeta, weight_zeta in one_dimensional
    )


@dataclass(frozen=True, slots=True)
class Hex20Definition:
    """Immutable spatial definition shared by Hex20 mechanics paths."""

    gauss_order: int = 3

    canonical_type = "Hex20"
    aliases = ("C3D20",)
    node_count = 20

    def __post_init__(self) -> None:
        if isinstance(self.gauss_order, bool) or self.gauss_order != 3:
            raise ValueError("Hex20 gauss_order must be 3")

    def gauss_points(self) -> tuple[tuple[float, float, float, float], ...]:
        return hex20_gauss_points(self.gauss_order)

    @staticmethod
    def shape_functions(xi: float, eta: float, zeta: float) -> np.ndarray:
        return hex20_shape_functions(xi, eta, zeta)

    @staticmethod
    def shape_gradients(xi: float, eta: float, zeta: float) -> np.ndarray:
        return hex20_shape_gradients(xi, eta, zeta)


__all__ = [
    "Hex20Definition",
    "HEX20_FACE_NODE_INDICES",
    "HEX20_NATURAL_NODE_COORDS",
    "hex20_gauss_points",
    "hex20_shape_functions",
    "hex20_shape_gradients",
    "hex20_shape_funcs_grads",
]
