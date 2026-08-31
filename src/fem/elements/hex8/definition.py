"""Geometric definition of the trilinear eight-node hexahedron."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


_HEX8_NODE_SIGNS = np.array(
    [
        (-1.0, -1.0, -1.0),
        (1.0, -1.0, -1.0),
        (1.0, 1.0, -1.0),
        (-1.0, 1.0, -1.0),
        (-1.0, -1.0, 1.0),
        (1.0, -1.0, 1.0),
        (1.0, 1.0, 1.0),
        (-1.0, 1.0, 1.0),
    ],
    dtype=float,
)


def hex8_shape_functions(xi: float, eta: float, zeta: float) -> np.ndarray:
    """Return the trilinear shape functions in standard corner order."""

    coordinates = np.array([xi, eta, zeta], dtype=float)
    return 0.125 * np.prod(1.0 + _HEX8_NODE_SIGNS * coordinates, axis=1)


def hex8_shape_gradients(xi: float, eta: float, zeta: float) -> np.ndarray:
    """Return ``dN/d(xi, eta, zeta)`` in standard corner order."""

    coordinates = np.array([xi, eta, zeta], dtype=float)
    gradients = np.empty((3, 8), dtype=float)
    for direction in range(3):
        factors = 1.0 + _HEX8_NODE_SIGNS * coordinates
        factors[:, direction] = _HEX8_NODE_SIGNS[:, direction]
        gradients[direction] = 0.125 * np.prod(factors, axis=1)
    return gradients


def hex8_shape_funcs_grads(
    xi: float,
    eta: float,
    zeta: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return ``N`` and the three parent-coordinate gradient rows."""

    gradients = hex8_shape_gradients(xi, eta, zeta)
    return (
        hex8_shape_functions(xi, eta, zeta),
        gradients[0],
        gradients[1],
        gradients[2],
    )


def hex8_gauss_points(gauss_order: int = 2) -> tuple[tuple[float, float, float, float], ...]:
    """Return the standard 2x2x2 full integration rule."""

    if gauss_order != 2:
        raise ValueError("Hex8 supports only the 2x2x2 full integration rule")
    a = 1.0 / np.sqrt(3.0)
    return (
        (-a, -a, -a, 1.0),
        (a, -a, -a, 1.0),
        (a, a, -a, 1.0),
        (-a, a, -a, 1.0),
        (-a, -a, a, 1.0),
        (a, -a, a, 1.0),
        (a, a, a, 1.0),
        (-a, a, a, 1.0),
    )


@dataclass(frozen=True, slots=True)
class Hex8Definition:
    """Immutable spatial definition shared by Hex8 mechanics paths."""

    gauss_order: int = 2

    canonical_type = "Hex8"
    aliases = ("C3D8",)
    node_count = 8

    def __post_init__(self) -> None:
        if isinstance(self.gauss_order, bool) or self.gauss_order != 2:
            raise ValueError("Hex8 gauss_order must be 2")

    def gauss_points(self) -> tuple[tuple[float, float, float, float], ...]:
        return hex8_gauss_points(self.gauss_order)

    @staticmethod
    def shape_functions(xi: float, eta: float, zeta: float) -> np.ndarray:
        return hex8_shape_functions(xi, eta, zeta)

    @staticmethod
    def shape_gradients(xi: float, eta: float, zeta: float) -> np.ndarray:
        return hex8_shape_gradients(xi, eta, zeta)


__all__ = [
    "Hex8Definition",
    "hex8_gauss_points",
    "hex8_shape_functions",
    "hex8_shape_gradients",
    "hex8_shape_funcs_grads",
]
