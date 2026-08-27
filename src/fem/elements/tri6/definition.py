"""Geometric definition of the six-node quadratic triangle."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


TRI6_NATURAL_NODE_COORDS = np.array(
    [
        (0.0, 0.0),
        (1.0, 0.0),
        (0.0, 1.0),
        (0.5, 0.0),
        (0.5, 0.5),
        (0.0, 0.5),
    ],
    dtype=float,
)


def tri6_shape_functions(xi: float, eta: float) -> np.ndarray:
    """Return Tri6 barycentric shape functions in vertex-edge order."""

    l1 = 1.0 - xi - eta
    l2 = xi
    l3 = eta
    return np.array(
        [
            l1 * (2.0 * l1 - 1.0),
            l2 * (2.0 * l2 - 1.0),
            l3 * (2.0 * l3 - 1.0),
            4.0 * l1 * l2,
            4.0 * l2 * l3,
            4.0 * l3 * l1,
        ],
        dtype=float,
    )


def tri6_shape_gradients(xi: float, eta: float) -> np.ndarray:
    """Return ``dN/d(xi, eta)`` in vertex-edge order."""

    l1 = 1.0 - xi - eta
    l2 = xi
    l3 = eta
    return np.array(
        [
            [1.0 - 4.0 * l1, 4.0 * l2 - 1.0, 0.0,
             4.0 * (l1 - l2), 4.0 * l3, -4.0 * l3],
            [1.0 - 4.0 * l1, 0.0, 4.0 * l3 - 1.0,
             -4.0 * l2, 4.0 * l2, 4.0 * (l1 - l3)],
        ],
        dtype=float,
    )


def tri6_shape_funcs_grads(
    xi: float,
    eta: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``N`` and the two parent-coordinate gradient rows."""

    gradients = tri6_shape_gradients(xi, eta)
    return tri6_shape_functions(xi, eta), gradients[0], gradients[1]


def tri6_gauss_points() -> tuple[tuple[float, float, float], ...]:
    """Return the three-point full rule on the reference triangle."""

    return (
        (1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0),
        (2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0),
        (1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0),
    )


@dataclass(frozen=True, slots=True)
class Tri6Definition:
    """Immutable spatial definition shared by Tri6 mechanics paths."""

    gauss_order: int = 3

    canonical_type = "Tri6"
    aliases = ("CPS6", "CPE6")
    node_count = 6

    def __post_init__(self) -> None:
        if isinstance(self.gauss_order, bool) or self.gauss_order != 3:
            raise ValueError("Tri6 gauss_order must be 3")

    def gauss_points(self) -> tuple[tuple[float, float, float], ...]:
        return tri6_gauss_points()

    @staticmethod
    def shape_functions(xi: float, eta: float) -> np.ndarray:
        return tri6_shape_functions(xi, eta)

    @staticmethod
    def shape_gradients(xi: float, eta: float) -> np.ndarray:
        return tri6_shape_gradients(xi, eta)


__all__ = [
    "Tri6Definition",
    "TRI6_NATURAL_NODE_COORDS",
    "tri6_gauss_points",
    "tri6_shape_functions",
    "tri6_shape_gradients",
    "tri6_shape_funcs_grads",
]
