"""Geometric definition of the ten-node quadratic tetrahedron."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


_TET10_BARYCENTRIC_GRADIENTS = np.array(
    [
        (-1.0, -1.0, -1.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    ],
    dtype=float,
)

TET10_NATURAL_NODE_COORDS = (
    (0.0, 0.0, 0.0),
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
    (0.5, 0.0, 0.0),
    (0.5, 0.5, 0.0),
    (0.0, 0.5, 0.0),
    (0.0, 0.0, 0.5),
    (0.5, 0.0, 0.5),
    (0.0, 0.5, 0.5),
)


def tet10_shape_functions(xi: float, eta: float, zeta: float) -> np.ndarray:
    """Return Tet10 barycentric shape functions in vertex-edge order."""

    barycentric = np.array(
        [1.0 - xi - eta - zeta, xi, eta, zeta],
        dtype=float,
    )
    l1, l2, l3, l4 = barycentric
    return np.array(
        [
            l1 * (2.0 * l1 - 1.0),
            l2 * (2.0 * l2 - 1.0),
            l3 * (2.0 * l3 - 1.0),
            l4 * (2.0 * l4 - 1.0),
            4.0 * l1 * l2,
            4.0 * l2 * l3,
            4.0 * l1 * l3,
            4.0 * l1 * l4,
            4.0 * l2 * l4,
            4.0 * l3 * l4,
        ],
        dtype=float,
    )


def tet10_shape_gradients(
    xi: float,
    eta: float,
    zeta: float,
) -> np.ndarray:
    """Return ``dN/d(xi, eta, zeta)`` in vertex-edge order."""

    barycentric = np.array(
        [1.0 - xi - eta - zeta, xi, eta, zeta],
        dtype=float,
    )
    gradients = np.empty((3, 10), dtype=float)
    for index in range(4):
        gradients[:, index] = (
            (4.0 * barycentric[index] - 1.0)
            * _TET10_BARYCENTRIC_GRADIENTS[index]
        )

    edge_pairs = (
        (0, 1),
        (1, 2),
        (0, 2),
        (0, 3),
        (1, 3),
        (2, 3),
    )
    for node_index, (first, second) in enumerate(edge_pairs, start=4):
        gradients[:, node_index] = 4.0 * (
            barycentric[second] * _TET10_BARYCENTRIC_GRADIENTS[first]
            + barycentric[first] * _TET10_BARYCENTRIC_GRADIENTS[second]
        )
    return gradients


def tet10_shape_funcs_grads(
    xi: float,
    eta: float,
    zeta: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return ``N`` and the three parent-coordinate gradient rows."""

    gradients = tet10_shape_gradients(xi, eta, zeta)
    return (
        tet10_shape_functions(xi, eta, zeta),
        gradients[0],
        gradients[1],
        gradients[2],
    )


def tet10_gauss_points() -> tuple[tuple[float, float, float, float], ...]:
    """Return the four-point Hammer rule on the reference tetrahedron."""

    n = 0.58541020
    a = (1.0 - n) / 4.0
    b = (1.0 + 3.0 * n) / 4.0
    weight = 1.0 / 24.0
    return (
        (a, a, a, weight),
        (b, a, a, weight),
        (a, b, a, weight),
        (a, a, b, weight),
    )


@dataclass(frozen=True, slots=True)
class Tet10Definition:
    """Immutable spatial definition shared by Tet10 mechanics paths."""

    gauss_order: int = 4

    canonical_type = "Tet10"
    aliases = ("C3D10",)
    node_count = 10

    def __post_init__(self) -> None:
        if isinstance(self.gauss_order, bool) or self.gauss_order != 4:
            raise ValueError("Tet10 gauss_order must be 4")

    def gauss_points(self) -> tuple[tuple[float, float, float, float], ...]:
        return tet10_gauss_points()

    @staticmethod
    def shape_functions(xi: float, eta: float, zeta: float) -> np.ndarray:
        return tet10_shape_functions(xi, eta, zeta)

    @staticmethod
    def shape_gradients(xi: float, eta: float, zeta: float) -> np.ndarray:
        return tet10_shape_gradients(xi, eta, zeta)


__all__ = [
    "Tet10Definition",
    "TET10_NATURAL_NODE_COORDS",
    "tet10_gauss_points",
    "tet10_shape_functions",
    "tet10_shape_gradients",
    "tet10_shape_funcs_grads",
]
