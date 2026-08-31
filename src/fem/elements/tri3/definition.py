"""Geometric definition of the linear three-node triangle."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def tri3_shape_functions(xi: float, eta: float) -> np.ndarray:
    """Return the barycentric shape functions on the reference triangle."""

    return np.array([1.0 - xi - eta, xi, eta], dtype=float)


def tri3_shape_gradients(xi: float, eta: float) -> np.ndarray:
    """Return the constant ``dN/d(xi, eta)`` matrix."""

    del xi, eta
    return np.array(
        [
            [-1.0, 1.0, 0.0],
            [-1.0, 0.0, 1.0],
        ],
        dtype=float,
    )


def tri3_gauss_points(gauss_order: int = 1) -> tuple[tuple[float, float, float], ...]:
    """Return the one-point full rule for the linear triangle."""

    if gauss_order != 1:
        raise ValueError("Tri3 supports only the one-point full integration rule")
    return ((1.0 / 3.0, 1.0 / 3.0, 0.5),)


@dataclass(frozen=True, slots=True)
class Tri3Definition:
    """Immutable spatial definition shared by linear and nonlinear Tri3 paths."""

    gauss_order: int = 1

    canonical_type = "Tri3"
    aliases = ("CPS3", "CPE3")
    node_count = 3

    def __post_init__(self) -> None:
        if isinstance(self.gauss_order, bool) or self.gauss_order != 1:
            raise ValueError("Tri3 gauss_order must be 1")

    def gauss_points(self) -> tuple[tuple[float, float, float], ...]:
        return tri3_gauss_points(self.gauss_order)

    @staticmethod
    def shape_functions(xi: float, eta: float) -> np.ndarray:
        return tri3_shape_functions(xi, eta)

    @staticmethod
    def shape_gradients(xi: float, eta: float) -> np.ndarray:
        return tri3_shape_gradients(xi, eta)


__all__ = [
    "Tri3Definition",
    "tri3_gauss_points",
    "tri3_shape_functions",
    "tri3_shape_gradients",
]
