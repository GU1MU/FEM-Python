"""Geometric definition of the two-node truss element."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .._line import (
    LINE2_NATURAL_NODE_COORDINATES,
    line2_gauss_points,
    line2_shape_functions,
    line2_shape_gradients,
)


TRUSS2_NATURAL_NODE_COORDINATES = LINE2_NATURAL_NODE_COORDINATES


def truss2_shape_functions(xi: float) -> np.ndarray:
    """Return the two-node truss interpolation at ``xi``."""

    return line2_shape_functions(xi)


def truss2_shape_gradients(xi: float) -> np.ndarray:
    """Return the parent-coordinate truss interpolation gradients."""

    return line2_shape_gradients(xi)


def truss2_gauss_points(
    gauss_order: int = 2,
) -> tuple[tuple[float, float], ...]:
    """Return the truss integration rule."""

    return line2_gauss_points(gauss_order)


@dataclass(frozen=True, slots=True)
class Truss2Definition:
    """Immutable spatial definition of the two-node truss."""

    gauss_order: int = 2

    canonical_type = "Truss2"
    aliases: tuple[str, ...] = ()
    node_count = 2

    def __post_init__(self) -> None:
        if isinstance(self.gauss_order, bool) or self.gauss_order != 2:
            raise ValueError("Truss2 gauss_order must be 2")

    def gauss_points(self) -> tuple[tuple[float, float], ...]:
        return truss2_gauss_points(self.gauss_order)

    @staticmethod
    def shape_functions(xi: float) -> np.ndarray:
        return truss2_shape_functions(xi)

    @staticmethod
    def shape_gradients(xi: float) -> np.ndarray:
        return truss2_shape_gradients(xi)


__all__ = [
    "TRUSS2_NATURAL_NODE_COORDINATES",
    "Truss2Definition",
    "truss2_gauss_points",
    "truss2_shape_functions",
    "truss2_shape_gradients",
]
