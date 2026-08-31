"""Geometric definition of the two-node beam element."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .._line import (
    LINE2_NATURAL_NODE_COORDINATES,
    line2_gauss_points,
    line2_shape_functions,
    line2_shape_gradients,
)


BEAM2_NATURAL_NODE_COORDINATES = LINE2_NATURAL_NODE_COORDINATES


def beam2_shape_functions(xi: float) -> np.ndarray:
    """Return the two-node beam interpolation at ``xi``."""

    return line2_shape_functions(xi)


def beam2_shape_gradients(xi: float) -> np.ndarray:
    """Return the parent-coordinate beam interpolation gradients."""

    return line2_shape_gradients(xi)


def beam2_gauss_points(
    gauss_order: int = 2,
) -> tuple[tuple[float, float], ...]:
    """Return the beam integration rule."""

    return line2_gauss_points(gauss_order)


@dataclass(frozen=True, slots=True)
class Beam2Definition:
    """Immutable spatial definition of the two-node beam."""

    gauss_order: int = 2

    canonical_type = "Beam2"
    aliases: tuple[str, ...] = ()
    node_count = 2

    def __post_init__(self) -> None:
        if isinstance(self.gauss_order, bool) or self.gauss_order != 2:
            raise ValueError("Beam2 gauss_order must be 2")

    def gauss_points(self) -> tuple[tuple[float, float], ...]:
        return beam2_gauss_points(self.gauss_order)

    @staticmethod
    def shape_functions(xi: float) -> np.ndarray:
        return beam2_shape_functions(xi)

    @staticmethod
    def shape_gradients(xi: float) -> np.ndarray:
        return beam2_shape_gradients(xi)


__all__ = [
    "BEAM2_NATURAL_NODE_COORDINATES",
    "Beam2Definition",
    "beam2_gauss_points",
    "beam2_shape_functions",
    "beam2_shape_gradients",
]
