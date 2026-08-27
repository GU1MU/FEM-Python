"""Shared reference interpolation for two-node line elements."""

from __future__ import annotations

import numpy as np


LINE2_NATURAL_NODE_COORDINATES = ((-1.0,), (1.0,))


def line2_shape_functions(xi: float) -> np.ndarray:
    """Return the two-node Lagrange interpolation at ``xi``."""

    value = float(xi)
    return np.array((0.5 * (1.0 - value), 0.5 * (1.0 + value)), dtype=float)


def line2_shape_gradients(xi: float) -> np.ndarray:
    """Return ``dN/dxi`` for a two-node line."""

    del xi
    return np.array(((-0.5, 0.5),), dtype=float)


def line2_gauss_points(
    gauss_order: int = 2,
) -> tuple[tuple[float, float], ...]:
    """Return the standard full-integration rule on ``[-1, 1]``."""

    if gauss_order != 2:
        raise ValueError("Line2 supports only the two-point full rule")
    radius = 1.0 / np.sqrt(3.0)
    return ((-radius, 1.0), (radius, 1.0))


__all__ = [
    "LINE2_NATURAL_NODE_COORDINATES",
    "line2_gauss_points",
    "line2_shape_functions",
    "line2_shape_gradients",
]
