"""Three-node triangular reference element."""

from __future__ import annotations

from .definition import (
    Tri3Definition,
    tri3_gauss_points,
    tri3_shape_functions,
    tri3_shape_gradients,
)

__all__ = [
    "Tri3Definition",
    "tri3_gauss_points",
    "tri3_shape_functions",
    "tri3_shape_gradients",
]
