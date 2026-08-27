"""Six-node quadratic triangle reference element."""

from __future__ import annotations

from .definition import (
    TRI6_NATURAL_NODE_COORDS,
    Tri6Definition,
    tri6_gauss_points,
    tri6_shape_functions,
    tri6_shape_gradients,
    tri6_shape_funcs_grads,
)

__all__ = [
    "TRI6_NATURAL_NODE_COORDS",
    "Tri6Definition",
    "tri6_gauss_points",
    "tri6_shape_functions",
    "tri6_shape_gradients",
    "tri6_shape_funcs_grads",
]
