"""Ten-node quadratic tetrahedron reference element."""

from __future__ import annotations

from .definition import (
    TET10_NATURAL_NODE_COORDS,
    Tet10Definition,
    tet10_gauss_points,
    tet10_shape_functions,
    tet10_shape_gradients,
    tet10_shape_funcs_grads,
)

__all__ = [
    "Tet10Definition",
    "TET10_NATURAL_NODE_COORDS",
    "tet10_gauss_points",
    "tet10_shape_functions",
    "tet10_shape_gradients",
    "tet10_shape_funcs_grads",
]
