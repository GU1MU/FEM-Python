from __future__ import annotations

from typing import Any, Dict

import numpy as np

from ._common import add_kernel_load, require_element, validate_vector
from .condition import EdgeTraction, SurfaceTraction


def add_surface_forces(
    mesh: Any,
    tractions: list[SurfaceTraction],
    F: np.ndarray,
    elem_lookup: Dict[int, Any],
    node_lookup: Dict[int, Any],
    spatial_dim: int,
) -> None:
    """Assemble element face tractions."""
    if not tractions:
        return
    if spatial_dim == 2:
        raise ValueError("2D surface tractions are not supported; use edge tractions")
    if spatial_dim != 3:
        raise ValueError(f"unsupported mesh spatial dimension: {spatial_dim}")
    for traction in tractions:
        elem = require_element(elem_lookup, traction.elem_id)
        validate_vector(traction.vector, spatial_dim, "surface traction")
        add_kernel_load(
            mesh,
            elem,
            node_lookup,
            F,
            "face_traction",
            traction.vector,
            local_index=traction.local_index,
        )


def add_edge_forces(
    mesh: Any,
    tractions: list[EdgeTraction],
    F: np.ndarray,
    elem_lookup: Dict[int, Any],
    node_lookup: Dict[int, Any],
    spatial_dim: int,
) -> None:
    """Assemble element edge tractions."""
    if not tractions:
        return
    if spatial_dim == 3:
        raise NotImplementedError("3D edge loads are not supported")
    if spatial_dim != 2:
        raise ValueError(f"unsupported mesh spatial dimension: {spatial_dim}")
    for traction in tractions:
        elem = require_element(elem_lookup, traction.elem_id)
        validate_vector(traction.vector, spatial_dim, "edge traction")
        add_kernel_load(
            mesh,
            elem,
            node_lookup,
            F,
            "edge_traction",
            traction.vector,
            local_index=traction.local_index,
        )
