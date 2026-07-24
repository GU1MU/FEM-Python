from __future__ import annotations

from typing import Any

import numpy as np

from . import body, line, nodal, traction
from ._common import spatial_dim
from .condition import BoundaryCondition


def build_load_vector(mesh: Any, bc: BoundaryCondition) -> np.ndarray:
    """Build global load vector from boundary conditions."""
    num_dofs = int(mesh.num_dofs)
    F = np.zeros(num_dofs, dtype=float)
    nodal.add_forces(F, bc.nodal_forces, num_dofs)

    elem_lookup_cache: dict[int, Any] | None = None
    node_lookup_cache: dict[int, Any] | None = None

    def elem_lookup() -> dict[int, Any]:
        nonlocal elem_lookup_cache
        if elem_lookup_cache is None:
            elem_lookup_cache = {elem.id: elem for elem in mesh.elements}
        return elem_lookup_cache

    def node_lookup() -> dict[int, Any]:
        nonlocal node_lookup_cache
        if node_lookup_cache is None:
            node_lookup_cache = {node.id: node for node in mesh.nodes}
        return node_lookup_cache

    dim = spatial_dim(mesh)

    if bc.body_forces:
        body.add_forces(mesh, bc.body_forces, F, elem_lookup(), node_lookup(), dim)
    if bc.gravity is not None:
        body.add_gravity(mesh, bc.gravity, F, node_lookup(), dim)
    if bc.element_gravities:
        body.add_element_gravities(
            mesh,
            bc.element_gravities,
            F,
            elem_lookup(),
            node_lookup(),
            dim,
        )
    if bc.line_loads:
        line.add_forces(mesh, bc.line_loads, F, elem_lookup(), node_lookup())
    if bc.surface_tractions:
        traction.add_surface_forces(
            mesh,
            bc.surface_tractions,
            F,
            elem_lookup(),
            node_lookup(),
            dim,
        )
    if bc.edge_tractions:
        traction.add_edge_forces(
            mesh,
            bc.edge_tractions,
            F,
            elem_lookup(),
            node_lookup(),
            dim,
        )

    if not np.all(np.isfinite(F)):
        raise ValueError("assembled load vector contains non-finite values")

    return F
