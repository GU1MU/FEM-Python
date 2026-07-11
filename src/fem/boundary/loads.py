from __future__ import annotations

from typing import Any

import numpy as np

from . import body, nodal, traction
from ._common import spatial_dim
from .condition import BoundaryCondition


def build_load_vector(mesh: Any, bc: BoundaryCondition) -> np.ndarray:
    """Build global load vector from boundary conditions."""
    num_dofs = int(mesh.num_dofs)
    F = np.zeros(num_dofs, dtype=float)
    nodal.add_forces(F, bc.nodal_forces, num_dofs)

    elem_lookup = {elem.id: elem for elem in mesh.elements}
    node_lookup = {node.id: node for node in mesh.nodes}
    dim = spatial_dim(mesh)

    body.add_forces(mesh, bc.body_forces, F, elem_lookup, node_lookup, dim)
    body.add_gravity(mesh, bc.gravity, F, node_lookup, dim)
    traction.add_surface_forces(mesh, bc.surface_tractions, F, elem_lookup, node_lookup, dim)
    traction.add_edge_forces(mesh, bc.edge_tractions, F, elem_lookup, node_lookup, dim)

    if not np.all(np.isfinite(F)):
        raise ValueError("assembled load vector contains non-finite values")

    return F
