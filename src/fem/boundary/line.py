from __future__ import annotations

from typing import Any, Dict

import numpy as np

from ..elements import get_element_kernel
from .condition import LineElementLoad


def add_forces(
    mesh: Any,
    loads: list[LineElementLoad],
    force: np.ndarray,
    elem_lookup: Dict[int, Any],
    node_lookup: Dict[int, Any],
) -> None:
    """Assemble resolved Beam2 line loads."""
    for load in loads:
        elem = elem_lookup.get(load.elem_id)
        if elem is None:
            raise KeyError(f"Element {load.elem_id} not found in mesh")
        if str(elem.type).casefold() != "beam2":
            raise ValueError("line loads may target only Beam2 elements")
        kernel = get_element_kernel(elem.type)
        element_force = np.asarray(
            kernel.line_load(
                mesh,
                elem,
                load.vector,
                load.coordinate_system,
                node_lookup,
            ),
            dtype=float,
        )
        dofs = list(mesh.element_dofs(elem))
        if element_force.shape != (len(dofs),):
            raise ValueError(
                f"Element {elem.id} line load vector shape {element_force.shape} "
                f"does not match {len(dofs)} element DOFs"
            )
        if not np.all(np.isfinite(element_force)):
            raise ValueError(f"Element {elem.id} line load vector contains non-finite values")
        force[dofs] += element_force
