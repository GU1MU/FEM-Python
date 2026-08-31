from __future__ import annotations

from typing import Dict

import numpy as np


def add_forces(F: np.ndarray, nodal_forces: Dict[int, float], num_dofs: int) -> None:
    """Assemble nodal forces into F."""
    for dof_id, value in nodal_forces.items():
        if dof_id < 0 or dof_id >= num_dofs:
            raise IndexError(f"DOF index {dof_id} out of bounds [0, {num_dofs})")
        value = float(value)
        if not np.isfinite(value):
            raise ValueError(f"nodal force at DOF {dof_id} must be finite, got {value!r}")
        F[dof_id] += value
