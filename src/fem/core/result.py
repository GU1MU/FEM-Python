from __future__ import annotations

import operator
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class ModelResult:
    """Result data for one solved model step."""
    model: Any
    step: Any
    U: np.ndarray
    reactions: np.ndarray
    name: str | None = None

    def __post_init__(self) -> None:
        num_dofs = int(self.model.mesh.num_dofs)
        self.U = _result_vector("U", self.U, num_dofs)
        self.reactions = _result_vector("reactions", self.reactions, num_dofs)

    def nodal_displacement(self, node_id: int, component: int) -> float:
        """Return one nodal displacement component using 1-based numbering."""
        dof = _nodal_dof(self.model.mesh, node_id, component)
        return float(self.U[dof])

    def nodal_reaction(self, node_id: int, component: int) -> float:
        """Return one nodal reaction component using 1-based numbering."""
        dof = _nodal_dof(self.model.mesh, node_id, component)
        return float(self.reactions[dof])


@dataclass
class ModelResults:
    """Collection of solved model step results."""
    model: Any
    results: tuple[ModelResult, ...]

    def __iter__(self) -> Iterator[ModelResult]:
        return iter(self.results)

    def __len__(self) -> int:
        return len(self.results)

    def __getitem__(
        self,
        index: int | slice,
    ) -> ModelResult | tuple[ModelResult, ...]:
        return self.results[index]


def _nodal_dof(mesh: Any, node_id: int, component: int) -> int:
    if isinstance(component, bool):
        raise TypeError("component must be an integer")
    try:
        component_number = operator.index(component)
    except TypeError as exc:
        raise TypeError("component must be an integer") from exc
    if component_number < 1 or component_number > mesh.dofs_per_node:
        raise IndexError(
            f"component {component_number} out of range for "
            f"{mesh.dofs_per_node} DOFs per node; components are 1-based"
        )
    return mesh.global_dof(node_id, component_number - 1)


def _result_vector(name: str, values: Any, num_dofs: int) -> np.ndarray:
    """Return an owned, finite one-dimensional result vector."""
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {array.shape}")
    if array.shape[0] != num_dofs:
        raise ValueError(
            f"{name} must have length {num_dofs}, got {array.shape[0]}"
        )
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array.copy()
