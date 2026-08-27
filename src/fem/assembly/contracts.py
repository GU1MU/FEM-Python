"""Local and global contribution contracts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np
from scipy.sparse import csr_matrix, issparse

from fem.model import DofSpace
from fem.state import EvaluationContext, SolutionState
from fem.physics.contracts import LocalOutputBatch


@dataclass(frozen=True, slots=True)
class AssemblyResult:
    """Global assembled contributions and detached outputs."""

    residual: np.ndarray
    tangent: csr_matrix
    mass: csr_matrix | None = None
    damping: csr_matrix | None = None
    outputs: Mapping[str, Any] = field(default_factory=dict)
    local_outputs: tuple[LocalOutputBatch, ...] = ()

    def __post_init__(self) -> None:
        residual = np.asarray(self.residual, dtype=float)
        tangent = (
            csr_matrix(self.tangent, dtype=float)
            if issparse(self.tangent)
            else csr_matrix(np.asarray(self.tangent, dtype=float))
        )
        if residual.ndim != 1 or tangent.shape != (residual.size, residual.size):
            raise ValueError("assembled tangent must be square and match residual")
        if not np.all(np.isfinite(residual)) or not np.all(np.isfinite(tangent.data)):
            raise ValueError("assembly result must be finite")
        object.__setattr__(self, "residual", np.array(residual, copy=True))
        tangent = tangent.copy()
        tangent.sum_duplicates()
        object.__setattr__(self, "tangent", tangent)
        for name in ("mass", "damping"):
            value = getattr(self, name)
            if value is None:
                continue
            matrix = csr_matrix(value, dtype=float)
            if matrix.shape != tangent.shape or not np.all(np.isfinite(matrix.data)):
                raise ValueError(
                    f"assembled {name} must be finite and match tangent"
                )
            matrix.sum_duplicates()
            object.__setattr__(self, name, matrix)
        object.__setattr__(self, "outputs", dict(self.outputs))
        batches = tuple(self.local_outputs)
        if any(type(batch) is not LocalOutputBatch for batch in batches):
            raise TypeError("local_outputs must contain LocalOutputBatch values")
        object.__setattr__(self, "local_outputs", batches)


@runtime_checkable
class Assembly(Protocol):
    """Global domain assembly consumed by a Problem."""

    @property
    def num_dofs(self) -> int:
        """Return the global degree-of-freedom count."""

    @property
    def dof_space(self) -> DofSpace:
        """Return the compiled named-field DOF layout."""

    def assemble(
        self,
        solution: SolutionState,
        *,
        context: EvaluationContext | None = None,
    ) -> AssemblyResult:
        """Return contributions for one analysis context."""

    def begin_increment(self) -> None:
        """Start a state transaction."""

    def commit(self) -> None:
        """Commit all local states."""

    def rollback(self) -> None:
        """Rollback all local states."""


__all__ = [
    "Assembly",
    "AssemblyResult",
]
