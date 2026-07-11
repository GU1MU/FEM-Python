from __future__ import annotations

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


@dataclass
class ModelResults:
    """Collection of solved model step results."""
    model: Any
    results: tuple[ModelResult, ...]


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
