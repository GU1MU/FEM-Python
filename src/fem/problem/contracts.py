"""Small, capability-oriented contracts between analysis and solvers.

The contracts in this module deliberately do not assemble matrices, solve
systems, or know concrete materials and elements.  They describe the
mathematical quantities a solver may request from a concrete problem.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable

import numpy as np
from scipy.sparse import spmatrix

from fem.model import DofSpace
from fem.state import EvaluationContext, SolutionState
from fem.physics.contracts import LocalOutputBatch


@dataclass(frozen=True, slots=True)
class ProblemContext:
    """Global solution and procedure coordinates for one equation evaluation."""

    solution: SolutionState | None = None
    evaluation: EvaluationContext = field(default_factory=EvaluationContext)

    def __post_init__(self) -> None:
        if self.solution is not None and type(self.solution) is not SolutionState:
            raise TypeError("solution must be exactly SolutionState or None")
        if type(self.evaluation) is not EvaluationContext:
            raise TypeError("evaluation must be exactly EvaluationContext")

    @property
    def time(self) -> float:
        return self.evaluation.time

    @property
    def load_factor(self) -> float:
        return self.evaluation.load_factor

    @property
    def parameters(self) -> Mapping[str, Any]:
        return self.evaluation.parameters


@dataclass(frozen=True, slots=True)
class ProblemEvaluation:
    """Mathematical contributions returned by one problem evaluation.

    The residual convention is ``R = F_int - F_ext``.  A Newton solver using
    this evaluation therefore solves ``K_t @ du = -R``.  ``mass``, ``damping``,
    ``constraints`` and ``outputs`` are optional capabilities; a linear static
    problem may leave them unset.
    """

    residual: np.ndarray | None = None
    tangent: spmatrix | None = None
    mass: spmatrix | None = None
    damping: spmatrix | None = None
    constraints: Any | None = None
    outputs: Mapping[str, Any] = field(default_factory=dict)
    local_outputs: tuple[LocalOutputBatch, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "outputs", MappingProxyType(dict(self.outputs)))
        local_outputs = tuple(self.local_outputs)
        if any(type(batch) is not LocalOutputBatch for batch in local_outputs):
            raise TypeError(
                "local_outputs must contain only LocalOutputBatch values"
            )
        object.__setattr__(self, "local_outputs", local_outputs)


@runtime_checkable
class Problem(Protocol):
    """Base contract consumed by an analysis or solver implementation."""

    @property
    def dof_space(self) -> DofSpace:
        """Return the global named-field layout of the equation system."""

    def evaluate(self, context: ProblemContext) -> ProblemEvaluation:
        """Return mathematical contributions for the supplied context."""


@runtime_checkable
class StatefulProblem(Problem, Protocol):
    """Optional lifecycle contract for history-dependent problems."""

    def begin_increment(self, context: ProblemContext) -> None:
        """Start a trial increment without committing material history."""

    def commit(self) -> None:
        """Commit the last converged trial state."""

    def rollback(self) -> None:
        """Discard the current trial state and restore committed state."""


__all__ = [
    "Problem",
    "ProblemContext",
    "ProblemEvaluation",
    "StatefulProblem",
]
