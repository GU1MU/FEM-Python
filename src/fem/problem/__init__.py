"""Global mathematical equations consumed by numerical solvers."""

from .contracts import (
    Problem,
    ProblemContext,
    ProblemEvaluation,
    StatefulProblem,
)
from .static import StaticEquilibriumProblem
from .dynamic import (
    ExplicitDynamicProblem,
    LinearDynamicProblem,
    NonlinearDynamicProblem,
)

__all__ = [
    "Problem",
    "ProblemContext",
    "ProblemEvaluation",
    "StatefulProblem",
    "StaticEquilibriumProblem",
    "ExplicitDynamicProblem",
    "LinearDynamicProblem",
    "NonlinearDynamicProblem",
]
