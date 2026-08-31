from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from fem.problem import ProblemContext, ProblemEvaluation
from fem.model import DofSpace
from fem.solver import newton
from fem.state import SolutionState


@dataclass(slots=True)
class _QuadraticProblem:
    evaluations: int = 0

    @property
    def dof_space(self):
        return DofSpace.single_field("U", 1)

    def evaluate(self, context: ProblemContext) -> ProblemEvaluation:
        self.evaluations += 1
        if context.solution is None:
            raise ValueError("quadratic problem requires a solution")
        displacement = float(context.solution.values[0])
        return ProblemEvaluation(
            residual=np.array([displacement**2 - 2.0]),
            tangent=csr_matrix([[2.0 * displacement]]),
        )


@dataclass(slots=True)
class _StatefulLinearProblem:
    target: float = 2.0
    committed: float = 0.0
    trial: float = 0.0
    begin_calls: int = 0
    commit_calls: int = 0
    rollback_calls: int = 0

    @property
    def dof_space(self):
        return DofSpace.single_field("U", 1)

    def evaluate(self, context: ProblemContext) -> ProblemEvaluation:
        if context.solution is None:
            raise ValueError("stateful problem requires a solution")
        self.trial = float(context.solution.values[0])
        return ProblemEvaluation(
            residual=np.array([self.trial - self.target]),
            tangent=csr_matrix([[1.0]]),
        )

    def begin_increment(self, context: ProblemContext) -> None:
        self.begin_calls += 1
        self.trial = self.committed

    def commit(self) -> None:
        self.commit_calls += 1
        self.committed = self.trial

    def rollback(self) -> None:
        self.rollback_calls += 1
        self.trial = self.committed


@dataclass(slots=True)
class _StatefulQuadraticProblem:
    target: float = 2.0
    committed: float = 0.5
    trial: float = 0.5
    rollback_calls: int = 0

    @property
    def dof_space(self):
        return DofSpace.single_field("U", 1)

    def evaluate(self, context: ProblemContext) -> ProblemEvaluation:
        if context.solution is None:
            raise ValueError("stateful problem requires a solution")
        self.trial = float(context.solution.values[0])
        return ProblemEvaluation(
            residual=np.array([self.trial**2 - self.target]),
            tangent=csr_matrix([[2.0 * self.trial]]),
        )

    def begin_increment(self, context: ProblemContext) -> None:
        self.trial = self.committed

    def commit(self) -> None:
        self.committed = self.trial

    def rollback(self) -> None:
        self.rollback_calls += 1
        self.trial = self.committed


@dataclass(slots=True)
class _NeverConverges:
    committed: float = 0.0
    trial: float = 0.0
    rollback_calls: int = 0

    @property
    def dof_space(self):
        return DofSpace.single_field("U", 1)

    def evaluate(self, context: ProblemContext) -> ProblemEvaluation:
        if context.solution is None:
            raise ValueError("stateful problem requires a solution")
        self.trial = float(context.solution.values[0])
        return ProblemEvaluation(
            residual=np.array([1.0]),
            tangent=csr_matrix([[1.0]]),
        )

    def begin_increment(self, context: ProblemContext) -> None:
        self.trial = self.committed

    def commit(self) -> None:
        self.committed = self.trial

    def rollback(self) -> None:
        self.rollback_calls += 1
        self.trial = self.committed


def test_newton_solver_converges_on_minimal_nonlinear_problem() -> None:
    problem = _QuadraticProblem()

    result = newton.solve(
        problem,
        SolutionState(problem.dof_space, np.array([1.0])),
        residual_tolerance=1.0e-12,
        max_iterations=10,
    )

    assert result.converged is True
    assert result.iterations < 10
    assert result.residual_norm <= 1.0e-12
    assert result.solution.values[0] == pytest.approx(np.sqrt(2.0))
    assert len(result.residual_history) == result.iterations + 1
    assert problem.evaluations == result.iterations + 1


def test_newton_solver_commits_state_only_after_convergence() -> None:
    problem = _StatefulLinearProblem()

    result = newton.solve(
        problem,
        SolutionState(problem.dof_space, np.array([0.0])),
        residual_tolerance=1.0e-12,
        max_iterations=3,
    )

    assert result.solution.values[0] == pytest.approx(2.0)
    assert problem.begin_calls == 1
    assert problem.commit_calls == 1
    assert problem.rollback_calls == 0
    assert problem.committed == problem.trial == pytest.approx(2.0)


def test_stateful_newton_backtracks_a_valid_trial_that_increases_residual() -> None:
    problem = _StatefulQuadraticProblem()

    result = newton.solve(
        problem,
        SolutionState(problem.dof_space, np.array([0.5])),
        residual_tolerance=1.0e-12,
        max_iterations=10,
    )

    assert result.converged is True
    assert result.solution.values[0] == pytest.approx(np.sqrt(2.0))
    assert result.trial_scale_history[0] == pytest.approx(0.5)
    assert all(0.0 < scale <= 1.0 for scale in result.trial_scale_history)
    assert problem.rollback_calls >= 1
    assert problem.committed == problem.trial == pytest.approx(np.sqrt(2.0))


def test_newton_solver_rolls_back_state_when_iteration_limit_is_reached() -> None:
    problem = _NeverConverges()

    with pytest.raises(newton.NewtonConvergenceError) as error:
        newton.solve(
            problem,
            SolutionState(problem.dof_space, np.array([0.0])),
            residual_tolerance=1.0e-12,
            max_iterations=2,
        )

    assert error.value.result.iterations == 2
    assert problem.rollback_calls == 1
    assert problem.trial == problem.committed == 0.0


def test_newton_solver_cancellation_rolls_back_state_at_safe_boundary() -> None:
    problem = _StatefulLinearProblem()
    checks = 0

    def should_cancel() -> bool:
        nonlocal checks
        checks += 1
        # The first two checks validate controls/initial state. The third one
        # is reached inside the started increment, where rollback is required.
        return checks >= 3

    with pytest.raises(newton.SolveCancelled):
        newton.solve(
            problem,
            SolutionState(problem.dof_space, np.array([0.0])),
            residual_tolerance=1.0e-12,
            max_iterations=3,
            should_cancel=should_cancel,
        )

    assert problem.rollback_calls == 1
    assert problem.trial == problem.committed == pytest.approx(0.0)


def test_newton_solver_rejects_invalid_controls() -> None:
    problem = _QuadraticProblem()

    with pytest.raises(ValueError, match="max_iterations"):
        newton.solve(problem, max_iterations=0)
    with pytest.raises(ValueError, match="residual_tolerance"):
        newton.solve(problem, residual_tolerance=0.0)
