from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from fem.problem import ProblemContext, ProblemEvaluation
from fem.analysis import incremental
from fem.solver import newton
from fem.model import DofSpace
from fem.state import EvaluationContext, SolutionState


@dataclass(slots=True)
class _RampProblem:
    """Small stateful problem whose converged displacement equals load factor."""

    num_dofs: int = 1
    committed: float = 0.0
    trial: float = 0.0

    @property
    def dof_space(self):
        return DofSpace.single_field("U", self.num_dofs)

    def evaluate(self, context: ProblemContext) -> ProblemEvaluation:
        solution = context.solution
        if solution is None:
            raise ValueError("ramp problem requires a solution")
        self.trial = float(solution.values[0])
        return ProblemEvaluation(
            residual=np.array([self.trial - context.load_factor]),
            tangent=csr_matrix([[1.0]]),
            outputs={"trial": np.array([self.trial])},
        )

    def begin_increment(self, context: ProblemContext) -> None:
        self.trial = self.committed

    def commit(self) -> None:
        self.committed = self.trial

    def rollback(self) -> None:
        self.trial = self.committed


@dataclass(slots=True)
class _FailAtFullLoadProblem(_RampProblem):
    def evaluate(self, context: ProblemContext) -> ProblemEvaluation:
        solution = context.solution
        if solution is None:
            raise ValueError("ramp problem requires a solution")
        self.trial = float(solution.values[0])
        residual = 1.0 if context.load_factor >= 1.0 else self.trial - context.load_factor
        return ProblemEvaluation(
            residual=np.array([residual]),
            tangent=csr_matrix([[1.0]]),
        )


@dataclass(slots=True)
class _IncrementLimitProblem(_RampProblem):
    """Reject an indivisible increment so adaptive cutback can be tested."""

    maximum_increment: float = 0.15

    def evaluate(self, context: ProblemContext) -> ProblemEvaluation:
        solution = context.solution
        if solution is None:
            raise ValueError("ramp problem requires a solution")
        self.trial = float(solution.values[0])
        if abs(float(context.load_factor) - self.committed) > self.maximum_increment:
            residual = 1.0
        else:
            residual = self.trial - context.load_factor
        return ProblemEvaluation(
            residual=np.array([residual]),
            tangent=csr_matrix([[1.0]]),
        )


@dataclass(slots=True)
class _ConstitutiveFailureProblem(_RampProblem):
    """Expose a local constitutive failure that is safe to retry by cutback."""

    maximum_increment: float = 0.15

    def evaluate(self, context: ProblemContext) -> ProblemEvaluation:
        solution = context.solution
        if solution is None:
            raise ValueError("ramp problem requires a solution")
        self.trial = float(solution.values[0])
        if abs(float(context.load_factor) - self.committed) > self.maximum_increment:
            raise RuntimeError("constitutive return mapping failed: local step too large")
        return ProblemEvaluation(
            residual=np.array([self.trial - context.load_factor]),
            tangent=csr_matrix([[1.0]]),
        )


@dataclass(slots=True)
class _CancelAfterFirstCommitProblem(_RampProblem):
    cancel_requested: bool = False

    def commit(self) -> None:
        _RampProblem.commit(self)
        if self.committed >= 0.25:
            self.cancel_requested = True


def test_incremental_solver_commits_each_converged_load_factor():
    problem = _RampProblem()

    result = incremental.solve(
        problem,
        [0.25, 0.5, 1.0],
        residual_tolerance=1.0e-12,
    )

    assert [item.load_factor for item in result.increments] == [0.25, 0.5, 1.0]
    np.testing.assert_allclose(
        [item.outputs["trial"][0] for item in result.increments],
        [0.25, 0.5, 1.0],
    )
    assert not result.increments[0].outputs["trial"].flags.writeable
    np.testing.assert_allclose(result.final_solution.values, [1.0])
    assert problem.committed == pytest.approx(1.0)
    assert problem.trial == pytest.approx(1.0)


def test_incremental_solver_propagates_cancellation_between_increments():
    problem = _CancelAfterFirstCommitProblem()

    with pytest.raises(newton.SolveCancelled):
        incremental.solve(
            problem,
            [0.25, 0.5],
            residual_tolerance=1.0e-12,
            should_cancel=lambda: problem.cancel_requested,
        )

    assert problem.committed == pytest.approx(0.25)
    assert problem.trial == pytest.approx(0.25)


def test_incremental_solver_reports_failed_factor_and_preserves_last_commit():
    problem = _FailAtFullLoadProblem()

    with pytest.raises(incremental.IncrementalConvergenceError) as caught:
        incremental.solve(problem, [0.5, 1.0], max_iterations=2)

    error = caught.value
    assert error.failed_load_factor == pytest.approx(1.0)
    assert [item.load_factor for item in error.completed.increments] == [0.5]
    np.testing.assert_allclose(error.completed.final_solution.values, [0.5])
    assert problem.committed == pytest.approx(0.5)
    assert problem.trial == pytest.approx(0.5)


def test_incremental_solver_can_retry_a_failed_target_with_bounded_cutback():
    problem = _IncrementLimitProblem()

    result = incremental.solve(
        problem,
        [0.2, 0.4],
        max_iterations=2,
        residual_tolerance=1.0e-12,
        adaptive=True,
        maximum_increments=10,
    )

    assert [item.load_factor for item in result.increments] == pytest.approx(
        [0.1, 0.2, 0.3, 0.4]
    )
    assert problem.committed == pytest.approx(0.4)


def test_incremental_solver_rejects_a_cutback_budget_smaller_than_targets():
    with pytest.raises(ValueError, match="maximum_increments"):
        incremental.solve(
            _RampProblem(),
            [0.5, 1.0],
            adaptive=True,
            maximum_increments=1,
        )


def test_incremental_solver_explains_when_minimum_increment_blocks_cutback():
    problem = _IncrementLimitProblem()

    with pytest.raises(incremental.IncrementalConvergenceError) as caught:
        incremental.solve(
            problem,
            [0.2],
            max_iterations=2,
            residual_tolerance=1.0e-12,
            adaptive=True,
            maximum_increments=10,
            minimum_increment=0.11,
        )

    message = str(caught.value)
    assert "无法自动切步" in message
    assert "最小增量" in message


def test_incremental_solver_cutbacks_a_constitutive_return_mapping_failure():
    result = incremental.solve(
        _ConstitutiveFailureProblem(),
        [0.2],
        max_iterations=4,
        residual_tolerance=1.0e-12,
        adaptive=True,
        maximum_increments=10,
    )

    assert [item.load_factor for item in result.increments] == pytest.approx(
        [0.1, 0.2]
    )


@pytest.mark.parametrize("factors", [[], [0.5, 0.5], [np.nan]])
def test_incremental_solver_validates_load_factor_sequence(factors):
    with pytest.raises(ValueError):
        incremental.solve(_RampProblem(), factors)


def test_incremental_solver_uses_context_displacement_when_provided():
    problem = _RampProblem()
    context = ProblemContext(
        solution=SolutionState(
            DofSpace.single_field("U", 1),
            np.array([-0.25]),
        ),
        evaluation=EvaluationContext(),
    )

    result = incremental.solve(problem, [0.5], context=context)

    np.testing.assert_allclose(result.final_solution.values, [0.5])


def test_incremental_solver_supports_unloading_and_load_reversal():
    problem = _RampProblem()

    result = incremental.solve(
        problem,
        [0.5, 1.0, 0.5, 0.0, -0.5],
        residual_tolerance=1.0e-12,
    )

    assert [item.load_factor for item in result.increments] == [
        0.5,
        1.0,
        0.5,
        0.0,
        -0.5,
    ]
    np.testing.assert_allclose(result.final_solution.values, [-0.5])
    assert problem.committed == pytest.approx(-0.5)
