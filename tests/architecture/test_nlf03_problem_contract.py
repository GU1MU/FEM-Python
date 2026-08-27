from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix

from fem.model import DofSpace
from fem.state import EvaluationContext
from fem.problem import (
    Problem,
    ProblemContext,
    ProblemEvaluation,
    StatefulProblem,
)


class _LinearProbeProblem:
    @property
    def dof_space(self):
        return DofSpace.single_field("U", 2)

    def evaluate(self, context: ProblemContext) -> ProblemEvaluation:
        assert context.load_factor == 1.0
        return ProblemEvaluation(
            residual=np.array([-2.0, -4.0]),
            tangent=csr_matrix(np.diag([2.0, 2.0])),
        )


class _StatefulProbeProblem(_LinearProbeProblem):
    def begin_increment(self, context: ProblemContext) -> None:
        self.context = context

    def commit(self) -> None:
        self.committed = True

    def rollback(self) -> None:
        self.rolled_back = True


def test_nlf03_base_problem_contract_expresses_linearized_math() -> None:
    problem = _LinearProbeProblem()
    context = ProblemContext(
        evaluation=EvaluationContext(load_factor=1.0),
    )

    assert isinstance(problem, Problem)
    evaluation = problem.evaluate(context)

    assert evaluation.residual is not None
    assert evaluation.tangent is not None
    np.testing.assert_allclose(evaluation.residual, [-2.0, -4.0])
    np.testing.assert_allclose(
        evaluation.tangent.toarray(),
        [[2.0, 0.0], [0.0, 2.0]],
    )
    assert evaluation.mass is None
    assert evaluation.damping is None
    assert evaluation.constraints is None


def test_nlf03_optional_outputs_are_isolated_from_source_mapping() -> None:
    source = {"stress": np.array([1.0, 2.0])}
    evaluation = ProblemEvaluation(outputs=source)

    source["strain"] = np.array([3.0, 4.0])

    assert tuple(evaluation.outputs) == ("stress",)


def test_nlf03_stateful_lifecycle_is_optional_and_explicit() -> None:
    problem = _StatefulProbeProblem()
    context = ProblemContext(
        evaluation=EvaluationContext(time=0.5, load_factor=0.25),
    )

    assert isinstance(problem, Problem)
    assert isinstance(problem, StatefulProblem)

    problem.begin_increment(context)
    problem.commit()
    problem.rollback()

    assert problem.context is context
    assert problem.committed is True
    assert problem.rolled_back is True
