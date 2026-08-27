from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import diags

from fem.model import DofSpace
from fem.problem import (
    Problem,
    ProblemContext,
    ProblemEvaluation,
    StatefulProblem,
)
from fem.state import EvaluationContext, SolutionState


@dataclass(slots=True)
class _ToyNonlinearProblem:
    """Small contract-only residual/tangent example."""

    @property
    def dof_space(self):
        return DofSpace.single_field("U", 2)

    def evaluate(self, context: ProblemContext) -> ProblemEvaluation:
        displacement = (
            np.zeros(2)
            if context.solution is None
            else context.solution.values
        )
        displacement = np.asarray(displacement, dtype=float)
        residual = displacement**3 - context.load_factor
        tangent = diags(3.0 * displacement**2, format="csr")
        return ProblemEvaluation(residual=residual, tangent=tangent)


@dataclass(slots=True)
class _ToyDynamicProblem:
    """Contract-only example showing optional time and dynamic quantities."""

    @property
    def dof_space(self):
        return DofSpace.single_field("U", 1)

    def evaluate(self, context: ProblemContext) -> ProblemEvaluation:
        solution = context.solution
        if solution is None:
            raise ValueError("dynamic toy problem requires a solution")
        return ProblemEvaluation(
            residual=np.zeros(1),
            mass=diags([2.0], format="csr"),
            damping=diags([0.1], format="csr"),
            outputs={
                "time": context.time,
                "velocity": solution.first_derivative,
                "acceleration": solution.second_derivative,
            },
        )


@dataclass(slots=True)
class _ToyStatefulProblem:
    """Contract-only trial/commit/rollback lifecycle example."""

    committed: float = 0.0
    trial: float = 0.0

    @property
    def dof_space(self):
        return DofSpace.single_field("U", 1)

    def evaluate(self, context: ProblemContext) -> ProblemEvaluation:
        return ProblemEvaluation(residual=np.array([self.trial]))

    def begin_increment(self, context: ProblemContext) -> None:
        displacement = 0.0
        if context.solution is not None:
            displacement = float(context.solution.values[0])
        self.trial = self.committed + displacement

    def commit(self) -> None:
        self.committed = self.trial

    def rollback(self) -> None:
        self.trial = self.committed


def test_minimal_problem_can_supply_nonlinear_residual_and_tangent() -> None:
    problem = _ToyNonlinearProblem()
    evaluation = problem.evaluate(
        ProblemContext(
            solution=SolutionState(
                DofSpace.single_field("U", 2),
                np.array([2.0, -1.0]),
            ),
            evaluation=EvaluationContext(load_factor=0.5),
        )
    )

    assert isinstance(problem, Problem)
    np.testing.assert_allclose(evaluation.residual, [7.5, -1.5])
    np.testing.assert_allclose(evaluation.tangent.diagonal(), [12.0, 3.0])


def test_dynamic_quantities_are_optional_problem_capabilities() -> None:
    velocity = np.array([1.5])
    acceleration = np.array([-0.25])
    problem = _ToyDynamicProblem()
    evaluation = problem.evaluate(
        ProblemContext(
            solution=SolutionState(
                DofSpace.single_field("U", 1),
                np.zeros(1),
                first_derivative=velocity,
                second_derivative=acceleration,
            ),
            evaluation=EvaluationContext(time=0.25),
        )
    )

    assert isinstance(problem, Problem)
    np.testing.assert_allclose(evaluation.mass.diagonal(), [2.0])
    np.testing.assert_allclose(evaluation.damping.diagonal(), [0.1])
    assert evaluation.outputs["time"] == 0.25
    np.testing.assert_allclose(evaluation.outputs["velocity"], velocity)
    np.testing.assert_allclose(evaluation.outputs["acceleration"], acceleration)


def test_stateful_problem_lifecycle_is_explicit_and_reversible() -> None:
    problem = _ToyStatefulProblem()

    assert isinstance(problem, StatefulProblem)
    problem.begin_increment(
        ProblemContext(
            solution=SolutionState(
                DofSpace.single_field("U", 1),
                np.array([2.0]),
            )
        )
    )
    assert problem.trial == 2.0
    problem.rollback()
    assert problem.trial == problem.committed == 0.0

    problem.begin_increment(
        ProblemContext(
            solution=SolutionState(
                DofSpace.single_field("U", 1),
                np.array([3.0]),
            )
        )
    )
    problem.commit()
    assert problem.trial == problem.committed == 3.0

    problem.begin_increment(
        ProblemContext(
            solution=SolutionState(
                DofSpace.single_field("U", 1),
                np.array([-1.0]),
            )
        )
    )
    problem.rollback()
    assert problem.trial == problem.committed == 3.0


def test_evaluation_outputs_can_carry_nodal_integration_and_history_fields() -> None:
    evaluation = ProblemEvaluation(
        outputs={
            "nodal": {"U": np.array([[0.0, 0.1]])},
            "integration_points": {"stress": np.array([[1.0, 2.0]])},
            "history": {"plastic_strain": np.array([[0.02]])},
        }
    )

    assert set(evaluation.outputs) == {
        "nodal",
        "integration_points",
        "history",
    }
    np.testing.assert_allclose(evaluation.outputs["nodal"]["U"], [[0.0, 0.1]])
    np.testing.assert_allclose(
        evaluation.outputs["integration_points"]["stress"],
        [[1.0, 2.0]],
    )
