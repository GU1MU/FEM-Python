from __future__ import annotations

import numpy as np

from fem.application import solve_analysis
from fem.problem import StaticEquilibriumProblem
from fem.analysis import linear_static as static_linear
from fem.solver import newton
from fem.state import SolutionState
from tests.architecture.test_nlf04_linear_static_problem_adapter import (
    _make_model,
    _problem,
)


def test_static_problem_exposes_newton_solvable_constraint_equations() -> None:
    problem = _problem(_make_model())

    assert isinstance(problem, StaticEquilibriumProblem)
    evaluation = problem.evaluate()
    constrained_dofs = tuple(evaluation.constraints.prescribed_values)
    assert constrained_dofs
    for dof in constrained_dofs:
        assert evaluation.tangent[dof, dof] == 1.0
        assert evaluation.residual[dof] == 0.0


def test_generic_newton_and_specialized_linear_path_agree() -> None:
    model = _make_model()
    problem = _problem(model)

    generic = newton.solve(
        problem,
        SolutionState.zeros(problem.dof_space),
        residual_tolerance=1.0e-12,
        max_iterations=3,
    )
    specialized = static_linear.solve(model, "pull")

    assert generic.iterations == 1
    np.testing.assert_allclose(generic.solution.values, specialized.U)
    np.testing.assert_allclose(
        problem.reactions(generic.solution),
        specialized.reactions,
        atol=1.0e-12,
    )


def test_result_publication_stays_above_problem_and_solver() -> None:
    model = _make_model()
    result = solve_analysis(model, "pull")
    specialized = static_linear.solve(model, "pull")

    assert not hasattr(_problem(model), "solve")
    np.testing.assert_allclose(result.U, specialized.U)
    np.testing.assert_allclose(result.reactions, specialized.reactions)
