from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from fem import materials
from fem.model import authoring as steps
from fem.assembly import AssemblyResult
from fem.analysis.compilation.boundary.compiled import compile_boundary
from fem.analysis.compilation.boundary.loads import build_load_vector
from fem.io import inp
from fem.model import (
    DofSpace,
    ElementSet,
    FEMModel,
    NodeSet,
    add_material,
    assign_section,
)
from fem.problem import Problem, ProblemContext, StaticEquilibriumProblem
from fem.state import EvaluationContext, SolutionState
from fem.selection import nodes
from fem.analysis import linear_static as static_linear
from fem.solver import newton
from tests.helpers.mesh_builders import make_quad4_stiffness_mesh


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INP_FIXTURE = PROJECT_ROOT / "examples" / "examples_data" / "cantilever_beam_3d.inp"


@dataclass
class _PreassembledLinearSystem:
    tangent: object

    @property
    def num_dofs(self) -> int:
        return int(self.tangent.shape[0])

    @property
    def dof_space(self):
        return DofSpace.single_field("U", self.num_dofs)

    def assemble(
        self,
        solution: SolutionState,
        *,
        context: EvaluationContext | None = None,
    ) -> AssemblyResult:
        del context
        return AssemblyResult(
            residual=np.asarray(self.tangent @ solution.values, dtype=float),
            tangent=self.tangent,
        )

    def begin_increment(self) -> None:
        pass

    def commit(self) -> None:
        pass

    def rollback(self) -> None:
        pass


def _make_model() -> FEMModel:
    model = FEMModel(
        mesh=make_quad4_stiffness_mesh(),
        name="nlf04_quad4_problem",
        node_sets={
            "fixed": NodeSet("fixed", (1, 4)),
            "loaded": NodeSet("loaded", (2, 3)),
        },
        element_sets={"plate": ElementSet("plate", (1,))},
    )
    add_material(
        model,
        materials.linear_elastic.material("steel", E=210.0, nu=0.3),
    )
    assign_section(model, "steel", "plate")
    step = steps.static("pull")
    steps.displacement(step, "fixed", components=(1, 2))
    steps.nodal_load(step, "loaded", component=1, value=1.0)
    steps.add(model, step)
    return model


def _problem(model: FEMModel, step: object = "pull") -> StaticEquilibriumProblem:
    prepared = static_linear.prepare(model, copy_model=False)
    boundary = compile_boundary(model, step)
    return StaticEquilibriumProblem(
        assembly=_PreassembledLinearSystem(prepared.base_stiffness),
        constraints=boundary.constraints,
        reference_load=build_load_vector(model.mesh, boundary.loads),
    )


def test_static_equilibrium_problem_is_solver_independent() -> None:
    problem = _problem(_make_model())

    assert isinstance(problem, Problem)
    assert not hasattr(problem, "solve")
    evaluation = problem.evaluate()
    assert evaluation.residual is not None
    assert evaluation.tangent is not None
    assert evaluation.residual.shape == (problem.num_dofs,)


def test_static_problem_uses_trial_displacement_and_load_factor() -> None:
    problem = _problem(_make_model())
    zero = problem.evaluate(
        ProblemContext(
            solution=SolutionState.zeros(problem.dof_space),
        )
    )
    trial_displacement = np.linspace(0.0, 0.001, problem.num_dofs)
    trial = problem.evaluate(
        ProblemContext(
            solution=SolutionState(problem.dof_space, trial_displacement),
            evaluation=EvaluationContext(load_factor=2.0),
        )
    )

    assert trial.residual is not None and zero.residual is not None
    assert trial.outputs["load_factor"] == 2.0


def test_generic_newton_matches_specialized_linear_solver() -> None:
    model = _make_model()
    problem = _problem(model)

    solved = newton.solve(
        problem,
        SolutionState.zeros(problem.dof_space),
        residual_tolerance=1.0e-12,
        max_iterations=3,
    )
    specialized = static_linear.solve(model, "pull")

    np.testing.assert_allclose(solved.solution.values, specialized.U)
    np.testing.assert_allclose(
        problem.reactions(solved.solution),
        specialized.reactions,
        atol=1.0e-12,
    )


def test_static_problem_accepts_preassembled_3d_system() -> None:
    model = inp.read(INP_FIXTURE)
    problem = _problem(model, None)
    solved = newton.solve(
        problem,
        SolutionState.zeros(problem.dof_space),
        # The imported 3D fixture is intentionally ill-conditioned; the
        # direct linear solve leaves an infinity-norm residual around 1e-8.
        residual_tolerance=1.0e-7,
        max_iterations=3,
    )
    specialized = static_linear.solve(model)

    tip_node = nodes.nearest(model.mesh, 5.0, 5.0, 0.0)
    dof = model.mesh.global_dof(tip_node, 1)
    assert solved.solution.values[dof] == pytest.approx(specialized.U[dof])


def test_static_problem_rejects_invalid_trial_vector() -> None:
    problem = _problem(_make_model())
    with pytest.raises(ValueError, match="solution DOF space"):
        problem.evaluate(
            ProblemContext(
                solution=SolutionState(
                    DofSpace.single_field("U", problem.num_dofs + 1),
                    np.zeros(problem.num_dofs + 1),
                )
            )
        )
