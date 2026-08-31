from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from fem.assembly import AssemblyResult
from fem.analysis import incremental
from fem.constraints import ConstraintSet
from fem.model import DofSpace, NewtonStrategy, StaticControlMode
from fem.problem import ProblemContext, ProblemEvaluation, StaticEquilibriumProblem
from fem.state import EvaluationContext, SolutionState


@dataclass(slots=True)
class _MatrixAssembly:
    matrix: np.ndarray

    @property
    def num_dofs(self) -> int:
        return int(self.matrix.shape[0])

    @property
    def dof_space(self) -> DofSpace:
        return DofSpace.single_field("U", self.num_dofs)

    def assemble(
        self,
        solution: SolutionState,
        *,
        context: EvaluationContext | None = None,
    ) -> AssemblyResult:
        del context
        return AssemblyResult(
            residual=self.matrix @ solution.values,
            tangent=csr_matrix(self.matrix),
        )

    def begin_increment(self) -> None:
        pass

    def commit(self) -> None:
        pass

    def rollback(self) -> None:
        pass


@dataclass(slots=True)
class _CubicRamp:
    committed: float = 0.0
    trial: float = 0.0

    @property
    def dof_space(self) -> DofSpace:
        return DofSpace.single_field("U", 1)

    def evaluate(self, context: ProblemContext) -> ProblemEvaluation:
        if context.solution is None:
            raise ValueError("cubic ramp requires a solution")
        self.trial = float(context.solution.values[0])
        error = self.trial - float(context.load_factor)
        return ProblemEvaluation(
            residual=np.array([error + 0.1 * error**3]),
            tangent=csr_matrix([[1.0 + 0.3 * error**2]]),
        )

    def begin_increment(self, context: ProblemContext) -> None:
        del context
        self.trial = self.committed

    def commit(self) -> None:
        self.committed = self.trial

    def rollback(self) -> None:
        self.trial = self.committed


def test_mixed_displacement_control_projects_targets_and_separates_force_residual():
    matrix = np.array(
        [[3.0, -1.0, 0.0], [-1.0, 3.0, -1.0], [0.0, -1.0, 2.0]],
        dtype=float,
    )
    problem = StaticEquilibriumProblem(
        assembly=_MatrixAssembly(matrix),
        constraints=ConstraintSet({0: 0.0, 2: 4.0}),
        reference_load=np.zeros(3),
        control_mode=StaticControlMode.MIXED,
    )

    result = incremental.solve(
        problem,
        [0.25, 0.5, 1.0],
        residual_tolerance=1.0e-12,
        relative_residual_tolerance=1.0e-10,
        displacement_tolerance=1.0e-10,
        energy_tolerance=1.0e-12,
        constraint_tolerance=1.0e-12,
        predictor=False,
    )

    assert [item.solution.values[2] for item in result.increments] == pytest.approx(
        [1.0, 2.0, 4.0]
    )
    assert all(
        item.outputs["constraint_error"].max(initial=0.0) == pytest.approx(0.0)
        for item in result.increments
    )
    assert all(
        item.outputs["free_residual"].max(initial=0.0) == pytest.approx(0.0)
        for item in result.increments
    )
    assert result.increments[-1].outputs["convergence"]["constraint_norm"] == pytest.approx(
        0.0
    )


def test_modified_newton_reuses_one_tangent_and_predicts_from_committed_frames():
    result = incremental.solve(
        _CubicRamp(),
        [0.25, 0.5],
        max_iterations=20,
        residual_tolerance=1.0e-12,
        tangent_strategy=NewtonStrategy.MODIFIED,
        predictor=True,
    )

    assert [item.solution.values[0] for item in result.increments] == pytest.approx(
        [0.25, 0.5]
    )
    assert all(
        item.record is not None and item.record.tangent_strategy == "modified"
        for item in result.increments
    )
    assert result.increments[0].record is not None
    assert result.increments[0].record.tangent_evaluations == 1
    # The predictor lands exactly on the second target, so that increment
    # converges before a tangent is needed at all.
    assert result.increments[1].record is not None
    assert result.increments[1].record.tangent_evaluations == 0
    assert result.increments[0].record is not None
    assert result.increments[1].record is not None
    assert result.increments[0].record.predictor_used is False
    assert result.increments[1].record.predictor_used is True


def test_adaptive_growth_generates_exact_targets_from_current_increment():
    result = incremental.solve(
        _CubicRamp(),
        [0.2, 0.4, 0.6, 0.8, 1.0],
        max_iterations=20,
        residual_tolerance=1.0e-12,
        predictor=False,
        adaptive_growth=True,
        growth_factor=1.5,
        growth_iteration_threshold=10,
        maximum_increment=1.0,
    )

    assert [item.load_factor for item in result.increments] == pytest.approx(
        [0.2, 0.5, 0.95, 1.0]
    )
    assert result.final_solution.values[0] == pytest.approx(1.0)
    assert all(item.record is not None for item in result.increments)


def test_static_control_metadata_round_trips_phase2_policies():
    from fem.model import StaticStepControls

    controls = StaticStepControls(
        initial_increment=0.25,
        maximum_increments=8,
        newton_strategy=NewtonStrategy.MODIFIED,
        line_search=False,
        predictor=False,
        automatic_cutback=False,
        minimum_increment=1.0e-5,
        maximum_increment=0.75,
        adaptive_growth=True,
        growth_factor=1.8,
        growth_iteration_threshold=6,
    )

    assert StaticStepControls.from_metadata(controls.to_metadata()) == controls
