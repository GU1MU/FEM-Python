"""Static equilibrium equation independent of solver and element types."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix

from fem.assembly import Assembly
from fem.constraints import ConstraintSet, apply_dirichlet_equations
from fem.model import DofSpace, StaticControlMode
from fem.state import SolutionState

from .contracts import ProblemContext, ProblemEvaluation


@dataclass(slots=True)
class StaticEquilibriumProblem:
    """Global equilibrium shared by linear and nonlinear static analyses."""

    assembly: Assembly
    constraints: ConstraintSet
    reference_load: np.ndarray
    output_metadata: Mapping[str, Any] = field(default_factory=dict)
    control_mode: StaticControlMode = StaticControlMode.LOAD

    def __post_init__(self) -> None:
        if not isinstance(self.assembly, Assembly):
            raise TypeError("assembly must implement the Assembly contract")
        if type(self.constraints) is not ConstraintSet:
            raise TypeError("constraints must be exactly ConstraintSet")
        try:
            control_mode = StaticControlMode(
                str(self.control_mode).strip().casefold()
            )
        except ValueError as error:
            raise TypeError(
                "control_mode must be StaticControlMode or a supported value"
            ) from error
        self.control_mode = control_mode
        load = np.asarray(self.reference_load, dtype=float)
        if load.shape != (self.num_dofs,):
            raise ValueError(
                f"reference_load must have shape ({self.num_dofs},), "
                f"got {load.shape}"
            )
        if not np.all(np.isfinite(load)):
            raise ValueError("reference_load must contain finite values")
        owned = np.array(load, copy=True)
        owned.flags.writeable = False
        self.reference_load = owned
        self.output_metadata = MappingProxyType(dict(self.output_metadata))

    @property
    def num_dofs(self) -> int:
        return int(self.assembly.num_dofs)

    @property
    def dof_space(self) -> DofSpace:
        return self.assembly.dof_space

    def effective_constraints(self, load_factor: float) -> ConstraintSet:
        """Return the authored boundary targets at one control factor."""

        if self.control_mode is StaticControlMode.LOAD:
            return self.constraints
        return self.constraints.scaled(load_factor)

    def project_solution(
        self,
        solution: SolutionState,
        context: ProblemContext | None = None,
    ) -> SolutionState:
        """Project constrained DOFs before Newton evaluates equilibrium.

        Prescribed DOFs are kinematic targets, not unknowns that should be
        discovered by the global residual norm.  Projecting them first makes
        the Newton system solve only the remaining equilibrium correction.
        """

        if type(solution) is not SolutionState:
            raise TypeError("solution must be exactly SolutionState")
        if solution.dof_space != self.assembly.dof_space:
            raise ValueError("solution DOF space must match the problem assembly")
        context = context or ProblemContext()
        constraints = self.effective_constraints(context.load_factor)
        if not constraints.prescribed_values:
            return solution
        values = np.array(solution.values, copy=True)
        for raw_dof, raw_value in constraints.prescribed_values.items():
            dof = int(raw_dof)
            if dof < 0 or dof >= values.size:
                raise IndexError(f"constraint DOF {dof} is out of bounds")
            values[dof] = float(raw_value)
        return solution.with_values(values)

    def evaluate(self, context: ProblemContext | None = None) -> ProblemEvaluation:
        context = context or ProblemContext()
        solution = context.solution or SolutionState.zeros(self.assembly.dof_space)
        if solution.dof_space != self.assembly.dof_space:
            raise ValueError("solution DOF space must match the problem assembly")
        load_factor = _load_factor(context.load_factor)
        external = (
            self.reference_load
            if self.control_mode is StaticControlMode.DISPLACEMENT
            else self.reference_load * load_factor
        )
        constraints = self.effective_constraints(load_factor)
        assembled = self.assembly.assemble(
            solution,
            context=context.evaluation,
        )
        physical_residual = assembled.residual - external
        residual = physical_residual.copy()
        need_tangent = bool(
            context.evaluation.parameters.get("_fem_need_tangent", True)
        )
        constrained_dofs = tuple(
            int(dof) for dof in constraints.prescribed_values
        )
        if need_tangent:
            tangent = apply_dirichlet_equations(
                assembled.tangent,
                residual,
                solution.values,
                constraints,
            )
        else:
            tangent = csr_matrix((self.num_dofs, self.num_dofs), dtype=float)
            for dof in constrained_dofs:
                residual[dof] = (
                    solution.values[dof] - constraints.prescribed_values[dof]
                )
        constrained_lookup = set(constrained_dofs)
        free_dofs = tuple(
            dof
            for dof in range(self.num_dofs)
            if dof not in constrained_lookup
        )
        constraint_error = np.array(
            [
                solution.values[dof] - constraints.prescribed_values[dof]
                for dof in constrained_dofs
            ],
            dtype=float,
        )
        outputs = dict(self.output_metadata)
        outputs.update(assembled.outputs)
        outputs.update(
            {
                "load_factor": load_factor,
                "internal_force": assembled.residual,
                "external_force": external,
                "physical_residual": physical_residual,
                "free_residual": physical_residual[list(free_dofs)],
                "free_dofs": free_dofs,
                "constrained_dofs": constrained_dofs,
                "constraint_error": constraint_error,
                "constraint_values": dict(constraints.prescribed_values),
                "control_mode": self.control_mode.value,
                "consistent_tangent": csr_matrix(assembled.tangent),
            }
        )
        return ProblemEvaluation(
            residual=residual,
            tangent=tangent,
            mass=assembled.mass,
            damping=assembled.damping,
            constraints=constraints,
            outputs=outputs,
            local_outputs=assembled.local_outputs,
        )

    def begin_increment(self, context: ProblemContext) -> None:
        del context
        self.assembly.begin_increment()

    def commit(self) -> None:
        self.assembly.commit()

    def rollback(self) -> None:
        self.assembly.rollback()

    def reactions(
        self,
        solution: SolutionState,
        context: ProblemContext | None = None,
    ) -> np.ndarray:
        base_context = context or ProblemContext()
        evaluation = self.evaluate(replace(base_context, solution=solution))
        return np.asarray(evaluation.outputs["physical_residual"], dtype=float)


def _load_factor(value: Any) -> float:
    factor = float(value)
    if not np.isfinite(factor):
        raise ValueError("load_factor must be finite")
    return factor


__all__ = ["StaticEquilibriumProblem"]
