"""Linear transient dynamics problem contract."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix

from fem.assembly import Assembly
from fem.constraints import ConstraintSet, apply_dirichlet_equations
from fem.model import DofSpace, InitialConditionSet, TimeAmplitude
from fem.state import EvaluationContext, SolutionState

from .contracts import ProblemContext, ProblemEvaluation


@dataclass(slots=True)
class LinearDynamicProblem:
    """One compiled small-strain linear system for Newmark integration."""

    dof_space: DofSpace
    stiffness: csr_matrix
    mass: csr_matrix
    damping: csr_matrix
    constraints: ConstraintSet
    reference_load: np.ndarray
    amplitude: TimeAmplitude = field(default_factory=TimeAmplitude)
    initial_conditions: InitialConditionSet = field(
        default_factory=InitialConditionSet,
    )
    output_metadata: dict[str, Any] = field(default_factory=dict)
    _mass_diagonal: np.ndarray = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if type(self.dof_space) is not DofSpace:
            raise TypeError("dof_space must be exactly DofSpace")
        size = self.dof_space.num_dofs
        for name in ("stiffness", "mass", "damping"):
            matrix = csr_matrix(getattr(self, name), dtype=float)
            if matrix.shape != (size, size):
                raise ValueError(f"{name} must have shape ({size}, {size})")
            if not np.all(np.isfinite(matrix.data)):
                raise ValueError(f"{name} must contain finite values")
            matrix.sum_duplicates()
            object.__setattr__(self, name, matrix)
        mass_diagonal = np.array(self.mass.diagonal(), dtype=float, copy=True)
        mass_diagonal.flags.writeable = False
        object.__setattr__(self, "_mass_diagonal", mass_diagonal)
        if type(self.constraints) is not ConstraintSet:
            raise TypeError("constraints must be exactly ConstraintSet")
        for dof in self.constraints.prescribed_values:
            if int(dof) >= size:
                raise IndexError(f"constraint DOF {dof} is out of bounds")
        load = np.asarray(self.reference_load, dtype=float)
        if load.shape != (size,) or not np.all(np.isfinite(load)):
            raise ValueError(f"reference_load must have shape ({size},) and be finite")
        owned_load = np.array(load, copy=True)
        owned_load.flags.writeable = False
        object.__setattr__(self, "reference_load", owned_load)
        if not isinstance(self.amplitude, TimeAmplitude):
            raise TypeError("amplitude must be a TimeAmplitude")
        if not isinstance(self.initial_conditions, InitialConditionSet):
            raise TypeError("initial_conditions must be InitialConditionSet")
        for name in ("displacement", "velocity", "acceleration"):
            invalid = [
                int(dof)
                for dof in getattr(self.initial_conditions, name)
                if int(dof) >= size
            ]
            if invalid:
                raise IndexError(f"initial {name} DOF {invalid[0]} is out of bounds")
        object.__setattr__(
            self,
            "output_metadata",
            MappingProxyType(dict(self.output_metadata)),
        )

    @property
    def num_dofs(self) -> int:
        return self.dof_space.num_dofs

    @property
    def mass_diagonal(self) -> np.ndarray:
        """Return the immutable diagonal used by explicit stepping."""

        return self._mass_diagonal

    @property
    def assembly(self) -> "LinearDynamicProblem":
        """Expose a procedure-neutral compiled-system view."""

        return self

    @property
    def bindings(self) -> tuple[Any, ...]:
        return ()

    def evaluate(self, context: ProblemContext) -> ProblemEvaluation:
        if type(context) is not ProblemContext:
            raise TypeError("context must be exactly ProblemContext")
        solution = context.solution or SolutionState.zeros(self.dof_space)
        if solution.dof_space != self.dof_space:
            raise ValueError("solution DOF space must match the dynamic problem")
        amplitude = self.amplitude.value_at(context.time)
        external = self.reference_load * amplitude
        physical_residual = self.stiffness @ solution.values - external
        residual = physical_residual.copy()
        tangent = apply_dirichlet_equations(
            self.stiffness,
            residual,
            solution.values,
            self.constraints,
        )
        outputs = dict(self.output_metadata)
        outputs.update(
            {
                "time": context.time,
                "load_factor": amplitude,
                "internal_force": self.stiffness @ solution.values,
                "external_force": external,
                "physical_residual": physical_residual,
            }
        )
        return ProblemEvaluation(
            residual=residual,
            tangent=tangent,
            mass=self.mass,
            damping=self.damping,
            constraints=self.constraints,
            outputs=outputs,
        )

    def initial_vectors(self) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """Return owned global initial displacement, velocity, and acceleration."""

        size = self.num_dofs
        vectors = [np.zeros(size, dtype=float) for _ in range(3)]
        for index, name in enumerate(("displacement", "velocity", "acceleration")):
            for dof, value in getattr(self.initial_conditions, name).items():
                vectors[index][int(dof)] = float(value)
        explicit_acceleration = (
            vectors[2]
            if self.initial_conditions.acceleration
            else None
        )
        return vectors[0], vectors[1], explicit_acceleration

    def reactions(self, displacement: np.ndarray, velocity: np.ndarray, acceleration: np.ndarray, time: float) -> np.ndarray:
        """Return the dynamic residual, whose constrained entries are reactions."""

        u = np.asarray(displacement, dtype=float)
        v = np.asarray(velocity, dtype=float)
        a = np.asarray(acceleration, dtype=float)
        external = self.reference_load * self.amplitude.value_at(time)
        return np.asarray(self.stiffness @ u + self.damping @ v + self.mass @ a - external)

    def project_solution(self, solution: SolutionState) -> SolutionState:
        """Project a central-difference trial state onto fixed DOFs."""

        if type(solution) is not SolutionState:
            raise TypeError("solution must be exactly SolutionState")
        if solution.dof_space != self.dof_space:
            raise ValueError("solution DOF space must match the dynamic problem")
        values = np.array(solution.values, copy=True)
        velocity = (
            None
            if solution.first_derivative is None
            else np.array(solution.first_derivative, copy=True)
        )
        acceleration = (
            None
            if solution.second_derivative is None
            else np.array(solution.second_derivative, copy=True)
        )
        for dof, value in self.constraints.prescribed_values.items():
            index = int(dof)
            values[index] = float(value)
            if velocity is not None:
                velocity[index] = 0.0
            if acceleration is not None:
                acceleration[index] = 0.0
        return SolutionState(self.dof_space, values, velocity, acceleration)

    def evaluate_internal(
        self,
        solution: SolutionState,
        *,
        time: float,
    ) -> ProblemEvaluation:
        """Return the internal-force evaluation used by explicit stepping."""

        projected = self.project_solution(solution)
        internal = np.asarray(self.stiffness @ projected.values, dtype=float)
        external = self.reference_load * self.amplitude.value_at(time)
        outputs = dict(self.output_metadata)
        outputs.update(
            {
                "time": float(time),
                "load_factor": self.amplitude.value_at(time),
                "internal_force": np.array(internal, copy=True),
                "external_force": np.array(external, copy=True),
            }
        )
        return ProblemEvaluation(
            residual=internal - external,
            tangent=self.stiffness,
            mass=self.mass,
            damping=self.damping,
            constraints=self.constraints,
            outputs=outputs,
        )

    def begin_increment(self) -> None:
        """Keep the explicit problem lifecycle uniform with stateful paths."""

    def commit(self) -> None:
        """Linear explicit assembly has no material history to commit."""

    def rollback(self) -> None:
        """Linear explicit assembly has no trial state to roll back."""

    def stable_time_increment(
        self,
        solution: SolutionState | None = None,
    ) -> float:
        """Estimate a conservative diagonal central-difference increment."""

        del solution
        stiffness_diagonal = np.asarray(self.stiffness.diagonal(), dtype=float)
        mass_diagonal = self.mass_diagonal
        positive = (stiffness_diagonal > 0.0) & (mass_diagonal > 0.0)
        if not np.any(positive):
            return float("inf")
        return float(
            np.min(
                np.sqrt(
                    2.0
                    * mass_diagonal[positive]
                    / stiffness_diagonal[positive]
                )
            )
        )




@dataclass(slots=True)
class NonlinearDynamicProblem:
    """Implicit dynamic equilibrium over a stateful assembled model.

    The Newmark driver supplies predictor quantities through the evaluation
    context.  This keeps time integration in the solver while the Problem
    owns only inertia, damping, constraints, and internal-force assembly.
    """

    assembly: Assembly
    mass: csr_matrix
    damping: csr_matrix
    constraints: ConstraintSet
    reference_load: np.ndarray
    amplitude: TimeAmplitude = field(default_factory=TimeAmplitude)
    initial_conditions: InitialConditionSet = field(
        default_factory=InitialConditionSet,
    )
    output_metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.assembly, Assembly):
            raise TypeError("assembly must implement the Assembly contract")
        size = self.assembly.num_dofs
        for name in ("mass", "damping"):
            matrix = csr_matrix(getattr(self, name), dtype=float)
            if matrix.shape != (size, size):
                raise ValueError(f"{name} must have shape ({size}, {size})")
            if not np.all(np.isfinite(matrix.data)):
                raise ValueError(f"{name} must contain finite values")
            matrix.sum_duplicates()
            object.__setattr__(self, name, matrix)
        if type(self.constraints) is not ConstraintSet:
            raise TypeError("constraints must be exactly ConstraintSet")
        load = np.asarray(self.reference_load, dtype=float)
        if load.shape != (size,) or not np.all(np.isfinite(load)):
            raise ValueError(
                f"reference_load must have shape ({size},) and be finite"
            )
        owned_load = np.array(load, copy=True)
        owned_load.flags.writeable = False
        object.__setattr__(self, "reference_load", owned_load)
        if not isinstance(self.amplitude, TimeAmplitude):
            raise TypeError("amplitude must be a TimeAmplitude")
        if not isinstance(self.initial_conditions, InitialConditionSet):
            raise TypeError("initial_conditions must be InitialConditionSet")
        for name in ("displacement", "velocity", "acceleration"):
            invalid = [
                int(dof)
                for dof in getattr(self.initial_conditions, name)
                if int(dof) >= size
            ]
            if invalid:
                raise IndexError(
                    f"initial {name} DOF {invalid[0]} is out of bounds"
                )
        object.__setattr__(
            self,
            "output_metadata",
            MappingProxyType(dict(self.output_metadata)),
        )

    @property
    def dof_space(self) -> DofSpace:
        return self.assembly.dof_space

    @property
    def num_dofs(self) -> int:
        return self.assembly.num_dofs

    def project_solution(
        self,
        solution: SolutionState,
        context: ProblemContext | None = None,
    ) -> SolutionState:
        if type(solution) is not SolutionState:
            raise TypeError("solution must be exactly SolutionState")
        if solution.dof_space != self.dof_space:
            raise ValueError("solution DOF space must match the dynamic problem")
        values = np.array(solution.values, copy=True)
        velocity = (
            None
            if solution.first_derivative is None
            else np.array(solution.first_derivative, copy=True)
        )
        acceleration = (
            None
            if solution.second_derivative is None
            else np.array(solution.second_derivative, copy=True)
        )
        for dof, value in self.constraints.prescribed_values.items():
            index = int(dof)
            values[index] = float(value)
            if velocity is not None:
                velocity[index] = 0.0
            if acceleration is not None:
                acceleration[index] = 0.0
        return SolutionState(
            self.dof_space,
            values,
            velocity,
            acceleration,
        )

    def evaluate(self, context: ProblemContext) -> ProblemEvaluation:
        if type(context) is not ProblemContext:
            raise TypeError("context must be exactly ProblemContext")
        solution = context.solution or SolutionState.zeros(self.dof_space)
        if solution.dof_space != self.dof_space:
            raise ValueError("solution DOF space must match the dynamic problem")
        parameters = context.evaluation.parameters
        required = ("dynamic_predictor_u", "dynamic_predictor_v", "dynamic_a0", "dynamic_a1", "dynamic_gamma", "dynamic_dt")
        missing = [name for name in required if name not in parameters]
        if missing:
            raise ValueError(
                "implicit dynamic evaluation is missing "
                + ", ".join(missing)
            )
        predictor_u = np.asarray(parameters["dynamic_predictor_u"], dtype=float)
        predictor_v = np.asarray(parameters["dynamic_predictor_v"], dtype=float)
        if predictor_u.shape != (self.num_dofs,) or predictor_v.shape != (self.num_dofs,):
            raise ValueError("dynamic predictor vectors must match the DOF space")
        a0 = float(parameters["dynamic_a0"])
        a1 = float(parameters["dynamic_a1"])
        gamma = float(parameters["dynamic_gamma"])
        dt = float(parameters["dynamic_dt"])
        if not np.isfinite(a0) or not np.isfinite(a1) or not np.isfinite(gamma) or not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("dynamic Newmark parameters must be finite")
        acceleration = a0 * (solution.values - predictor_u)
        velocity = predictor_v + gamma * dt * acceleration
        evaluation_context = context.evaluation
        assembled = self.assembly.assemble(
            solution,
            context=evaluation_context,
        )
        external = self.reference_load * self.amplitude.value_at(context.time)
        physical_residual = (
            assembled.residual
            + self.damping @ velocity
            + self.mass @ acceleration
            - external
        )
        residual = physical_residual.copy()
        need_tangent = bool(
            context.evaluation.parameters.get("_fem_need_tangent", True)
        )
        if need_tangent:
            effective_tangent = (
                assembled.tangent + a0 * self.mass + a1 * self.damping
            )
            tangent = apply_dirichlet_equations(
                effective_tangent,
                residual,
                solution.values,
                self.constraints,
            )
        else:
            effective_tangent = csr_matrix((self.num_dofs, self.num_dofs), dtype=float)
            for dof, value in self.constraints.prescribed_values.items():
                index = int(dof)
                residual[index] = solution.values[index] - float(value)
            tangent = effective_tangent
        outputs = dict(self.output_metadata)
        outputs.update(assembled.outputs)
        outputs.update(
            {
                "time": context.time,
                "load_factor": self.amplitude.value_at(context.time),
                "velocity": np.array(velocity, copy=True),
                "acceleration": np.array(acceleration, copy=True),
                "internal_force": np.array(assembled.residual, copy=True),
                "external_force": np.array(external, copy=True),
                "physical_residual": np.array(physical_residual, copy=True),
                "effective_tangent": effective_tangent,
            }
        )
        return ProblemEvaluation(
            residual=residual,
            tangent=tangent,
            mass=self.mass,
            damping=self.damping,
            constraints=self.constraints,
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

    def initial_vectors(self) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        size = self.num_dofs
        vectors = [np.zeros(size, dtype=float) for _ in range(3)]
        for index, name in enumerate(("displacement", "velocity", "acceleration")):
            for dof, value in getattr(self.initial_conditions, name).items():
                vectors[index][int(dof)] = float(value)
        return vectors[0], vectors[1], vectors[2] if self.initial_conditions.acceleration else None


@dataclass(slots=True)
class ExplicitDynamicProblem:
    """Explicit dynamic problem backed by the common assembled operators."""

    assembly: Assembly
    mass: csr_matrix
    damping: csr_matrix
    constraints: ConstraintSet
    reference_load: np.ndarray
    amplitude: TimeAmplitude = field(default_factory=TimeAmplitude)
    initial_conditions: InitialConditionSet = field(
        default_factory=InitialConditionSet,
    )
    output_metadata: dict[str, Any] = field(default_factory=dict)
    _mass_diagonal: np.ndarray = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.assembly, Assembly):
            raise TypeError("assembly must implement the Assembly contract")
        size = self.assembly.num_dofs
        mass = csr_matrix(self.mass, dtype=float)
        damping = csr_matrix(self.damping, dtype=float)
        if mass.shape != (size, size) or damping.shape != (size, size):
            raise ValueError("explicit dynamic matrices must match the assembly DOFs")
        if not np.all(np.isfinite(mass.data)) or not np.all(np.isfinite(damping.data)):
            raise ValueError("explicit dynamic matrices must be finite")
        diagonal = mass.diagonal()
        if np.any(diagonal <= 0.0) or mass.nnz != len(diagonal):
            raise ValueError("explicit dynamics requires a positive diagonal mass matrix")
        object.__setattr__(self, "mass", mass)
        object.__setattr__(self, "damping", damping)
        mass_diagonal = np.array(diagonal, dtype=float, copy=True)
        mass_diagonal.flags.writeable = False
        object.__setattr__(self, "_mass_diagonal", mass_diagonal)
        if type(self.constraints) is not ConstraintSet:
            raise TypeError("constraints must be exactly ConstraintSet")
        load = np.asarray(self.reference_load, dtype=float)
        if load.shape != (size,) or not np.all(np.isfinite(load)):
            raise ValueError("explicit reference_load must match the assembly DOFs")
        owned = np.array(load, copy=True)
        owned.flags.writeable = False
        object.__setattr__(self, "reference_load", owned)
        if not isinstance(self.amplitude, TimeAmplitude):
            raise TypeError("amplitude must be a TimeAmplitude")
        if not isinstance(self.initial_conditions, InitialConditionSet):
            raise TypeError("initial_conditions must be InitialConditionSet")
        object.__setattr__(self, "output_metadata", MappingProxyType(dict(self.output_metadata)))

    @property
    def dof_space(self) -> DofSpace:
        return self.assembly.dof_space

    @property
    def num_dofs(self) -> int:
        return self.assembly.num_dofs

    @property
    def mass_diagonal(self) -> np.ndarray:
        """Return the immutable diagonal used by central-difference updates."""

        return self._mass_diagonal

    def project_solution(self, solution: SolutionState) -> SolutionState:
        values = np.array(solution.values, copy=True)
        velocity = (
            None
            if solution.first_derivative is None
            else np.array(solution.first_derivative, copy=True)
        )
        acceleration = (
            None
            if solution.second_derivative is None
            else np.array(solution.second_derivative, copy=True)
        )
        for dof, value in self.constraints.prescribed_values.items():
            index = int(dof)
            values[index] = float(value)
            if velocity is not None:
                velocity[index] = 0.0
            if acceleration is not None:
                acceleration[index] = 0.0
        return SolutionState(
            self.dof_space,
            values,
            velocity,
            acceleration,
        )

    def evaluate_internal(
        self,
        solution: SolutionState,
        *,
        time: float,
    ) -> ProblemEvaluation:
        assembled = self.assembly.assemble(
            self.project_solution(solution),
            context=EvaluationContext(time=time, load_factor=self.amplitude.value_at(time)),
        )
        external = self.reference_load * self.amplitude.value_at(time)
        outputs = dict(self.output_metadata)
        outputs.update(assembled.outputs)
        outputs.update(
            {
                "time": time,
                "load_factor": self.amplitude.value_at(time),
                "internal_force": np.array(assembled.residual, copy=True),
                "external_force": np.array(external, copy=True),
            }
        )
        return ProblemEvaluation(
            residual=assembled.residual - external,
            tangent=assembled.tangent,
            mass=self.mass,
            damping=self.damping,
            constraints=self.constraints,
            outputs=outputs,
            local_outputs=assembled.local_outputs,
        )

    def evaluate(self, context: ProblemContext) -> ProblemEvaluation:
        """Expose the common Problem contract for compiled validation."""

        if type(context) is not ProblemContext:
            raise TypeError("context must be exactly ProblemContext")
        solution = context.solution or SolutionState.zeros(self.dof_space)
        return self.evaluate_internal(solution, time=context.time)

    def begin_increment(self) -> None:
        self.assembly.begin_increment()

    def commit(self) -> None:
        self.assembly.commit()

    def rollback(self) -> None:
        self.assembly.rollback()

    def initial_vectors(self) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        size = self.num_dofs
        vectors = [np.zeros(size, dtype=float) for _ in range(3)]
        for index, name in enumerate(("displacement", "velocity", "acceleration")):
            for dof, value in getattr(self.initial_conditions, name).items():
                vectors[index][int(dof)] = float(value)
        return vectors[0], vectors[1], vectors[2] if self.initial_conditions.acceleration else None

    def stable_time_increment(
        self,
        solution: SolutionState | None = None,
    ) -> float:
        """Estimate a conservative central-difference time increment."""

        state = (
            SolutionState.zeros(self.dof_space)
            if solution is None
            else self.project_solution(solution)
        )
        try:
            self.begin_increment()
            evaluation = self.evaluate_internal(state, time=0.0)
        finally:
            self.rollback()
        stiffness_diagonal = np.asarray(evaluation.tangent.diagonal(), dtype=float)
        mass_diagonal = self.mass_diagonal
        positive = stiffness_diagonal > 0.0
        if not np.any(positive):
            return float("inf")
        return float(np.min(np.sqrt(2.0 * mass_diagonal[positive] / stiffness_diagonal[positive])))


__all__ = [
    "ExplicitDynamicProblem",
    "LinearDynamicProblem",
    "NonlinearDynamicProblem",
]
