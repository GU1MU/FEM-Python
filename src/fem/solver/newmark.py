"""Newmark average-acceleration driver for linear transient dynamics."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix

from fem.problem import LinearDynamicProblem
from fem.state import SolutionState

from .linear import solve as solve_linear


class DynamicSolveCancelled(RuntimeError):
    """Raised when cooperative cancellation reaches a time increment."""


@dataclass(frozen=True, slots=True)
class DynamicFrame:
    """One accepted transient time frame."""

    time: float
    time_increment: float
    solution: SolutionState
    reactions: np.ndarray
    residual_norm: float
    outputs: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        time = float(self.time)
        increment = float(self.time_increment)
        residual = float(self.residual_norm)
        if not np.isfinite(time) or time < 0.0:
            raise ValueError("dynamic frame time must be finite and >= 0")
        if not np.isfinite(increment) or increment <= 0.0:
            raise ValueError("dynamic frame time_increment must be finite and > 0")
        if not np.isfinite(residual) or residual < 0.0:
            raise ValueError("dynamic frame residual_norm must be finite and >= 0")
        if type(self.solution) is not SolutionState:
            raise TypeError("dynamic frame solution must be SolutionState")
        reactions = np.asarray(self.reactions, dtype=float)
        if reactions.shape != (self.solution.dof_space.num_dofs,):
            raise ValueError("dynamic frame reactions must match the DOF space")
        if not np.all(np.isfinite(reactions)):
            raise ValueError("dynamic frame reactions must be finite")
        owned = np.array(reactions, copy=True)
        owned.flags.writeable = False
        object.__setattr__(self, "time", time)
        object.__setattr__(self, "time_increment", increment)
        object.__setattr__(self, "residual_norm", residual)
        object.__setattr__(self, "reactions", owned)
        object.__setattr__(self, "outputs", MappingProxyType(dict(self.outputs)))


@dataclass(frozen=True, slots=True)
class TransientSolveResult:
    """All accepted frames from one linear transient solve."""

    frames: tuple[DynamicFrame, ...]

    def __post_init__(self) -> None:
        frames = tuple(self.frames)
        if not frames or any(type(frame) is not DynamicFrame for frame in frames):
            raise ValueError("transient solve must contain DynamicFrame values")
        times = tuple(frame.time for frame in frames)
        if times != tuple(sorted(times)) or len(set(times)) != len(times):
            raise ValueError("dynamic frame times must be strictly increasing")
        object.__setattr__(self, "frames", frames)

    @property
    def final(self) -> DynamicFrame:
        return self.frames[-1]


def solve(
    problem: LinearDynamicProblem,
    times: Sequence[float],
    *,
    beta: float = 0.25,
    gamma: float = 0.5,
    monitor: Any | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> TransientSolveResult:
    """Integrate ``M a + C v + K u = F(t)`` with Newmark's method."""

    if not isinstance(problem, LinearDynamicProblem):
        raise TypeError("problem must be LinearDynamicProblem")
    beta = _positive_finite("beta", beta)
    gamma = _positive_finite("gamma", gamma)
    if beta <= 0.0 or gamma <= 0.0:
        raise ValueError("Newmark beta and gamma must be > 0")
    target_times = tuple(float(value) for value in times)
    if not target_times or any(
        not np.isfinite(value) or value <= 0.0 for value in target_times
    ):
        raise ValueError("dynamic times must be finite and > 0")
    if any(right <= left for left, right in zip(target_times, target_times[1:])):
        raise ValueError("dynamic times must be strictly increasing")

    u, v, acceleration = _initial_state(problem)
    previous_time = 0.0
    previous_force = problem.reference_load * problem.amplitude.value_at(0.0)
    if acceleration is None:
        acceleration = _solve_constrained(
            problem.mass,
            previous_force - problem.stiffness @ u - problem.damping @ v,
            {int(dof): 0.0 for dof in problem.constraints.prescribed_values},
        )
    else:
        acceleration = np.array(acceleration, copy=True)
    for dof in problem.constraints.prescribed_values:
        acceleration[int(dof)] = 0.0

    frames: list[DynamicFrame] = []
    external_work = 0.0
    for increment_number, time in enumerate(target_times, start=1):
        _check_cancelled(should_cancel)
        dt = time - previous_time
        callback = getattr(monitor, "dynamic_increment_started", None)
        if callable(callback):
            callback(
                increment_number,
                time,
                dt,
                solver_kind="dynamic_implicit",
            )
        predictor_u = u + dt * v + dt * dt * (0.5 - beta) * acceleration
        predictor_v = v + dt * (1.0 - gamma) * acceleration
        a0 = 1.0 / (beta * dt * dt)
        a1 = gamma / (beta * dt)
        effective = problem.stiffness + a0 * problem.mass + a1 * problem.damping
        force = problem.reference_load * problem.amplitude.value_at(time)
        rhs = (
            force
            + a0 * (problem.mass @ predictor_u)
            + a1 * (problem.damping @ predictor_u)
            - problem.damping @ predictor_v
        )
        displacement = _solve_constrained(
            effective,
            rhs,
            problem.constraints.prescribed_values,
        )
        acceleration_new = a0 * (displacement - predictor_u)
        velocity_new = predictor_v + gamma * dt * acceleration_new
        _project_constraints(
            displacement,
            velocity_new,
            acceleration_new,
            problem.constraints.prescribed_values,
        )
        physical_residual = problem.reactions(
            displacement,
            velocity_new,
            acceleration_new,
            time,
        )
        constrained = tuple(int(dof) for dof in problem.constraints.prescribed_values)
        free = np.array(
            [dof for dof in range(problem.num_dofs) if dof not in constrained],
            dtype=int,
        )
        residual_norm = (
            0.0
            if free.size == 0
            else float(np.linalg.norm(physical_residual[free], ord=np.inf))
        )
        external_work += 0.5 * float(
            (previous_force + force) @ (displacement - u)
        )
        state = SolutionState(
            problem.dof_space,
            displacement,
            velocity_new,
            acceleration_new,
        )
        kinetic_energy = 0.5 * float(
            velocity_new @ (problem.mass @ velocity_new)
        )
        strain_energy = 0.5 * float(
            displacement @ (problem.stiffness @ displacement)
        )
        total_energy = kinetic_energy + strain_energy
        outputs = {
            "time": time,
            "step_time": time,
            "total_time": time,
            "time_increment": dt,
            "velocity": np.array(velocity_new, copy=True),
            "acceleration": np.array(acceleration_new, copy=True),
            "physical_residual": np.array(physical_residual, copy=True),
            "external_force": np.array(force, copy=True),
            "kinetic_energy": kinetic_energy,
            "strain_energy": strain_energy,
            "external_work": external_work,
            "total_energy": total_energy,
            "energy_balance_error": total_energy - external_work,
            "increment_number": increment_number,
        }
        frames.append(
            DynamicFrame(
                time=time,
                time_increment=dt,
                solution=state,
                reactions=physical_residual,
                residual_norm=residual_norm,
                outputs=outputs,
            )
        )
        callback = getattr(monitor, "dynamic_increment_converged", None)
        if callable(callback):
            callback(
                increment_number,
                time,
                dt,
                residual_norm,
                solver_kind="dynamic_implicit",
                iterations=0,
                kinetic_energy=kinetic_energy,
                internal_energy=strain_energy,
                external_work=external_work,
                damping_dissipation=None,
                total_energy=total_energy,
                energy_balance_error=total_energy - external_work,
            )
        previous_time = time
        previous_force = force
        u, v, acceleration = displacement, velocity_new, acceleration_new
    return TransientSolveResult(tuple(frames))


def _initial_state(
    problem: LinearDynamicProblem,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    displacement, velocity, acceleration = problem.initial_vectors()
    for dof, value in problem.constraints.prescribed_values.items():
        index = int(dof)
        if index in problem.initial_conditions.displacement:
            if not np.isclose(displacement[index], value, rtol=0.0, atol=1.0e-12):
                raise ValueError(
                    f"initial displacement conflicts with prescribed DOF {index}"
                )
        displacement[index] = float(value)
        if abs(velocity[index]) > 1.0e-12:
            raise ValueError(f"initial velocity on prescribed DOF {index} must be zero")
        if acceleration is not None and abs(acceleration[index]) > 1.0e-12:
            raise ValueError(
                f"initial acceleration on prescribed DOF {index} must be zero"
            )
        velocity[index] = 0.0
    return displacement, velocity, acceleration


def _solve_constrained(
    matrix: csr_matrix,
    rhs: np.ndarray,
    constraints: Mapping[int, float],
) -> np.ndarray:
    matrix = csr_matrix(matrix, dtype=float)
    values = np.asarray(rhs, dtype=float)
    size = matrix.shape[0]
    if matrix.shape != (size, size) or values.shape != (size,):
        raise ValueError("dynamic linear system shapes do not match")
    constrained = tuple(sorted(int(dof) for dof in constraints))
    constrained_set = set(constrained)
    free = np.asarray(
        [dof for dof in range(size) if dof not in constrained_set],
        dtype=int,
    )
    solution = np.zeros(size, dtype=float)
    if constrained:
        solution[list(constrained)] = [float(constraints[dof]) for dof in constrained]
    if free.size:
        reduced = matrix[free][:, free]
        reduced_rhs = values[free]
        if constrained:
            reduced_rhs = reduced_rhs - matrix[free][:, list(constrained)] @ solution[list(constrained)]
        solution[free] = solve_linear(reduced.tocsr(), np.asarray(reduced_rhs).ravel())
    return solution


def _project_constraints(
    displacement: np.ndarray,
    velocity: np.ndarray,
    acceleration: np.ndarray,
    constraints: Mapping[int, float],
) -> None:
    for dof, value in constraints.items():
        index = int(dof)
        displacement[index] = float(value)
        velocity[index] = 0.0
        acceleration[index] = 0.0


def _positive_finite(name: str, value: Any) -> float:
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and > 0")
    return result


def _check_cancelled(callback: Callable[[], bool] | None) -> None:
    if callback is not None and callback():
        raise DynamicSolveCancelled("dynamic analysis cancelled")


__all__ = [
    "DynamicFrame",
    "DynamicSolveCancelled",
    "TransientSolveResult",
    "solve",
]
