"""Implicit Newmark integration for nonlinear dynamic Problems."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import replace
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix

from fem.problem import NonlinearDynamicProblem, ProblemContext
from fem.state import EvaluationContext, SolutionState

from . import newmark, newton
from .linear import solve as solve_linear


def solve(
    problem: NonlinearDynamicProblem,
    times: Sequence[float],
    *,
    beta: float = 0.25,
    gamma: float = 0.5,
    max_iterations: int = 25,
    residual_tolerance: float = 1.0e-6,
    relative_residual_tolerance: float | None = 1.0e-8,
    displacement_tolerance: float | None = 1.0e-8,
    energy_tolerance: float | None = 1.0e-8,
    constraint_tolerance: float | None = 1.0e-10,
    tangent_strategy: Any = "full",
    line_search: bool = True,
    monitor: Any | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> newmark.TransientSolveResult:
    """Integrate nonlinear dynamics with Newmark and a Newton inner solve."""

    if not isinstance(problem, NonlinearDynamicProblem):
        raise TypeError("problem must be NonlinearDynamicProblem")
    beta = _positive("beta", beta)
    gamma = _positive("gamma", gamma)
    target_times = _times(times)
    if isinstance(max_iterations, bool) or not isinstance(max_iterations, int) or max_iterations < 1:
        raise ValueError("max_iterations must be an integer >= 1")
    _check_cancelled(should_cancel)

    displacement, velocity, acceleration = problem.initial_vectors()
    if acceleration is None:
        acceleration = _initial_acceleration(problem, displacement, velocity)
    state = _project_state(
        problem,
        SolutionState(problem.dof_space, displacement, velocity, acceleration),
    )
    previous_time = 0.0
    frames: list[newmark.DynamicFrame] = []
    previous_external = problem.reference_load * problem.amplitude.value_at(0.0)
    external_work = 0.0
    damping_dissipation = 0.0

    for increment_number, time in enumerate(target_times, start=1):
        _check_cancelled(should_cancel)
        dt = float(time) - previous_time
        previous_displacement = np.array(state.values, copy=True)
        callback = getattr(monitor, "dynamic_increment_started", None)
        if callable(callback):
            callback(
                increment_number,
                time,
                dt,
                solver_kind="dynamic_implicit",
                attempt=1,
            )
        a0 = 1.0 / (beta * dt * dt)
        a1 = gamma / (beta * dt)
        predictor_u = state.values + dt * state.first_derivative + dt * dt * (0.5 - beta) * state.second_derivative
        predictor_v = state.first_derivative + dt * (1.0 - gamma) * state.second_derivative
        predictor = SolutionState(
            problem.dof_space,
            predictor_u,
            predictor_v,
            state.second_derivative,
        )
        context = ProblemContext(
            solution=predictor,
            evaluation=EvaluationContext(
                time=time,
                load_factor=problem.amplitude.value_at(time),
                parameters={
                    "dynamic_predictor_u": np.array(predictor_u, copy=True),
                    "dynamic_predictor_v": np.array(predictor_v, copy=True),
                    "dynamic_a0": a0,
                    "dynamic_a1": a1,
                    "dynamic_gamma": gamma,
                    "dynamic_dt": dt,
                },
            ),
        )
        try:
            result = newton.solve(
                problem,
                predictor,
                context=context,
                max_iterations=max_iterations,
                residual_tolerance=residual_tolerance,
                relative_residual_tolerance=relative_residual_tolerance,
                displacement_tolerance=displacement_tolerance,
                energy_tolerance=energy_tolerance,
                constraint_tolerance=constraint_tolerance,
                tangent_strategy=tangent_strategy,
                line_search=line_search,
                monitor=monitor,
                increment_index=increment_number,
                attempt_index=1,
                should_cancel=should_cancel,
            )
        except newton.SolveCancelled:
            raise

        displacement = result.solution.values
        acceleration = a0 * (displacement - predictor_u)
        velocity = predictor_v + gamma * dt * acceleration
        state = _project_state(
            problem,
            SolutionState(
                problem.dof_space,
                displacement,
                velocity,
                acceleration,
            ),
        )
        final_evaluation = result.evaluation
        if final_evaluation is None:
            final_evaluation = problem.evaluate(
                replace(context, solution=state)
            )
        physical_residual = np.asarray(
            final_evaluation.outputs["physical_residual"],
            dtype=float,
        )
        external = problem.reference_load * problem.amplitude.value_at(time)
        external_work += 0.5 * float(
            (previous_external + external)
            @ (state.values - previous_displacement)
        )
        damping_dissipation += dt * float(
            state.first_derivative @ (problem.damping @ state.first_derivative)
        )
        residual_norm = _free_norm(problem, physical_residual)
        outputs = dict(final_evaluation.outputs)
        kinetic_energy = 0.5 * float(
            state.first_derivative @ (problem.mass @ state.first_derivative)
        )
        provided_internal = final_evaluation.outputs.get(
            "internal_energy",
            final_evaluation.outputs.get("strain_energy"),
        )
        strain_energy = (
            0.5
            * float(
                state.values
                @ np.asarray(
                    final_evaluation.outputs["internal_force"],
                    dtype=float,
                )
            )
            if provided_internal is None
            else _finite_energy(provided_internal, "internal_energy")
        )
        total_energy = kinetic_energy + strain_energy
        outputs.update(
            {
                "time": time,
                "step_time": time,
                "total_time": time,
                "time_increment": dt,
                "velocity": np.array(state.first_derivative, copy=True),
                "acceleration": np.array(state.second_derivative, copy=True),
                "newton_iterations": int(result.iterations),
                "newton_residual_norm": float(result.residual_norm),
                "kinetic_energy": kinetic_energy,
                "strain_energy": strain_energy,
                "internal_energy": strain_energy,
                "external_work": external_work,
                "damping_dissipation": damping_dissipation,
                "total_energy": total_energy,
                "energy_balance_error": total_energy - external_work + damping_dissipation,
                "increment_number": increment_number,
            }
        )
        frames.append(
            newmark.DynamicFrame(
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
                iterations=int(result.iterations),
                kinetic_energy=kinetic_energy,
                internal_energy=strain_energy,
                external_work=external_work,
                damping_dissipation=damping_dissipation,
                total_energy=total_energy,
                energy_balance_error=total_energy - external_work + damping_dissipation,
            )
        previous_time = time
        previous_external = external

    return newmark.TransientSolveResult(tuple(frames))


def _initial_acceleration(
    problem: NonlinearDynamicProblem,
    displacement: np.ndarray,
    velocity: np.ndarray,
) -> np.ndarray:
    solution = _project_state(
        problem,
        SolutionState(problem.dof_space, displacement, velocity),
    )
    try:
        problem.assembly.begin_increment()
        assembled = problem.assembly.assemble(
            solution,
            context=EvaluationContext(
                time=0.0,
                load_factor=problem.amplitude.value_at(0.0),
            ),
        )
        problem.assembly.commit()
    except BaseException:
        problem.assembly.rollback()
        raise
    external = problem.reference_load * problem.amplitude.value_at(0.0)
    rhs = external - assembled.residual - problem.damping @ velocity
    acceleration = _solve_mass(problem.mass, rhs, problem.constraints.prescribed_values)
    for dof in problem.constraints.prescribed_values:
        acceleration[int(dof)] = 0.0
    return acceleration


def _solve_mass(
    mass: csr_matrix,
    rhs: np.ndarray,
    constraints: dict[int, float] | Any,
) -> np.ndarray:
    values = np.asarray(rhs, dtype=float)
    constrained = tuple(sorted(int(dof) for dof in constraints))
    constrained_set = set(constrained)
    free = np.fromiter(
        (dof for dof in range(values.size) if dof not in constrained_set),
        dtype=int,
    )
    result = np.zeros(values.size, dtype=float)
    if free.size:
        reduced = mass[free][:, free]
        result[free] = solve_linear(reduced.tocsr(), values[free])
    return result


def _project_state(problem: NonlinearDynamicProblem, state: SolutionState) -> SolutionState:
    projected = problem.project_solution(state)
    values = np.array(projected.values, copy=True)
    velocity = (
        None
        if projected.first_derivative is None
        else np.array(projected.first_derivative, copy=True)
    )
    acceleration = (
        None
        if projected.second_derivative is None
        else np.array(projected.second_derivative, copy=True)
    )
    for dof in problem.constraints.prescribed_values:
        index = int(dof)
        if velocity is not None:
            velocity[index] = 0.0
        if acceleration is not None:
            acceleration[index] = 0.0
    return SolutionState(problem.dof_space, values, velocity, acceleration)


def _times(values: Sequence[float]) -> tuple[float, ...]:
    target = tuple(float(value) for value in values)
    if not target or any(not np.isfinite(value) or value <= 0.0 for value in target):
        raise ValueError("dynamic times must be finite and > 0")
    if any(right <= left for left, right in zip(target, target[1:])):
        raise ValueError("dynamic times must be strictly increasing")
    return target


def _free_norm(problem: NonlinearDynamicProblem, vector: np.ndarray) -> float:
    constrained = set(int(dof) for dof in problem.constraints.prescribed_values)
    free = [index for index in range(vector.size) if index not in constrained]
    return 0.0 if not free else float(np.linalg.norm(vector[free], ord=np.inf))


def _positive(name: str, value: Any) -> float:
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and > 0")
    return result


def _finite_energy(value: Any, name: str) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _check_cancelled(callback: Callable[[], bool] | None) -> None:
    if callback is not None and callback():
        raise newmark.DynamicSolveCancelled("dynamic analysis cancelled")


__all__ = ["solve"]
