"""Central-difference integration for explicit dynamics."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from fem.problem import ExplicitDynamicProblem, LinearDynamicProblem
from fem.state import SolutionState

from . import newmark


def solve(
    problem: ExplicitDynamicProblem | LinearDynamicProblem,
    *,
    time_period: float,
    initial_time_increment: float,
    minimum_time_increment: float,
    maximum_time_increment: float,
    maximum_increments: int,
    stable_time_step_scale: float = 0.9,
    monitor: Any | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> newmark.TransientSolveResult:
    """Integrate one explicit problem with an automatically stable step."""

    if not isinstance(problem, (ExplicitDynamicProblem, LinearDynamicProblem)):
        raise TypeError(
            "problem must be ExplicitDynamicProblem or LinearDynamicProblem"
        )
    period = _positive("time_period", time_period)
    initial = _positive("initial_time_increment", initial_time_increment)
    minimum = _positive("minimum_time_increment", minimum_time_increment)
    maximum = _positive("maximum_time_increment", maximum_time_increment)
    if not minimum <= initial <= maximum:
        raise ValueError("explicit time increments must satisfy minimum <= initial <= maximum")
    if isinstance(maximum_increments, bool) or not isinstance(maximum_increments, int) or maximum_increments < 1:
        raise ValueError("maximum_increments must be an integer >= 1")
    scale = float(stable_time_step_scale)
    if not np.isfinite(scale) or not 0.0 < scale < 1.0:
        raise ValueError("stable_time_step_scale must be between 0 and 1")
    _check_cancelled(should_cancel)

    displacement, velocity, acceleration = problem.initial_vectors()
    state = problem.project_solution(
        SolutionState(problem.dof_space, displacement, velocity, acceleration)
    )
    initial_evaluation = _accepted_evaluation(problem, state, time=0.0)
    if acceleration is None:
        acceleration = _solve_acceleration(
            problem,
            np.asarray(initial_evaluation.outputs["internal_force"], dtype=float),
            velocity,
            time=0.0,
        )
        state = problem.project_solution(
            SolutionState(problem.dof_space, displacement, velocity, acceleration)
        )
    stable = problem.stable_time_increment(state)
    stable_limit = float("inf") if not np.isfinite(stable) else scale * stable
    if stable_limit < minimum:
        raise ValueError(
            "explicit stable time increment is below minimum_time_increment"
        )

    frames: list[newmark.DynamicFrame] = []
    current_time = 0.0
    previous_acceleration = np.array(state.second_derivative, copy=True)
    previous_external = problem.reference_load * problem.amplitude.value_at(0.0)
    external_work = 0.0
    damping_dissipation = 0.0
    for increment_number in range(1, maximum_increments + 1):
        _check_cancelled(should_cancel)
        remaining = period - current_time
        if remaining <= 1.0e-12:
            break
        stable = problem.stable_time_increment(state)
        stable_limit = float("inf") if not np.isfinite(stable) else scale * stable
        if stable_limit < minimum:
            raise ValueError(
                "explicit stable time increment fell below minimum_time_increment"
            )
        dt = min(initial, maximum, stable_limit, remaining)
        if dt < minimum and remaining > minimum + 1.0e-12:
            raise ValueError(
                "explicit time increment fell below minimum_time_increment"
            )
        time = current_time + dt
        previous_displacement = np.array(state.values, copy=True)
        callback = getattr(monitor, "dynamic_increment_started", None)
        if callable(callback):
            callback(
                increment_number,
                time,
                dt,
                solver_kind="dynamic_explicit",
                stable_time_increment=stable,
            )
        predictor_displacement = (
            state.values
            + dt * state.first_derivative
            + 0.5 * dt * dt * previous_acceleration
        )
        half_velocity = state.first_derivative + 0.5 * dt * previous_acceleration
        trial = problem.project_solution(
            SolutionState(problem.dof_space, predictor_displacement, half_velocity)
        )
        try:
            problem.begin_increment()
            evaluation = problem.evaluate_internal(trial, time=time)
            internal = np.asarray(evaluation.outputs["internal_force"], dtype=float)
            acceleration_new = _solve_acceleration(
                problem,
                internal,
                half_velocity,
                time=time,
            )
            velocity_new = half_velocity + 0.5 * dt * acceleration_new
            state = problem.project_solution(
                SolutionState(
                    problem.dof_space,
                    predictor_displacement,
                    velocity_new,
                    acceleration_new,
                )
            )
            problem.commit()
        except BaseException:
            problem.rollback()
            raise

        external = problem.reference_load * problem.amplitude.value_at(time)
        external_work += 0.5 * float(
            (previous_external + external)
            @ (state.values - previous_displacement)
        )
        damping_dissipation += dt * float(
            state.first_derivative @ (problem.damping @ state.first_derivative)
        )
        physical_residual = (
            internal
            + problem.damping @ state.first_derivative
            + problem.mass @ state.second_derivative
            - external
        )
        residual_norm = _free_norm(problem, physical_residual)
        kinetic = 0.5 * float(
            state.first_derivative @ (problem.mass @ state.first_derivative)
        )
        provided_internal = evaluation.outputs.get(
            "internal_energy",
            evaluation.outputs.get("strain_energy"),
        )
        strain = (
            0.5 * float(state.values @ internal)
            if provided_internal is None
            else _finite_energy(provided_internal, "internal_energy")
        )
        total = kinetic + strain
        outputs = dict(evaluation.outputs)
        outputs.update(
            {
                "time": time,
                "step_time": time,
                "total_time": time,
                "time_increment": dt,
                "stable_time_increment": stable,
                "actual_time_increment": dt,
                "velocity": np.array(state.first_derivative, copy=True),
                "acceleration": np.array(state.second_derivative, copy=True),
                "physical_residual": np.array(physical_residual, copy=True),
                "kinetic_energy": kinetic,
                "strain_energy": strain,
                "internal_energy": strain,
                "external_work": external_work,
                "damping_dissipation": damping_dissipation,
                "total_energy": total,
                "energy_balance_error": total - external_work + damping_dissipation,
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
                solver_kind="dynamic_explicit",
                stable_time_increment=stable,
                kinetic_energy=kinetic,
                internal_energy=strain,
                external_work=external_work,
                damping_dissipation=damping_dissipation,
                total_energy=total,
                energy_balance_error=total - external_work + damping_dissipation,
            )
        current_time = time
        previous_acceleration = np.array(state.second_derivative, copy=True)
        previous_external = external

    if not frames or current_time < period - 1.0e-10:
        raise ValueError(
            "explicit analysis did not reach the requested time period within maximum_increments"
        )
    return newmark.TransientSolveResult(tuple(frames))


def _accepted_evaluation(
    problem: ExplicitDynamicProblem | LinearDynamicProblem,
    state: SolutionState,
    *,
    time: float,
):
    try:
        problem.begin_increment()
        evaluation = problem.evaluate_internal(state, time=time)
        problem.commit()
        return evaluation
    except BaseException:
        problem.rollback()
        raise


def _solve_acceleration(
    problem: ExplicitDynamicProblem | LinearDynamicProblem,
    internal: np.ndarray,
    velocity: np.ndarray,
    *,
    time: float,
) -> np.ndarray:
    external = problem.reference_load * problem.amplitude.value_at(time)
    rhs = external - internal - problem.damping @ velocity
    diagonal = np.asarray(problem.mass_diagonal, dtype=float)
    acceleration = np.zeros(problem.num_dofs, dtype=float)
    free = [
        index
        for index in range(problem.num_dofs)
        if index not in problem.constraints.prescribed_values
    ]
    if free:
        acceleration[free] = rhs[free] / diagonal[free]
    return acceleration


def _free_norm(problem: ExplicitDynamicProblem, vector: np.ndarray) -> float:
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
