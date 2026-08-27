"""Newton algorithm over the generic Problem and SolutionState contracts."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix

from fem.model import NewtonStrategy
from fem.problem import Problem, ProblemContext, ProblemEvaluation, StatefulProblem
from fem.state import SolutionState

from . import convergence, linear

_MAX_STATEFUL_TRIAL_BACKTRACKS = 12
_TRIAL_BACKTRACK_FACTOR = 0.5


@dataclass(frozen=True, slots=True)
class NewtonResult:
    """Converged global solution and iteration diagnostics."""

    solution: SolutionState
    iterations: int
    residual_norm: float
    residual_history: tuple[float, ...]
    increment_history: tuple[float, ...]
    converged: bool = True
    trial_scale_history: tuple[float, ...] = ()
    force_history: tuple[float, ...] = ()
    relative_force_history: tuple[float, ...] = ()
    constraint_history: tuple[float, ...] = ()
    displacement_history: tuple[float, ...] = ()
    energy_history: tuple[float, ...] = ()
    tangent_strategy: str = NewtonStrategy.FULL.value
    tangent_evaluations: int = 0
    line_search_backtracks: int = 0
    evaluation: ProblemEvaluation | None = None

    def __post_init__(self) -> None:
        if type(self.solution) is not SolutionState:
            raise TypeError("solution must be exactly SolutionState")
        if self.evaluation is not None and type(self.evaluation) is not ProblemEvaluation:
            raise TypeError("evaluation must be exactly ProblemEvaluation or None")


class NewtonConvergenceError(RuntimeError):
    """Raised when a Newton increment reaches its iteration limit."""

    def __init__(self, message: str, result: NewtonResult) -> None:
        super().__init__(message)
        self.result = result


class SolveCancelled(RuntimeError):
    """Raised when a cooperative cancellation request reaches a solver."""


def check_cancelled(should_cancel: Callable[[], bool] | None) -> None:
    """Raise :class:`SolveCancelled` when the owning task asked to stop."""

    if should_cancel is None:
        return
    if not callable(should_cancel):
        raise TypeError("should_cancel must be callable or None")
    if should_cancel():
        raise SolveCancelled("solve cancelled")


def solve(
    problem: Problem,
    initial_state: SolutionState | None = None,
    *,
    context: ProblemContext | None = None,
    max_iterations: int = 25,
    residual_tolerance: float = 1.0e-8,
    relative_residual_tolerance: float | None = None,
    displacement_tolerance: float | None = None,
    energy_tolerance: float | None = None,
    constraint_tolerance: float | None = 1.0e-10,
    tangent_strategy: NewtonStrategy | str = NewtonStrategy.FULL,
    line_search: bool = True,
    monitor: Any | None = None,
    increment_index: int = 1,
    attempt_index: int = 1,
    should_cancel: Callable[[], bool] | None = None,
) -> NewtonResult:
    """Solve one nonlinear equation increment without knowing its physics.

    The algorithm updates the single global vector in ``SolutionState``. Named
    fields, elements, materials, time integration, and assembly remain owned by
    the Problem and lower layers.
    """

    _validate_controls(
        max_iterations,
        residual_tolerance,
        relative_residual_tolerance=relative_residual_tolerance,
        displacement_tolerance=displacement_tolerance,
        energy_tolerance=energy_tolerance,
        constraint_tolerance=constraint_tolerance,
        tangent_strategy=tangent_strategy,
        line_search=line_search,
    )
    strategy = _normalize_tangent_strategy(tangent_strategy)
    check_cancelled(should_cancel)
    base_context = context or ProblemContext()
    solution = _initial_state(problem, initial_state, base_context.solution)
    solution = _project_solution(problem, solution, base_context)
    current_context = replace(base_context, solution=solution)
    stateful = isinstance(problem, StatefulProblem)
    state_started = False
    # Integration-point output records are needed for the accepted increment,
    # not for every stateful Newton trial.  Keep the historical per-iteration
    # capture behavior available to callers that explicitly request it.  A
    # stateless Problem keeps the one-evaluation-per-Newton contract because
    # the generic solver cannot safely reconstruct its result payload.
    capture_outputs_each_iteration = bool(
        base_context.evaluation.parameters.get(
            "_fem_capture_outputs",
            not stateful,
        )
    )
    if stateful:
        check_cancelled(should_cancel)
        problem.begin_increment(current_context)
        state_started = True

    residual_history: list[float] = []
    increment_history: list[float] = []
    trial_scale_history: list[float] = []
    force_history: list[float] = []
    relative_force_history: list[float] = []
    constraint_history: list[float] = []
    displacement_history: list[float] = []
    energy_history: list[float] = []
    last_increment: np.ndarray | None = None
    cached_tangent: csr_matrix | None = None
    cached_factor: linear.SparseFactorization | None = None
    tangent_evaluations = 0
    line_search_backtracks = 0
    try:
        for iteration in range(max_iterations + 1):
            check_cancelled(should_cancel)
            evaluation_context = (
                current_context
                if capture_outputs_each_iteration
                else _evaluation_context(
                    current_context,
                    _fem_capture_outputs=False,
                )
            )
            evaluation = problem.evaluate(evaluation_context)
            check_cancelled(should_cancel)
            residual = _residual_vector(evaluation.residual)
            _validate_state_size(solution, residual.size)
            metrics = convergence.evaluate(
                evaluation,
                increment=last_increment,
                residual_tolerance=residual_tolerance,
            )
            residual_history.append(metrics.residual_norm)
            force_history.append(metrics.force_norm)
            relative_force_history.append(metrics.relative_force_norm)
            constraint_history.append(metrics.constraint_norm)
            displacement_history.append(metrics.displacement_norm)
            energy_history.append(metrics.energy_norm)
            if monitor is not None:
                monitor.newton_iteration(
                    increment_index,
                    attempt_index,
                    iteration + 1,
                    metrics.residual_norm,
                    increment_history[-1] if increment_history else None,
                    trial_scale=(
                        trial_scale_history[-1] if trial_scale_history else None
                    ),
                    force_norm=metrics.force_norm,
                    relative_force_norm=metrics.relative_force_norm,
                    constraint_norm=metrics.constraint_norm,
                    displacement_norm=metrics.displacement_norm,
                    energy_norm=metrics.energy_norm,
                )

            if convergence.converged(
                metrics,
                residual_tolerance=residual_tolerance,
                relative_residual_tolerance=relative_residual_tolerance,
                displacement_tolerance=displacement_tolerance,
                energy_tolerance=energy_tolerance,
                constraint_tolerance=constraint_tolerance,
            ):
                check_cancelled(should_cancel)
                if not capture_outputs_each_iteration and stateful:
                    # Re-evaluate the accepted solution once so result
                    # consumers receive the same complete output payload as
                    # before this optimization.  Keep tangent construction
                    # out of this output-only pass: the evaluation immediately
                    # above is at the same accepted solution and already owns
                    # the converged tangent.  Recomputing the material tangent
                    # here used to duplicate the most expensive constitutive
                    # work for every accepted increment.  The tangent is
                    # restored below so public NewtonResult semantics remain
                    # unchanged.
                    output_evaluation = problem.evaluate(
                        _evaluation_context(
                            current_context,
                            _fem_capture_outputs=True,
                            _fem_need_tangent=False,
                        )
                    )
                    check_cancelled(should_cancel)
                    if evaluation.tangent is not None:
                        output_values = dict(output_evaluation.outputs)
                        for tangent_key in (
                            "consistent_tangent",
                            "effective_tangent",
                            "material_tangent",
                            "geometric_tangent",
                        ):
                            if tangent_key in evaluation.outputs:
                                output_values[tangent_key] = evaluation.outputs[
                                    tangent_key
                                ]
                        evaluation = replace(
                            output_evaluation,
                            tangent=evaluation.tangent,
                            outputs=output_values,
                        )
                    else:
                        evaluation = output_evaluation
                result = _result(
                    solution,
                    iteration,
                    metrics,
                    residual_history,
                    increment_history,
                    trial_scale_history,
                    force_history,
                    relative_force_history,
                    constraint_history,
                    displacement_history,
                    energy_history,
                    tangent_strategy=strategy.value,
                    tangent_evaluations=tangent_evaluations,
                    line_search_backtracks=line_search_backtracks,
                    evaluation=evaluation,
                )
                if stateful:
                    problem.commit()
                    state_started = False
                return result

            if iteration == max_iterations:
                raise NewtonConvergenceError(
                    "Newton solver did not converge within max_iterations",
                    _result(
                        solution,
                        iteration,
                        metrics,
                        residual_history,
                        increment_history,
                        trial_scale_history,
                        force_history,
                        relative_force_history,
                        constraint_history,
                        displacement_history,
                        energy_history,
                        tangent_strategy=strategy.value,
                        tangent_evaluations=tangent_evaluations,
                        line_search_backtracks=line_search_backtracks,
                        converged=False,
                    ),
                )

            if cached_tangent is None or strategy is NewtonStrategy.FULL:
                next_tangent = _tangent_matrix(evaluation.tangent, residual.size)
                next_factor = linear.factorize(next_tangent)
                if cached_factor is not None:
                    cached_factor.close()
                cached_tangent = next_tangent
                cached_factor = next_factor
                tangent_evaluations += 1
            if cached_factor is None:
                raise RuntimeError("Newton solver has no sparse tangent factorization")
            check_cancelled(should_cancel)
            increment = cached_factor.solve(-residual)
            check_cancelled(should_cancel)
            increment_norm = float(np.linalg.norm(increment, ord=np.inf))
            if not np.isfinite(increment_norm):
                raise RuntimeError("Newton solver produced a non-finite increment")
            # Stateful material/geometric problems need an evaluated trial to
            # decide whether the Newton step should be shortened.  A pure
            # stateless Problem keeps the original one-evaluation-per-Newton
            # contract; callers that need line search can expose trial state
            # through StatefulProblem.
            if stateful:
                previous_solution = solution
                solution, current_context, backtracks = _trial_step(
                    problem,
                    base_context,
                    solution,
                    increment,
                    iteration=iteration,
                    current_metrics=metrics,
                    residual_history=residual_history,
                    increment_history=increment_history,
                    trial_scale_history=trial_scale_history,
                    force_history=force_history,
                    relative_force_history=relative_force_history,
                    constraint_history=constraint_history,
                    displacement_history=displacement_history,
                    energy_history=energy_history,
                    should_cancel=should_cancel,
                    residual_tolerance=residual_tolerance,
                    line_search=line_search,
                    stateful=stateful,
                    tangent_strategy=strategy.value,
                    tangent_evaluations=tangent_evaluations,
                    line_search_backtracks=line_search_backtracks,
                )
                last_increment = solution.values - previous_solution.values
                line_search_backtracks += backtracks
            else:
                increment_history.append(increment_norm)
                trial_scale_history.append(1.0)
                previous_solution = solution
                solution = _project_solution(
                    problem,
                    solution.with_values(solution.values + increment),
                    base_context,
                )
                last_increment = solution.values - previous_solution.values
                current_context = replace(base_context, solution=solution)
    except Exception:
        if stateful and state_started:
            problem.rollback()
        raise
    finally:
        if cached_factor is not None:
            cached_factor.close()

    raise AssertionError("Newton solver loop exited unexpectedly")


def _evaluation_context(
    context: ProblemContext,
    **parameters: Any,
) -> ProblemContext:
    """Return ``context`` with private evaluation flags overridden."""

    merged = dict(context.evaluation.parameters)
    merged.update(parameters)
    return replace(
        context,
        evaluation=replace(context.evaluation, parameters=merged),
    )


def _validate_controls(
    max_iterations: int,
    residual_tolerance: float,
    *,
    relative_residual_tolerance: float | None,
    displacement_tolerance: float | None,
    energy_tolerance: float | None,
    constraint_tolerance: float | None,
    tangent_strategy: NewtonStrategy | str,
    line_search: bool,
) -> None:
    if isinstance(max_iterations, bool) or not isinstance(max_iterations, int):
        raise ValueError("max_iterations must be an integer >= 1")
    if max_iterations < 1:
        raise ValueError("max_iterations must be an integer >= 1")
    try:
        tolerance = float(residual_tolerance)
    except (TypeError, ValueError) as exc:
        raise ValueError("residual_tolerance must be finite and > 0") from exc
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("residual_tolerance must be finite and > 0")
    for name, value in (
        ("relative_residual_tolerance", relative_residual_tolerance),
        ("displacement_tolerance", displacement_tolerance),
        ("energy_tolerance", energy_tolerance),
        ("constraint_tolerance", constraint_tolerance),
    ):
        if value is None:
            continue
        try:
            converted = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be finite and > 0") from exc
        if not np.isfinite(converted) or converted <= 0.0:
            raise ValueError(f"{name} must be finite and > 0")
    _normalize_tangent_strategy(tangent_strategy)
    if type(line_search) is not bool:
        raise TypeError("line_search must be bool")


def _normalize_tangent_strategy(value: NewtonStrategy | str) -> NewtonStrategy:
    try:
        return NewtonStrategy(str(value).strip().casefold())
    except ValueError as error:
        raise ValueError("tangent_strategy must be 'full' or 'modified'") from error


def _initial_state(
    problem: Problem,
    initial: SolutionState | None,
    context_state: SolutionState | None,
) -> SolutionState:
    value = initial if initial is not None else context_state
    if value is None:
        dof_space = getattr(problem, "dof_space", None)
        if dof_space is None:
            raise ValueError(
                "initial_state is required when Problem has no dof_space"
            )
        return SolutionState.zeros(dof_space)
    if type(value) is not SolutionState:
        raise TypeError("initial_state must be exactly SolutionState")
    problem_space = getattr(problem, "dof_space", None)
    if problem_space is not None and value.dof_space != problem_space:
        raise ValueError("initial_state DOF space must match the Problem")
    return value


def _validate_state_size(solution: SolutionState, size: int) -> None:
    if solution.dof_space.num_dofs != size:
        raise ValueError(
            "Problem residual size must match the SolutionState DOF space"
        )


def _residual_vector(value: Any) -> np.ndarray:
    if value is None:
        raise ValueError("Problem evaluation must provide residual")
    try:
        residual = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Problem residual must be a numeric vector") from exc
    if residual.ndim != 1 or residual.size == 0:
        raise ValueError("Problem residual must be a non-empty vector")
    if not np.all(np.isfinite(residual)):
        raise ValueError("Problem residual must contain finite values")
    return residual


def _tangent_matrix(value: Any, size: int) -> csr_matrix:
    if value is None:
        raise ValueError("Problem evaluation must provide tangent")
    try:
        tangent = csr_matrix(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("Problem tangent must be a numeric matrix") from exc
    if tangent.shape != (size, size):
        raise ValueError(
            f"Problem tangent must have shape ({size}, {size}), got {tangent.shape}"
        )
    if not np.all(np.isfinite(tangent.data)):
        raise ValueError("Problem tangent must contain finite values")
    return tangent


def _trial_step(
    problem: Problem,
    base_context: ProblemContext,
    solution: SolutionState,
    increment: np.ndarray,
    *,
    iteration: int,
    current_metrics: convergence.ConvergenceMetrics,
    residual_history: list[float],
    increment_history: list[float],
    trial_scale_history: list[float],
    force_history: list[float],
    relative_force_history: list[float],
    constraint_history: list[float],
    displacement_history: list[float],
    energy_history: list[float],
    should_cancel: Callable[[], bool] | None,
    residual_tolerance: float,
    line_search: bool,
    stateful: bool,
    tangent_strategy: str,
    tangent_evaluations: int,
    line_search_backtracks: int,
) -> tuple[SolutionState, ProblemContext, int]:
    """Accept a trial update, optionally reducing a non-decreasing residual.

    The same path is used for stateful and stateless Problems.  Stateful
    Problems roll back rejected material trials; stateless Problems simply
    discard the candidate vector.  This keeps line search a solver concern
    rather than a constitutive special case.
    """

    last_error: ValueError | None = None
    last_rejection: str | None = None
    scale = 1.0
    backtracks = 0
    maximum_trials = _MAX_STATEFUL_TRIAL_BACKTRACKS if line_search else 0
    for _ in range(maximum_trials + 1):
        check_cancelled(should_cancel)
        candidate = solution.with_values(solution.values + scale * increment)
        candidate = _project_solution(problem, candidate, base_context)
        candidate_parameters = dict(base_context.evaluation.parameters)
        candidate_parameters.update(
            {
                "_fem_need_tangent": False,
                "_fem_capture_outputs": False,
            }
        )
        candidate_context = replace(
            base_context,
            solution=candidate,
            evaluation=replace(
                base_context.evaluation,
                parameters=candidate_parameters,
            ),
        )
        try:
            evaluation = problem.evaluate(candidate_context)
            candidate_metrics = convergence.evaluate(
                evaluation,
                increment=candidate.values - solution.values,
                residual_tolerance=residual_tolerance,
                reference_scale=current_metrics.force_reference,
            )
            check_cancelled(should_cancel)
        except ValueError as error:
            if not line_search:
                raise
            last_error = error
            if stateful:
                assert isinstance(problem, StatefulProblem)
                problem.rollback()
            scale *= _TRIAL_BACKTRACK_FACTOR
            backtracks += 1
            continue

        if line_search and candidate_metrics.merit > current_metrics.merit:
            last_rejection = (
                "trial residual increased from "
                f"{current_metrics.merit:.3e} to "
                f"{candidate_metrics.merit:.3e}"
            )
            if stateful:
                assert isinstance(problem, StatefulProblem)
                problem.rollback()
            scale *= _TRIAL_BACKTRACK_FACTOR
            backtracks += 1
            continue

        accepted_increment = scale * increment
        increment_history.append(
            float(np.linalg.norm(accepted_increment, ord=np.inf))
        )
        trial_scale_history.append(float(scale))
        # The reduced candidate context is only for this residual check.  The
        # next Newton iteration must use the normal full-evaluation context.
        return candidate, replace(base_context, solution=candidate), backtracks

    result = _result(
        solution,
        iteration,
        current_metrics,
        residual_history,
        increment_history,
        trial_scale_history,
        force_history,
        relative_force_history,
        constraint_history,
        displacement_history,
        energy_history,
        tangent_strategy=tangent_strategy,
        tangent_evaluations=tangent_evaluations,
        line_search_backtracks=line_search_backtracks + backtracks,
        converged=False,
    )
    if last_rejection is not None:
        raise NewtonConvergenceError(
            "Newton trial step did not reduce the residual after backtracking: "
            + last_rejection,
            result,
        )
    if last_error is not None:
        raise NewtonConvergenceError(
            "Newton trial step remained invalid after backtracking: "
            + str(last_error),
            result,
        ) from last_error
    raise NewtonConvergenceError("Newton trial step could not be evaluated", result)


def _project_solution(
    problem: Problem,
    solution: SolutionState,
    context: ProblemContext,
) -> SolutionState:
    projector = getattr(problem, "project_solution", None)
    if projector is None:
        return solution
    if not callable(projector):
        raise TypeError("problem.project_solution must be callable")
    projected = projector(solution, context)
    if type(projected) is not SolutionState:
        raise TypeError("problem.project_solution must return SolutionState")
    return projected


def _result(
    solution: SolutionState,
    iterations: int,
    metrics: convergence.ConvergenceMetrics,
    residual_history: list[float],
    increment_history: list[float],
    trial_scale_history: list[float],
    force_history: list[float],
    relative_force_history: list[float],
    constraint_history: list[float],
    displacement_history: list[float],
    energy_history: list[float],
    *,
    converged: bool = True,
    tangent_strategy: str = NewtonStrategy.FULL.value,
    tangent_evaluations: int = 0,
    line_search_backtracks: int = 0,
    evaluation: ProblemEvaluation | None = None,
) -> NewtonResult:
    return NewtonResult(
        solution=solution,
        iterations=iterations,
        residual_norm=metrics.residual_norm,
        residual_history=tuple(residual_history),
        increment_history=tuple(increment_history),
        converged=converged,
        trial_scale_history=tuple(trial_scale_history),
        force_history=tuple(force_history),
        relative_force_history=tuple(relative_force_history),
        constraint_history=tuple(constraint_history),
        displacement_history=tuple(displacement_history),
        energy_history=tuple(energy_history),
        tangent_strategy=str(tangent_strategy),
        tangent_evaluations=int(tangent_evaluations),
        line_search_backtracks=int(line_search_backtracks),
        evaluation=evaluation,
    )


__all__ = [
    "NewtonConvergenceError",
    "NewtonResult",
    "SolveCancelled",
    "check_cancelled",
    "solve",
]
