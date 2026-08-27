"""Generic load-increment driver built on the Problem/Newton contracts."""

from __future__ import annotations

from copy import deepcopy
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from time import perf_counter
from typing import Any, Sequence

import numpy as np

from fem.state import SolutionState

from ..problem import Problem, ProblemContext
from ..solver import newton


@dataclass(frozen=True, slots=True)
class IncrementAttemptRecord:
    """Neutral fact record for one load-increment attempt.

    This record belongs to the numerical incremental driver.  It deliberately
    contains no GUI or application objects, so the Job Monitor and result
    materializer can both consume the same accepted/failed attempt facts.
    """

    increment: int
    attempt: int
    previous_load_factor: float
    target_load_factor: float
    status: str = "converged"
    load_factor: float | None = None
    iterations: int = 0
    residual_norm: float | None = None
    duration_seconds: float | None = None
    cutback_to: float | None = None
    error: str | None = None
    failure_code: str | None = None
    force_norm: float | None = None
    relative_force_norm: float | None = None
    constraint_norm: float | None = None
    displacement_norm: float | None = None
    energy_norm: float | None = None
    tangent_strategy: str | None = None
    tangent_evaluations: int | None = None
    line_search_backtracks: int | None = None
    predictor_used: bool = False

    def __post_init__(self) -> None:
        for name in ("increment", "attempt", "iterations"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            minimum = 0 if name == "iterations" else 1
            if value < minimum:
                raise ValueError(f"{name} must be >= {minimum}")
        if self.status not in {"converged", "cutback", "failed"}:
            raise ValueError(
                "status must be 'converged', 'cutback', or 'failed'"
            )
        for name in (
            "previous_load_factor",
            "target_load_factor",
            "load_factor",
            "residual_norm",
            "duration_seconds",
            "cutback_to",
            "force_norm",
            "relative_force_norm",
            "constraint_norm",
            "displacement_norm",
            "energy_norm",
        ):
            value = getattr(self, name)
            if value is None:
                continue
            numeric = float(value)
            if not np.isfinite(numeric):
                raise ValueError(f"{name} must be finite")
            if name == "duration_seconds" and numeric < 0.0:
                raise ValueError("duration_seconds must be >= 0")
            object.__setattr__(self, name, numeric)
        if self.status == "converged" and self.load_factor is None:
            raise ValueError("converged attempts require load_factor")
        if self.error is not None and not isinstance(self.error, str):
            raise TypeError("error must be a string or None")
        if self.failure_code is not None and not isinstance(
            self.failure_code,
            str,
        ):
            raise TypeError("failure_code must be a string or None")
        for name in (
            "tangent_evaluations",
            "line_search_backtracks",
        ):
            value = getattr(self, name)
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer or None")
            if value < 0:
                raise ValueError(f"{name} must be >= 0")
        if self.tangent_strategy is not None and not isinstance(
            self.tangent_strategy,
            str,
        ):
            raise TypeError("tangent_strategy must be a string or None")
        if type(self.predictor_used) is not bool:
            raise TypeError("predictor_used must be bool")


@dataclass(frozen=True, slots=True)
class IncrementResult:
    """One converged load increment and its Newton diagnostics."""

    load_factor: float
    newton: newton.NewtonResult
    solution: SolutionState
    outputs: Mapping[str, Any] = field(default_factory=dict)
    local_outputs: tuple[Any, ...] = ()
    record: IncrementAttemptRecord | None = None

    def __post_init__(self) -> None:
        if type(self.solution) is not SolutionState:
            raise TypeError("solution must be exactly SolutionState")
        if self.newton.solution is not self.solution:
            raise ValueError("increment solution must match the Newton result")
        if not isinstance(self.outputs, Mapping):
            raise TypeError("outputs must be a mapping")
        object.__setattr__(self, "outputs", _owned_outputs(self.outputs))
        object.__setattr__(self, "local_outputs", tuple(self.local_outputs))
        if self.record is not None and type(self.record) is not IncrementAttemptRecord:
            raise TypeError("record must be IncrementAttemptRecord or None")


@dataclass(frozen=True, slots=True)
class IncrementalSolveResult:
    """All converged increments from one incremental solve."""

    increments: tuple[IncrementResult, ...]
    attempts: tuple[IncrementAttemptRecord, ...] = ()

    def __post_init__(self) -> None:
        increments = tuple(self.increments)
        if any(type(item) is not IncrementResult for item in increments):
            raise TypeError("increments must contain IncrementResult values")
        attempts = tuple(self.attempts)
        if any(type(item) is not IncrementAttemptRecord for item in attempts):
            raise TypeError(
                "attempts must contain IncrementAttemptRecord values"
            )
        if not attempts:
            attempts = tuple(
                item.record
                for item in increments
                if item.record is not None
            )
        object.__setattr__(self, "increments", increments)
        object.__setattr__(self, "attempts", attempts)

    @property
    def final_solution(self) -> SolutionState:
        """Return the last converged immutable solution state."""

        if not self.increments:
            raise ValueError("incremental solve result has no increments")
        return self.increments[-1].solution

    @property
    def final_load_factor(self) -> float:
        """Return the last converged load factor."""

        if not self.increments:
            raise ValueError("incremental solve result has no increments")
        return float(self.increments[-1].load_factor)


class IncrementalConvergenceError(RuntimeError):
    """Raised when one load increment fails after previous increments converge."""

    def __init__(
        self,
        message: str,
        *,
        failed_load_factor: float,
        completed: IncrementalSolveResult,
        cause: BaseException,
        cutbacks: int = 0,
    ) -> None:
        super().__init__(message)
        self.failed_load_factor = float(failed_load_factor)
        self.completed = completed
        self.cause = cause
        self.failed_newton = getattr(cause, "result", None)
        self.cutbacks = int(cutbacks)


def solve(
    problem: Problem,
    load_factors: Sequence[float],
    initial_state: SolutionState | None = None,
    *,
    context: ProblemContext | None = None,
    max_iterations: int = 25,
    residual_tolerance: float = 1.0e-8,
    relative_residual_tolerance: float | None = None,
    displacement_tolerance: float | None = None,
    energy_tolerance: float | None = None,
    constraint_tolerance: float | None = 1.0e-10,
    tangent_strategy: Any = "full",
    line_search: bool = True,
    predictor: bool = True,
    adaptive: bool = False,
    adaptive_growth: bool = False,
    growth_factor: float = 1.5,
    growth_iteration_threshold: int = 4,
    maximum_increments: int | None = None,
    minimum_increment: float = 1.0e-6,
    maximum_increment: float | None = None,
    monitor: Any | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> IncrementalSolveResult:
    """Solve a finite load-factor path one increment at a time.

    Successive load factors may increase or decrease for unloading and load
    reversal, but repeated adjacent factors are rejected. The converged
    solution from one increment is the initial state of the next.
    Stateful Problems are committed or rolled back by ``newton.solve`` at
    each increment boundary. When ``adaptive`` is true, a Newton iteration
    failure may replace the current target with a midpoint retry, subject to
    ``maximum_increments`` and ``minimum_increment``. When
    ``adaptive_growth`` is enabled on a monotonic path, an easy increment
    creates the next target from its actual size instead of selecting a
    rounded target from the original path.
    """

    factors = _validated_load_factors(load_factors)
    _validate_adaptive_controls(
        adaptive=adaptive,
        adaptive_growth=adaptive_growth,
        growth_factor=growth_factor,
        growth_iteration_threshold=growth_iteration_threshold,
        maximum_increments=maximum_increments,
        minimum_increment=minimum_increment,
        maximum_increment=maximum_increment,
        initial_increment=abs(factors[0]),
        required_increments=len(factors),
    )
    if type(predictor) is not bool:
        raise TypeError("predictor must be bool")
    newton.check_cancelled(should_cancel)
    base_context = context or ProblemContext()
    current = _initial_state(problem, initial_state, base_context)
    completed: list[IncrementResult] = []
    attempts: list[IncrementAttemptRecord] = []
    pending = list(factors)
    dynamic_growth = adaptive_growth and _is_monotonic_path(factors)
    final_factor = float(factors[-1])
    previous_factor = 0.0
    previous_solution: SolutionState | None = None
    previous_factor_for_predictor: float | None = None
    cutbacks = 0
    increment_index = 1
    attempt_index = 0

    while pending:
        newton.check_cancelled(should_cancel)
        factor = pending.pop(0)
        attempt_index += 1
        attempt_started = perf_counter()
        evaluation_context = replace(
            base_context.evaluation,
            load_factor=factor,
        )
        increment_context = replace(
            base_context,
            evaluation=evaluation_context,
            solution=current,
        )
        initial_for_increment, predictor_used = _predict_initial_state(
            current,
            previous_solution,
            previous_factor_for_predictor,
            previous_factor,
            factor,
            enabled=predictor,
        )
        if monitor is not None:
            monitor.increment_started(
                increment_index,
                attempt_index,
                factor,
                previous_factor,
            )
        try:
            result = newton.solve(
                problem,
                initial_for_increment,
                context=increment_context,
                max_iterations=max_iterations,
                residual_tolerance=residual_tolerance,
                relative_residual_tolerance=relative_residual_tolerance,
                displacement_tolerance=displacement_tolerance,
                energy_tolerance=energy_tolerance,
                constraint_tolerance=constraint_tolerance,
                tangent_strategy=tangent_strategy,
                line_search=line_search,
                monitor=monitor,
                increment_index=increment_index,
                attempt_index=attempt_index,
                should_cancel=should_cancel,
            )
        except newton.SolveCancelled:
            raise
        except Exception as exc:
            retry_factor = None
            within_limit = False
            retry_block_reason: str | None = None
            if adaptive and _is_cutback_retryable(exc):
                retry_factor = _cutback_factor(
                    previous_factor,
                    factor,
                    minimum_increment=minimum_increment,
                )
                if retry_factor is None:
                    retry_block_reason = _minimum_increment_reason(
                        previous_factor,
                        factor,
                        minimum_increment=minimum_increment,
                    )
                within_limit = (
                    maximum_increments is None
                    or len(completed) + len(pending) + 1
                    < maximum_increments
                )
                if retry_factor is not None and not within_limit:
                    retry_block_reason = (
                        "已达到最大增量数 "
                        f"{int(maximum_increments)}"
                        if maximum_increments is not None
                        else None
                    )
            will_retry = retry_factor is not None and within_limit
            failed_newton = getattr(exc, "result", None)
            failure_text = str(exc)
            if retry_block_reason:
                failure_text += f"；无法自动切步：{retry_block_reason}"
            failed_record = IncrementAttemptRecord(
                increment=increment_index,
                attempt=attempt_index,
                previous_load_factor=previous_factor,
                target_load_factor=factor,
                status="cutback" if will_retry else "failed",
                iterations=(
                    0
                    if failed_newton is None
                    else int(failed_newton.iterations)
                ),
                residual_norm=(
                    None
                    if failed_newton is None
                    else float(failed_newton.residual_norm)
                ),
                duration_seconds=max(0.0, perf_counter() - attempt_started),
                cutback_to=retry_factor if will_retry else None,
                error=failure_text,
                failure_code=_failure_code(exc),
                force_norm=_last_metric(failed_newton, "force_history"),
                relative_force_norm=_last_metric(
                    failed_newton,
                    "relative_force_history",
                ),
                constraint_norm=_last_metric(
                    failed_newton,
                    "constraint_history",
                ),
                displacement_norm=_last_metric(
                    failed_newton,
                    "displacement_history",
                ),
                energy_norm=_last_metric(failed_newton, "energy_history"),
                tangent_strategy=(
                    None
                    if failed_newton is None
                    else str(getattr(failed_newton, "tangent_strategy", "full"))
                ),
                tangent_evaluations=(
                    None
                    if failed_newton is None
                    else int(getattr(failed_newton, "tangent_evaluations", 0))
                ),
                line_search_backtracks=(
                    None
                    if failed_newton is None
                    else int(
                        getattr(failed_newton, "line_search_backtracks", 0)
                    )
                ),
                predictor_used=predictor_used,
            )
            attempts.append(failed_record)
            if monitor is not None:
                monitor.increment_failed(
                    increment_index,
                    attempt_index,
                    getattr(failed_newton, "residual_norm", None),
                    failure_text,
                    retryable=will_retry,
                    record=failed_record,
                )
            if will_retry:
                if monitor is not None:
                    monitor.cutback(
                        increment_index,
                        attempt_index,
                        factor,
                        retry_factor,
                    )
                pending.insert(0, factor)
                pending.insert(0, retry_factor)
                cutbacks += 1
                continue
            completed_result = IncrementalSolveResult(
                tuple(completed),
                tuple(attempts),
            )
            raise IncrementalConvergenceError(
                "incremental solve failed at "
                f"load_factor={factor:g}"
                + (
                    f"; 无法自动切步：{retry_block_reason}"
                    if retry_block_reason
                    else ""
                ),
                failed_load_factor=factor,
                completed=completed_result,
                cause=exc,
                cutbacks=cutbacks,
            ) from exc
        newton.check_cancelled(should_cancel)
        accepted_previous_solution = current
        current = result.solution
        evaluation = result.evaluation
        if evaluation is None:
            # Keep compatibility with externally constructed NewtonResult
            # values while the normal solver path reuses its final evaluation.
            evaluation = problem.evaluate(
                replace(increment_context, solution=current)
            )
        newton.check_cancelled(should_cancel)
        outputs = dict(evaluation.outputs)
        outputs["convergence"] = {
            "force_norm": _last_metric(result, "force_history"),
            "relative_force_norm": _last_metric(
                result,
                "relative_force_history",
            ),
            "constraint_norm": _last_metric(
                result,
                "constraint_history",
            ),
            "displacement_norm": _last_metric(
                result,
                "displacement_history",
            ),
            "energy_norm": _last_metric(result, "energy_history"),
        }
        outputs["solver_diagnostics"] = {
            "tangent_strategy": result.tangent_strategy,
            "tangent_evaluations": result.tangent_evaluations,
            "line_search_backtracks": result.line_search_backtracks,
            "predictor_used": predictor_used,
        }
        accepted_record = IncrementAttemptRecord(
            increment=increment_index,
            attempt=attempt_index,
            previous_load_factor=previous_factor,
            target_load_factor=factor,
            status="converged",
            load_factor=factor,
            iterations=int(result.iterations),
            residual_norm=float(result.residual_norm),
            duration_seconds=max(0.0, perf_counter() - attempt_started),
            force_norm=_last_metric(result, "force_history"),
            relative_force_norm=_last_metric(
                result,
                "relative_force_history",
            ),
            constraint_norm=_last_metric(result, "constraint_history"),
            displacement_norm=_last_metric(
                result,
                "displacement_history",
            ),
            energy_norm=_last_metric(result, "energy_history"),
            tangent_strategy=result.tangent_strategy,
            tangent_evaluations=result.tangent_evaluations,
            line_search_backtracks=result.line_search_backtracks,
            predictor_used=predictor_used,
        )
        attempts.append(accepted_record)
        completed.append(
            IncrementResult(
                load_factor=factor,
                newton=result,
                solution=current,
                outputs=outputs,
                local_outputs=evaluation.local_outputs,
                record=accepted_record,
            )
        )
        if monitor is not None:
            monitor.increment_converged(
                increment_index,
                attempt_index,
                factor,
                len(result.residual_history),
                result.residual_norm,
                record=accepted_record,
            )
        previous_solution = accepted_previous_solution
        previous_factor_for_predictor = previous_factor
        if dynamic_growth and result.iterations <= growth_iteration_threshold:
            pending = _grown_pending_targets(
                previous_factor,
                factor,
                pending,
                growth_factor=growth_factor,
                maximum_increment=maximum_increment,
                final_factor=final_factor,
            )
        previous_factor = factor
        increment_index += 1
        attempt_index = 0
    return IncrementalSolveResult(tuple(completed), tuple(attempts))


def _last_metric(result: Any, name: str) -> float | None:
    if result is None:
        return None
    values = getattr(result, name, ())
    if not values:
        return None
    return float(values[-1])


def _failure_code(error: BaseException) -> str:
    if isinstance(error, newton.NewtonConvergenceError):
        if "backtracking" in str(error).casefold():
            return "line_search_failure"
        return "newton_convergence"
    if isinstance(error, newton.SolveCancelled):
        return "cancelled"
    if isinstance(error, (ValueError, KeyError)):
        return "invalid_state"
    return "increment_failure"


def _validate_adaptive_controls(
    *,
    adaptive: bool,
    adaptive_growth: bool,
    growth_factor: float,
    growth_iteration_threshold: int,
    maximum_increments: int | None,
    minimum_increment: float,
    maximum_increment: float | None,
    initial_increment: float,
    required_increments: int,
) -> None:
    """Validate optional cutback controls before touching Problem state."""

    if type(adaptive) is not bool:
        raise ValueError("adaptive must be a bool")
    if type(adaptive_growth) is not bool:
        raise ValueError("adaptive_growth must be a bool")
    if maximum_increments is not None:
        if (
            isinstance(maximum_increments, bool)
            or not isinstance(maximum_increments, int)
            or maximum_increments < 1
        ):
            raise ValueError("maximum_increments must be an integer >= 1")
        if maximum_increments < required_increments and not adaptive_growth:
            raise ValueError(
                "maximum_increments must be at least the number of "
                "requested load factors"
            )
    try:
        minimum = float(minimum_increment)
    except (TypeError, ValueError) as exc:
        raise ValueError("minimum_increment must be finite and > 0") from exc
    if not np.isfinite(minimum) or minimum <= 0.0:
        raise ValueError("minimum_increment must be finite and > 0")
    if maximum_increment is not None:
        try:
            maximum = float(maximum_increment)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "maximum_increment must be finite and > 0"
            ) from exc
        if not np.isfinite(maximum) or maximum <= 0.0:
            raise ValueError("maximum_increment must be finite and > 0")
        if maximum < float(initial_increment):
            raise ValueError(
                "maximum_increment must be at least initial_increment"
            )
    try:
        growth = float(growth_factor)
    except (TypeError, ValueError) as exc:
        raise ValueError("growth_factor must be finite and > 1") from exc
    if not np.isfinite(growth) or growth <= 1.0:
        raise ValueError("growth_factor must be finite and > 1")
    if (
        isinstance(growth_iteration_threshold, bool)
        or not isinstance(growth_iteration_threshold, int)
        or growth_iteration_threshold < 1
    ):
        raise ValueError("growth_iteration_threshold must be an integer >= 1")


def _predict_initial_state(
    current: SolutionState,
    previous: SolutionState | None,
    previous_factor: float | None,
    current_factor: float,
    target_factor: float,
    *,
    enabled: bool,
) -> tuple[SolutionState, bool]:
    """Return a bounded secant predictor from the last two converged states."""

    if not enabled or previous is None or previous_factor is None:
        return current, False
    denominator = float(current_factor) - float(previous_factor)
    numerator = float(target_factor) - float(current_factor)
    if (
        not np.isfinite(denominator)
        or not np.isfinite(numerator)
        or denominator == 0.0
        or numerator == 0.0
        or numerator * denominator <= 0.0
    ):
        return current, False
    ratio = numerator / denominator
    if not np.isfinite(ratio) or ratio <= 0.0 or ratio > 1.5:
        return current, False
    predicted = current.with_values(
        current.values + ratio * (current.values - previous.values)
    )
    if not np.all(np.isfinite(predicted.values)):
        return current, False
    return predicted, True


def _grown_pending_targets(
    previous_factor: float,
    current_factor: float,
    pending: list[float],
    *,
    growth_factor: float,
    maximum_increment: float | None,
    final_factor: float,
) -> list[float]:
    """Generate the next exact target after an easy accepted increment."""

    if not pending:
        return pending
    increment = abs(float(current_factor) - float(previous_factor))
    if not np.isfinite(increment) or increment <= 0.0:
        return pending
    remaining = abs(float(final_factor) - float(current_factor))
    if not np.isfinite(remaining) or remaining <= 0.0:
        return pending
    direction = 1.0 if final_factor > current_factor else -1.0
    if (float(current_factor) - float(previous_factor)) * direction <= 0.0:
        return pending
    next_increment = increment * float(growth_factor)
    if maximum_increment is not None:
        next_increment = min(next_increment, float(maximum_increment))
    next_increment = min(next_increment, remaining)
    if not np.isfinite(next_increment) or next_increment <= 0.0:
        return pending
    if remaining <= next_increment + 1.0e-14:
        return [float(final_factor)]
    next_factor = float(current_factor) + direction * next_increment
    return [float(next_factor), float(final_factor)]


def _is_monotonic_path(factors: Sequence[float]) -> bool:
    """Return whether automatic growth can follow one load direction."""

    if not factors or float(factors[0]) == 0.0:
        return False
    direction = 1.0 if float(factors[0]) > 0.0 else -1.0
    return all(
        (float(current) - float(previous)) * direction > 0.0
        for previous, current in zip(factors, factors[1:])
    )


def _cutback_factor(
    previous_factor: float,
    target_factor: float,
    *,
    minimum_increment: float,
) -> float | None:
    """Return a midpoint retry target unless the interval is already tiny."""

    midpoint = previous_factor + 0.5 * (target_factor - previous_factor)
    if (
        not np.isfinite(midpoint)
        or midpoint == previous_factor
        or midpoint == target_factor
        or abs(midpoint - previous_factor) < minimum_increment
    ):
        return None
    return float(midpoint)


def _minimum_increment_reason(
    previous_factor: float,
    target_factor: float,
    *,
    minimum_increment: float,
) -> str:
    """Explain why midpoint cutback stopped at the configured floor."""

    interval = abs(float(target_factor) - float(previous_factor))
    return (
        f"当前载荷增量 {interval:g} 小于最小增量 "
        f"{float(minimum_increment):g}"
    )


def _is_cutback_retryable(error: BaseException) -> bool:
    """Return whether an increment failure is safe to retry at a midpoint."""

    if isinstance(error, newton.NewtonConvergenceError):
        return True
    if not isinstance(error, (RuntimeError, ValueError)):
        return False
    message = str(error).casefold()
    return (
        (
            "deformation gradient" in message
            and "determinant" in message
        )
        or "constitutive return mapping failed" in message
    )


def _validated_load_factors(values: Sequence[float]) -> tuple[float, ...]:
    """Validate a non-empty finite load path without repeated steps."""

    if isinstance(values, (str, bytes)):
        raise ValueError("load_factors must be a non-empty numeric sequence")
    try:
        factors = tuple(float(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise ValueError("load_factors must be a non-empty numeric sequence") from exc
    if not factors or not np.all(np.isfinite(factors)):
        raise ValueError("load_factors must be a non-empty finite sequence")
    if any(current == previous for previous, current in zip(factors, factors[1:])):
        raise ValueError("load_factors must not repeat adjacent values")
    return factors


def _initial_state(
    problem: Problem,
    initial: SolutionState | None,
    context: ProblemContext,
) -> SolutionState:
    """Resolve the initial global state for the first increment."""

    value = initial if initial is not None else context.solution
    if value is None:
        dof_space = getattr(problem, "dof_space", None)
        if dof_space is None:
            raise ValueError(
                "initial_state is required for Problems without dof_space"
            )
        return SolutionState.zeros(dof_space)
    if type(value) is not SolutionState:
        raise TypeError("initial_state must be exactly SolutionState")
    problem_space = getattr(problem, "dof_space", None)
    if problem_space is not None and value.dof_space != problem_space:
        raise ValueError("initial_state DOF space must match the Problem")
    return value


def _owned_outputs(value: Mapping[str, Any]) -> dict[str, Any]:
    """Detach arrays and nested history values captured at an increment."""

    return {
        str(key): _owned_output(item)
        for key, item in value.items()
    }


def _owned_output(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        owned = np.array(value, dtype=value.dtype, copy=True)
        owned.flags.writeable = False
        return owned
    if isinstance(value, Mapping):
        return {
            key: _owned_output(item)
            for key, item in value.items()
        }
    if isinstance(value, tuple):
        return tuple(_owned_output(item) for item in value)
    if isinstance(value, list):
        return [_owned_output(item) for item in value]
    try:
        return deepcopy(value)
    except (TypeError, ValueError):
        return value


__all__ = [
    "IncrementResult",
    "IncrementalConvergenceError",
    "IncrementalSolveResult",
    "solve",
]
