"""Application-level dispatch for the supported static analysis paths.

The GUI owns the task/session lifecycle, while this module owns the decision
about which numerical path is allowed to execute.  Keeping that decision here
prevents a nonlinear solver from leaking into widgets or from being selected
by accident for an ordinary linear step.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

import numpy as np

from fem.analysis import (
    AnalysisCancelled,
    AnalysisRequest,
    AnalysisPath,
    DEFAULT_ANALYSIS_EXECUTOR,
    ProcedureContext,
    analysis_path_for,
    check_cancelled,
    execution_plan_cache_scope,
    incremental,
    resolve_analysis_request,
)
from fem.model import AnalysisStep, resolve_analysis_step
from fem.results import ModelResult

from .run_monitor import SolveMonitor


AnalysisSolverKind = Literal[
    "linear_static",
    "nonlinear_static",
    "dynamic_implicit",
    "dynamic_explicit",
]


class AnalysisConvergenceError(RuntimeError):
    """Expose a failed nonlinear increment without losing prior frames."""

    def __init__(
        self,
        message: str,
        *,
        failed_load_factor: float,
        completed: incremental.IncrementalSolveResult,
        partial_result: ModelResult | None,
        cause: BaseException,
        cutbacks: int = 0,
    ) -> None:
        if type(completed) is not incremental.IncrementalSolveResult:
            raise TypeError(
                "completed must be IncrementalSolveResult"
            )
        if partial_result is not None and type(partial_result) is not ModelResult:
            raise TypeError(
                "partial_result must be ModelResult or None"
            )
        if not isinstance(cause, BaseException):
            raise TypeError("cause must be an exception")
        try:
            failed_factor = float(failed_load_factor)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "failed_load_factor must be a finite scalar"
            ) from error
        if not np.isfinite(failed_factor):
            raise ValueError("failed_load_factor must be a finite scalar")
        self.failed_load_factor = failed_factor
        self.completed = completed
        self.partial_result = partial_result
        self.cause = cause
        self.failed_newton = getattr(cause, "result", None)
        self.cutbacks = int(cutbacks)
        last = completed.increments[-1] if completed.increments else None
        completed_text = f"已保留 {len(completed.increments)} 个收敛增量"
        if last is None:
            diagnostics = "尚无可用的 Newton 收敛历史"
        else:
            diagnostics = (
                f"最后一次 Newton: iterations={last.newton.iterations}, "
                f"residual_norm={last.newton.residual_norm:.3e}"
            )
        failed = self.failed_newton
        if failed is not None:
            failed_diagnostics = (
                f"失败增量 Newton: iterations={failed.iterations}, "
                f"residual_norm={failed.residual_norm:.3e}"
            )
            cause_text = str(cause).strip()
            if cause_text:
                failed_diagnostics += f"；失败原因: {cause_text}"
        else:
            failed_diagnostics = f"失败原因: {cause}"
        cutback_text = (
            f"，自动切步重试 {self.cutbacks} 次"
            if self.cutbacks
            else ""
        )
        detail = (
            f"{message}; failed_load_factor={failed_factor:g}; "
            f"{completed_text}{cutback_text}; {diagnostics}; "
            f"{failed_diagnostics}"
        )
        super().__init__(detail)


def selected_analysis_step(
    model: Any,
    step: str | int | AnalysisStep | None,
) -> AnalysisStep:
    """Resolve one step through the same boundary used by Problems."""

    selected = resolve_analysis_step(model, step)
    if selected is None:
        raise ValueError("an analysis step is required")
    return selected


def analysis_request_for_step(
    model: Any,
    step: str | int | AnalysisStep | None,
    *,
    max_iterations: int | None = None,
    residual_tolerance: float | None = None,
) -> AnalysisRequest:
    """Resolve one selected step into the public execution snapshot."""

    selected = selected_analysis_step(model, step)
    return resolve_analysis_request(
        selected,
        max_iterations=max_iterations,
        residual_tolerance=residual_tolerance,
    )


def analysis_solver_kind(
    model: Any,
    step: str | int | AnalysisStep | None,
    *,
    request: AnalysisRequest | None = None,
) -> AnalysisSolverKind:
    """Return the numerical path selected by one resolved request.

    Existing callers may still provide only ``model`` and ``step``.  The
    application solve path passes its already-resolved request so the same
    step metadata is not decoded a second time during one execution.
    """

    if request is None:
        request = analysis_request_for_step(model, step)
    return analysis_path_for(model, request).value


@execution_plan_cache_scope()
def solve_analysis(
    model: Any,
    step: str | int | AnalysisStep | None = None,
    *,
    name: str | None = None,
    prepared_system: Any | None = None,
    timings: dict[str, float] | None = None,
    max_iterations: int | None = None,
    residual_tolerance: float | None = None,
    monitor: SolveMonitor | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> ModelResult:
    """Execute one supported analysis and publish one common run contract.

    Linear statics keeps its direct sparse factorization internally, while
    implicit and explicit dynamics expose the same application-level
    lifecycle: monitor events, accepted increments, and result frames.
    """

    _start_monitor(monitor)
    try:
        result = _solve_analysis_impl(
            model,
            step,
            name=name,
            prepared_system=prepared_system,
            timings=timings,
            max_iterations=max_iterations,
            residual_tolerance=residual_tolerance,
            monitor=monitor,
            should_cancel=should_cancel,
        )
    except AnalysisCancelled:
        _finish_monitor_cancelled(monitor)
        raise
    except BaseException as error:
        _finish_monitor_failed(monitor, error)
        raise
    _finish_monitor_succeeded(monitor)
    return result


def _solve_analysis_impl(
    model: Any,
    step: str | int | AnalysisStep | None = None,
    *,
    name: str | None = None,
    prepared_system: Any | None = None,
    timings: dict[str, float] | None = None,
    max_iterations: int | None = None,
    residual_tolerance: float | None = None,
    monitor: SolveMonitor | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> ModelResult:
    """Execute one supported analysis through the application seam.

    ``prepared_system`` is meaningful only for the linear path.  Nonlinear
    execution deliberately starts from the issued model and creates its own
    Problem/material-point state, so the existing Session cache cannot be
    mistaken for a nonlinear tangent or history cache.
    """

    check_cancelled(should_cancel)
    selected = selected_analysis_step(model, step)
    request = resolve_analysis_request(
        selected,
        max_iterations=max_iterations,
        residual_tolerance=residual_tolerance,
    )
    path = analysis_path_for(model, request)
    procedure = DEFAULT_ANALYSIS_EXECUTOR.procedure_for(request, model=model)
    procedure.validate(model, request)
    if path is AnalysisPath.LINEAR_STATIC:
        if monitor is not None:
            monitor.stage_changed("线性静力求解")
    elif path is AnalysisPath.DYNAMIC_IMPLICIT:
        if monitor is not None:
            monitor.stage_changed("隐式动力学求解")
    elif path is AnalysisPath.DYNAMIC_EXPLICIT:
        if monitor is not None:
            monitor.stage_changed("显式动力学求解")
    else:
        if monitor is not None:
            monitor.stage_changed("非线性 Newton 求解")

    if prepared_system is not None and path not in {
        AnalysisPath.LINEAR_STATIC,
        AnalysisPath.DYNAMIC_IMPLICIT,
        AnalysisPath.DYNAMIC_EXPLICIT,
    }:
        raise ValueError(
            "nonlinear static analysis cannot use a linear prepared system"
        )

    check_cancelled(should_cancel)
    prepared = prepared_system
    if prepared is None:
        prepared = DEFAULT_ANALYSIS_EXECUTOR.prepare(
            model,
            request,
            timings=timings,
        )
    context = ProcedureContext(
        monitor=monitor,
        should_cancel=should_cancel,
        timings=timings,
    )
    try:
        numerical_result = DEFAULT_ANALYSIS_EXECUTOR.run(
            model,
            selected,
            request,
            prepared,
            name=name,
            context=context,
        )
    except incremental.IncrementalConvergenceError as error:
        partial_result = (
            DEFAULT_ANALYSIS_EXECUTOR.materialize(
                model,
                selected,
                request,
                error.completed,
                name=name,
                prepared=prepared,
            )
            if error.completed.increments
            else None
        )
        raise AnalysisConvergenceError(
            str(error),
            failed_load_factor=error.failed_load_factor,
            completed=error.completed,
            partial_result=partial_result,
            cause=error.cause,
            cutbacks=error.cutbacks,
        ) from error
    return DEFAULT_ANALYSIS_EXECUTOR.materialize(
        model,
        selected,
        request,
        numerical_result,
        name=name,
        prepared=prepared,
    )


def prepare_linear_analysis(
    model: Any,
    *,
    timings: dict[str, float] | None = None,
) -> Any:
    """Prepare the optimized linear procedure behind the application seam."""

    selected = selected_analysis_step(model, None)
    if selected is None:
        raise ValueError("a linear static analysis step is required")
    request = resolve_analysis_request(selected)
    if analysis_path_for(model, request) is not AnalysisPath.LINEAR_STATIC:
        raise ValueError("a linear static analysis step is required")
    return DEFAULT_ANALYSIS_EXECUTOR.prepare(
        model,
        request,
        timings=timings,
    )


def _start_monitor(monitor: SolveMonitor | None) -> None:
    """Start a lifecycle-aware monitor without requiring one from callers."""

    if monitor is None:
        return
    start = getattr(monitor, "start", None)
    if not callable(start):
        return
    snapshot = getattr(monitor, "snapshot", None)
    if callable(snapshot):
        try:
            status = snapshot().status
        except Exception:
            status = None
        if status not in (None, "pending"):
            return
    start()


def _finish_monitor_succeeded(monitor: SolveMonitor | None) -> None:
    if monitor is None:
        return
    callback = getattr(monitor, "succeeded", None)
    if callable(callback):
        callback()


def _finish_monitor_failed(
    monitor: SolveMonitor | None,
    error: BaseException,
) -> None:
    if monitor is None:
        return
    callback = getattr(monitor, "failed", None)
    if callable(callback):
        callback(str(error))


def _finish_monitor_cancelled(monitor: SolveMonitor | None) -> None:
    if monitor is None:
        return
    callback = getattr(monitor, "cancelled", None)
    if callable(callback):
        callback()


__all__ = [
    "AnalysisConvergenceError",
    "AnalysisSolverKind",
    "analysis_request_for_step",
    "analysis_solver_kind",
    "prepare_linear_analysis",
    "selected_analysis_step",
    "solve_analysis",
]
