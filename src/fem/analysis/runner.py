"""Procedure dispatch over a fully compiled analysis."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from fem.solver import newton

from . import incremental
from .compiled import CompiledAnalysis


class AnalysisCancelled(RuntimeError):
    """Raised when cooperative cancellation reaches an analysis boundary."""


def check_cancelled(should_cancel: Callable[[], bool] | None) -> None:
    if should_cancel is None:
        return
    if not callable(should_cancel):
        raise TypeError("should_cancel must be callable or None")
    if should_cancel():
        raise AnalysisCancelled("analysis cancelled")


def run_compiled_analysis(
    compiled: CompiledAnalysis,
    *,
    monitor: Any | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> incremental.IncrementalSolveResult:
    """Run the procedure selected by one immutable compiled request."""

    if type(compiled) is not CompiledAnalysis:
        raise TypeError("compiled must be exactly CompiledAnalysis")
    check_cancelled(should_cancel)
    request = compiled.request
    if request.step.procedure != "static":
        raise NotImplementedError(
            "compiled runner currently supports static analysis only"
        )
    controls = request.controls
    try:
        return incremental.solve(
            compiled.problem,
            controls.load_factors,
            max_iterations=controls.newton_max_iterations,
            residual_tolerance=controls.residual_tolerance,
            relative_residual_tolerance=controls.relative_residual_tolerance,
            displacement_tolerance=controls.displacement_tolerance,
            energy_tolerance=controls.energy_tolerance,
            constraint_tolerance=controls.constraint_tolerance,
            tangent_strategy=controls.newton_strategy,
            line_search=controls.line_search,
            predictor=controls.predictor,
            adaptive=controls.automatic_cutback,
            adaptive_growth=controls.adaptive_growth,
            growth_factor=controls.growth_factor,
            growth_iteration_threshold=controls.growth_iteration_threshold,
            maximum_increments=controls.maximum_increments,
            minimum_increment=controls.minimum_increment,
            maximum_increment=controls.maximum_increment,
            monitor=monitor,
            should_cancel=should_cancel,
        )
    except newton.SolveCancelled as exc:
        raise AnalysisCancelled(str(exc)) from exc


__all__ = ["AnalysisCancelled", "check_cancelled", "run_compiled_analysis"]
