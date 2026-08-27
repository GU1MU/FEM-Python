"""Typed execution planning for static analysis.

The plan is the boundary where geometry and constitutive behavior meet.  A
step selects the kinematic model; the compiled material table selects the
constitutive behavior.  The numerical strategy is then derived from both
facts, never from the historical ``NLGEOM`` spelling alone.
"""

from __future__ import annotations

from contextvars import ContextVar
from contextlib import ContextDecorator
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from fem.model import DynamicProcedureKind, GeometryMode

from .compilation.materials import (
    CompiledMaterialAssignments,
    compile_material_assignments,
)
from .contracts import AnalysisRequest


class ExecutionStrategy(StrEnum):
    """Numerical driver selected after procedure and response resolution."""

    DIRECT_LINEAR = "direct_linear"
    INCREMENTAL_NEWTON = "incremental_newton"
    IMPLICIT_DYNAMIC_LINEAR = "linear_dynamic_newmark"
    LINEAR_DYNAMIC_NEWMARK = "linear_dynamic_newmark"
    IMPLICIT_DYNAMIC_NONLINEAR = "nonlinear_dynamic_newmark"
    EXPLICIT_DYNAMIC_LINEAR = "linear_dynamic_central_difference"
    EXPLICIT_DYNAMIC_NONLINEAR = "nonlinear_dynamic_central_difference"


@dataclass(frozen=True, slots=True)
class AnalysisExecutionPlan:
    """Immutable decision and compiled material table for one execution."""

    geometry_mode: GeometryMode
    strategy: ExecutionStrategy
    material_models: tuple[str, ...]
    material_algorithms: tuple[str, ...]
    material_assignments: CompiledMaterialAssignments

    @property
    def requires_incremental_driver(self) -> bool:
        return self.strategy in {
            ExecutionStrategy.INCREMENTAL_NEWTON,
            ExecutionStrategy.IMPLICIT_DYNAMIC_NONLINEAR,
        }

    @property
    def has_material_history(self) -> bool:
        return any(model != "linear_elastic" for model in self.material_models)


class execution_plan_cache_scope(ContextDecorator):
    """Memoize one request's plan for the duration of one analysis operation.

    Plan resolution is pure with respect to the authored model, but it is
    intentionally not a process-wide cache: models and their element
    properties are mutable at the application boundary.  A short-lived
    context avoids compiling the same material/section table repeatedly
    inside one preflight or solve while keeping mutation invalidation safe.
    """

    _current: ContextVar[dict[tuple[int, str], AnalysisExecutionPlan] | None] = (
        ContextVar("fem_execution_plan_cache", default=None)
    )

    def _recreate_cm(self) -> "execution_plan_cache_scope":
        """Give ``ContextDecorator`` each invocation its own token holder."""

        # ``ContextDecorator`` may execute the same decorated function in
        # more than one worker at once.  Keeping the reset token on the
        # decorator object itself would make those calls race during exit.
        return type(self)()

    def __enter__(self) -> "execution_plan_cache_scope":
        current = self._current.get()
        self._token = None
        if current is None:
            self._token = self._current.set({})
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        if self._token is not None:
            self._current.reset(self._token)
        return None


def resolve_execution_plan(
    model: Any,
    request: AnalysisRequest,
    *,
    material_assignments: CompiledMaterialAssignments | None = None,
) -> AnalysisExecutionPlan:
    """Resolve geometry, material behavior, and solver strategy together."""

    if type(request) is not AnalysisRequest:
        raise TypeError("request must be exactly AnalysisRequest")
    cache = execution_plan_cache_scope._current.get()
    # Resolution creates equivalent request snapshots at a few boundaries
    # (preflight, validation and execution).  Object identity therefore
    # misses the useful short-lived cache.  ``repr`` is deliberately scoped
    # to this operation and avoids making mutable model state hashable.
    cache_key = (id(model), repr(request))
    if material_assignments is None and cache is not None:
        cached = cache.get(cache_key)
        if cached is not None:
            return cached
    assignments = (
        compile_material_assignments(model)
        if material_assignments is None
        else material_assignments
    )
    models = tuple(sorted({item.constitutive_model for item in assignments.assignments}))
    algorithms = tuple(
        sorted(
            {
                str(item.algorithm).strip().casefold()
                for item in assignments.assignments
                if item.algorithm is not None
                and str(item.algorithm).strip()
            }
        )
    )
    if not models:
        raise ValueError("analysis requires at least one material model")
    if request.step.procedure == "dynamic":
        controls = request.controls
        procedure_kind = getattr(
            controls,
            "procedure_kind",
            DynamicProcedureKind.IMPLICIT,
        )
        if not isinstance(procedure_kind, DynamicProcedureKind):
            procedure_kind = DynamicProcedureKind(str(procedure_kind).casefold())
        nonlinear = (
            request.geometry_mode is GeometryMode.FINITE_STRAIN
            or models != ("linear_elastic",)
        )
        if procedure_kind is DynamicProcedureKind.IMPLICIT:
            strategy = (
                ExecutionStrategy.IMPLICIT_DYNAMIC_NONLINEAR
                if nonlinear
                else ExecutionStrategy.IMPLICIT_DYNAMIC_LINEAR
            )
        else:
            strategy = (
                ExecutionStrategy.EXPLICIT_DYNAMIC_NONLINEAR
                if nonlinear
                else ExecutionStrategy.EXPLICIT_DYNAMIC_LINEAR
            )
        plan = AnalysisExecutionPlan(
            geometry_mode=request.geometry_mode,
            strategy=strategy,
            material_models=models,
            material_algorithms=algorithms,
            material_assignments=assignments,
        )
        if cache is not None and material_assignments is None:
            cache[cache_key] = plan
        return plan
    if request.step.procedure != "static":
        raise NotImplementedError(
            f"analysis procedure {request.step.procedure!r} is not planned yet"
        )
    strategy = (
        ExecutionStrategy.DIRECT_LINEAR
        if request.geometry_mode is GeometryMode.SMALL_STRAIN
        and models == ("linear_elastic",)
        else ExecutionStrategy.INCREMENTAL_NEWTON
    )
    plan = AnalysisExecutionPlan(
        geometry_mode=request.geometry_mode,
        strategy=strategy,
        material_models=models,
        material_algorithms=algorithms,
        material_assignments=assignments,
    )
    if cache is not None and material_assignments is None:
        cache[cache_key] = plan
    return plan


__all__ = [
    "AnalysisExecutionPlan",
    "ExecutionStrategy",
    "execution_plan_cache_scope",
    "resolve_execution_plan",
]
