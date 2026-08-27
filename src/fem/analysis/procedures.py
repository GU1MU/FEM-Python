"""Procedure-level execution contracts for analysis.

The application layer should select one procedure through this registry.  A
procedure owns its numerical preparation and execution strategy, while the
application owns task identity, cancellation, and publication lifecycle.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from enum import StrEnum
from time import perf_counter
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable

import numpy as np

from fem.analysis import incremental, linear_static
from fem.analysis.compiler import compile_analysis
from fem.analysis.contracts import AnalysisRequest
from fem.analysis.execution_plan import (
    ExecutionStrategy,
    resolve_execution_plan,
)
from fem.analysis.runner import AnalysisCancelled, run_compiled_analysis
from fem.model import (
    AnalysisStep,
    DynamicProcedureKind,
    DynamicStepControls,
    GeometryMode,
)
from fem.results import (
    ModelResult,
    ResultFrame,
    dynamic_frame_data_from_outputs,
)
from fem.solver import explicit_dynamic, implicit_dynamic, newmark, newton


class AnalysisPath(StrEnum):
    """Numerical formulation selected by a typed analysis request."""

    LINEAR_STATIC = "linear_static"
    NONLINEAR_STATIC = "nonlinear_static"
    DYNAMIC_IMPLICIT = "dynamic_implicit"
    LINEAR_DYNAMIC = "dynamic_implicit"
    DYNAMIC_EXPLICIT = "dynamic_explicit"


@dataclass(frozen=True, slots=True)
class ProcedureContext:
    """Runtime callbacks shared by all procedure implementations."""

    monitor: Any | None = None
    should_cancel: Callable[[], bool] | None = None
    timings: dict[str, float] | None = None


@runtime_checkable
class PreparedAnalysis(Protocol):
    """Opaque preparation owned by one analysis procedure."""

    def clone(self) -> "PreparedAnalysis":
        """Return an isolated task-owned copy."""

    def model_for_task(self) -> Any:
        """Return the preparation's detached task model."""

    def shares_cache_with(self, other: "PreparedAnalysis") -> bool:
        """Return whether two preparations share immutable cached data."""


@runtime_checkable
class AnalysisProcedure(Protocol):
    """One complete validate/prepare/run/materialize analysis procedure."""

    path: AnalysisPath

    def validate(self, model: Any, request: AnalysisRequest) -> None:
        """Validate procedure-specific entry conditions."""

    def prepare(
        self,
        model: Any,
        request: AnalysisRequest,
        *,
        timings: dict[str, float] | None = None,
    ) -> Any:
        """Build one procedure-owned prepared or compiled artifact."""

    def run(
        self,
        model: Any,
        step: AnalysisStep,
        request: AnalysisRequest,
        prepared: Any,
        *,
        name: str | None,
        context: ProcedureContext,
    ) -> Any:
        """Execute the prepared numerical path."""

    def materialize(
        self,
        model: Any,
        step: AnalysisStep,
        numerical_result: Any,
        *,
        name: str | None,
    ) -> ModelResult:
        """Project one detached numerical result into the public result type."""


@runtime_checkable
class AnalysisExecutor(Protocol):
    """Application-neutral executor for the common procedure lifecycle."""

    def prepare(
        self,
        model: Any,
        request: AnalysisRequest,
        *,
        prepared: Any | None = None,
        timings: dict[str, float] | None = None,
    ) -> Any:
        """Validate a request and return its procedure-owned preparation."""

    def run(
        self,
        model: Any,
        step: Any,
        request: AnalysisRequest,
        prepared: Any,
        *,
        name: str | None,
        context: ProcedureContext,
    ) -> Any:
        """Run one already-prepared procedure."""

    def validate_prepared(
        self,
        model: Any,
        step: Any,
        request: AnalysisRequest,
        prepared: Any,
    ) -> None:
        """Run procedure-specific numerical stability checks."""

    def materialize(
        self,
        model: Any,
        step: Any,
        request: AnalysisRequest,
        numerical_result: Any,
        *,
        name: str | None,
        prepared: Any | None = None,
    ) -> ModelResult:
        """Project a numerical result into the common public result contract."""


class LinearStaticProcedure:
    """Optimized linear static procedure behind the common protocol."""

    path = AnalysisPath.LINEAR_STATIC

    def validate(self, model: Any, request: AnalysisRequest) -> None:
        if (
            request.step.procedure != "static"
            or request.geometry_mode is not GeometryMode.SMALL_STRAIN
            or resolve_execution_plan(model, request).strategy
            is not ExecutionStrategy.DIRECT_LINEAR
        ):
            raise ValueError(
                "direct linear static procedure requires small-strain linear elasticity"
            )

    def prepare(
        self,
        model: Any,
        request: AnalysisRequest,
        *,
        timings: dict[str, float] | None = None,
    ) -> Any:
        self.validate(model, request)
        return linear_static.prepare(
            model,
            copy_model=False,
            timings=timings,
        )

    def validate_prepared(self, prepared: Any, step: Any) -> None:
        """Check constrained stiffness through the linear procedure boundary."""

        if not isinstance(prepared, PreparedAnalysis):
            raise TypeError("linear preparation must implement PreparedAnalysis")
        validator = getattr(prepared, "validate_stiffness", None)
        if not callable(validator):
            raise TypeError("linear preparation cannot validate stiffness")
        validator(step)

    def run(
        self,
        model: Any,
        step: AnalysisStep,
        request: AnalysisRequest,
        prepared: Any,
        *,
        name: str | None,
        context: ProcedureContext,
    ) -> ModelResult:
        self.validate(model, request)
        monitor = context.monitor
        should_cancel = context.should_cancel
        _check_cancelled(should_cancel)
        started = perf_counter()
        if monitor is not None:
            monitor.increment_started(1, 1, 1.0, 0.0)
        try:
            solved = linear_static.solve(
                model,
                request.step,
                name,
                _prepared_system=prepared,
                timings=context.timings,
            )
        except BaseException as error:
            if monitor is not None:
                record = _linear_attempt_record(
                    started,
                    status="failed",
                    error=str(error),
                )
                monitor.increment_failed(
                    1,
                    1,
                    None,
                    str(error),
                    retryable=False,
                    record=record,
                )
            raise
        duration = max(0.0, perf_counter() - started)
        _check_cancelled(should_cancel)
        record = _linear_attempt_record(
            started,
            status="converged",
            duration=duration,
        )
        if monitor is not None:
            monitor.increment_converged(
                1,
                1,
                1.0,
                1,
                0.0,
                record=record,
            )
        outputs = dict(solved.outputs)
        outputs.update(
            {
                "step": step,
                "increment_record": record,
                "increment_number": 1,
                "attempt_number": 1,
                "previous_load_factor": 0.0,
                "load_factor_increment": 1.0,
                "increment_duration_seconds": duration,
                "increment_status": "converged",
                "cutbacks": 0,
                "increment_attempts": (record,),
                "attempt_count": 1,
                "cutback_count": 0,
            }
        )
        frame = ResultFrame(
            model=model,
            step=step,
            U=solved.U,
            reactions=solved.reactions,
            frame_index=1,
            load_factor=1.0,
            name=solved.name,
            outputs=outputs,
            iterations=1,
            residual_norm=0.0,
        )
        return ModelResult(
            model=model,
            step=step,
            U=solved.U,
            reactions=solved.reactions,
            name=solved.name,
            outputs=outputs,
            load_factor=1.0,
            iterations=1,
            residual_norm=0.0,
            frames=(frame,),
        )

    def materialize(
        self,
        model: Any,
        step: AnalysisStep,
        numerical_result: Any,
        *,
        name: str | None,
    ) -> ModelResult:
        if type(numerical_result) is not ModelResult:
            raise TypeError("linear procedure must return ModelResult")
        return _rebind_result_identity(
            numerical_result,
            model=model,
            step=step,
            name=name,
        )


class ImplicitDynamicProcedure:
    """Abaqus-style direct-integration implicit dynamics."""

    path = AnalysisPath.DYNAMIC_IMPLICIT

    def validate(self, model: Any, request: AnalysisRequest) -> None:
        if request.step.procedure != "dynamic":
            raise ValueError("implicit dynamics requires a dynamic step")
        if not isinstance(request.controls, DynamicStepControls):
            raise TypeError("implicit dynamics requires DynamicStepControls")
        if request.controls.procedure_kind is not DynamicProcedureKind.IMPLICIT:
            raise ValueError("implicit procedure received explicit controls")
        plan = resolve_execution_plan(model, request)
        if plan.strategy not in {
            ExecutionStrategy.IMPLICIT_DYNAMIC_LINEAR,
            ExecutionStrategy.IMPLICIT_DYNAMIC_NONLINEAR,
        }:
            raise ValueError("analysis plan is not an implicit dynamic plan")

    def prepare(
        self,
        model: Any,
        request: AnalysisRequest,
        *,
        timings: dict[str, float] | None = None,
    ) -> Any:
        self.validate(model, request)
        started = perf_counter()
        prepared = compile_analysis(model, request)
        if timings is not None:
            timings["隐式动力学问题准备"] = timings.get(
                "隐式动力学问题准备",
                0.0,
            ) + (perf_counter() - started)
        return prepared

    def run(
        self,
        model: Any,
        step: AnalysisStep,
        request: AnalysisRequest,
        prepared: Any,
        *,
        name: str | None,
        context: ProcedureContext,
    ) -> newmark.TransientSolveResult:
        del step, name
        self.validate(model, request)
        if not hasattr(prepared, "problem"):
            raise TypeError("implicit dynamic preparation must contain a problem")
        controls = request.controls
        plan = resolve_execution_plan(model, request)
        try:
            if plan.strategy is ExecutionStrategy.IMPLICIT_DYNAMIC_LINEAR:
                return newmark.solve(
                    prepared.problem,
                    controls.time_grid,
                    beta=controls.beta,
                    gamma=controls.gamma,
                    monitor=context.monitor,
                    should_cancel=context.should_cancel,
                )
            return implicit_dynamic.solve(
                prepared.problem,
                controls.time_grid,
                beta=controls.beta,
                gamma=controls.gamma,
                monitor=context.monitor,
                should_cancel=context.should_cancel,
            )
        except (newmark.DynamicSolveCancelled, newton.SolveCancelled) as error:
            raise AnalysisCancelled(str(error)) from error

    def materialize(
        self,
        model: Any,
        step: AnalysisStep,
        numerical_result: Any,
        *,
        name: str | None,
    ) -> ModelResult:
        return _materialize_dynamic_result(
            model,
            step,
            numerical_result,
            name=name,
        )


class ExplicitDynamicProcedure:
    """Abaqus-style central-difference explicit dynamics."""

    path = AnalysisPath.DYNAMIC_EXPLICIT

    def validate(self, model: Any, request: AnalysisRequest) -> None:
        if request.step.procedure != "dynamic":
            raise ValueError("explicit dynamics requires a dynamic step")
        if not isinstance(request.controls, DynamicStepControls):
            raise TypeError("explicit dynamics requires DynamicStepControls")
        if request.controls.procedure_kind is not DynamicProcedureKind.EXPLICIT:
            raise ValueError("explicit procedure received implicit controls")
        plan = resolve_execution_plan(model, request)
        if plan.strategy not in {
            ExecutionStrategy.EXPLICIT_DYNAMIC_LINEAR,
            ExecutionStrategy.EXPLICIT_DYNAMIC_NONLINEAR,
        }:
            raise ValueError("analysis plan is not an explicit dynamic plan")

    def prepare(
        self,
        model: Any,
        request: AnalysisRequest,
        *,
        timings: dict[str, float] | None = None,
    ) -> Any:
        self.validate(model, request)
        started = perf_counter()
        prepared = compile_analysis(model, request)
        if timings is not None:
            timings["显式动力学问题准备"] = timings.get(
                "显式动力学问题准备",
                0.0,
            ) + (perf_counter() - started)
        return prepared

    def run(
        self,
        model: Any,
        step: AnalysisStep,
        request: AnalysisRequest,
        prepared: Any,
        *,
        name: str | None,
        context: ProcedureContext,
    ) -> newmark.TransientSolveResult:
        del step, name
        self.validate(model, request)
        if not hasattr(prepared, "problem"):
            raise TypeError("explicit dynamic preparation must contain a problem")
        controls = request.controls
        try:
            return explicit_dynamic.solve(
                prepared.problem,
                time_period=controls.time_period,
                initial_time_increment=controls.initial_time_increment,
                minimum_time_increment=controls.minimum_time_increment,
                maximum_time_increment=controls.maximum_time_increment,
                maximum_increments=controls.maximum_increments,
                monitor=context.monitor,
                should_cancel=context.should_cancel,
            )
        except newmark.DynamicSolveCancelled as error:
            raise AnalysisCancelled(str(error)) from error

    def materialize(
        self,
        model: Any,
        step: AnalysisStep,
        numerical_result: Any,
        *,
        name: str | None,
    ) -> ModelResult:
        return _materialize_dynamic_result(
            model,
            step,
            numerical_result,
            name=name,
        )


def _materialize_dynamic_result(
    model: Any,
    step: AnalysisStep,
    numerical_result: Any,
    *,
    name: str | None,
) -> ModelResult:
    if not isinstance(numerical_result, newmark.TransientSolveResult):
        raise TypeError("dynamic procedure must return TransientSolveResult")
    controls = getattr(step, "controls", None)
    if not isinstance(controls, DynamicStepControls):
        controls = DynamicStepControls.from_metadata(getattr(step, "metadata", {}))
    solver_kind = (
        "dynamic_explicit"
        if controls.procedure_kind is DynamicProcedureKind.EXPLICIT
        else "dynamic_implicit"
    )
    frames: list[ResultFrame] = []
    for index, frame in enumerate(numerical_result.frames, start=1):
        outputs = dict(frame.outputs)
        outputs["step"] = step
        outputs["load_factor"] = controls.amplitude.value_at(frame.time)
        dynamic_data = dynamic_frame_data_from_outputs(
            time=frame.time,
            time_increment=frame.time_increment,
            outputs=outputs,
            residual_norm=frame.residual_norm,
            iterations=int(outputs.get("newton_iterations", 0)),
            solver_kind=solver_kind,
            amplitude=float(outputs["load_factor"]),
        )
        frames.append(
            ResultFrame(
                model=model,
                step=step,
                U=frame.solution.values,
                reactions=frame.reactions,
                frame_index=index,
                load_factor=float(outputs["load_factor"]),
                name=name,
                outputs=outputs,
                iterations=int(outputs.get("newton_iterations", 0)),
                residual_norm=frame.residual_norm,
                dynamic_data=dynamic_data,
            )
        )
    final = frames[-1]
    outputs = dict(final.outputs)
    outputs.update(
        {
            "step": step,
            "frames": tuple(frames),
            "time_period": numerical_result.final.time,
            "frame_count": len(frames),
        }
    )
    return ModelResult(
        model=model,
        step=step,
        U=final.U,
        reactions=final.reactions,
        name=name,
        outputs=outputs,
        load_factor=final.load_factor,
        iterations=final.iterations,
        residual_norm=final.residual_norm,
        frames=tuple(frames),
        dynamic_data=final.dynamic_data,
    )


class NonlinearStaticProcedure:
    """Incremental nonlinear static procedure behind the common protocol."""

    path = AnalysisPath.NONLINEAR_STATIC

    def validate(self, model: Any, request: AnalysisRequest) -> None:
        if (
            request.step.procedure != "static"
            or resolve_execution_plan(model, request).strategy
            is not ExecutionStrategy.INCREMENTAL_NEWTON
        ):
            raise ValueError(
                "incremental static procedure requires material or geometric nonlinearity"
            )

    def prepare(
        self,
        model: Any,
        request: AnalysisRequest,
        *,
        timings: dict[str, float] | None = None,
    ) -> Any:
        self.validate(model, request)
        started = perf_counter()
        prepared = compile_analysis(model, request)
        if timings is not None:
            timings["非线性问题准备"] = timings.get("非线性问题准备", 0.0) + (
                perf_counter() - started
            )
        return prepared

    def run(
        self,
        model: Any,
        step: AnalysisStep,
        request: AnalysisRequest,
        prepared: Any,
        *,
        name: str | None,
        context: ProcedureContext,
    ) -> incremental.IncrementalSolveResult:
        del step, name
        self.validate(model, request)
        return run_compiled_analysis(
            prepared,
            monitor=context.monitor,
            should_cancel=context.should_cancel,
        )

    def materialize(
        self,
        model: Any,
        step: AnalysisStep,
        numerical_result: Any,
        *,
        name: str | None,
    ) -> ModelResult:
        if type(numerical_result) is not incremental.IncrementalSolveResult:
            raise TypeError(
                "nonlinear procedure must return IncrementalSolveResult"
            )
        return materialize_incremental_result(
            model,
            step,
            numerical_result,
            name=name,
        )


class AnalysisProcedureRegistry:
    """Immutable lookup table for the application's analysis procedures."""

    def __init__(self, procedures: tuple[AnalysisProcedure, ...]):
        entries = tuple(procedures)
        if not entries:
            raise ValueError("analysis procedure registry must not be empty")
        by_path: dict[AnalysisPath, AnalysisProcedure] = {}
        for procedure in entries:
            if not isinstance(procedure.path, AnalysisPath):
                raise TypeError("procedure path must be an AnalysisPath")
            if procedure.path in by_path:
                raise ValueError(f"duplicate analysis procedure {procedure.path}")
            by_path[procedure.path] = procedure
        self._procedures = entries
        self._by_path = MappingProxyType(by_path)

    def resolve(self, path: AnalysisPath) -> AnalysisProcedure:
        if not isinstance(path, AnalysisPath):
            raise TypeError("analysis path must be an AnalysisPath")
        try:
            return self._by_path[path]
        except KeyError as exc:
            raise NotImplementedError(
                f"analysis path {path.value!r} is not registered"
            ) from exc


class RegisteredAnalysisExecutor:
    """Run every registered analysis through one lifecycle implementation."""

    def __init__(self, registry: AnalysisProcedureRegistry):
        if not isinstance(registry, AnalysisProcedureRegistry):
            raise TypeError(
                "registry must be an AnalysisProcedureRegistry"
            )
        self._registry = registry

    def procedure_for(
        self,
        request: AnalysisRequest,
        *,
        model: Any | None = None,
    ) -> AnalysisProcedure:
        """Resolve the only procedure allowed to execute a request."""

        return self._registry.resolve(analysis_path_for(model, request))

    def prepare(
        self,
        model: Any,
        request: AnalysisRequest,
        *,
        prepared: Any | None = None,
        timings: dict[str, float] | None = None,
    ) -> Any:
        procedure = self.procedure_for(request, model=model)
        procedure.validate(model, request)
        if prepared is not None:
            path = analysis_path_for(model, request)
            if path is AnalysisPath.LINEAR_STATIC:
                if not isinstance(prepared, PreparedAnalysis):
                    raise TypeError(
                        "injected linear preparation must implement PreparedAnalysis"
                    )
            elif path in {
                AnalysisPath.DYNAMIC_IMPLICIT,
                AnalysisPath.DYNAMIC_EXPLICIT,
            }:
                if not hasattr(prepared, "problem"):
                    raise TypeError(
                        "dynamic preparation must expose a compiled problem"
                    )
            else:
                raise ValueError("the supplied preparation is not valid for this analysis path")
            return prepared
        return procedure.prepare(model, request, timings=timings)

    def run(
        self,
        model: Any,
        step: Any,
        request: AnalysisRequest,
        prepared: Any,
        *,
        name: str | None,
        context: ProcedureContext,
    ) -> Any:
        procedure = self.procedure_for(request, model=model)
        procedure.validate(model, request)
        return procedure.run(
            model,
            step,
            request,
            prepared,
            name=name,
            context=context,
        )

    def validate_prepared(
        self,
        model: Any,
        step: Any,
        request: AnalysisRequest,
        prepared: Any,
    ) -> None:
        """Validate a preparation without exposing a concrete solver type."""

        procedure = self.procedure_for(request, model=model)
        validator = getattr(procedure, "validate_prepared", None)
        if not callable(validator):
            return
        validator(prepared, step)

    def materialize(
        self,
        model: Any,
        step: Any,
        request: AnalysisRequest,
        numerical_result: Any,
        *,
        name: str | None,
        prepared: Any | None = None,
    ) -> ModelResult:
        procedure = self.procedure_for(request, model=model)
        result = procedure.materialize(
            model,
            step,
            numerical_result,
            name=name,
        )
        if prepared is None:
            return result
        model_for_result = getattr(prepared, "model_for_result", None)
        if not callable(model_for_result):
            return result
        compiled_model = model_for_result()
        if compiled_model is None or compiled_model is result.model:
            return result
        frames = tuple(
            replace(frame, compiled_model=compiled_model)
            for frame in result.frames
        )
        return replace(
            result,
            frames=frames,
            compiled_model=compiled_model,
        )

    def execute(
        self,
        model: Any,
        step: Any,
        request: AnalysisRequest,
        *,
        name: str | None = None,
        prepared: Any | None = None,
        monitor: Any | None = None,
        should_cancel: Callable[[], bool] | None = None,
        timings: dict[str, float] | None = None,
    ) -> ModelResult:
        """Execute validate → prepare → run → materialize as one operation."""

        preparation = self.prepare(
            model,
            request,
            prepared=prepared,
            timings=timings,
        )
        numerical_result = self.run(
            model,
            step,
            request,
            preparation,
            name=name,
            context=ProcedureContext(
                monitor=monitor,
                should_cancel=should_cancel,
                timings=timings,
            ),
        )
        return self.materialize(
            model,
            step,
            request,
            numerical_result,
            name=name,
            prepared=preparation,
        )


LinearDynamicProcedure = ImplicitDynamicProcedure


DEFAULT_ANALYSIS_PROCEDURES = AnalysisProcedureRegistry(
    (
        LinearStaticProcedure(),
        NonlinearStaticProcedure(),
        ImplicitDynamicProcedure(),
        ExplicitDynamicProcedure(),
    )
)

DEFAULT_ANALYSIS_EXECUTOR = RegisteredAnalysisExecutor(
    DEFAULT_ANALYSIS_PROCEDURES
)


def analysis_path_for(
    model: Any,
    request: AnalysisRequest | None = None,
) -> AnalysisPath:
    """Select a procedure from the complete geometry/material execution plan."""

    if request is None:
        # Keep the request-only inspection form useful for callers that do not
        # own an authoring model. Real execution always passes both values so
        # constitutive behavior participates in dispatch.
        request = model
        model = None
    if not isinstance(request, AnalysisRequest):
        raise TypeError("request must be an AnalysisRequest")
    if request.step.procedure == "dynamic":
        procedure_kind = request.controls.procedure_kind
        return (
            AnalysisPath.DYNAMIC_EXPLICIT
            if procedure_kind is DynamicProcedureKind.EXPLICIT
            else AnalysisPath.DYNAMIC_IMPLICIT
        )
    if model is None:
        # A request-only inspection is retained for callers that only need
        # the geometry projection.  Executable dispatch always supplies the
        # model so material behavior can participate in planning.
        return (
            AnalysisPath.NONLINEAR_STATIC
            if request.geometry_mode is GeometryMode.FINITE_STRAIN
            else AnalysisPath.LINEAR_STATIC
        )
    return (
        AnalysisPath.NONLINEAR_STATIC
        if resolve_execution_plan(model, request).strategy
        is ExecutionStrategy.INCREMENTAL_NEWTON
        else AnalysisPath.LINEAR_STATIC
    )


def materialize_incremental_result(
    model: Any,
    step: AnalysisStep,
    solved: incremental.IncrementalSolveResult,
    *,
    name: str | None,
) -> ModelResult:
    """Project detached incremental frames into the public result contract."""

    if not solved.increments:
        raise ValueError("incremental solve returned no converged increments")
    frames: list[ResultFrame] = []
    for index, increment in enumerate(solved.increments, start=1):
        outputs = dict(increment.outputs)
        record = increment.record
        if increment.local_outputs:
            outputs["local_output_batches"] = tuple(increment.local_outputs)
        if record is not None:
            cutbacks = sum(
                attempt.status == "cutback"
                and attempt.increment == record.increment
                and attempt.attempt < record.attempt
                for attempt in solved.attempts
            )
            outputs.update(
                {
                    "increment_record": record,
                    "increment_number": record.increment,
                    "attempt_number": record.attempt,
                    "previous_load_factor": record.previous_load_factor,
                    "load_factor_increment": (
                        record.target_load_factor
                        - record.previous_load_factor
                    ),
                    "increment_duration_seconds": record.duration_seconds,
                    "increment_status": record.status,
                    "cutbacks": cutbacks,
                }
            )
        outputs["step"] = step
        physical_residual = outputs.get("physical_residual")
        reactions = (
            np.zeros(int(model.mesh.num_dofs), dtype=float)
            if physical_residual is None
            else np.asarray(physical_residual, dtype=float)
        )
        frames.append(
            ResultFrame(
                model=model,
                step=step,
                U=increment.solution.field("U"),
                reactions=reactions,
                frame_index=index,
                load_factor=increment.load_factor,
                name=name,
                outputs=outputs,
                iterations=increment.newton.iterations,
                residual_norm=increment.newton.residual_norm,
            )
        )
    final = frames[-1]
    final_outputs = dict(final.outputs)
    final_outputs.update(
        {
            "increment_attempts": tuple(solved.attempts),
            "attempt_count": len(solved.attempts),
            "cutback_count": sum(
                attempt.status == "cutback" for attempt in solved.attempts
            ),
        }
    )
    return ModelResult(
        model=model,
        step=step,
        U=final.U,
        reactions=final.reactions,
        name=name,
        outputs=final_outputs,
        load_factor=final.load_factor,
        iterations=final.iterations,
        residual_norm=final.residual_norm,
        frames=tuple(frames),
    )


def _rebind_result_identity(
    result: ModelResult,
    *,
    model: Any,
    step: Any,
    name: str | None,
) -> ModelResult:
    """Attach public authoring identity only at the result boundary."""

    outputs = dict(result.outputs)
    outputs["step"] = step
    frames = tuple(
        ResultFrame(
            model=model,
            step=step,
            U=frame.U,
            reactions=frame.reactions,
            frame_index=frame.frame_index,
            load_factor=frame.load_factor,
            name=frame.name,
            outputs={**dict(frame.outputs), "step": step},
            iterations=frame.iterations,
            residual_norm=frame.residual_norm,
            converged=frame.converged,
            dynamic_data=frame.dynamic_data,
        )
        for frame in result.frames
    )
    return ModelResult(
        model=model,
        step=step,
        U=result.U,
        reactions=result.reactions,
        name=name if name is not None else result.name,
        outputs=outputs,
        load_factor=result.load_factor,
        iterations=result.iterations,
        residual_norm=result.residual_norm,
        frames=frames,
        dynamic_data=result.dynamic_data,
    )


def _linear_attempt_record(
    started: float,
    *,
    status: str,
    duration: float | None = None,
    error: str | None = None,
) -> incremental.IncrementAttemptRecord:
    return incremental.IncrementAttemptRecord(
        increment=1,
        attempt=1,
        previous_load_factor=0.0,
        target_load_factor=1.0,
        status=status,
        load_factor=1.0 if status == "converged" else None,
        iterations=1 if status == "converged" else 0,
        residual_norm=0.0 if status == "converged" else None,
        duration_seconds=(
            max(0.0, perf_counter() - started)
            if duration is None
            else max(0.0, duration)
        ),
        error=error,
    )


def _check_cancelled(callback: Callable[[], bool] | None) -> None:
    if callback is not None and callback():
        raise AnalysisCancelled("analysis cancelled")


__all__ = [
    "AnalysisPath",
    "AnalysisExecutor",
    "AnalysisProcedure",
    "AnalysisProcedureRegistry",
    "RegisteredAnalysisExecutor",
    "DEFAULT_ANALYSIS_EXECUTOR",
    "DEFAULT_ANALYSIS_PROCEDURES",
    "LinearStaticProcedure",
    "LinearDynamicProcedure",
    "ImplicitDynamicProcedure",
    "ExplicitDynamicProcedure",
    "NonlinearStaticProcedure",
    "PreparedAnalysis",
    "ProcedureContext",
    "analysis_path_for",
    "materialize_incremental_result",
]
