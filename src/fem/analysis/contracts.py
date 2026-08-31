"""Analysis-level contracts independent of elements and materials."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Protocol, runtime_checkable

from fem.model import (
    AnalysisStepSnapshot,
    DynamicStepControls,
    GeometryMode,
    StaticAnalysisOptions,
    StaticFormulation,
    StaticStepControls,
)


@dataclass(frozen=True, slots=True)
class AnalysisRequest:
    """Immutable, typed snapshot consumed by one analysis execution."""

    step: AnalysisStepSnapshot
    controls: StaticStepControls | DynamicStepControls = field(
        default_factory=StaticStepControls,
    )
    time_values: tuple[float, ...] = ()
    formulation: StaticFormulation = StaticFormulation.LINEAR
    geometry_mode: GeometryMode | None = None
    options: StaticAnalysisOptions = field(default_factory=StaticAnalysisOptions)

    def __post_init__(self) -> None:
        if type(self.step) is not AnalysisStepSnapshot:
            raise TypeError("AnalysisRequest.step must be AnalysisStepSnapshot")
        if not isinstance(self.formulation, StaticFormulation):
            raise TypeError(
                "AnalysisRequest.formulation must be StaticFormulation"
            )
        geometry_mode = self.geometry_mode
        if geometry_mode is None:
            geometry_mode = getattr(self.step, "geometry_mode", None)
        if geometry_mode is None:
            geometry_mode = GeometryMode.from_formulation(self.formulation)
        if not isinstance(geometry_mode, GeometryMode):
            raise TypeError("AnalysisRequest.geometry_mode must be GeometryMode")
        if geometry_mode.formulation is not self.formulation:
            raise ValueError(
                "AnalysisRequest geometry_mode must match formulation"
            )
        if self.step.geometry_mode is not geometry_mode:
            raise ValueError(
                "AnalysisRequest geometry_mode must match its step snapshot"
            )
        object.__setattr__(self, "geometry_mode", geometry_mode)
        procedure = self.step.procedure
        if procedure == "static" and not isinstance(self.controls, StaticStepControls):
            raise TypeError("AnalysisRequest.controls must be StaticStepControls")
        if procedure == "dynamic" and not isinstance(self.controls, DynamicStepControls):
            raise TypeError(
                "AnalysisRequest.controls must be DynamicStepControls"
            )
        if procedure not in {"static", "dynamic"}:
            raise NotImplementedError(
                f"analysis procedure {procedure!r} is not supported"
            )
        if not isinstance(self.options, StaticAnalysisOptions):
            raise TypeError(
                "AnalysisRequest.options must be StaticAnalysisOptions"
            )
        if self.step.formulation is not self.formulation:
            raise ValueError(
                "AnalysisRequest formulation must match its step snapshot"
            )
        if self.step.controls != self.controls:
            raise ValueError(
                "AnalysisRequest controls must match its step snapshot"
            )
        if procedure == "dynamic":
            # The dynamic procedure is independent from the response regime.
            # Material behavior and geometry determine linear/nonlinear during
            # execution-plan resolution.
            if not getattr(self.controls, "procedure_kind", None):
                raise ValueError("dynamic controls must select a procedure kind")
        times = tuple(float(value) for value in self.time_values)
        object.__setattr__(self, "time_values", times)

@runtime_checkable
class AnalysisRequestFactory(Protocol):
    """Boundary contract that resolves authoring data into a request."""

    def request(self, model: Any, step: Any | None = None) -> AnalysisRequest:
        """Resolve one model analysis into an immutable request."""


@dataclass(frozen=True, slots=True)
class NonlinearStaticAnalysis:
    """Static analysis procedure with typed increment/Newton controls."""

    controls: StaticStepControls = field(default_factory=StaticStepControls)

    def request(self, model: Any, step: Any | None = None) -> AnalysisRequest:
        del model
        if step is not None:
            # Resolve procedure options at the same boundary as ordinary
            # application execution, while this procedure supplies the
            # explicit controls selected by its caller.
            from .resolution import resolve_analysis_request

            resolved = resolve_analysis_request(step)
            if resolved.formulation is not StaticFormulation.NONLINEAR:
                raise ValueError(
                    "NonlinearStaticAnalysis requires a nonlinear formulation"
                )
            return AnalysisRequest(
                step=replace(resolved.step, controls=self.controls),
                formulation=resolved.formulation,
                geometry_mode=resolved.geometry_mode,
                controls=self.controls,
                options=resolved.options,
            )
        raise ValueError("a step is required to create a nonlinear analysis request")


__all__ = [
    "AnalysisRequestFactory",
    "AnalysisRequest",
    "NonlinearStaticAnalysis",
]
