"""Resolve persisted analysis-step data into execution contracts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any

from fem.model import (
    AnalysisStep,
    AnalysisStepSnapshot,
    DynamicStepControls,
    GeometryMode,
    InitialConditionSet,
    StaticAnalysisOptions,
    StaticFormulation,
    StaticStepControls,
)
from .contracts import AnalysisRequest


def resolve_analysis_request(
    step: Any,
    *,
    max_iterations: int | None = None,
    residual_tolerance: float | None = None,
) -> AnalysisRequest:
    """Create the sole typed runtime snapshot for one static step.

    Project metadata is decoded here exactly once. Numerical solvers and
    Problems receive the resulting value object and never interpret metadata
    keys themselves.
    """

    procedure = str(getattr(step, "procedure", "static")).strip().casefold()
    if procedure == "dynamic":
        typed_controls = getattr(step, "controls", None)
        if typed_controls is None:
            controls = DynamicStepControls.from_metadata(
                getattr(step, "metadata", {})
            )
        elif not isinstance(typed_controls, DynamicStepControls):
            raise TypeError(
                "dynamic analysis step controls must be DynamicStepControls"
            )
        else:
            controls = typed_controls
        geometry_mode = resolve_geometry_mode(step)
        formulation = geometry_mode.formulation
        initial_conditions = getattr(
            step,
            "initial_conditions",
            InitialConditionSet(),
        )
        metadata = getattr(step, "metadata", {})
        if initial_conditions.is_empty and isinstance(metadata, Mapping):
            persisted = metadata.get("initial_conditions")
            if persisted is not None:
                initial_conditions = InitialConditionSet.from_metadata(persisted)
        return AnalysisRequest(
            step=AnalysisStepSnapshot.from_step(
                step,
                controls=controls,
                formulation=formulation,
                geometry_mode=geometry_mode,
                initial_conditions=initial_conditions,
            ),
            formulation=formulation,
            geometry_mode=geometry_mode,
            controls=controls,
            options=_resolve_analysis_options(
                step,
                getattr(step, "metadata", {}),
            ),
        )
    typed_controls = getattr(step, "controls", None)
    if typed_controls is not None and not isinstance(
        typed_controls,
        StaticStepControls,
    ):
        raise TypeError("analysis step controls must be StaticStepControls")
    metadata = getattr(step, "metadata", {})
    if typed_controls is None:
        controls = StaticStepControls.from_metadata(metadata)
    else:
        # The typed authoring value is the single source of truth.  Metadata
        # is only a persistence mirror for the current project boundary.
        controls = typed_controls
    if max_iterations is not None:
        controls = replace(controls, newton_max_iterations=max_iterations)
    if residual_tolerance is not None:
        controls = replace(controls, residual_tolerance=residual_tolerance)
    geometry_mode = resolve_geometry_mode(step)
    formulation = geometry_mode.formulation
    return AnalysisRequest(
        step=AnalysisStepSnapshot.from_step(
            step,
            controls=controls,
            formulation=formulation,
            geometry_mode=geometry_mode,
        ),
        formulation=formulation,
        geometry_mode=geometry_mode,
        controls=controls,
        options=_resolve_analysis_options(step, metadata),
    )


def resolve_formulation(step: Any) -> StaticFormulation:
    """Return the persistence-facing projection of the geometry mode."""

    return resolve_geometry_mode(step).formulation


def resolve_geometry_mode(step: Any) -> GeometryMode:
    """Resolve geometry independently from constitutive material behavior."""

    typed = getattr(step, "geometry_mode", None)
    if typed is not None:
        if not isinstance(typed, GeometryMode):
            raise TypeError("analysis step geometry_mode must be GeometryMode")
        return typed

    typed = getattr(step, "formulation", None)
    if typed is not None:
        if not isinstance(typed, StaticFormulation):
            raise TypeError("analysis step formulation must be StaticFormulation")
        return GeometryMode.from_formulation(typed)
    return (
        GeometryMode.FINITE_STRAIN
        if _resolve_persisted_nlgeom(step)
        else GeometryMode.SMALL_STRAIN
    )


def resolve_nlgeom(step: Any) -> bool:
    """Return the nonlinear formulation flag at the persistence boundary."""

    return resolve_geometry_mode(step) is GeometryMode.FINITE_STRAIN


def _resolve_persisted_nlgeom(step: Any) -> bool:
    """Decode the current project-file spelling only at this boundary."""

    metadata = getattr(step, "metadata", {})
    if metadata is None:
        return False
    if not hasattr(metadata, "items"):
        raise TypeError("analysis step metadata must be a mapping")
    value = next(
        (
            candidate
            for key, candidate in reversed(tuple(metadata.items()))
            if str(key).strip().casefold() == "nlgeom"
        ),
        None,
    )
    return _truthy_option(value)


def is_nlgeom_enabled(step: AnalysisStep | None) -> bool:
    """Return whether one authored step enables geometric nonlinearity."""

    return False if step is None else resolve_nlgeom(step)


def require_nlgeom(step: AnalysisStep | None, owner: str) -> AnalysisStep:
    """Require a static step with geometric nonlinearity enabled."""

    if step is None:
        raise ValueError(f"{owner} requires an analysis step with nlgeom enabled")
    if str(step.procedure).strip().casefold() != "static":
        raise ValueError(
            f"{owner} requires procedure 'static', got {step.procedure!r}"
        )
    if not resolve_nlgeom(step):
        raise ValueError(f"{owner} requires nlgeom=true")
    return step


def _resolve_analysis_options(
    step: Any,
    metadata: Any,
) -> StaticAnalysisOptions:
    """Resolve typed options, decoding persistence keys only at the boundary."""

    typed_options = getattr(step, "options", None)
    if typed_options is not None:
        if not isinstance(typed_options, StaticAnalysisOptions):
            raise TypeError("analysis step options must be StaticAnalysisOptions")
        return typed_options

    if not hasattr(metadata, "items"):
        raise TypeError("analysis step metadata must be a mapping")
    normalized = {
        str(key).strip().casefold(): value
        for key, value in metadata.items()
    }
    for key in (
        "material_algorithm",
        "j2_algorithm",
        "finite_strain_material_model",
        "material_model",
    ):
        if key in normalized:
            return StaticAnalysisOptions(material_algorithm=normalized[key])
    return StaticAnalysisOptions()


def _truthy_option(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized in {"", "0", "false", "no", "off"}:
            return False
        if normalized in {"1", "true", "yes", "on"}:
            return True
    return bool(value)


__all__ = [
    "is_nlgeom_enabled",
    "require_nlgeom",
    "resolve_analysis_request",
    "resolve_geometry_mode",
    "resolve_formulation",
    "resolve_nlgeom",
]
