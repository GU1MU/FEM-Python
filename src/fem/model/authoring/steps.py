from __future__ import annotations

from dataclasses import replace
from typing import Any

from ..entities import AnalysisStep
from ..controls import (
    DynamicStepControls,
    GeometryMode,
    InitialConditionSet,
    StaticAnalysisOptions,
    StaticFormulation,
    StaticStepControls,
)


_STATIC_CONTROL_METADATA_KEYS = frozenset(
    StaticStepControls().to_metadata()
)
_STATIC_OPTION_METADATA_KEYS = frozenset(
    {
        "j2_algorithm",
        "finite_strain_material_model",
        "material_model",
    }
)
_DYNAMIC_CONTROL_METADATA_KEYS = frozenset(
    DynamicStepControls().to_metadata()
)


def update_static_step(
    step: AnalysisStep,
    *,
    name: str,
    formulation: StaticFormulation,
    controls: StaticStepControls | None,
) -> AnalysisStep:
    """Return a complete static-step update without dropping authored state.

    ``AnalysisStep`` keeps a metadata mirror for the project-file boundary,
    but the typed formulation and controls are the authoring source of truth.
    This helper owns the mirror maintenance needed when a GUI edits an
    existing step; all loads, boundaries, outputs, options, and unrelated
    metadata are copied from ``step`` by ``dataclasses.replace``.
    """

    if type(step) is not AnalysisStep:
        raise TypeError("step must be exactly AnalysisStep")
    if not isinstance(formulation, StaticFormulation):
        raise TypeError("formulation must be StaticFormulation")
    if controls is not None and type(controls) is not StaticStepControls:
        raise TypeError("controls must be StaticStepControls or None")

    control_keys = {
        str(key).strip().casefold()
        for key in _STATIC_CONTROL_METADATA_KEYS
    }
    option_keys = {
        str(key).strip().casefold()
        for key in _STATIC_OPTION_METADATA_KEYS
    }
    dynamic_control_keys = {
        str(key).strip().casefold()
        for key in _DYNAMIC_CONTROL_METADATA_KEYS
    }
    metadata = dict(step.metadata)
    for key in tuple(metadata):
        normalized = str(key).strip().casefold()
        if (
            normalized == "nlgeom"
            or normalized in control_keys
            or normalized in dynamic_control_keys
            or normalized == "initial_conditions"
        ):
            metadata.pop(key)
        elif (
            formulation is StaticFormulation.LINEAR
            and normalized in option_keys
        ):
            metadata.pop(key)

    if formulation is StaticFormulation.NONLINEAR:
        metadata["NLGEOM"] = True
    if controls is not None:
        metadata.update(controls.to_metadata())

    return replace(
        step,
        name=str(name),
        procedure="static",
        metadata=metadata,
        controls=controls,
        formulation=formulation,
        geometry_mode=GeometryMode.from_formulation(formulation),
        initial_conditions=InitialConditionSet(),
    )


def update_dynamic_step(
    step: AnalysisStep,
    *,
    name: str,
    controls: DynamicStepControls,
    initial_conditions: InitialConditionSet | None = None,
    formulation: StaticFormulation = StaticFormulation.LINEAR,
    geometry_mode: GeometryMode | None = None,
) -> AnalysisStep:
    """Return a complete implicit or explicit transient-step update.

    The procedure switch is deliberately handled here, at the authoring
    boundary.  A GUI edit can therefore change a static step into a dynamic
    step without leaving stale static control keys in the persistence mirror,
    while all authored boundaries, loads, and output requests remain intact.
    """

    if type(step) is not AnalysisStep:
        raise TypeError("step must be exactly AnalysisStep")
    if type(controls) is not DynamicStepControls:
        raise TypeError("controls must be exactly DynamicStepControls")
    if not isinstance(formulation, StaticFormulation):
        raise TypeError("formulation must be StaticFormulation")
    if geometry_mode is None:
        geometry_mode = GeometryMode.from_formulation(formulation)
    elif not isinstance(geometry_mode, GeometryMode):
        raise TypeError("geometry_mode must be GeometryMode or None")
    if geometry_mode.formulation is not formulation:
        raise ValueError("formulation and geometry_mode must agree")
    selected_initial_conditions = (
        step.initial_conditions
        if initial_conditions is None
        else initial_conditions
    )
    if type(selected_initial_conditions) is not InitialConditionSet:
        raise TypeError(
            "initial_conditions must be exactly InitialConditionSet or None"
        )

    static_keys = {
        str(key).strip().casefold()
        for key in _STATIC_CONTROL_METADATA_KEYS
    }
    dynamic_keys = {
        str(key).strip().casefold()
        for key in _DYNAMIC_CONTROL_METADATA_KEYS
    }
    static_option_keys = {
        str(key).strip().casefold()
        for key in _STATIC_OPTION_METADATA_KEYS
    }
    metadata = dict(step.metadata)
    for key in tuple(metadata):
        normalized = str(key).strip().casefold()
        if (
            normalized in static_keys
            or normalized in dynamic_keys
            or normalized in static_option_keys
            or normalized in {"nlgeom", "initial_conditions"}
        ):
            metadata.pop(key)
    metadata.update(controls.to_metadata())
    if formulation is StaticFormulation.NONLINEAR:
        metadata["NLGEOM"] = True

    return replace(
        step,
        name=str(name),
        procedure="dynamic",
        metadata=metadata,
        controls=controls,
        formulation=formulation,
        geometry_mode=geometry_mode,
        options=None,
        initial_conditions=selected_initial_conditions,
    )


def static(
    name: str = "Step-1",
    *,
    controls: StaticStepControls | None = None,
    formulation: StaticFormulation | None = None,
    geometry_mode: GeometryMode | None = None,
    options: StaticAnalysisOptions | None = None,
    **metadata: Any,
) -> AnalysisStep:
    """Create a static analysis step from typed formulation and controls.

    ``metadata`` is retained only as the persistence mirror used by the
    current project-file boundary.  Execution reads ``formulation`` and
    ``controls`` from the authored value object.
    """
    if geometry_mode is not None and not isinstance(geometry_mode, GeometryMode):
        raise TypeError("geometry_mode must be GeometryMode or None")
    if formulation is None and geometry_mode is None:
        formulation = _formulation_from_persistence_metadata(metadata)
    elif formulation is None:
        formulation = geometry_mode.formulation
    elif geometry_mode is None:
        geometry_mode = GeometryMode.from_formulation(formulation)
    elif geometry_mode.formulation is not formulation:
        raise ValueError("formulation and geometry_mode must agree")
    if not isinstance(formulation, StaticFormulation):
        raise TypeError("formulation must be StaticFormulation")
    if options is not None and not isinstance(options, StaticAnalysisOptions):
        raise TypeError("options must be StaticAnalysisOptions")
    metadata = dict(metadata)
    if formulation is StaticFormulation.NONLINEAR:
        metadata["NLGEOM"] = True
    else:
        metadata.pop("NLGEOM", None)
    if controls is not None:
        if not isinstance(controls, StaticStepControls):
            raise TypeError("controls must be StaticStepControls")
        metadata = {**metadata, **controls.to_metadata()}
    return AnalysisStep(
        str(name),
        procedure="static",
        metadata=metadata,
        controls=controls,
        formulation=formulation,
        geometry_mode=geometry_mode,
        options=options,
    )


def transient_dynamic(
    name: str = "DynamicStep-1",
    *,
    controls: DynamicStepControls | None = None,
    initial_conditions: InitialConditionSet | None = None,
    formulation: StaticFormulation = StaticFormulation.LINEAR,
    geometry_mode: GeometryMode | None = None,
    **metadata: Any,
) -> AnalysisStep:
    """Create an Abaqus-style implicit or explicit dynamic step."""

    selected_controls = (
        DynamicStepControls() if controls is None else controls
    )
    if not isinstance(selected_controls, DynamicStepControls):
        raise TypeError("controls must be DynamicStepControls")
    selected_initial_conditions = (
        InitialConditionSet()
        if initial_conditions is None
        else initial_conditions
    )
    if not isinstance(selected_initial_conditions, InitialConditionSet):
        raise TypeError("initial_conditions must be InitialConditionSet")
    if not isinstance(formulation, StaticFormulation):
        raise TypeError("formulation must be StaticFormulation")
    if geometry_mode is None:
        geometry_mode = GeometryMode.from_formulation(formulation)
    elif not isinstance(geometry_mode, GeometryMode):
        raise TypeError("geometry_mode must be GeometryMode or None")
    if geometry_mode.formulation is not formulation:
        raise ValueError("formulation and geometry_mode must agree")
    if formulation is StaticFormulation.NONLINEAR:
        metadata["NLGEOM"] = True
    return AnalysisStep(
        str(name),
        procedure="dynamic",
        metadata={**metadata, **selected_controls.to_metadata()},
        controls=selected_controls,
        formulation=formulation,
        geometry_mode=geometry_mode,
        initial_conditions=selected_initial_conditions,
    )


def _formulation_from_persistence_metadata(
    metadata: dict[str, Any],
) -> StaticFormulation:
    """Translate the current persistence spelling at the authoring edge."""

    for key, value in reversed(tuple(metadata.items())):
        if str(key).strip().casefold() != "nlgeom":
            continue
        if isinstance(value, str):
            enabled = value.strip().casefold() not in {
                "",
                "0",
                "false",
                "no",
                "off",
            }
        else:
            enabled = bool(value)
        return (
            StaticFormulation.NONLINEAR
            if enabled
            else StaticFormulation.LINEAR
        )
    return StaticFormulation.LINEAR


def add(model: Any, step: AnalysisStep) -> AnalysisStep:
    """Add a step to a model."""
    model.steps.append(step)
    return step


__all__ = [
    "add",
    "static",
    "transient_dynamic",
    "update_dynamic_step",
    "update_static_step",
]
