"""A5 strict, named linear-static analysis authoring."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from fem.application import UnitContext

from .naming import NamePolicy


class AnalysisAuthoringError(ValueError):
    """An A5 analysis definition is incomplete, ambiguous, or unsupported."""


@dataclass(frozen=True, slots=True)
class ConfirmedDisplacement:
    name: str
    step_name: str
    target_scope: str
    entity_type: str
    first_component: int
    last_component: int
    value: float
    unit: str
    distribution: str
    confirmed: bool

    def __post_init__(self) -> None:
        NamePolicy().validate(self.name)
        if not NamePolicy.has_type(self.name, "Displacement"):
            raise AnalysisAuthoringError("displacement name must use type Displacement")
        _nonblank(self.step_name, "displacement step")
        _nonblank(self.target_scope, "displacement target scope")
        if self.entity_type not in {"node_set", "edge", "surface"}:
            raise AnalysisAuthoringError(
                "displacement entity_type must be node_set, edge, or surface"
            )
        if (
            type(self.first_component) is not int
            or type(self.last_component) is not int
            or self.first_component < 1
            or self.last_component < self.first_component
        ):
            raise AnalysisAuthoringError(
                "displacement components must be one explicit contiguous range"
            )
        _finite(self.value, "displacement value")
        _nonblank(self.unit, "displacement unit")
        if self.distribution != "uniform":
            raise AnalysisAuthoringError(
                "A5 displacement distribution must be explicitly uniform"
            )
        if self.confirmed is not True:
            raise AnalysisAuthoringError(
                "displacement engineering fields are not confirmed"
            )

    def summary(self) -> dict[str, object]:
        translations = tuple(
            component
            for component in range(self.first_component, self.last_component + 1)
            if component <= 3
        )
        rotations = tuple(
            component
            for component in range(self.first_component, self.last_component + 1)
            if component >= 4
        )
        return {
            "name": self.name,
            "step": self.step_name,
            "target_scope": self.target_scope,
            "entity_type": self.entity_type,
            "degrees_of_freedom": [
                self.first_component,
                self.last_component,
            ],
            "component_direction": (
                f"U{self.first_component}"
                if self.first_component == self.last_component
                else f"U{self.first_component}..U{self.last_component}"
            ),
            "translation_components": list(translations),
            "rotation_components": list(rotations),
            "translation_unit": self.unit if translations else None,
            "rotation_unit": "rad" if rotations else None,
            "signed_value": self.value,
            "unit": self.unit,
            "distribution": self.distribution,
        }


@dataclass(frozen=True, slots=True)
class ConfirmedLoad:
    name: str
    step_name: str
    target_scope: str | None
    entity_type: str
    load_type: str
    component: int | None
    vector: tuple[float, ...]
    magnitude: float | None
    direction: str
    unit: str
    distribution: str
    confirmed: bool
    coordinate_system: str | None = None
    acceleration: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        NamePolicy().validate(self.name)
        if not NamePolicy.has_type(self.name, "Load"):
            raise AnalysisAuthoringError("load name must use type Load")
        _nonblank(self.step_name, "load step")
        if self.target_scope is not None:
            _nonblank(self.target_scope, "load target scope")
        if self.load_type not in {
            "nodal",
            "edge_traction",
            "edge_pressure",
            "surface_traction",
            "surface_pressure",
            "line",
            "body",
            "gravity",
        }:
            raise AnalysisAuthoringError("load_type is unsupported by A5")
        expected_entity = {
            "nodal": "node",
            "edge_traction": "edge",
            "edge_pressure": "edge",
            "surface_traction": "surface",
            "surface_pressure": "surface",
            "line": "element",
            "body": "element",
            "gravity": "element",
        }[self.load_type]
        if self.entity_type != expected_entity:
            raise AnalysisAuthoringError(
                "load entity_type does not match load_type"
            )
        values = tuple(self.vector)
        if any(not _is_finite(value) for value in values):
            raise AnalysisAuthoringError("load vector must contain finite numbers")
        object.__setattr__(self, "vector", tuple(float(value) for value in values))
        if self.magnitude is not None:
            _finite(self.magnitude, "load magnitude")
        acceleration = tuple(self.acceleration)
        if any(not _is_finite(value) for value in acceleration):
            raise AnalysisAuthoringError(
                "gravity acceleration must contain finite numbers"
            )
        object.__setattr__(
            self,
            "acceleration",
            tuple(float(value) for value in acceleration),
        )
        if self.load_type != "gravity" and self.target_scope is None:
            raise AnalysisAuthoringError(
                "only gravity may target the whole model"
            )
        if self.load_type == "nodal":
            if type(self.component) is not int or self.component < 1:
                raise AnalysisAuthoringError(
                    "nodal load requires one explicit component"
                )
            if values or self.magnitude is None:
                raise AnalysisAuthoringError(
                    "nodal load requires only one signed magnitude"
                )
            if self.distribution != "concentrated":
                raise AnalysisAuthoringError(
                    "nodal load distribution must be concentrated"
                )
        elif self.load_type in {"line", "body"}:
            if self.component is not None or not values or self.magnitude is not None:
                raise AnalysisAuthoringError(
                    f"{self.load_type} load requires an explicit signed vector only"
                )
            if acceleration:
                raise AnalysisAuthoringError(
                    f"{self.load_type} load cannot contain acceleration"
                )
            if self.distribution != "uniform":
                raise AnalysisAuthoringError(
                    f"{self.load_type} load must be explicitly uniform"
                )
            if self.load_type == "line":
                if self.coordinate_system not in {"global", "local"}:
                    raise AnalysisAuthoringError(
                        "line load coordinate_system must be global or local"
                    )
                if self.direction != self.coordinate_system:
                    raise AnalysisAuthoringError(
                        "line load direction must match coordinate_system"
                    )
            else:
                if self.coordinate_system not in {None, "global"}:
                    raise AnalysisAuthoringError(
                        "body force coordinate_system must be global"
                    )
                if self.direction != "global":
                    raise AnalysisAuthoringError(
                        "body force direction must be global"
                    )
        elif self.load_type == "gravity":
            if self.component is not None or values or self.magnitude is not None:
                raise AnalysisAuthoringError(
                    "gravity requires acceleration instead of vector or magnitude"
                )
            if not acceleration:
                raise AnalysisAuthoringError(
                    "gravity requires explicit acceleration"
                )
            if self.coordinate_system not in {None, "global"}:
                raise AnalysisAuthoringError(
                    "gravity coordinate_system must be global"
                )
            if self.direction != "global" or self.distribution != "uniform":
                raise AnalysisAuthoringError(
                    "gravity must be global and uniform"
                )
        elif self.load_type.endswith("traction"):
            if self.component is not None or not values or self.magnitude is not None:
                raise AnalysisAuthoringError(
                    "traction requires an explicit signed vector only"
                )
            if self.distribution != "uniform":
                raise AnalysisAuthoringError(
                    "traction distribution must be explicitly uniform"
                )
        else:
            if self.component is not None or values or self.magnitude is None:
                raise AnalysisAuthoringError(
                    "pressure requires one signed magnitude only"
                )
            if self.direction not in {"outward_normal", "inward_normal"}:
                raise AnalysisAuthoringError(
                    "pressure requires an explicit normal direction"
                )
            if (
                self.direction == "inward_normal"
                and float(self.magnitude) <= 0.0
            ) or (
                self.direction == "outward_normal"
                and float(self.magnitude) >= 0.0
            ):
                raise AnalysisAuthoringError(
                    "pressure sign must be positive inward and negative outward"
                )
            if self.distribution != "uniform":
                raise AnalysisAuthoringError(
                    "pressure distribution must be explicitly uniform"
                )
        _nonblank(self.direction, "load direction")
        _nonblank(self.unit, "load unit")
        if self.confirmed is not True:
            raise AnalysisAuthoringError(
                "load engineering fields are not confirmed"
            )

    def summary(self) -> dict[str, object]:
        nodal_family = (
            None
            if self.load_type != "nodal"
            else "force" if int(self.component or 0) <= 3 else "moment"
        )
        return {
            "name": self.name,
            "step": self.step_name,
            "target_scope": self.target_scope,
            "entity_type": self.entity_type,
            "load_type": self.load_type,
            "component": self.component,
            "nodal_family": nodal_family,
            "direction": self.direction,
            "signed_components": list(self.vector),
            "signed_magnitude": self.magnitude,
            "coordinate_system": self.coordinate_system,
            "acceleration": list(self.acceleration),
            "unit": self.unit,
            "distribution": self.distribution,
        }


@dataclass(frozen=True, slots=True)
class ConfirmedResultRequest:
    name: str
    step_name: str
    kind: str
    target: str
    variables: tuple[str, ...]
    units: tuple[str, ...]
    confirmed: bool

    def __post_init__(self) -> None:
        NamePolicy().validate(self.name)
        if not NamePolicy.has_type(self.name, "Output"):
            raise AnalysisAuthoringError(
                "result request name must use type Output"
            )
        _nonblank(self.step_name, "result request step")
        if self.kind != "field":
            raise AnalysisAuthoringError("A5 supports only field output requests")
        supported = {
            "node": frozenset({"U", "UR", "RF", "RM"}),
            "element": frozenset({"SF", "SM", "LE", "S"}),
        }
        if self.target not in supported:
            raise AnalysisAuthoringError("result request target is unsupported")
        variables = tuple(self.variables)
        if (
            not variables
            or any(type(item) is not str for item in variables)
            or not set(variables).issubset(supported[self.target])
        ):
            raise AnalysisAuthoringError(
                "result variables are missing or unsupported for their target"
            )
        if len(self.units) != len(variables) or any(
            type(item) is not str or not item.strip() for item in self.units
        ):
            raise AnalysisAuthoringError(
                "result variables require explicit aligned units"
            )
        object.__setattr__(self, "variables", variables)
        object.__setattr__(self, "units", tuple(self.units))
        if self.confirmed is not True:
            raise AnalysisAuthoringError(
                "result request engineering fields are not confirmed"
            )

    def summary(self) -> dict[str, object]:
        return {
            "name": self.name,
            "step": self.step_name,
            "kind": self.kind,
            "target": self.target,
            "variables": list(self.variables),
            "units": list(self.units),
        }


def expected_result_units(
    units: UnitContext,
    variables: Sequence[str],
) -> tuple[str, ...]:
    """Return exact project units for canonical result variables."""

    mapping = {
        "U": units.length,
        "UR": "rad",
        "RF": units.force,
        "RM": f"{units.force}*{units.length}",
        "SF": units.force,
        "SM": f"{units.force}*{units.length}",
        "LE": "1",
        "S": units.stress,
    }
    try:
        return tuple(mapping[variable] for variable in variables)
    except KeyError as error:
        raise AnalysisAuthoringError(
            f"result variable {error.args[0]!r} has no unit semantics"
        ) from error


def _nonblank(value: object, label: str) -> str:
    if type(value) is not str or not value.strip() or value != value.strip():
        raise AnalysisAuthoringError(
            f"{label} must be a nonblank string without outer whitespace"
        )
    return value


def _is_finite(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and type(value) in {int, float}
        and math.isfinite(float(value))
    )


def _finite(value: object, label: str) -> float:
    if not _is_finite(value):
        raise AnalysisAuthoringError(f"{label} must be a finite number")
    return float(value)


__all__ = [
    "AnalysisAuthoringError",
    "ConfirmedDisplacement",
    "ConfirmedLoad",
    "ConfirmedResultRequest",
]
