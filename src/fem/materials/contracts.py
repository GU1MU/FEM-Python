"""Measure-explicit material-point contracts shared by physics operators."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable

import numpy as np

from fem.state import EvaluationContext


class StressMeasure(str, Enum):
    """Stress tensor represented by a constitutive response."""

    CAUCHY = "cauchy"
    FIRST_PIOLA = "first_piola"
    SECOND_PIOLA = "second_piola"
    KIRCHHOFF = "kirchhoff"


class KinematicMeasure(str, Enum):
    """Kinematic quantity differentiated by a constitutive tangent."""

    SMALL_STRAIN = "small_strain"
    DEFORMATION_GRADIENT = "deformation_gradient"
    GREEN_LAGRANGE = "green_lagrange"
    LOGARITHMIC_STRAIN = "logarithmic_strain"


@dataclass(frozen=True, slots=True)
class MaterialPointInput:
    """Pure input supplied to one material-point evaluation.

    ``fields`` carries interpolated coupled quantities such as temperature,
    concentration, or electric potential. Adding another physical field does
    not change this contract.
    """

    kinematics: Any
    committed_state: Mapping[str, Any] = field(default_factory=dict)
    context: EvaluationContext = field(default_factory=EvaluationContext)
    fields: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.committed_state, Mapping):
            raise TypeError("committed_state must be a mapping")
        if type(self.context) is not EvaluationContext:
            raise TypeError("context must be exactly EvaluationContext")
        if not isinstance(self.fields, Mapping):
            raise TypeError("fields must be a mapping")
        object.__setattr__(
            self,
            "committed_state",
            MappingProxyType(_owned_mapping(self.committed_state)),
        )
        object.__setattr__(
            self,
            "fields",
            MappingProxyType(_owned_mapping(self.fields)),
        )


@dataclass(frozen=True, slots=True)
class MaterialResponse:
    """Stress and consistent tangent with explicit input/output measures."""

    stress: np.ndarray
    tangent: np.ndarray
    stress_measure: StressMeasure
    tangent_input: KinematicMeasure
    trial_state: Mapping[str, Any] = field(default_factory=dict)
    outputs: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        stress = np.asarray(self.stress, dtype=float)
        tangent = np.asarray(self.tangent, dtype=float)
        if stress.ndim != 2 or stress.shape[0] != stress.shape[1]:
            raise ValueError("material stress must be a square tensor")
        if tangent.ndim != 2:
            raise ValueError("material tangent must be a matrix")
        if not np.all(np.isfinite(stress)) or not np.all(np.isfinite(tangent)):
            raise ValueError("material response must contain finite values")
        try:
            stress_measure = StressMeasure(self.stress_measure)
            tangent_input = KinematicMeasure(self.tangent_input)
        except ValueError as exc:
            raise ValueError("material response uses an unsupported measure") from exc
        if not isinstance(self.trial_state, Mapping):
            raise TypeError("trial_state must be a mapping")
        if not isinstance(self.outputs, Mapping):
            raise TypeError("outputs must be a mapping")
        owned_stress = np.array(stress, copy=True)
        owned_tangent = np.array(tangent, copy=True)
        owned_stress.flags.writeable = False
        owned_tangent.flags.writeable = False
        object.__setattr__(self, "stress", owned_stress)
        object.__setattr__(self, "tangent", owned_tangent)
        object.__setattr__(self, "stress_measure", stress_measure)
        object.__setattr__(self, "tangent_input", tangent_input)
        object.__setattr__(
            self,
            "trial_state",
            MappingProxyType(_owned_mapping(self.trial_state)),
        )
        object.__setattr__(
            self,
            "outputs",
            MappingProxyType(_owned_mapping(self.outputs)),
        )

    def require(
        self,
        *,
        stress_measure: StressMeasure,
        tangent_input: KinematicMeasure,
        stress_shape: tuple[int, ...] | None = None,
        tangent_shape: tuple[int, ...] | None = None,
    ) -> None:
        """Validate the exact constitutive contract expected by an operator."""

        if self.stress_measure is not StressMeasure(stress_measure):
            raise ValueError(
                f"physics operator requires stress measure {stress_measure.value}, "
                f"got {self.stress_measure.value}"
            )
        if self.tangent_input is not KinematicMeasure(tangent_input):
            raise ValueError(
                f"physics operator requires tangent input {tangent_input.value}, "
                f"got {self.tangent_input.value}"
            )
        if stress_shape is not None and self.stress.shape != stress_shape:
            raise ValueError(
                f"material stress must have shape {stress_shape}, got {self.stress.shape}"
            )
        if tangent_shape is not None and self.tangent.shape != tangent_shape:
            raise ValueError(
                "material tangent must have shape "
                f"{tangent_shape}, got {self.tangent.shape}"
            )


def _owned_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    if getattr(value, "_fem_immutable_mapping", False):
        return dict(value)
    return deepcopy(dict(value))


@runtime_checkable
class MaterialModel(Protocol):
    """Stateless constitutive algorithm with transactional external history."""

    def initial_state(self) -> Mapping[str, Any]:
        """Return one detached state for a new integration point."""

    def evaluate(self, point: MaterialPointInput) -> MaterialResponse:
        """Return stress, tangent, outputs, and an uncommitted trial state."""


__all__ = [
    "KinematicMeasure",
    "MaterialModel",
    "MaterialPointInput",
    "MaterialResponse",
    "StressMeasure",
]
