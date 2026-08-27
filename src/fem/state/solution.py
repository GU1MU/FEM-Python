"""Global and element-local solution state shared by all procedures."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import numpy as np

from fem.model.dof_space import DofSpace


@dataclass(frozen=True, slots=True)
class SolutionState:
    """One global state over a compiled :class:`DofSpace`.

    ``values`` stores all primary unknowns.  The optional derivatives use the
    same global layout, so static procedures leave them unset while transient
    procedures can supply velocity/rate and acceleration without changing
    Problem or physics interfaces.
    """

    dof_space: DofSpace
    values: np.ndarray
    first_derivative: np.ndarray | None = None
    second_derivative: np.ndarray | None = None

    def __post_init__(self) -> None:
        if type(self.dof_space) is not DofSpace:
            raise TypeError("dof_space must be exactly DofSpace")
        object.__setattr__(
            self,
            "values",
            _owned_vector(self.values, self.dof_space.num_dofs, "values"),
        )
        for name in ("first_derivative", "second_derivative"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(
                    self,
                    name,
                    _owned_vector(value, self.dof_space.num_dofs, name),
                )

    @classmethod
    def zeros(cls, dof_space: DofSpace) -> "SolutionState":
        if type(dof_space) is not DofSpace:
            raise TypeError("dof_space must be exactly DofSpace")
        return cls(dof_space, np.zeros(dof_space.num_dofs, dtype=float))

    def field(self, name: str, *, derivative: int = 0) -> np.ndarray:
        if derivative not in {0, 1, 2}:
            raise ValueError("derivative must be 0, 1, or 2")
        source = (
            self.values
            if derivative == 0
            else self.first_derivative
            if derivative == 1
            else self.second_derivative
        )
        if source is None:
            raise ValueError(
                f"solution derivative order {derivative} is not available"
            )
        dofs = np.asarray(self.dof_space.field(name).dofs, dtype=int)
        result = np.array(source[dofs], dtype=float, copy=True)
        result.flags.writeable = False
        return result

    def with_values(self, values: np.ndarray) -> "SolutionState":
        return SolutionState(
            self.dof_space,
            values,
            self.first_derivative,
            self.second_derivative,
        )


@dataclass(frozen=True, slots=True)
class LocalFieldState:
    """Element-local values and optional time derivatives for one field."""

    values: np.ndarray
    first_derivative: np.ndarray | None = None
    second_derivative: np.ndarray | None = None

    def __post_init__(self) -> None:
        shape = np.asarray(self.values, dtype=float).shape
        if not shape:
            raise ValueError("local field values must be an array")
        object.__setattr__(self, "values", _owned_array(self.values, shape, "values"))
        for name in ("first_derivative", "second_derivative"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _owned_array(value, shape, name))


@dataclass(frozen=True, slots=True)
class EvaluationContext:
    """Procedure coordinates and scalar parameters for one evaluation."""

    time: float = 0.0
    load_factor: float = 1.0
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        time = float(self.time)
        load_factor = float(self.load_factor)
        if not np.isfinite(time) or not np.isfinite(load_factor):
            raise ValueError("evaluation time and load_factor must be finite")
        parameters = dict(self.parameters)
        object.__setattr__(self, "time", time)
        object.__setattr__(self, "load_factor", load_factor)
        object.__setattr__(self, "parameters", MappingProxyType(parameters))


def _owned_vector(value: object, size: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},), got {array.shape}")
    return _owned_array(array, (size,), name)


def _owned_array(value: object, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain finite values")
    owned = np.array(array, dtype=float, copy=True)
    owned.flags.writeable = False
    return owned


__all__ = ["EvaluationContext", "LocalFieldState", "SolutionState"]
