"""Typed transient-result data shared by solvers and post-processing."""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
from typing import Any, Mapping

import numpy as np


def _optional_finite(value: Any, *, name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number or None")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be finite")
    return numeric


@dataclass(frozen=True, slots=True)
class DynamicEnergy:
    """Energy quantities recorded at one accepted dynamic frame."""

    kinetic: float | None = None
    internal: float | None = None
    external_work: float | None = None
    damping_dissipation: float | None = None
    plastic_dissipation: float | None = None
    artificial: float | None = None
    total: float | None = None
    balance_error: float | None = None

    def __post_init__(self) -> None:
        for name in (
            "kinetic",
            "internal",
            "external_work",
            "damping_dissipation",
            "plastic_dissipation",
            "artificial",
            "total",
            "balance_error",
        ):
            object.__setattr__(self, name, _optional_finite(getattr(self, name), name=name))


@dataclass(frozen=True, slots=True)
class DynamicDiagnostics:
    """Procedure-specific diagnostics for one accepted frame."""

    solver_kind: str
    increment: int
    attempt: int = 1
    iterations: int | None = None
    residual_norm: float | None = None
    stable_time_increment: float | None = None
    critical_element: int | None = None
    status: str = "converged"

    def __post_init__(self) -> None:
        if not isinstance(self.solver_kind, str) or not self.solver_kind.strip():
            raise ValueError("solver_kind must be a non-empty string")
        if isinstance(self.increment, bool) or not isinstance(self.increment, int) or self.increment < 0:
            raise ValueError("increment must be an integer >= 0")
        if isinstance(self.attempt, bool) or not isinstance(self.attempt, int) or self.attempt < 1:
            raise ValueError("attempt must be an integer >= 1")
        if self.iterations is not None and (
            isinstance(self.iterations, bool)
            or not isinstance(self.iterations, int)
            or self.iterations < 0
        ):
            raise ValueError("iterations must be an integer >= 0 or None")
        object.__setattr__(
            self,
            "residual_norm",
            _optional_finite(self.residual_norm, name="residual_norm"),
        )
        stable = _optional_finite(
            self.stable_time_increment,
            name="stable_time_increment",
        )
        if stable is not None and stable <= 0.0:
            raise ValueError("stable_time_increment must be > 0")
        object.__setattr__(self, "stable_time_increment", stable)
        if self.critical_element is not None and (
            isinstance(self.critical_element, bool)
            or not isinstance(self.critical_element, int)
            or self.critical_element < 1
        ):
            raise ValueError("critical_element must be an integer >= 1 or None")
        if not isinstance(self.status, str) or not self.status.strip():
            raise ValueError("status must be a non-empty string")


@dataclass(frozen=True, slots=True)
class DynamicFrameData:
    """Time, derivative and diagnostic data belonging to one result frame."""

    solver_kind: str
    step_time: float
    total_time: float
    time_increment: float
    amplitude: float | None = None
    velocity: np.ndarray | None = None
    acceleration: np.ndarray | None = None
    energy: DynamicEnergy = DynamicEnergy()
    diagnostics: DynamicDiagnostics | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.solver_kind, str) or not self.solver_kind.strip():
            raise ValueError("solver_kind must be a non-empty string")
        for name in ("step_time", "total_time", "time_increment"):
            value = _optional_finite(getattr(self, name), name=name)
            if value is None or value < 0.0 or (name == "time_increment" and value < 0.0):
                raise ValueError(f"{name} must be finite and >= 0")
            object.__setattr__(self, name, value)
        if self.step_time < 0.0 or self.total_time < 0.0:
            raise ValueError("dynamic time values must be >= 0")
        object.__setattr__(self, "amplitude", _optional_finite(self.amplitude, name="amplitude"))
        if type(self.energy) is not DynamicEnergy:
            raise TypeError("energy must be DynamicEnergy")
        if self.diagnostics is not None and type(self.diagnostics) is not DynamicDiagnostics:
            raise TypeError("diagnostics must be DynamicDiagnostics or None")
        for name in ("velocity", "acceleration"):
            raw = getattr(self, name)
            if raw is None:
                continue
            vector = np.asarray(raw, dtype=float)
            if vector.ndim != 1 or not np.all(np.isfinite(vector)):
                raise ValueError(f"{name} must be a finite one-dimensional vector")
            owned = np.array(vector, copy=True)
            owned.flags.writeable = False
            object.__setattr__(self, name, owned)

    @property
    def is_initial(self) -> bool:
        return self.step_time == 0.0 and self.time_increment == 0.0


def dynamic_frame_data_from_outputs(
    *,
    time: float,
    time_increment: float,
    outputs: Mapping[str, Any],
    residual_norm: float | None,
    iterations: int | None,
    solver_kind: str | None = None,
    amplitude: float | None = None,
    velocity: np.ndarray | None = None,
    acceleration: np.ndarray | None = None,
) -> DynamicFrameData:
    """Build the public transient contract from one solver evaluation."""

    kind = solver_kind or str(outputs.get("procedure", "dynamic_implicit"))
    if amplitude is None:
        amplitude = outputs.get("load_factor")
    if velocity is None:
        velocity = outputs.get("velocity")
    if acceleration is None:
        acceleration = outputs.get("acceleration")
    energy = DynamicEnergy(
        kinetic=outputs.get("kinetic_energy"),
        internal=outputs.get("internal_energy", outputs.get("strain_energy")),
        external_work=outputs.get("external_work"),
        damping_dissipation=outputs.get("damping_dissipation"),
        plastic_dissipation=outputs.get("plastic_dissipation"),
        artificial=outputs.get("artificial_energy"),
        total=outputs.get("total_energy"),
        balance_error=outputs.get("energy_balance_error"),
    )
    diagnostics = DynamicDiagnostics(
        solver_kind=kind,
        increment=int(outputs.get("increment_number", 0)),
        attempt=int(outputs.get("attempt_number", 1)),
        iterations=iterations,
        residual_norm=residual_norm,
        stable_time_increment=outputs.get("stable_time_increment"),
        critical_element=outputs.get("critical_element"),
        status=str(outputs.get("increment_status", "converged")),
    )
    return DynamicFrameData(
        solver_kind=kind,
        step_time=float(outputs.get("step_time", time)),
        total_time=float(outputs.get("total_time", time)),
        time_increment=float(time_increment),
        amplitude=amplitude,
        velocity=velocity,
        acceleration=acceleration,
        energy=energy,
        diagnostics=diagnostics,
    )


__all__ = [
    "DynamicDiagnostics",
    "DynamicEnergy",
    "DynamicFrameData",
    "dynamic_frame_data_from_outputs",
]
