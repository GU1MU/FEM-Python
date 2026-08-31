"""Convergence metrics for implicit static Newton iterations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class ConvergenceMetrics:
    """Detached convergence facts for one evaluated Newton state."""

    residual_norm: float
    force_norm: float
    force_reference: float
    relative_force_norm: float
    constraint_norm: float
    displacement_norm: float
    energy_norm: float
    merit: float
    has_constraints: bool

    def __post_init__(self) -> None:
        for name in (
            "residual_norm",
            "force_norm",
            "force_reference",
            "relative_force_norm",
            "constraint_norm",
            "displacement_norm",
            "energy_norm",
            "merit",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and >= 0")
            object.__setattr__(self, name, value)
        if type(self.has_constraints) is not bool:
            raise TypeError("has_constraints must be bool")

    def as_mapping(self) -> dict[str, float | bool]:
        """Return JSON-friendly diagnostic values."""

        return {
            "residual_norm": self.residual_norm,
            "force_norm": self.force_norm,
            "force_reference": self.force_reference,
            "relative_force_norm": self.relative_force_norm,
            "constraint_norm": self.constraint_norm,
            "displacement_norm": self.displacement_norm,
            "energy_norm": self.energy_norm,
            "merit": self.merit,
            "has_constraints": self.has_constraints,
        }


def evaluate(
    evaluation: Any,
    *,
    increment: np.ndarray | None,
    residual_tolerance: float,
    reference_scale: float | None = None,
) -> ConvergenceMetrics:
    """Extract dimensionally separated metrics from one problem evaluation.

    Static problems publish ``free_residual`` and ``constraint_error`` in
    their output contract.  Small contract-only Problems may publish neither;
    those fall back to the original full residual behavior.
    """

    outputs = getattr(evaluation, "outputs", {})
    raw_force = outputs.get("free_residual", getattr(evaluation, "residual", None))
    force = _vector(raw_force, "force residual")
    raw_constraint = outputs.get("constraint_error")
    has_constraints = (
        raw_constraint is not None
        and _vector(raw_constraint, "constraint error").size > 0
    )
    constraint = (
        _vector(raw_constraint, "constraint error")
        if has_constraints
        else np.zeros(0, dtype=float)
    )
    displacement = (
        np.zeros(0, dtype=float)
        if increment is None
        else _vector(increment, "Newton increment")
    )
    free_dofs = outputs.get("free_dofs")
    free_increment = displacement
    if free_dofs is not None and displacement.size:
        indexes = tuple(int(value) for value in free_dofs)
        if any(index < 0 or index >= displacement.size for index in indexes):
            raise ValueError("free_dofs contains an out-of-range DOF")
        free_increment = displacement[list(indexes)]
    energy = (
        0.0
        if increment is None
        else abs(float(np.dot(free_increment, force)))
    )
    force_norm = _norm(force)
    constraint_norm = _norm(constraint)
    external = _vector(outputs.get("external_force"), "external force", allow_none=True)
    external_norm = _norm(external)
    tolerance = _positive(residual_tolerance, "residual_tolerance")
    force_reference = (
        max(external_norm, force_norm, tolerance)
        if reference_scale is None
        else max(_positive(reference_scale, "reference_scale"), tolerance)
    )
    relative_force = force_norm / force_reference
    force_merit = force_norm / max(force_reference, tolerance)
    constraint_scale = max(
        _mapping_norm(outputs.get("constraint_values")),
        constraint_norm,
        tolerance,
    )
    constraint_merit = constraint_norm / constraint_scale
    return ConvergenceMetrics(
        residual_norm=max(force_norm, constraint_norm),
        force_norm=force_norm,
        force_reference=force_reference,
        relative_force_norm=relative_force,
        constraint_norm=constraint_norm,
        displacement_norm=_norm(displacement),
        energy_norm=energy,
        merit=max(force_merit, constraint_merit),
        has_constraints=has_constraints,
    )


def converged(
    metrics: ConvergenceMetrics,
    *,
    residual_tolerance: float,
    relative_residual_tolerance: float | None = None,
    displacement_tolerance: float | None = None,
    energy_tolerance: float | None = None,
    constraint_tolerance: float | None = None,
) -> bool:
    """Apply the configured multi-criterion convergence policy."""

    force_ok = metrics.force_norm <= _positive(
        residual_tolerance,
        "residual_tolerance",
    )
    if relative_residual_tolerance is not None:
        relative_ok = metrics.relative_force_norm <= _positive(
            relative_residual_tolerance,
            "relative_residual_tolerance",
        )
        # A displacement-controlled step has no external-force scale.  In
        # that case the absolute force criterion is the meaningful one.
        if metrics.force_reference > float(residual_tolerance):
            force_ok = force_ok and relative_ok
    if not metrics.has_constraints:
        constraint_ok = True
    elif constraint_tolerance is None:
        constraint_ok = metrics.constraint_norm == 0.0
    else:
        constraint_ok = metrics.constraint_norm <= _positive(
            constraint_tolerance,
            "constraint_tolerance",
        )
    displacement_ok = (
        displacement_tolerance is None
        or not metrics.has_constraints
        or metrics.displacement_norm
        <= _positive(displacement_tolerance, "displacement_tolerance")
    )
    energy_ok = (
        energy_tolerance is None
        or metrics.energy_norm
        <= _positive(energy_tolerance, "energy_tolerance")
    )
    return force_ok and constraint_ok and displacement_ok and energy_ok


def _vector(
    value: Any,
    label: str,
    *,
    allow_none: bool = False,
) -> np.ndarray:
    if value is None and allow_none:
        return np.zeros(0, dtype=float)
    if value is None:
        raise ValueError(f"{label} must be provided")
    array = np.asarray(value, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{label} must be one-dimensional")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{label} must contain finite values")
    return array


def _norm(value: np.ndarray) -> float:
    return 0.0 if value.size == 0 else float(np.linalg.norm(value, ord=np.inf))


def _mapping_norm(value: Any) -> float:
    if value is None:
        return 0.0
    if not hasattr(value, "values"):
        raise ValueError("constraint values must be a mapping")
    return _norm(_vector(tuple(value.values()), "constraint values"))


def _positive(value: Any, label: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{label} must be finite and > 0")
    return result


__all__ = ["ConvergenceMetrics", "converged", "evaluate"]
