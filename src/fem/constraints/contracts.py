"""Immutable equation-constraint data used by Problems."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any


@dataclass(frozen=True, slots=True)
class ConstraintSet:
    """Prescribed equations for one compiled analysis step."""

    prescribed_values: Mapping[int, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        values = {
            int(dof): _finite(value, "prescribed value")
            for dof, value in self.prescribed_values.items()
        }
        if any(dof < 0 for dof in values):
            raise ValueError("constraint DOF ids must be non-negative")
        object.__setattr__(
            self,
            "prescribed_values",
            MappingProxyType(values),
        )

    def scaled(self, factor: float) -> "ConstraintSet":
        """Return the proportional target for one load-path factor."""

        scale = _finite(factor, "constraint scale")
        return ConstraintSet(
            {
                dof: value * scale
                for dof, value in self.prescribed_values.items()
            }
        )


def _finite(value: Any, label: str) -> float:
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{label} must be finite")
    return converted


__all__ = ["ConstraintSet"]
