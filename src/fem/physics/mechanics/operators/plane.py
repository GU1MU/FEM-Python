"""Validated properties shared by two-dimensional plane elements."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class PlaneProperties:
    """Normalized elastic section properties for a 2D plane element."""

    E: float
    nu: float
    thickness: float = 1.0
    plane_type: str = "stress"

    def __post_init__(self) -> None:
        E = _positive_finite(self.E, "E")
        nu = _finite(self.nu, "nu")
        if not -1.0 < nu < 0.5:
            raise ValueError("nu must satisfy -1 < nu < 0.5")
        thickness = _positive_finite(self.thickness, "thickness")
        plane_type = normalize_plane_type(self.plane_type)
        object.__setattr__(self, "E", E)
        object.__setattr__(self, "nu", nu)
        object.__setattr__(self, "thickness", thickness)
        object.__setattr__(self, "plane_type", plane_type)

    @classmethod
    def from_mapping(
        cls,
        properties: Mapping[str, Any],
        *,
        element_id: int | None = None,
        element_type: str | None = None,
        label: str = "plane element",
    ) -> "PlaneProperties":
        """Build normalized properties without mutating the source mapping."""

        if not isinstance(properties, Mapping):
            raise TypeError("plane element properties must be a mapping")
        if element_id is not None:
            label = f"{label} {int(element_id)}"
        try:
            E = properties["E"]
        except KeyError as exc:
            raise KeyError(f"{label} missing property E") from exc
        try:
            nu = properties["nu"]
        except KeyError as exc:
            raise KeyError(f"{label} missing property nu") from exc
        default_plane = (
            "strain"
            if str(element_type or "").strip().upper().startswith("CPE")
            else "stress"
        )
        try:
            return cls(
                E=E,
                nu=nu,
                thickness=properties.get("thickness", 1.0),
                plane_type=properties.get("plane_type", default_plane),
            )
        except (TypeError, ValueError) as exc:
            raise type(exc)(f"{label} properties invalid: {exc}") from exc

    def as_mapping(self) -> dict[str, Any]:
        """Return a detached normalized mapping for downstream adapters."""

        return {
            "E": self.E,
            "nu": self.nu,
            "thickness": self.thickness,
            "plane_type": self.plane_type,
        }


def normalize_plane_type(value: Any) -> str:
    """Normalize plane-stress/plane-strain aliases to ``stress``/``strain``."""

    if not isinstance(value, str):
        raise ValueError("plane_type must be 'stress' or 'strain'")
    normalized = value.strip().casefold()
    if normalized.startswith("plane_"):
        normalized = normalized[6:]
    if normalized.startswith("stress"):
        return "stress"
    if normalized.startswith("strain"):
        return "strain"
    raise ValueError("plane_type must be 'stress' or 'strain'")


def plane_thickness(
    properties: Mapping[str, Any],
    *,
    element_id: int | None = None,
    label: str = "plane element",
) -> float:
    """Resolve thickness for loads that do not need elastic parameters."""

    if not isinstance(properties, Mapping):
        raise TypeError("plane element properties must be a mapping")
    try:
        return _positive_finite(properties.get("thickness", 1.0), "thickness")
    except (TypeError, ValueError) as exc:
        if element_id is not None:
            label = f"{label} {int(element_id)}"
        raise type(exc)(f"{label} thickness invalid: {exc}") from exc


def _finite(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _positive_finite(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite and > 0") from exc
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and > 0")
    return result


__all__ = ["PlaneProperties", "normalize_plane_type", "plane_thickness"]
