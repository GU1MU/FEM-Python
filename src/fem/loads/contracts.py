"""Immutable external-load data used by compiled analyses."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any


@dataclass(frozen=True)
class ElementLoad:
    """Constant element load vector."""

    elem_id: int
    vector: tuple[float, ...]


@dataclass(frozen=True)
class ElementGravityLoad:
    """Gravity acceleration resolved to one element."""

    elem_id: int
    acceleration: tuple[float, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "elem_id", int(self.elem_id))
        object.__setattr__(self, "acceleration", tuple(self.acceleration))


@dataclass(frozen=True)
class SurfaceTraction:
    """Constant traction on one local element face."""

    elem_id: int
    local_index: int
    vector: tuple[float, ...]


@dataclass(frozen=True)
class EdgeTraction:
    """Constant traction on one local element edge."""

    elem_id: int
    local_index: int
    vector: tuple[float, ...]


@dataclass(frozen=True)
class LineElementLoad:
    """Resolved constant line-element load."""

    elem_id: int
    vector: tuple[float, ...]
    coordinate_system: str


@dataclass(frozen=True, slots=True)
class LoadSet:
    """Resolved external loads for one compiled analysis step."""

    nodal_forces: Mapping[int, float] = field(default_factory=dict)
    body_forces: tuple[ElementLoad, ...] = ()
    surface_tractions: tuple[SurfaceTraction, ...] = ()
    edge_tractions: tuple[EdgeTraction, ...] = ()
    gravity: tuple[float, ...] | None = None
    line_loads: tuple[LineElementLoad, ...] = ()
    element_gravities: tuple[ElementGravityLoad, ...] = ()

    def __post_init__(self) -> None:
        nodal = {
            int(dof): _finite(value, "nodal force")
            for dof, value in self.nodal_forces.items()
        }
        if any(dof < 0 for dof in nodal):
            raise ValueError("load DOF ids must be non-negative")
        gravity = (
            None
            if self.gravity is None
            else tuple(_finite(value, "gravity component") for value in self.gravity)
        )
        object.__setattr__(self, "nodal_forces", MappingProxyType(nodal))
        object.__setattr__(self, "body_forces", tuple(self.body_forces))
        object.__setattr__(self, "surface_tractions", tuple(self.surface_tractions))
        object.__setattr__(self, "edge_tractions", tuple(self.edge_tractions))
        object.__setattr__(self, "gravity", gravity)
        object.__setattr__(self, "line_loads", tuple(self.line_loads))
        object.__setattr__(self, "element_gravities", tuple(self.element_gravities))


def _finite(value: Any, label: str) -> float:
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{label} must be finite")
    return converted


__all__ = [
    "EdgeTraction",
    "ElementGravityLoad",
    "ElementLoad",
    "LineElementLoad",
    "LoadSet",
    "SurfaceTraction",
]
