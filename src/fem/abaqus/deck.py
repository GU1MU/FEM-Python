from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class AbaqusElement:
    """Element record parsed from an input deck."""
    id: int
    node_ids: tuple[int, ...]
    type: str
    element_set: str | None = None


@dataclass(frozen=True)
class AbaqusSurfaceFace:
    """Surface entry referencing an element or element set face."""
    target: str | int
    face_label: str


@dataclass
class AbaqusMaterial:
    """Material properties parsed from Abaqus keywords."""
    name: str
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AbaqusSection:
    """Section assignment parsed from an Abaqus input deck."""
    element_set: str
    material: str
    section_type: str = "solid"
    element_ids: tuple[int, ...] = ()


@dataclass(frozen=True)
class AbaqusBoundary:
    """Raw boundary line using Abaqus target and component notation."""
    target: str | int
    first_component: int | str
    last_component: int | None = None
    value: float = 0.0


@dataclass(frozen=True)
class AbaqusCload:
    """Raw concentrated nodal load line."""
    target: str | int
    component: int
    value: float


@dataclass(frozen=True)
class AbaqusDistributedLoad:
    """Raw distributed load from DLOAD or DSLOAD."""
    target: str | int | None
    label: str
    magnitude: float
    source: str
    extra: tuple[float, ...] = ()


@dataclass(frozen=True)
class AbaqusOutputRequest:
    """Raw output request parsed from a step."""
    kind: str
    target: str
    variables: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AbaqusStep:
    """Abaqus analysis step data kept before model construction."""
    name: str
    procedure: str = "static"
    boundaries: list[AbaqusBoundary] = field(default_factory=list)
    cloads: list[AbaqusCload] = field(default_factory=list)
    distributed_loads: list[AbaqusDistributedLoad] = field(default_factory=list)
    output_requests: list[AbaqusOutputRequest] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AbaqusDeck:
    """Parsed Abaqus input deck independent of FEM mesh classes."""
    name: str
    nodes: dict[int, tuple[float, float, float]] = field(default_factory=dict)
    elements: list[AbaqusElement] = field(default_factory=list)
    node_sets: dict[str, list[int]] = field(default_factory=dict)
    node_set_scopes: dict[str, str] = field(default_factory=dict)
    element_sets: dict[str, list[int]] = field(default_factory=dict)
    element_set_scopes: dict[str, str] = field(default_factory=dict)
    surfaces: dict[str, list[AbaqusSurfaceFace]] = field(default_factory=dict)
    surface_scopes: dict[str, str] = field(default_factory=dict)
    materials: dict[str, AbaqusMaterial] = field(default_factory=dict)
    sections: list[AbaqusSection] = field(default_factory=list)
    steps: list[AbaqusStep] = field(default_factory=list)
