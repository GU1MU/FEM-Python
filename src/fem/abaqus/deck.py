from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .errors import AbaqusSourceLocation


@dataclass(frozen=True, slots=True)
class AbaqusSourceSpan:
    """Logical source span assembled from one or more physical lines."""

    start: AbaqusSourceLocation
    end: AbaqusSourceLocation
    physical_locations: tuple[AbaqusSourceLocation, ...]

    @property
    def location(self) -> AbaqusSourceLocation:
        return self.start


@dataclass(frozen=True, slots=True)
class AbaqusKeywordOccurrence:
    """Raw keyword evidence retained for family-aware semantic auditing."""

    name: str
    params: tuple[tuple[str, str], ...]
    flags: tuple[str, ...]
    span: AbaqusSourceSpan
    raw_lines: tuple[str, ...] = ()

    @property
    def location(self) -> AbaqusSourceLocation:
        return self.span.start


@dataclass(frozen=True, slots=True)
class AbaqusDataRecordEvidence:
    """Presence, shape, and source evidence for one positional data record."""

    present: bool
    blank: bool
    field_count: int
    location: AbaqusSourceLocation | None
    fields: tuple[str, ...] = ()
    raw: str | None = None
    values: tuple[float, ...] = ()

    @classmethod
    def missing(cls) -> AbaqusDataRecordEvidence:
        return cls(False, False, 0, None)


@dataclass(frozen=True, slots=True)
class AbaqusNodeRecord:
    """Node coordinates plus any uninterpreted normal-component fields."""

    id: int
    coordinates: tuple[float, float, float]
    extra_fields: tuple[str, ...] = ()
    keyword_location: AbaqusSourceLocation | None = None
    location: AbaqusSourceLocation | None = None
    raw_fields: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class AbaqusSolidSectionData:
    """Raw homogeneous solid-section attribute with record presence evidence."""

    attribute: float | None
    record_present: bool
    blank: bool
    field_count: int
    location: AbaqusSourceLocation | None
    fields: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class AbaqusBeamSectionData:
    """Standard B31 profile geometry and approximate-n1 source evidence."""

    profile: str
    dimensions: tuple[float, ...]
    approximate_n1: tuple[float, float, float] | None
    geometry: AbaqusDataRecordEvidence
    orientation: AbaqusDataRecordEvidence


@dataclass(frozen=True)
class AbaqusElement:
    """Element record parsed from an input deck."""
    id: int
    node_ids: tuple[int, ...]
    type: str
    element_set: str | None = None
    keyword_location: AbaqusSourceLocation | None = None
    data_location: AbaqusSourceLocation | None = None
    raw_fields: tuple[str, ...] = ()


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
    elastic_records: list[AbaqusDataRecordEvidence] = field(
        default_factory=list
    )
    density_records: list[AbaqusDataRecordEvidence] = field(
        default_factory=list
    )
    elastic_keyword_count: int = 0
    density_keyword_count: int = 0


@dataclass(frozen=True)
class AbaqusSection:
    """Section assignment parsed from an Abaqus input deck."""
    element_set: str
    material: str
    section_type: str = "solid"
    element_ids: tuple[int, ...] = ()
    data: AbaqusSolidSectionData | AbaqusBeamSectionData | None = None
    keyword_location: AbaqusSourceLocation | None = None
    target_was_defined: bool = False


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
    keyword_location: AbaqusSourceLocation | None = None
    location: AbaqusSourceLocation | None = None
    raw_fields: tuple[str, ...] = ()
    keyword_params: tuple[tuple[str, str], ...] = ()
    keyword_flags: tuple[str, ...] = ()


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
    keyword_location: AbaqusSourceLocation | None = None
    procedure_present: bool = False
    procedure_location: AbaqusSourceLocation | None = None
    procedure_count: int = 0


@dataclass
class AbaqusDeck:
    """Parsed Abaqus input deck independent of FEM mesh classes."""
    name: str
    nodes: dict[int, tuple[float, float, float]] = field(default_factory=dict)
    node_records: dict[int, AbaqusNodeRecord] = field(default_factory=dict)
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
    keyword_occurrences: list[AbaqusKeywordOccurrence] = field(
        default_factory=list
    )


__all__ = [
    "AbaqusBeamSectionData",
    "AbaqusBoundary",
    "AbaqusCload",
    "AbaqusDataRecordEvidence",
    "AbaqusDeck",
    "AbaqusDistributedLoad",
    "AbaqusElement",
    "AbaqusKeywordOccurrence",
    "AbaqusMaterial",
    "AbaqusNodeRecord",
    "AbaqusOutputRequest",
    "AbaqusSection",
    "AbaqusSolidSectionData",
    "AbaqusSourceSpan",
    "AbaqusStep",
    "AbaqusSurfaceFace",
]
