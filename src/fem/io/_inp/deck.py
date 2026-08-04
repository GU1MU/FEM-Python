from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field, fields as dataclass_fields
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


@dataclass(frozen=True, slots=True, order=True)
class AbaqusElementEndIdentity:
    """One B31 structural node at a source-defined local element end."""

    element_id: int
    local_end: int
    node_id: int

    def __post_init__(self) -> None:
        element_id = int(self.element_id)
        local_end = int(self.local_end)
        node_id = int(self.node_id)
        if local_end not in (1, 2):
            raise ValueError("B31 local_end must be 1 or 2")
        object.__setattr__(self, "element_id", element_id)
        object.__setattr__(self, "local_end", local_end)
        object.__setattr__(self, "node_id", node_id)


@dataclass(frozen=True, slots=True)
class AbaqusNodeNormalRecord:
    """Typed n2 source attached to one ``*NODE`` record."""

    vector: tuple[float, float, float]
    node_id: int
    span: AbaqusSourceSpan | None = None
    fields: tuple[str, ...] = ()
    raw: str | None = None

    def __post_init__(self) -> None:
        vector = tuple(float(value) for value in self.vector)
        if len(vector) != 3:
            raise ValueError("node normal must contain exactly three components")
        node_id = int(self.node_id)
        object.__setattr__(self, "vector", vector)
        object.__setattr__(self, "node_id", node_id)
        object.__setattr__(self, "fields", tuple(str(value) for value in self.fields))

    @property
    def components(self) -> tuple[float, float, float]:
        """Return the normal components under the source terminology."""

        return self.vector

    @property
    def source_kind(self) -> str:
        return "node"

    @property
    def location(self) -> AbaqusSourceLocation | None:
        return None if self.span is None else self.span.location

    @property
    def source_span(self) -> AbaqusSourceSpan | None:
        return self.span


@dataclass(frozen=True, slots=True)
class AbaqusNormalRecord:
    """One ``*NORMAL, TYPE=ELEMENT`` source record.

    ``element`` and ``node`` intentionally retain their original scalar or set
    identity.  ``identities`` is filled by the adapter's validation pass after
    set expansion, so the parser never has to discard the source targets.
    """

    element: int | str
    node: int | str
    normal: tuple[float, float, float]
    span: AbaqusSourceSpan
    raw_fields: tuple[str, ...] = ()
    raw: str | None = None
    keyword_location: AbaqusSourceLocation | None = None
    identities: tuple[AbaqusElementEndIdentity, ...] = ()

    def __post_init__(self) -> None:
        normal = tuple(float(value) for value in self.normal)
        if len(normal) != 3:
            raise ValueError("*NORMAL must contain exactly three components")
        object.__setattr__(self, "normal", normal)
        object.__setattr__(self, "raw_fields", tuple(str(value) for value in self.raw_fields))
        object.__setattr__(
            self,
            "identities",
            tuple(sorted(self.identities)),
        )

    @property
    def element_target(self) -> int | str:
        return self.element

    @property
    def node_target(self) -> int | str:
        return self.node

    @property
    def element_id(self) -> int | None:
        return self.element if isinstance(self.element, int) else None

    @property
    def node_id(self) -> int | None:
        return self.node if isinstance(self.node, int) else None

    @property
    def element_end_identities(self) -> tuple[AbaqusElementEndIdentity, ...]:
        return self.identities

    @property
    def identity(self) -> AbaqusElementEndIdentity | None:
        return self.identities[0] if len(self.identities) == 1 else None

    @property
    def local_end(self) -> int | None:
        identity = self.identity
        return None if identity is None else identity.local_end

    @property
    def vector(self) -> tuple[float, float, float]:
        return self.normal

    @property
    def location(self) -> AbaqusSourceLocation:
        return self.span.location

    @property
    def source_span(self) -> AbaqusSourceSpan:
        return self.span

    @property
    def source_kind(self) -> str:
        return "normal"


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
    normal: AbaqusNodeNormalRecord | None = None

    @property
    def normal_components(self) -> tuple[float, float, float]:
        """Return typed node normal components, or no complete normal."""

        return () if self.normal is None else self.normal.vector

    @property
    def normal_source(self) -> AbaqusNodeNormalRecord | None:
        return self.normal


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
    orientation_node_id: int | None = None

    @property
    def structural_node_ids(self) -> tuple[int, ...]:
        """Return connectivity nodes that own structural DOFs.

        The fallback keeps manually constructed pre-Phase-3 DTOs usable while
        parser-produced B31 records store the extra node explicitly.
        """

        if (
            str(self.type).upper() == "B31"
            and self.orientation_node_id is None
            and len(self.node_ids) == 3
        ):
            return tuple(int(value) for value in self.node_ids[:2])
        return tuple(int(value) for value in self.node_ids)

    @property
    def additional_orientation_node_id(self) -> int | None:
        if self.orientation_node_id is not None:
            return int(self.orientation_node_id)
        if str(self.type).upper() == "B31" and len(self.node_ids) == 3:
            return int(self.node_ids[2])
        return None

    @property
    def additional_orientation_node(self) -> int | None:
        return self.additional_orientation_node_id

    @property
    def orientation_node(self) -> int | None:
        return self.additional_orientation_node_id


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
    parent_parameters: tuple[tuple[str, str], ...] = ()
    parent_flags: tuple[str, ...] = ()
    child_parameters: tuple[tuple[str, str], ...] = ()
    child_flags: tuple[str, ...] = ()


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
    normal_records: list[AbaqusNormalRecord] = field(default_factory=list)

    @property
    def normals(self) -> list[AbaqusNormalRecord]:
        """Compatibility spelling for the typed ``*NORMAL`` records."""

        return self.normal_records

    def snapshot(self) -> AbaqusDeck:
        """Return a detached, owned deck with deterministic map ordering."""

        return deepcopy(self)

    def __deepcopy__(self, memo: dict[int, Any]) -> AbaqusDeck:
        """Copy parser state without sharing mutable containers.

        Source lists retain their physical order because that is provenance;
        maps are rebuilt in stable key order so background-task snapshots do
        not depend on incidental insertion order.
        """

        existing = memo.get(id(self))
        if existing is not None:
            return existing
        clone = object.__new__(type(self))
        memo[id(self)] = clone
        for item in dataclass_fields(self):
            setattr(clone, item.name, deepcopy(getattr(self, item.name), memo))

        for name in (
            "nodes",
            "node_records",
            "node_sets",
            "node_set_scopes",
            "element_sets",
            "element_set_scopes",
            "surfaces",
            "surface_scopes",
            "materials",
        ):
            value = getattr(clone, name)
            setattr(
                clone,
                name,
                dict(sorted(value.items(), key=lambda item: (type(item[0]).__name__, repr(item[0])))),
            )
        return clone


__all__ = [
    "AbaqusBeamSectionData",
    "AbaqusBoundary",
    "AbaqusCload",
    "AbaqusDataRecordEvidence",
    "AbaqusDeck",
    "AbaqusDistributedLoad",
    "AbaqusElementEndIdentity",
    "AbaqusElement",
    "AbaqusKeywordOccurrence",
    "AbaqusMaterial",
    "AbaqusNodeNormalRecord",
    "AbaqusNodeRecord",
    "AbaqusNormalRecord",
    "AbaqusOutputRequest",
    "AbaqusSection",
    "AbaqusSolidSectionData",
    "AbaqusSourceSpan",
    "AbaqusStep",
    "AbaqusSurfaceFace",
]
