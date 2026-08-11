"""Pure B31 element-end orientation resolution.

The resolver in this module is deliberately detached from the FEM model and
from Abaqus/CAE.  It consumes source deck data (or an equivalent detached
topology) and returns a typed field for every structural B31 element end.

The semantic rules are concentrated in :class:`AbaqusOrientationPolicy`.
They follow the Abaqus Elements Guide, ``Beam Element Cross-Section
Orientation`` and the Analysis User's Guide, ``Normal Definitions at Nodes``:

* the local system is the right-handed ``(t, n1, n2)`` system;
* an additional connectivity node takes precedence over section ``n1``;
* a user-specified ``*NORMAL`` takes precedence over a node-defined normal;
* generated normals at a node are grouped within 20 degrees; groups of at
  most 30 remaining elements are averaged only when the final group is a
  pairwise 20-degree group; and
* a node may retain multiple generated normal groups.

Primary references:

* https://docs.software.vt.edu/abaqusv2025/English/SIMACAEELMRefMap/simaelm-c-beamcrosssection.htm
* https://docs.software.vt.edu/abaqusv2024/English/SIMACAEMODRefMap/simamod-c-nodalnormals.htm
* https://docs.software.vt.edu/abaqusv2024/English/SIMACAEKEYRefMap/simakey-r-normal.htm
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from math import cos, isfinite, radians
from types import MappingProxyType
from typing import Any, Literal

import numpy as np

from ...elements import (
    BEAM_DEFAULT_LOCAL_Y_REFERENCE,
    BEAM_ORIENTATION_PARALLEL_TOLERANCE,
)
from .deck import (
    AbaqusBeamSectionData,
    AbaqusDeck,
    AbaqusElement,
    AbaqusElementEndIdentity,
    AbaqusNodeRecord,
    AbaqusNormalRecord,
    AbaqusSection,
)
from .errors import (
    AbaqusBuildError,
    AbaqusSourceLocation,
    UnsupportedAbaqusFeatureError,
)


Vector3 = tuple[float, float, float]
OrientationResolutionKind = Literal[
    "explicit",
    "default-generated",
    "averaged",
    "split-group",
]


@dataclass(frozen=True, slots=True)
class AbaqusOrientationPolicy:
    """Central numerical and grouping policy for B31 normal resolution.

    ``20`` degrees and ``30`` remaining elements are stated by the Abaqus
    normal-averaging algorithm.  The parallel and validation tolerances are
    the numerical contract used by the existing Beam2 frame validator; they
    are kept here so the adapter does not make a second independent choice.
    """

    continuity_angle_degrees: float = 20.0
    max_averaging_elements: int = 30
    nonparallel_tolerance: float = BEAM_ORIENTATION_PARALLEL_TOLERANCE
    unit_tolerance: float = 1.0e-10
    comparison_tolerance: float = 1.0e-10
    frame_tolerance: float = 1.0e-8

    def __post_init__(self) -> None:
        angle = float(self.continuity_angle_degrees)
        maximum = int(self.max_averaging_elements)
        nonparallel = float(self.nonparallel_tolerance)
        unit = float(self.unit_tolerance)
        comparison = float(self.comparison_tolerance)
        frame = float(self.frame_tolerance)
        if not isfinite(angle) or not 0.0 < angle < 90.0:
            raise ValueError("orientation continuity angle must be in (0, 90)")
        if maximum < 1:
            raise ValueError("orientation averaging maximum must be positive")
        if not all(
            isfinite(value) and value > 0.0
            for value in (nonparallel, unit, comparison, frame)
        ):
            raise ValueError("orientation tolerances must be finite and positive")
        object.__setattr__(self, "continuity_angle_degrees", angle)
        object.__setattr__(self, "max_averaging_elements", maximum)
        object.__setattr__(self, "nonparallel_tolerance", nonparallel)
        object.__setattr__(self, "unit_tolerance", unit)
        object.__setattr__(self, "comparison_tolerance", comparison)
        object.__setattr__(self, "frame_tolerance", frame)

    @property
    def continuity_cosine(self) -> float:
        """Cosine threshold for the official 20-degree normal grouping."""

        return cos(radians(self.continuity_angle_degrees))


DEFAULT_ABAQUS_ORIENTATION_POLICY = AbaqusOrientationPolicy()
# A shorter name makes the single policy easy to discover in adapter code.
ABAQUS_ORIENTATION_POLICY = DEFAULT_ABAQUS_ORIENTATION_POLICY


@dataclass(frozen=True, slots=True, order=True)
class AbaqusNormalGroupIdentity:
    """Stable identity for one generated normal group at one node."""

    node_id: int
    group_index: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "node_id", int(self.node_id))
        object.__setattr__(self, "group_index", int(self.group_index))
        if self.group_index < 0:
            raise ValueError("normal group index must be nonnegative")

    @property
    def index(self) -> int:
        """Compatibility spelling for the stable group ordinal."""

        return self.group_index


@dataclass(frozen=True, slots=True)
class AbaqusOrientationTopology:
    """Detached, owned topology used by the resolver."""

    nodes: Mapping[int, Vector3]
    elements: tuple[AbaqusElement, ...]
    node_records: Mapping[int, AbaqusNodeRecord]
    element_sets: Mapping[str, tuple[int, ...]]
    node_sets: Mapping[str, tuple[int, ...]]
    sections: tuple[AbaqusSection, ...]
    normal_records: tuple[AbaqusNormalRecord, ...] = ()

    @classmethod
    def from_deck(cls, deck: AbaqusDeck) -> AbaqusOrientationTopology:
        return cls(
            nodes=deck.nodes,
            elements=tuple(deck.elements),
            node_records=deck.node_records,
            element_sets=deck.element_sets,
            node_sets=deck.node_sets,
            sections=tuple(deck.sections),
            normal_records=tuple(deck.normal_records),
        )

    def __post_init__(self) -> None:
        nodes = {
            int(node_id): _as_vector(coordinates, label="node coordinates")
            for node_id, coordinates in self.nodes.items()
        }
        node_records = {
            int(node_id): record
            for node_id, record in self.node_records.items()
        }
        element_sets = {
            str(name): tuple(sorted({int(value) for value in values}))
            for name, values in self.element_sets.items()
        }
        node_sets = {
            str(name): tuple(sorted({int(value) for value in values}))
            for name, values in self.node_sets.items()
        }
        object.__setattr__(self, "nodes", MappingProxyType(dict(sorted(nodes.items()))))
        object.__setattr__(
            self,
            "node_records",
            MappingProxyType(dict(sorted(node_records.items()))),
        )
        object.__setattr__(
            self,
            "element_sets",
            MappingProxyType(dict(sorted(element_sets.items()))),
        )
        object.__setattr__(
            self,
            "node_sets",
            MappingProxyType(dict(sorted(node_sets.items()))),
        )
        object.__setattr__(self, "elements", tuple(self.elements))
        object.__setattr__(self, "sections", tuple(self.sections))
        object.__setattr__(
            self,
            "normal_records",
            tuple(sorted(self.normal_records, key=_normal_record_key)),
        )


@dataclass(frozen=True, slots=True)
class AbaqusElementEndOrientation:
    """One validated effective frame at a source-defined B31 element end."""

    identity: AbaqusElementEndIdentity
    tangent: Vector3
    n1: Vector3
    normal: Vector3
    approximate_n1: Vector3
    resolution_kind: OrientationResolutionKind
    reference_source: Literal[
        "orientation-node", "section-n1", "default-n1"
    ]
    normal_source: Literal[
        "element-normal", "node-normal", "generated-normal", "averaged-normal"
    ]
    normal_group: AbaqusNormalGroupIdentity | None = None
    source_locations: tuple[AbaqusSourceLocation, ...] = ()
    frame_tolerance: float = DEFAULT_ABAQUS_ORIENTATION_POLICY.frame_tolerance

    def __post_init__(self) -> None:
        object.__setattr__(self, "tangent", _as_vector(self.tangent, label="tangent"))
        object.__setattr__(self, "n1", _as_vector(self.n1, label="n1"))
        object.__setattr__(self, "normal", _as_vector(self.normal, label="n2"))
        object.__setattr__(
            self,
            "approximate_n1",
            _as_vector(self.approximate_n1, label="approximate n1"),
        )
        object.__setattr__(self, "source_locations", tuple(self.source_locations))
        frame_tolerance = float(self.frame_tolerance)
        if not isfinite(frame_tolerance) or frame_tolerance <= 0.0:
            raise ValueError("frame tolerance must be finite and positive")
        object.__setattr__(self, "frame_tolerance", frame_tolerance)
        _validate_right_handed_frame(
            self.tangent,
            self.n1,
            self.normal,
            tolerance=frame_tolerance,
        )

    @property
    def element_id(self) -> int:
        return self.identity.element_id

    @property
    def local_end(self) -> int:
        return self.identity.local_end

    @property
    def node_id(self) -> int:
        return self.identity.node_id

    @property
    def n2(self) -> Vector3:
        """Return the effective second beam-section direction."""

        return self.normal

    @property
    def effective_n1(self) -> Vector3:
        return self.n1

    @property
    def effective_n2(self) -> Vector3:
        return self.normal

    @property
    def reference(self) -> Vector3:
        """Return the value consumed as the Beam2 local-y reference."""

        return self.n1

    @property
    def source_kind(self) -> OrientationResolutionKind:
        return self.resolution_kind

    @property
    def group_id(self) -> AbaqusNormalGroupIdentity | None:
        return self.normal_group


@dataclass(frozen=True, slots=True)
class AbaqusElementEndOrientationField:
    """Stable element-end orientation entries sorted by exact identity."""

    entries: tuple[AbaqusElementEndOrientation, ...] = ()

    def __post_init__(self) -> None:
        ordered = tuple(sorted(self.entries, key=lambda item: item.identity))
        if len({item.identity for item in ordered}) != len(ordered):
            raise ValueError("orientation field contains duplicate element ends")
        object.__setattr__(self, "entries", ordered)

    @property
    def orientations(self) -> tuple[AbaqusElementEndOrientation, ...]:
        return self.entries

    @property
    def element_end_orientations(self) -> tuple[AbaqusElementEndOrientation, ...]:
        return self.entries

    @property
    def by_identity(self) -> Mapping[AbaqusElementEndIdentity, AbaqusElementEndOrientation]:
        return MappingProxyType({item.identity: item for item in self.entries})

    def for_identity(
        self,
        identity: AbaqusElementEndIdentity,
    ) -> AbaqusElementEndOrientation | None:
        return self.by_identity.get(identity)

    def for_element(self, element_id: int) -> tuple[AbaqusElementEndOrientation, ...]:
        return tuple(
            item for item in self.entries if item.element_id == int(element_id)
        )

    def constant_reference(
        self,
        element_id: int,
        *,
        tolerance: float | None = None,
    ) -> Vector3 | None:
        """Return one Beam2 reference when both source ends reduce to it."""

        entries = self.for_element(element_id)
        if len(entries) != 2:
            return None
        first = np.asarray(entries[0].n1, dtype=float)
        resolved_tolerance = (
            DEFAULT_ABAQUS_ORIENTATION_POLICY.comparison_tolerance
            if tolerance is None
            else float(tolerance)
        )
        if not np.allclose(
            first,
            np.asarray(entries[1].n1, dtype=float),
            rtol=resolved_tolerance,
            atol=resolved_tolerance,
        ):
            return None
        return entries[0].n1

    def varies_by_element(self, *, tolerance: float | None = None) -> tuple[int, ...]:
        element_ids = sorted({item.element_id for item in self.entries})
        return tuple(
            element_id
            for element_id in element_ids
            if self.constant_reference(element_id, tolerance=tolerance) is None
            and len(self.for_element(element_id)) == 2
        )


@dataclass(frozen=True, slots=True)
class AbaqusOrientationGroup:
    """One final generated normal group at a node."""

    identity: AbaqusNormalGroupIdentity
    identities: tuple[AbaqusElementEndIdentity, ...]
    candidate_normals: tuple[Vector3, ...]
    effective_normal: Vector3
    averaged: bool
    split_reason: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "identities", tuple(sorted(self.identities)))
        object.__setattr__(
            self,
            "candidate_normals",
            tuple(_as_vector(value, label="candidate normal") for value in self.candidate_normals),
        )
        object.__setattr__(
            self,
            "effective_normal",
            _as_vector(self.effective_normal, label="effective group normal"),
        )

    @property
    def group_id(self) -> AbaqusNormalGroupIdentity:
        return self.identity

    @property
    def node_id(self) -> int:
        return self.identity.node_id


@dataclass(frozen=True, slots=True)
class AbaqusOrientationReportEntry:
    """One successful resolution event or diagnostic evidence item."""

    kind: str
    code: str
    message: str
    identities: tuple[AbaqusElementEndIdentity, ...] = ()
    locations: tuple[AbaqusSourceLocation, ...] = ()
    normal_group: AbaqusNormalGroupIdentity | None = None
    record: Any = None
    remediation: str | None = None
    severity: Literal["info", "error", "capability"] = "info"

    def __post_init__(self) -> None:
        object.__setattr__(self, "identities", tuple(sorted(self.identities)))
        object.__setattr__(self, "locations", _stable_locations(self.locations))

    @property
    def source_locations(self) -> tuple[AbaqusSourceLocation, ...]:
        return self.locations

    @property
    def group_id(self) -> AbaqusNormalGroupIdentity | None:
        return self.normal_group


# This name is useful to callers that use “diagnostic” for all report values.
AbaqusOrientationDiagnostic = AbaqusOrientationReportEntry


@dataclass(frozen=True, slots=True)
class AbaqusOrientationResolutionReport:
    """Typed provenance, grouping, and failure report for one resolution."""

    events: tuple[AbaqusOrientationReportEntry, ...] = ()
    diagnostics: tuple[AbaqusOrientationReportEntry, ...] = ()
    groups: tuple[AbaqusOrientationGroup, ...] = ()
    policy: AbaqusOrientationPolicy = DEFAULT_ABAQUS_ORIENTATION_POLICY

    def __post_init__(self) -> None:
        object.__setattr__(self, "events", tuple(self.events))
        object.__setattr__(self, "diagnostics", tuple(self.diagnostics))
        object.__setattr__(self, "groups", tuple(self.groups))

    @property
    def entries(self) -> tuple[AbaqusOrientationReportEntry, ...]:
        return self.events

    @property
    def issues(self) -> tuple[AbaqusOrientationReportEntry, ...]:
        return self.diagnostics

    @property
    def ok(self) -> bool:
        return not self.diagnostics

    @property
    def passed(self) -> bool:
        return self.ok

    @property
    def has_errors(self) -> bool:
        return any(item.severity == "error" for item in self.diagnostics)

    @property
    def has_capability_diagnostics(self) -> bool:
        return any(item.severity == "capability" for item in self.diagnostics)

    def of_kind(self, kind: str) -> tuple[AbaqusOrientationReportEntry, ...]:
        return tuple(item for item in self.events if item.kind == kind)

    @property
    def explicit(self) -> tuple[AbaqusOrientationReportEntry, ...]:
        return self.of_kind("explicit")

    @property
    def default_generated(self) -> tuple[AbaqusOrientationReportEntry, ...]:
        return self.of_kind("default-generated")

    @property
    def averaged(self) -> tuple[AbaqusOrientationReportEntry, ...]:
        return self.of_kind("averaged")

    @property
    def split_groups(self) -> tuple[AbaqusOrientationReportEntry, ...]:
        return self.of_kind("split-group")

    @property
    def conflicts(self) -> tuple[AbaqusOrientationReportEntry, ...]:
        return tuple(item for item in self.diagnostics if item.kind == "conflict")

    @property
    def unsupported_variations(self) -> tuple[AbaqusOrientationReportEntry, ...]:
        return tuple(
            item
            for item in self.diagnostics
            if item.kind == "unsupported-variation"
        )

    def raise_if_invalid(self) -> None:
        """Raise the first deterministic typed diagnostic, if any."""

        if not self.diagnostics:
            return
        diagnostic = self.diagnostics[0]
        error_type = (
            UnsupportedAbaqusFeatureError
            if diagnostic.severity == "capability"
            or diagnostic.kind == "unsupported-variation"
            else AbaqusBuildError
        )
        raise error_type(
            diagnostic.message,
            code=diagnostic.code,
            location=diagnostic.locations[0] if diagnostic.locations else None,
            locations=diagnostic.locations,
            record=diagnostic.record,
            remediation=diagnostic.remediation,
        )


@dataclass(frozen=True, slots=True)
class AbaqusOrientationResolution:
    """The pure resolver result: field plus its complete report."""

    field: AbaqusElementEndOrientationField
    report: AbaqusOrientationResolutionReport

    @property
    def orientation_field(self) -> AbaqusElementEndOrientationField:
        return self.field

    @property
    def resolution_report(self) -> AbaqusOrientationResolutionReport:
        return self.report

    @property
    def entries(self) -> tuple[AbaqusElementEndOrientation, ...]:
        return self.field.entries

    @property
    def passed(self) -> bool:
        return self.report.passed

    def raise_if_invalid(self) -> None:
        self.report.raise_if_invalid()


@dataclass(frozen=True, slots=True)
class _EndpointBase:
    identity: AbaqusElementEndIdentity
    tangent: Vector3
    approximate_n1: Vector3
    reference_source: Literal[
        "orientation-node", "section-n1", "default-n1"
    ]
    source_locations: tuple[AbaqusSourceLocation, ...]
    generated_normal: Vector3


def resolve_b31_orientations(
    deck: AbaqusDeck,
    topology: AbaqusOrientationTopology | Mapping[str, Any] | Any | None = None,
    *,
    policy: AbaqusOrientationPolicy = DEFAULT_ABAQUS_ORIENTATION_POLICY,
    strict: bool = False,
    _source_is_owned: bool = False,
) -> AbaqusOrientationResolution:
    """Resolve all structural B31 element ends from detached source data.

    The default mode returns a field and a report even when the report carries
    a conflict or a Phase 5 capability diagnostic.  ``strict=True`` is the
    builder-facing convenience that raises the first deterministic diagnostic.
    """

    if (
        topology is None
        and isinstance(deck, AbaqusDeck)
        and not any(
            str(element.type).upper() == "B31"
            for element in deck.elements
        )
        and not deck.normal_records
        and not any(
            record.extra_fields or record.normal is not None
            for record in deck.node_records.values()
        )
    ):
        result = _make_resolution((), (), (), (), policy)
        if strict:
            result.raise_if_invalid()
        return result

    source_deck = (
        deck
        if _source_is_owned or not isinstance(deck, AbaqusDeck)
        else deck.snapshot()
    )
    detached = _coerce_topology(source_deck, topology)
    events: list[AbaqusOrientationReportEntry] = []
    diagnostics: list[AbaqusOrientationReportEntry] = []
    groups: list[AbaqusOrientationGroup] = []

    elements = tuple(
        sorted(
            (
                element
                for element in detached.elements
                if str(element.type).upper() == "B31"
            ),
            key=_element_key,
        )
    )
    element_lookup: dict[int, AbaqusElement] = {}
    for element in elements:
        element_id = int(element.id)
        previous = element_lookup.get(element_id)
        if previous is not None and previous != element:
            diagnostics.append(
                _diagnostic(
                    "conflict",
                    "abaqus.b31.element_duplicate",
                    f"B31 element {element_id} has multiple source records",
                    locations=_source_locations_for_element(element, previous),
                    record=element_id,
                    remediation="Define one source record for each B31 element.",
                )
            )
        else:
            element_lookup[element_id] = element

    if not elements:
        diagnostics.extend(_normal_without_b31_diagnostics(detached))
        if any(record.extra_fields or record.normal is not None for record in detached.node_records.values()):
            record = next(
                item
                for item in detached.node_records.values()
                if item.extra_fields or item.normal is not None
            )
            diagnostics.append(
                _diagnostic(
                    "unsupported",
                    "abaqus.line.nodal_normals_unsupported",
                    "nodal normal components require a B31 structural element",
                    locations=_node_locations(detached, record.id),
                    record=record.extra_fields or record.normal.vector,
                    remediation="Use nodal normals only with supported B31 elements.",
                )
            )
        result = _make_resolution((), events, diagnostics, groups, policy)
        if strict:
            result.raise_if_invalid()
        return result

    node_normals = _resolve_node_normals(detached, diagnostics)
    explicit_normals = _resolve_element_normals(
        detached,
        element_lookup,
        diagnostics,
        policy,
    )

    endpoint_bases: dict[AbaqusElementEndIdentity, _EndpointBase] = {}
    generated_candidates: dict[int, list[tuple[_EndpointBase, Vector3]]] = {}

    for element in elements:
        element_id = int(element.id)
        structural = tuple(int(value) for value in element.structural_node_ids)
        if len(structural) != 2:
            diagnostics.append(
                _diagnostic(
                    "invalid",
                    "abaqus.line.connectivity_shape",
                    "B31 connectivity must contain exactly two structural nodes",
                    locations=_source_locations_for_element(element),
                    record=element.raw_fields or element.node_ids,
                )
            )
            continue

        coordinates: list[np.ndarray] = []
        missing_node = False
        for node_id in structural:
            coordinate = _node_coordinates(detached, node_id)
            if coordinate is None:
                diagnostics.append(
                    _diagnostic(
                        "invalid",
                        "abaqus.b31.node_missing",
                        f"B31 element {element_id} references missing node {node_id}",
                        identities=(
                            AbaqusElementEndIdentity(
                                element_id,
                                1 if node_id == structural[0] else 2,
                                node_id,
                            ),
                        ),
                        locations=_source_locations_for_element(element),
                        record={"element": element_id, "node": node_id},
                        remediation="Define both structural B31 nodes.",
                    )
                )
                missing_node = True
                break
            coordinates.append(np.asarray(coordinate, dtype=float))
        if missing_node:
            continue

        tangent_raw = coordinates[1] - coordinates[0]
        tangent_norm = float(np.linalg.norm(tangent_raw))
        if (
            tangent_raw.shape != (3,)
            or not np.all(np.isfinite(tangent_raw))
            or not isfinite(tangent_norm)
            or tangent_norm <= 0.0
        ):
            diagnostics.append(
                _diagnostic(
                    "invalid",
                    "abaqus.b31.geometry_invalid",
                    f"B31 element {element_id} has invalid structural geometry",
                    locations=_source_locations_for_element(element),
                    record=structural,
                )
            )
            continue
        tangent = tangent_raw / tangent_norm
        base_reference, reference_source, reference_locations = _base_reference(
            detached,
            element,
            coordinates[0],
            diagnostics,
        )
        if base_reference is None:
            continue
        base_norm = float(np.linalg.norm(base_reference))
        if (
            base_reference.shape != (3,)
            or not np.all(np.isfinite(base_reference))
            or not isfinite(base_norm)
            or base_norm <= 0.0
        ):
            diagnostics.append(
                _diagnostic(
                    "invalid",
                    "beam.orientation.invalid",
                    f"B31 element {element_id} n1 reference must be finite and nonzero",
                    locations=_stable_locations(
                        (*_source_locations_for_element(element), *reference_locations)
                    ),
                    record={
                        "element": element_id,
                        "nodes": structural,
                        "reference": tuple(float(value) for value in base_reference),
                    },
                )
            )
            continue
        base_projected = base_reference - float(base_reference @ tangent) * tangent
        base_projected_norm = float(np.linalg.norm(base_projected))
        if (
            not isfinite(base_projected_norm)
            or base_projected_norm <= policy.nonparallel_tolerance
        ):
            diagnostics.append(
                _diagnostic(
                    "invalid",
                    "beam.orientation.parallel",
                    f"B31 element {element_id} n1 reference is parallel to its tangent",
                    locations=_stable_locations(
                        (*_source_locations_for_element(element), *reference_locations)
                    ),
                    record={
                        "element": element_id,
                        "nodes": structural,
                        "reference": tuple(float(value) for value in base_reference),
                        "tangent": tuple(float(value) for value in tangent),
                    },
                )
            )
            continue
        approximate_n1 = base_projected / base_projected_norm
        generated_normal_raw = np.cross(tangent, approximate_n1)
        generated_normal = _unit_or_none(generated_normal_raw, policy)
        if generated_normal is None:
            diagnostics.append(
                _diagnostic(
                    "invalid",
                    "abaqus.b31.orientation_invalid",
                    f"B31 element {element_id} cannot form a right-handed frame",
                    locations=_source_locations_for_element(element),
                    record=structural,
                )
            )
            continue

        for local_end, node_id in enumerate(structural, start=1):
            identity = AbaqusElementEndIdentity(element_id, local_end, node_id)
            base = _EndpointBase(
                identity=identity,
                tangent=tuple(float(value) for value in tangent),
                approximate_n1=tuple(float(value) for value in approximate_n1),
                reference_source=reference_source,
                source_locations=_stable_locations(
                    (*_source_locations_for_element(element), *reference_locations)
                ),
                generated_normal=tuple(float(value) for value in generated_normal),
            )
            endpoint_bases[identity] = base
            explicit = explicit_normals.get(identity)
            if explicit is not None:
                continue
            node_normal = node_normals.get(node_id)
            if node_normal is not None:
                continue
            generated_candidates.setdefault(node_id, []).append(
                (base, base.generated_normal)
            )

    resolved: dict[AbaqusElementEndIdentity, AbaqusElementEndOrientation] = {}
    # Explicit and node-defined normals are never included in generated
    # averaging groups, exactly as required by the Abaqus “remaining” rule.
    for identity, base in sorted(endpoint_bases.items()):
        if identity in explicit_normals or identity.node_id in node_normals:
            source = explicit_normals.get(identity)
            if source is None:
                source_locations = base.source_locations
                normal = node_normals[identity.node_id][0]
                normal_locations = node_normals[identity.node_id][1]
                normal_source: Literal["element-normal", "node-normal"] = "node-normal"
            else:
                source_locations = base.source_locations
                normal = source[0]
                normal_locations = source[1]
                normal_source = "element-normal"
            orientation = _make_from_normal(
                base,
                normal,
                normal_source=normal_source,
                resolution_kind="explicit",
                locations=(*source_locations, *normal_locations),
                diagnostics=diagnostics,
                policy=policy,
            )
            if orientation is not None:
                resolved[identity] = orientation
                events.append(
                    _event(
                        "explicit",
                        "abaqus.b31.orientation.explicit",
                        (
                            "explicit *NORMAL resolved for one element end"
                            if source is not None
                            else "node-defined normal resolved for one element end"
                        ),
                        identity,
                        locations=orientation.source_locations,
                    )
                )

    for node_id in sorted(generated_candidates):
        candidates = sorted(
            generated_candidates[node_id],
            key=lambda item: item[0].identity,
        )
        final_groups, group_split_reasons, split_reason = _generated_groups(
            candidates,
            policy,
        )
        if split_reason is not None:
            events.append(
                _event(
                    "split-group",
                    "abaqus.b31.orientation.split_group",
                    "generated normals were kept in separate groups",
                    *(item[0].identity for item in candidates),
                    locations=tuple(
                        location
                        for item in candidates
                        for location in item[0].source_locations
                    ),
                    record={"node": node_id, "reason": split_reason},
                )
            )
        for group_index, group_candidates in enumerate(final_groups):
            group_split_reason = group_split_reasons[group_index]
            identities = tuple(item[0].identity for item in group_candidates)
            candidate_normals = tuple(item[1] for item in group_candidates)
            if len(group_candidates) == 1 and group_split_reason is None:
                averaged = False
                group_kind: OrientationResolutionKind = "default-generated"
                normal = candidate_normals[0]
                event_kind = "default-generated"
                event_code = "abaqus.b31.orientation.default_generated"
                message = "Abaqus default/generated normal resolved for one element end"
            elif group_split_reason is not None:
                averaged = False
                group_kind = "split-group"
                normal = candidate_normals[0]
                event_kind = None
                event_code = "abaqus.b31.orientation.split_group"
                message = "generated normal kept as a split normal group"
            else:
                averaged = True
                group_kind = "averaged"
                normal = _average_normal(candidate_normals, policy)
                event_kind = "averaged"
                event_code = "abaqus.b31.orientation.averaged"
                message = "generated normals were averaged within a continuous group"
            group_identity = AbaqusNormalGroupIdentity(node_id, group_index)
            effective_group_normal = _validate_generated_normal(
                normal,
                policy,
                diagnostics,
                identities=identities,
                locations=tuple(
                    location
                    for item in group_candidates
                    for location in item[0].source_locations
                ),
            )
            if effective_group_normal is None:
                continue
            groups.append(
                AbaqusOrientationGroup(
                    identity=group_identity,
                    identities=identities,
                    candidate_normals=candidate_normals,
                    effective_normal=effective_group_normal,
                    averaged=averaged,
                    split_reason=group_split_reason,
                )
            )
            if event_kind is not None:
                events.append(
                    _event(
                        event_kind,
                        event_code,
                        message,
                        *identities,
                        normal_group=group_identity,
                        locations=tuple(
                            location
                            for item in group_candidates
                            for location in item[0].source_locations
                        ),
                    )
                )
            for base, _candidate in group_candidates:
                orientation = _make_from_normal(
                    base,
                    effective_group_normal,
                    normal_source=(
                        "averaged-normal" if averaged else "generated-normal"
                    ),
                    resolution_kind=group_kind,
                    normal_group=group_identity,
                    locations=base.source_locations,
                    diagnostics=diagnostics,
                    policy=policy,
                )
                if orientation is not None:
                    resolved[base.identity] = orientation

    field = AbaqusElementEndOrientationField(tuple(resolved.values()))
    events = _stable_events(events)
    diagnostics = _stable_diagnostics(diagnostics)
    groups = sorted(groups, key=lambda group: group.identity)
    result = AbaqusOrientationResolution(
        field=field,
        report=AbaqusOrientationResolutionReport(
            events=tuple(events),
            diagnostics=tuple(diagnostics),
            groups=tuple(groups),
            policy=policy,
        ),
    )
    if strict:
        result.raise_if_invalid()
    return result


def resolve_orientations(
    deck: AbaqusDeck,
    topology: AbaqusOrientationTopology | Mapping[str, Any] | Any | None = None,
    *,
    policy: AbaqusOrientationPolicy = DEFAULT_ABAQUS_ORIENTATION_POLICY,
    strict: bool = False,
) -> AbaqusOrientationResolution:
    """Short public alias for :func:`resolve_b31_orientations`."""

    return resolve_b31_orientations(
        deck,
        topology,
        policy=policy,
        strict=strict,
    )


def resolve_orientation_field(
    deck: AbaqusDeck,
    topology: AbaqusOrientationTopology | Mapping[str, Any] | Any | None = None,
    *,
    policy: AbaqusOrientationPolicy = DEFAULT_ABAQUS_ORIENTATION_POLICY,
    strict: bool = False,
) -> AbaqusOrientationResolution:
    """Compatibility alias emphasizing the returned field/report pair."""

    return resolve_b31_orientations(
        deck,
        topology,
        policy=policy,
        strict=strict,
    )


def _make_resolution(
    entries: Iterable[AbaqusElementEndOrientation],
    events: Iterable[AbaqusOrientationReportEntry],
    diagnostics: Iterable[AbaqusOrientationReportEntry],
    groups: Iterable[AbaqusOrientationGroup],
    policy: AbaqusOrientationPolicy,
) -> AbaqusOrientationResolution:
    return AbaqusOrientationResolution(
        AbaqusElementEndOrientationField(tuple(entries)),
        AbaqusOrientationResolutionReport(
            events=_stable_events(events),
            diagnostics=_stable_diagnostics(diagnostics),
            groups=tuple(sorted(groups, key=lambda group: group.identity)),
            policy=policy,
        ),
    )


def _coerce_topology(
    deck: AbaqusDeck,
    topology: AbaqusOrientationTopology | Mapping[str, Any] | Any | None,
) -> AbaqusOrientationTopology:
    if topology is None:
        return AbaqusOrientationTopology.from_deck(deck)
    if isinstance(topology, AbaqusOrientationTopology):
        return topology
    if isinstance(topology, AbaqusDeck):
        return AbaqusOrientationTopology.from_deck(topology.snapshot())

    def get_value(name: str, default: Any) -> Any:
        if isinstance(topology, Mapping):
            return topology.get(name, default)
        return getattr(topology, name, default)

    elements_value = get_value("elements", deck.elements)
    if isinstance(elements_value, Mapping):
        elements_value = tuple(elements_value.values())
    elements = tuple(_coerce_element(element) for element in elements_value)
    nodes_value = get_value("nodes", deck.nodes)
    if isinstance(nodes_value, Mapping):
        nodes = {
            int(node_id): _node_value_coordinates(value)
            for node_id, value in nodes_value.items()
        }
    else:
        nodes = dict(deck.nodes)
    node_records = get_value("node_records", deck.node_records)
    element_sets = get_value("element_sets", deck.element_sets)
    node_sets = get_value("node_sets", deck.node_sets)
    sections = get_value("sections", deck.sections)
    normal_records = get_value("normal_records", deck.normal_records)
    return AbaqusOrientationTopology(
        nodes=nodes,
        elements=elements,
        node_records=node_records,
        element_sets=element_sets,
        node_sets=node_sets,
        sections=tuple(sections),
        normal_records=tuple(normal_records),
    )


def _coerce_element(value: Any) -> AbaqusElement:
    if isinstance(value, AbaqusElement):
        return value
    element_type = str(getattr(value, "type", ""))
    node_ids = tuple(int(node_id) for node_id in getattr(value, "node_ids", ()))
    orientation_node_id = getattr(value, "orientation_node_id", None)
    if (
        orientation_node_id is None
        and element_type.upper() == "B31"
        and len(node_ids) == 3
    ):
        orientation_node_id = node_ids[2]
        node_ids = node_ids[:2]
    props = getattr(value, "props", {})
    element_set = getattr(value, "element_set", None)
    if element_set is None and isinstance(props, Mapping):
        element_set = props.get("element_set")
    return AbaqusElement(
        id=int(getattr(value, "id")),
        node_ids=node_ids,
        type=element_type,
        element_set=element_set,
        keyword_location=getattr(value, "keyword_location", None),
        data_location=getattr(value, "data_location", None),
        raw_fields=tuple(getattr(value, "raw_fields", ())),
        orientation_node_id=(
            None if orientation_node_id is None else int(orientation_node_id)
        ),
    )


def _node_value_coordinates(value: Any) -> Vector3:
    if isinstance(value, Mapping):
        return _as_vector(
            (value.get("x"), value.get("y"), value.get("z", 0.0)),
            label="node coordinates",
        )
    if all(hasattr(value, name) for name in ("x", "y")):
        return _as_vector(
            (value.x, value.y, getattr(value, "z", 0.0)),
            label="node coordinates",
        )
    return _as_vector(value, label="node coordinates")


def _element_key(element: AbaqusElement) -> tuple[Any, ...]:
    return (
        int(element.id),
        tuple(int(value) for value in element.structural_node_ids),
        int(element.additional_orientation_node_id or -1),
        str(element.type).upper(),
        _location_key(element.data_location or element.keyword_location),
    )


def _node_coordinates(
    topology: AbaqusOrientationTopology,
    node_id: int,
) -> Vector3 | None:
    value = topology.nodes.get(int(node_id))
    if value is not None:
        return value
    record = topology.node_records.get(int(node_id))
    return None if record is None else tuple(float(value) for value in record.coordinates)


def _base_reference(
    topology: AbaqusOrientationTopology,
    element: AbaqusElement,
    origin: np.ndarray,
    diagnostics: list[AbaqusOrientationReportEntry],
) -> tuple[
    np.ndarray | None,
    Literal["orientation-node", "section-n1", "default-n1"],
    tuple[AbaqusSourceLocation, ...],
]:
    orientation_node_id = element.additional_orientation_node_id
    if orientation_node_id is not None:
        coordinates = _node_coordinates(topology, orientation_node_id)
        if coordinates is None:
            diagnostics.append(
                _diagnostic(
                    "invalid",
                    "abaqus.b31.orientation_node_missing",
                    (
                        f"B31 element {element.id} references missing orientation "
                        f"node {orientation_node_id}"
                    ),
                    locations=_source_locations_for_element(element),
                    record={
                        "element": int(element.id),
                        "orientation_node": int(orientation_node_id),
                    },
                    remediation="Define the orientation node with finite coordinates.",
                )
            )
            return None, "orientation-node", ()
        orientation = np.asarray(coordinates, dtype=float)
        if orientation.shape != (3,) or not np.all(np.isfinite(orientation)):
            diagnostics.append(
                _diagnostic(
                    "invalid",
                    "abaqus.b31.orientation_node_invalid",
                    f"B31 orientation node {orientation_node_id} has invalid coordinates",
                    locations=_stable_locations(
                        (*_source_locations_for_element(element), *_node_locations(topology, orientation_node_id))
                    ),
                    record=tuple(float(value) for value in orientation),
                    remediation="Provide three finite orientation-node coordinates.",
                )
            )
            return None, "orientation-node", ()
        return (
            orientation - origin,
            "orientation-node",
            _node_locations(topology, orientation_node_id),
        )

    section = _section_for_element(topology, int(element.id))
    if section is not None and isinstance(section.data, AbaqusBeamSectionData):
        if section.data.approximate_n1 is not None:
            try:
                section_reference = _as_vector(
                    section.data.approximate_n1,
                    label="*BEAM SECTION n1",
                )
            except ValueError as exc:
                diagnostics.append(
                    _diagnostic(
                        "invalid",
                        "beam.orientation.invalid",
                        str(exc),
                        locations=_section_locations(section),
                        record=section.data.approximate_n1,
                    )
                )
                return None, "section-n1", _section_locations(section)
            return (
                np.asarray(section_reference, dtype=float),
                "section-n1",
                _section_reference_locations(section),
            )
    return (
        np.asarray(BEAM_DEFAULT_LOCAL_Y_REFERENCE, dtype=float),
        "default-n1",
        (),
    )


def _section_for_element(
    topology: AbaqusOrientationTopology,
    element_id: int,
) -> AbaqusSection | None:
    selected: AbaqusSection | None = None
    for section in topology.sections:
        if int(element_id) in _section_element_ids(topology, section):
            selected = section
    return selected


def _section_element_ids(
    topology: AbaqusOrientationTopology,
    section: AbaqusSection,
) -> tuple[int, ...]:
    captured = {int(value) for value in section.element_ids}
    if getattr(section, "target_was_defined", None) is True:
        return tuple(sorted(captured))
    if captured:
        return tuple(sorted(captured))
    return tuple(sorted(int(value) for value in topology.element_sets.get(section.element_set, ())))


def _resolve_node_normals(
    topology: AbaqusOrientationTopology,
    diagnostics: list[AbaqusOrientationReportEntry],
) -> dict[int, tuple[Vector3, tuple[AbaqusSourceLocation, ...]]]:
    result: dict[int, tuple[Vector3, tuple[AbaqusSourceLocation, ...]]] = {}
    b31_present = any(str(element.type).upper() == "B31" for element in topology.elements)
    for node_id, record in sorted(topology.node_records.items()):
        extras = tuple(record.extra_fields)
        if record.normal is None and not extras:
            continue
        if record.normal is None:
            code = (
                "abaqus.b31.node_normal_empty"
                if len(extras) == 3
                else "abaqus.b31.node_normal_shape"
            )
            diagnostics.append(
                _diagnostic(
                    "invalid" if b31_present else "unsupported",
                    code if b31_present else "abaqus.line.nodal_normals_unsupported",
                    "*NODE normal data is empty or incomplete",
                    locations=_node_locations(topology, node_id),
                    record=extras,
                    remediation="Provide all three finite nodal normal components.",
                )
            )
            continue
        if extras and len(extras) != 3:
            diagnostics.append(
                _diagnostic(
                    "invalid",
                    "abaqus.b31.node_normal_shape",
                    "*NODE normal data must contain exactly three components",
                    locations=_node_locations(topology, node_id),
                    record=extras,
                )
            )
            continue
        vector = _valid_source_vector(
            record.normal.vector,
            code="abaqus.b31.node_normal_zero",
            message=f"node {node_id} normal must be finite and nonzero",
            locations=_node_locations(topology, node_id),
            record=record.normal.vector,
            diagnostics=diagnostics,
        )
        if vector is not None:
            result[int(node_id)] = (vector, _node_locations(topology, node_id))
    return result


def _resolve_element_normals(
    topology: AbaqusOrientationTopology,
    element_lookup: Mapping[int, AbaqusElement],
    diagnostics: list[AbaqusOrientationReportEntry],
    policy: AbaqusOrientationPolicy,
) -> dict[
    AbaqusElementEndIdentity,
    tuple[Vector3, tuple[AbaqusSourceLocation, ...]],
]:
    result: dict[
        AbaqusElementEndIdentity,
        tuple[Vector3, tuple[AbaqusSourceLocation, ...]],
    ] = {}
    records = tuple(topology.normal_records)
    for record in records:
        vector = _valid_source_vector(
            record.normal,
            code="abaqus.b31.normal_zero",
            message="*NORMAL vector must be finite and nonzero",
            locations=(record.location,),
            record=record.raw_fields,
            diagnostics=diagnostics,
        )
        if vector is None:
            continue
        element_ids = _normal_target_ids(
            record.element,
            topology.element_sets,
            kind="element",
            record=record,
            diagnostics=diagnostics,
        )
        node_ids = _normal_target_ids(
            record.node,
            topology.node_sets,
            kind="node",
            record=record,
            diagnostics=diagnostics,
        )
        if element_ids is None or node_ids is None:
            continue
        for element_id in element_ids:
            element = element_lookup.get(int(element_id))
            if element is None:
                diagnostics.append(
                    _diagnostic(
                        "invalid",
                        "abaqus.b31.normal.element_missing",
                        f"*NORMAL references undefined element {element_id}",
                        locations=(record.location,),
                        record=record.raw_fields,
                        remediation="Define the referenced B31 element.",
                    )
                )
                continue
            if str(element.type).upper() != "B31":
                diagnostics.append(
                    _diagnostic(
                        "unsupported",
                        "abaqus.normal.element_type_unsupported",
                        "*NORMAL, TYPE=ELEMENT currently supports B31 targets only",
                        locations=(record.location,),
                        record=record.raw_fields,
                        remediation="Target a B31 element.",
                    )
                )
                continue
            structural = tuple(int(value) for value in element.structural_node_ids)
            for node_id in node_ids:
                if int(node_id) not in structural:
                    diagnostics.append(
                        _diagnostic(
                            "invalid",
                            "abaqus.b31.normal.local_end_invalid",
                            (
                                f"*NORMAL node {node_id} is not a local end of "
                                f"element {element_id}"
                            ),
                            locations=(record.location,),
                            record={
                                "element": int(element_id),
                                "node": int(node_id),
                            },
                            remediation="Use one of the two structural B31 nodes.",
                        )
                    )
                    continue
                local_end = structural.index(int(node_id)) + 1
                identity = AbaqusElementEndIdentity(
                    int(element_id),
                    local_end,
                    int(node_id),
                )
                previous = result.get(identity)
                if previous is not None and not _same_direction(
                    previous[0], vector, policy
                ):
                    diagnostics.append(
                        _diagnostic(
                            "conflict",
                            "abaqus.b31.normal.conflict",
                            "conflicting *NORMAL vectors share one exact element-end identity",
                            identities=(identity,),
                            locations=(*previous[1], record.location),
                            record={
                                "identity": identity,
                                "first": previous[0],
                                "second": vector,
                            },
                            remediation="Define one vector for each element and local end.",
                        )
                    )
                    continue
                if previous is None:
                    result[identity] = (vector, (record.location,))
                else:
                    result[identity] = (
                        previous[0],
                        _stable_locations((*previous[1], record.location)),
                    )
    return result


def _normal_target_ids(
    target: int | str,
    collections: Mapping[str, tuple[int, ...]],
    *,
    kind: str,
    record: AbaqusNormalRecord,
    diagnostics: list[AbaqusOrientationReportEntry],
) -> tuple[int, ...] | None:
    if isinstance(target, int):
        return (int(target),)
    if target not in collections:
        code = (
            "abaqus.b31.normal.element_set_missing"
            if kind == "element"
            else "abaqus.b31.normal.node_set_missing"
        )
        diagnostics.append(
            _diagnostic(
                "invalid",
                code,
                f"*NORMAL {kind} set {target!r} is not defined",
                locations=(record.location,),
                record=record.raw_fields,
                remediation=f"Define the referenced {kind} set.",
            )
        )
        return None
    return tuple(sorted({int(value) for value in collections[target]}))


def _normal_without_b31_diagnostics(
    topology: AbaqusOrientationTopology,
) -> tuple[AbaqusOrientationReportEntry, ...]:
    diagnostics: list[AbaqusOrientationReportEntry] = []
    records = topology.normal_records
    if records:
        record = sorted(records, key=_normal_record_key)[0]
        diagnostics.append(
            _diagnostic(
                "unsupported",
                "abaqus.normal.element_type_unsupported",
                "*NORMAL, TYPE=ELEMENT is supported here only for B31 beams",
                locations=(record.location,),
                record=record.raw_fields,
                remediation="Target a B31 element or remove the *NORMAL block.",
            )
        )
    return tuple(diagnostics)


def _generated_groups(
    candidates: list[tuple[_EndpointBase, Vector3]],
    policy: AbaqusOrientationPolicy,
) -> tuple[
    list[list[tuple[_EndpointBase, Vector3]]],
    tuple[str | None, ...],
    str | None,
]:
    if len(candidates) > policy.max_averaging_elements:
        groups = [[candidate] for candidate in candidates]
        reason = "more-than-30-remaining-elements"
        return groups, tuple(reason for _group in groups), reason

    remaining = {item[0].identity: item for item in candidates}
    components: list[list[tuple[_EndpointBase, Vector3]]] = []
    while remaining:
        seed_identity = min(remaining)
        component = [remaining.pop(seed_identity)]
        frontier = list(component)
        while frontier:
            current = frontier.pop()
            for identity in sorted(tuple(remaining)):
                candidate = remaining[identity]
                if _within_continuity(current[1], candidate[1], policy):
                    remaining.pop(identity)
                    component.append(candidate)
                    frontier.append(candidate)
        components.append(sorted(component, key=lambda item: item[0].identity))

    final_with_reasons: list[
        tuple[list[tuple[_EndpointBase, Vector3]], str | None]
    ] = []
    non_clique_split = False
    for component in components:
        if _all_pairwise_within(component, policy):
            final_with_reasons.append((component, None))
        else:
            non_clique_split = True
            final_with_reasons.extend(
                (
                    [candidate],
                    "non-clique-continuity-group",
                )
                for candidate in component
            )
    final_with_reasons.sort(key=lambda item: item[0][0][0].identity)
    final = [group for group, _reason in final_with_reasons]
    final.sort(key=lambda group: group[0][0].identity)
    group_reasons_by_identity = {
        group[0][0].identity: reason
        for group, reason in final_with_reasons
    }
    if len(components) > 1:
        for group in final:
            identity = group[0][0].identity
            if len(group) == 1 and group_reasons_by_identity[identity] is None:
                group_reasons_by_identity[identity] = "disjoint-continuity-group"
    ordered_reasons = tuple(
        group_reasons_by_identity[group[0][0].identity] for group in final
    )
    split_reason = (
        "non-clique-continuity-group"
        if non_clique_split
        else "disjoint-continuity-groups"
        if len(components) > 1
        else None
    )
    return final, ordered_reasons, split_reason


def _all_pairwise_within(
    candidates: list[tuple[_EndpointBase, Vector3]],
    policy: AbaqusOrientationPolicy,
) -> bool:
    return all(
        _within_continuity(first[1], second[1], policy)
        for index, first in enumerate(candidates)
        for second in candidates[index + 1 :]
    )


def _within_continuity(
    first: Vector3,
    second: Vector3,
    policy: AbaqusOrientationPolicy,
) -> bool:
    return float(np.dot(first, second)) >= policy.continuity_cosine


def _average_normal(
    normals: tuple[Vector3, ...],
    policy: AbaqusOrientationPolicy,
) -> Vector3:
    value = np.sum(np.asarray(normals, dtype=float), axis=0)
    unit = _unit_or_none(value, policy)
    if unit is None:
        raise ValueError("generated normal average is zero or non-finite")
    return unit


def _make_from_normal(
    base: _EndpointBase,
    normal: Vector3,
    *,
    normal_source: Literal[
        "element-normal", "node-normal", "generated-normal", "averaged-normal"
    ],
    resolution_kind: OrientationResolutionKind,
    locations: Iterable[AbaqusSourceLocation],
    diagnostics: list[AbaqusOrientationReportEntry],
    policy: AbaqusOrientationPolicy,
    normal_group: AbaqusNormalGroupIdentity | None = None,
) -> AbaqusElementEndOrientation | None:
    vector = _valid_source_vector(
        normal,
        code="abaqus.b31.normal_zero",
        message="normal must be finite and nonzero",
        locations=tuple(locations),
        record=normal,
        diagnostics=diagnostics,
    )
    if vector is None:
        return None
    tangent = np.asarray(base.tangent, dtype=float)
    approximate_n1 = np.asarray(base.approximate_n1, dtype=float)
    expected_normal = np.cross(tangent, approximate_n1)
    if float(np.dot(vector, expected_normal)) < 0.0:
        vector = tuple(-value for value in vector)
    raw = np.asarray(vector, dtype=float)
    projected = raw - float(raw @ tangent) * tangent
    projected_norm = float(np.linalg.norm(projected))
    if not isfinite(projected_norm) or projected_norm <= policy.nonparallel_tolerance:
        diagnostics.append(
            _diagnostic(
                "invalid",
                "abaqus.b31.normal_parallel",
                "normal is parallel or nearly parallel to its element tangent",
                identities=(base.identity,),
                locations=tuple(locations),
                record={
                    "identity": base.identity,
                    "normal": tuple(float(value) for value in raw),
                    "tangent": base.tangent,
                },
            )
        )
        return None
    effective_normal = projected / projected_norm
    effective_n1 = np.cross(effective_normal, tangent)
    effective_n1_unit = _unit_or_none(effective_n1, policy)
    if effective_n1_unit is None:
        diagnostics.append(
            _diagnostic(
                "invalid",
                "abaqus.b31.normal_parallel",
                "normal cannot define a unique n1 direction",
                identities=(base.identity,),
                locations=tuple(locations),
                record=tuple(float(value) for value in effective_normal),
            )
        )
        return None
    tangent_tuple = base.tangent
    normal_tuple = tuple(float(value) for value in effective_normal)
    n1_tuple = tuple(float(value) for value in effective_n1_unit)
    try:
        _validate_right_handed_frame(
            tangent_tuple,
            n1_tuple,
            normal_tuple,
            tolerance=policy.frame_tolerance,
        )
    except ValueError as exc:
        diagnostics.append(
            _diagnostic(
                "invalid",
                "abaqus.b31.orientation_invalid",
                str(exc),
                identities=(base.identity,),
                locations=tuple(locations),
                record={
                    "identity": base.identity,
                    "tangent": tangent_tuple,
                    "n1": n1_tuple,
                    "n2": normal_tuple,
                },
            )
        )
        return None
    return AbaqusElementEndOrientation(
        identity=base.identity,
        tangent=tangent_tuple,
        n1=n1_tuple,
        normal=normal_tuple,
        approximate_n1=base.approximate_n1,
        resolution_kind=resolution_kind,
        reference_source=base.reference_source,
        normal_source=normal_source,
        normal_group=normal_group,
        source_locations=_stable_locations(locations),
        frame_tolerance=policy.frame_tolerance,
    )


def _validate_generated_normal(
    normal: Vector3,
    policy: AbaqusOrientationPolicy,
    diagnostics: list[AbaqusOrientationReportEntry],
    *,
    identities: tuple[AbaqusElementEndIdentity, ...],
    locations: tuple[AbaqusSourceLocation, ...],
) -> Vector3 | None:
    try:
        value = _unit_or_none(normal, policy)
        if value is None:
            raise ValueError("generated normal average is non-finite or zero")
        return value
    except ValueError as exc:
        diagnostics.append(
            _diagnostic(
                "invalid",
                "abaqus.b31.normal_invalid",
                str(exc),
                identities=identities,
                locations=locations,
                record=normal,
            )
        )
        return None


def _valid_source_vector(
    value: Any,
    *,
    code: str,
    message: str,
    locations: Iterable[AbaqusSourceLocation],
    record: Any,
    diagnostics: list[AbaqusOrientationReportEntry],
) -> Vector3 | None:
    try:
        vector = _as_vector(value, label="normal")
    except (TypeError, ValueError) as exc:
        diagnostics.append(
            _diagnostic(
                "invalid",
                "abaqus.b31.normal_invalid",
                str(exc),
                locations=tuple(locations),
                record=record,
            )
        )
        return None
    if max(abs(value) for value in vector) == 0.0:
        diagnostics.append(
            _diagnostic(
                "invalid",
                code,
                message,
                locations=tuple(locations),
                record=record,
            )
        )
        return None
    return vector


def _unit_or_none(
    value: Any,
    policy: AbaqusOrientationPolicy,
) -> Vector3 | None:
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    if array.shape != (3,) or not np.all(np.isfinite(array)):
        return None
    norm = float(np.linalg.norm(array))
    if not isfinite(norm) or norm <= policy.nonparallel_tolerance:
        return None
    unit = array / norm
    return tuple(float(value) for value in unit)


def _validate_right_handed_frame(
    tangent: Vector3,
    n1: Vector3,
    n2: Vector3,
    *,
    tolerance: float,
) -> None:
    arrays = tuple(np.asarray(value, dtype=float) for value in (tangent, n1, n2))
    if any(array.shape != (3,) or not np.all(np.isfinite(array)) for array in arrays):
        raise ValueError("orientation frame must contain finite 3-vectors")
    if any(abs(float(np.linalg.norm(array)) - 1.0) > tolerance for array in arrays):
        raise ValueError("orientation frame vectors must be unit length")
    tangent_array, n1_array, n2_array = arrays
    if any(
        abs(float(np.dot(first, second))) > tolerance
        for first, second in (
            (tangent_array, n1_array),
            (tangent_array, n2_array),
            (n1_array, n2_array),
        )
    ):
        raise ValueError("orientation frame vectors must be mutually nonparallel")
    expected_n2 = np.cross(tangent_array, n1_array)
    if not np.allclose(expected_n2, n2_array, rtol=tolerance, atol=tolerance):
        raise ValueError("orientation frame must satisfy n2 = tangent cross n1")
    determinant = float(np.linalg.det(np.vstack((tangent_array, n1_array, n2_array))))
    if not isfinite(determinant) or abs(determinant - 1.0) > tolerance * 10.0:
        raise ValueError("orientation frame must be right-handed")


def _same_direction(
    first: Vector3,
    second: Vector3,
    policy: AbaqusOrientationPolicy,
) -> bool:
    first_unit = _unit_unchecked(first)
    second_unit = _unit_unchecked(second)
    return first_unit is not None and second_unit is not None and float(
        np.dot(first_unit, second_unit)
    ) >= 1.0 - policy.comparison_tolerance


def _unit_unchecked(value: Any) -> np.ndarray | None:
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    if array.shape != (3,) or not np.all(np.isfinite(array)):
        return None
    norm = float(np.linalg.norm(array))
    if not isfinite(norm) or norm <= 0.0:
        return None
    return array / norm


def _as_vector(value: Any, *, label: str) -> Vector3:
    try:
        values = tuple(float(component) for component in value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must contain exactly three finite components") from exc
    if len(values) != 3 or not all(isfinite(value) for value in values):
        raise ValueError(f"{label} must contain exactly three finite components")
    return values  # type: ignore[return-value]


def _event(
    kind: str,
    code: str,
    message: str,
    *identities: AbaqusElementEndIdentity,
    locations: Iterable[AbaqusSourceLocation] = (),
    normal_group: AbaqusNormalGroupIdentity | None = None,
    record: Any = None,
) -> AbaqusOrientationReportEntry:
    return AbaqusOrientationReportEntry(
        kind=kind,
        code=code,
        message=message,
        identities=tuple(identities),
        locations=tuple(locations),
        normal_group=normal_group,
        record=record,
    )


def _diagnostic(
    kind: str,
    code: str,
    message: str,
    *,
    identities: Iterable[AbaqusElementEndIdentity] = (),
    locations: Iterable[AbaqusSourceLocation] = (),
    record: Any = None,
    remediation: str | None = None,
    severity: Literal["info", "error", "capability"] | None = None,
) -> AbaqusOrientationReportEntry:
    resolved_severity = severity or (
        "capability" if kind == "unsupported" or kind == "unsupported-variation" else "error"
    )
    return AbaqusOrientationReportEntry(
        kind=kind,
        code=code,
        message=message,
        identities=tuple(identities),
        locations=tuple(locations),
        record=record,
        remediation=remediation,
        severity=resolved_severity,  # type: ignore[arg-type]
    )


def _stable_events(
    events: Iterable[AbaqusOrientationReportEntry],
) -> list[AbaqusOrientationReportEntry]:
    return sorted(
        events,
        key=lambda item: (
            item.identities[0] if item.identities else AbaqusElementEndIdentity(0, 1, 0),
            item.kind,
            item.normal_group or AbaqusNormalGroupIdentity(0, 0),
            tuple(_location_key(location) for location in item.locations),
            item.message,
        ),
    )


def _stable_diagnostics(
    diagnostics: Iterable[AbaqusOrientationReportEntry],
) -> list[AbaqusOrientationReportEntry]:
    return sorted(
        diagnostics,
        key=lambda item: (
            item.identities[0] if item.identities else AbaqusElementEndIdentity(0, 1, 0),
            item.kind,
            item.code,
            tuple(_location_key(location) for location in item.locations),
            item.message,
        ),
    )


def _normal_record_key(record: AbaqusNormalRecord) -> tuple[Any, ...]:
    return (
        _target_key(record.element),
        _target_key(record.node),
        tuple(float(value) for value in record.normal),
        _location_key(record.location),
    )


def _target_key(value: int | str) -> tuple[str, str]:
    return (type(value).__name__, str(value))


def _source_locations_for_element(
    *elements: AbaqusElement,
) -> tuple[AbaqusSourceLocation, ...]:
    return _stable_locations(
        location
        for element in elements
        for location in (element.data_location, element.keyword_location)
        if location is not None
    )


def _section_locations(section: AbaqusSection) -> tuple[AbaqusSourceLocation, ...]:
    locations: list[AbaqusSourceLocation | None] = [section.keyword_location]
    data = section.data
    if isinstance(data, AbaqusBeamSectionData):
        locations.append(data.orientation.location)
        locations.append(data.geometry.location)
    return _stable_locations(location for location in locations if location is not None)


def _section_reference_locations(
    section: AbaqusSection,
) -> tuple[AbaqusSourceLocation, ...]:
    data = section.data
    if isinstance(data, AbaqusBeamSectionData) and data.orientation.location is not None:
        return (data.orientation.location,)
    return () if section.keyword_location is None else (section.keyword_location,)


def _node_locations(
    topology: AbaqusOrientationTopology,
    node_id: int,
) -> tuple[AbaqusSourceLocation, ...]:
    record = topology.node_records.get(int(node_id))
    if record is None:
        return ()
    location = record.normal.location if record.normal is not None else record.location
    return () if location is None else (location,)


def _location_key(location: AbaqusSourceLocation | None) -> tuple[str, int, str]:
    if location is None:
        return ("", 0, "")
    return (str(location.path) if location.path is not None else "", int(location.line), str(location.keyword or ""))


def _stable_locations(
    locations: Iterable[AbaqusSourceLocation],
) -> tuple[AbaqusSourceLocation, ...]:
    unique = {_location_key(location): location for location in locations if location is not None}
    return tuple(unique[key] for key in sorted(unique))


__all__ = [
    "ABAQUS_ORIENTATION_POLICY",
    "DEFAULT_ABAQUS_ORIENTATION_POLICY",
    "AbaqusElementEndOrientation",
    "AbaqusElementEndOrientationField",
    "AbaqusNormalGroupIdentity",
    "AbaqusOrientationDiagnostic",
    "AbaqusOrientationGroup",
    "AbaqusOrientationPolicy",
    "AbaqusOrientationReportEntry",
    "AbaqusOrientationResolution",
    "AbaqusOrientationResolutionReport",
    "AbaqusOrientationTopology",
    "resolve_b31_orientations",
    "resolve_orientation_field",
    "resolve_orientations",
]
