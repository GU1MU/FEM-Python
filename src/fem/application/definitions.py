"""Headless editable definitions and their single detached compiler."""

from __future__ import annotations

from array import array
from collections.abc import Iterable, Iterator, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from itertools import chain
from typing import Any, Callable

from fem.core.model import MaterialDefinition, SectionAssignment
from fem.elements import (
    BEAM_ELEMENT_LOCAL_Y_REFERENCE_KEY,
    BEAM_FRAME_FIELD_KEY,
    BEAM_FRAME_FIELD_REFERENCE_KEY,
    BEAM_LOCAL_Y_REFERENCE_KEY,
    BeamOrientation,
    BeamOrientationError,
    get_element_capabilities,
    parse_beam_orientation,
    resolve_beam_frame_field,
)
from fem.geometry.references import LogicalEntityRef, logical_ref_sort_key

from .capabilities import RegionRef
from .diagnostics import (
    PreflightDiagnostic,
    PreflightSeverity,
    PreflightStage,
)
from .analysis_identity import validate_analysis_object_names


from .native_part import NativePart


@dataclass(frozen=True, slots=True)
class FeatureRecord:
    """One item in the shallow native feature history."""

    name: str
    kind: str
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class MeshEntityRef:
    """Stable reference to one entity in one generated finite-element mesh."""

    kind: str
    node_id: int | None = None
    element_id: int | None = None
    local_index: int | None = None
    node_ids: tuple[int, ...] = ()
    part_id: str | None = None

    def __post_init__(self) -> None:
        if self.part_id is not None:
            from .native_part import normalize_part_id

            object.__setattr__(
                self,
                "part_id",
                normalize_part_id(self.part_id, "mesh entity part_id"),
            )
        if self.kind not in {"node", "edge", "face", "element"}:
            raise ValueError(f"unsupported mesh entity kind: {self.kind!r}")
        node_ids = tuple(int(node_id) for node_id in self.node_ids)
        if len(set(node_ids)) != len(node_ids):
            raise ValueError("mesh entity node_ids must be unique")
        object.__setattr__(self, "node_ids", node_ids)
        if self.kind == "node":
            if (
                isinstance(self.node_id, bool)
                or not isinstance(self.node_id, int)
                or self.element_id is not None
                or self.local_index is not None
                or node_ids
            ):
                raise ValueError("node reference requires only one integer node_id")
            return
        if (
            isinstance(self.element_id, bool)
            or not isinstance(self.element_id, int)
            or self.node_id is not None
        ):
            raise ValueError(
                f"{self.kind} reference requires one integer element_id"
            )
        if self.kind == "element":
            if self.local_index is not None or node_ids:
                raise ValueError(
                    "element reference does not accept local_index or node_ids"
                )
            return
        if (
            isinstance(self.local_index, bool)
            or not isinstance(self.local_index, int)
            or self.local_index < 0
            or not node_ids
        ):
            raise ValueError(
                f"{self.kind} reference requires local_index and node_ids"
            )

    @classmethod
    def node(
        cls,
        node_id: int,
        *,
        part_id: str | None = None,
    ) -> MeshEntityRef:
        return cls("node", node_id=int(node_id), part_id=part_id)

    @classmethod
    def element(
        cls,
        element_id: int,
        *,
        part_id: str | None = None,
    ) -> MeshEntityRef:
        return cls(
            "element",
            element_id=int(element_id),
            part_id=part_id,
        )

    @classmethod
    def edge(
        cls,
        element_id: int,
        local_index: int,
        node_ids: Iterable[int],
        *,
        part_id: str | None = None,
    ) -> MeshEntityRef:
        return cls(
            "edge",
            element_id=int(element_id),
            local_index=int(local_index),
            node_ids=tuple(int(node_id) for node_id in node_ids),
            part_id=part_id,
        )

    @classmethod
    def face(
        cls,
        element_id: int,
        local_index: int,
        node_ids: Iterable[int],
        *,
        part_id: str | None = None,
    ) -> MeshEntityRef:
        return cls(
            "face",
            element_id=int(element_id),
            local_index=int(local_index),
            node_ids=tuple(int(node_id) for node_id in node_ids),
            part_id=part_id,
        )

    @property
    def identity(self) -> tuple[int, int]:
        """Return the integer identity used for canonical ordering."""

        if self.kind == "node":
            return int(self.node_id), -1
        return int(self.element_id), (
            -1 if self.local_index is None else int(self.local_index)
        )


def mesh_entity_ref_sort_key(
    reference: MeshEntityRef,
) -> tuple[str, int, int, int, tuple[int, ...]]:
    """Return a deterministic ordering key for mesh entity references."""

    kind_order = {"node": 0, "edge": 1, "face": 2, "element": 3}
    primary, local_index = reference.identity
    return (
        "" if reference.part_id is None else reference.part_id,
        kind_order[reference.kind],
        primary,
        local_index,
        reference.node_ids,
    )


class MeshTopologyDirectory:
    """Revision-bound connectivity used to inflate compact edge/face refs."""

    __slots__ = ("_revision", "_rows")

    def __init__(
        self,
        mesh_revision: int | None,
        rows: Mapping[
            tuple[str | None, str, int, int], Iterable[int]
        ] | None = None,
    ) -> None:
        if mesh_revision is not None and (
            type(mesh_revision) is not int or mesh_revision < 0
        ):
            raise ValueError("mesh_revision must be a non-negative integer or None")
        normalized: dict[
            tuple[str | None, str, int, int], tuple[int, ...]
        ] = {}
        for raw_key, raw_node_ids in (rows or {}).items():
            part_id, kind, element_id, local_index = raw_key
            if kind not in {"edge", "face"}:
                raise ValueError("topology directory accepts only edge/face rows")
            key = (
                part_id,
                kind,
                int(element_id),
                int(local_index),
            )
            node_ids = tuple(int(value) for value in raw_node_ids)
            if not node_ids or len(set(node_ids)) != len(node_ids):
                raise ValueError("topology directory rows require unique node IDs")
            existing = normalized.get(key)
            if existing is not None and existing != node_ids:
                raise ValueError(f"conflicting topology directory row: {key!r}")
            normalized[key] = node_ids
        self._revision = mesh_revision
        self._rows = normalized

    @property
    def mesh_revision(self) -> int | None:
        return self._revision

    def resolve(
        self,
        part_id: str | None,
        kind: str,
        element_id: int,
        local_index: int,
        *,
        mesh_revision: int | None,
    ) -> tuple[int, ...]:
        if self._revision != mesh_revision:
            raise ValueError(
                "mesh topology directory revision does not match compact references"
            )
        key = (part_id, kind, int(element_id), int(local_index))
        try:
            return self._rows[key]
        except KeyError as error:
            raise ValueError(
                f"mesh topology is unavailable for compact {kind} reference: "
                f"{key!r}"
            ) from error

    def rows(
        self,
    ) -> tuple[tuple[str | None, str, int, int, tuple[int, ...]], ...]:
        return tuple(
            (*key, node_ids)
            for key, node_ids in sorted(
                self._rows.items(),
                key=lambda item: (
                    "" if item[0][0] is None else item[0][0],
                    0 if item[0][1] == "edge" else 1,
                    item[0][2],
                    item[0][3],
                ),
            )
        )

    def rebound(self, mesh_revision: int) -> MeshTopologyDirectory:
        return MeshTopologyDirectory(mesh_revision, self._rows)

    def __deepcopy__(self, memo: dict[int, Any]) -> MeshTopologyDirectory:
        del memo
        return self


class CompressedMeshEntityRefs(Sequence[MeshEntityRef]):
    """Canonical compact sequence of references from one mesh entity kind."""

    __slots__ = ("_kind", "_groups", "_length", "_mesh_revision", "_topology")

    def __init__(
        self,
        references: Iterable[MeshEntityRef],
        *,
        mesh_revision: int | None = None,
        topology: MeshTopologyDirectory | None = None,
    ) -> None:
        iterator = iter(references)
        try:
            first = next(iterator)
        except StopIteration:
            raise TypeError(
                "compressed mesh references require non-empty MeshEntityRef values"
            ) from None
        if type(first) is not MeshEntityRef:
            raise TypeError(
                "compressed mesh references require non-empty MeshEntityRef values"
            )
        kind = first.kind
        inferred_rows: dict[
            tuple[str | None, str, int, int], tuple[int, ...]
        ] = {}
        raw_groups: dict[str | None, array] = {}
        length = 0
        for value in chain((first,), iterator):
            if type(value) is not MeshEntityRef:
                raise TypeError(
                    "compressed mesh references require non-empty "
                    "MeshEntityRef values"
                )
            if value.kind != kind:
                raise ValueError("one compact reference set cannot mix entity kinds")
            packed = raw_groups.setdefault(value.part_id, array("q"))
            if kind in {"node", "element"}:
                packed.append(
                    int(value.node_id if kind == "node" else value.element_id)
                )
            else:
                pair = (int(value.element_id), int(value.local_index))
                key = (value.part_id, kind, *pair)
                if key in inferred_rows:
                    raise ValueError("mesh reference identities must be unique")
                packed.extend(pair)
                inferred_rows[key] = value.node_ids
            length += 1
        grouped: list[tuple[str | None, array]] = []
        for part_id in sorted(
            raw_groups,
            key=lambda value: "" if value is None else value,
        ):
            raw = raw_groups[part_id]
            packed = array("q")
            if kind in {"node", "element"}:
                ids = (
                    raw
                    if all(
                        raw[index - 1] <= raw[index]
                        for index in range(1, len(raw))
                    )
                    else sorted(raw)
                )
                start = previous = ids[0]
                for identity in ids[1:]:
                    if identity == previous:
                        raise ValueError("mesh reference identities must be unique")
                    if identity == previous + 1:
                        previous = identity
                        continue
                    packed.extend((start, previous))
                    start = previous = identity
                packed.extend((start, previous))
            else:
                pairs = [
                    (raw[index], raw[index + 1])
                    for index in range(0, len(raw), 2)
                ]
                if any(
                    pairs[index - 1] > pairs[index]
                    for index in range(1, len(pairs))
                ):
                    pairs.sort()
                for pair in pairs:
                    packed.extend(pair)
            grouped.append((part_id, packed))
        if topology is None and inferred_rows:
            topology = MeshTopologyDirectory(mesh_revision, inferred_rows)
        if topology is not None and topology.mesh_revision != mesh_revision:
            raise ValueError("topology and compact references must share mesh revision")
        self._kind = kind
        self._groups = tuple(grouped)
        self._length = length
        self._mesh_revision = mesh_revision
        self._topology = topology

    @classmethod
    def from_compact(
        cls,
        kind: str,
        groups: Iterable[tuple[str | None, Iterable[int]]],
        *,
        mesh_revision: int | None,
        topology: MeshTopologyDirectory | None = None,
    ) -> CompressedMeshEntityRefs:
        if kind not in {"node", "edge", "face", "element"}:
            raise ValueError(f"unsupported mesh entity kind: {kind!r}")
        instance = object.__new__(cls)
        normalized_groups: list[tuple[str | None, array]] = []
        length = 0
        previous_part: str | None = None
        first = True
        for raw_part_id, raw_values in groups:
            if raw_part_id is None:
                part_id = None
            else:
                from .native_part import normalize_part_id

                part_id = normalize_part_id(
                    raw_part_id,
                    "compact mesh reference part_id",
                )
            if not first and ("" if previous_part is None else previous_part) >= (
                "" if part_id is None else part_id
            ):
                raise ValueError("compact reference part groups are not canonical")
            first = False
            previous_part = part_id
            packed = array("q", (int(value) for value in raw_values))
            if len(packed) == 0 or len(packed) % 2:
                raise ValueError("compact reference arrays require integer pairs")
            if kind in {"node", "element"}:
                last_end: int | None = None
                for index in range(0, len(packed), 2):
                    start, end = packed[index], packed[index + 1]
                    if start > end or (
                        last_end is not None and start <= last_end + 1
                    ):
                        raise ValueError("compact ID ranges are not canonical")
                    length += end - start + 1
                    last_end = end
            else:
                pairs = list(zip(packed[::2], packed[1::2]))
                if pairs != sorted(set(pairs)):
                    raise ValueError("compact boundary identities are not canonical")
                length += len(pairs)
            normalized_groups.append((part_id, packed))
        if not normalized_groups:
            raise ValueError("compact reference groups must not be empty")
        if mesh_revision is not None and (
            type(mesh_revision) is not int or mesh_revision < 0
        ):
            raise ValueError("mesh_revision must be a non-negative integer or None")
        if topology is not None and topology.mesh_revision != mesh_revision:
            raise ValueError("topology and compact references must share mesh revision")
        instance._kind = kind
        instance._groups = tuple(normalized_groups)
        instance._length = length
        instance._mesh_revision = mesh_revision
        instance._topology = topology
        return instance

    @property
    def kind(self) -> str:
        return self._kind

    @property
    def mesh_revision(self) -> int | None:
        return self._mesh_revision

    @property
    def topology(self) -> MeshTopologyDirectory | None:
        return self._topology

    def compact_groups(self) -> tuple[tuple[str | None, tuple[int, ...]], ...]:
        return tuple((part_id, tuple(values)) for part_id, values in self._groups)

    def bind_mesh_revision(
        self,
        mesh_revision: int,
        topology: MeshTopologyDirectory | None = None,
    ) -> CompressedMeshEntityRefs:
        if type(mesh_revision) is not int or mesh_revision < 0:
            raise ValueError("mesh_revision must be a non-negative integer")
        if (
            self._mesh_revision is not None
            and self._mesh_revision != mesh_revision
        ):
            raise ValueError(
                "compact mesh references are stale for the requested mesh revision"
            )
        resolved_topology = topology
        if resolved_topology is None and self._topology is not None:
            resolved_topology = self._topology.rebound(mesh_revision)
        return self.from_compact(
            self._kind,
            self.compact_groups(),
            mesh_revision=mesh_revision,
            topology=resolved_topology,
        )

    def filter_part_ids(
        self,
        active_part_ids: frozenset[str],
    ) -> CompressedMeshEntityRefs | None:
        """Retain active Part groups without inflating entity references."""

        groups = tuple(
            (part_id, packed)
            for part_id, packed in self._groups
            if part_id is None or part_id in active_part_ids
        )
        if not groups:
            return None
        return self.from_compact(
            self._kind,
            groups,
            mesh_revision=self._mesh_revision,
            topology=self._topology,
        )

    def remap_part_ids(
        self,
        resolver: Callable[[str | None, str, int], str | None],
    ) -> CompressedMeshEntityRefs:
        """Return a regrouped set without inflating MeshEntityRef objects."""

        by_part: dict[str | None, list[tuple[int, int]]] = {}
        topology_rows: dict[
            tuple[str | None, str, int, int], tuple[int, ...]
        ] = {}
        for part_id, packed in self._groups:
            if self._kind in {"node", "element"}:
                for offset in range(0, len(packed), 2):
                    for identity in range(packed[offset], packed[offset + 1] + 1):
                        owner = resolver(part_id, self._kind, identity)
                        ranges = by_part.setdefault(owner, [])
                        if ranges and ranges[-1][1] + 1 == identity:
                            ranges[-1] = (ranges[-1][0], identity)
                        else:
                            ranges.append((identity, identity))
                continue
            if self._topology is None:
                raise ValueError(
                    "compact edge/face references require a topology directory"
                )
            for offset in range(0, len(packed), 2):
                element_id, local_index = packed[offset], packed[offset + 1]
                owner = resolver(part_id, self._kind, element_id)
                by_part.setdefault(owner, []).append((element_id, local_index))
                topology_rows[(owner, self._kind, element_id, local_index)] = (
                    self._topology.resolve(
                        part_id,
                        self._kind,
                        element_id,
                        local_index,
                        mesh_revision=self._mesh_revision,
                    )
                )
        groups: list[tuple[str | None, tuple[int, ...]]] = []
        for part_id, pairs in sorted(
            by_part.items(),
            key=lambda item: "" if item[0] is None else item[0],
        ):
            canonical_pairs = sorted(pairs)
            if self._kind in {"node", "element"}:
                merged: list[tuple[int, int]] = []
                for start, end in canonical_pairs:
                    if merged and start <= merged[-1][1] + 1:
                        merged[-1] = (merged[-1][0], max(merged[-1][1], end))
                    else:
                        merged.append((start, end))
                canonical_pairs = merged
            groups.append(
                (
                    part_id,
                    tuple(value for pair in canonical_pairs for value in pair),
                )
            )
        topology = (
            MeshTopologyDirectory(self._mesh_revision, topology_rows)
            if topology_rows
            else None
        )
        return self.from_compact(
            self._kind,
            groups,
            mesh_revision=self._mesh_revision,
            topology=topology,
        )

    def require_mesh_revision(self, mesh_revision: int) -> None:
        if self._mesh_revision != mesh_revision:
            raise ValueError(
                "compact mesh references are stale for the current mesh revision"
            )

    def __len__(self) -> int:
        return self._length

    def __iter__(self) -> Iterator[MeshEntityRef]:
        factory = {
            "node": MeshEntityRef.node,
            "element": MeshEntityRef.element,
            "edge": MeshEntityRef.edge,
            "face": MeshEntityRef.face,
        }[self._kind]
        for part_id, packed in self._groups:
            if self._kind in {"node", "element"}:
                for offset in range(0, len(packed), 2):
                    for identity in range(packed[offset], packed[offset + 1] + 1):
                        yield factory(identity, part_id=part_id)
                continue
            if self._topology is None:
                raise ValueError(
                    "compact edge/face references require a topology directory"
                )
            for offset in range(0, len(packed), 2):
                element_id, local_index = packed[offset], packed[offset + 1]
                node_ids = self._topology.resolve(
                    part_id,
                    self._kind,
                    element_id,
                    local_index,
                    mesh_revision=self._mesh_revision,
                )
                yield factory(
                    element_id,
                    local_index,
                    node_ids,
                    part_id=part_id,
                )

    def __getitem__(self, index: int | slice) -> MeshEntityRef | tuple[MeshEntityRef, ...]:
        if isinstance(index, slice):
            return tuple(list(self)[index])
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError("compact mesh reference index out of range")
        for offset, value in enumerate(self):
            if offset == index:
                return value
        raise IndexError("compact mesh reference index out of range")

    def __contains__(self, candidate: object) -> bool:
        if type(candidate) is not MeshEntityRef or candidate.kind != self._kind:
            return False
        group = next(
            (packed for part_id, packed in self._groups if part_id == candidate.part_id),
            None,
        )
        if group is None:
            return False
        if self._kind in {"node", "element"}:
            identity = int(
                candidate.node_id if self._kind == "node" else candidate.element_id
            )
            low = 0
            high = len(group) // 2
            while low < high:
                middle = (low + high) // 2
                if group[middle * 2] <= identity:
                    low = middle + 1
                else:
                    high = middle
            position = low - 1
            return (
                position >= 0
                and identity <= group[position * 2 + 1]
            )
        pair = (int(candidate.element_id), int(candidate.local_index))
        low = 0
        high = len(group) // 2
        while low < high:
            middle = (low + high) // 2
            current = (group[middle * 2], group[middle * 2 + 1])
            if current < pair:
                low = middle + 1
            else:
                high = middle
        if low >= len(group) // 2 or (
            group[low * 2], group[low * 2 + 1]
        ) != pair:
            return False
        if self._topology is None:
            return False
        return self._topology.resolve(
            candidate.part_id,
            self._kind,
            *pair,
            mesh_revision=self._mesh_revision,
        ) == candidate.node_ids

    def __eq__(self, other: object) -> bool:
        if isinstance(other, CompressedMeshEntityRefs):
            if (
                self._kind != other._kind
                or self._groups != other._groups
            ):
                return False
            if self._kind in {"node", "element"}:
                return True
            if self._topology is None or other._topology is None:
                return self._topology is other._topology
            for part_id, packed in self._groups:
                for offset in range(0, len(packed), 2):
                    element_id, local_index = packed[offset], packed[offset + 1]
                    if self._topology.resolve(
                        part_id,
                        self._kind,
                        element_id,
                        local_index,
                        mesh_revision=self._mesh_revision,
                    ) != other._topology.resolve(
                        part_id,
                        other._kind,
                        element_id,
                        local_index,
                        mesh_revision=other._mesh_revision,
                    ):
                        return False
            return True
        if isinstance(other, Sequence):
            return tuple(self) == tuple(other)
        return NotImplemented

    def __hash__(self) -> int:
        topology = ()
        if self._kind in {"edge", "face"} and self._topology is not None:
            topology = tuple(
                self._topology.resolve(
                    part_id,
                    self._kind,
                    packed[offset],
                    packed[offset + 1],
                    mesh_revision=self._mesh_revision,
                )
                for part_id, packed in self._groups
                for offset in range(0, len(packed), 2)
            )
        return hash((self._kind, self.compact_groups(), topology))

    def __deepcopy__(self, memo: dict[int, Any]) -> CompressedMeshEntityRefs:
        del memo
        return self


@dataclass(frozen=True, slots=True)
class NamedRegion:
    """One user-authored scope on a generated finite-element mesh.

    Logical references remain readable for compatibility with older project
    files. New GUI authoring always stores :class:`MeshEntityRef` values.
    """

    name: str
    references: (
        CompressedMeshEntityRefs
        | tuple[MeshEntityRef | LogicalEntityRef, ...]
    )

    def __post_init__(self) -> None:
        if type(self.name) is not str or not self.name.strip():
            raise ValueError("named region name must be a non-empty string")
        compact_input = isinstance(
            self.references, CompressedMeshEntityRefs
        )
        if compact_input:
            references = self.references
            object.__setattr__(self, "name", self.name.strip())
            return
        references = tuple(self.references)
        if not references:
            raise ValueError("named region references must not be empty")
        reference_types = {type(reference) for reference in references}
        if not reference_types.issubset({MeshEntityRef, LogicalEntityRef}):
            raise TypeError(
                "named region references must contain MeshEntityRef or "
                "LogicalEntityRef values"
            )
        if len(reference_types) != 1:
            raise ValueError(
                "one named region cannot mix mesh and logical references"
            )
        if len(set(references)) != len(references):
            raise ValueError("named region references must be unique")
        kinds = {reference.kind for reference in references}
        if len(kinds) != 1:
            raise ValueError("one named region cannot mix entity kinds")
        object.__setattr__(self, "name", self.name.strip())
        sort_key = (
            mesh_entity_ref_sort_key
            if reference_types == {MeshEntityRef}
            else logical_ref_sort_key
        )
        if reference_types == {MeshEntityRef}:
            compact = CompressedMeshEntityRefs(references)
            object.__setattr__(self, "references", compact)
        else:
            object.__setattr__(
                self,
                "references",
                tuple(sorted(references, key=sort_key)),
            )

    def __deepcopy__(self, memo: dict[int, Any]) -> NamedRegion:
        """Reuse this immutable value when command and snapshot state is copied."""

        del memo
        return self

    @property
    def entity_kind(self) -> str:
        """Return the single kind derived from the canonical references."""

        if isinstance(self.references, CompressedMeshEntityRefs):
            return self.references.kind
        return self.references[0].kind

    @property
    def mesh_revision(self) -> int | None:
        if isinstance(self.references, CompressedMeshEntityRefs):
            return self.references.mesh_revision
        return None

    def bind_mesh_revision(
        self,
        mesh_revision: int,
        topology: MeshTopologyDirectory | None = None,
    ) -> NamedRegion:
        if not isinstance(self.references, CompressedMeshEntityRefs):
            return self
        return NamedRegion(
            self.name,
            self.references.bind_mesh_revision(mesh_revision, topology),
        )

    @property
    def logical_ids(self) -> tuple[str, ...]:
        """Return legacy logical IDs for compatibility consumers."""

        return tuple(
            reference.logical_id
            for reference in self.references
            if type(reference) is LogicalEntityRef
        )


@dataclass(frozen=True, slots=True)
class SectionDefinition:
    """Editable section definition with a material linked by name."""

    name: str
    material: str
    section_type: str = "solid"
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RegionAssignment:
    """Assign one named section to an existing element region."""

    section_name: str
    region_name: str
    beam_orientation: BeamOrientation | None = None


@dataclass(frozen=True, slots=True)
class ModelDefinitions:
    """Detached, application-owned editable model definitions."""

    materials: tuple[MaterialDefinition, ...] = ()
    sections: tuple[SectionDefinition, ...] = ()
    assignments: tuple[RegionAssignment, ...] = ()
    steps: tuple[Any, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "materials",
            deepcopy(tuple(self.materials)),
        )
        object.__setattr__(
            self,
            "sections",
            deepcopy(tuple(self.sections)),
        )
        object.__setattr__(
            self,
            "assignments",
            deepcopy(tuple(self.assignments)),
        )
        object.__setattr__(self, "steps", deepcopy(tuple(self.steps)))


@dataclass(frozen=True, slots=True)
class DefinitionCompileResult:
    """Result of compiling definitions without mutating the base model."""

    definitions: ModelDefinitions | None
    model: Any | None
    diagnostics: tuple[PreflightDiagnostic, ...] = ()

    @property
    def passed(self) -> bool:
        return self.model is not None and not any(
            diagnostic.blocking for diagnostic in self.diagnostics
        )

    def require_model(self, *, detach: bool = True) -> Any:
        """Return the compiled model, detached by default, or reject it."""

        if type(detach) is not bool:
            raise TypeError("detach must be a bool")
        if not self.passed:
            raise DefinitionRejected(self.diagnostics)
        return deepcopy(self.model) if detach else self.model


class DefinitionRejected(ValueError):
    """A definitions command was rejected before Session state changed."""

    def __init__(
        self,
        diagnostics: Iterable[PreflightDiagnostic],
    ) -> None:
        self.diagnostics = tuple(deepcopy(tuple(diagnostics)))
        message = "; ".join(
            diagnostic.message for diagnostic in self.diagnostics
        )
        super().__init__(message or "model definitions were rejected")

    @classmethod
    def from_error(cls, error: Exception) -> DefinitionRejected:
        """Create a stable rejection for one input-validation error."""

        return cls((_definition_diagnostic(error),))


def normalize_model_definitions(
    materials: (
        ModelDefinitions
        | Mapping[str, Any]
        | Iterable[Any]
    ) = (),
    sections: Iterable[SectionDefinition] | None = None,
    assignments: Iterable[RegionAssignment] | None = None,
    steps: Iterable[Any] | None = None,
) -> ModelDefinitions:
    """Own, normalize, and validate editable definition inputs."""

    try:
        return _normalize_model_definitions(
            materials,
            sections,
            assignments,
            steps,
        )
    except DefinitionRejected:
        raise
    except (KeyError, TypeError, ValueError) as error:
        raise DefinitionRejected.from_error(error) from error


def _normalize_model_definitions(
    materials: (
        ModelDefinitions
        | Mapping[str, Any]
        | Iterable[Any]
    ),
    sections: Iterable[SectionDefinition] | None,
    assignments: Iterable[RegionAssignment] | None,
    steps: Iterable[Any] | None,
) -> ModelDefinitions:
    if isinstance(materials, ModelDefinitions):
        if any(value is not None for value in (sections, assignments, steps)):
            raise TypeError(
                "separate definition collections cannot accompany "
                "ModelDefinitions"
            )
        source = deepcopy(materials)
        material_values = source.materials
        section_values = source.sections
        assignment_values = source.assignments
        step_values = source.steps
    else:
        material_values = _mapping_values(materials)
        section_values = tuple(() if sections is None else sections)
        assignment_values = tuple(
            () if assignments is None else assignments
        )
        step_values = tuple(() if steps is None else steps)

    owned_materials_list: list[MaterialDefinition] = []
    for index, material in enumerate(deepcopy(tuple(material_values))):
        name = _required_name(material, "material")
        properties = deepcopy(dict(getattr(material, "properties", {})))
        if BEAM_LOCAL_Y_REFERENCE_KEY in properties:
            raise DefinitionRejected(
                (
                    _beam_orientation_diagnostic(
                        code="beam.orientation.invalid",
                        message=(
                            f"material {name!r} must not define reserved "
                            f"property {BEAM_LOCAL_Y_REFERENCE_KEY!r}"
                        ),
                        subject=name,
                        path=(
                            "definitions",
                            "materials",
                            str(index),
                            "properties",
                            BEAM_LOCAL_Y_REFERENCE_KEY,
                        ),
                        details={
                            "material_index": index,
                            "property": BEAM_LOCAL_Y_REFERENCE_KEY,
                        },
                    ),
                )
            )
        owned_materials_list.append(MaterialDefinition(name, properties))
    owned_materials = tuple(owned_materials_list)
    owned_sections_list: list[SectionDefinition] = []
    for index, section in enumerate(deepcopy(tuple(section_values))):
        properties = deepcopy(dict(section.properties))
        if BEAM_LOCAL_Y_REFERENCE_KEY in properties:
            raise DefinitionRejected(
                (
                    _beam_orientation_diagnostic(
                        code="beam.orientation.invalid",
                        message=(
                            f"section {_required_name(section, 'section')!r} "
                            f"must not define reserved property "
                            f"{BEAM_LOCAL_Y_REFERENCE_KEY!r}"
                        ),
                        subject=_required_name(section, "section"),
                        path=(
                            "definitions",
                            "sections",
                            str(index),
                            "properties",
                            BEAM_LOCAL_Y_REFERENCE_KEY,
                        ),
                        details={
                            "section_index": index,
                            "property": BEAM_LOCAL_Y_REFERENCE_KEY,
                        },
                    ),
                )
            )
        owned_sections_list.append(
            SectionDefinition(
                name=_required_name(section, "section"),
                material=str(section.material).strip(),
                section_type=str(section.section_type).strip().casefold(),
                properties=properties,
            )
        )
    owned_sections = tuple(owned_sections_list)

    owned_assignments_list: list[RegionAssignment] = []
    for index, assignment in enumerate(
        deepcopy(tuple(assignment_values))
    ):
        section_name = str(assignment.section_name).strip()
        region_name = str(assignment.region_name).strip()
        orientation = _normalize_beam_orientation(
            getattr(assignment, "beam_orientation", None),
            assignment_index=index,
            region_name=region_name,
        )
        owned_assignments_list.append(
            RegionAssignment(
                section_name=section_name,
                region_name=region_name,
                beam_orientation=orientation,
            )
        )
    owned_assignments = tuple(owned_assignments_list)
    owned_steps = deepcopy(tuple(step_values))
    for step in owned_steps:
        name = _required_name(step, "analysis step")
        step.name = name

    _validate_unique_names(owned_materials, "material")
    _validate_unique_names(owned_sections, "section")
    _validate_unique_names(owned_steps, "analysis step")
    validate_analysis_object_names(owned_steps, require_all=False)
    _validate_definition_links(
        owned_materials,
        owned_sections,
        owned_assignments,
    )
    return ModelDefinitions(
        owned_materials,
        owned_sections,
        owned_assignments,
        owned_steps,
    )


def definitions_from_model(model: Any) -> ModelDefinitions:
    """Project a kernel model into one detached editable snapshot."""

    materials = deepcopy(tuple(getattr(model, "materials", {}).values()))
    sections: list[SectionDefinition] = []
    assignments: list[RegionAssignment] = []
    for index, section in enumerate(
        getattr(model, "sections", ()),
        start=1,
    ):
        name = f"Section-{index}"
        properties = deepcopy(dict(section.properties))
        orientation = None
        if BEAM_LOCAL_Y_REFERENCE_KEY in properties:
            raw_orientation = properties.pop(BEAM_LOCAL_Y_REFERENCE_KEY)
            try:
                orientation = parse_beam_orientation(raw_orientation)
            except BeamOrientationError as error:
                raise DefinitionRejected(
                    (
                        _beam_orientation_diagnostic(
                            code="beam.orientation.invalid",
                            message=str(error),
                            subject=_element_set_subject(
                                str(section.element_set)
                            ),
                            path=(
                                "definitions",
                                "assignments",
                                str(index - 1),
                                "beam_orientation",
                            ),
                            details={
                                "assignment_index": index - 1,
                                "element_set": str(section.element_set),
                                "reference": deepcopy(raw_orientation),
                                "error_type": type(error).__name__,
                            },
                        ),
                    )
                ) from error
        sections.append(
            SectionDefinition(
                name=name,
                material=str(section.material),
                section_type=str(section.section_type),
                properties=properties,
            )
        )
        assignments.append(
            RegionAssignment(
                section_name=name,
                region_name=str(section.element_set),
                beam_orientation=orientation,
            )
        )
    definitions = normalize_model_definitions(
        materials,
        sections,
        assignments,
        deepcopy(tuple(getattr(model, "steps", ()))),
    )
    target_diagnostics = _orientation_target_diagnostics(
        model,
        definitions,
    )
    if target_diagnostics:
        raise DefinitionRejected(target_diagnostics)
    if assignments:
        orientation_diagnostics = _compiled_orientation_diagnostics(
            model,
            definitions,
        )
        if orientation_diagnostics:
            raise DefinitionRejected(orientation_diagnostics)
    return definitions


def compile_model_definitions(
    base_model: Any,
    definitions: (
        ModelDefinitions
        | Mapping[str, Any]
        | Iterable[Any]
    ),
    sections: Iterable[SectionDefinition] | None = None,
    assignments: Iterable[RegionAssignment] | None = None,
    steps: Iterable[Any] | None = None,
    *,
    detach_model: bool = True,
) -> DefinitionCompileResult:
    """Compile definitions, detaching the model by default, and report errors."""

    try:
        if type(detach_model) is not bool:
            raise TypeError("detach_model must be a bool")
        normalized = normalize_model_definitions(
            definitions,
            sections,
            assignments,
            steps,
        )
        compiled = deepcopy(base_model) if detach_model else base_model
        material_map = {
            material.name: MaterialDefinition(
                material.name,
                deepcopy(dict(material.properties)),
            )
            for material in normalized.materials
        }
        section_map = {
            section.name: section for section in normalized.sections
        }
        metadata = getattr(compiled, "metadata", {})
        element_sets = dict(
            metadata.get("_abaqus_internal_element_sets", {})
        )
        element_sets.update(dict(getattr(compiled, "element_sets", {})))

        compiled_sections: list[SectionAssignment] = []
        for assignment in normalized.assignments:
            section = section_map[assignment.section_name]
            if assignment.region_name not in element_sets:
                raise KeyError(
                    f"region {assignment.region_name!r} is not an element set"
                )
            properties = deepcopy(dict(section.properties))
            if assignment.beam_orientation is not None:
                properties[BEAM_LOCAL_Y_REFERENCE_KEY] = tuple(
                    assignment.beam_orientation.local_y_reference
                )
            compiled_sections.append(
                SectionAssignment(
                    element_set=assignment.region_name,
                    material=section.material,
                    section_type=section.section_type,
                    properties=properties,
                )
            )

        compiled.materials = material_map
        compiled.sections = compiled_sections
        compiled.steps = deepcopy(list(normalized.steps))
        if compiled_sections:
            _invalidate_changed_beam_frame_fields(
                base_model,
                compiled,
                normalized.assignments,
                element_sets,
            )
            target_diagnostics = _orientation_target_diagnostics(
                compiled,
                normalized,
            )
            if target_diagnostics:
                return DefinitionCompileResult(
                    definitions=deepcopy(normalized),
                    model=None,
                    diagnostics=target_diagnostics,
                )
            from fem.materials import resolve_sections

            resolution = resolve_sections(compiled)
            if resolution.issues:
                return DefinitionCompileResult(
                    definitions=deepcopy(normalized),
                    model=None,
                    diagnostics=tuple(
                        _section_resolution_diagnostic(
                            issue,
                            normalized,
                        )
                        for issue in resolution.issues
                    ),
                )
            orientation_diagnostics = (
                _compiled_orientation_diagnostics(
                    compiled,
                    normalized,
                )
            )
            if orientation_diagnostics:
                return DefinitionCompileResult(
                    definitions=deepcopy(normalized),
                    model=None,
                    diagnostics=orientation_diagnostics,
                )
    except DefinitionRejected as error:
        return DefinitionCompileResult(
            definitions=None,
            model=None,
            diagnostics=error.diagnostics,
        )
    except (KeyError, TypeError, ValueError) as error:
        diagnostic = _definition_diagnostic(error)
        return DefinitionCompileResult(
            definitions=None,
            model=None,
            diagnostics=(diagnostic,),
        )
    return DefinitionCompileResult(
        definitions=deepcopy(normalized),
        model=compiled,
        diagnostics=(),
    )


def _invalidate_changed_beam_frame_fields(
    base_model: Any,
    compiled_model: Any,
    assignments: tuple[RegionAssignment, ...],
    element_sets: Mapping[str, Any],
) -> None:
    """Drop imported frame fields when editable assignment orientation changes."""

    base_orientations: dict[
        str,
        tuple[float, float, float] | None,
    ] = {}
    for section in tuple(getattr(base_model, "sections", ())):
        properties = getattr(section, "properties", {})
        raw_orientation = (
            properties.get(BEAM_LOCAL_Y_REFERENCE_KEY)
            if isinstance(properties, Mapping)
            else None
        )
        orientation = (
            None
            if raw_orientation is None
            else parse_beam_orientation(raw_orientation).local_y_reference
        )
        base_orientations[str(getattr(section, "element_set", ""))] = (
            orientation
        )

    changed_regions = {
        assignment.region_name
        for assignment in assignments
        if (
            None
            if assignment.beam_orientation is None
            else assignment.beam_orientation.local_y_reference
        )
        != base_orientations.get(assignment.region_name)
    }
    if not changed_regions:
        return

    element_lookup = {
        int(element.id): element
        for element in tuple(
            getattr(getattr(compiled_model, "mesh", None), "elements", ())
        )
    }
    for region_name in changed_regions:
        element_set = element_sets.get(region_name)
        if element_set is None:
            continue
        for raw_element_id in tuple(
            getattr(element_set, "element_ids", ())
        ):
            element = element_lookup.get(int(raw_element_id))
            properties = getattr(element, "props", None)
            if not isinstance(properties, dict):
                continue
            properties.pop(BEAM_FRAME_FIELD_KEY, None)
            properties.pop(BEAM_FRAME_FIELD_REFERENCE_KEY, None)
            properties.pop(BEAM_ELEMENT_LOCAL_Y_REFERENCE_KEY, None)


def compiled_model_snapshot(
    base_model: Any,
    definitions: ModelDefinitions,
) -> Any:
    """Return a detached compiled model or raise ``DefinitionRejected``."""

    return compile_model_definitions(
        base_model,
        definitions,
    ).require_model()


def _normalize_beam_orientation(
    value: Any,
    *,
    assignment_index: int,
    region_name: str,
) -> BeamOrientation | None:
    if value is None:
        return None

    if not isinstance(value, BeamOrientation):
        error = TypeError(
            "assignment beam_orientation must be BeamOrientation or None"
        )
        raise DefinitionRejected(
            (
                _beam_orientation_diagnostic(
                    code="beam.orientation.invalid",
                    message=str(error),
                    subject=_element_set_subject(region_name),
                    path=(
                        "definitions",
                        "assignments",
                        str(assignment_index),
                        "beam_orientation",
                    ),
                    details={
                        "assignment_index": assignment_index,
                        "element_set": region_name,
                        "value_type": type(value).__name__,
                    },
                ),
            )
        ) from error
    return deepcopy(value)


def _compiled_orientation_diagnostics(
    model: Any,
    definitions: ModelDefinitions,
) -> tuple[PreflightDiagnostic, ...]:
    """Validate every authored explicit direction against its whole target."""

    from fem.materials import (
        MaterialPropertyError,
        SectionCompatibilityError,
        SectionPropertyError,
        resolve_section_properties,
        restored_element_properties,
    )

    element_lookup = {
        int(element.id): element
        for element in getattr(
            getattr(model, "mesh", None),
            "elements",
            (),
        )
    }
    metadata = getattr(model, "metadata", {})
    element_sets = dict(
        metadata.get("_abaqus_internal_element_sets", {})
    )
    element_sets.update(dict(getattr(model, "element_sets", {})))
    core_sections = tuple(getattr(model, "sections", ()))
    materials = getattr(model, "materials", {})
    diagnostics: list[PreflightDiagnostic] = []

    for assignment_index, assignment in enumerate(definitions.assignments):
        orientation = assignment.beam_orientation
        if orientation is None:
            continue
        if assignment_index >= len(core_sections):
            continue
        core_section = core_sections[assignment_index]
        element_set = element_sets.get(assignment.region_name)
        material = materials.get(str(core_section.material))
        if element_set is None or material is None:
            continue
        for raw_element_id in getattr(element_set, "element_ids", ()):
            element_id = int(raw_element_id)
            element = element_lookup.get(element_id)
            if element is None:
                continue
            try:
                resolved = resolve_section_properties(
                    str(element.type),
                    material.properties,
                    str(core_section.section_type),
                    core_section.properties,
                    baseline_properties=restored_element_properties(
                        model,
                        element_id,
                        element,
                    ),
                )
            except (
                MaterialPropertyError,
                SectionCompatibilityError,
                SectionPropertyError,
                NotImplementedError,
                KeyError,
                TypeError,
                ValueError,
            ):
                # The ordinary section-resolution diagnostics own these
                # schema/reference failures.
                continue
            try:
                resolve_beam_frame_field(
                    model.mesh,
                    element,
                    properties=deepcopy(resolved.effective_properties),
                )
            except BeamOrientationError as error:
                details = {
                    "assignment_index": assignment_index,
                    "element_set": assignment.region_name,
                    "element_id": element_id,
                    "reference": tuple(orientation.local_y_reference),
                    "operation": "section.assignment",
                    "error_type": type(error).__name__,
                }
                tangent = getattr(error, "tangent", None)
                if tangent is not None:
                    details["element_tangent"] = deepcopy(tangent)
                diagnostics.append(
                    _beam_orientation_diagnostic(
                        code=_beam_frame_error_code(error),
                        message=str(error),
                        subject=_element_set_subject(
                            assignment.region_name
                        ),
                        path=(
                            "definitions",
                            "assignments",
                            str(assignment_index),
                            "beam_orientation",
                        ),
                        details=details,
                    )
                )
    return tuple(diagnostics)


def _orientation_target_diagnostics(
    model: Any,
    definitions: ModelDefinitions,
) -> tuple[PreflightDiagnostic, ...]:
    oriented_assignments = tuple(
        (index, assignment)
        for index, assignment in enumerate(definitions.assignments)
        if assignment.beam_orientation is not None
    )
    if not oriented_assignments:
        return ()

    element_lookup = {
        int(element.id): element
        for element in getattr(
            getattr(model, "mesh", None),
            "elements",
            (),
        )
    }
    element_sets = dict(
        getattr(model, "metadata", {}).get(
            "_abaqus_internal_element_sets",
            {},
        )
    )
    element_sets.update(dict(getattr(model, "element_sets", {})))
    diagnostics: list[PreflightDiagnostic] = []

    for assignment_index, assignment in oriented_assignments:
        orientation = assignment.beam_orientation
        assert orientation is not None
        element_set = element_sets.get(assignment.region_name)
        if element_set is None:
            # The ordinary section-resolution path reports this reference.
            continue
        target_element_ids = tuple(
            getattr(element_set, "element_ids", ())
        )
        if not target_element_ids:
            diagnostics.append(
                _beam_orientation_diagnostic(
                    code="beam.orientation.unsupported_target",
                    message=(
                        "Beam orientation requires a non-empty Beam2 "
                        f"element set; {assignment.region_name!r} is empty"
                    ),
                    subject=_element_set_subject(
                        assignment.region_name
                    ),
                    path=(
                        "definitions",
                        "assignments",
                        str(assignment_index),
                        "beam_orientation",
                    ),
                    details={
                        "assignment_index": assignment_index,
                        "element_set": assignment.region_name,
                        "reference": tuple(
                            orientation.local_y_reference
                        ),
                        "operation": "section.assignment",
                    },
                )
            )
            continue
        for raw_element_id in target_element_ids:
            element_id = int(raw_element_id)
            element = element_lookup.get(element_id)
            if element is None:
                continue
            try:
                descriptor = get_element_capabilities(str(element.type))
            except (KeyError, NotImplementedError, TypeError, ValueError):
                descriptor = None
            if (
                descriptor is None
                or descriptor.canonical_type != "Beam2"
            ):
                diagnostics.append(
                    _beam_orientation_diagnostic(
                        code="beam.orientation.unsupported_target",
                        message=(
                            "Beam orientation can target only Beam2 "
                            f"elements; element {element_id} has type "
                            f"{element.type!r}"
                        ),
                        subject=_element_set_subject(
                            assignment.region_name
                        ),
                        path=(
                            "definitions",
                            "assignments",
                            str(assignment_index),
                            "beam_orientation",
                        ),
                        details={
                            "assignment_index": assignment_index,
                            "element_set": assignment.region_name,
                            "element_id": element_id,
                            "element_type": str(element.type),
                            "reference": tuple(
                                orientation.local_y_reference
                            ),
                            "operation": "section.assignment",
                        },
                    )
                )
    return tuple(diagnostics)


def _beam_frame_error_code(error: Exception) -> str:
    code = getattr(error, "code", None)
    if code in {
        "beam.orientation.invalid",
        "beam.orientation.parallel",
        "beam.orientation.unsupported_target",
    }:
        return str(code)
    return "beam.orientation.invalid"


def _element_set_subject(name: str) -> Any:
    try:
        return RegionRef("element_set", name)
    except ValueError:
        return str(name)


def _mapping_values(
    value: Mapping[Any, Any] | Iterable[Any],
) -> tuple[Any, ...]:
    if isinstance(value, Mapping):
        return tuple(value.values())
    return tuple(value)


def _required_name(value: Any, label: str) -> str:
    if not hasattr(value, "name"):
        raise TypeError(f"{label} is missing a name")
    name = str(value.name).strip()
    if not name:
        raise ValueError(f"{label} name must not be empty")
    return name


def _validate_unique_names(values: Iterable[Any], label: str) -> None:
    seen: dict[str, str] = {}
    for value in values:
        name = str(value.name)
        key = name.casefold()
        if key in seen:
            raise ValueError(
                f"{label} names must be unique ignoring case: "
                f"{seen[key]!r} and {name!r}"
            )
        seen[key] = name


def _validate_definition_links(
    materials: tuple[MaterialDefinition, ...],
    sections: tuple[SectionDefinition, ...],
    assignments: tuple[RegionAssignment, ...],
) -> None:
    material_names = {material.name for material in materials}
    section_names = {section.name for section in sections}
    for section in sections:
        if not section.material:
            raise ValueError(
                f"section {section.name!r} material must not be empty"
            )
        if section.material not in material_names:
            raise ValueError(
                f"section {section.name!r} references missing material "
                f"{section.material!r}"
            )
    for assignment in assignments:
        if not assignment.section_name:
            raise ValueError(
                "assignment section name must not be empty"
            )
        if not assignment.region_name:
            raise ValueError("assignment region name must not be empty")
        if assignment.section_name not in section_names:
            raise ValueError(
                "assignment references missing section "
                f"{assignment.section_name!r}"
            )


def _definition_diagnostic(error: Exception) -> PreflightDiagnostic:
    message = str(error)
    lowered = message.casefold()
    if "material" in lowered:
        code = "definition.material.missing"
    elif "section" in lowered:
        code = "definition.section.missing"
    else:
        code = "step.reference.invalid"
    return PreflightDiagnostic(
        code=code,
        severity=PreflightSeverity.ERROR,
        stage=PreflightStage.DEFINITIONS,
        message=message,
        subject="model_definitions",
        path=("definitions",),
        remediation="请修正名称、引用和目标区域后重试。",
        details={"error_type": type(error).__name__},
    )


def _beam_orientation_diagnostic(
    *,
    code: str,
    message: str,
    subject: Any,
    path: Iterable[str],
    details: Mapping[str, Any],
) -> PreflightDiagnostic:
    remediation = {
        "beam.orientation.invalid": (
            "请提供三个有限、非零的全局局部 y 参考方向分量。"
        ),
        "beam.orientation.parallel": (
            "请让参考方向与目标梁单元轴线保持明显非平行。"
        ),
        "beam.orientation.unsupported_target": (
            "请仅将 Beam orientation 用于完全由 Beam2 单元组成的区域。"
        ),
    }.get(code, "请修正 Beam orientation 后重试。")
    return PreflightDiagnostic(
        code=code,
        severity=PreflightSeverity.ERROR,
        stage=PreflightStage.DEFINITIONS,
        message=str(message),
        subject=subject,
        path=tuple(path),
        remediation=remediation,
        details=details,
    )


def _section_resolution_diagnostic(
    issue: Any,
    definitions: ModelDefinitions | None = None,
) -> PreflightDiagnostic:
    code = (
        "definition.section.missing"
        if issue.code == "definition.section.reference_missing"
        else issue.code
    )
    if str(code).startswith("beam.orientation."):
        assignment_index = issue.assignment_index
        orientation = None
        if (
            definitions is not None
            and assignment_index is not None
            and 0 <= int(assignment_index) < len(definitions.assignments)
        ):
            orientation = definitions.assignments[
                int(assignment_index)
            ].beam_orientation
        details: dict[str, Any] = {
            "assignment_index": assignment_index,
            "element_id": issue.element_id,
            "element_set": issue.element_set,
            "material": issue.material,
            "section_type": issue.section_type,
            "operation": "section.assignment",
        }
        if orientation is not None:
            details["reference"] = tuple(
                orientation.local_y_reference
            )
        return _beam_orientation_diagnostic(
            code=str(code),
            message=issue.message,
            subject=(
                _element_set_subject(issue.element_set)
                if issue.element_set is not None
                else issue.element_id
            ),
            path=(
                "definitions",
                "assignments",
                str(assignment_index),
                "beam_orientation",
            ),
            details=details,
        )
    return PreflightDiagnostic(
        code=code,
        severity=PreflightSeverity.ERROR,
        stage=PreflightStage.DEFINITIONS,
        message=issue.message,
        subject=(
            RegionRef("element_set", issue.element_set)
            if issue.element_set is not None
            else issue.element_id
        ),
        path=(
            "definitions",
            "sections",
            str(issue.assignment_index),
        ),
        remediation="请修复材料、截面参数或目标单元集。",
        details={
            "element_id": issue.element_id,
            "material": issue.material,
            "section_type": issue.section_type,
        },
    )


__all__ = [
    "DefinitionCompileResult",
    "DefinitionRejected",
    "FeatureRecord",
    "ModelDefinitions",
    "NamedRegion",
    "NativePart",
    "RegionAssignment",
    "SectionDefinition",
    "compile_model_definitions",
    "compiled_model_snapshot",
    "definitions_from_model",
    "normalize_model_definitions",
]
