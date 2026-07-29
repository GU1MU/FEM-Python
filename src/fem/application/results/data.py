"""Owned, immutable result data shared by headless application consumers."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from enum import Enum
import math
from numbers import Real
from typing import Any

import numpy as np

from fem.post.fields import ResultRegionKey

from .fields import (
    FieldAssociation,
    FieldMaterializationKey,
    FieldPosition,
    PhysicalQuantity,
    ResultFieldId,
    ResultSourceKey,
    ScalarFieldSelection,
    field_materialization_sort_key,
)


class FieldState(str, Enum):
    """Availability of one fully resolved materialization key."""

    READY = "ready"
    LAZY = "lazy"
    UNAVAILABLE = "unavailable"


_POSITION_ASSOCIATIONS = {
    FieldPosition.NODE: FieldAssociation.NODE,
    FieldPosition.INTEGRATION_POINT: FieldAssociation.INTEGRATION_POINT,
    FieldPosition.CENTROID: FieldAssociation.ELEMENT,
    FieldPosition.ELEMENT_NODAL: FieldAssociation.ELEMENT_NODE,
    FieldPosition.NODE_REGION: FieldAssociation.NODE_REGION,
    FieldPosition.RESOLVED_NODAL: FieldAssociation.RESOLVED_NODAL,
    FieldPosition.SECTION_END: FieldAssociation.ELEMENT_NODE,
    FieldPosition.SECTION_NODE_ENVELOPE: FieldAssociation.NODE,
}


class _FrozenJsonObject(Mapping[str, Any]):
    """Small immutable mapping used for detached diagnostic details."""

    __slots__ = ("_items",)

    def __init__(self, items: tuple[tuple[str, Any], ...]) -> None:
        object.__setattr__(self, "_items", items)

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise TypeError("frozen JSON mappings are immutable")

    def __delattr__(self, name: str) -> None:
        del name
        raise TypeError("frozen JSON mappings are immutable")

    def __getitem__(self, key: str) -> Any:
        for candidate, value in self._items:
            if candidate == key:
                return value
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return (key for key, _value in self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __repr__(self) -> str:
        return repr(dict(self._items))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Mapping):
            return False
        return dict(self._items) == dict(other.items())

    def __hash__(self) -> int:
        return hash(self._items)

    def __copy__(self) -> _FrozenJsonObject:
        return self

    def __deepcopy__(self, memo: dict[int, Any]) -> _FrozenJsonObject:
        del memo
        return self


@dataclass(frozen=True, slots=True)
class FieldDescriptor:
    """Registry-owned engineering description of one canonical field."""

    field_id: ResultFieldId
    association: FieldAssociation
    quantity: PhysicalQuantity
    components: tuple[str, ...]
    derived_components: tuple[str, ...]
    label_key: str
    unit_label: str | None
    default_component: str
    order: int

    def __post_init__(self) -> None:
        if type(self.field_id) is not ResultFieldId:
            raise TypeError("field_id must be ResultFieldId")
        if type(self.association) is not FieldAssociation:
            raise TypeError("association must be FieldAssociation")
        if type(self.quantity) is not PhysicalQuantity:
            raise TypeError("quantity must be PhysicalQuantity")
        expected_association = _POSITION_ASSOCIATIONS[self.field_id.position]
        if self.association is not expected_association:
            raise ValueError(
                f"{self.field_id.position.value} fields require "
                f"{expected_association.value} association"
            )

        components = _strict_string_tuple(
            self.components,
            label="components",
        )
        derived = _strict_string_tuple(
            self.derived_components,
            label="derived_components",
        )
        columns = components + derived
        if not columns:
            raise ValueError("descriptor must publish at least one component")
        if len(set(columns)) != len(columns):
            raise ValueError("descriptor component names must be unique")
        object.__setattr__(self, "components", components)
        object.__setattr__(self, "derived_components", derived)

        _require_nonblank_string(self.label_key, label="label_key")
        if self.unit_label is not None:
            _require_nonblank_string(self.unit_label, label="unit_label")
        _require_nonblank_string(
            self.default_component,
            label="default_component",
        )
        if self.default_component not in columns:
            raise ValueError(
                "default_component must name a descriptor component"
            )
        if type(self.order) is not int:
            raise TypeError("order must be an integer")
        if self.order < 0:
            raise ValueError("order must be non-negative")

    @property
    def columns(self) -> tuple[str, ...]:
        """Return storage columns in their canonical order."""

        return self.components + self.derived_components


@dataclass(frozen=True, slots=True)
class FieldLocation:
    """Typed FEM identity and physical sample location for one field row."""

    association: FieldAssociation
    coordinates: tuple[float, float, float]
    displacement: tuple[float, float, float] | None
    node_id: int | None = None
    element_id: int | None = None
    integration_point: int | None = None
    local_node: int | None = None
    region_key: ResultRegionKey | None = None
    averaged: bool | None = None

    def __post_init__(self) -> None:
        if type(self.association) is not FieldAssociation:
            raise TypeError("association must be FieldAssociation")
        object.__setattr__(
            self,
            "coordinates",
            _finite_triplet(self.coordinates, label="coordinates"),
        )
        if self.displacement is not None:
            object.__setattr__(
                self,
                "displacement",
                _finite_triplet(
                    self.displacement,
                    label="displacement",
                ),
            )

        _validate_location_identity(self)


@dataclass(frozen=True, slots=True)
class ResultDiagnostic:
    """Detached, machine-readable result-domain diagnostic."""

    code: str
    severity: str
    message: str
    path: tuple[object, ...]
    remediation: str
    details: Mapping[str, Any]

    def __post_init__(self) -> None:
        _require_nonblank_string(self.code, label="code")
        _require_nonblank_string(self.severity, label="severity")
        _require_nonblank_string(self.message, label="message")
        _require_nonblank_string(self.remediation, label="remediation")
        if type(self.path) is not tuple:
            raise TypeError("path must be a tuple")
        object.__setattr__(
            self,
            "path",
            tuple(
                _freeze_json_value(
                    item,
                    path=f"path[{index}]",
                    ancestors=set(),
                )
                for index, item in enumerate(self.path)
            ),
        )
        if not isinstance(self.details, Mapping):
            raise TypeError("details must be a mapping")
        object.__setattr__(
            self,
            "details",
            _freeze_json_mapping(
                self.details,
                path="details",
                ancestors=set(),
            ),
        )


@dataclass(frozen=True, slots=True)
class FieldAvailability:
    """Catalog state for one fully resolved field key."""

    key: FieldMaterializationKey
    descriptor: FieldDescriptor
    state: FieldState
    diagnostics: tuple[ResultDiagnostic, ...] = ()

    def __post_init__(self) -> None:
        if type(self.key) is not FieldMaterializationKey:
            raise TypeError("key must be FieldMaterializationKey")
        if type(self.descriptor) is not FieldDescriptor:
            raise TypeError("descriptor must be FieldDescriptor")
        if type(self.state) is not FieldState:
            raise TypeError("state must be FieldState")
        _validate_diagnostic_tuple(self.diagnostics)
        if self.key.request.field_id != self.descriptor.field_id:
            raise ValueError("key field_id must match descriptor field_id")


@dataclass(frozen=True, slots=True)
class ResultCatalog:
    """Immutable published catalog with an optional default when empty."""

    source: ResultSourceKey
    fields: tuple[FieldAvailability, ...]
    default_selection: ScalarFieldSelection | None
    diagnostics: tuple[ResultDiagnostic, ...] = ()

    def __post_init__(self) -> None:
        if type(self.source) is not ResultSourceKey:
            raise TypeError("source must be ResultSourceKey")
        if type(self.fields) is not tuple:
            raise TypeError("fields must be a tuple")
        for availability in self.fields:
            if type(availability) is not FieldAvailability:
                raise TypeError(
                    "fields must contain only FieldAvailability values"
                )
        _validate_diagnostic_tuple(self.diagnostics)
        _validate_unique_sorted_keys(
            tuple(item.key for item in self.fields),
            label="catalog fields",
        )
        if not self.fields:
            if self.default_selection is not None:
                raise ValueError(
                    "empty result catalogs cannot have a default selection"
                )
            return
        if type(self.default_selection) is not ScalarFieldSelection:
            raise TypeError(
                "non-empty catalogs require a ScalarFieldSelection default"
            )
        selected = tuple(
            item
            for item in self.fields
            if item.key == self.default_selection.field_key
        )
        if len(selected) != 1:
            raise ValueError(
                "default_selection must reference exactly one catalog field"
            )
        availability = selected[0]
        if availability.state is FieldState.UNAVAILABLE:
            raise ValueError(
                "default_selection cannot reference an unavailable field"
            )
        if (
            self.default_selection.component
            not in availability.descriptor.columns
        ):
            raise ValueError(
                "default selection component is not in the descriptor"
            )


@dataclass(frozen=True, slots=True, eq=False, init=False)
class ResultTopologyProjection:
    """Owned mesh topology and deformation data in canonical mesh order."""

    source: ResultSourceKey
    node_ids: tuple[int, ...]
    _node_coordinates: np.ndarray = field(repr=False)
    _nodal_displacements: np.ndarray = field(repr=False)
    element_ids: tuple[int, ...]
    element_types: tuple[str, ...]
    connectivity: tuple[tuple[int, ...], ...]
    element_region_keys: tuple[ResultRegionKey, ...]

    def __init__(
        self,
        source: ResultSourceKey,
        node_ids: tuple[int, ...],
        node_coordinates: np.ndarray,
        nodal_displacements: np.ndarray,
        element_ids: tuple[int, ...],
        element_types: tuple[str, ...],
        connectivity: tuple[tuple[int, ...], ...],
        element_region_keys: tuple[ResultRegionKey, ...],
    ) -> None:
        if type(source) is not ResultSourceKey:
            raise TypeError("source must be ResultSourceKey")
        checked_node_ids = _positive_unique_id_tuple(
            node_ids,
            label="node_ids",
        )
        coordinates = _owned_finite_matrix(
            node_coordinates,
            label="node_coordinates",
            columns=3,
            rows=len(checked_node_ids),
        )
        displacements = _owned_finite_matrix(
            nodal_displacements,
            label="nodal_displacements",
            columns=3,
            rows=len(checked_node_ids),
        )
        checked_element_ids = _positive_unique_id_tuple(
            element_ids,
            label="element_ids",
        )
        checked_element_types = _strict_string_tuple(
            element_types,
            label="element_types",
        )
        checked_connectivity = _connectivity_tuple(
            connectivity,
            node_ids=frozenset(checked_node_ids),
        )
        checked_region_keys = _region_key_tuple(element_region_keys)
        element_count = len(checked_element_ids)
        if len(checked_element_types) != element_count:
            raise ValueError(
                "element_types length must equal element_ids length"
            )
        if len(checked_connectivity) != element_count:
            raise ValueError(
                "connectivity length must equal element_ids length"
            )
        if len(checked_region_keys) != element_count:
            raise ValueError(
                "element_region_keys length must equal element_ids length"
            )

        object.__setattr__(self, "source", source)
        object.__setattr__(self, "node_ids", checked_node_ids)
        object.__setattr__(self, "_node_coordinates", coordinates)
        object.__setattr__(self, "_nodal_displacements", displacements)
        object.__setattr__(self, "element_ids", checked_element_ids)
        object.__setattr__(self, "element_types", checked_element_types)
        object.__setattr__(self, "connectivity", checked_connectivity)
        object.__setattr__(
            self,
            "element_region_keys",
            checked_region_keys,
        )

    @property
    def node_coordinates(self) -> np.ndarray:
        """Return a detached readonly copy of base node coordinates."""

        return _public_array_copy(self._node_coordinates)

    @property
    def nodal_displacements(self) -> np.ndarray:
        """Return a detached readonly copy of nodal translations."""

        return _public_array_copy(self._nodal_displacements)


@dataclass(frozen=True, slots=True, eq=False, init=False)
class FieldData:
    """Owned numeric rows for one complete materialization key."""

    descriptor: FieldDescriptor
    source: ResultSourceKey
    key: FieldMaterializationKey
    locations: tuple[FieldLocation, ...]
    _values: np.ndarray = field(repr=False)

    def __init__(
        self,
        descriptor: FieldDescriptor,
        source: ResultSourceKey,
        key: FieldMaterializationKey,
        locations: tuple[FieldLocation, ...],
        values: np.ndarray,
    ) -> None:
        if type(descriptor) is not FieldDescriptor:
            raise TypeError("descriptor must be FieldDescriptor")
        if type(source) is not ResultSourceKey:
            raise TypeError("source must be ResultSourceKey")
        if type(key) is not FieldMaterializationKey:
            raise TypeError("key must be FieldMaterializationKey")
        if key.request.field_id != descriptor.field_id:
            raise ValueError("key field_id must match descriptor field_id")
        if type(locations) is not tuple:
            raise TypeError("locations must be a tuple")

        identities: set[tuple[object, ...]] = set()
        for location in locations:
            if type(location) is not FieldLocation:
                raise TypeError(
                    "locations must contain only FieldLocation values"
                )
            if location.association is not descriptor.association:
                raise ValueError(
                    "location association must match descriptor association"
                )
            identity = _field_location_identity_key(location)
            if identity in identities:
                raise ValueError("field locations must have unique identities")
            identities.add(identity)

        owned_values = _owned_finite_matrix(
            values,
            label="values",
            columns=len(descriptor.columns),
            rows=len(locations),
        )
        object.__setattr__(self, "descriptor", descriptor)
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "key", key)
        object.__setattr__(self, "locations", locations)
        object.__setattr__(self, "_values", owned_values)

    @property
    def values(self) -> np.ndarray:
        """Return a detached readonly copy of the canonical value matrix."""

        return _public_array_copy(self._values)


@dataclass(frozen=True, slots=True)
class ResultMaterializationPatch:
    """Atomic set of newly recovered fields for one result source."""

    source: ResultSourceKey
    fields: tuple[FieldData, ...]
    diagnostics: tuple[ResultDiagnostic, ...] = ()

    def __post_init__(self) -> None:
        if type(self.source) is not ResultSourceKey:
            raise TypeError("source must be ResultSourceKey")
        _validate_field_tuple(
            self.fields,
            source=self.source,
            label="patch fields",
        )
        _validate_diagnostic_tuple(self.diagnostics)


@dataclass(frozen=True, slots=True)
class ResultMaterializationSnapshot:
    """Accepted immutable field set for one monotonically increasing generation."""

    source: ResultSourceKey
    generation: int
    topology: ResultTopologyProjection
    fields: tuple[FieldData, ...]

    def __post_init__(self) -> None:
        if type(self.source) is not ResultSourceKey:
            raise TypeError("source must be ResultSourceKey")
        if type(self.generation) is not int:
            raise TypeError("generation must be an integer")
        if self.generation < 0:
            raise ValueError("generation must be non-negative")
        if type(self.topology) is not ResultTopologyProjection:
            raise TypeError("topology must be ResultTopologyProjection")
        if self.topology.source != self.source:
            raise ValueError("topology source must match snapshot source")
        _validate_field_tuple(
            self.fields,
            source=self.source,
            label="snapshot fields",
        )


@dataclass(frozen=True, slots=True, init=False)
class ResultExportSnapshot:
    """Atomically related topology, field, generation, and scalar selection."""

    source: ResultSourceKey
    materialization_generation: int
    topology: ResultTopologyProjection
    field: FieldData
    selection: ScalarFieldSelection

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        raise TypeError(
            "ResultExportSnapshot must be created by "
            "prepare_result_export_snapshot()"
        )

    @classmethod
    def _create(
        cls,
        *,
        source: ResultSourceKey,
        materialization_generation: int,
        topology: ResultTopologyProjection,
        field_data: FieldData,
        selection: ScalarFieldSelection,
    ) -> ResultExportSnapshot:
        instance = object.__new__(cls)
        object.__setattr__(instance, "source", source)
        object.__setattr__(
            instance,
            "materialization_generation",
            materialization_generation,
        )
        object.__setattr__(instance, "topology", topology)
        object.__setattr__(instance, "field", field_data)
        object.__setattr__(instance, "selection", selection)
        return instance


def build_initial_materialization(
    source: ResultSourceKey,
    topology: ResultTopologyProjection,
    base_fields: tuple[FieldData, ...],
    eager_patches: tuple[ResultMaterializationPatch, ...] = (),
) -> ResultMaterializationSnapshot:
    """Combine base fields and successful eager patches at generation zero."""

    if type(source) is not ResultSourceKey:
        raise TypeError("source must be ResultSourceKey")
    if type(topology) is not ResultTopologyProjection:
        raise TypeError("topology must be ResultTopologyProjection")
    if topology.source != source:
        raise ValueError("topology source must match materialization source")
    _validate_field_tuple(
        base_fields,
        source=source,
        label="base_fields",
        require_sorted=False,
    )
    if type(eager_patches) is not tuple:
        raise TypeError("eager_patches must be a tuple")

    combined = list(base_fields)
    for patch in eager_patches:
        if type(patch) is not ResultMaterializationPatch:
            raise TypeError(
                "eager_patches must contain ResultMaterializationPatch values"
            )
        if patch.source != source:
            raise ValueError("eager patch source must match materialization source")
        combined.extend(patch.fields)
    ordered = tuple(
        sorted(combined, key=lambda item: field_materialization_sort_key(item.key))
    )
    _validate_field_tuple(
        ordered,
        source=source,
        label="initial materialization fields",
    )
    return ResultMaterializationSnapshot(
        source=source,
        generation=0,
        topology=topology,
        fields=ordered,
    )


def advance_materialization(
    current_snapshot: ResultMaterializationSnapshot,
    patch: ResultMaterializationPatch,
) -> ResultMaterializationSnapshot:
    """Atomically add a non-empty disjoint patch and advance one generation."""

    if type(current_snapshot) is not ResultMaterializationSnapshot:
        raise TypeError(
            "current_snapshot must be ResultMaterializationSnapshot"
        )
    if type(patch) is not ResultMaterializationPatch:
        raise TypeError("patch must be ResultMaterializationPatch")
    if patch.source != current_snapshot.source:
        raise ValueError("patch source must match current snapshot source")
    if not patch.fields:
        raise ValueError("patch must add at least one field")

    existing_keys = {field_data.key for field_data in current_snapshot.fields}
    repeated = tuple(
        field_data.key
        for field_data in patch.fields
        if field_data.key in existing_keys
    )
    if repeated:
        raise ValueError("patch cannot replace an existing field key")

    combined = tuple(
        sorted(
            current_snapshot.fields + patch.fields,
            key=lambda item: field_materialization_sort_key(item.key),
        )
    )
    return ResultMaterializationSnapshot(
        source=current_snapshot.source,
        generation=current_snapshot.generation + 1,
        topology=current_snapshot.topology,
        fields=combined,
    )


def prepare_result_export_snapshot(
    materialization: ResultMaterializationSnapshot,
    selection: ScalarFieldSelection,
) -> ResultExportSnapshot:
    """Resolve one exact field key and bind it to its accepted generation."""

    if type(materialization) is not ResultMaterializationSnapshot:
        raise TypeError(
            "materialization must be ResultMaterializationSnapshot"
        )
    if type(selection) is not ScalarFieldSelection:
        raise TypeError("selection must be ScalarFieldSelection")
    matches = tuple(
        field_data
        for field_data in materialization.fields
        if field_data.key == selection.field_key
    )
    if len(matches) != 1:
        raise KeyError("selection field key is not materialized")
    field_data = matches[0]
    if field_data.source != materialization.source:
        raise ValueError("field source must match materialization source")
    if materialization.topology.source != materialization.source:
        raise ValueError("topology source must match materialization source")
    if field_data.key != selection.field_key:
        raise ValueError("selection field key must match field key")
    if selection.component not in field_data.descriptor.columns:
        raise ValueError("selection component is not in the field descriptor")
    return ResultExportSnapshot._create(
        source=materialization.source,
        materialization_generation=materialization.generation,
        topology=materialization.topology,
        field_data=field_data,
        selection=selection,
    )


def _strict_string_tuple(
    value: object,
    *,
    label: str,
) -> tuple[str, ...]:
    if type(value) is not tuple:
        raise TypeError(f"{label} must be a tuple")
    for item in value:
        _require_nonblank_string(item, label=f"{label} item")
    return value


def _positive_unique_id_tuple(
    value: object,
    *,
    label: str,
) -> tuple[int, ...]:
    if type(value) is not tuple:
        raise TypeError(f"{label} must be a tuple")
    seen: set[int] = set()
    for item in value:
        if type(item) is not int:
            raise TypeError(f"{label} must contain integers")
        if item <= 0:
            raise ValueError(f"{label} must contain positive IDs")
        if item in seen:
            raise ValueError(f"{label} must contain unique IDs")
        seen.add(item)
    return value


def _region_key_tuple(value: object) -> tuple[ResultRegionKey, ...]:
    if type(value) is not tuple:
        raise TypeError("element_region_keys must be a tuple")
    for region_key in value:
        if type(region_key) is not ResultRegionKey:
            raise TypeError(
                "element_region_keys must contain ResultRegionKey values"
            )
    return value


def _connectivity_tuple(
    value: object,
    *,
    node_ids: frozenset[int],
) -> tuple[tuple[int, ...], ...]:
    if type(value) is not tuple:
        raise TypeError("connectivity must be a tuple")
    for element_index, connected_ids in enumerate(value):
        if type(connected_ids) is not tuple:
            raise TypeError("connectivity rows must be tuples")
        if not connected_ids:
            raise ValueError("connectivity rows must not be empty")
        for node_id in connected_ids:
            if type(node_id) is not int:
                raise TypeError("connectivity node IDs must be integers")
            if node_id not in node_ids:
                raise ValueError(
                    "connectivity references unknown node ID "
                    f"{node_id} at element ordinal {element_index}"
                )
    return value


def _owned_finite_matrix(
    value: object,
    *,
    label: str,
    columns: int,
    rows: int,
) -> np.ndarray:
    if type(value) is not np.ndarray:
        raise TypeError(f"{label} must be a numpy.ndarray")
    if value.ndim != 2:
        raise ValueError(f"{label} must be a two-dimensional matrix")
    if value.shape != (rows, columns):
        raise ValueError(
            f"{label} shape must be ({rows}, {columns}), got {value.shape}"
        )
    if np.issubdtype(value.dtype, np.bool_):
        raise TypeError(f"{label} must not contain boolean values")
    if not np.issubdtype(value.dtype, np.number):
        raise TypeError(f"{label} must contain real numeric values")
    if np.issubdtype(value.dtype, np.complexfloating):
        raise TypeError(f"{label} must contain real numeric values")
    owned = np.array(value, dtype=float, order="C", copy=True)
    if not bool(np.isfinite(owned).all()):
        raise ValueError(f"{label} must contain only finite values")
    owned.setflags(write=False)
    return owned


def _public_array_copy(owner: np.ndarray) -> np.ndarray:
    result = np.array(owner, dtype=owner.dtype, order="C", copy=True)
    result.setflags(write=False)
    return result


def _finite_triplet(
    value: object,
    *,
    label: str,
) -> tuple[float, float, float]:
    if type(value) is not tuple:
        raise TypeError(f"{label} must be a tuple")
    if len(value) != 3:
        raise ValueError(f"{label} must contain exactly three values")
    converted: list[float] = []
    for item in value:
        if isinstance(item, (bool, np.bool_)) or not isinstance(item, Real):
            raise TypeError(f"{label} must contain real numeric values")
        numeric = float(item)
        if not math.isfinite(numeric):
            raise ValueError(f"{label} must contain only finite values")
        converted.append(numeric)
    return converted[0], converted[1], converted[2]


def _validate_location_identity(location: FieldLocation) -> None:
    association = location.association
    required: set[str]
    if association is FieldAssociation.NODE:
        required = {"node_id"}
    elif association is FieldAssociation.ELEMENT:
        required = {"element_id"}
    elif association is FieldAssociation.INTEGRATION_POINT:
        required = {"element_id", "integration_point"}
    elif association is FieldAssociation.ELEMENT_NODE:
        required = {"element_id", "local_node", "node_id"}
    elif association is FieldAssociation.NODE_REGION:
        required = {"node_id", "region_key"}
    elif association is FieldAssociation.RESOLVED_NODAL:
        required = {"node_id", "region_key", "averaged"}
        if location.averaged is False:
            required.update({"element_id", "local_node"})
    else:
        raise AssertionError(f"unhandled association {association!r}")

    identity_values = {
        "node_id": location.node_id,
        "element_id": location.element_id,
        "integration_point": location.integration_point,
        "local_node": location.local_node,
        "region_key": location.region_key,
        "averaged": location.averaged,
    }
    for name, value in identity_values.items():
        if name in required:
            if value is None:
                raise ValueError(
                    f"{association.value} locations require {name}"
                )
        elif value is not None:
            raise ValueError(
                f"{association.value} locations do not allow {name}"
            )

    for name in ("node_id", "element_id", "integration_point", "local_node"):
        value = identity_values[name]
        if value is None:
            continue
        if type(value) is not int:
            raise TypeError(f"{name} must be an integer or None")
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    if location.region_key is not None:
        if type(location.region_key) is not ResultRegionKey:
            raise TypeError("region_key must be ResultRegionKey or None")
    if location.averaged is not None and type(location.averaged) is not bool:
        raise TypeError("averaged must be a bool or None")


def _field_location_identity_key(
    location: FieldLocation,
) -> tuple[object, ...]:
    if location.association is FieldAssociation.NODE:
        return location.association, location.node_id
    if location.association is FieldAssociation.ELEMENT:
        return location.association, location.element_id
    if location.association is FieldAssociation.INTEGRATION_POINT:
        return (
            location.association,
            location.element_id,
            location.integration_point,
        )
    if location.association is FieldAssociation.ELEMENT_NODE:
        return (
            location.association,
            location.element_id,
            location.local_node,
            location.node_id,
        )
    if location.association is FieldAssociation.NODE_REGION:
        return location.association, location.node_id, location.region_key
    if location.averaged:
        return (
            location.association,
            location.node_id,
            location.region_key,
            True,
        )
    return (
        location.association,
        location.node_id,
        location.region_key,
        False,
        location.element_id,
        location.local_node,
    )


def _validate_field_tuple(
    value: object,
    *,
    source: ResultSourceKey,
    label: str,
    require_sorted: bool = True,
) -> None:
    if type(value) is not tuple:
        raise TypeError(f"{label} must be a tuple")
    for field_data in value:
        if type(field_data) is not FieldData:
            raise TypeError(f"{label} must contain only FieldData values")
        if field_data.source != source:
            raise ValueError(f"{label} source must match its owner source")
    keys = tuple(field_data.key for field_data in value)
    _validate_unique_sorted_keys(
        keys,
        label=label,
        require_sorted=require_sorted,
    )


def _validate_unique_sorted_keys(
    keys: tuple[FieldMaterializationKey, ...],
    *,
    label: str,
    require_sorted: bool = True,
) -> None:
    if len(set(keys)) != len(keys):
        raise ValueError(f"{label} must use unique materialization keys")
    if require_sorted and keys != tuple(
        sorted(keys, key=field_materialization_sort_key)
    ):
        raise ValueError(
            f"{label} must follow field_materialization_sort_key order"
        )


def _validate_diagnostic_tuple(value: object) -> None:
    if type(value) is not tuple:
        raise TypeError("diagnostics must be a tuple")
    for diagnostic in value:
        if type(diagnostic) is not ResultDiagnostic:
            raise TypeError(
                "diagnostics must contain only ResultDiagnostic values"
            )


def _freeze_json_mapping(
    value: Mapping[object, object],
    *,
    path: str,
    ancestors: set[int],
) -> _FrozenJsonObject:
    identity = id(value)
    if identity in ancestors:
        raise ValueError(f"{path} contains a cyclic JSON value")
    ancestors.add(identity)
    try:
        items: list[tuple[str, Any]] = []
        keys = tuple(value)
        for key in keys:
            if type(key) is not str:
                raise TypeError(f"{path} mapping keys must be strings")
        for key in sorted(keys):
            items.append(
                (
                    key,
                    _freeze_json_value(
                        value[key],
                        path=f"{path}.{key}",
                        ancestors=ancestors,
                    ),
                )
            )
        return _FrozenJsonObject(tuple(items))
    finally:
        ancestors.remove(identity)


def _freeze_json_value(
    value: object,
    *,
    path: str,
    ancestors: set[int],
) -> object:
    value_type = type(value)
    if value is None or value_type in {bool, int, str}:
        return value
    if value_type is float:
        if not math.isfinite(value):
            raise ValueError(f"{path} must contain only finite JSON numbers")
        return value
    if isinstance(value, Mapping):
        return _freeze_json_mapping(
            value,
            path=path,
            ancestors=ancestors,
        )
    if value_type in {list, tuple}:
        identity = id(value)
        if identity in ancestors:
            raise ValueError(f"{path} contains a cyclic JSON value")
        ancestors.add(identity)
        try:
            return tuple(
                _freeze_json_value(
                    item,
                    path=f"{path}[{index}]",
                    ancestors=ancestors,
                )
                for index, item in enumerate(value)
            )
        finally:
            ancestors.remove(identity)
    raise TypeError(
        f"{path} contains unsupported JSON value type {value_type.__name__}"
    )


def _require_nonblank_string(value: object, *, label: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{label} must be a string")
    if not value.strip():
        raise ValueError(f"{label} must not be blank")
    return value


__all__ = [
    "FieldAvailability",
    "FieldData",
    "FieldDescriptor",
    "FieldLocation",
    "FieldState",
    "ResultCatalog",
    "ResultDiagnostic",
    "ResultExportSnapshot",
    "ResultMaterializationPatch",
    "ResultMaterializationSnapshot",
    "ResultTopologyProjection",
    "advance_materialization",
    "build_initial_materialization",
    "prepare_result_export_snapshot",
]
