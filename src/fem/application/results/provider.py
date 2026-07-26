"""Immutable headless provider for primary and lazily derived result fields."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
import math
from numbers import Integral, Real
from typing import Any

import numpy as np

from fem.core.result import ModelResult
from fem.elements import get_element_capabilities
from fem.post.fields import result_region_key_for_element

from .data import (
    FieldAvailability,
    FieldData,
    FieldLocation,
    FieldState,
    ResultCatalog,
    ResultMaterializationPatch,
    ResultMaterializationSnapshot,
    ResultTopologyProjection,
    build_initial_materialization,
)
from .fields import (
    FieldAssociation,
    FieldMaterializationKey,
    FieldPosition,
    FieldRequest,
    ResultSourceKey,
    ResultVariable,
    ScalarFieldSelection,
    field_materialization_sort_key,
)
from .registry import (
    ElementResultProfile,
    FieldRecoveryKind,
    FieldRegistryEntry,
    ResultModelFamily,
    catalog_diagnostics,
    catalog_entries,
    classify_result_model,
    registry_entry_for,
)
from .query import (
    ResultQuery,
    ResultQueryResult,
    evaluate_result_query,
)
from ._materializers import (
    check_cancellation,
    materialize_derived_fields,
)
from ._ownership import deep_owned_materialization, deep_owned_result


@dataclass(frozen=True, slots=True, eq=False)
class ResultProvider:
    """One result-bound immutable provider without live Session dependencies."""

    _owned_result: ModelResult = field(repr=False, compare=False)
    _profile: ElementResultProfile
    _catalog: ResultCatalog
    _snapshot: ResultMaterializationSnapshot

    def __post_init__(self) -> None:
        if type(self._owned_result) is not ModelResult:
            raise TypeError("_owned_result must be ModelResult")
        if type(self._profile) is not ElementResultProfile:
            raise TypeError("_profile must be ElementResultProfile")
        if type(self._catalog) is not ResultCatalog:
            raise TypeError("_catalog must be ResultCatalog")
        if type(self._snapshot) is not ResultMaterializationSnapshot:
            raise TypeError(
                "_snapshot must be ResultMaterializationSnapshot"
            )
        if self._catalog.source != self._snapshot.source:
            raise ValueError("provider catalog and snapshot sources must match")
        ready_catalog_keys = {
            item.key
            for item in self._catalog.fields
            if item.state is FieldState.READY
        }
        snapshot_keys = {
            field_data.key for field_data in self._snapshot.fields
        }
        if ready_catalog_keys != snapshot_keys:
            raise ValueError(
                "provider READY catalog keys must equal snapshot field keys"
            )

    @property
    def source(self) -> ResultSourceKey:
        """Return the exact accepted-result identity."""

        return self._snapshot.source

    @property
    def profile(self) -> ElementResultProfile:
        """Return the immutable contextual element-result profile."""

        return self._profile

    @property
    def snapshot(self) -> ResultMaterializationSnapshot:
        """Return the current immutable materialization snapshot."""

        return self._snapshot

    def catalog(self) -> ResultCatalog:
        """Return contextual READY/LAZY field availability."""

        return self._catalog

    def resolve_request(
        self,
        request: FieldRequest,
    ) -> FieldMaterializationKey:
        """Resolve numerical intent to this registry's recovery contract."""

        if type(request) is not FieldRequest:
            raise TypeError("request must be FieldRequest")
        entry = _entry_for_request(self._profile, request)
        return FieldMaterializationKey(
            request=request,
            recovery_contract=entry.recovery_contract,
        )

    def field_status(
        self,
        key: FieldMaterializationKey,
    ) -> FieldAvailability:
        """Return status for one exact full key without field-ID fallback."""

        if type(key) is not FieldMaterializationKey:
            raise TypeError("key must be FieldMaterializationKey")
        for availability in self._catalog.fields:
            if availability.key == key:
                return availability
        entry = _entry_for_key(self._profile, key)
        return FieldAvailability(
            key=key,
            descriptor=entry.descriptor,
            state=FieldState.LAZY,
        )

    def field(self, key: FieldMaterializationKey) -> FieldData:
        """Return one exact ready field or fail without a placeholder."""

        availability = self.field_status(key)
        if availability.state is not FieldState.READY:
            raise KeyError(key)
        matches = tuple(
            field_data
            for field_data in self._snapshot.fields
            if field_data.key == key
        )
        if len(matches) != 1:
            raise RuntimeError(
                "catalog READY state does not match materialization snapshot"
            )
        return matches[0]

    def query(self, query: ResultQuery) -> ResultQueryResult:
        """Evaluate one exact scalar query over this immutable snapshot."""

        return evaluate_result_query(self._snapshot, query)

    def materialize(
        self,
        keys: Iterable[FieldMaterializationKey],
        *,
        cancellation: object | None = None,
    ) -> ResultMaterializationPatch:
        """Recover one atomic set; ready cache hits are omitted."""

        check_cancellation(cancellation)
        try:
            requested = tuple(keys)
        except TypeError as error:
            raise TypeError(
                "keys must be an iterable of FieldMaterializationKey values"
            ) from error
        if any(type(key) is not FieldMaterializationKey for key in requested):
            raise TypeError(
                "keys must contain only FieldMaterializationKey values"
            )
        unique = tuple(
            sorted(
                set(requested),
                key=field_materialization_sort_key,
            )
        )
        ready_keys = {
            field_data.key for field_data in self._snapshot.fields
        }
        targets = tuple(
            (key, _entry_for_key(self._profile, key))
            for key in unique
            if key not in ready_keys
        )
        if not targets:
            check_cancellation(cancellation)
            return ResultMaterializationPatch(
                source=self.source,
                fields=(),
            )
        if any(
            entry.recovery_kind is FieldRecoveryKind.PRIMARY
            for _key, entry in targets
        ):
            raise RuntimeError(
                "provider snapshot is missing an eager primary field"
            )
        fields = materialize_derived_fields(
            source=self.source,
            result=self._owned_result,
            topology=self._snapshot.topology,
            profile=self._profile,
            targets=targets,
            cancellation=cancellation,
        )
        return ResultMaterializationPatch(
            source=self.source,
            fields=fields,
        )

    def apply(
        self,
        patch: ResultMaterializationPatch,
    ) -> ResultProvider:
        """Return a same-generation worker draft with a validated field overlay."""

        if type(patch) is not ResultMaterializationPatch:
            raise TypeError("patch must be ResultMaterializationPatch")
        if patch.source != self.source:
            raise ValueError("patch source must match provider source")
        if not patch.fields:
            return self

        existing_keys = {
            field_data.key for field_data in self._snapshot.fields
        }
        checked: list[tuple[FieldData, FieldRegistryEntry]] = []
        for field_data in patch.fields:
            entry = _entry_for_key(self._profile, field_data.key)
            if field_data.key in existing_keys:
                raise ValueError("patch cannot replace a READY field")
            if field_data.descriptor != entry.descriptor:
                raise ValueError(
                    "patch descriptor must match the contextual registry"
                )
            checked.append((field_data, entry))

        combined = tuple(
            sorted(
                self._snapshot.fields
                + tuple(field_data for field_data, _entry in checked),
                key=lambda item: field_materialization_sort_key(item.key),
            )
        )
        draft_snapshot = ResultMaterializationSnapshot(
            source=self.source,
            generation=self._snapshot.generation,
            topology=self._snapshot.topology,
            fields=combined,
        )
        draft_catalog = _catalog_with_ready_patch(
            self._catalog,
            tuple(checked),
        )
        return ResultProvider(
            _owned_result=self._owned_result,
            _profile=self._profile,
            _catalog=draft_catalog,
            _snapshot=draft_snapshot,
        )


def build_result_provider(
    source: ResultSourceKey,
    result: ModelResult,
) -> ResultProvider:
    """Deep-own one solved result and build topology plus eager primary fields."""

    if type(source) is not ResultSourceKey:
        raise TypeError("source must be ResultSourceKey")
    if type(result) is not ModelResult:
        raise TypeError("result must be ModelResult")

    owned_result = deep_owned_result(result)
    profile = classify_result_model(owned_result.model)
    if not profile.primary_compatible:
        raise ValueError(
            "result model does not have one exact common primary DOF profile"
        )
    topology = _build_topology(source, owned_result, profile)
    entries = catalog_entries(profile)
    base_fields = tuple(
        _primary_field(source, owned_result, topology, profile, entry)
        for entry in entries
        if entry.recovery_kind is FieldRecoveryKind.PRIMARY
    )
    if not base_fields:
        raise ValueError("result profile does not publish any primary fields")
    snapshot = build_initial_materialization(
        source=source,
        topology=topology,
        base_fields=base_fields,
    )
    catalog = _build_catalog(source, profile, entries, snapshot)
    return ResultProvider(
        _owned_result=owned_result,
        _profile=profile,
        _catalog=catalog,
        _snapshot=snapshot,
    )


def restore_result_provider(
    result: ModelResult,
    materialization: ResultMaterializationSnapshot,
) -> ResultProvider:
    """Rebuild a provider from one accepted materialization snapshot."""

    if type(result) is not ModelResult:
        raise TypeError("result must be ModelResult")
    if type(materialization) is not ResultMaterializationSnapshot:
        raise TypeError(
            "materialization must be ResultMaterializationSnapshot"
        )

    owned_result = deep_owned_result(result)
    profile = classify_result_model(owned_result.model)
    if not profile.primary_compatible:
        raise ValueError(
            "result model does not have one exact common primary DOF profile"
        )

    snapshot = deep_owned_materialization(materialization)
    expected_topology = _build_topology(
        snapshot.source,
        owned_result,
        profile,
    )
    _require_exact_topology(snapshot.topology, expected_topology)

    entries = catalog_entries(profile)
    checked = _validate_restored_fields(
        result=owned_result,
        profile=profile,
        entries=entries,
        snapshot=snapshot,
        expected_topology=expected_topology,
    )
    catalog = _catalog_with_ready_patch(
        _build_catalog(snapshot.source, profile, entries, snapshot),
        checked,
    )
    return ResultProvider(
        _owned_result=owned_result,
        _profile=profile,
        _catalog=catalog,
        _snapshot=snapshot,
    )


def _require_exact_topology(
    actual: ResultTopologyProjection,
    expected: ResultTopologyProjection,
) -> None:
    matches = (
        actual.source == expected.source
        and actual.node_ids == expected.node_ids
        and np.array_equal(
            actual.node_coordinates,
            expected.node_coordinates,
        )
        and np.array_equal(
            actual.nodal_displacements,
            expected.nodal_displacements,
        )
        and actual.element_ids == expected.element_ids
        and actual.element_types == expected.element_types
        and actual.connectivity == expected.connectivity
        and actual.element_region_keys == expected.element_region_keys
    )
    if not matches:
        raise ValueError(
            "materialization topology does not exactly match the result model"
        )


def _validate_restored_fields(
    *,
    result: ModelResult,
    profile: ElementResultProfile,
    entries: tuple[FieldRegistryEntry, ...],
    snapshot: ResultMaterializationSnapshot,
    expected_topology: ResultTopologyProjection,
) -> tuple[tuple[FieldData, FieldRegistryEntry], ...]:
    checked: list[tuple[FieldData, FieldRegistryEntry]] = []
    for field_data in snapshot.fields:
        entry = _entry_for_request(profile, field_data.key.request)
        if field_data.descriptor != entry.descriptor:
            raise ValueError(
                "materialization field descriptor does not match the "
                "current registry"
            )
        checked.append((field_data, entry))

    expected_primary = {
        field_data.key: field_data
        for field_data in (
            _primary_field(
                snapshot.source,
                result,
                expected_topology,
                profile,
                entry,
            )
            for entry in entries
            if entry.recovery_kind is FieldRecoveryKind.PRIMARY
        )
    }
    actual_by_key = {
        field_data.key: field_data for field_data, _entry in checked
    }
    if not set(expected_primary).issubset(actual_by_key):
        raise ValueError(
            "materialization must contain every current eager primary field key"
        )
    for key, expected in expected_primary.items():
        actual = actual_by_key[key]
        if (
            actual.descriptor != expected.descriptor
            or actual.source != expected.source
            or actual.key != expected.key
            or actual.locations != expected.locations
            or not np.array_equal(actual.values, expected.values)
        ):
            raise ValueError(
                "materialization eager primary field does not exactly match "
                "the result model"
            )
    return tuple(checked)


def _build_topology(
    source: ResultSourceKey,
    result: ModelResult,
    profile: ElementResultProfile,
) -> ResultTopologyProjection:
    mesh = result.model.mesh
    try:
        raw_nodes = tuple(mesh.nodes)
        raw_node_ids = tuple(mesh.node_ids)
        raw_elements = tuple(mesh.elements)
    except AttributeError as error:
        raise TypeError(
            "result model mesh must expose nodes, node_ids, and elements"
        ) from error

    node_lookup: dict[int, Any] = {}
    for node in raw_nodes:
        try:
            node_id = _positive_id(node.id, label="node id")
        except AttributeError as error:
            raise TypeError("mesh nodes must expose id") from error
        if node_id in node_lookup:
            raise ValueError(f"mesh node id {node_id} is duplicated")
        node_lookup[node_id] = node
    node_ids = tuple(
        _positive_id(value, label="mesh.node_ids item")
        for value in raw_node_ids
    )
    if len(set(node_ids)) != len(node_ids):
        raise ValueError("mesh.node_ids must be unique")
    if set(node_ids) != set(node_lookup):
        raise ValueError(
            "mesh.node_ids must identify exactly the current mesh nodes"
        )

    coordinates = np.asarray(
        [_node_coordinates(node_lookup[node_id]) for node_id in node_ids],
        dtype=float,
    ).reshape((len(node_ids), 3))
    nodal_displacements = np.asarray(
        [
            _nodal_translation(
                mesh,
                result.U,
                node_id,
                profile.dof_labels,
            )
            for node_id in node_ids
        ],
        dtype=float,
    ).reshape((len(node_ids), 3))

    element_ids: list[int] = []
    element_types: list[str] = []
    connectivity: list[tuple[int, ...]] = []
    element_region_keys = []
    for element in raw_elements:
        try:
            element_id = _positive_id(element.id, label="element id")
            descriptor = get_element_capabilities(element.type)
            connected = tuple(
                _positive_id(value, label="element node id")
                for value in element.node_ids
            )
        except AttributeError as error:
            raise TypeError(
                "mesh elements must expose id, type, and node_ids"
            ) from error
        if element_id in element_ids:
            raise ValueError(f"mesh element id {element_id} is duplicated")
        if len(connected) != descriptor.node_count:
            raise ValueError(
                f"element {element_id} connectivity must contain "
                f"{descriptor.node_count} node IDs"
            )
        if len(set(connected)) != len(connected):
            raise ValueError(
                f"element {element_id} connectivity must not repeat node IDs"
            )
        missing = tuple(
            node_id for node_id in connected if node_id not in node_lookup
        )
        if missing:
            raise ValueError(
                f"element {element_id} references missing node {missing[0]}"
            )
        element_ids.append(element_id)
        element_types.append(descriptor.canonical_type)
        connectivity.append(connected)
        element_region_keys.append(result_region_key_for_element(element))

    return ResultTopologyProjection(
        source=source,
        node_ids=node_ids,
        node_coordinates=coordinates,
        nodal_displacements=nodal_displacements,
        element_ids=tuple(element_ids),
        element_types=tuple(element_types),
        connectivity=tuple(connectivity),
        element_region_keys=tuple(element_region_keys),
    )


def _primary_field(
    source: ResultSourceKey,
    result: ModelResult,
    topology: ResultTopologyProjection,
    profile: ElementResultProfile,
    entry: FieldRegistryEntry,
) -> FieldData:
    descriptor = entry.descriptor
    variable = descriptor.field_id.variable
    if entry.recovery_kind is not FieldRecoveryKind.PRIMARY:
        raise ValueError("primary field builder requires a primary registry entry")
    if variable in {ResultVariable.U, ResultVariable.UR}:
        vector = result.U
    elif variable in {ResultVariable.RF, ResultVariable.RM}:
        vector = result.reactions
    else:
        raise ValueError(f"{variable.value} is not a primary result variable")

    source_labels = tuple(
        _source_dof_label(variable, component)
        for component in descriptor.components
    )
    component_indices = tuple(
        _dof_label_index(profile.dof_labels, label)
        for label in source_labels
    )
    values = np.asarray(
        [
            [
                _vector_at_dof(
                    result.model.mesh,
                    vector,
                    node_id,
                    component_index,
                )
                for component_index in component_indices
            ]
            for node_id in topology.node_ids
        ],
        dtype=float,
    ).reshape((len(topology.node_ids), len(component_indices)))
    if descriptor.derived_components:
        if descriptor.derived_components != ("Magnitude",):
            raise ValueError(
                "primary fields only support the Magnitude derived component"
            )
        magnitude = np.linalg.norm(values, axis=1, keepdims=True)
        values = np.hstack((values, magnitude))

    coordinates = topology.node_coordinates
    displacements = topology.nodal_displacements
    locations = tuple(
        FieldLocation(
            association=FieldAssociation.NODE,
            coordinates=tuple(float(value) for value in coordinates[index]),
            displacement=tuple(
                float(value) for value in displacements[index]
            ),
            node_id=node_id,
        )
        for index, node_id in enumerate(topology.node_ids)
    )
    return FieldData(
        descriptor=descriptor,
        source=source,
        key=entry.default_key(),
        locations=locations,
        values=values,
    )


def _build_catalog(
    source: ResultSourceKey,
    profile: ElementResultProfile,
    entries: tuple[FieldRegistryEntry, ...],
    snapshot: ResultMaterializationSnapshot,
) -> ResultCatalog:
    ready_keys = {field_data.key for field_data in snapshot.fields}
    fields = tuple(
        sorted(
            (
                FieldAvailability(
                    key=entry.default_key(),
                    descriptor=entry.descriptor,
                    state=(
                        FieldState.READY
                        if entry.default_key() in ready_keys
                        else FieldState.LAZY
                    ),
                )
                for entry in entries
            ),
            key=lambda availability: field_materialization_sort_key(
                availability.key
            ),
        )
    )
    default = next(
        (
            availability
            for availability in fields
            if (
                availability.state is FieldState.READY
                and availability.descriptor.field_id.variable
                is ResultVariable.U
            )
        ),
        None,
    )
    if default is None:
        default = next(
            (
                availability
                for availability in fields
                if availability.state is FieldState.READY
            ),
            None,
        )
    if default is None:
        raise ValueError("result catalog requires at least one ready field")
    return ResultCatalog(
        source=source,
        fields=fields,
        default_selection=ScalarFieldSelection(
            field_key=default.key,
            component=default.descriptor.default_component,
        ),
        diagnostics=catalog_diagnostics(profile),
    )


def _catalog_with_ready_patch(
    catalog: ResultCatalog,
    checked: tuple[tuple[FieldData, FieldRegistryEntry], ...],
) -> ResultCatalog:
    by_key = {availability.key: availability for availability in catalog.fields}
    for field_data, entry in checked:
        by_key[field_data.key] = FieldAvailability(
            key=field_data.key,
            descriptor=entry.descriptor,
            state=FieldState.READY,
        )
    fields = tuple(
        sorted(
            by_key.values(),
            key=lambda availability: field_materialization_sort_key(
                availability.key
            ),
        )
    )
    return ResultCatalog(
        source=catalog.source,
        fields=fields,
        default_selection=catalog.default_selection,
        diagnostics=catalog.diagnostics,
    )


def _entry_for_request(
    profile: ElementResultProfile,
    request: FieldRequest,
) -> FieldRegistryEntry:
    try:
        entry = registry_entry_for(profile, request.field_id)
    except KeyError as error:
        raise KeyError(request.field_id) from error
    if (
        request.gauss_order is not None
        and entry.recovery_kind is not FieldRecoveryKind.CONTINUUM_STRESS
    ):
        raise ValueError(
            "gauss_order is unavailable for this contextual result family"
        )
    if request.gauss_order is not None:
        supported = _common_gauss_orders(
            profile,
            request.field_id.position,
        )
        if request.gauss_order not in supported:
            if supported:
                detail = ", ".join(str(value) for value in sorted(supported))
                raise ValueError(
                    f"gauss_order {request.gauss_order} is unavailable for "
                    f"this model and position; supported orders: {detail}"
                )
            raise ValueError(
                "explicit gauss_order is unavailable for this model and "
                "position"
            )
    return entry


def _entry_for_key(
    profile: ElementResultProfile,
    key: FieldMaterializationKey,
) -> FieldRegistryEntry:
    if type(key) is not FieldMaterializationKey:
        raise TypeError("key must be FieldMaterializationKey")
    entry = _entry_for_request(profile, key.request)
    if key.recovery_contract != entry.recovery_contract:
        raise KeyError(key)
    return entry


def _common_gauss_orders(
    profile: ElementResultProfile,
    position: FieldPosition,
) -> frozenset[int]:
    if profile.family not in {
        ResultModelFamily.PLANE_CONTINUUM,
        ResultModelFamily.SOLID_CONTINUUM,
    }:
        return frozenset()
    per_type = tuple(
        _gauss_orders_for_type(element_type, position)
        for element_type in profile.canonical_element_types
    )
    if not per_type:
        return frozenset()
    common = set(per_type[0])
    for orders in per_type[1:]:
        common.intersection_update(orders)
    return frozenset(common)


def _gauss_orders_for_type(
    element_type: str,
    position: FieldPosition,
) -> frozenset[int]:
    if element_type in {"Tri3", "Tet4", "Tet10"}:
        return frozenset()
    if element_type == "Tri6":
        return frozenset({3})
    if element_type == "Quad4":
        if position in {
            FieldPosition.INTEGRATION_POINT,
            FieldPosition.CENTROID,
        }:
            return frozenset({1, 2})
        return frozenset({2})
    if element_type == "Quad8":
        return frozenset({2, 3})
    if element_type == "Hex8":
        return frozenset({2})
    if element_type == "Hex20":
        return frozenset({3})
    raise ValueError(
        f"no contextual gauss-order contract for {element_type!r}"
    )


def _node_coordinates(node: Any) -> tuple[float, float, float]:
    try:
        raw = (node.x, node.y, getattr(node, "z", 0.0))
    except AttributeError as error:
        raise TypeError("mesh nodes must expose x and y coordinates") from error
    return tuple(
        _finite_number(value, label="node coordinate")
        for value in raw
    )


def _nodal_translation(
    mesh: Any,
    vector: np.ndarray,
    node_id: int,
    dof_labels: tuple[str, ...],
) -> tuple[float, float, float]:
    values = [0.0, 0.0, 0.0]
    for direction in range(1, 4):
        label = f"U{direction}"
        if label not in dof_labels:
            continue
        component_index = dof_labels.index(label)
        values[direction - 1] = _vector_at_dof(
            mesh,
            vector,
            node_id,
            component_index,
        )
    return values[0], values[1], values[2]


def _vector_at_dof(
    mesh: Any,
    vector: np.ndarray,
    node_id: int,
    component_index: int,
) -> float:
    try:
        raw_dof = mesh.global_dof(node_id, component_index)
    except (AttributeError, IndexError, KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"cannot map node {node_id} component {component_index}"
        ) from error
    if isinstance(raw_dof, bool) or not isinstance(raw_dof, Integral):
        raise TypeError("mesh.global_dof must return an integer")
    dof = int(raw_dof)
    if dof < 0 or dof >= vector.shape[0]:
        raise ValueError("mesh.global_dof returned an out-of-range index")
    return _finite_number(vector[dof], label="result vector value")


def _source_dof_label(
    variable: ResultVariable,
    component: str,
) -> str:
    if variable in {ResultVariable.U, ResultVariable.UR}:
        return component
    if variable is ResultVariable.RF and component.startswith("RF"):
        return f"U{component[2:]}"
    if variable is ResultVariable.RM and component.startswith("RM"):
        return f"UR{component[2:]}"
    raise ValueError(
        f"component {component!r} is invalid for {variable.value}"
    )


def _dof_label_index(
    dof_labels: tuple[str, ...],
    label: str,
) -> int:
    try:
        return dof_labels.index(label)
    except ValueError as error:
        raise ValueError(
            f"primary result DOF label {label!r} is unavailable"
        ) from error


def _positive_id(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{label} must be an integer")
    converted = int(value)
    if converted <= 0:
        raise ValueError(f"{label} must be positive")
    return converted


def _finite_number(value: Any, *, label: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{label} must be a real number")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{label} must be finite")
    return converted


__all__ = [
    "ResultProvider",
    "build_result_provider",
    "restore_result_provider",
]
