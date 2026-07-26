"""Internal adapters from canonical pure-post recovery to application fields."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
import math
from numbers import Real
from typing import Any

import numpy as np

from fem.core.result import ModelResult
from fem.post.averaging import ResolvedStressField, resolve_nodal_stress
from fem.post.fields import ResultRegionKey
from fem.post.stress import beam, truss
from fem.post.stress.field import (
    StressField,
    StressPosition,
    StressRecord,
    StressRecovery,
    natural_shape_values,
)

from .data import (
    FieldData,
    FieldDescriptor,
    FieldLocation,
    ResultTopologyProjection,
)
from .fields import (
    FieldAssociation,
    FieldMaterializationKey,
    FieldPosition,
    ResultSourceKey,
    field_materialization_sort_key,
)
from .registry import (
    ElementResultProfile,
    FieldRecoveryKind,
    FieldRegistryEntry,
    ResultModelFamily,
)


_CHECKPOINT_INTERVAL = 128


def check_cancellation(cancellation: object | None) -> None:
    """Run the project cancellation checkpoint without translating errors."""

    if cancellation is None:
        return
    checkpoint = getattr(cancellation, "checkpoint", None)
    if callable(checkpoint):
        checkpoint()
        return
    if callable(cancellation):
        cancellation()
        return
    raise TypeError(
        "cancellation must be callable, expose checkpoint(), or be None"
    )


def _checkpoint_for_cancellation(
    cancellation: object | None,
) -> Callable[[], None] | None:
    if cancellation is None:
        return None

    def checkpoint() -> None:
        check_cancellation(cancellation)

    return checkpoint


def materialize_derived_fields(
    *,
    source: ResultSourceKey,
    result: ModelResult,
    topology: ResultTopologyProjection,
    profile: ElementResultProfile,
    targets: tuple[tuple[FieldMaterializationKey, FieldRegistryEntry], ...],
    cancellation: object | None = None,
) -> tuple[FieldData, ...]:
    """Materialize one already-validated atomic batch in canonical key order."""

    if type(source) is not ResultSourceKey:
        raise TypeError("source must be ResultSourceKey")
    if type(result) is not ModelResult:
        raise TypeError("result must be ModelResult")
    if type(topology) is not ResultTopologyProjection:
        raise TypeError("topology must be ResultTopologyProjection")
    if topology.source != source:
        raise ValueError("topology source must match materialization source")
    if type(profile) is not ElementResultProfile:
        raise TypeError("profile must be ElementResultProfile")
    if type(targets) is not tuple:
        raise TypeError("targets must be a tuple")
    for target in targets:
        if (
            type(target) is not tuple
            or len(target) != 2
            or type(target[0]) is not FieldMaterializationKey
            or type(target[1]) is not FieldRegistryEntry
        ):
            raise TypeError(
                "targets must contain (FieldMaterializationKey, "
                "FieldRegistryEntry) tuples"
            )
        if target[0].request.field_id != target[1].descriptor.field_id:
            raise ValueError("target key and descriptor field IDs must match")
        if target[0].recovery_contract != target[1].recovery_contract:
            raise ValueError("target recovery contracts must match")

    check_cancellation(cancellation)
    if not targets:
        return ()
    lookup = _TopologyLookup(topology)
    if profile.family in {
        ResultModelFamily.PLANE_CONTINUUM,
        ResultModelFamily.SOLID_CONTINUUM,
    }:
        fields = _materialize_continuum(
            source,
            result,
            lookup,
            targets,
            cancellation,
        )
    elif profile.family is ResultModelFamily.TRUSS:
        fields = _materialize_truss(
            source,
            result,
            targets,
            cancellation,
        )
    elif profile.family is ResultModelFamily.BEAM:
        fields = _materialize_beam(
            source,
            result,
            targets,
            cancellation,
        )
    else:
        raise ValueError("mixed result models do not publish derived fields")
    check_cancellation(cancellation)
    ordered = tuple(
        sorted(fields, key=lambda field: field_materialization_sort_key(field.key))
    )
    expected = {key for key, _entry in targets}
    actual = {field.key for field in ordered}
    if len(ordered) != len(expected) or actual != expected:
        raise RuntimeError(
            "derived materializer did not return exactly the requested keys"
        )
    return ordered


def _materialize_continuum(
    source: ResultSourceKey,
    result: ModelResult,
    lookup: _TopologyLookup,
    targets: tuple[tuple[FieldMaterializationKey, FieldRegistryEntry], ...],
    cancellation: object | None,
) -> tuple[FieldData, ...]:
    if any(
        entry.recovery_kind is not FieldRecoveryKind.CONTINUUM_STRESS
        for _key, entry in targets
    ):
        raise ValueError(
            "continuum materialization requires continuum stress targets"
        )
    grouped: dict[
        int | None,
        list[tuple[FieldMaterializationKey, FieldRegistryEntry]],
    ] = {}
    for target in targets:
        grouped.setdefault(target[0].request.gauss_order, []).append(target)

    fields: list[FieldData] = []
    checkpoint = _checkpoint_for_cancellation(cancellation)
    for gauss_order in sorted(
        grouped,
        key=lambda value: (value is not None, 0 if value is None else value),
    ):
        check_cancellation(cancellation)
        recovery = StressRecovery(
            result.model.mesh,
            result.U,
            gauss_order=gauss_order,
            checkpoint=checkpoint,
        )
        check_cancellation(cancellation)
        for key, entry in grouped[gauss_order]:
            check_cancellation(cancellation)
            if key.request.field_id.position is FieldPosition.RESOLVED_NODAL:
                element_nodal = recovery.collect(
                    StressPosition.ELEMENT_NODAL,
                    checkpoint=checkpoint,
                )
                recovered: StressField | ResolvedStressField = (
                    resolve_nodal_stress(
                        element_nodal,
                        key.request.averaging_policy,
                        node_ids=result.model.mesh.node_ids,
                        element_ids=tuple(
                            int(element.id)
                            for element in result.model.mesh.elements
                        ),
                        checkpoint=checkpoint,
                    )
                )
            else:
                recovered = recovery.collect(
                    _stress_position(key.request.field_id.position),
                    checkpoint=checkpoint,
                )
            check_cancellation(cancellation)
            fields.append(
                _continuum_field_data(
                    source,
                    key,
                    entry.descriptor,
                    recovered,
                    lookup,
                    cancellation,
                )
            )
    return tuple(fields)


def _materialize_truss(
    source: ResultSourceKey,
    result: ModelResult,
    targets: tuple[tuple[FieldMaterializationKey, FieldRegistryEntry], ...],
    cancellation: object | None,
) -> tuple[FieldData, ...]:
    allowed = {
        FieldRecoveryKind.TRUSS_STRAIN,
        FieldRecoveryKind.TRUSS_STRESS,
    }
    if any(entry.recovery_kind not in allowed for _key, entry in targets):
        raise ValueError("Truss materialization received a non-Truss target")
    check_cancellation(cancellation)
    recovered = truss.recover(
        result.model.mesh,
        result.U,
        checkpoint=_checkpoint_for_cancellation(cancellation),
    )
    check_cancellation(cancellation)
    fields = []
    for key, entry in targets:
        fields.append(
            _simple_rows_field(
                source=source,
                key=key,
                descriptor=entry.descriptor,
                rows=recovered.rows,
                association=FieldAssociation.ELEMENT,
                cancellation=cancellation,
            )
        )
    return tuple(fields)


def _materialize_beam(
    source: ResultSourceKey,
    result: ModelResult,
    targets: tuple[tuple[FieldMaterializationKey, FieldRegistryEntry], ...],
    cancellation: object | None,
) -> tuple[FieldData, ...]:
    allowed = {
        FieldRecoveryKind.BEAM_SECTION_END,
        FieldRecoveryKind.BEAM_NODE_ENVELOPE,
    }
    if any(entry.recovery_kind not in allowed for _key, entry in targets):
        raise ValueError("Beam materialization received a non-Beam target")
    check_cancellation(cancellation)
    checkpoint = _checkpoint_for_cancellation(cancellation)
    section_end = beam.recover_section_end_stress(
        result,
        checkpoint=checkpoint,
    )
    check_cancellation(cancellation)
    envelope = None
    fields = []
    for key, entry in targets:
        if entry.recovery_kind is FieldRecoveryKind.BEAM_SECTION_END:
            fields.append(
                _simple_rows_field(
                    source=source,
                    key=key,
                    descriptor=entry.descriptor,
                    rows=section_end.rows,
                    association=FieldAssociation.ELEMENT_NODE,
                    cancellation=cancellation,
                )
            )
            continue
        if envelope is None:
            check_cancellation(cancellation)
            envelope = beam.section_node_envelope(
                section_end,
                checkpoint=checkpoint,
            )
            check_cancellation(cancellation)
        fields.append(
            _simple_rows_field(
                source=source,
                key=key,
                descriptor=entry.descriptor,
                rows=envelope.rows,
                association=FieldAssociation.NODE,
                cancellation=cancellation,
            )
        )
    return tuple(fields)


def _continuum_field_data(
    source: ResultSourceKey,
    key: FieldMaterializationKey,
    descriptor: FieldDescriptor,
    recovered: StressField | ResolvedStressField,
    topology: _TopologyLookup,
    cancellation: object | None,
) -> FieldData:
    if tuple(recovered.component_names) != descriptor.components:
        raise ValueError(
            "continuum recovery components do not match the descriptor"
        )
    locations = []
    values = []
    for index, record in enumerate(recovered.records):
        if index % _CHECKPOINT_INTERVAL == 0:
            check_cancellation(cancellation)
        if type(record) is not StressRecord:
            raise TypeError("continuum recovery rows must be StressRecord")
        topology.validate_region(record)
        locations.append(
            _continuum_location(
                descriptor.association,
                record,
                topology,
            )
        )
        row_values = record.values(recovered.component_names)
        values.append(
            tuple(float(row_values[column]) for column in descriptor.columns)
        )
    return FieldData(
        descriptor=descriptor,
        source=source,
        key=key,
        locations=tuple(locations),
        values=_value_matrix(values, len(descriptor.columns)),
    )


def _continuum_location(
    association: FieldAssociation,
    record: StressRecord,
    topology: _TopologyLookup,
) -> FieldLocation:
    coordinates = _triplet(record.coordinates, label="stress coordinates")
    displacement = topology.displacement_for(record)
    if association is FieldAssociation.INTEGRATION_POINT:
        return FieldLocation(
            association=association,
            coordinates=coordinates,
            displacement=displacement,
            element_id=record.elem_id,
            integration_point=record.integration_point,
        )
    if association is FieldAssociation.ELEMENT:
        return FieldLocation(
            association=association,
            coordinates=coordinates,
            displacement=displacement,
            element_id=record.elem_id,
        )
    if association is FieldAssociation.ELEMENT_NODE:
        return FieldLocation(
            association=association,
            coordinates=coordinates,
            displacement=displacement,
            element_id=record.elem_id,
            local_node=record.local_node,
            node_id=record.node_id,
        )
    if association is FieldAssociation.NODE_REGION:
        return FieldLocation(
            association=association,
            coordinates=coordinates,
            displacement=displacement,
            node_id=record.node_id,
            region_key=record.region_key,
        )
    if association is FieldAssociation.RESOLVED_NODAL:
        return FieldLocation(
            association=association,
            coordinates=coordinates,
            displacement=displacement,
            node_id=record.node_id,
            region_key=record.region_key,
            averaged=record.averaged,
            element_id=(record.elem_id if record.averaged is False else None),
            local_node=(
                record.local_node if record.averaged is False else None
            ),
        )
    raise ValueError(
        f"unsupported continuum field association {association.value}"
    )


def _simple_rows_field(
    *,
    source: ResultSourceKey,
    key: FieldMaterializationKey,
    descriptor: FieldDescriptor,
    rows: tuple[Any, ...],
    association: FieldAssociation,
    cancellation: object | None,
) -> FieldData:
    if descriptor.association is not association:
        raise ValueError("post row association does not match descriptor")
    locations = []
    values = []
    for index, row in enumerate(rows):
        if index % _CHECKPOINT_INTERVAL == 0:
            check_cancellation(cancellation)
        row_values = row.values()
        values.append(
            tuple(float(row_values[column]) for column in descriptor.columns)
        )
        if association is FieldAssociation.ELEMENT:
            locations.append(
                FieldLocation(
                    association=association,
                    coordinates=_triplet(
                        row.coordinates,
                        label="element coordinates",
                    ),
                    displacement=_triplet(
                        row.displacement,
                        label="element displacement",
                    ),
                    element_id=row.element_id,
                )
            )
        elif association is FieldAssociation.ELEMENT_NODE:
            locations.append(
                FieldLocation(
                    association=association,
                    coordinates=_triplet(
                        row.coordinates,
                        label="element-node coordinates",
                    ),
                    displacement=_triplet(
                        row.displacement,
                        label="element-node displacement",
                    ),
                    element_id=row.element_id,
                    local_node=row.local_node,
                    node_id=row.node_id,
                )
            )
        elif association is FieldAssociation.NODE:
            locations.append(
                FieldLocation(
                    association=association,
                    coordinates=_triplet(
                        row.coordinates,
                        label="node coordinates",
                    ),
                    displacement=_triplet(
                        row.displacement,
                        label="node displacement",
                    ),
                    node_id=row.node_id,
                )
            )
        else:
            raise ValueError(
                f"unsupported simple row association {association.value}"
            )
    return FieldData(
        descriptor=descriptor,
        source=source,
        key=key,
        locations=tuple(locations),
        values=_value_matrix(values, len(descriptor.columns)),
    )


def _stress_position(position: FieldPosition) -> StressPosition:
    mapping = {
        FieldPosition.INTEGRATION_POINT: StressPosition.INTEGRATION_POINT,
        FieldPosition.CENTROID: StressPosition.CENTROID,
        FieldPosition.ELEMENT_NODAL: StressPosition.ELEMENT_NODAL,
        FieldPosition.NODE_REGION: StressPosition.NODAL,
    }
    try:
        return mapping[position]
    except KeyError as error:
        raise ValueError(
            f"unsupported continuum stress position {position.value}"
        ) from error


@dataclass(frozen=True, slots=True)
class _TopologyLookup:
    topology: ResultTopologyProjection
    _node_order: dict[int, int] = field(init=False, repr=False)
    _element_order: dict[int, int] = field(init=False, repr=False)
    _nodal_displacements: np.ndarray = field(init=False, repr=False)
    _regions_by_node: dict[int, set[ResultRegionKey]] = field(
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        node_order = {
            node_id: index
            for index, node_id in enumerate(self.topology.node_ids)
        }
        element_order = {
            element_id: index
            for index, element_id in enumerate(self.topology.element_ids)
        }
        object.__setattr__(self, "_node_order", node_order)
        object.__setattr__(self, "_element_order", element_order)
        object.__setattr__(
            self,
            "_nodal_displacements",
            self.topology.nodal_displacements,
        )
        regions_by_node: dict[int, set[ResultRegionKey]] = {}
        for index, connected in enumerate(self.topology.connectivity):
            region_key = self.topology.element_region_keys[index]
            for node_id in connected:
                regions_by_node.setdefault(node_id, set()).add(region_key)
        object.__setattr__(self, "_regions_by_node", regions_by_node)

    def validate_region(self, record: StressRecord) -> None:
        if type(record.region_key) is not ResultRegionKey:
            raise TypeError(
                "continuum stress records require ResultRegionKey"
            )
        if record.elem_id is not None:
            try:
                element_index = self._element_order[record.elem_id]
            except KeyError as error:
                raise ValueError(
                    f"stress row references unknown element {record.elem_id}"
                ) from error
            expected = self.topology.element_region_keys[element_index]
            if record.region_key != expected:
                raise ValueError(
                    "stress row region does not match topology element region"
                )
            return
        if record.node_id is None:
            raise ValueError(
                "region-only stress rows require a node or element identity"
            )
        if record.region_key not in self._regions_by_node.get(
            record.node_id,
            set(),
        ):
            raise ValueError(
                "stress row region is not incident to its topology node"
            )

    def displacement_for(
        self,
        record: StressRecord,
    ) -> tuple[float, float, float]:
        if record.displacement is not None:
            return _triplet(
                record.displacement,
                label="stress displacement",
            )
        if record.node_id is not None:
            try:
                index = self._node_order[record.node_id]
            except KeyError as error:
                raise ValueError(
                    f"stress row references unknown node {record.node_id}"
                ) from error
            return tuple(
                float(value) for value in self._nodal_displacements[index]
            )
        if record.elem_id is None or record.natural_coordinates is None:
            raise ValueError(
                "stress sample cannot derive its translational displacement"
            )
        try:
            element_index = self._element_order[record.elem_id]
        except KeyError as error:
            raise ValueError(
                f"stress row references unknown element {record.elem_id}"
            ) from error
        connected = self.topology.connectivity[element_index]
        type_key = self.topology.element_types[element_index].casefold()
        shape_values = np.asarray(
            natural_shape_values(type_key, record.natural_coordinates),
            dtype=float,
        )
        if shape_values.shape != (len(connected),):
            raise ValueError(
                "stress sample shape values do not match element connectivity"
            )
        displacements = np.asarray(
            [
                self._nodal_displacements[self._node_order[node_id]]
                for node_id in connected
            ],
            dtype=float,
        )
        interpolated = shape_values @ displacements
        return tuple(float(value) for value in interpolated)


def _triplet(
    values: object,
    *,
    label: str,
) -> tuple[float, float, float]:
    try:
        raw = tuple(values)  # type: ignore[arg-type]
    except TypeError as error:
        raise TypeError(f"{label} must contain two or three numbers") from error
    if len(raw) == 2:
        raw = (*raw, 0.0)
    if len(raw) != 3:
        raise ValueError(f"{label} must contain two or three values")
    result = []
    for value in raw:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
            raise TypeError(f"{label} must contain real numbers")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"{label} must contain finite numbers")
        result.append(number)
    return result[0], result[1], result[2]


def _value_matrix(
    rows: list[tuple[float, ...]],
    columns: int,
) -> np.ndarray:
    return np.asarray(rows, dtype=float).reshape((len(rows), columns))


__all__ = [
    "check_cancellation",
    "materialize_derived_fields",
]
