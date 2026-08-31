"""Internal adapters from canonical pure-post recovery to application fields."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
import math
from numbers import Real
from typing import Any

import numpy as np

from fem.results import ModelResult
from fem.post.averaging import (
    NodalAveragingPolicy,
    ResolvedStressField,
    resolve_nodal_stress,
)
from fem.post.fields import ResultRegionKey, result_region_sort_key
from fem.post.stress import beam, dispatch, truss
from fem.post.stress._common import element_volume, node_lookup
from fem.post.stress.field import (
    CANONICAL_PLANE_COMPONENT_NAMES,
    CANONICAL_SOLID_COMPONENT_NAMES,
    CapturedStressRecovery,
    FiniteStrainStateField,
    FiniteStrainStatePosition,
    FiniteStrainStateRecord,
    FiniteStrainStateRecovery,
    StressField,
    StressPosition,
    StressRecord,
    StressRecovery,
    natural_shape_values,
)
from fem.materials import linear_elastic
from fem.physics.mechanics.operators.continuum_quad4 import (
    _quad4_centroid_recovery_matrix,
    _quad4_extrapolation_matrix,
    quad4_gauss_points,
    quad4_shape_grad_xi_eta,
    quad4_shape_functions,
)
from fem.physics.mechanics.operators.plane import PlaneProperties

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
    FieldRequest,
    ResultFieldId,
    ResultSourceKey,
    ResultVariable,
    field_materialization_sort_key,
)
from .registry import (
    ElementResultProfile,
    FieldRecoveryKind,
    FieldRegistryEntry,
    ResultModelFamily,
)


_CHECKPOINT_INTERVAL = 128

# ``NODE_REGION`` has exactly one field-location identity per
# (node, result-region) pair.  Unlike ``RESOLVED_NODAL``, it cannot retain
# element-local contributions when the nodal values vary, so its state-field
# projection must always reduce a region's contributions to one value.
_NODE_REGION_AVERAGING_POLICY = NodalAveragingPolicy(
    threshold_percent=100.0,
)


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
    existing_fields: tuple[FieldData, ...] = (),
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
    if profile.family in {
        ResultModelFamily.PLANE_CONTINUUM,
        ResultModelFamily.SOLID_CONTINUUM,
    }:
        lookup = _TopologyLookup(topology)
        fields = _materialize_continuum(
            source,
            result,
            lookup,
            targets,
            existing_fields,
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
    existing_fields: tuple[FieldData, ...],
    cancellation: object | None,
) -> tuple[FieldData, ...]:
    allowed = {
        FieldRecoveryKind.CONTINUUM_STRESS,
        FieldRecoveryKind.FINITE_STRAIN_TENSOR,
        FieldRecoveryKind.FINITE_STRAIN_SCALAR,
        FieldRecoveryKind.MATERIAL_STATE_SCALAR,
    }
    if any(entry.recovery_kind not in allowed for _key, entry in targets):
        raise ValueError(
            "continuum materialization received an unsupported target"
        )
    grouped: dict[
        int | None,
        list[tuple[FieldMaterializationKey, FieldRegistryEntry]],
    ] = {}
    for target in targets:
        grouped.setdefault(target[0].request.gauss_order, []).append(target)

    fields: list[FieldData] = []
    checkpoint = _checkpoint_for_cancellation(cancellation)
    recovery_model = (
        result.compiled_model
        if result.compiled_model is not None
        else result.model
    )
    recovery_mesh = recovery_model.mesh
    for gauss_order in sorted(
        grouped,
        key=lambda value: (value is not None, 0 if value is None else value),
    ):
        check_cancellation(cancellation)
        recovery: StressRecovery | CapturedStressRecovery | None = None
        state_recoveries: dict[str, FiniteStrainStateRecovery] = {}
        for key, entry in grouped[gauss_order]:
            check_cancellation(cancellation)
            if entry.recovery_kind is FieldRecoveryKind.CONTINUUM_STRESS:
                if (
                    key.request.field_id.variable is ResultVariable.S
                    and key.request.field_id.position
                    is FieldPosition.RESOLVED_NODAL
                    and not _has_captured_continuum_stress_outputs(result)
                ):
                    element_nodal = _existing_element_nodal_stress_field(
                        source,
                        key,
                        existing_fields + tuple(fields),
                    )
                    if element_nodal is not None:
                        fast_field = (
                            _materialize_resolved_nodal_stress_from_field(
                                source,
                                key,
                                entry.descriptor,
                                element_nodal,
                                recovery_mesh,
                                lookup,
                                cancellation,
                            )
                        )
                        if fast_field is not None:
                            fields.append(fast_field)
                            continue
                if (
                    key.request.field_id.variable is ResultVariable.S
                    and key.request.field_id.position
                    is FieldPosition.RESOLVED_NODAL
                    and not _has_captured_continuum_stress_outputs(result)
                ):
                    fast_field = (
                        _materialize_linear_plane_resolved_nodal_stress_from_mesh(
                            source,
                            key,
                            entry.descriptor,
                            recovery_mesh,
                            lookup,
                            cancellation,
                        )
                    )
                    if fast_field is not None:
                        fields.append(fast_field)
                        continue
                if (
                    key.request.field_id.position is not FieldPosition.RESOLVED_NODAL
                    and not _has_captured_continuum_stress_outputs(result)
                ):
                    fast_field = _materialize_linear_plane_stress_from_mesh(
                        source,
                        key,
                        entry.descriptor,
                        recovery_mesh,
                        lookup,
                        cancellation,
                    )
                    if fast_field is not None:
                        fields.append(fast_field)
                        continue
                if (
                    key.request.field_id.position is not FieldPosition.RESOLVED_NODAL
                    and not _has_captured_continuum_stress_outputs(result)
                ):
                    fast_field = _materialize_quad4_linear_stress_from_mesh(
                        source,
                        key,
                        entry.descriptor,
                        recovery_mesh,
                        lookup,
                        cancellation,
                    )
                    if fast_field is not None:
                        fields.append(fast_field)
                        continue
                if recovery is None:
                    if _has_captured_continuum_stress_outputs(result):
                        recovery = CapturedStressRecovery(
                            result,
                            gauss_order=gauss_order,
                            checkpoint=checkpoint,
                        )
                    else:
                        recovery = StressRecovery(
                            recovery_mesh,
                            result.U,
                            gauss_order=gauss_order,
                            checkpoint=checkpoint,
                        )
                if key.request.field_id.position is FieldPosition.RESOLVED_NODAL:
                    fast_field = _materialize_quad4_captured_resolved_nodal_stress_field(
                        source,
                        key,
                        entry.descriptor,
                        recovery,
                        lookup,
                        cancellation,
                    )
                    if fast_field is not None:
                        fields.append(fast_field)
                        continue
                    element_nodal = recovery.collect(
                        StressPosition.ELEMENT_NODAL,
                        checkpoint=checkpoint,
                    )
                    recovered: StressField | ResolvedStressField = (
                        resolve_nodal_stress(
                            element_nodal,
                            key.request.averaging_policy,
                            node_ids=recovery_mesh.node_ids,
                            element_ids=tuple(
                                int(element.id)
                                for element in recovery_mesh.elements
                            ),
                            checkpoint=checkpoint,
                        )
                    )
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
                    continue
                else:
                    fast_field = _materialize_quad4_linear_stress_field(
                        source,
                        key,
                        entry.descriptor,
                        recovery,
                        lookup,
                        cancellation,
                    )
                    if fast_field is not None:
                        fields.append(fast_field)
                        continue
                    recovered = recovery.collect(
                        _stress_position(key.request.field_id.position),
                        checkpoint=checkpoint,
                    )
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
                continue

            state_variable = (
                "green_lagrange_strain"
                if entry.recovery_kind is FieldRecoveryKind.FINITE_STRAIN_TENSOR
                else "equivalent_plastic_strain"
            )
            state_recovery = state_recoveries.get(state_variable)
            if state_recovery is None:
                state_recovery = FiniteStrainStateRecovery(
                    result,
                    state_variable,
                    gauss_order=gauss_order,
                    checkpoint=checkpoint,
                )
                state_recoveries[state_variable] = state_recovery
            position = key.request.field_id.position
            if position is FieldPosition.RESOLVED_NODAL:
                recovered_state = state_recovery.collect(
                    FiniteStrainStatePosition.NODAL,
                    averaging_policy=key.request.averaging_policy,
                    checkpoint=checkpoint,
                )
            elif position is FieldPosition.NODE_REGION:
                recovered_state = state_recovery.collect(
                    FiniteStrainStatePosition.NODAL,
                    averaging_policy=_NODE_REGION_AVERAGING_POLICY,
                    checkpoint=checkpoint,
                )
            else:
                recovered_state = state_recovery.collect(
                    _state_position(position),
                    checkpoint=checkpoint,
                )
            fields.append(
                _finite_strain_state_field_data(
                    source,
                    key,
                    entry.descriptor,
                    recovered_state,
                    lookup,
                    cancellation,
                )
            )
    return tuple(fields)


def _materialize_linear_plane_resolved_nodal_stress_from_mesh(
    source: ResultSourceKey,
    key: FieldMaterializationKey,
    descriptor: FieldDescriptor,
    mesh: Any,
    topology: _TopologyLookup,
    cancellation: object | None,
) -> FieldData | None:
    """Batch the common uncaptured Tri3/Quad4 resolved-stress path.

    A direct request for resolved nodal stress used to build a complete
    ``StressRecovery`` object, then allocate one ``StressRecord`` and one
    invariant object per integration/node contribution before reducing them.
    For the linear plane contract the element formulas are independent and
    can be evaluated in two NumPy blocks.  The final typed ``FieldLocation``
    objects are still retained, so this changes traversal/allocation strategy
    only; it does not change the public field schema or averaging rules.
    """

    field_id = key.request.field_id
    if field_id.variable is not ResultVariable.S:
        return None
    if field_id.position is not FieldPosition.RESOLVED_NODAL:
        return None
    if descriptor.association is not FieldAssociation.RESOLVED_NODAL:
        return None
    if descriptor.components != CANONICAL_PLANE_COMPONENT_NAMES:
        return None
    if key.request.gauss_order not in (None, 2):
        return None
    if not isinstance(key.request.averaging_policy, NodalAveragingPolicy):
        return None

    elements = tuple(getattr(mesh, "elements", ()))
    element_ids = tuple(int(value) for value in topology.topology.element_ids)
    if (
        not elements
        or tuple(int(element.id) for element in elements) != element_ids
    ):
        return None
    type_keys = tuple(dispatch.type_key_from_name(element.type) for element in elements)
    if any(type_key not in {"tri3", "quad4"} for type_key in type_keys):
        return None
    if key.request.gauss_order == 2 and "tri3" in type_keys:
        # Tri3 has one fixed centroid point and rejects an explicit order.
        return None
    check_cancellation(cancellation)

    element_count = len(elements)
    element_values: dict[str, np.ndarray] = {}
    element_coordinates: dict[str, np.ndarray] = {}
    element_displacements: dict[str, np.ndarray] = {}
    element_positions: dict[str, np.ndarray] = {}
    for type_key in ("tri3", "quad4"):
        positions = np.asarray(
            [index for index, candidate in enumerate(type_keys) if candidate == type_key],
            dtype=np.int64,
        )
        if positions.size == 0:
            continue
        connectivity = np.asarray(
            [topology._connectivity_indices[int(index)] for index in positions],
            dtype=np.int64,
        )
        reference = topology.topology._node_coordinates[connectivity]
        displacement = topology._nodal_displacements[connectivity]
        if type_key == "tri3":
            values = _linear_tri3_stress_components(
                elements,
                positions,
                reference,
                displacement,
            )
        else:
            values = _linear_quad4_stress_components(
                elements,
                positions,
                reference,
                displacement,
            )
        if not np.all(np.isfinite(values)):
            raise ValueError("linear plane stress components must be finite")
        element_values[type_key] = values
        element_coordinates[type_key] = reference
        element_displacements[type_key] = displacement
        element_positions[type_key] = positions

    node_counts = np.asarray(
        [3 if type_key == "tri3" else 4 for type_key in type_keys],
        dtype=np.int64,
    )
    row_offsets = np.concatenate(
        (np.asarray((0,), dtype=np.int64), np.cumsum(node_counts[:-1]))
    )
    row_count = int(np.sum(node_counts))
    values_components = np.empty((row_count, 4), dtype=float)
    coordinates = np.empty((row_count, 3), dtype=float)
    displacements = np.empty((row_count, 3), dtype=float)
    node_ids = np.empty(row_count, dtype=np.int64)
    row_element_ids = np.empty(row_count, dtype=np.int64)
    local_nodes = np.empty(row_count, dtype=np.int64)
    row_regions: list[ResultRegionKey] = [
        topology.topology.element_region_keys[0]
    ] * row_count
    for type_key in ("tri3", "quad4"):
        positions = element_positions.get(type_key)
        if positions is None:
            continue
        local_count = 3 if type_key == "tri3" else 4
        row_indices = (
            row_offsets[positions, None]
            + np.arange(local_count, dtype=np.int64)[None, :]
        ).reshape(-1)
        values = element_values[type_key]
        coords = element_coordinates[type_key]
        element_u = element_displacements[type_key]
        connectivity = np.asarray(
            [topology._connectivity_indices[int(index)] for index in positions],
            dtype=np.int64,
        )
        values_components[row_indices] = values.reshape(-1, 4)
        coordinates[row_indices] = coords.reshape(-1, 3)
        displacements[row_indices] = element_u.reshape(-1, 3)
        node_ids[row_indices] = np.asarray(
            [
                topology.topology.node_ids[int(node_index)]
                for node_index in connectivity.reshape(-1)
            ],
            dtype=np.int64,
        )
        row_element_ids[row_indices] = np.repeat(
            np.asarray(element_ids, dtype=np.int64)[positions],
            local_count,
        )
        local_nodes[row_indices] = np.tile(
            np.arange(1, local_count + 1, dtype=np.int64),
            len(positions),
        )
        for position in positions:
            start = int(row_offsets[position])
            stop = start + local_count
            row_regions[start:stop] = (
                topology.topology.element_region_keys[int(position)],
            ) * local_count

    element_nodal_field_id = ResultFieldId(
        ResultVariable.S,
        FieldPosition.ELEMENT_NODAL,
        section_point_number=field_id.section_point_number,
    )
    element_nodal_descriptor = replace(
        descriptor,
        field_id=element_nodal_field_id,
        association=FieldAssociation.ELEMENT_NODE,
    )
    element_nodal_key = FieldMaterializationKey(
        FieldRequest(
            element_nodal_field_id,
            gauss_order=key.request.gauss_order,
        ),
        recovery_contract=key.recovery_contract,
    )
    element_nodal_field = _quad4_resolved_nodal_field_data(
        source=source,
        key=element_nodal_key,
        descriptor=element_nodal_descriptor,
        values_components=values_components,
        coordinates=coordinates,
        displacements=displacements,
        node_ids=node_ids,
        element_ids=[int(value) for value in row_element_ids],
        local_nodes=[int(value) for value in local_nodes],
        region_keys=tuple(row_regions),
        averaged=np.zeros(row_count, dtype=bool),
        cancellation=cancellation,
    )
    return _materialize_resolved_nodal_stress_from_field(
        source,
        key,
        descriptor,
        element_nodal_field,
        mesh,
        topology,
        cancellation,
    )


def _materialize_linear_plane_stress_from_mesh(
    source: ResultSourceKey,
    key: FieldMaterializationKey,
    descriptor: FieldDescriptor,
    mesh: Any,
    topology: _TopologyLookup,
    cancellation: object | None,
) -> FieldData | None:
    """Batch the common uncaptured linear Tri3/Quad4 stress positions.

    The general recovery service is deliberately retained for captured,
    nonlinear, solid, and high-order results.  Imported linear plane models
    are a frequent GUI path, however, and the old fallback evaluated every
    element through a Python kernel before allocating a record for every row.
    This adapter evaluates each element family in one NumPy block and only
    creates the final typed locations required by the result contract.
    """

    field_id = key.request.field_id
    position = field_id.position
    if field_id.variable is not ResultVariable.S:
        return None
    if position not in {
        FieldPosition.INTEGRATION_POINT,
        FieldPosition.CENTROID,
        FieldPosition.ELEMENT_NODAL,
    }:
        return None
    if descriptor.components != CANONICAL_PLANE_COMPONENT_NAMES:
        return None
    expected_association = {
        FieldPosition.INTEGRATION_POINT: FieldAssociation.INTEGRATION_POINT,
        FieldPosition.CENTROID: FieldAssociation.ELEMENT,
        FieldPosition.ELEMENT_NODAL: FieldAssociation.ELEMENT_NODE,
    }[position]
    if descriptor.association is not expected_association:
        return None
    if key.request.gauss_order not in (None, 2):
        return None

    elements = tuple(getattr(mesh, "elements", ()))
    element_ids = tuple(int(value) for value in topology.topology.element_ids)
    if (
        not elements
        or tuple(int(element.id) for element in elements) != element_ids
    ):
        return None
    type_keys = tuple(
        dispatch.type_key_from_name(element.type) for element in elements
    )
    if any(type_key not in {"tri3", "quad4"} for type_key in type_keys):
        return None
    # Keep the existing homogeneous Quad4 fast path and its established
    # location semantics untouched. This adapter is specifically for the
    # mixed Tri3/Quad4 case that otherwise falls all the way back to Python
    # per-element recovery.
    if len(set(type_keys)) != 2:
        return None
    # Tri3 has a fixed one-point rule and must reject an explicit order of 2,
    # matching the canonical recovery service's public behavior.
    if key.request.gauss_order == 2 and "tri3" in type_keys:
        return None
    check_cancellation(cancellation)

    blocks: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for type_key in ("tri3", "quad4"):
        positions = np.asarray(
            [
                index
                for index, candidate in enumerate(type_keys)
                if candidate == type_key
            ],
            dtype=np.int64,
        )
        if positions.size == 0:
            continue
        connectivity = np.stack(
            [topology._connectivity_indices[int(index)] for index in positions],
            axis=0,
        )
        reference = np.asarray(
            topology.topology._node_coordinates[connectivity],
            dtype=float,
        )
        displacement = np.asarray(
            topology._nodal_displacements[connectivity],
            dtype=float,
        )
        if type_key == "tri3":
            values = _linear_tri3_stress_components(
                elements,
                positions,
                reference,
                displacement,
            )
        else:
            values = _linear_quad4_integration_stress_components(
                elements,
                positions,
                reference,
                displacement,
            )
        if not np.all(np.isfinite(values)):
            raise ValueError("linear plane stress components must be finite")
        blocks[type_key] = (reference, displacement, values)

    row_counts = np.asarray(
        [
            (
                1
                if position is FieldPosition.INTEGRATION_POINT
                else 3
            )
            if type_key == "tri3"
            else 4
            for type_key in type_keys
        ],
        dtype=np.int64,
    )
    if position is FieldPosition.CENTROID:
        row_count = len(elements)
        values_components = np.empty((row_count, 4), dtype=float)
        coordinates = np.empty((row_count, 3), dtype=float)
        displacements = np.empty((row_count, 3), dtype=float)
        for type_key in ("tri3", "quad4"):
            positions = np.asarray(
                [
                    index
                    for index, candidate in enumerate(type_keys)
                    if candidate == type_key
                ],
                dtype=np.int64,
            )
            if positions.size == 0:
                continue
            reference, displacement, values = blocks[type_key]
            if type_key == "tri3":
                centroid_values = values[:, 0, :]
                shape = np.asarray(
                    [[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]],
                    dtype=float,
                )
            else:
                centroid_values = np.einsum(
                    "cq,eqs->ecs",
                    _quad4_centroid_recovery_matrix(2),
                    values,
                )[:, 0, :]
                shape = np.asarray(
                    [[0.25, 0.25, 0.25, 0.25]],
                    dtype=float,
                )
            coordinates[positions] = np.einsum(
                "qa,eak->eqk",
                shape,
                reference,
            )[:, 0, :]
            displacements[positions] = np.einsum(
                "qa,eak->eqk",
                shape,
                displacement,
            )[:, 0, :]
            values_components[positions] = centroid_values
        return _linear_plane_field_data_from_arrays(
            source=source,
            key=key,
            descriptor=descriptor,
            values_components=values_components,
            coordinates=coordinates,
            displacements=displacements,
            element_ids=np.asarray(element_ids, dtype=np.int64),
            integration_points=None,
            node_ids=None,
            local_nodes=None,
            region_keys=tuple(topology.topology.element_region_keys),
            cancellation=cancellation,
        )

    row_offsets = np.concatenate(
        (np.asarray((0,), dtype=np.int64), np.cumsum(row_counts[:-1]))
    )
    row_count = int(np.sum(row_counts))
    values_components = np.empty((row_count, 4), dtype=float)
    coordinates = np.empty((row_count, 3), dtype=float)
    displacements = np.empty((row_count, 3), dtype=float)
    row_element_ids = np.repeat(
        np.asarray(element_ids, dtype=np.int64),
        row_counts,
    )
    row_regions = tuple(
        region
        for region, count in zip(
            topology.topology.element_region_keys,
            row_counts,
        )
        for _ in range(int(count))
    )
    integration_points: np.ndarray | None = None
    node_ids: np.ndarray | None = None
    local_nodes: np.ndarray | None = None
    if position is FieldPosition.INTEGRATION_POINT:
        integration_points = np.concatenate(
            [
                np.arange(1, int(count) + 1, dtype=np.int64)
                for count in row_counts
            ]
        )
    else:
        local_nodes = np.concatenate(
            [
                np.arange(1, int(count) + 1, dtype=np.int64)
                for count in row_counts
            ]
        )
        node_ids = np.concatenate(
            [
                np.asarray(
                    [
                        topology.topology.node_ids[int(node_index)]
                        for node_index in topology._connectivity_indices[index]
                    ],
                    dtype=np.int64,
                )
                for index in range(len(elements))
            ]
        )

    for type_key in ("tri3", "quad4"):
        positions = np.asarray(
            [
                index
                for index, candidate in enumerate(type_keys)
                if candidate == type_key
            ],
            dtype=np.int64,
        )
        if positions.size == 0:
            continue
        reference, displacement, values = blocks[type_key]
        if type_key == "tri3":
            natural = np.asarray(
                [[1.0 / 3.0, 1.0 / 3.0]],
                dtype=float,
            )
            shape = (
                np.asarray(
                    [[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]],
                    dtype=float,
                )
                if position is FieldPosition.INTEGRATION_POINT
                else np.eye(3, dtype=float)
            )
            block_values = (
                values[:, :1, :]
                if position is FieldPosition.INTEGRATION_POINT
                else values
            )
            block_count = (
                1
                if position is FieldPosition.INTEGRATION_POINT
                else 3
            )
        elif position is FieldPosition.INTEGRATION_POINT:
            natural = np.asarray(
                [(xi, eta) for xi, eta, _weight in quad4_gauss_points(2)],
                dtype=float,
            )
            shape = np.asarray(
                [
                    quad4_shape_functions(xi, eta)
                    for xi, eta, _weight in quad4_gauss_points(2)
                ],
                dtype=float,
            )
            block_values = values
            block_count = 4
        else:
            natural = None
            shape = np.eye(4, dtype=float)
            block_values = np.einsum(
                "nq,eqs->ens",
                _quad4_extrapolation_matrix(2),
                values,
            )
            block_count = 4
        del natural
        row_indices = (
            row_offsets[positions, None]
            + np.arange(block_count, dtype=np.int64)[None, :]
        ).reshape(-1)
        values_components[row_indices] = block_values.reshape(-1, 4)
        coordinates[row_indices] = np.einsum(
            "qa,eak->eqk",
            shape,
            reference,
        ).reshape(-1, 3)
        displacements[row_indices] = np.einsum(
            "qa,eak->eqk",
            shape,
            displacement,
        ).reshape(-1, 3)

    return _linear_plane_field_data_from_arrays(
        source=source,
        key=key,
        descriptor=descriptor,
        values_components=values_components,
        coordinates=coordinates,
        displacements=displacements,
        element_ids=row_element_ids,
        integration_points=integration_points,
        node_ids=node_ids,
        local_nodes=local_nodes,
        region_keys=row_regions,
        cancellation=cancellation,
    )


def _linear_plane_matrices(
    elements: tuple[Any, ...],
    positions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build cached linear plane matrices for one homogeneous type block."""

    matrices = np.empty((len(positions), 3, 3), dtype=float)
    plane_strain = np.zeros(len(positions), dtype=bool)
    poisson = np.empty(len(positions), dtype=float)
    property_cache: dict[tuple[object, ...], PlaneProperties] = {}
    matrix_cache: dict[tuple[float, float, str], np.ndarray] = {}
    for block_index, element_index in enumerate(positions):
        element = elements[int(element_index)]
        source_properties = getattr(element, "props", {})
        try:
            cache_key = (
                source_properties.get("E"),
                source_properties.get("nu"),
                source_properties.get("thickness", 1.0),
                source_properties.get("plane_type", "stress"),
            )
            hash(cache_key)
        except (AttributeError, TypeError):
            cache_key = None
        properties = (
            None if cache_key is None else property_cache.get(cache_key)
        )
        if properties is None:
            properties = PlaneProperties.from_mapping(
                source_properties,
                element_id=element.id,
                element_type=getattr(element, "type", None),
            )
            if cache_key is not None:
                property_cache[cache_key] = properties
        matrix_key = (properties.E, properties.nu, properties.plane_type)
        matrix = matrix_cache.get(matrix_key)
        if matrix is None:
            matrix = linear_elastic.plane_matrix(
                properties.E,
                properties.nu,
                properties.plane_type,
            )
            matrix_cache[matrix_key] = matrix
        matrices[block_index] = matrix
        plane_strain[block_index] = properties.plane_type == "strain"
        poisson[block_index] = properties.nu
    return matrices, plane_strain, poisson


def _linear_tri3_stress_components(
    elements: tuple[Any, ...],
    positions: np.ndarray,
    reference: np.ndarray,
    displacement: np.ndarray,
) -> np.ndarray:
    """Return complete S11/S22/S33/S12 values for a Tri3 block."""

    xy = reference[:, :, :2]
    x = xy[:, :, 0]
    y = xy[:, :, 1]
    determinant = (
        x[:, 1] * y[:, 2]
        - x[:, 2] * y[:, 1]
        - x[:, 0] * y[:, 2]
        + x[:, 2] * y[:, 0]
        + x[:, 0] * y[:, 1]
        - x[:, 1] * y[:, 0]
    )
    if np.any(~np.isfinite(determinant)) or np.any(determinant <= 0.0):
        invalid = int(np.flatnonzero(determinant <= 0.0)[0])
        raise ValueError(
            f"Element {elements[int(positions[invalid])].id} has non-positive signed area"
        )
    B = np.zeros((len(positions), 3, 6), dtype=float)
    b = np.stack((y[:, 1] - y[:, 2], y[:, 2] - y[:, 0], y[:, 0] - y[:, 1]), axis=1)
    c = np.stack((x[:, 2] - x[:, 1], x[:, 0] - x[:, 2], x[:, 1] - x[:, 0]), axis=1)
    inverse_determinant = 1.0 / determinant
    columns = 2 * np.arange(3)
    B[:, 0, columns] = b * inverse_determinant[:, None]
    B[:, 1, columns + 1] = c * inverse_determinant[:, None]
    B[:, 2, columns] = c * inverse_determinant[:, None]
    B[:, 2, columns + 1] = b * inverse_determinant[:, None]
    strains = np.einsum(
        "eij,ej->ei",
        B,
        displacement[:, :, :2].reshape(len(positions), 6),
    )
    matrices, plane_strain, poisson = _linear_plane_matrices(elements, positions)
    raw = np.einsum("eij,ej->ei", matrices, strains)
    complete = np.empty((len(positions), 3, 4), dtype=float)
    complete[:, :, :2] = raw[:, None, :2]
    transverse = poisson * (raw[:, 0] + raw[:, 1])
    complete[:, :, 2] = np.where(
        plane_strain[:, None],
        transverse[:, None],
        0.0,
    )
    complete[:, :, 3] = raw[:, None, 2]
    return complete


def _linear_quad4_stress_components(
    elements: tuple[Any, ...],
    positions: np.ndarray,
    reference: np.ndarray,
    displacement: np.ndarray,
) -> np.ndarray:
    """Return complete S11/S22/S33/S12 values for a Quad4 block."""

    gauss = tuple(quad4_gauss_points(2))
    natural_gradients = np.asarray(
        [quad4_shape_grad_xi_eta(xi, eta) for xi, eta, _weight in gauss],
        dtype=float,
    )
    jacobians = np.einsum(
        "qia,eaj->eqij",
        natural_gradients,
        reference[:, :, :2],
    )
    determinants = (
        jacobians[:, :, 0, 0] * jacobians[:, :, 1, 1]
        - jacobians[:, :, 0, 1] * jacobians[:, :, 1, 0]
    )
    if np.any(~np.isfinite(determinants)) or np.any(determinants <= 0.0):
        invalid = int(np.flatnonzero(np.any(determinants <= 0.0, axis=1))[0])
        raise ValueError(
            f"Element {elements[int(positions[invalid])].id} has non-positive Jacobian determinant"
        )
    inverse = np.empty_like(jacobians)
    inverse[:, :, 0, 0] = jacobians[:, :, 1, 1] / determinants
    inverse[:, :, 0, 1] = -jacobians[:, :, 0, 1] / determinants
    inverse[:, :, 1, 0] = -jacobians[:, :, 1, 0] / determinants
    inverse[:, :, 1, 1] = jacobians[:, :, 0, 0] / determinants
    gradients = np.einsum("eqij,qja->eqia", inverse, natural_gradients)
    B = np.zeros((len(positions), 4, 3, 8), dtype=float)
    columns = 2 * np.arange(4)
    B[:, :, 0, columns] = gradients[:, :, 0, :]
    B[:, :, 1, columns + 1] = gradients[:, :, 1, :]
    B[:, :, 2, columns] = gradients[:, :, 1, :]
    B[:, :, 2, columns + 1] = gradients[:, :, 0, :]
    strains = np.einsum(
        "eqij,ej->eqi",
        B,
        displacement[:, :, :2].reshape(len(positions), 8),
    )
    matrices, plane_strain, poisson = _linear_plane_matrices(elements, positions)
    raw = np.einsum("eij,eqj->eqi", matrices, strains)
    complete = np.empty((len(positions), 4, 4), dtype=float)
    complete[:, :, :2] = raw[:, :, :2]
    complete[:, :, 2] = np.where(
        plane_strain[:, None],
        poisson[:, None] * (raw[:, :, 0] + raw[:, :, 1]),
        0.0,
    )
    complete[:, :, 3] = raw[:, :, 2]
    return np.einsum(
        "nq,eqs->ens",
        _quad4_extrapolation_matrix(2),
        complete,
    )


def _linear_quad4_integration_stress_components(
    elements: tuple[Any, ...],
    positions: np.ndarray,
    reference: np.ndarray,
    displacement: np.ndarray,
) -> np.ndarray:
    """Return complete S11/S22/S33/S12 values at the four Gauss points."""

    gauss = tuple(quad4_gauss_points(2))
    natural_gradients = np.asarray(
        [quad4_shape_grad_xi_eta(xi, eta) for xi, eta, _weight in gauss],
        dtype=float,
    )
    jacobians = np.einsum(
        "qia,eaj->eqij",
        natural_gradients,
        reference[:, :, :2],
    )
    determinants = (
        jacobians[:, :, 0, 0] * jacobians[:, :, 1, 1]
        - jacobians[:, :, 0, 1] * jacobians[:, :, 1, 0]
    )
    if np.any(~np.isfinite(determinants)) or np.any(determinants <= 0.0):
        invalid = int(np.flatnonzero(np.any(determinants <= 0.0, axis=1))[0])
        raise ValueError(
            f"Element {elements[int(positions[invalid])].id} has non-positive Jacobian determinant"
        )
    inverse = np.empty_like(jacobians)
    inverse[:, :, 0, 0] = jacobians[:, :, 1, 1] / determinants
    inverse[:, :, 0, 1] = -jacobians[:, :, 0, 1] / determinants
    inverse[:, :, 1, 0] = -jacobians[:, :, 1, 0] / determinants
    inverse[:, :, 1, 1] = jacobians[:, :, 0, 0] / determinants
    gradients = np.einsum("eqij,qja->eqia", inverse, natural_gradients)
    B = np.zeros((len(positions), 4, 3, 8), dtype=float)
    columns = 2 * np.arange(4)
    B[:, :, 0, columns] = gradients[:, :, 0, :]
    B[:, :, 1, columns + 1] = gradients[:, :, 1, :]
    B[:, :, 2, columns] = gradients[:, :, 1, :]
    B[:, :, 2, columns + 1] = gradients[:, :, 0, :]
    strains = np.einsum(
        "eqij,ej->eqi",
        B,
        displacement[:, :, :2].reshape(len(positions), 8),
    )
    matrices, plane_strain, poisson = _linear_plane_matrices(elements, positions)
    raw = np.einsum("eij,eqj->eqi", matrices, strains)
    complete = np.empty((len(positions), 4, 4), dtype=float)
    complete[:, :, :2] = raw[:, :, :2]
    complete[:, :, 2] = np.where(
        plane_strain[:, None],
        poisson[:, None] * (raw[:, :, 0] + raw[:, :, 1]),
        0.0,
    )
    complete[:, :, 3] = raw[:, :, 2]
    return complete


def _existing_element_nodal_stress_field(
    source: ResultSourceKey,
    resolved_key: FieldMaterializationKey,
    fields: tuple[FieldData, ...],
) -> FieldData | None:
    """Find a ready element-nodal stress field for one resolved request."""

    request = resolved_key.request
    for field_data in fields:
        field_request = field_data.key.request
        field_id = field_request.field_id
        if (
            field_data.source == source
            and field_id.variable is ResultVariable.S
            and field_id.position is FieldPosition.ELEMENT_NODAL
            and field_id.section_point_number is None
            and field_request.gauss_order == request.gauss_order
            and field_data.descriptor.association is FieldAssociation.ELEMENT_NODE
            and tuple(field_data.descriptor.components)
            in {
                CANONICAL_PLANE_COMPONENT_NAMES,
                CANONICAL_SOLID_COMPONENT_NAMES,
            }
        ):
            return field_data
    return None


def _materialize_resolved_nodal_stress_from_field(
    source: ResultSourceKey,
    key: FieldMaterializationKey,
    descriptor: FieldDescriptor,
    element_nodal: FieldData,
    recovery_mesh: Any,
    topology: _TopologyLookup,
    cancellation: object | None,
) -> FieldData | None:
    """Resolve an existing element-nodal field without redoing stress recovery.

    The canonical resolver works on ``StressRecord`` objects.  The result
    provider already owns the same rows in a validated columnar ``FieldData``
    object, so constructing a second Python object graph is unnecessary.  This
    path mirrors the canonical ordering, region threshold, and weighted
    averaging rules while only allocating the final resolved rows.
    """

    if descriptor.association is not FieldAssociation.RESOLVED_NODAL:
        return None
    if element_nodal.source != source:
        return None
    component_names = tuple(element_nodal.descriptor.components)
    if component_names not in {
        CANONICAL_PLANE_COMPONENT_NAMES,
        CANONICAL_SOLID_COMPONENT_NAMES,
    }:
        return None
    if tuple(descriptor.components) != component_names:
        return None
    locations = element_nodal.locations
    if not locations:
        return None
    component_count = len(component_names)
    values = np.asarray(element_nodal._values, dtype=float)
    if (
        values.ndim != 2
        or values.shape[0] != len(locations)
        or values.shape[1] < component_count
    ):
        return None
    component_values = values[:, :component_count]

    node_indices = np.empty(len(locations), dtype=np.int64)
    element_indices = np.empty(len(locations), dtype=np.int64)
    local_nodes = np.empty(len(locations), dtype=np.int64)
    node_ids = np.empty(len(locations), dtype=np.int64)
    element_ids = np.empty(len(locations), dtype=np.int64)
    region_keys: list[ResultRegionKey] = []
    element_by_id = {
        int(element.id): element for element in recovery_mesh.elements
    }
    node_lookup_: dict[int, Any] | None = None
    weights = np.empty(len(locations), dtype=float)
    weight_by_element: dict[int, float] = {}
    for index, location in enumerate(locations):
        if index % _CHECKPOINT_INTERVAL == 0:
            check_cancellation(cancellation)
        if (
            type(location) is not FieldLocation
            or location.association is not FieldAssociation.ELEMENT_NODE
            or location.node_id is None
            or location.element_id is None
            or location.local_node is None
        ):
            return None
        node_id = int(location.node_id)
        element_id = int(location.element_id)
        try:
            node_index = topology._node_order[node_id]
            element_index = topology._element_order[element_id]
        except KeyError:
            return None
        expected_region = topology.topology.element_region_keys[element_index]
        element = element_by_id.get(element_id)
        if element is None:
            return None
        node_indices[index] = node_index
        element_indices[index] = element_index
        local_nodes[index] = int(location.local_node)
        node_ids[index] = node_id
        element_ids[index] = element_id
        region_keys.append(expected_region)
        weight = weight_by_element.get(element_id)
        if weight is None:
            type_key = dispatch.type_key_from_name(element.type)
            if type_key in {"tet4", "tet10"}:
                if node_lookup_ is None:
                    node_lookup_ = node_lookup(recovery_mesh)
                weight = element_volume(recovery_mesh, element, node_lookup_)
            else:
                weight = 1.0
            if not math.isfinite(float(weight)) or float(weight) <= 0.0:
                return None
            weight_by_element[element_id] = float(weight)
        weights[index] = float(weight)

    policy = key.request.averaging_policy
    if type(policy) is not NodalAveragingPolicy:
        return None
    region_order = tuple(
        sorted(set(region_keys), key=result_region_sort_key)
    )
    region_rank = {
        region_key: index
        for index, region_key in enumerate(region_order)
    }
    region_ranks = np.asarray(
        [region_rank[region_key] for region_key in region_keys],
        dtype=np.int64,
    )
    region_ranges: list[np.ndarray] = []
    region_tolerances: list[np.ndarray] = []
    for region_index in range(len(region_order)):
        check_cancellation(cancellation)
        region_values = component_values[region_ranks == region_index]
        if len(region_values) == 0:
            return None
        region_ranges.append(np.ptp(region_values, axis=0))
        region_tolerances.append(
            np.finfo(float).eps
            * np.maximum(1.0, np.max(np.abs(region_values), axis=0))
            * 32.0
        )

    order = np.lexsort(
        (
            local_nodes,
            element_indices,
            region_ranks,
            node_indices,
        )
    )
    sorted_nodes = node_indices[order]
    sorted_regions = region_ranks[order]
    split_points = np.flatnonzero(
        (sorted_nodes[1:] != sorted_nodes[:-1])
        | (sorted_regions[1:] != sorted_regions[:-1])
    ) + 1
    group_starts = np.concatenate(([0], split_points))
    group_stops = np.concatenate((split_points, [len(order)]))

    output_components: list[np.ndarray] = []
    output_locations: list[FieldLocation] = []
    threshold = float(policy.threshold_percent)
    for group_index, (start, stop) in enumerate(
        zip(group_starts, group_stops)
    ):
        if group_index % _CHECKPOINT_INTERVAL == 0:
            check_cancellation(cancellation)
        selected = order[start:stop]
        region_key = region_keys[int(selected[0])]
        group_values = component_values[selected]
        average = False
        if len(selected) > 1 and threshold != 0.0:
            node_ranges = np.ptp(group_values, axis=0)
            ranges = region_ranges[region_rank[region_key]]
            tolerances = region_tolerances[region_rank[region_key]]
            relative_variation = np.zeros_like(node_ranges)
            outside_tolerance = ranges > tolerances
            relative_variation[outside_tolerance] = (
                100.0
                * node_ranges[outside_tolerance]
                / ranges[outside_tolerance]
            )
            average = bool(np.all(relative_variation <= threshold))
        if average:
            first_index = int(selected[0])
            averaged_components = np.average(
                group_values,
                axis=0,
                weights=weights[selected],
            )
            first_location = locations[first_index]
            output_components.append(averaged_components)
            output_locations.append(
                FieldLocation._from_validated_components(
                    descriptor.association,
                    first_location.coordinates,
                    first_location.displacement
                    if first_location.displacement is not None
                    else tuple(
                        float(value)
                        for value in topology._nodal_displacements[
                            node_indices[first_index]
                        ]
                    ),
                    node_id=int(node_ids[first_index]),
                    region_key=region_key,
                    averaged=True,
                )
            )
            continue
        for selected_index in selected:
            selected_index = int(selected_index)
            location = locations[selected_index]
            output_components.append(component_values[selected_index])
            output_locations.append(
                FieldLocation._from_validated_components(
                    descriptor.association,
                    location.coordinates,
                    location.displacement
                    if location.displacement is not None
                    else tuple(
                        float(value)
                        for value in topology._nodal_displacements[
                            node_indices[selected_index]
                        ]
                    ),
                    node_id=int(node_ids[selected_index]),
                    region_key=region_key,
                    averaged=False,
                    element_id=int(element_ids[selected_index]),
                    local_node=int(local_nodes[selected_index]),
                )
            )
    resolved_components = np.asarray(
        output_components,
        dtype=float,
    ).reshape((-1, component_count))
    return FieldData._from_materialized_values(
        descriptor=descriptor,
        source=source,
        key=key,
        locations=tuple(output_locations),
        values=_stress_value_matrix(resolved_components, descriptor),
    )


def _materialize_quad4_linear_stress_field(
    source: ResultSourceKey,
    key: FieldMaterializationKey,
    descriptor: FieldDescriptor,
    recovery: StressRecovery | CapturedStressRecovery,
    topology: _TopologyLookup,
    cancellation: object | None,
) -> FieldData | None:
    """Build common Quad4 stress fields without intermediate StressRecords.

    ``StressRecord`` remains the public recovery API.  The result provider,
    however, only needs columnar values and typed locations.  For homogeneous
    linear Quad4 centroid/integration-point fields, skipping the transient
    record and invariant dataclass allocations removes a large Python object
    graph while preserving every stored component and location identity.
    """

    position = key.request.field_id.position
    if position not in {
        FieldPosition.CENTROID,
        FieldPosition.INTEGRATION_POINT,
    }:
        return None
    if key.request.gauss_order not in (None, 2):
        return None
    if descriptor.components != ("S11", "S22", "S33", "S12"):
        return None
    if not isinstance(recovery, (StressRecovery, CapturedStressRecovery)):
        return None
    fields = tuple(getattr(recovery, "_ip_fields", ()))
    if not fields or any(
        item.type_key != "quad4"
        or item.gauss_order not in (None, 2)
        or item.components.shape != (4, 4)
        for item in fields
    ):
        return None

    element_indices = []
    for item in fields:
        try:
            element_index = topology._element_order[int(item.elem.id)]
        except KeyError:
            return None
        if (
            topology.topology.element_region_keys[element_index]
            != item.region_key
        ):
            return None
        element_indices.append(element_index)
    element_coordinates = topology.topology._node_coordinates[
        np.asarray(
            [
                [
                    topology._node_order[int(node_id)]
                    for node_id in topology.topology.connectivity[index]
                ]
                for index in element_indices
            ],
            dtype=int,
        )
    ]
    element_displacements = topology._nodal_displacements[
        np.asarray(
            [topology._connectivity_indices[index] for index in element_indices],
            dtype=int,
        )
    ]
    integration_values = np.asarray(
        [item.components for item in fields],
        dtype=float,
    )
    if position is FieldPosition.CENTROID:
        recovery_matrix = _quad4_centroid_recovery_matrix(2)
        values_components = np.einsum(
            "cq,eqs->ecs",
            recovery_matrix,
            integration_values,
        )[:, 0, :]
        shape_values = np.asarray(
            [quad4_shape_functions(0.0, 0.0)],
            dtype=float,
        )
        element_ids = np.asarray(
            [int(item.elem.id) for item in fields],
            dtype=int,
        )
        integration_points = None
        region_keys = tuple(item.region_key for item in fields)
    else:
        values_components = integration_values.reshape(-1, 4)
        shape_values = np.asarray(
            [
                quad4_shape_functions(xi, eta)
                for xi, eta, _weight in quad4_gauss_points(2)
            ],
            dtype=float,
        )
        element_ids = np.repeat(
            np.asarray([int(item.elem.id) for item in fields], dtype=int),
            4,
        )
        integration_points = np.tile(np.arange(1, 5, dtype=int), len(fields))
        region_keys = tuple(
            item.region_key for item in fields for _ in range(4)
        )

    return _quad4_field_data_from_components(
        source=source,
        key=key,
        descriptor=descriptor,
        values_components=values_components,
        shape_values=shape_values,
        element_coordinates=element_coordinates,
        element_displacements=element_displacements,
        element_ids=element_ids,
        integration_points=integration_points,
        region_keys=region_keys,
        cancellation=cancellation,
    )


def _quad4_field_data_from_components(
    *,
    source: ResultSourceKey,
    key: FieldMaterializationKey,
    descriptor: FieldDescriptor,
    values_components: np.ndarray,
    shape_values: np.ndarray,
    element_coordinates: np.ndarray,
    element_displacements: np.ndarray,
    element_ids: np.ndarray,
    integration_points: np.ndarray | None,
    region_keys: tuple[ResultRegionKey, ...],
    cancellation: object | None,
) -> FieldData:
    coordinates = np.einsum(
        "qa,eak->eqk",
        shape_values,
        element_coordinates,
    ).reshape(-1, 3)
    displacements = np.einsum(
        "qa,eak->eqk",
        shape_values,
        element_displacements,
    ).reshape(-1, 3)
    if not (
        np.all(np.isfinite(values_components))
        and np.all(np.isfinite(coordinates))
        and np.all(np.isfinite(displacements))
    ):
        raise ValueError("Quad4 materialized stress values must be finite")

    sig_x = values_components[:, 0]
    sig_y = values_components[:, 1]
    sig_z = values_components[:, 2]
    tau_xy = values_components[:, 3]
    mises = np.sqrt(
        0.5 * (
            (sig_x - sig_y) ** 2
            + (sig_y - sig_z) ** 2
            + (sig_z - sig_x) ** 2
        )
        + 3.0 * tau_xy**2
    )
    mean = 0.5 * (sig_x + sig_y)
    radius = np.hypot(0.5 * (sig_x - sig_y), tau_xy)
    principal = np.sort(
        np.column_stack((mean - radius, mean + radius, sig_z)),
        axis=1,
    )
    derived = {
        "Mises": mises,
        "MaxPrincipal": principal[:, 2],
        "MidPrincipal": principal[:, 1],
        "MinPrincipal": principal[:, 0],
    }
    column_values = []
    for column in descriptor.columns:
        if column in {"S11", "S22", "S33", "S12"}:
            column_values.append(
                values_components[:, ("S11", "S22", "S33", "S12").index(column)]
            )
        else:
            try:
                column_values.append(derived[column])
            except KeyError as error:
                raise ValueError(f"unsupported stress result column {column!r}") from error
    value_matrix = np.column_stack(column_values)
    value_matrix = np.asarray(value_matrix, dtype=float, order="C")
    value_matrix.flags.writeable = False

    association = descriptor.association
    locations = []
    for index in range(len(value_matrix)):
        if index % _CHECKPOINT_INTERVAL == 0:
            check_cancellation(cancellation)
        locations.append(
            FieldLocation._from_validated_components(
                association,
                tuple(float(value) for value in coordinates[index]),
                tuple(float(value) for value in displacements[index]),
                element_id=int(element_ids[index]),
                integration_point=(
                    None
                    if integration_points is None
                    else int(integration_points[index])
                ),
                region_key=region_keys[index],
            )
        )
    return FieldData._from_materialized_values(
        descriptor=descriptor,
        source=source,
        key=key,
        locations=tuple(locations),
        values=value_matrix,
    )


def _materialize_quad4_captured_resolved_nodal_stress_field(
    source: ResultSourceKey,
    key: FieldMaterializationKey,
    descriptor: FieldDescriptor,
    recovery: StressRecovery | CapturedStressRecovery,
    topology: _TopologyLookup,
    cancellation: object | None,
) -> FieldData | None:
    """Project captured Quad4 stress to resolved nodes without row objects.

    This path is intentionally restricted to the canonical four-point plane
    Quad4 contract.  It reproduces ``resolve_nodal_stress``'s global
    per-region threshold test and deterministic node/element ordering, while
    keeping the large intermediate tensor data in NumPy arrays instead of
    creating one ``StressRecord`` for every element-node contribution.
    """

    position = key.request.field_id.position
    if position is not FieldPosition.RESOLVED_NODAL:
        return None
    if not isinstance(recovery, CapturedStressRecovery):
        return None
    if descriptor.association is not FieldAssociation.RESOLVED_NODAL:
        return None
    if descriptor.components != CANONICAL_PLANE_COMPONENT_NAMES:
        return None
    if key.request.gauss_order not in (None, 2):
        return None
    policy = key.request.averaging_policy
    if not isinstance(policy, NodalAveragingPolicy):
        return None

    fields = tuple(getattr(recovery, "_ip_fields", ()))
    if not fields or any(
        item.type_key != "quad4"
        or item.gauss_order not in (None, 2)
        or np.asarray(item.components, dtype=float).shape != (4, 4)
        for item in fields
    ):
        return None
    field_by_element = {int(item.elem.id): item for item in fields}
    element_ids = np.asarray(topology.topology.element_ids, dtype=int)
    if set(field_by_element) != set(int(value) for value in element_ids):
        return None
    ordered_fields = tuple(field_by_element[int(value)] for value in element_ids)
    if any(
        item.region_key != topology.topology.element_region_keys[index]
        for index, item in enumerate(ordered_fields)
    ):
        return None
    integration_values = np.asarray(
        [item.components for item in ordered_fields],
        dtype=float,
    )
    if integration_values.shape != (len(element_ids), 4, 4):
        return None
    check_cancellation(cancellation)

    nodal_values = np.einsum(
        "nq,eqs->ens",
        _quad4_extrapolation_matrix(2),
        integration_values,
    )
    if not np.all(np.isfinite(nodal_values)):
        raise ValueError("captured Quad4 nodal stress values must be finite")

    connectivity_indices = np.stack(topology._connectivity_indices, axis=0)
    connectivity_ids = np.asarray(topology.topology.connectivity, dtype=int)
    element_positions = np.repeat(np.arange(len(element_ids), dtype=int), 4)
    local_nodes = np.tile(np.arange(1, 5, dtype=int), len(element_ids))
    node_indices = connectivity_indices.reshape(-1)
    node_ids = connectivity_ids.reshape(-1)
    values = nodal_values.reshape(-1, 4)
    weights = np.repeat(
        np.asarray([float(item.weight) for item in ordered_fields], dtype=float),
        4,
    )
    region_values = topology.topology.element_region_keys
    region_order = tuple(
        sorted(
            set(region_values),
            key=result_region_sort_key,
        )
    )
    region_rank_by_key = {
        region_key: index
        for index, region_key in enumerate(region_order)
    }
    flat_region_ranks = np.asarray(
        [
            region_rank_by_key[region_values[element_index]]
            for element_index in element_positions
        ],
        dtype=int,
    )
    order = np.lexsort(
        (
            local_nodes,
            element_positions,
            flat_region_ranks,
            node_indices,
        )
    )
    sorted_regions = flat_region_ranks[order]
    sorted_nodes = node_indices[order]
    split_points = np.flatnonzero(
        (sorted_nodes[1:] != sorted_nodes[:-1])
        | (sorted_regions[1:] != sorted_regions[:-1])
    ) + 1
    group_starts = np.concatenate(([0], split_points))
    group_stops = np.concatenate((split_points, [len(order)]))

    region_ranges: dict[ResultRegionKey, np.ndarray] = {}
    region_tolerances: dict[ResultRegionKey, np.ndarray] = {}
    for region_index, region_key in enumerate(region_order):
        mask = flat_region_ranks == region_index
        region_data = values[mask]
        region_ranges[region_key] = np.ptp(region_data, axis=0)
        region_tolerances[region_key] = (
            np.finfo(float).eps
            * np.maximum(1.0, np.max(np.abs(region_data), axis=0))
            * 32.0
        )

    row_values: list[np.ndarray] = []
    row_coordinates: list[np.ndarray] = []
    row_displacements: list[np.ndarray] = []
    row_node_ids: list[int] = []
    row_element_ids: list[int | None] = []
    row_local_nodes: list[int | None] = []
    row_regions: list[ResultRegionKey] = []
    row_averaged: list[bool] = []
    threshold = float(policy.threshold_percent)
    for group_index, (start, stop) in enumerate(
        zip(group_starts, group_stops)
    ):
        if group_index % _CHECKPOINT_INTERVAL == 0:
            check_cancellation(cancellation)
        selected = order[start:stop]
        region_key = region_values[int(element_positions[selected[0]])]
        group_values = values[selected]
        average = False
        if len(selected) > 1 and threshold != 0.0:
            node_ranges = np.ptp(group_values, axis=0)
            ranges = region_ranges[region_key]
            tolerances = region_tolerances[region_key]
            relative_variation = np.zeros_like(node_ranges)
            outside_tolerance = ranges > tolerances
            relative_variation[outside_tolerance] = (
                100.0 * node_ranges[outside_tolerance]
                / ranges[outside_tolerance]
            )
            average = bool(np.all(relative_variation <= threshold))
        if average:
            first = int(selected[0])
            row_values.append(
                np.average(
                    group_values,
                    axis=0,
                    weights=weights[selected],
                )
            )
            row_node_index = int(node_indices[first])
            row_coordinates.append(
                topology.topology._node_coordinates[row_node_index]
            )
            row_displacements.append(
                topology._nodal_displacements[row_node_index]
            )
            row_node_ids.append(int(node_ids[first]))
            row_element_ids.append(None)
            row_local_nodes.append(None)
            row_regions.append(region_key)
            row_averaged.append(True)
            continue
        for selected_index in selected:
            selected_index = int(selected_index)
            row_values.append(values[selected_index])
            row_node_index = int(node_indices[selected_index])
            row_coordinates.append(
                topology.topology._node_coordinates[row_node_index]
            )
            row_displacements.append(
                topology._nodal_displacements[row_node_index]
            )
            row_node_ids.append(int(node_ids[selected_index]))
            row_element_ids.append(
                int(element_ids[element_positions[selected_index]])
            )
            row_local_nodes.append(int(local_nodes[selected_index]))
            row_regions.append(region_key)
            row_averaged.append(False)

    return _quad4_resolved_nodal_field_data(
        source=source,
        key=key,
        descriptor=descriptor,
        values_components=np.asarray(row_values, dtype=float),
        coordinates=np.asarray(row_coordinates, dtype=float),
        displacements=np.asarray(row_displacements, dtype=float),
        node_ids=np.asarray(row_node_ids, dtype=int),
        element_ids=row_element_ids,
        local_nodes=row_local_nodes,
        region_keys=tuple(row_regions),
        averaged=np.asarray(row_averaged, dtype=bool),
        cancellation=cancellation,
    )


def _stress_value_matrix(
    values_components: np.ndarray,
    descriptor: FieldDescriptor,
) -> np.ndarray:
    """Return canonical stress components plus requested invariant columns."""

    base_names = (
        CANONICAL_PLANE_COMPONENT_NAMES
        if descriptor.components == CANONICAL_PLANE_COMPONENT_NAMES
        else CANONICAL_SOLID_COMPONENT_NAMES
    )
    values_components = np.asarray(values_components, dtype=float)
    if values_components.ndim != 2 or values_components.shape[1] != len(base_names):
        raise ValueError(
            "stress components must have shape "
            f"(n, {len(base_names)})"
        )
    sig_x = values_components[:, 0]
    sig_y = values_components[:, 1]
    sig_z = values_components[:, 2]
    tau_xy = values_components[:, 3]
    tau_yz = (
        values_components[:, 4]
        if len(base_names) == len(CANONICAL_SOLID_COMPONENT_NAMES)
        else np.zeros(len(values_components), dtype=float)
    )
    tau_zx = (
        values_components[:, 5]
        if len(base_names) == len(CANONICAL_SOLID_COMPONENT_NAMES)
        else np.zeros(len(values_components), dtype=float)
    )
    mises = np.sqrt(
        0.5 * (
            (sig_x - sig_y) ** 2
            + (sig_y - sig_z) ** 2
            + (sig_z - sig_x) ** 2
        )
        + 3.0 * (tau_xy**2 + tau_yz**2 + tau_zx**2)
    )
    principal = np.empty((len(values_components), 3), dtype=float)
    zero_out_of_plane_shear = (tau_yz == 0.0) & (tau_zx == 0.0)
    if np.any(zero_out_of_plane_shear):
        mean = 0.5 * (sig_x + sig_y)
        radius = np.hypot(0.5 * (sig_x - sig_y), tau_xy)
        principal[zero_out_of_plane_shear] = np.sort(
            np.column_stack(
                (
                    mean[zero_out_of_plane_shear]
                    - radius[zero_out_of_plane_shear],
                    mean[zero_out_of_plane_shear]
                    + radius[zero_out_of_plane_shear],
                    sig_z[zero_out_of_plane_shear],
                )
            ),
            axis=1,
        )
    full_tensor = ~zero_out_of_plane_shear
    if np.any(full_tensor):
        tensors = np.zeros(
            (int(np.count_nonzero(full_tensor)), 3, 3),
            dtype=float,
        )
        tensors[:, 0, 0] = sig_x[full_tensor]
        tensors[:, 1, 1] = sig_y[full_tensor]
        tensors[:, 2, 2] = sig_z[full_tensor]
        tensors[:, 0, 1] = tensors[:, 1, 0] = tau_xy[full_tensor]
        tensors[:, 1, 2] = tensors[:, 2, 1] = tau_yz[full_tensor]
        tensors[:, 0, 2] = tensors[:, 2, 0] = tau_zx[full_tensor]
        principal[full_tensor] = np.linalg.eigvalsh(tensors)
    derived = {
        "Mises": mises,
        "MaxPrincipal": principal[:, 2],
        "MidPrincipal": principal[:, 1],
        "MinPrincipal": principal[:, 0],
    }
    columns = []
    for column in descriptor.columns:
        if column in base_names:
            columns.append(values_components[:, base_names.index(column)])
            continue
        try:
            columns.append(derived[column])
        except KeyError as error:
            raise ValueError(
                f"unsupported stress result column {column!r}"
            ) from error
    value_matrix = np.asarray(
        np.column_stack(columns),
        dtype=float,
        order="C",
    )
    value_matrix.flags.writeable = False
    return value_matrix


def _linear_plane_field_data_from_arrays(
    *,
    source: ResultSourceKey,
    key: FieldMaterializationKey,
    descriptor: FieldDescriptor,
    values_components: np.ndarray,
    coordinates: np.ndarray,
    displacements: np.ndarray,
    element_ids: np.ndarray,
    integration_points: np.ndarray | None,
    node_ids: np.ndarray | None,
    local_nodes: np.ndarray | None,
    region_keys: tuple[ResultRegionKey, ...],
    cancellation: object | None,
) -> FieldData:
    """Build typed linear-plane rows after batched numerical recovery."""

    values_components = np.asarray(values_components, dtype=float)
    coordinates = np.asarray(coordinates, dtype=float)
    displacements = np.asarray(displacements, dtype=float)
    element_ids = np.asarray(element_ids, dtype=np.int64)
    row_count = len(values_components)
    if not (
        values_components.shape == (row_count, 4)
        and coordinates.shape == (row_count, 3)
        and displacements.shape == (row_count, 3)
        and element_ids.shape == (row_count,)
        and len(region_keys) == row_count
        and np.all(np.isfinite(values_components))
        and np.all(np.isfinite(coordinates))
        and np.all(np.isfinite(displacements))
    ):
        raise ValueError("linear plane stress rows have inconsistent shapes")
    if descriptor.components != CANONICAL_PLANE_COMPONENT_NAMES:
        raise ValueError("linear plane stress descriptor components are invalid")
    if descriptor.association is FieldAssociation.INTEGRATION_POINT:
        if (
            integration_points is None
            or integration_points.shape != (row_count,)
            or node_ids is not None
            or local_nodes is not None
        ):
            raise ValueError("integration-point row identities are invalid")
    elif descriptor.association is FieldAssociation.ELEMENT:
        if (
            integration_points is not None
            or node_ids is not None
            or local_nodes is not None
        ):
            raise ValueError("element row identities are invalid")
    elif descriptor.association is FieldAssociation.ELEMENT_NODE:
        if (
            node_ids is None
            or local_nodes is None
            or integration_points is not None
            or node_ids.shape != (row_count,)
            or local_nodes.shape != (row_count,)
        ):
            raise ValueError("element-node row identities are invalid")
    else:
        raise ValueError(
            f"unsupported linear plane stress association {descriptor.association.value}"
        )

    value_matrix = _stress_value_matrix(values_components, descriptor)
    locations: list[FieldLocation] = []
    for index in range(row_count):
        if index % _CHECKPOINT_INTERVAL == 0:
            check_cancellation(cancellation)
        identity: dict[str, object] = {
            "element_id": int(element_ids[index]),
            # Canonical continuum element/IP rows validate region ownership
            # internally but do not publish region_key in their locations.
            "region_key": None,
        }
        if descriptor.association is FieldAssociation.INTEGRATION_POINT:
            assert integration_points is not None
            identity["integration_point"] = int(integration_points[index])
        elif descriptor.association is FieldAssociation.ELEMENT_NODE:
            assert node_ids is not None and local_nodes is not None
            identity["node_id"] = int(node_ids[index])
            identity["local_node"] = int(local_nodes[index])
        locations.append(
            FieldLocation._from_validated_components(
                descriptor.association,
                tuple(float(value) for value in coordinates[index]),
                tuple(float(value) for value in displacements[index]),
                **identity,
            )
        )
    return FieldData._from_materialized_values(
        descriptor=descriptor,
        source=source,
        key=key,
        locations=tuple(locations),
        values=value_matrix,
    )


def _quad4_resolved_nodal_field_data(
    *,
    source: ResultSourceKey,
    key: FieldMaterializationKey,
    descriptor: FieldDescriptor,
    values_components: np.ndarray,
    coordinates: np.ndarray,
    displacements: np.ndarray,
    node_ids: np.ndarray,
    element_ids: list[int | None],
    local_nodes: list[int | None],
    region_keys: tuple[ResultRegionKey, ...],
    averaged: np.ndarray,
    cancellation: object | None,
) -> FieldData:
    value_matrix = _stress_value_matrix(values_components, descriptor)
    if not (
        coordinates.shape == (len(value_matrix), 3)
        and displacements.shape == (len(value_matrix), 3)
        and node_ids.shape == (len(value_matrix),)
        and averaged.shape == (len(value_matrix),)
        and len(element_ids) == len(value_matrix)
        and len(local_nodes) == len(value_matrix)
        and len(region_keys) == len(value_matrix)
    ):
        raise ValueError("resolved Quad4 stress rows have inconsistent shapes")
    locations = []
    for index in range(len(value_matrix)):
        if index % _CHECKPOINT_INTERVAL == 0:
            check_cancellation(cancellation)
        is_averaged = bool(averaged[index])
        locations.append(
            FieldLocation._from_validated_components(
                descriptor.association,
                tuple(float(value) for value in coordinates[index]),
                tuple(float(value) for value in displacements[index]),
                node_id=int(node_ids[index]),
                region_key=region_keys[index],
                averaged=is_averaged,
                element_id=None if is_averaged else element_ids[index],
                local_node=None if is_averaged else local_nodes[index],
            )
        )
    return FieldData._from_materialized_values(
        descriptor=descriptor,
        source=source,
        key=key,
        locations=tuple(locations),
        values=value_matrix,
    )


def _materialize_quad4_linear_stress_from_mesh(
    source: ResultSourceKey,
    key: FieldMaterializationKey,
    descriptor: FieldDescriptor,
    mesh: Any,
    topology: _TopologyLookup,
    cancellation: object | None,
) -> FieldData | None:
    """Directly batch-recover the common uncaptured linear Quad4 frame."""

    position = key.request.field_id.position
    if position not in {
        FieldPosition.CENTROID,
        FieldPosition.INTEGRATION_POINT,
    }:
        return None
    if key.request.gauss_order not in (None, 2):
        return None
    if descriptor.components != ("S11", "S22", "S33", "S12"):
        return None
    elements = tuple(getattr(mesh, "elements", ()))
    if not elements or tuple(int(elem.id) for elem in elements) != topology.topology.element_ids:
        return None
    if any(str(elem.type).strip().casefold() != "quad4" for elem in elements):
        return None
    check_cancellation(cancellation)

    connectivity = np.stack(topology._connectivity_indices, axis=0)
    reference = topology.topology._node_coordinates[connectivity]
    element_displacements = topology._nodal_displacements[connectivity]
    gauss = tuple(quad4_gauss_points(2))
    natural_gradients = np.asarray(
        [quad4_shape_grad_xi_eta(xi, eta) for xi, eta, _weight in gauss],
        dtype=float,
    )
    jacobians = np.einsum(
        "qia,eaj->eqij",
        natural_gradients,
        reference[:, :, :2],
    )
    determinants = (
        jacobians[:, :, 0, 0] * jacobians[:, :, 1, 1]
        - jacobians[:, :, 0, 1] * jacobians[:, :, 1, 0]
    )
    if np.any(~np.isfinite(determinants)) or np.any(determinants <= 0.0):
        invalid = int(np.flatnonzero(np.any(determinants <= 0.0, axis=1))[0])
        raise ValueError(
            f"Element {elements[invalid].id} has non-positive Jacobian determinant"
        )
    inverse = np.empty_like(jacobians)
    inverse[:, :, 0, 0] = jacobians[:, :, 1, 1] / determinants
    inverse[:, :, 0, 1] = -jacobians[:, :, 0, 1] / determinants
    inverse[:, :, 1, 0] = -jacobians[:, :, 1, 0] / determinants
    inverse[:, :, 1, 1] = jacobians[:, :, 0, 0] / determinants
    gradients = np.einsum("eqij,qja->eqia", inverse, natural_gradients)
    B = np.zeros((len(elements), 4, 3, 8), dtype=float)
    columns = 2 * np.arange(4)
    B[:, :, 0, columns] = gradients[:, :, 0, :]
    B[:, :, 1, columns + 1] = gradients[:, :, 1, :]
    B[:, :, 2, columns] = gradients[:, :, 1, :]
    B[:, :, 2, columns + 1] = gradients[:, :, 0, :]
    strains = np.einsum("eqij,ej->eqi", B, element_displacements[:, :, :2].reshape(len(elements), 8))

    matrices = np.empty((len(elements), 3, 3), dtype=float)
    plane_strain = np.zeros(len(elements), dtype=bool)
    poisson = np.empty(len(elements), dtype=float)
    property_cache: dict[tuple[object, ...], PlaneProperties] = {}
    matrix_cache: dict[tuple[float, float, str], np.ndarray] = {}
    for index, elem in enumerate(elements):
        source_properties = getattr(elem, "props", {})
        cache_key: tuple[object, ...] | None
        try:
            cache_key = (
                source_properties.get("E"),
                source_properties.get("nu"),
                source_properties.get("thickness", 1.0),
                source_properties.get("plane_type", "stress"),
            )
            hash(cache_key)
        except (AttributeError, TypeError):
            cache_key = None
        properties = (
            None
            if cache_key is None
            else property_cache.get(cache_key)
        )
        if properties is None:
            properties = PlaneProperties.from_mapping(
                source_properties,
                element_id=elem.id,
                element_type=getattr(elem, "type", None),
            )
            if cache_key is not None:
                property_cache[cache_key] = properties
        matrix_key = (properties.E, properties.nu, properties.plane_type)
        matrix = matrix_cache.get(matrix_key)
        if matrix is None:
            matrix = linear_elastic.plane_matrix(
                properties.E,
                properties.nu,
                properties.plane_type,
            )
            matrix_cache[matrix_key] = matrix
        matrices[index] = matrix
        plane_strain[index] = properties.plane_type == "strain"
        poisson[index] = properties.nu
    raw_components = np.einsum("eij,eqj->eqi", matrices, strains)
    complete = np.empty((len(elements), 4, 4), dtype=float)
    complete[:, :, :2] = raw_components[:, :, :2]
    complete[:, :, 2] = np.where(
        plane_strain[:, None],
        poisson[:, None] * (raw_components[:, :, 0] + raw_components[:, :, 1]),
        0.0,
    )
    complete[:, :, 3] = raw_components[:, :, 2]
    if position is FieldPosition.CENTROID:
        values_components = np.einsum(
            "cq,eqs->ecs",
            _quad4_centroid_recovery_matrix(2),
            complete,
        )[:, 0, :]
        shape_values = np.asarray(
            [quad4_shape_functions(0.0, 0.0)],
            dtype=float,
        )
        element_ids = np.asarray(topology.topology.element_ids, dtype=int)
        integration_points = None
        region_keys = tuple(topology.topology.element_region_keys)
    else:
        values_components = complete.reshape(-1, 4)
        shape_values = np.asarray(
            [quad4_shape_functions(xi, eta) for xi, eta, _weight in gauss],
            dtype=float,
        )
        element_ids = np.repeat(
            np.asarray(topology.topology.element_ids, dtype=int),
            4,
        )
        integration_points = np.tile(np.arange(1, 5, dtype=int), len(elements))
        region_keys = tuple(
            region
            for region in topology.topology.element_region_keys
            for _ in range(4)
        )
    # Keep only the stress block and geometry needed by the location builder;
    # the temporary Jacobian/B/strain arrays otherwise remain live through
    # every FieldLocation allocation and inflate the peak of large frames.
    values_components = np.array(
        values_components,
        dtype=float,
        order="C",
        copy=True,
    )
    del (
        B,
        complete,
        determinants,
        gradients,
        inverse,
        jacobians,
        matrices,
        natural_gradients,
        raw_components,
        strains,
    )
    return _quad4_field_data_from_components(
        source=source,
        key=key,
        descriptor=descriptor,
        values_components=values_components,
        shape_values=shape_values,
        element_coordinates=reference[:, :, :3],
        element_displacements=element_displacements,
        element_ids=element_ids,
        integration_points=integration_points,
        region_keys=region_keys,
        cancellation=cancellation,
    )


def _has_captured_continuum_stress_outputs(result: ModelResult) -> bool:
    """Return whether the solver already captured continuum stress values."""

    outputs = result.outputs
    integration = outputs.get("integration_points")
    return (
        isinstance(integration, Mapping)
        and "element_id" in integration
        and "natural_coordinates" in integration
        and (
            "cauchy_stress" in integration
            or "kirchhoff_stress" in integration
        )
    )


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
        FieldRecoveryKind.BEAM_INTEGRATION_POINT_SF,
        FieldRecoveryKind.BEAM_INTEGRATION_POINT_SM,
        FieldRecoveryKind.BEAM_INTEGRATION_POINT_S11,
    }
    if any(
        entry.recovery_kind not in allowed
        for _key, entry in targets
    ):
        raise ValueError("Beam materialization received a non-Beam target")
    check_cancellation(cancellation)
    checkpoint = _checkpoint_for_cancellation(cancellation)
    integration_point = beam.recover_integration_point_stress(
        result,
        checkpoint=checkpoint,
    )
    fields = []
    for key, entry in targets:
        if entry.recovery_kind is FieldRecoveryKind.BEAM_INTEGRATION_POINT_S11:
            point_number = key.request.field_id.section_point_number
            assert point_number is not None
            rows = integration_point.point_field(point_number).rows
        else:
            rows = integration_point.section_forces.rows
        fields.append(
            _simple_rows_field(
                source=source,
                key=key,
                descriptor=entry.descriptor,
                rows=rows,
                association=FieldAssociation.INTEGRATION_POINT,
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
    component_indexes = {
        name: component_index
        for component_index, name in enumerate(recovered.component_names)
    }
    invariant_attributes = {
        "Mises": "mises",
        "MaxPrincipal": "max_principal",
        "MidPrincipal": "mid_principal",
        "MinPrincipal": "min_principal",
    }
    columns: list[tuple[int | None, str | None]] = []
    for column in descriptor.columns:
        component_index = component_indexes.get(column)
        invariant_attribute = (
            None
            if component_index is not None
            else invariant_attributes.get(column)
        )
        if component_index is None and invariant_attribute is None:
            raise ValueError(f"unsupported stress result column {column!r}")
        columns.append((component_index, invariant_attribute))
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
        values.append(
            tuple(
                (
                    float(record.components[component_index])
                    if component_index is not None
                    else float(
                        getattr(record.invariants, invariant_attribute)
                    )
                )
                for component_index, invariant_attribute in columns
            )
        )
    value_matrix = _value_matrix(values, len(descriptor.columns))
    return FieldData._from_materialized_values(
        descriptor=descriptor,
        source=source,
        key=key,
        locations=tuple(locations),
        values=value_matrix,
    )


def _finite_strain_state_field_data(
    source: ResultSourceKey,
    key: FieldMaterializationKey,
    descriptor: FieldDescriptor,
    recovered: FiniteStrainStateField,
    topology: _TopologyLookup,
    cancellation: object | None,
) -> FieldData:
    if tuple(recovered.component_names) != descriptor.components:
        raise ValueError(
            "finite-strain state recovery components do not match the descriptor"
        )
    if descriptor.derived_components:
        raise ValueError(
            "finite-strain state fields do not publish derived components"
        )
    locations = []
    values = []
    for index, record in enumerate(recovered.records):
        if index % _CHECKPOINT_INTERVAL == 0:
            check_cancellation(cancellation)
        if type(record) is not FiniteStrainStateRecord:
            raise TypeError(
                "finite-strain state recovery rows must be "
                "FiniteStrainStateRecord"
            )
        topology.validate_region(record)
        locations.append(
            _continuum_location(
                descriptor.association,
                record,
                topology,
            )
        )
        values.append(tuple(float(value) for value in record.components))
    return FieldData._from_materialized_values(
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
        return FieldLocation._from_finite_components(
            association,
            coordinates,
            displacement,
            element_id=record.elem_id,
            integration_point=record.integration_point,
        )
    if association is FieldAssociation.ELEMENT:
        return FieldLocation._from_finite_components(
            association,
            coordinates,
            displacement,
            element_id=record.elem_id,
        )
    if association is FieldAssociation.ELEMENT_NODE:
        return FieldLocation._from_finite_components(
            association,
            coordinates,
            displacement,
            element_id=record.elem_id,
            local_node=record.local_node,
            node_id=record.node_id,
        )
    if association is FieldAssociation.NODE_REGION:
        return FieldLocation._from_finite_components(
            association,
            coordinates,
            displacement,
            node_id=record.node_id,
            region_key=record.region_key,
        )
    if association is FieldAssociation.RESOLVED_NODAL:
        return FieldLocation._from_finite_components(
            association,
            coordinates,
            displacement,
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
        row_values = (
            row.section_values()
            if "S12AbsMax" in descriptor.columns
            and callable(getattr(row, "section_values", None))
            else row.values()
        )
        values.append(
            tuple(float(row_values[column]) for column in descriptor.columns)
        )
        if association is FieldAssociation.ELEMENT:
            locations.append(
                FieldLocation._from_finite_components(
                    association,
                    _triplet(
                        row.coordinates,
                        label="element coordinates",
                    ),
                    _triplet(
                        row.displacement,
                        label="element displacement",
                    ),
                    element_id=row.element_id,
                )
            )
        elif association is FieldAssociation.ELEMENT_NODE:
            locations.append(
                FieldLocation._from_finite_components(
                    association,
                    _triplet(
                        row.coordinates,
                        label="element-node coordinates",
                    ),
                    _triplet(
                        row.displacement,
                        label="element-node displacement",
                    ),
                    element_id=row.element_id,
                    local_node=row.local_node,
                    node_id=row.node_id,
                    section_point=getattr(row, "section_point", None),
                )
            )
        elif association is FieldAssociation.INTEGRATION_POINT:
            locations.append(
                FieldLocation._from_finite_components(
                    association,
                    _triplet(
                        row.coordinates,
                        label="integration-point coordinates",
                    ),
                    _triplet(
                        row.displacement,
                        label="integration-point displacement",
                    ),
                    element_id=row.element_id,
                    integration_point=row.integration_point,
                    section_point=getattr(row, "section_point", None),
                )
            )
        elif association is FieldAssociation.NODE:
            locations.append(
                FieldLocation._from_finite_components(
                    association,
                    _triplet(
                        row.coordinates,
                        label="node coordinates",
                    ),
                    _triplet(
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
    value_matrix = _value_matrix(values, len(descriptor.columns))
    return FieldData._from_materialized_values(
        descriptor=descriptor,
        source=source,
        key=key,
        locations=tuple(locations),
        values=value_matrix,
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


def _state_position(position: FieldPosition) -> FiniteStrainStatePosition:
    mapping = {
        FieldPosition.INTEGRATION_POINT: (
            FiniteStrainStatePosition.INTEGRATION_POINT
        ),
        FieldPosition.CENTROID: FiniteStrainStatePosition.CENTROID,
        FieldPosition.ELEMENT_NODAL: FiniteStrainStatePosition.ELEMENT_NODAL,
    }
    try:
        return mapping[position]
    except KeyError as error:
        raise ValueError(
            f"unsupported finite-strain state position {position.value}"
        ) from error


@dataclass(frozen=True, slots=True)
class _TopologyLookup:
    topology: ResultTopologyProjection
    _node_order: dict[int, int] = field(init=False, repr=False)
    _element_order: dict[int, int] = field(init=False, repr=False)
    _nodal_displacements: np.ndarray = field(init=False, repr=False)
    _connectivity_indices: tuple[np.ndarray, ...] = field(
        init=False,
        repr=False,
    )
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
            self.topology._nodal_displacements,
        )
        regions_by_node: dict[int, set[ResultRegionKey]] = {}
        connectivity_indices: list[np.ndarray] = []
        for index, connected in enumerate(self.topology.connectivity):
            region_key = self.topology.element_region_keys[index]
            indices = np.fromiter(
                (node_order[node_id] for node_id in connected),
                dtype=np.int64,
                count=len(connected),
            )
            indices.setflags(write=False)
            connectivity_indices.append(indices)
            for node_id in connected:
                regions_by_node.setdefault(node_id, set()).add(region_key)
        object.__setattr__(
            self,
            "_connectivity_indices",
            tuple(connectivity_indices),
        )
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
        displacements = self._nodal_displacements[
            self._connectivity_indices[element_index]
        ]
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
    values = np.asarray(rows, dtype=float)
    if values.shape != (len(rows), columns):
        values = values.reshape((len(rows), columns))
    if not values.flags.owndata:
        values = values.copy()
    values.setflags(write=False)
    return values


__all__ = [
    "check_cancellation",
    "materialize_derived_fields",
]
