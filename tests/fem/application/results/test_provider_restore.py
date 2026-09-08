from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from fem.application.results import provider as provider_module
from fem.application.results.data import (
    FieldData,
    FieldState,
    ResultMaterializationPatch,
    ResultMaterializationSnapshot,
    ResultTopologyProjection,
    advance_materialization,
)
from fem.application.results.fields import (
    FieldMaterializationKey,
    FieldPosition,
    FieldRequest,
    ResultFieldId,
    ResultSourceKey,
    ResultVariable,
    field_materialization_sort_key,
)
from fem.application.results.provider import (
    build_result_provider,
    restore_result_provider,
)
from fem.core.mesh import Element2D, Mesh2D, Node2D
from fem.core.model import FEMModel
from fem.core.result import ModelResult
from fem.post.averaging import NodalAveragingPolicy


def _source(suffix: str = "1") -> ResultSourceKey:
    return ResultSourceKey(
        result_id=f"result-{suffix}",
        session_id="session-1",
        artifact_id="artifact-1",
        model_revision=2,
        step_name="Step-1",
        run_id=f"run-{suffix}",
    )


def _quad4_result() -> ModelResult:
    mesh = Mesh2D(
        nodes=[
            Node2D(1, 0.0, 0.0),
            Node2D(2, 1.0, 0.0),
            Node2D(3, 1.0, 1.0),
            Node2D(4, 0.0, 1.0),
        ],
        elements=[
            Element2D(
                10,
                [1, 2, 3, 4],
                "Quad4",
                {
                    "E": 100.0,
                    "nu": 0.25,
                    "plane_type": "stress",
                    "thickness": 1.0,
                },
            )
        ],
    )
    displacement = np.zeros(mesh.num_dofs)
    displacement[mesh.global_dof(2, 0)] = 0.1
    displacement[mesh.global_dof(3, 0)] = 0.1
    return ModelResult(
        FEMModel(mesh=mesh),
        None,
        displacement,
        np.zeros(mesh.num_dofs),
    )


def _key(
    provider,
    position: FieldPosition,
    *,
    policy: NodalAveragingPolicy | None = None,
    gauss_order: int | None = None,
) -> FieldMaterializationKey:
    return provider.resolve_request(
        FieldRequest(
            ResultFieldId(ResultVariable.S, position),
            averaging_policy=policy,
            gauss_order=gauss_order,
        )
    )


def _copy_field(
    field_data: FieldData,
    *,
    descriptor=None,
    source: ResultSourceKey | None = None,
    key: FieldMaterializationKey | None = None,
    locations=None,
    values: np.ndarray | None = None,
) -> FieldData:
    return FieldData(
        descriptor=(
            field_data.descriptor if descriptor is None else descriptor
        ),
        source=field_data.source if source is None else source,
        key=field_data.key if key is None else key,
        locations=field_data.locations if locations is None else locations,
        values=field_data.values if values is None else values,
    )


def _snapshot_with(
    original: ResultMaterializationSnapshot,
    *,
    topology: ResultTopologyProjection | None = None,
    fields: tuple[FieldData, ...] | None = None,
) -> ResultMaterializationSnapshot:
    selected = original.fields if fields is None else fields
    return ResultMaterializationSnapshot(
        source=original.source,
        generation=original.generation,
        topology=original.topology if topology is None else topology,
        fields=tuple(
            sorted(
                selected,
                key=lambda item: field_materialization_sort_key(item.key),
            )
        ),
    )


def _assert_same_field(actual: FieldData, expected: FieldData) -> None:
    assert actual.descriptor == expected.descriptor
    assert actual.source == expected.source
    assert actual.key == expected.key
    assert actual.locations == expected.locations
    assert np.array_equal(actual.values, expected.values)


def test_restore_preserves_accepted_keys_and_never_recovers_ready_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _quad4_result()
    base = build_result_provider(_source(), result)
    centroid = _key(base, FieldPosition.CENTROID)
    custom_ip = _key(
        base,
        FieldPosition.INTEGRATION_POINT,
        gauss_order=1,
    )
    custom_resolved = _key(
        base,
        FieldPosition.RESOLVED_NODAL,
        policy=NodalAveragingPolicy(25.0),
    )
    default_ip = _key(base, FieldPosition.INTEGRATION_POINT)
    default_resolved = _key(
        base,
        FieldPosition.RESOLVED_NODAL,
        policy=NodalAveragingPolicy(),
    )
    expected_lazy = base.materialize((default_ip,))
    accepted = advance_materialization(
        base.snapshot,
        base.materialize((custom_resolved, custom_ip, centroid)),
    )
    accepted_keys = tuple(field.key for field in accepted.fields)

    calls = []
    original_materializer = provider_module.materialize_derived_fields

    def counted_materializer(**kwargs):
        calls.append(kwargs["targets"])
        return original_materializer(**kwargs)

    monkeypatch.setattr(
        provider_module,
        "materialize_derived_fields",
        counted_materializer,
    )

    restored = restore_result_provider(result, accepted)

    assert calls == []
    assert restored.source == accepted.source
    assert restored.snapshot.generation == accepted.generation == 1
    assert restored.snapshot is not accepted
    assert restored.snapshot.topology is not accepted.topology
    assert restored._owned_result is not result
    assert restored._owned_result.model is not result.model
    assert not np.shares_memory(restored._owned_result.U, result.U)
    assert not np.shares_memory(
        restored._owned_result.reactions,
        result.reactions,
    )
    assert not np.shares_memory(
        restored.snapshot.topology._node_coordinates,
        accepted.topology._node_coordinates,
    )
    assert not np.shares_memory(
        restored.snapshot.topology._nodal_displacements,
        accepted.topology._nodal_displacements,
    )
    assert tuple(field.key for field in restored.snapshot.fields) == accepted_keys
    assert all(
        restored_field is not accepted_field
        for restored_field, accepted_field in zip(
            restored.snapshot.fields,
            accepted.fields,
            strict=True,
        )
    )
    assert all(
        not np.shares_memory(
            restored_field._values,
            accepted_field._values,
        )
        for restored_field, accepted_field in zip(
            restored.snapshot.fields,
            accepted.fields,
            strict=True,
        )
    )
    assert np.array_equal(
        restored.snapshot.topology.node_coordinates,
        accepted.topology.node_coordinates,
    )
    assert np.array_equal(
        restored.snapshot.topology.nodal_displacements,
        accepted.topology.nodal_displacements,
    )
    for restored_field, accepted_field in zip(
        restored.snapshot.fields,
        accepted.fields,
        strict=True,
    ):
        _assert_same_field(restored_field, accepted_field)

    expected_catalog_keys = tuple(
        sorted(
            {
                *(item.key for item in base.catalog().fields),
                *accepted_keys,
            },
            key=field_materialization_sort_key,
        )
    )
    assert tuple(item.key for item in restored.catalog().fields) == (
        expected_catalog_keys
    )
    assert restored.field_status(centroid).state is FieldState.READY
    assert restored.field_status(custom_ip).state is FieldState.READY
    assert restored.field_status(custom_resolved).state is FieldState.READY
    assert restored.field_status(default_ip).state is FieldState.LAZY
    assert restored.field_status(default_resolved).state is FieldState.LAZY

    ready_hit = restored.materialize(tuple(reversed(accepted_keys)))
    assert ready_hit.source == restored.source
    assert ready_hit.fields == ()
    assert calls == []

    restored_coordinates = restored.snapshot.topology.node_coordinates
    restored_displacements = restored.snapshot.topology.nodal_displacements
    restored_centroid_values = restored.field(centroid).values
    accepted.topology._node_coordinates.setflags(write=True)
    accepted.topology._node_coordinates[0, 0] = 321.0
    accepted.topology._nodal_displacements.setflags(write=True)
    accepted.topology._nodal_displacements[0, 0] = 654.0
    accepted_centroid = next(
        field for field in accepted.fields if field.key == centroid
    )
    accepted_centroid._values.setflags(write=True)
    accepted_centroid._values[0, 0] = 987.0
    result.U[:] = 5.0
    result.model.mesh.nodes[0].x = -20.0
    result.model.mesh.elements[0].props["E"] = 900.0

    assert np.array_equal(
        restored.snapshot.topology.node_coordinates,
        restored_coordinates,
    )
    assert np.array_equal(
        restored.snapshot.topology.nodal_displacements,
        restored_displacements,
    )
    assert np.array_equal(
        restored.field(centroid).values,
        restored_centroid_values,
    )

    actual_lazy = restored.materialize((default_ip,))
    assert len(calls) == 1
    assert len(actual_lazy.fields) == len(expected_lazy.fields) == 1
    _assert_same_field(actual_lazy.fields[0], expected_lazy.fields[0])


@pytest.mark.parametrize("member", ["coordinates", "connectivity"])
def test_restore_rejects_topology_that_differs_from_result_model(
    member: str,
) -> None:
    result = _quad4_result()
    base = build_result_provider(_source(), result)
    topology = base.snapshot.topology
    coordinates = topology.node_coordinates.copy()
    connectivity = topology.connectivity
    if member == "coordinates":
        coordinates[0, 0] += 0.25
    else:
        connectivity = ((1, 2, 4, 3),)
    tampered_topology = ResultTopologyProjection(
        source=topology.source,
        node_ids=topology.node_ids,
        node_coordinates=coordinates,
        nodal_displacements=topology.nodal_displacements,
        element_ids=topology.element_ids,
        element_types=topology.element_types,
        connectivity=connectivity,
        element_region_keys=topology.element_region_keys,
    )

    with pytest.raises(ValueError, match="topology"):
        restore_result_provider(
            result,
            _snapshot_with(base.snapshot, topology=tampered_topology),
        )


def test_restore_rejects_missing_primary_field() -> None:
    result = _quad4_result()
    base = build_result_provider(_source(), result)
    fields = tuple(
        field
        for field in base.snapshot.fields
        if field.key.request.field_id.variable is not ResultVariable.U
    )

    with pytest.raises(ValueError, match="primary field key"):
        restore_result_provider(
            result,
            _snapshot_with(base.snapshot, fields=fields),
        )


@pytest.mark.parametrize("member", ["values", "locations"])
def test_restore_rejects_tampered_primary_field(member: str) -> None:
    result = _quad4_result()
    base = build_result_provider(_source(), result)
    primary = next(
        field
        for field in base.snapshot.fields
        if field.key.request.field_id.variable is ResultVariable.U
    )
    values = primary.values.copy()
    locations = primary.locations
    if member == "values":
        values[0, 0] += 0.125
    else:
        first = replace(
            locations[0],
            coordinates=(0.125, 0.0, 0.0),
        )
        locations = (first, *locations[1:])
    replacement = _copy_field(
        primary,
        values=values,
        locations=locations,
    )
    fields = tuple(
        replacement if field.key == primary.key else field
        for field in base.snapshot.fields
    )

    with pytest.raises(ValueError, match="primary field"):
        restore_result_provider(
            result,
            _snapshot_with(base.snapshot, fields=fields),
        )


def test_restore_rejects_registry_descriptor_tampering() -> None:
    result = _quad4_result()
    base = build_result_provider(_source(), result)
    primary = base.snapshot.fields[0]
    replacement = _copy_field(
        primary,
        descriptor=replace(
            primary.descriptor,
            label_key="result.field.tampered",
        ),
    )
    fields = tuple(
        replacement if field.key == primary.key else field
        for field in base.snapshot.fields
    )

    with pytest.raises(ValueError, match="current registry"):
        restore_result_provider(
            result,
            _snapshot_with(base.snapshot, fields=fields),
        )


def test_restore_preserves_ready_historic_recovery_contracts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _quad4_result()
    base = build_result_provider(_source(), result)
    centroid = _key(base, FieldPosition.CENTROID)
    derived = base.materialize((centroid,)).fields[0]
    historic_key = replace(
        derived.key,
        recovery_contract=derived.key.recovery_contract + 1,
    )
    historic_values = derived.values.copy()
    historic_values[0, 0] += 7.0
    historic_field = _copy_field(
        derived,
        key=historic_key,
        values=historic_values,
    )
    snapshot = _snapshot_with(
        base.snapshot,
        fields=(*base.snapshot.fields, derived, historic_field),
    )
    custom_ip = _key(
        base,
        FieldPosition.INTEGRATION_POINT,
        gauss_order=1,
    )
    current_ip_field = base.materialize((custom_ip,)).fields[0]
    historic_ip_field = _copy_field(
        current_ip_field,
        key=replace(
            current_ip_field.key,
            recovery_contract=current_ip_field.key.recovery_contract + 1,
        ),
    )
    forbidden_calls = []

    def forbidden_materializer(**kwargs):
        forbidden_calls.append(kwargs)
        raise AssertionError("READY historic fields must remain cache hits")

    monkeypatch.setattr(
        provider_module,
        "materialize_derived_fields",
        forbidden_materializer,
    )

    restored = restore_result_provider(result, snapshot)

    assert restored.resolve_request(derived.key.request) == derived.key
    assert restored.field_status(derived.key).state is FieldState.READY
    assert restored.field_status(historic_key).state is FieldState.READY
    _assert_same_field(restored.field(derived.key), derived)
    _assert_same_field(restored.field(historic_key), historic_field)
    cache_hit = restored.materialize((historic_key, derived.key))
    assert cache_hit.fields == ()
    assert forbidden_calls == []

    absent_contract = replace(
        historic_key,
        recovery_contract=historic_key.recovery_contract + 1,
    )
    with pytest.raises(KeyError):
        restored.field_status(absent_contract)
    with pytest.raises(KeyError):
        restored.field(absent_contract)
    with pytest.raises(KeyError):
        restored.materialize((absent_contract,))
    assert forbidden_calls == []

    with pytest.raises(KeyError):
        restored.apply(
            ResultMaterializationPatch(
                source=restored.source,
                fields=(historic_ip_field,),
            )
        )


def test_restore_rejects_unknown_contextual_field() -> None:
    result = _quad4_result()
    base = build_result_provider(_source(), result)
    centroid = _key(base, FieldPosition.CENTROID)
    derived = base.materialize((centroid,)).fields[0]
    unknown_id = ResultFieldId(
        ResultVariable.LE,
        FieldPosition.CENTROID,
    )
    unknown_key = FieldMaterializationKey(
        replace(derived.key.request, field_id=unknown_id),
        recovery_contract=derived.key.recovery_contract,
    )
    unknown_field = _copy_field(
        derived,
        descriptor=replace(derived.descriptor, field_id=unknown_id),
        key=unknown_key,
    )
    snapshot = _snapshot_with(
        base.snapshot,
        fields=(*base.snapshot.fields, unknown_field),
    )

    with pytest.raises(KeyError):
        restore_result_provider(result, snapshot)


def test_restore_revalidates_field_source_ownership() -> None:
    result = _quad4_result()
    base = build_result_provider(_source(), result)
    copied_fields = tuple(_copy_field(field) for field in base.snapshot.fields)
    snapshot = _snapshot_with(base.snapshot, fields=copied_fields)
    object.__setattr__(snapshot.fields[0], "source", _source("foreign"))

    with pytest.raises(ValueError, match="field source"):
        restore_result_provider(result, snapshot)


def test_restore_validates_argument_types() -> None:
    result = _quad4_result()
    snapshot = build_result_provider(_source(), result).snapshot

    with pytest.raises(TypeError, match="result must"):
        restore_result_provider(object(), snapshot)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="materialization must"):
        restore_result_provider(result, object())  # type: ignore[arg-type]
