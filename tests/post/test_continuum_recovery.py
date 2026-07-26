from __future__ import annotations

import pytest

from fem.post import averaging as averaging_module
from fem.post.averaging import NodalAveragingPolicy, resolve_nodal_stress
from fem.post.fields import ResultRegionKey, result_region_sort_key
from fem.post.stress import field as field_module
from fem.post.stress.field import StressPosition, StressRecovery
from tests.helpers.phase8_result_characterization import (
    make_continuum_nodal_semantics_result,
)


class _RecoveryCancelled(RuntimeError):
    pass


@pytest.mark.parametrize(
    ("threshold", "expected"),
    (
        (0.0, [(10.0, 1, False), (30.0, 2, False), (50.0, 3, False)]),
        (75.0, [(10.0, 1, False), (30.0, 2, False), (50.0, 3, False)]),
        (100.0, [(20.0, None, True), (50.0, 3, False)]),
    ),
)
def test_canonical_continuum_resolution_unifies_threshold_oracle(
    threshold: float,
    expected: list[tuple[float, int | None, bool]],
) -> None:
    result = make_continuum_nodal_semantics_result()
    mesh = result.model.mesh
    element_nodal = StressRecovery(mesh, result.U).collect(
        StressPosition.ELEMENT_NODAL
    )

    resolved = resolve_nodal_stress(
        element_nodal,
        NodalAveragingPolicy(threshold),
        node_ids=mesh.node_ids,
        element_ids=tuple(element.id for element in mesh.elements),
    )
    center = [
        (record.components[0], record.elem_id, bool(record.averaged))
        for record in resolved.records
        if record.node_id == 1
    ]

    assert center == pytest.approx(expected)
    assert all(type(record.region_key) is ResultRegionKey for record in resolved.records)
    assert all(record.node_id != 8 for record in resolved.records)


def test_element_nodal_records_carry_sample_displacement_and_exact_region() -> None:
    result = make_continuum_nodal_semantics_result()
    mesh = result.model.mesh
    recovery = StressRecovery(mesh, result.U)
    element_nodal = recovery.collect(StressPosition.ELEMENT_NODAL)

    node_two = next(
        record for record in element_nodal.records if record.node_id == 2
    )
    assert node_two.displacement == pytest.approx((0.1, 0.0))
    assert type(node_two.region_key) is ResultRegionKey

    node_region = recovery.collect(StressPosition.NODAL)
    regions_by_node: dict[int, list[ResultRegionKey]] = {}
    for record in node_region.records:
        regions_by_node.setdefault(int(record.node_id), []).append(
            record.region_key
        )
        assert record.averaged is None
    assert regions_by_node[1] == sorted(
        regions_by_node[1],
        key=result_region_sort_key,
    )


@pytest.mark.parametrize(
    "position",
    (
        StressPosition.INTEGRATION_POINT,
        StressPosition.CENTROID,
        StressPosition.ELEMENT_NODAL,
        StressPosition.NODAL,
    ),
)
def test_position_transform_cancellation_never_caches_partial_field(
    position: StressPosition,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = make_continuum_nodal_semantics_result()
    recovery = StressRecovery(result.model.mesh, result.U)
    if position is StressPosition.NODAL:
        recovery.collect(StressPosition.ELEMENT_NODAL)

    original_make_record = field_module._make_record
    completed_records = 0
    cancellation_enabled = True

    def counted_make_record(*args, **kwargs):
        nonlocal completed_records
        record = original_make_record(*args, **kwargs)
        completed_records += 1
        return record

    def checkpoint() -> None:
        if cancellation_enabled and completed_records:
            raise _RecoveryCancelled("cancelled during position transform")

    monkeypatch.setattr(
        field_module,
        "_make_record",
        counted_make_record,
    )

    with pytest.raises(_RecoveryCancelled, match="position transform"):
        recovery.collect(position, checkpoint=checkpoint)

    assert completed_records >= 1
    assert position not in recovery._cache
    cancellation_enabled = False
    retried = recovery.collect(position, checkpoint=checkpoint)
    clean = StressRecovery(
        result.model.mesh,
        result.U,
    ).collect(position)
    assert retried == clean


def test_position_transform_final_checkpoint_precedes_cache_install(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = make_continuum_nodal_semantics_result()
    expected = StressRecovery(result.model.mesh, result.U).collect(
        StressPosition.CENTROID
    )
    recovery = StressRecovery(result.model.mesh, result.U)
    original_centroid_records = field_module._centroid_records
    transform_finished = False
    cancellation_enabled = True

    def marked_centroid_records(*args, **kwargs):
        nonlocal transform_finished
        records = original_centroid_records(*args[:-1], None, **kwargs)
        transform_finished = True
        return records

    def checkpoint() -> None:
        if cancellation_enabled and transform_finished:
            raise _RecoveryCancelled("cancelled at final checkpoint")

    monkeypatch.setattr(
        field_module,
        "_centroid_records",
        marked_centroid_records,
    )

    with pytest.raises(_RecoveryCancelled, match="final checkpoint"):
        recovery.collect(StressPosition.CENTROID, checkpoint=checkpoint)

    assert StressPosition.CENTROID not in recovery._cache
    cancellation_enabled = False
    assert recovery.collect(
        StressPosition.CENTROID,
        checkpoint=checkpoint,
    ) == expected


def test_resolved_nodal_cancellation_stops_between_input_records_and_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = make_continuum_nodal_semantics_result()
    mesh = result.model.mesh
    element_nodal = StressRecovery(mesh, result.U).collect(
        StressPosition.ELEMENT_NODAL
    )
    original_weight = averaging_module._positive_finite_weight
    checked_records = 0

    def counted_weight(value):
        nonlocal checked_records
        checked_records += 1
        return original_weight(value)

    def checkpoint() -> None:
        if checked_records:
            raise _RecoveryCancelled("cancelled during nodal resolution")

    monkeypatch.setattr(
        averaging_module,
        "_positive_finite_weight",
        counted_weight,
    )

    with pytest.raises(_RecoveryCancelled, match="nodal resolution"):
        resolve_nodal_stress(
            element_nodal,
            node_ids=mesh.node_ids,
            element_ids=tuple(
                element.id for element in mesh.elements
            ),
            checkpoint=checkpoint,
        )

    assert checked_records == 1
    retried = resolve_nodal_stress(
        element_nodal,
        node_ids=mesh.node_ids,
        element_ids=tuple(element.id for element in mesh.elements),
    )
    clean = resolve_nodal_stress(
        element_nodal,
        node_ids=mesh.node_ids,
        element_ids=tuple(element.id for element in mesh.elements),
    )
    assert retried == clean
