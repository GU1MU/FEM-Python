from __future__ import annotations

import pytest

from fem.post.averaging import NodalAveragingPolicy, resolve_nodal_stress
from fem.post.fields import ResultRegionKey, result_region_sort_key
from fem.post.stress.field import StressPosition, StressRecovery
from tests.helpers.phase8_result_characterization import (
    make_continuum_nodal_semantics_result,
)


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
