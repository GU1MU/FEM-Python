from __future__ import annotations

import math

import numpy as np
import pytest

from fem.post.averaging import NodalAveragingPolicy, resolve_nodal_stress
from fem.post.fields import (
    ResultRegionKey,
    make_result_region_signature,
)
from fem.post.stress.field import (
    CANONICAL_PLANE_COMPONENT_NAMES,
    StressField,
    StressPosition,
    StressRecord,
    StressRegionKey,
)
from fem.post.stress.invariants import derive_stress_invariants


def _region(name: str) -> ResultRegionKey:
    return ResultRegionKey(
        make_result_region_signature({"material": name}),
        make_result_region_signature({"section": "plane"}),
    )


def _record(
    *,
    node_id: int,
    elem_id: int,
    local_node: int,
    components: tuple[float, float, float, float],
    region_key: ResultRegionKey,
    weight: float = 1.0,
    coordinate_tag: float | None = None,
) -> StressRecord:
    coordinates = (
        float(node_id if coordinate_tag is None else coordinate_tag),
        0.0,
    )
    displacement = (
        float(node_id) / 100.0,
        -float(node_id) / 100.0,
    )
    return StressRecord(
        position=StressPosition.ELEMENT_NODAL,
        coordinates=coordinates,
        components=components,
        invariants=derive_stress_invariants(
            components,
            CANONICAL_PLANE_COMPONENT_NAMES,
        ),
        elem_id=elem_id,
        node_id=node_id,
        local_node=local_node,
        region_key=region_key,
        weight=weight,
        displacement=displacement,
    )


def _field(*records: StressRecord) -> StressField:
    return StressField(
        StressPosition.ELEMENT_NODAL,
        CANONICAL_PLANE_COMPONENT_NAMES,
        tuple(records),
    )


@pytest.mark.parametrize("value", (True, False, "75", None, object()))
def test_policy_rejects_non_numeric_thresholds_and_bool(value: object) -> None:
    with pytest.raises(TypeError, match="threshold_percent"):
        NodalAveragingPolicy(value)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "value",
    (-0.01, 100.01, math.nan, math.inf, -math.inf),
)
def test_policy_rejects_nonfinite_or_out_of_range_threshold(
    value: float,
) -> None:
    with pytest.raises(ValueError, match="threshold_percent"):
        NodalAveragingPolicy(value)


def test_policy_requires_preserved_region_boundaries() -> None:
    with pytest.raises(ValueError, match="must be True"):
        NodalAveragingPolicy(preserve_region_boundaries=False)
    with pytest.raises(TypeError, match="must be bool"):
        NodalAveragingPolicy(preserve_region_boundaries=1)  # type: ignore[arg-type]


def test_resolver_uses_mesh_and_region_order_and_omits_isolated_nodes() -> None:
    region_a = _region("a")
    region_z = _region("z")
    field = _field(
        _record(
            node_id=10,
            elem_id=100,
            local_node=2,
            components=(30.0, 0.0, 0.0, 0.0),
            region_key=region_z,
            weight=3.0,
            coordinate_tag=100.0,
        ),
        _record(
            node_id=10,
            elem_id=300,
            local_node=1,
            components=(50.0, 0.0, 0.0, 0.0),
            region_key=region_a,
        ),
        _record(
            node_id=30,
            elem_id=200,
            local_node=1,
            components=(0.0, 0.0, 0.0, 0.0),
            region_key=region_z,
        ),
        _record(
            node_id=10,
            elem_id=200,
            local_node=3,
            components=(10.0, 0.0, 0.0, 0.0),
            region_key=region_z,
            weight=1.0,
            coordinate_tag=200.0,
        ),
    )

    resolved = resolve_nodal_stress(
        field,
        NodalAveragingPolicy(100.0),
        node_ids=(30, 10, 99),
        element_ids=(200, 100, 300),
    )

    assert [
        (
            record.node_id,
            record.region_key,
            record.elem_id,
            record.local_node,
            record.averaged,
        )
        for record in resolved.records
    ] == [
        (30, region_z, 200, 1, False),
        (10, region_a, 300, 1, False),
        (10, region_z, None, None, True),
    ]
    averaged = resolved.records[-1]
    assert averaged.components == pytest.approx((25.0, 0.0, 0.0, 0.0))
    assert averaged.coordinates == (200.0, 0.0)
    assert averaged.displacement == pytest.approx((0.1, -0.1))
    assert averaged.weight == pytest.approx(4.0)
    assert averaged.invariants == derive_stress_invariants(
        averaged.components,
        CANONICAL_PLANE_COMPONENT_NAMES,
    )
    assert all(record.node_id != 99 for record in resolved.records)


def test_threshold_is_inclusive_and_zero_always_keeps_multiple_raw_samples() -> None:
    region = _region("only")
    field = _field(
        _record(
            node_id=1,
            elem_id=2,
            local_node=1,
            components=(10.0, 0.0, 0.0, 0.0),
            region_key=region,
        ),
        _record(
            node_id=1,
            elem_id=1,
            local_node=2,
            components=(30.0, 0.0, 0.0, 0.0),
            region_key=region,
        ),
        _record(
            node_id=2,
            elem_id=2,
            local_node=2,
            components=(0.0, 0.0, 0.0, 0.0),
            region_key=region,
        ),
        _record(
            node_id=3,
            elem_id=1,
            local_node=1,
            components=(40.0, 0.0, 0.0, 0.0),
            region_key=region,
        ),
    )

    exact = resolve_nodal_stress(
        field,
        NodalAveragingPolicy(50.0),
        node_ids=(1, 2, 3),
        element_ids=(2, 1),
    )
    below = resolve_nodal_stress(
        field,
        NodalAveragingPolicy(49.999),
        node_ids=(1, 2, 3),
        element_ids=(2, 1),
    )
    zero = resolve_nodal_stress(
        field,
        NodalAveragingPolicy(0.0),
        node_ids=(1, 2, 3),
        element_ids=(2, 1),
    )

    assert [
        record.averaged
        for record in exact.records
        if record.node_id == 1
    ] == [True]
    assert [
        (record.elem_id, record.local_node, record.averaged)
        for record in below.records
        if record.node_id == 1
    ] == [(2, 1, False), (1, 2, False)]
    assert [
        (record.elem_id, record.local_node, record.averaged)
        for record in zero.records
        if record.node_id == 1
    ] == [(2, 1, False), (1, 2, False)]


def test_plane_s33_participates_in_the_single_tensor_decision() -> None:
    region = _region("plane")
    field = _field(
        _record(
            node_id=1,
            elem_id=1,
            local_node=1,
            components=(5.0, 2.0, 0.0, 1.0),
            region_key=region,
        ),
        _record(
            node_id=1,
            elem_id=2,
            local_node=1,
            components=(5.0, 2.0, 10.0, 1.0),
            region_key=region,
        ),
        _record(
            node_id=2,
            elem_id=1,
            local_node=2,
            components=(5.0, 2.0, -10.0, 1.0),
            region_key=region,
        ),
        _record(
            node_id=3,
            elem_id=2,
            local_node=2,
            components=(5.0, 2.0, 30.0, 1.0),
            region_key=region,
        ),
    )

    resolved = resolve_nodal_stress(
        field,
        NodalAveragingPolicy(20.0),
        node_ids=(1, 2, 3),
        element_ids=(1, 2),
    )

    assert [
        (record.elem_id, record.averaged)
        for record in resolved.records
        if record.node_id == 1
    ] == [(1, False), (2, False)]


def test_region_scale_tolerance_handles_roundoff_sized_range() -> None:
    region = _region("roundoff")
    epsilon = np.finfo(float).eps
    field = _field(
        _record(
            node_id=1,
            elem_id=1,
            local_node=1,
            components=(1.0, 0.0, 0.0, 0.0),
            region_key=region,
        ),
        _record(
            node_id=1,
            elem_id=2,
            local_node=1,
            components=(1.0 + 16.0 * epsilon, 0.0, 0.0, 0.0),
            region_key=region,
        ),
    )

    resolved = resolve_nodal_stress(
        field,
        NodalAveragingPolicy(1.0),
        node_ids=(1,),
        element_ids=(1, 2),
    )

    assert len(resolved.records) == 1
    assert resolved.records[0].averaged


@pytest.mark.parametrize("weight", (0.0, -1.0, math.nan, math.inf))
def test_resolver_rejects_nonpositive_or_nonfinite_weights(
    weight: float,
) -> None:
    region = _region("weight")
    field = _field(
        _record(
            node_id=1,
            elem_id=1,
            local_node=1,
            components=(1.0, 0.0, 0.0, 0.0),
            region_key=region,
            weight=weight,
        )
    )

    with pytest.raises(ValueError, match="positive and finite"):
        resolve_nodal_stress(
            field,
            node_ids=(1,),
            element_ids=(1,),
        )


def test_legacy_region_factory_returns_exact_result_region_identity() -> None:
    region = StressRegionKey(
        ("material", "steel"),
        ("section", (("thickness", 1.0),)),
    )

    assert type(region) is ResultRegionKey
