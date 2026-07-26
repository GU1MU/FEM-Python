from __future__ import annotations

import pytest

from fem.application.results import (
    FieldMaterializationKey,
    FieldPosition,
    FieldRequest,
    ResultFieldId,
    ResultProvider,
    ResultQuery,
    ResultQueryValidationError,
    ResultSourceKey,
    ResultVariable,
    advance_materialization,
    build_result_provider,
    evaluate_result_query,
)
from fem.post.averaging import NodalAveragingPolicy
from fem.post.fields import (
    ResultRegionKey,
    make_result_region_signature,
)
from tests.helpers.phase8_result_characterization import (
    make_continuum_nodal_semantics_result,
)


def _source() -> ResultSourceKey:
    return ResultSourceKey(
        result_id="result-query",
        session_id="session-query",
        artifact_id="artifact-query",
        model_revision=7,
        step_name="Step-1",
        run_id="run-query",
    )


def _provider() -> ResultProvider:
    return build_result_provider(
        _source(),
        make_continuum_nodal_semantics_result(),
    )


def _key(
    provider: ResultProvider,
    variable: ResultVariable,
    position: FieldPosition,
    *,
    policy: NodalAveragingPolicy | None = None,
) -> FieldMaterializationKey:
    return provider.resolve_request(
        FieldRequest(
            ResultFieldId(variable, position),
            averaging_policy=policy,
        )
    )


def _accepted_snapshot(
    provider: ResultProvider,
    *keys: FieldMaterializationKey,
):
    return advance_materialization(
        provider.snapshot,
        provider.materialize(keys),
    )


def _foreign_region() -> ResultRegionKey:
    return ResultRegionKey(
        make_result_region_signature({"material": "foreign"}),
        make_result_region_signature({"section": "foreign"}),
    )


def test_query_reads_all_or_filtered_rows_in_original_field_order() -> None:
    provider = _provider()
    key = _key(provider, ResultVariable.U, FieldPosition.NODE)
    field_data = provider.field(key)

    all_query = ResultQuery(key, "U1")
    all_result = evaluate_result_query(provider.snapshot, all_query)
    filtered_query = ResultQuery(key, "U1", node_ids=(8, 1))
    filtered = evaluate_result_query(provider.snapshot, filtered_query)

    assert tuple(record.location for record in all_result.records) == (
        field_data.locations
    )
    assert tuple(
        record.location.node_id for record in filtered.records
    ) == (1, 8)
    assert filtered.source == provider.snapshot.source
    assert filtered.materialization_generation == 0
    assert filtered.query is filtered_query
    assert provider.query(filtered_query) == filtered


def test_query_filters_are_conjunctive_and_preserve_multi_region_rows() -> None:
    provider = _provider()
    raw_key = _key(
        provider,
        ResultVariable.S,
        FieldPosition.RESOLVED_NODAL,
        policy=NodalAveragingPolicy(0.0),
    )
    region_key = _key(
        provider,
        ResultVariable.S,
        FieldPosition.NODE_REGION,
    )
    snapshot = _accepted_snapshot(provider, raw_key, region_key)
    region_field = next(
        field_data
        for field_data in snapshot.fields
        if field_data.key == region_key
    )

    all_node_regions = evaluate_result_query(
        snapshot,
        ResultQuery(region_key, "S11", node_ids=(1,)),
    )
    expected_node_regions = tuple(
        location
        for location in region_field.locations
        if location.node_id == 1
    )
    assert tuple(
        record.location for record in all_node_regions.records
    ) == expected_node_regions
    assert len(all_node_regions.records) == 2
    assert len(
        {
            record.location.region_key
            for record in all_node_regions.records
        }
    ) == 2

    target_location = next(
        location
        for field_data in snapshot.fields
        if field_data.key == raw_key
        for location in field_data.locations
        if location.node_id == 1 and location.element_id == 2
    )
    exact = evaluate_result_query(
        snapshot,
        ResultQuery(
            raw_key,
            "S11",
            node_ids=(1,),
            element_ids=(2,),
            region_keys=(target_location.region_key,),
        ),
    )

    assert tuple(record.location for record in exact.records) == (
        target_location,
    )
    assert exact.materialization_generation == 1
    assert exact.source == snapshot.source


def test_query_uses_the_complete_field_key_and_exact_component() -> None:
    provider = _provider()
    raw_key = _key(
        provider,
        ResultVariable.S,
        FieldPosition.RESOLVED_NODAL,
        policy=NodalAveragingPolicy(0.0),
    )
    averaged_key = _key(
        provider,
        ResultVariable.S,
        FieldPosition.RESOLVED_NODAL,
        policy=NodalAveragingPolicy(100.0),
    )
    snapshot = _accepted_snapshot(provider, raw_key, averaged_key)

    raw = evaluate_result_query(snapshot, ResultQuery(raw_key, "S11"))
    averaged = evaluate_result_query(
        snapshot,
        ResultQuery(averaged_key, "S11"),
    )

    assert len(raw.records) == 9
    assert len(averaged.records) == 8
    assert raw.query.field_key == raw_key
    assert averaged.query.field_key == averaged_key

    wrong_contract = FieldMaterializationKey(
        raw_key.request,
        raw_key.recovery_contract + 1,
    )
    with pytest.raises(ResultQueryValidationError) as missing:
        evaluate_result_query(
            snapshot,
            ResultQuery(wrong_contract, "S11"),
        )
    assert missing.value.code == "result.query.field_not_materialized"

    with pytest.raises(ResultQueryValidationError) as component:
        evaluate_result_query(snapshot, ResultQuery(raw_key, "s11"))
    assert component.value.code == "result.query.component_not_available"


@pytest.mark.parametrize(
    ("filters", "code"),
    (
        ({"node_ids": (999,)}, "result.query.unknown_node_ids"),
        ({"element_ids": (999,)}, "result.query.unknown_element_ids"),
        (
            {"region_keys": (_foreign_region(),)},
            "result.query.unknown_region_keys",
        ),
    ),
)
def test_query_rejects_unknown_topology_filters(
    filters: dict[str, tuple[object, ...]],
    code: str,
) -> None:
    provider = _provider()
    key = _key(provider, ResultVariable.U, FieldPosition.NODE)

    with pytest.raises(ResultQueryValidationError) as captured:
        evaluate_result_query(
            provider.snapshot,
            ResultQuery(key, "U1", **filters),
        )

    assert captured.value.code == code


def test_lazy_query_fails_without_materializing_or_falling_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _provider()
    lazy_key = _key(
        provider,
        ResultVariable.S,
        FieldPosition.CENTROID,
    )
    calls: list[tuple[object, ...]] = []

    def unexpected_materialize(
        self: ResultProvider,
        keys,
        *,
        cancellation=None,
    ):
        del self, cancellation
        calls.append(tuple(keys))
        raise AssertionError("query must not materialize a lazy field")

    monkeypatch.setattr(
        ResultProvider,
        "materialize",
        unexpected_materialize,
    )

    with pytest.raises(ResultQueryValidationError) as captured:
        provider.query(ResultQuery(lazy_key, "S11"))

    assert captured.value.code == "result.query.field_not_materialized"
    assert calls == []
