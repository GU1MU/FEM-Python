from __future__ import annotations

from dataclasses import replace

import pytest

from fem.application.results import (
    FieldAssociation,
    FieldAvailability,
    FieldMaterializationKey,
    FieldPosition,
    FieldRequest,
    FieldState,
    ResultCatalog,
    ResultDiagnostic,
    ResultFieldId,
    ResultProvider,
    ResultQueryValidationError,
    ResultSourceKey,
    ResultVariable,
    advance_materialization,
    build_result_provider,
    field_materialization_sort_key,
    restore_result_provider,
)
from fem.application.results.inspection import (
    ElementResultInspectionRequest,
    NodeResultInspectionRequest,
    inspect_result_snapshot,
)
from fem.post.averaging import NodalAveragingPolicy
from tests.helpers.phase8_result_characterization import (
    make_continuum_nodal_semantics_result,
)


_NODE_ASSOCIATIONS = frozenset(
    {
        FieldAssociation.NODE,
        FieldAssociation.NODE_REGION,
        FieldAssociation.RESOLVED_NODAL,
        FieldAssociation.ELEMENT_NODE,
    }
)
_ELEMENT_ASSOCIATIONS = frozenset(
    {
        FieldAssociation.ELEMENT,
        FieldAssociation.INTEGRATION_POINT,
        FieldAssociation.ELEMENT_NODE,
    }
)


def _source() -> ResultSourceKey:
    return ResultSourceKey(
        result_id="result-inspection",
        session_id="session-inspection",
        artifact_id="artifact-inspection",
        model_revision=4,
        step_name="Step-1",
        run_id="run-inspection",
    )


def _key(
    provider: ResultProvider,
    position: FieldPosition,
) -> FieldMaterializationKey:
    return provider.resolve_request(
        FieldRequest(
            ResultFieldId(ResultVariable.S, position),
            averaging_policy=(
                NodalAveragingPolicy()
                if position is FieldPosition.RESOLVED_NODAL
                else None
            ),
        )
    )


def _base_provider() -> ResultProvider:
    return build_result_provider(
        _source(),
        make_continuum_nodal_semantics_result(),
    )


def _fully_materialized_provider() -> ResultProvider:
    result = make_continuum_nodal_semantics_result()
    provider = build_result_provider(_source(), result)
    keys = tuple(
        _key(provider, position)
        for position in (
            FieldPosition.INTEGRATION_POINT,
            FieldPosition.CENTROID,
            FieldPosition.ELEMENT_NODAL,
            FieldPosition.NODE_REGION,
            FieldPosition.RESOLVED_NODAL,
        )
    )
    snapshot = advance_materialization(
        provider.snapshot,
        provider.materialize(keys),
    )
    return restore_result_provider(result, snapshot)


def _expected_locations(
    provider: ResultProvider,
    availability: FieldAvailability,
    *,
    node_id: int | None = None,
    element_id: int | None = None,
):
    return tuple(
        location
        for location in provider.field(availability.key).locations
        if (
            (node_id is None or location.node_id == node_id)
            and (
                element_id is None
                or location.element_id == element_id
            )
        )
    )


def test_node_inspection_follows_catalog_component_and_location_order() -> None:
    provider = _fully_materialized_provider()
    request = NodeResultInspectionRequest(1)

    result = provider.inspect_result(request)
    expected_availability = tuple(
        availability
        for availability in provider.catalog().fields
        if availability.descriptor.association in _NODE_ASSOCIATIONS
    )

    assert result.source == provider.source
    assert result.materialization_generation == 1
    assert result.request is request
    assert tuple(
        field_entry.availability for field_entry in result.fields
    ) == expected_availability
    for field_entry in result.fields:
        availability = field_entry.availability
        expected_locations = _expected_locations(
            provider,
            availability,
            node_id=1,
        )
        assert tuple(
            query_result.query.component
            for query_result in field_entry.component_results
        ) == availability.descriptor.columns
        for query_result in field_entry.component_results:
            assert query_result.query.field_key == availability.key
            assert tuple(
                record.location for record in query_result.records
            ) == expected_locations

    node_region = next(
        field_entry
        for field_entry in result.fields
        if (
            field_entry.availability.descriptor.association
            is FieldAssociation.NODE_REGION
        )
    )
    node_region_rows = node_region.component_results[0].records
    assert len(node_region_rows) == 2
    assert len(
        {record.location.region_key for record in node_region_rows}
    ) == 2

    element_node = next(
        field_entry
        for field_entry in result.fields
        if (
            field_entry.availability.descriptor.association
            is FieldAssociation.ELEMENT_NODE
        )
    )
    assert tuple(
        record.location.element_id
        for record in element_node.component_results[0].records
    ) == (1, 2, 3)


def test_element_inspection_includes_only_element_associated_rows() -> None:
    provider = _fully_materialized_provider()
    request = ElementResultInspectionRequest(2)

    result = inspect_result_snapshot(
        provider.snapshot,
        provider.catalog(),
        request,
    )
    expected_availability = tuple(
        availability
        for availability in provider.catalog().fields
        if availability.descriptor.association in _ELEMENT_ASSOCIATIONS
    )

    assert tuple(
        field_entry.availability for field_entry in result.fields
    ) == expected_availability
    for field_entry in result.fields:
        availability = field_entry.availability
        expected_locations = _expected_locations(
            provider,
            availability,
            element_id=2,
        )
        for query_result in field_entry.component_results:
            assert tuple(
                record.location for record in query_result.records
            ) == expected_locations
            assert all(
                record.location.element_id == 2
                for record in query_result.records
            )


def test_lazy_and_unavailable_fields_remain_explicit_without_recovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _base_provider()
    centroid = next(
        availability
        for availability in provider.catalog().fields
        if (
            availability.descriptor.field_id.position
            is FieldPosition.CENTROID
        )
    )
    diagnostic = ResultDiagnostic(
        code="result.field.unavailable",
        severity="warning",
        message="Field is unavailable for inspection.",
        path=("results",),
        remediation="Choose another field.",
        details={"position": "centroid"},
    )
    unavailable = FieldAvailability(
        key=FieldMaterializationKey(
            centroid.key.request,
            centroid.key.recovery_contract + 100,
        ),
        descriptor=centroid.descriptor,
        state=FieldState.UNAVAILABLE,
        diagnostics=(diagnostic,),
    )
    catalog = ResultCatalog(
        source=provider.source,
        fields=tuple(
            sorted(
                provider.catalog().fields + (unavailable,),
                key=lambda item: field_materialization_sort_key(item.key),
            )
        ),
        default_selection=provider.catalog().default_selection,
        diagnostics=provider.catalog().diagnostics,
    )
    provider = replace(provider, _catalog=catalog)
    calls: list[tuple[object, ...]] = []

    def unexpected_materialize(
        self: ResultProvider,
        keys,
        *,
        cancellation=None,
    ):
        del self, cancellation
        calls.append(tuple(keys))
        raise AssertionError("inspection must not materialize fields")

    monkeypatch.setattr(
        ResultProvider,
        "materialize",
        unexpected_materialize,
    )

    result = provider.inspect_result(ElementResultInspectionRequest(1))
    node_result = provider.inspect_result(NodeResultInspectionRequest(1))
    states = tuple(
        field_entry.availability.state for field_entry in result.fields
    )
    node_states = tuple(
        field_entry.availability.state for field_entry in node_result.fields
    )

    assert FieldState.LAZY in states
    assert FieldState.UNAVAILABLE in states
    assert all(
        field_entry.component_results == ()
        for field_entry in result.fields
    )
    unavailable_entry = next(
        field_entry
        for field_entry in result.fields
        if field_entry.availability.state is FieldState.UNAVAILABLE
    )
    assert unavailable_entry.availability.diagnostics == (diagnostic,)
    assert FieldState.READY in node_states
    assert FieldState.LAZY in node_states
    assert all(
        bool(field_entry.component_results)
        == (field_entry.availability.state is FieldState.READY)
        for field_entry in node_result.fields
    )
    assert calls == []


@pytest.mark.parametrize(
    ("inspection_request", "code"),
    (
        (
            NodeResultInspectionRequest(999),
            "result.query.unknown_node_ids",
        ),
        (
            ElementResultInspectionRequest(999),
            "result.query.unknown_element_ids",
        ),
    ),
)
def test_inspection_rejects_unknown_topology_entities(
    inspection_request: (
        NodeResultInspectionRequest | ElementResultInspectionRequest
    ),
    code: str,
) -> None:
    provider = _base_provider()

    with pytest.raises(ResultQueryValidationError) as captured:
        provider.inspect_result(inspection_request)

    assert captured.value.code == code
