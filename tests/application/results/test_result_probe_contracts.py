from __future__ import annotations

import pytest

from fem.results import (
    FieldAssociation,
    FieldLocation,
    FieldMaterializationKey,
    FieldPosition,
    FieldRequest,
    ResultFieldId,
    ResultFrameKey,
    ResultProbeKind,
    ResultProbeRequest,
    ResultProbeTarget,
    ResultProbeValidationError,
    ResultQuery,
    ResultQueryRecord,
    ResultQueryResult,
    ResultSourceKey,
    ResultVariable,
    ScalarFieldSelection,
    probe_result_from_query_result,
    result_query_for_probe,
)
def _source() -> ResultSourceKey:
    return ResultSourceKey(
        result_id="probe-result",
        session_id="probe-session",
        artifact_id="probe-artifact",
        model_revision=1,
        step_name="Step-1",
        run_id="probe-run",
    )


def _selection() -> ScalarFieldSelection:
    key = FieldMaterializationKey(
        FieldRequest(
            ResultFieldId(
                ResultVariable.S,
                FieldPosition.INTEGRATION_POINT,
            ),
        ),
        recovery_contract=1,
    )
    return ScalarFieldSelection(key, "Mises")


def _location(
    association: FieldAssociation,
    *,
    element_id: int | None = None,
    node_id: int | None = None,
    integration_point: int | None = None,
    local_node: int | None = None,
) -> FieldLocation:
    return FieldLocation(
        association=association,
        coordinates=(float(element_id or node_id or 0), 0.0, 0.0),
        displacement=(0.0, 0.0, 0.0),
        element_id=element_id,
        node_id=node_id,
        integration_point=integration_point,
        local_node=local_node,
    )


def _query_result(
    query: ResultQuery,
    records: tuple[ResultQueryRecord, ...],
) -> ResultQueryResult:
    return ResultQueryResult(
        source=_source(),
        materialization_generation=4,
        query=query,
        records=records,
    )


def test_probe_target_requires_exact_identity_for_each_kind() -> None:
    assert ResultProbeTarget(ResultProbeKind.NODE, node_id=3).node_id == 3
    assert ResultProbeTarget(
        ResultProbeKind.INTEGRATION_POINT,
        element_id=8,
        integration_point=2,
    ).integration_point == 2

    with pytest.raises(ValueError):
        ResultProbeTarget(ResultProbeKind.NODE, node_id=3, element_id=8)
    with pytest.raises(ValueError):
        ResultProbeTarget(ResultProbeKind.ELEMENT, element_id=0)
    with pytest.raises(ValueError):
        ResultProbeTarget(
            ResultProbeKind.ELEMENT_NODE,
            element_id=8,
            local_node=0,
        )


def test_probe_translates_node_and_element_targets_without_averaging() -> None:
    node_request = ResultProbeRequest(
        _selection(),
        ResultProbeTarget(ResultProbeKind.NODE, node_id=3),
    )
    node_query = result_query_for_probe(node_request)
    assert node_query.node_ids == (3,)
    assert node_query.element_ids == ()

    element_request = ResultProbeRequest(
        _selection(),
        ResultProbeTarget(ResultProbeKind.ELEMENT, element_id=8),
    )
    element_query = result_query_for_probe(element_request)
    assert element_query.node_ids == ()
    assert element_query.element_ids == (8,)


def test_probe_filters_one_integration_point_and_keeps_frame_identity() -> None:
    request = ResultProbeRequest(
        _selection(),
        ResultProbeTarget(
            ResultProbeKind.INTEGRATION_POINT,
            element_id=8,
            integration_point=2,
        ),
    )
    query = result_query_for_probe(request)
    source = _source()
    result = _query_result(
        query,
        (
            ResultQueryRecord(
                source,
                _location(
                    FieldAssociation.INTEGRATION_POINT,
                    element_id=8,
                    integration_point=1,
                ),
                10.0,
            ),
            ResultQueryRecord(
                source,
                _location(
                    FieldAssociation.INTEGRATION_POINT,
                    element_id=8,
                    integration_point=2,
                ),
                20.0,
            ),
            ResultQueryRecord(
                source,
                _location(
                    FieldAssociation.INTEGRATION_POINT,
                    element_id=9,
                    integration_point=2,
                ),
                30.0,
            ),
        ),
    )

    probed = probe_result_from_query_result(
        result,
        request,
        frame_key=ResultFrameKey(source, 3),
    )

    assert probed.frame_key == ResultFrameKey(source, 3)
    assert tuple(record.value for record in probed.records) == (20.0,)
    assert probed.records[0].location.integration_point == 2


def test_element_node_probe_keeps_raw_element_side_record() -> None:
    selection = _selection()
    request = ResultProbeRequest(
        selection,
        ResultProbeTarget(
            ResultProbeKind.ELEMENT_NODE,
            element_id=10,
            local_node=2,
        ),
    )
    query = result_query_for_probe(request)
    source = _source()
    result = _query_result(
        query,
        (
            ResultQueryRecord(
                source,
                _location(
                    FieldAssociation.ELEMENT_NODE,
                    element_id=10,
                    local_node=1,
                    node_id=4,
                ),
                1.0,
            ),
            ResultQueryRecord(
                source,
                _location(
                    FieldAssociation.ELEMENT_NODE,
                    element_id=10,
                    local_node=2,
                    node_id=5,
                ),
                2.0,
            ),
        ),
    )

    probed = probe_result_from_query_result(result, request)
    assert len(probed.records) == 1
    assert probed.records[0].location.local_node == 2
    assert probed.records[0].location.node_id == 5


def test_probe_rejects_a_query_for_a_different_target() -> None:
    request = ResultProbeRequest(
        _selection(),
        ResultProbeTarget(ResultProbeKind.NODE, node_id=3),
    )
    foreign_query = ResultQuery(
        request.selection.field_key,
        request.selection.component,
        node_ids=(4,),
    )

    with pytest.raises(ResultProbeValidationError) as captured:
        probe_result_from_query_result(
            _query_result(foreign_query, ()),
            request,
        )
    assert captured.value.code == "result.probe.query_mismatch"
