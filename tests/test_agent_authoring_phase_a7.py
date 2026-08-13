from __future__ import annotations

import json

import pytest

from fem_agent.result_authoring import (
    AcceptedResultSource,
    AgentResultAggregation,
    AgentResultCatalog,
    AgentResultCatalogResponse,
    AgentResultField,
    AgentResultLocation,
    AgentResultQuery,
    AgentResultQueryBridge,
    AgentResultQueryResponse,
    AgentResultScalar,
    AgentResultVariable,
    FakeAgentResultQueryPort,
    ResultAuthoringError,
    explain_result_response,
    result_catalog_tool_schema,
    result_query_tool_schema,
)


def _source() -> AcceptedResultSource:
    return AcceptedResultSource(
        result_id="result-a7",
        session_id="session-a7",
        artifact_id="artifact-a7",
        model_revision=7,
        step_name="分析步-静力",
        run_id="run-a7",
    )


def _query(
    *,
    variable: AgentResultVariable = AgentResultVariable.DISPLACEMENT,
    component: str = "Magnitude",
    position: str = "node",
    region: str = "all_nodes",
    aggregation: AgentResultAggregation = AgentResultAggregation.MAXIMUM,
) -> AgentResultQuery:
    return AgentResultQuery(
        variable=variable,
        component=component,
        position=position,
        region=region,
        aggregation=aggregation,
        expected_source=_source(),
        expected_materialization_generation=3,
    )


def test_a7_query_schema_is_strict_complete_and_round_trips() -> None:
    query = _query()

    assert AgentResultQuery.from_dict(query.to_dict()) == query
    assert AgentResultQuery.from_dict(json.loads(query.to_json())) == query

    unknown = query.to_dict()
    unknown["display_result"] = True
    with pytest.raises(ResultAuthoringError, match="fields"):
        AgentResultQuery.from_dict(unknown)

    widened = query.to_dict()
    widened["expected_materialization_generation"] = False
    with pytest.raises(TypeError, match="integer"):
        AgentResultQuery.from_dict(widened)


def test_a7_sum_is_limited_to_reaction_force() -> None:
    with pytest.raises(ResultAuthoringError, match="only for reaction"):
        _query(aggregation=AgentResultAggregation.SUM)

    reaction = _query(
        variable=AgentResultVariable.REACTION_FORCE,
        component="RF1",
        region="边-固定端",
        aggregation=AgentResultAggregation.SUM,
    )
    assert reaction.aggregation is AgentResultAggregation.SUM


def test_a7_fake_port_explanation_uses_only_returned_scalar() -> None:
    request = _query()
    scalar = AgentResultScalar(
        variable=request.variable,
        component=request.component,
        position=request.position,
        region=request.region,
        aggregation=request.aggregation,
        value=1.25,
        unit="mm",
        source=request.expected_source,
        materialization_generation=3,
        location=AgentResultLocation("node", node_id=42),
    )
    response = AgentResultQueryResponse.success(scalar)
    catalog = AgentResultCatalog(
        source=request.expected_source,
        materialization_generation=3,
        fields=(
            AgentResultField(
                AgentResultVariable.DISPLACEMENT,
                "node",
                ("U1", "Magnitude"),
                "mm",
            ),
        ),
        nodal_regions=("all_nodes",),
        element_regions=("all_elements",),
    )
    port = FakeAgentResultQueryPort(
        {request: response},
        catalog_response=AgentResultCatalogResponse.success(catalog),
    )
    bridge = AgentResultQueryBridge(port)

    assert bridge.catalog().catalog is catalog
    actual = bridge.query(request.to_dict())
    explanation = explain_result_response(actual)

    assert actual is response
    assert port.catalog_calls == 1
    assert port.calls == [request]
    assert "1.25 mm" in explanation
    assert "节点 42" in explanation
    assert "run run-a7" in explanation
    assert "step 分析步-静力" in explanation


def test_a7_no_result_failure_has_no_engineering_scalar() -> None:
    request = _query()
    response = FakeAgentResultQueryPort().query(request)

    assert response.ok is False
    assert response.scalar is None
    assert response.diagnostics[0].code == "result.query.not_configured"
    assert "not configured" in explain_result_response(response)


def test_a7_provider_payload_is_bounded_and_contains_no_local_bulk_data() -> None:
    request = _query()
    scalar = AgentResultScalar(
        variable=request.variable,
        component=request.component,
        position=request.position,
        region=request.region,
        aggregation=AgentResultAggregation.ABSOLUTE_EXTREME,
        value=-2.0,
        unit="mm",
        source=request.expected_source,
        materialization_generation=3,
        location=AgentResultLocation(
            "element_node",
            node_id=4,
            element_id=8,
            local_node=2,
        ),
    )
    payload = AgentResultQueryResponse.success(scalar).to_json()

    assert len(payload.encode("utf-8")) < 4096
    for forbidden in (
        "coordinates",
        "connectivity",
        "records",
        "values",
        "vtk",
        "absolute_path",
        "C:\\",
    ):
        assert forbidden not in payload.casefold()


def test_a7_tool_catalog_exposes_query_without_display_or_confirmation() -> None:
    query_schema = result_query_tool_schema()
    catalog_schema = result_catalog_tool_schema()
    encoded = json.dumps(
        (catalog_schema, query_schema),
        ensure_ascii=False,
        sort_keys=True,
    )

    assert catalog_schema["name"] == "read_accepted_result_catalog"
    assert catalog_schema["input_schema"]["properties"] == {
        "run_id": {
            "type": "string",
            "minLength": 1,
            "maxLength": 128,
        }
    }
    assert "required" not in catalog_schema["input_schema"]
    assert query_schema["name"] == "query_accepted_result"
    assert query_schema["input_schema"]["additionalProperties"] is False
    assert "display" not in encoded.casefold()
    assert "confirm" not in encoded.casefold()
    assert "accept_proposal" not in encoded
