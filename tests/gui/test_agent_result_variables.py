from __future__ import annotations

import pytest

from fem_agent.result_authoring import (
    AcceptedResultReference,
    AcceptedResultSource,
    AgentResultAggregation,
    AgentResultComparisonQuery,
    AgentResultQueryBridge,
    AgentResultQuery,
    AgentResultQueryResponse,
    AgentResultQueryIdentity,
    AgentResultScalar,
    AgentResultVariable,
    FakeAgentResultQueryPort,
    ResultAuthoringError,
    result_comparison_tool_schema,
    result_query_tool_schema,
)
from fem_gui.agent_authoring import SessionResultQueryPort
from tests.helpers.agent_authoring_workflows import (
    dispatch_authoring_tool,
    make_authoring_controller,
)
from tests.helpers.agent_beam_fixtures import (
    BEAM_STEP_NAME as BEAM_STEP,
    apply_beam_definitions,
)
from tests.helpers.agent_line_fixtures import (
    TRUSS_STEP_NAME as TRUSS_STEP,
    apply_definition_action,
    apply_truss_definitions,
    make_meshed_line_session,
    solve_authoring_session,
)
from tests.helpers.agent_session_fixtures import _a5_session as _plate_session


ALL_VARIABLES = {"U", "UR", "RF", "RM", "SF", "SM", "LE", "S"}


def _source(value: object) -> AcceptedResultSource:
    return AcceptedResultSource(
        value.result_id,
        value.session_id,
        value.artifact_id,
        value.model_revision,
        value.step_name,
        value.run_id,
    )


def _query(
    session: object,
    variable: AgentResultVariable,
    component: str,
    position: str,
    region: str,
    aggregation: AgentResultAggregation,
):
    source, generation = session.current_result_identity()
    return AgentResultQuery(
        variable,
        component,
        position,
        region,
        aggregation,
        _source(source),
        generation,
    )


def test_schemas_publish_all_variables_and_rm_sum_contract() -> None:
    assert {item.value for item in AgentResultVariable} == ALL_VARIABLES
    for schema in (result_query_tool_schema(), result_comparison_tool_schema()):
        assert set(
            schema["input_schema"]["properties"]["variable"]["enum"]
        ) == ALL_VARIABLES

    AgentResultQueryIdentity(
        AgentResultVariable.REACTION_MOMENT,
        "RM1",
        "node",
        "all_nodes",
        AgentResultAggregation.SUM,
    )
    with pytest.raises(ResultAuthoringError, match="RF and RM"):
        AgentResultQueryIdentity(
            AgentResultVariable.ROTATION,
            "UR1",
            "node",
            "all_nodes",
            AgentResultAggregation.SUM,
        )


def test_agent_result_request_uses_model_capability_and_exact_units() -> None:
    beam = make_meshed_line_session("Beam2")
    controller, _bridge = make_authoring_controller(beam)
    apply_beam_definitions(controller, beam)

    for name, target, variables, units in (
        ("结果请求-转角力矩", "node", ["UR", "RM"], ["rad", "N*mm"]),
        ("结果请求-截面力", "element", ["SF", "SM"], ["N", "N*mm"]),
    ):
        outcome = dispatch_authoring_tool(
            controller,
            beam,
            "apply_model_definition",
            {
                "action": "create_result_request",
                "parameters": {
                    "name": name,
                    "step_name": BEAM_STEP,
                    "target": target,
                    "variables": variables,
                    "units": units,
                    "confirmed": True,
                },
            },
            f"phase9-{target}",
        )
        assert outcome.ok, outcome.to_json()

    edited = dispatch_authoring_tool(
        controller,
        beam,
        "edit_model_object",
        {
            "object_type": "result_request",
            "target_id": "结果请求-节点",
            "step_name": BEAM_STEP,
            "changes": {
                "variables": ["U", "UR", "RF", "RM"],
                "units": ["mm", "rad", "N", "N*mm"],
                "confirmed": True,
            },
        },
        "phase9-edit-node-output",
    )
    assert edited.ok, edited.to_json()
    revision = beam.session_revision
    rejected_edit = dispatch_authoring_tool(
        controller,
        beam,
        "edit_model_object",
        {
            "object_type": "result_request",
            "target_id": "结果请求-节点",
            "step_name": BEAM_STEP,
            "changes": {
                "variables": ["U", "UR", "RF", "RM"],
                "units": ["mm", "rad", "N", "N"],
                "confirmed": True,
            },
        },
        "phase9-edit-wrong-unit",
    )
    assert not rejected_edit.ok
    assert beam.session_revision == revision

    plate = _plate_session()
    plate_controller, _bridge = make_authoring_controller(plate)
    step = dispatch_authoring_tool(
        plate_controller,
        plate,
        "apply_model_definition",
        {"action": "create_static_step", "parameters": {"name": "分析步-静力"}},
        "phase9-plate-step",
    )
    assert step.ok
    revision = plate.session_revision
    for variable, unit in (("UR", "rad"), ("LE", "1"), ("SF", "N")):
        rejected = dispatch_authoring_tool(
            plate_controller,
            plate,
            "apply_model_definition",
            {
                "action": "create_result_request",
                "parameters": {
                    "name": f"结果请求-拒绝-{variable}",
                    "step_name": "分析步-静力",
                    "target": "node" if variable == "UR" else "element",
                    "variables": [variable],
                    "units": [unit],
                    "confirmed": True,
                },
            },
            f"phase9-reject-{variable.casefold()}",
        )
        assert not rejected.ok
        assert plate.session_revision == revision


def test_beam_catalog_and_queries_cover_rotational_and_section_fields() -> None:
    session = make_meshed_line_session("Beam2")
    controller, bridge = make_authoring_controller(session)
    apply_beam_definitions(controller, session)
    for name, target, variables, units in (
        ("结果请求-转角力矩", "node", ["UR", "RM"], ["rad", "N*mm"]),
        ("结果请求-截面力矩", "element", ["SF", "SM"], ["N", "N*mm"]),
    ):
        apply_definition_action(
            controller,
            session,
            "create_result_request",
            {
                "name": name,
                "step_name": BEAM_STEP,
                "target": target,
                "variables": variables,
                "units": units,
                "confirmed": True,
            },
            f"phase9-solve-{target}",
        )
    solve_authoring_session(controller, bridge, session)

    port = SessionResultQueryPort(session)
    catalog = port.catalog().catalog
    fields = {(item.variable.value, item.position, item.unit) for item in catalog.fields}
    assert ("UR", "node", "rad") in fields
    assert ("RM", "node", "N*mm") in fields
    assert ("SF", "integration_point", "N") in fields
    assert ("SM", "integration_point", "N*mm") in fields

    for variable, component, position, aggregation in (
        (AgentResultVariable.ROTATION, "UR1", "node", AgentResultAggregation.MAXIMUM),
        (AgentResultVariable.REACTION_MOMENT, "RM1", "node", AgentResultAggregation.SUM),
        (AgentResultVariable.SECTION_FORCE, "N", "integration_point", AgentResultAggregation.MAXIMUM),
        (AgentResultVariable.SECTION_MOMENT, "T", "integration_point", AgentResultAggregation.MAXIMUM),
    ):
        response = port.query(
            _query(
                session,
                variable,
                component,
                position,
                "all_nodes" if position == "node" else "all_elements",
                aggregation,
            )
        )
        assert response.ok, response.to_json()


def test_truss_le_catalog_and_centroid_query() -> None:
    session = make_meshed_line_session("Truss2")
    controller, bridge = make_authoring_controller(session)
    apply_truss_definitions(controller, session)
    apply_definition_action(
        controller,
        session,
        "create_result_request",
        {
            "name": "结果请求-对数应变",
            "step_name": TRUSS_STEP,
            "target": "element",
            "variables": ["LE"],
            "units": ["1"],
            "confirmed": True,
        },
        "phase9-le",
    )
    solve_authoring_session(controller, bridge, session)

    port = SessionResultQueryPort(session)
    catalog = port.catalog().catalog
    field = next(item for item in catalog.fields if item.variable.value == "LE")
    assert (field.position, field.components, field.unit) == (
        "centroid",
        ("LE11",),
        "1",
    )
    response = port.query(
        _query(
            session,
            AgentResultVariable.LOGARITHMIC_STRAIN,
            "LE11",
            "centroid",
            "域-杆件",
            AgentResultAggregation.MAXIMUM,
        )
    )
    assert response.ok, response.to_json()


def test_new_variable_comparison_uses_common_identity() -> None:
    first_source = AcceptedResultSource("r1", "s", "a", 1, "Step", "run-1")
    second_source = AcceptedResultSource("r2", "s", "a", 1, "Step", "run-2")
    query = AgentResultComparisonQuery(
        AgentResultQueryIdentity(
            AgentResultVariable.REACTION_MOMENT,
            "RM1",
            "node",
            "all_nodes",
            AgentResultAggregation.SUM,
        ),
        AcceptedResultReference(first_source, 2),
        AcceptedResultReference(second_source, 3),
    )

    baseline_query = query.result_query(query.baseline)
    candidate_query = query.result_query(query.candidate)

    def response(
        source: AcceptedResultSource,
        generation: int,
        value: float,
    ) -> AgentResultQueryResponse:
        return AgentResultQueryResponse.success(
            AgentResultScalar(
                AgentResultVariable.REACTION_MOMENT,
                "RM1",
                "node",
                "all_nodes",
                AgentResultAggregation.SUM,
                value,
                "N*mm",
                source,
                generation,
            )
        )

    compared = AgentResultQueryBridge(
        FakeAgentResultQueryPort(
            {
                baseline_query: response(first_source, 2, 10.0),
                candidate_query: response(second_source, 3, 15.0),
            }
        )
    ).compare(query)

    assert compared.ok
    assert compared.comparison.delta == 5.0
    assert compared.comparison.unit == "N*mm"
