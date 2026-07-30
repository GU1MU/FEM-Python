from __future__ import annotations

import json
import threading

import pytest

from fem_agent.authoring import (
    AgentProposal,
    AuthoringAuthorizationError,
    AuthoringContext,
    ClarificationRequiredError,
    FakeAuthoringPort,
    LocalModelBinding,
    ModelOperation,
    OperationKind,
    ProposalKind,
    ProposalState,
)
from fem_agent.authoring_runtime import (
    AuthoringToolOutcome,
    AuthoringWorkflowController,
    AuthoringWorkflowStage,
    provider_safe_authoring_payload,
)
from fem_agent.engine import AgentSessionEngine
from fem_agent.providers.base import (
    AssistantMessage,
    ProviderResponse,
    ToolCall,
)
from fem_agent.providers.fake import FakeProvider
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
)
from fem_agent.tools.registry import ToolExecutionContext
from fem_gui.agent_authoring import AgentAuthoringBridge


def _response(*calls: ToolCall, text: str | None = None) -> ProviderResponse:
    return ProviderResponse(
        AssistantMessage(
            "assistant",
            content=text,
            tool_calls=tuple(calls),
        ),
        finish_reason="tool_calls" if calls else "stop",
    )


def _context() -> AuthoringContext:
    return AuthoringContext(
        binding=LocalModelBinding(
            "document:a8",
            "native-a8",
            0,
            "native",
            True,
        ),
        model_name="模型-偏心孔板",
        active_part_id=None,
    )


def _requirements() -> dict[str, object]:
    return {
        "modeling_assumption": "plane_stress",
        "length_unit": "mm",
        "force_unit": "N",
        "stress_unit": "MPa",
        "plate_width": 120.0,
        "plate_height": 60.0,
        "plate_thickness": 2.0,
        "hole_radius": 8.0,
        "hole_center_x": 68.0,
        "hole_center_y": 26.0,
        "young_modulus": 210000.0,
        "poisson_ratio": 0.3,
        "mesh_cell_shape": "quadrilateral",
        "mesh_order": 1,
        "mesh_global_size": 5.0,
        "hole_mesh_size": 2.0,
        "fixed_dofs": [1, 2],
        "load_type": "edge_traction",
        "load_direction": "x",
        "load_magnitude": 100.0,
        "load_unit": "N/mm",
        "load_distribution": "uniform",
        "analysis_procedure": "static",
        "nlgeom": False,
        "result_requests": ["U", "S", "RF"],
    }


_REQUIREMENT_GROUP_KEYS = {
    "geometry": (
        "length_unit",
        "force_unit",
        "stress_unit",
        "plate_width",
        "plate_height",
        "hole_radius",
        "hole_center_x",
        "hole_center_y",
    ),
    "mesh": (
        "mesh_cell_shape",
        "mesh_order",
        "mesh_global_size",
        "hole_mesh_size",
    ),
    "definitions": (
        "modeling_assumption",
        "plate_thickness",
        "young_modulus",
        "poisson_ratio",
    ),
    "analysis": (
        "fixed_dofs",
        "load_type",
        "load_direction",
        "load_magnitude",
        "load_unit",
        "load_distribution",
        "analysis_procedure",
        "nlgeom",
        "result_requests",
    ),
}


def _requirements_for(group: str) -> dict[str, object]:
    values = _requirements()
    return {key: values[key] for key in _REQUIREMENT_GROUP_KEYS[group]}


def _proposal(kind: ProposalKind, suffix: str) -> AgentProposal:
    operation, parameters = {
        ProposalKind.GEOMETRY: (
            OperationKind.ADD_NATIVE_PART,
            {
                "part_name": "部件-偏心孔板",
                "recipe": {"kind": "plate_with_hole"},
            },
        ),
        ProposalKind.MESH: (
            OperationKind.REQUEST_MESH,
            {
                "part_id": "part-a8",
                "mesh_intent_hash": "0" * 64,
            },
        ),
        ProposalKind.SOLVE: (
            OperationKind.REQUEST_SOLVE,
            {
                "step_name": "分析步-静力",
                "validation_stamp": "0" * 64,
            },
        ),
    }[kind]
    return AgentProposal.create(
        proposal_id=f"proposal-{suffix}",
        proposal_kind=kind,
        agent_session_id="agent-a8",
        turn_id=f"turn-{suffix}",
        source_tool_call_ids=(f"call-{suffix}",),
        target_document_id="document:a8",
        target_session_id="native-a8",
        base_session_revision=0,
        draft_revision=1,
        operations=(ModelOperation(operation, parameters),),
        preconditions={"confirmed": True},
        expected_changes={"operation": suffix},
        invalidation_impact={"results": kind is not ProposalKind.SOLVE},
        display_summary={
            "title": f"{kind.value} proposal",
            "summary": f"准备 {suffix}",
            "confirm_label": "开始",
        },
    )


def _result_bridge() -> tuple[AgentResultQueryBridge, tuple[AgentResultQuery, ...]]:
    source = AcceptedResultSource(
        "result-a8",
        "native-a8",
        "artifact-a8",
        4,
        "分析步-静力",
        "run-a8",
    )
    queries = (
        AgentResultQuery(
            AgentResultVariable.DISPLACEMENT,
            "Magnitude",
            "node",
            "all_nodes",
            AgentResultAggregation.MAXIMUM,
            source,
            2,
        ),
        AgentResultQuery(
            AgentResultVariable.STRESS,
            "Mises",
            "element",
            "all_elements",
            AgentResultAggregation.ABSOLUTE_EXTREME,
            source,
            2,
        ),
        AgentResultQuery(
            AgentResultVariable.REACTION_FORCE,
            "RF1",
            "node",
            "边-固定端",
            AgentResultAggregation.SUM,
            source,
            2,
        ),
    )
    scalars = (
        AgentResultScalar(
            queries[0].variable,
            queries[0].component,
            queries[0].position,
            queries[0].region,
            queries[0].aggregation,
            0.42,
            "mm",
            source,
            2,
            AgentResultLocation("node", node_id=23),
        ),
        AgentResultScalar(
            queries[1].variable,
            queries[1].component,
            queries[1].position,
            queries[1].region,
            queries[1].aggregation,
            123.0,
            "MPa",
            source,
            2,
            AgentResultLocation("element", element_id=17),
        ),
        AgentResultScalar(
            queries[2].variable,
            queries[2].component,
            queries[2].position,
            queries[2].region,
            queries[2].aggregation,
            -1000.0,
            "N",
            source,
            2,
        ),
    )
    catalog = AgentResultCatalog(
        source,
        2,
        (
            AgentResultField(
                AgentResultVariable.DISPLACEMENT,
                "node",
                ("Magnitude",),
                "mm",
            ),
            AgentResultField(
                AgentResultVariable.STRESS,
                "element",
                ("Mises",),
                "MPa",
            ),
            AgentResultField(
                AgentResultVariable.REACTION_FORCE,
                "node",
                ("RF1",),
                "N",
            ),
        ),
        ("all_nodes", "边-固定端"),
        ("all_elements",),
    )
    port = FakeAgentResultQueryPort(
        {
            query: AgentResultQueryResponse.success(scalar)
            for query, scalar in zip(queries, scalars, strict=True)
        },
        catalog_response=AgentResultCatalogResponse.success(catalog),
    )
    return AgentResultQueryBridge(port), queries


def _controller(
    bridge: AgentAuthoringBridge,
) -> tuple[AuthoringWorkflowController, list[str], tuple[AgentResultQuery, ...]]:
    result_bridge, queries = _result_bridge()
    calls: list[str] = []

    def proposal_handler(
        kind: ProposalKind,
        suffix: str,
    ):
        def handle(_arguments, _controller):
            proposal = _proposal(kind, suffix)
            bridge.register_proposal(proposal)
            calls.append(suffix)
            return AuthoringToolOutcome(
                f"{suffix} proposal is waiting for GUI confirmation.",
                {
                    "proposal_id": proposal.proposal_id,
                    "proposal_hash": proposal.proposal_hash,
                    "state": "pending_confirmation",
                },
            )

        return handle

    def scopes(_arguments, controller):
        controller.confirmed_requirements("definitions")
        calls.append("scopes")
        return AuthoringToolOutcome(
            "Scopes and materials applied through one reversible patch.",
            {
                "state": "succeeded",
                "undo_available": True,
                "names": [
                    "边-固定端",
                    "边-加载端",
                    "边-孔边",
                    "域-板体",
                    "材料-结构钢",
                    "截面-平面应力",
                ],
            },
        )

    def analysis(_arguments, controller):
        controller.confirmed_requirements("analysis")
        calls.append("analysis")
        return AuthoringToolOutcome(
            "Analysis definitions applied through one reversible patch.",
            {
                "state": "succeeded",
                "undo_available": True,
                "names": [
                    "分析步-静力",
                    "位移-固定端",
                    "载荷-拉伸",
                    "结果请求-位移应力",
                ],
            },
        )

    def preflight(_arguments, _controller):
        calls.append("preflight")
        return AuthoringToolOutcome(
            "Deterministic native preflight passed.",
            {"passed": True, "blocking_diagnostic_count": 0},
        )

    def catalog(_arguments, _controller):
        response = result_bridge.catalog()
        calls.append("catalog")
        return AuthoringToolOutcome(
            "Accepted result catalog read locally.",
            response.to_dict(),
            ok=response.ok,
        )

    def query(arguments, _controller):
        response = result_bridge.query(arguments)
        calls.append("query")
        return AuthoringToolOutcome(
            "One accepted result scalar read locally.",
            response.to_dict(),
            ok=response.ok,
        )

    handlers = {
        "prepare_geometry_proposal": proposal_handler(
            ProposalKind.GEOMETRY,
            "geometry",
        ),
        "prepare_mesh_proposal": proposal_handler(
            ProposalKind.MESH,
            "mesh",
        ),
        "apply_scopes_and_materials": scopes,
        "apply_analysis_definitions": analysis,
        "run_native_preflight": preflight,
        "prepare_solve_proposal": proposal_handler(
            ProposalKind.SOLVE,
            "solve",
        ),
        "read_accepted_result_catalog": catalog,
        "query_accepted_result": query,
    }
    return AuthoringWorkflowController(lambda: _context(), handlers), calls, queries


def _dispatch(
    controller: AuthoringWorkflowController,
    name: str,
    arguments: dict[str, object],
    index: int,
):
    return controller.dispatch(
        name,
        arguments,
        ToolExecutionContext("session-a8", 0, f"key-{index}"),
    )


def test_a8_dynamic_catalog_requires_gui_review_and_never_publishes_confirmation() -> (
    None
):
    bridge = AgentAuthoringBridge(FakeAuthoringPort())
    bridge.bind_context(_context())
    controller, calls, _queries = _controller(bridge)

    initial_names = {item.name for item in controller.definitions}
    assert {
        "read_authoring_context",
        "set_authoring_requirements",
        "request_requirement_review",
    } <= initial_names
    assert not any(
        fragment in name
        for name in initial_names
        for fragment in ("accept", "confirm", "reject", "cancel")
    )
    unavailable = _dispatch(
        controller,
        "prepare_geometry_proposal",
        {},
        1,
    )
    assert unavailable.ok is False
    assert calls == []

    recorded = _dispatch(
        controller,
        "set_authoring_requirements",
        {
            "turn_id": "turn-requirements",
            "requirements": _requirements_for("geometry"),
        },
        2,
    )
    review_result = _dispatch(
        controller,
        "request_requirement_review",
        {},
        3,
    )
    assert recorded.ok is True
    assert review_result.ok is True
    assert controller.stage is AuthoringWorkflowStage.REVIEW_PENDING
    assert {item.name for item in controller.definitions} == {
        "read_authoring_context"
    }

    pending = controller.pending_review
    assert pending is not None
    confirmed = bridge.confirm_requirement_review_from_gui(
        controller.ledger,
        pending,
    )
    controller.resolve_requirement_review(confirmed)
    assert controller.stage is AuthoringWorkflowStage.GEOMETRY_READY
    assert {item.name for item in controller.definitions} == {
        "read_authoring_context",
        "prepare_geometry_proposal",
    }


@pytest.mark.parametrize(
    ("stage", "group", "operation"),
    [
        (
            AuthoringWorkflowStage.MESH_READY,
            "mesh",
            "prepare_mesh_proposal",
        ),
        (
            AuthoringWorkflowStage.DEFINITIONS_READY,
            "definitions",
            "apply_scopes_and_materials",
        ),
        (
            AuthoringWorkflowStage.ANALYSIS_DEFINITIONS_READY,
            "analysis",
            "apply_analysis_definitions",
        ),
    ],
)
def test_a8_ready_stage_opens_its_operation_only_after_stage_review(
    stage,
    group,
    operation,
) -> None:
    controller = AuthoringWorkflowController(
        lambda: _context(),
        {
            operation: lambda _arguments, _controller: AuthoringToolOutcome(
                "Stage operation completed.",
                {"state": "succeeded"},
            )
        },
    )
    controller._stage = stage
    bridge = AgentAuthoringBridge(FakeAuthoringPort())
    bridge.bind_context(_context())

    names = {item.name for item in controller.definitions}
    assert {
        "set_authoring_requirements",
        "request_requirement_review",
    } <= names
    assert operation not in names
    schema = next(
        item
        for item in controller.definitions
        if item.name == "set_authoring_requirements"
    ).parameters["properties"]["requirements"]["properties"]
    assert set(schema) == set(_REQUIREMENT_GROUP_KEYS[group])

    assert _dispatch(
        controller,
        "set_authoring_requirements",
        {
            "turn_id": f"turn-{group}",
            "requirements": _requirements_for(group),
        },
        30,
    ).ok
    assert _dispatch(
        controller,
        "request_requirement_review",
        {},
        31,
    ).ok
    pending = controller.pending_review
    assert pending is not None
    confirmed = bridge.confirm_requirement_review_from_gui(
        controller.ledger,
        pending,
    )
    controller.resolve_requirement_review(confirmed)

    names = {item.name for item in controller.definitions}
    assert operation in names
    assert "set_authoring_requirements" not in names
    assert "request_requirement_review" not in names


def test_a8_requirement_schema_and_review_fail_closed_for_unsupported_2d_values() -> (
    None
):
    controller = AuthoringWorkflowController(lambda: _context(), {})
    requirement_schema = next(
        item
        for item in controller.definitions
        if item.name == "set_authoring_requirements"
    ).parameters["properties"]["requirements"]["properties"]
    assert set(requirement_schema) == set(_REQUIREMENT_GROUP_KEYS["geometry"])

    controller._stage = AuthoringWorkflowStage.DEFINITIONS_READY
    definition_schema = next(
        item
        for item in controller.definitions
        if item.name == "set_authoring_requirements"
    ).parameters["properties"]["requirements"]["properties"]
    assert definition_schema["modeling_assumption"]["enum"] == [
        "plane_stress",
        "plane_strain",
    ]

    controller._stage = AuthoringWorkflowStage.ANALYSIS_DEFINITIONS_READY
    analysis_schema = next(
        item
        for item in controller.definitions
        if item.name == "set_authoring_requirements"
    ).parameters["properties"]["requirements"]["properties"]
    assert analysis_schema["load_type"]["enum"] == [
        "edge_traction",
        "edge_pressure",
    ]
    assert "z" not in analysis_schema["load_direction"]["enum"]
    assert analysis_schema["fixed_dofs"]["items"] == {
        "type": "integer",
        "minimum": 1,
        "maximum": 2,
    }

    for key, value in (
        ("modeling_assumption", "solid_3d"),
        ("load_type", "surface_traction"),
        ("load_direction", "z"),
        ("fixed_dofs", [0, 1]),
        ("fixed_dofs", [1, 4]),
    ):
        candidate = AuthoringWorkflowController(lambda: _context(), {})
        candidate._stage = (
            AuthoringWorkflowStage.DEFINITIONS_READY
            if key == "modeling_assumption"
            else AuthoringWorkflowStage.ANALYSIS_DEFINITIONS_READY
        )
        rejected = _dispatch(
            candidate,
            "set_authoring_requirements",
            {
                "turn_id": "turn-unsupported",
                "requirements": {key: value},
            },
            20,
        )
        assert rejected.ok is False

    for load_type, direction, magnitude in (
        ("edge_traction", "inward_normal", 10.0),
        ("edge_pressure", "x", 10.0),
        ("edge_pressure", "inward_normal", -10.0),
        ("edge_pressure", "outward_normal", 10.0),
    ):
        incompatible = _requirements_for("analysis")
        incompatible.update(
            {
                "load_type": load_type,
                "load_direction": direction,
                "load_magnitude": magnitude,
            }
        )
        candidate = AuthoringWorkflowController(lambda: _context(), {})
        candidate._stage = AuthoringWorkflowStage.ANALYSIS_DEFINITIONS_READY
        assert _dispatch(
            candidate,
            "set_authoring_requirements",
            {
                "turn_id": "turn-incompatible",
                "requirements": incompatible,
            },
            21,
        ).ok is True
        review = _dispatch(candidate, "request_requirement_review", {}, 22)
        assert review.ok is False
        assert candidate.pending_review is None
        assert (
            candidate.stage
            is AuthoringWorkflowStage.ANALYSIS_DEFINITIONS_READY
        )


def test_a8_requirement_batch_validation_is_atomic() -> None:
    controller = AuthoringWorkflowController(lambda: _context(), {})
    before_revision = controller.ledger.revision
    before_entries = controller.ledger.entries

    rejected = _dispatch(
        controller,
        "set_authoring_requirements",
        {
            "turn_id": "turn-mixed-validity",
            "requirements": {
                "length_unit": "mm",
                "fixed_dofs": [0, 1],
            },
        },
        26,
    )

    assert rejected.ok is False
    assert controller.ledger.revision == before_revision
    assert controller.ledger.entries == before_entries


def test_a8_seeded_binding_invalidates_confirmed_ledger_without_context_read() -> None:
    initial = _context()
    switched = AuthoringContext(
        binding=LocalModelBinding(
            "document:switched",
            "native-switched",
            0,
            "native",
            True,
        ),
        model_name="模型-新文档",
        active_part_id=None,
    )
    current = [initial]
    controller = AuthoringWorkflowController(lambda: current[0], {})
    controller.observe_binding(initial)
    bridge = AgentAuthoringBridge(FakeAuthoringPort())
    bridge.bind_context(initial)

    assert _dispatch(
        controller,
        "set_authoring_requirements",
        {
            "turn_id": "turn-before-switch",
            "requirements": _requirements_for("geometry"),
        },
        23,
    ).ok is True
    assert _dispatch(
        controller,
        "request_requirement_review",
        {},
        24,
    ).ok is True
    pending = controller.pending_review
    assert pending is not None
    confirmed = bridge.confirm_requirement_review_from_gui(
        controller.ledger,
        pending,
    )
    controller.resolve_requirement_review(confirmed)
    assert controller.stage is AuthoringWorkflowStage.GEOMETRY_READY

    current[0] = switched
    assert controller.observe_binding(switched) is False
    assert controller.stage is AuthoringWorkflowStage.STALE
    assert controller.ledger.entries == ()
    assert {item.name for item in controller.definitions} == {
        "read_authoring_context"
    }
    with pytest.raises(ClarificationRequiredError):
        controller.confirmed_requirements()

    assert _dispatch(
        controller,
        "read_authoring_context",
        {},
        25,
    ).ok is True
    assert controller.stage is AuthoringWorkflowStage.REQUIREMENTS


def test_a8_binding_allows_expected_pending_revision_only() -> None:
    initial = _context()
    next_revision = AuthoringContext(
        binding=LocalModelBinding(
            initial.binding.document_id,
            initial.binding.session_id,
            1,
            initial.binding.source_kind,
            initial.binding.supported,
        ),
        model_name=initial.model_name,
        active_part_id="part-a8",
    )
    controller = AuthoringWorkflowController(lambda: initial, {})
    controller.observe_binding(initial)
    controller._stage = AuthoringWorkflowStage.GEOMETRY_PENDING
    controller._pending_operation = "geometry"

    assert controller.observe_binding(next_revision) is True
    assert controller.stage is AuthoringWorkflowStage.GEOMETRY_PENDING
    controller.record_proposal_state("geometry", ProposalState.SUCCEEDED)
    assert controller.stage is AuthoringWorkflowStage.MESH_READY

    external_revision = AuthoringContext(
        binding=LocalModelBinding(
            initial.binding.document_id,
            initial.binding.session_id,
            2,
            initial.binding.source_kind,
            initial.binding.supported,
        ),
        model_name=initial.model_name,
        active_part_id="part-a8",
    )
    assert controller.observe_binding(external_revision) is False
    assert controller.stage is AuthoringWorkflowStage.STALE

    switched_session = AuthoringContext(
        binding=LocalModelBinding(
            "document:new-native",
            "new-native",
            1,
            "native",
            True,
        ),
        model_name="模型-新建",
        active_part_id="part-new",
    )
    blocked = AuthoringWorkflowController(lambda: switched_session, {})
    blocked.observe_binding(initial)
    blocked._stage = AuthoringWorkflowStage.GEOMETRY_PENDING
    blocked._pending_operation = "geometry"
    assert blocked.observe_binding(
        switched_session,
        proposal_staled=True,
    ) is False
    assert blocked.stage is AuthoringWorkflowStage.STALE


def test_a8_provider_payload_rejects_paths_bulk_arrays_and_unsafe_summaries() -> (
    None
):
    for payload in (
        {"message": "local file D:\\private\\model.femproj"},
        {"message": "local file \\\\server\\share\\model.femproj"},
        {"message": "local file /home/user/model.femproj"},
        {"node_ids": [1, 2]},
        {"element_ids": [1, 2]},
        {"coordinates": [[0.0, 0.0]]},
        {"result_provider": "hidden"},
        {"model_patch": {"schema_version": "1.0"}},
    ):
        with pytest.raises(ValueError):
            provider_safe_authoring_payload(payload)

    assert provider_safe_authoring_payload(
        {"node_id": 7, "element_id": 9, "value": 1.5}
    ) == {"node_id": 7, "element_id": 9, "value": 1.5}

    with pytest.raises(ValueError):
        AuthoringToolOutcome(
            "failed at D:\\private\\model.femproj",
            {"code": "local.failure"},
        )

    def unsafe(_arguments, _controller):
        raise ValueError("failed at D:\\private\\secret.femproj")

    controller = AuthoringWorkflowController(
        lambda: _context(),
        {"prepare_geometry_proposal": unsafe},
    )
    controller._stage = AuthoringWorkflowStage.GEOMETRY_READY
    result = _dispatch(controller, "prepare_geometry_proposal", {}, 4)
    serialized = result.to_json()
    assert "D:" not in serialized
    assert "private" not in serialized
    assert "ValueError" in serialized


def test_a8_fake_provider_runs_dynamic_engine_chain_with_three_gui_confirmations(
    tmp_path,
) -> None:
    bridge = AgentAuthoringBridge(FakeAuthoringPort())
    bridge.bind_context(_context())
    controller, local_calls, queries = _controller(bridge)
    provider = FakeProvider(
        [
            _response(text="请先确认板尺寸、孔尺寸、孔位置和项目单位制。"),
            _response(
                ToolCall(
                    "geometry-requirements",
                    "set_authoring_requirements",
                    {
                        "turn_id": "turn-geometry",
                        "requirements": _requirements_for("geometry"),
                    },
                ),
                ToolCall("geometry-review", "request_requirement_review", {}),
            ),
            _response(text="请确认几何需求。"),
            _response(
                ToolCall("geometry", "prepare_geometry_proposal", {})
            ),
            _response(text="几何提案等待 GUI 确认。"),
            _response(
                ToolCall(
                    "mesh-requirements",
                    "set_authoring_requirements",
                    {
                        "turn_id": "turn-mesh",
                        "requirements": _requirements_for("mesh"),
                    },
                ),
                ToolCall("mesh-review", "request_requirement_review", {}),
            ),
            _response(text="请确认网格需求。"),
            _response(ToolCall("mesh", "prepare_mesh_proposal", {})),
            _response(text="网格提案等待 GUI 确认。"),
            _response(
                ToolCall(
                    "definition-requirements",
                    "set_authoring_requirements",
                    {
                        "turn_id": "turn-definitions",
                        "requirements": _requirements_for("definitions"),
                    },
                ),
                ToolCall(
                    "definition-review",
                    "request_requirement_review",
                    {},
                ),
            ),
            _response(text="请确认材料与截面需求。"),
            _response(
                ToolCall(
                    "scopes",
                    "apply_scopes_and_materials",
                    {},
                )
            ),
            _response(text="作用域与材料已自动应用并可撤销。"),
            _response(
                ToolCall(
                    "analysis-requirements",
                    "set_authoring_requirements",
                    {
                        "turn_id": "turn-analysis",
                        "requirements": _requirements_for("analysis"),
                    },
                ),
                ToolCall(
                    "analysis-review",
                    "request_requirement_review",
                    {},
                ),
            ),
            _response(text="请确认边界条件、载荷与分析需求。"),
            _response(
                ToolCall(
                    "analysis",
                    "apply_analysis_definitions",
                    {},
                )
            ),
            _response(text="分析定义已自动应用并可撤销。"),
            _response(ToolCall("preflight", "run_native_preflight", {})),
            _response(text="预检通过。"),
            _response(
                ToolCall("solve", "prepare_solve_proposal", {})
            ),
            _response(text="求解提案等待 GUI 确认。"),
            _response(
                ToolCall(
                    "catalog",
                    "read_accepted_result_catalog",
                    {},
                ),
                *(
                    ToolCall(
                        f"query-{index}",
                        "query_accepted_result",
                        query.to_dict(),
                    )
                    for index, query in enumerate(queries, start=1)
                ),
            ),
            _response(
                text="最大位移、应力极值和固定端反力已由本地结果读取。"
            ),
        ]
    )
    engine = AgentSessionEngine(
        tmp_path / "agent-private",
        provider,
        session_id="session-a8",
        dynamic_tools=controller,
    )

    engine.send_message("帮我建立一个偏心的带孔平板模型，孔的位置偏离板的中心")
    assert local_calls == []
    authoring_prompt = provider.requests[0].messages[0].content
    assert "strict attention boundary" in authoring_prompt
    assert "Do not ask for or mention mesh, material" in authoring_prompt
    assert "full-project questionnaire" in authoring_prompt
    assert "Do not volunteer or enumerate FEM Agent features" in (
        authoring_prompt
    )
    assert "smallest useful set" in authoring_prompt
    assert "`empty` does not mean" in authoring_prompt
    engine.send_message("以下是明确的几何和单位参数。")
    pending = controller.pending_review
    assert pending is not None
    assert {item.key for item in pending.fields} == set(
        _REQUIREMENT_GROUP_KEYS["geometry"]
    )
    confirmed = bridge.confirm_requirement_review_from_gui(
        controller.ledger,
        pending,
    )
    controller.resolve_requirement_review(confirmed)

    engine.send_message("准备几何提案。")
    with pytest.raises(AuthoringAuthorizationError):
        bridge.accept_proposal("proposal-geometry")
    assert bridge.accept_from_gui_control("proposal-geometry").state is ProposalState.ACCEPTED
    controller.record_proposal_state("geometry", ProposalState.SUCCEEDED)

    engine.send_message("以下是明确的网格参数。")
    pending = controller.pending_review
    assert pending is not None
    assert {item.key for item in pending.fields} == set(
        _REQUIREMENT_GROUP_KEYS["mesh"]
    )
    confirmed = bridge.confirm_requirement_review_from_gui(
        controller.ledger,
        pending,
    )
    controller.resolve_requirement_review(confirmed)

    engine.send_message("准备网格提案。")
    assert bridge.accept_from_gui_control("proposal-mesh").state is ProposalState.ACCEPTED
    controller.record_proposal_state("mesh", ProposalState.SUCCEEDED)

    engine.send_message("以下是明确的材料与截面参数。")
    pending = controller.pending_review
    assert pending is not None
    assert {item.key for item in pending.fields} == set(
        _REQUIREMENT_GROUP_KEYS["definitions"]
    )
    confirmed = bridge.confirm_requirement_review_from_gui(
        controller.ledger,
        pending,
    )
    controller.resolve_requirement_review(confirmed)

    engine.send_message("应用作用域与材料。")

    engine.send_message("以下是明确的边界条件、载荷与分析参数。")
    pending = controller.pending_review
    assert pending is not None
    assert {item.key for item in pending.fields} == set(
        _REQUIREMENT_GROUP_KEYS["analysis"]
    )
    confirmed = bridge.confirm_requirement_review_from_gui(
        controller.ledger,
        pending,
    )
    controller.resolve_requirement_review(confirmed)

    engine.send_message("应用分析定义。")
    engine.send_message("运行预检。")
    engine.send_message("准备求解提案。")
    assert bridge.accept_from_gui_control("proposal-solve").state is ProposalState.ACCEPTED
    controller.record_proposal_state("solve", ProposalState.SUCCEEDED)

    events = engine.send_message("读取三项已接受结果。")
    assert controller.stage is AuthoringWorkflowStage.RESULTS_READY
    assert local_calls == [
        "geometry",
        "mesh",
        "scopes",
        "analysis",
        "preflight",
        "solve",
        "catalog",
        "query",
        "query",
        "query",
    ]
    tool_payloads = [
        event.data["result"]
        for event in events
        if event.event.value == "tool_completed"
    ]
    values = [
        payload["data"]["scalar"]["value"]
        for payload in tool_payloads
        if payload["data"] is not None
        and payload["data"].get("scalar") is not None
    ]
    assert values == [0.42, 123.0, -1000.0]

    all_tool_names = {
        definition.name
        for request in provider.requests
        for definition in request.tools
    }
    assert not any(
        fragment in name
        for name in all_tool_names
        for fragment in (
            "accept_proposal",
            "confirm_mesh",
            "confirm_solve",
            "confirm_requirement",
        )
    )
    captured = json.dumps(
        [
            {
                "messages": [
                    {
                        "role": message.role,
                        "content": message.content,
                    }
                    for message in request.messages
                ],
                "tools": [item.name for item in request.tools],
            }
            for request in provider.requests
        ],
        ensure_ascii=False,
    ).casefold()
    for forbidden in (
        "node_ids",
        "element_ids",
        "connectivity",
        "result_arrays",
        "modelsession",
        "raw_patch",
        "vtk",
    ):
        assert forbidden not in captured


@pytest.mark.parametrize(
    ("operation", "terminal_state", "pending_stage", "failure_stage"),
    [
        (
            "geometry",
            ProposalState.CANCELLED,
            AuthoringWorkflowStage.GEOMETRY_PENDING,
            AuthoringWorkflowStage.GEOMETRY_READY,
        ),
        (
            "mesh",
            ProposalState.FAILED,
            AuthoringWorkflowStage.MESH_PENDING,
            AuthoringWorkflowStage.MESH_READY,
        ),
        (
            "solve",
            ProposalState.FAILED,
            AuthoringWorkflowStage.SOLVE_PENDING,
            AuthoringWorkflowStage.SOLVE_READY,
        ),
    ],
)
def test_a8_proposal_failure_cancel_and_stale_have_one_terminal(
    operation,
    terminal_state,
    pending_stage,
    failure_stage,
) -> None:
    bridge = AgentAuthoringBridge(FakeAuthoringPort())
    bridge.bind_context(_context())
    controller, _calls, _queries = _controller(bridge)
    controller._stage = pending_stage
    controller._pending_operation = operation

    controller.record_proposal_state(operation, terminal_state, "terminal")

    assert controller.stage is failure_stage
    assert controller.terminal_records[-1].state == terminal_state.value
    assert controller.terminal_records[-1].operation == operation
    with pytest.raises(ValueError):
        controller.record_proposal_state(
            operation,
            ProposalState.FAILED,
            "late failure",
        )

    controller.invalidate_binding("document switched")
    assert controller.stage is AuthoringWorkflowStage.STALE
    assert controller.terminal_records[-1].state == "stale"


def test_a8_dynamic_dispatch_is_serial_and_thread_safe() -> None:
    entered: list[int] = []

    def handler(_arguments, _controller):
        entered.append(threading.get_ident())
        return AuthoringToolOutcome("Geometry proposal prepared.", {"ok": True})

    controller = AuthoringWorkflowController(
        lambda: _context(),
        {"prepare_geometry_proposal": handler},
    )
    controller._stage = AuthoringWorkflowStage.GEOMETRY_READY
    result = _dispatch(controller, "prepare_geometry_proposal", {}, 10)

    assert result.ok is True
    assert entered == [threading.get_ident()]
    assert controller.stage is AuthoringWorkflowStage.GEOMETRY_PENDING
