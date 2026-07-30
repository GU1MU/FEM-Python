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
    MeshSummary,
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
                    "proposal_view": {
                        "proposal_id": proposal.proposal_id,
                        "proposal_hash": proposal.proposal_hash,
                        "proposal_kind": proposal.proposal_kind.value,
                        "title": f"{suffix} proposal",
                        "summary": f"{suffix} proposal",
                        "impact": "Apply after local confirmation.",
                        "confirm_label": "加入模型",
                        "target_document_id": proposal.target_document_id,
                        "target_session_id": proposal.target_session_id,
                        "base_session_revision": (
                            proposal.base_session_revision
                        ),
                    },
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


def test_a8_geometry_uses_one_operation_confirmation_without_requirement_review() -> None:
    calls: list[str] = []

    def geometry(_arguments, _controller):
        calls.append("geometry")
        return AuthoringToolOutcome(
            "Geometry proposal registered.",
            {"state": "pending_confirmation"},
        )

    controller = AuthoringWorkflowController(
        lambda: _context(),
        {"prepare_geometry_proposal": geometry},
    )
    initial_names = {item.name for item in controller.definitions}
    assert "set_authoring_requirements" in initial_names
    assert "prepare_geometry_proposal" not in initial_names
    assert "request_requirement_review" not in initial_names
    assert not any(
        fragment in name
        for name in initial_names
        for fragment in ("accept", "confirm", "reject", "cancel")
    )

    recorded = _dispatch(
        controller,
        "set_authoring_requirements",
        {
            "turn_id": "turn-geometry",
            "requirements": _requirements_for("geometry"),
        },
        1,
    )
    assert recorded.ok
    assert recorded.data["operation_confirmation_required"] is True
    names = {item.name for item in controller.definitions}
    assert "prepare_geometry_proposal" in names
    assert "set_authoring_requirements" in names
    assert "request_requirement_review" not in names

    prepared = _dispatch(controller, "prepare_geometry_proposal", {}, 2)
    assert prepared.ok
    assert calls == ["geometry"]
    assert controller.stage is AuthoringWorkflowStage.GEOMETRY_PENDING


def test_a8_mesh_uses_one_confirmation_and_definitions_are_direct() -> None:
    calls: list[str] = []

    def handler(arguments, _controller):
        calls.append(str(arguments.get("action", "mesh")))
        return AuthoringToolOutcome("Applied.", {"state": "succeeded"})

    controller = AuthoringWorkflowController(
        lambda: _context(),
        {
            "prepare_mesh_proposal": handler,
            "apply_model_definition": handler,
            "run_native_preflight": handler,
        },
    )
    controller._stage = AuthoringWorkflowStage.MESH_READY
    assert "prepare_mesh_proposal" not in {
        item.name for item in controller.definitions
    }

    assert _dispatch(
        controller,
        "set_authoring_requirements",
        {
            "turn_id": "turn-mesh",
            "requirements": _requirements_for("mesh"),
        },
        3,
    ).ok
    names = {item.name for item in controller.definitions}
    assert "prepare_mesh_proposal" in names
    assert "request_requirement_review" not in names

    controller._stage = AuthoringWorkflowStage.DEFINITIONS_READY
    names = {item.name for item in controller.definitions}
    assert {"apply_model_definition", "run_native_preflight"} <= names
    assert "set_authoring_requirements" not in names
    assert "request_requirement_review" not in names

    applied = _dispatch(
        controller,
        "apply_model_definition",
        {
            "action": "create_material",
            "parameters": {
                "name": "材料-铝合金",
                "properties": {"E": 70000.0, "nu": 0.33},
            },
        },
        4,
    )
    assert applied.ok
    assert calls == ["create_material"]
    assert controller.stage is AuthoringWorkflowStage.DEFINITIONS_READY


def test_a8_only_geometry_mesh_and_solve_publish_execution_proposals() -> None:
    handler = lambda _arguments, _controller: AuthoringToolOutcome(
        "Registered.",
        {"state": "pending_confirmation"},
    )
    controller = AuthoringWorkflowController(
        lambda: _context(),
        {
            "prepare_geometry_proposal": handler,
            "prepare_mesh_proposal": handler,
            "prepare_solve_proposal": handler,
            "apply_model_definition": handler,
            "edit_model_object": handler,
        },
    )

    assert _dispatch(
        controller,
        "set_authoring_requirements",
        {
            "turn_id": "turn-geometry",
            "requirements": _requirements_for("geometry"),
        },
        5,
    ).ok
    assert "prepare_geometry_proposal" in {
        item.name for item in controller.definitions
    }

    controller._stage = AuthoringWorkflowStage.MESH_READY
    assert _dispatch(
        controller,
        "set_authoring_requirements",
        {
            "turn_id": "turn-mesh",
            "requirements": _requirements_for("mesh"),
        },
        6,
    ).ok
    assert "prepare_mesh_proposal" in {
        item.name for item in controller.definitions
    }

    controller._stage = AuthoringWorkflowStage.SOLVE_READY
    names = {item.name for item in controller.definitions}
    assert "prepare_solve_proposal" in names
    assert "apply_model_definition" in names
    assert "request_requirement_review" not in names


def test_a8_direct_definition_schema_is_granular() -> None:
    controller = AuthoringWorkflowController(
        lambda: _context(),
        {
            "apply_model_definition": lambda _arguments, _controller: (
                AuthoringToolOutcome("Applied.", {"state": "succeeded"})
            ),
        },
    )
    controller._stage = AuthoringWorkflowStage.DEFINITIONS_READY
    tool = next(
        item
        for item in controller.definitions
        if item.name == "apply_model_definition"
    )
    actions = tool.parameters["properties"]["action"]["enum"]
    assert actions == [
        "create_plate_scopes",
        "create_material",
        "create_section",
        "assign_section",
        "create_static_step",
        "create_boundary_condition",
        "create_load",
        "create_result_request",
    ]
    assert "apply_scopes_and_materials" not in {
        item.name for item in controller.definitions
    }
    assert "apply_analysis_definitions" not in {
        item.name for item in controller.definitions
    }


def test_a8_existing_current_mesh_exposes_direct_definitions_immediately() -> None:
    context = AuthoringContext(
        binding=LocalModelBinding(
            "document:existing",
            "native-existing",
            7,
            "native",
            True,
        ),
        model_name="模型-既有",
        active_part_id="part-existing",
        mesh=MeshSummary(True, True, 20, 10),
    )
    controller = AuthoringWorkflowController(
        lambda: context,
        {
            "apply_model_definition": lambda _arguments, _controller: (
                AuthoringToolOutcome("Applied.", {"state": "succeeded"})
            ),
        },
    )
    controller.observe_binding(context)

    assert "apply_model_definition" in {
        item.name for item in controller.definitions
    }


def test_a8_requirement_batch_validation_is_atomic() -> None:
    controller = AuthoringWorkflowController(lambda: _context(), {})
    before_revision = controller.ledger.revision
    before_entries = controller.ledger.entries

    rejected = _dispatch(
        controller,
        "set_authoring_requirements",
        {
            "turn_id": "turn-invalid",
            "requirements": {
                "plate_width": 100.0,
                "plate_height": -1.0,
            },
        },
        7,
    )

    assert rejected.ok is False
    assert controller.ledger.revision == before_revision
    assert controller.ledger.entries == before_entries


def test_a8_binding_change_clears_collected_requirements() -> None:
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
    assert _dispatch(
        controller,
        "set_authoring_requirements",
        {
            "turn_id": "turn-before-switch",
            "requirements": _requirements_for("geometry"),
        },
        8,
    ).ok
    assert controller.collected_requirements("geometry")["plate_width"] == 120.0

    current[0] = switched
    assert controller.observe_binding(switched) is False
    assert controller.stage is AuthoringWorkflowStage.STALE
    assert controller.ledger.entries == ()
    with pytest.raises(ValueError, match="clarification_required"):
        controller.collected_requirements("geometry")


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
    controller.record_proposal_state("geometry", ProposalState.SUCCEEDED)
    assert controller.stage is AuthoringWorkflowStage.MESH_READY


def test_a8_provider_payload_rejects_paths_bulk_arrays_and_unsafe_summaries() -> None:
    for payload in (
        {"message": "local file D:\\private\\model.femproj"},
        {"node_ids": [1, 2]},
        {"coordinates": [[0.0, 0.0]]},
        {"model_patch": {"schema_version": "1.0"}},
    ):
        with pytest.raises(ValueError):
            provider_safe_authoring_payload(payload)

    def unsafe(_arguments, _controller):
        raise ValueError("failed at D:\\private\\secret.femproj")

    controller = AuthoringWorkflowController(
        lambda: _context(),
        {"prepare_geometry_proposal": unsafe},
    )
    assert _dispatch(
        controller,
        "set_authoring_requirements",
        {
            "turn_id": "turn-geometry",
            "requirements": _requirements_for("geometry"),
        },
        9,
    ).ok
    result = _dispatch(controller, "prepare_geometry_proposal", {}, 10)
    serialized = result.to_json()
    assert "D:" not in serialized
    assert "private" not in serialized
    assert "ValueError" in serialized


@pytest.mark.parametrize(
    ("operation", "terminal", "expected"),
    [
        ("geometry", ProposalState.FAILED, AuthoringWorkflowStage.GEOMETRY_READY),
        ("mesh", ProposalState.CANCELLED, AuthoringWorkflowStage.MESH_READY),
        ("solve", ProposalState.STALE, AuthoringWorkflowStage.SOLVE_READY),
    ],
)
def test_a8_proposal_terminal_returns_to_the_operation_boundary(
    operation,
    terminal,
    expected,
) -> None:
    controller = AuthoringWorkflowController(lambda: _context(), {})
    controller._pending_operation = operation
    controller._stage = {
        "geometry": AuthoringWorkflowStage.GEOMETRY_PENDING,
        "mesh": AuthoringWorkflowStage.MESH_PENDING,
        "solve": AuthoringWorkflowStage.SOLVE_PENDING,
    }[operation]

    controller.record_proposal_state(operation, terminal, "terminal")
    assert controller.stage is expected
    assert controller.terminal_records[-1].state == terminal.value


def test_a8_dynamic_dispatch_is_serial_and_thread_safe() -> None:
    entered = threading.Event()
    release = threading.Event()

    def blocked(_arguments, _controller):
        entered.set()
        release.wait(timeout=2.0)
        return AuthoringToolOutcome("Applied.", {"state": "succeeded"})

    controller = AuthoringWorkflowController(
        lambda: _context(),
        {"apply_model_definition": blocked},
    )
    controller._stage = AuthoringWorkflowStage.DEFINITIONS_READY
    results = []

    def invoke() -> None:
        results.append(
            _dispatch(
                controller,
                "apply_model_definition",
                {
                    "action": "create_material",
                    "parameters": {
                        "name": "材料-测试",
                        "properties": {"E": 1.0},
                    },
                },
                len(results) + 20,
            )
        )

    first = threading.Thread(target=invoke)
    second = threading.Thread(target=invoke)
    first.start()
    assert entered.wait(timeout=1.0)
    second.start()
    release.set()
    first.join(timeout=2.0)
    second.join(timeout=2.0)

    assert len(results) == 2
    assert all(item.ok for item in results)
