from __future__ import annotations

import json

import pytest

from fem.application import ModelSession
from fem_agent.authoring import ProposalState
from fem_agent.engine import AgentSessionEngine, EngineEventType
from fem_agent.providers.base import ToolCall
from fem_agent.providers.fake import FakeProvider
from fem_agent.tools.registry import ToolExecutionContext

from tests.helpers.agent_authoring_workflow_fixtures import ControllerDynamicTools
from tests.helpers.agent_planar_construction import (
    make_planar_authoring_controller,
    build_planar_arguments,
    build_rectangle_arguments,
    dispatch_planar_construction,
)
from tests.helpers.agent_provider_fixtures import tool_response, text_response


pytestmark = pytest.mark.local_session


def test_planar_construction_publishes_bounded_context() -> None:
    _bridge, controller = make_planar_authoring_controller(ModelSession())
    assert "prepare_planar_construction_proposal" in {
        definition.name for definition in controller.definitions
    }
    result = controller.dispatch(
        "read_authoring_context", {}, ToolExecutionContext("planar", 0, "context")
    )
    assert result.ok
    capability = result.data["context"]["planar_construction_ir"]
    assert set(capability["output_kinds"]) == {
        "planar", "extrusion", "revolution", "path_sweep"
    }
    assert capability["budgets"]["max_node_count"] == 64
    assert capability["budgets"]["max_pattern_instances"] == 256


def test_disjoint_cutter_is_rejected_without_a_proposal() -> None:
    session = ModelSession()
    bridge, controller = make_planar_authoring_controller(session)
    before = session.snapshot()
    arguments = build_rectangle_arguments()
    construction = arguments["construction"]
    construction["nodes"].extend([
        {"id": "hole", "kind": "circle", "center_x": -5, "center_y": -5, "radius": 1},
        {"id": "cut", "kind": "difference", "base": "plate", "subtract": ["hole"]},
    ])
    construction["result_node_id"] = "cut"
    result = dispatch_planar_construction(controller, arguments=arguments)
    assert not result.ok
    assert result.data["diagnostic"]["code"] == "planar-ir.subtract-no-effect"
    assert bridge._records == {}
    assert session.snapshot() == before


def test_closed_centerline_returns_repair_guidance_without_mutation() -> None:
    session = ModelSession()
    bridge, controller = make_planar_authoring_controller(session)
    initial = dispatch_planar_construction(controller, arguments=build_rectangle_arguments())
    receipt = bridge.accept_from_gui_control(initial.data["proposal_id"])
    assert receipt.state is ProposalState.SUCCEEDED
    controller.record_proposal_state("geometry", receipt.state, receipt.message)
    before = session.snapshot()
    result = controller.dispatch(
        "prepare_geometry_edit",
        {"part_id": "P1", "edit": {
            "operation": "add_path_slot",
            "points": [{"x": 10, "y": 10}, {"x": 20, "y": 20}, {"x": 10, "y": 10}],
            "width": 2, "cap": "butt", "join": "miter",
        }},
        ToolExecutionContext("planar", before.session_revision, "closed-slot"),
    )
    assert not result.ok
    error = result.data["error"]
    assert error["code"] == "planar-ir.invalid-path-stroke"
    assert error["remediation"]["preserve_representation"] is True
    assert session.snapshot() == before


def test_fake_provider_uses_one_card_and_continues_from_new_snapshot(
    tmp_path,
) -> None:
    session = ModelSession()
    bridge, controller = make_planar_authoring_controller(session)
    dynamic = ControllerDynamicTools(controller)
    provider = FakeProvider(
        [
            tool_response(
                ToolCall(
                    "prepare-ir",
                    "prepare_planar_construction_proposal",
                    build_rectangle_arguments(),
                ),
            )
        ]
    )
    engine = AgentSessionEngine(
        tmp_path / "phase3-fake-provider",
        provider,
        dynamic_tools=dynamic,
    )
    before = session.snapshot()

    events = engine.send_message("创建二维矩形板")

    assert session.snapshot() == before
    assert [
        event.data["tool"]
        for event in events
        if event.event is EngineEventType.TOOL_STARTED
    ] == ["prepare_planar_construction_proposal"]
    assert len(bridge._records) == 1
    audit = json.loads(engine._audit_path().read_text(encoding="utf-8"))
    assert audit["entries"][-1]["tool_call_flags"]["called_tool_names"] == [
        "prepare_planar_construction_proposal"
    ]
    proposal_id, record = next(iter(bridge._records.items()))
    assert record.state is ProposalState.PENDING_CONFIRMATION
    checkpoint = next(
        event.data["result"]["data"]["continuation_checkpoint"]
        for event in events
        if event.event is EngineEventType.TOOL_COMPLETED
    )

    receipt = bridge.accept_from_gui_control(proposal_id)
    controller.record_proposal_state("geometry", receipt.state, receipt.message)
    dynamic.refresh_turn_snapshot(tuple(item.name for item in controller.definitions))
    provider.queue(
        tool_response(ToolCall("read-new", "read_authoring_context", {})),
        text_response("二维部件已进入后续建模阶段。"),
    )
    continuation = engine.continue_after_proposal(
        proposal_id,
        checkpoint["proposal_hash"],
        checkpoint["source_turn_id"],
        checkpoint["model_revision"],
        receipt.state.value,
        receipt.message,
    )

    assert session.snapshot().parts[0].dimension == 2
    assert dynamic.provider_snapshot.active_part_dimension == 2
    assert any(
        event.event is EngineEventType.TOOL_COMPLETED
        and event.data["tool"] == "read_authoring_context"
        for event in continuation
    )
    assert len([item for item in engine._history if item.role == "user"]) == 1
    assert not any(
        "confirm" in str(event.data.get("text", "")).casefold()
        or "确认" in str(event.data.get("text", ""))
        for event in continuation
        if event.event is EngineEventType.MESSAGE_DELTA
    )


def test_invalid_ir_fails_without_a_card_or_model_change() -> None:
    session = ModelSession()
    bridge, controller = make_planar_authoring_controller(session)
    before = session.snapshot()
    arguments = build_planar_arguments()
    arguments["construction"]["nodes"][-1]["subtract"] = ["missing"]

    result = controller.dispatch(
        "prepare_planar_construction_proposal",
        arguments,
        ToolExecutionContext("phase3-planar", 0, "invalid"),
    )

    assert not result.ok
    assert result.data["diagnostic"]["code"] == "planar-ir.reference-missing"
    assert result.data["diagnostic"]["model_unchanged"] is True
    assert "proposal_id" not in result.data
    assert not bridge._records
    assert session.snapshot() == before
