from __future__ import annotations

import json

import pytest

from fem.application import ModelSession, UnitContext
from fem.geometry import SketchCircle
from fem_agent.authoring import ProposalState
from fem_agent.engine import AgentSessionEngine, EngineEventType
from fem_agent.geometry_authoring import planar_sketch_geometry
from fem_agent.providers.base import ProviderResponse
from fem_agent.providers.fake import FakeProvider


from tests.helpers.agent_planar_construction import (
    ControllerDynamicTools as _ControllerDynamicTools,
    make_planar_authoring_controller,
    text_response as _text,
    tool_response as _tool,
)


pytestmark = pytest.mark.local_session


def _refusal() -> ProviderResponse:
    return _text("拉伸不受支持；必须先生成网格。")


def _session_controller(recipe, *, name: str = "Phase 6 native"):
    session = ModelSession()
    session.create_native_project_with_first_part(
        name,
        UnitContext("mm", "N", "MPa"),
        recipe,
        part_name="Sketch",
    )
    bridge, controller = make_planar_authoring_controller(session)
    return session, bridge, controller


def _ring_recipe():
    return planar_sketch_geometry(
        "Ring profile",
        contours=(
            SketchCircle("material", 0.0, 0.0, 5.0),
            SketchCircle("cut", 0.0, 0.0, 2.0),
        ),
    ).recipe


def _current_state_message(request) -> dict[str, object]:
    content = next(
        item.content
        for item in request.messages
        if item.role == "system"
        and isinstance(item.content, str)
        and item.content.startswith("Current local state")
    )
    assert isinstance(content, str)
    return json.loads(content.split(": ", 1)[1])


@pytest.mark.usefixtures("real_gmsh")
def test_fake_provider_guard_prepare_accept_continuation_uses_new_snapshot(
    tmp_path,
) -> None:
    session, bridge, controller = _session_controller(_ring_recipe())
    dynamic = _ControllerDynamicTools(controller)
    provider = FakeProvider(
        [
            _refusal(),
            _tool("read", "read_profile_transform_context", {"part_id": "P1"}),
            _tool(
                "prepare",
                "prepare_profile_extrusion",
                {
                    "part_id": "P1",
                    "profile_selection": "unique_material_profile",
                    "height": 4.0,
                },
            ),
            _tool("next", "read_authoring_context", {}),
            _text("geometry accepted; mesh stage is ready"),
        ]
    )
    engine = AgentSessionEngine(
        tmp_path / "phase6-fake-provider",
        provider,
        dynamic_tools=dynamic,
    )

    before = session.snapshot()
    before_snapshot_generation = dynamic.provider_snapshot.snapshot_generation
    events = engine.send_message("拉伸成3d，尺寸任意")
    assert [
        event.data["tool"]
        for event in events
        if event.event is EngineEventType.TOOL_STARTED
    ] == ["read_profile_transform_context", "prepare_profile_extrusion"]
    assert session.snapshot() == before

    prepare_event = next(
        event
        for event in events
        if event.event is EngineEventType.TOOL_COMPLETED
        and event.data["tool"] == "prepare_profile_extrusion"
    )
    checkpoint = prepare_event.data["result"]["data"]["continuation_checkpoint"]
    proposal_id = checkpoint["proposal_id"]
    assert bridge._records[proposal_id].state is ProposalState.PENDING_CONFIRMATION

    receipt = bridge.accept_from_gui_control(proposal_id)
    assert receipt.state is ProposalState.SUCCEEDED
    controller.record_proposal_state("geometry", receipt.state, receipt.message)
    assert controller.stage.value == "mesh_ready"
    assert session.snapshot().parts[0].dimension == 3
    dynamic.refresh_turn_snapshot(tuple(item.name for item in controller.definitions))
    assert dynamic.provider_snapshot.active_part_dimension == 3
    assert dynamic.provider_snapshot.snapshot_generation > before_snapshot_generation

    request_count = len(provider.requests)
    continuation_events = engine.continue_after_proposal(
        proposal_id,
        checkpoint["proposal_hash"],
        checkpoint["source_turn_id"],
        int(checkpoint["model_revision"]),
        receipt.state.value,
        receipt.message,
    )
    assert any(
        event.event is EngineEventType.TOOL_COMPLETED
        and event.data["tool"] == "read_authoring_context"
        for event in continuation_events
    )
    continuation_requests = provider.requests[request_count:]
    assert continuation_requests
    state = _current_state_message(continuation_requests[-1])
    assert state["authoring_turn_snapshot"]["active_part_dimension"] == 3
    assert state["authoring_turn_snapshot"]["workflow_stage"] == "mesh_ready"
