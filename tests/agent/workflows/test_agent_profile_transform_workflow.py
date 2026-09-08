"""Provider and controller contracts for profile transform authoring.

These tests deliberately keep the Fake Provider deterministic: they prove the
local route/guard/controller/GUI seams, not the
language ability of a remote model.  The optional real-provider smoke remains
gated by the explicit external-config contract shared with the agent tests.
"""

from __future__ import annotations

import json
import os

import pytest

from fem.application import ModelSession, UnitContext
from fem.geometry import SketchCircle, SketchRectangle
from fem_agent.authoring import ProposalState
from fem_agent.authoring_runtime import (
    AUTHORING_TURN_SNAPSHOT_MAX_BYTES,
    AuthoringTurnSnapshot,
)
from fem_agent.engine import AgentSessionEngine, EngineEventType
from fem_agent.diagnostics import PROFILE_TRANSFORM_DIAGNOSTIC_CODES
from fem_agent.geometry_authoring import planar_sketch_geometry
from fem_agent.providers.base import AssistantMessage, ProviderResponse, ToolCall
from fem_agent.providers.base import ToolDefinition
from fem_agent.providers.deepseek import DeepSeekProvider
from fem_agent.providers.fake import FakeProvider
from fem_agent.result_authoring import AgentResultQueryBridge
from fem_agent.routing import geometry_route_hint
from fem_agent.tools.registry import ToolExecutionContext
from fem_gui.agent_authoring import (
    AgentAuthoringBridge,
    SessionGeometryAuthoringPort,
    SessionResultQueryPort,
    create_session_authoring_workflow_controller,
)


pytestmark = pytest.mark.local_session


def _text(value: str) -> ProviderResponse:
    return ProviderResponse(
        AssistantMessage("assistant", content=value),
        finish_reason="stop",
    )


def _tool(call_id: str, name: str, arguments: dict[str, object]) -> ProviderResponse:
    return ProviderResponse(
        AssistantMessage(
            "assistant",
            tool_calls=(ToolCall(call_id, name, arguments),),
        ),
        finish_reason="tool_calls",
    )


def _refusal() -> ProviderResponse:
    return _text("拉伸不受支持；必须先生成网格。")


class _ControllerDynamicTools:
    """Provider-facing adapter around the real GUI authoring controller."""

    def __init__(self, controller) -> None:
        self.controller = controller
        self._snapshot = controller.set_published_tool_names(
            tuple(item.name for item in controller.definitions)
        )

    @property
    def definitions(self):
        return tuple(self.controller.definitions)

    @property
    def provider_snapshot(self) -> AuthoringTurnSnapshot:
        return self._snapshot

    def refresh_turn_snapshot(self, published_tool_names=()):
        names = tuple(published_tool_names) or tuple(
            item.name for item in self.controller.definitions
        )
        self._snapshot = self.controller.set_published_tool_names(names)
        return self._snapshot

    def dispatch(self, name, arguments, context):
        return self.controller.dispatch(name, arguments, context)


def _session_controller(recipe, *, name: str = "Phase 6 native"):
    session = ModelSession()
    session.create_native_project_with_first_part(
        name,
        UnitContext("mm", "N", "MPa"),
        recipe,
        part_name="Sketch",
    )
    holder: dict[str, object] = {}

    def refresh() -> None:
        bridge = holder["bridge"]
        assert isinstance(bridge, AgentAuthoringBridge)
        bridge.bind_snapshot(session.snapshot())
        controller = holder.get("controller")
        if controller is not None:
            controller.observe_binding(bridge.context)

    bridge = AgentAuthoringBridge(SessionGeometryAuthoringPort(session, refresh))
    holder["bridge"] = bridge
    bridge.bind_snapshot(session.snapshot())
    controller = create_session_authoring_workflow_controller(
        session,
        bridge,
        AgentResultQueryBridge(SessionResultQueryPort(session)),
    )
    holder["controller"] = controller
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


_PHASE6_PROVIDER_SMOKE_CASES = (
    ("拉伸成3d", "extrude_profiles", False),
    ("把这个截面加厚到 20 mm", "extrude_profiles", True),
    ("extrude this profile by 10 mm", "extrude_profiles", True),
    ("沿 A-B-C 这条路径扫掠", "path_sweep_profile", True),
    ("做扫掠六面体网格", "swept_mesh", True),
    ("这个功能支持吗", "read_only", True),
)


_PHASE6_PROVIDER_TOOLS = (
    ToolDefinition(
        "read_profile_transform_context",
        "Read the bounded canonical Profile transform context for Part P1.",
        {
            "type": "object",
            "properties": {"part_id": {"type": "string"}},
            "required": ["part_id"],
            "additionalProperties": False,
        },
    ),
    ToolDefinition(
        "prepare_profile_extrusion",
        "Prepare a positive GUI-confirmed Profile extrusion proposal.",
        {
            "type": "object",
            "properties": {
                "part_id": {"type": "string"},
                "profile_selection": {
                    "type": "string",
                    "const": "unique_material_profile",
                },
                "height": {"type": "number", "exclusiveMinimum": 0},
            },
            "required": ["part_id", "profile_selection", "height"],
            "additionalProperties": False,
        },
    ),
    ToolDefinition(
        "prepare_profile_path_sweep",
        "Prepare a GUI-confirmed ordered open path sweep proposal.",
        {
            "type": "object",
            "properties": {
                "part_id": {"type": "string"},
                "profile_selection": {
                    "type": "string",
                    "const": "unique_material_profile",
                },
                "path": {"type": "object"},
                "frame_strategy": {
                    "type": "string",
                    "enum": ["fixed", "transport"],
                },
            },
            "required": [
                "part_id",
                "profile_selection",
                "path",
                "frame_strategy",
            ],
            "additionalProperties": False,
        },
    ),
    ToolDefinition(
        "prepare_mesh_proposal",
        "Prepare a GUI-confirmed native tetrahedral mesh proposal.",
        {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
    ),
)


def test_provider_smoke_case_matrix_is_offline_and_policy_complete() -> None:
    assert [item[0] for item in _PHASE6_PROVIDER_SMOKE_CASES] == [
        "拉伸成3d",
        "把这个截面加厚到 20 mm",
        "extrude this profile by 10 mm",
        "沿 A-B-C 这条路径扫掠",
        "做扫掠六面体网格",
        "这个功能支持吗",
    ]
    for prompt, operation, complete in _PHASE6_PROVIDER_SMOKE_CASES:
        hint = geometry_route_hint(prompt)
        if operation == "read_only":
            assert hint is None
            continue
        assert hint is not None
        assert hint.requested_operation == operation
        if operation == "swept_mesh":
            assert hint.intent_kind == "meshing"
            assert hint.required_probe_tool is None
        else:
            assert hint.required_probe_tool == "read_profile_transform_context"
            assert hint.required_prepare_tool == (
                "prepare_profile_extrusion"
                if operation == "extrude_profiles"
                else "prepare_profile_path_sweep"
            )
        assert complete is (operation != "extrude_profiles" or prompt != "拉伸成3d")


def _smoke_tool_result(call: ToolCall) -> AssistantMessage:
    if call.name == "read_profile_transform_context":
        payload = {
            "dimension": 2,
            "material_profile_count": 1,
            "profiles": [{"face_id": "face:profile/p1"}],
            "operations": {"extrusion": {"available": True}},
        }
    elif call.name == "prepare_profile_extrusion":
        payload = {
            "state": "pending_confirmation",
            "proposal_id": "smoke-extrusion",
        }
    elif call.name == "prepare_profile_path_sweep":
        payload = {
            "state": "pending_confirmation",
            "proposal_id": "smoke-path",
        }
    else:
        payload = {"state": "pending_confirmation", "proposal_id": "smoke-mesh"}
    return AssistantMessage(
        "tool",
        content=json.dumps(payload, ensure_ascii=False, sort_keys=True),
        tool_call_id=call.call_id,
    )


@pytest.mark.cloud
def test_opt_in_provider_smoke_matrix() -> None:
    """Run six bounded policy prompts only with the existing explicit cloud gate."""

    from tests.helpers.agent_session_fixtures import _cloud_smoke_config

    try:
        config, reason = _cloud_smoke_config(os.environ)
    except Exception as error:
        pytest.fail(f"invalid cloud smoke configuration: {type(error).__name__}")
    if config is None:
        pytest.skip(reason)

    provider = DeepSeekProvider(
        config.provider_config(),
        environ=config.provider_environment({}),
    )
    system = (
        "You are a deterministic FEM authoring acceptance target. The current "
        "native context is one editable 2D Part P1 with exactly one material "
        "Profile. For an explicit transform, call "
        "read_profile_transform_context before the operation-specific prepare "
        "tool; geometry never requires a mesh. Use GUI-confirmed proposals only. "
        "For swept hex mesh, use prepare_mesh_proposal and never a geometry sweep. "
        "For a support question, answer read-only and never create a proposal."
    )
    for prompt, operation, complete in _PHASE6_PROVIDER_SMOKE_CASES:
        messages = [AssistantMessage("system", system), AssistantMessage("user", prompt)]
        called: list[str] = []
        for _round in range(3):
            response = provider.complete(messages, _PHASE6_PROVIDER_TOOLS)
            calls = tuple(response.message.tool_calls)
            called.extend(call.name for call in calls)
            messages.append(response.message)
            if not calls:
                break
            messages.extend(_smoke_tool_result(call) for call in calls)
        if operation == "read_only":
            assert not {
                "prepare_profile_extrusion",
                "prepare_profile_path_sweep",
                "prepare_mesh_proposal",
            }.intersection(called), prompt
        elif operation == "swept_mesh":
            assert "prepare_mesh_proposal" in called, prompt
            assert not {
                "prepare_profile_extrusion",
                "prepare_profile_path_sweep",
            }.intersection(called), prompt
        else:
            assert "read_profile_transform_context" in called, prompt
            expected = (
                "prepare_profile_extrusion"
                if operation == "extrude_profiles"
                else "prepare_profile_path_sweep"
            )
            if complete:
                assert expected in called, prompt
            assert "prepare_mesh_proposal" not in called, prompt


def test_contract_matrix_is_bounded_and_provider_discoverable() -> None:
    session, _bridge, controller = _session_controller(_ring_recipe())
    dynamic = _ControllerDynamicTools(controller)
    definitions = {item.name: item for item in dynamic.definitions}

    expected = {
        "read_profile_transform_context",
        "prepare_profile_extrusion",
        "prepare_profile_revolution",
        "prepare_profile_path_sweep",
    }
    assert expected <= definitions.keys()
    assert all(
        definitions[name].parameters.get("additionalProperties") is False
        for name in expected
    )
    assert definitions["read_profile_transform_context"].parameters["required"] == [
        "part_id"
    ]
    assert definitions["prepare_profile_extrusion"].parameters["required"] == [
        "part_id",
        "profile_selection",
        "height",
    ]
    assert definitions["prepare_profile_path_sweep"].parameters["required"] == [
        "part_id",
        "profile_selection",
        "path",
        "frame_strategy",
    ]
    generic = definitions["prepare_geometry_edit"].parameters
    assert all(
        variant["properties"]["operation"].get("const")
        not in {"extrude_profiles", "revolve_profile", "path_sweep_profile"}
        for variant in generic["properties"]["edit"]["oneOf"]
    )

    snapshot = dynamic.provider_snapshot
    payload = snapshot.to_provider_dict()
    assert snapshot.available is True
    assert snapshot.active_part_dimension == 2
    assert expected <= set(snapshot.published_tool_names)
    assert len(json.dumps(payload, ensure_ascii=False).encode("utf-8")) <= (
        AUTHORING_TURN_SNAPSHOT_MAX_BYTES
    )
    assert snapshot == controller.turn_snapshot

    assert geometry_route_hint("拉伸成3d") == geometry_route_hint(
        "extrude this profile"
    )
    assert geometry_route_hint("沿路径扫掠 A-B-C").required_prepare_tool == (
        "prepare_profile_path_sweep"
    )
    assert geometry_route_hint("做扫掠六面体网格").intent_kind == "meshing"

    for code in PROFILE_TRANSFORM_DIAGNOSTIC_CODES:
        # The Phase 5 diagnostic contract is the stable recovery surface used
        # by this phase's negative E2E paths.
        result = controller.dispatch(
            "read_profile_transform_context",
            {"part_id": "P404"},
            ToolExecutionContext(
                "phase6-contract",
                session.snapshot().session_revision,
                f"diag-{code.replace('.', '-')}",
            ),
        )
        assert result.data["diagnostic"]["code"] == (
            "profile-transform.part-not-found"
        )
        assert set(("message", "retryable", "required_fields", "preserve_draft")) <= set(
            result.data["diagnostic"]
        )


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
    assert len(provider.requests) == 3
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
    assert len([item for item in engine._history if item.role == "user"]) == 1
    continuation_requests = provider.requests[request_count:]
    assert continuation_requests
    terminal_prefix = "Local GUI proposal terminal (trusted control result): "
    for request in continuation_requests:
        state = _current_state_message(request)
        assert state["authoring_turn_snapshot"]["active_part_dimension"] == 3
        assert state["authoring_turn_snapshot"]["workflow_stage"] == "mesh_ready"
        terminals = [
            message.content[len(terminal_prefix):]
            for message in request.messages
            if message.role == "system"
            and (message.content or "").startswith(terminal_prefix)
        ]
        assert len(terminals) == 1
        terminal, _ = json.JSONDecoder().raw_decode(terminals[0])
        assert terminal["kind"] == "proposal_terminal"
        assert terminal["proposal_id"] == proposal_id
        assert terminal["proposal_hash"] == checkpoint["proposal_hash"]
        assert terminal["source_turn_id"] == checkpoint["source_turn_id"]
        assert terminal["model_revision"] == checkpoint["model_revision"]
        assert terminal["status"] == "succeeded"


@pytest.mark.usefixtures("real_gmsh")
def test_explicit_multi_profile_selection_matches_proposal_part_count():
    recipe = planar_sketch_geometry(
        "Two independent material profiles",
        contours=(
            SketchRectangle("material", 0.0, 0.0, 1.0, 1.0),
            SketchRectangle("material", 3.0, 0.0, 1.0, 1.0),
        ),
    ).recipe
    session, bridge, controller = _session_controller(recipe, name="Phase 6 multi")
    before = session.snapshot()
    context = controller.dispatch(
        "read_profile_transform_context",
        {"part_id": "P1"},
        ToolExecutionContext("phase6-multi", before.session_revision, "read-multi"),
    )
    profiles = context.data["profiles"]
    assert isinstance(profiles, list) and len(profiles) == 2
    candidates = [item["face_id"] for item in profiles]
    prepared = controller.dispatch(
        "prepare_profile_extrusion",
        {
            "part_id": "P1",
            "profile_selection": candidates,
            "context_revision": before.session_revision,
            "height": 2.0,
        },
        ToolExecutionContext("phase6-multi", before.session_revision, "prepare-multi"),
    )
    assert prepared.ok, prepared.summary
    assert session.snapshot() == before
    proposal = bridge._records[prepared.data["proposal_id"]].proposal
    assert proposal.display_summary["expected_part_count"] == len(candidates)
    assert proposal.display_summary["expected_entity_count"] == len(candidates)
    receipt = bridge.accept_from_gui_control(prepared.data["proposal_id"])
    assert receipt.state is ProposalState.SUCCEEDED
    controller.record_proposal_state("geometry", receipt.state)
    accepted = session.snapshot()
    assert len(accepted.parts) == len(candidates)
    assert all(part.dimension == 3 for part in accepted.parts)


def test_negative_paths_are_atomic_and_stable() -> None:
    session, bridge, controller = _session_controller(_ring_recipe())
    before = session.snapshot()
    zero = controller.dispatch(
        "prepare_profile_extrusion",
        {
            "part_id": "P1",
            "profile_selection": "unique_material_profile",
            "height": 0.0,
        },
        ToolExecutionContext("phase6-negative", 1, "zero"),
    )
    assert zero.data["diagnostic"]["code"] == "profile-transform.nonpositive-height"
    assert session.snapshot() == before

    ambiguous_recipe = planar_sketch_geometry(
        "Two profiles",
        contours=(
            SketchRectangle("material", 0.0, 0.0, 1.0, 1.0),
            SketchRectangle("material", 3.0, 0.0, 1.0, 1.0),
        ),
    ).recipe
    ambiguous, _ambiguous_bridge, ambiguous_controller = _session_controller(
        ambiguous_recipe,
        name="Phase 6 ambiguity",
    )
    result = ambiguous_controller.dispatch(
        "prepare_profile_extrusion",
        {
            "part_id": "P1",
            "profile_selection": "unique_material_profile",
            "height": 2.0,
        },
        ToolExecutionContext("phase6-negative", 1, "ambiguous"),
    )
    assert result.data["diagnostic"]["code"] == (
        "profile-transform.ambiguous-material-profiles"
    )
    assert result.data["diagnostic"]["candidates"]
    assert ambiguous.snapshot().session_revision == 1

    context = controller.dispatch(
        "read_profile_transform_context",
        {"part_id": "P1"},
        ToolExecutionContext("phase6-negative", 1, "hole-read"),
    )
    profiles = context.data["profiles"]
    assert isinstance(profiles, list) and len(profiles) == 1
    # Hole boundaries are lineage, not selectable material Profiles.  A
    # provider-supplied hole-like logical ID must therefore be rejected.
    hole_id = "face:hole"
    hole = controller.dispatch(
        "prepare_profile_extrusion",
        {
            "part_id": "P1",
            "profile_selection": [hole_id],
            "context_revision": 1,
            "height": 2.0,
        },
        ToolExecutionContext("phase6-negative", 1, "hole-only"),
    )
    assert hole.data["diagnostic"]["code"] == "profile-transform.invalid-source-id"
    assert session.snapshot() == before

    stale = controller.dispatch(
        "prepare_profile_extrusion",
        {
            "part_id": "P1",
            "profile_selection": "unique_material_profile",
            "height": 2.0,
        },
        ToolExecutionContext("phase6-negative", 1, "stale-prepare"),
    )
    assert stale.ok
    session.rename_native_part("P1", "Changed")
    stale_state = session.snapshot()
    failed = bridge.accept_from_gui_control(stale.data["proposal_id"])
    assert failed.state is ProposalState.FAILED
    assert session.snapshot() == stale_state
