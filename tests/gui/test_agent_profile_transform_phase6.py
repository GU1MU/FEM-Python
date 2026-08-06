"""Phase 6 cross-layer acceptance for Profile transform authoring.

These tests deliberately keep the Fake Provider deterministic: they prove the
local route/guard/controller/GUI seams and the exact geometry results, not the
language ability of a remote model.  The optional real-provider smoke remains
owned by ``tests/test_agent_cloud_smoke.py`` and is gated by its explicit
external-config contract.
"""

from __future__ import annotations

import json
import os

import pytest

from fem.application import ModelSession, UnitContext
from fem.application.preprocessing import generate_fem_model
from fem.geometry import (
    ExtrudedGeometry,
    PathSweptGeometry,
    SketchCircle,
    SketchRectangle,
    describe_recipe_topology,
)
from fem.io.project import decode_project, encode_project
from fem.mesh.settings import MeshSettings
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


def _blank_session_controller():
    session = ModelSession()
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


def _source_face(context_data: dict[str, object]) -> str:
    profiles = context_data.get("profiles")
    assert isinstance(profiles, list) and profiles
    face_id = profiles[0].get("face_id")
    assert isinstance(face_id, str)
    return face_id


def _ring_recipe():
    return planar_sketch_geometry(
        "Ring profile",
        contours=(
            SketchCircle("material", 0.0, 0.0, 5.0),
            SketchCircle("cut", 0.0, 0.0, 2.0),
        ),
    ).recipe


def _rectangle_recipe():
    return planar_sketch_geometry(
        "Sweep profile",
        contours=(SketchRectangle("material", 0.0, 0.0, 1.0, 1.0),),
    ).recipe


def _path(frame_strategy: str = "transport") -> dict[str, object]:
    return {
        "points": [
            {"name": "A", "x": 0.0, "y": 0.0, "z": 0.0},
            {"name": "B", "x": 0.0, "y": 0.0, "z": 2.0},
            {"name": "C", "x": 1.0, "y": 0.0, "z": 3.0},
        ],
        "members": [
            {"name": "AB", "start": "A", "end": "B"},
            {"name": "BC", "start": "B", "end": "C"},
        ],
    }


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


def test_phase6_provider_smoke_case_matrix_is_offline_and_policy_complete() -> None:
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
@pytest.mark.integration
def test_phase6_opt_in_provider_smoke_matrix() -> None:
    """Run six bounded policy prompts only with the existing explicit cloud gate."""

    from tests.test_agent_cloud_smoke import _cloud_smoke_config

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


def test_phase6_contract_matrix_is_bounded_and_provider_discoverable() -> None:
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


def test_phase6_fake_provider_guard_prepare_accept_continuation_uses_new_snapshot(
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
            _text("proposal is waiting for local GUI confirmation"),
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
    events = engine.send_message("拉伸成3d，尺寸任意")
    assert len(provider.requests) == 4
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
    assert dynamic.provider_snapshot.snapshot_generation > before.session_revision

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
    continuation_request = provider.requests[-2]
    state = _current_state_message(continuation_request)
    assert state["authoring_turn_snapshot"]["active_part_dimension"] == 3
    assert state["authoring_turn_snapshot"]["workflow_stage"] == "mesh_ready"
    assert "proposal_terminal" in (continuation_request.messages[-1].content or "")


@pytest.mark.gmsh
@pytest.mark.integration
def test_phase6_ring_dedicated_transform_tet4_tet10_save_reopen_and_hole_lineage():
    session, bridge, controller = _session_controller(_ring_recipe(), name="Phase 6 ring")
    context = controller.dispatch(
        "read_profile_transform_context",
        {"part_id": "P1"},
        ToolExecutionContext("phase6-ring", session.snapshot().session_revision, "read"),
    )
    source = _source_face(context.data)
    prepared = controller.dispatch(
        "prepare_profile_extrusion",
        {
            "part_id": "P1",
            "profile_selection": [source],
            "context_revision": session.snapshot().session_revision,
            "height": 4.0,
        },
        ToolExecutionContext("phase6-ring", session.snapshot().session_revision, "prepare"),
    )
    assert prepared.ok, prepared.summary
    before_accept = session.snapshot()
    receipt = bridge.accept_from_gui_control(prepared.data["proposal_id"])
    assert receipt.state is ProposalState.SUCCEEDED
    controller.record_proposal_state("geometry", receipt.state)
    accepted = session.snapshot()
    assert accepted.session_revision > before_accept.session_revision
    recipe = accepted.parts[0].geometry_recipe
    assert type(recipe) is ExtrudedGeometry
    topology = describe_recipe_topology(recipe)
    assert topology.exact
    assert len(topology.entities_of("body", selectable_only=True)) == 1
    assert any(entity.semantic_role == "sweep.boundary.hole" for entity in topology.entities)
    reopened = decode_project(encode_project(session.prepare_project_save())).snapshot
    assert reopened.parts[0].geometry_recipe == recipe

    for order, expected in ((1, "Tet4"), (2, "Tet10")):
        mesh = generate_fem_model(
            recipe,
            MeshSettings(3.0, order=order, cell_shape="tetrahedron"),
        )
        assert mesh.mesh.elements
        assert {element.type for element in mesh.mesh.elements} == {expected}


@pytest.mark.gmsh
@pytest.mark.integration
def test_phase6_blank_composite_ring_is_one_final_proposal_with_hole_selection(
    tmp_path,
):
    session, bridge, controller = _blank_session_controller()
    dynamic = _ControllerDynamicTools(controller)
    geometry = {
        "kind": "extruded_profiles",
        "profiles": [
            {
                "kind": "circle",
                "center_x": 0.0,
                "center_y": 0.0,
                "radius": 5.0,
                "role": "material",
            },
            {
                "kind": "circle",
                "center_x": 0.0,
                "center_y": 0.0,
                "radius": 2.0,
                "role": "hole",
            },
        ],
        "height": 4.0,
    }
    provider = FakeProvider(
        [
            _tool(
                "blank-prepare",
                "prepare_geometry_proposal",
                {"part_function": "blank-hollow-cylinder", "geometry": geometry},
            ),
            _text("中空圆柱提案等待 GUI 确认"),
            _tool("blank-next", "read_authoring_context", {}),
            _text("几何已接受；现在可以进入网格阶段"),
        ]
    )
    engine = AgentSessionEngine(
        tmp_path / "phase6-blank-composite",
        provider,
        dynamic_tools=dynamic,
    )
    before = session.snapshot()
    events = engine.send_message("创建外半径 5、内半径 2、高度 4 的中空圆柱")
    assert len(provider.requests) == 2
    prepare_events = [
        event
        for event in events
        if event.event is EngineEventType.TOOL_COMPLETED
        and event.data["tool"] == "prepare_geometry_proposal"
    ]
    assert len(prepare_events) == 1
    result = prepare_events[0].data["result"]
    assert result["ok"] is True
    assert session.snapshot() == before
    checkpoint = result["data"]["continuation_checkpoint"]
    proposal_id = checkpoint["proposal_id"]
    assert session.snapshot() == before
    proposal = bridge._records[proposal_id].proposal
    assert proposal.display_summary["dimension"] == 3
    assert proposal.display_summary["expected_entity_count"] == 1
    assert len(proposal.display_summary["expected_new_objects"]) == 1
    summary = proposal.display_summary["summary"]
    assert all(value in summary for value in ("半径=5", "半径=2", "拉伸高=4"))
    receipt = bridge.accept_from_gui_control(proposal_id)
    assert receipt.state is ProposalState.SUCCEEDED
    controller.record_proposal_state("geometry", receipt.state)
    accepted = session.snapshot()
    dynamic.refresh_turn_snapshot(tuple(item.name for item in controller.definitions))
    assert dynamic.provider_snapshot.active_part_dimension == 3
    recipe = accepted.parts[0].geometry_recipe
    assert type(recipe) is ExtrudedGeometry
    topology = describe_recipe_topology(recipe)
    hole_faces = tuple(
        entity.logical_id
        for entity in topology.entities_of("face")
        if entity.semantic_role == "sweep.boundary.hole"
    )
    assert hole_faces
    assert all(topology.entity(face_id).selectable for face_id in hole_faces)
    mesh = generate_fem_model(recipe, MeshSettings(3.0, cell_shape="tetrahedron"))
    assert {element.type for element in mesh.mesh.elements} == {"Tet4"}
    reopened = decode_project(encode_project(session.prepare_project_save())).snapshot
    assert reopened.parts[0].geometry_recipe == recipe
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
    assert not [
        event
        for event in continuation_events
        if event.event is EngineEventType.TOOL_COMPLETED
        and event.data["tool"] == "prepare_geometry_proposal"
    ]


@pytest.mark.gmsh
@pytest.mark.integration
def test_phase6_blank_center_hole_plate_has_canonical_hole_side_and_tet4():
    session, bridge, controller = _blank_session_controller()
    before = session.snapshot()
    result = controller.dispatch(
        "prepare_geometry_proposal",
        {
            "part_function": "center-hole-plate",
            "geometry": {
                "kind": "extruded_profiles",
                "profiles": [
                    {
                        "kind": "rectangle",
                        "x": -5.0,
                        "y": -3.0,
                        "width": 10.0,
                        "height": 6.0,
                    },
                    {
                        "kind": "circle",
                        "center_x": 0.0,
                        "center_y": 0.0,
                        "radius": 1.0,
                    },
                ],
                "height": 2.0,
            },
        },
        ToolExecutionContext(
            "phase6-center-hole",
            before.session_revision,
            "prepare-center-hole",
        ),
    )
    assert result.ok, result.summary
    assert session.snapshot() == before
    proposal = bridge._records[result.data["proposal_id"]].proposal
    assert proposal.display_summary["dimension"] == 3
    assert proposal.display_summary["expected_entity_count"] == 1
    assert len(proposal.display_summary["expected_new_objects"]) == 1
    assert all(
        value in proposal.display_summary["summary"]
        for value in ("10", "6", "1", "2")
    )

    receipt = bridge.accept_from_gui_control(result.data["proposal_id"])
    assert receipt.state is ProposalState.SUCCEEDED
    accepted = session.snapshot()
    assert accepted.session_revision > before.session_revision
    assert len(accepted.parts) == 1
    recipe = accepted.parts[0].geometry_recipe
    assert type(recipe) is ExtrudedGeometry
    topology = describe_recipe_topology(recipe)
    assert topology.exact
    assert tuple(entity.logical_id for entity in topology.entities_of("body")) == (
        "body:domain",
    )
    assert topology.entity("body:domain").selectable
    assert topology.entity("face:bottom").selectable
    assert topology.entity("face:top").selectable
    assert topology.entity("face:bottom").semantic_role == (
        "copy.bottom.sketch.profile"
    )
    assert topology.entity("face:top").semantic_role == (
        "copy.top.sketch.profile"
    )
    hole_side_ids = tuple(
        entity.logical_id
        for entity in topology.entities_of("face")
        if entity.semantic_role == "sweep.boundary.hole"
    )
    assert hole_side_ids == ("face:side/C1",)
    assert topology.entity(hole_side_ids[0]).selectable
    assert hole_side_ids[0] in topology.signature.logical_ids

    mesh = generate_fem_model(
        recipe,
        MeshSettings(2.0, cell_shape="tetrahedron", strict_cell_shape=True),
    )
    assert mesh.mesh.elements
    assert {element.type for element in mesh.mesh.elements} == {"Tet4"}
    reopened = decode_project(encode_project(session.prepare_project_save())).snapshot
    assert reopened.parts[0].geometry_recipe == recipe
    reopened_topology = describe_recipe_topology(reopened.parts[0].geometry_recipe)
    assert reopened_topology.entity("face:side/C1").selectable


def test_phase6_explicit_multi_profile_selection_matches_proposal_part_count():
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


@pytest.mark.gmsh
@pytest.mark.integration
@pytest.mark.parametrize("frame_strategy", ("fixed", "transport"))
def test_phase6_path_dedicated_transform_preserves_order_frame_and_tet_mesh(
    frame_strategy: str,
):
    session, bridge, controller = _session_controller(
        _rectangle_recipe(),
        name=f"Phase 6 path {frame_strategy}",
    )
    context = controller.dispatch(
        "read_profile_transform_context",
        {"part_id": "P1"},
        ToolExecutionContext("phase6-path", 1, f"read-{frame_strategy}"),
    )
    source = _source_face(context.data)
    prepared = controller.dispatch(
        "prepare_profile_path_sweep",
        {
            "part_id": "P1",
            "profile_selection": [source],
            "context_revision": 1,
            "path": _path(frame_strategy),
            "frame_strategy": frame_strategy,
        },
        ToolExecutionContext("phase6-path", 1, f"prepare-{frame_strategy}"),
    )
    assert prepared.ok, prepared.summary
    receipt = bridge.accept_from_gui_control(prepared.data["proposal_id"])
    assert receipt.state is ProposalState.SUCCEEDED
    controller.record_proposal_state("geometry", receipt.state)
    recipe = session.snapshot().parts[0].geometry_recipe
    assert type(recipe) is PathSweptGeometry
    assert recipe.frame_strategy == frame_strategy
    assert tuple(member.name for member in recipe.path.members) == ("AB", "BC")
    assert describe_recipe_topology(recipe).exact
    assert len(describe_recipe_topology(recipe).entities_of("body", selectable_only=True)) == 1
    reopened = decode_project(encode_project(session.prepare_project_save())).snapshot
    assert reopened.parts[0].geometry_recipe == recipe
    mesh = generate_fem_model(recipe, MeshSettings(1.0, cell_shape="tetrahedron"))
    assert mesh.mesh.elements
    assert {element.type for element in mesh.mesh.elements} == {"Tet4"}


def test_phase6_negative_paths_are_atomic_and_stable() -> None:
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
