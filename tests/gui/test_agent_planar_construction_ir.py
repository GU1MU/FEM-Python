from __future__ import annotations

from copy import deepcopy
import json

import pytest

from fem import geometry as geometry_runtime
from fem.application import ModelSession, derive_feature_history
from fem.geometry import BooleanGeometry, SketchGeometry
from fem_agent.authoring import AgentProposal, ProposalState
from fem_agent.authoring_runtime import (
    AuthoringTurnSnapshot,
    AuthoringWorkflowStage,
)
from fem_agent.engine import AgentSessionEngine, EngineEventType
from fem_agent.providers.base import AssistantMessage, ProviderResponse, ToolCall
from fem_agent.providers.fake import FakeProvider
from fem_agent.result_authoring import AgentResultQueryBridge
from fem_agent.tools.registry import ToolExecutionContext
from fem_gui.agent_authoring import (
    AgentAuthoringBridge,
    SessionGeometryAuthoringPort,
    SessionResultQueryPort,
    create_session_authoring_workflow_controller,
)
from tests.fixtures.planar_construction_phase0 import EXPECTED_H_CONSTRUCTION


def _tool(call_id: str, name: str, arguments: dict[str, object]) -> ProviderResponse:
    return ProviderResponse(
        AssistantMessage(
            "assistant",
            tool_calls=(ToolCall(call_id, name, arguments),),
        ),
        finish_reason="tool_calls",
    )


def _text(value: str) -> ProviderResponse:
    return ProviderResponse(
        AssistantMessage("assistant", content=value),
        finish_reason="stop",
    )


class _ControllerDynamicTools:
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


def _controller(session: ModelSession):
    holder: dict[str, object] = {}

    def refresh() -> None:
        bridge.bind_snapshot(session.snapshot())
        controller = holder.get("controller")
        if controller is not None:
            controller.observe_binding(bridge.context)  # type: ignore[arg-type]

    bridge = AgentAuthoringBridge(SessionGeometryAuthoringPort(session, refresh))
    bridge.bind_snapshot(session.snapshot())
    controller = create_session_authoring_workflow_controller(
        session,
        bridge,
        AgentResultQueryBridge(SessionResultQueryPort(session)),
    )
    holder["controller"] = controller
    return bridge, controller


def _h_plate_construction() -> dict[str, object]:
    return deepcopy(EXPECTED_H_CONSTRUCTION)


def _arguments() -> dict[str, object]:
    return {
        "part_function": "带组合槽和四角孔的二维板",
        "construction": _h_plate_construction(),
        "output": "planar",
    }


def _dispatch(controller, *, key: str = "phase3"):
    return controller.dispatch(
        "prepare_planar_construction_proposal",
        _arguments(),
        ToolExecutionContext("phase3-planar", 0, key),
    )


def test_phase3_publishes_strict_schema_and_bounded_context() -> None:
    _bridge, controller = _controller(ModelSession())
    definition = next(
        item
        for item in controller.definitions
        if item.name == "prepare_planar_construction_proposal"
    )
    outputs = definition.parameters["properties"]["output"]["oneOf"]
    assert outputs[0] == {"const": "planar"}
    assert {item["properties"]["kind"]["const"] for item in outputs[1:]} == {
        "planar",
        "extrusion",
        "revolution",
        "path_sweep",
    }
    planar_object = next(
        item
        for item in outputs[1:]
        if item["properties"]["kind"]["const"] == "planar"
    )
    assert planar_object["required"] == ["kind"]
    assert planar_object["additionalProperties"] is False
    construction = definition.parameters["properties"]["construction"]
    variants = construction["properties"]["nodes"]["items"]["oneOf"]
    assert {item["properties"]["kind"]["const"] for item in variants} == {
        "rectangle",
        "circle",
        "polygon",
        "path_stroke",
        "union",
        "difference",
        "intersection",
        "translate",
        "rotate",
        "mirror",
        "linear_pattern",
        "rectangular_pattern",
        "circular_pattern",
    }
    assert all(item["additionalProperties"] is False for item in variants)
    rectangle = next(
        item for item in variants if item["properties"]["kind"]["const"] == "rectangle"
    )
    circle = next(
        item for item in variants if item["properties"]["kind"]["const"] == "circle"
    )
    assert "lower-left" in rectangle["properties"]["x"]["description"]
    assert "not the center" in rectangle["properties"]["y"]["description"]
    assert "never diameter" in circle["properties"]["radius"]["description"]

    context = controller.dispatch(
        "read_authoring_context",
        {},
        ToolExecutionContext("phase3-planar", 0, "context"),
    )
    capability = context.data["context"]["planar_construction_ir"]
    assert capability["schema_version"] == 1
    assert capability["plane"] == "XY"
    assert capability["budgets"]["max_node_count"] == 64
    assert capability["budgets"]["max_pattern_instances"] == 256
    assert capability["coordinate_conventions"] == {
        "rectangle_anchor": "lower_left",
        "rectangle_extent": "x..x+width, y..y+height",
        "circle_position": "center_x, center_y",
        "circle_size": "radius; use diameter/2 when the request gives a diameter",
        "pattern_seed": "included_as_instance_zero",
    }
    serialized = json.dumps(capability).casefold()
    assert "gmsh" not in serialized
    assert "occ" not in serialized


def test_phase3_rejects_misanchored_cutters_before_presenting_a_card() -> None:
    session = ModelSession()
    bridge, controller = _controller(session)
    before = session.snapshot()
    construction = {
        "schema_version": 1,
        "name": "misanchored_plate",
        "plane": "XY",
        "nodes": [
            {
                "id": "plate",
                "kind": "rectangle",
                "x": 0,
                "y": 0,
                "width": 100,
                "height": 300,
            },
            {
                "id": "centered_as_if_x_y_were_centers",
                "kind": "rectangle",
                "x": -20,
                "y": -40,
                "width": 40,
                "height": 80,
            },
            {
                "id": "outside_hole",
                "kind": "circle",
                "center_x": -35,
                "center_y": -135,
                "radius": 3,
            },
            {
                "id": "result",
                "kind": "difference",
                "base": "plate",
                "subtract": ["centered_as_if_x_y_were_centers", "outside_hole"],
            },
        ],
        "result_node_id": "result",
    }

    result = controller.dispatch(
        "prepare_planar_construction_proposal",
        {
            "part_function": "错误锚点回归样例",
            "construction": construction,
            "output": "planar",
        },
        ToolExecutionContext("phase3-planar", 0, "misanchored"),
    )

    assert result.ok is False
    assert result.data["diagnostic"]["code"] == "planar-ir.subtract-no-effect"
    assert result.data["diagnostic"]["node_id"] in {
        "centered_as_if_x_y_were_centers",
        "outside_hole",
    }
    assert result.data["diagnostic"]["model_unchanged"] is True
    assert bridge._records == {}
    assert session.snapshot() == before


def test_phase3_h_plate_is_proven_before_one_card_and_accepts_one_strict_part() -> None:
    session = ModelSession()
    bridge, controller = _controller(session)
    before = session.snapshot()

    result = _dispatch(controller)

    assert result.ok, result.summary
    assert session.snapshot() == before
    assert len(bridge._records) == 1
    assert result.data["state"] == ProposalState.PENDING_CONFIRMATION.value
    assert result.data["authoring_path"] == "planar_construction_ir_v1"
    assert result.data["construction_summary"]["node_kind_counts"] == {
        "circle": 1,
        "difference": 1,
        "rectangle": 4,
        "rectangular_pattern": 1,
        "union": 2,
    }
    assert result.data["proof_summary"]["equivalent"] is True
    assert result.data["proof_summary"]["material_profile_count"] == 1
    assert result.data["proof_summary"]["hole_count"] == 5
    assert set(result.data["proposal_view"]) == {
        "proposal_id",
        "proposal_hash",
        "proposal_kind",
        "title",
        "summary",
        "impact",
        "confirm_label",
        "target_document_id",
        "target_session_id",
        "base_session_revision",
    }

    proposal = bridge._records[result.data["proposal_id"]].proposal
    evidence = proposal.preconditions["local_evidence"]
    assert evidence["construction_digest"].startswith(
        result.data["construction_digest_short"]
    )
    assert evidence["recipe_proof_digest"].startswith(
        result.data["recipe_proof_digest_short"]
    )

    def proposal_hash_with(preconditions: object) -> str:
        return AgentProposal.create(
            proposal_id=proposal.proposal_id,
            proposal_kind=proposal.proposal_kind,
            agent_session_id=proposal.agent_session_id,
            turn_id=proposal.turn_id,
            source_tool_call_ids=proposal.source_tool_call_ids,
            target_document_id=proposal.target_document_id,
            target_session_id=proposal.target_session_id,
            base_session_revision=proposal.base_session_revision,
            draft_revision=proposal.draft_revision,
            operations=proposal.operations,
            preconditions=preconditions,
            expected_changes=proposal.expected_changes,
            invalidation_impact=proposal.invalidation_impact,
            display_summary=proposal.display_summary,
        ).proposal_hash

    changed_ir = deepcopy(proposal.preconditions)
    changed_ir["local_evidence"]["construction_digest"] = "0" * 64
    changed_proof = deepcopy(proposal.preconditions)
    changed_proof["local_evidence"]["recipe_proof_digest"] = "0" * 64
    assert proposal_hash_with(changed_ir) != proposal.proposal_hash
    assert proposal_hash_with(changed_proof) != proposal.proposal_hash

    receipt = bridge.accept_from_gui_control(result.data["proposal_id"])

    assert receipt.state is ProposalState.SUCCEEDED
    accepted = session.snapshot()
    assert accepted.session_revision == before.session_revision + 1
    assert len(accepted.parts) == 1
    recipe = accepted.parts[0].geometry_recipe
    assert type(recipe) is BooleanGeometry
    assert recipe.planar_context is not None and recipe.planar_context.proven
    assert [feature.kind for feature in derive_feature_history(recipe)] == [
        "sketch",
        "cut",
        "cut",
    ]
    assert type(recipe.object_geometry) is BooleanGeometry
    assert type(recipe.object_geometry.object_geometry) is SketchGeometry
    h_tool = recipe.object_geometry.tool_geometry
    hole_tool = recipe.tool_geometry
    assert type(h_tool) is SketchGeometry
    assert type(hole_tool) is SketchGeometry
    assert (
        sum(
            profile.is_material
            for profile in geometry_runtime.analyze_sketch_profiles(h_tool).profiles
        )
        == 1
    )
    assert (
        sum(
            profile.is_material
            for profile in geometry_runtime.analyze_sketch_profiles(hole_tool).profiles
        )
        == 4
    )


def test_phase3_planar_object_output_normalizes_to_the_same_2d_recipe() -> None:
    session = ModelSession()
    bridge, controller = _controller(session)
    arguments = _arguments()
    arguments["output"] = {"kind": "planar"}

    result = controller.dispatch(
        "prepare_planar_construction_proposal",
        arguments,
        ToolExecutionContext("phase3-planar", 0, "object-planar"),
    )

    assert result.ok, result.summary
    proposal = bridge._records[result.data["proposal_id"]].proposal
    assert proposal.preconditions["local_evidence"]["output_kind"] == "planar"
    receipt = bridge.accept_from_gui_control(result.data["proposal_id"])
    assert receipt.state is ProposalState.SUCCEEDED
    assert session.snapshot().parts[0].dimension == 2


def test_agent_appends_a_second_round_path_slot_as_a_new_cut_feature() -> None:
    session = ModelSession()
    bridge, controller = _controller(session)
    initial = _dispatch(controller, key="feature-history-initial")
    assert initial.ok, initial.summary
    initial_receipt = bridge.accept_from_gui_control(str(initial.data["proposal_id"]))
    assert initial_receipt.state is ProposalState.SUCCEEDED
    controller.record_proposal_state(
        "geometry", initial_receipt.state, initial_receipt.message
    )

    edited = controller.dispatch(
        "prepare_geometry_edit",
        {
            "part_id": "P1",
            "edit": {
                "operation": "add_path_slot",
                "points": [
                    {"x": 75.0, "y": 82.0},
                    {"x": 40.0, "y": 82.0},
                    {"x": 40.0, "y": 50.0},
                    {"x": 75.0, "y": 50.0},
                    {"x": 75.0, "y": 18.0},
                    {"x": 40.0, "y": 18.0},
                ],
                "width": 6.0,
                "cap": "round",
                "join": "round",
            },
        },
        ToolExecutionContext(
            session.session_id,
            session.session_revision,
            "feature-history-second-cut",
        ),
    )
    assert edited.ok, edited.summary
    assert (
        bridge.accept_from_gui_control(str(edited.data["proposal_id"])).state
        is ProposalState.SUCCEEDED
    )

    recipe = session.snapshot().parts[0].geometry_recipe
    assert type(recipe) is BooleanGeometry
    assert recipe.operation == "cut"
    history = derive_feature_history(recipe)
    assert [feature.kind for feature in history] == [
        "sketch",
        "cut",
        "cut",
        "cut",
    ]
    assert [feature.name for feature in history] == [
        "Sketch-1",
        "Cut-1",
        "Cut-2",
        "Cut-3",
    ]


def test_phase3_accept_refreshes_revision_before_clearing_pending_operation() -> None:
    session = ModelSession()
    bridge, controller = _controller(session)
    result = _dispatch(controller, key="accept-refresh-order")
    bridge.set_lifecycle_listener(
        lambda proposal, state, message: controller.record_proposal_state(
            proposal.proposal_kind.value,
            state,
            message,
        )
    )

    receipt = bridge.accept_from_gui_control(result.data["proposal_id"])

    assert receipt.state is ProposalState.SUCCEEDED
    assert controller.stage is AuthoringWorkflowStage.MESH_READY
    assert {item.name for item in controller.definitions}.issuperset(
        {"read_geometry_edit_context", "prepare_geometry_edit"}
    )


def test_phase3_fake_provider_uses_one_card_and_continues_from_new_snapshot(
    tmp_path,
) -> None:
    session = ModelSession()
    bridge, controller = _controller(session)
    dynamic = _ControllerDynamicTools(controller)
    provider = FakeProvider(
        [
            _tool(
                "prepare-ir",
                "prepare_planar_construction_proposal",
                _arguments(),
            )
        ]
    )
    engine = AgentSessionEngine(
        tmp_path / "phase3-fake-provider",
        provider,
        dynamic_tools=dynamic,
    )
    before = session.snapshot()

    events = engine.send_message("创建带 H 形槽和四角孔的二维板")

    assert session.snapshot() == before
    assert len(provider.requests) == 1
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
        _tool("read-new", "read_authoring_context", {}),
        _text("二维部件已进入后续建模阶段。"),
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


@pytest.mark.parametrize("terminal", ["reject", "stale"])
def test_phase3_reject_and_stale_keep_the_blank_session_unchanged(
    terminal: str,
) -> None:
    session = ModelSession()
    bridge, controller = _controller(session)
    before = session.snapshot()
    result = _dispatch(controller, key=terminal)
    proposal_id = result.data["proposal_id"]

    if terminal == "reject":
        receipt = bridge.reject_from_gui_control(proposal_id)
        assert receipt.state is ProposalState.REJECTED
    else:
        assert bridge.stale_pending_proposals_from_gui("binding changed") == (
            proposal_id,
        )
        assert bridge.state(proposal_id) is ProposalState.STALE

    assert session.snapshot() == before


def test_phase3_invalid_ir_fails_without_a_card_or_model_change() -> None:
    session = ModelSession()
    bridge, controller = _controller(session)
    before = session.snapshot()
    arguments = _arguments()
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


def test_phase3_cancel_and_accept_failure_keep_session_unchanged(monkeypatch) -> None:
    cancelled_session = ModelSession()
    _cancelled_bridge, cancelled_controller = _controller(cancelled_session)
    cancelled_before = cancelled_session.snapshot()
    result = _dispatch(cancelled_controller, key="cancel")
    assert result.ok

    cancelled_controller.cancel_turn("provider operation cancelled")

    assert cancelled_session.snapshot() == cancelled_before

    failed_session = ModelSession()
    failed_bridge, failed_controller = _controller(failed_session)
    failed_before = failed_session.snapshot()
    failed = _dispatch(failed_controller, key="failed-accept")

    def fail_commit(*_args, **_kwargs) -> None:
        raise RuntimeError("injected commit failure")

    monkeypatch.setattr(
        failed_session,
        "create_native_project_with_first_part",
        fail_commit,
    )
    receipt = failed_bridge.accept_from_gui_control(failed.data["proposal_id"])

    assert receipt.state is ProposalState.FAILED
    assert failed_session.snapshot() == failed_before


def test_phase3_legacy_planar_profiles_remains_callable_and_auditable() -> None:
    session = ModelSession()
    _bridge, controller = _controller(session)
    assert {item.name for item in controller.definitions} >= {
        "prepare_geometry_proposal",
        "prepare_planar_construction_proposal",
    }

    result = controller.dispatch(
        "prepare_geometry_proposal",
        {
            "part_function": "兼容矩形板",
            "geometry": {
                "kind": "planar_profiles",
                "profiles": [
                    {
                        "kind": "rectangle",
                        "x": 0.0,
                        "y": 0.0,
                        "width": 10.0,
                        "height": 5.0,
                    }
                ],
            },
        },
        ToolExecutionContext("phase3-legacy", 0, "legacy"),
    )

    assert result.ok
    assert result.data["authoring_path"] == "legacy_planar_profiles"
