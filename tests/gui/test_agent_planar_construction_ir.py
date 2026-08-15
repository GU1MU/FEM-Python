from __future__ import annotations

from copy import deepcopy
import json

import pytest

from fem import geometry as geometry_runtime
from fem.application import ModelSession
from fem.geometry import SketchGeometry
from fem_agent.authoring import AgentProposal, ProposalState
from fem_agent.authoring_runtime import AuthoringTurnSnapshot
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
    construction = deepcopy(EXPECTED_H_CONSTRUCTION)
    construction["nodes"] = [
        node for node in construction["nodes"] if node["id"] != "all_cuts"
    ]
    construction["nodes"][-1]["subtract"] = ["h_slot", "holes"]
    return construction


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
        "extrusion",
        "revolution",
        "path_sweep",
    }
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
    serialized = json.dumps(capability).casefold()
    assert "gmsh" not in serialized
    assert "occ" not in serialized


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
        "union": 1,
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
    assert type(recipe) is SketchGeometry
    assert recipe.is_strict
    analysis = geometry_runtime.analyze_sketch_profiles(recipe)
    assert analysis.valid
    assert sum(profile.is_material for profile in analysis.profiles) == 1
    assert sum(profile.is_hole for profile in analysis.profiles) == 5


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
