from __future__ import annotations

import json
from dataclasses import replace

from fem_agent.authoring import (
    AuthoringContext,
    CapabilitySummary,
    LocalModelBinding,
    ProposalState,
)
from fem_agent.authoring_runtime import (
    AuthoringToolOutcome,
    AuthoringWorkflowController,
    AuthoringWorkflowStage,
)
from fem_agent.engine import AgentSessionEngine
from fem_agent.providers.fake import FakeProvider
from fem_agent.tools.registry import AgentToolRegistry, ToolExecutionContext
from fem_gui.agent_events import AgentEvent, AgentEventProjector, EventType


def _native_context(revision: int = 3) -> AuthoringContext:
    return AuthoringContext(
        LocalModelBinding(
            "document:save",
            "native-save",
            revision,
            "native",
            True,
        ),
        "模型-1",
        "part-1",
        capabilities=(
            CapabilitySummary("request_project_save", True),
        ),
    )


def _controller(
    current: list[AuthoringContext],
) -> AuthoringWorkflowController:
    def request_save(_arguments, controller):
        record = controller.register_project_save_proposal(
            "proposal-project-save",
            current[0],
        )
        return AuthoringToolOutcome(
            "Project save is waiting for the local GUI control.",
            {
                "proposal_id": record.proposal_id,
                "proposal_hash": record.proposal_hash,
                "state": record.state.value,
                "proposal_view": {
                    "proposal_id": record.proposal_id,
                    "proposal_hash": record.proposal_hash,
                    "proposal_kind": "project_save",
                    "title": "保存当前自主项目",
                    "summary": "保存当前已接受的模型状态",
                    "impact": "确认后调用本地项目保存",
                    "confirm_label": "保存模型",
                    "target_document_id": record.target_document_id,
                    "target_session_id": record.target_session_id,
                    "base_session_revision": record.base_session_revision,
                },
            },
        )

    controller = AuthoringWorkflowController(
        lambda: current[0],
        {"request_project_save": request_save},
    )
    controller.observe_binding(current[0])
    return controller


def _dispatch(controller: AuthoringWorkflowController):
    return controller.dispatch(
        "request_project_save",
        {},
        ToolExecutionContext("agent-save", 0, "save-key"),
    )


def test_project_save_tool_is_path_free_and_dispatch_only_creates_local_request(
    tmp_path,
) -> None:
    current = [_native_context()]
    controller = _controller(current)

    definition = next(
        item
        for item in controller.definitions
        if item.name == "request_project_save"
    )
    assert definition.parameters == {
        "type": "object",
        "properties": {},
        "required": [],
        "additionalProperties": False,
    }

    before = tuple(tmp_path.iterdir())
    result = _dispatch(controller)

    assert result.ok
    assert tuple(tmp_path.iterdir()) == before
    assert controller.stage is AuthoringWorkflowStage.PROJECT_SAVE_PENDING
    record = controller.project_save_record
    assert record is not None
    assert record.state is ProposalState.PENDING_CONFIRMATION
    payload = json.dumps(result.data, ensure_ascii=False)
    assert "project_save" in payload
    assert "project_path" not in payload
    assert str(tmp_path) not in payload
    assert "confirm_project_save" not in {
        item.name for item in controller.definitions
    }

    current[0] = replace(
        current[0],
        binding=replace(
            current[0].binding,
            source_kind="imported",
            supported=False,
        ),
    )
    controller.reset_for_binding()
    assert "request_project_save" not in {
        item.name for item in controller.definitions
    }


def test_project_save_saved_state_revision_is_expected_but_other_change_is_stale() -> (
    None
):
    current = [_native_context()]
    controller = _controller(current)
    result = _dispatch(controller)
    view = result.data["proposal_view"]
    assert isinstance(view, dict)

    controller.begin_project_save_from_gui(
        view["proposal_id"],
        view["proposal_hash"],
        current[0],
    )
    current[0] = replace(
        current[0],
        binding=replace(
            current[0].binding,
            session_revision=current[0].binding.session_revision + 1,
        ),
    )
    assert controller.observe_binding(
        current[0],
        saved_state_transition=True,
    )
    controller.record_project_save_state(
        view["proposal_id"],
        view["proposal_hash"],
        ProposalState.SUCCEEDED,
        "saved",
    )
    assert controller.stage is AuthoringWorkflowStage.REQUIREMENTS
    assert controller.project_save_record.state is ProposalState.SUCCEEDED

    result = _dispatch(controller)
    next_view = result.data["proposal_view"]
    assert isinstance(next_view, dict)
    current[0] = replace(
        current[0],
        binding=replace(
            current[0].binding,
            session_revision=current[0].binding.session_revision + 1,
        ),
    )
    assert not controller.observe_binding(current[0])
    assert controller.stage is AuthoringWorkflowStage.STALE
    assert controller.project_save_record.state is ProposalState.STALE
    assert not controller.can_accept_project_save_from_gui(
        next_view["proposal_id"],
        next_view["proposal_hash"],
        current[0],
    )


def test_v0_prompt_and_catalog_do_not_publish_project_save(tmp_path) -> None:
    registry = AgentToolRegistry(tmp_path / "registry")
    assert "request_project_save" not in {
        item.name for item in registry.definitions
    }

    engine = AgentSessionEngine(
        tmp_path / "engine",
        FakeProvider(),
    )
    try:
        assert "project-save request" not in engine._system_prompt
        assert "request_project_save" not in {
            item.name for item in engine.registry.definitions
        }
    finally:
        engine.close_session()


def test_project_save_event_is_a_strict_independent_proposal_view() -> None:
    projector = AgentEventProjector()
    projector.apply(
        AgentEvent.create(
            event_id="event-save-start",
            session_id="agent-save",
            turn_id="turn-save",
            sequence=1,
            event_type=EventType.TURN_STARTED,
            payload={"user_message": "保存模型"},
            timestamp="2026-07-30T00:00:00Z",
        )
    )
    projector.apply(
        AgentEvent.create(
            event_id="event-save-request",
            session_id="agent-save",
            turn_id="turn-save",
            sequence=2,
            event_type=EventType.PROPOSAL_REQUESTED,
            payload={
                "proposal_id": "proposal-save",
                "proposal_hash": "a" * 64,
                "proposal_kind": "project_save",
                "title": "保存当前自主项目",
                "summary": "保存当前已接受的模型状态",
                "impact": "确认后调用本地项目保存",
                "confirm_label": "保存模型",
                "target_document_id": "document:save",
                "target_session_id": "native-save",
                "base_session_revision": 3,
            },
            timestamp="2026-07-30T00:00:01Z",
        )
    )

    proposal = projector.presentation.turns[0].proposals[0]
    assert proposal.proposal_kind == "project_save"
    assert proposal.base_session_revision == 3
