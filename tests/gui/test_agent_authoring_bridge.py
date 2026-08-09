from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QLabel, QToolButton

from fem.core.model import (
    AnalysisStep,
    DisplacementConstraint,
    FEMModel,
    NodalLoad,
)
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
    RequirementLedger,
)
from fem_gui.agent_authoring import (
    AgentAuthoringBridge,
    authoring_context_from_snapshot,
)
from fem_gui.agent_events import (
    AgentEvent,
    AgentEventError,
    AgentEventProjector,
    EventType,
    ProposalViewStatus,
)
from fem_gui.agent_runtime import QtAgentRuntime
from fem_gui.widgets.agent_chat import (
    AgentChatDrawer,
    _AGENT_CHAT_STYLESHEET,
)
from tests.helpers.mesh_builders import make_selection_quad_mesh


def _application() -> QApplication:
    return QApplication.instance() or QApplication([])


def _context(*, revision: int = 4) -> AuthoringContext:
    return AuthoringContext(
        binding=LocalModelBinding(
            document_id="document:session-1",
            session_id="session-1",
            session_revision=revision,
            source_kind="native",
            supported=True,
        ),
        model_name="模型-孔板",
        active_part_id=None,
    )


def _proposal(
    proposal_id: str = "proposal-1",
    *,
    tool_call_id: str = "call-1",
    invalidates_results: bool = False,
) -> AgentProposal:
    return AgentProposal.create(
        proposal_id=proposal_id,
        proposal_kind=ProposalKind.GEOMETRY,
        agent_session_id="agent-session-1",
        turn_id="turn-1",
        source_tool_call_ids=(tool_call_id,),
        target_document_id="document:session-1",
        target_session_id="session-1",
        base_session_revision=4,
        draft_revision=1,
        operations=(
            ModelOperation(
                OperationKind.ADD_NATIVE_PART,
                {
                    "part_name": "部件-偏心孔板",
                    "recipe": {"kind": "PlateWithHoleGeometry"},
                },
            ),
        ),
        preconditions={"source_kind": "native"},
        expected_changes={"part_count_delta": 1},
        invalidation_impact={
            "mesh": False,
            "results": invalidates_results,
        },
        display_summary={
            "title": "加入偏心孔板",
            "summary": "A1 静态提案，不修改真实 ModelSession",
        },
    )


def test_snapshot_adapter_omits_paths_arrays_and_gui_objects() -> None:
    class _GuiObject:
        pass

    model = SimpleNamespace(
        nodes=[(0.0, 0.0), (1.0, 0.0)],
        elements=[(0, 1)],
        secret_gui=_GuiObject(),
    )
    snapshot = SimpleNamespace(
        session_id="session-1",
        session_revision=7,
        source_kind="native",
        source_path="D:\\private\\model.femproj",
        model_name="模型-孔板",
        active_part_id=None,
        parts=(),
        named_regions={},
        materials=(),
        sections=(),
        assignments=(),
        steps=(),
        artifact=SimpleNamespace(model=model),
        validations={},
        runs=(),
        displayed_result=None,
        mesh_current=True,
        qt_object=_GuiObject(),
    )

    context = authoring_context_from_snapshot(snapshot)
    payload = context.to_provider_dict()
    encoded = json.dumps(payload, ensure_ascii=False)

    assert payload["mesh"]["node_count"] == 2
    assert payload["mesh"]["element_count"] == 1
    assert "nodes" not in payload
    assert "elements" not in payload
    assert "private" not in encoded
    assert "_GuiObject" not in encoded
    assert context.binding.document_id == "document:session-1"


def test_snapshot_adapter_reads_counts_from_actual_fem_model_mesh() -> None:
    model = FEMModel(make_selection_quad_mesh(), name="模型-已划分网格")
    snapshot = SimpleNamespace(
        session_id="session-meshed",
        session_revision=8,
        source_kind="native",
        can_save=True,
        model_name=model.name,
        active_part_id=None,
        parts=(),
        named_regions={},
        materials=(),
        sections=(),
        assignments=(),
        steps=(),
        artifact=SimpleNamespace(model=model),
        validations={},
        runs=(),
        displayed_result=None,
        mesh_current=True,
        unit_context=None,
    )

    context = authoring_context_from_snapshot(snapshot)

    assert context.mesh.present
    assert context.mesh.current
    assert context.mesh.node_count == 4
    assert context.mesh.element_count == 1

    empty_model = FEMModel(type(model.mesh)([], []), name="模型-仅几何")
    empty_snapshot = SimpleNamespace(
        **{
            **vars(snapshot),
            "artifact": SimpleNamespace(model=empty_model),
        }
    )
    empty_context = authoring_context_from_snapshot(empty_snapshot)

    assert not empty_context.mesh.present
    assert not empty_context.mesh.current
    assert empty_context.mesh.node_count == 0
    assert empty_context.mesh.element_count == 0


def test_snapshot_adapter_skips_engineering_edits_without_units() -> None:
    model = FEMModel(make_selection_quad_mesh(), name="模型-无单位")
    snapshot = SimpleNamespace(
        session_id="session-without-units",
        session_revision=9,
        source_kind="native",
        can_save=True,
        model_name=model.name,
        active_part_id=None,
        parts=(),
        named_regions={},
        materials=(),
        sections=(),
        assignments=(),
        steps=(
            AnalysisStep(
                "Load",
                boundaries=(
                    DisplacementConstraint("Root", 1, 2, name="Fixed"),
                ),
                cloads=(NodalLoad("Tip", 1, 10.0, name="Force"),),
            ),
        ),
        artifact=SimpleNamespace(model=model),
        validations={},
        runs=(),
        displayed_result=None,
        mesh_current=True,
        unit_context=None,
    )

    context = authoring_context_from_snapshot(snapshot)
    edit_capability = next(
        item
        for item in context.capabilities
        if item.operation == "edit_model_objects"
    )

    assert not edit_capability.enabled


def test_bridge_gui_authorization_replay_stale_and_exception_paths() -> None:
    port = FakeAuthoringPort()
    bridge = AgentAuthoringBridge(port)
    bridge.bind_context(_context())
    proposal = _proposal()

    pending = bridge.register_proposal(proposal)
    replay = bridge.register_proposal(proposal)
    idempotent = bridge.register_proposal(_proposal("proposal-replay"))

    assert pending.state is ProposalState.PENDING_CONFIRMATION
    assert replay.replayed
    assert idempotent.replayed
    assert idempotent.proposal_id == proposal.proposal_id
    assert [call for call in port.calls if call[0] == "present"] == [
        ("present", proposal.proposal_id)
    ]

    with pytest.raises(AuthoringAuthorizationError):
        bridge.accept_proposal(proposal.proposal_id)
    with pytest.raises(AuthoringAuthorizationError):
        bridge.accept_proposal(proposal.proposal_id, "可以")

    assert bridge.accept_from_gui_control(
        proposal.proposal_id
    ).state is ProposalState.ACCEPTED
    with pytest.raises(AuthoringAuthorizationError):
        bridge.accept_from_gui_control(proposal.proposal_id)

    rejected = _proposal(
        "proposal-rejected",
        tool_call_id="call-rejected",
    )
    bridge.register_proposal(rejected)
    assert bridge.reject_from_gui_control(
        rejected.proposal_id
    ).state is ProposalState.REJECTED

    stale = _proposal("proposal-stale", tool_call_id="call-stale")
    bridge.register_proposal(stale)
    stale_ids = bridge.bind_context(_context(revision=5))
    assert stale_ids == (stale.proposal_id,)
    assert bridge.state(stale.proposal_id) is ProposalState.STALE

    failing_port = FakeAuthoringPort()
    failing_port.accept_error = RuntimeError("fake port failure")
    failing = AgentAuthoringBridge(failing_port)
    failing.bind_context(_context())
    failed_proposal = _proposal(
        "proposal-failed",
        tool_call_id="call-failed",
    )
    failing.register_proposal(failed_proposal)
    receipt = failing.accept_from_gui_control(failed_proposal.proposal_id)
    assert receipt.state is ProposalState.FAILED
    assert receipt.message == "fake port failure"


def test_result_invalidating_proposal_uses_configured_unsaved_result_gate() -> None:
    port = FakeAuthoringPort()
    bridge = AgentAuthoringBridge(port)
    bridge.bind_context(_context())
    proposal = _proposal(
        "proposal-result-invalidation",
        tool_call_id="call-result-invalidation",
        invalidates_results=True,
    )
    bridge.register_proposal(proposal)
    allow = False
    confirmations: list[bool] = []

    def confirm() -> bool:
        confirmations.append(allow)
        return allow

    bridge.set_result_invalidation_confirmation(confirm)
    with pytest.raises(AuthoringAuthorizationError, match="cancelled"):
        bridge.accept_from_gui_control(proposal.proposal_id)
    assert bridge.state(proposal.proposal_id) is ProposalState.PENDING_CONFIRMATION
    assert not [call for call in port.calls if call[0] == "accept"]

    allow = True
    accepted = bridge.accept_from_gui_control(proposal.proposal_id)
    assert accepted.state is ProposalState.ACCEPTED
    assert confirmations == [False, True]


def test_requirement_review_confirmation_only_enters_through_gui_bridge() -> None:
    ledger = RequirementLedger()
    ledger.record(
        "geometry.dimension",
        field_type="integer",
        stage="geometry",
        value=2,
        source_turn_id="turn-1",
    )
    review = ledger.create_review(
        "review-1",
        ("geometry.dimension",),
    )
    bridge = AgentAuthoringBridge(FakeAuthoringPort())
    bridge.bind_context(_context())

    with pytest.raises(ClarificationRequiredError):
        ledger.require_confirmed("geometry", ("geometry.dimension",))

    confirmed = bridge.confirm_requirement_review_from_gui(ledger, review)

    assert confirmed.status.value == "confirmed"
    assert ledger.require_confirmed(
        "geometry",
        ("geometry.dimension",),
    )[0].value == 2


def test_gui_authorization_rejects_a_background_caller() -> None:
    bridge = AgentAuthoringBridge(FakeAuthoringPort())
    bridge.bind_context(_context())
    proposal = _proposal()
    bridge.register_proposal(proposal)
    outcome: list[Exception] = []

    def call_from_worker() -> None:
        try:
            bridge.accept_from_gui_control(proposal.proposal_id)
        except Exception as exc:
            outcome.append(exc)

    worker = threading.Thread(target=call_from_worker)
    worker.start()
    worker.join(timeout=1.0)

    assert not worker.is_alive()
    assert len(outcome) == 1
    assert isinstance(outcome[0], AuthoringAuthorizationError)
    assert bridge.state(proposal.proposal_id) is (
        ProposalState.PENDING_CONFIRMATION
    )


@dataclass
class _Events:
    sequence: int = 1

    def make(
        self,
        event_type: EventType,
        payload: dict[str, object],
        *,
        turn_id: str = "turn-1",
    ) -> AgentEvent:
        sequence = self.sequence
        self.sequence += 1
        return AgentEvent.create(
            event_id=f"proposal-event-{sequence}",
            session_id="agent-session-1",
            turn_id=turn_id,
            sequence=sequence,
            event_type=event_type,
            payload=payload,
            timestamp="2026-07-30T08:00:00Z",
        )


def _proposal_payload(proposal_id: str, proposal_hash: str) -> dict[str, object]:
    return {
        "proposal_id": proposal_id,
        "proposal_hash": proposal_hash,
        "proposal_kind": "geometry",
        "title": "加入偏心孔板",
        "summary": "本地生成的有界摘要",
        "impact": "新增一个部件；A1 不修改模型",
        "confirm_label": "加入模型",
        "target_document_id": "document:session-1",
        "target_session_id": "session-1",
        "base_session_revision": 4,
    }


def test_proposal_events_are_strict_and_replay_all_lifecycle_paths() -> None:
    events = _Events()
    log = [
        events.make(
            EventType.TURN_STARTED,
            {"user_message": "建立偏心孔板"},
        )
    ]
    hashes = {
        name: (str(index) * 64)
        for index, name in enumerate(
            ("success", "reject", "stale", "failed", "cancelled"),
            start=1,
        )
    }
    for name, proposal_hash in hashes.items():
        log.append(
            events.make(
                EventType.PROPOSAL_REQUESTED,
                _proposal_payload(f"proposal-{name}", proposal_hash),
            )
        )
    log.append(events.make(EventType.TURN_COMPLETE, {}))
    log.extend(
        (
            events.make(
                EventType.PROPOSAL_ACCEPTED,
                {
                    "proposal_id": "proposal-success",
                    "proposal_hash": hashes["success"],
                },
            ),
            events.make(
                EventType.PROPOSAL_STARTED,
                {
                    "proposal_id": "proposal-success",
                    "proposal_hash": hashes["success"],
                },
            ),
            events.make(
                EventType.PROPOSAL_PROGRESS,
                {
                    "proposal_id": "proposal-success",
                    "proposal_hash": hashes["success"],
                    "progress": 0.5,
                    "message": "处理中",
                },
            ),
            events.make(
                EventType.PROPOSAL_SUCCEEDED,
                {
                    "proposal_id": "proposal-success",
                    "proposal_hash": hashes["success"],
                    "summary": "完成",
                },
            ),
            events.make(
                EventType.PROPOSAL_REJECTED,
                {
                    "proposal_id": "proposal-reject",
                    "proposal_hash": hashes["reject"],
                    "reason": "用户拒绝",
                },
            ),
            events.make(
                EventType.PROPOSAL_STALE,
                {
                    "proposal_id": "proposal-stale",
                    "proposal_hash": hashes["stale"],
                    "reason": "revision 改变",
                },
            ),
            events.make(
                EventType.PROPOSAL_FAILED,
                {
                    "proposal_id": "proposal-failed",
                    "proposal_hash": hashes["failed"],
                    "reason": "Fake Port 异常",
                },
            ),
            events.make(
                EventType.PROPOSAL_ACCEPTED,
                {
                    "proposal_id": "proposal-cancelled",
                    "proposal_hash": hashes["cancelled"],
                },
            ),
            events.make(
                EventType.PROPOSAL_STARTED,
                {
                    "proposal_id": "proposal-cancelled",
                    "proposal_hash": hashes["cancelled"],
                },
            ),
            events.make(
                EventType.PROPOSAL_CANCELLED,
                {
                    "proposal_id": "proposal-cancelled",
                    "proposal_hash": hashes["cancelled"],
                    "reason": "用户取消",
                },
            ),
        )
    )

    projected = AgentEventProjector.replay(log)
    restored = AgentEventProjector.restore_event_log(
        projected.export_event_log()
    )
    snapshot = projected.presentation.to_snapshot()

    assert restored.presentation.to_snapshot() == snapshot
    assert {
        item["proposal_id"]: item["status"]
        for item in snapshot["turns"][0]["proposals"]
    } == {
        "proposal-success": "succeeded",
        "proposal-reject": "rejected",
        "proposal-stale": "stale",
        "proposal-failed": "failed",
        "proposal-cancelled": "cancelled",
    }

    unknown = _proposal_payload("proposal-unknown", "a" * 64)
    unknown["raw_patch"] = {"operations": []}
    with pytest.raises(AgentEventError, match="未知 payload"):
        events.make(EventType.PROPOSAL_REQUESTED, unknown)


def test_minimal_gui_card_binds_and_only_buttons_authorize() -> None:
    application = _application()
    bridge = AgentAuthoringBridge(FakeAuthoringPort())
    bridge.bind_context(_context())
    proposal = _proposal()
    bridge.register_proposal(proposal)
    events = _Events()
    log = (
        events.make(
            EventType.TURN_STARTED,
            {"user_message": "建立偏心孔板"},
        ),
        events.make(
            EventType.PROPOSAL_REQUESTED,
            _proposal_payload(
                proposal.proposal_id,
                proposal.proposal_hash,
            ),
        ),
        events.make(EventType.TURN_COMPLETE, {}),
    )

    drawer = AgentChatDrawer(authoring_bridge=bridge)
    drawer.setStyleSheet(_AGENT_CHAT_STYLESHEET)
    drawer.replay_agent_events(log)
    drawer.show()
    application.processEvents()

    accept = drawer.findChild(
        QToolButton,
        "agentChatProposalAcceptButton",
    )
    reject = drawer.findChild(
        QToolButton,
        "agentChatProposalRejectButton",
    )

    assert accept is not None and accept.isEnabled()
    assert reject is not None and reject.isEnabled()
    assert accept.text() == "加入模型"
    assert reject.text() == "拒绝"
    assert accept.palette().color(accept.foregroundRole()).name() == "#ffffff"
    assert drawer.findChild(QLabel, "agentChatProposalImpact") is None
    assert drawer.findChild(QLabel, "agentChatProposalRevision") is None
    assert bridge.state(proposal.proposal_id) is (
        ProposalState.PENDING_CONFIRMATION
    )

    QTest.mouseClick(accept, Qt.MouseButton.LeftButton)
    application.processEvents()

    assert bridge.state(proposal.proposal_id) is ProposalState.ACCEPTED
    assert (
        drawer.event_presentation.turns[0].proposals[0].status
        is ProposalViewStatus.ACCEPTED
    )
    accept = drawer.findChild(
        QToolButton,
        "agentChatProposalAcceptButton",
    )
    assert accept is not None and not accept.isEnabled()
    assert "A1 Fake Port" in drawer.composer_hint.text()
    drawer.close()


def test_runtime_emits_ordered_unique_lifecycle_and_stales_bad_identity(
    tmp_path,
) -> None:
    application = _application()
    runtime = QtAgentRuntime(tmp_path / "agent-private")
    collector: list[AgentEvent] = []
    runtime.agentEventReady.connect(collector.append)
    events = _Events()
    proposal_hash = "c" * 64
    log = (
        events.make(
            EventType.TURN_STARTED,
            {"user_message": "建立偏心孔板"},
        ),
        events.make(
            EventType.PROPOSAL_REQUESTED,
            _proposal_payload("proposal-success", proposal_hash),
        ),
        events.make(EventType.TURN_COMPLETE, {}),
    )
    projection = AgentEventProjector.replay(log).presentation
    runtime.synchronize_event_projection_from_gui(projection)

    assert runtime.record_proposal_lifecycle_from_gui(
        "proposal-success",
        proposal_hash,
        "agent-session-1",
        "turn-1",
        ProposalState.SUCCEEDED,
        "",
    )
    application.processEvents()
    assert [event.event_type for event in collector] == [
        EventType.PROPOSAL_ACCEPTED,
        EventType.PROPOSAL_STARTED,
        EventType.PROPOSAL_SUCCEEDED,
    ]
    assert collector[-1].payload["summary"] == "已完成"
    assert not runtime.record_proposal_lifecycle_from_gui(
        "proposal-success",
        proposal_hash,
        "agent-session-1",
        "turn-1",
        ProposalState.FAILED,
        "迟到失败",
    )
    assert (
        AgentEventProjector.replay((*log, *collector))
        .presentation.turns[0]
        .proposals[0]
        .status
        is ProposalViewStatus.SUCCEEDED
    )

    events = _Events()
    stale_hash = "d" * 64
    stale_log = (
        events.make(
            EventType.TURN_STARTED,
            {"user_message": "建立另一个部件"},
        ),
        events.make(
            EventType.PROPOSAL_REQUESTED,
            _proposal_payload("proposal-stale-identity", stale_hash),
        ),
        events.make(EventType.TURN_COMPLETE, {}),
    )
    runtime.synchronize_event_projection_from_gui(
        AgentEventProjector.replay(stale_log).presentation
    )
    collector.clear()
    assert runtime.record_proposal_lifecycle_from_gui(
        "proposal-stale-identity",
        "e" * 64,
        "wrong-agent-session",
        "turn-1",
        ProposalState.ACCEPTED,
    )
    application.processEvents()
    assert [event.event_type for event in collector] == [
        EventType.PROPOSAL_STALE
    ]
    assert collector[0].payload["proposal_hash"] == stale_hash
    runtime.shutdown()


def test_pending_proposal_is_only_rendered_in_the_composer() -> None:
    _application()
    events = _Events()
    proposal_hash = "b" * 64
    log = (
        events.make(
            EventType.TURN_STARTED,
            {"user_message": "建立偏心孔板"},
        ),
        events.make(
            EventType.MESSAGE_START,
            {
                "message_id": "before-proposal",
                "role": "assistant",
                "format": "restricted_markdown",
            },
        ),
        events.make(
            EventType.MESSAGE_DELTA,
            {
                "message_id": "before-proposal",
                "delta": "几何参数已经记录。",
            },
        ),
        events.make(
            EventType.MESSAGE_COMPLETE,
            {"message_id": "before-proposal"},
        ),
        events.make(
            EventType.PROPOSAL_REQUESTED,
            _proposal_payload("proposal-last", proposal_hash),
        ),
        events.make(
            EventType.MESSAGE_START,
            {
                "message_id": "after-proposal",
                "role": "assistant",
                "format": "restricted_markdown",
            },
        ),
        events.make(
            EventType.MESSAGE_DELTA,
            {
                "message_id": "after-proposal",
                "delta": "请确认是否创建。",
            },
        ),
        events.make(
            EventType.MESSAGE_COMPLETE,
            {"message_id": "after-proposal"},
        ),
        events.make(EventType.TURN_COMPLETE, {}),
    )

    drawer = AgentChatDrawer()
    drawer.replay_agent_events(log)

    assert drawer.findChild(QLabel, "agentChatProposalStatus") is None
    assert drawer.composer_stack.currentWidget() is drawer.composer_task_surface
    assert drawer.composer_task_title.text() == "加入偏心孔板"
    assert drawer.composer_task_impact.isHidden()
    assert drawer.composer_task_status.text() == "等待确认"
    drawer.close()


def test_succeeded_proposal_hides_terminal_detail_and_empty_continuation_user(
) -> None:
    _application()
    events = _Events()
    proposal_hash = "f" * 64
    proposal_id = "proposal-completed"
    identity = {
        "proposal_id": proposal_id,
        "proposal_hash": proposal_hash,
    }
    log = (
        events.make(
            EventType.TURN_STARTED,
            {"user_message": "建立偏心孔板"},
        ),
        events.make(
            EventType.PROPOSAL_REQUESTED,
            _proposal_payload(proposal_id, proposal_hash),
        ),
        events.make(EventType.TURN_COMPLETE, {}),
        events.make(EventType.PROPOSAL_ACCEPTED, identity),
        events.make(EventType.PROPOSAL_STARTED, identity),
        events.make(
            EventType.PROPOSAL_SUCCEEDED,
            {**identity, "summary": "提案执行完成"},
        ),
        events.make(
            EventType.CONTINUATION_STARTED,
            {
                **identity,
                "source_turn_id": "turn-1",
                "status": "succeeded",
            },
            turn_id="continuation-turn",
        ),
        events.make(
            EventType.MESSAGE_START,
            {
                "message_id": "completion-message",
                "role": "assistant",
                "format": "restricted_markdown",
            },
            turn_id="continuation-turn",
        ),
        events.make(
            EventType.MESSAGE_DELTA,
            {
                "message_id": "completion-message",
                "delta": "几何已加入模型。",
            },
            turn_id="continuation-turn",
        ),
        events.make(
            EventType.MESSAGE_COMPLETE,
            {"message_id": "completion-message"},
            turn_id="continuation-turn",
        ),
        events.make(
            EventType.TURN_COMPLETE,
            {},
            turn_id="continuation-turn",
        ),
    )

    drawer = AgentChatDrawer()
    drawer.replay_agent_events(log)

    user_labels = drawer.findChildren(QLabel, "agentChatUserLabel")
    proposal_status = drawer.findChild(QLabel, "agentChatProposalStatus")
    assert [label.text() for label in user_labels] == ["建立偏心孔板"]
    assert proposal_status is not None
    assert proposal_status.text() == "已完成"
    drawer.close()
