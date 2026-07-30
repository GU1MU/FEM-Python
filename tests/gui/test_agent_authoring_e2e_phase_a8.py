from __future__ import annotations

import os
import threading
import time

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QToolButton

from fem_agent.authoring import (
    AuthoringContext,
    FakeAuthoringPort,
    LocalModelBinding,
    ProposalState,
)
from fem_agent.authoring_runtime import (
    AuthoringToolOutcome,
    AuthoringWorkflowController,
    AuthoringWorkflowStage,
)
from fem_agent.providers.base import (
    AssistantMessage,
    ProviderResponse,
    ToolCall,
)
from fem_agent.providers.fake import FakeProvider
from fem_agent.result_authoring import AgentResultQueryBridge
from fem_agent.tools.registry import ToolExecutionContext
from fem_gui.agent_authoring import (
    AgentAuthoringBridge,
    SessionGeometryAuthoringPort,
    SessionResultQueryPort,
    create_session_authoring_workflow_controller,
)
from fem_gui.agent_events import EventType
from fem_gui.agent_runtime import QtAgentRuntime
from fem_gui.main_window import FEMMainWindow
from fem_gui.widgets.agent_chat import AgentChatDrawer
from tests.test_agent_authoring_phase_a4 import _session as _a4_session
from tests.test_agent_authoring_phase_a8 import (
    _context,
    _controller,
    _requirements_for,
)


def _tool_response(call: ToolCall) -> ProviderResponse:
    return ProviderResponse(
        AssistantMessage("assistant", tool_calls=(call,)),
        finish_reason="tool_calls",
    )


def _text_response(text: str) -> ProviderResponse:
    return ProviderResponse(
        AssistantMessage("assistant", content=text),
        finish_reason="stop",
    )


def _application() -> QApplication:
    return QApplication.instance() or QApplication([])


def _wait_until(predicate, *, timeout_ms: int = 15_000) -> None:
    deadline = time.monotonic() + timeout_ms / 1000
    application = _application()
    while not predicate() and time.monotonic() < deadline:
        application.processEvents()
        QTest.qWait(10)
    application.processEvents()
    assert predicate()


def test_a8_qt_runtime_dispatches_dynamic_tool_on_owner_thread(
    tmp_path,
) -> None:
    _application()
    owner_thread = threading.get_ident()
    reader_threads: list[int] = []

    def read_context():
        reader_threads.append(threading.get_ident())
        return {
            "schema_version": "1.0",
            "binding": {
                "document_id": "document:a8",
                "session_id": "native-a8",
                "session_revision": 0,
                "source_kind": "blank",
                "supported": True,
            },
            "model_name": None,
            "parts": [],
            "mesh": {"generated": False},
            "definitions": {"analysis_step_count": 0},
        }

    controller = AuthoringWorkflowController(read_context, {})
    provider = FakeProvider(
        [
            _tool_response(
                ToolCall("context-a8", "read_authoring_context", {})
            ),
            _text_response("已读取有界建模上下文。"),
        ]
    )
    runtime = QtAgentRuntime(
        tmp_path / "agent-private",
        provider_factory=lambda: provider,
        authoring_controller=controller,
    )
    events = []
    runtime.agentEventReady.connect(events.append)

    assert runtime.send_message("读取当前建模上下文")
    _wait_until(lambda: not runtime.busy)

    assert reader_threads == [owner_thread]
    assert provider.requests
    available_names = {
        item.name for item in provider.requests[0].tools
    }
    assert "read_authoring_context" in available_names
    assert "set_authoring_requirements" in available_names
    assert "set_unit_context" not in available_names
    assert any(
        event.event_type.value == "tool_requested"
        and event.payload["tool_name"] == "read_authoring_context"
        for event in events
    )
    assert any(event.event_type.value == "tool_result" for event in events)
    runtime.shutdown()


def test_a8_qt_runtime_gui_terminal_notifications_require_owner_thread(
    tmp_path,
) -> None:
    _application()
    controller = AuthoringWorkflowController(lambda: {}, {})
    runtime = QtAgentRuntime(
        tmp_path / "agent-private",
        provider_factory=FakeProvider,
        authoring_controller=controller,
    )
    controller._stage = AuthoringWorkflowStage.MESH_PENDING
    controller._pending_operation = "mesh"

    runtime.record_authoring_proposal_state_from_gui(
        "mesh",
        ProposalState.FAILED,
        "deterministic mesh failure",
    )
    assert controller.stage is AuthoringWorkflowStage.MESH_READY
    assert controller.terminal_records[-1].state == "failed"

    failures: list[BaseException] = []

    def cross_thread_call() -> None:
        try:
            runtime.invalidate_authoring_binding_from_gui(
                "cross-thread document switch"
            )
        except BaseException as error:
            failures.append(error)

    thread = threading.Thread(target=cross_thread_call)
    thread.start()
    thread.join(timeout=5.0)

    assert len(failures) == 1
    assert isinstance(failures[0], RuntimeError)
    runtime.shutdown()


@pytest.mark.parametrize(
    ("abort_kind", "error_type"),
    [
        ("timeout", TimeoutError),
        ("shutdown", RuntimeError),
    ],
)
def test_a8_delayed_authoring_invocation_cannot_run_after_abort(
    tmp_path,
    monkeypatch,
    abort_kind,
    error_type,
) -> None:
    _application()
    calls = []

    def handler(_arguments, _controller):
        calls.append("mutated")
        return AuthoringToolOutcome("unexpected", {"state": "succeeded"})

    controller = AuthoringWorkflowController(
        lambda: _context(),
        {"prepare_geometry_proposal": handler},
    )
    controller._stage = AuthoringWorkflowStage.GEOMETRY_READY
    runtime = QtAgentRuntime(
        tmp_path / f"agent-private-{abort_kind}",
        provider_factory=FakeProvider,
        authoring_controller=controller,
    )
    errors = []

    if abort_kind == "timeout":
        monkeypatch.setattr(
            "fem_gui.agent_runtime._AUTHORING_TOOL_OWNER_TIMEOUT_SECONDS",
            0.05,
        )

    def dispatch_from_worker() -> None:
        try:
            runtime._dispatch_authoring_tool(
                "prepare_geometry_proposal",
                {},
                ToolExecutionContext("session-a8", 0, f"{abort_kind}-a8"),
            )
        except BaseException as error:
            errors.append(error)

    worker = threading.Thread(target=dispatch_from_worker)
    worker.start()
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        with runtime._lock:
            if runtime._authoring_invocations:
                break
        time.sleep(0.005)
    else:
        raise AssertionError("authoring invocation was not queued")

    if abort_kind == "shutdown":
        runtime.shutdown()
    worker.join(timeout=2.0)
    assert not worker.is_alive()
    assert len(errors) == 1
    assert isinstance(errors[0], error_type)
    assert calls == []

    _application().processEvents()
    assert calls == []
    if abort_kind == "timeout":
        runtime.shutdown()


def test_a8_geometry_operation_emits_the_only_local_confirmation_card(
    tmp_path,
) -> None:
    _application()
    bridge = AgentAuthoringBridge(FakeAuthoringPort())
    bridge.bind_context(_context())
    controller, _calls, _queries = _controller(bridge)
    provider = FakeProvider(
        [
            ProviderResponse(
                AssistantMessage(
                    "assistant",
                    tool_calls=(
                        ToolCall(
                            "requirements-a8",
                            "set_authoring_requirements",
                            {
                                "turn_id": "turn-a8",
                                "requirements": _requirements_for("geometry"),
                            },
                        ),
                    ),
                ),
                finish_reason="tool_calls",
            ),
            ProviderResponse(
                AssistantMessage(
                    "assistant",
                    tool_calls=(
                        ToolCall(
                            "geometry-a8",
                            "prepare_geometry_proposal",
                            {},
                        ),
                    ),
                ),
                finish_reason="tool_calls",
            ),
            _text_response("几何创建等待确认。"),
        ]
    )
    runtime = QtAgentRuntime(
        tmp_path / "agent-private-operation-card",
        provider_factory=lambda: provider,
        authoring_controller=controller,
    )
    events = []
    runtime.agentEventReady.connect(events.append)

    assert runtime.send_message("创建这个带孔平板")
    _wait_until(lambda: not runtime.busy)

    proposals = [
        event
        for event in events
        if event.event_type is EventType.PROPOSAL_REQUESTED
    ]
    assert len(proposals) == 1
    proposal = proposals[0]
    assert proposal.payload["proposal_kind"] == "geometry"
    assert proposal.payload["confirm_label"] == "加入模型"
    assert controller.pending_review is None
    assert controller.stage is AuthoringWorkflowStage.GEOMETRY_PENDING
    runtime.shutdown()


@pytest.mark.parametrize(
    ("button_name", "expected_stage"),
    [
        (
            "agentChatProposalAcceptButton",
            AuthoringWorkflowStage.GEOMETRY_PENDING,
        ),
        (
            "agentChatProposalRejectButton",
            AuthoringWorkflowStage.GEOMETRY_READY,
        ),
    ],
)
def test_a8_geometry_card_buttons_reach_gui_boundary(
    tmp_path,
    button_name,
    expected_stage,
) -> None:
    application = _application()
    bridge = AgentAuthoringBridge(FakeAuthoringPort())
    bridge.bind_context(_context())
    controller, _calls, _queries = _controller(bridge)
    provider = FakeProvider(
        [
            ProviderResponse(
                AssistantMessage(
                    "assistant",
                    tool_calls=(
                        ToolCall(
                            "requirements-a8",
                            "set_authoring_requirements",
                            {
                                "turn_id": "turn-a8",
                                "requirements": _requirements_for("geometry"),
                            },
                        ),
                    ),
                ),
                finish_reason="tool_calls",
            ),
            ProviderResponse(
                AssistantMessage(
                    "assistant",
                    tool_calls=(
                        ToolCall(
                            "geometry-a8",
                            "prepare_geometry_proposal",
                            {},
                        ),
                    ),
                ),
                finish_reason="tool_calls",
            ),
            _text_response("几何创建等待确认。"),
        ]
    )
    runtime = QtAgentRuntime(
        tmp_path / f"agent-private-{button_name}",
        provider_factory=lambda: provider,
        authoring_controller=controller,
    )
    drawer = AgentChatDrawer(
        agent_runtime=runtime,
        authoring_bridge=bridge,
    )
    drawer.resize(720, 800)
    drawer.show()
    assert runtime.send_message("创建这个带孔平板")
    _wait_until(lambda: not runtime.busy)

    button = drawer.findChild(QToolButton, button_name)
    assert controller.stage is AuthoringWorkflowStage.GEOMETRY_PENDING
    assert button is not None and button.isEnabled()

    QTest.mouseClick(button, Qt.MouseButton.LeftButton)
    application.processEvents()

    assert controller.stage is expected_stage, drawer.composer_hint.text()
    drawer.close()
    runtime.shutdown()


def test_a8_new_agent_session_discards_pending_geometry_proposal(
    tmp_path,
) -> None:
    _application()
    bridge = AgentAuthoringBridge(FakeAuthoringPort())
    bridge.bind_context(_context())
    controller, _calls, _queries = _controller(bridge)
    controller.dispatch(
        "set_authoring_requirements",
        {
            "turn_id": "turn-a8",
            "requirements": _requirements_for("geometry"),
        },
        ToolExecutionContext("session-a8", 0, "requirements-before-reset"),
    )
    prepared = controller.dispatch(
        "prepare_geometry_proposal",
        {},
        ToolExecutionContext("session-a8", 0, "geometry-before-reset"),
    )
    proposal_id = prepared.data["proposal_id"]
    assert controller.stage is AuthoringWorkflowStage.GEOMETRY_PENDING
    assert bridge.state(proposal_id) is ProposalState.PENDING_CONFIRMATION

    runtime = QtAgentRuntime(
        tmp_path / "agent-private-session-reset",
        provider_factory=FakeProvider,
        authoring_controller=controller,
    )
    drawer = AgentChatDrawer(
        agent_runtime=runtime,
        authoring_bridge=bridge,
    )

    assert runtime.new_session()
    _wait_until(lambda: not runtime.busy)

    assert controller.stage is AuthoringWorkflowStage.REQUIREMENTS
    assert controller.ledger.entries == ()
    assert bridge.state(proposal_id) is ProposalState.STALE
    drawer.close()
    runtime.shutdown()


def test_a8_production_main_window_injects_the_real_controller() -> None:
    _application()
    window = FEMMainWindow()
    runtime = window.viewport_panel.agent_chat_drawer.agent_runtime

    assert runtime.authoring_controller is window.agent_authoring_controller
    assert runtime.authoring_controller is not None
    assert "read_authoring_context" in {
        item.name for item in runtime.authoring_controller.definitions
    }

    runtime.shutdown()
    window.close()


def test_a8_production_geometry_waits_for_one_gui_acceptance() -> None:
    _application()
    window = FEMMainWindow()
    controller = window.agent_authoring_controller
    bridge = window.agent_authoring_bridge
    before = window.session.snapshot()
    assert before.source_kind is None

    recorded = controller.dispatch(
        "set_authoring_requirements",
        {
            "turn_id": "turn-a8",
            "requirements": _requirements_for("geometry"),
        },
        ToolExecutionContext("session-a8", 0, "requirements-a8"),
    )
    prepared = controller.dispatch(
        "prepare_geometry_proposal",
        {},
        ToolExecutionContext("session-a8", 0, "geometry-a8"),
    )
    proposal_id = prepared.data["proposal_id"]

    draft_state = window.session.snapshot()
    assert recorded.ok and prepared.ok
    assert controller.pending_review is None
    assert controller.stage is AuthoringWorkflowStage.GEOMETRY_PENDING
    assert draft_state.session_revision == before.session_revision
    assert draft_state.parts == ()

    accepted = bridge.accept_from_gui_control(proposal_id)
    controller.record_proposal_state("geometry", accepted.state)
    after = window.session.snapshot()

    assert accepted.state is ProposalState.SUCCEEDED
    assert controller.stage is AuthoringWorkflowStage.MESH_READY
    assert after.source_kind == "native"
    assert after.session_revision == before.session_revision + 1
    assert [part.name for part in after.parts] == ["部件-偏心孔板"]

    window.viewport_panel.agent_chat_drawer.agent_runtime.shutdown()
    window.close()


def test_a8_direct_definition_actions_apply_one_by_one_and_refresh_gui() -> None:
    session = _a4_session()
    projections = []
    controller_holder = {}

    def project_definition_delta(delta) -> None:
        projections.append(delta)
        stale_ids = bridge.bind_snapshot(session.snapshot())
        controller_holder["controller"].observe_binding(
            bridge.context,
            proposal_staled=bool(stale_ids),
        )

    port = SessionGeometryAuthoringPort(
        session,
        lambda: None,
        apply_definition_delta=project_definition_delta,
    )
    bridge = AgentAuthoringBridge(port)
    bridge.bind_snapshot(session.snapshot())
    controller = create_session_authoring_workflow_controller(
        session,
        bridge,
        AgentResultQueryBridge(SessionResultQueryPort(session)),
    )
    controller_holder["controller"] = controller
    controller._stage = AuthoringWorkflowStage.DEFINITIONS_READY
    before_revision = session.session_revision

    actions = [
        ("create_plate_scopes", {}),
        (
            "create_material",
            {
                "name": "材料-铝合金",
                "properties": {"E": 70000.0, "nu": 0.33},
            },
        ),
        (
            "create_section",
            {
                "name": "截面-平面应力",
                "material": "材料-铝合金",
                "plane_type": "stress",
                "thickness": 1.0,
            },
        ),
        (
            "assign_section",
            {
                "section_name": "截面-平面应力",
                "region_name": "域-板体",
            },
        ),
        ("create_static_step", {"name": "分析步-静力"}),
        (
            "create_boundary_condition",
            {
                "name": "位移-固定端",
                "step_name": "分析步-静力",
                "target_scope": "边-固定端",
                "target_kind": "edge",
                "first_component": 1,
                "last_component": 2,
                "value": 0.0,
            },
        ),
        (
            "create_load",
            {
                "name": "载荷-拉伸",
                "step_name": "分析步-静力",
                "target_scope": "边-加载端",
                "load_type": "edge_traction",
                "vector": [10.0, 0.0],
            },
        ),
        (
            "create_result_request",
            {
                "name": "结果请求-位移反力",
                "step_name": "分析步-静力",
                "target": "node",
                "variables": ["U", "RF"],
            },
        ),
    ]

    results = []
    for index, (action, parameters) in enumerate(actions):
        result = controller.dispatch(
            "apply_model_definition",
            {"action": action, "parameters": parameters},
            ToolExecutionContext(
                "session-a8",
                session.session_revision,
                f"direct-{index}",
            ),
        )
        assert result.ok, result.to_json()
        assert result.data["gui_synchronized"] is True
        assert "proposal_id" not in result.data
        results.append(result)

    after = session.snapshot()
    assert after.session_revision == before_revision + len(actions)
    assert len(projections) == len(actions)
    assert set(after.named_regions) == {
        "边-固定端",
        "边-加载端",
        "边-孔边",
        "域-板体",
    }
    assert [item.name for item in after.materials] == ["材料-铝合金"]
    assert [item.name for item in after.sections] == ["截面-平面应力"]
    assert [item.name for item in after.steps] == ["分析步-静力"]
    assert len(after.steps[0].boundaries) == 1
    assert len(after.steps[0].edge_loads) == 1
    assert len(after.steps[0].outputs) == 1
    assert controller.stage is AuthoringWorkflowStage.DEFINITIONS_READY
