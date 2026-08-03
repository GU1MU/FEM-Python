from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QObject, Qt, Slot
from PySide6.QtTest import QSignalSpy, QTest
from PySide6.QtWidgets import QApplication, QWidget

from fem_agent.engine import (
    AgentSessionEngine,
    EngineEvent,
    EngineEventType,
)
from fem_agent.providers.base import (
    AssistantMessage,
    ProviderResponse,
    ToolCall,
)
from fem_agent.providers.fake import FakeProvider
from fem_agent.tools.registry import AgentToolRegistry
from fem_agent.worker import IsolatedFEMWorker
from fem_gui.agent_events import (
    AgentEvent,
    AgentEventProjector,
    EventType,
    TurnStatus,
)
from fem_gui.agent_runtime import QtAgentRuntime
from fem_gui.agent_workspace import (
    WorkspaceCommandHandler,
    WorkspaceFileReference,
    build_workspace_file_reference,
    normalize_user_workspace,
)
from fem_gui.widgets.agent_chat import (
    AgentChatDrawer,
    ModelViewportOverlayHost,
)
from tests.helpers.abaqus_builders import (
    write_perforated_plate_style_inp,
)


def _application() -> QApplication:
    return QApplication.instance() or QApplication([])


def _wait_until(predicate, *, timeout_ms: int = 15_000) -> None:
    deadline = time.monotonic() + timeout_ms / 1_000
    application = _application()
    while not predicate() and time.monotonic() < deadline:
        application.processEvents()
        QTest.qWait(10)
    application.processEvents()
    assert predicate()


class _EventCollector(QObject):
    def __init__(self) -> None:
        super().__init__()
        self.events: list[AgentEvent] = []
        self.thread_ids: list[int] = []

    @Slot(object)
    def receive(self, event: AgentEvent) -> None:
        self.events.append(event)
        self.thread_ids.append(threading.get_ident())


class _RecordingEngine:
    def __init__(
        self,
        root: Path,
        provider,
        event_sink,
        calls: dict[str, list[int]],
    ) -> None:
        calls.setdefault("ctor", []).append(threading.get_ident())
        self._calls = calls
        self._delegate = AgentSessionEngine(
            root,
            provider,
            event_sink=event_sink,
        )

    @property
    def session_id(self) -> str:
        return self._delegate.session_id

    def reset_operation_start_signal(self) -> None:
        self._delegate.reset_operation_start_signal()

    def wait_for_operation_start(
        self,
        timeout_seconds: float | None = None,
    ) -> bool:
        return self._delegate.wait_for_operation_start(timeout_seconds)

    def send_message(self, text: str, *, request_context=None):
        self._calls.setdefault("send", []).append(threading.get_ident())
        return self._delegate.send_message(
            text,
            request_context=request_context,
        )

    def create_session(self):
        self._calls.setdefault("new", []).append(threading.get_ident())
        return self._delegate.create_session()

    def attach_artifact(self, artifact_id, *, replace_existing=False):
        self._calls.setdefault("attach", []).append(threading.get_ident())
        return self._delegate.attach_artifact(
            artifact_id,
            replace_existing=replace_existing,
        )

    def confirm_revision(self):
        self._calls.setdefault("confirm", []).append(threading.get_ident())
        return self._delegate.confirm_revision()

    def cancel_active_operation(self):
        self._calls.setdefault("cancel", []).append(threading.get_ident())
        return self._delegate.cancel_active_operation()

    def close_session(self):
        self._calls.setdefault("close", []).append(threading.get_ident())
        return self._delegate.close_session()

    def get_snapshot(self):
        return self._delegate.get_snapshot()


class _CancellableFakeProvider(FakeProvider):
    def __init__(self) -> None:
        super().__init__()
        self.started = threading.Event()
        self.release = threading.Event()
        self.cancel_calls = 0

    def complete(self, messages, tools):
        from fem_agent.providers.fake import CapturedRequest

        self.requests.append(CapturedRequest(tuple(messages), tuple(tools)))
        self.started.set()
        self.release.wait(timeout=5.0)
        return ProviderResponse(
            AssistantMessage("assistant", "这是一条晚到回复。"),
            finish_reason="stop",
        )

    def cancel_active_request(self) -> None:
        self.cancel_calls += 1
        self.release.set()


class _ConfirmableEngine:
    def __init__(self, _root, _provider, event_sink) -> None:
        self.session_id = "ses_confirmable"
        self.revision = 7
        self.revision_hash = "a" * 64
        self._event_sink = event_sink
        self._operation_started = threading.Event()
        self.confirm_calls = 0

    def _emit(self, event_type, data):
        event = EngineEvent(
            event_type,
            self.session_id,
            data,
            "2026-07-30T00:00:00Z",
        )
        self._event_sink(event)
        return event

    def reset_operation_start_signal(self) -> None:
        self._operation_started.clear()

    def wait_for_operation_start(
        self,
        timeout_seconds: float | None = None,
    ) -> bool:
        return self._operation_started.wait(timeout_seconds)

    def send_message(self, _text, *, request_context=None):
        del request_context
        self._operation_started.set()
        events = (
            self._emit(
                EngineEventType.ANALYSIS_SUMMARY,
                {
                    "analysis_summary": {
                        "revision": self.revision,
                        "revision_hash": self.revision_hash,
                        "model_name": "frame",
                        "node_count": 4,
                        "element_count": 3,
                        "analysis_step": {"name": "Step-1"},
                        "diagnostics": [],
                    }
                },
            ),
            self._emit(
                EngineEventType.MESSAGE_DELTA,
                {"text": "模型已准备好，可确认求解。"},
            ),
        )
        return events

    def create_session(self):
        raise AssertionError("not used by this test")

    def attach_artifact(self, _artifact_id, *, replace_existing=False):
        del replace_existing
        return ()

    def confirm_revision(self):
        self._operation_started.set()
        self.confirm_calls += 1
        return (
            self._emit(
                EngineEventType.RUN_COMPLETED,
                {
                    "status": "succeeded",
                    "run_id": "run_confirmed",
                    "artifacts": [{"artifact_id": "result-summary"}],
                },
            ),
        )

    def cancel_active_operation(self):
        return ()

    def close_session(self):
        return ()

    def get_snapshot(self):
        return SimpleNamespace(
            revision=self.revision,
            revision_hash=self.revision_hash,
        )


def _workspace_handler(
    user_workspace: Path,
    agent_data_root: Path,
) -> WorkspaceCommandHandler:
    return WorkspaceCommandHandler(
        directory_chooser=lambda _parent, _initial: user_workspace,
        agent_data_root=agent_data_root,
    )


def _reference(
    user_workspace: Path,
    relative_path: str = "frame.inp",
) -> WorkspaceFileReference:
    reference = build_workspace_file_reference(
        normalize_user_workspace(user_workspace),
        relative_path,
    )
    return reference.at_text_range(3, 3 + len(reference.mention_text))


def _write_provider_config(
    path: Path,
    *,
    enabled: bool,
) -> None:
    path.write_text(
        json.dumps(
            {
                "provider": "fake",
                "model": "fake-phase5",
                "base_url": "https://example.invalid",
                "enabled": enabled,
            }
        ),
        encoding="utf-8",
    )


def test_fake_engine_runs_off_main_thread_and_ui_consumes_queued_events(
    tmp_path,
):
    _application()
    main_thread_id = threading.get_ident()
    user_workspace = tmp_path / "user-workspace"
    user_workspace.mkdir()
    source = user_workspace / "notes.txt"
    source.write_text("PRIVATE-FILE-CONTENT", encoding="utf-8")
    before = {path.relative_to(user_workspace) for path in user_workspace.rglob("*")}
    agent_root = tmp_path / "agent-private"
    fake = FakeProvider(
        [
            ProviderResponse(
                AssistantMessage(
                    "assistant",
                    "**Fake Engine 已连接。**",
                ),
                finish_reason="stop",
            )
        ]
    )
    calls: dict[str, list[int]] = {}

    def engine_factory(root, provider, sink):
        return _RecordingEngine(root, provider, sink, calls)

    runtime = QtAgentRuntime(
        agent_root,
        provider_factory=lambda: fake,
        engine_factory=engine_factory,
    )
    handler = _workspace_handler(user_workspace, agent_root)
    drawer = AgentChatDrawer(
        workspace_commands=handler,
        agent_runtime=runtime,
    )
    assert str(agent_root.resolve()) not in drawer.workspace_state.toolTip()
    applied_thread_ids: list[int] = []
    drawer.agentEventApplied.connect(
        lambda _event: applied_thread_ids.append(threading.get_ident())
    )
    drawer.show()
    workspace_variant = str(user_workspace.resolve()).upper().replace("\\", "/")
    private_variant = str(agent_root.resolve()).upper().replace("\\", "/")
    drawer.input.setPlainText(
        f"检查 @notes.txt；本地路径 {workspace_variant} {private_variant}"
    )
    drawer._workspace_references.append(
        _reference(user_workspace, "notes.txt")
    )
    drawer.send_button.click()

    _wait_until(lambda: not runtime.busy)
    _wait_until(
        lambda: (
            drawer.event_presentation.turns
            and drawer.event_presentation.turns[-1].status is TurnStatus.COMPLETED
        )
    )

    assert calls["ctor"][0] != main_thread_id
    assert calls["send"][0] != main_thread_id
    assert applied_thread_ids
    assert set(applied_thread_ids) == {main_thread_id}
    event_log = drawer.event_projector.export_event_log()
    assert [record["sequence"] for record in event_log] == list(
        range(1, len(event_log) + 1)
    )
    replayed = AgentEventProjector.restore_event_log(event_log)
    assert replayed.presentation == drawer.event_presentation

    request = fake.requests[0]
    provider_text = [
        message.content for message in request.messages if message.role == "user"
    ][-1]
    assert provider_text == (
        "检查 @notes.txt；本地路径 "
        "<本地路径已隐藏> <本地路径已隐藏>"
    )
    serialized_request = repr(request)
    assert str(user_workspace.resolve()) not in serialized_request
    assert str(agent_root.resolve()) not in serialized_request
    assert "PRIVATE-FILE-CONTENT" in serialized_request
    assert request.tools
    assert drawer.input.isEnabled()
    assert drawer.send_state.currentWidget() is drawer.send_button
    assert {
        path.relative_to(user_workspace) for path in user_workspace.rglob("*")
    } == before
    assert agent_root.exists()
    conversation = next(
        (agent_root / "sessions").glob("*/conversation.json")
    ).read_text(encoding="utf-8")
    assert "PRIVATE-FILE-CONTENT" not in conversation

    drawer.close()
    assert not runtime.is_shutdown
    drawer.shutdown_runtime()
    assert calls["close"][0] != main_thread_id
    assert runtime.is_shutdown
    assert all(not thread.is_alive() for thread in runtime._session_executor._threads)


def test_default_provider_factory_honors_enabled_config(
    tmp_path,
    monkeypatch,
):
    config_path = tmp_path / "fem-agent.config.json"
    _write_provider_config(config_path, enabled=True)
    monkeypatch.setattr(
        "fem_gui.agent_runtime.find_main_config",
        lambda: config_path,
    )
    runtime = QtAgentRuntime(tmp_path / "agent-private")
    collector = _EventCollector()
    runtime.agentEventReady.connect(
        collector.receive,
        Qt.ConnectionType.QueuedConnection,
    )
    provider_ready = QSignalSpy(runtime.providerReady)

    assert runtime.send_message("检查配置化 Provider")
    _wait_until(lambda: not runtime.busy)
    _wait_until(
        lambda: any(
            event.event_type is EventType.TURN_COMPLETE
            for event in collector.events
        )
    )

    assert provider_ready.count() == 1
    assert list(provider_ready.at(0)) == ["fake", "fake-phase5"]
    runtime.shutdown()


def test_default_provider_factory_rejects_disabled_config_locally(
    tmp_path,
    monkeypatch,
):
    config_path = tmp_path / "fem-agent.config.json"
    _write_provider_config(config_path, enabled=False)
    monkeypatch.setattr(
        "fem_gui.agent_runtime.find_main_config",
        lambda: config_path,
    )
    runtime = QtAgentRuntime(tmp_path / "agent-private")
    collector = _EventCollector()
    runtime.agentEventReady.connect(
        collector.receive,
        Qt.ConnectionType.QueuedConnection,
    )

    assert runtime.send_message("不应调用 Provider")
    _wait_until(lambda: not runtime.busy)
    _wait_until(
        lambda: any(
            event.event_type is EventType.TURN_FAILED
            for event in collector.events
        )
    )

    diagnostic = next(
        event
        for event in collector.events
        if event.event_type is EventType.DIAGNOSTIC
    )
    assert diagnostic.payload["code"] == "GUI-AGENT-CONFIG"
    assert "enabled" in diagnostic.payload["message"]
    assert not (tmp_path / "agent-private").exists()
    runtime.shutdown()


def test_inp_reference_is_attached_locally_without_provider_body(
    tmp_path,
):
    user_workspace = tmp_path / "user-workspace"
    user_workspace.mkdir()
    source = write_perforated_plate_style_inp(
        user_workspace,
        "frame.inp",
        ("*Cload", "Set-right, 1, 10."),
    )
    source_bytes = source.read_bytes()
    source_text = source.read_text(encoding="utf-8")
    fake = FakeProvider(
        [
            ProviderResponse(
                AssistantMessage("assistant", "模型输入已附加。"),
                finish_reason="stop",
            )
        ]
    )
    agent_root = tmp_path / "agent-private"
    runtime = QtAgentRuntime(
        agent_root,
        provider_factory=lambda: fake,
    )
    collector = _EventCollector()
    runtime.agentEventReady.connect(
        collector.receive,
        Qt.ConnectionType.QueuedConnection,
    )

    assert runtime.send_message(
        "检查 @frame.inp",
        (_reference(user_workspace),),
        workspace_root=user_workspace,
    )
    _wait_until(lambda: not runtime.busy, timeout_ms=45_000)
    _wait_until(
        lambda: any(
            event.event_type is EventType.TURN_COMPLETE
            for event in collector.events
        )
    )

    serialized_request = repr(fake.requests[0])
    assert "local_fem_input" in serialized_request
    assert source_text not in serialized_request
    assert str(source.resolve()) not in serialized_request
    private_copy = next(
        (agent_root / "sessions").glob("*/inputs/*/frame.inp")
    )
    assert private_copy.read_bytes() == source_bytes
    assert source.read_bytes() == source_bytes
    runtime.shutdown()


def test_provider_tool_call_reaches_registry_and_projects_tool_events(
    tmp_path,
    monkeypatch,
):
    fake = FakeProvider(
        [
            ProviderResponse(
                AssistantMessage(
                    "assistant",
                    tool_calls=(
                        ToolCall("call-1", "show_capabilities", {}),
                    ),
                ),
                finish_reason="tool_calls",
            ),
            ProviderResponse(
                AssistantMessage(
                    "assistant",
                    "能力检查完成。",
                ),
                finish_reason="stop",
            ),
        ]
    )
    dispatched: list[str] = []
    worker_runs: list[bool] = []
    original_dispatch = AgentToolRegistry.dispatch

    def recording_dispatch(self, name, *args, **kwargs):
        dispatched.append(name)
        return original_dispatch(
            self,
            name,
            *args,
            **kwargs,
        )

    monkeypatch.setattr(
        AgentToolRegistry,
        "dispatch",
        recording_dispatch,
    )
    monkeypatch.setattr(
        IsolatedFEMWorker,
        "run",
        lambda _self, *_args, **_kwargs: worker_runs.append(True),
    )
    runtime = QtAgentRuntime(
        tmp_path / "agent-private",
        provider_factory=lambda: fake,
    )
    collector = _EventCollector()
    runtime.agentEventReady.connect(
        collector.receive,
        Qt.ConnectionType.QueuedConnection,
    )

    assert runtime.send_message("检查当前能力")
    _wait_until(lambda: not runtime.busy)
    _wait_until(
        lambda: any(
            event.event_type is EventType.TURN_COMPLETE
            for event in collector.events
        )
    )

    assert dispatched == ["show_capabilities"]
    assert worker_runs == []
    assert len(fake.requests) == 2
    assert [event.event_type for event in collector.events] == [
        EventType.TURN_STARTED,
        EventType.TOOL_REQUESTED,
        EventType.TOOL_STARTED,
        EventType.TOOL_RESULT,
        EventType.MESSAGE_START,
        EventType.MESSAGE_DELTA,
        EventType.MESSAGE_COMPLETE,
        EventType.TURN_COMPLETE,
    ]
    projected = AgentEventProjector.replay(collector.events).presentation
    tool = projected.turns[-1].tool_groups[0].calls[0]
    assert tool.tool_name == "show_capabilities"
    assert tool.display_name == "检查 Agent 能力"
    runtime.shutdown()


def test_tool_diagnostic_stays_inside_tool_details(tmp_path):
    fake = FakeProvider(
        [
            ProviderResponse(
                AssistantMessage(
                    "assistant",
                    tool_calls=(
                        ToolCall(
                            "call-invalid-units",
                            "set_unit_context",
                            {
                                "length": "mm",
                                "force": "N",
                                "stress": "MPa",
                                "density": "tonne/mm^3",
                                "acceleration": "mm/s^2",
                            },
                        ),
                    ),
                ),
                finish_reason="tool_calls",
            ),
            ProviderResponse(
                AssistantMessage(
                    "assistant",
                    "请继续提供建模参数。",
                ),
                finish_reason="stop",
            ),
        ]
    )
    runtime = QtAgentRuntime(
        tmp_path / "agent-private",
        provider_factory=lambda: fake,
    )
    collector = _EventCollector()
    runtime.agentEventReady.connect(
        collector.receive,
        Qt.ConnectionType.QueuedConnection,
    )

    assert runtime.send_message("记录单位")
    _wait_until(lambda: not runtime.busy)
    _wait_until(
        lambda: any(
            event.event_type is EventType.TURN_COMPLETE
            for event in collector.events
        )
    )

    assert EventType.TOOL_FAILED in {
        event.event_type for event in collector.events
    }
    assert EventType.DIAGNOSTIC not in {
        event.event_type for event in collector.events
    }
    projected = AgentEventProjector.replay(collector.events).presentation
    call = projected.turns[-1].tool_groups[0].calls[0]
    assert call.diagnostics
    assert "RevisionNotFoundError" in call.diagnostics[0]
    runtime.shutdown()


def test_text_around_tool_loop_keeps_message_tool_message_timeline(
    tmp_path,
):
    fake = FakeProvider(
        [
            ProviderResponse(
                AssistantMessage(
                    "assistant",
                    "先检查能力。",
                    tool_calls=(
                        ToolCall("call-1", "show_capabilities", {}),
                    ),
                ),
                finish_reason="tool_calls",
            ),
            ProviderResponse(
                AssistantMessage("assistant", "能力检查完成。"),
                finish_reason="stop",
            ),
        ]
    )
    runtime = QtAgentRuntime(
        tmp_path / "agent-private",
        provider_factory=lambda: fake,
    )
    collector = _EventCollector()
    runtime.agentEventReady.connect(
        collector.receive,
        Qt.ConnectionType.QueuedConnection,
    )

    assert runtime.send_message("检查能力")
    _wait_until(lambda: not runtime.busy)
    _wait_until(
        lambda: any(
            event.event_type is EventType.TURN_COMPLETE
            for event in collector.events
        )
    )

    presentation = AgentEventProjector.replay(
        collector.events
    ).presentation
    turn = presentation.turns[-1]
    assert [message.text for message in turn.messages] == [
        "先检查能力。",
        "能力检查完成。",
    ]
    assert [
        item.kind.value
        for item in turn.timeline
    ] == ["message", "tool_group", "message"]
    event_types = [event.event_type for event in collector.events]
    first_complete = event_types.index(EventType.MESSAGE_COMPLETE)
    assert event_types[first_complete - 1 : first_complete + 3] == [
        EventType.MESSAGE_DELTA,
        EventType.MESSAGE_COMPLETE,
        EventType.TOOL_REQUESTED,
        EventType.TOOL_STARTED,
    ]
    runtime.shutdown()


def test_revision_bound_gui_confirmation_starts_solve_and_projects_result(
    tmp_path,
):
    engine_holder = {}

    def engine_factory(root, provider, sink):
        engine = _ConfirmableEngine(root, provider, sink)
        engine_holder["engine"] = engine
        return engine

    runtime = QtAgentRuntime(
        tmp_path / "agent-private",
        provider_factory=FakeProvider,
        engine_factory=engine_factory,
    )
    collector = _EventCollector()
    runtime.agentEventReady.connect(
        collector.receive,
        Qt.ConnectionType.QueuedConnection,
    )
    solve_finished = QSignalSpy(runtime.solveFinished)

    assert runtime.send_message("检查模型并给出分析摘要")
    _wait_until(lambda: not runtime.busy)
    _wait_until(
        lambda: any(
            event.event_type is EventType.CONFIRMATION_REQUESTED
            for event in collector.events
        )
    )
    confirmation = next(
        event
        for event in collector.events
        if event.event_type is EventType.CONFIRMATION_REQUESTED
    )
    revision = confirmation.payload["revision"]
    revision_hash = confirmation.payload["revision_hash"]
    before_confirm = len(collector.events)
    summary_turn = AgentEventProjector.replay(
        collector.events
    ).presentation.turns[-1]
    assert [
        item.kind.value
        for item in summary_turn.timeline
    ] == ["message", "confirmation"]

    assert engine_holder["engine"].confirm_calls == 0
    assert runtime.confirm_solve(revision, revision_hash)
    _wait_until(lambda: not runtime.busy)
    _wait_until(lambda: solve_finished.count() == 1)

    assert engine_holder["engine"].confirm_calls == 1
    assert list(solve_finished.at(0)) == [
        revision,
        revision_hash,
        True,
    ]
    solve_events = collector.events[before_confirm:]
    assert [event.event_type for event in solve_events] == [
        EventType.TURN_STARTED,
        EventType.TOOL_REQUESTED,
        EventType.TOOL_STARTED,
        EventType.TOOL_RESULT,
        EventType.TURN_COMPLETE,
    ]
    projected = AgentEventProjector.replay(collector.events).presentation
    solve_turn = projected.turns[-1]
    assert solve_turn.status is TurnStatus.COMPLETED
    assert solve_turn.tool_groups[0].calls[0].tool_name == (
        "solve_confirmed_analysis"
    )
    runtime.shutdown()


def test_busy_session_rejects_duplicate_send_and_drawer_hide_does_not_cancel(
    tmp_path,
):
    application = _application()
    provider = _CancellableFakeProvider()
    agent_root = tmp_path / "agent-private"
    runtime = QtAgentRuntime(
        agent_root,
        provider_factory=lambda: provider,
    )
    viewport = QWidget()
    host = ModelViewportOverlayHost(
        viewport,
        workspace_commands=_workspace_handler(
            tmp_path,
            agent_root,
        ),
        agent_runtime=runtime,
    )
    host.resize(700, 450)
    host.show()
    application.processEvents()
    baseline_viewport_geometry = viewport.geometry()
    collector = _EventCollector()
    runtime.agentEventReady.connect(
        collector.receive,
        Qt.ConnectionType.QueuedConnection,
    )
    rejected = QSignalSpy(runtime.operationRejected)
    event_rejections = QSignalSpy(runtime.eventRejected)

    assert runtime.send_message("第一条")
    assert not runtime.send_message("第二条")
    assert rejected.count() == 1
    assert provider.started.wait(timeout=3.0)
    duplicate = EngineEvent(
        EngineEventType.DIAGNOSTIC,
        runtime.session_id,
        {
            "diagnostic": {
                "code": "TEST-DIAGNOSTIC",
                "severity": "info",
                "message": "仅用于重复事件边界测试。",
            }
        },
        "2026-07-30T00:00:00Z",
    )
    runtime._receive_engine_event(duplicate)
    runtime._receive_engine_event(duplicate)
    runtime._receive_engine_event(
        EngineEvent(
            EngineEventType.DIAGNOSTIC,
            "other-session",
            duplicate.data,
            "2026-07-30T00:00:01Z",
        )
    )
    runtime._receive_engine_event(
        EngineEvent(
            duplicate.event,
            duplicate.session_id,
            duplicate.data,
            duplicate.timestamp,
        )
    )
    application.processEvents()
    assert event_rejections.count() == 2
    assert len(runtime._active_turn.seen_engine_events) == 2

    host.set_drawer_open(False, animated=False)
    application.processEvents()
    assert runtime.busy
    assert provider.cancel_calls == 0
    assert len(provider.requests) == 1

    host.set_drawer_open(True, animated=False)
    host.agent_chat_drawer.stop_button.click()
    _wait_until(lambda: not runtime.busy)
    _wait_until(
        lambda: any(
            event.event_type is EventType.TURN_CANCELLED for event in collector.events
        )
    )
    assert provider.cancel_calls == 1
    assert viewport.geometry() == baseline_viewport_geometry
    assert not any(
        event.event_type is EventType.MESSAGE_DELTA for event in collector.events
    )

    rejected_before_manual_late_event = event_rejections.count()
    runtime._receive_engine_event(
        EngineEvent(
            EngineEventType.MESSAGE_DELTA,
            runtime.session_id,
            {"text": "late"},
            "2026-07-30T00:00:00Z",
        )
    )
    application.processEvents()
    assert event_rejections.count() == rejected_before_manual_late_event + 1
    assert [event.sequence for event in collector.events] == list(
        range(1, len(collector.events) + 1)
    )
    AgentEventProjector.replay(collector.events)
    host.close()


def test_new_session_resets_projector_and_runs_in_background(tmp_path):
    application = _application()
    fake = FakeProvider(
        [
            ProviderResponse(
                AssistantMessage("assistant", "第一轮"),
                finish_reason="stop",
            )
        ]
    )
    calls: dict[str, list[int]] = {}

    def engine_factory(root, provider, sink):
        return _RecordingEngine(root, provider, sink, calls)

    runtime = QtAgentRuntime(
        tmp_path / "agent-private",
        provider_factory=lambda: fake,
        engine_factory=engine_factory,
    )
    drawer = AgentChatDrawer(agent_runtime=runtime)
    drawer.show()
    assert runtime.send_message("第一轮")
    _wait_until(lambda: not runtime.busy)
    _wait_until(lambda: bool(drawer.event_presentation.turns))
    first_session = runtime.session_id

    assert runtime.new_session()
    _wait_until(lambda: not runtime.busy)
    _wait_until(lambda: runtime.session_id != first_session)
    application.processEvents()

    assert calls["new"]
    assert calls["new"][0] != threading.get_ident()
    assert drawer.event_presentation.turns == []
    assert drawer.event_presentation.session_id == ""
    drawer.close()
    assert not runtime.is_shutdown
    drawer.shutdown_runtime()


def test_provider_failure_is_diagnostic_and_failed_turn(tmp_path):
    runtime = QtAgentRuntime(
        tmp_path / "agent-private",
        provider_factory=lambda: FakeProvider(
            [RuntimeError("private provider detail")]
        ),
    )
    collector = _EventCollector()
    runtime.agentEventReady.connect(
        collector.receive,
        Qt.ConnectionType.QueuedConnection,
    )

    assert runtime.send_message("触发 Fake 失败")
    _wait_until(lambda: not runtime.busy)
    _wait_until(lambda: len(collector.events) >= 3)

    assert [event.event_type for event in collector.events] == [
        EventType.TURN_STARTED,
        EventType.DIAGNOSTIC,
        EventType.TURN_FAILED,
    ]
    assert "private provider detail" not in repr(collector.events)
    runtime.shutdown()


def test_selected_workspace_root_is_redacted_without_file_reference(tmp_path):
    _application()
    user_workspace = tmp_path / "selected-workspace"
    user_workspace.mkdir()
    agent_root = tmp_path / "agent-private"
    fake = FakeProvider(
        [
            ProviderResponse(
                AssistantMessage("assistant", "路径已隐藏。"),
                finish_reason="stop",
            )
        ]
    )
    handler = _workspace_handler(user_workspace, agent_root)
    selection = handler.execute("/workspace")
    assert selection.succeeded
    runtime = QtAgentRuntime(
        agent_root,
        provider_factory=lambda: fake,
    )
    drawer = AgentChatDrawer(
        workspace_commands=handler,
        agent_runtime=runtime,
    )
    drawer.input.setPlainText(
        "不要发送 "
        + str(user_workspace.resolve()).upper().replace("\\", "/")
        + "；也不要发送 D:\\outside\\secret.inp"
    )
    drawer.send_button.click()

    _wait_until(lambda: not runtime.busy)

    provider_text = [
        message.content
        for message in fake.requests[0].messages
        if message.role == "user"
    ][-1]
    assert provider_text == (
        "不要发送 <本地路径已隐藏>；"
        "也不要发送 <绝对路径已隐藏>"
    )
    assert str(user_workspace.resolve()) not in repr(
        drawer.event_projector.export_event_log()
    )
    drawer.shutdown_runtime()


def test_long_fake_reply_is_split_into_replayable_message_deltas(tmp_path):
    long_reply = "有限元🙂" * 5_000
    runtime = QtAgentRuntime(
        tmp_path / "agent-private",
        provider_factory=lambda: FakeProvider(
            [
                ProviderResponse(
                    AssistantMessage("assistant", long_reply),
                    finish_reason="stop",
                )
            ]
        ),
    )
    collector = _EventCollector()
    runtime.agentEventReady.connect(
        collector.receive,
        Qt.ConnectionType.QueuedConnection,
    )

    assert runtime.send_message("返回较长文本")
    _wait_until(lambda: not runtime.busy)
    _wait_until(
        lambda: any(
            event.event_type is EventType.TURN_COMPLETE
            for event in collector.events
        )
    )

    deltas = [
        event.payload["delta"]
        for event in collector.events
        if event.event_type is EventType.MESSAGE_DELTA
    ]
    assert len(deltas) > 1
    assert "".join(deltas) == long_reply
    replayed = AgentEventProjector.replay(collector.events)
    assert replayed.presentation.turns[-1].messages[-1].text == long_reply
    runtime.shutdown()


def test_runtime_coalesces_high_frequency_provider_chunks(tmp_path):
    class BurstStreamingProvider(FakeProvider):
        def __init__(self) -> None:
            super().__init__()
            self.reply = "流" * 1_000

        def complete_stream(self, messages, tools, on_text_delta):
            del messages, tools
            waiter = threading.Event()
            for batch in range(4):
                for character in self.reply[batch * 250 : (batch + 1) * 250]:
                    on_text_delta(character)
                waiter.wait(0.035)
            return ProviderResponse(
                AssistantMessage("assistant", self.reply),
                finish_reason="stop",
            )

    provider = BurstStreamingProvider()
    runtime = QtAgentRuntime(
        tmp_path / "agent-private",
        provider_factory=lambda: provider,
    )
    collector = _EventCollector()
    runtime.agentEventReady.connect(
        collector.receive,
        Qt.ConnectionType.QueuedConnection,
    )

    assert runtime.send_message("逐字符返回")
    _wait_until(lambda: not runtime.busy)
    _wait_until(
        lambda: any(
            event.event_type is EventType.TURN_COMPLETE
            for event in collector.events
        )
    )

    deltas = [
        event.payload["delta"]
        for event in collector.events
        if event.event_type is EventType.MESSAGE_DELTA
    ]
    assert 3 <= len(deltas) <= 8
    assert "".join(deltas) == provider.reply
    replayed = AgentEventProjector.replay(collector.events)
    assert replayed.presentation.turns[-1].messages[-1].text == provider.reply
    runtime.shutdown()


def test_runtime_adapts_delta_batch_to_backlog_and_reports_metrics(tmp_path):
    class BackloggedStreamingProvider(FakeProvider):
        def __init__(self) -> None:
            super().__init__()
            self.reply = "积" * 10_000

        def complete_stream(self, messages, tools, on_text_delta):
            del messages, tools
            on_text_delta(self.reply)
            threading.Event().wait(0.06)
            return ProviderResponse(
                AssistantMessage("assistant", self.reply),
                finish_reason="stop",
            )

    provider = BackloggedStreamingProvider()
    runtime = QtAgentRuntime(
        tmp_path / "agent-private",
        provider_factory=lambda: provider,
    )
    collector = _EventCollector()
    runtime.agentEventReady.connect(
        collector.receive,
        Qt.ConnectionType.QueuedConnection,
    )

    assert runtime.send_message("制造待刷新积压")
    _wait_until(lambda: not runtime.busy)
    _wait_until(
        lambda: any(
            event.event_type is EventType.TURN_COMPLETE
            for event in collector.events
        )
    )

    deltas = [
        event.payload["delta"]
        for event in collector.events
        if event.event_type is EventType.MESSAGE_DELTA
    ]
    assert "".join(deltas) == provider.reply
    assert len(deltas) <= 2
    assert max(map(len, deltas)) > 8_000
    metrics = runtime.stream_backlog_metrics
    assert metrics["pending_characters"] == 0
    assert metrics["max_pending_characters"] >= 8_000
    assert metrics["max_wait_seconds"] >= 0.02
    runtime.shutdown()


def test_phase5_rejects_unregistered_provider_before_complete(tmp_path):
    class ForbiddenProvider:
        provider_name = "network"
        model_name = "network-model"

        def __init__(self) -> None:
            self.called = False

        def complete(self, _messages, _tools):
            self.called = True
            raise AssertionError("network-like provider must remain unreachable")

    provider = ForbiddenProvider()
    runtime = QtAgentRuntime(
        tmp_path / "agent-private",
        provider_factory=lambda: provider,
    )
    collector = _EventCollector()
    runtime.agentEventReady.connect(
        collector.receive,
        Qt.ConnectionType.QueuedConnection,
    )

    assert runtime.send_message("保持离线")
    _wait_until(lambda: not runtime.busy)
    _wait_until(
        lambda: any(
            event.event_type is EventType.TURN_FAILED for event in collector.events
        )
    )
    assert not provider.called
    runtime.shutdown()


def test_drawer_close_keeps_active_call_until_explicit_shutdown(tmp_path):
    provider = _CancellableFakeProvider()
    runtime = QtAgentRuntime(
        tmp_path / "agent-private",
        provider_factory=lambda: provider,
    )
    drawer = AgentChatDrawer(agent_runtime=runtime)
    drawer.show()

    assert runtime.send_message("等待取消")
    assert provider.started.wait(timeout=3.0)
    drawer.close()

    assert not runtime.is_shutdown
    assert runtime.busy
    assert provider.cancel_calls == 0

    drawer.shutdown_runtime()

    assert runtime.is_shutdown
    assert provider.cancel_calls == 1
    assert all(
        not thread.is_alive()
        for thread in (
            *runtime._session_executor._threads,
            *runtime._control_executor._threads,
        )
    )
