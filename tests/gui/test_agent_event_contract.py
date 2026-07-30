from __future__ import annotations

import ast
import os
from dataclasses import dataclass
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QLabel, QToolButton, QWidget

from fem_gui.agent_events import (
    AGENT_EVENT_SCHEMA_VERSION,
    AgentEvent,
    AgentEventError,
    AgentEventProjector,
    DiagnosticSeverity,
    EventType,
    FakeAgentEventStream,
    MessageStatus,
    ToolStatus,
    TurnStatus,
    safe_tool_summary,
)
from fem_gui.widgets.agent_chat import (
    AgentChatDrawer,
    ToolActivityPreview,
    _AGENT_CHAT_STYLESHEET,
)


def _application() -> QApplication:
    return QApplication.instance() or QApplication([])


@dataclass
class _Events:
    session_id: str = "session-1"
    sequence: int = 1

    def make(
        self,
        event_type: EventType,
        payload: dict,
        *,
        turn_id: str = "turn-1",
        session_id: str | None = None,
        event_id: str | None = None,
        sequence: int | None = None,
    ) -> AgentEvent:
        current_sequence = self.sequence if sequence is None else sequence
        if sequence is None:
            self.sequence += 1
        return AgentEvent.create(
            event_id=event_id or f"event-{current_sequence}",
            session_id=session_id or self.session_id,
            turn_id=turn_id,
            sequence=current_sequence,
            event_type=event_type,
            payload=payload,
            timestamp="2026-07-29T08:00:00Z",
        )


def _turn_start(events: _Events, *, turn_id: str = "turn-1") -> AgentEvent:
    return events.make(
        EventType.TURN_STARTED,
        {"user_message": "检查模型"},
        turn_id=turn_id,
    )


def _message_events(
    events: _Events,
    *,
    turn_id: str = "turn-1",
    message_id: str = "message-1",
    deltas: tuple[str, ...] = ("第一段", "第二段"),
) -> list[AgentEvent]:
    result = [
        events.make(
            EventType.MESSAGE_START,
            {
                "message_id": message_id,
                "role": "assistant",
                "format": "restricted_markdown",
            },
            turn_id=turn_id,
        )
    ]
    result.extend(
        events.make(
            EventType.MESSAGE_DELTA,
            {"message_id": message_id, "delta": delta},
            turn_id=turn_id,
        )
        for delta in deltas
    )
    result.append(
        events.make(
            EventType.MESSAGE_COMPLETE,
            {"message_id": message_id},
            turn_id=turn_id,
        )
    )
    return result


def _tool_events(
    events: _Events,
    call_id: str,
    *,
    turn_id: str = "turn-1",
    terminal: EventType = EventType.TOOL_RESULT,
    duration_ms: int = 100,
) -> list[AgentEvent]:
    result = [
        events.make(
            EventType.TOOL_REQUESTED,
            {
                "call_id": call_id,
                "tool_name": f"tool_{call_id}",
                "display_name": f"工具 {call_id}",
                "request": {"file": "model.inp"},
            },
            turn_id=turn_id,
        ),
        events.make(
            EventType.TOOL_STARTED,
            {"call_id": call_id},
            turn_id=turn_id,
        ),
    ]
    if terminal is EventType.TOOL_RESULT:
        payload = {
            "call_id": call_id,
            "result": {"ok": True},
            "duration_ms": duration_ms,
        }
    elif terminal is EventType.TOOL_WARNING:
        payload = {
            "call_id": call_id,
            "warning": "需要检查单位",
            "result": {"ok": True},
            "duration_ms": duration_ms,
        }
    else:
        payload = {
            "call_id": call_id,
            "error": "工具失败",
            "diagnostic": "输入不完整",
            "duration_ms": duration_ms,
        }
    result.append(
        events.make(
            terminal,
            payload,
            turn_id=turn_id,
        )
    )
    return result


def test_event_contract_module_is_ui_and_agent_independent():
    module_path = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "fem_gui"
        / "agent_events.py"
    )
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    imported_roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(
                alias.name.split(".", 1)[0]
                for alias in node.names
            )
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported_roots.add(node.module.split(".", 1)[0])

    assert "PySide6" not in imported_roots
    assert "fem_agent" not in imported_roots
    assert imported_roots <= {
        "__future__",
        "copy",
        "dataclasses",
        "datetime",
        "enum",
        "json",
        "math",
        "re",
        "types",
        "typing",
    }


def test_event_schema_round_trip_and_unknown_values_are_rejected():
    event = _turn_start(_Events())
    restored = AgentEvent.from_dict(event.to_dict())

    assert restored == event
    assert restored.schema_version == AGENT_EVENT_SCHEMA_VERSION
    assert restored.event_type is EventType.TURN_STARTED

    unknown_schema = event.to_dict()
    unknown_schema["schema_version"] = "2.0"
    with pytest.raises(AgentEventError, match="schema_version"):
        AgentEvent.from_dict(unknown_schema)

    unknown_type = event.to_dict()
    unknown_type["event_type"] = "terminal_text"
    with pytest.raises(AgentEventError, match="event_type"):
        AgentEvent.from_dict(unknown_type)

    unknown_payload = event.to_dict()
    unknown_payload["payload"]["raw_cli_text"] = "unstructured"
    with pytest.raises(AgentEventError, match="未知 payload"):
        AgentEvent.from_dict(unknown_payload)


def test_event_payload_is_immutable_and_to_dict_is_detached():
    event = AgentEvent.create(
        event_id="event-immutable",
        session_id="session-1",
        turn_id="turn-1",
        sequence=1,
        event_type=EventType.TOOL_REQUESTED,
        payload={
            "call_id": "call-1",
            "tool_name": "inspect",
            "display_name": "检查",
            "request": {
                "files": ["frame.inp"],
                "options": {"summary": True},
            },
        },
        timestamp="2026-07-29T08:00:00Z",
    )

    with pytest.raises(TypeError):
        event.payload["request"]["options"]["summary"] = False
    with pytest.raises(AttributeError):
        event.payload["request"]["files"].append("other.inp")

    exported = event.to_dict()
    exported["payload"]["request"]["options"]["summary"] = False
    assert event.payload["request"]["options"]["summary"] is True


def test_projector_rejects_duplicate_inverse_gap_and_cross_boundaries():
    events = _Events()
    projector = AgentEventProjector()
    start = _turn_start(events)
    projector.apply(start)

    with pytest.raises(AgentEventError, match="重复 event_id"):
        projector.apply(start)

    inverse = events.make(
        EventType.MESSAGE_START,
        {
            "message_id": "message-inverse",
            "role": "assistant",
            "format": "restricted_markdown",
        },
        sequence=1,
        event_id="event-inverse",
    )
    with pytest.raises(AgentEventError, match="sequence"):
        projector.apply(inverse)

    gap = events.make(
        EventType.MESSAGE_START,
        {
            "message_id": "message-gap",
            "role": "assistant",
            "format": "restricted_markdown",
        },
        sequence=3,
        event_id="event-gap",
    )
    with pytest.raises(AgentEventError, match="sequence"):
        projector.apply(gap)

    cross_session = events.make(
        EventType.MESSAGE_START,
        {
            "message_id": "message-cross-session",
            "role": "assistant",
            "format": "restricted_markdown",
        },
        session_id="session-2",
    )
    with pytest.raises(AgentEventError, match="session"):
        projector.apply(cross_session)

    cross_turn = events.make(
        EventType.MESSAGE_START,
        {
            "message_id": "message-cross-turn",
            "role": "assistant",
            "format": "restricted_markdown",
        },
        turn_id="turn-2",
        sequence=2,
    )
    with pytest.raises(AgentEventError, match="turn"):
        projector.apply(cross_turn)


def test_message_deltas_join_and_single_chunk_uses_the_same_interface():
    events = _Events()
    log = [_turn_start(events)]
    log.extend(_message_events(events, deltas=("模型", "检查", "完成")))
    log.append(events.make(EventType.TURN_COMPLETE, {}))

    state = AgentEventProjector.replay(log).presentation
    message = state.turns[0].messages[0]

    assert message.text == "模型检查完成"
    assert message.status is MessageStatus.COMPLETED
    assert state.turns[0].status is TurnStatus.COMPLETED

    one_chunk = _Events(session_id="session-whole")
    whole_log = [_turn_start(one_chunk)]
    whole_log.extend(_message_events(one_chunk, deltas=("整段回复",)))
    whole_log.append(one_chunk.make(EventType.TURN_COMPLETE, {}))
    whole_message = AgentEventProjector.replay(
        whole_log
    ).presentation.turns[0].messages[0]
    assert whole_message.text == "整段回复"


def test_consecutive_tools_group_and_message_breaks_the_next_group():
    events = _Events()
    log = [_turn_start(events)]
    log.extend(_tool_events(events, "a", duration_ms=200))
    log.extend(
        _tool_events(
            events,
            "b",
            terminal=EventType.TOOL_WARNING,
            duration_ms=300,
        )
    )
    log.extend(_message_events(events, deltas=("中间回复",)))
    log.extend(
        _tool_events(
            events,
            "c",
            terminal=EventType.TOOL_FAILED,
            duration_ms=400,
        )
    )
    log.append(events.make(EventType.TURN_COMPLETE, {}))

    turn = AgentEventProjector.replay(log).presentation.turns[0]

    assert [len(group.calls) for group in turn.tool_groups] == [2, 1]
    first, second = turn.tool_groups
    assert first.completed_count == 1
    assert first.warning_count == 1
    assert first.failed_count == 0
    assert first.total_duration_ms == 500
    assert second.failed_count == 1
    assert second.total_duration_ms == 400


def test_diagnostic_between_tool_events_breaks_the_next_tool_group():
    events = _Events()
    log = [
        _turn_start(events),
        events.make(
            EventType.TOOL_REQUESTED,
            {
                "call_id": "a",
                "tool_name": "tool_a",
                "display_name": "工具 A",
                "request": {},
            },
        ),
        events.make(EventType.TOOL_STARTED, {"call_id": "a"}),
        events.make(
            EventType.DIAGNOSTIC,
            {
                "diagnostic_id": "between-tools",
                "title": "中间诊断",
                "message": "需要继续检查",
                "severity": "warning",
            },
        ),
        events.make(
            EventType.TOOL_RESULT,
            {"call_id": "a", "result": {}, "duration_ms": 100},
        ),
    ]
    log.extend(_tool_events(events, "b"))
    log.append(events.make(EventType.TURN_COMPLETE, {}))

    turn = AgentEventProjector.replay(log).presentation.turns[0]

    assert [len(group.calls) for group in turn.tool_groups] == [1, 1]
    assert [item.kind.value for item in turn.timeline] == [
        "tool_group",
        "diagnostic",
        "tool_group",
    ]


def test_each_turn_keeps_its_own_collapsed_tool_groups():
    events = _Events()
    log = [_turn_start(events)]
    log.extend(_tool_events(events, "a"))
    log.append(events.make(EventType.TURN_COMPLETE, {}))
    log.append(_turn_start(events, turn_id="turn-2"))
    log.extend(_tool_events(events, "b", turn_id="turn-2"))
    log.append(
        events.make(
            EventType.TURN_COMPLETE,
            {},
            turn_id="turn-2",
        )
    )

    state = AgentEventProjector.replay(log).presentation

    assert len(state.turns) == 2
    assert [len(turn.tool_groups) for turn in state.turns] == [1, 1]
    assert state.turns[0].tool_groups[0].group_id.startswith("turn-1:")
    assert state.turns[1].tool_groups[0].group_id.startswith("turn-2:")


def test_tool_summaries_redact_secrets_paths_and_unknown_objects():
    events = _Events()
    log = [_turn_start(events)]
    log.extend(
        (
            events.make(
                EventType.TOOL_REQUESTED,
                {
                    "call_id": "secret-call",
                    "tool_name": "inspect_file",
                    "display_name": "检查文件",
                    "request": {
                        "api_key": "sk-abcdefghijklmnop",
                        "path": r"C:\Users\person\private\model.inp",
                        "nested": {"token": "Bearer abcdefghijklmnop"},
                    },
                },
            ),
            events.make(
                EventType.TOOL_STARTED,
                {"call_id": "secret-call"},
            ),
            events.make(
                EventType.TOOL_RESULT,
                {
                    "call_id": "secret-call",
                    "result": {
                        "output": "/home/person/private/result.odb",
                    },
                    "duration_ms": 10,
                },
            ),
            events.make(EventType.TURN_COMPLETE, {}),
        )
    )

    call = (
        AgentEventProjector.replay(log)
        .presentation.turns[0]
        .tool_groups[0]
        .calls[0]
    )
    rendered = call.request_summary + call.result_summary

    assert "sk-abcdefghijklmnop" not in rendered
    assert "Bearer abcdefghijklmnop" not in rendered
    assert "C:" not in rendered
    assert "/home/person" not in rendered
    assert "敏感信息已隐藏" in rendered
    assert "绝对路径已隐藏" in rendered

    class _Unsafe:
        def __repr__(self) -> str:
            raise AssertionError("不应调用任意对象 repr")

    assert safe_tool_summary(_Unsafe()) == "<不支持的值>"

    spaced_paths = safe_tool_summary(
        {
            "windows": r"C:\Users\Jane Doe\private model.inp",
            "posix": "/home/Jane Doe/private model.inp",
            "url": "https://example.test/v1/models",
            "note": "password=two word secret",
        }
    )
    assert "Jane Doe" not in spaced_paths
    assert "two word secret" not in spaced_paths
    assert "https://example.test/v1/models" in spaced_paths
    file_uri = safe_tool_summary(
        "preview=file:///C:/Users/Jane Doe/private model.inp"
    )
    assert "Jane Doe" not in file_uri
    assert "绝对路径已隐藏" in file_uri


def test_confirmation_requires_exact_revision_hash_and_text_cannot_authorize():
    events = _Events()
    log = [_turn_start(events)]
    log.extend(_message_events(events, deltas=("确认执行",)))
    revision_hash = "a" * 64
    log.append(
        events.make(
            EventType.CONFIRMATION_REQUESTED,
            {
                "confirmation_id": "confirmation-1",
                "title": "需要确认",
                "summary": "确认分析输入",
                "revision": 7,
                "revision_hash": revision_hash,
            },
        )
    )
    log.append(events.make(EventType.TURN_COMPLETE, {}))

    turn = AgentEventProjector.replay(log).presentation.turns[0]

    assert len(turn.confirmations) == 1
    confirmation = turn.confirmations[0]
    assert confirmation.revision == 7
    assert confirmation.revision_hash == revision_hash
    assert not confirmation.authorized

    invalid = _Events()
    _turn_start(invalid)
    with pytest.raises(AgentEventError, match="revision_hash"):
        invalid.make(
            EventType.CONFIRMATION_REQUESTED,
            {
                "confirmation_id": "bad-confirmation",
                "title": "需要确认",
                "summary": "摘要",
                "revision": 7,
                "revision_hash": "short",
            },
        )


def test_cancel_and_failure_close_unfinished_messages_and_tools():
    events = _Events()
    log = [
        _turn_start(events),
        events.make(
            EventType.MESSAGE_START,
            {
                "message_id": "message-cancel",
                "role": "assistant",
                "format": "restricted_markdown",
            },
        ),
        events.make(
            EventType.MESSAGE_DELTA,
            {"message_id": "message-cancel", "delta": "部分回复"},
        ),
        events.make(
            EventType.TOOL_REQUESTED,
            {
                "call_id": "running-call",
                "tool_name": "running_tool",
                "display_name": "运行中工具",
                "request": {},
            },
        ),
        events.make(
            EventType.TOOL_STARTED,
            {"call_id": "running-call"},
        ),
        events.make(
            EventType.TURN_CANCELLED,
            {"reason": "用户停止"},
        ),
        _turn_start(events, turn_id="turn-2"),
        events.make(
            EventType.MESSAGE_START,
            {
                "message_id": "message-failed",
                "role": "assistant",
                "format": "restricted_markdown",
            },
            turn_id="turn-2",
        ),
        events.make(
            EventType.MESSAGE_DELTA,
            {"message_id": "message-failed", "delta": "未完成"},
            turn_id="turn-2",
        ),
        events.make(
            EventType.TURN_FAILED,
            {"reason": "Fake 流失败"},
            turn_id="turn-2",
        ),
    ]

    state = AgentEventProjector.replay(log).presentation
    cancelled, failed = state.turns

    assert cancelled.status is TurnStatus.CANCELLED
    assert cancelled.messages[0].status is MessageStatus.CANCELLED
    assert (
        cancelled.tool_groups[0].calls[0].status
        is ToolStatus.CANCELLED
    )
    assert failed.status is TurnStatus.FAILED
    assert failed.messages[0].status is MessageStatus.INTERRUPTED


def test_turn_complete_rejects_unfinished_stream():
    events = _Events()
    projector = AgentEventProjector()
    projector.apply(_turn_start(events))
    projector.apply(
        events.make(
            EventType.MESSAGE_START,
            {
                "message_id": "message-running",
                "role": "assistant",
                "format": "restricted_markdown",
            },
        )
    )

    with pytest.raises(AgentEventError, match="未完成消息"):
        projector.apply(events.make(EventType.TURN_COMPLETE, {}))


def test_event_log_replay_restore_produces_an_equivalent_snapshot():
    stream = FakeAgentEventStream()
    source = AgentEventProjector.replay(stream.review_preview())
    restored = AgentEventProjector.restore_event_log(
        source.export_event_log()
    )

    assert restored.presentation == source.presentation
    assert (
        restored.presentation.to_snapshot()
        == source.presentation.to_snapshot()
    )


def test_diagnostic_severity_is_structured_and_separate_from_tools():
    events = _Events()
    log = [_turn_start(events)]
    log.extend(_tool_events(events, "a"))
    log.append(
        events.make(
            EventType.DIAGNOSTIC,
            {
                "diagnostic_id": "diagnostic-1",
                "title": "阻塞",
                "message": "缺少确认",
                "severity": "blocking",
                "code": "CONFIRMATION-REQUIRED",
            },
        )
    )
    log.append(events.make(EventType.TURN_COMPLETE, {}))

    turn = AgentEventProjector.replay(log).presentation.turns[0]

    assert turn.diagnostics[0].severity is DiagnosticSeverity.BLOCKING
    assert len(turn.tool_groups[0].calls[0].diagnostics) == 0
    assert turn.timeline[-1].item_id == "diagnostic-1"


def test_fake_event_stream_drives_tool_message_diagnostic_and_confirmation_ui():
    application = _application()
    drawer = AgentChatDrawer()
    drawer.replay_agent_events(FakeAgentEventStream().review_preview())
    drawer.resize(440, 760)
    drawer.show()
    application.processEvents()

    state = drawer.event_presentation
    assert state.session_id == "phase3-preview"
    assert "模型预检查" in state.turns[0].messages[0].text
    assert len(state.turns[0].tool_groups[0].calls) == 3

    tools = drawer.findChild(ToolActivityPreview)
    assert tools is not None
    assert "3 个工具" in tools.summary_button.text()
    assert "3 项完成" in tools.summary_button.text()

    confirmation_button = drawer.findChild(
        QToolButton,
        "agentChatConfirmationButton",
    )
    assert confirmation_button is not None
    confirmation = state.turns[0].confirmations[0]
    assert confirmation_button.property("revision") == confirmation.revision
    assert (
        confirmation_button.property("revisionHash")
        == confirmation.revision_hash
    )
    assert (
        drawer.findChild(QLabel, "agentChatConfirmationRevision")
        is None
    )
    assert not confirmation_button.property("authorized")
    assert not confirmation_button.isEnabled()
    drawer.close()


def test_failed_turn_keeps_diagnostic_without_duplicate_status_box():
    application = _application()
    events = _Events(session_id="failed-turn-ui")
    drawer = AgentChatDrawer()
    drawer.setStyleSheet(_AGENT_CHAT_STYLESHEET)
    drawer.replay_agent_events(
        (
            _turn_start(events),
            events.make(
                EventType.DIAGNOSTIC,
                {
                    "diagnostic_id": "config-error",
                    "title": "Agent 配置不可用",
                    "message": "FEM Agent 配置无效或缺少凭据。",
                    "severity": "error",
                    "code": "GUI-AGENT-CONFIG",
                },
            ),
            events.make(
                EventType.TURN_FAILED,
                {"reason": "FEM Agent 配置无效或缺少凭据。"},
            ),
        )
    )
    drawer.show()
    application.processEvents()

    assert drawer.findChild(QWidget, "agentChatDiagnostic") is not None
    assert drawer.findChild(QLabel, "agentChatTurnStatus") is None
    title = drawer.findChild(QLabel, "agentChatDiagnosticTitle")
    code = drawer.findChild(QLabel, "agentChatDiagnosticCode")
    assert title is not None
    assert code is not None
    assert title.property("severity") == "error"
    assert title.palette().color(title.foregroundRole()).name() == "#b42318"
    assert code.text() == "GUI-AGENT-CONFIG"
    assert code.font().pointSizeF() == 7.5
    drawer.close()


def test_incremental_ui_escapes_raw_html_and_shows_stream_status():
    application = _application()
    events = _Events(session_id="ui-session")
    start = _turn_start(events)
    message_start = events.make(
        EventType.MESSAGE_START,
        {
            "message_id": "ui-message",
            "role": "assistant",
            "format": "restricted_markdown",
        },
    )
    drawer = AgentChatDrawer()
    drawer.replay_agent_events((start, message_start))
    drawer.show()
    drawer.apply_agent_event(
        events.make(
            EventType.MESSAGE_DELTA,
            {
                "message_id": "ui-message",
                "delta": "<img src='https://example.test/a.png'>"
                "<script>alert(1)</script> **安全文本**",
            },
        )
    )
    application.processEvents()

    labels = drawer.findChildren(QLabel, "agentChatAgentMessage")
    assert len(labels) == 1
    label = labels[0]
    assert label.property("messageStatus") == "streaming"
    assert "&lt;img" in label.text()
    assert "<img src=" not in label.text()
    assert "<b>安全文本</b>" in label.text()
    assert not label.openExternalLinks()
    drawer.close()


def test_restricted_markdown_renders_ordered_and_unordered_lists():
    application = _application()
    events = _Events(session_id="markdown-list-session")
    drawer = AgentChatDrawer()
    drawer.replay_agent_events(
        (
            _turn_start(events),
            events.make(
                EventType.MESSAGE_START,
                {
                    "message_id": "markdown-list-message",
                    "role": "assistant",
                    "format": "restricted_markdown",
                },
            ),
        )
    )
    drawer.show()
    drawer.apply_agent_event(
        events.make(
            EventType.MESSAGE_DELTA,
            {
                "message_id": "markdown-list-message",
                "delta": (
                    "支持的操作：\n"
                    "- **模型检查**：读取模型\n"
                    "* `求解`：等待确认\n\n"
                    "执行顺序：\n"
                    "1. 建立几何\n"
                    "2. 生成网格\n"
                    "<img src='https://example.test/unsafe.png'>"
                ),
            },
        )
    )
    application.processEvents()

    label = drawer.findChild(QLabel, "agentChatAgentMessage")
    assert label is not None
    rendered = label.text()
    assert "<ul " in rendered
    assert "<li><b>模型检查</b>：读取模型</li>" in rendered
    assert "<li><span style='font-family:monospace'>求解</span>" in rendered
    assert "<ol " in rendered
    assert "<li>建立几何</li>" in rendered
    assert "<li>生成网格</li>" in rendered
    assert "&lt;img src=&#x27;https://example.test/unsafe.png&#x27;&gt;" in (
        rendered
    )
    assert "<img src=" not in rendered
    assert not label.openExternalLinks()
    drawer.close()


def test_restricted_markdown_renders_safe_aligned_tables():
    application = _application()
    events = _Events(session_id="markdown-table-session")
    drawer = AgentChatDrawer()
    drawer.replay_agent_events(
        (
            _turn_start(events),
            events.make(
                EventType.MESSAGE_START,
                {
                    "message_id": "markdown-table-message",
                    "role": "assistant",
                    "format": "restricted_markdown",
                },
            ),
        )
    )
    drawer.show()
    drawer.apply_agent_event(
        events.make(
            EventType.MESSAGE_DELTA,
            {
                "message_id": "markdown-table-message",
                "delta": (
                    "| 参数 | 默认值 | 说明 |\n"
                    "| :--- | ---: | :---: |\n"
                    "| **厚度** | 1 mm | `plane stress` |\n"
                    "| 载荷 | 10 MPa | <script>unsafe</script> |\n"
                    "| 转义 | A \\| B | `x|y` |"
                ),
            },
        )
    )
    application.processEvents()

    label = drawer.findChild(QLabel, "agentChatAgentMessage")
    assert label is not None
    rendered = label.text()
    assert "<table " in rendered
    assert rendered.count("<th ") == 3
    assert rendered.count("<td ") == 9
    assert "text-align:left" in rendered
    assert "text-align:right" in rendered
    assert "text-align:center" in rendered
    assert "<b>厚度</b>" in rendered
    assert "font-family:monospace" in rendered
    assert "A | B" in rendered
    assert "&lt;script&gt;unsafe&lt;/script&gt;" in rendered
    assert "<script>" not in rendered
    drawer.close()


def test_conversation_follows_stream_until_user_scrolls_up():
    application = _application()
    events = _Events(session_id="scroll-follow-session")
    start = _turn_start(events)
    message_start = events.make(
        EventType.MESSAGE_START,
        {
            "message_id": "scrolling-message",
            "role": "assistant",
            "format": "restricted_markdown",
        },
    )
    first_delta = events.make(
        EventType.MESSAGE_DELTA,
        {
            "message_id": "scrolling-message",
            "delta": "\n".join(f"初始内容 {index}" for index in range(40)),
        },
    )
    drawer = AgentChatDrawer()
    drawer.setStyleSheet(_AGENT_CHAT_STYLESHEET)
    drawer.resize(420, 320)
    drawer.replay_agent_events((start, message_start, first_delta))
    drawer.show()
    application.processEvents()
    QTest.qWait(10)

    scroll_bar = drawer.conversation_scroll.verticalScrollBar()
    assert scroll_bar.maximum() > 0
    assert scroll_bar.width() <= 10
    assert scroll_bar.value() == scroll_bar.maximum()

    previous_value = scroll_bar.maximum() // 3
    scroll_bar.setValue(previous_value)
    drawer.apply_agent_event(
        events.make(
            EventType.MESSAGE_DELTA,
            {
                "message_id": "scrolling-message",
                "delta": "\n用户上滑后新增的流式内容",
            },
        )
    )
    application.processEvents()
    QTest.qWait(10)

    assert scroll_bar.value() == previous_value
    assert scroll_bar.value() < scroll_bar.maximum()

    scroll_bar.setValue(scroll_bar.maximum())
    drawer.apply_agent_event(
        events.make(
            EventType.MESSAGE_DELTA,
            {
                "message_id": "scrolling-message",
                "delta": "\n恢复跟随后的新增内容",
            },
        )
    )
    application.processEvents()
    QTest.qWait(10)

    assert scroll_bar.value() == scroll_bar.maximum()
    drawer.close()


def test_incremental_render_preserves_expanded_tool_group():
    application = _application()
    events = _Events(session_id="expanded-session")
    log = [_turn_start(events)]
    log.extend(_tool_events(events, "visible-tool"))
    log.append(
        events.make(
            EventType.MESSAGE_START,
            {
                "message_id": "streaming-message",
                "role": "assistant",
                "format": "restricted_markdown",
            },
        )
    )
    drawer = AgentChatDrawer()
    drawer.setStyleSheet(_AGENT_CHAT_STYLESHEET)
    drawer.replay_agent_events(log)
    drawer.show()
    application.processEvents()

    tools = drawer.findChild(ToolActivityPreview)
    assert (
        tools.palette().color(tools.backgroundRole()).name()
        == "#f2f4f6"
    )
    tools.summary_button.click()
    assert tools.details.isVisible()

    drawer.apply_agent_event(
        events.make(
            EventType.MESSAGE_DELTA,
            {
                "message_id": "streaming-message",
                "delta": "继续输出",
            },
        )
    )
    application.processEvents()

    refreshed_tools = drawer.findChild(ToolActivityPreview)
    assert refreshed_tools.summary_button.isChecked()
    assert refreshed_tools.details.isVisible()
    drawer.close()


def test_structured_titles_and_tool_details_are_forced_plain_text():
    application = _application()
    events = _Events(session_id="plain-text-session")
    unsafe_markup = "<img src='file:///private/model.png'>"
    log = [
        _turn_start(events),
        events.make(
            EventType.TOOL_REQUESTED,
            {
                "call_id": "markup-tool",
                "tool_name": "inspect",
                "display_name": unsafe_markup,
                "request": {"markup": "<b>request</b>"},
            },
        ),
        events.make(
            EventType.TOOL_STARTED,
            {"call_id": "markup-tool"},
        ),
        events.make(
            EventType.TOOL_RESULT,
            {
                "call_id": "markup-tool",
                "result": {"markup": "<script>result</script>"},
                "duration_ms": 1,
            },
        ),
        events.make(
            EventType.DIAGNOSTIC,
            {
                "diagnostic_id": "markup-diagnostic",
                "title": unsafe_markup,
                "message": "<b>diagnostic</b>",
                "severity": "warning",
            },
        ),
        events.make(
            EventType.CONFIRMATION_REQUESTED,
            {
                "confirmation_id": "markup-confirmation",
                "title": unsafe_markup,
                "summary": "<b>confirmation</b>",
                "revision": 1,
                "revision_hash": "a" * 64,
            },
        ),
        events.make(EventType.TURN_COMPLETE, {}),
    ]
    drawer = AgentChatDrawer()
    drawer.replay_agent_events(log)
    drawer.show()
    application.processEvents()

    labels = [
        label
        for label in drawer.findChildren(QLabel)
        if "<" in label.text()
    ]
    assert labels
    assert all(
        label.textFormat() == Qt.TextFormat.PlainText
        for label in labels
    )
    drawer.close()
