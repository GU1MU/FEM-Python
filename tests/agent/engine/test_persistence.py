import threading

import pytest

from fem_agent.diagnostics import DiagnosticCode
from fem_agent.engine import AgentSessionEngine, EngineConfig, EngineEventType
from fem_agent.providers.base import ToolCall
from fem_agent.providers.fake import FakeProvider

from tests.helpers.agent_engine_fixtures import _attached_engine
from tests.helpers.agent_provider_fixtures import tool_response, text_response


pytestmark = pytest.mark.integration


def test_request_context_is_ephemeral_across_provider_tool_loop(tmp_path):
    provider = FakeProvider(
        [
            tool_response(
                ToolCall(
                    "call_capabilities",
                    "show_capabilities",
                    {"detail": "summary"},
                )
            ),
            text_response("能力检查完成。"),
        ]
    )
    engine = AgentSessionEngine(
        tmp_path / "workspace",
        provider,
        session_id="ses_ephemeral_context",
    )
    request_context = (
        "The following JSON is user-selected workspace data for this turn.\n"
        '{"files":[{"path":"notes.md","content":"PRIVATE-CONTEXT"}]}'
    )

    events = engine.send_message(
        "结合 @notes.md 检查能力。",
        request_context=request_context,
    )

    assert len(provider.requests) == 2
    for request in provider.requests:
        assert sum(
            message.role == "user"
            and message.content == request_context
            for message in request.messages
        ) == 1
        context_index = next(
            index
            for index, message in enumerate(request.messages)
            if message.content == request_context
        )
        assert request.messages[context_index + 1].content == (
            "结合 @notes.md 检查能力。"
        )
    tool_started = next(
        event
        for event in events
        if event.event == EngineEventType.TOOL_STARTED
    )
    assert tool_started.data["arguments"] == {"detail": "summary"}
    conversation = (
        engine.workspace
        / "sessions"
        / engine.session_id
        / "conversation.json"
    ).read_text(encoding="utf-8")
    assert "PRIVATE-CONTEXT" not in conversation
    assert "@notes.md" in conversation


def test_credential_in_request_context_is_rejected_before_provider(tmp_path):
    provider = FakeProvider([text_response("不应调用。")])
    engine = AgentSessionEngine(
        tmp_path / "workspace",
        provider,
        session_id="ses_context_credential",
    )

    events = engine.send_message(
        "读取文件。",
        request_context=(
            "workspace data\n"
            '{"content":"DEEPSEEK_API_KEY=sk-abcdefghijklmnop"}'
        ),
    )

    assert provider.requests == []
    assert any(
        event.event == EngineEventType.DIAGNOSTIC
        and event.data["diagnostic"]["code"]
        == DiagnosticCode.INVALID_INPUT.value
        for event in events
    )
    conversation = (
        engine.workspace
        / "sessions"
        / engine.session_id
        / "conversation.json"
    )
    assert not conversation.exists()


def test_engine_conversation_can_be_reopened_without_provider_objects(tmp_path):
    provider = FakeProvider([text_response("已记录。")])
    engine, _ = _attached_engine(tmp_path, provider)
    engine.send_message("保留这条会话记录。")
    session_id = engine.session_id

    reopened = AgentSessionEngine(
        engine.workspace,
        FakeProvider([text_response("继续。")]),
        session_id=session_id,
    )
    events = reopened.send_message("继续。")

    assert any(
        event.event == EngineEventType.MESSAGE_DELTA
        and event.data["text"] == "继续。"
        for event in events
    )


def test_conversation_storage_is_byte_bounded_and_reopenable(tmp_path):
    provider = FakeProvider(
        [text_response("答" * 500) for _ in range(10)]
    )
    config = EngineConfig(
        max_provider_message_chars=2_000,
        max_user_message_chars=2_000,
        max_conversation_storage_bytes=4_096,
    )
    engine = AgentSessionEngine(
        tmp_path / "workspace",
        provider,
        session_id="ses_bounded_conversation",
        config=config,
    )

    for index in range(10):
        engine.send_message(f"{index}:" + "问" * 500)

    conversation = (
        engine.workspace
        / "sessions"
        / engine.session_id
        / "conversation.json"
    )
    assert conversation.stat().st_size <= 4_096

    reopened = AgentSessionEngine(
        engine.workspace,
        FakeProvider([text_response("继续")]),
        session_id=engine.session_id,
        config=config,
    )
    assert reopened.send_message("继续")


def test_conversation_window_keeps_complete_tool_result_for_provider(tmp_path):
    observed = {}

    def inspect_tool_result(messages, tools):
        observed["roles"] = [message.role for message in messages]
        observed["tool_payload"] = next(
            message.content
            for message in messages
            if message.role == "tool"
        )
        return text_response("已读取工具结果。")

    provider = FakeProvider(
        [
            tool_response(
                ToolCall(
                    "small_window_capabilities",
                    "show_capabilities",
                    {},
                )
            ),
            inspect_tool_result,
        ]
    )
    engine = AgentSessionEngine(
        tmp_path / "workspace",
        provider,
        session_id="ses_small_conversation_window",
        config=EngineConfig(
            max_cloud_turns=2,
            max_tool_calls=1,
            max_conversation_messages=3,
        ),
    )

    events = engine.send_message("查看能力。")

    assert observed["roles"][-3:] == ["user", "assistant", "tool"]
    assert '"ok":true' in observed["tool_payload"]
    assert any(
        event.event == EngineEventType.MESSAGE_DELTA
        and event.data["text"] == "已读取工具结果。"
        for event in events
    )


def test_conversation_window_rejects_incomplete_maximum_tool_turn():
    with pytest.raises(ValueError, match="complete tool turn"):
        EngineConfig(
            max_cloud_turns=2,
            max_tool_calls=1,
            max_conversation_messages=2,
        )


def test_tool_audit_is_byte_bounded_and_remains_appendable(tmp_path):
    responses = []
    for index in range(20):
        responses.extend(
            (
                tool_response(
                    ToolCall(
                        f"audit_{index}",
                        "show_capabilities",
                        {},
                    )
                ),
                text_response("能力已列出。"),
            )
        )
    engine = AgentSessionEngine(
        tmp_path / "workspace",
        FakeProvider(responses),
        session_id="ses_bounded_audit",
        config=EngineConfig(max_tool_audit_storage_bytes=2_048),
    )

    for index in range(20):
        engine.send_message(f"第 {index} 次列出能力。")

    audit = (
        engine.workspace
        / "sessions"
        / engine.session_id
        / "tool-audit.json"
    )
    assert audit.stat().st_size <= 2_048
    reopened = AgentSessionEngine(
        engine.workspace,
        FakeProvider(
            [
                tool_response(
                    ToolCall("audit_final", "show_capabilities", {})
                ),
                text_response("完成。"),
            ]
        ),
        session_id=engine.session_id,
        config=EngineConfig(max_tool_audit_storage_bytes=2_048),
    )
    assert reopened.send_message("再列出一次。")
    assert audit.stat().st_size <= 2_048


def test_unstorable_provider_turn_returns_resource_diagnostic(tmp_path):
    engine = AgentSessionEngine(
        tmp_path / "workspace",
        FakeProvider([text_response("答" * 500)]),
        session_id="ses_oversized_turn",
        config=EngineConfig(
            max_provider_message_chars=1_000,
            max_conversation_storage_bytes=1_024,
        ),
    )

    events = engine.send_message("一个短问题。")

    assert any(
        event.event == EngineEventType.DIAGNOSTIC
        and event.data["diagnostic"]["code"] == "RESOURCE_LIMIT"
        for event in events
    )
    reopened = AgentSessionEngine(
        engine.workspace,
        FakeProvider([text_response("可继续。")]),
        session_id=engine.session_id,
        config=EngineConfig(
            max_provider_message_chars=1_000,
            max_conversation_storage_bytes=1_024,
        ),
    )
    assert reopened.send_message("继续。")


def test_session_switch_is_rejected_while_provider_operation_is_active(
    monkeypatch,
    tmp_path,
):
    provider = FakeProvider()
    engine = AgentSessionEngine(
        tmp_path / "workspace",
        provider,
        session_id="ses_concurrent_session",
    )
    entered = threading.Event()
    release = threading.Event()

    def complete(*args, **kwargs):
        entered.set()
        assert release.wait(2.0)
        return text_response("done")

    monkeypatch.setattr(provider, "complete", complete)
    original_session = engine.session_id
    thread = threading.Thread(target=lambda: engine.send_message("hello"))

    thread.start()
    assert entered.wait(2.0)
    rejected = engine.create_session()
    release.set()
    thread.join(2.0)

    assert not thread.is_alive()
    assert engine.session_id == original_session
    assert any(
        event.event == EngineEventType.DIAGNOSTIC
        and event.data["diagnostic"]["code"]
        == DiagnosticCode.OPERATION_IN_PROGRESS.value
        for event in rejected
    )
