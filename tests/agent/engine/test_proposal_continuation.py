import pytest

from fem_agent.engine import AgentSessionEngine, EngineEventType
from fem_agent.providers.base import AssistantMessage, ProviderResponse, ToolCall
from fem_agent.providers.fake import FakeProvider
from fem_agent.schemas import ToolResult

from tests.helpers.agent_engine_providers import (
    _tool_response,
    _text_response,
    _StreamingFakeProvider,
    _ReasoningStreamingFakeProvider,
)
from tests.helpers.agent_engine_registry_fixtures import (
    _GeometryEditToolRegistry,
)


pytestmark = pytest.mark.integration


def test_unbacked_proposal_execution_claim_is_not_exposed(tmp_path):
    provider = FakeProvider(
        [
            _text_response("方案已确认，等待本地操作执行完成。"),
            _text_response("当前没有可执行的本地提案。"),
        ]
    )
    engine = AgentSessionEngine(
        tmp_path / "workspace",
        provider,
        session_id="ses_unbacked_proposal_claim",
        dynamic_tools=_GeometryEditToolRegistry(),
    )

    events = engine.send_message("当前状态是什么？")

    assert len(provider.requests) == 2
    assert "proposal-grounding correction" in (
        provider.requests[1].messages[-1].content or ""
    )
    assert not any(
        "等待本地操作执行完成" in str(event.data.get("text", ""))
        for event in events
        if event.event is EngineEventType.MESSAGE_DELTA
    )
    assert any(
        event.event is EngineEventType.MESSAGE_DELTA
        and event.data["text"] == "当前没有可执行的本地提案。"
        for event in events
    )


def _register_test_continuation(
    engine,
    *,
    revision=4,
    proposal_kind="",
):
    engine._register_continuation_from_result(
        ToolResult(
            ok=True,
            session_id=engine.session_id,
            input_revision=0,
            idempotency_key="checkpoint-result",
            summary="proposal waiting",
            data={
                "continuation_checkpoint": {
                    "session_id": engine.session_id,
                    "source_turn_id": "source-turn-1",
                    "proposal_id": "proposal-continue-1",
                    "proposal_hash": "a" * 64,
                    "model_revision": revision,
                    "proposal_kind": proposal_kind,
                }
            },
        )
    )


def test_proposal_continuation_uses_system_envelope_and_consumes_once(tmp_path):
    provider = FakeProvider(
        [_text_response("等待本地确认"), _text_response("继续下一阶段")]
    )
    engine = AgentSessionEngine(
        tmp_path / "workspace",
        provider,
        session_id="ses_continuation_success",
    )
    engine.send_message("建立模型")
    _register_test_continuation(engine)

    events = engine.continue_after_proposal(
        "proposal-continue-1",
        "a" * 64,
        "source-turn-1",
        4,
        "succeeded",
        "几何创建完成",
    )

    assert any(
        event.event is EngineEventType.MESSAGE_DELTA
        and event.data["text"] == "继续下一阶段"
        for event in events
    )
    continuation_request = provider.requests[1]
    assert continuation_request.messages[-1].role == "system"
    assert "proposal_terminal" in continuation_request.messages[-1].content
    assert "Treat this terminal status as authoritative" in (
        continuation_request.messages[-1].content
    )
    assert "do not add a standalone completion acknowledgement" in (
        continuation_request.messages[-1].content
    )
    assert sum(
        message.role == "user" for message in continuation_request.messages
    ) == 1
    assert continuation_request.tools
    assert engine.continue_after_proposal(
        "proposal-continue-1",
        "a" * 64,
        "source-turn-1",
        4,
        "succeeded",
    ) == ()
    assert len(provider.requests) == 2

    _register_test_continuation(engine, revision=9)
    assert engine.continue_after_proposal(
        "proposal-continue-1",
        "a" * 64,
        "source-turn-1",
        9,
        "cancelled",
    ) == ()
    assert len(provider.requests) == 2


def test_succeeded_proposal_suppresses_bare_completion_echo(tmp_path):
    provider = FakeProvider(
        [_text_response("等待本地确认"), _text_response("已完成")]
    )
    engine = AgentSessionEngine(
        tmp_path / "workspace",
        provider,
        session_id="ses_redundant_completion",
    )
    engine.send_message("建立模型")
    _register_test_continuation(engine, proposal_kind="geometry")

    events = engine.continue_after_proposal(
        "proposal-continue-1",
        "a" * 64,
        "source-turn-1",
        4,
        "succeeded",
        "已完成",
    )

    assert not any(
        event.event
        in {EngineEventType.MESSAGE_STARTED, EngineEventType.MESSAGE_DELTA}
        for event in events
    )
    assert "已完成" not in tuple(
        message.content
        for message in engine._history
        if message.role == "assistant" and message.content
    )
    assert len(provider.requests) == 2


def test_succeeded_proposal_suppresses_reconfirmation_and_refusal(tmp_path):
    contradiction = (
        "当前无法创建几何，请你在本地 UI 中再次点击并确认这个提案。"
    )
    provider = FakeProvider(
        [_text_response("等待本地确认"), _text_response(contradiction)]
    )
    engine = AgentSessionEngine(
        tmp_path / "workspace",
        provider,
        session_id="ses_continuation_guard",
    )
    engine.send_message("建立模型")
    _register_test_continuation(engine)

    events = engine.continue_after_proposal(
        "proposal-continue-1",
        "a" * 64,
        "source-turn-1",
        4,
        "succeeded",
        "几何创建完成",
    )

    deltas = [
        event.data["text"]
        for event in events
        if event.event is EngineEventType.MESSAGE_DELTA
    ]
    assert deltas == ["几何创建完成"]
    assert engine._history[-1] == AssistantMessage(
        "assistant",
        "几何创建完成",
    )
    assert contradiction not in tuple(
        message.content for message in engine._history if message.content
    )


def test_succeeded_proposal_suppresses_unpublished_tool_call(tmp_path):
    provider = FakeProvider(
        [
            _text_response("等待本地确认"),
            _tool_response(
                ToolCall("missing-tool", "draft_native_geometry", {})
            ),
        ]
    )
    engine = AgentSessionEngine(
        tmp_path / "workspace",
        provider,
        session_id="ses_continuation_tool_guard",
    )
    engine.send_message("建立模型")
    _register_test_continuation(engine)

    events = engine.continue_after_proposal(
        "proposal-continue-1",
        "a" * 64,
        "source-turn-1",
        4,
        "succeeded",
        "几何创建完成",
    )

    assert not any(
        event.event is EngineEventType.TOOL_STARTED for event in events
    )
    assert [
        event.data["text"]
        for event in events
        if event.event is EngineEventType.MESSAGE_DELTA
    ] == ["几何创建完成"]


def test_failed_or_revision_changed_continuation_cannot_advance_tools(tmp_path):
    provider = FakeProvider(
        [_text_response("等待本地确认"), _text_response("请修正后重试")]
    )
    engine = AgentSessionEngine(
        tmp_path / "workspace",
        provider,
        session_id="ses_continuation_failure",
    )
    engine.send_message("建立模型")
    _register_test_continuation(engine)

    engine.continue_after_proposal(
        "proposal-continue-1",
        "a" * 64,
        "source-turn-1",
        4,
        "failed",
        "本地任务失败",
    )

    assert provider.requests[1].tools == ()
    _register_test_continuation(engine, revision=7)
    assert engine.continue_after_proposal(
        "proposal-continue-1",
        "a" * 64,
        "source-turn-1",
        8,
        "succeeded",
    ) == ()
    assert len(provider.requests) == 2


def test_additional_model_request_retries_bare_delete_completion(tmp_path):
    provider = FakeProvider(
        [
            _text_response("等待本地确认"),
            _text_response("已完成"),
            _text_response("新增模型尚未创建，当前缺少所需工具。"),
        ]
    )
    engine = AgentSessionEngine(
        tmp_path / "workspace",
        provider,
        session_id="ses_additional_model_continuation",
    )
    engine.send_message("另建立一个2D平板")
    _register_test_continuation(
        engine,
        proposal_kind="destructive_edit",
    )

    events = engine.continue_after_proposal(
        "proposal-continue-1",
        "a" * 64,
        "source-turn-1",
        4,
        "succeeded",
        "部件已删除",
    )

    assert len(provider.requests) == 3
    correction = provider.requests[2].messages[-1]
    assert correction.role == "system"
    assert "create_native_model_document" in (correction.content or "")
    assert [
        event.data["text"]
        for event in events
        if event.event is EngineEventType.MESSAGE_DELTA
    ] == ["新增模型尚未创建，当前缺少所需工具。"]
    assert "已完成" not in tuple(
        message.content for message in engine._history if message.content
    )


def test_post_terminal_streamed_contradiction_is_held_back_from_gui(tmp_path):
    contradiction = "网格操作卡已生成（待确认），内容如下：全局尺寸 5"
    provider = _StreamingFakeProvider(
        [_text_response("等待本地确认"), _text_response(contradiction)]
    )
    streamed = []
    engine = AgentSessionEngine(
        tmp_path / "workspace",
        provider,
        session_id="ses_post_terminal_stream_guard",
        event_sink=streamed.append,
    )
    engine.send_message("建立模型")
    _register_test_continuation(engine, proposal_kind="mesh")
    first_round_events = len(streamed)

    events = engine.continue_after_proposal(
        "proposal-continue-1",
        "a" * 64,
        "source-turn-1",
        4,
        "succeeded",
        "网格意图和生成模型已原子提交",
    )

    continuation_deltas = [
        event.data["text"]
        for event in streamed[first_round_events:]
        if event.event is EngineEventType.MESSAGE_DELTA
    ]
    assert continuation_deltas == ["网格意图和生成模型已原子提交"]
    assert not any(
        "待确认" in event.data["text"]
        for event in events
        if event.event is EngineEventType.MESSAGE_DELTA
    )
    assert contradiction not in tuple(
        message.content for message in engine._history if message.content
    )


def test_post_terminal_streamed_summary_replays_once_after_guards(tmp_path):
    provider = _StreamingFakeProvider(
        [_text_response("等待本地确认"), _text_response("下一阶段开始")]
    )
    streamed = []
    engine = AgentSessionEngine(
        tmp_path / "workspace",
        provider,
        session_id="ses_post_terminal_stream_replay",
        event_sink=streamed.append,
    )
    engine.send_message("建立模型")
    _register_test_continuation(engine)
    first_round_events = len(streamed)

    engine.continue_after_proposal(
        "proposal-continue-1",
        "a" * 64,
        "source-turn-1",
        4,
        "succeeded",
        "几何创建完成",
    )

    first_round_deltas = [
        event.data["text"]
        for event in streamed[:first_round_events]
        if event.event is EngineEventType.MESSAGE_DELTA
    ]
    assert len(first_round_deltas) == 2
    assert "".join(first_round_deltas) == "等待本地确认"
    continuation_deltas = [
        event.data["text"]
        for event in streamed[first_round_events:]
        if event.event is EngineEventType.MESSAGE_DELTA
    ]
    assert continuation_deltas == ["下一阶段开始"]


def test_post_terminal_streamed_reasoning_is_also_held_back(tmp_path):
    contradiction = "网格操作卡已生成（待确认），内容如下：全局尺寸 5"
    provider = _ReasoningStreamingFakeProvider(
        [
            _text_response("等待本地确认"),
            ProviderResponse(
                AssistantMessage(
                    "assistant",
                    content=contradiction,
                    reasoning_content="The card is pending user confirmation.",
                ),
                finish_reason="stop",
            ),
        ]
    )
    streamed = []
    engine = AgentSessionEngine(
        tmp_path / "workspace",
        provider,
        session_id="ses_post_terminal_reasoning_guard",
        event_sink=streamed.append,
    )
    engine.send_message("建立模型")
    _register_test_continuation(engine, proposal_kind="mesh")
    first_round_events = len(streamed)

    events = engine.continue_after_proposal(
        "proposal-continue-1",
        "a" * 64,
        "source-turn-1",
        4,
        "succeeded",
        "网格意图和生成模型已原子提交",
    )

    continuation_deltas = [
        event.data["text"]
        for event in streamed[first_round_events:]
        if event.event is EngineEventType.MESSAGE_DELTA
    ]
    assert continuation_deltas == ["网格意图和生成模型已原子提交"]
    assert not any(
        event.data["text"].startswith(("The card", "网格操作卡"))
        for event in events
        if event.event is EngineEventType.MESSAGE_DELTA
    )
