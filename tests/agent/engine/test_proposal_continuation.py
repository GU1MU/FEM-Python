from fem_agent.engine import AgentSessionEngine, EngineEventType
from fem_agent.providers.base import ToolCall
from fem_agent.providers.fake import FakeProvider
from fem_agent.schemas import ToolResult

from tests.helpers.agent_provider_fixtures import (
    tool_response,
    text_response,
    StreamingFakeProvider,
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
        [text_response("等待本地确认"), text_response("继续下一阶段")]
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


def test_succeeded_proposal_suppresses_unpublished_tool_call(tmp_path):
    provider = FakeProvider(
        [
            text_response("等待本地确认"),
            tool_response(
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
        [text_response("等待本地确认"), text_response("请修正后重试")]
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


def test_post_terminal_streamed_summary_replays_once_after_guards(tmp_path):
    provider = StreamingFakeProvider(
        [text_response("等待本地确认"), text_response("下一阶段开始")]
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
