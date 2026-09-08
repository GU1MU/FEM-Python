import json

import pytest

from fem_agent.engine import AgentSessionEngine, EngineEventType
from fem_agent.providers.base import (
    AssistantMessage,
    ProviderResponse,
    ToolCall,
    ToolDefinition,
)
from fem_agent.providers.fake import FakeProvider
from fem_agent.schemas import ToolResult

from tests.helpers.agent_engine_registry_fixtures import (
    _AdditionalModelToolRegistry,
    _RetryingGeometryEditWithCatalogToolRegistry,
)
from tests.helpers.agent_provider_fixtures import (
    tool_response,
    text_response,
    StreamingFakeProvider,
    ReasoningStreamingFakeProvider,
)


class _PatchToolRegistry:
    def __init__(self):
        self.definitions = (
            ToolDefinition(
                "apply_model_definition",
                "apply_model_definition",
                {
                    "type": "object",
                    "properties": {
                        "action": {"type": "string"},
                        "parameters": {"type": "object"},
                    },
                    "required": ["action", "parameters"],
                    "additionalProperties": False,
                },
            ),
        )
        self.calls = []

    provider_snapshot = None

    def refresh_turn_snapshot(self, published_tool_names=()):
        del published_tool_names
        return None

    def dispatch(self, name, arguments, context):
        self.calls.append((name, dict(arguments)))
        return ToolResult(
            ok=True,
            session_id=context.session_id,
            input_revision=context.expected_revision,
            idempotency_key=context.idempotency_key,
            summary="材料已创建",
            data={"state": "succeeded", "undo_available": True},
        )


def test_engine_forwards_provider_text_deltas_without_rebuffering(tmp_path):
    provider = StreamingFakeProvider([text_response("正在检查当前模型")])
    streamed = []
    engine = AgentSessionEngine(
        tmp_path / "workspace",
        provider,
        session_id="ses_streaming_text",
        event_sink=streamed.append,
    )

    returned = engine.send_message("检查模型")

    sink_deltas = [
        event.data["text"]
        for event in streamed
        if event.event is EngineEventType.MESSAGE_DELTA
    ]
    returned_deltas = [
        event.data["text"]
        for event in returned
        if event.event is EngineEventType.MESSAGE_DELTA
    ]
    assert sink_deltas == ["正在检查", "当前模型"]
    assert returned_deltas == sink_deltas
    assert "".join(sink_deltas) == "正在检查当前模型"


def test_streamed_formal_response_finalizes_semantic_presentation(tmp_path):
    provider = StreamingFakeProvider(
        [text_response("Please provide the target value.")]
    )
    engine = AgentSessionEngine(
        tmp_path / "workspace",
        provider,
        session_id="ses_streaming_presentation",
    )

    events = engine.send_message("Help me choose a target value.")

    presentation = next(
        event
        for event in events
        if event.event is EngineEventType.MESSAGE_PRESENTATION
    )
    assert presentation.data["presentation_kind"] == "decision_request"


@pytest.mark.parametrize(
    "provider_type, formal_event",
    [
        (FakeProvider, EngineEventType.MESSAGE_STARTED),
        (ReasoningStreamingFakeProvider, EngineEventType.MESSAGE_PRESENTATION),
    ],
)
def test_reasoning_precedes_formal_content_and_survives_reopen(
    tmp_path, provider_type, formal_event
):
    reasoning, formal = "Checking the current values.", "The requested values are valid."
    provider = provider_type(
        [
            ProviderResponse(
                AssistantMessage("assistant", content=formal, reasoning_content=reasoning),
                finish_reason="stop",
            )
        ]
    )
    engine = AgentSessionEngine(tmp_path / "workspace", provider)

    events = engine.send_message("Inspect the current values.")

    starts = [
        event for event in events if event.event is EngineEventType.MESSAGE_STARTED
    ]
    assert starts[0].data["presentation_kind"] == "process"
    assert "".join(
        event.data["text"]
        for event in events
        if event.event is EngineEventType.MESSAGE_DELTA
    ) == reasoning + formal
    assert any(
        event.event is formal_event
        and event.data["presentation_kind"] == "result_summary"
        for event in events
    )

    replay = FakeProvider([text_response("Continue.")])
    reopened = AgentSessionEngine(engine.workspace, replay, session_id=engine.session_id)
    reopened.send_message("Continue.")
    restored = next(
        message
        for message in replay.requests[0].messages
        if message.role == "assistant" and message.content == formal
    )
    assert restored.reasoning_content == reasoning


def test_chinese_turn_retries_and_hides_an_english_only_response(tmp_path):
    provider = FakeProvider(
        [
            text_response("I will inspect the current model."),
            text_response("我会检查当前模型。"),
        ]
    )
    engine = AgentSessionEngine(
        tmp_path / "workspace",
        provider,
        session_id="ses_chinese_language_retry",
    )

    events = engine.send_message("请检查当前模型。")

    deltas = [
        event.data["text"]
        for event in events
        if event.event is EngineEventType.MESSAGE_DELTA
    ]
    assert deltas == ["我会检查当前模型。"]
    assert len(provider.requests) == 2
    state = json.loads(
        (provider.requests[0].messages[1].content or "").split(": ", 1)[1]
    )
    assert state["required_response_language"] == "zh-CN"


def test_chinese_turn_drops_english_tool_narration_but_keeps_the_tool_call(tmp_path):
    provider = FakeProvider(
        [
            ProviderResponse(
                AssistantMessage(
                    "assistant",
                    content="I will inspect the available capabilities.",
                    tool_calls=(ToolCall("call_language_tool", "show_capabilities", {}),),
                ),
                finish_reason="tool_calls",
            ),
            text_response("已读取当前能力。"),
        ]
    )
    engine = AgentSessionEngine(
        tmp_path / "workspace",
        provider,
        session_id="ses_chinese_tool_language",
    )

    events = engine.send_message("查看当前能力。")

    deltas = [
        event.data["text"]
        for event in events
        if event.event is EngineEventType.MESSAGE_DELTA
    ]
    assert deltas == ["正在读取当前可用能力。", "已读取当前能力。"]
    assistant_tool_message = next(
        message for message in engine._history if message.tool_calls
    )
    assert assistant_tool_message.content is None
    assert assistant_tool_message.tool_calls[0].name == "show_capabilities"


def test_proposal_preview_precedes_the_proposal_result(tmp_path):
    arguments = {
        "part_function": "2D平板",
        "construction": {
            "schema_version": 1,
            "name": "平板",
            "plane": "XY",
            "nodes": [
                {
                    "id": "plate", "kind": "rectangle", "x": 0, "y": 0,
                    "width": 300, "height": 100,
                }
            ],
            "result_node_id": "plate",
        },
        "output": "planar",
    }
    tool = "prepare_planar_construction_proposal"
    provider = FakeProvider([tool_response(ToolCall("proposal", tool, arguments))])
    engine = AgentSessionEngine(
        tmp_path / "workspace", provider, dynamic_tools=_AdditionalModelToolRegistry()
    )

    events = engine.send_message("创建一个300×100的2D平板")

    preview = next(
        i for i, event in enumerate(events)
        if event.event is EngineEventType.MESSAGE_STARTED
        and event.data.get("presentation_kind") == "proposal_preview"
    )
    completed = next(
        i for i, event in enumerate(events)
        if event.event is EngineEventType.TOOL_COMPLETED and event.data["tool"] == tool
    )
    assert preview < completed
    assert events[preview + 1].event is EngineEventType.MESSAGE_DELTA
    assert events[preview + 1].data["text"]
    assert events[completed].data["result"]["data"]["state"] == "pending_confirmation"


def test_automatic_model_patch_has_natural_preview_before_tool_execution(
    tmp_path,
):
    arguments = {
        "action": "create_material",
        "parameters": {
            "name": "Steel",
            "properties": {"E": 210000, "nu": 0.3},
        },
    }
    provider = FakeProvider(
        [
            ProviderResponse(
                AssistantMessage(
                    "assistant",
                    content=(
                        'apply_model_definition {"action":"create_material"}'
                    ),
                    tool_calls=(
                        ToolCall(
                            "apply-material",
                            "apply_model_definition",
                            arguments,
                        ),
                    ),
                ),
                finish_reason="tool_calls",
            ),
            text_response("材料已创建。"),
        ]
    )
    tools = _PatchToolRegistry()
    engine = AgentSessionEngine(
        tmp_path / "workspace",
        provider,
        session_id="ses_patch_preview",
        dynamic_tools=tools,
    )

    events = engine.send_message("创建一个钢材料，弹性模量210000，泊松比0.3")

    tool_index = next(
        index
        for index, event in enumerate(events)
        if event.event is EngineEventType.TOOL_STARTED
        and event.data["tool"] == "apply_model_definition"
    )
    preview_start_index = next(
        index
        for index, event in enumerate(events[:tool_index])
        if event.event is EngineEventType.MESSAGE_STARTED
        and event.data.get("presentation_kind") == "patch_preview"
    )
    preview = events[preview_start_index + 1]

    assert preview.event is EngineEventType.MESSAGE_DELTA
    assert not any(
        event.event is EngineEventType.MESSAGE_DELTA
        and "apply_model_definition" in event.data["text"]
        for event in events
    )
    assert tools.calls == [("apply_model_definition", arguments)]


def test_failed_tool_self_correction_is_presented_as_process(tmp_path):
    edit = {
        "part_id": "P1",
        "edit": {
            "operation": "add_polygon",
            "vertices": [
                {"x": 20, "y": 20},
                {"x": 40, "y": 20},
                {"x": 40, "y": 40},
                {"x": 20, "y": 40},
            ],
        },
    }
    self_correction = (
        "此前的轮廓推导包含多段坐标映射。局部校验失败后，"
        "这些中间推演不能作为已完成结果，需要在后续请求中继续处理。"
    )
    provider = FakeProvider(
        [
            tool_response(
                ToolCall(
                    "read-edit",
                    "read_geometry_edit_context",
                    {"part_id": "P1"},
                )
            ),
            tool_response(
                ToolCall("prepare-invalid", "prepare_geometry_edit", edit)
            ),
            text_response(self_correction),
        ]
    )
    tools = _RetryingGeometryEditWithCatalogToolRegistry()
    engine = AgentSessionEngine(
        tmp_path / "workspace",
        provider,
        session_id="ses_failed_tool_process_presentation",
        dynamic_tools=tools,
    )

    events = engine.send_message("在现有平板上增加一个槽")

    correction_delta_index = next(
        index
        for index, event in enumerate(events)
        if event.event is EngineEventType.MESSAGE_DELTA
        and event.data.get("text") == self_correction
    )
    correction_start = events[correction_delta_index - 1]
    assert correction_start.event is EngineEventType.MESSAGE_STARTED
    assert correction_start.data["presentation_kind"] == "process"
