import json

import pytest

from fem_agent.engine import (
    AgentSessionEngine,
    EngineEventType,
    _geometry_edit_preview,
    _point_chain,
)
from fem_agent.providers.base import (
    AssistantMessage,
    ProviderResponse,
    ToolCall,
    ToolDefinition,
)
from fem_agent.providers.fake import FakeProvider
from fem_agent.schemas import ToolResult

from tests.helpers.agent_engine_providers import (
    _tool_response,
    _text_response,
    _StreamingFakeProvider,
    _ReasoningStreamingFakeProvider,
)
from tests.helpers.agent_engine_registry_fixtures import (
    _AdditionalModelToolRegistry,
    _RetryingGeometryEditWithCatalogToolRegistry,
)


pytestmark = pytest.mark.integration


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

    @property
    def provider_snapshot(self):
        return None

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
    provider = _StreamingFakeProvider([_text_response("正在检查当前模型")])
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
    provider = _StreamingFakeProvider(
        [_text_response("Please provide the target value.")]
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


def test_reasoning_stream_is_process_and_formal_content_stays_separate(tmp_path):
    reasoning = "I should inspect the current state and verify the requested values."
    formal = "The requested values are valid."
    provider = _ReasoningStreamingFakeProvider(
        [
            ProviderResponse(
                AssistantMessage(
                    "assistant",
                    content=formal,
                    reasoning_content=reasoning,
                ),
                finish_reason="stop",
            )
        ]
    )
    engine = AgentSessionEngine(
        tmp_path / "workspace",
        provider,
        session_id="ses_streaming_reasoning_presentation",
    )

    events = engine.send_message("Inspect the current values.")

    process_start = next(
        event
        for event in events
        if event.event is EngineEventType.MESSAGE_STARTED
        and event.data.get("presentation_kind") == "process"
    )
    assert process_start.data["presentation_kind"] == "process"
    assert any(
        event.event is EngineEventType.MESSAGE_PRESENTATION
        and event.data.get("presentation_kind") == "result_summary"
        for event in events
    )
    deltas = [
        event.data["text"]
        for event in events
        if event.event is EngineEventType.MESSAGE_DELTA
    ]
    assert "".join(deltas) == reasoning + formal


def test_buffered_reasoning_is_presented_before_formal_content(tmp_path):
    reasoning = "先读取当前状态，再核对参数。"
    formal = "参数已经核对完成。"
    provider = FakeProvider(
        [
            ProviderResponse(
                AssistantMessage(
                    "assistant",
                    content=formal,
                    reasoning_content=reasoning,
                ),
                finish_reason="stop",
            )
        ]
    )
    engine = AgentSessionEngine(
        tmp_path / "workspace",
        provider,
        session_id="ses_buffered_reasoning_presentation",
    )

    events = engine.send_message("检查当前参数。")

    displayed = [
        (
            event.data.get("presentation_kind"),
            events[index + 1].data.get("text"),
        )
        for index, event in enumerate(events[:-1])
        if event.event is EngineEventType.MESSAGE_STARTED
        and events[index + 1].event is EngineEventType.MESSAGE_DELTA
        and events[index + 1].data.get("text") in {reasoning, formal}
    ]
    assert displayed == [
        ("process", reasoning),
        ("result_summary", formal),
    ]

    continuation_provider = FakeProvider([_text_response("继续。")])
    reopened = AgentSessionEngine(
        engine.workspace,
        continuation_provider,
        session_id=engine.session_id,
    )
    reopened.send_message("继续。")
    replayed = next(
        message
        for message in continuation_provider.requests[0].messages
        if message.role == "assistant" and message.content == formal
    )
    assert replayed.reasoning_content == reasoning


def test_chinese_turn_retries_and_hides_an_english_only_response(tmp_path):
    provider = FakeProvider(
        [
            _text_response("I will inspect the current model."),
            _text_response("我会检查当前模型。"),
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
    correction = provider.requests[1].messages[-1]
    assert correction.role == "system"
    assert "Simplified Chinese" in (correction.content or "")
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
            _text_response("已读取当前能力。"),
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


def test_provider_prompt_contains_restrained_engineering_response_contract(
    tmp_path,
):
    provider = FakeProvider([_text_response("结论。")])
    engine = AgentSessionEngine(
        tmp_path / "workspace",
        provider,
        session_id="ses_response_contract",
    )

    engine.send_message("请简洁回答。")

    system_prompt = provider.requests[0].messages[0].content
    contract_text = system_prompt.split(
        "<response_contract>\n",
        1,
    )[1].split("\n</response_contract>", 1)[0]
    contract = json.loads(contract_text)
    assert contract["language"] == "match_user"
    assert contract["tone"] == [
        "academic",
        "concise",
        "restrained",
        "rational",
        "engineering-focused",
    ]
    assert contract["implementation_details"] == (
        "only_when_explicitly_requested_or_required_by_material_diagnostic"
    )
    assert contract["abaqus_comparison"] == (
        "only_when_explicitly_requested_and_reference_evidence_is_available"
    )
    assert contract["generic_disclaimers"] == "omit"
    assert system_prompt.startswith(
        "You are FEM Agent, an in-application assistant"
    )
    assert "FEM Agent V0" not in system_prompt
    assert "local deterministic fem package" not in system_prompt.casefold()
    assert "do not write guessed or inferred values" not in system_prompt


def test_planar_path_preview_renders_json_coordinate_pairs():
    assert _point_chain([[140, 35], [160, 35], [160, 65]]) == (
        "(140, 35) → (160, 35) → (160, 65)"
    )


def test_validated_plan_preview_precedes_card_and_is_not_repeated_after_accept(
    tmp_path,
):
    arguments = {
        "part_function": "2D平板",
        "construction": {
            "schema_version": 1,
            "name": "2D平板",
            "plane": "XY",
            "nodes": [
                {
                    "id": "plate",
                    "kind": "rectangle",
                    "x": 0,
                    "y": 0,
                    "width": 300,
                    "height": 100,
                }
            ],
            "result_node_id": "plate",
        },
        "output": "planar",
    }
    provider = FakeProvider(
        [
            ProviderResponse(
                AssistantMessage(
                    "assistant",
                    content="让我先调用工具，稍后再说明方案。",
                    tool_calls=(
                        ToolCall(
                            "prepare-preview",
                            "prepare_planar_construction_proposal",
                            arguments,
                        ),
                    ),
                ),
                finish_reason="tool_calls",
            ),
        ]
    )
    tools = _AdditionalModelToolRegistry()
    engine = AgentSessionEngine(
        tmp_path / "workspace",
        provider,
        session_id="ses_plan_preview_order",
        dynamic_tools=tools,
    )

    events = engine.send_message("创建一个300×100的2D平板")

    preview_index = next(
        index
        for index, event in enumerate(events)
        if event.event is EngineEventType.MESSAGE_DELTA
        and "方案预览" in event.data["text"]
    )
    proposal_result_index = next(
        index
        for index, event in enumerate(events)
        if event.event is EngineEventType.TOOL_COMPLETED
        and event.data["tool"] == "prepare_planar_construction_proposal"
    )
    stage_index = next(
        index
        for index, event in enumerate(events)
        if event.event is EngineEventType.MESSAGE_DELTA
        and "正在构造二维轮廓" in event.data["text"]
    )
    proposal_start_index = next(
        index
        for index, event in enumerate(events)
        if event.event is EngineEventType.TOOL_STARTED
        and event.data["tool"] == "prepare_planar_construction_proposal"
    )
    assert stage_index < proposal_start_index < preview_index
    assert preview_index < proposal_result_index
    preview = events[preview_index].data["text"]
    assert "矩形轮廓：左下角 (0, 0)，尺寸 300 × 100" in preview
    assert "形成 1 个材料区域和 0 个切除区域" in preview
    assert "plate：" not in preview
    preview_start = next(
        event
        for event in events[:preview_index]
        if event.event is EngineEventType.MESSAGE_STARTED
        and event.data.get("presentation_kind") == "proposal_preview"
    )
    assert preview_start.data["presentation_kind"] == "proposal_preview"
    assert not any(
        event.event is EngineEventType.MESSAGE_DELTA
        and "稍后再说明方案" in event.data["text"]
        for event in events
    )

    provider.queue(
        _text_response(
            "已生成设计方案：2D 平板几何提案，方案已就绪。"
        )
    )
    continuation = engine.continue_after_proposal(
        "proposal-new-model-geometry",
        "b" * 64,
        "source-turn-model",
        0,
        "succeeded",
        "已完成",
    )

    assert not any(
        event.event in {
            EngineEventType.MESSAGE_STARTED,
            EngineEventType.MESSAGE_DELTA,
        }
        for event in continuation
    )


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
            _text_response("材料已创建。"),
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
    assert "创建材料“Steel”" in preview.data["text"]
    assert "弹性模量 210000" in preview.data["text"]
    assert "泊松比 0.3" in preview.data["text"]
    assert "apply_model_definition" not in preview.data["text"]
    assert "create_material" not in preview.data["text"]
    assert "{" not in preview.data["text"]
    assert not any(
        event.event is EngineEventType.MESSAGE_DELTA
        and "apply_model_definition" in event.data["text"]
        for event in events
    )
    assert tools.calls == [("apply_model_definition", arguments)]


def test_batch_geometry_edit_preview_uses_natural_language():
    preview = "\n".join(
        _geometry_edit_preview(
            {
                "part_id": "P1",
                "spatial_relation": {
                    "reference_feature_id": "PB1",
                    "relation": "above",
                    "clearance": 10,
                },
                "edit": {
                    "operation": "batch",
                    "edits": [
                        {
                            "operation": "add_rectangle",
                            "x": 85,
                            "y": 30,
                            "width": 30,
                            "height": 12,
                        },
                        {
                            "operation": "add_rectangle",
                            "x": 88,
                            "y": 49,
                            "width": 24,
                            "height": 10,
                        },
                        {
                            "operation": "add_rectangle",
                            "x": 85,
                            "y": 68,
                            "width": 30,
                            "height": 12,
                        },
                    ],
                },
            },
            "增加三个矩形轮廓",
            True,
        )
    )

    assert "一次完成 3 项草图修改" in preview
    assert "位于特征 PB1 上方，要求净间距 10（提交时由本地校验）" in preview
    assert "增加矩形轮廓，左下角 (85, 30)，尺寸 30 × 12" in preview
    assert "P1" not in preview
    assert "add_rectangle" not in preview
    assert '"operation"' not in preview
    assert "{" not in preview


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
            _tool_response(
                ToolCall(
                    "read-edit",
                    "read_geometry_edit_context",
                    {"part_id": "P1"},
                )
            ),
            _tool_response(
                ToolCall("prepare-invalid", "prepare_geometry_edit", edit)
            ),
            _text_response(self_correction),
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
