import json
import threading
from dataclasses import replace

import pytest

from fem_agent.artifacts import ArtifactStore
from fem_agent.authoring_runtime import AuthoringWorkflowController
from fem_agent.diagnostics import DiagnosticCode
from fem_agent.engine import (
    AgentSessionEngine,
    EngineConfig,
    EngineEventType,
    _geometry_edit_preview,
    _missing_requested_geometry_features,
    _point_chain,
)
from fem_agent.providers.base import (
    AssistantMessage,
    ProviderConfig,
    ProviderResponse,
    ToolCall,
    ToolDefinition,
)
from fem_agent.providers.deepseek import DeepSeekProvider
from fem_agent.providers.fake import FakeProvider
from fem_agent.schemas import RunStatus, SessionPhase, ToolResult
from fem_agent.tools.registry import ToolExecutionContext
from fem_agent.worker import (
    InspectionWorkerError,
    WorkerResponse,
    WorkerResponseIntegrityError,
)
from tests.helpers.abaqus_builders import write_perforated_plate_style_inp


pytestmark = pytest.mark.integration


def _tool_response(*calls):
    return ProviderResponse(
        AssistantMessage("assistant", tool_calls=tuple(calls)),
        finish_reason="tool_calls",
    )


def _text_response(text):
    return ProviderResponse(
        AssistantMessage("assistant", content=text),
        finish_reason="stop",
    )


class _AdditionalModelToolRegistry:
    def __init__(self):
        no_arguments = {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        }
        self.definitions = tuple(
            ToolDefinition(name, name, no_arguments)
            for name in (
                "create_native_model_document",
                "read_authoring_context",
                "prepare_planar_construction_proposal",
            )
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
        data = {}
        if name == "create_native_model_document":
            data = {
                "state": "succeeded",
                "next_action": (
                    "read_authoring_context_then_prepare_requested_geometry"
                ),
            }
        elif name == "prepare_planar_construction_proposal":
            data = {
                "state": "pending_confirmation",
                "proposal_view": {
                    "proposal_id": "proposal-new-model-geometry",
                    "proposal_hash": "b" * 64,
                    "proposal_kind": "geometry",
                    "title": "加入部件",
                    "summary": (
                        "设计提案：2D 平面构造（节点=1，材料区=1，孔洞=0）；"
                        "单位制 mm-N-MPa（默认）"
                    ),
                    "impact": "确认后创建该二维几何并刷新 GUI",
                    "confirm_label": "加入部件",
                    "target_document_id": "1",
                    "target_session_id": "native-session",
                    "base_session_revision": 0,
                },
                "proof_summary": {
                    "material_profile_count": 1,
                    "hole_count": 0,
                    "component_count": 1,
                },
                "continuation_checkpoint": {
                    "session_id": context.session_id,
                    "source_turn_id": "source-turn-model",
                    "proposal_id": "proposal-new-model-geometry",
                    "proposal_hash": "b" * 64,
                    "model_revision": 0,
                    "proposal_kind": "geometry",
                },
            }
        return ToolResult(
            ok=True,
            session_id=context.session_id,
            input_revision=context.expected_revision,
            idempotency_key=context.idempotency_key,
            summary=f"{name} completed",
            data=data,
        )


class _StageProposalToolRegistry:
    def __init__(self):
        no_arguments = {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        }
        self.definitions = tuple(
            ToolDefinition(name, name, no_arguments)
            for name in (
                "set_authoring_requirements",
                "prepare_mesh_proposal",
            )
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
        data = {}
        if name == "set_authoring_requirements":
            data = {
                "requirement_stage": "mesh",
                "recorded": sorted(dict(arguments.get("requirements", {}))),
                "missing_requirements": [],
                "operation_confirmation_required": True,
                "next_action": "prepare_stage_proposal",
            }
        elif name == "prepare_mesh_proposal":
            data = {
                "state": "pending_confirmation",
                "proposal_view": {
                    "proposal_id": "proposal-mesh-stage",
                    "proposal_hash": "c" * 64,
                    "proposal_kind": "mesh",
                    "title": "生成网格",
                    "summary": "网格方案：二次三角形，全局 10 mm，孔边局部加密",
                    "impact": "确认后调用 Gmsh 生成网格并刷新 GUI",
                    "confirm_label": "生成网格",
                    "target_document_id": "1",
                    "target_session_id": "native-session",
                    "base_session_revision": 0,
                },
                "continuation_checkpoint": {
                    "session_id": context.session_id,
                    "source_turn_id": "source-turn-mesh",
                    "proposal_id": "proposal-mesh-stage",
                    "proposal_hash": "c" * 64,
                    "model_revision": 0,
                    "proposal_kind": "mesh",
                },
            }
        return ToolResult(
            ok=True,
            session_id=context.session_id,
            input_revision=context.expected_revision,
            idempotency_key=context.idempotency_key,
            summary=f"{name} completed",
            data=data,
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


class _GeometryEditToolRegistry:
    def __init__(self):
        no_arguments = {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        }
        self.definitions = tuple(
            ToolDefinition(name, name, no_arguments)
            for name in (
                "read_geometry_edit_context",
                "prepare_geometry_edit",
            )
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
        data = {}
        if name == "prepare_geometry_edit":
            data = {
                "state": "pending_confirmation",
                "proposal_view": {
                    "summary": "修改当前二维草图",
                    "impact": "确认后更新该部件并刷新 GUI",
                },
                "continuation_checkpoint": {
                    "session_id": context.session_id,
                    "source_turn_id": "source-turn-edit",
                    "proposal_id": "proposal-planar-edit",
                    "proposal_hash": "c" * 64,
                    "model_revision": context.expected_revision,
                    "proposal_kind": "geometry",
                },
            }
        return ToolResult(
            ok=True,
            session_id=context.session_id,
            input_revision=context.expected_revision,
            idempotency_key=context.idempotency_key,
            summary=f"{name} completed",
            data=data,
        )


class _StaleGeometryEditToolRegistry:
    def __init__(self):
        self.stage = "stale"
        self.calls = []

    @property
    def definitions(self):
        no_arguments = {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        }
        names = (
            ("read_authoring_context",)
            if self.stage == "stale"
            else (
                "read_authoring_context",
                "read_geometry_edit_context",
                "prepare_geometry_edit",
            )
        )
        return tuple(
            ToolDefinition(name, name, no_arguments)
            for name in names
        )

    @property
    def provider_snapshot(self):
        return {
            "available": True,
            "workflow_stage": self.stage,
            "published_tool_names": [item.name for item in self.definitions],
        }

    def refresh_turn_snapshot(self, published_tool_names=()):
        del published_tool_names
        return self.provider_snapshot

    def dispatch(self, name, arguments, context):
        self.calls.append((name, dict(arguments)))
        data = {}
        if name == "read_authoring_context":
            self.stage = "mesh_ready"
        elif name == "prepare_geometry_edit":
            data = {
                "state": "pending_confirmation",
                "proposal_view": {
                    "summary": "修正连续定宽槽",
                    "impact": "确认后更新该部件并刷新 GUI",
                },
                "continuation_checkpoint": {
                    "session_id": context.session_id,
                    "source_turn_id": "source-turn-stale-edit",
                    "proposal_id": "proposal-stale-edit",
                    "proposal_hash": "d" * 64,
                    "model_revision": context.expected_revision,
                    "proposal_kind": "geometry",
                },
            }
        return ToolResult(
            ok=True,
            session_id=context.session_id,
            input_revision=context.expected_revision,
            idempotency_key=context.idempotency_key,
            summary=f"{name} completed",
            data=data,
        )


class _GeometryEditWithCatalogToolRegistry(_GeometryEditToolRegistry):
    def __init__(self):
        super().__init__()
        no_arguments = {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        }
        self.definitions = (
            *self.definitions,
            ToolDefinition(
                "read_geometry_feature_catalog",
                "read_geometry_feature_catalog",
                no_arguments,
            ),
        )


class _RetryingGeometryEditWithCatalogToolRegistry(
    _GeometryEditWithCatalogToolRegistry
):
    def __init__(self):
        super().__init__()
        self.prepare_attempts = 0

    def dispatch(self, name, arguments, context):
        if name == "prepare_geometry_edit":
            self.prepare_attempts += 1
            if self.prepare_attempts == 1:
                self.calls.append((name, dict(arguments)))
                return ToolResult(
                    ok=False,
                    session_id=context.session_id,
                    input_revision=context.expected_revision,
                    idempotency_key=context.idempotency_key,
                    summary="geometry validation needs refreshed feature context",
                    data={"retryable": True},
                )
        return super().dispatch(name, arguments, context)


class _StreamingFakeProvider(FakeProvider):
    def complete_stream(self, messages, tools, on_text_delta):
        response = super().complete(messages, tools)
        content = response.message.content or ""
        split = max(1, len(content) // 2)
        for delta in (content[:split], content[split:]):
            if delta:
                on_text_delta(delta)
        return response


class _ReasoningStreamingFakeProvider(FakeProvider):
    supports_reasoning_stream = True

    def complete_stream(
        self,
        messages,
        tools,
        on_text_delta,
        on_reasoning_delta,
    ):
        response = super().complete(messages, tools)
        reasoning = response.message.reasoning_content or ""
        reasoning_split = max(1, len(reasoning) // 2)
        for delta in (reasoning[:reasoning_split], reasoning[reasoning_split:]):
            if delta:
                on_reasoning_delta(delta)
        content = response.message.content or ""
        content_split = max(1, len(content) // 2)
        for delta in (content[:content_split], content[content_split:]):
            if delta:
                on_text_delta(delta)
        return response


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


def test_authoring_prompt_uses_proposal_first_geometry_and_local_unit_defaults(
    tmp_path,
):
    provider = FakeProvider([_text_response("已准备设计提案。")])
    controller = AuthoringWorkflowController(lambda: {}, {})
    engine = AgentSessionEngine(
        tmp_path / "workspace",
        provider,
        session_id="ses_proposal_first",
        dynamic_tools=controller,
    )

    engine.send_message("建立一个模型")

    system_prompt = provider.requests[0].messages[0].content
    assert "Use a proposal-first policy for native geometry." in system_prompt
    assert "propose a basic planar rectangular Part" in system_prompt
    assert "local defaults length=mm, force=N, and" in system_prompt
    assert "never create a separate unit-selection" in system_prompt
    assert "do not add a natural-language instruction asking" in system_prompt
    assert "material removal is represented by a closed" in system_prompt
    assert "use one non-self-intersecting add_polygon edit as the primary" in (
        system_prompt
    )
    assert "add_polygon edit" in system_prompt
    assert "add_path_slot as the preferred geometry-edit entry" in system_prompt
    assert "planar_boolean(tool.kind=path_stroke) as its lower-level equivalent" in (
        system_prompt
    )
    assert "keep the path-slot representation" in system_prompt
    assert "A malformed centerline does not" in system_prompt
    assert "junction or the intended width varies" in system_prompt
    assert "use every returned diagnostic and affected logical" in system_prompt
    assert "use prepare_planar_construction_proposal as the sole" in system_prompt
    assert "Use one polygon as the default representation" in system_prompt
    assert "path_stroke as the preferred compact representation" in system_prompt
    assert "multiple bends" in system_prompt
    assert "not as the default construction" in system_prompt
    assert "submit an open centerline as a wire" in system_prompt
    assert "one ordered, open, non-branching" in system_prompt
    assert "strictly inside its material target with positive" in system_prompt
    assert "preserve one connected material component" in system_prompt
    assert "S-shaped" not in system_prompt
    assert "U-shaped" not in system_prompt
    assert "H-shaped" not in system_prompt
    assert "Never use a user-visible" in system_prompt
    assert "diagnostic probe" in system_prompt
    assert "call create_native_model_document" in system_prompt
    assert "Never delete the current Part" in system_prompt


def test_blank_geometry_omitted_units_cannot_create_a_unit_question(tmp_path):
    question = (
        "我需要先确认一下你的项目单位制。请告诉我：你希望使用什么单位制？"
    )
    provider = FakeProvider(
        [_text_response(question), _text_response(question)]
    )
    controller = AuthoringWorkflowController(lambda: {}, {})
    engine = AgentSessionEngine(
        tmp_path / "workspace",
        provider,
        session_id="ses_default_native_units",
        dynamic_tools=controller,
    )

    events = engine.send_message("帮我建立一个平板，上面切除出一个S形")

    assert len(provider.requests) == 2
    first_state = provider.requests[0].messages[1].content or ""
    assert '"blank_native_geometry_unit_policy"' in first_state
    assert all(value in first_state for value in ('"mm"', '"N"', '"MPa"'))
    correction = provider.requests[1].messages[-1]
    assert correction.role == "system"
    assert "length=mm, force=N, and stress=MPa" in (correction.content or "")
    deltas = [
        event.data["text"]
        for event in events
        if event.event is EngineEventType.MESSAGE_DELTA
    ]
    assert len(deltas) == 1
    assert "默认单位制 mm-N-MPa" in deltas[0]
    assert "几何建模工具" in deltas[0]
    assert question not in tuple(
        message.content for message in engine._history if message.content
    )


def test_explicit_native_geometry_units_do_not_activate_default_unit_guard(
    tmp_path,
):
    response = "将按用户指定的 m-kN-kPa 单位制继续。"
    provider = FakeProvider([_text_response(response)])
    controller = AuthoringWorkflowController(lambda: {}, {})
    engine = AgentSessionEngine(
        tmp_path / "workspace",
        provider,
        session_id="ses_explicit_native_units",
        dynamic_tools=controller,
    )

    events = engine.send_message("请用 m-kN-kPa 创建一个平板")

    assert len(provider.requests) == 1
    assert any(
        event.event is EngineEventType.MESSAGE_DELTA
        and event.data["text"] == response
        for event in events
    )


def test_new_model_tool_cannot_stop_before_requested_geometry_proposal(tmp_path):
    provider = FakeProvider(
        [
            _tool_response(
                ToolCall("create-model", "create_native_model_document", {})
            ),
            _text_response("新模型已创建。"),
            _tool_response(
                ToolCall("read-new-model", "read_authoring_context", {})
            ),
            _tool_response(
                ToolCall(
                    "prepare-new-geometry",
                    "prepare_planar_construction_proposal",
                    {
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
                                    "width": 10,
                                    "height": 5,
                                }
                            ],
                            "result_node_id": "plate",
                        },
                        "output": "planar",
                    },
                )
            ),
        ]
    )
    tools = _AdditionalModelToolRegistry()
    engine = AgentSessionEngine(
        tmp_path / "workspace",
        provider,
        session_id="ses_new_model_follow_up",
        dynamic_tools=tools,
    )

    events = engine.send_message("另建立一个2D平板")

    assert len(provider.requests) == 4
    correction = provider.requests[2].messages[-1]
    assert correction.role == "system"
    assert "prepare_planar_construction_proposal" in (correction.content or "")
    assert [name for name, _arguments in tools.calls] == [
        "create_native_model_document",
        "read_authoring_context",
        "prepare_planar_construction_proposal",
    ]
    assert "新模型已创建。" not in tuple(
        event.data.get("text")
        for event in events
        if event.event is EngineEventType.MESSAGE_DELTA
    )
    visible_text = tuple(
        str(event.data.get("text", ""))
        for event in events
        if event.event is EngineEventType.MESSAGE_DELTA
    )
    assert any("正在创建新的模型文档" in text for text in visible_text)
    assert any("正在读取当前模型状态和建模约束" in text for text in visible_text)
    assert any("正在构造二维轮廓" in text for text in visible_text)


def test_completed_stage_requirements_cannot_stop_before_the_proposal_card(
    tmp_path,
):
    provider = FakeProvider(
        [
            _tool_response(
                ToolCall(
                    "record-mesh-requirements",
                    "set_authoring_requirements",
                    {
                        "turn_id": "turn-mesh",
                        "requirements": {
                            "mesh_global_size": 10,
                            "mesh_order": 2,
                        },
                    },
                )
            ),
            _text_response(
                "网格方案如下，操作卡片已就绪：二次三角形、全局 10 mm。"
                "确认后即生成网格。"
            ),
            _tool_response(
                ToolCall("prepare-mesh-card", "prepare_mesh_proposal", {})
            ),
        ]
    )
    tools = _StageProposalToolRegistry()
    engine = AgentSessionEngine(
        tmp_path / "workspace",
        provider,
        session_id="ses_stage_proposal_card",
        dynamic_tools=tools,
    )

    events = engine.send_message("划分网格，2次三角形，孔边加密")

    assert len(provider.requests) == 3
    correction = provider.requests[2].messages[-1]
    assert correction.role == "system"
    assert "prepare_mesh_proposal" in (correction.content or "")
    assert [
        message.role
        for message in provider.requests[2].messages
    ] == ["system", "system", "user", "assistant", "tool", "system"]
    assert [name for name, _arguments in tools.calls] == [
        "set_authoring_requirements",
        "prepare_mesh_proposal",
    ]
    assert "操作卡片已就绪" not in tuple(
        event.data.get("text")
        for event in events
        if event.event is EngineEventType.MESSAGE_DELTA
    )
    assert any(
        event.event is EngineEventType.TOOL_COMPLETED
        and event.data["tool"] == "prepare_mesh_proposal"
        for event in events
    )


def test_stage_proposal_correction_retry_limit_recovers_locally(tmp_path):
    provider = FakeProvider(
        [
            _tool_response(
                ToolCall(
                    "record-mesh-requirements",
                    "set_authoring_requirements",
                    {
                        "turn_id": "turn-mesh",
                        "requirements": {"mesh_global_size": 10},
                    },
                )
            ),
            _text_response("网格参数已记录，操作卡片已就绪。"),
            _text_response("网格参数已记录，操作卡片仍然就绪。"),
        ]
    )
    tools = _StageProposalToolRegistry()
    engine = AgentSessionEngine(
        tmp_path / "workspace",
        provider,
        session_id="ses_stage_proposal_retry_limit",
        dynamic_tools=tools,
    )

    events = engine.send_message("划分网格")

    assert len(provider.requests) == 3
    deltas = tuple(
        str(event.data.get("text", ""))
        for event in events
        if event.event is EngineEventType.MESSAGE_DELTA
    )
    assert any("未能生成确认卡片" in text for text in deltas)
    assert not any("操作卡片已就绪" in text for text in deltas)
    assert [name for name, _arguments in tools.calls] == [
        "set_authoring_requirements",
    ]


def test_explicit_2d_request_cannot_fall_back_to_a_derived_3d_output(tmp_path):
    construction = {
        "schema_version": 1,
        "name": "2D平板",
        "plane": "XY",
        "nodes": [
            {
                "id": "plate",
                "kind": "rectangle",
                "x": 0,
                "y": 0,
                "width": 100,
                "height": 300,
            }
        ],
        "result_node_id": "plate",
    }
    provider = FakeProvider(
        [
            _tool_response(
                ToolCall(
                    "wrong-3d",
                    "prepare_planar_construction_proposal",
                    {
                        "part_function": "2D平板",
                        "construction": construction,
                        "output": {
                            "kind": "extrusion",
                            "profile_selection": "unique_material_profile",
                            "height": 10,
                        },
                    },
                )
            ),
            _tool_response(
                ToolCall(
                    "correct-2d",
                    "prepare_planar_construction_proposal",
                    {
                        "part_function": "2D平板",
                        "construction": construction,
                        "output": {"kind": "planar"},
                    },
                )
            ),
        ]
    )
    tools = _AdditionalModelToolRegistry()
    engine = AgentSessionEngine(
        tmp_path / "workspace",
        provider,
        session_id="ses_planar_dimension_guard",
        dynamic_tools=tools,
    )

    events = engine.send_message("创建一个二维平板")

    assert len(provider.requests) == 2
    assert tools.calls == [
        (
            "prepare_planar_construction_proposal",
            {
                "part_function": "2D平板",
                "construction": construction,
                "output": {"kind": "planar"},
            },
        )
    ]
    first_result = next(
        event.data["result"]
        for event in events
        if event.event is EngineEventType.TOOL_COMPLETED
        and event.data["call_id"] == "wrong-3d"
    )
    assert first_result["data"]["required_output"] == "planar"


def test_planar_retry_limit_stops_provider_after_three_failed_calls(tmp_path):
    class RetryLimitedTools:
        def __init__(self):
            self.calls = 0
            self.definitions = (
                ToolDefinition(
                    "prepare_planar_construction_proposal",
                    "prepare planar construction",
                    {"type": "object"},
                ),
            )

        @property
        def provider_snapshot(self):
            return None

        def refresh_turn_snapshot(self, published_tool_names=()):
            del published_tool_names
            return None

        def dispatch(self, name, arguments, context):
            del name, arguments
            self.calls += 1
            exhausted = self.calls >= 3
            return ToolResult(
                ok=False,
                session_id=context.session_id,
                input_revision=context.expected_revision,
                idempotency_key=context.idempotency_key,
                summary="planar construction failed",
                data={
                    "retry": {
                        "attempt": self.calls,
                        "limit": 3,
                        "retryable": not exhausted,
                        "blocker": (
                            "Planar construction retry limit reached after three attempts."
                            if exhausted
                            else None
                        ),
                    }
                },
            )

    provider = FakeProvider(
        [
            _tool_response(
                ToolCall(
                    f"retry-{index}",
                    "prepare_planar_construction_proposal",
                    {"output": "planar"},
                )
            )
            for index in range(1, 5)
        ]
    )
    tools = RetryLimitedTools()
    engine = AgentSessionEngine(
        tmp_path / "workspace",
        provider,
        session_id="ses_planar_retry_limit",
        dynamic_tools=tools,
    )

    events = engine.send_message("创建一个二维平板")

    assert tools.calls == 3
    assert len(provider.requests) == 3
    assert any(
        event.event is EngineEventType.MESSAGE_DELTA
        and "本轮已停止继续提交" in event.data["text"]
        for event in events
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


def test_planar_edit_cannot_claim_submission_without_a_proposal_tool_call(tmp_path):
    provider = FakeProvider(
        [
            _text_response("我现在提交修订方案。"),
            _tool_response(
                ToolCall("read-edit", "read_geometry_edit_context", {})
            ),
            _text_response("提交。"),
            _tool_response(
                ToolCall(
                    "prepare-edit",
                    "prepare_geometry_edit",
                    {
                        "part_id": "P1",
                        "edit": {
                            "operation": "add_path_slot",
                            "points": [{"x": 0, "y": 0}, {"x": 1, "y": 0}],
                            "width": 0.1,
                            "cap": "square",
                            "join": "miter",
                        },
                    },
                )
            ),
        ]
    )
    tools = _GeometryEditToolRegistry()
    engine = AgentSessionEngine(
        tmp_path / "workspace",
        provider,
        session_id="ses_planar_edit_progress",
        dynamic_tools=tools,
    )

    events = engine.send_message("当然，切除出S形状的槽即可")

    assert len(provider.requests) == 4
    assert [name for name, _arguments in tools.calls] == [
        "read_geometry_edit_context",
        "prepare_geometry_edit",
    ]
    assert "read_geometry_edit_context" in (
        provider.requests[1].messages[-1].content or ""
    )
    assert "prepare_geometry_edit" in (
        provider.requests[3].messages[-1].content or ""
    )
    assert not any(
        "提交" in str(event.data.get("text", ""))
        for event in events
        if event.event is EngineEventType.MESSAGE_DELTA
    )
    assert any(
        event.event is EngineEventType.TOOL_COMPLETED
        and event.data["tool"] == "prepare_geometry_edit"
        and event.data["result"]["data"]["state"] == "pending_confirmation"
        for event in events
    )


def test_planar_edit_blocks_premature_prepare_but_audit_distinguishes_it(tmp_path):
    edit = {
        "part_id": "P1",
        "edit": {
            "operation": "add_rectangle",
            "x": 20,
            "y": 20,
            "width": 10,
            "height": 10,
        },
    }
    provider = FakeProvider(
        [
            _tool_response(
                ToolCall("premature-prepare", "prepare_geometry_edit", edit)
            ),
            _tool_response(
                ToolCall(
                    "read-edit",
                    "read_geometry_edit_context",
                    {"part_id": "P1"},
                )
            ),
            _tool_response(
                ToolCall("prepared", "prepare_geometry_edit", edit)
            ),
        ]
    )
    tools = _GeometryEditWithCatalogToolRegistry()
    engine = AgentSessionEngine(
        tmp_path / "workspace",
        provider,
        session_id="ses_planar_edit_premature_prepare",
        dynamic_tools=tools,
    )

    engine.send_message("在现有平板上增加一个矩形槽")

    assert [name for name, _arguments in tools.calls] == [
        "read_geometry_edit_context",
        "prepare_geometry_edit",
    ]
    audit = json.loads(engine._audit_path().read_text(encoding="utf-8"))
    assert audit["entries"][0]["tool_call_flags"]["called_tool_names"] == [
        "prepare_geometry_edit"
    ]
    assert audit["entries"][0]["tool_call_flags"]["accepted_tool_names"] == []
    assert audit["entries"][1]["tool_call_flags"]["accepted_tool_names"] == [
        "read_geometry_edit_context"
    ]


def test_planar_edit_allows_supplemental_read_before_prepare(tmp_path):
    edit = {
        "part_id": "P1",
        "edit": {
            "operation": "add_path_slot",
            "points": [
                {"x": 190, "y": 75},
                {"x": 190, "y": 25},
                {"x": 225, "y": 25},
                {"x": 225, "y": 75},
            ],
            "width": 6,
            "cap": "square",
            "join": "miter",
        },
    }
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
                ToolCall(
                    "read-features",
                    "read_geometry_feature_catalog",
                    {},
                )
            ),
            _tool_response(
                ToolCall("prepare-edit", "prepare_geometry_edit", edit)
            ),
        ]
    )
    tools = _GeometryEditWithCatalogToolRegistry()
    engine = AgentSessionEngine(
        tmp_path / "workspace",
        provider,
        session_id="ses_planar_edit_supplemental_read",
        dynamic_tools=tools,
    )

    events = engine.send_message("在H槽旁边加入一个U形槽")

    assert len(provider.requests) == 3
    assert [name for name, _arguments in tools.calls] == [
        "read_geometry_edit_context",
        "read_geometry_feature_catalog",
        "prepare_geometry_edit",
    ]
    assert not any(
        event.event is EngineEventType.MESSAGE_DELTA
        and event.data.get("text") == "当前几何能力检查未完成，请重试。"
        for event in events
    )
    assert any(
        event.event is EngineEventType.TOOL_COMPLETED
        and event.data["tool"] == "prepare_geometry_edit"
        and event.data["result"]["data"]["state"] == "pending_confirmation"
        for event in events
    )


def test_planar_edit_allows_read_only_discovery_before_required_probe(tmp_path):
    edit = {
        "part_id": "P1",
        "edit": {
            "operation": "add_polygon",
            "vertices": [
                {"x": 190, "y": 75},
                {"x": 190, "y": 25},
                {"x": 225, "y": 25},
                {"x": 225, "y": 75},
            ],
        },
    }
    provider = FakeProvider(
        [
            _tool_response(
                ToolCall(
                    "read-features",
                    "read_geometry_feature_catalog",
                    {},
                )
            ),
            _tool_response(
                ToolCall(
                    "read-edit",
                    "read_geometry_edit_context",
                    {"part_id": "P1"},
                )
            ),
            _tool_response(
                ToolCall("prepare-edit", "prepare_geometry_edit", edit)
            ),
        ]
    )
    tools = _GeometryEditWithCatalogToolRegistry()
    engine = AgentSessionEngine(
        tmp_path / "workspace",
        provider,
        session_id="ses_planar_edit_discovery_before_probe",
        dynamic_tools=tools,
    )

    engine.send_message("在现有平板上增加一个槽")

    assert [name for name, _arguments in tools.calls] == [
        "read_geometry_feature_catalog",
        "read_geometry_edit_context",
        "prepare_geometry_edit",
    ]


def test_planar_edit_allows_context_reread_after_failed_prepare(tmp_path):
    edit = {
        "part_id": "P1",
        "edit": {
            "operation": "add_polygon",
            "vertices": [
                {"x": 190, "y": 75},
                {"x": 190, "y": 25},
                {"x": 225, "y": 25},
                {"x": 225, "y": 75},
            ],
        },
    }
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
            _tool_response(
                ToolCall(
                    "read-features",
                    "read_geometry_feature_catalog",
                    {},
                )
            ),
            _tool_response(
                ToolCall("prepare-corrected", "prepare_geometry_edit", edit)
            ),
        ]
    )
    tools = _RetryingGeometryEditWithCatalogToolRegistry()
    engine = AgentSessionEngine(
        tmp_path / "workspace",
        provider,
        session_id="ses_planar_edit_reread_after_failure",
        dynamic_tools=tools,
    )

    events = engine.send_message("在现有平板上增加一个槽")

    assert [name for name, _arguments in tools.calls] == [
        "read_geometry_edit_context",
        "prepare_geometry_edit",
        "read_geometry_feature_catalog",
        "prepare_geometry_edit",
    ]
    assert any(
        event.event is EngineEventType.TOOL_COMPLETED
        and event.data["call_id"] == "prepare-corrected"
        and event.data["result"]["data"]["state"] == "pending_confirmation"
        for event in events
    )


def test_planar_edit_allows_clarification_after_context_read(tmp_path):
    provider = FakeProvider(
        [
            _tool_response(
                ToolCall(
                    "read-edit",
                    "read_geometry_edit_context",
                    {"part_id": "P1"},
                )
            ),
            _text_response("请提供U形槽的槽宽和外包尺寸。"),
        ]
    )
    tools = _GeometryEditWithCatalogToolRegistry()
    engine = AgentSessionEngine(
        tmp_path / "workspace",
        provider,
        session_id="ses_planar_edit_clarification",
        dynamic_tools=tools,
    )

    events = engine.send_message("在H槽旁边加入一个U形槽")

    assert len(provider.requests) == 2
    assert [name for name, _arguments in tools.calls] == [
        "read_geometry_edit_context"
    ]
    assert any(
        event.event is EngineEventType.MESSAGE_DELTA
        and event.data.get("text") == "请提供U形槽的槽宽和外包尺寸。"
        for event in events
    )
    clarification_start = next(
        event
        for event in events
        if event.event is EngineEventType.MESSAGE_STARTED
        and event.data.get("presentation_kind") == "decision_request"
    )
    assert clarification_start.data["presentation_kind"] == "decision_request"


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


def test_tool_round_keeps_explicit_user_decision_visible(tmp_path):
    process = "我先读取现有特征位置。"
    decision = "请确认开口深度。"
    provider = FakeProvider(
        [
            ProviderResponse(
                AssistantMessage(
                    "assistant",
                    content=f"{process}\n\n{decision}",
                    tool_calls=(
                        ToolCall(
                            "read-edit",
                            "read_geometry_edit_context",
                            {"part_id": "P1"},
                        ),
                    ),
                ),
                finish_reason="tool_calls",
            ),
            _text_response("读取完成。"),
        ]
    )
    tools = _GeometryEditWithCatalogToolRegistry()
    engine = AgentSessionEngine(
        tmp_path / "workspace",
        provider,
        session_id="ses_tool_round_decision_presentation",
        dynamic_tools=tools,
    )

    events = engine.send_message("在现有平板上增加一个槽")

    displayed = [
        (
            event.data.get("presentation_kind"),
            events[index + 1].data.get("text"),
        )
        for index, event in enumerate(events[:-1])
        if event.event is EngineEventType.MESSAGE_STARTED
        and events[index + 1].event is EngineEventType.MESSAGE_DELTA
        and events[index + 1].data.get("text") in {process, decision}
    ]
    assert displayed == [
        ("process", process),
        ("decision_request", decision),
    ]


def test_failed_tool_mixed_process_and_decision_are_presented_separately(
    tmp_path,
):
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
    process = "本地校验拒绝了刚才的轮廓，我重新核对了已有尺寸。"
    decision = "请提供开口深度。"
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
            _text_response(f"{process}\n\n{decision}"),
        ]
    )
    tools = _RetryingGeometryEditWithCatalogToolRegistry()
    engine = AgentSessionEngine(
        tmp_path / "workspace",
        provider,
        session_id="ses_mixed_process_decision_presentation",
        dynamic_tools=tools,
    )

    events = engine.send_message("在现有平板上增加一个槽")

    displayed = [
        (
            event.data.get("presentation_kind"),
            events[index + 1].data.get("text"),
        )
        for index, event in enumerate(events[:-1])
        if event.event is EngineEventType.MESSAGE_STARTED
        and events[index + 1].event is EngineEventType.MESSAGE_DELTA
        and events[index + 1].data.get("text") in {process, decision}
    ]
    assert displayed == [
        ("process", process),
        ("decision_request", decision),
    ]


def test_undo_stale_edit_resynchronizes_before_new_proposal(tmp_path):
    corrected_edit = {
        "part_id": "P1",
        "edit": {
            "operation": "add_polygon",
            "vertices": [
                {"x": 20, "y": 20},
                {"x": 35, "y": 20},
                {"x": 35, "y": 35},
                {"x": 20, "y": 35},
            ],
        },
    }
    provider = FakeProvider(
        [
            _text_response("我先按旧版本说明。"),
            _tool_response(
                ToolCall("sync", "read_authoring_context", {})
            ),
            _text_response("修正轮廓如下。"),
            _tool_response(
                ToolCall("read-edit", "read_geometry_edit_context", {})
            ),
            _text_response("方案已确认，等待本地操作执行完成。"),
            _tool_response(
                ToolCall("prepare-edit", "prepare_geometry_edit", corrected_edit)
            ),
        ]
    )
    tools = _StaleGeometryEditToolRegistry()
    engine = AgentSessionEngine(
        tmp_path / "workspace",
        provider,
        session_id="ses_stale_undo_edit",
        dynamic_tools=tools,
    )

    events = engine.send_message("撤销后重新生成这个槽轮廓")

    assert [name for name, _arguments in tools.calls] == [
        "read_authoring_context",
        "read_geometry_edit_context",
        "prepare_geometry_edit",
    ]
    assert tools.calls[-1][1] == corrected_edit
    assert "read_authoring_context" in (
        provider.requests[1].messages[-1].content or ""
    )
    assert "read_geometry_edit_context" in (
        provider.requests[3].messages[-1].content or ""
    )
    assert "proposal-grounding correction" in (
        provider.requests[5].messages[-1].content or ""
    )
    assert not any(
        "旧版本" in str(event.data.get("text", ""))
        or "等待本地操作执行完成" in str(event.data.get("text", ""))
        for event in events
        if event.event is EngineEventType.MESSAGE_DELTA
    )
    assert any(
        event.event is EngineEventType.TOOL_COMPLETED
        and event.data["tool"] == "prepare_geometry_edit"
        and event.data["result"]["data"]["state"] == "pending_confirmation"
        for event in events
    )


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


def test_branching_slot_rejects_single_path_before_dispatch(tmp_path):
    invalid = {
        "part_function": "二维平板，中央分叉槽",
        "construction": {
            "schema_version": 1,
            "name": "invalid_branching_slot",
            "plane": "XY",
            "nodes": [
                {
                    "id": "plate",
                    "kind": "rectangle",
                    "x": 0,
                    "y": 0,
                    "width": 300,
                    "height": 100,
                },
                {
                    "id": "slot",
                    "kind": "path_stroke",
                    "points": [
                        {"x": 140, "y": 35},
                        {"x": 160, "y": 35},
                        {"x": 160, "y": 65},
                        {"x": 140, "y": 65},
                    ],
                    "width": 10,
                    "cap": "butt",
                    "join": "miter",
                },
                {
                    "id": "result",
                    "kind": "difference",
                    "base": "plate",
                    "subtract": ["slot"],
                },
            ],
            "result_node_id": "result",
        },
        "output": {"kind": "planar"},
    }
    corrected = {
        "part_function": "二维平板，中央分叉槽",
        "construction": {
            "schema_version": 1,
            "name": "connected_branching_slot",
            "plane": "XY",
            "nodes": [
                {
                    "id": "plate",
                    "kind": "rectangle",
                    "x": 0,
                    "y": 0,
                    "width": 300,
                    "height": 100,
                },
                {
                    "id": "left_stem",
                    "kind": "rectangle",
                    "x": 130,
                    "y": 30,
                    "width": 10,
                    "height": 40,
                },
                {
                    "id": "cross_stem",
                    "kind": "rectangle",
                    "x": 130,
                    "y": 45,
                    "width": 40,
                    "height": 10,
                },
                {
                    "id": "right_stem",
                    "kind": "rectangle",
                    "x": 160,
                    "y": 30,
                    "width": 10,
                    "height": 40,
                },
                {
                    "id": "slot",
                    "kind": "union",
                    "operands": ["left_stem", "cross_stem", "right_stem"],
                },
                {
                    "id": "result",
                    "kind": "difference",
                    "base": "plate",
                    "subtract": ["slot"],
                },
            ],
            "result_node_id": "result",
        },
        "output": {"kind": "planar"},
    }
    provider = FakeProvider(
        [
            _tool_response(
                ToolCall(
                    "invalid-slot",
                    "prepare_planar_construction_proposal",
                    invalid,
                )
            ),
            _tool_response(
                ToolCall(
                    "corrected-slot",
                    "prepare_planar_construction_proposal",
                    corrected,
                )
            ),
        ]
    )
    tools = _AdditionalModelToolRegistry()
    engine = AgentSessionEngine(
        tmp_path / "workspace",
        provider,
        session_id="ses_branching_slot_guard",
        dynamic_tools=tools,
    )

    events = engine.send_message(
        "创建一个2D平板，在中央做一条中心线包含分叉节点的槽"
    )

    assert tools.calls == [
        ("prepare_planar_construction_proposal", corrected)
    ]
    assert len(provider.requests) == 2
    assert "single non-branching open centerline" in (
        provider.requests[1].messages[-1].content or ""
    )
    completed = [
        event
        for event in events
        if event.event is EngineEventType.TOOL_COMPLETED
        and event.data["tool"] == "prepare_planar_construction_proposal"
    ]
    assert completed[0].data["result"]["ok"] is False
    assert completed[-1].data["result"]["data"]["state"] == "pending_confirmation"


def test_nonbranching_path_slot_rejects_disconnected_rectangle_fallback(tmp_path):
    correct_edit = {
        "part_id": "P1",
        "edit": {
            "operation": "add_path_slot",
            "points": [
                {"x": 75, "y": 80},
                {"x": 40, "y": 80},
                {"x": 40, "y": 50},
                {"x": 75, "y": 50},
                {"x": 75, "y": 20},
                {"x": 40, "y": 20},
            ],
            "width": 6,
            "cap": "square",
            "join": "miter",
        },
    }
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
                ToolCall(
                    "bad-edit",
                    "prepare_geometry_edit",
                    {
                        "part_id": "P1",
                        "edit": {
                            "operation": "batch",
                            "edits": [
                                {
                                    "operation": "add_rectangle",
                                    "x": 40,
                                    "y": 70,
                                    "width": 35,
                                    "height": 6,
                                },
                                {
                                    "operation": "add_rectangle",
                                    "x": 40,
                                    "y": 47,
                                    "width": 35,
                                    "height": 6,
                                },
                                {
                                    "operation": "add_rectangle",
                                    "x": 40,
                                    "y": 24,
                                    "width": 35,
                                    "height": 6,
                                },
                            ],
                        },
                    },
                )
            ),
            _tool_response(
                ToolCall("correct-edit", "prepare_geometry_edit", correct_edit)
            ),
        ]
    )
    tools = _GeometryEditToolRegistry()
    engine = AgentSessionEngine(
        tmp_path / "workspace",
        provider,
        session_id="ses_nonbranching_path_slot_guard",
        dynamic_tools=tools,
    )

    events = engine.send_message(
        "请加入一条由单条开放无分叉中心线定义的连续定宽槽"
    )

    assert [name for name, _arguments in tools.calls] == [
        "read_geometry_edit_context",
        "prepare_geometry_edit",
    ]
    assert tools.calls[-1][1] == correct_edit
    bad_result = next(
        event.data["result"]
        for event in events
        if event.event is EngineEventType.TOOL_COMPLETED
        and event.data["call_id"] == "bad-edit"
    )
    assert not bad_result["ok"]
    assert bad_result["data"]["required_operation"] == "add_path_slot"
    assert any(
        event.event is EngineEventType.TOOL_COMPLETED
        and event.data["call_id"] == "correct-edit"
        and event.data["result"]["data"]["state"] == "pending_confirmation"
        for event in events
    )
    preview = next(
        event.data["text"]
        for event in events
        if event.event is EngineEventType.MESSAGE_DELTA
        and "方案预览" in event.data["text"]
    )
    assert "切除一条连续定宽槽" in preview
    assert "(75, 80) → (40, 80)" in preview
    assert "add_path_slot" not in preview
    assert "P1" not in preview
    assert "{" not in preview
    assert any(
        event.event is EngineEventType.MESSAGE_DELTA
        and "上一版方案未通过校验" in event.data["text"]
        for event in events
    )


def test_generic_geometry_feature_guard_requires_requested_slot_and_holes():
    partial = {
        "part_function": "宽平板",
        "geometry": {
            "kind": "planar_profiles",
            "profiles": [
                {
                    "kind": "rectangle",
                    "x": 0,
                    "y": 0,
                    "width": 500,
                    "height": 300,
                }
            ],
        },
    }
    complete = {
        "part_function": "宽平板，中央开槽，四周开孔",
        "geometry": {
            "kind": "planar_profiles",
            "profiles": [
                {
                    "kind": "rectangle",
                    "x": 0,
                    "y": 0,
                    "width": 500,
                    "height": 300,
                    "role": "material",
                },
                {
                    "kind": "rectangle",
                    "x": 225,
                    "y": 110,
                    "width": 50,
                    "height": 80,
                    "role": "hole",
                },
                {
                    "kind": "circle",
                    "center_x": 40,
                    "center_y": 40,
                    "radius": 10,
                    "role": "hole",
                },
            ],
        },
    }
    request = "做一个宽平板，中央开槽，四周开孔"
    assert _missing_requested_geometry_features(request, partial) == (
        "slot_or_cutout",
        "holes",
    )
    assert _missing_requested_geometry_features(request, complete) == ()


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


def _attached_engine(tmp_path, provider):
    source = write_perforated_plate_style_inp(
        tmp_path,
        "engine_model.inp",
        ("*Boundary", "Set-right, 1, 1, 0.05"),
    )
    workspace = tmp_path / "workspace"
    engine = AgentSessionEngine(
        workspace,
        provider,
        session_id="ses_engine",
    )
    artifact = ArtifactStore(workspace).copy_input(engine.session_id, source)
    engine.attach_artifact(artifact.artifact_id)
    return engine, source


def _ready_engine(tmp_path, provider):
    engine, _ = _attached_engine(tmp_path, provider)
    first = engine.revisions.require_current(engine.session_id)
    engine.registry.dispatch(
        "set_unit_context",
        {
            "length": "mm",
            "force": "N",
            "stress": "MPa",
            "density": "tonne/mm^3",
            "acceleration": "mm/s^2",
        },
        ToolExecutionContext(engine.session_id, first.revision, "ready_units"),
    )
    second = engine.revisions.require_current(engine.session_id)
    engine.registry.dispatch(
        "set_result_requests",
        {
            "queries": [{"kind": "max_displacement_magnitude"}],
            "export_formats": [],
        },
        ToolExecutionContext(
            engine.session_id,
            second.revision,
            "ready_results",
        ),
    )
    engine.get_analysis_summary()
    return engine


def test_attach_does_not_report_draft_requirements_as_input_errors(tmp_path):
    source = write_perforated_plate_style_inp(
        tmp_path,
        "attach_diagnostics.inp",
        ("*Cload", "Set-right, 1, 10."),
        section_data=("1.,",),
    )
    workspace = tmp_path / "workspace"
    engine = AgentSessionEngine(workspace, FakeProvider())
    artifact = ArtifactStore(workspace).copy_input(engine.session_id, source)

    events = engine.attach_artifact(artifact.artifact_id)

    diagnostic_codes = {
        event.data["diagnostic"]["code"]
        for event in events
        if event.event == EngineEventType.DIAGNOSTIC
    }
    assert diagnostic_codes.isdisjoint(
        {
            "UNIT_CONTEXT_REQUIRED",
            "RESULT_REQUEST_REQUIRED",
            "INVALID_INPUT",
        }
    )
    current = engine.revisions.require_current(engine.session_id)
    assert current.spec.analysis_step == "Step-1"
    assert engine.get_snapshot().phase == SessionPhase.INSPECTED


def test_fake_provider_completes_unit_result_and_summary_tool_loop(tmp_path):
    provider = FakeProvider(
        [
            _tool_response(
                ToolCall(
                    "call_units",
                    "set_unit_context",
                    {
                        "length": "mm",
                        "force": "N",
                        "stress": "MPa",
                        "density": "tonne/mm^3",
                        "acceleration": "mm/s^2",
                    },
                ),
                ToolCall(
                    "call_results",
                    "set_result_requests",
                    {
                        "queries": [
                            {"kind": "max_displacement_magnitude"},
                            {
                                "kind": "reaction_sum",
                                "component": 1,
                                "node_set": "Set-left",
                            },
                        ],
                        "export_formats": ["csv", "vtk"],
                    },
                ),
            ),
            _tool_response(
                ToolCall("call_summary", "get_analysis_summary", {})
            ),
            _text_response("分析摘要已准备好，请检查后输入 /confirm。"),
        ]
    )
    engine, _ = _attached_engine(tmp_path, provider)

    events = engine.send_message("单位和结果要求如下，请形成分析摘要。")

    assert engine.get_snapshot().phase == SessionPhase.AWAITING_CONFIRMATION
    assert engine.get_snapshot().revision == 3
    assert any(
        event.event == EngineEventType.MESSAGE_DELTA
        and "/confirm" in event.data["text"]
        for event in events
    )
    completed = [
        event
        for event in events
        if event.event == EngineEventType.TOOL_COMPLETED
    ]
    assert [event.data["tool"] for event in completed] == [
        "set_unit_context",
        "set_result_requests",
        "get_analysis_summary",
    ]
    assert any(
        event.event == EngineEventType.ANALYSIS_SUMMARY
        for event in events
    )


def test_inspection_worker_failure_prevents_same_turn_tool_retry(
    monkeypatch,
    tmp_path,
):
    def finish_without_tools(messages, tools):
        assert tools == ()
        return _text_response("模型检查进程暂时失败，请重试。")

    provider = FakeProvider(
        [
            _tool_response(
                ToolCall("call_summary_failed", "get_analysis_summary", {})
            ),
            finish_without_tools,
        ]
    )
    engine, _ = _attached_engine(tmp_path, provider)
    inspection_calls = 0

    def fail_inspection(*args, **kwargs):
        nonlocal inspection_calls
        inspection_calls += 1
        raise InspectionWorkerError("The inspection protocol failed.")

    monkeypatch.setattr(
        engine.registry.inspector,
        "inspect",
        fail_inspection,
    )

    first_events = engine.send_message("请生成分析摘要。")

    assert inspection_calls == 1
    assert any(
        event.event == EngineEventType.DIAGNOSTIC
        and event.data["diagnostic"]["code"]
        == DiagnosticCode.WORKER_CRASH.value
        for event in first_events
    )
    assert [
        event.data["tool"]
        for event in first_events
        if event.event == EngineEventType.TOOL_STARTED
    ] == ["get_analysis_summary"]

    provider.queue(
        _tool_response(
            ToolCall("call_summary_retried", "get_analysis_summary", {})
        ),
        finish_without_tools,
    )
    second_events = engine.send_message("重试生成摘要。")

    assert inspection_calls == 2
    assert provider.requests[-2].tools
    assert [
        event.data["tool"]
        for event in second_events
        if event.event == EngineEventType.TOOL_STARTED
    ] == ["get_analysis_summary"]


@pytest.mark.integration
def test_solved_model_is_queried_then_explained_by_agent_without_new_run(
    tmp_path,
):
    provider = FakeProvider()
    engine, source = _attached_engine(tmp_path, provider)
    first = engine.revisions.require_current(engine.session_id)
    engine.registry.dispatch(
        "set_unit_context",
        {
            "length": "mm",
            "force": "N",
            "stress": "MPa",
            "density": "tonne/mm^3",
            "acceleration": "mm/s^2",
        },
        ToolExecutionContext(
            engine.session_id,
            first.revision,
            "postsolve_units",
        ),
    )
    engine.get_analysis_summary()
    completed = engine.confirm_revision()
    run_id = next(
        event.data["run_id"]
        for event in completed
        if event.event == EngineEventType.RUN_COMPLETED
    )
    solved = engine.get_snapshot()
    assert solved.phase == SessionPhase.SOLVED
    assert solved.revision == 2

    def explain_result(messages, tools):
        tool_message = next(
            message
            for message in reversed(messages)
            if message.role == "tool"
        )
        payload = json.loads(tool_message.content)
        scalar = payload["data"]["result_summary"]["scalars"][0]
        assert scalar["region"] == "Surf-right"
        assert scalar["unit"] == "mm"
        return _text_response(
            f"自由端最大位移为 {scalar['value']:.6g} mm，"
            f"位于节点 {scalar['node_id']}。"
        )

    provider.queue(
        _tool_response(
            ToolCall(
                "postsolve_edge_displacement",
                "query_results",
                {
                    "queries": [
                        {
                            "kind": "max_displacement_magnitude",
                            "edge": "Surf-right",
                        }
                    ]
                },
            )
        ),
        explain_result,
    )

    events = engine.send_message("分析自由端的最大位移，并说明位置。")

    tool_result = next(
        event.data["result"]
        for event in events
        if event.event == EngineEventType.TOOL_COMPLETED
        and event.data["tool"] == "query_results"
    )
    assert tool_result["ok"] is True
    tool_payload = next(
        message.content
        for request in reversed(provider.requests)
        for message in reversed(request.messages)
        if message.role == "tool"
    )
    assert "solution.npy" not in tool_payload
    assert '"reactions"' not in tool_payload
    assert str(source) not in tool_payload
    provider_result = json.loads(tool_payload)
    assert set(provider_result["data"]) == {"result_summary"}
    provider_summary = provider_result["data"]["result_summary"]
    assert set(provider_summary) == {
        "schema_version",
        "run_id",
        "step",
        "finite_vectors",
        "scalars",
        "diagnostics",
    }
    assert len(provider_summary["scalars"]) == 1
    assert set(provider_summary["scalars"][0]) == {
        "schema_version",
        "query_kind",
        "value",
        "unit",
        "measure",
        "run_id",
        "step",
        "node_id",
        "element_id",
        "region",
    }
    expected_value = (
        f"{provider_summary['scalars'][0]['value']:.6g}"
    )
    assert any(
        event.event == EngineEventType.MESSAGE_DELTA
        and "自由端最大位移为" in event.data["text"]
        and expected_value in event.data["text"]
        and "mm" in event.data["text"]
        and str(provider_summary["scalars"][0]["node_id"])
        in event.data["text"]
        for event in events
    )
    visible_text = "".join(
        event.data["text"]
        for event in events
        if event.event == EngineEventType.MESSAGE_DELTA
    )
    for unwanted in (
        "本地 FEM",
        "本地结果",
        "由本地",
        "未与 Abaqus",
        "没有与 Abaqus",
    ):
        assert unwanted not in visible_text
    after = engine.get_snapshot()
    assert after.phase == SessionPhase.SOLVED
    assert after.revision == solved.revision
    assert after.revision_hash == solved.revision_hash
    assert after.active_run_id == run_id
    runs = (
        engine.workspace
        / "sessions"
        / engine.session_id
        / "runs"
    )
    assert len(list(runs.iterdir())) == 1


@pytest.mark.integration
def test_reopened_solved_session_can_query_saved_solution(tmp_path):
    engine = _ready_engine(tmp_path, FakeProvider())
    engine.confirm_revision()
    session_id = engine.session_id
    run_id = engine.get_snapshot().active_run_id
    provider = FakeProvider(
        [
            _tool_response(
                ToolCall(
                    "reopened_query",
                    "query_results",
                    {
                        "queries": [
                            {
                                "kind": "max_displacement_magnitude",
                                "edge": "Surf-right",
                            }
                        ]
                    },
                )
            ),
            _text_response("已从保存的解中分析自由端位移。"),
        ]
    )
    reopened = AgentSessionEngine(
        engine.workspace,
        provider,
        session_id=session_id,
    )

    events = reopened.send_message("继续分析自由端位移。")

    assert any(
        event.event == EngineEventType.TOOL_COMPLETED
        and event.data["tool"] == "query_results"
        and event.data["result"]["ok"] is True
        for event in events
    )
    assert reopened.get_snapshot().phase == SessionPhase.SOLVED
    assert reopened.get_snapshot().active_run_id == run_id


@pytest.mark.integration
def test_postsolve_result_configuration_cannot_discard_active_run(tmp_path):
    engine = _ready_engine(tmp_path, FakeProvider())
    engine.confirm_revision()
    before = engine.get_snapshot()
    current = engine.revisions.require_current(engine.session_id)

    result = engine.registry.dispatch(
        "set_result_requests",
        {
            "queries": [{"kind": "max_displacement_magnitude"}],
            "export_formats": [],
        },
        ToolExecutionContext(
            engine.session_id,
            current.revision,
            "postsolve_wrong_tool",
            completed_run=engine._active_run,
        ),
    )

    after = engine.get_snapshot()
    assert result.ok is False
    assert result.diagnostics[0].code == "INVALID_TOOL_ARGUMENTS"
    assert "query_results" in result.diagnostics[0].message
    assert after.revision == before.revision
    assert after.active_run_id == before.active_run_id
    assert after.phase == SessionPhase.SOLVED


@pytest.mark.integration
def test_legacy_run_can_return_its_matching_precomputed_summary(tmp_path):
    engine = _ready_engine(tmp_path, FakeProvider())
    engine.confirm_revision()
    current = engine.revisions.require_current(engine.session_id)
    legacy_response = replace(
        engine._active_run,
        artifacts=tuple(
            item
            for item in engine._active_run.artifacts
            if item.kind != "solution"
        ),
    )

    result = engine.registry.dispatch(
        "query_results",
        {
            "queries": [
                item.to_dict()
                for item in current.spec.requested_queries
            ]
        },
        ToolExecutionContext(
            engine.session_id,
            current.revision,
            "legacy_precomputed_query",
            completed_run=legacy_response,
        ),
    )

    assert result.ok is True
    assert result.data["result_summary"] == (
        legacy_response.result_summary.to_dict()
    )
    legacy_default = engine.registry.dispatch(
        "query_results",
        {},
        ToolExecutionContext(
            engine.session_id,
            current.revision,
            "legacy_default_query",
            completed_run=legacy_response,
        ),
    )
    assert legacy_default.ok is True
    assert legacy_default.data == result.data

    different = engine.registry.dispatch(
        "query_results",
        {
            "queries": [
                {
                    "kind": "max_displacement_component",
                    "component": 1,
                }
            ]
        },
        ToolExecutionContext(
            engine.session_id,
            current.revision,
            "legacy_different_query",
            completed_run=legacy_response,
        ),
    )
    assert different.ok is False
    assert different.diagnostics[0].code == "RESULT_QUERY_FAILED"
    assert "predates reusable" in different.diagnostics[0].message
    assert engine.get_snapshot().phase == SessionPhase.SOLVED


@pytest.mark.integration
def test_failed_postsolve_query_does_not_change_successful_run(tmp_path):
    engine = _ready_engine(tmp_path, FakeProvider())
    engine.confirm_revision()
    before = engine.get_snapshot()
    current = engine.revisions.require_current(engine.session_id)

    result = engine.registry.dispatch(
        "query_results",
        {
            "queries": [
                {
                    "kind": "max_displacement_magnitude",
                    "edge": "missing-edge",
                }
            ]
        },
        ToolExecutionContext(
            engine.session_id,
            current.revision,
            "missing_postsolve_region",
            completed_run=engine._active_run,
        ),
    )

    after = engine.get_snapshot()
    assert result.ok is False
    assert result.diagnostics[0].code == "RESULT_QUERY_FAILED"
    assert after.phase == SessionPhase.SOLVED
    assert after.confirmed is True
    assert after.revision_hash == before.revision_hash
    assert after.active_run_id == before.active_run_id
    assert engine._active_run.status == RunStatus.SUCCEEDED
    assert len(
        list(
            (
                engine.workspace
                / "sessions"
                / engine.session_id
                / "runs"
            ).iterdir()
        )
    ) == 1


def test_missing_deepseek_key_produces_actionable_engine_diagnostic(tmp_path):
    engine = AgentSessionEngine(
        tmp_path / "workspace",
        DeepSeekProvider(
            ProviderConfig(max_retries=0),
            environ={},
        ),
    )

    events = engine.send_message("检查状态")

    diagnostic = next(
        event.data["diagnostic"]
        for event in events
        if event.event == EngineEventType.DIAGNOSTIC
    )
    assert diagnostic["code"] == "PROVIDER_AUTHENTICATION_FAILED"
    assert "DEEPSEEK_API_KEY" in diagnostic["message"]


def test_natural_language_and_model_tool_call_cannot_bypass_confirm(tmp_path):
    provider = FakeProvider(
        [
            _tool_response(
                ToolCall("call_solve", "solve_confirmed_analysis", {})
            ),
            _text_response("请在本地输入 /confirm。"),
        ]
    )
    engine, _ = _attached_engine(tmp_path, provider)

    events = engine.send_message("我确认了，直接求解。")

    tool_event = next(
        event
        for event in events
        if event.event == EngineEventType.TOOL_COMPLETED
    )
    assert tool_event.data["result"]["ok"] is False
    assert (
        tool_event.data["result"]["diagnostics"][0]["code"]
        == "CONFIRMATION_REQUIRED"
    )
    assert engine.get_snapshot().active_run_id is None


def test_attached_local_path_and_raw_input_are_absent_from_provider_requests(tmp_path):
    provider = FakeProvider([_text_response("请先提供单位和结果要求。")])
    engine, source = _attached_engine(tmp_path, provider)
    raw_text = source.read_text(encoding="utf-8")

    engine.send_message("检查已附加的模型。")

    serialized = "\n".join(
        message.content or ""
        for request in provider.requests
        for message in request.messages
    )
    assert str(source) not in serialized
    assert raw_text not in serialized
    assert "*Node" not in serialized


def test_request_context_is_ephemeral_across_provider_tool_loop(tmp_path):
    provider = FakeProvider(
        [
            _tool_response(
                ToolCall(
                    "call_capabilities",
                    "show_capabilities",
                    {"detail": "summary"},
                )
            ),
            _text_response("能力检查完成。"),
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
    provider = FakeProvider([_text_response("不应调用。")])
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
    provider = FakeProvider([_text_response("已记录。")])
    engine, _ = _attached_engine(tmp_path, provider)
    engine.send_message("保留这条会话记录。")
    session_id = engine.session_id

    reopened = AgentSessionEngine(
        engine.workspace,
        FakeProvider([_text_response("继续。")]),
        session_id=session_id,
    )
    events = reopened.send_message("继续。")

    assert any(
        event.event == EngineEventType.MESSAGE_DELTA
        and event.data["text"] == "继续。"
        for event in events
    )


def test_provider_retry_of_identical_mutation_is_idempotent(tmp_path):
    call = ToolCall(
        "call_units_retry",
        "set_unit_context",
        {
            "length": "mm",
            "force": "N",
            "stress": "MPa",
            "density": "tonne/mm^3",
            "acceleration": "mm/s^2",
        },
    )
    provider = FakeProvider(
        [
            _tool_response(call),
            _tool_response(call),
            _text_response("单位已记录。"),
        ]
    )
    engine, _ = _attached_engine(tmp_path, provider)

    engine.send_message("记录单位。")

    assert engine.get_snapshot().revision == 2
    assert len(engine.revisions.list_records(engine.session_id)) == 2


def test_conversation_storage_is_byte_bounded_and_reopenable(tmp_path):
    provider = FakeProvider(
        [_text_response("答" * 500) for _ in range(10)]
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
        FakeProvider([_text_response("继续")]),
        session_id=engine.session_id,
        config=config,
    )
    assert reopened.send_message("继续")


def test_same_provider_call_id_in_a_new_user_turn_is_not_stale(tmp_path):
    call = ToolCall(
        "reused_call_id",
        "set_unit_context",
        {
            "length": "mm",
            "force": "N",
            "stress": "MPa",
            "density": "tonne/mm^3",
            "acceleration": "mm/s^2",
        },
    )
    provider = FakeProvider(
        [
            _tool_response(call),
            _text_response("第一次记录完成。"),
            _tool_response(call),
            _text_response("第二次记录完成。"),
        ]
    )
    engine, _ = _attached_engine(tmp_path, provider)

    engine.send_message("记录单位。")
    first_revision = engine.get_snapshot().revision
    engine.send_message("再次确认同一单位。")

    assert first_revision == 2
    assert engine.get_snapshot().revision == 3


def test_conversation_window_keeps_complete_tool_result_for_provider(tmp_path):
    observed = {}

    def inspect_tool_result(messages, tools):
        observed["roles"] = [message.role for message in messages]
        observed["tool_payload"] = next(
            message.content
            for message in messages
            if message.role == "tool"
        )
        return _text_response("已读取工具结果。")

    provider = FakeProvider(
        [
            _tool_response(
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
                _tool_response(
                    ToolCall(
                        f"audit_{index}",
                        "show_capabilities",
                        {},
                    )
                ),
                _text_response("能力已列出。"),
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
                _tool_response(
                    ToolCall("audit_final", "show_capabilities", {})
                ),
                _text_response("完成。"),
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
        FakeProvider([_text_response("答" * 500)]),
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
        FakeProvider([_text_response("可继续。")]),
        session_id=engine.session_id,
        config=EngineConfig(
            max_provider_message_chars=1_000,
            max_conversation_storage_bytes=1_024,
        ),
    )
    assert reopened.send_message("继续。")


def test_repeated_confirm_after_success_does_not_start_another_run(
    tmp_path,
):
    engine, _ = _attached_engine(tmp_path, FakeProvider())
    first = engine.revisions.require_current(engine.session_id)
    engine.registry.dispatch(
        "set_unit_context",
        {
            "length": "mm",
            "force": "N",
            "stress": "MPa",
            "density": "tonne/mm^3",
            "acceleration": "mm/s^2",
        },
        ToolExecutionContext(engine.session_id, first.revision, "units"),
    )
    second = engine.revisions.require_current(engine.session_id)
    engine.registry.dispatch(
        "set_result_requests",
        {
            "queries": [{"kind": "max_displacement_magnitude"}],
            "export_formats": [],
        },
        ToolExecutionContext(
            engine.session_id,
            second.revision,
            "requested_results",
        ),
    )
    engine.get_analysis_summary()
    completed = engine.confirm_revision()
    run_id = next(
        event.data["run_id"]
        for event in completed
        if event.event == EngineEventType.RUN_COMPLETED
    )

    repeated = engine.confirm_revision()

    assert not any(
        event.event == EngineEventType.RUN_COMPLETED
        for event in repeated
    )
    assert any(
        event.event == EngineEventType.CONFIRMATION_REQUIRED
        and event.data["reason"] == "invalid_session_phase"
        for event in repeated
    )
    assert engine.get_snapshot().active_run_id == run_id


def test_confirm_rejects_wrong_result_region_type_before_worker(
    monkeypatch,
    tmp_path,
):
    engine = _ready_engine(tmp_path, FakeProvider())
    current = engine.revisions.require_current(engine.session_id)
    updated = engine.registry.dispatch(
        "set_result_requests",
        {
            "queries": [
                {
                    "kind": "max_displacement_magnitude",
                    "node_set": "Surf-right",
                }
            ],
            "export_formats": [],
        },
        ToolExecutionContext(
            engine.session_id,
            current.revision,
            "wrong_region_type",
        ),
    )
    assert updated.ok is True

    def unexpected_worker_run(*args, **kwargs):
        raise AssertionError("worker must not run for an invalid result target")

    monkeypatch.setattr(engine.worker, "run", unexpected_worker_run)

    events = engine.confirm_revision()

    diagnostic = next(
        event.data["diagnostic"]
        for event in events
        if event.event == EngineEventType.DIAGNOSTIC
        and event.data["diagnostic"]["code"] == "RESULT_QUERY_FAILED"
    )
    assert "defined as an edge" in diagnostic["message"]
    assert any(
        event.event == EngineEventType.CONFIRMATION_REQUIRED
        and event.data["accepted"] is False
        for event in events
    )
    assert not any(
        event.event == EngineEventType.RUN_PROGRESS
        for event in events
    )


def test_worker_protocol_exception_becomes_retryable_engine_run(
    monkeypatch,
    tmp_path,
):
    engine = _ready_engine(tmp_path, FakeProvider())
    calls = 0

    def run(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise WorkerResponseIntegrityError("damaged response")
        record = engine.revisions.require_current(engine.session_id)
        return WorkerResponse(
            session_id=engine.session_id,
            revision=record.revision,
            revision_hash=record.revision_hash,
            run_id="run_retry_succeeded",
            status=RunStatus.SUCCEEDED,
            result_summary=None,
            artifacts=(),
            diagnostics=(),
            elapsed_seconds=0.01,
        )

    monkeypatch.setattr(engine.worker, "run", run)

    failed = engine.confirm_revision()
    retried = engine.retry_transient_run()

    assert calls == 2
    assert any(event.event == EngineEventType.ERROR for event in failed)
    assert any(
        event.event == EngineEventType.DIAGNOSTIC
        and event.data["diagnostic"]["code"]
        == DiagnosticCode.WORKER_CRASH.value
        for event in failed
    )
    assert any(
        event.event == EngineEventType.STATE_CHANGED
        and event.data["phase"] == SessionPhase.CONFIRMED.value
        for event in failed
    )
    assert any(
        event.event == EngineEventType.RUN_PROGRESS
        and event.data["stage"] == "worker_retry_started"
        for event in retried
    )
    assert engine.get_snapshot().phase == SessionPhase.SOLVED


def test_event_subscriber_receives_worker_progress_before_operation_returns(
    monkeypatch,
    tmp_path,
):
    engine = _ready_engine(tmp_path, FakeProvider())
    progress_seen = threading.Event()
    release_worker = threading.Event()
    received = []

    def sink(event):
        received.append(event)
        if (
            event.event == EngineEventType.RUN_PROGRESS
            and event.data["stage"] == "worker_started"
        ):
            progress_seen.set()

    unsubscribe = engine.subscribe(sink)

    def blocked_run(*args, **kwargs):
        assert release_worker.wait(2.0)
        record = engine.revisions.require_current(engine.session_id)
        return WorkerResponse(
            session_id=engine.session_id,
            revision=record.revision,
            revision_hash=record.revision_hash,
            run_id="run_live_event",
            status=RunStatus.SUCCEEDED,
            result_summary=None,
            artifacts=(),
            diagnostics=(),
            elapsed_seconds=0.01,
        )

    monkeypatch.setattr(engine.worker, "run", blocked_run)
    thread = threading.Thread(target=engine.confirm_revision)

    thread.start()
    assert progress_seen.wait(2.0)
    assert thread.is_alive()
    release_worker.set()
    thread.join(2.0)
    unsubscribe()

    assert not thread.is_alive()
    assert any(
        event.event == EngineEventType.RUN_COMPLETED
        for event in received
    )


def test_cancel_during_confirmation_preflight_prevents_worker_launch(
    monkeypatch,
    tmp_path,
):
    engine = _ready_engine(tmp_path, FakeProvider())
    entered = threading.Event()
    release = threading.Event()
    original = engine.registry.analysis_summary

    def blocked_summary(record):
        entered.set()
        assert release.wait(2.0)
        return original(record)

    monkeypatch.setattr(engine.registry, "analysis_summary", blocked_summary)
    monkeypatch.setattr(
        engine.worker,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("worker must not launch after cancellation")
        ),
    )
    result: list[tuple] = []
    thread = threading.Thread(
        target=lambda: result.append(engine.confirm_revision()),
    )

    thread.start()
    assert entered.wait(2.0)
    cancelled = engine.cancel_active_operation()
    release.set()
    thread.join(2.0)

    assert not thread.is_alive()
    assert cancelled[0].data["scope"] == "operation"
    assert any(
        event.event == EngineEventType.OPERATION_CANCELLED
        and event.data["scope"] == "confirmation"
        for event in result[0]
    )
    assert engine.get_snapshot().phase == SessionPhase.AWAITING_CONFIRMATION


def test_cancelled_attachment_inspection_does_not_commit_a_revision(
    monkeypatch,
    tmp_path,
):
    source = write_perforated_plate_style_inp(
        tmp_path,
        "cancel_attach.inp",
        ("*Boundary", "Set-right, 1, 1, 0.05"),
    )
    workspace = tmp_path / "workspace"
    engine = AgentSessionEngine(workspace, FakeProvider())
    artifact = ArtifactStore(workspace).copy_input(engine.session_id, source)
    entered = threading.Event()

    def blocked_inspection(*args, **kwargs):
        entered.set()
        cancel_event = kwargs["cancel_event"]
        assert cancel_event.wait(2.0)
        raise InspectionWorkerError("cancelled")

    monkeypatch.setattr(
        engine.registry.inspector,
        "inspect",
        blocked_inspection,
    )
    result: list[tuple] = []
    thread = threading.Thread(
        target=lambda: result.append(
            engine.attach_artifact(artifact.artifact_id)
        ),
    )

    thread.start()
    assert entered.wait(2.0)
    engine.cancel_active_operation()
    thread.join(2.0)

    assert not thread.is_alive()
    assert engine.revisions.latest(engine.session_id) is None
    assert any(
        event.event == EngineEventType.OPERATION_CANCELLED
        and event.data["scope"] == "inspection"
        for event in result[0]
    )


def test_idle_cancel_does_not_poison_the_next_summary(tmp_path):
    engine, _ = _attached_engine(tmp_path, FakeProvider())

    cancelled = engine.cancel_active_operation()
    summary = engine.get_analysis_summary()

    assert cancelled[0].data["scope"] == "idle"
    assert summary.revision == 1


def test_show_summary_can_be_cancelled_during_local_inspection(
    monkeypatch,
    tmp_path,
):
    engine, _ = _attached_engine(tmp_path, FakeProvider())
    entered = threading.Event()
    release = threading.Event()
    original = engine.registry.analysis_summary

    def blocked_summary(record):
        entered.set()
        assert release.wait(2.0)
        return original(record)

    monkeypatch.setattr(engine.registry, "analysis_summary", blocked_summary)
    result: list[tuple] = []
    thread = threading.Thread(
        target=lambda: result.append(engine.show_analysis_summary()),
    )

    thread.start()
    assert entered.wait(2.0)
    engine.cancel_active_operation()
    release.set()
    thread.join(2.0)

    assert not thread.is_alive()
    assert any(
        event.event == EngineEventType.OPERATION_CANCELLED
        and event.data["scope"] == "inspection"
        for event in result[0]
    )


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
        return _text_response("done")

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


def test_reopened_engine_rejects_response_that_disagrees_with_manifest(
    tmp_path,
):
    engine = _ready_engine(tmp_path, FakeProvider())
    completed = engine.confirm_revision()
    run_id = next(
        event.data["run_id"]
        for event in completed
        if event.event == EngineEventType.RUN_COMPLETED
    )
    response_path = (
        engine.workspace
        / "sessions"
        / engine.session_id
        / "runs"
        / run_id
        / "logs"
        / "worker-response.json"
    )
    payload = json.loads(response_path.read_text(encoding="utf-8"))
    payload["elapsed_seconds"] += 1.0
    response_path.write_text(
        json.dumps(payload, ensure_ascii=False),
        encoding="utf-8",
    )

    reopened = AgentSessionEngine(
        engine.workspace,
        FakeProvider(),
        session_id=engine.session_id,
    )

    assert reopened.get_snapshot().active_run_id is None
    assert reopened.get_snapshot().phase == SessionPhase.CONFIRMED


def test_oversized_tool_call_batch_is_rejected_before_persistence(tmp_path):
    provider = FakeProvider(
        [
            _tool_response(
                *(
                    ToolCall(
                        f"too_many_{index}",
                        "show_capabilities",
                        {},
                    )
                    for index in range(13)
                )
            )
        ]
    )
    engine = AgentSessionEngine(
        tmp_path / "workspace",
        provider,
        session_id="ses_tool_batch",
        config=EngineConfig(max_tool_calls=12),
    )

    events = engine.send_message("列出能力。")

    assert any(
        event.event == EngineEventType.DIAGNOSTIC
        and event.data["diagnostic"]["code"] == "TOOL_LIMIT_EXCEEDED"
        for event in events
    )
    reopened = AgentSessionEngine(
        engine.workspace,
        FakeProvider([_text_response("会话仍可继续。")]),
        session_id=engine.session_id,
    )
    assert reopened.send_message("继续。")
