import pytest

from fem_agent.authoring_runtime import AuthoringWorkflowController
from fem_agent.engine import AgentSessionEngine, EngineEventType
from fem_agent.providers.base import ToolCall, ToolDefinition
from fem_agent.providers.fake import FakeProvider
from fem_agent.schemas import ToolResult

from tests.helpers.agent_engine_providers import (
    _tool_response,
    _text_response,
)
from tests.helpers.agent_engine_registry_fixtures import (
    _AdditionalModelToolRegistry,
)


pytestmark = pytest.mark.integration


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


def test_authoring_prompt_drives_solve_preflight_and_card_chain(tmp_path):
    provider = FakeProvider([_text_response("需要界面确认。")])
    controller = AuthoringWorkflowController(lambda: {}, {})
    engine = AgentSessionEngine(
        tmp_path / "workspace",
        provider,
        session_id="ses_solve_chain",
        dynamic_tools=controller,
    )

    engine.send_message("提交求解")

    system_prompt = provider.requests[0].messages[0].content
    assert "drive the whole local" in system_prompt
    assert "call run_native_preflight for the current step" in system_prompt
    assert "never state that the preflight passed" in system_prompt
    assert "prepare_solve_proposal in that same flow" in system_prompt
    assert "no confirmable control" in system_prompt
    assert "before that card has been presented" in system_prompt


def test_authoring_prompt_forbids_preflight_result_polling(tmp_path):
    provider = FakeProvider([_text_response("等待预检。")])
    controller = AuthoringWorkflowController(lambda: {}, {})
    engine = AgentSessionEngine(
        tmp_path / "workspace",
        provider,
        session_id="ses_solve_polling",
        dynamic_tools=controller,
    )

    engine.send_message("提交求解")

    system_prompt = provider.requests[0].messages[0].content
    # run_native_preflight waits locally and returns the terminal state, so
    # re-reading the authoring context in a polling loop must be forbidden:
    # it previously exhausted the bounded tool-call budget while the GUI
    # background check was still running.
    assert "exactly once" in system_prompt
    assert "returns its terminal state" in system_prompt
    assert "not re-read the authoring context to wait for it" in system_prompt
    assert "do not poll read_authoring_context in a loop" in system_prompt
    assert "read it at most" in system_prompt
    assert "end the turn" in system_prompt


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
