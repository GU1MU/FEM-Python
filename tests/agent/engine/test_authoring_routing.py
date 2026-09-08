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

    provider_snapshot = None

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
    deltas = [
        event.data["text"]
        for event in events
        if event.event is EngineEventType.MESSAGE_DELTA
    ]
    assert len(deltas) == 1
    assert "默认单位制 mm-N-MPa" in deltas[0]
    assert "几何建模工具" in deltas[0]


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
    assert [name for name, _arguments in tools.calls] == [
        "create_native_model_document",
        "read_authoring_context",
        "prepare_planar_construction_proposal",
    ]


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
    assert [name for name, _arguments in tools.calls] == [
        "set_authoring_requirements",
        "prepare_mesh_proposal",
    ]
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
    assert [name for name, _arguments in tools.calls] == [
        "set_authoring_requirements",
    ]
