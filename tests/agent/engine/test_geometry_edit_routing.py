import pytest

from fem_agent.engine import AgentSessionEngine, EngineEventType
from fem_agent.providers.base import ToolCall, ToolDefinition
from fem_agent.providers.fake import FakeProvider
from fem_agent.schemas import ToolResult

from tests.helpers.agent_engine_providers import (
    _tool_response,
    _text_response,
)
from tests.helpers.agent_engine_registry_fixtures import (
    _GeometryEditToolRegistry,
    _GeometryEditWithCatalogToolRegistry,
)


pytestmark = pytest.mark.integration


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
    assert any(
        event.event is EngineEventType.TOOL_COMPLETED
        and event.data["tool"] == "prepare_geometry_edit"
        and event.data["result"]["data"]["state"] == "pending_confirmation"
        for event in events
    )


def test_geometry_proposal_requires_current_edit_context(tmp_path):
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

    events = engine.send_message("在现有平板上增加一个矩形槽")

    assert [name for name, _arguments in tools.calls] == [
        "read_geometry_edit_context",
        "prepare_geometry_edit",
    ]
    completed = [
        event.data
        for event in events
        if event.event is EngineEventType.TOOL_COMPLETED
    ]
    assert all(item["call_id"] != "premature-prepare" for item in completed)
    assert completed[-1]["result"]["data"]["state"] == "pending_confirmation"


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
