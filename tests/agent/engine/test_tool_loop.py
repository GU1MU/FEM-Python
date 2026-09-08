import pytest

from fem_agent.artifacts import ArtifactStore
from fem_agent.diagnostics import DiagnosticCode
from fem_agent.engine import AgentSessionEngine, EngineConfig, EngineEventType
from fem_agent.providers.base import (
    AssistantMessage,
    ProviderConfig,
    ProviderResponse,
    ToolCall,
)
from fem_agent.providers.deepseek import DeepSeekProvider
from fem_agent.providers.fake import FakeProvider
from fem_agent.schemas import SessionPhase
from fem_agent.worker import InspectionWorkerError

from tests.helpers.abaqus_builders import write_perforated_plate_style_inp
from tests.helpers.agent_engine_fixtures import _attached_engine
from tests.helpers.agent_engine_registry_fixtures import (
    _GeometryEditWithCatalogToolRegistry,
    _RetryingGeometryEditWithCatalogToolRegistry,
)
from tests.helpers.agent_provider_fixtures import tool_response, text_response


pytestmark = pytest.mark.integration


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
            text_response("读取完成。"),
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
            text_response(f"{process}\n\n{decision}"),
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
            tool_response(
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
            tool_response(
                ToolCall("call_summary", "get_analysis_summary", {})
            ),
            text_response("分析摘要已准备好，请检查后输入 /confirm。"),
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
        return text_response("模型检查进程暂时失败，请重试。")

    provider = FakeProvider(
        [
            tool_response(
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
        tool_response(
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
            tool_response(call),
            tool_response(call),
            text_response("单位已记录。"),
        ]
    )
    engine, _ = _attached_engine(tmp_path, provider)

    engine.send_message("记录单位。")

    assert engine.get_snapshot().revision == 2
    assert len(engine.revisions.list_records(engine.session_id)) == 2


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
            tool_response(call),
            text_response("第一次记录完成。"),
            tool_response(call),
            text_response("第二次记录完成。"),
        ]
    )
    engine, _ = _attached_engine(tmp_path, provider)

    engine.send_message("记录单位。")
    first_revision = engine.get_snapshot().revision
    engine.send_message("再次确认同一单位。")

    assert first_revision == 2
    assert engine.get_snapshot().revision == 3


def test_oversized_tool_call_batch_is_rejected_before_persistence(tmp_path):
    provider = FakeProvider(
        [
            tool_response(
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
        FakeProvider([text_response("会话仍可继续。")]),
        session_id=engine.session_id,
    )
    assert reopened.send_message("继续。")
