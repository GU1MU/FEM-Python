import json
import threading
from dataclasses import replace

import pytest

from fem_agent.artifacts import ArtifactStore
from fem_agent.diagnostics import DiagnosticCode
from fem_agent.engine import (
    AgentSessionEngine,
    EngineConfig,
    EngineEventType,
)
from fem_agent.providers.base import (
    AssistantMessage,
    ProviderConfig,
    ProviderResponse,
    ToolCall,
)
from fem_agent.providers.deepseek import DeepSeekProvider
from fem_agent.providers.fake import FakeProvider
from fem_agent.schemas import RunStatus, SessionPhase
from fem_agent.tools.registry import ToolExecutionContext
from fem_agent.worker import (
    InspectionWorkerError,
    WorkerResponse,
    WorkerResponseIntegrityError,
)
from tests.helpers.abaqus_builders import write_perforated_plate_style_inp


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
    assert "local deterministic fem package" not in system_prompt.casefold()


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
        assert release_worker.wait(5.0)
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
    assert progress_seen.wait(5.0)
    assert thread.is_alive()
    release_worker.set()
    thread.join(5.0)
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
        assert release.wait(5.0)
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
    assert entered.wait(5.0)
    cancelled = engine.cancel_active_operation()
    release.set()
    thread.join(5.0)

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
        assert cancel_event.wait(5.0)
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
    assert entered.wait(5.0)
    engine.cancel_active_operation()
    thread.join(5.0)

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
        assert release.wait(5.0)
        return original(record)

    monkeypatch.setattr(engine.registry, "analysis_summary", blocked_summary)
    result: list[tuple] = []
    thread = threading.Thread(
        target=lambda: result.append(engine.show_analysis_summary()),
    )

    thread.start()
    assert entered.wait(5.0)
    engine.cancel_active_operation()
    release.set()
    thread.join(5.0)

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
        assert release.wait(5.0)
        return _text_response("done")

    monkeypatch.setattr(provider, "complete", complete)
    original_session = engine.session_id
    thread = threading.Thread(target=lambda: engine.send_message("hello"))

    thread.start()
    assert entered.wait(5.0)
    rejected = engine.create_session()
    release.set()
    thread.join(5.0)

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
