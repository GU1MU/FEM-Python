import json
import threading

import pytest

import fem_agent.worker as worker_module
from fem_agent.artifacts import atomic_write_json
from fem_agent.diagnostics import DiagnosticCode
from fem_agent.engine import AgentSessionEngine, EngineEventType
from fem_agent.providers.base import ToolCall
from fem_agent.providers.fake import FakeProvider
from fem_agent.schemas import RunStatus, SessionPhase
from fem_agent.tools.registry import ToolExecutionContext
from fem_agent.worker import WorkerResponse, WorkerResponseIntegrityError

from tests.helpers.agent_engine_fixtures import (
    InProcessFEMInspector,
    _attached_engine,
    _ready_engine,
)
from tests.helpers.agent_provider_fixtures import tool_response, text_response


@pytest.fixture(autouse=True)
def in_process_worker_transport(monkeypatch):
    """Exercise engine state and real FEM work without child startup overhead.

    Process isolation, deadlines, and wire transport remain covered by
    test_worker_subprocess.py and test_e2e.py.
    """
    def query(self, response, queries, *, timeout_seconds=None, cancel_event=None):
        request = worker_module.ResultQueryRequest(
            session_id=response.session_id,
            revision=response.revision,
            revision_hash=response.revision_hash,
            run_id=response.run_id,
            queries=tuple(queries),
        )
        try:
            return worker_module.execute_result_query_request(
                self.artifacts.root, request,
            )
        except Exception as error:
            raise worker_module.ResultQueryWorkerError(str(error)) from error

    def launch(
        self, request, record, run, request_path, response_path, timeout, cancel_event,
    ):
        response = worker_module.execute_worker_request(self.artifacts.root, request)
        atomic_write_json(response_path, response.to_dict())
        return worker_module.load_verified_worker_response(
            self.artifacts, record, run.run_id,
        )

    monkeypatch.setattr(
        worker_module.IsolatedFEMInspector, "inspect", InProcessFEMInspector.inspect,
    )
    monkeypatch.setattr(worker_module.IsolatedFEMResultQuerier, "query", query)
    monkeypatch.setattr(worker_module.IsolatedFEMWorker, "_launch", launch)


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
        return text_response(
            f"自由端最大位移为 {scalar['value']:.6g} mm，"
            f"位于节点 {scalar['node_id']}。"
        )

    provider.queue(
        tool_response(
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


def test_reopened_solved_session_can_query_saved_solution(tmp_path):
    engine = _ready_engine(tmp_path, FakeProvider())
    engine.confirm_revision()
    session_id = engine.session_id
    run_id = engine.get_snapshot().active_run_id
    provider = FakeProvider(
        [
            tool_response(
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
            text_response("已从保存的解中分析自由端位移。"),
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
