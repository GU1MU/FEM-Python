import threading

import pytest

from fem_agent.artifacts import ArtifactStore
from fem_agent.engine import AgentSessionEngine, EngineEventType
from fem_agent.providers.base import ToolCall
from fem_agent.providers.fake import FakeProvider
from fem_agent.schemas import SessionPhase
from fem_agent.tools.registry import ToolExecutionContext
from fem_agent.worker import InspectionWorkerError

from tests.helpers.abaqus_builders import write_perforated_plate_style_inp
from tests.helpers.agent_engine_fixtures import _attached_engine, _ready_engine
from tests.helpers.agent_provider_fixtures import tool_response, text_response


pytestmark = pytest.mark.integration


def test_natural_language_and_model_tool_call_cannot_bypass_confirm(tmp_path):
    provider = FakeProvider(
        [
            tool_response(
                ToolCall("call_solve", "solve_confirmed_analysis", {})
            ),
            text_response("请在本地输入 /confirm。"),
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
