from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path
from time import monotonic

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import QThread
from PySide6.QtWidgets import QApplication

from fem.io.inp import read
from fem.application import (
    ModelSession,
    PreflightDiagnostic,
    PreflightSeverity,
    PreflightStage,
    RunStatus,
    UnitContext,
    ValidationRecord,
    run_static_preflight,
)
from fem.application.results import build_solve_result_bundle
from fem.geometry import PlateWithHoleGeometry
from fem.mesh.settings import MeshSettings
from fem.solvers import static_linear
from fem_agent.authoring import (
    AgentProposal,
    AuthoringAuthorizationError,
    ModelOperation,
    OperationKind,
    ProposalKind,
    ProposalState,
)
from fem_agent.solve_authoring import (
    SolveValidationStamp,
    create_solve_proposal,
)
from fem_gui.agent_authoring import (
    AgentAuthoringBridge,
    AgentPreflightState,
    AgentSolveTaskRequest,
    SessionGeometryAuthoringPort,
)
from fem_gui.main_window import FEMMainWindow
from fem_gui.task_controller import (
    BackgroundTaskState,
    TaskApplyStatus,
    TaskCompletion,
)
from fem_gui.visualization.model_adapter import build_model_geometry


_FIXTURE = (
    Path(__file__).parents[1]
    / "fixtures"
    / "inp"
    / "abaqus_standard"
    / "truss2_tension.inp"
)
STEP_NAME = "Tension"


def _application() -> QApplication:
    return QApplication.instance() or QApplication([])


def _wait_for_task(window: FEMMainWindow) -> None:
    deadline = monotonic() + 2.0
    application = _application()
    while window.task_controller.busy and monotonic() < deadline:
        application.processEvents()
        QThread.msleep(1)
    application.processEvents()
    assert not window.task_controller.busy


def _recipe() -> PlateWithHoleGeometry:
    return PlateWithHoleGeometry(
        "实体-偏心孔板",
        10.0,
        6.0,
        6.5,
        2.0,
        1.0,
    )


def _seed_session(
    session: ModelSession,
    *,
    validated: bool,
) -> object:
    model = read(_FIXTURE)
    session.create_native_project_with_first_part(
        "模型-偏心孔板",
        UnitContext("mm", "N", "MPa"),
        _recipe(),
        part_name="部件-偏心孔板",
    )
    session.replace_part_mesh_settings(
        "P1",
        MeshSettings(1.0, cell_shape="triangle"),
    )
    mesh = session.prepare_mesh_generation()
    session.accept_generated_model(mesh.token, model)
    if validated:
        task = session.prepare_validation(STEP_NAME)
        report = run_static_preflight(
            task.model,
            task.step_name,
            token=task.token,
        )
        assert report.passed
        session.accept_validation(task.token, report)
    return model


def _seed_window(
    window: FEMMainWindow,
    *,
    validated: bool,
) -> None:
    _seed_session(window.session, validated=validated)
    window.document = window.session.projection_snapshot()
    window._applied_session_revision = window.document.session_revision
    window.geometry = build_model_geometry(window.document.model)
    window._current_step_name = STEP_NAME
    window.agent_authoring_bridge.bind_snapshot(window.document)


def _proposal(
    session: ModelSession,
    proposal_id: str = "proposal-a6",
    *,
    job_name: str = "作业-静力1",
) -> AgentProposal:
    return create_solve_proposal(
        proposal_id=proposal_id,
        agent_session_id="agent-session-a6",
        turn_id=f"turn-{proposal_id}",
        source_tool_call_ids=(f"call-{proposal_id}",),
        snapshot=session.snapshot(),
        draft_revision=6,
        step_name=STEP_NAME,
        job_name=job_name,
    )


def test_a6_agent_automatically_preflights_then_gui_click_solves_in_background(
    monkeypatch,
) -> None:
    _application()
    window = FEMMainWindow()
    _seed_window(window, validated=False)
    solve_threads: list[bool] = []
    original_solve = static_linear.solve

    def tracked_solve(*args, **kwargs):
        solve_threads.append(QThread.currentThread() is window.thread())
        return original_solve(*args, **kwargs)

    monkeypatch.setattr(static_linear, "solve", tracked_solve)

    preflight = window.agent_authoring_bridge.request_preflight(STEP_NAME)

    assert preflight.state is AgentPreflightState.RUNNING
    assert window.session.validation_for(STEP_NAME) is None
    _wait_for_task(window)
    completed_preflight = (
        window.agent_authoring_bridge.port.preflight_record(
            preflight.request_id
        )
    )
    assert completed_preflight.state is AgentPreflightState.PASSED
    assert completed_preflight.validation_stamp is not None

    proposal = _proposal(window.session)
    window.agent_authoring_bridge.register_proposal(proposal)
    receipt = window.agent_authoring_bridge.accept_from_gui_control(
        proposal.proposal_id
    )
    assert receipt.state in {ProposalState.RUNNING, ProposalState.SUCCEEDED}
    _wait_for_task(window)

    assert (
        window.agent_authoring_bridge.state(proposal.proposal_id)
        is ProposalState.SUCCEEDED
    )
    run = window.session.find_run("作业-静力1")
    assert run is not None and run.status is RunStatus.SUCCEEDED
    provenance = window.session.result_provenance_for(run.run_id)
    assert provenance is not None
    assert provenance.artifact_id == proposal.operations[0].parameters[
        "artifact_id"
    ]
    assert provenance.model_revision == proposal.operations[0].parameters[
        "model_revision"
    ]
    assert provenance.run_id == run.run_id
    assert solve_threads == [False]
    window.close()


def test_a6_only_gui_control_can_start_once_and_terminal_mapping_is_exact() -> None:
    session = ModelSession()
    _seed_session(session, validated=True)
    requests: list[AgentSolveTaskRequest] = []
    port = SessionGeometryAuthoringPort(
        session,
        lambda: None,
        start_solve_task=lambda request: requests.append(request) is None,
    )
    bridge = AgentAuthoringBridge(port)
    bridge.bind_snapshot(session.snapshot())
    proposal = _proposal(session)
    bridge.register_proposal(proposal)

    with pytest.raises(AuthoringAuthorizationError):
        bridge.accept_proposal(proposal.proposal_id)
    running = bridge.accept_from_gui_control(proposal.proposal_id)
    assert running.state is ProposalState.RUNNING
    assert len(requests) == 1
    with pytest.raises(AuthoringAuthorizationError):
        bridge.accept_from_gui_control(proposal.proposal_id)
    assert len(requests) == 1

    port.progress_solve(proposal.proposal_id, "正在求解")
    port.complete_solve(
        proposal.proposal_id,
        ProposalState.CANCELLED,
        "cancelled",
    )
    assert bridge.state(proposal.proposal_id) is ProposalState.CANCELLED


def test_a6_cancelled_run_rejects_late_solver_result() -> None:
    session = ModelSession()
    _seed_session(session, validated=True)
    solve_tasks = []

    def start(request: AgentSolveTaskRequest) -> bool:
        task = session.prepare_solve(
            request.step_name,
            request.job_name,
            expected_session_revision=request.base_session_revision,
        )
        session.begin_run(task.token)
        solve_tasks.append(task)
        return True

    port = SessionGeometryAuthoringPort(
        session,
        lambda: None,
        start_solve_task=start,
    )
    bridge = AgentAuthoringBridge(port)
    bridge.bind_snapshot(session.snapshot())
    proposal = _proposal(session, "proposal-cancel-late")
    bridge.register_proposal(proposal)
    bridge.accept_from_gui_control(proposal.proposal_id)
    task = solve_tasks[0]
    result = static_linear.solve(
        task.model,
        task.step_name,
        name=task.run_name,
    )
    bundle = build_solve_result_bundle(task, result)

    cancelled = session.accept_run_cancelled(task.token)
    port.complete_solve(
        proposal.proposal_id,
        ProposalState.CANCELLED,
        "cancelled",
    )
    late = session.accept_run_succeeded(task.token, bundle)

    assert cancelled.accepted
    assert not late.accepted
    assert session.result_provenance_for(task.run_id) is None
    assert bridge.state(proposal.proposal_id) is ProposalState.CANCELLED


def test_a6_revision_and_validation_stamp_changes_disable_old_proposals() -> None:
    session = ModelSession()
    _seed_session(session, validated=True)
    requests: list[AgentSolveTaskRequest] = []
    port = SessionGeometryAuthoringPort(
        session,
        lambda: None,
        start_solve_task=lambda request: requests.append(request) is None,
    )
    bridge = AgentAuthoringBridge(port)
    bridge.bind_snapshot(session.snapshot())

    revision_proposal = _proposal(session, "proposal-revision")
    bridge.register_proposal(revision_proposal)
    session.rename_native_model("模型-用户修改")
    bridge.bind_snapshot(session.snapshot())
    assert (
        bridge.state(revision_proposal.proposal_id)
        is ProposalState.STALE
    )
    assert not bridge.can_accept_from_gui_control(
        revision_proposal.proposal_id
    )

    fresh = _proposal(
        session,
        "proposal-stamp",
        job_name="作业-静力2",
    )
    current_record = session.validation_for(STEP_NAME)
    assert current_record is not None
    stale_report = replace(
        current_record.report,
        diagnostics=(
            PreflightDiagnostic(
                code="test.changed-warning",
                severity=PreflightSeverity.WARNING,
                stage=PreflightStage.OUTPUT,
                message="changed without a session mutation",
            ),
        ),
    )
    stale_stamp = SolveValidationStamp.from_record(
        ValidationRecord(current_record.stamp, stale_report)
    )
    parameters = dict(fresh.operations[0].parameters)
    parameters["validation_stamp"] = stale_stamp.to_dict()
    stale_operation = ModelOperation(
        OperationKind.REQUEST_SOLVE,
        parameters,
    )
    stale = AgentProposal.create(
        proposal_id="proposal-stale-stamp",
        proposal_kind=ProposalKind.SOLVE,
        agent_session_id=fresh.agent_session_id,
        turn_id="turn-stale-stamp",
        source_tool_call_ids=("call-stale-stamp",),
        target_document_id=fresh.target_document_id,
        target_session_id=fresh.target_session_id,
        base_session_revision=fresh.base_session_revision,
        draft_revision=fresh.draft_revision,
        operations=(stale_operation,),
        preconditions=fresh.preconditions,
        expected_changes=fresh.expected_changes,
        invalidation_impact=fresh.invalidation_impact,
        display_summary=fresh.display_summary,
    )
    bridge.register_proposal(stale)

    assert not bridge.can_accept_from_gui_control(stale.proposal_id)
    failed = bridge.accept_from_gui_control(stale.proposal_id)
    assert failed.state is ProposalState.FAILED
    assert requests == []


def test_a6_port_rejects_legacy_solve_proposal_shape() -> None:
    session = ModelSession()
    _seed_session(session, validated=True)
    port = SessionGeometryAuthoringPort(
        session,
        lambda: None,
        start_solve_task=lambda _request: True,
    )
    bridge = AgentAuthoringBridge(port)
    bridge.bind_snapshot(session.snapshot())
    legacy = AgentProposal.create(
        proposal_id="proposal-legacy",
        proposal_kind=ProposalKind.SOLVE,
        agent_session_id="agent-session-a1",
        turn_id="turn-legacy",
        source_tool_call_ids=("call-legacy",),
        target_document_id=f"document:{session.session_id}",
        target_session_id=session.session_id,
        base_session_revision=session.session_revision,
        draft_revision=1,
        operations=(
            ModelOperation(
                OperationKind.REQUEST_SOLVE,
                {
                    "step_name": STEP_NAME,
                    "validation_stamp": "legacy",
                },
            ),
        ),
        preconditions={"authoring_phase": "A6"},
        expected_changes={},
        invalidation_impact={},
        display_summary={"title": "legacy"},
    )

    with pytest.raises(ValueError, match="exact schema"):
        bridge.register_proposal(legacy)


def test_a6_busy_rejects_before_run_and_start_failure_terminalizes_created_run(
    monkeypatch,
    dispose_gui_widget,
) -> None:
    _application()
    busy_window = FEMMainWindow()
    _seed_window(busy_window, validated=True)
    busy_proposal = _proposal(busy_window.session, "proposal-busy")
    busy_window.agent_authoring_bridge.register_proposal(busy_proposal)
    busy_window.task_controller._active = object()
    try:
        failed = busy_window.agent_authoring_bridge.accept_from_gui_control(
            busy_proposal.proposal_id
        )
    finally:
        busy_window.task_controller._active = None
    assert failed.state is ProposalState.FAILED
    assert busy_window.session.snapshot().runs == ()
    dispose_gui_widget(busy_window)

    start_window = FEMMainWindow()
    _seed_window(start_window, validated=True)
    start_proposal = _proposal(
        start_window.session,
        "proposal-start-false",
    )
    start_window.agent_authoring_bridge.register_proposal(start_proposal)
    monkeypatch.setattr(start_window, "_start_task", lambda *_args, **_kwargs: False)

    failed = start_window.agent_authoring_bridge.accept_from_gui_control(
        start_proposal.proposal_id
    )

    assert failed.state is ProposalState.FAILED
    runs = start_window.session.snapshot().runs
    assert len(runs) == 1
    assert runs[0].status is RunStatus.FAILED
    start_window.close()


@pytest.mark.parametrize(
    ("background_state", "proposal_state"),
    [
        (BackgroundTaskState.FAILED, ProposalState.FAILED),
        (BackgroundTaskState.CANCELLED, ProposalState.CANCELLED),
        (BackgroundTaskState.DISCARDED, ProposalState.STALE),
    ],
)
def test_a6_existing_job_terminal_states_map_to_proposal(
    monkeypatch,
    background_state: BackgroundTaskState,
    proposal_state: ProposalState,
) -> None:
    _application()
    window = FEMMainWindow()
    _seed_window(window, validated=True)
    proposal = _proposal(window.session, f"proposal-{proposal_state.value}")
    window.agent_authoring_bridge.register_proposal(proposal)
    completions = []

    def fake_begin(*_args, completion, **_kwargs):
        completions.append(completion)
        return object()

    monkeypatch.setattr(window, "_begin_submit_run", fake_begin)
    running = window.agent_authoring_bridge.accept_from_gui_control(
        proposal.proposal_id
    )
    assert running.state is ProposalState.RUNNING

    completions[0].complete(
        TaskCompletion(
            1,
            "Agent solve",
            background_state,
            message=background_state.value,
            apply_status=(
                None
                if background_state
                in {
                    BackgroundTaskState.FAILED,
                    BackgroundTaskState.CANCELLED,
                }
                else TaskApplyStatus.STALE
            ),
        )
    )

    assert (
        window.agent_authoring_bridge.state(proposal.proposal_id)
        is proposal_state
    )
    window.close()
