from __future__ import annotations

from threading import Event

import pytest
from PySide6.QtWidgets import QApplication

from fem.application import ModelSession, UnitContext
from fem.core.model import FEMModel
from fem.geometry import PlateWithHoleGeometry
from fem.mesh.settings import MeshSettings
from fem_agent.authoring import (
    AuthoringAuthorizationError,
    ProposalState,
)
from fem_agent.mesh_authoring import MeshIntent, create_mesh_proposal
from fem_gui.agent_authoring import (
    AgentAuthoringBridge,
    AgentMeshTaskRequest,
    SessionGeometryAuthoringPort,
    authoring_context_from_snapshot,
)
from fem_gui.main_window import FEMMainWindow
from fem_gui.task_controller import (
    BackgroundTaskState,
    TaskApplyStatus,
    TaskCompletion,
)
from fem_gui.workers import TaskContext
from tests.helpers.mesh_builders import make_selection_quad_mesh


def _recipe() -> PlateWithHoleGeometry:
    return PlateWithHoleGeometry(
        "实体-偏心孔板",
        10.0,
        6.0,
        6.5,
        2.0,
        1.0,
    )


def _model(name: str) -> FEMModel:
    return FEMModel(make_selection_quad_mesh(), name=name)


def _session() -> ModelSession:
    session = ModelSession()
    session.create_native_project_with_first_part(
        "模型-偏心孔板",
        UnitContext("mm", "N", "MPa"),
        _recipe(),
        part_name="部件-偏心孔板",
    )
    return session


def _proposal(session: ModelSession, proposal_id: str):
    return create_mesh_proposal(
        proposal_id=proposal_id,
        agent_session_id="agent-session-a3",
        turn_id=f"turn-{proposal_id}",
        source_tool_call_ids=(f"call-{proposal_id}",),
        context=authoring_context_from_snapshot(session.snapshot()),
        draft_revision=1,
        part_id="P1",
        mesh_intent=MeshIntent(
            "quadrilateral",
            1,
            global_size=0.5,
        ),
    )


def test_a3_bridge_calls_no_mesh_work_before_gui_start_and_reject_is_noop() -> (
    None
):
    session = _session()
    requests: list[AgentMeshTaskRequest] = []
    port = SessionGeometryAuthoringPort(
        session,
        lambda: None,
        lambda request: requests.append(request) is None,
    )
    bridge = AgentAuthoringBridge(port)
    bridge.bind_snapshot(session.snapshot())
    proposal = _proposal(session, "proposal-reject")
    before = session.snapshot()

    bridge.register_proposal(proposal)
    assert requests == []
    with pytest.raises(AuthoringAuthorizationError):
        bridge.accept_proposal(proposal.proposal_id)
    rejected = bridge.reject_from_gui_control(proposal.proposal_id)

    assert rejected.state is ProposalState.REJECTED
    assert requests == []
    assert session.snapshot() == before


@pytest.mark.parametrize(
    "starter",
    [
        pytest.param(lambda _request: False, id="controller-busy"),
        pytest.param(
            lambda _request: (_ for _ in ()).throw(RuntimeError("start failed")),
            id="starter-error",
        ),
    ],
)
def test_a3_mesh_start_failure_consumes_token_without_session_change(
    starter,
) -> None:
    session = _session()
    port = SessionGeometryAuthoringPort(session, lambda: None, starter)
    bridge = AgentAuthoringBridge(port)
    bridge.bind_snapshot(session.snapshot())
    proposal = _proposal(session, "proposal-start-failed")
    bridge.register_proposal(proposal)
    before = session.snapshot()

    receipt = bridge.accept_from_gui_control(proposal.proposal_id)

    assert receipt.state is ProposalState.FAILED
    assert session.snapshot() == before


def test_a3_running_failure_cancel_and_stale_keep_current_model() -> None:
    for terminal in (
        ProposalState.FAILED,
        ProposalState.CANCELLED,
    ):
        session = _session()
        requests: list[AgentMeshTaskRequest] = []
        port = SessionGeometryAuthoringPort(
            session,
            lambda: None,
            lambda request: requests.append(request) is None,
        )
        bridge = AgentAuthoringBridge(port)
        bridge.bind_snapshot(session.snapshot())
        proposal = _proposal(session, f"proposal-{terminal.value}")
        bridge.register_proposal(proposal)
        before = session.snapshot()

        running = bridge.accept_from_gui_control(proposal.proposal_id)
        assert running.state is ProposalState.RUNNING
        assert session.snapshot() == before
        port.terminate_mesh(proposal.proposal_id, terminal, terminal.value)

        assert bridge.state(proposal.proposal_id) is terminal
        assert session.snapshot() == before
        assert session.validate_task_token(
            requests[0].task.token
        ).value == "already_completed"

    session = _session()
    requests = []
    port = SessionGeometryAuthoringPort(
        session,
        lambda: None,
        lambda request: requests.append(request) is None,
    )
    bridge = AgentAuthoringBridge(port)
    bridge.bind_snapshot(session.snapshot())
    proposal = _proposal(session, "proposal-stale")
    bridge.register_proposal(proposal)
    bridge.accept_from_gui_control(proposal.proposal_id)
    session.rename_native_model("模型-用户修改")
    current = session.snapshot()

    delta = port.accept_mesh_result(proposal.proposal_id, _model("晚到网格"))
    assert delta.accepted is False
    port.terminate_mesh(
        proposal.proposal_id,
        ProposalState.STALE,
        "revision changed",
    )
    assert bridge.state(proposal.proposal_id) is ProposalState.STALE
    assert session.snapshot() == current


def test_a3_port_success_calls_atomic_session_accept() -> None:
    session = _session()
    requests: list[AgentMeshTaskRequest] = []
    refreshes: list[int] = []
    port = SessionGeometryAuthoringPort(
        session,
        lambda: refreshes.append(session.session_revision),
        lambda request: requests.append(request) is None,
    )
    bridge = AgentAuthoringBridge(port)
    bridge.bind_snapshot(session.snapshot())
    proposal = _proposal(session, "proposal-success")
    bridge.register_proposal(proposal)
    before = session.snapshot()

    running = bridge.accept_from_gui_control(proposal.proposal_id)
    delta = port.accept_mesh_result(proposal.proposal_id, _model("新网格"))
    after = session.snapshot()

    assert running.state is ProposalState.RUNNING
    assert delta.accepted
    assert bridge.state(proposal.proposal_id) is ProposalState.SUCCEEDED
    assert after.session_revision == before.session_revision + 1
    assert after.parts[0].mesh_settings == MeshSettings(
        0.5,
        cell_shape="quadrilateral",
        strict_cell_shape=True,
    )
    assert after.artifact is not None
    assert after.artifact.model.name == "新网格"
    # A3 projection is owned by the task controller after CAS acceptance.
    assert refreshes == []


def test_a3_main_window_success_projects_exactly_once(
    monkeypatch,
) -> None:
    _application = QApplication.instance() or QApplication([])
    window = FEMMainWindow()
    window.session.create_native_project_with_first_part(
        "模型-偏心孔板",
        UnitContext("mm", "N", "MPa"),
        _recipe(),
        part_name="部件-偏心孔板",
    )
    window.document = window.session.projection_snapshot()
    window.agent_authoring_bridge.bind_snapshot(window.document)
    proposal = _proposal(window.session, "proposal-window")
    window.agent_authoring_bridge.register_proposal(proposal)
    candidate = _model("窗口网格")
    projections: list[int] = []

    monkeypatch.setattr(
        "fem_gui.main_window.generate_fem_model",
        lambda *_args, **_kwargs: candidate,
    )
    monkeypatch.setattr(
        "fem_gui.main_window.build_model_geometry",
        lambda _model: object(),
    )

    def apply_delta(delta, **_kwargs):
        projections.append(delta.session_revision)
        window.document = window.session.projection_snapshot()
        return True

    monkeypatch.setattr(window, "_apply_session_delta", apply_delta)

    def synchronous_start(
        workload,
        *,
        task_name,
        apply_result,
        project_result,
        on_terminal,
        **_kwargs,
    ):
        context = TaskContext(1, Event(), lambda *_args: None)
        value = workload(context)
        outcome = apply_result(value)
        assert outcome.status is TaskApplyStatus.ACCEPTED
        project_result(outcome.projection_value)
        on_terminal(
            TaskCompletion(
                1,
                task_name,
                BackgroundTaskState.SUCCEEDED,
                apply_status=TaskApplyStatus.ACCEPTED,
            )
        )
        return 1

    monkeypatch.setattr(window.task_controller, "start", synchronous_start)

    receipt = window.agent_authoring_bridge.accept_from_gui_control(
        proposal.proposal_id
    )

    assert receipt.state is ProposalState.SUCCEEDED
    assert projections == [window.session.session_revision]
    assert window.session.snapshot().artifact.model.name == "窗口网格"
