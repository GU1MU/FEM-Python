from __future__ import annotations

from pathlib import Path

import pytest

from fem.application import ModelSession, UnitContext, run_static_preflight
from fem.geometry import PlateWithHoleGeometry
from fem.io.inp import read
from fem.mesh.settings import MeshSettings
from fem_agent.authoring import AuthoringContractError
from fem_agent.authoring_runtime import AuthoringWorkflowStage
from fem_agent.result_authoring import AgentResultQueryBridge
from fem_agent.solve_authoring import create_solve_proposal
from fem_agent.tools.registry import ToolExecutionContext
from fem_gui.agent_authoring import (
    AgentAuthoringBridge,
    SessionGeometryAuthoringPort,
    SessionResultQueryPort,
    create_session_authoring_workflow_controller,
)

from tests.helpers.agent_session_fixtures import make_defined_plate_session


_FIXTURE = (
    Path(__file__).parents[2]
    / "helpers" / "fixtures"
    / "inp"
    / "abaqus_standard"
    / "truss2_tension.inp"
)
STEP_NAME = "Tension"

def test_context_read_restores_stale_workspace_binding() -> None:
    session = make_defined_plate_session()
    bridge = AgentAuthoringBridge(
        SessionGeometryAuthoringPort(session, lambda: None)
    )
    bridge.bind_snapshot(session.snapshot(), document_id=2)
    assert bridge.context.binding.document_id == "2"
    controller = create_session_authoring_workflow_controller(
        session,
        bridge,
        AgentResultQueryBridge(SessionResultQueryPort(session)),
    )
    controller.invalidate_binding("document is read-only")
    assert controller.stage is AuthoringWorkflowStage.STALE

    outcome = controller.dispatch(
        "read_authoring_context",
        {},
        ToolExecutionContext(
            session.session_id,
            session.session_revision,
            "rebind",
        ),
    )

    assert outcome.ok
    assert controller.stage is not AuthoringWorkflowStage.STALE


def _validated_solve_session() -> ModelSession:
    model = read(_FIXTURE)
    session = ModelSession()
    session.create_native_project_with_first_part(
        "模型-桁架",
        UnitContext("mm", "N", "MPa"),
        PlateWithHoleGeometry("实体-偏心孔板", 10.0, 6.0, 6.5, 2.0, 1.0),
        part_name="部件-桁架",
    )
    session.replace_part_mesh_settings(
        "P1",
        MeshSettings(1.0, cell_shape="triangle"),
    )
    mesh = session.prepare_mesh_generation()
    assert session.accept_generated_model(mesh.token, model).accepted
    task = session.prepare_validation(STEP_NAME)
    report = run_static_preflight(task.model, task.step_name, token=task.token)
    assert report.passed
    assert session.accept_validation(task.token, report).accepted
    return session


def _solve_proposal(session: ModelSession, *, target_document_id: str | None):
    return create_solve_proposal(
        proposal_id=f"proposal-binding-{target_document_id}",
        agent_session_id="agent-binding",
        turn_id="turn-binding-solve",
        source_tool_call_ids=("call-binding-solve",),
        snapshot=session.snapshot(),
        draft_revision=6,
        step_name=STEP_NAME,
        job_name="作业-静力1",
        target_document_id=target_document_id,
    )


def test_solve_acceptance_rejects_mismatched_document_target() -> None:
    session = _validated_solve_session()
    bridge = AgentAuthoringBridge(
        SessionGeometryAuthoringPort(session, lambda: None)
    )
    bridge.bind_snapshot(session.snapshot(), document_id=2)
    matching = _solve_proposal(session, target_document_id="2")
    mismatched = _solve_proposal(session, target_document_id="999")

    before = session.snapshot()
    assert bridge.register_proposal(matching).state.name == "PENDING_CONFIRMATION"
    with pytest.raises(AuthoringContractError, match="target is stale"):
        bridge.register_proposal(mismatched)
    assert session.snapshot() == before
