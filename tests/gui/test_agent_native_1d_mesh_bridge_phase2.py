from __future__ import annotations

import pytest

from fem.application import ModelSession, UnitContext
from fem.core.model import FEMModel
from fem.geometry.recipes import WireGeometry, WireMember, WirePoint
from fem.mesh.settings import MeshSettings
from fem_agent.authoring import ProposalState
from fem_agent.authoring_runtime import AuthoringWorkflowStage
from fem_agent.mesh_authoring import MeshIntent, create_mesh_proposal
from fem_agent.result_authoring import AgentResultQueryBridge
from fem_agent.tools.registry import ToolExecutionContext
from fem_gui.agent_authoring import (
    AgentAuthoringBridge,
    AgentMeshTaskRequest,
    SessionGeometryAuthoringPort,
    SessionResultQueryPort,
    authoring_context_from_snapshot,
    create_session_authoring_workflow_controller,
)
from tests.helpers.mesh_builders import make_dof_order_meshes


def _wire() -> WireGeometry:
    return WireGeometry(
        "Bar",
        (
            WirePoint("Root", 0.0, 0.0, 0.0),
            WirePoint("Tip", 1.0, 0.0, 0.0),
        ),
        (WireMember("Bar", "Root", "Tip"),),
    )


def _session() -> ModelSession:
    session = ModelSession()
    session.create_native_project_with_first_part(
        "Line model",
        UnitContext("mm", "N", "MPa"),
        _wire(),
        part_name="Wire",
    )
    return session


def _model(line_element_type: str) -> FEMModel:
    index = 0 if line_element_type == "Truss2" else 1
    return FEMModel(
        make_dof_order_meshes()[index],
        name=f"{line_element_type} model",
    )


def _proposal(
    session: ModelSession,
    proposal_id: str,
    line_element_type: str,
):
    return create_mesh_proposal(
        proposal_id=proposal_id,
        agent_session_id="agent-line-phase2",
        turn_id=f"turn-{proposal_id}",
        source_tool_call_ids=(f"call-{proposal_id}",),
        context=authoring_context_from_snapshot(session.snapshot()),
        draft_revision=1,
        part_id="P1",
        mesh_intent=MeshIntent(
            "line",
            1,
            global_size=0.25,
            line_element_type=line_element_type,
        ),
    )


def test_session_controller_prepares_strict_line_proposal_without_mutation() -> (
    None
):
    session = _session()
    port = SessionGeometryAuthoringPort(session, lambda: None, lambda _request: True)
    bridge = AgentAuthoringBridge(port)
    before = session.snapshot()
    bridge.bind_snapshot(before)
    controller = create_session_authoring_workflow_controller(
        session,
        bridge,
        AgentResultQueryBridge(SessionResultQueryPort(session)),
    )

    assert controller.stage is AuthoringWorkflowStage.MESH_READY
    recorded = controller.dispatch(
        "set_authoring_requirements",
        {
            "turn_id": "turn-controller-line-phase2",
            "requirements": {
                "mesh_cell_shape": "line",
                "mesh_order": 1,
                "mesh_global_size": 0.25,
                "line_element_type": "Beam2",
            },
        },
        ToolExecutionContext(
            before.session_id,
            before.session_revision,
            "controller-line-requirements",
        ),
    )
    prepared = controller.dispatch(
        "prepare_mesh_proposal",
        {},
        ToolExecutionContext(
            before.session_id,
            before.session_revision,
            "controller-line-proposal",
        ),
    )

    assert recorded.ok
    assert prepared.ok
    assert prepared.data["state"] == ProposalState.PENDING_CONFIRMATION.value
    proposal_id = str(prepared.data["proposal_id"])
    proposal = bridge._records[proposal_id].proposal
    intent_payload = proposal.operations[0].parameters["mesh_intent"]
    assert intent_payload["schema_version"] == "1.1"
    assert intent_payload["cell_shape"] == "line"
    assert intent_payload["order"] == 1
    assert intent_payload["line_element_type"] == "Beam2"
    assert proposal.display_summary["line_element_type"] == "Beam2"
    assert "线单元 Beam2" in prepared.data["proposal_view"]["summary"]
    assert session.snapshot() == before


@pytest.mark.parametrize(
    ("line_element_type", "expected_dofs"),
    [("Truss2", 3), ("Beam2", 6)],
)
def test_line_mesh_success_commits_once_with_the_explicit_formulation(
    line_element_type: str,
    expected_dofs: int,
) -> None:
    session = _session()
    requests: list[AgentMeshTaskRequest] = []
    port = SessionGeometryAuthoringPort(
        session,
        lambda: None,
        lambda request: requests.append(request) is None,
    )
    bridge = AgentAuthoringBridge(port)
    bridge.bind_snapshot(session.snapshot())
    proposal = _proposal(
        session,
        f"proposal-{line_element_type}",
        line_element_type,
    )
    bridge.register_proposal(proposal)
    before = session.snapshot()

    running = bridge.accept_from_gui_control(proposal.proposal_id)
    during = session.snapshot()
    accepted = port.accept_mesh_result(
        proposal.proposal_id,
        _model(line_element_type),
    )
    after = session.snapshot()

    assert running.state is ProposalState.RUNNING
    assert during == before
    assert len(requests) == 1
    assert accepted.accepted
    assert bridge.state(proposal.proposal_id) is ProposalState.SUCCEEDED
    assert after.session_revision == before.session_revision + 1
    assert after.mesh_settings == MeshSettings(
        0.25,
        order=1,
        cell_shape="line",
        line_element_type=line_element_type,
    )
    assert after.artifact.model.mesh.dofs_per_node == expected_dofs


@pytest.mark.parametrize(
    "terminal",
    [ProposalState.CANCELLED, ProposalState.STALE],
)
def test_line_remesh_cancel_and_stale_preserve_the_installed_model(
    terminal: ProposalState,
) -> None:
    session = _session()
    session.replace_mesh_settings(
        MeshSettings(
            0.5,
            cell_shape="line",
            line_element_type="Truss2",
        )
    )
    initial_task = session.prepare_mesh_generation()
    assert session.accept_generated_model(
        initial_task.token,
        _model("Truss2"),
    ).accepted
    requests: list[AgentMeshTaskRequest] = []
    port = SessionGeometryAuthoringPort(
        session,
        lambda: None,
        lambda request: requests.append(request) is None,
    )
    bridge = AgentAuthoringBridge(port)
    bridge.bind_snapshot(session.snapshot())
    proposal = _proposal(session, f"proposal-{terminal.value}", "Beam2")
    bridge.register_proposal(proposal)
    bridge.accept_from_gui_control(proposal.proposal_id)
    running_snapshot = session.snapshot()

    if terminal is ProposalState.STALE:
        session.rename_native_model("User changed model")
        current = session.snapshot()
        assert not port.accept_mesh_result(
            proposal.proposal_id,
            _model("Beam2"),
        ).accepted
        port.terminate_mesh(proposal.proposal_id, terminal, "revision changed")
        after = session.snapshot()
        assert after == current
    else:
        port.terminate_mesh(proposal.proposal_id, terminal, "cancelled")
        after = session.snapshot()
        assert after == running_snapshot

    assert bridge.state(proposal.proposal_id) is terminal
    assert after.mesh_settings.line_element_type == "Truss2"
    assert {element.type for element in after.artifact.model.mesh.elements} == {
        "Truss2"
    }
    assert session.validate_task_token(requests[0].task.token).value == (
        "already_completed"
    )
