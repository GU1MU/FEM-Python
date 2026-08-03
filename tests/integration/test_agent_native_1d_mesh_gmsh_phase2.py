from __future__ import annotations

import pytest

from fem.application import MeshEntityRef, ModelSession, UnitContext
from fem.application.native_scope_materialization import (
    mesh_references_for_logical_entities,
)
from fem.application.preprocessing import generate_fem_model
from fem.geometry import LogicalEntityRef
from fem.geometry.part_namespace import namespace_part_logical_id
from fem.geometry.recipes import WireGeometry, WireMember, WirePoint
from fem.io.project_v3 import load_project_v3, save_project_v3
from fem_agent.authoring import ProposalState
from fem_agent.mesh_authoring import MeshIntent, create_mesh_proposal
from fem_gui.agent_authoring import (
    AgentAuthoringBridge,
    AgentMeshTaskRequest,
    SessionGeometryAuthoringPort,
    authoring_context_from_snapshot,
)


def _wire() -> WireGeometry:
    return WireGeometry(
        "Frame",
        (
            WirePoint("A", 0.0, 0.0, 0.0),
            WirePoint("B", 1.0, 0.0, 0.0),
            WirePoint("C", 1.0, 1.0, 0.5),
        ),
        (
            WireMember("AB", "A", "B"),
            WireMember("BC", "B", "C"),
        ),
    )


def _logical(logical_id: str) -> LogicalEntityRef:
    return LogicalEntityRef(namespace_part_logical_id("P1", logical_id))


@pytest.mark.parametrize(
    ("line_element_type", "expected_dofs"),
    [("Truss2", 3), ("Beam2", 6)],
)
def test_real_agent_line_mesh_materializes_stable_scopes_and_reopens(
    real_gmsh,
    tmp_path,
    line_element_type: str,
    expected_dofs: int,
) -> None:
    del real_gmsh
    session = ModelSession()
    session.create_native_project_with_first_part(
        "Line model",
        UnitContext("mm", "N", "MPa"),
        _wire(),
        part_name="Wire",
    )
    requests: list[AgentMeshTaskRequest] = []
    port = SessionGeometryAuthoringPort(
        session,
        lambda: None,
        lambda request: requests.append(request) is None,
    )
    bridge = AgentAuthoringBridge(port)
    bridge.bind_snapshot(session.snapshot())
    intent = MeshIntent(
        "line",
        1,
        global_size=0.2,
        line_element_type=line_element_type,
    )
    proposal = create_mesh_proposal(
        proposal_id=f"proposal-real-{line_element_type}",
        agent_session_id="agent-real-line-phase2",
        turn_id=f"turn-real-{line_element_type}",
        source_tool_call_ids=(f"call-real-{line_element_type}",),
        context=authoring_context_from_snapshot(session.snapshot()),
        draft_revision=1,
        part_id="P1",
        mesh_intent=intent,
    )
    bridge.register_proposal(proposal)
    before_revision = session.session_revision

    running = bridge.accept_from_gui_control(proposal.proposal_id)
    generated = generate_fem_model(requests[0].task)
    accepted = port.accept_mesh_result(proposal.proposal_id, generated)
    installed = session.snapshot().artifact.model

    assert running.state is ProposalState.RUNNING
    assert accepted.accepted
    assert session.session_revision == before_revision + 1
    assert installed.mesh.dofs_per_node == expected_dofs
    assert {element.type for element in installed.mesh.elements} == {
        line_element_type
    }
    for point_name in ("A", "B", "C"):
        references = mesh_references_for_logical_entities(
            installed,
            (_logical(f"point:{point_name}"),),
            mesh_kind="node",
        )
        assert len(references) == 1
    member_references = {
        member_name: mesh_references_for_logical_entities(
            installed,
            (_logical(f"edge:{member_name}"),),
            mesh_kind="element",
        )
        for member_name in ("AB", "BC")
    }
    assert all(member_references.values())
    assert set(member_references["AB"]).isdisjoint(member_references["BC"])
    assert {
        reference
        for references in member_references.values()
        for reference in references
    } == {
        MeshEntityRef.element(element.id, part_id="P1")
        for element in installed.mesh.elements
    }

    save_task = session.prepare_project_save()
    path = save_project_v3(
        tmp_path / f"agent-{line_element_type}.femproj",
        save_task.snapshot,
    )
    assert session.accept_project_saved(save_task.token, path).accepted
    reopened = ModelSession()
    assert reopened.replace_from_snapshot(load_project_v3(path)).accepted
    regenerated_task = reopened.prepare_mesh_generation()
    regenerated = generate_fem_model(regenerated_task)
    assert reopened.accept_generated_model(
        regenerated_task.token,
        regenerated,
    ).accepted
    reopened_snapshot = reopened.snapshot()

    assert reopened_snapshot.mesh_settings.line_element_type == line_element_type
    assert reopened_snapshot.artifact.model.mesh.dofs_per_node == expected_dofs
    assert {
        element.type for element in reopened_snapshot.artifact.model.mesh.elements
    } == {line_element_type}
    assert mesh_references_for_logical_entities(
        reopened_snapshot.artifact.model,
        (LogicalEntityRef("point:B"),),
        mesh_kind="node",
    )
    assert mesh_references_for_logical_entities(
        reopened_snapshot.artifact.model,
        (LogicalEntityRef("edge:BC"),),
        mesh_kind="element",
    )
