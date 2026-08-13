from __future__ import annotations

import pytest
from PySide6.QtWidgets import QApplication

from fem.application import ModelSession, UnitContext
from fem.core.model import MaterialDefinition
from fem.geometry import (
    BooleanGeometry,
    BoxGeometry,
    ExtrudedGeometry,
    MovedGeometry,
    MultiBodyGeometry,
    PathSweptGeometry,
    RevolvedGeometry,
    SketchRectangle,
    WireGeometry,
    WireMember,
    WirePoint,
    describe_recipe_topology,
)
from fem_agent.authoring import ProposalState
from fem_agent.geometry_authoring import (
    create_profile_extrusion_proposal,
    create_profile_path_sweep_proposal,
    create_profile_revolution_proposal,
    planar_sketch_geometry,
)
from fem_agent.result_authoring import AgentResultQueryBridge
from fem_agent.tools.registry import ToolExecutionContext
from fem_gui.agent_authoring import (
    AgentAuthoringBridge,
    SessionGeometryAuthoringPort,
    SessionResultQueryPort,
    authoring_context_from_snapshot,
    create_session_authoring_workflow_controller,
)
from fem_gui.model_iteration import ModelIterationService, geometry_edit_policy
from fem_gui.main_window import FEMMainWindow
from fem_gui.workspace import FEMWorkspace
from tests.geometry.test_profile_extrusion import profile_face_id, two_profile_sketch
from tests.gui.test_agent_exact_boolean_phase4 import (
    _body_call,
    _multi_body_session,
    _part_call,
    _part_session,
)
from tests.gui.test_agent_result_query_phase_a7 import _solved_session


def _single_profile_session() -> tuple[ModelSession, object, str]:
    sketch = planar_sketch_geometry(
        "Profile",
        contours=(SketchRectangle("material", 0.0, 0.0, 2.0, 1.0),),
    ).recipe
    face_id = next(
        entity.logical_id
        for entity in describe_recipe_topology(sketch).entities
        if entity.kind == "face" and entity.semantic_role == "sketch.profile"
    )
    session = ModelSession()
    session.create_native_project_with_first_part(
        "Profile model",
        UnitContext("mm", "N", "MPa"),
        sketch,
    )
    return session, sketch, face_id


def _profile_proposal(kind: str, session: ModelSession, document_id: int, mode: str):
    context = authoring_context_from_snapshot(
        session.snapshot(),
        document_id=document_id,
    )
    common = {
        "proposal_id": f"proposal-{kind}-{mode}",
        "agent_session_id": "agent-phase7",
        "turn_id": f"turn-{kind}-{mode}",
        "source_tool_call_ids": (f"call-{kind}-{mode}",),
        "context": context,
        "draft_revision": 1,
        "part_id": "P1",
        "summary": f"{kind} profile",
        "edit_mode": mode,
    }
    if kind == "extrude":
        sketch = session.snapshot().parts[0].geometry_recipe
        face_ids = tuple(
            entity.logical_id
            for entity in describe_recipe_topology(sketch).entities
            if entity.kind == "face" and entity.semantic_role == "sketch.profile"
        )
        return create_profile_extrusion_proposal(
            **common,
            base_recipe=sketch,
            source_face_ids=face_ids,
            height=2.0,
        )
    sketch = session.snapshot().parts[0].geometry_recipe
    face_id = profile_face_id(sketch, "L1") if kind == "revolve" else next(
        entity.logical_id
        for entity in describe_recipe_topology(sketch).entities
        if entity.kind == "face" and entity.semantic_role == "sketch.profile"
    )
    if kind == "revolve":
        return create_profile_revolution_proposal(
            **common,
            base_recipe=sketch,
            source_face_id=face_id,
            axis="y",
            angle_degrees=180.0,
        )
    return create_profile_path_sweep_proposal(
        **common,
        base_recipe=sketch,
        source_face_id=face_id,
        path=WireGeometry(
            "Path",
            (
                WirePoint("A", 0.0, 0.0, 0.0),
                WirePoint("B", 0.0, 0.0, 3.0),
            ),
            (WireMember("AB", "A", "B"),),
        ),
        frame_strategy="transport",
    )


def _profile_session(kind: str) -> ModelSession:
    if kind in {"extrude", "revolve"}:
        session = ModelSession()
        sketch = two_profile_sketch() if kind == "extrude" else two_profile_sketch()
        session.create_native_project_with_first_part(
            "Profile model",
            UnitContext("mm", "N", "MPa"),
            sketch,
        )
        return session
    return _single_profile_session()[0]


def _branch_bridge(session: ModelSession):
    workspace = FEMWorkspace()
    document = workspace.add_model(session, session.projection_snapshot())
    workspace.activate(document)

    def commit(mutation, revision):
        result = ModelIterationService(workspace).branch_geometry_mutation(
            document.document_id,
            mutation.affected_part_ids,
            mutation.apply,
            expected_source_session_revision=revision,
        )
        return {"mode": "branch", **result.report.to_dict()}

    port = SessionGeometryAuthoringPort(
        session,
        lambda: None,
        geometry_edit_mode=lambda: geometry_edit_policy(document).value,
        commit_geometry_edit=commit,
    )
    bridge = AgentAuthoringBridge(port)
    bridge.bind_snapshot(session.snapshot(), document_id=document.document_id)
    return workspace, document, bridge


@pytest.mark.parametrize(
    ("kind", "recipe_type"),
    (
        ("extrude", ExtrudedGeometry),
        ("revolve", RevolvedGeometry),
        ("sweep", PathSweptGeometry),
    ),
)
def test_profile_transforms_are_in_place_during_pure_geometry(
    kind: str,
    recipe_type: type,
) -> None:
    session = _profile_session(kind)
    port = SessionGeometryAuthoringPort(session, lambda: None)
    bridge = AgentAuthoringBridge(port)
    bridge.bind_snapshot(session.snapshot(), document_id=1)
    proposal = _profile_proposal(kind, session, 1, "in_place")
    bridge.register_proposal(proposal)

    receipt = bridge.accept_from_gui_control(proposal.proposal_id)

    assert receipt.state is ProposalState.SUCCEEDED
    assert isinstance(session.snapshot().parts[0].geometry_recipe, recipe_type)
    assert port.latest_geometry_iteration_report()["mode"] == "in_place"


@pytest.mark.parametrize(
    ("kind", "recipe_type"),
    (
        ("extrude", ExtrudedGeometry),
        ("revolve", RevolvedGeometry),
        ("sweep", PathSweptGeometry),
    ),
)
def test_profile_transforms_branch_with_downstream_state(
    kind: str,
    recipe_type: type,
) -> None:
    session = _profile_session(kind)
    session.replace_model_definitions(
        (MaterialDefinition("Steel", {"E": 1.0}),),
        (),
        (),
        (),
    )
    workspace, source, bridge = _branch_bridge(session)
    source_before = source.session.snapshot()
    proposal = _profile_proposal(kind, session, source.document_id, "branch")
    bridge.register_proposal(proposal)

    receipt = bridge.accept_from_gui_control(proposal.proposal_id)

    assert receipt.state is ProposalState.SUCCEEDED
    child = workspace.active_document()
    assert child is not None and child is not source
    assert source.session.snapshot() == source_before
    child_snapshot = child.session.snapshot()
    assert any(
        isinstance(part.geometry_recipe, recipe_type)
        for part in child_snapshot.parts
        if not part.suppressed
    )
    assert [item.name for item in child_snapshot.materials] == ["Steel"]
    assert child_snapshot.artifact is None
    assert not child_snapshot.validations
    assert not child_snapshot.runs
    assert not child_snapshot.result_generations
    report = bridge.port.latest_geometry_iteration_report()
    assert report["mode"] == "branch"
    assert report["affected_part_ids"]


@pytest.mark.parametrize("kind", ("revolve", "sweep"))
def test_derived_profile_branch_display_matches_top_level_impact(kind: str) -> None:
    session = _profile_session(kind)
    proposal = _profile_proposal(kind, session, 1, "branch")

    assert proposal.invalidation_impact == {
        "mesh": False,
        "definitions": False,
        "results": False,
    }
    assert proposal.display_summary["invalidation_impact"] == (
        proposal.invalidation_impact
    )
    assert proposal.display_summary["invalidated_objects"] == []


@pytest.mark.gmsh
@pytest.mark.parametrize("kind", ("part", "body"))
def test_boolean_operations_branch_without_mutating_source(kind: str) -> None:
    session = (
        _part_session(
            BoxGeometry("Target", 2.0, 1.0, 1.0),
            MovedGeometry(BoxGeometry("Tool", 1.0, 1.0, 1.0), 1.5, 0.0, 0.0),
        )
        if kind == "part"
        else _multi_body_session()
    )
    session.replace_model_definitions(
        (MaterialDefinition("Steel", {"E": 1.0}),), (), (), ()
    )
    workspace, source, bridge = _branch_bridge(session)
    controller = create_session_authoring_workflow_controller(
        session,
        bridge,
        AgentResultQueryBridge(SessionResultQueryPort(session)),
    )
    source_before = source.session.snapshot()
    prepared = controller.dispatch(
        "prepare_geometry_edit",
        _part_call("fuse") if kind == "part" else _body_call("fuse"),
        ToolExecutionContext("agent-phase7", session.session_revision, kind),
    )
    assert prepared.ok, prepared.summary
    assert prepared.data["geometry_edit_mode"] == "branch"

    receipt = bridge.accept_from_gui_control(str(prepared.data["proposal_id"]))

    assert receipt.state is ProposalState.SUCCEEDED
    assert source.session.snapshot() == source_before
    child = workspace.active_document()
    assert child is not None and child is not source
    child_snapshot = child.session.snapshot()
    assert [item.name for item in child_snapshot.materials] == ["Steel"]
    assert not child_snapshot.runs
    if kind == "part":
        assert [part.suppressed for part in child_snapshot.parts[:2]] == [True, True]
        assert isinstance(child_snapshot.parts[-1].geometry_recipe, BooleanGeometry)
    else:
        geometry = child_snapshot.parts[0].geometry_recipe
        assert isinstance(geometry, MultiBodyGeometry)
        assert {body.id for body in geometry.bodies} == {"B1", "B3"}
    proposal = bridge._records[str(prepared.data["proposal_id"])]
    assert proposal.proposal.invalidation_impact["results"] is False


def test_policy_change_and_headless_branch_fail_closed() -> None:
    session = _profile_session("revolve")
    mode = "in_place"
    port = SessionGeometryAuthoringPort(
        session,
        lambda: None,
        geometry_edit_mode=lambda: mode,
    )
    bridge = AgentAuthoringBridge(port)
    bridge.bind_snapshot(session.snapshot(), document_id=1)
    proposal = _profile_proposal("revolve", session, 1, "in_place")
    bridge.register_proposal(proposal)
    mode = "branch"
    assert bridge.accept_from_gui_control(proposal.proposal_id).state is ProposalState.FAILED
    assert session.snapshot().parts[0].geometry_recipe == two_profile_sketch()

    branch_session = _profile_session("revolve")
    branch_port = SessionGeometryAuthoringPort(
        branch_session,
        lambda: None,
        geometry_edit_mode=lambda: "branch",
    )
    branch_bridge = AgentAuthoringBridge(branch_port)
    branch_bridge.bind_snapshot(branch_session.snapshot(), document_id=2)
    branch = _profile_proposal("revolve", branch_session, 2, "branch")
    branch_bridge.register_proposal(branch)
    before = branch_session.snapshot()
    receipt = branch_bridge.accept_from_gui_control(branch.proposal_id)
    assert receipt.state is ProposalState.FAILED
    assert "workspace-aware commit seam" in receipt.message
    assert branch_session.snapshot() == before


def test_profile_branch_activation_failure_rolls_back_workspace_and_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    QApplication.instance() or QApplication([])
    window = FEMMainWindow()
    sketch = two_profile_sketch()
    window.session.create_native_project_with_first_part(
        "Profile model",
        UnitContext("mm", "N", "MPa"),
        sketch,
    )
    window.session.replace_model_definitions(
        (MaterialDefinition("Steel", {"E": 1.0}),), (), (), ()
    )
    window._rebuild_full_projection()
    source = window.workspace.active_document()
    assert source is not None
    source_before = source.session.snapshot()
    proposal = _profile_proposal(
        "revolve",
        source.session,
        source.document_id,
        "branch",
    )
    monkeypatch.setattr(window, "_activate_workspace_context", lambda _child: False)
    window.agent_authoring_bridge.port._commit_geometry_edit_callback = (
        window._commit_agent_geometry_edit
    )
    window.agent_authoring_bridge.register_proposal(proposal)

    receipt = window.agent_authoring_bridge.accept_from_gui_control(
        proposal.proposal_id
    )

    assert receipt.state is ProposalState.FAILED
    assert window.workspace.document_count == 1
    assert window.workspace.active_document_id == source.document_id
    assert source.session.snapshot() == source_before
    assert window.session is source.session
    assert window.agent_authoring_bridge.port.session is source.session
    assert window.agent_authoring_bridge.context.binding.document_id == str(
        source.document_id
    )
    window.close()


def test_profile_branch_retains_source_result_and_records_lineage() -> None:
    source_session = _solved_session()
    source_run_id = source_session.snapshot().displayed_result_run_id
    assert source_run_id is not None
    workspace = FEMWorkspace()
    source = workspace.add_model(
        source_session,
        source_session.projection_snapshot(),
    )
    source_before = source_session.snapshot()
    source_projection_before = source_session.project_snapshot_for_branch()
    recipe = ExtrudedGeometry(
        source_before.parts[0].geometry_recipe,
        2.0,
        ("face:domain",),
    )

    result = ModelIterationService(workspace).branch_geometry_mutation(
        source.document_id,
        ("P1",),
        lambda session, revision: session.replace_part_geometry(
            "P1",
            recipe,
            expected_session_revision=revision,
        ),
        source_run_id=source_run_id,
    )

    assert source_session.project_snapshot_for_branch() == source_projection_before
    assert source_session.session_revision == source_before.session_revision
    assert tuple(run.run_id for run in source_session.snapshot().runs) == tuple(
        run.run_id for run in source_before.runs
    )
    assert source_session.result_identity_for(source_run_id) is not None
    child_snapshot = result.document.session.snapshot()
    assert not child_snapshot.runs
    assert not child_snapshot.result_generations
    assert result.document.lineage is not None
    assert result.document.lineage.source_run_id == source_run_id
    assert result.report.source_run_id == source_run_id
