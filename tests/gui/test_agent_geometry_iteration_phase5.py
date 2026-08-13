from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import pytest
from PySide6.QtWidgets import QApplication

from fem.application import ModelSession, UnitContext
from fem.core.model import MaterialDefinition
from fem.geometry import (
    SketchCircle,
    SketchGeometry,
    SketchRadiusDimension,
    SketchRectangle,
)
from fem_agent.authoring import ProposalState
from fem_agent.geometry_authoring import (
    add_planar_circle,
    apply_planar_edit_batch,
    create_geometry_edit_proposal,
    delete_planar_circles,
    planar_geometry_catalog,
    planar_sketch_geometry,
    replace_planar_circle_pattern,
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
from fem_gui.main_window import FEMMainWindow
from tests.gui.test_agent_result_query_phase_a7 import _solved_session


def _three_hole_sketch() -> SketchGeometry:
    return planar_sketch_geometry(
        "Plate",
        contours=(
            SketchRectangle("material", 0.0, 0.0, 20.0, 20.0),
            SketchCircle("cut", 5.0, 10.0, 1.0),
            SketchCircle("cut", 10.0, 10.0, 1.0),
            SketchCircle("cut", 15.0, 10.0, 1.0),
        ),
    ).recipe


def _circle_ids(recipe: object) -> list[str]:
    return [
        str(item["id"])
        for item in planar_geometry_catalog(recipe)["curves"]
        if item["kind"] == "circle"
    ]


def test_pattern_replaces_three_horizontal_holes_with_five_vertical_holes() -> None:
    original = _three_hole_sketch()
    original_non_circles = tuple(
        curve for curve in original.curves if not isinstance(curve, SketchCircle)
    )

    first = replace_planar_circle_pattern(
        original,
        target_circle_ids=_circle_ids(original),
        count=5,
        start_center_x=10.0,
        start_center_y=4.0,
        spacing_x=0.0,
        spacing_y=3.0,
        radius=0.75,
    )
    catalog = planar_geometry_catalog(first.recipe)
    circles = [item for item in catalog["curves"] if item["kind"] == "circle"]

    assert len(circles) == 5
    assert [(item["center_x"], item["center_y"]) for item in circles] == [
        (10.0, 4.0),
        (10.0, 7.0),
        (10.0, 10.0),
        (10.0, 13.0),
        (10.0, 16.0),
    ]
    assert {item["radius"] for item in circles} == {0.75}
    assert tuple(
        curve
        for curve in first.recipe.curves
        if not isinstance(curve, SketchCircle)
    ) == original_non_circles
    assert not set(_circle_ids(original)) & set(_circle_ids(first.recipe))

    second = replace_planar_circle_pattern(
        first.recipe,
        target_circle_ids=_circle_ids(first.recipe),
        count=3,
        start_center_x=5.0,
        start_center_y=10.0,
        spacing_x=5.0,
        spacing_y=0.0,
        radius=1.0,
    )
    assert len(_circle_ids(second.recipe)) == 3


def test_delete_pattern_and_failed_batch_are_strict_and_atomic() -> None:
    recipe = _three_hole_sketch()
    circle_ids = _circle_ids(recipe)
    constrained = SketchGeometry(
        recipe.name,
        recipe.plane,
        recipe.points,
        recipe.curves,
        (SketchRadiusDimension("D1", circle_ids[0], 1.0),),
    )

    with pytest.raises(Exception, match="constraint"):
        delete_planar_circles(constrained, circle_ids=[circle_ids[0]])
    with pytest.raises(ValueError, match="existing"):
        delete_planar_circles(recipe, circle_ids=["missing"])
    with pytest.raises(ValueError, match="unique"):
        delete_planar_circles(recipe, circle_ids=[circle_ids[0], circle_ids[0]])
    with pytest.raises(ValueError, match="non-zero"):
        replace_planar_circle_pattern(
            recipe,
            target_circle_ids=circle_ids,
            count=2,
            start_center_x=1.0,
            start_center_y=1.0,
            spacing_x=0.0,
            spacing_y=0.0,
            radius=1.0,
        )
    before = deepcopy(recipe)
    with pytest.raises(ValueError, match="existing"):
        apply_planar_edit_batch(
            recipe,
            edits=(
                {
                    "operation": "add_circle",
                    "center_x": 2.0,
                    "center_y": 2.0,
                    "radius": 0.25,
                },
                {"operation": "delete_circles", "circle_ids": ["missing"]},
            ),
        )
    assert recipe == before


def test_batch_preparation_registers_one_proposal_and_commits_one_revision() -> None:
    session = ModelSession()
    session.create_native_project_with_first_part(
        "Plate Model",
        UnitContext("mm", "N", "MPa"),
        _three_hole_sketch(),
    )
    refreshes: list[int] = []
    port = SessionGeometryAuthoringPort(
        session, lambda: refreshes.append(session.session_revision)
    )
    bridge = AgentAuthoringBridge(port)
    bridge.bind_snapshot(session.snapshot())
    controller = create_session_authoring_workflow_controller(
        session,
        bridge,
        AgentResultQueryBridge(SessionResultQueryPort(session)),
    )
    before = session.snapshot()
    read = controller.dispatch(
        "read_geometry_edit_context",
        {"part_id": "P1"},
        ToolExecutionContext(session.session_id, session.session_revision, "read"),
    )
    assert {
        "delete_circles",
        "replace_circle_pattern",
        "batch",
    } <= set(read.data["supported_edits"])
    outcome = controller.dispatch(
        "prepare_geometry_edit",
        {
            "part_id": "P1",
            "edit": {
                "operation": "batch",
                "edits": [
                    {
                        "operation": "replace_circle_pattern",
                        "target_circle_ids": _circle_ids(
                            before.parts[0].geometry_recipe
                        ),
                        "count": 5,
                        "start_center_x": 10.0,
                        "start_center_y": 4.0,
                        "spacing_x": 0.0,
                        "spacing_y": 3.0,
                        "radius": 0.75,
                    }
                ],
            },
        },
        ToolExecutionContext(session.session_id, session.session_revision, "batch"),
    )

    assert outcome.ok
    assert outcome.data["geometry_edit_mode"] == "in_place"
    assert session.session_revision == before.session_revision
    receipt = bridge.accept_from_gui_control(str(outcome.data["proposal_id"]))
    assert receipt.state is ProposalState.SUCCEEDED
    assert session.session_revision == before.session_revision + 1
    assert refreshes == [session.session_revision]
    assert len(_circle_ids(session.snapshot().parts[0].geometry_recipe)) == 5


def test_main_window_geometry_edit_uses_in_place_then_automatic_branch() -> None:
    QApplication.instance() or QApplication([])
    window = FEMMainWindow()
    units = UnitContext("mm", "N", "MPa")
    window.session.create_native_project_with_first_part(
        "Plate Model",
        units,
        _three_hole_sketch(),
    )
    window._rebuild_full_projection()
    source = window.workspace.active_document()
    assert source is not None

    in_place_draft = add_planar_circle(
        window.session.snapshot().parts[0].geometry_recipe,
        center_x=2.0,
        center_y=2.0,
        radius=0.25,
    )
    in_place = create_geometry_edit_proposal(
        proposal_id="proposal-in-place",
        agent_session_id="agent-phase5",
        turn_id="turn-in-place",
        source_tool_call_ids=("call-in-place",),
        context=authoring_context_from_snapshot(
            window.session.snapshot(), document_id=source.document_id
        ),
        draft_revision=1,
        part_id="P1",
        draft=in_place_draft,
        summary="add one hole",
        edit_mode="in_place",
    )
    window.agent_authoring_bridge.register_proposal(in_place)
    assert (
        window.agent_authoring_bridge.accept_from_gui_control(in_place.proposal_id).state
        is ProposalState.SUCCEEDED
    )
    assert window.workspace.active_document_id == source.document_id
    assert window.agent_authoring_bridge.port.latest_geometry_iteration_report()[
        "mode"
    ] == "in_place"

    window.session.replace_model_definitions(
        (MaterialDefinition("Steel", {"E": 210000.0, "nu": 0.3}),),
        (),
        (),
        (),
    )
    window._rebuild_full_projection()
    source_before = source.session.project_snapshot_for_branch()
    branch_draft = add_planar_circle(
        window.session.snapshot().parts[0].geometry_recipe,
        center_x=18.0,
        center_y=18.0,
        radius=0.25,
    )
    branch = create_geometry_edit_proposal(
        proposal_id="proposal-branch",
        agent_session_id="agent-phase5",
        turn_id="turn-branch",
        source_tool_call_ids=("call-branch",),
        context=authoring_context_from_snapshot(
            window.session.snapshot(), document_id=source.document_id
        ),
        draft_revision=2,
        part_id="P1",
        draft=branch_draft,
        summary="create iteration",
        edit_mode="branch",
    )
    assert branch.display_summary["geometry_edit_mode"] == "branch"
    assert branch.expected_changes["creates_iteration_model"] is True
    assert branch.invalidation_impact["results"] is False
    window.agent_authoring_bridge.register_proposal(branch)
    receipt = window.agent_authoring_bridge.accept_from_gui_control(branch.proposal_id)

    assert receipt.state is ProposalState.SUCCEEDED
    child = window.workspace.active_document()
    assert child is not None and child.document_id != source.document_id
    assert source.session.project_snapshot_for_branch() == source_before
    assert child.lineage is not None
    assert child.lineage.source_document_id == source.document_id
    assert window.session is child.session
    assert window.agent_authoring_bridge.context.binding.document_id == str(
        child.document_id
    )
    report = window.agent_authoring_bridge.port.latest_geometry_iteration_report()
    assert report is not None
    assert report["mode"] == "branch"
    assert report["runs"] == "not_migrated"
    assert report["results"] == "not_migrated"
    window.close()


def test_branch_preserves_source_result_without_result_loss_confirmation() -> None:
    QApplication.instance() or QApplication([])
    window = FEMMainWindow()
    solved = _solved_session()
    source = window.workspace.active_document()
    assert source is not None
    window.session = solved
    window.document = solved.projection_snapshot()
    source = window.workspace.active_document()
    assert source is not None
    window._bind_agent_document(source)
    window._rebuild_full_projection()
    before = solved.snapshot()
    displayed_run_id = before.displayed_result_run_id
    assert displayed_run_id is not None
    confirmation_calls: list[bool] = []
    window.agent_authoring_bridge.set_result_invalidation_confirmation(
        lambda: confirmation_calls.append(True) or False
    )
    draft = add_planar_circle(
        before.parts[0].geometry_recipe,
        center_x=5.0,
        center_y=2.0,
        radius=0.5,
    )
    proposal = create_geometry_edit_proposal(
        proposal_id="proposal-result-branch",
        agent_session_id="agent-phase5",
        turn_id="turn-result-branch",
        source_tool_call_ids=("call-result-branch",),
        context=authoring_context_from_snapshot(
            before, document_id=source.document_id
        ),
        draft_revision=3,
        part_id="P1",
        draft=draft,
        summary="branch result model",
        edit_mode="branch",
    )
    window.agent_authoring_bridge.register_proposal(proposal)
    receipt = window.agent_authoring_bridge.accept_from_gui_control(
        proposal.proposal_id
    )

    assert receipt.state is ProposalState.SUCCEEDED
    assert confirmation_calls == []
    assert solved.result_identity_for(displayed_run_id) is not None
    assert any(run.run_id == displayed_run_id for run in solved.snapshot().runs)
    child = window.workspace.active_document()
    assert child is not None and child.lineage is not None
    assert child.lineage.source_run_id == displayed_run_id
    child_snapshot = child.session.snapshot()
    assert not child_snapshot.runs
    assert not child_snapshot.result_generations
    report = window.agent_authoring_bridge.port.latest_geometry_iteration_report()
    assert report["source_state"] == {"runs": "retained", "results": "retained"}
    assert report["target_state"]["runs"] == "not_migrated"
    assert report["target_state"]["results"] == "not_migrated"
    window.close()


def test_branch_activation_failure_restores_workspace_window_and_agent(
    monkeypatch,
) -> None:
    QApplication.instance() or QApplication([])
    window = FEMMainWindow()
    window.session.create_native_project_with_first_part(
        "Plate Model",
        UnitContext("mm", "N", "MPa"),
        _three_hole_sketch(),
    )
    window.session.replace_model_definitions(
        (MaterialDefinition("Steel", {"E": 1.0}),), (), (), ()
    )
    window._rebuild_full_projection()
    source = window.workspace.active_document()
    assert source is not None
    source_snapshot = source.session.project_snapshot_for_branch()
    draft = add_planar_circle(
        source.session.snapshot().parts[0].geometry_recipe,
        center_x=2.0,
        center_y=2.0,
        radius=0.25,
    )
    proposal = create_geometry_edit_proposal(
        proposal_id="proposal-activation-failure",
        agent_session_id="agent-phase5",
        turn_id="turn-activation-failure",
        source_tool_call_ids=("call-activation-failure",),
        context=authoring_context_from_snapshot(
            source.session.snapshot(), document_id=source.document_id
        ),
        draft_revision=4,
        part_id="P1",
        draft=draft,
        summary="fail activation",
        edit_mode="branch",
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
    assert window.session is source.session
    assert source.session.project_snapshot_for_branch() == source_snapshot
    assert window.agent_authoring_bridge.context.binding.document_id == str(
        source.document_id
    )
    assert window.agent_authoring_bridge.context.binding.session_id == (
        source.session.session_id
    )
    window.close()


def test_latest_iteration_report_is_owned_and_document_bound() -> None:
    session = ModelSession()
    session.create_native_project_with_first_part(
        "Plate Model",
        UnitContext("mm", "N", "MPa"),
        _three_hole_sketch(),
    )
    port = SessionGeometryAuthoringPort(session, lambda: None)
    bridge = AgentAuthoringBridge(port)
    bridge.bind_snapshot(session.snapshot(), document_id=1)
    draft = add_planar_circle(
        session.snapshot().parts[0].geometry_recipe,
        center_x=2.0,
        center_y=2.0,
        radius=0.25,
    )
    proposal = create_geometry_edit_proposal(
        proposal_id="proposal-report-ownership",
        agent_session_id="agent-phase5",
        turn_id="turn-report-ownership",
        source_tool_call_ids=("call-report-ownership",),
        context=authoring_context_from_snapshot(session.snapshot(), document_id=1),
        draft_revision=5,
        part_id="P1",
        draft=draft,
        summary="report ownership",
    )
    bridge.register_proposal(proposal)
    assert (
        bridge.accept_from_gui_control(proposal.proposal_id).state
        is ProposalState.SUCCEEDED
    )
    first = port.latest_geometry_iteration_report()
    first["target"]["session_id"] = "mutated"
    assert port.latest_geometry_iteration_report()["target"]["session_id"] == (
        session.session_id
    )

    bridge.bind_snapshot(session.snapshot(), document_id=1)
    assert port.latest_geometry_iteration_report() is not None
    bridge.bind_snapshot(session.snapshot(), document_id=2)
    assert port.latest_geometry_iteration_report() is None


def test_source_run_selection_prefers_selected_before_provider_and_latest() -> None:
    QApplication.instance() or QApplication([])
    window = FEMMainWindow()
    selected_source = SimpleNamespace(run_id="run-selected")
    latest_source = SimpleNamespace(run_id="run-latest")
    identities = {
        "run-selected": (selected_source, 1),
        "run-latest": (latest_source, 2),
    }
    fake_session = SimpleNamespace(
        session_id="session-source",
        snapshot=lambda: SimpleNamespace(
            displayed_result_run_id=None,
            selected_run_id="run-selected",
            runs=(SimpleNamespace(run_id="run-latest"),),
        ),
        result_identity_for=lambda run_id: identities.get(run_id),
    )
    source = SimpleNamespace(session=fake_session)

    assert window._agent_geometry_edit_source_run_id(source) == "run-selected"
    window.close()


def test_partial_agent_port_bind_failure_restores_every_binding(
    monkeypatch,
) -> None:
    QApplication.instance() or QApplication([])
    window = FEMMainWindow()
    window.session.create_native_project_with_first_part(
        "Plate Model",
        UnitContext("mm", "N", "MPa"),
        _three_hole_sketch(),
    )
    window.session.replace_model_definitions(
        (MaterialDefinition("Steel", {"E": 1.0}),), (), (), ()
    )
    window._rebuild_full_projection()
    source = window.workspace.active_document()
    assert source is not None
    source_snapshot = source.session.project_snapshot_for_branch()
    authoring_port = window.agent_authoring_bridge.port
    result_port = window.agent_result_query_bridge.port
    original_result_bind = result_port.bind_session
    failed = False

    def fail_child_once(session, *, idle=None):
        nonlocal failed
        if session is not source.session and not failed:
            failed = True
            raise RuntimeError("partial result-port bind failure")
        return original_result_bind(session, idle=idle)

    monkeypatch.setattr(result_port, "bind_session", fail_child_once)
    draft = add_planar_circle(
        source.session.snapshot().parts[0].geometry_recipe,
        center_x=2.0,
        center_y=2.0,
        radius=0.25,
    )
    proposal = create_geometry_edit_proposal(
        proposal_id="proposal-partial-bind-failure",
        agent_session_id="agent-phase5",
        turn_id="turn-partial-bind-failure",
        source_tool_call_ids=("call-partial-bind-failure",),
        context=authoring_context_from_snapshot(
            source.session.snapshot(), document_id=source.document_id
        ),
        draft_revision=6,
        part_id="P1",
        draft=draft,
        summary="partial bind failure",
        edit_mode="branch",
    )
    window.agent_authoring_bridge.register_proposal(proposal)
    receipt = window.agent_authoring_bridge.accept_from_gui_control(
        proposal.proposal_id
    )

    assert receipt.state is ProposalState.FAILED
    assert failed
    assert window.workspace.document_count == 1
    assert window.workspace.active_document_id == source.document_id
    assert window.session is source.session
    assert source.session.project_snapshot_for_branch() == source_snapshot
    assert authoring_port.session is source.session
    assert result_port.session is source.session
    assert window.agent_authoring_bridge.context.binding.document_id == str(
        source.document_id
    )
    runtime = window.viewport_panel.agent_chat_drawer.agent_runtime
    assert runtime.target_identity == (
        str(source.document_id),
        source.session.session_id,
    )
    window.close()
