from __future__ import annotations

import os
from dataclasses import replace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QToolButton

from fem.application import AnalysisRun, ModelSession, RunStatus, UnitContext
from fem.mesh.settings import MeshSettings
from fem_agent.authoring import (
    AuthoringAuthorizationError,
    ModelPatch,
    ProposalState,
)
from fem_agent.definition_authoring import create_scope_definition_change
from fem_gui.agent_authoring import (
    AgentAuthoringBridge,
    AppliedPatchState,
    SessionGeometryAuthoringPort,
    authoring_context_from_snapshot,
)
from fem_gui.widgets.agent_chat import AgentChatDrawer
from tests.test_agent_authoring_phase_a4 import (
    _change,
    _plate_model,
    _recipe,
    _session,
)


def _application() -> QApplication:
    return QApplication.instance() or QApplication([])


def test_a4_bridge_applies_once_and_gui_card_undoes_once() -> None:
    _application()
    session = _session()
    full_refreshes: list[str] = []
    definition_deltas: list[object] = []
    port = SessionGeometryAuthoringPort(
        session,
        lambda: full_refreshes.append("full"),
        apply_definition_delta=definition_deltas.append,
    )
    bridge = AgentAuthoringBridge(port)
    bridge.bind_snapshot(session.snapshot())
    drawer = AgentChatDrawer(authoring_bridge=bridge)

    record = bridge.apply_automatic_patch(_change(session))
    replay = bridge.apply_automatic_patch(record.patch)

    assert record.undo_available
    assert replay.replayed
    assert len(definition_deltas) == 1
    assert full_refreshes == []
    assert session.snapshot().session_revision == record.session_revision
    assert record.inverse_patch.base_session_revision == record.session_revision
    assert len(record.inverse_patch.patch_id) <= 128

    undo = drawer.findChild(
        QToolButton,
        "agentChatPatchUndoButton",
    )
    assert undo is not None and undo.isEnabled()

    QTest.mouseClick(undo, Qt.MouseButton.LeftButton)

    restored = port.patch_record(record.patch.patch_id)
    assert restored.state is AppliedPatchState.UNDONE
    assert not restored.undo_available
    assert len(definition_deltas) == 2
    assert full_refreshes == []
    assert not session.snapshot().named_regions
    assert not session.snapshot().materials


def test_a4_automatic_port_rejects_destructive_inverse_as_forward_patch() -> None:
    session = _session()
    port = SessionGeometryAuthoringPort(
        session,
        lambda: None,
        apply_definition_delta=lambda _delta: None,
    )
    bridge = AgentAuthoringBridge(port)
    bridge.bind_snapshot(session.snapshot())
    record = bridge.apply_automatic_patch(_change(session))

    with pytest.raises(ValueError, match="cannot overwrite or remove"):
        port.apply_patch(record.inverse_patch)

    assert session.snapshot().named_regions
    assert session.snapshot().materials


def test_a4_revision_change_disables_old_undo_entry() -> None:
    _application()
    session = _session()
    port = SessionGeometryAuthoringPort(
        session,
        lambda: None,
        apply_definition_delta=lambda _delta: None,
    )
    bridge = AgentAuthoringBridge(port)
    bridge.bind_snapshot(session.snapshot())
    record = bridge.apply_automatic_patch(_change(session))
    current = session.snapshot()
    session.replace_model_definitions(
        current.materials,
        current.sections,
        current.assignments,
        current.steps,
    )

    drawer = AgentChatDrawer(authoring_bridge=bridge)
    drawer.show_applied_patch(record)
    undo = drawer.findChild(
        QToolButton,
        "agentChatPatchUndoButton",
    )

    assert undo is not None and not undo.isEnabled()
    assert not bridge.can_undo_patch(record.patch.patch_id)


def test_a4_result_invalidating_proposal_rejection_keeps_model_unchanged() -> None:
    _application()
    session = _session()
    snapshot = session.snapshot()
    artifact = snapshot.artifact
    run = AnalysisRun(
        "run-a4",
        "作业-旧结果",
        "分析步-旧",
        artifact.artifact_id,
        artifact.model_revision,
        status=RunStatus.SUCCEEDED,
        result_id="result-a4",
    )
    result_snapshot = replace(snapshot, runs=(run,))
    proposal = create_scope_definition_change(
        patch_id="patch-with-result",
        proposal_id="proposal-with-result",
        agent_session_id="agent-a4",
        turn_id="turn-a4",
        source_tool_call_ids=("call-a4",),
        context=authoring_context_from_snapshot(result_snapshot),
        snapshot=result_snapshot,
        draft_revision=4,
        material_function="结构钢",
        material_properties={"E": 210000.0, "nu": 0.3},
        section_function="平面应力",
        plane_type="stress",
        thickness=2.0,
    )
    port = SessionGeometryAuthoringPort(
        session,
        lambda: None,
        apply_definition_delta=lambda _delta: None,
    )
    bridge = AgentAuthoringBridge(port)
    bridge.bind_snapshot(snapshot)
    bridge.register_proposal(proposal)

    receipt = bridge.reject_from_gui_control(proposal.proposal_id)

    assert receipt.state is ProposalState.REJECTED
    assert session.snapshot().session_revision == snapshot.session_revision
    assert not session.snapshot().named_regions
    assert not session.snapshot().materials


def test_a4_automatic_apply_fails_closed_if_port_sees_accepted_result(
    monkeypatch,
) -> None:
    session = _session()
    patch = _change(session)
    snapshot = session.snapshot()
    artifact = snapshot.artifact
    result_snapshot = replace(
        snapshot,
        runs=(
            AnalysisRun(
                "run-a4",
                "作业-旧结果",
                "分析步-旧",
                artifact.artifact_id,
                artifact.model_revision,
                status=RunStatus.SUCCEEDED,
                result_id="result-a4",
            ),
        ),
    )
    port = SessionGeometryAuthoringPort(
        session,
        lambda: None,
        apply_definition_delta=lambda _delta: None,
    )
    bridge = AgentAuthoringBridge(port)
    bridge.bind_snapshot(snapshot)
    monkeypatch.setattr(session, "snapshot", lambda: result_snapshot)

    with pytest.raises(
        AuthoringAuthorizationError,
        match="requires GUI confirmation",
    ):
        bridge.apply_automatic_patch(patch)

    assert session.session_revision == snapshot.session_revision


def test_a4_inverse_id_is_bounded_for_maximum_forward_id() -> None:
    session = _session()
    port = SessionGeometryAuthoringPort(
        session,
        lambda: None,
        apply_definition_delta=lambda _delta: None,
    )
    bridge = AgentAuthoringBridge(port)
    bridge.bind_snapshot(session.snapshot())
    snapshot = session.snapshot()
    original = _change(session)
    patch = ModelPatch.create(
        patch_id="p" * 128,
        agent_session_id=original.agent_session_id,
        turn_id=original.turn_id,
        source_tool_call_ids=original.source_tool_call_ids,
        target_document_id=original.target_document_id,
        target_session_id=original.target_session_id,
        base_session_revision=snapshot.session_revision,
        draft_revision=original.draft_revision,
        operations=original.operations,
        preconditions=original.preconditions,
        expected_changes=original.expected_changes,
        invalidation_impact=original.invalidation_impact,
        display_summary=original.display_summary,
    )

    record = bridge.apply_automatic_patch(patch)

    assert record.inverse_patch.patch_id.startswith("inverse-")
    assert len(record.inverse_patch.patch_id) <= 128


def test_a4_main_window_definition_projection_does_not_rebuild_mesh_actors(
    monkeypatch,
) -> None:
    _application()
    from fem_gui.main_window import FEMMainWindow

    window = FEMMainWindow()
    session: ModelSession = window.session
    session.create_native_project_with_first_part(
        "模型-偏心孔板",
        UnitContext("mm", "N", "MPa"),
        _recipe(),
        part_name="部件-偏心孔板",
    )
    task = session.prepare_agent_mesh_generation(
        "P1",
        MeshSettings(1.0),
        "a" * 64,
        expected_session_revision=session.session_revision,
    )
    session.accept_agent_generated_model(task.token, _plate_model())
    window._rebuild_full_projection()
    window.agent_authoring_bridge.bind_snapshot(session.snapshot())
    patch = _change(session)
    actor_rebuilds: list[object] = []
    monkeypatch.setattr(
        window.viewport,
        "set_model",
        lambda *args, **kwargs: actor_rebuilds.append((args, kwargs)),
    )

    record = window.agent_authoring_bridge.apply_automatic_patch(patch)

    assert actor_rebuilds == []
    assert window.document.session_revision == record.session_revision
    assert window.document.named_regions
    assert window.document.materials
    window.close()
