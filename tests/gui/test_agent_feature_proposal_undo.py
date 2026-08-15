from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QToolButton

from fem.application import ModelSession, UnitContext
from fem.geometry import (
    BooleanGeometry,
    DiskGeometry,
    MovedGeometry,
    RectangleGeometry,
)
from fem_agent.geometry_authoring import (
    create_geometry_edit_proposal,
    geometry_draft,
)
from fem_gui.agent_authoring import (
    AgentAuthoringBridge,
    AppliedPatchState,
    SessionGeometryAuthoringPort,
    authoring_context_from_snapshot,
)
from fem_gui.agent_events import ProposalView, ProposalViewStatus
from fem_gui.widgets.agent_chat import AgentChatDrawer


def _application() -> QApplication:
    return QApplication.instance() or QApplication([])


def _accepted_feature_proposal():
    session = ModelSession()
    base = RectangleGeometry("Plate", 10.0, 4.0)
    session.create_native_project_with_first_part(
        "模型-特征撤销",
        UnitContext("mm", "N", "MPa"),
        base,
        part_name="部件-板",
    )
    cut = BooleanGeometry(
        "Cut",
        "cut",
        base,
        MovedGeometry(DiskGeometry("Hole", 0.5), 5.0, 2.0),
    )
    proposal = create_geometry_edit_proposal(
        proposal_id="proposal-feature-undo",
        agent_session_id="agent-feature-undo",
        turn_id="turn-feature-undo",
        source_tool_call_ids=("call-feature-undo",),
        context=authoring_context_from_snapshot(session.snapshot(), document_id=1),
        draft_revision=1,
        part_id="P1",
        draft=geometry_draft(cut),
        summary="在板上切除圆孔",
    )
    refreshes: list[str] = []
    port = SessionGeometryAuthoringPort(
        session,
        lambda: refreshes.append("refresh"),
    )
    bridge = AgentAuthoringBridge(port)
    bridge.bind_snapshot(session.snapshot(), document_id=1)
    bridge.register_proposal(proposal)
    bridge.accept_from_gui_control(proposal.proposal_id)
    return session, base, proposal, port, bridge, refreshes


def test_feature_proposal_undo_restores_exact_pre_feature_recipe_once() -> None:
    session, base, proposal, port, bridge, refreshes = _accepted_feature_proposal()

    assert bridge.can_undo_proposal(proposal.proposal_id)
    record = bridge.undo_proposal_from_gui_control(proposal.proposal_id)

    assert record.state is AppliedPatchState.UNDONE
    assert record.feature_name == "Cut-1"
    assert session.snapshot().parts[0].geometry_recipe == base
    assert refreshes == ["refresh", "refresh"]
    assert not bridge.can_undo_proposal(proposal.proposal_id)
    assert port.feature_proposal_record(proposal.proposal_id).state is (
        AppliedPatchState.UNDONE
    )


def test_later_geometry_revision_disables_old_feature_proposal_undo() -> None:
    session, _base, proposal, _port, bridge, _refreshes = (
        _accepted_feature_proposal()
    )
    current = session.snapshot().parts[0].geometry_recipe
    session.replace_part_geometry(
        "P1",
        MovedGeometry(current, 1.0, 0.0),
        expected_session_revision=session.session_revision,
    )

    assert not bridge.can_undo_proposal(proposal.proposal_id)


def test_plain_recipe_replacement_is_not_mislabeled_as_feature_undo() -> None:
    session = ModelSession()
    session.create_native_project_with_first_part(
        "模型-参数修改",
        UnitContext("mm", "N", "MPa"),
        RectangleGeometry("Plate", 10.0, 4.0),
    )
    proposal = create_geometry_edit_proposal(
        proposal_id="proposal-plain-replacement",
        agent_session_id="agent-feature-undo",
        turn_id="turn-plain-replacement",
        source_tool_call_ids=("call-plain-replacement",),
        context=authoring_context_from_snapshot(session.snapshot(), document_id=1),
        draft_revision=1,
        part_id="P1",
        draft=geometry_draft(RectangleGeometry("Plate", 12.0, 4.0)),
        summary="调整板长",
    )
    bridge = AgentAuthoringBridge(
        SessionGeometryAuthoringPort(session, lambda: None)
    )
    bridge.bind_snapshot(session.snapshot(), document_id=1)
    bridge.register_proposal(proposal)
    bridge.accept_from_gui_control(proposal.proposal_id)

    assert bridge.feature_proposal_record(proposal.proposal_id) is None
    assert not bridge.can_undo_proposal(proposal.proposal_id)


def test_completed_modify_part_card_exposes_feature_undo_button() -> None:
    application = _application()
    session, base, proposal, _port, bridge, _refreshes = (
        _accepted_feature_proposal()
    )
    drawer = AgentChatDrawer(authoring_bridge=bridge)
    drawer._add_proposal_card(
        ProposalView(
            proposal.proposal_id,
            proposal.proposal_hash,
            proposal.proposal_kind.value,
            "修改部件",
            "在板上切除圆孔",
            "网格与结果需要更新",
            "确认修改",
            proposal.target_document_id,
            proposal.target_session_id,
            proposal.base_session_revision,
            status=ProposalViewStatus.SUCCEEDED,
        ),
        proposal.turn_id,
    )
    undo = drawer.findChild(QToolButton, "agentChatProposalUndoButton")

    assert undo is not None and undo.isEnabled()
    assert undo.text() == "撤销"
    QTest.mouseClick(undo, Qt.MouseButton.LeftButton)
    application.processEvents()

    assert session.snapshot().parts[0].geometry_recipe == base
    assert not bridge.can_undo_proposal(proposal.proposal_id)
    drawer.close()
