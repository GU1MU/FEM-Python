from __future__ import annotations

import json
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QToolButton

from fem.geometry import SketchGeometry, SketchRectangle
from fem_agent.authoring_runtime import AuthoringWorkflowStage
from fem_agent.tools.registry import ToolExecutionContext
from fem_gui.agent_events import ProposalView
from fem_gui.main_window import FEMMainWindow


def _application() -> QApplication:
    return QApplication.instance() or QApplication([])


def _native_window() -> FEMMainWindow:
    _application()
    window = FEMMainWindow()
    window._set_native_geometry(
        SketchGeometry(
            "Accepted plate",
            (SketchRectangle("material", 0.0, 0.0, 2.0, 1.0),),
        ),
        "草图",
    )
    controller = window.agent_authoring_controller
    controller.reset_for_binding()
    assert window.agent_authoring_bridge.context is not None
    controller.observe_binding(window.agent_authoring_bridge.context)
    window._confirm_workspace_context_close = lambda *_args, **_kwargs: True
    window.show()
    window.viewport_panel.agent_chat_drawer.show()
    _application().processEvents()
    return window


def test_delete_part_runs_only_from_the_unique_gui_confirmation() -> None:
    window = _native_window()
    controller = window.agent_authoring_controller
    tool_context = ToolExecutionContext(
        "agent-delete-gui",
        0,
        "delete-part-gui",
    )
    catalog_result = controller.dispatch(
        "read_deletable_objects",
        {},
        tool_context,
    )
    part = next(
        item
        for item in catalog_result.data["objects"]
        if item["object_type"] == "part"
    )
    proposal_result = controller.dispatch(
        "prepare_delete_proposal",
        {
            "object_type": "part",
            "target_id": part["target_id"],
        },
        tool_context,
    )
    proposal = ProposalView(**proposal_result.data["proposal_view"])
    drawer = window.viewport_panel.agent_chat_drawer
    drawer._add_proposal_card(proposal, "turn-delete-part")
    _application().processEvents()
    buttons = [
        button
        for button in drawer.findChildren(
            QToolButton,
            "agentChatProposalAcceptButton",
        )
        if button.property("proposalId") == proposal.proposal_id
    ]
    button = buttons[-1]

    assert catalog_result.ok and proposal_result.ok
    assert controller.stage is AuthoringWorkflowStage.DESTRUCTIVE_EDIT_PENDING
    assert len(window.session.snapshot().parts) == 1
    assert "path" not in json.dumps(proposal_result.data, ensure_ascii=False)
    assert button.isEnabled()

    button.click()
    _application().processEvents()

    assert window.session.snapshot().parts == ()
    assert controller.stage is AuthoringWorkflowStage.REQUIREMENTS
    current_buttons = [
        current
        for current in drawer.findChildren(
            QToolButton,
            "agentChatProposalAcceptButton",
        )
        if current.property("proposalId") == proposal.proposal_id
    ]
    assert not current_buttons or not current_buttons[-1].isEnabled()
    window.close()
