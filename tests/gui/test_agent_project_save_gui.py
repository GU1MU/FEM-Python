from __future__ import annotations

import json
import os
import time
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QToolButton

from fem.application import NativePart
from fem.geometry import SketchGeometry, SketchRectangle
from fem_agent.authoring import ProposalState
from fem_agent.authoring_runtime import AuthoringWorkflowStage
from fem_agent.tools.registry import ToolExecutionContext
from fem_gui import main_window as main_window_module
from fem_gui.agent_events import ProposalView, ProposalViewStatus
from fem_gui.main_window import FEMMainWindow


def _application() -> QApplication:
    return QApplication.instance() or QApplication([])


def _wait_until(predicate, *, timeout_ms: int = 2_000) -> None:
    deadline = time.monotonic() + timeout_ms / 1000
    application = _application()
    while not predicate() and time.monotonic() < deadline:
        application.processEvents()
        QTest.qWait(1)
    application.processEvents()
    assert predicate()


def _native_window() -> FEMMainWindow:
    _application()
    window = FEMMainWindow()
    window._set_native_geometry(
        SketchGeometry(
            "Accepted plate",
            (
                SketchRectangle(
                    "material",
                    0.0,
                    0.0,
                    2.0,
                    1.0,
                ),
            ),
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


def _request_save(
    window: FEMMainWindow,
    *,
    index: int,
) -> ProposalView:
    controller = window.agent_authoring_controller
    result = controller.dispatch(
        "request_project_save",
        {},
        ToolExecutionContext(
            "agent-project-save",
            0,
            f"project-save-{index}",
        ),
    )
    assert result.ok
    view = result.data["proposal_view"]
    assert isinstance(view, dict)
    encoded = json.dumps(result.data, ensure_ascii=False)
    assert "project_path" not in encoded
    assert "source_path" not in encoded
    proposal = ProposalView(**view)
    window.viewport_panel.agent_chat_drawer._add_proposal_card(
        proposal,
        f"turn-save-{index}",
    )
    _application().processEvents()
    return proposal


def _proposal_button(
    window: FEMMainWindow,
    proposal: ProposalView,
    object_name: str = "agentChatProposalAcceptButton",
) -> QToolButton:
    matches = [
        button
        for button in window.viewport_panel.agent_chat_drawer.findChildren(
            QToolButton,
            object_name,
        )
        if button.property("proposalId") == proposal.proposal_id
    ]
    return matches[-1]


def _click_accept(
    window: FEMMainWindow,
    proposal: ProposalView,
) -> None:
    button = _proposal_button(window, proposal)
    assert button.isEnabled()
    button.click()
    _application().processEvents()


def _change_accepted_geometry(window: FEMMainWindow, width: float) -> None:
    changed = SketchGeometry(
        f"Accepted plate {width}",
        (
            SketchRectangle(
                "material",
                0.0,
                0.0,
                width,
                1.0,
            ),
        ),
    )
    delta = window.session.replace_native_geometry_inputs(
        (NativePart(),),
        changed,
        expected_session_revision=window.document.session_revision,
    )
    window._accepted_command(window._next_command_id(), delta)


def test_agent_gui_save_uses_save_as_then_reuses_existing_path(
    tmp_path,
    monkeypatch,
) -> None:
    window = _native_window()
    target = tmp_path / "agent-model.femproj"
    dialog_calls: list[str] = []
    monkeypatch.setattr(
        main_window_module.QFileDialog,
        "getSaveFileName",
        lambda *_args, **_kwargs: (
            dialog_calls.append("save-as") or str(target),
            "",
        ),
    )
    real_save_native_project = window.save_native_project
    save_command_calls: list[dict[str, object]] = []

    def save_spy(**kwargs):
        save_command_calls.append(dict(kwargs))
        return real_save_native_project(**kwargs)

    monkeypatch.setattr(window, "save_native_project", save_spy)
    controller = window.agent_authoring_controller
    controller.dispatch(
        "set_authoring_requirements",
        {
            "turn_id": "turn-private-draft",
            "requirements": {
                "length_unit": "secret-draft-unit",
            },
        },
        ToolExecutionContext("agent-project-save", 0, "draft-key"),
    )

    first = _request_save(window, index=1)
    _click_accept(window, first)
    _wait_until(
        lambda: (
            controller.project_save_record is not None
            and controller.project_save_record.state
            is ProposalState.SUCCEEDED
            and not window.busy
        )
    )

    output = target.with_suffix(".fempy")
    assert output.is_file()
    assert window.document.project_path == output
    assert not window.document.dirty
    assert dialog_calls == ["save-as"]
    assert len(save_command_calls) == 1
    assert "secret-draft-unit" not in output.read_text(encoding="utf-8")
    assert controller.stage is AuthoringWorkflowStage.MESH_READY

    _change_accepted_geometry(window, 3.0)
    controller.reset_for_binding()
    assert window.agent_authoring_bridge.context is not None
    controller.observe_binding(window.agent_authoring_bridge.context)
    second = _request_save(window, index=2)
    _click_accept(window, second)
    _wait_until(
        lambda: (
            controller.project_save_record is not None
            and controller.project_save_record.state
            is ProposalState.SUCCEEDED
            and not window.busy
        )
    )

    assert dialog_calls == ["save-as"]
    assert len(save_command_calls) == 2
    assert not window.document.dirty
    assert controller.stage is AuthoringWorkflowStage.MESH_READY
    window.close()


def test_agent_gui_save_cancel_failure_and_reject_are_terminal(
    tmp_path,
    monkeypatch,
) -> None:
    window = _native_window()
    controller = window.agent_authoring_controller
    monkeypatch.setattr(
        main_window_module.QFileDialog,
        "getSaveFileName",
        lambda *_args, **_kwargs: ("", ""),
    )

    cancelled = _request_save(window, index=3)
    _click_accept(window, cancelled)
    assert controller.project_save_record.state is ProposalState.CANCELLED
    assert cancelled.status is ProposalViewStatus.CANCELLED
    assert controller.stage is AuthoringWorkflowStage.MESH_READY

    rejected = _request_save(window, index=4)
    reject = _proposal_button(
        window,
        rejected,
        "agentChatProposalRejectButton",
    )
    reject.click()
    assert controller.project_save_record.state is ProposalState.REJECTED
    assert rejected.status is ProposalViewStatus.REJECTED

    target = tmp_path / "failed.femproj"
    monkeypatch.setattr(
        main_window_module.QFileDialog,
        "getSaveFileName",
        lambda *_args, **_kwargs: (str(target), ""),
    )
    monkeypatch.setattr(
        main_window_module,
        "save_project",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            OSError("private path must remain local")
        ),
    )
    monkeypatch.setattr(window, "_show_error", lambda *_args: None)
    failed = _request_save(window, index=5)
    _click_accept(window, failed)
    _wait_until(lambda: not window.busy)

    assert (
        failed.status is ProposalViewStatus.FAILED
    ), controller.project_save_record
    assert controller.project_save_record.message == "保存自主项目失败"
    assert not target.exists()
    assert controller.stage is AuthoringWorkflowStage.MESH_READY
    window.close()
