from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path

from PySide6.QtCore import QObject, Signal

from fem.application import ModelSession, SessionDelta, UnitContext
from fem.geometry import RectangleGeometry
from fem_agent.geometry_authoring import GEOMETRY_FEATURE_CATALOG_TOOL_NAME
from fem_gui.agent_authoring import (
    AgentAuthoringBridge,
    SessionGeometryAuthoringPort,
    SessionResultQueryPort,
    create_session_authoring_workflow_controller,
)
from fem_gui.main_window import FEMMainWindow
from fem_gui.task_controller import (
    BackgroundTaskState,
    TaskApplyOutcome,
    TaskCompletion,
)
from fem_agent.result_authoring import AgentResultQueryBridge
from fem_agent.tools.registry import ToolExecutionContext


class ManualTaskController(QObject):
    busy_changed = Signal(bool)
    state_changed = Signal(object)
    cancelling_changed = Signal(bool)
    progress = Signal(int, str)
    completed = Signal(object)
    projection_failed = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self._busy = False
        self._task_id = 0
        self._callbacks: dict[str, object] = {}
        self._cancel_requested = False

    @property
    def busy(self) -> bool:
        return self._busy

    @property
    def cancel_requested(self) -> bool:
        return self._cancel_requested

    @property
    def current_task_name(self) -> str:
        return "phase6-task" if self._busy else ""

    def start(self, workload, *, task_name, apply_result, project_result=None,
              rebuild_projection=None, on_terminal=None, on_progress=None,
              on_projection_error=None):
        del workload, rebuild_projection, on_projection_error
        if self._busy:
            return None
        self._task_id += 1
        self._callbacks = {
            "apply": apply_result,
            "project": project_result,
            "terminal": on_terminal,
            "progress": on_progress,
            "name": task_name,
        }
        self._busy = True
        self.busy_changed.emit(True)
        return self._task_id

    def request_cancel(self, *, after_cleanup=None) -> bool:
        if not self._busy:
            return False
        self._cancel_requested = True
        if after_cleanup is not None:
            self._callbacks["after_cleanup"] = after_cleanup
        return True

    def emit_progress(self, message: str) -> None:
        callback = self._callbacks.get("progress")
        if callback is not None:
            callback(message)

    def finish_success(self, value: object) -> None:
        apply = self._callbacks["apply"]
        outcome = apply(value)
        project = self._callbacks.get("project")
        if outcome.status.value == "accepted" and project is not None:
            project(outcome.projection_value)
        terminal = self._callbacks.get("terminal")
        if terminal is not None:
            terminal(
                TaskCompletion(
                    self._task_id,
                    str(self._callbacks["name"]),
                    BackgroundTaskState.SUCCEEDED,
                    apply_status=outcome.status,
                    value=outcome.projection_value,
                )
            )
        self._busy = False
        self.busy_changed.emit(False)
        callback = self._callbacks.get("after_cleanup")
        if callback is not None:
            callback()

    def finish_failure(self, message: str) -> None:
        terminal = self._callbacks.get("terminal")
        if terminal is not None:
            terminal(
                TaskCompletion(
                    self._task_id,
                    str(self._callbacks["name"]),
                    BackgroundTaskState.FAILED,
                    message=message,
                )
            )
        self._busy = False
        self.busy_changed.emit(False)


def test_document_controller_busy_isolated_and_inactive_completion_updates_one_root(
    monkeypatch,
    dispose_gui_widget,
):
    window = FEMMainWindow()
    try:
        first = window.workspace.active_document()
        assert first is not None
        first.task_controller = ManualTaskController()
        window._bind_task_controller(first)
        second = window.workspace.add_model(
            session=ModelSession(),
            display_name="Model-B",
            source_path=Path("phase6-b.inp"),
            task_controller=ManualTaskController(),
        )
        assert window._activate_workspace_context(first)

        projected: list[object] = []
        viewport_calls: list[str] = []
        ribbon_calls: list[str] = []
        combo_calls: list[str] = []
        status_calls: list[str] = []
        for name in ("render", "fit", "set_model", "clear_model"):
            monkeypatch.setattr(
                window.viewport,
                name,
                lambda *args, _name=name, **kwargs: viewport_calls.append(_name),
            )
        monkeypatch.setattr(
            window.ribbon,
            "set_current",
            lambda *args, **kwargs: ribbon_calls.append("ribbon"),
        )
        monkeypatch.setattr(
            window.status_panel,
            "set_state",
            lambda *args, **kwargs: status_calls.append("status"),
        )
        for combo in (window.result_variable_combo, window.result_component_combo,
                      window.result_position_combo):
            monkeypatch.setattr(
                combo,
                "setEnabled",
                lambda *args, **kwargs: combo_calls.append("combo"),
            )

        task = first.session.new_native_project("A")
        assert window._start_task(
            lambda _context: None,
            projected.append,
            "phase6",
            apply_result=lambda value: TaskApplyOutcome.accepted(task),
            controller=first.task_controller,
            context=first,
        )
        assert first.task_controller.busy
        assert not second.task_controller.busy
        assert window.busy

        assert window._activate_workspace_context(second)
        assert not window.busy
        assert first.task_controller.busy
        viewport_calls.clear()
        ribbon_calls.clear()
        combo_calls.clear()
        status_calls.clear()
        first.task_controller.emit_progress("A progress")
        assert projected == []
        first.task_controller.finish_success(task)

        assert projected == []
        assert first.projection.session_revision == first.session.session_revision
        assert viewport_calls == []
        assert ribbon_calls == []
        assert combo_calls == []
        assert status_calls == []
        assert first.document_id in window.model_tree._roots
        assert second.document_id in window.model_tree._roots
    finally:
        dispose_gui_widget(window)


def test_agent_runtime_busy_blocks_activation_and_idle_rebinds_target(
    monkeypatch,
    dispose_gui_widget,
):
    window = FEMMainWindow()
    try:
        first = window.workspace.active_document()
        assert first is not None
        second = window.workspace.add_model(
            session=ModelSession(),
            display_name="Model-B",
            source_path=Path("phase6-agent-b.inp"),
        )
        runtime = window.viewport_panel.agent_chat_drawer.agent_runtime
        with runtime._lock:
            runtime._busy = True
        assert not window._activate_workspace_context(second)
        assert window.workspace.active_document() is first
        with runtime._lock:
            runtime._busy = False
        assert window._activate_workspace_context(second)
        assert runtime.target_identity == (
            str(second.document_id),
            second.session.session_id,
        )
        assert window.agent_authoring_bridge.context is not None
        assert window.agent_authoring_bridge.context.binding.document_id == str(
            second.document_id
        )
    finally:
        dispose_gui_widget(window)


def test_rebindable_authoring_and_result_ports_require_idle_callback():
    first = ModelSession()
    second = ModelSession()
    result_port = SessionResultQueryPort(first)
    result_port.bind_session(second, idle=lambda: True)
    assert result_port.session is second
    try:
        result_port.bind_session(first, idle=lambda: False)
    except RuntimeError:
        pass
    else:
        raise AssertionError("busy Agent rebinding must be rejected")

    first.create_native_project_with_first_part(
        "First",
        UnitContext("mm", "N", "MPa"),
        RectangleGeometry("First-Part", 1.0, 1.0),
    )
    second.create_native_project_with_first_part(
        "Second",
        UnitContext("mm", "N", "MPa"),
        RectangleGeometry("Second-Part", 2.0, 2.0),
    )
    second.add_native_part(
        RectangleGeometry("Second-Extra", 1.0, 1.0),
    )
    authoring_port = SessionGeometryAuthoringPort(
        first,
        lambda: None,
    )
    bridge = AgentAuthoringBridge(authoring_port)
    bridge.bind_snapshot(first.projection_snapshot(), document_id="phase6-a")
    controller = create_session_authoring_workflow_controller(
        first,
        bridge,
        AgentResultQueryBridge(result_port),
    )
    authoring_port.bind_session(second, idle=lambda: True)
    bridge.bind_snapshot(second.projection_snapshot(), document_id="phase6-b")
    assert bridge.context is not None
    controller.reset_for_binding()
    controller.observe_binding(bridge.context)

    catalog = controller.dispatch(
        GEOMETRY_FEATURE_CATALOG_TOOL_NAME,
        {},
        ToolExecutionContext("phase6", 0, "rebound-catalog"),
    )

    assert catalog.ok
    assert catalog.data["session_revision"] == second.session_revision
    assert [part["part_id"] for part in catalog.data["parts"]] == ["P1", "P2"]


def test_detached_controller_does_not_capture_active_document(
    dispose_gui_widget,
):
    window = FEMMainWindow()
    try:
        active = window.workspace.active_document()
        assert active is not None
        detached = ManualTaskController()
        projected: list[object] = []
        assert window._start_task(
            lambda _context: None,
            projected.append,
            "detached",
            controller=detached,
        )
        assert detached.busy
        assert not active.task_controller.busy
        detached.finish_success("opened")
        assert projected == ["opened"]
    finally:
        dispose_gui_widget(window)


def test_inactive_failure_uses_only_target_state_callback(
    monkeypatch,
    dispose_gui_widget,
):
    window = FEMMainWindow()
    try:
        first = window.workspace.active_document()
        assert first is not None
        first.task_controller = ManualTaskController()
        second = window.workspace.add_model(
            session=ModelSession(),
            display_name="Model-B",
        )
        visible_failures: list[str] = []
        target_failures: list[str] = []
        errors: list[str] = []
        statuses: list[str] = []
        monkeypatch.setattr(
            window,
            "_show_error",
            lambda *_args: errors.append("error"),
        )
        monkeypatch.setattr(
            window.status_panel,
            "set_state",
            lambda *_args, **_kwargs: statuses.append("status"),
        )
        assert window._start_task(
            lambda _context: None,
            lambda _value: None,
            "phase6",
            visible_failures.append,
            on_inactive_failure=target_failures.append,
            controller=first.task_controller,
            context=first,
        )
        assert window._activate_workspace_context(second)
        errors.clear()
        statuses.clear()
        first.task_controller.finish_failure("failed")

        assert visible_failures == []
        assert target_failures == ["failed"]
        assert errors == []
        assert statuses == []
    finally:
        dispose_gui_widget(window)
