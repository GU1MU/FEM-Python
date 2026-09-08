from __future__ import annotations

import gc
import os
from time import monotonic

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("FEM_GUI_OFFSCREEN", "1")

import pytest

from tests.helpers.file_builders import write_inp


_gui_cleanup_count = 0
_GUI_TEARDOWN_TIMEOUT_SECONDS = 2.0
_quarantined_gui_widgets: list[object] = []


@pytest.fixture(scope="session")
def _gui_application():
    """Keep exactly one QApplication wrapper alive for the GUI test session."""

    from PySide6.QtWidgets import QApplication

    application = QApplication.instance()
    if application is None:
        application = QApplication([])
    application.setQuitOnLastWindowClosed(False)
    yield application


@pytest.fixture
def gui_application(gui_runtime):
    """Opt into Qt setup and per-test cleanup while reusing the session app."""
    return gui_runtime


@pytest.fixture
def dispose_gui_widget(gui_application):
    """Destroy a closed Qt window before the test constructs its replacement."""

    from PySide6.QtCore import QEvent

    def dispose(widget) -> None:
        widget.hide()
        assert widget.close()
        widget.deleteLater()
        gui_application.sendPostedEvents(None, QEvent.Type.DeferredDelete)

    return dispose


@pytest.fixture
def gui_runtime(monkeypatch, _gui_application):
    """Guard modal loops and release GUI windows/tasks after each opted-in test."""

    from PySide6.QtCore import QEvent, QThread
    from PySide6.QtWidgets import QDialog, QFileDialog, QInputDialog, QMessageBox

    unstubbed_modals: list[str] = []

    def reject_dialog(*args, **_kwargs):
        dialog_name = type(args[0]).__name__ if args else "QDialog"
        unstubbed_modals.append(f"{dialog_name}.exec")
        return QDialog.DialogCode.Rejected

    def reject_message_box(method: str):
        def reject(*_args, **_kwargs):
            unstubbed_modals.append(f"QMessageBox.{method}")
            return QMessageBox.StandardButton.NoButton

        return reject

    def reject_file_dialog(method: str):
        def reject(*_args, **_kwargs):
            unstubbed_modals.append(f"QFileDialog.{method}")
            return "", ""

        return reject

    def reject_input_dialog(method: str):
        def reject(*_args, **_kwargs):
            unstubbed_modals.append(f"QInputDialog.{method}")
            if method == "getDouble":
                return 0.0, False
            if method == "getInt":
                return 0, False
            return "", False

        return reject

    monkeypatch.setattr(QDialog, "exec", reject_dialog)
    monkeypatch.setattr(QMessageBox, "exec", reject_dialog)
    for method in (
        "critical",
        "information",
        "question",
        "warning",
    ):
        monkeypatch.setattr(QMessageBox, method, reject_message_box(method))
    for method in ("getOpenFileName", "getSaveFileName"):
        monkeypatch.setattr(QFileDialog, method, reject_file_dialog(method))
    for method in ("getDouble", "getInt", "getItem", "getText"):
        monkeypatch.setattr(QInputDialog, method, reject_input_dialog(method))

    yield _gui_application

    global _gui_cleanup_count

    application = _gui_application
    application.processEvents()
    widgets = tuple(
        widget
        for widget in application.topLevelWidgets()
        if widget.parentWidget() is None
    )

    widget_services: list[tuple[object, object | None, object | None]] = []
    for widget in widgets:
        viewport_panel = getattr(widget, "viewport_panel", None)
        overlay_host = getattr(viewport_panel, "overlay_host", None)
        if overlay_host is None and hasattr(widget, "agent_chat_drawer"):
            overlay_host = widget
        drawer = getattr(overlay_host, "agent_chat_drawer", None)
        runtime = getattr(widget, "agent_runtime", None)
        if runtime is None:
            runtime = getattr(drawer, "agent_runtime", None)
        widget_services.append((widget, overlay_host, runtime))

    controllers = tuple(
        controller
        for widget in widgets
        if (controller := getattr(widget, "task_controller", None)) is not None
    )
    for controller in controllers:
        if bool(getattr(controller, "busy", False)):
            controller.request_cancel()
    for widget, overlay_host, _runtime in widget_services:
        shutdown_overlay = getattr(overlay_host, "shutdown", None)
        if callable(shutdown_overlay):
            shutdown_overlay(wait=False)
        shutdown_runtime = getattr(widget, "shutdown_runtime", None)
        if callable(shutdown_runtime):
            shutdown_runtime(wait=False)

    deadline = monotonic() + _GUI_TEARDOWN_TIMEOUT_SECONDS
    while (
        (
            any(
                bool(getattr(controller, "busy", False))
                for controller in controllers
            )
            or any(
                bool(getattr(runtime, "busy", False))
                for _widget, _overlay, runtime in widget_services
                if runtime is not None
            )
        )
        and monotonic() < deadline
    ):
        application.processEvents()
        QThread.msleep(1)
    application.processEvents()

    stuck_widgets = tuple(
        widget
        for widget, _overlay, runtime in widget_services
        if (
            bool(getattr(getattr(widget, "task_controller", None), "busy", False))
            or bool(getattr(runtime, "busy", False))
        )
    )
    for widget, _overlay_host, _runtime in widget_services:
        stuck = widget in stuck_widgets
        widget.hide()
        if stuck:
            if widget not in _quarantined_gui_widgets:
                _quarantined_gui_widgets.append(widget)
            continue
        widget.close()
        viewport = getattr(widget, "viewport", None)
        shutdown_backend = getattr(viewport, "shutdown_backend", None)
        if callable(shutdown_backend):
            shutdown_backend()
        if widget in _quarantined_gui_widgets:
            _quarantined_gui_widgets.remove(widget)
        widget.deleteLater()
    application.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    application.processEvents()
    application.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    _gui_cleanup_count += 1
    gc.collect(0)
    if _gui_cleanup_count % 50 == 0:
        gc.collect()
    teardown_failures: list[str] = []
    if stuck_widgets:
        names = ", ".join(
            str(getattr(widget, "objectName", lambda: "")())
            or type(widget).__name__
            for widget in stuck_widgets
        )
        teardown_failures.append(
            "GUI teardown timed out while background tasks were still running: "
            f"{names}"
        )
    if unstubbed_modals:
        teardown_failures.append(
            "GUI tests must stub modal dialogs before opening them: "
            + ", ".join(unstubbed_modals)
        )
    if teardown_failures:
        pytest.fail("; ".join(teardown_failures))


@pytest.fixture
def gui_inp_path(tmp_path):
    """生成可稳定求解的小型平面应力验收模型。"""
    return write_inp(
        tmp_path,
        "gui_plate.inp",
        [
            "*Heading",
            "GUI plate",
            "*Node",
            "1, 0., 0.",
            "2, 1., 0.",
            "3, 1., 1.",
            "4, 0., 1.",
            "*Element, type=CPS4, elset=SOLID",
            "1, 1, 2, 3, 4",
            "*Nset, nset=LEFT",
            "1, 4",
            "*Nset, nset=RIGHT",
            "2, 3",
            "*Elset, elset=SOLID",
            "1",
            "*Material, name=STEEL",
            "*Elastic",
            "210000., 0.3",
            "*Solid Section, elset=SOLID, material=STEEL",
            "*Boundary",
            "LEFT, 1, 2, 0.",
            "*Step, name=Static-1",
            "*Static",
            "*Cload",
            "RIGHT, 1, 10.",
            "*Output, field",
            "*Node Output",
            "U, RF",
            "*Element Output",
            "S",
            "*End Step",
        ],
    )
