from __future__ import annotations

from time import monotonic

from PySide6.QtCore import QThread
from PySide6.QtWidgets import QApplication

from fem_gui.main_window import FEMMainWindow


def wait_for_analysis_task(window: FEMMainWindow, application: QApplication) -> None:
    """Process GUI events until an already-running analysis task completes."""
    controller = window.task_controller
    assert controller.busy
    deadline = monotonic() + 2.0
    while controller.busy and monotonic() < deadline:
        application.processEvents()
        QThread.msleep(1)
    application.processEvents()
    assert not controller.busy
