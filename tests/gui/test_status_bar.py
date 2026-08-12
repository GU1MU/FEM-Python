from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from fem_gui.widgets.status_bar import CAEStatusBar


def _application() -> QApplication:
    return QApplication.instance() or QApplication([])


def test_state_field_is_wide_and_keeps_full_status_as_tooltip() -> None:
    application = _application()
    status_bar = CAEStatusBar()
    status_bar.resize(1200, 22)
    status_bar.show()
    application.processEvents()

    message = "在目标 XY 平面绘制闭合轮廓"
    status_bar.set_state(message)

    assert status_bar.state_label.maximumWidth() == 320
    assert status_bar.state_label.width() > status_bar.selection_label.width() * 2
    assert status_bar.state_label.text() == f"状态：{message}"
    assert status_bar.state_label.toolTip() == message
    assert (
        status_bar.state_label.fontMetrics().horizontalAdvance(
            status_bar.state_label.text()
        )
        < status_bar.state_label.maximumWidth()
    )
    status_bar.close()
