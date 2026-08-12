from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
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


@pytest.mark.parametrize(
    ("mode", "label"),
    (
        ("geometry_point", "点"),
        ("geometry_edge", "边"),
        ("geometry_face", "面"),
        ("geometry_body", "体"),
        ("mesh_edge", "边"),
        ("mesh_face", "面"),
        ("mesh_body", "体"),
    ),
)
def test_selection_field_uses_semantic_entity_names(
    mode: str,
    label: str,
) -> None:
    _application()
    status_bar = CAEStatusBar()

    status_bar.set_selection_mode(mode)

    assert status_bar.selection_label.text() == f"选择：{label}"
    status_bar.close()


def test_object_field_is_wider_than_selection_field() -> None:
    application = _application()
    status_bar = CAEStatusBar()
    status_bar.resize(1200, 22)
    status_bar.show()
    application.processEvents()

    assert status_bar.selection_label.maximumWidth() == 90
    assert status_bar.object_label.maximumWidth() == 130
    assert status_bar.object_label.width() > status_bar.selection_label.width()
    status_bar.close()
