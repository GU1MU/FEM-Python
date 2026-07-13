"""通用只读对象信息弹窗。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)


def show_information(
    parent: QWidget,
    title: str,
    rows: Sequence[tuple[str, object]],
) -> None:
    """显示一个只读、可复制的对象信息窗口。"""
    dialog = QDialog(parent)
    dialog.setWindowTitle(title)
    dialog.setMinimumWidth(460)
    layout = QVBoxLayout(dialog)
    form = QFormLayout()
    form.setHorizontalSpacing(22)
    form.setVerticalSpacing(8)
    for name, value in rows:
        label = QLabel(_format_value(value), dialog)
        label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        label.setWordWrap(True)
        form.addRow(f"{name}：", label)
    layout.addLayout(form)
    buttons = QDialogButtonBox(
        QDialogButtonBox.StandardButton.Close,
        parent=dialog,
    )
    buttons.button(QDialogButtonBox.StandardButton.Close).setText("关闭")
    buttons.rejected.connect(dialog.reject)
    layout.addWidget(buttons)
    dialog.exec()


def _format_value(value: object) -> str:
    if value is None or value == "":
        return "无"
    if isinstance(value, Mapping):
        return "；".join(f"{key}={item}" for key, item in value.items()) or "无"
    if isinstance(value, (list, tuple, set)):
        return "，".join(str(item) for item in value) or "无"
    return str(value)
