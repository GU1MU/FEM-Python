"""通用只读对象信息弹窗。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)


class CompactDoubleSpinBox(QDoubleSpinBox):
    """Keep numeric precision while hiding insignificant trailing zeroes."""

    def __init__(self, parent=None, *, minimum_display_decimals: int = 2) -> None:
        super().__init__(parent)
        self._minimum_display_decimals = max(
            0,
            int(minimum_display_decimals),
        )

    def textFromValue(self, value: float) -> str:
        text = super().textFromValue(value)
        decimal_point = self.locale().decimalPoint()
        if decimal_point not in text:
            return text
        whole, fraction = text.split(decimal_point, 1)
        fraction = fraction.rstrip("0")
        minimum = min(self._minimum_display_decimals, self.decimals())
        fraction = fraction.ljust(minimum, "0")
        return whole if not fraction else f"{whole}{decimal_point}{fraction}"


def configure_form_layout(form: QFormLayout) -> None:
    """Apply the compact alignment shared by modal parameter dialogs."""
    form.setLabelAlignment(
        Qt.AlignmentFlag.AlignRight
        | Qt.AlignmentFlag.AlignVCenter
    )
    form.setFormAlignment(Qt.AlignmentFlag.AlignTop)
    form.setHorizontalSpacing(12)
    form.setVerticalSpacing(8)


def normalize_dialog_message(value: object) -> str:
    """Remove punctuation excluded by the shared dialog copy style."""

    return str(value).replace("：", " ").replace("。", "；").rstrip("；")


def show_information(
    parent: QWidget,
    title: str,
    rows: Sequence[tuple[str, object]],
) -> None:
    """显示一个只读、可复制的对象信息窗口。"""
    dialog = QDialog(parent)
    dialog.setWindowTitle(title)
    dialog.setMinimumWidth(330)
    layout = QVBoxLayout(dialog)
    form = QFormLayout()
    configure_form_layout(form)
    for name, value in rows:
        label = QLabel(_format_value(value), dialog)
        label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        label.setWordWrap(True)
        form.addRow(str(name), label)
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
