"""通用只读对象信息弹窗。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from PySide6.QtCore import QLocale, Qt
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


class AdaptivePrecisionDoubleSpinBox(QDoubleSpinBox):
    """Show compact defaults while preserving precision typed by the user."""

    def __init__(
        self,
        parent=None,
        *,
        input_decimals: int = 12,
        default_display_decimals: int = 2,
    ) -> None:
        self._user_decimals: int | None = None
        self._default_display_decimals = max(
            0,
            int(default_display_decimals),
        )
        super().__init__(parent)
        self.setDecimals(max(0, int(input_decimals)))
        self.lineEdit().textEdited.connect(self._remember_user_precision)

    def setValue(self, value: float) -> None:
        self._user_decimals = None
        super().setValue(value)

    def stepBy(self, steps: int) -> None:
        self._user_decimals = None
        super().stepBy(steps)

    def textFromValue(self, value: float) -> str:
        decimals = (
            min(self._default_display_decimals, self.decimals())
            if self._user_decimals is None
            else min(self._user_decimals, self.decimals())
        )
        locale = QLocale(self.locale())
        locale.setNumberOptions(
            locale.numberOptions() | QLocale.NumberOption.OmitGroupSeparator
        )
        text = locale.toString(float(value), "f", decimals)
        if self._user_decimals is not None:
            return text
        decimal_point = self.locale().decimalPoint()
        if decimal_point not in text:
            return text
        return text.rstrip("0").rstrip(decimal_point)

    def _remember_user_precision(self, text: str) -> None:
        decimal_point = self.locale().decimalPoint()
        if decimal_point not in text:
            self._user_decimals = 0
            return
        fraction = text.split(decimal_point, 1)[1]
        self._user_decimals = min(
            len(fraction.rstrip(self.suffix())),
            self.decimals(),
        )


def configure_form_layout(form: QFormLayout) -> None:
    """Apply the compact alignment shared by modal parameter dialogs."""
    form.setLabelAlignment(
        Qt.AlignmentFlag.AlignRight
        | Qt.AlignmentFlag.AlignVCenter
    )
    form.setFormAlignment(Qt.AlignmentFlag.AlignTop)
    form.setHorizontalSpacing(12)
    form.setVerticalSpacing(8)


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
