"""通用只读对象信息弹窗。"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from PySide6.QtCore import QEvent, QLocale, Qt
from PySide6.QtGui import QValidator
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)


def _scientific_text(value: float, significant_digits: int = 4) -> str:
    """Format a nonzero value that would disappear in fixed notation."""

    places = max(1, int(significant_digits) - 1)
    mantissa, exponent = f"{float(value):.{places}E}".split("E")
    return f"{mantissa}e{int(exponent):+d}"


def _numeric_text_value(widget: QDoubleSpinBox, text: str) -> float | None:
    normalized = str(text).strip()
    suffix = widget.suffix().strip()
    if suffix and normalized.endswith(suffix):
        normalized = normalized[: -len(suffix)].rstrip()
    normalized = normalized.replace(widget.locale().groupSeparator(), "")
    normalized = normalized.replace(widget.locale().decimalPoint(), ".")
    try:
        value = float(normalized)
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def _insert_scientific_key(widget: QDoubleSpinBox, event: object) -> bool:
    text = str(getattr(event, "text", lambda: "")())
    current = widget.lineEdit().text().casefold()
    if text.casefold() == "e" or (
        text in {"+", "-"} and "e" in current
    ):
        widget.lineEdit().insert(text)
        return True
    return False


class CompactDoubleSpinBox(QDoubleSpinBox):
    """Keep numeric precision while hiding insignificant trailing zeroes."""

    def __init__(
        self,
        parent=None,
        *,
        minimum_display_decimals: int = 2,
        display_decimals: int | None = None,
    ) -> None:
        super().__init__(parent)
        self.lineEdit().installEventFilter(self)
        self._minimum_display_decimals = max(
            0,
            int(minimum_display_decimals),
        )
        self._display_decimals = (
            None
            if display_decimals is None
            else max(0, int(display_decimals))
        )
        self.lineEdit().setValidator(None)

    def setRange(self, minimum: float, maximum: float) -> None:
        super().setRange(minimum, maximum)
        self.lineEdit().setValidator(None)

    def textFromValue(self, value: float) -> str:
        decimals = (
            self.decimals()
            if self._display_decimals is None
            else min(self._display_decimals, self.decimals())
        )
        locale = QLocale(self.locale())
        locale.setNumberOptions(
            locale.numberOptions() | QLocale.NumberOption.OmitGroupSeparator
        )
        text = locale.toString(float(value), "f", decimals)
        if value != 0.0 and abs(value) < 0.5 * 10.0 ** (-decimals):
            return _scientific_text(value)
        decimal_point = self.locale().decimalPoint()
        if decimal_point not in text:
            return text
        whole, fraction = text.split(decimal_point, 1)
        fraction = fraction.rstrip("0")
        minimum = min(self._minimum_display_decimals, self.decimals())
        fraction = fraction.ljust(minimum, "0")
        return whole if not fraction else f"{whole}{decimal_point}{fraction}"

    def validate(self, text: str, pos: int):
        value = _numeric_text_value(self, text)
        if value is not None and self.minimum() <= value <= self.maximum():
            return QValidator.State.Acceptable, text, pos
        return super().validate(text, pos)

    def valueFromText(self, text: str) -> float:
        value = _numeric_text_value(self, text)
        return super().valueFromText(text) if value is None else value

    def keyPressEvent(self, event) -> None:
        if _insert_scientific_key(self, event):
            return
        super().keyPressEvent(event)

    def eventFilter(self, watched, event) -> bool:
        if (
            watched is self.lineEdit()
            and event.type() == QEvent.Type.KeyPress
            and _insert_scientific_key(self, event)
        ):
            return True
        return super().eventFilter(watched, event)


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
        self.lineEdit().installEventFilter(self)
        self.setDecimals(max(0, int(input_decimals)))
        self.lineEdit().setValidator(None)
        self.lineEdit().textEdited.connect(self._remember_user_precision)

    def setRange(self, minimum: float, maximum: float) -> None:
        super().setRange(minimum, maximum)
        self.lineEdit().setValidator(None)

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
            if value != 0.0 and abs(value) < 0.5 * 10.0 ** (-decimals):
                return _scientific_text(value)
            return text
        decimal_point = self.locale().decimalPoint()
        if decimal_point not in text:
            return text
        compact = text.rstrip("0").rstrip(decimal_point)
        if value != 0.0 and compact in {"0", "-0"}:
            return _scientific_text(value)
        return compact

    def valueFromText(self, text: str) -> float:
        """Accept decimal and scientific notation independently of locale."""

        value = _numeric_text_value(self, text)
        return super().valueFromText(text) if value is None else value

    def validate(self, text: str, pos: int):
        value = _numeric_text_value(self, text)
        if value is not None and self.minimum() <= value <= self.maximum():
            return QValidator.State.Acceptable, text, pos
        return super().validate(text, pos)

    def keyPressEvent(self, event) -> None:
        if _insert_scientific_key(self, event):
            return
        super().keyPressEvent(event)

    def eventFilter(self, watched, event) -> bool:
        if (
            watched is self.lineEdit()
            and event.type() == QEvent.Type.KeyPress
            and _insert_scientific_key(self, event)
        ):
            return True
        return super().eventFilter(watched, event)

    def _remember_user_precision(self, text: str) -> None:
        if "e" in text.casefold():
            self._user_decimals = None
            return
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
