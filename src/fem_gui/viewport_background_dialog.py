"""视口背景设置对话框。"""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QPushButton,
    QVBoxLayout,
)

from .viewport_background import ViewportBackgroundSettings


PRESETS = (
    ("清浅蓝灰", ViewportBackgroundSettings()),
    ("白色", ViewportBackgroundSettings("solid", "#ffffff", "#ffffff")),
    ("浅灰", ViewportBackgroundSettings("solid", "#f2f4f5", "#f2f4f5")),
    ("深蓝渐变", ViewportBackgroundSettings("gradient", "#607d92", "#1e3448")),
    ("黑色", ViewportBackgroundSettings("solid", "#16191c", "#16191c")),
)


class ViewportBackgroundDialog(QDialog):
    """选择背景预设或自定义渐变，并支持实时预览。"""

    previewRequested = Signal(object)
    applyRequested = Signal(object, bool)

    def __init__(
        self,
        settings: ViewportBackgroundSettings,
        remember: bool,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("viewportBackgroundDialog")
        self.setWindowTitle("视口背景")
        self.setMinimumWidth(420)
        self._baseline = settings.normalized()
        self._bottom_color = self._baseline.bottom_color
        self._top_color = self._baseline.top_color

        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.preset_combo = QComboBox(self)
        self.preset_combo.setObjectName("backgroundPreset")
        for label, value in PRESETS:
            self.preset_combo.addItem(label, value)
        self.preset_combo.addItem("自定义", None)
        self.style_combo = QComboBox(self)
        self.style_combo.setObjectName("backgroundStyle")
        self.style_combo.addItem("纯色", "solid")
        self.style_combo.addItem("渐变", "gradient")
        self.bottom_button = QPushButton(self)
        self.bottom_button.setObjectName("backgroundBottomColor")
        self.top_button = QPushButton(self)
        self.top_button.setObjectName("backgroundTopColor")
        form.addRow("预设：", self.preset_combo)
        form.addRow("背景样式：", self.style_combo)
        form.addRow("底部颜色：", self.bottom_button)
        form.addRow("顶部颜色：", self.top_button)
        layout.addLayout(form)
        self.auto_contrast = QCheckBox("自动调整文字和边线颜色", self)
        self.auto_contrast.setChecked(self._baseline.auto_contrast)
        self.remember = QCheckBox("记住设置", self)
        self.remember.setChecked(remember)
        layout.addWidget(self.auto_contrast)
        layout.addWidget(self.remember)

        command_row = QHBoxLayout()
        reset = QPushButton("恢复默认", self)
        reset.clicked.connect(
            lambda _checked=False: self._set_controls(ViewportBackgroundSettings())
        )
        command_row.addWidget(reset)
        command_row.addStretch(1)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Apply
            | QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel,
            parent=self,
        )
        buttons.button(QDialogButtonBox.StandardButton.Apply).setText("应用")
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("确定")
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("取消")
        command_row.addWidget(buttons)
        layout.addLayout(command_row)

        self.preset_combo.currentIndexChanged.connect(self._preset_changed)
        self.style_combo.currentIndexChanged.connect(self._controls_changed)
        self.bottom_button.clicked.connect(lambda: self._choose_color("bottom"))
        self.top_button.clicked.connect(lambda: self._choose_color("top"))
        self.auto_contrast.toggled.connect(self._controls_changed)
        buttons.button(QDialogButtonBox.StandardButton.Apply).clicked.connect(self.apply)
        buttons.accepted.connect(self.accept_with_apply)
        buttons.rejected.connect(self.reject)
        self.rejected.connect(lambda: self.previewRequested.emit(self._baseline))
        self._set_controls(self._baseline, emit_preview=False)

    def settings(self) -> ViewportBackgroundSettings:
        return ViewportBackgroundSettings(
            str(self.style_combo.currentData()),
            self._bottom_color,
            self._top_color,
            self.auto_contrast.isChecked(),
        ).normalized()

    def apply(self) -> None:
        self._baseline = self.settings()
        self.applyRequested.emit(self._baseline, self.remember.isChecked())

    def accept_with_apply(self) -> None:
        self.apply()
        self.accept()

    def _preset_changed(self, index: int) -> None:
        preset = self.preset_combo.itemData(index)
        if isinstance(preset, ViewportBackgroundSettings):
            self._set_controls(preset)

    def _set_controls(
        self,
        settings: ViewportBackgroundSettings,
        *,
        emit_preview: bool = True,
    ) -> None:
        value = settings.normalized()
        self.style_combo.blockSignals(True)
        self.auto_contrast.blockSignals(True)
        self.style_combo.setCurrentIndex(max(0, self.style_combo.findData(value.style)))
        self.auto_contrast.setChecked(value.auto_contrast)
        self.style_combo.blockSignals(False)
        self.auto_contrast.blockSignals(False)
        self._bottom_color = value.bottom_color
        self._top_color = value.top_color
        self._update_color_buttons()
        self._select_matching_preset(value)
        self.top_button.setEnabled(value.style == "gradient")
        if emit_preview:
            self.previewRequested.emit(self.settings())

    def _select_matching_preset(self, value: ViewportBackgroundSettings) -> None:
        index = len(PRESETS)
        for candidate, (_label, preset) in enumerate(PRESETS):
            if value == preset:
                index = candidate
                break
        self.preset_combo.blockSignals(True)
        self.preset_combo.setCurrentIndex(index)
        self.preset_combo.blockSignals(False)

    def _controls_changed(self, *_args) -> None:
        self.top_button.setEnabled(self.style_combo.currentData() == "gradient")
        self._select_matching_preset(self.settings())
        self.previewRequested.emit(self.settings())

    def _choose_color(self, target: str) -> None:
        current = self._bottom_color if target == "bottom" else self._top_color
        color = QColorDialog.getColor(QColor(current), self, "选择视口背景颜色")
        if not color.isValid():
            return
        if target == "bottom":
            self._bottom_color = color.name()
        else:
            self._top_color = color.name()
        self._update_color_buttons()
        self._select_matching_preset(self.settings())
        self.previewRequested.emit(self.settings())

    def _update_color_buttons(self) -> None:
        for button, color in (
            (self.bottom_button, self._bottom_color),
            (self.top_button, self._top_color),
        ):
            foreground = "#ffffff" if QColor(color).lightnessF() < 0.5 else "#20262d"
            button.setText(color.upper())
            button.setStyleSheet(
                f"QPushButton {{ background:{color}; color:{foreground}; border:1px solid #9da5ab; }}"
            )
