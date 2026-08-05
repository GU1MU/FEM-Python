"""视口图片导出设置对话框。"""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QSpinBox,
    QVBoxLayout,
)

from .dialogs import configure_form_layout


MIN_IMAGE_DIMENSION = 64
MAX_IMAGE_DIMENSION = 16384


@dataclass(frozen=True, slots=True)
class ViewportImageExportOptions:
    """已经解析、可直接传给视口截图接口的导出设置。"""

    scale: int
    window_size: tuple[int, int] | None
    transparent_background: bool


class ViewportImageExportDialog(QDialog):
    """选择本次视口图片导出的分辨率与背景选项。"""

    def __init__(
        self,
        current_size: tuple[int, int],
        supports_transparent_background: bool,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("viewportImageExportDialog")
        self.setWindowTitle("视口图片导出设置")
        self.setMinimumWidth(390)
        self._current_size = (int(current_size[0]), int(current_size[1]))
        self._supports_transparent_background = bool(supports_transparent_background)

        layout = QVBoxLayout(self)
        form = QFormLayout()
        configure_form_layout(form)

        self.current_size_label = QLabel(self._format_size(self._current_size), self)
        self.current_size_label.setObjectName("viewportCurrentSize")
        self.quality_combo = QComboBox(self)
        self.quality_combo.setObjectName("viewportExportQuality")
        self.quality_combo.addItem("当前分辨率", 1)
        self.quality_combo.addItem("高清 2×", 2)
        self.quality_combo.addItem("超清 4×", 4)
        self.quality_combo.addItem("自定义", "custom")
        self.quality_combo.setCurrentIndex(1)

        initial_width = self._clamp_dimension(self._current_size[0] * 2)
        initial_height = self._clamp_dimension(self._current_size[1] * 2)
        self.custom_width_spin = self._dimension_spin_box(initial_width)
        self.custom_width_spin.setObjectName("viewportExportWidth")
        self.custom_height_spin = self._dimension_spin_box(initial_height)
        self.custom_height_spin.setObjectName("viewportExportHeight")
        self.output_size_label = QLabel(self)
        self.output_size_label.setObjectName("viewportExportOutputSize")
        self.transparent_background_check = QCheckBox("透明背景", self)
        self.transparent_background_check.setObjectName(
            "viewportExportTransparentBackground"
        )
        self.transparent_background_check.setChecked(False)
        self.transparent_background_check.setEnabled(
            self._supports_transparent_background
        )

        form.addRow("当前视口：", self.current_size_label)
        form.addRow("导出质量：", self.quality_combo)
        form.addRow("自定义宽度：", self.custom_width_spin)
        form.addRow("自定义高度：", self.custom_height_spin)
        form.addRow("输出尺寸：", self.output_size_label)
        form.addRow("背景：", self.transparent_background_check)
        layout.addLayout(form)

        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            parent=self,
        )
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setText("确定")
        self.buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("取消")
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)

        self.quality_combo.currentIndexChanged.connect(self._update_controls)
        self.custom_width_spin.valueChanged.connect(self._update_output_size)
        self.custom_height_spin.valueChanged.connect(self._update_output_size)
        self._update_controls()

    @property
    def options(self) -> ViewportImageExportOptions:
        """返回当前控件状态对应的截图参数。"""
        quality = self.quality_combo.currentData()
        if quality == "custom":
            scale = 1
            window_size = (
                self.custom_width_spin.value(),
                self.custom_height_spin.value(),
            )
        else:
            scale = int(quality)
            window_size = None
        return ViewportImageExportOptions(
            scale=scale,
            window_size=window_size,
            transparent_background=(
                self._supports_transparent_background
                and self.transparent_background_check.isChecked()
            ),
        )

    @property
    def output_size(self) -> tuple[int, int]:
        """返回当前选择会生成的图片像素尺寸。"""
        quality = self.quality_combo.currentData()
        if quality == "custom":
            return (
                self.custom_width_spin.value(),
                self.custom_height_spin.value(),
            )
        scale = int(quality)
        return (
            self._current_size[0] * scale,
            self._current_size[1] * scale,
        )

    def _dimension_spin_box(self, value: int) -> QSpinBox:
        spin = QSpinBox(self)
        spin.setRange(MIN_IMAGE_DIMENSION, MAX_IMAGE_DIMENSION)
        spin.setSuffix(" px")
        spin.setValue(value)
        return spin

    def _update_controls(self, *_args) -> None:
        custom = self.quality_combo.currentData() == "custom"
        self.custom_width_spin.setEnabled(custom)
        self.custom_height_spin.setEnabled(custom)
        self._update_output_size()

    def _update_output_size(self, *_args) -> None:
        self.output_size_label.setText(self._format_size(self.output_size))

    @staticmethod
    def _format_size(size: tuple[int, int]) -> str:
        return f"{size[0]} × {size[1]} px"

    @staticmethod
    def _clamp_dimension(value: int) -> int:
        return min(MAX_IMAGE_DIMENSION, max(MIN_IMAGE_DIMENSION, int(value)))
