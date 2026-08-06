"""视口图片导出设置对话框。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)

from .dialogs import configure_form_layout


MIN_IMAGE_DIMENSION = 64
MAX_IMAGE_DIMENSION = 16384
IMAGE_FILE_FILTER = "PNG 图片 (*.png);;JPEG 图片 (*.jpg *.jpeg)"


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
        default_path: str,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("viewportImageExportDialog")
        self.setWindowTitle("视口图片导出设置")
        self.setMinimumWidth(520)
        self._current_size = (int(current_size[0]), int(current_size[1]))
        self._default_path = str(default_path)

        layout = QVBoxLayout(self)
        form = QFormLayout()
        configure_form_layout(form)

        self.path_edit = QLineEdit(self)
        self.path_edit.setObjectName("viewportExportPath")
        self.path_edit.setReadOnly(True)
        self.path_edit.setPlaceholderText("请选择图片保存路径")
        self.browse_button = QPushButton("浏览…", self)
        self.browse_button.setObjectName("viewportExportBrowse")
        path_layout = QHBoxLayout()
        path_layout.setContentsMargins(0, 0, 0, 0)
        path_layout.addWidget(self.path_edit, 1)
        path_layout.addWidget(self.browse_button)

        self.quality_combo = QComboBox(self)
        self.quality_combo.setObjectName("viewportExportQuality")
        self.quality_combo.addItem("当前分辨率", 1)
        self.quality_combo.addItem("高清 2×", 2)
        self.quality_combo.addItem("超清 4×", 4)
        self.quality_combo.addItem("自定义", "custom")
        self.quality_combo.setCurrentIndex(1)

        initial_width = self._clamp_dimension(self._current_size[0] * 2)
        initial_height = self._clamp_dimension(self._current_size[1] * 2)
        self.width_spin = self._dimension_spin_box(initial_width)
        self.width_spin.setObjectName("viewportExportWidth")
        self.height_spin = self._dimension_spin_box(initial_height)
        self.height_spin.setObjectName("viewportExportHeight")
        self.transparent_background_check = QCheckBox("透明背景", self)
        self.transparent_background_check.setObjectName(
            "viewportExportTransparentBackground"
        )
        self.transparent_background_check.setChecked(False)
        self.transparent_background_check.setEnabled(False)

        form.addRow("保存路径：", path_layout)
        form.addRow("导出质量：", self.quality_combo)
        form.addRow("宽度：", self.width_spin)
        form.addRow("高度：", self.height_spin)
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

        self.browse_button.clicked.connect(self._choose_target_path)
        self.quality_combo.currentIndexChanged.connect(self._update_controls)
        self._update_controls()
        self._update_target_format()

    @property
    def target_path(self) -> str:
        """返回用户选定并完成扩展名规范化的保存路径。"""
        return self.path_edit.text().strip()

    @property
    def options(self) -> ViewportImageExportOptions:
        """返回当前控件状态对应的截图参数。"""
        quality = self.quality_combo.currentData()
        if quality == "custom":
            scale = 1
            window_size = (
                self.width_spin.value(),
                self.height_spin.value(),
            )
        else:
            scale = int(quality)
            window_size = None
        return ViewportImageExportOptions(
            scale=scale,
            window_size=window_size,
            transparent_background=(
                Path(self.target_path).suffix.lower() == ".png"
                and self.transparent_background_check.isChecked()
            ),
        )

    @property
    def output_size(self) -> tuple[int, int]:
        """返回当前选择会生成的图片像素尺寸。"""
        return (
            self.width_spin.value(),
            self.height_spin.value(),
        )

    def _dimension_spin_box(self, value: int) -> QSpinBox:
        spin = QSpinBox(self)
        spin.setRange(MIN_IMAGE_DIMENSION, MAX_IMAGE_DIMENSION)
        spin.setSuffix(" px")
        spin.setValue(value)
        return spin

    def _update_controls(self, *_args) -> None:
        quality = self.quality_combo.currentData()
        custom = quality == "custom"
        if custom:
            self.width_spin.setMinimum(MIN_IMAGE_DIMENSION)
            self.height_spin.setMinimum(MIN_IMAGE_DIMENSION)
            self.width_spin.setMaximum(MAX_IMAGE_DIMENSION)
            self.height_spin.setMaximum(MAX_IMAGE_DIMENSION)
        else:
            scale = int(quality)
            self._set_fixed_dimension(self.width_spin, self._current_size[0] * scale)
            self._set_fixed_dimension(self.height_spin, self._current_size[1] * scale)
        self.width_spin.setEnabled(custom)
        self.height_spin.setEnabled(custom)

    def _choose_target_path(self) -> None:
        path, _selected_filter = QFileDialog.getSaveFileName(
            self,
            "选择视口图片保存位置",
            self.target_path or self._default_path,
            IMAGE_FILE_FILTER,
        )
        if not path:
            return
        target = Path(path)
        if target.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            target = target.with_suffix(".png")
        self.path_edit.setText(str(target))
        self._update_target_format()

    def _update_target_format(self) -> None:
        supports_transparency = Path(self.target_path).suffix.lower() == ".png"
        self.transparent_background_check.setEnabled(supports_transparency)
        if not supports_transparency:
            self.transparent_background_check.setChecked(False)
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(
            bool(self.target_path)
        )

    @staticmethod
    def _set_fixed_dimension(spin: QSpinBox, value: int) -> None:
        spin.setMinimum(1)
        spin.setMaximum(max(MAX_IMAGE_DIMENSION, int(value)))
        spin.setValue(int(value))

    @staticmethod
    def _clamp_dimension(value: int) -> int:
        return min(MAX_IMAGE_DIMENSION, max(MIN_IMAGE_DIMENSION, int(value)))
