"""Viewport image export settings dialog."""

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
from .theme import COLORS


MIN_IMAGE_DIMENSION = 64
MAX_IMAGE_DIMENSION = 16384
IMAGE_FILE_FILTER = "PNG images (*.png);;JPEG images (*.jpg *.jpeg)"


@dataclass(frozen=True, slots=True)
class ViewportImageExportOptions:
    """Parsed export settings ready for the viewport screenshot interface."""

    scale: int
    window_size: tuple[int, int] | None
    transparent_background: bool


class ViewportImageExportDialog(QDialog):
    """Select resolution and background options for this viewport image export."""

    def __init__(
        self,
        current_size: tuple[int, int],
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("viewportImageExportDialog")
        self.setWindowTitle("Viewport Image Export Settings")
        self.setMinimumWidth(520)
        self._current_size = (int(current_size[0]), int(current_size[1]))

        layout = QVBoxLayout(self)
        form = QFormLayout()
        configure_form_layout(form)

        self.path_edit = QLineEdit(self)
        self.path_edit.setObjectName("viewportExportPath")
        self.path_edit.setReadOnly(True)
        self.path_edit.setPlaceholderText("Select an image save path")
        self.browse_button = QPushButton("Browse...", self)
        self.browse_button.setObjectName("viewportExportBrowse")
        path_layout = QHBoxLayout()
        path_layout.setContentsMargins(0, 0, 0, 0)
        path_layout.addWidget(self.path_edit, 1)
        path_layout.addWidget(self.browse_button)

        self.quality_combo = QComboBox(self)
        self.quality_combo.setObjectName("viewportExportQuality")
        self.quality_combo.addItem("Current resolution", 1)
        self.quality_combo.addItem("High quality 2×", 2)
        self.quality_combo.addItem("Ultra quality 4×", 4)
        self.quality_combo.addItem("Custom", "custom")
        self.quality_combo.setCurrentIndex(1)

        initial_width = self._clamp_dimension(self._current_size[0] * 2)
        initial_height = self._clamp_dimension(self._current_size[1] * 2)
        self.width_spin = self._dimension_spin_box(initial_width)
        self.width_spin.setObjectName("viewportExportWidth")
        self.height_spin = self._dimension_spin_box(initial_height)
        self.height_spin.setObjectName("viewportExportHeight")
        self.setStyleSheet(
            f"""
            QSpinBox#viewportExportWidth:disabled,
            QSpinBox#viewportExportHeight:disabled {{
                background: {COLORS['background']};
                color: {COLORS['disabled']};
                border-color: {COLORS['soft_border']};
            }}
            """
        )
        self.transparent_background_check = QCheckBox("Transparent background", self)
        self.transparent_background_check.setObjectName(
            "viewportExportTransparentBackground"
        )
        self.transparent_background_check.setChecked(False)
        self.transparent_background_check.setEnabled(False)

        form.addRow("Save path:", path_layout)
        form.addRow("Export quality:", self.quality_combo)
        form.addRow("Width:", self.width_spin)
        form.addRow("Height:", self.height_spin)
        form.addRow("Background:", self.transparent_background_check)
        layout.addLayout(form)

        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            parent=self,
        )
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setText("OK")
        self.buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("Cancel")
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)

        self.browse_button.clicked.connect(self._choose_target_path)
        self.quality_combo.currentIndexChanged.connect(self._update_controls)
        self._update_controls()
        self._update_target_format()

    @property
    def target_path(self) -> str:
        """Return the selected save path with its extension normalized."""
        return self.path_edit.text().strip()

    @property
    def options(self) -> ViewportImageExportOptions:
        """Return screenshot parameters for the current control state."""
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
        """Return the image pixel dimensions for the current selection."""
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
            "Select Viewport Image Save Location",
            self.target_path,
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
        suffix = Path(self.target_path).suffix.lower()
        supports_transparency = not suffix or suffix == ".png"
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
