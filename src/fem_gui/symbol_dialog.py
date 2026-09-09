"""Constraint and load symbol display settings."""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QVBoxLayout,
)

from .visualization.symbols import SymbolSettings
from .dialogs import CompactDoubleSpinBox, configure_form_layout


COLOR_OPTIONS = (
    ("Blue gray", "#5F7F96"),
    ("Brick red", "#A65D54"),
    ("Ochre orange", "#B47A4B"),
    ("Gray green", "#638477"),
    ("Gray purple", "#80728F"),
)


class SymbolSettingsDialog(QDialog):
    """Configure constraint and load symbols for the current step."""

    applyRequested = Signal(object)

    def __init__(
        self,
        settings: SymbolSettings,
        step_names: tuple[str, ...],
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Constraint and Load Display")
        self.setMinimumWidth(400)
        layout = QVBoxLayout(self)
        form = QFormLayout()
        configure_form_layout(form)
        self.step_combo = QComboBox(self)
        for name in step_names:
            self.step_combo.addItem(name, name)
        index = self.step_combo.findData(settings.step_name)
        self.step_combo.setCurrentIndex(index if index >= 0 else 0)
        self.show_constraints = QCheckBox("Show displacement constraints", self)
        self.show_constraints.setChecked(settings.show_constraints)
        self.show_nodal_loads = QCheckBox("Show nodal forces", self)
        self.show_nodal_loads.setChecked(settings.show_nodal_loads)
        self.show_edge_loads = QCheckBox("Show edge forces", self)
        self.show_edge_loads.setChecked(settings.show_edge_loads)
        self.show_surface_loads = QCheckBox("Show surface forces", self)
        self.show_surface_loads.setChecked(settings.show_surface_loads)
        self.show_line_loads = QCheckBox("Show beam edge forces", self)
        self.show_line_loads.setChecked(settings.show_line_loads)
        self.show_values = QCheckBox("Show value labels", self)
        self.show_values.setChecked(settings.show_values)
        self.scale = CompactDoubleSpinBox(self)
        self.scale.setRange(0.01, 100.0)
        self.scale.setDecimals(3)
        self.scale.setValue(settings.scale)
        self.normalize = QCheckBox("Normalize arrows", self)
        self.normalize.setChecked(settings.normalize_arrows)
        self.density = QComboBox(self)
        for label, key in (("Low", "low"), ("Medium", "medium"), ("High", "high")):
            self.density.addItem(label, key)
        self.density.setCurrentIndex(max(0, self.density.findData(settings.sampling_density)))
        self.constraint_color = _color_combo(settings.constraint_color, self)
        self.load_color = _color_combo(settings.load_color, self)
        form.addRow("Step:", self.step_combo)
        form.addRow(self.show_constraints)
        form.addRow(self.show_nodal_loads)
        form.addRow(self.show_edge_loads)
        form.addRow(self.show_surface_loads)
        form.addRow(self.show_line_loads)
        form.addRow(self.show_values)
        form.addRow("Symbol scale:", self.scale)
        form.addRow(self.normalize)
        form.addRow("Region symbol density:", self.density)
        form.addRow("Constraint color:", self.constraint_color)
        form.addRow("Load color:", self.load_color)
        layout.addLayout(form)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Apply
            | QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel,
            parent=self,
        )
        buttons.button(QDialogButtonBox.StandardButton.Apply).setText("Apply")
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("OK")
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("Cancel")
        buttons.button(QDialogButtonBox.StandardButton.Apply).clicked.connect(self.apply)
        buttons.accepted.connect(self.accept_with_apply)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def settings(self) -> SymbolSettings:
        """Return the current settings."""
        step_data = self.step_combo.currentData()
        return SymbolSettings(
            step_name=str(step_data) if step_data is not None else None,
            show_constraints=self.show_constraints.isChecked(),
            show_nodal_loads=self.show_nodal_loads.isChecked(),
            show_edge_loads=self.show_edge_loads.isChecked(),
            show_surface_loads=self.show_surface_loads.isChecked(),
            show_line_loads=self.show_line_loads.isChecked(),
            show_values=self.show_values.isChecked(),
            scale=float(self.scale.value()),
            normalize_arrows=self.normalize.isChecked(),
            sampling_density=str(self.density.currentData()),
            constraint_color=str(self.constraint_color.currentData()),
            load_color=str(self.load_color.currentData()),
        )

    def apply(self) -> None:
        self.applyRequested.emit(self.settings())

    def accept_with_apply(self) -> None:
        self.apply()
        self.accept()


def _color_combo(selected: str, parent) -> QComboBox:
    combo = QComboBox(parent)
    for label, value in COLOR_OPTIONS:
        combo.addItem(label, value)
    combo.setCurrentIndex(max(0, combo.findData(selected)))
    return combo
