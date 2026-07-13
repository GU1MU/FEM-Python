"""约束与载荷符号显示设置。"""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QVBoxLayout,
)

from .visualization.symbols import SymbolSettings


COLOR_OPTIONS = (
    ("蓝色", "#2980b9"),
    ("红色", "#c0392b"),
    ("橙色", "#e67e22"),
    ("绿色", "#278a5b"),
    ("紫色", "#7d3c98"),
)


class SymbolSettingsDialog(QDialog):
    """设置当前分析步的约束与载荷符号。"""

    applyRequested = Signal(object)

    def __init__(
        self,
        settings: SymbolSettings,
        step_names: tuple[str, ...],
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("约束与载荷显示")
        self.setMinimumWidth(440)
        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.step_combo = QComboBox(self)
        for name in step_names:
            self.step_combo.addItem(name, name)
        index = self.step_combo.findData(settings.step_name)
        self.step_combo.setCurrentIndex(index if index >= 0 else 0)
        self.show_constraints = QCheckBox("显示位移约束", self)
        self.show_constraints.setChecked(settings.show_constraints)
        self.show_nodal_loads = QCheckBox("显示节点载荷", self)
        self.show_nodal_loads.setChecked(settings.show_nodal_loads)
        self.show_edge_loads = QCheckBox("显示边载荷", self)
        self.show_edge_loads.setChecked(settings.show_edge_loads)
        self.show_surface_loads = QCheckBox("显示面压力与面牵引", self)
        self.show_surface_loads.setChecked(settings.show_surface_loads)
        self.show_values = QCheckBox("显示数值标签", self)
        self.show_values.setChecked(settings.show_values)
        self.scale = QDoubleSpinBox(self)
        self.scale.setRange(0.01, 100.0)
        self.scale.setDecimals(3)
        self.scale.setValue(settings.scale)
        self.normalize = QCheckBox("箭头归一化", self)
        self.normalize.setChecked(settings.normalize_arrows)
        self.density = QComboBox(self)
        for label, key in (("低", "low"), ("中等", "medium"), ("高", "high")):
            self.density.addItem(label, key)
        self.density.setCurrentIndex(max(0, self.density.findData(settings.sampling_density)))
        self.constraint_color = _color_combo(settings.constraint_color, self)
        self.load_color = _color_combo(settings.load_color, self)
        form.addRow("分析步：", self.step_combo)
        form.addRow(self.show_constraints)
        form.addRow(self.show_nodal_loads)
        form.addRow(self.show_edge_loads)
        form.addRow(self.show_surface_loads)
        form.addRow(self.show_values)
        form.addRow("符号比例：", self.scale)
        form.addRow(self.normalize)
        form.addRow("区域符号密度：", self.density)
        form.addRow("约束颜色：", self.constraint_color)
        form.addRow("载荷颜色：", self.load_color)
        layout.addLayout(form)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Apply
            | QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel,
            parent=self,
        )
        buttons.button(QDialogButtonBox.StandardButton.Apply).setText("应用")
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("确定")
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("取消")
        buttons.button(QDialogButtonBox.StandardButton.Apply).clicked.connect(self.apply)
        buttons.accepted.connect(self.accept_with_apply)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def settings(self) -> SymbolSettings:
        """返回当前设置。"""
        step_data = self.step_combo.currentData()
        return SymbolSettings(
            step_name=str(step_data) if step_data is not None else None,
            show_constraints=self.show_constraints.isChecked(),
            show_nodal_loads=self.show_nodal_loads.isChecked(),
            show_edge_loads=self.show_edge_loads.isChecked(),
            show_surface_loads=self.show_surface_loads.isChecked(),
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
