"""结果查询、显示和云图设置弹窗。"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .visualization.query import (
    ELEMENT_STRESS,
    NODAL_STRESS,
    QueryRecord,
    available_components,
    available_query_types,
    parse_object_ids,
    query_records,
)
from .visualization.result_adapter import ResultData


@dataclass(frozen=True, slots=True)
class ResultDisplaySettings:
    """结果显示弹窗返回的完整状态。"""

    shape_mode: str
    contour_enabled: bool
    field_key: str
    scale_mode: str
    scale_value: float
    overlay_undeformed: bool
    show_edges: bool


class ResultDisplayDialog(QDialog):
    """统一选择显示模式、字段位置、分量和变形比例。"""

    applyRequested = Signal(object)

    def __init__(
        self,
        fields: Mapping[str, Any],
        *,
        step_name: str,
        current_field: str | None,
        shape_mode: str,
        contour_enabled: bool,
        scale_mode: str,
        scale_value: float,
        overlay_undeformed: bool,
        show_edges: bool,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("结果显示")
        self.setMinimumWidth(460)
        self._records = _field_records(fields)
        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.step_combo = QComboBox(self)
        self.step_combo.addItem(step_name, step_name)
        self.shape_combo = QComboBox(self)
        self.shape_combo.addItem("未变形形状", "undeformed")
        self.shape_combo.addItem("变形形状", "deformed")
        self.shape_combo.setCurrentIndex(max(0, self.shape_combo.findData(shape_mode)))
        self.contour_checkbox = QCheckBox("显示云图", self)
        self.contour_checkbox.setChecked(contour_enabled)
        self.family_combo = QComboBox(self)
        self.position_combo = QComboBox(self)
        self.component_combo = QComboBox(self)
        for family in dict.fromkeys(record[1] for record in self._records):
            self.family_combo.addItem(family)
        self.family_combo.currentTextChanged.connect(self._sync_positions)
        self.position_combo.currentTextChanged.connect(self._sync_components)
        form.addRow("结果步：", self.step_combo)
        form.addRow("几何形状：", self.shape_combo)
        form.addRow(self.contour_checkbox)
        form.addRow("场变量：", self.family_combo)
        form.addRow("位置：", self.position_combo)
        form.addRow("分量：", self.component_combo)
        layout.addLayout(form)

        self.scale_group = QGroupBox("变形比例", self)
        scale_layout = QVBoxLayout(self.scale_group)
        self.auto_scale = QRadioButton("自动", self.scale_group)
        self.real_scale = QRadioButton("真实比例", self.scale_group)
        self.custom_scale = QRadioButton("指定比例", self.scale_group)
        scale_buttons = QButtonGroup(self.scale_group)
        for button in (self.auto_scale, self.real_scale, self.custom_scale):
            scale_buttons.addButton(button)
        self.scale_value = QDoubleSpinBox(self.scale_group)
        self.scale_value.setRange(0.0, 1.0e12)
        self.scale_value.setDecimals(6)
        self.scale_value.setValue(scale_value)
        custom_row = QHBoxLayout()
        custom_row.addWidget(self.custom_scale)
        custom_row.addWidget(self.scale_value, 1)
        scale_layout.addWidget(self.auto_scale)
        scale_layout.addWidget(self.real_scale)
        scale_layout.addLayout(custom_row)
        {"auto": self.auto_scale, "real": self.real_scale, "custom": self.custom_scale}.get(scale_mode, self.auto_scale).setChecked(True)
        layout.addWidget(self.scale_group)

        self.overlay_checkbox = QCheckBox("叠加未变形轮廓", self)
        self.overlay_checkbox.setChecked(overlay_undeformed)
        self.edges_checkbox = QCheckBox("显示单元边", self)
        self.edges_checkbox.setChecked(show_edges)
        layout.addWidget(self.overlay_checkbox)
        layout.addWidget(self.edges_checkbox)
        buttons = _dialog_buttons(self)
        buttons.button(QDialogButtonBox.StandardButton.Apply).clicked.connect(self.apply)
        buttons.accepted.connect(self.accept_with_apply)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self._select_field(current_field)
        self.shape_combo.currentIndexChanged.connect(self._refresh_mode_state)
        self.contour_checkbox.toggled.connect(self._refresh_mode_state)
        self._refresh_mode_state()

    def settings(self) -> ResultDisplaySettings:
        """返回当前选择。"""
        field_key = next(
            (
                key for key, family, position, component in self._records
                if family == self.family_combo.currentText()
                and position == self.position_combo.currentText()
                and component == self.component_combo.currentText()
            ),
            self._records[0][0],
        )
        scale_mode = "auto" if self.auto_scale.isChecked() else "real" if self.real_scale.isChecked() else "custom"
        return ResultDisplaySettings(
            shape_mode=str(self.shape_combo.currentData()),
            contour_enabled=self.contour_checkbox.isChecked(),
            field_key=field_key,
            scale_mode=scale_mode,
            scale_value=float(self.scale_value.value()),
            overlay_undeformed=self.overlay_checkbox.isChecked(),
            show_edges=self.edges_checkbox.isChecked(),
        )

    def apply(self) -> None:
        self.applyRequested.emit(self.settings())

    def accept_with_apply(self) -> None:
        self.apply()
        self.accept()

    def _select_field(self, field_key: str | None) -> None:
        record = next((record for record in self._records if record[0] == field_key), self._records[0])
        self.family_combo.setCurrentText(record[1])
        self._sync_positions()
        self.position_combo.setCurrentText(record[2])
        self._sync_components()
        self.component_combo.setCurrentText(record[3])

    def _sync_positions(self) -> None:
        current = self.position_combo.currentText()
        self.position_combo.blockSignals(True)
        self.position_combo.clear()
        for position in dict.fromkeys(record[2] for record in self._records if record[1] == self.family_combo.currentText()):
            self.position_combo.addItem(position)
        self.position_combo.setCurrentText(current)
        self.position_combo.blockSignals(False)
        self._sync_components()

    def _sync_components(self) -> None:
        current = self.component_combo.currentText()
        self.component_combo.clear()
        for component in dict.fromkeys(
            record[3] for record in self._records
            if record[1] == self.family_combo.currentText()
            and record[2] == self.position_combo.currentText()
        ):
            self.component_combo.addItem(component)
        self.component_combo.setCurrentText(current)

    def _refresh_mode_state(self) -> None:
        contour_enabled = self.contour_checkbox.isChecked()
        for control in (self.family_combo, self.position_combo, self.component_combo):
            control.setEnabled(contour_enabled)
        deformed = self.shape_combo.currentData() == "deformed"
        self.scale_group.setEnabled(deformed)
        self.overlay_checkbox.setEnabled(deformed)


class ContourSettingsDialog(QDialog):
    """控制云图范围、色带、数值格式、图例和极值。"""

    applyRequested = Signal(object)

    def __init__(self, options: dict[str, Any], parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("云图设置")
        self.setMinimumWidth(470)
        layout = QVBoxLayout(self)
        range_group = QGroupBox("范围", self)
        range_layout = QVBoxLayout(range_group)
        self.auto_range = QRadioButton("自动", range_group)
        self.manual_range = QRadioButton("手动", range_group)
        (self.manual_range if options.get("manual") else self.auto_range).setChecked(True)
        range_buttons = QButtonGroup(range_group)
        range_buttons.addButton(self.auto_range)
        range_buttons.addButton(self.manual_range)
        self.minimum = QDoubleSpinBox(range_group)
        self.maximum = QDoubleSpinBox(range_group)
        for spin in (self.minimum, self.maximum):
            spin.setRange(-1.0e30, 1.0e30)
            spin.setDecimals(8)
        self.minimum.setValue(float(options.get("minimum", 0.0)))
        self.maximum.setValue(float(options.get("maximum", 1.0)))
        manual_row = QHBoxLayout()
        manual_row.addWidget(self.manual_range)
        manual_row.addWidget(QLabel("最小值", self))
        manual_row.addWidget(self.minimum)
        manual_row.addWidget(QLabel("最大值", self))
        manual_row.addWidget(self.maximum)
        range_layout.addWidget(self.auto_range)
        range_layout.addLayout(manual_row)
        layout.addWidget(range_group)
        form = QFormLayout()
        self.colormap = QComboBox(self)
        for label, key in (("彩虹", "jet"), ("维里迪斯", "viridis"), ("等离子", "plasma"), ("冷暖", "coolwarm"), ("灰度", "gray")):
            self.colormap.addItem(label, key)
        self.colormap.setCurrentIndex(max(0, self.colormap.findData(options.get("colormap", "jet"))))
        self.levels = QSpinBox(self)
        self.levels.setRange(2, 256)
        self.levels.setValue(int(options.get("levels", 12)))
        self.style = QComboBox(self)
        self.style.addItem("分段云图", "segmented")
        self.style.addItem("连续云图", "continuous")
        self.style.setCurrentIndex(max(0, self.style.findData(options.get("style", "continuous"))))
        self.style.currentIndexChanged.connect(
            lambda: self.levels.setEnabled(self.style.currentData() == "segmented")
        )
        self.levels.setEnabled(self.style.currentData() == "segmented")
        self.averaging_threshold = QDoubleSpinBox(self)
        self.averaging_threshold.setRange(0.0, 100.0)
        self.averaging_threshold.setDecimals(1)
        self.averaging_threshold.setSuffix(" %")
        self.averaging_threshold.setValue(float(options.get("averaging_threshold", 75.0)))
        self.averaging_threshold.setToolTip(
            "仅用于节点平均应力；超过阈值的当前分量按单元侧分开显示"
        )
        self.number_format = QComboBox(self)
        for label, key in (("自动", "general"), ("定点小数", "fixed"), ("科学计数法", "scientific")):
            self.number_format.addItem(label, key)
        self.number_format.setCurrentIndex(max(0, self.number_format.findData(options.get("number_format", "general"))))
        self.decimals = QSpinBox(self)
        self.decimals.setRange(0, 12)
        self.decimals.setValue(int(options.get("decimals", 5)))
        self.orientation = QComboBox(self)
        self.orientation.addItem("横向", "horizontal")
        self.orientation.addItem("纵向", "vertical")
        self.orientation.setCurrentIndex(max(0, self.orientation.findData(options.get("orientation", "horizontal"))))
        form.addRow("色带：", self.colormap)
        form.addRow("云图样式：", self.style)
        form.addRow("色带级数：", self.levels)
        form.addRow("节点平均阈值：", self.averaging_threshold)
        form.addRow("数值格式：", self.number_format)
        form.addRow("小数位：", self.decimals)
        form.addRow("图例方向：", self.orientation)
        layout.addLayout(form)
        self.legend = QCheckBox("显示图例", self)
        self.legend.setChecked(bool(options.get("legend", True)))
        self.show_minimum = QCheckBox("显示最小值", self)
        self.show_minimum.setChecked(bool(options.get("show_minimum", True)))
        self.show_maximum = QCheckBox("显示最大值", self)
        self.show_maximum.setChecked(bool(options.get("show_maximum", True)))
        self.show_ids = QCheckBox("显示对象编号", self)
        self.show_ids.setChecked(bool(options.get("show_ids", False)))
        self.show_edges = QCheckBox("显示单元边", self)
        self.show_edges.setChecked(bool(options.get("edges", False)))
        for checkbox in (self.legend, self.show_minimum, self.show_maximum, self.show_ids, self.show_edges):
            layout.addWidget(checkbox)
        buttons = _dialog_buttons(self)
        buttons.button(QDialogButtonBox.StandardButton.Apply).clicked.connect(self.apply)
        buttons.accepted.connect(self.accept_with_apply)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def settings(self) -> dict[str, Any]:
        return {
            "manual": self.manual_range.isChecked(),
            "minimum": float(self.minimum.value()),
            "maximum": float(self.maximum.value()),
            "colormap": str(self.colormap.currentData()),
            "style": str(self.style.currentData()),
            "levels": int(self.levels.value()),
            "number_format": str(self.number_format.currentData()),
            "decimals": int(self.decimals.value()),
            "orientation": str(self.orientation.currentData()),
            "legend": self.legend.isChecked(),
            "show_minimum": self.show_minimum.isChecked(),
            "show_maximum": self.show_maximum.isChecked(),
            "show_ids": self.show_ids.isChecked(),
            "edges": self.show_edges.isChecked(),
            "averaging_threshold": float(self.averaging_threshold.value()),
        }

    def apply(self) -> None:
        if self.manual_range.isChecked() and self.minimum.value() >= self.maximum.value():
            QMessageBox.warning(self, "云图设置", "手动范围的最小值必须小于最大值。")
            return
        self.applyRequested.emit(self.settings())

    def accept_with_apply(self) -> None:
        if self.manual_range.isChecked() and self.minimum.value() >= self.maximum.value():
            self.apply()
            return
        self.apply()
        self.accept()


class ResultQueryDialog(QDialog):
    """查询节点或单元结果并与视口选择联动。"""

    locateRequested = Signal(str, int)

    def __init__(
        self,
        data: ResultData,
        *,
        step_name: str,
        node_ids: tuple[int, ...],
        element_ids: tuple[int, ...],
        selected_kind: str | None,
        selected_id: int | None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("查询结果")
        self.resize(780, 520)
        self._data = data
        self._node_ids = node_ids
        self._element_ids = element_ids
        self._selected_kind = selected_kind
        self._selected_id = selected_id
        self._records: tuple[QueryRecord, ...] = ()
        self._components: tuple[str, ...] = ()
        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.step_combo = QComboBox(self)
        self.step_combo.addItem(step_name, step_name)
        self.kind_combo = QComboBox(self)
        self.kind_combo.addItem("节点", "node")
        self.kind_combo.addItem("单元", "element")
        self.type_combo = QComboBox(self)
        self.component_combo = QComboBox(self)
        self.ids_edit = QLineEdit(self)
        self.ids_edit.setPlaceholderText("例如：1, 3, 5-8")
        use_selection = QPushButton("使用当前选择", self)
        use_selection.clicked.connect(self.use_current_selection)
        id_row = QHBoxLayout()
        id_row.addWidget(self.ids_edit, 1)
        id_row.addWidget(use_selection)
        form.addRow("结果步：", self.step_combo)
        form.addRow("对象类型：", self.kind_combo)
        form.addRow("结果类型：", self.type_combo)
        form.addRow("分量：", self.component_combo)
        form.addRow("对象编号：", id_row)
        layout.addLayout(form)
        command_row = QHBoxLayout()
        query_button = QPushButton("查询", self)
        locate_button = QPushButton("在视口定位", self)
        copy_button = QPushButton("复制", self)
        export_button = QPushButton("导出 CSV", self)
        close_button = QPushButton("关闭", self)
        query_button.clicked.connect(self.run_query)
        locate_button.clicked.connect(self.locate_current)
        copy_button.clicked.connect(self.copy_table)
        export_button.clicked.connect(self.export_csv)
        close_button.clicked.connect(self.close)
        for button in (query_button, locate_button, copy_button, export_button):
            command_row.addWidget(button)
        command_row.addStretch(1)
        command_row.addWidget(close_button)
        layout.addLayout(command_row)
        self.table = QTableWidget(self)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.doubleClicked.connect(self.locate_current)
        layout.addWidget(self.table, 1)
        self.kind_combo.currentIndexChanged.connect(self._sync_query_types)
        self.type_combo.currentIndexChanged.connect(self._sync_components)
        if selected_kind in {"node", "element"}:
            self.kind_combo.setCurrentIndex(self.kind_combo.findData(selected_kind))
        self._sync_query_types()
        self.use_current_selection()

    def use_current_selection(self) -> None:
        if self._selected_id is None or self._selected_kind != self.kind_combo.currentData():
            return
        self.ids_edit.setText(str(self._selected_id))

    def run_query(self) -> None:
        try:
            ids = parse_object_ids(self.ids_edit.text(), self._valid_ids())
        except ValueError as error:
            QMessageBox.warning(self, "查询结果", str(error))
            return
        self._records = query_records(self._data, str(self.type_combo.currentData()), ids)
        selected_component = self.component_combo.currentData()
        self._components = (
            available_components(self._data, str(self.type_combo.currentData()))
            if selected_component is None
            else (str(selected_component),)
        )
        self._populate_table()

    def locate_current(self, *_args) -> None:
        row = self.table.currentRow()
        if row < 0 or row >= len(self._records):
            return
        self.locateRequested.emit(str(self.kind_combo.currentData()), self._records[row].object_id)

    def copy_table(self) -> None:
        QApplication.clipboard().setText(self._table_text("\t"))

    def export_csv(self) -> None:
        path, _filter = QFileDialog.getSaveFileName(self, "导出查询结果", "query_results.csv", "CSV 文件 (*.csv)")
        if not path:
            return
        target = Path(path).with_suffix(".csv")
        with target.open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.writer(stream)
            for line in self._table_rows():
                writer.writerow(line)

    def _sync_query_types(self) -> None:
        current = self.type_combo.currentData()
        self.type_combo.blockSignals(True)
        self.type_combo.clear()
        for query_type in available_query_types(self._data, str(self.kind_combo.currentData())):
            self.type_combo.addItem(query_type, query_type)
        index = self.type_combo.findData(current)
        self.type_combo.setCurrentIndex(index if index >= 0 else 0)
        self.type_combo.blockSignals(False)
        self._sync_components()
        self.use_current_selection()

    def _sync_components(self) -> None:
        self.component_combo.clear()
        self.component_combo.addItem("全部分量", None)
        for component in available_components(self._data, str(self.type_combo.currentData())):
            self.component_combo.addItem(_component_label(component), component)

    def _valid_ids(self) -> tuple[int, ...]:
        return self._element_ids if self.kind_combo.currentData() == "element" else self._node_ids

    def _populate_table(self) -> None:
        include_source = any(record.source_element_id is not None for record in self._records)
        headers = ["编号"]
        if include_source:
            headers.append("来源单元")
        headers.extend(_component_label(component) for component in self._components)
        self.table.clear()
        self.table.setColumnCount(len(headers))
        self.table.setHorizontalHeaderLabels(headers)
        self.table.setRowCount(len(self._records))
        for row, record in enumerate(self._records):
            values: list[object] = [record.object_id]
            if include_source:
                values.append("平均" if record.source_element_id is None else record.source_element_id)
            values.extend(record.values.get(component, float("nan")) for component in self._components)
            for column, value in enumerate(values):
                text = f"{value:.8g}" if isinstance(value, float) else str(value)
                self.table.setItem(row, column, QTableWidgetItem(text))
        self.table.resizeColumnsToContents()
        if self.table.rowCount():
            self.table.selectRow(0)

    def _table_rows(self) -> list[list[str]]:
        return [
            [self.table.horizontalHeaderItem(column).text() for column in range(self.table.columnCount())],
            *[
                [self.table.item(row, column).text() for column in range(self.table.columnCount())]
                for row in range(self.table.rowCount())
            ],
        ]

    def _table_text(self, separator: str) -> str:
        return "\n".join(separator.join(row) for row in self._table_rows())


def _field_records(fields: Mapping[str, Any]) -> list[tuple[str, str, str, str]]:
    records: list[tuple[str, str, str, str]] = []
    for key, field in fields.items():
        if key in {"U", "U1", "U2", "U3", "R3"}:
            family = "位移"
        elif key in {"RF", "RF1", "RF2", "RF3", "RM3"}:
            family = "反力"
        else:
            family = "应力"
        position = "节点" if field.association == "point" else "单元中心"
        if key.startswith("N:"):
            position = "节点平均"
        elif key.startswith("EN:"):
            position = "单元节点（不平均）"
        component = key.split(":", 1)[-1] if family == "应力" else field.label
        component = _component_label(component)
        records.append((key, family, position, component))
    return records


def _component_label(component: str) -> str:
    return {
        "U": "总位移",
        "RF": "总反力",
        "MaxPrincipal": "最大主应力",
        "MinPrincipal": "最小主应力",
    }.get(component, component)


def _dialog_buttons(parent: QWidget) -> QDialogButtonBox:
    buttons = QDialogButtonBox(
        QDialogButtonBox.StandardButton.Apply
        | QDialogButtonBox.StandardButton.Ok
        | QDialogButtonBox.StandardButton.Cancel,
        parent=parent,
    )
    buttons.button(QDialogButtonBox.StandardButton.Apply).setText("应用")
    buttons.button(QDialogButtonBox.StandardButton.Ok).setText("确定")
    buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("取消")
    return buttons
