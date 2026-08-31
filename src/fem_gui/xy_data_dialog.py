"""XY data extraction and curve display for increment-history results."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from fem.results import (
    FieldAvailability,
    FieldPosition,
    FieldState,
    ResultCatalog,
    ResultFrameCatalog,
    ResultProbeKind,
    ResultProbeTarget,
    ResultXYAxis,
    ResultXYRequest,
    ResultXYSeries,
    ScalarFieldSelection,
    xy_difference,
    xy_derivative,
    xy_envelope,
    xy_integral,
)
from PySide6.QtCharts import QChart, QChartView, QLineSeries, QValueAxis
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QColorDialog,
    QDialog,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .dialogs import configure_form_layout
from .result_presentation import (
    result_field_position_label,
    result_variable_label,
    visible_result_fields,
)


class XYDataDialog(QDialog):
    """Generate one exact result-location curve over retained increments."""

    generateRequested = Signal(object)
    pathRequested = Signal()

    def __init__(
        self,
        catalog: ResultCatalog,
        *,
        frame_catalog: ResultFrameCatalog,
        current_selection: ScalarFieldSelection | None = None,
        node_ids: tuple[int, ...] = (),
        element_ids: tuple[int, ...] = (),
        section_point_labels: Mapping[int, str] | None = None,
        parent=None,
    ) -> None:
        if type(catalog) is not ResultCatalog:
            raise TypeError("catalog must be ResultCatalog")
        if type(frame_catalog) is not ResultFrameCatalog:
            raise TypeError("frame_catalog must be ResultFrameCatalog")
        fields = visible_result_fields(catalog.fields)
        if not fields:
            raise ValueError("XY data requires a non-empty result catalog")
        if any(type(value) is not int or value <= 0 for value in node_ids):
            raise TypeError("node_ids must contain positive integers")
        if any(type(value) is not int or value <= 0 for value in element_ids):
            raise TypeError("element_ids must contain positive integers")

        super().__init__(parent)
        self.setWindowTitle("XY 数据")
        self.setObjectName("xyDataDialog")
        self.resize(1120, 700)
        self.setMinimumSize(900, 560)
        self._catalog = catalog
        self._frame_catalog = frame_catalog
        self._fields = fields
        self._node_ids = tuple(node_ids)
        self._element_ids = tuple(element_ids)
        self._section_point_labels = dict(section_point_labels or {})
        self._pending = False
        self._series_records: list[dict[str, object]] = []
        self._curve_counter = 0

        root = QVBoxLayout(self)
        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(self._build_controls())
        splitter.addWidget(self._build_results())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        root.addWidget(splitter, 1)

        button_row = QHBoxLayout()
        self.generate_button = QPushButton("生成曲线", self)
        self.generate_button.setObjectName("xyGenerateButton")
        self.copy_button = QPushButton("复制数据", self)
        self.copy_button.setObjectName("xyCopyButton")
        self.save_button = QPushButton("保存数据", self)
        self.rename_button = QPushButton("重命名", self)
        self.delete_button = QPushButton("删除曲线", self)
        self.path_button = QPushButton("路径提取", self)
        self.close_button = QPushButton("关闭", self)
        self.close_button.setObjectName("xyCloseButton")
        button_row.addStretch(1)
        button_row.addWidget(self.generate_button)
        button_row.addWidget(self.copy_button)
        button_row.addWidget(self.save_button)
        button_row.addWidget(self.rename_button)
        button_row.addWidget(self.delete_button)
        button_row.addWidget(self.path_button)
        button_row.addWidget(self.close_button)
        root.addLayout(button_row)

        self.generate_button.clicked.connect(self.request_curve)
        self.copy_button.clicked.connect(self.copy_data)
        self.save_button.clicked.connect(self.save_data)
        self.rename_button.clicked.connect(self.rename_selected_series)
        self.delete_button.clicked.connect(self.delete_selected_series)
        self.path_button.clicked.connect(self.pathRequested.emit)
        self.close_button.clicked.connect(self.reject)
        self.field_combo.currentIndexChanged.connect(self._field_changed)
        self.target_combo.currentIndexChanged.connect(
            self._target_changed
        )

        initial = self._initial_selection(current_selection)
        self._select_field(initial)
        self._populate_axes()
        self._refresh_target_controls()
        self._refresh_availability()
        self._set_chart_message("请选择结果字段和目标位置后生成曲线")

    def _build_controls(self) -> QWidget:
        host = QWidget(self)
        layout = QVBoxLayout(host)

        result_group = QGroupBox("结果变量", host)
        form = QFormLayout(result_group)
        configure_form_layout(form)
        self.field_combo = QComboBox(result_group)
        self.component_combo = QComboBox(result_group)
        self.availability_label = QLabel(result_group)
        self.availability_label.setWordWrap(True)
        for availability in self._fields:
            self.field_combo.addItem(
                self._field_label(availability),
            )
        form.addRow("场变量：", self.field_combo)
        form.addRow("分量：", self.component_combo)
        form.addRow("字段状态：", self.availability_label)

        target_group = QGroupBox("结果位置", host)
        target_form = QFormLayout(target_group)
        configure_form_layout(target_form)
        self.target_combo = QComboBox(target_group)
        self.target_combo.addItem("节点")
        self.target_combo.addItem("单元")
        self.target_combo.addItem("积分点")
        self.target_combo.addItem("单元局部节点")
        self.target_id_label = QLabel("节点编号", target_group)
        self.target_id_spin = QSpinBox(target_group)
        self.target_id_spin.setRange(1, 1_000_000_000)
        self.integration_point_label = QLabel(
            "积分点编号",
            target_group,
        )
        self.integration_point_spin = QSpinBox(target_group)
        self.integration_point_spin.setRange(1, 1_000_000)
        self.local_node_label = QLabel("局部节点编号", target_group)
        self.local_node_spin = QSpinBox(target_group)
        self.local_node_spin.setRange(1, 1_000_000)
        target_form.addRow("目标类型：", self.target_combo)
        target_form.addRow(self.target_id_label, self.target_id_spin)
        target_form.addRow(
            self.integration_point_label,
            self.integration_point_spin,
        )
        target_form.addRow(self.local_node_label, self.local_node_spin)

        axis_group = QGroupBox("横轴", host)
        axis_form = QFormLayout(axis_group)
        configure_form_layout(axis_form)
        self.axis_combo = QComboBox(axis_group)
        axis_form.addRow("自变量：", self.axis_combo)

        layout.addWidget(result_group)
        layout.addWidget(target_group)
        layout.addWidget(axis_group)

        appearance_group = QGroupBox("曲线显示", host)
        appearance_form = QFormLayout(appearance_group)
        configure_form_layout(appearance_form)
        self.curve_color_button = QPushButton("#1F77B4", appearance_group)
        self.curve_color_button.setObjectName("xyCurveColorButton")
        self.curve_color_button.clicked.connect(self._choose_curve_color)
        self.curve_line_style_combo = QComboBox(appearance_group)
        for label, key in (
            ("实线", "solid"),
            ("虚线", "dashed"),
            ("点划线", "dashdot"),
            ("点线", "dotted"),
        ):
            self.curve_line_style_combo.addItem(label, key)
        self.curve_grid_checkbox = QCheckBox("显示网格", appearance_group)
        self.curve_grid_checkbox.setChecked(True)
        self.curve_legend_checkbox = QCheckBox("显示图例", appearance_group)
        self.curve_legend_checkbox.setChecked(True)
        appearance_form.addRow("颜色：", self.curve_color_button)
        appearance_form.addRow("线型：", self.curve_line_style_combo)
        display_host = QWidget(appearance_group)
        display_layout = QHBoxLayout(display_host)
        display_layout.setContentsMargins(0, 0, 0, 0)
        display_layout.addWidget(self.curve_grid_checkbox)
        display_layout.addWidget(self.curve_legend_checkbox)
        display_layout.addStretch(1)
        appearance_form.addRow("显示：", display_host)
        self.curve_grid_checkbox.toggled.connect(self._refresh_chart_options)
        self.curve_legend_checkbox.toggled.connect(self._refresh_chart_options)
        self.curve_line_style_combo.currentIndexChanged.connect(
            self._apply_current_curve_style
        )
        self.curve_color_button.setStyleSheet(
            "QPushButton { background-color: #1f77b4; color: white; padding: 3px 14px; }"
        )
        layout.addWidget(appearance_group)

        operation_group = QGroupBox("曲线运算", host)
        operation_form = QFormLayout(operation_group)
        configure_form_layout(operation_form)
        self.operation_combo = QComboBox(operation_group)
        for label, key in (
            ("差值（选择两条）", "difference"),
            ("导数（选择一条）", "derivative"),
            ("积分（选择一条）", "integral"),
            ("包络（选择一条）", "envelope"),
        ):
            self.operation_combo.addItem(label, key)
        self.operation_button = QPushButton("生成运算曲线", operation_group)
        self.operation_button.clicked.connect(self.apply_operation)
        operation_form.addRow("运算：", self.operation_combo)
        operation_form.addRow("操作：", self.operation_button)
        layout.addWidget(operation_group)
        layout.addStretch(1)
        host.setMinimumWidth(300)
        host.setMaximumWidth(380)
        return host

    def _build_results(self) -> QWidget:
        host = QWidget(self)
        layout = QVBoxLayout(host)
        self.chart = QChart()
        self.chart.legend().hide()
        self.chart_view = QChartView(self.chart, host)
        self.chart_view.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.chart_view.setMinimumHeight(330)
        self.series_list = QListWidget(host)
        self.series_list.setObjectName("xySeriesList")
        self.series_list.setSelectionMode(
            QListWidget.SelectionMode.ExtendedSelection
        )
        self.series_list.setMaximumHeight(125)
        self.series_list.currentRowChanged.connect(self._series_selection_changed)
        self.series_list.itemChanged.connect(self._series_visibility_changed)
        self.table = QTableWidget(host)
        self.table.setObjectName("xyDataTable")
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(("增量", "X", "Y"))
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(
            QTableWidget.SelectionBehavior.SelectRows
        )
        self.table.setAlternatingRowColors(True)
        self.status_label = QLabel(host)
        self.status_label.setWordWrap(True)
        layout.addWidget(self.chart_view, 3)
        layout.addWidget(QLabel("曲线列表", host))
        layout.addWidget(self.series_list)
        layout.addWidget(self.table, 2)
        layout.addWidget(self.status_label)
        return host

    def _initial_selection(
        self,
        current_selection: ScalarFieldSelection | None,
    ) -> ScalarFieldSelection:
        if current_selection is not None:
            if type(current_selection) is not ScalarFieldSelection:
                raise TypeError(
                    "current_selection must be ScalarFieldSelection or None"
                )
            if any(
                availability.key == current_selection.field_key
                for availability in self._fields
            ) and current_selection.component in next(
                availability.descriptor.columns
                for availability in self._fields
                if availability.key == current_selection.field_key
            ):
                return current_selection
        default = self._catalog.default_selection
        if default is not None and any(
            availability.key == default.field_key for availability in self._fields
        ):
            return default
        availability = self._fields[0]
        return ScalarFieldSelection(
            availability.key,
            availability.descriptor.default_component,
        )

    def _select_field(self, selection: ScalarFieldSelection) -> None:
        field_index = next(
            (
                index
                for index, availability in enumerate(self._fields)
                if availability.key == selection.field_key
            ),
            0,
        )
        self.field_combo.blockSignals(True)
        self.field_combo.setCurrentIndex(field_index)
        self.field_combo.blockSignals(False)
        self._sync_components(preferred=selection.component)

    def _field_label(self, availability: FieldAvailability) -> str:
        field_id = availability.descriptor.field_id
        label = (
            f"{result_variable_label(field_id.variable)} · "
            f"{result_field_position_label(field_id, section_point_labels=self._section_point_labels)}"
        )
        return f"{label}（{self._state_label(availability.state)}）"

    @staticmethod
    def _state_label(state: FieldState) -> str:
        return {
            FieldState.READY: "已就绪",
            FieldState.LAZY: "待物化",
            FieldState.UNAVAILABLE: "不可用",
        }[state]

    def _current_availability(self) -> FieldAvailability:
        index = self.field_combo.currentIndex()
        if index < 0 or index >= len(self._fields):
            raise RuntimeError("当前结果字段无效")
        return self._fields[index]

    def _sync_components(self, *, preferred: str | None = None) -> None:
        availability = self._current_availability()
        if preferred is None:
            current = self.component_combo.currentData()
            preferred = current if isinstance(current, str) else None
        self.component_combo.blockSignals(True)
        self.component_combo.clear()
        for component in availability.descriptor.columns:
            self.component_combo.addItem(component)
        index = self.component_combo.findText(preferred or "")
        if index < 0:
            index = self.component_combo.findText(
                availability.descriptor.default_component
            )
        self.component_combo.setCurrentIndex(index if index >= 0 else 0)
        self.component_combo.blockSignals(False)

    def _field_changed(self, _index: int) -> None:
        if self._pending:
            return
        self._sync_components()
        self._refresh_target_controls()
        self._refresh_availability()

    def _target_changed(self, _index: int) -> None:
        if not self._pending:
            self._refresh_target_controls()

    def _populate_axes(self) -> None:
        self.axis_combo.clear()
        metadata = self._frame_catalog.frames
        dynamic = bool(metadata) and all(
            frame.solver_kind is not None
            and str(frame.solver_kind).startswith("dynamic")
            for frame in metadata
        )
        if dynamic:
            axes = []
            if all(frame.step_time is not None for frame in metadata):
                axes.append(("步时间", ResultXYAxis.STEP_TIME))
            if all(frame.total_time is not None for frame in metadata):
                axes.append(("总时间", ResultXYAxis.TOTAL_TIME))
        else:
            axes = [("增量", ResultXYAxis.INCREMENT)]
            if metadata and all(frame.step_time is not None for frame in metadata):
                axes.append(("步时间", ResultXYAxis.STEP_TIME))
            if metadata and all(frame.load_factor is not None for frame in metadata):
                axes.append(("载荷因子", ResultXYAxis.LOAD_FACTOR))
        for label, axis in axes:
            self.axis_combo.addItem(label)
        self._axis_values = tuple(axis for _label, axis in axes)

    def _refresh_target_controls(self) -> None:
        kind = self._current_target_kind()
        self.integration_point_label.setVisible(
            kind is ResultProbeKind.INTEGRATION_POINT
        )
        self.integration_point_spin.setVisible(
            kind is ResultProbeKind.INTEGRATION_POINT
        )
        self.local_node_label.setVisible(
            kind is ResultProbeKind.ELEMENT_NODE
        )
        self.local_node_spin.setVisible(
            kind is ResultProbeKind.ELEMENT_NODE
        )
        is_node = kind is ResultProbeKind.NODE
        self.target_id_label.setText("节点编号" if is_node else "单元编号")
        if is_node and self._node_ids and self.target_id_spin.value() not in self._node_ids:
            self.target_id_spin.setValue(self._node_ids[0])
        if not is_node and self._element_ids and self.target_id_spin.value() not in self._element_ids:
            self.target_id_spin.setValue(self._element_ids[0])

    def _current_target_kind(self) -> ResultProbeKind:
        kinds = (
            ResultProbeKind.NODE,
            ResultProbeKind.ELEMENT,
            ResultProbeKind.INTEGRATION_POINT,
            ResultProbeKind.ELEMENT_NODE,
        )
        index = self.target_combo.currentIndex()
        return kinds[index if 0 <= index < len(kinds) else 0]

    def _default_target_index(self) -> int:
        position = self._current_availability().descriptor.field_id.position
        if position in {
            FieldPosition.NODE,
            FieldPosition.NODE_REGION,
            FieldPosition.RESOLVED_NODAL,
            FieldPosition.SECTION_NODE_ENVELOPE,
        }:
            return 0
        if position is FieldPosition.CENTROID:
            return 1
        if position is FieldPosition.INTEGRATION_POINT:
            return 2
        return 3

    def _refresh_availability(self) -> None:
        try:
            availability = self._current_availability()
        except RuntimeError as error:
            self.availability_label.setText(str(error))
            return
        self.availability_label.setText(
            {
                FieldState.READY: "字段已就绪；可直接提取。",
                FieldState.LAZY: "字段将在后台按每个增量分别物化，不改变最终结果快照。",
                FieldState.UNAVAILABLE: "字段不可用；请更换场变量。",
            }[availability.state]
        )
        self.target_combo.blockSignals(True)
        self.target_combo.setCurrentIndex(self._default_target_index())
        self.target_combo.blockSignals(False)
        self._refresh_target_controls()

    def current_request(self) -> ResultXYRequest:
        availability = self._current_availability()
        field_id = availability.descriptor.field_id
        if field_id.section_point_number is not None:
            raise ValueError(
                "当前 XY 数据暂不支持带截面点的结果字段；"
                "请先选择节点、质心或积分点结果。"
            )
        component = self.component_combo.currentText().strip()
        if not component:
            raise ValueError("请选择结果分量")
        target_kind = self._current_target_kind()
        expected_index = self._default_target_index()
        if self.target_combo.currentIndex() != expected_index:
            raise ValueError(
                "当前结果位置只能使用“"
                f"{self.target_combo.itemText(expected_index)}”目标"
            )
        target_id = int(self.target_id_spin.value())
        if target_kind is ResultProbeKind.NODE:
            target = ResultProbeTarget(target_kind, node_id=target_id)
        elif target_kind is ResultProbeKind.ELEMENT:
            target = ResultProbeTarget(target_kind, element_id=target_id)
        elif target_kind is ResultProbeKind.INTEGRATION_POINT:
            target = ResultProbeTarget(
                target_kind,
                element_id=target_id,
                integration_point=int(self.integration_point_spin.value()),
            )
        else:
            target = ResultProbeTarget(
                target_kind,
                element_id=target_id,
                local_node=int(self.local_node_spin.value()),
            )
        axis_index = self.axis_combo.currentIndex()
        axis = self._axis_values[axis_index]
        return ResultXYRequest(
            ScalarFieldSelection(availability.key, component),
            target,
            axis,
        )

    def request_curve(self) -> None:
        if self._pending:
            return
        try:
            request = self.current_request()
        except (RuntimeError, TypeError, ValueError) as error:
            QMessageBox.warning(self, "XY 数据", str(error))
            return
        self.generateRequested.emit(request)

    def set_pending(self, pending: bool) -> None:
        if type(pending) is not bool:
            raise TypeError("pending must be a bool")
        self._pending = pending
        for widget in (
            self.field_combo,
            self.component_combo,
            self.target_combo,
            self.target_id_spin,
            self.integration_point_spin,
            self.local_node_spin,
            self.axis_combo,
            self.curve_color_button,
            self.curve_line_style_combo,
            self.curve_grid_checkbox,
            self.curve_legend_checkbox,
            self.operation_combo,
            self.operation_button,
            self.path_button,
        ):
            widget.setEnabled(not pending)
        self.generate_button.setEnabled(not pending)
        has_series = bool(self._series_records)
        self.copy_button.setEnabled(not pending and has_series)
        self.save_button.setEnabled(not pending and has_series)
        self.rename_button.setEnabled(not pending and has_series)
        self.delete_button.setEnabled(not pending and has_series)
        if pending:
            self._set_chart_message("正在按增量读取结果并生成曲线……")

    def set_message(self, message: str) -> None:
        self.status_label.setText(str(message).strip() or "XY 数据未生成")

    def set_series(self, series: ResultXYSeries) -> None:
        if type(series) is not ResultXYSeries:
            raise TypeError("series must be ResultXYSeries")
        if series.source != self._catalog.source:
            raise ValueError("XY series source does not match dialog catalog")
        self._curve_counter += 1
        record = {
            "name": f"曲线-{self._curve_counter}",
            "series": series,
            "color": self.curve_color_button.text().strip() or "#1F77B4",
            "line_style": str(self.curve_line_style_combo.currentData() or "solid"),
            "visible": True,
        }
        self._series_records.append(record)
        self._refresh_series_list(select_index=len(self._series_records) - 1)
        self._refresh_table_for_selected()
        target = series.request.target
        self.set_message(
            f"已生成 {len(series.points)} 个点 · "
            f"{_target_label(target)} · 横轴：{_axis_label(series.request.axis)}"
        )
        self.copy_button.setEnabled(not self._pending)
        self.save_button.setEnabled(not self._pending)
        self.rename_button.setEnabled(not self._pending)
        self.delete_button.setEnabled(not self._pending)

    def _refresh_series_list(self, *, select_index: int | None = None) -> None:
        self.series_list.blockSignals(True)
        self.series_list.clear()
        for index, record in enumerate(self._series_records):
            item = QListWidgetItem(str(record["name"]), self.series_list)
            item.setData(Qt.ItemDataRole.UserRole, index)
            item.setFlags(
                item.flags()
                | Qt.ItemFlag.ItemIsUserCheckable
            )
            item.setCheckState(
                Qt.CheckState.Checked
                if bool(record.get("visible", True))
                else Qt.CheckState.Unchecked
            )
        self.series_list.blockSignals(False)
        if self._series_records:
            row = (
                len(self._series_records) - 1
                if select_index is None
                else max(0, min(select_index, len(self._series_records) - 1))
            )
            self.series_list.setCurrentRow(row)
        else:
            self.table.clearContents()
            self.table.setRowCount(0)
            self._set_chart_message("请选择结果字段和目标位置后生成曲线")

    def _selected_record_indices(self) -> tuple[int, ...]:
        indices: list[int] = []
        for item in self.series_list.selectedItems():
            index = item.data(Qt.ItemDataRole.UserRole)
            if type(index) is int and 0 <= index < len(self._series_records):
                indices.append(index)
        if not indices:
            row = self.series_list.currentRow()
            if 0 <= row < len(self._series_records):
                indices.append(row)
        return tuple(dict.fromkeys(indices))

    def _current_record(self) -> dict[str, object] | None:
        indices = self._selected_record_indices()
        return self._series_records[indices[0]] if indices else None

    def _series_selection_changed(self, _row: int) -> None:
        self._load_current_curve_style()
        self._refresh_table_for_selected()
        self._refresh_chart_options()

    def _load_current_curve_style(self) -> None:
        """Keep the editor controls synchronized with the active curve.

        The curve list is intentionally multi-selectable for operations, but
        the colour/style editor always describes the current row.  Without
        this synchronization it is very easy to edit curve B while the
        button still displays curve A's style.
        """

        record = self._current_record()
        if record is None:
            return
        color = str(record.get("color") or "#1F77B4")
        self.curve_color_button.setText(color)
        self.curve_color_button.setStyleSheet(
            "QPushButton { background-color: "
            f"{color}; color: white; padding: 3px 14px; }}"
        )
        style = str(record.get("line_style") or "solid")
        self.curve_line_style_combo.blockSignals(True)
        index = self.curve_line_style_combo.findData(style)
        self.curve_line_style_combo.setCurrentIndex(max(0, index))
        self.curve_line_style_combo.blockSignals(False)

    def _series_visibility_changed(self, item: QListWidgetItem) -> None:
        index = item.data(Qt.ItemDataRole.UserRole)
        if type(index) is int and 0 <= index < len(self._series_records):
            self._series_records[index]["visible"] = (
                item.checkState() is Qt.CheckState.Checked
            )
        self._replace_chart()

    def _refresh_table_for_selected(self) -> None:
        record = self._current_record()
        series = record.get("series") if record is not None else None
        if not isinstance(series, ResultXYSeries):
            self.table.setRowCount(0)
            return
        self.table.setRowCount(len(series.points))
        for row, point in enumerate(series.points):
            values = (
                str(point.frame_index),
                f"{point.x_value:.8g}",
                f"{point.y_value:.8g}",
            )
            for column, value in enumerate(values):
                self.table.setItem(row, column, QTableWidgetItem(value))
        self.table.resizeColumnsToContents()

    def _choose_curve_color(self) -> None:
        current = QColor(self.curve_color_button.text().strip() or "#1F77B4")
        color = QColorDialog.getColor(current, self, "选择曲线颜色")
        if not color.isValid():
            return
        value = color.name(QColor.NameFormat.HexArgb)
        self.curve_color_button.setText(value)
        self.curve_color_button.setStyleSheet(
            f"QPushButton {{ background-color: {value}; color: white; padding: 3px 14px; }}"
        )
        self._apply_current_curve_style()

    def _apply_current_curve_style(self) -> None:
        record = self._current_record()
        if record is None:
            return
        record["color"] = self.curve_color_button.text().strip() or "#1F77B4"
        record["line_style"] = str(
            self.curve_line_style_combo.currentData() or "solid"
        )
        self._replace_chart()

    def _refresh_chart_options(self, *_args: object) -> None:
        self.chart.legend().setVisible(self.curve_legend_checkbox.isChecked())
        self._replace_chart()

    def _replace_chart(self, _series: ResultXYSeries | None = None) -> None:
        records = [
            record
            for record in self._series_records
            if bool(record.get("visible", True))
            and isinstance(record.get("series"), ResultXYSeries)
        ]
        chart = self.chart
        chart.removeAllSeries()
        for axis in tuple(chart.axes()):
            chart.removeAxis(axis)
        if not records:
            self._set_chart_message(
                "请选择结果字段和目标位置后生成曲线"
                if not self._series_records
                else "曲线已隐藏；请在曲线列表中勾选要显示的曲线"
            )
            return
        series_values = [record["series"] for record in records]
        first_series = series_values[0]
        assert isinstance(first_series, ResultXYSeries)
        lines: list[QLineSeries] = []
        all_x: list[float] = []
        all_y: list[float] = []
        for record in records:
            series = record["series"]
            assert isinstance(series, ResultXYSeries)
            line = QLineSeries(chart)
            line.setName(str(record["name"]))
            pen = QPen(QColor(str(record.get("color") or "#1F77B4")))
            pen.setWidthF(2.0)
            pen.setStyle(_line_style(str(record.get("line_style") or "solid")))
            line.setPen(pen)
            for point in series.points:
                line.append(point.x_value, point.y_value)
                all_x.append(point.x_value)
                all_y.append(point.y_value)
            chart.addSeries(line)
            lines.append(line)
        x_axis = QValueAxis(chart)
        y_axis = QValueAxis(chart)
        x_axis.setTitleText(_axis_label(first_series.request.axis))
        y_axis.setTitleText(first_series.request.selection.component)
        x_min = min(all_x)
        x_max = max(all_x)
        y_min = min(all_y)
        y_max = max(all_y)
        x_axis.setRange(*_expanded_range(x_min, x_max))
        y_axis.setRange(*_expanded_range(y_min, y_max))
        x_axis.setGridLineVisible(self.curve_grid_checkbox.isChecked())
        y_axis.setGridLineVisible(self.curve_grid_checkbox.isChecked())
        chart.addAxis(x_axis, Qt.AlignmentFlag.AlignBottom)
        chart.addAxis(y_axis, Qt.AlignmentFlag.AlignLeft)
        for line in lines:
            line.attachAxis(x_axis)
            line.attachAxis(y_axis)
        chart.legend().setVisible(self.curve_legend_checkbox.isChecked())
        chart.setTitle(
            f"{first_series.request.selection.component} · "
            f"{_axis_label(first_series.request.axis)}"
        )
        self.chart_view.update()

    def _set_chart_message(self, message: str) -> None:
        self.chart.removeAllSeries()
        for axis in tuple(self.chart.axes()):
            self.chart.removeAxis(axis)
        self.chart.setTitle(str(message))

    def copy_data(self) -> None:
        self._refresh_table_for_selected()
        headers = [
            self.table.horizontalHeaderItem(column).text()
            for column in range(self.table.columnCount())
        ]
        rows = [headers]
        rows.extend(
            [
                self.table.item(row, column).text()
                for column in range(self.table.columnCount())
            ]
            for row in range(self.table.rowCount())
        )
        QApplication.clipboard().setText(
            "\n".join("\t".join(row) for row in rows)
        )

    def save_data(self) -> None:
        record = self._current_record()
        series = record.get("series") if record is not None else None
        if not isinstance(series, ResultXYSeries):
            return
        name = str(record.get("name") or "xy-data")
        path, _filter = QFileDialog.getSaveFileName(
            self,
            "保存 XY 数据",
            f"{name}.csv",
            "CSV 文件 (*.csv);;文本文件 (*.txt)",
        )
        if not path:
            return
        target = Path(path)
        lines = ["曲线,增量,X,Y"]
        lines.extend(
            f"{name},{point.frame_index},{point.x_value:.16g},{point.y_value:.16g}"
            for point in series.points
        )
        try:
            target.write_text("\n".join(lines) + "\n", encoding="utf-8")
        except OSError as error:
            QMessageBox.warning(self, "保存 XY 数据", str(error))
            return
        self.set_message(f"已保存曲线：{target.name}")

    def rename_selected_series(self) -> None:
        indices = self._selected_record_indices()
        if not indices:
            return
        index = indices[0]
        current = str(self._series_records[index].get("name") or "")
        name, accepted = QInputDialog.getText(
            self,
            "重命名曲线",
            "曲线名称：",
            text=current,
        )
        if not accepted or not name.strip():
            return
        self._series_records[index]["name"] = name.strip()
        self._refresh_series_list(select_index=index)
        self._replace_chart()

    def delete_selected_series(self) -> None:
        indices = set(self._selected_record_indices())
        if not indices:
            return
        self._series_records = [
            record
            for index, record in enumerate(self._series_records)
            if index not in indices
        ]
        self._refresh_series_list(
            select_index=min(indices) if self._series_records else None
        )
        self._refresh_table_for_selected()
        self._replace_chart()

    def apply_operation(self) -> None:
        indices = self._selected_record_indices()
        operation = str(self.operation_combo.currentData() or "")
        required = 2 if operation == "difference" else 1
        if len(indices) != required:
            QMessageBox.warning(
                self,
                "曲线运算",
                f"{self.operation_combo.currentText()}需要选择 {required} 条曲线。",
            )
            return
        selected = [self._series_records[index].get("series") for index in indices]
        if not all(isinstance(series, ResultXYSeries) for series in selected):
            return
        try:
            if operation == "difference":
                result = xy_difference(selected[0], selected[1])
            elif operation == "derivative":
                result = xy_derivative(selected[0])
            elif operation == "integral":
                result = xy_integral(selected[0])
            else:
                result = xy_envelope(selected[0])
        except (TypeError, ValueError, RuntimeError) as error:
            QMessageBox.warning(self, "曲线运算", str(error))
            return
        self.set_series(result)
        record = self._series_records[-1]
        record["name"] = {
            "difference": "差值",
            "derivative": "导数",
            "integral": "积分",
            "envelope": "包络",
        }.get(operation, "运算") + f"-{self._curve_counter}"
        self._refresh_series_list(select_index=len(self._series_records) - 1)
        self._replace_chart()


def _expanded_range(minimum: float, maximum: float) -> tuple[float, float]:
    if minimum == maximum:
        padding = max(abs(minimum) * 0.05, 1.0)
        return minimum - padding, maximum + padding
    padding = (maximum - minimum) * 0.05
    return minimum - padding, maximum + padding


def _axis_label(axis: ResultXYAxis) -> str:
    return {
        ResultXYAxis.INCREMENT: "增量",
        ResultXYAxis.STEP_TIME: "步时间",
        ResultXYAxis.TOTAL_TIME: "总时间",
        ResultXYAxis.LOAD_FACTOR: "载荷因子",
    }[axis]


def _target_label(target: ResultProbeTarget) -> str:
    if target.kind is ResultProbeKind.NODE:
        return f"节点 {target.node_id}"
    if target.kind is ResultProbeKind.ELEMENT:
        return f"单元 {target.element_id}"
    if target.kind is ResultProbeKind.INTEGRATION_POINT:
        return f"单元 {target.element_id} · 积分点 {target.integration_point}"
    return f"单元 {target.element_id} · 局部节点 {target.local_node}"


def _line_style(value: str) -> Qt.PenStyle:
    return {
        "solid": Qt.PenStyle.SolidLine,
        "dashed": Qt.PenStyle.DashLine,
        "dashdot": Qt.PenStyle.DashDotLine,
        "dotted": Qt.PenStyle.DotLine,
    }.get(value, Qt.PenStyle.SolidLine)


__all__ = ["XYDataDialog"]
