"""结果查询、显示和云图设置弹窗。"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
import re
from typing import Any

from fem.application.results import (
    FieldAssociation,
    FieldAvailability,
    FieldMaterializationKey,
    FieldState,
    ResultCatalog,
    ResultProvider,
    ResultQuery,
    ResultQueryRecord,
    ResultQueryResult,
    ResultSourceKey,
    ScalarFieldSelection,
)
from fem.post.fields import encode_result_region_key
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
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

from .dialogs import CompactDoubleSpinBox, configure_form_layout
from .result_presentation import (
    result_field_is_visible,
    visible_result_fields,
)
from .visualization.colormaps import ABAQUS_RAINBOW
from .visualization.contour_rendering import (
    CONTOUR_EDGE_ALL,
    CONTOUR_EDGE_EXTERIOR,
    CONTOUR_EDGE_FEATURE,
    CONTOUR_EDGE_FREE,
    CONTOUR_EDGE_NONE,
    CONTOUR_RENDER_FILLED,
    CONTOUR_RENDER_SHADED,
)


@dataclass(frozen=True, slots=True)
class TypedResultDisplaySettings:
    """Catalog-native result display state with a complete field selection."""

    shape_mode: str
    contour_enabled: bool
    selection: ScalarFieldSelection
    scale_mode: str
    scale_value: float
    overlay_undeformed: bool
    show_edges: bool

    def __post_init__(self) -> None:
        _validate_typed_display_options(
            shape_mode=self.shape_mode,
            contour_enabled=self.contour_enabled,
            selection=self.selection,
            scale_mode=self.scale_mode,
            scale_value=self.scale_value,
            overlay_undeformed=self.overlay_undeformed,
            show_edges=self.show_edges,
        )


@dataclass(frozen=True, slots=True)
class _TypedQueryMode:
    association: FieldAssociation

    def __post_init__(self) -> None:
        if type(self.association) is not FieldAssociation:
            raise TypeError("association must be FieldAssociation")


class TypedResultDisplayDialog(QDialog):
    """从 immutable result catalog 选择完整的 scalar field identity。"""

    applyRequested = Signal(TypedResultDisplaySettings)

    def __init__(
        self,
        catalog: ResultCatalog,
        *,
        current_selection: ScalarFieldSelection,
        shape_mode: str,
        contour_enabled: bool,
        scale_mode: str,
        scale_value: float,
        overlay_undeformed: bool,
        show_edges: bool,
        parent=None,
    ) -> None:
        if type(catalog) is not ResultCatalog:
            raise TypeError("catalog must be ResultCatalog")
        _validate_typed_display_selection(catalog, current_selection)
        initial = TypedResultDisplaySettings(
            shape_mode=shape_mode,
            contour_enabled=contour_enabled,
            selection=current_selection,
            scale_mode=scale_mode,
            scale_value=scale_value,
            overlay_undeformed=overlay_undeformed,
            show_edges=show_edges,
        )

        super().__init__(parent)
        self.setWindowTitle("结果显示")
        self.setMinimumWidth(420)
        self._catalog = catalog

        layout = QVBoxLayout(self)
        form = QFormLayout()
        configure_form_layout(form)
        self.step_combo = QComboBox(self)
        self.step_combo.addItem(catalog.source.step_name, catalog.source)
        self.shape_combo = QComboBox(self)
        self.shape_combo.addItem("未变形形状", "undeformed")
        self.shape_combo.addItem("变形形状", "deformed")
        self.shape_combo.setCurrentIndex(
            self.shape_combo.findData(initial.shape_mode)
        )
        self.contour_checkbox = QCheckBox("显示云图", self)
        self.contour_checkbox.setChecked(initial.contour_enabled)
        self.field_combo = QComboBox(self)
        self.component_combo = QComboBox(self)
        self.availability_label = QLabel(self)
        self.availability_label.setWordWrap(True)
        for availability in visible_result_fields(catalog.fields):
            self.field_combo.addItem(
                _typed_result_display_field_label(availability),
                availability.key,
            )
        self.field_combo.setCurrentIndex(
            self.field_combo.findData(current_selection.field_key)
        )
        form.addRow("结果步：", self.step_combo)
        form.addRow("几何形状：", self.shape_combo)
        form.addRow(self.contour_checkbox)
        form.addRow("场变量：", self.field_combo)
        form.addRow("分量：", self.component_combo)
        form.addRow("字段状态：", self.availability_label)
        layout.addLayout(form)

        self.scale_group = QGroupBox("变形比例", self)
        scale_layout = QVBoxLayout(self.scale_group)
        self.auto_scale = QRadioButton("自动", self.scale_group)
        self.real_scale = QRadioButton("真实比例", self.scale_group)
        self.custom_scale = QRadioButton("指定比例", self.scale_group)
        scale_buttons = QButtonGroup(self.scale_group)
        for button in (
            self.auto_scale,
            self.real_scale,
            self.custom_scale,
        ):
            scale_buttons.addButton(button)
        self.scale_value = CompactDoubleSpinBox(self.scale_group)
        self.scale_value.setRange(0.0, 1.0e12)
        self.scale_value.setDecimals(6)
        self.scale_value.setValue(initial.scale_value)
        custom_row = QHBoxLayout()
        custom_row.addWidget(self.custom_scale)
        custom_row.addWidget(self.scale_value, 1)
        scale_layout.addWidget(self.auto_scale)
        scale_layout.addWidget(self.real_scale)
        scale_layout.addLayout(custom_row)
        {
            "auto": self.auto_scale,
            "real": self.real_scale,
            "custom": self.custom_scale,
        }[initial.scale_mode].setChecked(True)
        layout.addWidget(self.scale_group)

        self.overlay_checkbox = QCheckBox("叠加未变形轮廓", self)
        self.overlay_checkbox.setChecked(initial.overlay_undeformed)
        self.edges_checkbox = QCheckBox("显示单元边", self)
        self.edges_checkbox.setChecked(initial.show_edges)
        layout.addWidget(self.overlay_checkbox)
        layout.addWidget(self.edges_checkbox)

        self.button_box = _dialog_buttons(self)
        self.apply_button = self.button_box.button(
            QDialogButtonBox.StandardButton.Apply
        )
        self.ok_button = self.button_box.button(
            QDialogButtonBox.StandardButton.Ok
        )
        self.apply_button.clicked.connect(self.apply)
        self.button_box.accepted.connect(self.accept_with_apply)
        self.button_box.rejected.connect(self.reject)
        layout.addWidget(self.button_box)

        self._sync_components(
            preferred_component=current_selection.component,
        )
        self.field_combo.currentIndexChanged.connect(
            self._field_changed
        )
        self.shape_combo.currentIndexChanged.connect(
            self._refresh_mode_state
        )
        self.contour_checkbox.toggled.connect(
            self._refresh_mode_state
        )
        self._refresh_availability()
        self._refresh_mode_state()

    @property
    def catalog(self) -> ResultCatalog:
        """返回 dialog 绑定的 exact immutable catalog。"""

        return self._catalog

    @property
    def source(self) -> ResultSourceKey:
        """返回 catalog 所属的 exact result source。"""

        return self._catalog.source

    def current_availability(self) -> FieldAvailability:
        """返回当前字段的完整 catalog entry。"""

        key = self.field_combo.currentData()
        if type(key) is not FieldMaterializationKey:
            raise RuntimeError("no typed result field is selected")
        for availability in self._catalog.fields:
            if availability.key == key:
                return availability
        raise RuntimeError("field key is outside the dialog catalog")

    def current_selection(self) -> ScalarFieldSelection:
        """返回当前完整 materialization key 与 descriptor component。"""

        availability = self.current_availability()
        component = self.component_combo.currentData()
        if type(component) is not str:
            raise RuntimeError("no typed scalar component is selected")
        if component not in availability.descriptor.columns:
            raise RuntimeError(
                "selected component is outside the field descriptor"
            )
        return ScalarFieldSelection(availability.key, component)

    def settings(self) -> TypedResultDisplaySettings:
        """返回当前 catalog-native 显示设置。"""

        scale_mode = (
            "auto"
            if self.auto_scale.isChecked()
            else "real"
            if self.real_scale.isChecked()
            else "custom"
        )
        return TypedResultDisplaySettings(
            shape_mode=self.shape_combo.currentData(),
            contour_enabled=self.contour_checkbox.isChecked(),
            selection=self.current_selection(),
            scale_mode=scale_mode,
            scale_value=float(self.scale_value.value()),
            overlay_undeformed=self.overlay_checkbox.isChecked(),
            show_edges=self.edges_checkbox.isChecked(),
        )

    def apply(self) -> None:
        """仅为 READY/LAZY selection 发出 typed settings。"""

        if self.current_availability().state is FieldState.UNAVAILABLE:
            return
        self.applyRequested.emit(self.settings())

    def accept_with_apply(self) -> None:
        """提交可显示字段并关闭对话框。"""

        if self.current_availability().state is FieldState.UNAVAILABLE:
            return
        self.apply()
        self.accept()

    def _field_changed(self, *_args: object) -> None:
        self._sync_components()
        self._refresh_availability()

    def _sync_components(
        self,
        *,
        preferred_component: str | None = None,
    ) -> None:
        if preferred_component is None:
            candidate = self.component_combo.currentData()
            if type(candidate) is str:
                preferred_component = candidate
        self.component_combo.blockSignals(True)
        self.component_combo.clear()
        availability = self.current_availability()
        for component in availability.descriptor.columns:
            self.component_combo.addItem(component, component)
        selected_index = self.component_combo.findData(preferred_component)
        if selected_index < 0:
            selected_index = self.component_combo.findData(
                availability.descriptor.default_component
            )
        self.component_combo.setCurrentIndex(
            selected_index if selected_index >= 0 else 0
        )
        self.component_combo.blockSignals(False)

    def _refresh_availability(self) -> None:
        availability = self.current_availability()
        self.availability_label.setText(
            _typed_result_display_availability_text(availability)
        )
        can_submit = availability.state is not FieldState.UNAVAILABLE
        self.apply_button.setEnabled(can_submit)
        self.ok_button.setEnabled(can_submit)

    def _refresh_mode_state(self) -> None:
        contour_enabled = self.contour_checkbox.isChecked()
        self.field_combo.setEnabled(contour_enabled)
        self.component_combo.setEnabled(contour_enabled)
        deformed = self.shape_combo.currentData() == "deformed"
        self.scale_group.setEnabled(deformed)
        self.overlay_checkbox.setEnabled(deformed)


class ContourSettingsDialog(QDialog):
    """控制云图范围、色带、数值格式、图例和极值。"""

    applyRequested = Signal(object)

    def __init__(self, options: dict[str, Any], parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("云图设置")
        self.setMinimumWidth(450)
        layout = QVBoxLayout(self)
        range_group = QGroupBox("范围", self)
        range_layout = QVBoxLayout(range_group)
        self.auto_range = QRadioButton("自动", range_group)
        self.manual_range = QRadioButton("手动", range_group)
        (self.manual_range if options.get("manual") else self.auto_range).setChecked(True)
        range_buttons = QButtonGroup(range_group)
        range_buttons.addButton(self.auto_range)
        range_buttons.addButton(self.manual_range)
        self.minimum = CompactDoubleSpinBox(range_group)
        self.maximum = CompactDoubleSpinBox(range_group)
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
        configure_form_layout(form)
        self.colormap = QComboBox(self)
        for label, key in (
            ("Abaqus 彩虹", ABAQUS_RAINBOW),
            ("维里迪斯", "viridis"),
            ("等离子", "plasma"),
            ("冷暖", "coolwarm"),
            ("灰度", "gray"),
        ):
            self.colormap.addItem(label, key)
        selected_colormap = options.get("colormap", ABAQUS_RAINBOW)
        if selected_colormap == "jet":
            selected_colormap = ABAQUS_RAINBOW
        self.colormap.setCurrentIndex(
            max(0, self.colormap.findData(selected_colormap))
        )
        self.levels = QSpinBox(self)
        self.levels.setRange(2, 256)
        self.levels.setValue(int(options.get("levels", 12)))
        self.style = QComboBox(self)
        self.style.addItem("分段云图", "segmented")
        self.style.addItem("连续云图", "continuous")
        self.style.setCurrentIndex(max(0, self.style.findData(options.get("style", "segmented"))))
        self.style.currentIndexChanged.connect(
            lambda: self.levels.setEnabled(self.style.currentData() == "segmented")
        )
        self.levels.setEnabled(self.style.currentData() == "segmented")
        self.render_mode = QComboBox(self)
        self.render_mode.addItem("光影", CONTOUR_RENDER_SHADED)
        self.render_mode.addItem("填充", CONTOUR_RENDER_FILLED)
        self.render_mode.setCurrentIndex(
            max(
                0,
                self.render_mode.findData(
                    options.get(
                        "render_mode",
                        CONTOUR_RENDER_SHADED,
                    )
                ),
            )
        )
        self.edge_mode = QComboBox(self)
        for label, key in (
            ("全部边", CONTOUR_EDGE_ALL),
            ("外部边", CONTOUR_EDGE_EXTERIOR),
            ("特征边", CONTOUR_EDGE_FEATURE),
            ("自由边", CONTOUR_EDGE_FREE),
            ("无边", CONTOUR_EDGE_NONE),
        ):
            self.edge_mode.addItem(label, key)
        selected_edge_mode = options.get(
            "edge_mode",
            (
                CONTOUR_EDGE_ALL
                if options.get("edges", False)
                else CONTOUR_EDGE_NONE
            ),
        )
        if not options.get("edges", False):
            selected_edge_mode = CONTOUR_EDGE_NONE
        self.edge_mode.setCurrentIndex(
            max(0, self.edge_mode.findData(selected_edge_mode))
        )
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
        self.orientation.setCurrentIndex(max(0, self.orientation.findData(options.get("orientation", "vertical"))))
        form.addRow("色带：", self.colormap)
        form.addRow("云图样式：", self.style)
        form.addRow("渲染模式：", self.render_mode)
        form.addRow("显示轮廓：", self.edge_mode)
        form.addRow("色带级数：", self.levels)
        form.addRow("节点平均阈值：", self.averaging_threshold)
        form.addRow("数值格式：", self.number_format)
        form.addRow("小数位：", self.decimals)
        form.addRow("图例方向：", self.orientation)
        layout.addLayout(form)
        self.legend = QCheckBox("显示图例", self)
        self.legend.setChecked(bool(options.get("legend", True)))
        self.show_minimum = QCheckBox("显示最小值", self)
        self.show_minimum.setChecked(bool(options.get("show_minimum", False)))
        self.show_maximum = QCheckBox("显示最大值", self)
        self.show_maximum.setChecked(bool(options.get("show_maximum", False)))
        self.show_ids = QCheckBox("显示对象编号", self)
        self.show_ids.setChecked(bool(options.get("show_ids", False)))
        for checkbox in (
            self.legend,
            self.show_minimum,
            self.show_maximum,
            self.show_ids,
        ):
            layout.addWidget(checkbox)
        buttons = _dialog_buttons(self)
        buttons.button(QDialogButtonBox.StandardButton.Apply).clicked.connect(self.apply)
        buttons.accepted.connect(self.accept_with_apply)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def settings(self) -> dict[str, Any]:
        edge_mode = str(self.edge_mode.currentData())
        return {
            "manual": self.manual_range.isChecked(),
            "minimum": float(self.minimum.value()),
            "maximum": float(self.maximum.value()),
            "colormap": str(self.colormap.currentData()),
            "style": str(self.style.currentData()),
            "render_mode": str(self.render_mode.currentData()),
            "edge_mode": edge_mode,
            "levels": int(self.levels.value()),
            "number_format": str(self.number_format.currentData()),
            "decimals": int(self.decimals.value()),
            "orientation": str(self.orientation.currentData()),
            "legend": self.legend.isChecked(),
            "show_minimum": self.show_minimum.isChecked(),
            "show_maximum": self.show_maximum.isChecked(),
            "show_ids": self.show_ids.isChecked(),
            "edges": edge_mode != CONTOUR_EDGE_NONE,
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


class TypedResultQueryDialog(QDialog):
    """用 provider catalog 构造精确查询，数值工作由外层命令完成。"""

    selectionRequested = Signal(object)
    queryRequested = Signal(object)

    def __init__(
        self,
        provider: ResultProvider,
        catalog: ResultCatalog | None = None,
        *,
        parent=None,
    ) -> None:
        if type(provider) is not ResultProvider:
            raise TypeError("provider must be ResultProvider")
        provider_catalog = provider.catalog()
        if catalog is None:
            catalog = provider_catalog
        elif type(catalog) is not ResultCatalog:
            raise TypeError("catalog must be ResultCatalog or None")
        elif catalog != provider_catalog:
            raise ValueError("catalog must exactly match provider.catalog()")

        super().__init__(parent)
        self.setWindowTitle("查询结果")
        self.resize(900, 520)
        self._catalog = catalog
        self._source = provider.source
        self._initial_generation = provider.snapshot.generation
        self._node_ids = provider.snapshot.topology.node_ids
        self._element_ids = provider.snapshot.topology.element_ids
        self._last_query: ResultQuery | None = None
        self._displayed_generation: int | None = None
        self._query_pending = False

        layout = QVBoxLayout(self)
        form = QFormLayout()
        configure_form_layout(form)
        self.step_combo = QComboBox(self)
        self.step_combo.addItem(self._source.step_name, self._source)
        self.association_combo = QComboBox(self)
        self.association_combo.addItem(
            "节点",
            _TypedQueryMode(FieldAssociation.NODE),
        )
        self.association_combo.addItem(
            "单元",
            _TypedQueryMode(FieldAssociation.ELEMENT),
        )
        self.field_combo = QComboBox(self)
        self.component_combo = QComboBox(self)
        self.ids_edit = QLineEdit(self)
        self.ids_edit.setPlaceholderText("留空查询全部；例如：1, 3, 5-8")
        self.availability_label = QLabel(self)
        form.addRow("结果步：", self.step_combo)
        form.addRow("对象类型：", self.association_combo)
        form.addRow("场变量：", self.field_combo)
        form.addRow("分量：", self.component_combo)
        form.addRow("对象编号：", self.ids_edit)
        form.addRow("字段状态：", self.availability_label)
        layout.addLayout(form)

        command_row = QHBoxLayout()
        self.query_button = QPushButton("查询", self)
        copy_button = QPushButton("复制", self)
        close_button = QPushButton("关闭", self)
        self.query_button.clicked.connect(self.request_query)
        copy_button.clicked.connect(self.copy_table)
        close_button.clicked.connect(self.close)
        command_row.addWidget(self.query_button)
        command_row.addWidget(copy_button)
        command_row.addStretch(1)
        command_row.addWidget(close_button)
        layout.addLayout(command_row)

        self.result_summary = QLabel("尚未查询", self)
        layout.addWidget(self.result_summary)
        self.table = QTableWidget(self)
        self.table.setEditTriggers(
            QTableWidget.EditTrigger.NoEditTriggers
        )
        self.table.setSelectionBehavior(
            QTableWidget.SelectionBehavior.SelectRows
        )
        layout.addWidget(self.table, 1)
        self._prepare_table()

        default_availability = self._availability_for_key(
            self._catalog.default_selection.field_key
        )
        default_mode = (
            FieldAssociation.NODE
            if _typed_query_association_matches(
                FieldAssociation.NODE,
                default_availability.descriptor.association,
            )
            else FieldAssociation.ELEMENT
        )
        self.association_combo.setCurrentIndex(
            0 if default_mode is FieldAssociation.NODE else 1
        )
        self._sync_fields(
            preferred_key=self._catalog.default_selection.field_key,
            emit_selection=False,
        )
        self.association_combo.currentIndexChanged.connect(
            self._association_changed
        )
        self.field_combo.currentIndexChanged.connect(self._field_changed)
        self.component_combo.currentIndexChanged.connect(
            self._component_changed
        )

    @property
    def catalog(self) -> ResultCatalog:
        """返回 dialog 绑定的 exact immutable catalog。"""

        return self._catalog

    @property
    def source(self) -> ResultSourceKey:
        """返回 dialog 打开时绑定的完整结果来源。"""

        return self._source

    @property
    def query_pending(self) -> bool:
        """返回是否已有一个 typed query 正在执行。"""

        return self._query_pending

    def set_query_pending(self, pending: bool) -> None:
        """在一个查询生命周期内冻结所有 query intent 控件。"""

        if type(pending) is not bool:
            raise TypeError("pending must be a bool")
        self._query_pending = pending
        enabled = not pending
        self.association_combo.setEnabled(enabled)
        self.field_combo.setEnabled(enabled)
        self.component_combo.setEnabled(enabled)
        self.ids_edit.setEnabled(enabled)
        if pending:
            self.result_summary.setText("正在查询……")
        self._refresh_availability()

    def set_query_message(self, message: str) -> None:
        """显示不携带 records 的查询状态消息。"""

        if type(message) is not str:
            raise TypeError("message must be a string")
        self.result_summary.setText(message.strip() or "结果查询未完成")

    def current_availability(self) -> FieldAvailability:
        """返回当前字段的 typed catalog entry。"""

        key = self.field_combo.currentData()
        if type(key) is not FieldMaterializationKey:
            raise RuntimeError("no typed result field is selected")
        return self._availability_for_key(key)

    def current_selection(self) -> ScalarFieldSelection:
        """返回当前完整 field key 与 scalar component。"""

        availability = self.current_availability()
        component = self.component_combo.currentData()
        if type(component) is not str:
            raise RuntimeError("no typed scalar component is selected")
        if component not in availability.descriptor.columns:
            raise RuntimeError(
                "selected component is outside the field descriptor"
            )
        return ScalarFieldSelection(availability.key, component)

    def current_query(self) -> ResultQuery:
        """根据 typed association 与 FEM ID 输入构造精确查询。"""

        availability = self.current_availability()
        if availability.state is FieldState.UNAVAILABLE:
            raise ValueError("当前字段不可查询。")
        selection = self.current_selection()
        mode = self.association_combo.currentData()
        if type(mode) is not _TypedQueryMode:
            raise RuntimeError("query association must be typed")
        if mode.association is FieldAssociation.NODE:
            node_ids = _parse_typed_query_ids(
                self.ids_edit.text(),
                self._node_ids,
            )
            element_ids: tuple[int, ...] = ()
        elif mode.association is FieldAssociation.ELEMENT:
            node_ids = ()
            element_ids = _parse_typed_query_ids(
                self.ids_edit.text(),
                self._element_ids,
            )
        else:
            raise RuntimeError("query association must be typed")
        return ResultQuery(
            field_key=selection.field_key,
            component=selection.component,
            node_ids=node_ids,
            element_ids=element_ids,
        )

    def request_query(self, *_args: object) -> None:
        """把 selection/query 交给外层，不在 dialog 内恢复或读取字段。"""

        if self._query_pending:
            return
        try:
            selection = self.current_selection()
            query = self.current_query()
        except (RuntimeError, ValueError) as error:
            QMessageBox.warning(self, "查询结果", str(error))
            return
        self._last_query = query
        self.selectionRequested.emit(selection)
        self.queryRequested.emit(query)

    def set_query_result(self, result: ResultQueryResult) -> None:
        """按 application 结果原序显示全部 query records。"""

        if type(result) is not ResultQueryResult:
            raise TypeError("result must be ResultQueryResult")
        if result.source != self._source:
            raise ValueError("query result source must match the dialog source")
        minimum_generation = self._initial_generation
        if self._displayed_generation is not None:
            minimum_generation = max(
                minimum_generation,
                self._displayed_generation,
            )
        if result.materialization_generation < minimum_generation:
            raise ValueError("query result generation is stale")
        expected_query = self._last_query
        if expected_query is None:
            expected_query = self.current_query()
        if result.query != expected_query:
            raise ValueError("query result must match the latest dialog query")

        self.table.setRowCount(len(result.records))
        for row, record in enumerate(result.records):
            location = record.location
            values = (
                _typed_association_text(location.association),
                _optional_identity_text(location.node_id),
                _optional_identity_text(location.element_id),
                _optional_identity_text(location.integration_point),
                _optional_identity_text(location.local_node),
                (
                    ""
                    if location.region_key is None
                    else encode_result_region_key(location.region_key)
                ),
                _averaged_text(location.averaged),
                _number_text(location.coordinates[0]),
                _number_text(location.coordinates[1]),
                _number_text(location.coordinates[2]),
                _number_text(record.value),
            )
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                if column == 0:
                    item.setData(Qt.ItemDataRole.UserRole, record)
                self.table.setItem(row, column, item)
        self.table.resizeColumnsToContents()
        if self.table.rowCount():
            self.table.selectRow(0)
        self._displayed_generation = result.materialization_generation
        self.result_summary.setText(
            f"共 {len(result.records)} 行 · "
            f"materialization generation "
            f"{result.materialization_generation}"
        )

    def show_result(self, result: ResultQueryResult) -> None:
        """兼容 Qt slot 风格的显式展示入口。"""

        self.set_query_result(result)

    def record_at(self, row: int) -> ResultQueryRecord:
        """返回显示行绑定的 exact ResultQueryRecord。"""

        if type(row) is not int:
            raise TypeError("row must be an integer")
        if row < 0 or row >= self.table.rowCount():
            raise IndexError(row)
        item = self.table.item(row, 0)
        if item is None:
            raise RuntimeError("query table row is incomplete")
        record = item.data(Qt.ItemDataRole.UserRole)
        if type(record) is not ResultQueryRecord:
            raise RuntimeError("query table row lost its typed record")
        return record

    def copy_table(self) -> None:
        """复制当前 typed query 表格。"""

        QApplication.clipboard().setText(self._table_text("\t"))

    def _association_changed(self, *_args: object) -> None:
        if self._query_pending:
            return
        self._last_query = None
        self._sync_fields(emit_selection=True)

    def _field_changed(self, *_args: object) -> None:
        if self._query_pending:
            return
        self._last_query = None
        self._sync_components()
        self._refresh_availability()
        self._emit_selection_requested()

    def _component_changed(self, *_args: object) -> None:
        if self._query_pending:
            return
        self._last_query = None
        self._emit_selection_requested()

    def _sync_fields(
        self,
        *,
        preferred_key: FieldMaterializationKey | None = None,
        emit_selection: bool,
    ) -> None:
        if preferred_key is None:
            candidate = self.field_combo.currentData()
            if type(candidate) is FieldMaterializationKey:
                preferred_key = candidate
        mode = self.association_combo.currentData()
        self.field_combo.blockSignals(True)
        self.field_combo.clear()
        if type(mode) is _TypedQueryMode:
            for availability in visible_result_fields(
                self._catalog.fields
            ):
                if _typed_query_association_matches(
                    mode.association,
                    availability.descriptor.association,
                ):
                    self.field_combo.addItem(
                        _typed_field_label(availability),
                        availability.key,
                    )
        selected_index = self.field_combo.findData(preferred_key)
        if selected_index < 0:
            default_key = self._catalog.default_selection.field_key
            selected_index = self.field_combo.findData(default_key)
        self.field_combo.setCurrentIndex(
            selected_index if selected_index >= 0 else 0
        )
        self.field_combo.blockSignals(False)
        self._sync_components()
        self._refresh_availability()
        if emit_selection:
            self._emit_selection_requested()

    def _sync_components(self) -> None:
        current = self.component_combo.currentData()
        self.component_combo.blockSignals(True)
        self.component_combo.clear()
        try:
            availability = self.current_availability()
        except RuntimeError:
            self.component_combo.blockSignals(False)
            return
        for component in availability.descriptor.columns:
            self.component_combo.addItem(component, component)
        selected_component = current
        if (
            availability.key
            == self._catalog.default_selection.field_key
        ):
            selected_component = self._catalog.default_selection.component
        index = self.component_combo.findData(selected_component)
        if index < 0:
            index = self.component_combo.findData(
                availability.descriptor.default_component
            )
        self.component_combo.setCurrentIndex(index if index >= 0 else 0)
        self.component_combo.blockSignals(False)

    def _refresh_availability(self) -> None:
        try:
            availability = self.current_availability()
        except RuntimeError:
            self.availability_label.setText("没有适用于当前对象的字段")
            self.query_button.setEnabled(False)
            return
        self.availability_label.setText(
            _typed_availability_text(availability)
        )
        self.query_button.setEnabled(
            not self._query_pending
            and availability.state is not FieldState.UNAVAILABLE
        )

    def _emit_selection_requested(self) -> None:
        try:
            availability = self.current_availability()
            selection = self.current_selection()
        except RuntimeError:
            return
        if availability.state is not FieldState.UNAVAILABLE:
            self.selectionRequested.emit(selection)

    def _availability_for_key(
        self,
        key: FieldMaterializationKey,
    ) -> FieldAvailability:
        for availability in self._catalog.fields:
            if availability.key == key:
                return availability
        raise RuntimeError("field key is outside the dialog catalog")

    def _prepare_table(self) -> None:
        headers = (
            "关联",
            "节点",
            "单元",
            "积分点",
            "局部节点",
            "区域",
            "平均状态",
            "X",
            "Y",
            "Z",
            "值",
        )
        self.table.clear()
        self.table.setColumnCount(len(headers))
        self.table.setHorizontalHeaderLabels(headers)
        self.table.setRowCount(0)

    def _table_rows(self) -> list[list[str]]:
        return [
            [
                self.table.horizontalHeaderItem(column).text()
                for column in range(self.table.columnCount())
            ],
            *[
                [
                    self.table.item(row, column).text()
                    for column in range(self.table.columnCount())
                ]
                for row in range(self.table.rowCount())
            ],
        ]

    def _table_text(self, separator: str) -> str:
        return "\n".join(
            separator.join(row) for row in self._table_rows()
        )


def _typed_query_association_matches(
    mode: FieldAssociation,
    association: FieldAssociation,
) -> bool:
    if mode is FieldAssociation.NODE:
        if association is FieldAssociation.NODE:
            return True
        if association is FieldAssociation.ELEMENT_NODE:
            return True
        if association is FieldAssociation.NODE_REGION:
            return True
        return association is FieldAssociation.RESOLVED_NODAL
    if mode is FieldAssociation.ELEMENT:
        if association is FieldAssociation.ELEMENT:
            return True
        if association is FieldAssociation.INTEGRATION_POINT:
            return True
        return association is FieldAssociation.ELEMENT_NODE
    raise TypeError("mode must be a query association")


def _typed_association_text(association: FieldAssociation) -> str:
    labels = {
        FieldAssociation.NODE: "节点",
        FieldAssociation.ELEMENT_NODE: "节点",
        FieldAssociation.ELEMENT: "单元",
        FieldAssociation.INTEGRATION_POINT: "积分点",
        FieldAssociation.NODE_REGION: "节点",
        FieldAssociation.RESOLVED_NODAL: "节点",
    }
    return labels[association]


def _parse_typed_query_ids(
    text: str,
    valid_ids: tuple[int, ...],
) -> tuple[int, ...]:
    if type(text) is not str:
        raise TypeError("text must be a string")
    if type(valid_ids) is not tuple or any(
        type(value) is not int for value in valid_ids
    ):
        raise TypeError("valid_ids must be a tuple of integers")
    if not text.strip():
        return ()

    valid = frozenset(valid_ids)
    parsed: list[int] = []
    for token in re.split(r"[\s,，;；]+", text.strip()):
        if not token:
            continue
        match = re.fullmatch(r"(-?\d+)\s*[-~～]\s*(-?\d+)", token)
        if match is None:
            try:
                candidates = (int(token),)
            except ValueError as error:
                raise ValueError(
                    f"无法识别的有限元编号：{token}"
                ) from error
        else:
            first, last = (int(value) for value in match.groups())
            step = 1 if last >= first else -1
            candidates = range(first, last + step, step)
        for candidate in candidates:
            if candidate not in valid:
                raise ValueError(f"有限元编号不存在：{candidate}")
            if candidate not in parsed:
                parsed.append(candidate)
    return tuple(parsed)


def _typed_field_label(availability: FieldAvailability) -> str:
    return _typed_result_display_field_label(availability)


_TYPED_RESULT_FIELD_LABELS = {
    "result.field.u.node": "位移 U",
    "result.field.ur.node": "转角 UR",
    "result.field.rf.node": "反力 RF",
    "result.field.rm.node": "反力矩 RM",
    "result.field.le.centroid": "对数应变 LE（单元质心）",
    "result.field.s.element_nodal": "应力 S（节点）",
}
_TYPED_RESULT_FIELD_STATE_LABELS = {
    FieldState.READY: "就绪",
    FieldState.LAZY: "按需加载",
    FieldState.UNAVAILABLE: "不可用",
}


def _typed_result_display_field_label(
    availability: FieldAvailability,
) -> str:
    descriptor = availability.descriptor
    label = _TYPED_RESULT_FIELD_LABELS.get(
        descriptor.label_key,
        descriptor.label_key,
    )
    return (
        f"{label}"
        f"（{_TYPED_RESULT_FIELD_STATE_LABELS[availability.state]}）"
    )


def _typed_result_display_availability_text(
    availability: FieldAvailability,
) -> str:
    if availability.state is FieldState.READY:
        return "已就绪"
    if availability.state is FieldState.LAZY:
        return "待物化；应用后由外层命令加载"
    if availability.diagnostics:
        return "\n".join(
            diagnostic.message for diagnostic in availability.diagnostics
        )
    return "不可用"


def _validate_typed_display_selection(
    catalog: ResultCatalog,
    selection: ScalarFieldSelection,
) -> None:
    if type(selection) is not ScalarFieldSelection:
        raise TypeError("current_selection must be ScalarFieldSelection")
    matches = tuple(
        availability
        for availability in catalog.fields
        if (
            availability.key == selection.field_key
            and result_field_is_visible(availability)
        )
    )
    if len(matches) != 1:
        raise ValueError(
            "current_selection must reference exactly one catalog field"
        )
    if selection.component not in matches[0].descriptor.columns:
        raise ValueError(
            "current_selection component is outside the field descriptor"
        )


def _validate_typed_display_options(
    *,
    shape_mode: str,
    contour_enabled: bool,
    selection: ScalarFieldSelection,
    scale_mode: str,
    scale_value: float,
    overlay_undeformed: bool,
    show_edges: bool,
) -> None:
    if type(shape_mode) is not str:
        raise TypeError("shape_mode must be a string")
    if shape_mode not in {"undeformed", "deformed"}:
        raise ValueError("shape_mode must be undeformed or deformed")
    if type(contour_enabled) is not bool:
        raise TypeError("contour_enabled must be a boolean")
    if type(selection) is not ScalarFieldSelection:
        raise TypeError("selection must be ScalarFieldSelection")
    if type(scale_mode) is not str:
        raise TypeError("scale_mode must be a string")
    if scale_mode not in {"auto", "real", "custom"}:
        raise ValueError("scale_mode must be auto, real, or custom")
    if type(scale_value) is not float:
        raise TypeError("scale_value must be a float")
    if not isfinite(scale_value) or scale_value < 0.0:
        raise ValueError("scale_value must be finite and non-negative")
    if type(overlay_undeformed) is not bool:
        raise TypeError("overlay_undeformed must be a boolean")
    if type(show_edges) is not bool:
        raise TypeError("show_edges must be a boolean")


def _typed_availability_text(availability: FieldAvailability) -> str:
    if availability.state is FieldState.READY:
        return "已就绪"
    if availability.state is FieldState.LAZY:
        return "待物化；选择会交给外层命令"
    if availability.diagnostics:
        return availability.diagnostics[0].message
    return "不可用"


def _optional_identity_text(value: int | None) -> str:
    return "" if value is None else str(value)


def _averaged_text(value: bool | None) -> str:
    if value is None:
        return "缺失"
    return "是" if value else "否"


def _number_text(value: float) -> str:
    return f"{value:.8g}"


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
