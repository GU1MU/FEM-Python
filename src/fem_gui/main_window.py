"""第一版中文有限元主窗口。"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime
import logging
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

from PySide6.QtCore import QSettings, Qt, QThread, Signal, Slot
from PySide6.QtGui import QAction, QCloseEvent
from PySide6.QtWidgets import (
    QComboBox, QDoubleSpinBox, QFileDialog, QGridLayout,
    QLabel, QMainWindow, QMessageBox, QSizePolicy, QSplitter, QVBoxLayout, QWidget,
)

from fem.abaqus import build_model as build_abaqus_model, parse_file
from fem.post.vtk.export import from_result as export_vtk_result
from fem.boundary.step import boundary_for_step
from fem.solvers import static_linear

from .actions import build_actions
from .analysis_dialogs import JobManagerDialog, JobSubmitDialog
from .analysis_jobs import AnalysisJob, JobStatus
from .dialogs import show_information
from .document import FEMDocument
from .inspection_dialogs import EntityInfoDialog
from .inspection_service import InspectionService
from .mesh_browser import MeshBrowserDialog
from .postprocessing_dialogs import (
    ContourSettingsDialog,
    ResultDisplayDialog,
    ResultDisplaySettings,
    ResultQueryDialog,
)
from .symbol_dialog import SymbolSettingsDialog
from .viewport_background import (
    ViewportBackgroundSettings,
    load_background_settings,
    save_background_settings,
)
from .viewport_background_dialog import ViewportBackgroundDialog
from .visualization.model_adapter import ModelGeometry, build_model_geometry
from .visualization.result_adapter import (
    ResultData, automatic_deformation_scale, build_result_data,
    ensure_stress_data,
)
from .visualization.selection import SelectionState
from .visualization.scene import DisplayState
from .visualization.symbols import SymbolSettings
from .widgets.navigation_panel import NavigationPanel
from .widgets.ribbon import RibbonPage, RibbonWidget
from .widgets.status_bar import CAEStatusBar
from .widgets.viewport import FEMViewport
from .widgets.viewport_toolbar import ViewportPanel
from .workers import TaskWorker


def initial_display_policy(element_count: int, node_count: int) -> dict[str, bool]:
    """Return the explicit first-display degradation policy for large models."""
    return {
        "show_edges": int(element_count) <= 100_000,
        "show_symbols": int(element_count) <= 200_000,
        "show_nodes": False,
        "show_labels": False,
        "simplified": int(element_count) > 100_000 or int(node_count) > 200_000,
    }


class FEMMainWindow(QMainWindow):
    """只暴露当前内核已经实现的有限元工作流。"""

    importStageChanged = Signal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("有限元分析")
        self.resize(1280, 800)
        self.document = FEMDocument()
        self.geometry: ModelGeometry | None = None
        self.result_data: ResultData | None = None
        self.inspection_service: InspectionService | None = None
        self._inspection_windows: list[QWidget] = []
        self._mesh_browser: MeshBrowserDialog | None = None
        self.selection = SelectionState()
        self.actions: dict[str, QAction] = {}
        self._thread: QThread | None = None
        self._worker: TaskWorker | None = None
        self._task_success_callback: Callable[[object], None] | None = None
        self._task_failure_callback: Callable[[str], None] | None = None
        self._task_error_title = "操作失败"
        self._job_manager: JobManagerDialog | None = None
        self._display = DisplayState()
        self._model_edges_visible = True
        self._scale_mode = "auto"
        self._scale_value = 1.0
        self._overlay_undeformed = False
        self._symbol_settings = SymbolSettings()
        self._application_settings = QSettings("fem-project", "fem-gui")
        self._background_settings, self._remember_background = load_background_settings(
            self._application_settings
        )
        self._contour_options: dict[str, Any] = {
            "manual": False, "minimum": 0.0, "maximum": 1.0,
            "levels": 12, "colormap": "jet", "style": "continuous", "legend": True,
            "number_format": "general", "decimals": 5,
            "orientation": "horizontal", "show_minimum": True,
            "show_maximum": True, "show_ids": False,
            "edges": False,
            "averaging_threshold": 75.0,
        }
        self._step_combos: list[QComboBox] = []
        self._build_actions()
        self._build_menus()
        self._build_ribbon()
        self._build_central_area()
        self._build_status_bar()
        self.importStageChanged.connect(self.status_panel.set_state)
        self._refresh_result_controls()
        self._update_action_states()

    @property
    def busy(self) -> bool:
        return self._thread is not None

    def _build_actions(self) -> None:
        self.actions = build_actions(self)

    def _build_menus(self) -> None:
        file_menu = self.menuBar().addMenu("文件")
        file_menu.setObjectName("menuFile")
        file_menu.addActions([self.actions[name] for name in ("open", "reload", "close")])
        file_menu.addSeparator()
        file_menu.addAction(self.actions["exit"])
        edit_menu = self.menuBar().addMenu("编辑")
        edit_menu.setObjectName("menuEdit")
        edit_menu.addActions([self.actions[name] for name in ("select_node", "select_element", "clear_selection")])
        view_menu = self.menuBar().addMenu("视图")
        view_menu.setObjectName("menuView")
        view_menu.addActions([self.actions[name] for name in (
            "fit", "top", "bottom", "front", "back", "left", "right", "iso",
        )])
        view_menu.addSeparator()
        view_menu.addActions([self.actions[name] for name in ("orthographic", "perspective")])
        view_menu.addSeparator()
        view_menu.addActions([self.actions[name] for name in ("edges", "nodes", "node_labels", "element_labels", "symbols")])
        view_menu.addSeparator()
        view_menu.addActions([
            self.actions["symbol_settings"], self.actions["viewport_background"],
        ])
        analysis_menu = self.menuBar().addMenu("分析")
        analysis_menu.setObjectName("menuAnalysis")
        analysis_menu.addAction(self.actions["step_info"])
        analysis_menu.addSeparator()
        analysis_menu.addAction(self.actions["check_model"])
        analysis_menu.addSeparator()
        analysis_menu.addActions([
            self.actions["submit_job"], self.actions["resubmit_job"],
            self.actions["job_manager"],
        ])
        result_menu = self.menuBar().addMenu("结果")
        result_menu.setObjectName("menuResult")
        result_menu.addActions([self.actions[name] for name in ("undeformed", "deformed", "contour")])
        result_menu.addSeparator()
        result_menu.addActions([self.actions[name] for name in ("overlay", "field", "scale", "contour_options", "query", "export", "screenshot")])
        help_menu = self.menuBar().addMenu("帮助")
        help_menu.setObjectName("menuHelp")
        help_menu.addAction(self.actions["about"])

    def _build_ribbon(self) -> None:
        self.ribbon = RibbonWidget(self)
        self._add_ribbon_page("项目", (
            ("文件", ("open", "reload", "close"), ("open",)),
            ("信息", ("model_info",), ()),
            ("分析", ("submit_job",), ("submit_job",)),
            ("输出", ("export", "screenshot"), ()),
        ), step_group="分析")
        self._add_ribbon_page("模型", (
            ("选择", ("select_node", "select_element", "clear_selection", "selected_info"), ()),
            ("显示", ("nodes", "edges", "node_labels", "element_labels"), ()),
            ("符号", ("symbols", "symbol_settings"), ()),
        ))
        self._add_ribbon_page("分析", (
            ("分析步", ("step_info",), ()),
            ("检查", ("check_model",), ()),
            ("作业", ("submit_job", "resubmit_job"), ("submit_job",)),
            ("管理", ("job_manager",), ()),
        ), step_group="分析步")
        self._build_result_ribbon_page()
        self._add_ribbon_page("视图", (
            ("视角", ("top", "bottom", "front", "back", "left", "right", "iso"),
             ("top", "bottom", "front", "back", "left", "right", "iso")),
            ("相机", ("fit", "orthographic", "perspective", "viewport_background"), ()),
            ("标注", ("nodes", "edges", "node_labels", "element_labels", "symbols"), ()),
        ))
        self.ribbon.moduleChanged.connect(self._on_module_changed)

    def _build_result_ribbon_page(self) -> None:
        page = self.ribbon.add_page("结果")
        shape_group = page.add_group("形状")
        for name in ("undeformed", "deformed"):
            shape_group.add_action(self.actions[name])
        contour_group = page.add_group("云图")
        contour_group.add_action(self.actions["contour"], large=True)

        field_group = page.add_group("主变量")
        field_host = QWidget(field_group)
        field_host.setObjectName("resultFieldControls")
        field_host.setMinimumWidth(246)
        field_host.setMaximumWidth(330)
        field_host.setSizePolicy(
            QSizePolicy.Policy.Preferred,
            QSizePolicy.Policy.Preferred,
        )
        field_layout = QGridLayout(field_host)
        field_layout.setContentsMargins(4, 0, 4, 0)
        field_layout.setHorizontalSpacing(5)
        field_layout.setVerticalSpacing(4)
        self.result_variable_combo = QComboBox(field_host)
        self.result_variable_combo.setObjectName("resultVariableCombo")
        self.result_component_combo = QComboBox(field_host)
        self.result_component_combo.setObjectName("resultComponentCombo")
        self.result_position_combo = QComboBox(field_host)
        self.result_position_combo.setObjectName("resultPositionCombo")
        for combo in (
            self.result_variable_combo,
            self.result_component_combo,
            self.result_position_combo,
        ):
            combo.setFixedHeight(24)
            combo.setEnabled(False)
        self.result_variable_combo.setMinimumWidth(76)
        self.result_component_combo.setMinimumWidth(86)
        self.result_position_combo.setMinimumWidth(199)
        for combo in (
            self.result_variable_combo,
            self.result_component_combo,
            self.result_position_combo,
        ):
            combo.setSizePolicy(
                QSizePolicy.Policy.Expanding,
                QSizePolicy.Policy.Fixed,
            )
        variable_label = QLabel("变量", field_host)
        component_label = QLabel("分量", field_host)
        position_label = QLabel("位置", field_host)
        for label in (variable_label, component_label, position_label):
            label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            label.setFixedSize(36, 24)
        field_layout.addWidget(variable_label, 0, 0)
        field_layout.addWidget(self.result_variable_combo, 0, 1)
        field_layout.addWidget(component_label, 0, 2)
        field_layout.addWidget(self.result_component_combo, 0, 3)
        field_layout.addWidget(position_label, 1, 0)
        field_layout.addWidget(self.result_position_combo, 1, 1, 1, 3)
        field_layout.setColumnStretch(1, 1)
        field_layout.setColumnStretch(3, 1)
        field_group.add_widget(field_host)
        self.result_variable_combo.activated.connect(self._result_variable_changed)
        self.result_component_combo.activated.connect(self._result_component_changed)
        self.result_position_combo.activated.connect(self._result_position_changed)

        deformation_group = page.add_group("变形")
        scale_host = QWidget(deformation_group)
        scale_layout = QGridLayout(scale_host)
        scale_layout.setContentsMargins(0, 0, 0, 0)
        scale_layout.setHorizontalSpacing(3)
        scale_layout.setVerticalSpacing(2)
        self.result_scale_combo = QComboBox(scale_host)
        self.result_scale_combo.setObjectName("resultScaleCombo")
        self.result_scale_combo.addItem("自动比例", "auto")
        self.result_scale_combo.addItem("真实比例", "real")
        self.result_scale_combo.addItem("自定义比例", "custom")
        self.result_scale_combo.setFixedWidth(100)
        self.result_scale_value = QDoubleSpinBox(scale_host)
        self.result_scale_value.setObjectName("resultScaleValue")
        self.result_scale_value.setRange(0.0, 1.0e12)
        self.result_scale_value.setDecimals(5)
        self.result_scale_value.setValue(self._scale_value)
        self.result_scale_value.setFixedWidth(100)
        self.result_scale_value.setEnabled(False)
        scale_layout.addWidget(self.result_scale_combo, 0, 0)
        scale_layout.addWidget(self.result_scale_value, 1, 0)
        deformation_group.add_widget(scale_host)
        deformation_group.add_action(self.actions["overlay"], large=True)
        self.result_scale_combo.activated.connect(self._result_scale_mode_changed)
        self.result_scale_value.valueChanged.connect(self._result_scale_value_changed)

        display_group = page.add_group("显示设置")
        display_group.add_action(self.actions["field"])
        display_group.add_action(self.actions["contour_options"])
        output_group = page.add_group("查询与导出")
        output_group.add_action(self.actions["query"], large=True)
        for name in ("export", "screenshot"):
            output_group.add_action(self.actions[name])

    def _add_ribbon_page(
        self,
        name: str,
        groups: tuple[tuple[str, tuple[str, ...], tuple[str, ...]], ...],
        *,
        step_group: str | None = None,
    ) -> RibbonPage:
        page = self.ribbon.add_page(name)
        for title, action_names, large_names in groups:
            group = page.add_group(title)
            if title == step_group:
                group.add_widget(self._create_step_combo(name))
            for action_name in action_names:
                large = action_name in large_names
                group.add_action(
                    self.actions[action_name],
                    large=large,
                    compact=not large and len(action_names) == 1,
                )
        return page

    def _create_step_combo(self, module_name: str) -> QComboBox:
        combo = QComboBox(self)
        combo.setObjectName(f"stepCombo_{module_name}")
        combo.setToolTip("当前分析步")
        combo.setFixedWidth(145)
        combo.addItem("—", None)
        combo.activated.connect(lambda _index, source=combo: self._step_combo_changed(source))
        self._step_combos.append(combo)
        return combo

    def _build_central_area(self) -> None:
        self.navigation = NavigationPanel(self)
        self.model_tree = self.navigation.model_tree
        self.result_tree = self.navigation.result_tree
        self.viewport = FEMViewport(self)
        self.viewport.set_background_settings(self._background_settings)
        self.viewport_panel = ViewportPanel(self.viewport, self.actions, self)
        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        splitter.setObjectName("mainSplitter")
        splitter.addWidget(self.navigation)
        splitter.addWidget(self.viewport_panel)
        splitter.setSizes([260, 1020])
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setCollapsible(0, False)
        host = QWidget(self)
        host.setObjectName("centralWorkspace")
        layout = QVBoxLayout(host)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.ribbon)
        layout.addWidget(splitter, 1)
        self.setCentralWidget(host)
        self.model_tree.highlightRequested.connect(self._highlight_tree_entry)
        self.model_tree.informationRequested.connect(self._show_entry_information)
        self.result_tree.fieldActivated.connect(self._activate_result_field)
        self.viewport.entityPicked.connect(self._on_viewport_pick)

    def _build_status_bar(self) -> None:
        self.status_panel = CAEStatusBar(self)
        self.setStatusBar(self.status_panel)

    def _on_module_changed(self, module_name: str) -> None:
        if module_name == "结果":
            self.navigation.show_result()
        elif module_name in {"项目", "模型", "分析"}:
            self.navigation.show_model()

    def _step_combo_changed(self, combo: QComboBox) -> None:
        step_name = combo.currentData()
        if step_name is None or self.document.model is None:
            return
        self._set_current_step(str(step_name))

    def _sync_step_combos(self) -> None:
        names = self.document.runnable_step_names()
        for combo in self._step_combos:
            combo.blockSignals(True)
            combo.clear()
            if not names:
                combo.addItem("—", None)
            else:
                for name in names:
                    combo.addItem(name, name)
                index = combo.findData(self.document.step_name)
                combo.setCurrentIndex(index if index >= 0 else 0)
            combo.setEnabled(bool(names) and not self.busy)
            combo.blockSignals(False)

    def _set_current_step(self, name: str) -> None:
        self.document.step_name = name
        self._symbol_settings = replace(self._symbol_settings, step_name=name)
        self.viewport.set_symbol_settings(self._symbol_settings)
        self._sync_step_combos()
        self.status_panel.set_step(name)
        self.status_panel.set_state(f"已选择分析步：{name}", 4000)

    @staticmethod
    def _field_family(field_key: str) -> str:
        if field_key in {"U", "U1", "U2", "U3", "R3"}:
            return "U"
        if field_key in {"RF", "RF1", "RF2", "RF3", "RM3"}:
            return "RF"
        return "S"

    def _refresh_result_controls(self) -> None:
        data = self.result_data
        current_field = self._display.field_key
        self.result_variable_combo.blockSignals(True)
        self.result_variable_combo.clear()
        if data is None or not data.fields:
            self.result_variable_combo.addItem("—", None)
            self.result_component_combo.clear()
            self.result_component_combo.addItem("—", None)
            self.result_position_combo.clear()
            self.result_position_combo.addItem("—", None)
            self.result_variable_combo.blockSignals(False)
            return
        available = {self._field_family(key) for key in data.fields}
        for family, label in (("U", "位移 U"), ("RF", "反力 RF"), ("S", "应力 S")):
            if family in available:
                self.result_variable_combo.addItem(label, family)
        family = self._field_family(current_field) if current_field in data.fields else str(self.result_variable_combo.itemData(0))
        self.result_variable_combo.setCurrentIndex(max(0, self.result_variable_combo.findData(family)))
        self.result_variable_combo.blockSignals(False)
        self._populate_result_positions(current_field)
        self._populate_result_components(current_field)

    def _populate_result_positions(self, preferred_field: str | None = None) -> None:
        self.result_position_combo.blockSignals(True)
        self.result_position_combo.clear()
        family = str(self.result_variable_combo.currentData())
        if family == "S" and self.result_data is not None:
            prefixes = {key.split(":", 1)[0] for key in self.result_data.fields if ":" in key}
            for prefix, label in (
                ("N", "节点平均"),
                ("EN", "单元节点（不平均）"),
                ("E", "单元中心"),
            ):
                if prefix in prefixes:
                    self.result_position_combo.addItem(label, prefix)
            preferred = preferred_field.split(":", 1)[0] if preferred_field and ":" in preferred_field else "N"
        else:
            self.result_position_combo.addItem("节点", "")
            preferred = ""
        index = self.result_position_combo.findData(preferred)
        self.result_position_combo.setCurrentIndex(index if index >= 0 else 0)
        self.result_position_combo.blockSignals(False)

    def _populate_result_components(self, preferred_field: str | None = None) -> None:
        self.result_component_combo.blockSignals(True)
        self.result_component_combo.clear()
        if self.result_data is not None:
            family = str(self.result_variable_combo.currentData())
            position = str(self.result_position_combo.currentData() or "")
            records = [
                (key, field)
                for key, field in self.result_data.fields.items()
                if self._field_family(key) == family
                and (family != "S" or key.startswith(f"{position}:"))
            ]
            records.sort(key=lambda item: self._field_sort_key(item[0]))
            for key, field in records:
                label = field.label
                if family == "S":
                    label = key.split(":", 1)[1]
                    label = {
                        "MaxPrincipal": "最大主应力",
                        "MinPrincipal": "最小主应力",
                    }.get(label, label)
                self.result_component_combo.addItem(label, key)
            index = self.result_component_combo.findData(preferred_field)
            self.result_component_combo.setCurrentIndex(index if index >= 0 else 0)
        self.result_component_combo.blockSignals(False)

    @staticmethod
    def _field_sort_key(field_key: str) -> tuple[int, int]:
        component = field_key.split(":", 1)[-1]
        order = (
            "U", "U1", "U2", "U3", "R3",
            "RF", "RF1", "RF2", "RF3", "RM3",
            "S11", "S22", "S33", "S12", "S13", "S23",
            "Mises", "MaxPrincipal", "MinPrincipal", "LE11",
        )
        prefix = field_key.split(":", 1)[0] if ":" in field_key else ""
        association_order = {"N": 0, "EN": 1, "E": 2}.get(prefix, 0)
        return (order.index(component) if component in order else len(order), association_order)

    def _result_variable_changed(self, _index: int) -> None:
        self._populate_result_positions()
        self._populate_result_components()
        self._result_component_changed(self.result_component_combo.currentIndex())

    def _result_position_changed(self, _index: int) -> None:
        component = None
        current = self.result_component_combo.currentData()
        if current:
            component = str(current).split(":", 1)[-1]
        prefix = str(self.result_position_combo.currentData() or "")
        self._populate_result_components(f"{prefix}:{component}" if prefix and component else None)
        self._result_component_changed(self.result_component_combo.currentIndex())

    def _result_component_changed(self, _index: int) -> None:
        field_key = self.result_component_combo.currentData()
        if self.result_data is None or field_key not in self.result_data.fields:
            return
        self._activate_result_field(str(field_key))

    def _result_scale_mode_changed(self, _index: int) -> None:
        self._scale_mode = str(self.result_scale_combo.currentData())
        self.result_scale_value.setEnabled(self._scale_mode == "custom" and self.document.has_result)
        self._apply_scale()

    def _result_scale_value_changed(self, value: float) -> None:
        self._scale_value = float(value)
        if self._scale_mode == "custom":
            self._apply_scale()

    def _update_action_states(self) -> None:
        has_model = self.document.has_model
        has_result = self.document.has_result
        busy = self.busy
        self.actions["open"].setEnabled(not busy)
        self.actions["reload"].setEnabled(has_model and not busy)
        self.actions["close"].setEnabled(has_model and not busy)
        has_step = has_model and self.document.step_name is not None
        self.actions["step_info"].setEnabled(has_step)
        self.actions["check_model"].setEnabled(has_step and not busy)
        self.actions["submit_job"].setEnabled(has_step and self.geometry is not None and not busy)
        active = self.document.find_job(self.document.active_job_name)
        resubmittable = (
            active
            if active is not None and active.status in {JobStatus.COMPLETED, JobStatus.FAILED}
            else self.document.latest_resubmittable_job()
        )
        self.actions["resubmit_job"].setEnabled(not busy and resubmittable is not None)
        self.actions["job_manager"].setEnabled(has_model)
        for name in (
            "fit", "front", "back", "left", "right", "top", "bottom", "iso",
            "orthographic", "perspective", "edges", "nodes", "node_labels",
            "element_labels", "select_node", "select_element", "clear_selection",
            "symbols", "symbol_settings", "model_info",
        ):
            self.actions[name].setEnabled(has_model)
        self.actions["selected_info"].setEnabled(
            has_model and (self.selection.node_id is not None or self.selection.element_id is not None)
        )
        for name in (
            "undeformed", "deformed", "contour", "overlay", "field", "scale",
            "contour_options", "query", "export",
            "screenshot",
        ):
            self.actions[name].setEnabled(has_result)
        self.actions["field"].setEnabled(has_result and not busy)
        self.actions["query"].setEnabled(has_result and not busy)
        self.actions["screenshot"].setEnabled(
            has_result and self.viewport.can_capture
        )
        self.result_variable_combo.setEnabled(has_result and not busy)
        self.result_component_combo.setEnabled(has_result and not busy)
        self.result_position_combo.setEnabled(has_result and not busy)
        self.result_scale_combo.setEnabled(has_result)
        self.result_scale_value.setEnabled(has_result and self._scale_mode == "custom")
        self._sync_step_combos()

    def open_inp(self) -> None:
        path, _filter = QFileDialog.getOpenFileName(
            self, "打开 Abaqus INP", "", "Abaqus INP 文件 (*.inp);;所有文件 (*)"
        )
        if path:
            self._load_path(Path(path))

    def reload_model(self) -> None:
        if self.document.path is not None:
            self._load_path(self.document.path)

    def _load_path(self, path: Path) -> None:
        self.status_panel.set_state("正在导入模型……")

        self.status_panel.set_state("正在解析 INP……")

        def workload():
            timings: dict[str, float] = {}
            self.importStageChanged.emit("正在解析 INP……")
            started = perf_counter()
            deck = parse_file(path)
            timings["INP 解析"] = perf_counter() - started
            self.importStageChanged.emit("正在构建有限元模型……")
            started = perf_counter()
            model = build_abaqus_model(deck)
            timings["FEMModel 构建"] = perf_counter() - started
            self.importStageChanged.emit("正在生成显示网格……")
            started = perf_counter()
            geometry = build_model_geometry(model)
            timings["VTK 显示几何构建"] = perf_counter() - started
            return model, geometry, timings

        self._start_task(workload, lambda value: self._model_loaded(path, value), "模型加载失败")

    def _model_loaded(self, path: Path, value: object) -> None:
        if len(value) == 2:
            model, geometry = value
            timings = {}
        else:
            model, geometry, timings = value
        self.status_panel.set_state("正在初始化视口……")
        self._close_inspection_windows()
        self._close_job_manager()
        self.document.set_model(path, model)
        self.geometry = geometry
        self.result_data = None
        self.selection.clear()
        self._display = DisplayState()
        self._overlay_undeformed = False
        self.actions["undeformed"].setChecked(True)
        self.actions["contour"].setChecked(False)
        self.actions["overlay"].setChecked(False)
        self._symbol_settings = replace(
            self._symbol_settings,
            step_name=self.document.step_name,
        )
        started = perf_counter()
        self.inspection_service = InspectionService(model)
        timings["InspectionService 初始化"] = perf_counter() - started
        started = perf_counter()
        self.model_tree.set_model(model)
        timings["模型树更新"] = perf_counter() - started
        self.result_tree.clear_result()
        self.navigation.show_model()
        element_count = len(model.mesh.elements)
        node_count = len(model.mesh.nodes)
        policy = initial_display_policy(element_count, node_count)
        simplified = policy["simplified"]
        self._model_edges_visible = policy["show_edges"]
        self.actions["edges"].setChecked(policy["show_edges"])
        self.actions["nodes"].setChecked(policy["show_nodes"])
        self.actions["node_labels"].setChecked(policy["show_labels"])
        self.actions["element_labels"].setChecked(policy["show_labels"])
        self.actions["symbols"].setChecked(policy["show_symbols"])

        started = perf_counter()
        self.viewport.set_model(model, geometry, refresh_symbols=False, render=False)
        self.viewport.set_edges_visible(self.actions["edges"].isChecked(), render=False)
        timings["视口网格创建"] = perf_counter() - started
        self.viewport.set_symbol_settings(self._symbol_settings, refresh=False, render=False)
        self.viewport.set_symbols_visible(
            self.actions["symbols"].isChecked(), refresh=False, render=False
        )
        started = perf_counter()
        self.viewport.show_boundary_and_loads(render=False)
        timings["载荷约束符号创建"] = perf_counter() - started
        started = perf_counter()
        self.viewport.render()
        timings["首次渲染"] = perf_counter() - started
        logging.info(
            "模型导入性能 %s: %s (总计 %.3fs)",
            path,
            ", ".join(f"{name}={seconds:.3f}s" for name, seconds in timings.items()),
            sum(timings.values()),
        )
        self.setWindowTitle(f"有限元分析 — {path.name}")
        self.status_panel.set_object()
        self.status_panel.set_step(self.document.step_name)
        self.status_panel.set_result()
        self.status_panel.set_state("模型加载完成", 5000)
        if simplified:
            self.status_panel.set_state(
                "大型模型：已简化首次显示，可在视口工具栏中开启单元边和载荷符号。",
                8000,
            )
        else:
            self.status_panel.set_state("模型加载完成", 5000)
        self._refresh_result_controls()
        self._sync_step_combos()
        self._update_action_states()

    def close_model(self) -> None:
        if self.busy:
            return
        self.document.close()
        self._close_inspection_windows()
        self._close_job_manager()
        self.inspection_service = None
        self.geometry = None
        self.result_data = None
        self.selection.clear()
        self._display = DisplayState()
        self._overlay_undeformed = False
        self.actions["undeformed"].setChecked(True)
        self.actions["contour"].setChecked(False)
        self.actions["overlay"].setChecked(False)
        self.model_tree.clear_model()
        self.result_tree.clear_result()
        self.navigation.show_model()
        self.viewport.clear_model()
        self.setWindowTitle("有限元分析")
        self.status_panel.reset_document()
        self.status_panel.set_selection_mode(self.selection.mode)
        self._refresh_result_controls()
        self._update_action_states()

    def show_current_step_information(self) -> EntityInfoDialog | None:
        """复用现有只读信息窗口显示当前分析步。"""
        if self.document.model is None or self.document.step_name is None:
            return None
        for index, step in enumerate(self.document.model.steps):
            if step.name == self.document.step_name:
                return self.show_entity_information("step", index)
        return None

    def check_current_model(self, show_success: bool = True) -> bool:
        """以正式线性静力验证规则检查当前模型。"""
        model = self.document.model
        step_name = self.document.step_name
        if model is None or step_name is None:
            return False
        try:
            selected_step = static_linear.validate_problem(model, step_name)
            boundary = boundary_for_step(model, selected_step)
        except Exception as error:
            self._show_error("模型检查失败", str(error))
            return False
        if show_success:
            mesh = model.mesh
            show_information(self, "模型检查", [
                ("模型名称", model.name or "未命名模型"),
                ("当前分析步", selected_step.name if selected_step else "—"),
                ("分析类型", "线性静力"),
                ("节点数", len(mesh.nodes)),
                ("单元数", len(mesh.elements)),
                ("总自由度数", mesh.num_dofs),
                ("材料数量", len(model.materials)),
                ("截面数量", len(model.sections)),
                ("位移边界条件数量", len(boundary.prescribed_displacements)),
                ("节点载荷数量", len(boundary.nodal_forces)),
                ("表面载荷数量", len(boundary.surface_tractions)),
                ("边载荷数量", len(boundary.edge_tractions)),
                ("检查结果", "通过"),
            ])
        self.status_panel.set_state("模型检查通过", 4000)
        return True

    def create_and_submit_job(self) -> None:
        """显示创建窗口后提交一个新的会话作业。"""
        if self.document.model is None or self.geometry is None or self.busy:
            return
        dialog = JobSubmitDialog(
            self.document.next_job_name(),
            self.document.runnable_step_names(),
            self.document.step_name,
            self,
        )
        if dialog.exec():
            self._submit_job(dialog.job_name, dialog.step_name)

    def resubmit_job(self, source_name: str | None = None) -> None:
        """以当前模型状态重新提交某个已完成或失败作业。"""
        if self.busy or self.document.model is None or self.geometry is None:
            return
        source = self.document.find_job(source_name or self.document.active_job_name)
        if source is None or source.status not in {JobStatus.COMPLETED, JobStatus.FAILED}:
            source = self.document.latest_resubmittable_job()
        if source is None:
            return
        dialog = JobSubmitDialog(
            self.document.next_job_name(),
            self.document.runnable_step_names(),
            source.step_name,
            self,
        )
        if dialog.exec():
            self._submit_job(
                dialog.job_name,
                dialog.step_name,
                source_job_name=source.name,
            )

    def _submit_job(
        self,
        name: str,
        step_name: str,
        *,
        source_job_name: str | None = None,
    ) -> AnalysisJob | None:
        """验证并后台提交作业；全部状态均仅保留在内存中。"""
        if self.document.model is None or self.geometry is None or self.busy:
            return None
        clean_name = str(name).strip()
        clean_step = str(step_name).strip()
        if not clean_name:
            self._show_error("创建作业失败", "作业名称不能为空。")
            return None
        if len(clean_name) > 64:
            self._show_error("创建作业失败", "作业名称不能超过 64 个字符。")
            return None
        if self.document.find_job(clean_name) is not None:
            self._show_error("创建作业失败", f"作业名称已存在：{clean_name}")
            return None
        if clean_step not in self.document.runnable_step_names():
            self._show_error("创建作业失败", f"分析步不存在：{clean_step}")
            return None
        job = AnalysisJob(
            clean_name,
            clean_step,
            JobStatus.RUNNING,
            started_at=datetime.now(),
            source_job_name=source_job_name,
        )
        job.add_message("开始检查模型")
        validation_started = perf_counter()
        try:
            selected_step = static_linear.validate_problem(
                self.document.model, clean_step
            )
        except Exception as error:
            job.status = JobStatus.FAILED
            job.finished_at = datetime.now()
            job.error = str(error)
            job.add_message("分析失败")
            job.add_message(job.error)
            self._show_error("模型检查失败", job.error)
            return None
        job.add_message("模型检查通过")
        validation_elapsed = perf_counter() - validation_started
        if source_job_name is not None:
            job.add_message(f"基于当前模型重新提交自 {source_job_name}")
        job.add_message("开始线性静力分析")
        self.document.add_job(job)
        self.document.active_job_name = job.name
        model = self.document.model
        geometry = self.geometry
        self.status_panel.set_state(f"正在分析：{job.name}")
        self._refresh_job_manager()

        def workload() -> tuple[object, ResultData, dict[str, float]]:
            timings = {"模型验证": validation_elapsed}
            result = static_linear.solve(
                model,
                clean_step,
                name=job.name,
                _validated_step=selected_step,
                timings=timings,
            )
            started = perf_counter()
            data = build_result_data(result, geometry, include_stress=False)
            timings["位移与反力结果"] = perf_counter() - started
            return result, data, timings

        self._start_task(
            workload,
            lambda value, submitted=job: self._job_succeeded(submitted, value),
            "分析运行失败",
            lambda message, submitted=job: self._job_failed(submitted, message),
        )
        return job

    def _job_succeeded(self, job: AnalysisJob, value: object) -> None:
        if len(value) == 2:
            result, data = value
            timings = {}
        else:
            result, data, timings = value
        job.status = JobStatus.COMPLETED
        job.finished_at = datetime.now()
        job.model_result = result
        job.result_data = data
        job.timings.update(timings)
        for stage, seconds in timings.items():
            job.add_message(f"{stage}：{seconds:.3f} s")
        job.add_message("线性静力分析完成")
        self.document.active_job_name = job.name
        activation_started = perf_counter()
        self._activate_job_result(job, completion=True)
        job.timings["首次结果显示"] = perf_counter() - activation_started
        job.add_message(f"首次结果显示：{job.timings['首次结果显示']:.3f} s")
        self._refresh_job_manager()
        self.status_panel.set_state(f"分析完成：{job.name}", 5000)
        self.ribbon.set_current("结果")

    def _job_failed(self, job: AnalysisJob, message: str) -> None:
        job.status = JobStatus.FAILED
        job.finished_at = datetime.now()
        job.error = message
        job.add_message("分析失败")
        job.add_message(message)
        self._refresh_job_manager()
        self.status_panel.set_state(f"分析失败：{job.name}", 5000)
        self._show_error("分析运行失败", message)
        self.status_panel.set_state(f"分析失败：{job.name}", 5000)

    def _activate_job_result(self, job: AnalysisJob, *, completion: bool = False) -> None:
        """将一个已完成会话作业的结果接入现有后处理流程。"""
        if not job.has_result:
            return
        self.document.result = job.model_result
        self.document.active_job_name = job.name
        self.document.step_name = job.step_name
        self.result_data = job.result_data
        data = job.result_data
        field_key = "U" if "U" in data.fields else next(iter(data.fields), None)
        self._display = DisplayState("deformed", True, field_key)
        if self.inspection_service is not None:
            self.inspection_service.update_result_data(data)
        self.viewport.set_result_data(data)
        self._apply_scale()
        self.actions["deformed"].setChecked(True)
        self.actions["contour"].setChecked(True)
        self.actions["symbols"].setChecked(False)
        self.viewport.set_symbols_visible(False, render=False)
        self.actions["node_labels"].setChecked(False)
        self.actions["element_labels"].setChecked(False)
        self.viewport.set_node_labels_visible(False)
        self.viewport.set_element_labels_visible(False)
        self.viewport.hide_selection_highlight(render=False)
        self._apply_display()
        self.result_tree.set_result(f"{job.name} · {job.step_name}", data)
        self._refresh_result_controls()
        self._sync_step_combos()
        self.status_panel.set_result(self._result_status_text())
        if not completion:
            self.status_panel.set_state(f"已打开结果：{job.name}", 5000)

    def show_job_manager(self) -> JobManagerDialog | None:
        """显示唯一的会话作业管理器。"""
        if self.document.model is None:
            return None
        if self._job_manager is None:
            dialog = JobManagerDialog(self.document.jobs, self)
            dialog.resubmitRequested.connect(self.resubmit_job)
            dialog.openResultRequested.connect(self.open_job_result)
            dialog.destroyed.connect(
                lambda _object=None, target=dialog: self._forget_job_manager(target)
            )
            self._job_manager = dialog
            dialog.show()
        else:
            self._refresh_job_manager()
            self._job_manager.show()
            self._job_manager.raise_()
            self._job_manager.activateWindow()
        return self._job_manager

    def _refresh_job_manager(self) -> None:
        if self._job_manager is not None:
            self._job_manager.refresh(self.document.jobs)

    def _close_job_manager(self) -> None:
        if self._job_manager is not None:
            self._job_manager.close()
            self._job_manager = None

    def _forget_job_manager(self, dialog: JobManagerDialog) -> None:
        if self._job_manager is dialog:
            self._job_manager = None

    def open_job_result(self, name: str) -> None:
        """打开一个已完成作业的内存结果，不重新求解。"""
        job = self.document.find_job(name)
        if job is None or not job.has_result:
            return
        self._activate_job_result(job)
        self._refresh_job_manager()
        self.ribbon.set_current("结果")

    def _start_task(
        self,
        workload: Callable[[], object],
        on_success: Callable[[object], None],
        error_title: str,
        on_failure: Callable[[str], None] | None = None,
    ) -> None:
        if self.busy:
            return
        thread = QThread(self)
        worker = TaskWorker(workload)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.succeeded.connect(self._task_succeeded)
        worker.failed.connect(self._task_failed)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._task_ended)
        self._thread = thread
        self._worker = worker
        self._task_success_callback = on_success
        self._task_failure_callback = on_failure
        self._task_error_title = error_title
        self._update_action_states()
        thread.start()

    @Slot(object)
    def _task_succeeded(self, value: object) -> None:
        """在 GUI 主线程应用后台任务结果。"""
        if QThread.currentThread() is not self.thread():
            raise RuntimeError("后台任务结果必须在 GUI 主线程处理")
        if self._task_success_callback is not None:
            self._task_success_callback(value)

    @Slot(str)
    def _task_failed(self, message: str) -> None:
        """在 GUI 主线程显示后台任务错误。"""
        if QThread.currentThread() is not self.thread():
            raise RuntimeError("后台任务错误必须在 GUI 主线程处理")
        if self._task_failure_callback is not None:
            self._task_failure_callback(message)
        else:
            self._show_error(self._task_error_title, message)

    def _task_ended(self) -> None:
        self._thread = None
        self._worker = None
        self._task_success_callback = None
        self._task_failure_callback = None
        self._update_action_states()

    def _show_error(self, title: str, message: str) -> None:
        self.status_panel.set_state("操作失败", 5000)
        box = QMessageBox(QMessageBox.Icon.Critical, title, message, parent=self)
        box.setStandardButtons(QMessageBox.StandardButton.Close)
        box.button(QMessageBox.StandardButton.Close).setText("关闭")
        box.exec()

    def viewport_fit(self) -> None:
        self.viewport.fit()

    def _toggle_edges(self, checked: bool) -> None:
        if self._display.contour_enabled:
            self._contour_options["edges"] = bool(checked)
        else:
            self._model_edges_visible = bool(checked)
        self.viewport.set_edges_visible(checked)

    def _toggle_nodes(self, checked: bool) -> None:
        self.viewport.set_nodes_visible(checked)

    def _toggle_node_labels(self, checked: bool) -> None:
        self.viewport.set_node_labels_visible(checked)

    def _toggle_element_labels(self, checked: bool) -> None:
        self.viewport.set_element_labels_visible(checked)

    def _toggle_symbols(self, checked: bool) -> None:
        self.viewport.set_symbols_visible(checked)

    def _set_selection_mode(self, mode: str) -> None:
        self.selection.mode = "element" if mode == "element" else "node"
        self.viewport.set_selection_mode(self.selection.mode)
        self.status_panel.set_selection_mode(self.selection.mode)

    def clear_selection(self) -> None:
        self.selection.clear()
        self.viewport.clear_selection()
        self.status_panel.set_object()
        self.actions["selected_info"].setEnabled(False)

    def _on_viewport_pick(self, kind: str, key: int) -> None:
        if kind == "node":
            self.selection.select_node(key)
            self.viewport.highlight_node(key)
        else:
            self.selection.select_element(key)
            self.viewport.highlight_element(key)
        self.model_tree.select_entity(kind, key)
        self.status_panel.set_selection_mode(kind)
        self.status_panel.set_object(
            f"{'节点' if kind == 'node' else '单元'} {key}",
            self._entity_coordinates(kind, key),
        )
        self.actions["selected_info"].setEnabled(True)

    def _entity_coordinates(self, kind: str, key: int) -> str:
        if self.geometry is None:
            return "—"
        if kind == "node":
            index = self.geometry.node_id_to_point_index.get(int(key))
            point = None if index is None else self.geometry.points[index]
        else:
            index = self.geometry.element_id_to_cell_index.get(int(key))
            point = None if index is None else self.geometry.points[list(self.geometry.cells[index])].mean(axis=0)
        if point is None:
            return "—"
        return ", ".join(f"{axis}={float(value):.6g}" for axis, value in zip("xyz", point))

    def _highlight_tree_entry(self, kind: str, key: object) -> None:
        if self.inspection_service is None:
            return
        self.highlight_entity(kind, key)
        if kind not in {"node", "element"}:
            if kind in {"model", "mesh"}:
                self.status_panel.set_object("模型" if kind == "model" else "网格")
                return
            names = {
                "node_set": "节点集", "element_set": "单元集", "surface": "表面",
                "edge": "边集合", "material": "材料", "section": "截面", "step": "分析步",
            }
            self.status_panel.set_object(f"{names.get(kind, '对象')} {key}")

    def set_shape_mode(self, shape_mode: str) -> None:
        if self.result_data is None:
            return
        shape = "deformed" if shape_mode == "deformed" else "undeformed"
        self._display = replace(self._display, shape_mode=shape)
        self.actions[shape].setChecked(True)
        self._apply_scale()
        self._apply_display()

    def _toggle_contour(self, checked: bool) -> None:
        if self.result_data is None:
            return
        self._display = replace(self._display, contour_enabled=bool(checked))
        self._apply_display()

    def _toggle_undeformed_overlay(self, checked: bool) -> None:
        self._overlay_undeformed = bool(checked)
        self.viewport.set_undeformed_overlay_visible(checked)

    def show_result_mode(self, mode: str) -> None:
        """兼容既有调用，并转换为新的独立显示状态。"""
        if mode == "contour":
            self.actions["contour"].setChecked(True)
            self._toggle_contour(True)
        elif mode in {"undeformed", "deformed"}:
            self.actions["contour"].setChecked(False)
            self._display = replace(self._display, contour_enabled=False)
            self.set_shape_mode(mode)

    def _apply_display(self) -> None:
        show_edges = (
            bool(self._contour_options["edges"])
            if self._display.contour_enabled
            else self._model_edges_visible
        )
        self.actions["edges"].setChecked(show_edges)
        self.viewport.set_edges_visible(show_edges, render=False)
        self.viewport.set_display(
            self._display.shape_mode,
            self._display.contour_enabled,
            self._display.field_key,
        )
        self.status_panel.set_result(self._result_status_text())

    def _activate_result_field(self, field_key: str) -> None:
        if self.result_data is None or field_key not in self.result_data.fields:
            return
        if not self.result_data.field_ready(field_key):
            prefix = field_key.split(":", 1)[0]
            self._ensure_result_stress(
                (prefix,),
                lambda key=field_key: self._activate_result_field(key),
            )
            return
        self._display = replace(self._display, field_key=field_key, contour_enabled=True)
        self.actions["contour"].setChecked(True)
        self._refresh_result_controls()
        self._apply_display()

    def _ensure_result_stress(
        self,
        prefixes: tuple[str, ...],
        on_ready: Callable[[], None],
    ) -> bool:
        """Recover requested stress fields once in the existing background worker."""
        data = self.result_data
        if data is None:
            return False
        required = tuple(
            prefix
            for prefix in prefixes
            if any(
                key.startswith(f"{prefix}:") and not scalar.ready
                for key, scalar in data.fields.items()
            )
        )
        if not required:
            on_ready()
            return True
        if self.busy:
            self.status_panel.set_state("当前任务完成后才能恢复应力结果", 4000)
            return False

        active_job = self.document.find_job(self.document.active_job_name)
        self.status_panel.set_state("正在恢复应力结果……")

        def workload() -> float:
            started = perf_counter()
            ensure_stress_data(data, required)
            return perf_counter() - started

        def succeeded(seconds: object) -> None:
            if self.result_data is not data:
                return
            elapsed = float(seconds)
            if active_job is not None:
                active_job.timings["应力恢复"] = (
                    active_job.timings.get("应力恢复", 0.0) + elapsed
                )
                active_job.add_message(f"应力恢复：{elapsed:.3f} s")
            if self.inspection_service is not None:
                self.inspection_service.update_result_data(data)
            self.viewport.set_result_data(data)
            self._refresh_result_controls()
            self._refresh_job_manager()
            self.status_panel.set_state("应力结果恢复完成", 4000)
            on_ready()

        self._start_task(
            workload,
            succeeded,
            "应力结果恢复失败",
        )
        return False

    def _result_status_text(self) -> str:
        if self.result_data is None:
            return "—"
        shape = "变形" if self._display.shape_mode == "deformed" else "未变形"
        if not self._display.contour_enabled:
            return f"{shape} / 无云图"
        field = self.result_data.fields.get(self._display.field_key or "")
        if field is None:
            return f"{shape} / 云图"
        prefix = (self._display.field_key or "").split(":", 1)[0]
        position = {
            "N": "节点平均",
            "EN": "单元节点",
            "E": "单元中心",
        }.get(prefix)
        component = (self._display.field_key or "").split(":", 1)[-1]
        result_name = f"S {component}（{position}）" if position else field.label
        return f"{shape} / {result_name}"

    def show_result_display_dialog(self) -> None:
        if self.result_data is None:
            return
        step_name = self.document.step_name or "分析结果"
        dialog = ResultDisplayDialog(
            self.result_data.fields,
            step_name=step_name,
            current_field=self._display.field_key,
            shape_mode=self._display.shape_mode,
            contour_enabled=self._display.contour_enabled,
            scale_mode=self._scale_mode,
            scale_value=self._scale_value,
            overlay_undeformed=self._overlay_undeformed,
            show_edges=self.actions["edges"].isChecked(),
            parent=self,
        )
        dialog.applyRequested.connect(self._apply_result_display_settings)
        dialog.exec()

    def _apply_result_display_settings(self, settings: ResultDisplaySettings) -> None:
        if (
            settings.contour_enabled
            and self.result_data is not None
            and settings.field_key in self.result_data.fields
            and not self.result_data.field_ready(settings.field_key)
        ):
            prefix = settings.field_key.split(":", 1)[0]
            self._ensure_result_stress(
                (prefix,),
                lambda value=settings: self._apply_result_display_settings(value),
            )
            return
        self._display = DisplayState(
            settings.shape_mode,
            settings.contour_enabled,
            settings.field_key,
        )
        self._scale_mode = settings.scale_mode
        self._scale_value = settings.scale_value
        self._overlay_undeformed = settings.overlay_undeformed
        self.actions[settings.shape_mode].setChecked(True)
        self.actions["contour"].setChecked(settings.contour_enabled)
        self.actions["overlay"].setChecked(settings.overlay_undeformed)
        if settings.contour_enabled:
            self._contour_options["edges"] = settings.show_edges
        else:
            self._model_edges_visible = settings.show_edges
        self._apply_scale()
        self.viewport.set_undeformed_overlay_visible(settings.overlay_undeformed)
        self.result_scale_combo.setCurrentIndex(max(0, self.result_scale_combo.findData(settings.scale_mode)))
        self.result_scale_value.setValue(settings.scale_value)
        self.result_scale_value.setEnabled(settings.scale_mode == "custom")
        self._refresh_result_controls()
        self._apply_display()

    def _apply_scale(self) -> None:
        if self.geometry is None or self.result_data is None:
            return
        scale = automatic_deformation_scale(self.geometry, self.result_data) if self._scale_mode == "auto" else 1.0 if self._scale_mode == "real" else self._scale_value
        self.viewport.set_deformation_scale(scale)

    def show_contour_dialog(self) -> None:
        dialog = ContourSettingsDialog(dict(self._contour_options), self)
        dialog.applyRequested.connect(self._set_contour_options)
        dialog.exec()

    def _set_contour_options(self, options: dict[str, Any]) -> None:
        self._contour_options.update(options)
        if self._display.contour_enabled:
            show_edges = bool(self._contour_options["edges"])
            self.actions["edges"].setChecked(show_edges)
            self.viewport.set_edges_visible(show_edges, render=False)
        self.viewport.set_contour_options(self._contour_options)

    def show_symbol_settings_dialog(self) -> None:
        if self.document.model is None:
            return
        dialog = SymbolSettingsDialog(
            self._symbol_settings,
            self.document.runnable_step_names(),
            self,
        )
        dialog.applyRequested.connect(self._apply_symbol_settings)
        dialog.exec()

    def show_viewport_background_dialog(self) -> None:
        """打开可实时预览的视口背景设置。"""
        dialog = ViewportBackgroundDialog(
            self._background_settings,
            self._remember_background,
            self,
        )
        dialog.previewRequested.connect(self.viewport.set_background_settings)
        dialog.applyRequested.connect(self._apply_background_settings)
        dialog.exec()

    def _apply_background_settings(
        self,
        settings: ViewportBackgroundSettings,
        remember: bool,
    ) -> None:
        self._background_settings = settings.normalized()
        self._remember_background = bool(remember)
        self.viewport.set_background_settings(self._background_settings)
        save_background_settings(
            self._application_settings,
            self._background_settings,
            self._remember_background,
        )

    def _apply_symbol_settings(self, settings: SymbolSettings) -> None:
        self._symbol_settings = settings
        if settings.step_name is not None:
            self.document.step_name = settings.step_name
        self.viewport.set_symbol_settings(settings)
        self._sync_step_combos()
        self.status_panel.set_step(self.document.step_name)

    def query_result(self) -> None:
        if self.result_data is None or self.geometry is None:
            return
        prefixes = self.result_data.available_stress_prefixes()
        if any(
            not scalar.ready
            for key, scalar in self.result_data.fields.items()
            if key.split(":", 1)[0] in prefixes
        ):
            self._ensure_result_stress(prefixes, self.query_result)
            return
        selected_kind = None
        selected_id = None
        if self.selection.node_id is not None:
            selected_kind, selected_id = "node", self.selection.node_id
        elif self.selection.element_id is not None:
            selected_kind, selected_id = "element", self.selection.element_id
        dialog = ResultQueryDialog(
            self.result_data,
            step_name=self.document.step_name or "分析结果",
            node_ids=tuple(self.geometry.node_id_to_point_index),
            element_ids=tuple(self.geometry.element_id_to_cell_index),
            selected_kind=selected_kind,
            selected_id=selected_id,
            parent=self,
        )
        dialog.locateRequested.connect(self._on_query_locate)
        dialog.exec()

    def _on_query_locate(self, kind: str, identifier: int) -> None:
        self._on_viewport_pick(kind, identifier)

    def export_vtk(self) -> None:
        if self.document.result is None:
            return
        default = (self.document.path.stem if self.document.path else "result") + ".vtk"
        path, _filter = QFileDialog.getSaveFileName(self, "导出 VTK", default, "VTK 文件 (*.vtk)")
        if not path:
            return
        target = Path(path).with_suffix(".vtk")
        try:
            export_vtk_result(self.document.result, output_dir=target.parent, name=target.stem, overwrite=True)
        except Exception as error:
            self._show_error("导出 VTK 失败", str(error))
            return
        self.status_panel.set_state("VTK 导出完成", 5000)

    def export_viewport_image(self) -> None:
        default = (self.document.path.stem if self.document.path else "viewport") + ".png"
        path, _filter = QFileDialog.getSaveFileName(
            self,
            "导出视口图片",
            default,
            "PNG 图片 (*.png);;JPEG 图片 (*.jpg *.jpeg)",
        )
        if not path:
            return
        target = Path(path)
        if target.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            target = target.with_suffix(".png")
        try:
            self.viewport.save_screenshot(str(target))
        except Exception as error:
            self._show_error("导出视口图片失败", str(error))
            return
        self.status_panel.set_state("视口图片保存完成", 5000)

    def show_model_information(self) -> None:
        self.show_entity_information("model", None)

    def show_about(self) -> None:
        show_information(self, "关于", [("软件", "有限元分析"), ("功能", "Abaqus INP 线性静力分析与结果查看"), ("界面", "PySide6、PyVistaQt、VTK")])

    def _show_entry_information(self, kind: str, key: object) -> None:
        if kind == "mesh":
            self.show_mesh_browser()
        else:
            self.show_entity_information(kind, key)

    def show_entity_information(self, kind: str, key: object) -> EntityInfoDialog | None:
        if self.inspection_service is None:
            return
        if self.result_data is not None and kind in {"node", "element"}:
            prefix = "N" if kind == "node" else "E"
            if any(
                field_key.startswith(f"{prefix}:") and not scalar.ready
                for field_key, scalar in self.result_data.fields.items()
            ):
                self._ensure_result_stress(
                    (prefix,),
                    lambda entity_kind=kind, entity_key=key: self.show_entity_information(
                        entity_kind, entity_key
                    ),
                )
                return None
        dialog = EntityInfoDialog(self.inspection_service.inspect(kind, key), self)
        dialog.highlightRequested.connect(self.highlight_entity)
        dialog.locateRequested.connect(self.locate_entity)
        dialog.entityRequested.connect(self.show_entity_information)
        self._track_inspection_window(dialog)
        dialog.show()
        return dialog

    def show_mesh_browser(self) -> MeshBrowserDialog | None:
        if self.inspection_service is None:
            return None
        if self._mesh_browser is not None:
            self._mesh_browser.show()
            self._mesh_browser.raise_()
            return self._mesh_browser
        dialog = MeshBrowserDialog(self.inspection_service, self)
        dialog.entityInformationRequested.connect(self.show_entity_information)
        dialog.highlightRequested.connect(self.highlight_entity)
        dialog.locateRequested.connect(self.locate_entity)
        dialog.destroyed.connect(lambda: setattr(self, "_mesh_browser", None))
        self._mesh_browser = dialog
        self._track_inspection_window(dialog)
        dialog.show()
        return dialog

    def show_selected_information(self) -> EntityInfoDialog | None:
        if self.selection.node_id is not None:
            return self.show_entity_information("node", self.selection.node_id)
        if self.selection.element_id is not None:
            return self.show_entity_information("element", self.selection.element_id)
        return None

    def highlight_entity(self, kind: str, key: object) -> None:
        if self.inspection_service is None:
            return
        if kind in {"surface", "edge", "surface_load", "edge_load"}:
            region_kind = "surface" if kind in {"surface", "surface_load"} else "edge"
            if kind in {"surface_load", "edge_load"}:
                step_index, load_index = key
                step = self.document.model.steps[step_index]
                key = (
                    step.surface_loads[load_index].surface
                    if region_kind == "surface"
                    else step.edge_loads[load_index].edge
                )
            members = (
                self.document.model.surfaces[str(key)].faces
                if region_kind == "surface"
                else self.document.model.edges[str(key)].edges
            )
            self.viewport.highlight_region(members, region_kind)
            return
        selection = self.inspection_service.selection_for(kind, key)
        if len(selection.node_ids) == 1 and not selection.element_ids:
            self._on_viewport_pick("node", selection.node_ids[0])
        elif len(selection.element_ids) == 1 and not selection.node_ids:
            self._on_viewport_pick("element", selection.element_ids[0])
        elif selection.node_ids:
            self.viewport.highlight_nodes(selection.node_ids)
        elif selection.element_ids:
            self.viewport.highlight_elements(selection.element_ids)

    def locate_entity(self, kind: str, key: object) -> None:
        if self.inspection_service is None:
            return
        selection = self.inspection_service.selection_for(kind, key)
        if selection.node_ids:
            self.viewport.locate_nodes(selection.node_ids)
        elif selection.element_ids:
            self.viewport.locate_elements(selection.element_ids)

    def _track_inspection_window(self, window: QWidget) -> None:
        self._inspection_windows.append(window)
        window.destroyed.connect(lambda _object=None, target=window: self._forget_inspection_window(target))

    def _forget_inspection_window(self, window: QWidget) -> None:
        if window in self._inspection_windows:
            self._inspection_windows.remove(window)

    def _close_inspection_windows(self) -> None:
        windows = tuple(self._inspection_windows)
        self._inspection_windows.clear()
        self._mesh_browser = None
        for window in windows:
            window.close()

    def closeEvent(self, event: QCloseEvent) -> None:
        if self.busy:
            self._show_error("任务正在运行", "请等待当前导入或分析任务完成后再退出。")
            event.ignore()
            return
        self._close_inspection_windows()
        self._close_job_manager()
        self.document.clear_jobs()
        event.accept()
