"""第一版中文有限元主窗口。"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import logging
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

from PySide6.QtCore import QSettings, Qt, QThread, QTimer, Slot
from PySide6.QtGui import QAction, QCloseEvent
from PySide6.QtWidgets import (
    QApplication, QComboBox, QFileDialog, QGridLayout,
    QLabel, QMainWindow, QMessageBox, QSizePolicy, QSplitter, QVBoxLayout, QWidget,
)

from fem.abaqus import build_model as build_abaqus_model, parse_file
from fem.application import (
    AuthoringStatus,
    DefinitionRejected,
    ModelCapabilityReport,
    ModelSession,
    PreflightDiagnostic,
    PreflightReport,
    RegionRef,
    describe_model_capabilities,
    describe_native_authoring_capabilities,
    safe_static_preflight,
)
from fem.application.preprocessing import generate_fem_model
from fem.core.model import (
    EdgeLoad,
    GravityLoad,
    LineLoad,
    NodalLoad,
    SurfaceLoad,
)
from fem.geometry.recipe_topology import can_preserve_logical_references
from fem.solvers import static_linear
from fem.io.project_v1 import load_project_v1, save_project_v1

from .actions import build_actions
from .analysis_dialogs import JobManagerDialog, JobSubmitDialog
from .analysis_definition_dialogs import (
    AnalysisDefinitionManagerDialog,
    DisplacementDialog,
    LoadDialog,
    StaticStepDialog,
)
from .analysis_jobs import AnalysisJob, JobStatus
from .dialogs import CompactDoubleSpinBox, show_information
from .document import FeatureRecord, NamedRegion
from .model_dialogs import (
    MaterialEditDialog,
    MaterialManagerDialog,
    RegionAssignmentDialog,
    SectionManagerDialog,
)
from .inspection_dialogs import EntityInfoDialog
from .inspection_service import InspectionService
from .mesh_browser import MeshBrowserDialog
from .mesh_quality import analyze_mesh
from .postprocessing_dialogs import (
    ContourSettingsDialog,
    ResultDisplayDialog,
    ResultDisplaySettings,
    ResultQueryDialog,
)
from .preprocessing import (
    BoxGeometry,
    BooleanGeometry,
    CylinderGeometry,
    DiskGeometry,
    ExtrudedGeometry,
    MeshSettings,
    MovedGeometry,
    NATIVE_GEOMETRY_TYPES,
    PlateWithHoleGeometry,
    RectangleGeometry,
    RotatedGeometry,
    SketchGeometry,
    build_geometry_preview,
    geometry_characteristic_size,
    geometry_dimension,
    geometry_feature_rows,
    supports_hexahedron,
)
from .preprocessing_dialogs import (
    BoxGeometryDialog,
    CylinderGeometryDialog,
    DiskGeometryDialog,
    MeshSettingsDialog,
    MeshControlsDialog,
    LocalMeshControlDialog,
    MoveGeometryDialog,
    PlateWithHoleGeometryDialog,
    RectangleGeometryDialog,
    RotateGeometryDialog,
    ExtrudeGeometryDialog,
    GeometryManagerDialog,
    BooleanGeometryDialog,
    SketchGeometryDialog,
    NamedRegionDialog,
    NamedRegionManagerDialog,
)
from .symbol_dialog import SymbolSettingsDialog
from .viewport_background import (
    ViewportBackgroundSettings,
    load_background_settings,
    save_background_settings,
)
from .viewport_background_dialog import ViewportBackgroundDialog
from .visualization.model_adapter import ModelGeometry, build_model_geometry
from .visualization.csv_export import export_field_csv
from .visualization.result_adapter import (
    ResultData, automatic_deformation_scale, build_result_data,
    field_family, recovered_stress_data,
)
from .visualization.selection import SelectionState
from .visualization.scene import DisplayState
from .visualization.symbols import SymbolSettings
from .widgets.navigation_panel import NavigationPanel
from .widgets.ribbon import RibbonPage, RibbonWidget
from .widgets.status_bar import CAEStatusBar
from .widgets.viewport import FEMViewport
from .widgets.viewport_toolbar import ViewportPanel
from .workers import TaskContext, TaskWorker


def initial_display_policy(element_count: int, node_count: int) -> dict[str, bool]:
    """Return the explicit first-display degradation policy for large models."""
    return {
        "show_edges": int(element_count) <= 100_000,
        "show_symbols": int(element_count) <= 200_000,
        "show_nodes": False,
        "show_labels": False,
        "simplified": int(element_count) > 100_000 or int(node_count) > 200_000,
    }


def native_feature_history(recipe: Any) -> list[FeatureRecord]:
    """Derive stable, shallow feature names from the existing recipe chain."""
    records: list[FeatureRecord] = []
    counters: dict[str, int] = {}

    def add(kind: str, row: str) -> None:
        counters[kind] = counters.get(kind, 0) + 1
        records.append(FeatureRecord(f"{kind}-{counters[kind]}", kind.casefold(), {"summary": row}))

    def visit(item: Any) -> None:
        if isinstance(item, SketchGeometry):
            add("Sketch", geometry_feature_rows(item)[0])
        elif isinstance(item, MovedGeometry):
            visit(item.base)
            add("Move", geometry_feature_rows(item)[-1])
        elif isinstance(item, RotatedGeometry):
            visit(item.base)
            add("Rotate", geometry_feature_rows(item)[-1])
        elif isinstance(item, ExtrudedGeometry):
            visit(item.base)
            add("Extrude", geometry_feature_rows(item)[-1])
        elif isinstance(item, BooleanGeometry):
            visit(item.object_geometry)
            add({"fuse": "Fuse", "cut": "Cut", "fragment": "Partition"}[item.operation], geometry_feature_rows(item)[-1])
        else:
            add("Base", geometry_feature_rows(item)[0])

    visit(recipe)
    return records


class FEMMainWindow(QMainWindow):
    """只暴露当前内核已经实现的有限元工作流。"""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("有限元分析")
        self.resize(1280, 800)
        self.session = ModelSession()
        self.document = self.session.snapshot()
        self._applied_session_revision = self.document.session_revision
        self._current_step_name: str | None = None
        self.geometry: ModelGeometry | None = None
        self.result_data: ResultData | None = None
        self.inspection_service: InspectionService | None = None
        self._inspection_windows: list[QWidget] = []
        self._mesh_browser: MeshBrowserDialog | None = None
        self._selected_geometry_kind: str | None = None
        self._selected_geometry_id: int | None = None
        self._selected_geometry_ids: set[int] = set()
        self._geometry_selection_mode = "body"
        self._pending_local_mesh_selection = False
        self._pending_analysis_selection: str | None = None
        self.selection = SelectionState()
        self.actions: dict[str, QAction] = {}
        self._thread: QThread | None = None
        self._worker: TaskWorker | None = None
        self._task_counter = 0
        self._active_task_id: int | None = None
        self._active_task_name = ""
        self._task_terminal_state: str | None = None
        self._task_cancel_requested = False
        self._task_callback_active = False
        self._task_thread_finished = False
        self._task_success_callback: Callable[[object], None] | None = None
        self._task_failure_callback: Callable[[str], None] | None = None
        self._task_cancel_callback: Callable[[], None] | None = None
        self._task_error_title = "操作失败"
        self._active_session_task_token: object | None = None
        self._close_after_task_cancel = False
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
            "orientation": "horizontal", "show_minimum": False,
            "show_maximum": False, "show_ids": False,
            "edges": False,
            "averaging_threshold": 75.0,
        }
        self._step_combos: list[QComboBox] = []
        self._build_actions()
        self._build_menus()
        self._build_ribbon()
        self._build_central_area()
        self._build_status_bar()
        self._refresh_result_controls()
        self._update_action_states()

    @property
    def busy(self) -> bool:
        return self._thread is not None

    def _apply_session_delta(
        self,
        delta: object,
        *,
        model_geometry: ModelGeometry | None = None,
        result_projection: ResultData | None = None,
        timings: dict[str, float] | None = None,
        source_label: str | None = None,
    ) -> bool:
        """Project one accepted Session transition into every GUI cache."""
        if not bool(getattr(delta, "accepted", True)):
            return False
        revision = int(getattr(delta, "session_revision"))
        if revision <= self._applied_session_revision:
            return False

        snapshot = self.session.snapshot()
        if snapshot.session_revision < revision:
            return False
        if snapshot.session_revision > revision:
            # A newer transition was already accepted.  Project only the newest
            # state and let the older delta become an ordered no-op.
            revision = snapshot.session_revision

        previous_artifact_id = (
            self.document.artifact.artifact_id
            if self.document.artifact is not None
            else None
        )
        self.document = snapshot

        step_names = self.session.runnable_step_names()
        if self._current_step_name not in step_names:
            self._current_step_name = self.session.default_step_name()
        self._symbol_settings = replace(
            self._symbol_settings,
            step_name=self._current_step_name,
        )

        artifact = snapshot.artifact
        if artifact is None:
            self._clear_model_projection()
            recipe = snapshot.geometry_recipe
            if isinstance(recipe, NATIVE_GEOMETRY_TYPES):
                self.model_tree.set_geometry_preview(
                    recipe.name,
                    tuple(item.name for item in snapshot.feature_history),
                    part_name=(
                        snapshot.parts[0].name
                        if snapshot.parts
                        else "Part-1"
                    ),
                )
                try:
                    self.viewport.show_geometry_preview(
                        build_geometry_preview(recipe)
                    )
                finally:
                    # The Session transition is already committed.  Keep action
                    # gates aligned with it even when the optional renderer
                    # cannot display the preview.
                    self._update_action_states()
                self.viewport_panel.set_geometry_context(True)
                self.status_panel.set_object(recipe.name)
            else:
                self.viewport_panel.set_geometry_context(False)
                self.status_panel.set_object()
        elif (
            previous_artifact_id != artifact.artifact_id
            or self.geometry is None
            or self.geometry.artifact_id != artifact.artifact_id
            or self.viewport.artifact_id != artifact.artifact_id
        ):
            geometry = model_geometry or build_model_geometry(artifact.model)
            geometry = replace(geometry, artifact_id=artifact.artifact_id)
            self._install_model(
                artifact.model,
                geometry,
                dict(timings or {}),
                source_label=source_label or self._session_source_label(),
            )

        current_result = self.session.current_result()
        current_run_id = (
            current_result.provenance.run_id
            if current_result is not None
            else None
        )
        if current_run_id is None:
            self._clear_result_projection()
        elif (
            result_projection is not None
            and artifact is not None
            and result_projection.artifact_id == artifact.artifact_id
            and result_projection.run_id == current_run_id
        ):
            self._install_result_projection(result_projection)
        elif (
            self.result_data is None
            or self.result_data.run_id != current_run_id
            or self.result_data.artifact_id != artifact.artifact_id
        ):
            core_result = getattr(
                current_result,
                "result",
                getattr(current_result, "model_result", None),
            )
            if core_result is not None and self.geometry is not None:
                self._install_result_projection(
                    replace(
                        build_result_data(
                            core_result,
                            self.geometry,
                            include_stress=False,
                        ),
                        artifact_id=artifact.artifact_id,
                        run_id=current_run_id,
                    )
                )
        elif (
            self.viewport.run_id != current_run_id
            or (
                self.inspection_service is not None
                and self.inspection_service.result_data is not self.result_data
            )
        ):
            self._install_result_projection(self.result_data)

        self._sync_step_combos()
        self._refresh_result_controls()
        self._update_action_states()
        self._applied_session_revision = revision
        return True

    def _session_source_label(self) -> str:
        path = self.document.source_path or self.document.project_path
        if path is not None:
            return Path(path).name
        recipe = self.document.geometry_recipe
        if recipe is not None:
            return str(getattr(recipe, "name", "") or "Model-1")
        model = self.document.model
        return str(getattr(model, "name", "") or "模型")

    def _clear_model_projection(self) -> None:
        self._close_inspection_windows()
        self._close_job_manager()
        self.inspection_service = None
        self.geometry = None
        self.result_data = None
        self.selection.clear()
        self._display = DisplayState()
        self.model_tree.clear_model()
        self.result_tree.clear_result()
        self.navigation.show_model()
        self.viewport.clear_model()
        self.status_panel.set_step(self._current_step_name)
        self.status_panel.set_result()

    def _clear_result_projection(self) -> None:
        if self.inspection_service is not None:
            self.inspection_service.update_result_data(None)
        if self.result_data is None and self.viewport.run_id is None:
            self.result_tree.clear_result()
            return
        self.result_data = None
        self._display = DisplayState()
        self.result_tree.clear_result()
        self.navigation.show_model()
        if self.document.artifact is not None and self.geometry is not None:
            self.viewport.set_model(
                self.document.artifact.model,
                self.geometry,
                refresh_symbols=False,
                render=False,
            )
            self.viewport.set_symbol_settings(
                self._symbol_settings,
                refresh=False,
                render=False,
            )
            self.viewport.show_boundary_and_loads(render=False)
            self.viewport.render()
        self.status_panel.set_result()

    def _install_result_projection(self, data: ResultData) -> None:
        if (
            self.document.artifact is None
            or data.artifact_id != self.document.artifact.artifact_id
            or self.geometry is None
            or self.geometry.artifact_id != data.artifact_id
            or self.viewport.artifact_id != data.artifact_id
        ):
            return
        self.result_data = data
        if self.inspection_service is not None:
            self.inspection_service.update_result_data(data)
        self.viewport.set_result_data(data)
        step_name = self._current_step_name or self.session.default_step_name()
        self.result_tree.set_result(step_name or "", data)
        self.status_panel.set_result("分析结果")

    def _build_actions(self) -> None:
        self.actions = build_actions(self)

    def _build_menus(self) -> None:
        file_menu = self.menuBar().addMenu("文件")
        file_menu.setObjectName("menuFile")
        file_menu.addActions([self.actions[name] for name in ("new_native", "open_project", "save_project", "open", "reload", "close")])
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
            ("文件", ("new_native", "open_project", "save_project", "open", "reload", "close"), ("new_native", "open")),
            ("信息", ("model_info",), ()),
            ("分析", ("submit_job",), ("submit_job",)),
            ("输出", ("export", "screenshot"), ()),
        ), step_group="分析")
        self._add_ribbon_page("几何", (
            ("创建", ("geometry_sketch",), ("geometry_sketch",)),
            (
                "特征",
                ("geometry_extrude", "geometry_move", "geometry_rotate"),
                ("geometry_extrude",),
            ),
            (
                "布尔",
                ("geometry_fuse", "geometry_cut"),
                (),
            ),
            (
                "选择",
                (
                    "geometry_select_point", "geometry_select_edge",
                    "geometry_select_face", "geometry_select_body",
                    "geometry_region", "geometry_regions",
                ),
                (),
            ),
            (
                "编辑",
                ("geometry_manager", "geometry_undo", "geometry_delete"),
                (),
            ),
        ))
        self._add_ribbon_page("网格", (
            (
                "设置",
                ("mesh_settings", "mesh_local_control", "mesh_controls"),
                ("mesh_settings",),
            ),
            ("划分", ("mesh_generate", "mesh_clear"), ("mesh_generate",)),
            (
                "检查",
                ("mesh_verify", "mesh_statistics", "mesh_quality"),
                (),
            ),
        ))
        self._add_ribbon_page("模型", (
            ("定义", ("material_manager", "section_manager", "section_assign"), ("material_manager",)),
            ("选择", ("select_node", "select_element", "clear_selection", "selected_info"), ()),
            ("显示", ("nodes", "edges", "node_labels", "element_labels"), ()),
            ("符号", ("symbols", "symbol_settings"), ()),
        ))
        self._add_ribbon_page("分析", (
            ("分析步", ("step_create", "step_info"), ("step_create",)),
            ("边界与载荷", ("boundary_create", "load_create", "output_create"), ()),
            ("检查", ("check_model",), ()),
            ("作业", ("submit_job", "resubmit_job"), ("submit_job",)),
            ("管理", ("analysis_manager", "job_manager"), ()),
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
        self.result_scale_value = CompactDoubleSpinBox(scale_host)
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
        self.model_tree.editRequested.connect(self._edit_tree_entry)
        self.result_tree.fieldActivated.connect(self._activate_result_field)
        self.viewport.entityPicked.connect(self._on_viewport_pick)
        self.viewport.selectionMissed.connect(self._on_viewport_pick_missed)
        self.viewport.selectionConfirmed.connect(self._confirm_guided_selection)
        self.viewport.selectionCancelled.connect(self._cancel_guided_selection)

    def _build_status_bar(self) -> None:
        self.status_panel = CAEStatusBar(self)
        self.status_panel.cancelRequested.connect(self.cancel_current_task)
        self.setStatusBar(self.status_panel)

    def _on_module_changed(self, module_name: str) -> None:
        geometry_context = (
            module_name in {"几何", "网格"}
            and isinstance(self.document.geometry_recipe, NATIVE_GEOMETRY_TYPES)
        )
        self.viewport_panel.set_geometry_context(geometry_context)
        if geometry_context:
            action = self.actions[f"geometry_select_{self._geometry_selection_mode}"]
            action.setChecked(True)
            self._set_geometry_selection_mode(self._geometry_selection_mode)
        elif self.document.has_model:
            action = self.actions[
                "select_element" if self.selection.mode == "element" else "select_node"
            ]
            action.setChecked(True)
            self._set_selection_mode(self.selection.mode)
        if module_name == "结果":
            self.navigation.show_result()
        elif module_name in {"项目", "几何", "网格", "模型", "分析"}:
            self.navigation.show_model()

    def _step_combo_changed(self, combo: QComboBox) -> None:
        step_name = combo.currentData()
        if step_name is None:
            return
        self._set_current_step(str(step_name))

    def _sync_step_combos(self) -> None:
        names = self.session.runnable_step_names()
        for combo in self._step_combos:
            combo.blockSignals(True)
            combo.clear()
            if not names:
                combo.addItem("—", None)
            else:
                for name in names:
                    combo.addItem(name, name)
                index = combo.findData(self._current_step_name)
                combo.setCurrentIndex(index if index >= 0 else 0)
            combo.setEnabled(bool(names) and not self.busy)
            combo.blockSignals(False)

    def _set_current_step(self, name: str) -> None:
        if name not in self.session.runnable_step_names():
            return
        self._current_step_name = name
        self._symbol_settings = replace(self._symbol_settings, step_name=name)
        self.viewport.set_symbol_settings(self._symbol_settings)
        self._sync_step_combos()
        self.status_panel.set_step(name)
        self.status_panel.set_state(f"已选择分析步：{name}", 4000)
        self._update_action_states()

    @staticmethod
    def _field_family(field_key: str) -> str:
        return field_family(field_key)

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
        for family, label in (
            ("U", "位移 U"),
            ("R", "转角 R"),
            ("RF", "反力 RF"),
            ("RM", "反力矩 RM"),
            ("S", "应力 S"),
        ):
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
            prefixes = self.result_data.available_stress_prefixes()
            for prefix in prefixes:
                self.result_position_combo.addItem(
                    self.result_data.stress_position_label(prefix),
                    prefix,
                )
            preferred = (
                preferred_field.split(":", 1)[0]
                if preferred_field and ":" in preferred_field
                else prefixes[0] if prefixes else ""
            )
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
                        "MidPrincipal": "中间主应力",
                        "MinPrincipal": "最小主应力",
                        "LE11": "轴向应变",
                        "S11Max": "最大轴向应力",
                        "S11Min": "最小轴向应力",
                        "S11AbsMax": "最大绝对值轴向应力",
                    }.get(label, label)
                self.result_component_combo.addItem(label, key)
            index = self.result_component_combo.findData(preferred_field)
            if index < 0 and family == "S":
                index = self.result_component_combo.findData(
                    f"{position}:S11AbsMax"
                )
            self.result_component_combo.setCurrentIndex(index if index >= 0 else 0)
        self.result_component_combo.blockSignals(False)

    @staticmethod
    def _field_sort_key(field_key: str) -> tuple[int, int]:
        component = field_key.split(":", 1)[-1]
        order = (
            "U", "U1", "U2", "U3", "R1", "R2", "R3",
            "RF", "RF1", "RF2", "RF3", "RM1", "RM2", "RM3",
            "S11", "S22", "S33", "S12", "S13", "S23",
            "Mises", "MaxPrincipal", "MidPrincipal", "MinPrincipal", "LE11",
            "S11Max", "S11Min", "S11AbsMax",
        )
        prefix = field_key.split(":", 1)[0] if ":" in field_key else ""
        association_order = {
            "IP": 0,
            "CENTROID": 1,
            "EN": 2,
            "NODAL": 3,
        }.get(prefix, 0)
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
        self.result_scale_value.setEnabled(
            self._scale_mode == "custom"
            and self.session.current_result() is not None
        )
        self._apply_scale()

    def _result_scale_value_changed(self, value: float) -> None:
        self._scale_value = float(value)
        if self._scale_mode == "custom":
            self._apply_scale()

    def _update_action_states(self) -> None:
        has_model = self.document.artifact is not None
        has_result = self.session.current_result() is not None
        busy = self.busy
        source_kind = self.document.source_kind
        self._set_action_available(
            "open",
            not busy,
            "后台任务运行时不能打开 INP",
        )
        self._set_action_available("new_native", not busy, "后台任务运行时不能新建项目")
        self._set_action_available("open_project", not busy, "后台任务运行时不能打开项目")
        self._set_action_available(
            "save_project",
            source_kind == "native"
            and self.document.geometry_recipe is not None
            and not busy,
            "请先创建自主草图或几何；INP 模型保持原文件工作流",
        )
        self._set_action_available(
            "reload",
            source_kind == "imported"
            and self.document.source_path is not None
            and not busy,
            "只有已打开的 INP 模型可以重新加载",
        )
        self._set_action_available(
            "close",
            source_kind is not None and not busy,
            "当前没有打开的模型或项目",
        )
        geometry_reason = (
            "请先新建模型"
            if source_kind is None
            else "INP 模型没有可编辑 CAD；请新建自主模型"
        )
        self._set_action_available(
            "geometry_sketch",
            source_kind == "native" and not busy,
            geometry_reason,
        )
        recipe = self.document.geometry_recipe
        has_native_geometry = isinstance(recipe, NATIVE_GEOMETRY_TYPES)
        for name in ("geometry_move", "geometry_rotate"):
            self._set_action_available(
                name,
                has_native_geometry and not busy,
                "请先创建自主几何",
            )
        self._set_action_available(
            "geometry_extrude",
            has_native_geometry
            and geometry_dimension(recipe) == 2
            and not busy,
            "请先创建二维草图或平面几何",
        )
        self._set_action_available(
            "geometry_manager",
            has_native_geometry and not busy,
            "请先创建自主几何",
        )
        self._set_action_available(
            "geometry_undo",
            isinstance(
                recipe,
                (MovedGeometry, RotatedGeometry, ExtrudedGeometry, BooleanGeometry),
            )
            and not busy,
            "当前没有可撤销的几何特征",
        )
        self._set_action_available(
            "geometry_delete",
            has_native_geometry and not busy,
            "请先创建自主几何",
        )
        for name in ("geometry_fuse", "geometry_cut"):
            self._set_action_available(
                name,
                has_native_geometry and not busy,
                "请先创建自主几何",
            )
        for name in (
            "geometry_select_point", "geometry_select_edge",
            "geometry_select_face", "geometry_select_body",
        ):
            self._set_action_available(
                name,
                has_native_geometry and not busy,
                "请先创建自主几何",
            )
        self._set_action_available(
            "geometry_region",
            has_native_geometry
            and self._selected_geometry_kind in {"point", "edge", "face", "body"}
            and bool(self._selected_geometry_ids)
            and not busy,
            "请先在视口中选择点、边、面或体",
        )
        self._set_action_available(
            "geometry_regions",
            has_native_geometry
            and bool(self.document.named_regions)
            and not busy,
            "当前没有命名区域",
        )
        self._set_action_available(
            "mesh_settings",
            has_native_geometry and not busy,
            "请先创建自主草图；INP 模型保留已有网格，不能反向编辑 CAD",
        )
        self._set_action_available(
            "mesh_generate",
            has_native_geometry
            and isinstance(self.document.mesh_settings, MeshSettings)
            and not busy,
            "请先创建自主几何并设置网格参数",
        )
        for name in ("mesh_controls", "mesh_local_control"):
            self._set_action_available(
                name,
                has_native_geometry
                and isinstance(self.document.mesh_settings, MeshSettings)
                and not busy,
                "请先创建自主几何并设置网格参数",
            )
        self._set_action_available(
            "mesh_clear",
            source_kind == "native"
            and has_model
            and not busy,
            "当前没有可清除的自主网格",
        )
        for name in ("mesh_statistics", "mesh_quality", "mesh_verify"):
            self._set_action_available(
                name,
                has_model and not busy,
                "请先生成网格或打开 INP 模型",
            )
        self._set_action_available(
            "material_manager",
            source_kind is not None and not busy,
            "请先新建模型或打开 INP",
        )
        self._set_action_available(
            "section_manager",
            source_kind is not None
            and bool(self.document.material_definitions)
            and not busy,
            "请先新建模型或打开 INP，并创建材料",
        )
        self._set_action_available(
            "section_assign",
            bool(self.document.section_definitions)
            and (has_model or has_native_geometry)
            and not busy,
            "请先创建几何或打开 INP，并创建截面",
        )
        has_step = bool(self.session.runnable_step_names())
        self._set_action_available(
            "step_create",
            source_kind is not None and not busy,
            "请先新建模型或打开 INP",
        )
        native_analysis_regions = bool(self._native_analysis_region_names())
        self._set_action_available(
            "boundary_create",
            has_step
            and (
                bool(self.document.model.node_sets if has_model else ())
                or native_analysis_regions
                or has_native_geometry
            )
            and not busy,
            "请先创建分析步，并准备可选择的几何或节点区域",
        )
        supported_load_regions = self._supported_load_regions()
        has_load_region = any(supported_load_regions)
        capability_report = self._model_capability_report()
        blocking_capability = next(
            (
                diagnostic
                for diagnostic in (
                    capability_report.diagnostics
                    if capability_report is not None
                    else ()
                )
                if diagnostic.blocking
            ),
            None,
        )
        if not has_step:
            load_reason = "请先创建分析步"
        elif blocking_capability is not None:
            load_reason = (
                f"[{blocking_capability.code}] "
                f"{blocking_capability.remediation or blocking_capability.message}"
            )
        else:
            load_reason = "当前 capability report 没有可用的载荷目标区域"
        self._set_action_available(
            "load_create",
            has_step
            and (has_load_region or has_native_geometry)
            and not busy,
            load_reason,
        )
        self._set_action_available(
            "output_create",
            False,
            "当前求解链不会执行输出请求；既有请求仅可查看或删除",
        )
        self._set_action_available(
            "analysis_manager",
            bool(self.document.analysis_definitions) and not busy,
            "当前没有可管理的分析定义",
        )
        self._set_action_available(
            "step_info",
            has_step and not busy,
            "当前没有可查看的分析步",
        )
        can_check_current = (
            self._current_step_name is not None
            and self.session.can_check(self._current_step_name)
        )
        self._set_action_available(
            "check_model",
            can_check_current and not busy,
            "请先生成网格或打开包含分析步的 INP 模型",
        )
        self._set_action_available(
            "submit_job",
            has_step
            and self.geometry is not None
            and self._current_step_name is not None
            and self.session.can_submit(self._current_step_name)
            and not busy,
            "请先通过当前分析步的模型检查",
        )
        resubmittable = next(
            (
                run
                for run in reversed(self.document.runs)
                if str(getattr(run.status, "value", run.status)).lower()
                in {"succeeded", "failed", "cancelled"}
            ),
            None,
        )
        self._set_action_available(
            "resubmit_job",
            not busy and resubmittable is not None,
            "当前没有已完成或失败的作业可重新提交",
        )
        self._set_action_available(
            "job_manager",
            has_model,
            "请先生成网格或打开 INP 模型",
        )
        self._set_action_available(
            "model_info",
            source_kind is not None and not busy,
            "当前没有打开的模型或项目",
        )
        for name in (
            "edges", "nodes", "node_labels", "element_labels",
            "select_node", "select_element", "symbols", "symbol_settings",
        ):
            self._set_action_available(
                name,
                has_model,
                "请先生成网格或打开 INP 模型",
            )
        for name in (
            "fit", "front", "back", "left", "right", "top", "bottom", "iso",
            "orthographic", "perspective", "clear_selection",
        ):
            self._set_action_available(
                name,
                (has_model or has_native_geometry) and not busy,
                "请先创建几何、生成网格或打开 INP 模型",
            )
        self._set_action_available(
            "selected_info",
            has_model
            and (
                self.selection.node_id is not None
                or self.selection.element_id is not None
            ),
            "请先选择节点或单元",
        )
        for name in (
            "undeformed", "deformed", "contour", "overlay", "field", "scale",
            "contour_options", "query", "export",
        ):
            self._set_action_available(
                name,
                has_result and (not busy or name not in {"field", "query"}),
                "当前没有可查看的分析结果",
            )
        self._set_action_available(
            "screenshot",
            has_result and self.viewport.can_capture,
            "当前没有可截图的分析结果，或视口不支持截图",
        )
        for action in self.actions.values():
            if not action.isEnabled() and action.toolTip() == action.text():
                action.setToolTip(f"{action.text()}（当前状态下不可用）")
                action.setStatusTip("当前状态下不可用")
        self.result_variable_combo.setEnabled(has_result and not busy)
        self.result_component_combo.setEnabled(has_result and not busy)
        self.result_position_combo.setEnabled(has_result and not busy)
        self.result_scale_combo.setEnabled(has_result)
        self.result_scale_value.setEnabled(has_result and self._scale_mode == "custom")
        self._sync_step_combos()
        self._update_window_title()

    def _update_window_title(self) -> None:
        """Keep source and save state visible without duplicating workflow state."""
        source_kind = self.document.source_kind
        if source_kind is None:
            self.setWindowTitle("有限元分析")
            return
        if source_kind == "imported":
            name = (
                self.document.path.name
                if self.document.path is not None
                else str(getattr(self.document.model, "name", "") or "模型")
            )
            source_label = "INP"
        else:
            recipe_name = str(
                getattr(self.document.geometry_recipe, "name", "") or ""
            )
            name = (
                self.document.native_project_path.name
                if self.document.native_project_path is not None
                else recipe_name or "Model-1"
            )
            source_label = "自主"
        dirty_marker = " *" if self.document.dirty else ""
        self.setWindowTitle(
            f"有限元分析 — {name} [{source_label}]{dirty_marker}"
        )

    def _set_action_available(self, name: str, available: bool, reason: str) -> None:
        """Keep disabled workflow commands explainable instead of silently grey."""
        action = self.actions[name]
        action.setEnabled(bool(available))
        if available:
            action.setToolTip(action.text())
            action.setStatusTip(action.text())
            return
        message = str(reason).strip() or "当前状态不可用"
        action.setToolTip(f"{action.text()}（{message}）")
        action.setStatusTip(message)

    def create_sketch_geometry(self) -> None:
        if self.document.source_kind == "imported":
            self._show_error(
                "几何编辑不可用",
                "当前 INP 只包含有限元模型和网格，不能反向转换为可编辑 CAD；请新建自主模型。",
            )
            return
        current = self.document.geometry_recipe
        dialog = SketchGeometryDialog(
            current if isinstance(current, SketchGeometry) else None,
            self,
        )
        if dialog.exec():
            self._set_native_geometry(dialog.recipe(), "草图")

    def create_rectangle_geometry(self) -> None:
        current = self.document.geometry_recipe
        dialog = RectangleGeometryDialog(
            current if isinstance(current, RectangleGeometry) else None,
            self,
        )
        if not dialog.exec():
            return
        self._set_native_geometry(dialog.recipe(), "矩形")

    def create_disk_geometry(self) -> None:
        current = self.document.geometry_recipe
        dialog = DiskGeometryDialog(
            current if isinstance(current, DiskGeometry) else None,
            self,
        )
        if dialog.exec():
            self._set_native_geometry(dialog.recipe(), "圆盘")

    def create_box_geometry(self) -> None:
        current = self.document.geometry_recipe
        dialog = BoxGeometryDialog(
            current if isinstance(current, BoxGeometry) else None,
            self,
        )
        if dialog.exec():
            self._set_native_geometry(dialog.recipe(), "长方体")

    def create_cylinder_geometry(self) -> None:
        current = self.document.geometry_recipe
        dialog = CylinderGeometryDialog(
            current if isinstance(current, CylinderGeometry) else None,
            self,
        )
        if dialog.exec():
            self._set_native_geometry(dialog.recipe(), "圆柱")

    def create_plate_with_hole_geometry(self) -> None:
        current = self.document.geometry_recipe
        dialog = PlateWithHoleGeometryDialog(
            current if isinstance(current, PlateWithHoleGeometry) else None,
            self,
        )
        if not dialog.exec():
            return
        self._set_native_geometry(dialog.recipe(), "带圆孔矩形板")

    def move_geometry(self) -> None:
        current = self.document.geometry_recipe
        if not isinstance(current, NATIVE_GEOMETRY_TYPES):
            return
        dialog = MoveGeometryDialog(
            current,
            self,
            is_3d=geometry_dimension(current) == 3,
        )
        if dialog.exec():
            self._set_native_geometry(dialog.recipe(), "移动后的")

    def rotate_geometry(self) -> None:
        current = self.document.geometry_recipe
        if not isinstance(current, NATIVE_GEOMETRY_TYPES):
            return
        dialog = RotateGeometryDialog(
            current,
            self,
            is_3d=geometry_dimension(current) == 3,
        )
        if dialog.exec():
            self._set_native_geometry(dialog.recipe(), "旋转后的")

    def extrude_geometry(self) -> None:
        current = self.document.geometry_recipe
        if (
            not isinstance(current, NATIVE_GEOMETRY_TYPES)
            or geometry_dimension(current) != 2
        ):
            return
        dialog = ExtrudeGeometryDialog(current, self)
        if dialog.exec():
            self._set_native_geometry(dialog.recipe(), "拉伸实体")

    def fuse_geometry(self) -> None:
        self._boolean_geometry("fuse", "合并后的")

    def cut_geometry(self) -> None:
        self._boolean_geometry("cut", "切除后的")

    def _boolean_geometry(self, operation: str, label: str) -> None:
        current = self.document.geometry_recipe
        if not isinstance(current, NATIVE_GEOMETRY_TYPES):
            return
        if isinstance(current, SketchGeometry):
            dialog = SketchGeometryDialog(
                current,
                self,
                new_contour_operation=(
                    "cut" if operation == "cut" else "material"
                ),
            )
            if dialog.exec():
                self._set_native_geometry(dialog.recipe(), label)
            return
        dialog = BooleanGeometryDialog(
            current,
            operation,
            self,
            is_3d=geometry_dimension(current) == 3,
        )
        if dialog.exec():
            self._set_native_geometry(dialog.recipe(), label)

    @staticmethod
    def _root_geometry(recipe: object) -> object:
        current = recipe
        while isinstance(
            current,
            (MovedGeometry, RotatedGeometry, ExtrudedGeometry),
        ):
            current = current.base
        while isinstance(current, BooleanGeometry):
            current = current.object_geometry
            while isinstance(
                current,
                (MovedGeometry, RotatedGeometry, ExtrudedGeometry),
            ):
                current = current.base
        return current

    @classmethod
    def _replace_root_geometry(
        cls,
        recipe: object,
        new_root: object,
    ) -> object:
        if isinstance(recipe, (MovedGeometry, RotatedGeometry, ExtrudedGeometry)):
            return replace(
                recipe,
                base=cls._replace_root_geometry(recipe.base, new_root),
            )
        if isinstance(recipe, BooleanGeometry):
            return replace(
                recipe,
                object_geometry=cls._replace_root_geometry(
                    recipe.object_geometry,
                    new_root,
                ),
            )
        return new_root

    def show_geometry_manager(self) -> None:
        current = self.document.geometry_recipe
        if not isinstance(current, NATIVE_GEOMETRY_TYPES):
            return
        if isinstance(current, SketchGeometry):
            dialog = SketchGeometryDialog(current, self)
            if dialog.exec():
                self._set_native_geometry(dialog.recipe(), "草图")
            return
        root = self._root_geometry(current)
        dialog = GeometryManagerDialog(
            current,
            self,
            can_edit_base=isinstance(root, SketchGeometry),
        )
        if not dialog.exec():
            return
        if dialog.operation == "edit" and isinstance(root, SketchGeometry):
            editor = SketchGeometryDialog(root, self)
            if editor.exec():
                rebuilt = self._replace_root_geometry(
                    current,
                    editor.recipe(),
                )
                self._set_native_geometry(rebuilt, "重新生成后的")
        elif dialog.operation == "delete" and isinstance(
            current,
            (MovedGeometry, RotatedGeometry, ExtrudedGeometry),
        ):
            self._set_native_geometry(current.base, "撤销后的")
        elif dialog.operation == "delete" and isinstance(current, BooleanGeometry):
            self._set_native_geometry(current.object_geometry, "撤销后的")
        elif dialog.operation == "clear":
            self._apply_session_delta(self.session.clear_geometry())
            self.status_panel.set_state("当前几何已清空", 5000)

    def undo_geometry_feature(self) -> None:
        current = self.document.geometry_recipe
        if isinstance(current, (MovedGeometry, RotatedGeometry, ExtrudedGeometry)):
            self._set_native_geometry(current.base, "撤销后的")
        elif isinstance(current, BooleanGeometry):
            self._set_native_geometry(current.object_geometry, "撤销后的")

    def delete_geometry(self) -> None:
        if not isinstance(self.document.geometry_recipe, NATIVE_GEOMETRY_TYPES):
            return
        self._apply_session_delta(self.session.clear_geometry())
        self._selected_geometry_kind = None
        self._selected_geometry_id = None
        self._selected_geometry_ids.clear()
        self.viewport_panel.set_geometry_context(False)
        self.status_panel.set_state("当前几何已删除", 5000)

    def _set_native_geometry(self, recipe: object, label: str) -> None:
        if not isinstance(recipe, NATIVE_GEOMETRY_TYPES):
            raise TypeError(f"不支持的几何定义：{type(recipe).__name__}")
        if self.document.source_kind != "native":
            self._apply_session_delta(self.session.new_native_project())
        prior_recipe = self.document.geometry_recipe
        current_settings = self.document.mesh_settings
        preserve_topology_references = can_preserve_logical_references(
            prior_recipe,
            recipe,
        )
        invalidated_regions: tuple[str, ...] = ()
        if (
            prior_recipe is not None
            and not preserve_topology_references
            and self.document.named_regions
        ):
            invalidated_regions = tuple(self.document.named_regions)
        invalidated_mesh_references = (
            prior_recipe is not None
            and not preserve_topology_references
            and isinstance(current_settings, MeshSettings)
            and (
                current_settings.local_size is not None
                or bool(current_settings.local_controls)
            )
        )
        invalidated_definitions = (
            prior_recipe is not None
            and not preserve_topology_references
            and bool(
                self.document.region_assignments
                or self.document.analysis_definitions
            )
        )
        preserved_references = preserve_topology_references and bool(
            self.document.named_regions
            or self.document.region_assignments
            or self.document.analysis_definitions
            or (
                isinstance(current_settings, MeshSettings)
                and (
                    current_settings.local_size is not None
                    or bool(current_settings.local_controls)
                )
            )
        )
        delta = self.session.replace_geometry(
            tuple(self.document.parts),
            recipe,
            feature_history=tuple(native_feature_history(recipe)),
        )
        self._apply_session_delta(delta)
        self._pending_local_mesh_selection = False
        self._pending_analysis_selection = None
        self._selected_geometry_kind = None
        self._selected_geometry_id = None
        self._selected_geometry_ids.clear()
        dimension = geometry_dimension(recipe)
        target_shape = "tetrahedron" if dimension == 3 else "triangle"
        if isinstance(current_settings, MeshSettings):
            valid_shape = (
                current_settings.cell_shape in (
                    {"tetrahedron", "hexahedron"}
                    if supports_hexahedron(recipe)
                    else {"tetrahedron"}
                )
                if dimension == 3
                else current_settings.cell_shape in {"triangle", "quadrilateral"}
            )
            settings = replace(
                current_settings,
                cell_shape=current_settings.cell_shape if valid_shape else target_shape,
                local_size=(
                    current_settings.local_size
                    if preserve_topology_references
                    else None
                ),
                local_controls=(
                    current_settings.local_controls
                    if preserve_topology_references
                    else ()
                ),
            )
        else:
            settings = MeshSettings(
                geometry_characteristic_size(recipe) / 10.0,
                cell_shape=target_shape,
            )
        self._apply_session_delta(
            self.session.replace_mesh_settings(settings)
        )
        self._geometry_selection_mode = "body"
        self.actions["geometry_select_body"].setChecked(True)
        self.viewport.set_selection_mode("geometry_body")
        self.status_panel.set_selection_mode("geometry_body")
        self.viewport_panel.set_geometry_context(True)
        message = (
            f"{label}几何已创建；网格、模型和结果已标记过期，"
            "请进入网格模块生成网格"
        )
        if invalidated_regions:
            message += (
                f"；{len(invalidated_regions)} 个旧命名区域已失效，"
                "请重新选择同名区域"
            )
        if invalidated_mesh_references:
            message += "；旧局部网格设置已失效"
        if invalidated_definitions:
            message += "；依赖旧拓扑的区域分配和分析步已失效"
        if preserved_references:
            message += "；逻辑拓扑未变化，已有拓扑引用已保留"
        self.status_panel.set_state(message, 6000)

    def create_named_geometry_region(self) -> None:
        self._create_region_from_current_geometry_selection()

    def _create_region_from_current_geometry_selection(self) -> str | None:
        kind, entity_id = self._selected_geometry_kind, self._selected_geometry_id
        if kind not in {"point", "edge", "face", "body"} or entity_id is None:
            return None
        entity_ids = tuple(sorted(self._selected_geometry_ids or {entity_id}))
        for region in self.document.named_regions.values():
            if region.entity_kind == kind and region.entity_ids == entity_ids:
                return region.name
        dialog = NamedRegionDialog(
            kind,
            entity_ids,
            self,
            suggested_name=self._next_named_region_name(kind),
        )
        if not dialog.exec():
            return None
        try:
            name = dialog.region_name()
        except ValueError as error:
            self._show_error("创建命名区域", str(error))
            return None
        if name in self.document.named_regions:
            self._show_error("创建命名区域", f"区域名称已存在：{name}")
            return None
        regions = dict(self.document.named_regions)
        regions[name] = NamedRegion(name, kind, entity_ids)
        self._apply_session_delta(
            self.session.replace_named_regions(regions)
        )
        self.status_panel.set_state(
            f"已创建几何区域 {name}；网格生成后可使用对应的网格集合进行分配和分析",
            5000,
        )
        self._update_action_states()
        return name

    def _next_named_region_name(self, kind: str) -> str:
        prefixes = {
            "point": "PointSet",
            "edge": "EdgeSet",
            "face": "Surface",
            "body": "BodySet",
        }
        prefix = prefixes.get(kind, "Region")
        existing = {
            name.casefold() for name in self.document.named_regions
        }
        index = 1
        while f"{prefix}-{index}".casefold() in existing:
            index += 1
        return f"{prefix}-{index}"

    def _native_analysis_region_names(
        self,
        kinds: set[str] | None = None,
    ) -> list[str]:
        return [
            region.name
            for region in self.document.named_regions.values()
            if kinds is None or region.entity_kind in kinds
        ]

    def _referenced_region_names(self) -> set[str]:
        names = {
            assignment.region_name
            for assignment in self.document.region_assignments
        }
        for step in self.document.analysis_definitions:
            names.update(str(item.target) for item in step.boundaries)
            names.update(str(item.target) for item in step.cloads)
            names.update(item.edge for item in step.edge_loads)
            names.update(item.surface for item in step.surface_loads)
            names.update(
                str(item.target)
                for item in step.line_loads
                if isinstance(item.target, str)
            )
            names.update(
                str(item.target)
                for item in step.gravity_loads
                if isinstance(item.target, str)
            )
        return names

    def show_named_region_manager(self) -> None:
        if not self.document.named_regions:
            return
        previous = dict(self.document.named_regions)
        dialog = NamedRegionManagerDialog(previous, self)
        if not dialog.exec():
            return
        updated = dialog.values()
        old_by_signature = {
            (region.entity_kind, region.entity_ids): name
            for name, region in previous.items()
        }
        new_by_signature = {
            (region.entity_kind, region.entity_ids): name
            for name, region in updated.items()
        }
        renames = {
            old_name: new_by_signature[signature]
            for signature, old_name in old_by_signature.items()
            if signature in new_by_signature
            and new_by_signature[signature] != old_name
        }
        deleted = set(previous) - set(renames) - set(updated)
        referenced_deleted = deleted & self._referenced_region_names()
        if referenced_deleted:
            self._show_error(
                "命名区域管理",
                "以下区域仍被截面、边界或载荷引用，不能删除："
                + "、".join(sorted(referenced_deleted)),
            )
            return
        try:
            delta = self.session.replace_named_regions(
                updated,
                renames=renames,
            )
        except ValueError as error:
            self._show_error("命名区域管理", str(error))
            return
        self._apply_session_delta(delta)
        self.status_panel.set_state(
            "命名区域已更新，请重新生成网格",
            5000,
        )

    def _analysis_region_names(
        self,
    ) -> tuple[list[str], list[str], list[str]]:
        model = self.document.model
        node_regions = list(model.node_sets) if model is not None else []
        edge_regions = list(model.edges) if model is not None else []
        face_regions = list(model.surfaces) if model is not None else []

        def extend_unique(target: list[str], values: list[str]) -> None:
            target.extend(value for value in values if value not in target)

        extend_unique(
            node_regions,
            self._native_analysis_region_names({"point", "edge", "face"}),
        )
        extend_unique(edge_regions, self._native_analysis_region_names({"edge"}))
        extend_unique(face_regions, self._native_analysis_region_names({"face"}))
        return node_regions, edge_regions, face_regions

    def _model_capability_report(
        self,
    ) -> ModelCapabilityReport | None:
        """Return the headless capability report for current authoring state."""

        model = self.document.model
        if model is not None:
            return describe_model_capabilities(model)
        recipe = self.document.geometry_recipe
        settings = self.document.mesh_settings
        if isinstance(recipe, NATIVE_GEOMETRY_TYPES):
            return describe_native_authoring_capabilities(
                recipe,
                settings,
            )
        return None

    def _supported_load_regions(
        self,
    ) -> tuple[list[str], list[str], list[str], list[str]]:
        """Return targets filtered by the application capability report."""

        node_regions, edge_regions, face_regions = self._analysis_region_names()
        report = self._model_capability_report()
        if report is None:
            return node_regions, [], [], []
        supported = set(report.load_kinds)
        line_regions = [
            item.region.name
            for item in report.regions
            if item.region.kind == "element_set"
            and item.compatible
            and item.families == ("beam",)
            and item.supports_distributed_load("line")
        ]
        return (
            node_regions if "node" in supported else [],
            edge_regions if "edge" in supported else [],
            face_regions if "surface" in supported else [],
            line_regions,
        )

    def _request_analysis_geometry_selection(self, operation: str) -> None:
        recipe = self.document.geometry_recipe
        if not isinstance(recipe, NATIVE_GEOMETRY_TYPES):
            return
        self._pending_analysis_selection = operation
        default_kind = "face" if geometry_dimension(recipe) == 3 else "edge"
        self.actions[f"geometry_select_{default_kind}"].setChecked(True)
        self._set_geometry_selection_mode(default_kind)
        label = "边界条件" if operation == "boundary" else "载荷"
        self.status_panel.set_state(
            f"请在视口中选择要施加{label}的"
            f"{'面' if default_kind == 'face' else '边'}；"
            "Ctrl 多选，Enter 完成，Esc 取消",
            0,
        )

    def edit_mesh_settings(self) -> None:
        recipe = self.document.geometry_recipe
        if not isinstance(recipe, NATIVE_GEOMETRY_TYPES):
            return
        current = self.document.mesh_settings
        if not isinstance(current, MeshSettings):
            current = MeshSettings(
                geometry_characteristic_size(recipe) / 10.0,
                cell_shape="tetrahedron" if geometry_dimension(recipe) == 3 else "triangle",
            )
        dialog = MeshSettingsDialog(
            current,
            self,
            mesh_dimension=geometry_dimension(recipe),
            allow_hexahedron=supports_hexahedron(recipe),
        )
        if not dialog.exec():
            return
        self._apply_session_delta(
            self.session.replace_mesh_settings(dialog.settings())
        )
        self.status_panel.set_state("网格设置已更新，请生成网格", 5000)

    def show_mesh_controls(self) -> None:
        settings = self.document.mesh_settings
        if not isinstance(settings, MeshSettings):
            return
        dialog = MeshControlsDialog(settings, self)
        if not dialog.exec():
            return
        updated = dialog.settings()
        if updated == settings:
            return
        self._apply_session_delta(
            self.session.replace_mesh_settings(updated)
        )
        self.status_panel.set_state(
            "局部网格控制已更新，请重新生成网格",
            5000,
        )

    def set_local_mesh_control(self) -> None:
        settings = self.document.mesh_settings
        recipe = self.document.geometry_recipe
        if not isinstance(settings, MeshSettings) or not isinstance(
            recipe,
            NATIVE_GEOMETRY_TYPES,
        ):
            return
        supported_kinds = {"point", "edge", "face"}
        if (
            self._selected_geometry_kind not in supported_kinds
            or self._selected_geometry_id is None
        ):
            self._pending_local_mesh_selection = True
            self.viewport_panel.set_geometry_context(True)
            default_kind = "face" if geometry_dimension(recipe) == 3 else "edge"
            self.actions[f"geometry_select_{default_kind}"].setChecked(True)
            self._set_geometry_selection_mode(default_kind)
            self.status_panel.set_state(
                f"请选择需要设置局部网格的{'面' if default_kind == 'face' else '边'}；"
                "Ctrl 多选，Enter 完成，Esc 取消",
                0,
            )
            return
        selected_id = self._selected_geometry_id
        dialog = LocalMeshControlDialog(
            self._selected_geometry_kind,
            selected_id,
            settings.size,
            self,
        )
        if not dialog.exec():
            return
        control = dialog.control()
        selected_ids = tuple(sorted(
            self._selected_geometry_ids
            if self._selected_geometry_kind == control.entity_kind
            else {control.entity_id}
        ))
        controls = tuple(
            item
            for item in settings.local_controls
            if not (
                item.entity_kind == control.entity_kind
                and item.entity_id in selected_ids
            )
        ) + tuple(
            replace(control, entity_id=entity_id)
            for entity_id in selected_ids
        )
        self._apply_session_delta(
            self.session.replace_mesh_settings(
                replace(settings, local_controls=controls)
            )
        )
        kind_name = {
            "point": "点",
            "edge": "边",
            "face": "面",
        }.get(control.entity_kind, "实体")
        self.status_panel.set_state(
            f"已设置所选{kind_name}的局部尺寸，请重新生成网格",
            5000,
        )

    def clear_native_mesh(self) -> None:
        recipe = self.document.geometry_recipe
        if (
            self.document.source_kind != "native"
            or not isinstance(recipe, NATIVE_GEOMETRY_TYPES)
        ):
            return
        self._apply_session_delta(
            self.session.clear_generated_model()
        )
        self._selected_geometry_kind = None
        self._selected_geometry_id = None
        self._selected_geometry_ids.clear()
        self.setWindowTitle(f"有限元分析 — {recipe.name}（几何）")
        self.status_panel.set_state("网格已清除，几何与网格控制已保留", 5000)

    def show_mesh_statistics(self) -> None:
        self._start_mesh_analysis("statistics")

    def show_mesh_quality(self) -> None:
        self._start_mesh_analysis("quality")

    def show_mesh_verification(self) -> None:
        self._start_mesh_analysis("verification")

    def _start_mesh_analysis(self, report_kind: str) -> None:
        model = self.document.model
        revision = self.document.model_revision
        if model is None or self.busy:
            return

        def workload(context: TaskContext):
            context.report("正在检查网格……")
            report = analyze_mesh(model)
            context.checkpoint()
            return report

        def succeeded(report: object) -> None:
            if (
                self.document.model is not model
                or self.document.model_revision != revision
            ):
                self.status_panel.set_state(
                    "模型已发生变化，已忽略旧的网格检查结果",
                    5000,
                )
                return
            self._show_mesh_analysis(report_kind, model, report)

        self._start_task(
            workload,
            succeeded,
            "网格检查失败",
            task_name="网格检查",
        )

    def _show_mesh_analysis(
        self,
        report_kind: str,
        model: object,
        report: object,
    ) -> None:
        if report_kind == "statistics":
            show_information(self, "网格统计", [
                ("节点数", report.node_count),
                ("单元数", report.element_count),
                (
                    "单元类型",
                    "；".join(
                        f"{name}={count}"
                        for name, count in report.element_types
                    ),
                ),
                ("节点集", len(model.node_sets)),
                ("单元集", len(model.element_sets)),
            ])
            return
        worst = "；".join(
            f"{element_id} ({score:.4f})"
            for element_id, score in report.worst_elements
        ) or "无可检查单元"
        if report_kind == "quality":
            show_information(self, "网格质量检查", [
                ("指标", "归一化形状质量（1 为理想，0 为退化）"),
                ("已检查单元", f"{report.checked_count} / {report.element_count}"),
                ("最小值", f"{report.minimum:.6f}"),
                ("平均值", f"{report.mean:.6f}"),
                ("最大值", f"{report.maximum:.6f}"),
                ("最差单元", worst),
            ])
            return
        show_information(self, "检查网格", [
            ("节点数", report.node_count),
            ("单元数", report.element_count),
            (
                "单元类型",
                "；".join(
                    f"{name}={count}" for name, count in report.element_types
                ),
            ),
            ("已检查", f"{report.checked_count} / {report.element_count}"),
            ("最小质量", f"{report.minimum:.6f}"),
            ("平均质量", f"{report.mean:.6f}"),
            ("最大质量", f"{report.maximum:.6f}"),
            ("最差单元", worst),
        ])

    def generate_native_mesh(self) -> None:
        recipe = self.document.geometry_recipe
        settings = self.document.mesh_settings
        if not isinstance(recipe, NATIVE_GEOMETRY_TYPES) or not isinstance(
            settings,
            MeshSettings,
        ):
            return
        task = self.session.prepare_mesh_generation()
        self.status_panel.set_state("正在生成网格……")

        def workload(context: TaskContext):
            timings: dict[str, float] = {}
            context.report("正在生成网格……")
            started = perf_counter()
            model = generate_fem_model(task)
            timings["Gmsh 几何与网格"] = perf_counter() - started
            context.report("正在准备显示网格……")
            started = perf_counter()
            display_geometry = build_model_geometry(model)
            timings["VTK 显示几何构建"] = perf_counter() - started
            context.checkpoint()
            return model, display_geometry, timings

        def succeeded(value: object) -> None:
            self._generated_model_loaded(
                value,
                token=task.token,
            )
            self.ribbon.set_current("模型")

        self._start_task(
            workload,
            succeeded,
            "网格生成失败",
            lambda message: self._session_task_failed(
                task.token,
                "网格生成失败",
                message,
            ),
            task_name="网格生成",
            on_cancelled=lambda: self._session_task_cancelled(task.token),
        )

    def open_inp(self) -> None:
        path, _filter = QFileDialog.getOpenFileName(
            self, "打开 Abaqus INP", "", "Abaqus INP 文件 (*.inp);;所有文件 (*)"
        )
        if path:
            self._load_path(Path(path))

    def new_native_model(self) -> None:
        if self.busy:
            return
        if not self._confirm_discard_changes():
            return
        self._apply_session_delta(
            self.session.new_native_project()
        )
        self.model_tree.set_geometry_preview("Model-1", (), part_name="Part-1")
        self.viewport_panel.set_geometry_context(True)
        self.status_panel.set_state("已新建自主模型，请进入几何模块创建草图", 5000)
        self.ribbon.set_current("几何")

    def open_native_project(self) -> None:
        if self.busy:
            return
        path, _filter = QFileDialog.getOpenFileName(
            self, "打开自主项目", "", "FEM 自主项目 (*.femproj);;所有文件 (*)"
        )
        if not path:
            return
        if not self._confirm_discard_changes():
            return
        expected_revision = self.document.session_revision
        try:
            project = load_project_v1(path)
            delta = self.session.replace_from_snapshot(
                project,
                expected_session_revision=expected_revision,
            )
            self._apply_session_delta(
                delta,
                source_label=Path(path).name,
            )
            self.status_panel.set_state("自主项目已打开，请生成网格并检查模型", 6000)
            self.ribbon.set_current("几何")
        except Exception as error:
            self._show_error("打开自主项目失败", str(error))

    def save_native_project(self) -> bool:
        if self.document.source_kind != "native" or self.document.geometry_recipe is None:
            return False
        path = self.document.project_path
        if path is None:
            filename, _filter = QFileDialog.getSaveFileName(
                self, "保存自主项目", "Model-1.femproj", "FEM 自主项目 (*.femproj)"
            )
            if not filename:
                return False
            path = Path(filename)
            if path.suffix.lower() != ".femproj":
                path = path.with_suffix(".femproj")
        save_snapshot = None
        marked_clean = False
        try:
            save_snapshot = self.session.prepare_project_save()
            target = save_project_v1(path, save_snapshot)
            marked_clean = self._apply_session_delta(
                self.session.accept_project_saved(
                    save_snapshot.token,
                    target,
                )
            )
        except Exception as error:
            if save_snapshot is not None:
                self._apply_session_delta(
                    self.session.accept_task_failed(
                        save_snapshot.token,
                        error,
                    )
                )
            self._show_error("保存自主项目失败", str(error))
            return False
        if not marked_clean:
            self.status_panel.set_state(
                "已保存发起操作时的项目快照；当前修改仍未保存",
                6000,
            )
            return False
        self.status_panel.set_state(f"自主项目已保存：{target.name}", 5000)
        return True

    def reload_model(self) -> None:
        if self.document.source_path is not None:
            self._load_path(self.document.source_path)

    def _load_path(self, path: Path) -> None:
        if not self._confirm_discard_changes():
            return
        task = self.session.prepare_import(path)
        self.status_panel.set_state("正在导入模型……")

        self.status_panel.set_state("正在解析 INP……")

        def workload(context: TaskContext):
            timings: dict[str, float] = {}
            context.report("正在解析 INP……")
            started = perf_counter()
            deck = parse_file(path)
            timings["INP 解析"] = perf_counter() - started
            context.report("正在构建有限元模型……")
            started = perf_counter()
            model = build_abaqus_model(deck)
            timings["FEMModel 构建"] = perf_counter() - started
            context.report("正在生成显示网格……")
            started = perf_counter()
            geometry = build_model_geometry(model)
            timings["VTK 显示几何构建"] = perf_counter() - started
            context.checkpoint()
            return model, geometry, timings

        self._start_task(
            workload,
            lambda value: self._model_loaded(
                path,
                value,
                token=task.token,
            ),
            "模型加载失败",
            lambda message: self._session_task_failed(
                task.token,
                "模型加载失败",
                message,
            ),
            task_name="INP 导入",
            on_cancelled=lambda: self._session_task_cancelled(task.token),
        )

    def _model_loaded(
        self,
        path: Path,
        value: object,
        *,
        token: object | None = None,
    ) -> None:
        model, geometry, timings = self._unpack_model_load(value)
        task = self.session.prepare_import(path) if token is None else None
        active_token = token or task.token
        delta = self.session.accept_imported_model(active_token, model)
        accepted = self._apply_session_delta(
            delta,
            model_geometry=geometry,
            timings=timings,
            source_label=path.name,
        )
        if not accepted:
            self.status_panel.set_state(
                "导入结果已过期，未覆盖当前会话",
                5000,
            )

    def _generated_model_loaded(
        self,
        value: object,
        *,
        token: object | None = None,
    ) -> None:
        """Install a generated model through the same GUI path as an INP model."""
        model, geometry, timings = self._unpack_model_load(value)
        task = self.session.prepare_mesh_generation() if token is None else None
        active_token = token or task.token
        delta = self.session.accept_generated_model(active_token, model)
        accepted = self._apply_session_delta(
            delta,
            model_geometry=geometry,
            timings=timings,
            source_label=str(getattr(model, "name", None) or "未命名模型"),
        )
        if not accepted:
            self.status_panel.set_state(
                "网格结果已过期，未覆盖当前会话",
                5000,
            )

    @staticmethod
    def _unpack_model_load(value: object) -> tuple[object, ModelGeometry, dict[str, float]]:
        if len(value) == 2:
            model, geometry = value
            timings = {}
        else:
            model, geometry, timings = value
        return model, geometry, dict(timings)

    def _show_model_in_tree(self, model: object) -> None:
        if self.document.source_kind == "native" and self.document.parts:
            self.model_tree.set_model(
                model,
                feature_rows=tuple(
                    record.name for record in self.document.feature_history
                ),
                part_name=self.document.parts[0].name,
            )
            return
        self.model_tree.set_model(model)

    def _install_model(
        self,
        model: object,
        geometry: ModelGeometry,
        timings: dict[str, float],
        *,
        source_label: str,
    ) -> None:
        self.status_panel.set_state("正在初始化视口……")
        self._close_inspection_windows()
        self._close_job_manager()
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
            step_name=self._current_step_name,
        )
        started = perf_counter()
        self.inspection_service = InspectionService(model)
        timings["InspectionService 初始化"] = perf_counter() - started
        started = perf_counter()
        self._show_model_in_tree(model)
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
            source_label,
            ", ".join(f"{name}={seconds:.3f}s" for name, seconds in timings.items()),
            sum(timings.values()),
        )
        self.setWindowTitle(f"有限元分析 — {source_label}")
        self.status_panel.set_object()
        self.status_panel.set_step(self._current_step_name)
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

    def _confirm_discard_changes(self) -> bool:
        if not self.document.dirty:
            return True
        box = QMessageBox(self)
        box.setWindowTitle("未保存的修改")
        box.setIcon(QMessageBox.Icon.Warning)
        box.setText("当前模型包含尚未保存的修改。")
        save_button = None
        if (
            self.document.source_kind == "native"
            and self.document.geometry_recipe is not None
        ):
            save_button = box.addButton(
                "保存",
                QMessageBox.ButtonRole.AcceptRole,
            )
        discard_button = box.addButton(
            "放弃修改",
            QMessageBox.ButtonRole.DestructiveRole,
        )
        cancel_button = box.addButton(
            "取消",
            QMessageBox.ButtonRole.RejectRole,
        )
        box.setDefaultButton(cancel_button)
        box.exec()
        clicked = box.clickedButton()
        if clicked is save_button:
            return self.save_native_project()
        return clicked is discard_button

    def close_model(self, *, confirm: bool = True) -> bool:
        if self.busy:
            return False
        if confirm and not self._confirm_discard_changes():
            return False
        self._apply_session_delta(self.session.close())
        self._selected_geometry_kind = None
        self._selected_geometry_id = None
        self._selected_geometry_ids.clear()
        self._display = DisplayState()
        self._overlay_undeformed = False
        self.actions["undeformed"].setChecked(True)
        self.actions["contour"].setChecked(False)
        self.actions["overlay"].setChecked(False)
        self.status_panel.reset_document()
        self.status_panel.set_selection_mode(self.selection.mode)
        return True

    def _apply_model_definition_changes(
        self,
        reason: str,
        *,
        materials: object | None = None,
        sections: object | None = None,
        assignments: object | None = None,
        steps: object | None = None,
    ) -> bool:
        """Atomically compile editable definitions through the Session."""
        try:
            delta = self.session.replace_model_definitions(
                tuple(
                    self.document.material_definitions
                    if materials is None
                    else materials
                ),
                tuple(
                    self.document.section_definitions
                    if sections is None
                    else sections
                ),
                tuple(
                    self.document.region_assignments
                    if assignments is None
                    else assignments
                ),
                tuple(
                    self.document.analysis_definitions
                    if steps is None
                    else steps
                ),
            )
        except DefinitionRejected as error:
            self._show_error(
                "模型定义",
                self._render_diagnostics(error.diagnostics),
            )
            return False
        self._apply_session_delta(delta)
        self.status_panel.set_state(reason, 5000)
        return True

    @staticmethod
    def _render_diagnostics(
        diagnostics: object,
    ) -> str:
        """Render structured diagnostics without reinterpreting their rules."""

        values = tuple(diagnostics)
        if any(
            not isinstance(item, PreflightDiagnostic)
            for item in values
        ):
            raise TypeError(
                "diagnostic renderer requires PreflightDiagnostic values"
            )
        return "\n".join(
            (
                f"[{item.code}] {item.message}"
                + (
                    f"\n建议：{item.remediation}"
                    if item.remediation
                    else ""
                )
            )
            for item in values
        ) or "操作未通过模型定义校验。"

    def show_material_manager(self) -> None:
        dialog = MaterialManagerDialog(self.document.material_definitions, self)
        if not dialog.exec():
            return
        values = tuple(dialog.values())
        old_materials = tuple(self.document.material_definitions)
        sections = tuple(self.document.section_definitions)
        if len(values) == len(old_materials):
            renames = {
                old.name: new.name
                for old, new in zip(old_materials, values)
                if old.name != new.name
            }
            sections = tuple(
                replace(
                    section,
                    material=renames.get(section.material, section.material),
                )
                for section in sections
            )
        available = {material.name for material in values}
        missing = sorted({
            section.material
            for section in sections
            if section.material not in available
        })
        if missing:
            self._show_error(
                "材料管理",
                f"材料仍被截面引用，不能删除：{'、'.join(missing)}",
            )
            return
        try:
            self._apply_model_definition_changes(
                "材料已修改，模型需要重新检查",
                materials=values,
                sections=sections,
            )
        except ValueError as error:
            self._show_error("材料管理", str(error))

    def edit_material(self, material_name: str) -> None:
        """Edit one tree material and compile it back into the current model."""
        row = next(
            (
                index
                for index, material in enumerate(
                    self.document.material_definitions
                )
                if material.name == material_name
            ),
            None,
        )
        if row is None:
            self._show_error("编辑材料", f"材料不存在：{material_name}")
            return
        dialog = MaterialEditDialog(
            self.document.material_definitions[row],
            self,
        )
        if not dialog.exec():
            return
        try:
            updated = dialog.material()
        except ValueError as error:
            self._show_error("编辑材料", str(error))
            return
        if any(
            index != row and material.name == updated.name
            for index, material in enumerate(
                self.document.material_definitions
            )
        ):
            self._show_error(
                "编辑材料",
                f"材料名称已存在：{updated.name}",
            )
            return

        materials = list(self.document.material_definitions)
        materials[row] = updated
        sections = tuple(self.document.section_definitions)
        if updated.name != material_name:
            sections = tuple(
                replace(section, material=updated.name)
                if section.material == material_name
                else section
                for section in sections
            )
        try:
            self._apply_model_definition_changes(
                "材料已修改，模型需要重新检查",
                materials=materials,
                sections=sections,
            )
        except ValueError as error:
            self._show_error("编辑材料", str(error))

    def show_section_manager(self) -> None:
        capability_report = self._model_capability_report()
        section_authoring = (
            capability_report.operation("section.create")
            if capability_report is not None
            else None
        )
        dialog = SectionManagerDialog(
            self.document.material_definitions,
            self.document.section_definitions,
            self,
            model_dimension=(
                capability_report.topological_dimension
                if capability_report is not None
                and capability_report.topological_dimension is not None
                else 3
            ),
            section_presets=(
                capability_report.section_presets
                if capability_report is not None
                else ()
            ),
            authoring_enabled=(
                section_authoring is not None
                and section_authoring.status
                in {AuthoringStatus.ENABLED, AuthoringStatus.LIMITED}
            ),
        )
        if not dialog.exec():
            return
        values = tuple(dialog.values())
        old_sections = tuple(self.document.section_definitions)
        assignments = tuple(self.document.region_assignments)
        if len(values) == len(old_sections):
            renames = {
                old.name: new.name
                for old, new in zip(old_sections, values)
                if old.name != new.name
            }
            assignments = tuple(
                replace(
                    assignment,
                    section_name=renames.get(
                        assignment.section_name,
                        assignment.section_name,
                    ),
                )
                for assignment in assignments
            )
        available = {section.name for section in values}
        missing = sorted({
            assignment.section_name
            for assignment in assignments
            if assignment.section_name not in available
        })
        if missing:
            self._show_error(
                "截面管理",
                f"截面仍被区域引用，不能删除：{'、'.join(missing)}",
            )
            return
        try:
            self._apply_model_definition_changes(
                "截面已修改，模型需要重新检查",
                sections=values,
                assignments=assignments,
            )
        except ValueError as error:
            self._show_error("截面管理", str(error))

    def assign_section_to_region(self) -> None:
        if not self.document.section_definitions:
            return
        capability_report = self._model_capability_report()
        regions: list[RegionRef] = []
        if capability_report is not None:
            regions.extend(
                item.region
                for item in capability_report.regions
                if item.region.kind == "element_set"
            )
        if isinstance(self.document.geometry_recipe, NATIVE_GEOMETRY_TYPES):
            domain = RegionRef("element_set", "DOMAIN")
            if domain not in regions:
                regions.insert(0, domain)
            for name in self._native_analysis_region_names({"body"}):
                reference = RegionRef("element_set", name)
                if reference not in regions:
                    regions.append(reference)
        if not regions:
            self._show_error("截面分配", "当前模型没有可分配的单元区域")
            return
        compatible_targets = {
            section.name: tuple(
                region
                for region in regions
                if (
                    capability_report is not None
                    and (
                        capability_report.region(region).supports_section(
                            section.section_type
                        )
                        if capability_report.regions
                        else capability_report.supports_section(
                            section.section_type
                        )
                    )
                )
            )
            for section in self.document.section_definitions
        }
        dialog = RegionAssignmentDialog(
            self.document.section_definitions,
            regions,
            self,
            compatible_targets=compatible_targets,
        )
        if not dialog.exec():
            return
        assignment = dialog.assignment()
        assignments = [
            current
            for current in self.document.region_assignments
            if current.region_name != assignment.region_name
        ] + [assignment]
        self._apply_model_definition_changes(
            "截面分配已修改，模型需要重新检查",
            assignments=assignments,
        )

    def _analysis_definitions_changed(
        self,
        reason: str,
        steps: object,
    ) -> None:
        self._apply_model_definition_changes(reason, steps=steps)

    def create_static_step(self) -> None:
        if self.document.source_kind is None:
            return
        definitions = list(deepcopy(self.document.analysis_definitions))
        name = f"Step-{len(definitions) + 1}"
        dialog = StaticStepDialog(name, self)
        if not dialog.exec():
            return
        try:
            step = dialog.step()
        except ValueError as error:
            self._show_error("创建分析步", str(error))
            return
        if any(
            existing.name.casefold() == step.name.casefold()
            for existing in definitions
        ):
            self._show_error("创建分析步", f"分析步名称已存在：{step.name}")
            return
        definitions.append(step)
        self._analysis_definitions_changed(
            "分析步已修改，模型需要重新检查",
            definitions,
        )

    def create_displacement_boundary(self) -> None:
        model = self.document.model
        if not self.session.runnable_step_names():
            return
        selected_region = None
        if (
            self.document.source_kind == "native"
            and self._selected_geometry_kind in {"point", "edge", "face"}
            and self._selected_geometry_id is not None
        ):
            selected_region = self._create_region_from_current_geometry_selection()
            if selected_region is None:
                return
        node_regions, _edge_regions, _face_regions = self._analysis_region_names()
        if not node_regions and isinstance(
            self.document.geometry_recipe,
            NATIVE_GEOMETRY_TYPES,
        ):
            self._request_analysis_geometry_selection("boundary")
            return
        capability_report = self._model_capability_report()
        dimensions = (
            capability_report.dofs_per_node
            if capability_report is not None
            and capability_report.dofs_per_node is not None
            else model.mesh.dofs_per_node
            if model is not None
            else geometry_dimension(self.document.geometry_recipe)
        )
        dialog = DisplacementDialog(
            list(self.session.runnable_step_names()),
            node_regions,
            dimensions,
            self,
            selected_region=selected_region,
            labels=(
                capability_report.dof_labels
                if capability_report is not None
                else ()
            ),
        )
        if not dialog.exec():
            return
        try:
            step_name, boundaries = dialog.definitions()
        except ValueError as error:
            self._show_error("位移边界条件", str(error))
            return
        definitions = list(deepcopy(self.document.analysis_definitions))
        step = next(step for step in definitions if step.name == step_name)
        step.boundaries = tuple(step.boundaries) + boundaries
        self._analysis_definitions_changed(
            "边界条件已修改，模型需要重新检查",
            definitions,
        )

    def create_load(self) -> None:
        model = self.document.model
        if not self.session.runnable_step_names():
            return
        selected_region = None
        preferred_kind = None
        capability_report = self._model_capability_report()
        if capability_report is None:
            return
        if (
            self.document.source_kind == "native"
            and self._selected_geometry_kind in {"point", "edge", "face"}
            and self._selected_geometry_id is not None
        ):
            preferred_kind = {
                "point": "node",
                "edge": "edge",
                "face": "surface",
            }[self._selected_geometry_kind]
            if preferred_kind not in capability_report.load_kinds:
                self._show_error(
                    "创建载荷",
                    "所选区域不支持当前模型的分布载荷契约。",
                )
                return
            selected_region = self._create_region_from_current_geometry_selection()
            if selected_region is None:
                return
        node_regions, edge_regions, face_regions, line_regions = (
            self._supported_load_regions()
        )
        dimensions = (
            capability_report.dofs_per_node
            if capability_report.dofs_per_node is not None
            else model.mesh.dofs_per_node
            if model is not None
            else 3
        )
        dialog = LoadDialog(
            list(self.session.runnable_step_names()),
            node_regions,
            edge_regions,
            face_regions,
            dimensions,
            self,
            spatial_dimensions=(
                capability_report.spatial_dimension or 3
            ),
            line_regions=line_regions,
            selected_region=selected_region,
            preferred_kind=preferred_kind,
            labels=capability_report.force_labels,
        )
        if not dialog.exec():
            return
        try:
            step_name, load = dialog.definition()
        except ValueError as error:
            self._show_error("创建载荷", str(error))
            return
        definitions = list(deepcopy(self.document.analysis_definitions))
        step = next(step for step in definitions if step.name == step_name)
        if isinstance(load, NodalLoad):
            step.cloads = tuple(step.cloads) + (load,)
        elif isinstance(load, EdgeLoad):
            step.edge_loads = tuple(step.edge_loads) + (load,)
        elif isinstance(load, SurfaceLoad):
            step.surface_loads = tuple(step.surface_loads) + (load,)
        elif isinstance(load, LineLoad):
            step.line_loads = tuple(step.line_loads) + (load,)
        elif isinstance(load, GravityLoad):
            step.gravity_loads = tuple(step.gravity_loads) + (load,)
        self._analysis_definitions_changed(
            "载荷已修改，模型需要重新检查",
            definitions,
        )

    def create_output_request(self) -> None:
        self._show_error(
            "输出请求",
            "当前求解链不会执行输出请求，因此不能新建；"
            "既有请求仍可在分析定义管理中查看或删除。",
        )

    def _analysis_manager_dialog(
        self,
    ) -> AnalysisDefinitionManagerDialog | None:
        if not self.document.analysis_definitions:
            return None
        node_regions, edge_regions, face_regions, line_regions = (
            self._supported_load_regions()
        )
        capability_report = self._model_capability_report()
        dimensions = (
            capability_report.dofs_per_node
            if capability_report is not None
            and capability_report.dofs_per_node is not None
            else 3
        )
        return AnalysisDefinitionManagerDialog(
            self.document.analysis_definitions,
            node_regions,
            edge_regions,
            face_regions,
            dimensions,
            self,
            spatial_dimensions=(
                capability_report.spatial_dimension
                if capability_report is not None
                and capability_report.spatial_dimension is not None
                else 3
            ),
            line_regions=line_regions,
            dof_labels=(
                capability_report.dof_labels
                if capability_report is not None
                else ()
            ),
            force_labels=(
                capability_report.force_labels
                if capability_report is not None
                else ()
            ),
        )

    def show_analysis_manager(self) -> None:
        dialog = self._analysis_manager_dialog()
        if dialog is None:
            return
        if not dialog.exec():
            return
        self._analysis_definitions_changed(
            "分析步、边界、载荷或输出请求已修改，模型需要重新检查",
            dialog.values(),
        )

    def edit_analysis_definition(self, kind: str, key: object) -> None:
        """Open the parameter dialog for one concrete model-tree definition."""
        manager_kind = {
            "step": "step",
            "boundary": "boundary",
            "cload": "node_load",
            "edge_load": "edge_load",
            "surface_load": "surface_load",
            "line_load": "line_load",
            "gravity_load": "gravity_load",
            "output": "output",
        }.get(kind)
        if manager_kind is None:
            self.show_entity_information(kind, key)
            return
        if kind == "step":
            definition_key = (manager_kind, int(key), None)
        else:
            step_index, item_index = key
            definition_key = (
                manager_kind,
                int(step_index),
                int(item_index),
            )
        dialog = self._analysis_manager_dialog()
        if dialog is None or not dialog.edit_definition(definition_key):
            return
        self._analysis_definitions_changed(
            "分析步、边界、载荷或输出请求已修改，模型需要重新检查",
            dialog.values(),
        )

    def show_current_step_information(self) -> EntityInfoDialog | None:
        """复用现有只读信息窗口显示当前分析步。"""
        if self._current_step_name is None:
            return None
        if self.document.model is None:
            step = next(
                (
                    item
                    for item in self.document.analysis_definitions
                    if item.name == self._current_step_name
                ),
                None,
            )
            if step is None:
                return None
            show_information(self, "分析步信息", [
                ("名称", step.name),
                ("过程", "线性静力" if step.procedure == "static" else step.procedure),
                ("边界条件", len(step.boundaries)),
                (
                    "载荷",
                    len(step.cloads)
                    + len(step.edge_loads)
                    + len(step.surface_loads)
                    + len(step.line_loads)
                    + len(step.gravity_loads),
                ),
                ("输出请求", len(step.outputs)),
                ("状态", "将在生成网格后编译到有限元模型"),
            ])
            return None
        for index, step in enumerate(self.document.model.steps):
            if step.name == self._current_step_name:
                return self.show_entity_information("step", index)
        return None

    def check_current_model(self, show_success: bool = True) -> bool:
        """Run the same structured static preflight used by background checks."""
        task = self._prepare_model_check()
        if task is None:
            return False
        report = self._evaluate_model_check(
            task.model,
            task.step_name,
            task.token,
        )
        return self._complete_model_check(
            task.token,
            report,
            show_success=show_success,
        )

    def start_model_check(self, _checked: bool = False) -> bool:
        """Run the user-facing model check without blocking the Qt event loop."""
        if self.busy:
            return False
        task = self._prepare_model_check()
        if task is None:
            return False

        def workload(context: TaskContext):
            context.report("正在检查模型……")
            result = self._evaluate_model_check(
                task.model,
                task.step_name,
                task.token,
            )
            context.checkpoint()
            return result

        def succeeded(value: object) -> None:
            self._complete_model_check(
                task.token,
                value,
                show_success=True,
            )

        def failed(message: str) -> None:
            self._apply_session_delta(
                self.session.accept_task_failed(
                    task.token,
                    message,
                )
            )
            self._show_error("模型检查失败", message)

        def cancelled() -> None:
            self._apply_session_delta(
                self.session.accept_task_cancelled(task.token)
            )

        return self._start_task(
            workload,
            succeeded,
            "模型检查失败",
            failed,
            task_name="模型检查",
            on_cancelled=cancelled,
        )

    def _prepare_model_check(self) -> object | None:
        step_name = self._current_step_name
        if step_name is None or not self.session.can_check(step_name):
            return None
        return self.session.prepare_validation(step_name)

    @staticmethod
    def _evaluate_model_check(
        model: object,
        step_name: str,
        token: object | None = None,
    ) -> PreflightReport:
        return safe_static_preflight(
            model,
            step_name,
            token=token,
        )

    def _complete_model_check(
        self,
        token: object,
        report: object,
        *,
        show_success: bool,
    ) -> bool:
        if not isinstance(report, PreflightReport):
            raise TypeError("model check must return PreflightReport")
        delta = self.session.accept_validation(token, report)
        if not self._apply_session_delta(delta):
            self.status_panel.set_state(
                "模型已发生变化，已忽略旧的检查结果",
                5000,
            )
            return False
        if not report.passed:
            message = self._render_diagnostics(report.errors)
            self._show_error(
                "模型检查失败",
                message or "模型检查未通过",
            )
            self.status_panel.set_state("模型检查未通过", 5000)
            return False
        if show_success:
            facts = report.facts
            warnings = "；".join(
                f"[{item.code}] {item.message}"
                for item in report.warnings
            )
            show_information(self, "模型检查", [
                ("模型名称", facts.model_name or "未命名模型"),
                ("当前分析步", facts.step_name or "—"),
                ("分析类型", facts.procedure or "线性静力"),
                ("节点数", facts.node_count),
                ("单元数", facts.element_count),
                ("总自由度数", facts.dof_count),
                ("材料数量", facts.material_count),
                ("截面数量", facts.section_count),
                ("位移边界条件数量", facts.displacement_count),
                ("节点载荷数量", facts.nodal_load_count),
                ("表面载荷数量", facts.surface_load_count),
                ("边载荷数量", facts.edge_load_count),
                ("梁线载荷数量", facts.line_load_count),
                ("重力载荷数量", facts.gravity_load_count),
                (
                    "数值稳定性",
                    (
                        "已检查"
                        if report.numerical_stability_checked
                        else "未执行"
                    ),
                ),
                ("警告/限制", warnings or "无"),
                ("检查结果", "通过"),
            ])
        self.status_panel.set_state(
            "模型检查通过（有警告）"
            if report.warnings
            else "模型检查通过",
            4000,
        )
        return True

    def create_and_submit_job(self) -> None:
        """显示创建窗口后提交一个新的会话作业。"""
        if self.document.model is None or self.geometry is None or self.busy:
            return
        dialog = JobSubmitDialog(
            self.session.next_run_name(),
            self.session.runnable_step_names(),
            self._current_step_name,
            self,
        )
        if dialog.exec():
            self._submit_job(dialog.job_name, dialog.step_name)

    def resubmit_job(self, source_name: str | None = None) -> None:
        """以当前模型状态重新提交某个已完成或失败作业。"""
        if self.busy or self.document.model is None or self.geometry is None:
            return
        source = self.session.find_run(source_name)
        if source is None or source.status not in {
            JobStatus.COMPLETED,
            JobStatus.FAILED,
            JobStatus.CANCELLED,
        }:
            source = self.session.latest_resubmittable_run()
        if source is None:
            return
        dialog = JobSubmitDialog(
            self.session.next_run_name(),
            self.session.runnable_step_names(),
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
        if self.session.find_run(clean_name) is not None:
            self._show_error("创建作业失败", f"作业名称已存在：{clean_name}")
            return None
        if clean_step not in self.session.runnable_step_names():
            self._show_error("创建作业失败", f"分析步不存在：{clean_step}")
            return None
        try:
            task = self.session.prepare_solve(clean_step, clean_name)
        except (KeyError, RuntimeError, ValueError) as error:
            self._show_error("创建作业失败", str(error))
            return None
        if task.delta is not None:
            self._apply_session_delta(task.delta)
        self._apply_session_delta(self.session.begin_run(task.token))
        job = self.session.find_run(task.run_id)
        if job is None:
            return None
        geometry = self.geometry
        self.status_panel.set_state(f"正在分析：{job.name}")
        self._refresh_job_manager()
        stage = {"name": "模型验证"}

        def workload(
            context: TaskContext,
        ) -> tuple[object, ResultData, dict[str, float]]:
            timings: dict[str, float] = {}
            solve_model = task.model
            stage["name"] = "求解"
            context.report("正在装配并求解……")
            result = static_linear.solve(
                solve_model,
                task.step_name,
                name=task.run_name,
                timings=timings,
            )
            context.report("正在准备结果……")
            started = perf_counter()
            data = build_result_data(result, geometry, include_stress=False)
            data = replace(
                data,
                artifact_id=task.token.artifact_id,
                run_id=task.run_id,
            )
            timings["位移与反力结果"] = perf_counter() - started
            context.checkpoint()
            return result, data, timings

        self._start_task(
            workload,
            lambda value, token=task.token: self._job_succeeded(token, value),
            "分析运行失败",
            lambda message, token=task.token, current_stage=stage: self._job_failed(
                token,
                message,
                validation_failure=current_stage["name"] == "模型验证",
            ),
            task_name=f"作业 {job.name}",
            on_cancelled=lambda token=task.token: self._job_cancelled(token),
        )
        self._active_session_task_token = task.token
        return job

    def _job_succeeded(self, token: object, value: object) -> None:
        if len(value) == 2:
            result, data = value
            timings = {}
        else:
            result, data, timings = value
        data = replace(
            data,
            artifact_id=token.artifact_id,
            run_id=token.run_id,
        )
        delta = self.session.accept_run_result(
            token,
            result,
            timings=timings,
        )
        if not self._apply_session_delta(
            delta,
            result_projection=data,
        ):
            self.status_panel.set_state(
                "求解结果已过期，未覆盖当前会话",
                5000,
            )
            return
        job = self.session.find_run(token.run_id)
        if job is None:
            return
        activation_started = perf_counter()
        self._activate_job_result(job, completion=True)
        timings["首次结果显示"] = perf_counter() - activation_started
        self._refresh_job_manager()
        self.status_panel.set_state(f"分析完成：{job.name}", 5000)
        self.ribbon.set_current("结果")

    def _job_failed(
        self,
        token: object,
        message: str,
        *,
        validation_failure: bool = False,
    ) -> None:
        self._apply_session_delta(
            self.session.accept_run_failed(token, message)
        )
        job = self.session.find_run(token.run_id)
        if job is None:
            return
        self._refresh_job_manager()
        state = "模型检查失败" if validation_failure else "分析失败"
        self.status_panel.set_state(f"{state}：{job.name}", 5000)
        self._show_error(
            "模型检查失败" if validation_failure else "分析运行失败",
            message,
        )

    def _job_cancelled(self, token: object) -> None:
        self._apply_session_delta(
            self.session.accept_run_cancelled(token)
        )
        job = self.session.find_run(token.run_id)
        if job is None:
            return
        self._refresh_job_manager()
        self.status_panel.set_state(f"分析已取消：{job.name}", 5000)

    def _activate_job_result(self, job: AnalysisJob, *, completion: bool = False) -> None:
        """将一个已完成会话作业的结果接入现有后处理流程。"""
        if not job.has_result:
            return
        if self.document.displayed_result_run_id != job.run_id:
            projection = self.session.prepare_result_projection(job.run_id)
            if self.geometry is None:
                return
            data = replace(
                build_result_data(
                    projection.record.result,
                    self.geometry,
                    include_stress=False,
                ),
                artifact_id=projection.token.artifact_id,
                run_id=projection.run_id,
            )
            if not self._apply_session_delta(
                self.session.accept_result_projection(
                    projection.token
                )
            ):
                return
            self._apply_session_delta(
                self.session.select_result(job.run_id),
                result_projection=data,
            )
        self._set_current_step(job.step_name)
        data = self.result_data
        if data is None or data.run_id != job.run_id:
            return
        field_key = "U" if "U" in data.fields else next(iter(data.fields), None)
        self._display = DisplayState("deformed", True, field_key)
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
        job = self.session.find_run(name)
        if job is None or not job.has_result:
            return
        self._activate_job_result(job)
        self._refresh_job_manager()
        self.ribbon.set_current("结果")

    def _session_task_failed(
        self,
        token: object,
        title: str,
        message: str,
    ) -> None:
        if self._apply_session_delta(
            self.session.accept_task_failed(token, message)
        ):
            self._show_error(title, message)

    def _session_task_cancelled(self, token: object) -> None:
        self._apply_session_delta(
            self.session.accept_task_cancelled(token)
        )

    def _start_task(
        self,
        workload: Callable[[TaskContext], object],
        on_success: Callable[[object], None],
        error_title: str,
        on_failure: Callable[[str], None] | None = None,
        *,
        task_name: str = "后台任务",
        on_cancelled: Callable[[], None] | None = None,
    ) -> bool:
        if self.busy:
            self.status_panel.set_state(
                f"当前任务正在运行：{self._active_task_name or '后台任务'}",
                4000,
            )
            return False
        self._task_counter += 1
        task_id = self._task_counter
        thread = QThread(self)
        worker = TaskWorker(task_id, workload)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(self._task_progress)
        worker.succeeded.connect(self._task_succeeded)
        worker.failed.connect(self._task_failed)
        worker.cancelled.connect(self._task_cancelled)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(self._task_ended)
        thread.finished.connect(thread.deleteLater)
        self._thread = thread
        self._worker = worker
        self._active_task_id = task_id
        self._active_task_name = str(task_name)
        self._task_terminal_state = None
        self._task_cancel_requested = False
        self._task_callback_active = False
        self._task_thread_finished = False
        self._task_success_callback = on_success
        self._task_failure_callback = on_failure
        self._task_cancel_callback = on_cancelled
        self._task_error_title = error_title
        self.status_panel.set_task_active(True)
        self._update_action_states()
        thread.start()
        return True

    @Slot(int, str)
    def _task_progress(self, task_id: int, stage: str) -> None:
        if (
            task_id == self._active_task_id
            and not self._task_cancel_requested
        ):
            self.status_panel.set_state(stage)

    @Slot(int, object)
    def _task_succeeded(self, task_id: int, value: object) -> None:
        """在 GUI 主线程应用后台任务结果。"""
        if QThread.currentThread() is not self.thread():
            raise RuntimeError("后台任务结果必须在 GUI 主线程处理")
        if task_id != self._active_task_id or self._task_terminal_state is not None:
            return
        if self._task_cancel_requested:
            self._task_cancelled(task_id)
            return
        self._task_terminal_state = "succeeded"
        self._task_callback_active = True
        try:
            if self._task_success_callback is not None:
                self._task_success_callback(value)
        except Exception as error:
            logging.exception("GUI background task result application failed")
            self._task_terminal_state = "failed"
            self._apply_task_failure(
                str(error).strip() or type(error).__name__
            )
        finally:
            self._task_callback_active = False
            self._maybe_finalize_task()

    @Slot(int, str)
    def _task_failed(self, task_id: int, message: str) -> None:
        """在 GUI 主线程显示后台任务错误。"""
        if QThread.currentThread() is not self.thread():
            raise RuntimeError("后台任务错误必须在 GUI 主线程处理")
        if task_id != self._active_task_id or self._task_terminal_state is not None:
            return
        self._task_terminal_state = "failed"
        self._task_callback_active = True
        try:
            self._apply_task_failure(message)
        finally:
            self._task_callback_active = False
            self._maybe_finalize_task()

    def _apply_task_failure(self, message: str) -> None:
        try:
            if self._task_failure_callback is not None:
                self._task_failure_callback(message)
            else:
                self._show_error(self._task_error_title, message)
        except Exception:
            logging.exception("GUI background task failure callback failed")
            self._show_error(self._task_error_title, message)

    @Slot(int)
    def _task_cancelled(self, task_id: int) -> None:
        if task_id != self._active_task_id or self._task_terminal_state is not None:
            return
        self._task_terminal_state = "cancelled"
        self._task_callback_active = True
        try:
            if self._task_cancel_callback is not None:
                self._task_cancel_callback()
        except Exception as error:
            logging.exception("GUI background task cancellation callback failed")
            self._show_error(
                self._task_error_title,
                str(error).strip() or type(error).__name__,
            )
        finally:
            self._task_callback_active = False
            self.status_panel.set_state(
                f"已取消：{self._active_task_name or '后台任务'}",
                4000,
            )
            self._maybe_finalize_task()

    def cancel_current_task(self) -> bool:
        if (
            self._worker is None
            or self._active_task_id is None
            or self._task_terminal_state is not None
            or self._task_cancel_requested
        ):
            return False
        token = self._active_session_task_token
        if token is not None and getattr(token, "run_id", None) is not None:
            try:
                self._apply_session_delta(
                    self.session.request_cancel(token.run_id)
                )
            except (KeyError, RuntimeError):
                pass
        self._task_cancel_requested = True
        self._worker.request_cancel()
        self.status_panel.set_task_active(True, cancelling=True)
        self.status_panel.set_state(
            f"正在取消：{self._active_task_name or '后台任务'}",
        )
        return True

    @Slot()
    def _task_ended(self) -> None:
        if QThread.currentThread() is not self.thread():
            raise RuntimeError("后台任务清理必须在 GUI 主线程处理")
        if self.sender() is not self._thread:
            return
        self._task_thread_finished = True
        self._maybe_finalize_task()

    def _maybe_finalize_task(self) -> None:
        if (
            not self._task_thread_finished
            or self._task_terminal_state is None
            or self._task_callback_active
        ):
            return
        self._finalize_task()

    def _finalize_task(self) -> None:
        self._thread = None
        self._worker = None
        self._active_task_id = None
        self._active_task_name = ""
        self._task_terminal_state = None
        self._task_cancel_requested = False
        self._task_callback_active = False
        self._task_thread_finished = False
        self._task_success_callback = None
        self._task_failure_callback = None
        self._task_cancel_callback = None
        self._active_session_task_token = None
        self.status_panel.set_task_active(False)
        self._update_action_states()
        if self._close_after_task_cancel:
            self._close_after_task_cancel = False
            QTimer.singleShot(0, self.close)

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
        normalized = "element" if mode == "element" else "node"
        if (
            self._selected_geometry_kind is not None
            or self.viewport._selection_mode != normalized
        ):
            self.selection.clear()
            self._selected_geometry_kind = None
            self._selected_geometry_id = None
            self._selected_geometry_ids.clear()
            self.viewport.clear_selection()
            self.status_panel.set_object()
            self.actions["selected_info"].setEnabled(False)
        self.selection.mode = normalized
        self.viewport.set_selection_mode(self.selection.mode)
        self.status_panel.set_selection_mode(self.selection.mode)

    def _set_geometry_selection_mode(self, mode: str) -> None:
        normalized = mode if mode in {"point", "edge", "face", "body"} else "body"
        has_fem_selection = (
            self.selection.node_id is not None
            or self.selection.element_id is not None
        )
        if (
            has_fem_selection
            or (
                self._selected_geometry_kind is not None
                and self._selected_geometry_kind != normalized
            )
        ):
            self._selected_geometry_kind = None
            self._selected_geometry_id = None
            self._selected_geometry_ids.clear()
            self.viewport.clear_selection()
            self.status_panel.set_object()
        self.selection.clear()
        self.actions["selected_info"].setEnabled(False)
        self._geometry_selection_mode = normalized
        self.viewport.set_selection_mode(f"geometry_{normalized}")
        self.status_panel.set_selection_mode(f"geometry_{normalized}")

    def clear_selection(self) -> None:
        self.selection.clear()
        self._pending_local_mesh_selection = False
        self._pending_analysis_selection = None
        self._selected_geometry_kind = None
        self._selected_geometry_id = None
        self._selected_geometry_ids.clear()
        self.viewport.clear_selection()
        self.status_panel.set_object()
        self.actions["selected_info"].setEnabled(False)
        self._update_action_states()

    @staticmethod
    def _geometry_pick_is_additive() -> bool:
        return bool(
            QApplication.keyboardModifiers()
            & Qt.KeyboardModifier.ControlModifier
        )

    def _on_viewport_pick(self, kind: str, key: int) -> None:
        if kind.startswith("geometry_"):
            geometry_kind = kind.removeprefix("geometry_")
            additive = self._geometry_pick_is_additive()
            if self._selected_geometry_kind != geometry_kind or not additive:
                self._selected_geometry_ids = {int(key)}
            elif int(key) in self._selected_geometry_ids:
                self._selected_geometry_ids.remove(int(key))
            else:
                self._selected_geometry_ids.add(int(key))
            self._selected_geometry_kind = geometry_kind
            self._selected_geometry_id = (
                int(key)
                if int(key) in self._selected_geometry_ids
                else min(self._selected_geometry_ids, default=None)
            )
            self.viewport.highlight_geometry_entities(
                kind,
                tuple(self._selected_geometry_ids),
            )
            labels = {
                "geometry_point": "点",
                "geometry_edge": "边",
                "geometry_face": "面",
                "geometry_body": "体",
            }
            self.status_panel.set_selection_mode(kind)
            selected_count = len(self._selected_geometry_ids)
            self.status_panel.set_object(
                (
                    f"已选择 {selected_count} 个"
                    f"{labels.get(kind, '几何实体')}"
                )
                if selected_count
                else "—"
            )
            self.actions["selected_info"].setEnabled(False)
            self._update_action_states()
            return
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

    def _on_viewport_pick_missed(self, kind: str) -> None:
        """Clear a replace-selection click without cancelling guided selection."""
        if self._geometry_pick_is_additive():
            return
        if kind.startswith("geometry_"):
            self._selected_geometry_kind = None
            self._selected_geometry_id = None
            self._selected_geometry_ids.clear()
        else:
            self.selection.clear()
        self.viewport.clear_selection()
        self.status_panel.set_object()
        self.actions["selected_info"].setEnabled(False)
        self._update_action_states()

    def _confirm_guided_selection(self) -> None:
        if not self._selected_geometry_ids:
            if self._pending_local_mesh_selection or self._pending_analysis_selection:
                self.status_panel.set_state("请先选择至少一个几何对象", 3000)
            return
        if self._pending_local_mesh_selection:
            self._pending_local_mesh_selection = False
            QTimer.singleShot(0, self.set_local_mesh_control)
            return
        if self._pending_analysis_selection is not None:
            operation = self._pending_analysis_selection
            self._pending_analysis_selection = None
            callback = (
                self.create_displacement_boundary
                if operation == "boundary"
                else self.create_load
            )
            QTimer.singleShot(0, callback)

    def _cancel_guided_selection(self) -> None:
        if not self._pending_local_mesh_selection and self._pending_analysis_selection is None:
            return
        self._pending_local_mesh_selection = False
        self._pending_analysis_selection = None
        self.clear_selection()
        self.status_panel.set_state("已取消区域选择", 3000)

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

    def _current_result_projection(self) -> ResultData | None:
        record = self.session.current_result()
        data = self.result_data
        artifact = self.document.artifact
        if record is None or data is None or artifact is None:
            return None
        provenance = record.provenance
        if (
            provenance.artifact_id != artifact.artifact_id
            or provenance.run_id != data.run_id
            or data.artifact_id != artifact.artifact_id
            or self.geometry is None
            or self.geometry.artifact_id != artifact.artifact_id
            or self.viewport.artifact_id != artifact.artifact_id
            or self.viewport.run_id != provenance.run_id
        ):
            return None
        return data

    def set_shape_mode(self, shape_mode: str) -> None:
        if self._current_result_projection() is None:
            return
        shape = "deformed" if shape_mode == "deformed" else "undeformed"
        self._display = replace(self._display, shape_mode=shape)
        self.actions[shape].setChecked(True)
        self._apply_scale()
        self._apply_display()

    def _toggle_contour(self, checked: bool) -> None:
        if self._current_result_projection() is None:
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
        if self._current_result_projection() is None:
            return
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
        data = self._current_result_projection()
        if data is None or field_key not in data.fields:
            return
        if not data.field_ready(field_key):
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

        if data.run_id is None:
            return False
        projection = self.session.prepare_result_projection(data.run_id)
        self.status_panel.set_state("正在恢复应力结果……")

        def workload(context: TaskContext) -> tuple[ResultData, float]:
            context.report("正在恢复应力结果……")
            started = perf_counter()
            updated = recovered_stress_data(data, required)
            context.checkpoint()
            return updated, perf_counter() - started

        def succeeded(value: object) -> None:
            updated, _seconds = value
            if not self._apply_session_delta(
                self.session.accept_result_projection(
                    projection.token
                ),
                result_projection=updated,
            ):
                return
            if (
                self.result_data is None
                or self.result_data.run_id != projection.run_id
            ):
                return
            self._refresh_job_manager()
            self.status_panel.set_state("应力结果恢复完成", 4000)
            on_ready()

        self._start_task(
            workload,
            succeeded,
            "应力结果恢复失败",
            lambda message: self._session_task_failed(
                projection.token,
                "应力结果恢复失败",
                message,
            ),
            task_name="应力恢复",
            on_cancelled=lambda: self._session_task_cancelled(
                projection.token
            ),
        )
        return False

    def _result_status_text(self) -> str:
        if self._current_result_projection() is None:
            return "—"
        shape = "变形" if self._display.shape_mode == "deformed" else "未变形"
        if not self._display.contour_enabled:
            return f"{shape} / 无云图"
        field = self.result_data.fields.get(self._display.field_key or "")
        if field is None:
            return f"{shape} / 云图"
        prefix = (self._display.field_key or "").split(":", 1)[0]
        position = (
            self.result_data.stress_position_label(prefix)
            if ":" in (self._display.field_key or "")
            else None
        )
        component = (self._display.field_key or "").split(":", 1)[-1]
        result_name = f"S {component}（{position}）" if position else field.label
        return f"{shape} / {result_name}"

    def show_result_display_dialog(self) -> None:
        if self._current_result_projection() is None:
            return
        step_name = self._current_step_name or "分析结果"
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
        if self.geometry is None or self._current_result_projection() is None:
            return
        scale = automatic_deformation_scale(self.geometry, self.result_data) if self._scale_mode == "auto" else 1.0 if self._scale_mode == "real" else self._scale_value
        self.viewport.set_deformation_scale(scale)

    def show_contour_dialog(self) -> None:
        if self._current_result_projection() is None:
            return
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
            self.session.runnable_step_names(),
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
            self._current_step_name = settings.step_name
        self.viewport.set_symbol_settings(settings)
        self._sync_step_combos()
        self.status_panel.set_step(self._current_step_name)
        self._update_action_states()

    def query_result(self) -> None:
        if self._current_result_projection() is None or self.geometry is None:
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
            step_name=self._current_step_name or "分析结果",
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

    def export_csv(self) -> None:
        if self._current_result_projection() is None:
            return
        field_key = self._display.field_key
        if field_key is None or field_key not in self.result_data.fields:
            self._show_error("导出 CSV 失败", "请先选择一个结果字段")
            return
        scalar = self.result_data.fields[field_key]
        if not scalar.ready:
            prefix = field_key.split(":", 1)[0]
            self._ensure_result_stress((prefix,), self.export_csv)
            return
        stem = self.document.path.stem if self.document.path else "result"
        safe_field = field_key.replace(":", "_")
        default = f"{stem}_{safe_field}.csv"
        path, _filter = QFileDialog.getSaveFileName(
            self, "导出当前结果字段", default, "CSV 文件 (*.csv)"
        )
        if not path:
            return
        target = Path(path).with_suffix(".csv")
        data = self.result_data
        self.status_panel.set_state("正在导出 CSV……")

        def workload(context: TaskContext):
            context.report("正在导出 CSV……")
            result = export_field_csv(data, field_key, target)
            context.checkpoint()
            return result

        self._start_task(
            workload,
            lambda _value: self.status_panel.set_state("CSV 导出完成", 5000),
            "导出 CSV 失败",
            task_name="CSV 导出",
        )

    def export_viewport_image(self) -> None:
        if self._current_result_projection() is None:
            return
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
        if self.inspection_service is None:
            source = {
                "native": "自主模型",
                "imported": "INP 模型",
            }.get(self.document.source_kind, "未打开")
            show_information(self, "模型概况", [
                ("模型来源", source),
                (
                    "几何状态",
                    "已创建" if self.document.has_native_geometry else "未创建",
                ),
                (
                    "网格状态",
                    "当前"
                    if self.document.mesh_is_current
                    else "未生成或已过期",
                ),
                ("材料数量", len(self.document.material_definitions)),
                ("截面数量", len(self.document.section_definitions)),
                ("分析步数量", len(self.document.analysis_definitions)),
                (
                    "当前状态",
                    "未保存"
                    if self.document.dirty
                    else "模型检查已通过"
                    if self._current_step_name
                    and self.session.can_submit(self._current_step_name)
                    else "就绪",
                ),
            ])
            return
        self.show_entity_information("model", None)

    def show_about(self) -> None:
        show_information(self, "关于", [("软件", "有限元分析"), ("功能", "Abaqus INP 线性静力分析与结果查看"), ("界面", "PySide6、PyVistaQt、VTK")])

    def _show_entry_information(self, kind: str, key: object) -> None:
        if kind == "mesh":
            self.show_mesh_browser()
        else:
            self.show_entity_information(kind, key)

    def _edit_tree_entry(self, kind: str, key: object) -> None:
        if kind == "material":
            self.edit_material(str(key))
        elif kind in {
            "step",
            "boundary",
            "cload",
            "edge_load",
            "surface_load",
            "gravity_load",
            "output",
        }:
            self.edit_analysis_definition(kind, key)
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
            box = QMessageBox(
                QMessageBox.Icon.Question,
                "任务正在运行",
                "是否取消当前任务并在安全退出后关闭程序？",
                parent=self,
            )
            cancel_button = box.addButton(
                "取消任务并退出",
                QMessageBox.ButtonRole.AcceptRole,
            )
            box.addButton(
                "继续等待",
                QMessageBox.ButtonRole.RejectRole,
            )
            box.exec()
            if box.clickedButton() is cancel_button:
                self._close_after_task_cancel = True
                self.cancel_current_task()
            event.ignore()
            return
        if self.isVisible() and not self._confirm_discard_changes():
            event.ignore()
            return
        self._close_inspection_windows()
        self._close_job_manager()
        if self.document.is_open:
            self._apply_session_delta(self.session.close())
        event.accept()
