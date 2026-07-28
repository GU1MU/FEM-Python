"""第一版中文有限元主窗口。"""

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from dataclasses import replace
import logging
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

import numpy as np
from PySide6.QtCore import QSettings, Qt, QTimer, Signal
from PySide6.QtGui import QAction, QCloseEvent
from PySide6.QtWidgets import (
    QApplication, QComboBox, QDialog, QFileDialog, QGridLayout,
    QInputDialog, QLabel, QMainWindow, QMessageBox, QSizePolicy, QSplitter,
    QVBoxLayout, QWidget,
)

from fem.abaqus import (
    build_model_with_report as build_abaqus_model_with_report,
    parse_file,
)
from fem.application import (
    AnalysisRun,
    AuthoringCapability,
    AuthoringStatus,
    BeamFrameReport,
    DefinitionEditBatch,
    DeleteIntent,
    DefinitionRejected,
    ModelCapabilityReport,
    ModelDefinitions,
    ModelSession,
    MeshEntityRef,
    NamedRegion,
    NamedRegionEditBatch,
    NativePart,
    PreflightDiagnostic,
    PreflightReport,
    RegionRef,
    RegionAssignment,
    RenameIntent,
    RevisionConflictError,
    RunStatus,
    SessionDelta,
    TokenStatus,
    TransitionEffect,
    UNSET,
    describe_model_capabilities,
    describe_native_authoring_capabilities,
    describe_session_authoring,
    evaluate_authoring_candidate,
    evaluate_native_assignment_candidate,
    evaluate_native_line_load_candidate,
    resolve_effective_beam_frames,
    safe_static_preflight,
)
from fem.application.preprocessing import generate_fem_model
from fem.application.results import (
    FieldAvailability,
    FieldPosition,
    FieldState,
    ResultQuery,
    ResultQueryResult,
    ResultQueryValidationError,
    ResultExportSnapshot,
    ResultMaterializationPatch,
    ResultProvider,
    ResultSourceKey,
    ResultVariable,
    ScalarFieldSelection,
    advance_materialization,
    build_solve_result_bundle,
    prepare_result_export_snapshot,
    project_scalar_field_topology,
    restore_result_provider,
)
from fem.core.model import (
    EdgeLoad,
    GravityLoad,
    LineLoad,
    NodalLoad,
    OutputRequest,
    SurfaceLoad,
)
from fem.geometry import (
    BooleanGeometry,
    BoxGeometry,
    CylinderGeometry,
    DiskGeometry,
    ExtrudedGeometry,
    LogicalEntityRef,
    MovedGeometry,
    NATIVE_GEOMETRY_TYPES,
    PlateWithHoleGeometry,
    RectangleGeometry,
    RotatedGeometry,
    SketchGeometry,
    WireGeometry,
    geometry_dimension,
    logical_ref_sort_key,
    recipe_characteristic_size,
    supports_structured_hexahedron,
)
from fem.io.project import load_project, save_project
from fem.io.result_csv import write_result_csv
from fem.io.result_vtk import write_result_vtk
from fem.mesh.quality import analyze_mesh
from fem.mesh.settings import MeshSettings
from fem.solvers import static_linear

from .actions import build_actions
from .action_state import GuiActionContext, derive_action_availability
from .analysis_dialogs import JobManagerDialog, JobSubmitDialog
from .analysis_definition_dialogs import (
    AnalysisDefinitionManagerDialog,
    DisplacementDialog,
    LoadDialog,
    OutputRequestDialog,
    StaticStepDialog,
)
from .commands import (
    CloseSessionCommand,
    GuiCommandCompletion,
    GuiCommandDiagnostic,
    GuiCommandOutcome,
    GuiCommandReceipt,
    GuiCommandStatus,
    MeshInputEdit,
    NativeGeometryEdit,
    NewNativeProjectCommand,
    ResultCsvExportSpec,
    ResultVtkExportSpec,
)
from .dialogs import CompactDoubleSpinBox, show_information
from .model_dialogs import (
    MaterialEditDialog,
    MaterialManagerDialog,
    RegionAssignmentDialog,
    SectionManagerDialog,
)
from .inspection_dialogs import EntityInfoDialog
from .inspection_service import InspectionService
from .mesh_browser import MeshBrowserDialog
from .geometry_preview import build_geometry_preview
from .scope_selection import (
    ScopeSelectionTopology,
    build_scope_selection_topology,
)
from .postprocessing_dialogs import (
    ContourSettingsDialog,
    TypedResultDisplayDialog,
    TypedResultDisplaySettings,
    TypedResultQueryDialog,
)
from .preprocessing_dialogs import (
    BasicSolidCreationDialog,
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
    GeometryCreationDialog,
    BooleanGeometryDialog,
    SketchGeometryDialog,
    NamedRegionDialog,
    NamedRegionManagerDialog,
)
from .symbol_dialog import SymbolSettingsDialog
from .task_controller import (
    BackgroundTaskController,
    BackgroundTaskState,
    TaskApplyOutcome,
    TaskCompletion,
)
from .viewport_background import (
    ViewportBackgroundSettings,
    load_background_settings,
    save_background_settings,
)
from .viewport_background_dialog import ViewportBackgroundDialog
from .sketch_editor import SketchDraftController, SketchDraftValidationError
from .wire_editor import WireDraftController, WireDraftValidationError
from .visualization.model_adapter import ModelGeometry, build_model_geometry
from .visualization.result_renderer import (
    ResultRenderPayload,
    build_result_render_payload,
)
from .visualization.selection import SelectionState
from .visualization.scene import DisplayState
from .visualization.symbols import SymbolSettings
from .widgets.navigation_panel import NavigationPanel
from .widgets.ribbon import RibbonPage, RibbonWidget
from .widgets.sketch_editor_panel import SketchEditorPanel
from .widgets.wire_editor_panel import WireEditorPanel
from .widgets.status_bar import CAEStatusBar
from .widgets.viewport import FEMViewport
from .widgets.viewport_toolbar import ViewportPanel
from .workers import TaskContext


_IMPORTED_OUTPUT_REQUEST_WARNING = (
    "此修改只保留在当前 Session；"
    "重新加载原 INP 后会恢复源文件中的输出请求。"
)

_RESULT_VARIABLE_LABELS = {
    ResultVariable.U: "位移 U",
    ResultVariable.UR: "转角 UR",
    ResultVariable.RF: "反力 RF",
    ResultVariable.RM: "反力矩 RM",
    ResultVariable.LE: "对数应变 LE",
    ResultVariable.S: "应力 S",
}
_RESULT_POSITION_LABELS = {
    FieldPosition.NODE: "节点",
    FieldPosition.INTEGRATION_POINT: "积分点",
    FieldPosition.CENTROID: "单元质心",
    FieldPosition.ELEMENT_NODAL: "单元节点",
    FieldPosition.NODE_REGION: "节点区域",
    FieldPosition.RESOLVED_NODAL: "平均节点",
    FieldPosition.SECTION_END: "截面端点",
    FieldPosition.SECTION_NODE_ENVELOPE: "截面节点包络",
}
_RESULT_FIELD_STATE_LABELS = {
    FieldState.LAZY: "按需加载",
    FieldState.UNAVAILABLE: "不可用",
}


def initial_display_policy(
    element_count: int,
    node_count: int,
    *,
    line_mesh: bool = False,
) -> dict[str, bool]:
    """Return the explicit first-display degradation policy for large models."""
    return {
        "show_edges": int(element_count) <= 100_000,
        "show_symbols": int(element_count) <= 200_000,
        "show_nodes": bool(line_mesh) and int(node_count) <= 20_000,
        "show_labels": False,
        "simplified": int(element_count) > 100_000 or int(node_count) > 200_000,
    }


class _ExactDataComboBox(QComboBox):
    """Keep typed Python user data from being coerced by QVariant."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._exact_user_data: list[object] = []

    def addItem(self, text: str, user_data: object = None) -> None:
        super().addItem(text)
        self._exact_user_data.append(user_data)

    def clear(self) -> None:
        super().clear()
        self._exact_user_data.clear()

    def itemData(
        self,
        index: int,
        role: int = Qt.ItemDataRole.UserRole,
    ) -> object:
        if role == Qt.ItemDataRole.UserRole:
            if 0 <= index < len(self._exact_user_data):
                return self._exact_user_data[index]
            return None
        return super().itemData(index, role)

    def currentData(
        self,
        role: int = Qt.ItemDataRole.UserRole,
    ) -> object:
        return self.itemData(self.currentIndex(), role)

    def findData(
        self,
        data: object,
        role: int = Qt.ItemDataRole.UserRole,
        flags: Qt.MatchFlag = (
            Qt.MatchFlag.MatchExactly | Qt.MatchFlag.MatchCaseSensitive
        ),
    ) -> int:
        if role == Qt.ItemDataRole.UserRole:
            return next(
                (
                    index
                    for index, candidate in enumerate(self._exact_user_data)
                    if type(candidate) is type(data) and candidate == data
                ),
                -1,
            )
        return super().findData(data, role, flags)


class FEMMainWindow(QMainWindow):
    """只暴露当前内核已经实现的有限元工作流。"""

    resultQueryCompleted = Signal(object)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("有限元分析")
        self.resize(1280, 800)
        self.session = ModelSession()
        self.document = self.session.snapshot()
        self._applied_session_revision = self.document.session_revision
        self._import_notices: tuple[object, ...] = ()
        self._current_step_name: str | None = None
        self.geometry: ModelGeometry | None = None
        self.result_provider: ResultProvider | None = None
        self.result_selection: ScalarFieldSelection | None = None
        self._pending_result_selection: ScalarFieldSelection | None = None
        self._pending_result_source: ResultSourceKey | None = None
        self._pending_result_generation: int | None = None
        self._pending_result_query: ResultQuery | None = None
        self._pending_result_query_source: ResultSourceKey | None = None
        self._pending_result_query_generation: int | None = None
        self.inspection_service: InspectionService | None = None
        self._inspection_windows: list[QWidget] = []
        self._mesh_browser: MeshBrowserDialog | None = None
        self._selected_geometry_refs: set[LogicalEntityRef] = set()
        self._selected_mesh_scope_refs: set[MeshEntityRef] = set()
        self._geometry_selection_mode = "body"
        self._pending_local_mesh_selection = False
        self._pending_analysis_selection: str | None = None
        self._pending_scope_kind: str | None = None
        self._scope_selection_overlay_active = False
        self._scope_selection_topology_cache: (
            ScopeSelectionTopology | None
        ) = None
        self._wire_editor_controller: WireDraftController | None = None
        self._wire_editor_original_recipe: object | None = None
        self._wire_editor_base_revision: int | None = None
        self._sketch_editor_controller: SketchDraftController | None = None
        self._sketch_editor_original_recipe: object | None = None
        self._sketch_editor_base_revision: int | None = None
        self.selection = SelectionState()
        self.actions: dict[str, QAction] = {}
        self.task_controller = BackgroundTaskController(self)
        self._command_counter = 0
        self._job_manager: JobManagerDialog | None = None
        self._viewport_fit_pending = False
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
        self.task_controller.busy_changed.connect(self._task_busy_changed)
        self.task_controller.cancelling_changed.connect(
            self._task_cancelling_changed
        )
        self._refresh_result_controls()
        self._update_action_states()

    @property
    def busy(self) -> bool:
        return self.task_controller.busy

    @property
    def import_notices(self) -> tuple[object, ...]:
        """Return detached, non-authoritative notices for the current import."""

        return deepcopy(self._import_notices)

    def _next_command_id(self) -> int:
        self._command_counter += 1
        return self._command_counter

    @staticmethod
    def _rejected_command(
        command_id: int,
        code: str,
        error: object,
        remediation: str = "",
    ) -> GuiCommandReceipt:
        message = str(error).strip() or type(error).__name__
        return GuiCommandReceipt.rejected(
            command_id,
            GuiCommandDiagnostic(code, message, remediation),
        )

    def _show_command_rejection(
        self,
        title: str,
        receipt: GuiCommandReceipt,
    ) -> None:
        diagnostic = receipt.diagnostic
        if diagnostic is None:
            raise ValueError("rejected command receipt requires a diagnostic")
        message = diagnostic.message
        if diagnostic.remediation:
            message += f"\n建议：{diagnostic.remediation}"
        self._show_error(title, message)

    def _accepted_command(
        self,
        command_id: int,
        delta: SessionDelta,
        **projection: object,
    ) -> GuiCommandReceipt:
        rebuild_required = False
        try:
            applied = self._apply_session_delta(delta, **projection)
        except Exception:
            logging.exception("synchronous GUI command projection failed")
            rebuild_required = True
        else:
            if delta.accepted and not applied:
                rebuild_required = True
        if rebuild_required:
            try:
                self._rebuild_full_projection()
            except Exception as error:
                logging.exception("synchronous full GUI projection rebuild failed")
                self._task_projection_failed(
                    str(error).strip() or type(error).__name__
                )
        return GuiCommandReceipt.accepted(command_id, delta)

    def new_native_project(
        self,
        command: NewNativeProjectCommand,
    ) -> GuiCommandReceipt:
        command_id = self._next_command_id()
        if type(command) is not NewNativeProjectCommand:
            return self._rejected_command(
                command_id,
                "command.type.invalid",
                "command must be NewNativeProjectCommand",
            )
        if self.busy:
            return self._rejected_command(
                command_id,
                "task.busy",
                "a background task is already running",
            )
        try:
            delta = self.session.new_native_project(
                command.name,
                expected_session_revision=command.expected_session_revision,
            )
            receipt = self._accepted_command(command_id, delta)
        except (RevisionConflictError, TypeError, ValueError) as error:
            return self._rejected_command(
                command_id,
                "session.new_native.rejected",
                error,
            )
        self._import_notices = ()
        return receipt

    def close_session(
        self,
        command: CloseSessionCommand,
    ) -> GuiCommandReceipt:
        command_id = self._next_command_id()
        if type(command) is not CloseSessionCommand:
            return self._rejected_command(
                command_id,
                "command.type.invalid",
                "command must be CloseSessionCommand",
            )
        if self.busy:
            return self._rejected_command(
                command_id,
                "task.busy",
                "a background task is already running",
            )
        try:
            delta = self.session.close(
                expected_session_revision=command.expected_session_revision
            )
            receipt = self._accepted_command(command_id, delta)
        except (RevisionConflictError, RuntimeError, TypeError, ValueError) as error:
            return self._rejected_command(
                command_id,
                "session.close.rejected",
                error,
            )
        self._import_notices = ()
        return receipt

    def open_project_path(self, path: str | Path) -> GuiCommandReceipt:
        command_id = self._next_command_id()
        if self.busy:
            return self._rejected_command(
                command_id,
                "task.busy",
                "a background task is already running",
            )
        if (
            self._wire_editor_controller is not None
            or self._sketch_editor_controller is not None
        ):
            return self._rejected_command(
                command_id,
                "sketch_editor.active",
                "请先完成或取消当前草图编辑，再打开项目",
            )
        target = Path(path)
        try:
            loaded = load_project(target)
            delta = self.session.replace_from_snapshot(
                loaded.snapshot,
                expected_session_revision=self.document.session_revision,
            )
            receipt = self._accepted_command(
                command_id,
                delta,
                source_label=target.name,
            )
        except Exception as error:
            return self._rejected_command(
                command_id,
                "project.open.rejected",
                error,
                "请检查项目文件版本和内容。",
            )
        self._import_notices = deepcopy(tuple(loaded.notices))
        if loaded.notices:
            self.status_panel.set_state(
                "；".join(notice.message for notice in loaded.notices),
                12000,
            )
        return receipt

    def open_inp_path(self, path: str | Path) -> GuiCommandReceipt:
        command_id = self._next_command_id()
        if self.busy:
            return self._rejected_command(
                command_id,
                "task.busy",
                "a background task is already running",
            )
        try:
            target = Path(path)
            completion = GuiCommandCompletion(command_id)
            started = self._begin_import(target, completion=completion)
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            return self._rejected_command(
                command_id,
                "import.open.rejected",
                error,
            )
        if not started:
            return self._rejected_command(
                command_id,
                "task.start.rejected",
                "the import task could not be started",
            )
        return GuiCommandReceipt.pending(command_id, completion)

    def reload_imported_source(self) -> GuiCommandReceipt:
        command_id = self._next_command_id()
        if self.busy:
            return self._rejected_command(
                command_id,
                "task.busy",
                "a background task is already running",
            )
        source_path = self.document.source_path
        if self.document.source_kind != "imported" or source_path is None:
            return self._rejected_command(
                command_id,
                "import.reload.unavailable",
                "the current session has no imported source to reload",
            )
        try:
            completion = GuiCommandCompletion(command_id)
            started = self._begin_import(
                Path(source_path),
                completion=completion,
            )
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            return self._rejected_command(
                command_id,
                "import.reload.rejected",
                error,
            )
        if not started:
            return self._rejected_command(
                command_id,
                "task.start.rejected",
                "the reload task could not be started",
            )
        return GuiCommandReceipt.pending(command_id, completion)

    def save_project_path(self, path: str | Path) -> GuiCommandReceipt:
        command_id = self._next_command_id()
        if not self.document.can_save:
            return self._rejected_command(
                command_id,
                "project.save.unavailable",
                "current session cannot be saved as a native project",
            )
        target = Path(path)
        if target.suffix.casefold() != ".femproj":
            target = target.with_suffix(".femproj")
        save_snapshot = None
        try:
            save_snapshot = self.session.prepare_project_save()
            saved_path = save_project(target, save_snapshot)
            delta = self.session.accept_project_saved(
                save_snapshot.token,
                saved_path,
            )
            return self._accepted_command(command_id, delta)
        except Exception as error:
            if save_snapshot is not None:
                failure = self.session.accept_task_failed(
                    save_snapshot.token,
                    error,
                )
                if failure.accepted:
                    self._apply_session_delta(failure)
            return self._rejected_command(
                command_id,
                "project.save.rejected",
                error,
            )

    def apply_native_geometry_edit(
        self,
        edit: NativeGeometryEdit,
    ) -> GuiCommandReceipt:
        command_id = self._next_command_id()
        if type(edit) is not NativeGeometryEdit:
            return self._rejected_command(
                command_id,
                "command.type.invalid",
                "edit must be NativeGeometryEdit",
            )
        try:
            delta = self.session.replace_native_geometry_inputs(
                edit.parts,
                edit.recipe,
                mesh_settings=edit.mesh_settings,
                expected_session_revision=edit.base_session_revision,
            )
            return self._accepted_command(command_id, delta)
        except (RevisionConflictError, RuntimeError, TypeError, ValueError) as error:
            return self._rejected_command(
                command_id,
                "geometry.edit.rejected",
                error,
            )

    def apply_mesh_input_edit(
        self,
        edit: MeshInputEdit,
    ) -> GuiCommandReceipt:
        command_id = self._next_command_id()
        if type(edit) is not MeshInputEdit:
            return self._rejected_command(
                command_id,
                "command.type.invalid",
                "edit must be MeshInputEdit",
            )
        try:
            delta = self.session.replace_mesh_settings(
                edit.settings,
                expected_session_revision=edit.base_session_revision,
            )
            return self._accepted_command(command_id, delta)
        except (RevisionConflictError, RuntimeError, TypeError, ValueError) as error:
            return self._rejected_command(
                command_id,
                "mesh.edit.rejected",
                error,
            )

    def apply_named_region_edit(
        self,
        batch: NamedRegionEditBatch,
    ) -> GuiCommandReceipt:
        command_id = self._next_command_id()
        try:
            delta = self.session.apply_named_region_edit(batch)
            return self._accepted_command(command_id, delta)
        except (RevisionConflictError, RuntimeError, TypeError, ValueError) as error:
            return self._rejected_command(
                command_id,
                "named_region.edit.rejected",
                error,
            )

    def apply_definition_edit(
        self,
        batch: DefinitionEditBatch,
    ) -> GuiCommandReceipt:
        command_id = self._next_command_id()
        try:
            delta = self.session.apply_definition_edit(batch)
            return self._accepted_command(command_id, delta)
        except DefinitionRejected as error:
            return self._rejected_command(
                command_id,
                "definition.edit.rejected",
                self._render_diagnostics(error.diagnostics),
            )
        except (RevisionConflictError, RuntimeError, TypeError, ValueError) as error:
            return self._rejected_command(
                command_id,
                "definition.edit.rejected",
                error,
            )

    def clear_generated_mesh(
        self,
        base_session_revision: int,
    ) -> GuiCommandReceipt:
        command_id = self._next_command_id()
        try:
            delta = self.session.clear_generated_model(
                expected_session_revision=base_session_revision
            )
            return self._accepted_command(command_id, delta)
        except (RevisionConflictError, RuntimeError, TypeError, ValueError) as error:
            return self._rejected_command(
                command_id,
                "mesh.clear.rejected",
                error,
            )

    def generate_mesh(self) -> GuiCommandReceipt:
        command_id = self._next_command_id()
        if self.busy:
            return self._rejected_command(
                command_id,
                "task.busy",
                "a background task is already running",
            )
        if not isinstance(
            self.document.geometry_recipe,
            NATIVE_GEOMETRY_TYPES,
        ) or not isinstance(self.document.mesh_settings, MeshSettings):
            return self._rejected_command(
                command_id,
                "mesh.generate.unavailable",
                "native geometry and mesh settings are required",
            )
        if self.document.model is not None:
            cleared = self.clear_generated_mesh(
                self.document.session_revision
            )
            if cleared.diagnostic is not None:
                return self._rejected_command(
                    command_id,
                    "mesh.generate.clear_failed",
                    cleared.diagnostic.message,
                )
        try:
            completion = GuiCommandCompletion(command_id)
            started = self._begin_mesh_generation(completion=completion)
        except (RuntimeError, TypeError, ValueError) as error:
            return self._rejected_command(
                command_id,
                "mesh.generate.rejected",
                error,
            )
        if not started:
            return self._rejected_command(
                command_id,
                "task.start.rejected",
                "the mesh-generation task could not be started",
            )
        return GuiCommandReceipt.pending(command_id, completion)

    def check_step(self, step_name: str) -> GuiCommandReceipt:
        command_id = self._next_command_id()
        if self.busy:
            return self._rejected_command(
                command_id,
                "task.busy",
                "a background task is already running",
            )
        if type(step_name) is not str or not step_name.strip():
            return self._rejected_command(
                command_id,
                "validation.step.invalid",
                "step_name must be a non-empty string",
            )
        clean_step = step_name.strip()
        if not self.session.can_check(clean_step):
            return self._rejected_command(
                command_id,
                "validation.unavailable",
                f"step cannot be checked: {clean_step}",
            )
        try:
            completion = GuiCommandCompletion(command_id)
            started = self._begin_model_check(
                clean_step,
                completion=completion,
                show_success=False,
            )
        except (KeyError, RuntimeError, TypeError, ValueError) as error:
            return self._rejected_command(
                command_id,
                "validation.rejected",
                error,
            )
        if not started:
            return self._rejected_command(
                command_id,
                "task.start.rejected",
                "the model-check task could not be started",
            )
        return GuiCommandReceipt.pending(command_id, completion)

    def submit_run(self, name: str, step_name: str) -> GuiCommandReceipt:
        command_id = self._next_command_id()
        if self.busy:
            return self._rejected_command(
                command_id,
                "task.busy",
                "a background task is already running",
            )
        try:
            completion = GuiCommandCompletion(command_id)
            job = self._begin_submit_run(
                name,
                step_name,
                completion=completion,
            )
        except (KeyError, RuntimeError, TypeError, ValueError) as error:
            return self._rejected_command(
                command_id,
                "run.submit.rejected",
                error,
            )
        if job is None:
            return self._rejected_command(
                command_id,
                "task.start.rejected",
                "the analysis task could not be started",
            )
        return GuiCommandReceipt.pending(command_id, completion)

    def select_run_result(self, run_id: str) -> GuiCommandReceipt:
        command_id = self._next_command_id()
        try:
            delta = self.session.select_result(str(run_id))
            receipt = self._accepted_command(command_id, delta)
        except (KeyError, RuntimeError, TypeError, ValueError) as error:
            return self._rejected_command(
                command_id,
                "result.select.rejected",
                error,
            )
        job = self.session.find_run(str(run_id))
        if job is not None and job.has_result:
            self._activate_job_result(job)
        return receipt

    def select_result_field(
        self,
        selection: ScalarFieldSelection,
    ) -> GuiCommandReceipt:
        """Select one exact scalar field, materializing LAZY data on demand."""

        command_id = self._next_command_id()
        if type(selection) is not ScalarFieldSelection:
            return self._rejected_command(
                command_id,
                "command.type.invalid",
                "selection must be a ScalarFieldSelection",
            )
        if self.busy:
            return self._rejected_command(
                command_id,
                "task.busy",
                "a background task is already running",
            )
        provider = self._current_result_provider()
        if provider is None:
            return self._rejected_command(
                command_id,
                "result.current.unavailable",
                "there is no current accepted result provider",
            )
        try:
            availability = self._catalog_availability_for_selection(
                provider,
                selection,
            )
        except (KeyError, TypeError, ValueError) as error:
            return self._rejected_command(
                command_id,
                "result.field.selection.invalid",
                error,
            )
        if availability.state is FieldState.LAZY:
            return self._begin_result_field_materialization(
                command_id,
                provider,
                selection,
            )
        if availability.state is FieldState.UNAVAILABLE:
            diagnostic = next(
                (
                    item
                    for item in availability.diagnostics
                    if item.message.strip()
                ),
                None,
            )
            return self._rejected_command(
                command_id,
                "result.field.unavailable",
                (
                    diagnostic.message
                    if diagnostic is not None
                    else "the selected field is unavailable"
                ),
                (
                    diagnostic.remediation
                    if diagnostic is not None
                    else ""
                ),
            )

        outcome = self._result_selection_outcome(provider, selection)
        if self.result_selection == selection:
            return GuiCommandReceipt.accepted(
                command_id,
                outcome=outcome,
            )
        try:
            self._install_ready_result_selection(provider, selection)
        except (KeyError, RuntimeError, TypeError, ValueError) as error:
            return self._rejected_command(
                command_id,
                "result.field.projection.failed",
                error,
            )
        return GuiCommandReceipt.accepted(
            command_id,
            outcome=outcome,
        )

    def _begin_result_field_materialization(
        self,
        command_id: int,
        provider: ResultProvider,
        selection: ScalarFieldSelection,
    ) -> GuiCommandReceipt:
        """Start one exact-key, generation-bound lazy field request."""

        task = None
        try:
            task = self.session.prepare_result_materialization(
                provider.source.run_id,
                (selection.field_key,),
            )
            completion = GuiCommandCompletion(command_id)
            accepted_delta: SessionDelta | None = None
            self._pending_result_selection = selection
            self._pending_result_source = task.materialization.source
            self._pending_result_generation = (
                task.materialization.generation
            )
            self._update_action_states()

            def workload(
                context: TaskContext,
            ) -> ResultMaterializationPatch:
                context.report("正在按需加载结果字段……")
                detached_provider = restore_result_provider(
                    task.record.result,
                    task.materialization,
                )
                return detached_provider.materialize(
                    task.field_keys,
                    cancellation=context,
                )

            def apply_result(value: object) -> TaskApplyOutcome:
                nonlocal accepted_delta
                if type(value) is not ResultMaterializationPatch:
                    raise TypeError(
                        "result materialization worker must return "
                        "ResultMaterializationPatch"
                    )
                delta = self.session.accept_result_materialization(
                    task.token,
                    value,
                )
                if not delta.accepted:
                    status = delta.token_status
                    message = delta.reason or (
                        status.value
                        if status is not None
                        else "result materialization was rejected"
                    )
                    if status in {
                        TokenStatus.WRONG_KIND,
                        TokenStatus.INVALID_STATE,
                    }:
                        return TaskApplyOutcome.rejected(message)
                    return TaskApplyOutcome.stale(message)

                accepted_delta = delta
                materialized = next(
                    (
                        field_data
                        for field_data in value.fields
                        if field_data.key == selection.field_key
                    ),
                    None,
                )
                outcome = GuiCommandOutcome(
                    source=task.materialization.source,
                    materialization_generation=(
                        task.materialization.generation
                        + (1 if value.fields else 0)
                    ),
                    selection=selection,
                    record_count=(
                        None
                        if materialized is None
                        else len(materialized.locations)
                    ),
                )
                return TaskApplyOutcome.accepted(outcome)

            def succeeded(value: object) -> None:
                if (
                    type(value) is not GuiCommandOutcome
                    or accepted_delta is None
                ):
                    raise RuntimeError(
                        "accepted result materialization has no outcome"
                    )
                if not self._apply_session_delta(accepted_delta):
                    raise RuntimeError(
                        "accepted result materialization could not be projected"
                    )
                current = self._current_result_provider()
                if (
                    current is not None
                    and current.source == value.source
                    and current.snapshot.generation
                    == value.materialization_generation
                ):
                    self._install_ready_result_selection(
                        current,
                        selection,
                    )
                self._refresh_job_manager()
                self.status_panel.set_state(
                    "结果字段按需加载完成",
                    4000,
                )

            started = self._start_task(
                workload,
                succeeded,
                "结果字段按需加载失败",
                lambda message: self._session_task_failed(
                    task.token,
                    "结果字段按需加载失败",
                    message,
                ),
                task_name="结果字段按需加载",
                on_cancelled=lambda: self._session_task_cancelled(
                    task.token
                ),
                apply_result=apply_result,
                completion=completion,
            )
        except (KeyError, RuntimeError, TypeError, ValueError) as error:
            if task is not None:
                failure = self.session.accept_task_failed(
                    task.token,
                    error,
                )
                self._apply_revision_neutral_task_receipt(failure)
            self._clear_pending_result_materialization()
            self._update_action_states()
            return self._rejected_command(
                command_id,
                "result.field.materialization.rejected",
                error,
            )

        if not started:
            cancelled = self.session.accept_task_cancelled(task.token)
            self._apply_revision_neutral_task_receipt(cancelled)
            self._clear_pending_result_materialization()
            self._update_action_states()
            return self._rejected_command(
                command_id,
                "task.start.rejected",
                "the result materialization task could not be started",
            )

        def finished(terminal: TaskCompletion) -> None:
            self._finish_result_field_materialization(
                selection,
                task.materialization.source,
                task.materialization.generation,
                terminal,
            )

        completion.observe(finished)
        return GuiCommandReceipt.pending(command_id, completion)

    def _finish_result_field_materialization(
        self,
        selection: ScalarFieldSelection,
        source: ResultSourceKey,
        generation: int,
        terminal: TaskCompletion,
    ) -> None:
        if (
            self._pending_result_selection != selection
            or self._pending_result_source != source
            or self._pending_result_generation != generation
        ):
            return
        self._clear_pending_result_materialization()
        projected = self.result_provider
        restore_generation = (
            generation + 1
            if (
                terminal.state is BackgroundTaskState.SUCCEEDED
                and terminal.projection_error is not None
            )
            else generation
        )
        if (
            (
                terminal.state is not BackgroundTaskState.SUCCEEDED
                or terminal.projection_error is not None
            )
            and type(projected) is ResultProvider
            and projected.source == source
            and projected.snapshot.generation
            in {generation, restore_generation}
        ):
            current_selection = self.result_selection
            if type(current_selection) is ScalarFieldSelection:
                self.result_tree.select_selection(current_selection)
        self._refresh_result_controls()
        self._update_action_states()

    def _clear_pending_result_materialization(self) -> None:
        self._pending_result_selection = None
        self._pending_result_source = None
        self._pending_result_generation = None

    def query_result(
        self,
        query: ResultQuery,
    ) -> GuiCommandReceipt:
        """Query one exact typed field without reading GUI selection state."""

        return self._submit_result_query(query)

    def _submit_result_query(
        self,
        query: ResultQuery,
        *,
        expected_source: ResultSourceKey | None = None,
    ) -> GuiCommandReceipt:
        command_id = self._next_command_id()
        if type(query) is not ResultQuery:
            return self._rejected_command(
                command_id,
                "command.type.invalid",
                "query must be a ResultQuery",
            )
        if (
            expected_source is not None
            and type(expected_source) is not ResultSourceKey
        ):
            return self._rejected_command(
                command_id,
                "command.type.invalid",
                "expected_source must be a ResultSourceKey or None",
            )
        provider = self._current_result_provider()
        if provider is None:
            return self._rejected_command(
                command_id,
                "result.current.unavailable",
                "there is no current accepted result provider",
            )
        if (
            expected_source is not None
            and provider.source != expected_source
        ):
            return self._rejected_command(
                command_id,
                "result.query.source.stale",
                "the query dialog source is no longer current",
            )
        if self.busy:
            return self._rejected_command(
                command_id,
                "task.busy",
                "a background task is already running",
            )

        try:
            availability = provider.validate_query(query)
        except ResultQueryValidationError as error:
            return self._rejected_command(
                command_id,
                error.code,
                error,
            )
        except (KeyError, RuntimeError, TypeError, ValueError) as error:
            return self._rejected_command(
                command_id,
                "result.query.validation.rejected",
                error,
            )

        if availability.state is FieldState.LAZY:
            return self._begin_result_query_materialization(
                command_id,
                provider,
                query,
            )
        if availability.state is not FieldState.READY:
            return self._rejected_command(
                command_id,
                "result.query.field.unavailable",
                "the selected query field is unavailable",
            )

        try:
            result = provider.query(query)
            self._validate_result_query_result(
                provider,
                query,
                result,
            )
        except ResultQueryValidationError as error:
            return self._rejected_command(
                command_id,
                error.code,
                error,
            )
        except (KeyError, RuntimeError, TypeError, ValueError) as error:
            return self._rejected_command(
                command_id,
                "result.query.rejected",
                error,
            )

        outcome = self._result_query_outcome(result)
        self.resultQueryCompleted.emit(result)
        return GuiCommandReceipt.accepted(
            command_id,
            outcome=outcome,
        )

    def _begin_result_query_materialization(
        self,
        command_id: int,
        provider: ResultProvider,
        query: ResultQuery,
    ) -> GuiCommandReceipt:
        """Materialize one lazy query field through the Session CAS gate."""

        task = None
        try:
            task = self.session.prepare_result_materialization(
                provider.source.run_id,
                (query.field_key,),
            )
            completion = GuiCommandCompletion(command_id)
            accepted_delta: SessionDelta | None = None
            accepted_result: ResultQueryResult | None = None
            self._pending_result_query = query
            self._pending_result_query_source = (
                task.materialization.source
            )
            self._pending_result_query_generation = (
                task.materialization.generation
            )
            self._update_action_states()

            def workload(
                context: TaskContext,
            ) -> ResultMaterializationPatch:
                context.report("正在按需加载查询字段……")
                detached_provider = restore_result_provider(
                    task.record.result,
                    task.materialization,
                )
                return detached_provider.materialize(
                    task.field_keys,
                    cancellation=context,
                )

            def apply_result(value: object) -> TaskApplyOutcome:
                nonlocal accepted_delta, accepted_result
                if type(value) is not ResultMaterializationPatch:
                    raise TypeError(
                        "result query worker must return "
                        "ResultMaterializationPatch"
                    )
                delta = self.session.accept_result_materialization(
                    task.token,
                    value,
                )
                if not delta.accepted:
                    status = delta.token_status
                    message = delta.reason or (
                        status.value
                        if status is not None
                        else "result query materialization was rejected"
                    )
                    if status in {
                        TokenStatus.WRONG_KIND,
                        TokenStatus.INVALID_STATE,
                    }:
                        return TaskApplyOutcome.rejected(message)
                    return TaskApplyOutcome.stale(message)

                accepted_delta = delta
                try:
                    materialization = advance_materialization(
                        task.materialization,
                        value,
                    )
                    accepted_provider = restore_result_provider(
                        task.record.result,
                        materialization,
                    )
                    result = accepted_provider.query(query)
                    self._validate_result_query_result(
                        accepted_provider,
                        query,
                        result,
                    )
                except Exception:
                    self._project_accepted_result_query_delta(delta)
                    raise

                accepted_result = result
                return TaskApplyOutcome.accepted(
                    self._result_query_outcome(result)
                )

            def succeeded(value: object) -> None:
                if (
                    type(value) is not GuiCommandOutcome
                    or accepted_delta is None
                    or accepted_result is None
                ):
                    raise RuntimeError(
                        "accepted result query has no typed outcome"
                    )
                if not self._project_accepted_result_query_delta(
                    accepted_delta
                ):
                    raise RuntimeError(
                        "accepted result query could not be projected"
                    )
                current = self._current_result_provider()
                if (
                    current is not None
                    and current.source == accepted_result.source
                    and current.snapshot.generation
                    == accepted_result.materialization_generation
                ):
                    self.resultQueryCompleted.emit(accepted_result)
                self._refresh_job_manager()
                self.status_panel.set_state(
                    "结果查询完成",
                    4000,
                )

            started = self._start_task(
                workload,
                succeeded,
                "结果查询失败",
                lambda message: self._session_task_failed(
                    task.token,
                    "结果查询失败",
                    message,
                ),
                task_name="结果查询",
                on_cancelled=lambda: self._session_task_cancelled(
                    task.token
                ),
                apply_result=apply_result,
                completion=completion,
            )
        except (KeyError, RuntimeError, TypeError, ValueError) as error:
            if task is not None:
                failure = self.session.accept_task_failed(
                    task.token,
                    error,
                )
                self._apply_revision_neutral_task_receipt(failure)
            self._clear_pending_result_query_materialization()
            self._update_action_states()
            return self._rejected_command(
                command_id,
                "result.query.materialization.rejected",
                error,
            )

        if not started:
            cancelled = self.session.accept_task_cancelled(task.token)
            self._apply_revision_neutral_task_receipt(cancelled)
            self._clear_pending_result_query_materialization()
            self._update_action_states()
            return self._rejected_command(
                command_id,
                "task.start.rejected",
                "the result query task could not be started",
            )

        def finished(terminal: TaskCompletion) -> None:
            self._finish_result_query_materialization(
                query,
                task.materialization.source,
                task.materialization.generation,
                terminal,
            )

        completion.observe(finished)
        return GuiCommandReceipt.pending(command_id, completion)

    def _finish_result_query_materialization(
        self,
        query: ResultQuery,
        source: ResultSourceKey,
        generation: int,
        _terminal: TaskCompletion,
    ) -> None:
        if (
            self._pending_result_query != query
            or self._pending_result_query_source != source
            or self._pending_result_query_generation != generation
        ):
            return
        self._clear_pending_result_query_materialization()
        self._update_action_states()

    def _clear_pending_result_query_materialization(self) -> None:
        self._pending_result_query = None
        self._pending_result_query_source = None
        self._pending_result_query_generation = None

    def _project_accepted_result_query_delta(
        self,
        delta: SessionDelta,
    ) -> bool:
        if delta.changed or delta.invalidated:
            return self._apply_session_delta(delta)
        return self._apply_revision_neutral_task_receipt(delta)

    @staticmethod
    def _validate_result_query_result(
        provider: ResultProvider,
        query: ResultQuery,
        result: ResultQueryResult,
    ) -> None:
        if type(result) is not ResultQueryResult:
            raise TypeError("provider query must return ResultQueryResult")
        if (
            result.source != provider.source
            or result.materialization_generation
            != provider.snapshot.generation
            or result.query != query
        ):
            raise ValueError(
                "query result must match the provider source, "
                "generation, and exact query"
            )

    @staticmethod
    def _result_query_outcome(
        result: ResultQueryResult,
    ) -> GuiCommandOutcome:
        return GuiCommandOutcome(
            source=result.source,
            materialization_generation=(
                result.materialization_generation
            ),
            selection=ScalarFieldSelection(
                result.query.field_key,
                result.query.component,
            ),
            record_count=len(result.records),
        )

    def export_result_csv(
        self,
        path: str | Path,
        spec: ResultCsvExportSpec,
    ) -> GuiCommandReceipt:
        """Export one exact ready scalar selection through canonical CSV I/O."""

        command_id = self._next_command_id()
        if self.busy:
            return self._rejected_command(
                command_id,
                "task.busy",
                "a background task is already running",
            )
        if type(spec) is not ResultCsvExportSpec:
            return self._rejected_command(
                command_id,
                "command.type.invalid",
                "spec must be ResultCsvExportSpec",
            )
        try:
            target = Path(path)
            if target.suffix.casefold() != ".csv":
                raise ValueError(
                    "canonical result CSV target must use the .csv extension"
                )
            export = self._prepare_result_export(spec)
            completion = GuiCommandCompletion(command_id)

            def workload(context: TaskContext) -> GuiCommandOutcome:
                installed = write_result_csv(
                    target,
                    export,
                    checkpoint=context.checkpoint,
                )
                return GuiCommandOutcome(
                    output_path=installed,
                    source=export.source,
                    materialization_generation=(
                        export.materialization_generation
                    ),
                    selection=export.selection,
                    record_count=len(export.field.locations),
                )

            started = self._start_task(
                workload,
                lambda _outcome: self.status_panel.set_state(
                    "CSV 导出完成",
                    5000,
                ),
                "导出 CSV 失败",
                task_name="CSV 导出",
                completion=completion,
            )
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            return self._rejected_command(
                command_id,
                "result.csv_export.rejected",
                error,
            )
        if not started:
            return self._rejected_command(
                command_id,
                "task.start.rejected",
                "the CSV export task could not be started",
            )
        return GuiCommandReceipt.pending(command_id, completion)

    def export_result_vtk(
        self,
        path: str | Path,
        spec: ResultVtkExportSpec,
    ) -> GuiCommandReceipt:
        """Export one exact ready scalar selection through canonical VTK I/O."""

        command_id = self._next_command_id()
        if self.busy:
            return self._rejected_command(
                command_id,
                "task.busy",
                "a background task is already running",
            )
        if type(spec) is not ResultVtkExportSpec:
            return self._rejected_command(
                command_id,
                "command.type.invalid",
                "spec must be ResultVtkExportSpec",
            )
        try:
            target = Path(path)
            if target.suffix.casefold() != ".vtk":
                raise ValueError(
                    "canonical result VTK target must use the .vtk extension"
                )
            export = self._prepare_result_export(spec)
            completion = GuiCommandCompletion(command_id)

            def workload(context: TaskContext) -> GuiCommandOutcome:
                installed = write_result_vtk(
                    target,
                    export,
                    spec.deformation_scale,
                    checkpoint=context.checkpoint,
                )
                return GuiCommandOutcome(
                    output_path=installed,
                    source=export.source,
                    materialization_generation=(
                        export.materialization_generation
                    ),
                    selection=export.selection,
                    record_count=len(export.field.locations),
                )

            started = self._start_task(
                workload,
                lambda _outcome: self.status_panel.set_state(
                    "VTK 导出完成",
                    5000,
                ),
                "导出 VTK 失败",
                task_name="VTK 导出",
                completion=completion,
            )
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            return self._rejected_command(
                command_id,
                "result.vtk_export.rejected",
                error,
            )
        if not started:
            return self._rejected_command(
                command_id,
                "task.start.rejected",
                "the VTK export task could not be started",
            )
        return GuiCommandReceipt.pending(command_id, completion)

    def _prepare_result_export(
        self,
        spec: ResultCsvExportSpec | ResultVtkExportSpec,
    ) -> ResultExportSnapshot:
        record = self.session.current_result()
        if record is None:
            raise RuntimeError("there is no current accepted result")
        materialization = record.materialization
        if materialization.source != spec.source:
            raise RuntimeError(
                "export source does not match the current accepted result"
            )
        if (
            materialization.generation
            != spec.materialization_generation
        ):
            raise RuntimeError(
                "export generation does not match the current materialization"
            )
        return prepare_result_export_snapshot(
            materialization,
            spec.selection,
        )

    def _apply_session_delta(
        self,
        delta: object,
        *,
        model_geometry: ModelGeometry | None = None,
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
        if (
            snapshot.artifact is None
            or previous_artifact_id
            != snapshot.artifact.artifact_id
        ):
            self._scope_selection_topology_cache = None

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
                if (
                    geometry_dimension(recipe) == 1
                    and self._geometry_selection_mode == "face"
                ):
                    self._selected_geometry_refs.clear()
                    self._geometry_selection_mode = "body"
                    self.actions["geometry_select_body"].setChecked(True)
                    self.actions["geometry_select_face"].setChecked(False)
                    self.viewport.set_selection_mode("geometry_body")
                try:
                    preview = build_geometry_preview(recipe)
                    if (
                        self._sketch_editor_controller is not None
                        or self._wire_editor_controller is not None
                    ):
                        self.viewport.show_geometry_preview(
                            preview,
                            render=False,
                        )
                    else:
                        self.viewport.show_geometry_preview(preview)
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
        if current_result is None:
            self._clear_result_projection()
        elif artifact is not None and self.geometry is not None:
            materialization = current_result.materialization
            provider = self.result_provider
            provider_changed = (
                type(provider) is not ResultProvider
                or provider.source != materialization.source
                or provider.snapshot.generation != materialization.generation
            )
            if provider_changed:
                provider = restore_result_provider(
                    current_result.result,
                    materialization,
                )

            selection_is_current = self._selection_belongs_to_catalog(
                provider,
                self.result_selection,
            )
            viewport_payload = self.viewport._result_render_payload
            viewport_is_current = (
                type(viewport_payload) is ResultRenderPayload
                and viewport_payload.topology.source == provider.source
                and viewport_payload.topology.materialization_generation
                == provider.snapshot.generation
                and viewport_payload.topology.selection
                == self.result_selection
                and self.viewport.run_id == provider.source.run_id
            )
            consumers_are_current = (
                self.result_provider is provider
                and selection_is_current
                and viewport_is_current
                and self.result_tree.catalog is provider.catalog()
                and self.result_tree.has_selection(self.result_selection)
                and (
                    self.inspection_service is None
                    or self.inspection_service.result_provider is provider
                )
            )
            if (
                provider_changed
                or not consumers_are_current
            ):
                self._install_result_provider_projection(provider)

        self._sync_step_combos()
        self._refresh_result_controls()
        self._update_action_states()
        self._applied_session_revision = revision
        return True

    def _apply_revision_neutral_task_receipt(
        self,
        receipt: SessionDelta,
    ) -> bool:
        """Consume an accepted no-state-change task receipt."""

        return not (
            type(receipt) is not SessionDelta
            or not receipt.accepted
            or receipt.changed
            or receipt.invalidated
        )

    def _session_source_label(self) -> str:
        path = self.document.source_path or self.document.project_path
        if path is not None:
            return Path(path).name
        recipe = self.document.geometry_recipe
        if recipe is not None:
            return str(getattr(recipe, "name", "") or "Model-1")
        model = self.document.model
        return str(getattr(model, "name", "") or "模型")

    @staticmethod
    def _session_task_outcome(
        delta: SessionDelta,
        projection_value: object,
    ) -> TaskApplyOutcome:
        if type(delta) is not SessionDelta:
            raise TypeError("Session task acceptance must return SessionDelta")
        if delta.accepted:
            return TaskApplyOutcome.accepted((delta, projection_value))
        status = delta.token_status
        message = delta.reason or (
            status.value if status is not None else "task result was rejected"
        )
        if status in {TokenStatus.WRONG_KIND, TokenStatus.INVALID_STATE}:
            return TaskApplyOutcome.rejected(message)
        return TaskApplyOutcome.stale(message)

    def _clear_model_projection(self) -> None:
        self._close_inspection_windows()
        self._close_job_manager()
        self._pending_analysis_selection = None
        self._pending_scope_kind = None
        self.viewport_panel.scope_creation_bar.finish()
        self.inspection_service = None
        self.geometry = None
        self.result_provider = None
        self.result_selection = None
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
            self.inspection_service.update_result_provider(None)
        had_projection = not (
            self.result_provider is None
            and self.viewport._result_render_payload is None
        )
        self.result_provider = None
        self.result_selection = None
        self._display = DisplayState()
        self.result_tree.clear_result()
        self.navigation.show_model()
        self.status_panel.set_result()
        if not had_projection:
            return
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

    def _install_result_provider_projection(
        self,
        provider: ResultProvider,
    ) -> None:
        """Install one exact provider across the typed GUI consumers."""

        if type(provider) is not ResultProvider:
            raise TypeError("provider must be exactly ResultProvider")
        record = self.session.current_result()
        if (
            record is None
            or provider.source != record.materialization.source
            or provider.snapshot.generation
            != record.materialization.generation
        ):
            raise RuntimeError(
                "provider does not match the current accepted result"
            )
        source = provider.source
        if (
            self.document.artifact is None
            or source.artifact_id
            != self.document.artifact.artifact_id
            or self.geometry is None
            or self.geometry.artifact_id != source.artifact_id
            or self.viewport.artifact_id != source.artifact_id
        ):
            raise RuntimeError(
                "provider projection does not match the current model"
            )

        catalog = provider.catalog()
        selection = (
            self.result_selection
            if (
                type(self.result_provider) is ResultProvider
                and self.result_provider.source == provider.source
                and self._selection_belongs_to_catalog(
                    provider,
                    self.result_selection,
                )
            )
            else catalog.default_selection
        )
        payload = self._build_result_render_payload(
            provider,
            selection,
        )
        step_name = self._current_step_name or self.session.default_step_name()
        previous_catalog = self.result_tree.catalog
        previous_selection = self.result_selection
        inspection = self.inspection_service
        previous_inspection_provider = (
            None if inspection is None else inspection.result_provider
        )
        try:
            self.result_tree.set_catalog(step_name or "", catalog)
            if not self.result_tree.select_selection(selection):
                raise RuntimeError(
                    "installed selection is missing from the result tree"
                )
            if inspection is not None:
                inspection.update_result_provider(provider)
            self._install_viewport_result_payload(
                payload,
                shape_mode=self._display.shape_mode,
                contour_enabled=self._display.contour_enabled,
            )
        except Exception:
            try:
                if previous_catalog is None:
                    self.result_tree.clear_result()
                else:
                    self.result_tree.set_catalog(
                        previous_catalog.source.step_name,
                        previous_catalog,
                    )
                    if type(previous_selection) is ScalarFieldSelection:
                        self.result_tree.select_selection(
                            previous_selection
                        )
            except Exception:
                logging.exception(
                    "failed to restore the previous result tree"
                )
            if (
                inspection is not None
                and inspection is self.inspection_service
            ):
                try:
                    inspection.update_result_provider(
                        previous_inspection_provider
                    )
                except Exception:
                    logging.exception(
                        "failed to restore the previous inspection result"
                    )
            raise
        self.result_provider = provider
        self.result_selection = selection
        self.status_panel.set_result(self._result_status_text())

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
        edit_menu.addActions([self.actions[name] for name in ("select_node", "select_element", "select_edge")])
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
        result_menu.addActions([self.actions[name] for name in ("overlay", "field", "scale", "contour_options", "query", "export_csv", "export_vtk", "screenshot")])
        help_menu = self.menuBar().addMenu("帮助")
        help_menu.setObjectName("menuHelp")
        help_menu.addAction(self.actions["about"])

    def _build_ribbon(self) -> None:
        self.ribbon = RibbonWidget(self)
        self._add_ribbon_page("项目", (
            ("文件", ("new_native", "open_project", "save_project", "open", "reload", "close"), ("new_native", "open")),
            ("信息", ("model_info",), ()),
            ("分析", ("submit_job",), ("submit_job",)),
            ("输出", ("export_csv", "export_vtk"), ()),
        ), step_group="分析")
        self._add_ribbon_page("几何", (
            ("创建", ("geometry_create",), ("geometry_create",)),
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
                "作用域",
                ("geometry_region", "geometry_regions"),
                (),
            ),
            (
                "检查",
                ("mesh_verify", "mesh_statistics"),
                (),
            ),
        ))
        self._add_ribbon_page("模型", (
            ("定义", ("material_manager", "section_manager", "section_assign"), ("material_manager",)),
            ("选择", ("select_node", "select_element", "select_edge", "selected_info"), ()),
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
        self.result_variable_combo = _ExactDataComboBox(field_host)
        self.result_variable_combo.setObjectName("resultVariableCombo")
        self.result_component_combo = _ExactDataComboBox(field_host)
        self.result_component_combo.setObjectName("resultComponentCombo")
        self.result_position_combo = _ExactDataComboBox(field_host)
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
        for name in ("export_csv", "screenshot"):
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
        self.viewport_panel.scope_creation_bar.createRequested.connect(
            self._complete_scope_creation_from_bar
        )
        self.wire_editor_panel = WireEditorPanel(parent=self)
        self.wire_editor_panel.hide()
        self.sketch_editor_panel = SketchEditorPanel(parent=self)
        self.sketch_editor_panel.hide()
        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        splitter.setObjectName("mainSplitter")
        splitter.addWidget(self.navigation)
        splitter.addWidget(self.viewport_panel)
        splitter.addWidget(self.wire_editor_panel)
        splitter.addWidget(self.sketch_editor_panel)
        splitter.setSizes([260, 1020, 0, 0])
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        splitter.setStretchFactor(3, 0)
        splitter.setCollapsible(0, False)
        splitter.setCollapsible(2, True)
        splitter.setCollapsible(3, True)
        self.main_splitter = splitter
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
        self.result_tree.fieldSelectionActivated.connect(
            self._activate_result_selection
        )
        self.viewport.entityPicked.connect(self._on_viewport_pick)
        self.viewport.geometryEntityPicked.connect(
            self._on_geometry_entity_pick
        )
        self.viewport.geometryEntitiesBoxSelected.connect(
            self._on_geometry_entities_box_selected
        )
        self.viewport.meshEntityPicked.connect(
            self._on_mesh_scope_entity_pick
        )
        self.viewport.meshEntitiesBoxSelected.connect(
            self._on_mesh_entities_box_selected
        )
        self.viewport.selectionMissed.connect(self._on_viewport_pick_missed)
        self.viewport.selectionConfirmed.connect(self._confirm_guided_selection)
        self.viewport.selectionCancelled.connect(self._cancel_guided_selection)
        self.wire_editor_panel.finishRequested.connect(self.finish_wire_geometry)
        self.wire_editor_panel.cancelRequested.connect(self.cancel_wire_geometry)
        self.wire_editor_panel.workPlaneChanged.connect(
            self._wire_editor_work_plane_changed
        )
        self.wire_editor_panel.entityFocusRequested.connect(
            self.viewport.focus_wire_draft_entity
        )
        self.sketch_editor_panel.finishRequested.connect(
            self.finish_sketch_geometry
        )
        self.sketch_editor_panel.cancelRequested.connect(
            self.cancel_sketch_geometry
        )
        self.sketch_editor_panel.entityFocusRequested.connect(
            self.viewport.focus_sketch_draft_entity
        )

    def _build_status_bar(self) -> None:
        self.status_panel = CAEStatusBar(self)
        self.status_panel.cancelRequested.connect(self.cancel_current_task)
        self.wire_editor_panel.statusChanged.connect(
            lambda message: self.status_panel.set_state(message, 5000)
        )
        self.sketch_editor_panel.statusChanged.connect(
            lambda message: self.status_panel.set_state(message, 5000)
        )
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

    def _refresh_result_controls(self) -> None:
        provider = self._current_result_provider()
        selection = (
            self.result_selection
            if (
                provider is not None
                and self._selection_belongs_to_catalog(
                    provider,
                    self.result_selection,
                )
            )
            else None
        )
        self.result_variable_combo.blockSignals(True)
        self.result_variable_combo.clear()
        if provider is None or not provider.catalog().fields:
            self.result_variable_combo.addItem("—", None)
            self.result_variable_combo.blockSignals(False)
            self.result_position_combo.blockSignals(True)
            self.result_position_combo.clear()
            self.result_position_combo.addItem("—", None)
            self.result_position_combo.blockSignals(False)
            self.result_component_combo.blockSignals(True)
            self.result_component_combo.clear()
            self.result_component_combo.addItem("—", None)
            self.result_component_combo.blockSignals(False)
            return

        variables: list[ResultVariable] = []
        for availability in provider.catalog().fields:
            if availability.state is FieldState.UNAVAILABLE:
                continue
            variable = availability.descriptor.field_id.variable
            if variable not in variables:
                variables.append(variable)
                self.result_variable_combo.addItem(
                    _RESULT_VARIABLE_LABELS.get(
                        variable,
                        variable.value,
                    ),
                    variable,
                )
        preferred_variable = (
            None
            if selection is None
            else selection.field_key.request.field_id.variable
        )
        variable_index = self.result_variable_combo.findData(
            preferred_variable
        )
        self.result_variable_combo.setCurrentIndex(
            variable_index if variable_index >= 0 else 0
        )
        self.result_variable_combo.blockSignals(False)
        self._populate_result_positions(selection)
        self._populate_result_components(
            preferred_selection=selection,
        )

    def _populate_result_positions(
        self,
        preferred_selection: ScalarFieldSelection | None = None,
    ) -> None:
        self.result_position_combo.blockSignals(True)
        self.result_position_combo.clear()
        provider = self._current_result_provider()
        variable = self.result_variable_combo.currentData()
        if (
            provider is None
            or type(variable) is not ResultVariable
        ):
            self.result_position_combo.addItem("—", None)
            self.result_position_combo.blockSignals(False)
            return

        positions: list[FieldPosition] = []
        for availability in provider.catalog().fields:
            if availability.state is FieldState.UNAVAILABLE:
                continue
            field_id = availability.descriptor.field_id
            if (
                field_id.variable is variable
                and field_id.position not in positions
            ):
                positions.append(field_id.position)
                self.result_position_combo.addItem(
                    _RESULT_POSITION_LABELS.get(
                        field_id.position,
                        field_id.position.value,
                    ),
                    field_id.position,
                )
        preferred_position = None
        if (
            type(preferred_selection) is ScalarFieldSelection
            and preferred_selection.field_key.request.field_id.variable
            is variable
        ):
            preferred_position = (
                preferred_selection.field_key.request.field_id.position
            )
        position_index = self.result_position_combo.findData(
            preferred_position
        )
        self.result_position_combo.setCurrentIndex(
            position_index if position_index >= 0 else 0
        )
        self.result_position_combo.blockSignals(False)

    def _populate_result_components(
        self,
        *,
        preferred_selection: ScalarFieldSelection | None = None,
        preferred_component: str | None = None,
    ) -> None:
        self.result_component_combo.blockSignals(True)
        self.result_component_combo.clear()
        provider = self._current_result_provider()
        variable = self.result_variable_combo.currentData()
        position = self.result_position_combo.currentData()
        if (
            provider is None
            or type(variable) is not ResultVariable
            or type(position) is not FieldPosition
        ):
            self.result_component_combo.addItem("—", None)
            self.result_component_combo.blockSignals(False)
            return

        availabilities = tuple(
            availability
            for availability in provider.catalog().fields
            if (
                availability.state is not FieldState.UNAVAILABLE
                and availability.descriptor.field_id.variable is variable
                and availability.descriptor.field_id.position is position
            )
        )
        component_counts: dict[str, int] = {}
        for availability in availabilities:
            for component in availability.descriptor.columns:
                component_counts[component] = (
                    component_counts.get(component, 0) + 1
                )
        fallback_index = -1
        for availability in availabilities:
            for component in availability.descriptor.columns:
                selection = ScalarFieldSelection(
                    availability.key,
                    component,
                )
                label = component
                if component_counts[component] > 1:
                    label = (
                        f"{label} · contract "
                        f"{availability.key.recovery_contract}"
                    )
                state_label = _RESULT_FIELD_STATE_LABELS.get(
                    availability.state
                )
                if state_label is not None:
                    label = f"{label}（{state_label}）"
                self.result_component_combo.addItem(label, selection)
                if (
                    fallback_index < 0
                    and component == preferred_component
                ):
                    fallback_index = (
                        self.result_component_combo.count() - 1
                    )

        selection_index = self.result_component_combo.findData(
            preferred_selection
        )
        if selection_index < 0:
            selection_index = fallback_index
        self.result_component_combo.setCurrentIndex(
            selection_index if selection_index >= 0 else 0
        )
        self.result_component_combo.blockSignals(False)

    def _result_variable_changed(self, _index: int) -> None:
        current = self.result_selection
        self._populate_result_positions()
        self._populate_result_components(
            preferred_component=(
                None
                if current is None
                else current.component
            )
        )
        self._result_component_changed(
            self.result_component_combo.currentIndex()
        )

    def _result_position_changed(self, _index: int) -> None:
        current = self.result_component_combo.currentData()
        self._populate_result_components(
            preferred_component=(
                current.component
                if type(current) is ScalarFieldSelection
                else None
            )
        )
        self._result_component_changed(
            self.result_component_combo.currentIndex()
        )

    def _result_component_changed(self, _index: int) -> None:
        selection = self.result_component_combo.currentData()
        if type(selection) is not ScalarFieldSelection:
            return
        self._activate_result_selection(selection)

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
        authoring = describe_session_authoring(self.document)
        selection_kind = (
            "node"
            if self.selection.node_id is not None
            else "element"
            if self.selection.element_id is not None
            else None
        )
        provider = self._current_result_provider()
        selected_availability: FieldAvailability | None = None
        action_selection = (
            self._pending_result_selection
            if (
                self._pending_result_selection is not None
                and provider is not None
                and provider.source == self._pending_result_source
            )
            else self.result_selection
        )
        if provider is not None and type(
            action_selection
        ) is ScalarFieldSelection:
            try:
                selected_availability = (
                    self._catalog_availability_for_selection(
                        provider,
                        action_selection,
                    )
                )
            except (KeyError, TypeError, ValueError):
                selected_availability = None
        context = GuiActionContext(
            busy=self.busy,
            selected_step_name=self._current_step_name,
            geometry_selection=tuple(
                sorted(
                    self._selected_geometry_refs,
                    key=logical_ref_sort_key,
                )
            ),
            fem_selection_kind=selection_kind,
            display_backend_available=self.viewport.can_capture,
            open_dialog_keys=(
                frozenset({"job_manager"})
                if self._job_manager is not None
                else frozenset()
            ),
            result_source_current=provider is not None,
            catalog_available=(
                provider is not None
                and bool(provider.catalog().fields)
            ),
            selected_field_exists=selected_availability is not None,
            selected_field_state=(
                None
                if selected_availability is None
                else selected_availability.state
            ),
            materialization_pending=(
                self._pending_result_selection is not None
                or self._pending_result_query is not None
            ),
            result_task_busy=(
                self._pending_result_selection is not None
                or self._pending_result_query is not None
            ),
            viewport_scene_available=(
                provider is not None
                and self.viewport.run_id == provider.source.run_id
            ),
            wire_editor_active=self._wire_editor_controller is not None,
            sketch_editor_active=self._sketch_editor_controller is not None,
        )
        for availability in derive_action_availability(
            self.document,
            authoring,
            context,
        ):
            self._set_action_available(
                availability.key.value,
                availability.enabled,
                availability.reason,
            )
        has_result = provider is not None
        self.result_variable_combo.setEnabled(has_result and not self.busy)
        self.result_component_combo.setEnabled(has_result and not self.busy)
        self.result_position_combo.setEnabled(has_result and not self.busy)
        self.result_scale_combo.setEnabled(has_result)
        self.result_scale_value.setEnabled(
            has_result and self._scale_mode == "custom"
        )
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
                self.document.project_path.name
                if self.document.project_path is not None
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

    def start_wire_geometry(self) -> None:
        """Start a detached Wire draft from the Geometry ribbon."""

        if (
            self.busy
            or self._wire_editor_controller is not None
            or self._sketch_editor_controller is not None
        ):
            return
        if self.document.source_kind != "native":
            self._show_error(
                "新建线体",
                "请先新建自主模型，再创建线体。",
            )
            return
        current = self.document.geometry_recipe
        if current is not None and not self._confirm_wire_replacement():
            return
        self._begin_wire_editor(
            None,
            original_recipe=None,
        )

    def _begin_wire_editor(
        self,
        root: WireGeometry | None,
        *,
        original_recipe: object | None,
    ) -> None:
        if (
            self._wire_editor_controller is not None
            or self._sketch_editor_controller is not None
        ):
            return
        controller = (
            WireDraftController(root=root)
            if root is not None
            else WireDraftController(name="线体-1")
        )
        self._wire_editor_controller = controller
        self._wire_editor_original_recipe = original_recipe
        self._wire_editor_base_revision = self.document.session_revision
        self.wire_editor_panel.set_controller(
            controller,
            base_snapshot=controller.snapshot(),
        )
        self.wire_editor_panel.begin(self.viewport)
        self.main_splitter.setSizes([260, 760, 360, 0])
        self._wire_editor_work_plane_changed(
            str(self.wire_editor_panel.work_plane_combo.currentData())
        )
        self.ribbon.set_current("几何")
        self.status_panel.set_state(
            "线体编辑已启动，请在视图区添加点并连接杆件",
            0,
        )
        self._update_action_states()

    def _confirm_wire_replacement(self) -> bool:
        if self.document.geometry_recipe is None:
            return True
        answer = QMessageBox.question(
            self,
            "替换几何",
            "创建线体会替换当前自主几何，是否继续？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        return answer == QMessageBox.StandardButton.Yes

    def finish_wire_geometry(self) -> None:
        controller = self._wire_editor_controller
        base_revision = self._wire_editor_base_revision
        if controller is None or base_revision is None:
            return
        try:
            root = controller.to_geometry()
        except WireDraftValidationError as error:
            self.wire_editor_panel.show_status(str(error))
            return
        original = self._wire_editor_original_recipe
        recipe = (
            self._replace_root_geometry(original, root)
            if original is not None
            else root
        )
        receipt = self.apply_native_geometry_edit(
            NativeGeometryEdit(
                base_session_revision=base_revision,
                parts=tuple(self.document.parts) or (NativePart(),),
                recipe=recipe,
                mesh_settings=UNSET,
            )
        )
        if receipt.diagnostic is not None:
            self.wire_editor_panel.show_status(
                f"{receipt.diagnostic.code}: {receipt.diagnostic.message}"
            )
            return
        self._exit_wire_editor()
        self.status_panel.set_state(
            "线体几何已创建，请在网格设置中选择桁架或梁单元",
            6000,
        )
        self.ribbon.set_current("几何")

    def cancel_wire_geometry(self) -> None:
        controller = self._wire_editor_controller
        if controller is None:
            return
        if controller.dirty and not self._confirm_wire_editor_discard():
            return
        self._exit_wire_editor()
        self._rebuild_full_projection()
        self.status_panel.set_state("已取消线体编辑", 4000)

    def _exit_wire_editor(self) -> None:
        self.wire_editor_panel.end()
        self._wire_editor_controller = None
        self._wire_editor_original_recipe = None
        self._wire_editor_base_revision = None
        self.main_splitter.setSizes([260, 1020, 0, 0])
        self._update_action_states()
        self._schedule_viewport_fit()

    def _confirm_wire_editor_discard(self) -> bool:
        answer = QMessageBox.question(
            self,
            "放弃线体草图",
            "线体草图包含未保存的修改，是否放弃这些修改？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        return answer == QMessageBox.StandardButton.Yes

    def _wire_editor_work_plane_changed(self, plane: str) -> None:
        views = {"XY": "top", "XZ": "front", "YZ": "left"}
        view = views.get(str(plane).upper())
        if view is not None:
            self.viewport.set_view(view)

    def create_geometry(self) -> None:
        """Open the single 1D/2D/3D geometry creation entry point."""

        if (
            self.busy
            or self._wire_editor_controller is not None
            or self._sketch_editor_controller is not None
        ):
            return
        if self.document.source_kind != "native":
            self._show_error(
                "创建草图",
                "请先新建自主模型；INP 模型不能反向转换为可编辑 CAD。",
            )
            return
        dialog = GeometryCreationDialog(self)
        if not self._exec_dialog(dialog):
            return
        creation_kind = dialog.creation_kind()
        if creation_kind == "3d":
            solid_dialog = BasicSolidCreationDialog(self)
            if not self._exec_dialog(solid_dialog):
                return
            creation_kind = f"3d_{solid_dialog.solid_kind()}"
        handlers = {
            "1d": self.start_wire_geometry,
            "2d": self.start_sketch_geometry,
            "3d_box": self.create_box_geometry,
            "3d_cylinder": self.create_cylinder_geometry,
        }
        handler = handlers.get(creation_kind)
        if handler is None:
            raise RuntimeError("geometry creation dialog returned an unknown kind")
        handler()

    def create_sketch_geometry(self) -> None:
        """Compatibility action: enter the interactive 2D sketch workflow."""

        self.start_sketch_geometry()

    def start_sketch_geometry(self) -> None:
        if (
            self.busy
            or self._wire_editor_controller is not None
            or self._sketch_editor_controller is not None
        ):
            return
        if self.document.source_kind != "native":
            self._show_error(
                "新建二维草图",
                "请先新建自主模型，再创建二维草图。",
            )
            return
        self._begin_sketch_editor(None, original_recipe=None)

    def _begin_sketch_editor(
        self,
        root: SketchGeometry | None,
        *,
        original_recipe: object | None,
    ) -> None:
        if (
            self._wire_editor_controller is not None
            or self._sketch_editor_controller is not None
        ):
            return
        controller = (
            SketchDraftController(root=root)
            if root is not None
            else SketchDraftController(name="草图-1")
        )
        self._sketch_editor_controller = controller
        self._sketch_editor_original_recipe = original_recipe
        self._sketch_editor_base_revision = self.document.session_revision
        self.sketch_editor_panel.set_controller(
            controller,
            base_snapshot=controller.snapshot(),
        )
        self.sketch_editor_panel.begin(self.viewport)
        self.main_splitter.setSizes([260, 720, 0, 400])
        self.ribbon.set_current("几何")
        self.viewport.set_view("top")
        self.status_panel.set_state(
            "二维草图编辑已启动，请在 XY 工作平面绘制闭合轮廓",
            0,
        )
        self._update_action_states()

    def _confirm_sketch_replacement(self) -> bool:
        if self.document.geometry_recipe is None:
            return True
        answer = QMessageBox.question(
            self,
            "替换几何",
            "完成草图将替换当前自主几何，是否提交？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        return answer == QMessageBox.StandardButton.Yes

    def finish_sketch_geometry(self) -> None:
        controller = self._sketch_editor_controller
        base_revision = self._sketch_editor_base_revision
        if controller is None or base_revision is None:
            return
        try:
            root = controller.to_sketch_geometry()
        except SketchDraftValidationError as error:
            self.sketch_editor_panel.show_status(str(error))
            return
        if (
            self.document.geometry_recipe is not None
            and not self._confirm_sketch_replacement()
        ):
            self.sketch_editor_panel.show_status(
                "已保留当前草稿；可继续编辑后再次完成"
            )
            return
        original = self._sketch_editor_original_recipe
        recipe = (
            self._replace_root_geometry(original, root)
            if original is not None
            else root
        )
        receipt = self.apply_native_geometry_edit(
            NativeGeometryEdit(
                base_session_revision=base_revision,
                parts=tuple(self.document.parts) or (NativePart(),),
                recipe=recipe,
                mesh_settings=UNSET,
            )
        )
        if receipt.diagnostic is not None:
            self.sketch_editor_panel.show_status(
                f"{receipt.diagnostic.code}: {receipt.diagnostic.message}"
            )
            return
        self._exit_sketch_editor()
        self.status_panel.set_state(
            "二维草图已创建；可继续拉伸、布尔运算或设置网格",
            6000,
        )
        self.ribbon.set_current("几何")

    def cancel_sketch_geometry(self) -> None:
        controller = self._sketch_editor_controller
        if controller is None:
            return
        if controller.dirty and not self._confirm_sketch_editor_discard():
            return
        self._exit_sketch_editor()
        self._rebuild_full_projection()
        self.status_panel.set_state("已取消二维草图编辑", 4000)

    def _exit_sketch_editor(self) -> None:
        self.sketch_editor_panel.end()
        self._sketch_editor_controller = None
        self._sketch_editor_original_recipe = None
        self._sketch_editor_base_revision = None
        self.main_splitter.setSizes([260, 1020, 0, 0])
        self._update_action_states()
        self._schedule_viewport_fit()

    def _confirm_sketch_editor_discard(self) -> bool:
        answer = QMessageBox.question(
            self,
            "放弃二维草图",
            "二维草图包含未保存的修改，是否放弃这些修改？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        return answer == QMessageBox.StandardButton.Yes

    def create_rectangle_geometry(self) -> None:
        current = self.document.geometry_recipe
        dialog = RectangleGeometryDialog(
            current if isinstance(current, RectangleGeometry) else None,
            self,
        )
        if not self._exec_dialog(dialog):
            return
        self._set_native_geometry(dialog.recipe(), "矩形")

    def create_disk_geometry(self) -> None:
        current = self.document.geometry_recipe
        dialog = DiskGeometryDialog(
            current if isinstance(current, DiskGeometry) else None,
            self,
        )
        if self._exec_dialog(dialog):
            self._set_native_geometry(dialog.recipe(), "圆盘")

    def create_box_geometry(self) -> None:
        current = self.document.geometry_recipe
        dialog = BoxGeometryDialog(
            current if isinstance(current, BoxGeometry) else None,
            self,
        )
        if self._exec_dialog(dialog):
            self._set_native_geometry(dialog.recipe(), "长方体")

    def create_cylinder_geometry(self) -> None:
        current = self.document.geometry_recipe
        dialog = CylinderGeometryDialog(
            current if isinstance(current, CylinderGeometry) else None,
            self,
        )
        if self._exec_dialog(dialog):
            self._set_native_geometry(dialog.recipe(), "圆柱")

    def create_plate_with_hole_geometry(self) -> None:
        current = self.document.geometry_recipe
        dialog = PlateWithHoleGeometryDialog(
            current if isinstance(current, PlateWithHoleGeometry) else None,
            self,
        )
        if not self._exec_dialog(dialog):
            return
        self._set_native_geometry(dialog.recipe(), "带圆孔矩形板")

    def move_geometry(self) -> None:
        current = self.document.geometry_recipe
        if not isinstance(current, NATIVE_GEOMETRY_TYPES):
            return
        dialog = MoveGeometryDialog(
            current,
            self,
            is_3d=geometry_dimension(current) != 2,
        )
        if self._exec_dialog(dialog):
            self._set_native_geometry(dialog.recipe(), "移动后的")

    def rotate_geometry(self) -> None:
        current = self.document.geometry_recipe
        if not isinstance(current, NATIVE_GEOMETRY_TYPES):
            return
        dialog = RotateGeometryDialog(
            current,
            self,
            is_3d=geometry_dimension(current) != 2,
        )
        if self._exec_dialog(dialog):
            self._set_native_geometry(dialog.recipe(), "旋转后的")

    def extrude_geometry(self) -> None:
        current = self.document.geometry_recipe
        if (
            not isinstance(current, NATIVE_GEOMETRY_TYPES)
            or geometry_dimension(current) != 2
        ):
            return
        dialog = ExtrudeGeometryDialog(current, self)
        if self._exec_dialog(dialog):
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
            if current.is_strict:
                self._begin_sketch_editor(
                    current,
                    original_recipe=current,
                )
                action = "孔轮廓" if operation == "cut" else "材料轮廓"
                self.status_panel.set_state(
                    f"请绘制新的闭合{action}；轮廓嵌套关系将在完成草图时解析",
                    0,
                )
                return
            dialog = SketchGeometryDialog(
                current,
                self,
                new_contour_operation=(
                    "cut" if operation == "cut" else "material"
                ),
            )
            if self._exec_dialog(dialog):
                self._set_native_geometry(dialog.recipe(), label)
            return
        dialog = BooleanGeometryDialog(
            current,
            operation,
            self,
            is_3d=geometry_dimension(current) == 3,
        )
        if self._exec_dialog(dialog):
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
            self._begin_sketch_editor(current, original_recipe=current)
            return
        root = self._root_geometry(current)
        dialog = GeometryManagerDialog(
            current,
            self,
            can_edit_base=isinstance(root, (SketchGeometry, WireGeometry)),
            base_label=(
                "Edit base geometry"
                if isinstance(root, WireGeometry)
                else "编辑基础草图"
            ),
        )
        if not self._exec_dialog(dialog):
            return
        if dialog.operation == "edit" and isinstance(root, SketchGeometry):
            self._begin_sketch_editor(root, original_recipe=current)
        elif dialog.operation == "edit" and isinstance(root, WireGeometry):
            self._begin_wire_editor(root, original_recipe=current)
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
        receipt = self.apply_native_geometry_edit(
            NativeGeometryEdit(
                base_session_revision=self.document.session_revision,
                parts=(),
                recipe=None,
            )
        )
        if receipt.diagnostic is not None:
            self._show_command_rejection("删除几何", receipt)
            return
        self._selected_geometry_refs.clear()
        self._selected_mesh_scope_refs.clear()
        self.viewport_panel.set_geometry_context(False)
        self.status_panel.set_state("当前几何已删除", 5000)

    def _set_native_geometry(self, recipe: object, label: str) -> None:
        if not isinstance(recipe, NATIVE_GEOMETRY_TYPES):
            raise TypeError(f"不支持的几何定义：{type(recipe).__name__}")
        if self.document.source_kind != "native":
            new_receipt = self.new_native_project(NewNativeProjectCommand())
            if new_receipt.diagnostic is not None:
                self._show_command_rejection("新建项目", new_receipt)
                return
        prior_region_count = len(self.document.named_regions)
        receipt = self.apply_native_geometry_edit(
            NativeGeometryEdit(
                base_session_revision=self.document.session_revision,
                parts=tuple(self.document.parts) or (NativePart(),),
                recipe=recipe,
                mesh_settings=UNSET,
            )
        )
        if receipt.diagnostic is not None:
            self._show_command_rejection("编辑几何", receipt)
            return
        delta = receipt.delta
        if delta is None:
            raise RuntimeError("geometry command did not return a Session delta")
        self._pending_local_mesh_selection = False
        self._pending_analysis_selection = None
        self._selected_geometry_refs.clear()
        self._selected_mesh_scope_refs.clear()
        self._geometry_selection_mode = "body"
        self.actions["geometry_select_body"].setChecked(True)
        self.viewport.set_selection_mode("geometry_body")
        self.status_panel.set_selection_mode("geometry_body")
        self.viewport_panel.set_geometry_context(True)
        message = (
            f"{label}几何已创建；网格、模型和结果已标记过期，"
            "请进入网格模块生成网格"
        )
        if TransitionEffect.NAMED_REGIONS_CLEARED in delta.effects:
            message += (
                f"；{prior_region_count} 个旧作用域已失效，"
                "请重新创建作用域"
            )
        if TransitionEffect.LOCAL_CONTROLS_CLEARED in delta.effects:
            message += "；旧局部网格设置已失效"
        if delta.effects & {
            TransitionEffect.ASSIGNMENTS_CLEARED,
            TransitionEffect.STEPS_CLEARED,
        }:
            message += "；依赖旧拓扑的区域分配和分析步已失效"
        if TransitionEffect.MESH_SHAPE_NORMALIZED in delta.effects:
            message += "；网格单元形状已按新几何规范化"
        if TransitionEffect.REFERENCES_PRESERVED in delta.effects:
            message += "；逻辑拓扑未变化，已有拓扑引用已保留"
        self.status_panel.set_state(message, 6000)

    def create_named_geometry_region(self) -> None:
        if self.document.model is None:
            return
        if self._pending_analysis_selection == "scope":
            self._show_scope_creation_bar(
                self._pending_scope_kind or "node"
            )
            return
        kind = self._choose_mesh_scope_kind()
        if kind is not None:
            self._request_analysis_geometry_selection("scope", kind)

    def _choose_mesh_scope_kind(self) -> str | None:
        if self.document.model is None:
            return None
        topology = self._scope_selection_topology()
        available_kinds = {
            reference.kind
            for reference in topology.mesh_references
        }
        kinds = [
            ("Set", "node"),
            *((("Edge", "edge"),) if "edge" in available_kinds else ()),
            *((("Surface", "face"),) if "face" in available_kinds else ()),
            *((("Volume", "body"),) if "body" in available_kinds else ()),
        ]
        labels = tuple(label for label, _kind in kinds)
        selected, accepted = QInputDialog.getItem(
            self,
            "创建作用域",
            "作用域类型",
            labels,
            0,
            False,
        )
        self._schedule_viewport_fit()
        if not accepted:
            return None
        return dict(kinds).get(str(selected))

    def _scope_selection_topology(self) -> ScopeSelectionTopology:
        model = self.document.model
        if model is None:
            raise RuntimeError("scope selection requires a generated mesh")
        if self._scope_selection_topology_cache is None:
            self._scope_selection_topology_cache = (
                build_scope_selection_topology(
                    model,
                    self.document.geometry_recipe,
                )
            )
        return self._scope_selection_topology_cache

    def _start_edge_scope_selection(self) -> None:
        if self.document.model is None:
            return
        self._request_analysis_geometry_selection("scope", "edge")

    def _create_region_from_current_mesh_selection(
        self,
        *,
        requested_name: str | None = None,
    ) -> str | None:
        references = self._canonical_mesh_scope_selection()
        if not references:
            return None
        kind = references[0].kind
        for region in self.document.named_regions.values():
            if region.references == references:
                return region.name
        if requested_name is None:
            dialog = NamedRegionDialog(
                references,
                self,
                suggested_name=self._next_named_region_name(kind),
            )
            if not self._exec_dialog(dialog):
                return None
            try:
                name = dialog.region_name()
            except ValueError as error:
                self._show_error("创建作用域", str(error))
                return None
        else:
            name = str(requested_name).strip()
            if not name:
                self._show_error("创建作用域", "作用域名称不能为空")
                return None
        if name in self.document.named_regions:
            self._show_error("创建作用域", f"作用域名称已存在：{name}")
            return None
        regions = dict(self.document.named_regions)
        base_revision = self.document.session_revision
        try:
            regions[name] = NamedRegion(name, references)
            batch = NamedRegionEditBatch(
                base_session_revision=base_revision,
                regions=tuple(regions.values()),
            )
        except (TypeError, ValueError) as error:
            self._show_error("创建作用域", str(error))
            return None
        receipt = self.apply_named_region_edit(batch)
        if receipt.diagnostic is not None:
            self._show_command_rejection("创建作用域", receipt)
            return None
        self.status_panel.set_state(
            f"已创建作用域 {name}",
            5000,
        )
        self._update_action_states()
        return name

    def _next_named_region_name(self, kind: str) -> str:
        if (
            isinstance(self.document.geometry_recipe, NATIVE_GEOMETRY_TYPES)
            and geometry_dimension(self.document.geometry_recipe) == 1
        ):
            prefixes = {
                "node": "NodeSet",
                "edge": "EdgeSet",
                "element": "ElementSet",
            }
            prefix = prefixes.get(kind, "Region")
            existing = {name.casefold() for name in self.document.named_regions}
            index = 1
            while f"{prefix}-{index}".casefold() in existing:
                index += 1
            return f"{prefix}-{index}"
        prefixes = {
            "node": "NodeSet",
            "edge": "EdgeSet",
            "face": "Surface",
            "body": "Volume",
            "element": "ElementSet",
        }
        prefix = prefixes.get(kind, "Region")
        existing = {
            name.casefold() for name in self.document.named_regions
        }
        index = 1
        while f"{prefix}-{index}".casefold() in existing:
            index += 1
        return f"{prefix}-{index}"

    def show_named_region_manager(self) -> None:
        if not self.document.named_regions:
            return
        base_revision = self.document.session_revision
        dialog = NamedRegionManagerDialog(
            dict(self.document.named_regions),
            self,
        )
        if not self._exec_dialog(dialog):
            return
        updated = dialog.values()
        try:
            batch = NamedRegionEditBatch(
                base_session_revision=base_revision,
                regions=tuple(updated.values()),
                renames=dialog.rename_intents(),
                deletes=dialog.delete_intents(),
            )
        except (TypeError, ValueError) as error:
            self._show_error("作用域管理", str(error))
            return
        receipt = self.apply_named_region_edit(batch)
        if receipt.diagnostic is not None:
            self._show_command_rejection("作用域管理", receipt)
            return
        self.status_panel.set_state(
            "作用域已更新",
            5000,
        )

    def _scope_authoring_targets(
        self,
        authoring: Any | None = None,
    ) -> tuple[Any, ...]:
        projection = (
            authoring
            if authoring is not None
            else describe_session_authoring(self.document)
        )
        targets = projection.targets
        if self.document.source_kind != "native":
            return targets
        authored_names = frozenset(self.document.named_regions)
        return tuple(
            target
            for target in targets
            if target.region.name in authored_names
        )

    def _analysis_region_names(
        self,
    ) -> tuple[list[RegionRef], list[RegionRef], list[RegionRef]]:
        authoring = describe_session_authoring(self.document)
        targets = self._scope_authoring_targets(authoring)
        return tuple(
            [target.region for target in targets if target.region.kind == kind]
            for kind in ("node_set", "edge", "surface")
        )

    def _analysis_element_regions(
        self,
        capability_report: ModelCapabilityReport | None = None,
    ) -> list[RegionRef]:
        del capability_report
        authoring = describe_session_authoring(self.document)
        return [
            target.region
            for target in self._scope_authoring_targets(authoring)
            if target.region.kind == "element_set"
        ]

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
                named_regions=tuple(self.document.named_regions.values()),
            )
        return None

    def _supported_load_regions(
        self,
    ) -> tuple[
        list[RegionRef],
        list[RegionRef],
        list[RegionRef],
        list[RegionRef],
    ]:
        """Return targets filtered by the application capability report."""

        authoring = describe_session_authoring(self.document)
        targets = self._scope_authoring_targets(authoring)
        operations = (
            ("node_set", "load.node"),
            ("edge", "load.edge"),
            ("surface", "load.surface"),
            ("element_set", "load.line.global"),
        )
        return tuple(
            [
                target.region
                for target in targets
                if target.region.kind == kind
                and target.operation(operation).can_submit
            ]
            for kind, operation in operations
        )

    def _request_analysis_geometry_selection(
        self,
        operation: str,
        scope_kind: str | None = None,
    ) -> None:
        if self.document.model is None:
            return
        self._pending_analysis_selection = operation
        topology = self._scope_selection_topology()
        dimension = topology.preview.topological_dimension
        domain_kind = {
            1: "edge",
            2: "face",
            3: "body",
        }[dimension]
        requested_kind = {
            "node": "node",
            "edge": "edge",
            "surface": "face",
            "face": "face",
            "body": "body",
            "volume": "body",
            "line": domain_kind,
            "element": "element",
            "element_set": domain_kind,
        }.get(str(scope_kind or ""))
        semantic_kinds = {
            reference.kind
            for reference in topology.mesh_references
        }
        default_kind = requested_kind or (
            domain_kind if domain_kind in semantic_kinds else "node"
        )
        available = {"node", "element"} | semantic_kinds
        if default_kind not in available:
            self._show_error(
                "创建作用域",
                f"当前网格不支持 {default_kind} 作用域",
            )
            self._pending_analysis_selection = None
            return
        self._pending_scope_kind = default_kind
        self.viewport_panel.set_geometry_context(False)
        if default_kind in {"edge", "face", "body"}:
            self._selected_geometry_refs.clear()
            self._selected_mesh_scope_refs.clear()
            self.viewport.show_geometry_preview(
                topology.preview,
                preserve_model=True,
            )
            self._scope_selection_overlay_active = True
            self._set_geometry_selection_mode(default_kind)
        else:
            self._selected_geometry_refs.clear()
            self._selected_mesh_scope_refs.clear()
            self._set_mesh_scope_selection_mode(default_kind)
        self._show_scope_creation_bar(default_kind)
        label = {
            "boundary": "边界条件",
            "load": "载荷",
            "scope": "作用域",
            "section": "截面分配",
        }.get(operation, "作用域")
        kind_label = {
            "node": "节点",
            "edge": "边",
            "face": "面",
            "body": "体",
            "element": "单元",
        }[default_kind]
        self.status_panel.set_state(
            f"请选择用于{label}的{kind_label}；可点选或框选，"
            "Ctrl 多选，使用视图底部的“创建”完成，Esc 取消",
            0,
        )

    def _show_scope_creation_bar(self, semantic_kind: str) -> None:
        type_label = {
            "node": "Set",
            "edge": "Edge",
            "face": "Surface",
            "body": "Volume",
            "element": "Set",
        }[semantic_kind]
        bar = self.viewport_panel.scope_creation_bar
        bar.begin(
            type_label,
            self._next_named_region_name(semantic_kind),
        )
        bar.set_selection_ready(
            bool(self._canonical_mesh_scope_selection())
        )

    def _complete_scope_creation_from_bar(self) -> None:
        if self._pending_analysis_selection is None:
            return
        if not self._canonical_mesh_scope_selection():
            self.status_panel.set_state("请先选择至少一个对象", 3000)
            return
        bar = self.viewport_panel.scope_creation_bar
        name = self._create_region_from_current_mesh_selection(
            requested_name=bar.scope_name(),
        )
        if name is None:
            return
        operation = self._pending_analysis_selection
        self._pending_analysis_selection = None
        self._pending_scope_kind = None
        if self._scope_selection_overlay_active:
            self.viewport.hide_geometry_selection_overlay()
            self._scope_selection_overlay_active = False
        bar.finish()
        callback = {
            "boundary": self.create_displacement_boundary,
            "load": self.create_load,
            "section": self.assign_section_to_region,
        }.get(operation)
        if callback is not None:
            QTimer.singleShot(0, callback)
        elif operation != "scope":
            raise RuntimeError(
                f"unsupported guided scope operation: {operation}"
            )

    def edit_mesh_settings(self) -> None:
        recipe = self.document.geometry_recipe
        if not isinstance(recipe, NATIVE_GEOMETRY_TYPES):
            return
        current = self.document.mesh_settings
        if not isinstance(current, MeshSettings) and geometry_dimension(recipe) != 1:
            current = MeshSettings(
                recipe_characteristic_size(recipe) / 10.0,
                cell_shape="tetrahedron" if geometry_dimension(recipe) == 3 else "triangle",
            )
        dialog = MeshSettingsDialog(
            current,
            self,
            mesh_dimension=geometry_dimension(recipe),
            allow_hexahedron=supports_structured_hexahedron(recipe),
            suggested_size=recipe_characteristic_size(recipe) / 10.0,
        )
        if not self._exec_dialog(dialog):
            return
        receipt = self.apply_mesh_input_edit(
            MeshInputEdit(
                self.document.session_revision,
                dialog.settings(),
            )
        )
        if receipt.diagnostic is not None:
            self._show_command_rejection("网格设置", receipt)
            return
        self.status_panel.set_state("网格设置已更新，请生成网格", 5000)

    def show_mesh_controls(self) -> None:
        settings = self.document.mesh_settings
        if not isinstance(settings, MeshSettings):
            return
        dialog = MeshControlsDialog(settings, self)
        if not self._exec_dialog(dialog):
            return
        updated = dialog.settings()
        if updated == settings:
            return
        receipt = self.apply_mesh_input_edit(
            MeshInputEdit(self.document.session_revision, updated)
        )
        if receipt.diagnostic is not None:
            self._show_command_rejection("网格控制", receipt)
            return
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
        local_control_action = self.actions["mesh_local_control"]
        if not local_control_action.isEnabled():
            self.status_panel.set_state(
                local_control_action.statusTip(),
                6000,
            )
            return
        is_wire = geometry_dimension(recipe) == 1
        supported_kinds = {"point", "edge"} if is_wire else {"point", "edge", "face"}
        selected_references = self._canonical_geometry_selection()
        if (
            not selected_references
            or selected_references[0].kind not in supported_kinds
            or (is_wire and len(selected_references) != 1)
        ):
            self._pending_local_mesh_selection = True
            self.viewport_panel.set_geometry_context(True)
            default_kind = (
                "face"
                if geometry_dimension(recipe) == 3
                else "point"
                if is_wire
                else "edge"
            )
            self.actions[f"geometry_select_{default_kind}"].setChecked(True)
            self._set_geometry_selection_mode(default_kind)
            self.status_panel.set_state(
                f"请选择需要设置局部网格的{'面' if default_kind == 'face' else '边'}；"
                "Ctrl 多选，Enter 完成，Esc 取消",
                0,
            )
            return
        dialog = LocalMeshControlDialog(
            selected_references[0],
            settings.size,
            self,
        )
        if not self._exec_dialog(dialog):
            return
        control = dialog.control()
        selected_references = (
            selected_references
            if selected_references[0].kind == control.target.kind
            else (control.target,)
        )
        controls = tuple(
            item
            for item in settings.local_controls
            if not (
                item.target in selected_references
                and item.falloff == control.falloff
            )
        ) + tuple(
            replace(control, target=reference)
            for reference in selected_references
        )
        receipt = self.apply_mesh_input_edit(
            MeshInputEdit(
                self.document.session_revision,
                replace(settings, local_controls=controls),
            )
        )
        if receipt.diagnostic is not None:
            self._show_command_rejection("局部网格控制", receipt)
            return
        kind_name = {
            "point": "点",
            "edge": "边",
            "face": "面",
        }.get(control.target.kind, "实体")
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
        receipt = self.clear_generated_mesh(self.document.session_revision)
        if receipt.diagnostic is not None:
            self._show_command_rejection("清除网格", receipt)
            return
        self._selected_geometry_refs.clear()
        self._selected_mesh_scope_refs.clear()
        self.setWindowTitle(f"有限元分析 — {recipe.name}（几何）")
        self.status_panel.set_state(
            "网格已清除；旧作用域及其依赖定义已失效",
            5000,
        )

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
            report = analyze_mesh(model.mesh)
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
            self._show_information("网格统计", [
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
            self._show_information("网格质量检查", [
                ("指标", "归一化形状质量（1 为理想，0 为退化）"),
                ("已检查单元", f"{report.checked_count} / {report.element_count}"),
                ("最小值", f"{report.minimum:.6f}"),
                ("平均值", f"{report.mean:.6f}"),
                ("最大值", f"{report.maximum:.6f}"),
                ("最差单元", worst),
            ])
            return
        self._show_information("检查网格", [
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
        receipt = self.generate_mesh()
        if receipt.diagnostic is not None:
            self._show_command_rejection("网格生成失败", receipt)

    def _begin_mesh_generation(
        self,
        *,
        completion: GuiCommandCompletion | None = None,
    ) -> bool:
        recipe = self.document.geometry_recipe
        settings = self.document.mesh_settings
        if not isinstance(recipe, NATIVE_GEOMETRY_TYPES) or not isinstance(
            settings,
            MeshSettings,
        ):
            raise RuntimeError("native geometry and mesh settings are required")
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
            delta, payload = value
            model, geometry, timings, _notices = self._unpack_model_load(
                payload
            )
            if not self._apply_session_delta(
                delta,
                model_geometry=geometry,
                timings=timings,
                source_label=str(
                    getattr(model, "name", None) or "未命名模型"
                ),
            ):
                raise RuntimeError("已接受的网格结果无法投影")
            self._import_notices = ()
            self.ribbon.set_current("模型")

        def apply_result(value: object) -> TaskApplyOutcome:
            model, _geometry, _timings, _notices = self._unpack_model_load(
                value
            )
            return self._session_task_outcome(
                self.session.accept_generated_model(task.token, model),
                value,
            )

        return self._start_task(
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
            apply_result=apply_result,
            completion=completion,
        )

    def open_inp(self) -> None:
        path, _filter = QFileDialog.getOpenFileName(
            self, "打开 Abaqus INP", "", "Abaqus INP 文件 (*.inp);;所有文件 (*)"
        )
        if path and self._confirm_discard_changes():
            receipt = self.open_inp_path(Path(path))
            if receipt.diagnostic is not None:
                self._show_command_rejection("模型加载失败", receipt)

    def new_native_model(self) -> None:
        if self.busy:
            return
        if not self._confirm_discard_changes():
            return
        receipt = self.new_native_project(NewNativeProjectCommand())
        if receipt.diagnostic is not None:
            self._show_command_rejection("新建自主项目", receipt)
            return
        self.model_tree.set_geometry_preview("Model-1", (), part_name="Part-1")
        self.viewport_panel.set_geometry_context(True)
        self.status_panel.set_state(
            "Native model created. Create a sketch, solid, or wire in Geometry.",
            5000,
        )
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
        receipt = self.open_project_path(path)
        if receipt.diagnostic is not None:
            self._show_command_rejection("打开自主项目失败", receipt)
            return
        if not self.import_notices:
            self.status_panel.set_state(
                "自主项目已打开，请生成网格并检查模型",
                6000,
            )
        self.ribbon.set_current("几何")

    def save_native_project(self) -> bool:
        if not self.document.can_save:
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
        receipt = self.save_project_path(path)
        if receipt.diagnostic is not None:
            self._show_command_rejection("保存自主项目失败", receipt)
            return False
        if self.document.dirty:
            self.status_panel.set_state(
                "已保存发起操作时的项目快照；当前修改仍未保存",
                6000,
            )
            return False
        self.status_panel.set_state(f"自主项目已保存：{path.name}", 5000)
        return True

    def reload_model(self) -> None:
        if (
            self.document.source_path is not None
            and self._confirm_discard_changes()
        ):
            receipt = self.reload_imported_source()
            if receipt.diagnostic is not None:
                self._show_command_rejection("重新加载失败", receipt)

    def _load_path(self, path: Path) -> None:
        receipt = self.open_inp_path(path)
        if receipt.diagnostic is not None:
            self._show_command_rejection("模型加载失败", receipt)

    def _begin_import(
        self,
        path: Path,
        *,
        completion: GuiCommandCompletion | None = None,
    ) -> bool:
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
            build_result = build_abaqus_model_with_report(deck)
            model = build_result.model
            timings["FEMModel 构建"] = perf_counter() - started
            context.report("正在生成显示网格……")
            started = perf_counter()
            geometry = build_model_geometry(model)
            timings["VTK 显示几何构建"] = perf_counter() - started
            context.checkpoint()
            return model, geometry, timings, build_result.notices

        def apply_result(value: object) -> TaskApplyOutcome:
            model, _geometry, _timings, _notices = self._unpack_model_load(
                value
            )
            return self._session_task_outcome(
                self.session.accept_imported_model(task.token, model),
                value,
            )

        def project_result(value: object) -> None:
            delta, payload = value
            _model, geometry, timings, notices = self._unpack_model_load(
                payload
            )
            if not self._apply_session_delta(
                delta,
                model_geometry=geometry,
                timings=timings,
                source_label=path.name,
            ):
                raise RuntimeError("已接受的导入结果无法投影")
            self._install_import_notices(notices)

        return self._start_task(
            workload,
            project_result,
            "模型加载失败",
            lambda message: self._session_task_failed(
                task.token,
                "模型加载失败",
                message,
            ),
            task_name="INP 导入",
            on_cancelled=lambda: self._session_task_cancelled(task.token),
            apply_result=apply_result,
            completion=completion,
        )

    def _model_loaded(
        self,
        path: Path,
        value: object,
        *,
        token: object | None = None,
    ) -> None:
        model, geometry, timings, notices = self._unpack_model_load(value)
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
            return
        self._install_import_notices(notices)

    def _generated_model_loaded(
        self,
        value: object,
        *,
        token: object | None = None,
    ) -> None:
        """Install a generated model through the same GUI path as an INP model."""
        model, geometry, timings, _notices = self._unpack_model_load(value)
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
            return
        self._import_notices = ()

    @staticmethod
    def _unpack_model_load(
        value: object,
    ) -> tuple[
        object,
        ModelGeometry,
        dict[str, float],
        tuple[object, ...],
    ]:
        if len(value) == 2:
            model, geometry = value
            timings = {}
            notices = ()
        elif len(value) == 3:
            model, geometry, timings = value
            notices = ()
        elif len(value) == 4:
            model, geometry, timings, notices = value
        else:
            raise ValueError(
                "model load result must contain model, geometry, optional "
                "timings, and optional notices"
            )
        return model, geometry, dict(timings), tuple(notices)

    def _install_import_notices(
        self,
        notices: tuple[object, ...],
    ) -> None:
        """Replace notices only after the imported Session CAS was projected."""

        self._import_notices = deepcopy(tuple(notices))
        if not self._import_notices:
            return
        self.status_panel.set_state(
            "；".join(
                str(getattr(notice, "message", notice))
                for notice in self._import_notices
            ),
            12000,
        )

    def _show_model_in_tree(self, model: object) -> None:
        definition_options = {
            "section_definitions": tuple(
                self.document.sections
            ),
            "region_assignments": tuple(
                self.document.assignments
            ),
        }
        if self.document.source_kind == "native" and self.document.parts:
            self.model_tree.set_model(
                model,
                feature_rows=tuple(
                    record.name for record in self.document.feature_history
                ),
                part_name=self.document.parts[0].name,
                scope_names=frozenset(self.document.named_regions),
                **definition_options,
            )
            return
        self.model_tree.set_model(model, **definition_options)

    def _install_model(
        self,
        model: object,
        geometry: ModelGeometry,
        timings: dict[str, float],
        *,
        source_label: str,
    ) -> None:
        self.status_panel.set_state("正在初始化视口……")
        self._scope_selection_overlay_active = False
        self._scope_selection_topology_cache = None
        self._close_inspection_windows()
        self._close_job_manager()
        self.geometry = geometry
        self.result_provider = None
        self.result_selection = None
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
        definitions = ModelDefinitions(
            materials=tuple(self.document.materials),
            sections=tuple(self.document.sections),
            assignments=tuple(self.document.assignments),
            steps=tuple(self.document.steps),
        )
        def frame_query(target: RegionRef | int) -> BeamFrameReport:
            return resolve_effective_beam_frames(model, target)
        started = perf_counter()
        self.inspection_service = InspectionService(
            model,
            definitions=definitions,
            effective_frame_query=frame_query,
        )
        timings["InspectionService 初始化"] = perf_counter() - started
        started = perf_counter()
        self._show_model_in_tree(model)
        timings["模型树更新"] = perf_counter() - started
        self.result_tree.clear_result()
        self.navigation.show_model()
        element_count = len(model.mesh.elements)
        node_count = len(model.mesh.nodes)
        policy = initial_display_policy(
            element_count,
            node_count,
            line_mesh=geometry.is_line_mesh,
        )
        simplified = policy["simplified"]
        self._model_edges_visible = policy["show_edges"]
        self.actions["edges"].setChecked(policy["show_edges"])
        self.actions["nodes"].setChecked(policy["show_nodes"])
        self.actions["node_labels"].setChecked(policy["show_labels"])
        self.actions["element_labels"].setChecked(policy["show_labels"])
        self.actions["symbols"].setChecked(policy["show_symbols"])

        started = perf_counter()
        self.viewport.set_model(
            model,
            geometry,
            refresh_symbols=False,
            render=False,
            effective_frame_query=frame_query,
        )
        self.viewport.set_edges_visible(self.actions["edges"].isChecked(), render=False)
        self.viewport.set_nodes_visible(self.actions["nodes"].isChecked(), render=False)
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
        wire_editor_active = self._wire_editor_controller is not None
        sketch_editor_active = self._sketch_editor_controller is not None
        editor_active = wire_editor_active or sketch_editor_active
        if wire_editor_active and self._wire_editor_controller.dirty:
            if not self._confirm_wire_editor_discard():
                return False
        if sketch_editor_active and self._sketch_editor_controller.dirty:
            if not self._confirm_sketch_editor_discard():
                return False
        if not self.document.dirty:
            if wire_editor_active:
                self._exit_wire_editor()
            elif sketch_editor_active:
                self._exit_sketch_editor()
            if editor_active:
                self._rebuild_full_projection()
            return True
        box = QMessageBox(self)
        box.setWindowTitle("未保存的修改")
        box.setIcon(QMessageBox.Icon.Warning)
        box.setText("当前模型包含尚未保存的修改。")
        save_button = None
        if self.document.can_save:
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
            accepted = self.save_native_project()
        else:
            accepted = clicked is discard_button
        if accepted and wire_editor_active:
            self._exit_wire_editor()
            self._rebuild_full_projection()
        elif accepted and sketch_editor_active:
            self._exit_sketch_editor()
            self._rebuild_full_projection()
        return accepted

    def close_model(self, *, confirm: bool = True) -> bool:
        if self.busy:
            return False
        if confirm and not self._confirm_discard_changes():
            return False
        receipt = self.close_session(
            CloseSessionCommand(self.document.session_revision)
        )
        if receipt.diagnostic is not None:
            self._show_command_rejection("关闭模型", receipt)
            return False
        self._selected_geometry_refs.clear()
        self._selected_mesh_scope_refs.clear()
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
        base_session_revision: int | None = None,
        material_renames: tuple[RenameIntent, ...] = (),
        section_renames: tuple[RenameIntent, ...] = (),
        material_deletes: tuple[DeleteIntent, ...] = (),
        section_deletes: tuple[DeleteIntent, ...] = (),
    ) -> bool:
        """Atomically compile editable definitions through the Session."""
        try:
            batch = DefinitionEditBatch(
                base_session_revision=(
                    self.document.session_revision
                    if base_session_revision is None
                    else base_session_revision
                ),
                materials=tuple(
                    self.document.materials
                    if materials is None
                    else materials
                ),
                sections=tuple(
                    self.document.sections
                    if sections is None
                    else sections
                ),
                assignments=tuple(
                    self.document.assignments
                    if assignments is None
                    else assignments
                ),
                steps=tuple(
                    self.document.steps if steps is None else steps
                ),
                material_renames=material_renames,
                section_renames=section_renames,
                material_deletes=material_deletes,
                section_deletes=section_deletes,
            )
        except (TypeError, ValueError) as error:
            self._show_error("模型定义", str(error))
            return False
        receipt = self.apply_definition_edit(batch)
        if receipt.diagnostic is not None:
            self._show_command_rejection("模型定义", receipt)
            return False
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
        base_revision = self.document.session_revision
        dialog = MaterialManagerDialog(self.document.materials, self)
        if not self._exec_dialog(dialog):
            return
        values = tuple(dialog.values())
        self._apply_model_definition_changes(
            "材料已修改，模型需要重新检查",
            materials=values,
            base_session_revision=base_revision,
            material_renames=dialog.rename_intents(),
            material_deletes=dialog.delete_intents(),
        )

    def edit_material(self, material_name: str) -> None:
        """Edit one tree material and compile it back into the current model."""
        row = next(
            (
                index
                for index, material in enumerate(
                    self.document.materials
                )
                if material.name == material_name
            ),
            None,
        )
        if row is None:
            self._show_error("编辑材料", f"材料不存在：{material_name}")
            return
        dialog = MaterialEditDialog(
            self.document.materials[row],
            self,
        )
        if not self._exec_dialog(dialog):
            return
        try:
            updated = dialog.material()
        except ValueError as error:
            self._show_error("编辑材料", str(error))
            return
        if any(
            index != row and material.name == updated.name
            for index, material in enumerate(
                self.document.materials
            )
        ):
            self._show_error(
                "编辑材料",
                f"材料名称已存在：{updated.name}",
            )
            return

        materials = list(self.document.materials)
        materials[row] = updated
        self._apply_model_definition_changes(
            "材料已修改，模型需要重新检查",
            materials=materials,
            material_renames=(
                (RenameIntent(material_name, updated.name),)
                if updated.name != material_name
                else ()
            ),
        )

    def show_section_manager(self) -> None:
        base_revision = self.document.session_revision
        capability_report = self._model_capability_report()
        section_authoring = (
            capability_report.operation("section.create")
            if capability_report is not None
            else None
        )
        dialog = SectionManagerDialog(
            self.document.materials,
            self.document.sections,
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
        if not self._exec_dialog(dialog):
            return
        values = tuple(dialog.values())
        self._apply_model_definition_changes(
            "截面已修改，模型需要重新检查",
            sections=values,
            base_session_revision=base_revision,
            section_renames=dialog.rename_intents(),
            section_deletes=dialog.delete_intents(),
        )

    def assign_section_to_region(self) -> None:
        if not self.document.sections:
            return
        selected_region_name = None
        if self._mesh_scope_selection_kind() == "element":
            selected_region_name = (
                self._create_region_from_current_mesh_selection()
            )
            if selected_region_name is None:
                return
        dialog = self._region_assignment_dialog(
            selected_region_name=selected_region_name,
        )
        if dialog is None:
            return
        if not self._exec_dialog(dialog):
            requested_scope_kind = dialog.requested_scope_kind()
            if requested_scope_kind is not None:
                self._request_analysis_geometry_selection(
                    "section",
                    requested_scope_kind,
                )
            return
        try:
            assignment = dialog.assignment()
            decision = dialog.candidate_decision(assignment)
        except (TypeError, ValueError) as error:
            self._show_error("截面分配", str(error))
            return
        if not self._strict_authoring_decision_enabled(decision):
            self._show_authoring_decision_error("截面分配", decision)
            return
        assignments = [
            current
            for current in self.document.assignments
            if current.region_name != assignment.region_name
        ] + [assignment]
        self._apply_model_definition_changes(
            "截面分配已修改，模型需要重新检查",
            assignments=assignments,
        )

    def edit_region_assignment(self, assignment_index: int) -> None:
        assignments = list(self.document.assignments)
        index = int(assignment_index)
        if index < 0 or index >= len(assignments):
            self._show_error(
                "截面分配",
                f"截面分配不存在：{assignment_index}",
            )
            return
        dialog = self._region_assignment_dialog(
            assignments[index],
            assignment_index=index,
        )
        if dialog is None or not self._exec_dialog(dialog):
            return
        try:
            updated = dialog.assignment()
            decision = dialog.candidate_decision(updated)
        except (TypeError, ValueError) as error:
            self._show_error("截面分配", str(error))
            return
        if not self._strict_authoring_decision_enabled(decision):
            self._show_authoring_decision_error("截面分配", decision)
            return
        assignments[index] = updated
        self._apply_model_definition_changes(
            "截面分配已修改，模型需要重新检查",
            assignments=assignments,
        )

    def _region_assignment_dialog(
        self,
        current: object | None = None,
        *,
        assignment_index: int | None = None,
        selected_region_name: str | None = None,
    ) -> RegionAssignmentDialog | None:
        capability_report = self._model_capability_report()
        regions = self._analysis_element_regions(capability_report)
        if current is not None:
            existing = RegionRef(
                "element_set",
                str(current.region_name),
            )
            if existing not in regions:
                regions.append(existing)
        allow_scope_selection = (
            current is None
            and self.document.model is not None
        )
        if not regions and not allow_scope_selection:
            self._show_error("截面分配", "当前模型没有可分配的单元作用域")
            return None
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
            for section in self.document.sections
        }
        if current is not None:
            existing_targets = list(
                compatible_targets.get(current.section_name, ())
            )
            existing_region = RegionRef(
                "element_set",
                str(current.region_name),
            )
            if existing_region not in existing_targets:
                existing_targets.append(existing_region)
            compatible_targets[current.section_name] = tuple(
                existing_targets
            )
        return RegionAssignmentDialog(
            self.document.sections,
            regions,
            self,
            compatible_targets=compatible_targets,
            current=current,
            candidate_evaluator=(
                lambda candidate, index=assignment_index:
                self._evaluate_region_assignment_candidate(
                    candidate,
                    candidate_index=index,
                )
            ),
            orientation_suggester=self._suggest_region_assignment_orientation,
            allow_scope_selection=allow_scope_selection,
            selected_region_name=selected_region_name,
        )

    def _suggest_region_assignment_orientation(
        self,
        region: RegionRef,
    ) -> BeamFrameReport | None:
        model = self.document.model
        if model is None:
            return None
        return resolve_effective_beam_frames(model, region)

    def _evaluate_region_assignment_candidate(
        self,
        candidate: RegionAssignment,
        *,
        candidate_index: int | None = None,
    ) -> AuthoringCapability:
        if type(candidate) is not RegionAssignment:
            raise TypeError("candidate must be RegionAssignment")
        section = next(
            (
                item
                for item in self.document.sections
                if item.name == candidate.section_name
            ),
            None,
        )
        if section is None:
            raise ValueError(f"unknown section: {candidate.section_name}")
        section_type = str(section.section_type).strip().casefold()
        if section_type == "beam":
            section_type = str(
                section.properties.get("section_type", section_type)
            ).strip().casefold()
        operation = f"section.{section_type}"
        model = self.document.model
        if model is None:
            return evaluate_native_assignment_candidate(
                self.document,
                candidate,
                candidate_index=candidate_index,
            )
        definitions = ModelDefinitions(
            materials=tuple(self.document.materials),
            sections=tuple(self.document.sections),
            assignments=tuple(self.document.assignments),
            steps=tuple(self.document.steps),
        )
        return evaluate_authoring_candidate(
            model,
            definitions,
            operation=operation,
            candidate=candidate,
            candidate_index=candidate_index,
        )

    def _show_authoring_decision_error(
        self,
        title: str,
        decision: AuthoringCapability,
    ) -> None:
        if type(decision) is not AuthoringCapability:
            raise TypeError("candidate decision must be AuthoringCapability")
        diagnostics = decision.diagnostics
        message = (
            self._render_diagnostics(diagnostics)
            if diagnostics
            else "当前 authoring capability 不允许提交该定义。"
        )
        self._show_error(title, message)

    def _analysis_definitions_changed(
        self,
        reason: str,
        steps: object,
    ) -> None:
        self._apply_model_definition_changes(reason, steps=steps)

    def create_static_step(self) -> None:
        if self.document.source_kind is None:
            return
        definitions = list(deepcopy(self.document.steps))
        name = f"Step-{len(definitions) + 1}"
        dialog = StaticStepDialog(name, self)
        if not self._exec_dialog(dialog):
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
        if self.document.source_kind == "native" and model is None:
            return
        selected_region = None
        if (
            self._mesh_scope_selection_kind() == "node"
        ):
            selected_name = self._create_region_from_current_mesh_selection()
            if selected_name is None:
                return
            selected_region = RegionRef("node_set", selected_name)
        node_regions, _edge_regions, _face_regions = self._analysis_region_names()
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
            allow_scope_selection=(
                self.document.model is not None
            ),
        )
        if not self._exec_dialog(dialog):
            requested_scope_kind = (
                dialog.requested_scope_kind()
                if hasattr(dialog, "requested_scope_kind")
                else None
            )
            if requested_scope_kind is not None:
                self._request_analysis_geometry_selection(
                    "boundary",
                    requested_scope_kind,
                )
            return
        try:
            step_name, boundaries = dialog.definitions()
        except ValueError as error:
            self._show_error("位移边界条件", str(error))
            return
        definitions = list(deepcopy(self.document.steps))
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
        if self.document.source_kind == "native" and model is None:
            return
        selected_region = None
        preferred_kind = None
        capability_report = self._model_capability_report()
        if capability_report is None:
            return
        selected_mesh_kind = self._mesh_scope_selection_kind()
        if (
            selected_mesh_kind in {"node", "edge", "face", "element"}
        ):
            preferred_kind = {
                "node": "node",
                "edge": "edge",
                "face": "surface",
                "element": "line",
            }[selected_mesh_kind]
            if preferred_kind not in capability_report.load_kinds:
                self._show_error(
                    "创建载荷",
                    "所选区域不支持当前模型的分布载荷契约。",
                )
                return
            selected_name = self._create_region_from_current_mesh_selection()
            if selected_name is None:
                return
            selected_region = RegionRef(
                {
                    "node": "node_set",
                    "edge": "edge",
                    "surface": "surface",
                    "line": "element_set",
                }[preferred_kind],
                selected_name,
            )
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
            candidate_evaluator=self._evaluate_line_load_candidate,
            scope_selection_kinds=(
                tuple(
                    kind
                    for kind in ("node", "edge", "surface", "line")
                    if kind in capability_report.load_kinds
                )
                if self.document.model is not None
                else ()
            ),
        )
        if not self._exec_dialog(dialog):
            requested_scope_kind = (
                dialog.requested_scope_kind()
                if hasattr(dialog, "requested_scope_kind")
                else None
            )
            if requested_scope_kind is not None:
                self._request_analysis_geometry_selection(
                    "load",
                    requested_scope_kind,
                )
            return
        try:
            step_name, load = dialog.definition()
        except ValueError as error:
            self._show_error("创建载荷", str(error))
            return
        if (
            isinstance(load, LineLoad)
            and load.coordinate_system == "local"
        ):
            decision = dialog.candidate_decision(load, step_name)
            if not self._strict_authoring_decision_enabled(decision):
                self._show_authoring_decision_error(
                    "创建梁线载荷",
                    decision,
                )
                return
        definitions = list(deepcopy(self.document.steps))
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
        authoring = describe_session_authoring(self.document)
        capability = authoring.operation("output_request.create")
        if not capability.can_submit:
            self._show_authoring_decision_error(
                "创建输出请求",
                capability,
            )
            return
        catalog = authoring.output_request_catalog
        candidates = () if catalog is None else catalog.candidates
        if not candidates:
            self._show_error(
                "创建输出请求",
                "当前结果能力目录没有受支持的输出请求候选。",
            )
            return
        step_names = [
            step.name
            for step in self.document.steps
            if step.name.strip().casefold() != "initial"
        ]
        dialog = OutputRequestDialog(
            step_names,
            self,
            candidates=candidates,
        )
        if not self._exec_dialog(dialog):
            return
        try:
            step_name, request = dialog.definition()
        except (TypeError, ValueError) as error:
            self._show_error("创建输出请求", str(error))
            return
        if type(request) is not OutputRequest:
            self._show_error(
                "创建输出请求",
                "输出请求候选必须生成 typed OutputRequest。",
            )
            return
        definitions = list(deepcopy(self.document.steps))
        target = next(
            (
                step
                for step in definitions
                if step.name == step_name
                and step.name.strip().casefold() != "initial"
            ),
            None,
        )
        if target is None:
            self._show_error(
                "创建输出请求",
                f"分析步不存在或不可编辑：{step_name}",
            )
            return
        target.outputs = tuple(target.outputs) + (deepcopy(request),)
        self._warn_imported_output_overlay()
        self._analysis_definitions_changed(
            "输出请求已创建，模型需要重新检查",
            definitions,
        )

    def _warn_imported_output_overlay(self) -> None:
        if self.document.source_kind != "imported":
            return
        QMessageBox.warning(
            self,
            "输出请求",
            _IMPORTED_OUTPUT_REQUEST_WARNING,
        )

    @staticmethod
    def _output_collections_changed(
        before: object,
        after: object,
    ) -> bool:
        return tuple(
            (step.name, tuple(step.outputs))
            for step in before
            if step.outputs
        ) != tuple(
            (step.name, tuple(step.outputs))
            for step in after
            if step.outputs
        )

    def _analysis_manager_dialog(
        self,
    ) -> AnalysisDefinitionManagerDialog | None:
        if not self.document.steps:
            return None
        node_regions, edge_regions, face_regions, line_regions = (
            self._supported_load_regions()
        )
        authoring = describe_session_authoring(self.document)
        capability_report = self._model_capability_report()
        dimensions = (
            capability_report.dofs_per_node
            if capability_report is not None
            and capability_report.dofs_per_node is not None
            else 3
        )
        return AnalysisDefinitionManagerDialog(
            self.document.steps,
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
            candidate_evaluator=self._evaluate_line_load_candidate,
            output_view_capability=authoring.operation(
                "output_request.view"
            ),
            output_delete_capability=authoring.operation(
                "output_request.delete"
            ),
        )

    def _evaluate_line_load_candidate(
        self,
        candidate: LineLoad,
        step_name: str,
        *,
        candidate_index: int | None = None,
    ) -> AuthoringCapability:
        model = self.document.model
        if model is None:
            return evaluate_native_line_load_candidate(
                self.document,
                candidate,
                step_name,
                candidate_index=candidate_index,
            )
        definitions = ModelDefinitions(
            materials=tuple(self.document.materials),
            sections=tuple(self.document.sections),
            assignments=tuple(self.document.assignments),
            steps=tuple(self.document.steps),
        )
        return evaluate_authoring_candidate(
            model,
            definitions,
            operation="load.line.local",
            candidate=candidate,
            step_name=str(step_name),
            candidate_index=candidate_index,
        )

    @staticmethod
    def _strict_authoring_decision_enabled(
        decision: AuthoringCapability,
    ) -> bool:
        if type(decision) is not AuthoringCapability:
            raise TypeError("candidate decision must be AuthoringCapability")
        return decision.can_submit

    def show_analysis_manager(self) -> None:
        dialog = self._analysis_manager_dialog()
        if dialog is None:
            return
        if not self._exec_dialog(dialog):
            return
        values = dialog.values()
        current = tuple(self.document.steps)
        if tuple(values) == current:
            return
        if self._output_collections_changed(current, values):
            capability = describe_session_authoring(
                self.document
            ).operation("output_request.delete")
            if not capability.can_submit:
                self._show_authoring_decision_error(
                    "删除输出请求",
                    capability,
                )
                return
            self._warn_imported_output_overlay()
        self._analysis_definitions_changed(
            "分析步、边界、载荷或输出请求已修改，模型需要重新检查",
            values,
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
                    for item in self.document.steps
                    if item.name == self._current_step_name
                ),
                None,
            )
            if step is None:
                return None
            self._show_information("分析步信息", [
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
        step_name = self._current_step_name
        if step_name is None:
            return False
        receipt = self.check_step(step_name)
        if receipt.diagnostic is not None:
            self._show_command_rejection("模型检查失败", receipt)
            return False
        completion = receipt.completion
        if completion is not None:
            def show_report(record: TaskCompletion) -> None:
                if record.state is not BackgroundTaskState.SUCCEEDED:
                    return
                validation = self.session.validation_for(step_name)
                if validation is not None and validation.report.passed:
                    self._show_model_check_report(validation.report)

            completion.observe(show_report)
        return receipt.completion is not None

    def _begin_model_check(
        self,
        step_name: str,
        *,
        completion: GuiCommandCompletion | None = None,
        show_success: bool,
    ) -> bool:
        task = self._prepare_model_check(step_name)
        if task is None:
            raise RuntimeError(f"step cannot be checked: {step_name}")

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
            delta, report = value
            self._project_model_check(
                delta,
                report,
                show_success=show_success,
            )

        def apply_result(value: object) -> TaskApplyOutcome:
            if not isinstance(value, PreflightReport):
                raise TypeError("model check must return PreflightReport")
            return self._session_task_outcome(
                self.session.accept_validation(task.token, value),
                value,
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
            apply_result=apply_result,
            completion=completion,
        )

    def _prepare_model_check(
        self,
        step_name: str | None = None,
    ) -> object | None:
        step_name = self._current_step_name if step_name is None else step_name
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
        return self._project_model_check(
            delta,
            report,
            show_success=show_success,
        )

    def _project_model_check(
        self,
        delta: SessionDelta,
        report: PreflightReport,
        *,
        show_success: bool,
    ) -> bool:
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
            self._show_model_check_report(report)
        self.status_panel.set_state(
            "模型检查通过（有警告）"
            if report.warnings
            else "模型检查通过",
            4000,
        )
        return True

    def _show_model_check_report(self, report: PreflightReport) -> None:
        facts = report.facts
        warnings = "；".join(
            f"[{item.code}] {item.message}"
            for item in report.warnings
        )
        self._show_information("模型检查", [
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
        if self._exec_dialog(dialog):
            receipt = self.submit_run(dialog.job_name, dialog.step_name)
            if receipt.diagnostic is not None:
                self._show_command_rejection("创建作业失败", receipt)

    def resubmit_job(self, source_name: str | None = None) -> None:
        """以当前模型状态重新提交某个已完成或失败作业。"""
        if self.busy or self.document.model is None or self.geometry is None:
            return
        source = self.session.find_run(source_name)
        if source is None or source.status not in {
            RunStatus.SUCCEEDED,
            RunStatus.FAILED,
            RunStatus.CANCELLED,
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
        if self._exec_dialog(dialog):
            receipt = self.submit_run(
                dialog.job_name,
                dialog.step_name,
            )
            if receipt.diagnostic is not None:
                self._show_command_rejection("重新提交作业失败", receipt)

    def _submit_job(
        self,
        name: str,
        step_name: str,
        *,
        source_job_name: str | None = None,
    ) -> AnalysisRun | None:
        """验证并后台提交作业；全部状态均仅保留在内存中。"""
        del source_job_name
        receipt = self.submit_run(name, step_name)
        if receipt.diagnostic is not None:
            self._show_command_rejection("创建作业失败", receipt)
            return None
        return self.session.find_run(str(name).strip())

    def _begin_submit_run(
        self,
        name: str,
        step_name: str,
        *,
        completion: GuiCommandCompletion | None = None,
    ) -> AnalysisRun | None:
        if self.document.model is None or self.geometry is None or self.busy:
            raise RuntimeError("a current model is required and the task controller must be idle")
        if type(name) is not str or type(step_name) is not str:
            raise TypeError("name and step_name must be strings")
        clean_name = name.strip()
        clean_step = step_name.strip()
        if not clean_name:
            raise ValueError("作业名称不能为空。")
        if len(clean_name) > 64:
            raise ValueError("作业名称不能超过 64 个字符。")
        if self.session.find_run(clean_name) is not None:
            raise ValueError(f"作业名称已存在：{clean_name}")
        if clean_step not in self.session.runnable_step_names():
            raise ValueError(f"分析步不存在：{clean_step}")
        task = self.session.prepare_solve(clean_step, clean_name)
        if task.delta is not None:
            self._apply_session_delta(task.delta)
        self._apply_session_delta(self.session.begin_run(task.token))
        job = self.session.find_run(task.run_id)
        if job is None:
            return None
        self.status_panel.set_state(f"正在分析：{job.name}")
        self._refresh_job_manager()
        stage = {"name": "模型验证"}

        def workload(
            context: TaskContext,
        ) -> tuple[object, dict[str, float]]:
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
            context.report("正在执行输出请求……")
            started = perf_counter()
            bundle = build_solve_result_bundle(
                task,
                result,
                cancellation=context,
            )
            timings["输出请求与初始结果"] = perf_counter() - started
            context.checkpoint()
            return bundle, timings

        def apply_result(value: object) -> TaskApplyOutcome:
            bundle, timings = value
            delta = self.session.accept_run_succeeded(
                task.token,
                bundle,
                timings=timings,
            )
            return self._session_task_outcome(
                delta,
                timings,
            )

        def project_result(value: object) -> None:
            delta, timings = value
            if not self._apply_session_delta(delta):
                raise RuntimeError("已接受的求解结果无法投影")
            completed = self.session.find_run(task.token.run_id)
            if completed is None:
                raise RuntimeError("已接受的分析作业不存在")
            activation_started = perf_counter()
            self._activate_job_result(completed, completion=True)
            timings["首次结果显示"] = perf_counter() - activation_started
            self._refresh_job_manager()
            self.status_panel.set_state(f"分析完成：{completed.name}", 5000)
            self.ribbon.set_current("结果")

        started = self._start_task(
            workload,
            project_result,
            "分析运行失败",
            lambda message, token=task.token, current_stage=stage: self._job_failed(
                token,
                message,
                validation_failure=current_stage["name"] == "模型验证",
            ),
            task_name=f"作业 {job.name}",
            on_cancelled=lambda token=task.token: self._job_cancelled(token),
            apply_result=apply_result,
            completion=completion,
        )
        if started:
            return job
        self._apply_session_delta(
            self.session.accept_run_failed(
                task.token,
                "analysis task could not be started",
            )
        )
        return None

    def _job_succeeded(self, token: object, value: object) -> None:
        bundle, timings = value
        delta = self.session.accept_run_succeeded(
            token,
            bundle,
            timings=timings,
        )
        if not self._apply_session_delta(delta):
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

    def _activate_job_result(
        self,
        job: AnalysisRun,
        *,
        completion: bool = False,
    ) -> None:
        """将一个已完成会话作业的结果接入现有后处理流程。"""
        if not job.has_result:
            return
        if self.document.displayed_result_run_id != job.run_id:
            projection = self.session.prepare_result_projection(job.run_id)
            if not self._apply_revision_neutral_task_receipt(
                self.session.accept_result_projection(
                    projection.token
                )
            ):
                return
            self._apply_session_delta(self.session.select_result(job.run_id))
        self._set_current_step(job.step_name)
        provider = self._current_result_provider()
        selection = self.result_selection
        if (
            provider is None
            or provider.source.run_id != job.run_id
            or type(selection) is not ScalarFieldSelection
        ):
            return
        self._display = DisplayState("deformed", True)
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
        self.result_tree.set_catalog(
            f"{job.name} · {job.step_name}",
            provider.catalog(),
        )
        if not self.result_tree.select_selection(selection):
            raise RuntimeError(
                "current result selection is missing from the result tree"
            )
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
            dialog = JobManagerDialog(self.document.runs, self)
            dialog.resubmitRequested.connect(self.resubmit_job)
            dialog.openResultRequested.connect(self.open_job_result)
            self._fit_viewport_when_dialog_finishes(dialog)
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
            self._job_manager.refresh(self.document.runs)

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
        receipt = self.select_run_result(job.run_id)
        if receipt.diagnostic is not None:
            self._show_command_rejection("打开结果", receipt)
            return
        self._refresh_job_manager()
        self.ribbon.set_current("结果")

    def _session_task_failed(
        self,
        token: object,
        title: str,
        message: str,
    ) -> None:
        delta = self.session.accept_task_failed(token, message)
        applied = (
            self._apply_revision_neutral_task_receipt(delta)
            if not delta.changed and not delta.invalidated
            else self._apply_session_delta(delta)
        )
        if applied:
            self._show_error(title, message)

    def _session_task_cancelled(self, token: object) -> None:
        delta = self.session.accept_task_cancelled(token)
        if not delta.changed and not delta.invalidated:
            self._apply_revision_neutral_task_receipt(delta)
            return
        self._apply_session_delta(delta)

    def _start_task(
        self,
        workload: Callable[[TaskContext], object],
        on_success: Callable[[object], None],
        error_title: str,
        on_failure: Callable[[str], None] | None = None,
        *,
        task_name: str = "后台任务",
        on_cancelled: Callable[[], None] | None = None,
        apply_result: Callable[[object], TaskApplyOutcome] | None = None,
        completion: GuiCommandCompletion | None = None,
    ) -> bool:
        if self.busy:
            self.status_panel.set_state(
                "当前任务正在运行："
                f"{self.task_controller.current_task_name or '后台任务'}",
                4000,
            )
            return False
        result_applier = apply_result or TaskApplyOutcome.accepted

        def project_terminal(record: TaskCompletion) -> None:
            if record.state is BackgroundTaskState.FAILED:
                message = record.message or "后台任务失败"
                try:
                    if on_failure is None:
                        self._show_error(error_title, message)
                    else:
                        on_failure(message)
                except Exception:
                    logging.exception(
                        "GUI background task failure callback failed"
                    )
                    self._show_error(error_title, message)
            elif record.state is BackgroundTaskState.CANCELLED:
                try:
                    if on_cancelled is not None:
                        on_cancelled()
                except Exception as error:
                    logging.exception(
                        "GUI background task cancellation callback failed"
                    )
                    self._show_error(
                        error_title,
                        str(error).strip() or type(error).__name__,
                    )
                self.status_panel.set_state(
                    f"已取消：{record.task_name}",
                    4000,
                )
            elif record.state is BackgroundTaskState.DISCARDED:
                self.status_panel.set_state(
                    record.message or "任务结果已过期，未应用",
                    5000,
                )

        def terminal(record: TaskCompletion) -> None:
            try:
                project_terminal(record)
            finally:
                if completion is not None:
                    completion.complete(record)

        task_id = self.task_controller.start(
            workload,
            task_name=task_name,
            apply_result=result_applier,
            project_result=on_success,
            rebuild_projection=self._rebuild_full_projection,
            on_terminal=terminal,
            on_progress=self.status_panel.set_state,
            on_projection_error=self._task_projection_failed,
        )
        if task_id is not None and completion is not None:
            completion.bind_task_id(task_id)
        return task_id is not None

    def _task_busy_changed(self, busy: bool) -> None:
        self.status_panel.set_task_active(bool(busy))
        self._update_action_states()

    def _task_cancelling_changed(self, cancelling: bool) -> None:
        if cancelling:
            self.status_panel.set_task_active(True, cancelling=True)
            self.status_panel.set_state(
                "正在取消，等待当前后端调用返回："
                f"{self.task_controller.current_task_name or '后台任务'}"
            )

    def _task_projection_failed(self, message: str) -> None:
        logging.error("GUI task projection failed: %s", message)
        self.status_panel.set_state("任务已接受，但界面刷新失败", 8000)

    def _rebuild_full_projection(self) -> None:
        snapshot = self.session.snapshot()
        self._applied_session_revision = -1
        if not self._apply_session_delta(
            SessionDelta(
                session_revision=snapshot.session_revision,
                reason="full GUI projection rebuild",
            )
        ):
            raise RuntimeError("无法从最新 Session snapshot 重建界面")

    def cancel_current_task(
        self,
        *,
        after_cleanup: Callable[[], None] | None = None,
    ) -> bool:
        running = next(
            (
                run
                for run in self.document.runs
                if run.status is RunStatus.RUNNING
            ),
            None,
        )
        if running is not None:
            try:
                self._apply_session_delta(
                    self.session.request_cancel(running.run_id)
                )
            except (KeyError, RuntimeError):
                pass
        return self.task_controller.request_cancel(
            after_cleanup=after_cleanup
        )

    def _show_error(self, title: str, message: str) -> None:
        self.status_panel.set_state("操作失败", 5000)
        box = QMessageBox(QMessageBox.Icon.Critical, title, message, parent=self)
        box.setStandardButtons(QMessageBox.StandardButton.Close)
        box.button(QMessageBox.StandardButton.Close).setText("关闭")
        box.exec()

    def viewport_fit(self) -> None:
        self.viewport.fit()

    def _schedule_viewport_fit(self) -> None:
        """Fit once after the current dialog/layout transition has settled."""

        if self._viewport_fit_pending:
            return
        self._viewport_fit_pending = True
        QTimer.singleShot(0, self._run_scheduled_viewport_fit)

    def _run_scheduled_viewport_fit(self) -> None:
        self._viewport_fit_pending = False
        self.viewport.fit()

    def _exec_dialog(self, dialog: QDialog) -> int:
        """Execute one FEM dialog and fit after its caller finishes projection."""

        try:
            return int(dialog.exec())
        finally:
            self._schedule_viewport_fit()

    def _show_information(
        self,
        title: str,
        rows: Sequence[tuple[str, object]],
    ) -> None:
        show_information(self, title, rows)
        self._schedule_viewport_fit()

    def _fit_viewport_when_dialog_finishes(self, dialog: QDialog) -> None:
        """Restore full-model framing after a view-affecting dialog closes."""

        dialog.finished.connect(self._fit_viewport_after_dialog)

    def _fit_viewport_after_dialog(self, _result: int = 0) -> None:
        self._schedule_viewport_fit()

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

    def _canonical_geometry_selection(
        self,
    ) -> tuple[LogicalEntityRef, ...]:
        return tuple(
            sorted(
                self._selected_geometry_refs,
                key=logical_ref_sort_key,
            )
        )

    def _geometry_selection_kind(self) -> str | None:
        kinds = {
            reference.kind
            for reference in self._selected_geometry_refs
        }
        if not kinds:
            return None
        if len(kinds) != 1:
            raise RuntimeError(
                "geometry selection contains incompatible entity kinds"
            )
        return next(iter(kinds))

    def _canonical_mesh_scope_selection(
        self,
    ) -> tuple[MeshEntityRef, ...]:
        return tuple(
            sorted(
                self._selected_mesh_scope_refs,
                key=lambda reference: (
                    reference.kind,
                    reference.identity,
                    reference.node_ids,
                ),
            )
        )

    def _mesh_scope_selection_kind(self) -> str | None:
        kinds = {
            reference.kind
            for reference in self._selected_mesh_scope_refs
        }
        if not kinds:
            return None
        if len(kinds) != 1:
            raise RuntimeError(
                "mesh scope selection contains incompatible entity kinds"
            )
        return next(iter(kinds))

    def _set_selection_mode(self, mode: str) -> None:
        normalized = "element" if mode == "element" else "node"
        if (
            self._selected_geometry_refs
            or self.viewport._selection_mode != normalized
        ):
            self.selection.clear()
            self._selected_geometry_refs.clear()
            self._selected_mesh_scope_refs.clear()
            self.viewport.clear_selection()
            self.status_panel.set_object()
            self.actions["selected_info"].setEnabled(False)
        self.selection.mode = normalized
        self.viewport.set_selection_mode(self.selection.mode)
        self.status_panel.set_selection_mode(self.selection.mode)

    def _set_geometry_selection_mode(self, mode: str) -> None:
        normalized = mode if mode in {"point", "edge", "face", "body"} else "body"
        recipe = self.document.geometry_recipe
        if (
            normalized == "face"
            and isinstance(recipe, NATIVE_GEOMETRY_TYPES)
            and geometry_dimension(recipe) == 1
        ):
            normalized = "body"
        has_fem_selection = (
            self.selection.node_id is not None
            or self.selection.element_id is not None
        )
        if (
            has_fem_selection
            or (
                self._selected_geometry_refs
                and self._geometry_selection_kind() != normalized
            )
        ):
            self._selected_geometry_refs.clear()
            self.viewport.clear_selection()
            self.status_panel.set_object()
        self.selection.clear()
        self.actions["selected_info"].setEnabled(False)
        self._geometry_selection_mode = normalized
        self.viewport.set_selection_mode(f"geometry_{normalized}")
        self.status_panel.set_selection_mode(f"geometry_{normalized}")

    def _set_mesh_scope_selection_mode(self, mode: str) -> None:
        normalized = (
            mode
            if mode in {"node", "edge", "face", "element"}
            else "node"
        )
        if (
            self._selected_mesh_scope_refs
            and self._mesh_scope_selection_kind() != normalized
        ):
            self._selected_mesh_scope_refs.clear()
            self.viewport.clear_selection()
            self.status_panel.set_object()
        self.selection.clear()
        self._selected_geometry_refs.clear()
        self.actions["selected_info"].setEnabled(False)
        self.viewport.set_selection_mode(f"mesh_{normalized}")
        self.status_panel.set_selection_mode(f"mesh_{normalized}")

    def clear_selection(self) -> None:
        self.selection.clear()
        self._pending_local_mesh_selection = False
        self._pending_analysis_selection = None
        self._pending_scope_kind = None
        self.viewport_panel.scope_creation_bar.finish()
        if self._scope_selection_overlay_active:
            self.viewport.hide_geometry_selection_overlay()
            self._scope_selection_overlay_active = False
        self._selected_geometry_refs.clear()
        self._selected_mesh_scope_refs.clear()
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
        if kind == "node":
            self.selection.select_node(key)
            self.viewport.highlight_node(key)
        elif kind == "element":
            self.selection.select_element(key)
            self.viewport.highlight_element(key)
        else:
            raise ValueError(
                "entityPicked 只接受 FEM node 或 element"
            )
        self.model_tree.select_entity(kind, key)
        self.status_panel.set_selection_mode(kind)
        self.status_panel.set_object(
            f"{'节点' if kind == 'node' else '单元'} {key}",
            self._entity_coordinates(kind, key),
        )
        self.actions["selected_info"].setEnabled(True)

    def _on_geometry_entity_pick(
        self,
        reference: LogicalEntityRef,
    ) -> None:
        if type(reference) is not LogicalEntityRef:
            raise TypeError(
                "geometryEntityPicked 必须携带 LogicalEntityRef"
            )
        additive = self._geometry_pick_is_additive()
        if (
            self._geometry_selection_kind() != reference.kind
            or not additive
        ):
            self._selected_geometry_refs = {reference}
        elif reference in self._selected_geometry_refs:
            self._selected_geometry_refs.remove(reference)
        else:
            self._selected_geometry_refs.add(reference)
        self._refresh_geometry_scope_selection(reference.kind)

    def _on_geometry_entities_box_selected(
        self,
        references: object,
    ) -> None:
        selected = tuple(references)
        if any(type(reference) is not LogicalEntityRef for reference in selected):
            raise TypeError(
                "geometry box selection requires LogicalEntityRef values"
            )
        if not selected:
            self._on_viewport_pick_missed(
                f"geometry_{self._geometry_selection_mode}"
            )
            return
        kind = selected[0].kind
        if any(reference.kind != kind for reference in selected):
            raise ValueError(
                "geometry box selection cannot mix entity kinds"
            )
        if self._geometry_pick_is_additive():
            self._selected_geometry_refs.symmetric_difference_update(
                selected
            )
        else:
            self._selected_geometry_refs = set(selected)
        self._refresh_geometry_scope_selection(kind)

    def _refresh_geometry_scope_selection(self, kind: str) -> None:
        references = self._canonical_geometry_selection()
        if self._pending_scope_kind in {"edge", "face", "body"}:
            self._expand_semantic_scope_selection()
        self.viewport_panel.scope_creation_bar.set_selection_ready(
            bool(self._canonical_mesh_scope_selection())
        )
        self.viewport.highlight_geometry_entities(references)
        mode = f"geometry_{kind}"
        labels = {
            "point": "点",
            "edge": "边",
            "face": "面",
            "body": "体",
        }
        self.status_panel.set_selection_mode(mode)
        selected_count = len(references)
        wire_single_label = None
        if (
            selected_count == 1
            and isinstance(self.document.geometry_recipe, NATIVE_GEOMETRY_TYPES)
            and geometry_dimension(self.document.geometry_recipe) == 1
        ):
            reference = references[0]
            semantic_name = reference.logical_id.partition(":")[2]
            wire_single_label = {
                "point": f"连接点 {semantic_name}",
                "edge": f"杆件 {semantic_name}",
                "body": "线体区域",
            }.get(kind)
        self.status_panel.set_object(
            wire_single_label
            if wire_single_label is not None
            else (
                f"已选择 {selected_count} 个"
                f"{labels.get(kind, '几何实体')}"
            )
            if selected_count
            else "—"
        )
        self.actions["selected_info"].setEnabled(False)
        self._update_action_states()

    def _expand_semantic_scope_selection(self) -> None:
        kind = self._pending_scope_kind
        if kind not in {"edge", "face", "body"}:
            return
        topology = self._scope_selection_topology()
        expanded = {
            mesh_reference
            for logical_reference in self._selected_geometry_refs
            for mesh_reference in topology.mesh_references.get(
                logical_reference,
                (),
            )
        }
        self._selected_mesh_scope_refs = expanded

    def _on_mesh_scope_entity_pick(
        self,
        reference: MeshEntityRef,
    ) -> None:
        if type(reference) is not MeshEntityRef:
            raise TypeError(
                "meshEntityPicked 必须携带 MeshEntityRef"
            )
        additive = self._geometry_pick_is_additive()
        if (
            self._mesh_scope_selection_kind() != reference.kind
            or not additive
        ):
            self._selected_mesh_scope_refs = {reference}
        elif reference in self._selected_mesh_scope_refs:
            self._selected_mesh_scope_refs.remove(reference)
        else:
            self._selected_mesh_scope_refs.add(reference)
        self._refresh_mesh_scope_selection(reference.kind)

    def _on_mesh_entities_box_selected(
        self,
        references: object,
    ) -> None:
        selected = tuple(references)
        if any(type(reference) is not MeshEntityRef for reference in selected):
            raise TypeError(
                "mesh box selection requires MeshEntityRef values"
            )
        if not selected:
            self._on_viewport_pick_missed(
                f"mesh_{self._pending_scope_kind or 'node'}"
            )
            return
        kind = selected[0].kind
        if any(reference.kind != kind for reference in selected):
            raise ValueError("mesh box selection cannot mix entity kinds")
        if self._geometry_pick_is_additive():
            self._selected_mesh_scope_refs.symmetric_difference_update(
                selected
            )
        else:
            self._selected_mesh_scope_refs = set(selected)
        self._refresh_mesh_scope_selection(kind)

    def _refresh_mesh_scope_selection(self, kind: str) -> None:
        references = self._canonical_mesh_scope_selection()
        self.viewport_panel.scope_creation_bar.set_selection_ready(
            bool(references)
        )
        self.viewport.highlight_mesh_entities(references)
        labels = {
            "node": "节点",
            "edge": "单元边",
            "face": "单元面",
            "element": "单元",
        }
        self.status_panel.set_selection_mode(f"mesh_{kind}")
        self.status_panel.set_object(
            (
                f"已选择 {len(references)} 个"
                f"{labels.get(kind, '网格实体')}"
            )
            if references
            else "—"
        )
        self.actions["selected_info"].setEnabled(False)
        self._update_action_states()

    def _on_viewport_pick_missed(self, kind: str) -> None:
        """Clear a replace-selection click without cancelling guided selection."""
        if self._geometry_pick_is_additive():
            return
        if kind.startswith("geometry_"):
            self._selected_geometry_refs.clear()
            if self._pending_scope_kind in {"edge", "face", "body"}:
                self._selected_mesh_scope_refs.clear()
        elif kind.startswith("mesh_"):
            self._selected_mesh_scope_refs.clear()
        else:
            self.selection.clear()
        self.viewport.clear_selection()
        self.viewport_panel.scope_creation_bar.set_selection_ready(False)
        self.status_panel.set_object()
        self.actions["selected_info"].setEnabled(False)
        self._update_action_states()

    def _confirm_guided_selection(self) -> None:
        pending_scope_selection = self._pending_analysis_selection is not None
        active_selection = (
            self._selected_mesh_scope_refs
            if pending_scope_selection
            else self._selected_geometry_refs
        )
        if not active_selection:
            if self._pending_local_mesh_selection or self._pending_analysis_selection:
                self.status_panel.set_state("请先选择至少一个对象", 3000)
            return
        if self._pending_local_mesh_selection:
            self._pending_local_mesh_selection = False
            QTimer.singleShot(0, self.set_local_mesh_control)
            return
        if self._pending_analysis_selection is not None:
            self._complete_scope_creation_from_bar()

    def _cancel_guided_selection(self) -> None:
        if not self._pending_local_mesh_selection and self._pending_analysis_selection is None:
            return
        self._pending_local_mesh_selection = False
        self._pending_analysis_selection = None
        self._pending_scope_kind = None
        self.viewport_panel.scope_creation_bar.finish()
        if self._scope_selection_overlay_active:
            self.viewport.hide_geometry_selection_overlay()
            self._scope_selection_overlay_active = False
        self.clear_selection()
        self.status_panel.set_state("已取消作用域选择", 3000)

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

    def _current_result_provider(self) -> ResultProvider | None:
        """Return the exact provider for the currently displayed Session result."""

        provider = self.result_provider
        record = self.session.current_result()
        artifact = self.document.artifact
        geometry = self.geometry
        if (
            type(provider) is not ResultProvider
            or record is None
            or artifact is None
            or geometry is None
            or provider.source != record.materialization.source
            or provider.snapshot.generation
            != record.materialization.generation
            or provider.source.artifact_id != artifact.artifact_id
            or geometry.artifact_id != artifact.artifact_id
        ):
            return None
        return provider

    @staticmethod
    def _catalog_availability_for_selection(
        provider: ResultProvider,
        selection: ScalarFieldSelection,
    ) -> FieldAvailability:
        if type(provider) is not ResultProvider:
            raise TypeError("provider must be exactly ResultProvider")
        if type(selection) is not ScalarFieldSelection:
            raise TypeError("selection must be a ScalarFieldSelection")
        matches = tuple(
            availability
            for availability in provider.catalog().fields
            if availability.key == selection.field_key
        )
        if len(matches) != 1:
            raise KeyError("selection is outside the current result catalog")
        availability = matches[0]
        if selection.component not in availability.descriptor.columns:
            raise ValueError(
                "selection component is outside the field descriptor"
            )
        return availability

    @classmethod
    def _selection_belongs_to_catalog(
        cls,
        provider: ResultProvider,
        selection: ScalarFieldSelection | None,
    ) -> bool:
        if type(selection) is not ScalarFieldSelection:
            return False
        try:
            cls._catalog_availability_for_selection(
                provider,
                selection,
            )
        except (KeyError, TypeError, ValueError):
            return False
        return True

    @staticmethod
    def _result_selection_outcome(
        provider: ResultProvider,
        selection: ScalarFieldSelection,
    ) -> GuiCommandOutcome:
        field_data = provider.field(selection.field_key)
        return GuiCommandOutcome(
            source=provider.source,
            materialization_generation=provider.snapshot.generation,
            selection=selection,
            record_count=len(field_data.locations),
        )

    def _result_deformation_scale(
        self,
        provider: ResultProvider,
        *,
        shape_mode: str | None = None,
        scale_mode: str | None = None,
        scale_value: float | None = None,
    ) -> float:
        shape = self._display.shape_mode if shape_mode is None else shape_mode
        if shape != "deformed":
            return 0.0
        mode = self._scale_mode if scale_mode is None else scale_mode
        if mode == "real":
            return 1.0
        if mode == "custom":
            value = self._scale_value if scale_value is None else scale_value
            scale = float(value)
            if not np.isfinite(scale) or scale < 0.0:
                raise ValueError(
                    "custom deformation scale must be finite and non-negative"
                )
            return scale
        if mode != "auto":
            raise ValueError("unknown deformation scale mode")
        topology = provider.snapshot.topology
        coordinates = topology.node_coordinates
        displacements = topology.nodal_displacements
        if len(coordinates) == 0:
            return 1.0
        span = float(np.linalg.norm(np.ptp(coordinates, axis=0)))
        maximum = float(
            np.max(np.linalg.norm(displacements, axis=1))
        )
        return (
            1.0
            if maximum <= 0.0 or span <= 0.0
            else 0.1 * span / maximum
        )

    def _build_result_render_payload(
        self,
        provider: ResultProvider,
        selection: ScalarFieldSelection,
        *,
        shape_mode: str | None = None,
        scale_mode: str | None = None,
        scale_value: float | None = None,
    ) -> ResultRenderPayload:
        availability = self._catalog_availability_for_selection(
            provider,
            selection,
        )
        if availability.state is not FieldState.READY:
            raise KeyError("only a READY catalog field can be rendered")
        export = prepare_result_export_snapshot(
            provider.snapshot,
            selection,
        )
        topology = project_scalar_field_topology(
            export,
            deformation_scale=self._result_deformation_scale(
                provider,
                shape_mode=shape_mode,
                scale_mode=scale_mode,
                scale_value=scale_value,
            ),
        )
        return build_result_render_payload(topology)

    def _install_ready_result_selection(
        self,
        provider: ResultProvider,
        selection: ScalarFieldSelection,
    ) -> None:
        if provider is not self._current_result_provider():
            raise RuntimeError("provider is no longer current")
        payload = self._build_result_render_payload(
            provider,
            selection,
        )
        if not self.result_tree.has_selection(selection):
            raise RuntimeError(
                "selected field is missing from the result tree"
            )
        self._install_viewport_result_payload(
            payload,
            shape_mode=self._display.shape_mode,
            contour_enabled=self._display.contour_enabled,
        )
        self.result_selection = selection
        if not self.result_tree.select_selection(selection):
            raise RuntimeError(
                "selected field disappeared from the result tree"
            )
        self._refresh_result_controls()
        self.status_panel.set_result(self._result_status_text())
        self._update_action_states()

    def _install_viewport_result_payload(
        self,
        payload: ResultRenderPayload,
        *,
        shape_mode: str,
        contour_enabled: bool,
    ) -> None:
        """Install one payload and restore the prior scene on renderer failure."""

        previous_payload = self.viewport._result_render_payload
        previous_display = self.viewport._display
        try:
            self.viewport.set_result_render_payload(payload)
            self.viewport.set_display(
                shape_mode,
                contour_enabled,
            )
        except Exception:
            if previous_payload is not None:
                try:
                    FEMViewport.set_result_render_payload(
                        self.viewport,
                        previous_payload,
                    )
                    FEMViewport.set_display(
                        self.viewport,
                        previous_display.shape_mode,
                        previous_display.contour_enabled,
                    )
                except Exception:
                    logging.exception(
                        "failed to restore viewport result payload"
                    )
            else:
                try:
                    self._restore_viewport_model_scene()
                except Exception:
                    logging.exception(
                        "failed to restore the model viewport scene"
                    )
            raise

    def _restore_viewport_model_scene(self) -> None:
        artifact = self.document.artifact
        geometry = self.geometry
        if artifact is None or geometry is None:
            FEMViewport.clear_model(self.viewport)
            return
        FEMViewport.set_model(
            self.viewport,
            artifact.model,
            geometry,
            refresh_symbols=False,
            render=False,
        )
        FEMViewport.set_symbol_settings(
            self.viewport,
            self._symbol_settings,
            refresh=False,
            render=False,
        )
        FEMViewport.show_boundary_and_loads(
            self.viewport,
            render=False,
        )
        FEMViewport.render(self.viewport)

    def _activate_result_selection(
        self,
        selection: ScalarFieldSelection,
    ) -> None:
        provider = self._current_result_provider()
        receipt = self.select_result_field(selection)
        if receipt.diagnostic is not None:
            current = self._current_result_provider()
            installed = self.result_selection
            if (
                current is provider
                and type(installed) is ScalarFieldSelection
            ):
                self.result_tree.select_selection(installed)
            self.status_panel.set_state(
                receipt.diagnostic.message,
                5000,
            )
            self._refresh_result_controls()
            return
        if receipt.status is GuiCommandStatus.PENDING:
            completion = receipt.completion
            if completion is not None and provider is not None:
                def finished(terminal: TaskCompletion) -> None:
                    self._finish_activated_result_selection(
                        selection,
                        provider.source,
                        completion,
                        terminal,
                    )

                completion.observe(finished)
            self.status_panel.set_state(
                "正在按需加载结果字段……",
                4000,
            )
            return
        self._display = replace(
            self._display,
            contour_enabled=True,
        )
        self.actions["contour"].setChecked(True)
        self.viewport.set_display(
            self._display.shape_mode,
            True,
        )
        self.status_panel.set_result(self._result_status_text())

    def _finish_activated_result_selection(
        self,
        selection: ScalarFieldSelection,
        source: ResultSourceKey,
        completion: GuiCommandCompletion,
        terminal: TaskCompletion,
    ) -> None:
        provider = self._current_result_provider()
        outcome = completion.outcome
        if (
            terminal.state is not BackgroundTaskState.SUCCEEDED
            or terminal.projection_error is not None
            or type(outcome) is not GuiCommandOutcome
            or outcome.source != source
            or provider is None
            or provider.source != source
            or provider.snapshot.generation
            != outcome.materialization_generation
            or self.result_selection != selection
        ):
            return
        self._display = replace(
            self._display,
            contour_enabled=True,
        )
        self.actions["contour"].setChecked(True)
        self.viewport.set_display(
            self._display.shape_mode,
            True,
        )
        self.status_panel.set_result(self._result_status_text())

    def set_shape_mode(self, shape_mode: str) -> None:
        if self._current_result_provider() is None:
            return
        shape = "deformed" if shape_mode == "deformed" else "undeformed"
        self._display = replace(self._display, shape_mode=shape)
        self.actions[shape].setChecked(True)
        self._apply_display()

    def _toggle_contour(self, checked: bool) -> None:
        if self._current_result_provider() is None:
            return
        self._display = replace(self._display, contour_enabled=bool(checked))
        self._apply_display()

    def _toggle_undeformed_overlay(self, checked: bool) -> None:
        self._overlay_undeformed = bool(checked)
        self.viewport.set_undeformed_overlay_visible(checked)

    def _apply_display(self) -> None:
        provider = self._current_result_provider()
        selection = self.result_selection
        if (
            provider is None
            or type(selection) is not ScalarFieldSelection
        ):
            return
        payload = self._build_result_render_payload(
            provider,
            selection,
        )
        show_edges = (
            bool(self._contour_options["edges"])
            if self._display.contour_enabled
            else self._model_edges_visible
        )
        self.actions["edges"].setChecked(show_edges)
        self.viewport.set_edges_visible(show_edges, render=False)
        self.viewport.set_result_render_payload(payload)
        self.viewport.set_display(
            self._display.shape_mode,
            self._display.contour_enabled,
        )
        self.status_panel.set_result(self._result_status_text())

    def _result_status_text(self) -> str:
        provider = self._current_result_provider()
        selection = self.result_selection
        if (
            provider is None
            or type(selection) is not ScalarFieldSelection
        ):
            return "—"
        shape = "变形" if self._display.shape_mode == "deformed" else "未变形"
        if not self._display.contour_enabled:
            return f"{shape} / 无云图"
        try:
            availability = self._catalog_availability_for_selection(
                provider,
                selection,
            )
        except (KeyError, TypeError, ValueError):
            return f"{shape} / 云图"
        field_id = availability.descriptor.field_id
        result_name = (
            f"{field_id.variable.value} {selection.component}"
            f"（{field_id.position.value}）"
        )
        return f"{shape} / {result_name}"

    def show_result_display_dialog(self) -> None:
        provider = self._current_result_provider()
        selection = self.result_selection
        if (
            provider is None
            or type(selection) is not ScalarFieldSelection
        ):
            return
        dialog = TypedResultDisplayDialog(
            provider.catalog(),
            current_selection=selection,
            shape_mode=self._display.shape_mode,
            contour_enabled=self._display.contour_enabled,
            scale_mode=self._scale_mode,
            scale_value=self._scale_value,
            overlay_undeformed=self._overlay_undeformed,
            show_edges=self.actions["edges"].isChecked(),
            parent=self,
        )
        dialog.applyRequested.connect(
            lambda settings, source=provider.source: (
                self._apply_typed_result_display_settings(
                    settings,
                    expected_source=source,
                )
            )
        )
        self._exec_dialog(dialog)

    def _apply_typed_result_display_settings(
        self,
        settings: TypedResultDisplaySettings,
        *,
        expected_source: ResultSourceKey | None = None,
        _materialization_completion: bool = False,
    ) -> None:
        if type(settings) is not TypedResultDisplaySettings:
            raise TypeError(
                "settings must be TypedResultDisplaySettings"
            )
        if (
            expected_source is not None
            and type(expected_source) is not ResultSourceKey
        ):
            raise TypeError(
                "expected_source must be ResultSourceKey or None"
            )
        if self.busy and not _materialization_completion:
            self.status_panel.set_state(
                "结果任务正在运行，请等待完成后再应用显示设置",
                5000,
            )
            return
        provider = self._current_result_provider()
        if provider is None:
            return
        if (
            expected_source is not None
            and provider.source != expected_source
        ):
            self.status_panel.set_state(
                "结果已切换，请重新打开显示设置",
                5000,
            )
            return
        try:
            availability = self._catalog_availability_for_selection(
                provider,
                settings.selection,
            )
        except (KeyError, TypeError, ValueError) as error:
            self._show_error("结果显示失败", str(error))
            return
        if availability.state is not FieldState.READY:
            receipt = self.select_result_field(settings.selection)
            if receipt.diagnostic is not None:
                self.status_panel.set_state(
                    receipt.diagnostic.message,
                    5000,
                )
            elif (
                receipt.status is GuiCommandStatus.PENDING
                and receipt.completion is not None
            ):
                receipt.completion.observe(
                    lambda terminal, value=settings, source=provider.source: (
                        self._finish_typed_result_display_materialization(
                            terminal,
                            value,
                            source,
                        )
                    )
                )
            return
        try:
            payload = self._build_result_render_payload(
                provider,
                settings.selection,
                shape_mode=settings.shape_mode,
                scale_mode=settings.scale_mode,
                scale_value=settings.scale_value,
            )
        except (KeyError, RuntimeError, TypeError, ValueError) as error:
            self._show_error("结果显示失败", str(error))
            return
        if not self.result_tree.has_selection(settings.selection):
            self._show_error(
                "结果显示失败",
                "selected field is missing from the result tree",
            )
            return

        try:
            self._install_viewport_result_payload(
                payload,
                shape_mode=settings.shape_mode,
                contour_enabled=settings.contour_enabled,
            )
        except (RuntimeError, TypeError, ValueError) as error:
            self._show_error("结果显示失败", str(error))
            return

        self.result_selection = settings.selection
        self._display = DisplayState(
            settings.shape_mode,
            settings.contour_enabled,
        )
        self._scale_mode = settings.scale_mode
        self._scale_value = settings.scale_value
        self._overlay_undeformed = settings.overlay_undeformed
        if not self.result_tree.select_selection(settings.selection):
            raise RuntimeError(
                "selected field disappeared from the result tree"
            )
        self.viewport.set_undeformed_overlay_visible(
            settings.overlay_undeformed
        )
        self.viewport.set_edges_visible(
            settings.show_edges,
            render=False,
        )
        self.actions[settings.shape_mode].setChecked(True)
        self.actions["contour"].setChecked(
            settings.contour_enabled
        )
        self.actions["overlay"].setChecked(
            settings.overlay_undeformed
        )
        if settings.contour_enabled:
            self._contour_options["edges"] = settings.show_edges
        else:
            self._model_edges_visible = settings.show_edges
        self.actions["edges"].setChecked(settings.show_edges)
        self.result_scale_combo.setCurrentIndex(
            max(
                0,
                self.result_scale_combo.findData(settings.scale_mode),
            )
        )
        self.result_scale_value.setValue(settings.scale_value)
        self.result_scale_value.setEnabled(
            settings.scale_mode == "custom"
        )
        self._refresh_result_controls()
        self.status_panel.set_result(self._result_status_text())
        self._update_action_states()

    def _finish_typed_result_display_materialization(
        self,
        terminal: TaskCompletion,
        settings: TypedResultDisplaySettings,
        source: ResultSourceKey,
    ) -> None:
        if (
            terminal.state is BackgroundTaskState.SUCCEEDED
            and terminal.projection_error is None
        ):
            self._apply_typed_result_display_settings(
                settings,
                expected_source=source,
                _materialization_completion=True,
            )

    def _apply_scale(self) -> None:
        if self._current_result_provider() is None:
            return
        self._apply_display()

    def show_contour_dialog(self) -> None:
        if self._current_result_provider() is None:
            return
        dialog = ContourSettingsDialog(dict(self._contour_options), self)
        dialog.applyRequested.connect(self._set_contour_options)
        self._exec_dialog(dialog)

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
        self._exec_dialog(dialog)

    def show_viewport_background_dialog(self) -> None:
        """打开可实时预览的视口背景设置。"""
        dialog = ViewportBackgroundDialog(
            self._background_settings,
            self._remember_background,
            self,
        )
        dialog.previewRequested.connect(self.viewport.set_background_settings)
        dialog.applyRequested.connect(self._apply_background_settings)
        self._exec_dialog(dialog)

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

    def show_result_query_dialog(self) -> None:
        """Open a catalog-only query dialog without eager recovery."""

        provider = self._current_result_provider()
        if provider is None:
            return
        source = provider.source
        dialog = TypedResultQueryDialog(provider, parent=self)
        closed = False
        active_query: ResultQuery | None = None

        def deliver_result(result: object) -> None:
            if (
                closed
                or type(result) is not ResultQueryResult
                or result.query != active_query
            ):
                return
            current = self._current_result_provider()
            if (
                current is None
                or current.source != source
                or result.source != source
                or current.snapshot.generation
                != result.materialization_generation
            ):
                return
            dialog.set_query_result(result)

        def submit_query(query: object) -> None:
            nonlocal active_query
            if type(query) is not ResultQuery:
                return
            active_query = query
            dialog.set_query_pending(True)
            receipt = self._submit_result_query(
                query,
                expected_source=source,
            )
            if receipt.status is GuiCommandStatus.REJECTED:
                dialog.set_query_pending(False)
                diagnostic = receipt.diagnostic
                if diagnostic is not None:
                    dialog.set_query_message(diagnostic.message)
                    self.status_panel.set_state(
                        diagnostic.message,
                        5000,
                    )
                return
            if receipt.status is GuiCommandStatus.ACCEPTED:
                dialog.set_query_pending(False)
                return

            completion = receipt.completion
            if completion is None:
                dialog.set_query_pending(False)
                return

            def query_finished(terminal: TaskCompletion) -> None:
                if closed:
                    return
                dialog.set_query_pending(False)
                if terminal.state is not BackgroundTaskState.SUCCEEDED:
                    dialog.set_query_message(
                        terminal.message or "结果查询未完成"
                    )

            completion.observe(query_finished)

        self.resultQueryCompleted.connect(deliver_result)
        dialog.queryRequested.connect(submit_query)
        try:
            self._exec_dialog(dialog)
        finally:
            closed = True
            try:
                self.resultQueryCompleted.disconnect(deliver_result)
            except (RuntimeError, TypeError):
                pass

    def export_csv(self) -> None:
        export_identity = self._current_result_export_identity("导出 CSV 失败")
        if export_identity is None:
            return
        source, generation, selection, field_key = export_identity
        stem = self.document.path.stem if self.document.path else "result"
        safe_field = field_key.replace(":", "_")
        default = f"{stem}_{safe_field}.csv"
        path, _filter = QFileDialog.getSaveFileName(
            self, "导出当前结果字段", default, "CSV 文件 (*.csv)"
        )
        if not path:
            return
        target = Path(path).with_suffix(".csv")
        self.status_panel.set_state("正在导出 CSV……")
        receipt = self.export_result_csv(
            target,
            ResultCsvExportSpec(
                source,
                generation,
                selection,
            ),
        )
        if receipt.diagnostic is not None:
            self._show_command_rejection("导出 CSV 失败", receipt)

    def export_vtk(self) -> None:
        export_identity = self._current_result_export_identity("导出 VTK 失败")
        if export_identity is None:
            return
        source, generation, selection, field_key = export_identity
        stem = self.document.path.stem if self.document.path else "result"
        safe_field = field_key.replace(":", "_")
        default = f"{stem}_{safe_field}.vtk"
        path, _filter = QFileDialog.getSaveFileName(
            self,
            "导出当前结果字段",
            default,
            "VTK 文件 (*.vtk)",
        )
        if not path:
            return
        target = Path(path).with_suffix(".vtk")
        self.status_panel.set_state("正在导出 VTK……")
        receipt = self.export_result_vtk(
            target,
            ResultVtkExportSpec(
                source,
                generation,
                selection,
                self._current_result_export_deformation_scale(),
            ),
        )
        if receipt.diagnostic is not None:
            self._show_command_rejection("导出 VTK 失败", receipt)

    def _current_result_export_identity(
        self,
        error_title: str,
    ) -> tuple[ResultSourceKey, int, ScalarFieldSelection, str] | None:
        provider = self._current_result_provider()
        selection = self.result_selection
        if (
            provider is None
            or type(selection) is not ScalarFieldSelection
        ):
            return None
        try:
            availability = self._catalog_availability_for_selection(
                provider,
                selection,
            )
        except (KeyError, TypeError, ValueError) as error:
            self._show_error(error_title, str(error))
            return None
        if availability.state is not FieldState.READY:
            self._show_error(error_title, "当前结果字段尚未就绪")
            return None
        field_id = availability.descriptor.field_id
        field_label = "_".join(
            (
                field_id.variable.value,
                field_id.position.value,
                selection.component,
            )
        )
        return (
            provider.source,
            provider.snapshot.generation,
            selection,
            field_label,
        )

    def _current_result_export_deformation_scale(self) -> float:
        provider = self._current_result_provider()
        if provider is None:
            return 0.0
        return self._result_deformation_scale(provider)

    def export_viewport_image(self) -> None:
        if self._current_result_provider() is None:
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
            self._show_information("模型概况", [
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
                ("材料数量", len(self.document.materials)),
                ("截面数量", len(self.document.sections)),
                ("分析步数量", len(self.document.steps)),
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
        self._show_information(
            "关于",
            [
                ("软件", "有限元分析"),
                ("功能", "Abaqus INP 线性静力分析与结果查看"),
                ("界面", "PySide6、PyVistaQt、VTK"),
            ],
        )

    def _show_entry_information(self, kind: str, key: object) -> None:
        if kind == "mesh":
            self.show_mesh_browser()
        else:
            self.show_entity_information(kind, key)

    def _edit_tree_entry(self, kind: str, key: object) -> None:
        if kind == "material":
            self.edit_material(str(key))
        elif kind == "assignment":
            self.edit_region_assignment(int(key))
        elif kind in {
            "step",
            "boundary",
            "cload",
            "edge_load",
            "surface_load",
            "line_load",
            "gravity_load",
            "output",
        }:
            self.edit_analysis_definition(kind, key)
        else:
            self.show_entity_information(kind, key)

    def show_entity_information(self, kind: str, key: object) -> EntityInfoDialog | None:
        if self.inspection_service is None:
            return
        dialog = EntityInfoDialog(self.inspection_service.inspect(kind, key), self)
        dialog.highlightRequested.connect(self.highlight_entity)
        dialog.locateRequested.connect(self.locate_entity)
        dialog.entityRequested.connect(self.show_entity_information)
        self._fit_viewport_when_dialog_finishes(dialog)
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
        self._fit_viewport_when_dialog_finishes(dialog)
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
        beam_frame_target: RegionRef | int | None = None
        if kind == "element":
            beam_frame_target = int(key)
        elif kind == "element_set":
            beam_frame_target = RegionRef("element_set", str(key))
        elif kind == "assignment":
            assignment = self.document.assignments[int(key)]
            beam_frame_target = RegionRef(
                "element_set",
                str(assignment.region_name),
            )
        elif kind == "line_load":
            step_index, load_index = key
            target = self.document.model.steps[
                int(step_index)
            ].line_loads[int(load_index)].target
            beam_frame_target = (
                int(target)
                if isinstance(target, int)
                else RegionRef("element_set", str(target))
            )
        selection = self.inspection_service.selection_for(kind, key)
        if len(selection.node_ids) == 1 and not selection.element_ids:
            self._on_viewport_pick("node", selection.node_ids[0])
        elif len(selection.element_ids) == 1 and not selection.node_ids:
            self._on_viewport_pick("element", selection.element_ids[0])
        elif selection.node_ids:
            self.viewport.highlight_nodes(selection.node_ids)
        elif selection.element_ids:
            self.viewport.highlight_elements(selection.element_ids)
        if (
            beam_frame_target is not None
            and len(selection.element_ids) != 1
        ):
            self.viewport.show_beam_frame_preview(
                beam_frame_target
            )

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
                self.cancel_current_task(
                    after_cleanup=lambda: QTimer.singleShot(0, self.close)
                )
            event.ignore()
            return
        if self.isVisible() and not self._confirm_discard_changes():
            event.ignore()
            return
        self._close_inspection_windows()
        self._close_job_manager()
        if self.document.is_open:
            receipt = self.close_session(
                CloseSessionCommand(self.document.session_revision)
            )
            if receipt.diagnostic is not None:
                event.ignore()
                return
        event.accept()
