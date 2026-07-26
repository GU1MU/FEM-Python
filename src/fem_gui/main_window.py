"""第一版中文有限元主窗口。"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import logging
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

from PySide6.QtCore import QSettings, Qt, QTimer
from PySide6.QtGui import QAction, QCloseEvent
from PySide6.QtWidgets import (
    QApplication, QComboBox, QFileDialog, QGridLayout,
    QLabel, QMainWindow, QMessageBox, QSizePolicy, QSplitter, QVBoxLayout, QWidget,
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
    describe_native_regions,
    describe_model_capabilities,
    describe_native_authoring_capabilities,
    describe_session_authoring,
    evaluate_authoring_candidate,
    resolve_effective_beam_frames,
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
    geometry_dimension,
    logical_ref_sort_key,
    recipe_characteristic_size,
    supports_structured_hexahedron,
)
from fem.io.project import load_project, save_project
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
    StaticStepDialog,
)
from .commands import (
    CloseSessionCommand,
    GuiCommandCompletion,
    GuiCommandDiagnostic,
    GuiCommandReceipt,
    MeshInputEdit,
    NativeGeometryEdit,
    NewNativeProjectCommand,
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
from .postprocessing_dialogs import (
    ContourSettingsDialog,
    ResultDisplayDialog,
    ResultDisplaySettings,
    ResultQueryDialog,
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
from .workers import TaskContext


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
        self.result_data: ResultData | None = None
        self.inspection_service: InspectionService | None = None
        self._inspection_windows: list[QWidget] = []
        self._mesh_browser: MeshBrowserDialog | None = None
        self._selected_geometry_refs: set[LogicalEntityRef] = set()
        self._geometry_selection_mode = "body"
        self._pending_local_mesh_selection = False
        self._pending_analysis_selection: str | None = None
        self.selection = SelectionState()
        self.actions: dict[str, QAction] = {}
        self.task_controller = BackgroundTaskController(self)
        self._command_counter = 0
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
            core_result = current_result.result
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
        self.viewport.geometryEntityPicked.connect(
            self._on_geometry_entity_pick
        )
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
        authoring = describe_session_authoring(self.document)
        selection_kind = (
            "node"
            if self.selection.node_id is not None
            else "element"
            if self.selection.element_id is not None
            else None
        )
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
        has_result = self.document.displayed_result is not None
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
                f"；{prior_region_count} 个旧命名区域已失效，"
                "请重新选择同名区域"
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
        self._create_region_from_current_geometry_selection()

    def _create_region_from_current_geometry_selection(self) -> str | None:
        references = self._canonical_geometry_selection()
        if not references:
            return None
        kind = references[0].kind
        for region in self.document.named_regions.values():
            if region.references == references:
                return region.name
        dialog = NamedRegionDialog(
            references,
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
        base_revision = self.document.session_revision
        try:
            regions[name] = NamedRegion(name, references)
            batch = NamedRegionEditBatch(
                base_session_revision=base_revision,
                regions=tuple(regions.values()),
            )
        except (TypeError, ValueError) as error:
            self._show_error("创建命名区域", str(error))
            return None
        receipt = self.apply_named_region_edit(batch)
        if receipt.diagnostic is not None:
            self._show_command_rejection("创建命名区域", receipt)
            return None
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

    def _native_region_catalog(self) -> tuple[Any, ...]:
        recipe = self.document.geometry_recipe
        if not isinstance(recipe, NATIVE_GEOMETRY_TYPES):
            return ()
        return describe_native_regions(
            recipe,
            self.document.named_regions,
        )

    def _native_region_names(self, product: str) -> list[str]:
        return [
            descriptor.name
            for descriptor in self._native_region_catalog()
            if product in descriptor.products
        ]

    def show_named_region_manager(self) -> None:
        if not self.document.named_regions:
            return
        base_revision = self.document.session_revision
        dialog = NamedRegionManagerDialog(
            dict(self.document.named_regions),
            self,
        )
        if not dialog.exec():
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
            self._show_error("命名区域管理", str(error))
            return
        receipt = self.apply_named_region_edit(batch)
        if receipt.diagnostic is not None:
            self._show_command_rejection("命名区域管理", receipt)
            return
        self.status_panel.set_state(
            "命名区域已更新，请重新生成网格",
            5000,
        )

    def _analysis_region_names(
        self,
    ) -> tuple[list[RegionRef], list[RegionRef], list[RegionRef]]:
        targets = describe_session_authoring(self.document).targets
        return tuple(
            [target.region for target in targets if target.region.kind == kind]
            for kind in ("node_set", "edge", "surface")
        )

    def _analysis_element_regions(
        self,
        capability_report: ModelCapabilityReport | None = None,
    ) -> list[RegionRef]:
        del capability_report
        return [
            target.region
            for target in describe_session_authoring(self.document).targets
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

        projection = describe_session_authoring(self.document)
        operations = (
            ("node_set", "load.node"),
            ("edge", "load.edge"),
            ("surface", "load.surface"),
            ("element_set", "load.line.global"),
        )
        return tuple(
            [
                target.region
                for target in projection.targets
                if target.region.kind == kind
                and target.operation(operation).can_submit
            ]
            for kind, operation in operations
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
                recipe_characteristic_size(recipe) / 10.0,
                cell_shape="tetrahedron" if geometry_dimension(recipe) == 3 else "triangle",
            )
        dialog = MeshSettingsDialog(
            current,
            self,
            mesh_dimension=geometry_dimension(recipe),
            allow_hexahedron=supports_structured_hexahedron(recipe),
        )
        if not dialog.exec():
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
        if not dialog.exec():
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
        supported_kinds = {"point", "edge", "face"}
        selected_references = self._canonical_geometry_selection()
        if (
            not selected_references
            or selected_references[0].kind not in supported_kinds
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
        dialog = LocalMeshControlDialog(
            selected_references[0],
            settings.size,
            self,
        )
        if not dialog.exec():
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
        policy = initial_display_policy(element_count, node_count)
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
            return self.save_native_project()
        return clicked is discard_button

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
        if not dialog.exec():
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
        if not dialog.exec():
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
        dialog = self._region_assignment_dialog()
        if dialog is None or not dialog.exec():
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
        if dialog is None or not dialog.exec():
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
        if not regions:
            self._show_error("截面分配", "当前模型没有可分配的单元区域")
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
        model = self.document.model
        if model is None:
            raise RuntimeError("region assignment candidate requires a model")
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
        definitions = ModelDefinitions(
            materials=tuple(self.document.materials),
            sections=tuple(self.document.sections),
            assignments=tuple(self.document.assignments),
            steps=tuple(self.document.steps),
        )
        return evaluate_authoring_candidate(
            model,
            definitions,
            operation=f"section.{section_type}",
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
        selected_geometry_kind = self._geometry_selection_kind()
        if (
            self.document.source_kind == "native"
            and selected_geometry_kind in {"point", "edge", "face"}
        ):
            selected_name = self._create_region_from_current_geometry_selection()
            if selected_name is None:
                return
            selected_region = RegionRef("node_set", selected_name)
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
        selected_region = None
        preferred_kind = None
        capability_report = self._model_capability_report()
        if capability_report is None:
            return
        selected_geometry_kind = self._geometry_selection_kind()
        if (
            self.document.source_kind == "native"
            and selected_geometry_kind in {"point", "edge", "face"}
        ):
            preferred_kind = {
                "point": "node",
                "edge": "edge",
                "face": "surface",
            }[selected_geometry_kind]
            if preferred_kind not in capability_report.load_kinds:
                self._show_error(
                    "创建载荷",
                    "所选区域不支持当前模型的分布载荷契约。",
                )
                return
            selected_name = self._create_region_from_current_geometry_selection()
            if selected_name is None:
                return
            selected_region = RegionRef(
                {
                    "node": "node_set",
                    "edge": "edge",
                    "surface": "surface",
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
        )
        if not dialog.exec():
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
        self._show_error(
            "输出请求",
            "当前求解链不会执行输出请求，因此不能新建；"
            "既有请求仍可在分析定义管理中查看或删除。",
        )

    def _analysis_manager_dialog(
        self,
    ) -> AnalysisDefinitionManagerDialog | None:
        if not self.document.steps:
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
            raise RuntimeError("line-load candidate requires a model")
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
        if not dialog.exec():
            return
        values = dialog.values()
        if tuple(values) == tuple(
            self.document.steps
        ):
            return
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
        if dialog.exec():
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

        def apply_result(value: object) -> TaskApplyOutcome:
            if len(value) == 2:
                result, data = value
                timings: dict[str, float] = {}
            else:
                result, data, timings = value
            data = replace(
                data,
                artifact_id=task.token.artifact_id,
                run_id=task.token.run_id,
            )
            delta = self.session.accept_run_result(
                task.token,
                result,
                timings=timings,
            )
            return self._session_task_outcome(
                delta,
                (data, timings),
            )

        def project_result(value: object) -> None:
            delta, payload = value
            data, timings = payload
            if not self._apply_session_delta(
                delta,
                result_projection=data,
            ):
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
            dialog = JobManagerDialog(self.document.runs, self)
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

    def _set_selection_mode(self, mode: str) -> None:
        normalized = "element" if mode == "element" else "node"
        if (
            self._selected_geometry_refs
            or self.viewport._selection_mode != normalized
        ):
            self.selection.clear()
            self._selected_geometry_refs.clear()
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

    def clear_selection(self) -> None:
        self.selection.clear()
        self._pending_local_mesh_selection = False
        self._pending_analysis_selection = None
        self._selected_geometry_refs.clear()
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
        references = self._canonical_geometry_selection()
        self.viewport.highlight_geometry_entities(references)
        mode = f"geometry_{reference.kind}"
        labels = {
            "point": "点",
            "edge": "边",
            "face": "面",
            "body": "体",
        }
        self.status_panel.set_selection_mode(mode)
        selected_count = len(references)
        self.status_panel.set_object(
            (
                f"已选择 {selected_count} 个"
                f"{labels.get(reference.kind, '几何实体')}"
            )
            if selected_count
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
        else:
            self.selection.clear()
        self.viewport.clear_selection()
        self.status_panel.set_object()
        self.actions["selected_info"].setEnabled(False)
        self._update_action_states()

    def _confirm_guided_selection(self) -> None:
        if not self._selected_geometry_refs:
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
            delta, updated = value
            if not self._apply_session_delta(
                delta,
                result_projection=updated,
            ):
                raise RuntimeError("已接受的应力恢复结果无法投影")
            if (
                self.result_data is None
                or self.result_data.run_id != projection.run_id
            ):
                return
            self._refresh_job_manager()
            self.status_panel.set_state("应力结果恢复完成", 4000)
            on_ready()

        def apply_result(value: object) -> TaskApplyOutcome:
            updated, _seconds = value
            return self._session_task_outcome(
                self.session.accept_result_projection(projection.token),
                updated,
            )

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
            apply_result=apply_result,
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
        show_information(self, "关于", [("软件", "有限元分析"), ("功能", "Abaqus INP 线性静力分析与结果查看"), ("界面", "PySide6、PyVistaQt、VTK")])

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
