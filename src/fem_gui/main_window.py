"""第一版中文有限元主窗口。"""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass, replace
import logging
from pathlib import Path
from time import perf_counter, sleep
from typing import Any, Callable
from weakref import WeakSet

import numpy as np
from PySide6.QtCore import QSignalBlocker, QSize, QSettings, Qt, QTimer, Signal
from PySide6.QtGui import QAction, QCloseEvent
from PySide6.QtWidgets import (
    QApplication, QComboBox, QDialog, QFileDialog, QGridLayout,
    QHBoxLayout, QInputDialog, QLabel, QMainWindow, QMessageBox, QSizePolicy,
    QRubberBand, QSplitter,
    QVBoxLayout, QWidget,
)

from fem import geometry as geometry_runtime
from fem.io.inp import read_with_report as read_inp_with_report
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
    NamedRegionEditTaskSnapshot,
    PreparedNamedRegionEdit,
    PreflightDiagnostic,
    PreparedPreflight,
    PreflightReport,
    RegionRef,
    RegionAssignment,
    RenameIntent,
    RevisionConflictError,
    RunStatus,
    SessionAuthoringProjection,
    SessionDelta,
    StrictBodyBooleanPreview,
    StrictPartBooleanResult,
    StrictPlanarBooleanPreview,
    StrictPlanarBooleanResult,
    FaceSketchBooleanResult,
    TokenStatus,
    TransitionEffect,
    describe_model_capabilities,
    describe_session_authoring,
    evaluate_authoring_candidate,
    evaluate_native_assignment_candidate,
    evaluate_native_line_load_candidate,
    derive_geometry_feature_rows,
    resolve_effective_beam_frames,
    safe_static_preflight,
    safe_prepare_static_preflight,
    compile_named_region_edit,
    prepare_part_boolean,
    prepare_strict_part_recipe_preview,
    prepare_strict_body_recipe_preview,
    prepare_planar_boolean,
    prepare_face_sketch_boolean,
    prepare_strict_planar_recipe_preview,
)
from fem.application.definitions import mesh_entity_ref_sort_key
from fem.application.preprocessing import generate_fem_model
from fem.application.recipe_compiler import compile_recipe
from fem.application.changes import ChangeKind
from fem.application.results import (
    FieldAvailability,
    FieldMaterializationKey,
    FieldPosition,
    FieldRequest,
    FieldState,
    NodalAveragingPolicy,
    OutputRequestProjection,
    ResultCapabilityCatalog,
    ResultCatalog,
    ResultQuery,
    ResultQueryResult,
    ResultQueryValidationError,
    ResultExportSnapshot,
    ResultFieldId,
    ResultFieldTopologyTemplate,
    ResultMaterializationPatch,
    ResultProvider,
    ResultSourceKey,
    ResultVariable,
    ScalarFieldSelection,
    build_solve_result_bundle,
    build_result_field_topology_template,
    classify_result_model,
    prepare_result_export_snapshot,
    project_output_requests,
    project_scalar_field_topology,
    project_scalar_field_topology_from_template,
)
from fem.core.model import (
    AnalysisStep,
    BodyForce,
    EdgeLoad,
    GravityLoad,
    LineLoad,
    NodalLoad,
    OutputRequest,
    SurfaceLoad,
)
from fem.core._constraint_targets import displacement_target_kind
from fem.geometry import (
    BooleanGeometry,
    BoxGeometry,
    CylinderGeometry,
    DiskGeometry,
    ExtrudedGeometry,
    ExtrusionSourceResolutionError,
    LogicalEntityRef,
    FaceSketchBooleanDirection,
    FaceSketchBooleanGeometry,
    FaceSketchBooleanOperation,
    FaceWorkplaneResolutionError,
    MovedGeometry,
    MultiBodyGeometry,
    NATIVE_GEOMETRY_TYPES,
    PlateWithHoleGeometry,
    PathSweptGeometry,
    RectangleGeometry,
    RevolvedGeometry,
    RotatedGeometry,
    SketchGeometry,
    SketchReferencePoint,
    WireGeometry,
    geometry_dimension,
    analyze_sketch_profiles,
    logical_ref_sort_key,
    recipe_characteristic_size,
    resolve_extrusion_source_faces,
    supports_structured_hexahedron,
    add_solid_body,
    delete_solid_body,
    part_id_from_logical_id,
    namespace_part_logical_id,
    strip_part_logical_id,
    rename_solid_body,
    provide_face_reference_points,
    resolve_face_workplane,
    undo_solid_body_feature,
)
from fem.geometry.gmsh_coordinator import GmshExecutionCancelled
from fem.geometry.part_namespace import part_id_sort_key
from fem.io.project import (
    LEGACY_MODEL_FILE_SUFFIXES,
    MODEL_FILE_SUFFIX,
    LoadedProject,
    load_project,
    save_project,
)
from fem.io.result_csv import write_result_table_csv
from fem.io.result_vtk import write_result_vtk
from fem.io.result_archive import (
    LoadedResultArchive,
    RESULT_FILE_SUFFIX,
    load_result_archive,
    save_result_archive,
)
from fem.mesh.quality import analyze_mesh
from fem.mesh.settings import MeshSettings
from fem.solvers import static_linear

from .actions import build_actions
from .action_state import GuiActionContext, derive_action_availability
from .agent_authoring import (
    AgentAuthoringBridge,
    AgentGeometryMutation,
    AgentMeshTaskRequest,
    AgentPreflightState,
    AgentPreflightTaskRequest,
    AgentResultQueryBridge,
    AgentSolveTaskRequest,
    ProposalState,
    SessionGeometryAuthoringPort,
    SessionResultQueryPort,
    create_session_authoring_workflow_controller,
)
from .part_boolean import PartBooleanController
from .planar_boolean import PlanarBooleanController, planar_reference_points
from .analysis_dialogs import JobManagerDialog, JobSubmitDialog
from .analysis_definition_dialogs import (
    AnalysisDefinitionManagerDialog,
    DisplacementDialog,
    DisplacementDialogState,
    LoadDialog,
    LoadDialogState,
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
    ResultArchiveOpenSpec,
    ResultArchiveSaveSpec,
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
from .model_iteration import ModelIterationService, geometry_edit_policy
from .geometry_preview import (
    GeometryPreview,
    build_geometry_preview,
    build_face_sketch_boolean_committed_preview,
    build_face_sketch_boolean_display,
    build_face_sketch_boolean_result_preview,
    namespace_part_geometry_preview,
    build_strict_body_boolean_previews,
    build_strict_part_boolean_preview,
    build_strict_planar_boolean_preview,
    build_strict_sketch_draft_preview,
)
from .agent_workspace_catalog import create_workspace_catalog_bridge
from .face_sketch_boolean_dialog import (
    FaceSketchBooleanDialog,
    FaceSketchBooleanParameters,
)
from .face_sketch_editor import (
    FaceSketchBooleanFeatureRequest,
    FaceSupportedSketchController,
)
from .part_geometry_preview import (
    build_multi_part_geometry_preview,
    localize_part_geometry_preview,
)
from .scope_selection import (
    MeshSelectionTopology,
    ScopeSelectionTopology,
    build_mesh_selection_topology,
    build_scope_selection_topology,
)
from .postprocessing_dialogs import (
    ContourSettingsDialog,
    DisplaySettingsDialog,
    TypedResultDisplayDialog,
    TypedResultDisplaySettings,
    TypedResultQueryDialog,
)
from .result_presentation import (
    result_field_is_beam_section,
    result_field_position_label,
    result_provider_section_point_labels,
    result_variable_label,
    visible_result_fields,
)
from .result_csv_export_dialog import ResultCsvExportDialog
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
    SweepGeometryDialog,
    GeometryManagerDialog,
    GeometryCreationDialog,
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
from .workspace import (
    DocumentPresentationState,
    FEMWorkspace,
    WorkspaceDocument,
    canonical_path,
)
from .viewport_background import (
    ViewportBackgroundSettings,
    load_background_settings,
    save_background_settings,
)
from .viewport_background_dialog import ViewportBackgroundDialog
from .viewport_image_export_dialog import ViewportImageExportDialog
from .sketch_editor import SketchDraftController, SketchDraftValidationError
from .wire_editor import WireDraftController, WireDraftValidationError
from .visualization.colormaps import ABAQUS_RAINBOW
from .visualization.contour_rendering import (
    CONTOUR_EDGE_ALL,
    CONTOUR_EDGE_GEOMETRY,
    CONTOUR_EDGE_NONE,
    CONTOUR_RENDER_SHADED,
)
from .visualization.model_adapter import (
    ModelGeometry,
    build_model_geometry,
    build_result_archive_geometry,
    build_result_archive_model_view,
    ArchiveModelView,
)
from .visualization.result_renderer import (
    ResultRenderPayload,
    build_result_render_payload,
)
from .visualization.selection import SelectionContextState, SelectionState
from .visualization.scene import DisplayState
from .visualization.symbols import SymbolSettings
from .widgets.navigation_panel import NavigationPanel
from .widgets.model_tree import (
    native_feature_kind_label,
    native_feature_label,
)
from .widgets.boolean_feature_panel import BooleanFeaturePanel
from .widgets.planar_boolean_panel import PlanarBooleanPanel
from .widgets.ribbon import RibbonPage, RibbonWidget
from .widgets.sketch_editor_panel import SketchEditorPanel
from .widgets.wire_editor_panel import WireEditorPanel
from .widgets.status_bar import CAEStatusBar
from .widgets.viewport import (
    FEMViewport,
    _capture_camera_state,
    _restore_camera_state,
)
from .widgets.viewport_toolbar import ViewportPanel
from .workers import TaskContext


_IMPORTED_OUTPUT_REQUEST_WARNING = (
    "此修改只保留在当前 Session；"
    "重新加载原 INP 后会恢复源文件中的输出请求。"
)

_RESULT_FIELD_STATE_LABELS = {
    FieldState.LAZY: "按需加载",
    FieldState.UNAVAILABLE: "不可用",
}
_NUMERICAL_MODEL_CHECK_DOF_LIMIT = 50_000
_NUMERICAL_MODEL_CHECK_ELEMENT_LIMIT = 100_000
_DEFAULT_SCOPE_BACKGROUND_REFERENCE_THRESHOLD = 10_000
_SYNCHRONOUS_GUI_COMMAND_TIMEOUT_SECONDS = 5.0
_TREE_KEY_MISSING = object()


_VISIBLE_OUTPUT_VARIABLES = frozenset({"u", "rf", "s"})


def _native_model_open_filter() -> str:
    """Build the native-model chooser filter from the public suffix contract."""

    suffixes = " ".join(
        f"*{suffix}"
        for suffix in (MODEL_FILE_SUFFIX, *LEGACY_MODEL_FILE_SUFFIXES)
    )
    return f"FEM 自主项目 ({suffixes});;所有文件 (*)"


def _native_model_save_filter() -> str:
    """Build the native-model save filter from the current suffix contract."""

    return f"FEM 自主项目 (*{MODEL_FILE_SUFFIX})"


def _output_request_projections_by_step(
    steps: Sequence[AnalysisStep],
    catalog: ResultCapabilityCatalog | None,
) -> dict[str, tuple[OutputRequestProjection, ...]]:
    if catalog is None:
        return {step.name: () for step in steps}
    return {
        step.name: project_output_requests(tuple(step.outputs), catalog)
        for step in steps
    }


def _executable_output_requests(
    projections: Sequence[OutputRequestProjection],
) -> tuple[OutputRequest, ...]:
    return tuple(
        request
        for projection in projections
        if (request := projection.executable_authoring_request) is not None
    )


def _with_required_displacement_output(
    requests: tuple[OutputRequest, ...],
    candidates: Sequence[object],
) -> tuple[OutputRequest, ...]:
    if any(
        request.kind.casefold() == "field"
        and request.target.casefold() == "node"
        and variable.strip().casefold() == "u"
        for request in requests
        for variable in request.variables
    ):
        return requests
    required = next(
        (
            candidate.authoring_request
            for candidate in candidates
            if candidate.authoring_request.variables == ("U",)
        ),
        None,
    )
    if type(required) is not OutputRequest:
        raise ValueError("当前模型不支持必需的位移场 U 输出")
    return (deepcopy(required), *requests)


def _replace_visible_output_requests(
    requests: tuple[OutputRequest, ...],
    existing: tuple[OutputRequest, ...],
) -> tuple[OutputRequest, ...]:
    preserved: list[OutputRequest] = []
    for request in existing:
        variables = tuple(
            variable
            for variable in request.variables
            if variable.strip().casefold() not in _VISIBLE_OUTPUT_VARIABLES
        )
        if not variables:
            continue
        preserved.append(
            request
            if variables == tuple(request.variables)
            else replace(request, variables=variables)
        )
    return (*preserved, *deepcopy(requests))


def _resolve_analysis_object_key(
    steps: Sequence[AnalysisStep],
    collection_name: str,
    key: object,
) -> tuple[int, int] | None:
    """Resolve legacy indices or stable `(step_name, object_name)` identity."""

    if not isinstance(key, (tuple, list)) or len(key) != 2:
        return None
    step_key, item_key = key
    if type(step_key) is str and type(item_key) is str:
        for step_index, step in enumerate(steps):
            if step.name != step_key:
                continue
            for item_index, item in enumerate(
                tuple(getattr(step, collection_name))
            ):
                if getattr(item, "name", None) == item_key:
                    return step_index, item_index
        return None
    try:
        return int(step_key), int(item_key)
    except (TypeError, ValueError):
        return None


_ANALYSIS_COLLECTION_BY_TREE_KIND = {
    "boundary": "boundaries",
    "cload": "cloads",
    "edge_load": "edge_loads",
    "surface_load": "surface_loads",
    "line_load": "line_loads",
    "body_load": "body_loads",
    "gravity_load": "gravity_loads",
    "output": "outputs",
}


def initial_display_policy(
    element_count: int,
    node_count: int,
    *,
    line_mesh: bool = False,
) -> dict[str, bool]:
    """Return the explicit first-display degradation policy for large models."""
    element_count = int(element_count)
    node_count = int(node_count)
    simplified = element_count > 100_000 or node_count > 200_000
    return {
        "show_edges": element_count <= 100_000,
        "show_symbols": True,
        "reduce_symbols": simplified,
        "show_nodes": bool(line_mesh) and node_count <= 20_000,
        "show_labels": False,
        "simplified": simplified,
    }


def should_run_numerical_model_check(model: object) -> bool:
    """Return whether a full stiffness factorization is suitable for preflight."""

    mesh = getattr(model, "mesh", None)
    if mesh is None:
        return True
    try:
        element_count = len(mesh.elements)
        dof_count = int(mesh.num_dofs)
    except (AttributeError, TypeError, ValueError):
        return True
    return (
        element_count <= _NUMERICAL_MODEL_CHECK_ELEMENT_LIMIT
        and dof_count <= _NUMERICAL_MODEL_CHECK_DOF_LIMIT
    )


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


class _PreferredWidthHost(QWidget):
    """Prefer a wider ribbon section while allowing compact windows."""

    def __init__(
        self,
        preferred_width: int,
        minimum_width: int,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._preferred_width = preferred_width
        self._minimum_width = minimum_width
        self.setMinimumWidth(minimum_width)

    def sizeHint(self) -> QSize:
        hint = super().sizeHint()
        hint.setWidth(max(hint.width(), self._preferred_width))
        return hint

    def minimumSizeHint(self) -> QSize:
        hint = super().minimumSizeHint()
        hint.setWidth(self._minimum_width)
        return hint


@dataclass(frozen=True, slots=True)
class _ResultArchiveDisplayPayload:
    """Worker-owned decoded archive plus display-only structural adapters."""

    loaded: LoadedResultArchive
    geometry: ModelGeometry
    model_view: ArchiveModelView
    timings: dict[str, float]


class FEMMainWindow(QMainWindow):
    """只暴露当前内核已经实现的有限元工作流。"""

    resultQueryCompleted = Signal(object)
    faceSketchBooleanFeatureRequested = Signal(object)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("有限元分析")
        self.resize(1280, 800)
        # Phase 1 keeps the visible GUI single-document while moving ownership
        # of the Session and its task controller into a workspace context.
        # The public ``session``, ``document`` and ``task_controller`` names
        # below remain compatibility aliases for the active context.
        self._session_alias = ModelSession()
        self._document_alias = self._session_alias.projection_snapshot()
        self.workspace = FEMWorkspace(parent=self)
        self._task_callback_context: WorkspaceDocument | None = None
        self._task_controller_connections: WeakSet[BackgroundTaskController] = (
            WeakSet()
        )
        # A context switch batches every viewport mutation and emits one
        # explicit render after the target presentation has been installed.
        self._workspace_activation = False
        self._task_controller_alias = BackgroundTaskController(self)
        self._active_context = self.workspace.add_model(
            session=self._session_alias,
            projection=self._document_alias,
            display_name="模型-1",
            task_controller=self._task_controller_alias,
        )
        self.workspace.activate(self._active_context)
        self.agent_result_query_bridge = AgentResultQueryBridge(
            SessionResultQueryPort(self.session, self.workspace)
        )
        self.agent_authoring_bridge = AgentAuthoringBridge(
            SessionGeometryAuthoringPort(
                self.session,
                self._rebuild_full_projection,
                self._begin_agent_mesh_generation,
                self._apply_agent_definition_delta,
                self._begin_agent_solve,
                self._begin_agent_preflight,
                self._agent_geometry_edit_mode,
                self._commit_agent_geometry_edit,
            )
        )
        self.agent_authoring_bridge.set_result_invalidation_confirmation(
            lambda: self._confirm_result_invalidation()
        )
        self.agent_authoring_bridge.bind_snapshot(
            self.document,
            document_id=self._active_context.document_id,
        )
        self.agent_authoring_controller = (
            create_session_authoring_workflow_controller(
                self.session,
                self.agent_authoring_bridge,
                self.agent_result_query_bridge,
                next_job_name=self.workspace.next_job_name,
                workspace_catalog_bridge=create_workspace_catalog_bridge(
                    self.workspace
                ),
            )
        )
        self._applied_session_revision = self.document.session_revision
        self._import_notices: tuple[object, ...] = ()
        # Keep the extension compatibility decision explicit in the GUI.  A
        # legacy ``.femproj`` document must be routed through Save As so the
        # old file is never silently replaced by a new-schema writer.
        self._legacy_project_extension = False
        self._current_step_name: str | None = None
        self.geometry: ModelGeometry | None = None
        self._result_archive_geometry: ModelGeometry | None = None
        self._result_archive_model_view: ArchiveModelView | None = None
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
        self._selection_context = SelectionContextState()
        self._temporary_selection_context: SelectionContextState | None = None
        self._temporary_selection_owner: str | None = None
        self._pending_local_mesh_selection = False
        self._pending_analysis_selection: str | None = None
        self._pending_analysis_requested_scope_kind: str | None = None
        self._pending_analysis_dialog_state: (
            DisplacementDialogState | LoadDialogState | None
        ) = None
        self._pending_scope_kind: str | None = None
        self._pending_analysis_edit: (
            tuple[
                tuple[str, int, int | None],
                str,
                tuple[AnalysisStep, ...],
            ]
            | None
        ) = None
        self._scope_selection_overlay_active = False
        self._scope_selection_topology_cache: (
            ScopeSelectionTopology | None
        ) = None
        self._mesh_selection_topology_cache: (
            MeshSelectionTopology | None
        ) = None
        self._session_authoring_cache: (
            tuple[object, object, SessionAuthoringProjection] | None
        ) = None
        self._result_model_capability_cache: (
            tuple[object, object, ModelCapabilityReport] | None
        ) = None
        self._wire_editor_controller: WireDraftController | None = None
        self._wire_editor_original_recipe: object | None = None
        self._wire_editor_base_revision: int | None = None
        self._wire_editor_part_name: str | None = None
        self._sketch_editor_controller: SketchDraftController | None = None
        self._sketch_editor_original_recipe: object | None = None
        self._sketch_editor_base_revision: int | None = None
        self._sketch_editor_part_name: str | None = None
        self._body_boolean_controller: PartBooleanController | None = None
        self._body_boolean_preview_result: object | None = None
        self._body_boolean_preview_generation = 0
        self._planar_boolean_controller: PlanarBooleanController | None = None
        self._planar_boolean_preview_result: StrictPlanarBooleanResult | None = None
        self._planar_boolean_preview_generation = 0
        self._planar_boolean_original_selection: set[LogicalEntityRef] = set()
        self._sketch_editor_is_planar_boolean_tool = False
        self._sketch_editor_is_face_sketch = False
        self._face_sketch_controller: FaceSupportedSketchController | None = None
        self._face_sketch_dialog: FaceSketchBooleanDialog | None = None
        self._face_sketch_parameters: FaceSketchBooleanParameters | None = None
        self._face_sketch_preview_result: FaceSketchBooleanResult | None = None
        self._face_sketch_preview_generation = 0
        self._face_sketch_reference_points: tuple[SketchReferencePoint, ...] = ()
        self._face_sketch_original_selection: set[LogicalEntityRef] = set()
        self._solid_face_boolean_operation: (
            FaceSketchBooleanOperation | None
        ) = None
        self._solid_face_boolean_original_selection: set[
            LogicalEntityRef
        ] = set()
        self._face_sketch_selection_cache: dict[
            tuple[str, str, int, str], bool
        ] = {}
        self.faceSketchBooleanFeatureRequested.connect(
            self._commit_face_sketch_boolean_feature
        )
        self._geometry_preview_cache: (
            tuple[str, object, GeometryPreview] | None
        ) = None
        self._pending_exact_boolean_preview_key: (
            tuple[str, str, int, int] | None
        ) = None
        self.selection = SelectionState()
        self.actions: dict[str, QAction] = {}
        self.scope_background_reference_threshold = (
            _DEFAULT_SCOPE_BACKGROUND_REFERENCE_THRESHOLD
        )
        self._pending_mesh_topology_selection_filter: str | None = None
        self._pending_mesh_topology_callback: Callable[[], None] | None = None
        self._queued_named_region_edit: (
            tuple[int, NamedRegionEditTaskSnapshot, GuiCommandCompletion] | None
        ) = None
        self._command_counter = 0
        self._job_manager: JobManagerDialog | None = None
        self._viewport_fit_pending = False
        self._display = DisplayState()
        self._show_suppressed_part_ghosts = False
        self._model_edges_visible = True
        self._scale_mode = "auto"
        self._scale_value = 1.0
        self._overlay_undeformed = False
        self._symbol_settings = SymbolSettings()
        self._application_settings = QSettings("fem-project", "fem-gui")
        self._background_settings, self._remember_background = load_background_settings(
            self._application_settings
        )
        self._default_contour_options: dict[str, Any] = {
            "manual": False, "minimum": 0.0, "maximum": 1.0,
            "levels": 12, "colormap": ABAQUS_RAINBOW,
            "style": "segmented", "legend": True,
            "render_mode": CONTOUR_RENDER_SHADED,
            "edge_mode": CONTOUR_EDGE_GEOMETRY,
            "edge_style": "solid", "edge_width": 1.0,
            "number_format": "scientific", "decimals": 2,
            "orientation": "vertical", "show_minimum": False,
            "show_maximum": False, "show_ids": False,
            "legend_font": "Arial", "legend_font_size": 14,
            "show_coordinate_system": True,
            "edges": True,
            "averaging_threshold": 75.0,
        }
        self._contour_options: dict[str, Any] = dict(
            self._default_contour_options
        )
        self._result_visualization_provider_cache: (
            tuple[
                ResultSourceKey,
                int,
                FieldMaterializationKey,
                ResultProvider,
            ]
            | None
        ) = None
        self._result_topology_template_cache: (
            tuple[object, ResultFieldTopologyTemplate] | None
        ) = None
        self._result_deformation_scale_cache: (
            tuple[object, str, str, float, float] | None
        ) = None
        self._step_combos: list[QComboBox] = []
        self._closing = False
        self._exit_pending = False
        self._shared_resources_shutdown = False
        self._deferred_ui_callbacks: list[Callable[[], None]] = []
        self._deferred_ui_timer = QTimer(self)
        self._deferred_ui_timer.setSingleShot(True)
        self._deferred_ui_timer.timeout.connect(
            self._run_deferred_ui_callbacks
        )
        self._build_actions()
        self._build_menus()
        self._build_ribbon()
        self._build_central_area()
        self._build_status_bar()
        self._bind_task_controller(self._active_context)
        self._bind_agent_document(self._active_context)
        self._refresh_result_controls()
        self._update_action_states()
        self._refresh_model_tree_for_context(self._active_context)
        self._refresh_result_tree_for_context(self._active_context)
        self.model_tree.set_active_document(self._active_context.document_id)
        self.result_tree.set_active_document(self._active_context.document_id)

    @property
    def busy(self) -> bool:
        return self.task_controller.busy

    def _task_context_or_active(self) -> WorkspaceDocument | None:
        return self._task_callback_context or self._active_workspace_context()

    def _bind_task_controller(self, context: WorkspaceDocument) -> None:
        """Connect one document controller without duplicating signal slots."""

        controller = context.task_controller
        if controller in self._task_controller_connections:
            return
        self._task_controller_connections.add(controller)
        controller.busy_changed.connect(
            lambda busy, target=context: self._task_busy_changed_for_context(
                target,
                bool(busy),
            )
        )
        controller.cancelling_changed.connect(
            lambda cancelling, target=context: self._task_cancelling_changed_for_context(
                target,
                bool(cancelling),
            )
        )

    def _task_context_is_active(self, context: WorkspaceDocument) -> bool:
        return self.workspace.active_document_id == context.document_id

    def _task_busy_changed_for_context(
        self,
        context: WorkspaceDocument,
        busy: bool,
    ) -> None:
        if self._task_context_is_active(context):
            self._task_busy_changed(busy)

    def _task_cancelling_changed_for_context(
        self,
        context: WorkspaceDocument,
        cancelling: bool,
    ) -> None:
        if self._task_context_is_active(context):
            self._task_cancelling_changed(cancelling)

    def _bind_agent_document(self, context: WorkspaceDocument) -> None:
        """Rebind the one Agent bridge/port pair to an idle target document."""

        panel = getattr(self, "viewport_panel", None)
        drawer = None if panel is None else getattr(panel, "agent_chat_drawer", None)
        runtime = None if drawer is None else getattr(drawer, "agent_runtime", None)
        if runtime is not None:
            runtime.bind_target(context.document_id, context.session.session_id)
        idle = None if runtime is None else (lambda: not runtime.busy)
        for bridge in (
            self.agent_authoring_bridge,
            self.agent_result_query_bridge,
        ):
            port = bridge.port
            binder = getattr(port, "bind_session", None)
            if callable(binder):
                binder(context.session, idle=idle)
        # Document switches intentionally do not preserve unfinished Agent
        # workflow state.  Reset before observing the new typed binding so the
        # controller becomes usable for the target instead of remaining stale.
        self.agent_authoring_controller.reset_for_binding()
        if context.is_result:
            self.agent_authoring_bridge.unbind_context(
                "结果只读文档不支持 Agent 建模操作"
            )
            self.agent_authoring_controller.invalidate_binding(
                "结果只读文档不支持 Agent 建模操作"
            )
        else:
            self.agent_authoring_bridge.bind_snapshot(
                context.projection,
                document_id=context.document_id,
            )
            current = self.agent_authoring_bridge.context
            if current is not None:
                self.agent_authoring_controller.observe_binding(current)
        if drawer is not None:
            drawer.refresh_authoring_binding()

    def _active_workspace_context(self) -> WorkspaceDocument | None:
        """Return the cached active context, refreshing after workspace moves."""

        workspace = getattr(self, "workspace", None)
        context = getattr(self, "_active_context", None)
        if (
            workspace is not None
            and context is not None
            and workspace.active_document_id == context.document_id
        ):
            return context
        if workspace is None:
            return context
        context = workspace.active_document()
        if context is not None:
            self._active_context = context
        return context

    def _activate_workspace_context(
        self,
        context: WorkspaceDocument,
    ) -> bool:
        """Activate one context while reusing its presentation adapters.

        The workspace owns the long-lived geometry/inspection references.  The
        window only installs those references into its single viewport; VTK
        actors are consequently rebuilt for the active document and never
        retained by an inactive context.
        """

        previous_context = getattr(self, "_active_context", None)
        previous_active_id = self.workspace.active_document_id
        previous_active_kind = self.workspace.active_kind
        previous_revision = self._applied_session_revision
        if (
            previous_context is context
            and previous_active_id == context.document_id
        ):
            self.model_tree.set_active_document(context.document_id)
            self.result_tree.set_active_document(context.document_id)
            self._task_busy_changed(context.task_controller.busy)
            if context.task_controller.cancel_requested:
                self._task_cancelling_changed(True)
            return True
        # Embedded editors own transient geometry and cannot be detached from
        # the current viewport safely.  Keep the current context active.
        if self._active_editor():
            return False
        try:
            self._bind_agent_document(context)
        except (RuntimeError, TypeError, ValueError):
            if previous_context is not None:
                try:
                    self._bind_agent_document(previous_context)
                except (RuntimeError, TypeError, ValueError):
                    logging.exception(
                        "failed to restore Agent binding after partial bind"
                    )
            return False

        if previous_context is not None:
            self._capture_document_presentation(previous_context)
        previous_presentation = (
            replace(previous_context.presentation_state)
            if previous_context is not None
            else None
        )
        previous_aliases = {
            "document": self.document,
            "geometry": self.geometry,
            "inspection_service": self.inspection_service,
            "result_provider": self.result_provider,
            "result_selection": self.result_selection,
            "display": self._display,
            "step_name": self._current_step_name,
            "legacy_project_extension": self._legacy_project_extension,
            "result_archive_geometry": self._result_archive_geometry,
            "result_archive_model_view": self._result_archive_model_view,
            "result_visualization_provider_cache": (
                self._result_visualization_provider_cache
            ),
            "model_edges_visible": self._model_edges_visible,
            "symbol_settings": self._symbol_settings,
            "overlay_undeformed": self._overlay_undeformed,
            "selection_context": replace(self._selection_context),
            "geometry_selection_mode": self._geometry_selection_mode,
            "selection_mode": self.selection.mode,
            "selection_node_id": self.selection.node_id,
            "selection_element_id": self.selection.element_id,
            "selected_geometry_refs": set(self._selected_geometry_refs),
            "selected_mesh_scope_refs": set(self._selected_mesh_scope_refs),
        }

        viewport_render_suppressed = getattr(
            self.viewport,
            "_render_suppressed",
            False,
        )
        try:
            self._clear_workspace_transients()
            self.workspace.activate(context)
            self._active_context = context
            self._applied_session_revision = -1
            self._legacy_project_extension = bool(
                context.projection.project_path is not None
                and context.projection.project_path.suffix.casefold()
                in LEGACY_MODEL_FILE_SUFFIXES
            )
            self._prepare_document_presentation(context)
            cache = context.presentation_cache
            artifact = context.projection.artifact
            artifact_id = None if artifact is None else artifact.artifact_id
            model_geometry = (
                cache.model_geometry
                if cache.matches_artifact(artifact_id)
                else None
            )
            result_identity = context.session.current_result_identity()
            result_model_view = None
            if result_identity is not None and cache.matches_result(*result_identity):
                result_model_view = cache.result_model_view

            self._workspace_activation = True
            self.viewport._render_suppressed = True
            delta = SessionDelta(
                session_revision=int(context.session.session_revision),
                reason="workspace document activated",
            )
            applied = self._apply_session_delta(
                delta,
                context=context,
                model_geometry=model_geometry,
                result_model_view=result_model_view,
            )
            if not applied:
                raise RuntimeError("target context projection could not be installed")
            self._restore_document_presentation(context, render=False)
            self._project_viewport_for_module(
                context.presentation_state.module_name
                or self._current_module_name(),
                render=False,
                reset_camera=False,
            )
            self._set_ribbon_module_silent(
                context.presentation_state.module_name
                or self._current_module_name()
            )
            # First display fits once; warm displays restore the saved camera
            # and do not reset/fit it.  All preceding calls use render=False.
            self.viewport._render_suppressed = viewport_render_suppressed
            camera_state = context.presentation_state.camera_state
            if camera_state is not None and self.viewport._plotter is not None:
                _restore_camera_state(self.viewport._plotter, camera_state)
                self.viewport._refresh_symbols_for_camera(render=False)
                self.viewport.render()
            else:
                try:
                    self.viewport.fit(render=False)
                except TypeError:
                    # Keep compatibility with lightweight viewport probes in
                    # existing GUI tests that expose ``fit()`` only.
                    self.viewport.fit()
                self.viewport.render()
            self.model_tree.set_active_document(context.document_id)
            self.result_tree.set_active_document(context.document_id)
            self._task_busy_changed(context.task_controller.busy)
            if context.task_controller.cancel_requested:
                self._task_cancelling_changed(True)
            return True
        except Exception:
            logging.exception("workspace document activation failed")
            applied = False
        finally:
            self._workspace_activation = False
            self.viewport._render_suppressed = viewport_render_suppressed

        # Activation can fail after the workspace registry has already moved.
        # Restore the previous pair so aliases and registries stay in lockstep.
        previous_still_registered = False
        if previous_context is not None:
            try:
                previous_still_registered = (
                    self.workspace.document(previous_context.document_id)
                    is previous_context
                )
            except (KeyError, TypeError, ValueError):
                previous_still_registered = False
        if previous_still_registered:
            try:
                self.workspace.activate(previous_context)
            except (KeyError, TypeError, ValueError):
                self.workspace.active_document_id = previous_active_id
                self.workspace.active_kind = previous_active_kind
        else:
            fallback = self.workspace.active_document()
            if fallback is not None:
                previous_context = fallback
            else:
                self.workspace.active_document_id = previous_active_id
                self.workspace.active_kind = previous_active_kind
        self._active_context = previous_context
        self._applied_session_revision = previous_revision
        if previous_context is not None:
            try:
                self._bind_agent_document(previous_context)
            except (RuntimeError, TypeError, ValueError):
                logging.exception("failed to restore Agent document binding")
        if previous_context is not None:
            previous_context.presentation_state = previous_presentation
        self._restore_failed_workspace_activation(
            previous_context,
            previous_aliases,
            viewport_render_suppressed=viewport_render_suppressed,
        )
        if previous_context is None:
            self.model_tree.set_active_document(None)
            self.result_tree.set_active_document(None)
        elif self.workspace.active_document_id == previous_context.document_id:
            self.model_tree.set_active_document(previous_context.document_id)
            self.result_tree.set_active_document(previous_context.document_id)
        return applied

    def _restore_failed_workspace_activation(
        self,
        previous_context: WorkspaceDocument | None,
        previous_aliases: dict[str, object],
        *,
        viewport_render_suppressed: bool,
    ) -> None:
        """Restore aliases and the previous scene after a failed activation."""

        self.document = previous_aliases["document"]
        self.geometry = previous_aliases["geometry"]
        self.inspection_service = previous_aliases["inspection_service"]
        self.result_provider = previous_aliases["result_provider"]
        self.result_selection = previous_aliases["result_selection"]
        self._display = previous_aliases["display"]
        self._current_step_name = previous_aliases["step_name"]
        self._legacy_project_extension = previous_aliases[
            "legacy_project_extension"
        ]
        self._result_archive_geometry = previous_aliases[
            "result_archive_geometry"
        ]
        self._result_archive_model_view = previous_aliases[
            "result_archive_model_view"
        ]
        self._result_visualization_provider_cache = previous_aliases[
            "result_visualization_provider_cache"
        ]
        self._model_edges_visible = previous_aliases["model_edges_visible"]
        self._symbol_settings = previous_aliases["symbol_settings"]
        self._overlay_undeformed = previous_aliases["overlay_undeformed"]
        self._selection_context = previous_aliases["selection_context"]
        self._geometry_selection_mode = previous_aliases[
            "geometry_selection_mode"
        ]
        self.selection.mode = previous_aliases["selection_mode"]
        self.selection.node_id = previous_aliases["selection_node_id"]
        self.selection.element_id = previous_aliases["selection_element_id"]
        self._selected_geometry_refs = set(
            previous_aliases["selected_geometry_refs"]
        )
        self._selected_mesh_scope_refs = set(
            previous_aliases["selected_mesh_scope_refs"]
        )

        # Rebuild only the shared viewport scene.  All actor and display
        # mutations remain render-suppressed until the final explicit repaint.
        self.viewport._render_suppressed = True
        try:
            if previous_context is None:
                FEMViewport.clear_model(self.viewport)
            elif self.document.artifact is not None and self.geometry is not None:
                self._restore_viewport_model_scene(
                    render=False,
                    reset_camera=False,
                )
            else:
                preview = self._current_native_geometry_preview()
                if preview is not None:
                    self.viewport.show_geometry_preview(
                        preview,
                        render=False,
                        reset_camera=False,
                    )
                else:
                    FEMViewport.clear_model(self.viewport)
            if self.result_provider is not None and self.result_selection is not None:
                try:
                    self._apply_display(render=False)
                except Exception:
                    logging.exception(
                        "failed to restore previous result display after activation"
                    )
            self._sync_step_combos()
            self._refresh_result_controls()
            self.status_panel.set_step(self._current_step_name)
            self._update_action_states()
        except Exception:
            logging.exception(
                "failed to restore previous viewport after activation failure"
            )
        finally:
            self.viewport._render_suppressed = viewport_render_suppressed
        camera_state = (
            previous_context.presentation_state.camera_state
            if previous_context is not None
            else None
        )
        if camera_state is not None and self.viewport._plotter is not None:
            try:
                _restore_camera_state(self.viewport._plotter, camera_state)
                self.viewport._refresh_symbols_for_camera(render=False)
            except Exception:
                logging.exception(
                    "failed to restore previous camera after activation failure"
                )
        try:
            self.viewport.render()
        except Exception:
            logging.exception("failed to repaint previous viewport after activation failure")

    def _capture_document_presentation(
        self,
        context: WorkspaceDocument,
    ) -> None:
        """Save the lightweight visible state before leaving ``context``."""

        if context is not self._active_workspace_context():
            return
        state = context.presentation_state
        state.module_name = self._current_module_name() or None
        state.step_name = self._current_step_name
        state.result_selection = self.result_selection
        state.display_state = self._display
        state.selection_mode = getattr(self.viewport, "_selection_mode", None)
        state.result_scale_mode = str(self._scale_mode)
        state.result_scale_value = float(self._scale_value)
        state.contour_options = dict(self._contour_options)
        state.overlay_undeformed = bool(self._overlay_undeformed)
        plotter = getattr(self.viewport, "_plotter", None)
        state.camera_state = (
            _capture_camera_state(plotter) if plotter is not None else None
        )

        cache = context.presentation_cache
        artifact = context.projection.artifact
        artifact_id = None if artifact is None else artifact.artifact_id
        if artifact_id is None:
            cache.invalidate_model()
        elif (
            self.geometry is not None
            and self.geometry.artifact_id == artifact_id
        ):
            cache.artifact_id = artifact_id
            cache.model_geometry = self.geometry
            cache.inspection_service = self.inspection_service
        identity = self.session.current_result_identity()
        provider = self.result_provider
        if identity is None or provider is None:
            cache.invalidate_result()
        elif provider.source == identity[0]:
            cache.result_source = identity[0]
            cache.result_generation = identity[1]
            cache.result_model_view = self._result_archive_model_view

    def _prepare_document_presentation(self, context: WorkspaceDocument) -> None:
        """Seed window aliases from ``context`` before projection callbacks run."""

        state = context.presentation_state
        names = context.session.runnable_step_names()
        self._current_step_name = (
            state.step_name if state.step_name in names else context.session.default_step_name()
        )
        self._display = (
            state.display_state
            if isinstance(state.display_state, DisplayState)
            else DisplayState()
        )
        self.result_selection = (
            state.result_selection
            if isinstance(state.result_selection, ScalarFieldSelection)
            else None
        )
        scale_mode = state.result_scale_mode
        self._scale_mode = (
            scale_mode
            if scale_mode in {"auto", "real", "custom"}
            else "auto"
        )
        try:
            self._scale_value = float(state.result_scale_value)
        except (TypeError, ValueError):
            self._scale_value = 1.0
        saved_contour_options = (
            state.contour_options
            if isinstance(state.contour_options, dict)
            and state.contour_options
            else self._default_contour_options
        )
        self._contour_options = dict(self._default_contour_options)
        self._contour_options.update(saved_contour_options)
        self._overlay_undeformed = bool(state.overlay_undeformed)
        selection_mode = state.selection_mode
        if isinstance(selection_mode, str):
            if selection_mode.startswith("geometry_"):
                self._selection_context.set_space("geometry")
                self._geometry_selection_mode = selection_mode.removeprefix(
                    "geometry_"
                )
            elif selection_mode.startswith("mesh_"):
                self._selection_context.set_space("mesh")
                mesh_filter = selection_mode.removeprefix("mesh_")
                self._selection_context.set_filter(
                    "point" if mesh_filter == "node" else mesh_filter
                )
            elif selection_mode in {"node", "element"}:
                self.selection.mode = selection_mode
        else:
            self._selection_context = SelectionContextState()
            self._geometry_selection_mode = "body"
            self.selection.mode = "node"
            try:
                self.viewport.set_selection_mode("node", render=False)
            except TypeError:
                self.viewport.set_selection_mode("node")

    def _restore_document_presentation(
        self,
        context: WorkspaceDocument,
        *,
        render: bool,
    ) -> None:
        """Install state controls after the target scene has been projected."""

        state = context.presentation_state
        self._current_step_name = (
            state.step_name
            if state.step_name in self.session.runnable_step_names()
            else self.session.default_step_name()
        )
        if isinstance(state.display_state, DisplayState):
            self._display = state.display_state
            for name, checked in (
                ("undeformed", self._display.shape_mode == "undeformed"),
                ("deformed", self._display.shape_mode == "deformed"),
                ("contour", self._display.contour_enabled),
            ):
                action = self.actions.get(name)
                if action is not None:
                    with QSignalBlocker(action):
                        action.setChecked(checked)
        self.result_scale_combo.blockSignals(True)
        scale_index = self.result_scale_combo.findData(self._scale_mode)
        if scale_index >= 0:
            self.result_scale_combo.setCurrentIndex(scale_index)
        self.result_scale_combo.blockSignals(False)
        self.result_scale_value.blockSignals(True)
        self.result_scale_value.setValue(float(self._scale_value))
        self.result_scale_value.blockSignals(False)
        overlay_action = self.actions.get("overlay")
        if overlay_action is not None:
            with QSignalBlocker(overlay_action):
                overlay_action.setChecked(self._overlay_undeformed)
        try:
            self.viewport.set_undeformed_overlay_visible(
                self._overlay_undeformed
            )
            self.viewport.set_contour_options(
                {
                    key: value
                    for key, value in self._contour_options.items()
                    if key != "averaging_threshold"
                }
            )
            self.viewport.set_contour_metadata(
                {
                    "averaging_threshold": float(
                        self._contour_options.get("averaging_threshold", 20.0)
                    )
                }
            )
        except (AttributeError, TypeError, ValueError):
            pass
        selection_mode = state.selection_mode
        if isinstance(selection_mode, str):
            try:
                self.viewport.set_selection_mode(selection_mode, render=render)
            except (TypeError, ValueError):
                # Older/fake viewports may not expose the render keyword.
                self.viewport.set_selection_mode(selection_mode)
        self._sync_step_combos()
        self._refresh_result_controls()
        self.status_panel.set_step(self._current_step_name)
        self._update_action_states()

    def _clear_workspace_transients(self) -> None:
        """Remove selections and inspection popups before a context switch."""

        self._close_inspection_windows()
        self._close_job_manager()
        self._temporary_selection_context = None
        self._temporary_selection_owner = None
        self._pending_local_mesh_selection = False
        self._pending_analysis_selection = None
        self._pending_analysis_requested_scope_kind = None
        self._pending_scope_kind = None
        self._scope_selection_overlay_active = False
        self.selection.clear()
        self._selected_geometry_refs.clear()
        self._selected_mesh_scope_refs.clear()
        try:
            self.viewport.clear_selection(render=False)
        except TypeError:
            self.viewport.clear_selection()

    def _set_ribbon_module_silent(self, module_name: str | None) -> None:
        if not module_name or not hasattr(self, "ribbon"):
            return
        tab_bar = self.ribbon.tab_bar
        for index in range(tab_bar.count()):
            if tab_bar.tabText(index) == module_name:
                with QSignalBlocker(tab_bar):
                    tab_bar.setCurrentIndex(index)
                self.ribbon.stack.setCurrentIndex(index)
                return

    @property
    def session(self) -> ModelSession:
        """Compatibility alias for the active workspace Session."""

        context = self._task_context_or_active()
        if context is not None:
            return context.session
        return self._session_alias

    @session.setter
    def session(self, value: ModelSession) -> None:
        self._session_alias = value
        context = self._active_workspace_context()
        if context is not None:
            context.session = value

    @property
    def document(self) -> object:
        """Compatibility alias for the active context projection."""

        context = self._task_context_or_active()
        if context is not None:
            return context.projection
        return self._document_alias

    @document.setter
    def document(self, value: object) -> None:
        self._document_alias = value
        context = self._active_workspace_context()
        if context is not None:
            self.workspace.update_projection(context, value)

    @property
    def task_controller(self) -> BackgroundTaskController:
        """Compatibility alias for the active context task controller."""

        context = self._task_context_or_active()
        if context is not None:
            return context.task_controller
        return self._task_controller_alias

    @task_controller.setter
    def task_controller(self, value: BackgroundTaskController) -> None:
        self._task_controller_alias = value
        context = self._active_workspace_context()
        if context is not None:
            context.task_controller = value

    def _defer_ui(self, callback: Callable[[], None]) -> None:
        if self._closing:
            return
        self._deferred_ui_callbacks.append(callback)
        if not self._deferred_ui_timer.isActive():
            self._deferred_ui_timer.start(0)

    def _run_deferred_ui_callbacks(self) -> None:
        callbacks = tuple(self._deferred_ui_callbacks)
        self._deferred_ui_callbacks.clear()
        if self._closing:
            return
        for callback in callbacks:
            if self._closing:
                break
            callback()

    @property
    def import_notices(self) -> tuple[object, ...]:
        """Return detached, non-authoritative notices for the current import."""

        return deepcopy(self._import_notices)

    @property
    def legacy_project_extension(self) -> bool:
        """Whether the current native document came from ``.femproj``."""

        return self._legacy_project_extension

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

    def _accepted_definition_only_command(
        self,
        command_id: int,
        delta: SessionDelta,
    ) -> GuiCommandReceipt:
        """Project a definition-only artifact without rebuilding mesh actors."""

        try:
            self._apply_definition_only_delta(delta)
        except Exception:
            logging.exception("definition-only GUI command projection failed")
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
                part_name=command.part_name,
                body_name=command.body_name,
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
        *,
        document_id: int | None = None,
    ) -> GuiCommandReceipt:
        command_id = self._next_command_id()
        if type(command) is not CloseSessionCommand:
            return self._rejected_command(
                command_id,
                "command.type.invalid",
                "command must be CloseSessionCommand",
            )
        target_context = (
            self.workspace.active_document()
            if document_id is None
            else self.workspace.document(int(document_id))
        )
        if target_context is None:
            return self._rejected_command(
                command_id,
                "workspace.document.missing",
                "no model context is available",
            )
        if target_context.task_controller.busy:
            return self._rejected_command(
                command_id,
                "task.busy",
                "a background task is already running",
            )
        try:
            delta = target_context.session.close(
                expected_session_revision=command.expected_session_revision
            )
            receipt = self._accepted_command(
                command_id,
                delta,
                context=target_context,
            )
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
        if (
            self._wire_editor_controller is not None
            or self._sketch_editor_controller is not None
            or self._body_boolean_controller is not None
            or self._planar_boolean_controller is not None
            or self._solid_face_boolean_operation is not None
        ):
            return self._rejected_command(
                command_id,
                "sketch_editor.active",
                "请先完成或取消当前草图编辑，再打开项目",
            )
        target = Path(path)
        existing_id = self.workspace.model_paths.get(canonical_path(target))
        if existing_id is not None:
            existing = self.workspace.models[existing_id]
            if not self._activate_workspace_context(existing):
                return self._rejected_command(
                    command_id,
                    "workspace.activate.rejected",
                    "the already-open project could not be activated",
                )
            return GuiCommandReceipt.accepted(
                command_id,
                outcome=GuiCommandOutcome(
                    output_path=target,
                    diagnostic_summary="已激活已打开的模型",
                ),
            )
        open_controller = self.workspace.ensure_open_controller()
        if open_controller.busy:
            return self._rejected_command(
                command_id,
                "task.busy",
                "a project open task is already running",
            )
        completion = GuiCommandCompletion(command_id)
        accepted_context: WorkspaceDocument | None = None

        def workload(context: TaskContext) -> LoadedProject:
            context.report("正在读取并验证自主项目……")
            context.checkpoint()
            loaded = load_project(target)
            context.checkpoint()
            return loaded

        def apply_result(payload: object) -> TaskApplyOutcome:
            nonlocal accepted_context
            if type(payload) is not LoadedProject:
                raise TypeError(
                    "project loader worker must return LoadedProject"
                )
            try:
                # Decode remains detached from every existing Session.  Only
                # after successful validation do we create a new context and
                # install the snapshot into its own writer.
                session = ModelSession()
                accepted_context = self.workspace.add_model(
                    session=session,
                    display_name=(
                        str(payload.snapshot.model_name or "").strip()
                        or target.stem
                    ),
                    source_path=payload.path or target,
                )
                delta = session.replace_from_snapshot(payload.snapshot)
            except (RevisionConflictError, TypeError, ValueError) as error:
                if accepted_context is not None:
                    self.workspace.remove(accepted_context)
                    accepted_context = None
                return TaskApplyOutcome.rejected(str(error))
            return TaskApplyOutcome.accepted((payload, accepted_context, delta))

        def on_success(value: object) -> None:
            if (
                type(value) is not tuple
                or len(value) != 3
                or type(value[0]) is not LoadedProject
                or not isinstance(value[1], WorkspaceDocument)
                or type(value[2]) is not SessionDelta
            ):
                raise RuntimeError(
                    "accepted project load has no detached project or delta"
                )
            payload, context, delta = value
            self._apply_session_delta(delta, context=context)
            if not self._activate_workspace_context(context):
                raise RuntimeError("opened project context could not be activated")
            self._import_notices = deepcopy(tuple(payload.notices))
            if payload.notices:
                self.status_panel.set_state(
                    "；".join(notice.message for notice in payload.notices),
                    12000,
                )

        try:
            started = self._start_task(
                workload,
                on_success,
                "打开自主项目失败",
                task_name="打开自主项目",
                apply_result=apply_result,
                completion=completion,
                controller=open_controller,
            )
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            return self._rejected_command(
                command_id,
                "project.open.rejected",
                error,
                "请检查项目文件版本和内容。",
            )
        if not started:
            return self._rejected_command(
                command_id,
                "task.start.rejected",
                "the project open task could not be started",
            )
        return GuiCommandReceipt.pending(command_id, completion)

    def open_inp_path(self, path: str | Path) -> GuiCommandReceipt:
        command_id = self._next_command_id()
        if self._active_editor():
            return self._rejected_command(
                command_id,
                "editor.active",
                "请先完成或取消当前编辑，再打开 INP",
            )
        try:
            target = Path(path)
            existing_id = self.workspace.model_paths.get(canonical_path(target))
            if existing_id is not None:
                existing = self.workspace.models[existing_id]
                if not self._activate_workspace_context(existing):
                    return self._rejected_command(
                        command_id,
                        "workspace.activate.rejected",
                        "the already-open INP model could not be activated",
                    )
                return GuiCommandReceipt.accepted(
                    command_id,
                    outcome=GuiCommandOutcome(
                        output_path=target,
                        diagnostic_summary="已激活已打开的 INP 模型",
                    ),
                )
            open_controller = self.workspace.ensure_open_controller()
            if open_controller.busy:
                return self._rejected_command(
                    command_id,
                    "task.busy",
                    "a model open task is already running",
                )
            session = ModelSession()
            task = session.prepare_import(target)
            completion = GuiCommandCompletion(command_id)
            accepted_context: WorkspaceDocument | None = None

            def workload(context: TaskContext):
                timings: dict[str, float] = {}
                context.report("正在解析并构建 INP……")
                started_at = perf_counter()
                import_result = read_inp_with_report(target)
                timings["INP 解析与构建"] = perf_counter() - started_at
                model = import_result.model
                context.report("正在生成显示网格……")
                started_at = perf_counter()
                geometry = build_model_geometry(model)
                timings["VTK 显示几何构建"] = perf_counter() - started_at
                context.report("正在准备独立模型文档……")
                started_at = perf_counter()
                prepared = session.prepare_owned_imported_model_transfer(model)
                timings["Session 模型所有权准备"] = perf_counter() - started_at
                context.checkpoint()
                return prepared, geometry, timings, import_result.notices

            def apply_result(value: object) -> TaskApplyOutcome:
                nonlocal accepted_context
                prepared, _geometry, _timings, _notices = self._unpack_model_load(
                    value
                )
                try:
                    accepted_context = self.workspace.add_model(
                        session=session,
                        display_name=target.stem,
                        source_path=target,
                    )
                    delta = session.accept_imported_model_transfer(
                        task.token,
                        prepared,
                    )
                    if not delta.accepted:
                        raise RuntimeError(delta.reason or "INP import was rejected")
                except (RevisionConflictError, RuntimeError, TypeError, ValueError) as error:
                    if accepted_context is not None:
                        self.workspace.remove(accepted_context)
                        accepted_context = None
                    return TaskApplyOutcome.rejected(str(error))
                return TaskApplyOutcome.accepted(
                    (accepted_context, delta, value)
                )

            def remove_failed_context(context: WorkspaceDocument) -> None:
                self.model_tree.remove_document(context.document_id)
                self.result_tree.remove_model_runs(context.document_id)
                self.workspace.remove(context)

            def project_result(value: object) -> None:
                if (
                    type(value) is not tuple
                    or len(value) != 3
                    or not isinstance(value[0], WorkspaceDocument)
                    or type(value[1]) is not SessionDelta
                ):
                    raise RuntimeError(
                        "accepted INP import has no detached document and delta"
                    )
                context, delta, payload = value
                _model, geometry, timings, notices = self._unpack_model_load(
                    payload
                )
                if not self._apply_session_delta(
                    delta,
                    context=context,
                    model_geometry=geometry,
                    timings=timings,
                    source_label=target.name,
                ):
                    remove_failed_context(context)
                    raise RuntimeError("已接受的 INP 模型无法投影")
                artifact = context.projection.artifact
                if artifact is None:
                    remove_failed_context(context)
                    raise RuntimeError("已接受的 INP 模型缺少模型构件")
                if geometry.artifact_id != artifact.artifact_id:
                    geometry = replace(geometry, artifact_id=artifact.artifact_id)
                context.presentation_cache.artifact_id = artifact.artifact_id
                context.presentation_cache.model_geometry = geometry
                if not self._activate_workspace_context(context):
                    remove_failed_context(context)
                    raise RuntimeError("INP 模型文档无法激活")
                self._install_import_notices(notices)

            self.status_panel.set_state("正在导入新模型……")
            started = self._start_task(
                workload,
                project_result,
                "模型加载失败",
                task_name="INP 导入",
                apply_result=apply_result,
                completion=completion,
                controller=open_controller,
            )
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

    def save_project_path(
        self,
        path: str | Path,
        *,
        document_id: int | None = None,
    ) -> GuiCommandReceipt:
        command_id = self._next_command_id()
        target_context = (
            self.workspace.active_document()
            if document_id is None
            else self.workspace.document(int(document_id))
        )
        if target_context is None:
            return self._rejected_command(
                command_id,
                "workspace.document.missing",
                "no model context is available",
            )
        target_document = target_context.projection
        if not target_document.can_save:
            return self._rejected_command(
                command_id,
                "project.save.unavailable",
                "current session cannot be saved as a native project",
            )
        target = Path(path)
        if target.suffix.casefold() != MODEL_FILE_SUFFIX:
            target = target.with_suffix(MODEL_FILE_SUFFIX)
        try:
            save_snapshot = target_context.session.prepare_project_save()

            completion = GuiCommandCompletion(command_id)

            def workload(context: TaskContext) -> Path:
                context.report("正在验证并保存自主项目……")
                return save_project(
                    target,
                    save_snapshot,
                    checkpoint=context.checkpoint,
                )

            def apply_result(payload: object) -> TaskApplyOutcome:
                if not isinstance(payload, Path):
                    raise TypeError(
                        "project save worker must return pathlib.Path"
                    )
                return self._session_task_outcome(
                    target_context.session.accept_project_saved(
                        save_snapshot.token,
                        payload,
                    ),
                    payload,
                )

            def on_success(value: object) -> None:
                if (
                    type(value) is not tuple
                    or len(value) != 2
                    or type(value[0]) is not SessionDelta
                    or not isinstance(value[1], Path)
                ):
                    raise TypeError(
                        "accepted project save must carry SessionDelta and Path"
                    )
                delta, _saved_path = value
                self._accepted_command(
                    command_id,
                    delta,
                    context=target_context,
                )

            started = self._start_task(
                workload,
                on_success,
                "保存自主项目失败",
                lambda message: self._target_session_task_failed(
                    target_context,
                    save_snapshot.token,
                    "保存自主项目失败",
                    message,
                ),
                task_name="保存自主项目",
                on_cancelled=lambda: self._target_session_task_cancelled(
                    target_context,
                    save_snapshot.token,
                ),
                on_inactive_failure=lambda message: self._target_session_task_failed(
                    target_context,
                    save_snapshot.token,
                    "保存自主项目失败",
                    message,
                ),
                on_inactive_cancelled=lambda: self._target_session_task_cancelled(
                    target_context,
                    save_snapshot.token,
                ),
                apply_result=apply_result,
                completion=completion,
                controller=target_context.task_controller,
                context=target_context,
            )
        except Exception as error:
            return self._rejected_command(
                command_id,
                "project.save.rejected",
                error,
            )
        if not started:
            return self._rejected_command(
                command_id,
                "task.start.rejected",
                "the project save task could not be started",
            )
        return GuiCommandReceipt.pending(command_id, completion)

    def save_result_path(self, path: str | Path) -> GuiCommandReceipt:
        """Save the displayed successful result through the shared worker."""

        command_id = self._next_command_id()
        if self.busy:
            return self._rejected_command(
                command_id,
                "task.busy",
                "a background task is already running",
            )
        save_snapshot = None
        started = False
        try:
            spec = ResultArchiveSaveSpec(Path(path))
            run_id = self.document.displayed_result_run_id
            if run_id is None:
                raise RuntimeError("当前没有可保存的成功结果")
            snapshot = self.session.prepare_result_archive_save(run_id)
            save_snapshot = snapshot
            completion = GuiCommandCompletion(command_id)

            def workload(context: TaskContext) -> Path:
                context.report("正在编码并保存分析结果……")
                return save_result_archive(
                    spec.path,
                    snapshot.archive,
                    checkpoint=context.checkpoint,
                )

            def apply_result(payload: object) -> TaskApplyOutcome:
                if not isinstance(payload, Path):
                    raise TypeError("result save worker must return pathlib.Path")
                return self._session_task_outcome(
                    self.session.accept_result_archive_saved(
                        snapshot.token,
                        payload,
                    ),
                    payload,
                )

            def on_success(value: object) -> None:
                if (
                    type(value) is not tuple
                    or len(value) != 2
                    or type(value[0]) is not SessionDelta
                    or not isinstance(value[1], Path)
                ):
                    raise TypeError(
                        "accepted result save must carry SessionDelta and Path"
                    )
                delta, saved_path = value
                self._accepted_command(command_id, delta)
                self.status_panel.set_state(
                    f"分析结果已保存：{saved_path.name}",
                    5000,
                )

            started = self._start_task(
                workload,
                on_success,
                "保存分析结果失败",
                lambda message: self._session_task_failed(
                    snapshot.token,
                    "保存分析结果失败",
                    message,
                ),
                task_name="保存分析结果",
                on_cancelled=lambda: self._session_task_cancelled(snapshot.token),
                on_inactive_failure=lambda message: self._session_task_failed(
                    snapshot.token,
                    "保存分析结果失败",
                    message,
                ),
                on_inactive_cancelled=lambda: self._session_task_cancelled(
                    snapshot.token
                ),
                apply_result=apply_result,
                completion=completion,
            )
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            if save_snapshot is not None:
                self.session.accept_result_archive_save_cancelled(
                    save_snapshot.token
                )
            return self._rejected_command(
                command_id,
                "result.save.rejected",
                error,
            )
        if not started:
            if save_snapshot is not None:
                self.session.accept_result_archive_save_cancelled(
                    save_snapshot.token
                )
            return self._rejected_command(
                command_id,
                "task.start.rejected",
                "the result save task could not be started",
            )
        return GuiCommandReceipt.pending(command_id, completion)

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
        if not self._confirm_result_invalidation():
            return self._rejected_command(
                command_id,
                "document.transition.cancelled",
                "用户取消了未保存结果确认",
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
        if not self._confirm_result_invalidation():
            return self._rejected_command(
                command_id,
                "document.transition.cancelled",
                "用户取消了未保存结果确认",
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
        if type(batch) is not NamedRegionEditBatch:
            return self._rejected_command(
                command_id,
                "command.type.invalid",
                "batch must be a NamedRegionEditBatch",
            )
        if not self._confirm_result_invalidation():
            return self._rejected_command(
                command_id,
                "document.transition.cancelled",
                "用户取消了未保存结果确认",
            )
        reference_count = sum(
            len(region.references) for region in batch.regions
        )
        if reference_count > self.scope_background_reference_threshold:
            return self._apply_named_region_edit_in_background(
                command_id,
                batch,
            )
        try:
            delta = self.session.apply_named_region_edit(batch)
            return self._accepted_definition_only_command(command_id, delta)
        except (RevisionConflictError, RuntimeError, TypeError, ValueError) as error:
            return self._rejected_command(
                command_id,
                "named_region.edit.rejected",
                error,
            )

    def _apply_named_region_edit_in_background(
        self,
        command_id: int,
        batch: NamedRegionEditBatch,
    ) -> GuiCommandReceipt:
        """Start one revision-bound large scope materialization task."""

        if self.busy:
            if (
                self.task_controller.current_task_name != "提交作用域"
                or self._queued_named_region_edit is not None
            ):
                return self._rejected_command(
                    command_id,
                    "task.busy",
                    "a background task is already running",
                )
            try:
                task = self.session.prepare_named_region_edit(batch)
                completion = GuiCommandCompletion(command_id)
            except (
                RevisionConflictError,
                RuntimeError,
                TypeError,
                ValueError,
            ) as error:
                return self._rejected_command(
                    command_id,
                    "named_region.edit.rejected",
                    error,
                )
            self._queued_named_region_edit = (
                command_id,
                task,
                completion,
            )
            self.task_controller.request_cancel(
                after_cleanup=self._launch_queued_named_region_edit
            )
            return GuiCommandReceipt.pending(command_id, completion)
        try:
            task = self.session.prepare_named_region_edit(batch)
            completion = GuiCommandCompletion(command_id)
            started = self._start_named_region_edit_task(
                command_id,
                task,
                completion,
            )
        except (RevisionConflictError, RuntimeError, TypeError, ValueError) as error:
            return self._rejected_command(
                command_id,
                "named_region.edit.rejected",
                error,
            )
        if not started:
            self.session.terminate_named_region_edit(
                task,
                "named-region task start rejected",
            )
            return self._rejected_command(
                command_id,
                "task.start.rejected",
                "the scope task could not be started",
            )
        return GuiCommandReceipt.pending(command_id, completion)

    def _launch_queued_named_region_edit(self) -> None:
        queued = self._queued_named_region_edit
        self._queued_named_region_edit = None
        if queued is None:
            return
        command_id, task, completion = queued
        if not self._start_named_region_edit_task(
            command_id,
            task,
            completion,
        ):
            self.session.terminate_named_region_edit(
                task,
                "queued named-region task start rejected",
            )
            self.status_panel.set_state(
                "后续作用域提交未能启动",
                5000,
            )

    def _start_named_region_edit_task(
        self,
        command_id: int,
        task: NamedRegionEditTaskSnapshot,
        completion: GuiCommandCompletion,
    ) -> bool:
        """Launch one already-prepared scope snapshot on the shared worker."""

        def workload(context: TaskContext) -> object:
            context.report("正在后台提交作用域……")
            prepared = compile_named_region_edit(task)
            context.checkpoint()
            return prepared

        def apply_result(payload: object) -> TaskApplyOutcome:
            if type(payload) is not PreparedNamedRegionEdit:
                raise TypeError(
                    "scope worker must return PreparedNamedRegionEdit"
                )
            delta = self.session.accept_named_region_edit(task, payload)
            if not delta.accepted:
                return TaskApplyOutcome.stale(
                    "作用域提交结果已过期，未应用"
                )
            return TaskApplyOutcome.accepted(delta)

        def on_success(value: object) -> None:
            if type(value) is not SessionDelta:
                raise TypeError(
                    "accepted scope edit must carry a SessionDelta"
                )
            self._accepted_definition_only_command(command_id, value)
            self.status_panel.set_state("作用域已更新", 5000)

        def on_failure(message: str) -> None:
            terminated = self.session.terminate_named_region_edit(
                task,
                message,
            )
            if terminated.accepted:
                self._show_error("提交作用域失败", message)

        def on_cancelled() -> None:
            self.session.terminate_named_region_edit(
                task,
                "named-region edit cancelled",
            )

        def on_inactive_failure(message: str) -> None:
            self.session.terminate_named_region_edit(task, message)

        return self._start_task(
            workload,
            on_success,
            "提交作用域失败",
            on_failure,
            task_name="提交作用域",
            on_cancelled=on_cancelled,
            on_inactive_failure=on_inactive_failure,
            on_inactive_cancelled=on_cancelled,
            apply_result=apply_result,
            completion=completion,
        )

    def apply_definition_edit(
        self,
        batch: DefinitionEditBatch,
    ) -> GuiCommandReceipt:
        command_id = self._next_command_id()
        if not self._confirm_result_invalidation():
            return self._rejected_command(
                command_id,
                "document.transition.cancelled",
                "用户取消了未保存结果确认",
            )
        try:
            delta = self.session.apply_definition_edit(batch)
            artifact = self.document.artifact
            if (
                artifact is not None
                and self.geometry is not None
                and self.viewport.artifact_id == artifact.artifact_id
            ):
                return self._accepted_definition_only_command(
                    command_id,
                    delta,
                )
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
        if not self._confirm_result_invalidation():
            return self._rejected_command(
                command_id,
                "document.transition.cancelled",
                "用户取消了未保存结果确认",
            )
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

    def check_step(
        self,
        step_name: str,
        *,
        expected_session_revision: int | None = None,
    ) -> GuiCommandReceipt:
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
                expected_session_revision=expected_session_revision,
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

    def create_run(self, name: str, step_name: str) -> GuiCommandReceipt:
        """Create one pending analysis run without starting the solver."""

        command_id = self._next_command_id()
        if self.busy:
            return self._rejected_command(
                command_id,
                "task.busy",
                "a background task is already running",
            )
        try:
            if self.document.model is None or self.geometry is None:
                raise RuntimeError("a current model is required")
            if type(name) is not str or type(step_name) is not str:
                raise TypeError("name and step_name must be strings")
            clean_name = name.strip()
            clean_step = step_name.strip()
            if not clean_name:
                raise ValueError("作业名称不能为空。")
            if len(clean_name) > 64:
                raise ValueError("作业名称不能超过 64 个字符。")
            if self.workspace.job_name_exists(clean_name):
                raise ValueError(f"作业名称已存在：{clean_name}")
            if clean_step not in self.session.runnable_step_names():
                raise ValueError(f"分析步不存在：{clean_step}")
            delta = self.session.create_run(clean_step, clean_name)
            self.workspace.remember_job_name(clean_name)
            receipt = self._accepted_command(command_id, delta)
            self._refresh_job_manager()
            self.status_panel.set_state(
                f"作业已创建：{clean_name}；请在作业管理器中提交",
                5000,
            )
            return receipt
        except (KeyError, RuntimeError, TypeError, ValueError) as error:
            return self._rejected_command(
                command_id,
                "run.create.rejected",
                error,
            )

    def submit_created_run(self, name: str) -> GuiCommandReceipt:
        """Submit one existing pending run from the job manager."""

        command_id = self._next_command_id()
        if self.busy:
            return self._rejected_command(
                command_id,
                "task.busy",
                "a background task is already running",
            )
        try:
            job = self.session.find_run(name)
            if job is None:
                raise KeyError(f"作业不存在：{name}")
            completion = GuiCommandCompletion(command_id)
            started = self._begin_submit_run(
                job.name,
                job.step_name,
                completion=completion,
                existing_run=True,
            )
        except (KeyError, RuntimeError, TypeError, ValueError) as error:
            return self._rejected_command(
                command_id,
                "run.submit.rejected",
                error,
            )
        if started is None:
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
            field_keys = self._result_materialization_keys(
                provider,
                selection,
            )
            task = self.session.prepare_result_materialization(
                provider.source.run_id,
                field_keys,
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
                return provider.materialize(
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
                on_inactive_failure=lambda message: self._session_task_failed(
                    task.token,
                    "结果字段按需加载失败",
                    message,
                ),
                on_inactive_cancelled=lambda: self._session_task_cancelled(
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

    @staticmethod
    def _result_materialization_keys(
        provider: ResultProvider,
        selection: ScalarFieldSelection,
    ) -> tuple[FieldMaterializationKey, ...]:
        """Batch the five Beam section fields behind one recovery pass."""

        field_id = selection.field_key.request.field_id
        if not result_field_is_beam_section(field_id):
            return (selection.field_key,)
        keys = tuple(
            availability.key
            for availability in visible_result_fields(provider.catalog().fields)
            if (
                availability.state is FieldState.LAZY
                and result_field_is_beam_section(availability.descriptor.field_id)
            )
        )
        return keys or (selection.field_key,)

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
                active_context = self.workspace.active_document()
                self.result_tree.select_selection(
                    current_selection,
                    document_id=(
                        None
                        if active_context is None
                        else active_context.document_id
                    ),
                    source=projected.source,
                )
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
                return provider.materialize(
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
                    accepted_provider = provider.advance(value)
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
                on_inactive_failure=lambda message: self._session_task_failed(
                    task.token,
                    "结果查询失败",
                    message,
                ),
                on_inactive_cancelled=lambda: self._session_task_cancelled(
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
        """Export one or more components from one exact ready field."""

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
                    "result CSV target must use the .csv extension"
                )
            exports = self._prepare_result_csv_exports(spec)
            export = exports[0]
            completion = GuiCommandCompletion(command_id)

            def workload(context: TaskContext) -> GuiCommandOutcome:
                installed = write_result_table_csv(
                    target,
                    exports,
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
        provider = self._current_result_provider()
        if provider is None:
            raise RuntimeError("there is no current accepted result")
        materialization = provider.snapshot
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

    def _prepare_result_csv_exports(
        self,
        spec: ResultCsvExportSpec,
    ) -> tuple[ResultExportSnapshot, ...]:
        provider = self._current_result_provider()
        if provider is None:
            raise RuntimeError("there is no current accepted result")
        materialization = provider.snapshot
        if materialization.source != spec.source:
            raise RuntimeError(
                "export source does not match the current accepted result"
            )
        if materialization.generation != spec.materialization_generation:
            raise RuntimeError(
                "export generation does not match the current materialization"
            )
        return tuple(
            prepare_result_export_snapshot(materialization, selection)
            for selection in spec.selections
        )

    def _apply_session_delta(
        self,
        delta: object,
        *,
        context: WorkspaceDocument | None = None,
        model_geometry: ModelGeometry | None = None,
        result_model_view: ArchiveModelView | None = None,
        geometry_preview: GeometryPreview | None = None,
        timings: dict[str, float] | None = None,
        source_label: str | None = None,
    ) -> bool:
        """Project one accepted Session transition into every GUI cache."""
        if not bool(getattr(delta, "accepted", True)):
            return False
        target_context = context
        if target_context is None:
            target_context = self._task_callback_context
        if target_context is None:
            target_context = self.workspace.active_document()
        if target_context is None:
            return False
        revision = int(getattr(delta, "session_revision"))
        requested_revision = revision
        active_context = self.workspace.active_document()
        if target_context is not active_context:
            if revision <= target_context.projection.session_revision:
                return False
            previous_target_artifact = target_context.projection.artifact
            snapshot = self._snapshot_for_delta(
                delta,
                revision,
                context=target_context,
            )
            if snapshot.session_revision < revision:
                return False
            self.workspace.update_projection(target_context, snapshot)
            next_artifact = snapshot.artifact
            if (
                previous_target_artifact is None
                or next_artifact is None
                or previous_target_artifact.artifact_id != next_artifact.artifact_id
            ):
                target_context.presentation_cache.invalidate_model()
            next_target_result = target_context.session.current_result_identity()
            if next_target_result is None or not target_context.presentation_cache.matches_result(
                *next_target_result
            ):
                target_context.presentation_cache.invalidate_result()
            self._refresh_model_tree_for_context(target_context)
            self._refresh_result_tree_for_context(target_context)
            return True
        if revision <= self._applied_session_revision:
            return False

        snapshot = self._snapshot_for_delta(
            delta,
            revision,
            context=target_context,
        )
        if snapshot.session_revision < revision:
            return False
        if snapshot.session_revision > revision:
            # A newer transition was already accepted.  Project only the newest
            # state and let the older delta become an ordered no-op.
            revision = snapshot.session_revision

        if geometry_preview is not None:
            if type(geometry_preview) is not GeometryPreview:
                raise TypeError("geometry_preview must be a GeometryPreview")
            if (
                snapshot.session_revision == requested_revision
                and snapshot.source_kind == "native"
                and snapshot.artifact is None
                and snapshot.active_part_id is not None
            ):
                self._geometry_preview_cache = (
                    snapshot.session_id,
                    self._native_part_preview_cache_key(snapshot),
                    geometry_preview,
                )

        previous_artifact_id = (
            self.document.artifact.artifact_id
            if self.document.artifact is not None
            else None
        )
        next_artifact_id = (
            snapshot.artifact.artifact_id
            if snapshot.artifact is not None
            else None
        )
        target_cache = target_context.presentation_cache
        if previous_artifact_id != next_artifact_id:
            target_cache.invalidate_model()
        self.document = snapshot
        self._legacy_project_extension = bool(
            snapshot.source_kind == "native"
            and snapshot.project_path is not None
            and snapshot.project_path.suffix.casefold()
            in LEGACY_MODEL_FILE_SUFFIXES
        )
        if snapshot.source_kind == "result":
            stale_agent_proposals = self.agent_authoring_bridge.unbind_context(
                "结果只读文档不支持 Agent 建模操作"
            )
            self.agent_authoring_controller.invalidate_binding(
                "结果只读文档不支持 Agent 建模操作"
            )
            current_authoring_context = None
        else:
            stale_agent_proposals = self.agent_authoring_bridge.bind_snapshot(
                snapshot,
                document_id=target_context.document_id,
            )
            current_authoring_context = self.agent_authoring_bridge.context
        if current_authoring_context is not None:
            self.agent_authoring_controller.observe_binding(
                current_authoring_context,
                proposal_staled=bool(stale_agent_proposals),
                saved_state_transition=(
                    isinstance(delta, SessionDelta)
                    and delta.changed
                    == frozenset({ChangeKind.SAVED_STATE})
                ),
                project_save_terminal_transition=(
                    isinstance(delta, SessionDelta)
                    and delta.changed
                    == frozenset({ChangeKind.SESSION})
                    and delta.reason
                    in {
                        "project_save task failed",
                        "project_save task cancelled",
                    }
                ),
            )
        if hasattr(self, "viewport_panel"):
            self.viewport_panel.agent_chat_drawer.refresh_authoring_binding()
        if (
            snapshot.artifact is None
            or previous_artifact_id
            != snapshot.artifact.artifact_id
        ):
            self._scope_selection_topology_cache = None
            self._mesh_selection_topology_cache = None

        step_names = self.session.runnable_step_names()
        if self._current_step_name not in step_names:
            self._current_step_name = self.session.default_step_name()
        self._symbol_settings = replace(
            self._symbol_settings,
            step_name=self._current_step_name,
        )

        artifact = snapshot.artifact
        if artifact is None:
            self._clear_model_projection(clear_tree=False)
            self._refresh_model_tree_for_context(target_context)
            recipe = snapshot.geometry_recipe
            canonical_parts = (
                tuple(snapshot.parts)
                if snapshot.source_kind == "native"
                and all(
                    part.geometry_recipe is not None
                    for part in snapshot.parts
                )
                else ()
            )
            if canonical_parts:
                visible_parts = tuple(
                    part
                    for part in canonical_parts
                    if (
                        part.id == snapshot.active_part_id
                        and not part.suppressed
                    )
                )
                preview_key = self._native_part_preview_cache_key(snapshot)
                cached = self._geometry_preview_cache
                cached_is_current = (
                    cached is not None
                    and cached[0] == snapshot.session_id
                    and cached[1] == preview_key
                )
                preview = (
                    cached[2]
                    if cached_is_current
                    else build_multi_part_geometry_preview(visible_parts)
                )
                if not cached_is_current:
                    self._geometry_preview_cache = (
                        snapshot.session_id,
                        preview_key,
                        preview,
                    )
                selectable_ids = {
                    logical_id
                    for values in (
                        preview.face_logical_ids,
                        preview.edge_logical_ids,
                        preview.point_logical_ids,
                        preview.face_body_logical_ids,
                        preview.edge_body_logical_ids,
                        preview.point_body_logical_ids,
                    )
                    for logical_id in values
                    if logical_id is not None
                }
                self._selected_geometry_refs = {
                    reference
                    for reference in self._selected_geometry_refs
                    if reference.logical_id in selectable_ids
                }
                suppressed_preview = (
                    build_multi_part_geometry_preview(
                        tuple(
                            part
                            for part in canonical_parts
                            if part.suppressed
                        ),
                        include_suppressed=True,
                    )
                    if self._show_suppressed_part_ghosts
                    and any(part.suppressed for part in canonical_parts)
                    else None
                )
                self.viewport.set_geometry_ghost_preview(
                    suppressed_preview
                )
                if (
                    recipe is not None
                    and geometry_dimension(recipe) == 1
                    and self._geometry_selection_mode == "face"
                ):
                    self._selected_geometry_refs.clear()
                    self._geometry_selection_mode = "body"
                    self._selection_context.geometry_filter = "body"
                    if self._selection_context.space == "geometry":
                        self.actions["select_body"].setChecked(True)
                        self.viewport.set_selection_mode(
                            "geometry_body",
                            render=not self._workspace_activation,
                        )
                try:
                    if (
                        self._sketch_editor_controller is not None
                        or self._wire_editor_controller is not None
                    ):
                        self.viewport.show_geometry_preview(
                            preview,
                            render=False,
                            reset_camera=not self._workspace_activation,
                        )
                    else:
                        self.viewport.show_geometry_preview(
                            preview,
                            render=not self._workspace_activation,
                            reset_camera=not self._workspace_activation,
                        )
                    if (
                        not cached_is_current
                        and recipe is not None
                        and self._recipe_contains_strict_boolean(recipe)
                    ):
                        self._schedule_exact_boolean_preview(
                            snapshot,
                            recipe,
                        )
                finally:
                    # The Session transition is already committed.  Keep action
                    # gates aligned with it even when the optional renderer
                    # cannot display the preview.
                    self._update_action_states()
                self.viewport_panel.set_geometry_context(True)
                active = snapshot.active_part
                self.status_panel.set_object(
                    "—"
                    if active is None
                    else active.name
                )
            elif snapshot.source_kind == "native":
                self.viewport.clear_model()
                self.viewport_panel.set_geometry_context(True)
                self.status_panel.set_object("尚未创建部件")
            else:
                self.viewport_panel.set_geometry_context(False)
                self.status_panel.set_object()
        elif (
            previous_artifact_id != artifact.artifact_id
            or self.geometry is None
            or self.geometry.artifact_id != artifact.artifact_id
            or self.viewport.artifact_id != artifact.artifact_id
        ):
            if snapshot.source_kind == "result":
                if result_model_view is not None:
                    self._result_archive_model_view = result_model_view
                if model_geometry is not None:
                    self._result_archive_geometry = model_geometry
                geometry = (
                    model_geometry
                    or self._result_archive_geometry
                )
                model_for_view = (
                    result_model_view
                    or self._result_archive_model_view
                )
                if geometry is None or model_for_view is None:
                    raise RuntimeError(
                        "result-only projection requires worker display payload"
                    )
            else:
                if model_geometry is None and target_cache.matches_artifact(
                    artifact.artifact_id
                ):
                    model_geometry = target_cache.model_geometry
                geometry = model_geometry or build_model_geometry(artifact.model)
            if geometry.artifact_id != artifact.artifact_id:
                geometry = replace(geometry, artifact_id=artifact.artifact_id)
            if snapshot.source_kind != "result":
                model_for_view = artifact.model
            cached_inspection = (
                target_cache.inspection_service
                if target_cache.matches_artifact(artifact.artifact_id)
                else None
            )
            self._install_model(
                model_for_view,
                geometry,
                dict(timings or {}),
                source_label=source_label or self._session_source_label(),
                inspection_service=cached_inspection,
                render=not self._workspace_activation,
                reset_camera=not self._workspace_activation,
                preserve_display=self._workspace_activation,
            )
            target_cache.artifact_id = artifact.artifact_id
            target_cache.model_geometry = geometry
            target_cache.inspection_service = self.inspection_service
        else:
            # The model is already installed; update only this document's
            # tree root for definition/result metadata changes.
            self._refresh_model_tree_for_context(target_context)

        result_identity = self.session.current_result_identity()
        if result_identity is None:
            self._clear_result_projection()
            target_cache.invalidate_result()
        else:
            source, generation = result_identity
            provider = self.result_provider
            provider_changed = (
                type(provider) is not ResultProvider
                or provider.source != source
                or provider.snapshot.generation != generation
            )
            if provider_changed:
                provider = self.session.current_result_provider()
                if provider is None:
                    raise RuntimeError(
                        "current result identity has no provider"
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
                and self.result_tree.has_selection(
                    self.result_selection,
                    document_id=target_context.document_id,
                    source=(None if provider is None else provider.source),
                )
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
            target_cache.result_source = source
            target_cache.result_generation = generation
            target_cache.result_model_view = self._result_archive_model_view
            target_cache.inspection_service = self.inspection_service

        self._sync_step_combos()
        self._refresh_result_controls()
        self._update_action_states()
        self._refresh_result_tree_for_context(target_context)
        if not self._workspace_activation:
            self._project_viewport_for_module(self._current_module_name())
        self.model_tree.set_active_document(target_context.document_id)
        self._applied_session_revision = revision
        return True

    def _snapshot_for_delta(
        self,
        delta: object,
        revision: int,
        *,
        context: WorkspaceDocument | None = None,
    ) -> object:
        """Project one ordered delta without detaching full model/results."""

        target_context = (
            context
            if context is not None
            else self.workspace.active_document()
        )
        if target_context is None:
            return self.session.projection_snapshot()

        if (
            isinstance(delta, SessionDelta)
            and target_context.session.session_revision == revision
        ):
            return target_context.session.projection_snapshot(
                target_context.projection,
                delta.changed,
            )
        return target_context.session.projection_snapshot()

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
            return str(getattr(recipe, "name", "") or "模型-1")
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

    def _clear_model_projection(self, *, clear_tree: bool = True) -> None:
        self._close_inspection_windows()
        self._close_job_manager()
        self._restore_temporary_selection_context()
        self._pending_analysis_selection = None
        self._pending_analysis_requested_scope_kind = None
        self._pending_analysis_dialog_state = None
        self._pending_scope_kind = None
        self._pending_analysis_edit = None
        self._pending_local_mesh_selection = False
        self._temporary_selection_context = None
        self._temporary_selection_owner = None
        self._selected_geometry_refs.clear()
        self._selected_mesh_scope_refs.clear()
        self._scope_selection_overlay_active = False
        self.viewport_panel.scope_creation_bar.finish()
        self.viewport_panel.planar_boolean_face_bar.finish()
        self.inspection_service = None
        self.geometry = None
        self._result_archive_geometry = None
        self._result_archive_model_view = None
        self.result_provider = None
        self.result_selection = None
        self._result_visualization_provider_cache = None
        self.selection.clear()
        self._display = DisplayState()
        if clear_tree:
            active_context = self.workspace.active_document()
            if active_context is None:
                self.model_tree.clear_model()
            else:
                self._refresh_model_tree_for_context(active_context)
        active_context = self.workspace.active_document()
        if active_context is not None:
            self._refresh_result_tree_for_context(active_context)
        elif self.result_tree.roots:
            # Empty workspaces retain the historical placeholder.
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
        self._result_visualization_provider_cache = None
        self._display = DisplayState()
        active_context = self.workspace.active_document()
        if active_context is not None:
            self._refresh_result_tree_for_context(active_context)
        else:
            self.result_tree.clear_result()
        self.navigation.show_model()
        self.status_panel.set_result()
        if not had_projection:
            return
        self.selection.clear()
        self._selected_geometry_refs.clear()
        self._selected_mesh_scope_refs.clear()
        self.viewport.clear_selection()
        if self.document.source_kind == "result":
            self.viewport.clear_model()
            return
        if self.document.artifact is not None and self.geometry is not None:
            self.viewport.set_model(
                self.document.artifact.model,
                self.geometry,
                refresh_symbols=False,
                render=False,
                mesh_selection_topology_provider=(
                    self._mesh_selection_topology
                ),
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
        identity = self.session.current_result_identity()
        if (
            identity is None
            or provider.source != identity[0]
            or provider.snapshot.generation != identity[1]
        ):
            raise RuntimeError(
                "provider does not match the current accepted result"
            )
        source = provider.source
        previous_provider = self.result_provider
        if (
            type(previous_provider) is ResultProvider
            and previous_provider.source != provider.source
        ):
            self.selection.clear()
            self._selected_geometry_refs.clear()
            self._selected_mesh_scope_refs.clear()
            self.viewport.clear_selection()
        cached = self._result_visualization_provider_cache
        if (
            cached is not None
            and (
                cached[0] != provider.source
                or cached[1] != provider.snapshot.generation
            )
        ):
            self._result_visualization_provider_cache = None

        catalog = provider.catalog()
        if not catalog.fields:
            inspection = self.inspection_service
            self._clear_result_projection()
            self.result_provider = provider
            self.result_selection = None
            active_context = self.workspace.active_document()
            if active_context is not None:
                self._refresh_result_tree_for_context(
                    active_context,
                    catalog=catalog,
                )
            if inspection is not None:
                inspection.update_result_provider(provider)
            self.status_panel.set_result()
            return
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
        render_provider, render_selection = (
            self._result_visualization_provider(provider, selection)
        )
        payload = self._build_result_render_payload(
            render_provider,
            render_selection,
        )
        self._prepare_viewport_for_result_source(source)
        previous_catalog = self.result_tree.catalog
        previous_selection = self.result_selection
        inspection = self.inspection_service
        previous_inspection_provider = (
            None if inspection is None else inspection.result_provider
        )
        try:
            active_context = self.workspace.active_document()
            if active_context is not None:
                self._refresh_result_tree_for_context(
                    active_context,
                    catalog=catalog,
                )
            if not self.result_tree.select_selection(
                selection,
                document_id=active_context.document_id
                if active_context is not None
                else None,
                source=provider.source,
            ):
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
                    active_context = self.workspace.active_document()
                    if active_context is not None:
                        self._refresh_result_tree_for_context(active_context)
                else:
                    active_context = self.workspace.active_document()
                    if active_context is not None:
                        self._refresh_result_tree_for_context(
                            active_context,
                            catalog=previous_catalog,
                        )
                    if type(previous_selection) is ScalarFieldSelection:
                        self.result_tree.select_selection(
                            previous_selection,
                            document_id=active_context.document_id,
                            source=previous_catalog.source,
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
        visual_selection = self._result_averaging_visual_selection(
            provider,
            selection,
        )
        if (
            visual_selection != selection
            and render_selection != visual_selection
        ):
            self._defer_ui(self._apply_result_averaging_threshold)

    def _prepare_viewport_for_result_source(
        self,
        source: ResultSourceKey,
    ) -> None:
        """Detach a different model scene before installing run-owned mesh."""

        if self.viewport.artifact_id not in {None, source.artifact_id}:
            self.viewport.clear_model()

    def _build_actions(self) -> None:
        self.actions = build_actions(self)

    def _build_menus(self) -> None:
        file_menu = self.menuBar().addMenu("文件")
        file_menu.setObjectName("menuFile")
        file_menu.addActions([
            self.actions[name]
            for name in (
                "new_native",
                "open_project",
                "save_project",
                "save_project_as",
                "open",
                "save_result",
                "save_result_as",
                "open_result",
            )
        ])
        file_menu.addSeparator()
        file_menu.addAction(self.actions["exit"])
        edit_menu = self.menuBar().addMenu("编辑")
        edit_menu.setObjectName("menuEdit")
        edit_menu.addActions([
            self.actions[name]
            for name in (
                "select_point", "select_element", "select_edge", "select_face", "select_body",
            )
        ])
        view_menu = self.menuBar().addMenu("视图")
        view_menu.setObjectName("menuView")
        view_menu.addActions([self.actions[name] for name in (
            "fit", "front", "back", "top", "bottom", "left", "right", "iso",
        )])
        view_menu.addSeparator()
        view_menu.addActions([self.actions[name] for name in ("orthographic", "perspective")])
        view_menu.addSeparator()
        view_menu.addActions([self.actions[name] for name in ("edges", "nodes", "node_labels", "element_labels", "symbols")])
        view_menu.addSeparator()
        view_menu.addActions([
            self.actions["symbol_settings"],
            self.actions["viewport_background"],
            self.actions["suppressed_part_ghosts"],
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
        result_menu.addActions([
            self.actions[name]
            for name in (
                "overlay",
                "display_settings",
                "scale",
                "contour_options",
                "query",
                "export_csv",
                "export_vtk",
                "screenshot",
            )
        ])
        help_menu = self.menuBar().addMenu("帮助")
        help_menu.setObjectName("menuHelp")
        help_menu.addAction(self.actions["about"])

    def _build_ribbon(self) -> None:
        self.ribbon = RibbonWidget(self)
        scope_group = (
            "作用域",
            ("geometry_region", "geometry_regions"),
            (),
        )
        self._add_ribbon_page("项目", (
            (
                "文件",
                (
                    "new_native",
                    "open_project",
                    "save_project",
                    "open",
                    "save_result",
                    "open_result",
                    "model_info",
                ),
                ("new_native", "open"),
            ),
            ("输出", ("export_csv", "screenshot"), ()),
        ))
        self._add_ribbon_page("几何", (
            (
                "创建",
                ("geometry_create",),
                ("geometry_create",),
            ),
            (
                "特征",
                (
                    "geometry_extrude",
                    "geometry_sweep",
                    "geometry_move",
                    "geometry_rotate",
                    "geometry_fuse",
                    "geometry_cut",
                ),
                ("geometry_extrude", "geometry_sweep"),
            ),
            (
                "选择",
                (
                    "select_point", "select_edge", "select_face", "select_body",
                ),
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
            scope_group,
            (
                "检查",
                ("mesh_verify", "mesh_statistics"),
                (),
            ),
        ))
        self._add_ribbon_page("模型", (
            ("定义", ("material_manager", "section_manager", "section_assign"), ("material_manager",)),
            scope_group,
            (
                "选择",
                (
                    "select_point", "select_element", "select_edge",
                    "select_face", "select_body",
                ),
                (),
            ),
            ("显示", ("nodes", "edges", "node_labels", "element_labels"), ()),
            ("符号", ("symbols", "symbol_settings"), ()),
        ))
        self._build_analysis_ribbon_page()
        self._build_result_ribbon_page()
        self._add_ribbon_page("视图", (
            ("视角", ("front", "back", "top", "bottom", "left", "right", "iso"),
             ("front", "back", "top", "bottom", "left", "right", "iso")),
            ("相机", ("fit", "orthographic", "perspective", "viewport_background"), ()),
            ("标注", ("nodes", "edges", "node_labels", "element_labels", "symbols"), ()),
        ))
        self.ribbon.moduleChanged.connect(self._on_module_changed)

    def _build_analysis_ribbon_page(self) -> None:
        page = self.ribbon.add_page("分析")

        step_group = page.add_group("分析步")
        for name in ("step_create", "step_info"):
            step_group.add_action(self.actions[name])
        step_group.add_widget(self._create_step_combo("分析"))
        step_group.add_action(self.actions["output_create"], compact=True)

        scope_group = page.add_group("作用域")
        for name in ("geometry_region", "geometry_regions"):
            scope_group.add_action(self.actions[name])

        boundary_group = page.add_group("边界条件")
        for name in ("boundary_create", "load_create"):
            boundary_group.add_action(self.actions[name])

        job_group = page.add_group("作业")
        for name in (
            "check_model",
            "submit_job",
            "analysis_manager",
            "job_manager",
        ):
            job_group.add_action(self.actions[name])

    def _build_result_ribbon_page(self) -> None:
        page = self.ribbon.add_page("结果")
        shape_group = page.add_group("形状")
        for name in ("undeformed", "deformed"):
            shape_group.add_action(self.actions[name])
        contour_group = page.add_group("云图")
        contour_group.add_action(self.actions["contour"], large=True)

        field_group = page.add_group("主变量")
        field_host = _PreferredWidthHost(312, 246, field_group)
        field_host.setObjectName("resultFieldControls")
        field_host.setMaximumWidth(390)
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
        self.result_averaging_threshold = CompactDoubleSpinBox(
            field_host,
            minimum_display_decimals=0,
        )
        self.result_averaging_threshold.setObjectName(
            "resultAveragingThreshold"
        )
        self.result_averaging_threshold.setRange(0.0, 100.0)
        self.result_averaging_threshold.setDecimals(0)
        self.result_averaging_threshold.setSingleStep(1.0)
        self.result_averaging_threshold.setKeyboardTracking(False)
        self.result_averaging_threshold.setValue(
            float(self._contour_options["averaging_threshold"])
        )
        self.result_averaging_threshold.setFixedHeight(24)
        self.result_averaging_threshold.setFixedWidth(58)
        self.result_averaging_threshold.setToolTip(
            "仅控制平均节点应力云图；不改变查询或 CSV 导出的结果数据。"
        )
        for combo in (
            self.result_variable_combo,
            self.result_component_combo,
            self.result_position_combo,
        ):
            combo.setFixedHeight(24)
            combo.setEnabled(False)
        self.result_variable_combo.setMinimumWidth(76)
        component_minimum_width = max(
            138,
            (
                self.result_component_combo.fontMetrics().horizontalAdvance(
                    "MaxPrincipal"
                )
                + 44
            ),
        )
        self.result_component_combo.setMinimumWidth(
            component_minimum_width
        )
        self.result_component_combo.view().setMinimumWidth(
            component_minimum_width
        )
        self.result_position_combo.setMinimumWidth(90)
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
        self.result_averaging_threshold_label = QLabel(
            "阈值（%）",
            field_host,
        )
        for label in (variable_label, component_label, position_label):
            label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            label.setFixedSize(36, 24)
        self.result_averaging_threshold_label.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        self.result_averaging_threshold_label.setFixedHeight(24)
        position_host = QWidget(field_host)
        position_layout = QHBoxLayout(position_host)
        position_layout.setContentsMargins(0, 0, 0, 0)
        position_layout.setSpacing(5)
        position_layout.addWidget(self.result_position_combo, 1)
        position_layout.addWidget(self.result_averaging_threshold_label)
        position_layout.addWidget(self.result_averaging_threshold)
        field_layout.addWidget(variable_label, 0, 0)
        field_layout.addWidget(self.result_variable_combo, 0, 1)
        field_layout.addWidget(component_label, 0, 2)
        field_layout.addWidget(self.result_component_combo, 0, 3)
        field_layout.addWidget(position_label, 1, 0)
        field_layout.addWidget(position_host, 1, 1, 1, 3)
        field_layout.setColumnStretch(1, 1)
        field_layout.setColumnStretch(3, 2)
        field_group.add_widget(field_host)
        self.result_variable_combo.activated.connect(self._result_variable_changed)
        self.result_component_combo.activated.connect(self._result_component_changed)
        self.result_position_combo.activated.connect(self._result_position_changed)
        self.result_averaging_threshold.valueChanged.connect(
            self._result_averaging_threshold_changed
        )
        self._sync_result_averaging_threshold_control()

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
        self.result_scale_value.setDecimals(2)
        self.result_scale_value.setValue(self._scale_value)
        self.result_scale_value.setFixedWidth(100)
        self.result_scale_value.setEnabled(False)
        scale_layout.addWidget(self.result_scale_combo, 0, 0)
        scale_layout.addWidget(self.result_scale_value, 1, 0)
        deformation_group.add_widget(scale_host)
        deformation_group.add_action(self.actions["overlay"], large=True)
        self.result_scale_combo.activated.connect(self._result_scale_mode_changed)
        self.result_scale_value.valueChanged.connect(self._result_scale_value_changed)

        display_group = page.add_group("设置")
        display_group.add_action(self.actions["display_settings"])
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
        combo.setMinimumWidth(100)
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
        self.viewport_panel = ViewportPanel(
            self.viewport,
            self.actions,
            self,
            authoring_bridge=self.agent_authoring_bridge,
            authoring_controller=self.agent_authoring_controller,
        )
        self.viewport_panel.overlay_host.viewportGeometryCommitted.connect(
            self._viewport_geometry_committed
        )
        self.viewport_panel.agent_chat_drawer.set_project_save_handler(
            self._start_agent_project_save
        )
        self.viewport_panel.scope_creation_bar.createRequested.connect(
            self._complete_scope_creation_from_bar
        )
        self.viewport_panel.scope_creation_bar.cancelRequested.connect(
            self._cancel_guided_selection
        )
        self.viewport_panel.planar_boolean_face_bar.cancelRequested.connect(
            self._cancel_boolean_face_prompt
        )
        self.viewport_panel.planar_boolean_face_bar.confirmRequested.connect(
            self._confirm_boolean_face_prompt
        )
        self.wire_editor_panel = WireEditorPanel(parent=self)
        self.wire_editor_panel.hide()
        self.sketch_editor_panel = SketchEditorPanel(
            parent=self,
            settings=self._application_settings,
        )
        self.sketch_editor_panel.hide()
        self.boolean_feature_panel = BooleanFeaturePanel(parent=self)
        self.planar_boolean_panel = PlanarBooleanPanel(parent=self)
        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        splitter.setObjectName("mainSplitter")
        splitter.addWidget(self.navigation)
        splitter.addWidget(self.viewport_panel)
        splitter.addWidget(self.wire_editor_panel)
        splitter.addWidget(self.sketch_editor_panel)
        splitter.addWidget(self.boolean_feature_panel)
        splitter.addWidget(self.planar_boolean_panel)
        # Keep the VTK viewport at a stable size while the user drags a handle.
        # Qt shows a rubber-band preview and commits the new layout on release.
        splitter.setOpaqueResize(False)
        splitter.splitterMoved.connect(self._main_splitter_moved)
        splitter.setSizes([260, 1020, 0, 0, 0, 0])
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        splitter.setStretchFactor(3, 0)
        splitter.setStretchFactor(4, 0)
        splitter.setStretchFactor(5, 0)
        splitter.setCollapsible(0, False)
        splitter.setCollapsible(2, True)
        splitter.setCollapsible(3, True)
        splitter.setCollapsible(4, True)
        splitter.setCollapsible(5, True)
        self.main_splitter = splitter
        host = QWidget(self)
        host.setObjectName("centralWorkspace")
        layout = QVBoxLayout(host)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.ribbon)
        layout.addWidget(splitter, 1)
        self.setCentralWidget(host)
        self.model_tree.highlightRequested[int, str, object].connect(
            self._highlight_tree_entry
        )
        self.model_tree.highlightResetRequested.connect(
            self._reset_tree_entry_highlight
        )
        self.model_tree.informationRequested[int, str, object].connect(
            self._show_entry_information
        )
        self.model_tree.editRequested[int, str, object].connect(
            self._edit_tree_entry
        )
        self.model_tree.deleteRequested[int, str, object].connect(
            self._delete_tree_entry
        )
        self.model_tree.renameRequested[int, str, object].connect(
            self._rename_tree_entry
        )
        self.model_tree.rootActionRequested.connect(
            self._model_root_action_requested
        )
        self.result_tree.fieldSelectionRouted.connect(
            self._activate_routed_result_selection
        )
        self.result_tree.runActivated.connect(
            self._activate_routed_result_run
        )
        self.result_tree.rootActionRequested.connect(
            self._result_root_action_requested
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
        self.boolean_feature_panel.selectionRequested.connect(
            self._request_body_boolean_selection
        )
        self.boolean_feature_panel.operationChanged.connect(
            self._body_boolean_operation_changed
        )
        self.boolean_feature_panel.finishRequested.connect(
            self.finish_body_boolean
        )
        self.boolean_feature_panel.cancelRequested.connect(
            self.cancel_body_boolean
        )
        self.planar_boolean_panel.targetSelectionRequested.connect(
            self._request_planar_boolean_target
        )
        self.planar_boolean_panel.targetSelectionCleared.connect(
            self._clear_planar_boolean_target
        )
        self.planar_boolean_panel.toolSketchRequested.connect(
            self._edit_planar_boolean_tool
        )
        self.planar_boolean_panel.toolSketchDeleted.connect(
            self._delete_planar_boolean_tool
        )
        self.planar_boolean_panel.operationChanged.connect(
            self._planar_boolean_operation_changed
        )
        self.planar_boolean_panel.finishRequested.connect(
            self.finish_planar_boolean
        )
        self.planar_boolean_panel.cancelRequested.connect(
            self.cancel_planar_boolean
        )

    def _main_splitter_moved(self, _position: int, _index: int) -> None:
        """Clear the non-opaque drag marker before repainting the VTK surface."""

        self._clear_main_splitter_drag_marker()
        self.viewport.schedule_resize_repaint()

    def _viewport_geometry_committed(self) -> None:
        """Coalesce drawer resizes with QtInteractor's queued resize paint."""

        self.viewport.schedule_resize_repaint()

    def _clear_main_splitter_drag_marker(self) -> None:
        self.main_splitter.setRubberBand(-1)
        for marker in self.main_splitter.findChildren(QRubberBand):
            if marker.parent() is self.main_splitter:
                marker.hide()
        self.main_splitter.update()

    def _build_status_bar(self) -> None:
        self.status_panel = CAEStatusBar(self)
        self.status_panel.cancelRequested.connect(self.cancel_current_task)
        self.wire_editor_panel.statusChanged.connect(
            lambda message: self.status_panel.set_state(message, 5000)
        )
        self.sketch_editor_panel.statusChanged.connect(
            lambda message: self.status_panel.set_state(message, 5000)
        )
        self.planar_boolean_panel.statusChanged.connect(
            lambda message: self.status_panel.set_state(message, 5000)
        )
        self.setStatusBar(self.status_panel)

    def _on_module_changed(self, module_name: str) -> None:
        if self._temporary_selection_context is None:
            self._set_selection_space(
                "geometry" if module_name == "几何" else "mesh"
            )
        if module_name == "结果":
            self.navigation.show_result()
        elif module_name in {"项目", "几何", "网格", "模型", "分析"}:
            self.navigation.show_model()
        self._project_viewport_for_module(module_name)

    def _current_module_name(self) -> str:
        tab_bar = getattr(getattr(self, "ribbon", None), "tab_bar", None)
        if tab_bar is None or tab_bar.currentIndex() < 0:
            return ""
        return tab_bar.tabText(tab_bar.currentIndex())

    def _project_viewport_for_module(
        self,
        module_name: str,
        *,
        render: bool = True,
        reset_camera: bool = True,
    ) -> None:
        """Project the stable geometry, mesh, or result scene for a module."""

        if self._temporary_selection_context is not None or any(
            controller is not None
            for controller in (
                self._wire_editor_controller,
                self._sketch_editor_controller,
                self._body_boolean_controller,
                self._planar_boolean_controller,
                self._face_sketch_controller,
            )
        ):
            return
        if module_name == "结果":
            provider = self._current_result_provider()
            selection = self.result_selection
            if provider is not None and type(selection) is ScalarFieldSelection:
                if self._viewport_result_scene_is_current(provider, selection):
                    return
                artifact = self.document.artifact
                if (
                    artifact is not None
                    and artifact.artifact_id == provider.source.artifact_id
                ):
                    self._restore_viewport_model_scene(
                        render=False,
                        reset_camera=reset_camera,
                    )
                self._apply_display(render=render)
            else:
                self._project_mesh_or_geometry_fallback(
                    render=render,
                    reset_camera=reset_camera,
                )
            return
        if module_name == "几何":
            preview = self._current_native_geometry_preview()
            if preview is not None:
                self.viewport.show_geometry_preview(
                    preview,
                    render=render,
                    reset_camera=reset_camera,
                )
            else:
                self._project_mesh_or_geometry_fallback(
                    render=render,
                    reset_camera=reset_camera,
                )
            return
        if module_name in {"网格", "模型", "分析"}:
            self._project_mesh_or_geometry_fallback(
                render=render,
                reset_camera=reset_camera,
            )

    def _project_mesh_or_geometry_fallback(
        self,
        *,
        render: bool = True,
        reset_camera: bool = True,
    ) -> None:
        if self.document.artifact is not None and self.geometry is not None:
            self.actions["edges"].setChecked(self._model_edges_visible)
            self._restore_viewport_model_scene(
                render=render,
                reset_camera=reset_camera,
            )
            return
        preview = self._current_native_geometry_preview()
        if preview is not None:
            self.viewport.show_geometry_preview(
                preview,
                render=render,
                reset_camera=reset_camera,
            )
            return
        self.viewport.clear_model()

    def _viewport_result_scene_is_current(
        self,
        provider: ResultProvider,
        selection: ScalarFieldSelection,
    ) -> bool:
        render_provider, render_selection = (
            self._result_visualization_provider(provider, selection)
        )
        return self.viewport.result_scene_is_current(
            render_provider.source,
            render_selection,
            materialization_generation=render_provider.snapshot.generation,
            deformation_scale=self._result_deformation_scale(render_provider),
            display=self._display,
        )

    def _current_native_geometry_preview(self) -> GeometryPreview | None:
        if self.document.source_kind != "native":
            return None
        active = self.document.active_part
        if (
            active is None
            or active.suppressed
            or active.geometry_recipe is None
        ):
            return None
        preview_key = self._native_part_preview_cache_key(self.document)
        cached = self._geometry_preview_cache
        preview = (
            cached[2]
            if (
                cached is not None
                and cached[0] == self.document.session_id
                and cached[1] == preview_key
            )
            else build_multi_part_geometry_preview((active,))
        )
        if (
            cached is None
            or cached[0] != self.document.session_id
            or cached[1] != preview_key
        ):
            self._geometry_preview_cache = (
                self.document.session_id,
                preview_key,
                preview,
            )
        suppressed_parts = tuple(
            part
            for part in self.document.parts
            if part.suppressed and part.geometry_recipe is not None
        )
        ghost_preview = (
            build_multi_part_geometry_preview(
                suppressed_parts,
                include_suppressed=True,
            )
            if self._show_suppressed_part_ghosts and suppressed_parts
            else None
        )
        self.viewport.set_geometry_ghost_preview(ghost_preview)
        return preview

    def _step_combo_changed(self, combo: QComboBox) -> None:
        step_name = combo.currentData()
        if step_name is None:
            return
        self._set_current_step(str(step_name))

    def _sync_step_combos(self) -> None:
        names = self.session.runnable_step_names()
        selected = (
            self._current_step_name
            if self._current_step_name in names
            else names[0]
            if names
            else None
        )
        self._current_step_name = selected
        self._symbol_settings = replace(
            self._symbol_settings,
            step_name=selected,
        )
        for combo in self._step_combos:
            combo.blockSignals(True)
            combo.clear()
            if not names:
                combo.addItem("—", None)
            else:
                for name in names:
                    combo.addItem(name, name)
                index = combo.findData(selected)
                combo.setCurrentIndex(index if index >= 0 else 0)
            combo.setEnabled(bool(names) and not self.busy)
            combo.blockSignals(False)

    def _set_current_step(
        self,
        name: str,
        *,
        refresh_viewport: bool = True,
    ) -> None:
        if name not in self.session.runnable_step_names():
            return
        self._current_step_name = name
        self._symbol_settings = replace(self._symbol_settings, step_name=name)
        self.viewport.set_symbol_settings(
            self._symbol_settings,
            refresh=refresh_viewport,
            render=refresh_viewport,
        )
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
            self._sync_result_averaging_threshold_control()
            return

        variables: list[ResultVariable] = []
        for availability in visible_result_fields(
            provider.catalog().fields
        ):
            if availability.state is FieldState.UNAVAILABLE:
                continue
            variable = availability.descriptor.field_id.variable
            if variable not in variables:
                variables.append(variable)
                self.result_variable_combo.addItem(
                    result_variable_label(variable),
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
        self._sync_result_averaging_threshold_control()

    def _sync_result_averaging_threshold_control(self) -> None:
        variable = self.result_variable_combo.currentData()
        field_id = self.result_position_combo.currentData()
        visible = (
            variable is ResultVariable.S
            and type(field_id) is ResultFieldId
            and field_id.position is FieldPosition.RESOLVED_NODAL
        )
        enabled = (
            visible
            and self._current_result_provider() is not None
            and not self.busy
        )
        self.result_averaging_threshold_label.setVisible(visible)
        self.result_averaging_threshold.setVisible(visible)
        self.result_averaging_threshold.setEnabled(enabled)

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

        field_ids: list[ResultFieldId] = []
        section_point_labels = result_provider_section_point_labels(provider)
        for availability in visible_result_fields(
            provider.catalog().fields
        ):
            if availability.state is FieldState.UNAVAILABLE:
                continue
            field_id = availability.descriptor.field_id
            if (
                field_id.variable is variable
                and field_id not in field_ids
            ):
                field_ids.append(field_id)
                self.result_position_combo.addItem(
                    result_field_position_label(
                        field_id,
                        section_point_labels=section_point_labels,
                    ),
                    field_id,
                )
        preferred_field_id = None
        if (
            type(preferred_selection) is ScalarFieldSelection
            and preferred_selection.field_key.request.field_id.variable
            is variable
        ):
            preferred_field_id = preferred_selection.field_key.request.field_id
        position_index = self.result_position_combo.findData(
            preferred_field_id
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
        selected_field_id = self.result_position_combo.currentData()
        if (
            provider is None
            or type(variable) is not ResultVariable
            or type(selected_field_id) is not ResultFieldId
        ):
            self.result_component_combo.addItem("—", None)
            self.result_component_combo.blockSignals(False)
            return

        availabilities = tuple(
            availability
            for availability in visible_result_fields(
                provider.catalog().fields
            )
            if (
                availability.state is not FieldState.UNAVAILABLE
                and availability.descriptor.field_id.variable is variable
                and availability.descriptor.field_id == selected_field_id
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
        self._sync_result_averaging_threshold_control()

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
        self._sync_result_averaging_threshold_control()

    def _result_component_changed(self, _index: int) -> None:
        selection = self.result_component_combo.currentData()
        if type(selection) is not ScalarFieldSelection:
            return
        self._activate_result_selection(selection)

    def _result_averaging_threshold_changed(self, value: float) -> None:
        threshold = float(value)
        if (
            threshold
            == float(self._contour_options["averaging_threshold"])
        ):
            return
        self._contour_options["averaging_threshold"] = threshold
        self.viewport.set_contour_metadata(
            {"averaging_threshold": threshold}
        )
        self._apply_result_averaging_threshold()

    def _result_scale_mode_changed(self, _index: int) -> None:
        self._scale_mode = str(self.result_scale_combo.currentData())
        self._sync_result_scale_control()
        self._apply_scale()

    def _result_scale_value_changed(self, value: float) -> None:
        self._scale_value = float(value)
        if self._scale_mode == "custom":
            self._apply_scale()

    def _sync_result_scale_control(self) -> None:
        provider = self._current_result_provider()
        displayed_value = self._scale_value
        if provider is not None and self._scale_mode != "custom":
            displayed_value = self._result_deformation_scale(
                provider,
                shape_mode="deformed",
                scale_mode=self._scale_mode,
            )
        self.result_scale_value.blockSignals(True)
        self.result_scale_value.setValue(displayed_value)
        self.result_scale_value.blockSignals(False)
        self.result_scale_value.setEnabled(
            provider is not None and self._scale_mode == "custom"
        )

    def _face_sketch_selection_is_valid(self) -> bool:
        if (
            self.busy
            or self._face_sketch_controller is not None
            or self.document.source_kind != "native"
            or len(self._selected_geometry_refs) != 1
        ):
            return False
        reference = next(iter(self._selected_geometry_refs))
        if reference.kind != "face":
            return False
        part_id = part_id_from_logical_id(reference.logical_id)
        if part_id is None:
            part_id = self.document.active_part_id
            local_id = reference.logical_id
        else:
            try:
                local_id = strip_part_logical_id(part_id, reference.logical_id)
            except ValueError:
                return False
        if part_id is None or part_id != self.document.active_part_id:
            return False
        try:
            part = self.document.part(part_id)
            part_revision = self.document.part_revision(part_id)
        except KeyError:
            return False
        recipe = part.geometry_recipe
        if recipe is None or part.suppressed or geometry_dimension(recipe) != 3:
            return False
        key = (
            self.document.session_id,
            part_id,
            part_revision,
            local_id,
        )
        cached = self._face_sketch_selection_cache.get(key)
        if cached is not None:
            return cached
        valid = False
        try:
            with geometry_runtime.model(
                f"{getattr(recipe, 'name', 'solid')}-face-selection",
                dimension=3,
            ) as cad:
                compiled = compile_recipe(cad, recipe)
                resolve_face_workplane(cad, compiled.logical_entities, local_id)
            valid = True
        except (FaceWorkplaneResolutionError, RuntimeError, TypeError, ValueError):
            valid = False
        self._face_sketch_selection_cache[key] = valid
        return valid

    def _update_action_states(self) -> None:
        authoring = self._session_authoring_projection()
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
            boolean_editor_active=(
                self._body_boolean_controller is not None
                or self._planar_boolean_controller is not None
                or self._face_sketch_controller is not None
                or self._solid_face_boolean_operation is not None
            ),
            planar_solid_face_selected=(
                self._face_sketch_selection_is_valid()
            ),
            selection_space=self._selection_context.space,
            selection_filter=self._selection_context.active_filter,
            selection_topological_dimension=(
                geometry_dimension(self.document.geometry_recipe)
                if self._selection_context.space == "geometry"
                and isinstance(
                    self.document.geometry_recipe,
                    NATIVE_GEOMETRY_TYPES,
                )
                else authoring.report.topological_dimension
                if self._selection_context.space == "mesh"
                else None
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
        self._sync_selection_action_state()
        has_result = provider is not None
        self.result_variable_combo.setEnabled(has_result and not self.busy)
        self.result_component_combo.setEnabled(has_result and not self.busy)
        self.result_position_combo.setEnabled(has_result and not self.busy)
        self.result_scale_combo.setEnabled(has_result)
        self._sync_result_scale_control()
        self._sync_result_averaging_threshold_control()
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
        elif source_kind == "result":
            name = (
                self.document.source_path.name
                if self.document.source_path is not None
                else "结果"
            )
            self.setWindowTitle(f"有限元分析 — {name} [结果只读]")
            return
        else:
            recipe_name = str(
                getattr(self.document.geometry_recipe, "name", "") or ""
            )
            name = (
                self.document.project_path.name
                if self.document.project_path is not None
                else recipe_name or "模型-1"
            )
            source_label = "自主"
        dirty_marker = " *" if self.document.dirty else ""
        self.setWindowTitle(
            f"有限元分析 — {name} [{source_label}]{dirty_marker}"
        )

    def _set_action_available(self, name: str, available: bool, _reason: str) -> None:
        """Apply availability while keeping the command's plain label."""
        action = self.actions[name]
        action.setEnabled(bool(available))
        action.setToolTip(action.text())
        action.setStatusTip(action.text())

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
        default_name = (
            f"部件-{self.session.next_native_part_id[1:]}"
        )
        part_name, accepted = QInputDialog.getText(
            self,
            "新建线体部件",
            "部件名称：",
            text=default_name,
        )
        if not accepted or not part_name.strip():
            return
        self._begin_wire_editor(
            None,
            original_recipe=None,
            part_name=part_name.strip(),
        )

    def _begin_wire_editor(
        self,
        root: WireGeometry | None,
        *,
        original_recipe: object | None,
        part_name: str | None = None,
        display_size: float | None = None,
    ) -> None:
        if (
            self._wire_editor_controller is not None
            or self._sketch_editor_controller is not None
            or self._planar_boolean_controller is not None
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
        self._wire_editor_part_name = part_name
        self.wire_editor_panel.set_controller(
            controller,
            base_snapshot=controller.snapshot(),
        )
        self.wire_editor_panel.begin(
            self.viewport,
            display_size=display_size,
        )
        self.main_splitter.setSizes([260, 760, 360, 0, 0, 0])
        self._wire_editor_work_plane_changed(
            str(self.wire_editor_panel.work_plane_combo.currentData())
        )
        self.ribbon.set_current("几何")
        self.status_panel.set_state("线体编辑：添加点并连接杆件", 0)
        self._update_action_states()

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
        if not self._confirm_result_invalidation(preserve_editor=True):
            return
        try:
            if original is None:
                delta = self.session.add_native_part(
                    recipe,
                    name=self._wire_editor_part_name,
                    expected_session_revision=base_revision,
                )
            else:
                active_id = self.document.active_part_id
                if active_id is None:
                    raise RuntimeError("没有当前部件")
                delta = self.session.replace_part_geometry(
                    active_id,
                    recipe,
                    expected_part_revision=self.document.part_revision(
                        active_id
                    ),
                    expected_session_revision=base_revision,
                )
            self._apply_session_delta(delta)
        except (RuntimeError, TypeError, ValueError, KeyError) as error:
            self.wire_editor_panel.show_status(
                f"geometry.part.edit-rejected: {error}"
            )
            return
        self._exit_wire_editor()
        self.status_panel.set_state("线体已创建，请选择桁架或梁单元", 6000)
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
        self._wire_editor_part_name = None
        self.main_splitter.setSizes([260, 1020, 0, 0, 0, 0])
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
        views = {"XY": "front", "XZ": "bottom", "YZ": "left"}
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
                "新建部件",
                "请先新建自主模型；INP 模型不能反向转换为可编辑 CAD。",
            )
            return
        dialog = GeometryCreationDialog(
            self,
            default_part_name=(
                f"部件-{self.session.next_native_part_id[1:]}"
            ),
        )
        if not self._exec_dialog(dialog):
            return
        creation_kind = dialog.creation_kind()
        part_name = dialog.part_name()
        if creation_kind == "3d":
            solid_dialog = BasicSolidCreationDialog(self)
            if not self._exec_dialog(solid_dialog):
                return
            creation_kind = f"3d_{solid_dialog.solid_kind()}"
        if creation_kind == "1d":
            self._begin_wire_editor(
                None,
                original_recipe=None,
                part_name=part_name,
                display_size=dialog.sketch_size(),
            )
        elif creation_kind == "2d":
            self._begin_sketch_editor(
                None,
                original_recipe=None,
                part_name=part_name,
                display_size=dialog.sketch_size(),
            )
        elif creation_kind in {"3d_box", "3d_cylinder"}:
            self._create_basic_solid_part(creation_kind, part_name)
        else:
            raise RuntimeError(
                "geometry creation dialog returned an unknown kind"
            )

    def create_sketch_geometry(self) -> None:
        """Compatibility action: enter the interactive 2D sketch workflow."""

        self.start_sketch_geometry()

    def start_face_sketch_boolean(self) -> None:
        """Resolve the selected solid plane Face before opening a detached draft."""

        if (
            self.busy
            or self._wire_editor_controller is not None
            or self._sketch_editor_controller is not None
            or self._face_sketch_controller is not None
            or not self._face_sketch_selection_is_valid()
        ):
            return
        selected = next(iter(self._selected_geometry_refs))
        part_id = part_id_from_logical_id(selected.logical_id)
        if part_id is None:
            part_id = self.document.active_part_id
            support_face_id = selected.logical_id
        else:
            support_face_id = strip_part_logical_id(
                part_id,
                selected.logical_id,
            )
        if part_id is None:
            return
        part = self.document.part(part_id)
        source_geometry = part.geometry_recipe
        if source_geometry is None:
            return
        session_id = self.document.session_id
        session_revision = self.document.session_revision
        part_revision = self.document.part_revision(part_id)
        original_selection = set(self._selected_geometry_refs)
        self.status_panel.set_state("正在解析草图工作面…", 0)

        def workload(context: TaskContext) -> object:
            context.report("正在解析实体平面工作面和关联参考点…")
            with geometry_runtime.model(
                f"{getattr(source_geometry, 'name', 'solid')}-face-sketch",
                dimension=3,
            ) as cad:
                compiled = compile_recipe(cad, source_geometry)
                workplane = resolve_face_workplane(
                    cad,
                    compiled.logical_entities,
                    support_face_id,
                )
                reference_points = provide_face_reference_points(
                    cad,
                    compiled.logical_entities,
                    workplane,
                )
            context.checkpoint()
            return workplane, reference_points

        def apply_result(payload: object) -> TaskApplyOutcome:
            current = self.document
            if (
                current.session_id != session_id
                or current.session_revision != session_revision
                or current.part_revision(part_id) != part_revision
                or current.part(part_id).geometry_recipe != source_geometry
            ):
                return TaskApplyOutcome.stale("工作面解析结果已过期，未进入草图")
            return TaskApplyOutcome.accepted(payload)

        def on_success(payload: object) -> None:
            workplane, reference_points = payload
            try:
                controller = FaceSupportedSketchController(
                    self.document,
                    part_id,
                    workplane,
                    reference_points=tuple(reference_points),
                )
            except (KeyError, RuntimeError, TypeError, ValueError) as error:
                self._show_error("在面上创建草图", str(error))
                return
            self._face_sketch_controller = controller
            self._face_sketch_reference_points = tuple(reference_points)
            self._face_sketch_original_selection = original_selection
            self._face_sketch_parameters = None
            self._face_sketch_preview_result = None
            self._begin_face_sketch_editor()

        def restore_face_prompt(message: str) -> None:
            operation = self._solid_face_boolean_operation
            if operation is None:
                self._show_error("在面上创建草图", message)
                return
            face_bar = self.viewport_panel.planar_boolean_face_bar
            face_bar.begin(operation.value)
            face_bar.set_selection_ready(
                self._face_sketch_selection_is_valid()
            )
            self.status_panel.set_state(
                f"目标面解析失败，请重新选择：{message}",
                5000,
            )

        started = self._start_task(
            workload,
            on_success,
            "在面上创建草图",
            restore_face_prompt,
            task_name="解析面草图工作面",
            on_cancelled=(
                self._cancel_solid_face_boolean
                if self._solid_face_boolean_operation is not None
                else None
            ),
            apply_result=apply_result,
        )
        if not started and self._solid_face_boolean_operation is not None:
            restore_face_prompt("当前有其他后台任务正在运行")

    def _begin_face_sketch_editor(self) -> None:
        controller = self._face_sketch_controller
        if controller is None:
            return
        draft = controller.draft
        self._sketch_editor_controller = draft
        self._sketch_editor_original_recipe = None
        self._sketch_editor_base_revision = controller.launch_snapshot.session_revision
        self._sketch_editor_part_name = None
        self._sketch_editor_is_face_sketch = True
        self.sketch_editor_panel.set_controller(
            draft,
            base_snapshot=draft.snapshot(),
        )
        self.sketch_editor_panel.set_reference_points(
            self._face_sketch_reference_points,
            refresh_controller=False,
        )
        self.sketch_editor_panel.begin(self.viewport, purpose="face_sketch")
        self.main_splitter.setSizes([260, 720, 0, 400, 0, 0])
        launch = controller.launch_snapshot
        cached = self._geometry_preview_cache
        preview_key = self._native_part_preview_cache_key(self.document)
        source_preview = (
            localize_part_geometry_preview(launch.part_id, cached[2])
            if (
                cached is not None
                and cached[0] == self.document.session_id
                and cached[1] == preview_key
            )
            else build_geometry_preview(launch.part.geometry_recipe)
        )
        self.viewport.show_sketch_reference_preview(
            source_preview,
            support_face_id=launch.workplane.support_face_id,
            target_body_id=launch.workplane.target_body_id,
        )
        self.ribbon.set_current("几何")
        self._schedule_viewport_fit()
        self.status_panel.set_state("面草图编辑：绘制轮廓后创建", 0)
        self._update_action_states()

    def _open_face_sketch_boolean_dialog(self, sketch: SketchGeometry) -> None:
        controller = self._face_sketch_controller
        if controller is None:
            return
        analysis = (
            controller.draft.derive_profiles()
            if type(controller) is FaceSupportedSketchController
            else analyze_sketch_profiles(sketch)
        )
        material_ids = tuple(
            profile.id
            for profile in analysis.profiles
            if profile.is_material
        )
        previous = self._face_sketch_parameters
        selected_ids = (
            material_ids
            if previous is None
            else tuple(
                profile_id
                for profile_id in previous.participating_profile_ids
                if profile_id in material_ids
            )
        )
        if not selected_ids:
            selected_ids = material_ids
        dialog = FaceSketchBooleanDialog(self)
        dialog.set_profiles(material_ids, selected_ids=selected_ids)
        operation = self._solid_face_boolean_operation
        if previous is not None or operation is not None:
            defaults = dialog.parameters()
            if defaults is not None:
                dialog.set_parameters(
                    FaceSketchBooleanParameters(
                        (
                            operation
                            if operation is not None
                            else previous.operation
                        ),
                        (
                            previous.direction
                            if previous is not None
                            else FaceSketchBooleanDirection.INWARD
                            if operation is FaceSketchBooleanOperation.CUT
                            else defaults.direction
                        ),
                        (
                            previous.distance
                            if previous is not None
                            else defaults.distance
                        ),
                        selected_ids,
                    )
                )
        if operation is not None:
            dialog.fix_operation(operation)
        dialog.parametersChanged.connect(
            self._face_sketch_boolean_parameters_changed
        )
        dialog.createFeatureRequested.connect(
            self._request_face_sketch_boolean_feature
        )
        dialog.returnSketchRequested.connect(
            self._return_to_face_sketch
        )
        dialog.cancelRequested.connect(self.cancel_face_sketch_boolean)
        self._face_sketch_dialog = dialog
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()
        self._face_sketch_boolean_parameters_changed(dialog.parameters())
        self.status_panel.set_state("正在准备拉伸布尔预览", 0)
        self._update_action_states()

    def _face_sketch_boolean_parameters_changed(self, payload: object) -> None:
        dialog = self._face_sketch_dialog
        controller = self._face_sketch_controller
        if dialog is None or controller is None:
            return
        self._face_sketch_preview_generation += 1
        generation = self._face_sketch_preview_generation
        self._face_sketch_preview_result = None
        dialog.set_preview_running(generation)
        if type(payload) is not FaceSketchBooleanParameters:
            self._face_sketch_parameters = None
            dialog.set_preview_invalid(generation, dialog.validation_reason())
            return
        self._face_sketch_parameters = payload
        self._launch_face_sketch_boolean_preview(payload, generation)

    def _launch_face_sketch_boolean_preview(
        self,
        parameters: FaceSketchBooleanParameters,
        generation: int,
    ) -> None:
        controller = self._face_sketch_controller
        dialog = self._face_sketch_dialog
        if controller is None or dialog is None:
            return
        if generation != self._face_sketch_preview_generation:
            return
        if self.busy:
            dialog.set_preview_invalid(generation, "等待上一代精确预览结束…")
            return
        try:
            sketch = controller.draft.to_sketch_geometry()
            sketch_snapshot = controller.sketch_snapshot()
            launch = controller.launch_snapshot
            feature_ids = {
                str(record.payload.get("feature_id"))
                for record in launch.part.feature_history
                if record.payload.get("feature_id")
            }
            feature_index = 1
            while f"FSB{feature_index}" in feature_ids:
                feature_index += 1
            operation_kind = (
                "face_sketch_boolean_fuse"
                if parameters.operation is FaceSketchBooleanOperation.FUSE
                else "face_sketch_boolean_cut"
            )
            operation_index = 1 + sum(
                record.kind == operation_kind
                for record in launch.part.feature_history
            )
            recipe = FaceSketchBooleanGeometry(
                launch.part.geometry_recipe,
                f"FSB{feature_index}",
                (
                    f"拉伸合并-{operation_index}"
                    if parameters.operation is FaceSketchBooleanOperation.FUSE
                    else f"拉伸切除-{operation_index}"
                ),
                launch.workplane.support_face_id,
                launch.workplane.strategy,
                sketch,
                parameters.operation,
                parameters.direction,
                parameters.distance,
                parameters.participating_profile_ids,
                sketch_snapshot.external_references,
                sketch_snapshot.external_coincidences,
            )
        except (RuntimeError, TypeError, ValueError) as error:
            dialog.set_preview_invalid(generation, str(error))
            return
        sketch_revision = sketch_snapshot.revision
        dialog.set_preview_running(generation)

        def workload(context: TaskContext) -> FaceSketchBooleanResult:
            context.report("正在执行精确拉伸布尔并验证拓扑…")
            with geometry_runtime.model(
                f"{recipe.name}-{generation}",
                dimension=3,
            ) as cad:
                result = prepare_face_sketch_boolean(cad, recipe)
            context.checkpoint()
            return result

        def apply_result(payload: object) -> TaskApplyOutcome:
            return TaskApplyOutcome.accepted(payload)

        def on_success(payload: object) -> None:
            current = self._face_sketch_controller
            current_dialog = self._face_sketch_dialog
            is_current = (
                type(payload) is FaceSketchBooleanResult
                and current is controller
                and current_dialog is dialog
                and generation == self._face_sketch_preview_generation
                and current.sketch_snapshot().revision == sketch_revision
                and self._face_sketch_parameters == parameters
                and current.launch_is_current(self.document)
            )
            if not is_current:
                self._defer_ui(self._refresh_current_face_sketch_preview)
                return
            display = build_face_sketch_boolean_display(
                sketch,
                parameters.participating_profile_ids,
                launch.workplane.direction_vector(parameters.direction),
                parameters.distance,
            )
            exact = build_face_sketch_boolean_result_preview(payload.preview)
            cached = self._geometry_preview_cache
            preview_key = self._native_part_preview_cache_key(self.document)
            target = (
                localize_part_geometry_preview(launch.part_id, cached[2])
                if (
                    cached is not None
                    and cached[0] == self.document.session_id
                    and cached[1] == preview_key
                )
                else build_geometry_preview(launch.part.geometry_recipe)
            )
            self._face_sketch_preview_result = payload
            self.viewport.show_face_sketch_boolean_preview(
                target,
                display,
                exact,
                target_body_id=launch.workplane.target_body_id,
                origin=launch.workplane.origin,
                direction=launch.workplane.direction_vector(parameters.direction),
                distance=parameters.distance,
                operation_name=parameters.operation.chinese_name,
            )
            dialog.set_preview_valid(generation)

        def on_failure(message: str) -> None:
            if (
                self._face_sketch_controller is controller
                and self._face_sketch_dialog is dialog
                and generation == self._face_sketch_preview_generation
            ):
                self._face_sketch_preview_result = None
                dialog.set_preview_invalid(
                    generation,
                    f"精确预览失败：{message}",
                )
            else:
                self._defer_ui(self._refresh_current_face_sketch_preview)

        started = self._start_task(
            workload,
            on_success,
            "拉伸布尔",
            on_failure,
            task_name="拉伸布尔精确预览",
            on_cancelled=self._refresh_current_face_sketch_preview,
            apply_result=apply_result,
        )
        if not started:
            dialog.set_preview_invalid(generation, "等待上一代精确预览结束…")

    def _refresh_current_face_sketch_preview(self) -> None:
        parameters = self._face_sketch_parameters
        dialog = self._face_sketch_dialog
        if parameters is None or dialog is None:
            return
        generation = self._face_sketch_preview_generation
        if dialog.preview_is_valid:
            return
        self._launch_face_sketch_boolean_preview(parameters, generation)

    def _request_face_sketch_boolean_feature(
        self,
        parameters: object,
        generation: int,
    ) -> None:
        controller = self._face_sketch_controller
        result = self._face_sketch_preview_result
        dialog = self._face_sketch_dialog
        if (
            controller is None
            or result is None
            or dialog is None
            or type(parameters) is not FaceSketchBooleanParameters
            or parameters != self._face_sketch_parameters
            or int(generation) != self._face_sketch_preview_generation
            or not dialog.preview_is_valid
        ):
            return
        request = FaceSketchBooleanFeatureRequest(
            controller.launch_snapshot,
            result.geometry,
            controller.sketch_snapshot().revision,
            int(generation),
        )
        self.faceSketchBooleanFeatureRequested.emit(request)
        self.status_panel.set_state("拉伸布尔待提交", 5000)

    def _commit_face_sketch_boolean_feature(self, payload: object) -> None:
        controller = self._face_sketch_controller
        result = self._face_sketch_preview_result
        dialog = self._face_sketch_dialog
        parameters = self._face_sketch_parameters
        if (
            type(payload) is not FaceSketchBooleanFeatureRequest
            or controller is None
            or result is None
            or dialog is None
            or parameters is None
        ):
            return
        launch = controller.launch_snapshot
        current_sketch = controller.sketch_snapshot()
        request_is_current = (
            payload.launch == launch
            and payload.geometry == result.geometry
            and payload.sketch_revision == current_sketch.revision
            and payload.preview_generation == self._face_sketch_preview_generation
            and dialog.preview_is_valid
            and controller.launch_is_current(self.document)
            and payload.geometry.sketch == controller.draft.to_sketch_geometry()
            and payload.geometry.external_references
            == current_sketch.external_references
            and payload.geometry.external_coincidences
            == current_sketch.external_coincidences
            and payload.geometry.support_face_id == launch.workplane.support_face_id
            and payload.geometry.workplane_strategy == launch.workplane.strategy
            and payload.geometry.operation is parameters.operation
            and payload.geometry.direction is parameters.direction
            and payload.geometry.distance == parameters.distance
            and payload.geometry.participating_profile_ids
            == tuple(sorted(parameters.participating_profile_ids))
        )
        if not request_is_current:
            dialog.set_preview_invalid(
                self._face_sketch_preview_generation,
                "提交前状态已变化，精确预览已过期，请重新生成预览",
            )
            self.status_panel.set_state("状态已变化，请重建预览", 6000)
            return
        if not self._confirm_result_invalidation(preserve_editor=True):
            return
        try:
            delta = self.session.commit_face_sketch_boolean(
                launch.part_id,
                launch.body_id,
                payload.geometry,
                expected_session_id=launch.session_id,
                expected_part_revision=launch.part_revision,
                expected_session_revision=launch.session_revision,
                expected_body_recipe=launch.body.recipe,
                expected_support_face_id=launch.workplane.support_face_id,
                expected_workplane_strategy=launch.workplane.strategy,
                expected_sketch_revision=payload.sketch_revision,
                sketch_revision=current_sketch.revision,
                expected_preview_generation=payload.preview_generation,
                preview_generation=self._face_sketch_preview_generation,
            )
            committed_preview = namespace_part_geometry_preview(
                launch.part_id,
                build_face_sketch_boolean_committed_preview(
                    payload.geometry,
                    result.preview,
                ),
            )
            if not self._apply_session_delta(
                delta,
                geometry_preview=committed_preview,
            ):
                raise RuntimeError("Session 未接受拉伸布尔提交")
        except (KeyError, RuntimeError, TypeError, ValueError) as error:
            dialog.set_preview_invalid(
                self._face_sketch_preview_generation,
                f"提交被拒绝：{error}",
            )
            self.status_panel.set_state(
                f"拉伸布尔提交被拒绝，模型未修改：{error}",
                7000,
            )
            return

        feature_name = self.document.part(launch.part_id).feature_history[-1].name
        dialog.close_for_workflow()
        self._face_sketch_dialog = None
        self._face_sketch_controller = None
        self._face_sketch_parameters = None
        self._face_sketch_preview_result = None
        self._face_sketch_reference_points = ()
        self._face_sketch_original_selection = set()
        self._solid_face_boolean_operation = None
        self._solid_face_boolean_original_selection = set()
        self._selected_geometry_refs.clear()
        self._selected_mesh_scope_refs.clear()
        self.status_panel.set_state(
            f"{feature_name} 已作为一个撤销单元提交",
            6000,
        )
        self._update_action_states()

    def _return_to_face_sketch(self, parameters: object) -> None:
        if (
            type(parameters) is not FaceSketchBooleanParameters
            or self._face_sketch_controller is None
        ):
            return
        self._face_sketch_parameters = parameters
        self._face_sketch_preview_result = None
        self._face_sketch_preview_generation += 1
        dialog = self._face_sketch_dialog
        self._face_sketch_dialog = None
        if dialog is not None:
            dialog.close_for_workflow()
        self._begin_face_sketch_editor()

    def cancel_face_sketch_boolean(self) -> None:
        if self._face_sketch_controller is None:
            return
        self._face_sketch_preview_generation += 1
        original_selection = set(self._face_sketch_original_selection)
        dialog = self._face_sketch_dialog
        self._face_sketch_dialog = None
        if dialog is not None:
            dialog.close_for_workflow()

        def cleanup() -> None:
            if self._sketch_editor_is_face_sketch:
                self._exit_sketch_editor()
            self._face_sketch_controller = None
            self._face_sketch_parameters = None
            self._face_sketch_preview_result = None
            self._face_sketch_reference_points = ()
            self._face_sketch_original_selection = set()
            self._solid_face_boolean_operation = None
            self._solid_face_boolean_original_selection = set()
            self._rebuild_full_projection()
            self._selected_geometry_refs = original_selection
            if original_selection:
                self.viewport.highlight_geometry_entities(
                    tuple(sorted(original_selection, key=logical_ref_sort_key))
                )
            self.status_panel.set_state("拉伸布尔已取消", 4000)
            self._update_action_states()

        if self.task_controller.current_task_name == "拉伸布尔精确预览":
            if self.cancel_current_task(after_cleanup=cleanup):
                return
        cleanup()

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
        current_part_name = (
            f"部件-{self.session.next_native_part_id[1:]}"
        )
        part_name, accepted = QInputDialog.getText(
            self,
            "新建二维草图",
            "部件名称：",
            text=current_part_name,
        )
        if not accepted:
            return
        part_name = part_name.strip()
        if not part_name:
            self._show_error("新建二维草图", "部件名称不能为空。")
            return
        self._begin_sketch_editor(
            None,
            original_recipe=None,
            part_name=part_name,
        )

    def _begin_sketch_editor(
        self,
        root: SketchGeometry | None,
        *,
        original_recipe: object | None,
        part_name: str | None = None,
        display_size: float | None = None,
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
        self._sketch_editor_part_name = part_name
        self.sketch_editor_panel.set_controller(
            controller,
            base_snapshot=controller.snapshot(),
        )
        self.sketch_editor_panel.begin(
            self.viewport,
            display_size=display_size,
        )
        self.main_splitter.setSizes([260, 720, 0, 400, 0, 0])
        self.ribbon.set_current("几何")
        self.viewport.set_view("front")
        self._schedule_viewport_fit()
        self.status_panel.set_state("二维草图编辑：绘制闭合轮廓", 0)
        self._update_action_states()

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
        if self._sketch_editor_is_face_sketch:
            face_controller = self._face_sketch_controller
            if (
                face_controller is None
                or not face_controller.launch_is_current(self.document)
            ):
                self.sketch_editor_panel.show_status(
                    "项目已变化；面草图未应用，请取消后重新开始"
                )
                return
            self._exit_sketch_editor()
            self._open_face_sketch_boolean_dialog(root)
            return
        if self._sketch_editor_is_planar_boolean_tool:
            planar = self._planar_boolean_controller
            if (
                planar is None
                or self.document.session_revision != planar.base_session_revision
                or base_revision != planar.base_session_revision
            ):
                self.sketch_editor_panel.show_status(
                    "项目已变化；工具草图未应用，请取消二维布尔后重试"
                )
                return
            try:
                snapshot = controller.snapshot()
                planar.set_tool_recipe(
                    root,
                    external_references=snapshot.external_references,
                    external_coincidences=snapshot.external_coincidences,
                    unresolved_reference_ids=snapshot.unresolved_reference_ids,
                )
            except (TypeError, ValueError) as error:
                self.sketch_editor_panel.show_status(str(error))
                return
            self._exit_sketch_editor()
            self.status_panel.set_state(
                f"{len(planar.tool_face_ids)} 个轮廓完成，正在执行二维布尔",
                0,
            )
            self._refresh_planar_boolean_preview(auto_commit=True)
            return
        original = self._sketch_editor_original_recipe
        try:
            recipe = (
                self._replace_root_geometry(original, root)
                if original is not None
                else root
            )
        except ExtrusionSourceResolutionError as error:
            self.sketch_editor_panel.show_status(
                f"{error.code}: 所选拉伸 Profile 已失效，请恢复该 Profile "
                "或取消编辑后重新创建拉伸"
            )
            return
        if not self._confirm_result_invalidation(preserve_editor=True):
            return
        rendered_draft = self.viewport._sketch_draft_render_data
        local_preview = (
            GeometryPreview(
                rendered_draft.points,
                rendered_draft.faces,
                rendered_draft.curves,
                tuple(f"face:{value}" for value in rendered_draft.face_ids),
                tuple(f"edge:{value}" for value in rendered_draft.curve_ids),
                tuple(
                    None if value is None else f"point:{value}"
                    for value in rendered_draft.point_ids
                ),
                "body:domain",
                2,
            )
            if (
                rendered_draft is not None
                and rendered_draft.geometry_revision
                == controller.snapshot().revision
                and rendered_draft.faces
            )
            else build_strict_sketch_draft_preview(
                root,
                analysis=controller.derive_profiles(),
            )
        )
        geometry_preview = (
            namespace_part_geometry_preview(
                self.session.next_native_part_id,
                local_preview,
            )
            if original is None
            else None
        )
        try:
            if original is None:
                delta = self.session.add_native_part(
                    recipe,
                    name=self._sketch_editor_part_name,
                    expected_session_revision=base_revision,
                )
            else:
                active_id = self.document.active_part_id
                if active_id is None:
                    raise RuntimeError("没有当前部件")
                delta = self.session.replace_part_geometry(
                    active_id,
                    recipe,
                    expected_part_revision=self.document.part_revision(
                        active_id
                    ),
                    expected_session_revision=base_revision,
                )
            self._apply_session_delta(
                delta,
                geometry_preview=geometry_preview,
            )
        except (RuntimeError, TypeError, ValueError, KeyError) as error:
            self.sketch_editor_panel.show_status(
                f"geometry.part.edit-rejected: {error}"
            )
            return
        self._exit_sketch_editor()
        self.status_panel.set_state("二维草图已创建", 6000)
        self.ribbon.set_current("几何")

    def cancel_sketch_geometry(self) -> None:
        controller = self._sketch_editor_controller
        if controller is None:
            return
        if controller.dirty and not self._confirm_sketch_editor_discard():
            return
        if self._sketch_editor_is_face_sketch:
            self.cancel_face_sketch_boolean()
            return
        tool_mode = self._sketch_editor_is_planar_boolean_tool
        self._exit_sketch_editor()
        if tool_mode:
            self.cancel_planar_boolean()
        else:
            self._rebuild_full_projection()
            self.status_panel.set_state("已取消二维草图编辑", 4000)

    def _exit_sketch_editor(
        self,
        *,
        return_to_planar_boolean: bool = False,
    ) -> None:
        self.sketch_editor_panel.end()
        self._sketch_editor_controller = None
        self._sketch_editor_original_recipe = None
        self._sketch_editor_base_revision = None
        self._sketch_editor_part_name = None
        self._sketch_editor_is_planar_boolean_tool = False
        self._sketch_editor_is_face_sketch = False
        if return_to_planar_boolean:
            self._restore_planar_boolean_source_projection()
        self.main_splitter.setSizes(
            [260, 720, 0, 0, 0, 400]
            if return_to_planar_boolean
            else [260, 1020, 0, 0, 0, 0]
        )
        self._update_action_states()
        self._schedule_viewport_fit()

    def _restore_planar_boolean_source_projection(self) -> None:
        controller = self._planar_boolean_controller
        active_part_id = self.document.active_part_id
        if controller is None or active_part_id is None:
            return
        preview_key = self._native_part_preview_cache_key(self.document)
        cached = self._geometry_preview_cache
        preview = (
            cached[2]
            if (
                cached is not None
                and cached[0] == self.document.session_id
                and cached[1] == preview_key
            )
            else namespace_part_geometry_preview(
                active_part_id,
                build_geometry_preview(controller.geometry),
            )
        )
        self.viewport.show_geometry_preview(preview)
        if controller.target_face_id is None:
            return
        target = LogicalEntityRef(
            namespace_part_logical_id(
                active_part_id,
                controller.target_face_id,
            )
        )
        self._selected_geometry_refs = {target}
        self.viewport.highlight_geometry_entities((target,))

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
            self._add_or_set_solid_source(dialog.recipe(), "长方体")

    def _create_basic_solid_part(
        self,
        creation_kind: str,
        part_name: str,
    ) -> None:
        dialog: BoxGeometryDialog | CylinderGeometryDialog
        if creation_kind == "3d_box":
            dialog = BoxGeometryDialog(parent=self)
        elif creation_kind == "3d_cylinder":
            dialog = CylinderGeometryDialog(parent=self)
        else:
            raise ValueError("不支持的基本实体类型")
        base_revision = self.document.session_revision
        if not self._exec_dialog(dialog):
            return
        if not self._confirm_result_invalidation():
            return
        try:
            delta = self.session.add_native_part(
                dialog.recipe(),
                name=part_name,
                expected_session_revision=base_revision,
            )
        except (RuntimeError, TypeError, ValueError) as error:
            self._show_error("新建部件", str(error))
            return
        self._apply_session_delta(delta)
        self.status_panel.set_state(
            f"{part_name} 已创建并设为当前部件",
            5000,
        )

    def create_cylinder_geometry(self) -> None:
        current = self.document.geometry_recipe
        dialog = CylinderGeometryDialog(
            current if isinstance(current, CylinderGeometry) else None,
            self,
        )
        if self._exec_dialog(dialog):
            self._add_or_set_solid_source(dialog.recipe(), "圆柱")

    def _add_or_set_solid_source(self, recipe: object, label: str) -> None:
        current = self.document.geometry_recipe
        if isinstance(current, MultiBodyGeometry):
            self._set_native_geometry(
                add_solid_body(current, recipe),
                f"新增{label}",
            )
        else:
            self._set_native_geometry(recipe, label)

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
        selected_body_id = self._selected_body_id(current)
        if isinstance(current, MultiBodyGeometry) and selected_body_id is None:
            self.status_panel.set_state("请先选择一个实体", 5000)
            return
        source = (
            current.body(selected_body_id).recipe
            if isinstance(current, MultiBodyGeometry)
            and selected_body_id is not None
            else current
        )
        dialog = MoveGeometryDialog(
            source,
            self,
            is_3d=geometry_dimension(source) != 2,
        )
        if self._exec_dialog(dialog):
            moved = dialog.recipe()
            if isinstance(current, MultiBodyGeometry):
                moved_geometry = replace(
                    current,
                    bodies=tuple(
                        replace(body, recipe=moved)
                        if body.id == selected_body_id
                        else body
                        for body in current.bodies
                    ),
                )
                self._set_native_geometry(moved_geometry, "移动后的")
            else:
                self._set_native_geometry(moved, "移动后的")

    def rotate_geometry(self) -> None:
        current = self.document.geometry_recipe
        if not isinstance(current, NATIVE_GEOMETRY_TYPES):
            return
        selected_body_id = self._selected_body_id(current)
        if isinstance(current, MultiBodyGeometry) and selected_body_id is None:
            self.status_panel.set_state("请先选择一个实体", 5000)
            return
        source = (
            current.body(selected_body_id).recipe
            if isinstance(current, MultiBodyGeometry)
            and selected_body_id is not None
            else current
        )
        dialog = RotateGeometryDialog(
            source,
            self,
            is_3d=geometry_dimension(source) != 2,
        )
        if self._exec_dialog(dialog):
            rotated = dialog.recipe()
            if isinstance(current, MultiBodyGeometry):
                rotated_geometry = replace(
                    current,
                    bodies=tuple(
                        replace(body, recipe=rotated)
                        if body.id == selected_body_id
                        else body
                        for body in current.bodies
                    ),
                )
                self._set_native_geometry(rotated_geometry, "旋转后的")
            else:
                self._set_native_geometry(rotated, "旋转后的")

    def extrude_geometry(self) -> None:
        current = self.document.geometry_recipe
        if (
            not isinstance(current, NATIVE_GEOMETRY_TYPES)
            or geometry_dimension(current) != 2
        ):
            return
        selected = self._canonical_geometry_selection()
        if selected and any(reference.kind != "face" for reference in selected):
            self.status_panel.set_state("当前选择包含非面实体", 5000)
            return
        active_id = self.document.active_part_id
        local_selected = tuple(
            LogicalEntityRef(
                strip_part_logical_id(
                    active_id,
                    reference.logical_id,
                )
            )
            if (
                active_id is not None
                and part_id_from_logical_id(reference.logical_id)
                is not None
            )
            else reference
            for reference in selected
        )
        try:
            all_sources = resolve_extrusion_source_faces(current)
            if local_selected:
                sources = resolve_extrusion_source_faces(
                    current,
                    local_selected,
                )
            elif len(all_sources.face_ids) == 1:
                sources = all_sources
            else:
                self.status_panel.set_state(
                "多轮廓草图：请选择二维面",
                    5000,
                )
                return
        except ExtrusionSourceResolutionError as error:
            self.status_panel.set_state(str(error), 5000)
            return
        base_session_revision = self.document.session_revision
        dialog = ExtrudeGeometryDialog(
            current,
            self,
            source_face_ids=sources.face_ids,
        )
        while self._exec_dialog(dialog):
            recipes = tuple(
                ExtrudedGeometry(
                    current,
                    dialog.height_spin.value(),
                    (face_id,),
                )
                for face_id in sources.face_ids
            )
            try:
                for recipe in recipes:
                    self._preflight_extruded_geometry(recipe)
            except Exception as error:
                logging.exception("extrusion OCC preflight failed")
                self._show_error(
                    "拉伸几何",
                    "临时 OCC 编译失败，当前几何和选择保持不变："
                    f"\n{error}",
                )
                continue
            if len(recipes) == 1:
                self._set_native_geometry(
                    recipes[0],
                    "拉伸实体",
                    base_session_revision=base_session_revision,
                )
            else:
                if active_id is None:
                    self._show_error("拉伸几何", "没有当前部件")
                    return
                if not self._confirm_result_invalidation():
                    return
                try:
                    delta = self.session.replace_part_with_extruded_siblings(
                        active_id,
                        recipes,
                        expected_part_revision=(
                            self.document.part_revision(active_id)
                        ),
                        expected_session_revision=base_session_revision,
                    )
                    self._apply_session_delta(delta)
                except (
                    RuntimeError,
                    TypeError,
                    ValueError,
                    KeyError,
                ) as error:
                    self._show_error("拉伸几何", str(error))
                    return
            return

    @staticmethod
    def _preflight_extruded_geometry(recipe: ExtrudedGeometry) -> None:
        with geometry_runtime.model(
            f"{recipe.name}-extrusion-preflight",
            dimension=3,
        ) as cad:
            compile_recipe(cad, recipe)

    def sweep_geometry(self) -> None:
        current = self.document.geometry_recipe
        if (
            not isinstance(current, NATIVE_GEOMETRY_TYPES)
            or geometry_dimension(current) != 2
        ):
            return
        selected = self._canonical_geometry_selection()
        if selected and any(reference.kind != "face" for reference in selected):
            self.status_panel.set_state("当前选择包含非面实体", 5000)
            return
        active_id = self.document.active_part_id
        local_selected = tuple(
            LogicalEntityRef(
                strip_part_logical_id(
                    active_id,
                    reference.logical_id,
                )
            )
            if (
                active_id is not None
                and part_id_from_logical_id(reference.logical_id)
                is not None
            )
            else reference
            for reference in selected
        )
        try:
            all_sources = resolve_extrusion_source_faces(current)
            if local_selected:
                sources = resolve_extrusion_source_faces(
                    current,
                    local_selected,
                )
            elif len(all_sources.face_ids) == 1:
                sources = all_sources
            else:
                self.status_panel.set_state(
                "多轮廓草图：请选择二维面",
                    5000,
                )
                return
        except ExtrusionSourceResolutionError as error:
            self.status_panel.set_state(str(error), 5000)
            return
        base_session_revision = self.document.session_revision
        dialog = SweepGeometryDialog(
            current,
            self,
            source_face_ids=sources.face_ids,
        )
        while self._exec_dialog(dialog):
            recipes = tuple(
                RevolvedGeometry(
                    current,
                    str(dialog.axis_combo.currentData()),
                    dialog.angle_spin.value(),
                    (face_id,),
                )
                for face_id in sources.face_ids
            )
            try:
                for recipe in recipes:
                    self._preflight_revolved_geometry(recipe)
            except Exception as error:
                logging.exception("sweep OCC preflight failed")
                self._show_error(
                    "扫掠几何",
                    "临时 OCC 编译失败，当前几何和选择保持不变："
                    f"\n{error}",
                )
                continue
            if len(recipes) == 1:
                self._set_native_geometry(
                    recipes[0],
                    "扫掠实体",
                    base_session_revision=base_session_revision,
                )
            else:
                if active_id is None:
                    self._show_error("扫掠几何", "没有当前部件")
                    return
                if not self._confirm_result_invalidation():
                    return
                try:
                    delta = self.session.replace_part_with_revolved_siblings(
                        active_id,
                        recipes,
                        expected_part_revision=(
                            self.document.part_revision(active_id)
                        ),
                        expected_session_revision=base_session_revision,
                    )
                    self._apply_session_delta(delta)
                except (
                    RuntimeError,
                    TypeError,
                    ValueError,
                    KeyError,
                ) as error:
                    self._show_error("扫掠几何", str(error))
                    return
            return

    @staticmethod
    def _preflight_revolved_geometry(recipe: RevolvedGeometry) -> None:
        with geometry_runtime.model(
            f"{recipe.name}-sweep-preflight",
            dimension=3,
        ) as cad:
            compile_recipe(cad, recipe)

    def fuse_geometry(self) -> None:
        self._boolean_geometry("fuse", "合并后的")

    def cut_geometry(self) -> None:
        self._boolean_geometry("cut", "切除后的")

    def _boolean_geometry(self, operation: str, label: str) -> None:
        current = self.document.geometry_recipe
        if not isinstance(current, NATIVE_GEOMETRY_TYPES):
            return
        if geometry_dimension(current) == 2:
            self._begin_planar_boolean(operation)
            return
        if geometry_dimension(current) == 3:
            self._begin_solid_face_boolean(operation)

    def _begin_solid_face_boolean(self, operation: str) -> None:
        current = self.document.geometry_recipe
        if (
            not isinstance(current, NATIVE_GEOMETRY_TYPES)
            or geometry_dimension(current) != 3
        ):
            return
        try:
            requested_operation = FaceSketchBooleanOperation(operation)
        except ValueError:
            self.status_panel.set_state("三维布尔操作必须是合并或切除", 5000)
            return
        if (
            self._wire_editor_controller is not None
            or self._sketch_editor_controller is not None
            or self._planar_boolean_controller is not None
            or self._body_boolean_controller is not None
            or self._face_sketch_controller is not None
            or self._solid_face_boolean_operation is not None
        ):
            self.status_panel.set_state("请先完成或取消当前几何编辑", 5000)
            return
        original_selection = set(self._selected_geometry_refs)
        self._solid_face_boolean_operation = requested_operation
        self._solid_face_boolean_original_selection = original_selection
        self._set_geometry_selection_mode("face")
        face_bar = self.viewport_panel.planar_boolean_face_bar
        face_bar.begin(operation)
        selected_face_is_valid = self._face_sketch_selection_is_valid()
        face_bar.set_selection_ready(selected_face_is_valid)
        if selected_face_is_valid:
            self.viewport.highlight_geometry_entities(
                self._canonical_geometry_selection()
            )
            self.status_panel.set_state(
            "目标面已选，请确认",
                0,
            )
        else:
            self.status_panel.set_state("请选择目标面", 0)
        self._update_action_states()

    def _begin_planar_boolean(self, operation: str) -> None:
        current = self.document.geometry_recipe
        if (
            not isinstance(current, NATIVE_GEOMETRY_TYPES)
            or geometry_dimension(current) != 2
        ):
            return
        if (
            self._wire_editor_controller is not None
            or self._sketch_editor_controller is not None
            or self._body_boolean_controller is not None
        ):
            self.status_panel.set_state(
                "请先完成或取消当前几何编辑",
                5000,
            )
            return
        if self._planar_boolean_controller is not None:
            self.cancel_planar_boolean()
        selected = self._canonical_geometry_selection()
        active_part_id = self.document.active_part_id
        initial_target = None
        if (
            active_part_id is not None
            and len(selected) == 1
            and selected[0].kind == "face"
        ):
            owner = part_id_from_logical_id(selected[0].logical_id)
            if owner is None:
                initial_target = selected[0].logical_id
            elif owner == active_part_id:
                initial_target = strip_part_logical_id(
                    active_part_id,
                    selected[0].logical_id,
                )
        try:
            controller = PlanarBooleanController(
                current,
                self.document.session_revision,
                operation,
                target_face_id=initial_target,
            )
        except (TypeError, ValueError) as error:
            self.status_panel.set_state(str(error), 5000)
            return
        self._planar_boolean_controller = controller
        self._planar_boolean_preview_result = None
        self._planar_boolean_preview_generation += 1
        self._planar_boolean_original_selection = set(
            self._selected_geometry_refs
        )
        self._set_geometry_selection_mode("face")
        controller.request_target_selection()
        face_bar = self.viewport_panel.planar_boolean_face_bar
        face_bar.begin(operation)
        if initial_target is not None:
            self.viewport.highlight_geometry_entities(selected)
            face_bar.set_selection_ready(True)
        self.status_panel.set_state("请选择目标面", 0)
        self._update_action_states()

    def _request_planar_boolean_target(self) -> None:
        controller = self._planar_boolean_controller
        if controller is None:
            return
        controller.request_target_selection()
        self._planar_boolean_preview_result = None
        self._planar_boolean_preview_generation += 1
        self.planar_boolean_panel.set_preview_valid(False)
        self._set_geometry_selection_mode("face")
        face_bar = self.viewport_panel.planar_boolean_face_bar
        face_bar.begin(controller.operation)
        face_bar.set_selection_ready(controller.target_face_id is not None)
        self.status_panel.set_state("请选择目标面", 0)

    def _clear_planar_boolean_target(self) -> None:
        controller = self._planar_boolean_controller
        if controller is None:
            return
        controller.clear_target()
        self._planar_boolean_preview_result = None
        self._planar_boolean_preview_generation += 1
        self.planar_boolean_panel.set_preview_valid(False)
        self._selected_geometry_refs.clear()
        self._rebuild_full_projection()
        self.viewport.clear_selection()
        self._set_geometry_selection_mode("face")
        self.viewport_panel.planar_boolean_face_bar.set_selection_ready(False)
        self.planar_boolean_panel.refresh()
        self.planar_boolean_panel.show_status("已取消目标面选择")

    def _assign_planar_boolean_reference(
        self,
        reference: LogicalEntityRef,
    ) -> bool:
        controller = self._planar_boolean_controller
        if controller is None or not controller.selecting_target:
            return False
        try:
            active_part_id = self.document.active_part_id
            local_reference = (
                LogicalEntityRef(
                    strip_part_logical_id(
                        active_part_id,
                        reference.logical_id,
                    )
                )
                if (
                    active_part_id is not None
                    and part_id_from_logical_id(reference.logical_id)
                    is not None
                )
                else reference
            )
            controller.assign_reference(local_reference)
        except (KeyError, TypeError, ValueError) as error:
            self.planar_boolean_panel.show_status(str(error))
            return True
        self._planar_boolean_preview_result = None
        self._planar_boolean_preview_generation += 1
        self.planar_boolean_panel.set_preview_valid(False)
        self._selected_geometry_refs = {reference}
        self.viewport.highlight_geometry_entities((reference,))
        controller.request_target_selection()
        self.viewport_panel.planar_boolean_face_bar.set_selection_ready(True)
        self.status_panel.set_state("目标面已选择，请点击“确定”继续", 0)
        return True

    def _assign_solid_face_boolean_reference(
        self,
        reference: LogicalEntityRef,
    ) -> bool:
        if self._solid_face_boolean_operation is None:
            return False
        face_bar = self.viewport_panel.planar_boolean_face_bar
        if reference.kind != "face":
            face_bar.set_selection_ready(False)
            self.status_panel.set_state("请选择一个实体平面面", 3000)
            return True
        owner = part_id_from_logical_id(reference.logical_id)
        if (
            owner is not None
            and owner != self.document.active_part_id
            and any(part.id == owner for part in self.document.parts)
        ):
            try:
                self._apply_session_delta(
                    self.session.set_active_native_part(
                        owner,
                        expected_session_revision=(
                            self.document.session_revision
                        ),
                    )
                )
            except (RuntimeError, TypeError, ValueError, KeyError) as error:
                face_bar.set_selection_ready(False)
                self.status_panel.set_state(str(error), 5000)
                return True
        self._selected_geometry_refs = {reference}
        self.viewport.highlight_geometry_entities((reference,))
        valid = self._face_sketch_selection_is_valid()
        face_bar.set_selection_ready(valid)
        self.status_panel.set_state(
            "目标面已选，请确认"
            if valid
            else "无效平面面，请重选",
            0 if valid else 4000,
        )
        self._update_action_states()
        return True

    def _confirm_boolean_face_prompt(self) -> None:
        if self._planar_boolean_controller is not None:
            self._confirm_planar_boolean_target()
            return
        if self._solid_face_boolean_operation is not None:
            self._confirm_solid_face_boolean_target()

    def _cancel_boolean_face_prompt(self) -> None:
        if self._planar_boolean_controller is not None:
            self.cancel_planar_boolean()
            return
        if self._solid_face_boolean_operation is not None:
            self._cancel_solid_face_boolean()

    def _confirm_planar_boolean_target(self) -> None:
        controller = self._planar_boolean_controller
        if controller is None:
            return
        try:
            controller.confirm_target_selection()
        except ValueError as error:
            self.viewport_panel.planar_boolean_face_bar.set_selection_ready(
                False
            )
            self.status_panel.set_state(str(error), 3000)
            return
        self.viewport_panel.planar_boolean_face_bar.finish()
        self.status_panel.set_state("正在进入工具草图", 0)
        self._edit_planar_boolean_tool()

    def _confirm_solid_face_boolean_target(self) -> None:
        operation = self._solid_face_boolean_operation
        if operation is None:
            return
        if not self._face_sketch_selection_is_valid():
            self.viewport_panel.planar_boolean_face_bar.set_selection_ready(
                False
            )
            self.status_panel.set_state("请先选择一个有效的实体平面面", 3000)
            return
        self.viewport_panel.planar_boolean_face_bar.finish()
        self.status_panel.set_state(
            "正在解析草图工作面",
            0,
        )
        self.start_face_sketch_boolean()

    def _cancel_solid_face_boolean(self) -> None:
        if self._solid_face_boolean_operation is None:
            return
        original_selection = set(self._solid_face_boolean_original_selection)
        self.viewport_panel.planar_boolean_face_bar.finish()
        self._solid_face_boolean_operation = None
        self._solid_face_boolean_original_selection = set()
        self._selected_geometry_refs = original_selection
        self._set_geometry_selection_mode(
            next(iter(original_selection)).kind
            if original_selection
            else "body"
        )
        if original_selection:
            self.viewport.highlight_geometry_entities(
                tuple(sorted(original_selection, key=logical_ref_sort_key))
            )
        else:
            self.viewport.clear_selection()
        self.status_panel.set_state("已取消布尔操作", 3000)
        self._update_action_states()

    def _edit_planar_boolean_tool(self) -> None:
        controller = self._planar_boolean_controller
        if controller is None:
            return
        if self._sketch_editor_controller is not None:
            return
        if controller.target_face_id is None or controller.selecting_target:
            if controller.target_face_id is None:
                controller.request_target_selection()
            face_bar = self.viewport_panel.planar_boolean_face_bar
            face_bar.begin(controller.operation)
            face_bar.set_selection_ready(
                controller.target_face_id is not None
            )
            message = (
                "目标面已选择，请点击“确定”继续"
                if controller.target_face_id is not None
                else "请选择目标面"
            )
            self.status_panel.set_state(message, 0)
            return
        if self.document.session_revision != controller.base_session_revision:
            self.status_panel.set_state(
                "项目已变化，请重开二维布尔",
                5000,
            )
            return
        root = controller.tool_geometry
        if root is not None:
            restored = SketchDraftController.snapshot_from_geometry(root)
            draft = SketchDraftController(
                snapshot=replace(
                    restored,
                    external_references=controller.external_references,
                    external_coincidences=controller.external_coincidences,
                    unresolved_reference_ids=controller.unresolved_reference_ids,
                )
            )
        else:
            draft = SketchDraftController(name="布尔工具草图")
        self._sketch_editor_controller = draft
        self._sketch_editor_original_recipe = None
        self._sketch_editor_base_revision = controller.base_session_revision
        self._sketch_editor_part_name = None
        self._sketch_editor_is_planar_boolean_tool = True
        self.sketch_editor_panel.set_controller(
            draft,
            base_snapshot=draft.snapshot(),
        )
        cached = self._geometry_preview_cache
        active_part_id = self.document.active_part_id
        cache_key = self._native_part_preview_cache_key(self.document)
        source_preview = (
            localize_part_geometry_preview(active_part_id, cached[2])
            if (
                cached is not None
                and active_part_id is not None
                and cached[0] == self.document.session_id
                and cached[1] == cache_key
            )
            else build_geometry_preview(controller.geometry)
        )
        target_faces = tuple(
            face
            for face, logical_id in zip(
                source_preview.faces,
                source_preview.face_logical_ids,
                strict=True,
            )
            if logical_id == controller.target_face_id
        )
        if not target_faces:
            target_faces = source_preview.faces
        if controller.target_face_id is not None:
            self.sketch_editor_panel.set_reference_points(
                planar_reference_points(
                    source_preview,
                    controller.target_face_id,
                    plane=draft.plane,
                )
            )
        self.sketch_editor_panel.begin(
            self.viewport,
            purpose="planar_boolean_tool",
        )
        self.main_splitter.setSizes([260, 720, 0, 400, 0, 0])
        self.viewport.set_view("front")
        if target_faces:
            self.viewport.show_sketch_reference_preview(
                GeometryPreview(
                    source_preview.points,
                    target_faces,
                    (),
                    (controller.target_face_id,) * len(target_faces),
                    (),
                    (),
                    topological_dimension=2,
                )
            )
        self._schedule_viewport_fit()
        self.status_panel.set_state(
            "在目标 XY 平面绘制闭合轮廓",
            0,
        )
        self._update_action_states()

    def _delete_planar_boolean_tool(self) -> None:
        controller = self._planar_boolean_controller
        if controller is None:
            return
        controller.clear_tool()
        self._planar_boolean_preview_result = None
        self._planar_boolean_preview_generation += 1
        self.planar_boolean_panel.set_preview_valid(False)
        self._rebuild_full_projection()
        self._set_geometry_selection_mode("face")
        active_part_id = self.document.active_part_id
        if (
            active_part_id is not None
            and controller.target_face_id is not None
        ):
            target = LogicalEntityRef(
                namespace_part_logical_id(
                    active_part_id,
                    controller.target_face_id,
                )
            )
            self._selected_geometry_refs = {target}
            self.viewport.highlight_geometry_entities((target,))
        self.planar_boolean_panel.refresh()
        self.planar_boolean_panel.show_status("工具轮廓已删除")

    def _planar_boolean_operation_changed(self, operation: str) -> None:
        controller = self._planar_boolean_controller
        if controller is None:
            return
        controller.set_operation(operation)
        self._planar_boolean_preview_result = None
        self._planar_boolean_preview_generation += 1
        self.planar_boolean_panel.set_preview_valid(False)
        if controller.ready:
            self._refresh_planar_boolean_preview()

    def _refresh_planar_boolean_preview(
        self,
        *,
        auto_commit: bool = False,
    ) -> None:
        controller = self._planar_boolean_controller
        if controller is None or not controller.ready:
            return
        if self.document.session_revision != controller.base_session_revision:
            self.status_panel.set_state(
                "项目已变化，请重开二维布尔",
                5000,
            )
            return
        if self.busy:
            self.status_panel.set_state(
                "后台任务运行中，请稍后",
                5000,
            )
            if auto_commit:
                self._defer_ui(self._edit_planar_boolean_tool)
            return
        self._planar_boolean_preview_generation += 1
        generation = self._planar_boolean_preview_generation
        base_revision = controller.base_session_revision
        operation = controller.operation
        target_face_id = controller.target_face_id
        tool_geometry = controller.tool_geometry
        tool_face_ids = controller.tool_face_ids
        source_geometry = controller.geometry
        self._planar_boolean_preview_result = None
        self.planar_boolean_panel.set_preview_valid(False)
        self.planar_boolean_panel.set_preview_running(True)
        self.planar_boolean_panel.show_status(
            "正在执行临时 OCC 平面布尔运算并验证拓扑…"
        )

        def workload(context: TaskContext) -> StrictPlanarBooleanResult:
            context.report("正在执行平面布尔运算并验证精确拓扑…")
            context.checkpoint()
            with geometry_runtime.model(
                f"{getattr(source_geometry, 'name', 'planar')}"
                f"-{operation}-preview",
                dimension=2,
            ) as cad:
                result = prepare_planar_boolean(
                    cad,
                    source_geometry,
                    target_face_id,
                    tool_geometry,
                    tool_face_ids,
                    operation,
                )
            context.checkpoint()
            return result

        def apply_result(payload: object) -> TaskApplyOutcome:
            current = self._planar_boolean_controller
            if (
                type(payload) is not StrictPlanarBooleanResult
                or current is not controller
                or generation != self._planar_boolean_preview_generation
                or self.document.session_revision != base_revision
                or current.operation != operation
                or current.target_face_id != target_face_id
                or current.tool_geometry != tool_geometry
                or current.tool_face_ids != tool_face_ids
            ):
                if current is controller:
                    self.planar_boolean_panel.set_preview_running(False)
                    self.planar_boolean_panel.set_preview_valid(False)
                return TaskApplyOutcome.stale(
                    "二维布尔预览已过期，未应用"
                )
            return TaskApplyOutcome.accepted(payload)

        def on_success(payload: object) -> None:
            if type(payload) is not StrictPlanarBooleanResult:
                raise TypeError(
                    "strict planar Boolean task returned invalid data"
                )
            preview = build_strict_planar_boolean_preview(
                payload.geometry,
                payload.preview,
            )
            active_id = self.document.active_part_id
            if active_id is not None:
                preview = namespace_part_geometry_preview(
                    active_id,
                    preview,
                )
            self._planar_boolean_preview_result = payload
            self.planar_boolean_panel.set_preview_running(False)
            self.planar_boolean_panel.set_preview_valid(True)
            if auto_commit:
                self.status_panel.set_state(
                    "验证通过，正在提交二维布尔",
                    0,
                )
                self.finish_planar_boolean(preview=preview)
            else:
                self.viewport.show_geometry_preview(preview)
                self.planar_boolean_panel.show_status(
                    "精确预览和拓扑验证已通过，可以完成"
                )

        def on_failure(message: str) -> None:
            if (
                self._planar_boolean_controller is controller
                and generation == self._planar_boolean_preview_generation
            ):
                self.planar_boolean_panel.set_preview_running(False)
                self.planar_boolean_panel.set_preview_valid(False)
                self.planar_boolean_panel.show_status(
                    "OCC 平面布尔运算或拓扑验证失败；"
                    f"已提交几何未变化：\n{message}"
                )
                self._planar_boolean_preview_result = None
                if auto_commit:
                    self._defer_ui(self._edit_planar_boolean_tool)

        def on_cancelled() -> None:
            if self._planar_boolean_controller is not controller:
                return
            self.planar_boolean_panel.set_preview_running(False)
            if auto_commit:
                self._defer_ui(self._edit_planar_boolean_tool)

        started = self._start_task(
            workload,
            on_success,
            "二维布尔预览",
            on_failure,
            task_name="二维布尔预览",
            on_cancelled=on_cancelled,
            apply_result=apply_result,
        )
        if not started:
            self.planar_boolean_panel.set_preview_running(False)
            if auto_commit:
                self._defer_ui(self._edit_planar_boolean_tool)

    def finish_planar_boolean(
        self,
        *,
        preview: GeometryPreview | None = None,
    ) -> None:
        controller = self._planar_boolean_controller
        result = self._planar_boolean_preview_result
        if controller is None or not controller.ready:
            return
        if result is None:
            self._refresh_planar_boolean_preview()
            return
        active_id = self.document.active_part_id
        if active_id is None:
            return
        if preview is None:
            preview = namespace_part_geometry_preview(
                active_id,
                build_strict_planar_boolean_preview(
                    result.geometry,
                    result.preview,
                ),
            )
        if not self._set_native_geometry(
            result.geometry,
            "二维布尔后的",
            base_session_revision=controller.base_session_revision,
            preserve_editor=True,
            geometry_preview=preview,
        ):
            self.status_panel.set_state(
                "提交未完成，已返回工具草图",
                5000,
            )
            self._edit_planar_boolean_tool()
            return
        face_references = {
            LogicalEntityRef(
                namespace_part_logical_id(
                    active_id,
                    entity.logical_id,
                )
            )
            for entity in result.proof.result_entities
            if entity.kind == "face"
        }
        self._exit_planar_boolean()
        self._selected_geometry_refs = face_references
        if face_references:
            self.viewport.highlight_geometry_entities(
                tuple(
                    sorted(
                        face_references,
                        key=logical_ref_sort_key,
                    )
                )
            )
        self._update_action_states()

    def cancel_planar_boolean(self) -> None:
        if self._planar_boolean_controller is None:
            return
        original_selection = set(self._planar_boolean_original_selection)
        self._planar_boolean_preview_generation += 1
        self._exit_planar_boolean()

        def restore_projection_and_selection() -> None:
            self._rebuild_full_projection()
            self._selected_geometry_refs = original_selection
            if original_selection:
                self.viewport.highlight_geometry_entities(
                    tuple(
                        sorted(
                            original_selection,
                            key=logical_ref_sort_key,
                        )
                    )
                )
            self.status_panel.set_state(
                "二维布尔已取消",
                4000,
            )

        if self.task_controller.current_task_name == "二维布尔预览":
            if self.cancel_current_task(
                after_cleanup=restore_projection_and_selection,
            ):
                return
        restore_projection_and_selection()

    def _exit_planar_boolean(self) -> None:
        self.viewport_panel.planar_boolean_face_bar.finish()
        self.planar_boolean_panel.end()
        self._planar_boolean_controller = None
        self._planar_boolean_preview_result = None
        self._planar_boolean_original_selection = set()
        self.main_splitter.setSizes([260, 1020, 0, 0, 0, 0])

    def _begin_body_boolean(self, operation: str) -> None:
        candidates = tuple(
            part
            for part in self.document.parts
            if not part.suppressed and part.dimension == 3
        )
        if len(candidates) < 2:
            self.status_panel.set_state(
                "实体布尔至少需要两个三维部件",
                5000,
            )
            return
        if (
            self._wire_editor_controller is not None
            or self._sketch_editor_controller is not None
            or self._planar_boolean_controller is not None
        ):
            self.status_panel.set_state("请先完成或取消当前几何编辑", 5000)
            return
        if self._body_boolean_controller is not None:
            self.cancel_body_boolean()
        controller = PartBooleanController(
            tuple(self.document.parts),
            self.document.session_revision,
            operation,
            target_part_id=(
                self.document.active_part_id
                if self.document.active_part_id
                in {part.id for part in candidates}
                else None
            ),
        )
        self._body_boolean_controller = controller
        self._body_boolean_preview_result = None
        self._body_boolean_preview_generation += 1
        self.boolean_feature_panel.begin(controller)
        self._set_geometry_selection_mode("body")
        self.main_splitter.setSizes([260, 720, 0, 0, 400, 0])
        self.status_panel.set_state(
            "实体布尔：选择目标和工具部件",
            0,
        )
        self._update_action_states()

    def _request_body_boolean_selection(self, slot: str) -> None:
        controller = self._body_boolean_controller
        if controller is None:
            return
        try:
            controller.request_selection(slot)
        except ValueError as error:
            self.boolean_feature_panel.show_status(str(error))
            return
        if self._body_boolean_preview_result is not None:
            self._body_boolean_preview_result = None
            self.boolean_feature_panel.set_preview_valid(False)
            self._rebuild_full_projection()
        self._set_geometry_selection_mode("body")
        role = "目标部件" if slot == "target" else "工具部件"
        self.boolean_feature_panel.show_status(
            f"请在视口或模型树中选择{role}"
        )
        self.status_panel.set_state(f"正在选择布尔操作{role}", 0)

    def _assign_body_boolean_reference(
        self,
        reference: LogicalEntityRef,
    ) -> bool:
        controller = self._body_boolean_controller
        if controller is None or controller.pending_slot is None:
            return False
        if (
            reference.kind == "body"
            and part_id_from_logical_id(reference.logical_id) is not None
        ):
            reference = LogicalEntityRef(
                f"part:{part_id_from_logical_id(reference.logical_id)}"
            )
        try:
            slot = controller.assign_reference(reference)
        except (KeyError, TypeError, ValueError) as error:
            self.boolean_feature_panel.show_status(str(error))
            return True
        self.boolean_feature_panel.refresh()
        self.viewport.highlight_body_boolean_operands(
            (
                None
                if controller.target_part_id is None
                else LogicalEntityRef(
                    f"body:{controller.target_part_id}/domain"
                )
            ),
            (
                None
                if controller.tool_part_id is None
                else LogicalEntityRef(
                    f"body:{controller.tool_part_id}/domain"
                )
            ),
        )
        role = "目标部件" if slot == "target" else "工具部件"
        self.boolean_feature_panel.show_status(f"{role}已设置")
        if controller.ready:
            self._refresh_body_boolean_preview()
        return True

    def _body_boolean_operation_changed(self, operation: str) -> None:
        controller = self._body_boolean_controller
        if controller is None:
            return
        controller.set_operation(operation)
        self._body_boolean_preview_result = None
        self._body_boolean_preview_generation += 1
        self.boolean_feature_panel.set_preview_valid(False)
        if controller.ready:
            self._refresh_body_boolean_preview()

    def _refresh_body_boolean_preview(self) -> None:
        controller = self._body_boolean_controller
        if controller is None or not controller.ready:
            return
        if self.document.session_revision != controller.base_session_revision:
            self.boolean_feature_panel.finish_button.setEnabled(False)
            self.boolean_feature_panel.show_status(
                "项目已变化；请取消后重新打开实体布尔"
            )
            return
        if self.busy:
            self.boolean_feature_panel.show_status(
                "当前有后台任务正在运行，请稍后重试"
            )
            return
        self._body_boolean_preview_generation += 1
        generation = self._body_boolean_preview_generation
        target_part_id = controller.target_part_id
        tool_part_id = controller.tool_part_id
        operation = controller.operation
        base_revision = controller.base_session_revision
        result_name = self.boolean_feature_panel.result_name()
        result_part_id = self.session.next_native_part_id
        feature_id = self.session.next_part_boolean_feature_id
        target_part = controller.part(str(target_part_id))
        tool_part = controller.part(str(tool_part_id))
        self._body_boolean_preview_result = None
        self.boolean_feature_panel.set_preview_valid(False)
        self.boolean_feature_panel.set_preview_running(True)
        self.boolean_feature_panel.show_status(
            "正在执行临时 OCC 布尔运算并验证谱系…"
        )

        def workload(context: TaskContext) -> StrictPartBooleanResult:
            context.report("正在执行临时 OCC 布尔运算并验证谱系…")
            context.checkpoint()
            with geometry_runtime.model(
                f"{result_name}-{operation}-preview",
                dimension=3,
            ) as cad:
                result = prepare_part_boolean(
                    cad,
                    target_part,
                    tool_part,
                    operation,
                    result_part_id=result_part_id,
                    feature_id=feature_id,
                    result_name=result_name,
                )
            context.checkpoint()
            return result

        def apply_result(payload: object) -> TaskApplyOutcome:
            current = self._body_boolean_controller
            if (
                type(payload) is not StrictPartBooleanResult
                or current is not controller
                or generation != self._body_boolean_preview_generation
                or self.document.session_revision != base_revision
                or current.target_part_id != target_part_id
                or current.tool_part_id != tool_part_id
                or current.operation != operation
                or self.boolean_feature_panel.result_name()
                != result_name
                or self.session.next_native_part_id != result_part_id
                or self.session.next_part_boolean_feature_id != feature_id
            ):
                if current is controller:
                    self.boolean_feature_panel.set_preview_running(False)
                    self.boolean_feature_panel.set_preview_valid(False)
                return TaskApplyOutcome.stale(
                    "实体布尔预览已过期，未应用"
                )
            return TaskApplyOutcome.accepted(payload)

        def on_success(payload: object) -> None:
            if type(payload) is not StrictPartBooleanResult:
                raise TypeError("实体布尔任务返回了无效结果")
            preview = build_strict_part_boolean_preview(payload.preview)
            self._body_boolean_preview_result = payload
            self.viewport.clear_body_boolean_highlights(render=False)
            self.viewport.show_geometry_preview(preview)
            self.boolean_feature_panel.set_preview_running(False)
            self.boolean_feature_panel.set_preview_valid(True)
            self.boolean_feature_panel.show_status(
                "精确预览与谱系验证已通过，可以完成"
            )

        def on_failure(message: str) -> None:
            if (
                self._body_boolean_controller is controller
                and generation == self._body_boolean_preview_generation
            ):
                self.boolean_feature_panel.set_preview_running(False)
                self.boolean_feature_panel.set_preview_valid(False)
                self.boolean_feature_panel.show_status(
                    "OCC 布尔运算或谱系验证失败；"
                    f"已提交部件未变化：\n{message}"
                )
                self._body_boolean_preview_result = None

        started = self._start_task(
            workload,
            on_success,
            "实体布尔预览",
            on_failure,
            task_name="实体布尔预览",
            on_cancelled=lambda: (
                self.boolean_feature_panel.set_preview_running(False)
                if self._body_boolean_controller is controller
                else None
            ),
            apply_result=apply_result,
        )
        if not started:
            self.boolean_feature_panel.set_preview_running(False)

    def finish_body_boolean(self) -> None:
        controller = self._body_boolean_controller
        if controller is None or not controller.ready:
            return
        if self._body_boolean_preview_result is None:
            self._refresh_body_boolean_preview()
        result = self._body_boolean_preview_result
        if type(result) is not StrictPartBooleanResult:
            return
        if result.recipe.name != self.boolean_feature_panel.result_name():
            self._body_boolean_preview_result = None
            self.boolean_feature_panel.set_preview_valid(False)
            self._refresh_body_boolean_preview()
            return
        preview = build_strict_part_boolean_preview(result.preview)
        target_id = str(controller.target_part_id)
        tool_id = str(controller.tool_part_id)
        if not self._confirm_result_invalidation(preserve_editor=True):
            return
        try:
            delta = self.session.apply_part_boolean(
                target_id,
                tool_id,
                controller.operation,
                self.boolean_feature_panel.result_name(),
                result=result,
                expected_target_revision=self.document.part_revision(
                    target_id
                ),
                expected_tool_revision=self.document.part_revision(tool_id),
                expected_session_revision=controller.base_session_revision,
            )
            self._apply_session_delta(
                delta,
                geometry_preview=preview,
            )
        except (RuntimeError, TypeError, ValueError, KeyError) as error:
            self.boolean_feature_panel.show_status(
                f"提交失败；已提交部件保持不变：{error}"
            )
            return
        self._exit_body_boolean()
        result_id = result.context.result_part_id
        target_reference = LogicalEntityRef(f"body:{result_id}/domain")
        self._selected_geometry_refs = {target_reference}
        self.viewport.highlight_geometry_entities((target_reference,))
        self._update_action_states()

    def cancel_body_boolean(self) -> None:
        if self._body_boolean_controller is None:
            return
        self._body_boolean_preview_generation += 1
        self._exit_body_boolean()
        if self.task_controller.current_task_name == "实体布尔预览":
            if self.cancel_current_task(
                after_cleanup=self._rebuild_full_projection,
            ):
                return
        self._rebuild_full_projection()
        self.status_panel.set_state(
            "实体布尔已取消",
            4000,
        )

    def _exit_body_boolean(self) -> None:
        self.viewport.clear_body_boolean_highlights(render=False)
        self.boolean_feature_panel.end()
        self._body_boolean_controller = None
        self._body_boolean_preview_result = None
        self.main_splitter.setSizes([260, 1020, 0, 0, 0, 0])

    @classmethod
    def _recipe_contains_strict_boolean(cls, recipe: object) -> bool:
        if isinstance(recipe, MultiBodyGeometry):
            return any(
                cls._recipe_contains_strict_boolean(body.recipe)
                for body in recipe.bodies
            )
        if isinstance(recipe, BooleanGeometry):
            return (
                recipe.body_context is not None
                and recipe.body_context.proven
            ) or (
                recipe.planar_context is not None
                and recipe.planar_context.proven
            ) or (
                recipe.part_context is not None
                and recipe.part_context.proven
            ) or cls._recipe_contains_strict_boolean(
                recipe.object_geometry
            ) or cls._recipe_contains_strict_boolean(recipe.tool_geometry)
        if isinstance(
            recipe,
            (MovedGeometry, RotatedGeometry, ExtrudedGeometry, RevolvedGeometry, PathSweptGeometry),
        ):
            return cls._recipe_contains_strict_boolean(recipe.base)
        return False

    @classmethod
    def _recipe_contains_strict_part_boolean(cls, recipe: object) -> bool:
        if isinstance(recipe, MultiBodyGeometry):
            return any(
                cls._recipe_contains_strict_part_boolean(body.recipe)
                for body in recipe.bodies
            )
        if isinstance(recipe, BooleanGeometry):
            return (
                recipe.part_context is not None
                and recipe.part_context.proven
            ) or cls._recipe_contains_strict_part_boolean(
                recipe.object_geometry
            ) or cls._recipe_contains_strict_part_boolean(
                recipe.tool_geometry
            )
        if isinstance(
            recipe,
            (MovedGeometry, RotatedGeometry, ExtrudedGeometry, RevolvedGeometry, PathSweptGeometry),
        ):
            return cls._recipe_contains_strict_part_boolean(recipe.base)
        return False

    @classmethod
    def _recipe_contains_strict_planar_boolean(cls, recipe: object) -> bool:
        if isinstance(recipe, MultiBodyGeometry):
            return any(
                cls._recipe_contains_strict_planar_boolean(body.recipe)
                for body in recipe.bodies
            )
        if isinstance(recipe, BooleanGeometry):
            return (
                recipe.planar_context is not None
                and recipe.planar_context.proven
            ) or cls._recipe_contains_strict_planar_boolean(
                recipe.object_geometry
            ) or cls._recipe_contains_strict_planar_boolean(
                recipe.tool_geometry
            )
        if isinstance(
            recipe,
            (MovedGeometry, RotatedGeometry, ExtrudedGeometry, RevolvedGeometry, PathSweptGeometry),
        ):
            return cls._recipe_contains_strict_planar_boolean(recipe.base)
        return False

    def _schedule_exact_boolean_preview(
        self,
        snapshot: object,
        recipe: object,
    ) -> None:
        part_id = getattr(snapshot, "active_part_id", None)
        if type(part_id) is not str:
            return
        key = (
            str(getattr(snapshot, "session_id")),
            part_id,
            int(getattr(snapshot, "part_revision")(part_id)),
            int(getattr(snapshot, "project_revision")),
        )
        if self._pending_exact_boolean_preview_key == key:
            return
        self._pending_exact_boolean_preview_key = key

        def launch() -> None:
            if (
                self.document.session_id != key[0]
                or self.document.active_part_id != key[1]
                or self.document.part_revision(key[1]) != key[2]
                or self.document.project_revision != key[3]
                or self.document.geometry_recipe != recipe
            ):
                if self._pending_exact_boolean_preview_key == key:
                    self._pending_exact_boolean_preview_key = None
                return

            if not isinstance(recipe, MultiBodyGeometry):
                if self._recipe_contains_strict_part_boolean(recipe):
                    self._launch_exact_part_boolean_preview(key, recipe)
                else:
                    self._launch_exact_planar_boolean_preview(key, recipe)
                return

            strict_bodies = tuple(
                body
                for body in recipe.bodies
                if self._recipe_contains_strict_boolean(body.recipe)
            )

            def workload(
                context: TaskContext,
            ) -> tuple[StrictBodyBooleanPreview, ...]:
                previews: list[StrictBodyBooleanPreview] = []
                for body in strict_bodies:
                    context.report(
                        f"正在重建 {body.name} [{body.id}] 的 OCC 预览…"
                    )
                    with geometry_runtime.model(
                        f"{recipe.name}-{body.id}-persisted-preview",
                        dimension=3,
                    ) as cad:
                        previews.append(
                            prepare_strict_body_recipe_preview(
                                cad,
                                body.id,
                                body.recipe,
                            )
                        )
                    context.checkpoint()
                return tuple(previews)

            def apply_result(payload: object) -> TaskApplyOutcome:
                if (
                    self.document.session_id != key[0]
                    or self.document.active_part_id != key[1]
                    or self.document.part_revision(key[1]) != key[2]
                    or self.document.project_revision != key[3]
                    or self.document.geometry_recipe != recipe
                    or not isinstance(payload, tuple)
                    or any(
                        type(item) is not StrictBodyBooleanPreview
                        for item in payload
                    )
                ):
                    if self._pending_exact_boolean_preview_key == key:
                        self._pending_exact_boolean_preview_key = None
                    return TaskApplyOutcome.stale(
                        "布尔 OCC 预览重建结果已过期"
                    )
                preview = build_strict_body_boolean_previews(
                    recipe,
                    payload,
                )
                return TaskApplyOutcome.accepted(preview)

            def on_success(payload: object) -> None:
                if type(payload) is not GeometryPreview:
                    raise TypeError(
                        "persisted Boolean preview task returned invalid data"
                    )
                self._pending_exact_boolean_preview_key = None
                self._geometry_preview_cache = (
                    key[0],
                    self._native_part_preview_cache_key(self.document),
                    payload,
                )
                if (
                    self.document.session_id == key[0]
                    and self.document.geometry_recipe == recipe
                    and self.document.artifact is None
                ):
                    self.viewport.show_geometry_preview(payload)
                    self.status_panel.set_state(
                        "布尔预览已重建",
                        4000,
                    )

            def cleanup_failure(message: str | None = None) -> None:
                if self._pending_exact_boolean_preview_key == key:
                    self._pending_exact_boolean_preview_key = None
                if message:
                    self.status_panel.set_state(
                        f"布尔 OCC 预览重建失败：{message}",
                        6000,
                    )

            started = self._start_task(
                workload,
                on_success,
                "布尔 OCC 预览",
                cleanup_failure,
                task_name="布尔 OCC 预览重建",
                on_cancelled=cleanup_failure,
                apply_result=apply_result,
            )
            if not started:
                cleanup_failure()

        self._defer_ui(launch)

    def _launch_exact_part_boolean_preview(
        self,
        key: tuple[str, str, int, int],
        recipe: object,
    ) -> None:
        def workload(context: TaskContext) -> StrictBodyBooleanPreview:
            context.report("正在重建部件布尔 OCC 预览…")
            with geometry_runtime.model(
                f"{getattr(recipe, 'name', 'part-boolean')}-persisted-preview",
                dimension=3,
            ) as cad:
                preview = prepare_strict_part_recipe_preview(
                    cad,
                    key[1],
                    recipe,
                )
            context.checkpoint()
            return preview

        def apply_result(payload: object) -> TaskApplyOutcome:
            if (
                self.document.session_id != key[0]
                or self.document.active_part_id != key[1]
                or self.document.part_revision(key[1]) != key[2]
                or self.document.project_revision != key[3]
                or self.document.geometry_recipe != recipe
                or type(payload) is not StrictBodyBooleanPreview
            ):
                if self._pending_exact_boolean_preview_key == key:
                    self._pending_exact_boolean_preview_key = None
                return TaskApplyOutcome.stale(
                    "部件布尔 OCC 预览重建结果已过期"
                )
            preview = build_strict_part_boolean_preview(payload)
            return TaskApplyOutcome.accepted(preview)

        def on_success(payload: object) -> None:
            if type(payload) is not GeometryPreview:
                raise TypeError(
                    "persisted Part Boolean preview returned invalid data"
                )
            self._pending_exact_boolean_preview_key = None
            self._geometry_preview_cache = (
                key[0],
                self._native_part_preview_cache_key(self.document),
                payload,
            )
            if (
                self.document.session_id == key[0]
                and self.document.active_part_id == key[1]
                and self.document.part_revision(key[1]) == key[2]
                and self.document.project_revision == key[3]
                and self.document.geometry_recipe == recipe
                and self.document.artifact is None
            ):
                self.viewport.show_geometry_preview(payload)
                self.status_panel.set_state(
                        "部件布尔预览已重建",
                    4000,
                )

        def cleanup_failure(message: str | None = None) -> None:
            if self._pending_exact_boolean_preview_key == key:
                self._pending_exact_boolean_preview_key = None
            if message:
                self.status_panel.set_state(
                    f"部件布尔 OCC 预览重建失败：{message}",
                    6000,
                )

        started = self._start_task(
            workload,
            on_success,
            "部件布尔 OCC 预览",
            cleanup_failure,
            task_name="部件布尔 OCC 预览重建",
            on_cancelled=cleanup_failure,
            apply_result=apply_result,
        )
        if not started:
            cleanup_failure()

    def _launch_exact_planar_boolean_preview(
        self,
        key: tuple[str, str, int, int],
        recipe: object,
    ) -> None:
        def workload(context: TaskContext) -> StrictPlanarBooleanPreview:
            context.report("正在重建二维布尔 OCC 预览…")
            with geometry_runtime.model(
                f"{getattr(recipe, 'name', 'planar')}-persisted-preview",
                dimension=geometry_dimension(recipe),
            ) as cad:
                preview = prepare_strict_planar_recipe_preview(cad, recipe)
            context.checkpoint()
            return preview

        def apply_result(payload: object) -> TaskApplyOutcome:
            if (
                self.document.session_id != key[0]
                or self.document.active_part_id != key[1]
                or self.document.part_revision(key[1]) != key[2]
                or self.document.project_revision != key[3]
                or self.document.geometry_recipe != recipe
                or type(payload) is not StrictPlanarBooleanPreview
            ):
                if self._pending_exact_boolean_preview_key == key:
                    self._pending_exact_boolean_preview_key = None
                return TaskApplyOutcome.stale(
                    "二维布尔 OCC 预览重建结果已过期"
                )
            preview = build_strict_planar_boolean_preview(recipe, payload)
            preview = namespace_part_geometry_preview(key[1], preview)
            return TaskApplyOutcome.accepted(preview)

        def on_success(payload: object) -> None:
            if type(payload) is not GeometryPreview:
                raise TypeError(
                    "persisted planar Boolean preview returned invalid data"
                )
            self._pending_exact_boolean_preview_key = None
            self._geometry_preview_cache = (
                key[0],
                self._native_part_preview_cache_key(self.document),
                payload,
            )
            if (
                self.document.session_id == key[0]
                and self.document.active_part_id == key[1]
                and self.document.part_revision(key[1]) == key[2]
                and self.document.project_revision == key[3]
                and self.document.geometry_recipe == recipe
                and self.document.artifact is None
            ):
                self.viewport.show_geometry_preview(payload)
                self.status_panel.set_state(
                        "二维布尔预览已重建",
                    4000,
                )

        def cleanup_failure(message: str | None = None) -> None:
            if self._pending_exact_boolean_preview_key == key:
                self._pending_exact_boolean_preview_key = None
            if message:
                self.status_panel.set_state(
                    f"二维布尔 OCC 预览重建失败：{message}",
                    6000,
                )

        started = self._start_task(
            workload,
            on_success,
            "二维布尔 OCC 预览",
            cleanup_failure,
            task_name="二维布尔 OCC 预览重建",
            on_cancelled=cleanup_failure,
            apply_result=apply_result,
        )
        if not started:
            cleanup_failure()

    @staticmethod
    def _root_geometry(recipe: object) -> object:
        current = recipe
        while isinstance(
            current,
            (
                MovedGeometry,
                RotatedGeometry,
                ExtrudedGeometry,
                RevolvedGeometry,
                PathSweptGeometry,
            ),
        ):
            current = current.base
        while isinstance(current, BooleanGeometry):
            current = current.object_geometry
            while isinstance(
                current,
                (
                    MovedGeometry,
                    RotatedGeometry,
                    ExtrudedGeometry,
                    RevolvedGeometry,
                    PathSweptGeometry,
                ),
            ):
                current = current.base
        return current

    @staticmethod
    def _native_part_preview_cache_key(snapshot: object) -> tuple[object, ...]:
        """Key previews by stable Part identity, recipe, suppression, and revision."""

        return (
            snapshot.active_part_id,
            tuple(
                (
                    part.id,
                    snapshot.part_revision(part.id),
                    part.geometry_recipe,
                    part.suppressed,
                )
                for part in snapshot.parts
            )
        )

    @classmethod
    def _replace_root_geometry(
        cls,
        recipe: object,
        new_root: object,
    ) -> object:
        if isinstance(
            recipe,
            (MovedGeometry, RotatedGeometry, ExtrudedGeometry, RevolvedGeometry, PathSweptGeometry),
        ):
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
        if isinstance(current, MultiBodyGeometry):
            body_id = self._selected_body_id(current)
            if body_id is None:
                self.status_panel.set_state(
                    "请在模型树或视口中选择一个实体",
                    5000,
                )
                return
            body = current.body(body_id)
            name, accepted = QInputDialog.getText(
                self,
                "实体管理",
                f"重命名 {body.name} [{body.id}]：",
                text=body.name,
            )
            if accepted:
                try:
                    updated = rename_solid_body(current, body.id, name)
                except (TypeError, ValueError) as error:
                    self._show_error("实体管理", str(error))
                    return
                self._set_native_geometry(updated, "重命名后的")
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
                "编辑基础几何"
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
            (
                MovedGeometry,
                RotatedGeometry,
                ExtrudedGeometry,
                RevolvedGeometry,
                PathSweptGeometry,
            ),
        ):
            self._set_native_geometry(current.base, "撤销后的")
        elif dialog.operation == "delete" and isinstance(current, BooleanGeometry):
            self._set_native_geometry(current.object_geometry, "撤销后的")
        elif dialog.operation == "clear":
            if not self._confirm_result_invalidation():
                return
            self._apply_session_delta(self.session.clear_geometry())
            self.status_panel.set_state("当前几何已清空", 5000)

    def undo_geometry_feature(self) -> None:
        current = self.document.geometry_recipe
        active_part_id = self.document.active_part_id
        if (
            active_part_id is not None
            and self.session.can_undo_face_sketch_boolean(active_part_id)
        ):
            if not self._confirm_result_invalidation():
                return
            try:
                self._apply_session_delta(
                    self.session.undo_face_sketch_boolean(
                        active_part_id,
                        expected_part_revision=self.document.part_revision(
                            active_part_id
                        ),
                        expected_session_revision=self.document.session_revision,
                    )
                )
            except (RuntimeError, TypeError, ValueError, KeyError) as error:
                self._show_error("撤销拉伸布尔", str(error))
                return
            self.status_panel.set_state(
                "拉伸布尔已撤销",
                5000,
            )
        elif isinstance(current, MultiBodyGeometry):
            body_id = self._selected_body_id(current)
            if body_id is None:
                self.status_panel.set_state("请先选择一个实体", 5000)
                return
            try:
                updated = undo_solid_body_feature(current, body_id)
            except ValueError as error:
                self.status_panel.set_state(str(error), 5000)
                return
            self._set_native_geometry(updated, "撤销后的")
        elif (
            isinstance(current, (ExtrudedGeometry, RevolvedGeometry))
            and self.document.active_part_id is not None
            and self.session.can_undo_part_extrusion(
                self.document.active_part_id
            )
        ):
            part_id = self.document.active_part_id
            if not self._confirm_result_invalidation():
                return
            try:
                self._apply_session_delta(
                    self.session.undo_part_extrusion(
                        part_id,
                        expected_part_revision=(
                            self.document.part_revision(part_id)
                        ),
                        expected_session_revision=(
                            self.document.session_revision
                        ),
                    )
                )
            except (RuntimeError, TypeError, ValueError, KeyError) as error:
                self._show_error(
                    (
                        "撤销拉伸"
                        if isinstance(current, ExtrudedGeometry)
                        else "撤销扫掠"
                    ),
                    str(error),
                )
        elif isinstance(
            current,
            (MovedGeometry, RotatedGeometry, ExtrudedGeometry, RevolvedGeometry, PathSweptGeometry),
        ):
            self._set_native_geometry(current.base, "撤销后的")
        elif (
            isinstance(current, BooleanGeometry)
            and current.part_context is not None
            and self.document.active_part_id is not None
        ):
            result_id = self.document.active_part_id
            if not self._confirm_result_invalidation():
                return
            try:
                self._apply_session_delta(
                    self.session.undo_part_boolean(
                        result_id,
                        expected_part_revision=(
                            self.document.part_revision(result_id)
                        ),
                        expected_session_revision=(
                            self.document.session_revision
                        ),
                    )
                )
            except (RuntimeError, TypeError, ValueError, KeyError) as error:
                self._show_error("撤销实体布尔", str(error))
        elif isinstance(current, BooleanGeometry):
            self._set_native_geometry(current.object_geometry, "撤销后的")

    def delete_geometry(self) -> None:
        current = self.document.geometry_recipe
        if not isinstance(current, NATIVE_GEOMETRY_TYPES):
            return
        if isinstance(current, MultiBodyGeometry):
            body_id = self._selected_body_id(current)
            if body_id is None:
                self.status_panel.set_state("请先选择要删除的实体", 5000)
                return
            body = current.body(body_id)
            prefix = f"{body_id}/"
            impacted_regions = tuple(
                region.name
                for region in self.document.named_regions.values()
                if any(
                    type(reference) is LogicalEntityRef
                    and (
                        reference.logical_id == f"body:{body_id}"
                        or reference.logical_id.split(":", 1)[1].startswith(
                            prefix
                        )
                    )
                    for reference in region.references
                )
            )
            impacted_controls = tuple(
                control
                for control in getattr(
                    self.document.mesh_settings,
                    "local_controls",
                    (),
                )
                if (
                    control.target.logical_id == f"body:{body_id}"
                    or control.target.logical_id.split(":", 1)[1].startswith(
                        prefix
                    )
                )
            )
            impact = (
                f"\n将移除 {len(impacted_regions)} 个命名区域、"
                f"{len(impacted_controls)} 个局部网格控制。"
                if impacted_regions or impacted_controls
                else "\n当前没有直接引用此实体的命名区域或局部网格控制。"
            )
            answer = QMessageBox.question(
                self,
                "删除实体",
                f"确认删除 {body.name} [{body.id}]？{impact}",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
            updated = delete_solid_body(current, body_id)
            if updated is not None:
                self._set_native_geometry(updated, "删除实体后的")
                return
        active_id = self.document.active_part_id
        if active_id is None:
            return
        active = self.document.part(active_id)
        answer = QMessageBox.question(
            self,
            "删除部件",
            f"确认删除 {active.name} [{active.id}]？"
            "\n仅该部件命名空间内的集合和指派会受影响。",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        if not self._confirm_result_invalidation():
            return
        try:
            delta = self.session.delete_native_part(
                active_id,
                expected_part_revision=self.document.part_revision(
                    active_id
                ),
                expected_session_revision=self.document.session_revision,
            )
            self._apply_session_delta(delta)
        except (RuntimeError, TypeError, ValueError, KeyError) as error:
            self._show_error("删除部件", str(error))
            return
        self._selected_geometry_refs.clear()
        self._selected_mesh_scope_refs.clear()
        self.viewport_panel.set_geometry_context(False)
        self.status_panel.set_state("当前部件已删除", 5000)

    def _set_native_geometry(
        self,
        recipe: object,
        label: str,
        *,
        base_session_revision: int | None = None,
        preserve_editor: bool = False,
        geometry_preview: GeometryPreview | None = None,
    ) -> bool:
        if not isinstance(recipe, NATIVE_GEOMETRY_TYPES):
            raise TypeError(f"不支持的几何定义：{type(recipe).__name__}")
        if geometry_preview is not None and type(
            geometry_preview
        ) is not GeometryPreview:
            raise TypeError("geometry_preview must be a GeometryPreview")
        if not self._confirm_result_invalidation(
            preserve_editor=preserve_editor
        ):
            return False
        if isinstance(recipe, MultiBodyGeometry):
            self._show_error(
                "编辑几何",
                "新项目不能创建多实体几何；请使用独立部件和实体布尔。",
            )
            return False
        self._geometry_preview_cache = None
        if self.document.source_kind != "native":
            new_receipt = self.new_native_project(NewNativeProjectCommand())
            if new_receipt.diagnostic is not None:
                self._show_command_rejection("新建项目", new_receipt)
                return False
        prior_region_count = len(self.document.named_regions)
        expected_session_revision = (
            self.document.session_revision
            if base_session_revision is None
            else base_session_revision
        )
        try:
            active_id = self.document.active_part_id
            if active_id is None:
                delta = self.session.add_native_part(
                    recipe,
                    name=(
                        f"部件-{self.session.next_native_part_id[1:]}"
                    ),
                    expected_session_revision=expected_session_revision,
                )
            else:
                delta = self.session.replace_part_geometry(
                    active_id,
                    recipe,
                    expected_part_revision=self.document.part_revision(
                        active_id
                    ),
                    expected_session_revision=expected_session_revision,
                    authenticate_geometry=geometry_preview is None,
                )
            if not self._apply_session_delta(
                delta,
                geometry_preview=geometry_preview,
            ):
                return False
        except (RuntimeError, TypeError, ValueError, KeyError) as error:
            self._show_error("编辑几何", str(error))
            return False
        self._pending_local_mesh_selection = False
        self._pending_analysis_selection = None
        self._pending_analysis_requested_scope_kind = None
        self._pending_analysis_dialog_state = None
        self._pending_scope_kind = None
        self._pending_analysis_edit = None
        self._selected_geometry_refs.clear()
        self._selected_mesh_scope_refs.clear()
        self._geometry_selection_mode = "body"
        self._selection_context.geometry_filter = "body"
        if self._selection_context.space == "geometry":
            self._set_selection_filter("body", force=True)
        message = f"{label}几何已创建，网格待更新"
        if TransitionEffect.NAMED_REGIONS_CLEARED in delta.effects:
            removed_region_count = (
                prior_region_count - len(self.document.named_regions)
            )
            message += f"；{removed_region_count} 个作用域失效"
        if TransitionEffect.LOCAL_CONTROLS_CLEARED in delta.effects:
            message += "；旧局部网格设置已失效"
        if delta.effects & {
            TransitionEffect.ASSIGNMENTS_CLEARED,
            TransitionEffect.STEPS_CLEARED,
        }:
            message += "；拓扑依赖失效"
        if TransitionEffect.MESH_SHAPE_NORMALIZED in delta.effects:
            message += "；单元类型已调整"
        if TransitionEffect.REFERENCES_PRESERVED in delta.effects:
            message += "；拓扑引用保留"
        self.status_panel.set_state(message, 6000)
        return True

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
        available_kinds = self._available_scope_selection_kinds()
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
        if not accepted:
            return None
        return dict(kinds).get(str(selected))

    def _scope_selection_topology(self) -> ScopeSelectionTopology:
        model = self._current_gui_model()
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

    def _mesh_selection_topology(self) -> MeshSelectionTopology:
        model = self._current_gui_model()
        if model is None:
            raise RuntimeError("mesh selection requires a generated mesh")
        if self._mesh_selection_topology_cache is None:
            scope_topology = self._scope_selection_topology_cache
            if scope_topology is None:
                scope_topology = build_scope_selection_topology(
                    model,
                    self.document.geometry_recipe,
                )
                self._scope_selection_topology_cache = scope_topology
            self._mesh_selection_topology_cache = (
                build_mesh_selection_topology(
                    model,
                    scope_topology=scope_topology,
                )
            )
        return self._mesh_selection_topology_cache

    def _mesh_selection_topology_requires_background(self) -> bool:
        """Return whether topology inference is large enough to leave the UI."""

        model = self._current_gui_model()
        return bool(
            model is not None
            and len(model.mesh.elements)
            > self.scope_background_reference_threshold
        )

    def _begin_mesh_selection_topology_preparation(
        self,
        selection_filter: str | None,
        *,
        on_ready: Callable[[], None] | None = None,
    ) -> None:
        """Prepare shared scope and mesh topology away from the GUI thread."""

        if (
            self.busy
            and self.task_controller.current_task_name
            == "准备网格选择拓扑"
        ):
            self._pending_mesh_topology_selection_filter = selection_filter
            self._pending_mesh_topology_callback = on_ready
            return
        model = self._current_gui_model()
        artifact = self.document.artifact
        if model is None or artifact is None:
            return
        artifact_id = artifact.artifact_id
        recipe = self.document.geometry_recipe

        def workload(context: TaskContext) -> object:
            context.report("正在分析网格作用域……")
            scope_topology = build_scope_selection_topology(model, recipe)
            context.checkpoint()
            context.report("正在准备边、面和体选择……")
            mesh_topology = build_mesh_selection_topology(
                model,
                scope_topology=scope_topology,
            )
            context.checkpoint()
            return artifact_id, scope_topology, mesh_topology

        def apply_result(value: object) -> TaskApplyOutcome:
            result_artifact_id, _scope_topology, _mesh_topology = value
            current_artifact = self.document.artifact
            if (
                current_artifact is None
                or current_artifact.artifact_id != result_artifact_id
            ):
                self._pending_mesh_topology_selection_filter = None
                self._pending_mesh_topology_callback = None
                return TaskApplyOutcome.stale("模型已变化，已丢弃选择拓扑")
            return TaskApplyOutcome.accepted(value)

        def project_result(value: object) -> None:
            _artifact_id, scope_topology, mesh_topology = value
            self._scope_selection_topology_cache = scope_topology
            self._mesh_selection_topology_cache = mesh_topology
            requested = self._pending_mesh_topology_selection_filter
            callback = self._pending_mesh_topology_callback
            self._pending_mesh_topology_selection_filter = None
            self._pending_mesh_topology_callback = None
            self.status_panel.set_state("网格选择拓扑已就绪", 3000)
            if callback is not None:
                callback()
                return
            active_filter = self._selection_context.active_filter
            if (
                requested is not None
                and self._selection_context.space == "mesh"
                and active_filter
                == ("point" if requested == "node" else requested)
            ):
                self._set_mesh_scope_selection_mode(requested)

        def clear_pending(*_args: object) -> None:
            self._pending_mesh_topology_selection_filter = None
            self._pending_mesh_topology_callback = None

        self._pending_mesh_topology_selection_filter = selection_filter
        self._pending_mesh_topology_callback = on_ready
        started = self._start_task(
            workload,
            project_result,
            "准备网格选择拓扑失败",
            clear_pending,
            task_name="准备网格选择拓扑",
            on_cancelled=clear_pending,
            apply_result=apply_result,
        )
        if not started:
            self._pending_mesh_topology_selection_filter = None
            self._pending_mesh_topology_callback = None

    def _current_gui_model(self) -> object | None:
        """Return the structural model facade used by GUI-only consumers."""

        if self.document.source_kind == "result":
            return self._result_archive_model_view
        return self.document.model

    def _available_scope_selection_kinds(
        self,
        capability_report: ModelCapabilityReport | None = None,
    ) -> set[str]:
        """Describe selectable scope kinds without materializing topology."""

        if self.document.model is None:
            return set()
        report = capability_report or self._model_capability_report()
        dimension = (
            report.topological_dimension
            if report is not None
            else None
        )
        kinds = {"node"}
        if dimension is not None and dimension >= 1:
            kinds.add("edge")
        if dimension is not None and dimension >= 2:
            kinds.add("face")
        if dimension is not None and dimension >= 3:
            kinds.add("body")
        return kinds

    def _start_edge_scope_selection(self) -> None:
        if self.document.model is None:
            return
        self._request_analysis_geometry_selection("scope", "edge")

    def _create_region_from_current_mesh_selection(
        self,
        *,
        requested_name: str | None = None,
        references: tuple[MeshEntityRef, ...] | None = None,
        on_committed: Callable[[str], None] | None = None,
    ) -> str | None:
        references = (
            self._canonical_mesh_scope_selection()
            if references is None
            else references
        )
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
        if receipt.status is GuiCommandStatus.PENDING:
            if on_committed is not None:
                completion = receipt.completion
                if completion is None:
                    raise RuntimeError(
                        "pending scope command requires a completion handle"
                    )

                def resume_after_commit(record: TaskCompletion) -> None:
                    if record.state is BackgroundTaskState.SUCCEEDED:
                        self._defer_ui(lambda: on_committed(name))

                completion.observe(resume_after_commit)
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
        authoring: SessionAuthoringProjection | None = None,
    ) -> tuple[Any, ...]:
        projection = (
            authoring
            if authoring is not None
            else self._session_authoring_projection()
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
        authoring = self._session_authoring_projection()
        targets = self._scope_authoring_targets(authoring)
        return tuple(
            [target.region for target in targets if target.region.kind == kind]
            for kind in ("node_set", "edge", "surface")
        )

    def _analysis_element_regions(
        self,
        authoring: SessionAuthoringProjection | None = None,
    ) -> list[RegionRef]:
        projection = authoring or self._session_authoring_projection()
        return [
            target.region
            for target in self._scope_authoring_targets(projection)
            if target.region.kind == "element_set"
        ]

    def _session_authoring_projection(self) -> SessionAuthoringProjection:
        """Reuse the immutable authoring projection for one Session snapshot."""

        builder = describe_session_authoring
        cached = self._session_authoring_cache
        if (
            cached is not None
            and cached[0] is self.document
            and cached[1] is builder
        ):
            return cached[2]
        projection = builder(self.document)
        self._session_authoring_cache = (
            self.document,
            builder,
            projection,
        )
        return projection

    def _model_capability_report(
        self,
        authoring: SessionAuthoringProjection | None = None,
    ) -> ModelCapabilityReport | None:
        """Return the headless capability report for current authoring state."""

        if self.document.source_kind != "result":
            return (
                authoring or self._session_authoring_projection()
            ).report
        model = self._current_gui_model()
        if model is not None:
            builder = describe_model_capabilities
            cached = self._result_model_capability_cache
            if (
                cached is not None
                and cached[0] is model
                and cached[1] is builder
            ):
                return cached[2]
            report = builder(model)
            self._result_model_capability_cache = (
                model,
                builder,
                report,
            )
            return report
        return None

    def _supported_load_regions(
        self,
        authoring: SessionAuthoringProjection | None = None,
    ) -> tuple[
        list[RegionRef],
        list[RegionRef],
        list[RegionRef],
        list[RegionRef],
        list[RegionRef],
    ]:
        """Return targets filtered by the application capability report."""

        projection = authoring or self._session_authoring_projection()
        targets = self._scope_authoring_targets(projection)
        operations = (
            ("node_set", "load.node"),
            ("edge", "load.edge"),
            ("surface", "load.surface"),
            ("element_set", "load.line.global"),
            ("element_set", "load.body"),
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

    def _supported_boundary_regions(
        self,
        authoring: SessionAuthoringProjection | None = None,
    ) -> list[RegionRef]:
        """Return typed regions that can expand to constrained mesh nodes."""

        projection = authoring or self._session_authoring_projection()
        return [
            target.region
            for target in self._scope_authoring_targets(projection)
            if target.region.kind in {"node_set", "edge", "surface"}
            and target.operation("boundary.displacement").can_submit
        ]

    def _request_analysis_geometry_selection(
        self,
        operation: str,
        scope_kind: str | None = None,
        *,
        resume_edit: (
            tuple[
                tuple[str, int, int | None],
                str,
                tuple[AnalysisStep, ...],
            ]
            | None
        ) = None,
        dialog_state: (
            DisplacementDialogState | LoadDialogState | None
        ) = None,
    ) -> None:
        if self.document.model is None:
            return
        self._pending_analysis_selection = operation
        self._pending_analysis_requested_scope_kind = (
            None if scope_kind is None else str(scope_kind)
        )
        self._pending_analysis_dialog_state = dialog_state
        self._pending_analysis_edit = resume_edit
        report = self._model_capability_report()
        dimension = (
            report.topological_dimension
            if report is not None
            else None
        )
        domain_kind = {
            1: "edge",
            2: "face",
            3: "body",
        }.get(dimension)
        if domain_kind is None:
            self._show_error("创建作用域", "当前模型缺少可选择的网格维度")
            self._pending_analysis_selection = None
            self._pending_analysis_requested_scope_kind = None
            self._pending_analysis_dialog_state = None
            self._pending_analysis_edit = None
            return
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
        semantic_kinds = self._available_scope_selection_kinds(report)
        semantic_kinds.discard("node")
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
            self._pending_analysis_requested_scope_kind = None
            self._pending_analysis_dialog_state = None
            self._pending_analysis_edit = None
            return
        self._pending_scope_kind = default_kind
        if default_kind in {"edge", "face", "body"}:
            if (
                self._scope_selection_topology_cache is None
                and self._mesh_selection_topology_requires_background()
            ):
                self._begin_mesh_selection_topology_preparation(
                    None,
                    on_ready=lambda: self._request_analysis_geometry_selection(
                        operation,
                        scope_kind,
                        resume_edit=resume_edit,
                        dialog_state=dialog_state,
                    ),
                )
                return
            topology = self._scope_selection_topology()
            self._selected_geometry_refs.clear()
            self._selected_mesh_scope_refs.clear()
            self.viewport.show_geometry_preview(
                topology.preview,
                preserve_model=True,
                render=False,
            )
            self._scope_selection_overlay_active = True
            self._begin_temporary_selection_context(
                "analysis_scope",
                "geometry",
                default_kind,
            )
        else:
            self._selected_geometry_refs.clear()
            self._selected_mesh_scope_refs.clear()
            self._begin_temporary_selection_context(
                "analysis_scope",
                "mesh",
                "point" if default_kind == "node" else default_kind,
            )
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
            f"选择{label}{kind_label}（Ctrl 多选，底栏完成）",
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
        references = self._canonical_mesh_scope_selection()
        if not references:
            self.status_panel.set_state("请先选择至少一个对象", 3000)
            return
        bar = self.viewport_panel.scope_creation_bar
        name = self._create_region_from_current_mesh_selection(
            requested_name=bar.scope_name(),
            references=references,
            on_committed=self._finish_scope_creation_from_bar,
        )
        if name is None:
            return
        self._finish_scope_creation_from_bar(name)

    def _finish_scope_creation_from_bar(self, name: str) -> None:
        """Finish a guided workflow only after its scope commit succeeds."""

        operation = self._pending_analysis_selection
        if operation is None:
            return
        bar = self.viewport_panel.scope_creation_bar
        resumed_edit = self._pending_analysis_edit
        dialog_state = getattr(
            self,
            "_pending_analysis_dialog_state",
            None,
        )
        selected_scope_kind = resumed_edit[1] if resumed_edit is not None else (
            getattr(
                self,
                "_pending_analysis_requested_scope_kind",
                None,
            )
            or getattr(dialog_state, "scope_kind", None)
            or {
                "node": "node",
                "edge": "edge",
                "face": "surface",
                "body": "body",
                "element": "line",
            }.get(getattr(self, "_pending_scope_kind", None))
        )
        self._pending_analysis_selection = None
        self._pending_analysis_requested_scope_kind = None
        self._pending_analysis_dialog_state = None
        self._pending_scope_kind = None
        self._pending_analysis_edit = None
        if self._scope_selection_overlay_active:
            self.viewport.hide_geometry_selection_overlay()
            self._scope_selection_overlay_active = False
        bar.finish()
        self._selected_geometry_refs.clear()
        self._selected_mesh_scope_refs.clear()
        self.viewport.clear_selection()
        self._restore_temporary_selection_context("analysis_scope")
        self.status_panel.set_object()
        self.actions["selected_info"].setEnabled(False)
        self._update_action_states()
        callback = None
        if resumed_edit is not None:
            definition_key, _requested_kind, edit_steps = resumed_edit
            region_kind = {
                "node": "node_set",
                "edge": "edge",
                "surface": "surface",
                "line": "element_set",
                "body": "element_set",
            }.get(str(selected_scope_kind))
            if region_kind is None:
                raise RuntimeError(
                    "unsupported analysis edit scope kind: "
                    f"{selected_scope_kind}"
                )
            selected_region = RegionRef(region_kind, name)

            def resume_definition_edit() -> None:
                kwargs = {
                    "selected_region": selected_region,
                    "steps": edit_steps,
                }
                if dialog_state is not None:
                    kwargs["dialog_state"] = dialog_state
                self._edit_analysis_definition_key(definition_key, **kwargs)

            callback = resume_definition_edit
        elif operation in {"boundary", "load"}:
            region_kind = {
                "node": "node_set",
                "edge": "edge",
                "surface": "surface",
                "line": "element_set",
                "body": "element_set",
            }.get(str(selected_scope_kind))
            if region_kind is None:
                raise RuntimeError(
                    "unsupported analysis scope kind: "
                    f"{selected_scope_kind}"
                )
            selected_region = RegionRef(region_kind, name)

            def resume_definition_create() -> None:
                kwargs = {}
                if dialog_state is not None:
                    kwargs["dialog_state"] = dialog_state
                if operation == "boundary":
                    self.create_displacement_boundary(
                        selected_region,
                        **kwargs,
                    )
                else:
                    self.create_load(selected_region, **kwargs)

            callback = resume_definition_create
        elif operation == "section":
            def resume_section_assignment() -> None:
                self.assign_section_to_region(name)

            callback = resume_section_assignment
        if callback is not None:
            self._defer_ui(callback)
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
        self.status_panel.set_state("网格设置已更新，需划分网格", 5000)

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
            "局部网格已更新，需重新划分",
            5000,
        )

    def set_local_mesh_control(self) -> None:
        settings = self.document.mesh_settings
        recipe = self.document.geometry_recipe
        if not isinstance(settings, MeshSettings) or not isinstance(
            recipe,
            NATIVE_GEOMETRY_TYPES,
        ):
            self._restore_temporary_selection_context("local_mesh")
            return
        local_control_action = self.actions["mesh_local_control"]
        if not local_control_action.isEnabled():
            self.status_panel.set_state(
                local_control_action.statusTip(),
                6000,
            )
            self._restore_temporary_selection_context("local_mesh")
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
            default_kind = (
                "face"
                if geometry_dimension(recipe) == 3
                else "point"
                if is_wire
                else "edge"
            )
            self._begin_temporary_selection_context(
                "local_mesh",
                "geometry",
                default_kind,
            )
            self.status_panel.set_state(
                f"选择局部网格{'面' if default_kind == 'face' else '边'}"
                "（Ctrl 多选，Enter 完成）",
                0,
            )
            return
        dialog = LocalMeshControlDialog(
            selected_references[0],
            settings.size,
            self,
        )
        if not self._exec_dialog(dialog):
            self._restore_temporary_selection_context("local_mesh")
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
            self._restore_temporary_selection_context("local_mesh")
            return
        kind_name = {
            "point": "点",
            "edge": "边",
            "face": "面",
        }.get(control.target.kind, "实体")
        self.status_panel.set_state(
            f"{kind_name}局部尺寸已设置，需重新划分",
            5000,
        )
        self._restore_temporary_selection_context("local_mesh")

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
            "网格已清除，相关定义已失效",
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
                    "模型已变化，旧网格检查已忽略",
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

    def _record_agent_workflow_proposal_state(
        self,
        operation: str,
        state: ProposalState,
        message: str,
    ) -> None:
        if not hasattr(self, "viewport_panel"):
            return
        runtime = self.viewport_panel.agent_chat_drawer.agent_runtime
        try:
            runtime.record_authoring_proposal_state_from_gui(
                operation,
                state,
                message,
            )
        except (RuntimeError, ValueError):
            pass

    def _record_agent_workflow_preflight_state(
        self,
        state: str,
        message: str,
    ) -> None:
        if not hasattr(self, "viewport_panel"):
            return
        runtime = self.viewport_panel.agent_chat_drawer.agent_runtime
        try:
            runtime.record_authoring_preflight_state_from_gui(
                state,
                message,
            )
        except (RuntimeError, ValueError):
            pass

    def _begin_agent_mesh_generation(
        self,
        request: AgentMeshTaskRequest,
    ) -> bool:
        """Start the A3 detached task after the bridge consumed GUI authority."""

        task = request.task
        port = self.agent_authoring_bridge.port
        complete_mesh = getattr(port, "complete_mesh", None)
        accept_mesh_result = getattr(port, "accept_mesh_result", None)
        terminate_mesh = getattr(port, "terminate_mesh", None)
        if not all(
            callable(item)
            for item in (complete_mesh, accept_mesh_result, terminate_mesh)
        ):
            raise RuntimeError("Agent mesh lifecycle port is unavailable")

        def workload(context: TaskContext) -> object:
            context.report("正在生成 Agent 网格……")
            started = perf_counter()
            try:
                model = generate_fem_model(
                    task,
                    cancelled=lambda: context.is_cancelled,
                )
            except GmshExecutionCancelled:
                context.checkpoint()
                raise
            timings = {
                "Gmsh 几何与网格": perf_counter() - started,
            }
            context.report("正在准备显示网格……")
            started = perf_counter()
            display_geometry = build_model_geometry(model)
            timings["VTK 显示几何构建"] = perf_counter() - started
            context.checkpoint()
            return model, display_geometry, timings

        def apply_result(value: object) -> TaskApplyOutcome:
            model, _geometry, _timings, _notices = self._unpack_model_load(
                value
            )
            outcome = self._session_task_outcome(
                accept_mesh_result(request.proposal_id, model),
                value,
            )
            return outcome

        def project_result(value: object) -> None:
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
                raise RuntimeError("已接受的 Agent 网格结果无法投影")
            self._import_notices = ()

        def terminal(record: TaskCompletion) -> None:
            if record.state is BackgroundTaskState.SUCCEEDED:
                self._record_agent_workflow_proposal_state(
                    "mesh",
                    ProposalState.SUCCEEDED,
                    "Agent 网格已原子安装",
                )
                return
            state = {
                BackgroundTaskState.CANCELLED: ProposalState.CANCELLED,
                BackgroundTaskState.DISCARDED: ProposalState.STALE,
            }.get(record.state, ProposalState.FAILED)
            terminate_mesh(
                request.proposal_id,
                state,
                record.message or record.state.value,
            )
            self._record_agent_workflow_proposal_state(
                "mesh",
                state,
                record.message or record.state.value,
            )
            if record.state is BackgroundTaskState.FAILED:
                self.status_panel.set_state(
                    record.message or "Agent 网格生成失败",
                    5000,
                )
            elif record.state is BackgroundTaskState.CANCELLED:
                self.status_panel.set_state("已取消 Agent 网格生成", 4000)
            else:
                self.status_panel.set_state(
                    record.message or "Agent 网格结果已陈旧，未应用",
                    5000,
                )

        task_id = self.task_controller.start(
            workload,
            task_name="Agent 网格生成",
            apply_result=apply_result,
            project_result=project_result,
            rebuild_projection=self._rebuild_full_projection,
            on_terminal=terminal,
            on_progress=self.status_panel.set_state,
            on_projection_error=self._task_projection_failed,
        )
        return task_id is not None

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
            on_inactive_failure=lambda message: self._session_task_failed(
                task.token,
                "网格生成失败",
                message,
            ),
            on_inactive_cancelled=lambda: self._session_task_cancelled(
                task.token
            ),
            apply_result=apply_result,
            completion=completion,
        )

    def open_inp(self) -> None:
        path, _filter = QFileDialog.getOpenFileName(
            self, "打开 Abaqus INP", "", "Abaqus INP 文件 (*.inp);;所有文件 (*)"
        )
        if path:
            receipt = self.open_inp_path(Path(path))
            if receipt.diagnostic is not None:
                self._show_command_rejection("模型加载失败", receipt)

    def save_current_result(
        self,
        *,
        wait: bool = False,
        force_save_as: bool = False,
    ) -> bool:
        """Save the displayed result, prompting when no target exists."""

        if self.busy or self._current_result_provider() is None:
            return False
        path = self.document.result_path
        run = self.session.find_run(self.document.displayed_result_run_id)
        model_name = str(self.document.model_name or "模型").strip() or "模型"
        job_name = str(getattr(run, "name", None) or "结果").strip() or "结果"
        if force_save_as or path is None:
            default_name = (
                Path(path).with_suffix(RESULT_FILE_SUFFIX).name
                if path is not None
                else f"{model_name}-{job_name}{RESULT_FILE_SUFFIX}"
            )
            filename, _filter = QFileDialog.getSaveFileName(
                self,
                "保存分析结果",
                default_name,
                "FEM-Python 结果 (*.femres)",
            )
            if not filename:
                return False
            path = Path(filename)
        receipt = self.save_result_path(path)
        if receipt.diagnostic is not None:
            self._show_command_rejection("保存分析结果失败", receipt)
            return False
        completion = receipt.completion
        if completion is None:
            return False
        if not wait:
            return True
        try:
            terminal = completion.result(_SYNCHRONOUS_GUI_COMMAND_TIMEOUT_SECONDS)
        except TimeoutError:
            self.cancel_current_task()
            self.status_panel.set_state("保存分析结果超时，正在取消任务", 5000)
            return False
        deadline = perf_counter() + 5.0
        while self.busy and perf_counter() < deadline:
            QApplication.processEvents()
            sleep(0.001)
        QApplication.processEvents()
        return (
            terminal.state is BackgroundTaskState.SUCCEEDED
            and not self.busy
        )

    def save_current_result_as(self, *, wait: bool = False) -> bool:
        return self.save_current_result(wait=wait, force_save_as=True)

    def open_result_file(self) -> None:
        """Show the result archive dialog and install only after confirmation."""

        if self.busy or self._active_editor():
            return
        path, _filter = QFileDialog.getOpenFileName(
            self,
            "打开分析结果",
            "",
            "FEM-Python 结果 (*.femres);;所有文件 (*)",
        )
        if not path:
            return
        receipt = self.open_result_path(path)
        if receipt.diagnostic is not None:
            self._show_command_rejection("打开分析结果失败", receipt)

    def new_native_model(self) -> None:
        if self.busy:
            return
        model_name, accepted = QInputDialog.getText(
            self,
            "新建模型",
            "模型名称：",
            text=f"模型-{self.workspace.next_model_number}",
        )
        if not accepted:
            return
        model_name = model_name.strip()
        if not model_name:
            self._show_error("新建模型", "模型名称不能为空。")
            return
        # New models are appended as independent workspace documents; the
        # current document remains intact, so no discard confirmation applies.
        self._create_native_model(model_name)

    def _create_native_model(self, model_name: str) -> None:
        """Create a native project after its model name has been collected."""

        if self.busy:
            return
        previous_context = self._active_context
        # New model commands append a fresh Session/context.  The existing
        # active model remains in the workspace and is only switched away
        # after the new context has been installed.
        context = self.workspace.add_model(display_name=model_name)
        if not self._activate_workspace_context(context):
            self.workspace.remove(context)
            return
        receipt = self.new_native_project(
            NewNativeProjectCommand(model_name)
        )
        if receipt.diagnostic is not None:
            self.model_tree.remove_document(context.document_id)
            self.workspace.remove(context)
            if previous_context is not None:
                self._activate_workspace_context(previous_context)
            else:
                self._active_context = None
            self._show_command_rejection("新建自主项目", receipt)
            return
        self.viewport_panel.set_geometry_context(True)
        self.status_panel.set_state(
            "模型已创建，请新建部件",
            5000,
        )
        self.ribbon.set_current("几何")

    def open_native_project(self) -> None:
        path, _filter = QFileDialog.getOpenFileName(
            self,
            "打开自主项目",
            "",
            _native_model_open_filter(),
        )
        if not path:
            return
        receipt = self.open_project_path(path)
        if receipt.diagnostic is not None:
            self._show_command_rejection("打开自主项目失败", receipt)
            return
        completion = receipt.completion
        if completion is not None:
            def project_opened(record: TaskCompletion) -> None:
                if record.state is not BackgroundTaskState.SUCCEEDED:
                    return
                self.ribbon.set_current("几何")
                if not self.import_notices:
                    self.status_panel.set_state(
                    "项目已打开，请划分并检查网格",
                        6000,
                    )

            completion.observe(project_opened)
        else:
            self.ribbon.set_current("几何")

    def save_native_project(
        self,
        *,
        wait: bool = False,
        force_save_as: bool = False,
        agent_terminal: (
            Callable[[ProposalState, str], None] | None
        ) = None,
    ) -> bool:
        if not self.document.can_save:
            if agent_terminal is not None:
                agent_terminal(
                    ProposalState.FAILED,
                    "当前项目无法保存",
                )
            return False
        path = self.document.project_path
        save_as = (
            force_save_as
            or
            path is None
            or self._legacy_project_extension
            or path.suffix.casefold() != MODEL_FILE_SUFFIX
        )
        if save_as:
            default_name = (
                Path(path).with_suffix(MODEL_FILE_SUFFIX).name
                if path is not None
                else f"{self.document.model_name or '模型-1'}{MODEL_FILE_SUFFIX}"
            )
            filename, _filter = QFileDialog.getSaveFileName(
                self,
                "保存自主项目",
                default_name,
                _native_model_save_filter(),
            )
            if not filename:
                if agent_terminal is not None:
                    agent_terminal(
                        ProposalState.CANCELLED,
                        "用户取消了另存为",
                    )
                return False
            path = Path(filename)
            if path.suffix.casefold() != MODEL_FILE_SUFFIX:
                path = path.with_suffix(MODEL_FILE_SUFFIX)
        receipt = self.save_project_path(path)
        if receipt.diagnostic is not None:
            self._show_command_rejection("保存自主项目失败", receipt)
            if agent_terminal is not None:
                agent_terminal(
                    ProposalState.FAILED,
                    "保存任务未能启动",
                )
            return False
        completion = receipt.completion
        if completion is None:
            if agent_terminal is not None:
                agent_terminal(
                    ProposalState.FAILED,
                    "保存任务未能启动",
                )
            return False

        def project_saved(record: TaskCompletion) -> None:
            if record.state is not BackgroundTaskState.SUCCEEDED:
                return
            if self.document.dirty:
                self.status_panel.set_state(
                    "项目快照已保存，当前修改未保存",
                    6000,
                )
            else:
                self.status_panel.set_state(
                    f"自主项目已保存：{path.name}",
                    5000,
                )

        completion.observe(project_saved)
        if agent_terminal is not None:

            def agent_save_finished(record: TaskCompletion) -> None:
                state = {
                    BackgroundTaskState.SUCCEEDED: ProposalState.SUCCEEDED,
                    BackgroundTaskState.CANCELLED: ProposalState.CANCELLED,
                    BackgroundTaskState.DISCARDED: ProposalState.STALE,
                }.get(record.state, ProposalState.FAILED)
                message = {
                    ProposalState.SUCCEEDED: "自主项目已保存",
                    ProposalState.CANCELLED: "保存任务已取消",
                    ProposalState.STALE: "保存快照已陈旧",
                    ProposalState.FAILED: "保存自主项目失败",
                }[state]
                agent_terminal(state, message)

            completion.observe(agent_save_finished)
        if not wait:
            return True
        try:
            terminal = completion.result(_SYNCHRONOUS_GUI_COMMAND_TIMEOUT_SECONDS)
        except TimeoutError:
            self.cancel_current_task()
            self.status_panel.set_state("保存自主项目超时，正在取消任务", 5000)
            return False
        return (
            terminal.state is BackgroundTaskState.SUCCEEDED
            and not self.document.dirty
        )

    def save_native_project_as(self, *, wait: bool = False) -> bool:
        return self.save_native_project(wait=wait, force_save_as=True)

    def _start_agent_project_save(
        self,
        terminal: Callable[[ProposalState, str], None],
    ) -> bool:
        """Enter the existing project-save command from the GUI card only."""

        if not callable(terminal):
            raise TypeError("project save terminal callback must be callable")
        return self.save_native_project(agent_terminal=terminal)

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
            context.report("正在解析并构建 INP……")
            started = perf_counter()
            import_result = read_inp_with_report(path)
            timings["INP 解析与构建"] = perf_counter() - started
            model = import_result.model
            context.report("正在生成显示网格……")
            started = perf_counter()
            geometry = build_model_geometry(model)
            timings["VTK 显示几何构建"] = perf_counter() - started
            context.report("正在准备会话模型……")
            started = perf_counter()
            prepared = self.session.prepare_owned_imported_model_transfer(
                model
            )
            timings["Session 模型所有权准备"] = perf_counter() - started
            context.checkpoint()
            return prepared, geometry, timings, import_result.notices

        def apply_result(value: object) -> TaskApplyOutcome:
            prepared, _geometry, _timings, _notices = self._unpack_model_load(
                value
            )
            return self._session_task_outcome(
                self.session.accept_imported_model_transfer(
                    task.token,
                    prepared,
                ),
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
            on_inactive_failure=lambda message: self._session_task_failed(
                task.token,
                "模型加载失败",
                message,
            ),
            on_inactive_cancelled=lambda: self._session_task_cancelled(
                task.token
            ),
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
            "导入结果过期，未应用",
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
            "网格结果过期，未应用",
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

        self._import_notices = tuple(notices)
        if not self._import_notices:
            return
        self.status_panel.set_state(
            "；".join(
                str(getattr(notice, "message", notice))
                for notice in self._import_notices
            ),
            12000,
        )

    def _show_model_in_tree(
        self,
        model: object,
        *,
        context: WorkspaceDocument | None = None,
    ) -> None:
        target_context = context or self.workspace.active_document()
        target_document_id = (
            None if target_context is None else target_context.document_id
        )
        target_document = (
            self.document if target_context is None else target_context.projection
        )
        output_catalog = ResultCapabilityCatalog.from_profile(
            classify_result_model(model)
        )
        output_projections = _output_request_projections_by_step(
            tuple(target_document.steps),
            output_catalog,
        )
        definition_options = {
            "section_definitions": tuple(
                target_document.sections
            ),
            "region_assignments": tuple(
                target_document.assignments
            ),
            "output_request_projections_by_step": output_projections,
        }
        source_path = target_document.source_path or target_document.project_path
        if target_document.source_kind == "native" and target_document.parts:
            self.model_tree.set_model(
                model,
                model_name=str(
                    target_context.display_name
                    if target_context is not None
                    else target_document.model_name or "模型-1"
                ),
                scope_names=frozenset(target_document.named_regions),
                native_parts=tuple(target_document.parts),
                active_part_id=target_document.active_part_id,
                document_id=target_document_id,
                source_path=source_path,
                **definition_options,
            )
            return
        self.model_tree.set_model(
            model,
            model_name=(
                None
                if target_context is None
                else target_context.display_name
            ),
            document_id=target_document_id,
            source_path=source_path,
            **definition_options,
        )

    def _refresh_model_tree_for_context(
        self,
        context: WorkspaceDocument,
    ) -> None:
        """Incrementally project one context's root into ``ModelTree``."""

        if context.is_result:
            # External result documents own display geometry in the result
            # tree; they must never add a model-document root.
            self.model_tree.remove_document(context.document_id)
            return
        snapshot = context.projection
        artifact = snapshot.artifact
        if artifact is not None:
            self._show_model_in_tree(artifact.model, context=context)
            return
        self.model_tree.set_geometry_preview(
            str(context.display_name or snapshot.model_name or "模型"),
            (),
            parts=tuple(snapshot.parts)
            if snapshot.source_kind == "native"
            else None,
            active_part_id=snapshot.active_part_id,
            document_id=context.document_id,
            source_path=snapshot.source_path or snapshot.project_path,
        )

    def _refresh_result_tree_for_context(
        self,
        context: WorkspaceDocument,
        *,
        catalog: object | None = None,
    ) -> None:
        """Incrementally project one model's jobs or one archive result root.

        ResultTree owns one indexed root per successful run or archive.  This helper
        deliberately performs no viewport, ribbon, or result-control work, so
        deltas for inactive documents stay on the cheap tree-only path.
        """

        provider = None
        if catalog is None:
            try:
                provider = context.session.current_result_provider()
            except (AttributeError, RuntimeError, ValueError):
                provider = None
            if provider is not None:
                try:
                    catalog = provider.catalog()
                except (AttributeError, RuntimeError, ValueError):
                    catalog = None
        labels = None
        if provider is not None:
            try:
                labels = result_provider_section_point_labels(provider)
            except (AttributeError, RuntimeError, ValueError):
                labels = None
        source_path = context.source_path or getattr(
            context.projection,
            "project_path",
            None,
        )
        if context.is_result:
            self.result_tree.upsert_archive(
                context.document_id,
                context.projection,
                display_name=context.display_name,
                source_path=source_path,
                catalog=catalog if isinstance(catalog, ResultCatalog) else None,
                section_point_labels=labels,
            )
        else:
            self.result_tree.upsert_model_runs(
                context.document_id,
                context.projection,
                display_name=context.display_name,
                source_path=source_path,
                catalog=catalog if isinstance(catalog, ResultCatalog) else None,
                section_point_labels=labels,
            )

    def open_result_path(self, path: str | Path) -> GuiCommandReceipt:
        """Decode an archive in the background and append it to the workspace."""

        command_id = self._next_command_id()
        if self._active_editor():
            return self._rejected_command(
                command_id,
                "editor.active",
                "active editor must be completed before opening a result",
            )
        try:
            spec = ResultArchiveOpenSpec(Path(path))
            existing_id = self.workspace.result_paths.get(canonical_path(spec.path))
            if existing_id is not None:
                context = self.workspace.document(existing_id)
                if not self._activate_workspace_context(context):
                    return self._rejected_command(
                        command_id,
                        "result.activate.rejected",
                        "the existing result document could not be activated",
                    )
                self.ribbon.set_current("结果")
                return GuiCommandReceipt.accepted(
                    command_id,
                    outcome=GuiCommandOutcome(output_path=spec.path),
                )
            open_controller = self.workspace.ensure_open_controller()
            if open_controller.busy:
                return self._rejected_command(
                    command_id,
                    "task.busy",
                    "a result archive is already being opened",
                )
            completion = GuiCommandCompletion(command_id)

            def workload(task_context: TaskContext):
                task_context.report("reading and validating result archive")
                task_context.checkpoint()
                loaded = load_result_archive(spec.path)
                task_context.checkpoint()
                started = perf_counter()
                model_view = build_result_archive_model_view(
                    loaded.snapshot.model_projection,
                    loaded.snapshot.profile,
                    name=str(loaded.snapshot.origin.model_name or "Result"),
                )
                geometry = build_result_archive_geometry(
                    loaded.snapshot.model_projection,
                )
                timings = {"result display adapters": perf_counter() - started}
                task_context.checkpoint()
                return _ResultArchiveDisplayPayload(
                    loaded=loaded,
                    geometry=geometry,
                    model_view=model_view,
                    timings=timings,
                )

            def on_success(value: object) -> None:
                if type(value) is not _ResultArchiveDisplayPayload:
                    raise TypeError("result open worker must return display payload")
                payload = value
                session = ModelSession()
                delta = session.replace_from_result_archive(
                    payload.loaded,
                    path=spec.path,
                )
                if not delta.accepted:
                    raise RuntimeError(delta.reason or "result archive was rejected")
                projection = session.projection_snapshot()
                display_name = spec.path.stem
                previous_context = self.workspace.active_document()
                context = self.workspace.add_result(
                    session=session,
                    projection=projection,
                    display_name=display_name,
                    source_path=spec.path,
                )
                context.presentation_state.module_name = "结果"
                artifact = projection.artifact
                context.presentation_cache.artifact_id = (
                    None if artifact is None else artifact.artifact_id
                )
                context.presentation_cache.model_geometry = payload.geometry
                context.presentation_cache.result_model_view = payload.model_view
                identity = session.current_result_identity()
                if identity is not None:
                    context.presentation_cache.result_source = identity[0]
                    context.presentation_cache.result_generation = identity[1]
                try:
                    self._refresh_result_tree_for_context(context)
                    if not self._activate_workspace_context(context):
                        raise RuntimeError(
                            "result document could not be activated"
                        )
                    result_module_was_current = (
                        self._current_module_name() == "结果"
                    )
                    self.ribbon.set_current("结果")
                    if result_module_was_current:
                        self._on_module_changed("结果")
                    self.status_panel.set_state(
                        f"结果文件已打开：{spec.path.name}",
                        6000,
                    )
                except Exception:
                    self.result_tree.remove_archive(context.document_id)
                    self.workspace.remove(context)
                    if previous_context is not None:
                        try:
                            self._activate_workspace_context(previous_context)
                        except Exception:
                            logging.exception(
                                "failed to restore prior context after result open rollback"
                            )
                    raise

            started = self._start_task(
                workload,
                on_success,
                "打开分析结果失败",
                task_name="打开分析结果",
                completion=completion,
                controller=open_controller,
            )
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            return self._rejected_command(command_id, "result.open.rejected", error)
        if not started:
            return self._rejected_command(
                command_id,
                "task.start.rejected",
                "the result open task could not be started",
            )
        return GuiCommandReceipt.pending(command_id, completion)

    def _install_model(
        self,
        model: object,
        geometry: ModelGeometry,
        timings: dict[str, float],
        *,
        source_label: str,
        inspection_service: InspectionService | None = None,
        render: bool = True,
        reset_camera: bool = True,
        preserve_display: bool = False,
    ) -> None:
        self.status_panel.set_state("正在初始化视口……")
        self._scope_selection_overlay_active = False
        self._scope_selection_topology_cache = None
        self._mesh_selection_topology_cache = None
        self._restore_temporary_selection_context()
        self._close_inspection_windows()
        self._close_job_manager()
        self.geometry = geometry
        self.result_provider = None
        if not preserve_display:
            self.result_selection = None
        self.selection.clear()
        self._selected_geometry_refs.clear()
        self._selected_mesh_scope_refs.clear()
        if not preserve_display:
            self._display = DisplayState()
        if not preserve_display:
            self._overlay_undeformed = False
            self.actions["undeformed"].setChecked(True)
            self.actions["contour"].setChecked(False)
            self.actions["overlay"].setChecked(False)
        self._symbol_settings = replace(
            self._symbol_settings,
            step_name=self._current_step_name,
        )
        frame_query = None
        if self.document.source_kind != "result":
            def frame_query(target: RegionRef | int) -> BeamFrameReport:
                return resolve_effective_beam_frames(model, target)
        if inspection_service is not None:
            self.inspection_service = inspection_service
        else:
            started = perf_counter()
            self.inspection_service = InspectionService(
                model,
                definitions=self.document,
                effective_frame_query=frame_query,
            )
            timings["InspectionService 初始化"] = perf_counter() - started
        started = perf_counter()
        active_context = self.workspace.active_document()
        if active_context is not None:
            if active_context.is_result:
                self.model_tree.remove_document(active_context.document_id)
            else:
                self._show_model_in_tree(model, context=active_context)
        timings["模型树更新"] = perf_counter() - started
        if active_context is not None:
            self._refresh_result_tree_for_context(active_context)
        else:
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
        self.viewport.set_symbol_sampling_density_override(
            None if self.document.source_kind == "result" else (
                "low" if policy["reduce_symbols"] else None
            )
        )

        started = perf_counter()
        self.viewport.set_model(
            model,
            geometry,
            refresh_symbols=False,
            render=False,
            reset_camera=reset_camera,
            show_edges=policy["show_edges"],
            show_nodes=policy["show_nodes"],
            show_node_labels=policy["show_labels"],
            show_element_labels=policy["show_labels"],
            effective_frame_query=frame_query,
            mesh_selection_topology_provider=(
                self._mesh_selection_topology
            ),
        )
        self.viewport.set_edges_visible(self.actions["edges"].isChecked(), render=False)
        self.viewport.set_nodes_visible(self.actions["nodes"].isChecked(), render=False)
        timings["视口网格创建"] = perf_counter() - started
        self.viewport.set_symbol_settings(self._symbol_settings, refresh=False, render=False)
        self.viewport.set_symbols_visible(
            self.actions["symbols"].isChecked(), refresh=False, render=False
        )
        if (
            self.document.source_kind != "result"
            and not self._workspace_activation
        ):
            started = perf_counter()
            self.viewport.show_boundary_and_loads(render=False)
            timings["载荷约束符号创建"] = perf_counter() - started
        if render:
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
                "大型模型：已简化首次显示",
                8000,
            )
        else:
            self.status_panel.set_state("模型加载完成", 5000)
        self._refresh_result_controls()
        self._sync_step_combos()
        self._update_action_states()

    def _active_editor(self) -> bool:
        return self._solid_face_boolean_operation is not None or any(
            controller is not None
            for controller in (
                self._wire_editor_controller,
                self._sketch_editor_controller,
                self._body_boolean_controller,
                self._planar_boolean_controller,
                self._face_sketch_controller,
            )
        )

    def _confirm_discard_changes(self) -> bool:
        """Compatibility entry point for the unified document transition gate."""

        return self._confirm_document_transition()

    def _confirm_result_invalidation(self, *, preserve_editor: bool = False) -> bool:
        """Allow model edits; completed results are immutable run history."""

        return True

    def _confirm_document_transition(self, *, preserve_editor: bool = False) -> bool:
        """Confirm editor, project, and result transitions before mutation."""

        wire_editor_active = (
            not preserve_editor and self._wire_editor_controller is not None
        )
        sketch_editor_active = (
            not preserve_editor and self._sketch_editor_controller is not None
        )
        body_boolean_active = (
            not preserve_editor and self._body_boolean_controller is not None
        )
        planar_boolean_active = (
            not preserve_editor and self._planar_boolean_controller is not None
        )
        face_sketch_active = (
            not preserve_editor and self._face_sketch_controller is not None
        )
        solid_face_boolean_active = (
            not preserve_editor
            and self._solid_face_boolean_operation is not None
            and self._face_sketch_controller is None
        )
        editor_active = (
            wire_editor_active
            or sketch_editor_active
            or body_boolean_active
            or planar_boolean_active
            or face_sketch_active
            or solid_face_boolean_active
        )
        if wire_editor_active and self._wire_editor_controller.dirty:
            if not self._confirm_wire_editor_discard():
                return False
        if sketch_editor_active and self._sketch_editor_controller.dirty:
            if not self._confirm_sketch_editor_discard():
                return False
        unsaved_result_count = self.session.unsaved_result_count
        if not self.document.dirty and not unsaved_result_count:
            if wire_editor_active:
                self._exit_wire_editor()
            elif sketch_editor_active:
                self._exit_sketch_editor()
            if face_sketch_active:
                self.cancel_face_sketch_boolean()
            if body_boolean_active:
                self._exit_body_boolean()
            if planar_boolean_active:
                self._exit_planar_boolean()
            if solid_face_boolean_active:
                self._cancel_solid_face_boolean()
            if editor_active:
                self._rebuild_full_projection()
            return True
        box = QMessageBox(self)
        box.setWindowTitle("未保存的文档内容")
        box.setIcon(QMessageBox.Icon.Warning)
        messages: list[str] = []
        if self.document.dirty:
            messages.append("当前模型包含尚未保存的修改。")
        if unsaved_result_count:
            current_run = self.session.find_run(
                self.document.displayed_result_run_id
            )
            job_name = getattr(current_run, "name", None) or "当前作业"
            messages.append(
                f"有 {unsaved_result_count} 个未保存结果（当前作业：{job_name}）。"
            )
        box.setText("\n".join(messages))
        save_button = None
        if self.document.can_save and not unsaved_result_count:
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
            accepted = self.save_native_project(wait=True)
        else:
            accepted = clicked is discard_button
        if accepted and wire_editor_active:
            self._exit_wire_editor()
            self._rebuild_full_projection()
        elif accepted and sketch_editor_active:
            self._exit_sketch_editor()
            self._rebuild_full_projection()
        if accepted and face_sketch_active:
            self.cancel_face_sketch_boolean()
        if accepted and body_boolean_active:
            self._exit_body_boolean()
            self._rebuild_full_projection()
        if accepted and planar_boolean_active:
            self._exit_planar_boolean()
            self._rebuild_full_projection()
        if accepted and solid_face_boolean_active:
            self._cancel_solid_face_boolean()
        return accepted

    def _confirm_workspace_context_close(
        self,
        context: WorkspaceDocument,
        confirm: bool,
    ) -> bool:
        """Apply the close guard to an arbitrary (possibly inactive) context."""

        dirty = bool(context.projection.dirty)
        unsaved_results = int(context.session.unsaved_result_count)
        if not dirty and not unsaved_results:
            return True
        if not confirm:
            # The caller has already performed the confirmation (for example
            # the deferred cooperative-cancellation close path).
            return True
        box = QMessageBox(self)
        box.setWindowTitle("未保存的文档内容")
        box.setIcon(QMessageBox.Icon.Warning)
        box.setText(
            f"模型“{context.display_name}”包含未保存的内容，是否关闭？"
        )
        discard = box.addButton(
            "放弃修改并关闭",
            QMessageBox.ButtonRole.DestructiveRole,
        )
        box.addButton("取消", QMessageBox.ButtonRole.RejectRole)
        box.exec()
        return box.clickedButton() is discard

    def _request_workspace_context_cancel(
        self,
        context: WorkspaceDocument,
        *,
        after_cleanup: Callable[[], None] | None = None,
    ) -> bool:
        """Request cooperative cancellation for one document-owned task."""

        running = next(
            (
                run
                for run in context.projection.runs
                if run.status is RunStatus.RUNNING
            ),
            None,
        )
        if running is not None:
            try:
                delta = context.session.request_cancel(running.run_id)
                self._apply_session_delta(delta, context=context)
            except (KeyError, RuntimeError, TypeError, ValueError):
                pass
        return context.task_controller.request_cancel(after_cleanup=after_cleanup)

    def _close_workspace_context_after_cancel(self, document_id: int) -> None:
        """Finish a deferred close once the target controller is idle."""

        try:
            context = self.workspace.document(int(document_id))
        except (KeyError, TypeError, ValueError):
            return
        if context.task_controller.busy:
            return
        self.close_model(confirm=False, document_id=context.document_id)

    def close_model(
        self,
        *,
        confirm: bool = True,
        document_id: int | None = None,
    ) -> bool:
        target_context = (
            self.workspace.active_document()
            if document_id is None
            else self.workspace.document(int(document_id))
        )
        if target_context is None:
            return False
        active_context = self.workspace.active_document()
        was_active = (
            target_context is self._active_context
            and active_context is target_context
        )
        if was_active:
            if confirm and not self._confirm_discard_changes():
                return False
        elif not self._confirm_workspace_context_close(target_context, confirm):
            return False
        if target_context.task_controller.busy:
            requested = self._request_workspace_context_cancel(
                target_context,
                after_cleanup=lambda document_id=target_context.document_id: self._close_workspace_context_after_cancel(
                    document_id
                ),
            )
            # A second close request while cancellation is already underway
            # still owns the same deferred release callback.
            return bool(requested or target_context.task_controller.cancel_requested)
        if (
            was_active
            and self._solid_face_boolean_operation is not None
        ):
            self._cancel_solid_face_boolean()
        replacement = None
        if was_active:
            same_kind = tuple(
                self.workspace.results.values()
                if target_context.is_result
                else self.workspace.models.values()
            )
            target_index = same_kind.index(target_context)
            if target_index + 1 < len(same_kind):
                replacement = same_kind[target_index + 1]
            elif target_index:
                replacement = same_kind[target_index - 1]
            else:
                replacement = next(
                    (
                        context
                        for context in self.workspace.documents()
                        if context is not target_context
                    ),
                    None,
                )
            if replacement is not None and not self._activate_workspace_context(
                replacement
            ):
                return False
        if target_context.projection.is_open:
            receipt = self.close_session(
                CloseSessionCommand(target_context.projection.session_revision),
                document_id=target_context.document_id,
            )
            if receipt.diagnostic is not None:
                self._show_command_rejection("关闭模型", receipt)
                return False
        closed_id = target_context.document_id
        target_context.presentation_cache.invalidate_model()
        target_context.presentation_cache.invalidate_result()
        target_context.presentation_state = DocumentPresentationState()
        if target_context.is_result:
            self.result_tree.remove_archive(closed_id)
        else:
            self.model_tree.remove_document(closed_id)
            self.result_tree.remove_model_runs(closed_id)
        self.workspace.remove(target_context)
        if replacement is None:
            replacement = self.workspace.active_document()
        if replacement is None:
            self._active_context = None
        elif was_active:
            # The replacement was preflighted before removal so activation
            # cannot fail after the target registry entry is gone.  Refresh
            # only its existing root for compatibility with the normal
            # post-close projection path.  Result documents have no model
            # artifact and therefore refresh the result root instead.
            if replacement.is_result:
                self._refresh_result_tree_for_context(replacement)
            else:
                self._refresh_model_tree_for_context(replacement)
        if was_active and replacement is None:
            self._clear_model_projection(clear_tree=True)
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

    def assign_section_to_region(
        self,
        selected_region_name: str | None = None,
    ) -> None:
        if not self.document.sections:
            return
        if (
            selected_region_name is None
            and self._mesh_scope_selection_kind() == "element"
        ):
            selected_region_name = (
                self._create_region_from_current_mesh_selection(
                    on_committed=self.assign_section_to_region,
                )
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
        regions = self._analysis_element_regions()
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
        name = f"分析步-{len(definitions) + 1}"
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

    def create_displacement_boundary(
        self,
        selected_region: RegionRef | None = None,
        *,
        dialog_state: DisplacementDialogState | None = None,
    ) -> None:
        model = self.document.model
        if not self.session.runnable_step_names():
            return
        if self.document.source_kind == "native" and model is None:
            return
        selected_mesh_kind = self._mesh_scope_selection_kind()
        if (
            selected_region is None
            and selected_mesh_kind in {"node", "edge", "face"}
        ):
            selected_region_kind = {
                "node": "node_set",
                "edge": "edge",
                "face": "surface",
            }[selected_mesh_kind]
            selected_name = self._create_region_from_current_mesh_selection(
                on_committed=lambda name: self.create_displacement_boundary(
                    RegionRef(selected_region_kind, name)
                ),
            )
            if selected_name is None:
                return
            selected_region = RegionRef(
                selected_region_kind,
                selected_name,
            )
        boundary_regions = self._supported_boundary_regions()
        capability_report = self._model_capability_report()
        dimensions = (
            capability_report.dofs_per_node
            if capability_report is not None
            and capability_report.dofs_per_node is not None
            else model.mesh.dofs_per_node
            if model is not None
            else geometry_dimension(self.document.geometry_recipe)
        )
        available_scope_kinds = self._available_scope_selection_kinds(
            capability_report
        )
        dialog = DisplacementDialog(
            list(self.session.runnable_step_names()),
            boundary_regions,
            dimensions,
            self,
            selected_region=selected_region,
            labels=(
                capability_report.dof_labels
                if capability_report is not None
                else ()
            ),
            scope_selection_kinds=tuple(
                kind
                for kind, mesh_kind in (
                    ("node", "node"),
                    ("edge", "edge"),
                    ("surface", "face"),
                )
                if mesh_kind in available_scope_kinds
            ),
            form_state=dialog_state,
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
                    dialog_state=(
                        dialog.form_state()
                        if hasattr(dialog, "form_state")
                        else None
                    ),
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

    def create_load(
        self,
        selected_region: RegionRef | None = None,
        *,
        dialog_state: LoadDialogState | None = None,
    ) -> None:
        model = self.document.model
        if not self.session.runnable_step_names():
            return
        if self.document.source_kind == "native" and model is None:
            return
        preferred_kind = None
        capability_report = self._model_capability_report()
        if capability_report is None:
            return
        selected_mesh_kind = self._mesh_scope_selection_kind()
        if (
            selected_region is None
            and selected_mesh_kind in {"node", "edge", "face", "element"}
        ):
            preferred_kind = {
                "node": "node",
                "edge": "edge",
                "face": "surface",
                "element": "line",
            }[selected_mesh_kind]
            if (
                selected_mesh_kind == "element"
                and "line" not in capability_report.load_kinds
                and "body" in capability_report.load_kinds
            ):
                preferred_kind = "body"
            if preferred_kind not in capability_report.load_kinds:
                self._show_error(
                    "创建载荷",
                    "所选区域不支持当前模型的分布载荷契约。",
                )
                return
            selected_region_kind = {
                "node": "node_set",
                "edge": "edge",
                "surface": "surface",
                "line": "element_set",
                "body": "element_set",
            }[preferred_kind]
            selected_name = self._create_region_from_current_mesh_selection(
                on_committed=lambda name: self.create_load(
                    RegionRef(selected_region_kind, name)
                ),
            )
            if selected_name is None:
                return
            selected_region = RegionRef(
                selected_region_kind,
                selected_name,
            )
        (
            node_regions,
            edge_regions,
            face_regions,
            line_regions,
            body_regions,
        ) = (
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
            body_regions=body_regions,
            selected_region=selected_region,
            preferred_kind=preferred_kind,
            labels=capability_report.force_labels,
            candidate_evaluator=self._evaluate_line_load_candidate,
            scope_selection_kinds=(
                tuple(
                    kind
                    for kind in ("node", "edge", "surface", "line", "body")
                    if kind in capability_report.load_kinds
                )
                if self.document.model is not None
                else ()
            ),
            form_state=dialog_state,
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
                    dialog_state=(
                        dialog.form_state()
                        if hasattr(dialog, "form_state")
                        else None
                    ),
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
                    "创建边力",
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
        elif isinstance(load, BodyForce):
            step.body_loads = tuple(step.body_loads) + (load,)
        elif isinstance(load, GravityLoad):
            step.gravity_loads = tuple(step.gravity_loads) + (load,)
        self._analysis_definitions_changed(
            "载荷已修改，模型需要重新检查",
            definitions,
        )

    def create_output_request(self) -> None:
        authoring = self._session_authoring_projection()
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
        existing_projections = _output_request_projections_by_step(
            tuple(self.document.steps),
            catalog,
        )
        dialog = OutputRequestDialog(
            step_names,
            self,
            candidates=candidates,
            existing_requests_by_step={
                step_name: _executable_output_requests(
                    existing_projections[step_name]
                )
                for step_name in step_names
            },
        )
        if not self._exec_dialog(dialog):
            return
        try:
            definitions_method = getattr(dialog, "definitions", None)
            if callable(definitions_method):
                step_name, requests = definitions_method()
            else:
                step_name, request = dialog.definition()
                requests = (request,)
        except (TypeError, ValueError) as error:
            self._show_error("创建输出请求", str(error))
            return
        if (
            type(requests) is not tuple
            or not requests
            or any(type(request) is not OutputRequest for request in requests)
        ):
            self._show_error(
                "创建输出请求",
                "输出请求候选必须生成非空的 typed OutputRequest 元组。",
            )
            return
        try:
            requests = _with_required_displacement_output(
                requests,
                candidates,
            )
        except ValueError as error:
            self._show_error("输出请求", str(error))
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
        updated_outputs = _replace_visible_output_requests(
            requests,
            tuple(target.outputs),
        )
        if updated_outputs == tuple(target.outputs):
            self.status_panel.set_state(
                "输出请求未更改",
                5000,
            )
            return
        target.outputs = updated_outputs
        self._warn_imported_output_overlay()
        self._analysis_definitions_changed(
            "输出请求已更新，模型需要重新检查",
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
        steps: Sequence[AnalysisStep] | None = None,
    ) -> AnalysisDefinitionManagerDialog | None:
        source_steps = (
            list(self.document.steps)
            if steps is None
            else list(steps)
        )
        if not source_steps:
            return None
        model = self.document.model
        authoring = self._session_authoring_projection()
        capability_report = self._model_capability_report(authoring)
        (
            node_regions,
            edge_regions,
            face_regions,
            line_regions,
            body_regions,
        ) = (
            self._supported_load_regions(authoring)
        )
        available_scope_kinds = self._available_scope_selection_kinds(
            capability_report
        )
        dimensions = (
            capability_report.dofs_per_node
            if capability_report is not None
            and capability_report.dofs_per_node is not None
            else 3
        )
        return AnalysisDefinitionManagerDialog(
            source_steps,
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
            body_regions=body_regions,
            boundary_regions=self._supported_boundary_regions(authoring),
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
            boundary_scope_selection_kinds=tuple(
                kind
                for kind, mesh_kind in (
                    ("node", "node"),
                    ("edge", "edge"),
                    ("surface", "face"),
                )
                if mesh_kind in available_scope_kinds
            ),
            load_scope_selection_kinds=(
                tuple(
                    kind
                    for kind in (
                        "node",
                        "edge",
                        "surface",
                        "line",
                        "body",
                    )
                    if capability_report is not None
                    and kind in capability_report.load_kinds
                )
                if model is not None
                else ()
            ),
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
            self._begin_analysis_definition_scope_selection(dialog)
            return
        values = dialog.values()
        current = tuple(self.document.steps)
        if tuple(values) == current:
            return
        if self._output_collections_changed(current, values):
            capability = self._session_authoring_projection().operation(
                "output_request.delete"
            )
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
            "body_load": "body_load",
            "gravity_load": "gravity_load",
            "output": "output",
        }.get(kind)
        if manager_kind is None:
            self.show_entity_information(kind, key)
            return
        if kind == "step":
            definition_key = (manager_kind, int(key), None)
        else:
            collection_name = {
                "boundary": "boundaries",
                "cload": "cloads",
                "edge_load": "edge_loads",
                "surface_load": "surface_loads",
                "line_load": "line_loads",
                "body_load": "body_loads",
                "gravity_load": "gravity_loads",
                "output": "outputs",
            }[kind]
            resolved = _resolve_analysis_object_key(
                tuple(self.document.steps),
                collection_name,
                key,
            )
            if resolved is None:
                return
            step_index, item_index = resolved
            definition_key = (
                manager_kind,
                step_index,
                item_index,
            )
        self._edit_analysis_definition_key(definition_key)

    def _edit_analysis_definition_key(
        self,
        definition_key: tuple[str, int, int | None],
        *,
        selected_region: RegionRef | None = None,
        steps: Sequence[AnalysisStep] | None = None,
        dialog_state: (
            DisplacementDialogState | LoadDialogState | None
        ) = None,
    ) -> None:
        """Edit one manager definition and resume guided scope creation."""

        dialog = self._analysis_manager_dialog(steps)
        if dialog is None:
            return
        edit_kwargs = {"selected_region": selected_region}
        if dialog_state is not None:
            edit_kwargs["dialog_state"] = dialog_state
        if not dialog.edit_definition(definition_key, **edit_kwargs):
            self._begin_analysis_definition_scope_selection(dialog)
            return
        self._analysis_definitions_changed(
            "分析步、边界、载荷或输出请求已修改，模型需要重新检查",
            dialog.values(),
        )

    def _begin_analysis_definition_scope_selection(
        self,
        dialog: AnalysisDefinitionManagerDialog,
    ) -> bool:
        request = dialog.requested_scope_selection()
        if request is None:
            return False
        requested_kind, definition_key = request
        operation = (
            "boundary"
            if definition_key[0] == "boundary"
            else "load"
        )
        self._request_analysis_geometry_selection(
            operation,
            requested_kind,
            resume_edit=(
                definition_key,
                requested_kind,
                tuple(dialog.values()),
            ),
            dialog_state=dialog.requested_scope_dialog_state(),
        )
        return self._pending_analysis_selection is not None

    def delete_analysis_definition(self, kind: str, key: object) -> None:
        """Delete one supported definition selected in the model tree."""
        collection_name = {
            "boundary": "boundaries",
            "cload": "cloads",
            "edge_load": "edge_loads",
            "surface_load": "surface_loads",
            "line_load": "line_loads",
            "body_load": "body_loads",
            "gravity_load": "gravity_loads",
        }.get(kind)
        if collection_name is None:
            return
        resolved = _resolve_analysis_object_key(
            tuple(self.document.steps),
            collection_name,
            key,
        )
        if resolved is None:
            return
        step_index, item_index = resolved
        definitions = list(deepcopy(self.document.steps))
        if not 0 <= step_index < len(definitions):
            return
        step = definitions[step_index]
        collection = tuple(getattr(step, collection_name))
        if not 0 <= item_index < len(collection):
            return
        setattr(
            step,
            collection_name,
            collection[:item_index] + collection[item_index + 1:],
        )
        self._analysis_definitions_changed(
            (
                "边界条件已删除，模型需要重新检查"
                if kind == "boundary"
                else "载荷已删除，模型需要重新检查"
            ),
            definitions,
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
        executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="fem-preflight",
        )
        future = executor.submit(
            self._evaluate_model_check,
            task.model,
            task.step_name,
            task.token,
        )
        timed_out = False
        try:
            evaluation = future.result(_SYNCHRONOUS_GUI_COMMAND_TIMEOUT_SECONDS)
        except TimeoutError:
            timed_out = True
            future.cancel()
            self.status_panel.set_state("模型检查超时", 5000)
            return False
        finally:
            executor.shutdown(
                wait=not timed_out,
                cancel_futures=timed_out,
            )
        return self._complete_model_check(
            task.token,
            evaluation,
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
        expected_session_revision: int | None = None,
        on_progress: Callable[[str], None] | None = None,
    ) -> bool:
        task = self._prepare_model_check(
            step_name,
            expected_session_revision=expected_session_revision,
        )
        if task is None:
            raise RuntimeError(f"step cannot be checked: {step_name}")
        full_numerical_check = should_run_numerical_model_check(task.model)

        def workload(context: TaskContext):
            context.report(
                "正在检查模型……"
                if full_numerical_check
                else "正在执行大模型快速检查……"
            )
            result = self._evaluate_model_check(
                task.model,
                task.step_name,
                task.token,
            )
            context.checkpoint()
            return result

        def succeeded(value: object) -> None:
            delta, evaluation = value
            self._project_model_check(
                delta,
                evaluation.report,
                show_success=show_success,
            )

        def apply_result(value: object) -> TaskApplyOutcome:
            if type(value) is not PreparedPreflight:
                raise TypeError("model check must return PreparedPreflight")
            return self._session_task_outcome(
                self.session.accept_validation_with_prepared_system(
                    task.token,
                    value.report,
                    value.prepared_system,
                ),
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

        def inactive_failed(message: str) -> None:
            self._apply_session_delta(
                self.session.accept_task_failed(task.token, message)
            )

        def inactive_cancelled() -> None:
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
            on_inactive_failure=inactive_failed,
            on_inactive_cancelled=inactive_cancelled,
            apply_result=apply_result,
            completion=completion,
            on_progress=on_progress,
        )

    def _prepare_model_check(
        self,
        step_name: str | None = None,
        *,
        expected_session_revision: int | None = None,
    ) -> object | None:
        step_name = self._current_step_name if step_name is None else step_name
        if step_name is None or not self.session.can_check(step_name):
            return None
        options: dict[str, object] = {"detach_model": False}
        if expected_session_revision is not None:
            options["expected_session_revision"] = expected_session_revision
        return self.session.prepare_validation(step_name, **options)

    def _begin_agent_preflight(
        self,
        request: AgentPreflightTaskRequest,
    ) -> bool:
        """Run A6 preflight automatically through the existing GUI task."""

        if type(request) is not AgentPreflightTaskRequest:
            raise TypeError("request must be exactly AgentPreflightTaskRequest")
        if self.busy:
            return False
        port = self.agent_authoring_bridge.port
        complete_preflight = getattr(port, "complete_preflight", None)
        if not callable(complete_preflight):
            raise RuntimeError("Agent preflight lifecycle port is unavailable")
        completion = GuiCommandCompletion(self._next_command_id())

        def terminal(record: TaskCompletion) -> None:
            if record.state is BackgroundTaskState.SUCCEEDED:
                validation = self.session.validation_for(request.step_name)
                if validation is None:
                    completed = complete_preflight(
                        request.request_id,
                        AgentPreflightState.STALE,
                        "预检结果未绑定当前模型",
                    )
                elif validation.passed:
                    completed = complete_preflight(
                        request.request_id,
                        AgentPreflightState.PASSED,
                        "确定性模型预检通过",
                    )
                else:
                    completed = complete_preflight(
                        request.request_id,
                        AgentPreflightState.BLOCKED,
                        (
                            f"存在 {len(validation.report.errors)} "
                            "项阻塞诊断"
                        ),
                    )
                self._record_agent_workflow_preflight_state(
                    completed.state.value,
                    completed.message,
                )
                return
            state = {
                BackgroundTaskState.CANCELLED: AgentPreflightState.CANCELLED,
                BackgroundTaskState.DISCARDED: AgentPreflightState.STALE,
            }.get(record.state, AgentPreflightState.FAILED)
            completed = complete_preflight(
                request.request_id,
                state,
                record.message or record.state.value,
            )
            self._record_agent_workflow_preflight_state(
                completed.state.value,
                completed.message,
            )

        completion.observe(terminal)
        return self._begin_model_check(
            request.step_name,
            completion=completion,
            show_success=False,
            expected_session_revision=request.base_session_revision,
        )

    @staticmethod
    def _evaluate_model_check(
        model: object,
        step_name: str,
        token: object | None = None,
    ) -> PreparedPreflight:
        full_numerical_check = should_run_numerical_model_check(model)
        options = {
            "token": token,
            "check_numerical_stability": full_numerical_check,
            "copy_model": full_numerical_check,
            "quick_check": not full_numerical_check,
        }
        if full_numerical_check:
            return safe_prepare_static_preflight(
                model,
                step_name,
                **options,
            )
        return PreparedPreflight(
            safe_static_preflight(
                model,
                step_name,
                **options,
            )
        )

    def _complete_model_check(
        self,
        token: object,
        evaluation: object,
        *,
        show_success: bool,
    ) -> bool:
        if type(evaluation) is not PreparedPreflight:
            raise TypeError("model check must return PreparedPreflight")
        report = evaluation.report
        delta = self.session.accept_validation_with_prepared_system(
            token,
            report,
            evaluation.prepared_system,
        )
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
                "模型已变化，旧检查已忽略",
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
        stiffness_skipped = any(
            item.code == "static.stiffness.skipped_large_model"
            for item in report.warnings
        )
        warning_messages = tuple(dict.fromkeys(
            self._render_model_check_warning(item)
            for item in report.warnings
        ))
        self._show_information("模型检查", [
            ("分析类型", facts.procedure or "线性静力"),
            ("节点数", facts.node_count),
            ("单元数", facts.element_count),
            ("总自由度数", facts.dof_count),
            (
                "数值稳定性",
                (
                    "已检查"
                    if report.numerical_stability_checked
                    else ("已跳过" if stiffness_skipped else "未执行")
                ),
            ),
            ("警告/限制", "\n".join(warning_messages) or "无"),
            ("检查结果", "通过"),
        ])

    @staticmethod
    def _render_model_check_warning(item: PreflightDiagnostic) -> str:
        details = dict(getattr(item, "details", ()))
        request_index = details.get("request_index")
        request_name = details.get("request_name")
        if type(request_index) is int:
            request_label = f"第 {request_index + 1} 条输出请求"
        else:
            request_label = "当前输出请求"
        if request_name:
            request_label += f"“{request_name}”"

        def values_text(value: object) -> str:
            if isinstance(value, (tuple, list)):
                return "、".join(str(part) for part in value)
            return "" if value is None else str(value)

        request_variables = values_text(details.get("request_variables"))
        variables_context = (
            f"（变量：{request_variables}）" if request_variables else ""
        )
        code = item.code
        if code == "output.request.kind_unsupported":
            kind = details.get("kind", details.get("request_kind", ""))
            return (
                f"{request_label}{variables_context}："
                f"类型“{kind}”暂不支持执行"
            )
        if code == "output.request.target_unsupported":
            target = details.get("target", details.get("request_target", ""))
            return (
                f"{request_label}{variables_context}："
                f"目标“{target}”暂不支持执行"
            )
        if code == "output.request.variables_empty":
            return f"{request_label}：未指定结果变量"
        if code == "output.request.variable_unsupported":
            variables = values_text(
                details.get("source_variables")
                or details.get("canonical_variable")
                or details.get("request_variables")
            )
            return f"变量 {variables} 暂不支持执行"
        if code == "output.request.model_family_unsupported":
            family = details.get("model_family", "")
            return (
                f"{request_label}{variables_context}："
                f"模型类型“{family}”暂不支持这些变量"
            )
        if code == "output.request.position_unsupported":
            position = details.get("position", "")
            return (
                f"{request_label}{variables_context}："
                f"结果位置“{position}”暂不支持"
            )
        if code == "output.request.metadata_unsupported":
            settings = values_text(
                details.get("options")
                or details.get("source_keys")
                or details.get("flags")
            )
            setting_text = f"设置“{settings}”" if settings else "部分设置"
            return (
                f"{request_label}{variables_context}："
                f"{setting_text}暂不支持"
            )
        if code == "output.request.frequency_unsupported":
            frequency = details.get("frequency", "")
            return (
                f"{request_label}{variables_context}："
                f"输出频率“{frequency}”暂不支持"
            )
        return str(item.message).rstrip("；;。.")

    def create_job(self) -> None:
        """Create a pending job; submission is owned by the job manager."""

        if self.document.model is None or self.geometry is None or self.busy:
            return
        dialog = JobSubmitDialog(
            self.workspace.next_job_name(),
            self.session.runnable_step_names(),
            self._current_step_name,
            self,
        )
        if self._exec_dialog(dialog):
            receipt = self.create_run(dialog.job_name, dialog.step_name)
            if receipt.diagnostic is not None:
                self._show_command_rejection("创建作业失败", receipt)

    def create_and_submit_job(self) -> None:
        """Compatibility entry point following the create-only GUI workflow."""

        self.create_job()

    def resubmit_job(self, source_name: str | None = None) -> None:
        """以历史作业的分析步设置创建一个新的待提交作业。"""
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
            self.workspace.next_job_name(),
            self.session.runnable_step_names(),
            source.step_name,
            self,
        )
        if self._exec_dialog(dialog):
            receipt = self.create_run(
                dialog.job_name,
                dialog.step_name,
            )
            if receipt.diagnostic is not None:
                self._show_command_rejection("复制作业失败", receipt)

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
            self._show_command_rejection("提交作业失败", receipt)
            return None
        return self.session.find_run(str(name).strip())

    def _begin_submit_run(
        self,
        name: str,
        step_name: str,
        *,
        completion: GuiCommandCompletion | None = None,
        expected_session_revision: int | None = None,
        on_progress: Callable[[str], None] | None = None,
        existing_run: bool = False,
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
        if not existing_run and self.workspace.job_name_exists(clean_name):
            raise ValueError(f"作业名称已存在：{clean_name}")
        if clean_step not in self.session.runnable_step_names():
            raise ValueError(f"分析步不存在：{clean_step}")
        task = (
            self.session.prepare_run_solve(
                clean_name,
                expected_session_revision=expected_session_revision,
            )
            if existing_run
            else self.session.prepare_solve(
                clean_step,
                clean_name,
                expected_session_revision=expected_session_revision,
            )
        )
        if not existing_run:
            self.workspace.remember_job_name(clean_name)
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
        ) -> tuple[object, dict[str, float], object, object | None]:
            timings: dict[str, float] = {}
            solve_model = task.model
            stage["name"] = "求解"
            context.report("正在装配并求解……")
            run_prepared = task.prepared_system
            if run_prepared is None:
                run_prepared = static_linear.prepare(
                    solve_model,
                    copy_model=False,
                    timings=timings,
                )
            result = static_linear.solve(
                solve_model,
                task.step_name,
                name=task.run_name,
                _prepared_system=run_prepared,
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
            cache_candidate = (
                run_prepared.clone()
                if task.prepared_system is None
                else None
            )
            context.checkpoint()
            return bundle, timings, run_prepared, cache_candidate

        def apply_result(value: object) -> TaskApplyOutcome:
            bundle, timings, run_prepared, cache_candidate = value
            delta = self.session.accept_run_succeeded_with_prepared_system(
                task.token,
                bundle,
                run_prepared,
                cache_candidate=cache_candidate,
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
            on_inactive_failure=(
                lambda message, token=task.token, current_stage=stage: self._job_failed(
                    token,
                    message,
                    validation_failure=current_stage["name"] == "模型验证",
                )
            ),
            on_inactive_cancelled=(
                lambda token=task.token: self._job_cancelled(token)
            ),
            apply_result=apply_result,
            completion=completion,
            on_progress=on_progress,
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

    def _begin_agent_solve(
        self,
        request: AgentSolveTaskRequest,
    ) -> bool:
        """Submit one A6 proposal through the existing GUI job lifecycle."""

        if type(request) is not AgentSolveTaskRequest:
            raise TypeError("request must be exactly AgentSolveTaskRequest")
        port = self.agent_authoring_bridge.port
        progress_solve = getattr(port, "progress_solve", None)
        complete_solve = getattr(port, "complete_solve", None)
        if not callable(progress_solve) or not callable(complete_solve):
            raise RuntimeError("Agent solve lifecycle port is unavailable")
        completion = GuiCommandCompletion(self._next_command_id())

        def terminal(record: TaskCompletion) -> None:
            if record.state is BackgroundTaskState.SUCCEEDED:
                run = self.session.find_run(request.job_name)
                provenance = (
                    None
                    if run is None
                    else self.session.result_provenance_for(run.run_id)
                )
                if (
                    run is None
                    or not run.has_result
                    or run.artifact_id != request.artifact_id
                    or run.model_revision != request.model_revision
                    or run.step_name != request.step_name
                    or provenance is None
                    or provenance.session_id != self.document.session_id
                    or provenance.artifact_id != request.artifact_id
                    or provenance.model_revision != request.model_revision
                    or provenance.step_name != request.step_name
                    or provenance.run_id != run.run_id
                ):
                    complete_solve(
                        request.proposal_id,
                        ProposalState.FAILED,
                        "求解成功终态缺少精确 artifact/run/model provenance",
                    )
                    self._record_agent_workflow_proposal_state(
                        "solve",
                        ProposalState.FAILED,
                        "求解成功终态缺少精确 provenance",
                    )
                    return
                completed = complete_solve(
                    request.proposal_id,
                    ProposalState.SUCCEEDED,
                    (
                        f"求解完成：artifact {request.artifact_id} · "
                        f"run {run.run_id} · model revision "
                        f"{request.model_revision}"
                    ),
                )
                self._record_agent_workflow_proposal_state(
                    "solve",
                    completed.state,
                    completed.message,
                )
                return
            state = {
                BackgroundTaskState.CANCELLED: ProposalState.CANCELLED,
                BackgroundTaskState.DISCARDED: ProposalState.STALE,
            }.get(record.state, ProposalState.FAILED)
            completed = complete_solve(
                request.proposal_id,
                state,
                record.message or record.state.value,
            )
            self._record_agent_workflow_proposal_state(
                "solve",
                completed.state,
                completed.message,
            )

        completion.observe(terminal)
        job = self._begin_submit_run(
            request.job_name,
            request.step_name,
            completion=completion,
            expected_session_revision=request.base_session_revision,
            on_progress=lambda message: progress_solve(
                request.proposal_id,
                message,
            ),
        )
        return job is not None

    def _job_succeeded(self, token: object, value: object) -> None:
        bundle, timings = value
        delta = self.session.accept_run_succeeded(
            token,
            bundle,
            timings=timings,
        )
        if not self._apply_session_delta(delta):
            self.status_panel.set_state(
                "求解结果过期，未应用",
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
        target_context = self._task_context_or_active()
        self._apply_session_delta(
            self.session.accept_run_failed(token, message)
        )
        job = self.session.find_run(token.run_id)
        if job is None:
            return
        if target_context is not None and not self._task_context_is_active(
            target_context
        ):
            return
        self._refresh_job_manager()
        state = "模型检查失败" if validation_failure else "分析失败"
        self.status_panel.set_state(f"{state}：{job.name}", 5000)
        self._show_error(
            "模型检查失败" if validation_failure else "分析运行失败",
            message,
        )

    def _job_cancelled(self, token: object) -> None:
        target_context = self._task_context_or_active()
        self._apply_session_delta(
            self.session.accept_run_cancelled(token)
        )
        job = self.session.find_run(token.run_id)
        if job is None:
            return
        if target_context is not None and not self._task_context_is_active(
            target_context
        ):
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
        selection_delta = None
        if self.document.displayed_result_run_id != job.run_id:
            projection = self.session.prepare_result_projection(job.run_id)
            if not self._apply_revision_neutral_task_receipt(
                self.session.accept_result_projection(
                    projection.token
                )
            ):
                return
            selection_delta = self.session.select_result(job.run_id)
        self._set_current_step(job.step_name, refresh_viewport=False)
        self._display = DisplayState("deformed", True)
        self.actions["deformed"].setChecked(True)
        self.actions["contour"].setChecked(True)
        self.actions["symbols"].setChecked(False)
        self.viewport.set_symbols_visible(False, render=False)
        self.actions["node_labels"].setChecked(False)
        self.actions["element_labels"].setChecked(False)
        self.viewport.set_labels_visible(False, False, render=False)
        self.viewport.hide_selection_highlight(render=False)
        if selection_delta is not None:
            self._apply_session_delta(selection_delta)
        provider = self._current_result_provider()
        selection = self.result_selection
        if (
            provider is None
            or provider.source.run_id != job.run_id
            or type(selection) is not ScalarFieldSelection
        ):
            return
        if not self._viewport_result_scene_is_current(provider, selection):
            self._apply_display()
        context = self.workspace.active_document()
        if context is not None:
            self._refresh_result_tree_for_context(
                context,
                catalog=provider.catalog(),
            )
        if not self.result_tree.select_selection(
            selection,
            document_id=context.document_id if context is not None else None,
            source=provider.source,
        ):
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
            dialog.submitRequested.connect(self.submit_job)
            dialog.terminateRequested.connect(self.terminate_job)
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

    def submit_job(self, name: str) -> bool:
        """Submit the pending job selected in the job manager."""

        receipt = self.submit_created_run(name)
        if receipt.diagnostic is not None:
            self._show_command_rejection("提交作业失败", receipt)
            self._refresh_job_manager()
            return False
        return True

    def terminate_job(self, name: str) -> bool:
        """请求终止作业管理窗口中选定的当前求解。"""
        job = self.session.find_run(name)
        if (
            job is None
            or job.status is not RunStatus.RUNNING
            or job.cancellation_requested
            or self.document.active_job_name != job.name
            or self.task_controller.current_task_name != f"作业 {job.name}"
        ):
            return False
        requested = self.cancel_current_task()
        self._refresh_job_manager()
        if requested:
            self.status_panel.set_state(f"正在终止求解：{job.name}")
        return requested

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
        target = self._task_context_or_active()
        if applied and (target is None or self._task_context_is_active(target)):
            self._show_error(title, message)

    def _session_task_cancelled(self, token: object) -> None:
        delta = self.session.accept_task_cancelled(token)
        if not delta.changed and not delta.invalidated:
            self._apply_revision_neutral_task_receipt(delta)
            return
        self._apply_session_delta(delta)

    def _target_session_task_failed(
        self,
        context: WorkspaceDocument,
        token: object,
        title: str,
        message: str,
    ) -> None:
        delta = context.session.accept_task_failed(token, message)
        applied = (
            self._apply_revision_neutral_task_receipt(delta)
            if not delta.changed and not delta.invalidated
            else self._apply_session_delta(delta, context=context)
        )
        if applied and self.workspace.active_document_id == context.document_id:
            self._show_error(title, message)

    def _target_session_task_cancelled(
        self,
        context: WorkspaceDocument,
        token: object,
    ) -> None:
        delta = context.session.accept_task_cancelled(token)
        if not delta.changed and not delta.invalidated:
            self._apply_revision_neutral_task_receipt(delta)
            return
        self._apply_session_delta(delta, context=context)

    def _task_target_is_live(self, context: WorkspaceDocument) -> bool:
        try:
            registered = self.workspace.document(context.document_id)
        except (KeyError, TypeError, ValueError):
            return False
        return registered is context and registered.session is context.session

    @staticmethod
    def _session_delta_from_task_value(value: object) -> SessionDelta | None:
        if type(value) is SessionDelta:
            return value
        if isinstance(value, (tuple, list)):
            for item in value:
                found = FEMMainWindow._session_delta_from_task_value(item)
                if found is not None:
                    return found
        if isinstance(value, dict):
            for item in value.values():
                found = FEMMainWindow._session_delta_from_task_value(item)
                if found is not None:
                    return found
        return None

    def _apply_inactive_task_projection(
        self,
        value: object,
        context: WorkspaceDocument,
    ) -> None:
        """Apply only the target projection/tree portion of a late task."""

        delta = self._session_delta_from_task_value(value)
        if delta is None or not self._task_target_is_live(context):
            return
        previous = self._task_callback_context
        self._task_callback_context = context
        try:
            self._apply_session_delta(delta, context=context)
        finally:
            self._task_callback_context = previous

    def _invoke_task_callback(
        self,
        context: WorkspaceDocument | None,
        callback: Callable[..., object],
        *args: object,
    ) -> object:
        previous = self._task_callback_context
        self._task_callback_context = context
        try:
            return callback(*args)
        finally:
            self._task_callback_context = previous

    def _start_task(
        self,
        workload: Callable[[TaskContext], object],
        on_success: Callable[[object], None],
        error_title: str,
        on_failure: Callable[[str], None] | None = None,
        *,
        task_name: str = "后台任务",
        on_cancelled: Callable[[], None] | None = None,
        on_inactive_failure: Callable[[str], None] | None = None,
        on_inactive_cancelled: Callable[[], None] | None = None,
        apply_result: Callable[[object], TaskApplyOutcome] | None = None,
        completion: GuiCommandCompletion | None = None,
        on_progress: Callable[[str], None] | None = None,
        controller: BackgroundTaskController | None = None,
        context: WorkspaceDocument | None = None,
    ) -> bool:
        target_context = context
        if target_context is None and controller is None:
            target_context = self._active_workspace_context()
        if target_context is not None:
            self._bind_task_controller(target_context)
        task_controller = (
            controller
            or (
                target_context.task_controller
                if target_context is not None
                else self.task_controller
            )
        )
        if task_controller.busy:
            if target_context is None or self._task_context_is_active(target_context):
                self.status_panel.set_state(
                    f"任务中：{task_controller.current_task_name or '后台任务'}",
                    4000,
                )
            return False
        result_applier = apply_result or TaskApplyOutcome.accepted

        def apply(value: object) -> TaskApplyOutcome:
            if target_context is not None and not self._task_target_is_live(
                target_context
            ):
                return TaskApplyOutcome.stale("目标文档已关闭")
            outcome = self._invoke_task_callback(
                target_context,
                result_applier,
                value,
            )
            if type(outcome) is not TaskApplyOutcome:
                raise TypeError("apply_result must return a TaskApplyOutcome")
            return outcome

        def project_terminal(record: TaskCompletion) -> None:
            active_target = (
                target_context is None
                or self._task_context_is_active(target_context)
            )
            if record.state is BackgroundTaskState.FAILED:
                message = record.message or "后台任务失败"
                try:
                    failure_callback = (
                        on_failure if active_target else on_inactive_failure
                    )
                    if failure_callback is None:
                        if active_target:
                            self._show_error(error_title, message)
                    else:
                        self._invoke_task_callback(
                            target_context,
                            failure_callback,
                            message,
                        )
                except Exception:
                    logging.exception(
                        "GUI background task failure callback failed"
                    )
                    if active_target:
                        self._show_error(error_title, message)
            elif record.state is BackgroundTaskState.CANCELLED:
                try:
                    cancelled_callback = (
                        on_cancelled if active_target else on_inactive_cancelled
                    )
                    if cancelled_callback is not None:
                        self._invoke_task_callback(
                            target_context,
                            cancelled_callback,
                        )
                except Exception as error:
                    logging.exception(
                        "GUI background task cancellation callback failed"
                    )
                    if active_target:
                        self._show_error(
                            error_title,
                            str(error).strip() or type(error).__name__,
                        )
                if active_target:
                    self.status_panel.set_state(
                        f"已取消：{record.task_name}",
                        4000,
                    )
            elif record.state is BackgroundTaskState.DISCARDED:
                if active_target:
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

        def progress(message: str) -> None:
            if target_context is None or self._task_context_is_active(target_context):
                self.status_panel.set_state(message)
                if on_progress is not None:
                    self._invoke_task_callback(target_context, on_progress, message)

        project_callback = (
            None
            if on_success is None
            else self._task_project_result_wrapper(target_context, on_success)
        )
        task_id = task_controller.start(
            workload,
            task_name=task_name,
            apply_result=apply,
            project_result=project_callback,
            rebuild_projection=self._task_rebuild_projection_wrapper(
                target_context,
            ),
            on_terminal=terminal,
            on_progress=progress,
            on_projection_error=self._task_projection_failed_wrapper(
                target_context,
            ),
        )
        if task_id is not None and completion is not None:
            completion.bind_task_id(task_id)
        return task_id is not None

    def _task_project_result_wrapper(
        self,
        context: WorkspaceDocument | None,
        callback: Callable[[object], None],
    ) -> Callable[[object], None]:
        def project(value: object) -> None:
            if context is not None and not self._task_target_is_live(context):
                return
            if context is not None and not self._task_context_is_active(context):
                self._apply_inactive_task_projection(value, context)
                return
            self._invoke_task_callback(context, callback, value)

        return project

    def _task_rebuild_projection_wrapper(
        self,
        context: WorkspaceDocument | None,
    ) -> Callable[[], None]:
        def rebuild() -> None:
            if context is not None and not self._task_context_is_active(context):
                return
            self._invoke_task_callback(context, self._rebuild_full_projection)

        return rebuild

    def _task_projection_failed_wrapper(
        self,
        context: WorkspaceDocument | None,
    ) -> Callable[[str], None]:
        def report(message: str) -> None:
            if context is not None and not self._task_context_is_active(context):
                return
            self._invoke_task_callback(context, self._task_projection_failed, message)

        return report

    def _task_busy_changed(self, busy: bool) -> None:
        self.status_panel.set_task_active(bool(busy))
        self._update_action_states()

    def _task_cancelling_changed(self, cancelling: bool) -> None:
        if cancelling:
            self.status_panel.set_task_active(True, cancelling=True)
            self.status_panel.set_state(
                f"取消中：{self.task_controller.current_task_name or '后台任务'}"
            )

    def _task_projection_failed(self, message: str) -> None:
        logging.error("GUI task projection failed: %s", message)
        self.status_panel.set_state("任务已接受，但界面刷新失败", 8000)

    def _rebuild_full_projection(self) -> None:
        snapshot = self.session.projection_snapshot()
        self._applied_session_revision = -1
        if not self._apply_session_delta(
            SessionDelta(
                session_revision=snapshot.session_revision,
                reason="full GUI projection rebuild",
            )
        ):
            raise RuntimeError("无法从最新 Session snapshot 重建界面")

    def _agent_geometry_edit_mode(self) -> str:
        context = self._active_workspace_context()
        if context is None:
            raise RuntimeError("没有活动模型可供几何修改")
        return geometry_edit_policy(context).value

    def _commit_agent_geometry_edit(
        self,
        mutation: AgentGeometryMutation,
        expected_session_revision: int,
    ) -> dict[str, object]:
        """Commit one validated geometry mutation through workspace policy."""

        source = self._active_workspace_context()
        if source is None or source.session is not self.session:
            raise RuntimeError("Agent 几何修改源模型已改变")
        mode = geometry_edit_policy(source)
        if mode.value == "in_place":
            mutation.apply(source.session, expected_session_revision)
            return {
                "mode": "in_place",
                "source": {
                    "document_id": source.document_id,
                    "session_id": source.session.session_id,
                },
                "target": {
                    "document_id": source.document_id,
                    "session_id": source.session.session_id,
                },
                "part_id": mutation.affected_part_ids[0],
                "affected_part_ids": list(mutation.affected_part_ids),
                "requires_remesh": True,
                "validations": "reset",
                "runs": "not_migrated",
                "results": "not_migrated",
            }
        result = ModelIterationService(self.workspace).branch_geometry_mutation(
            source.document_id,
            mutation.affected_part_ids,
            mutation.apply,
            source_run_id=self._agent_geometry_edit_source_run_id(source),
            expected_source_session_revision=expected_session_revision,
            activate_child=self._activate_workspace_context,
        )
        return {"mode": "branch", **result.report.to_dict()}

    def _agent_geometry_edit_source_run_id(
        self,
        source: WorkspaceDocument,
    ) -> str | None:
        """Choose deterministic accepted-result provenance for one branch."""

        snapshot = source.session.snapshot()
        candidates: list[str] = []
        if snapshot.displayed_result_run_id is not None:
            candidates.append(str(snapshot.displayed_result_run_id))
        if snapshot.selected_run_id is not None:
            candidates.append(str(snapshot.selected_run_id))
        provider = self.result_provider
        provider_source = None if provider is None else provider.source
        if (
            provider_source is not None
            and provider_source.session_id == source.session.session_id
        ):
            candidates.append(str(provider_source.run_id))
        candidates.extend(str(run.run_id) for run in reversed(snapshot.runs))
        seen: set[str] = set()
        for run_id in candidates:
            if run_id in seen:
                continue
            seen.add(run_id)
            identity = source.session.result_identity_for(run_id)
            if identity is not None and identity[0].run_id == run_id:
                return run_id
        return None

    def _apply_agent_definition_delta(self, delta: SessionDelta) -> None:
        """Project A4 scopes/definitions without rebuilding mesh actors."""

        self._apply_definition_only_delta(delta)

    def _apply_definition_only_delta(self, delta: SessionDelta) -> None:
        """Project changed scopes/definitions while retaining mesh topology caches."""

        if not delta.accepted:
            return
        revision = int(delta.session_revision)
        if revision <= self._applied_session_revision:
            return
        snapshot = self.session.projection_snapshot(
            self.document,
            delta.changed,
        )
        if snapshot.session_revision != revision:
            raise RuntimeError("Agent definition delta is not the current revision")
        artifact = snapshot.artifact
        if artifact is None:
            raise RuntimeError("Agent definitions require a current model artifact")

        previous_artifact = self.document.artifact
        geometry = self.geometry
        if (
            previous_artifact is None
            or geometry is None
            or self.viewport.artifact_id != previous_artifact.artifact_id
        ):
            raise RuntimeError(
                "definition-only projection requires the current mesh viewport"
            )

        self.document = snapshot
        bound_context = self._task_context_or_active()
        stale_agent_proposals = self.agent_authoring_bridge.bind_snapshot(
            snapshot,
            document_id=(
                None if bound_context is None else bound_context.document_id
            ),
        )
        current_authoring_context = self.agent_authoring_bridge.context
        if current_authoring_context is not None:
            self.agent_authoring_controller.observe_binding(
                current_authoring_context,
                proposal_staled=bool(stale_agent_proposals),
            )
        if hasattr(self, "viewport_panel"):
            self.viewport_panel.agent_chat_drawer.refresh_authoring_binding()
        self._close_inspection_windows()

        model = artifact.model

        def frame_query(target: RegionRef | int) -> BeamFrameReport:
            return resolve_effective_beam_frames(model, target)

        self.geometry = replace(geometry, artifact_id=artifact.artifact_id)
        self.viewport.rebind_model_artifact(
            model,
            self.geometry,
            effective_frame_query=frame_query,
        )

        self.inspection_service = InspectionService(
            model,
            definitions=snapshot,
            effective_frame_query=frame_query,
        )
        active_context = self.workspace.active_document()
        if active_context is not None:
            cache = active_context.presentation_cache
            cache.invalidate_model()
            cache.artifact_id = snapshot.artifact.artifact_id
            cache.model_geometry = self.geometry
            cache.inspection_service = self.inspection_service
        self._show_model_in_tree(model)
        had_result_projection = not (
            self.result_provider is None
            and self.viewport._result_render_payload is None
        )
        self._clear_result_projection()
        if not had_result_projection:
            self.viewport.show_boundary_and_loads()
        self._sync_step_combos()
        self._refresh_result_controls()
        self._refresh_job_manager()
        self._update_action_states()
        self._applied_session_revision = revision

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
        self._defer_ui(self._run_scheduled_viewport_fit)

    def _run_scheduled_viewport_fit(self) -> None:
        self._viewport_fit_pending = False
        if (
            self._pending_local_mesh_selection
            or self._pending_analysis_selection is not None
        ):
            return
        self.viewport.fit()

    def _exec_dialog(self, dialog: QDialog) -> int:
        """Execute one FEM dialog without changing the viewport framing."""

        return int(dialog.exec())

    def _exec_view_dialog(self, dialog: QDialog) -> int:
        """Execute a view-affecting dialog and restore full-model framing."""

        try:
            return self._exec_dialog(dialog)
        finally:
            self._schedule_viewport_fit()

    def _show_information(
        self,
        title: str,
        rows: Sequence[tuple[str, object]],
    ) -> None:
        show_information(self, title, rows)
        self._schedule_viewport_fit()

    def _show_save_success(self, content_name: str, path: str | Path) -> None:
        QMessageBox.information(
            self,
            "保存成功",
            f"{content_name}已保存成功\n\n{path}",
        )

    def _fit_viewport_when_dialog_finishes(self, dialog: QDialog) -> None:
        """Restore full-model framing after a view-affecting dialog closes."""

        dialog.finished.connect(self._fit_viewport_after_dialog)

    def _fit_viewport_after_dialog(self, _result: int = 0) -> None:
        self._schedule_viewport_fit()

    def _toggle_edges(self, checked: bool) -> None:
        if (
            self._current_module_name() == "结果"
            and self._display.contour_enabled
        ):
            self._contour_options["edges"] = bool(checked)
            if (
                checked
                and self._contour_options["edge_mode"]
                == CONTOUR_EDGE_NONE
            ):
                self._contour_options["edge_mode"] = CONTOUR_EDGE_ALL
        else:
            self._model_edges_visible = bool(checked)
        self.viewport.set_edges_visible(checked)

    def _toggle_suppressed_part_ghosts(self, checked: bool) -> None:
        self._show_suppressed_part_ghosts = bool(checked)
        parts = tuple(
            part
            for part in self.document.parts
            if part.suppressed and part.geometry_recipe is not None
        )
        preview = (
            build_multi_part_geometry_preview(
                parts,
                include_suppressed=True,
            )
            if self._show_suppressed_part_ghosts and parts
            else None
        )
        self.viewport.set_geometry_ghost_preview(preview)
        current = self.viewport._geometry_preview
        if current is not None:
            self.viewport.show_geometry_preview(current)

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

    def _selected_body_id(self, recipe: object) -> str | None:
        if not isinstance(recipe, MultiBodyGeometry):
            return None
        selected = self._canonical_geometry_selection()
        if len(selected) != 1 or selected[0].kind != "body":
            return None
        logical_id = selected[0].logical_id
        if logical_id == "body:domain":
            return None
        body_id = logical_id.split(":", 1)[1]
        try:
            recipe.body(body_id)
        except KeyError:
            return None
        return body_id

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
                key=mesh_entity_ref_sort_key,
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

    @staticmethod
    def _selection_action_name(selection_filter: str) -> str:
        return f"select_{selection_filter}"

    def _begin_temporary_selection_context(
        self,
        owner: str,
        space: str,
        selection_filter: str,
    ) -> None:
        """Override module selection semantics until one guided flow ends."""

        if self._temporary_selection_context is None:
            self._temporary_selection_context = replace(
                self._selection_context
            )
            self._temporary_selection_owner = str(owner)
        elif self._temporary_selection_owner != owner:
            raise RuntimeError("another temporary selection context is active")
        self._set_selection_space(space)
        self._set_selection_filter(selection_filter)

    def _restore_temporary_selection_context(
        self,
        owner: str | None = None,
    ) -> None:
        saved = self._temporary_selection_context
        if saved is None:
            return
        if owner is not None and self._temporary_selection_owner != owner:
            return
        self._temporary_selection_context = None
        self._temporary_selection_owner = None
        self.selection.clear()
        self._selected_geometry_refs.clear()
        self._selected_mesh_scope_refs.clear()
        self.viewport.clear_selection()
        module_name = ""
        ribbon = getattr(self, "ribbon", None)
        tab_bar = None if ribbon is None else ribbon.tab_bar
        if tab_bar is not None and tab_bar.currentIndex() >= 0:
            module_name = tab_bar.tabText(tab_bar.currentIndex())
        saved.set_space("geometry" if module_name == "几何" else "mesh")
        self._selection_context = saved
        self.viewport_panel.set_geometry_context(saved.space == "geometry")
        self._set_selection_filter(saved.active_filter, force=True)
        self.status_panel.set_object()
        self.actions["selected_info"].setEnabled(False)

    def _set_selection_space(self, space: str) -> None:
        normalized = "geometry" if space == "geometry" else "mesh"
        changed = self._selection_context.space != normalized
        selection_filter = self._selection_context.set_space(normalized)
        if changed:
            self.selection.clear()
            self._selected_geometry_refs.clear()
            self._selected_mesh_scope_refs.clear()
            self.viewport.clear_selection()
            self.status_panel.set_object()
            self.actions["selected_info"].setEnabled(False)
        self.viewport_panel.set_geometry_context(normalized == "geometry")
        self._set_selection_filter(selection_filter, force=True)

    def _set_selection_filter(
        self,
        selection_filter: str,
        *,
        force: bool = False,
    ) -> None:
        if selection_filter not in {
            "point", "element", "edge", "face", "body",
        }:
            raise ValueError("unsupported semantic selection filter")
        if self._selection_context.space == "geometry" and selection_filter == "element":
            self._sync_selection_action_state()
            return
        if selection_filter == "face":
            if (
                self._selection_context.space == "geometry"
                and isinstance(
                    self.document.geometry_recipe,
                    NATIVE_GEOMETRY_TYPES,
                )
                and geometry_dimension(self.document.geometry_recipe) == 1
            ):
                selection_filter = "body"
            elif self._selection_context.space == "mesh":
                report = self._model_capability_report()
                if report is not None and report.topological_dimension == 1:
                    selection_filter = "edge"
        changed = selection_filter != self._selection_context.active_filter
        if changed:
            self.selection.clear()
            self._selected_geometry_refs.clear()
            self._selected_mesh_scope_refs.clear()
            self.viewport.clear_selection()
            self.viewport_panel.scope_creation_bar.set_selection_ready(False)
            self.status_panel.set_object()
            self.actions["selected_info"].setEnabled(False)
        self._selection_context.set_filter(selection_filter)
        self.actions[self._selection_action_name(selection_filter)].setChecked(True)
        if self._selection_context.space == "geometry":
            self._set_geometry_selection_mode(selection_filter)
        else:
            self._set_mesh_scope_selection_mode(
                "node" if selection_filter == "point" else selection_filter
            )
        if changed or force:
            self._update_action_states()

    def _sync_selection_action_state(self) -> None:
        active_filter = self._selection_context.active_filter
        for selection_filter in ("point", "element", "edge", "face", "body"):
            action = self.actions[self._selection_action_name(selection_filter)]
            action.setChecked(selection_filter == active_filter)
        labels = (
            {
                "point": "选择点",
                "element": "选择单元",
                "edge": "选择边",
                "face": "选择面",
                "body": "选择体",
            }
            if self._selection_context.space == "geometry"
            else {
                "point": "选择点",
                "element": "选择单元",
                "edge": "选择边",
                "face": "选择面",
                "body": "选择体",
            }
        )
        for selection_filter, label in labels.items():
            action = self.actions[self._selection_action_name(selection_filter)]
            action.setToolTip(label)
            action.setStatusTip(label)

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
            if mode in {"node", "edge", "face", "element", "body"}
            else "node"
        )
        model = self._current_gui_model()
        reference_kind = normalized
        if model is not None:
            semantic_filter = "point" if normalized == "node" else normalized
            report = self._model_capability_report()
            dimension = (
                report.topological_dimension
                if report is not None
                else None
            )
            reference_kind = (
                "node"
                if semantic_filter == "point"
                else "element"
                if semantic_filter in {"element", "body"}
                else "element"
                if semantic_filter == "edge" and dimension == 1
                else "element"
                if semantic_filter == "face" and dimension == 2
                else semantic_filter
            )
        incompatible = (
            self._selected_mesh_scope_refs
            and self._mesh_scope_selection_kind() != reference_kind
        )
        if incompatible:
            self._selected_mesh_scope_refs.clear()
            self.viewport.clear_selection(render=False)
            self.status_panel.set_object()
        if (
            normalized in {"edge", "face", "body"}
            and model is not None
            and self._mesh_selection_topology_cache is None
            and self._mesh_selection_topology_requires_background()
        ):
            self.selection.clear()
            self._selected_geometry_refs.clear()
            self.actions["selected_info"].setEnabled(False)
            self.status_panel.set_selection_mode(f"mesh_{normalized}")
            self._begin_mesh_selection_topology_preparation(normalized)
            return
        if (
            normalized in {"edge", "face", "body"}
            and model is not None
            and self._mesh_selection_topology_cache is None
        ):
            self._mesh_selection_topology()
        if (
            self._selected_mesh_scope_refs
            and normalized in {"edge", "face", "body"}
            and model is not None
        ):
            topology = self._mesh_selection_topology()
            self._selected_mesh_scope_refs = {
                expanded
                for reference in self._selected_mesh_scope_refs
                for expanded in topology.expand(normalized, reference)
            }
        self.selection.clear()
        self._selected_geometry_refs.clear()
        self.actions["selected_info"].setEnabled(False)
        self.viewport.set_selection_mode(f"mesh_{normalized}")
        self.status_panel.set_selection_mode(f"mesh_{normalized}")
        if self._selected_mesh_scope_refs:
            self._refresh_mesh_scope_selection(normalized)

    def clear_selection(self) -> None:
        temporary_owner = self._temporary_selection_owner
        self.selection.clear()
        self._pending_local_mesh_selection = False
        self._pending_analysis_selection = None
        self._pending_analysis_requested_scope_kind = None
        self._pending_analysis_dialog_state = None
        self._pending_scope_kind = None
        self._pending_analysis_edit = None
        self.viewport_panel.scope_creation_bar.finish()
        if self._scope_selection_overlay_active:
            self.viewport.hide_geometry_selection_overlay()
            self._scope_selection_overlay_active = False
        self._selected_geometry_refs.clear()
        self._selected_mesh_scope_refs.clear()
        self.viewport.clear_selection()
        self.status_panel.set_object()
        self.actions["selected_info"].setEnabled(False)
        self._restore_temporary_selection_context(temporary_owner)
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
        if self._assign_solid_face_boolean_reference(reference):
            return
        if self._assign_planar_boolean_reference(reference):
            return
        if self._assign_body_boolean_reference(reference):
            return
        owner = part_id_from_logical_id(reference.logical_id)
        if (
            owner is not None
            and owner != self.document.active_part_id
            and any(part.id == owner for part in self.document.parts)
        ):
            try:
                self._apply_session_delta(
                    self.session.set_active_native_part(
                        owner,
                        expected_session_revision=(
                            self.document.session_revision
                        ),
                    )
                )
            except (RuntimeError, TypeError, ValueError, KeyError):
                return
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
        planar = self._planar_boolean_controller
        if planar is not None and planar.selecting_target:
            if len(selected) != 1 or selected[0].kind != "face":
                self.status_panel.set_state(
                    "二维布尔一次只能选择一个目标面",
                    4000,
                )
                return
            self._assign_planar_boolean_reference(selected[0])
            return
        if self._solid_face_boolean_operation is not None:
            if len(selected) != 1 or selected[0].kind != "face":
                self.status_panel.set_state(
                    "布尔操作一次只能选择一个目标面",
                    4000,
                )
                return
            self._assign_solid_face_boolean_reference(selected[0])
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
                f"{selected_count} 个"
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
        semantic_filter = self._active_mesh_selection_filter()
        expanded = self._expand_mesh_selection_reference(
            semantic_filter,
            reference,
        )
        if not expanded:
            self._on_viewport_pick_missed(f"mesh_{semantic_filter}")
            return
        self._apply_mesh_selection_groups(
            semantic_filter,
            (expanded,),
            additive=self._geometry_pick_is_additive(),
        )

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
        semantic_filter = self._active_mesh_selection_filter()
        groups = self._mesh_box_selection_groups(
            semantic_filter,
            selected,
        )
        if not groups:
            self._on_viewport_pick_missed(f"mesh_{semantic_filter}")
            return
        self._apply_mesh_selection_groups(
            semantic_filter,
            groups,
            additive=self._geometry_pick_is_additive(),
        )

    def _mesh_box_selection_groups(
        self,
        semantic_filter: str,
        references: tuple[MeshEntityRef, ...],
    ) -> tuple[tuple[MeshEntityRef, ...], ...]:
        """Expand box hits once per semantic mesh entity."""

        if (
            self._current_gui_model() is None
            or (
                self._pending_analysis_selection is not None
                and semantic_filter in {"point", "element"}
            )
            or semantic_filter in {"point", "element"}
        ):
            return tuple(
                expanded
                for reference in references
                if (
                    expanded := self._expand_mesh_selection_reference(
                        semantic_filter,
                        reference,
                    )
                )
            )
        topology = self._mesh_selection_topology()
        if semantic_filter == "body":
            owners = {
                topology.element_owners.get(int(reference.element_id))
                for reference in references
                if reference.element_id is not None
            }
            owners.discard(None)
            return tuple(
                topology.part_elements[owner]
                for owner in sorted(owners, key=part_id_sort_key)
            )
        expansions = (
            topology.edge_expansions
            if semantic_filter == "edge"
            else topology.face_expansions
            if semantic_filter == "face"
            else None
        )
        if expansions is None:
            raise ValueError("unsupported mesh selection filter")
        groups_by_identity: dict[int, tuple[MeshEntityRef, ...]] = {}
        for reference in references:
            group = expansions.get((reference.kind, reference.identity))
            if group:
                groups_by_identity.setdefault(id(group), group)
        return tuple(groups_by_identity.values())

    def _active_mesh_selection_filter(self) -> str:
        mode = self.viewport._selection_mode
        if mode.startswith("mesh_"):
            semantic_filter = mode.removeprefix("mesh_")
            return "point" if semantic_filter == "node" else semantic_filter
        return "point" if self.selection.mode == "node" else "element"

    def _expand_mesh_selection_reference(
        self,
        semantic_filter: str,
        reference: MeshEntityRef,
    ) -> tuple[MeshEntityRef, ...]:
        if self._current_gui_model() is None:
            return (reference,)
        if (
            self._pending_analysis_selection is not None
            and semantic_filter in {"point", "element"}
        ):
            return (reference,)
        if semantic_filter == "element" and reference.element_id is not None:
            owner = self.viewport._mesh_body_owner_by_element_id.get(
                int(reference.element_id)
            )
            if owner is not None:
                return (
                    MeshEntityRef.element(
                        int(reference.element_id),
                        part_id=owner,
                    ),
                )
        return self._mesh_selection_topology().expand(
            semantic_filter,
            reference,
        )

    def _apply_mesh_selection_groups(
        self,
        semantic_filter: str,
        groups: tuple[tuple[MeshEntityRef, ...], ...],
        *,
        additive: bool,
    ) -> None:
        unique_groups = tuple(group for group in groups if group)
        previous = set(self._selected_mesh_scope_refs)
        if additive:
            selected = set(previous)
            for group in sorted(
                unique_groups,
                key=lambda values: mesh_entity_ref_sort_key(values[0]),
            ):
                group_set = set(group)
                if group_set.issubset(selected):
                    selected.difference_update(group_set)
                else:
                    selected.update(group_set)
        else:
            selected = {
                reference
                for group in unique_groups
                for reference in group
            }
        self._selected_mesh_scope_refs = selected
        self._refresh_mesh_scope_selection(
            semantic_filter,
            changed_references=previous.symmetric_difference(selected),
        )

    def _refresh_mesh_scope_selection(
        self,
        kind: str,
        *,
        changed_references: set[MeshEntityRef] | None = None,
    ) -> None:
        references = self._selected_mesh_scope_refs
        reference_kind = (
            next(iter(references)).kind
            if references
            else (
                "node" if kind in {"node", "point"} else kind
            )
        )
        self.viewport_panel.scope_creation_bar.set_selection_ready(
            bool(references)
        )
        self.viewport.highlight_mesh_entities(
            references,
            changed_references=changed_references,
            entity_kind="body" if kind == "body" else reference_kind,
        )
        labels = {
            "point": "节点",
            "node": "节点",
            "edge": "拓扑边",
            "face": "拓扑面",
            "element": "单元",
            "body": "部件",
        }
        semantic_count = len(references)
        if (
            references
            and kind in {"edge", "face", "body"}
            and self._current_gui_model() is not None
        ):
            topology = self._mesh_selection_topology()
            if kind == "body":
                semantic_count = len(
                    {
                        topology.element_owners.get(
                            int(reference.element_id)
                        )
                        for reference in references
                        if reference.element_id is not None
                    }
                    - {None}
                )
            else:
                semantic_count = len(
                    {
                        id(group)
                        for reference in references
                        if (group := topology.expand(kind, reference))
                    }
                )
        status_kind = "node" if kind == "point" else kind
        self.status_panel.set_selection_mode(f"mesh_{status_kind}")
        self.status_panel.set_object(
            (
                f"{semantic_count} 个"
                f"{labels.get(kind, '网格实体')}"
            )
            if references
            else "—"
        )
        self.actions["selected_info"].setEnabled(False)

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
        if self._solid_face_boolean_operation is not None:
            self.viewport_panel.planar_boolean_face_bar.set_selection_ready(
                False
            )
            self.status_panel.set_state("请选择目标面", 0)
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
            self._defer_ui(self.set_local_mesh_control)
            return
        if self._pending_analysis_selection is not None:
            self._complete_scope_creation_from_bar()

    def _cancel_guided_selection(self) -> None:
        planar = self._planar_boolean_controller
        if planar is not None and planar.selecting_target:
            self.cancel_planar_boolean()
            return
        if self._solid_face_boolean_operation is not None:
            self._cancel_solid_face_boolean()
            return
        if not self._pending_local_mesh_selection and self._pending_analysis_selection is None:
            return
        self._pending_local_mesh_selection = False
        self._pending_analysis_selection = None
        self._pending_analysis_requested_scope_kind = None
        self._pending_analysis_dialog_state = None
        self._pending_scope_kind = None
        self._pending_analysis_edit = None
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

    def _prepare_tree_entry(
        self,
        document_id_or_kind: object,
        kind_or_key: object = None,
        key: object = _TREE_KEY_MISSING,
    ) -> tuple[int | None, str, object] | None:
        """Normalize legacy and document-routed tree callbacks.

        Phase 1 callers pass ``(kind, key)`` while the multi-document tree
        emits ``(document_id, kind, key)``.  Routed callbacks activate their
        target context before the existing handlers inspect ``self.document``.
        """

        if key is _TREE_KEY_MISSING:
            return None, str(document_id_or_kind), kind_or_key
        try:
            context = self.workspace.document(document_id_or_kind)
        except (KeyError, TypeError, ValueError):
            return None
        if not self._activate_workspace_context(context):
            return None
        return context.document_id, str(kind_or_key), key

    def _resolve_tree_analysis_key(
        self,
        kind: str,
        key: object,
    ) -> object | None:
        collection_name = _ANALYSIS_COLLECTION_BY_TREE_KIND.get(kind)
        if collection_name is None:
            return key
        return _resolve_analysis_object_key(
            tuple(self.document.steps),
            collection_name,
            key,
        )

    def _model_root_action_requested(self, document_id: int, action: str) -> None:
        """Handle model-root lifecycle actions after routing by document id."""

        try:
            context = self.workspace.document(int(document_id))
        except (KeyError, TypeError, ValueError):
            return
        if action == "activate":
            self._activate_workspace_context(context)
            return
        if action == "close":
            self.close_model(document_id=context.document_id)
            return
        if not self._activate_workspace_context(context):
            return
        if action == "save":
            self.save_native_project()
        elif action == "save_as":
            self.save_native_project(force_save_as=True)

    def _result_root_action_requested(self, document_id: int, action: str) -> None:
        """Handle result-root activation and independent close actions."""

        try:
            context = self.workspace.document(int(document_id))
        except (KeyError, TypeError, ValueError):
            return
        if action == "activate":
            self._activate_workspace_context(context)
        elif action == "close":
            self.close_model(document_id=context.document_id)

    def _highlight_tree_entry(
        self,
        document_id_or_kind: object,
        kind_or_key: object = None,
        key: object = _TREE_KEY_MISSING,
    ) -> None:
        entry = self._prepare_tree_entry(
            document_id_or_kind,
            kind_or_key,
            key,
        )
        if entry is None:
            return
        _document_id, kind, key = entry
        if kind == "part" and type(key) is str:
            if self._solid_face_boolean_operation is not None:
                self.status_panel.set_state("请在视口中选择目标面", 3000)
                return
            part_reference = LogicalEntityRef(f"part:{key}")
            if self._assign_body_boolean_reference(part_reference):
                return
            try:
                self._apply_session_delta(
                    self.session.set_active_native_part(
                        key,
                        expected_session_revision=(
                            self.document.session_revision
                        ),
                    )
                )
            except (RuntimeError, TypeError, ValueError, KeyError) as error:
                self._show_error("选择当前部件", str(error))
                return
            reference = LogicalEntityRef(f"body:{key}/domain")
            self._selected_geometry_refs = {reference}
            self._geometry_selection_mode = "body"
            self.viewport.set_selection_mode("geometry_body")
            self.viewport.highlight_geometry_entities((reference,))
            active_part = next(
                (
                    part
                    for part in self.document.parts
                    if part.id == key
                ),
                None,
            )
            self.status_panel.set_object(
                "当前部件"
                if active_part is None
                else active_part.name
            )
            self._update_action_states()
            return
        if kind == "geometry_body" and type(key) is str:
            reference = LogicalEntityRef(key)
            if self._assign_solid_face_boolean_reference(reference):
                return
            if self._assign_planar_boolean_reference(reference):
                return
            if self._assign_body_boolean_reference(reference):
                return
            self._selected_geometry_refs = {reference}
            self._geometry_selection_mode = "body"
            self.viewport.set_selection_mode("geometry_body")
            self.viewport.highlight_geometry_entities((reference,))
            self.status_panel.set_object(f"实体 {key}")
            self._update_action_states()
            return
        if self.inspection_service is None:
            return
        resolved_key = self._resolve_tree_analysis_key(kind, key)
        if resolved_key is None:
            return
        if kind == "boundary":
            step_index, boundary_index = resolved_key
            boundary = self.document.model.steps[step_index].boundaries[
                boundary_index
            ]
            target_kind = displacement_target_kind(boundary)
            target = boundary.target
            self.highlight_entity(
                "node" if target_kind == "node_set" and isinstance(target, int)
                else target_kind,
                target,
            )
            names = {
                "node_set": "节点集",
                "edge": "边集合",
                "surface": "表面",
            }
            self.status_panel.set_object(
                f"{names[target_kind]} {target}"
            )
            return
        key = resolved_key
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

    def _reset_tree_entry_highlight(self, followed_by_highlight: bool) -> None:
        """Clear the previous tree highlight before projecting a new item."""

        self.viewport.clear_selection(render=not followed_by_highlight)

    def _current_result_provider(self) -> ResultProvider | None:
        """Return the exact provider for the currently displayed Session result."""

        provider = self.result_provider
        identity = self.session.current_result_identity()
        if (
            type(provider) is not ResultProvider
            or identity is None
            or provider.source != identity[0]
            or provider.snapshot.generation != identity[1]
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
        availability = provider.field_status(selection.field_key)
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
        requested_value = (
            float(self._scale_value if scale_value is None else scale_value)
            if mode == "custom"
            else 0.0
        )
        cache = self._result_deformation_scale_cache
        if (
            cache is not None
            and cache[0] is provider.snapshot
            and cache[1] == shape
            and cache[2] == mode
            and cache[3] == requested_value
        ):
            return cache[4]
        if mode == "real":
            scale = 1.0
        elif mode == "custom":
            scale = requested_value
            if not np.isfinite(scale) or scale < 0.0:
                raise ValueError(
                    "custom deformation scale must be finite and non-negative"
                )
        elif mode != "auto":
            raise ValueError("unknown deformation scale mode")
        else:
            topology = provider.snapshot.topology
            coordinates = topology.node_coordinates
            displacements = topology.nodal_displacements
            if len(coordinates) == 0:
                scale = 1.0
            else:
                span = float(np.linalg.norm(np.ptp(coordinates, axis=0)))
                maximum = float(
                    np.max(np.linalg.norm(displacements, axis=1))
                )
                scale = (
                    1.0
                    if maximum <= 0.0 or span <= 0.0
                    else 0.1 * span / maximum
                )
        self._result_deformation_scale_cache = (
            provider.snapshot,
            shape,
            mode,
            requested_value,
            scale,
        )
        return scale

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
        deformation_scale = self._result_deformation_scale(
            provider,
            shape_mode=shape_mode,
            scale_mode=scale_mode,
            scale_value=scale_value,
        )
        cache = self._result_topology_template_cache
        if (
            cache is not None
            and cache[0] is provider.snapshot
            and cache[1].matches(export, deformation_scale)
        ):
            topology = project_scalar_field_topology_from_template(
                export,
                cache[1],
                deformation_scale,
            )
        else:
            topology = project_scalar_field_topology(
                export,
                deformation_scale=deformation_scale,
            )
            self._result_topology_template_cache = (
                provider.snapshot,
                build_result_field_topology_template(
                    topology,
                    export.field,
                ),
            )
        return build_result_render_payload(
            topology,
            reusable=self.viewport._result_render_payload,
        )

    def _result_averaging_visual_selection(
        self,
        provider: ResultProvider,
        selection: ScalarFieldSelection,
    ) -> ScalarFieldSelection:
        request = selection.field_key.request
        field_id = request.field_id
        if (
            field_id.variable is not ResultVariable.S
            or field_id.position not in {
                FieldPosition.ELEMENT_NODAL,
                FieldPosition.RESOLVED_NODAL,
            }
        ):
            return selection
        visual_request = FieldRequest(
            field_id=ResultFieldId(
                ResultVariable.S,
                FieldPosition.RESOLVED_NODAL,
            ),
            averaging_policy=NodalAveragingPolicy(
                threshold_percent=float(
                    self._contour_options["averaging_threshold"]
                )
            ),
            gauss_order=request.gauss_order,
        )
        return ScalarFieldSelection(
            provider.resolve_request(visual_request),
            selection.component,
        )

    def _result_visualization_provider(
        self,
        provider: ResultProvider,
        selection: ScalarFieldSelection,
    ) -> tuple[ResultProvider, ScalarFieldSelection]:
        visual_selection = self._result_averaging_visual_selection(
            provider,
            selection,
        )
        if visual_selection == selection:
            return provider, selection
        if (
            provider.field_status(visual_selection.field_key).state
            is FieldState.READY
        ):
            return provider, visual_selection
        cached = self._result_visualization_provider_cache
        if (
            cached is None
            or cached[0] != provider.source
            or cached[1] != provider.snapshot.generation
            or cached[2] != visual_selection.field_key
        ):
            return provider, selection
        if (
            cached[3].source != provider.source
            or cached[3].snapshot.generation
            != provider.snapshot.generation
        ):
            return provider, selection
        return (
            cached[3],
            visual_selection,
        )

    def _apply_result_averaging_threshold(self) -> None:
        self._sync_result_averaging_threshold_control()
        provider = self._current_result_provider()
        selection = self.result_selection
        if (
            provider is None
            or type(selection) is not ScalarFieldSelection
        ):
            return
        visual_selection = self._result_averaging_visual_selection(
            provider,
            selection,
        )
        render_provider, render_selection = (
            self._result_visualization_provider(provider, selection)
        )
        if (
            visual_selection == selection
            or render_selection == visual_selection
        ):
            self._apply_display()
            return
        if self.busy:
            self.status_panel.set_state(
            "结果任务中，完成后应用节点平均",
                4000,
            )
            return
        self._begin_result_averaging_visualization(
            provider,
            selection,
            visual_selection,
        )

    def _begin_result_averaging_visualization(
        self,
        provider: ResultProvider,
        selection: ScalarFieldSelection,
        visual_selection: ScalarFieldSelection,
    ) -> None:
        task = None
        try:
            task = self.session.prepare_result_projection(
                provider.source.run_id
            )
            if (
                self.session.current_result_identity()
                != (provider.source, provider.snapshot.generation)
                or task.token.result_id != provider.source.result_id
                or dict(task.token.dependency_revisions).get(
                    "materialization_generation"
                )
                != provider.snapshot.generation
            ):
                raise RuntimeError(
                    "result changed before averaging visualization started"
                )

            def workload(
                context: TaskContext,
            ) -> ResultMaterializationPatch:
                context.report("正在计算节点平均应力云图……")
                return provider.materialize(
                    (visual_selection.field_key,),
                    cancellation=context,
                )

            def apply_result(value: object) -> TaskApplyOutcome:
                if type(value) is not ResultMaterializationPatch:
                    raise TypeError(
                        "averaging visualization must return "
                        "ResultMaterializationPatch"
                    )
                return self._session_task_outcome(
                    self.session.accept_result_projection(task.token),
                    value,
                )

            def succeeded(value: object) -> None:
                delta, patch = value
                if not self._apply_revision_neutral_task_receipt(delta):
                    raise RuntimeError(
                        "averaging visualization receipt was not accepted"
                    )
                visual_provider = provider.apply(patch)
                if not any(
                    field_data.key == visual_selection.field_key
                    for field_data in visual_provider.snapshot.fields
                ):
                    raise RuntimeError(
                        "averaging visualization field was not materialized"
                    )
                current = self._current_result_provider()
                current_selection = self.result_selection
                if (
                    current is None
                    or current.source != provider.source
                    or current.snapshot.generation
                    != provider.snapshot.generation
                    or current_selection != selection
                    or self._result_averaging_visual_selection(
                        current,
                        current_selection,
                    )
                    != visual_selection
                ):
                    return
                self._result_visualization_provider_cache = (
                    provider.source,
                    provider.snapshot.generation,
                    visual_selection.field_key,
                    visual_provider,
                )
                self._apply_display()
                self.status_panel.set_state(
                    "节点平均应力云图已更新",
                    4000,
                )

            started = self._start_task(
                workload,
                succeeded,
                "节点平均应力云图计算失败",
                lambda message: self._session_task_failed(
                    task.token,
                    "节点平均应力云图计算失败",
                    message,
                ),
                task_name="节点平均应力云图",
                on_cancelled=lambda: self._session_task_cancelled(
                    task.token
                ),
                on_inactive_failure=lambda message: self._session_task_failed(
                    task.token,
                    "节点平均应力云图计算失败",
                    message,
                ),
                on_inactive_cancelled=lambda: self._session_task_cancelled(
                    task.token
                ),
                apply_result=apply_result,
            )
        except (KeyError, RuntimeError, TypeError, ValueError) as error:
            if task is not None:
                failure = self.session.accept_task_failed(
                    task.token,
                    error,
                )
                self._apply_revision_neutral_task_receipt(failure)
            self._show_error("节点平均应力云图计算失败", str(error))
            return
        if not started:
            cancelled = self.session.accept_task_cancelled(task.token)
            self._apply_revision_neutral_task_receipt(cancelled)

    def _install_ready_result_selection(
        self,
        provider: ResultProvider,
        selection: ScalarFieldSelection,
    ) -> None:
        if provider is not self._current_result_provider():
            raise RuntimeError("provider is no longer current")
        render_provider, render_selection = (
            self._result_visualization_provider(provider, selection)
        )
        payload = self._build_result_render_payload(
            render_provider,
            render_selection,
        )
        active_context = self.workspace.active_document()
        if not self.result_tree.has_selection(
            selection,
            document_id=(
                None if active_context is None else active_context.document_id
            ),
            source=provider.source,
        ):
            raise RuntimeError(
                "selected field is missing from the result tree"
            )
        self._install_viewport_result_payload(
            payload,
            shape_mode=self._display.shape_mode,
            contour_enabled=self._display.contour_enabled,
        )
        self.result_selection = selection
        if not self.result_tree.select_selection(
            selection,
            document_id=(
                None if active_context is None else active_context.document_id
            ),
            source=provider.source,
        ):
            raise RuntimeError(
                "selected field disappeared from the result tree"
            )
        self._refresh_result_controls()
        self.status_panel.set_result(self._result_status_text())
        self._update_action_states()
        visual_selection = self._result_averaging_visual_selection(
            provider,
            selection,
        )
        if (
            visual_selection != selection
            and render_selection != visual_selection
        ):
            self._defer_ui(self._apply_result_averaging_threshold)

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
                render=not self._workspace_activation,
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
                        render=not self._workspace_activation,
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

    def _restore_viewport_model_scene(
        self,
        *,
        render: bool = True,
        reset_camera: bool = True,
    ) -> None:
        artifact = self.document.artifact
        geometry = self.geometry
        if artifact is None or geometry is None:
            FEMViewport.clear_model(self.viewport)
            return
        model = self._current_gui_model()
        if model is None:
            FEMViewport.clear_model(self.viewport)
            return
        if self.viewport.model_scene_is_current(model, geometry):
            return
        frame_query = None
        if self.document.source_kind != "result":
            def frame_query(target: RegionRef | int) -> BeamFrameReport:
                return resolve_effective_beam_frames(model, target)
        FEMViewport.set_model(
            self.viewport,
            model,
            geometry,
            refresh_symbols=False,
            render=False,
            reset_camera=reset_camera,
            show_edges=self._model_edges_visible,
            show_nodes=self.actions["nodes"].isChecked(),
            show_node_labels=self.actions["node_labels"].isChecked(),
            show_element_labels=self.actions["element_labels"].isChecked(),
            effective_frame_query=frame_query,
            mesh_selection_topology_provider=(
                self._mesh_selection_topology
            ),
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
        selected_references = set(self._selected_mesh_scope_refs)
        if selected_references:
            FEMViewport.highlight_mesh_entities(
                self.viewport,
                selected_references,
                entity_kind=(
                    "body"
                    if self._selection_context.active_filter == "body"
                    else next(iter(selected_references)).kind
                ),
            )
        if render:
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
                active_context = self.workspace.active_document()
                self.result_tree.select_selection(
                    installed,
                    document_id=(
                        None
                        if active_context is None
                        else active_context.document_id
                    ),
                    source=(None if provider is None else provider.source),
                )
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
        if not self._display.contour_enabled:
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

    def _activate_routed_result_selection(
        self,
        document_id: int,
        run_id: str,
        source: object,
        selection: ScalarFieldSelection,
    ) -> None:
        """Activate the owning document/run before applying a field choice."""

        if type(selection) is not ScalarFieldSelection:
            return
        try:
            context = self.workspace.document(int(document_id))
        except (KeyError, TypeError, ValueError):
            return
        if self.workspace.active_document_id != context.document_id:
            if not self._activate_workspace_context(context):
                return
        if not context.is_result and run_id:
            identity = context.session.current_result_identity()
            if identity is None or identity[0].run_id != str(run_id):
                try:
                    receipt = self.select_run_result(str(run_id))
                except (RuntimeError, TypeError, ValueError):
                    return
                if receipt.diagnostic is not None:
                    return
        identity = context.session.current_result_identity()
        if source is not None and (
            identity is None or identity[0] != source
        ):
            return
        self._activate_result_selection(selection)

    def _activate_routed_result_run(
        self,
        document_id: int,
        run_id: str,
    ) -> None:
        """Activate a run item after routing to its owning workspace document."""

        try:
            context = self.workspace.document(int(document_id))
        except (KeyError, TypeError, ValueError):
            return
        if self.workspace.active_document_id != context.document_id:
            if not self._activate_workspace_context(context):
                return
        if context.is_result:
            was_result = self._current_module_name() == "结果"
            self.ribbon.set_current("结果")
            if was_result:
                self._on_module_changed("结果")
            return
        identity = context.session.current_result_identity()
        if identity is not None and identity[0].run_id == str(run_id):
            self._refresh_result_tree_for_context(context)
            was_result = self._current_module_name() == "结果"
            self.ribbon.set_current("结果")
            if was_result:
                self._on_module_changed("结果")
            return
        try:
            receipt = self.select_run_result(str(run_id))
        except (RuntimeError, TypeError, ValueError):
            return
        if receipt.diagnostic is not None:
            return
        self._refresh_result_tree_for_context(context)
        self.ribbon.set_current("结果")

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
        if not self._display.contour_enabled:
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

    def _apply_display(self, *, render: bool = True) -> None:
        provider = self._current_result_provider()
        selection = self.result_selection
        if (
            provider is None
            or type(selection) is not ScalarFieldSelection
        ):
            return
        render_provider, render_selection = (
            self._result_visualization_provider(provider, selection)
        )
        payload = self._build_result_render_payload(
            render_provider,
            render_selection,
        )
        show_edges = (
            bool(self._contour_options["edges"])
            if self._display.contour_enabled
            else self._model_edges_visible
        )
        self.actions["edges"].setChecked(show_edges)
        self._prepare_viewport_for_result_source(render_provider.source)
        self.viewport.set_edges_visible(show_edges, render=False)
        self.viewport.set_result_render_payload(payload)
        self.viewport.set_display(
            self._display.shape_mode,
            self._display.contour_enabled,
            render=render,
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
        position_label = result_field_position_label(
            field_id,
            section_point_labels=result_provider_section_point_labels(provider),
        )
        result_name = (
            f"{field_id.variable.value} {selection.component}"
            f"（{position_label}）"
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
            section_point_labels=result_provider_section_point_labels(provider),
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
        self._exec_view_dialog(dialog)

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
            "结果任务中，完成后应用设置",
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
            render_provider, render_selection = (
                self._result_visualization_provider(
                    provider,
                    settings.selection,
                )
            )
            payload = self._build_result_render_payload(
                render_provider,
                render_selection,
                shape_mode=settings.shape_mode,
                scale_mode=settings.scale_mode,
                scale_value=settings.scale_value,
            )
        except (KeyError, RuntimeError, TypeError, ValueError) as error:
            self._show_error("结果显示失败", str(error))
            return
        active_context = self.workspace.active_document()
        if not self.result_tree.has_selection(
            settings.selection,
            document_id=(
                None if active_context is None else active_context.document_id
            ),
            source=provider.source,
        ):
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
        if not self.result_tree.select_selection(
            settings.selection,
            document_id=(
                None if active_context is None else active_context.document_id
            ),
            source=provider.source,
        ):
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
            if (
                settings.show_edges
                and self._contour_options["edge_mode"]
                == CONTOUR_EDGE_NONE
            ):
                self._contour_options["edge_mode"] = CONTOUR_EDGE_ALL
        else:
            self._model_edges_visible = settings.show_edges
        self.actions["edges"].setChecked(settings.show_edges)
        self.result_scale_combo.setCurrentIndex(
            max(
                0,
                self.result_scale_combo.findData(settings.scale_mode),
            )
        )
        self._sync_result_scale_control()
        self._refresh_result_controls()
        self.status_panel.set_result(self._result_status_text())
        self._update_action_states()
        visual_selection = self._result_averaging_visual_selection(
            provider,
            settings.selection,
        )
        if (
            visual_selection != settings.selection
            and render_selection != visual_selection
        ):
            self._defer_ui(self._apply_result_averaging_threshold)

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
        options = dict(self._contour_options)
        automatic_range = self.viewport.current_contour_range()
        if automatic_range is not None:
            options["automatic_minimum"] = automatic_range[0]
            options["automatic_maximum"] = automatic_range[1]
        dialog = ContourSettingsDialog(options, self)
        dialog.applyRequested.connect(self._set_contour_options)
        self._exec_view_dialog(dialog)

    def show_display_settings_dialog(self) -> None:
        if self._current_result_provider() is None:
            return
        dialog = DisplaySettingsDialog(dict(self._contour_options), self)
        dialog.applyRequested.connect(self._set_contour_options)
        self._exec_view_dialog(dialog)

    def _set_contour_options(self, options: dict[str, Any]) -> None:
        previous_threshold = float(
            self._contour_options["averaging_threshold"]
        )
        updated_options = dict(options)
        if "edge_mode" in updated_options:
            updated_options["edges"] = (
                updated_options["edge_mode"] != CONTOUR_EDGE_NONE
            )
        elif (
            updated_options.get("edges")
            and self._contour_options["edge_mode"]
            == CONTOUR_EDGE_NONE
        ):
            updated_options["edge_mode"] = CONTOUR_EDGE_ALL
        self._contour_options.update(updated_options)
        threshold = float(
            self._contour_options["averaging_threshold"]
        )
        self.result_averaging_threshold.blockSignals(True)
        self.result_averaging_threshold.setValue(threshold)
        self.result_averaging_threshold.blockSignals(False)
        if self._display.contour_enabled:
            show_edges = bool(self._contour_options["edges"])
            self.actions["edges"].setChecked(show_edges)
            self.viewport.set_edges_visible(show_edges, render=False)
        self.viewport.set_contour_metadata(
            {"averaging_threshold": threshold}
        )
        self.viewport.set_contour_options(
            {
                key: value
                for key, value in self._contour_options.items()
                if key != "averaging_threshold"
            }
        )
        if threshold != previous_threshold:
            self._apply_result_averaging_threshold()

    def show_symbol_settings_dialog(self) -> None:
        if self.document.model is None:
            return
        dialog = SymbolSettingsDialog(
            self._symbol_settings,
            self.session.runnable_step_names(),
            self,
        )
        dialog.applyRequested.connect(self._apply_symbol_settings)
        self._exec_view_dialog(dialog)

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
        self.viewport.set_symbol_sampling_density_override(None)
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
        provider = self._current_result_provider()
        selection = self.result_selection
        if (
            provider is None
            or type(selection) is not ScalarFieldSelection
        ):
            return
        try:
            dialog = ResultCsvExportDialog(
                provider.catalog(),
                current_selection=selection,
                section_point_labels=result_provider_section_point_labels(
                    provider
                ),
                parent=self,
            )
        except (TypeError, ValueError) as error:
            self._show_error("导出 CSV 失败", str(error))
            return
        if self._exec_dialog(dialog) != QDialog.DialogCode.Accepted:
            return
        export_selections = dialog.current_selections()
        target = dialog.target_path()
        try:
            availabilities = tuple(
                self._catalog_availability_for_selection(
                    provider,
                    export_selection,
                )
                for export_selection in export_selections
            )
        except (KeyError, TypeError, ValueError) as error:
            self._show_error("导出 CSV 失败", str(error))
            return
        if any(
            availability.state is not FieldState.READY
            for availability in availabilities
        ):
            self._show_error("导出 CSV 失败", "所选结果字段尚未就绪")
            return
        self.status_panel.set_state("正在导出 CSV……")
        receipt = self.export_result_csv(
            target,
            ResultCsvExportSpec(
                provider.source,
                provider.snapshot.generation,
                export_selections,
            ),
        )
        if receipt.diagnostic is not None:
            self._show_command_rejection("导出 CSV 失败", receipt)
            return
        if receipt.completion is not None:
            def export_finished(terminal: TaskCompletion) -> None:
                if (
                    terminal.state is BackgroundTaskState.SUCCEEDED
                    and terminal.projection_error is None
                ):
                    self._show_save_success("CSV 文件", target)

            receipt.completion.observe(export_finished)

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
        position_token = field_id.position.value
        if field_id.position is FieldPosition.SECTION_POINT:
            position_token = (
                f"{position_token}_{field_id.section_point_number}"
            )
        field_label = "_".join(
            (
                field_id.variable.value,
                (
                    "node"
                    if field_id.position is FieldPosition.ELEMENT_NODAL
                    else position_token
                ),
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
        dialog = ViewportImageExportDialog(
            self.viewport.screenshot_size(),
            self,
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        options = dialog.options
        try:
            self.viewport.save_screenshot(
                dialog.target_path,
                scale=options.scale,
                window_size=options.window_size,
                transparent_background=options.transparent_background,
            )
        except Exception as error:
            self._show_error("导出视口图片失败", str(error))
            return
        self.status_panel.set_state("视口图片保存完成", 5000)
        self._show_save_success("视口图片", dialog.target_path)

    def show_model_information(self) -> None:
        if (
            self.inspection_service is None
            or self.document.source_kind == "native"
        ):
            source = {
                "native": "自主模型",
                "imported": "INP 模型",
            }.get(self.document.source_kind, "未打开")
            self._show_information("模型概况", [
                (
                    "模型名称",
                    self.document.model_name
                    or getattr(self.document.model, "name", "—"),
                ),
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
                    "节点数量",
                    len(getattr(getattr(self.document.model, "mesh", None), "nodes", ())),
                ),
                (
                    "单元数量",
                    len(getattr(getattr(self.document.model, "mesh", None), "elements", ())),
                ),
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

    def _show_entry_information(
        self,
        document_id_or_kind: object,
        kind_or_key: object = None,
        key: object = _TREE_KEY_MISSING,
    ) -> None:
        entry = self._prepare_tree_entry(
            document_id_or_kind,
            kind_or_key,
            key,
        )
        if entry is None:
            return
        _document_id, kind, key = entry
        if kind == "model":
            self.show_model_information()
        elif kind == "part":
            try:
                part = self.document.part(str(key))
            except KeyError:
                return
            self._show_information("部件信息", [
                ("名称", part.name),
                ("稳定标识", part.id),
                ("所属模型", self.document.model_name or "模型-1"),
                ("特征数量", len(part.feature_history)),
                ("维度", f"{part.dimension}D"),
                ("状态", "已抑制" if part.suppressed else "活动"),
                (
                    "几何状态",
                    "已创建",
                ),
            ])
        elif kind == "feature":
            record = next(
                (
                    item
                    for item in self.document.feature_history
                    if item.name == str(key)
                ),
                None,
            )
            rows: list[tuple[str, object]] = [
                ("名称", native_feature_label(key))
            ]
            if record is not None:
                rows.append(("类型", native_feature_kind_label(record.kind)))
                body_name = record.payload.get("body_name")
                if body_name:
                    rows.append(("实体", body_name))
            self._show_information("特征信息", rows)
        elif kind == "geometry_body":
            body_id = str(key).removeprefix("body:")
            recipe = self.document.geometry_recipe
            body = (
                next(
                    (
                        item
                        for item in recipe.bodies
                        if item.id == body_id
                    ),
                    None,
                )
                if isinstance(recipe, MultiBodyGeometry)
                else None
            )
            if body is None:
                return
            self._show_information("实体信息", [
                ("名称", body.name),
                ("标识", body.id),
                ("特征数量", len(derive_geometry_feature_rows(body.recipe))),
                ("几何类型", type(body.recipe).__name__),
            ])
        elif kind == "mesh":
            self.show_mesh_browser()
        else:
            resolved_key = self._resolve_tree_analysis_key(kind, key)
            if resolved_key is not None:
                self.show_entity_information(kind, resolved_key)

    def _rename_tree_entry(
        self,
        document_id_or_kind: object,
        kind_or_key: object = None,
        key: object = _TREE_KEY_MISSING,
    ) -> None:
        entry = self._prepare_tree_entry(
            document_id_or_kind,
            kind_or_key,
            key,
        )
        if entry is None:
            return
        _document_id, kind, _key = entry
        if self.busy or self.document.source_kind != "native":
            return
        if kind == "model":
            current = str(self.document.model_name or "模型-1")
            title = "重命名模型"
            prompt = "模型名称："
            rename = self.session.rename_native_model
        elif kind == "part" and type(_key) is str:
            try:
                part = self.document.part(_key)
            except KeyError:
                return
            current = part.name
            title = "重命名部件"
            prompt = "部件名称："
            rename = None
        else:
            return
        name, accepted = QInputDialog.getText(
            self,
            title,
            prompt,
            text=current,
        )
        if not accepted:
            return
        try:
            if kind == "part":
                delta = self.session.rename_native_part(
                    str(_key),
                    name,
                    expected_part_revision=self.document.part_revision(
                        str(_key)
                    ),
                    expected_session_revision=(
                        self.document.session_revision
                    ),
                )
            else:
                if rename is None:
                    raise RuntimeError("重命名命令不可用")
                delta = rename(
                    name,
                    expected_session_revision=(
                        self.document.session_revision
                    ),
                )
            self._apply_session_delta(delta)
            if self.document.artifact is not None:
                self._refresh_model_tree(self.document.model)
        except (RevisionConflictError, RuntimeError, TypeError, ValueError) as error:
            self._show_error(title, str(error))

    def _edit_tree_entry(
        self,
        document_id_or_kind: object,
        kind_or_key: object = None,
        key: object = _TREE_KEY_MISSING,
    ) -> None:
        entry = self._prepare_tree_entry(
            document_id_or_kind,
            kind_or_key,
            key,
        )
        if entry is None:
            return
        document_id, kind, key = entry
        if kind == "part" and type(key) is str:
            if document_id is None:
                self._highlight_tree_entry(kind, key)
            else:
                self._highlight_tree_entry(document_id, kind, key)
            if self.document.active_part_id != key:
                return
            self.show_geometry_manager()
        elif kind == "material":
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
            "body_load",
            "gravity_load",
            "output",
        }:
            self.edit_analysis_definition(kind, key)
        else:
            self.show_entity_information(kind, key)

    def _delete_tree_entry(
        self,
        document_id_or_kind: object,
        kind_or_key: object = None,
        key: object = _TREE_KEY_MISSING,
    ) -> None:
        entry = self._prepare_tree_entry(
            document_id_or_kind,
            kind_or_key,
            key,
        )
        if entry is None:
            return
        document_id, kind, key = entry
        if kind == "part" and type(key) is str:
            if document_id is None:
                self._highlight_tree_entry(kind, key)
            else:
                self._highlight_tree_entry(document_id, kind, key)
            if self.document.active_part_id != key:
                return
            self.delete_geometry()
            return
        self.delete_analysis_definition(kind, key)

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
        self.viewport.clear_selection(render=False)
        if kind == "boundary":
            step_index, boundary_index = key
            boundary = self.document.model.steps[int(step_index)].boundaries[
                int(boundary_index)
            ]
            target_kind = displacement_target_kind(boundary)
            self.highlight_entity(
                "node"
                if target_kind == "node_set"
                and isinstance(boundary.target, int)
                else target_kind,
                boundary.target,
            )
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
            reference_factory = (
                MeshEntityRef.face
                if region_kind == "surface"
                else MeshEntityRef.edge
            )
            references = tuple(
                reference_factory(
                    int(member.elem_id),
                    int(member.local_index),
                    member.node_ids,
                )
                for member in members
            )
            self.viewport.highlight_mesh_entities(
                references,
                entity_kind="face" if region_kind == "surface" else "edge",
            )
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
        scope_kind = (
            "node"
            if kind in {"node_set", "cload"}
            else "element"
            if kind in {
                "element_set",
                "line_load",
                "body_load",
                "gravity_load",
            }
            else None
        )
        if scope_kind == "node" and selection.node_ids:
            self.viewport.highlight_mesh_entities(
                tuple(
                    MeshEntityRef.node(int(node_id))
                    for node_id in selection.node_ids
                ),
                entity_kind="node",
            )
            return
        if scope_kind == "element" and selection.element_ids:
            self.viewport.highlight_mesh_entities(
                tuple(
                    MeshEntityRef.element(int(element_id))
                    for element_id in selection.element_ids
                ),
                entity_kind="element",
            )
            if beam_frame_target is not None:
                self.viewport.show_beam_frame_preview(
                    beam_frame_target,
                    render=False,
                )
            return
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

    def _confirm_workspace_task_exit(self, context: WorkspaceDocument) -> bool:
        """Ask whether one document-owned task may be cancelled for exit."""

        box = QMessageBox(self)
        box.setWindowTitle("任务正在运行")
        box.setIcon(QMessageBox.Icon.Question)
        box.setText(
            f"文档“{context.display_name}”有后台任务正在运行，是否取消任务并退出？"
        )
        cancel_button = box.addButton(
            "取消任务并退出",
            QMessageBox.ButtonRole.AcceptRole,
        )
        continue_button = box.addButton(
            "继续编辑",
            QMessageBox.ButtonRole.RejectRole,
        )
        box.setDefaultButton(continue_button)
        box.exec()
        return box.clickedButton() is cancel_button

    def _shutdown_shared_resources(self) -> None:
        """Shutdown the singleton viewport/Agent boundary at most once."""

        if self._shared_resources_shutdown:
            return
        self._shared_resources_shutdown = True
        self.viewport_panel.overlay_host.shutdown(wait=False)
        self.viewport.shutdown_backend()

    def _finish_workspace_exit(self) -> None:
        """Release every idle document, then close the shared window boundary."""

        if self._closing:
            return
        if self.workspace.any_busy():
            return
        for context in tuple(self.workspace.documents()):
            if not self._release_workspace_context_for_exit(context):
                return
        self._active_context = None
        self._clear_model_projection(clear_tree=True)
        self._exit_pending = False
        self._closing = True
        self._deferred_ui_timer.stop()
        self._deferred_ui_callbacks.clear()
        self._close_inspection_windows()
        self._close_job_manager()
        self._shutdown_shared_resources()
        self.close()

    def _release_workspace_context_for_exit(
        self,
        context: WorkspaceDocument,
    ) -> bool:
        """Release one confirmed document without activating its neighbors."""

        if context.task_controller.busy:
            return False
        if context.projection.is_open:
            try:
                context.session.close(
                    expected_session_revision=context.projection.session_revision
                )
            except (RevisionConflictError, RuntimeError, TypeError, ValueError):
                logging.exception("workspace document could not close during exit")
                return False
        context.presentation_cache.invalidate_model()
        context.presentation_cache.invalidate_result()
        context.presentation_state = DocumentPresentationState()
        if context.is_result:
            self.result_tree.remove_archive(context.document_id)
        else:
            self.model_tree.remove_document(context.document_id)
            self.result_tree.remove_model_runs(context.document_id)
        self.workspace.remove(context)
        return True

    def closeEvent(self, event: QCloseEvent) -> None:
        if self._closing:
            event.accept()
            return
        if self._exit_pending:
            event.ignore()
            return
        contexts = tuple(self.workspace.documents())
        if self.isVisible():
            # Confirm every context before mutating any registry entry.  This
            # keeps a later cancellation from leaving a partially closed
            # workspace.
            for context in contexts:
                if not self._confirm_workspace_context_close(context, True):
                    event.ignore()
                    return
                if context.task_controller.busy and not self._confirm_workspace_task_exit(
                    context
                ):
                    event.ignore()
                    return
        busy_contexts = tuple(
            context for context in contexts if context.task_controller.busy
        )
        if busy_contexts:
            self._exit_pending = True
            pending_ids = {context.document_id for context in busy_contexts}

            def release_after_cancel(document_id: int) -> None:
                pending_ids.discard(document_id)
                if pending_ids or not self._exit_pending:
                    return
                self._finish_workspace_exit()

            for context in busy_contexts:
                requested = self._request_workspace_context_cancel(
                    context,
                    after_cleanup=lambda document_id=context.document_id: release_after_cancel(
                        document_id
                    ),
                )
                if not requested and not context.task_controller.cancel_requested:
                    self._exit_pending = False
                    event.ignore()
                    return
            event.ignore()
            return
        self._finish_workspace_exit()
        if self._closing:
            event.accept()
        else:
            event.ignore()
