"""Qt-free action catalog and availability projection for the FEM GUI."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from fem.application import (
    AuthoringCapability,
    AuthoringStatus,
    SessionAuthoringProjection,
    SessionSnapshot,
)
from fem.application.results import FieldState
from fem.geometry import (
    BooleanGeometry,
    ExtrudedGeometry,
    ExtrusionSourceResolutionError,
    MovedGeometry,
    MultiBodyGeometry,
    NATIVE_GEOMETRY_TYPES,
    RevolvedGeometry,
    RotatedGeometry,
    geometry_dimension,
    resolve_extrusion_source_faces,
    analyze_body_relations,
)
from fem.geometry.references import LogicalEntityRef
from fem.mesh.settings import MeshSettings


class GuiActionKey(str, Enum):
    OPEN = "open"
    NEW_NATIVE = "new_native"
    DELETE_MODEL = "delete_model"
    OPEN_PROJECT = "open_project"
    SAVE_PROJECT = "save_project"
    SAVE_PROJECT_AS = "save_project_as"
    SAVE_RESULT = "save_result"
    SAVE_RESULT_AS = "save_result_as"
    OPEN_RESULT = "open_result"
    RELOAD = "reload"
    CLOSE = "close"
    EXIT = "exit"
    MODEL_INFO = "model_info"
    MATERIAL_MANAGER = "material_manager"
    SECTION_MANAGER = "section_manager"
    SECTION_ASSIGN = "section_assign"
    GEOMETRY_CREATE = "geometry_create"
    GEOMETRY_SKETCH = "geometry_sketch"
    GEOMETRY_FACE_SKETCH = "geometry_face_sketch"
    GEOMETRY_WIRE = "geometry_wire"
    GEOMETRY_MOVE = "geometry_move"
    GEOMETRY_ROTATE = "geometry_rotate"
    GEOMETRY_EXTRUDE = "geometry_extrude"
    GEOMETRY_SWEEP = "geometry_sweep"
    GEOMETRY_FUSE = "geometry_fuse"
    GEOMETRY_CUT = "geometry_cut"
    GEOMETRY_MANAGER = "geometry_manager"
    GEOMETRY_UNDO = "geometry_undo"
    GEOMETRY_DELETE = "geometry_delete"
    GEOMETRY_REGION = "geometry_region"
    GEOMETRY_REGIONS = "geometry_regions"
    MESH_SETTINGS = "mesh_settings"
    MESH_GENERATE = "mesh_generate"
    MESH_CLEAR = "mesh_clear"
    MESH_CONTROLS = "mesh_controls"
    MESH_LOCAL_CONTROL = "mesh_local_control"
    MESH_STATISTICS = "mesh_statistics"
    MESH_QUALITY = "mesh_quality"
    MESH_VERIFY = "mesh_verify"
    FIT = "fit"
    TOP = "top"
    BOTTOM = "bottom"
    FRONT = "front"
    BACK = "back"
    LEFT = "left"
    RIGHT = "right"
    ISO = "iso"
    ORTHOGRAPHIC = "orthographic"
    PERSPECTIVE = "perspective"
    VIEWPORT_BACKGROUND = "viewport_background"
    SUPPRESSED_PART_GHOSTS = "suppressed_part_ghosts"
    EDGES = "edges"
    NODES = "nodes"
    NODE_LABELS = "node_labels"
    ELEMENT_LABELS = "element_labels"
    SYMBOLS = "symbols"
    SYMBOL_SETTINGS = "symbol_settings"
    STEP_INFO = "step_info"
    STEP_CREATE = "step_create"
    BOUNDARY_CREATE = "boundary_create"
    LOAD_CREATE = "load_create"
    OUTPUT_CREATE = "output_create"
    ANALYSIS_MANAGER = "analysis_manager"
    CHECK_MODEL = "check_model"
    SUBMIT_JOB = "submit_job"
    RESUBMIT_JOB = "resubmit_job"
    JOB_MANAGER = "job_manager"
    UNDEFORMED = "undeformed"
    DEFORMED = "deformed"
    CONTOUR = "contour"
    OVERLAY = "overlay"
    FIELD = "field"
    DISPLAY_SETTINGS = "display_settings"
    SCALE = "scale"
    CONTOUR_OPTIONS = "contour_options"
    QUERY = "query"
    EXPORT_CSV = "export_csv"
    EXPORT_VTK = "export_vtk"
    SCREENSHOT = "screenshot"
    ABOUT = "about"
    SELECT_POINT = "select_point"
    SELECT_EDGE = "select_edge"
    SELECT_ELEMENT = "select_element"
    SELECT_FACE = "select_face"
    SELECT_BODY = "select_body"
    CLEAR_SELECTION = "clear_selection"
    SELECTED_INFO = "selected_info"


@dataclass(frozen=True, slots=True)
class GuiActionDescriptor:
    key: GuiActionKey
    text: str
    handler: str
    icon_name: str | None = None
    checkable: bool = False
    checked: bool = False
    group: str | None = None
    argument: object | None = None
    checked_only: bool = False


def _d(
    key: GuiActionKey,
    text: str,
    handler: str,
    icon_name: str | None = None,
    *,
    checkable: bool = False,
    checked: bool = False,
    group: str | None = None,
    argument: object | None = None,
    checked_only: bool = False,
) -> GuiActionDescriptor:
    return GuiActionDescriptor(
        key,
        text,
        handler,
        icon_name,
        checkable,
        checked,
        group,
        argument,
        checked_only,
    )


ACTION_DESCRIPTORS: tuple[GuiActionDescriptor, ...] = (
    _d(GuiActionKey.OPEN, "Open INP", "open_inp", "open_inp"),
    _d(GuiActionKey.NEW_NATIVE, "New Model", "new_native_model", "new_model"),
    _d(
        GuiActionKey.DELETE_MODEL,
        "Delete Model",
        "delete_current_model",
        "geometry_delete",
    ),
    _d(GuiActionKey.OPEN_PROJECT, "Open Model", "open_native_project", "open_project"),
    _d(GuiActionKey.SAVE_PROJECT, "Save Model", "save_native_project", "save_project"),
    _d(GuiActionKey.SAVE_PROJECT_AS, "Save Model As...", "save_native_project_as"),
    _d(GuiActionKey.RELOAD, "Reload", "reload_model", "reload"),
    _d(GuiActionKey.CLOSE, "Close Model", "close_model", "close"),
    _d(GuiActionKey.SAVE_RESULT, "Save Results", "save_current_result", "save_result"),
    _d(GuiActionKey.SAVE_RESULT_AS, "Save Results As...", "save_current_result_as"),
    _d(GuiActionKey.OPEN_RESULT, "Open Results", "open_result_file", "open_result"),
    _d(GuiActionKey.EXIT, "Exit", "close"),
    _d(GuiActionKey.MODEL_INFO, "Model Overview", "show_model_information", "model_info"),
    _d(GuiActionKey.MATERIAL_MANAGER, "Material Manager", "show_material_manager", "material"),
    _d(GuiActionKey.SECTION_MANAGER, "Section Manager", "show_section_manager", "section"),
    _d(GuiActionKey.SECTION_ASSIGN, "Assign Section", "assign_section_to_region", "section_assign"),
    _d(GuiActionKey.GEOMETRY_CREATE, "New Part", "create_geometry", "sketch"),
    _d(GuiActionKey.GEOMETRY_SKETCH, "New Sketch", "create_sketch_geometry", "sketch"),
    _d(
        GuiActionKey.GEOMETRY_FACE_SKETCH,
        "Sketch on Face",
        "start_face_sketch_boolean",
        "sketch",
    ),
    _d(GuiActionKey.GEOMETRY_WIRE, "New Wire", "start_wire_geometry", "wire"),
    _d(GuiActionKey.GEOMETRY_MOVE, "Move", "move_geometry", "geometry_move"),
    _d(GuiActionKey.GEOMETRY_ROTATE, "Rotate", "rotate_geometry", "geometry_rotate"),
    _d(GuiActionKey.GEOMETRY_EXTRUDE, "Extrude", "extrude_geometry", "extrude"),
    _d(GuiActionKey.GEOMETRY_SWEEP, "Sweep", "sweep_geometry", "sweep"),
    _d(GuiActionKey.GEOMETRY_FUSE, "Fuse", "fuse_geometry", "boolean_fuse"),
    _d(GuiActionKey.GEOMETRY_CUT, "Cut", "cut_geometry", "boolean_cut"),
    _d(GuiActionKey.GEOMETRY_MANAGER, "Edit", "show_geometry_manager", "feature_edit"),
    _d(GuiActionKey.GEOMETRY_UNDO, "Undo Feature", "undo_geometry_feature", "feature_undo"),
    _d(GuiActionKey.GEOMETRY_DELETE, "Delete Geometry", "delete_geometry", "geometry_delete"),
    _d(GuiActionKey.GEOMETRY_REGION, "Create Scope", "create_named_geometry_region", "named_region_create"),
    _d(GuiActionKey.GEOMETRY_REGIONS, "Scope Manager", "show_named_region_manager", "named_region_manager"),
    _d(GuiActionKey.MESH_SETTINGS, "Mesh Settings", "edit_mesh_settings", "mesh_settings"),
    _d(GuiActionKey.MESH_GENERATE, "Generate Mesh", "generate_native_mesh", "mesh"),
    _d(GuiActionKey.MESH_CLEAR, "Clear Mesh", "clear_native_mesh", "mesh_clear"),
    _d(GuiActionKey.MESH_CONTROLS, "Control Manager", "show_mesh_controls", "mesh_controls"),
    _d(GuiActionKey.MESH_LOCAL_CONTROL, "Local Mesh", "set_local_mesh_control", "mesh_local_control"),
    _d(GuiActionKey.MESH_STATISTICS, "Mesh Statistics", "show_mesh_statistics", "mesh_statistics"),
    _d(GuiActionKey.MESH_QUALITY, "Quality Check", "show_mesh_quality", "mesh_quality"),
    _d(GuiActionKey.MESH_VERIFY, "Verify Mesh", "show_mesh_verification", "mesh_verify"),
    _d(GuiActionKey.FIT, "Fit to Window", "viewport_fit", "fit"),
    _d(GuiActionKey.FRONT, "Front View", "viewport.set_view", "front", argument="front"),
    _d(GuiActionKey.BACK, "Back View", "viewport.set_view", "back", argument="back"),
    _d(GuiActionKey.TOP, "Top View", "viewport.set_view", "top", argument="top"),
    _d(GuiActionKey.BOTTOM, "Bottom View", "viewport.set_view", "bottom", argument="bottom"),
    _d(GuiActionKey.LEFT, "Left View", "viewport.set_view", "left", argument="left"),
    _d(GuiActionKey.RIGHT, "Right View", "viewport.set_view", "right", argument="right"),
    _d(GuiActionKey.ISO, "Isometric View", "viewport.set_view", "iso", argument="iso"),
    _d(GuiActionKey.ORTHOGRAPHIC, "Orthographic", "viewport.set_parallel_projection", "orthographic", checkable=True, checked=True, group="projection", argument=True, checked_only=True),
    _d(GuiActionKey.PERSPECTIVE, "Perspective", "viewport.set_parallel_projection", "perspective", checkable=True, group="projection", argument=False, checked_only=True),
    _d(GuiActionKey.VIEWPORT_BACKGROUND, "Viewport Background", "show_viewport_background_dialog", "background"),
    _d(GuiActionKey.SUPPRESSED_PART_GHOSTS, "Show Suppressed Source Parts", "_toggle_suppressed_part_ghosts", checkable=True),
    _d(GuiActionKey.EDGES, "Show Element Edges", "_toggle_edges", "edges", checkable=True, checked=True),
    _d(GuiActionKey.NODES, "Show Nodes", "_toggle_nodes", "nodes", checkable=True),
    _d(GuiActionKey.NODE_LABELS, "Show Node IDs", "_toggle_node_labels", "node_ids", checkable=True),
    _d(GuiActionKey.ELEMENT_LABELS, "Show Element IDs", "_toggle_element_labels", "element_ids", checkable=True),
    _d(GuiActionKey.SYMBOLS, "Show Constraints and Loads", "_toggle_symbols", "symbols", checkable=True, checked=True),
    _d(GuiActionKey.SYMBOL_SETTINGS, "Symbol Settings", "show_symbol_settings_dialog", "settings"),
    _d(GuiActionKey.STEP_INFO, "Step Info", "show_current_step_information", "step_info"),
    _d(GuiActionKey.STEP_CREATE, "Create Step", "create_static_step", "step_create"),
    _d(GuiActionKey.BOUNDARY_CREATE, "Displacement BC", "create_displacement_boundary", "boundary"),
    _d(GuiActionKey.LOAD_CREATE, "Load BC", "create_load", "load"),
    _d(GuiActionKey.OUTPUT_CREATE, "Output Requests", "create_output_request", "output"),
    _d(GuiActionKey.ANALYSIS_MANAGER, "Analysis Manager", "show_analysis_manager", "analysis_manager"),
    _d(GuiActionKey.CHECK_MODEL, "Check Model", "start_model_check", "check"),
    _d(GuiActionKey.SUBMIT_JOB, "Create Job", "create_job", "job"),
    _d(GuiActionKey.RESUBMIT_JOB, "Copy Job", "resubmit_job", "resubmit"),
    _d(GuiActionKey.JOB_MANAGER, "Job Manager", "show_job_manager", "job_manager"),
    _d(GuiActionKey.UNDEFORMED, "Undeformed", "set_shape_mode", "undeformed", checkable=True, checked=True, group="shape", argument="undeformed"),
    _d(GuiActionKey.DEFORMED, "Deformed", "set_shape_mode", "deformed", checkable=True, group="shape", argument="deformed"),
    _d(GuiActionKey.CONTOUR, "Contours", "_toggle_contour", "contour", checkable=True),
    _d(GuiActionKey.OVERLAY, "Undeformed Outline", "_toggle_undeformed_overlay", "overlay", checkable=True),
    _d(GuiActionKey.FIELD, "Field / Component", "show_result_display_dialog", "field"),
    _d(GuiActionKey.DISPLAY_SETTINGS, "Display Settings", "show_display_settings_dialog", "settings"),
    _d(GuiActionKey.SCALE, "Deformation Scale", "show_result_display_dialog", "scale"),
    _d(GuiActionKey.CONTOUR_OPTIONS, "Contour Settings", "show_contour_dialog", "settings"),
    _d(
        GuiActionKey.QUERY,
        "Query Results",
        "show_result_query_dialog",
        "query",
    ),
    _d(GuiActionKey.EXPORT_CSV, "Export CSV", "export_csv", "export"),
    _d(GuiActionKey.EXPORT_VTK, "Export VTK", "export_vtk", "export"),
    _d(GuiActionKey.SCREENSHOT, "Export Viewport", "export_viewport_image", "image"),
    _d(GuiActionKey.ABOUT, "About", "show_about"),
    _d(GuiActionKey.SELECT_POINT, "Select Points", "_set_selection_filter", "select_geometry_point", checkable=True, checked=True, group="selection", argument="point", checked_only=True),
    _d(GuiActionKey.SELECT_ELEMENT, "Select Elements", "_set_selection_filter", "select_element", checkable=True, group="selection", argument="element", checked_only=True),
    _d(GuiActionKey.SELECT_EDGE, "Select Edges", "_set_selection_filter", "select_geometry_edge", checkable=True, group="selection", argument="edge", checked_only=True),
    _d(GuiActionKey.SELECT_FACE, "Select Faces", "_set_selection_filter", "select_geometry_face", checkable=True, group="selection", argument="face", checked_only=True),
    _d(GuiActionKey.SELECT_BODY, "Select Bodies", "_set_selection_filter", "select_geometry_body", checkable=True, group="selection", argument="body", checked_only=True),
    _d(GuiActionKey.CLEAR_SELECTION, "Clear Selection", "clear_selection", "clear_selection"),
    _d(GuiActionKey.SELECTED_INFO, "Inspect Selection", "show_selected_information", "inspect"),
)


@dataclass(frozen=True, slots=True)
class GuiActionContext:
    busy: bool = False
    model_document_active: bool = False
    selected_step_name: str | None = None
    geometry_selection: tuple[LogicalEntityRef, ...] = ()
    fem_selection_kind: str | None = None
    display_backend_available: bool = True
    open_dialog_keys: frozenset[str] = frozenset()
    viewport_capture_active: bool = False
    # Compatibility defaults preserve the legacy displayed-result projection
    # until MainWindow supplies the derived Phase-8 facts explicitly.
    result_source_current: bool = True
    catalog_available: bool = True
    selected_field_exists: bool = True
    selected_field_state: FieldState | None = FieldState.READY
    materialization_pending: bool = False
    result_task_busy: bool = False
    viewport_scene_available: bool = False
    wire_editor_active: bool = False
    sketch_editor_active: bool = False
    boolean_editor_active: bool = False
    planar_solid_face_selected: bool = False
    selection_space: str = "mesh"
    selection_filter: str = "point"
    selection_topological_dimension: int | None = None
    symbol_display_disabled: bool = False

    def __post_init__(self) -> None:
        for name in (
            "busy",
            "model_document_active",
            "display_backend_available",
            "viewport_capture_active",
            "result_source_current",
            "catalog_available",
            "selected_field_exists",
            "materialization_pending",
            "result_task_busy",
            "viewport_scene_available",
            "wire_editor_active",
            "sketch_editor_active",
            "boolean_editor_active",
            "planar_solid_face_selected",
            "symbol_display_disabled",
        ):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be a bool")
        if (
            self.selected_field_state is not None
            and type(self.selected_field_state) is not FieldState
        ):
            raise TypeError("selected_field_state must be a FieldState or None")
        selection = tuple(self.geometry_selection)
        if any(type(item) is not LogicalEntityRef for item in selection):
            raise TypeError("geometry selection must contain LogicalEntityRef values")
        object.__setattr__(self, "geometry_selection", selection)
        object.__setattr__(
            self,
            "open_dialog_keys",
            frozenset(str(item) for item in self.open_dialog_keys),
        )
        if self.selection_space not in {"geometry", "mesh"}:
            raise ValueError("selection_space must be geometry or mesh")
        if self.selection_filter not in {
            "point", "element", "edge", "face", "body",
        }:
            raise ValueError("unsupported selection_filter")
        if (
            self.selection_topological_dimension is not None
            and self.selection_topological_dimension not in {1, 2, 3}
        ):
            raise ValueError("selection_topological_dimension must be 1, 2, 3, or None")


@dataclass(frozen=True, slots=True)
class ActionAvailability:
    key: GuiActionKey
    enabled: bool
    reason: str = ""


def derive_action_availability(
    snapshot: SessionSnapshot,
    authoring: SessionAuthoringProjection,
    context: GuiActionContext,
) -> tuple[ActionAvailability, ...]:
    """Return exactly one deterministic result for every production action."""

    if not isinstance(context, GuiActionContext):
        raise TypeError("context must be GuiActionContext")
    states = {
        descriptor.key: ActionAvailability(descriptor.key, True, "")
        for descriptor in ACTION_DESCRIPTORS
    }

    def set_state(key: GuiActionKey, enabled: bool, reason: str) -> None:
        states[key] = ActionAvailability(key, bool(enabled), "" if enabled else reason)

    busy = context.busy
    has_model = snapshot.artifact is not None
    has_result = snapshot.displayed_result is not None
    has_current_result = has_result and context.result_source_current
    has_result_catalog = has_current_result and context.catalog_available
    result_actions_idle = (
        not busy
        and not context.materialization_pending
        and not context.result_task_busy
    )
    recipe = snapshot.geometry_recipe
    active_part = snapshot.active_part
    active_part_editable = (
        active_part is not None
        and not active_part.suppressed
        and active_part.geometry_recipe is not None
    )
    set_state(
        GuiActionKey.SUPPRESSED_PART_GHOSTS,
        snapshot.source_kind == "native"
        and any(part.suppressed for part in snapshot.parts),
        "There are no suppressed source parts",
    )
    editor_active = (
        context.wire_editor_active
        or context.sketch_editor_active
        or context.boolean_editor_active
    )
    has_native_geometry = (
        snapshot.source_kind == "native"
        and isinstance(recipe, NATIVE_GEOMETRY_TYPES)
        and active_part_editable
    )
    is_multi_body = isinstance(recipe, MultiBodyGeometry)
    selected_body_count = len(
        {
            reference.logical_id
            for reference in context.geometry_selection
            if reference.kind == "body"
            and reference.logical_id != "body:domain"
        }
    )

    set_state(GuiActionKey.OPEN, not busy, "Cannot open an INP file while a background task is running")
    set_state(GuiActionKey.NEW_NATIVE, not busy, "Cannot create a project while a background task is running")
    set_state(
        GuiActionKey.DELETE_MODEL,
        context.model_document_active and not busy,
        (
            "Cannot delete a model while a background task is running"
            if busy
            else "No model is selected"
        ),
    )
    set_state(GuiActionKey.OPEN_PROJECT, not busy, "Cannot open a project while a background task is running")
    for key in (GuiActionKey.SAVE_PROJECT, GuiActionKey.SAVE_PROJECT_AS):
        set_state(
            key,
            snapshot.can_save and not busy,
            "Create a native part first; INP models use the original file workflow",
        )
    for key in (GuiActionKey.SAVE_RESULT, GuiActionKey.SAVE_RESULT_AS):
        set_state(
            key,
            has_current_result and result_actions_idle,
            (
                "Cannot save results while a background task is running"
                if busy
                else "No successful results are available to save"
                if not has_current_result
                else "A result task is running or the result source is stale"
            ),
        )
    set_state(
        GuiActionKey.OPEN_RESULT,
        not busy and not editor_active,
        "Cannot open results while a background task is running"
        if busy
        else "Finish or cancel the current edit first",
    )
    set_state(
        GuiActionKey.RELOAD,
        snapshot.can_reload and not busy,
        "Only an open INP model can be reloaded",
    )
    set_state(
        GuiActionKey.CLOSE,
        snapshot.source_kind is not None and not busy,
        "No model or project is open",
    )
    create_part_model_ready = (
        snapshot.source_kind == "native"
        or (
            snapshot.source_kind is None
            and context.model_document_active
        )
    )
    create_part_enabled = (
        create_part_model_ready
        and not snapshot.parts
        and not busy
    )
    create_part_reason = (
        "Cannot create a part while a background task is running"
        if busy
        else "The current model already has a part; creating multiple parts is not supported yet"
        if snapshot.parts
        else "Create a model first"
        if snapshot.source_kind is None
        else "INP models have no editable CAD; create a native model"
    )
    set_state(
        GuiActionKey.GEOMETRY_CREATE,
        create_part_enabled,
        create_part_reason,
    )
    set_state(
        GuiActionKey.GEOMETRY_SKETCH,
        create_part_enabled,
        create_part_reason,
    )
    set_state(
        GuiActionKey.GEOMETRY_FACE_SKETCH,
        has_native_geometry
        and geometry_dimension(recipe) == 3
        and context.planar_solid_face_selected
        and not busy
        and not editor_active,
        (
            "Finish the current geometry edit first"
            if editor_active
            else "Select a valid planar solid face on the current part"
        ),
    )
    set_state(
        GuiActionKey.GEOMETRY_WIRE,
        create_part_enabled,
        create_part_reason,
    )
    for key in (
        GuiActionKey.GEOMETRY_MOVE,
        GuiActionKey.GEOMETRY_ROTATE,
        GuiActionKey.GEOMETRY_MANAGER,
        GuiActionKey.GEOMETRY_DELETE,
        GuiActionKey.GEOMETRY_FUSE,
        GuiActionKey.GEOMETRY_CUT,
    ):
        if is_multi_body:
            is_boolean = key in {
                GuiActionKey.GEOMETRY_FUSE,
                GuiActionKey.GEOMETRY_CUT,
            }
            enabled = (
                has_native_geometry
                and (is_boolean or selected_body_count == 1)
                and not busy
                and not editor_active
            )
            reason = (
                "Finish the current geometry edit first"
                if is_boolean
                else "Select a body first"
            )
            set_state(key, enabled, reason)
            continue
        set_state(
            key,
            has_native_geometry
            and (
                key
                not in {GuiActionKey.GEOMETRY_FUSE, GuiActionKey.GEOMETRY_CUT}
                or geometry_dimension(recipe) in {2, 3}
            )
            and (
                key
                not in {GuiActionKey.GEOMETRY_FUSE, GuiActionKey.GEOMETRY_CUT}
                or not editor_active
            )
            and not busy,
            (
                "Finish the current geometry edit first"
                if editor_active
                else "Fuse and cut require 2D faces or 3D solids"
            ),
        )
    extrude_enabled = False
    extrude_reason = "Create a 2D sketch or planar geometry first"
    if (
        has_native_geometry
        and geometry_dimension(recipe) == 2
        and not busy
    ):
        if context.geometry_selection and any(
            reference.kind != "face"
            for reference in context.geometry_selection
        ):
            extrude_reason = "The current selection contains entities that are not faces"
        else:
            try:
                all_sources = resolve_extrusion_source_faces(recipe)
                requested_sources = (
                    resolve_extrusion_source_faces(
                        recipe,
                        context.geometry_selection,
                    )
                    if context.geometry_selection
                    else None
                )
            except ExtrusionSourceResolutionError as error:
                extrude_reason = {
                    "extrude.source-face.topology-unproven": (
                        "The current 2D topology cannot be safely extruded"
                    ),
                    "extrude.source-face.unknown": (
                        "The selected profile is no longer valid; select it again"
                    ),
                }.get(error.code, str(error))
            else:
                if (
                    len(all_sources.face_ids) > 1
                    and requested_sources is None
                ):
                    extrude_reason = (
                        "This sketch contains multiple profiles; select at least one 2D face first"
                    )
                else:
                    extrude_enabled = True
                    extrude_reason = ""
    set_state(
        GuiActionKey.GEOMETRY_EXTRUDE,
        extrude_enabled,
        extrude_reason,
    )
    set_state(
        GuiActionKey.GEOMETRY_SWEEP,
        extrude_enabled,
        extrude_reason,
    )
    set_state(
        GuiActionKey.GEOMETRY_UNDO,
        (
            selected_body_count == 1
            if is_multi_body
            else isinstance(
                recipe,
                (
                    MovedGeometry,
                    RotatedGeometry,
                    ExtrudedGeometry,
                    RevolvedGeometry,
                    BooleanGeometry,
                ),
            )
        )
        and not busy,
        "No geometry feature is available to undo",
    )
    set_state(
        GuiActionKey.GEOMETRY_REGION,
        has_model and not busy,
        "Generate a mesh first",
    )
    set_state(
        GuiActionKey.GEOMETRY_REGIONS,
        has_model
        and bool(snapshot.named_regions)
        and not busy,
        "Generate a mesh and create a scope first",
    )
    set_state(
        GuiActionKey.MESH_SETTINGS,
        has_native_geometry and not busy,
        "Create a native sketch first; INP models retain their existing mesh and cannot be edited as CAD",
    )
    has_mesh_settings = isinstance(snapshot.mesh_settings, MeshSettings)
    body_relations_ready = not (
        is_multi_body
        and any(
            relation.relation != "disjoint"
            for relation in analyze_body_relations(recipe)
        )
    )
    mesh_inputs_ready = (
        has_native_geometry
        and has_mesh_settings
        and not busy
        and body_relations_ready
    )
    truss_member_policy = bool(
        has_mesh_settings
        and snapshot.mesh_settings.line_element_type == "Truss2"
    )
    truss_controls_conflict = bool(
        truss_member_policy
        and snapshot.mesh_settings.local_controls
    )
    set_state(
        GuiActionKey.MESH_GENERATE,
        mesh_inputs_ready and not truss_controls_conflict,
        (
            "Each wire member uses one truss element; remove local sizes from mesh controls first"
            if truss_controls_conflict
            else (
                "Resolve overlapping or touching bodies with Boolean operations first"
                if not body_relations_ready
                else "Create native geometry and set mesh parameters first"
            )
        ),
    )
    set_state(
        GuiActionKey.MESH_CONTROLS,
        mesh_inputs_ready,
        "Create native geometry and set mesh parameters first",
    )
    set_state(
        GuiActionKey.MESH_LOCAL_CONTROL,
        mesh_inputs_ready and not truss_member_policy,
        (
            "Each wire member uses one truss element; local size controls are not supported"
            if truss_member_policy
            else "Create native geometry and set mesh parameters first"
        ),
    )
    set_state(
        GuiActionKey.MESH_CLEAR,
        snapshot.source_kind == "native" and has_model and not busy,
        "No native mesh is available to clear",
    )
    for key in (
        GuiActionKey.MESH_STATISTICS,
        GuiActionKey.MESH_QUALITY,
        GuiActionKey.MESH_VERIFY,
    ):
        set_state(key, has_model and not busy, "Generate a mesh or open an INP model first")

    set_state(
        GuiActionKey.MATERIAL_MANAGER,
        snapshot.source_kind is not None and not busy,
        "Create a model or open an INP file first",
    )
    section_capability = authoring.report.operation("section.create")
    set_state(
        GuiActionKey.SECTION_MANAGER,
        snapshot.source_kind is not None
        and bool(snapshot.materials)
        and (section_capability.can_enter or bool(snapshot.sections))
        and not busy,
        _capability_reason(section_capability, "Create a model or open an INP file, then create a material"),
    )
    visible_targets = tuple(
        target
        for target in authoring.targets
        if (
            snapshot.source_kind != "native"
            or target.region.name in snapshot.named_regions
        )
    )
    section_targets = tuple(
        target
        for target in visible_targets
        if target.region.kind == "element_set"
        and target.operation("section.assignment").can_submit
    )
    set_state(
        GuiActionKey.SECTION_ASSIGN,
        bool(snapshot.sections)
        and has_model
        and bool(section_targets or snapshot.source_kind == "native")
        and not busy,
        "Generate a mesh or open an INP file, then create a section",
    )

    has_step = bool(authoring.step_lifecycle)
    set_state(
        GuiActionKey.STEP_CREATE,
        snapshot.source_kind is not None and not busy,
        "Create a model or open an INP file first",
    )
    boundary_targets = tuple(
        target
        for target in visible_targets
        if target.operation("boundary.displacement").can_submit
    )
    set_state(
        GuiActionKey.BOUNDARY_CREATE,
        has_step
        and has_model
        and bool(boundary_targets or has_native_geometry)
        and not busy,
        "Create an analysis step and generate a mesh first",
    )
    load_operations = (
        "load.node",
        "load.edge",
        "load.surface",
        "load.body",
        "load.line.global",
        "load.line.local",
    )
    load_targets = tuple(
        target
        for target in visible_targets
        if any(target.operation(name).can_submit for name in load_operations)
    )
    load_reason = (
        "Create an analysis step first"
        if not has_step
        else _first_target_reason(authoring, load_operations)
    )
    set_state(
        GuiActionKey.LOAD_CREATE,
        has_step
        and has_model
        and bool(load_targets or has_native_geometry)
        and not busy,
        load_reason,
    )
    output_create = authoring.operation("output_request.create")
    set_state(
        GuiActionKey.OUTPUT_CREATE,
        output_create.can_submit and not busy,
        _capability_reason(
            output_create,
            "The current session does not allow output requests",
        ),
    )
    set_state(
        GuiActionKey.ANALYSIS_MANAGER,
        bool(snapshot.steps) and not busy,
        "No analysis definitions are available to manage",
    )
    set_state(GuiActionKey.STEP_INFO, has_step and not busy, "No analysis step is available to view")
    lifecycle = authoring.step(context.selected_step_name)
    set_state(
        GuiActionKey.CHECK_MODEL,
        lifecycle is not None and lifecycle.can_check and not busy,
        lifecycle.check_reason if lifecycle is not None else "No analysis step is available to check",
    )
    set_state(
        GuiActionKey.SUBMIT_JOB,
        lifecycle is not None and lifecycle.can_submit and not busy,
        lifecycle.submit_reason if lifecycle is not None else "Pass the model check for the current analysis step first",
    )
    resubmittable = any(
        str(getattr(run.status, "value", run.status)).casefold()
        in {"succeeded", "failed", "cancelled"}
        for run in snapshot.runs
    )
    set_state(
        GuiActionKey.RESUBMIT_JOB,
        not busy and resubmittable,
        "No previous job is available to copy",
    )
    set_state(GuiActionKey.JOB_MANAGER, has_model, "Generate a mesh or open an INP model first")
    set_state(
        GuiActionKey.MODEL_INFO,
        snapshot.source_kind is not None and not busy,
        "No model or project is open",
    )
    for key in (
        GuiActionKey.EDGES,
        GuiActionKey.NODES,
        GuiActionKey.NODE_LABELS,
        GuiActionKey.ELEMENT_LABELS,
        GuiActionKey.SYMBOL_SETTINGS,
    ):
        set_state(key, has_model, "Generate a mesh or open an INP model first")
    set_state(
        GuiActionKey.SYMBOLS,
        has_model and not context.symbol_display_disabled,
        (
            "Constraints and loads cannot be displayed in the current ribbon module"
            if context.symbol_display_disabled
            else "Generate a mesh or open an INP model first"
        ),
    )
    selection_keys = {
        "point": GuiActionKey.SELECT_POINT,
        "element": GuiActionKey.SELECT_ELEMENT,
        "edge": GuiActionKey.SELECT_EDGE,
        "face": GuiActionKey.SELECT_FACE,
        "body": GuiActionKey.SELECT_BODY,
    }
    selection_dimension = context.selection_topological_dimension
    if context.selection_space == "geometry":
        geometry_available = has_native_geometry and not busy
        for kind, key in selection_keys.items():
            enabled = geometry_available and kind != "element"
            reason = (
                "Geometry selection does not support elements"
                if kind == "element"
                else "Create native geometry first"
            )
            if kind == "face" and selection_dimension == 1:
                enabled = False
                reason = "1D geometry has no selectable faces"
            set_state(key, enabled, reason)
    else:
        for kind, key in selection_keys.items():
            enabled = has_model and not busy
            reason = "Generate a mesh or open an INP model first"
            if kind == "face" and selection_dimension == 1:
                enabled = False
                reason = "The current 1D mesh has no selectable topology faces"
            set_state(key, enabled, reason)
    for key in (
        GuiActionKey.FIT,
        GuiActionKey.FRONT,
        GuiActionKey.BACK,
        GuiActionKey.LEFT,
        GuiActionKey.RIGHT,
        GuiActionKey.TOP,
        GuiActionKey.BOTTOM,
        GuiActionKey.ISO,
        GuiActionKey.ORTHOGRAPHIC,
        GuiActionKey.PERSPECTIVE,
        GuiActionKey.CLEAR_SELECTION,
    ):
        set_state(
            key,
            (has_model or has_native_geometry) and not busy,
            "Create geometry, generate a mesh, or open an INP model first",
        )
    set_state(
        GuiActionKey.SELECTED_INFO,
        has_model and context.fem_selection_kind in {"node", "element"},
        "Select nodes or elements first",
    )
    for key in (
        GuiActionKey.UNDEFORMED,
        GuiActionKey.DEFORMED,
        GuiActionKey.CONTOUR,
        GuiActionKey.OVERLAY,
        GuiActionKey.SCALE,
        GuiActionKey.DISPLAY_SETTINGS,
        GuiActionKey.CONTOUR_OPTIONS,
    ):
        set_state(
            key,
            has_result,
            "No analysis results are available to view",
        )
    set_state(
        GuiActionKey.FIELD,
        has_result_catalog and result_actions_idle,
        "The current result catalog is unavailable or a result task is running",
    )
    set_state(
        GuiActionKey.QUERY,
        has_result_catalog and result_actions_idle,
        "The current result catalog is unavailable or a result task is running",
    )
    csv_export_enabled = has_result_catalog and result_actions_idle
    vtk_export_enabled = (
        has_result_catalog
        and context.selected_field_exists
        and context.selected_field_state is FieldState.READY
        and result_actions_idle
    )
    set_state(
        GuiActionKey.EXPORT_CSV,
        csv_export_enabled,
        "The current result catalog is unavailable or a result task is running",
    )
    set_state(
        GuiActionKey.EXPORT_VTK,
        vtk_export_enabled,
        "Select a ready field from the current results and wait for the result task to finish",
    )
    set_state(
        GuiActionKey.SCREENSHOT,
        context.viewport_scene_available
        and context.display_backend_available
        and not context.viewport_capture_active,
        "The viewport has no scene to capture or the screenshot backend is unavailable",
    )

    if editor_active:
        mutation_keys = (
            GuiActionKey.OPEN,
            GuiActionKey.NEW_NATIVE,
            GuiActionKey.DELETE_MODEL,
            GuiActionKey.OPEN_PROJECT,
            GuiActionKey.SAVE_PROJECT,
            GuiActionKey.SAVE_PROJECT_AS,
            GuiActionKey.RELOAD,
            GuiActionKey.CLOSE,
            GuiActionKey.MATERIAL_MANAGER,
            GuiActionKey.SECTION_MANAGER,
            GuiActionKey.SECTION_ASSIGN,
            GuiActionKey.GEOMETRY_CREATE,
            GuiActionKey.GEOMETRY_SKETCH,
            GuiActionKey.GEOMETRY_FACE_SKETCH,
            GuiActionKey.GEOMETRY_WIRE,
            GuiActionKey.GEOMETRY_MOVE,
            GuiActionKey.GEOMETRY_ROTATE,
            GuiActionKey.GEOMETRY_EXTRUDE,
            GuiActionKey.GEOMETRY_SWEEP,
            GuiActionKey.GEOMETRY_FUSE,
            GuiActionKey.GEOMETRY_CUT,
            GuiActionKey.GEOMETRY_MANAGER,
            GuiActionKey.GEOMETRY_UNDO,
            GuiActionKey.GEOMETRY_DELETE,
            GuiActionKey.GEOMETRY_REGION,
            GuiActionKey.GEOMETRY_REGIONS,
            GuiActionKey.MESH_SETTINGS,
            GuiActionKey.MESH_GENERATE,
            GuiActionKey.MESH_CLEAR,
            GuiActionKey.MESH_CONTROLS,
            GuiActionKey.MESH_LOCAL_CONTROL,
            GuiActionKey.MESH_STATISTICS,
            GuiActionKey.MESH_QUALITY,
            GuiActionKey.MESH_VERIFY,
            GuiActionKey.STEP_CREATE,
            GuiActionKey.BOUNDARY_CREATE,
            GuiActionKey.LOAD_CREATE,
            GuiActionKey.OUTPUT_CREATE,
            GuiActionKey.ANALYSIS_MANAGER,
            GuiActionKey.CHECK_MODEL,
            GuiActionKey.SUBMIT_JOB,
            GuiActionKey.RESUBMIT_JOB,
            GuiActionKey.JOB_MANAGER,
        )
        for key in mutation_keys:
            set_state(key, False, "Finish or cancel the current sketch edit first")
        for key in selection_keys.values():
            set_state(key, False, "Finish or cancel the current sketch edit first")
        for key in (
            GuiActionKey.FIT,
            GuiActionKey.TOP,
            GuiActionKey.BOTTOM,
            GuiActionKey.FRONT,
            GuiActionKey.BACK,
            GuiActionKey.LEFT,
            GuiActionKey.RIGHT,
            GuiActionKey.ISO,
            GuiActionKey.ORTHOGRAPHIC,
            GuiActionKey.PERSPECTIVE,
        ):
            set_state(key, not busy, "View controls are unavailable while a background task is running")

    if snapshot.source_kind == "result":
        readonly_reason = "Read-only result documents do not support modeling or analysis edits"
        for key in (
            GuiActionKey.MATERIAL_MANAGER,
            GuiActionKey.SECTION_MANAGER,
            GuiActionKey.SECTION_ASSIGN,
            GuiActionKey.GEOMETRY_CREATE,
            GuiActionKey.GEOMETRY_SKETCH,
            GuiActionKey.GEOMETRY_FACE_SKETCH,
            GuiActionKey.GEOMETRY_WIRE,
            GuiActionKey.GEOMETRY_MOVE,
            GuiActionKey.GEOMETRY_ROTATE,
            GuiActionKey.GEOMETRY_EXTRUDE,
            GuiActionKey.GEOMETRY_SWEEP,
            GuiActionKey.GEOMETRY_FUSE,
            GuiActionKey.GEOMETRY_CUT,
            GuiActionKey.GEOMETRY_MANAGER,
            GuiActionKey.GEOMETRY_UNDO,
            GuiActionKey.GEOMETRY_DELETE,
            GuiActionKey.GEOMETRY_REGION,
            GuiActionKey.GEOMETRY_REGIONS,
            GuiActionKey.MESH_SETTINGS,
            GuiActionKey.MESH_GENERATE,
            GuiActionKey.MESH_CLEAR,
            GuiActionKey.MESH_CONTROLS,
            GuiActionKey.MESH_LOCAL_CONTROL,
            GuiActionKey.STEP_CREATE,
            GuiActionKey.BOUNDARY_CREATE,
            GuiActionKey.LOAD_CREATE,
            GuiActionKey.OUTPUT_CREATE,
            GuiActionKey.ANALYSIS_MANAGER,
            GuiActionKey.CHECK_MODEL,
            GuiActionKey.SUBMIT_JOB,
            GuiActionKey.RESUBMIT_JOB,
            GuiActionKey.JOB_MANAGER,
        ):
            set_state(key, False, readonly_reason)

    for raw_key in context.open_dialog_keys:
        try:
            key = GuiActionKey(raw_key)
        except ValueError:
            continue
        if states[key].enabled:
            set_state(key, False, "This window is already open")

    result = tuple(states[descriptor.key] for descriptor in ACTION_DESCRIPTORS)
    if len(result) != len(GuiActionKey) or {item.key for item in result} != set(GuiActionKey):
        raise RuntimeError("action descriptor registry is incomplete or contains duplicates")
    return result


def _capability_reason(capability: AuthoringCapability, fallback: str) -> str:
    for diagnostic in capability.diagnostics:
        message = diagnostic.remediation or diagnostic.message
        if message:
            return f"[{diagnostic.code}] {message}"
    return fallback


def _first_target_reason(
    projection: SessionAuthoringProjection,
    operations: tuple[str, ...],
) -> str:
    for target in projection.targets:
        for operation in operations:
            capability = target.operation(operation)
            if capability.status in {AuthoringStatus.LIMITED, AuthoringStatus.UNAVAILABLE}:
                reason = _capability_reason(capability, "")
                if reason:
                    return reason
    for diagnostic in projection.report.diagnostics:
        if diagnostic.blocking:
            return f"[{diagnostic.code}] {diagnostic.remediation or diagnostic.message}"
    return "The current capability report has no available load target regions"


__all__ = [
    "ACTION_DESCRIPTORS",
    "ActionAvailability",
    "GuiActionContext",
    "GuiActionDescriptor",
    "GuiActionKey",
    "derive_action_availability",
]
