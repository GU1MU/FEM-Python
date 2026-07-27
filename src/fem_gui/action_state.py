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
    MovedGeometry,
    NATIVE_GEOMETRY_TYPES,
    RotatedGeometry,
    geometry_dimension,
)
from fem.geometry.references import LogicalEntityRef
from fem.mesh.settings import MeshSettings


class GuiActionKey(str, Enum):
    OPEN = "open"
    NEW_NATIVE = "new_native"
    OPEN_PROJECT = "open_project"
    SAVE_PROJECT = "save_project"
    RELOAD = "reload"
    CLOSE = "close"
    EXIT = "exit"
    MODEL_INFO = "model_info"
    MATERIAL_MANAGER = "material_manager"
    SECTION_MANAGER = "section_manager"
    SECTION_ASSIGN = "section_assign"
    GEOMETRY_SKETCH = "geometry_sketch"
    GEOMETRY_MOVE = "geometry_move"
    GEOMETRY_ROTATE = "geometry_rotate"
    GEOMETRY_EXTRUDE = "geometry_extrude"
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
    SCALE = "scale"
    CONTOUR_OPTIONS = "contour_options"
    QUERY = "query"
    EXPORT_CSV = "export_csv"
    EXPORT_VTK = "export_vtk"
    SCREENSHOT = "screenshot"
    ABOUT = "about"
    SELECT_NODE = "select_node"
    SELECT_ELEMENT = "select_element"
    GEOMETRY_SELECT_POINT = "geometry_select_point"
    GEOMETRY_SELECT_EDGE = "geometry_select_edge"
    GEOMETRY_SELECT_FACE = "geometry_select_face"
    GEOMETRY_SELECT_BODY = "geometry_select_body"
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
    _d(GuiActionKey.OPEN, "打开 INP", "open_inp", "open_inp"),
    _d(GuiActionKey.NEW_NATIVE, "新建模型", "new_native_model", "new_model"),
    _d(GuiActionKey.OPEN_PROJECT, "打开项目", "open_native_project", "open_project"),
    _d(GuiActionKey.SAVE_PROJECT, "保存项目", "save_native_project", "save_project"),
    _d(GuiActionKey.RELOAD, "重新加载", "reload_model", "reload"),
    _d(GuiActionKey.CLOSE, "关闭模型", "close_model", "close"),
    _d(GuiActionKey.EXIT, "退出", "close"),
    _d(GuiActionKey.MODEL_INFO, "模型概况", "show_model_information", "model_info"),
    _d(GuiActionKey.MATERIAL_MANAGER, "材料管理", "show_material_manager", "material"),
    _d(GuiActionKey.SECTION_MANAGER, "截面管理", "show_section_manager", "section"),
    _d(GuiActionKey.SECTION_ASSIGN, "截面分配", "assign_section_to_region", "section_assign"),
    _d(GuiActionKey.GEOMETRY_SKETCH, "新建草图", "create_sketch_geometry", "sketch"),
    _d(GuiActionKey.GEOMETRY_MOVE, "移动", "move_geometry", "geometry_move"),
    _d(GuiActionKey.GEOMETRY_ROTATE, "旋转", "rotate_geometry", "geometry_rotate"),
    _d(GuiActionKey.GEOMETRY_EXTRUDE, "拉伸", "extrude_geometry", "extrude"),
    _d(GuiActionKey.GEOMETRY_FUSE, "合并", "fuse_geometry", "boolean_fuse"),
    _d(GuiActionKey.GEOMETRY_CUT, "切除", "cut_geometry", "boolean_cut"),
    _d(GuiActionKey.GEOMETRY_MANAGER, "编辑", "show_geometry_manager", "feature_edit"),
    _d(GuiActionKey.GEOMETRY_UNDO, "撤销特征", "undo_geometry_feature", "feature_undo"),
    _d(GuiActionKey.GEOMETRY_DELETE, "删除几何", "delete_geometry", "geometry_delete"),
    _d(GuiActionKey.GEOMETRY_REGION, "创建命名区域", "create_named_geometry_region", "named_region_create"),
    _d(GuiActionKey.GEOMETRY_REGIONS, "区域管理", "show_named_region_manager", "named_region_manager"),
    _d(GuiActionKey.MESH_SETTINGS, "网格设置", "edit_mesh_settings", "mesh_settings"),
    _d(GuiActionKey.MESH_GENERATE, "生成网格", "generate_native_mesh", "mesh"),
    _d(GuiActionKey.MESH_CLEAR, "清除网格", "clear_native_mesh", "mesh_clear"),
    _d(GuiActionKey.MESH_CONTROLS, "控制管理", "show_mesh_controls", "mesh_controls"),
    _d(GuiActionKey.MESH_LOCAL_CONTROL, "局部网格", "set_local_mesh_control", "mesh_local_control"),
    _d(GuiActionKey.MESH_STATISTICS, "网格统计", "show_mesh_statistics", "mesh_statistics"),
    _d(GuiActionKey.MESH_QUALITY, "质量检查", "show_mesh_quality", "mesh_quality"),
    _d(GuiActionKey.MESH_VERIFY, "检查网格", "show_mesh_verification", "mesh_verify"),
    _d(GuiActionKey.FIT, "适合窗口", "viewport_fit", "fit"),
    _d(GuiActionKey.TOP, "XY 视图", "viewport.set_view", "top", argument="top"),
    _d(GuiActionKey.BOTTOM, "YX 视图", "viewport.set_view", "bottom", argument="bottom"),
    _d(GuiActionKey.FRONT, "XZ 视图", "viewport.set_view", "front", argument="front"),
    _d(GuiActionKey.BACK, "ZX 视图", "viewport.set_view", "back", argument="back"),
    _d(GuiActionKey.LEFT, "YZ 视图", "viewport.set_view", "left", argument="left"),
    _d(GuiActionKey.RIGHT, "ZY 视图", "viewport.set_view", "right", argument="right"),
    _d(GuiActionKey.ISO, "XYZ 轴测视图", "viewport.set_view", "iso", argument="iso"),
    _d(GuiActionKey.ORTHOGRAPHIC, "正交投影", "viewport.set_parallel_projection", "orthographic", checkable=True, checked=True, group="projection", argument=True, checked_only=True),
    _d(GuiActionKey.PERSPECTIVE, "透视投影", "viewport.set_parallel_projection", "perspective", checkable=True, group="projection", argument=False, checked_only=True),
    _d(GuiActionKey.VIEWPORT_BACKGROUND, "视口背景", "show_viewport_background_dialog", "background"),
    _d(GuiActionKey.EDGES, "显示单元边", "_toggle_edges", "edges", checkable=True, checked=True),
    _d(GuiActionKey.NODES, "显示节点", "_toggle_nodes", "nodes", checkable=True),
    _d(GuiActionKey.NODE_LABELS, "显示节点编号", "_toggle_node_labels", "node_ids", checkable=True),
    _d(GuiActionKey.ELEMENT_LABELS, "显示单元编号", "_toggle_element_labels", "element_ids", checkable=True),
    _d(GuiActionKey.SYMBOLS, "显示约束和载荷", "_toggle_symbols", "symbols", checkable=True, checked=True),
    _d(GuiActionKey.SYMBOL_SETTINGS, "符号设置", "show_symbol_settings_dialog", "settings"),
    _d(GuiActionKey.STEP_INFO, "分析步信息", "show_current_step_information", "step_info"),
    _d(GuiActionKey.STEP_CREATE, "创建分析步", "create_static_step", "step_create"),
    _d(GuiActionKey.BOUNDARY_CREATE, "位移边界条件", "create_displacement_boundary", "boundary"),
    _d(GuiActionKey.LOAD_CREATE, "创建载荷", "create_load", "load"),
    _d(GuiActionKey.OUTPUT_CREATE, "输出请求", "create_output_request", "output"),
    _d(GuiActionKey.ANALYSIS_MANAGER, "分析管理", "show_analysis_manager", "analysis_manager"),
    _d(GuiActionKey.CHECK_MODEL, "检查模型", "start_model_check", "check"),
    _d(GuiActionKey.SUBMIT_JOB, "创建并提交", "create_and_submit_job", "job"),
    _d(GuiActionKey.RESUBMIT_JOB, "重新提交", "resubmit_job", "resubmit"),
    _d(GuiActionKey.JOB_MANAGER, "作业管理器", "show_job_manager", "job_manager"),
    _d(GuiActionKey.UNDEFORMED, "未变形形状", "set_shape_mode", "undeformed", checkable=True, checked=True, group="shape", argument="undeformed"),
    _d(GuiActionKey.DEFORMED, "变形形状", "set_shape_mode", "deformed", checkable=True, group="shape", argument="deformed"),
    _d(GuiActionKey.CONTOUR, "显示云图", "_toggle_contour", "contour", checkable=True),
    _d(GuiActionKey.OVERLAY, "叠加未变形轮廓", "_toggle_undeformed_overlay", "overlay", checkable=True),
    _d(GuiActionKey.FIELD, "结果变量和分量", "show_result_display_dialog", "field"),
    _d(GuiActionKey.SCALE, "变形比例", "show_result_display_dialog", "scale"),
    _d(GuiActionKey.CONTOUR_OPTIONS, "云图设置", "show_contour_dialog", "settings"),
    _d(
        GuiActionKey.QUERY,
        "查询结果",
        "show_result_query_dialog",
        "query",
    ),
    _d(GuiActionKey.EXPORT_CSV, "导出 CSV", "export_csv", "export"),
    _d(GuiActionKey.EXPORT_VTK, "导出 VTK", "export_vtk", "export"),
    _d(GuiActionKey.SCREENSHOT, "保存视口图片", "export_viewport_image", "image"),
    _d(GuiActionKey.ABOUT, "关于", "show_about"),
    _d(GuiActionKey.SELECT_NODE, "选择节点", "_set_selection_mode", "select_node", checkable=True, checked=True, group="selection", argument="node"),
    _d(GuiActionKey.SELECT_ELEMENT, "选择单元", "_set_selection_mode", "select_element", checkable=True, group="selection", argument="element"),
    _d(GuiActionKey.GEOMETRY_SELECT_POINT, "选择点", "_set_geometry_selection_mode", "select_geometry_point", checkable=True, group="selection", argument="point"),
    _d(GuiActionKey.GEOMETRY_SELECT_EDGE, "选择边", "_set_geometry_selection_mode", "select_geometry_edge", checkable=True, group="selection", argument="edge"),
    _d(GuiActionKey.GEOMETRY_SELECT_FACE, "选择面", "_set_geometry_selection_mode", "select_geometry_face", checkable=True, group="selection", argument="face"),
    _d(GuiActionKey.GEOMETRY_SELECT_BODY, "选择体", "_set_geometry_selection_mode", "select_geometry_body", checkable=True, group="selection", argument="body"),
    _d(GuiActionKey.CLEAR_SELECTION, "清除选择", "clear_selection", "clear_selection"),
    _d(GuiActionKey.SELECTED_INFO, "查看所选信息", "show_selected_information", "inspect"),
)


@dataclass(frozen=True, slots=True)
class GuiActionContext:
    busy: bool = False
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

    def __post_init__(self) -> None:
        for name in (
            "busy",
            "display_backend_available",
            "viewport_capture_active",
            "result_source_current",
            "catalog_available",
            "selected_field_exists",
            "materialization_pending",
            "result_task_busy",
            "viewport_scene_available",
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
    has_native_geometry = (
        snapshot.source_kind == "native"
        and isinstance(recipe, NATIVE_GEOMETRY_TYPES)
    )

    set_state(GuiActionKey.OPEN, not busy, "后台任务运行时不能打开 INP")
    set_state(GuiActionKey.NEW_NATIVE, not busy, "后台任务运行时不能新建项目")
    set_state(GuiActionKey.OPEN_PROJECT, not busy, "后台任务运行时不能打开项目")
    set_state(
        GuiActionKey.SAVE_PROJECT,
        snapshot.can_save and not busy,
        "请先创建自主草图或几何；INP 模型保持原文件工作流",
    )
    set_state(
        GuiActionKey.RELOAD,
        snapshot.can_reload and not busy,
        "只有已打开的 INP 模型可以重新加载",
    )
    set_state(
        GuiActionKey.CLOSE,
        snapshot.source_kind is not None and not busy,
        "当前没有打开的模型或项目",
    )
    geometry_reason = (
        "请先新建模型"
        if snapshot.source_kind is None
        else "INP 模型没有可编辑 CAD；请新建自主模型"
    )
    set_state(
        GuiActionKey.GEOMETRY_SKETCH,
        snapshot.source_kind == "native" and not busy,
        geometry_reason,
    )
    for key in (
        GuiActionKey.GEOMETRY_MOVE,
        GuiActionKey.GEOMETRY_ROTATE,
        GuiActionKey.GEOMETRY_MANAGER,
        GuiActionKey.GEOMETRY_DELETE,
        GuiActionKey.GEOMETRY_FUSE,
        GuiActionKey.GEOMETRY_CUT,
    ):
        set_state(key, has_native_geometry and not busy, "请先创建自主几何")
    set_state(
        GuiActionKey.GEOMETRY_EXTRUDE,
        has_native_geometry and geometry_dimension(recipe) == 2 and not busy,
        "请先创建二维草图或平面几何",
    )
    set_state(
        GuiActionKey.GEOMETRY_UNDO,
        isinstance(
            recipe,
            (MovedGeometry, RotatedGeometry, ExtrudedGeometry, BooleanGeometry),
        ) and not busy,
        "当前没有可撤销的几何特征",
    )
    for key in (
        GuiActionKey.GEOMETRY_SELECT_POINT,
        GuiActionKey.GEOMETRY_SELECT_EDGE,
        GuiActionKey.GEOMETRY_SELECT_FACE,
        GuiActionKey.GEOMETRY_SELECT_BODY,
    ):
        set_state(key, has_native_geometry and not busy, "请先创建自主几何")
    set_state(
        GuiActionKey.GEOMETRY_REGION,
        has_native_geometry and bool(context.geometry_selection) and not busy,
        "请先在视口中选择点、边、面或体",
    )
    set_state(
        GuiActionKey.GEOMETRY_REGIONS,
        has_native_geometry and bool(snapshot.named_regions) and not busy,
        "当前没有命名区域",
    )
    set_state(
        GuiActionKey.MESH_SETTINGS,
        has_native_geometry and not busy,
        "请先创建自主草图；INP 模型保留已有网格，不能反向编辑 CAD",
    )
    has_mesh_settings = isinstance(snapshot.mesh_settings, MeshSettings)
    for key in (
        GuiActionKey.MESH_GENERATE,
        GuiActionKey.MESH_CONTROLS,
        GuiActionKey.MESH_LOCAL_CONTROL,
    ):
        set_state(
            key,
            has_native_geometry and has_mesh_settings and not busy,
            "请先创建自主几何并设置网格参数",
        )
    set_state(
        GuiActionKey.MESH_CLEAR,
        snapshot.source_kind == "native" and has_model and not busy,
        "当前没有可清除的自主网格",
    )
    for key in (
        GuiActionKey.MESH_STATISTICS,
        GuiActionKey.MESH_QUALITY,
        GuiActionKey.MESH_VERIFY,
    ):
        set_state(key, has_model and not busy, "请先生成网格或打开 INP 模型")

    set_state(
        GuiActionKey.MATERIAL_MANAGER,
        snapshot.source_kind is not None and not busy,
        "请先新建模型或打开 INP",
    )
    section_capability = authoring.report.operation("section.create")
    set_state(
        GuiActionKey.SECTION_MANAGER,
        snapshot.source_kind is not None
        and bool(snapshot.materials)
        and (section_capability.can_enter or bool(snapshot.sections))
        and not busy,
        _capability_reason(section_capability, "请先新建模型或打开 INP，并创建材料"),
    )
    section_targets = tuple(
        target
        for target in authoring.targets
        if target.region.kind == "element_set"
        and target.operation("section.assignment").can_submit
    )
    set_state(
        GuiActionKey.SECTION_ASSIGN,
        bool(snapshot.sections)
        and bool(section_targets or has_model or has_native_geometry)
        and not busy,
        "请先创建几何或打开 INP，并创建截面",
    )

    has_step = bool(authoring.step_lifecycle)
    set_state(
        GuiActionKey.STEP_CREATE,
        snapshot.source_kind is not None and not busy,
        "请先新建模型或打开 INP",
    )
    boundary_targets = tuple(
        target
        for target in authoring.targets
        if target.operation("boundary.displacement").can_submit
    )
    set_state(
        GuiActionKey.BOUNDARY_CREATE,
        has_step and bool(boundary_targets or has_native_geometry) and not busy,
        "请先创建分析步，并准备可选择的几何或节点区域",
    )
    load_operations = (
        "load.node",
        "load.edge",
        "load.surface",
        "load.line.global",
        "load.line.local",
    )
    load_targets = tuple(
        target
        for target in authoring.targets
        if any(target.operation(name).can_submit for name in load_operations)
    )
    load_reason = (
        "请先创建分析步"
        if not has_step
        else _first_target_reason(authoring, load_operations)
    )
    set_state(
        GuiActionKey.LOAD_CREATE,
        has_step and bool(load_targets or has_native_geometry) and not busy,
        load_reason,
    )
    output_create = authoring.operation("output_request.create")
    set_state(
        GuiActionKey.OUTPUT_CREATE,
        output_create.can_submit and not busy,
        _capability_reason(
            output_create,
            "当前 Session 不允许创建输出请求",
        ),
    )
    set_state(
        GuiActionKey.ANALYSIS_MANAGER,
        bool(snapshot.steps) and not busy,
        "当前没有可管理的分析定义",
    )
    set_state(GuiActionKey.STEP_INFO, has_step and not busy, "当前没有可查看的分析步")
    lifecycle = authoring.step(context.selected_step_name)
    set_state(
        GuiActionKey.CHECK_MODEL,
        lifecycle is not None and lifecycle.can_check and not busy,
        lifecycle.check_reason if lifecycle is not None else "当前没有可检查的分析步",
    )
    set_state(
        GuiActionKey.SUBMIT_JOB,
        lifecycle is not None and lifecycle.can_submit and not busy,
        lifecycle.submit_reason if lifecycle is not None else "请先通过当前分析步的模型检查",
    )
    resubmittable = any(
        str(getattr(run.status, "value", run.status)).casefold()
        in {"succeeded", "failed", "cancelled"}
        for run in snapshot.runs
    )
    set_state(
        GuiActionKey.RESUBMIT_JOB,
        not busy and resubmittable,
        "当前没有已完成或失败的作业可重新提交",
    )
    set_state(GuiActionKey.JOB_MANAGER, has_model, "请先生成网格或打开 INP 模型")
    set_state(
        GuiActionKey.MODEL_INFO,
        snapshot.source_kind is not None and not busy,
        "当前没有打开的模型或项目",
    )
    for key in (
        GuiActionKey.EDGES,
        GuiActionKey.NODES,
        GuiActionKey.NODE_LABELS,
        GuiActionKey.ELEMENT_LABELS,
        GuiActionKey.SELECT_NODE,
        GuiActionKey.SELECT_ELEMENT,
        GuiActionKey.SYMBOLS,
        GuiActionKey.SYMBOL_SETTINGS,
    ):
        set_state(key, has_model, "请先生成网格或打开 INP 模型")
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
            "请先创建几何、生成网格或打开 INP 模型",
        )
    set_state(
        GuiActionKey.SELECTED_INFO,
        has_model and context.fem_selection_kind in {"node", "element"},
        "请先选择节点或单元",
    )
    for key in (
        GuiActionKey.UNDEFORMED,
        GuiActionKey.DEFORMED,
        GuiActionKey.CONTOUR,
        GuiActionKey.OVERLAY,
        GuiActionKey.SCALE,
        GuiActionKey.CONTOUR_OPTIONS,
    ):
        set_state(
            key,
            has_result,
            "当前没有可查看的分析结果",
        )
    set_state(
        GuiActionKey.FIELD,
        has_result_catalog and result_actions_idle,
        "当前结果目录不可用，或结果任务正在运行",
    )
    set_state(
        GuiActionKey.QUERY,
        has_result_catalog and result_actions_idle,
        "当前结果目录不可用，或结果任务正在运行",
    )
    result_export_enabled = (
        has_result_catalog
        and context.selected_field_exists
        and context.selected_field_state is FieldState.READY
        and result_actions_idle
    )
    for key in (GuiActionKey.EXPORT_CSV, GuiActionKey.EXPORT_VTK):
        set_state(
            key,
            result_export_enabled,
            "请选择已就绪的当前结果字段，并等待结果任务完成",
        )
    set_state(
        GuiActionKey.SCREENSHOT,
        context.viewport_scene_available
        and context.display_backend_available
        and not context.viewport_capture_active,
        "当前视口没有可捕获场景，或截图后端不可用",
    )

    for raw_key in context.open_dialog_keys:
        try:
            key = GuiActionKey(raw_key)
        except ValueError:
            continue
        if states[key].enabled:
            set_state(key, False, "该窗口已经打开")

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
    return "当前 capability report 没有可用的载荷目标区域"


__all__ = [
    "ACTION_DESCRIPTORS",
    "ActionAvailability",
    "GuiActionContext",
    "GuiActionDescriptor",
    "GuiActionKey",
    "derive_action_availability",
]
