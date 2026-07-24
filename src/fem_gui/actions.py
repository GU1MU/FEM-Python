"""主窗口共享 QAction 的集中注册。"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from PySide6.QtGui import QAction, QActionGroup

from .icons import icon


def build_actions(owner: Any) -> dict[str, QAction]:
    """创建菜单、Ribbon 和视口工具栏共用的动作。"""
    actions: dict[str, QAction] = {}

    def add(
        key: str,
        text: str,
        callback: Callable[..., None],
        *,
        icon_name: str | None = None,
        checkable: bool = False,
        checked: bool = False,
    ) -> QAction:
        action = QAction(icon(icon_name), text, owner) if icon_name else QAction(text, owner)
        action.setObjectName(f"action_{key}")
        action.setToolTip(text)
        action.setStatusTip(text)
        action.setCheckable(checkable)
        action.setChecked(checked)
        action.triggered.connect(callback)
        actions[key] = action
        return action

    add("open", "打开 INP", owner.open_inp, icon_name="open_inp")
    add("new_native", "新建模型", owner.new_native_model, icon_name="new_model")
    add("open_project", "打开项目", owner.open_native_project, icon_name="open_project")
    add("save_project", "保存项目", owner.save_native_project, icon_name="save_project")
    add("reload", "重新加载", owner.reload_model, icon_name="reload")
    add("close", "关闭模型", owner.close_model, icon_name="close")
    add("exit", "退出", owner.close)
    add("model_info", "模型概况", owner.show_model_information, icon_name="model_info")
    add("material_manager", "材料管理", owner.show_material_manager, icon_name="material")
    add("section_manager", "截面管理", owner.show_section_manager, icon_name="section")
    add(
        "section_assign",
        "截面分配",
        owner.assign_section_to_region,
        icon_name="section_assign",
    )

    add(
        "geometry_sketch",
        "新建草图",
        owner.create_sketch_geometry,
        icon_name="sketch",
    )
    add("geometry_move", "移动", owner.move_geometry, icon_name="geometry_move")
    add("geometry_rotate", "旋转", owner.rotate_geometry, icon_name="geometry_rotate")
    add("geometry_extrude", "拉伸", owner.extrude_geometry, icon_name="extrude")
    add("geometry_fuse", "合并", owner.fuse_geometry, icon_name="boolean_fuse")
    add("geometry_cut", "切除", owner.cut_geometry, icon_name="boolean_cut")
    add("geometry_manager", "编辑", owner.show_geometry_manager, icon_name="feature_edit")
    add("geometry_undo", "撤销特征", owner.undo_geometry_feature, icon_name="feature_undo")
    add("geometry_delete", "删除几何", owner.delete_geometry, icon_name="geometry_delete")
    add(
        "geometry_region",
        "创建命名区域",
        owner.create_named_geometry_region,
        icon_name="named_region_create",
    )
    add(
        "geometry_regions",
        "区域管理",
        owner.show_named_region_manager,
        icon_name="named_region_manager",
    )
    add(
        "mesh_settings",
        "网格设置",
        owner.edit_mesh_settings,
        icon_name="mesh_settings",
    )
    add(
        "mesh_generate",
        "生成网格",
        owner.generate_native_mesh,
        icon_name="mesh",
    )
    add("mesh_clear", "清除网格", owner.clear_native_mesh, icon_name="mesh_clear")
    add("mesh_controls", "控制管理", owner.show_mesh_controls, icon_name="mesh_controls")
    add(
        "mesh_local_control",
        "局部网格",
        owner.set_local_mesh_control,
        icon_name="mesh_local_size",
    )
    add("mesh_statistics", "网格统计", owner.show_mesh_statistics, icon_name="mesh_statistics")
    add("mesh_quality", "质量检查", owner.show_mesh_quality, icon_name="mesh_quality")
    add("mesh_verify", "检查网格", owner.show_mesh_verification, icon_name="mesh_verify")

    add("fit", "适合窗口", owner.viewport_fit, icon_name="fit")
    for key, text in (
        ("top", "XY 视图"), ("bottom", "YX 视图"),
        ("front", "XZ 视图"), ("back", "ZX 视图"),
        ("left", "YZ 视图"), ("right", "ZY 视图"),
        ("iso", "XYZ 轴测视图"),
    ):
        add(
            key,
            text,
            lambda _checked=False, view=key: owner.viewport.set_view(view),
            icon_name=key,
        )

    projection = QActionGroup(owner)
    projection.setObjectName("projection_action_group")
    projection.setExclusive(True)
    projection.addAction(add(
        "orthographic", "正交投影",
        lambda checked=False: owner.viewport.set_parallel_projection(True) if checked else None,
        icon_name="orthographic", checkable=True, checked=True,
    ))
    projection.addAction(add(
        "perspective", "透视投影",
        lambda checked=False: owner.viewport.set_parallel_projection(False) if checked else None,
        icon_name="perspective", checkable=True,
    ))
    add(
        "viewport_background", "视口背景", owner.show_viewport_background_dialog,
        icon_name="background",
    )

    add("edges", "显示单元边", owner._toggle_edges, icon_name="edges", checkable=True, checked=True)
    add("nodes", "显示节点", owner._toggle_nodes, icon_name="nodes", checkable=True)
    add("node_labels", "显示节点编号", owner._toggle_node_labels, icon_name="node_ids", checkable=True)
    add("element_labels", "显示单元编号", owner._toggle_element_labels, icon_name="element_ids", checkable=True)
    add("symbols", "显示约束和载荷", owner._toggle_symbols, icon_name="symbols", checkable=True, checked=True)
    add("symbol_settings", "符号设置", owner.show_symbol_settings_dialog, icon_name="settings")

    add("step_info", "分析步信息", owner.show_current_step_information, icon_name="step_info")
    add("step_create", "创建分析步", owner.create_static_step, icon_name="step_create")
    add("boundary_create", "位移边界条件", owner.create_displacement_boundary, icon_name="boundary")
    add("load_create", "创建载荷", owner.create_load, icon_name="load")
    add("output_create", "输出请求", owner.create_output_request, icon_name="output")
    add(
        "analysis_manager",
        "分析管理",
        owner.show_analysis_manager,
        icon_name="analysis_manager",
    )
    add("check_model", "检查模型", owner.start_model_check, icon_name="check")
    add("submit_job", "创建并提交", owner.create_and_submit_job, icon_name="job")
    add("resubmit_job", "重新提交", owner.resubmit_job, icon_name="resubmit")
    add("job_manager", "作业管理器", owner.show_job_manager, icon_name="job_manager")

    shapes = QActionGroup(owner)
    shapes.setObjectName("result_shape_action_group")
    shapes.setExclusive(True)
    shapes.addAction(add(
        "undeformed", "未变形形状", lambda: owner.set_shape_mode("undeformed"),
        icon_name="undeformed", checkable=True, checked=True,
    ))
    shapes.addAction(add(
        "deformed", "变形形状", lambda: owner.set_shape_mode("deformed"),
        icon_name="deformed", checkable=True,
    ))
    add("contour", "显示云图", owner._toggle_contour, icon_name="contour", checkable=True)
    add(
        "overlay", "叠加未变形轮廓", owner._toggle_undeformed_overlay,
        icon_name="overlay", checkable=True,
    )
    add("field", "结果变量和分量", owner.show_result_display_dialog, icon_name="field")
    add("scale", "变形比例", owner.show_result_display_dialog, icon_name="scale")
    add("contour_options", "云图设置", owner.show_contour_dialog, icon_name="settings")
    add("query", "查询结果", owner.query_result, icon_name="query")
    add("export", "导出 CSV", owner.export_csv, icon_name="export")
    add("screenshot", "保存视口图片", owner.export_viewport_image, icon_name="image")
    add("about", "关于", owner.show_about)

    selection = QActionGroup(owner)
    selection.setObjectName("selection_action_group")
    selection.setExclusive(True)
    selection.addAction(add(
        "select_node", "选择节点", lambda: owner._set_selection_mode("node"),
        icon_name="select_node", checkable=True, checked=True,
    ))
    selection.addAction(add(
        "select_element", "选择单元", lambda: owner._set_selection_mode("element"),
        icon_name="select_element", checkable=True,
    ))
    for key, text, icon_name in (
        ("geometry_select_point", "选择点", "select_geometry_point"),
        ("geometry_select_edge", "选择边", "select_geometry_edge"),
        ("geometry_select_face", "选择面", "select_geometry_face"),
        ("geometry_select_body", "选择体", "select_geometry_body"),
    ):
        selection.addAction(add(
            key,
            text,
            lambda _checked=False, mode=key.removeprefix("geometry_select_"): owner._set_geometry_selection_mode(mode),
            icon_name=icon_name,
            checkable=True,
        ))
    add("clear_selection", "清除选择", owner.clear_selection, icon_name="clear_selection")
    add("selected_info", "查看所选信息", owner.show_selected_information, icon_name="inspect")
    return actions
