from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QSize
from PySide6.QtWidgets import QApplication, QLabel, QMenu, QToolBar, QToolButton

from fem.abaqus import read
from fem.application.results import (
    FieldState,
    ResultVariable,
    ScalarFieldSelection,
    build_solve_result_bundle,
)
from fem.solvers.static_linear import solve
from fem_gui.main_window import FEMMainWindow
from fem_gui.visualization.model_adapter import build_model_geometry


def _application() -> QApplication:
    return QApplication.instance() or QApplication([])


def test_main_window_has_modules_navigation_and_viewport_toolbar():
    _application()
    window = FEMMainWindow()

    assert [window.ribbon.tab_bar.tabText(i) for i in range(window.ribbon.tab_bar.count())] == [
        "项目", "几何", "网格", "模型", "分析", "结果", "视图",
    ]
    assert "主页" not in [window.ribbon.tab_bar.tabText(i) for i in range(window.ribbon.tab_bar.count())]
    assert [window.navigation.tabs.tabText(i) for i in range(window.navigation.tabs.count())] == ["模型", "结果"]
    assert window.viewport_panel.toolbar.objectName() == "viewportToolbar"
    assert window.viewport_panel.toolbar.height() == 44
    assert window.viewport._message.text() == ""
    assert window.findChild(QToolBar, "main_toolbar") is None
    assert window.statusBar().height() == 22
    for name in ("statusState", "statusSelection", "statusObject", "statusCoordinate", "statusStep", "statusResult"):
        assert window.statusBar().findChild(QLabel, name) is not None
    window.show()
    window.resize(800, 600)
    QApplication.processEvents()
    assert window.width() == 800
    window.ribbon.set_current("结果")
    QApplication.processEvents()
    variable_y = window.result_variable_combo.mapTo(
        window.ribbon, window.result_variable_combo.rect().topLeft()
    ).y()
    component_y = window.result_component_combo.mapTo(
        window.ribbon, window.result_component_combo.rect().topLeft()
    ).y()
    position_y = window.result_position_combo.mapTo(
        window.ribbon, window.result_position_combo.rect().topLeft()
    ).y()
    assert variable_y == component_y
    assert position_y > variable_y
    window.close()


def test_scope_creation_bar_uses_stable_tray_and_cancel_exits_selection():
    application = _application()
    window = FEMMainWindow()
    window.show()
    window.resize(1000, 700)
    application.processEvents()
    viewport_size = window.viewport.size()
    bar = window.viewport_panel.scope_creation_bar
    window._pending_analysis_selection = "scope"
    window._pending_scope_kind = "node"

    bar.begin("Set", "NodeSet-1")
    application.processEvents()

    assert window.viewport.size() == viewport_size
    tray = window.viewport_panel.scope_creation_tray
    assert window.viewport_panel.layout().indexOf(tray) >= 0
    assert tray.currentWidget() is bar
    assert window.viewport.geometry().bottom() < tray.geometry().top()
    assert bar.cancel_button.text() == "取消"

    bar.cancel_button.click()
    application.processEvents()

    assert bar.isHidden()
    assert window._pending_analysis_selection is None
    assert window._pending_scope_kind is None
    assert window.viewport.size() == viewport_size
    assert tray.currentWidget() is window.viewport_panel._scope_creation_idle
    window.close()


def test_menu_ribbon_and_viewport_toolbar_reuse_actions():
    _application()
    window = FEMMainWindow()
    fit = window.actions["fit"]

    view_menu = window.findChild(QMenu, "menuView")
    assert view_menu is not None
    assert fit in view_menu.actions()
    assert fit in window.viewport_panel.toolbar.actions()
    ribbon_actions = {
        button.defaultAction()
        for button in window.ribbon.findChildren(QToolButton)
        if button.defaultAction() is not None
    }
    assert fit in ribbon_actions
    background = window.actions["viewport_background"]
    assert background in view_menu.actions()
    assert background in ribbon_actions
    assert background not in window.viewport_panel.toolbar.actions()
    selected_info = window.actions["selected_info"]
    assert selected_info in window.viewport_panel.toolbar.actions()
    assert selected_info in ribbon_actions
    result_menu = window.findChild(QMenu, "menuResult")
    assert result_menu is not None
    tab_names = [
        window.ribbon.tab_bar.tabText(index)
        for index in range(window.ribbon.tab_bar.count())
    ]
    project_actions = {
        button.defaultAction()
        for button in window.ribbon.stack.widget(
            tab_names.index("项目")
        ).findChildren(QToolButton)
        if button.defaultAction() is not None
    }
    result_actions = {
        button.defaultAction()
        for button in window.ribbon.stack.widget(
            tab_names.index("结果")
        ).findChildren(QToolButton)
        if button.defaultAction() is not None
    }
    for name in ("export_csv", "export_vtk", "screenshot"):
        export_action = window.actions[name]
        assert export_action in result_menu.actions()
        assert export_action in ribbon_actions
    assert window.actions["export_csv"] in project_actions
    assert window.actions["export_vtk"] in project_actions
    assert window.actions["screenshot"] not in project_actions
    assert window.actions["export_csv"] in result_actions
    assert window.actions["screenshot"] in result_actions
    assert window.actions["export_vtk"] not in result_actions
    window.close()


def test_small_ribbon_commands_use_readable_icons():
    _application()
    window = FEMMainWindow()
    geometry_move = next(
        button
        for button in window.ribbon.findChildren(QToolButton)
        if button.defaultAction() is window.actions["geometry_move"]
    )

    assert geometry_move.iconSize() == QSize(24, 24)
    assert geometry_move.height() == 30
    window.close()


def test_model_page_replaces_clear_selection_with_edge_selection():
    _application()
    window = FEMMainWindow()
    tab_names = [
        window.ribbon.tab_bar.tabText(index)
        for index in range(window.ribbon.tab_bar.count())
    ]
    model_page = window.ribbon.stack.widget(tab_names.index("模型"))
    model_actions = {
        button.defaultAction().objectName()
        for button in model_page.findChildren(QToolButton)
        if button.defaultAction() is not None
    }

    assert model_actions == {
        "action_select_node",
        "action_select_element",
        "action_select_edge",
        "action_selected_info",
        "action_nodes",
        "action_edges",
        "action_node_labels",
        "action_element_labels",
        "action_symbols",
        "action_symbol_settings",
        "action_material_manager",
        "action_section_manager",
        "action_section_assign",
        "action_geometry_region",
        "action_geometry_regions",
    }
    geometry_page = window.ribbon.stack.widget(tab_names.index("几何"))
    mesh_page = window.ribbon.stack.widget(tab_names.index("网格"))
    geometry_actions = {
        button.defaultAction()
        for button in geometry_page.findChildren(QToolButton)
        if button.defaultAction() is not None
    }
    assert geometry_actions == {
        window.actions[name]
        for name in (
            "geometry_create",
            "geometry_extrude",
            "geometry_sweep",
            "geometry_move",
            "geometry_rotate",
            "geometry_fuse",
            "geometry_cut",
            "geometry_manager",
            "geometry_undo",
            "geometry_delete",
            "geometry_select_point",
            "geometry_select_edge",
            "geometry_select_face",
            "geometry_select_body",
        )
    }
    mesh_actions = {
        button.defaultAction()
        for button in mesh_page.findChildren(QToolButton)
        if button.defaultAction() is not None
    }
    assert mesh_actions == {
        window.actions[name]
        for name in (
            "mesh_settings",
            "mesh_local_control",
            "mesh_controls",
            "mesh_generate",
            "mesh_clear",
            "mesh_verify",
            "mesh_statistics",
            "geometry_region",
            "geometry_regions",
        )
    }
    assert window.actions["geometry_region"].text() == "创建作用域"
    assert window.actions["geometry_regions"].text() == "作用域管理"
    window.close()


def test_scope_group_is_available_in_mesh_model_and_analysis_pages():
    _application()
    window = FEMMainWindow()
    tab_names = [
        window.ribbon.tab_bar.tabText(index)
        for index in range(window.ribbon.tab_bar.count())
    ]
    expected_actions = {
        window.actions["geometry_region"],
        window.actions["geometry_regions"],
    }

    for page_name in ("网格", "模型", "分析"):
        page = window.ribbon.stack.widget(tab_names.index(page_name))
        title = next(
            label
            for label in page.findChildren(QLabel, "ribbonGroupTitle")
            if label.text() == "作用域"
        )
        group_actions = {
            button.defaultAction()
            for button in title.parentWidget().findChildren(QToolButton)
            if button.defaultAction() is not None
        }
        assert group_actions == expected_actions

    window.close()


def test_standard_views_use_coordinate_plane_names():
    _application()
    window = FEMMainWindow()
    assert window.actions["top"].text() == "XY 视图"
    assert window.actions["bottom"].text() == "YX 视图"
    assert window.actions["front"].text() == "XZ 视图"
    assert window.actions["back"].text() == "ZX 视图"
    assert window.actions["left"].text() == "YZ 视图"
    assert window.actions["right"].text() == "ZY 视图"
    assert window.actions["iso"].text() == "XYZ 轴测视图"
    window.close()


def test_viewport_toolbar_switches_to_geometry_selection_after_creation():
    _application()
    window = FEMMainWindow()
    from fem.geometry import RectangleGeometry

    window._set_native_geometry(RectangleGeometry("toolbar-geometry", 2.0, 1.0), "矩形")

    geometry_face = window.viewport_panel.toolbar.findChild(
        QToolButton,
        "viewportAction_geometry_select_face",
    )
    model_node = window.viewport_panel.toolbar.findChild(
        QToolButton,
        "viewportAction_select_node",
    )
    assert geometry_face is not None and not geometry_face.isHidden()
    assert geometry_face.defaultAction().isEnabled()
    assert model_node is not None and model_node.isHidden()

    window.ribbon.set_current("模型")
    assert geometry_face.isHidden()
    assert not model_node.isHidden()
    window.close()


def test_viewport_background_updates_placeholder_without_model():
    _application()
    window = FEMMainWindow()
    from fem_gui.viewport_background import ViewportBackgroundSettings

    settings = ViewportBackgroundSettings("solid", "#16191c", "#16191c")
    window.viewport.set_background_settings(settings)

    assert window.viewport._background_settings == settings
    assert window.viewport._background_settings.foreground_color == "#f2f5f7"
    assert "#16191c" in window.viewport._message.styleSheet()
    window.close()


def test_result_navigation_refreshes_and_activates_real_field(gui_inp_path):
    _application()
    window = FEMMainWindow()
    model = read(gui_inp_path)
    geometry = build_model_geometry(model)
    window._model_loaded(gui_inp_path, (model, geometry))
    window._on_viewport_pick("node", 2)
    assert window.status_panel.object_label.text() == "对象：节点 2"
    assert "x=1" in window.status_panel.coordinate_label.text()
    assert window.check_current_model(show_success=False)
    task = window.session.prepare_solve("Static-1", "Job-1")
    if task.delta is not None:
        assert window._apply_session_delta(task.delta)
    assert window._apply_session_delta(window.session.begin_run(task.token))
    result = solve(task.model, task.step_name)
    window._job_succeeded(
        task.token,
        (build_solve_result_bundle(task, result), {}),
    )

    text = []
    root = window.result_tree.invisibleRootItem()
    stack = [root.child(index) for index in range(root.childCount())]
    while stack:
        item = stack.pop()
        text.append(item.text(0))
        stack.extend(item.child(index) for index in range(item.childCount()))
    assert any(label.startswith("位移 U（") for label in text)
    assert any(label.startswith("反力 RF（") for label in text)
    assert any(label.startswith("应力 S（") for label in text)

    window.ribbon.set_current("结果")
    assert window.navigation.tabs.currentWidget() is window.result_tree
    provider = window.result_provider
    assert provider is not None
    reaction = next(
        availability
        for availability in provider.catalog().fields
        if (
            availability.state is FieldState.READY
            and availability.descriptor.field_id.variable
            is ResultVariable.RF
        )
    )
    selection = ScalarFieldSelection(
        reaction.key,
        reaction.descriptor.default_component,
    )
    window.result_tree.fieldSelectionActivated.emit(selection)
    assert window.result_selection == selection
    assert (
        window.viewport._result_render_payload.topology.selection
        == selection
    )
    assert window._display.contour_enabled
    assert window.actions["contour"].isChecked()
    window.ribbon.set_current("模型")
    assert window.navigation.tabs.currentWidget() is window.model_tree
    window.close()
