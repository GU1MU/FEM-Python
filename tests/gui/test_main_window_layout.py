from __future__ import annotations

import gc
import os
import weakref

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QAbstractAnimation, QPoint, QSize, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QLabel,
    QMenu,
    QRubberBand,
    QToolBar,
    QToolButton,
)

from fem_gui.main_window import FEMMainWindow


def _application() -> QApplication:
    return QApplication.instance() or QApplication([])


def test_closed_main_window_releases_python_wrapper(dispose_gui_widget):
    _application()
    window = FEMMainWindow()
    wrapper_reference = weakref.ref(window)

    dispose_gui_widget(window)
    del window
    gc.collect()

    assert wrapper_reference() is None


def test_close_discards_deferred_callbacks_and_stops_owned_timer():
    _application()
    window = FEMMainWindow()
    invoked: list[bool] = []
    window._defer_ui(lambda: invoked.append(True))

    assert window._deferred_ui_timer.isActive()
    assert window.close()
    assert window._deferred_ui_callbacks == []
    assert not window._deferred_ui_timer.isActive()

    QApplication.processEvents()

    assert invoked == []


def test_main_window_has_modules_navigation_and_viewport_toolbar():
    _application()
    window = FEMMainWindow()

    assert not window.main_splitter.opaqueResize()
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
    result_group_titles = {
        label.text()
        for label in window.ribbon.stack.currentWidget().findChildren(QLabel)
        if label.objectName() == "ribbonGroupTitle"
    }
    assert "设置" in result_group_titles
    assert "显示设置" not in result_group_titles
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
    component_text_width = (
        window.result_component_combo.fontMetrics().horizontalAdvance(
            "MaxPrincipal"
        )
    )
    assert (
        window.result_component_combo.minimumWidth()
        >= component_text_width + 44
    )
    assert (
        window.result_component_combo.view().minimumWidth()
        >= window.result_component_combo.minimumWidth()
    )
    window.close()


def test_splitter_resize_uses_preview_line_and_commits_one_repaint():
    application = _application()
    window = FEMMainWindow()
    window.show()
    window.resize(1000, 700)
    application.processEvents()
    repaint_calls = []
    window.viewport.schedule_resize_repaint = lambda: repaint_calls.append(True)
    splitter = window.main_splitter
    handle = splitter.handle(1)
    original_sizes = splitter.sizes()
    start = handle.rect().center()

    QTest.mousePress(
        handle,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
        start,
    )
    QTest.mouseMove(handle, start + QPoint(60, 0))
    application.processEvents()

    assert splitter.sizes() == original_sizes
    assert repaint_calls == []

    QTest.mouseRelease(
        handle,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
        start + QPoint(60, 0),
    )

    assert splitter.sizes() != original_sizes
    assert repaint_calls == []
    application.processEvents()
    assert repaint_calls == [True]
    assert not any(
        band.isVisible()
        for band in splitter.findChildren(QRubberBand)
    )
    window.close()


def test_agent_drawer_resize_previews_then_commits_viewport_geometry():
    application = _application()
    window = FEMMainWindow()
    window.show()
    window.resize(1000, 700)
    application.processEvents()
    repaint_calls = []
    window.viewport.schedule_resize_repaint = lambda: repaint_calls.append(True)
    host = window.viewport_panel.overlay_host
    baseline_width = window.viewport.width()

    host.set_drawer_open(True, animated=False)

    assert window.viewport.width() == baseline_width - host.drawer_width
    assert repaint_calls == [True]
    application.processEvents()
    assert repaint_calls == [True]

    repaint_calls.clear()
    handle = host.agent_chat_drawer.resize_handle
    start = handle.rect().center()
    drawer_width = host.drawer_width
    viewport_width = window.viewport.width()
    QTest.mousePress(
        handle,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
        start,
    )
    QTest.mouseMove(handle, start - QPoint(30, 0))
    application.processEvents()

    assert host._drawer_resize_preview.isVisible()
    assert host.drawer_width == drawer_width
    assert window.viewport.width() == viewport_width
    assert repaint_calls == []

    QTest.mouseRelease(
        handle,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
        start - QPoint(30, 0),
    )

    assert host._drawer_resize_preview.isHidden()
    assert host.drawer_width == drawer_width
    assert window.viewport.width() == viewport_width
    assert repaint_calls == []
    application.processEvents()

    assert host.drawer_width == drawer_width + 30
    assert window.viewport.width() == viewport_width - 30
    assert repaint_calls == [True]
    window.close()


def test_agent_drawer_controls_skip_native_window_animation_frames():
    application = _application()
    window = FEMMainWindow()
    window.show()
    window.resize(1000, 700)
    application.processEvents()
    host = window.viewport_panel.overlay_host
    baseline_width = window.viewport.width()
    animation_frames = []
    geometry_commits = []
    host._animation.valueChanged.connect(animation_frames.append)
    host.viewportGeometryCommitted.connect(lambda: geometry_commits.append(True))

    QTest.mouseClick(
        host.chat_launcher,
        Qt.MouseButton.LeftButton,
    )

    assert host.drawer_is_open
    assert host.agent_chat_drawer.isVisible()
    assert host.chat_launcher.isHidden()
    assert window.viewport.width() == baseline_width - host.drawer_width
    assert host._animation.state() == QAbstractAnimation.State.Stopped
    assert animation_frames == []
    assert geometry_commits == [True]

    QTest.mouseClick(
        host.agent_chat_drawer.close_button,
        Qt.MouseButton.LeftButton,
    )

    assert not host.drawer_is_open
    assert host.agent_chat_drawer.isHidden()
    assert host.chat_launcher.isVisible()
    assert window.viewport.width() == baseline_width
    assert host._animation.state() == QAbstractAnimation.State.Stopped
    assert animation_frames == []
    assert geometry_commits == [True, True]
    window.close()


def test_scope_creation_bar_overlays_viewport_and_cancel_exits_selection():
    application = _application()
    window = FEMMainWindow()
    window.show()
    window.resize(1000, 700)
    application.processEvents()
    viewport_size = window.viewport.size()
    bar = window.viewport_panel.scope_creation_bar
    host = window.viewport_panel.overlay_host
    assert bar.isHidden()
    window._pending_analysis_selection = "scope"
    window._pending_scope_kind = "node"

    bar.begin("Set", "NodeSet-1")
    application.processEvents()

    assert window.viewport.size() == viewport_size
    assert bar.isVisible()
    assert bar.isWindow()
    host_origin = host.mapToGlobal(QPoint(0, 0))
    assert bar.geometry().left() == host_origin.x()
    assert bar.geometry().bottom() == (
        host_origin.y() + host.height() - 1
    )
    assert bar.cancel_button.text() == "取消"

    bar.cancel_button.click()
    application.processEvents()

    assert bar.isHidden()
    assert window._pending_analysis_selection is None
    assert window._pending_scope_kind is None
    assert window.viewport.size() == viewport_size
    window.close()


def test_planar_boolean_face_bar_reuses_the_viewport_bottom_overlay():
    application = _application()
    window = FEMMainWindow()
    window.show()
    window.resize(1000, 700)
    application.processEvents()
    viewport_size = window.viewport.size()
    bar = window.viewport_panel.planar_boolean_face_bar
    host = window.viewport_panel.overlay_host

    bar.begin("cut")
    application.processEvents()

    assert window.viewport.size() == viewport_size
    assert bar.isVisible()
    assert bar.prompt_label.text() == "请选择目标面"
    assert bar.cancel_button.text() == "取消"
    assert bar.confirm_button.text() == "确定"
    assert not bar.confirm_button.isEnabled()
    bar.set_selection_ready(True)
    assert bar.confirm_button.isEnabled()
    host_origin = host.mapToGlobal(QPoint(0, 0))
    assert bar.geometry().left() == host_origin.x()
    assert bar.geometry().bottom() == host_origin.y() + host.height() - 1

    bar.finish()
    application.processEvents()
    assert bar.isHidden()
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
    assert selected_info not in window.viewport_panel.toolbar.actions()
    assert selected_info not in ribbon_actions
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
    project_action_order = [
        button.defaultAction()
        for button in window.ribbon.stack.widget(
            tab_names.index("项目")
        ).findChildren(QToolButton)
        if button.defaultAction() is not None
    ]
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
    assert window.actions["export_csv"] in ribbon_actions
    assert window.actions["screenshot"] in ribbon_actions
    assert window.actions["screenshot"] in window.viewport_panel.toolbar.actions()
    assert window.actions["export_vtk"] not in ribbon_actions
    assert window.actions["export_csv"] in project_actions
    assert window.actions["screenshot"] in project_actions
    assert window.actions["export_vtk"] not in project_actions
    assert window.actions["model_info"] in project_actions
    assert window.actions["submit_job"] not in project_actions
    assert window.actions["save_project_as"] not in project_action_order
    assert window.actions["save_result_as"] not in project_action_order
    project_group_titles = {
        label.text()
        for label in window.ribbon.stack.widget(
            tab_names.index("项目")
        ).findChildren(QLabel, "ribbonGroupTitle")
    }
    assert project_group_titles == {"文件", "输出"}
    assert window.actions["export_csv"] in result_actions
    assert window.actions["screenshot"] in result_actions
    assert window.actions["export_vtk"] not in result_actions
    assert window.actions["display_settings"] in result_menu.actions()
    assert window.actions["display_settings"] in result_actions
    for name in ("undeformed", "deformed", "contour"):
        postprocessing_action = window.actions[name]
        assert postprocessing_action in result_menu.actions()
        assert postprocessing_action in result_actions
        assert postprocessing_action not in window.viewport_panel.toolbar.actions()
    assert window.actions["field"] not in result_menu.actions()
    assert window.actions["field"] not in result_actions
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


def test_geometry_omits_element_selection_while_model_keeps_all_selection_actions():
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
        "action_select_point",
        "action_select_element",
        "action_select_edge",
        "action_select_face",
        "action_select_body",
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
    geometry_group_titles = [
        label.text()
        for label in geometry_page.findChildren(QLabel)
        if label.objectName() == "ribbonGroupTitle"
    ]
    assert geometry_group_titles == ["创建", "特征", "选择"]
    feature_title = next(
        label
        for label in geometry_page.findChildren(QLabel)
        if label.objectName() == "ribbonGroupTitle" and label.text() == "特征"
    )
    feature_actions = {
        button.defaultAction()
        for button in feature_title.parentWidget().findChildren(QToolButton)
        if button.defaultAction() is not None
    }
    assert feature_actions == {
        window.actions[name]
        for name in (
            "geometry_extrude",
            "geometry_sweep",
            "geometry_move",
            "geometry_rotate",
            "geometry_fuse",
            "geometry_cut",
        )
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
            "select_point",
            "select_edge",
            "select_face",
            "select_body",
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


def test_analysis_page_uses_compact_workflow_groups():
    application = _application()
    window = FEMMainWindow()
    window.show()
    window.resize(1600, 700)
    window.ribbon.set_current("分析")
    application.processEvents()
    page = window.ribbon.stack.currentWidget()
    groups = {
        label.text(): label.parentWidget()
        for label in page.findChildren(QLabel, "ribbonGroupTitle")
    }

    assert tuple(groups) == ("分析步", "作用域", "边界条件", "作业")

    def action_button(group_name, action_name):
        return next(
            button
            for button in groups[group_name].findChildren(QToolButton)
            if button.defaultAction() is window.actions[action_name]
        )

    def action_names(group_name):
        return {
            button.defaultAction().objectName().removeprefix("action_")
            for button in groups[group_name].findChildren(QToolButton)
            if button.defaultAction() is not None
        }

    step_create = action_button("分析步", "step_create")
    step_info = action_button("分析步", "step_info")
    output_create = action_button("分析步", "output_create")
    step_combo = groups["分析步"].findChild(QComboBox, "stepCombo_分析")
    assert step_combo is not None
    step_create_pos = step_create.mapTo(groups["分析步"], QPoint(0, 0))
    step_info_pos = step_info.mapTo(groups["分析步"], QPoint(0, 0))
    step_combo_pos = step_combo.mapTo(groups["分析步"], QPoint(0, 0))
    output_pos = output_create.mapTo(groups["分析步"], QPoint(0, 0))
    assert step_create_pos.x() == step_info_pos.x()
    assert step_create_pos.y() < step_info_pos.y()
    assert step_create_pos.x() + step_create.width() < step_combo_pos.x()
    assert step_combo_pos.x() + step_combo.width() < output_pos.x()
    assert action_names("分析步") == {
        "step_create",
        "step_info",
        "output_create",
    }

    boundary = action_button("边界条件", "boundary_create")
    load = action_button("边界条件", "load_create")
    boundary_pos = boundary.mapTo(groups["边界条件"], QPoint(0, 0))
    load_pos = load.mapTo(groups["边界条件"], QPoint(0, 0))
    assert boundary_pos.x() == load_pos.x()
    assert boundary_pos.y() < load_pos.y()
    assert action_names("边界条件") == {"boundary_create", "load_create"}

    check = action_button("作业", "check_model")
    submit = action_button("作业", "submit_job")
    analysis_manager = action_button("作业", "analysis_manager")
    job_manager = action_button("作业", "job_manager")
    check_pos = check.mapTo(groups["作业"], QPoint(0, 0))
    submit_pos = submit.mapTo(groups["作业"], QPoint(0, 0))
    analysis_manager_pos = analysis_manager.mapTo(groups["作业"], QPoint(0, 0))
    job_manager_pos = job_manager.mapTo(groups["作业"], QPoint(0, 0))
    assert check_pos.x() == submit_pos.x()
    assert check_pos.y() < submit_pos.y()
    assert analysis_manager_pos.x() == job_manager_pos.x()
    assert analysis_manager_pos.y() < job_manager_pos.y()
    assert check_pos.x() < analysis_manager_pos.x()
    assert action_names("作业") == {
        "check_model",
        "submit_job",
        "analysis_manager",
        "job_manager",
    }

    window.close()


def test_standard_views_use_abaqus_names():
    _application()
    window = FEMMainWindow()
    assert window.actions["front"].text() == "前视图"
    assert window.actions["back"].text() == "后视图"
    assert window.actions["top"].text() == "俯视图"
    assert window.actions["bottom"].text() == "仰视图"
    assert window.actions["left"].text() == "左视图"
    assert window.actions["right"].text() == "右视图"
    assert window.actions["iso"].text() == "轴测视图"
    window.close()


def test_viewport_toolbar_keeps_isometric_view_first():
    _application()
    window = FEMMainWindow()
    standard_views = {"front", "back", "top", "bottom", "left", "right", "iso"}
    toolbar_views = [
        action.objectName().removeprefix("action_")
        for action in window.viewport_panel.toolbar.actions()
        if action.objectName().removeprefix("action_") in standard_views
    ]
    assert toolbar_views == ["iso", "top", "front"]
    more_views = window.viewport_panel.toolbar.findChild(
        QToolButton,
        "viewportMoreViews",
    )
    assert more_views is not None
    assert more_views.menu() is not None
    assert [
        action.objectName().removeprefix("action_")
        for action in more_views.menu().actions()
    ] == ["bottom", "back", "left", "right"]
    window.close()


def test_viewport_toolbar_keeps_one_shared_five_action_selection_group():
    _application()
    window = FEMMainWindow()
    from fem.geometry import RectangleGeometry

    semantic_names = (
        "select_point", "select_element", "select_edge", "select_face", "select_body",
    )
    toolbar_actions = [
        action.objectName().removeprefix("action_")
        for action in window.viewport_panel.toolbar.actions()
        if action.objectName().removeprefix("action_") in semantic_names
    ]
    assert toolbar_actions == list(semantic_names)

    window.ribbon.set_current("几何")
    window._set_native_geometry(RectangleGeometry("toolbar-geometry", 2.0, 1.0), "矩形")
    window.show()
    _application().processEvents()

    geometry_face = window.viewport_panel.toolbar.findChild(
        QToolButton,
        "viewportAction_select_face",
    )
    model_point = window.viewport_panel.toolbar.findChild(
        QToolButton,
        "viewportAction_select_point",
    )
    assert geometry_face is not None and not geometry_face.isHidden()
    assert geometry_face.defaultAction().isEnabled()
    assert model_point is not None and not model_point.isHidden()
    assert not window.actions["select_element"].isEnabled()
    assert window.actions["select_face"].toolTip() == "选择面"
    assert window.actions["select_body"].text() == "选择体"
    assert window.actions["select_body"].toolTip() == "选择体"

    window.ribbon.set_current("模型")
    assert not geometry_face.isHidden()
    assert not model_point.isHidden()
    assert window._selection_context.space == "mesh"
    window.hide()
    window.close()


def test_module_switch_restores_each_selection_space_filter_without_stale_checks():
    _application()
    window = FEMMainWindow()
    names = ("select_point", "select_element", "select_edge", "select_face", "select_body")
    groups = {window.actions[name].actionGroup() for name in names}
    assert len(groups) == 1

    window.ribbon.set_current("几何")
    window._set_selection_filter("edge")
    assert window.viewport._selection_mode == "geometry_edge"

    window.ribbon.set_current("模型")
    assert window._selection_context.active_filter == "point"
    assert window.viewport._selection_mode == "mesh_node"
    window._set_selection_filter("face")
    assert window.viewport._selection_mode == "mesh_face"

    window.ribbon.set_current("几何")
    assert window._selection_context.active_filter == "edge"
    assert window.viewport._selection_mode == "geometry_edge"
    assert [name for name in names if window.actions[name].isChecked()] == ["select_edge"]
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
