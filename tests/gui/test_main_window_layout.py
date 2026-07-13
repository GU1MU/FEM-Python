from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QLabel, QMenu, QToolBar, QToolButton

from fem.abaqus import read
from fem.solvers.static_linear import solve
from fem_gui.main_window import FEMMainWindow
from fem_gui.analysis_jobs import AnalysisJob, JobStatus
from fem_gui.visualization.model_adapter import build_model_geometry
from fem_gui.visualization.result_adapter import build_result_data


def _application() -> QApplication:
    return QApplication.instance() or QApplication([])


def test_main_window_has_modules_navigation_and_viewport_toolbar():
    _application()
    window = FEMMainWindow()

    assert [window.ribbon.tab_bar.tabText(i) for i in range(window.ribbon.tab_bar.count())] == [
        "项目", "模型", "分析", "结果", "视图",
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
    window.ribbon.tab_bar.setCurrentIndex(3)
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
    result = solve(model)
    data = build_result_data(result, geometry)
    job = AnalysisJob("Job-1", "Static-1", JobStatus.RUNNING)
    window.document.add_job(job)
    window._job_succeeded(job, (result, data))

    text = []
    root = window.result_tree.invisibleRootItem()
    stack = [root.child(index) for index in range(root.childCount())]
    while stack:
        item = stack.pop()
        text.append(item.text(0))
        stack.extend(item.child(index) for index in range(item.childCount()))
    assert "位移 U" in text
    assert "反力 RF" in text
    assert "应力 S" in text

    window.ribbon.set_current("结果")
    assert window.navigation.tabs.currentWidget() is window.result_tree
    window.result_tree.fieldActivated.emit("U")
    assert window._display.field_key == "U"
    assert window._display.contour_enabled
    assert window.actions["contour"].isChecked()
    window.ribbon.set_current("模型")
    assert window.navigation.tabs.currentWidget() is window.model_tree
    window.close()
