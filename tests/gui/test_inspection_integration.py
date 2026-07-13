from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from fem.abaqus import read
from fem_gui.main_window import FEMMainWindow
from fem_gui.visualization.model_adapter import build_model_geometry
from fem_gui.widgets.model_tree import ROLE_KIND


def _application() -> QApplication:
    return QApplication.instance() or QApplication([])


def _find_kind(tree, kind):
    root = tree.invisibleRootItem()
    stack = [root.child(index) for index in range(root.childCount())]
    while stack:
        item = stack.pop()
        if item.data(0, ROLE_KIND) == kind:
            return item
        stack.extend(item.child(index) for index in range(item.childCount()))
    raise AssertionError(kind)


def test_tree_double_click_opens_mesh_browser_and_entity_dialog(gui_inp_path):
    _application()
    window = FEMMainWindow()
    model = read(gui_inp_path)
    window._model_loaded(gui_inp_path, (model, build_model_geometry(model)))

    window.model_tree._on_double_clicked(_find_kind(window.model_tree, "mesh"))
    assert window._mesh_browser is not None
    assert window._mesh_browser.node_model.rowCount() == 4
    window.model_tree._on_double_clicked(_find_kind(window.model_tree, "material"))
    info = next(item for item in window._inspection_windows if item.objectName() == "entityInfoDialog")
    assert info.inspection.kind == "material"
    assert info.inspection.key == "STEEL"
    window.close()


def test_selected_information_action_and_window_lifecycle(gui_inp_path):
    _application()
    window = FEMMainWindow()
    model = read(gui_inp_path)
    window._model_loaded(gui_inp_path, (model, build_model_geometry(model)))
    window._update_action_states()
    assert not window.actions["selected_info"].isEnabled()

    window._on_viewport_pick("node", 2)
    assert window.actions["selected_info"].isEnabled()
    dialog = window.show_selected_information()
    browser = window.show_mesh_browser()
    assert dialog.inspection.kind == "node"
    assert browser is window._mesh_browser
    assert window._inspection_windows

    window.close_model()
    QApplication.processEvents()
    assert window.inspection_service is None
    assert window._mesh_browser is None
    assert window._inspection_windows == []
    assert not window.actions["selected_info"].isEnabled()
    window.close()
