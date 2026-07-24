from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from fem.abaqus import read
from fem.core.model import (
    DisplacementConstraint,
    MaterialDefinition,
    NodalLoad,
)
from fem_gui.analysis_definition_dialogs import (
    DisplacementDialog,
    LoadDialog,
)
from fem_gui.main_window import FEMMainWindow
from fem_gui.model_dialogs import MaterialEditDialog
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
    window.model_tree._on_double_clicked(_find_kind(window.model_tree, "section"))
    info = next(item for item in window._inspection_windows if item.objectName() == "entityInfoDialog")
    assert info.inspection.kind == "section"
    assert info.inspection.key == 0
    window.close()


def test_tree_double_click_edits_imported_material_and_invalidates_model(
    monkeypatch,
    gui_inp_path,
):
    _application()
    window = FEMMainWindow()
    model = read(gui_inp_path)
    window._model_loaded(
        gui_inp_path,
        (model, build_model_geometry(model)),
    )
    monkeypatch.setattr(
        MaterialEditDialog,
        "exec",
        lambda _dialog: True,
    )
    monkeypatch.setattr(
        MaterialEditDialog,
        "material",
        lambda _dialog: MaterialDefinition(
            "STEEL",
            {"E": 123456.0, "nu": 0.28},
        ),
    )
    window.document.workflow.model_checked = True
    window.document.result = object()

    window.model_tree._on_double_clicked(
        _find_kind(window.model_tree, "material")
    )

    assert model.materials["STEEL"].properties == {
        "E": 123456.0,
        "nu": 0.28,
    }
    assert window.document.dirty
    assert not window.document.workflow.model_checked
    assert window.document.result is None
    window.close()


def test_tree_double_click_edits_imported_boundary_and_load(
    monkeypatch,
    gui_inp_path,
):
    _application()
    window = FEMMainWindow()
    model = read(gui_inp_path)
    window._model_loaded(
        gui_inp_path,
        (model, build_model_geometry(model)),
    )
    monkeypatch.setattr(
        DisplacementDialog,
        "exec",
        lambda _dialog: True,
    )
    monkeypatch.setattr(
        DisplacementDialog,
        "definitions",
        lambda _dialog: (
            "Initial",
            (DisplacementConstraint("LEFT", 1, 1, 0.25),),
        ),
    )

    window.model_tree._on_double_clicked(
        _find_kind(window.model_tree, "boundary")
    )

    boundary = model.steps[0].boundaries[0]
    assert (
        boundary.target,
        boundary.first_component,
        boundary.last_component,
        boundary.value,
    ) == ("LEFT", 1, 1, 0.25)

    monkeypatch.setattr(
        LoadDialog,
        "exec",
        lambda _dialog: True,
    )
    monkeypatch.setattr(
        LoadDialog,
        "definition",
        lambda _dialog: (
            "Static-1",
            NodalLoad("RIGHT", 2, 7.5),
        ),
    )
    window.model_tree._on_double_clicked(
        _find_kind(window.model_tree, "cload")
    )

    load_step = next(
        step for step in model.steps if step.name == "Static-1"
    )
    load = load_step.cloads[0]
    assert (load.target, load.component, load.value) == (
        "RIGHT",
        2,
        7.5,
    )
    assert window.document.dirty
    assert not window.document.workflow.model_checked
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
