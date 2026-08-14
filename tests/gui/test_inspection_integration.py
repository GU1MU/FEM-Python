from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from fem.io.inp import read
from fem.application import MeshEntityRef, NativePart, RegionRef
from fem.application.results import build_solve_result_bundle
from fem.core.model import (
    AnalysisStep,
    DisplacementConstraint,
    Edge,
    EdgeLoad,
    ElementEdge,
    FEMModel,
    MaterialDefinition,
    NodalLoad,
)
from fem.solvers import static_linear
from fem.geometry import BooleanGeometry, DiskGeometry, RectangleGeometry
import fem_gui.main_window as main_window_module
from fem_gui.analysis_definition_dialogs import (
    DisplacementDialog,
    LoadDialog,
)
from fem_gui.main_window import FEMMainWindow
from fem_gui.model_dialogs import MaterialEditDialog
from fem_gui.visualization.model_adapter import build_model_geometry
from fem_gui.widgets.model_tree import ROLE_KEY, ROLE_KIND
from tests.helpers.mesh_builders import make_selection_hex_mesh
from tests.helpers.preflight_builders import passing_preflight_report


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


def _seed_current_result(window: FEMMainWindow, step_name: str) -> None:
    validation = window.session.prepare_validation(step_name)
    assert window._apply_session_delta(
        window.session.accept_validation(
            validation.token,
            passing_preflight_report(validation.token),
        )
    )
    solve = window.session.prepare_solve(step_name, "Seed-Result")
    if solve.delta is not None:
        assert window._apply_session_delta(solve.delta)
    assert window._apply_session_delta(window.session.begin_run(solve.token))
    result = static_linear.solve(solve.model, step_name)
    window._job_succeeded(
        solve.token,
        (build_solve_result_bundle(solve, result), {}),
    )
    assert window.session.current_result() is not None


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


def test_native_tree_keeps_model_name_and_supports_info_and_renames(
    monkeypatch,
):
    _application()
    window = FEMMainWindow()
    window._apply_session_delta(
        window.session.new_native_project("Model-1")
    )
    recipe = RectangleGeometry("Sketch-1", 2.0, 1.0)
    window._apply_session_delta(
        window.session.replace_native_geometry_inputs(
            (NativePart(geometry_recipe=recipe),),
            recipe,
        )
    )

    root = window.model_tree.topLevelItem(0)
    part = root.child(0)
    feature = part.child(0)
    part_key = part.data(0, ROLE_KEY)
    assert root.text(0) == "Model-1"
    assert part.text(0) == "Part-1"

    information = []
    monkeypatch.setattr(
        window,
        "_show_information",
        lambda title, rows: information.append((title, rows)),
    )
    window._show_entry_information("model", None)
    window._show_entry_information("part", part_key)
    window._show_entry_information(
        "feature",
        feature.data(0, ROLE_KEY),
    )
    assert [title for title, _rows in information] == [
        "模型概况",
        "部件信息",
        "特征信息",
    ]

    names = iter((("Bracket", True), ("Mount", True)))
    monkeypatch.setattr(
        main_window_module.QInputDialog,
        "getText",
        lambda *_args, **_kwargs: next(names),
    )
    window._rename_tree_entry("model", None)
    window._rename_tree_entry("part", part_key)

    assert window.document.model_name == "Bracket"
    assert window.document.parts[0].name == "Mount"
    assert window.model_tree.topLevelItem(0).text(0) == "Bracket"
    assert window.model_tree.topLevelItem(0).child(0).text(0) == "Mount"
    window.close()


def test_native_feature_information_uses_tree_labels_without_summary(
    monkeypatch,
):
    _application()
    window = FEMMainWindow()
    window._apply_session_delta(
        window.session.new_native_project("模型-1")
    )
    recipe = BooleanGeometry(
        "Cut",
        "cut",
        RectangleGeometry("Plate", 4.0, 2.0),
        DiskGeometry("Hole", 0.25),
    )
    window._apply_session_delta(
        window.session.replace_native_geometry_inputs(
            (NativePart(geometry_recipe=recipe),),
            recipe,
        )
    )

    part = window.model_tree.topLevelItem(0).child(0)
    feature = part.child(part.childCount() - 2)
    assert feature.text(0) == "切除-1"
    assert feature.toolTip(0) == ""

    information = []
    monkeypatch.setattr(
        window,
        "_show_information",
        lambda title, rows: information.append((title, rows)),
    )
    window._show_entry_information(
        "feature",
        feature.data(0, ROLE_KEY),
    )

    assert information == [
        ("特征信息", [("名称", "切除-1"), ("类型", "切除")])
    ]
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
    _seed_current_result(window, "Static-1")
    monkeypatch.setattr(
        window,
        "_confirm_result_invalidation",
        lambda **_kwargs: True,
    )

    window.model_tree._on_double_clicked(
        _find_kind(window.model_tree, "material")
    )

    assert window.document.model.materials["STEEL"].properties == {
        "E": 123456.0,
        "nu": 0.28,
    }
    assert window.document.dirty
    assert window.session.validation_for("Static-1") is None
    assert window.session.current_result() is None
    assert not window.document.has_result
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

    boundary = window.document.model.steps[0].boundaries[0]
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
        step
        for step in window.document.model.steps
        if step.name == "Static-1"
    )
    load = load_step.cloads[0]
    assert (load.target, load.component, load.value) == (
        "RIGHT",
        2,
        7.5,
    )
    assert window.document.dirty
    assert not window.session.can_submit("Static-1")
    window.close()


def test_tree_boundary_click_reuses_mesh_scope_selection_highlight(monkeypatch) -> None:
    _application()
    model = FEMModel(
        make_selection_hex_mesh(),
        edges={
            "FIXED_EDGE": Edge(
                "FIXED_EDGE",
                (ElementEdge(1, 4, (5, 6)),),
            )
        },
        steps=(
            AnalysisStep(
                "Initial",
                boundaries=(
                    DisplacementConstraint(
                        "FIXED_EDGE",
                        1,
                        3,
                        target_kind="edge",
                        name="Fixed",
                    ),
                ),
                edge_loads=(
                    EdgeLoad(
                        "FIXED_EDGE",
                        (2.0, 0.0, 0.0),
                        name="Edge traction",
                    ),
                ),
            ),
        ),
    )
    window = FEMMainWindow()
    window._model_loaded(
        Path("edge-boundary.inp"),
        (model, build_model_geometry(model)),
    )
    highlighted_scopes = []
    highlighted_nodes = []
    window.viewport._actors["set_highlight"] = object()
    monkeypatch.setattr(
        window.viewport,
        "highlight_mesh_entities",
        lambda references, **options: highlighted_scopes.append(
            (tuple(references), options)
        ),
    )
    monkeypatch.setattr(
        window.viewport,
        "highlight_nodes",
        lambda node_ids: highlighted_nodes.append(node_ids),
    )

    window.model_tree._on_clicked(
        _find_kind(window.model_tree, "boundary")
    )

    expected = (
        MeshEntityRef.edge(1, 4, (5, 6)),
    )
    assert highlighted_scopes == [
        (expected, {"entity_kind": "edge"})
    ]
    assert highlighted_nodes == []
    assert "set_highlight" not in window.viewport._actors

    window.model_tree._on_clicked(
        _find_kind(window.model_tree, "edge_load")
    )

    assert highlighted_scopes[-1] == (
        expected,
        {"entity_kind": "edge"},
    )

    cleared = []
    monkeypatch.setattr(
        window.viewport,
        "clear_selection",
        lambda **options: cleared.append(options),
    )

    def reject_edge_load_editor(dialog):
        assert dialog.selected_scope() == RegionRef(
            "edge",
            "FIXED_EDGE",
        )
        assert highlighted_scopes[-1] == (
            expected,
            {"entity_kind": "edge"},
        )
        return False

    monkeypatch.setattr(LoadDialog, "exec", reject_edge_load_editor)

    window._edit_analysis_definition_key(("edge_load", 0, 0))

    assert cleared == [{"render": False}, {}]
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
