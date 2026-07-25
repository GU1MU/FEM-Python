from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from fem.abaqus import read
from fem.solvers.static_linear import solve
from fem_gui.main_window import FEMMainWindow
from fem_gui.analysis_jobs import AnalysisJob, JobStatus
from fem_gui.visualization.model_adapter import build_model_geometry
from fem_gui.visualization.result_adapter import build_result_data


def _application() -> QApplication:
    return QApplication.instance() or QApplication([])


def _solved_window(path) -> FEMMainWindow:
    window = FEMMainWindow()
    model = read(path)
    geometry = build_model_geometry(model)
    window._model_loaded(path, (model, geometry))
    result = solve(model)
    job = AnalysisJob("Job-1", "Static-1", JobStatus.RUNNING)
    window.document.add_job(job)
    window._job_succeeded(job, (result, build_result_data(result, geometry)))
    window._update_action_states()
    return window


def test_shape_and_contour_are_independent_for_all_four_states(gui_inp_path):
    _application()
    window = _solved_window(gui_inp_path)

    for shape_mode in ("undeformed", "deformed"):
        window.set_shape_mode(shape_mode)
        for contour_enabled in (False, True):
            window.actions["contour"].setChecked(contour_enabled)
            window._toggle_contour(contour_enabled)
            assert window._display.shape_mode == shape_mode
            assert window._display.contour_enabled is contour_enabled
            assert window.viewport._display == window._display
            assert ("无云图" not in window.status_panel.result_label.text()) is contour_enabled
    window.close()


def test_analysis_uses_clean_deformed_displacement_contour_defaults(gui_inp_path):
    _application()
    window = FEMMainWindow()
    model = read(gui_inp_path)
    geometry = build_model_geometry(model)
    window._model_loaded(gui_inp_path, (model, geometry))
    window.selection.select_node(1)
    window.viewport.highlight_node(1)
    result = solve(model)

    job = AnalysisJob("Job-1", "Static-1", JobStatus.RUNNING)
    window.document.add_job(job)
    window._job_succeeded(job, (result, build_result_data(result, geometry)))

    assert window._display.shape_mode == "deformed"
    assert window._display.contour_enabled
    assert window._display.field_key == "U"
    assert window._contour_options["style"] == "continuous"
    assert window._contour_options["colormap"] == "jet"
    assert window._contour_options["decimals"] == 5
    assert not window._contour_options["show_minimum"]
    assert not window._contour_options["show_maximum"]
    assert window._contour_options["orientation"] == "horizontal"
    assert not window.actions["symbols"].isChecked()
    assert not window.actions["node_labels"].isChecked()
    assert not window.actions["element_labels"].isChecked()
    assert not window.actions["edges"].isChecked()
    assert window.selection.node_id == 1
    assert window.viewport._selected_id == 1
    assert not window.viewport._selection_highlight_visible

    window.actions["contour"].setChecked(False)
    window._toggle_contour(False)
    assert window.actions["edges"].isChecked()
    window.close()


def test_result_ribbon_selects_real_fields_and_deformation_scale(gui_inp_path):
    _application()
    window = _solved_window(gui_inp_path)

    assert window.result_variable_combo.currentData() == "U"
    assert window.result_component_combo.currentData() == "U"
    reaction_index = window.result_variable_combo.findData("RF")
    window.result_variable_combo.setCurrentIndex(reaction_index)
    window._result_variable_changed(reaction_index)
    assert window.result_component_combo.currentData() == "RF"
    assert window._display.field_key == "RF"
    assert window._display.contour_enabled

    stress_index = window.result_variable_combo.findData("S")
    window.result_variable_combo.setCurrentIndex(stress_index)
    window._result_variable_changed(stress_index)
    assert window.result_position_combo.currentData() == "IP"
    assert str(window.result_component_combo.currentData()).startswith("IP:")
    unaveraged_index = window.result_position_combo.findData("EN")
    window.result_position_combo.setCurrentIndex(unaveraged_index)
    window._result_position_changed(unaveraged_index)
    assert str(window.result_component_combo.currentData()).startswith("EN:")
    center_index = window.result_position_combo.findData("CENTROID")
    window.result_position_combo.setCurrentIndex(center_index)
    window._result_position_changed(center_index)
    assert str(window.result_component_combo.currentData()).startswith("CENTROID:")

    custom_index = window.result_scale_combo.findData("custom")
    window.result_scale_combo.setCurrentIndex(custom_index)
    window._result_scale_mode_changed(custom_index)
    window.result_scale_value.setValue(8.0)
    assert window.result_scale_value.text() == "8.00"
    assert window._scale_mode == "custom"
    assert window.viewport._deformation_scale == 8.0

    real_index = window.result_scale_combo.findData("real")
    window.result_scale_combo.setCurrentIndex(real_index)
    window._result_scale_mode_changed(real_index)
    assert window.viewport._deformation_scale == 1.0
    window.close()


def test_overlay_and_contour_style_update_existing_scene_state(gui_inp_path):
    _application()
    window = _solved_window(gui_inp_path)
    window.set_shape_mode("deformed")
    window.actions["overlay"].setChecked(True)
    window._toggle_undeformed_overlay(True)
    window._set_contour_options({
        "style": "continuous",
        "manual": True,
        "minimum": 0.0,
        "maximum": 1.0,
        "legend": True,
        "show_minimum": True,
        "show_maximum": True,
        "edges": False,
    })

    assert window._overlay_undeformed
    assert window.viewport._overlay_undeformed
    assert window.viewport._contour["style"] == "continuous"
    assert window.viewport._contour["manual"]
    assert window.viewport._contour["averaging_threshold"] == 75.0
    assert not window.actions["edges"].isChecked()
    window.close()
