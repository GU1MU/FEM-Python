from __future__ import annotations

import os
from time import monotonic

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from fem.abaqus import read
from fem.application.results import (
    FieldPosition,
    ResultVariable,
    ScalarFieldSelection,
    build_solve_result_bundle,
)
from fem.io.result_csv import read_result_csv
from fem.solvers.static_linear import solve
from fem_gui.commands import ResultCsvExportSpec
from fem_gui.main_window import FEMMainWindow
from fem_gui.visualization.model_adapter import build_model_geometry


def _application() -> QApplication:
    return QApplication.instance() or QApplication([])


def _wait_for_tasks(window: FEMMainWindow) -> None:
    deadline = monotonic() + 5.0
    application = _application()
    while window.busy and monotonic() < deadline:
        application.processEvents()
    application.processEvents()
    assert not window.busy


def _solved_window(path) -> FEMMainWindow:
    window = FEMMainWindow()
    model = read(path)
    geometry = build_model_geometry(model)
    window._model_loaded(path, (model, geometry))
    _install_result(window)
    window._update_action_states()
    return window


def _install_result(window) -> None:
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


def _assert_current_ribbon_selection(
    window: FEMMainWindow,
    *,
    variable: ResultVariable,
    position: FieldPosition,
) -> ScalarFieldSelection:
    provider = window._current_result_provider()
    selection = window.result_component_combo.currentData()
    assert provider is not None
    assert type(window.result_variable_combo.currentData()) is ResultVariable
    assert window.result_variable_combo.currentData() is variable
    assert type(window.result_position_combo.currentData()) is FieldPosition
    assert window.result_position_combo.currentData() is position
    assert type(selection) is ScalarFieldSelection
    assert selection == window.result_selection
    assert selection.field_key.request.field_id.variable is variable
    assert selection.field_key.request.field_id.position is position
    availability = provider.field_status(selection.field_key)
    assert selection.component in availability.descriptor.columns
    assert (
        window.viewport._result_render_payload.topology.selection
        == selection
    )
    return selection


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
            assert window.viewport._display.shape_mode == shape_mode
            assert (
                window.viewport._display.contour_enabled
                is contour_enabled
            )
            assert (
                window.viewport._result_render_payload.topology.selection
                == window.result_selection
            )
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
    _install_result(window)

    assert window._display.shape_mode == "deformed"
    assert window._display.contour_enabled
    selection = _assert_current_ribbon_selection(
        window,
        variable=ResultVariable.U,
        position=FieldPosition.NODE,
    )
    provider = window._current_result_provider()
    assert provider is not None
    assert selection == provider.catalog().default_selection
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

    default_selection = _assert_current_ribbon_selection(
        window,
        variable=ResultVariable.U,
        position=FieldPosition.NODE,
    )
    provider = window._current_result_provider()
    assert provider is not None
    assert default_selection == provider.catalog().default_selection

    reaction_index = window.result_variable_combo.findData(ResultVariable.RF)
    assert reaction_index >= 0
    window.result_variable_combo.setCurrentIndex(reaction_index)
    window._result_variable_changed(reaction_index)
    reaction_selection = _assert_current_ribbon_selection(
        window,
        variable=ResultVariable.RF,
        position=FieldPosition.NODE,
    )
    assert reaction_selection != default_selection
    assert window._display.contour_enabled

    stress_index = window.result_variable_combo.findData(ResultVariable.S)
    assert stress_index >= 0
    window.result_variable_combo.setCurrentIndex(stress_index)
    window._result_variable_changed(stress_index)
    _wait_for_tasks(window)
    integration_point_selection = _assert_current_ribbon_selection(
        window,
        variable=ResultVariable.S,
        position=FieldPosition.INTEGRATION_POINT,
    )

    unaveraged_index = window.result_position_combo.findData(
        FieldPosition.ELEMENT_NODAL
    )
    assert unaveraged_index >= 0
    window.result_position_combo.setCurrentIndex(unaveraged_index)
    window._result_position_changed(unaveraged_index)
    _wait_for_tasks(window)
    element_nodal_selection = _assert_current_ribbon_selection(
        window,
        variable=ResultVariable.S,
        position=FieldPosition.ELEMENT_NODAL,
    )
    assert element_nodal_selection.component == integration_point_selection.component
    assert element_nodal_selection.field_key != integration_point_selection.field_key

    center_index = window.result_position_combo.findData(
        FieldPosition.CENTROID
    )
    assert center_index >= 0
    window.result_position_combo.setCurrentIndex(center_index)
    window._result_position_changed(center_index)
    _wait_for_tasks(window)
    centroid_selection = _assert_current_ribbon_selection(
        window,
        variable=ResultVariable.S,
        position=FieldPosition.CENTROID,
    )
    assert centroid_selection.component == element_nodal_selection.component
    assert centroid_selection.field_key != element_nodal_selection.field_key

    custom_index = window.result_scale_combo.findData("custom")
    window.result_scale_combo.setCurrentIndex(custom_index)
    window._result_scale_mode_changed(custom_index)
    window.result_scale_value.setValue(8.0)
    assert window.result_scale_value.text() == "8.00"
    assert window._scale_mode == "custom"
    assert (
        window.viewport._result_render_payload.topology.deformation_scale
        == 8.0
    )

    real_index = window.result_scale_combo.findData("real")
    window.result_scale_combo.setCurrentIndex(real_index)
    window._result_scale_mode_changed(real_index)
    assert (
        window.viewport._result_render_payload.topology.deformation_scale
        == 1.0
    )
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


def test_stress_averaging_threshold_is_visualization_only(
    gui_inp_path,
    tmp_path,
):
    _application()
    window = _solved_window(gui_inp_path)

    assert window.result_averaging_threshold.value() == 75.0
    assert window.result_averaging_threshold.isHidden()

    stress_index = window.result_variable_combo.findData(ResultVariable.S)
    assert stress_index >= 0
    window.result_variable_combo.setCurrentIndex(stress_index)
    window._result_variable_changed(stress_index)
    _wait_for_tasks(window)
    assert not window.result_averaging_threshold.isHidden()
    assert not window.result_averaging_threshold.isEnabled()

    resolved_index = window.result_position_combo.findData(
        FieldPosition.RESOLVED_NODAL
    )
    assert resolved_index >= 0
    window.result_position_combo.setCurrentIndex(resolved_index)
    window._result_position_changed(resolved_index)
    _wait_for_tasks(window)

    selection = window.result_selection
    provider = window._current_result_provider()
    assert type(selection) is ScalarFieldSelection
    assert provider is not None
    assert window.result_averaging_threshold.isEnabled()
    assert (
        selection.field_key.request.averaging_policy.threshold_percent
        == 75.0
    )
    generation = provider.snapshot.generation

    window.result_averaging_threshold.setValue(25.0)
    _wait_for_tasks(window)

    current = window._current_result_provider()
    rendered = window.viewport._result_render_payload.topology.selection
    assert current is not None
    assert current.snapshot.generation == generation
    assert window.result_selection == selection
    assert rendered.component == selection.component
    assert (
        rendered.field_key.request.averaging_policy.threshold_percent
        == 25.0
    )
    assert not any(
        availability.key == rendered.field_key
        for availability in current.catalog().fields
    )
    window.set_shape_mode("undeformed")
    assert (
        window.viewport._result_render_payload.topology.selection
        == rendered
    )

    target = tmp_path / "stress.csv"
    receipt = window.export_result_csv(
        target,
        ResultCsvExportSpec(
            current.source,
            current.snapshot.generation,
            selection,
        ),
    )
    assert receipt.completion is not None
    _wait_for_tasks(window)
    readback = read_result_csv(target)
    field = current.field(selection.field_key)
    component_index = field.descriptor.columns.index(selection.component)
    expected_values = tuple(
        float(value)
        for value in field.values[:, component_index]
    )
    assert readback.selection == selection
    assert tuple(record.value for record in readback.records) == expected_values
    window.close()
