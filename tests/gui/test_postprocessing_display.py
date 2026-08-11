from __future__ import annotations

import os
from time import monotonic

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication

from fem.application import MeshEntityRef
from fem.io.inp import read
from fem.application.results import (
    FieldPosition,
    ResultFieldId,
    ResultVariable,
    ScalarFieldSelection,
    build_solve_result_bundle,
)
from fem.solvers.static_linear import solve
from fem_gui.main_window import FEMMainWindow
from fem_gui.postprocessing_dialogs import (
    ContourSettingsDialog,
    DisplaySettingsDialog,
)
from fem_gui.visualization.model_adapter import build_model_geometry


def _application() -> QApplication:
    return QApplication.instance() or QApplication([])


def _wait_for_tasks(window: FEMMainWindow) -> None:
    deadline = monotonic() + 2.0
    application = _application()
    idle_cycles = 0
    while monotonic() < deadline:
        application.processEvents()
        if window.busy:
            idle_cycles = 0
            continue
        idle_cycles += 1
        if idle_cycles >= 3:
            break
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
    render_position: FieldPosition | None = None,
) -> ScalarFieldSelection:
    provider = window._current_result_provider()
    selection = window.result_component_combo.currentData()
    assert provider is not None
    assert type(window.result_variable_combo.currentData()) is ResultVariable
    assert window.result_variable_combo.currentData() is variable
    field_id = window.result_position_combo.currentData()
    assert type(field_id) is ResultFieldId
    assert field_id.variable is variable
    assert field_id.position is position
    assert type(selection) is ScalarFieldSelection
    assert selection == window.result_selection
    assert selection.field_key.request.field_id.variable is variable
    assert selection.field_key.request.field_id.position is position
    availability = provider.field_status(selection.field_key)
    assert selection.component in availability.descriptor.columns
    rendered_selection = (
        window.viewport._result_render_payload.topology.selection
    )
    expected_render_position = (
        position if render_position is None else render_position
    )
    assert (
        rendered_selection.field_key.request.field_id.position
        is expected_render_position
    )
    assert rendered_selection.component == selection.component
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
    window._set_selection_filter("point")
    window._on_mesh_scope_entity_pick(MeshEntityRef.node(1))
    selected_references = set(window._selected_mesh_scope_refs)
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
    assert window._contour_options["style"] == "segmented"
    assert window._contour_options["colormap"] == "abaqus_rainbow"
    assert window._contour_options["render_mode"] == "shaded"
    assert window._contour_options["edge_mode"] == "geometry"
    assert window.viewport._contour["edge_mode"] == "geometry"
    assert window._contour_options["number_format"] == "scientific"
    assert window._contour_options["decimals"] == 2
    assert not window._contour_options["show_minimum"]
    assert not window._contour_options["show_maximum"]
    assert window._contour_options["show_coordinate_system"]
    assert window._contour_options["orientation"] == "vertical"
    assert not window.actions["symbols"].isChecked()
    assert not window.actions["node_labels"].isChecked()
    assert not window.actions["element_labels"].isChecked()
    assert window.actions["edges"].isChecked()
    assert selected_references == {
        MeshEntityRef.node(1, part_id="P1")
    }
    assert window._selected_mesh_scope_refs == selected_references
    assert (
        window.viewport._mesh_scope_selected_references
        == selected_references
    )

    window.actions["contour"].setChecked(False)
    window._toggle_contour(False)
    assert window.actions["edges"].isChecked()
    window.close()


def test_ribbon_modules_switch_between_result_contour_and_mesh(
    gui_inp_path,
) -> None:
    _application()
    window = _solved_window(gui_inp_path)

    assert window.ribbon.tab_bar.tabText(
        window.ribbon.tab_bar.currentIndex()
    ) == "结果"
    assert window.viewport._result_render_payload is not None
    assert window.viewport._display.contour_enabled

    for module_name in ("分析", "模型", "网格"):
        window.ribbon.set_current(module_name)
        assert window.viewport._result_render_payload is None
        assert window.viewport._geometry_preview is None
        assert window.viewport.artifact_id == window.document.artifact.artifact_id

    window.ribbon.set_current("结果")
    assert window.viewport._result_render_payload is not None
    assert window.viewport._display.contour_enabled
    assert (
        window.viewport._result_render_payload.topology.selection
        == window.result_selection
    )
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
    element_nodal_selection = _assert_current_ribbon_selection(
        window,
        variable=ResultVariable.S,
        position=FieldPosition.ELEMENT_NODAL,
        render_position=FieldPosition.RESOLVED_NODAL,
    )
    assert element_nodal_selection.component
    assert window.result_position_combo.count() == 1
    assert window.result_position_combo.currentText() == "节点"

    custom_index = window.result_scale_combo.findData("custom")
    window.result_scale_combo.setCurrentIndex(custom_index)
    window._result_scale_mode_changed(custom_index)
    assert window.result_scale_value.isEnabled()
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
    assert not window.result_scale_value.isEnabled()
    assert window.result_scale_value.value() == 1.0

    auto_index = window.result_scale_combo.findData("auto")
    window.result_scale_combo.setCurrentIndex(auto_index)
    window._result_scale_mode_changed(auto_index)
    automatic_scale = (
        window.viewport._result_render_payload.topology.deformation_scale
    )
    assert not window.result_scale_value.isEnabled()
    assert window.result_scale_value.decimals() == 2
    assert window.result_scale_value.text() == f"{automatic_scale:.2f}"

    window.result_scale_combo.setCurrentIndex(custom_index)
    window._result_scale_mode_changed(custom_index)
    assert window.result_scale_value.isEnabled()
    assert window.result_scale_value.value() == 8.0
    window.close()


def test_display_settings_dialog_applies_viewport_options(gui_inp_path):
    _application()
    window = _solved_window(gui_inp_path)
    opened: list[DisplaySettingsDialog] = []
    window._exec_dialog = opened.append

    window.show_display_settings_dialog()

    assert len(opened) == 1
    dialog = opened[0]
    assert dialog.windowTitle() == "显示设置"
    dialog.engineering_format.setChecked(True)
    dialog.horizontal_orientation.setChecked(True)
    dialog.edge_style.setCurrentIndex(dialog.edge_style.findData("dashed"))
    dialog.edge_width.setValue(2.5)
    dialog.show_ids.setChecked(True)
    dialog.apply()

    assert window.viewport._contour["number_format"] == "engineering"
    assert window.viewport._contour["orientation"] == "horizontal"
    assert window.viewport._contour["edge_style"] == "dashed"
    assert window.viewport._contour["edge_width"] == 2.5
    assert window.viewport._contour["show_ids"]
    window.close()


def test_contour_dialog_auto_range_uses_current_result_extrema(gui_inp_path):
    _application()
    window = _solved_window(gui_inp_path)
    expected = window.viewport.current_contour_range()
    opened: list[ContourSettingsDialog] = []
    window._exec_dialog = opened.append

    window.show_contour_dialog()

    assert expected is not None
    assert len(opened) == 1
    dialog = opened[0]
    assert dialog.auto_range.isChecked()
    assert not dialog.minimum.isEnabled()
    assert not dialog.maximum.isEnabled()
    assert dialog.minimum.value() == pytest.approx(expected[0])
    assert dialog.maximum.value() == pytest.approx(expected[1])
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


def test_stress_exposes_no_discarded_position_controls(gui_inp_path):
    _application()
    window = _solved_window(gui_inp_path)

    assert window.result_averaging_threshold.value() == 75.0
    assert window.result_averaging_threshold.isHidden()

    stress_index = window.result_variable_combo.findData(ResultVariable.S)
    assert stress_index >= 0
    window.result_variable_combo.setCurrentIndex(stress_index)
    window._result_variable_changed(stress_index)
    _wait_for_tasks(window)

    selection = window.result_selection
    assert type(selection) is ScalarFieldSelection
    assert (
        selection.field_key.request.field_id.position
        is FieldPosition.ELEMENT_NODAL
    )
    assert window.result_position_combo.count() == 1
    assert window.result_position_combo.currentText() == "节点"
    assert window.result_averaging_threshold.isHidden()
    rendered_selection = (
        window.viewport._result_render_payload.topology.selection
    )
    assert (
        rendered_selection.field_key.request.field_id.position
        is FieldPosition.RESOLVED_NODAL
    )
    assert (
        rendered_selection.field_key.request.averaging_policy
        .threshold_percent
        == 75.0
    )
    window.close()


def test_stress_averaging_threshold_rebuilds_only_visual_field(gui_inp_path):
    _application()
    window = _solved_window(gui_inp_path)

    stress_index = window.result_variable_combo.findData(ResultVariable.S)
    window.result_variable_combo.setCurrentIndex(stress_index)
    window._result_variable_changed(stress_index)
    _wait_for_tasks(window)
    selected = window.result_selection

    window._set_contour_options({"averaging_threshold": 25.0})
    _wait_for_tasks(window)

    assert window.result_selection == selected
    rendered_selection = (
        window.viewport._result_render_payload.topology.selection
    )
    assert (
        rendered_selection.field_key.request.field_id.position
        is FieldPosition.RESOLVED_NODAL
    )
    assert (
        rendered_selection.field_key.request.averaging_policy
        .threshold_percent
        == 25.0
    )
    window.close()
