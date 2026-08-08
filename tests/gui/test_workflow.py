from __future__ import annotations

import os
from time import monotonic

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtCore import QThread
from PySide6.QtWidgets import QApplication, QDialog, QFileDialog

from fem.application.results import (
    FieldPosition,
    ResultProvider,
    ResultVariable,
    ScalarFieldSelection,
)
from fem.io.result_csv import read_result_csv
from fem.io.result_vtk import read_result_vtk
from fem_gui.main_window import FEMMainWindow
from fem_gui.postprocessing_dialogs import TypedResultDisplaySettings
from fem_gui.result_csv_export_dialog import ResultCsvExportDialog
from fem_gui.task_controller import (
    BackgroundTaskState,
    TaskApplyStatus,
    TaskCompletion,
)
from fem_gui.visualization.symbols import SymbolSettings


def _application() -> QApplication:
    return QApplication.instance() or QApplication([])


def _wait_for_task(window: FEMMainWindow) -> None:
    controller = window.task_controller
    assert controller.busy
    deadline = monotonic() + 10.0
    application = QApplication.instance()
    while controller.busy and monotonic() < deadline:
        application.processEvents()
        QThread.msleep(1)
    application.processEvents()
    assert not controller.busy


def _current_ready_selection(
    window: FEMMainWindow,
) -> tuple[ResultProvider, ScalarFieldSelection]:
    provider = window._current_result_provider()
    selection = window.result_selection
    assert type(provider) is ResultProvider
    assert type(selection) is ScalarFieldSelection
    field = provider.field(selection.field_key)
    assert field.source == provider.source
    assert field.key == selection.field_key
    assert selection.component in field.descriptor.columns
    assert (
        window.viewport._result_render_payload.topology.selection
        == selection
    )
    return provider, selection


def test_background_import_solve_and_result_state(gui_inp_path):
    _application()
    window = FEMMainWindow()

    window._load_path(gui_inp_path)
    assert not window.actions["open"].isEnabled()
    assert not window.actions["submit_job"].isEnabled()
    _wait_for_task(window)

    assert window.document.has_model
    completion = window.task_controller.last_completion
    assert isinstance(completion, TaskCompletion)
    assert completion.task_name == "INP 导入"
    assert completion.state is BackgroundTaskState.SUCCEEDED
    assert completion.apply_status is TaskApplyStatus.ACCEPTED
    assert not window.actions["submit_job"].isEnabled()
    assert window.check_current_model(show_success=False)
    assert window.actions["submit_job"].isEnabled()
    assert not window.actions["contour"].isEnabled()
    assert window.geometry.point_index_to_node_id[0] == 1

    job = window._submit_job("Job-1", "Static-1")
    assert job is not None
    assert not window.actions["submit_job"].isEnabled()
    _wait_for_task(window)

    assert window.document.has_result
    assert window.actions["deformed"].isEnabled()
    assert window.actions["query"].isEnabled()
    assert window.actions["deformed"].isChecked()
    provider, selection = _current_ready_selection(window)
    field = provider.field(selection.field_key)
    field_id = selection.field_key.request.field_id
    assert field_id.variable is ResultVariable.U
    assert field_id.position is FieldPosition.NODE
    component_index = field.descriptor.columns.index(selection.component)
    assert np.max(field.values[:, component_index]) > 0.0
    display = TypedResultDisplaySettings(
        shape_mode="undeformed",
        contour_enabled=True,
        selection=selection,
        scale_mode="custom",
        scale_value=5.0,
        overlay_undeformed=True,
        show_edges=False,
    )
    window._apply_typed_result_display_settings(
        display,
        expected_source=provider.source,
    )
    assert window._display.shape_mode == "undeformed"
    assert window._display.contour_enabled
    assert window.result_selection == selection
    assert (
        window.viewport._result_render_payload.topology.selection
        == selection
    )
    assert window._scale_value == 5.0
    assert window._overlay_undeformed
    assert not window.actions["edges"].isChecked()
    assert not window.actions["screenshot"].isEnabled()

    symbol_settings = SymbolSettings(step_name="Static-1", show_values=True, scale=1.5)
    window._apply_symbol_settings(symbol_settings)
    assert window._symbol_settings == symbol_settings
    window.close()


def test_reload_clears_selection_and_old_result(gui_inp_path):
    _application()
    window = FEMMainWindow()
    window._load_path(gui_inp_path)
    _wait_for_task(window)
    window.selection.select_node(1)
    assert window.check_current_model(show_success=False)
    window._submit_job("Job-1", "Static-1")
    _wait_for_task(window)
    assert window.document.has_result

    receipt = window.reload_imported_source()
    assert receipt.diagnostic is None
    _wait_for_task(window)

    assert window.selection.node_id is None
    assert window.selection.element_id is None
    assert not window.document.has_result
    assert not window.actions["query"].isEnabled()
    window.close_model(confirm=False)
    window.close()


def test_gui_exports_the_current_result_field_as_csv_and_vtk(
    monkeypatch,
    gui_inp_path,
    tmp_path,
):
    _application()
    window = FEMMainWindow()
    window._load_path(gui_inp_path)
    _wait_for_task(window)
    assert window.check_current_model(show_success=False)
    window._submit_job("Job-1", "Static-1")
    _wait_for_task(window)
    provider, selection = _current_ready_selection(window)
    window._apply_typed_result_display_settings(
        TypedResultDisplaySettings(
            shape_mode="deformed",
            contour_enabled=True,
            selection=selection,
            scale_mode="custom",
            scale_value=2.5,
            overlay_undeformed=False,
            show_edges=False,
        ),
        expected_source=provider.source,
    )
    assert window.result_selection == selection
    csv_target = tmp_path / "gui_result_test.csv"
    vtk_target = tmp_path / "gui_result_test.vtk"
    successes = []

    def execute_csv_dialog(dialog: QDialog) -> int:
        assert type(dialog) is ResultCsvExportDialog
        assert dialog.path_edit.text() == ""
        dialog.path_edit.setText(str(csv_target))
        return QDialog.DialogCode.Accepted

    monkeypatch.setattr(window, "_exec_dialog", execute_csv_dialog)
    monkeypatch.setattr(
        window,
        "_show_save_success",
        lambda content_name, path: successes.append((content_name, path)),
    )
    monkeypatch.setattr(
        QFileDialog,
        "getSaveFileName",
        staticmethod(
            lambda *_args, **_kwargs: (
                str(vtk_target),
                "VTK 文件 (*.vtk)",
            )
        ),
    )

    assert window.actions["export_csv"].isEnabled()
    assert window.actions["export_vtk"].isEnabled()
    window.actions["export_csv"].trigger()
    _wait_for_task(window)
    csv_readback = read_result_csv(csv_target)
    assert csv_readback.source == provider.source
    assert (
        csv_readback.materialization_generation
        == provider.snapshot.generation
    )
    assert csv_readback.selection == selection
    assert successes == [("CSV 文件", csv_target)]

    window.actions["export_vtk"].trigger()
    _wait_for_task(window)
    vtk_readback = read_result_vtk(vtk_target)
    assert vtk_readback.source == provider.source
    assert (
        vtk_readback.materialization_generation
        == provider.snapshot.generation
    )
    assert vtk_readback.selection == selection
    assert vtk_readback.deformation_scale == 2.5
    window.close_model(confirm=False)
    window.close()
