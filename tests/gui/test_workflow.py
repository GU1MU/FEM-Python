from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtCore import QEventLoop, QThread, QTimer
from PySide6.QtWidgets import QApplication, QFileDialog

from fem_gui.main_window import FEMMainWindow
from fem_gui.postprocessing_dialogs import ResultDisplaySettings
from fem_gui.visualization.symbols import SymbolSettings


ROOT = Path(__file__).resolve().parents[2]


def _application() -> QApplication:
    return QApplication.instance() or QApplication([])


def _wait_for_task(window: FEMMainWindow) -> None:
    assert window._thread is not None
    loop = QEventLoop()
    window._thread.finished.connect(loop.quit)
    QTimer.singleShot(10_000, loop.quit)
    loop.exec()
    assert not window.busy


def test_background_import_solve_and_result_state(gui_inp_path):
    _application()
    window = FEMMainWindow()
    callback_threads: list[QThread] = []
    original_model_loaded = window._model_loaded

    def record_model_loaded(path, value):
        callback_threads.append(QThread.currentThread())
        original_model_loaded(path, value)

    window._model_loaded = record_model_loaded

    window._load_path(gui_inp_path)
    assert not window.actions["open"].isEnabled()
    assert not window.actions["submit_job"].isEnabled()
    _wait_for_task(window)

    assert window.document.has_model
    assert callback_threads == [window.thread()]
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
    assert np.max(window.result_data.fields["U"].values) > 0.0
    display = ResultDisplaySettings(
        shape_mode="undeformed",
        contour_enabled=True,
        field_key="U",
        scale_mode="custom",
        scale_value=5.0,
        overlay_undeformed=True,
        show_edges=False,
    )
    window._apply_result_display_settings(display)
    assert window._display.shape_mode == "undeformed"
    assert window._display.contour_enabled
    assert window._display.field_key == "U"
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
    window._submit_job("Job-1", "Static-1")
    _wait_for_task(window)
    assert window.document.has_result

    window.reload_model()
    _wait_for_task(window)

    assert window.selection.node_id is None
    assert window.selection.element_id is None
    assert not window.document.has_result
    assert not window.actions["query"].isEnabled()
    window.close()


def test_gui_uses_official_vtk_result_export(monkeypatch, gui_inp_path):
    _application()
    window = FEMMainWindow()
    window._load_path(gui_inp_path)
    _wait_for_task(window)
    window._submit_job("Job-1", "Static-1")
    _wait_for_task(window)
    target = ROOT / "gui_result_test.vtk"
    generated = (
        target,
        ROOT / "gui_result_test_nodal_displacement.csv",
        ROOT / "gui_result_test_element_stress.csv",
        ROOT / "gui_result_test_nodal_stress.csv",
    )
    monkeypatch.setattr(
        QFileDialog,
        "getSaveFileName",
        staticmethod(lambda *_args, **_kwargs: (str(target), "VTK 文件 (*.vtk)")),
    )

    try:
        window.export_vtk()
        assert target.is_file()
        content = target.read_text(encoding="utf-8")
        assert "DATASET UNSTRUCTURED_GRID" in content
        assert "CELL_TYPES 1" in content
        assert "VECTORS displacement float" in content
    finally:
        for path in generated:
            path.unlink(missing_ok=True)
        window.close()
