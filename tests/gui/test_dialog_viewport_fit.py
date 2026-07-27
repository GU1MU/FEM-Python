from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from fem.abaqus import read
from fem_gui.main_window import FEMMainWindow
from fem_gui.visualization.model_adapter import build_model_geometry


def _application() -> QApplication:
    return QApplication.instance() or QApplication([])


def _loaded_window(gui_inp_path) -> FEMMainWindow:
    window = FEMMainWindow()
    model = read(gui_inp_path)
    window._model_loaded(
        gui_inp_path,
        (model, build_model_geometry(model)),
    )
    return window


def test_information_dialog_fits_viewport_when_closed(
    monkeypatch,
    gui_inp_path,
) -> None:
    _application()
    window = _loaded_window(gui_inp_path)
    fit_calls = []
    monkeypatch.setattr(window.viewport, "fit", lambda: fit_calls.append(True))

    dialog = window.show_entity_information("node", 1)
    assert dialog is not None
    dialog.reject()

    assert fit_calls == [True]
    window.close()


def test_mesh_browser_fits_viewport_when_closed(
    monkeypatch,
    gui_inp_path,
) -> None:
    _application()
    window = _loaded_window(gui_inp_path)
    fit_calls = []
    monkeypatch.setattr(window.viewport, "fit", lambda: fit_calls.append(True))

    dialog = window.show_mesh_browser()
    assert dialog is not None
    dialog.reject()

    assert fit_calls == [True]
    window.close()


def test_contour_dialog_fits_viewport_after_exec(
    monkeypatch,
) -> None:
    _application()
    window = FEMMainWindow()
    fit_calls = []
    monkeypatch.setattr(window.viewport, "fit", lambda: fit_calls.append(True))
    monkeypatch.setattr(
        window,
        "_current_result_provider",
        lambda: object(),
    )

    def reject_dialog(dialog) -> int:
        dialog.reject()
        return 0

    monkeypatch.setattr(
        "fem_gui.main_window.ContourSettingsDialog.exec",
        reject_dialog,
    )

    window.show_contour_dialog()

    assert fit_calls == [True]
    window.close()


def test_wire_editor_exit_fits_after_splitter_restores(
    monkeypatch,
) -> None:
    app = _application()
    window = FEMMainWindow()
    fit_calls = []
    window._wire_editor_controller = object()
    monkeypatch.setattr(window.wire_editor_panel, "end", lambda: None)
    monkeypatch.setattr(window.viewport, "fit", lambda: fit_calls.append(True))

    window._exit_wire_editor()
    app.processEvents()

    assert fit_calls == [True]
    window.close()
