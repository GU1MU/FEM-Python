from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pyvista as pv
import pytest
from PySide6.QtWidgets import QApplication, QDialog

from fem_gui import main_window as main_window_module
from fem_gui.main_window import FEMMainWindow
from fem_gui.viewport_image_export_dialog import (
    MAX_IMAGE_DIMENSION,
    MIN_IMAGE_DIMENSION,
    ViewportImageExportDialog,
    ViewportImageExportOptions,
)
from fem_gui.widgets.viewport import FEMViewport


def _application() -> QApplication:
    return QApplication.instance() or QApplication([])


def _dialog(
    size: tuple[int, int] = (800, 600),
    *,
    supports_transparency: bool = True,
) -> ViewportImageExportDialog:
    _application()
    return ViewportImageExportDialog(size, supports_transparency)


def test_export_dialog_defaults_to_two_times_current_viewport() -> None:
    dialog = _dialog()

    assert dialog.quality_combo.currentData() == 2
    assert dialog.output_size == (1600, 1200)
    assert dialog.output_size_label.text() == "1600 × 1200 px"
    assert dialog.options == ViewportImageExportOptions(2, None, False)
    assert not dialog.custom_width_spin.isEnabled()
    assert not dialog.custom_height_spin.isEnabled()


@pytest.mark.parametrize(
    ("quality", "expected_scale", "expected_size"),
    ((1, 1, (800, 600)), (4, 4, (3200, 2400))),
)
def test_export_dialog_fixed_quality_updates_scale_and_preview(
    quality: int,
    expected_scale: int,
    expected_size: tuple[int, int],
) -> None:
    dialog = _dialog()

    dialog.quality_combo.setCurrentIndex(dialog.quality_combo.findData(quality))

    assert dialog.output_size == expected_size
    assert dialog.options.scale == expected_scale
    assert dialog.options.window_size is None
    assert not dialog.custom_width_spin.isEnabled()
    assert not dialog.custom_height_spin.isEnabled()


def test_export_dialog_custom_quality_returns_exact_window_size() -> None:
    dialog = _dialog()
    custom_index = dialog.quality_combo.findData("custom")

    dialog.quality_combo.setCurrentIndex(custom_index)
    dialog.custom_width_spin.setValue(1234)
    dialog.custom_height_spin.setValue(987)

    assert dialog.custom_width_spin.isEnabled()
    assert dialog.custom_height_spin.isEnabled()
    assert dialog.output_size == (1234, 987)
    assert dialog.output_size_label.text() == "1234 × 987 px"
    assert dialog.options == ViewportImageExportOptions(
        1,
        (1234, 987),
        False,
    )

    dialog.quality_combo.setCurrentIndex(dialog.quality_combo.findData(2))
    assert not dialog.custom_width_spin.isEnabled()
    assert not dialog.custom_height_spin.isEnabled()
    assert dialog.output_size == (1600, 1200)


@pytest.mark.parametrize(
    ("size", "expected_custom_size"),
    (
        ((1, 1), (MIN_IMAGE_DIMENSION, MIN_IMAGE_DIMENSION)),
        ((10000, 9000), (MAX_IMAGE_DIMENSION, MAX_IMAGE_DIMENSION)),
    ),
)
def test_export_dialog_clamps_initial_custom_dimensions(
    size: tuple[int, int],
    expected_custom_size: tuple[int, int],
) -> None:
    dialog = _dialog(size)

    dialog.quality_combo.setCurrentIndex(dialog.quality_combo.findData("custom"))

    assert dialog.output_size == expected_custom_size


def test_export_dialog_transparency_is_available_only_for_png() -> None:
    png_dialog = _dialog(supports_transparency=True)
    png_dialog.transparent_background_check.setChecked(True)
    assert png_dialog.transparent_background_check.isEnabled()
    assert png_dialog.options.transparent_background

    jpeg_dialog = _dialog(supports_transparency=False)
    jpeg_dialog.transparent_background_check.setChecked(True)
    assert not jpeg_dialog.transparent_background_check.isEnabled()
    assert not jpeg_dialog.options.transparent_background


def test_viewport_screenshot_forwards_all_export_parameters() -> None:
    _application()
    viewport = FEMViewport()
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    viewport._plotter = SimpleNamespace(
        window_size=(720, 480),
        screenshot=lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    assert viewport.screenshot_size() == (720, 480)
    viewport.save_screenshot(
        "viewport.png",
        scale=1,
        window_size=(1400, 900),
        transparent_background=True,
    )

    assert calls == [
        (
            ("viewport.png",),
            {
                "scale": 1,
                "window_size": (1400, 900),
                "transparent_background": True,
                "return_img": False,
            },
        )
    ]
    viewport._plotter = None
    viewport.close()


def test_viewport_screenshot_size_falls_back_to_qt_widget() -> None:
    _application()
    viewport = FEMViewport()
    viewport.resize(640, 360)
    viewport._plotter = SimpleNamespace(window_size=(0, 0))

    assert viewport.screenshot_size() == (640, 360)

    viewport._plotter = None
    viewport.close()


class _ExportHarness:
    def __init__(self) -> None:
        self.document = SimpleNamespace(path=None)
        self.viewport = SimpleNamespace(
            screenshot_size=lambda: (900, 500),
            save_screenshot=self._save_screenshot,
        )
        self.status_calls: list[tuple[str, int]] = []
        self.status_panel = SimpleNamespace(set_state=self._set_state)
        self.screenshot_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
        self.errors: list[tuple[str, str]] = []

    def _current_result_provider(self) -> object:
        return object()

    def _save_screenshot(self, *args, **kwargs) -> None:
        self.screenshot_calls.append((args, kwargs))

    def _set_state(self, message: str, timeout: int) -> None:
        self.status_calls.append((message, timeout))

    def _show_error(self, title: str, message: str) -> None:
        self.errors.append((title, message))


def test_export_flow_cancels_before_or_after_settings_without_screenshot(
    monkeypatch,
) -> None:
    harness = _ExportHarness()
    dialog_calls = []
    monkeypatch.setattr(
        main_window_module.QFileDialog,
        "getSaveFileName",
        lambda *_args, **_kwargs: ("", ""),
    )
    monkeypatch.setattr(
        main_window_module,
        "ViewportImageExportDialog",
        lambda *_args, **_kwargs: dialog_calls.append((_args, _kwargs)),
    )

    FEMMainWindow.export_viewport_image(harness)

    assert dialog_calls == []
    assert harness.screenshot_calls == []

    class RejectedDialog:
        def __init__(self, *args, **kwargs) -> None:
            dialog_calls.append((args, kwargs))

        def exec(self) -> QDialog.DialogCode:
            return QDialog.DialogCode.Rejected

    monkeypatch.setattr(
        main_window_module.QFileDialog,
        "getSaveFileName",
        lambda *_args, **_kwargs: ("viewport.png", ""),
    )
    monkeypatch.setattr(
        main_window_module,
        "ViewportImageExportDialog",
        RejectedDialog,
    )

    FEMMainWindow.export_viewport_image(harness)

    assert len(dialog_calls) == 1
    assert harness.screenshot_calls == []


def test_export_flow_passes_options_and_keeps_success_feedback(monkeypatch) -> None:
    harness = _ExportHarness()
    created_with = []
    selected = ViewportImageExportOptions(4, None, False)

    class AcceptedDialog:
        options = selected

        def __init__(self, *args, **kwargs) -> None:
            created_with.append((args, kwargs))

        def exec(self) -> QDialog.DialogCode:
            return QDialog.DialogCode.Accepted

    monkeypatch.setattr(
        main_window_module.QFileDialog,
        "getSaveFileName",
        lambda *_args, **_kwargs: ("viewport.jpg", ""),
    )
    monkeypatch.setattr(
        main_window_module,
        "ViewportImageExportDialog",
        AcceptedDialog,
    )

    FEMMainWindow.export_viewport_image(harness)

    assert created_with[0][0][:2] == ((900, 500), False)
    assert harness.screenshot_calls == [
        (
            ("viewport.jpg",),
            {
                "scale": 4,
                "window_size": None,
                "transparent_background": False,
            },
        )
    ]
    assert harness.status_calls == [("视口图片保存完成", 5000)]
    assert harness.errors == []


def test_export_flow_keeps_existing_error_feedback(monkeypatch) -> None:
    harness = _ExportHarness()
    harness.viewport.save_screenshot = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        RuntimeError("capture failed")
    )

    class AcceptedDialog:
        options = ViewportImageExportOptions(1, (1024, 768), True)

        def __init__(self, *args, **kwargs) -> None:
            pass

        def exec(self) -> QDialog.DialogCode:
            return QDialog.DialogCode.Accepted

    monkeypatch.setattr(
        main_window_module.QFileDialog,
        "getSaveFileName",
        lambda *_args, **_kwargs: ("viewport", ""),
    )
    monkeypatch.setattr(
        main_window_module,
        "ViewportImageExportDialog",
        AcceptedDialog,
    )

    FEMMainWindow.export_viewport_image(harness)

    assert harness.errors == [("导出视口图片失败", "capture failed")]
    assert harness.status_calls == []


def test_custom_screenshot_restores_pyvista_size_and_camera(tmp_path: Path) -> None:
    _application()
    viewport = FEMViewport()
    plotter = pv.Plotter(off_screen=True, window_size=(320, 240))
    plotter.add_mesh(pv.Sphere())
    viewport._plotter = plotter
    before_size = tuple(plotter.window_size)
    before_camera = (
        tuple(plotter.camera.position),
        tuple(plotter.camera.focal_point),
        tuple(plotter.camera.up),
        plotter.camera.parallel_scale,
        tuple(plotter.camera.clipping_range),
        plotter.camera.view_angle,
    )
    output = tmp_path / "custom.png"

    viewport.save_screenshot(str(output), window_size=(640, 360))

    image = pv.read(output)
    after_camera = (
        tuple(plotter.camera.position),
        tuple(plotter.camera.focal_point),
        tuple(plotter.camera.up),
        plotter.camera.parallel_scale,
        tuple(plotter.camera.clipping_range),
        plotter.camera.view_angle,
    )
    assert image.dimensions[:2] == (640, 360)
    assert tuple(plotter.window_size) == before_size
    assert after_camera == before_camera

    plotter.close()
    viewport._plotter = None
    viewport.close()
