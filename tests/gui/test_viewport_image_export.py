from __future__ import annotations

from contextlib import nullcontext
import os
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pyvista as pv
import pytest
from PIL import Image, ImageChops, ImageStat
from PySide6.QtWidgets import QApplication, QDialog, QDialogButtonBox, QLabel

from fem_gui import main_window as main_window_module
from fem_gui import viewport_image_export_dialog as export_dialog_module
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
) -> ViewportImageExportDialog:
    _application()
    return ViewportImageExportDialog(size)


def test_export_dialog_defaults_to_two_times_current_viewport() -> None:
    dialog = _dialog()

    assert dialog.quality_combo.currentData() == 2
    assert dialog.output_size == (1600, 1200)
    assert dialog.width_spin.value() == 1600
    assert dialog.height_spin.value() == 1200
    assert dialog.options == ViewportImageExportOptions(2, None, False)
    assert not dialog.width_spin.isEnabled()
    assert not dialog.height_spin.isEnabled()

    labels = {label.text() for label in dialog.findChildren(QLabel)}
    assert "宽度：" in labels
    assert "高度：" in labels
    assert "当前视口：" not in labels
    assert "自定义宽度：" not in labels
    assert "自定义高度：" not in labels
    assert "输出尺寸：" not in labels


@pytest.mark.parametrize(
    ("quality", "expected_size"),
    (
        (1, (800, 600)),
        (4, (3200, 2400)),
    ),
)
def test_export_dialog_fixed_quality_updates_scale_and_preview(
    quality: int,
    expected_size: tuple[int, int],
) -> None:
    dialog = _dialog()

    dialog.quality_combo.setCurrentIndex(dialog.quality_combo.findData(quality))

    assert dialog.output_size == expected_size
    assert dialog.width_spin.value() == expected_size[0]
    assert dialog.height_spin.value() == expected_size[1]
    assert dialog.options.scale == quality
    assert dialog.options.window_size is None
    assert not dialog.width_spin.isEnabled()
    assert not dialog.height_spin.isEnabled()


def test_export_dialog_custom_quality_returns_exact_window_size() -> None:
    dialog = _dialog()
    custom_index = dialog.quality_combo.findData("custom")

    dialog.quality_combo.setCurrentIndex(custom_index)
    dialog.width_spin.setValue(1234)
    dialog.height_spin.setValue(987)

    assert dialog.width_spin.isEnabled()
    assert dialog.height_spin.isEnabled()
    assert dialog.output_size == (1234, 987)
    assert dialog.options == ViewportImageExportOptions(
        1,
        (1234, 987),
        False,
    )

    dialog.quality_combo.setCurrentIndex(dialog.quality_combo.findData(2))
    assert not dialog.width_spin.isEnabled()
    assert not dialog.height_spin.isEnabled()
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


def test_export_dialog_selects_path_and_updates_format_controls(monkeypatch) -> None:
    dialog = _dialog()
    ok_button = dialog.buttons.button(QDialogButtonBox.StandardButton.Ok)
    browse_calls = []

    assert dialog.target_path == ""
    assert not ok_button.isEnabled()
    assert dialog.transparent_background_check.isEnabled()
    dialog.transparent_background_check.setChecked(True)

    png_path = Path("exports") / "result"
    monkeypatch.setattr(
        export_dialog_module.QFileDialog,
        "getSaveFileName",
        lambda *args, **_kwargs: (
            browse_calls.append(args) or (str(png_path), "")
        ),
    )
    dialog.browse_button.click()

    assert browse_calls[0][2] == ""
    assert dialog.target_path == str(png_path.with_suffix(".png"))
    assert ok_button.isEnabled()
    assert dialog.transparent_background_check.isEnabled()
    assert dialog.options.transparent_background

    jpeg_path = Path("exports") / "result.jpg"
    monkeypatch.setattr(
        export_dialog_module.QFileDialog,
        "getSaveFileName",
        lambda *_args, **_kwargs: (str(jpeg_path), ""),
    )
    dialog.browse_button.click()

    assert dialog.target_path == str(jpeg_path)
    assert not dialog.transparent_background_check.isEnabled()
    assert not dialog.transparent_background_check.isChecked()
    assert not dialog.options.transparent_background


def test_export_dialog_keeps_path_when_browse_is_cancelled(monkeypatch) -> None:
    dialog = _dialog()
    existing = Path("exports") / "existing.png"
    dialog.path_edit.setText(str(existing))
    dialog._update_target_format()
    monkeypatch.setattr(
        export_dialog_module.QFileDialog,
        "getSaveFileName",
        lambda *_args, **_kwargs: ("", ""),
    )

    dialog.browse_button.click()

    assert dialog.target_path == str(existing)


def test_viewport_screenshot_forwards_all_export_parameters() -> None:
    _application()
    viewport = FEMViewport()
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    render_calls = []
    copied_camera = object()
    camera_calls = []
    camera = SimpleNamespace(
        copy=lambda: copied_camera,
        DeepCopy=lambda state: camera_calls.append(("copy", state)),
        Modified=lambda: camera_calls.append(("modified",)),
    )
    plotter = SimpleNamespace(
        window_size=(720, 480),
        screenshot=lambda *args, **kwargs: calls.append((args, kwargs)),
        camera=camera,
        renderer=SimpleNamespace(
            GetGradientBackground=lambda: True,
            GetBackground=lambda: (1.0, 1.0, 1.0),
            GetBackground2=lambda: (0.9, 0.9, 0.9),
            GetBackgroundAlpha=lambda: 0.0,
            GradientBackgroundOff=lambda: None,
            SetBackgroundAlpha=lambda _alpha: None,
            SetBackground=lambda *_color: None,
            SetBackground2=lambda *_color: None,
            SetGradientBackground=lambda _gradient: None,
        ),
        render=lambda: render_calls.append(True),
        window_size_context=lambda _size: nullcontext(),
    )
    viewport._plotter = plotter

    assert viewport.screenshot_size() == (720, 480)
    viewport.save_screenshot(
        "viewport.png",
        scale=1,
        transparent_background=True,
    )

    assert calls == [
        (
            ("viewport.png",),
            {
                "scale": 1,
                "window_size": None,
                "transparent_background": True,
                "return_img": False,
            },
        )
    ]
    assert plotter.camera is camera
    assert camera_calls == [("copy", copied_camera), ("modified",)]
    assert render_calls == [True, True]
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
        self.successes: list[tuple[str, object]] = []

    def _current_result_provider(self) -> object:
        return object()

    def _save_screenshot(self, *args, **kwargs) -> None:
        self.screenshot_calls.append((args, kwargs))

    def _set_state(self, message: str, timeout: int) -> None:
        self.status_calls.append((message, timeout))

    def _show_error(self, title: str, message: str) -> None:
        self.errors.append((title, message))

    def _show_save_success(self, content_name: str, path: object) -> None:
        self.successes.append((content_name, path))


def test_save_success_dialog_reports_selected_path(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(
        main_window_module.QMessageBox,
        "information",
        lambda *args: calls.append(args),
    )
    owner = object()

    FEMMainWindow._show_save_success(
        owner,
        "CSV 文件",
        Path("exports") / "selected.csv",
    )

    assert calls == [
        (
            owner,
            "保存成功",
            f"CSV 文件已保存成功\n\n{Path('exports') / 'selected.csv'}",
        )
    ]


def test_export_flow_cancels_combined_dialog_without_screenshot(monkeypatch) -> None:
    harness = _ExportHarness()
    dialog_calls = []

    class RejectedDialog:
        def __init__(self, *args, **kwargs) -> None:
            dialog_calls.append((args, kwargs))

        def exec(self) -> QDialog.DialogCode:
            return QDialog.DialogCode.Rejected

    monkeypatch.setattr(
        main_window_module,
        "ViewportImageExportDialog",
        RejectedDialog,
    )

    FEMMainWindow.export_viewport_image(harness)

    assert len(dialog_calls) == 1
    assert dialog_calls[0][0] == ((900, 500), harness)
    assert harness.screenshot_calls == []
    assert harness.successes == []


def test_export_flow_passes_options_and_keeps_success_feedback(monkeypatch) -> None:
    harness = _ExportHarness()
    created_with = []
    selected = ViewportImageExportOptions(1, (3600, 2000), False)

    class AcceptedDialog:
        options = selected
        target_path = "viewport.jpg"

        def __init__(self, *args, **kwargs) -> None:
            created_with.append((args, kwargs))

        def exec(self) -> QDialog.DialogCode:
            return QDialog.DialogCode.Accepted

    monkeypatch.setattr(
        main_window_module,
        "ViewportImageExportDialog",
        AcceptedDialog,
    )

    FEMMainWindow.export_viewport_image(harness)

    assert created_with[0][0] == ((900, 500), harness)
    assert harness.screenshot_calls == [
        (
            ("viewport.jpg",),
            {
                "scale": 1,
                "window_size": (3600, 2000),
                "transparent_background": False,
            },
        )
    ]
    assert harness.status_calls == [("视口图片保存完成", 5000)]
    assert harness.errors == []
    assert harness.successes == [("视口图片", "viewport.jpg")]


def test_export_flow_keeps_existing_error_feedback(monkeypatch) -> None:
    harness = _ExportHarness()
    harness.viewport.save_screenshot = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        RuntimeError("capture failed")
    )

    class AcceptedDialog:
        options = ViewportImageExportOptions(1, (1024, 768), True)
        target_path = "viewport.png"

        def __init__(self, *args, **kwargs) -> None:
            pass

        def exec(self) -> QDialog.DialogCode:
            return QDialog.DialogCode.Accepted

    monkeypatch.setattr(
        main_window_module,
        "ViewportImageExportDialog",
        AcceptedDialog,
    )

    FEMMainWindow.export_viewport_image(harness)

    assert harness.errors == [("导出视口图片失败", "capture failed")]
    assert harness.status_calls == []
    assert harness.successes == []


@pytest.mark.gui_native
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


@pytest.mark.gui_native
def test_scaled_screenshot_restores_pyvista_size_and_camera(tmp_path: Path) -> None:
    _application()
    viewport = FEMViewport()
    plotter = pv.Plotter(off_screen=True, window_size=(320, 240))
    plotter.add_mesh(pv.Sphere())
    plotter.camera_position = [(3.0, 2.0, 1.0), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)]
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
    output = tmp_path / "scaled.png"

    viewport.save_screenshot(str(output), scale=4)

    image = pv.read(output)
    after_camera = (
        tuple(plotter.camera.position),
        tuple(plotter.camera.focal_point),
        tuple(plotter.camera.up),
        plotter.camera.parallel_scale,
        tuple(plotter.camera.clipping_range),
        plotter.camera.view_angle,
    )
    assert image.dimensions[:2] == (1280, 960)
    assert tuple(plotter.window_size) == before_size
    assert after_camera == before_camera

    plotter.close()
    viewport._plotter = None
    viewport.close()


def test_scaled_screenshot_does_not_resize_live_render_window() -> None:
    _application()
    viewport = FEMViewport()
    screenshot_calls = []
    export_calls = []
    camera_state = object()
    camera = SimpleNamespace(
        copy=lambda: camera_state,
        DeepCopy=lambda _state: None,
        Modified=lambda: None,
    )
    plotter = SimpleNamespace(
        window_size=(720, 480),
        screenshot=lambda *args, **kwargs: screenshot_calls.append((args, kwargs)),
        camera=camera,
        renderer=SimpleNamespace(),
        render=lambda: None,
    )
    viewport._plotter = plotter
    viewport._save_offscreen_screenshot = lambda *args, **kwargs: (
        export_calls.append((args, kwargs))
    )

    viewport.save_screenshot("viewport.png", scale=4)

    assert screenshot_calls == []
    assert export_calls == [
        (
            ("viewport.png", (2880, 1920)),
            {
                "transparent_background": False,
            },
        )
    ]
    assert plotter.window_size == (720, 480)
    assert plotter.camera is camera
    viewport._plotter = None
    viewport.close()


@pytest.mark.gui_native
def test_scaled_screenshot_preserves_overlay_layout(tmp_path: Path) -> None:
    _application()
    viewport = FEMViewport()
    plotter = pv.Plotter(off_screen=True, window_size=(400, 300))
    plotter.render_window.SetDPI(144)
    mesh = pv.Sphere()
    mesh["height"] = mesh.points[:, 2]
    plotter.set_background("#eef7fb")
    plotter.add_mesh(
        mesh,
        scalars="height",
        scalar_bar_args={
            "title": "U, Magnitude",
            "position_x": 0.78,
            "position_y": 0.19,
            "width": 0.045,
            "height": 0.62,
            "title_font_size": 18,
            "label_font_size": 14,
            "unconstrained_font_size": True,
        },
    )
    plotter.add_axes()
    viewport._plotter = plotter
    normal = tmp_path / "normal.png"
    scaled = tmp_path / "scaled.png"

    viewport.save_screenshot(str(normal))
    viewport.save_screenshot(str(scaled), scale=2)

    normal_image = Image.open(normal).convert("RGB")
    scaled_image = Image.open(scaled).convert("RGB")
    reduced_image = scaled_image.resize(normal_image.size, Image.Resampling.LANCZOS)
    difference = ImageChops.difference(normal_image, reduced_image)
    assert ImageStat.Stat(difference).mean < [8.0, 8.0, 8.0]
    colorbar_box = (270, 0, 400, 300)
    colorbar_difference = ImageChops.difference(
        normal_image.crop(colorbar_box),
        reduced_image.crop(colorbar_box),
    )
    assert ImageStat.Stat(colorbar_difference).mean < [8.0, 8.0, 8.0]

    plotter.close()
    viewport._plotter = None
    viewport.close()


@pytest.mark.gui_native
def test_transparent_screenshot_handles_gradient_background(tmp_path: Path) -> None:
    _application()
    viewport = FEMViewport()
    plotter = pv.Plotter(off_screen=True, window_size=(320, 240))
    plotter.set_background("#eef7fb", top="#ffffff")
    mesh = pv.Sphere()
    mesh["height"] = mesh.points[:, 2]
    plotter.add_mesh(
        mesh,
        scalars="height",
        scalar_bar_args={
            "title_font_size": 18,
            "label_font_size": 14,
            "unconstrained_font_size": True,
        },
    )
    plotter.add_axes()
    viewport._plotter = plotter
    renderer = plotter.renderer
    scalar_bar = next(iter(plotter.scalar_bars.values()))
    before_font_sizes = (
        scalar_bar.GetTitleTextProperty().GetFontSize(),
        scalar_bar.GetLabelTextProperty().GetFontSize(),
        scalar_bar.GetAnnotationTextProperty().GetFontSize(),
        scalar_bar.GetTextPad(),
        scalar_bar.GetAnnotationLeaderPadding(),
        scalar_bar.GetVerticalTitleSeparation(),
    )
    before_background = (
        renderer.GetGradientBackground(),
        renderer.GetBackground(),
        renderer.GetBackground2(),
        renderer.GetBackgroundAlpha(),
    )
    output = tmp_path / "transparent.png"

    viewport.save_screenshot(
        str(output),
        window_size=(640, 480),
        transparent_background=True,
    )

    image = pv.read(output)
    pixels = image.point_data["PNGImage"]
    after_background = (
        renderer.GetGradientBackground(),
        renderer.GetBackground(),
        renderer.GetBackground2(),
        renderer.GetBackgroundAlpha(),
    )
    after_font_sizes = (
        scalar_bar.GetTitleTextProperty().GetFontSize(),
        scalar_bar.GetLabelTextProperty().GetFontSize(),
        scalar_bar.GetAnnotationTextProperty().GetFontSize(),
        scalar_bar.GetTextPad(),
        scalar_bar.GetAnnotationLeaderPadding(),
        scalar_bar.GetVerticalTitleSeparation(),
    )
    assert pixels.shape[1] == 4
    assert pixels[:, 3].min() == 0
    assert pixels[:, 3].max() == 255
    assert after_background == before_background
    assert after_font_sizes == before_font_sizes

    plotter.close()
    viewport._plotter = None
    viewport.close()
