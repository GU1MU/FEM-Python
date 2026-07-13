from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from fem_gui.main_window import FEMMainWindow, initial_display_policy
from fem_gui.visualization.model_adapter import build_model_geometry
from fem_gui.visualization.symbols import SymbolSettings
from fem_gui.widgets import viewport as viewport_module
from fem_gui.widgets.viewport import FEMViewport
from tests.helpers.model_builders import make_static_pull_truss_model


def _application() -> QApplication:
    return QApplication.instance() or QApplication([])


class _Camera:
    def __init__(self, *, parallel=True, scale=1.0, distance=10.0, angle=60.0) -> None:
        self.parallel = parallel
        self.parallel_scale = scale
        self.distance = distance
        self.angle = angle

    def GetParallelProjection(self):
        return self.parallel

    def GetParallelScale(self):
        return self.parallel_scale

    def GetPosition(self):
        return (0.0, 0.0, self.distance)

    def GetFocalPoint(self):
        return (0.0, 0.0, 0.0)

    def GetViewAngle(self):
        return self.angle


class _Plotter:
    def __init__(self, camera=None) -> None:
        self.render_count = 0
        self.camera = camera
        self.window_size = (800, 400)

    def render(self) -> None:
        self.render_count += 1


class _ViewCamera:
    def __init__(self) -> None:
        self.up = None
        self.orthogonalized = False
        self.position = (0.0, 0.0, 10.0)
        self.focal_point = (0.0, 0.0, 0.0)

    def SetViewUp(self, x, y, z) -> None:
        self.up = (x, y, z)

    def OrthogonalizeViewUp(self) -> None:
        self.orthogonalized = True

    def GetPosition(self):
        return self.position

    def GetFocalPoint(self):
        return self.focal_point

    def SetPosition(self, x, y, z) -> None:
        self.position = (x, y, z)


class _ViewPlotter:
    def __init__(self) -> None:
        self.camera = _ViewCamera()
        self.calls = []

    def view_xy(self) -> None:
        self.calls.append("xy")

    def view_xz(self) -> None:
        self.calls.append("xz")

    def view_yz(self) -> None:
        self.calls.append("yz")

    def view_isometric(self) -> None:
        self.calls.append("iso")

    def reset_camera_clipping_range(self) -> None:
        self.calls.append("clip")

    def render(self) -> None:
        self.calls.append("render")


def test_boundary_cache_reuses_step_and_is_cleared_by_new_model(monkeypatch):
    _application()
    model = make_static_pull_truss_model()
    geometry = build_model_geometry(model)
    viewport = FEMViewport()
    viewport.set_model(model, geometry, refresh_symbols=False, render=False)
    camera = _Camera(scale=1.0)
    viewport._plotter = _Plotter(camera)
    monkeypatch.setattr(viewport_module, "_pyvista", object())
    calls = []
    original = viewport_module.boundary_for_step

    def counted(current_model, step_name):
        calls.append((current_model, step_name))
        return original(current_model, step_name)

    monkeypatch.setattr(viewport_module, "boundary_for_step", counted)
    viewport.set_symbol_settings(SymbolSettings(
        step_name="pull", show_constraints=False, show_nodal_loads=False,
        show_edge_loads=False, show_surface_loads=False,
    ))
    viewport.show_boundary_and_loads("pull")
    assert len(calls) == 1
    original_scale = viewport._last_symbol_scale

    camera.parallel_scale = 2.0
    viewport._refresh_symbols_for_camera(render=False)
    assert len(calls) == 1
    assert viewport._last_symbol_scale == pytest.approx(2.0 * original_scale)

    new_model = make_static_pull_truss_model(load=25.0)
    viewport.set_model(
        new_model, build_model_geometry(new_model), refresh_symbols=False, render=False
    )
    assert viewport._boundary_cache == {}


def test_world_per_pixel_supports_parallel_and_perspective_cameras():
    _application()
    viewport = FEMViewport()
    viewport._plotter = _Plotter(_Camera(parallel=True, scale=4.0))
    assert viewport._world_per_pixel() == pytest.approx(0.02)

    viewport._plotter = _Plotter(_Camera(parallel=False, distance=10.0, angle=60.0))
    expected = 20.0 * np.tan(np.deg2rad(30.0)) / 400.0
    assert viewport._world_per_pixel() == pytest.approx(expected)


@pytest.mark.parametrize(
    ("view", "base", "up", "position"),
    [
        ("top", "xy", (0.0, 1.0, 0.0), (0.0, 0.0, -10.0)),
        ("bottom", "xy", (1.0, 0.0, 0.0), (0.0, 0.0, 10.0)),
        ("front", "xz", (0.0, 0.0, 1.0), (0.0, 10.0, 0.0)),
        ("back", "xz", (1.0, 0.0, 0.0), (0.0, -10.0, 0.0)),
        ("left", "yz", (0.0, 0.0, 1.0), (-10.0, 0.0, 0.0)),
        ("right", "yz", (0.0, 1.0, 0.0), (10.0, 0.0, 0.0)),
    ],
)
def test_coordinate_view_keeps_plane_and_swaps_screen_axes(view, base, up, position):
    _application()
    viewport = FEMViewport()
    plotter = _ViewPlotter()
    viewport._plotter = plotter

    viewport.set_view(view)

    assert base in plotter.calls
    assert plotter.camera.up == up
    assert plotter.camera.position == position
    assert plotter.camera.orthogonalized
    assert "clip" in plotter.calls


def test_model_load_batches_symbol_rebuild_and_final_render(monkeypatch):
    _application()
    model = make_static_pull_truss_model()
    geometry = build_model_geometry(model)
    window = FEMMainWindow()
    rebuilds = []
    renders = []
    original_show = window.viewport.show_boundary_and_loads

    def counted_show(*args, **kwargs):
        rebuilds.append(1)
        return original_show(*args, **kwargs)

    monkeypatch.setattr(window.viewport, "show_boundary_and_loads", counted_show)
    monkeypatch.setattr(window.viewport, "render", lambda: renders.append(1))
    window._model_loaded(Path("batch.inp"), (model, geometry))

    assert len(rebuilds) == 1
    assert len(renders) == 1
    window.close()


def test_large_model_first_display_policy_has_explicit_thresholds():
    assert initial_display_policy(100_000, 200_000)["show_edges"]
    assert not initial_display_policy(100_001, 10)["show_edges"]
    assert initial_display_policy(200_000, 10)["show_symbols"]
    assert not initial_display_policy(200_001, 10)["show_symbols"]
    assert initial_display_policy(10, 200_001)["simplified"]
    assert not initial_display_policy(10, 200_001)["show_nodes"]
    assert not initial_display_policy(10, 200_001)["show_labels"]
