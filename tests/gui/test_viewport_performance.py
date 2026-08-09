from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from fem.core.model import FEMModel, GravityLoad
import fem_gui.main_window as main_window_module
from fem_gui.main_window import FEMMainWindow, initial_display_policy
from fem_gui.visualization.model_adapter import build_model_geometry
from fem_gui.visualization.symbols import SymbolSettings
from fem_gui.widgets import viewport as viewport_module
from fem_gui.widgets.viewport import FEMViewport
from tests.helpers.mesh_builders import (
    make_selection_hex_mesh,
    make_selection_quad_mesh,
)
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


class _GravityActor:
    def __init__(self) -> None:
        self.pickable = True

    def SetPickable(self, pickable: bool) -> None:
        self.pickable = bool(pickable)


class _GravityPlotter(_Plotter):
    def __init__(self, camera=None) -> None:
        super().__init__(camera)
        self.arrow_calls = []
        self.gravity_actor = _GravityActor()

    def add_arrows(self, origins, vectors, **kwargs):
        self.arrow_calls.append(
            (np.asarray(origins), np.asarray(vectors), kwargs)
        )
        return self.gravity_actor


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

    def reset_camera(self, *, bounds=None, render=True) -> None:
        self.calls.append(("reset", bounds, render))
        if bounds is None:
            return
        focal = np.asarray(
            (
                0.5 * (bounds[0] + bounds[1]),
                0.5 * (bounds[2] + bounds[3]),
                0.5 * (bounds[4] + bounds[5]),
            ),
            dtype=float,
        )
        direction = (
            np.asarray(self.camera.position, dtype=float)
            - np.asarray(self.camera.focal_point, dtype=float)
        )
        direction /= np.linalg.norm(direction)
        self.camera.focal_point = tuple(focal)
        self.camera.position = tuple(focal + 10.0 * direction)

    def add_mesh(self, _data, **kwargs):
        self.calls.append(("mesh", kwargs["name"]))
        return _VisibilityActor()

    def render(self) -> None:
        self.calls.append("render")


class _VisibilityActor:
    def __init__(self) -> None:
        self.visible = True

    def SetVisibility(self, visible: bool) -> None:
        self.visible = bool(visible)


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
    assert viewport._beam_frame_cache == {}


def test_gravity_uses_one_centered_yellow_direction_arrow(monkeypatch):
    _application()
    model = make_static_pull_truss_model()
    model.steps[0].gravity_loads = (GravityLoad((0.0, -9.81, 0.0)),)
    geometry = build_model_geometry(model)
    viewport = FEMViewport()
    viewport.set_model(model, geometry, refresh_symbols=False, render=False)
    plotter = _GravityPlotter(_Camera(scale=1.0))
    viewport._plotter = plotter
    monkeypatch.setattr(viewport_module, "_pyvista", object())
    viewport.set_symbol_settings(
        SymbolSettings(
            step_name="pull",
            show_constraints=False,
            show_nodal_loads=False,
            show_edge_loads=False,
            show_surface_loads=False,
            show_line_loads=False,
        ),
        refresh=False,
    )

    viewport.show_boundary_and_loads("pull", render=False)

    assert len(plotter.arrow_calls) == 1
    origins, directions, options = plotter.arrow_calls[0]
    center = 0.5 * (
        np.min(geometry.points, axis=0)
        + np.max(geometry.points, axis=0)
    )
    assert directions[0] == pytest.approx((0.0, -1.0, 0.0))
    assert origins[0] + 0.5 * options["mag"] * directions[0] == pytest.approx(
        center
    )
    assert options["color"] == "#FFD400"
    assert options["name"] == "gravity"
    assert plotter.gravity_actor.pickable is False


def test_symbol_sampling_density_override_is_explicit_and_reversible():
    _application()
    viewport = FEMViewport()
    viewport.set_symbol_settings(
        SymbolSettings(sampling_density="high"),
        refresh=False,
    )

    assert viewport._effective_symbol_sampling_density() == "high"
    viewport.set_symbol_sampling_density_override("low")
    assert viewport._effective_symbol_sampling_density() == "low"
    viewport.set_symbol_sampling_density_override(None)
    assert viewport._effective_symbol_sampling_density() == "high"
    with pytest.raises(ValueError, match="符号采样密度"):
        viewport.set_symbol_sampling_density_override("invalid")
    viewport.close()


def test_world_per_pixel_supports_parallel_and_perspective_cameras():
    _application()
    viewport = FEMViewport()
    viewport._plotter = _Plotter(_Camera(parallel=True, scale=4.0))
    assert viewport._world_per_pixel() == pytest.approx(0.02)

    viewport._plotter = _Plotter(_Camera(parallel=False, distance=10.0, angle=60.0))
    expected = 20.0 * np.tan(np.deg2rad(30.0)) / 400.0
    assert viewport._world_per_pixel() == pytest.approx(expected)


def test_line_elements_are_drawn_thicker_than_continuum_edges():
    _application()
    viewport = FEMViewport()
    viewport._geometry = build_model_geometry(make_static_pull_truss_model())
    assert viewport._element_line_width() == 5
    assert viewport._line_render_options() == {"render_lines_as_tubes": True}
    assert viewport._node_point_size() == 11
    assert viewport._mesh_layer_color(viewport._visual_palette()) == "#3F6F8C"
    assert viewport._element_layer_color(viewport._visual_palette()) == "#3F6F8C"
    assert viewport._node_layer_color(viewport._visual_palette()) == "#9A6F3F"
    node_labels = viewport._label_render_options("node")
    element_labels = viewport._label_render_options("element")
    assert node_labels["always_visible"]
    assert node_labels["show_points"] is False
    assert node_labels["justification_vertical"] == "bottom"
    assert element_labels["justification_vertical"] == "top"

    viewport._geometry = None
    assert viewport._element_line_width() == 1
    assert viewport._line_render_options() == {}
    assert viewport._node_point_size() == 7
    assert viewport._mesh_layer_color(viewport._visual_palette()) == "#d8dde2"
    assert viewport._element_layer_color(viewport._visual_palette()) == "#3F6F8C"
    assert viewport._node_layer_color(viewport._visual_palette()) == "#9A6F3F"


@pytest.mark.parametrize(
    "mesh_factory",
    [make_selection_quad_mesh, make_selection_hex_mesh],
)
def test_2d_and_3d_meshes_share_element_and_node_colors(mesh_factory):
    _application()
    viewport = FEMViewport()
    viewport._geometry = build_model_geometry(FEMModel(mesh_factory()))
    palette = viewport._visual_palette()

    assert not viewport._is_line_mesh()
    assert viewport._mesh_layer_color(palette) == "#d8dde2"
    assert viewport._element_layer_color(palette) == "#3F6F8C"
    assert viewport._node_layer_color(palette) == "#9A6F3F"


@pytest.mark.parametrize(
    ("view", "up", "direction"),
    [
        ("top", (0.0, 1.0, 0.0), (0.0, 0.0, -1.0)),
        ("bottom", (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
        ("front", (0.0, 0.0, 1.0), (0.0, 1.0, 0.0)),
        ("back", (1.0, 0.0, 0.0), (0.0, -1.0, 0.0)),
        ("left", (0.0, 0.0, 1.0), (-1.0, 0.0, 0.0)),
        ("right", (0.0, 1.0, 0.0), (1.0, 0.0, 0.0)),
        (
            "iso",
            (-1.0, 2.0, -1.0),
            tuple(np.asarray((1.0, 1.0, 1.0)) / np.sqrt(3.0)),
        ),
    ],
)
def test_coordinate_view_keeps_axes_and_fits_off_origin_model(
    view,
    up,
    direction,
):
    _application()
    viewport = FEMViewport()
    plotter = _ViewPlotter()
    viewport._plotter = plotter
    viewport._grid = SimpleNamespace(
        points=np.asarray(
            (
                (10.0, -4.0, 2.0),
                (14.0, 8.0, 6.0),
            )
        )
    )

    viewport.set_view(view)

    bounds = (10.0, 14.0, -4.0, 8.0, 2.0, 6.0)
    focal = np.asarray((12.0, 2.0, 4.0))
    actual_direction = (
        np.asarray(plotter.camera.position)
        - np.asarray(plotter.camera.focal_point)
    )
    actual_direction /= np.linalg.norm(actual_direction)

    assert plotter.calls == [("reset", bounds, False), "render"]
    assert plotter.camera.focal_point == pytest.approx(focal)
    assert plotter.camera.up == up
    assert actual_direction == pytest.approx(direction)
    assert plotter.camera.orthogonalized


def test_base_model_layers_fit_stable_bounds_without_intermediate_render(
    monkeypatch,
):
    _application()
    viewport = FEMViewport()
    plotter = _ViewPlotter()
    viewport._plotter = plotter
    viewport._grid = SimpleNamespace(
        points=np.asarray(
            (
                (20.0, -8.0, 3.0),
                (24.0, 4.0, 9.0),
            )
        )
    )
    monkeypatch.setattr(
        viewport,
        "_refresh_node_layer",
        lambda *, render: None,
    )
    monkeypatch.setattr(
        viewport,
        "_refresh_labels",
        lambda *, render=False: None,
    )

    viewport._add_base_layers(reset_camera=True, render=False)

    bounds = (20.0, 24.0, -8.0, 4.0, 3.0, 9.0)
    assert plotter.calls == [
        ("mesh", "mesh_surface"),
        ("mesh", "element_edges"),
        ("reset", bounds, False),
    ]


def test_large_model_base_layers_skip_hidden_element_edges(monkeypatch):
    _application()
    viewport = FEMViewport()
    plotter = _ViewPlotter()
    viewport._plotter = plotter
    viewport._grid = SimpleNamespace(
        points=np.asarray(((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)))
    )
    viewport._show_edges = False
    monkeypatch.setattr(
        viewport,
        "_refresh_node_layer",
        lambda *, render: None,
    )
    monkeypatch.setattr(
        viewport,
        "_refresh_labels",
        lambda *, render=False: None,
    )

    viewport._add_base_layers(reset_camera=False, render=False)

    assert plotter.calls == [("mesh", "mesh_surface")]
    assert "element_edges" not in viewport._actors
    viewport.close()


def test_element_edges_are_created_once_when_enabled_after_initial_load():
    _application()
    viewport = FEMViewport()
    plotter = _ViewPlotter()
    viewport._plotter = plotter
    viewport._grid = object()
    viewport._show_edges = False

    viewport.set_edges_visible(True, render=False)
    viewport.set_edges_visible(True, render=False)

    assert plotter.calls == [("mesh", "element_edges")]
    assert viewport._actors["element_edges"].visible
    viewport.close()


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
    assert window.actions["nodes"].isChecked()
    assert window.viewport._show_nodes
    window.actions["node_labels"].trigger()
    window.actions["element_labels"].trigger()
    assert window.viewport._show_node_labels
    assert window.viewport._show_element_labels
    window.close()


def test_large_model_first_display_policy_has_explicit_thresholds():
    assert initial_display_policy(100_000, 200_000)["show_edges"]
    assert not initial_display_policy(100_001, 10)["show_edges"]
    assert initial_display_policy(100_000, 10)["show_symbols"]
    assert initial_display_policy(100_001, 10)["show_symbols"]
    assert initial_display_policy(100_001, 10)["reduce_symbols"]
    assert initial_display_policy(10, 200_001)["simplified"]
    assert initial_display_policy(10, 200_001)["show_symbols"]
    assert initial_display_policy(10, 200_001)["reduce_symbols"]
    assert not initial_display_policy(10, 200_000)["reduce_symbols"]
    assert not initial_display_policy(10, 200_001)["show_nodes"]
    assert not initial_display_policy(10, 200_001)["show_labels"]
    assert initial_display_policy(10, 20_000, line_mesh=True)["show_nodes"]
    assert not initial_display_policy(10, 20_001, line_mesh=True)["show_nodes"]


def test_large_model_load_uses_sparse_symbols_until_user_changes_settings(
    monkeypatch,
):
    _application()
    model = make_static_pull_truss_model()
    geometry = build_model_geometry(model)
    window = FEMMainWindow()
    monkeypatch.setattr(
        main_window_module,
        "initial_display_policy",
        lambda *_args, **_kwargs: {
            "show_edges": False,
            "show_symbols": True,
            "reduce_symbols": True,
            "show_nodes": False,
            "show_labels": False,
            "simplified": True,
        },
    )
    monkeypatch.setattr(window.viewport, "render", lambda: None)

    window._model_loaded(Path("large.inp"), (model, geometry))

    assert window.actions["symbols"].isChecked()
    assert window.viewport._effective_symbol_sampling_density() == "low"

    window._apply_symbol_settings(
        SymbolSettings(
            step_name=window._current_step_name,
            sampling_density="high",
        )
    )

    assert window.viewport._effective_symbol_sampling_density() == "high"
    window.close()
