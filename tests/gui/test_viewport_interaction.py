from __future__ import annotations

import os
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtWidgets import QApplication, QWidget

from fem.application import RegionRef
from fem.core.mesh import Element3D, Mesh3D, Node3D
from fem.core.model import FEMModel
from fem.geometry import LogicalEntityRef
from fem_gui.visualization.model_adapter import build_model_geometry
from fem_gui.widgets.viewport import (
    BEAM_FRAME_GLYPH_LIMIT,
    FEMViewport,
    PickHit,
    _effective_line_load_vector,
)


def _application() -> QApplication:
    return QApplication.instance() or QApplication([])


class _Camera:
    def __init__(self) -> None:
        self.position = np.array((0.0, 0.0, 10.0), dtype=float)
        self.focal = np.zeros(3, dtype=float)
        self.up = np.array((0.0, 1.0, 0.0), dtype=float)
        self.orthogonalized = False

    def GetPosition(self):
        return tuple(self.position)

    def GetFocalPoint(self):
        return tuple(self.focal)

    def GetViewUp(self):
        return tuple(self.up)

    def SetPosition(self, *position) -> None:
        self.position = np.asarray(position, dtype=float)

    def SetViewUp(self, *up) -> None:
        self.up = np.asarray(up, dtype=float)

    def OrthogonalizeViewUp(self) -> None:
        self.orthogonalized = True


class _Plotter(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.resize(800, 600)
        self.camera = _Camera()
        self.clipping_resets = 0
        self.render_count = 0

    def reset_camera_clipping_range(self) -> None:
        self.clipping_resets += 1

    def render(self) -> None:
        self.render_count += 1


def test_hover_reuses_preselection_for_the_same_semantic_target(
    monkeypatch,
) -> None:
    _application()
    viewport = FEMViewport()
    viewport._plotter = SimpleNamespace(
        height=lambda: 400,
        _getPixelRatio=lambda: 1.0,
        devicePixelRatioF=lambda: 1.0,
    )
    reference = LogicalEntityRef("body:domain")
    viewport._geometry_pick_to_ref = {101: reference, 102: reference}
    hits = iter(
        (
            PickHit(
                "geometry_body",
                101,
                "geometry_surface",
                (40.0, 50.0),
                (0.0, 0.0, 0.0),
                vtk_cell_id=1,
            ),
            PickHit(
                "geometry_body",
                102,
                "geometry_surface",
                (80.0, 90.0),
                (0.5, 0.5, 0.0),
                vtk_cell_id=2,
            ),
        )
    )
    shown = []
    monkeypatch.setattr(viewport, "_resolve_pick", lambda *_args: next(hits))
    monkeypatch.setattr(viewport, "_show_preselection", shown.append)

    viewport._pending_hover_position = (40.0, 349.0)
    viewport._update_preselection()
    viewport._pending_hover_position = (80.0, 309.0)
    viewport._update_preselection()

    assert len(shown) == 1
    assert viewport._hover_hit is not None
    assert viewport._hover_hit.vtk_cell_id == 2
    viewport._plotter = None
    viewport.close()


def test_mesh_body_hover_groups_elements_by_part_owner(monkeypatch) -> None:
    _application()
    viewport = FEMViewport()
    viewport._plotter = SimpleNamespace(
        height=lambda: 400,
        _getPixelRatio=lambda: 1.0,
        devicePixelRatioF=lambda: 1.0,
    )
    viewport._mesh_body_owner_by_element_id = {10: "P1", 11: "P1"}
    hits = iter(
        (
            PickHit(
                "mesh_body",
                10,
                "model_pick_grid",
                (40.0, 50.0),
                (0.0, 0.0, 0.0),
                vtk_cell_id=1,
            ),
            PickHit(
                "mesh_body",
                11,
                "model_pick_grid",
                (80.0, 90.0),
                (0.5, 0.5, 0.0),
                vtk_cell_id=2,
            ),
        )
    )
    shown = []
    monkeypatch.setattr(viewport, "_resolve_pick", lambda *_args: next(hits))
    monkeypatch.setattr(viewport, "_show_preselection", shown.append)

    viewport._pending_hover_position = (40.0, 349.0)
    viewport._update_preselection()
    viewport._pending_hover_position = (80.0, 309.0)
    viewport._update_preselection()

    assert len(shown) == 1
    assert viewport._hover_hit is not None
    assert viewport._hover_hit.pick_id == 11
    viewport._plotter = None
    viewport.close()


class _FitPlotter:
    def __init__(self) -> None:
        self.reset_calls = []
        self.render_count = 0

    def reset_camera(self, *, bounds=None, render=True) -> None:
        self.reset_calls.append((bounds, render))

    def render(self) -> None:
        self.render_count += 1


class _MouseEvent:
    def __init__(
        self,
        event_type: QEvent.Type,
        *,
        x: float,
        y: float,
        button: Qt.MouseButton,
        buttons: Qt.MouseButton,
        modifiers: Qt.KeyboardModifier,
    ) -> None:
        self._type = event_type
        self._position = QPointF(x, y)
        self._button = button
        self._buttons = buttons
        self._modifiers = modifiers

    def type(self):
        return self._type

    def position(self):
        return self._position

    def button(self):
        return self._button

    def buttons(self):
        return self._buttons

    def modifiers(self):
        return self._modifiers


def _viewport() -> tuple[FEMViewport, _Plotter]:
    _application()
    viewport = FEMViewport()
    plotter = _Plotter()
    viewport._plotter = plotter
    return viewport, plotter


def test_trackball_projects_center_and_outer_ring() -> None:
    viewport, _plotter = _viewport()

    assert viewport._trackball_point(400.0, 300.0) == pytest.approx((0.0, 0.0, 1.0))
    outer = viewport._trackball_point(800.0, 300.0)

    assert np.linalg.norm(outer) == pytest.approx(1.0)
    assert outer[2] == pytest.approx(0.0)


def test_ctrl_alt_left_drag_rotates_freely_around_focal_point() -> None:
    viewport, plotter = _viewport()
    modifiers = Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.AltModifier
    press = _MouseEvent(
        QEvent.Type.MouseButtonPress,
        x=400.0,
        y=300.0,
        button=Qt.MouseButton.LeftButton,
        buttons=Qt.MouseButton.LeftButton,
        modifiers=modifiers,
    )
    move = _MouseEvent(
        QEvent.Type.MouseMove,
        x=500.0,
        y=250.0,
        button=Qt.MouseButton.NoButton,
        buttons=Qt.MouseButton.LeftButton,
        modifiers=modifiers,
    )

    assert viewport.eventFilter(plotter, press)
    assert viewport.eventFilter(plotter, move)
    assert not np.allclose(plotter.camera.position, (0.0, 0.0, 10.0))
    assert plotter.camera.orthogonalized
    assert plotter.clipping_resets == 1
    assert plotter.render_count == 1


def test_trackball_outer_ring_rolls_camera_without_moving_focal_distance() -> None:
    viewport, plotter = _viewport()
    viewport._trackball_vector = viewport._trackball_point(800.0, 300.0)
    original_position = plotter.camera.position.copy()

    viewport._rotate_trackball(400.0, 0.0)

    assert plotter.camera.position == pytest.approx(original_position)
    assert not np.allclose(plotter.camera.up, (0.0, 1.0, 0.0))


def test_model_clear_resets_partial_mouse_gesture() -> None:
    viewport, _plotter = _viewport()
    viewport._selection_press_position = (10.0, 20.0)
    viewport._selection_dragged = True
    viewport._abaqus_view_button = Qt.MouseButton.LeftButton
    viewport._trackball_vector = np.ones(3)

    viewport.clear_model()

    assert viewport._selection_press_position is None
    assert not viewport._selection_dragged
    assert viewport._abaqus_view_button is None
    assert viewport._trackball_vector is None


def test_fit_uses_stable_model_bounds_and_renders_once() -> None:
    _application()
    viewport = FEMViewport()
    plotter = _FitPlotter()
    viewport._plotter = plotter
    viewport._grid = SimpleNamespace(
        points=np.asarray(
            (
                (10.0, -4.0, 2.0),
                (14.0, 8.0, 6.0),
                (11.0, 3.0, 5.0),
            )
        )
    )

    viewport.fit()
    viewport.fit()

    expected = (10.0, 14.0, -4.0, 8.0, 2.0, 6.0)
    assert plotter.reset_calls == [(expected, False), (expected, False)]
    assert plotter.render_count == 2


def test_fit_prefers_deformed_result_bounds_over_base_grid() -> None:
    _application()
    viewport = FEMViewport()
    plotter = _FitPlotter()
    viewport._plotter = plotter
    viewport._grid = SimpleNamespace(
        points=np.asarray(((0.0, 0.0, 0.0), (1.0, 1.0, 0.0)))
    )
    viewport._result_grid = SimpleNamespace(
        points=np.asarray(((20.0, 10.0, -2.0), (30.0, 15.0, 4.0)))
    )
    viewport._actors["result"] = object()

    viewport.fit()

    assert plotter.reset_calls == [
        ((20.0, 30.0, 10.0, 15.0, -2.0, 4.0), False)
    ]


def test_child_widget_mouse_move_uses_plotter_coordinates() -> None:
    viewport, plotter = _viewport()
    child = QWidget(plotter)
    child.move(100, 50)
    child.resize(200, 100)
    viewport._picker_event_targets = {child}
    press = _MouseEvent(
        QEvent.Type.MouseButtonPress,
        x=10.0,
        y=10.0,
        button=Qt.MouseButton.LeftButton,
        buttons=Qt.MouseButton.LeftButton,
        modifiers=Qt.KeyboardModifier.NoModifier,
    )
    move = _MouseEvent(
        QEvent.Type.MouseMove,
        x=11.0,
        y=10.0,
        button=Qt.MouseButton.NoButton,
        buttons=Qt.MouseButton.LeftButton,
        modifiers=Qt.KeyboardModifier.NoModifier,
    )

    assert viewport.eventFilter(child, press)
    assert viewport.eventFilter(child, move)
    assert not viewport._selection_dragged


class _FrameActor:
    def SetPickable(self, _value) -> None:
        return


class _FramePlotter:
    def __init__(self) -> None:
        self.camera = None
        self.arrow_calls = []
        self.label_calls = []
        self.removed = []
        self.render_count = 0

    def add_arrows(self, origins, vectors, **kwargs):
        self.arrow_calls.append(
            (np.asarray(origins), np.asarray(vectors), kwargs)
        )
        return _FrameActor()

    def add_point_labels(self, points, labels, **kwargs):
        self.label_calls.append(
            (np.asarray(points), tuple(labels), kwargs)
        )
        return _FrameActor()

    def remove_actor(self, actor, **_kwargs):
        self.removed.append(actor)

    def render(self) -> None:
        self.render_count += 1


def _many_beam_model(count: int) -> FEMModel:
    nodes = []
    elements = []
    for index in range(count):
        start = 2 * index + 1
        nodes.extend(
            (
                Node3D(start, 0.0, float(index), 0.0),
                Node3D(start + 1, 1.0, float(index), 0.0),
            )
        )
        elements.append(
            Element3D(index + 1, [start, start + 1], "Beam2", {})
        )
    return FEMModel(
        Mesh3D(nodes, elements, dofs_per_node=6),
        name="many beams",
    )


def test_selected_beam_frame_preview_is_cached_and_glyph_bounded() -> None:
    _application()
    model = _many_beam_model(BEAM_FRAME_GLYPH_LIMIT + 11)
    geometry = build_model_geometry(model)
    frame = SimpleNamespace(
        source="explicit",
        local_x=np.asarray((1.0, 0.0, 0.0)),
        local_y=np.asarray((0.0, 1.0, 0.0)),
        local_z=np.asarray((0.0, 0.0, 1.0)),
        rotation=np.eye(3),
    )
    report = SimpleNamespace(
        entries=tuple(
            SimpleNamespace(element_id=index + 1, frame=frame)
            for index in range(BEAM_FRAME_GLYPH_LIMIT + 11)
        )
    )
    queried = []

    def query(target):
        queried.append(target)
        return report

    viewport = FEMViewport()
    viewport.set_model(
        model,
        geometry,
        refresh_symbols=False,
        render=False,
        effective_frame_query=query,
    )
    plotter = _FramePlotter()
    viewport._plotter = plotter
    target = RegionRef("element_set", "BEAMS")

    viewport.show_beam_frame_preview(target)
    viewport.show_beam_frame_preview(target)

    assert queried == [target]
    assert all(
        len(origins) == BEAM_FRAME_GLYPH_LIMIT
        for origins, _vectors, _kwargs in plotter.arrow_calls[:3]
    )
    assert {
        call[2]["name"]
        for call in plotter.arrow_calls[:3]
    } == {
        "beam_frame_x_explicit",
        "beam_frame_y_explicit",
        "beam_frame_z_explicit",
    }
    assert len(plotter.label_calls[0][1]) == (
        3 * BEAM_FRAME_GLYPH_LIMIT
    )


def test_local_line_load_vector_uses_resolved_rotation_without_fallback() -> None:
    rotation = np.asarray(
        (
            (1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
            (0.0, -1.0, 0.0),
        )
    )
    frame = SimpleNamespace(rotation=rotation)

    assert _effective_line_load_vector(
        (0.0, 2.0, 0.0),
        "local",
        frame,
    ) == pytest.approx((0.0, 0.0, 2.0))
    assert _effective_line_load_vector(
        (0.0, 2.0, 0.0),
        "global",
        None,
    ) == pytest.approx((0.0, 2.0, 0.0))
    assert (
        _effective_line_load_vector(
            (0.0, 2.0, 0.0),
            "local",
            None,
        )
        is None
    )
