from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtWidgets import QApplication, QWidget

from fem_gui.widgets.viewport import FEMViewport


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
