"""Mouse gestures and paired Qt/VTK interaction resources."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from PySide6.QtCore import QEvent, QObject, QPoint, QTimer, Qt
from PySide6.QtWidgets import QWidget

from .viewport_camera import rotate_camera_trackball, trackball_point


@dataclass(frozen=True, slots=True)
class PointerInput:
    position: tuple[float, float]
    vtk_position: tuple[int, int]
    button: Qt.MouseButton
    buttons: Qt.MouseButton
    modifiers: Qt.KeyboardModifier
    global_position: QPoint
    press_position: tuple[float, float] | None = None
    dragged: bool = False


@dataclass(frozen=True, slots=True)
class SelectionInputPort:
    hover: Callable[[tuple[float, float]], None]
    clear_hover: Callable[[bool], bool]
    click: Callable[[float, float], None]
    box: Callable[[tuple[float, float], tuple[float, float]], None]
    confirm: Callable[[], None]
    cancel: Callable[[], None]
    camera_finished: Callable[[], None]
    render: Callable[[], None]


@dataclass(frozen=True, slots=True)
class ToolInputPort:
    active: Callable[[], bool]
    key: Callable[[int], bool]
    press: Callable[[PointerInput], bool]
    move: Callable[[PointerInput], None]
    release: Callable[[PointerInput], None]
    leave: Callable[[], None]
    reset: Callable[[], bool]


@dataclass(frozen=True, slots=True)
class InteractionBackend:
    widget: QWidget
    interactor: Any | None
    renderer: Any | None
    camera: Any | None
    to_vtk: Callable[[float, float], tuple[int, int]]
    reset_clipping: Callable[[], None]


class ViewportInteraction(QObject):
    """Own input bindings, pending frames and gesture state for one backend."""

    def __init__(
        self, selection: SelectionInputPort, tools: ToolInputPort,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._selection = selection
        self._tools = tools
        self._backend: InteractionBackend | None = None
        self._attachment_serial = 0
        self._event_targets: set[QWidget] = set()
        self._observer_id: int | None = None
        self._press_position: tuple[float, float] | None = None
        self._dragged = False
        self._camera_button: Qt.MouseButton | None = None
        self._trackball_vector: np.ndarray | None = None
        self._pending_hover_position: tuple[float, float] | None = None
        self._rubber_band: _SelectionRubberBand | None = None
        self._hover_timer = QTimer(self)
        self._hover_timer.setSingleShot(True)
        self._hover_timer.setInterval(40)
        self._hover_timer.timeout.connect(self._deliver_hover)
        self._band_render_timer = QTimer(self)
        self._band_render_timer.setSingleShot(True)
        self._band_render_timer.setInterval(16)
        self._band_render_timer.timeout.connect(self._render_band)

    def attach(self, backend: InteractionBackend) -> None:
        if self._backend == backend:
            return
        self.detach()
        self._backend = backend
        try:
            self._event_targets = {backend.widget, *backend.widget.findChildren(QWidget)}
            for target in self._event_targets:
                target.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
                target.setMouseTracking(True)
                target.installEventFilter(self)
            if backend.interactor is not None:
                serial = self._attachment_serial

                def camera_finished(_object=None, _event=None):
                    if self._backend is backend and self._attachment_serial == serial:
                        self._selection.camera_finished()

                self._observer_id = backend.interactor.AddObserver(
                    "EndInteractionEvent", camera_finished, 1.0,
                )
        except Exception:
            self.detach()
            raise

    def detach(self) -> None:
        backend = self._backend
        self._backend = None
        self._attachment_serial += 1
        self.reset()
        for target in self._event_targets:
            try:
                target.removeEventFilter(self)
            except RuntimeError:
                pass
        self._event_targets.clear()
        if backend is not None and self._observer_id is not None:
            try:
                backend.interactor.RemoveObserver(self._observer_id)
            except RuntimeError:
                pass
        self._observer_id = None
        if self._rubber_band is not None:
            self._rubber_band.detach()
            self._rubber_band = None

    def reset(self) -> bool:
        """Discard a gesture and pending work without rendering."""
        self._hover_timer.stop()
        self._band_render_timer.stop()
        self._pending_hover_position = None
        self._press_position = None
        had_drag = self._dragged
        self._dragged = False
        self._camera_button = None
        self._trackball_vector = None
        tool_changed = self._tools.reset()
        hover_changed = self._selection.clear_hover(False)
        band_changed = self.hide_rubber_band()
        return (tool_changed and had_drag) or hover_changed or band_changed

    def _reset_and_render(self) -> None:
        if self.reset() and self._backend is not None:
            self._selection.render()

    @staticmethod
    def _uses_camera_modifier(modifiers: Qt.KeyboardModifier) -> bool:
        return bool(modifiers & Qt.KeyboardModifier.ControlModifier) and bool(
            modifiers & Qt.KeyboardModifier.AltModifier
        )

    def _pointer_input(self, watched: QWidget, event: object) -> PointerInput:
        position = event.position()
        if watched is not self._backend.widget:
            position = watched.mapTo(self._backend.widget, position.toPoint())
        coordinates = (float(position.x()), float(position.y()))
        return PointerInput(
            coordinates, self._backend.to_vtk(*coordinates), event.button(),
            event.buttons(), event.modifiers(), event.globalPosition().toPoint(),
            self._press_position, self._dragged,
        )

    def eventFilter(self, watched: object, event: object) -> bool:
        if self._backend is None or watched not in self._event_targets:
            return False
        event_type = event.type()
        if event_type == QEvent.Type.KeyPress:
            key = event.key()
            if self._tools.active() and self._tools.key(key):
                if key in {Qt.Key.Key_Escape, Qt.Key.Key_Return, Qt.Key.Key_Enter}:
                    self._reset_and_render()
                return True
            if key in {Qt.Key.Key_Return, Qt.Key.Key_Enter}:
                self._reset_and_render()
                self._selection.confirm()
                return True
            if key == Qt.Key.Key_Escape:
                self._reset_and_render()
                self._selection.cancel()
                return True
        if event_type in {QEvent.Type.MouseButtonPress, QEvent.Type.MouseButtonDblClick}:
            pointer = self._pointer_input(watched, event)
            button = pointer.button
            if self._uses_camera_modifier(pointer.modifiers) and button in {
                Qt.MouseButton.LeftButton, Qt.MouseButton.MiddleButton, Qt.MouseButton.RightButton,
            }:
                self._reset_and_render()
                self._camera_button = button
                if button == Qt.MouseButton.LeftButton:
                    self._trackball_vector = trackball_point(
                        *pointer.position, self._backend.widget.width(), self._backend.widget.height(),
                    )
                    return True
                return False
            if self._tools.active():
                handled = self._tools.press(pointer)
                if button == Qt.MouseButton.LeftButton and handled:
                    self._hover_timer.stop()
                    self._pending_hover_position = None
                    self._press_position = pointer.position
                    self._dragged = False
                return handled
            if button == Qt.MouseButton.LeftButton:
                self._hover_timer.stop()
                self._pending_hover_position = None
                self._press_position = pointer.position
                self._dragged = False
                if self.hide_rubber_band():
                    self._selection.render()
                return True
            if button in {Qt.MouseButton.MiddleButton, Qt.MouseButton.RightButton}:
                return True
        if event_type == QEvent.Type.MouseMove:
            if self._camera_button == Qt.MouseButton.LeftButton:
                pointer = self._pointer_input(watched, event)
                self._rotate_trackball(*pointer.position)
                return True
            if self._camera_button is not None:
                return False
            pointer = self._pointer_input(watched, event)
            if self._tools.active():
                if self._press_position is not None and sum(
                    abs(value - start) for value, start in zip(pointer.position, self._press_position)
                ) > 2.0:
                    self._dragged = True
                    pointer = self._pointer_input(watched, event)
                self._tools.move(pointer)
                return True
            if self._press_position is not None:
                if sum(abs(value - start) for value, start in zip(pointer.position, self._press_position)) > 4.0:
                    self._dragged = True
                    self._selection.clear_hover(True)
                if self._dragged:
                    self.show_rubber_band(self._press_position, pointer.position)
                return True
            if pointer.buttons & (
                Qt.MouseButton.LeftButton | Qt.MouseButton.MiddleButton | Qt.MouseButton.RightButton
            ):
                return True
            self.queue_hover(pointer.position)
            return False
        if event_type == QEvent.Type.MouseButtonRelease:
            pointer = self._pointer_input(watched, event)
            button = pointer.button
            if self._camera_button == button:
                self._camera_button = None
                if button == Qt.MouseButton.LeftButton:
                    self._trackball_vector = None
                    self._selection.camera_finished()
                    return True
                return False
            if self._tools.active():
                if button == Qt.MouseButton.LeftButton and self._press_position is None:
                    return True
                self._press_position = None
                self._dragged = False
                self._tools.release(pointer)
                return True
            if button == Qt.MouseButton.LeftButton and self._press_position is not None:
                self._press_position = None
                self._dragged = False
                if self.hide_rubber_band():
                    self._selection.render()
                if pointer.dragged:
                    self._selection.box(pointer.press_position, pointer.position)
                else:
                    self._selection.click(*pointer.position)
                return True
            if button in {Qt.MouseButton.MiddleButton, Qt.MouseButton.RightButton}:
                return True
        if event_type == QEvent.Type.Leave:
            self._hover_timer.stop()
            self._pending_hover_position = None
            self._tools.leave()
            self._selection.clear_hover(True)
        return False

    def queue_hover(self, position: tuple[float, float]) -> None:
        if self._backend is None:
            return
        self._pending_hover_position = position
        self._hover_timer.start()

    def _deliver_hover(self) -> None:
        position = self._pending_hover_position
        self._pending_hover_position = None
        if self._backend is not None and position is not None:
            self._selection.hover(position)

    def show_rubber_band(self, start: tuple[float, float], end: tuple[float, float]) -> None:
        backend = self._backend
        if backend is None or backend.renderer is None:
            return
        if self._rubber_band is None:
            self._rubber_band = _SelectionRubberBand(backend.renderer)
        self._rubber_band.set_rectangle(backend.to_vtk(*start), backend.to_vtk(*end))
        if self._rubber_band.show():
            self._band_render_timer.stop()
            self._selection.render()
        elif not self._band_render_timer.isActive():
            self._band_render_timer.start()

    def hide_rubber_band(self) -> bool:
        self._band_render_timer.stop()
        return self._rubber_band is not None and self._rubber_band.hide()

    def _render_band(self) -> None:
        if self._backend is not None and self._rubber_band is not None and self._rubber_band._visible:
            self._selection.render()

    def _rotate_trackball(self, x: float, y: float) -> None:
        backend = self._backend
        if backend is None or self._trackball_vector is None or backend.camera is None:
            return
        current = trackball_point(x, y, backend.widget.width(), backend.widget.height())
        changed = rotate_camera_trackball(backend.camera, self._trackball_vector, current)
        self._trackball_vector = current
        if changed:
            backend.reset_clipping()
            self._selection.render()


class _SelectionRubberBand:
    """Draw a border-only rectangle in VTK display coordinates."""

    def __init__(self, renderer: Any) -> None:
        import vtk

        self._renderer = renderer
        self._visible = False
        self._points = vtk.vtkPoints()
        self._points.SetNumberOfPoints(4)
        lines = vtk.vtkCellArray()
        for start, end in ((0, 1), (1, 2), (2, 3), (3, 0)):
            lines.InsertNextCell(2)
            lines.InsertCellPoint(start)
            lines.InsertCellPoint(end)
        self._polydata = vtk.vtkPolyData()
        self._polydata.SetPoints(self._points)
        self._polydata.SetLines(lines)
        coordinate = vtk.vtkCoordinate()
        coordinate.SetCoordinateSystemToDisplay()
        mapper = vtk.vtkPolyDataMapper2D()
        mapper.SetInputData(self._polydata)
        mapper.SetTransformCoordinate(coordinate)
        self._actors = []
        for color, width in (
            ((1.0, 1.0, 1.0), 4.0),
            ((0.09, 0.55, 0.76), 2.0),
        ):
            actor = vtk.vtkActor2D()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(*color)
            actor.GetProperty().SetLineWidth(width)
            actor.GetProperty().SetLineStipplePattern(0xF0F0)
            actor.GetProperty().SetLineStippleRepeatFactor(1)
            actor.PickableOff()
            actor.SetVisibility(False)
            renderer.AddViewProp(actor)
            self._actors.append(actor)

    def set_rectangle(
        self,
        start: tuple[int, int],
        end: tuple[int, int],
    ) -> None:
        minimum_x, maximum_x = sorted((int(start[0]), int(end[0])))
        minimum_y, maximum_y = sorted((int(start[1]), int(end[1])))
        for index, point in enumerate(
            (
                (minimum_x, minimum_y, 0.0),
                (maximum_x, minimum_y, 0.0),
                (maximum_x, maximum_y, 0.0),
                (minimum_x, maximum_y, 0.0),
            )
        ):
            self._points.SetPoint(index, point)
        self._points.Modified()
        self._polydata.Modified()

    def show(self) -> bool:
        if self._visible:
            return False
        self._visible = True
        for actor in self._actors:
            actor.SetVisibility(True)
        return True

    def hide(self) -> bool:
        if not self._visible:
            return False
        self._visible = False
        for actor in self._actors:
            actor.SetVisibility(False)
        return True


    def detach(self) -> None:
        for actor in self._actors:
            self._renderer.RemoveViewProp(actor)
        self._actors.clear()
        self._renderer = None
        self._visible = False
