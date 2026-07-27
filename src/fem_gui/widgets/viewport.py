"""延迟加载 PyVistaQt 的分层有限元视口。"""

from __future__ import annotations

import inspect
import logging
import os
from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import numpy as np
from PySide6.QtCore import QEvent, QTimer, Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QLabel, QStackedLayout, QVBoxLayout, QWidget

from fem.application import RegionRef
from fem.application.results import FieldLocation, ResultCellKind, ResultValueLayout
from fem.boundary.step import boundary_for_step, get_step
from fem.geometry import LogicalEntityRef, logical_ref_sort_key
from ..geometry_preview import GeometryPreview
from ..viewport_background import ViewportBackgroundSettings
from ..visualization.model_adapter import ModelGeometry, pyvista_cell_array
from ..visualization.scene import DisplayState
from ..visualization.symbols import (
    SymbolSettings,
    arc_points,
    camera_facing_offset,
    constraint_rotation_axes,
    constraint_sample_indices,
    constraint_spatial_regions,
    constraint_symbol_dimensions,
    load_symbol_length,
    region_sample_indices,
    rotation_lock_points,
    sample_polyline,
    symbol_length,
)

if TYPE_CHECKING:
    from ..visualization.result_renderer import ResultRenderPayload

_pyvista = None
_QtInteractor = None
_backend_error: Exception | None = None
_backend_attempted = False

BEAM_FRAME_GLYPH_LIMIT = 64
BEAM_FRAME_CACHE_LIMIT = 256
_TYPED_RESULT_GRID_NAME = "typed_result_grid"


def _effective_line_load_vector(
    vector: object,
    coordinate_system: str,
    frame: Any | None,
) -> np.ndarray | None:
    """Project local components with the application-resolved Beam rotation."""

    values = np.asarray(vector, dtype=float)
    if coordinate_system != "local":
        return values
    if frame is None:
        return None
    return np.asarray(frame.rotation, dtype=float).T @ values


def _binding_error(interactor: type[object]) -> RuntimeError | None:
    """拒绝与主程序不同的 Qt 绑定。"""
    for base in interactor.mro():
        module = getattr(base, "__module__", "")
        if module.startswith("PySide6.QtWidgets"):
            return None
        if module.startswith(("PyQt5.QtWidgets", "PyQt6.QtWidgets", "PySide2.QtWidgets")):
            return RuntimeError(f"PyVistaQt 使用了不兼容的 Qt 绑定：{module}")
    return RuntimeError("无法确认 PyVistaQt 的 Qt 绑定来源")


def load_backend() -> tuple[object | None, type[object] | None, Exception | None]:
    """仅在需要显示网格时导入 VTK 相关包。"""
    global _pyvista, _QtInteractor, _backend_error, _backend_attempted
    if _backend_attempted:
        return _pyvista, _QtInteractor, _backend_error
    _backend_attempted = True
    try:
        import pyvista as pv
        from pyvistaqt import QtInteractor
        error = _binding_error(QtInteractor)
        if error is not None:
            raise error
    except Exception as error:
        _backend_error = error
        return None, None, error
    _pyvista, _QtInteractor, _backend_error = pv, QtInteractor, None
    return _pyvista, _QtInteractor, None


def is_offscreen_environment() -> bool:
    """判断是否应避免自动创建原生 OpenGL 控件。"""
    platform = os.environ.get("QT_QPA_PLATFORM", "").strip().lower()
    return platform in {"offscreen", "minimal", "minimalegl"} or os.environ.get(
        "FEM_GUI_OFFSCREEN", ""
    ).strip().lower() in {"1", "true", "yes", "on"}


def _geometry_edge_polydata(
    pyvista,
    points: np.ndarray,
    preview: GeometryPreview,
    pick_ids: tuple[int, ...],
):
    """Build line-only PolyData so logical edge ids match VTK cells exactly."""
    if len(pick_ids) != len(preview.edges):
        raise ValueError("edge pick tokens 必须与 display cells 数量一致")
    line_cells = np.hstack(
        [np.asarray((len(edge), *edge), dtype=np.int64) for edge in preview.edges]
    )
    edge_mesh = pyvista.PolyData()
    edge_mesh.points = points
    edge_mesh.lines = line_cells
    edge_mesh.cell_data["geometry_pick_id"] = np.asarray(
        pick_ids,
        dtype=np.int64,
    )
    edge_mesh.set_active_scalars(None)
    return edge_mesh


def _geometry_point_polydata(
    pyvista,
    points: np.ndarray,
    preview: GeometryPreview,
    pick_ids: tuple[int, ...],
):
    """Build pickable points from logical vertices, excluding display samples."""
    if len(pick_ids) != len(points):
        raise ValueError("point pick tokens 必须与 display points 数量一致")
    point_ids = np.asarray(pick_ids, dtype=np.int64)
    selectable = point_ids > 0
    point_mesh = pyvista.PolyData(points[selectable])
    point_mesh.point_data["geometry_pick_id"] = point_ids[selectable]
    return point_mesh


def _geometry_surface_polydata(
    pyvista,
    points: np.ndarray,
    preview: GeometryPreview,
    pick_ids: tuple[int, ...],
):
    """Triangulate display faces while preserving their logical geometry ids."""
    if len(pick_ids) != len(preview.faces):
        raise ValueError("face pick tokens 必须与 display cells 数量一致")
    face_cells = np.hstack(
        [np.asarray((len(face), *face), dtype=np.int64) for face in preview.faces]
    )
    surface = pyvista.PolyData(points, faces=face_cells)
    surface.cell_data["geometry_pick_id"] = np.asarray(
        pick_ids,
        dtype=np.int64,
    )
    surface = surface.triangulate()
    surface.set_active_scalars(None)
    return surface


@dataclass(frozen=True, slots=True)
class PickHit:
    """One resolved selectable object shared by hover and click."""

    kind: str
    pick_id: int
    dataset_name: str
    display_position: tuple[float, float]
    world_position: tuple[float, float, float]
    vtk_point_id: int | None = None
    vtk_cell_id: int | None = None


def _require_result_render_payload(
    payload: object,
) -> ResultRenderPayload:
    """Validate the exact typed viewport boundary without eagerly loading VTK."""

    from ..visualization.result_renderer import (
        validate_result_render_payload,
    )

    return validate_result_render_payload(payload)


def _point_to_segment_distance(
    point: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
) -> tuple[float, float]:
    """Return 2-D distance and segment fraction for one display-space segment."""
    vector = end - start
    length_squared = float(np.dot(vector, vector))
    if length_squared <= 0.0:
        return float(np.linalg.norm(point - start)), 0.0
    fraction = float(
        np.clip(np.dot(point - start, vector) / length_squared, 0.0, 1.0)
    )
    closest = start + fraction * vector
    return float(np.linalg.norm(point - closest)), fraction


class FEMViewport(QWidget):
    """维护网格、标注、选择、载荷与结果等独立 Actor。"""

    entityPicked = Signal(str, int)
    geometryEntityPicked = Signal(object)
    selectionMissed = Signal(str)
    selectionConfirmed = Signal()
    selectionCancelled = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._background_settings = ViewportBackgroundSettings()
        self._plotter = None
        self._grid = None
        self._result_grid = None
        self._result_point_index_to_node_id: dict[int, int] = {}
        self._result_point_index_to_element_id: dict[int, int] = {}
        self._result_cell_index_to_element_id: dict[int, int] = {}
        self._model = None
        self._geometry: ModelGeometry | None = None
        self._geometry_preview: GeometryPreview | None = None
        self._geometry_preview_surface = None
        self._geometry_preview_edges = None
        self._geometry_preview_points = None
        self._geometry_pick_to_ref: dict[int, LogicalEntityRef] = {}
        self._geometry_ref_to_pick_ids: dict[
            LogicalEntityRef,
            tuple[int, ...],
        ] = {}
        self._geometry_face_pick_ids: tuple[int, ...] = ()
        self._geometry_edge_pick_ids: tuple[int, ...] = ()
        self._geometry_point_pick_ids: tuple[int, ...] = ()
        self._geometry_body_pick_id = 0
        self._pick_grid = None
        self._pick_locators: dict[int, tuple[int, Any]] = {}
        self._result_render_payload: ResultRenderPayload | None = None
        # Runtime revalidation follows VTK Modified notifications. Full
        # representation validation still runs unconditionally on install/render.
        self._result_render_validated_mtime: int | None = None
        self._artifact_id: str | None = None
        self._run_id: str | None = None
        self._actors: dict[str, Any] = {}
        self._selection_mode = "node"
        self._selected_kind: str | None = None
        self._selected_id: int | None = None
        self._selection_highlight_visible = True
        self._show_edges = True
        self._show_nodes = False
        self._show_node_labels = False
        self._show_element_labels = False
        self._display = DisplayState()
        self._overlay_undeformed = False
        self._symbol_settings = SymbolSettings()
        self._symbols_visible = True
        self._boundary_cache: dict[str | None, Any] = {}
        self._effective_frame_query: (
            Callable[[RegionRef | int], Any] | None
        ) = None
        self._beam_frame_cache: dict[RegionRef | int, Any] = {}
        self._beam_frame_preview_target: RegionRef | int | None = None
        self._last_symbol_scale: float | None = None
        self._last_symbol_camera_position: np.ndarray | None = None
        self._updating_symbol_scale = False
        self._selection_press_position: tuple[float, float] | None = None
        self._selection_dragged = False
        self._picker_event_targets: set[QWidget] = set()
        self._abaqus_view_button: Qt.MouseButton | None = None
        self._trackball_vector: np.ndarray | None = None
        self._hover_hit: PickHit | None = None
        self._pending_hover_position: tuple[float, float] | None = None
        self._hover_timer = QTimer(self)
        self._hover_timer.setSingleShot(True)
        self._hover_timer.setInterval(40)
        self._hover_timer.timeout.connect(self._update_preselection)
        self._contour = {
            "manual": False, "minimum": 0.0, "maximum": 1.0, "levels": 12,
            "colormap": "jet", "style": "continuous", "legend": True, "edges": False,
            "number_format": "general", "decimals": 5,
            "orientation": "horizontal", "show_minimum": False,
            "show_maximum": False, "show_ids": False,
        }
        self._message = QLabel("", self)
        self._message.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._stack = QStackedLayout()
        self._stack.addWidget(self._message)
        host = QWidget(self)
        host.setLayout(self._stack)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(host)
        self._update_background_stylesheet()

    @property
    def backend_available(self) -> bool:
        pv, interactor, _error = load_backend()
        return pv is not None and interactor is not None

    @property
    def can_capture(self) -> bool:
        return self._plotter is not None

    @property
    def artifact_id(self) -> str | None:
        """Return the Session artifact represented by the model cache."""
        return self._artifact_id

    @property
    def run_id(self) -> str | None:
        """Return the Session run represented by the result cache."""
        return self._run_id

    @staticmethod
    def _uses_abaqus_view_modifier(modifiers: Qt.KeyboardModifier) -> bool:
        """Return whether Ctrl+Alt activates temporary view manipulation."""
        return bool(modifiers & Qt.KeyboardModifier.ControlModifier) and bool(
            modifiers & Qt.KeyboardModifier.AltModifier
        )

    def eventFilter(self, watched: object, event: object) -> bool:
        """Separate Abaqus-style camera drags from ordinary selection clicks."""
        if self._plotter is None:
            return False
        if self._picker_event_targets:
            is_picker_target = watched in self._picker_event_targets
        else:
            is_picker_target = watched is self._plotter
        if not is_picker_target:
            return super().eventFilter(watched, event)

        event_type = event.type()
        if event_type == QEvent.Type.KeyPress:
            if event.key() in {Qt.Key.Key_Return, Qt.Key.Key_Enter}:
                self.selectionConfirmed.emit()
                return True
            if event.key() == Qt.Key.Key_Escape:
                self._clear_preselection(render=True)
                self.selectionCancelled.emit()
                return True
        press_events = {
            QEvent.Type.MouseButtonPress,
            QEvent.Type.MouseButtonDblClick,
        }
        if event_type in press_events:
            button = event.button()
            if self._uses_abaqus_view_modifier(event.modifiers()) and button in {
                Qt.MouseButton.LeftButton,
                Qt.MouseButton.MiddleButton,
                Qt.MouseButton.RightButton,
            }:
                self._clear_preselection(render=True)
                self._abaqus_view_button = button
                self._selection_press_position = None
                if button == Qt.MouseButton.LeftButton:
                    position = self._plotter_event_position(watched, event)
                    self._trackball_vector = self._trackball_point(
                        position.x(), position.y()
                    )
                    return True
                return False
            if button == Qt.MouseButton.LeftButton:
                position = self._plotter_event_position(watched, event)
                self._selection_press_position = (position.x(), position.y())
                self._selection_dragged = False
                return True
            if button in {Qt.MouseButton.MiddleButton, Qt.MouseButton.RightButton}:
                return True

        if event_type == QEvent.Type.MouseMove:
            if self._abaqus_view_button == Qt.MouseButton.LeftButton:
                position = self._plotter_event_position(watched, event)
                self._rotate_trackball(position.x(), position.y())
                return True
            if self._abaqus_view_button is not None:
                return False
            if self._selection_press_position is not None:
                position = self._plotter_event_position(watched, event)
                start_x, start_y = self._selection_press_position
                if abs(position.x() - start_x) + abs(position.y() - start_y) > 4.0:
                    self._selection_dragged = True
                    self._clear_preselection(render=True)
                return True
            if event.buttons() & (
                Qt.MouseButton.LeftButton
                | Qt.MouseButton.MiddleButton
                | Qt.MouseButton.RightButton
            ):
                return True
            position = self._plotter_event_position(watched, event)
            self._pending_hover_position = (position.x(), position.y())
            self._hover_timer.start()
            return False

        if event_type == QEvent.Type.MouseButtonRelease:
            button = event.button()
            if self._abaqus_view_button == button:
                self._abaqus_view_button = None
                if button == Qt.MouseButton.LeftButton:
                    self._trackball_vector = None
                    self._refresh_symbols_for_camera(render=True)
                    return True
                return False
            if button == Qt.MouseButton.LeftButton and self._selection_press_position is not None:
                position = self._plotter_event_position(watched, event)
                should_pick = not self._selection_dragged
                self._selection_press_position = None
                self._selection_dragged = False
                if should_pick:
                    self._pick_qt_position(position.x(), position.y())
                return True
            if button in {Qt.MouseButton.MiddleButton, Qt.MouseButton.RightButton}:
                return True

        if event_type == QEvent.Type.Leave:
            self._pending_hover_position = None
            self._hover_timer.stop()
            self._clear_preselection(render=True)

        return super().eventFilter(watched, event)

    def _plotter_event_position(self, watched: object, event: object):
        """Return a mouse position in the outer QtInteractor coordinate system."""
        position = event.position()
        if watched is self._plotter or not isinstance(watched, QWidget):
            return position
        mapped = watched.mapTo(self._plotter, position.toPoint())
        return mapped

    def _trackball_point(self, x: float, y: float) -> np.ndarray:
        """Project a viewport position onto Abaqus' virtual rotation sphere."""
        if self._plotter is None:
            return np.array((0.0, 0.0, 1.0), dtype=float)
        width = max(float(self._plotter.width()), 1.0)
        height = max(float(self._plotter.height()), 1.0)
        radius = max(0.42 * min(width, height), 1.0)
        point_x = (float(x) - width * 0.5) / radius
        point_y = (height * 0.5 - float(y)) / radius
        distance_squared = point_x * point_x + point_y * point_y
        if distance_squared <= 1.0:
            point_z = float(np.sqrt(max(0.0, 1.0 - distance_squared)))
            return np.array((point_x, point_y, point_z), dtype=float)
        inverse_length = 1.0 / float(np.sqrt(distance_squared))
        return np.array(
            (point_x * inverse_length, point_y * inverse_length, 0.0),
            dtype=float,
        )

    @staticmethod
    def _rotate_vector(
        vector: np.ndarray,
        axis: np.ndarray,
        angle: float,
    ) -> np.ndarray:
        """Rotate a vector using Rodrigues' formula."""
        cosine = float(np.cos(angle))
        sine = float(np.sin(angle))
        return (
            vector * cosine
            + np.cross(axis, vector) * sine
            + axis * float(np.dot(axis, vector)) * (1.0 - cosine)
        )

    def _rotate_trackball(self, x: float, y: float) -> None:
        """Rotate the camera freely around its focal point using an arcball."""
        if self._plotter is None or self._trackball_vector is None:
            return
        current = self._trackball_point(x, y)
        previous = self._trackball_vector
        axis_camera = np.cross(current, previous)
        axis_length = float(np.linalg.norm(axis_camera))
        dot = float(np.clip(np.dot(previous, current), -1.0, 1.0))
        if axis_length <= 1.0e-12:
            self._trackball_vector = current
            return
        axis_camera /= axis_length
        angle = float(np.arctan2(axis_length, dot))

        camera = self._plotter.camera
        position = np.asarray(camera.GetPosition(), dtype=float)
        focal = np.asarray(camera.GetFocalPoint(), dtype=float)
        view_up = np.asarray(camera.GetViewUp(), dtype=float)
        direction = focal - position
        direction_length = float(np.linalg.norm(direction))
        up_length = float(np.linalg.norm(view_up))
        if direction_length <= 1.0e-12 or up_length <= 1.0e-12:
            self._trackball_vector = current
            return
        direction /= direction_length
        view_up /= up_length
        view_right = np.cross(direction, view_up)
        right_length = float(np.linalg.norm(view_right))
        if right_length <= 1.0e-12:
            self._trackball_vector = current
            return
        view_right /= right_length
        view_up = np.cross(view_right, direction)

        axis_world = (
            axis_camera[0] * view_right
            + axis_camera[1] * view_up
            - axis_camera[2] * direction
        )
        axis_world_length = float(np.linalg.norm(axis_world))
        if axis_world_length <= 1.0e-12:
            self._trackball_vector = current
            return
        axis_world /= axis_world_length

        offset = position - focal
        rotated_offset = self._rotate_vector(offset, axis_world, angle)
        rotated_up = self._rotate_vector(view_up, axis_world, angle)
        camera.SetPosition(*(focal + rotated_offset))
        camera.SetViewUp(*rotated_up)
        camera.OrthogonalizeViewUp()
        self._plotter.reset_camera_clipping_range()
        self._render()
        self._trackball_vector = current

    def set_model(
        self,
        model: Any,
        geometry: ModelGeometry,
        *,
        refresh_symbols: bool = True,
        render: bool = True,
        effective_frame_query: (
            Callable[[RegionRef | int], Any] | None
        ) = None,
    ) -> None:
        self._model = model
        self._geometry = geometry
        self._artifact_id = geometry.artifact_id
        self._run_id = None
        self._geometry_preview = None
        self._geometry_preview_surface = None
        self._geometry_preview_edges = None
        self._geometry_preview_points = None
        self._geometry_pick_to_ref.clear()
        self._geometry_ref_to_pick_ids.clear()
        self._geometry_face_pick_ids = ()
        self._geometry_edge_pick_ids = ()
        self._geometry_point_pick_ids = ()
        self._geometry_body_pick_id = 0
        self._pick_grid = None
        self._pick_locators.clear()
        self._result_render_payload = None
        self._result_render_validated_mtime = None
        self._selected_kind = None
        self._selected_id = None
        self._selection_highlight_visible = True
        self._selection_press_position = None
        self._selection_dragged = False
        self._abaqus_view_button = None
        self._trackball_vector = None
        self._hover_hit = None
        self._pending_hover_position = None
        self._hover_timer.stop()
        self._display = DisplayState()
        self._overlay_undeformed = False
        self._boundary_cache.clear()
        self._effective_frame_query = effective_frame_query
        self._beam_frame_cache.clear()
        self._beam_frame_preview_target = None
        self._last_symbol_scale = None
        self._last_symbol_camera_position = None
        if is_offscreen_environment():
            self._message.setText("模型已加载（当前环境未启用三维渲染）")
            return
        if not self._ensure_plotter():
            return
        self._remove_all_layers(render=False)
        self._grid = self._make_grid(geometry.points)
        self._pick_grid = self._grid
        self._add_base_layers(reset_camera=True, render=False)
        if refresh_symbols:
            self.show_boundary_and_loads(render=render)
        elif render:
            self._render()

    def clear_model(self) -> None:
        self._model = None
        self._geometry = None
        self._artifact_id = None
        self._run_id = None
        self._geometry_preview = None
        self._geometry_preview_surface = None
        self._geometry_preview_edges = None
        self._geometry_preview_points = None
        self._geometry_pick_to_ref.clear()
        self._geometry_ref_to_pick_ids.clear()
        self._geometry_face_pick_ids = ()
        self._geometry_edge_pick_ids = ()
        self._geometry_point_pick_ids = ()
        self._geometry_body_pick_id = 0
        self._pick_grid = None
        self._pick_locators.clear()
        self._result_render_payload = None
        self._result_render_validated_mtime = None
        self._display = DisplayState()
        self._overlay_undeformed = False
        self._selected_kind = None
        self._selected_id = None
        self._selection_highlight_visible = True
        self._selection_press_position = None
        self._selection_dragged = False
        self._abaqus_view_button = None
        self._trackball_vector = None
        self._hover_hit = None
        self._pending_hover_position = None
        self._hover_timer.stop()
        self._boundary_cache.clear()
        self._effective_frame_query = None
        self._beam_frame_cache.clear()
        self._beam_frame_preview_target = None
        self._last_symbol_scale = None
        self._last_symbol_camera_position = None
        self._grid = None
        self._result_grid = None
        self._result_point_index_to_node_id.clear()
        self._result_point_index_to_element_id.clear()
        self._result_cell_index_to_element_id.clear()
        self._remove_all_layers(render=False)
        self._message.clear()
        self._stack.setCurrentWidget(self._message)

    def show_geometry_preview(self, preview: GeometryPreview) -> None:
        """Display CAD-like shaded geometry without creating FE elements."""
        self._geometry_preview = preview
        self._install_geometry_pick_bindings(preview)
        if is_offscreen_environment():
            self._message.setText("几何预览已更新（当前环境未启用三维渲染）")
            self._stack.setCurrentWidget(self._message)
            return
        if not self._ensure_plotter():
            return
        self._remove_all_layers(render=False)
        points = np.asarray(preview.points, dtype=float)
        surface = _geometry_surface_polydata(
            _pyvista,
            points,
            preview,
            self._geometry_face_pick_ids,
        )
        self._geometry_preview_surface = surface
        self._actors["geometry_surface"] = self._plotter.add_mesh(
            surface,
            color="#c8d3dc",
            smooth_shading=False,
            show_edges=False,
            show_scalar_bar=False,
            name="geometry_surface",
            reset_camera=False,
        )
        if preview.edges:
            edge_mesh = _geometry_edge_polydata(
                _pyvista,
                points,
                preview,
                self._geometry_edge_pick_ids,
            )
            self._geometry_preview_edges = edge_mesh
            self._actors["geometry_edges"] = self._plotter.add_mesh(
                edge_mesh,
                color="#334b5f",
                line_width=2,
                show_scalar_bar=False,
                name="geometry_edges",
                reset_camera=False,
            )
        point_mesh = _geometry_point_polydata(
            _pyvista,
            points,
            preview,
            self._geometry_point_pick_ids,
        )
        self._geometry_preview_points = point_mesh
        if point_mesh.n_points:
            self._actors["geometry_points"] = self._plotter.add_mesh(
                point_mesh,
                color="#406f8f",
                point_size=6,
                render_points_as_spheres=True,
                show_scalar_bar=False,
                name="geometry_points",
                reset_camera=False,
            )
            self._actors["geometry_points"].SetVisibility(
                self._selection_mode == "geometry_point"
            )
        self._pick_grid = None
        self._pick_locators.clear()
        self._clear_preselection(render=False)
        self._plotter.reset_camera()
        self._render()

    def _install_geometry_pick_bindings(
        self,
        preview: GeometryPreview,
    ) -> None:
        pick_to_ref: dict[int, LogicalEntityRef] = {}
        ref_to_pick_ids: dict[LogicalEntityRef, set[int]] = {}
        next_pick_id = 1

        def allocate(
            logical_ids: tuple[str | None, ...],
        ) -> tuple[int, ...]:
            nonlocal next_pick_id
            tokens: list[int] = []
            for logical_id in logical_ids:
                if logical_id is None:
                    tokens.append(0)
                    continue
                reference = LogicalEntityRef(logical_id)
                pick_id = next_pick_id
                next_pick_id += 1
                tokens.append(pick_id)
                pick_to_ref[pick_id] = reference
                ref_to_pick_ids.setdefault(reference, set()).add(pick_id)
            return tuple(tokens)

        self._geometry_point_pick_ids = allocate(
            preview.point_logical_ids
        )
        self._geometry_edge_pick_ids = allocate(preview.edge_logical_ids)
        self._geometry_face_pick_ids = allocate(preview.face_logical_ids)
        if preview.body_logical_id is None:
            self._geometry_body_pick_id = 0
        else:
            body_reference = LogicalEntityRef(preview.body_logical_id)
            self._geometry_body_pick_id = next_pick_id
            pick_to_ref[next_pick_id] = body_reference
            ref_to_pick_ids.setdefault(body_reference, set()).add(
                next_pick_id
            )
        self._geometry_pick_to_ref = pick_to_ref
        self._geometry_ref_to_pick_ids = {
            reference: tuple(sorted(pick_ids))
            for reference, pick_ids in ref_to_pick_ids.items()
        }

    def set_result_render_payload(
        self,
        payload: ResultRenderPayload,
    ) -> None:
        """Install one already-projected scalar topology for direct rendering."""

        checked = _require_result_render_payload(payload)
        source = checked.topology.source
        if (
            self._artifact_id is not None
            and self._artifact_id != source.artifact_id
        ):
            raise ValueError(
                "payload artifact provenance does not match the viewport model"
            )
        self._result_render_payload = checked
        self._result_render_validated_mtime = int(
            checked.dataset.GetMTime()
        )
        self._artifact_id = source.artifact_id
        self._run_id = source.run_id
        self._index_result_render_provenance(checked)

    def _index_result_render_provenance(
        self,
        payload: ResultRenderPayload,
    ) -> None:
        """Index locations from a payload validated by the current caller."""

        topology = payload.topology
        self._result_point_index_to_node_id = {
            index: int(location.node_id)
            for index, location in enumerate(topology.point_locations)
            if location is not None and location.node_id is not None
        }
        self._result_point_index_to_element_id = {
            index: int(location.element_id)
            for index, location in enumerate(topology.point_locations)
            if location is not None and location.element_id is not None
        }
        cell_ids: list[int] = []
        for cell, location in zip(
            topology.cells,
            topology.cell_locations,
            strict=True,
        ):
            if location is not None and location.element_id is not None:
                cell_ids.append(int(location.element_id))
                continue
            candidates = {
                int(point_location.element_id)
                for point_index in cell
                if (
                    (point_location := topology.point_locations[point_index])
                    is not None
                    and point_location.element_id is not None
                )
            }
            cell_ids.append(candidates.pop() if len(candidates) == 1 else 0)
        self._result_cell_index_to_element_id = {
            index: element_id
            for index, element_id in enumerate(cell_ids)
            if element_id > 0
        }

    def _rendered_result_payload(self) -> ResultRenderPayload | None:
        """Return the rendered payload, honoring VTK-reported modifications.

        Mutations that bypass VTK's ``Modified`` protocol are detected at the
        next unconditional install/render boundary, not on every hover read.
        """

        payload = self._result_render_payload
        if payload is None or self._result_grid is not payload.dataset:
            return None
        modified = int(payload.dataset.GetMTime())
        if self._result_render_validated_mtime != modified:
            payload = _require_result_render_payload(payload)
            self._result_render_validated_mtime = modified
        return payload

    @staticmethod
    def _provenance_ids(
        length: int,
        index_to_id: dict[int, int],
    ) -> np.ndarray:
        return np.fromiter(
            (index_to_id.get(index, 0) for index in range(length)),
            dtype=np.int64,
            count=length,
        )

    def _typed_result_point_ids(self, dataset: Any) -> np.ndarray:
        return self._provenance_ids(
            int(dataset.n_points),
            self._result_point_index_to_node_id,
        )

    def _typed_result_point_element_ids(self, dataset: Any) -> np.ndarray:
        return self._provenance_ids(
            int(dataset.n_points),
            self._result_point_index_to_element_id,
        )

    def _typed_result_cell_ids(self, dataset: Any) -> np.ndarray:
        return self._provenance_ids(
            int(dataset.n_cells),
            self._result_cell_index_to_element_id,
        )

    def _typed_result_node_points(
        self,
        node_ids: tuple[int, ...],
    ) -> np.ndarray | None:
        payload = self._rendered_result_payload()
        if payload is None:
            return None
        ids = self._typed_result_point_ids(payload.dataset)
        requested = frozenset(int(node_id) for node_id in node_ids)
        available = frozenset(int(node_id) for node_id in ids if node_id > 0)
        if not requested or not requested.issubset(available):
            return None
        indices = np.flatnonzero(
            np.isin(ids, np.asarray(tuple(requested), dtype=np.int64))
        )
        if len(indices) == 0:
            return None
        return np.asarray(payload.dataset.points)[indices]

    def _typed_result_element_cells(
        self,
        element_ids: tuple[int, ...],
    ) -> Any | None:
        payload = self._rendered_result_payload()
        if payload is None:
            return None
        ids = self._typed_result_cell_ids(payload.dataset)
        requested = frozenset(int(element_id) for element_id in element_ids)
        available = frozenset(
            int(element_id) for element_id in ids if element_id > 0
        )
        if not requested or not requested.issubset(available):
            return None
        indices = np.flatnonzero(
            np.isin(ids, np.asarray(tuple(requested), dtype=np.int64))
        )
        if len(indices) == 0:
            return None
        return payload.dataset.extract_cells(indices)

    def set_selection_mode(self, mode: str) -> None:
        previous = self._selection_mode
        if mode in {
            "geometry_point", "geometry_edge", "geometry_face", "geometry_body",
        }:
            self._selection_mode = mode
        else:
            self._selection_mode = "element" if mode == "element" else "node"
        if previous != self._selection_mode:
            self._clear_preselection(render=False)
        points_actor = self._actors.get("geometry_points")
        if points_actor is not None:
            points_actor.SetVisibility(self._selection_mode == "geometry_point")
        self._update_pickable_actors()
        self._render()

    def clear_selection(self) -> None:
        self._selected_kind = None
        self._selected_id = None
        self._selection_highlight_visible = True
        self._remove_actor("selection")
        self._remove_actor("geometry_selection")
        self._clear_beam_frame_preview(render=False)
        self._clear_preselection(render=False)
        self._render()

    def highlight_geometry(self, reference: LogicalEntityRef) -> None:
        """Highlight one stable logical preview reference."""
        self.highlight_geometry_entities((reference,))

    def highlight_geometry_entities(
        self,
        references: Iterable[LogicalEntityRef],
    ) -> None:
        """Map logical refs back to every matching preview display cell."""
        self._remove_actor("geometry_selection")
        self._clear_beam_frame_preview(render=False)
        raw_references = tuple(references)
        if any(
            type(reference) is not LogicalEntityRef
            for reference in raw_references
        ):
            raise TypeError(
                "geometry highlight 只接受 LogicalEntityRef"
            )
        canonical_refs = tuple(
            sorted(
                set(raw_references),
                key=logical_ref_sort_key,
            )
        )
        if (
            self._plotter is None
            or self._geometry_preview is None
            or not canonical_refs
        ):
            return
        kinds = {reference.kind for reference in canonical_refs}
        if len(kinds) != 1:
            raise ValueError("geometry highlight 只能包含同一种实体类型")
        kind = f"geometry_{canonical_refs[0].kind}"
        pick_ids = tuple(
            sorted(
                {
                    pick_id
                    for reference in canonical_refs
                    for pick_id in self._geometry_ref_to_pick_ids.get(
                        reference,
                        (),
                    )
                }
            )
        )
        if not pick_ids:
            return
        if kind == "geometry_point" and self._geometry_preview_points is not None:
            ids = np.asarray(
                self._geometry_preview_points.point_data["geometry_pick_id"],
                dtype=np.int64,
            )
            indices = tuple(
                int(index)
                for index in np.flatnonzero(np.isin(ids, pick_ids))
            )
            if not indices:
                return
            data = _pyvista.PolyData(
                np.asarray(
                    self._geometry_preview_points.points,
                    dtype=float,
                )[np.asarray(indices, dtype=np.int64)]
            )
            kwargs = {"point_size": 13, "render_points_as_spheres": True}
        elif kind == "geometry_edge" and self._geometry_preview_edges is not None:
            ids = np.asarray(
                self._geometry_preview_edges.cell_data["geometry_pick_id"],
                dtype=np.int64,
            )
            cells = np.flatnonzero(np.isin(ids, pick_ids))
            if not len(cells):
                return
            data = self._geometry_preview_edges.extract_cells(cells)
            kwargs = {"line_width": 5}
        elif kind == "geometry_face" and self._geometry_preview_surface is not None:
            ids = np.asarray(
                self._geometry_preview_surface.cell_data["geometry_pick_id"],
                dtype=np.int64,
            )
            cells = np.flatnonzero(np.isin(ids, pick_ids))
            if not len(cells):
                return
            data = self._geometry_preview_surface.extract_cells(cells)
            kwargs = {"opacity": 0.8}
        elif kind == "geometry_body" and self._geometry_preview_surface is not None:
            data = self._geometry_preview_surface
            kwargs = {"opacity": 0.45}
        else:
            return
        self._actors["geometry_selection"] = self._plotter.add_mesh(
            data,
            color="#f5a623",
            show_edges=False,
            show_scalar_bar=False,
            name="geometry_selection",
            reset_camera=False,
            **kwargs,
        )
        self._offset_highlight_actor(self._actors["geometry_selection"])
        self._update_pickable_actors()
        self._render()

    def highlight_node(self, node_id: int) -> None:
        if self._geometry is None or node_id not in self._geometry.node_id_to_point_index:
            return
        index = self._geometry.node_id_to_point_index[node_id]
        self._selected_kind = "node"
        self._selected_id = int(node_id)
        self._selection_highlight_visible = True
        self._remove_actor("selection")
        self._clear_beam_frame_preview(render=False)
        if self._plotter is not None and _pyvista is not None:
            result_points = self._typed_result_node_points((int(node_id),))
            points = (
                self._model_display_points()[[index]]
                if result_points is None
                else result_points
            )
            point = _pyvista.PolyData(points)
            self._actors["selection"] = self._plotter.add_mesh(
                point, color="#d69a3a", point_size=14, render_points_as_spheres=True,
                name="selection", reset_camera=False,
            )
            self._update_pickable_actors()
            self._render()

    def highlight_element(self, element_id: int) -> None:
        if self._geometry is None or element_id not in self._geometry.element_id_to_cell_index:
            return
        index = self._geometry.element_id_to_cell_index[element_id]
        self._selected_kind = "element"
        self._selected_id = int(element_id)
        self._selection_highlight_visible = True
        self._remove_actor("selection")
        if self._pick_grid is not None and self._plotter is not None:
            selected = self._typed_result_element_cells((int(element_id),))
            if selected is None:
                selected = self._pick_grid.extract_cells([index])
            self._actors["selection"] = self._plotter.add_mesh(
                selected, color="#d69a3a", style="wireframe", line_width=3,
                name="selection", reset_camera=False,
            )
            self._offset_highlight_actor(self._actors["selection"])
            self._update_pickable_actors()
        self.show_beam_frame_preview(int(element_id), render=False)
        self._render()

    def highlight_nodes(self, node_ids: tuple[int, ...]) -> None:
        if self._geometry is None or _pyvista is None or self._plotter is None:
            return
        indices = [self._geometry.node_id_to_point_index[node_id] for node_id in node_ids if node_id in self._geometry.node_id_to_point_index]
        self._remove_actor("set_highlight")
        self._clear_beam_frame_preview(render=False)
        if indices:
            result_points = self._typed_result_node_points(
                tuple(
                    int(node_id)
                    for node_id in node_ids
                    if node_id in self._geometry.node_id_to_point_index
                )
            )
            points = (
                self._model_display_points()[indices]
                if result_points is None
                else result_points
            )
            self._actors["set_highlight"] = self._plotter.add_mesh(
                _pyvista.PolyData(points), color="#4f8fa8",
                point_size=12, render_points_as_spheres=True, name="set_highlight",
                reset_camera=False,
            )
        self._render()

    def highlight_elements(self, element_ids: tuple[int, ...]) -> None:
        if self._geometry is None or self._pick_grid is None or self._plotter is None:
            return
        indices = [self._geometry.element_id_to_cell_index[element_id] for element_id in element_ids if element_id in self._geometry.element_id_to_cell_index]
        self._remove_actor("set_highlight")
        self._clear_beam_frame_preview(render=False)
        if indices:
            selected = self._typed_result_element_cells(
                tuple(
                    int(element_id)
                    for element_id in element_ids
                    if element_id in self._geometry.element_id_to_cell_index
                )
            )
            if selected is None:
                selected = self._pick_grid.extract_cells(indices)
            self._actors["set_highlight"] = self._plotter.add_mesh(
                selected, color="#4f8fa8", style="wireframe",
                line_width=3, name="set_highlight", reset_camera=False,
            )
        self._render()

    def highlight_region(self, members: tuple[Any, ...], kind: str) -> None:
        """Highlight named surface faces or 2D boundary edges as one actor."""
        if self._geometry is None or _pyvista is None or self._plotter is None:
            return
        connectivity: list[int] = []
        self._clear_beam_frame_preview(render=False)
        for member in members:
            indices = [
                self._geometry.node_id_to_point_index[int(node_id)]
                for node_id in member.node_ids
                if int(node_id) in self._geometry.node_id_to_point_index
            ]
            if indices:
                connectivity.extend((len(indices), *indices))
        self._remove_actor("set_highlight")
        if not connectivity:
            self._render()
            return
        region = _pyvista.PolyData(self._model_display_points())
        if kind == "surface":
            region.faces = np.asarray(connectivity, dtype=np.int64)
            kwargs = {"opacity": 0.42, "show_edges": True, "line_width": 4}
        else:
            region.lines = np.asarray(connectivity, dtype=np.int64)
            kwargs = {"line_width": 6}
        self._actors["set_highlight"] = self._plotter.add_mesh(
            region, color="#4f8fa8", name="set_highlight", reset_camera=False,
            **kwargs,
        )
        self._render()

    def _effective_beam_frame_report(
        self,
        target: RegionRef | int,
    ) -> Any | None:
        cached = self._beam_frame_cache.get(target)
        if cached is not None:
            return cached
        if not callable(self._effective_frame_query):
            return None
        report = self._effective_frame_query(target)
        if len(self._beam_frame_cache) >= BEAM_FRAME_CACHE_LIMIT:
            oldest = next(iter(self._beam_frame_cache))
            self._beam_frame_cache.pop(oldest, None)
        self._beam_frame_cache[target] = report
        return report

    def _clear_beam_frame_preview(
        self,
        *,
        render: bool,
    ) -> None:
        self._beam_frame_preview_target = None
        for name in tuple(self._actors):
            if name.startswith("beam_frame_"):
                self._remove_actor(name)
        if render:
            self._render()

    def show_beam_frame_preview(
        self,
        target: RegionRef | int,
        *,
        render: bool = True,
    ) -> None:
        """Draw bounded assignment-aware Beam axes for one selected target."""

        self._clear_beam_frame_preview(render=False)
        self._beam_frame_preview_target = target
        if (
            self._plotter is None
            or self._geometry is None
            or self._model is None
        ):
            if render:
                self._render()
            return
        report = self._effective_beam_frame_report(target)
        entries = tuple(getattr(report, "entries", ()))
        if not entries:
            if render:
                self._render()
            return
        entries = entries[:BEAM_FRAME_GLYPH_LIMIT]
        element_lookup = {
            int(element.id): element
            for element in self._model.mesh.elements
        }
        points = self._current_points()
        frame_rows: list[tuple[np.ndarray, Any]] = []
        for entry in entries:
            element = element_lookup.get(int(entry.element_id))
            if element is None:
                continue
            indices = [
                self._geometry.node_id_to_point_index[int(node_id)]
                for node_id in element.node_ids
                if int(node_id) in self._geometry.node_id_to_point_index
            ]
            if not indices:
                continue
            frame_rows.append(
                (
                    np.mean(points[indices], axis=0),
                    entry.frame,
                )
            )
        if not frame_rows:
            if render:
                self._render()
            return
        glyph_scale = 0.75 * symbol_length(
            self._geometry.points,
            self._symbol_settings.scale,
            world_per_pixel=self._world_per_pixel(),
        )
        axes = (
            ("x", "#dc4b4b", "local_x"),
            ("y", "#45a565", "local_y"),
            ("z", "#477fd1", "local_z"),
        )
        label_points: list[np.ndarray] = []
        labels: list[str] = []
        for source in ("explicit", "automatic"):
            selected = [
                (origin, frame)
                for origin, frame in frame_rows
                if str(frame.source) == source
            ]
            if not selected:
                continue
            origins = np.asarray(
                [origin for origin, _frame in selected],
                dtype=float,
            )
            for axis_name, color, attribute in axes:
                vectors = np.asarray(
                    [
                        getattr(frame, attribute)
                        for _origin, frame in selected
                    ],
                    dtype=float,
                )
                actor_name = f"beam_frame_{axis_name}_{source}"
                self._actors[actor_name] = self._plotter.add_arrows(
                    origins,
                    vectors,
                    mag=glyph_scale,
                    color=color,
                    opacity=1.0 if source == "explicit" else 0.48,
                    name=actor_name,
                    reset_camera=False,
                )
                label_points.extend(
                    origins + glyph_scale * vectors
                )
                labels.extend(
                    (
                        axis_name
                        if source == "explicit"
                        else f"{axis_name} (automatic)"
                    )
                    for _ in selected
                )
        if label_points:
            self._actors["beam_frame_labels"] = (
                self._plotter.add_point_labels(
                    np.asarray(label_points, dtype=float),
                    labels,
                    point_size=0,
                    font_size=8,
                    shape_color=self._visual_palette()[
                        "label_background"
                    ],
                    name="beam_frame_labels",
                    reset_camera=False,
                )
            )
        self._update_pickable_actors()
        if render:
            self._render()

    def set_display(
        self,
        shape_mode: str,
        contour_enabled: bool,
    ) -> None:
        """独立设置已投影结果的几何形状和云图开关。"""
        shape = "deformed" if shape_mode == "deformed" else "undeformed"
        self._display = DisplayState(
            shape_mode=shape,
            contour_enabled=bool(contour_enabled),
        )
        self._update_result_layer()

    def set_contour_options(self, options: dict[str, Any]) -> None:
        self._contour.update(options)
        if self._display.contour_enabled:
            self._update_result_layer()

    def hide_selection_highlight(self, *, render: bool = True) -> None:
        """Hide the selection actor while preserving the selected FEM entity."""
        self._selection_highlight_visible = False
        self._remove_actor("selection")
        if render:
            self._render()

    def set_undeformed_overlay_visible(self, visible: bool) -> None:
        self._overlay_undeformed = bool(visible)
        self._refresh_undeformed_overlay()
        self._render()

    def set_symbol_settings(
        self, settings: SymbolSettings, *, refresh: bool = True, render: bool = True
    ) -> None:
        self._symbol_settings = settings
        if refresh:
            self.show_boundary_and_loads(settings.step_name, render=render)

    def set_symbols_visible(
        self, visible: bool, *, refresh: bool = True, render: bool = True
    ) -> None:
        """统一显示或隐藏当前分析步的约束与载荷符号。"""
        self._symbols_visible = bool(visible)
        symbol_names = self._symbol_actor_names()
        existing = [self._actors[name] for name in symbol_names if name in self._actors]
        if existing:
            for actor in existing:
                actor.SetVisibility(self._symbols_visible)
            if render:
                self._render()
        elif refresh and self._symbols_visible:
            self.show_boundary_and_loads(self._symbol_settings.step_name, render=render)

    def save_screenshot(self, path: str) -> None:
        """通过 VTK 帧缓冲保存当前视口。"""
        if self._plotter is None:
            raise RuntimeError("三维视口尚未初始化")
        self._plotter.screenshot(path)

    def set_background_settings(self, settings: ViewportBackgroundSettings) -> None:
        """更新视口背景和依赖背景对比度的显示层。"""
        self._background_settings = settings.normalized()
        self._update_background_stylesheet()
        if self._plotter is None:
            return
        self._apply_plotter_background()
        palette = self._visual_palette()
        for name, color in (
            ("mesh_surface", palette["mesh"]),
            ("element_edges", palette["edge"]),
            ("nodes", palette["node"]),
            ("undeformed_overlay", palette["overlay"]),
        ):
            self._set_actor_color(name, color)
        if not self._display.contour_enabled:
            self._set_actor_color("result", palette["result"])
        self._refresh_labels(render=False)
        if self._symbols_visible and self._model is not None:
            self.show_boundary_and_loads(self._symbol_settings.step_name, render=False)
        self._refresh_extrema_for_background()
        self._update_scalar_bar_text_color()
        self._render()

    def set_edges_visible(self, visible: bool, *, render: bool = True) -> None:
        self._show_edges = bool(visible)
        actor = self._actors.get("element_edges")
        if actor is not None:
            actor.SetVisibility(self._show_edges)
        if render:
            self._render()

    def set_nodes_visible(self, visible: bool) -> None:
        self._show_nodes = bool(visible)
        self._refresh_node_layer()

    def set_node_labels_visible(self, visible: bool) -> None:
        self._show_node_labels = bool(visible)
        self._refresh_labels()

    def set_element_labels_visible(self, visible: bool) -> None:
        self._show_element_labels = bool(visible)
        self._refresh_labels()

    def fit(self) -> None:
        if self._plotter is not None:
            self._plotter.reset_camera()
            self._refresh_symbols_for_camera(render=False)
            self._render()

    def render(self) -> None:
        """Render once after a caller completes a batch of viewport updates."""
        self._render()

    def invalidate_boundary_cache(self) -> None:
        """Drop GUI-only expanded boundary data after analysis definitions change."""
        self._boundary_cache.clear()

    def _world_per_pixel(self) -> float | None:
        if self._plotter is None:
            return None
        camera = getattr(self._plotter, "camera", None)
        if camera is None:
            return None
        window_size = getattr(self._plotter, "window_size", None)
        height = float(window_size[1]) if window_size and len(window_size) > 1 else float(self.height())
        if height <= 0.0:
            return None
        if bool(camera.GetParallelProjection()):
            scale = float(camera.GetParallelScale())
            return 2.0 * scale / height if scale > 0.0 else None
        position = np.asarray(camera.GetPosition(), dtype=float)
        focal_point = np.asarray(camera.GetFocalPoint(), dtype=float)
        distance = float(np.linalg.norm(position - focal_point))
        view_angle = float(camera.GetViewAngle())
        if distance <= 0.0 or view_angle <= 0.0:
            return None
        return 2.0 * distance * float(np.tan(np.deg2rad(0.5 * view_angle))) / height

    def _camera_position(self) -> np.ndarray | None:
        camera = getattr(self._plotter, "camera", None) if self._plotter is not None else None
        if camera is None:
            return None
        try:
            return np.asarray(camera.GetPosition(), dtype=float)
        except Exception:
            return None

    def _refresh_symbols_for_camera(self, *, render: bool) -> None:
        if (
            self._updating_symbol_scale
            or not self._symbols_visible
            or self._model is None
            or self._geometry is None
            or self._plotter is None
            or _pyvista is None
        ):
            return
        new_scale = symbol_length(
            self._geometry.points,
            self._symbol_settings.scale,
            world_per_pixel=self._world_per_pixel(),
        )
        previous = self._last_symbol_scale
        camera_position = self._camera_position()
        same_scale = previous is not None and abs(new_scale - previous) <= 0.02 * max(previous, 1.0e-12)
        same_camera = (
            camera_position is not None
            and self._last_symbol_camera_position is not None
            and np.allclose(camera_position, self._last_symbol_camera_position)
        )
        if same_scale and same_camera:
            return
        self._updating_symbol_scale = True
        try:
            self.show_boundary_and_loads(
                self._symbol_settings.step_name,
                render=False,
            )
            if self._beam_frame_preview_target is not None:
                self.show_beam_frame_preview(
                    self._beam_frame_preview_target,
                    render=False,
                )
            if render:
                self._render()
        finally:
            self._updating_symbol_scale = False

    def locate_nodes(self, node_ids: tuple[int, ...]) -> None:
        """仅调整相机以包含指定节点，不改变选择状态。"""
        if self._geometry is None:
            return
        indices = [
            self._geometry.node_id_to_point_index[node_id]
            for node_id in node_ids
            if node_id in self._geometry.node_id_to_point_index
        ]
        if indices:
            self._focus_points(self._current_points()[indices])

    def locate_elements(self, element_ids: tuple[int, ...]) -> None:
        """仅调整相机以包含指定单元，不改变选择状态。"""
        if self._geometry is None:
            return
        point_indices: set[int] = set()
        for element_id in element_ids:
            cell_index = self._geometry.element_id_to_cell_index.get(element_id)
            if cell_index is not None:
                point_indices.update(self._geometry.cells[cell_index])
        if point_indices:
            self._focus_points(self._current_points()[sorted(point_indices)])

    def _focus_points(self, points: np.ndarray) -> None:
        if self._plotter is None or len(points) == 0:
            return
        minimum = np.min(points, axis=0)
        maximum = np.max(points, axis=0)
        span = float(np.max(maximum - minimum))
        padding = max(span * 0.08, 1.0e-6)
        bounds = tuple(
            value
            for axis in range(3)
            for value in (minimum[axis] - padding, maximum[axis] + padding)
        )
        self._plotter.reset_camera(bounds=bounds)
        self._refresh_symbols_for_camera(render=False)
        self._render()

    def set_view(self, view: str) -> None:
        if self._plotter is None:
            return
        methods = {
            # Each tuple is (PyVista base view, camera-to-focal direction,
            # screen-up axis).  The directions deliberately follow the arrows
            # in the coordinate PNGs: positive X is left in XY/XZ, positive Y
            # is left in YZ, and the paired labels swap the screen axes.
            "front": ("view_xz", (0.0, -1.0, 0.0), (0.0, 0.0, 1.0)),  # XZ
            "back": ("view_xz", (0.0, 1.0, 0.0), (1.0, 0.0, 0.0)),   # ZX
            "left": ("view_yz", (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),   # YZ
            "right": ("view_yz", (-1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),  # ZY
            "top": ("view_xy", (0.0, 0.0, 1.0), (0.0, 1.0, 0.0)),    # XY
            "bottom": ("view_xy", (0.0, 0.0, -1.0), (1.0, 0.0, 0.0)), # YX
        }
        if view == "iso":
            self._plotter.view_isometric()
            up = (-1.0, 2.0, -1.0)
            camera = self._plotter.camera
        else:
            method_name, direction, up = methods[view]
            getattr(self._plotter, method_name)()
            camera = self._plotter.camera
            focal = np.asarray(camera.GetFocalPoint(), dtype=float)
            position = np.asarray(camera.GetPosition(), dtype=float)
            distance = float(np.linalg.norm(position - focal))
            if distance > 1.0e-12:
                direction_array = np.asarray(direction, dtype=float)
                direction_array /= np.linalg.norm(direction_array)
                camera.SetPosition(*(focal - direction_array * distance))
        camera.SetViewUp(*up)
        # VTK orthogonalizes the view-up vector against the current view
        # direction, avoiding a roll that would make the PNG legend disagree
        # with the actual screen axes.
        camera.OrthogonalizeViewUp()
        reset_clipping = getattr(self._plotter, "reset_camera_clipping_range", None)
        if reset_clipping is not None:
            reset_clipping()
        self._refresh_symbols_for_camera(render=False)
        self._render()

    def set_parallel_projection(self, enabled: bool) -> None:
        if self._plotter is not None:
            (self._plotter.enable_parallel_projection if enabled else self._plotter.disable_parallel_projection)()
            self._refresh_symbols_for_camera(render=False)
            self._render()

    @staticmethod
    def _symbol_actor_names() -> tuple[str, ...]:
        return (
            "constraints", "constraint_rotations",
            "constraint_labels", "loads", "load_moments", "load_moment_heads",
            "load_labels", "load_moment_labels", "load_regions", "load_region_edges",
        )

    def show_boundary_and_loads(
        self, step_name: str | None = None, *, render: bool = True
    ) -> None:
        for name in self._symbol_actor_names():
            self._remove_actor(name)
        if not self._symbols_visible:
            if render:
                self._render()
            return
        if self._model is None or self._geometry is None or self._plotter is None or _pyvista is None:
            return
        settings = self._symbol_settings
        selected_step = step_name if step_name is not None else settings.step_name
        try:
            if selected_step not in self._boundary_cache:
                self._boundary_cache[selected_step] = boundary_for_step(
                    self._model, selected_step
                )
            boundary = self._boundary_cache[selected_step]
        except Exception:
            return
        selected_definition = get_step(self._model, selected_step)
        glyph_scale = symbol_length(
            self._geometry.points,
            settings.scale,
            world_per_pixel=self._world_per_pixel(),
        )
        self._last_symbol_scale = glyph_scale
        self._last_symbol_camera_position = self._camera_position()
        constraint_scale, constraint_radius = constraint_symbol_dimensions(glyph_scale)
        load_scale = load_symbol_length(glyph_scale)
        is_3d = bool(self._model.mesh.nodes and hasattr(self._model.mesh.nodes[0], "z"))
        translation_count = 3 if is_3d else 2
        constraint_points: list[np.ndarray] = []
        constraint_vectors: list[np.ndarray] = []
        rotation_points: list[np.ndarray] = []
        rotation_axes: list[np.ndarray] = []
        constraint_label_points: dict[int, np.ndarray] = {}
        constraint_labels_by_node: dict[int, list[str]] = {}
        if settings.show_constraints:
            boundary_definitions = list(
                selected_definition.boundaries if selected_definition is not None else ()
            )
            initial_step = next(
                (
                    step for step in self._model.steps
                    if step.name.lower() == "initial"
                ),
                None,
            )
            if initial_step is not None and initial_step is not selected_definition:
                boundary_definitions = [*initial_step.boundaries, *boundary_definitions]
            constraints_by_target: dict[str | int, dict[int, float]] = {}
            for definition in boundary_definitions:
                components = constraints_by_target.setdefault(definition.target, {})
                for component in range(definition.first_component - 1, definition.last_component):
                    components[component] = float(definition.value)
            for target, components in constraints_by_target.items():
                if isinstance(target, int):
                    node_ids = (target,)
                elif target in self._model.node_sets:
                    node_ids = tuple(self._model.node_sets[target].node_ids)
                else:
                    continue
                target_points = np.asarray([
                    self._geometry.points[self._geometry.node_id_to_point_index[node_id]]
                    for node_id in node_ids
                ])
                regions = constraint_spatial_regions(
                    target_points, self._geometry.points
                )
                for region_indices in regions:
                    region_node_ids = tuple(node_ids[int(index)] for index in region_indices)
                    candidate_points = target_points[region_indices]
                    selected = constraint_sample_indices(
                        candidate_points, settings.sampling_density
                    )
                    camera_position = self._camera_position()
                    for selected_index in selected:
                        node_id = region_node_ids[int(selected_index)]
                        base = self._geometry.points[
                            self._geometry.node_id_to_point_index[node_id]
                        ]
                        display_base = base + camera_facing_offset(
                            base, camera_position, 0.28 * constraint_scale
                        )
                        for component, value in sorted(components.items()):
                            name = (
                                f"R{component - translation_count + 1}"
                                if translation_count == 3
                                and component >= translation_count
                                else "R3"
                                if component >= translation_count
                                else f"U{component + 1}"
                            )
                            constraint_label_points[node_id] = display_base
                            constraint_labels_by_node.setdefault(node_id, []).append(
                                f"{name}={value:.6g}" if value else name
                            )
                            if component >= translation_count:
                                continue
                            direction = np.zeros(3)
                            direction[component] = -1.0
                            constraint_points.append(display_base)
                            constraint_vectors.append(direction)
                        displayed_axes = constraint_rotation_axes(
                            tuple(sorted(components)),
                            is_3d=is_3d,
                            point=base,
                            camera_position=camera_position,
                        )
                        rotation_points.extend(display_base for _axis in displayed_axes)
                        rotation_axes.extend(displayed_axes)
        if constraint_points:
            cloud = _pyvista.PolyData(np.asarray(constraint_points))
            cloud["directions"] = np.asarray(constraint_vectors)
            wedge = _pyvista.Cone(
                center=(-0.5 * constraint_scale, 0.0, 0.0),
                direction=(1.0, 0.0, 0.0),
                height=constraint_scale,
                radius=constraint_radius,
                resolution=16,
            )
            glyphs = cloud.glyph(orient="directions", scale=False, geom=wedge)
            self._actors["constraints"] = self._plotter.add_mesh(
                glyphs, color=settings.constraint_color, name="constraints", reset_camera=False,
            )
        if rotation_points:
            self._add_rotation_constraint_symbols(
                rotation_points, rotation_axes, 0.36 * constraint_scale,
                settings.constraint_color, "constraint_rotations",
            )
        if settings.show_values and constraint_labels_by_node:
            node_ids = sorted(constraint_labels_by_node)
            label_points = [constraint_label_points[node_id] for node_id in node_ids]
            labels = ["，".join(constraint_labels_by_node[node_id]) for node_id in node_ids]
            self._actors["constraint_labels"] = self._plotter.add_point_labels(
                np.asarray(label_points), labels, point_size=0, font_size=9,
                shape_color=self._visual_palette()["label_background"],
                text_color=settings.constraint_color,
                name="constraint_labels", reset_camera=False,
            )
        origins: list[np.ndarray] = []
        vectors: list[np.ndarray] = []
        load_labels: list[str] = []
        moment_points: list[np.ndarray] = []
        moment_labels: list[str] = []
        if settings.show_nodal_loads:
            nodal_vectors: dict[int, np.ndarray] = {}
            nodal_moments: dict[int, np.ndarray] = {}
            for dof, value in boundary.nodal_forces.items():
                node_index, component = divmod(int(dof), self._model.mesh.dofs_per_node)
                node_id = self._model.mesh.dof_map.node_ids[node_index]
                if component < translation_count:
                    nodal_vectors.setdefault(node_id, np.zeros(3))[component] += float(value)
                else:
                    axis = component - translation_count if is_3d else 2
                    nodal_moments.setdefault(node_id, np.zeros(3))[axis] += float(value)
            for node_id, vector in nodal_vectors.items():
                if float(np.linalg.norm(vector)) <= 0.0:
                    continue
                base = self._geometry.points[self._geometry.node_id_to_point_index[node_id]]
                origins.append(base)
                vectors.append(vector)
                load_labels.append(f"F={tuple(float(value) for value in vector[:translation_count])}")
            for node_id, moment in nodal_moments.items():
                if float(np.linalg.norm(moment)) <= 0.0:
                    continue
                moment_points.append(self._geometry.points[self._geometry.node_id_to_point_index[node_id]])
                moment_labels.append(
                    "M=" + str(tuple(float(value) for value in moment))
                )
        node_lookup = {int(node.id): self._geometry.points[self._geometry.node_id_to_point_index[int(node.id)]] for node in self._model.mesh.nodes}
        face_cells: list[int] = []
        edge_lines: list[int] = []
        def add_distributed_group(definition, members, tractions, kind: str) -> None:
            candidates: list[np.ndarray] = []
            candidate_vectors: list[np.ndarray] = []
            for member, traction in zip(members, tractions):
                ids = member.node_ids
                points = np.asarray([node_lookup[int(node_id)] for node_id in ids])
                candidates.append(np.mean(points, axis=0))
                vector = np.pad(
                    np.asarray(traction.vector, dtype=float),
                    (0, max(0, 3 - len(traction.vector))),
                )[:3]
                candidate_vectors.append(vector)
                indices = [self._geometry.node_id_to_point_index[int(node_id)] for node_id in ids]
                target = face_cells if kind == "face" else edge_lines
                target.extend((len(indices), *indices))
            if not candidates:
                return
            candidate_array = np.asarray(candidates)
            selected = region_sample_indices(candidate_array, settings.sampling_density)
            label_index = int(selected[np.argmin(np.linalg.norm(
                candidate_array[selected] - np.mean(candidate_array, axis=0), axis=1
            ))])
            for index in selected:
                vector = candidate_vectors[int(index)]
                magnitude = float(np.linalg.norm(vector))
                if magnitude <= 0.0:
                    continue
                origins.append(candidate_array[int(index)])
                vectors.append(vector)
                prefix = "P" if definition.load_type == "pressure" else "T"
                load_labels.append(f"{prefix}={magnitude:.6g}" if int(index) == label_index else "")

        if selected_definition is not None:
            surface_offset = 0
            if settings.show_surface_loads:
                for definition in selected_definition.surface_loads:
                    members = self._model.surfaces[definition.surface].faces
                    count = len(members)
                    add_distributed_group(
                        definition, members,
                        boundary.surface_tractions[surface_offset:surface_offset + count],
                        "face",
                    )
                    surface_offset += count
            edge_offset = 0
            if settings.show_edge_loads:
                for definition in selected_definition.edge_loads:
                    members = self._model.edges[definition.edge].edges
                    count = len(members)
                    add_distributed_group(
                        definition, members,
                        boundary.edge_tractions[edge_offset:edge_offset + count],
                        "edge",
                    )
                    edge_offset += count
            if settings.show_line_loads:
                element_lookup = {
                    int(element.id): element
                    for element in self._model.mesh.elements
                }
                frame_by_element: dict[int, Any] = {}
                for definition in selected_definition.line_loads:
                    if definition.coordinate_system != "local":
                        continue
                    frame_target = (
                        int(definition.target)
                        if isinstance(definition.target, int)
                        else RegionRef(
                            "element_set",
                            str(definition.target),
                        )
                    )
                    report = self._effective_beam_frame_report(
                        frame_target
                    )
                    for entry in tuple(
                        getattr(report, "entries", ())
                    ):
                        frame_by_element[int(entry.element_id)] = (
                            entry.frame
                        )
                for load in boundary.line_loads:
                    element = element_lookup.get(int(load.elem_id))
                    if element is None:
                        continue
                    points = np.asarray([
                        node_lookup[int(node_id)]
                        for node_id in element.node_ids
                    ])
                    samples = sample_polyline(points, settings.sampling_density)
                    vector = _effective_line_load_vector(
                        load.vector,
                        load.coordinate_system,
                        frame_by_element.get(int(load.elem_id)),
                    )
                    if vector is None:
                        continue
                    if float(np.linalg.norm(vector)) <= 0.0:
                        continue
                    for sample_index, sample in enumerate(samples):
                        origins.append(sample)
                        vectors.append(vector)
                        load_labels.append(
                            f"q={tuple(float(value) for value in load.vector)}"
                            if sample_index == len(samples) // 2
                            else ""
                        )
                    indices = [
                        self._geometry.node_id_to_point_index[int(node_id)]
                        for node_id in element.node_ids
                    ]
                    edge_lines.extend((len(indices), *indices))
        if face_cells:
            region = _pyvista.PolyData(
                self._current_points(), faces=np.asarray(face_cells, dtype=np.int64)
            )
            self._actors["load_regions"] = self._plotter.add_mesh(
                region, color=settings.load_color, opacity=0.11, show_edges=False,
                name="load_regions", reset_camera=False,
            )
        if edge_lines:
            region_edges = _pyvista.PolyData(self._current_points())
            region_edges.lines = np.asarray(edge_lines, dtype=np.int64)
            self._actors["load_region_edges"] = self._plotter.add_mesh(
                region_edges, color=settings.load_color, line_width=3,
                name="load_region_edges", reset_camera=False,
            )
        self._add_load_arrows(origins, vectors, load_labels, load_scale)
        if moment_points:
            moment_axes = [
                moment / np.linalg.norm(moment)
                for moment in nodal_moments.values()
                if float(np.linalg.norm(moment)) > 0.0
            ]
            self._add_arc_symbols(
                moment_points, moment_axes, 0.65 * glyph_scale,
                settings.load_color, "load_moments",
            )
            if settings.show_values:
                self._actors["load_moment_labels"] = self._plotter.add_point_labels(
                    np.asarray(moment_points), moment_labels, point_size=0, font_size=9,
                    shape_color=self._visual_palette()["label_background"],
                    text_color=settings.load_color,
                    name="load_moment_labels", reset_camera=False,
                )
        if render:
            self._render()

    def _add_arc_symbols(
        self,
        centers: list[np.ndarray],
        axes: list[np.ndarray],
        radius: float,
        color: str,
        name: str,
    ) -> None:
        points: list[np.ndarray] = []
        lines: list[int] = []
        heads: list[np.ndarray] = []
        tangents: list[np.ndarray] = []
        for center, axis in zip(centers, axes):
            arc = arc_points(center, axis, radius)
            if len(arc) < 2:
                continue
            offset = len(points)
            points.extend(arc)
            lines.extend((len(arc), *(offset + index for index in range(len(arc)))))
            axis_start = len(points)
            points.extend((center - 0.45 * radius * axis, center + 0.45 * radius * axis))
            lines.extend((2, axis_start, axis_start + 1))
            heads.append(arc[-1])
            tangents.append(arc[-1] - arc[-2])
        if not points:
            return
        poly = _pyvista.PolyData(np.asarray(points))
        poly.lines = np.asarray(lines, dtype=np.int64)
        self._actors[name] = self._plotter.add_mesh(
            poly, color=color, line_width=3, name=name, reset_camera=False,
        )
        head_name = f"{name[:-1]}_heads" if name.endswith("s") else f"{name}_heads"
        self._actors[head_name] = self._plotter.add_arrows(
            np.asarray(heads), np.asarray(tangents), mag=0.22 * radius,
            color=color, name=head_name, reset_camera=False,
        )

    def _add_rotation_constraint_symbols(
        self,
        centers: list[np.ndarray],
        axes: list[np.ndarray],
        radius: float,
        color: str,
        name: str,
    ) -> None:
        points: list[np.ndarray] = []
        lines: list[int] = []
        for center, axis in zip(centers, axes):
            ring, bars = rotation_lock_points(center, axis, radius)
            if len(ring) == 0:
                continue
            ring_start = len(points)
            points.extend(ring)
            lines.extend((len(ring), *(ring_start + index for index in range(len(ring)))))
            bars_start = len(points)
            points.extend(bars)
            lines.extend((2, bars_start, bars_start + 1))
            lines.extend((2, bars_start + 2, bars_start + 3))
        if not points:
            return
        poly = _pyvista.PolyData(np.asarray(points))
        poly.lines = np.asarray(lines, dtype=np.int64)
        self._actors[name] = self._plotter.add_mesh(
            poly, color=color, line_width=3, name=name, reset_camera=False,
        )

    def _add_load_arrows(
        self,
        tips: list[np.ndarray],
        vectors: list[np.ndarray],
        labels: list[str],
        glyph_scale: float,
    ) -> None:
        if not vectors:
            return
        settings = self._symbol_settings
        display_vectors = np.asarray(vectors, dtype=float)
        norms = np.linalg.norm(display_vectors, axis=1)
        nonzero = norms > 0.0
        directions = np.zeros_like(display_vectors)
        directions[nonzero] = display_vectors[nonzero] / norms[nonzero, None]
        if settings.normalize_arrows:
            display_vectors = directions
        else:
            maximum = float(np.max(norms))
            if maximum > 0.0:
                factors = 0.55 + 0.45 * np.sqrt(norms / maximum)
                display_vectors = directions * factors[:, None]
        arrow_lengths = np.linalg.norm(display_vectors, axis=1)
        tip_points = np.asarray(tips)
        displayed_lengths = arrow_lengths * glyph_scale
        display_origins = tip_points - directions * displayed_lengths[:, None]
        arrow_points = _pyvista.PolyData(display_origins)
        arrow_points["directions"] = directions
        arrow_points["arrow_scale"] = displayed_lengths
        arrow = _pyvista.Arrow(
            start=(0.0, 0.0, 0.0), direction=(1.0, 0.0, 0.0),
            tip_length=0.24, tip_radius=0.11, tip_resolution=16,
            shaft_radius=0.035, shaft_resolution=12,
        )
        arrow_glyphs = arrow_points.glyph(
            orient="directions", scale="arrow_scale", factor=1.0, geom=arrow,
        )
        self._actors["loads"] = self._plotter.add_mesh(
            arrow_glyphs, color=settings.load_color, name="loads", reset_camera=False,
        )
        if settings.show_values and tips:
            existing = self._actors.get("load_labels")
            if existing is not None:
                self._remove_actor("load_labels")
            labelled = [index for index, label in enumerate(labels) if label]
            if labelled:
                self._actors["load_labels"] = self._plotter.add_point_labels(
                    display_origins[labelled], [labels[index] for index in labelled],
                    point_size=0, font_size=9,
                    shape_color=self._visual_palette()["label_background"],
                    text_color=settings.load_color, name="load_labels",
                    reset_camera=False,
                )

    def closeEvent(self, event) -> None:
        if self._plotter is not None:
            try:
                self._plotter.close()
            except Exception:
                pass
        super().closeEvent(event)

    def _ensure_plotter(self) -> bool:
        if self._plotter is not None:
            self._stack.setCurrentWidget(self._plotter)
            return True
        pv, interactor, error = load_backend()
        if pv is None or interactor is None:
            self._message.setText(f"三维视口无法加载：{error}")
            return False
        kwargs: dict[str, object] = {}
        try:
            if "off_screen" in inspect.signature(interactor).parameters and is_offscreen_environment():
                kwargs["off_screen"] = True
        except (TypeError, ValueError):
            pass
        try:
            self._plotter = interactor(self, **kwargs)
            self._apply_plotter_background()
            self._stack.addWidget(self._plotter)
            self._stack.setCurrentWidget(self._plotter)
            self._install_picker()
        except Exception as error:
            self._plotter = None
            self._message.setText(f"三维视口初始化失败：{error}")
            self._stack.setCurrentWidget(self._message)
            return False
        return True

    def _make_grid(self, points: np.ndarray):
        grid = _pyvista.UnstructuredGrid(
            pyvista_cell_array(self._geometry), self._geometry.cell_types, np.asarray(points, dtype=float)
        )
        grid.point_data["node_id"] = np.asarray(
            [
                self._geometry.point_index_to_node_id[index]
                for index in range(len(self._geometry.points))
            ],
            dtype=np.int64,
        )
        grid.cell_data["element_id"] = np.asarray(
            [
                self._geometry.cell_index_to_element_id[index]
                for index in range(len(self._geometry.cells))
            ],
            dtype=np.int64,
        )
        return grid

    def _model_display_points(self) -> np.ndarray:
        if self._geometry is None:
            return np.empty((0, 3), dtype=float)
        return np.asarray(self._geometry.points, dtype=float)

    def _refresh_pick_grid(self, points: np.ndarray | None = None) -> None:
        if self._geometry is None or _pyvista is None:
            self._pick_grid = None
        else:
            values = self._model_display_points() if points is None else points
            self._pick_grid = self._make_grid(np.asarray(values, dtype=float))
        self._pick_locators.clear()
        self._clear_preselection(render=False)

    def _element_line_width(self) -> int:
        if (
            self._geometry is not None
            and len(self._geometry.cell_types) > 0
            and np.all(np.asarray(self._geometry.cell_types) == 3)
        ):
            return 3
        return 1

    def _add_base_layers(self, reset_camera: bool, *, render: bool = True) -> None:
        palette = self._visual_palette()
        self._actors["mesh_surface"] = self._plotter.add_mesh(
            self._grid, color=palette["mesh"], show_edges=False, name="mesh_surface",
            line_width=self._element_line_width(), reset_camera=False,
        )
        self._actors["element_edges"] = self._plotter.add_mesh(
            self._grid, color=palette["edge"], style="wireframe",
            line_width=self._element_line_width(),
            name="element_edges", reset_camera=False,
        )
        self._refresh_node_layer(render=False)
        self._refresh_labels(render=False)
        if reset_camera:
            self._plotter.reset_camera()
        if render:
            self._render()

    def _update_result_layer(self) -> None:
        self._remove_actor("result")
        self._remove_actor("extrema")
        self._remove_scalar_bars()
        self._result_grid = None
        if self._result_render_payload is not None:
            self._update_result_render_payload_layer(
                self._result_render_payload
            )
            return
        self._result_point_index_to_node_id.clear()
        self._result_point_index_to_element_id.clear()
        self._result_cell_index_to_element_id.clear()

    def _update_result_render_payload_layer(
        self,
        payload: ResultRenderPayload,
    ) -> None:
        """Render an already-selected and already-deformed typed payload."""

        checked = _require_result_render_payload(payload)
        self._result_render_validated_mtime = int(
            checked.dataset.GetMTime()
        )
        self._index_result_render_provenance(checked)
        if self._plotter is None:
            return
        dataset = checked.dataset
        self._result_grid = dataset
        self._pick_locators.clear()
        self._clear_preselection(render=False)
        self._remove_actor("set_highlight")
        self._remove_actor("selection")
        base = self._actors.get("mesh_surface")
        if base is not None:
            base.SetVisibility(False)
        kwargs: dict[str, Any] = {
            "name": "result",
            "reset_camera": False,
            "line_width": self._element_line_width(),
            "show_edges": False,
        }
        if self._display.contour_enabled:
            kwargs.update(
                scalars=checked.scalar_name,
                cmap=self._contour["colormap"],
                n_colors=(
                    256
                    if self._contour.get("style") == "continuous"
                    else int(self._contour["levels"])
                ),
                interpolate_before_map=self._contour.get("style")
                == "continuous",
                show_scalar_bar=self._contour["legend"],
                scalar_bar_args={
                    "title": checked.topology.selection.component,
                    "vertical": self._contour["orientation"] == "vertical",
                    "fmt": self._scalar_format(),
                    "color": self._background_settings.foreground_color,
                },
            )
            if self._contour["manual"]:
                kwargs["clim"] = (
                    self._contour["minimum"],
                    self._contour["maximum"],
                )
        else:
            kwargs["color"] = self._visual_palette()["result"]
        self._actors["result"] = self._plotter.add_mesh(dataset, **kwargs)
        if (
            self._display.contour_enabled
            and (
                self._contour["show_minimum"]
                or self._contour["show_maximum"]
            )
        ):
            self._add_result_render_payload_extrema_labels(checked)
        self._refresh_undeformed_overlay()
        self._restore_selection()
        self._render()

    def _add_result_render_payload_extrema_labels(
        self,
        payload: ResultRenderPayload,
    ) -> None:
        """Label extrema from payload scalar and typed location provenance."""

        checked = _require_result_render_payload(payload)
        topology = checked.topology
        if topology.value_layout is ResultValueLayout.POINT:
            values = np.asarray(
                checked.dataset.point_data[checked.scalar_name]
            )
            points = np.asarray(checked.dataset.points)
            locations = topology.point_locations
        else:
            values = np.asarray(
                checked.dataset.cell_data[checked.scalar_name]
            )
            points = np.asarray(checked.dataset.cell_centers().points)
            locations = topology.cell_locations
        finite = np.flatnonzero(np.isfinite(values))
        if len(finite) == 0:
            return
        minimum_index = int(finite[np.argmin(values[finite])])
        maximum_index = int(finite[np.argmax(values[finite])])
        entries: list[tuple[int, str]] = []
        for enabled, index, title in (
            (self._contour["show_minimum"], minimum_index, "最小值"),
            (self._contour["show_maximum"], maximum_index, "最大值"),
        ):
            if not enabled:
                continue
            label = f"{title} {self._format_scalar(float(values[index]))}"
            if self._contour["show_ids"]:
                identity = self._result_location_identity(locations[index])
                if identity:
                    label += f"（{identity}）"
            entries.append((index, label))
        if not entries:
            return
        palette = self._visual_palette()
        self._actors["extrema"] = self._plotter.add_point_labels(
            points[[entry[0] for entry in entries]],
            [entry[1] for entry in entries],
            point_size=7,
            font_size=10,
            shape_color=palette["label_background"],
            text_color=self._background_settings.foreground_color,
            name="extrema",
            reset_camera=False,
        )

    @staticmethod
    def _result_location_identity(
        location: FieldLocation | None,
    ) -> str:
        if location is None:
            return ""
        values = []
        if location.node_id is not None:
            values.append(f"节点 {int(location.node_id)}")
        if location.element_id is not None:
            values.append(f"单元 {int(location.element_id)}")
        if location.integration_point is not None:
            values.append(f"积分点 {int(location.integration_point)}")
        if location.local_node is not None:
            values.append(f"局部节点 {int(location.local_node)}")
        return "，".join(values)

    def _refresh_geometry_dependent_layers(self, *, render: bool = True) -> None:
        self._remove_actor("element_edges")
        self._remove_actor("set_highlight")
        self._actors["element_edges"] = self._plotter.add_mesh(
            self._grid, color=self._visual_palette()["edge"], style="wireframe",
            line_width=self._element_line_width(),
            name="element_edges", reset_camera=False,
        )
        self._actors["element_edges"].SetVisibility(self._show_edges)
        self._refresh_node_layer(render=False)
        self._refresh_labels(render=False)
        self._remove_actor("selection")
        self._restore_selection()
        if render:
            self._render()

    def _refresh_undeformed_overlay(self) -> None:
        self._remove_actor("undeformed_overlay")
        if (
            not self._overlay_undeformed
            or self._display.shape_mode == "undeformed"
            or self._plotter is None
            or self._geometry is None
        ):
            return
        undeformed = self._make_grid(self._geometry.points)
        self._actors["undeformed_overlay"] = self._plotter.add_mesh(
            undeformed,
            color=self._visual_palette()["overlay"],
            style="wireframe",
            line_width=self._element_line_width(),
            opacity=0.65,
            name="undeformed_overlay",
            reset_camera=False,
        )

    def _restore_selection(self) -> None:
        if not self._selection_highlight_visible:
            return
        if self._selected_kind == "node" and self._selected_id is not None:
            self.highlight_node(self._selected_id)
        elif self._selected_kind == "element" and self._selected_id is not None:
            self.highlight_element(self._selected_id)

    def _scalar_format(self) -> str:
        decimals = int(self._contour["decimals"])
        mode = self._contour["number_format"]
        if mode == "fixed":
            return f"%.{decimals}f"
        if mode == "scientific":
            return f"%.{decimals}e"
        return f"%.{decimals}g"

    def _format_scalar(self, value: float) -> str:
        decimals = int(self._contour["decimals"])
        mode = self._contour["number_format"]
        if mode == "fixed":
            return f"{value:.{decimals}f}"
        if mode == "scientific":
            return f"{value:.{decimals}e}"
        return f"{value:.{decimals}g}"

    def _refresh_node_layer(self, *, render: bool = True) -> None:
        self._remove_actor("nodes")
        if self._show_nodes and self._plotter is not None and _pyvista is not None and self._geometry is not None:
            node_data = _pyvista.PolyData(self._model_display_points())
            node_data.point_data["node_id"] = np.asarray(
                [
                    self._geometry.point_index_to_node_id[index]
                    for index in range(len(self._geometry.points))
                ],
                dtype=np.int64,
            )
            self._actors["nodes"] = self._plotter.add_mesh(
                node_data,
                color=self._visual_palette()["node"], point_size=7,
                render_points_as_spheres=True, name="nodes", reset_camera=False,
            )
            self._update_pickable_actors()
        if render:
            self._render()

    def _refresh_labels(self, *, render: bool = True) -> None:
        self._remove_actor("node_labels")
        self._remove_actor("element_labels")
        if self._plotter is None or self._geometry is None or _pyvista is None:
            return
        if self._show_node_labels:
            labels = [str(self._geometry.point_index_to_node_id[index]) for index in range(len(self._geometry.points))]
            self._actors["node_labels"] = self._plotter.add_point_labels(
                self._model_display_points(), labels, point_size=0, font_size=10,
                shape=None, text_color=self._background_settings.foreground_color,
                name="node_labels", reset_camera=False,
            )
        if self._show_element_labels:
            centers = self._pick_grid.cell_centers().points
            labels = [str(self._geometry.cell_index_to_element_id[index]) for index in range(len(self._geometry.cells))]
            self._actors["element_labels"] = self._plotter.add_point_labels(
                centers, labels, point_size=0, font_size=10, shape=None,
                text_color=self._background_settings.foreground_color,
                name="element_labels", reset_camera=False,
            )
        if render:
            self._render()

    def _current_points(self) -> np.ndarray:
        return self._geometry.points if self._grid is None else np.asarray(self._grid.points)

    def _install_picker(self) -> None:
        try:
            interactor = self._plotter.iren.interactor

            def camera_interaction_finished(_obj=None, _event=None):
                self._refresh_symbols_for_camera(render=True)

            interactor.AddObserver("EndInteractionEvent", camera_interaction_finished, 1.0)
            self._picker_event_targets = {self._plotter}
            self._picker_event_targets.update(self._plotter.findChildren(QWidget))
            for target in self._picker_event_targets:
                target.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
                target.setMouseTracking(True)
                target.installEventFilter(self)
            self._update_pickable_actors()
            self._plotter.setToolTip(
                "左键选择；Ctrl+Alt+左键旋转；Ctrl+Alt+中键平移；"
                "Ctrl+Alt+右键缩放；滚轮缩放"
            )
        except Exception:
            return

    def _pick_qt_position(self, x: float, y: float) -> None:
        """Submit the same resolved candidate used by hover highlighting."""
        if self._plotter is None:
            return
        vtk_x, vtk_y = self._qt_to_vtk_position(x, y)
        pixel_ratio = self._device_pixel_ratio()
        hover = self._hover_hit
        if (
            hover is not None
            and hover.kind == self._selection_mode
            and float(
                np.linalg.norm(
                    np.asarray(hover.display_position)
                    - np.asarray((vtk_x, vtk_y))
                )
            )
            <= 3.0 * pixel_ratio
        ):
            hit = hover
        else:
            hit = self._resolve_pick(vtk_x, vtk_y)
        self._submit_pick(hit)

    def _pick_at(self, x: int, y: int) -> PickHit | None:
        """Resolve and submit one VTK display-coordinate selection."""
        hit = self._resolve_pick(int(x), int(y))
        self._submit_pick(hit)
        return hit

    def _submit_pick(self, hit: PickHit | None) -> None:
        if hit is None:
            self.selectionMissed.emit(self._selection_mode)
            return
        logging.debug(
            "viewport pick mode=%s dataset=%s vtk_point_id=%s "
            "vtk_cell_id=%s pick_id=%s world=%s display=%s",
            hit.kind,
            hit.dataset_name,
            hit.vtk_point_id,
            hit.vtk_cell_id,
            hit.pick_id,
            hit.world_position,
            hit.display_position,
        )
        if hit.kind.startswith("geometry_"):
            reference = self._geometry_pick_to_ref.get(hit.pick_id)
            if reference is None:
                self.selectionMissed.emit(self._selection_mode)
                return
            if hit.kind != f"geometry_{reference.kind}":
                raise ValueError(
                    "geometry pick kind 与 logical reference 不一致"
                )
            self.geometryEntityPicked.emit(reference)
            return
        if hit.kind not in {"node", "element"}:
            raise ValueError(
                "entityPicked 只接受 FEM node 或 element"
            )
        if isinstance(hit.pick_id, bool) or not isinstance(
            hit.pick_id,
            int,
        ):
            raise TypeError("FEM entity pick id 必须是整数")
        self.entityPicked.emit(hit.kind, hit.pick_id)

    def _device_pixel_ratio(self) -> float:
        if self._plotter is None:
            return 1.0
        try:
            return max(float(self._plotter._getPixelRatio()), 1.0)
        except (AttributeError, TypeError, ValueError):
            return max(float(self._plotter.devicePixelRatioF()), 1.0)

    def _qt_to_vtk_position(self, x: float, y: float) -> tuple[int, int]:
        """Convert one Qt logical top-left position to VTK device pixels."""
        ratio = self._device_pixel_ratio()
        return (
            int(round(float(x) * ratio)),
            int(round((float(self._plotter.height()) - float(y) - 1.0) * ratio)),
        )

    def _update_preselection(self) -> None:
        if self._pending_hover_position is None or self._plotter is None:
            return
        x, y = self._pending_hover_position
        vtk_x, vtk_y = self._qt_to_vtk_position(x, y)
        hit = self._resolve_pick(vtk_x, vtk_y)
        if hit == self._hover_hit:
            return
        self._hover_hit = hit
        self._show_preselection(hit)

    def _resolve_pick(self, x: int, y: int) -> PickHit | None:
        mode = self._selection_mode
        if mode == "geometry_point":
            return self._pick_screen_point(
                x,
                y,
                self._geometry_preview_points,
                "geometry_pick_id",
                "geometry_points",
                self._geometry_preview_surface,
                8.0,
            )
        if mode == "geometry_edge":
            return self._pick_screen_edge(x, y, 6.0)
        if mode in {"geometry_face", "geometry_body"}:
            hit = self._pick_cell(
                x,
                y,
                self._geometry_preview_surface,
                "geometry_pick_id",
                "geometry_surface",
                mode,
            )
            if hit is not None and hit.pick_id <= 0:
                return None
            if hit is not None and mode == "geometry_body":
                body_pick_id = self._geometry_body_pick_id
                if body_pick_id <= 0:
                    return None
                return replace(hit, pick_id=body_pick_id)
            return hit
        if mode == "node":
            payload = self._rendered_result_payload()
            if payload is not None:
                ids = self._typed_result_point_ids(payload.dataset)
                if bool(np.any(ids > 0)):
                    hit = self._pick_screen_point(
                        x,
                        y,
                        payload.dataset,
                        "node_id",
                        _TYPED_RESULT_GRID_NAME,
                        payload.dataset,
                        8.0,
                        provenance_ids=ids,
                    )
                    if hit is not None:
                        return hit
            return self._pick_screen_point(
                x,
                y,
                self._pick_grid,
                "node_id",
                "model_pick_grid",
                self._pick_grid,
                8.0,
            )
        if mode == "element":
            payload = self._rendered_result_payload()
            if payload is not None:
                ids = self._typed_result_cell_ids(payload.dataset)
                if bool(np.any(ids > 0)):
                    if all(
                        kind is ResultCellKind.SAMPLE_VERTEX
                        for kind in payload.topology.cell_kinds
                    ):
                        point_ids = self._typed_result_point_element_ids(
                            payload.dataset
                        )
                        hit = self._pick_screen_point(
                            x,
                            y,
                            payload.dataset,
                            "element_id",
                            _TYPED_RESULT_GRID_NAME,
                            payload.dataset,
                            8.0,
                            provenance_ids=point_ids,
                        )
                        if hit is not None:
                            return hit
                    hit = self._pick_cell(
                        x,
                        y,
                        payload.dataset,
                        "element_id",
                        _TYPED_RESULT_GRID_NAME,
                        mode,
                        provenance_ids=ids,
                    )
                    if hit is not None:
                        return hit
            return self._pick_cell(
                x,
                y,
                self._pick_grid,
                "element_id",
                "model_pick_grid",
                mode,
            )
        return None

    def _pick_screen_point(
        self,
        x: int,
        y: int,
        dataset: Any,
        id_array: str,
        dataset_name: str,
        occluder: Any,
        tolerance: float,
        *,
        provenance_ids: np.ndarray | None = None,
    ) -> PickHit | None:
        if dataset is None:
            return None
        if provenance_ids is not None:
            ids = np.asarray(provenance_ids, dtype=np.int64)
            if ids.shape != (int(dataset.n_points),):
                raise ValueError(
                    "point provenance ids must match dataset points"
                )
        elif id_array in dataset.point_data:
            ids = np.asarray(dataset.point_data[id_array], dtype=np.int64)
        else:
            return None
        mouse = np.asarray((float(x), float(y)), dtype=float)
        threshold = float(tolerance) * self._device_pixel_ratio()
        world_points = np.asarray(dataset.points, dtype=float)
        display_points = self._world_points_to_display(world_points)
        if display_points is None:
            return None
        distances = np.linalg.norm(display_points[:, :2] - mouse, axis=1)
        candidates = np.flatnonzero(
            np.isfinite(distances)
            & (distances <= threshold)
            & (display_points[:, 2] >= 0.0)
            & (display_points[:, 2] <= 1.0)
            & (ids > 0)
        )
        if not len(candidates):
            return None
        order = candidates[
            np.lexsort(
                (
                    display_points[candidates, 2],
                    distances[candidates],
                )
            )
        ]
        index = next(
            (
                int(candidate)
                for candidate in order
                if self._display_candidate_is_visible(
                    display_points[candidate],
                    occluder,
                )
            ),
            None,
        )
        if index is None:
            return None
        world = world_points[index]
        return PickHit(
            self._selection_mode,
            int(ids[index]),
            dataset_name,
            (float(x), float(y)),
            tuple(float(value) for value in world),
            vtk_point_id=index,
        )

    def _pick_screen_edge(
        self,
        x: int,
        y: int,
        tolerance: float,
    ) -> PickHit | None:
        preview = self._geometry_preview
        dataset = self._geometry_preview_edges
        if (
            preview is None
            or dataset is None
            or "geometry_pick_id" not in dataset.cell_data
        ):
            return None
        mouse = np.asarray((float(x), float(y)), dtype=float)
        threshold = float(tolerance) * self._device_pixel_ratio()
        points = np.asarray(preview.points, dtype=float)
        ids = np.asarray(dataset.cell_data["geometry_pick_id"], dtype=np.int64)
        display_points = self._world_points_to_display(points)
        if display_points is None:
            return None
        starts: list[int] = []
        ends: list[int] = []
        cells: list[int] = []
        for cell_index, edge in enumerate(preview.edges):
            for start_index, end_index in zip(edge, edge[1:]):
                starts.append(int(start_index))
                ends.append(int(end_index))
                cells.append(int(cell_index))
        if not starts:
            return None
        start_ids = np.asarray(starts, dtype=np.int64)
        end_ids = np.asarray(ends, dtype=np.int64)
        start_display = display_points[start_ids]
        vectors = display_points[end_ids] - start_display
        vector_2d = vectors[:, :2]
        length_squared = np.einsum("ij,ij->i", vector_2d, vector_2d)
        fractions = np.divide(
            np.einsum("ij,ij->i", mouse - start_display[:, :2], vector_2d),
            length_squared,
            out=np.zeros_like(length_squared),
            where=length_squared > 0.0,
        )
        fractions = np.clip(fractions, 0.0, 1.0)
        closest = start_display + fractions[:, None] * vectors
        distances = np.linalg.norm(closest[:, :2] - mouse, axis=1)
        candidates = np.flatnonzero(
            np.isfinite(distances)
            & (distances <= threshold)
            & (closest[:, 2] >= 0.0)
            & (closest[:, 2] <= 1.0)
            & (ids[np.asarray(cells, dtype=np.int64)] > 0)
        )
        if not len(candidates):
            return None
        order = candidates[
            np.lexsort((closest[candidates, 2], distances[candidates]))
        ]
        segment_index = next(
            (
                int(candidate)
                for candidate in order
                if self._display_candidate_is_visible(
                    closest[candidate],
                    self._geometry_preview_surface,
                )
            ),
            None,
        )
        if segment_index is None:
            return None
        cell_index = cells[segment_index]
        fraction = float(fractions[segment_index])
        world = (
            points[start_ids[segment_index]]
            + fraction
            * (
                points[end_ids[segment_index]]
                - points[start_ids[segment_index]]
            )
        )
        return PickHit(
            self._selection_mode,
            int(ids[cell_index]),
            "geometry_edges",
            (float(x), float(y)),
            tuple(float(value) for value in world),
            vtk_cell_id=cell_index,
        )

    def _pick_cell(
        self,
        x: int,
        y: int,
        dataset: Any,
        id_array: str,
        dataset_name: str,
        kind: str,
        *,
        provenance_ids: np.ndarray | None = None,
    ) -> PickHit | None:
        if dataset is None:
            return None
        if provenance_ids is not None:
            ids = np.asarray(provenance_ids, dtype=np.int64)
            if ids.shape != (int(dataset.n_cells),):
                raise ValueError(
                    "cell provenance ids must match dataset cells"
                )
        elif id_array in dataset.cell_data:
            ids = np.asarray(dataset.cell_data[id_array], dtype=np.int64)
        else:
            return None
        intersection = self._intersect_dataset(x, y, dataset)
        if intersection is None:
            return None
        cell_id, world = intersection
        if not 0 <= cell_id < len(ids) or int(ids[cell_id]) <= 0:
            return None
        return PickHit(
            kind,
            int(ids[cell_id]),
            dataset_name,
            (float(x), float(y)),
            tuple(float(value) for value in world),
            vtk_cell_id=cell_id,
        )

    def _display_to_world(self, x: float, y: float, depth: float) -> np.ndarray | None:
        renderer = getattr(self._plotter, "renderer", None)
        if renderer is None:
            return None
        renderer.SetDisplayPoint(float(x), float(y), float(depth))
        renderer.DisplayToWorld()
        homogeneous = np.asarray(renderer.GetWorldPoint(), dtype=float)
        if len(homogeneous) != 4 or abs(float(homogeneous[3])) <= 1.0e-15:
            return None
        return homogeneous[:3] / homogeneous[3]

    def _world_to_display(self, world: np.ndarray) -> np.ndarray | None:
        values = self._world_points_to_display(
            np.asarray(world, dtype=float).reshape((1, 3))
        )
        return None if values is None else values[0]

    def _world_points_to_display(self, points: np.ndarray) -> np.ndarray | None:
        """Project a point array in one NumPy operation for responsive hover."""
        renderer = getattr(self._plotter, "renderer", None)
        if renderer is None:
            return None
        camera = renderer.GetActiveCamera()
        aspect = float(renderer.GetTiledAspectRatio())
        matrix = camera.GetCompositeProjectionTransformMatrix(aspect, 0.0, 1.0)
        transform = np.asarray(
            [
                [matrix.GetElement(row, column) for column in range(4)]
                for row in range(4)
            ],
            dtype=float,
        )
        source = np.asarray(points, dtype=float).reshape((-1, 3))
        homogeneous = np.column_stack((source, np.ones(len(source), dtype=float)))
        clip = homogeneous @ transform.T
        valid = np.abs(clip[:, 3]) > 1.0e-15
        normalized = np.full((len(source), 3), np.nan, dtype=float)
        normalized[valid] = clip[valid, :3] / clip[valid, 3, None]
        render_window = renderer.GetRenderWindow()
        if render_window is not None:
            width, height = render_window.GetSize()
        else:
            ratio = self._device_pixel_ratio()
            width = int(round(self._plotter.width() * ratio))
            height = int(round(self._plotter.height() * ratio))
        x0, y0, x1, y1 = renderer.GetViewport()
        display = np.empty_like(normalized)
        display[:, 0] = (
            float(width) * x0
            + (normalized[:, 0] + 1.0) * 0.5 * float(width) * (x1 - x0)
        )
        display[:, 1] = (
            float(height) * y0
            + (normalized[:, 1] + 1.0) * 0.5 * float(height) * (y1 - y0)
        )
        display[:, 2] = normalized[:, 2]
        return display

    def _locator_for(self, dataset: Any):
        import vtk

        key = id(dataset)
        modified = int(dataset.GetMTime())
        cached = self._pick_locators.get(key)
        if cached is not None and cached[0] == modified:
            return cached[1]
        locator = vtk.vtkCellLocator()
        locator.SetDataSet(dataset)
        locator.BuildLocator()
        self._pick_locators[key] = (modified, locator)
        return locator

    def _intersect_dataset(
        self,
        x: float,
        y: float,
        dataset: Any,
    ) -> tuple[int, np.ndarray] | None:
        if dataset is None or int(dataset.GetNumberOfCells()) == 0:
            return None
        near = self._display_to_world(x, y, 0.0)
        far = self._display_to_world(x, y, 1.0)
        if near is None or far is None:
            return None
        import vtk

        hit_fraction = vtk.mutable(0.0)
        world = [0.0, 0.0, 0.0]
        pcoords = [0.0, 0.0, 0.0]
        sub_id = vtk.mutable(0)
        cell_id = vtk.mutable(-1)
        hit = self._locator_for(dataset).IntersectWithLine(
            tuple(near),
            tuple(far),
            1.0e-9,
            hit_fraction,
            world,
            pcoords,
            sub_id,
            cell_id,
        )
        if not hit or int(cell_id) < 0:
            return None
        return int(cell_id), np.asarray(world, dtype=float)

    def _display_candidate_is_visible(
        self,
        display: np.ndarray,
        occluder: Any,
    ) -> bool:
        intersection = self._intersect_dataset(
            float(display[0]),
            float(display[1]),
            occluder,
        )
        if intersection is None:
            return True
        _cell_id, front_world = intersection
        front_display = self._world_to_display(front_world)
        return (
            front_display is None
            or float(display[2]) <= float(front_display[2]) + 2.0e-3
        )

    def _show_preselection(self, hit: PickHit | None) -> None:
        self._remove_actor("preselection")
        if hit is None or self._plotter is None or _pyvista is None:
            self._render()
            return
        data = None
        kwargs: dict[str, Any] = {}
        if hit.kind == "geometry_point" and self._geometry_preview_points is not None:
            ids = np.asarray(
                self._geometry_preview_points.point_data["geometry_pick_id"]
            )
            indices = np.flatnonzero(ids == hit.pick_id)
            if len(indices):
                data = _pyvista.PolyData(
                    np.asarray(self._geometry_preview_points.points)[indices]
                )
                kwargs = {"point_size": 11, "render_points_as_spheres": True}
        elif hit.kind == "geometry_edge" and self._geometry_preview_edges is not None:
            ids = np.asarray(
                self._geometry_preview_edges.cell_data["geometry_pick_id"]
            )
            cells = np.flatnonzero(ids == hit.pick_id)
            if len(cells):
                data = self._geometry_preview_edges.extract_cells(cells)
                kwargs = {"line_width": 4}
        elif hit.kind in {"geometry_face", "geometry_body"} and self._geometry_preview_surface is not None:
            if hit.kind == "geometry_body":
                data = self._geometry_preview_surface
            else:
                ids = np.asarray(
                    self._geometry_preview_surface.cell_data["geometry_pick_id"]
                )
                cells = np.flatnonzero(ids == hit.pick_id)
                if len(cells):
                    data = self._geometry_preview_surface.extract_cells(cells)
            kwargs = {"opacity": 0.38}
        elif hit.kind == "node":
            if (
                hit.dataset_name == _TYPED_RESULT_GRID_NAME
                and self._rendered_result_payload() is not None
            ):
                dataset = self._result_grid
                ids = self._typed_result_point_ids(dataset)
            elif (
                self._pick_grid is not None
                and "node_id" in self._pick_grid.point_data
            ):
                dataset = self._pick_grid
                ids = np.asarray(dataset.point_data["node_id"])
            else:
                dataset = None
                ids = np.empty(0, dtype=np.int64)
            indices = np.flatnonzero(ids == hit.pick_id)
            if dataset is not None and len(indices):
                data = _pyvista.PolyData(np.asarray(dataset.points)[indices])
                kwargs = {"point_size": 11, "render_points_as_spheres": True}
        elif hit.kind == "element":
            if (
                hit.dataset_name == _TYPED_RESULT_GRID_NAME
                and self._rendered_result_payload() is not None
            ):
                dataset = self._result_grid
                ids = self._typed_result_cell_ids(dataset)
            elif (
                self._pick_grid is not None
                and "element_id" in self._pick_grid.cell_data
            ):
                dataset = self._pick_grid
                ids = np.asarray(dataset.cell_data["element_id"])
            else:
                dataset = None
                ids = np.empty(0, dtype=np.int64)
            cells = np.flatnonzero(ids == hit.pick_id)
            if dataset is not None and len(cells):
                data = dataset.extract_cells(cells)
                kwargs = {"style": "wireframe", "line_width": 2}
        if data is None:
            self._render()
            return
        self._actors["preselection"] = self._plotter.add_mesh(
            data,
            color="#38b8c8",
            show_edges=False,
            show_scalar_bar=False,
            name="preselection",
            reset_camera=False,
            **kwargs,
        )
        self._offset_highlight_actor(self._actors["preselection"])
        self._update_pickable_actors()
        self._render()

    def _clear_preselection(self, *, render: bool) -> None:
        had_preselection = self._hover_hit is not None or "preselection" in self._actors
        self._hover_hit = None
        self._remove_actor("preselection")
        if render and had_preselection:
            self._render()

    def _remove_actor(self, name: str) -> None:
        actor = self._actors.pop(name, None)
        if actor is not None and self._plotter is not None:
            try:
                self._plotter.remove_actor(actor, reset_camera=False, render=False)
            except Exception:
                pass

    def _update_pickable_actors(self) -> None:
        """Keep auxiliary display layers out of VTK's picking machinery."""
        target_names: set[str] = set()
        if self._selection_mode == "geometry_point":
            target_names.add("geometry_points")
        elif self._selection_mode == "geometry_edge":
            target_names.add("geometry_edges")
        elif self._selection_mode in {"geometry_face", "geometry_body"}:
            target_names.add("geometry_surface")
        elif self._selection_mode == "node" and "nodes" in self._actors:
            target_names.add("nodes")
        elif self._selection_mode == "element":
            target_names.add("mesh_surface")
        for name, actor in self._actors.items():
            try:
                actor.SetPickable(name in target_names)
            except (AttributeError, TypeError):
                continue

    @staticmethod
    def _offset_highlight_actor(actor: Any) -> None:
        """Avoid coplanar hover/selection flicker without moving geometry."""
        try:
            mapper = actor.GetMapper()
            mapper.SetResolveCoincidentTopologyToPolygonOffset()
            mapper.SetRelativeCoincidentTopologyPolygonOffsetParameters(-1.0, -1.0)
        except (AttributeError, TypeError):
            return

    def _update_background_stylesheet(self) -> None:
        settings = self._background_settings
        if settings.style == "gradient":
            background = (
                "qlineargradient(x1:0,y1:1,x2:0,y2:0,"
                f"stop:0 {settings.bottom_color},stop:1 {settings.top_color})"
            )
        else:
            background = settings.bottom_color
        foreground = settings.foreground_color
        self.setStyleSheet(f"background:{background};")
        self._message.setStyleSheet(
            f"background:{background}; color:{foreground}; font-size:11pt;"
        )

    def _apply_plotter_background(self) -> None:
        if self._plotter is None:
            return
        settings = self._background_settings
        top = settings.top_color if settings.style == "gradient" else None
        self._plotter.set_background(settings.bottom_color, top=top)
        try:
            self._plotter.hide_axes()
            self._plotter.add_axes(color=settings.foreground_color)
        except Exception:
            pass

    def _visual_palette(self) -> dict[str, str]:
        dark = self._background_settings.is_dark and self._background_settings.auto_contrast
        if dark:
            return {
                "mesh": "#718797", "edge": "#d9e2e8", "node": "#f0f3f5",
                "result": "#8295a5", "overlay": "#e0e6ea",
                "label_background": "#263746",
            }
        return {
            "mesh": "#d8dde2", "edge": "#4f5963", "node": "#35495e",
            "result": "#b9c6d2", "overlay": "#7f8c8d",
            "label_background": "#ffffff",
        }

    def _set_actor_color(self, name: str, color: str) -> None:
        actor = self._actors.get(name)
        if actor is None:
            return
        qcolor = QColor(color)
        try:
            actor.GetProperty().SetColor(qcolor.redF(), qcolor.greenF(), qcolor.blueF())
        except Exception:
            try:
                actor.prop.color = color
            except Exception:
                pass

    def _refresh_extrema_for_background(self) -> None:
        self._remove_actor("extrema")
        payload = self._result_render_payload
        if (
            payload is not None
            and self._result_grid is payload.dataset
            and self._display.contour_enabled
            and (self._contour["show_minimum"] or self._contour["show_maximum"])
        ):
            self._add_result_render_payload_extrema_labels(payload)

    def _update_scalar_bar_text_color(self) -> None:
        if self._plotter is None:
            return
        color = QColor(self._background_settings.foreground_color)
        rgb = (color.redF(), color.greenF(), color.blueF())
        try:
            for scalar_bar in self._plotter.scalar_bars.values():
                scalar_bar.GetTitleTextProperty().SetColor(*rgb)
                scalar_bar.GetLabelTextProperty().SetColor(*rgb)
        except Exception:
            pass

    def _remove_all_layers(self, *, render: bool = True) -> None:
        for name in tuple(self._actors):
            self._remove_actor(name)
        self._remove_scalar_bars()
        if render:
            self._render()

    def _remove_scalar_bars(self) -> None:
        """Remove stale legends before switching or rebuilding display layers."""
        if self._plotter is None:
            return
        scalar_bars = getattr(self._plotter, "scalar_bars", None)
        if scalar_bars is None:
            return
        for title in tuple(scalar_bars.keys()):
            try:
                self._plotter.remove_scalar_bar(title, render=False)
            except Exception:
                pass

    def _render(self) -> None:
        if self._plotter is not None:
            try:
                self._update_pickable_actors()
                self._plotter.render()
            except Exception:
                pass
