"""延迟加载 PyVistaQt 的分层有限元视口。"""

from __future__ import annotations

import inspect
import os
from dataclasses import replace
from typing import Any

import numpy as np
from PySide6.QtCore import QEvent, Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QLabel, QStackedLayout, QVBoxLayout, QWidget

from fem.boundary.step import boundary_for_step, get_step
from ..viewport_background import ViewportBackgroundSettings
from ..visualization.model_adapter import ModelGeometry, pyvista_cell_array
from ..visualization.result_adapter import ResultData, ScalarField, deformed_points
from ..visualization.stress_adapter import build_stress_render_geometry
from ..visualization.scene import DisplayState
from ..visualization.symbols import (
    SymbolSettings,
    arc_points,
    camera_facing_offset,
    constraint_sample_indices,
    constraint_spatial_regions,
    constraint_symbol_dimensions,
    load_symbol_length,
    region_sample_indices,
    symbol_length,
)

_pyvista = None
_QtInteractor = None
_backend_error: Exception | None = None
_backend_attempted = False


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


class FEMViewport(QWidget):
    """维护网格、标注、选择、载荷与结果等独立 Actor。"""

    entityPicked = Signal(str, int)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._background_settings = ViewportBackgroundSettings()
        self._plotter = None
        self._grid = None
        self._result_grid = None
        self._result_point_index_to_node_id: dict[int, int] = {}
        self._result_point_index_to_element_id: dict[int, int | None] = {}
        self._result_cell_index_to_element_id: dict[int, int] = {}
        self._model = None
        self._geometry: ModelGeometry | None = None
        self._result_data: ResultData | None = None
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
        self._deformation_scale = 1.0
        self._overlay_undeformed = False
        self._symbol_settings = SymbolSettings()
        self._symbols_visible = True
        self._boundary_cache: dict[str | None, Any] = {}
        self._last_symbol_scale: float | None = None
        self._last_symbol_camera_position: np.ndarray | None = None
        self._updating_symbol_scale = False
        self._selection_press_position: tuple[float, float] | None = None
        self._selection_dragged = False
        self._abaqus_view_button: Qt.MouseButton | None = None
        self._trackball_vector: np.ndarray | None = None
        self._contour = {
            "manual": False, "minimum": 0.0, "maximum": 1.0, "levels": 12,
            "colormap": "jet", "style": "continuous", "legend": True, "edges": False,
            "number_format": "general", "decimals": 5,
            "orientation": "horizontal", "show_minimum": True,
            "show_maximum": True, "show_ids": False,
            "averaging_threshold": 75.0,
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

    @staticmethod
    def _uses_abaqus_view_modifier(modifiers: Qt.KeyboardModifier) -> bool:
        """Return whether Ctrl+Alt activates temporary view manipulation."""
        return bool(modifiers & Qt.KeyboardModifier.ControlModifier) and bool(
            modifiers & Qt.KeyboardModifier.AltModifier
        )

    def eventFilter(self, watched: object, event: object) -> bool:
        """Separate Abaqus-style camera drags from ordinary selection clicks."""
        if watched is not self._plotter or self._plotter is None:
            return super().eventFilter(watched, event)

        event_type = event.type()
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
                self._abaqus_view_button = button
                self._selection_press_position = None
                if button == Qt.MouseButton.LeftButton:
                    position = event.position()
                    self._trackball_vector = self._trackball_point(
                        position.x(), position.y()
                    )
                    return True
                return False
            if button == Qt.MouseButton.LeftButton:
                position = event.position()
                self._selection_press_position = (position.x(), position.y())
                self._selection_dragged = False
                return True
            if button in {Qt.MouseButton.MiddleButton, Qt.MouseButton.RightButton}:
                return True

        if event_type == QEvent.Type.MouseMove:
            if self._abaqus_view_button == Qt.MouseButton.LeftButton:
                position = event.position()
                self._rotate_trackball(position.x(), position.y())
                return True
            if self._abaqus_view_button is not None:
                return False
            if self._selection_press_position is not None:
                position = event.position()
                start_x, start_y = self._selection_press_position
                if abs(position.x() - start_x) + abs(position.y() - start_y) > 4.0:
                    self._selection_dragged = True
                return True
            if event.buttons() & (
                Qt.MouseButton.LeftButton
                | Qt.MouseButton.MiddleButton
                | Qt.MouseButton.RightButton
            ):
                return True

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
                position = event.position()
                should_pick = not self._selection_dragged
                self._selection_press_position = None
                self._selection_dragged = False
                if should_pick:
                    self._pick_qt_position(position.x(), position.y())
                return True
            if button in {Qt.MouseButton.MiddleButton, Qt.MouseButton.RightButton}:
                return True

        return super().eventFilter(watched, event)

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
    ) -> None:
        self._model = model
        self._geometry = geometry
        self._result_data = None
        self._selected_kind = None
        self._selected_id = None
        self._selection_highlight_visible = True
        self._selection_press_position = None
        self._selection_dragged = False
        self._abaqus_view_button = None
        self._trackball_vector = None
        self._display = DisplayState()
        self._overlay_undeformed = False
        self._boundary_cache.clear()
        self._last_symbol_scale = None
        self._last_symbol_camera_position = None
        if is_offscreen_environment():
            self._message.setText("模型已加载（当前环境未启用三维渲染）")
            return
        if not self._ensure_plotter():
            return
        self._remove_all_layers(render=False)
        self._grid = self._make_grid(geometry.points)
        self._add_base_layers(reset_camera=True, render=False)
        if refresh_symbols:
            self.show_boundary_and_loads(render=render)
        elif render:
            self._render()

    def clear_model(self) -> None:
        self._model = None
        self._geometry = None
        self._result_data = None
        self._display = DisplayState()
        self._overlay_undeformed = False
        self._selected_kind = None
        self._selected_id = None
        self._selection_highlight_visible = True
        self._selection_press_position = None
        self._selection_dragged = False
        self._abaqus_view_button = None
        self._trackball_vector = None
        self._boundary_cache.clear()
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

    def set_result_data(self, data: ResultData) -> None:
        self._result_data = data
        if self._display.field_key not in data.fields:
            field_key = "U" if "U" in data.fields else next(iter(data.fields), None)
            self._display = replace(self._display, field_key=field_key)

    def set_selection_mode(self, mode: str) -> None:
        self._selection_mode = "element" if mode == "element" else "node"

    def clear_selection(self) -> None:
        self._selected_kind = None
        self._selected_id = None
        self._selection_highlight_visible = True
        self._remove_actor("selection")
        self._render()

    def highlight_node(self, node_id: int) -> None:
        if self._geometry is None or node_id not in self._geometry.node_id_to_point_index:
            return
        index = self._geometry.node_id_to_point_index[node_id]
        self._selected_kind = "node"
        self._selected_id = int(node_id)
        self._selection_highlight_visible = True
        self._remove_actor("selection")
        if self._plotter is not None and _pyvista is not None:
            point = _pyvista.PolyData(self._current_points()[[index]])
            self._actors["selection"] = self._plotter.add_mesh(
                point, color="#d69a3a", point_size=14, render_points_as_spheres=True,
                name="selection", reset_camera=False,
            )
            self._render()

    def highlight_element(self, element_id: int) -> None:
        if self._geometry is None or element_id not in self._geometry.element_id_to_cell_index:
            return
        index = self._geometry.element_id_to_cell_index[element_id]
        self._selected_kind = "element"
        self._selected_id = int(element_id)
        self._selection_highlight_visible = True
        self._remove_actor("selection")
        if self._grid is not None and self._plotter is not None:
            selected = self._grid.extract_cells([index])
            self._actors["selection"] = self._plotter.add_mesh(
                selected, color="#d69a3a", style="wireframe", line_width=3,
                name="selection", reset_camera=False,
            )
            self._render()

    def highlight_nodes(self, node_ids: tuple[int, ...]) -> None:
        if self._geometry is None or _pyvista is None or self._plotter is None:
            return
        indices = [self._geometry.node_id_to_point_index[node_id] for node_id in node_ids if node_id in self._geometry.node_id_to_point_index]
        self._remove_actor("set_highlight")
        if indices:
            self._actors["set_highlight"] = self._plotter.add_mesh(
                _pyvista.PolyData(self._current_points()[indices]), color="#4f8fa8",
                point_size=12, render_points_as_spheres=True, name="set_highlight",
                reset_camera=False,
            )
        self._render()

    def highlight_elements(self, element_ids: tuple[int, ...]) -> None:
        if self._geometry is None or self._grid is None or self._plotter is None:
            return
        indices = [self._geometry.element_id_to_cell_index[element_id] for element_id in element_ids if element_id in self._geometry.element_id_to_cell_index]
        self._remove_actor("set_highlight")
        if indices:
            self._actors["set_highlight"] = self._plotter.add_mesh(
                self._grid.extract_cells(indices), color="#4f8fa8", style="wireframe",
                line_width=3, name="set_highlight", reset_camera=False,
            )
        self._render()

    def highlight_region(self, members: tuple[Any, ...], kind: str) -> None:
        """Highlight named surface faces or 2D boundary edges as one actor."""
        if self._geometry is None or _pyvista is None or self._plotter is None:
            return
        connectivity: list[int] = []
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
        region = _pyvista.PolyData(self._current_points())
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

    def set_display(
        self,
        shape_mode: str,
        contour_enabled: bool,
        field_key: str | None = None,
    ) -> None:
        """独立设置几何形状、云图开关和主结果字段。"""
        shape = "deformed" if shape_mode == "deformed" else "undeformed"
        self._display = DisplayState(
            shape_mode=shape,
            contour_enabled=bool(contour_enabled),
            field_key=field_key if field_key is not None else self._display.field_key,
        )
        self._update_result_layer()

    def show_display(self, mode: str, field_key: str | None = None) -> None:
        """兼容旧调用，并将其转换为独立的形状与着色状态。"""
        if mode == "contour":
            self.set_display(self._display.shape_mode, True, field_key)
        else:
            self.set_display(mode, False, field_key)

    def set_field(self, key: str) -> None:
        self._display = replace(self._display, field_key=key)
        if self._display.contour_enabled:
            self._update_result_layer()

    def set_deformation_scale(self, scale: float) -> None:
        self._deformation_scale = float(scale)
        if self._display.shape_mode == "deformed":
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
                render=render,
            )
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
            "constraints", "constraint_rotations", "constraint_rotation_heads",
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
                            name = "R3" if component >= translation_count else f"U{component + 1}"
                            constraint_label_points[node_id] = display_base
                            constraint_labels_by_node.setdefault(node_id, []).append(
                                f"{name}={value:.6g}" if value else name
                            )
                            if component >= translation_count:
                                rotation_points.append(display_base)
                                axis = np.zeros(3)
                                axis[(component - translation_count + (2 if not is_3d else 0)) % 3] = 1.0
                                rotation_axes.append(axis)
                                continue
                            direction = np.zeros(3)
                            direction[component] = -1.0
                            constraint_points.append(display_base)
                            constraint_vectors.append(direction)
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
            self._add_arc_symbols(
                rotation_points, rotation_axes, 0.55 * constraint_scale,
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
            nodal_moments: dict[int, float] = {}
            for dof, value in boundary.nodal_forces.items():
                node_index, component = divmod(int(dof), self._model.mesh.dofs_per_node)
                node_id = self._model.mesh.dof_map.node_ids[node_index]
                if component < translation_count:
                    nodal_vectors.setdefault(node_id, np.zeros(3))[component] += float(value)
                else:
                    nodal_moments[node_id] = nodal_moments.get(node_id, 0.0) + float(value)
            for node_id, vector in nodal_vectors.items():
                if float(np.linalg.norm(vector)) <= 0.0:
                    continue
                base = self._geometry.points[self._geometry.node_id_to_point_index[node_id]]
                origins.append(base)
                vectors.append(vector)
                load_labels.append(f"F={tuple(float(value) for value in vector[:translation_count])}")
            for node_id, value in nodal_moments.items():
                moment_points.append(self._geometry.points[self._geometry.node_id_to_point_index[node_id]])
                moment_labels.append(f"M3={value:.6g}")
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
                np.array([0.0, 0.0, 1.0 if value >= 0.0 else -1.0])
                for value in nodal_moments.values()
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
        return _pyvista.UnstructuredGrid(
            pyvista_cell_array(self._geometry), self._geometry.cell_types, np.asarray(points, dtype=float)
        )

    def _add_base_layers(self, reset_camera: bool, *, render: bool = True) -> None:
        palette = self._visual_palette()
        self._actors["mesh_surface"] = self._plotter.add_mesh(
            self._grid, color=palette["mesh"], show_edges=False, name="mesh_surface",
            reset_camera=False,
        )
        self._actors["element_edges"] = self._plotter.add_mesh(
            self._grid, color=palette["edge"], style="wireframe", line_width=1,
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
        self._result_grid = None
        self._result_point_index_to_node_id.clear()
        self._result_point_index_to_element_id.clear()
        self._result_cell_index_to_element_id.clear()
        if self._plotter is None or self._geometry is None or self._result_data is None:
            return
        base = self._actors.get("mesh_surface")
        use_deformed = self._display.shape_mode == "deformed"
        use_contour = self._display.contour_enabled
        if not use_deformed and not use_contour:
            if base is not None:
                base.SetVisibility(True)
            self._grid = self._make_grid(self._geometry.points)
            self._refresh_geometry_dependent_layers()
            self._refresh_undeformed_overlay()
            return
        points = (
            deformed_points(self._geometry, self._result_data, self._deformation_scale)
            if use_deformed
            else self._geometry.points
        )
        grid = self._make_grid(points)
        self._grid = grid
        if base is not None:
            base.SetVisibility(False)
        kwargs: dict[str, Any] = {"name": "result", "reset_camera": False}
        field_key = self._display.field_key
        if (
            use_contour
            and field_key in self._result_data.fields
            and self._result_data.fields[field_key].ready
        ):
            scalar: ScalarField = self._result_data.fields[field_key]
            render_scalar = scalar
            if field_key.startswith(("N:", "EN:")):
                stress_geometry = build_stress_render_geometry(
                    self._geometry,
                    self._result_data,
                    field_key,
                    float(self._contour.get("averaging_threshold", 75.0)),
                )
                stress_points = np.asarray(stress_geometry.points, dtype=float).copy()
                if use_deformed:
                    for point_index, node_id in stress_geometry.point_index_to_node_id.items():
                        source = self._geometry.node_id_to_point_index[node_id]
                        stress_points[point_index] += (
                            self._deformation_scale
                            * self._result_data.displacement_vectors[source]
                        )
                grid = _pyvista.UnstructuredGrid(
                    stress_geometry.cell_array,
                    stress_geometry.cell_types,
                    stress_points,
                )
                render_scalar = ScalarField(
                    scalar.key, scalar.label, "point", stress_geometry.values
                )
                self._result_point_index_to_node_id = dict(
                    stress_geometry.point_index_to_node_id
                )
                self._result_point_index_to_element_id = dict(
                    stress_geometry.point_index_to_element_id
                )
                self._result_cell_index_to_element_id = dict(
                    stress_geometry.cell_index_to_element_id
                )
            if render_scalar.association == "point":
                grid.point_data[render_scalar.key] = render_scalar.values
            else:
                grid.cell_data[render_scalar.key] = render_scalar.values
            kwargs.update(
                scalars=scalar.key, cmap=self._contour["colormap"],
                n_colors=(
                    256
                    if self._contour.get("style") == "continuous"
                    else int(self._contour["levels"])
                ),
                interpolate_before_map=self._contour.get("style") == "continuous",
                show_scalar_bar=self._contour["legend"],
                scalar_bar_args={
                    "title": scalar.label,
                    "vertical": self._contour["orientation"] == "vertical",
                    "fmt": self._scalar_format(),
                    "color": self._background_settings.foreground_color,
                },
            )
            if self._contour["manual"]:
                kwargs["clim"] = (self._contour["minimum"], self._contour["maximum"])
        else:
            kwargs.update(color=self._visual_palette()["result"])
        # 单元边由独立 Actor 管理，避免与结果 Actor 重复绘制。
        kwargs["show_edges"] = False
        self._result_grid = grid
        self._actors["result"] = self._plotter.add_mesh(grid, **kwargs)
        if (
            use_contour
            and (self._contour["show_minimum"] or self._contour["show_maximum"])
            and field_key in self._result_data.fields
            and self._result_data.fields[field_key].ready
        ):
            self._add_extrema_labels(grid, render_scalar)
        self._refresh_geometry_dependent_layers()
        self._refresh_undeformed_overlay()

    def _add_extrema_labels(self, grid: Any, scalar: ScalarField) -> None:
        finite = np.flatnonzero(np.isfinite(scalar.values))
        if len(finite) == 0:
            return
        minimum_index = int(finite[np.argmin(scalar.values[finite])])
        maximum_index = int(finite[np.argmax(scalar.values[finite])])
        points = grid.points if scalar.association == "point" else grid.cell_centers().points
        entries: list[tuple[int, str]] = []
        for enabled, index, title in (
            (self._contour["show_minimum"], minimum_index, "最小值"),
            (self._contour["show_maximum"], maximum_index, "最大值"),
        ):
            if not enabled:
                continue
            label = f"{title} {self._format_scalar(float(scalar.values[index]))}"
            if self._contour["show_ids"]:
                object_id = (
                    self._result_point_index_to_node_id.get(
                        index, self._geometry.point_index_to_node_id.get(index, "—")
                    )
                    if scalar.association == "point"
                    else self._result_cell_index_to_element_id.get(
                        index, self._geometry.cell_index_to_element_id.get(index, "—")
                    )
                )
                kind = "节点" if scalar.association == "point" else "单元"
                label += f"（{kind} {object_id}）"
                if scalar.key.startswith("EN:"):
                    element_id = self._result_point_index_to_element_id.get(index)
                    if element_id is not None:
                        label = label.removesuffix("）") + f"，单元 {element_id}）"
            entries.append((index, label))
        indices = [entry[0] for entry in entries]
        labels = [entry[1] for entry in entries]
        if not entries:
            return
        palette = self._visual_palette()
        self._actors["extrema"] = self._plotter.add_point_labels(
            np.asarray(points)[indices], labels, point_size=7, font_size=10,
            shape_color=palette["label_background"],
            text_color=self._background_settings.foreground_color, name="extrema",
            reset_camera=False,
        )

    def _refresh_geometry_dependent_layers(self, *, render: bool = True) -> None:
        self._remove_actor("element_edges")
        self._remove_actor("set_highlight")
        self._actors["element_edges"] = self._plotter.add_mesh(
            self._grid, color=self._visual_palette()["edge"], style="wireframe", line_width=1,
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
            line_width=1,
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
            self._actors["nodes"] = self._plotter.add_mesh(
                _pyvista.PolyData(self._current_points()),
                color=self._visual_palette()["node"], point_size=7,
                render_points_as_spheres=True, name="nodes", reset_camera=False,
            )
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
                self._current_points(), labels, point_size=0, font_size=10,
                shape=None, text_color=self._background_settings.foreground_color,
                name="node_labels", reset_camera=False,
            )
        if self._show_element_labels:
            centers = self._grid.cell_centers().points
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
            self._plotter.installEventFilter(self)
            self._plotter.setToolTip(
                "左键选择；Ctrl+Alt+左键旋转；Ctrl+Alt+中键平移；"
                "Ctrl+Alt+右键缩放；滚轮缩放"
            )
        except Exception:
            return

    def _pick_qt_position(self, x: float, y: float) -> None:
        """Convert a Qt top-left position to VTK display coordinates and pick."""
        if self._plotter is None:
            return
        try:
            pixel_ratio = float(self._plotter._getPixelRatio())
        except (AttributeError, TypeError, ValueError):
            pixel_ratio = float(self._plotter.devicePixelRatioF())
        vtk_x = int(round(x * pixel_ratio))
        vtk_y = int(round((self._plotter.height() - y - 1.0) * pixel_ratio))
        self._pick_at(vtk_x, vtk_y)

    def _pick_at(self, x: int, y: int) -> None:
        """Pick a FEM entity at VTK display coordinates."""
        if self._geometry is None or self._plotter is None:
            return
        import vtk

        picker = vtk.vtkPointPicker() if self._selection_mode == "node" else vtk.vtkCellPicker()
        picker.SetTolerance(0.01)
        if not picker.Pick(x, y, 0, self._plotter.renderer):
            return
        picked_actor = picker.GetActor()
        allowed_names = (
            "mesh_surface", "result", "element_edges", "nodes",
            "undeformed_overlay",
        )
        allowed_actors = {
            self._actors[name]
            for name in allowed_names
            if name in self._actors
        }
        if picked_actor not in allowed_actors:
            return
        if self._selection_mode == "node":
            index = int(picker.GetPointId())
            mapping = (
                self._result_point_index_to_node_id
                if picked_actor == self._actors.get("result")
                and self._result_point_index_to_node_id
                else self._geometry.point_index_to_node_id
            )
            if index in mapping:
                self.entityPicked.emit("node", mapping[index])
            return
        index = int(picker.GetCellId())
        mapping = (
            self._result_cell_index_to_element_id
            if picked_actor == self._actors.get("result")
            and self._result_cell_index_to_element_id
            else self._geometry.cell_index_to_element_id
        )
        if index in mapping:
            self.entityPicked.emit("element", mapping[index])

    def _remove_actor(self, name: str) -> None:
        actor = self._actors.pop(name, None)
        if actor is not None and self._plotter is not None:
            try:
                self._plotter.remove_actor(actor, reset_camera=False, render=False)
            except Exception:
                pass

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
        field_key = self._display.field_key
        if (
            self._grid is not None
            and self._result_data is not None
            and self._display.contour_enabled
            and field_key in self._result_data.fields
            and (self._contour["show_minimum"] or self._contour["show_maximum"])
        ):
            self._add_extrema_labels(self._grid, self._result_data.fields[field_key])

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
        if render:
            self._render()

    def _render(self) -> None:
        if self._plotter is not None:
            try:
                self._plotter.render()
            except Exception:
                pass
