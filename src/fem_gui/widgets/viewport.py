"""延迟加载 PyVistaQt 的分层有限元视口。"""

from __future__ import annotations

import inspect
import logging
import math
import os
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

import numpy as np
from PySide6.QtCore import QEvent, QTimer, Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QLabel,
    QStackedLayout,
    QVBoxLayout,
    QWidget,
)

from fem.application import MeshEntityRef, RegionRef
from fem.application.results import (
    FieldLocation,
    FieldPosition,
    ResultCellKind,
    ResultValueLayout,
)
from fem.boundary.step import (
    boundary_for_step,
    effective_step_boundaries,
    get_step,
)
from fem.core._constraint_targets import (
    displacement_target_kind,
    resolve_displacement_node_ids,
)
from fem.geometry import (
    LogicalEntityRef,
    SketchArc,
    SketchCircle,
    SketchCurve,
    SketchLine,
    SketchPlane,
    SketchPoint,
    SketchReferencePoint,
    SketchSnapCandidate,
    logical_ref_sort_key,
    select_sketch_snap_candidate,
    sketch_intersections,
)
from ..geometry_preview import FaceSketchBooleanDisplay, GeometryPreview
from ..result_presentation import result_field_position_label
from ..scope_selection import build_mesh_selection_topology
from ..sketch_constraint_ui import SketchConstraintOverlay
from ..wire_editor import (
    intersect_ray_with_work_plane,
    snap_work_plane_point,
)
from ..viewport_background import ViewportBackgroundSettings
from ..visualization.colormaps import (
    ABAQUS_RAINBOW,
    resolve_contour_colormap,
)
from ..visualization.contour_rendering import (
    CONTOUR_EDGE_GEOMETRY,
    CONTOUR_EDGE_NONE,
    CONTOUR_RENDER_SHADED,
    contour_surface_options,
    extract_contour_edges,
    style_contour_edges,
)
from ..visualization.model_adapter import ModelGeometry, pyvista_cell_array
from ..visualization.scene import DisplayState
from ..visualization.symbols import (
    SymbolSettings,
    arc_points,
    camera_facing_offset,
    constraint_outward_direction,
    constraint_rotation_axes,
    constraint_sample_indices,
    constraint_spatial_regions,
    constraint_symbol_dimensions,
    load_arrow_origins,
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
_LINE_ELEMENT_WIDTH = 5
_LINE_NODE_POINT_SIZE = 11
_GRAVITY_SYMBOL_COLOR = "#FFD400"


class _SelectionRubberBand:
    """Draw a border-only rectangle in VTK display coordinates."""

    def __init__(self, renderer: Any) -> None:
        import vtk

        self._visible = False
        self._containment = True
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
            actor.PickableOff()
            actor.SetVisibility(False)
            renderer.AddViewProp(actor)
            self._actors.append(actor)

    def set_containment(self, containment: bool) -> None:
        normalized = bool(containment)
        if normalized == self._containment:
            return
        self._containment = normalized
        pattern = 0xFFFF if normalized else 0xF0F0
        for actor in self._actors:
            actor.GetProperty().SetLineStipplePattern(pattern)
            actor.GetProperty().SetLineStippleRepeatFactor(1)

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
    body_pick_ids: tuple[int, ...] = (),
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
    if body_pick_ids:
        edge_mesh.cell_data["geometry_body_pick_id"] = np.asarray(
            body_pick_ids,
            dtype=np.int64,
        )
    if any(part_id is not None for part_id in preview.edge_part_ids):
        edge_mesh.cell_data["geometry_part_id"] = np.asarray(
            tuple(part_id or "" for part_id in preview.edge_part_ids),
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
    if any(part_id is not None for part_id in preview.point_part_ids):
        point_mesh.point_data["geometry_part_id"] = np.asarray(
            tuple(part_id or "" for part_id in preview.point_part_ids),
        )[selectable]
    return point_mesh


def _geometry_surface_polydata(
    pyvista,
    points: np.ndarray,
    preview: GeometryPreview,
    pick_ids: tuple[int, ...],
    body_pick_ids: tuple[int, ...] = (),
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
    if body_pick_ids:
        surface.cell_data["geometry_body_pick_id"] = np.asarray(
            body_pick_ids,
            dtype=np.int64,
        )
    if any(part_id is not None for part_id in preview.face_part_ids):
        surface.cell_data["geometry_part_id"] = np.asarray(
            tuple(part_id or "" for part_id in preview.face_part_ids),
        )
    surface = surface.triangulate()
    surface.set_active_scalars(None)
    return surface


def _mesh_edge_polydata(
    pyvista,
    geometry: ModelGeometry,
    rows: tuple[tuple[int, int, tuple[int, ...]], ...],
    *,
    points: np.ndarray | None = None,
):
    """Build one pickable display cell for every boundary element edge."""

    cells = tuple(
        tuple(
            geometry.node_id_to_point_index[int(node_id)]
            for node_id in node_ids
        )
        for _element_id, _local_index, node_ids in rows
    )
    line_cells = np.hstack(
        [np.asarray((len(cell), *cell), dtype=np.int64) for cell in cells]
    )
    dataset = pyvista.PolyData()
    dataset.points = geometry.points if points is None else points
    dataset.lines = line_cells
    dataset.cell_data["mesh_scope_pick_id"] = np.arange(
        1,
        len(rows) + 1,
        dtype=np.int64,
    )
    dataset.set_active_scalars(None)
    return dataset, cells


def _line_only_polydata(pyvista, points: np.ndarray, lines: Iterable[int]):
    """Build line cells without PyVista's implicit vertex cells."""

    dataset = pyvista.PolyData()
    dataset.points = points
    dataset.lines = np.asarray(tuple(lines), dtype=np.int64)
    dataset.set_active_scalars(None)
    return dataset


def _mesh_face_polydata(
    pyvista,
    geometry: ModelGeometry,
    rows: tuple[tuple[int, int, tuple[int, ...]], ...],
    *,
    points: np.ndarray | None = None,
):
    """Build boundary-face polygons while retaining element-local identity."""

    display_node_ids = tuple(
        _face_display_node_ids(node_ids)
        for _element_id, _local_index, node_ids in rows
    )
    cells = tuple(
        tuple(
            geometry.node_id_to_point_index[int(node_id)]
            for node_id in node_ids
        )
        for node_ids in display_node_ids
    )
    face_cells = np.hstack(
        [np.asarray((len(cell), *cell), dtype=np.int64) for cell in cells]
    )
    dataset = pyvista.PolyData(
        geometry.points if points is None else points,
        faces=face_cells,
    )
    dataset.cell_data["mesh_scope_pick_id"] = np.arange(
        1,
        len(rows) + 1,
        dtype=np.int64,
    )
    dataset.set_active_scalars(None)
    return dataset


def _face_display_node_ids(node_ids: tuple[int, ...]) -> tuple[int, ...]:
    """Return corner nodes in perimeter order for supported quadratic faces."""

    if len(node_ids) == 8:
        return node_ids[:4]
    if len(node_ids) == 6:
        return node_ids[:3]
    return node_ids


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


@dataclass(slots=True)
class _MeshScopeHighlightPipeline:
    """One persistent VTK mask/filter/actor chain for a mesh entity kind."""

    kind: str
    dataset: Any
    algorithm: Any
    mask: np.ndarray
    vtk_mask: Any
    actor: Any
    selected_indices: set[int] = field(default_factory=set)

    def update(self, target: set[int]) -> bool:
        return self.update_changes(
            {
                index: index in target
                for index in self.selected_indices.symmetric_difference(target)
            }
        )

    def update_changes(self, changes: dict[int, bool]) -> bool:
        changed = False
        for index, selected in changes.items():
            if (index in self.selected_indices) == selected:
                continue
            self.mask[index] = 1 if selected else 0
            if selected:
                self.selected_indices.add(index)
            else:
                self.selected_indices.discard(index)
            changed = True
        if not changed:
            return False
        self.vtk_mask.Modified()
        self.dataset.Modified()
        self.algorithm.Modified()
        return True

    def clear(self) -> bool:
        if not self.selected_indices:
            return False
        for index in self.selected_indices:
            self.mask[index] = 0
        self.selected_indices.clear()
        self.vtk_mask.Modified()
        self.dataset.Modified()
        self.algorithm.Modified()
        return True


@dataclass(frozen=True, slots=True)
class WireDraftRenderData:
    """Detached display data for an incomplete wire draft."""

    points: tuple[tuple[float, float, float], ...]
    point_names: tuple[str, ...]
    members: tuple[tuple[int, int], ...]
    member_names: tuple[str, ...]
    pending_member_start: str | None = None

    def __post_init__(self) -> None:
        points = tuple(tuple(float(value) for value in point) for point in self.points)
        point_names = tuple(self.point_names)
        members = tuple(tuple(int(value) for value in member) for member in self.members)
        member_names = tuple(self.member_names)
        if len(points) != len(point_names):
            raise ValueError("wire draft point names must match point coordinates")
        if len(members) != len(member_names):
            raise ValueError("wire draft member names must match member cells")
        if any(len(member) != 2 for member in members):
            raise ValueError("wire draft members must be two-index cells")
        if any(
            index < 0 or index >= len(points)
            for member in members
            for index in member
        ):
            raise ValueError("wire draft member index is outside point data")
        if self.pending_member_start is not None and self.pending_member_start not in point_names:
            raise ValueError("pending wire member start must name a draft point")
        object.__setattr__(self, "points", points)
        object.__setattr__(self, "point_names", point_names)
        object.__setattr__(self, "members", members)
        object.__setattr__(self, "member_names", member_names)


@dataclass(frozen=True, slots=True)
class SketchDraftRenderData:
    """Detached display data for an incomplete planar sketch draft."""

    points: tuple[tuple[float, float, float], ...]
    point_ids: tuple[str | None, ...]
    curves: tuple[tuple[int, ...], ...]
    curve_ids: tuple[str, ...]
    faces: tuple[tuple[int, ...], ...] = ()
    face_ids: tuple[str, ...] = ()
    selected_kind: str | None = None
    selected_id: str | None = None
    plane: SketchPlane = field(default_factory=SketchPlane.xy)
    selected_ids: tuple[str, ...] = ()
    snap_midpoints: tuple[tuple[float, float, float], ...] = ()
    snap_centers: tuple[tuple[float, float, float], ...] = ()
    geometry_revision: int = 0
    constraint_status: str = "under_constrained"
    analytic_points: tuple[SketchPoint, ...] = ()
    analytic_curves: tuple[SketchCurve, ...] = ()
    constraint_overlays: tuple[SketchConstraintOverlay, ...] = ()
    inference_preview: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        points = tuple(
            tuple(float(value) for value in point)
            for point in self.points
        )
        point_ids = tuple(self.point_ids)
        curves = tuple(
            tuple(int(value) for value in curve)
            for curve in self.curves
        )
        curve_ids = tuple(str(value) for value in self.curve_ids)
        faces = tuple(
            tuple(int(value) for value in face)
            for face in self.faces
        )
        face_ids = tuple(str(value) for value in self.face_ids)
        analytic_points = tuple(self.analytic_points)
        analytic_curves = tuple(self.analytic_curves)
        constraint_overlays = tuple(self.constraint_overlays)
        if len(points) != len(point_ids):
            raise ValueError("sketch draft point IDs must match point coordinates")
        if len(curves) != len(curve_ids):
            raise ValueError("sketch draft curve IDs must match curve cells")
        if len(faces) != len(face_ids):
            raise ValueError("sketch draft face IDs must match face cells")
        if any(len(curve) < 2 for curve in curves):
            raise ValueError("sketch draft curves require at least two points")
        if any(len(face) < 3 for face in faces):
            raise ValueError("sketch draft faces require at least three points")
        if any(
            index < 0 or index >= len(points)
            for cell in (*curves, *faces)
            for index in cell
        ):
            raise ValueError("sketch draft cell index is outside point data")
        if isinstance(self.geometry_revision, bool) or not isinstance(
            self.geometry_revision, int
        ):
            raise TypeError("sketch geometry revision must be an integer")
        if self.constraint_status not in {
            "under_constrained",
            "fully_constrained",
            "redundant",
            "conflicting",
            "failed",
        }:
            raise ValueError("invalid sketch constraint status")
        if any(type(point) is not SketchPoint for point in analytic_points):
            raise TypeError("analytic_points must contain SketchPoint values")
        if any(
            type(curve) not in (SketchLine, SketchCircle, SketchArc)
            for curve in analytic_curves
        ):
            raise TypeError("analytic_curves must contain strict sketch curves")
        selected_ids = tuple(dict.fromkeys(str(value) for value in self.selected_ids))
        snap_midpoints = tuple(
            tuple(float(value) for value in point)
            for point in self.snap_midpoints
        )
        snap_centers = tuple(
            tuple(float(value) for value in point)
            for point in self.snap_centers
        )
        if any(len(point) != 3 for point in (*snap_midpoints, *snap_centers)):
            raise ValueError("sketch snap points must have three coordinates")
        if any(
            not np.all(np.isfinite(point))
            for point in (*snap_midpoints, *snap_centers)
        ):
            raise ValueError("sketch snap point coordinates must be finite")
        if self.selected_id is not None and not selected_ids:
            selected_ids = (str(self.selected_id),)
        selected_id = selected_ids[0] if selected_ids else None
        if self.selected_kind not in {None, "point", "curve", "profile"}:
            raise ValueError("invalid sketch draft selection kind")
        if selected_id is None and self.selected_kind is not None:
            raise ValueError("sketch selection kind requires an entity ID")
        if selected_id is not None and self.selected_kind is None:
            raise ValueError("sketch selection IDs require an entity kind")
        if type(self.plane) is not SketchPlane:
            raise TypeError("sketch draft plane must be a SketchPlane")
        object.__setattr__(self, "points", points)
        object.__setattr__(self, "point_ids", point_ids)
        object.__setattr__(self, "curves", curves)
        object.__setattr__(self, "curve_ids", curve_ids)
        object.__setattr__(self, "faces", faces)
        object.__setattr__(self, "face_ids", face_ids)
        object.__setattr__(self, "analytic_points", analytic_points)
        object.__setattr__(self, "analytic_curves", analytic_curves)
        object.__setattr__(self, "constraint_overlays", constraint_overlays)
        object.__setattr__(self, "selected_id", selected_id)
        object.__setattr__(self, "selected_ids", selected_ids)
        object.__setattr__(self, "snap_midpoints", snap_midpoints)
        object.__setattr__(self, "snap_centers", snap_centers)


def _sketch_geometry_color(constraint_status: str) -> str:
    return {
        "under_constrained": "#1976a8",
        "fully_constrained": "#2e7d32",
        "redundant": "#ef6c00",
        "conflicting": "#c62828",
        "failed": "#c62828",
    }[constraint_status]


@dataclass(frozen=True, slots=True)
class _ViewportCameraState:
    """Camera values that control framing and on-screen magnification."""

    position: tuple[float, float, float]
    focal_point: tuple[float, float, float]
    view_up: tuple[float, float, float]
    parallel_scale: float
    view_angle: float
    parallel_projection: int


@dataclass(frozen=True, slots=True)
class _WireGridLayout:
    """A work-plane grid whose visible lines stay on snap coordinates."""

    axes: tuple[int, int, int]
    center: tuple[float, float, float]
    plane_size: float
    resolution: int
    visible_spacing: float


def _wire_grid_layout(
    points: np.ndarray,
    plane: str,
    offset: float,
    snap_spacing: float,
) -> _WireGridLayout:
    """Size and align a bounded work-plane grid without losing snap alignment."""

    axes = {"XY": (0, 1, 2), "XZ": (0, 2, 1), "YZ": (1, 2, 0)}[plane]
    spacing = float(snap_spacing)
    spans = (
        tuple(float(np.ptp(points[:, axis])) for axis in axes[:2])
        if len(points)
        else (0.0, 0.0)
    )
    target_size = max(10.0, max(spans, default=0.0) + 2.0 * spacing)
    requested_intervals = max(1, int(math.ceil(target_size / spacing)))
    major_factor = max(1, int(math.ceil(requested_intervals / 2000)))
    visible_spacing = spacing * major_factor
    resolution = max(2, int(math.ceil(target_size / visible_spacing)))
    if resolution % 2:
        resolution += 1
    plane_size = resolution * visible_spacing
    center = np.mean(points, axis=0) if len(points) else np.zeros(3, dtype=float)
    center = np.asarray(center, dtype=float)
    for axis in axes[:2]:
        center[axis] = (
            math.floor(center[axis] / visible_spacing + 0.5) * visible_spacing
        )
    center[axes[2]] = float(offset)
    return _WireGridLayout(
        axes,
        tuple(float(value) for value in center),
        float(plane_size),
        resolution,
        float(visible_spacing),
    )


def _wire_grid_polydata(pyvista, layout: _WireGridLayout):
    """Build exact-spacing grid lines without creating a dense surface mesh."""

    center = np.asarray(layout.center, dtype=float)
    half_size = 0.5 * layout.plane_size
    first_axis, second_axis, _normal_axis = layout.axes
    points: list[np.ndarray] = []
    lines: list[tuple[int, int, int]] = []
    offsets = np.linspace(
        -half_size,
        half_size,
        layout.resolution + 1,
    )
    for offset in offsets:
        first_start = center.copy()
        first_end = center.copy()
        first_start[first_axis] -= half_size
        first_end[first_axis] += half_size
        first_start[second_axis] += offset
        first_end[second_axis] += offset
        first_index = len(points)
        points.extend((first_start, first_end))
        lines.append((2, first_index, first_index + 1))

        second_start = center.copy()
        second_end = center.copy()
        second_start[second_axis] -= half_size
        second_end[second_axis] += half_size
        second_start[first_axis] += offset
        second_end[first_axis] += offset
        second_index = len(points)
        points.extend((second_start, second_end))
        lines.append((2, second_index, second_index + 1))
    grid = pyvista.PolyData()
    grid.points = np.asarray(points, dtype=float)
    grid.lines = np.asarray(lines, dtype=np.int64)
    return grid


def _wire_coordinate_label(point: Iterable[float]) -> str:
    """Format one hover coordinate using glyphs supported by VTK's font."""

    x, y, z = tuple(float(value) for value in point)
    return f"({x:.2f}, {y:.2f}, {z:.2f})"


def _intersect_ray_with_sketch_plane(
    ray_start: Iterable[float],
    ray_end: Iterable[float],
    plane: SketchPlane,
) -> tuple[float, float, float] | None:
    """Intersect a display ray with an arbitrary immutable sketch frame."""

    if type(plane) is not SketchPlane:
        raise TypeError("plane must be a SketchPlane")
    start = np.asarray(tuple(ray_start), dtype=float)
    end = np.asarray(tuple(ray_end), dtype=float)
    origin = np.asarray(plane.origin, dtype=float)
    normal = np.asarray(plane.normal, dtype=float)
    direction = end - start
    denominator = float(np.dot(direction, normal))
    if abs(denominator) <= 1.0e-12:
        return None
    parameter = float(np.dot(origin - start, normal) / denominator)
    point = start + parameter * direction
    return tuple(float(value) for value in point)


def _snap_sketch_plane_point(
    point: Iterable[float],
    plane: SketchPlane,
    spacing: float,
) -> tuple[float, float, float]:
    """Snap in U/V coordinates and map the result back to global space."""

    clean_spacing = float(spacing)
    if not np.isfinite(clean_spacing) or clean_spacing <= 0.0:
        raise ValueError("sketch grid spacing must be positive")
    u, v = plane.to_local(tuple(float(value) for value in point))
    return plane.to_global(
        round(u / clean_spacing) * clean_spacing,
        round(v / clean_spacing) * clean_spacing,
    )


def _sketch_local_points(
    points: Iterable[tuple[float, float, float]],
    plane: SketchPlane,
) -> np.ndarray:
    values = tuple(points)
    if not values:
        return np.empty((0, 2), dtype=float)
    return np.asarray(
        tuple(plane.to_local(point) for point in values),
        dtype=float,
    )


def _sketch_grid_polydata(
    pyvista,
    plane: SketchPlane,
    points: Iterable[tuple[float, float, float]],
    spacing: float,
):
    """Build a bounded grid aligned to the frame's U/V axes."""

    local = _sketch_local_points(points, plane)
    local_3d = np.column_stack(
        (local, np.zeros(len(local), dtype=float))
    )
    layout = _wire_grid_layout(local_3d, "XY", 0.0, spacing)
    grid = _wire_grid_polydata(pyvista, layout)
    grid.points = np.asarray(
        tuple(
            plane.to_global(float(point[0]), float(point[1]))
            for point in grid.points
        ),
        dtype=float,
    )
    return grid, layout


def _sketch_axis_local_endpoints(
    layout: _WireGridLayout,
    axis: int,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Keep sketch axes anchored to the work-plane origin as the grid moves."""

    if axis not in {0, 1}:
        raise ValueError("sketch axis must be 0 or 1")
    center = np.asarray(layout.center[:2], dtype=float)
    half_size = 0.5 * layout.plane_size
    start = np.zeros(2, dtype=float)
    end = np.zeros(2, dtype=float)
    start[axis] = min(float(center[axis] - half_size), 0.0)
    end[axis] = max(float(center[axis] + half_size), 0.0)
    return (
        tuple(float(value) for value in start),
        tuple(float(value) for value in end),
    )


def _sketch_coordinate_label(
    point: Iterable[float],
    plane: SketchPlane,
) -> str:
    u, v = plane.to_local(tuple(float(value) for value in point))
    return f"(U={u:.2f}, V={v:.2f})"


def _sketch_snap_label(kind: str | None) -> str | None:
    return {
        "sketch_point": "草图点",
        "topology_vertex": "外部参考点",
        "face_center": "外部参考点",
        "line_midpoint": "中点",
        "circle_center": "圆心",
        "arc_center": "圆心",
        "intersection": "交点",
        "grid": "网格点",
    }.get(kind)


def _sketch_camera_bounds(
    points: Iterable[tuple[float, float, float]],
    spacing: float,
    plane: SketchPlane | None = None,
) -> tuple[float, float, float, float, float, float]:
    """Frame a useful authoring area independently of the full grid actor."""

    clean_spacing = float(spacing)
    if not np.isfinite(clean_spacing) or clean_spacing <= 0.0:
        raise ValueError("sketch grid spacing must be positive")
    frame = SketchPlane.xy() if plane is None else plane
    if type(frame) is not SketchPlane:
        raise TypeError("plane must be a SketchPlane")
    values = _sketch_local_points(points, frame)
    values = values[np.all(np.isfinite(values), axis=1)]
    if len(values):
        minimum = np.min(values, axis=0)
        maximum = np.max(values, axis=0)
        center = 0.5 * (minimum + maximum)
        spans = maximum - minimum
    else:
        center = np.zeros(2, dtype=float)
        spans = np.zeros(2, dtype=float)
    minimum_half_extent = max(20.0 * clean_spacing, 1.0)
    padding = max(
        4.0 * clean_spacing,
        0.1 * float(np.max(spans)),
        0.05,
    )
    half_extents = np.maximum(
        0.5 * spans + padding,
        minimum_half_extent,
    )
    depth = max(float(np.max(half_extents)) * 1.0e-6, 1.0e-6)
    corners = np.asarray(
        tuple(
            frame.to_global(u, v)
            for u in (
                center[0] - half_extents[0],
                center[0] + half_extents[0],
            )
            for v in (
                center[1] - half_extents[1],
                center[1] + half_extents[1],
            )
        ),
        dtype=float,
    )
    normal = np.asarray(frame.normal, dtype=float)
    corners = np.vstack(
        (corners - depth * normal, corners + depth * normal)
    )
    minimum = np.min(corners, axis=0)
    maximum = np.max(corners, axis=0)
    return tuple(
        float(value)
        for pair in zip(minimum, maximum, strict=True)
        for value in pair
    )


def _sketch_shape_preview_points(
    mode: str,
    pending_points: Iterable[tuple[float, float, float]],
    cursor_point: tuple[float, float, float] | None,
    plane: SketchPlane | None = None,
) -> tuple[tuple[float, float, float], ...]:
    """Build the transient rectangle or circle polyline for a second click."""

    pending = tuple(
        tuple(float(value) for value in point)
        for point in pending_points
    )
    normalized_mode = str(mode).strip().casefold()
    if (
        normalized_mode not in {"rectangle", "circle"}
        or len(pending) != 1
        or cursor_point is None
    ):
        return ()
    frame = SketchPlane.xy() if plane is None else plane
    if type(frame) is not SketchPlane:
        raise TypeError("plane must be a SketchPlane")
    start = np.asarray(frame.to_local(pending[0]), dtype=float)
    cursor = np.asarray(frame.to_local(cursor_point), dtype=float)
    if (
        start.shape != (2,)
        or cursor.shape != (2,)
        or not np.all(np.isfinite(start))
        or not np.all(np.isfinite(cursor))
    ):
        return ()
    if normalized_mode == "rectangle":
        if np.allclose(start, cursor, rtol=0.0, atol=1.0e-12):
            return ()
        return (
            frame.to_global(float(start[0]), float(start[1])),
            frame.to_global(float(cursor[0]), float(start[1])),
            frame.to_global(float(cursor[0]), float(cursor[1])),
            frame.to_global(float(start[0]), float(cursor[1])),
            frame.to_global(float(start[0]), float(start[1])),
        )
    radius = float(np.linalg.norm(cursor - start))
    if radius <= 1.0e-12:
        return ()
    angles = np.linspace(0.0, 2.0 * math.pi, 65)
    return tuple(
        frame.to_global(
            float(start[0] + radius * math.cos(angle)),
            float(start[1] + radius * math.sin(angle)),
        )
        for angle in angles
    )


def _sketch_curve_sample_count(
    radius: float,
    sweep: float,
    world_per_pixel: float,
    *,
    chord_error_pixels: float = 0.75,
) -> int:
    """Choose an arc segment count whose projected chord error stays below 1 px."""

    clean_radius = float(radius)
    clean_sweep = abs(float(sweep))
    clean_world_per_pixel = float(world_per_pixel)
    if (
        not all(
            math.isfinite(value)
            for value in (
                clean_radius,
                clean_sweep,
                clean_world_per_pixel,
                chord_error_pixels,
            )
        )
        or clean_radius <= 0.0
        or clean_sweep <= 0.0
        or clean_world_per_pixel <= 0.0
        or chord_error_pixels <= 0.0
    ):
        raise ValueError("adaptive curve sampling requires positive finite values")
    relative_error = min(1.0, chord_error_pixels * clean_world_per_pixel / clean_radius)
    maximum_angle = 2.0 * math.acos(max(-1.0, 1.0 - relative_error))
    if maximum_angle <= 0.0:
        return 16384
    minimum = 8 if math.isclose(clean_sweep, math.tau, abs_tol=1.0e-12) else 2
    return min(16384, max(minimum, int(math.ceil(clean_sweep / maximum_angle))))


def _sketch_intersection_points(
    data: SketchDraftRenderData,
) -> tuple[tuple[float, float, float], ...]:
    """Return distinct pairwise curve intersections for transient snapping."""

    if data.analytic_curves:
        point_map = {point.id: point for point in data.analytic_points}
        result = sketch_intersections(data.analytic_curves, point_map)
        intersections: list[tuple[float, float]] = []
        for item in result.intersections:
            candidate = (item.u, item.v)
            if not any(
                math.hypot(
                    candidate[0] - existing[0],
                    candidate[1] - existing[1],
                )
                <= 1.0e-9
                for existing in intersections
            ):
                intersections.append(candidate)
        return tuple(data.plane.to_global(u, v) for u, v in intersections)

    local = _sketch_local_points(data.points, data.plane)
    intersections: list[tuple[float, float]] = []
    for left_index, left_curve in enumerate(data.curves):
        for right_curve in data.curves[left_index + 1 :]:
            for left_start, left_end in zip(left_curve, left_curve[1:]):
                p = local[left_start]
                r = local[left_end] - p
                for right_start, right_end in zip(right_curve, right_curve[1:]):
                    q = local[right_start]
                    s = local[right_end] - q
                    denominator = float(r[0] * s[1] - r[1] * s[0])
                    if abs(denominator) <= 1.0e-12:
                        continue
                    delta = q - p
                    left_parameter = float(
                        (delta[0] * s[1] - delta[1] * s[0]) / denominator
                    )
                    right_parameter = float(
                        (delta[0] * r[1] - delta[1] * r[0]) / denominator
                    )
                    if not (
                        -1.0e-12 <= left_parameter <= 1.0 + 1.0e-12
                        and -1.0e-12 <= right_parameter <= 1.0 + 1.0e-12
                    ):
                        continue
                    value = p + left_parameter * r
                    candidate = (float(value[0]), float(value[1]))
                    if not any(
                        math.hypot(candidate[0] - existing[0], candidate[1] - existing[1])
                        <= 1.0e-9
                        for existing in intersections
                    ):
                        intersections.append(candidate)
    return tuple(data.plane.to_global(u, v) for u, v in intersections)


def _sketch_intersection_curve_ids_at(
    data: SketchDraftRenderData,
    u: float,
    v: float,
    *,
    tolerance: float = 1.0e-8,
) -> tuple[str, ...]:
    """Return the stable analytic curve pair for a snapped intersection."""

    if not data.analytic_curves:
        return ()
    point_map = {point.id: point for point in data.analytic_points}
    result = sketch_intersections(data.analytic_curves, point_map)
    for item in result.intersections:
        if math.hypot(item.u - u, item.v - v) <= tolerance:
            return item.left_curve_id, item.right_curve_id
    return ()


def _capture_camera_state(plotter: object) -> _ViewportCameraState | None:
    """Capture framing before transient actors are rebuilt."""

    try:
        camera = plotter.camera
        return _ViewportCameraState(
            tuple(float(value) for value in camera.GetPosition()),
            tuple(float(value) for value in camera.GetFocalPoint()),
            tuple(float(value) for value in camera.GetViewUp()),
            float(camera.GetParallelScale()),
            float(camera.GetViewAngle()),
            int(camera.GetParallelProjection()),
        )
    except (AttributeError, TypeError, ValueError):
        return None


def _restore_camera_state(plotter: object, state: _ViewportCameraState) -> None:
    """Restore framing while allowing the clipping range to follow new actors."""

    camera = plotter.camera
    camera.SetPosition(*state.position)
    camera.SetFocalPoint(*state.focal_point)
    camera.SetViewUp(*state.view_up)
    camera.SetParallelScale(state.parallel_scale)
    camera.SetViewAngle(state.view_angle)
    camera.SetParallelProjection(state.parallel_projection)
    camera.OrthogonalizeViewUp()
    reset_clipping = getattr(plotter, "reset_camera_clipping_range", None)
    if reset_clipping is not None:
        reset_clipping()


def _require_result_render_payload(
    payload: object,
) -> ResultRenderPayload:
    """Validate the exact typed viewport boundary without eagerly loading VTK."""

    from ..visualization.result_renderer import (
        validate_result_render_payload,
    )

    return validate_result_render_payload(payload)


def _reuse_result_render_dataset(
    current: ResultRenderPayload,
    candidate: ResultRenderPayload,
    *,
    candidate_validated: bool = False,
) -> tuple[ResultRenderPayload, bool]:
    from ..visualization.result_renderer import (
        reuse_result_render_dataset,
    )

    return reuse_result_render_dataset(
        current,
        candidate,
        candidate_validated=candidate_validated,
    )


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


def _display_point_in_polygon(
    point: np.ndarray,
    polygon: np.ndarray,
) -> bool:
    """Return whether one display-space point lies in a polygon cell."""

    if len(polygon) < 3:
        return False
    x, y = float(point[0]), float(point[1])
    inside = False
    for index in range(len(polygon)):
        start = polygon[index]
        end = polygon[(index + 1) % len(polygon)]
        distance, _fraction = _point_to_segment_distance(
            np.asarray((x, y), dtype=float),
            np.asarray(start[:2], dtype=float),
            np.asarray(end[:2], dtype=float),
        )
        if distance <= 1.0:
            return True
        if (start[1] > y) != (end[1] > y):
            crossing_x = (
                start[0]
                + (y - start[1])
                * (end[0] - start[0])
                / (end[1] - start[1])
            )
            if crossing_x > x:
                inside = not inside
    return inside


class FEMViewport(QWidget):
    """维护网格、标注、选择、载荷与结果等独立 Actor。"""

    nativeSurfaceUpdated = Signal()
    entityPicked = Signal(str, int)
    geometryEntityPicked = Signal(object)
    geometryEntitiesBoxSelected = Signal(object)
    meshEntityPicked = Signal(object)
    meshEntitiesBoxSelected = Signal(object)
    selectionMissed = Signal(str)
    selectionConfirmed = Signal()
    selectionCancelled = Signal()
    wireWorkPlanePointSelected = Signal(object)
    wireDraftPointSelected = Signal(str)
    wireDraftMemberSelected = Signal(str)
    wireMemberStartSelected = Signal(str)
    wireMemberEndpointsSelected = Signal(str, str)
    wireAuthoringMissed = Signal(str)
    wirePendingInteractionCancelled = Signal()
    wireAuthoringFinishRequested = Signal()
    wireAuthoringCancelled = Signal()
    sketchWorkPlanePointSelected = Signal(object)
    sketchReferencePointSelected = Signal(object)
    sketchDraftPointSelected = Signal(str)
    sketchDraftCurveSelected = Signal(str)
    sketchDraftProfileSelected = Signal(str)
    sketchDraftPointSelectionRequested = Signal(str, object)
    sketchDraftCurveSelectionRequested = Signal(str, object)
    sketchDraftProfileSelectionRequested = Signal(str, object)
    sketchTrimRequested = Signal(str, object)
    sketchAuthoringMissed = Signal(str)
    sketchPendingInteractionCancelled = Signal()
    sketchAuthoringFinishRequested = Signal()
    sketchAuthoringCancelled = Signal()
    sketchInferencePreviewChanged = Signal(object)
    sketchSnapConfirmed = Signal(object)
    sketchPointDragPreviewRequested = Signal(str, object)
    sketchPointDragCommitRequested = Signal(str, object)
    sketchConstraintSelectionConfirmed = Signal()
    sketchConstraintSelectionCancelled = Signal()
    sketchDeleteRequested = Signal()
    sketchContextMenuRequested = Signal(str, str, object)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._background_settings = ViewportBackgroundSettings()
        self._plotter = None
        self._grid = None
        self._result_grid = None
        self._result_point_index_to_node_id: dict[int, int] = {}
        self._result_point_index_to_element_id: dict[int, int] = {}
        self._result_cell_index_to_element_id: dict[int, int] = {}
        self._result_provenance_layout: (
            tuple[object, object, object] | None
        ) = None
        self._model = None
        self._geometry: ModelGeometry | None = None
        self._geometry_preview: GeometryPreview | None = None
        self._geometry_ghost_preview: GeometryPreview | None = None
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
        self._geometry_face_body_pick_ids: tuple[int, ...] = ()
        self._geometry_edge_body_pick_ids: tuple[int, ...] = ()
        self._geometry_point_body_pick_ids: tuple[int, ...] = ()
        self._mesh_scope_edges = None
        self._mesh_scope_faces = None
        self._mesh_scope_edge_rows: tuple[
            tuple[int, int, tuple[int, ...]], ...
        ] = ()
        self._mesh_scope_face_rows: tuple[
            tuple[int, int, tuple[int, ...]], ...
        ] = ()
        self._mesh_scope_edge_cells: tuple[tuple[int, ...], ...] = ()
        self._mesh_scope_pick_to_ref: dict[
            tuple[str, int],
            MeshEntityRef,
        ] = {}
        self._mesh_scope_ref_to_pick_id: dict[MeshEntityRef, int] = {}
        self._mesh_scope_identity_to_pick_id: dict[
            tuple[str, tuple[int, int]],
            int,
        ] = {}
        self._mesh_scope_pick_bindings_ready = False
        self._mesh_scope_highlight_pipelines: dict[
            str,
            _MeshScopeHighlightPipeline,
        ] = {}
        self._mesh_scope_highlight_indices: dict[
            tuple[str, tuple[int, int]],
            tuple[int, ...],
        ] = {}
        self._mesh_scope_selected_references: set[MeshEntityRef] = set()
        self._mesh_scope_highlight_kind: str | None = None
        self._pick_grid = None
        self._pick_locators: dict[int, tuple[int, Any]] = {}
        self._result_render_payload: ResultRenderPayload | None = None
        self._scalar_reuse_pending = False
        self._scalar_reuse_display: DisplayState | None = None
        # Runtime revalidation follows VTK Modified notifications. Full
        # representation validation still runs unconditionally on install/render.
        self._result_render_validated_mtime: int | None = None
        self._result_install_validation: (
            tuple[ResultRenderPayload, int] | None
        ) = None
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
        self._symbol_sampling_density_override: str | None = None
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
        self._selection_rubber_band: _SelectionRubberBand | None = None
        self._picker_event_targets: set[QWidget] = set()
        self._abaqus_view_button: Qt.MouseButton | None = None
        self._trackball_vector: np.ndarray | None = None
        self._hover_hit: PickHit | None = None
        self._pending_hover_position: tuple[float, float] | None = None
        self._wire_authoring_active = False
        self._wire_authoring_mode = "point"
        self._wire_work_plane = "XY"
        self._wire_plane_offset = 0.0
        self._wire_grid_snap = True
        self._wire_grid_spacing = 0.1
        self._wire_draft_render_data: WireDraftRenderData | None = None
        self._wire_pending_member_start: str | None = None
        self._wire_authoring_selection: tuple[str, str] | None = None
        self._wire_authoring_hover: tuple[str, str] | None = None
        self._wire_authoring_preview_point: tuple[float, float, float] | None = None
        self._sketch_authoring_active = False
        self._sketch_authoring_mode = "polyline"
        self._sketch_grid_visible = True
        self._sketch_grid_snap = True
        self._sketch_grid_spacing = 0.1
        self._sketch_snap_sketch_points = True
        self._sketch_snap_external_points = True
        self._sketch_snap_midpoints = True
        self._sketch_snap_centers = True
        self._sketch_snap_intersections = True
        self._sketch_screen_snap_tolerance = 9.0
        self._sketch_show_point_ids = True
        self._sketch_show_external_labels = True
        self._sketch_show_profile_fill = True
        self._sketch_show_work_plane_axes = True
        self._sketch_draft_render_data: SketchDraftRenderData | None = None
        self._sketch_intersection_revision: int | None = None
        self._sketch_intersection_cache: tuple[
            tuple[float, float, float], ...
        ] = ()
        self._sketch_reference_points: tuple[SketchReferencePoint, ...] = ()
        self._sketch_authoring_snap_reference: SketchReferencePoint | None = None
        self._sketch_authoring_snap_kind: str | None = None
        self._sketch_authoring_snap_point_id: str | None = None
        self._sketch_authoring_intersection_curve_ids: tuple[str, ...] = ()
        self._sketch_drag_point_id: str | None = None
        self._sketch_drag_moved = False
        self._sketch_pending_points: tuple[tuple[float, float, float], ...] = ()
        self._sketch_authoring_preview_point: (
            tuple[float, float, float] | None
        ) = None
        self._sketch_hover_entity: tuple[str, str] | None = None
        self._sketch_constraint_selection_active = False
        self._sketch_constraint_selection: tuple[tuple[str, str], ...] = ()
        self._hover_timer = QTimer(self)
        self._hover_timer.setSingleShot(True)
        self._hover_timer.setInterval(40)
        self._hover_timer.timeout.connect(self._update_preselection)
        self._mesh_scope_render_timer = QTimer(self)
        self._mesh_scope_render_timer.setSingleShot(True)
        self._mesh_scope_render_timer.setInterval(0)
        self._mesh_scope_render_timer.timeout.connect(
            self._render_mesh_scope_highlight
        )
        self._contour = {
            "manual": False, "minimum": 0.0, "maximum": 1.0, "levels": 12,
            "colormap": ABAQUS_RAINBOW, "style": "segmented",
            "legend": True, "edges": True,
            "render_mode": CONTOUR_RENDER_SHADED,
            "edge_mode": CONTOUR_EDGE_GEOMETRY,
            "edge_style": "solid", "edge_width": 1.0,
            "number_format": "scientific", "decimals": 2,
            "orientation": "vertical", "show_minimum": False,
            "show_maximum": False, "show_ids": False,
            "legend_font": "Arial", "legend_font_size": 14,
            "show_coordinate_system": True,
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
            if self._sketch_authoring_active:
                if self._sketch_constraint_selection_active:
                    if event.key() in {Qt.Key.Key_Return, Qt.Key.Key_Enter}:
                        self.sketchConstraintSelectionConfirmed.emit()
                        return True
                    if event.key() == Qt.Key.Key_Escape:
                        self.sketchConstraintSelectionCancelled.emit()
                        return True
                if (
                    event.key() == Qt.Key.Key_Delete
                    and self._sketch_authoring_mode == "select"
                ):
                    self.sketchDeleteRequested.emit()
                    return True
                if event.key() in {Qt.Key.Key_Return, Qt.Key.Key_Enter}:
                    if self.cancel_pending_sketch_interaction():
                        return True
                    self.sketchAuthoringFinishRequested.emit()
                    return True
                if event.key() == Qt.Key.Key_Escape:
                    if not self.cancel_pending_sketch_interaction():
                        self.sketchAuthoringCancelled.emit()
                    return True
            if self._wire_authoring_active:
                if event.key() in {Qt.Key.Key_Return, Qt.Key.Key_Enter}:
                    self.wireAuthoringFinishRequested.emit()
                    return True
                if event.key() == Qt.Key.Key_Escape:
                    if not self.cancel_pending_wire_interaction():
                        self.wireAuthoringCancelled.emit()
                    return True
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
            if self._sketch_authoring_active:
                if (
                    button == Qt.MouseButton.RightButton
                    and self._sketch_authoring_mode == "select"
                    and not self._sketch_constraint_selection_active
                ):
                    position = self._plotter_event_position(watched, event)
                    vtk_x, vtk_y = self._qt_to_vtk_position(
                        position.x(), position.y()
                    )
                    point_id = self._sketch_point_at(vtk_x, vtk_y)
                    curve_id = (
                        None
                        if point_id is not None
                        else self._sketch_curve_at(vtk_x, vtk_y)
                    )
                    entity_id = point_id or curve_id
                    if entity_id is not None:
                        self.sketchContextMenuRequested.emit(
                            "point" if point_id is not None else "curve",
                            entity_id,
                            event.globalPosition().toPoint(),
                        )
                        return True
                if button in {
                    Qt.MouseButton.MiddleButton,
                    Qt.MouseButton.RightButton,
                }:
                    self.cancel_pending_sketch_interaction()
                    return True
                if button == Qt.MouseButton.LeftButton:
                    position = self._plotter_event_position(watched, event)
                    vtk_x, vtk_y = self._qt_to_vtk_position(position.x(), position.y())
                    self._sketch_drag_point_id = (
                        self._sketch_point_at(vtk_x, vtk_y)
                        if self._sketch_authoring_mode == "select"
                        and not self._sketch_constraint_selection_active
                        else None
                    )
                    self._sketch_drag_moved = False
                    self._selection_press_position = (position.x(), position.y())
                    return True
            if self._wire_authoring_active:
                if button in {
                    Qt.MouseButton.MiddleButton,
                    Qt.MouseButton.RightButton,
                }:
                    self.cancel_pending_wire_interaction()
                    return True
                if button == Qt.MouseButton.LeftButton:
                    self._selection_press_position = None
                    return True
            if button == Qt.MouseButton.LeftButton:
                self._pending_hover_position = None
                self._hover_timer.stop()
                position = self._plotter_event_position(watched, event)
                self._selection_press_position = (position.x(), position.y())
                self._selection_dragged = False
                self._hide_selection_rubber_band()
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
            if self._sketch_authoring_active:
                position = self._plotter_event_position(watched, event)
                vtk_x, vtk_y = self._qt_to_vtk_position(
                    position.x(),
                    position.y(),
                )
                if self._sketch_authoring_mode in {"select", "trim"}:
                    point_id = (
                        self._sketch_point_at(vtk_x, vtk_y)
                        if self._sketch_authoring_mode == "select"
                        else None
                    )
                    curve_id = (
                        None
                        if point_id is not None
                        else self._sketch_curve_at(vtk_x, vtk_y)
                    )
                    self._set_sketch_entity_hover(
                        "point" if point_id is not None else "curve",
                        point_id or curve_id,
                    )
                    self._set_sketch_authoring_preview_point(None)
                    return True
                point, _reason = self._sketch_work_plane_point_at(
                    vtk_x,
                    vtk_y,
                    snap=self._sketch_authoring_mode != "trim",
                )
                if self._sketch_drag_point_id is not None:
                    start = self._selection_press_position
                    if start is not None and (
                        abs(position.x() - start[0]) + abs(position.y() - start[1]) > 2.0
                    ):
                        self._sketch_drag_moved = True
                    if point is not None and self._sketch_drag_moved:
                        self.sketchPointDragPreviewRequested.emit(
                            self._sketch_drag_point_id, point
                        )
                    return True
                self._set_sketch_authoring_preview_point(point)
                self.sketchInferencePreviewChanged.emit(
                    {
                        "point": point,
                        "snap_kind": self._sketch_authoring_snap_kind,
                        "point_id": self._sketch_authoring_snap_point_id,
                        "curve_ids": self._sketch_authoring_intersection_curve_ids,
                    }
                )
                return True
            if self._wire_authoring_active:
                position = self._plotter_event_position(watched, event)
                vtk_x, vtk_y = self._qt_to_vtk_position(
                    position.x(),
                    position.y(),
                )
                self._update_wire_authoring_hover(vtk_x, vtk_y)
                return True
            if self._selection_press_position is not None:
                position = self._plotter_event_position(watched, event)
                start_x, start_y = self._selection_press_position
                if abs(position.x() - start_x) + abs(position.y() - start_y) > 4.0:
                    self._selection_dragged = True
                    self._clear_preselection(render=True)
                if self._selection_dragged:
                    self._show_selection_rubber_band(
                        (start_x, start_y),
                        (position.x(), position.y()),
                    )
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
            if self._sketch_authoring_active:
                if button == Qt.MouseButton.LeftButton:
                    position = self._plotter_event_position(watched, event)
                    vtk_x, vtk_y = self._qt_to_vtk_position(
                        position.x(), position.y()
                    )
                    if self._sketch_drag_point_id is not None and self._sketch_drag_moved:
                        point, _reason = self._sketch_work_plane_point_at(vtk_x, vtk_y)
                        if point is not None:
                            self.sketchPointDragCommitRequested.emit(
                                self._sketch_drag_point_id, point
                            )
                        self._sketch_drag_point_id = None
                        self._sketch_drag_moved = False
                        self._selection_press_position = None
                        return True
                    self._sketch_drag_point_id = None
                    self._selection_press_position = None
                    self._sketch_authoring_click(
                        vtk_x,
                        vtk_y,
                        modifiers=event.modifiers(),
                    )
                    return True
                if button in {
                    Qt.MouseButton.MiddleButton,
                    Qt.MouseButton.RightButton,
                }:
                    self.cancel_pending_sketch_interaction()
                    return True
            if self._wire_authoring_active:
                if button == Qt.MouseButton.LeftButton:
                    position = self._plotter_event_position(watched, event)
                    vtk_x, vtk_y = self._qt_to_vtk_position(
                        position.x(), position.y()
                    )
                    self._wire_authoring_click(vtk_x, vtk_y)
                    return True
                if button in {
                    Qt.MouseButton.MiddleButton,
                    Qt.MouseButton.RightButton,
                }:
                    self.cancel_pending_wire_interaction()
                    return True
            if button == Qt.MouseButton.LeftButton and self._selection_press_position is not None:
                position = self._plotter_event_position(watched, event)
                start = self._selection_press_position
                should_pick = not self._selection_dragged
                self._selection_press_position = None
                self._selection_dragged = False
                self._hide_selection_rubber_band()
                if should_pick:
                    self._pick_qt_position(position.x(), position.y())
                else:
                    self._box_select_qt_positions(
                        start,
                        (position.x(), position.y()),
                    )
                return True
            if button in {Qt.MouseButton.MiddleButton, Qt.MouseButton.RightButton}:
                return True

        if event_type == QEvent.Type.Leave:
            self._pending_hover_position = None
            self._hover_timer.stop()
            if self._sketch_authoring_active:
                self._set_sketch_authoring_preview_point(None)
                self._set_sketch_entity_hover(None, None)
            if self._wire_authoring_active:
                self._set_wire_authoring_hover(None)
            self._clear_preselection(render=True)

        return super().eventFilter(watched, event)

    def _show_selection_rubber_band(
        self,
        start: tuple[float, float],
        end: tuple[float, float],
    ) -> None:
        if self._plotter is None:
            return
        if self._selection_rubber_band is None:
            renderer = getattr(self._plotter, "renderer", None)
            if renderer is None:
                return
            self._selection_rubber_band = _SelectionRubberBand(
                renderer
            )
        self._selection_rubber_band.set_containment(
            float(end[0]) >= float(start[0])
        )
        self._selection_rubber_band.set_rectangle(
            self._qt_to_vtk_position(*start),
            self._qt_to_vtk_position(*end),
        )
        self._selection_rubber_band.show()
        self._render()

    def _hide_selection_rubber_band(self) -> None:
        if (
            self._selection_rubber_band is not None
            and self._selection_rubber_band.hide()
        ):
            self._render()

    def _box_select_qt_positions(
        self,
        start: tuple[float, float],
        end: tuple[float, float],
    ) -> None:
        if self._selection_mode == "mesh_node":
            self.meshEntitiesBoxSelected.emit(
                self._mesh_nodes_in_qt_rectangle(start, end)
            )
            return
        if self._selection_mode in {
            "mesh_element",
            "mesh_edge",
            "mesh_face",
            "mesh_body",
        }:
            self.meshEntitiesBoxSelected.emit(
                self._mesh_entities_in_qt_rectangle(start, end)
            )
            return
        if self._selection_mode == "geometry_point":
            self.geometryEntitiesBoxSelected.emit(
                self._geometry_points_in_qt_rectangle(start, end)
            )
            return
        if self._selection_mode in {
            "geometry_edge",
            "geometry_face",
            "geometry_body",
        }:
            self.geometryEntitiesBoxSelected.emit(
                self._geometry_entities_in_qt_rectangle(start, end)
            )

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
        show_edges: bool | None = None,
        show_nodes: bool | None = None,
        show_node_labels: bool | None = None,
        show_element_labels: bool | None = None,
        effective_frame_query: (
            Callable[[RegionRef | int], Any] | None
        ) = None,
    ) -> None:
        if show_edges is not None:
            self._show_edges = bool(show_edges)
        if show_nodes is not None:
            self._show_nodes = bool(show_nodes)
        if show_node_labels is not None:
            self._show_node_labels = bool(show_node_labels)
        if show_element_labels is not None:
            self._show_element_labels = bool(show_element_labels)
        self._model = model
        self._geometry = geometry
        self._artifact_id = geometry.artifact_id
        self._run_id = None
        self._geometry_preview = None
        self._geometry_ghost_preview = None
        self._geometry_preview_surface = None
        self._geometry_preview_edges = None
        self._geometry_preview_points = None
        self._geometry_pick_to_ref.clear()
        self._geometry_ref_to_pick_ids.clear()
        self._geometry_face_pick_ids = ()
        self._geometry_edge_pick_ids = ()
        self._geometry_point_pick_ids = ()
        self._geometry_body_pick_id = 0
        self._geometry_face_body_pick_ids = ()
        self._geometry_edge_body_pick_ids = ()
        self._geometry_point_body_pick_ids = ()
        self._mesh_scope_selected_references.clear()
        self._reset_mesh_scope_highlight_pipelines()
        self._clear_mesh_scope_pick_bindings()
        self._pick_grid = None
        self._pick_locators.clear()
        self._result_render_payload = None
        self._scalar_reuse_pending = False
        self._scalar_reuse_display = None
        self._result_provenance_layout = None
        self._result_render_validated_mtime = None
        self._result_install_validation = None
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
        self._install_mesh_scope_pick_bindings()
        self._install_mesh_scope_highlight_pipelines()
        if refresh_symbols:
            self.show_boundary_and_loads(render=render)
        elif render:
            self._render()

    def rebind_model_artifact(
        self,
        model: Any,
        geometry: ModelGeometry,
        *,
        effective_frame_query: (
            Callable[[RegionRef | int], Any] | None
        ) = None,
    ) -> None:
        """Rebind unchanged mesh actors to a definition-only model artifact."""

        if self._geometry is None:
            raise RuntimeError("cannot rebind a viewport without a model")
        self._model = model
        self._geometry = geometry
        self._artifact_id = geometry.artifact_id
        self._run_id = None
        self._effective_frame_query = effective_frame_query
        self._boundary_cache.clear()
        self._beam_frame_cache.clear()

    def clear_model(self) -> None:
        if self._sketch_authoring_active:
            self.stop_sketch_authoring(render=False)
        if self._wire_authoring_active:
            self.stop_wire_authoring(render=False)
        self._model = None
        self._geometry = None
        self._artifact_id = None
        self._run_id = None
        self._geometry_preview = None
        self._geometry_ghost_preview = None
        self._geometry_preview_surface = None
        self._geometry_preview_edges = None
        self._geometry_preview_points = None
        self._geometry_pick_to_ref.clear()
        self._geometry_ref_to_pick_ids.clear()
        self._geometry_face_pick_ids = ()
        self._geometry_edge_pick_ids = ()
        self._geometry_point_pick_ids = ()
        self._geometry_body_pick_id = 0
        self._geometry_face_body_pick_ids = ()
        self._geometry_edge_body_pick_ids = ()
        self._geometry_point_body_pick_ids = ()
        self._mesh_scope_selected_references.clear()
        self._reset_mesh_scope_highlight_pipelines()
        self._clear_mesh_scope_pick_bindings()
        self._pick_grid = None
        self._pick_locators.clear()
        self._result_render_payload = None
        self._scalar_reuse_pending = False
        self._scalar_reuse_display = None
        self._result_provenance_layout = None
        self._result_render_validated_mtime = None
        self._result_install_validation = None
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

    @property
    def wire_authoring_active(self) -> bool:
        return self._wire_authoring_active

    @property
    def sketch_authoring_active(self) -> bool:
        return self._sketch_authoring_active

    def start_sketch_authoring(
        self,
        render_data: SketchDraftRenderData,
        *,
        snap: bool = True,
        spacing: float = 0.1,
        reference_points: tuple[SketchReferencePoint, ...] = (),
    ) -> None:
        """Enter transient planar sketch authoring without touching Session."""

        if type(render_data) is not SketchDraftRenderData:
            raise TypeError("render_data must be a SketchDraftRenderData")
        if not np.isfinite(float(spacing)) or float(spacing) <= 0.0:
            raise ValueError("sketch grid spacing must be positive")
        if self._wire_authoring_active:
            self.stop_wire_authoring()
        self._sketch_authoring_active = True
        self._sketch_authoring_mode = "polyline"
        self._sketch_grid_snap = bool(snap)
        self._sketch_grid_spacing = float(spacing)
        self._sketch_draft_render_data = render_data
        self._sketch_intersection_revision = None
        self.set_sketch_reference_points(reference_points, render=False)
        self._sketch_pending_points = ()
        self._sketch_authoring_preview_point = None
        self._sketch_hover_entity = None
        self._sketch_constraint_selection_active = False
        self._sketch_constraint_selection = ()
        self._clear_preselection(render=False)
        self._remove_all_layers(render=False)
        self._show_sketch_draft(render=False, reset_camera=True)

    def set_sketch_reference_points(
        self,
        reference_points: tuple[SketchReferencePoint, ...],
        *,
        render: bool = True,
    ) -> None:
        values = tuple(reference_points)
        if any(type(item) is not SketchReferencePoint for item in values):
            raise TypeError("reference_points must contain SketchReferencePoint values")
        self._sketch_reference_points = values
        if render and self._sketch_authoring_active:
            self._show_sketch_draft(render=True)

    def update_sketch_draft(
        self,
        render_data: SketchDraftRenderData,
    ) -> None:
        if type(render_data) is not SketchDraftRenderData:
            raise TypeError("render_data must be a SketchDraftRenderData")
        if not self._sketch_authoring_active:
            return
        self._sketch_draft_render_data = render_data
        self._show_sketch_draft(render=True)

    def update_sketch_selection(
        self,
        kind: str | None,
        entity_ids: Iterable[str] = (),
        *,
        render: bool = True,
    ) -> None:
        """Update only the transient sketch highlight actor."""

        data = self._sketch_draft_render_data
        if data is None:
            return
        ids = tuple(dict.fromkeys(str(value) for value in entity_ids))
        normalized_kind = None if not ids else str(kind)
        if normalized_kind == "edge":
            normalized_kind = "curve"
        if normalized_kind not in {None, "point", "curve", "profile"}:
            raise ValueError("invalid sketch draft selection kind")
        self._sketch_draft_render_data = replace(
            data,
            selected_kind=normalized_kind,
            selected_id=ids[0] if ids else None,
            selected_ids=ids,
        )
        self._show_sketch_selection(render=render)

    def show_sketch_reference_preview(
        self,
        preview: GeometryPreview,
        *,
        support_face_id: str | None = None,
        target_body_id: str | None = None,
        render: bool = True,
    ) -> None:
        """Show the target Body translucently and emphasize its support Face."""

        if type(preview) is not GeometryPreview:
            raise TypeError("preview must be a GeometryPreview")
        if not self._sketch_authoring_active:
            raise RuntimeError("sketch authoring must be active")
        self._remove_actor("sketch_reference_surface")
        self._remove_actor("sketch_support_face_highlight")
        if is_offscreen_environment():
            return
        if not preview.faces or not self._ensure_plotter() or _pyvista is None:
            return
        has_body_ids = any(
            body_id is not None
            for body_id in preview.face_body_logical_ids
        )
        face_indices = tuple(
            index
            for index, body_id in enumerate(
                preview.face_body_logical_ids
            )
            if (
                target_body_id is None
                or not has_body_ids
                or body_id == target_body_id
            )
        )
        if not face_indices:
            return
        surface = _pyvista.PolyData()
        surface.points = np.asarray(preview.points, dtype=float)
        surface.faces = np.hstack(
            tuple(
                (len(preview.faces[index]), *preview.faces[index])
                for index in face_indices
            )
        ).astype(np.int64)
        actor = self._plotter.add_mesh(
            surface,
            color="#7194ab",
            opacity=0.32,
            smooth_shading=False,
            show_edges=True,
            edge_color="#527084",
            line_width=1,
            name="sketch_reference_surface",
            reset_camera=False,
        )
        actor.SetPickable(False)
        self._actors["sketch_reference_surface"] = actor
        if support_face_id is not None:
            support_faces = tuple(
                preview.faces[index]
                for index in face_indices
                if preview.face_logical_ids[index] == support_face_id
            )
            if support_faces:
                highlight = _pyvista.PolyData()
                highlight.points = np.asarray(
                    preview.points,
                    dtype=float,
                )
                highlight.faces = np.hstack(
                    tuple(
                        (len(face), *face)
                        for face in support_faces
                    )
                ).astype(np.int64)
                support_actor = self._plotter.add_mesh(
                    highlight,
                    color="#f5a623",
                    opacity=0.52,
                    show_edges=True,
                    edge_color="#ff8c00",
                    line_width=4,
                    name="sketch_support_face_highlight",
                    reset_camera=False,
                    pickable=False,
                )
                support_actor.SetPickable(False)
                self._actors["sketch_support_face_highlight"] = (
                    support_actor
                )
        if render:
            self._render()

    def show_face_sketch_boolean_preview(
        self,
        target: GeometryPreview,
        display: FaceSketchBooleanDisplay,
        exact_result: GeometryPreview,
        *,
        target_body_id: str,
        origin: tuple[float, float, float],
        direction: tuple[float, float, float],
        distance: float,
        operation_name: str,
    ) -> None:
        """Show all detached layers for the current exact preview generation."""

        if type(target) is not GeometryPreview:
            raise TypeError("target must be a GeometryPreview")
        if type(display) is not FaceSketchBooleanDisplay:
            raise TypeError("display must be FaceSketchBooleanDisplay")
        if type(exact_result) is not GeometryPreview:
            raise TypeError("exact_result must be a GeometryPreview")
        for name in (
            "face_boolean_target",
            "face_boolean_unselected_profiles",
            "face_boolean_selected_profiles",
            "face_boolean_tool",
            "face_boolean_exact_result",
            "face_boolean_direction",
            "face_boolean_distance",
        ):
            self._remove_actor(name)
        if is_offscreen_environment():
            self._message.setText(
                f"拉伸布尔精确预览有效（{operation_name}，距离 {distance:g}）"
            )
            self._stack.setCurrentWidget(self._message)
            return
        if not self._ensure_plotter() or _pyvista is None:
            return

        def surface(
            name: str,
            preview: GeometryPreview,
            color: str,
            opacity: float,
            *,
            face_indices: tuple[int, ...] | None = None,
            show_edges: bool = False,
        ) -> None:
            indices = (
                tuple(range(len(preview.faces)))
                if face_indices is None
                else face_indices
            )
            if not indices:
                return
            mesh = _pyvista.PolyData()
            mesh.points = np.asarray(preview.points, dtype=float)
            mesh.faces = np.hstack(
                tuple(
                    (len(preview.faces[index]), *preview.faces[index])
                    for index in indices
                )
            ).astype(np.int64)
            actor = self._plotter.add_mesh(
                mesh,
                color=color,
                opacity=opacity,
                show_edges=show_edges,
                edge_color=color,
                line_width=2,
                name=name,
                reset_camera=False,
                pickable=False,
            )
            actor.SetPickable(False)
            self._actors[name] = actor

        has_body_ids = any(
            value is not None for value in target.face_body_logical_ids
        )
        target_indices = tuple(
            index
            for index, body_id in enumerate(target.face_body_logical_ids)
            if not has_body_ids or body_id == target_body_id
        )
        surface(
            "face_boolean_target",
            target,
            "#71808e",
            0.18,
            face_indices=target_indices,
            show_edges=True,
        )
        surface(
            "face_boolean_unselected_profiles",
            display.unselected_profiles,
            "#8c96a0",
            0.45,
            show_edges=True,
        )
        surface(
            "face_boolean_selected_profiles",
            display.selected_profiles,
            "#25a7d9",
            0.72,
            show_edges=True,
        )
        surface(
            "face_boolean_tool",
            display.tool,
            "#36a269" if operation_name == "合并材料" else "#d9544d",
            0.28,
            show_edges=True,
        )
        surface(
            "face_boolean_exact_result",
            exact_result,
            "#4f81bd",
            0.62,
            show_edges=True,
        )
        arrow = self._plotter.add_arrows(
            np.asarray((origin,), dtype=float),
            np.asarray((direction,), dtype=float),
            mag=float(distance),
            color="#ff9800",
            name="face_boolean_direction",
            reset_camera=False,
        )
        arrow.SetPickable(False)
        self._actors["face_boolean_direction"] = arrow
        endpoint = np.asarray(origin, dtype=float) + float(distance) * np.asarray(
            direction,
            dtype=float,
        )
        self._actors["face_boolean_distance"] = self._plotter.add_point_labels(
            np.asarray((endpoint,), dtype=float),
            (f"距离：{distance:g}",),
            point_size=0,
            font_size=12,
            shape=None,
            text_color="#ff9800",
            name="face_boolean_distance",
            reset_camera=False,
        )
        self._render()

    def set_sketch_authoring_mode(self, mode: str) -> None:
        normalized = str(mode).strip().casefold()
        if normalized not in {
            "select",
            "polyline",
            "rectangle",
            "circle",
            "arc",
            "trim",
        }:
            raise ValueError("unsupported sketch authoring mode")
        self._sketch_authoring_mode = normalized
        self._set_sketch_authoring_preview_point(None)
        self._set_sketch_entity_hover(None, None)
        self.cancel_pending_sketch_interaction()

    def set_sketch_grid(
        self,
        *,
        visible: bool | None = None,
        snap: bool | None = None,
        spacing: float | None = None,
    ) -> None:
        if spacing is not None and (
            not np.isfinite(float(spacing)) or float(spacing) <= 0.0
        ):
            raise ValueError("sketch grid spacing must be positive")
        if visible is not None:
            self._sketch_grid_visible = bool(visible)
        if snap is not None:
            self._sketch_grid_snap = bool(snap)
        if spacing is not None:
            self._sketch_grid_spacing = float(spacing)
        if self._sketch_authoring_active:
            self._show_sketch_draft(render=True)

    def set_sketch_preferences(
        self,
        *,
        snap_sketch_points: bool,
        snap_external_points: bool,
        snap_midpoints: bool,
        snap_centers: bool,
        snap_intersections: bool,
        screen_snap_tolerance: float,
        show_point_ids: bool,
        show_external_labels: bool,
        show_profile_fill: bool,
        show_work_plane_axes: bool,
    ) -> None:
        tolerance = float(screen_snap_tolerance)
        if not np.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("sketch screen snap tolerance must be non-negative")
        self._sketch_snap_sketch_points = bool(snap_sketch_points)
        self._sketch_snap_external_points = bool(snap_external_points)
        self._sketch_snap_midpoints = bool(snap_midpoints)
        self._sketch_snap_centers = bool(snap_centers)
        self._sketch_snap_intersections = bool(snap_intersections)
        self._sketch_screen_snap_tolerance = tolerance
        self._sketch_show_point_ids = bool(show_point_ids)
        self._sketch_show_external_labels = bool(show_external_labels)
        self._sketch_show_profile_fill = bool(show_profile_fill)
        self._sketch_show_work_plane_axes = bool(show_work_plane_axes)
        if self._sketch_authoring_active:
            self._show_sketch_draft(render=True)

    def set_sketch_pending_points(
        self,
        points: Iterable[tuple[float, float, float]],
    ) -> None:
        normalized = tuple(
            tuple(float(value) for value in point)
            for point in points
        )
        if normalized == self._sketch_pending_points:
            return
        self._sketch_pending_points = normalized
        if self._sketch_authoring_active:
            self._show_sketch_authoring_hover(render=True)

    def begin_sketch_constraint_selection(self) -> None:
        self._sketch_constraint_selection_active = True
        self._sketch_constraint_selection = ()
        self._show_sketch_constraint_selection(render=True)

    def set_sketch_constraint_selection(
        self,
        targets: Iterable[tuple[str, str]],
    ) -> None:
        normalized = tuple(
            ("curve" if kind == "edge" else str(kind), str(entity_id))
            for kind, entity_id in targets
        )
        if normalized == self._sketch_constraint_selection:
            return
        self._sketch_constraint_selection = normalized
        if self._sketch_authoring_active:
            self._show_sketch_constraint_selection(render=True)

    def end_sketch_constraint_selection(self) -> None:
        self._sketch_constraint_selection_active = False
        self._sketch_constraint_selection = ()
        self._show_sketch_constraint_selection(render=True)

    def set_sketch_inference_preview(self, kinds: Iterable[str]) -> None:
        """Update only transient inference labels, without rebuilding geometry."""

        normalized = tuple(str(kind) for kind in kinds)
        data = self._sketch_draft_render_data
        if data is None or normalized == data.inference_preview:
            return
        self._sketch_draft_render_data = replace(data, inference_preview=normalized)
        if self._sketch_authoring_active:
            self._show_sketch_authoring_hover(render=True)

    def cancel_pending_sketch_interaction(self) -> bool:
        if self._sketch_constraint_selection_active:
            self.sketchConstraintSelectionCancelled.emit()
            return True
        if self._sketch_drag_point_id is not None:
            self._sketch_drag_point_id = None
            self._sketch_drag_moved = False
            self._selection_press_position = None
            self.sketchPendingInteractionCancelled.emit()
            return True
        if not self._sketch_pending_points:
            return False
        self.set_sketch_pending_points(())
        self.sketchPendingInteractionCancelled.emit()
        return True

    def stop_sketch_authoring(self, *, render: bool = True) -> None:
        if (
            not self._sketch_authoring_active
            and self._sketch_draft_render_data is None
        ):
            return
        for name in (
            "sketch_work_plane_grid",
            "sketch_work_plane_axis_0",
            "sketch_work_plane_axis_1",
            "sketch_work_plane_origin",
            "sketch_work_plane_axis_labels",
            "sketch_draft_faces",
            "sketch_draft_curves",
            "sketch_draft_points",
            "sketch_draft_point_labels",
            "sketch_reference_points",
            "sketch_reference_labels",
            "sketch_authoring_selection",
            "sketch_authoring_shape_preview",
            "sketch_authoring_hover_outline",
            "sketch_authoring_hover",
            "sketch_authoring_hover_label",
            "sketch_entity_hover",
            "sketch_constraint_selection_points",
            "sketch_constraint_selection_curves",
            "sketch_constraint_overlays",
            "sketch_reference_surface",
            "sketch_support_face_highlight",
        ):
            self._remove_actor(name)
        self._sketch_authoring_active = False
        self._sketch_draft_render_data = None
        self._sketch_intersection_revision = None
        self._sketch_intersection_cache = ()
        self._sketch_reference_points = ()
        self._sketch_authoring_snap_reference = None
        self._sketch_authoring_snap_kind = None
        self._sketch_authoring_snap_point_id = None
        self._sketch_authoring_intersection_curve_ids = ()
        self._sketch_drag_point_id = None
        self._sketch_drag_moved = False
        self._sketch_pending_points = ()
        self._sketch_authoring_preview_point = None
        self._sketch_hover_entity = None
        self._sketch_constraint_selection_active = False
        self._sketch_constraint_selection = ()
        self._clear_preselection(render=False)
        if render:
            self._render()

    def focus_sketch_draft_entity(self, kind: str, entity_id: str) -> None:
        data = self._sketch_draft_render_data
        if data is None:
            return
        normalized_kind = "curve" if kind == "edge" else str(kind)
        self.update_sketch_selection(normalized_kind, (entity_id,), render=False)
        if self._plotter is not None:
            coordinates = self._sketch_entity_coordinates(
                self._sketch_draft_render_data,
                normalized_kind,
                str(entity_id),
            )
            if coordinates is not None and len(coordinates):
                center = np.mean(coordinates, axis=0)
                camera = self._plotter.camera
                old_focal = np.asarray(camera.focal_point, dtype=float)
                old_position = np.asarray(camera.position, dtype=float)
                camera.SetFocalPoint(*center)
                camera.SetPosition(*(old_position + center - old_focal))
                if camera.GetParallelProjection():
                    local = _sketch_local_points(
                        tuple(tuple(value) for value in coordinates),
                        data.plane,
                    )
                    if len(local) > 1:
                        span = max(
                            float(np.ptp(local[:, 0])),
                            float(np.ptp(local[:, 1])),
                            self._sketch_grid_spacing,
                        )
                        camera.SetParallelScale(max(span * 0.75, self._sketch_grid_spacing))
        self._render()

    def _show_sketch_selection(self, *, render: bool) -> None:
        self._remove_actor("sketch_authoring_selection")
        data = self._sketch_draft_render_data
        if (
            data is not None
            and not is_offscreen_environment()
            and self._ensure_plotter()
            and _pyvista is not None
        ):
            coordinates = np.asarray(data.points, dtype=float).reshape((-1, 3))
            selection = self._sketch_selection_polydata(data, coordinates)
            if selection is not None:
                self._actors["sketch_authoring_selection"] = self._plotter.add_mesh(
                    selection,
                    color="#ffb300",
                    line_width=8,
                    point_size=14,
                    render_points_as_spheres=True,
                    render_lines_as_tubes=True,
                    opacity=1.0,
                    name="sketch_authoring_selection",
                    reset_camera=False,
                    pickable=False,
                )
        if render:
            self._render()

    def _set_sketch_entity_hover(
        self,
        kind: str | None,
        entity_id: str | None,
    ) -> None:
        normalized = (
            None
            if kind is None or entity_id is None
            else ("curve" if kind == "edge" else str(kind), str(entity_id))
        )
        if normalized == self._sketch_hover_entity:
            return
        self._sketch_hover_entity = normalized
        self._show_sketch_entity_hover(render=True)

    def _show_sketch_entity_hover(self, *, render: bool) -> None:
        self._remove_actor("sketch_entity_hover")
        data = self._sketch_draft_render_data
        if (
            data is not None
            and self._sketch_hover_entity is not None
            and not is_offscreen_environment()
            and self._ensure_plotter()
            and _pyvista is not None
        ):
            kind, entity_id = self._sketch_hover_entity
            hover_data = replace(
                data,
                selected_kind=kind,
                selected_id=entity_id,
                selected_ids=(entity_id,),
            )
            coordinates = np.asarray(data.points, dtype=float).reshape((-1, 3))
            geometry = self._sketch_selection_polydata(hover_data, coordinates)
            if geometry is not None:
                self._actors["sketch_entity_hover"] = self._plotter.add_mesh(
                    geometry,
                    color="#ffd54f",
                    line_width=6,
                    point_size=12,
                    render_points_as_spheres=True,
                    render_lines_as_tubes=True,
                    opacity=0.95,
                    name="sketch_entity_hover",
                    reset_camera=False,
                    pickable=False,
                )
        if render:
            self._render()

    def _show_sketch_constraint_selection(self, *, render: bool) -> None:
        actor_names = {
            "point": "sketch_constraint_selection_points",
            "curve": "sketch_constraint_selection_curves",
        }
        for actor_name in actor_names.values():
            self._remove_actor(actor_name)
        data = self._sketch_draft_render_data
        if (
            data is not None
            and self._sketch_constraint_selection
            and not is_offscreen_environment()
            and self._ensure_plotter()
            and _pyvista is not None
        ):
            coordinates = np.asarray(data.points, dtype=float).reshape((-1, 3))
            for kind, actor_name in actor_names.items():
                ids = tuple(
                    entity_id
                    for entity_kind, entity_id in self._sketch_constraint_selection
                    if entity_kind == kind
                )
                if not ids:
                    continue
                selection_data = replace(
                    data,
                    selected_kind=kind,
                    selected_id=ids[0],
                    selected_ids=ids,
                )
                geometry = self._sketch_selection_polydata(
                    selection_data,
                    coordinates,
                )
                if geometry is None:
                    continue
                self._actors[actor_name] = self._plotter.add_mesh(
                    geometry,
                    color="#ffb300",
                    line_width=8,
                    point_size=14,
                    render_points_as_spheres=True,
                    render_lines_as_tubes=True,
                    opacity=1.0,
                    name=actor_name,
                    reset_camera=False,
                    pickable=False,
                )
        if render:
            self._render()

    @staticmethod
    def _sketch_entity_coordinates(
        data: SketchDraftRenderData,
        kind: str,
        entity_id: str,
    ) -> np.ndarray | None:
        coordinates = np.asarray(data.points, dtype=float).reshape((-1, 3))
        if kind == "point":
            indices = tuple(
                index
                for index, point_id in enumerate(data.point_ids)
                if point_id == entity_id
            )
            return coordinates[np.asarray(indices)] if indices else None
        if kind == "curve" and entity_id in data.curve_ids:
            return coordinates[np.asarray(data.curves[data.curve_ids.index(entity_id)])]
        if kind == "profile":
            indices = tuple(
                index
                for face, face_id in zip(data.faces, data.face_ids, strict=True)
                if face_id == entity_id
                for index in face
            )
            return coordinates[np.asarray(indices)] if indices else None
        return None

    def _show_sketch_draft(
        self,
        *,
        render: bool,
        reset_camera: bool = False,
    ) -> None:
        data = self._sketch_draft_render_data
        if data is None:
            return
        for name in (
            "sketch_work_plane_grid",
            "sketch_work_plane_axis_0",
            "sketch_work_plane_axis_1",
            "sketch_work_plane_origin",
            "sketch_work_plane_axis_labels",
            "sketch_draft_faces",
            "sketch_draft_curves",
            "sketch_draft_points",
            "sketch_draft_point_labels",
            "sketch_reference_points",
            "sketch_reference_labels",
            "sketch_authoring_selection",
            "sketch_authoring_shape_preview",
            "sketch_authoring_hover_outline",
            "sketch_authoring_hover",
            "sketch_authoring_hover_label",
            "sketch_entity_hover",
            "sketch_constraint_selection_points",
            "sketch_constraint_selection_curves",
        ):
            self._remove_actor(name)
        if is_offscreen_environment():
            self._message.setText("二维草图编辑中（当前环境未启用三维渲染）")
            self._stack.setCurrentWidget(self._message)
            return
        if not self._ensure_plotter() or _pyvista is None:
            return
        coordinates = np.asarray(data.points, dtype=float).reshape((-1, 3))
        geometry_color = _sketch_geometry_color(data.constraint_status)
        grid, grid_layout = _sketch_grid_polydata(
            _pyvista,
            data.plane,
            data.points,
            self._sketch_grid_spacing,
        )
        if self._sketch_grid_visible:
            self._actors["sketch_work_plane_grid"] = self._plotter.add_mesh(
                grid,
                color="#7f8f9f",
                opacity=0.35,
                line_width=1,
                name="sketch_work_plane_grid",
                reset_camera=False,
                pickable=False,
            )
        origin = np.asarray(data.plane.origin, dtype=float)
        label_points = [origin]
        axis_labels = ["O"]
        for index in range(2) if self._sketch_show_work_plane_axes else ():
            local_start, local_end = _sketch_axis_local_endpoints(
                grid_layout,
                index,
            )
            start = np.asarray(
                data.plane.to_global(*local_start),
                dtype=float,
            )
            end = np.asarray(
                data.plane.to_global(*local_end),
                dtype=float,
            )
            axis_line = _pyvista.PolyData()
            axis_line.points = np.asarray((start, end), dtype=float)
            axis_line.lines = np.asarray((2, 0, 1), dtype=np.int64)
            actor_name = f"sketch_work_plane_axis_{index}"
            self._actors[actor_name] = self._plotter.add_mesh(
                axis_line,
                color=("#d9534f", "#45a049")[index],
                line_width=4,
                name=actor_name,
                reset_camera=False,
                pickable=False,
            )
            label_points.append(end)
            axis_labels.append(("U", "V")[index])
        if self._sketch_show_work_plane_axes:
            self._actors["sketch_work_plane_origin"] = self._plotter.add_mesh(
                _pyvista.PolyData(np.asarray((origin,), dtype=float)),
                color="#ff8c00",
                point_size=13,
                render_points_as_spheres=True,
                name="sketch_work_plane_origin",
                reset_camera=False,
                pickable=False,
            )
            self._actors["sketch_work_plane_axis_labels"] = (
                self._plotter.add_point_labels(
                    np.asarray(label_points, dtype=float),
                    axis_labels,
                    point_size=0,
                    font_size=11,
                    shape=None,
                    text_color="#ff8c00",
                    name="sketch_work_plane_axis_labels",
                    reset_camera=False,
                )
            )
        if data.faces and self._sketch_show_profile_fill:
            surface = _pyvista.PolyData()
            surface.points = coordinates
            surface.faces = np.hstack(
                tuple((len(face), *face) for face in data.faces)
            ).astype(np.int64)
            self._actors["sketch_draft_faces"] = self._plotter.add_mesh(
                surface,
                color="#80b9d8",
                opacity=0.28,
                show_edges=False,
                name="sketch_draft_faces",
                reset_camera=False,
                pickable=False,
            )
        if data.curves:
            curves = _pyvista.PolyData()
            curves.points = coordinates
            curves.lines = np.hstack(
                tuple((len(curve), *curve) for curve in data.curves)
            ).astype(np.int64)
            self._actors["sketch_draft_curves"] = self._plotter.add_mesh(
                curves,
                color=geometry_color,
                line_width=3,
                name="sketch_draft_curves",
                reset_camera=False,
                pickable=False,
            )
        authoring_points = tuple(
            index
            for index, point_id in enumerate(data.point_ids)
            if point_id is not None
        )
        if authoring_points:
            point_coordinates = coordinates[np.asarray(authoring_points)]
            self._actors["sketch_draft_points"] = self._plotter.add_mesh(
                _pyvista.PolyData(point_coordinates),
                color=geometry_color,
                point_size=10,
                render_points_as_spheres=True,
                name="sketch_draft_points",
                reset_camera=False,
                pickable=False,
            )
            if self._sketch_show_point_ids:
                self._actors["sketch_draft_point_labels"] = (
                    self._plotter.add_point_labels(
                        point_coordinates,
                        tuple(str(index + 1) for index in range(len(authoring_points))),
                        point_size=0,
                        font_size=9,
                        shape=None,
                        text_color=geometry_color,
                        name="sketch_draft_point_labels",
                        reset_camera=False,
                    )
                )
        if self._sketch_reference_points:
            reference_coordinates = np.asarray(
                tuple(point.position for point in self._sketch_reference_points),
                dtype=float,
            )
            self._actors["sketch_reference_points"] = self._plotter.add_mesh(
                _pyvista.PolyData(reference_coordinates),
                color="#d32f2f",
                point_size=13,
                render_points_as_spheres=True,
                name="sketch_reference_points",
                reset_camera=False,
                pickable=False,
            )
            labels = {
                "topology_vertex": "顶点",
                "line_midpoint": "中点",
                "circle_center": "圆心",
                "arc_center": "弧心",
                "face_center": "面中心",
            }
            if self._sketch_show_external_labels:
                self._actors["sketch_reference_labels"] = (
                    self._plotter.add_point_labels(
                        reference_coordinates,
                        [
                            labels[point.derived_type.value]
                            for point in self._sketch_reference_points
                        ],
                        point_size=0,
                        font_size=9,
                        shape=None,
                        text_color="#d32f2f",
                        name="sketch_reference_labels",
                        reset_camera=False,
                    )
                )
        if data.constraint_overlays:
            self._actors["sketch_constraint_overlays"] = self._plotter.add_point_labels(
                np.asarray(tuple(item.position for item in data.constraint_overlays)),
                tuple(item.text for item in data.constraint_overlays),
                point_size=0,
                font_size=10,
                shape="rounded_rect",
                text_color="#c62828" if any(item.warning for item in data.constraint_overlays) else "#5d3a9b",
                name="sketch_constraint_overlays",
                reset_camera=False,
            )
        selection = self._sketch_selection_polydata(data, coordinates)
        if selection is not None:
            self._actors["sketch_authoring_selection"] = self._plotter.add_mesh(
                selection,
                color="#ffb300",
                line_width=8,
                point_size=14,
                render_points_as_spheres=True,
                render_lines_as_tubes=True,
                opacity=1.0,
                name="sketch_authoring_selection",
                reset_camera=False,
                pickable=False,
            )
        self._show_sketch_constraint_selection(render=False)
        self._show_sketch_entity_hover(render=False)
        self._show_sketch_authoring_hover(render=False)
        if reset_camera:
            bounds = _sketch_camera_bounds(
                data.points,
                self._sketch_grid_spacing,
                data.plane,
            )
            local_points = _sketch_local_points(
                data.points,
                data.plane,
            )
            local_center = (
                np.mean(local_points, axis=0)
                if len(local_points)
                else np.zeros(2, dtype=float)
            )
            focal = np.asarray(
                data.plane.to_global(*local_center),
                dtype=float,
            )
            span = max(
                bounds[1] - bounds[0],
                bounds[3] - bounds[2],
                bounds[5] - bounds[4],
                1.0,
            )
            position = focal + 2.0 * span * np.asarray(
                data.plane.normal
            )
            camera = self._plotter.camera
            camera.SetFocalPoint(*focal)
            camera.SetPosition(*position)
            camera.SetViewUp(*data.plane.y_direction)
            camera.ParallelProjectionOn()
            camera.OrthogonalizeViewUp()
            self._plotter.reset_camera(
                bounds=bounds,
                render=False,
            )
        self._refresh_sketch_curve_sampling(render=False)
        if render:
            self._render()

    def _refresh_sketch_curve_sampling(self, *, render: bool) -> None:
        data = self._sketch_draft_render_data
        world_per_pixel = self._world_per_pixel()
        if (
            data is None
            or not data.analytic_curves
            or world_per_pixel is None
            or self._plotter is None
            or _pyvista is None
        ):
            return
        point_map = {point.id: point for point in data.analytic_points}
        coordinates: list[tuple[float, float, float]] = []
        cells: list[tuple[int, ...]] = []

        def append_point(u: float, v: float) -> int:
            index = len(coordinates)
            coordinates.append(data.plane.to_global(u, v))
            return index

        for curve in data.analytic_curves:
            if isinstance(curve, SketchLine):
                start = point_map[curve.start_point_id]
                end = point_map[curve.end_point_id]
                cells.append(
                    (append_point(start.u, start.v), append_point(end.u, end.v))
                )
                continue
            center = point_map[curve.center_point_id]
            if isinstance(curve, SketchCircle):
                radius = curve.radius
                start_angle = 0.0
                sweep = math.tau
            else:
                start = point_map[curve.start_point_id]
                end = point_map[curve.end_point_id]
                radius = math.hypot(start.u - center.u, start.v - center.v)
                start_angle = math.atan2(start.v - center.v, start.u - center.u)
                end_angle = math.atan2(end.v - center.v, end.u - center.u)
                sweep = (
                    (end_angle - start_angle) % math.tau
                    if curve.orientation == "ccw"
                    else -((start_angle - end_angle) % math.tau)
                )
            count = _sketch_curve_sample_count(
                radius,
                sweep,
                world_per_pixel,
            )
            cells.append(
                tuple(
                    append_point(
                        center.u + radius * math.cos(start_angle + sweep * index / count),
                        center.v + radius * math.sin(start_angle + sweep * index / count),
                    )
                    for index in range(count + 1)
                )
            )

        polydata = _pyvista.PolyData()
        polydata.points = np.asarray(coordinates, dtype=float).reshape((-1, 3))
        polydata.lines = np.hstack(
            tuple((len(cell), *cell) for cell in cells)
        ).astype(np.int64)
        self._remove_actor("sketch_draft_curves")
        self._actors["sketch_draft_curves"] = self._plotter.add_mesh(
            polydata,
            color=_sketch_geometry_color(data.constraint_status),
            line_width=3,
            name="sketch_draft_curves",
            reset_camera=False,
            pickable=False,
        )
        if render:
            self._render()

    def _sketch_selection_polydata(
        self,
        data: SketchDraftRenderData,
        coordinates: np.ndarray,
    ):
        if not data.selected_ids or _pyvista is None:
            return None
        if data.selected_kind == "point":
            indices = tuple(
                index
                for index, point_id in enumerate(data.point_ids)
                if point_id in data.selected_ids
            )
            if indices:
                return _pyvista.PolyData(coordinates[np.asarray(indices)])
        if data.selected_kind == "curve":
            cells = tuple(
                curve
                for curve, curve_id in zip(data.curves, data.curve_ids, strict=True)
                if curve_id in data.selected_ids
            )
            if not cells:
                return None
            result = _pyvista.PolyData()
            result.points = coordinates
            result.lines = np.hstack(
                tuple((len(curve), *curve) for curve in cells)
            ).astype(np.int64)
            return result
        if data.selected_kind == "profile":
            cells = tuple(
                face
                for face, face_id in zip(
                    data.faces,
                    data.face_ids,
                    strict=True,
                )
                if face_id in data.selected_ids
            )
            if cells:
                result = _pyvista.PolyData()
                result.points = coordinates
                result.faces = np.hstack(
                    tuple((len(face), *face) for face in cells)
                ).astype(np.int64)
                return result
        return None

    @property
    def wire_work_plane(self) -> str:
        return self._wire_work_plane

    def start_wire_authoring(
        self,
        render_data: WireDraftRenderData,
        *,
        work_plane: str = "XY",
        offset: float = 0.0,
        snap: bool = True,
        spacing: float = 0.1,
    ) -> None:
        """Enter the transient wire tool without touching Session state."""

        if type(render_data) is not WireDraftRenderData:
            raise TypeError("render_data must be a WireDraftRenderData")
        clean_plane = str(work_plane).upper()
        if clean_plane not in {"XY", "XZ", "YZ"}:
            raise ValueError("work plane must be XY, XZ, or YZ")
        if not np.isfinite(float(offset)):
            raise ValueError("work plane offset must be finite")
        if not np.isfinite(float(spacing)) or float(spacing) <= 0.0:
            raise ValueError("grid spacing must be positive")
        self._wire_authoring_active = True
        self._wire_authoring_mode = "point"
        self._wire_work_plane = clean_plane
        self._wire_plane_offset = float(offset)
        self._wire_grid_snap = bool(snap)
        self._wire_grid_spacing = float(spacing)
        self._wire_pending_member_start = render_data.pending_member_start
        self._wire_authoring_selection = None
        self._wire_authoring_hover = None
        self._wire_authoring_preview_point = None
        self._wire_draft_render_data = render_data
        self._clear_preselection(render=False)
        self._remove_all_layers(render=False)
        self._show_wire_draft(render=False, reset_camera=True)

    def update_wire_draft(self, render_data: WireDraftRenderData) -> None:
        if type(render_data) is not WireDraftRenderData:
            raise TypeError("render_data must be a WireDraftRenderData")
        if not self._wire_authoring_active:
            return
        self._wire_draft_render_data = render_data
        self._wire_pending_member_start = render_data.pending_member_start
        selected = self._wire_authoring_selection
        if selected is not None:
            kind, name = selected
            names = (
                render_data.point_names
                if kind == "point"
                else render_data.member_names
            )
            if name not in names:
                self._wire_authoring_selection = None
        hovered = self._wire_authoring_hover
        if hovered is not None:
            kind, name = hovered
            names = (
                render_data.point_names
                if kind == "point"
                else render_data.member_names
            )
            if name not in names:
                self._wire_authoring_hover = None
        self._show_wire_draft(render=True, reset_camera=False)

    def set_wire_authoring_mode(self, mode: str) -> None:
        normalized = str(mode).strip().casefold()
        aliases = {"point": "point", "member": "member", "select": "select"}
        if normalized not in aliases:
            raise ValueError("wire authoring mode must be point, member, or select")
        self._wire_authoring_mode = aliases[normalized]
        self._wire_pending_member_start = None
        self._wire_authoring_selection = None
        self._wire_authoring_hover = None
        self._wire_authoring_preview_point = None
        if self._wire_draft_render_data is not None:
            self._wire_draft_render_data = replace(
                self._wire_draft_render_data,
                pending_member_start=None,
            )
            self._show_wire_draft(render=True)

    def set_wire_work_plane(
        self,
        plane: str,
        offset: float | None = None,
        *,
        snap: bool | None = None,
        spacing: float | None = None,
    ) -> None:
        clean_plane = str(plane).upper()
        if clean_plane not in {"XY", "XZ", "YZ"}:
            raise ValueError("work plane must be XY, XZ, or YZ")
        if offset is not None and not np.isfinite(float(offset)):
            raise ValueError("work plane offset must be finite")
        if spacing is not None and (
            not np.isfinite(float(spacing)) or float(spacing) <= 0.0
        ):
            raise ValueError("grid spacing must be positive")
        self._wire_work_plane = clean_plane
        if offset is not None:
            self._wire_plane_offset = float(offset)
        if snap is not None:
            self._wire_grid_snap = bool(snap)
        if spacing is not None:
            self._wire_grid_spacing = float(spacing)
        if self._wire_authoring_active:
            self._wire_authoring_hover = None
            self._wire_authoring_preview_point = None
            self._show_wire_draft(render=True)

    def cancel_pending_wire_interaction(self) -> bool:
        if self._wire_pending_member_start is None and self._wire_authoring_selection is None:
            return False
        self._wire_pending_member_start = None
        self._wire_authoring_selection = None
        if self._wire_draft_render_data is not None:
            self._wire_draft_render_data = replace(
                self._wire_draft_render_data,
                pending_member_start=None,
            )
            self._show_wire_draft(render=True)
        self.wirePendingInteractionCancelled.emit()
        return True

    def stop_wire_authoring(self, *, render: bool = True) -> None:
        """Leave the wire tool and remove only its transient display actors."""

        if (
            not self._wire_authoring_active
            and self._wire_draft_render_data is None
        ):
            return
        for name in (
            "wire_work_plane_grid",
            "wire_work_plane_axis_0",
            "wire_work_plane_axis_1",
            "wire_work_plane_origin",
            "wire_work_plane_axis_labels",
            "wire_draft_members",
            "wire_draft_points",
            "wire_pending_member_start",
            "wire_authoring_selection",
            "wire_authoring_selection_label",
            "wire_authoring_hover_outline",
            "wire_authoring_hover",
            "wire_authoring_hover_label",
        ):
            self._remove_actor(name)
        self._wire_authoring_active = False
        self._wire_draft_render_data = None
        self._wire_pending_member_start = None
        self._wire_authoring_selection = None
        self._wire_authoring_hover = None
        self._wire_authoring_preview_point = None
        self._clear_preselection(render=False)
        if render:
            self._render()

    def show_wire_draft(self, render_data: WireDraftRenderData) -> None:
        """Public alias used by the editor panel and focused tests."""

        self.update_wire_draft(render_data)

    def focus_wire_draft_entity(self, kind: str, name: str) -> None:
        """Highlight a draft entity selected from the editor table."""

        if kind not in {"point", "member"}:
            raise ValueError("wire draft focus kind must be point or member")
        data = self._wire_draft_render_data
        if data is None:
            return
        names = data.point_names if kind == "point" else data.member_names
        if name not in names:
            return
        self._wire_authoring_selection = (kind, name)
        self._show_wire_draft(render=True)

    def _show_wire_draft(
        self,
        *,
        render: bool,
        reset_camera: bool = False,
    ) -> None:
        data = self._wire_draft_render_data
        if data is None:
            return
        camera_state = (
            None
            if reset_camera or self._plotter is None
            else _capture_camera_state(self._plotter)
        )
        for name in (
            "wire_work_plane_grid",
            "wire_work_plane_axis_0",
            "wire_work_plane_axis_1",
            "wire_work_plane_origin",
            "wire_work_plane_axis_labels",
            "wire_draft_members",
            "wire_draft_points",
            "wire_pending_member_start",
            "wire_authoring_selection",
            "wire_authoring_selection_label",
            "wire_authoring_hover_outline",
            "wire_authoring_hover",
            "wire_authoring_hover_label",
        ):
            self._remove_actor(name)
        if is_offscreen_environment():
            self._message.setText("线体编辑中（当前环境未启用三维渲染）")
            self._stack.setCurrentWidget(self._message)
            return
        if not self._ensure_plotter():
            return
        if _pyvista is None:
            return
        points = np.asarray(data.points, dtype=float).reshape((-1, 3))
        grid_layout = _wire_grid_layout(
            points,
            self._wire_work_plane,
            self._wire_plane_offset,
            self._wire_grid_spacing,
        )
        axes = grid_layout.axes
        try:
            grid = _wire_grid_polydata(_pyvista, grid_layout)
            self._actors["wire_work_plane_grid"] = self._plotter.add_mesh(
                grid,
                color="#7f8f9f",
                opacity=0.45,
                line_width=2,
                name="wire_work_plane_grid",
                reset_camera=False,
                pickable=False,
            )
            center = np.asarray(grid_layout.center, dtype=float)
            origin = np.zeros(3, dtype=float)
            origin[axes[2]] = self._wire_plane_offset
            half_size = 0.5 * grid_layout.plane_size
            axis_colors = ("#d9534f", "#45a049", "#3b82c4")
            label_points = [origin]
            axis_labels = ["O"]
            for index, axis in enumerate(axes[:2]):
                start = origin.copy()
                end = origin.copy()
                start[axis] = center[axis] - half_size
                end[axis] = center[axis] + half_size
                axis_line = _pyvista.PolyData()
                axis_line.points = np.asarray((start, end), dtype=float)
                axis_line.lines = np.asarray((2, 0, 1), dtype=np.int64)
                actor_name = f"wire_work_plane_axis_{index}"
                self._actors[actor_name] = self._plotter.add_mesh(
                    axis_line,
                    color=axis_colors[axis],
                    line_width=4,
                    name=actor_name,
                    reset_camera=False,
                    pickable=False,
                )
                label_points.append(end)
                axis_labels.append("XYZ"[axis])
            self._actors["wire_work_plane_origin"] = self._plotter.add_mesh(
                _pyvista.PolyData(np.asarray((origin,), dtype=float)),
                color="#ff8c00",
                point_size=13,
                render_points_as_spheres=True,
                name="wire_work_plane_origin",
                reset_camera=False,
                pickable=False,
            )
            self._actors["wire_work_plane_axis_labels"] = (
                self._plotter.add_point_labels(
                    np.asarray(label_points, dtype=float),
                    axis_labels,
                    point_size=0,
                    font_size=11,
                    shape=None,
                    text_color="#ff8c00",
                    name="wire_work_plane_axis_labels",
                    reset_camera=False,
                )
            )
        except Exception:
            logging.exception("failed to render wire work plane")
        if data.members:
            line_cells = np.hstack(
                [np.asarray((2, *member), dtype=np.int64) for member in data.members]
            )
            members = _pyvista.PolyData()
            members.points = points
            members.lines = line_cells
            members.cell_data["draft_member_index"] = np.arange(
                len(data.members), dtype=np.int64
            )
            self._actors["wire_draft_members"] = self._plotter.add_mesh(
                members,
                color="#334b5f",
                line_width=3,
                name="wire_draft_members",
                reset_camera=False,
                pickable=False,
            )
        if len(points):
            draft_points = _pyvista.PolyData(points)
            draft_points.point_data["draft_point_index"] = np.arange(
                len(points), dtype=np.int64
            )
            self._actors["wire_draft_points"] = self._plotter.add_mesh(
                draft_points,
                color="#406f8f",
                point_size=10,
                render_points_as_spheres=True,
                name="wire_draft_points",
                reset_camera=False,
                pickable=False,
            )
            pending = data.pending_member_start
            if pending in data.point_names:
                pending_point = _pyvista.PolyData(
                    points[[data.point_names.index(pending)]]
                )
                self._actors["wire_pending_member_start"] = self._plotter.add_mesh(
                    pending_point,
                    color="#d69a3a",
                    point_size=15,
                    render_points_as_spheres=True,
                    name="wire_pending_member_start",
                    reset_camera=False,
                    pickable=False,
                )
        selected = self._wire_authoring_selection
        selection_label_point = None
        if selected is not None:
            kind, name = selected
            if kind == "point" and name in data.point_names:
                selected_points = points[[data.point_names.index(name)]]
                selected_point = _pyvista.PolyData(selected_points)
                self._actors["wire_authoring_selection"] = self._plotter.add_mesh(
                    selected_point,
                    color="#ff8c00",
                    point_size=20,
                    render_points_as_spheres=True,
                    name="wire_authoring_selection",
                    reset_camera=False,
                    pickable=False,
                )
                selection_label_point = selected_points
            elif kind == "member" and name in data.member_names:
                member_index = data.member_names.index(name)
                start, end = data.members[member_index]
                selected_member = _pyvista.PolyData()
                selected_member.points = points
                selected_member.lines = np.asarray(
                    (2, start, end),
                    dtype=np.int64,
                )
                self._actors["wire_authoring_selection"] = self._plotter.add_mesh(
                    selected_member,
                    color="#ff8c00",
                    line_width=8,
                    name="wire_authoring_selection",
                    reset_camera=False,
                    pickable=False,
                )
                selection_label_point = np.mean(
                    points[[start, end]],
                    axis=0,
                    keepdims=True,
                )
            if selection_label_point is not None:
                self._actors["wire_authoring_selection_label"] = (
                    self._plotter.add_point_labels(
                        selection_label_point,
                        [name],
                        point_size=0,
                        font_size=11,
                        shape=None,
                        text_color="#ff8c00",
                        name="wire_authoring_selection_label",
                        reset_camera=False,
                    )
                )
        self._show_wire_authoring_hover(render=False)
        if reset_camera:
            self._plotter.reset_camera()
        elif camera_state is not None:
            _restore_camera_state(self._plotter, camera_state)
        if render:
            self._render()

    def _set_sketch_authoring_preview_point(
        self,
        point: tuple[float, float, float] | None,
    ) -> None:
        normalized = (
            None
            if point is None
            else tuple(float(value) for value in point)
        )
        if normalized == self._sketch_authoring_preview_point:
            return
        self._sketch_authoring_preview_point = normalized
        self._show_sketch_authoring_hover(render=True)

    def _show_sketch_authoring_hover(self, *, render: bool) -> None:
        for actor_name in (
            "sketch_authoring_shape_preview",
            "sketch_authoring_hover_outline",
            "sketch_authoring_hover",
            "sketch_authoring_hover_label",
        ):
            self._remove_actor(actor_name)
        point = self._sketch_authoring_preview_point
        if (
            point is None
            or self._plotter is None
            or _pyvista is None
        ):
            if render and self._plotter is not None:
                self._render()
            return
        shape_points = _sketch_shape_preview_points(
            self._sketch_authoring_mode,
            self._sketch_pending_points,
            point,
            (
                None
                if self._sketch_draft_render_data is None
                else self._sketch_draft_render_data.plane
            ),
        )
        if shape_points:
            shape = _pyvista.PolyData()
            shape.points = np.asarray(shape_points, dtype=float)
            shape.lines = np.asarray(
                (len(shape_points), *range(len(shape_points))),
                dtype=np.int64,
            )
            self._actors["sketch_authoring_shape_preview"] = (
                self._plotter.add_mesh(
                    shape,
                    color="#00b8d4",
                    line_width=4,
                    name="sketch_authoring_shape_preview",
                    reset_camera=False,
                    pickable=False,
                )
            )
        label_point = np.asarray((point,), dtype=float)
        self._actors["sketch_authoring_hover_outline"] = (
            self._plotter.add_mesh(
                _pyvista.PolyData(label_point),
                color="#e0a800",
                point_size=14,
                render_points_as_spheres=True,
                name="sketch_authoring_hover_outline",
                reset_camera=False,
                pickable=False,
            )
        )
        self._actors["sketch_authoring_hover"] = self._plotter.add_mesh(
            _pyvista.PolyData(label_point),
            color="#ffd54f",
            point_size=9,
            render_points_as_spheres=True,
            name="sketch_authoring_hover",
            reset_camera=False,
            pickable=False,
        )
        coordinate_label = _sketch_coordinate_label(
            point,
            (
                SketchPlane.xy()
                if self._sketch_draft_render_data is None
                else self._sketch_draft_render_data.plane
            ),
        )
        snap_label = _sketch_snap_label(self._sketch_authoring_snap_kind)
        inference_labels = {
            "coincident": "重合",
            "point_on_curve": "点在曲线上",
            "horizontal": "水平",
            "vertical": "垂直",
        }
        inference = "、".join(
            inference_labels.get(kind, kind)
            for kind in (
                self._sketch_draft_render_data.inference_preview
                if self._sketch_draft_render_data is not None
                else ()
            )
        )
        hover_text = f"{snap_label} {coordinate_label}" if snap_label else coordinate_label
        if inference:
            hover_text += f"  推断：{inference}"
        self._actors["sketch_authoring_hover_label"] = (
            self._plotter.add_point_labels(
                label_point,
                [hover_text],
                point_size=0,
                font_size=11,
                shape=None,
                text_color="#a66a00",
                name="sketch_authoring_hover_label",
                reset_camera=False,
            )
        )
        if render:
            self._render()

    def _sketch_point_at(
        self,
        x: int,
        y: int,
        tolerance: float = 9.0,
    ) -> str | None:
        data = self._sketch_draft_render_data
        if data is None or not data.points:
            return None
        display = self._world_points_to_display(
            np.asarray(data.points, dtype=float)
        )
        if display is None:
            return None
        mouse = np.asarray((float(x), float(y)), dtype=float)
        candidates: list[tuple[float, int]] = []
        for index, point_id in enumerate(data.point_ids):
            if point_id is None:
                continue
            distance = float(np.linalg.norm(display[index, :2] - mouse))
            if distance <= tolerance * self._device_pixel_ratio():
                candidates.append((distance, index))
        if not candidates:
            return None
        return data.point_ids[min(candidates)[1]]

    def _sketch_curve_at(
        self,
        x: int,
        y: int,
        tolerance: float = 7.0,
    ) -> str | None:
        data = self._sketch_draft_render_data
        if data is None or not data.curves:
            return None
        display = self._world_points_to_display(
            np.asarray(data.points, dtype=float)
        )
        if display is None:
            return None
        mouse = np.asarray((float(x), float(y)), dtype=float)
        candidates: list[tuple[float, int]] = []
        for curve_index, curve in enumerate(data.curves):
            distance = min(
                _point_to_segment_distance(
                    mouse,
                    display[start, :2],
                    display[end, :2],
                )[0]
                for start, end in zip(curve, curve[1:])
            )
            if distance <= tolerance * self._device_pixel_ratio():
                candidates.append((distance, curve_index))
        if not candidates:
            return None
        return data.curve_ids[min(candidates)[1]]

    def _sketch_profile_at(self, x: int, y: int) -> str | None:
        data = self._sketch_draft_render_data
        if data is None or not data.faces:
            return None
        display = self._world_points_to_display(
            np.asarray(data.points, dtype=float)
        )
        if display is None:
            return None
        point = np.asarray((float(x), float(y)), dtype=float)
        for face, face_id in reversed(
            tuple(zip(data.faces, data.face_ids, strict=True))
        ):
            polygon = display[np.asarray(face), :2]
            if _display_point_in_polygon(point, polygon):
                return face_id
        return None

    def _sketch_work_plane_point_at(
        self,
        x: int,
        y: int,
        *,
        snap: bool = True,
    ) -> tuple[tuple[float, float, float] | None, str | None]:
        self._sketch_authoring_snap_reference = None
        self._sketch_authoring_snap_kind = None
        self._sketch_authoring_snap_point_id = None
        self._sketch_authoring_intersection_curve_ids = ()
        near = self._display_to_world(float(x), float(y), 0.0)
        far = self._display_to_world(float(x), float(y), 1.0)
        if near is None or far is None:
            return None, "point.ray"
        data = self._sketch_draft_render_data
        plane = SketchPlane.xy() if data is None else data.plane
        point = _intersect_ray_with_sketch_plane(near, far, plane)
        if point is None:
            return None, "point.parallel"
        if snap:
            candidate_world: list[tuple[float, float, float]] = []
            candidate_values: list[
                tuple[str, float, float, SketchReferencePoint | None, str | None]
            ] = []
            if data is not None and self._sketch_snap_sketch_points:
                for coordinate, point_id in zip(
                    data.points,
                    data.point_ids,
                    strict=True,
                ):
                    if point_id is None:
                        continue
                    u, v = plane.to_local(coordinate)
                    candidate_world.append(coordinate)
                    candidate_values.append(
                        ("sketch_point", u, v, None, point_id)
                    )
            if data is not None and self._sketch_snap_midpoints:
                for midpoint in data.snap_midpoints:
                    midpoint_u, midpoint_v = plane.to_local(midpoint)
                    candidate_world.append(midpoint)
                    candidate_values.append(
                        ("line_midpoint", midpoint_u, midpoint_v, None, None)
                    )
            if data is not None and self._sketch_snap_centers:
                for center in data.snap_centers:
                    center_u, center_v = plane.to_local(center)
                    candidate_world.append(center)
                    candidate_values.append(
                        ("circle_center", center_u, center_v, None, None)
                    )
            enabled_external_kinds = set()
            if self._sketch_snap_external_points:
                enabled_external_kinds.update(("topology_vertex", "face_center"))
            if self._sketch_snap_midpoints:
                enabled_external_kinds.add("line_midpoint")
            if self._sketch_snap_centers:
                enabled_external_kinds.update(("circle_center", "arc_center"))
            for reference_point in self._sketch_reference_points:
                if reference_point.derived_type.value not in enabled_external_kinds:
                    continue
                candidate_world.append(reference_point.position)
                candidate_values.append(
                    (
                        reference_point.derived_type.value,
                        reference_point.u,
                        reference_point.v,
                        reference_point,
                        None,
                    )
                )
            if data is not None and self._sketch_snap_intersections:
                if self._sketch_intersection_revision != data.geometry_revision:
                    self._sketch_intersection_revision = data.geometry_revision
                    self._sketch_intersection_cache = _sketch_intersection_points(
                        data
                    )
                for intersection in self._sketch_intersection_cache:
                    intersection_u, intersection_v = plane.to_local(intersection)
                    candidate_world.append(intersection)
                    candidate_values.append(
                        (
                            "intersection",
                            intersection_u,
                            intersection_v,
                            None,
                            None,
                        )
                    )
            if self._sketch_grid_snap:
                grid_point = _snap_sketch_plane_point(
                    point,
                    plane,
                    self._sketch_grid_spacing,
                )
                grid_u, grid_v = plane.to_local(grid_point)
                candidate_world.append(grid_point)
                candidate_values.append(("grid", grid_u, grid_v, None, None))
            display = self._world_points_to_display(
                np.asarray(candidate_world, dtype=float)
            ) if candidate_world else None
            if display is not None:
                candidates = tuple(
                    SketchSnapCandidate(
                        kind,
                        float(screen[0]),
                        float(screen[1]),
                        u,
                        v,
                        reference_point,
                        point_id,
                    )
                    for screen, (
                        kind,
                        u,
                        v,
                        reference_point,
                        point_id,
                    ) in zip(display, candidate_values, strict=True)
                )
                selected = select_sketch_snap_candidate(
                    candidates,
                    (float(x), float(y)),
                    pixel_threshold=(
                        self._sketch_screen_snap_tolerance
                        * self._device_pixel_ratio()
                    ),
                )
                if selected is not None:
                    point = plane.to_global(selected.u, selected.v)
                    self._sketch_authoring_snap_reference = selected.reference_point
                    self._sketch_authoring_snap_kind = selected.kind
                    self._sketch_authoring_snap_point_id = selected.sketch_point_id
                    if selected.kind == "intersection" and data is not None:
                        self._sketch_authoring_intersection_curve_ids = (
                            _sketch_intersection_curve_ids_at(data, selected.u, selected.v)
                        )
            elif self._sketch_grid_snap:
                point = grid_point
                self._sketch_authoring_snap_kind = "grid"
        return point, None

    def _sketch_authoring_click(
        self,
        x: int,
        y: int,
        *,
        modifiers: Qt.KeyboardModifier = Qt.KeyboardModifier.NoModifier,
    ) -> None:
        if self._sketch_authoring_mode == "select":
            self._sketch_hover_entity = None
            self._show_sketch_entity_hover(render=False)
            point_id = self._sketch_point_at(x, y)
            if point_id is not None:
                self.sketchDraftPointSelectionRequested.emit(point_id, modifiers)
                self.sketchDraftPointSelected.emit(point_id)
                return
            curve_id = self._sketch_curve_at(x, y)
            if curve_id is not None:
                self.sketchDraftCurveSelectionRequested.emit(curve_id, modifiers)
                self.sketchDraftCurveSelected.emit(curve_id)
                return
            profile_id = self._sketch_profile_at(x, y)
            if profile_id is not None:
                self.sketchDraftProfileSelectionRequested.emit(profile_id, modifiers)
                self.sketchDraftProfileSelected.emit(profile_id)
                return
            self.sketchAuthoringMissed.emit("select")
            return
        point, reason = self._sketch_work_plane_point_at(
            x,
            y,
            snap=self._sketch_authoring_mode != "trim",
        )
        if point is None:
            self.sketchAuthoringMissed.emit(reason or "point.ray")
            return
        if self._sketch_authoring_mode == "trim":
            self._sketch_hover_entity = None
            self._show_sketch_entity_hover(render=False)
            curve_id = self._sketch_curve_at(x, y)
            if curve_id is None:
                self.sketchAuthoringMissed.emit("trim")
                return
            self.sketchTrimRequested.emit(curve_id, point)
            return
        self.sketchReferencePointSelected.emit(
            self._sketch_authoring_snap_reference
        )
        self.sketchSnapConfirmed.emit(
            {
                "kind": self._sketch_authoring_snap_kind,
                "point_id": self._sketch_authoring_snap_point_id,
                "curve_ids": self._sketch_authoring_intersection_curve_ids,
                "point": point,
            }
        )
        self.sketchWorkPlanePointSelected.emit(point)

    def _wire_point_at(self, x: int, y: int, tolerance: float = 9.0) -> str | None:
        data = self._wire_draft_render_data
        if data is None or not data.points:
            return None
        display = self._world_points_to_display(np.asarray(data.points, dtype=float))
        if display is None:
            return None
        mouse = np.asarray((float(x), float(y)), dtype=float)
        distances = np.linalg.norm(display[:, :2] - mouse, axis=1)
        candidates = np.flatnonzero(
            np.isfinite(distances)
            & (distances <= tolerance * self._device_pixel_ratio())
            & (display[:, 2] >= 0.0)
            & (display[:, 2] <= 1.0)
        )
        if not len(candidates):
            return None
        index = int(candidates[np.argmin(distances[candidates])])
        return data.point_names[index]

    def _wire_member_at(self, x: int, y: int, tolerance: float = 7.0) -> str | None:
        data = self._wire_draft_render_data
        if data is None or not data.members:
            return None
        display = self._world_points_to_display(np.asarray(data.points, dtype=float))
        if display is None:
            return None
        mouse = np.asarray((float(x), float(y)), dtype=float)
        candidates: list[tuple[float, int]] = []
        for index, (start, end) in enumerate(data.members):
            distance, _fraction = _point_to_segment_distance(
                mouse,
                display[start, :2],
                display[end, :2],
            )
            if distance <= tolerance * self._device_pixel_ratio():
                candidates.append((distance, index))
        if not candidates:
            return None
        return data.member_names[min(candidates)[1]]

    def _wire_hover_at(self, x: int, y: int) -> tuple[str, str] | None:
        point = self._wire_point_at(x, y)
        if point is not None:
            return ("point", point)
        if self._wire_authoring_mode == "select":
            member = self._wire_member_at(x, y)
            if member is not None:
                return ("member", member)
        return None

    def _wire_work_plane_point_at(
        self,
        x: int,
        y: int,
    ) -> tuple[tuple[float, float, float] | None, str | None]:
        """Return the exact point that a point-mode click would create."""

        near = self._display_to_world(float(x), float(y), 0.0)
        far = self._display_to_world(float(x), float(y), 1.0)
        if near is None or far is None:
            return None, "point.ray"
        point = intersect_ray_with_work_plane(
            near,
            far,
            self._wire_work_plane,
            self._wire_plane_offset,
        )
        if point is None:
            return None, "point.parallel"
        if self._wire_grid_snap:
            point = snap_work_plane_point(
                point,
                self._wire_work_plane,
                self._wire_grid_spacing,
            )
        return point, None

    def _update_wire_authoring_hover(self, x: int, y: int) -> None:
        hovered = self._wire_hover_at(x, y)
        preview_point = None
        if self._wire_authoring_mode == "point" and hovered is None:
            preview_point, _reason = self._wire_work_plane_point_at(x, y)
        self._set_wire_authoring_hover(
            hovered,
            preview_point=preview_point,
        )

    def _set_wire_authoring_hover(
        self,
        hovered: tuple[str, str] | None,
        *,
        preview_point: tuple[float, float, float] | None = None,
    ) -> None:
        normalized_preview = (
            None
            if preview_point is None
            else tuple(float(value) for value in preview_point)
        )
        if (
            hovered == self._wire_authoring_hover
            and normalized_preview == self._wire_authoring_preview_point
        ):
            return
        self._wire_authoring_hover = hovered
        self._wire_authoring_preview_point = normalized_preview
        self._show_wire_authoring_hover(render=True)

    def _show_wire_authoring_hover(self, *, render: bool) -> None:
        for actor_name in (
            "wire_authoring_hover_outline",
            "wire_authoring_hover",
            "wire_authoring_hover_label",
        ):
            self._remove_actor(actor_name)
        data = self._wire_draft_render_data
        hovered = self._wire_authoring_hover
        preview_point = self._wire_authoring_preview_point
        if (
            (hovered is None and preview_point is None)
            or data is None
            or self._plotter is None
            or _pyvista is None
        ):
            if render and self._plotter is not None:
                self._render()
            return
        points = np.asarray(data.points, dtype=float).reshape((-1, 3))
        label_point = None
        label = None
        if preview_point is not None:
            label_point = np.asarray((preview_point,), dtype=float)
            label = _wire_coordinate_label(preview_point)
            self._actors["wire_authoring_hover_outline"] = (
                self._plotter.add_mesh(
                    _pyvista.PolyData(label_point),
                    color="#173443",
                    point_size=24,
                    render_points_as_spheres=True,
                    name="wire_authoring_hover_outline",
                    reset_camera=False,
                    pickable=False,
                )
            )
            self._actors["wire_authoring_hover"] = self._plotter.add_mesh(
                _pyvista.PolyData(label_point),
                color="#00e5ff",
                point_size=16,
                render_points_as_spheres=True,
                name="wire_authoring_hover",
                reset_camera=False,
                pickable=False,
            )
        elif hovered is not None:
            kind, name = hovered
            label = name
            if kind == "point" and name in data.point_names:
                label_point = points[[data.point_names.index(name)]]
                self._actors["wire_authoring_hover_outline"] = (
                    self._plotter.add_mesh(
                        _pyvista.PolyData(label_point),
                        color="#173443",
                        point_size=26,
                        render_points_as_spheres=True,
                        name="wire_authoring_hover_outline",
                        reset_camera=False,
                        pickable=False,
                    )
                )
                self._actors["wire_authoring_hover"] = self._plotter.add_mesh(
                    _pyvista.PolyData(label_point),
                    color="#00e5ff",
                    point_size=18,
                    render_points_as_spheres=True,
                    name="wire_authoring_hover",
                    reset_camera=False,
                    pickable=False,
                )
            elif kind == "member" and name in data.member_names:
                member_index = data.member_names.index(name)
                start, end = data.members[member_index]
                hovered_member = _pyvista.PolyData()
                hovered_member.points = points
                hovered_member.lines = np.asarray(
                    (2, start, end),
                    dtype=np.int64,
                )
                self._actors["wire_authoring_hover_outline"] = (
                    self._plotter.add_mesh(
                        hovered_member,
                        color="#173443",
                        line_width=10,
                        name="wire_authoring_hover_outline",
                        reset_camera=False,
                        pickable=False,
                    )
                )
                self._actors["wire_authoring_hover"] = self._plotter.add_mesh(
                    hovered_member,
                    color="#00e5ff",
                    line_width=6,
                    name="wire_authoring_hover",
                    reset_camera=False,
                    pickable=False,
                )
                label_point = np.mean(
                    points[[start, end]],
                    axis=0,
                    keepdims=True,
                )
        if label_point is not None and label is not None:
            label_anchor = self._wire_label_anchor(label_point)
            self._actors["wire_authoring_hover_label"] = (
                self._plotter.add_point_labels(
                    label_anchor,
                    [label],
                    point_size=0,
                    font_size=11,
                    shape=None,
                    text_color="#00e5ff",
                    always_visible=True,
                    justification_horizontal="left",
                    justification_vertical="bottom",
                    name="wire_authoring_hover_label",
                    reset_camera=False,
                )
            )
        if render:
            self._render()

    def _wire_label_anchor(self, point: np.ndarray) -> np.ndarray:
        """Move a hover label away from its marker by a stable screen gap."""

        world = np.asarray(point, dtype=float).reshape((-1, 3))
        if len(world) != 1:
            return world
        display = self._world_to_display(world[0])
        if display is None:
            return world
        ratio = self._device_pixel_ratio()
        shifted = self._display_to_world(
            float(display[0]) + 18.0 * ratio,
            float(display[1]) + 10.0 * ratio,
            float(display[2]),
        )
        if shifted is None:
            return world
        return np.asarray(shifted, dtype=float).reshape((1, 3))

    def _wire_authoring_click(self, x: int, y: int) -> None:
        self._set_wire_authoring_hover(None)
        if self._wire_authoring_mode == "select":
            point = self._wire_point_at(x, y)
            if point is not None:
                self._wire_authoring_selection = ("point", point)
                self.wireDraftPointSelected.emit(point)
                self._show_wire_draft(render=True)
                return
            member = self._wire_member_at(x, y)
            if member is not None:
                self._wire_authoring_selection = ("member", member)
                self.wireDraftMemberSelected.emit(member)
                self._show_wire_draft(render=True)
                return
            self.wireAuthoringMissed.emit("select")
            return
        if self._wire_authoring_mode == "member":
            point = self._wire_point_at(x, y)
            if point is None:
                self.wireAuthoringMissed.emit("member")
                return
            if self._wire_pending_member_start is None:
                self._wire_pending_member_start = point
                self.wireMemberStartSelected.emit(point)
            elif point == self._wire_pending_member_start:
                self.wireAuthoringMissed.emit("member.same_endpoint")
            else:
                start = self._wire_pending_member_start
                self._wire_pending_member_start = None
                self.wireMemberEndpointsSelected.emit(start, point)
            if self._wire_draft_render_data is not None:
                self._wire_draft_render_data = replace(
                    self._wire_draft_render_data,
                    pending_member_start=self._wire_pending_member_start,
                )
                self._show_wire_draft(render=True)
            return
        existing = self._wire_point_at(x, y)
        if existing is not None:
            self._wire_authoring_selection = ("point", existing)
            self.wireDraftPointSelected.emit(existing)
            self._show_wire_draft(render=True)
            return
        point, reason = self._wire_work_plane_point_at(x, y)
        if point is None:
            self.wireAuthoringMissed.emit(reason or "point.ray")
            return
        self.wireWorkPlanePointSelected.emit(point)

    def set_geometry_ghost_preview(
        self,
        preview: GeometryPreview | None,
    ) -> None:
        """Set an unpickable translucent preview for suppressed source Parts."""

        if preview is not None and type(preview) is not GeometryPreview:
            raise TypeError("ghost preview must be a GeometryPreview or None")
        self._geometry_ghost_preview = preview

    def show_geometry_preview(
        self,
        preview: GeometryPreview,
        *,
        preserve_model: bool = False,
        render: bool = True,
    ) -> None:
        """Display CAD geometry, optionally as a picking overlay on the mesh."""
        self._geometry_preview = preview
        self._install_geometry_pick_bindings(preview)
        if is_offscreen_environment():
            self._message.setText("几何预览已更新（当前环境未启用三维渲染）")
            if not preserve_model:
                self._stack.setCurrentWidget(self._message)
            return
        if not self._ensure_plotter():
            return
        if preserve_model:
            for name in (
                "geometry_surface",
                "geometry_edges",
                "geometry_points",
                "geometry_selection",
            ):
                self._remove_actor(name)
        else:
            self._remove_all_layers(render=False)
        self._show_geometry_ghost_preview()
        points = np.asarray(preview.points, dtype=float)
        self._geometry_preview_surface = None
        self._geometry_preview_edges = None
        self._geometry_preview_points = None
        if preview.faces:
            surface = _geometry_surface_polydata(
                _pyvista,
                points,
                preview,
                self._geometry_face_pick_ids,
                self._geometry_face_body_pick_ids,
            )
            self._geometry_preview_surface = surface
            if not preserve_model:
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
                self._geometry_edge_body_pick_ids,
            )
            self._geometry_preview_edges = edge_mesh
            if not preserve_model:
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
        if point_mesh.n_points and not preserve_model:
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
                preview.topological_dimension == 1
                or self._selection_mode == "geometry_point"
            )
        if not preserve_model:
            self._pick_grid = None
            self._pick_locators.clear()
        self._clear_preselection(render=False)
        self._update_pickable_actors()
        if not preserve_model:
            self._reset_camera_to_fit()
        if render:
            self._render()

    def _show_geometry_ghost_preview(self) -> None:
        preview = self._geometry_ghost_preview
        if preview is None or not preview.points or _pyvista is None:
            return
        points = np.asarray(preview.points, dtype=float)
        if preview.faces:
            surface = _geometry_surface_polydata(
                _pyvista,
                points,
                preview,
                (0,) * len(preview.faces),
                (0,) * len(preview.faces),
            )
            self._actors["geometry_ghost_surface"] = self._plotter.add_mesh(
                surface,
                color="#a7b0b7",
                opacity=0.18,
                smooth_shading=False,
                show_edges=False,
                show_scalar_bar=False,
                name="geometry_ghost_surface",
                reset_camera=False,
            )
        if preview.edges:
            edges = _geometry_edge_polydata(
                _pyvista,
                points,
                preview,
                (0,) * len(preview.edges),
                (0,) * len(preview.edges),
            )
            self._actors["geometry_ghost_edges"] = self._plotter.add_mesh(
                edges,
                color="#78838c",
                opacity=0.28,
                line_width=1,
                show_scalar_bar=False,
                name="geometry_ghost_edges",
                reset_camera=False,
            )

    def hide_geometry_selection_overlay(self) -> None:
        """Remove a scope-picking overlay while preserving the current mesh."""
        overlay_names = (
            "geometry_surface",
            "geometry_edges",
            "geometry_points",
            "geometry_selection",
        )
        had_visible_overlay = any(
            name in self._actors for name in (*overlay_names, "preselection")
        )
        for name in overlay_names:
            self._remove_actor(name)
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
        self._geometry_face_body_pick_ids = ()
        self._geometry_edge_body_pick_ids = ()
        self._geometry_point_body_pick_ids = ()
        self._clear_preselection(render=False)
        self._update_pickable_actors()
        if had_visible_overlay:
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
        self._geometry_face_body_pick_ids = allocate(
            preview.face_body_logical_ids
        )
        self._geometry_edge_body_pick_ids = allocate(
            preview.edge_body_logical_ids
        )
        self._geometry_point_body_pick_ids = allocate(
            preview.point_body_logical_ids
        )
        body_pick_ids = {
            pick_id
            for pick_id in (
                *self._geometry_face_body_pick_ids,
                *self._geometry_edge_body_pick_ids,
                *self._geometry_point_body_pick_ids,
            )
            if pick_id > 0
        }
        self._geometry_body_pick_id = (
            next(iter(body_pick_ids))
            if len(
                {
                    pick_to_ref[pick_id]
                    for pick_id in body_pick_ids
                }
            ) == 1
            else 0
        )
        self._geometry_pick_to_ref = pick_to_ref
        self._geometry_ref_to_pick_ids = {
            reference: tuple(sorted(pick_ids))
            for reference, pick_ids in ref_to_pick_ids.items()
        }

    def _install_mesh_scope_pick_bindings(self) -> None:
        """Build boundary edge/face picking data from the displayed FEM mesh."""

        self._clear_mesh_scope_pick_bindings()
        if (
            self._model is None
            or self._geometry is None
            or _pyvista is None
        ):
            return
        topology = build_mesh_selection_topology(self._model)
        edge_references = tuple(
            reference
            for reference in topology.pick_references("edge")
            if reference.kind == "edge"
        )
        face_references = tuple(
            reference
            for reference in topology.pick_references("face")
            if reference.kind == "face"
        )
        edge_rows = tuple(
            (
                int(reference.element_id),
                int(reference.local_index),
                tuple(reference.node_ids),
            )
            for reference in edge_references
        )
        face_rows = tuple(
            (
                int(reference.element_id),
                int(reference.local_index),
                tuple(reference.node_ids),
            )
            for reference in face_references
        )
        self._mesh_scope_edge_rows = edge_rows
        self._mesh_scope_face_rows = face_rows
        if edge_rows:
            (
                self._mesh_scope_edges,
                self._mesh_scope_edge_cells,
            ) = _mesh_edge_polydata(
                _pyvista,
                self._geometry,
                edge_rows,
            )
            for pick_id, reference in enumerate(edge_references, start=1):
                self._mesh_scope_pick_to_ref[("edge", pick_id)] = reference
                self._mesh_scope_ref_to_pick_id[reference] = pick_id
                self._mesh_scope_identity_to_pick_id[
                    (reference.kind, reference.identity)
                ] = pick_id
        if face_rows:
            self._mesh_scope_faces = _mesh_face_polydata(
                _pyvista,
                self._geometry,
                face_rows,
            )
            for pick_id, reference in enumerate(face_references, start=1):
                self._mesh_scope_pick_to_ref[("face", pick_id)] = reference
                self._mesh_scope_ref_to_pick_id[reference] = pick_id
                self._mesh_scope_identity_to_pick_id[
                    (reference.kind, reference.identity)
                ] = pick_id
        self._mesh_scope_pick_bindings_ready = True

    def _ensure_mesh_scope_pick_bindings(self) -> None:
        if self._mesh_scope_pick_bindings_ready:
            return
        self._install_mesh_scope_pick_bindings()

    def _clear_mesh_scope_pick_bindings(self) -> None:
        self._mesh_scope_edges = None
        self._mesh_scope_faces = None
        self._mesh_scope_edge_rows = ()
        self._mesh_scope_face_rows = ()
        self._mesh_scope_edge_cells = ()
        self._mesh_scope_pick_to_ref.clear()
        self._mesh_scope_ref_to_pick_id.clear()
        self._mesh_scope_identity_to_pick_id.clear()
        self._mesh_scope_pick_bindings_ready = False

    def _reset_mesh_scope_highlight_pipelines(self) -> None:
        self._mesh_scope_render_timer.stop()
        for kind in ("node", "element", "edge", "face"):
            self._remove_actor(f"mesh_scope_selection_{kind}")
        self._mesh_scope_highlight_pipelines.clear()
        self._mesh_scope_highlight_indices.clear()
        self._mesh_scope_highlight_kind = None

    def _install_mesh_scope_highlight_pipelines(self) -> None:
        """Create all four empty selection pipelines once for the installed mesh."""

        self._reset_mesh_scope_highlight_pipelines()
        if (
            self._plotter is None
            or _pyvista is None
            or self._geometry is None
            or self._pick_grid is None
        ):
            return
        import vtk

        payload = self._rendered_result_payload()
        if payload is None:
            node_dataset = _pyvista.PolyData(
                np.asarray(self._geometry.points, dtype=float)
            )
            element_dataset = self._pick_grid
            self._mesh_scope_highlight_indices.update(
                {
                    ("node", (int(node_id), -1)): (int(point_index),)
                    for node_id, point_index in (
                        self._geometry.node_id_to_point_index.items()
                    )
                }
            )
            self._mesh_scope_highlight_indices.update(
                {
                    ("element", (int(element_id), -1)): (int(cell_index),)
                    for element_id, cell_index in (
                        self._geometry.element_id_to_cell_index.items()
                    )
                }
            )
        else:
            result_dataset = payload.dataset
            node_ids = self._typed_result_point_ids(result_dataset)
            cell_ids = self._typed_result_cell_ids(result_dataset)
            node_dataset = (
                _pyvista.PolyData(
                    np.asarray(result_dataset.points, dtype=float)
                )
                if bool(np.any(node_ids > 0))
                else None
            )
            element_dataset = (
                result_dataset.copy(deep=True)
                if bool(np.any(cell_ids > 0))
                else None
            )
            for node_id in np.unique(node_ids[node_ids > 0]):
                self._mesh_scope_highlight_indices[
                    ("node", (int(node_id), -1))
                ] = tuple(
                    int(index)
                    for index in np.flatnonzero(node_ids == node_id)
                )
            for element_id in np.unique(cell_ids[cell_ids > 0]):
                self._mesh_scope_highlight_indices[
                    ("element", (int(element_id), -1))
                ] = tuple(
                    int(index)
                    for index in np.flatnonzero(cell_ids == element_id)
                )
        sources = {
            "node": (node_dataset, vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS),
            "element": (
                element_dataset,
                vtk.vtkDataObject.FIELD_ASSOCIATION_CELLS,
            ),
            "edge": (
                self._mesh_scope_edges,
                vtk.vtkDataObject.FIELD_ASSOCIATION_CELLS,
            ),
            "face": (
                self._mesh_scope_faces,
                vtk.vtkDataObject.FIELD_ASSOCIATION_CELLS,
            ),
        }
        styles: dict[str, dict[str, Any]] = {
            "node": {
                "point_size": 13,
                "render_points_as_spheres": True,
            },
            "element": {"style": "wireframe", "line_width": 3},
            "edge": {"line_width": 5},
            "face": {"opacity": 0.8},
        }
        for kind, (dataset, association) in sources.items():
            if dataset is None:
                continue
            if kind in {"edge", "face"}:
                source = (
                    self._mesh_scope_edge_rows
                    if kind == "edge"
                    else self._mesh_scope_face_rows
                )
                for index, row in enumerate(source):
                    self._mesh_scope_highlight_indices[
                        (kind, (int(row[0]), int(row[1])))
                    ] = (index,)
            mask = np.zeros(
                dataset.n_points
                if association == vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS
                else dataset.n_cells,
                dtype=np.uint8,
            )
            mask_name = f"mesh_scope_{kind}_mask"
            if association == vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS:
                dataset.point_data[mask_name] = mask
                algorithm = vtk.vtkThresholdPoints()
                vtk_mask = dataset.GetPointData().GetArray(mask_name)
            else:
                dataset.cell_data[mask_name] = mask
                algorithm = vtk.vtkThreshold()
                vtk_mask = dataset.GetCellData().GetArray(mask_name)
            algorithm.SetInputData(dataset)
            algorithm.SetInputArrayToProcess(
                0,
                0,
                0,
                association,
                mask_name,
            )
            algorithm.SetUpperThreshold(0.5)
            algorithm.SetThresholdFunction(algorithm.THRESHOLD_UPPER)
            actor_name = f"mesh_scope_selection_{kind}"
            actor = self._plotter.add_mesh(
                algorithm,
                color="#f5a623",
                show_edges=False,
                show_scalar_bar=False,
                name=actor_name,
                reset_camera=False,
                pickable=False,
                **styles[kind],
            )
            actor.SetVisibility(False)
            self._actors[actor_name] = actor
            self._offset_highlight_actor(actor)
            self._mesh_scope_highlight_pipelines[kind] = (
                _MeshScopeHighlightPipeline(
                    kind,
                    dataset,
                    algorithm,
                    np.asarray(
                        dataset.point_data[mask_name]
                        if association
                        == vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS
                        else dataset.cell_data[mask_name]
                    ),
                    vtk_mask,
                    actor,
                )
            )

    def _mesh_scope_highlight_index(self, reference: MeshEntityRef) -> int | None:
        indices = self._mesh_scope_highlight_reference_indices(
            reference
        )
        return None if not indices else int(indices[0])

    def _mesh_scope_highlight_reference_indices(
        self,
        reference: MeshEntityRef,
    ) -> tuple[int, ...]:
        indices = self._mesh_scope_highlight_indices.get(
            (reference.kind, reference.identity),
            (),
        )
        if (
            indices
            or self._rendered_result_payload() is not None
            or self._geometry is None
        ):
            return indices
        if reference.kind == "node":
            index = self._geometry.node_id_to_point_index.get(
                int(reference.node_id)
            )
        elif reference.kind == "element":
            index = self._geometry.element_id_to_cell_index.get(
                int(reference.element_id)
            )
        else:
            pick_id = self._mesh_scope_identity_to_pick_id.get(
                (reference.kind, reference.identity)
            )
            index = None if pick_id is None else int(pick_id) - 1
        return () if index is None else (int(index),)

    def _schedule_mesh_scope_render(self) -> None:
        if self._plotter is not None and not self._mesh_scope_render_timer.isActive():
            self._mesh_scope_render_timer.start()

    def _render_mesh_scope_highlight(self) -> None:
        if self._plotter is not None:
            self._render()

    def _clear_mesh_scope_highlight(self, *, schedule_render: bool) -> bool:
        changed = self._mesh_scope_highlight_kind is not None
        for pipeline in self._mesh_scope_highlight_pipelines.values():
            changed = pipeline.clear() or changed
            pipeline.actor.SetVisibility(False)
        self._mesh_scope_highlight_kind = None
        if changed and schedule_render:
            self._schedule_mesh_scope_render()
        return changed

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
        current = self._result_render_payload
        can_reuse = (
            current is not None
            and self._result_grid is current.dataset
            and "result" in self._actors
        )
        if can_reuse:
            checked, reused = _reuse_result_render_dataset(
                current,
                checked,
                candidate_validated=True,
            )
        else:
            reused = False
        self._result_render_payload = checked
        self._scalar_reuse_pending = reused
        self._scalar_reuse_display = (
            self._display if reused else None
        )
        self._result_render_validated_mtime = int(
            checked.dataset.GetMTime()
        )
        self._result_install_validation = (
            checked,
            self._result_render_validated_mtime,
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
        layout = (
            topology.cells,
            topology.point_locations,
            topology.cell_locations,
        )
        cached = self._result_provenance_layout
        if (
            cached is not None
            and cached[0] is layout[0]
            and cached[1] is layout[1]
            and cached[2] is layout[2]
        ):
            return
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
        self._result_provenance_layout = layout

    def _rendered_result_payload(self) -> ResultRenderPayload | None:
        """Return the rendered payload, honoring VTK-reported modifications.

        Mutations that bypass VTK's ``Modified`` protocol are detected at the
        next unconditional install/render boundary, not on every hover read.
        """

        payload = self._result_render_payload
        if (
            payload is None
            or self._scalar_reuse_pending
            or self._result_grid is not payload.dataset
        ):
            return None
        modified = int(payload.dataset.GetMTime())
        if self._result_render_validated_mtime != modified:
            payload = _require_result_render_payload(payload)
            self._result_render_validated_mtime = modified
        return payload

    def _consume_result_install_validation(
        self,
        payload: ResultRenderPayload,
    ) -> ResultRenderPayload:
        token = self._result_install_validation
        self._result_install_validation = None
        if (
            token is not None
            and token[0] is payload
            and token[1] == int(payload.dataset.GetMTime())
        ):
            return payload
        return _require_result_render_payload(payload)

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

    def _project_mesh_scope_datasets_to_result(
        self,
        payload: ResultRenderPayload,
    ) -> None:
        """Place mesh topology pick data on the currently rendered result."""

        if self._geometry is None or _pyvista is None:
            self._mesh_scope_edges = None
            self._mesh_scope_faces = None
            self._mesh_scope_edge_cells = ()
            return
        coordinates: dict[int, np.ndarray] = {}
        for index, location in enumerate(payload.topology.point_locations):
            if location is None or location.node_id is None:
                continue
            coordinates.setdefault(
                int(location.node_id),
                np.asarray(payload.dataset.points[index], dtype=float),
            )
        projected_points = np.asarray(self._geometry.points, dtype=float).copy()
        for node_id, point_index in self._geometry.node_id_to_point_index.items():
            point = coordinates.get(int(node_id))
            if point is not None:
                projected_points[int(point_index)] = point
        edge_node_ids = {
            int(node_id)
            for _element_id, _local_index, node_ids in self._mesh_scope_edge_rows
            for node_id in node_ids
        }
        face_node_ids = {
            int(node_id)
            for _element_id, _local_index, node_ids in self._mesh_scope_face_rows
            for node_id in _face_display_node_ids(node_ids)
        }
        if self._mesh_scope_edge_rows and edge_node_ids.issubset(coordinates):
            (
                self._mesh_scope_edges,
                self._mesh_scope_edge_cells,
            ) = _mesh_edge_polydata(
                _pyvista,
                self._geometry,
                self._mesh_scope_edge_rows,
                points=projected_points,
            )
        else:
            self._mesh_scope_edges = None
            self._mesh_scope_edge_cells = ()
        self._mesh_scope_faces = (
            _mesh_face_polydata(
                _pyvista,
                self._geometry,
                self._mesh_scope_face_rows,
                points=projected_points,
            )
            if self._mesh_scope_face_rows
            and face_node_ids.issubset(coordinates)
            else None
        )

    def set_selection_mode(self, mode: str) -> None:
        previous = self._selection_mode
        had_preselection = (
            self._hover_hit is not None or "preselection" in self._actors
        )
        if mode in {
            "geometry_point", "geometry_edge", "geometry_face", "geometry_body",
            "mesh_node", "mesh_edge", "mesh_face", "mesh_element", "mesh_body",
        }:
            self._selection_mode = mode
        else:
            self._selection_mode = "element" if mode == "element" else "node"
        if self._selection_mode in {"mesh_edge", "mesh_face"}:
            self._ensure_mesh_scope_pick_bindings()
        if previous != self._selection_mode:
            self._clear_preselection(render=False)
            highlighted_kind = self._mesh_scope_highlight_kind
            active_mesh_kind = (
                self._selection_mode.removeprefix("mesh_")
                if self._selection_mode.startswith("mesh_")
                else None
            )
            if (
                highlighted_kind is not None
                and highlighted_kind != active_mesh_kind
            ):
                self._clear_mesh_scope_highlight(schedule_render=True)
        points_actor = self._actors.get("geometry_points")
        points_visibility_changed = False
        if points_actor is not None:
            desired_visibility = (
                self._geometry_preview is not None
                and (
                    self._geometry_preview.topological_dimension == 1
                    or self._selection_mode == "geometry_point"
                )
            )
            try:
                points_visibility_changed = (
                    bool(points_actor.GetVisibility())
                    != desired_visibility
                )
            except (AttributeError, TypeError):
                points_visibility_changed = True
            points_actor.SetVisibility(desired_visibility)
        self._update_pickable_actors()
        if had_preselection or points_visibility_changed:
            self._render()

    def clear_selection(self) -> None:
        visual_names = {
            "selection",
            "geometry_selection",
            "preselection",
        }
        had_mesh_scope_selection = any(
            pipeline.selected_indices
            for pipeline in self._mesh_scope_highlight_pipelines.values()
        )
        had_visible_selection = any(
            name in visual_names or name.startswith("beam_frame_")
            for name in self._actors
        )
        self._hide_selection_rubber_band()
        self._selected_kind = None
        self._selected_id = None
        self._selection_highlight_visible = True
        self._mesh_scope_selected_references.clear()
        self._remove_actor("selection")
        self._remove_actor("geometry_selection")
        self._clear_mesh_scope_highlight(schedule_render=False)
        self._clear_beam_frame_preview(render=False)
        self._clear_preselection(render=False)
        self._update_pickable_actors()
        if had_visible_selection:
            self._render()
        elif had_mesh_scope_selection:
            self._schedule_mesh_scope_render()

    def highlight_geometry(self, reference: LogicalEntityRef) -> None:
        """Highlight one stable logical preview reference."""
        self.highlight_geometry_entities((reference,))

    def highlight_body_boolean_operands(
        self,
        target: LogicalEntityRef | None,
        tool: LogicalEntityRef | None,
    ) -> None:
        """Show detached target/tool Body roles in blue and orange."""

        self.clear_body_boolean_highlights(render=False)
        for actor_name, reference, color in (
            ("boolean_target", target, "#2f80ed"),
            ("boolean_tool", tool, "#f5a623"),
        ):
            if reference is None:
                continue
            if type(reference) is not LogicalEntityRef or reference.kind != "body":
                raise TypeError("Boolean operand highlight requires Body references")
            data = self._body_highlight_data(reference)
            if data is None or self._plotter is None:
                continue
            mesh, kwargs = data
            self._actors[actor_name] = self._plotter.add_mesh(
                mesh,
                color=color,
                show_edges=False,
                show_scalar_bar=False,
                name=actor_name,
                reset_camera=False,
                **kwargs,
            )
            self._offset_highlight_actor(self._actors[actor_name])
        self._update_pickable_actors()
        self._render()

    def clear_body_boolean_highlights(self, *, render: bool = True) -> None:
        """Remove transient strict Boolean target/tool actors."""

        had_actor = any(
            name in self._actors
            for name in ("boolean_target", "boolean_tool")
        )
        self._remove_actor("boolean_target")
        self._remove_actor("boolean_tool")
        self._update_pickable_actors()
        if render and had_actor:
            self._render()

    def _body_highlight_data(self, reference: LogicalEntityRef):
        pick_ids = tuple(
            sorted(self._geometry_ref_to_pick_ids.get(reference, ()))
        )
        if not pick_ids or self._geometry_preview is None:
            return None
        if self._geometry_preview_surface is not None:
            if (
                "geometry_body_pick_id"
                not in self._geometry_preview_surface.cell_data
            ):
                return self._geometry_preview_surface, {"opacity": 0.45}
            ids = np.asarray(
                self._geometry_preview_surface.cell_data[
                    "geometry_body_pick_id"
                ],
                dtype=np.int64,
            )
            cells = np.flatnonzero(np.isin(ids, pick_ids))
            if not len(cells):
                return None
            return (
                self._geometry_preview_surface.extract_cells(cells),
                {"opacity": 0.45},
            )
        if (
            self._geometry_preview.topological_dimension == 1
            and self._geometry_preview_edges is not None
        ):
            ids = np.asarray(
                self._geometry_preview_edges.cell_data[
                    "geometry_body_pick_id"
                ],
                dtype=np.int64,
            )
            cells = np.flatnonzero(np.isin(ids, pick_ids))
            if not len(cells):
                return None
            return (
                self._geometry_preview_edges.extract_cells(cells),
                {"line_width": 6},
            )
        return None

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
        elif kind == "geometry_body":
            if self._geometry_preview_surface is not None:
                if (
                    "geometry_body_pick_id"
                    not in self._geometry_preview_surface.cell_data
                ):
                    data = self._geometry_preview_surface
                else:
                    ids = np.asarray(
                        self._geometry_preview_surface.cell_data[
                            "geometry_body_pick_id"
                        ],
                        dtype=np.int64,
                    )
                    cells = np.flatnonzero(np.isin(ids, pick_ids))
                    if not len(cells):
                        return
                    data = self._geometry_preview_surface.extract_cells(cells)
                kwargs = {"opacity": 0.45}
            elif (
                self._geometry_preview is not None
                and self._geometry_preview.topological_dimension == 1
                and self._geometry_preview_edges is not None
            ):
                ids = np.asarray(
                    self._geometry_preview_edges.cell_data[
                        "geometry_body_pick_id"
                    ],
                    dtype=np.int64,
                )
                cells = np.flatnonzero(np.isin(ids, pick_ids))
                if not len(cells):
                    return
                data = self._geometry_preview_edges.extract_cells(cells)
                kwargs = {"line_width": 6}
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

    def highlight_mesh_entities(
        self,
        references: Iterable[MeshEntityRef],
        *,
        changed_references: Iterable[MeshEntityRef] | None = None,
        entity_kind: str | None = None,
    ) -> None:
        """Update one persistent mesh-selection mask by its changed references."""

        selected = references if isinstance(references, set) else set(references)
        if not selected:
            self._mesh_scope_selected_references.clear()
            self._clear_mesh_scope_highlight(schedule_render=True)
            return
        if entity_kind is None:
            kinds = {reference.kind for reference in selected}
            if len(kinds) != 1:
                raise ValueError("mesh scope highlight requires one entity kind")
            kind = next(iter(kinds))
        else:
            kind = str(entity_kind)
            if kind not in {"node", "element", "edge", "face"}:
                raise ValueError("unsupported mesh scope highlight kind")
        self._mesh_scope_selected_references = set(selected)
        pipeline = self._mesh_scope_highlight_pipelines.get(kind)
        if pipeline is None:
            return
        visibility_changed = self._mesh_scope_highlight_kind != kind
        if visibility_changed:
            self._clear_mesh_scope_highlight(schedule_render=False)
            pipeline.actor.SetVisibility(True)
            self._mesh_scope_highlight_kind = kind
        changes = None
        if changed_references is not None and not visibility_changed:
            changes = {}
            for reference in changed_references:
                for index in self._mesh_scope_highlight_reference_indices(
                    reference
                ):
                    changes[index] = reference in selected
        else:
            target = {
                index
                for reference in selected
                for index in self._mesh_scope_highlight_reference_indices(
                    reference
                )
            }
        updated = (
            pipeline.update_changes(changes)
            if changes is not None
            else pipeline.update(target)
        )
        if updated or visibility_changed:
            self._schedule_mesh_scope_render()

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
        previous_coordinate_system = bool(
            self._contour["show_coordinate_system"]
        )
        self._contour.update(options)
        coordinate_system_changed = (
            bool(self._contour["show_coordinate_system"])
            != previous_coordinate_system
        )
        if coordinate_system_changed:
            self._refresh_coordinate_system_axes()
        if (
            "edges" in options
            and self._display.contour_enabled
        ):
            self._show_edges = bool(options["edges"])
        if self._display.contour_enabled:
            self._update_result_layer()
        elif coordinate_system_changed:
            self._render()

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

    def set_symbol_sampling_density_override(
        self,
        density: str | None,
    ) -> None:
        if density not in {None, "low", "medium", "high"}:
            raise ValueError("符号采样密度必须是 low、medium、high 或 None")
        self._symbol_sampling_density_override = density

    def _effective_symbol_sampling_density(self) -> str:
        return (
            self._symbol_sampling_density_override
            or self._symbol_settings.sampling_density
        )

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

    def screenshot_size(self) -> tuple[int, int]:
        """返回截图使用的当前 VTK 视口像素尺寸。"""
        if self._plotter is not None:
            window_size = getattr(self._plotter, "window_size", None)
            if window_size is not None and len(window_size) >= 2:
                width, height = int(window_size[0]), int(window_size[1])
                if width > 0 and height > 0:
                    return width, height
        return max(1, self.width()), max(1, self.height())

    def save_screenshot(
        self,
        path: str,
        *,
        scale: int = 1,
        window_size: tuple[int, int] | None = None,
        transparent_background: bool = False,
    ) -> None:
        """通过 VTK 帧缓冲保存当前视口。"""
        if self._plotter is None:
            raise RuntimeError("三维视口尚未初始化")
        plotter = self._plotter
        current_size = self.screenshot_size()
        target_size = window_size
        if target_size is None and scale > 1:
            target_size = (
                current_size[0] * scale,
                current_size[1] * scale,
            )
        text_scale = (
            1.0
            if target_size is None
            else max(
                1.0,
                min(
                    target_size[0] / current_size[0],
                    target_size[1] / current_size[1],
                ),
            )
        )
        camera = plotter.camera
        camera_state = camera.copy()
        renderer = plotter.renderer
        try:
            scalar_bar_fonts = self._scale_scalar_bar_fonts(text_scale)
            background_state = None
            try:
                if transparent_background:
                    background_state = (
                        renderer.GetGradientBackground(),
                        renderer.GetBackground(),
                        renderer.GetBackground2(),
                        renderer.GetBackgroundAlpha(),
                    )
                    renderer.GradientBackgroundOff()
                    renderer.SetBackgroundAlpha(0.0)
                if target_size is None:
                    plotter.render()
                    plotter.screenshot(
                        path,
                        scale=1,
                        window_size=None,
                        transparent_background=transparent_background,
                        return_img=False,
                    )
                else:
                    self._save_offscreen_screenshot(
                        path,
                        target_size,
                        transparent_background=transparent_background,
                    )
            finally:
                if background_state is not None:
                    gradient, background, background2, background_alpha = (
                        background_state
                    )
                    renderer.SetBackground(*background)
                    renderer.SetBackground2(*background2)
                    renderer.SetBackgroundAlpha(background_alpha)
                    renderer.SetGradientBackground(gradient)
                self._restore_scalar_bar_fonts(scalar_bar_fonts)
        finally:
            camera.DeepCopy(camera_state)
            camera.Modified()
            self._render()

    def _save_offscreen_screenshot(
        self,
        path: str,
        window_size: tuple[int, int],
        *,
        transparent_background: bool,
    ) -> None:
        """将现有渲染层临时放入独立窗口，不改变屏幕上的 Qt 视口。"""
        import vtk

        plotter = self._plotter
        source_window = plotter.render_window
        renderer_collection = source_window.GetRenderers()
        renderer_collection.InitTraversal()
        renderers = [
            renderer_collection.GetNextItem()
            for _ in range(renderer_collection.GetNumberOfItems())
        ]

        export_window = vtk.vtkRenderWindow()
        export_window.SetOffScreenRendering(1)
        export_window.SetSize(int(window_size[0]), int(window_size[1]))
        export_window.SetDPI(source_window.GetDPI())
        export_window.SetNumberOfLayers(source_window.GetNumberOfLayers())
        export_window.SetMultiSamples(source_window.GetMultiSamples())
        export_window.SetAlphaBitPlanes(
            1 if transparent_background else source_window.GetAlphaBitPlanes()
        )
        try:
            for renderer in renderers:
                source_window.RemoveRenderer(renderer)
                export_window.AddRenderer(renderer)
            export_window.Render()

            capture = vtk.vtkWindowToImageFilter()
            capture.SetInput(export_window)
            if transparent_background:
                capture.SetInputBufferTypeToRGBA()
            else:
                capture.SetInputBufferTypeToRGB()
            capture.ReadFrontBufferOn()
            capture.Update()

            suffix = os.path.splitext(path)[1].lower()
            if suffix == ".png":
                writer = vtk.vtkPNGWriter()
            elif suffix in {".jpg", ".jpeg"}:
                writer = vtk.vtkJPEGWriter()
                writer.SetQuality(95)
            else:
                raise ValueError("仅支持导出 PNG 或 JPEG 图片")
            writer.SetFileName(path)
            writer.SetInputConnection(capture.GetOutputPort())
            writer.Write()
        finally:
            for renderer in renderers:
                export_window.RemoveRenderer(renderer)
                source_window.AddRenderer(renderer)
            export_window.Finalize()

    def _scale_scalar_bar_fonts(
        self,
        factor: float,
    ) -> list[tuple[Any, Any, Any, Any, int, int, int, int, float, int]]:
        scalar_bars = getattr(self._plotter, "scalar_bars", None)
        if scalar_bars is None or factor <= 1.0:
            return []
        states: list[
            tuple[Any, Any, Any, Any, int, int, int, int, float, int]
        ] = []
        for scalar_bar in scalar_bars.values():
            title = scalar_bar.GetTitleTextProperty()
            labels = scalar_bar.GetLabelTextProperty()
            annotation_text = scalar_bar.GetAnnotationTextProperty()
            title_size = title.GetFontSize()
            label_size = labels.GetFontSize()
            annotation_size = annotation_text.GetFontSize()
            text_pad = scalar_bar.GetTextPad()
            annotation_pad = scalar_bar.GetAnnotationLeaderPadding()
            title_separation = scalar_bar.GetVerticalTitleSeparation()
            states.append(
                (
                    scalar_bar,
                    title,
                    labels,
                    annotation_text,
                    title_size,
                    label_size,
                    annotation_size,
                    text_pad,
                    annotation_pad,
                    title_separation,
                )
            )
            title.SetFontSize(round(title_size * factor))
            labels.SetFontSize(round(label_size * factor))
            annotation_text.SetFontSize(round(annotation_size * factor))
            scalar_bar.SetTextPad(round(text_pad * factor))
            scalar_bar.SetAnnotationLeaderPadding(annotation_pad * factor)
            scalar_bar.SetVerticalTitleSeparation(
                round(title_separation * factor)
            )
        return states

    @staticmethod
    def _restore_scalar_bar_fonts(
        states: list[
            tuple[Any, Any, Any, Any, int, int, int, int, float, int]
        ],
    ) -> None:
        for (
            scalar_bar,
            title,
            labels,
            annotation_text,
            title_size,
            label_size,
            annotation_size,
            text_pad,
            annotation_pad,
            title_separation,
        ) in states:
            title.SetFontSize(title_size)
            labels.SetFontSize(label_size)
            annotation_text.SetFontSize(annotation_size)
            scalar_bar.SetTextPad(text_pad)
            scalar_bar.SetAnnotationLeaderPadding(annotation_pad)
            scalar_bar.SetVerticalTitleSeparation(title_separation)

    def set_background_settings(self, settings: ViewportBackgroundSettings) -> None:
        """更新视口背景和依赖背景对比度的显示层。"""
        self._background_settings = settings.normalized()
        self._update_background_stylesheet()
        if self._plotter is None:
            return
        self._apply_plotter_background()
        palette = self._visual_palette()
        for name, color in (
            ("mesh_surface", self._mesh_layer_color(palette)),
            ("element_edges", self._element_layer_color(palette)),
            ("result_edges", self._background_settings.foreground_color),
            ("nodes", self._node_layer_color(palette)),
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
        if self._display.contour_enabled:
            base_edges = self._actors.get("element_edges")
            if base_edges is not None:
                base_edges.SetVisibility(False)
            result_edges = self._actors.get("result_edges")
            if (
                self._show_edges
                and result_edges is None
                and self._plotter is not None
                and self._result_grid is not None
            ):
                result_edges = self._add_result_edges_layer(
                    self._result_grid
                )
            if result_edges is not None:
                result_edges.SetVisibility(self._show_edges)
            if render:
                self._render()
            return

        result_edges = self._actors.get("result_edges")
        if result_edges is not None:
            result_edges.SetVisibility(False)
        actor = self._actors.get("element_edges")
        if (
            self._show_edges
            and actor is None
            and self._plotter is not None
            and self._grid is not None
        ):
            actor = self._add_element_edges_layer()
        if actor is not None:
            actor.SetVisibility(self._show_edges)
        if render:
            self._render()

    def set_nodes_visible(self, visible: bool, *, render: bool = True) -> None:
        self._show_nodes = bool(visible)
        self._refresh_node_layer(render=render)

    def set_node_labels_visible(self, visible: bool) -> None:
        self._show_node_labels = bool(visible)
        self._refresh_labels()

    def set_element_labels_visible(self, visible: bool) -> None:
        self._show_element_labels = bool(visible)
        self._refresh_labels()

    def _fit_bounds(self) -> tuple[float, float, float, float, float, float] | None:
        """Return bounds for the displayed model, excluding auxiliary actors."""

        points: object | None = None
        if self._sketch_authoring_active:
            data = self._sketch_draft_render_data
            return _sketch_camera_bounds(
                () if data is None else data.points,
                self._sketch_grid_spacing,
                None if data is None else data.plane,
            )
        if self._wire_authoring_active:
            if (
                self._wire_draft_render_data is None
                or not self._wire_draft_render_data.points
            ):
                return None
            points = self._wire_draft_render_data.points
        elif self._geometry_preview is not None:
            points = self._geometry_preview.points
        elif self._result_grid is not None and "result" in self._actors:
            points = self._result_grid.points
        elif self._grid is not None:
            points = self._grid.points
        if points is None:
            return None

        values = np.asarray(points, dtype=float)
        if values.size == 0 or values.size % 3:
            return None
        values = values.reshape((-1, 3))
        values = values[np.all(np.isfinite(values), axis=1)]
        if len(values) == 0:
            return None

        minimum = np.min(values, axis=0)
        maximum = np.max(values, axis=0)
        if not np.any(maximum > minimum):
            reference = max(float(np.max(np.abs(values))), 1.0)
            padding = reference * 1.0e-6
            minimum -= padding
            maximum += padding
        return tuple(
            float(value)
            for axis in range(3)
            for value in (minimum[axis], maximum[axis])
        )

    def _reset_camera_to_fit(self) -> None:
        """Frame stable display bounds without including auxiliary actors."""

        if self._plotter is None:
            return
        bounds = self._fit_bounds()
        if bounds is None:
            self._plotter.reset_camera(render=False)
        else:
            self._plotter.reset_camera(bounds=bounds, render=False)

    def fit(self) -> None:
        if self._plotter is not None:
            self._reset_camera_to_fit()
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
        views = {
            # Each tuple is (focal-to-camera direction, screen-up axis).
            # The directions deliberately follow the arrows in the coordinate
            # PNGs: positive X is left in XY/XZ, positive Y is left in YZ, and
            # the paired labels swap the screen axes.
            "front": ((0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),   # XZ
            "back": ((0.0, -1.0, 0.0), (1.0, 0.0, 0.0)),   # ZX
            "left": ((-1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),   # YZ
            "right": ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),   # ZY
            "top": ((0.0, 0.0, -1.0), (0.0, 1.0, 0.0)),    # XY
            "bottom": ((0.0, 0.0, 1.0), (1.0, 0.0, 0.0)),  # YX
            "iso": ((1.0, 1.0, 1.0), (-1.0, 2.0, -1.0)),
        }
        direction, up = views[view]
        camera = self._plotter.camera
        focal = np.asarray(camera.GetFocalPoint(), dtype=float)
        position = np.asarray(camera.GetPosition(), dtype=float)
        distance = float(np.linalg.norm(position - focal))
        direction_array = np.asarray(direction, dtype=float)
        direction_array /= np.linalg.norm(direction_array)
        camera.SetPosition(*(focal + direction_array * max(distance, 1.0)))
        camera.SetViewUp(*up)
        # VTK orthogonalizes the view-up vector against the current view
        # direction, avoiding a roll that would make the PNG legend disagree
        # with the actual screen axes.
        camera.OrthogonalizeViewUp()
        self._reset_camera_to_fit()
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
            "gravity",
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
        sampling_density = self._effective_symbol_sampling_density()
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
        model_center = 0.5 * (
            np.min(self._geometry.points, axis=0)
            + np.max(self._geometry.points, axis=0)
        )
        if selected_definition is not None and selected_definition.gravity_loads:
            gravity_vector = np.zeros(3, dtype=float)
            for gravity_load in selected_definition.gravity_loads:
                acceleration = np.asarray(
                    gravity_load.acceleration,
                    dtype=float,
                ).reshape(-1)
                gravity_vector[:min(3, len(acceleration))] += acceleration[:3]
            self._add_gravity_arrow(
                model_center,
                gravity_vector,
                load_scale,
            )
        is_3d = bool(self._model.mesh.nodes and hasattr(self._model.mesh.nodes[0], "z"))
        translation_count = 3 if is_3d else 2
        constraint_points: list[np.ndarray] = []
        constraint_vectors: list[np.ndarray] = []
        rotation_points: list[np.ndarray] = []
        rotation_axes: list[np.ndarray] = []
        constraint_label_points: dict[int, np.ndarray] = {}
        constraint_labels_by_node: dict[int, list[str]] = {}
        if settings.show_constraints:
            boundary_definitions = effective_step_boundaries(
                self._model,
                selected_definition,
            )
            constraints_by_target: dict[
                tuple[str, str | int],
                dict[int, float],
            ] = {}
            representative_by_target: dict[
                tuple[str, str | int],
                object,
            ] = {}
            for definition in boundary_definitions:
                target_key = (
                    displacement_target_kind(definition),
                    definition.target,
                )
                representative_by_target[target_key] = definition
                components = constraints_by_target.setdefault(target_key, {})
                for component in range(definition.first_component - 1, definition.last_component):
                    components[component] = float(definition.value)
            for target_key, components in constraints_by_target.items():
                try:
                    node_ids = resolve_displacement_node_ids(
                        self._model,
                        representative_by_target[target_key],
                    )
                except (KeyError, TypeError, ValueError):
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
                        candidate_points, sampling_density
                    )
                    camera_position = self._camera_position()
                    for selected_index in selected:
                        node_id = region_node_ids[int(selected_index)]
                        base = self._geometry.points[
                            self._geometry.node_id_to_point_index[node_id]
                        ]
                        display_base = base + camera_facing_offset(
                            base, camera_position, 0.04 * constraint_scale
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
                            outward = constraint_outward_direction(
                                base, model_center, component
                            )
                            tip = display_base + 0.08 * constraint_scale * outward
                            constraint_points.append(
                                tip + constraint_scale * outward
                            )
                            constraint_vectors.append(-outward)
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
            cloud["constraint_scale"] = np.full(
                len(constraint_points), constraint_scale
            )
            marker = _pyvista.Arrow(
                start=(0.0, 0.0, 0.0),
                direction=(1.0, 0.0, 0.0),
                tip_length=0.34,
                tip_radius=constraint_radius / constraint_scale,
                tip_resolution=16,
                shaft_radius=0.028,
                shaft_resolution=12,
            )
            glyphs = cloud.glyph(
                orient="directions",
                scale="constraint_scale",
                factor=1.0,
                geom=marker,
            )
            self._actors["constraints"] = self._plotter.add_mesh(
                glyphs,
                color=settings.constraint_color,
                lighting=False,
                name="constraints",
                reset_camera=False,
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
        load_starts_at_anchor: list[bool] = []
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
                load_starts_at_anchor.append(False)
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
            selected = region_sample_indices(candidate_array, sampling_density)
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
                load_starts_at_anchor.append(kind == "edge")
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
                    samples = sample_polyline(points, sampling_density)
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
                        load_starts_at_anchor.append(False)
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
                lighting=False, name="load_regions", reset_camera=False,
            )
        if edge_lines:
            region_edges = _line_only_polydata(
                _pyvista,
                self._current_points(),
                edge_lines,
            )
            self._actors["load_region_edges"] = self._plotter.add_mesh(
                region_edges, color=settings.load_color, line_width=3,
                lighting=False, name="load_region_edges", reset_camera=False,
            )
        self._add_load_arrows(
            origins,
            vectors,
            load_starts_at_anchor,
            load_labels,
            load_scale,
        )
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
            poly, color=color, line_width=2, lighting=False,
            name=name, reset_camera=False,
        )
        head_name = f"{name[:-1]}_heads" if name.endswith("s") else f"{name}_heads"
        self._actors[head_name] = self._plotter.add_arrows(
            np.asarray(heads), np.asarray(tangents), mag=0.22 * radius,
            color=color, lighting=False, name=head_name, reset_camera=False,
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
            poly, color=color, line_width=2, lighting=False,
            name=name, reset_camera=False,
        )

    def _add_load_arrows(
        self,
        anchors: list[np.ndarray],
        vectors: list[np.ndarray],
        starts_at_anchor: list[bool],
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
        anchor_points = np.asarray(anchors)
        displayed_lengths = arrow_lengths * glyph_scale
        display_origins = load_arrow_origins(
            anchor_points,
            directions,
            displayed_lengths,
            np.asarray(starts_at_anchor),
        )
        arrow_points = _pyvista.PolyData(display_origins)
        arrow_points["directions"] = directions
        arrow_points["arrow_scale"] = displayed_lengths
        arrow = _pyvista.Arrow(
            start=(0.0, 0.0, 0.0), direction=(1.0, 0.0, 0.0),
            tip_length=0.20, tip_radius=0.07, tip_resolution=8,
            shaft_radius=0.015, shaft_resolution=8,
        )
        arrow_glyphs = arrow_points.glyph(
            orient="directions", scale="arrow_scale", factor=1.0, geom=arrow,
        )
        self._actors["loads"] = self._plotter.add_mesh(
            arrow_glyphs, color=settings.load_color, lighting=False,
            name="loads", reset_camera=False,
        )
        if settings.show_values and anchors:
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

    def _add_gravity_arrow(
        self,
        center: np.ndarray,
        acceleration: np.ndarray,
        glyph_scale: float,
    ) -> None:
        magnitude = float(np.linalg.norm(acceleration))
        if magnitude <= 0.0 or self._plotter is None:
            return
        direction = np.asarray(acceleration, dtype=float) / magnitude
        origin = np.asarray(center, dtype=float) - 0.5 * glyph_scale * direction
        actor = self._plotter.add_arrows(
            origin.reshape((1, 3)),
            direction.reshape((1, 3)),
            mag=glyph_scale,
            color=_GRAVITY_SYMBOL_COLOR,
            lighting=False,
            name="gravity",
            reset_camera=False,
        )
        actor.SetPickable(False)
        self._actors["gravity"] = actor

    def closeEvent(self, event) -> None:
        self.shutdown_backend()
        super().closeEvent(event)

    def shutdown_backend(self) -> None:
        """Release the native render backend exactly once."""

        plotter = self._plotter
        if plotter is None:
            return
        self._plotter = None
        self._stack.setCurrentWidget(self._message)
        try:
            self._stack.removeWidget(plotter)
        except (RuntimeError, TypeError):
            pass
        try:
            plotter.close()
        except Exception:
            pass

    def _ensure_plotter(self) -> bool:
        if self._plotter is not None:
            self._stack.setCurrentWidget(self._plotter)
            self.nativeSurfaceUpdated.emit()
            return True
        pv, interactor, error = load_backend()
        if pv is None or interactor is None:
            self._message.setText(f"三维视口无法加载：{error}")
            return False
        kwargs: dict[str, object] = {}
        try:
            parameters = inspect.signature(interactor).parameters
            if "auto_update" in parameters:
                kwargs["auto_update"] = False
            if "off_screen" in parameters and is_offscreen_environment():
                kwargs["off_screen"] = True
        except (TypeError, ValueError):
            pass
        try:
            self._plotter = interactor(self, **kwargs)
            self._apply_plotter_background()
            self._stack.addWidget(self._plotter)
            self._stack.setCurrentWidget(self._plotter)
            self._install_picker()
            self.nativeSurfaceUpdated.emit()
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
        if self._is_line_mesh():
            return _LINE_ELEMENT_WIDTH
        return 1

    def _is_line_mesh(self) -> bool:
        if self._geometry is None:
            return False
        declared = getattr(self._geometry, "is_line_mesh", None)
        if declared is not None:
            return bool(declared)
        cell_types = np.asarray(getattr(self._geometry, "cell_types", ()))
        return bool(len(cell_types)) and bool(np.all(cell_types == 3))

    def _line_render_options(self) -> dict[str, bool]:
        if self._is_line_mesh():
            return {"render_lines_as_tubes": True}
        return {}

    def _node_point_size(self) -> int:
        return _LINE_NODE_POINT_SIZE if self._is_line_mesh() else 7

    def _mesh_layer_color(self, palette: dict[str, str]) -> str:
        return (
            self._element_layer_color(palette)
            if self._is_line_mesh()
            else palette["mesh"]
        )

    @staticmethod
    def _element_layer_color(palette: dict[str, str]) -> str:
        return palette["element"]

    def _node_layer_color(self, palette: dict[str, str]) -> str:
        return palette["node"]

    def _add_element_edges_layer(self) -> Any:
        palette = self._visual_palette()
        actor = self._plotter.add_mesh(
            self._grid,
            color=self._element_layer_color(palette),
            style="wireframe",
            line_width=self._element_line_width(),
            lighting=False,
            name="element_edges",
            reset_camera=False,
            **self._line_render_options(),
        )
        self._actors["element_edges"] = actor
        return actor

    def _add_base_layers(self, reset_camera: bool, *, render: bool = True) -> None:
        palette = self._visual_palette()
        line_options = self._line_render_options()
        mesh_color = self._mesh_layer_color(palette)
        self._actors["mesh_surface"] = self._plotter.add_mesh(
            self._grid, color=mesh_color, show_edges=False, name="mesh_surface",
            line_width=self._element_line_width(), lighting=False,
            reset_camera=False,
            **line_options,
        )
        if self._show_edges:
            self._add_element_edges_layer()
        self._refresh_node_layer(render=False)
        self._refresh_labels(render=False)
        if reset_camera:
            self._reset_camera_to_fit()
        if render:
            self._render()

    def _update_result_layer(self) -> None:
        if (
            self._scalar_reuse_pending
            and self._scalar_reuse_display == self._display
            and self._update_reused_scalar_layer()
        ):
            self._scalar_reuse_pending = False
            self._scalar_reuse_display = None
            return
        self._scalar_reuse_pending = False
        self._scalar_reuse_display = None
        self._remove_actor("result")
        self._remove_actor("result_edges")
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
        self._result_provenance_layout = None
        self._result_install_validation = None

    def _update_result_render_payload_layer(
        self,
        payload: ResultRenderPayload,
    ) -> None:
        """Render an already-selected and already-deformed typed payload."""

        checked = self._consume_result_install_validation(payload)
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
        self._project_mesh_scope_datasets_to_result(checked)
        self._remove_actor("set_highlight")
        self._remove_actor("selection")
        base = self._actors.get("mesh_surface")
        if base is not None:
            base.SetVisibility(False)
        base_edges = self._actors.get("element_edges")
        if base_edges is not None:
            base_edges.SetVisibility(
                self._show_edges
                and not self._display.contour_enabled
            )
        kwargs: dict[str, Any] = {
            "name": "result",
            "reset_camera": False,
            "line_width": self._element_line_width(),
            "show_edges": False,
            **self._line_render_options(),
            **contour_surface_options(
                str(self._contour["render_mode"]),
                is_line_mesh=self._is_line_mesh(),
            ),
        }
        if self._display.contour_enabled:
            color_count = (
                256
                if self._contour.get("style") == "continuous"
                else int(self._contour["levels"])
            )
            kwargs.update(
                scalars=checked.scalar_name,
                cmap=resolve_contour_colormap(
                    str(self._contour["colormap"]),
                    color_count,
                ),
                n_colors=color_count,
                interpolate_before_map=self._contour.get("style")
                == "continuous",
                show_scalar_bar=self._contour["legend"],
                scalar_bar_args=self._contour_bar_args(checked),
            )
            if self._contour["manual"]:
                kwargs["clim"] = (
                    self._contour["minimum"],
                    self._contour["maximum"],
                )
        else:
            kwargs["color"] = self._visual_palette()["result"]
        self._actors["result"] = self._plotter.add_mesh(dataset, **kwargs)
        if self._display.contour_enabled and self._contour["legend"]:
            self._configure_contour_bar(
                checked,
                getattr(self._actors["result"], "mapper", None),
            )
        if (
            self._display.contour_enabled
            and self._show_edges
        ):
            self._add_result_edges_layer(dataset)
        if (
            self._display.contour_enabled
            and (
                self._contour["show_minimum"]
                or self._contour["show_maximum"]
            )
        ):
            self._add_result_render_payload_extrema_labels(
                checked,
                payload_validated=True,
            )
        selected_mesh_references = set(
            self._mesh_scope_selected_references
        )
        self._install_mesh_scope_highlight_pipelines()
        if selected_mesh_references:
            selected_kind = next(iter(selected_mesh_references)).kind
            self.highlight_mesh_entities(
                selected_mesh_references,
                entity_kind=selected_kind,
            )
        self._refresh_undeformed_overlay()
        self._restore_selection()
        self._render()

    def _update_reused_scalar_layer(self) -> bool:
        payload = self._result_render_payload
        actor = self._actors.get("result")
        mapper = None if actor is None else getattr(actor, "mapper", None)
        if (
            payload is None
            or self._plotter is None
            or self._result_grid is not payload.dataset
            or mapper is None
            or not self._display.contour_enabled
        ):
            return False

        checked = self._consume_result_install_validation(payload)
        mapper.array_name = checked.scalar_name
        mapper.scalar_visibility = True
        mapper.scalar_range = self._contour_data_range(checked)
        mapper.Update()
        self._remove_scalar_bars()
        if self._contour["legend"]:
            scalar_bar = self._plotter.add_scalar_bar(
                mapper=mapper,
                **self._contour_bar_args(checked),
            )
            self._configure_contour_bar(
                checked,
                mapper,
                scalar_bar=scalar_bar,
            )
        self._remove_actor("extrema")
        if (
            self._contour["show_minimum"]
            or self._contour["show_maximum"]
        ):
            self._add_result_render_payload_extrema_labels(
                checked,
                payload_validated=True,
            )
        self._result_render_validated_mtime = int(
            checked.dataset.GetMTime()
        )
        self._render()
        return True

    def _contour_data_range(
        self,
        payload: ResultRenderPayload,
    ) -> tuple[float, float]:
        if self._contour["manual"]:
            return (
                float(self._contour["minimum"]),
                float(self._contour["maximum"]),
            )
        return self._payload_data_range(payload)

    def current_contour_range(self) -> tuple[float, float] | None:
        """返回当前结果字段的实际数值范围。"""

        payload = self._result_render_payload
        if payload is None:
            return None
        return self._payload_data_range(payload)

    @staticmethod
    def _payload_data_range(
        payload: ResultRenderPayload,
    ) -> tuple[float, float]:
        preference = (
            "point"
            if payload.topology.value_layout is ResultValueLayout.POINT
            else "cell"
        )
        minimum, maximum = payload.dataset.get_data_range(
            payload.scalar_name,
            preference=preference,
        )
        return float(minimum), float(maximum)

    def _contour_bar_args(
        self,
        payload: ResultRenderPayload,
    ) -> dict[str, Any]:
        selection = payload.topology.selection
        field_id = selection.field_key.request.field_id
        variable = field_id.variable.value
        vertical = self._contour["orientation"] == "vertical"
        font_family = {
            "Arial": "arial",
            "Times New Roman": "times",
            "Courier New": "courier",
        }.get(str(self._contour["legend_font"]), "arial")
        font_size = int(self._contour["legend_font_size"])
        title = f"{variable}, {selection.component}"
        if field_id.position in {
            FieldPosition.SECTION_POINT,
            FieldPosition.SECTION_END,
        }:
            title += f"（{result_field_position_label(field_id)}）"
        options: dict[str, Any] = {
            "title": title,
            "vertical": vertical,
            "n_labels": min(int(self._contour["levels"]) + 1, 7),
            "fmt": self._scalar_format(),
            "color": self._background_settings.foreground_color,
            "outline": False,
            "title_font_size": font_size,
            "label_font_size": font_size,
            "font_family": font_family,
            "unconstrained_font_size": True,
        }
        if vertical:
            options.update(
                width=0.045,
                height=0.62,
                position_x=0.78,
                position_y=0.19,
            )
        else:
            options.update(
                width=0.46,
                height=0.065,
                position_x=0.27,
                position_y=0.08,
            )
        return options

    def _configure_contour_bar(
        self,
        payload: ResultRenderPayload,
        mapper: Any | None,
        *,
        scalar_bar: Any | None = None,
    ) -> None:
        if self._plotter is None or mapper is None:
            return
        if scalar_bar is None:
            scalar_bars = getattr(self._plotter, "scalar_bars", None)
            title = self._contour_bar_args(payload)["title"]
            if scalar_bars is None or title not in scalar_bars:
                return
            scalar_bar = scalar_bars[title]

        minimum, maximum = self._contour_data_range(payload)
        label_count = min(int(self._contour["levels"]) + 1, 7)
        values = (
            np.asarray((minimum,), dtype=float)
            if minimum == maximum
            else np.linspace(minimum, maximum, label_count)
        )
        mapper.lookup_table.annotations = {
            float(value): self._format_scalar(float(value))
            for value in values
        }

        scalar_bar.DrawTickLabelsOff()
        scalar_bar.DrawAnnotationsOn()
        scalar_bar.AnnotationTextScalingOff()
        scalar_bar.SetTextPositionToPrecedeScalarBar()
        scalar_bar.SetTextPad(12)
        scalar_bar.SetAnnotationLeaderPadding(18.0)
        scalar_bar.SetVerticalTitleSeparation(18)
        annotation_text = scalar_bar.GetAnnotationTextProperty()
        label_text = scalar_bar.GetLabelTextProperty()
        annotation_text.SetColor(*label_text.GetColor())
        annotation_text.SetFontSize(label_text.GetFontSize())

    def _add_result_edges_layer(self, dataset: Any) -> Any | None:
        edge_mode = str(self._contour["edge_mode"])
        if edge_mode == CONTOUR_EDGE_NONE:
            edge_mode = "all"
        edges = extract_contour_edges(dataset, edge_mode)
        if edges is None or edges.n_cells == 0:
            return None
        edges = style_contour_edges(
            edges,
            str(self._contour["edge_style"]),
        )
        line_width = float(self._contour["edge_width"])
        if self._contour["edge_style"] == "bold":
            line_width = max(line_width * 2.0, 3.0)
        actor = self._plotter.add_mesh(
            edges,
            color=self._background_settings.foreground_color,
            line_width=line_width,
            lighting=False,
            show_scalar_bar=False,
            pickable=False,
            name="result_edges",
            reset_camera=False,
            **self._line_render_options(),
        )
        self._offset_highlight_actor(actor)
        self._actors["result_edges"] = actor
        return actor

    def _add_result_render_payload_extrema_labels(
        self,
        payload: ResultRenderPayload,
        *,
        payload_validated: bool = False,
    ) -> None:
        """Label extrema from payload scalar and typed location provenance."""

        checked = (
            payload
            if payload_validated
            else _require_result_render_payload(payload)
        )
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
        self._actors["extrema"] = self._plotter.add_point_labels(
            points[[entry[0] for entry in entries]],
            [entry[1] for entry in entries],
            point_color="#d69a3a",
            point_size=14,
            render_points_as_spheres=True,
            font_size=14,
            shape=None,
            text_color="#000000",
            always_visible=True,
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
        if location.section_point is not None:
            values.append(f"截面点 {int(location.section_point.number)}")
            values.append(
                "截面坐标 "
                f"({location.section_point.local_y:.6g}, "
                f"{location.section_point.local_z:.6g})"
            )
        return "，".join(values)

    def _refresh_geometry_dependent_layers(self, *, render: bool = True) -> None:
        self._remove_actor("element_edges")
        self._remove_actor("set_highlight")
        if self._show_edges:
            self._add_element_edges_layer()
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
            **self._line_render_options(),
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
            return f"%.{decimals}E"
        if mode == "engineering":
            return f"%.{decimals}E"
        return f"%.{decimals}g"

    def _format_scalar(self, value: float) -> str:
        decimals = int(self._contour["decimals"])
        mode = self._contour["number_format"]
        if mode == "fixed":
            return f"{value:.{decimals}f}"
        if mode == "scientific":
            if value == 0.0:
                return "0"
            mantissa, exponent = f"{value:.{decimals}E}".split("E")
            return f"{mantissa}E{int(exponent):+d}"
        if mode == "engineering":
            if value == 0.0:
                return "0"
            exponent = 3 * math.floor(math.log10(abs(value)) / 3)
            mantissa = value / (10.0 ** exponent)
            rounded = float(f"{mantissa:.{decimals}f}")
            if abs(rounded) >= 1000.0:
                exponent += 3
                mantissa /= 1000.0
            return f"{mantissa:.{decimals}f}E{exponent:+d}"
        return f"{value:.{decimals}g}"

    def _refresh_node_layer(self, *, render: bool = True) -> None:
        self._remove_actor("nodes")
        if self._show_nodes and self._plotter is not None and _pyvista is not None and self._geometry is not None:
            palette = self._visual_palette()
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
                color=self._node_layer_color(palette),
                point_size=self._node_point_size(),
                render_points_as_spheres=True, lighting=False,
                name="nodes", reset_camera=False,
            )
            self._update_pickable_actors()
        if render:
            self._render()

    def _label_render_options(self, kind: str) -> dict[str, Any]:
        if kind not in {"node", "element"}:
            raise ValueError("label kind must be 'node' or 'element'")
        return {
            "show_points": False,
            "font_size": 11 if self._is_line_mesh() else 10,
            "shape": None,
            "shadow": True,
            "always_visible": True,
            "justification_vertical": "bottom" if kind == "node" else "top",
            "text_color": self._background_settings.foreground_color,
            "reset_camera": False,
        }

    def _refresh_labels(self, *, render: bool = True) -> None:
        self._remove_actor("node_labels")
        self._remove_actor("element_labels")
        if self._plotter is None or self._geometry is None or _pyvista is None:
            return
        if self._show_node_labels:
            labels = [str(self._geometry.point_index_to_node_id[index]) for index in range(len(self._geometry.points))]
            self._actors["node_labels"] = self._plotter.add_point_labels(
                self._model_display_points(),
                labels,
                name="node_labels",
                **self._label_render_options("node"),
            )
        if self._show_element_labels:
            centers = self._pick_grid.cell_centers().points
            labels = [str(self._geometry.cell_index_to_element_id[index]) for index in range(len(self._geometry.cells))]
            self._actors["element_labels"] = self._plotter.add_point_labels(
                centers,
                labels,
                name="element_labels",
                **self._label_render_options("element"),
            )
        if render:
            self._render()

    def _current_points(self) -> np.ndarray:
        return self._geometry.points if self._grid is None else np.asarray(self._grid.points)

    def _install_picker(self) -> None:
        try:
            interactor = self._plotter.iren.interactor

            def camera_interaction_finished(_obj=None, _event=None):
                if self._sketch_authoring_active:
                    self._refresh_sketch_curve_sampling(render=False)
                    self._render()
                else:
                    self._refresh_symbols_for_camera(render=True)

            interactor.AddObserver("EndInteractionEvent", camera_interaction_finished, 1.0)
            self._picker_event_targets = {self._plotter}
            self._picker_event_targets.update(self._plotter.findChildren(QWidget))
            for target in self._picker_event_targets:
                target.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
                target.setMouseTracking(True)
                target.installEventFilter(self)
            self._update_pickable_actors()
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
        if hit.kind.startswith("mesh_"):
            kind = hit.kind.removeprefix("mesh_")
            if kind == "node":
                reference = MeshEntityRef.node(hit.pick_id)
            elif kind in {"element", "body"}:
                reference = MeshEntityRef.element(hit.pick_id)
            else:
                reference = self._mesh_scope_pick_to_ref.get(
                    (kind, hit.pick_id)
                )
                if reference is None and hit.dataset_name in {
                    "model_pick_grid",
                    _TYPED_RESULT_GRID_NAME,
                }:
                    reference = MeshEntityRef.element(hit.pick_id)
                if reference is None:
                    self.selectionMissed.emit(self._selection_mode)
                    return
            self.meshEntityPicked.emit(reference)
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
            try:
                return max(float(self._plotter.devicePixelRatioF()), 1.0)
            except (AttributeError, TypeError, ValueError):
                return 1.0

    def _qt_to_vtk_position(self, x: float, y: float) -> tuple[int, int]:
        """Convert one Qt logical top-left position to VTK device pixels."""
        ratio = self._device_pixel_ratio()
        height = int(round(float(self._plotter.height()) * ratio))
        return (
            int(round(float(x) * ratio)),
            height - int(round(float(y) * ratio)) - 1,
        )

    def _mesh_nodes_in_qt_rectangle(
        self,
        start: tuple[float, float],
        end: tuple[float, float],
    ) -> tuple[MeshEntityRef, ...]:
        payload = self._rendered_result_payload()
        if payload is not None:
            points = np.asarray(payload.dataset.points, dtype=float)
            node_ids = self._typed_result_point_ids(payload.dataset)
            occluder = payload.dataset
        elif self._geometry is not None:
            points = self._model_display_points()
            node_ids = np.asarray(
                self._geometry.point_index_to_node_id,
                dtype=np.int64,
            )
            occluder = self._pick_grid
        else:
            return ()
        display = self._world_points_to_display(points)
        if display is None:
            return ()
        bounds, _containment = self._vtk_rectangle(start, end)
        selected: dict[int, MeshEntityRef] = {}
        for point_index, candidate in enumerate(display):
            node_id = int(node_ids[point_index])
            if node_id <= 0:
                continue
            if not self._rectangle_contains_point(bounds, candidate):
                continue
            if not self._display_candidate_is_visible(
                candidate,
                occluder,
            ):
                continue
            selected.setdefault(node_id, MeshEntityRef.node(node_id))
        return tuple(selected.values())

    def _mesh_entities_in_qt_rectangle(
        self,
        start: tuple[float, float],
        end: tuple[float, float],
    ) -> tuple[MeshEntityRef, ...]:
        mode = self._selection_mode
        payload = self._rendered_result_payload()
        if payload is not None and (
            (
                mode == "mesh_edge"
                and self._mesh_scope_edge_rows
                and self._mesh_scope_edges is None
            )
            or (
                mode == "mesh_face"
                and self._mesh_scope_face_rows
                and self._mesh_scope_faces is None
            )
        ):
            return ()
        if mode == "mesh_edge" and self._mesh_scope_edges is not None:
            dataset = self._mesh_scope_edges
            scalar_name = "mesh_scope_pick_id"
            reference_kind = "edge"
        elif mode == "mesh_face" and self._mesh_scope_faces is not None:
            dataset = self._mesh_scope_faces
            scalar_name = "mesh_scope_pick_id"
            reference_kind = "face"
        else:
            dataset = self._pick_grid if payload is None else payload.dataset
            scalar_name = "element_id"
            reference_kind = "element"
        if dataset is None:
            return ()
        display = self._world_points_to_display(
            np.asarray(dataset.points, dtype=float)
        )
        if display is None:
            return ()
        bounds, containment = self._vtk_rectangle(start, end)
        if reference_kind == "element" and payload is not None:
            pick_ids = self._typed_result_cell_ids(dataset)
        elif scalar_name in dataset.cell_data:
            pick_ids = np.asarray(
                dataset.cell_data[scalar_name],
                dtype=np.int64,
            )
        else:
            return ()
        occluder = (
            payload.dataset
            if payload is not None
            else self._pick_grid
        )
        selected: dict[tuple[str, tuple[int, int]], MeshEntityRef] = {}
        for cell_index, pick_id in enumerate(pick_ids):
            if int(pick_id) <= 0:
                continue
            cell = dataset.get_cell(cell_index)
            points = display[np.asarray(cell.point_ids, dtype=np.int64)]
            if not len(points):
                continue
            matches = (
                all(
                    self._rectangle_contains_point(bounds, point)
                    for point in points
                )
                if containment
                else self._rectangle_intersects_points(bounds, points)
            )
            if not matches or not self._display_candidate_is_visible(
                np.mean(points, axis=0),
                occluder,
            ):
                continue
            if reference_kind == "element":
                reference = MeshEntityRef.element(int(pick_id))
            else:
                reference = self._mesh_scope_pick_to_ref.get(
                    (reference_kind, int(pick_id))
                )
                if reference is None:
                    continue
            selected.setdefault(
                (reference.kind, reference.identity),
                reference,
            )
        return tuple(selected.values())

    def _geometry_points_in_qt_rectangle(
        self,
        start: tuple[float, float],
        end: tuple[float, float],
    ) -> tuple[LogicalEntityRef, ...]:
        dataset = self._geometry_preview_points
        if dataset is None or "geometry_pick_id" not in dataset.point_data:
            return ()
        display = self._world_points_to_display(
            np.asarray(dataset.points, dtype=float)
        )
        if display is None:
            return ()
        bounds, _containment = self._vtk_rectangle(start, end)
        pick_ids = np.asarray(
            dataset.point_data["geometry_pick_id"],
            dtype=np.int64,
        )
        selected: set[LogicalEntityRef] = set()
        for point_index, pick_id in enumerate(pick_ids):
            candidate = display[point_index]
            if int(pick_id) <= 0 or not self._rectangle_contains_point(
                bounds,
                candidate,
            ):
                continue
            if not self._display_candidate_is_visible(
                candidate,
                self._geometry_preview_surface,
            ):
                continue
            reference = self._geometry_pick_to_ref.get(int(pick_id))
            if reference is not None:
                selected.add(reference)
        return tuple(sorted(selected, key=logical_ref_sort_key))

    def _geometry_entities_in_qt_rectangle(
        self,
        start: tuple[float, float],
        end: tuple[float, float],
    ) -> tuple[LogicalEntityRef, ...]:
        dataset = (
            self._geometry_preview_edges
            if self._selection_mode == "geometry_edge"
            else self._geometry_preview_surface
        )
        scalar_name = (
            "geometry_body_pick_id"
            if self._selection_mode == "geometry_body"
            else "geometry_pick_id"
        )
        if dataset is None or scalar_name not in dataset.cell_data:
            return ()
        display = self._world_points_to_display(
            np.asarray(dataset.points, dtype=float)
        )
        if display is None:
            return ()
        bounds, containment = self._vtk_rectangle(start, end)
        pick_ids = np.asarray(
            dataset.cell_data[scalar_name],
            dtype=np.int64,
        )
        cells_by_reference: dict[LogicalEntityRef, list[np.ndarray]] = {}
        for cell_index, pick_id in enumerate(pick_ids):
            if int(pick_id) <= 0:
                continue
            reference = self._geometry_pick_to_ref.get(int(pick_id))
            if reference is None:
                continue
            cell = dataset.get_cell(cell_index)
            point_ids = np.asarray(cell.point_ids, dtype=np.int64)
            if not len(point_ids):
                continue
            cells_by_reference.setdefault(reference, []).append(
                display[point_ids]
            )
        selected: list[LogicalEntityRef] = []
        for reference, cells in sorted(
            cells_by_reference.items(),
            key=lambda item: item[0].logical_id,
        ):
            entity_points = np.vstack(cells)
            if containment:
                matches = all(
                    self._rectangle_contains_point(bounds, point)
                    for point in entity_points
                )
            else:
                matches = self._rectangle_intersects_points(
                    bounds,
                    entity_points,
                )
            if not matches:
                continue
            if not any(
                self._display_candidate_is_visible(
                    np.mean(cell, axis=0),
                    self._pick_grid,
                )
                for cell in cells
            ):
                continue
            selected.append(reference)
        return tuple(selected)

    def _vtk_rectangle(
        self,
        start: tuple[float, float],
        end: tuple[float, float],
    ) -> tuple[tuple[float, float, float, float], bool]:
        first = self._qt_to_vtk_position(*start)
        second = self._qt_to_vtk_position(*end)
        return (
            (
                float(min(first[0], second[0])),
                float(max(first[0], second[0])),
                float(min(first[1], second[1])),
                float(max(first[1], second[1])),
            ),
            start[0] <= end[0],
        )

    @staticmethod
    def _rectangle_contains_point(
        bounds: tuple[float, float, float, float],
        point: np.ndarray,
    ) -> bool:
        minimum_x, maximum_x, minimum_y, maximum_y = bounds
        return (
            minimum_x <= float(point[0]) <= maximum_x
            and minimum_y <= float(point[1]) <= maximum_y
            and 0.0 <= float(point[2]) <= 1.0
        )

    @staticmethod
    def _rectangle_intersects_points(
        bounds: tuple[float, float, float, float],
        points: np.ndarray,
    ) -> bool:
        minimum_x, maximum_x, minimum_y, maximum_y = bounds
        finite = points[np.isfinite(points).all(axis=1)]
        if not len(finite):
            return False
        return (
            float(np.max(finite[:, 0])) >= minimum_x
            and float(np.min(finite[:, 0])) <= maximum_x
            and float(np.max(finite[:, 1])) >= minimum_y
            and float(np.min(finite[:, 1])) <= maximum_y
            and float(np.max(finite[:, 2])) >= 0.0
            and float(np.min(finite[:, 2])) <= 1.0
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
            if (
                mode == "geometry_body"
                and self._geometry_preview is not None
                and self._geometry_preview.topological_dimension == 1
            ):
                hit = self._pick_screen_edge(x, y, 6.0)
                if hit is None:
                    return None
                if hit.vtk_cell_id is None:
                    return None
                dataset = self._geometry_preview_edges
                if (
                    dataset is None
                    or "geometry_body_pick_id" not in dataset.cell_data
                ):
                    return None
                pick_id = int(
                    dataset.cell_data["geometry_body_pick_id"][
                        hit.vtk_cell_id
                    ]
                )
                return (
                    None
                    if pick_id <= 0
                    else replace(hit, kind="geometry_body", pick_id=pick_id)
                )
            hit = self._pick_cell(
                x,
                y,
                self._geometry_preview_surface,
                (
                    "geometry_body_pick_id"
                    if mode == "geometry_body"
                    else "geometry_pick_id"
                ),
                "geometry_surface",
                mode,
            )
            if hit is not None and hit.pick_id <= 0:
                return None
            return hit
        if mode == "mesh_node":
            payload = self._rendered_result_payload()
            if payload is not None:
                ids = self._typed_result_point_ids(payload.dataset)
                if not bool(np.any(ids > 0)):
                    return None
                return self._pick_screen_point(
                    x,
                    y,
                    payload.dataset,
                    "node_id",
                    _TYPED_RESULT_GRID_NAME,
                    payload.dataset,
                    8.0,
                    provenance_ids=ids,
                )
            return self._pick_screen_point(
                x,
                y,
                self._pick_grid,
                "node_id",
                "model_pick_grid",
                self._pick_grid,
                8.0,
            )
        if mode == "mesh_edge":
            hit = self._pick_mesh_scope_edge(x, y, 6.0)
            if hit is not None or self._mesh_scope_edges is not None:
                return hit
            if self._rendered_result_payload() is not None:
                return (
                    None
                    if self._mesh_scope_edge_rows
                    else self._pick_mesh_element_or_body(x, y, mode)
                )
            return self._pick_cell(
                x,
                y,
                self._pick_grid,
                "element_id",
                "model_pick_grid",
                mode,
            )
        if mode == "mesh_face":
            if self._mesh_scope_faces is not None:
                return self._pick_cell(
                    x,
                    y,
                    self._mesh_scope_faces,
                    "mesh_scope_pick_id",
                    "mesh_scope_faces",
                    mode,
                )
            if self._rendered_result_payload() is not None:
                return (
                    None
                    if self._mesh_scope_face_rows
                    else self._pick_mesh_element_or_body(x, y, mode)
                )
            return self._pick_cell(
                x,
                y,
                self._pick_grid,
                "element_id",
                "model_pick_grid",
                mode,
            )
        if mode == "mesh_element":
            return self._pick_mesh_element_or_body(x, y, mode)
        if mode == "mesh_body":
            return self._pick_mesh_element_or_body(x, y, mode)
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

    def _pick_mesh_element_or_body(
        self,
        x: int,
        y: int,
        mode: str,
    ) -> PickHit | None:
        payload = self._rendered_result_payload()
        if payload is not None:
            cell_ids = self._typed_result_cell_ids(payload.dataset)
            if not bool(np.any(cell_ids > 0)):
                return None
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
                    return replace(hit, kind=mode)
            return self._pick_cell(
                x,
                y,
                payload.dataset,
                "element_id",
                _TYPED_RESULT_GRID_NAME,
                mode,
                provenance_ids=cell_ids,
            )
        return self._pick_cell(
            x,
            y,
            self._pick_grid,
            "element_id",
            "model_pick_grid",
            mode,
        )

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

    def _pick_mesh_scope_edge(
        self,
        x: int,
        y: int,
        tolerance: float,
    ) -> PickHit | None:
        dataset = self._mesh_scope_edges
        if (
            dataset is None
            or not self._mesh_scope_edge_cells
            or "mesh_scope_pick_id" not in dataset.cell_data
        ):
            return None
        points = np.asarray(dataset.points, dtype=float)
        display_points = self._world_points_to_display(points)
        if display_points is None:
            return None
        mouse = np.asarray((float(x), float(y)), dtype=float)
        threshold = float(tolerance) * self._device_pixel_ratio()
        ids = np.asarray(
            dataset.cell_data["mesh_scope_pick_id"],
            dtype=np.int64,
        )
        candidates: list[tuple[float, float, int, np.ndarray]] = []
        for cell_index, cell in enumerate(self._mesh_scope_edge_cells):
            for start_index, end_index in zip(cell, cell[1:]):
                start = display_points[start_index]
                end = display_points[end_index]
                distance, fraction = _point_to_segment_distance(
                    mouse,
                    start[:2],
                    end[:2],
                )
                closest = start + fraction * (end - start)
                if (
                    distance <= threshold
                    and 0.0 <= closest[2] <= 1.0
                    and self._display_candidate_is_visible(
                        closest,
                        (
                            self._result_grid
                            if self._rendered_result_payload() is not None
                            else self._pick_grid
                        ),
                    )
                ):
                    world = (
                        points[start_index]
                        + fraction * (points[end_index] - points[start_index])
                    )
                    candidates.append(
                        (distance, float(closest[2]), cell_index, world)
                    )
        if not candidates:
            return None
        _distance, _depth, cell_index, world = min(candidates)
        return PickHit(
            self._selection_mode,
            int(ids[cell_index]),
            "mesh_scope_edges",
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
        logical_pick_ids = (hit.pick_id,)
        if hit.kind.startswith("geometry_"):
            reference = self._geometry_pick_to_ref.get(hit.pick_id)
            if reference is not None:
                logical_pick_ids = self._geometry_ref_to_pick_ids.get(
                    reference,
                    logical_pick_ids,
                )
        if hit.kind == "geometry_point" and self._geometry_preview_points is not None:
            ids = np.asarray(
                self._geometry_preview_points.point_data["geometry_pick_id"]
            )
            indices = np.flatnonzero(np.isin(ids, logical_pick_ids))
            if len(indices):
                data = _pyvista.PolyData(
                    np.asarray(self._geometry_preview_points.points)[indices]
                )
                kwargs = {"point_size": 11, "render_points_as_spheres": True}
        elif hit.kind == "geometry_edge" and self._geometry_preview_edges is not None:
            ids = np.asarray(
                self._geometry_preview_edges.cell_data["geometry_pick_id"]
            )
            cells = np.flatnonzero(np.isin(ids, logical_pick_ids))
            if len(cells):
                data = self._geometry_preview_edges.extract_cells(cells)
                kwargs = {"line_width": 4}
        elif hit.kind in {"geometry_face", "geometry_body"}:
            if self._geometry_preview_surface is not None:
                if hit.kind == "geometry_body":
                    if (
                        "geometry_body_pick_id"
                        not in self._geometry_preview_surface.cell_data
                    ):
                        data = self._geometry_preview_surface
                    else:
                        ids = np.asarray(
                            self._geometry_preview_surface.cell_data[
                                "geometry_body_pick_id"
                            ]
                        )
                        cells = np.flatnonzero(
                            np.isin(ids, logical_pick_ids)
                        )
                        if len(cells):
                            data = (
                                self._geometry_preview_surface.extract_cells(
                                    cells
                                )
                            )
                else:
                    ids = np.asarray(
                        self._geometry_preview_surface.cell_data["geometry_pick_id"]
                    )
                    cells = np.flatnonzero(np.isin(ids, logical_pick_ids))
                    if len(cells):
                        data = self._geometry_preview_surface.extract_cells(cells)
                kwargs = {"opacity": 0.38}
            elif (
                hit.kind == "geometry_body"
                and self._geometry_preview is not None
                and self._geometry_preview.topological_dimension == 1
                and self._geometry_preview_edges is not None
            ):
                ids = np.asarray(
                    self._geometry_preview_edges.cell_data[
                        "geometry_body_pick_id"
                    ]
                )
                cells = np.flatnonzero(np.isin(ids, logical_pick_ids))
                if len(cells):
                    data = self._geometry_preview_edges.extract_cells(cells)
                kwargs = {"line_width": 5}
        elif (
            hit.kind in {"mesh_edge", "mesh_face"}
            and hit.dataset_name in {
                "mesh_scope_edges",
                "mesh_scope_faces",
            }
        ):
            dataset = (
                self._mesh_scope_edges
                if hit.kind == "mesh_edge"
                else self._mesh_scope_faces
            )
            if dataset is not None:
                ids = np.asarray(
                    dataset.cell_data["mesh_scope_pick_id"],
                    dtype=np.int64,
                )
                cells = np.flatnonzero(ids == hit.pick_id)
                if len(cells):
                    data = dataset.extract_cells(cells)
                    kwargs = (
                        {"line_width": 4}
                        if hit.kind == "mesh_edge"
                        else {"opacity": 0.38}
                    )
        elif hit.kind in {"node", "mesh_node"}:
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
        elif hit.kind in {
            "element",
            "mesh_element",
            "mesh_body",
            "mesh_edge",
            "mesh_face",
        }:
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
            if (
                self._selection_mode == "geometry_body"
                and self._geometry_preview is not None
                and self._geometry_preview.topological_dimension == 1
            ):
                target_names.add("geometry_edges")
            else:
                target_names.add("geometry_surface")
        elif self._selection_mode in {"node", "mesh_node"} and "nodes" in self._actors:
            target_names.add("nodes")
        elif self._selection_mode in {
            "element",
            "mesh_element",
            "mesh_body",
        }:
            target_names.add("mesh_surface")
        elif self._selection_mode == "mesh_edge":
            target_names.add("mesh_scope_edges")
        elif self._selection_mode == "mesh_face":
            target_names.add("mesh_scope_faces")
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
            mapper.SetRelativeCoincidentTopologyPolygonOffsetParameters(
                -4.0,
                -4.0,
            )
            mapper.SetRelativeCoincidentTopologyLineOffsetParameters(
                -4.0,
                -4.0,
            )
            mapper.SetRelativeCoincidentTopologyPointOffsetParameter(-4.0)
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
        self._refresh_coordinate_system_axes()

    def _refresh_coordinate_system_axes(self) -> None:
        if self._plotter is None:
            return
        try:
            self._plotter.hide_axes()
            if self._contour.get("show_coordinate_system", True):
                self._plotter.add_axes(
                    color=self._background_settings.foreground_color
                )
        except Exception:
            pass

    def _visual_palette(self) -> dict[str, str]:
        dark = self._background_settings.is_dark and self._background_settings.auto_contrast
        if dark:
            return {
                "mesh": "#718797", "element": "#6FA6BC", "node": "#D0A05B",
                "result": "#8295a5", "overlay": "#e0e6ea",
                "label_background": "#263746",
            }
        return {
            "mesh": "#d8dde2", "element": "#3F6F8C", "node": "#9A6F3F",
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
                scalar_bar.GetAnnotationTextProperty().SetColor(*rgb)
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
            self.nativeSurfaceUpdated.emit()
