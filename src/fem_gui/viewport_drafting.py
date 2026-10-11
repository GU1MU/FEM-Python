"""Detached wire/sketch display data and work-plane drafting geometry."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np

from fem.geometry import (
    SketchArc,
    SketchCircle,
    SketchCurve,
    SketchLine,
    SketchPlane,
    SketchPoint,
    sketch_intersections,
)

from .sketch_constraint_ui import SketchConstraintOverlay

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


@dataclass(frozen=True, slots=True)
class WireGridLayout:
    """A work-plane grid whose visible lines stay on snap coordinates."""

    axes: tuple[int, int, int]
    center: tuple[float, float, float]
    plane_size: float
    resolution: int
    visible_spacing: float


def wire_grid_layout(
    points: np.ndarray,
    plane: str,
    offset: float,
    snap_spacing: float,
    minimum_size: float = 10.0,
) -> WireGridLayout:
    """Size and align a bounded work-plane grid without losing snap alignment."""

    axes = {"XY": (0, 1, 2), "XZ": (0, 2, 1), "YZ": (1, 2, 0)}[plane]
    spacing = float(snap_spacing)
    clean_minimum_size = float(minimum_size)
    if not np.isfinite(clean_minimum_size) or clean_minimum_size <= 0.0:
        raise ValueError("grid minimum size must be positive")
    if len(points):
        minimum = np.min(points[:, axes[:2]], axis=0)
        maximum = np.max(points[:, axes[:2]], axis=0)
        spans = tuple(float(value) for value in maximum - minimum)
        local_center = 0.5 * (minimum + maximum)
    else:
        spans = (0.0, 0.0)
        local_center = np.zeros(2, dtype=float)
    geometry_span = max(spans, default=0.0)
    padding = max(2.0 * spacing, 0.1 * geometry_span)
    target_size = max(
        clean_minimum_size,
        geometry_span + 2.0 * padding,
    )
    requested_intervals = max(1, int(math.ceil(target_size / spacing)))
    major_factor = max(1, int(math.ceil(requested_intervals / 2000)))
    visible_spacing = spacing * major_factor
    resolution = max(2, int(math.ceil(target_size / visible_spacing)))
    if resolution % 2:
        resolution += 1
    plane_size = resolution * visible_spacing
    center = np.zeros(3, dtype=float)
    for index, axis in enumerate(axes[:2]):
        center[axis] = (
            math.floor(local_center[index] / visible_spacing + 0.5)
            * visible_spacing
        )
    center[axes[2]] = float(offset)
    return WireGridLayout(
        axes,
        tuple(float(value) for value in center),
        float(plane_size),
        resolution,
        float(visible_spacing),
    )


def wire_grid_polydata(pyvista, layout: WireGridLayout):
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


def wire_camera_bounds(
    points: np.ndarray,
    plane: str,
    offset: float,
    spacing: float,
    minimum_size: float = 10.0,
) -> tuple[float, float, float, float, float, float]:
    """Frame the square wire work plane instead of only its draft points."""

    layout = wire_grid_layout(
        points,
        plane,
        offset,
        spacing,
        minimum_size,
    )
    center = np.asarray(layout.center, dtype=float)
    half_size = 0.5 * layout.plane_size
    minimum = center.copy()
    maximum = center.copy()
    for axis in layout.axes[:2]:
        minimum[axis] -= half_size
        maximum[axis] += half_size
    depth = max(half_size * 1.0e-6, 1.0e-6)
    minimum[layout.axes[2]] -= depth
    maximum[layout.axes[2]] += depth
    return tuple(
        float(value)
        for axis in range(3)
        for value in (minimum[axis], maximum[axis])
    )


def wire_coordinate_label(point: Iterable[float]) -> str:
    """Format one hover coordinate using glyphs supported by VTK's font."""

    x, y, z = tuple(float(value) for value in point)
    return f"({x:.2f}, {y:.2f}, {z:.2f})"


def intersect_ray_with_sketch_plane(
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


def snap_sketch_plane_point(
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


def sketch_local_points(
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


def sketch_grid_polydata(
    pyvista,
    plane: SketchPlane,
    points: Iterable[tuple[float, float, float]],
    spacing: float,
    minimum_size: float = 10.0,
):
    """Build a bounded grid aligned to the frame's U/V axes."""

    local = sketch_local_points(points, plane)
    local_3d = np.column_stack(
        (local, np.zeros(len(local), dtype=float))
    )
    layout = wire_grid_layout(
        local_3d,
        "XY",
        0.0,
        spacing,
        minimum_size,
    )
    grid = wire_grid_polydata(pyvista, layout)
    grid.points = np.asarray(
        tuple(
            plane.to_global(float(point[0]), float(point[1]))
            for point in grid.points
        ),
        dtype=float,
    )
    return grid, layout


def sketch_axis_local_endpoints(
    layout: WireGridLayout,
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


def sketch_coordinate_label(
    point: Iterable[float],
    plane: SketchPlane,
) -> str:
    u, v = plane.to_local(tuple(float(value) for value in point))
    return f"U = {u:.2f}\nV = {v:.2f}"


def sketch_snap_label(kind: str | None) -> str | None:
    return {
        "sketch_point": "Sketch point",
        "topology_vertex": "External reference point",
        "face_center": "External reference point",
        "line_midpoint": "Midpoint",
        "circle_center": "Circle center",
        "arc_center": "Circle center",
        "intersection": "Intersection",
        "grid": "Grid point",
    }.get(kind)


def sketch_camera_bounds(
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
    values = sketch_local_points(points, frame)
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


def sketch_shape_preview_points(
    mode: str,
    pending_points: Iterable[tuple[float, float, float]],
    cursor_point: tuple[float, float, float] | None,
    plane: SketchPlane | None = None,
) -> tuple[tuple[float, float, float], ...]:
    """Build the transient shape polyline while a multi-click tool is active."""

    pending = tuple(
        tuple(float(value) for value in point)
        for point in pending_points
    )
    normalized_mode = str(mode).strip().casefold()
    if cursor_point is None:
        return ()
    frame = SketchPlane.xy() if plane is None else plane
    if type(frame) is not SketchPlane:
        raise TypeError("plane must be a SketchPlane")
    if normalized_mode == "arc":
        if len(pending) == 1:
            return pending[0], tuple(float(value) for value in cursor_point)
        if len(pending) == 2:
            return sketch_three_point_arc_preview(
                pending[0],
                pending[1],
                tuple(float(value) for value in cursor_point),
                frame,
            )
        return ()
    if normalized_mode not in {"rectangle", "circle"} or len(pending) != 1:
        return ()
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


def sketch_three_point_arc_preview(
    first: tuple[float, float, float],
    through: tuple[float, float, float],
    last: tuple[float, float, float],
    plane: SketchPlane,
) -> tuple[tuple[float, float, float], ...]:
    local_first = plane.to_local(first)
    local_through = plane.to_local(through)
    local_last = plane.to_local(last)
    denominator = 2.0 * (
        local_first[0] * (local_through[1] - local_last[1])
        + local_through[0] * (local_last[1] - local_first[1])
        + local_last[0] * (local_first[1] - local_through[1])
    )
    if abs(denominator) <= 1.0e-12:
        return first, through, last
    first_square = local_first[0] ** 2 + local_first[1] ** 2
    through_square = local_through[0] ** 2 + local_through[1] ** 2
    last_square = local_last[0] ** 2 + local_last[1] ** 2
    center_u = (
        first_square * (local_through[1] - local_last[1])
        + through_square * (local_last[1] - local_first[1])
        + last_square * (local_first[1] - local_through[1])
    ) / denominator
    center_v = (
        first_square * (local_last[0] - local_through[0])
        + through_square * (local_first[0] - local_last[0])
        + last_square * (local_through[0] - local_first[0])
    ) / denominator
    start_angle = math.atan2(
        local_first[1] - center_v,
        local_first[0] - center_u,
    )
    end_angle = math.atan2(
        local_last[1] - center_v,
        local_last[0] - center_u,
    )
    cross = (
        (local_through[0] - center_u) * (local_last[1] - center_v)
        - (local_through[1] - center_v) * (local_last[0] - center_u)
    )
    sweep = (
        (end_angle - start_angle) % math.tau
        if cross >= 0.0
        else -((start_angle - end_angle) % math.tau)
    )
    radius = math.hypot(
        local_first[0] - center_u,
        local_first[1] - center_v,
    )
    count = max(8, int(math.ceil(48 * abs(sweep) / math.tau)))
    return tuple(
        plane.to_global(
            center_u + radius * math.cos(start_angle + sweep * index / count),
            center_v + radius * math.sin(start_angle + sweep * index / count),
        )
        for index in range(count + 1)
    )


def sketch_curve_sample_count(
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


def sketch_intersection_points(
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

    local = sketch_local_points(data.points, data.plane)
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


def sketch_intersection_curve_ids_at(
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
