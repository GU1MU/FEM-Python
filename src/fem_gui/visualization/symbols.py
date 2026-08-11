"""约束与载荷符号的状态和采样辅助。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class SymbolSettings:
    """约束与载荷符号的完整显示状态。"""

    step_name: str | None = None
    show_constraints: bool = True
    show_nodal_loads: bool = True
    show_edge_loads: bool = True
    show_surface_loads: bool = True
    show_line_loads: bool = True
    show_values: bool = False
    scale: float = 1.0
    normalize_arrows: bool = True
    sampling_density: str = "low"
    constraint_color: str = "#5F7F96"
    load_color: str = "#A65D54"


def sample_polyline(points: np.ndarray, density: str) -> np.ndarray:
    """沿折线按弧长生成载荷符号采样点。"""
    points = np.asarray(points, dtype=float)
    count = {"low": 1, "medium": 3, "high": 7}.get(density, 3)
    if len(points) < 2:
        return points[:1]
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    total = float(np.sum(lengths))
    if total <= 0.0:
        return points[:1]
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    targets = np.linspace(0.0, total, count + 2)[1:-1]
    samples = []
    for target in targets:
        segment = min(int(np.searchsorted(cumulative, target, side="right") - 1), len(lengths) - 1)
        fraction = (target - cumulative[segment]) / lengths[segment]
        samples.append(points[segment] + fraction * (points[segment + 1] - points[segment]))
    return np.asarray(samples)


def sample_face(points: np.ndarray, density: str) -> np.ndarray:
    """在面中心附近生成稳定的可视化采样点。"""
    points = np.asarray(points, dtype=float)
    if len(points) == 0:
        return np.empty((0, 3), dtype=float)
    center = np.mean(points, axis=0)
    if density == "low":
        return center.reshape((1, 3))
    fraction = 0.35 if density == "medium" else 0.6
    limit = min(len(points), 4 if density == "medium" else len(points))
    offsets = [center + fraction * (point - center) for point in points[:limit]]
    return np.asarray([center, *offsets])


def symbol_length(
    points: np.ndarray,
    multiplier: float = 1.0,
    *,
    world_per_pixel: float | None = None,
    minimum_pixels: float = 24.0,
    maximum_pixels: float = 56.0,
) -> float:
    """Return a feature-based glyph length constrained to a useful screen size."""
    points = np.asarray(points, dtype=float)
    if len(points) == 0:
        base = 0.015
    else:
        sides = np.ptp(points, axis=0)
        active = sides[sides > max(float(np.max(sides)) * 1.0e-9, 1.0e-12)]
        effective = float(np.median(active)) if len(active) else 1.0
        base = 0.04 * effective
    if world_per_pixel is not None and float(world_per_pixel) > 0.0:
        pixel_size = float(world_per_pixel)
        base = float(np.clip(
            base,
            float(minimum_pixels) * pixel_size,
            float(maximum_pixels) * pixel_size,
        ))
    return max(base * float(multiplier), 1.0e-9)


def constraint_symbol_dimensions(glyph_length: float) -> tuple[float, float]:
    """Return a slender constraint-marker size from the shared screen scale."""
    length = 1.35 * float(glyph_length)
    return length, 0.12 * length


def constraint_outward_direction(
    point: np.ndarray,
    model_center: np.ndarray,
    component: int,
) -> np.ndarray:
    """Place a translational constraint marker on the nearest exterior axis side."""
    point = np.asarray(point, dtype=float)
    model_center = np.asarray(model_center, dtype=float)
    direction = np.zeros(3, dtype=float)
    direction[int(component)] = (
        1.0 if point[int(component)] > model_center[int(component)] else -1.0
    )
    return direction


def load_symbol_length(glyph_length: float) -> float:
    """Return the displayed shaft-and-head length for force vectors."""
    return 2.0 * float(glyph_length)


def load_arrow_origins(
    anchors: np.ndarray,
    directions: np.ndarray,
    lengths: np.ndarray,
    start_aligned: np.ndarray,
) -> np.ndarray:
    """Resolve arrow origins for mixed start- and tip-aligned load glyphs."""
    anchors = np.asarray(anchors, dtype=float)
    directions = np.asarray(directions, dtype=float)
    lengths = np.asarray(lengths, dtype=float)
    start_aligned = np.asarray(start_aligned, dtype=bool)
    origins = anchors - directions * lengths[:, None]
    origins[start_aligned] = anchors[start_aligned]
    return origins


def region_sample_indices(points: np.ndarray, density: str) -> np.ndarray:
    """Spatially sample one semantic region without following mesh refinement."""
    return _spatial_sample_indices(points, density, {"low": 4, "medium": 12, "high": 24})


def constraint_sample_indices(points: np.ndarray, density: str) -> np.ndarray:
    """Sample a support along its visible boundary instead of across its area."""
    points = np.asarray(points, dtype=float)
    if len(points) == 0:
        return np.empty(0, dtype=np.int64)
    limits = {"low": 3, "medium": 6, "high": 12}
    limit = limits.get(density, limits["medium"])
    centered = points - np.mean(points, axis=0)
    _left, singular_values, right = np.linalg.svd(centered, full_matrices=False)
    tolerance = max(float(singular_values[0]) * 1.0e-8, 1.0e-12)
    dimension = int(np.count_nonzero(singular_values > tolerance))
    if dimension == 0:
        return np.asarray([0], dtype=np.int64)
    if dimension == 1:
        if len(points) <= limit:
            return np.arange(len(points), dtype=np.int64)
        order = np.argsort(centered @ right[0])
        offsets = np.linspace(0, len(order) - 1, limit, dtype=np.int64)
        return order[offsets].astype(np.int64)
    if dimension == 2:
        projected = centered @ right[:2].T
        hull = _convex_hull_indices(projected)
        if len(hull) >= 3:
            hull_indices = np.asarray(hull, dtype=np.int64)
            boundary_limit = max(4, limit)
            if len(hull_indices) <= boundary_limit:
                return hull_indices
            offsets = np.linspace(
                0,
                len(hull_indices),
                boundary_limit,
                endpoint=False,
                dtype=np.int64,
            )
            return hull_indices[offsets]
    if len(points) <= limit:
        return np.arange(len(points), dtype=np.int64)
    return _spatial_sample_indices(points, density, limits)


def constraint_spatial_regions(
    points: np.ndarray,
    model_points: np.ndarray,
    *,
    reference_axis: np.ndarray | None = None,
) -> tuple[np.ndarray, ...]:
    """Separate support regions divided by a large gap along the model axis."""
    points = np.asarray(points, dtype=float)
    if len(points) <= 6:
        return (np.arange(len(points), dtype=np.int64),)
    if reference_axis is None:
        reference = np.asarray(model_points, dtype=float)
        centered_reference = reference - np.mean(reference, axis=0)
        _left, _singular_values, right = np.linalg.svd(
            centered_reference, full_matrices=False
        )
        axis = right[0]
    else:
        axis = np.asarray(reference_axis, dtype=float)
    projection = points @ axis
    order = np.argsort(projection)
    ordered_projection = projection[order]
    gaps = np.diff(ordered_projection)
    if len(gaps) == 0:
        return (np.arange(len(points), dtype=np.int64),)
    split = int(np.argmax(gaps))
    span = float(ordered_projection[-1] - ordered_projection[0])
    if span <= 1.0e-12 or float(gaps[split]) < 0.35 * span:
        return (np.arange(len(points), dtype=np.int64),)
    return (
        np.sort(order[:split + 1]).astype(np.int64),
        np.sort(order[split + 1:]).astype(np.int64),
    )


def camera_facing_offset(
    point: np.ndarray,
    camera_position: np.ndarray | None,
    distance: float,
) -> np.ndarray:
    """Return a view-ray offset that preserves an object's screen position."""
    if camera_position is None or distance <= 0.0:
        return np.zeros(3)
    direction = np.asarray(camera_position, dtype=float) - np.asarray(point, dtype=float)
    norm = float(np.linalg.norm(direction))
    if norm <= 1.0e-12:
        return np.zeros(3)
    return float(distance) * direction / norm


def _convex_hull_indices(points: np.ndarray) -> list[int]:
    """Return counter-clockwise hull vertices for projected support points."""
    def cross(origin: int, first: int, second: int) -> float:
        first_edge = points[first] - points[origin]
        second_edge = points[second] - points[origin]
        return float(
            first_edge[0] * second_edge[1]
            - first_edge[1] * second_edge[0]
        )

    ordered = sorted(range(len(points)), key=lambda index: (points[index, 0], points[index, 1]))
    lower: list[int] = []
    for index in ordered:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], index) <= 1.0e-12:
            lower.pop()
        lower.append(index)
    upper: list[int] = []
    for index in reversed(ordered):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], index) <= 1.0e-12:
            upper.pop()
        upper.append(index)
    return lower[:-1] + upper[:-1]


def _spatial_sample_indices(
    points: np.ndarray,
    density: str,
    limits: dict[str, int],
) -> np.ndarray:
    """Select evenly spread representative points without mesh-order bias."""
    points = np.asarray(points, dtype=float)
    count = len(points)
    limit = limits.get(density, limits["medium"])
    if count <= limit:
        return np.arange(count, dtype=np.int64)
    # Start at the geometric center and add the point farthest from the already
    # selected set.  The old "first point in a bin" approach made the result
    # visibly depend on element numbering: a refined region could look random
    # even though its load definition was uniform.
    centroid = np.mean(points, axis=0)
    first = int(np.argmin(np.einsum("ij,ij->i", points - centroid, points - centroid)))
    selected = [first]
    nearest_distance2 = np.einsum(
        "ij,ij->i", points - points[first], points - points[first]
    )
    nearest_distance2[first] = -1.0
    while len(selected) < limit:
        index = int(np.argmax(nearest_distance2))
        if nearest_distance2[index] < 0.0:
            break
        selected.append(index)
        distance2 = np.einsum(
            "ij,ij->i", points - points[index], points - points[index]
        )
        nearest_distance2 = np.minimum(nearest_distance2, distance2)
        nearest_distance2[selected] = -1.0
    return np.asarray(selected, dtype=np.int64)


def arc_points(
    center: np.ndarray,
    axis: np.ndarray,
    radius: float,
    *,
    segments: int = 18,
) -> np.ndarray:
    """Create a three-quarter circular arc around an axis for moment symbols."""
    center = np.asarray(center, dtype=float)
    axis = np.asarray(axis, dtype=float)
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm <= 0.0:
        return np.empty((0, 3), dtype=float)
    axis = axis / axis_norm
    reference = np.array([1.0, 0.0, 0.0])
    if abs(float(np.dot(reference, axis))) > 0.9:
        reference = np.array([0.0, 1.0, 0.0])
    first = np.cross(axis, reference)
    first /= np.linalg.norm(first)
    second = np.cross(axis, first)
    angles = np.linspace(-0.25 * np.pi, 1.25 * np.pi, segments + 1)
    return center + radius * (
        np.cos(angles)[:, None] * first + np.sin(angles)[:, None] * second
    )


def rotation_lock_points(
    center: np.ndarray,
    axis: np.ndarray,
    radius: float,
    *,
    segments: int = 24,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a closed ring and crossed bars for a restrained rotation."""
    center = np.asarray(center, dtype=float)
    axis = np.asarray(axis, dtype=float)
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm <= 0.0:
        empty = np.empty((0, 3), dtype=float)
        return empty, empty
    axis = axis / axis_norm
    reference = np.array([1.0, 0.0, 0.0])
    if abs(float(np.dot(reference, axis))) > 0.9:
        reference = np.array([0.0, 1.0, 0.0])
    first = np.cross(axis, reference)
    first /= np.linalg.norm(first)
    second = np.cross(axis, first)
    angles = np.linspace(0.0, 2.0 * np.pi, segments + 1)
    ring = center + radius * (
        np.cos(angles)[:, None] * first + np.sin(angles)[:, None] * second
    )
    bar_radius = 0.68 * float(radius)
    bars = np.asarray((
        center - bar_radius * first,
        center + bar_radius * first,
        center - bar_radius * second,
        center + bar_radius * second,
    ))
    return ring, bars


def constraint_rotation_axes(
    components: tuple[int, ...],
    *,
    is_3d: bool,
    point: np.ndarray,
    camera_position: np.ndarray | None,
) -> np.ndarray:
    """Collapse a full 3D rotational lock to one camera-facing symbol."""
    translation_count = 3 if is_3d else 2
    rotations = tuple(
        component for component in components if component >= translation_count
    )
    if not rotations:
        return np.empty((0, 3), dtype=float)
    if is_3d and set(rotations) == {3, 4, 5}:
        if camera_position is not None:
            view_axis = np.asarray(camera_position, dtype=float) - np.asarray(point, dtype=float)
            norm = float(np.linalg.norm(view_axis))
            if norm > 1.0e-12:
                return (view_axis / norm).reshape((1, 3))
        return np.asarray(((0.0, 0.0, 1.0),))
    axes = []
    for component in rotations:
        axis = np.zeros(3)
        axis[(component - translation_count + (2 if not is_3d else 0)) % 3] = 1.0
        axes.append(axis)
    return np.asarray(axes)
