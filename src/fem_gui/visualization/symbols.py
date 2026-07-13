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
    show_values: bool = False
    scale: float = 1.0
    normalize_arrows: bool = True
    sampling_density: str = "medium"
    constraint_color: str = "#2F6F9F"
    load_color: str = "#B24A3A"


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
    minimum_pixels: float = 18.0,
    maximum_pixels: float = 32.0,
) -> float:
    """Return a feature-based glyph length constrained to a useful screen size."""
    points = np.asarray(points, dtype=float)
    if len(points) == 0:
        base = 0.015
    else:
        sides = np.ptp(points, axis=0)
        active = sides[sides > max(float(np.max(sides)) * 1.0e-9, 1.0e-12)]
        effective = float(np.median(active)) if len(active) else 1.0
        base = 0.015 * effective
    if world_per_pixel is not None and float(world_per_pixel) > 0.0:
        pixel_size = float(world_per_pixel)
        base = float(np.clip(
            base,
            float(minimum_pixels) * pixel_size,
            float(maximum_pixels) * pixel_size,
        ))
    return max(base * float(multiplier), 1.0e-9)


def constraint_symbol_dimensions(glyph_length: float) -> tuple[float, float]:
    """Return a compact constraint glyph size from the shared screen scale."""
    length = 1.65 * float(glyph_length)
    return length, 0.20 * length


def load_symbol_length(glyph_length: float) -> float:
    """Return the displayed shaft-and-head length for force vectors."""
    return 3.3 * float(glyph_length)


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
            return np.asarray(hull, dtype=np.int64)
    if len(points) <= limit:
        return np.arange(len(points), dtype=np.int64)
    return _spatial_sample_indices(points, density, limits)


def constraint_spatial_regions(
    points: np.ndarray,
    model_points: np.ndarray,
) -> tuple[np.ndarray, ...]:
    """Separate support regions divided by a large gap along the model axis."""
    points = np.asarray(points, dtype=float)
    if len(points) <= 6:
        return (np.arange(len(points), dtype=np.int64),)
    reference = np.asarray(model_points, dtype=float)
    centered_reference = reference - np.mean(reference, axis=0)
    _left, _singular_values, right = np.linalg.svd(
        centered_reference, full_matrices=False
    )
    projection = points @ right[0]
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
        return float(np.cross(
            points[first] - points[origin], points[second] - points[origin]
        ))

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
