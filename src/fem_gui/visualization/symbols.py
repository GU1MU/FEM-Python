"""约束与载荷符号的状态和采样辅助。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


_DISTRIBUTED_SYMBOL_LIMITS = {"low": 6, "medium": 12, "high": 24}


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
    count = _DISTRIBUTED_SYMBOL_LIMITS.get(
        density,
        _DISTRIBUTED_SYMBOL_LIMITS["medium"],
    )
    if len(points) < 2:
        return points[:1]
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    total = float(np.sum(lengths))
    if total <= 0.0:
        return points[:1]
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    targets = total * (np.arange(count, dtype=float) + 0.5) / count
    samples = []
    for target in targets:
        segment = min(int(np.searchsorted(cumulative, target, side="right") - 1), len(lengths) - 1)
        fraction = (target - cumulative[segment]) / lengths[segment]
        samples.append(points[segment] + fraction * (points[segment + 1] - points[segment]))
    return np.asarray(samples)


def sample_distributed_polyline(
    member_points: list[np.ndarray],
    member_vectors: list[np.ndarray],
    density: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample connected line members at equal aggregate arc-length intervals."""
    if len(member_points) != len(member_vectors):
        raise ValueError("member points and vectors must have the same length")
    records: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    endpoints: list[tuple[tuple[float, ...], tuple[float, ...]]] = []
    for raw_points, raw_vector in zip(member_points, member_vectors):
        points = np.asarray(raw_points, dtype=float)
        if len(points) < 2:
            continue
        vector = np.asarray(raw_vector, dtype=float)
        records.append((points, vector, np.linalg.norm(np.diff(points, axis=0), axis=1)))
        endpoints.append((_point_key(points[0]), _point_key(points[-1])))
    if not records:
        empty = np.empty((0, 3), dtype=float)
        return empty, empty

    adjacency: dict[tuple[float, ...], list[int]] = {}
    for index, (first, last) in enumerate(endpoints):
        adjacency.setdefault(first, []).append(index)
        adjacency.setdefault(last, []).append(index)
    remaining = set(range(len(records)))
    ordered_segments: list[tuple[np.ndarray, np.ndarray, np.ndarray, float]] = []
    while remaining:
        seed = min(
            remaining,
            key=lambda index: (
                min(endpoints[index]),
                max(endpoints[index]),
            ),
        )
        component = _line_member_component(seed, remaining, endpoints, adjacency)
        degrees: dict[tuple[float, ...], int] = {}
        for index in component:
            first, last = endpoints[index]
            degrees[first] = degrees.get(first, 0) + 1
            degrees[last] = degrees.get(last, 0) + 1
        open_ends = [point for point, degree in degrees.items() if degree == 1]
        current = min(open_ends or list(degrees))
        component_remaining = set(component)
        while component_remaining:
            options = [
                index
                for index in adjacency.get(current, ())
                if index in component_remaining
            ]
            if not options:
                current = min(
                    point
                    for index in component_remaining
                    for point in endpoints[index]
                )
                continue
            index = min(
                options,
                key=lambda candidate: (
                    _other_endpoint(endpoints[candidate], current),
                    candidate,
                ),
            )
            points, vector, lengths = records[index]
            first, last = endpoints[index]
            if current != first:
                points = points[::-1]
                lengths = lengths[::-1]
                next_point = first
            else:
                next_point = last
            for segment, length in enumerate(lengths):
                if float(length) > 0.0:
                    ordered_segments.append(
                        (points[segment], points[segment + 1], vector, float(length))
                    )
            component_remaining.remove(index)
            remaining.remove(index)
            current = next_point

    if not ordered_segments:
        empty = np.empty((0, 3), dtype=float)
        return empty, empty
    lengths = np.asarray([segment[3] for segment in ordered_segments], dtype=float)
    cumulative = np.cumsum(lengths)
    total = float(cumulative[-1])
    count = _DISTRIBUTED_SYMBOL_LIMITS.get(
        density,
        _DISTRIBUTED_SYMBOL_LIMITS["medium"],
    )
    targets = total * (np.arange(count, dtype=float) + 0.5) / count
    sampled_points: list[np.ndarray] = []
    sampled_vectors: list[np.ndarray] = []
    for target in targets:
        segment_index = min(
            int(np.searchsorted(cumulative, target, side="right")),
            len(ordered_segments) - 1,
        )
        start_distance = cumulative[segment_index] - lengths[segment_index]
        fraction = (target - start_distance) / lengths[segment_index]
        first, last, vector, _length = ordered_segments[segment_index]
        sampled_points.append(first + fraction * (last - first))
        sampled_vectors.append(vector)
    return np.asarray(sampled_points), np.asarray(sampled_vectors)


def sample_distributed_faces(
    face_points: list[np.ndarray],
    face_vectors: list[np.ndarray],
    density: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample polygonal members by area with a deterministic spatial ordering."""
    if len(face_points) != len(face_vectors):
        raise ValueError("face points and vectors must have the same length")
    groups: dict[int, list[tuple[np.ndarray, np.ndarray]]] = {}
    for raw_points, raw_vector in zip(face_points, face_vectors):
        points = np.asarray(raw_points, dtype=float)
        if len(points) >= 3:
            groups.setdefault(len(points), []).append(
                (points, np.asarray(raw_vector, dtype=float))
            )
    triangle_first: list[np.ndarray] = []
    triangle_second: list[np.ndarray] = []
    triangle_third: list[np.ndarray] = []
    triangle_vectors: list[np.ndarray] = []
    triangle_areas: list[np.ndarray] = []
    for vertex_count, group in groups.items():
        polygons = np.asarray([entry[0] for entry in group])
        vectors = np.asarray([entry[1] for entry in group])
        triangles_per_face = vertex_count - 2
        first = np.repeat(polygons[:, :1, :], triangles_per_face, axis=1).reshape((-1, 3))
        second = polygons[:, 1:-1, :].reshape((-1, 3))
        third = polygons[:, 2:, :].reshape((-1, 3))
        areas = 0.5 * np.linalg.norm(
            np.cross(second - first, third - first),
            axis=1,
        )
        valid = areas > 0.0
        triangle_first.append(first[valid])
        triangle_second.append(second[valid])
        triangle_third.append(third[valid])
        triangle_vectors.append(
            np.repeat(vectors, triangles_per_face, axis=0)[valid]
        )
        triangle_areas.append(areas[valid])
    if not triangle_first or not any(len(group) for group in triangle_first):
        empty = np.empty((0, 3), dtype=float)
        return empty, empty

    first = np.concatenate(triangle_first)
    second = np.concatenate(triangle_second)
    third = np.concatenate(triangle_third)
    vectors = np.concatenate(triangle_vectors)
    areas = np.concatenate(triangle_areas)
    centroids = (first + second + third) / 3.0
    spatial_keys = _morton_keys(centroids)
    order = np.lexsort(
        (centroids[:, 2], centroids[:, 1], centroids[:, 0], spatial_keys)
    )
    first = first[order]
    second = second[order]
    third = third[order]
    vectors = vectors[order]
    areas = areas[order]
    cumulative = np.cumsum(areas)
    count = _DISTRIBUTED_SYMBOL_LIMITS.get(
        density,
        _DISTRIBUTED_SYMBOL_LIMITS["medium"],
    )
    candidate_count = 12 * count
    targets = (
        float(cumulative[-1])
        * (np.arange(candidate_count, dtype=float) + 0.5)
        / candidate_count
    )
    triangle_indices = np.searchsorted(cumulative, targets, side="right")
    triangle_indices = np.clip(triangle_indices, 0, len(first) - 1)
    roots = np.sqrt(np.asarray([
        _radical_inverse(index + 1, 2) for index in range(candidate_count)
    ]))
    across = np.asarray([
        _radical_inverse(index + 1, 3) for index in range(candidate_count)
    ])
    candidates = (
        (1.0 - roots[:, None]) * first[triangle_indices]
        + (roots * (1.0 - across))[:, None] * second[triangle_indices]
        + (roots * across)[:, None] * third[triangle_indices]
    )
    candidate_vectors = vectors[triangle_indices]
    centers = candidates[
        np.linspace(0, candidate_count - 1, count, dtype=np.int64)
    ].copy()
    assignments = np.zeros(candidate_count, dtype=np.int64)
    for _iteration in range(6):
        delta = candidates[:, None, :] - centers[None, :, :]
        assignments = np.argmin(np.einsum("ijk,ijk->ij", delta, delta), axis=1)
        for cluster in range(count):
            members = candidates[assignments == cluster]
            if len(members):
                centers[cluster] = np.mean(members, axis=0)
    selected: list[int] = []
    for cluster, center in enumerate(centers):
        members = np.flatnonzero(assignments == cluster)
        if len(members) == 0:
            continue
        offsets = candidates[members] - center
        selected.append(int(members[np.argmin(np.einsum("ij,ij->i", offsets, offsets))]))
    return candidates[selected], candidate_vectors[selected]


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
    return _spatial_sample_indices(points, density, _DISTRIBUTED_SYMBOL_LIMITS)


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
        coordinates = centered[order] @ right[0]
        targets = np.linspace(coordinates[0], coordinates[-1], limit)
        offsets = _nearest_sorted_offsets(coordinates, targets)
        return order[offsets].astype(np.int64)
    if dimension == 2:
        projected = centered @ right[:2].T
        hull = _convex_hull_indices(projected)
        if len(hull) >= 3:
            hull_indices = np.asarray(hull, dtype=np.int64)
            boundary_limits = {"low": 6, "medium": 12, "high": 24}
            boundary_limit = boundary_limits.get(
                density,
                boundary_limits["medium"],
            )
            if len(hull_indices) <= boundary_limit:
                return hull_indices
            offsets = _closed_polyline_sample_offsets(
                projected[hull_indices],
                boundary_limit,
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
        while len(lower) >= 2 and cross(lower[-2], lower[-1], index) < -1.0e-12:
            lower.pop()
        lower.append(index)
    upper: list[int] = []
    for index in reversed(ordered):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], index) < -1.0e-12:
            upper.pop()
        upper.append(index)
    return lower[:-1] + upper[:-1]


def _point_key(point: np.ndarray) -> tuple[float, ...]:
    return tuple(float(value) for value in np.asarray(point, dtype=float))


def _other_endpoint(
    endpoints: tuple[tuple[float, ...], tuple[float, ...]],
    point: tuple[float, ...],
) -> tuple[float, ...]:
    return endpoints[1] if endpoints[0] == point else endpoints[0]


def _line_member_component(
    seed: int,
    remaining: set[int],
    endpoints: list[tuple[tuple[float, ...], tuple[float, ...]]],
    adjacency: dict[tuple[float, ...], list[int]],
) -> set[int]:
    component: set[int] = set()
    pending = [seed]
    while pending:
        index = pending.pop()
        if index in component or index not in remaining:
            continue
        component.add(index)
        for endpoint in endpoints[index]:
            pending.extend(adjacency[endpoint])
    return component


def _morton_keys(points: np.ndarray) -> np.ndarray:
    sides = np.ptp(points, axis=0)
    normalized = np.zeros_like(points)
    active = sides > 1.0e-12
    normalized[:, active] = (
        points[:, active] - np.min(points[:, active], axis=0)
    ) / sides[active]
    quantized = np.rint(1023.0 * normalized).astype(np.uint64)
    keys = np.zeros(len(points), dtype=np.uint64)
    for bit in range(10):
        for axis in range(3):
            keys |= ((quantized[:, axis] >> bit) & 1) << (3 * bit + axis)
    return keys


def _radical_inverse(value: int, base: int) -> float:
    result = 0.0
    fraction = 1.0 / base
    while value:
        value, digit = divmod(value, base)
        result += digit * fraction
        fraction /= base
    return result


def _nearest_sorted_offsets(
    coordinates: np.ndarray,
    targets: np.ndarray,
) -> np.ndarray:
    upper = np.searchsorted(coordinates, targets, side="left")
    upper = np.clip(upper, 0, len(coordinates) - 1)
    lower = np.maximum(upper - 1, 0)
    choose_lower = (
        np.abs(targets - coordinates[lower])
        <= np.abs(coordinates[upper] - targets)
    )
    offsets = np.where(choose_lower, lower, upper)
    return np.asarray(list(dict.fromkeys(int(value) for value in offsets)), dtype=np.int64)


def _closed_polyline_sample_offsets(points: np.ndarray, count: int) -> np.ndarray:
    closed = np.vstack((points, points[0]))
    lengths = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    total = float(np.sum(lengths))
    if total <= 0.0:
        return np.asarray([0], dtype=np.int64)
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    targets = total * np.arange(count, dtype=float) / count
    segments = np.searchsorted(cumulative, targets, side="right") - 1
    segments = np.clip(segments, 0, len(points) - 1)
    next_segments = (segments + 1) % len(points)
    local = targets - cumulative[segments]
    choose_next = local > 0.5 * lengths[segments]
    offsets = np.where(choose_next, next_segments, segments)
    return np.asarray(list(dict.fromkeys(int(value) for value in offsets)), dtype=np.int64)


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
