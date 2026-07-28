"""Pure, backend-neutral analysis of native geometry recipes."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Literal

from .recipes import (
    BooleanGeometry,
    BoxGeometry,
    CylinderGeometry,
    DiskGeometry,
    ExtrudedGeometry,
    MovedGeometry,
    NATIVE_GEOMETRY_TYPES,
    NativeGeometry,
    PlateWithHoleGeometry,
    RectangleGeometry,
    RotatedGeometry,
    SketchArc,
    SketchCircle,
    SketchContour,
    SketchGeometry,
    SketchLine,
    SketchPlane,
    SketchPoint,
    SketchRectangle,
    WireGeometry,
)


_AXIS_ALIGNMENT_ABS_TOLERANCE = 1.0e-12


@dataclass(frozen=True, slots=True)
class RectangleFrame:
    """One proven axis-aligned rectangle in global XY coordinates."""

    x: float
    y: float
    width: float
    height: float

    def __post_init__(self) -> None:
        values = tuple(
            float(value) for value in (self.x, self.y, self.width, self.height)
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("rectangle frame values must be finite")
        if values[2] <= 0.0 or values[3] <= 0.0:
            raise ValueError("rectangle frame dimensions must be positive")
        for name, value in zip(
            ("x", "y", "width", "height"),
            values,
            strict=True,
        ):
            object.__setattr__(self, name, value)

    def strictly_contains_rectangle(self, other: RectangleFrame) -> bool:
        """Return whether ``other`` lies strictly inside this frame."""

        if type(other) is not RectangleFrame:
            raise TypeError("other must be a RectangleFrame")
        return (
            self.x < other.x
            and other.x + other.width < self.x + self.width
            and self.y < other.y
            and other.y + other.height < self.y + self.height
        )

    def __iter__(self):
        yield self.x
        yield self.y
        yield self.width
        yield self.height

    def strictly_contains_circle(self, other: CircleFrame) -> bool:
        """Return whether ``other`` lies strictly inside this frame."""

        if type(other) is not CircleFrame:
            raise TypeError("other must be a CircleFrame")
        return (
            self.x < other.center_x - other.radius
            and other.center_x + other.radius < self.x + self.width
            and self.y < other.center_y - other.radius
            and other.center_y + other.radius < self.y + self.height
        )


@dataclass(frozen=True, slots=True)
class CircleFrame:
    """One proven circle in global XY coordinates."""

    center_x: float
    center_y: float
    radius: float

    def __post_init__(self) -> None:
        values = tuple(
            float(value) for value in (self.center_x, self.center_y, self.radius)
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("circle frame values must be finite")
        if values[2] <= 0.0:
            raise ValueError("circle frame radius must be positive")
        for name, value in zip(
            ("center_x", "center_y", "radius"),
            values,
            strict=True,
        ):
            object.__setattr__(self, name, value)

    @property
    def x(self) -> float:
        """Return the global X coordinate of the center."""

        return self.center_x

    @property
    def y(self) -> float:
        """Return the global Y coordinate of the center."""

        return self.center_y

    def __iter__(self):
        yield self.center_x
        yield self.center_y
        yield self.radius


@dataclass(frozen=True, slots=True)
class SketchDiagnostic:
    """One actionable, backend-neutral sketch validation diagnostic."""

    code: str
    message: str
    entity_ids: tuple[str, ...] = ()
    blocking: bool = True
    severity: Literal["error", "warning"] = "error"

    def __post_init__(self) -> None:
        if type(self.code) is not str or not self.code.strip():
            raise ValueError("sketch diagnostic code cannot be empty")
        if type(self.message) is not str or not self.message.strip():
            raise ValueError("sketch diagnostic message cannot be empty")
        entity_ids = tuple(self.entity_ids)
        if any(type(item) is not str or not item.strip() for item in entity_ids):
            raise TypeError("sketch diagnostic entity IDs must be non-empty strings")
        if type(self.blocking) is not bool:
            raise TypeError("sketch diagnostic blocking must be a boolean")
        if self.severity not in {"error", "warning"}:
            raise ValueError("sketch diagnostic severity must be error or warning")
        object.__setattr__(self, "entity_ids", entity_ids)

    @property
    def affected_ids(self) -> tuple[str, ...]:
        """Compatibility alias used by editor and topology projections."""

        return self.entity_ids

    @property
    def ids(self) -> tuple[str, ...]:
        return self.entity_ids

    @property
    def entity_id(self) -> str | None:
        return self.entity_ids[0] if len(self.entity_ids) == 1 else None


@dataclass(frozen=True, slots=True)
class SketchProfile:
    """One deterministic closed curve loop and its nesting classification."""

    id: str
    curve_ids: tuple[str, ...]
    nesting_depth: int
    role: Literal["outer", "hole"]
    signed_area: float
    bounding_box: tuple[float, float, float, float]
    parent_profile_id: str | None = None

    def __post_init__(self) -> None:
        if type(self.id) is not str or not self.id.strip():
            raise ValueError("sketch profile id cannot be empty")
        curve_ids = tuple(self.curve_ids)
        if not curve_ids or any(type(item) is not str or not item for item in curve_ids):
            raise ValueError("sketch profile must contain ordered curve IDs")
        if isinstance(self.nesting_depth, bool) or self.nesting_depth < 0:
            raise ValueError("sketch profile nesting depth must be non-negative")
        if self.role not in {"outer", "hole"}:
            raise ValueError("sketch profile role must be outer or hole")
        if not math.isfinite(float(self.signed_area)) or self.signed_area == 0.0:
            raise ValueError("sketch profile area must be finite and non-zero")
        if len(self.bounding_box) != 4 or not all(
            math.isfinite(float(value)) for value in self.bounding_box
        ):
            raise ValueError("sketch profile bounding_box must contain finite values")
        if self.bounding_box[2] < self.bounding_box[0] or self.bounding_box[3] < self.bounding_box[1]:
            raise ValueError("sketch profile bounding_box must be ordered")
        object.__setattr__(self, "curve_ids", curve_ids)
        object.__setattr__(self, "signed_area", float(self.signed_area))
        object.__setattr__(self, "bounding_box", tuple(float(value) for value in self.bounding_box))

    @property
    def ordered_curve_ids(self) -> tuple[str, ...]:
        """Return the ordered signed curve references."""

        return self.curve_ids

    @property
    def curve_graph(self) -> tuple[str, ...]:
        """Return the curve members without changing their signed order."""

        return self.curve_ids

    @property
    def is_hole(self) -> bool:
        return self.role == "hole"

    @property
    def is_material(self) -> bool:
        return self.role == "outer"

    @property
    def area(self) -> float:
        return self.signed_area

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        return self.bounding_box

    @property
    def parent_id(self) -> str | None:
        return self.parent_profile_id

    @property
    def profile_id(self) -> str:
        return self.id


@dataclass(frozen=True, slots=True)
class SketchProfileAnalysis:
    """Pure profile-analysis output that can be unpacked by callers."""

    profiles: tuple[SketchProfile, ...]
    diagnostics: tuple[SketchDiagnostic, ...]

    def __post_init__(self) -> None:
        profiles = tuple(self.profiles)
        diagnostics = tuple(self.diagnostics)
        if any(type(item) is not SketchProfile for item in profiles):
            raise TypeError("profiles must contain only SketchProfile values")
        if any(type(item) is not SketchDiagnostic for item in diagnostics):
            raise TypeError("diagnostics must contain only SketchDiagnostic values")
        object.__setattr__(self, "profiles", profiles)
        object.__setattr__(self, "diagnostics", diagnostics)

    @property
    def valid(self) -> bool:
        return bool(self.profiles) and not any(
            diagnostic.blocking for diagnostic in self.diagnostics
        )

    @property
    def blocking_diagnostics(self) -> tuple[SketchDiagnostic, ...]:
        return tuple(item for item in self.diagnostics if item.blocking)

    @property
    def errors(self) -> tuple[SketchDiagnostic, ...]:
        return self.blocking_diagnostics

    @property
    def is_valid(self) -> bool:
        return self.valid

    def __len__(self) -> int:
        return len(self.profiles)

    def __getitem__(self, index: int) -> SketchProfile:
        return self.profiles[index]

    def __iter__(self):
        yield self.profiles
        yield self.diagnostics


@dataclass(frozen=True, slots=True)
class _SketchLoop:
    curve_ids: tuple[str, ...]
    polyline: tuple[tuple[float, float], ...]
    area: float
    bounding_box: tuple[float, float, float, float]


def legacy_sketch_to_strict(recipe: SketchGeometry) -> SketchGeometry:
    """Convert a legacy material/cut contour sketch into a curve graph.

    The conversion is deterministic and intentionally leaves material-versus-
    hole classification to loop containment analysis.  It is useful for
    opening old projects and for compatibility tools; new files are encoded
    directly from the strict curve graph.
    """

    if not isinstance(recipe, SketchGeometry):
        raise TypeError("recipe must be a SketchGeometry")
    if recipe.is_strict:
        return recipe
    points: list[SketchPoint] = []
    curves: list[SketchLine | SketchCircle] = []
    point_counter = 0
    curve_counter = 0

    def add_point(u: float, v: float) -> str:
        nonlocal point_counter
        point_counter += 1
        point_id = f"P{point_counter}"
        points.append(SketchPoint(point_id, u, v))
        return point_id

    def add_line(start: str, end: str) -> None:
        nonlocal curve_counter
        curve_counter += 1
        curves.append(SketchLine(f"L{curve_counter}", start, end))

    for contour in recipe.contours:
        if isinstance(contour, SketchRectangle):
            left = contour.x
            right = contour.x + contour.width
            bottom = contour.y
            top = contour.y + contour.height
            bottom_left = add_point(left, bottom)
            bottom_right = add_point(right, bottom)
            top_right = add_point(right, top)
            top_left = add_point(left, top)
            add_line(bottom_left, bottom_right)
            add_line(bottom_right, top_right)
            add_line(top_right, top_left)
            add_line(top_left, bottom_left)
        elif isinstance(contour, SketchCircle) and contour.is_legacy:
            center_id = add_point(contour.x, contour.y)
            curve_counter += 1
            curves.append(SketchCircle(f"C{curve_counter}", center_id, contour.radius))
        else:  # pragma: no cover - SketchGeometry validates legacy contours
            raise TypeError(f"Unsupported legacy sketch contour: {type(contour).__name__}")
    return SketchGeometry(recipe.name, SketchPlane.xy(), tuple(points), tuple(curves))


def legacy_sketches_to_strict(recipe: NativeGeometry) -> NativeGeometry:
    """Recursively canonicalize legacy sketches in one feature tree."""

    _require_native_geometry(recipe)
    if type(recipe) is SketchGeometry:
        return legacy_sketch_to_strict(recipe) if recipe.is_legacy else recipe
    if type(recipe) is MovedGeometry:
        base = legacy_sketches_to_strict(recipe.base)
        if base is recipe.base:
            return recipe
        return MovedGeometry(
            base,
            recipe.dx,
            recipe.dy,
            recipe.dz,
        )
    if type(recipe) is RotatedGeometry:
        base = legacy_sketches_to_strict(recipe.base)
        if base is recipe.base:
            return recipe
        return RotatedGeometry(
            base,
            recipe.axis,
            recipe.angle_degrees,
        )
    if type(recipe) is ExtrudedGeometry:
        base = legacy_sketches_to_strict(recipe.base)
        if base is recipe.base:
            return recipe
        return ExtrudedGeometry(
            base,
            recipe.height,
            recipe.source_face_ids,
        )
    if type(recipe) is BooleanGeometry:
        object_geometry = legacy_sketches_to_strict(
            recipe.object_geometry
        )
        tool_geometry = legacy_sketches_to_strict(recipe.tool_geometry)
        if (
            object_geometry is recipe.object_geometry
            and tool_geometry is recipe.tool_geometry
        ):
            return recipe
        return BooleanGeometry(
            recipe.name,
            recipe.operation,
            object_geometry,
            tool_geometry,
        )
    return recipe


def analyze_sketch_profiles(
    sketch: SketchGeometry,
    *,
    tolerance: float = 1.0e-8,
) -> SketchProfileAnalysis:
    """Analyze strict sketch loops without constructing OCC geometry."""

    if not isinstance(sketch, SketchGeometry):
        raise TypeError("sketch must be a SketchGeometry")
    if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)):
        raise TypeError("tolerance must be a finite real number")
    tolerance_value = float(tolerance)
    if not math.isfinite(tolerance_value) or tolerance_value <= 0.0:
        raise ValueError("tolerance must be a positive finite real number")
    if sketch.is_legacy:
        sketch = legacy_sketch_to_strict(sketch)
    return _analyze_strict_sketch(sketch, tolerance_value)


def _analyze_strict_sketch(
    sketch: SketchGeometry,
    tolerance: float,
) -> SketchProfileAnalysis:
    points = {point.id: (point.u, point.v) for point in sketch.points}
    diagnostics: list[SketchDiagnostic] = []
    edges = tuple(sketch.curves)
    edge_by_id = {curve.id: curve for curve in edges}
    point_ids_by_curve = {
        curve.id: _curve_point_ids_for_analysis(curve) for curve in edges
    }
    polylines = {
        curve.id: _curve_polyline_for_analysis(curve, points, tolerance)
        for curve in edges
    }

    adjacency: dict[str, list[tuple[str, str]]] = {}
    for curve in edges:
        curve_point_ids = point_ids_by_curve[curve.id]
        if isinstance(curve, SketchCircle):
            continue
        start, end = curve_point_ids[0], curve_point_ids[-1]
        adjacency.setdefault(start, []).append((curve.id, end))
        adjacency.setdefault(end, []).append((curve.id, start))
    for point_id, connections in sorted(adjacency.items()):
        degree = len(connections)
        if degree == 1:
            diagnostics.append(
                SketchDiagnostic(
                    "sketch.open-loop",
                    f"点 {point_id} 是开放轮廓端点",
                    (point_id,),
                )
            )
        elif degree > 2:
            diagnostics.append(
                SketchDiagnostic(
                    "sketch.t-junction",
                    f"点 {point_id} 连接了 {degree} 条曲线，轮廓不是唯一闭环",
                    (point_id,) + tuple(sorted(item[0] for item in connections)),
                )
            )
        elif degree == 0:  # pragma: no cover - adjacency omits isolated points
            diagnostics.append(
                SketchDiagnostic(
                    "sketch.unused-point",
                    f"点 {point_id} 没有连接曲线",
                    (point_id,),
                    blocking=False,
                    severity="warning",
                )
            )

    loops: list[_SketchLoop] = []
    visited_edges: set[str] = set()
    for curve in sorted(edges, key=lambda item: item.id.casefold()):
        if curve.id in visited_edges:
            continue
        if isinstance(curve, SketchCircle):
            visited_edges.add(curve.id)
            polyline = polylines[curve.id]
            loop = _make_loop((curve.id,), polyline, tolerance)
            loops.append(
                _replace_loop_area(
                    loop,
                    abs(_exact_curve_loop_area((curve.id,), edge_by_id, points)),
                )
            )
            continue
        component_edges, component_points = _component_for_edge(
            curve.id,
            adjacency,
            edge_by_id,
        )
        visited_edges.update(component_edges)
        if any(
            len(adjacency.get(point_id, ())) != 2 for point_id in component_points
        ):
            continue
        traced = _trace_component_loop(
            component_edges,
            component_points,
            adjacency,
            edge_by_id,
            polylines,
        )
        if traced is None:
            diagnostics.append(
                SketchDiagnostic(
                    "sketch.open-loop",
                    "曲线图无法按唯一顺序闭合",
                    tuple(sorted(component_edges)),
                )
            )
            continue
        signed_ids, polyline = traced
        loop = _make_loop(signed_ids, polyline, tolerance)
        loop = _replace_loop_area(
            loop,
            abs(_exact_curve_loop_area(signed_ids, edge_by_id, points)),
        )
        if abs(loop.area) <= tolerance * tolerance:
            diagnostics.append(
                SketchDiagnostic(
                    "sketch.zero-area-profile",
                    "闭环面积在几何容差内为零",
                    tuple(sorted(component_edges)),
                )
            )
            continue
        loops.append(loop)

    diagnostics.extend(_curve_intersection_diagnostics(edges, polylines, points, tolerance))
    if not loops:
        diagnostics.append(
            SketchDiagnostic(
                "sketch.no-profile",
                "草图没有可以提交的闭合 Profile",
            )
        )
        return SketchProfileAnalysis((), tuple(_deduplicate_diagnostics(diagnostics)))

    containments = _loop_containment_diagnostics(loops, tolerance)
    diagnostics.extend(containments[1])
    contains = containments[0]
    loop_ids = tuple(_profile_id(loop.curve_ids) for loop in loops)
    profile_values: list[SketchProfile] = []
    for index, loop in enumerate(loops):
        parent_indices = [candidate for candidate in contains[index]]
        depth = len(parent_indices)
        parent_index = None
        if parent_indices:
            parent_index = max(
                parent_indices,
                key=lambda candidate: abs(loops[candidate].area),
            )
        role: Literal["outer", "hole"] = "outer" if depth % 2 == 0 else "hole"
        profile_values.append(
            SketchProfile(
                loop_ids[index],
                loop.curve_ids,
                depth,
                role,
                loop.area,
                loop.bounding_box,
                None if parent_index is None else loop_ids[parent_index],
            )
        )
    profiles = tuple(
        sorted(
            profile_values,
            key=lambda profile: (
                profile.nesting_depth,
                profile.bounding_box[0],
                profile.bounding_box[1],
                profile.id,
            ),
        )
    )
    if not any(profile.is_material for profile in profiles):
        diagnostics.append(
            SketchDiagnostic(
                "sketch.no-material-profile",
                "草图没有材料 Profile",
            )
        )
    return SketchProfileAnalysis(
        profiles,
        tuple(_deduplicate_diagnostics(diagnostics)),
    )


def _curve_point_ids_for_analysis(curve: object) -> tuple[str, ...]:
    if isinstance(curve, SketchLine):
        return curve.start_point_id, curve.end_point_id
    if isinstance(curve, SketchArc):
        return curve.start_point_id, curve.center_point_id, curve.end_point_id
    if isinstance(curve, SketchCircle) and curve.is_curve:
        return (curve.center_point_id,)
    raise TypeError(f"Unsupported strict sketch curve: {type(curve).__name__}")


def _curve_polyline_for_analysis(
    curve: object,
    points: dict[str, tuple[float, float]],
    tolerance: float,
    *,
    reverse: bool = False,
) -> tuple[tuple[float, float], ...]:
    if isinstance(curve, SketchLine):
        result = (points[curve.start_point_id], points[curve.end_point_id])
        return result[::-1] if reverse else result
    if isinstance(curve, SketchCircle):
        center = points[curve.center_point_id]
        count = 72
        result = tuple(
            (
                center[0] + curve.radius * math.cos(2.0 * math.pi * index / count),
                center[1] + curve.radius * math.sin(2.0 * math.pi * index / count),
            )
            for index in range(count)
        )
        return tuple(reversed(result)) if reverse else result
    if isinstance(curve, SketchArc):
        start_id = curve.start_point_id
        end_id = curve.end_point_id
        orientation = curve.orientation
        if reverse:
            start_id, end_id = end_id, start_id
            orientation = "cw" if orientation == "ccw" else "ccw"
        start = points[start_id]
        center = points[curve.center_point_id]
        end = points[end_id]
        start_angle = math.atan2(start[1] - center[1], start[0] - center[0])
        end_angle = math.atan2(end[1] - center[1], end[0] - center[0])
        if orientation == "ccw":
            sweep = (end_angle - start_angle) % (2.0 * math.pi)
        else:
            sweep = -((start_angle - end_angle) % (2.0 * math.pi))
        if abs(sweep) <= tolerance:
            sweep = 2.0 * math.pi if orientation == "ccw" else -2.0 * math.pi
        radius = math.hypot(start[0] - center[0], start[1] - center[1])
        count = max(8, min(256, int(math.ceil(abs(sweep) * radius / 0.25))))
        return tuple(
            (
                center[0] + radius * math.cos(start_angle + sweep * index / count),
                center[1] + radius * math.sin(start_angle + sweep * index / count),
            )
            for index in range(count + 1)
        )
    raise TypeError(f"Unsupported strict sketch curve: {type(curve).__name__}")


def _component_for_edge(
    curve_id: str,
    adjacency: dict[str, list[tuple[str, str]]],
    edge_by_id: dict[str, object],
) -> tuple[set[str], set[str]]:
    del edge_by_id
    edge_ids: set[str] = set()
    point_ids: set[str] = set()
    queue = [curve_id]
    while queue:
        current = queue.pop()
        if current in edge_ids:
            continue
        edge_ids.add(current)
        for point_id, connections in adjacency.items():
            if any(edge == current for edge, _other in connections):
                point_ids.add(point_id)
                queue.extend(
                    edge for edge, _other in connections if edge not in edge_ids
                )
    return edge_ids, point_ids


def _trace_component_loop(
    component_edges: set[str],
    component_points: set[str],
    adjacency: dict[str, list[tuple[str, str]]],
    edge_by_id: dict[str, object],
    polylines: dict[str, tuple[tuple[float, float], ...]],
) -> tuple[tuple[str, ...], tuple[tuple[float, float], ...]] | None:
    start = min(component_points, key=str.casefold)
    current = start
    previous_edge: str | None = None
    signed_ids: list[str] = []
    path: list[tuple[float, float]] = []
    max_steps = len(component_edges) + 1
    for _ in range(max_steps):
        candidates = sorted(adjacency[current], key=lambda item: item[0].casefold())
        candidates = [item for item in candidates if item[0] != previous_edge]
        if not candidates:
            return None
        edge_id, next_point = candidates[0]
        if edge_id in signed_ids or f"-{edge_id}" in signed_ids:
            return None if next_point != start else (tuple(signed_ids), tuple(path))
        curve = edge_by_id[edge_id]
        curve_points = _curve_point_ids_for_analysis(curve)
        forward = current == curve_points[0]
        signed_ids.append(edge_id if forward else f"-{edge_id}")
        edge_polyline = polylines[edge_id]
        if not forward:
            edge_polyline = tuple(reversed(edge_polyline))
        if not path:
            path.extend(edge_polyline)
        else:
            path.extend(edge_polyline[1:])
        previous_edge = edge_id
        current = next_point
        if current == start:
            if len(signed_ids) != len(component_edges):
                return None
            return tuple(signed_ids), tuple(path)
    return None


def _make_loop(
    curve_ids: tuple[str, ...],
    polyline: tuple[tuple[float, float], ...],
    tolerance: float,
) -> _SketchLoop:
    area = _polyline_area(polyline)
    if area < 0.0:
        curve_ids = tuple(_flip_signed_curve_id(item) for item in reversed(curve_ids))
        polyline = tuple(reversed(polyline))
        area = -area
    coordinates = polyline[:-1] if polyline and polyline[0] == polyline[-1] else polyline
    xs = tuple(point[0] for point in coordinates)
    ys = tuple(point[1] for point in coordinates)
    if not xs or not ys:
        return _SketchLoop(curve_ids, polyline, area, (0.0, 0.0, 0.0, 0.0))
    del tolerance
    return _SketchLoop(
        curve_ids,
        polyline,
        area,
        (min(xs), min(ys), max(xs), max(ys)),
    )


def _replace_loop_area(loop: _SketchLoop, area: float) -> _SketchLoop:
    return _SketchLoop(loop.curve_ids, loop.polyline, area, loop.bounding_box)


def _exact_curve_loop_area(
    signed_curve_ids: tuple[str, ...],
    edge_by_id: dict[str, object],
    points: dict[str, tuple[float, float]],
) -> float:
    total = 0.0
    for signed_id in signed_curve_ids:
        reversed_curve = signed_id.startswith("-")
        curve = edge_by_id[signed_id.lstrip("-")]
        if isinstance(curve, SketchLine):
            start_id, end_id = curve.start_point_id, curve.end_point_id
            if reversed_curve:
                start_id, end_id = end_id, start_id
            start = points[start_id]
            end = points[end_id]
            total += 0.5 * (start[0] * end[1] - end[0] * start[1])
            continue
        if isinstance(curve, SketchCircle):
            sign = -1.0 if reversed_curve else 1.0
            total += sign * math.pi * curve.radius * curve.radius
            continue
        if isinstance(curve, SketchArc):
            start_id, end_id = curve.start_point_id, curve.end_point_id
            orientation = curve.orientation
            if reversed_curve:
                start_id, end_id = end_id, start_id
                orientation = "cw" if orientation == "ccw" else "ccw"
            start = points[start_id]
            end = points[end_id]
            center = points[curve.center_point_id]
            start_angle = math.atan2(start[1] - center[1], start[0] - center[0])
            end_angle = math.atan2(end[1] - center[1], end[0] - center[0])
            if orientation == "ccw":
                delta = (end_angle - start_angle) % (2.0 * math.pi)
            else:
                delta = -((start_angle - end_angle) % (2.0 * math.pi))
            radius = math.hypot(start[0] - center[0], start[1] - center[1])
            total += 0.5 * (
                radius * center[0] * (math.sin(end_angle) - math.sin(start_angle))
                + radius * center[1] * (math.cos(start_angle) - math.cos(end_angle))
                + radius * radius * delta
            )
            continue
        raise TypeError(f"Unsupported strict sketch curve: {type(curve).__name__}")
    return total


def _flip_signed_curve_id(value: str) -> str:
    return value[1:] if value.startswith("-") else f"-{value}"


def _polyline_area(polyline: tuple[tuple[float, float], ...]) -> float:
    if len(polyline) < 3:
        return 0.0
    coordinates = polyline
    if coordinates[0] != coordinates[-1]:
        coordinates = coordinates + (coordinates[0],)
    return 0.5 * sum(
        left[0] * right[1] - right[0] * left[1]
        for left, right in zip(coordinates, coordinates[1:])
    )


def _profile_id(curve_ids: tuple[str, ...]) -> str:
    members = "|".join(sorted(item.lstrip("-") for item in curve_ids))
    digest = hashlib.sha1(members.encode("utf-8")).hexdigest()[:16]
    return f"profile/{digest}"


def _curve_intersection_diagnostics(
    curves: tuple[object, ...],
    polylines: dict[str, tuple[tuple[float, float], ...]],
    points: dict[str, tuple[float, float]],
    tolerance: float,
) -> tuple[SketchDiagnostic, ...]:
    diagnostics: list[SketchDiagnostic] = []
    for index, left in enumerate(curves):
        for right in curves[index + 1 :]:
            left_id = left.id
            right_id = right.id
            left_segments = _polyline_segments_for_intersections(left, polylines[left_id])
            right_segments = _polyline_segments_for_intersections(right, polylines[right_id])
            pair_code: str | None = None
            for left_segment in left_segments:
                for right_segment in right_segments:
                    intersection = _segment_intersection(
                        left_segment[0],
                        left_segment[1],
                        right_segment[0],
                        right_segment[1],
                        tolerance,
                    )
                    if intersection is None:
                        continue
                    kind, location = intersection
                    if kind == "overlap":
                        pair_code = "sketch.overlap"
                        break
                    if _is_shared_endpoint(location, left, right, points, tolerance):
                        continue
                    left_endpoint = _is_segment_endpoint(location, left_segment, tolerance)
                    right_endpoint = _is_segment_endpoint(location, right_segment, tolerance)
                    if left_endpoint and right_endpoint:
                        pair_code = "sketch.tangent-ambiguity"
                    elif left_endpoint or right_endpoint:
                        pair_code = "sketch.t-junction"
                    else:
                        pair_code = "sketch.crossing"
                    break
                if pair_code is not None:
                    break
            if pair_code is not None:
                diagnostics.append(
                    SketchDiagnostic(
                        pair_code,
                        f"曲线 {left_id} 和 {right_id} 的几何关系无法唯一解释",
                        (left_id, right_id),
                    )
                )
    return tuple(diagnostics)


def _polyline_segments_for_intersections(
    curve: object,
    polyline: tuple[tuple[float, float], ...],
) -> tuple[tuple[tuple[float, float], tuple[float, float]], ...]:
    if isinstance(curve, SketchCircle) and polyline:
        return tuple(zip(polyline, (*polyline[1:], polyline[0])))
    return tuple(zip(polyline, polyline[1:]))


def _segment_intersection(
    left_start: tuple[float, float],
    left_end: tuple[float, float],
    right_start: tuple[float, float],
    right_end: tuple[float, float],
    tolerance: float,
) -> tuple[str, tuple[float, float]] | None:
    def cross(a: tuple[float, float], b: tuple[float, float]) -> float:
        return a[0] * b[1] - a[1] * b[0]

    def subtract(a: tuple[float, float], b: tuple[float, float]) -> tuple[float, float]:
        return a[0] - b[0], a[1] - b[1]

    left_vector = subtract(left_end, left_start)
    right_vector = subtract(right_end, right_start)
    offset = subtract(right_start, left_start)
    denominator = cross(left_vector, right_vector)
    if abs(denominator) <= tolerance:
        if abs(cross(offset, left_vector)) > tolerance:
            return None
        length = math.hypot(*left_vector)
        if length <= tolerance:
            return None
        projection = (
            ((right_start[0] - left_start[0]) * left_vector[0]
             + (right_start[1] - left_start[1]) * left_vector[1]) / (length * length),
            ((right_end[0] - left_start[0]) * left_vector[0]
             + (right_end[1] - left_start[1]) * left_vector[1]) / (length * length),
        )
        low = max(0.0, min(projection))
        high = min(1.0, max(projection))
        if high - low <= tolerance:
            if high < -tolerance or low > 1.0 + tolerance:
                return None
            point = (
                left_start[0] + low * left_vector[0],
                left_start[1] + low * left_vector[1],
            )
            return "point", point
        midpoint = 0.5 * (low + high)
        return "overlap", (
            left_start[0] + midpoint * left_vector[0],
            left_start[1] + midpoint * left_vector[1],
        )
    t = cross(offset, right_vector) / denominator
    u = cross(offset, left_vector) / denominator
    if -tolerance <= t <= 1.0 + tolerance and -tolerance <= u <= 1.0 + tolerance:
        return "point", (
            left_start[0] + t * left_vector[0],
            left_start[1] + t * left_vector[1],
        )
    return None


def _is_segment_endpoint(
    location: tuple[float, float],
    segment: tuple[tuple[float, float], tuple[float, float]],
    tolerance: float,
) -> bool:
    return any(
        math.hypot(location[0] - point[0], location[1] - point[1]) <= tolerance
        for point in segment
    )


def _is_shared_endpoint(
    location: tuple[float, float],
    left: object,
    right: object,
    points: dict[str, tuple[float, float]],
    tolerance: float,
) -> bool:
    left_ids = set(_curve_point_ids_for_analysis(left))
    right_ids = set(_curve_point_ids_for_analysis(right))
    for point_id in left_ids & right_ids:
        point = points.get(point_id)
        if point is not None and math.hypot(location[0] - point[0], location[1] - point[1]) <= tolerance:
            return True
    return False


def _loop_containment_diagnostics(
    loops: list[_SketchLoop],
    tolerance: float,
) -> tuple[tuple[set[int], ...], tuple[SketchDiagnostic, ...]]:
    contains = [set() for _ in loops]
    diagnostics: list[SketchDiagnostic] = []
    for left_index, left in enumerate(loops):
        for right_index, right in enumerate(loops):
            if left_index == right_index:
                continue
            if not _bbox_may_contain(left.bounding_box, right.bounding_box, tolerance):
                continue
            candidate = right.polyline[0]
            location = _point_in_polyline(candidate, left.polyline, tolerance)
            if location == -1:
                diagnostics.append(
                    SketchDiagnostic(
                        "sketch.tangent-ambiguity",
                        "两个 Profile 的边界在几何容差内相切或重合",
                        tuple(sorted((*left.curve_ids, *right.curve_ids))),
                    )
                )
            elif location == 1:
                contains[right_index].add(left_index)
    return tuple(contains), tuple(diagnostics)


def _bbox_may_contain(
    outer: tuple[float, float, float, float],
    inner: tuple[float, float, float, float],
    tolerance: float,
) -> bool:
    return (
        outer[0] - tolerance <= inner[0]
        and inner[2] <= outer[2] + tolerance
        and outer[1] - tolerance <= inner[1]
        and inner[3] <= outer[3] + tolerance
    )


def _point_in_polyline(
    point: tuple[float, float],
    polyline: tuple[tuple[float, float], ...],
    tolerance: float,
) -> int:
    coordinates = polyline
    if coordinates[0] != coordinates[-1]:
        coordinates = coordinates + (coordinates[0],)
    inside = False
    for start, end in zip(coordinates, coordinates[1:]):
        if _point_on_segment(point, start, end, tolerance):
            return -1
        if (start[1] > point[1]) != (end[1] > point[1]):
            x_intersection = (end[0] - start[0]) * (point[1] - start[1]) / (
                end[1] - start[1]
            ) + start[0]
            if point[0] < x_intersection:
                inside = not inside
    return 1 if inside else 0


def _point_on_segment(
    point: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
    tolerance: float,
) -> bool:
    length = math.hypot(end[0] - start[0], end[1] - start[1])
    if length <= tolerance:
        return math.hypot(point[0] - start[0], point[1] - start[1]) <= tolerance
    cross = (point[0] - start[0]) * (end[1] - start[1]) - (
        point[1] - start[1]
    ) * (end[0] - start[0])
    if abs(cross) > tolerance * length:
        return False
    dot = (point[0] - start[0]) * (end[0] - start[0]) + (
        point[1] - start[1]
    ) * (end[1] - start[1])
    return -tolerance * length <= dot <= length * length + tolerance * length


def _deduplicate_diagnostics(
    diagnostics: list[SketchDiagnostic],
) -> tuple[SketchDiagnostic, ...]:
    result: list[SketchDiagnostic] = []
    seen: set[tuple[str, tuple[str, ...]]] = set()
    for diagnostic in diagnostics:
        key = diagnostic.code, diagnostic.entity_ids
        if key in seen:
            continue
        seen.add(key)
        result.append(diagnostic)
    return tuple(result)


def expand_sketch_recipe(recipe: SketchGeometry) -> NativeGeometry:
    """Return a detached evaluator recipe for one planar sketch.

    Contour indices and transforms follow the authoring order.  Material
    contours are fused in that order before cut contours are applied in their
    own authoring order, matching the native CAD construction contract.
    """

    if not isinstance(recipe, SketchGeometry):
        raise TypeError("recipe must be a SketchGeometry")
    if recipe.is_strict:
        raise ValueError(
            "strict curve-based sketches must be compiled through the sketch compiler"
        )

    def contour_geometry(
        contour: SketchContour,
        index: int,
    ) -> NativeGeometry:
        name = f"{recipe.name}-Contour-{index}"
        if isinstance(contour, SketchRectangle):
            result: NativeGeometry = RectangleGeometry(
                name,
                contour.width,
                contour.height,
            )
        elif isinstance(contour, SketchCircle):
            result = DiskGeometry(name, contour.radius)
        else:  # pragma: no cover - SketchGeometry validates contour types
            raise TypeError(f"Unsupported sketch contour: {type(contour).__name__}")
        if contour.x != 0.0 or contour.y != 0.0:
            result = MovedGeometry(result, contour.x, contour.y)
        return result

    indexed_contours = tuple(enumerate(recipe.contours, start=1))
    material = tuple(
        contour_geometry(contour, index)
        for index, contour in indexed_contours
        if contour.operation == "material"
    )
    cuts = tuple(
        contour_geometry(contour, index)
        for index, contour in indexed_contours
        if contour.operation == "cut"
    )
    result = material[0]
    for tool in material[1:]:
        result = BooleanGeometry(recipe.name, "fuse", result, tool)
    for tool in cuts:
        result = BooleanGeometry(recipe.name, "cut", result, tool)
    return result


def axis_aligned_rectangle(recipe: NativeGeometry) -> RectangleFrame | None:
    """Return the proven global XY rectangle frame, if one exists."""

    _require_native_geometry(recipe)
    return _axis_aligned_rectangle(recipe)


def _axis_aligned_rectangle(recipe: NativeGeometry) -> RectangleFrame | None:
    if isinstance(recipe, SketchGeometry):
        if recipe.is_strict:
            analysis = analyze_sketch_profiles(recipe)
            material = tuple(profile for profile in analysis.profiles if profile.is_material)
            if len(material) != 1 or analysis.blocking_diagnostics:
                return None
            profile = material[0]
            curve_ids = {item.lstrip("-") for item in profile.curve_ids}
            if len(curve_ids) != 4 or any(
                not isinstance(recipe.curve(curve_id), SketchLine)
                for curve_id in curve_ids
            ):
                return None
            x_min, y_min, x_max, y_max = profile.bounding_box
            return RectangleFrame(x_min, y_min, x_max - x_min, y_max - y_min)
        return _axis_aligned_rectangle(expand_sketch_recipe(recipe))
    if isinstance(recipe, RectangleGeometry):
        return RectangleFrame(0.0, 0.0, recipe.width, recipe.height)
    if isinstance(recipe, MovedGeometry):
        frame = _axis_aligned_rectangle(recipe.base)
        if frame is None or recipe.dz != 0.0:
            return None
        return RectangleFrame(
            frame.x + recipe.dx,
            frame.y + recipe.dy,
            frame.width,
            frame.height,
        )
    if isinstance(recipe, RotatedGeometry) and math.isclose(
        recipe.angle_degrees % 360.0,
        0.0,
        abs_tol=_AXIS_ALIGNMENT_ABS_TOLERANCE,
    ):
        return _axis_aligned_rectangle(recipe.base)
    return None


def transformed_circle(recipe: NativeGeometry) -> CircleFrame | None:
    """Return the proven global XY circle frame, if one exists."""

    _require_native_geometry(recipe)
    return _transformed_circle(recipe)


def _transformed_circle(recipe: NativeGeometry) -> CircleFrame | None:
    if isinstance(recipe, SketchGeometry):
        if recipe.is_strict:
            analysis = analyze_sketch_profiles(recipe)
            material = tuple(profile for profile in analysis.profiles if profile.is_material)
            if len(material) != 1 or analysis.blocking_diagnostics:
                return None
            curve_ids = tuple(item.lstrip("-") for item in material[0].curve_ids)
            if len(curve_ids) != 1:
                return None
            curve = recipe.curve(curve_ids[0])
            if not isinstance(curve, SketchCircle):
                return None
            center = recipe.point(curve.center_point_id)
            return CircleFrame(center.u, center.v, curve.radius)
        return _transformed_circle(expand_sketch_recipe(recipe))
    if isinstance(recipe, DiskGeometry):
        return CircleFrame(0.0, 0.0, recipe.radius)
    if isinstance(recipe, MovedGeometry):
        frame = _transformed_circle(recipe.base)
        if frame is None or recipe.dz != 0.0:
            return None
        return CircleFrame(
            frame.center_x + recipe.dx,
            frame.center_y + recipe.dy,
            frame.radius,
        )
    if isinstance(recipe, RotatedGeometry):
        frame = _transformed_circle(recipe.base)
        if frame is None:
            return None
        angle = math.radians(recipe.angle_degrees)
        cosine, sine = math.cos(angle), math.sin(angle)
        return CircleFrame(
            frame.center_x * cosine - frame.center_y * sine,
            frame.center_x * sine + frame.center_y * cosine,
            frame.radius,
        )
    return None


def recipe_characteristic_size(recipe: NativeGeometry) -> float:
    """Return the deterministic characteristic size used by native meshing."""

    _require_native_geometry(recipe)
    if isinstance(recipe, BooleanGeometry):
        return min(
            recipe_characteristic_size(recipe.object_geometry),
            recipe_characteristic_size(recipe.tool_geometry),
        )
    if isinstance(recipe, (MovedGeometry, RotatedGeometry)):
        return recipe_characteristic_size(recipe.base)
    if isinstance(recipe, ExtrudedGeometry):
        return min(recipe_characteristic_size(recipe.base), recipe.height)
    if isinstance(recipe, SketchGeometry):
        if recipe.is_strict:
            analysis = analyze_sketch_profiles(recipe)
            if not analysis.profiles:
                return 0.0
            return min(
                max(
                    profile.bounding_box[2] - profile.bounding_box[0],
                    profile.bounding_box[3] - profile.bounding_box[1],
                )
                for profile in analysis.profiles
            )
        return recipe_characteristic_size(expand_sketch_recipe(recipe))
    if isinstance(recipe, WireGeometry):
        points = {point.name: point for point in recipe.points}
        return max(
            math.hypot(
                points[member.start].x - points[member.end].x,
                points[member.start].y - points[member.end].y,
                points[member.start].z - points[member.end].z,
            )
            for member in recipe.members
        )
    if isinstance(recipe, (RectangleGeometry, PlateWithHoleGeometry)):
        return min(recipe.width, recipe.height)
    if isinstance(recipe, DiskGeometry):
        return 2.0 * recipe.radius
    if isinstance(recipe, BoxGeometry):
        return min(recipe.width, recipe.depth, recipe.height)
    if isinstance(recipe, CylinderGeometry):
        return min(2.0 * recipe.radius, recipe.height)
    raise AssertionError("native geometry dispatch is incomplete")


def supports_structured_hexahedron(recipe: NativeGeometry) -> bool:
    """Return whether the existing structured Hex mesher supports ``recipe``."""

    _require_native_geometry(recipe)
    if isinstance(recipe, (MovedGeometry, RotatedGeometry)):
        return supports_structured_hexahedron(recipe.base)
    if isinstance(recipe, BoxGeometry):
        return True
    if not isinstance(recipe, ExtrudedGeometry) or not isinstance(
        recipe.base,
        SketchGeometry,
    ):
        return False
    if recipe.base.is_strict:
        return False
    contours = recipe.base.contours
    return (
        len(contours) == 1
        and isinstance(contours[0], SketchRectangle)
        and contours[0].operation == "material"
    )


def _require_native_geometry(recipe: object) -> None:
    if not isinstance(recipe, NATIVE_GEOMETRY_TYPES):
        raise TypeError(
            f"Unsupported native geometry recipe: {type(recipe).__name__}"
        )


__all__ = [
    "CircleFrame",
    "RectangleFrame",
    "SketchDiagnostic",
    "SketchProfile",
    "SketchProfileAnalysis",
    "analyze_sketch_profiles",
    "axis_aligned_rectangle",
    "expand_sketch_recipe",
    "legacy_sketch_to_strict",
    "legacy_sketches_to_strict",
    "recipe_characteristic_size",
    "supports_structured_hexahedron",
    "transformed_circle",
]
