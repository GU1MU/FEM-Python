"""Detached bounded point and line previews for native geometry recipes."""

from __future__ import annotations

import math
from dataclasses import dataclass

from fem.geometry import (
    BooleanGeometry,
    BoxGeometry,
    CylinderGeometry,
    DiskGeometry,
    ExtrudedGeometry,
    MovedGeometry,
    MultiBodyGeometry,
    PlateWithHoleGeometry,
    PathSweptGeometry,
    RectangleGeometry,
    RevolvedGeometry,
    RotatedGeometry,
    SketchArc,
    SketchCircle,
    SketchGeometry,
    SketchLine,
    WireGeometry,
    geometry_dimension,
    planar_geometry_normal,
)

from .geometry_validation import _MAX_PREVIEW_POINTS, _as_strict_planar_sketch


_PREVIEW_SEGMENTS = 24


@dataclass(frozen=True, slots=True)
class StaticGeometryPreview:
    """Detached line preview with a fixed point budget."""

    dimension: int
    points: tuple[tuple[float, float, float], ...]
    lines: tuple[tuple[int, ...], ...]
    bounds: tuple[float, float, float, float, float, float]
    point_names: tuple[str, ...] = ()
    member_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.dimension not in {1, 2, 3}:
            raise ValueError("preview dimension must be 1, 2, or 3")
        points = tuple(self.points)
        lines = tuple(self.lines)
        point_names = tuple(self.point_names)
        member_names = tuple(self.member_names)
        if not points or len(points) > _MAX_PREVIEW_POINTS:
            raise ValueError("preview point count is outside the A2 bound")
        if any(
            len(point) != 3
            or any(not math.isfinite(float(component)) for component in point)
            for point in points
        ):
            raise ValueError("preview points must be finite XYZ triples")
        if any(
            len(line) < 2 or any(index < 0 or index >= len(points) for index in line)
            for line in lines
        ):
            raise ValueError("preview lines contain invalid point indices")
        if len(self.bounds) != 6 or any(
            not math.isfinite(float(value)) for value in self.bounds
        ):
            raise ValueError("preview bounds must contain six finite values")
        if point_names and len(point_names) != len(points):
            raise ValueError("preview point names must match preview points")
        if member_names and len(member_names) != len(lines):
            raise ValueError("preview member names must match preview lines")
        if any(not name.strip() for name in (*point_names, *member_names)):
            raise ValueError("preview entity names must not be blank")
        object.__setattr__(self, "point_names", point_names)
        object.__setattr__(self, "member_names", member_names)

    def to_dict(self) -> dict[str, object]:
        payload = {
            "kind": "bounded_wireframe",
            "dimension": self.dimension,
            "point_count": len(self.points),
            "line_count": len(self.lines),
            "points": [list(point) for point in self.points],
            "lines": [list(line) for line in self.lines],
            "bounds": list(self.bounds),
        }
        if self.point_names:
            payload["point_names"] = list(self.point_names)
        if self.member_names:
            payload["member_names"] = list(self.member_names)
        return payload


def _preview(
    recipe: object,
    *,
    point_budget: int = _MAX_PREVIEW_POINTS,
) -> StaticGeometryPreview:
    if (
        isinstance(point_budget, bool)
        or not isinstance(point_budget, int)
        or not 1 <= point_budget <= _MAX_PREVIEW_POINTS
    ):
        raise ValueError("preview point budget is outside the A2 bound")
    if type(recipe) is MovedGeometry:
        base = _preview(recipe.base, point_budget=point_budget)
        points = tuple(
            (x + recipe.dx, y + recipe.dy, z + recipe.dz) for x, y, z in base.points
        )
        return _make_preview(
            base.dimension,
            points,
            base.lines,
            point_names=base.point_names,
            member_names=base.member_names,
        )
    if type(recipe) is RotatedGeometry:
        base = _preview(recipe.base, point_budget=point_budget)
        angle = math.radians(recipe.angle_degrees)
        cosine, sine = math.cos(angle), math.sin(angle)

        def rotated(point: tuple[float, float, float]) -> tuple[float, float, float]:
            x, y, z = point
            if recipe.axis == "x":
                return x, y * cosine - z * sine, y * sine + z * cosine
            if recipe.axis == "y":
                return x * cosine + z * sine, y, -x * sine + z * cosine
            return x * cosine - y * sine, x * sine + y * cosine, z

        return _make_preview(
            base.dimension,
            tuple(map(rotated, base.points)),
            base.lines,
            point_names=base.point_names,
            member_names=base.member_names,
        )
    if type(recipe) is ExtrudedGeometry:
        base = _preview(
            recipe.base,
            point_budget=max(1, point_budget // 2),
        )
        count = len(base.points)
        if count * 2 > point_budget:
            raise ValueError("extruded preview exceeds the point budget")
        nx, ny, nz = planar_geometry_normal(recipe.base)
        points = base.points + tuple(
            (
                x + nx * recipe.height,
                y + ny * recipe.height,
                z + nz * recipe.height,
            )
            for x, y, z in base.points
        )
        lines = (
            base.lines
            + tuple(tuple(index + count for index in line) for line in base.lines)
            + tuple((index, index + count) for index in range(count))
        )
        return _make_preview(3, points, lines)
    if type(recipe) is RevolvedGeometry:
        base = _preview(recipe.base)
        samples = min(4, max(2, _MAX_PREVIEW_POINTS // len(base.points)))
        previews = []
        for index in range(samples):
            angle = math.radians(recipe.angle_degrees * index / (samples - 1))
            cosine, sine = math.cos(angle), math.sin(angle)

            def rotate(point: tuple[float, float, float]) -> tuple[float, float, float]:
                x, y, z = point
                if recipe.axis == "x":
                    return x, y * cosine - z * sine, y * sine + z * cosine
                if recipe.axis == "y":
                    return x * cosine + z * sine, y, -x * sine + z * cosine
                return x * cosine - y * sine, x * sine + y * cosine, z

            previews.append(
                _make_preview(
                    3,
                    tuple(rotate(point) for point in base.points),
                    base.lines,
                )
            )
        return _combine_previews(tuple(previews), 3)
    if type(recipe) is PathSweptGeometry:
        return _combine_previews((_preview(recipe.base), _preview(recipe.path)), 3)
    if type(recipe) is BooleanGeometry:
        object_budget = max(1, point_budget * 3 // 4)
        tool_budget = max(1, point_budget - object_budget)
        try:
            return _combine_previews(
                (
                    _preview(recipe.object_geometry, point_budget=object_budget),
                    _preview(recipe.tool_geometry, point_budget=tool_budget),
                ),
                geometry_dimension(recipe),
            )
        except ValueError:
            return _preview(recipe.tool_geometry, point_budget=point_budget)
    if type(recipe) is MultiBodyGeometry:
        return _combine_previews(
            tuple(_preview(body.recipe) for body in recipe.bodies),
            3,
        )
    if type(recipe) is WireGeometry:
        point_indexes = {point.name: index for index, point in enumerate(recipe.points)}
        return _make_preview(
            1,
            tuple((point.x, point.y, point.z) for point in recipe.points),
            tuple(
                (point_indexes[member.start], point_indexes[member.end])
                for member in recipe.members
            ),
            point_names=tuple(point.name for point in recipe.points),
            member_names=tuple(member.name for member in recipe.members),
        )
    if type(recipe) is RectangleGeometry:
        points = (
            (0.0, 0.0, 0.0),
            (recipe.width, 0.0, 0.0),
            (recipe.width, recipe.height, 0.0),
            (0.0, recipe.height, 0.0),
        )
        return _make_preview(2, points, ((0, 1, 2, 3, 0),))
    if type(recipe) is DiskGeometry:
        points = _circle_points(0.0, 0.0, recipe.radius, 0.0)
        return _make_preview(2, points, (tuple(range(len(points))) + (0,),))
    if type(recipe) is PlateWithHoleGeometry:
        outer = (
            (0.0, 0.0, 0.0),
            (recipe.width, 0.0, 0.0),
            (recipe.width, recipe.height, 0.0),
            (0.0, recipe.height, 0.0),
        )
        hole = _circle_points(
            recipe.hole_x,
            recipe.hole_y,
            recipe.hole_radius,
            0.0,
        )
        return _make_preview(
            2,
            outer + hole,
            (
                (0, 1, 2, 3, 0),
                tuple(range(4, 4 + len(hole))) + (4,),
            ),
        )
    if type(recipe) is SketchGeometry:
        sketch = _as_strict_planar_sketch(recipe)
        assert sketch.plane is not None
        points = tuple(
            sketch.plane.to_global(point.u, point.v) for point in sketch.points
        )
        if len(points) > point_budget:
            raise ValueError("sketch preview exceeds the point budget")
        point_indexes = {point.id: index for index, point in enumerate(sketch.points)}
        preview_points = list(points)
        lines: list[tuple[int, ...]] = []
        point_by_id = {point.id: point for point in sketch.points}
        curved_count = sum(
            isinstance(curve, (SketchCircle, SketchArc)) for curve in sketch.curves
        )
        curved_samples = (
            0
            if curved_count == 0
            else min(
                _PREVIEW_SEGMENTS,
                max(
                    0,
                    (point_budget - len(preview_points)) // curved_count,
                ),
            )
        )
        for curve in sketch.curves:
            if isinstance(curve, SketchLine):
                lines.append(
                    (
                        point_indexes[curve.start_point_id],
                        point_indexes[curve.end_point_id],
                    )
                )
            elif isinstance(curve, SketchCircle):
                if curved_samples < 4:
                    continue
                center = point_by_id[curve.center_point_id]
                circle = tuple(
                    sketch.plane.to_global(
                        center.u
                        + curve.radius
                        * math.cos(2.0 * math.pi * index / curved_samples),
                        center.v
                        + curve.radius
                        * math.sin(2.0 * math.pi * index / curved_samples),
                    )
                    for index in range(curved_samples)
                )
                start = len(preview_points)
                preview_points.extend(circle)
                lines.append(tuple(range(start, start + len(circle))) + (start,))
            elif isinstance(curve, SketchArc):
                if curved_samples < 2:
                    continue
                start_point = point_by_id[curve.start_point_id]
                center = point_by_id[curve.center_point_id]
                end_point = point_by_id[curve.end_point_id]
                start_angle = math.atan2(
                    start_point.v - center.v,
                    start_point.u - center.u,
                )
                end_angle = math.atan2(
                    end_point.v - center.v,
                    end_point.u - center.u,
                )
                sweep = end_angle - start_angle
                if curve.orientation == "ccw" and sweep <= 0.0:
                    sweep += 2.0 * math.pi
                elif curve.orientation == "cw" and sweep >= 0.0:
                    sweep -= 2.0 * math.pi
                radius = math.hypot(
                    start_point.u - center.u,
                    start_point.v - center.v,
                )
                arc = tuple(
                    sketch.plane.to_global(
                        center.u
                        + radius
                        * math.cos(start_angle + sweep * index / (curved_samples - 1)),
                        center.v
                        + radius
                        * math.sin(start_angle + sweep * index / (curved_samples - 1)),
                    )
                    for index in range(curved_samples)
                )
                start = len(preview_points)
                preview_points.extend(arc)
                lines.append(tuple(range(start, start + len(arc))))
        return _make_preview(
            2,
            tuple(preview_points),
            tuple(lines),
        )
    if type(recipe) is BoxGeometry:
        points = tuple(
            (x, y, z)
            for z in (0.0, recipe.height)
            for y in (0.0, recipe.depth)
            for x in (0.0, recipe.width)
        )
        lines = (
            (0, 1),
            (1, 3),
            (3, 2),
            (2, 0),
            (4, 5),
            (5, 7),
            (7, 6),
            (6, 4),
            (0, 4),
            (1, 5),
            (2, 6),
            (3, 7),
        )
        return _make_preview(3, points, lines)
    if type(recipe) is CylinderGeometry:
        bottom = _circle_points(0.0, 0.0, recipe.radius, 0.0)
        top = _circle_points(0.0, 0.0, recipe.radius, recipe.height)
        count = len(bottom)
        lines = (
            tuple(range(count)) + (0,),
            tuple(range(count, count * 2)) + (count,),
            (0, count),
            (count // 4, count + count // 4),
            (count // 2, count + count // 2),
            (3 * count // 4, count + 3 * count // 4),
        )
        return _make_preview(3, bottom + top, lines)
    raise TypeError("recipe is outside the A2 preview subset")


def _circle_points(
    center_x: float,
    center_y: float,
    radius: float,
    z: float,
) -> tuple[tuple[float, float, float], ...]:
    return tuple(
        (
            center_x + radius * math.cos(2.0 * math.pi * index / _PREVIEW_SEGMENTS),
            center_y + radius * math.sin(2.0 * math.pi * index / _PREVIEW_SEGMENTS),
            z,
        )
        for index in range(_PREVIEW_SEGMENTS)
    )


def _combine_previews(
    previews: tuple[StaticGeometryPreview, ...],
    dimension: int,
) -> StaticGeometryPreview:
    points: list[tuple[float, float, float]] = []
    lines: list[tuple[int, ...]] = []
    for preview in previews:
        if len(points) + len(preview.points) > _MAX_PREVIEW_POINTS:
            raise ValueError("combined preview exceeds the point budget")
        offset = len(points)
        points.extend(preview.points)
        lines.extend(tuple(index + offset for index in line) for line in preview.lines)
    return _make_preview(dimension, tuple(points), tuple(lines))


def _make_preview(
    dimension: int,
    points: tuple[tuple[float, float, float], ...],
    lines: tuple[tuple[int, ...], ...],
    *,
    point_names: tuple[str, ...] = (),
    member_names: tuple[str, ...] = (),
) -> StaticGeometryPreview:
    xs = tuple(point[0] for point in points)
    ys = tuple(point[1] for point in points)
    zs = tuple(point[2] for point in points)
    return StaticGeometryPreview(
        dimension=dimension,
        points=points,
        lines=lines,
        bounds=(min(xs), max(xs), min(ys), max(ys), min(zs), max(zs)),
        point_names=point_names,
        member_names=member_names,
    )


__all__ = [
    "StaticGeometryPreview",
]
