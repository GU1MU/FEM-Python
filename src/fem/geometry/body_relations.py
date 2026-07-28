"""Conservative Body relation analysis used by mesh preflight."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

from .recipes import (
    BooleanGeometry,
    BoxGeometry,
    CylinderGeometry,
    DiskGeometry,
    ExtrudedGeometry,
    MovedGeometry,
    MultiBodyGeometry,
    PlateWithHoleGeometry,
    RectangleGeometry,
    RotatedGeometry,
    SketchGeometry,
)


BodyRelationKind = Literal["disjoint", "touching", "overlapping", "unknown"]


@dataclass(frozen=True, slots=True)
class BodyRelation:
    """One conservative relation between two stable Bodies."""

    first_body_id: str
    second_body_id: str
    relation: BodyRelationKind


class BodyOverlapError(ValueError):
    """Multi-Body geometry is unsafe for independent-domain meshing."""


def analyze_body_relations(
    recipe: MultiBodyGeometry,
    *,
    tolerance: float = 1.0e-9,
) -> tuple[BodyRelation, ...]:
    """Classify every Body pair from deterministic recipe bounds."""

    if type(recipe) is not MultiBodyGeometry:
        raise TypeError("recipe must be a MultiBodyGeometry")
    tolerance = float(tolerance)
    if tolerance < 0.0 or not math.isfinite(tolerance):
        raise ValueError("tolerance must be a finite non-negative number")
    bounds = {
        body.id: _recipe_bounds(body.recipe)
        for body in recipe.bodies
    }
    relations: list[BodyRelation] = []
    for index, first in enumerate(recipe.bodies):
        for second in recipe.bodies[index + 1:]:
            first_bounds = bounds[first.id]
            second_bounds = bounds[second.id]
            if first_bounds is None or second_bounds is None:
                relation: BodyRelationKind = "unknown"
            else:
                gaps = _bounds_gaps(first_bounds, second_bounds)
                if any(gap > tolerance for gap in gaps):
                    relation = "disjoint"
                else:
                    relation = (
                        _classify_known_solids(
                            first.recipe,
                            second.recipe,
                            tolerance,
                        )
                        or "unknown"
                    )
            relations.append(
                BodyRelation(first.id, second.id, relation)
            )
    return tuple(relations)


def require_meshable_body_relations(recipe: object) -> None:
    """Reject overlap, touching, and unknown multi-Body mesh inputs."""

    if not isinstance(recipe, MultiBodyGeometry):
        return
    blocked = tuple(
        relation
        for relation in analyze_body_relations(recipe)
        if relation.relation != "disjoint"
    )
    if not blocked:
        return
    details = ", ".join(
        f"{item.first_body_id}/{item.second_body_id}={item.relation}"
        for item in blocked
    )
    raise BodyOverlapError(
        "body.overlap.mesh-blocked: resolve Body relations with Boolean "
        f"before meshing ({details})"
    )


def _classify_known_solids(
    first: object,
    second: object,
    tolerance: float,
) -> BodyRelationKind | None:
    first_primitive = _translated_primitive(first)
    second_primitive = _translated_primitive(second)
    if first_primitive is None or second_primitive is None:
        return None
    first_kind, first_recipe, first_offset = first_primitive
    second_kind, second_recipe, second_offset = second_primitive
    first_bounds = _translated_primitive_bounds(
        first_kind,
        first_recipe,
        first_offset,
    )
    second_bounds = _translated_primitive_bounds(
        second_kind,
        second_recipe,
        second_offset,
    )
    if first_kind == "box" and second_kind == "box":
        return _classify_separation(
            _bounds_gaps(first_bounds, second_bounds),
            tolerance,
        )
    if first_kind == "cylinder" and second_kind == "cylinder":
        first_center = first_offset[:2]
        second_center = second_offset[:2]
        radial_gap = math.hypot(
            first_center[0] - second_center[0],
            first_center[1] - second_center[1],
        ) - (first_recipe.radius + second_recipe.radius)
        z_gap = max(
            first_bounds[2] - second_bounds[5],
            second_bounds[2] - first_bounds[5],
        )
        return _classify_separation((radial_gap, z_gap), tolerance)
    if {first_kind, second_kind} == {"box", "cylinder"}:
        if first_kind == "box":
            box_bounds = first_bounds
            cylinder = second_recipe
            cylinder_offset = second_offset
            cylinder_bounds = second_bounds
        else:
            box_bounds = second_bounds
            cylinder = first_recipe
            cylinder_offset = first_offset
            cylinder_bounds = first_bounds
        closest_x = min(
            max(cylinder_offset[0], box_bounds[0]),
            box_bounds[3],
        )
        closest_y = min(
            max(cylinder_offset[1], box_bounds[1]),
            box_bounds[4],
        )
        radial_gap = math.hypot(
            cylinder_offset[0] - closest_x,
            cylinder_offset[1] - closest_y,
        ) - cylinder.radius
        z_gap = max(
            box_bounds[2] - cylinder_bounds[5],
            cylinder_bounds[2] - box_bounds[5],
        )
        return _classify_separation((radial_gap, z_gap), tolerance)
    return None


def _translated_primitive(
    recipe: object,
) -> tuple[
    Literal["box", "cylinder"],
    BoxGeometry | CylinderGeometry,
    tuple[float, float, float],
] | None:
    dx = dy = dz = 0.0
    current = recipe
    while isinstance(current, MovedGeometry):
        dx += current.dx
        dy += current.dy
        dz += current.dz
        current = current.base
    if isinstance(current, BoxGeometry):
        return "box", current, (dx, dy, dz)
    if isinstance(current, CylinderGeometry):
        return "cylinder", current, (dx, dy, dz)
    return None


def _translated_primitive_bounds(
    kind: Literal["box", "cylinder"],
    recipe: BoxGeometry | CylinderGeometry,
    offset: tuple[float, float, float],
) -> tuple[float, float, float, float, float, float]:
    dx, dy, dz = offset
    if kind == "box":
        return (
            dx,
            dy,
            dz,
            dx + recipe.width,
            dy + recipe.depth,
            dz + recipe.height,
        )
    return (
        dx - recipe.radius,
        dy - recipe.radius,
        dz,
        dx + recipe.radius,
        dy + recipe.radius,
        dz + recipe.height,
    )


def _bounds_gaps(
    first: tuple[float, float, float, float, float, float],
    second: tuple[float, float, float, float, float, float],
) -> tuple[float, float, float]:
    return tuple(
        max(
            first[axis] - second[axis + 3],
            second[axis] - first[axis + 3],
        )
        for axis in range(3)
    )


def _classify_separation(
    gaps: tuple[float, ...],
    tolerance: float,
) -> BodyRelationKind:
    if any(gap > tolerance for gap in gaps):
        return "disjoint"
    if any(abs(gap) <= tolerance for gap in gaps):
        return "touching"
    return "overlapping"


def _recipe_bounds(
    recipe: object,
) -> tuple[float, float, float, float, float, float] | None:
    if isinstance(recipe, BoxGeometry):
        return (0.0, 0.0, 0.0, recipe.width, recipe.depth, recipe.height)
    if isinstance(recipe, CylinderGeometry):
        return (
            -recipe.radius,
            -recipe.radius,
            0.0,
            recipe.radius,
            recipe.radius,
            recipe.height,
        )
    if isinstance(recipe, RectangleGeometry):
        return (0.0, 0.0, 0.0, recipe.width, recipe.height, 0.0)
    if isinstance(recipe, (DiskGeometry, PlateWithHoleGeometry)):
        if isinstance(recipe, DiskGeometry):
            return (
                -recipe.radius,
                -recipe.radius,
                0.0,
                recipe.radius,
                recipe.radius,
                0.0,
            )
        return (0.0, 0.0, 0.0, recipe.width, recipe.height, 0.0)
    if isinstance(recipe, SketchGeometry):
        if recipe.is_strict:
            coordinates = tuple(
                (point.u, point.v, 0.0)
                for point in recipe.points
            )
        else:
            coordinates = tuple(
                coordinate
                for contour in recipe.contours
                for coordinate in _contour_bounds_points(contour)
            )
        return _point_bounds(coordinates)
    if isinstance(recipe, ExtrudedGeometry):
        base = _recipe_bounds(recipe.base)
        if base is None:
            return None
        return (
            base[0],
            base[1],
            base[2],
            base[3],
            base[4],
            base[5] + recipe.height,
        )
    if isinstance(recipe, MovedGeometry):
        base = _recipe_bounds(recipe.base)
        if base is None:
            return None
        return (
            base[0] + recipe.dx,
            base[1] + recipe.dy,
            base[2] + recipe.dz,
            base[3] + recipe.dx,
            base[4] + recipe.dy,
            base[5] + recipe.dz,
        )
    if isinstance(recipe, RotatedGeometry):
        base = _recipe_bounds(recipe.base)
        if base is None:
            return None
        angle = math.radians(recipe.angle_degrees)
        cosine, sine = math.cos(angle), math.sin(angle)

        def rotate(point: tuple[float, float, float]):
            x, y, z = point
            if recipe.axis == "x":
                return x, y * cosine - z * sine, y * sine + z * cosine
            if recipe.axis == "y":
                return x * cosine + z * sine, y, -x * sine + z * cosine
            return x * cosine - y * sine, x * sine + y * cosine, z

        return _point_bounds(tuple(rotate(point) for point in _corners(base)))
    if isinstance(recipe, BooleanGeometry):
        target = _recipe_bounds(recipe.object_geometry)
        tool = _recipe_bounds(recipe.tool_geometry)
        if recipe.operation == "cut":
            return target
        if target is None or tool is None:
            return None
        return (
            min(target[0], tool[0]),
            min(target[1], tool[1]),
            min(target[2], tool[2]),
            max(target[3], tool[3]),
            max(target[4], tool[4]),
            max(target[5], tool[5]),
        )
    return None


def _contour_bounds_points(contour: object) -> tuple[tuple[float, float, float], ...]:
    if hasattr(contour, "width") and hasattr(contour, "height"):
        x = float(getattr(contour, "x"))
        y = float(getattr(contour, "y"))
        width = float(getattr(contour, "width"))
        height = float(getattr(contour, "height"))
        return (
            (x, y, 0.0),
            (x + width, y + height, 0.0),
        )
    if hasattr(contour, "radius"):
        x = float(getattr(contour, "x"))
        y = float(getattr(contour, "y"))
        radius = float(getattr(contour, "radius"))
        return (
            (x - radius, y - radius, 0.0),
            (x + radius, y + radius, 0.0),
        )
    return ()


def _corners(
    bounds: tuple[float, float, float, float, float, float],
) -> tuple[tuple[float, float, float], ...]:
    return tuple(
        (x, y, z)
        for x in (bounds[0], bounds[3])
        for y in (bounds[1], bounds[4])
        for z in (bounds[2], bounds[5])
    )


def _point_bounds(
    points: tuple[tuple[float, float, float], ...],
) -> tuple[float, float, float, float, float, float] | None:
    if not points:
        return None
    return (
        min(point[0] for point in points),
        min(point[1] for point in points),
        min(point[2] for point in points),
        max(point[0] for point in points),
        max(point[1] for point in points),
        max(point[2] for point in points),
    )


__all__ = [
    "BodyOverlapError",
    "BodyRelation",
    "BodyRelationKind",
    "analyze_body_relations",
    "require_meshable_body_relations",
]
