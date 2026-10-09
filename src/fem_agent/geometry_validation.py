"""Geometry bounds, numeric validation, and strict sketch conversion."""

from __future__ import annotations

import math
from numbers import Real

from fem.geometry import (
    DiskGeometry,
    PlateWithHoleGeometry,
    RectangleGeometry,
    SketchCircle,
    SketchGeometry,
    SketchLine,
    SketchPlane,
    SketchPoint,
)


_MAX_PREVIEW_POINTS = 128


def _finite(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{field_name} must be a finite real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field_name} must be a finite real number")
    return result


def _as_strict_planar_sketch(recipe: object) -> SketchGeometry:
    if type(recipe) is SketchGeometry:
        if not recipe.is_strict:
            raise TypeError("Agent geometry requires a strict sketch")
        return recipe
    if type(recipe) in {RectangleGeometry, PlateWithHoleGeometry}:
        points = (
            SketchPoint("P1", 0.0, 0.0),
            SketchPoint("P2", recipe.width, 0.0),
            SketchPoint("P3", recipe.width, recipe.height),
            SketchPoint("P4", 0.0, recipe.height),
        )
        curves = (
            SketchLine("L1", "P1", "P2"),
            SketchLine("L2", "P2", "P3"),
            SketchLine("L3", "P3", "P4"),
            SketchLine("L4", "P4", "P1"),
        )
        if type(recipe) is PlateWithHoleGeometry:
            points += (SketchPoint("P5", recipe.hole_x, recipe.hole_y),)
            curves += (SketchCircle("C5", "P5", recipe.hole_radius),)
        return SketchGeometry(recipe.name, SketchPlane.xy(), points, curves)
    if type(recipe) is DiskGeometry:
        return SketchGeometry(
            recipe.name,
            SketchPlane.xy(),
            (SketchPoint("P1", 0.0, 0.0),),
            (SketchCircle("C1", "P1", recipe.radius),),
        )
    raise TypeError("incremental planar edits require a planar primitive or sketch")
