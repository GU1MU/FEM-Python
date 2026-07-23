"""Selection predicates for OCC points."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal

from ..geometry import EntityRef, GeometryModel
from ._geometry import (
    _adjacent_to,
    _all,
    _by_coord,
    _in_box,
    _nearest_point,
)


def all(cad: GeometryModel) -> tuple[EntityRef, ...]:
    """Return all live OCC points."""
    return _all(cad, 0)


def by_coord(
    cad: GeometryModel,
    entities: Iterable[EntityRef] | None = None,
    *,
    x: float | None = None,
    y: float | None = None,
    z: float | None = None,
    tolerance: float = 1.0e-8,
) -> tuple[EntityRef, ...]:
    """Select points matching every supplied coordinate."""
    return _by_coord(
        cad,
        entities,
        dimension=0,
        operation="points.by_coord",
        x=x,
        y=y,
        z=z,
        tolerance=tolerance,
    )


def by_x(
    cad: GeometryModel,
    value: float,
    entities: Iterable[EntityRef] | None = None,
    *,
    tolerance: float = 1.0e-8,
) -> tuple[EntityRef, ...]:
    """Select points matching an x coordinate."""
    return by_coord(cad, entities, x=value, tolerance=tolerance)


def by_y(
    cad: GeometryModel,
    value: float,
    entities: Iterable[EntityRef] | None = None,
    *,
    tolerance: float = 1.0e-8,
) -> tuple[EntityRef, ...]:
    """Select points matching a y coordinate."""
    return by_coord(cad, entities, y=value, tolerance=tolerance)


def by_z(
    cad: GeometryModel,
    value: float,
    entities: Iterable[EntityRef] | None = None,
    *,
    tolerance: float = 1.0e-8,
) -> tuple[EntityRef, ...]:
    """Select points matching a z coordinate."""
    return by_coord(cad, entities, z=value, tolerance=tolerance)


def in_box(
    cad: GeometryModel,
    entities: Iterable[EntityRef] | None = None,
    *,
    xmin: float | None = None,
    xmax: float | None = None,
    ymin: float | None = None,
    ymax: float | None = None,
    zmin: float | None = None,
    zmax: float | None = None,
    tolerance: float = 1.0e-8,
) -> tuple[EntityRef, ...]:
    """Select points completely contained in a closed axis-aligned box."""
    return _in_box(
        cad,
        entities,
        dimension=0,
        operation="points.in_box",
        xmin=xmin,
        xmax=xmax,
        ymin=ymin,
        ymax=ymax,
        zmin=zmin,
        zmax=zmax,
        tolerance=tolerance,
    )


def nearest(
    cad: GeometryModel,
    x: float,
    y: float,
    z: float | None = None,
    *,
    entities: Iterable[EntityRef] | None = None,
) -> EntityRef | None:
    """Return the nearest point, choosing the first equal-distance candidate."""
    return _nearest_point(cad, x, y, z, entities=entities)


def adjacent_to(
    cad: GeometryModel,
    anchors: Iterable[EntityRef],
    entities: Iterable[EntityRef] | None = None,
    *,
    mode: Literal["any", "all"] = "any",
) -> tuple[EntityRef, ...]:
    """Select points immediately adjacent to any or all anchor curves."""
    return _adjacent_to(
        cad,
        anchors,
        entities,
        dimension=0,
        operation="points.adjacent_to",
        mode=mode,
    )


__all__ = [
    "adjacent_to",
    "all",
    "by_coord",
    "by_x",
    "by_y",
    "by_z",
    "in_box",
    "nearest",
]
