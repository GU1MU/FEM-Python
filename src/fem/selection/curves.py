"""Selection predicates for OCC curves."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal

from ..geometry import EntityRef, GeometryModel
from ._geometry import (
    _adjacent_to,
    _all,
    _by_center,
    _by_coord,
    _by_measure,
    _by_measure_range,
    _in_box,
    _intersects_box,
    _nearest_to,
    _within_distance,
)


def all(cad: GeometryModel) -> tuple[EntityRef, ...]:
    """Return all live OCC curves."""
    return _all(cad, 1)


def by_coord(
    cad: GeometryModel,
    entities: Iterable[EntityRef] | None = None,
    *,
    x: float | None = None,
    y: float | None = None,
    z: float | None = None,
    tolerance: float = 1.0e-8,
) -> tuple[EntityRef, ...]:
    """Select curves lying completely on every supplied coordinate plane."""
    return _by_coord(
        cad,
        entities,
        dimension=1,
        operation="curves.by_coord",
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
    """Select curves lying completely on an x-coordinate plane."""
    return by_coord(cad, entities, x=value, tolerance=tolerance)


def by_y(
    cad: GeometryModel,
    value: float,
    entities: Iterable[EntityRef] | None = None,
    *,
    tolerance: float = 1.0e-8,
) -> tuple[EntityRef, ...]:
    """Select curves lying completely on a y-coordinate plane."""
    return by_coord(cad, entities, y=value, tolerance=tolerance)


def by_z(
    cad: GeometryModel,
    value: float,
    entities: Iterable[EntityRef] | None = None,
    *,
    tolerance: float = 1.0e-8,
) -> tuple[EntityRef, ...]:
    """Select curves lying completely on a z-coordinate plane."""
    return by_coord(cad, entities, z=value, tolerance=tolerance)


def by_center(
    cad: GeometryModel,
    entities: Iterable[EntityRef] | None = None,
    *,
    x: float | None = None,
    y: float | None = None,
    z: float | None = None,
    tolerance: float = 1.0e-8,
) -> tuple[EntityRef, ...]:
    """Select curves whose centers of mass match supplied coordinates."""
    return _by_center(
        cad,
        entities,
        dimension=1,
        operation="curves.by_center",
        x=x,
        y=y,
        z=z,
        tolerance=tolerance,
    )


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
    """Select curves completely contained in a closed axis-aligned box."""
    return _in_box(
        cad,
        entities,
        dimension=1,
        operation="curves.in_box",
        xmin=xmin,
        xmax=xmax,
        ymin=ymin,
        ymax=ymax,
        zmin=zmin,
        zmax=zmax,
        tolerance=tolerance,
    )


def intersects_box(
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
    """Select curves whose bounding boxes overlap a closed query region.

    This is a conservative axis-aligned bounding-box test, not an exact
    geometric Boolean intersection. A curve can match when its bounding box
    overlaps even if its geometry does not enter the query region.
    """
    return _intersects_box(
        cad,
        entities,
        dimension=1,
        operation="curves.intersects_box",
        xmin=xmin,
        xmax=xmax,
        ymin=ymin,
        ymax=ymax,
        zmin=zmin,
        zmax=zmax,
        tolerance=tolerance,
    )


def by_length(
    cad: GeometryModel,
    entities: Iterable[EntityRef] | None = None,
    *,
    value: float,
    tolerance: float = 1.0e-8,
) -> tuple[EntityRef, ...]:
    """Select curves whose lengths match one non-negative value."""
    return _by_measure(
        cad,
        entities,
        dimension=1,
        operation="curves.by_length",
        query_name="length",
        value=value,
        tolerance=tolerance,
    )


def by_length_range(
    cad: GeometryModel,
    entities: Iterable[EntityRef] | None = None,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    tolerance: float = 1.0e-8,
) -> tuple[EntityRef, ...]:
    """Select curves in a closed, optionally one-sided length range."""
    return _by_measure_range(
        cad,
        entities,
        dimension=1,
        operation="curves.by_length_range",
        query_name="length",
        minimum=minimum,
        maximum=maximum,
        tolerance=tolerance,
    )


def nearest_to(
    cad: GeometryModel,
    anchor: EntityRef,
    entities: Iterable[EntityRef] | None = None,
) -> EntityRef | None:
    """Return the first curve nearest to one OCC entity.

    Distance is the unsigned minimum Euclidean distance between the entity
    sets. Touching, intersection, containment, and self-distance are zero;
    this is neither boundary-only nor Hausdorff distance.
    """
    return _nearest_to(
        cad,
        anchor,
        entities,
        dimension=1,
        operation="curves.nearest_to",
    )


def within_distance(
    cad: GeometryModel,
    anchor: EntityRef,
    entities: Iterable[EntityRef] | None = None,
    *,
    max_distance: float,
    tolerance: float = 1.0e-8,
) -> tuple[EntityRef, ...]:
    """Select curves within an OCC entity-set distance threshold.

    Distance is the unsigned minimum Euclidean distance between the entity
    sets. Touching, intersection, containment, and self-distance are zero;
    this is neither boundary-only nor Hausdorff distance.
    """
    return _within_distance(
        cad,
        anchor,
        entities,
        dimension=1,
        operation="curves.within_distance",
        max_distance=max_distance,
        tolerance=tolerance,
    )


def adjacent_to(
    cad: GeometryModel,
    anchors: Iterable[EntityRef],
    entities: Iterable[EntityRef] | None = None,
    *,
    mode: Literal["any", "all"] = "any",
) -> tuple[EntityRef, ...]:
    """Select curves immediately adjacent to any or all anchor entities."""
    return _adjacent_to(
        cad,
        anchors,
        entities,
        dimension=1,
        operation="curves.adjacent_to",
        mode=mode,
    )


__all__ = [
    "adjacent_to",
    "all",
    "by_center",
    "by_coord",
    "by_length",
    "by_length_range",
    "by_x",
    "by_y",
    "by_z",
    "in_box",
    "intersects_box",
    "nearest_to",
    "within_distance",
]
