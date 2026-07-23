"""Selection predicates for OCC volumes."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal

from ..geometry import EntityRef, GeometryModel
from ._geometry import (
    _adjacent_to,
    _all,
    _by_center,
    _by_measure,
    _by_measure_range,
    _in_box,
    _intersects_box,
    _nearest_to,
    _within_distance,
)


def all(cad: GeometryModel) -> tuple[EntityRef, ...]:
    """Return all live OCC volumes."""
    return _all(cad, 3)


def by_center(
    cad: GeometryModel,
    entities: Iterable[EntityRef] | None = None,
    *,
    x: float | None = None,
    y: float | None = None,
    z: float | None = None,
    tolerance: float = 1.0e-8,
) -> tuple[EntityRef, ...]:
    """Select volumes whose centers of mass match supplied coordinates."""
    return _by_center(
        cad,
        entities,
        dimension=3,
        operation="volumes.by_center",
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
    """Select volumes completely contained in a closed axis-aligned box."""
    return _in_box(
        cad,
        entities,
        dimension=3,
        operation="volumes.in_box",
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
    """Select volumes whose bounding boxes overlap a closed query region.

    This is a conservative axis-aligned bounding-box test, not an exact
    geometric Boolean intersection. A volume can match when its bounding box
    overlaps even if its geometry does not enter the query region.
    """
    return _intersects_box(
        cad,
        entities,
        dimension=3,
        operation="volumes.intersects_box",
        xmin=xmin,
        xmax=xmax,
        ymin=ymin,
        ymax=ymax,
        zmin=zmin,
        zmax=zmax,
        tolerance=tolerance,
    )


def by_volume(
    cad: GeometryModel,
    entities: Iterable[EntityRef] | None = None,
    *,
    value: float,
    tolerance: float = 1.0e-8,
) -> tuple[EntityRef, ...]:
    """Select volumes whose volumes match one non-negative value."""
    return _by_measure(
        cad,
        entities,
        dimension=3,
        operation="volumes.by_volume",
        query_name="volume",
        value=value,
        tolerance=tolerance,
    )


def by_volume_range(
    cad: GeometryModel,
    entities: Iterable[EntityRef] | None = None,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    tolerance: float = 1.0e-8,
) -> tuple[EntityRef, ...]:
    """Select volumes in a closed, optionally one-sided volume range."""
    return _by_measure_range(
        cad,
        entities,
        dimension=3,
        operation="volumes.by_volume_range",
        query_name="volume",
        minimum=minimum,
        maximum=maximum,
        tolerance=tolerance,
    )


def nearest_to(
    cad: GeometryModel,
    anchor: EntityRef,
    entities: Iterable[EntityRef] | None = None,
) -> EntityRef | None:
    """Return the first volume nearest to one OCC entity.

    Distance is the unsigned minimum Euclidean distance between the entity
    sets. Touching, intersection, containment, and self-distance are zero;
    this is neither boundary-only nor Hausdorff distance.
    """
    return _nearest_to(
        cad,
        anchor,
        entities,
        dimension=3,
        operation="volumes.nearest_to",
    )


def within_distance(
    cad: GeometryModel,
    anchor: EntityRef,
    entities: Iterable[EntityRef] | None = None,
    *,
    max_distance: float,
    tolerance: float = 1.0e-8,
) -> tuple[EntityRef, ...]:
    """Select volumes within an OCC entity-set distance threshold.

    Distance is the unsigned minimum Euclidean distance between the entity
    sets. Touching, intersection, containment, and self-distance are zero;
    this is neither boundary-only nor Hausdorff distance.
    """
    return _within_distance(
        cad,
        anchor,
        entities,
        dimension=3,
        operation="volumes.within_distance",
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
    """Select volumes immediately adjacent to any or all anchor surfaces."""
    return _adjacent_to(
        cad,
        anchors,
        entities,
        dimension=3,
        operation="volumes.adjacent_to",
        mode=mode,
    )


__all__ = [
    "adjacent_to",
    "all",
    "by_center",
    "by_volume",
    "by_volume_range",
    "in_box",
    "intersects_box",
    "nearest_to",
    "within_distance",
]
