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
    _in_box,
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
    "in_box",
]
