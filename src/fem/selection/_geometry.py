"""Shared implementation for public CAD-entity selectors."""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Literal

from ..geometry import EntityRef, GeometryModel


_Mode = Literal["any", "all"]


def _all(cad: GeometryModel, dimension: int) -> tuple[EntityRef, ...]:
    model = _require_model(cad)
    return model.entities(dimension)


def _by_coord(
    cad: GeometryModel,
    entities: Iterable[EntityRef] | None,
    *,
    dimension: int,
    operation: str,
    x: float | None,
    y: float | None,
    z: float | None,
    tolerance: float,
) -> tuple[EntityRef, ...]:
    model = _require_model(cad)
    coordinates, tolerance_value = _coordinates(
        operation,
        x=x,
        y=y,
        z=z,
        tolerance=tolerance,
    )
    candidates = _resolve_entities(
        model,
        entities,
        dimension=dimension,
        operation=operation,
    )
    if not candidates:
        return ()
    matches = frozenset(
        model.select(
            candidates,
            x=coordinates[0],
            y=coordinates[1],
            z=coordinates[2],
            tolerance=tolerance_value,
        )
    )
    return tuple(entity for entity in candidates if entity in matches)


def _by_center(
    cad: GeometryModel,
    entities: Iterable[EntityRef] | None,
    *,
    dimension: int,
    operation: str,
    x: float | None,
    y: float | None,
    z: float | None,
    tolerance: float,
) -> tuple[EntityRef, ...]:
    model = _require_model(cad)
    coordinates, tolerance_value = _coordinates(
        operation,
        x=x,
        y=y,
        z=z,
        tolerance=tolerance,
    )
    candidates = _resolve_entities(
        model,
        entities,
        dimension=dimension,
        operation=operation,
    )
    matches: list[EntityRef] = []
    for entity in candidates:
        center = model.center_of_mass(entity)
        if all(
            coordinate is None
            or abs(center[axis] - coordinate) <= tolerance_value
            for axis, coordinate in enumerate(coordinates)
        ):
            matches.append(entity)
    return tuple(matches)


def _in_box(
    cad: GeometryModel,
    entities: Iterable[EntityRef] | None,
    *,
    dimension: int,
    operation: str,
    xmin: float | None,
    xmax: float | None,
    ymin: float | None,
    ymax: float | None,
    zmin: float | None,
    zmax: float | None,
    tolerance: float,
) -> tuple[EntityRef, ...]:
    model = _require_model(cad)
    bounds, tolerance_value = _box_bounds(
        operation,
        xmin=xmin,
        xmax=xmax,
        ymin=ymin,
        ymax=ymax,
        zmin=zmin,
        zmax=zmax,
        tolerance=tolerance,
    )
    candidates = _resolve_entities(
        model,
        entities,
        dimension=dimension,
        operation=operation,
    )
    effective_tolerance = model.effective_bounding_box_tolerance(
        tolerance_value
    )
    matches: list[EntityRef] = []
    for entity in candidates:
        entity_bounds = model.bounding_box(entity)
        if all(
            lower is None
            or entity_bounds[axis] >= lower - effective_tolerance
            for axis, lower in enumerate(bounds[:3])
        ) and all(
            upper is None
            or entity_bounds[axis + 3] <= upper + effective_tolerance
            for axis, upper in enumerate(bounds[3:])
        ):
            matches.append(entity)
    return tuple(matches)


def _intersects_box(
    cad: GeometryModel,
    entities: Iterable[EntityRef] | None,
    *,
    dimension: int,
    operation: str,
    xmin: float | None,
    xmax: float | None,
    ymin: float | None,
    ymax: float | None,
    zmin: float | None,
    zmax: float | None,
    tolerance: float,
) -> tuple[EntityRef, ...]:
    model = _require_model(cad)
    bounds, tolerance_value = _box_bounds(
        operation,
        xmin=xmin,
        xmax=xmax,
        ymin=ymin,
        ymax=ymax,
        zmin=zmin,
        zmax=zmax,
        tolerance=tolerance,
    )
    candidates = _resolve_entities(
        model,
        entities,
        dimension=dimension,
        operation=operation,
    )
    effective_tolerance = model.effective_bounding_box_tolerance(
        tolerance_value
    )
    matches: list[EntityRef] = []
    for entity in candidates:
        entity_bounds = model.bounding_box(entity)
        if all(
            lower is None
            or entity_bounds[axis + 3] >= lower - effective_tolerance
            for axis, lower in enumerate(bounds[:3])
        ) and all(
            upper is None
            or entity_bounds[axis] <= upper + effective_tolerance
            for axis, upper in enumerate(bounds[3:])
        ):
            matches.append(entity)
    return tuple(matches)


def _by_measure(
    cad: GeometryModel,
    entities: Iterable[EntityRef] | None,
    *,
    dimension: int,
    operation: str,
    query_name: Literal["area", "length", "volume"],
    value: float,
    tolerance: float,
) -> tuple[EntityRef, ...]:
    model = _require_model(cad)
    target_value = _nonnegative_float(value, "value")
    tolerance_value = _nonnegative_float(tolerance, "tolerance")
    candidates = _resolve_entities(
        model,
        entities,
        dimension=dimension,
        operation=operation,
    )
    query = getattr(model, query_name)
    return tuple(
        entity
        for entity in candidates
        if abs(query(entity) - target_value) <= tolerance_value
    )


def _by_measure_range(
    cad: GeometryModel,
    entities: Iterable[EntityRef] | None,
    *,
    dimension: int,
    operation: str,
    query_name: Literal["area", "length", "volume"],
    minimum: float | None,
    maximum: float | None,
    tolerance: float,
) -> tuple[EntityRef, ...]:
    model = _require_model(cad)
    lower, upper, tolerance_value = _measure_range(
        operation,
        minimum=minimum,
        maximum=maximum,
        tolerance=tolerance,
    )
    candidates = _resolve_entities(
        model,
        entities,
        dimension=dimension,
        operation=operation,
    )
    query = getattr(model, query_name)
    matches: list[EntityRef] = []
    for entity in candidates:
        measure = query(entity)
        if (
            lower is None or measure >= lower - tolerance_value
        ) and (
            upper is None or measure <= upper + tolerance_value
        ):
            matches.append(entity)
    return tuple(matches)


def _nearest_point(
    cad: GeometryModel,
    x: float,
    y: float,
    z: float | None,
    *,
    entities: Iterable[EntityRef] | None,
) -> EntityRef | None:
    model = _require_model(cad)
    target = (
        _finite_float(x, "x"),
        _finite_float(y, "y"),
        None if z is None else _finite_float(z, "z"),
    )
    candidates = _resolve_entities(
        model,
        entities,
        dimension=0,
        operation="points.nearest",
    )
    best: EntityRef | None = None
    best_distance: float | None = None
    for entity in candidates:
        center = model.center_of_mass(entity)
        deltas = (
            center[0] - target[0],
            center[1] - target[1],
            *(() if target[2] is None else (center[2] - target[2],)),
        )
        distance = math.hypot(*deltas)
        if best_distance is None or distance < best_distance:
            best = entity
            best_distance = distance
    return best


def _nearest_to(
    cad: GeometryModel,
    anchor: EntityRef,
    entities: Iterable[EntityRef] | None,
    *,
    dimension: int,
    operation: str,
) -> EntityRef | None:
    model = _require_model(cad)
    target = _distance_anchor(
        anchor,
        operation=operation,
    )
    candidates = _resolve_entities(
        model,
        entities,
        dimension=dimension,
        operation=operation,
    )
    distances = model.distances_to(target, candidates)
    best: EntityRef | None = None
    best_distance: float | None = None
    for candidate, distance in zip(candidates, distances, strict=True):
        if best_distance is None or distance < best_distance:
            best = candidate
            best_distance = distance
    return best


def _within_distance(
    cad: GeometryModel,
    anchor: EntityRef,
    entities: Iterable[EntityRef] | None,
    *,
    dimension: int,
    operation: str,
    max_distance: float,
    tolerance: float,
) -> tuple[EntityRef, ...]:
    model = _require_model(cad)
    target = _distance_anchor(
        anchor,
        operation=operation,
    )
    maximum = _nonnegative_float(max_distance, "max_distance")
    tolerance_value = _nonnegative_float(tolerance, "tolerance")
    candidates = _resolve_entities(
        model,
        entities,
        dimension=dimension,
        operation=operation,
    )
    distances = model.distances_to(target, candidates)
    return tuple(
        candidate
        for candidate, distance in zip(candidates, distances, strict=True)
        if distance <= maximum + tolerance_value
    )


def _adjacent_to(
    cad: GeometryModel,
    anchors: Iterable[EntityRef],
    entities: Iterable[EntityRef] | None,
    *,
    dimension: int,
    operation: str,
    mode: str,
) -> tuple[EntityRef, ...]:
    model = _require_model(cad)
    normalized_mode = _mode(mode)
    explicit_candidates = entities is not None
    candidates = _resolve_entities(
        model,
        entities,
        dimension=dimension,
        operation=operation,
    )
    normalized_anchors = _anchors(
        anchors,
        dimension=dimension,
        operation=operation,
    )

    if explicit_candidates and candidates:
        if dimension > model.dimension:
            # No valid target-dimension entity can belong to this model. Retain
            # the established per-reference exception for explicit candidates.
            for candidate in candidates:
                model.bounding_box(candidate)
        else:
            live_entities = frozenset(model.entities(dimension))
            # Snapshot membership validates current local identities without a
            # numerical query. The public fallback preserves precise ownership
            # and stale-reference errors for anything absent from the snapshot.
            for candidate in candidates:
                if candidate not in live_entities:
                    model.bounding_box(candidate)

    adjacency_sets = tuple(
        frozenset(model.adjacent(anchor, dimension=dimension))
        for anchor in normalized_anchors
    )
    if normalized_mode == "any":
        adjacent = frozenset().union(*adjacency_sets)
    else:
        adjacent = adjacency_sets[0].intersection(*adjacency_sets[1:])
    return tuple(candidate for candidate in candidates if candidate in adjacent)


def _require_model(cad: object) -> GeometryModel:
    if not isinstance(cad, GeometryModel):
        raise TypeError(f"cad must be a GeometryModel, got {cad!r}")
    return cad


def _resolve_entities(
    cad: GeometryModel,
    entities: Iterable[EntityRef] | None,
    *,
    dimension: int,
    operation: str,
) -> tuple[EntityRef, ...]:
    if entities is None:
        return cad.entities(dimension)
    try:
        candidates = tuple(entities)
    except TypeError as exc:
        raise TypeError(f"{operation} entities must be iterable") from exc
    if not candidates:
        # Probe a universally valid entity dimension so an explicit empty
        # candidate set still requires a live, queryable geometry model.
        cad.entities(0)
        return ()

    seen: set[EntityRef] = set()
    unique: list[EntityRef] = []
    for entity in candidates:
        if not isinstance(entity, EntityRef):
            raise TypeError(
                f"{operation} entities must contain only EntityRef values"
            )
        if entity.dimension != dimension:
            raise ValueError(
                f"{operation} entities must have dimension {dimension}"
            )
        if entity not in seen:
            seen.add(entity)
            unique.append(entity)
    return tuple(unique)


def _distance_anchor(
    anchor: EntityRef,
    *,
    operation: str,
) -> EntityRef:
    if not isinstance(anchor, EntityRef):
        raise TypeError(f"{operation} anchor must be an EntityRef")
    return anchor


def _anchors(
    anchors: Iterable[EntityRef],
    *,
    dimension: int,
    operation: str,
) -> tuple[EntityRef, ...]:
    try:
        normalized = tuple(anchors)
    except TypeError as exc:
        raise TypeError(f"{operation} anchors must be iterable") from exc
    if not normalized:
        raise ValueError(f"{operation} requires at least one anchor")

    anchor_dimension: int | None = None
    seen: set[EntityRef] = set()
    unique: list[EntityRef] = []
    for anchor in normalized:
        if not isinstance(anchor, EntityRef):
            raise TypeError(
                f"{operation} anchors must contain only EntityRef values"
            )
        if anchor_dimension is None:
            anchor_dimension = anchor.dimension
        elif anchor.dimension != anchor_dimension:
            raise ValueError(
                f"{operation} anchors must have one common dimension"
            )
        if anchor not in seen:
            seen.add(anchor)
            unique.append(anchor)
    if anchor_dimension is None or abs(anchor_dimension - dimension) != 1:
        raise ValueError(
            f"{operation} anchor dimension must differ from target dimension by one"
        )
    return tuple(unique)


def _coordinates(
    operation: str,
    *,
    x: float | None,
    y: float | None,
    z: float | None,
    tolerance: float,
) -> tuple[tuple[float | None, float | None, float | None], float]:
    raw = (x, y, z)
    if all(value is None for value in raw):
        raise ValueError(f"{operation} requires at least one coordinate")
    coordinates = tuple(
        None if value is None else _finite_float(value, axis)
        for axis, value in zip(("x", "y", "z"), raw, strict=True)
    )
    return (
        (coordinates[0], coordinates[1], coordinates[2]),
        _nonnegative_float(tolerance, "tolerance"),
    )


def _box_bounds(
    operation: str,
    *,
    xmin: float | None,
    xmax: float | None,
    ymin: float | None,
    ymax: float | None,
    zmin: float | None,
    zmax: float | None,
    tolerance: float,
) -> tuple[
    tuple[
        float | None,
        float | None,
        float | None,
        float | None,
        float | None,
        float | None,
    ],
    float,
]:
    raw = (xmin, ymin, zmin, xmax, ymax, zmax)
    if all(value is None for value in raw):
        raise ValueError(f"{operation} requires at least one box bound")
    labels = ("xmin", "ymin", "zmin", "xmax", "ymax", "zmax")
    normalized = tuple(
        None if value is None else _finite_float(value, label)
        for label, value in zip(labels, raw, strict=True)
    )
    for axis, label in enumerate(("x", "y", "z")):
        lower = normalized[axis]
        upper = normalized[axis + 3]
        if lower is not None and upper is not None and lower > upper:
            raise ValueError(f"{label}min must not exceed {label}max")
    return (
        (
            normalized[0],
            normalized[1],
            normalized[2],
            normalized[3],
            normalized[4],
            normalized[5],
        ),
        _nonnegative_float(tolerance, "tolerance"),
    )


def _measure_range(
    operation: str,
    *,
    minimum: float | None,
    maximum: float | None,
    tolerance: float,
) -> tuple[float | None, float | None, float]:
    if minimum is None and maximum is None:
        raise ValueError(
            f"{operation} requires at least one of minimum and maximum"
        )
    lower = (
        None
        if minimum is None
        else _nonnegative_float(minimum, "minimum")
    )
    upper = (
        None
        if maximum is None
        else _nonnegative_float(maximum, "maximum")
    )
    if lower is not None and upper is not None and lower > upper:
        raise ValueError("minimum must not exceed maximum")
    return lower, upper, _nonnegative_float(tolerance, "tolerance")


def _mode(mode: str) -> _Mode:
    if mode not in ("any", "all"):
        raise ValueError("mode must be 'any' or 'all'")
    return mode


def _finite_float(value: object, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be finite, got {value!r}")
    try:
        normalized = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be finite, got {value!r}") from exc
    if not math.isfinite(normalized):
        raise ValueError(f"{label} must be finite, got {value!r}")
    return normalized


def _nonnegative_float(value: object, label: str) -> float:
    normalized = _finite_float(value, label)
    if normalized < 0.0:
        raise ValueError(f"{label} must be finite and >= 0, got {value!r}")
    return normalized


__all__: list[str] = []
