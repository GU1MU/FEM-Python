"""Neutral scalar-field topology shared by GUI and headless serializers."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from numbers import Real
from typing import TypeVar

import numpy as np

from fem.post.fields import ResultRegionKey

from .data import (
    FieldData,
    FieldLocation,
    ResultExportSnapshot,
    ResultTopologyProjection,
)
from .fields import (
    FieldAssociation,
    ResultSourceKey,
    ScalarFieldSelection,
)


class ResultValueLayout(str, Enum):
    """Placement of the selected scalar values in a neutral topology."""

    POINT = "point"
    CELL = "cell"


class ResultCellKind(str, Enum):
    """Meaning of one neutral cell without a VTK numeric type."""

    FEM_ELEMENT = "fem_element"
    SAMPLE_VERTEX = "sample_vertex"


@dataclass(frozen=True, slots=True, eq=False, init=False)
class ResultFieldTopology:
    """Owned, immutable render topology for one exact scalar selection."""

    source: ResultSourceKey
    materialization_generation: int
    selection: ScalarFieldSelection
    deformation_scale: float
    _points: np.ndarray = field(repr=False)
    cells: tuple[tuple[int, ...], ...]
    cell_kinds: tuple[ResultCellKind, ...]
    canonical_element_types: tuple[str | None, ...]
    _values: np.ndarray = field(repr=False)
    value_layout: ResultValueLayout
    point_locations: tuple[FieldLocation | None, ...]
    cell_locations: tuple[FieldLocation | None, ...]

    def __init__(
        self,
        *,
        source: ResultSourceKey,
        materialization_generation: int,
        selection: ScalarFieldSelection,
        deformation_scale: float,
        points: np.ndarray,
        cells: tuple[tuple[int, ...], ...],
        cell_kinds: tuple[ResultCellKind, ...],
        canonical_element_types: tuple[str | None, ...],
        values: np.ndarray,
        value_layout: ResultValueLayout,
        point_locations: tuple[FieldLocation | None, ...],
        cell_locations: tuple[FieldLocation | None, ...],
    ) -> None:
        if type(source) is not ResultSourceKey:
            raise TypeError("source must be ResultSourceKey")
        if type(materialization_generation) is not int:
            raise TypeError("materialization_generation must be an integer")
        if materialization_generation < 0:
            raise ValueError("materialization_generation must be non-negative")
        if type(selection) is not ScalarFieldSelection:
            raise TypeError("selection must be ScalarFieldSelection")
        checked_scale = _finite_scale(deformation_scale)
        owned_points = _owned_points(points)
        checked_cells = _projected_cells(cells, point_count=len(owned_points))
        checked_kinds = _cell_kinds(cell_kinds, len(checked_cells))
        _validate_cell_shapes(checked_cells, checked_kinds)
        checked_types = _canonical_element_types(
            canonical_element_types,
            checked_kinds,
        )
        if type(value_layout) is not ResultValueLayout:
            raise TypeError("value_layout must be ResultValueLayout")
        expected_values = (
            len(owned_points)
            if value_layout is ResultValueLayout.POINT
            else len(checked_cells)
        )
        owned_values = _owned_values(values, expected_values)
        checked_point_locations = _location_tuple(
            point_locations,
            length=len(owned_points),
            label="point_locations",
        )
        checked_cell_locations = _location_tuple(
            cell_locations,
            length=len(checked_cells),
            label="cell_locations",
        )

        object.__setattr__(self, "source", source)
        object.__setattr__(
            self,
            "materialization_generation",
            materialization_generation,
        )
        object.__setattr__(self, "selection", selection)
        object.__setattr__(self, "deformation_scale", checked_scale)
        object.__setattr__(self, "_points", owned_points)
        object.__setattr__(self, "cells", checked_cells)
        object.__setattr__(self, "cell_kinds", checked_kinds)
        object.__setattr__(
            self,
            "canonical_element_types",
            checked_types,
        )
        object.__setattr__(self, "_values", owned_values)
        object.__setattr__(self, "value_layout", value_layout)
        object.__setattr__(
            self,
            "point_locations",
            checked_point_locations,
        )
        object.__setattr__(
            self,
            "cell_locations",
            checked_cell_locations,
        )

    @property
    def points(self) -> np.ndarray:
        """Return a detached readonly copy of projected point coordinates."""

        return _public_array_copy(self._points)

    @property
    def values(self) -> np.ndarray:
        """Return a detached readonly copy of selected scalar values."""

        return _public_array_copy(self._values)


def project_scalar_field_topology(
    export: ResultExportSnapshot,
    deformation_scale: float = 0.0,
) -> ResultFieldTopology:
    """Project one accepted scalar field into its sole neutral render layout."""

    if type(export) is not ResultExportSnapshot:
        raise TypeError("export must be ResultExportSnapshot")
    scale = _finite_scale(deformation_scale)
    _validate_export_snapshot(export)
    field_data = export.field
    component_index = field_data.descriptor.columns.index(export.selection.component)
    scalar_values = field_data.values[:, component_index]
    association = field_data.descriptor.association

    if association is FieldAssociation.NODE:
        projected = _project_node(
            export.topology,
            field_data,
            scalar_values,
            scale,
        )
    elif association is FieldAssociation.ELEMENT:
        projected = _project_element(
            export.topology,
            field_data,
            scalar_values,
            scale,
        )
    elif association is FieldAssociation.INTEGRATION_POINT:
        projected = _project_integration_points(
            export.topology,
            field_data,
            scalar_values,
            scale,
        )
    elif association is FieldAssociation.ELEMENT_NODE:
        projected = _project_element_nodes(
            export.topology,
            field_data,
            scalar_values,
            scale,
        )
    elif association is FieldAssociation.NODE_REGION:
        projected = _project_node_regions(
            export.topology,
            field_data,
            scalar_values,
            scale,
        )
    elif association is FieldAssociation.RESOLVED_NODAL:
        projected = _project_resolved_nodes(
            export.topology,
            field_data,
            scalar_values,
            scale,
        )
    else:
        raise ValueError(f"unsupported field association {association.value}")

    return ResultFieldTopology(
        source=export.source,
        materialization_generation=export.materialization_generation,
        selection=export.selection,
        deformation_scale=scale,
        **projected,
    )


def _validate_export_snapshot(export: ResultExportSnapshot) -> None:
    if export.topology.source != export.source:
        raise ValueError("topology source must match export source")
    if export.field.source != export.source:
        raise ValueError("field source must match export source")
    if export.selection.field_key != export.field.key:
        raise ValueError("selection field key must match export field key")
    if export.selection.component not in export.field.descriptor.columns:
        raise ValueError("selection component is not in the field descriptor")


def _project_node(
    topology: ResultTopologyProjection,
    field_data: FieldData,
    scalar_values: np.ndarray,
    scale: float,
) -> dict[str, object]:
    rows = _unique_row_map(
        field_data,
        key=lambda location: location.node_id,
        label="node",
    )
    used: set[int] = set()
    locations: list[FieldLocation] = []
    values: list[float] = []
    coordinates = topology.node_coordinates
    displacements = topology.nodal_displacements
    for index, node_id in enumerate(topology.node_ids):
        row_index = _required_row(rows, node_id, label=f"node {node_id}")
        location = field_data.locations[row_index]
        _validate_nodal_sample(
            location,
            node_id=node_id,
            coordinates=coordinates[index],
            displacement=displacements[index],
        )
        used.add(row_index)
        locations.append(location)
        values.append(float(scalar_values[row_index]))
    _require_complete_row_use(field_data, used)
    return _fem_projection(
        topology,
        points=_deformed_points(coordinates, displacements, scale),
        values=np.asarray(values, dtype=float),
        value_layout=ResultValueLayout.POINT,
        point_locations=tuple(locations),
        cell_locations=(None,) * len(topology.element_ids),
    )


def _project_element(
    topology: ResultTopologyProjection,
    field_data: FieldData,
    scalar_values: np.ndarray,
    scale: float,
) -> dict[str, object]:
    rows = _unique_row_map(
        field_data,
        key=lambda location: location.element_id,
        label="element",
    )
    used: set[int] = set()
    locations: list[FieldLocation] = []
    values: list[float] = []
    for element_id in topology.element_ids:
        row_index = _required_row(
            rows,
            element_id,
            label=f"element {element_id}",
        )
        used.add(row_index)
        locations.append(field_data.locations[row_index])
        values.append(float(scalar_values[row_index]))
    _require_complete_row_use(field_data, used)
    return _fem_projection(
        topology,
        points=_deformed_points(
            topology.node_coordinates,
            topology.nodal_displacements,
            scale,
        ),
        values=np.asarray(values, dtype=float),
        value_layout=ResultValueLayout.CELL,
        point_locations=(None,) * len(topology.node_ids),
        cell_locations=tuple(locations),
    )


def _project_integration_points(
    topology: ResultTopologyProjection,
    field_data: FieldData,
    scalar_values: np.ndarray,
    scale: float,
) -> dict[str, object]:
    known_elements = frozenset(topology.element_ids)
    points: list[tuple[float, float, float]] = []
    for location in field_data.locations:
        if location.element_id not in known_elements:
            raise ValueError("integration-point row references an unknown element")
        points.append(_deformed_location(location, scale))
    locations = field_data.locations
    count = len(locations)
    return {
        "points": np.asarray(points, dtype=float).reshape((count, 3)),
        "cells": tuple((index,) for index in range(count)),
        "cell_kinds": (ResultCellKind.SAMPLE_VERTEX,) * count,
        "canonical_element_types": (None,) * count,
        "values": scalar_values,
        "value_layout": ResultValueLayout.POINT,
        "point_locations": locations,
        "cell_locations": locations,
    }


def _project_element_nodes(
    topology: ResultTopologyProjection,
    field_data: FieldData,
    scalar_values: np.ndarray,
    scale: float,
) -> dict[str, object]:
    rows = _unique_row_map(
        field_data,
        key=_element_node_key,
        label="element-node",
    )
    return _project_element_local_rows(
        topology,
        field_data,
        scalar_values,
        scale,
        row_for=lambda element_id, local_node, node_id, _region: (
            _required_row(
                rows,
                (element_id, local_node, node_id),
                label=(
                    f"element {element_id} local node {local_node} (node {node_id})"
                ),
            )
        ),
    )


def _project_node_regions(
    topology: ResultTopologyProjection,
    field_data: FieldData,
    scalar_values: np.ndarray,
    scale: float,
) -> dict[str, object]:
    rows = _unique_row_map(
        field_data,
        key=lambda location: (location.node_id, location.region_key),
        label="node-region",
    )
    projected_by_region: dict[object, int] = {}

    def row_for(
        element_id: int,
        local_node: int,
        node_id: int,
        region_key: ResultRegionKey,
    ) -> int:
        del element_id, local_node
        return _required_row(
            rows,
            (node_id, region_key),
            label=f"node {node_id} in its element region",
        )

    return _project_element_local_rows(
        topology,
        field_data,
        scalar_values,
        scale,
        row_for=row_for,
        reusable_point_key=lambda row_index: (
            field_data.locations[row_index].node_id,
            field_data.locations[row_index].region_key,
        ),
        projected_by_key=projected_by_region,
    )


def _project_resolved_nodes(
    topology: ResultTopologyProjection,
    field_data: FieldData,
    scalar_values: np.ndarray,
    scale: float,
) -> dict[str, object]:
    averaged_rows: dict[tuple[int, ResultRegionKey], int] = {}
    raw_rows: dict[tuple[int, int, int, ResultRegionKey], int] = {}
    for row_index, location in enumerate(field_data.locations):
        if location.averaged is True:
            key = _checked_region_pair(location)
            _insert_unique(
                averaged_rows,
                key,
                row_index,
                label="resolved averaged",
            )
        elif location.averaged is False:
            key = (
                _required_identity(location.element_id, "element_id"),
                _required_identity(location.local_node, "local_node"),
                _required_identity(location.node_id, "node_id"),
                _required_region(location.region_key),
            )
            _insert_unique(
                raw_rows,
                key,
                row_index,
                label="resolved raw",
            )
        else:
            raise ValueError("resolved-nodal rows require an averaged state")
    projected_averaged: dict[object, int] = {}

    def row_for(
        element_id: int,
        local_node: int,
        node_id: int,
        region_key: ResultRegionKey,
    ) -> int:
        averaged_key = (node_id, region_key)
        averaged = averaged_rows.get(averaged_key)
        if averaged is not None:
            return averaged
        return _required_row(
            raw_rows,
            (element_id, local_node, node_id, region_key),
            label=(
                f"resolved node {node_id}, element {element_id}, "
                f"local node {local_node}, and region"
            ),
        )

    def reusable_key(
        row_index: int,
    ) -> tuple[int, ResultRegionKey] | None:
        location = field_data.locations[row_index]
        if location.averaged is not True:
            return None
        return _checked_region_pair(location)

    return _project_element_local_rows(
        topology,
        field_data,
        scalar_values,
        scale,
        row_for=row_for,
        reusable_point_key=reusable_key,
        projected_by_key=projected_averaged,
    )


_RowKey = TypeVar("_RowKey")


def _project_element_local_rows(
    topology: ResultTopologyProjection,
    field_data: FieldData,
    scalar_values: np.ndarray,
    scale: float,
    *,
    row_for: object,
    reusable_point_key: object | None = None,
    projected_by_key: dict[object, int] | None = None,
) -> dict[str, object]:
    if not callable(row_for):
        raise TypeError("row_for must be callable")
    if reusable_point_key is not None and not callable(reusable_point_key):
        raise TypeError("reusable_point_key must be callable or None")
    point_cache = {} if projected_by_key is None else projected_by_key
    node_order = {node_id: index for index, node_id in enumerate(topology.node_ids)}
    coordinates = topology.node_coordinates
    displacements = topology.nodal_displacements
    points: list[tuple[float, float, float]] = []
    values: list[float] = []
    locations: list[FieldLocation] = []
    cells: list[tuple[int, ...]] = []
    used: set[int] = set()

    for element_id, connected, region_key in zip(
        topology.element_ids,
        topology.connectivity,
        topology.element_region_keys,
        strict=True,
    ):
        projected_cell: list[int] = []
        for local_node, node_id in enumerate(connected, start=1):
            row_index = row_for(
                element_id,
                local_node,
                node_id,
                region_key,
            )
            if type(row_index) is not int:
                raise TypeError("row projector must return an integer index")
            if row_index < 0 or row_index >= len(field_data.locations):
                raise ValueError("row projector returned an invalid index")
            location = field_data.locations[row_index]
            topology_node_index = node_order[node_id]
            _validate_nodal_sample(
                location,
                node_id=node_id,
                coordinates=coordinates[topology_node_index],
                displacement=displacements[topology_node_index],
            )
            used.add(row_index)
            cache_key = (
                None if reusable_point_key is None else reusable_point_key(row_index)
            )
            point_index = None if cache_key is None else point_cache.get(cache_key)
            if point_index is None:
                point_index = len(points)
                points.append(_deformed_location(location, scale))
                values.append(float(scalar_values[row_index]))
                locations.append(location)
                if cache_key is not None:
                    point_cache[cache_key] = point_index
            projected_cell.append(point_index)
        cells.append(tuple(projected_cell))

    _require_complete_row_use(field_data, used)
    return {
        "points": np.asarray(points, dtype=float).reshape((len(points), 3)),
        "cells": tuple(cells),
        "cell_kinds": (ResultCellKind.FEM_ELEMENT,) * len(cells),
        "canonical_element_types": topology.element_types,
        "values": np.asarray(values, dtype=float),
        "value_layout": ResultValueLayout.POINT,
        "point_locations": tuple(locations),
        "cell_locations": (None,) * len(cells),
    }


def _fem_projection(
    topology: ResultTopologyProjection,
    *,
    points: np.ndarray,
    values: np.ndarray,
    value_layout: ResultValueLayout,
    point_locations: tuple[FieldLocation | None, ...],
    cell_locations: tuple[FieldLocation | None, ...],
) -> dict[str, object]:
    node_order = {node_id: index for index, node_id in enumerate(topology.node_ids)}
    cells = tuple(
        tuple(node_order[node_id] for node_id in connected)
        for connected in topology.connectivity
    )
    return {
        "points": points,
        "cells": cells,
        "cell_kinds": (ResultCellKind.FEM_ELEMENT,) * len(cells),
        "canonical_element_types": topology.element_types,
        "values": values,
        "value_layout": value_layout,
        "point_locations": point_locations,
        "cell_locations": cell_locations,
    }


def _unique_row_map(
    field_data: FieldData,
    *,
    key: object,
    label: str,
) -> dict[object, int]:
    if not callable(key):
        raise TypeError("key must be callable")
    rows: dict[object, int] = {}
    for row_index, location in enumerate(field_data.locations):
        row_key = key(location)
        _insert_unique(rows, row_key, row_index, label=label)
    return rows


def _insert_unique(
    rows: dict[_RowKey, int],
    key: _RowKey,
    row_index: int,
    *,
    label: str,
) -> None:
    if key in rows:
        raise ValueError(f"{label} rows contain a duplicate identity")
    rows[key] = row_index


def _required_row(
    rows: dict[_RowKey, int],
    key: _RowKey,
    *,
    label: str,
) -> int:
    try:
        return rows[key]
    except KeyError as error:
        raise ValueError(f"missing exact field row for {label}") from error


def _require_complete_row_use(
    field_data: FieldData,
    used: set[int],
) -> None:
    if len(used) != len(field_data.locations):
        unused = tuple(
            index for index in range(len(field_data.locations)) if index not in used
        )
        raise ValueError(
            f"field contains rows outside the projected topology at ordinals {unused}"
        )


def _element_node_key(
    location: FieldLocation,
) -> tuple[int, int, int]:
    return (
        _required_identity(location.element_id, "element_id"),
        _required_identity(location.local_node, "local_node"),
        _required_identity(location.node_id, "node_id"),
    )


def _checked_region_pair(
    location: FieldLocation,
) -> tuple[int, ResultRegionKey]:
    return (
        _required_identity(location.node_id, "node_id"),
        _required_region(location.region_key),
    )


def _required_identity(value: int | None, label: str) -> int:
    if type(value) is not int:
        raise ValueError(f"field location requires {label}")
    return value


def _required_region(value: ResultRegionKey | None) -> ResultRegionKey:
    if type(value) is not ResultRegionKey:
        raise ValueError("field location requires an exact region key")
    return value


def _validate_nodal_sample(
    location: FieldLocation,
    *,
    node_id: int,
    coordinates: np.ndarray,
    displacement: np.ndarray,
) -> None:
    if location.node_id != node_id:
        raise ValueError("field row node identity does not match topology")
    if tuple(float(value) for value in coordinates) != location.coordinates:
        raise ValueError("field row coordinates do not match topology node")
    if location.displacement is None:
        raise ValueError("deformable nodal field rows require sample displacement")
    if tuple(float(value) for value in displacement) != location.displacement:
        raise ValueError("field row displacement does not match topology node")


def _deformed_location(
    location: FieldLocation,
    scale: float,
) -> tuple[float, float, float]:
    if location.displacement is None:
        raise ValueError("field sample does not provide deformation displacement")
    point = _deformed_points(
        np.asarray((location.coordinates,), dtype=float),
        np.asarray((location.displacement,), dtype=float),
        scale,
    )
    return tuple(float(value) for value in point[0])


def _deformed_points(
    coordinates: np.ndarray,
    displacements: np.ndarray,
    scale: float,
) -> np.ndarray:
    with np.errstate(over="ignore", invalid="ignore"):
        points = np.asarray(coordinates, dtype=float) + (
            scale * np.asarray(displacements, dtype=float)
        )
    if not bool(np.isfinite(points).all()):
        raise ValueError("deformed point coordinates must be finite")
    return points


def _finite_scale(value: object) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError("deformation_scale must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("deformation_scale must be finite")
    return result


def _owned_points(value: object) -> np.ndarray:
    if type(value) is not np.ndarray:
        raise TypeError("points must be a numpy.ndarray")
    if value.ndim != 2 or value.shape[1:] != (3,):
        raise ValueError("points must have shape (point_count, 3)")
    return _owned_finite_array(value, label="points")


def _owned_values(value: object, length: int) -> np.ndarray:
    if type(value) is not np.ndarray:
        raise TypeError("values must be a numpy.ndarray")
    if value.ndim != 1 or value.shape != (length,):
        raise ValueError(f"values must have shape ({length},)")
    return _owned_finite_array(value, label="values")


def _owned_finite_array(value: np.ndarray, *, label: str) -> np.ndarray:
    if np.issubdtype(value.dtype, np.bool_):
        raise TypeError(f"{label} must not contain boolean values")
    if not np.issubdtype(value.dtype, np.number):
        raise TypeError(f"{label} must contain real numeric values")
    if np.issubdtype(value.dtype, np.complexfloating):
        raise TypeError(f"{label} must contain real numeric values")
    owned = np.array(value, dtype=float, order="C", copy=True)
    if not bool(np.isfinite(owned).all()):
        raise ValueError(f"{label} must contain only finite values")
    owned.setflags(write=False)
    return owned


def _projected_cells(
    value: object,
    *,
    point_count: int,
) -> tuple[tuple[int, ...], ...]:
    if type(value) is not tuple:
        raise TypeError("cells must be a tuple")
    for cell in value:
        if type(cell) is not tuple:
            raise TypeError("cell connectivity rows must be tuples")
        if not cell:
            raise ValueError("cell connectivity rows must not be empty")
        if len(set(cell)) != len(cell):
            raise ValueError("cell connectivity must not repeat point indexes")
        for point_index in cell:
            if type(point_index) is not int:
                raise TypeError("cell point indexes must be integers")
            if point_index < 0 or point_index >= point_count:
                raise ValueError(
                    "cell point indexes must be zero-based projected indexes"
                )
    return value


def _cell_kinds(
    value: object,
    length: int,
) -> tuple[ResultCellKind, ...]:
    if type(value) is not tuple:
        raise TypeError("cell_kinds must be a tuple")
    if len(value) != length:
        raise ValueError("cell_kinds length must match cells")
    for item in value:
        if type(item) is not ResultCellKind:
            raise TypeError("cell_kinds must contain only ResultCellKind values")
    return value


def _validate_cell_shapes(
    cells: tuple[tuple[int, ...], ...],
    cell_kinds: tuple[ResultCellKind, ...],
) -> None:
    for cell, kind in zip(cells, cell_kinds, strict=True):
        if kind is ResultCellKind.SAMPLE_VERTEX and len(cell) != 1:
            raise ValueError("sample-vertex cells must contain exactly one point")


def _canonical_element_types(
    value: object,
    cell_kinds: tuple[ResultCellKind, ...],
) -> tuple[str | None, ...]:
    if type(value) is not tuple:
        raise TypeError("canonical_element_types must be a tuple")
    if len(value) != len(cell_kinds):
        raise ValueError("canonical_element_types length must match cells")
    for element_type, cell_kind in zip(value, cell_kinds, strict=True):
        if cell_kind is ResultCellKind.SAMPLE_VERTEX:
            if element_type is not None:
                raise ValueError("sample-vertex cells require a None element type")
            continue
        if type(element_type) is not str:
            raise TypeError("FEM-element cells require a canonical element type")
        if not element_type.strip():
            raise ValueError("canonical element types must not be blank")
    return value


def _location_tuple(
    value: object,
    *,
    length: int,
    label: str,
) -> tuple[FieldLocation | None, ...]:
    if type(value) is not tuple:
        raise TypeError(f"{label} must be a tuple")
    if len(value) != length:
        raise ValueError(f"{label} length does not match its topology")
    for location in value:
        if location is not None and type(location) is not FieldLocation:
            raise TypeError(f"{label} must contain FieldLocation or None values")
    return value


def _public_array_copy(owner: np.ndarray) -> np.ndarray:
    result = np.array(owner, dtype=owner.dtype, order="C", copy=True)
    result.setflags(write=False)
    return result


__all__ = [
    "ResultCellKind",
    "ResultFieldTopology",
    "ResultValueLayout",
    "project_scalar_field_topology",
]
