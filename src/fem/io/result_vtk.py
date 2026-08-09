"""Canonical, snapshot-bound scalar result VTK interchange."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import math
from numbers import Real
from pathlib import Path
from typing import Any

from fem.elements.beam_section import BeamSectionPoint
from fem.application.results.data import (
    FieldLocation,
    ResultExportSnapshot,
)
from fem.application.results.fields import (
    FieldAssociation,
    FieldMaterializationKey,
    FieldPosition,
    FieldRequest,
    PhysicalQuantity,
    ResultFieldId,
    ResultSourceKey,
    ResultVariable,
    ScalarFieldSelection,
)
from fem.application.results.topology import (
    ResultCellKind,
    ResultValueLayout,
    project_scalar_field_topology,
)
from fem.post.averaging import NodalAveragingPolicy
from fem.post.fields import (
    ResultRegionKey,
    decode_result_region_key,
    encode_result_region_key,
)
from fem.post.vtk.cells import vtk_cell_type
from fem.post.vtk.writer import (
    append_legacy_ascii_unstructured_grid_geometry,
)

from ._atomic_text import atomic_write_verified_text


RESULT_VTK_FORMAT_NAME = "fem-python-result-field"
RESULT_VTK_SCHEMA_VERSION = 1
RESULT_VTK_TITLE = "fem-python canonical result field"

_METADATA_NAMES = (
    "format_utf8",
    "schema",
    "result_id_utf8",
    "session_id_utf8",
    "artifact_id_utf8",
    "model_revision",
    "run_id_utf8",
    "step_name_utf8",
    "materialization_generation",
    "field_variable_utf8",
    "field_position_utf8",
    "component_utf8",
    "averaging_policy_present",
    "averaging_threshold_percent",
    "averaging_preserve_region_boundaries",
    "gauss_order_present",
    "gauss_order",
    "recovery_contract",
    "field_quantity_utf8",
    "field_association_utf8",
    "deformation_scale",
    "region_count",
)
_IDENTITY_NAMES = (
    "fem_node_id",
    "fem_node_id_valid",
    "fem_element_id",
    "fem_element_id_valid",
    "integration_point",
    "integration_point_valid",
    "local_node",
    "local_node_valid",
    "region_index",
    "region_index_valid",
    "averaged_state",
)
_SECTION_POINT_METADATA_NAMES = ("section_point_number",)
_SECTION_POINT_IDENTITY_NAMES = (
    "section_point_number",
    "section_point_valid",
    "section_point_local_y",
    "section_point_local_z",
)
_EXPECTED_ASSOCIATION = {
    FieldPosition.NODE: FieldAssociation.NODE,
    FieldPosition.INTEGRATION_POINT: FieldAssociation.INTEGRATION_POINT,
    FieldPosition.CENTROID: FieldAssociation.ELEMENT,
    FieldPosition.ELEMENT_NODAL: FieldAssociation.ELEMENT_NODE,
    FieldPosition.NODE_REGION: FieldAssociation.NODE_REGION,
    FieldPosition.RESOLVED_NODAL: FieldAssociation.RESOLVED_NODAL,
    FieldPosition.SECTION_END: FieldAssociation.ELEMENT_NODE,
    FieldPosition.SECTION_POINT: FieldAssociation.ELEMENT_NODE,
    FieldPosition.SECTION_NODE_ENVELOPE: FieldAssociation.NODE,
}
_EXPECTED_QUANTITY = {
    ResultVariable.U: PhysicalQuantity.DISPLACEMENT,
    ResultVariable.UR: PhysicalQuantity.ROTATION,
    ResultVariable.RF: PhysicalQuantity.FORCE,
    ResultVariable.RM: PhysicalQuantity.MOMENT,
    ResultVariable.S: PhysicalQuantity.STRESS,
    ResultVariable.LE: PhysicalQuantity.STRAIN,
}
_VTK_CELL_POINT_COUNTS = {
    1: 1,
    3: 2,
    5: 3,
    9: 4,
    10: 4,
    12: 8,
    22: 6,
    23: 8,
    24: 10,
    25: 20,
}


class ResultVtkError(ValueError):
    """Base error for canonical result VTK validation or serialization."""


class ResultVtkEncodeError(ResultVtkError):
    """The supplied export snapshot cannot be represented canonically."""


class ResultVtkDecodeError(ResultVtkError):
    """The input is not a canonical result VTK document."""


class ResultVtkEmptySelectionError(ResultVtkError):
    """The selected scalar projection contains no values."""


@dataclass(frozen=True, slots=True)
class ResultVtkLocationIdentity:
    """The exact FEM identity portion of one projected field location."""

    node_id: int | None = None
    element_id: int | None = None
    integration_point: int | None = None
    local_node: int | None = None
    region_key: ResultRegionKey | None = None
    averaged: bool | None = None
    section_point: BeamSectionPoint | None = None

    def __post_init__(self) -> None:
        for label in (
            "node_id",
            "element_id",
            "integration_point",
            "local_node",
        ):
            value = getattr(self, label)
            if value is None:
                continue
            if type(value) is not int:
                raise TypeError(f"{label} must be an integer or None")
            if value <= 0:
                raise ValueError(f"{label} must be positive")
        if self.region_key is not None and type(self.region_key) is not ResultRegionKey:
            raise TypeError("region_key must be ResultRegionKey or None")
        if self.averaged is not None and type(self.averaged) is not bool:
            raise TypeError("averaged must be bool or None")
        if self.section_point is not None and type(self.section_point) is not BeamSectionPoint:
            raise TypeError("section_point must be BeamSectionPoint or None")

    @classmethod
    def from_location(
        cls,
        location: FieldLocation,
    ) -> ResultVtkLocationIdentity:
        """Detach the identity fields from one canonical field location."""

        if type(location) is not FieldLocation:
            raise TypeError("location must be FieldLocation")
        return cls(
            node_id=location.node_id,
            element_id=location.element_id,
            integration_point=location.integration_point,
            local_node=location.local_node,
            region_key=location.region_key,
            averaged=location.averaged,
            section_point=location.section_point,
        )


@dataclass(frozen=True, slots=True)
class ResultVtkReadback:
    """Complete semantics reconstructed from one canonical VTK document."""

    source: ResultSourceKey
    materialization_generation: int
    selection: ScalarFieldSelection
    quantity: PhysicalQuantity
    association: FieldAssociation
    deformation_scale: float
    points: tuple[tuple[float, float, float], ...]
    cells: tuple[tuple[int, ...], ...]
    cell_types: tuple[int, ...]
    values: tuple[float, ...]
    value_layout: ResultValueLayout
    point_locations: tuple[ResultVtkLocationIdentity | None, ...]
    cell_locations: tuple[ResultVtkLocationIdentity | None, ...]
    region_table: tuple[ResultRegionKey, ...]

    def __post_init__(self) -> None:
        if type(self.source) is not ResultSourceKey:
            raise TypeError("source must be ResultSourceKey")
        if type(self.materialization_generation) is not int:
            raise TypeError("materialization_generation must be an integer")
        if self.materialization_generation < 0:
            raise ValueError("materialization_generation must be non-negative")
        if type(self.selection) is not ScalarFieldSelection:
            raise TypeError("selection must be ScalarFieldSelection")
        if type(self.quantity) is not PhysicalQuantity:
            raise TypeError("quantity must be PhysicalQuantity")
        if type(self.association) is not FieldAssociation:
            raise TypeError("association must be FieldAssociation")
        object.__setattr__(
            self,
            "deformation_scale",
            _finite_number(
                self.deformation_scale,
                label="deformation_scale",
            ),
        )
        object.__setattr__(
            self,
            "points",
            _points_tuple(self.points),
        )
        object.__setattr__(
            self,
            "cells",
            _cells_tuple(self.cells, point_count=len(self.points)),
        )
        object.__setattr__(
            self,
            "cell_types",
            _cell_type_tuple(self.cell_types, self.cells),
        )
        object.__setattr__(
            self,
            "values",
            _finite_value_tuple(self.values),
        )
        if not self.values:
            raise ValueError("values must not be empty")
        if type(self.value_layout) is not ResultValueLayout:
            raise TypeError("value_layout must be ResultValueLayout")
        expected_values = (
            len(self.points)
            if self.value_layout is ResultValueLayout.POINT
            else len(self.cells)
        )
        if len(self.values) != expected_values:
            raise ValueError("values length must match its point or cell layout")
        _validate_location_tuple(
            self.point_locations,
            length=len(self.points),
            label="point_locations",
            association=self.association,
        )
        _validate_location_tuple(
            self.cell_locations,
            length=len(self.cells),
            label="cell_locations",
            association=self.association,
        )
        scalar_locations = (
            self.point_locations
            if self.value_layout is ResultValueLayout.POINT
            else self.cell_locations
        )
        if any(location is None for location in scalar_locations):
            raise ValueError(
                "every selected scalar requires an exact location identity"
            )
        if len(set(scalar_locations)) != len(scalar_locations):
            raise ValueError("selected scalar locations must use unique identities")
        _validate_region_table(
            self.region_table,
            self.point_locations,
            self.cell_locations,
        )

        field_id = self.selection.field_key.request.field_id
        if self.association is not _EXPECTED_ASSOCIATION[field_id.position]:
            raise ValueError("association does not match the selected field position")
        if self.quantity is not _EXPECTED_QUANTITY[field_id.variable]:
            raise ValueError("quantity does not match the selected field variable")


@dataclass(frozen=True, slots=True)
class _FieldArray:
    name: str
    components: int
    tuples: int
    data_type: str
    values: tuple[str, ...]


class _LineReader:
    def __init__(self, lines: list[str]) -> None:
        self._lines = lines
        self._index = 0

    def take(self, label: str) -> str:
        if self._index >= len(self._lines):
            raise ResultVtkDecodeError(f"result VTK is missing {label}")
        value = self._lines[self._index]
        self._index += 1
        return value

    def require_end(self) -> None:
        if self._index != len(self._lines):
            raise ResultVtkDecodeError(
                "result VTK contains unexpected trailing content"
            )


def dumps_result_vtk(
    snapshot: ResultExportSnapshot,
    deformation_scale: float = 0.0,
) -> str:
    """Serialize one selected scalar field as canonical VTK Legacy ASCII."""

    projected = _project_result_vtk(snapshot, deformation_scale)
    return _serialize_result_vtk(projected)


def read_result_vtk(path: str | Path) -> ResultVtkReadback:
    """Read and strictly validate one canonical result VTK file."""

    raw = Path(path).read_bytes()
    try:
        text = raw.decode("ascii", errors="strict")
    except UnicodeDecodeError as error:
        raise ResultVtkDecodeError(
            "canonical result VTK must contain ASCII wire text"
        ) from error
    if "\r" in text:
        raise ResultVtkDecodeError("result VTK must use LF line endings")
    if not text.endswith("\n"):
        raise ResultVtkDecodeError("result VTK must end with a final newline")
    reader = _LineReader(text[:-1].split("\n"))

    _expect_line(
        reader,
        "# vtk DataFile Version 3.0",
        label="VTK Legacy 3.0 header",
    )
    _expect_line(reader, RESULT_VTK_TITLE, label="canonical title")
    _expect_line(reader, "ASCII", label="ASCII declaration")
    _expect_line(
        reader,
        "DATASET UNSTRUCTURED_GRID",
        label="UNSTRUCTURED_GRID declaration",
    )

    points = _read_points(reader)
    cells = _read_cells(reader, point_count=len(points))
    cell_types = _read_cell_types(reader, cells)
    metadata_arrays = _read_field_arrays(
        reader,
        expected_name="ResultMetadata",
    )
    (
        source,
        generation,
        selection,
        quantity,
        association,
        deformation_scale,
        region_table,
    ) = _decode_metadata(metadata_arrays)

    point_count = _read_data_count(
        reader,
        keyword="POINT_DATA",
        expected=len(points),
    )
    point_values, point_identity = _read_data_section(
        reader,
        field_name="PointIdentity",
        count=point_count,
        region_table=region_table,
    )
    cell_count = _read_data_count(
        reader,
        keyword="CELL_DATA",
        expected=len(cells),
    )
    cell_values, cell_identity = _read_data_section(
        reader,
        field_name="CellIdentity",
        count=cell_count,
        region_table=region_table,
    )
    reader.require_end()

    if point_values is not None and cell_values is not None:
        raise ResultVtkDecodeError(
            "result VTK must contain exactly one selected scalar array"
        )
    if point_values is None and cell_values is None:
        raise ResultVtkDecodeError("result VTK selected scalar array is missing")
    value_layout = (
        ResultValueLayout.POINT if point_values is not None else ResultValueLayout.CELL
    )
    values = point_values if point_values is not None else cell_values
    assert values is not None
    try:
        return ResultVtkReadback(
            source=source,
            materialization_generation=generation,
            selection=selection,
            quantity=quantity,
            association=association,
            deformation_scale=deformation_scale,
            points=points,
            cells=cells,
            cell_types=cell_types,
            values=values,
            value_layout=value_layout,
            point_locations=point_identity,
            cell_locations=cell_identity,
            region_table=region_table,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ResultVtkDecodeError(
            f"result VTK semantics are invalid: {error}"
        ) from error


def write_result_vtk(
    path: str | Path,
    snapshot: ResultExportSnapshot,
    deformation_scale: float = 0.0,
    *,
    checkpoint: Callable[[], Any] | None = None,
    before_replace: Callable[[], Any] | None = None,
) -> Path:
    """Durably verify and atomically install one canonical result VTK."""

    if checkpoint is not None and not callable(checkpoint):
        raise TypeError("checkpoint must be callable or None")
    target = Path(path)
    if target.suffix.casefold() != ".vtk":
        raise ResultVtkEncodeError(
            "canonical result VTK target must use the .vtk extension"
        )
    projected = _project_result_vtk(snapshot, deformation_scale)
    if checkpoint is not None:
        checkpoint()
    serialized = _serialize_result_vtk(projected)
    return atomic_write_verified_text(
        target,
        serialized,
        verifier=read_result_vtk,
        semantic_encoder=_semantic_identity,
        expected_semantic=projected,
        error_type=ResultVtkEncodeError,
        mismatch_message=("temporary result VTK semantic verification failed"),
        checkpoint=checkpoint,
        before_replace=before_replace,
    )


def _project_result_vtk(
    snapshot: ResultExportSnapshot,
    deformation_scale: float,
) -> ResultVtkReadback:
    if type(snapshot) is not ResultExportSnapshot:
        raise TypeError("snapshot must be ResultExportSnapshot")
    if not snapshot.field.locations:
        raise ResultVtkEmptySelectionError(
            "result VTK export requires at least one scalar value"
        )
    try:
        topology = project_scalar_field_topology(
            snapshot,
            deformation_scale,
        )
        values = tuple(float(value) for value in topology.values)
        if not values:
            raise ResultVtkEmptySelectionError(
                "result VTK export requires at least one scalar value"
            )
        cell_types = tuple(
            (
                1
                if kind is ResultCellKind.SAMPLE_VERTEX
                else vtk_cell_type(_require_element_type(element_type))
            )
            for kind, element_type in zip(
                topology.cell_kinds,
                topology.canonical_element_types,
                strict=True,
            )
        )
        point_locations = tuple(
            (
                None
                if location is None
                else ResultVtkLocationIdentity.from_location(location)
            )
            for location in topology.point_locations
        )
        cell_locations = tuple(
            (
                None
                if location is None
                else ResultVtkLocationIdentity.from_location(location)
            )
            for location in topology.cell_locations
        )
        region_table = _build_region_table(
            point_locations,
            cell_locations,
        )
        descriptor = snapshot.field.descriptor
        return ResultVtkReadback(
            source=topology.source,
            materialization_generation=(topology.materialization_generation),
            selection=topology.selection,
            quantity=descriptor.quantity,
            association=descriptor.association,
            deformation_scale=topology.deformation_scale,
            points=tuple(
                tuple(float(component) for component in point)
                for point in topology.points
            ),
            cells=topology.cells,
            cell_types=cell_types,
            values=values,
            value_layout=topology.value_layout,
            point_locations=point_locations,
            cell_locations=cell_locations,
            region_table=region_table,
        )
    except ResultVtkError:
        raise
    except (KeyError, TypeError, ValueError) as error:
        raise ResultVtkEncodeError(
            f"snapshot cannot be represented as result VTK: {error}"
        ) from error


def _serialize_result_vtk(projected: ResultVtkReadback) -> str:
    lines: list[str] = []
    append_legacy_ascii_unstructured_grid_geometry(
        lines,
        title=RESULT_VTK_TITLE,
        points=projected.points,
        cells=projected.cells,
        cell_types=projected.cell_types,
        numeric_declaration="double",
        format_float=_format_float,
    )
    _write_field_arrays(
        lines,
        "ResultMetadata",
        _metadata_arrays(projected),
    )

    lines.append(f"POINT_DATA {len(projected.points)}")
    if projected.value_layout is ResultValueLayout.POINT:
        _write_scalar_array(lines, projected.values)
    _write_field_arrays(
        lines,
        "PointIdentity",
        _identity_arrays(
            projected.point_locations,
            projected.region_table,
        ),
    )

    lines.append(f"CELL_DATA {len(projected.cells)}")
    if projected.value_layout is ResultValueLayout.CELL:
        _write_scalar_array(lines, projected.values)
    _write_field_arrays(
        lines,
        "CellIdentity",
        _identity_arrays(
            projected.cell_locations,
            projected.region_table,
        ),
    )
    return "\n".join(lines) + "\n"


def _metadata_arrays(
    projected: ResultVtkReadback,
) -> tuple[_FieldArray, ...]:
    source = projected.source
    selection = projected.selection
    key = selection.field_key
    request = key.request
    field_id = request.field_id
    policy = request.averaging_policy
    arrays = (
        _utf8_array("format_utf8", RESULT_VTK_FORMAT_NAME),
        _scalar_array("schema", "int", RESULT_VTK_SCHEMA_VERSION),
        _utf8_array("result_id_utf8", source.result_id),
        _utf8_array("session_id_utf8", source.session_id),
        _utf8_array("artifact_id_utf8", source.artifact_id),
        _scalar_array("model_revision", "long", source.model_revision),
        _utf8_array("run_id_utf8", source.run_id),
        _utf8_array("step_name_utf8", source.step_name),
        _scalar_array(
            "materialization_generation",
            "long",
            projected.materialization_generation,
        ),
        _utf8_array("field_variable_utf8", field_id.variable.value),
        _utf8_array("field_position_utf8", field_id.position.value),
        _utf8_array("component_utf8", selection.component),
        _scalar_array(
            "averaging_policy_present",
            "unsigned_char",
            int(policy is not None),
        ),
        _scalar_array(
            "averaging_threshold_percent",
            "double",
            0.0 if policy is None else policy.threshold_percent,
        ),
        _scalar_array(
            "averaging_preserve_region_boundaries",
            "unsigned_char",
            (0 if policy is None else int(policy.preserve_region_boundaries)),
        ),
        _scalar_array(
            "gauss_order_present",
            "unsigned_char",
            int(request.gauss_order is not None),
        ),
        _scalar_array(
            "gauss_order",
            "long",
            0 if request.gauss_order is None else request.gauss_order,
        ),
        _scalar_array(
            "recovery_contract",
            "long",
            key.recovery_contract,
        ),
        _utf8_array("field_quantity_utf8", projected.quantity.value),
        _utf8_array(
            "field_association_utf8",
            projected.association.value,
        ),
        _scalar_array(
            "deformation_scale",
            "double",
            projected.deformation_scale,
        ),
        _scalar_array(
            "region_count",
            "long",
            len(projected.region_table),
        ),
    )
    point_arrays = (
        ()
        if field_id.section_point_number is None
        else (
            _scalar_array(
                "section_point_number",
                "long",
                field_id.section_point_number,
            ),
        )
    )
    region_arrays = tuple(
        _utf8_array(
            f"region_{index}_utf8",
            encode_result_region_key(region_key),
        )
        for index, region_key in enumerate(projected.region_table)
    )
    return arrays + point_arrays + region_arrays


def _identity_arrays(
    locations: tuple[ResultVtkLocationIdentity | None, ...],
    region_table: tuple[ResultRegionKey, ...],
) -> tuple[_FieldArray, ...]:
    region_indexes = {
        region_key: index for index, region_key in enumerate(region_table)
    }
    node_values, node_validity = _optional_identity_values(
        locations,
        "node_id",
    )
    element_values, element_validity = _optional_identity_values(
        locations,
        "element_id",
    )
    integration_values, integration_validity = _optional_identity_values(
        locations,
        "integration_point",
    )
    local_values, local_validity = _optional_identity_values(
        locations,
        "local_node",
    )
    point_validity = tuple(
        int(location is not None and location.section_point is not None)
        for location in locations
    )
    point_numbers = tuple(
        0
        if location is None or location.section_point is None
        else location.section_point.number
        for location in locations
    )
    point_local_y = tuple(
        0.0
        if location is None or location.section_point is None
        else location.section_point.local_y
        for location in locations
    )
    point_local_z = tuple(
        0.0
        if location is None or location.section_point is None
        else location.section_point.local_z
        for location in locations
    )
    region_values = tuple(
        (
            0
            if location is None or location.region_key is None
            else region_indexes[location.region_key]
        )
        for location in locations
    )
    region_validity = tuple(
        int(location is not None and location.region_key is not None)
        for location in locations
    )
    averaged_state = tuple(
        (
            0
            if location is None or location.averaged is None
            else (2 if location.averaged else 1)
        )
        for location in locations
    )
    count = len(locations)
    return (
        _vector_array("fem_node_id", "long", node_values, count),
        _vector_array(
            "fem_node_id_valid",
            "unsigned_char",
            node_validity,
            count,
        ),
        _vector_array(
            "fem_element_id",
            "long",
            element_values,
            count,
        ),
        _vector_array(
            "fem_element_id_valid",
            "unsigned_char",
            element_validity,
            count,
        ),
        _vector_array(
            "integration_point",
            "long",
            integration_values,
            count,
        ),
        _vector_array(
            "integration_point_valid",
            "unsigned_char",
            integration_validity,
            count,
        ),
        _vector_array("local_node", "long", local_values, count),
        _vector_array(
            "local_node_valid",
            "unsigned_char",
            local_validity,
            count,
        ),
        *(
            (
                _vector_array(
                    "section_point_number", "long", point_numbers, count
                ),
                _vector_array(
                    "section_point_valid",
                    "unsigned_char",
                    point_validity,
                    count,
                ),
                _vector_array(
                    "section_point_local_y",
                    "double",
                    point_local_y,
                    count,
                ),
                _vector_array(
                    "section_point_local_z",
                    "double",
                    point_local_z,
                    count,
                ),
            )
            if any(point_validity)
            else ()
        ),
        _vector_array("region_index", "long", region_values, count),
        _vector_array(
            "region_index_valid",
            "unsigned_char",
            region_validity,
            count,
        ),
        _vector_array(
            "averaged_state",
            "unsigned_char",
            averaged_state,
            count,
        ),
    )


def _write_scalar_array(lines: list[str], values: tuple[float, ...]) -> None:
    lines.extend(
        (
            "SCALARS selected_scalar double 1",
            "LOOKUP_TABLE default",
        )
    )
    lines.extend(_format_float(value) for value in values)


def _write_field_arrays(
    lines: list[str],
    name: str,
    arrays: tuple[_FieldArray, ...],
) -> None:
    lines.append(f"FIELD {name} {len(arrays)}")
    for array in arrays:
        lines.append(
            f"{array.name} {array.components} {array.tuples} {array.data_type}"
        )
        lines.append(" ".join(array.values))


def _read_points(
    reader: _LineReader,
) -> tuple[tuple[float, float, float], ...]:
    parts = reader.take("POINTS declaration").split()
    if len(parts) != 3 or parts[0] != "POINTS" or parts[2] != "double":
        raise ResultVtkDecodeError("result VTK POINTS must declare double precision")
    count = _parse_integer(parts[1], label="point count", minimum=0)
    points = []
    for index in range(count):
        values = reader.take(f"point {index}").split()
        if len(values) != 3:
            raise ResultVtkDecodeError(
                "each result VTK point must contain three components"
            )
        points.append(
            tuple(_parse_float(value, label=f"point {index}") for value in values)
        )
    return tuple(points)


def _read_cells(
    reader: _LineReader,
    *,
    point_count: int,
) -> tuple[tuple[int, ...], ...]:
    parts = reader.take("CELLS declaration").split()
    if len(parts) != 3 or parts[0] != "CELLS":
        raise ResultVtkDecodeError("result VTK CELLS declaration is invalid")
    count = _parse_integer(parts[1], label="cell count", minimum=0)
    declared_size = _parse_integer(
        parts[2],
        label="cell list size",
        minimum=0,
    )
    cells = []
    actual_size = 0
    for cell_index in range(count):
        values = reader.take(f"cell {cell_index}").split()
        if not values:
            raise ResultVtkDecodeError("result VTK cell row is empty")
        cell_size = _parse_integer(
            values[0],
            label=f"cell {cell_index} size",
            minimum=1,
        )
        if len(values) != cell_size + 1:
            raise ResultVtkDecodeError(
                f"cell {cell_index} connectivity length is invalid"
            )
        connectivity = tuple(
            _parse_integer(
                value,
                label=f"cell {cell_index} point index",
                minimum=0,
            )
            for value in values[1:]
        )
        if any(point_index >= point_count for point_index in connectivity):
            raise ResultVtkDecodeError(
                f"cell {cell_index} references an unknown point index"
            )
        if len(set(connectivity)) != len(connectivity):
            raise ResultVtkDecodeError(f"cell {cell_index} repeats a point index")
        cells.append(connectivity)
        actual_size += cell_size + 1
    if actual_size != declared_size:
        raise ResultVtkDecodeError(
            "result VTK CELLS list size does not match connectivity"
        )
    return tuple(cells)


def _read_cell_types(
    reader: _LineReader,
    cells: tuple[tuple[int, ...], ...],
) -> tuple[int, ...]:
    parts = reader.take("CELL_TYPES declaration").split()
    if len(parts) != 2 or parts[0] != "CELL_TYPES":
        raise ResultVtkDecodeError("result VTK CELL_TYPES declaration is invalid")
    count = _parse_integer(parts[1], label="cell type count", minimum=0)
    if count != len(cells):
        raise ResultVtkDecodeError("CELL_TYPES count must match CELLS count")
    result = tuple(
        _parse_integer(
            reader.take(f"cell type {index}"),
            label=f"cell type {index}",
            minimum=1,
        )
        for index in range(count)
    )
    for index, (cell, cell_type) in enumerate(zip(cells, result, strict=True)):
        expected = _VTK_CELL_POINT_COUNTS.get(cell_type)
        if expected is None:
            raise ResultVtkDecodeError(
                f"cell {index} uses an unsupported VTK cell type"
            )
        if len(cell) != expected:
            raise ResultVtkDecodeError(
                f"cell {index} connectivity does not match its VTK type"
            )
    return result


def _read_field_arrays(
    reader: _LineReader,
    *,
    expected_name: str,
) -> tuple[_FieldArray, ...]:
    parts = reader.take(f"FIELD {expected_name} declaration").split()
    if len(parts) != 3 or parts[0] != "FIELD" or parts[1] != expected_name:
        raise ResultVtkDecodeError(
            f"result VTK FIELD {expected_name} declaration is invalid"
        )
    count = _parse_integer(
        parts[2],
        label=f"{expected_name} array count",
        minimum=0,
    )
    return _read_field_array_body(
        reader,
        expected_name=expected_name,
        count=count,
    )


def _read_field_array_body(
    reader: _LineReader,
    *,
    expected_name: str,
    count: int,
) -> tuple[_FieldArray, ...]:
    arrays = []
    names: set[str] = set()
    for index in range(count):
        header = reader.take(f"{expected_name} array {index} declaration").split()
        if len(header) != 4:
            raise ResultVtkDecodeError(f"{expected_name} array declaration is invalid")
        name, components_text, tuples_text, data_type = header
        if not name or name in names:
            raise ResultVtkDecodeError(f"{expected_name} array names must be unique")
        components = _parse_integer(
            components_text,
            label=f"{name} component count",
            minimum=1,
        )
        tuples = _parse_integer(
            tuples_text,
            label=f"{name} tuple count",
            minimum=0,
        )
        if data_type not in {"double", "int", "long", "unsigned_char"}:
            raise ResultVtkDecodeError(f"{name} uses an unsupported FIELD data type")
        value_line = reader.take(f"{name} values")
        values = tuple(value_line.split()) if value_line else ()
        if len(values) != components * tuples:
            raise ResultVtkDecodeError(
                f"{name} FIELD value count does not match its declaration"
            )
        names.add(name)
        arrays.append(
            _FieldArray(
                name=name,
                components=components,
                tuples=tuples,
                data_type=data_type,
                values=values,
            )
        )
    return tuple(arrays)


def _decode_metadata(
    arrays: tuple[_FieldArray, ...],
) -> tuple[
    ResultSourceKey,
    int,
    ScalarFieldSelection,
    PhysicalQuantity,
    FieldAssociation,
    float,
    tuple[ResultRegionKey, ...],
]:
    if len(arrays) < len(_METADATA_NAMES):
        raise ResultVtkDecodeError(
            "ResultMetadata does not contain the complete schema"
        )
    if tuple(array.name for array in arrays[: len(_METADATA_NAMES)]) != (
        _METADATA_NAMES
    ):
        raise ResultVtkDecodeError("ResultMetadata arrays do not match canonical order")
    by_name = {array.name: array for array in arrays}
    if _decode_utf8(by_name["format_utf8"]) != RESULT_VTK_FORMAT_NAME:
        raise ResultVtkDecodeError("result VTK format metadata is invalid")
    if (
        _decode_scalar_int(
            by_name["schema"],
            data_type="int",
            minimum=1,
        )
        != RESULT_VTK_SCHEMA_VERSION
    ):
        raise ResultVtkDecodeError("result VTK schema metadata is invalid")
    region_count = _decode_scalar_int(
        by_name["region_count"],
        data_type="long",
        minimum=0,
    )
    point_names = (
        _SECTION_POINT_METADATA_NAMES
        if len(arrays) > len(_METADATA_NAMES)
        and arrays[len(_METADATA_NAMES)].name == "section_point_number"
        else ()
    )
    expected_names = _METADATA_NAMES + point_names + tuple(
        f"region_{index}_utf8" for index in range(region_count)
    )
    if tuple(array.name for array in arrays) != expected_names:
        raise ResultVtkDecodeError("ResultMetadata region table arrays are invalid")

    try:
        source = ResultSourceKey(
            result_id=_decode_utf8(by_name["result_id_utf8"]),
            session_id=_decode_utf8(by_name["session_id_utf8"]),
            artifact_id=_decode_utf8(by_name["artifact_id_utf8"]),
            model_revision=_decode_scalar_int(
                by_name["model_revision"],
                data_type="long",
                minimum=0,
            ),
            step_name=_decode_utf8(by_name["step_name_utf8"]),
            run_id=_decode_utf8(by_name["run_id_utf8"]),
        )
        generation = _decode_scalar_int(
            by_name["materialization_generation"],
            data_type="long",
            minimum=0,
        )
        variable = ResultVariable(_decode_utf8(by_name["field_variable_utf8"]))
        position = FieldPosition(_decode_utf8(by_name["field_position_utf8"]))
        section_point_number = (
            _decode_scalar_int(
                by_name["section_point_number"],
                data_type="long",
                minimum=1,
            )
            if point_names
            else None
        )
        policy_present = _decode_flag(by_name["averaging_policy_present"])
        threshold = _decode_scalar_float(by_name["averaging_threshold_percent"])
        preserve = _decode_flag(by_name["averaging_preserve_region_boundaries"])
        if not policy_present and (threshold != 0.0 or preserve):
            raise ValueError(
                "absent averaging policy must use canonical zero placeholders"
            )
        policy = (
            NodalAveragingPolicy(
                threshold_percent=threshold,
                preserve_region_boundaries=preserve,
            )
            if policy_present
            else None
        )
        gauss_present = _decode_flag(by_name["gauss_order_present"])
        gauss_order = _decode_scalar_int(
            by_name["gauss_order"],
            data_type="long",
            minimum=0,
        )
        if not gauss_present and gauss_order != 0:
            raise ValueError("absent gauss order must use a canonical zero placeholder")
        if gauss_present and gauss_order == 0:
            raise ValueError("present gauss order must be positive")
        request = FieldRequest(
            field_id=ResultFieldId(
                variable,
                position,
                section_point_number=section_point_number,
            ),
            averaging_policy=policy,
            gauss_order=gauss_order if gauss_present else None,
        )
        key = FieldMaterializationKey(
            request=request,
            recovery_contract=_decode_scalar_int(
                by_name["recovery_contract"],
                data_type="long",
                minimum=1,
            ),
        )
        selection = ScalarFieldSelection(
            field_key=key,
            component=_decode_utf8(by_name["component_utf8"]),
        )
        quantity = PhysicalQuantity(_decode_utf8(by_name["field_quantity_utf8"]))
        association = FieldAssociation(_decode_utf8(by_name["field_association_utf8"]))
        deformation_scale = _decode_scalar_float(by_name["deformation_scale"])
        region_texts = tuple(
            _decode_utf8(by_name[f"region_{index}_utf8"])
            for index in range(region_count)
        )
        if tuple(sorted(set(region_texts))) != region_texts:
            raise ValueError(
                "region table must contain unique canonical sorted strings"
            )
        region_table = tuple(decode_result_region_key(value) for value in region_texts)
        if (
            tuple(encode_result_region_key(region) for region in region_table)
            != region_texts
        ):
            raise ValueError("region table must use the canonical codec")
        return (
            source,
            generation,
            selection,
            quantity,
            association,
            deformation_scale,
            region_table,
        )
    except ResultVtkError:
        raise
    except (KeyError, TypeError, UnicodeDecodeError, ValueError) as error:
        raise ResultVtkDecodeError(
            f"ResultMetadata semantics are invalid: {error}"
        ) from error


def _read_data_count(
    reader: _LineReader,
    *,
    keyword: str,
    expected: int,
) -> int:
    parts = reader.take(f"{keyword} declaration").split()
    if len(parts) != 2 or parts[0] != keyword:
        raise ResultVtkDecodeError(f"result VTK {keyword} declaration is invalid")
    count = _parse_integer(
        parts[1],
        label=f"{keyword} count",
        minimum=0,
    )
    if count != expected:
        raise ResultVtkDecodeError(
            f"{keyword} count does not match the projected topology"
        )
    return count


def _read_data_section(
    reader: _LineReader,
    *,
    field_name: str,
    count: int,
    region_table: tuple[ResultRegionKey, ...],
) -> tuple[
    tuple[float, ...] | None,
    tuple[ResultVtkLocationIdentity | None, ...],
]:
    line = reader.take(f"{field_name} data")
    values: tuple[float, ...] | None = None
    if line == "SCALARS selected_scalar double 1":
        _expect_line(
            reader,
            "LOOKUP_TABLE default",
            label="selected scalar lookup table",
        )
        values = tuple(
            _parse_float(
                reader.take(f"selected scalar {index}"),
                label=f"selected scalar {index}",
            )
            for index in range(count)
        )
        line = reader.take(f"FIELD {field_name} declaration")
    arrays = _read_field_arrays_from_declaration(
        reader,
        line=line,
        expected_name=field_name,
    )
    return values, _decode_identity_arrays(
        arrays,
        count=count,
        region_table=region_table,
    )


def _read_field_arrays_from_declaration(
    reader: _LineReader,
    *,
    line: str,
    expected_name: str,
) -> tuple[_FieldArray, ...]:
    parts = line.split()
    if len(parts) != 3 or parts[0] != "FIELD" or parts[1] != expected_name:
        raise ResultVtkDecodeError(
            f"result VTK FIELD {expected_name} declaration is invalid"
        )
    count = _parse_integer(
        parts[2],
        label=f"{expected_name} array count",
        minimum=0,
    )
    return _read_field_array_body(
        reader,
        expected_name=expected_name,
        count=count,
    )


def _decode_identity_arrays(
    arrays: tuple[_FieldArray, ...],
    *,
    count: int,
    region_table: tuple[ResultRegionKey, ...],
) -> tuple[ResultVtkLocationIdentity | None, ...]:
    point_offset = _IDENTITY_NAMES.index("region_index")
    expanded_names = (
        _IDENTITY_NAMES[:point_offset]
        + _SECTION_POINT_IDENTITY_NAMES
        + _IDENTITY_NAMES[point_offset:]
    )
    actual_names = tuple(array.name for array in arrays)
    if actual_names not in {_IDENTITY_NAMES, expanded_names}:
        raise ResultVtkDecodeError(
            "result VTK identity arrays do not match canonical order"
        )
    has_section_points = actual_names == expanded_names
    by_name = {array.name: array for array in arrays}
    integer_names = tuple(
        name
        for name in actual_names
        if name not in {"section_point_local_y", "section_point_local_z"}
    )
    values = {
        name: _decode_vector_int(
            by_name[name],
            data_type=(
                "unsigned_char"
                if name.endswith("_valid")
                or name in {"averaged_state", "section_point_valid"}
                else "long"
            ),
            count=count,
            minimum=0,
        )
        for name in integer_names
    }
    point_local_y = (
        _decode_vector_float(by_name["section_point_local_y"], count=count)
        if has_section_points
        else (0.0,) * count
    )
    point_local_z = (
        _decode_vector_float(by_name["section_point_local_z"], count=count)
        if has_section_points
        else (0.0,) * count
    )
    result = []
    for index in range(count):
        node_id = _decode_optional_identity(
            values["fem_node_id"][index],
            values["fem_node_id_valid"][index],
            label="fem_node_id",
        )
        element_id = _decode_optional_identity(
            values["fem_element_id"][index],
            values["fem_element_id_valid"][index],
            label="fem_element_id",
        )
        integration_point = _decode_optional_identity(
            values["integration_point"][index],
            values["integration_point_valid"][index],
            label="integration_point",
        )
        local_node = _decode_optional_identity(
            values["local_node"][index],
            values["local_node_valid"][index],
            label="local_node",
        )
        point_number = (
            _decode_optional_identity(
                values["section_point_number"][index],
                values["section_point_valid"][index],
                label="section_point_number",
            )
            if has_section_points
            else None
        )
        if point_number is None and (
            point_local_y[index] != 0.0 or point_local_z[index] != 0.0
        ):
            raise ResultVtkDecodeError(
                "absent section point must use zero coordinate placeholders"
            )
        section_point = (
            None
            if point_number is None
            else BeamSectionPoint(
                point_number,
                point_local_y[index],
                point_local_z[index],
            )
        )
        region_index = _decode_optional_index(
            values["region_index"][index],
            values["region_index_valid"][index],
            count=len(region_table),
        )
        state = values["averaged_state"][index]
        if state not in {0, 1, 2}:
            raise ResultVtkDecodeError("averaged_state must be missing, false, or true")
        averaged = None if state == 0 else state == 2
        region_key = None if region_index is None else region_table[region_index]
        if (
            node_id is None
            and element_id is None
            and integration_point is None
            and local_node is None
            and region_key is None
            and averaged is None
            and section_point is None
        ):
            result.append(None)
            continue
        try:
            result.append(
                ResultVtkLocationIdentity(
                    node_id=node_id,
                    element_id=element_id,
                    integration_point=integration_point,
                    local_node=local_node,
                    region_key=region_key,
                    averaged=averaged,
                    section_point=section_point,
                )
            )
        except (TypeError, ValueError) as error:
            raise ResultVtkDecodeError(
                f"result VTK identity row {index} is invalid: {error}"
            ) from error
    return tuple(result)


def _utf8_array(name: str, value: str) -> _FieldArray:
    if type(value) is not str or not value:
        raise ResultVtkEncodeError(f"{name} must be a non-empty string")
    try:
        encoded = value.encode("utf-8", errors="strict")
    except UnicodeEncodeError as error:
        raise ResultVtkEncodeError(f"{name} must contain valid Unicode text") from error
    return _FieldArray(
        name=name,
        components=1,
        tuples=len(encoded),
        data_type="unsigned_char",
        values=tuple(str(byte) for byte in encoded),
    )


def _scalar_array(
    name: str,
    data_type: str,
    value: int | float,
) -> _FieldArray:
    if data_type == "double":
        text = _format_float(value)
    else:
        if type(value) is not int:
            raise ResultVtkEncodeError(f"{name} must be an integer FIELD scalar")
        text = str(value)
    return _FieldArray(name, 1, 1, data_type, (text,))


def _vector_array(
    name: str,
    data_type: str,
    values: tuple[int | float, ...],
    count: int,
) -> _FieldArray:
    if len(values) != count:
        raise ResultVtkEncodeError(f"{name} must contain one value per identity row")
    if data_type == "double":
        encoded = tuple(_format_float(value) for value in values)
    else:
        if any(type(value) is not int for value in values):
            raise ResultVtkEncodeError(
                f"{name} must contain one integer per identity row"
            )
        encoded = tuple(str(value) for value in values)
    return _FieldArray(
        name,
        1,
        count,
        data_type,
        encoded,
    )


def _decode_utf8(array: _FieldArray) -> str:
    _require_array_shape(array, data_type="unsigned_char")
    if array.tuples == 0:
        raise ResultVtkDecodeError(f"{array.name} UTF-8 metadata must not be empty")
    encoded = bytes(
        _parse_integer(
            value,
            label=f"{array.name} byte",
            minimum=0,
            maximum=255,
        )
        for value in array.values
    )
    try:
        decoded = encoded.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise ResultVtkDecodeError(
            f"{array.name} does not contain valid UTF-8 bytes"
        ) from error
    if not decoded:
        raise ResultVtkDecodeError(f"{array.name} UTF-8 metadata must not be empty")
    return decoded


def _decode_scalar_int(
    array: _FieldArray,
    *,
    data_type: str,
    minimum: int,
) -> int:
    _require_array_shape(
        array,
        data_type=data_type,
        tuples=1,
    )
    return _parse_integer(
        array.values[0],
        label=array.name,
        minimum=minimum,
    )


def _decode_scalar_float(array: _FieldArray) -> float:
    _require_array_shape(array, data_type="double", tuples=1)
    return _parse_float(array.values[0], label=array.name)


def _decode_flag(array: _FieldArray) -> bool:
    value = _decode_scalar_int(
        array,
        data_type="unsigned_char",
        minimum=0,
    )
    if value not in {0, 1}:
        raise ResultVtkDecodeError(f"{array.name} must be zero or one")
    return bool(value)


def _decode_vector_int(
    array: _FieldArray,
    *,
    data_type: str,
    count: int,
    minimum: int,
) -> tuple[int, ...]:
    _require_array_shape(
        array,
        data_type=data_type,
        tuples=count,
    )
    return tuple(
        _parse_integer(
            value,
            label=f"{array.name} value",
            minimum=minimum,
        )
        for value in array.values
    )


def _decode_vector_float(
    array: _FieldArray,
    *,
    count: int,
) -> tuple[float, ...]:
    _require_array_shape(array, data_type="double", tuples=count)
    return tuple(
        _parse_float(value, label=f"{array.name} value")
        for value in array.values
    )


def _require_array_shape(
    array: _FieldArray,
    *,
    data_type: str,
    tuples: int | None = None,
) -> None:
    if array.components != 1 or array.data_type != data_type:
        raise ResultVtkDecodeError(f"{array.name} FIELD declaration is not canonical")
    if tuples is not None and array.tuples != tuples:
        raise ResultVtkDecodeError(f"{array.name} FIELD tuple count is invalid")


def _optional_identity_values(
    locations: tuple[ResultVtkLocationIdentity | None, ...],
    attribute: str,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    return (
        tuple(
            (
                0
                if location is None or getattr(location, attribute) is None
                else getattr(location, attribute)
            )
            for location in locations
        ),
        tuple(
            int(location is not None and getattr(location, attribute) is not None)
            for location in locations
        ),
    )


def _decode_optional_identity(
    value: int,
    validity: int,
    *,
    label: str,
) -> int | None:
    if validity not in {0, 1}:
        raise ResultVtkDecodeError(f"{label} validity must be zero or one")
    if not validity:
        if value != 0:
            raise ResultVtkDecodeError(
                f"missing {label} must use a canonical zero placeholder"
            )
        return None
    if value <= 0:
        raise ResultVtkDecodeError(f"present {label} must be positive")
    return value


def _decode_optional_index(
    value: int,
    validity: int,
    *,
    count: int,
) -> int | None:
    if validity not in {0, 1}:
        raise ResultVtkDecodeError("region_index validity must be zero or one")
    if not validity:
        if value != 0:
            raise ResultVtkDecodeError(
                "missing region_index must use a canonical zero placeholder"
            )
        return None
    if value < 0 or value >= count:
        raise ResultVtkDecodeError(
            "present region_index is outside the canonical region table"
        )
    return value


def _expect_line(
    reader: _LineReader,
    expected: str,
    *,
    label: str,
) -> None:
    if reader.take(label) != expected:
        raise ResultVtkDecodeError(f"result VTK {label} is invalid")


def _format_float(value: Real) -> str:
    numeric = _finite_number(value, label="VTK numeric value")
    return format(numeric, ".17g")


def _parse_float(value: str, *, label: str) -> float:
    if not value:
        raise ResultVtkDecodeError(f"{label} must not be empty")
    try:
        numeric = float(value)
    except ValueError as error:
        raise ResultVtkDecodeError(
            f"{label} must be a finite canonical number"
        ) from error
    if not math.isfinite(numeric):
        raise ResultVtkDecodeError(f"{label} must be finite")
    if _format_float(numeric) != value:
        raise ResultVtkDecodeError(f"{label} must use canonical numeric text")
    return numeric


def _parse_integer(
    value: str,
    *,
    label: str,
    minimum: int,
    maximum: int | None = None,
) -> int:
    if not value or not value.isascii() or not value.isdecimal():
        raise ResultVtkDecodeError(f"{label} must be a canonical integer")
    numeric = int(value)
    if str(numeric) != value:
        raise ResultVtkDecodeError(f"{label} must be a canonical integer")
    if numeric < minimum:
        raise ResultVtkDecodeError(f"{label} must be at least {minimum}")
    if maximum is not None and numeric > maximum:
        raise ResultVtkDecodeError(f"{label} must not exceed {maximum}")
    return numeric


def _finite_number(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{label} must be a real number")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{label} must be finite")
    return numeric


def _points_tuple(
    value: object,
) -> tuple[tuple[float, float, float], ...]:
    if type(value) is not tuple:
        raise TypeError("points must be a tuple")
    points = []
    for point in value:
        if type(point) is not tuple or len(point) != 3:
            raise TypeError("points must contain three-component tuples")
        points.append(
            tuple(
                _finite_number(component, label="point component")
                for component in point
            )
        )
    return tuple(points)


def _cells_tuple(
    value: object,
    *,
    point_count: int,
) -> tuple[tuple[int, ...], ...]:
    if type(value) is not tuple:
        raise TypeError("cells must be a tuple")
    for cell in value:
        if type(cell) is not tuple or not cell:
            raise TypeError("cells must contain non-empty tuples")
        if len(set(cell)) != len(cell):
            raise ValueError("cell connectivity must not repeat point indexes")
        for point_index in cell:
            if type(point_index) is not int:
                raise TypeError("cell point indexes must be integers")
            if point_index < 0 or point_index >= point_count:
                raise ValueError("cell connectivity references an unknown point index")
    return value


def _cell_type_tuple(
    value: object,
    cells: tuple[tuple[int, ...], ...],
) -> tuple[int, ...]:
    if type(value) is not tuple:
        raise TypeError("cell_types must be a tuple")
    if len(value) != len(cells):
        raise ValueError("cell_types length must match cells")
    for cell, cell_type in zip(cells, value, strict=True):
        if type(cell_type) is not int:
            raise TypeError("cell_types must contain integers")
        expected = _VTK_CELL_POINT_COUNTS.get(cell_type)
        if expected is None:
            raise ValueError("cell_types contains an unsupported VTK type")
        if len(cell) != expected:
            raise ValueError("cell connectivity does not match its VTK cell type")
    return value


def _finite_value_tuple(value: object) -> tuple[float, ...]:
    if type(value) is not tuple:
        raise TypeError("values must be a tuple")
    return tuple(_finite_number(item, label="scalar value") for item in value)


def _validate_location_tuple(
    value: object,
    *,
    length: int,
    label: str,
    association: FieldAssociation,
) -> None:
    if type(value) is not tuple:
        raise TypeError(f"{label} must be a tuple")
    if len(value) != length:
        raise ValueError(f"{label} length must match its topology")
    for identity in value:
        if identity is None:
            continue
        if type(identity) is not ResultVtkLocationIdentity:
            raise TypeError(f"{label} must contain location identities or None")
        try:
            FieldLocation(
                association=association,
                coordinates=(0.0, 0.0, 0.0),
                displacement=None,
                node_id=identity.node_id,
                element_id=identity.element_id,
                integration_point=identity.integration_point,
                local_node=identity.local_node,
                region_key=identity.region_key,
                averaged=identity.averaged,
                section_point=identity.section_point,
            )
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"{label} contains an invalid {association.value} identity: {error}"
            ) from error


def _validate_region_table(
    value: object,
    point_locations: tuple[ResultVtkLocationIdentity | None, ...],
    cell_locations: tuple[ResultVtkLocationIdentity | None, ...],
) -> None:
    if type(value) is not tuple:
        raise TypeError("region_table must be a tuple")
    if any(type(region) is not ResultRegionKey for region in value):
        raise TypeError("region_table must contain only ResultRegionKey values")
    encoded = tuple(encode_result_region_key(region) for region in value)
    if tuple(sorted(set(encoded))) != encoded:
        raise ValueError("region_table must use unique canonical sorted identities")
    referenced = {
        identity.region_key
        for identity in point_locations + cell_locations
        if identity is not None and identity.region_key is not None
    }
    if referenced != set(value):
        raise ValueError("region_table must exactly equal referenced location regions")


def _build_region_table(
    point_locations: tuple[ResultVtkLocationIdentity | None, ...],
    cell_locations: tuple[ResultVtkLocationIdentity | None, ...],
) -> tuple[ResultRegionKey, ...]:
    by_text = {
        encode_result_region_key(identity.region_key): identity.region_key
        for identity in point_locations + cell_locations
        if identity is not None and identity.region_key is not None
    }
    return tuple(by_text[text] for text in sorted(by_text))


def _require_element_type(value: str | None) -> str:
    if type(value) is not str or not value:
        raise ResultVtkEncodeError(
            "FEM-element VTK cells require a canonical element type"
        )
    return value


def _semantic_identity(value: ResultVtkReadback) -> ResultVtkReadback:
    return value


__all__ = [
    "RESULT_VTK_FORMAT_NAME",
    "RESULT_VTK_SCHEMA_VERSION",
    "RESULT_VTK_TITLE",
    "ResultVtkDecodeError",
    "ResultVtkEmptySelectionError",
    "ResultVtkEncodeError",
    "ResultVtkError",
    "ResultVtkLocationIdentity",
    "ResultVtkReadback",
    "dumps_result_vtk",
    "read_result_vtk",
    "write_result_vtk",
]
