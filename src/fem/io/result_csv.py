"""Canonical, snapshot-bound scalar result CSV interchange."""

from __future__ import annotations

from collections.abc import Callable
import csv
from dataclasses import dataclass
import io
import math
from numbers import Real
from pathlib import Path
from typing import Any

from fem.application.results.data import (
    FieldLocation,
    ResultExportSnapshot,
    ResultMaterializationSnapshot,
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
from fem.application.results.query import (
    ResultQueryResult,
    evaluate_result_query,
)
from fem.post.averaging import NodalAveragingPolicy
from fem.post.fields import (
    decode_result_region_key,
    encode_result_region_key,
)

from ._atomic_text import atomic_write_verified_text


RESULT_CSV_FORMAT_NAME = "fem-python-result-field"
RESULT_CSV_SCHEMA_VERSION = 1
RESULT_CSV_HEADER = (
    "format",
    "schema",
    "result_id",
    "session_id",
    "artifact_id",
    "model_revision",
    "run_id",
    "step",
    "materialization_generation",
    "field_variable",
    "field_position",
    "component",
    "averaging_threshold_percent",
    "averaging_preserve_region_boundaries",
    "gauss_order",
    "recovery_contract",
    "quantity",
    "association",
    "node_id",
    "element_id",
    "integration_point",
    "local_node",
    "region",
    "averaged",
    "x",
    "y",
    "z",
    "value",
    "unit",
)

_UTF8_BOM = b"\xef\xbb\xbf"
_EXPECTED_ASSOCIATION = {
    FieldPosition.NODE: FieldAssociation.NODE,
    FieldPosition.INTEGRATION_POINT: FieldAssociation.INTEGRATION_POINT,
    FieldPosition.CENTROID: FieldAssociation.ELEMENT,
    FieldPosition.ELEMENT_NODAL: FieldAssociation.ELEMENT_NODE,
    FieldPosition.NODE_REGION: FieldAssociation.NODE_REGION,
    FieldPosition.RESOLVED_NODAL: FieldAssociation.RESOLVED_NODAL,
    FieldPosition.SECTION_END: FieldAssociation.ELEMENT_NODE,
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


class ResultCsvError(ValueError):
    """Base error for canonical result CSV validation or serialization."""


class ResultCsvEncodeError(ResultCsvError):
    """The supplied snapshot or query cannot be encoded canonically."""


class ResultCsvDecodeError(ResultCsvError):
    """The input is not a canonical result CSV document."""


class ResultCsvEmptySelectionError(ResultCsvError):
    """The requested full field or exact query contains no scalar rows."""


@dataclass(frozen=True, slots=True)
class ResultCsvRecord:
    """One decoded scalar value at a canonical result location."""

    location: FieldLocation
    value: float

    def __post_init__(self) -> None:
        if type(self.location) is not FieldLocation:
            raise TypeError("location must be FieldLocation")
        value = self.value
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError("value must be a real number")
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError("value must be finite")
        object.__setattr__(self, "value", numeric)


@dataclass(frozen=True, slots=True)
class ResultCsvReadback:
    """Complete semantic projection reconstructed from canonical CSV."""

    source: ResultSourceKey
    materialization_generation: int
    selection: ScalarFieldSelection
    quantity: PhysicalQuantity
    association: FieldAssociation
    unit_label: str | None
    records: tuple[ResultCsvRecord, ...]

    def __post_init__(self) -> None:
        if type(self.source) is not ResultSourceKey:
            raise TypeError("source must be ResultSourceKey")
        if type(self.materialization_generation) is not int:
            raise TypeError("materialization_generation must be an integer")
        if self.materialization_generation < 0:
            raise ValueError(
                "materialization_generation must be non-negative"
            )
        if type(self.selection) is not ScalarFieldSelection:
            raise TypeError("selection must be ScalarFieldSelection")
        if type(self.quantity) is not PhysicalQuantity:
            raise TypeError("quantity must be PhysicalQuantity")
        if type(self.association) is not FieldAssociation:
            raise TypeError("association must be FieldAssociation")
        if self.unit_label is not None:
            _require_nonblank_string(self.unit_label, label="unit_label")
        if type(self.records) is not tuple:
            raise TypeError("records must be a tuple")
        if not self.records:
            raise ValueError("records must not be empty")

        field_id = self.selection.field_key.request.field_id
        if self.association is not _EXPECTED_ASSOCIATION[field_id.position]:
            raise ValueError(
                "association does not match the field position"
            )
        if self.quantity is not _EXPECTED_QUANTITY[field_id.variable]:
            raise ValueError(
                "quantity does not match the field variable"
            )

        identities: set[tuple[object, ...]] = set()
        for record in self.records:
            if type(record) is not ResultCsvRecord:
                raise TypeError(
                    "records must contain only ResultCsvRecord values"
                )
            if record.location.association is not self.association:
                raise ValueError(
                    "record association must match CSV association"
                )
            identity = _location_identity(record.location)
            if identity in identities:
                raise ValueError(
                    "result CSV records must use unique locations"
                )
            identities.add(identity)


@dataclass(frozen=True, slots=True)
class _ResultCsvMetadata:
    source: ResultSourceKey
    materialization_generation: int
    selection: ScalarFieldSelection
    quantity: PhysicalQuantity
    association: FieldAssociation
    unit_label: str | None


def dumps_result_csv(
    snapshot: ResultExportSnapshot,
    query_result: ResultQueryResult | None = None,
) -> str:
    """Serialize a complete field or exact query as deterministic CSV."""

    projected = _project_result_csv(snapshot, query_result)
    return _serialize_result_csv(projected)


def read_result_csv(path: str | Path) -> ResultCsvReadback:
    """Read and strictly validate one canonical result CSV file."""

    raw = Path(path).read_bytes()
    if not raw.startswith(_UTF8_BOM):
        raise ResultCsvDecodeError("result CSV must start with a UTF-8 BOM")
    try:
        text = raw.decode("utf-8-sig", errors="strict")
    except UnicodeDecodeError as error:
        raise ResultCsvDecodeError(
            "result CSV must contain valid UTF-8"
        ) from error
    if "\r" in text:
        raise ResultCsvDecodeError("result CSV must use LF line endings")
    if not text.endswith("\n"):
        raise ResultCsvDecodeError(
            "result CSV must end with a final newline"
        )

    try:
        rows = list(
            csv.reader(
                io.StringIO(text, newline=""),
                strict=True,
            )
        )
    except csv.Error as error:
        raise ResultCsvDecodeError("result CSV syntax is invalid") from error
    if not rows:
        raise ResultCsvDecodeError("result CSV header is missing")
    if tuple(rows[0]) != RESULT_CSV_HEADER:
        raise ResultCsvDecodeError("result CSV header does not match schema 1")
    if len(rows) == 1:
        raise ResultCsvEmptySelectionError(
            "result CSV must contain at least one scalar row"
        )

    metadata: _ResultCsvMetadata | None = None
    records: list[ResultCsvRecord] = []
    for row_number, row in enumerate(rows[1:], start=2):
        if len(row) != len(RESULT_CSV_HEADER):
            raise ResultCsvDecodeError(
                f"result CSV row {row_number} has the wrong column count"
            )
        values = dict(zip(RESULT_CSV_HEADER, row, strict=True))
        try:
            row_metadata, record = _decode_row(values)
        except ResultCsvError:
            raise
        except (KeyError, TypeError, ValueError) as error:
            raise ResultCsvDecodeError(
                f"result CSV row {row_number} is invalid: {error}"
            ) from error
        if metadata is None:
            metadata = row_metadata
        elif row_metadata != metadata:
            raise ResultCsvDecodeError(
                f"result CSV row {row_number} changes field metadata"
            )
        records.append(record)

    assert metadata is not None
    try:
        return ResultCsvReadback(
            source=metadata.source,
            materialization_generation=(
                metadata.materialization_generation
            ),
            selection=metadata.selection,
            quantity=metadata.quantity,
            association=metadata.association,
            unit_label=metadata.unit_label,
            records=tuple(records),
        )
    except (TypeError, ValueError) as error:
        raise ResultCsvDecodeError(
            f"result CSV semantics are invalid: {error}"
        ) from error


def write_result_csv(
    path: str | Path,
    snapshot: ResultExportSnapshot,
    query_result: ResultQueryResult | None = None,
    *,
    before_replace: Callable[[], Any] | None = None,
) -> Path:
    """Durably verify and atomically install one canonical result CSV."""

    projected = _project_result_csv(snapshot, query_result)
    serialized = _serialize_result_csv(projected)
    return atomic_write_verified_text(
        path,
        serialized,
        verifier=read_result_csv,
        semantic_encoder=_semantic_identity,
        expected_semantic=projected,
        error_type=ResultCsvError,
        mismatch_message=(
            "temporary result CSV semantic verification failed"
        ),
        before_replace=before_replace,
    )


def _project_result_csv(
    snapshot: ResultExportSnapshot,
    query_result: ResultQueryResult | None,
) -> ResultCsvReadback:
    if type(snapshot) is not ResultExportSnapshot:
        raise TypeError("snapshot must be ResultExportSnapshot")
    if query_result is not None and type(query_result) is not ResultQueryResult:
        raise TypeError("query_result must be ResultQueryResult or None")

    field_data = snapshot.field
    try:
        component_index = field_data.descriptor.columns.index(
            snapshot.selection.component
        )
    except ValueError as error:
        raise ResultCsvEncodeError(
            "snapshot selection component is not in the field descriptor"
        ) from error

    if query_result is None:
        values = field_data.values
        source_rows = tuple(
            (location, float(values[index, component_index]))
            for index, location in enumerate(field_data.locations)
        )
    else:
        _validate_query_result(snapshot, query_result)
        source_rows = tuple(
            (record.location, record.value)
            for record in query_result.records
        )
    if not source_rows:
        raise ResultCsvEmptySelectionError(
            "result CSV export requires at least one scalar row"
        )

    records = tuple(
        ResultCsvRecord(
            location=_csv_location(location),
            value=value,
        )
        for location, value in source_rows
    )
    descriptor = field_data.descriptor
    try:
        return ResultCsvReadback(
            source=snapshot.source,
            materialization_generation=(
                snapshot.materialization_generation
            ),
            selection=snapshot.selection,
            quantity=descriptor.quantity,
            association=descriptor.association,
            unit_label=descriptor.unit_label,
            records=records,
        )
    except (TypeError, ValueError) as error:
        raise ResultCsvEncodeError(
            f"snapshot cannot be represented as result CSV: {error}"
        ) from error


def _validate_query_result(
    snapshot: ResultExportSnapshot,
    query_result: ResultQueryResult,
) -> None:
    if query_result.source != snapshot.source:
        raise ResultCsvEncodeError(
            "query result source must exactly match the export snapshot"
        )
    if (
        query_result.materialization_generation
        != snapshot.materialization_generation
    ):
        raise ResultCsvEncodeError(
            "query result generation must exactly match the export snapshot"
        )
    if query_result.query.field_key != snapshot.selection.field_key:
        raise ResultCsvEncodeError(
            "query field key must exactly match the export selection"
        )
    if query_result.query.component != snapshot.selection.component:
        raise ResultCsvEncodeError(
            "query component must exactly match the export selection"
        )

    materialization = ResultMaterializationSnapshot(
        source=snapshot.source,
        generation=snapshot.materialization_generation,
        topology=snapshot.topology,
        fields=(snapshot.field,),
    )
    try:
        expected = evaluate_result_query(
            materialization,
            query_result.query,
        )
    except (TypeError, ValueError) as error:
        raise ResultCsvEncodeError(
            f"query result cannot be validated: {error}"
        ) from error
    if query_result != expected:
        raise ResultCsvEncodeError(
            "query records must be the exact ordered snapshot subset"
        )


def _serialize_result_csv(projected: ResultCsvReadback) -> str:
    output = io.StringIO(newline="")
    output.write("\ufeff")
    writer = csv.writer(
        output,
        lineterminator="\n",
        quoting=csv.QUOTE_MINIMAL,
    )
    writer.writerow(RESULT_CSV_HEADER)

    metadata = _encode_metadata(projected)
    for record in projected.records:
        location = record.location
        row = {
            **metadata,
            "node_id": _format_optional_int(location.node_id),
            "element_id": _format_optional_int(location.element_id),
            "integration_point": _format_optional_int(
                location.integration_point
            ),
            "local_node": _format_optional_int(location.local_node),
            "region": (
                ""
                if location.region_key is None
                else encode_result_region_key(location.region_key)
            ),
            "averaged": _format_optional_bool(location.averaged),
            "x": _format_float(location.coordinates[0]),
            "y": _format_float(location.coordinates[1]),
            "z": _format_float(location.coordinates[2]),
            "value": _format_float(record.value),
        }
        writer.writerow(tuple(row[name] for name in RESULT_CSV_HEADER))
    serialized = output.getvalue()
    if "\r" in serialized:
        raise ResultCsvEncodeError(
            "result CSV text values must not contain carriage returns"
        )
    return serialized


def _encode_metadata(projected: ResultCsvReadback) -> dict[str, str]:
    source = projected.source
    selection = projected.selection
    key = selection.field_key
    request = key.request
    field_id = request.field_id
    policy = request.averaging_policy
    for label, value in (
        ("result_id", source.result_id),
        ("session_id", source.session_id),
        ("artifact_id", source.artifact_id),
        ("run_id", source.run_id),
        ("step", source.step_name),
        ("component", selection.component),
    ):
        _reject_carriage_return(value, label=label)
    if projected.unit_label is not None:
        _reject_carriage_return(projected.unit_label, label="unit")

    return {
        "format": RESULT_CSV_FORMAT_NAME,
        "schema": str(RESULT_CSV_SCHEMA_VERSION),
        "result_id": source.result_id,
        "session_id": source.session_id,
        "artifact_id": source.artifact_id,
        "model_revision": str(source.model_revision),
        "run_id": source.run_id,
        "step": source.step_name,
        "materialization_generation": str(
            projected.materialization_generation
        ),
        "field_variable": field_id.variable.value,
        "field_position": field_id.position.value,
        "component": selection.component,
        "averaging_threshold_percent": (
            "" if policy is None else _format_float(policy.threshold_percent)
        ),
        "averaging_preserve_region_boundaries": (
            ""
            if policy is None
            else _format_bool(policy.preserve_region_boundaries)
        ),
        "gauss_order": _format_optional_int(request.gauss_order),
        "recovery_contract": str(key.recovery_contract),
        "quantity": projected.quantity.value,
        "association": projected.association.value,
        "unit": "" if projected.unit_label is None else projected.unit_label,
    }


def _decode_row(
    values: dict[str, str],
) -> tuple[_ResultCsvMetadata, ResultCsvRecord]:
    if values["format"] != RESULT_CSV_FORMAT_NAME:
        raise ValueError("format must be fem-python-result-field")
    if values["schema"] != str(RESULT_CSV_SCHEMA_VERSION):
        raise ValueError("schema must be 1")

    source = ResultSourceKey(
        result_id=values["result_id"],
        session_id=values["session_id"],
        artifact_id=values["artifact_id"],
        model_revision=_parse_integer(
            values["model_revision"],
            label="model_revision",
            minimum=0,
        ),
        step_name=values["step"],
        run_id=values["run_id"],
    )
    field_id = ResultFieldId(
        variable=ResultVariable(values["field_variable"]),
        position=FieldPosition(values["field_position"]),
    )
    threshold_text = values["averaging_threshold_percent"]
    preserve_text = values["averaging_preserve_region_boundaries"]
    if bool(threshold_text) != bool(preserve_text):
        raise ValueError(
            "averaging policy columns must both be empty or both be present"
        )
    policy = (
        None
        if not threshold_text
        else NodalAveragingPolicy(
            threshold_percent=_parse_float(
                threshold_text,
                label="averaging_threshold_percent",
            ),
            preserve_region_boundaries=_parse_bool(
                preserve_text,
                label="averaging_preserve_region_boundaries",
            ),
        )
    )
    request = FieldRequest(
        field_id=field_id,
        averaging_policy=policy,
        gauss_order=_parse_optional_integer(
            values["gauss_order"],
            label="gauss_order",
            minimum=1,
        ),
    )
    key = FieldMaterializationKey(
        request=request,
        recovery_contract=_parse_integer(
            values["recovery_contract"],
            label="recovery_contract",
            minimum=1,
        ),
    )
    selection = ScalarFieldSelection(
        field_key=key,
        component=values["component"],
    )
    association = FieldAssociation(values["association"])
    quantity = PhysicalQuantity(values["quantity"])
    unit_label = values["unit"] or None
    region_text = values["region"]
    region_key = (
        None
        if not region_text
        else decode_result_region_key(region_text)
    )
    if region_key is not None:
        if encode_result_region_key(region_key) != region_text:
            raise ValueError("region must use the canonical region codec")

    location = FieldLocation(
        association=association,
        coordinates=(
            _parse_float(values["x"], label="x"),
            _parse_float(values["y"], label="y"),
            _parse_float(values["z"], label="z"),
        ),
        displacement=None,
        node_id=_parse_optional_integer(
            values["node_id"],
            label="node_id",
            minimum=1,
        ),
        element_id=_parse_optional_integer(
            values["element_id"],
            label="element_id",
            minimum=1,
        ),
        integration_point=_parse_optional_integer(
            values["integration_point"],
            label="integration_point",
            minimum=1,
        ),
        local_node=_parse_optional_integer(
            values["local_node"],
            label="local_node",
            minimum=1,
        ),
        region_key=region_key,
        averaged=_parse_optional_bool(
            values["averaged"],
            label="averaged",
        ),
    )
    metadata = _ResultCsvMetadata(
        source=source,
        materialization_generation=_parse_integer(
            values["materialization_generation"],
            label="materialization_generation",
            minimum=0,
        ),
        selection=selection,
        quantity=quantity,
        association=association,
        unit_label=unit_label,
    )
    return metadata, ResultCsvRecord(
        location=location,
        value=_parse_float(values["value"], label="value"),
    )


def _csv_location(location: FieldLocation) -> FieldLocation:
    return FieldLocation(
        association=location.association,
        coordinates=location.coordinates,
        displacement=None,
        node_id=location.node_id,
        element_id=location.element_id,
        integration_point=location.integration_point,
        local_node=location.local_node,
        region_key=location.region_key,
        averaged=location.averaged,
    )


def _location_identity(location: FieldLocation) -> tuple[object, ...]:
    association = location.association
    if association is FieldAssociation.NODE:
        return association, location.node_id
    if association is FieldAssociation.ELEMENT:
        return association, location.element_id
    if association is FieldAssociation.INTEGRATION_POINT:
        return association, location.element_id, location.integration_point
    if association is FieldAssociation.ELEMENT_NODE:
        return (
            association,
            location.element_id,
            location.local_node,
            location.node_id,
        )
    if association is FieldAssociation.NODE_REGION:
        return association, location.node_id, location.region_key
    if location.averaged:
        return association, location.node_id, location.region_key, True
    return (
        association,
        location.node_id,
        location.region_key,
        False,
        location.element_id,
        location.local_node,
    )


def _format_float(value: float) -> str:
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ResultCsvEncodeError(
            "result CSV numeric values must be finite"
        )
    return format(numeric, ".17g")


def _parse_float(value: str, *, label: str) -> float:
    if not value:
        raise ValueError(f"{label} must not be empty")
    try:
        numeric = float(value)
    except ValueError as error:
        raise ValueError(f"{label} must be a finite number") from error
    if not math.isfinite(numeric):
        raise ValueError(f"{label} must be finite")
    if _format_float(numeric) != value:
        raise ValueError(f"{label} must use canonical numeric text")
    return numeric


def _format_optional_int(value: int | None) -> str:
    return "" if value is None else str(value)


def _parse_integer(value: str, *, label: str, minimum: int) -> int:
    if not value or not value.isascii() or not value.isdecimal():
        raise ValueError(f"{label} must be a canonical integer")
    numeric = int(value)
    if value != str(numeric):
        raise ValueError(f"{label} must be a canonical integer")
    if numeric < minimum:
        raise ValueError(f"{label} must be at least {minimum}")
    return numeric


def _parse_optional_integer(
    value: str,
    *,
    label: str,
    minimum: int,
) -> int | None:
    if not value:
        return None
    return _parse_integer(value, label=label, minimum=minimum)


def _format_bool(value: bool) -> str:
    return "true" if value else "false"


def _format_optional_bool(value: bool | None) -> str:
    return "" if value is None else _format_bool(value)


def _parse_bool(value: str, *, label: str) -> bool:
    if value == "true":
        return True
    if value == "false":
        return False
    raise ValueError(f"{label} must be lowercase true or false")


def _parse_optional_bool(value: str, *, label: str) -> bool | None:
    return None if not value else _parse_bool(value, label=label)


def _require_nonblank_string(value: object, *, label: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{label} must be a string")
    if not value.strip():
        raise ValueError(f"{label} must not be blank")
    return value


def _reject_carriage_return(value: str, *, label: str) -> None:
    if "\r" in value:
        raise ResultCsvEncodeError(
            f"{label} must not contain carriage returns"
        )


def _semantic_identity(value: ResultCsvReadback) -> ResultCsvReadback:
    return value


__all__ = [
    "RESULT_CSV_FORMAT_NAME",
    "RESULT_CSV_HEADER",
    "RESULT_CSV_SCHEMA_VERSION",
    "ResultCsvDecodeError",
    "ResultCsvEmptySelectionError",
    "ResultCsvEncodeError",
    "ResultCsvError",
    "ResultCsvReadback",
    "ResultCsvRecord",
    "dumps_result_csv",
    "read_result_csv",
    "write_result_csv",
]
