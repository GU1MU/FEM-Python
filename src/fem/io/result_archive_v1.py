"""Strict ZIP/NumPy schema-v1 codec for FEM-Python result archives."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from datetime import datetime, timezone
from io import BytesIO
import hashlib
import json
import math
from pathlib import Path
import re
import zipfile
from typing import Any

import numpy as np

from fem.elements.beam_section import BeamSectionPoint

from fem.application.results.archive import (
    LoadedResultArchive,
    ResultArchiveModelProjection,
    ResultArchiveOrigin,
    ResultArchiveRun,
    ResultArchiveSnapshot,
    archive_region_dictionary,
)
from fem.application.results.data import (
    FieldAvailability,
    FieldData,
    FieldDescriptor,
    FieldLocation,
    FieldState,
    ResultCatalog,
    ResultDiagnostic,
    ResultMaterializationSnapshot,
    ResultTopologyProjection,
)
from fem.application.results.execution import (
    OutputExecutionStatus,
    OutputRequestExecution,
    OutputVariableExecution,
    ResultExecutionReport,
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
from fem.application.results.output_requests import (
    ExecutableOutputRequest,
    OutputVariableProjection,
)
from fem.application.results.registry import (
    ElementResultProfile,
    ResultModelFamily,
    registry_entry_for,
)
from fem.application.units import UnitContext
from fem.io._atomic_binary import atomic_write_verified_binary
from fem.io._result_archive_errors import (
    ResultArchiveDecodeError,
    ResultArchiveEncodeError,
    UnsupportedResultArchiveSchemaError,
)
from fem.post.averaging import NodalAveragingPolicy
from fem.post.fields import decode_result_region_key, encode_result_region_key


FORMAT_NAME = "fem-python-result"
SCHEMA_VERSION = 1
RESULT_ARCHIVE_FORMAT_NAME = FORMAT_NAME
RESULT_ARCHIVE_SCHEMA_VERSION = SCHEMA_VERSION
RESULT_FILE_SUFFIX = ".femres"
RESULT_ARCHIVE_FILE_SUFFIX = RESULT_FILE_SUFFIX
MANIFEST_NAME = "manifest.json"

# Archive inputs are untrusted.  Keep these limits deliberately generous for
# ordinary solver output while bounding decompression and NumPy metadata work
# before any array is materialized.  The limits are part of the decoder's
# safety policy, not the schema's numerical contract.
_MAX_ZIP_ENTRY_BYTES = 512 * 1024 * 1024
_MAX_ZIP_TOTAL_BYTES = 1024 * 1024 * 1024
_MAX_ZIP_COMPRESSION_RATIO = 10_000
_MAX_ZIP_CONTAINER_BYTES = 1024 * 1024 * 1024
_MAX_ZIP_ENTRY_COUNT = 4096
_MAX_MANIFEST_BYTES = 8 * 1024 * 1024
_MAX_ARRAY_BYTES = 512 * 1024 * 1024

_TOP_LEVEL_KEYS = frozenset(
    {
        "format",
        "schema",
        "archive_id",
        "created_at",
        "producer_version",
        "source",
        "origin",
        "run",
        "unit_context",
        "profile",
        "catalog",
        "materialization_generation",
        "model_projection",
        "inspection_summaries",
        "topology",
        "fields",
        "arrays",
    }
)
_FIELD_ID_RE = re.compile(r"^[a-z0-9_]+-[a-f0-9]{24}$")
_TOPOLOGY_ARRAY_ORDER = (
    "node_ids",
    "node_coordinates",
    "nodal_displacements",
    "element_ids",
    "connectivity_offsets",
    "connectivity",
    "element_type_indices",
    "region_indices",
)
_FIELD_ARRAY_ORDER = (
    "values",
    "coordinates",
    "displacement",
    "displacement_mask",
    "node_id",
    "element_id",
    "integration_point",
    "local_node",
    "region_index",
    "averaged",
    "averaged_mask",
    "section_point_number",
    "section_point_local_y",
    "section_point_local_z",
)


def _json_plain(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _json_plain(item) for key, item in value.items()}
    if type(value) in {tuple, list}:
        return [_json_plain(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"unsupported JSON value {type(value).__name__}")


def _canonical_json(value: object) -> bytes:
    try:
        return json.dumps(
            _json_plain(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8", errors="strict")
    except (RecursionError, TypeError, ValueError, UnicodeError) as error:
        raise ResultArchiveEncodeError(f"manifest JSON encoding failed: {error}") from error


def _reject_json_constant(value: str) -> Any:
    raise ResultArchiveDecodeError(f"manifest contains non-finite JSON constant {value!r}")


def _reject_duplicate_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ResultArchiveDecodeError(f"manifest contains duplicate key {key!r}")
        result[key] = value
    return result


def _parse_manifest(data: bytes) -> dict[str, Any]:
    try:
        text = data.decode("utf-8", errors="strict")
        payload = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_object,
            parse_constant=_reject_json_constant,
        )
    except ResultArchiveDecodeError:
        raise
    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
        OverflowError,
        RecursionError,
        TypeError,
        ValueError,
    ) as error:
        raise ResultArchiveDecodeError(f"manifest is not strict UTF-8 JSON: {error}") from error
    if type(payload) is not dict:
        raise ResultArchiveDecodeError("manifest root must be a JSON object")
    _exact_keys(payload, _TOP_LEVEL_KEYS, label="manifest")
    return payload


def _parse_manifest_header(data: bytes) -> tuple[str, int]:
    """Read only the version-neutral fields required for codec dispatch."""

    try:
        text = data.decode("utf-8", errors="strict")
        payload = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_object,
            parse_constant=_reject_json_constant,
        )
    except ResultArchiveDecodeError:
        raise
    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
        OverflowError,
        RecursionError,
        TypeError,
        ValueError,
    ) as error:
        raise ResultArchiveDecodeError(
            f"manifest header is not strict UTF-8 JSON: {error}"
        ) from error
    if type(payload) is not dict:
        raise ResultArchiveDecodeError("manifest root must be a JSON object")
    format_name = payload.get("format")
    schema = payload.get("schema")
    if type(format_name) is not str or not format_name.strip():
        raise ResultArchiveDecodeError("manifest.format must be a nonblank string")
    if type(schema) is not int:
        raise ResultArchiveDecodeError("manifest.schema must be a strict integer")
    return format_name, schema


def _exact_keys(value: object, expected: set[str] | frozenset[str], *, label: str) -> None:
    if type(value) is not dict:
        raise ResultArchiveDecodeError(f"{label} must be an object")
    actual = frozenset(value)
    if actual != frozenset(expected):
        missing = sorted(frozenset(expected) - actual)
        unknown = sorted(actual - frozenset(expected))
        raise ResultArchiveDecodeError(
            f"{label} fields do not match schema (missing={missing}, unknown={unknown})"
        )


def _strict_string(value: object, *, label: str) -> str:
    if type(value) is not str or not value.strip():
        raise ResultArchiveDecodeError(f"{label} must be a nonblank string")
    return value


def _strict_bool(value: object, *, label: str) -> bool:
    if type(value) is not bool:
        raise ResultArchiveDecodeError(f"{label} must be boolean")
    return value


def _strict_int(value: object, *, label: str, minimum: int | None = None) -> int:
    if type(value) is not int:
        raise ResultArchiveDecodeError(f"{label} must be a strict integer")
    if minimum is not None and value < minimum:
        raise ResultArchiveDecodeError(f"{label} must be >= {minimum}")
    return value


def _strict_float(value: object, *, label: str, minimum: float | None = None) -> float:
    if type(value) not in {int, float}:
        raise ResultArchiveDecodeError(f"{label} must be a real number")
    try:
        result = float(value)
    except (OverflowError, ValueError) as error:
        raise ResultArchiveDecodeError(f"{label} must be a finite real number") from error
    if not np.isfinite(result):
        raise ResultArchiveDecodeError(f"{label} must be finite")
    if minimum is not None and result < minimum:
        raise ResultArchiveDecodeError(f"{label} must be >= {minimum}")
    return result


def _datetime_to_json(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _datetime_from_json(value: object, *, label: str) -> datetime:
    text = _strict_string(value, label=label)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as error:
        raise ResultArchiveDecodeError(f"{label} is not an ISO-8601 datetime") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ResultArchiveDecodeError(f"{label} must be timezone-aware")
    return parsed.astimezone(timezone.utc)


def _source_to_json(source: ResultSourceKey) -> dict[str, object]:
    return {
        "result_id": source.result_id,
        "session_id": source.session_id,
        "artifact_id": source.artifact_id,
        "model_revision": source.model_revision,
        "step_name": source.step_name,
        "run_id": source.run_id,
    }


def _source_from_json(value: object, *, label: str) -> ResultSourceKey:
    if type(value) is not dict:
        raise ResultArchiveDecodeError(f"{label} must be an object")
    _exact_keys(value, {"result_id", "session_id", "artifact_id", "model_revision", "step_name", "run_id"}, label=label)
    try:
        return ResultSourceKey(
            result_id=_strict_string(value["result_id"], label=f"{label}.result_id"),
            session_id=_strict_string(value["session_id"], label=f"{label}.session_id"),
            artifact_id=_strict_string(value["artifact_id"], label=f"{label}.artifact_id"),
            model_revision=_strict_int(value["model_revision"], label=f"{label}.model_revision", minimum=0),
            step_name=_strict_string(value["step_name"], label=f"{label}.step_name"),
            run_id=_strict_string(value["run_id"], label=f"{label}.run_id"),
        )
    except (TypeError, ValueError) as error:
        raise ResultArchiveDecodeError(f"invalid {label}: {error}") from error


def _field_id_to_json(field_id: ResultFieldId) -> dict[str, object]:
    result: dict[str, object] = {
        "variable": field_id.variable.value,
        "position": field_id.position.value,
    }
    if field_id.section_point_number is not None:
        result["section_point_number"] = field_id.section_point_number
    return result


def _field_id_from_json(value: object, *, label: str) -> ResultFieldId:
    if type(value) is not dict:
        raise ResultArchiveDecodeError(f"{label} must be an object")
    keys = set(value)
    if keys not in (
        {"variable", "position"},
        {"variable", "position", "section_point_number"},
    ):
        raise ResultArchiveDecodeError(f"{label} has unexpected object keys")
    point_number = value.get("section_point_number")
    if point_number is not None:
        point_number = _strict_int(
            point_number,
            label=f"{label}.section_point_number",
            minimum=1,
        )
    try:
        return ResultFieldId(
            ResultVariable(_strict_string(value["variable"], label=f"{label}.variable")),
            FieldPosition(_strict_string(value["position"], label=f"{label}.position")),
            section_point_number=point_number,
        )
    except (TypeError, ValueError) as error:
        raise ResultArchiveDecodeError(f"invalid {label}: {error}") from error


def _policy_to_json(policy: NodalAveragingPolicy | None) -> dict[str, object] | None:
    if policy is None:
        return None
    return {
        "threshold_percent": policy.threshold_percent,
        "preserve_region_boundaries": policy.preserve_region_boundaries,
    }


def _policy_from_json(value: object, *, label: str) -> NodalAveragingPolicy | None:
    if value is None:
        return None
    if type(value) is not dict:
        raise ResultArchiveDecodeError(f"{label} must be an object or null")
    _exact_keys(value, {"threshold_percent", "preserve_region_boundaries"}, label=label)
    try:
        return NodalAveragingPolicy(
            threshold_percent=_strict_float(value["threshold_percent"], label=f"{label}.threshold_percent"),
            preserve_region_boundaries=_strict_bool(value["preserve_region_boundaries"], label=f"{label}.preserve_region_boundaries"),
        )
    except (TypeError, ValueError) as error:
        raise ResultArchiveDecodeError(f"invalid {label}: {error}") from error


def _field_request_to_json(request: FieldRequest) -> dict[str, object]:
    return {
        "field_id": _field_id_to_json(request.field_id),
        "averaging_policy": _policy_to_json(request.averaging_policy),
        "gauss_order": request.gauss_order,
    }


def _field_request_from_json(value: object, *, label: str) -> FieldRequest:
    if type(value) is not dict:
        raise ResultArchiveDecodeError(f"{label} must be an object")
    _exact_keys(value, {"field_id", "averaging_policy", "gauss_order"}, label=label)
    gauss = value["gauss_order"]
    if gauss is not None:
        gauss = _strict_int(gauss, label=f"{label}.gauss_order", minimum=1)
    try:
        return FieldRequest(
            _field_id_from_json(value["field_id"], label=f"{label}.field_id"),
            averaging_policy=_policy_from_json(value["averaging_policy"], label=f"{label}.averaging_policy"),
            gauss_order=gauss,
        )
    except (TypeError, ValueError) as error:
        raise ResultArchiveDecodeError(f"invalid {label}: {error}") from error


def _key_to_json(key: FieldMaterializationKey) -> dict[str, object]:
    return {"request": _field_request_to_json(key.request), "recovery_contract": key.recovery_contract}


def _key_from_json(value: object, *, label: str) -> FieldMaterializationKey:
    if type(value) is not dict:
        raise ResultArchiveDecodeError(f"{label} must be an object")
    _exact_keys(value, {"request", "recovery_contract"}, label=label)
    try:
        return FieldMaterializationKey(
            _field_request_from_json(value["request"], label=f"{label}.request"),
            _strict_int(value["recovery_contract"], label=f"{label}.recovery_contract", minimum=1),
        )
    except (TypeError, ValueError) as error:
        raise ResultArchiveDecodeError(f"invalid {label}: {error}") from error


def stable_field_id(key: FieldMaterializationKey) -> str:
    """Return a deterministic identity derived from the typed field key."""

    if type(key) is not FieldMaterializationKey:
        raise TypeError("key must be FieldMaterializationKey")
    payload = _canonical_json(_key_to_json(key))
    prefix = f"{key.request.field_id.variable.value.lower()}_{key.request.field_id.position.value}"
    digest = hashlib.sha256(payload).hexdigest()[:24]
    return f"{prefix}-{digest}"


def _descriptor_to_json(descriptor: FieldDescriptor) -> dict[str, object]:
    return {
        "field_id": _field_id_to_json(descriptor.field_id),
        "association": descriptor.association.value,
        "quantity": descriptor.quantity.value,
        "components": list(descriptor.components),
        "derived_components": list(descriptor.derived_components),
        "label_key": descriptor.label_key,
        "unit_label": descriptor.unit_label,
        "default_component": descriptor.default_component,
        "order": descriptor.order,
    }


def _descriptor_from_json(value: object, *, label: str) -> FieldDescriptor:
    if type(value) is not dict:
        raise ResultArchiveDecodeError(f"{label} must be an object")
    _exact_keys(value, {"field_id", "association", "quantity", "components", "derived_components", "label_key", "unit_label", "default_component", "order"}, label=label)
    components = value["components"]
    derived = value["derived_components"]
    if type(components) is not list or any(type(item) is not str for item in components):
        raise ResultArchiveDecodeError(f"{label}.components must be a string array")
    if type(derived) is not list or any(type(item) is not str for item in derived):
        raise ResultArchiveDecodeError(f"{label}.derived_components must be a string array")
    unit = value["unit_label"]
    if unit is not None:
        unit = _strict_string(unit, label=f"{label}.unit_label")
    try:
        return FieldDescriptor(
            field_id=_field_id_from_json(value["field_id"], label=f"{label}.field_id"),
            association=FieldAssociation(_strict_string(value["association"], label=f"{label}.association")),
            quantity=PhysicalQuantity(_strict_string(value["quantity"], label=f"{label}.quantity")),
            components=tuple(components),
            derived_components=tuple(derived),
            label_key=_strict_string(value["label_key"], label=f"{label}.label_key"),
            unit_label=unit,
            default_component=_strict_string(value["default_component"], label=f"{label}.default_component"),
            order=_strict_int(value["order"], label=f"{label}.order", minimum=0),
        )
    except (TypeError, ValueError) as error:
        raise ResultArchiveDecodeError(f"invalid {label}: {error}") from error


def _diagnostic_to_json(diagnostic: ResultDiagnostic) -> dict[str, object]:
    return {
        "code": diagnostic.code,
        "severity": diagnostic.severity,
        "message": diagnostic.message,
        "path": _json_plain(diagnostic.path),
        "remediation": diagnostic.remediation,
        "details": _json_plain(diagnostic.details),
    }


def _diagnostics_to_json(values: tuple[ResultDiagnostic, ...]) -> list[object]:
    return [_diagnostic_to_json(item) for item in values]


def _diagnostic_from_json(value: object, *, label: str) -> ResultDiagnostic:
    if type(value) is not dict:
        raise ResultArchiveDecodeError(f"{label} must be an object")
    _exact_keys(value, {"code", "severity", "message", "path", "remediation", "details"}, label=label)
    path = value["path"]
    if type(path) is not list:
        raise ResultArchiveDecodeError(f"{label}.path must be an array")
    details = value["details"]
    if type(details) is not dict:
        raise ResultArchiveDecodeError(f"{label}.details must be an object")
    try:
        return ResultDiagnostic(
            code=_strict_string(value["code"], label=f"{label}.code"),
            severity=_strict_string(value["severity"], label=f"{label}.severity"),
            message=_strict_string(value["message"], label=f"{label}.message"),
            path=tuple(path),
            remediation=_strict_string(value["remediation"], label=f"{label}.remediation"),
            details=details,
        )
    except (TypeError, ValueError) as error:
        raise ResultArchiveDecodeError(f"invalid {label}: {error}") from error


def _diagnostics_from_json(value: object, *, label: str) -> tuple[ResultDiagnostic, ...]:
    if type(value) is not list:
        raise ResultArchiveDecodeError(f"{label} must be an array")
    return tuple(_diagnostic_from_json(item, label=f"{label}[{index}]") for index, item in enumerate(value))


def _profile_to_json(profile: ElementResultProfile) -> dict[str, object]:
    return {
        "family": profile.family.value,
        "canonical_element_types": list(profile.canonical_element_types),
        "element_families": list(profile.element_families),
        "dofs_per_node": profile.dofs_per_node,
        "dof_labels": list(profile.dof_labels),
        "force_labels": list(profile.force_labels),
        "primary_compatible": profile.primary_compatible,
        "stress_compatible": profile.stress_compatible,
    }


def _profile_from_json(value: object, *, label: str) -> ElementResultProfile:
    if type(value) is not dict:
        raise ResultArchiveDecodeError(f"{label} must be an object")
    _exact_keys(value, {"family", "canonical_element_types", "element_families", "dofs_per_node", "dof_labels", "force_labels", "primary_compatible", "stress_compatible"}, label=label)
    sequences = {}
    for key in ("canonical_element_types", "element_families", "dof_labels", "force_labels"):
        items = value[key]
        if type(items) is not list or any(type(item) is not str or not item for item in items):
            raise ResultArchiveDecodeError(f"{label}.{key} must be an array of nonblank strings")
        sequences[key] = tuple(items)
    dofs = value["dofs_per_node"]
    if dofs is not None:
        dofs = _strict_int(dofs, label=f"{label}.dofs_per_node", minimum=1)
    try:
        return ElementResultProfile(
            family=ResultModelFamily(_strict_string(value["family"], label=f"{label}.family")),
            canonical_element_types=sequences["canonical_element_types"],
            element_families=sequences["element_families"],
            dofs_per_node=dofs,
            dof_labels=sequences["dof_labels"],
            force_labels=sequences["force_labels"],
            primary_compatible=_strict_bool(value["primary_compatible"], label=f"{label}.primary_compatible"),
            stress_compatible=_strict_bool(value["stress_compatible"], label=f"{label}.stress_compatible"),
        )
    except (TypeError, ValueError) as error:
        raise ResultArchiveDecodeError(f"invalid {label}: {error}") from error


def _catalog_to_json(catalog: ResultCatalog) -> dict[str, object]:
    return {
        "source": _source_to_json(catalog.source),
        "fields": [
            {
                "key": _key_to_json(item.key),
                "descriptor": _descriptor_to_json(item.descriptor),
                "state": item.state.value,
                "diagnostics": _diagnostics_to_json(item.diagnostics),
            }
            for item in catalog.fields
        ],
        "default_selection": (
            None
            if catalog.default_selection is None
            else {
                "field_key": _key_to_json(catalog.default_selection.field_key),
                "component": catalog.default_selection.component,
            }
        ),
        "diagnostics": _diagnostics_to_json(catalog.diagnostics),
    }


def _catalog_from_json(value: object, *, label: str) -> ResultCatalog:
    if type(value) is not dict:
        raise ResultArchiveDecodeError(f"{label} must be an object")
    _exact_keys(value, {"source", "fields", "default_selection", "diagnostics"}, label=label)
    source = _source_from_json(value["source"], label=f"{label}.source")
    fields_value = value["fields"]
    if type(fields_value) is not list:
        raise ResultArchiveDecodeError(f"{label}.fields must be an array")
    fields: list[FieldAvailability] = []
    for index, item in enumerate(fields_value):
        item_label = f"{label}.fields[{index}]"
        if type(item) is not dict:
            raise ResultArchiveDecodeError(f"{item_label} must be an object")
        _exact_keys(item, {"key", "descriptor", "state", "diagnostics"}, label=item_label)
        try:
            fields.append(
                FieldAvailability(
                    key=_key_from_json(item["key"], label=f"{item_label}.key"),
                    descriptor=_descriptor_from_json(item["descriptor"], label=f"{item_label}.descriptor"),
                    state=FieldState(_strict_string(item["state"], label=f"{item_label}.state")),
                    diagnostics=_diagnostics_from_json(item["diagnostics"], label=f"{item_label}.diagnostics"),
                )
            )
        except (TypeError, ValueError) as error:
            raise ResultArchiveDecodeError(f"invalid {item_label}: {error}") from error
    selection_value = value["default_selection"]
    selection = None
    if selection_value is not None:
        if type(selection_value) is not dict:
            raise ResultArchiveDecodeError(f"{label}.default_selection must be an object or null")
        _exact_keys(selection_value, {"field_key", "component"}, label=f"{label}.default_selection")
        try:
            selection = ScalarFieldSelection(
                _key_from_json(selection_value["field_key"], label=f"{label}.default_selection.field_key"),
                _strict_string(selection_value["component"], label=f"{label}.default_selection.component"),
            )
        except (TypeError, ValueError) as error:
            raise ResultArchiveDecodeError(f"invalid {label}.default_selection: {error}") from error
    try:
        return ResultCatalog(
            source=source,
            fields=tuple(fields),
            default_selection=selection,
            diagnostics=_diagnostics_from_json(value["diagnostics"], label=f"{label}.diagnostics"),
        )
    except (TypeError, ValueError) as error:
        raise ResultArchiveDecodeError(f"invalid {label}: {error}") from error


def _output_variable_to_json(value: OutputVariableProjection) -> dict[str, object]:
    return {
        "source_variable_indices": list(value.source_variable_indices),
        "source_variables": list(value.source_variables),
        "canonical_variable": None if value.canonical_variable is None else value.canonical_variable.value,
        "field_requests": [_field_request_to_json(item) for item in value.field_requests],
        "diagnostics": _diagnostics_to_json(value.diagnostics),
    }


def _output_variable_from_json(value: object, *, label: str) -> OutputVariableProjection:
    if type(value) is not dict:
        raise ResultArchiveDecodeError(f"{label} must be an object")
    _exact_keys(value, {"source_variable_indices", "source_variables", "canonical_variable", "field_requests", "diagnostics"}, label=label)
    indices = value["source_variable_indices"]
    names = value["source_variables"]
    requests = value["field_requests"]
    if type(indices) is not list or any(type(item) is not int for item in indices):
        raise ResultArchiveDecodeError(f"{label}.source_variable_indices must be an integer array")
    if type(names) is not list or any(type(item) is not str for item in names):
        raise ResultArchiveDecodeError(f"{label}.source_variables must be a string array")
    if type(requests) is not list:
        raise ResultArchiveDecodeError(f"{label}.field_requests must be an array")
    canonical = value["canonical_variable"]
    try:
        canonical_value = None if canonical is None else ResultVariable(_strict_string(canonical, label=f"{label}.canonical_variable"))
        return OutputVariableProjection(
            source_variable_indices=tuple(indices),
            source_variables=tuple(names),
            canonical_variable=canonical_value,
            field_requests=tuple(_field_request_from_json(item, label=f"{label}.field_requests[{index}]") for index, item in enumerate(requests)),
            diagnostics=_diagnostics_from_json(value["diagnostics"], label=f"{label}.diagnostics"),
        )
    except (TypeError, ValueError) as error:
        raise ResultArchiveDecodeError(f"invalid {label}: {error}") from error


def _executable_request_to_json(value: ExecutableOutputRequest) -> dict[str, object]:
    return {
        "request_index": value.request_index,
        "kind": value.kind,
        "target": value.target,
        "frequency": value.frequency,
        "variables": [_output_variable_to_json(item) for item in value.variables],
        "field_requests": [_field_request_to_json(item) for item in value.field_requests],
    }


def _executable_request_from_json(value: object, *, label: str) -> ExecutableOutputRequest:
    if type(value) is not dict:
        raise ResultArchiveDecodeError(f"{label} must be an object")
    _exact_keys(value, {"request_index", "kind", "target", "frequency", "variables", "field_requests"}, label=label)
    variables = value["variables"]
    requests = value["field_requests"]
    if type(variables) is not list or type(requests) is not list:
        raise ResultArchiveDecodeError(f"{label}.variables and field_requests must be arrays")
    try:
        return ExecutableOutputRequest(
            request_index=_strict_int(value["request_index"], label=f"{label}.request_index", minimum=0),
            kind=_strict_string(value["kind"], label=f"{label}.kind"),
            target=_strict_string(value["target"], label=f"{label}.target"),
            frequency=_strict_int(value["frequency"], label=f"{label}.frequency", minimum=1),
            variables=tuple(_output_variable_from_json(item, label=f"{label}.variables[{index}]") for index, item in enumerate(variables)),
            field_requests=tuple(_field_request_from_json(item, label=f"{label}.field_requests[{index}]") for index, item in enumerate(requests)),
        )
    except (TypeError, ValueError) as error:
        raise ResultArchiveDecodeError(f"invalid {label}: {error}") from error


def _output_execution_to_json(value: OutputVariableExecution) -> dict[str, object]:
    return {
        "source_variable_indices": list(value.source_variable_indices),
        "canonical_variable": None if value.canonical_variable is None else value.canonical_variable.value,
        "field_keys": [_key_to_json(item) for item in value.field_keys],
        "status": value.status.value,
        "diagnostics": _diagnostics_to_json(value.diagnostics),
    }


def _output_execution_from_json(value: object, *, label: str) -> OutputVariableExecution:
    if type(value) is not dict:
        raise ResultArchiveDecodeError(f"{label} must be an object")
    _exact_keys(value, {"source_variable_indices", "canonical_variable", "field_keys", "status", "diagnostics"}, label=label)
    indices = value["source_variable_indices"]
    keys = value["field_keys"]
    if type(indices) is not list or any(type(item) is not int for item in indices):
        raise ResultArchiveDecodeError(f"{label}.source_variable_indices must be an integer array")
    if type(keys) is not list:
        raise ResultArchiveDecodeError(f"{label}.field_keys must be an array")
    canonical = value["canonical_variable"]
    try:
        return OutputVariableExecution(
            source_variable_indices=tuple(indices),
            canonical_variable=None if canonical is None else ResultVariable(_strict_string(canonical, label=f"{label}.canonical_variable")),
            field_keys=tuple(_key_from_json(item, label=f"{label}.field_keys[{index}]") for index, item in enumerate(keys)),
            status=OutputExecutionStatus(_strict_string(value["status"], label=f"{label}.status")),
            diagnostics=_diagnostics_from_json(value["diagnostics"], label=f"{label}.diagnostics"),
        )
    except (TypeError, ValueError) as error:
        raise ResultArchiveDecodeError(f"invalid {label}: {error}") from error


def _request_execution_to_json(value: OutputRequestExecution) -> dict[str, object]:
    return {
        "request_index": value.request_index,
        "status": value.status.value,
        "executable_request": None if value.executable_request is None else _executable_request_to_json(value.executable_request),
        "variables": [_output_execution_to_json(item) for item in value.variables],
        "diagnostics": _diagnostics_to_json(value.diagnostics),
    }


def _request_execution_from_json(value: object, *, label: str) -> OutputRequestExecution:
    if type(value) is not dict:
        raise ResultArchiveDecodeError(f"{label} must be an object")
    _exact_keys(value, {"request_index", "status", "executable_request", "variables", "diagnostics"}, label=label)
    variables = value["variables"]
    if type(variables) is not list:
        raise ResultArchiveDecodeError(f"{label}.variables must be an array")
    try:
        return OutputRequestExecution(
            request_index=_strict_int(value["request_index"], label=f"{label}.request_index", minimum=0),
            status=OutputExecutionStatus(_strict_string(value["status"], label=f"{label}.status")),
            executable_request=None if value["executable_request"] is None else _executable_request_from_json(value["executable_request"], label=f"{label}.executable_request"),
            variables=tuple(_output_execution_from_json(item, label=f"{label}.variables[{index}]") for index, item in enumerate(variables)),
            diagnostics=_diagnostics_from_json(value["diagnostics"], label=f"{label}.diagnostics"),
        )
    except (TypeError, ValueError) as error:
        raise ResultArchiveDecodeError(f"invalid {label}: {error}") from error


def _report_to_json(value: ResultExecutionReport | None) -> object:
    if value is None:
        return None
    return {
        "source": _source_to_json(value.source),
        "requests": [_request_execution_to_json(item) for item in value.requests],
        "diagnostics": _diagnostics_to_json(value.diagnostics),
    }


def _report_from_json(value: object, *, label: str) -> ResultExecutionReport | None:
    if value is None:
        return None
    if type(value) is not dict:
        raise ResultArchiveDecodeError(f"{label} must be an object or null")
    _exact_keys(value, {"source", "requests", "diagnostics"}, label=label)
    requests = value["requests"]
    if type(requests) is not list:
        raise ResultArchiveDecodeError(f"{label}.requests must be an array")
    try:
        return ResultExecutionReport(
            source=_source_from_json(value["source"], label=f"{label}.source"),
            requests=tuple(_request_execution_from_json(item, label=f"{label}.requests[{index}]") for index, item in enumerate(requests)),
            diagnostics=_diagnostics_from_json(value["diagnostics"], label=f"{label}.diagnostics"),
        )
    except (TypeError, ValueError) as error:
        raise ResultArchiveDecodeError(f"invalid {label}: {error}") from error


def _unit_to_json(value: UnitContext | None) -> object:
    return None if value is None else value.to_dict()


def _unit_from_json(value: object, *, label: str) -> UnitContext | None:
    if value is None:
        return None
    if type(value) is not dict:
        raise ResultArchiveDecodeError(f"{label} must be an object or null")
    try:
        return UnitContext.from_dict(value)
    except (TypeError, ValueError) as error:
        raise ResultArchiveDecodeError(f"invalid {label}: {error}") from error


def _array_bytes(array: np.ndarray) -> bytes:
    if type(array) is not np.ndarray:
        raise ResultArchiveEncodeError("archive arrays must be numpy arrays")
    if array.dtype.hasobject:
        raise ResultArchiveEncodeError("object dtype arrays are forbidden")
    if np.issubdtype(array.dtype, np.complexfloating):
        raise ResultArchiveEncodeError("complex arrays are forbidden")
    stream = BytesIO()
    try:
        np.save(stream, np.ascontiguousarray(array), allow_pickle=False)
    except (TypeError, ValueError) as error:
        raise ResultArchiveEncodeError(f"NumPy array encoding failed: {error}") from error
    return stream.getvalue()


def _array_meta(data: bytes, array: np.ndarray) -> dict[str, object]:
    return {
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "nbytes": array.nbytes,
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def _checked_array_meta(value: object, *, label: str) -> dict[str, object]:
    if type(value) is not dict:
        raise ResultArchiveDecodeError(f"{label} must be an object")
    _exact_keys(value, {"dtype", "shape", "nbytes", "sha256"}, label=label)
    dtype = _strict_string(value["dtype"], label=f"{label}.dtype")
    try:
        parsed = np.dtype(dtype)
    except TypeError as error:
        raise ResultArchiveDecodeError(f"{label}.dtype is invalid") from error
    if parsed.hasobject or np.issubdtype(parsed, np.complexfloating):
        raise ResultArchiveDecodeError(f"{label}.dtype cannot be object or complex")
    shape = value["shape"]
    if type(shape) is not list or any(type(item) is not int or item < 0 for item in shape):
        raise ResultArchiveDecodeError(f"{label}.shape must be nonnegative integer array")
    nbytes = _strict_int(value["nbytes"], label=f"{label}.nbytes", minimum=0)
    try:
        declared_nbytes = math.prod(shape, start=1) * parsed.itemsize
    except (OverflowError, ValueError) as error:
        raise ResultArchiveDecodeError(f"{label}.shape is too large") from error
    if declared_nbytes > _MAX_ARRAY_BYTES:
        raise ResultArchiveDecodeError(
            f"{label} declared byte size exceeds archive safety limit"
        )
    if nbytes != declared_nbytes:
        raise ResultArchiveDecodeError(
            f"{label} declared byte size does not match dtype/shape"
        )
    sha = _strict_string(value["sha256"], label=f"{label}.sha256")
    if not re.fullmatch(r"[0-9a-f]{64}", sha):
        raise ResultArchiveDecodeError(f"{label}.sha256 must be lowercase hexadecimal")
    return {"dtype": parsed.str, "shape": tuple(shape), "nbytes": nbytes, "sha256": sha}


def _checked_array(
    raw: bytes,
    meta: dict[str, object],
    *,
    label: str,
    expected_dtype: str,
    expected_shape: tuple[int, ...],
    finite: bool = False,
) -> np.ndarray:
    if hashlib.sha256(raw).hexdigest() != meta["sha256"]:
        raise ResultArchiveDecodeError(f"{label} SHA-256 checksum mismatch")
    try:
        stream = BytesIO(raw)
        array = np.load(stream, allow_pickle=False)
        if stream.tell() != len(raw):
            raise ResultArchiveDecodeError(f"{label} contains trailing bytes")
    except ResultArchiveDecodeError:
        raise
    except (ValueError, OSError, EOFError) as error:
        raise ResultArchiveDecodeError(f"{label} is not a valid non-pickle .npy array") from error
    if type(array) is not np.ndarray:
        raise ResultArchiveDecodeError(f"{label} must contain a plain ndarray")
    if array.dtype.hasobject or np.issubdtype(array.dtype, np.complexfloating):
        raise ResultArchiveDecodeError(f"{label} cannot contain object or complex dtype")
    try:
        expected = np.dtype(expected_dtype)
    except TypeError as error:
        raise ResultArchiveDecodeError(f"{label} expected dtype is invalid") from error
    if array.dtype.str != expected.str or meta["dtype"] != expected.str:
        raise ResultArchiveDecodeError(f"{label} dtype does not match manifest")
    if tuple(array.shape) != expected_shape or tuple(meta["shape"]) != expected_shape:
        raise ResultArchiveDecodeError(f"{label} shape does not match manifest")
    if array.nbytes != meta["nbytes"]:
        raise ResultArchiveDecodeError(f"{label} byte size does not match manifest")
    if finite and not bool(np.isfinite(array).all()):
        raise ResultArchiveDecodeError(f"{label} contains non-finite values")
    if np.issubdtype(array.dtype, np.integer) and array.dtype.itemsize < 1:
        raise ResultArchiveDecodeError(f"{label} integer dtype is invalid")
    array = np.array(array, dtype=array.dtype, order="C", copy=True)
    array.setflags(write=False)
    return array


class _ArchiveBuilder:
    def __init__(self) -> None:
        self.arrays: OrderedDict[str, tuple[dict[str, object], bytes]] = OrderedDict()
        self.expanded_bytes = 0

    def add(self, name: str, array: np.ndarray) -> str:
        if not _safe_entry_name(name):
            raise ResultArchiveEncodeError(f"unsafe archive entry name {name!r}")
        if name in self.arrays:
            raise ResultArchiveEncodeError(f"duplicate archive entry {name!r}")
        owned = np.ascontiguousarray(array)
        if owned.nbytes > _MAX_ARRAY_BYTES:
            raise ResultArchiveEncodeError(
                f"archive array {name!r} exceeds the array size safety limit"
            )
        raw = _array_bytes(owned)
        if len(raw) > _MAX_ZIP_ENTRY_BYTES:
            raise ResultArchiveEncodeError(
                f"archive entry {name!r} exceeds the size safety limit"
            )
        _validate_entry_count(
            len(self.arrays) + 2,
            error_type=ResultArchiveEncodeError,
        )
        if self.expanded_bytes + len(raw) > _MAX_ZIP_TOTAL_BYTES:
            raise ResultArchiveEncodeError(
                "result archive exceeds the total size safety limit"
            )
        self.arrays[name] = (_array_meta(raw, owned), raw)
        self.expanded_bytes += len(raw)
        return name


def _safe_entry_name(name: str) -> bool:
    if type(name) is not str or not name or "\\" in name or name.startswith("/") or ":" in name:
        return False
    parts = name.split("/")
    return all(part not in {"", ".", ".."} for part in parts)


def _validate_container_bytes(
    size: int,
    *,
    error_type: type[Exception],
) -> None:
    if size > _MAX_ZIP_CONTAINER_BYTES:
        raise error_type(
            "result archive container exceeds the size safety limit"
        )


def _validate_entry_count(
    count: int,
    *,
    error_type: type[Exception],
) -> None:
    if count > _MAX_ZIP_ENTRY_COUNT:
        raise error_type("result archive contains too many ZIP entries")


def _validate_manifest_bytes(
    size: int,
    *,
    error_type: type[Exception],
) -> None:
    if size > _MAX_MANIFEST_BYTES:
        raise error_type("result archive manifest exceeds the size safety limit")


def _validate_zip_infos(
    infos: list[zipfile.ZipInfo],
    *,
    error_type: type[Exception],
) -> int:
    """Validate central-directory size/count/ratio limits and return total raw bytes."""

    _validate_entry_count(len(infos), error_type=error_type)
    total_size = 0
    for info in infos:
        if info.file_size < 0 or info.file_size > _MAX_ZIP_ENTRY_BYTES:
            raise error_type(
                f"ZIP entry {info.filename!r} exceeds the size safety limit"
            )
        total_size += info.file_size
        if total_size > _MAX_ZIP_TOTAL_BYTES:
            raise error_type("result archive exceeds the total size safety limit")
        if info.file_size and info.compress_size <= 0:
            raise error_type(
                f"ZIP entry {info.filename!r} has an invalid compression size"
            )
        if (
            info.compress_size
            and info.file_size / info.compress_size > _MAX_ZIP_COMPRESSION_RATIO
        ):
            raise error_type(
                f"ZIP entry {info.filename!r} exceeds the compression-ratio safety limit"
            )
    return total_size


def _encode_arrays(snapshot: ResultArchiveSnapshot, builder: _ArchiveBuilder) -> tuple[dict[str, object], dict[str, object], list[object]]:
    topology = snapshot.topology
    regions = archive_region_dictionary(snapshot)
    region_index = {key: index for index, key in enumerate(regions)}
    element_types: list[str] = []
    element_type_index: dict[str, int] = {}
    for element_type in topology.element_types:
        if element_type not in element_type_index:
            element_type_index[element_type] = len(element_types)
            element_types.append(element_type)
    topology_arrays = {
        "node_ids": builder.add("topology/node_ids.npy", np.asarray(topology.node_ids, dtype="<i8")),
        "node_coordinates": builder.add("topology/node_coordinates.npy", np.asarray(topology.node_coordinates, dtype="<f8")),
        "nodal_displacements": builder.add("topology/nodal_displacements.npy", np.asarray(topology.nodal_displacements, dtype="<f8")),
        "element_ids": builder.add("topology/element_ids.npy", np.asarray(topology.element_ids, dtype="<i8")),
        "connectivity_offsets": builder.add("topology/connectivity_offsets.npy", _connectivity_offsets(topology.connectivity)),
        "connectivity": builder.add("topology/connectivity.npy", np.asarray([node_id for row in topology.connectivity for node_id in row], dtype="<i8")),
        "element_type_indices": builder.add("topology/element_type_indices.npy", np.asarray([element_type_index[item] for item in topology.element_types], dtype="<i8")),
        "region_indices": builder.add("topology/region_indices.npy", np.asarray([region_index[item] for item in topology.element_region_keys], dtype="<i8")),
    }
    field_entries: list[object] = []
    seen_ids: set[str] = set()
    for field_data in snapshot.fields:
        field_id = stable_field_id(field_data.key)
        if field_id in seen_ids:
            raise ResultArchiveEncodeError(f"duplicate stable field id {field_id!r}")
        seen_ids.add(field_id)
        prefix = f"fields/{field_id}"
        rows = len(field_data.locations)
        coordinates = np.asarray([item.coordinates for item in field_data.locations], dtype="<f8").reshape((rows, 3))
        displacement = np.zeros((rows, 3), dtype="<f8")
        displacement_mask = np.zeros(rows, dtype="<u1")
        ids = {name: np.full(rows, -1, dtype="<i8") for name in ("node_id", "element_id", "integration_point", "local_node", "region_index")}
        averaged = np.zeros(rows, dtype="<u1")
        averaged_mask = np.zeros(rows, dtype="<u1")
        for row, location in enumerate(field_data.locations):
            if location.displacement is not None:
                displacement[row] = location.displacement
                displacement_mask[row] = 1
            if location.node_id is not None:
                ids["node_id"][row] = location.node_id
            if location.element_id is not None:
                ids["element_id"][row] = location.element_id
            if location.integration_point is not None:
                ids["integration_point"][row] = location.integration_point
            if location.local_node is not None:
                ids["local_node"][row] = location.local_node
            if location.region_key is not None:
                ids["region_index"][row] = region_index[location.region_key]
            if location.averaged is not None:
                averaged[row] = int(location.averaged)
                averaged_mask[row] = 1
        arrays = {
            "values": builder.add(f"{prefix}/values.npy", np.asarray(field_data.values, dtype="<f8")),
            "coordinates": builder.add(f"{prefix}/locations-coordinates.npy", coordinates),
            "displacement": builder.add(f"{prefix}/locations-displacement.npy", displacement),
            "displacement_mask": builder.add(f"{prefix}/locations-displacement-mask.npy", displacement_mask),
            "node_id": builder.add(f"{prefix}/locations-node-ids.npy", ids["node_id"]),
            "element_id": builder.add(f"{prefix}/locations-element-ids.npy", ids["element_id"]),
            "integration_point": builder.add(f"{prefix}/locations-integration-points.npy", ids["integration_point"]),
            "local_node": builder.add(f"{prefix}/locations-local-nodes.npy", ids["local_node"]),
            "region_index": builder.add(f"{prefix}/locations-region-indices.npy", ids["region_index"]),
            "averaged": builder.add(f"{prefix}/locations-averaged.npy", averaged),
            "averaged_mask": builder.add(f"{prefix}/locations-averaged-mask.npy", averaged_mask),
        }
        if any(location.section_point is not None for location in field_data.locations):
            if any(location.section_point is None for location in field_data.locations):
                raise ResultArchiveEncodeError(
                    "section-point fields cannot mix point and non-point rows"
                )
            points = tuple(location.section_point for location in field_data.locations)
            arrays.update(
                {
                    "section_point_number": builder.add(
                        f"{prefix}/locations-section-point-numbers.npy",
                        np.asarray([point.number for point in points], dtype="<i8"),
                    ),
                    "section_point_local_y": builder.add(
                        f"{prefix}/locations-section-point-local-y.npy",
                        np.asarray([point.local_y for point in points], dtype="<f8"),
                    ),
                    "section_point_local_z": builder.add(
                        f"{prefix}/locations-section-point-local-z.npy",
                        np.asarray([point.local_z for point in points], dtype="<f8"),
                    ),
                }
            )
        field_entries.append({
            "id": field_id,
            "key": _key_to_json(field_data.key),
            "descriptor": _descriptor_to_json(field_data.descriptor),
            "location_rows": rows,
            "arrays": arrays,
        })
    return (
        {
            "arrays": topology_arrays,
            "node_count": len(topology.node_ids),
            "element_count": len(topology.element_ids),
            "element_types": element_types,
            "region_keys": [encode_result_region_key(item) for item in regions],
        },
        {"regions": [encode_result_region_key(item) for item in regions]},
        field_entries,
    )


def _connectivity_offsets(connectivity: tuple[tuple[int, ...], ...]) -> np.ndarray:
    offsets = [0]
    for row in connectivity:
        offsets.append(offsets[-1] + len(row))
    return np.asarray(offsets, dtype="<i8")


def _manifest_for_snapshot(snapshot: ResultArchiveSnapshot, builder: _ArchiveBuilder) -> dict[str, object]:
    topology_meta, _regions_meta, fields = _encode_arrays(snapshot, builder)
    model_projection = snapshot.model_projection
    return {
        "format": FORMAT_NAME,
        "schema": SCHEMA_VERSION,
        "archive_id": snapshot.archive_id,
        "created_at": _datetime_to_json(snapshot.created_at),
        "producer_version": snapshot.producer_version,
        "source": _source_to_json(snapshot.source),
        "origin": {
            "model_name": snapshot.origin.model_name,
            "source_basename": snapshot.origin.source_basename,
            "model_fingerprint": snapshot.origin.model_fingerprint,
            "provenance": _json_plain(snapshot.origin.provenance),
        },
        "run": {
            "name": snapshot.run.name,
            "step_name": snapshot.run.step_name,
            "created_at": _datetime_to_json(snapshot.run.created_at),
            "started_at": None if snapshot.run.started_at is None else _datetime_to_json(snapshot.run.started_at),
            "finished_at": None if snapshot.run.finished_at is None else _datetime_to_json(snapshot.run.finished_at),
            "timings": _json_plain(snapshot.run.timings),
            "messages": list(snapshot.run.messages),
            "output_report": _report_to_json(snapshot.run.output_report),
        },
        "unit_context": _unit_to_json(snapshot.unit_context),
        "profile": _profile_to_json(snapshot.profile),
        "catalog": _catalog_to_json(snapshot.catalog),
        "materialization_generation": snapshot.materialization.generation,
        "model_projection": {
            "unit_context": _unit_to_json(model_projection.unit_context),
            "named_region_node_ids": _json_plain(model_projection.named_region_node_ids),
            "named_region_element_ids": _json_plain(model_projection.named_region_element_ids),
            "summaries": _json_plain(model_projection.summaries),
        },
        "inspection_summaries": _json_plain(model_projection.summaries),
        "topology": topology_meta,
        "fields": fields,
        "arrays": {name: meta for name, (meta, _raw) in builder.arrays.items()},
    }


def encode_result_archive_v1(snapshot: ResultArchiveSnapshot) -> bytes:
    """Encode one immutable snapshot to deterministic ZIP bytes."""

    if type(snapshot) is not ResultArchiveSnapshot:
        raise TypeError("snapshot must be ResultArchiveSnapshot")
    try:
        builder = _ArchiveBuilder()
        manifest = _manifest_for_snapshot(snapshot, builder)
        manifest_bytes = _canonical_json(manifest)
        _validate_manifest_bytes(
            len(manifest_bytes),
            error_type=ResultArchiveEncodeError,
        )
        _validate_entry_count(
            len(builder.arrays) + 1,
            error_type=ResultArchiveEncodeError,
        )
        expanded_size = len(manifest_bytes) + builder.expanded_bytes
        if expanded_size > _MAX_ZIP_TOTAL_BYTES:
            raise ResultArchiveEncodeError(
                "result archive exceeds the total size safety limit"
            )
        output = BytesIO()
        with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6, strict_timestamps=True) as archive:
            _writestr_deterministic(archive, MANIFEST_NAME, manifest_bytes)
            for name, (_meta, raw) in builder.arrays.items():
                _writestr_deterministic(archive, name, raw)
            _validate_zip_infos(
                archive.infolist(),
                error_type=ResultArchiveEncodeError,
            )
        serialized = output.getvalue()
        _validate_container_bytes(
            len(serialized),
            error_type=ResultArchiveEncodeError,
        )
        return serialized
    except ResultArchiveEncodeError:
        raise
    except (OSError, TypeError, ValueError, zipfile.BadZipFile) as error:
        raise ResultArchiveEncodeError(f"result archive ZIP encoding failed: {error}") from error


def _writestr_deterministic(archive: zipfile.ZipFile, name: str, data: bytes) -> None:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 0
    info.external_attr = 0
    archive.writestr(info, data)


def _open_archive_bytes(data: bytes) -> tuple[dict[str, Any], dict[str, bytes]]:
    if type(data) is not bytes:
        raise TypeError("archive data must be bytes")
    _validate_container_bytes(len(data), error_type=ResultArchiveDecodeError)
    try:
        archive = zipfile.ZipFile(BytesIO(data), "r")
    except (zipfile.BadZipFile, OSError, ValueError) as error:
        raise ResultArchiveDecodeError(f"result archive is not a valid ZIP: {error}") from error
    try:
        infos = archive.infolist()
        _validate_zip_infos(infos, error_type=ResultArchiveDecodeError)
        names = [item.filename for item in infos]
        if len(names) != len(set(names)):
            raise ResultArchiveDecodeError("result archive contains duplicate entries")
        if any(not _safe_entry_name(name) for name in names):
            raise ResultArchiveDecodeError("result archive contains unsafe entry path")
        if MANIFEST_NAME not in names:
            raise ResultArchiveDecodeError("result archive is missing manifest.json")
        manifest_info = archive.getinfo(MANIFEST_NAME)
        _validate_manifest_bytes(
            manifest_info.file_size,
            error_type=ResultArchiveDecodeError,
        )
        manifest_bytes = _read_zip_entry_bounded(
            archive,
            MANIFEST_NAME,
            _MAX_MANIFEST_BYTES,
        )
        total_read = len(manifest_bytes)
        if total_read > _MAX_ZIP_TOTAL_BYTES:
            raise ResultArchiveDecodeError(
                "result archive exceeds the total size safety limit"
            )
        manifest = _parse_manifest(manifest_bytes)
        _validate_array_entry_names(manifest)
        array_meta = manifest["arrays"]
        if type(array_meta) is not dict:
            raise ResultArchiveDecodeError("manifest.arrays must be an object")
        arrays: dict[str, bytes] = {}
        expected_names = {MANIFEST_NAME, *array_meta}
        if set(names) != expected_names:
            unknown = sorted(set(names) - expected_names)
            missing = sorted(expected_names - set(names))
            raise ResultArchiveDecodeError(f"ZIP entries do not match manifest (missing={missing}, unknown={unknown})")
        expected_order = [MANIFEST_NAME]
        expected_order.extend(
            manifest["topology"]["arrays"][key]
            for key in _TOPOLOGY_ARRAY_ORDER
        )
        for field in manifest["fields"]:
            expected_order.extend(
                field["arrays"][key]
                for key in _FIELD_ARRAY_ORDER
                if key in field["arrays"]
            )
        if names != expected_order:
            raise ResultArchiveDecodeError("ZIP entries are not in canonical order")
        for name in array_meta:
            if not _safe_entry_name(name) or name == MANIFEST_NAME:
                raise ResultArchiveDecodeError(f"manifest declares unsafe array entry {name!r}")
            _checked_array_meta(array_meta[name], label=f"manifest.arrays[{name!r}]")
            raw = _read_zip_entry_bounded(
                archive,
                name,
                _MAX_ZIP_ENTRY_BYTES,
            )
            total_read += len(raw)
            if total_read > _MAX_ZIP_TOTAL_BYTES:
                raise ResultArchiveDecodeError(
                    "result archive exceeds the total size safety limit"
                )
            arrays[name] = raw
        return manifest, arrays
    except ResultArchiveDecodeError:
        raise
    except (KeyError, RuntimeError, zipfile.BadZipFile, OSError) as error:
        raise ResultArchiveDecodeError(f"result archive read failed: {error}") from error
    finally:
        archive.close()


def _inspect_result_archive_header(
    archive: zipfile.ZipFile,
) -> tuple[str, int]:
    infos = archive.infolist()
    _validate_zip_infos(infos, error_type=ResultArchiveDecodeError)
    names = [item.filename for item in infos]
    if len(names) != len(set(names)):
        raise ResultArchiveDecodeError("result archive contains duplicate entries")
    if any(not _safe_entry_name(name) for name in names):
        raise ResultArchiveDecodeError("result archive contains unsafe entry path")
    if MANIFEST_NAME not in names:
        raise ResultArchiveDecodeError("result archive is missing manifest.json")
    manifest_info = archive.getinfo(MANIFEST_NAME)
    _validate_manifest_bytes(
        manifest_info.file_size,
        error_type=ResultArchiveDecodeError,
    )
    manifest_bytes = _read_zip_entry_bounded(
        archive,
        MANIFEST_NAME,
        _MAX_MANIFEST_BYTES,
    )
    return _parse_manifest_header(manifest_bytes)


def inspect_result_archive_header_bytes(data: bytes) -> tuple[str, int]:
    """Return ``(format, schema)`` without decoding archive arrays."""

    if type(data) is not bytes:
        raise TypeError("archive data must be bytes")
    _validate_container_bytes(len(data), error_type=ResultArchiveDecodeError)
    try:
        with zipfile.ZipFile(BytesIO(data), "r") as archive:
            return _inspect_result_archive_header(archive)
    except ResultArchiveDecodeError:
        raise
    except (KeyError, OSError, RuntimeError, ValueError, zipfile.BadZipFile) as error:
        raise ResultArchiveDecodeError(
            f"result archive header could not be read: {error}"
        ) from error


def inspect_result_archive_header_path(
    path: str | Path,
) -> tuple[str, int]:
    """Read a bounded archive header directly from a filesystem path."""

    source = Path(path)
    try:
        declared_size = source.stat().st_size
        _validate_container_bytes(
            declared_size,
            error_type=ResultArchiveDecodeError,
        )
        with zipfile.ZipFile(source, "r") as archive:
            return _inspect_result_archive_header(archive)
    except ResultArchiveDecodeError:
        raise
    except (KeyError, OSError, RuntimeError, TypeError, ValueError, zipfile.BadZipFile) as error:
        raise ResultArchiveDecodeError(
            f"result archive header could not be read: {error}"
        ) from error


def _read_zip_entry_bounded(
    archive: zipfile.ZipFile,
    name: str,
    limit: int,
) -> bytes:
    try:
        declared_size = archive.getinfo(name).file_size
        if declared_size > limit:
            raise ResultArchiveDecodeError(
                f"result archive entry {name!r} exceeds the size safety limit"
            )
        with archive.open(name, "r") as stream:
            data = stream.read(declared_size + 1)
    except (KeyError, OSError, RuntimeError, zipfile.BadZipFile) as error:
        raise ResultArchiveDecodeError(
            f"result archive entry {name!r} could not be read: {error}"
        ) from error
    if len(data) != declared_size:
        raise ResultArchiveDecodeError(
            f"result archive entry {name!r} size changed during read"
        )
    return data


def _array_from_manifest(
    arrays: Mapping[str, bytes],
    manifest_arrays: Mapping[str, object],
    name: object,
    *,
    label: str,
    expected_dtype: str,
    expected_shape: tuple[int, ...],
    finite: bool = False,
) -> np.ndarray:
    if type(name) is not str:
        raise ResultArchiveDecodeError(f"{label} entry name must be a string")
    if name not in arrays or name not in manifest_arrays:
        raise ResultArchiveDecodeError(f"{label} references missing array entry {name!r}")
    meta = _checked_array_meta(manifest_arrays[name], label=f"manifest.arrays[{name!r}]")
    return _checked_array(arrays[name], meta, label=label, expected_dtype=expected_dtype, expected_shape=expected_shape, finite=finite)


def _decode_origin(value: object) -> ResultArchiveOrigin:
    if type(value) is not dict:
        raise ResultArchiveDecodeError("manifest.origin must be an object")
    _exact_keys(value, {"model_name", "source_basename", "model_fingerprint", "provenance"}, label="manifest.origin")
    for key in ("model_name", "source_basename", "model_fingerprint"):
        if value[key] is not None:
            _strict_string(value[key], label=f"manifest.origin.{key}")
    if type(value["provenance"]) is not dict:
        raise ResultArchiveDecodeError("manifest.origin.provenance must be an object")
    try:
        return ResultArchiveOrigin(
            model_name=value["model_name"],
            source_basename=value["source_basename"],
            model_fingerprint=value["model_fingerprint"],
            provenance=value["provenance"],
        )
    except (RecursionError, TypeError, ValueError) as error:
        raise ResultArchiveDecodeError(f"invalid manifest.origin: {error}") from error


def _decode_run(value: object) -> ResultArchiveRun:
    if type(value) is not dict:
        raise ResultArchiveDecodeError("manifest.run must be an object")
    _exact_keys(value, {"name", "step_name", "created_at", "started_at", "finished_at", "timings", "messages", "output_report"}, label="manifest.run")
    timings = value["timings"]
    messages = value["messages"]
    if type(timings) is not dict:
        raise ResultArchiveDecodeError("manifest.run.timings must be an object")
    if type(messages) is not list or any(type(item) is not str for item in messages):
        raise ResultArchiveDecodeError("manifest.run.messages must be a string array")
    checked_timings = {key: _strict_float(item, label=f"manifest.run.timings.{key}", minimum=0.0) for key, item in timings.items()}
    try:
        return ResultArchiveRun(
            name=_strict_string(value["name"], label="manifest.run.name"),
            step_name=_strict_string(value["step_name"], label="manifest.run.step_name"),
            created_at=_datetime_from_json(value["created_at"], label="manifest.run.created_at"),
            started_at=None if value["started_at"] is None else _datetime_from_json(value["started_at"], label="manifest.run.started_at"),
            finished_at=None if value["finished_at"] is None else _datetime_from_json(value["finished_at"], label="manifest.run.finished_at"),
            timings=checked_timings,
            messages=tuple(messages),
            output_report=_report_from_json(value["output_report"], label="manifest.run.output_report"),
        )
    except (TypeError, ValueError) as error:
        raise ResultArchiveDecodeError(f"invalid manifest.run: {error}") from error


def _decode_named_region_mapping(
    value: object,
    *,
    label: str,
    allowed_ids: frozenset[int],
) -> dict[str, tuple[int, ...]]:
    if type(value) is not dict:
        raise ResultArchiveDecodeError(f"{label} must be an object")
    result: dict[str, tuple[int, ...]] = {}
    for name, raw_ids in value.items():
        if type(name) is not str or not name.strip():
            raise ResultArchiveDecodeError(f"{label} keys must be nonblank strings")
        if type(raw_ids) is not list:
            raise ResultArchiveDecodeError(f"{label}.{name} must be an integer array")
        if any(type(item) is not int or item <= 0 for item in raw_ids):
            raise ResultArchiveDecodeError(f"{label}.{name} must contain positive integers")
        ids = tuple(raw_ids)
        if len(set(ids)) != len(ids):
            raise ResultArchiveDecodeError(f"{label}.{name} contains duplicate IDs")
        if not set(ids).issubset(allowed_ids):
            raise ResultArchiveDecodeError(f"{label}.{name} references an unknown ID")
        result[name] = ids
    return result


def _decode_topology(
    manifest: Mapping[str, object],
    arrays: Mapping[str, bytes],
    source: ResultSourceKey,
) -> ResultTopologyProjection:
    topology = manifest["topology"]
    if type(topology) is not dict:
        raise ResultArchiveDecodeError("manifest.topology must be an object")
    _exact_keys(topology, {"arrays", "node_count", "element_count", "element_types", "region_keys"}, label="manifest.topology")
    node_count = _strict_int(topology["node_count"], label="manifest.topology.node_count", minimum=1)
    element_count = _strict_int(topology["element_count"], label="manifest.topology.element_count", minimum=1)
    element_types = topology["element_types"]
    region_keys_value = topology["region_keys"]
    if type(element_types) is not list or any(type(item) is not str or not item for item in element_types):
        raise ResultArchiveDecodeError("manifest.topology.element_types must be a string array")
    if len(set(element_types)) != len(element_types):
        raise ResultArchiveDecodeError("manifest.topology.element_types contains duplicates")
    if type(region_keys_value) is not list or any(type(item) is not str for item in region_keys_value):
        raise ResultArchiveDecodeError("manifest.topology.region_keys must be a string array")
    try:
        regions = tuple(decode_result_region_key(item) for item in region_keys_value)
    except (TypeError, ValueError) as error:
        raise ResultArchiveDecodeError(f"invalid topology region dictionary: {error}") from error
    if len(set(regions)) != len(regions):
        raise ResultArchiveDecodeError("topology region dictionary contains duplicates")
    if regions != tuple(sorted(regions, key=encode_result_region_key)):
        raise ResultArchiveDecodeError(
            "topology region dictionary must be in canonical order"
        )
    entries = topology["arrays"]
    if type(entries) is not dict:
        raise ResultArchiveDecodeError("manifest.topology.arrays must be an object")
    _exact_keys(entries, {"node_ids", "node_coordinates", "nodal_displacements", "element_ids", "connectivity_offsets", "connectivity", "element_type_indices", "region_indices"}, label="manifest.topology.arrays")
    node_ids = _array_from_manifest(arrays, manifest["arrays"], entries["node_ids"], label="topology.node_ids", expected_dtype="<i8", expected_shape=(node_count,))
    coordinates = _array_from_manifest(arrays, manifest["arrays"], entries["node_coordinates"], label="topology.node_coordinates", expected_dtype="<f8", expected_shape=(node_count, 3), finite=True)
    displacements = _array_from_manifest(arrays, manifest["arrays"], entries["nodal_displacements"], label="topology.nodal_displacements", expected_dtype="<f8", expected_shape=(node_count, 3), finite=True)
    element_ids = _array_from_manifest(arrays, manifest["arrays"], entries["element_ids"], label="topology.element_ids", expected_dtype="<i8", expected_shape=(element_count,))
    offsets = _array_from_manifest(arrays, manifest["arrays"], entries["connectivity_offsets"], label="topology.connectivity_offsets", expected_dtype="<i8", expected_shape=(element_count + 1,))
    connectivity_values_meta = _checked_array_meta(manifest["arrays"].get(entries["connectivity"], {}), label="topology.connectivity") if entries["connectivity"] in manifest["arrays"] else None
    if connectivity_values_meta is None:
        raise ResultArchiveDecodeError("topology.connectivity entry is missing")
    connectivity = _array_from_manifest(arrays, manifest["arrays"], entries["connectivity"], label="topology.connectivity", expected_dtype="<i8", expected_shape=tuple(connectivity_values_meta["shape"]))
    type_indices = _array_from_manifest(arrays, manifest["arrays"], entries["element_type_indices"], label="topology.element_type_indices", expected_dtype="<i8", expected_shape=(element_count,))
    region_indices = _array_from_manifest(arrays, manifest["arrays"], entries["region_indices"], label="topology.region_indices", expected_dtype="<i8", expected_shape=(element_count,))
    if offsets[0] != 0 or np.any(offsets[1:] < offsets[:-1]) or offsets[-1] != len(connectivity):
        raise ResultArchiveDecodeError("topology connectivity offsets are invalid")
    if np.any(type_indices < 0) or np.any(type_indices >= len(element_types)):
        raise ResultArchiveDecodeError("topology element type index is out of range")
    first_seen_types = tuple(dict.fromkeys(int(item) for item in type_indices))
    if first_seen_types != tuple(range(len(element_types))):
        raise ResultArchiveDecodeError(
            "topology element type dictionary must be fully used in first-seen order"
        )
    if np.any(region_indices < 0) or np.any(region_indices >= len(regions)):
        raise ResultArchiveDecodeError("topology region index is out of range")
    rows = tuple(tuple(int(item) for item in connectivity[int(offsets[i]): int(offsets[i + 1])]) for i in range(element_count))
    try:
        return ResultTopologyProjection(
            source=source,
            node_ids=tuple(int(item) for item in node_ids),
            node_coordinates=coordinates,
            nodal_displacements=displacements,
            element_ids=tuple(int(item) for item in element_ids),
            element_types=tuple(element_types[int(item)] for item in type_indices),
            connectivity=rows,
            element_region_keys=tuple(regions[int(item)] for item in region_indices),
        )
    except (TypeError, ValueError) as error:
        raise ResultArchiveDecodeError(f"invalid topology: {error}") from error


def _decode_field(
    value: object,
    manifest: Mapping[str, object],
    arrays: Mapping[str, bytes],
    source: ResultSourceKey,
    regions: tuple[Any, ...],
) -> FieldData:
    if type(value) is not dict:
        raise ResultArchiveDecodeError("manifest.fields entries must be objects")
    _exact_keys(value, {"id", "key", "descriptor", "location_rows", "arrays"}, label="manifest.fields[]")
    field_id = _strict_string(value["id"], label="manifest.fields[].id")
    if not _FIELD_ID_RE.fullmatch(field_id):
        raise ResultArchiveDecodeError("manifest field id is not canonical")
    key = _key_from_json(value["key"], label=f"field[{field_id}].key")
    if stable_field_id(key) != field_id:
        raise ResultArchiveDecodeError(f"field id {field_id!r} does not match typed key")
    descriptor = _descriptor_from_json(value["descriptor"], label=f"field[{field_id}].descriptor")
    rows = _strict_int(value["location_rows"], label=f"field[{field_id}].location_rows", minimum=0)
    entry = value["arrays"]
    if type(entry) is not dict:
        raise ResultArchiveDecodeError(f"field[{field_id}].arrays must be an object")
    base_array_keys = set(_FIELD_ARRAY_ORDER[:11])
    point_array_keys = set(_FIELD_ARRAY_ORDER[11:])
    if set(entry) not in (base_array_keys, base_array_keys | point_array_keys):
        raise ResultArchiveDecodeError(
            f"field[{field_id}].arrays has unexpected object keys"
        )
    field_values = _array_from_manifest(arrays, manifest["arrays"], entry["values"], label=f"field[{field_id}].values", expected_dtype="<f8", expected_shape=(rows, len(descriptor.columns)), finite=True)
    coordinates = _array_from_manifest(arrays, manifest["arrays"], entry["coordinates"], label=f"field[{field_id}].coordinates", expected_dtype="<f8", expected_shape=(rows, 3), finite=True)
    displacement = _array_from_manifest(arrays, manifest["arrays"], entry["displacement"], label=f"field[{field_id}].displacement", expected_dtype="<f8", expected_shape=(rows, 3), finite=True)
    displacement_mask = _array_from_manifest(arrays, manifest["arrays"], entry["displacement_mask"], label=f"field[{field_id}].displacement_mask", expected_dtype="<u1", expected_shape=(rows,))
    id_arrays = {
        name: _array_from_manifest(arrays, manifest["arrays"], entry[name], label=f"field[{field_id}].{name}", expected_dtype="<i8", expected_shape=(rows,))
        for name in ("node_id", "element_id", "integration_point", "local_node", "region_index")
    }
    averaged = _array_from_manifest(arrays, manifest["arrays"], entry["averaged"], label=f"field[{field_id}].averaged", expected_dtype="<u1", expected_shape=(rows,))
    averaged_mask = _array_from_manifest(arrays, manifest["arrays"], entry["averaged_mask"], label=f"field[{field_id}].averaged_mask", expected_dtype="<u1", expected_shape=(rows,))
    point_numbers = point_local_y = point_local_z = None
    if point_array_keys.issubset(entry):
        point_numbers = _array_from_manifest(
            arrays,
            manifest["arrays"],
            entry["section_point_number"],
            label=f"field[{field_id}].section_point_number",
            expected_dtype="<i8",
            expected_shape=(rows,),
        )
        point_local_y = _array_from_manifest(
            arrays,
            manifest["arrays"],
            entry["section_point_local_y"],
            label=f"field[{field_id}].section_point_local_y",
            expected_dtype="<f8",
            expected_shape=(rows,),
            finite=True,
        )
        point_local_z = _array_from_manifest(
            arrays,
            manifest["arrays"],
            entry["section_point_local_z"],
            label=f"field[{field_id}].section_point_local_z",
            expected_dtype="<f8",
            expected_shape=(rows,),
            finite=True,
        )
    for name, mask in (("displacement_mask", displacement_mask), ("averaged_mask", averaged_mask), ("averaged", averaged)):
        if np.any(mask > 1):
            raise ResultArchiveDecodeError(f"field[{field_id}].{name} contains invalid mask value")
    locations: list[FieldLocation] = []
    for row in range(rows):
        region_index = int(id_arrays["region_index"][row])
        if region_index < -1 or region_index >= len(regions):
            raise ResultArchiveDecodeError(f"field[{field_id}] region index is out of range")
        try:
            locations.append(
                FieldLocation(
                    association=descriptor.association,
                    coordinates=tuple(float(item) for item in coordinates[row]),
                    displacement=(tuple(float(item) for item in displacement[row]) if displacement_mask[row] else None),
                    node_id=_optional_positive_int(id_arrays["node_id"][row]),
                    element_id=_optional_positive_int(id_arrays["element_id"][row]),
                    integration_point=_optional_positive_int(id_arrays["integration_point"][row]),
                    local_node=_optional_positive_int(id_arrays["local_node"][row]),
                    region_key=None if region_index < 0 else regions[region_index],
                    averaged=(None if not averaged_mask[row] else bool(averaged[row])),
                    section_point=(
                        None
                        if point_numbers is None
                        else BeamSectionPoint(
                            int(point_numbers[row]),
                            float(point_local_y[row]),
                            float(point_local_z[row]),
                        )
                    ),
                )
            )
        except (TypeError, ValueError) as error:
            raise ResultArchiveDecodeError(f"invalid field[{field_id}] location {row}: {error}") from error
    try:
        return FieldData(descriptor=descriptor, source=source, key=key, locations=tuple(locations), values=field_values)
    except (TypeError, ValueError) as error:
        raise ResultArchiveDecodeError(f"invalid field[{field_id}]: {error}") from error


def _validate_array_references(manifest: Mapping[str, object]) -> None:
    """Require every declared array entry to have exactly one owner slot."""

    references: list[str] = []
    topology = manifest["topology"]
    if type(topology) is not dict or type(topology.get("arrays")) is not dict:
        raise ResultArchiveDecodeError("manifest.topology.arrays must be an object")
    references.extend(topology["arrays"].values())
    fields = manifest["fields"]
    if type(fields) is not list:
        raise ResultArchiveDecodeError("manifest.fields must be an array")
    for index, field in enumerate(fields):
        if type(field) is not dict or type(field.get("arrays")) is not dict:
            raise ResultArchiveDecodeError(f"manifest.fields[{index}].arrays must be an object")
        references.extend(field["arrays"].values())
    if any(type(name) is not str for name in references):
        raise ResultArchiveDecodeError("array references must be strings")
    if len(references) != len(set(references)):
        raise ResultArchiveDecodeError("manifest contains duplicate array references")
    declared = set(manifest["arrays"])
    referenced = set(references)
    if declared != referenced:
        raise ResultArchiveDecodeError("manifest array entries are not exactly referenced")


def _validate_array_entry_names(manifest: Mapping[str, object]) -> None:
    topology = manifest["topology"]
    if type(topology) is not dict or type(topology.get("arrays")) is not dict:
        raise ResultArchiveDecodeError("manifest.topology.arrays must be an object")
    expected_topology = {
        "node_ids": "topology/node_ids.npy",
        "node_coordinates": "topology/node_coordinates.npy",
        "nodal_displacements": "topology/nodal_displacements.npy",
        "element_ids": "topology/element_ids.npy",
        "connectivity_offsets": "topology/connectivity_offsets.npy",
        "connectivity": "topology/connectivity.npy",
        "element_type_indices": "topology/element_type_indices.npy",
        "region_indices": "topology/region_indices.npy",
    }
    for key, expected in expected_topology.items():
        if topology["arrays"].get(key) != expected:
            raise ResultArchiveDecodeError(f"topology array entry {key!r} is not canonical")
    expected_field_suffixes = {
        "values": "values.npy",
        "coordinates": "locations-coordinates.npy",
        "displacement": "locations-displacement.npy",
        "displacement_mask": "locations-displacement-mask.npy",
        "node_id": "locations-node-ids.npy",
        "element_id": "locations-element-ids.npy",
        "integration_point": "locations-integration-points.npy",
        "local_node": "locations-local-nodes.npy",
        "region_index": "locations-region-indices.npy",
        "averaged": "locations-averaged.npy",
        "averaged_mask": "locations-averaged-mask.npy",
    }
    fields = manifest["fields"]
    if type(fields) is not list:
        raise ResultArchiveDecodeError("manifest.fields must be an array")
    for index, field in enumerate(fields):
        if type(field) is not dict:
            raise ResultArchiveDecodeError(f"manifest.fields[{index}] must be an object")
        field_id = field.get("id")
        arrays = field.get("arrays")
        if type(field_id) is not str or type(arrays) is not dict:
            raise ResultArchiveDecodeError(f"manifest.fields[{index}] has invalid id/arrays")
        for key, suffix in expected_field_suffixes.items():
            if arrays.get(key) != f"fields/{field_id}/{suffix}":
                raise ResultArchiveDecodeError(f"field array entry {field_id!r}/{key!r} is not canonical")


def _validate_region_dictionary(
    manifest: Mapping[str, object],
    topology: ResultTopologyProjection,
    fields: tuple[FieldData, ...],
) -> None:
    regions = _decode_regions(manifest["topology"])
    used = set(topology.element_region_keys)
    for field_data in fields:
        used.update(
            location.region_key
            for location in field_data.locations
            if location.region_key is not None
        )
    if used != set(regions):
        raise ResultArchiveDecodeError(
            "topology region dictionary must contain exactly the used regions"
        )


def _optional_positive_int(value: object) -> int | None:
    integer = int(value)
    return None if integer == -1 else integer


def decode_result_archive_v1(data: bytes) -> ResultArchiveSnapshot:
    """Decode and strictly validate one schema-v1 archive byte sequence."""

    if type(data) is not bytes:
        raise TypeError("archive data must be bytes")
    _validate_container_bytes(len(data), error_type=ResultArchiveDecodeError)
    manifest, arrays = _open_archive_bytes(data)
    if manifest["format"] != FORMAT_NAME:
        raise ResultArchiveDecodeError("manifest.format is not fem-python-result")
    schema = manifest["schema"]
    if type(schema) is not int:
        raise ResultArchiveDecodeError("manifest.schema must be a strict integer")
    if schema != SCHEMA_VERSION:
        raise UnsupportedResultArchiveSchemaError(f"unsupported result archive schema {schema!r}")
    source = _source_from_json(manifest["source"], label="manifest.source")
    origin = _decode_origin(manifest["origin"])
    run = _decode_run(manifest["run"])
    profile = _profile_from_json(manifest["profile"], label="manifest.profile")
    catalog = _catalog_from_json(manifest["catalog"], label="manifest.catalog")
    if catalog.source != source:
        raise ResultArchiveDecodeError("catalog source does not match manifest.source")
    _validate_profile_bound_catalog(profile, catalog)
    topology = _decode_topology(manifest, arrays, source)
    fields_value = manifest["fields"]
    if type(fields_value) is not list:
        raise ResultArchiveDecodeError("manifest.fields must be an array")
    _validate_array_entry_names(manifest)
    _validate_array_references(manifest)
    fields = tuple(_decode_field(item, manifest, arrays, source, tuple(_decode_regions(manifest["topology"]))) for item in fields_value)
    _validate_profile_bound_fields(profile, fields)
    _validate_region_dictionary(manifest, topology, fields)
    if tuple(sorted(fields, key=lambda item: _field_sort(item.key))) != fields:
        raise ResultArchiveDecodeError("materialized fields are not in canonical order")
    materialization_generation = _strict_int(manifest["materialization_generation"], label="manifest.materialization_generation", minimum=0)
    try:
        materialization = ResultMaterializationSnapshot(source=source, generation=materialization_generation, topology=topology, fields=fields)
    except (TypeError, ValueError) as error:
        raise ResultArchiveDecodeError(f"invalid materialization: {error}") from error
    model_projection_value = manifest["model_projection"]
    if type(model_projection_value) is not dict:
        raise ResultArchiveDecodeError("manifest.model_projection must be an object")
    _exact_keys(model_projection_value, {"unit_context", "named_region_node_ids", "named_region_element_ids", "summaries"}, label="manifest.model_projection")
    if type(manifest["inspection_summaries"]) is not dict:
        raise ResultArchiveDecodeError("manifest.inspection_summaries must be an object")
    if manifest["inspection_summaries"] != model_projection_value["summaries"]:
        raise ResultArchiveDecodeError("inspection summaries do not match model projection")
    for key in ("named_region_node_ids", "named_region_element_ids", "summaries"):
        if type(model_projection_value[key]) is not dict:
            raise ResultArchiveDecodeError(f"manifest.model_projection.{key} must be an object")
    try:
        projection = ResultArchiveModelProjection(
            topology=topology,
            unit_context=_unit_from_json(model_projection_value["unit_context"], label="manifest.model_projection.unit_context"),
            named_region_node_ids=_decode_named_region_mapping(
                model_projection_value["named_region_node_ids"],
                label="manifest.model_projection.named_region_node_ids",
                allowed_ids=frozenset(topology.node_ids),
            ),
            named_region_element_ids=_decode_named_region_mapping(
                model_projection_value["named_region_element_ids"],
                label="manifest.model_projection.named_region_element_ids",
                allowed_ids=frozenset(topology.element_ids),
            ),
            summaries=model_projection_value["summaries"],
        )
        snapshot = ResultArchiveSnapshot(
            archive_id=_strict_string(manifest["archive_id"], label="manifest.archive_id"),
            created_at=_datetime_from_json(manifest["created_at"], label="manifest.created_at"),
            producer_version=_strict_string(manifest["producer_version"], label="manifest.producer_version"),
            origin=origin,
            run=run,
            profile=profile,
            catalog=catalog,
            materialization=materialization,
            model_projection=projection,
            unit_context=_unit_from_json(manifest["unit_context"], label="manifest.unit_context"),
        )
    except (OverflowError, RecursionError, TypeError, ValueError) as error:
        raise ResultArchiveDecodeError(f"invalid result archive snapshot: {error}") from error
    # Ensure all READY catalog fields have exactly one materialized field and
    # no materialized field is hidden from the catalog.
    ready = {item.key for item in catalog.fields if item.state is FieldState.READY}
    materialized = {item.key for item in fields}
    if ready != materialized:
        raise ResultArchiveDecodeError(
            "catalog READY keys must exactly match materialized field keys"
        )
    return snapshot


def _validate_profile_bound_catalog(
    profile: ElementResultProfile,
    catalog: ResultCatalog,
) -> None:
    for item in catalog.fields:
        try:
            expected = registry_entry_for(profile, item.key.request.field_id)
        except (KeyError, TypeError, ValueError) as error:
            raise ResultArchiveDecodeError("catalog field is outside the profile registry") from error
        if (
            item.descriptor != expected.descriptor
            and not _is_legacy_beam_section_field(
                profile,
                item.key,
                item.descriptor,
            )
        ):
            raise ResultArchiveDecodeError("catalog descriptor does not match profile registry")


def _validate_profile_bound_fields(
    profile: ElementResultProfile,
    fields: tuple[FieldData, ...],
) -> None:
    for field_data in fields:
        try:
            expected = registry_entry_for(profile, field_data.key.request.field_id)
        except (KeyError, TypeError, ValueError) as error:
            raise ResultArchiveDecodeError("materialized field is outside the profile registry") from error
        if (
            field_data.descriptor != expected.descriptor
            and not _is_legacy_beam_section_field(
                profile,
                field_data.key,
                field_data.descriptor,
            )
        ):
            raise ResultArchiveDecodeError("materialized descriptor does not match profile registry")


def _is_legacy_beam_section_field(
    profile: ElementResultProfile,
    key: FieldMaterializationKey,
    descriptor: FieldDescriptor,
) -> bool:
    """Recognize schema-v1 Beam extrema without inventing point rows."""

    return (
        profile.family is ResultModelFamily.BEAM
        and key.recovery_contract == 1
        and key.request.field_id
        == ResultFieldId(ResultVariable.S, FieldPosition.SECTION_END)
        and descriptor.field_id == key.request.field_id
        and descriptor.association is FieldAssociation.ELEMENT_NODE
        and descriptor.quantity is PhysicalQuantity.STRESS
        and descriptor.components == ("S11Max", "S11Min")
        and descriptor.derived_components == ("S11AbsMax",)
    )


def _decode_regions(topology: object) -> tuple[Any, ...]:
    if type(topology) is not dict or type(topology.get("region_keys")) is not list:
        raise ResultArchiveDecodeError("manifest.topology.region_keys must be an array")
    try:
        return tuple(decode_result_region_key(item) for item in topology["region_keys"])
    except (TypeError, ValueError) as error:
        raise ResultArchiveDecodeError(f"invalid region dictionary: {error}") from error


def _field_sort(key: FieldMaterializationKey) -> tuple[object, ...]:
    from fem.application.results.fields import field_materialization_sort_key

    return field_materialization_sort_key(key)


def dumps_result_archive_v1(snapshot: ResultArchiveSnapshot) -> bytes:
    return encode_result_archive_v1(snapshot)


def loads_result_archive_v1(data: bytes | bytearray) -> ResultArchiveSnapshot:
    if type(data) not in {bytes, bytearray}:
        raise TypeError("archive data must be bytes or bytearray")
    if len(data) > _MAX_ZIP_CONTAINER_BYTES:
        raise ResultArchiveDecodeError(
            "result archive container exceeds the size safety limit"
        )
    return decode_result_archive_v1(bytes(data))


def save_result_archive_v1(
    path: str | Path,
    snapshot: ResultArchiveSnapshot,
    *,
    checkpoint: Any | None = None,
    before_replace: Any | None = None,
) -> Path:
    """Atomically save and readback-verify one archive snapshot."""

    if checkpoint is not None:
        checkpoint()
    serialized = encode_result_archive_v1(snapshot)
    if checkpoint is not None:
        checkpoint()
    expected = hashlib.sha256(serialized).hexdigest()

    def verifier(temporary: Path) -> ResultArchiveSnapshot:
        return decode_result_archive_v1(
            _read_path_bounded(temporary, _MAX_ZIP_CONTAINER_BYTES)
        )

    def semantic(snapshot_value: ResultArchiveSnapshot) -> str:
        return hashlib.sha256(encode_result_archive_v1(snapshot_value)).hexdigest()

    try:
        return atomic_write_verified_binary(
            path,
            serialized,
            verifier=verifier,
            semantic_encoder=semantic,
            expected_semantic=expected,
            error_type=ResultArchiveEncodeError,
            checkpoint=checkpoint,
            before_replace=before_replace,
        )
    except ResultArchiveEncodeError:
        raise
    except (OSError, ValueError, TypeError) as error:
        raise ResultArchiveEncodeError(f"result archive save failed: {error}") from error


def load_result_archive_v1(path: str | Path) -> LoadedResultArchive:
    source = Path(path)
    try:
        snapshot = decode_result_archive_v1(
            _read_path_bounded(source, _MAX_ZIP_CONTAINER_BYTES)
        )
    except ResultArchiveDecodeError:
        raise
    except (OSError, ValueError, TypeError) as error:
        raise ResultArchiveDecodeError(f"result archive load failed: {error}") from error
    return LoadedResultArchive(snapshot=snapshot, path=source, source_schema=SCHEMA_VERSION)


def _read_path_bounded(path: Path, limit: int) -> bytes:
    declared_size = path.stat().st_size
    if declared_size > limit:
        raise ResultArchiveDecodeError(
            "result archive container exceeds the size safety limit"
        )
    with path.open("rb") as stream:
        data = stream.read(declared_size + 1)
    if len(data) != declared_size:
        raise ResultArchiveDecodeError(
            "result archive container size changed during read"
        )
    return data


read_result_archive_v1 = load_result_archive_v1
write_result_archive_v1 = save_result_archive_v1


__all__ = [
    "FORMAT_NAME",
    "MANIFEST_NAME",
    "RESULT_ARCHIVE_FILE_SUFFIX",
    "RESULT_ARCHIVE_FORMAT_NAME",
    "RESULT_ARCHIVE_SCHEMA_VERSION",
    "RESULT_FILE_SUFFIX",
    "SCHEMA_VERSION",
    "decode_result_archive_v1",
    "dumps_result_archive_v1",
    "encode_result_archive_v1",
    "inspect_result_archive_header_bytes",
    "inspect_result_archive_header_path",
    "load_result_archive_v1",
    "loads_result_archive_v1",
    "read_result_archive_v1",
    "save_result_archive_v1",
    "stable_field_id",
    "write_result_archive_v1",
]
