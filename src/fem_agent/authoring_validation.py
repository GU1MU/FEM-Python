"""Strict authoring tool argument validation without workflow state."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
import math
import re
from typing import Any

from .providers.base import ToolDefinition


class _AuthoringArgumentSchemaError(ValueError):
    def __init__(
        self,
        path: str,
        reason: str,
        *,
        allowed_values: Sequence[object] = (),
    ) -> None:
        self.path = str(path)
        self.reason = str(reason)
        self.allowed_values = tuple(allowed_values)
        super().__init__(f"{self.path}: {self.reason}")


def _schema_type_matches(value: object, expected: str) -> bool:
    return {
        "object": isinstance(value, Mapping),
        "array": isinstance(value, Sequence)
        and not isinstance(value, (str, bytes, bytearray)),
        "string": type(value) is str,
        "number": isinstance(value, (int, float)) and not isinstance(value, bool),
        "integer": type(value) is int,
        "boolean": type(value) is bool,
        "null": value is None,
    }.get(expected, True)


def _schema_discriminator_branch(
    value: object,
    branches: Sequence[Mapping[str, object]],
    path: str,
) -> Mapping[str, object] | None:
    if not isinstance(value, Mapping):
        return None
    for field in ("operation", "kind"):
        raw_value = value.get(field)
        choices: dict[object, Mapping[str, object]] = {}
        for branch in branches:
            properties = branch.get("properties")
            field_schema = (
                properties.get(field) if isinstance(properties, Mapping) else None
            )
            constant = (
                field_schema.get("const")
                if isinstance(field_schema, Mapping)
                else None
            )
            if constant is not None:
                choices[constant] = branch
        if choices:
            if field not in value:
                raise _AuthoringArgumentSchemaError(
                    f"{path}.{field}",
                    "required field is missing",
                    allowed_values=tuple(choices),
                )
            if raw_value not in choices:
                raise _AuthoringArgumentSchemaError(
                    f"{path}.{field}",
                    f"unsupported value {raw_value!r}",
                    allowed_values=tuple(choices),
                )
            return choices[raw_value]
    return None


def _validate_schema_value(
    value: object,
    schema: Mapping[str, object],
    path: str,
) -> None:
    raw_branches = schema.get("oneOf")
    if isinstance(raw_branches, Sequence):
        branches = tuple(
            item for item in raw_branches if isinstance(item, Mapping)
        )
        selected = _schema_discriminator_branch(value, branches, path)
        if selected is not None:
            _validate_schema_value(value, selected, path)
            return
        matched = 0
        first_error: _AuthoringArgumentSchemaError | None = None
        for branch in branches:
            try:
                _validate_schema_value(value, branch, path)
            except _AuthoringArgumentSchemaError as error:
                if first_error is None:
                    first_error = error
            else:
                matched += 1
        if matched != 1:
            if first_error is not None:
                raise first_error
            raise _AuthoringArgumentSchemaError(path, "must match exactly one schema")
        return

    expected_type = schema.get("type")
    if isinstance(expected_type, str) and not _schema_type_matches(value, expected_type):
        raise _AuthoringArgumentSchemaError(
            path,
            f"expected {expected_type}, got {type(value).__name__}",
        )
    if "const" in schema and value != schema["const"]:
        raise _AuthoringArgumentSchemaError(
            path,
            f"must equal {schema['const']!r}",
            allowed_values=(schema["const"],),
        )
    raw_enum = schema.get("enum")
    if isinstance(raw_enum, Sequence) and value not in raw_enum:
        raise _AuthoringArgumentSchemaError(
            path,
            f"unsupported value {value!r}",
            allowed_values=tuple(raw_enum),
        )

    if isinstance(value, Mapping):
        properties = schema.get("properties")
        declared = properties if isinstance(properties, Mapping) else {}
        required = schema.get("required")
        required_fields = tuple(required) if isinstance(required, Sequence) else ()
        missing = [field for field in required_fields if field not in value]
        if missing:
            field = str(missing[0])
            raise _AuthoringArgumentSchemaError(
                f"{path}.{field}",
                "required field is missing",
            )
        if schema.get("additionalProperties") is False:
            extras = [field for field in value if field not in declared]
            if extras:
                field = str(extras[0])
                raise _AuthoringArgumentSchemaError(
                    f"{path}.{field}",
                    "field is not allowed",
                    allowed_values=tuple(declared),
                )
        minimum_properties = schema.get("minProperties")
        if type(minimum_properties) is int and len(value) < minimum_properties:
            raise _AuthoringArgumentSchemaError(
                path,
                f"requires at least {minimum_properties} fields",
            )
        for field, item in value.items():
            field_schema = declared.get(field)
            if isinstance(field_schema, Mapping):
                _validate_schema_value(item, field_schema, f"{path}.{field}")

    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        minimum_items = schema.get("minItems")
        maximum_items = schema.get("maxItems")
        if type(minimum_items) is int and len(value) < minimum_items:
            raise _AuthoringArgumentSchemaError(
                path,
                f"requires at least {minimum_items} items",
            )
        if type(maximum_items) is int and len(value) > maximum_items:
            raise _AuthoringArgumentSchemaError(
                path,
                f"allows at most {maximum_items} items",
            )
        if schema.get("uniqueItems") is True:
            canonical = [
                json.dumps(item, sort_keys=True, separators=(",", ":"))
                for item in value
            ]
            if len(canonical) != len(set(canonical)):
                raise _AuthoringArgumentSchemaError(path, "items must be unique")
        item_schema = schema.get("items")
        if isinstance(item_schema, Mapping):
            for index, item in enumerate(value):
                _validate_schema_value(item, item_schema, f"{path}[{index}]")

    if type(value) is str:
        minimum_length = schema.get("minLength")
        maximum_length = schema.get("maxLength")
        if type(minimum_length) is int and len(value) < minimum_length:
            raise _AuthoringArgumentSchemaError(
                path,
                f"requires at least {minimum_length} characters",
            )
        if type(maximum_length) is int and len(value) > maximum_length:
            raise _AuthoringArgumentSchemaError(
                path,
                f"allows at most {maximum_length} characters",
            )
        pattern = schema.get("pattern")
        if isinstance(pattern, str) and re.search(pattern, value) is None:
            raise _AuthoringArgumentSchemaError(
                path,
                f"must match pattern {pattern}",
            )

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if not math.isfinite(float(value)):
            raise _AuthoringArgumentSchemaError(path, "number must be finite")
        minimum = schema.get("minimum")
        maximum = schema.get("maximum")
        exclusive_minimum = schema.get("exclusiveMinimum")
        exclusive_maximum = schema.get("exclusiveMaximum")
        if isinstance(minimum, (int, float)) and value < minimum:
            raise _AuthoringArgumentSchemaError(path, f"must be >= {minimum}")
        if isinstance(maximum, (int, float)) and value > maximum:
            raise _AuthoringArgumentSchemaError(path, f"must be <= {maximum}")
        if isinstance(exclusive_minimum, (int, float)) and value <= exclusive_minimum:
            raise _AuthoringArgumentSchemaError(
                path,
                f"must be > {exclusive_minimum}",
            )
        if isinstance(exclusive_maximum, (int, float)) and value >= exclusive_maximum:
            raise _AuthoringArgumentSchemaError(
                path,
                f"must be < {exclusive_maximum}",
            )


def _validate_geometry_edit_arguments(
    arguments: Mapping[str, object],
    definition: ToolDefinition,
) -> None:
    _validate_schema_value(arguments, definition.parameters, "arguments")


def _require_exact_fields(
    value: Mapping[str, Any],
    required: set[str],
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("tool arguments must be an object")
    data = dict(value)
    if set(data) != required:
        raise ValueError("tool argument fields do not match the strict schema")
    return data


def _nonblank_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"{label} must be a non-blank string")
    return value.strip()


def _validate_requirement_value(
    key: str,
    value: object,
    spec: Mapping[str, object],
) -> None:
    expected = spec["type"]
    if expected == "string":
        normalized = _nonblank_string(value, key)
        if "enum" in spec and normalized not in spec["enum"]:
            raise ValueError(f"{key} is outside the supported values")
    elif expected == "number":
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise TypeError(f"{key} must be a finite number")
        numeric = float(value)
        if "exclusiveMinimum" in spec and numeric <= float(spec["exclusiveMinimum"]):
            raise ValueError(f"{key} is below its strict lower bound")
        if "exclusiveMaximum" in spec and numeric >= float(spec["exclusiveMaximum"]):
            raise ValueError(f"{key} is above its strict upper bound")
    elif expected == "integer":
        if isinstance(value, bool) or type(value) is not int:
            raise TypeError(f"{key} must be an integer")
        if "minimum" in spec and value < int(spec["minimum"]):
            raise ValueError(f"{key} is below its lower bound")
        if "maximum" in spec and value > int(spec["maximum"]):
            raise ValueError(f"{key} is above its upper bound")
        if "enum" in spec and value not in spec["enum"]:
            raise ValueError(f"{key} is outside the supported values")
    elif expected == "boolean":
        if type(value) is not bool:
            raise TypeError(f"{key} must be a boolean")
        if "enum" in spec and value not in spec["enum"]:
            raise ValueError(f"{key} is outside the supported values")
    elif expected == "array":
        if not isinstance(value, list):
            raise TypeError(f"{key} must be an array")
        if not int(spec["minItems"]) <= len(value) <= int(spec["maxItems"]):
            raise ValueError(f"{key} has an unsupported number of values")
        if len({json.dumps(item, sort_keys=True) for item in value}) != len(value):
            raise ValueError(f"{key} must contain unique values")
        item_spec = spec["items"]
        assert isinstance(item_spec, Mapping)
        for item in value:
            _validate_requirement_value(key, item, item_spec)
    else:
        raise RuntimeError("unsupported requirement schema")
