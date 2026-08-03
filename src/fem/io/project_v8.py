"""Strict schema-v8 persistence for native-project unit context."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any

from fem.application.session import ProjectSaveSnapshot, ProjectSnapshot
from fem.application.units import UnitContext

from ._project_codec import (
    ProjectFieldCodecPolicy,
    atomic_write_project,
    dumps_canonical_json,
    loads_json_strict,
    unwrap_project_snapshot,
)
from ._project_errors import ProjectDecodeError, ProjectEncodeError, ProjectError
from .project_v7 import decode_project_v7, encode_project_v7


SCHEMA_VERSION = 8
FORMAT_NAME = "fem-python-project"


class ProjectV8Error(ProjectError):
    """Base error for schema-v8 project processing."""


class ProjectV8DecodeError(ProjectV8Error, ProjectDecodeError):
    """A schema-v8 payload is malformed or incompatible."""


class ProjectV8EncodeError(ProjectV8Error, ProjectEncodeError):
    """A snapshot cannot be represented losslessly by schema v8."""


def loads_project_v8(
    data: str | bytes | bytearray,
    *,
    source_path: str | Path | None = None,
) -> ProjectSnapshot:
    payload = loads_json_strict(
        data,
        error_type=ProjectV8DecodeError,
        document_label="v8 项目",
    )
    return decode_project_v8(payload, source_path=source_path)


def decode_project_v8(
    payload: Mapping[str, Any] | str | bytes | bytearray,
    *,
    source_path: str | Path | None = None,
    _field_policy: ProjectFieldCodecPolicy | None = None,
) -> ProjectSnapshot:
    """Decode v8 after removing its one additive field for the v7 core."""

    if isinstance(payload, (str, bytes, bytearray)):
        return loads_project_v8(payload, source_path=source_path)
    try:
        root = _mapping(payload, "$")
        _exact_keys(root, "$", {"format", "schema", "project"})
        if root["format"] != FORMAT_NAME:
            raise ProjectV8DecodeError(f"$.format 必须精确等于 {FORMAT_NAME!r}")
        if type(root["schema"]) is not int or root["schema"] != SCHEMA_VERSION:
            raise ProjectV8DecodeError(f"v8 decoder 不能读取 schema {root['schema']!r}")
        project = _mapping(root["project"], "$.project")
        _exact_keys(project, "$.project", {"kind", "authoring"})
        authoring = _mapping(project["authoring"], "$.project.authoring")
        v7_fields = {
            "model_name",
            "active_part_id",
            "parts",
            "retired_part_ids",
            "retired_part_boolean_feature_ids",
            "named_regions",
            "definitions",
            "part_boolean_undo_records",
        }
        _exact_keys(
            authoring,
            "$.project.authoring",
            {*v7_fields, "unit_context"},
        )
        units = _decode_unit_context(
            authoring["unit_context"],
            "$.project.authoring.unit_context",
        )
        v7_payload = deepcopy(dict(root))
        v7_payload["schema"] = 7
        v7_authoring = v7_payload["project"]["authoring"]
        del v7_authoring["unit_context"]
        decode_options = (
            {} if _field_policy is None else {"_field_policy": _field_policy}
        )
        snapshot = decode_project_v7(
            v7_payload,
            source_path=source_path,
            **decode_options,
        )
        return replace(snapshot, unit_context=units)
    except ProjectV8Error:
        raise
    except Exception as error:
        raise ProjectV8DecodeError(f"$.project.authoring 无效：{error}") from error


def encode_project_v8(
    snapshot: ProjectSnapshot | ProjectSaveSnapshot,
    *,
    _field_policy: ProjectFieldCodecPolicy | None = None,
) -> dict[str, Any]:
    """Encode canonical v7 authoring plus the typed unit convention."""

    try:
        project = unwrap_project_snapshot(snapshot)
        encode_options = (
            {} if _field_policy is None else {"_field_policy": _field_policy}
        )
        payload = encode_project_v7(project, **encode_options)
        payload["schema"] = SCHEMA_VERSION
        payload["project"]["authoring"]["unit_context"] = (
            None if project.unit_context is None else project.unit_context.to_dict()
        )
        dumps_canonical_json(payload, error_type=ProjectV8EncodeError)
        return payload
    except ProjectV8Error:
        raise
    except Exception as error:
        raise ProjectV8EncodeError(f"snapshot 无法由 v8 无损表示：{error}") from error


def _decode_unit_context(value: Any, path: str) -> UnitContext | None:
    if value is None:
        return None
    data = _mapping(value, path)
    fields = {
        "length",
        "force",
        "stress",
        "density",
        "acceleration",
        "convention",
    }
    _exact_keys(data, path, fields)
    try:
        return UnitContext.from_dict(dict(data))
    except (TypeError, ValueError) as error:
        raise ProjectV8DecodeError(f"{path} 无效：{error}") from error


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProjectV8DecodeError(f"{path} 必须是 object")
    return value


def _exact_keys(
    value: Mapping[str, Any],
    path: str,
    expected: set[str],
) -> None:
    actual = set(value)
    missing = expected - actual
    extra = actual - expected
    if missing:
        raise ProjectV8DecodeError(f"{path} 缺少字段 {sorted(missing)[0]!r}")
    if extra:
        raise ProjectV8DecodeError(f"{path} 包含未知字段 {sorted(extra)[0]!r}")


def dumps_project_v8(
    snapshot: ProjectSnapshot | ProjectSaveSnapshot,
) -> str:
    return dumps_canonical_json(
        encode_project_v8(snapshot),
        error_type=ProjectV8EncodeError,
    )


def load_project_v8(path: str | Path) -> ProjectSnapshot:
    source = Path(path)
    return loads_project_v8(source.read_bytes(), source_path=source)


def save_project_v8(
    path: str | Path,
    snapshot: ProjectSnapshot | ProjectSaveSnapshot,
    *,
    checkpoint: Callable[[], Any] | None = None,
) -> Path:
    if checkpoint is not None:
        checkpoint()
    payload = encode_project_v8(snapshot)
    if checkpoint is not None:
        checkpoint()
    serialized = dumps_canonical_json(
        payload,
        error_type=ProjectV8EncodeError,
    )
    return atomic_write_project(
        path,
        serialized,
        verifier=load_project_v8,
        semantic_encoder=encode_project_v8,
        expected_semantic=payload,
        error_type=ProjectV8EncodeError,
        mismatch_message=(
            "临时 v8 项目回读后的单位及 canonical Part authoring 与 snapshot 不一致"
        ),
        checkpoint=checkpoint,
    )


read_project_v8 = load_project_v8
write_project_v8 = save_project_v8


__all__ = [
    "FORMAT_NAME",
    "ProjectV8DecodeError",
    "ProjectV8EncodeError",
    "ProjectV8Error",
    "SCHEMA_VERSION",
    "decode_project_v8",
    "dumps_project_v8",
    "encode_project_v8",
    "load_project_v8",
    "loads_project_v8",
    "read_project_v8",
    "save_project_v8",
    "write_project_v8",
]
