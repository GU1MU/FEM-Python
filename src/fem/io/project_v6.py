"""Schema-v6 persistence for strict planar Boolean features."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from fem.application.session import ProjectSaveSnapshot, ProjectSnapshot

from ._project_codec import (
    ProjectFieldCodecPolicy,
    atomic_write_project,
    dumps_canonical_json,
    loads_json_strict,
)
from ._project_errors import ProjectDecodeError, ProjectEncodeError, ProjectError
from .project_v5 import decode_project_v5, encode_project_v5


SCHEMA_VERSION = 6
FORMAT_NAME = "fem-python-project"


class ProjectV6Error(ProjectError):
    """Base error for schema-v6 project processing."""


class ProjectV6DecodeError(ProjectV6Error, ProjectDecodeError):
    """A schema-v6 payload is malformed or incompatible."""


class ProjectV6EncodeError(ProjectV6Error, ProjectEncodeError):
    """A snapshot cannot be represented losslessly by schema v6."""


_V6_FIELD_POLICY = ProjectFieldCodecPolicy(
    version_label="v6",
    decode_error=ProjectV6DecodeError,
    encode_error=ProjectV6EncodeError,
    require_current_fields=True,
    assignment_orientation=True,
    allow_wire_geometry=True,
    allow_strict_sketch=True,
    extrusion_source_faces=True,
    displacement_region_targets=True,
    body_force_loads=True,
    allow_multi_body=True,
    allow_planar_boolean=True,
)


def loads_project_v6(
    data: str | bytes | bytearray,
    *,
    source_path: str | Path | None = None,
) -> ProjectSnapshot:
    payload = loads_json_strict(
        data,
        error_type=ProjectV6DecodeError,
        document_label="v6 项目",
    )
    return decode_project_v6(payload, source_path=source_path)


def decode_project_v6(
    payload: Mapping[str, Any] | str | bytes | bytearray,
    *,
    source_path: str | Path | None = None,
) -> ProjectSnapshot:
    """Decode canonical v6 through the shared strict current-field codec."""

    if isinstance(payload, (str, bytes, bytearray)):
        return loads_project_v6(payload, source_path=source_path)
    if not isinstance(payload, Mapping):
        raise TypeError("decode_project_v6 requires a mapping or JSON data")
    if payload.get("schema") != SCHEMA_VERSION:
        raise ProjectV6DecodeError(
            f"v6 decoder 不能读取 schema {payload.get('schema')!r}"
        )
    compatible = dict(payload)
    compatible["schema"] = 5
    try:
        return decode_project_v5(
            compatible,
            source_path=source_path,
            _field_policy=_V6_FIELD_POLICY,
        )
    except Exception as error:
        if isinstance(error, ProjectV6DecodeError):
            raise
        raise ProjectV6DecodeError(str(error)) from error


def encode_project_v6(
    snapshot: ProjectSnapshot | ProjectSaveSnapshot,
) -> dict[str, Any]:
    """Encode canonical current authoring and mark it as schema v6."""

    try:
        payload = encode_project_v5(
            snapshot,
            _field_policy=_V6_FIELD_POLICY,
        )
    except Exception as error:
        if isinstance(error, ProjectV6EncodeError):
            raise
        raise ProjectV6EncodeError(str(error)) from error
    payload["schema"] = SCHEMA_VERSION
    return payload


def dumps_project_v6(
    snapshot: ProjectSnapshot | ProjectSaveSnapshot,
) -> str:
    return dumps_canonical_json(
        encode_project_v6(snapshot),
        error_type=ProjectV6EncodeError,
    )


def load_project_v6(path: str | Path) -> ProjectSnapshot:
    source = Path(path)
    return loads_project_v6(source.read_bytes(), source_path=source)


def save_project_v6(
    path: str | Path,
    snapshot: ProjectSnapshot | ProjectSaveSnapshot,
    *,
    checkpoint: Callable[[], Any] | None = None,
) -> Path:
    if checkpoint is not None:
        checkpoint()
    payload = encode_project_v6(snapshot)
    if checkpoint is not None:
        checkpoint()
    serialized = dumps_canonical_json(
        payload,
        error_type=ProjectV6EncodeError,
    )
    return atomic_write_project(
        path,
        serialized,
        verifier=load_project_v6,
        semantic_encoder=encode_project_v6,
        expected_semantic=payload,
        error_type=ProjectV6EncodeError,
        mismatch_message=(
            "临时 v6 项目回读后的 canonical authoring 值与保存 snapshot 不一致"
        ),
        checkpoint=checkpoint,
    )


read_project_v6 = load_project_v6
write_project_v6 = save_project_v6


__all__ = [
    "FORMAT_NAME",
    "ProjectV6DecodeError",
    "ProjectV6EncodeError",
    "ProjectV6Error",
    "SCHEMA_VERSION",
    "decode_project_v6",
    "dumps_project_v6",
    "encode_project_v6",
    "load_project_v6",
    "loads_project_v6",
    "read_project_v6",
    "save_project_v6",
    "write_project_v6",
]
