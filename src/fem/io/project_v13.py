"""Schema-v13 persistence for strict sketch constraints."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

from fem.application.session import ProjectSaveSnapshot, ProjectSnapshot

from ._project_codec import (
    atomic_write_project,
    dumps_canonical_json,
    loads_json_strict,
    sketch_constraint_codec,
)
from ._project_errors import ProjectDecodeError, ProjectEncodeError, ProjectError
from .project_v12 import decode_project_v12, encode_project_v12


SCHEMA_VERSION = 13
FORMAT_NAME = "fem-python-project"


class ProjectV13Error(ProjectError):
    """Base error for schema-v13 processing."""


class ProjectV13DecodeError(ProjectV13Error, ProjectDecodeError):
    """A schema-v13 payload is malformed."""


class ProjectV13EncodeError(ProjectV13Error, ProjectEncodeError):
    """A snapshot cannot be represented losslessly by schema v13."""


def loads_project_v13(
    data: str | bytes | bytearray,
    *,
    source_path: str | Path | None = None,
) -> ProjectSnapshot:
    return decode_project_v13(
        loads_json_strict(
            data,
            error_type=ProjectV13DecodeError,
            document_label="v13 项目",
        ),
        source_path=source_path,
    )


def decode_project_v13(
    payload: Mapping[str, Any] | str | bytes | bytearray,
    *,
    source_path: str | Path | None = None,
) -> ProjectSnapshot:
    if isinstance(payload, (str, bytes, bytearray)):
        return loads_project_v13(payload, source_path=source_path)
    try:
        root = dict(payload)
        if set(root) != {"format", "schema", "project"}:
            raise ProjectV13DecodeError("$ 字段必须精确为 format、schema、project")
        if root["format"] != FORMAT_NAME:
            raise ProjectV13DecodeError(
                f"$.format 必须精确等于 {FORMAT_NAME!r}"
            )
        if type(root["schema"]) is not int or root["schema"] != SCHEMA_VERSION:
            raise ProjectV13DecodeError(
                f"v13 decoder 不能读取 schema {root['schema']!r}"
            )
        v12_payload = deepcopy(root)
        v12_payload["schema"] = 12
        with sketch_constraint_codec():
            return decode_project_v12(v12_payload, source_path=source_path)
    except ProjectV13Error:
        raise
    except Exception as error:
        raise ProjectV13DecodeError(f"schema v13 项目无效：{error}") from error


def encode_project_v13(
    snapshot: ProjectSnapshot | ProjectSaveSnapshot,
) -> dict[str, Any]:
    try:
        with sketch_constraint_codec():
            payload = encode_project_v12(snapshot)
        payload["schema"] = SCHEMA_VERSION
        dumps_canonical_json(payload, error_type=ProjectV13EncodeError)
        return payload
    except ProjectV13Error:
        raise
    except Exception as error:
        raise ProjectV13EncodeError(
            f"snapshot 无法由 v13 无损表示：{error}"
        ) from error


def dumps_project_v13(snapshot: ProjectSnapshot | ProjectSaveSnapshot) -> str:
    return dumps_canonical_json(
        encode_project_v13(snapshot), error_type=ProjectV13EncodeError
    )


def load_project_v13(path: str | Path) -> ProjectSnapshot:
    source = Path(path)
    return loads_project_v13(source.read_bytes(), source_path=source)


def save_project_v13(
    path: str | Path,
    snapshot: ProjectSnapshot | ProjectSaveSnapshot,
    *,
    checkpoint: Callable[[], Any] | None = None,
) -> Path:
    if checkpoint is not None:
        checkpoint()
    payload = encode_project_v13(snapshot)
    if checkpoint is not None:
        checkpoint()
    serialized = dumps_canonical_json(payload, error_type=ProjectV13EncodeError)
    return atomic_write_project(
        path,
        serialized,
        verifier=load_project_v13,
        semantic_encoder=encode_project_v13,
        expected_semantic=payload,
        error_type=ProjectV13EncodeError,
        mismatch_message="临时 v13 项目回读后的草图约束状态不一致",
        checkpoint=checkpoint,
    )


read_project_v13 = load_project_v13
write_project_v13 = save_project_v13


__all__ = [
    "FORMAT_NAME",
    "SCHEMA_VERSION",
    "ProjectV13DecodeError",
    "ProjectV13EncodeError",
    "ProjectV13Error",
    "decode_project_v13",
    "dumps_project_v13",
    "encode_project_v13",
    "load_project_v13",
    "loads_project_v13",
    "read_project_v13",
    "save_project_v13",
    "write_project_v13",
]
