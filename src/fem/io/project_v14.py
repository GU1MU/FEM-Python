"""Schema-v14 persistence for an accepted native mesh artifact."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

from fem.application.session import ProjectSaveSnapshot, ProjectSnapshot

from ._atomic_text import atomic_write_verified_text_stream
from ._project_codec import (
    borrow_project_snapshot,
    dumps_compact_canonical_json,
    loads_json_strict,
)
from ._project_errors import ProjectDecodeError, ProjectEncodeError, ProjectError
from ._project_model_codec import decode_project_model, encode_project_model
from .project_v13 import decode_project_v13, encode_project_v13


SCHEMA_VERSION = 14
FORMAT_NAME = "fem-python-project"


class ProjectV14Error(ProjectError):
    """Base error for schema-v14 processing."""


class ProjectV14DecodeError(ProjectV14Error, ProjectDecodeError):
    """A schema-v14 payload is malformed."""


class ProjectV14EncodeError(ProjectV14Error, ProjectEncodeError):
    """A snapshot cannot be represented losslessly by schema v14."""


def loads_project_v14(
    data: str | bytes | bytearray,
    *,
    source_path: str | Path | None = None,
) -> ProjectSnapshot:
    return decode_project_v14(
        loads_json_strict(
            data,
            error_type=ProjectV14DecodeError,
            document_label="v14 project",
        ),
        source_path=source_path,
    )


def decode_project_v14(
    payload: Mapping[str, Any] | str | bytes | bytearray,
    *,
    source_path: str | Path | None = None,
) -> ProjectSnapshot:
    if isinstance(payload, (str, bytes, bytearray)):
        return loads_project_v14(payload, source_path=source_path)
    try:
        root = dict(payload)
        if set(root) != {"format", "schema", "project"}:
            raise ProjectV14DecodeError("$ fields must be exactly format, schema, project")
        if root["format"] != FORMAT_NAME:
            raise ProjectV14DecodeError(f"$.format must equal exactly {FORMAT_NAME!r}")
        if type(root["schema"]) is not int or root["schema"] != SCHEMA_VERSION:
            raise ProjectV14DecodeError(
                f"v14 decoder cannot read schema {root['schema']!r}"
            )
        project = root["project"]
        if not isinstance(project, Mapping):
            raise ProjectV14DecodeError("$.project must be an object")
        if set(project) != {"kind", "authoring", "model_artifact"}:
            raise ProjectV14DecodeError(
                "$.project fields must be exactly kind, authoring, model_artifact"
            )
        raw_model = project["model_artifact"]
        v13_payload = {
            "format": root["format"],
            "schema": 13,
            "project": {
                "kind": project["kind"],
                "authoring": project["authoring"],
            },
        }
        decoded = decode_project_v13(v13_payload, source_path=source_path)
        model = (
            None
            if raw_model is None
            else decode_project_model(
                raw_model,
                error_type=ProjectV14DecodeError,
            )
        )
        return replace(decoded, model=model)
    except ProjectV14Error:
        raise
    except Exception as error:
        raise ProjectV14DecodeError(f"invalid schema v14 project: {error}") from error


def encode_project_v14(
    snapshot: ProjectSnapshot | ProjectSaveSnapshot,
) -> dict[str, Any]:
    try:
        project = borrow_project_snapshot(
            snapshot,
            error_type=ProjectV14EncodeError,
        )
        payload = encode_project_v13(replace(project, model=None))
        payload["schema"] = SCHEMA_VERSION
        payload["project"]["model_artifact"] = (
            None
            if project.model is None
            else encode_project_model(
                project.model,
                error_type=ProjectV14EncodeError,
            )
        )
        return payload
    except ProjectV14Error:
        raise
    except Exception as error:
        raise ProjectV14EncodeError(f"snapshot cannot be represented losslessly by v14: {error}") from error


def dumps_project_v14(snapshot: ProjectSnapshot | ProjectSaveSnapshot) -> str:
    return dumps_compact_canonical_json(
        encode_project_v14(snapshot),
        error_type=ProjectV14EncodeError,
    )


def load_project_v14(path: str | Path) -> ProjectSnapshot:
    source = Path(path)
    return loads_project_v14(source.read_bytes(), source_path=source)


def save_project_v14(
    path: str | Path,
    snapshot: ProjectSnapshot | ProjectSaveSnapshot,
    *,
    checkpoint: Callable[[], Any] | None = None,
) -> Path:
    if checkpoint is not None:
        checkpoint()
    serialized = dumps_project_v14(snapshot)
    if checkpoint is not None:
        checkpoint()
    return atomic_write_verified_text_stream(
        path,
        lambda stream: stream.write(serialized),
        error_type=ProjectV14EncodeError,
        mismatch_message="UTF-8 bytes differ after rereading the temporary v14 project",
        checkpoint=checkpoint,
    )


read_project_v14 = load_project_v14
write_project_v14 = save_project_v14


__all__ = [
    "FORMAT_NAME",
    "SCHEMA_VERSION",
    "ProjectV14DecodeError",
    "ProjectV14EncodeError",
    "ProjectV14Error",
    "decode_project_v14",
    "dumps_project_v14",
    "encode_project_v14",
    "load_project_v14",
    "loads_project_v14",
    "read_project_v14",
    "save_project_v14",
    "write_project_v14",
]
