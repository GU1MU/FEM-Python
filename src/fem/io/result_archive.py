"""Version-neutral ``.femres`` persistence API."""

from __future__ import annotations

from pathlib import Path

from fem.application.results.archive import LoadedResultArchive, ResultArchiveSnapshot

from ._result_archive_errors import (
    ResultArchiveDecodeError,
    ResultArchiveEncodeError,
    ResultArchiveError,
    UnsupportedResultArchiveSchemaError,
)
from .result_archive_v1 import (
    FORMAT_NAME,
    SCHEMA_VERSION,
    decode_result_archive_v1,
    dumps_result_archive_v1,
    encode_result_archive_v1,
    inspect_result_archive_header_bytes,
    inspect_result_archive_header_path,
    load_result_archive_v1,
    loads_result_archive_v1,
    read_result_archive_v1,
    save_result_archive_v1,
    write_result_archive_v1,
)


RESULT_ARCHIVE_FORMAT_NAME = FORMAT_NAME
CURRENT_RESULT_ARCHIVE_SCHEMA = SCHEMA_VERSION
RESULT_ARCHIVE_SCHEMA_VERSION = SCHEMA_VERSION
RESULT_FILE_SUFFIX = ".femres"
RESULT_ARCHIVE_FILE_SUFFIX = RESULT_FILE_SUFFIX


def _require_supported_schema(format_name: str, schema: int) -> None:
    if format_name != FORMAT_NAME:
        raise ResultArchiveDecodeError(
            f"unsupported result archive format {format_name!r}"
        )
    if schema != SCHEMA_VERSION:
        raise UnsupportedResultArchiveSchemaError(
            f"unsupported result archive schema {schema!r}"
        )


def save_result_archive(
    path: str | Path,
    snapshot: ResultArchiveSnapshot,
    *,
    checkpoint: object | None = None,
    before_replace: object | None = None,
) -> Path:
    """Route a snapshot to the current binary result archive writer."""

    return save_result_archive_v1(
        path,
        snapshot,
        checkpoint=checkpoint,
        before_replace=before_replace,
    )


def load_result_archive(path: str | Path) -> LoadedResultArchive:
    """Route a result archive path through strict schema dispatch."""

    _require_supported_schema(*inspect_result_archive_header_path(path))
    return load_result_archive_v1(path)


def encode_result_archive(snapshot: ResultArchiveSnapshot) -> bytes:
    return encode_result_archive_v1(snapshot)


def decode_result_archive(data: bytes | bytearray) -> ResultArchiveSnapshot:
    return loads_result_archive(data)


def dumps_result_archive(snapshot: ResultArchiveSnapshot) -> bytes:
    return dumps_result_archive_v1(snapshot)


def loads_result_archive(data: bytes | bytearray) -> ResultArchiveSnapshot:
    if type(data) not in {bytes, bytearray}:
        raise TypeError("archive data must be bytes or bytearray")
    raw = bytes(data)
    _require_supported_schema(*inspect_result_archive_header_bytes(raw))
    return loads_result_archive_v1(raw)


read_result_archive = load_result_archive
write_result_archive = save_result_archive


__all__ = [
    "CURRENT_RESULT_ARCHIVE_SCHEMA",
    "FORMAT_NAME",
    "RESULT_ARCHIVE_FORMAT_NAME",
    "RESULT_ARCHIVE_FILE_SUFFIX",
    "RESULT_ARCHIVE_SCHEMA_VERSION",
    "RESULT_FILE_SUFFIX",
    "SCHEMA_VERSION",
    "LoadedResultArchive",
    "ResultArchiveDecodeError",
    "ResultArchiveEncodeError",
    "ResultArchiveError",
    "ResultArchiveSnapshot",
    "UnsupportedResultArchiveSchemaError",
    "decode_result_archive",
    "decode_result_archive_v1",
    "dumps_result_archive",
    "dumps_result_archive_v1",
    "encode_result_archive",
    "encode_result_archive_v1",
    "load_result_archive",
    "load_result_archive_v1",
    "loads_result_archive",
    "loads_result_archive_v1",
    "read_result_archive",
    "read_result_archive_v1",
    "write_result_archive",
    "write_result_archive_v1",
    "save_result_archive",
    "save_result_archive_v1",
]
