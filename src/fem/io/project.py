"""Version-neutral native project persistence API.

Readers dispatch schema 1 compatibility migration or schema 2-14 codecs after
one strict JSON parse.  Writers always emit the current schema.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from fem.application.session import ProjectSaveSnapshot, ProjectSnapshot
from fem.application.analysis_identity import with_compatibility_analysis_names

from ._project_codec import borrow_project_snapshot, loads_json_strict
from ._project_errors import (
    ProjectDecodeError,
    ProjectEncodeError,
    ProjectError,
    UnsupportedProjectSchemaError,
)
from .project_migration import (
    ProjectMigrationNotice,
    ProjectV1MigrationError,
    migrate_project_snapshot_to_v5,
    migrate_project_snapshot_to_v7,
)
from .project_v1 import _decode_project_v1_loaded
from .project_v2 import decode_project_v2
from .project_v3 import (
    decode_project_v3,
)
from .project_v4 import decode_project_v4
from .project_v5 import (
    decode_project_v5,
)
from .project_v6 import (
    decode_project_v6,
)
from .project_v7 import (
    decode_project_v7,
)
from .project_v8 import (
    decode_project_v8,
)
from .project_v9 import (
    decode_project_v9,
)
from .project_v10 import (
    decode_project_v10,
)
from .project_v11 import decode_project_v11
from .project_v12 import decode_project_v12
from .project_v13 import (
    decode_project_v13,
)
from .project_v14 import (
    decode_project_v14,
    dumps_project_v14,
    encode_project_v14,
    save_project_v14,
)


CURRENT_PROJECT_SCHEMA = 14

# Public model filename contracts.  ``.femproj`` remains a read-compatible
# legacy extension; writers and GUI save flows use ``.fempy`` exclusively.
MODEL_FILE_SUFFIX = ".fempy"
LEGACY_MODEL_FILE_SUFFIXES = (".femproj",)


@dataclass(frozen=True, slots=True)
class LoadedProject:
    """One detached decoded project plus non-authoritative load metadata."""

    snapshot: ProjectSnapshot
    path: Path | None
    source_schema: int
    notices: tuple[ProjectMigrationNotice, ...]

    def __post_init__(self) -> None:
        if type(self.snapshot) is not ProjectSnapshot:
            raise TypeError("snapshot must be a ProjectSnapshot")
        if self.path is not None and not isinstance(self.path, Path):
            raise TypeError("path must be a pathlib.Path or None")
        if type(self.source_schema) is not int or self.source_schema <= 0:
            raise TypeError("source_schema must be a positive integer")
        if type(self.notices) is not tuple or any(
            type(notice) is not ProjectMigrationNotice for notice in self.notices
        ):
            raise TypeError(
                "notices must be a tuple of ProjectMigrationNotice values"
            )
        if self.snapshot.source_path != self.path:
            raise ValueError(
                "loaded project path must equal snapshot.source_path"
            )


def load_project(path: str | Path) -> LoadedProject:
    """Read and decode a supported schema 1-14 project from *path*."""

    source = Path(path)
    return loads_project(source.read_bytes(), source_path=source)


def loads_project(
    data: str | bytes | bytearray,
    *,
    source_path: str | Path | None = None,
) -> LoadedProject:
    """Strictly parse and decode a supported schema 1-14 JSON document."""

    payload = loads_json_strict(
        data,
        error_type=ProjectDecodeError,
        document_label="项目",
    )
    return decode_project(payload, source_path=source_path)


def decode_project(
    payload: Mapping[str, Any],
    *,
    source_path: str | Path | None = None,
) -> LoadedProject:
    """Dispatch one already-parsed project object by its strict schema value."""

    if not isinstance(payload, Mapping):
        raise TypeError(
            "decode_project() 只接受已解析的 mapping；"
            "文本或 bytes 请使用 loads_project()"
        )
    if "schema" not in payload:
        raise ProjectDecodeError(
            "$.schema 缺失；项目必须声明受支持的严格整数 schema"
        )
    schema = payload["schema"]
    if type(schema) is not int:
        raise ProjectDecodeError(
            "$.schema 必须是严格整数；请使用受支持的项目格式"
        )

    resolved_path = None if source_path is None else Path(source_path)
    if schema == 1:
        snapshot, notices = _decode_project_v1_loaded(
            payload,
            source_path=resolved_path,
        )
    elif schema == 2:
        snapshot = decode_project_v2(
            payload,
            source_path=resolved_path,
        )
        notices = ()
    elif schema == 3:
        snapshot = decode_project_v3(
            payload,
            source_path=resolved_path,
        )
        notices = ()
    elif schema == 4:
        snapshot = decode_project_v4(
            payload,
            source_path=resolved_path,
        )
        notices = ()
    elif schema == 5:
        snapshot = decode_project_v5(
            payload,
            source_path=resolved_path,
        )
        notices = ()
    elif schema == 6:
        snapshot = decode_project_v6(
            payload,
            source_path=resolved_path,
        )
        notices = ()
    elif schema == 7:
        snapshot = decode_project_v7(
            payload,
            source_path=resolved_path,
        )
        notices = ()
    elif schema == 8:
        snapshot = decode_project_v8(
            payload,
            source_path=resolved_path,
        )
        notices = ()
    elif schema == 9:
        snapshot = decode_project_v9(
            payload,
            source_path=resolved_path,
        )
        notices = ()
    elif schema == 10:
        snapshot = decode_project_v10(
            payload,
            source_path=resolved_path,
        )
        notices = ()
    elif schema == 11:
        snapshot = decode_project_v11(
            payload,
            source_path=resolved_path,
        )
        notices = ()
    elif schema == 12:
        snapshot = decode_project_v12(
            payload,
            source_path=resolved_path,
        )
        notices = ()
    elif schema == 13:
        snapshot = decode_project_v13(payload, source_path=resolved_path)
        notices = ()
    elif schema == CURRENT_PROJECT_SCHEMA:
        snapshot = decode_project_v14(payload, source_path=resolved_path)
        notices = ()
    else:
        raise UnsupportedProjectSchemaError(
            f"$.schema={schema!r} 不受支持；"
            "当前版本可读取 schema 1、2、3、4、5、6、7、8、9、10 和 "
            f"11、12、13 和 {CURRENT_PROJECT_SCHEMA}"
        )
    if schema < 7:
        try:
            snapshot, v5_notices = migrate_project_snapshot_to_v5(snapshot)
        except ProjectV1MigrationError as error:
            raise ProjectDecodeError(
                f"schema {schema} 无法原子迁移到 schema {CURRENT_PROJECT_SCHEMA}：{error}"
            ) from error
        notices = (*notices, *v5_notices)
        try:
            snapshot, v7_notices = migrate_project_snapshot_to_v7(snapshot)
        except ProjectV1MigrationError as error:
            raise ProjectDecodeError(
                f"schema {schema} 无法原子迁移到 schema {CURRENT_PROJECT_SCHEMA}：{error}"
            ) from error
        notices = (*notices, *v7_notices)
    if schema < CURRENT_PROJECT_SCHEMA:
        snapshot = replace(
            snapshot,
            analysis_definitions=with_compatibility_analysis_names(
                snapshot.analysis_definitions
            ),
        )
    return LoadedProject(
        snapshot=snapshot,
        path=resolved_path,
        source_schema=schema,
        notices=tuple(notices),
    )


def encode_project(
    snapshot: ProjectSnapshot | ProjectSaveSnapshot,
) -> dict[str, Any]:
    """Encode a detached snapshot using the current project schema."""

    return encode_project_v14(_canonical_writer_snapshot(snapshot))


def dumps_project(
    snapshot: ProjectSnapshot | ProjectSaveSnapshot,
) -> str:
    """Serialize a detached snapshot using canonical current-schema JSON."""

    return dumps_project_v14(_canonical_writer_snapshot(snapshot))


def save_project(
    path: str | Path,
    snapshot: ProjectSnapshot | ProjectSaveSnapshot,
    *,
    checkpoint: Callable[[], Any] | None = None,
) -> Path:
    """Atomically save a detached snapshot using the current schema."""

    return save_project_v14(
        path,
        _canonical_writer_snapshot(snapshot),
        checkpoint=checkpoint,
    )


def _canonical_writer_snapshot(
    snapshot: ProjectSnapshot | ProjectSaveSnapshot,
) -> ProjectSnapshot | ProjectSaveSnapshot:
    """Upgrade compatibility snapshots before the strict v10 writer."""

    project = borrow_project_snapshot(snapshot)
    if project.geometry_recipe is not None and not (
        project.parts
        and all(part.geometry_recipe is not None for part in project.parts)
    ):
        project, _notices = migrate_project_snapshot_to_v7(project)
    analysis_definitions = with_compatibility_analysis_names(
        project.analysis_definitions
    )
    if analysis_definitions == project.analysis_definitions:
        return project
    return replace(project, analysis_definitions=analysis_definitions)


__all__ = [
    "CURRENT_PROJECT_SCHEMA",
    "LEGACY_MODEL_FILE_SUFFIXES",
    "LoadedProject",
    "MODEL_FILE_SUFFIX",
    "ProjectDecodeError",
    "ProjectEncodeError",
    "ProjectError",
    "ProjectMigrationNotice",
    "UnsupportedProjectSchemaError",
    "decode_project",
    "dumps_project",
    "encode_project",
    "load_project",
    "loads_project",
    "save_project",
]
