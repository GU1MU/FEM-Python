"""Strict schema-v9 persistence for explicit and automatic mesh intent."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any

from fem.application.native_part import NativePart
from fem.application.session import ProjectSaveSnapshot, ProjectSnapshot
from fem.mesh.settings import MeshSettings

from ._project_codec import (
    atomic_write_project,
    dumps_canonical_json,
    loads_json_strict,
    unwrap_project_snapshot,
)
from ._project_errors import ProjectDecodeError, ProjectEncodeError, ProjectError
from .project_v8 import decode_project_v8, encode_project_v8


SCHEMA_VERSION = 9
FORMAT_NAME = "fem-python-project"
_V8_MESH_FIELDS = {
    "size",
    "order",
    "cell_shape",
    "local_controls",
    "line_element_type",
}
_V9_MESH_FIELDS = {
    *_V8_MESH_FIELDS,
    "intent_mode",
    "auto_level",
    "strict_cell_shape",
}


class ProjectV9Error(ProjectError):
    """Base error for schema-v9 processing."""


class ProjectV9DecodeError(ProjectV9Error, ProjectDecodeError):
    """A schema-v9 payload is malformed or incompatible."""


class ProjectV9EncodeError(ProjectV9Error, ProjectEncodeError):
    """A snapshot cannot be represented losslessly by schema v9."""


def loads_project_v9(
    data: str | bytes | bytearray,
    *,
    source_path: str | Path | None = None,
) -> ProjectSnapshot:
    return decode_project_v9(
        loads_json_strict(
            data,
            error_type=ProjectV9DecodeError,
            document_label="v9 项目",
        ),
        source_path=source_path,
    )


def decode_project_v9(
    payload: Mapping[str, Any] | str | bytes | bytearray,
    *,
    source_path: str | Path | None = None,
) -> ProjectSnapshot:
    if isinstance(payload, (str, bytes, bytearray)):
        return loads_project_v9(payload, source_path=source_path)
    try:
        root = _mapping(payload, "$")
        _exact_keys(root, "$", {"format", "schema", "project"})
        if root["format"] != FORMAT_NAME:
            raise ProjectV9DecodeError(f"$.format 必须精确等于 {FORMAT_NAME!r}")
        if type(root["schema"]) is not int or root["schema"] != SCHEMA_VERSION:
            raise ProjectV9DecodeError(
                f"v9 decoder 不能读取 schema {root['schema']!r}"
            )
        project = _mapping(root["project"], "$.project")
        authoring = _mapping(project.get("authoring"), "$.project.authoring")
        raw_parts = authoring.get("parts")
        if not isinstance(raw_parts, list):
            raise ProjectV9DecodeError("$.project.authoring.parts 必须是 array")

        auto_levels: dict[str, int | None] = {}
        strict_shapes: dict[str, bool] = {}
        v8_payload = deepcopy(dict(root))
        v8_payload["schema"] = 8
        v8_parts = v8_payload["project"]["authoring"]["parts"]
        for index, raw_part in enumerate(v8_parts):
            part_path = f"$.project.authoring.parts[{index}]"
            part = _mapping(raw_part, part_path)
            part_id = part.get("id")
            if type(part_id) is not str:
                raise ProjectV9DecodeError(f"{part_path}.id 必须是 string")
            raw_settings = part.get("mesh_settings")
            if raw_settings is None:
                auto_levels[part_id] = None
                strict_shapes[part_id] = False
                continue
            settings = _mapping(raw_settings, f"{part_path}.mesh_settings")
            _exact_keys(settings, f"{part_path}.mesh_settings", _V9_MESH_FIELDS)
            mode = settings["intent_mode"]
            level = settings["auto_level"]
            strict = settings["strict_cell_shape"]
            if type(strict) is not bool:
                raise ProjectV9DecodeError(
                    f"{part_path}.mesh_settings.strict_cell_shape 必须是 boolean"
                )
            strict_shapes[part_id] = strict
            if mode == "explicit":
                if level is not None:
                    raise ProjectV9DecodeError(
                        f"{part_path}.mesh_settings.auto_level "
                        "在 explicit 模式必须为 null"
                    )
                auto_levels[part_id] = None
            elif mode == "automatic":
                if (
                    isinstance(level, bool)
                    or type(level) is not int
                    or level not in {1, 2, 3, 4, 5}
                ):
                    raise ProjectV9DecodeError(
                        f"{part_path}.mesh_settings.auto_level 必须是 1 到 5"
                    )
                auto_levels[part_id] = level
            else:
                raise ProjectV9DecodeError(
                    f"{part_path}.mesh_settings.intent_mode 无效"
                )
            del settings["intent_mode"]
            del settings["auto_level"]
            del settings["strict_cell_shape"]

        decoded = decode_project_v8(v8_payload, source_path=source_path)
        parts: list[NativePart] = []
        for part in decoded.parts:
            settings = part.mesh_settings
            level = auto_levels.get(part.id)
            if settings is not None:
                settings = replace(
                    settings,
                    auto_level=level,
                    strict_cell_shape=strict_shapes.get(part.id, False),
                )
            if (
                part.provenance is not None
                and settings is not None
                and settings.auto_level is not None
            ):
                raise ProjectV9DecodeError(
                    "schema v9 首轮不允许 Boolean 结果 Part 使用 AutoMesh"
                )
            parts.append(replace(part, mesh_settings=settings))
        active = next(
            (
                part
                for part in parts
                if part.id == decoded.active_part_id
            ),
            None,
        )
        return replace(
            decoded,
            parts=tuple(parts),
            mesh_settings=(None if active is None else active.mesh_settings),
        )
    except ProjectV9Error:
        raise
    except Exception as error:
        raise ProjectV9DecodeError(f"schema v9 项目无效：{error}") from error


def encode_project_v9(
    snapshot: ProjectSnapshot | ProjectSaveSnapshot,
) -> dict[str, Any]:
    try:
        project = unwrap_project_snapshot(snapshot)
        if any(
            part.provenance is not None
            and part.mesh_settings is not None
            and part.mesh_settings.auto_level is not None
            for part in project.parts
        ):
            raise ProjectV9EncodeError(
                "schema v9 首轮不允许 Boolean 结果 Part 使用 AutoMesh"
            )
        payload = encode_project_v8(project)
        payload["schema"] = SCHEMA_VERSION
        by_id = {part.id: part for part in project.parts}
        for raw_part in payload["project"]["authoring"]["parts"]:
            source = by_id[str(raw_part["id"])]
            encoded_settings = raw_part["mesh_settings"]
            if encoded_settings is None:
                continue
            settings = source.mesh_settings
            if type(settings) is not MeshSettings:
                raise ProjectV9EncodeError(
                    f"Part {source.id} mesh_settings 不是 MeshSettings"
                )
            encoded_settings["intent_mode"] = (
                "automatic" if settings.auto_level is not None else "explicit"
            )
            encoded_settings["auto_level"] = settings.auto_level
            encoded_settings["strict_cell_shape"] = settings.strict_cell_shape
        dumps_canonical_json(payload, error_type=ProjectV9EncodeError)
        return payload
    except ProjectV9Error:
        raise
    except Exception as error:
        raise ProjectV9EncodeError(
            f"snapshot 无法由 v9 无损表示：{error}"
        ) from error


def dumps_project_v9(
    snapshot: ProjectSnapshot | ProjectSaveSnapshot,
) -> str:
    return dumps_canonical_json(
        encode_project_v9(snapshot),
        error_type=ProjectV9EncodeError,
    )


def load_project_v9(path: str | Path) -> ProjectSnapshot:
    source = Path(path)
    return loads_project_v9(source.read_bytes(), source_path=source)


def save_project_v9(
    path: str | Path,
    snapshot: ProjectSnapshot | ProjectSaveSnapshot,
    *,
    checkpoint: Callable[[], Any] | None = None,
) -> Path:
    if checkpoint is not None:
        checkpoint()
    payload = encode_project_v9(snapshot)
    if checkpoint is not None:
        checkpoint()
    serialized = dumps_canonical_json(
        payload,
        error_type=ProjectV9EncodeError,
    )
    return atomic_write_project(
        path,
        serialized,
        verifier=load_project_v9,
        semantic_encoder=encode_project_v9,
        expected_semantic=payload,
        error_type=ProjectV9EncodeError,
        mismatch_message=(
            "临时 v9 项目回读后的 mesh intent 与 canonical authoring 不一致"
        ),
        checkpoint=checkpoint,
    )


read_project_v9 = load_project_v9
write_project_v9 = save_project_v9


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProjectV9DecodeError(f"{path} 必须是 object")
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
        raise ProjectV9DecodeError(f"{path} 缺少字段 {sorted(missing)[0]!r}")
    if extra:
        raise ProjectV9DecodeError(f"{path} 包含未知字段 {sorted(extra)[0]!r}")


__all__ = [
    "FORMAT_NAME",
    "ProjectV9DecodeError",
    "ProjectV9EncodeError",
    "ProjectV9Error",
    "SCHEMA_VERSION",
    "decode_project_v9",
    "dumps_project_v9",
    "encode_project_v9",
    "load_project_v9",
    "loads_project_v9",
    "read_project_v9",
    "save_project_v9",
    "write_project_v9",
]
