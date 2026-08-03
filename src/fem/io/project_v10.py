"""Strict schema-v10 persistence for named analysis objects."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any

from fem.application.analysis_identity import (
    ANALYSIS_OBJECT_COLLECTIONS,
    validate_analysis_object_names,
    without_analysis_object_names,
)
from fem.application.session import ProjectSaveSnapshot, ProjectSnapshot

from ._project_codec import (
    ProjectFieldCodecPolicy,
    atomic_write_project,
    dumps_canonical_json,
    loads_json_strict,
    unwrap_project_snapshot,
)
from ._project_errors import ProjectDecodeError, ProjectEncodeError, ProjectError
from .project_v9 import decode_project_v9, encode_project_v9


SCHEMA_VERSION = 10
FORMAT_NAME = "fem-python-project"


class ProjectV10Error(ProjectError):
    """Base error for schema-v10 processing."""


class ProjectV10DecodeError(ProjectV10Error, ProjectDecodeError):
    """A schema-v10 payload is malformed or incompatible."""


class ProjectV10EncodeError(ProjectV10Error, ProjectEncodeError):
    """A snapshot cannot be represented losslessly by schema v10."""


def loads_project_v10(
    data: str | bytes | bytearray,
    *,
    source_path: str | Path | None = None,
) -> ProjectSnapshot:
    return decode_project_v10(
        loads_json_strict(
            data,
            error_type=ProjectV10DecodeError,
            document_label="v10 项目",
        ),
        source_path=source_path,
    )


def decode_project_v10(
    payload: Mapping[str, Any] | str | bytes | bytearray,
    *,
    source_path: str | Path | None = None,
    _field_policy: ProjectFieldCodecPolicy | None = None,
) -> ProjectSnapshot:
    if isinstance(payload, (str, bytes, bytearray)):
        return loads_project_v10(payload, source_path=source_path)
    try:
        root = _mapping(payload, "$")
        _exact_keys(root, "$", {"format", "schema", "project"})
        if root["format"] != FORMAT_NAME:
            raise ProjectV10DecodeError(
                f"$.format 必须精确等于 {FORMAT_NAME!r}"
            )
        if type(root["schema"]) is not int or root["schema"] != SCHEMA_VERSION:
            raise ProjectV10DecodeError(
                f"v10 decoder 不能读取 schema {root['schema']!r}"
            )

        v9_payload = deepcopy(dict(root))
        v9_payload["schema"] = 9
        steps = _steps(v9_payload)
        names: list[dict[str, tuple[str, ...]]] = []
        for step_index, raw_step in enumerate(steps):
            step_path = (
                "$.project.authoring.definitions.steps"
                f"[{step_index}]"
            )
            step = _mapping(raw_step, step_path)
            step_names: dict[str, tuple[str, ...]] = {}
            for collection in ANALYSIS_OBJECT_COLLECTIONS:
                raw_values = step.get(collection)
                if not isinstance(raw_values, list):
                    raise ProjectV10DecodeError(
                        f"{step_path}.{collection} 必须是 array"
                    )
                collected: list[str] = []
                for item_index, raw_item in enumerate(raw_values):
                    item_path = (
                        f"{step_path}.{collection}[{item_index}]"
                    )
                    item = _mapping(raw_item, item_path)
                    if "name" not in item:
                        raise ProjectV10DecodeError(
                            f"{item_path} 缺少字段 'name'"
                        )
                    name = item["name"]
                    if (
                        type(name) is not str
                        or not name.strip()
                        or name != name.strip()
                    ):
                        raise ProjectV10DecodeError(
                            f"{item_path}.name 必须是无首尾空白的非空 string"
                        )
                    collected.append(name)
                    del item["name"]
                step_names[collection] = tuple(collected)
            names.append(step_names)

        decode_options = (
            {} if _field_policy is None else {"_field_policy": _field_policy}
        )
        decoded = decode_project_v9(
            v9_payload,
            source_path=source_path,
            **decode_options,
        )
        if len(decoded.analysis_definitions) != len(names):
            raise ProjectV10DecodeError("分析步名称映射数量不匹配")
        named_steps = deepcopy(tuple(decoded.analysis_definitions))
        for step_index, step in enumerate(named_steps):
            for collection in ANALYSIS_OBJECT_COLLECTIONS:
                values = tuple(getattr(step, collection))
                collection_names = names[step_index][collection]
                if len(values) != len(collection_names):
                    raise ProjectV10DecodeError(
                        f"分析步 {step_index} 的 {collection} 名称数量不匹配"
                    )
                setattr(
                    step,
                    collection,
                    tuple(
                        replace(item, name=collection_names[index])
                        for index, item in enumerate(values)
                    ),
                )
        validate_analysis_object_names(named_steps, require_all=True)
        return replace(decoded, analysis_definitions=named_steps)
    except ProjectV10Error:
        raise
    except Exception as error:
        raise ProjectV10DecodeError(
            f"schema v10 项目无效：{error}"
        ) from error


def encode_project_v10(
    snapshot: ProjectSnapshot | ProjectSaveSnapshot,
    *,
    _field_policy: ProjectFieldCodecPolicy | None = None,
) -> dict[str, Any]:
    try:
        project = unwrap_project_snapshot(snapshot)
        validate_analysis_object_names(
            project.analysis_definitions,
            require_all=True,
        )
        legacy_project = replace(
            project,
            analysis_definitions=without_analysis_object_names(
                project.analysis_definitions
            ),
        )
        encode_options = (
            {} if _field_policy is None else {"_field_policy": _field_policy}
        )
        payload = encode_project_v9(legacy_project, **encode_options)
        payload["schema"] = SCHEMA_VERSION
        raw_steps = _steps(payload)
        for step_index, source_step in enumerate(
            project.analysis_definitions
        ):
            raw_step = raw_steps[step_index]
            for collection in ANALYSIS_OBJECT_COLLECTIONS:
                raw_values = raw_step[collection]
                source_values = tuple(getattr(source_step, collection))
                if len(raw_values) != len(source_values):
                    raise ProjectV10EncodeError(
                        f"分析步 {step_index} 的 {collection} 数量不匹配"
                    )
                for raw_item, source_item in zip(
                    raw_values,
                    source_values,
                    strict=True,
                ):
                    name = getattr(source_item, "name", None)
                    if type(name) is not str:
                        raise ProjectV10EncodeError(
                            f"{collection} 对象缺少稳定名称"
                        )
                    raw_item["name"] = name
        dumps_canonical_json(payload, error_type=ProjectV10EncodeError)
        return payload
    except ProjectV10Error:
        raise
    except Exception as error:
        raise ProjectV10EncodeError(
            f"snapshot 无法由 v10 无损表示：{error}"
        ) from error


def dumps_project_v10(
    snapshot: ProjectSnapshot | ProjectSaveSnapshot,
) -> str:
    return dumps_canonical_json(
        encode_project_v10(snapshot),
        error_type=ProjectV10EncodeError,
    )


def load_project_v10(path: str | Path) -> ProjectSnapshot:
    source = Path(path)
    return loads_project_v10(source.read_bytes(), source_path=source)


def save_project_v10(
    path: str | Path,
    snapshot: ProjectSnapshot | ProjectSaveSnapshot,
    *,
    checkpoint: Callable[[], Any] | None = None,
) -> Path:
    if checkpoint is not None:
        checkpoint()
    payload = encode_project_v10(snapshot)
    if checkpoint is not None:
        checkpoint()
    serialized = dumps_canonical_json(
        payload,
        error_type=ProjectV10EncodeError,
    )
    return atomic_write_project(
        path,
        serialized,
        verifier=load_project_v10,
        semantic_encoder=encode_project_v10,
        expected_semantic=payload,
        error_type=ProjectV10EncodeError,
        mismatch_message=(
            "临时 v10 项目回读后的分析对象身份与 canonical authoring 不一致"
        ),
        checkpoint=checkpoint,
    )


read_project_v10 = load_project_v10
write_project_v10 = save_project_v10


def _steps(payload: Mapping[str, Any]) -> list[Any]:
    project = _mapping(payload.get("project"), "$.project")
    authoring = _mapping(project.get("authoring"), "$.project.authoring")
    definitions = _mapping(
        authoring.get("definitions"),
        "$.project.authoring.definitions",
    )
    steps = definitions.get("steps")
    if not isinstance(steps, list):
        raise ProjectV10DecodeError(
            "$.project.authoring.definitions.steps 必须是 array"
        )
    return steps


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProjectV10DecodeError(f"{path} 必须是 object")
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
        raise ProjectV10DecodeError(
            f"{path} 缺少字段 {sorted(missing)[0]!r}"
        )
    if extra:
        raise ProjectV10DecodeError(
            f"{path} 包含未知字段 {sorted(extra)[0]!r}"
        )


__all__ = [
    "FORMAT_NAME",
    "ProjectV10DecodeError",
    "ProjectV10EncodeError",
    "ProjectV10Error",
    "SCHEMA_VERSION",
    "decode_project_v10",
    "dumps_project_v10",
    "encode_project_v10",
    "load_project_v10",
    "loads_project_v10",
    "read_project_v10",
    "save_project_v10",
    "write_project_v10",
]
