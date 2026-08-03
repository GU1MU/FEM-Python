"""Strict schema-v11 persistence for planar-face sketch Boolean features."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any

from fem.application.session import (
    FaceSketchBooleanUndoRecord,
    ProjectSaveSnapshot,
    ProjectSnapshot,
)
from fem.geometry.recipes import FaceSketchBooleanGeometry

from . import project_v7 as _v7
from ._project_codec import (
    ProjectFieldCodecPolicy,
    atomic_write_project,
    decode_assignment_field,
    decode_step_field,
    dumps_canonical_json,
    encode_assignment_field,
    encode_step_field,
    loads_json_strict,
    unwrap_project_snapshot,
)
from ._project_errors import ProjectDecodeError, ProjectEncodeError, ProjectError
from .project_v10 import decode_project_v10, encode_project_v10


SCHEMA_VERSION = 11
FORMAT_NAME = "fem-python-project"


class ProjectV11Error(ProjectError):
    """Base error for schema-v11 processing."""


class ProjectV11DecodeError(ProjectV11Error, ProjectDecodeError):
    """A schema-v11 payload is malformed or fails exact replay."""


class ProjectV11EncodeError(ProjectV11Error, ProjectEncodeError):
    """A snapshot cannot be represented losslessly by schema v11."""


_V11_FIELD_POLICY = ProjectFieldCodecPolicy(
    version_label="v11",
    decode_error=ProjectV11DecodeError,
    encode_error=ProjectV11EncodeError,
    require_current_fields=True,
    assignment_orientation=True,
    allow_wire_geometry=True,
    allow_strict_sketch=True,
    extrusion_source_faces=True,
    displacement_region_targets=True,
    body_force_loads=True,
    allow_multi_body=True,
    allow_planar_boolean=True,
    allow_part_boolean=True,
    allow_revolved_geometry=True,
    allow_path_swept_geometry=True,
    allow_face_sketch_boolean=True,
)


def loads_project_v11(
    data: str | bytes | bytearray,
    *,
    source_path: str | Path | None = None,
) -> ProjectSnapshot:
    return decode_project_v11(
        loads_json_strict(
            data,
            error_type=ProjectV11DecodeError,
            document_label="v11 项目",
        ),
        source_path=source_path,
    )


def decode_project_v11(
    payload: Mapping[str, Any] | str | bytes | bytearray,
    *,
    source_path: str | Path | None = None,
) -> ProjectSnapshot:
    if isinstance(payload, (str, bytes, bytearray)):
        return loads_project_v11(payload, source_path=source_path)
    try:
        root = _mapping(payload, "$")
        _exact_keys(root, "$", {"format", "schema", "project"})
        if root["format"] != FORMAT_NAME:
            raise ProjectV11DecodeError(
                f"$.format 必须精确等于 {FORMAT_NAME!r}"
            )
        if type(root["schema"]) is not int or root["schema"] != SCHEMA_VERSION:
            raise ProjectV11DecodeError(
                f"v11 decoder 不能读取 schema {root['schema']!r}"
            )
        project = _mapping(root["project"], "$.project")
        authoring = _mapping(project.get("authoring"), "$.project.authoring")
        required_records = {
            "face_sketch_boolean_undo_records",
            "face_sketch_boolean_redo_records",
        }
        if not required_records.issubset(authoring):
            missing = sorted(required_records - set(authoring))[0]
            raise ProjectV11DecodeError(
                f"$.project.authoring 缺少字段 {missing!r}"
            )
        undo_records = _decode_record_array(
            authoring["face_sketch_boolean_undo_records"],
            "$.project.authoring.face_sketch_boolean_undo_records",
        )
        redo_records = _decode_record_array(
            authoring["face_sketch_boolean_redo_records"],
            "$.project.authoring.face_sketch_boolean_redo_records",
        )
        v10_payload = deepcopy(dict(root))
        v10_payload["schema"] = 10
        v10_authoring = v10_payload["project"]["authoring"]
        del v10_authoring["face_sketch_boolean_undo_records"]
        del v10_authoring["face_sketch_boolean_redo_records"]
        decoded = decode_project_v10(
            v10_payload,
            source_path=source_path,
            _field_policy=_V11_FIELD_POLICY,
        )
        result = replace(
            decoded,
            face_sketch_boolean_undo_records=undo_records,
            face_sketch_boolean_redo_records=redo_records,
        )
        _validate_face_sketch_state(result, encode=False)
        return result
    except ProjectV11Error:
        raise
    except Exception as error:
        raise ProjectV11DecodeError(f"schema v11 项目无效：{error}") from error


def encode_project_v11(
    snapshot: ProjectSnapshot | ProjectSaveSnapshot,
) -> dict[str, Any]:
    try:
        project = unwrap_project_snapshot(snapshot)
        _validate_face_sketch_state(project, encode=True)
        payload = encode_project_v10(project, _field_policy=_V11_FIELD_POLICY)
        payload["schema"] = SCHEMA_VERSION
        authoring = payload["project"]["authoring"]
        authoring["face_sketch_boolean_undo_records"] = [
            _encode_record(
                record,
                f"snapshot.face_sketch_boolean_undo_records[{index}]",
            )
            for index, record in enumerate(
                project.face_sketch_boolean_undo_records
            )
        ]
        authoring["face_sketch_boolean_redo_records"] = [
            _encode_record(
                record,
                f"snapshot.face_sketch_boolean_redo_records[{index}]",
            )
            for index, record in enumerate(
                project.face_sketch_boolean_redo_records
            )
        ]
        dumps_canonical_json(payload, error_type=ProjectV11EncodeError)
        return payload
    except ProjectV11Error:
        raise
    except Exception as error:
        raise ProjectV11EncodeError(
            f"snapshot 无法由 v11 无损表示：{error}"
        ) from error


def dumps_project_v11(
    snapshot: ProjectSnapshot | ProjectSaveSnapshot,
) -> str:
    return dumps_canonical_json(
        encode_project_v11(snapshot),
        error_type=ProjectV11EncodeError,
    )


def load_project_v11(path: str | Path) -> ProjectSnapshot:
    source = Path(path)
    return loads_project_v11(source.read_bytes(), source_path=source)


def save_project_v11(
    path: str | Path,
    snapshot: ProjectSnapshot | ProjectSaveSnapshot,
    *,
    checkpoint: Callable[[], Any] | None = None,
) -> Path:
    if checkpoint is not None:
        checkpoint()
    payload = encode_project_v11(snapshot)
    if checkpoint is not None:
        checkpoint()
    serialized = dumps_canonical_json(
        payload,
        error_type=ProjectV11EncodeError,
    )
    return atomic_write_project(
        path,
        serialized,
        verifier=load_project_v11,
        semantic_encoder=encode_project_v11,
        expected_semantic=payload,
        error_type=ProjectV11EncodeError,
        mismatch_message="临时 v11 项目回读后的面草图特征与撤销状态不一致",
        checkpoint=checkpoint,
    )


def _decode_record_array(
    value: Any,
    path: str,
) -> tuple[FaceSketchBooleanUndoRecord, ...]:
    if not isinstance(value, list):
        raise ProjectV11DecodeError(f"{path} 必须是 array")
    return tuple(
        _decode_record(item, f"{path}[{index}]")
        for index, item in enumerate(value)
    )


def _decode_record(value: Any, path: str) -> FaceSketchBooleanUndoRecord:
    data = _mapping(value, path)
    _exact_keys(
        data,
        path,
        {
            "feature_id",
            "part_id",
            "before_part",
            "after_part",
            "before_named_regions",
            "after_named_regions",
            "before_assignments",
            "after_assignments",
            "before_steps",
            "after_steps",
        },
    )
    return FaceSketchBooleanUndoRecord(
        _string(data["feature_id"], f"{path}.feature_id"),
        _string(data["part_id"], f"{path}.part_id"),
        _v7._decode_part(
            data["before_part"],
            f"{path}.before_part",
            field_policy=_V11_FIELD_POLICY,
        ),
        _v7._decode_part(
            data["after_part"],
            f"{path}.after_part",
            field_policy=_V11_FIELD_POLICY,
        ),
        _decode_regions(data["before_named_regions"], f"{path}.before_named_regions"),
        _decode_regions(data["after_named_regions"], f"{path}.after_named_regions"),
        _decode_values(
            data["before_assignments"],
            f"{path}.before_assignments",
            decode_assignment_field,
        ),
        _decode_values(
            data["after_assignments"],
            f"{path}.after_assignments",
            decode_assignment_field,
        ),
        _decode_values(data["before_steps"], f"{path}.before_steps", decode_step_field),
        _decode_values(data["after_steps"], f"{path}.after_steps", decode_step_field),
    )


def _encode_record(
    record: FaceSketchBooleanUndoRecord,
    path: str,
) -> dict[str, Any]:
    if type(record) is not FaceSketchBooleanUndoRecord:
        raise ProjectV11EncodeError(
            f"{path} 必须是 FaceSketchBooleanUndoRecord"
        )
    return {
        "feature_id": record.feature_id,
        "part_id": record.part_id,
        "before_part": _v7._encode_part(
            record.before_part,
            f"{path}.before_part",
            field_policy=_V11_FIELD_POLICY,
        ),
        "after_part": _v7._encode_part(
            record.after_part,
            f"{path}.after_part",
            field_policy=_V11_FIELD_POLICY,
        ),
        "before_named_regions": _encode_regions(
            record.before_named_regions, f"{path}.before_named_regions"
        ),
        "after_named_regions": _encode_regions(
            record.after_named_regions, f"{path}.after_named_regions"
        ),
        "before_assignments": _encode_values(
            record.before_assignments,
            f"{path}.before_assignments",
            encode_assignment_field,
        ),
        "after_assignments": _encode_values(
            record.after_assignments,
            f"{path}.after_assignments",
            encode_assignment_field,
        ),
        "before_steps": _encode_values(
            record.before_steps, f"{path}.before_steps", encode_step_field
        ),
        "after_steps": _encode_values(
            record.after_steps, f"{path}.after_steps", encode_step_field
        ),
    }


def _decode_regions(value: Any, path: str) -> tuple[Any, ...]:
    if not isinstance(value, list):
        raise ProjectV11DecodeError(f"{path} 必须是 array")
    return tuple(
        _v7._decode_named_region(item, f"{path}[{index}]")
        for index, item in enumerate(value)
    )


def _encode_regions(values: tuple[Any, ...], path: str) -> list[Any]:
    return [
        _v7._encode_named_region(item, f"{path}[{index}]")
        for index, item in enumerate(values)
    ]


def _decode_values(value: Any, path: str, decoder: Any) -> tuple[Any, ...]:
    if not isinstance(value, list):
        raise ProjectV11DecodeError(f"{path} 必须是 array")
    return tuple(
        decoder(item, f"{path}[{index}]", policy=_V11_FIELD_POLICY)
        for index, item in enumerate(value)
    )


def _encode_values(values: tuple[Any, ...], path: str, encoder: Any) -> list[Any]:
    return [
        encoder(item, f"{path}[{index}]", policy=_V11_FIELD_POLICY)
        for index, item in enumerate(values)
    ]


def _validate_face_sketch_state(
    snapshot: ProjectSnapshot,
    *,
    encode: bool,
) -> None:
    error_type = ProjectV11EncodeError if encode else ProjectV11DecodeError
    undo = tuple(snapshot.face_sketch_boolean_undo_records)
    redo = tuple(snapshot.face_sketch_boolean_redo_records)
    overlap = {
        (record.part_id, record.feature_id) for record in undo
    } & {
        (record.part_id, record.feature_id) for record in redo
    }
    if overlap:
        raise error_type("同一面草图特征不能同时位于撤销栈和重做栈")
    by_part = {part.id: part for part in snapshot.parts}
    for record in (*undo, *redo):
        if record.part_id not in by_part:
            raise error_type(
                f"面草图撤销记录引用了不存在的 Part：{record.part_id}"
            )
        _validate_record_transition(record, snapshot, encode=encode)
    for part_id, live_part in by_part.items():
        undo_stack = tuple(record for record in undo if record.part_id == part_id)
        redo_stack = tuple(record for record in redo if record.part_id == part_id)
        for left, right in zip(undo_stack, undo_stack[1:]):
            if left.after_part != right.before_part:
                raise error_type(f"Part {part_id} 的面草图撤销栈不连续")
        for left, right in zip(redo_stack, redo_stack[1:]):
            if left.before_part != right.after_part:
                raise error_type(f"Part {part_id} 的面草图重做栈不连续")
        expected = (
            redo_stack[-1].before_part
            if redo_stack
            else undo_stack[-1].after_part
            if undo_stack
            else None
        )
        if expected is not None and live_part != expected:
            raise error_type(f"Part {part_id} 的当前状态与面草图撤销栈不一致")


def _validate_record_transition(
    record: FaceSketchBooleanUndoRecord,
    snapshot: ProjectSnapshot,
    *,
    encode: bool,
) -> None:
    error_type = ProjectV11EncodeError if encode else ProjectV11DecodeError
    before_features = _face_features(record.before_part.geometry_recipe)
    after_features = _face_features(record.after_part.geometry_recipe)
    if (
        len(after_features) != len(before_features) + 1
        or after_features[:-1] != before_features
        or after_features[-1].feature_id != record.feature_id
    ):
        raise error_type("面草图撤销记录不是单一特征状态转换")
    for part in (record.before_part, record.after_part):
        _v7._authenticate_part(part, encode=encode)
    other_parts = tuple(
        part for part in snapshot.parts if part.id != record.part_id
    )
    _v7._validate_references(
        (*other_parts, record.before_part), record.before_named_regions
    )
    _v7._validate_references(
        (*other_parts, record.after_part), record.after_named_regions
    )


def _face_features(recipe: object) -> tuple[FaceSketchBooleanGeometry, ...]:
    if isinstance(recipe, FaceSketchBooleanGeometry):
        return (*_face_features(recipe.base), recipe)
    bodies = getattr(recipe, "bodies", None)
    if bodies is not None:
        return tuple(
            feature for body in bodies for feature in _face_features(body.recipe)
        )
    base = getattr(recipe, "base", None)
    return () if base is None else _face_features(base)


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProjectV11DecodeError(f"{path} 必须是 object")
    return value


def _string(value: Any, path: str) -> str:
    if type(value) is not str or not value.strip() or value != value.strip():
        raise ProjectV11DecodeError(f"{path} 必须是无首尾空白的非空 string")
    return value


def _exact_keys(value: Mapping[str, Any], path: str, expected: set[str]) -> None:
    missing = expected - set(value)
    extra = set(value) - expected
    if missing:
        raise ProjectV11DecodeError(f"{path} 缺少字段 {sorted(missing)[0]!r}")
    if extra:
        raise ProjectV11DecodeError(f"{path} 包含未知字段 {sorted(extra)[0]!r}")


read_project_v11 = load_project_v11
write_project_v11 = save_project_v11


__all__ = [
    "FORMAT_NAME",
    "ProjectV11DecodeError",
    "ProjectV11EncodeError",
    "ProjectV11Error",
    "SCHEMA_VERSION",
    "decode_project_v11",
    "dumps_project_v11",
    "encode_project_v11",
    "load_project_v11",
    "loads_project_v11",
    "read_project_v11",
    "save_project_v11",
    "write_project_v11",
]
