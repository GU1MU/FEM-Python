"""Schema-v12 persistence for compact, revision-bound mesh scopes."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any

from fem.application.definitions import (
    CompressedMeshEntityRefs,
    MeshTopologyDirectory,
    NamedRegion,
)
from fem.application.session import ProjectSaveSnapshot, ProjectSnapshot

from . import project_v7 as _v7
from ._project_codec import (
    atomic_write_project,
    dumps_canonical_json,
    loads_json_strict,
    unwrap_project_snapshot,
)
from ._project_errors import ProjectDecodeError, ProjectEncodeError, ProjectError
from .project_v11 import decode_project_v11, encode_project_v11


SCHEMA_VERSION = 12
FORMAT_NAME = "fem-python-project"
_TOPOLOGY_ID = "mesh-v1"


class ProjectV12Error(ProjectError):
    """Base error for schema-v12 processing."""


class ProjectV12DecodeError(ProjectV12Error, ProjectDecodeError):
    """A schema-v12 payload is malformed."""


class ProjectV12EncodeError(ProjectV12Error, ProjectEncodeError):
    """A snapshot cannot be represented losslessly by schema v12."""


def loads_project_v12(
    data: str | bytes | bytearray,
    *,
    source_path: str | Path | None = None,
) -> ProjectSnapshot:
    return decode_project_v12(
        loads_json_strict(
            data,
            error_type=ProjectV12DecodeError,
            document_label="v12 项目",
        ),
        source_path=source_path,
    )


def decode_project_v12(
    payload: Mapping[str, Any] | str | bytes | bytearray,
    *,
    source_path: str | Path | None = None,
) -> ProjectSnapshot:
    if isinstance(payload, (str, bytes, bytearray)):
        return loads_project_v12(payload, source_path=source_path)
    try:
        root = _mapping(payload, "$")
        _exact_keys(root, "$", {"format", "schema", "project"})
        if root["format"] != FORMAT_NAME:
            raise ProjectV12DecodeError(
                f"$.format 必须精确等于 {FORMAT_NAME!r}"
            )
        if type(root["schema"]) is not int or root["schema"] != SCHEMA_VERSION:
            raise ProjectV12DecodeError(
                f"v12 decoder 不能读取 schema {root['schema']!r}"
            )
        v11_payload = deepcopy(dict(root))
        project = _mapping(v11_payload["project"], "$.project")
        _mapping(project.get("authoring"), "$.project.authoring")
        authoring = v11_payload["project"]["authoring"]
        topology_rows = _decode_topology_directory(
            authoring.pop("mesh_scope_topology", None),
            "$.project.authoring.mesh_scope_topology",
        )
        directory = MeshTopologyDirectory(None, topology_rows)
        v11_payload["schema"] = 11
        with _v7._compact_region_decoding(
            lambda data, path: _decode_compact_region(
                data,
                directory,
                path,
            )
        ):
            return decode_project_v11(v11_payload, source_path=source_path)
    except ProjectV12Error:
        raise
    except Exception as error:
        raise ProjectV12DecodeError(f"schema v12 项目无效：{error}") from error


def encode_project_v12(
    snapshot: ProjectSnapshot | ProjectSaveSnapshot,
) -> dict[str, Any]:
    try:
        project = unwrap_project_snapshot(snapshot)
        topology_rows: dict[
            tuple[str | None, str, int, int], tuple[int, ...]
        ] = {}
        with _v7._compact_region_encoding(
            lambda region, path: _encode_compact_region(
                region,
                topology_rows,
                path,
            )
        ):
            payload = encode_project_v11(project)
        payload["schema"] = SCHEMA_VERSION
        _compress_regions(payload["project"], topology_rows, "$.project")
        authoring = payload["project"]["authoring"]
        authoring["mesh_scope_topology"] = _encode_topology_directory(
            topology_rows
        )
        dumps_canonical_json(payload, error_type=ProjectV12EncodeError)
        return payload
    except ProjectV12Error:
        raise
    except Exception as error:
        raise ProjectV12EncodeError(
            f"snapshot 无法由 v12 无损表示：{error}"
        ) from error


def dumps_project_v12(snapshot: ProjectSnapshot | ProjectSaveSnapshot) -> str:
    return dumps_canonical_json(
        encode_project_v12(snapshot),
        error_type=ProjectV12EncodeError,
    )


def load_project_v12(path: str | Path) -> ProjectSnapshot:
    source = Path(path)
    return loads_project_v12(source.read_bytes(), source_path=source)


def save_project_v12(
    path: str | Path,
    snapshot: ProjectSnapshot | ProjectSaveSnapshot,
    *,
    checkpoint: Callable[[], Any] | None = None,
) -> Path:
    if checkpoint is not None:
        checkpoint()
    payload = encode_project_v12(snapshot)
    if checkpoint is not None:
        checkpoint()
    serialized = dumps_canonical_json(
        payload,
        error_type=ProjectV12EncodeError,
    )
    return atomic_write_project(
        path,
        serialized,
        verifier=load_project_v12,
        semantic_encoder=encode_project_v12,
        expected_semantic=payload,
        error_type=ProjectV12EncodeError,
        mismatch_message="临时 v12 项目回读后的压缩作用域状态不一致",
        checkpoint=checkpoint,
    )


def _compress_regions(
    value: Any,
    topology_rows: dict[
        tuple[str | None, str, int, int], tuple[int, ...]
    ],
    path: str,
) -> None:
    if isinstance(value, dict):
        if set(value) == {"name", "references"} and isinstance(
            value["references"], list
        ):
            references = value["references"]
            if references and all(
                isinstance(reference, Mapping)
                and reference.get("kind") in {"node", "edge", "face", "element"}
                for reference in references
            ):
                value["compact_references"] = _compact_reference_payload(
                    references,
                    topology_rows,
                    f"{path}.references",
                )
                del value["references"]
                return
        for key, item in tuple(value.items()):
            _compress_regions(item, topology_rows, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _compress_regions(item, topology_rows, f"{path}[{index}]")


def _compact_reference_payload(
    references: list[Mapping[str, Any]],
    topology_rows: dict[
        tuple[str | None, str, int, int], tuple[int, ...]
    ],
    path: str,
) -> dict[str, Any]:
    kind = references[0]["kind"]
    if any(reference["kind"] != kind for reference in references):
        raise ProjectV12EncodeError(f"{path} 混用了网格实体类型")
    groups: list[dict[str, Any]] = []
    part_ids = sorted(
        {reference.get("part_id") for reference in references},
        key=lambda value: "" if value is None else value,
    )
    for part_id in part_ids:
        selected = [
            reference
            for reference in references
            if reference.get("part_id") == part_id
        ]
        group: dict[str, Any] = {"part_id": part_id}
        if kind in {"node", "element"}:
            field = "node_id" if kind == "node" else "element_id"
            ids = [int(reference[field]) for reference in selected]
            ranges: list[list[int]] = []
            start = previous = ids[0]
            for identity in ids[1:]:
                if identity == previous + 1:
                    previous = identity
                    continue
                ranges.append([start, previous])
                start = previous = identity
            ranges.append([start, previous])
            group["ranges"] = ranges
        else:
            identities: list[list[int]] = []
            for reference in selected:
                element_id = int(reference["element_id"])
                local_index = int(reference["local_index"])
                identities.append([element_id, local_index])
                key = (part_id, kind, element_id, local_index)
                node_ids = tuple(int(value) for value in reference["node_ids"])
                previous_nodes = topology_rows.get(key)
                if previous_nodes is not None and previous_nodes != node_ids:
                    raise ProjectV12EncodeError(
                        f"{path} 在共享拓扑目录中存在冲突引用 {key!r}"
                    )
                topology_rows[key] = node_ids
            group["identities"] = identities
        groups.append(group)
    result: dict[str, Any] = {"kind": kind, "groups": groups}
    if kind in {"edge", "face"}:
        result["topology_id"] = _TOPOLOGY_ID
    return result


def _encode_compact_region(
    region: NamedRegion,
    topology_rows: dict[
        tuple[str | None, str, int, int], tuple[int, ...]
    ],
    path: str,
) -> dict[str, Any]:
    references = region.references
    if not isinstance(references, CompressedMeshEntityRefs):
        raise ProjectV12EncodeError(f"{path} 缺少压缩网格引用")
    groups: list[dict[str, Any]] = []
    for part_id, packed in references.compact_groups():
        group: dict[str, Any] = {"part_id": part_id}
        pairs = [list(packed[index : index + 2]) for index in range(0, len(packed), 2)]
        if references.kind in {"node", "element"}:
            group["ranges"] = pairs
        else:
            group["identities"] = pairs
            topology = references.topology
            if topology is None:
                raise ProjectV12EncodeError(
                    f"{path} 的边/面引用缺少拓扑目录"
                )
            for element_id, local_index in pairs:
                key = (
                    part_id,
                    references.kind,
                    element_id,
                    local_index,
                )
                node_ids = topology.resolve(
                    *key,
                    mesh_revision=references.mesh_revision,
                )
                previous = topology_rows.get(key)
                if previous is not None and previous != node_ids:
                    raise ProjectV12EncodeError(
                        f"{path} 在共享拓扑目录中存在冲突引用 {key!r}"
                    )
                topology_rows[key] = node_ids
        groups.append(group)
    compact: dict[str, Any] = {
        "kind": references.kind,
        "groups": groups,
    }
    if references.kind in {"edge", "face"}:
        compact["topology_id"] = _TOPOLOGY_ID
    return {"name": region.name, "compact_references": compact}


def _inflate_regions(
    value: Any,
    topology_rows: Mapping[
        tuple[str | None, str, int, int], tuple[int, ...]
    ],
    path: str,
) -> None:
    if isinstance(value, dict):
        if set(value) == {"name", "compact_references"}:
            value["references"] = _inflate_reference_payload(
                value.pop("compact_references"),
                topology_rows,
                f"{path}.compact_references",
            )
            return
        for key, item in tuple(value.items()):
            _inflate_regions(item, topology_rows, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _inflate_regions(item, topology_rows, f"{path}[{index}]")


def _inflate_reference_payload(
    value: Any,
    topology_rows: Mapping[
        tuple[str | None, str, int, int], tuple[int, ...]
    ],
    path: str,
) -> list[dict[str, Any]]:
    data = _mapping(value, path)
    kind = data.get("kind")
    if kind not in {"node", "edge", "face", "element"}:
        raise ProjectV12DecodeError(f"{path}.kind 不受支持")
    expected = {"kind", "groups"}
    if kind in {"edge", "face"}:
        expected.add("topology_id")
    _exact_keys(data, path, expected)
    if kind in {"edge", "face"} and data["topology_id"] != _TOPOLOGY_ID:
        raise ProjectV12DecodeError(f"{path}.topology_id 不存在")
    groups = _array(data["groups"], f"{path}.groups")
    references: list[dict[str, Any]] = []
    previous_part_key: str | None = None
    for index, raw_group in enumerate(groups):
        group_path = f"{path}.groups[{index}]"
        group = _mapping(raw_group, group_path)
        required = {"part_id", "ranges" if kind in {"node", "element"} else "identities"}
        _exact_keys(group, group_path, required)
        part_id = group["part_id"]
        if part_id is not None and type(part_id) is not str:
            raise ProjectV12DecodeError(f"{group_path}.part_id 必须是 string 或 null")
        part_key = "" if part_id is None else part_id
        if previous_part_key is not None and previous_part_key >= part_key:
            raise ProjectV12DecodeError(f"{path}.groups 不是 canonical 顺序")
        previous_part_key = part_key
        if kind in {"node", "element"}:
            last_end: int | None = None
            for pair_index, raw_pair in enumerate(
                _array(group["ranges"], f"{group_path}.ranges")
            ):
                start, end = _integer_pair(
                    raw_pair,
                    f"{group_path}.ranges[{pair_index}]",
                )
                if start > end or (last_end is not None and start <= last_end + 1):
                    raise ProjectV12DecodeError(
                        f"{group_path}.ranges 不是 canonical 区间"
                    )
                for identity in range(start, end + 1):
                    reference = {"kind": kind}
                    if part_id is not None:
                        reference["part_id"] = part_id
                    reference["node_id" if kind == "node" else "element_id"] = identity
                    references.append(reference)
                last_end = end
        else:
            previous_pair: tuple[int, int] | None = None
            for pair_index, raw_pair in enumerate(
                _array(group["identities"], f"{group_path}.identities")
            ):
                pair = _integer_pair(
                    raw_pair,
                    f"{group_path}.identities[{pair_index}]",
                )
                if previous_pair is not None and previous_pair >= pair:
                    raise ProjectV12DecodeError(
                        f"{group_path}.identities 不是 canonical 顺序"
                    )
                key = (part_id, kind, *pair)
                if key not in topology_rows:
                    raise ProjectV12DecodeError(
                        f"{group_path}.identities 缺少共享拓扑 {key!r}"
                    )
                reference = {
                    "kind": kind,
                    "element_id": pair[0],
                    "local_index": pair[1],
                    "node_ids": list(topology_rows[key]),
                }
                if part_id is not None:
                    reference["part_id"] = part_id
                references.append(reference)
                previous_pair = pair
    if not references:
        raise ProjectV12DecodeError(f"{path} 必须包含至少一个引用")
    return references


def _decode_compact_region(
    value: Mapping[str, Any],
    directory: MeshTopologyDirectory,
    path: str,
) -> NamedRegion:
    data = _mapping(value, path)
    _exact_keys(data, path, {"name", "compact_references"})
    name = data["name"]
    if type(name) is not str or not name.strip() or name != name.strip():
        raise ProjectV12DecodeError(
            f"{path}.name 必须是无首尾空白的非空 string"
        )
    compact_path = f"{path}.compact_references"
    compact = _mapping(data["compact_references"], compact_path)
    kind = compact.get("kind")
    if kind not in {"node", "edge", "face", "element"}:
        raise ProjectV12DecodeError(f"{compact_path}.kind 不受支持")
    expected = {"kind", "groups"}
    if kind in {"edge", "face"}:
        expected.add("topology_id")
    _exact_keys(compact, compact_path, expected)
    if kind in {"edge", "face"} and compact["topology_id"] != _TOPOLOGY_ID:
        raise ProjectV12DecodeError(f"{compact_path}.topology_id 不存在")
    groups: list[tuple[str | None, tuple[int, ...]]] = []
    for index, raw_group in enumerate(
        _array(compact["groups"], f"{compact_path}.groups")
    ):
        group_path = f"{compact_path}.groups[{index}]"
        group = _mapping(raw_group, group_path)
        field = "ranges" if kind in {"node", "element"} else "identities"
        _exact_keys(group, group_path, {"part_id", field})
        part_id = group["part_id"]
        if part_id is not None and type(part_id) is not str:
            raise ProjectV12DecodeError(
                f"{group_path}.part_id 必须是 string 或 null"
            )
        packed = tuple(
            value
            for pair_index, raw_pair in enumerate(
                _array(group[field], f"{group_path}.{field}")
            )
            for value in _integer_pair(
                raw_pair,
                f"{group_path}.{field}[{pair_index}]",
            )
        )
        groups.append((part_id, packed))
    topology = directory if kind in {"edge", "face"} else None
    references = CompressedMeshEntityRefs.from_compact(
        kind,
        groups,
        mesh_revision=None,
        topology=topology,
    )
    if topology is not None:
        for part_id, packed in references.compact_groups():
            for offset in range(0, len(packed), 2):
                topology.resolve(
                    part_id,
                    kind,
                    packed[offset],
                    packed[offset + 1],
                    mesh_revision=None,
                )
    return NamedRegion(name, references)


def _encode_topology_directory(
    rows: Mapping[tuple[str | None, str, int, int], tuple[int, ...]],
) -> dict[str, Any]:
    return {
        "id": _TOPOLOGY_ID,
        "rows": [
            {
                "part_id": part_id,
                "kind": kind,
                "element_id": element_id,
                "local_index": local_index,
                "node_ids": list(node_ids),
            }
            for (part_id, kind, element_id, local_index), node_ids in sorted(
                rows.items(),
                key=lambda item: (
                    "" if item[0][0] is None else item[0][0],
                    0 if item[0][1] == "edge" else 1,
                    item[0][2],
                    item[0][3],
                ),
            )
        ],
    }


def _decode_topology_directory(
    value: Any,
    path: str,
) -> dict[tuple[str | None, str, int, int], tuple[int, ...]]:
    data = _mapping(value, path)
    _exact_keys(data, path, {"id", "rows"})
    if data["id"] != _TOPOLOGY_ID:
        raise ProjectV12DecodeError(f"{path}.id 不受支持")
    result: dict[tuple[str | None, str, int, int], tuple[int, ...]] = {}
    for index, raw_row in enumerate(_array(data["rows"], f"{path}.rows")):
        row_path = f"{path}.rows[{index}]"
        row = _mapping(raw_row, row_path)
        _exact_keys(
            row,
            row_path,
            {"part_id", "kind", "element_id", "local_index", "node_ids"},
        )
        part_id = row["part_id"]
        if part_id is not None and type(part_id) is not str:
            raise ProjectV12DecodeError(f"{row_path}.part_id 必须是 string 或 null")
        kind = row["kind"]
        if kind not in {"edge", "face"}:
            raise ProjectV12DecodeError(f"{row_path}.kind 只接受 edge/face")
        key = (
            part_id,
            kind,
            _integer(row["element_id"], f"{row_path}.element_id"),
            _integer(row["local_index"], f"{row_path}.local_index"),
        )
        node_ids = tuple(
            _integer(item, f"{row_path}.node_ids[{node_index}]")
            for node_index, item in enumerate(
                _array(row["node_ids"], f"{row_path}.node_ids")
            )
        )
        if not node_ids or len(set(node_ids)) != len(node_ids) or key in result:
            raise ProjectV12DecodeError(f"{row_path} 不是唯一有效的拓扑行")
        result[key] = node_ids
    return result


def _share_topology_directory(
    snapshot: ProjectSnapshot,
    directory: MeshTopologyDirectory,
) -> ProjectSnapshot:
    def shared_region(region: NamedRegion) -> NamedRegion:
        references = region.references
        if not isinstance(references, CompressedMeshEntityRefs):
            return region
        topology = directory if references.kind in {"edge", "face"} else None
        compact = CompressedMeshEntityRefs.from_compact(
            references.kind,
            references.compact_groups(),
            mesh_revision=None,
            topology=topology,
        )
        return NamedRegion(region.name, compact)

    def shared_regions(regions: Any) -> tuple[NamedRegion, ...]:
        return tuple(shared_region(region) for region in regions)

    updates: dict[str, Any] = {
        "named_regions": shared_regions(snapshot.named_regions)
    }
    for field_name in (
        "boolean_reference_undo_records",
        "part_boolean_undo_records",
        "face_sketch_boolean_undo_records",
        "face_sketch_boolean_redo_records",
    ):
        records = []
        for record in getattr(snapshot, field_name):
            records.append(
                replace(
                    record,
                    before_named_regions=shared_regions(record.before_named_regions),
                    after_named_regions=shared_regions(record.after_named_regions),
                )
            )
        updates[field_name] = tuple(records)
    return replace(snapshot, **updates)


def _mapping(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ProjectV12DecodeError(f"{path} 必须是 object")
    if any(type(key) is not str for key in value):
        raise ProjectV12DecodeError(f"{path} 的字段名必须是 string")
    return dict(value)


def _array(value: Any, path: str) -> list[Any]:
    if not isinstance(value, list):
        raise ProjectV12DecodeError(f"{path} 必须是 array")
    return value


def _integer(value: Any, path: str) -> int:
    if type(value) is not int:
        raise ProjectV12DecodeError(f"{path} 必须是严格整数")
    return value


def _integer_pair(value: Any, path: str) -> tuple[int, int]:
    items = _array(value, path)
    if len(items) != 2:
        raise ProjectV12DecodeError(f"{path} 必须包含两个整数")
    return _integer(items[0], f"{path}[0]"), _integer(items[1], f"{path}[1]")


def _exact_keys(value: Mapping[str, Any], path: str, expected: set[str]) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        details = []
        if missing:
            details.append(f"缺少 {missing!r}")
        if extra:
            details.append(f"多出 {extra!r}")
        raise ProjectV12DecodeError(f"{path} 字段不匹配：{'; '.join(details)}")


read_project_v12 = load_project_v12
write_project_v12 = save_project_v12


__all__ = [
    "FORMAT_NAME",
    "SCHEMA_VERSION",
    "ProjectV12DecodeError",
    "ProjectV12EncodeError",
    "ProjectV12Error",
    "decode_project_v12",
    "dumps_project_v12",
    "encode_project_v12",
    "load_project_v12",
    "loads_project_v12",
    "read_project_v12",
    "save_project_v12",
    "write_project_v12",
]
