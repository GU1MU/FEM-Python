"""Lossless, detached persistence for native ``.femproj`` v1 projects.

The codec deliberately has no knowledge of a live :class:`ModelSession`.  A
load fully decodes and validates a detached ``ProjectSnapshot``; installing
that snapshot and accepting a completed save remain application-layer
transactions.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import fields
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, NoReturn, TYPE_CHECKING

from fem.application.definitions import (
    FeatureRecord,
    NamedRegion,
    NativePart,
    RegionAssignment,
    SectionDefinition,
)
from fem.core.model import (
    AnalysisStep,
    DisplacementConstraint,
    EdgeLoad,
    GravityLoad,
    LineLoad,
    MaterialDefinition,
    NodalLoad,
    OutputRequest,
    SurfaceLoad,
)
from fem.geometry.recipes import (
    BooleanGeometry,
    BoxGeometry,
    CylinderGeometry,
    DiskGeometry,
    ExtrudedGeometry,
    MovedGeometry,
    PlateWithHoleGeometry,
    RectangleGeometry,
    RotatedGeometry,
    SketchCircle,
    SketchGeometry,
    SketchRectangle,
)
from fem.mesh.settings import LocalMeshControl, MeshSettings

if TYPE_CHECKING:
    from fem.application.session import ProjectSaveSnapshot, ProjectSnapshot


SCHEMA_VERSION = 1
LOGICAL_TOPOLOGY_VERSION = 1


class ProjectV1Error(ValueError):
    """Base error for a project that cannot be represented losslessly."""


class ProjectV1DecodeError(ProjectV1Error):
    """The serialized project is malformed, incomplete, or unsupported."""


class ProjectV1EncodeError(ProjectV1Error):
    """The snapshot contains state that v1 cannot encode losslessly."""


def loads_project_v1(
    data: str | bytes | bytearray,
    *,
    source_path: str | Path | None = None,
) -> ProjectSnapshot:
    """Decode a complete JSON document into a detached project snapshot."""

    if isinstance(data, (bytes, bytearray)):
        try:
            text = bytes(data).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ProjectV1DecodeError("项目文件不是有效的 UTF-8 文本") from exc
    elif isinstance(data, str):
        text = data
    else:
        raise TypeError("data 必须是 str、bytes 或 bytearray")

    try:
        payload = json.loads(
            text,
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except ProjectV1DecodeError:
        raise
    except (json.JSONDecodeError, RecursionError) as exc:
        raise ProjectV1DecodeError(f"项目 JSON 无效：{exc}") from exc
    return decode_project_v1(payload, source_path=source_path)


def decode_project_v1(
    payload: Mapping[str, Any] | str | bytes | bytearray,
    *,
    source_path: str | Path | None = None,
) -> ProjectSnapshot:
    """Validate a parsed v1 payload and return a detached snapshot.

    No caller-owned mapping is retained and no live application state is
    consulted or modified.
    """

    if isinstance(payload, (str, bytes, bytearray)):
        return loads_project_v1(payload, source_path=source_path)
    root = _mapping(payload, "$", error_type=ProjectV1DecodeError)
    _keys(
        root,
        "$",
        required={"schema", "source", "geometry"},
        optional={
            "logical_topology_version",
            "parts",
            "mesh_settings",
            "feature_history",
            "named_regions",
            "materials",
            "sections",
            "assignments",
            "steps",
        },
        error_type=ProjectV1DecodeError,
    )
    schema = _integer(root["schema"], "$.schema", error_type=ProjectV1DecodeError)
    if schema != SCHEMA_VERSION:
        raise ProjectV1DecodeError(f"不支持的项目 schema：{schema!r}")
    topology_version = (
        None
        if "logical_topology_version" not in root
        else _integer(
            root["logical_topology_version"],
            "$.logical_topology_version",
            error_type=ProjectV1DecodeError,
        )
    )
    if topology_version not in {None, LOGICAL_TOPOLOGY_VERSION}:
        raise ProjectV1DecodeError(
            f"不支持的逻辑拓扑契约版本：{topology_version!r}"
        )
    source_kind = _string(root["source"], "$.source", error_type=ProjectV1DecodeError)
    if source_kind != "native":
        raise ProjectV1DecodeError("v1 项目只支持 source='native'")

    geometry = _decode_geometry(root["geometry"], "$.geometry")
    mesh_settings = _decode_mesh_settings(root.get("mesh_settings"), "$.mesh_settings")
    parts = tuple(
        _decode_part(item, f"$.parts[{index}]")
        for index, item in enumerate(
            _array(root.get("parts", ()), "$.parts", error_type=ProjectV1DecodeError)
        )
    ) or (NativePart(),)

    if "feature_history" in root:
        feature_history = tuple(
            _decode_feature(item, f"$.feature_history[{index}]")
            for index, item in enumerate(
                _array(
                    root["feature_history"],
                    "$.feature_history",
                    error_type=ProjectV1DecodeError,
                )
            )
        )
    else:
        feature_history = tuple(_history_for_recipe(geometry))

    named_regions = tuple(
        _decode_named_region(item, f"$.named_regions[{index}]")
        for index, item in enumerate(
            _array(
                root.get("named_regions", ()),
                "$.named_regions",
                error_type=ProjectV1DecodeError,
            )
        )
    )
    _require_unique_names(named_regions, "$.named_regions")
    if topology_version is None and (
        named_regions
        or (
            mesh_settings is not None
            and bool(mesh_settings.local_controls)
        )
    ):
        raise ProjectV1DecodeError(
            "旧项目缺少逻辑拓扑契约版本，无法安全恢复命名区域或局部网格控制；"
            "请移除这些实体引用后打开项目并重新选择"
        )

    materials = tuple(
        _decode_material(item, f"$.materials[{index}]")
        for index, item in enumerate(
            _array(
                root.get("materials", ()),
                "$.materials",
                error_type=ProjectV1DecodeError,
            )
        )
    )
    _require_unique_names(materials, "$.materials")

    sections = tuple(
        _decode_section(item, f"$.sections[{index}]")
        for index, item in enumerate(
            _array(
                root.get("sections", ()),
                "$.sections",
                error_type=ProjectV1DecodeError,
            )
        )
    )
    _require_unique_names(sections, "$.sections")

    assignments = tuple(
        _decode_assignment(item, f"$.assignments[{index}]")
        for index, item in enumerate(
            _array(
                root.get("assignments", ()),
                "$.assignments",
                error_type=ProjectV1DecodeError,
            )
        )
    )
    steps = tuple(
        _decode_step(item, f"$.steps[{index}]")
        for index, item in enumerate(
            _array(root.get("steps", ()), "$.steps", error_type=ProjectV1DecodeError)
        )
    )
    _require_unique_names(steps, "$.steps")

    try:
        from fem.application.session import ProjectSnapshot

        return ProjectSnapshot(
            source_kind="native",
            source_path=None if source_path is None else Path(source_path),
            parts=parts,
            geometry_recipe=geometry,
            mesh_settings=mesh_settings,
            feature_history=feature_history,
            named_regions=named_regions,
            material_definitions=materials,
            section_definitions=sections,
            region_assignments=assignments,
            analysis_definitions=steps,
        )
    except ProjectV1Error:
        raise
    except (TypeError, ValueError) as exc:
        raise ProjectV1DecodeError(f"项目 snapshot 无效：{exc}") from exc


def load_project_v1(path: str | Path) -> ProjectSnapshot:
    """Read and fully decode a project without modifying a live Session."""

    source = Path(path)
    try:
        data = source.read_bytes()
    except OSError:
        raise
    return loads_project_v1(data, source_path=source)


def encode_project_v1(
    snapshot: ProjectSnapshot | ProjectSaveSnapshot,
) -> dict[str, Any]:
    """Encode a detached/save snapshot as a lossless v1 JSON object."""

    project = _unwrap_project_snapshot(snapshot)
    source_kind = _snapshot_attr(project, "source_kind")
    if source_kind != "native":
        raise ProjectV1EncodeError("v1 项目只支持保存 native Session")

    geometry = _snapshot_attr(project, "geometry_recipe")
    if geometry is None:
        raise ProjectV1EncodeError("请先创建草图或几何后再保存项目")

    parts = _snapshot_sequence(project, "parts")
    features = _snapshot_sequence(project, "feature_history")
    named_regions = _snapshot_sequence(project, "named_regions", mapping_values=True)
    materials = _snapshot_sequence(project, "material_definitions")
    sections = _snapshot_sequence(project, "section_definitions")
    assignments = _snapshot_sequence(project, "region_assignments")
    steps = _snapshot_sequence(project, "analysis_definitions")

    if not parts:
        raise ProjectV1EncodeError("native 项目至少需要一个 NativePart")
    _require_unique_names(named_regions, "snapshot.named_regions", encode=True)
    _require_unique_names(materials, "snapshot.material_definitions", encode=True)
    _require_unique_names(sections, "snapshot.section_definitions", encode=True)
    _require_unique_names(steps, "snapshot.analysis_definitions", encode=True)

    payload = {
        "schema": SCHEMA_VERSION,
        "logical_topology_version": LOGICAL_TOPOLOGY_VERSION,
        "source": "native",
        "parts": [
            _encode_part(item, f"snapshot.parts[{index}]")
            for index, item in enumerate(parts)
        ],
        "geometry": _encode_geometry(geometry, "snapshot.geometry_recipe", set()),
        "mesh_settings": _encode_mesh_settings(
            _snapshot_attr(project, "mesh_settings"),
            "snapshot.mesh_settings",
        ),
        "feature_history": [
            _encode_feature(item, f"snapshot.feature_history[{index}]")
            for index, item in enumerate(features)
        ],
        "named_regions": [
            _encode_named_region(item, f"snapshot.named_regions[{index}]")
            for index, item in enumerate(named_regions)
        ],
        "materials": [
            _encode_material(item, f"snapshot.material_definitions[{index}]")
            for index, item in enumerate(materials)
        ],
        "sections": [
            _encode_section(item, f"snapshot.section_definitions[{index}]")
            for index, item in enumerate(sections)
        ],
        "assignments": [
            _encode_assignment(item, f"snapshot.region_assignments[{index}]")
            for index, item in enumerate(assignments)
        ],
        "steps": [
            _encode_step(item, f"snapshot.analysis_definitions[{index}]")
            for index, item in enumerate(steps)
        ],
    }
    # This catches non-finite numbers and excessive/cyclic JSON metadata before
    # any destination file is touched.
    try:
        json.dumps(payload, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError, RecursionError) as exc:
        raise ProjectV1EncodeError(f"项目包含无法无损编码的 JSON 值：{exc}") from exc
    return payload


def dumps_project_v1(
    snapshot: ProjectSnapshot | ProjectSaveSnapshot,
    *,
    indent: int | None = 2,
) -> str:
    """Serialize a project snapshot as deterministic UTF-8 JSON text."""

    payload = encode_project_v1(snapshot)
    try:
        return json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            indent=indent,
        )
    except (TypeError, ValueError, RecursionError) as exc:
        raise ProjectV1EncodeError(f"项目包含无法编码的值：{exc}") from exc


def save_project_v1(
    path: str | Path,
    snapshot: ProjectSnapshot | ProjectSaveSnapshot,
) -> Path:
    """Atomically save an immutable project snapshot.

    The temporary file is created beside the target, flushed to disk, decoded
    again for complete validation, and only then installed with ``os.replace``.
    """

    target = Path(path)
    serialized = dumps_project_v1(snapshot)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    installed = False
    try:
        stream = os.fdopen(descriptor, "w", encoding="utf-8", newline="\n")
        descriptor = -1
        with stream:
            stream.write(serialized)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())

        # Re-read the bytes that will actually be installed.  This validates
        # UTF-8, JSON structure, all v1 fields, and snapshot construction.
        verified = load_project_v1(temporary)
        if encode_project_v1(verified) != encode_project_v1(snapshot):
            raise ProjectV1EncodeError("临时项目文件校验后与保存 snapshot 不一致")

        os.replace(temporary, target)
        installed = True
        return target
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if not installed:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass


# Explicit file-oriented aliases make call sites read naturally while keeping
# the primary names symmetric with ``json.loads`` / ``json.dumps``.
read_project_v1 = load_project_v1
write_project_v1 = save_project_v1
load_native_project = load_project_v1
save_native_project = save_project_v1


def _decode_part(value: Any, path: str) -> NativePart:
    data = _mapping(value, path, error_type=ProjectV1DecodeError)
    _keys(
        data,
        path,
        required=set(),
        optional={"name", "body_name"},
        error_type=ProjectV1DecodeError,
    )
    return _construct_decode(
        NativePart,
        path,
        name=_string(
            data.get("name", "Part-1"), f"{path}.name", error_type=ProjectV1DecodeError
        ),
        body_name=_string(
            data.get("body_name", "Body-1"),
            f"{path}.body_name",
            error_type=ProjectV1DecodeError,
        ),
    )


def _decode_feature(value: Any, path: str) -> FeatureRecord:
    data = _mapping(value, path, error_type=ProjectV1DecodeError)
    _keys(
        data,
        path,
        required={"name", "kind"},
        optional={"payload"},
        error_type=ProjectV1DecodeError,
    )
    return _construct_decode(
        FeatureRecord,
        path,
        name=_string(data["name"], f"{path}.name", error_type=ProjectV1DecodeError),
        kind=_string(data["kind"], f"{path}.kind", error_type=ProjectV1DecodeError),
        payload=_json_object(
            data.get("payload", {}),
            f"{path}.payload",
            error_type=ProjectV1DecodeError,
        ),
    )


def _decode_named_region(value: Any, path: str) -> NamedRegion:
    data = _mapping(value, path, error_type=ProjectV1DecodeError)
    _keys(
        data,
        path,
        required={"name", "entity_kind"},
        optional={"entity_ids"},
        error_type=ProjectV1DecodeError,
    )
    entity_kind = _string(
        data["entity_kind"], f"{path}.entity_kind", error_type=ProjectV1DecodeError
    )
    if entity_kind not in {"point", "edge", "face", "body"}:
        raise ProjectV1DecodeError(f"{path}.entity_kind 不是受支持的实体类型")
    entity_ids = tuple(
        _positive_integer(item, f"{path}.entity_ids[{index}]", ProjectV1DecodeError)
        for index, item in enumerate(
            _array(
                data.get("entity_ids", ()),
                f"{path}.entity_ids",
                error_type=ProjectV1DecodeError,
            )
        )
    )
    if len(set(entity_ids)) != len(entity_ids):
        raise ProjectV1DecodeError(f"{path}.entity_ids 包含重复实体编号")
    return _construct_decode(
        NamedRegion,
        path,
        name=_string(data["name"], f"{path}.name", error_type=ProjectV1DecodeError),
        entity_kind=entity_kind,
        entity_ids=entity_ids,
    )


def _decode_material(value: Any, path: str) -> MaterialDefinition:
    data = _mapping(value, path, error_type=ProjectV1DecodeError)
    _keys(
        data,
        path,
        required={"name"},
        optional={"properties"},
        error_type=ProjectV1DecodeError,
    )
    return _construct_decode(
        MaterialDefinition,
        path,
        name=_string(data["name"], f"{path}.name", error_type=ProjectV1DecodeError),
        properties=_json_object(
            data.get("properties", {}),
            f"{path}.properties",
            error_type=ProjectV1DecodeError,
        ),
    )


def _decode_section(value: Any, path: str) -> SectionDefinition:
    data = _mapping(value, path, error_type=ProjectV1DecodeError)
    _keys(
        data,
        path,
        required={"name", "material"},
        optional={"section_type", "properties"},
        error_type=ProjectV1DecodeError,
    )
    return _construct_decode(
        SectionDefinition,
        path,
        name=_string(data["name"], f"{path}.name", error_type=ProjectV1DecodeError),
        material=_string(
            data["material"], f"{path}.material", error_type=ProjectV1DecodeError
        ),
        section_type=_string(
            data.get("section_type", "solid"),
            f"{path}.section_type",
            error_type=ProjectV1DecodeError,
        ),
        properties=_json_object(
            data.get("properties", {}),
            f"{path}.properties",
            error_type=ProjectV1DecodeError,
        ),
    )


def _decode_assignment(value: Any, path: str) -> RegionAssignment:
    data = _mapping(value, path, error_type=ProjectV1DecodeError)
    _keys(
        data,
        path,
        required={"section_name", "region_name"},
        optional=set(),
        error_type=ProjectV1DecodeError,
    )
    return _construct_decode(
        RegionAssignment,
        path,
        section_name=_string(
            data["section_name"],
            f"{path}.section_name",
            error_type=ProjectV1DecodeError,
        ),
        region_name=_string(
            data["region_name"],
            f"{path}.region_name",
            error_type=ProjectV1DecodeError,
        ),
    )


def _decode_geometry(value: Any, path: str) -> Any:
    data = _mapping(value, path, error_type=ProjectV1DecodeError)
    kind = _string(
        data.get("type"), f"{path}.type", error_type=ProjectV1DecodeError
    )
    if kind == "SketchGeometry":
        _keys(
            data,
            path,
            required={"type", "name", "contours"},
            optional=set(),
            error_type=ProjectV1DecodeError,
        )
        contours = tuple(
            _decode_contour(item, f"{path}.contours[{index}]")
            for index, item in enumerate(
                _array(
                    data["contours"],
                    f"{path}.contours",
                    error_type=ProjectV1DecodeError,
                )
            )
        )
        return _construct_decode(
            SketchGeometry,
            path,
            _string(data["name"], f"{path}.name", error_type=ProjectV1DecodeError),
            contours,
        )
    if kind == "RectangleGeometry":
        _keys(
            data,
            path,
            required={"type", "name", "width", "height"},
            optional=set(),
            error_type=ProjectV1DecodeError,
        )
        return _construct_decode(
            RectangleGeometry,
            path,
            _string(data["name"], f"{path}.name", error_type=ProjectV1DecodeError),
            _number(data["width"], f"{path}.width", ProjectV1DecodeError),
            _number(data["height"], f"{path}.height", ProjectV1DecodeError),
        )
    if kind == "DiskGeometry":
        _keys(
            data,
            path,
            required={"type", "name", "radius"},
            optional=set(),
            error_type=ProjectV1DecodeError,
        )
        return _construct_decode(
            DiskGeometry,
            path,
            _string(data["name"], f"{path}.name", error_type=ProjectV1DecodeError),
            _number(data["radius"], f"{path}.radius", ProjectV1DecodeError),
        )
    if kind == "BoxGeometry":
        _keys(
            data,
            path,
            required={"type", "name", "width", "depth", "height"},
            optional=set(),
            error_type=ProjectV1DecodeError,
        )
        return _construct_decode(
            BoxGeometry,
            path,
            _string(data["name"], f"{path}.name", error_type=ProjectV1DecodeError),
            _number(data["width"], f"{path}.width", ProjectV1DecodeError),
            _number(data["depth"], f"{path}.depth", ProjectV1DecodeError),
            _number(data["height"], f"{path}.height", ProjectV1DecodeError),
        )
    if kind == "CylinderGeometry":
        _keys(
            data,
            path,
            required={"type", "name", "radius", "height"},
            optional=set(),
            error_type=ProjectV1DecodeError,
        )
        return _construct_decode(
            CylinderGeometry,
            path,
            _string(data["name"], f"{path}.name", error_type=ProjectV1DecodeError),
            _number(data["radius"], f"{path}.radius", ProjectV1DecodeError),
            _number(data["height"], f"{path}.height", ProjectV1DecodeError),
        )
    if kind == "PlateWithHoleGeometry":
        _keys(
            data,
            path,
            required={
                "type",
                "name",
                "width",
                "height",
                "hole_x",
                "hole_y",
                "hole_radius",
            },
            optional=set(),
            error_type=ProjectV1DecodeError,
        )
        return _construct_decode(
            PlateWithHoleGeometry,
            path,
            _string(data["name"], f"{path}.name", error_type=ProjectV1DecodeError),
            _number(data["width"], f"{path}.width", ProjectV1DecodeError),
            _number(data["height"], f"{path}.height", ProjectV1DecodeError),
            _number(data["hole_x"], f"{path}.hole_x", ProjectV1DecodeError),
            _number(data["hole_y"], f"{path}.hole_y", ProjectV1DecodeError),
            _number(
                data["hole_radius"], f"{path}.hole_radius", ProjectV1DecodeError
            ),
        )
    if kind == "MovedGeometry":
        _keys(
            data,
            path,
            required={"type", "base", "dx", "dy"},
            optional={"dz"},
            error_type=ProjectV1DecodeError,
        )
        return _construct_decode(
            MovedGeometry,
            path,
            _decode_geometry(data["base"], f"{path}.base"),
            _number(data["dx"], f"{path}.dx", ProjectV1DecodeError),
            _number(data["dy"], f"{path}.dy", ProjectV1DecodeError),
            _number(data.get("dz", 0.0), f"{path}.dz", ProjectV1DecodeError),
        )
    if kind == "RotatedGeometry":
        _keys(
            data,
            path,
            required={"type", "base", "axis", "angle_degrees"},
            optional=set(),
            error_type=ProjectV1DecodeError,
        )
        return _construct_decode(
            RotatedGeometry,
            path,
            _decode_geometry(data["base"], f"{path}.base"),
            _string(data["axis"], f"{path}.axis", error_type=ProjectV1DecodeError),
            _number(
                data["angle_degrees"],
                f"{path}.angle_degrees",
                ProjectV1DecodeError,
            ),
        )
    if kind == "ExtrudedGeometry":
        _keys(
            data,
            path,
            required={"type", "base", "height"},
            optional=set(),
            error_type=ProjectV1DecodeError,
        )
        return _construct_decode(
            ExtrudedGeometry,
            path,
            _decode_geometry(data["base"], f"{path}.base"),
            _number(data["height"], f"{path}.height", ProjectV1DecodeError),
        )
    if kind == "BooleanGeometry":
        _keys(
            data,
            path,
            required={"type", "name", "operation", "object", "tool"},
            optional=set(),
            error_type=ProjectV1DecodeError,
        )
        return _construct_decode(
            BooleanGeometry,
            path,
            _string(data["name"], f"{path}.name", error_type=ProjectV1DecodeError),
            _string(
                data["operation"],
                f"{path}.operation",
                error_type=ProjectV1DecodeError,
            ),
            _decode_geometry(data["object"], f"{path}.object"),
            _decode_geometry(data["tool"], f"{path}.tool"),
        )
    raise ProjectV1DecodeError(f"{path}.type 是未知几何类型：{kind!r}")


def _decode_contour(value: Any, path: str) -> SketchRectangle | SketchCircle:
    data = _mapping(value, path, error_type=ProjectV1DecodeError)
    kind = _string(
        data.get("type"), f"{path}.type", error_type=ProjectV1DecodeError
    )
    operation = _string(
        data.get("operation"),
        f"{path}.operation",
        error_type=ProjectV1DecodeError,
    )
    if kind == "rectangle":
        _keys(
            data,
            path,
            required={"type", "operation", "x", "y", "width", "height"},
            optional=set(),
            error_type=ProjectV1DecodeError,
        )
        return _construct_decode(
            SketchRectangle,
            path,
            operation,
            _number(data["x"], f"{path}.x", ProjectV1DecodeError),
            _number(data["y"], f"{path}.y", ProjectV1DecodeError),
            _number(data["width"], f"{path}.width", ProjectV1DecodeError),
            _number(data["height"], f"{path}.height", ProjectV1DecodeError),
        )
    if kind == "circle":
        _keys(
            data,
            path,
            required={"type", "operation", "x", "y", "radius"},
            optional=set(),
            error_type=ProjectV1DecodeError,
        )
        return _construct_decode(
            SketchCircle,
            path,
            operation,
            _number(data["x"], f"{path}.x", ProjectV1DecodeError),
            _number(data["y"], f"{path}.y", ProjectV1DecodeError),
            _number(data["radius"], f"{path}.radius", ProjectV1DecodeError),
        )
    raise ProjectV1DecodeError(f"{path}.type 是未知草图轮廓：{kind!r}")


def _decode_mesh_settings(value: Any, path: str) -> MeshSettings | None:
    if value is None:
        return None
    data = _mapping(value, path, error_type=ProjectV1DecodeError)
    _keys(
        data,
        path,
        required={"size"},
        optional={"order", "cell_shape", "local_size", "local_controls"},
        error_type=ProjectV1DecodeError,
    )
    local_size_value = data.get("local_size")
    local_size = (
        None
        if local_size_value is None
        else _number(local_size_value, f"{path}.local_size", ProjectV1DecodeError)
    )
    controls = tuple(
        _decode_local_control(item, f"{path}.local_controls[{index}]")
        for index, item in enumerate(
            _array(
                data.get("local_controls", ()),
                f"{path}.local_controls",
                error_type=ProjectV1DecodeError,
            )
        )
    )
    return _construct_decode(
        MeshSettings,
        path,
        _number(data["size"], f"{path}.size", ProjectV1DecodeError),
        _integer(
            data.get("order", 1), f"{path}.order", error_type=ProjectV1DecodeError
        ),
        _string(
            data.get("cell_shape", "triangle"),
            f"{path}.cell_shape",
            error_type=ProjectV1DecodeError,
        ),
        local_size,
        controls,
    )


def _decode_local_control(value: Any, path: str) -> LocalMeshControl:
    data = _mapping(value, path, error_type=ProjectV1DecodeError)
    _keys(
        data,
        path,
        required={"entity_kind", "entity_id", "size"},
        optional=set(),
        error_type=ProjectV1DecodeError,
    )
    return _construct_decode(
        LocalMeshControl,
        path,
        _string(
            data["entity_kind"],
            f"{path}.entity_kind",
            error_type=ProjectV1DecodeError,
        ),
        _positive_integer(data["entity_id"], f"{path}.entity_id", ProjectV1DecodeError),
        _number(data["size"], f"{path}.size", ProjectV1DecodeError),
    )


def _decode_step(value: Any, path: str) -> AnalysisStep:
    data = _mapping(value, path, error_type=ProjectV1DecodeError)
    _keys(
        data,
        path,
        required={"name"},
        optional={
            "procedure",
            "metadata",
            "boundaries",
            "cloads",
            "edge_loads",
            "surface_loads",
            "line_loads",
            "gravity_loads",
            "outputs",
        },
        error_type=ProjectV1DecodeError,
    )
    return _construct_decode(
        AnalysisStep,
        path,
        name=_string(data["name"], f"{path}.name", error_type=ProjectV1DecodeError),
        procedure=_string(
            data.get("procedure", "static"),
            f"{path}.procedure",
            error_type=ProjectV1DecodeError,
        ),
        boundaries=tuple(
            _decode_boundary(item, f"{path}.boundaries[{index}]")
            for index, item in enumerate(
                _array(
                    data.get("boundaries", ()),
                    f"{path}.boundaries",
                    error_type=ProjectV1DecodeError,
                )
            )
        ),
        cloads=tuple(
            _decode_cload(item, f"{path}.cloads[{index}]")
            for index, item in enumerate(
                _array(
                    data.get("cloads", ()),
                    f"{path}.cloads",
                    error_type=ProjectV1DecodeError,
                )
            )
        ),
        edge_loads=tuple(
            _decode_edge_load(item, f"{path}.edge_loads[{index}]")
            for index, item in enumerate(
                _array(
                    data.get("edge_loads", ()),
                    f"{path}.edge_loads",
                    error_type=ProjectV1DecodeError,
                )
            )
        ),
        surface_loads=tuple(
            _decode_surface_load(item, f"{path}.surface_loads[{index}]")
            for index, item in enumerate(
                _array(
                    data.get("surface_loads", ()),
                    f"{path}.surface_loads",
                    error_type=ProjectV1DecodeError,
                )
            )
        ),
        line_loads=tuple(
            _decode_line_load(item, f"{path}.line_loads[{index}]")
            for index, item in enumerate(
                _array(
                    data.get("line_loads", ()),
                    f"{path}.line_loads",
                    error_type=ProjectV1DecodeError,
                )
            )
        ),
        gravity_loads=tuple(
            _decode_gravity_load(item, f"{path}.gravity_loads[{index}]")
            for index, item in enumerate(
                _array(
                    data.get("gravity_loads", ()),
                    f"{path}.gravity_loads",
                    error_type=ProjectV1DecodeError,
                )
            )
        ),
        outputs=tuple(
            _decode_output(item, f"{path}.outputs[{index}]")
            for index, item in enumerate(
                _array(
                    data.get("outputs", ()),
                    f"{path}.outputs",
                    error_type=ProjectV1DecodeError,
                )
            )
        ),
        metadata=_json_object(
            data.get("metadata", {}),
            f"{path}.metadata",
            error_type=ProjectV1DecodeError,
        ),
    )


def _decode_boundary(value: Any, path: str) -> DisplacementConstraint:
    data = _mapping(value, path, error_type=ProjectV1DecodeError)
    _keys(
        data,
        path,
        required={"target", "first_component", "last_component"},
        optional={"value"},
        error_type=ProjectV1DecodeError,
    )
    return _construct_decode(
        DisplacementConstraint,
        path,
        _target(data["target"], f"{path}.target", ProjectV1DecodeError),
        _integer(
            data["first_component"],
            f"{path}.first_component",
            error_type=ProjectV1DecodeError,
        ),
        _integer(
            data["last_component"],
            f"{path}.last_component",
            error_type=ProjectV1DecodeError,
        ),
        _number(data.get("value", 0.0), f"{path}.value", ProjectV1DecodeError),
    )


def _decode_cload(value: Any, path: str) -> NodalLoad:
    data = _mapping(value, path, error_type=ProjectV1DecodeError)
    _keys(
        data,
        path,
        required={"target", "component", "value"},
        optional=set(),
        error_type=ProjectV1DecodeError,
    )
    return _construct_decode(
        NodalLoad,
        path,
        _target(data["target"], f"{path}.target", ProjectV1DecodeError),
        _integer(
            data["component"], f"{path}.component", error_type=ProjectV1DecodeError
        ),
        _number(data["value"], f"{path}.value", ProjectV1DecodeError),
    )


def _decode_edge_load(value: Any, path: str) -> EdgeLoad:
    data = _mapping(value, path, error_type=ProjectV1DecodeError)
    _keys(
        data,
        path,
        required={"edge"},
        optional={"vector", "magnitude", "load_type"},
        error_type=ProjectV1DecodeError,
    )
    magnitude_value = data.get("magnitude")
    return _construct_decode(
        EdgeLoad,
        path,
        _string(data["edge"], f"{path}.edge", error_type=ProjectV1DecodeError),
        _number_array(data.get("vector", ()), f"{path}.vector", ProjectV1DecodeError),
        None
        if magnitude_value is None
        else _number(magnitude_value, f"{path}.magnitude", ProjectV1DecodeError),
        _string(
            data.get("load_type", "traction"),
            f"{path}.load_type",
            error_type=ProjectV1DecodeError,
        ),
    )


def _decode_surface_load(value: Any, path: str) -> SurfaceLoad:
    data = _mapping(value, path, error_type=ProjectV1DecodeError)
    _keys(
        data,
        path,
        required={"surface"},
        optional={"vector", "magnitude", "load_type"},
        error_type=ProjectV1DecodeError,
    )
    magnitude_value = data.get("magnitude")
    return _construct_decode(
        SurfaceLoad,
        path,
        _string(
            data["surface"], f"{path}.surface", error_type=ProjectV1DecodeError
        ),
        _number_array(data.get("vector", ()), f"{path}.vector", ProjectV1DecodeError),
        None
        if magnitude_value is None
        else _number(magnitude_value, f"{path}.magnitude", ProjectV1DecodeError),
        _string(
            data.get("load_type", "traction"),
            f"{path}.load_type",
            error_type=ProjectV1DecodeError,
        ),
    )


def _decode_line_load(value: Any, path: str) -> LineLoad:
    data = _mapping(value, path, error_type=ProjectV1DecodeError)
    _keys(
        data,
        path,
        required={"target", "vector"},
        optional={"coordinate_system"},
        error_type=ProjectV1DecodeError,
    )
    vector = _number_array(data["vector"], f"{path}.vector", ProjectV1DecodeError)
    coordinate_system = _string(
        data.get("coordinate_system", "global"),
        f"{path}.coordinate_system",
        error_type=ProjectV1DecodeError,
    )
    if coordinate_system not in {"global", "local"}:
        raise ProjectV1DecodeError(
            f"{path}.coordinate_system 必须是 'global' 或 'local'"
        )
    return _construct_decode(
        LineLoad,
        path,
        _target(data["target"], f"{path}.target", ProjectV1DecodeError),
        vector,
        coordinate_system,
    )


def _decode_gravity_load(value: Any, path: str) -> GravityLoad:
    data = _mapping(value, path, error_type=ProjectV1DecodeError)
    _keys(
        data,
        path,
        required={"acceleration"},
        optional={"target"},
        error_type=ProjectV1DecodeError,
    )
    target = data.get("target")
    return _construct_decode(
        GravityLoad,
        path,
        _number_array(
            data["acceleration"], f"{path}.acceleration", ProjectV1DecodeError
        ),
        None
        if target is None
        else _target(target, f"{path}.target", ProjectV1DecodeError),
    )


def _decode_output(value: Any, path: str) -> OutputRequest:
    data = _mapping(value, path, error_type=ProjectV1DecodeError)
    _keys(
        data,
        path,
        required={"kind", "target"},
        optional={"variables", "metadata"},
        error_type=ProjectV1DecodeError,
    )
    variables = tuple(
        _string(item, f"{path}.variables[{index}]", error_type=ProjectV1DecodeError)
        for index, item in enumerate(
            _array(
                data.get("variables", ()),
                f"{path}.variables",
                error_type=ProjectV1DecodeError,
            )
        )
    )
    return _construct_decode(
        OutputRequest,
        path,
        _string(data["kind"], f"{path}.kind", error_type=ProjectV1DecodeError),
        _string(data["target"], f"{path}.target", error_type=ProjectV1DecodeError),
        variables,
        _json_object(
            data.get("metadata", {}),
            f"{path}.metadata",
            error_type=ProjectV1DecodeError,
        ),
    )


def _encode_part(part: Any, path: str) -> dict[str, Any]:
    _exact_dataclass(part, NativePart, {"name", "body_name"}, path)
    return {
        "name": _string(part.name, f"{path}.name", error_type=ProjectV1EncodeError),
        "body_name": _string(
            part.body_name, f"{path}.body_name", error_type=ProjectV1EncodeError
        ),
    }


def _encode_feature(feature: Any, path: str) -> dict[str, Any]:
    _exact_dataclass(feature, FeatureRecord, {"name", "kind", "payload"}, path)
    return {
        "name": _string(
            feature.name, f"{path}.name", error_type=ProjectV1EncodeError
        ),
        "kind": _string(
            feature.kind, f"{path}.kind", error_type=ProjectV1EncodeError
        ),
        "payload": _json_object(
            feature.payload, f"{path}.payload", error_type=ProjectV1EncodeError
        ),
    }


def _encode_named_region(region: Any, path: str) -> dict[str, Any]:
    _exact_dataclass(
        region,
        NamedRegion,
        {"name", "entity_kind", "entity_ids"},
        path,
    )
    entity_kind = _string(
        region.entity_kind, f"{path}.entity_kind", error_type=ProjectV1EncodeError
    )
    if entity_kind not in {"point", "edge", "face", "body"}:
        raise ProjectV1EncodeError(f"{path}.entity_kind 不是受支持的实体类型")
    entity_ids = [
        _positive_integer(item, f"{path}.entity_ids[{index}]", ProjectV1EncodeError)
        for index, item in enumerate(
            _runtime_sequence(region.entity_ids, f"{path}.entity_ids")
        )
    ]
    if len(set(entity_ids)) != len(entity_ids):
        raise ProjectV1EncodeError(f"{path}.entity_ids 包含重复实体编号")
    return {
        "name": _string(
            region.name, f"{path}.name", error_type=ProjectV1EncodeError
        ),
        "entity_kind": entity_kind,
        "entity_ids": entity_ids,
    }


def _encode_material(material: Any, path: str) -> dict[str, Any]:
    _exact_dataclass(material, MaterialDefinition, {"name", "properties"}, path)
    return {
        "name": _string(
            material.name, f"{path}.name", error_type=ProjectV1EncodeError
        ),
        "properties": _json_object(
            material.properties,
            f"{path}.properties",
            error_type=ProjectV1EncodeError,
        ),
    }


def _encode_section(section: Any, path: str) -> dict[str, Any]:
    _exact_dataclass(
        section,
        SectionDefinition,
        {"name", "material", "section_type", "properties"},
        path,
    )
    return {
        "name": _string(
            section.name, f"{path}.name", error_type=ProjectV1EncodeError
        ),
        "material": _string(
            section.material, f"{path}.material", error_type=ProjectV1EncodeError
        ),
        "section_type": _string(
            section.section_type,
            f"{path}.section_type",
            error_type=ProjectV1EncodeError,
        ),
        "properties": _json_object(
            section.properties,
            f"{path}.properties",
            error_type=ProjectV1EncodeError,
        ),
    }


def _encode_assignment(assignment: Any, path: str) -> dict[str, Any]:
    _exact_dataclass(
        assignment,
        RegionAssignment,
        {"section_name", "region_name", "beam_orientation"},
        path,
    )
    if assignment.beam_orientation is not None:
        raise ProjectV1EncodeError(
            f"{path}.beam_orientation 无法由 .femproj v1 无损表示；"
            "v1 不支持 Beam orientation"
        )
    return {
        "section_name": _string(
            assignment.section_name,
            f"{path}.section_name",
            error_type=ProjectV1EncodeError,
        ),
        "region_name": _string(
            assignment.region_name,
            f"{path}.region_name",
            error_type=ProjectV1EncodeError,
        ),
    }


def _encode_geometry(
    recipe: Any,
    path: str,
    ancestors: set[int],
) -> dict[str, Any]:
    identity = id(recipe)
    if identity in ancestors:
        raise ProjectV1EncodeError(f"{path} 包含循环几何引用")
    ancestors.add(identity)
    try:
        recipe_type = type(recipe)
        if recipe_type is SketchGeometry:
            _exact_dataclass(recipe, SketchGeometry, {"name", "contours"}, path)
            return {
                "type": "SketchGeometry",
                "name": _string(
                    recipe.name, f"{path}.name", error_type=ProjectV1EncodeError
                ),
                "contours": [
                    _encode_contour(item, f"{path}.contours[{index}]")
                    for index, item in enumerate(
                        _runtime_sequence(recipe.contours, f"{path}.contours")
                    )
                ],
            }
        if recipe_type is RectangleGeometry:
            _exact_dataclass(
                recipe, RectangleGeometry, {"name", "width", "height"}, path
            )
            return {
                "type": "RectangleGeometry",
                "name": _string(
                    recipe.name, f"{path}.name", error_type=ProjectV1EncodeError
                ),
                "width": _number(
                    recipe.width, f"{path}.width", ProjectV1EncodeError
                ),
                "height": _number(
                    recipe.height, f"{path}.height", ProjectV1EncodeError
                ),
            }
        if recipe_type is DiskGeometry:
            _exact_dataclass(recipe, DiskGeometry, {"name", "radius"}, path)
            return {
                "type": "DiskGeometry",
                "name": _string(
                    recipe.name, f"{path}.name", error_type=ProjectV1EncodeError
                ),
                "radius": _number(
                    recipe.radius, f"{path}.radius", ProjectV1EncodeError
                ),
            }
        if recipe_type is BoxGeometry:
            _exact_dataclass(
                recipe,
                BoxGeometry,
                {"name", "width", "depth", "height"},
                path,
            )
            return {
                "type": "BoxGeometry",
                "name": _string(
                    recipe.name, f"{path}.name", error_type=ProjectV1EncodeError
                ),
                "width": _number(
                    recipe.width, f"{path}.width", ProjectV1EncodeError
                ),
                "depth": _number(
                    recipe.depth, f"{path}.depth", ProjectV1EncodeError
                ),
                "height": _number(
                    recipe.height, f"{path}.height", ProjectV1EncodeError
                ),
            }
        if recipe_type is CylinderGeometry:
            _exact_dataclass(
                recipe, CylinderGeometry, {"name", "radius", "height"}, path
            )
            return {
                "type": "CylinderGeometry",
                "name": _string(
                    recipe.name, f"{path}.name", error_type=ProjectV1EncodeError
                ),
                "radius": _number(
                    recipe.radius, f"{path}.radius", ProjectV1EncodeError
                ),
                "height": _number(
                    recipe.height, f"{path}.height", ProjectV1EncodeError
                ),
            }
        if recipe_type is PlateWithHoleGeometry:
            _exact_dataclass(
                recipe,
                PlateWithHoleGeometry,
                {"name", "width", "height", "hole_x", "hole_y", "hole_radius"},
                path,
            )
            return {
                "type": "PlateWithHoleGeometry",
                "name": _string(
                    recipe.name, f"{path}.name", error_type=ProjectV1EncodeError
                ),
                "width": _number(
                    recipe.width, f"{path}.width", ProjectV1EncodeError
                ),
                "height": _number(
                    recipe.height, f"{path}.height", ProjectV1EncodeError
                ),
                "hole_x": _number(
                    recipe.hole_x, f"{path}.hole_x", ProjectV1EncodeError
                ),
                "hole_y": _number(
                    recipe.hole_y, f"{path}.hole_y", ProjectV1EncodeError
                ),
                "hole_radius": _number(
                    recipe.hole_radius,
                    f"{path}.hole_radius",
                    ProjectV1EncodeError,
                ),
            }
        if recipe_type is MovedGeometry:
            _exact_dataclass(recipe, MovedGeometry, {"base", "dx", "dy", "dz"}, path)
            return {
                "type": "MovedGeometry",
                "base": _encode_geometry(recipe.base, f"{path}.base", ancestors),
                "dx": _number(recipe.dx, f"{path}.dx", ProjectV1EncodeError),
                "dy": _number(recipe.dy, f"{path}.dy", ProjectV1EncodeError),
                "dz": _number(recipe.dz, f"{path}.dz", ProjectV1EncodeError),
            }
        if recipe_type is RotatedGeometry:
            _exact_dataclass(
                recipe, RotatedGeometry, {"base", "axis", "angle_degrees"}, path
            )
            return {
                "type": "RotatedGeometry",
                "base": _encode_geometry(recipe.base, f"{path}.base", ancestors),
                "axis": _string(
                    recipe.axis, f"{path}.axis", error_type=ProjectV1EncodeError
                ),
                "angle_degrees": _number(
                    recipe.angle_degrees,
                    f"{path}.angle_degrees",
                    ProjectV1EncodeError,
                ),
            }
        if recipe_type is ExtrudedGeometry:
            _exact_dataclass(recipe, ExtrudedGeometry, {"base", "height"}, path)
            return {
                "type": "ExtrudedGeometry",
                "base": _encode_geometry(recipe.base, f"{path}.base", ancestors),
                "height": _number(
                    recipe.height, f"{path}.height", ProjectV1EncodeError
                ),
            }
        if recipe_type is BooleanGeometry:
            _exact_dataclass(
                recipe,
                BooleanGeometry,
                {"name", "operation", "object_geometry", "tool_geometry"},
                path,
            )
            return {
                "type": "BooleanGeometry",
                "name": _string(
                    recipe.name, f"{path}.name", error_type=ProjectV1EncodeError
                ),
                "operation": _string(
                    recipe.operation,
                    f"{path}.operation",
                    error_type=ProjectV1EncodeError,
                ),
                "object": _encode_geometry(
                    recipe.object_geometry, f"{path}.object_geometry", ancestors
                ),
                "tool": _encode_geometry(
                    recipe.tool_geometry, f"{path}.tool_geometry", ancestors
                ),
            }
        raise ProjectV1EncodeError(
            f"{path} 的几何类型无法由 v1 无损编码：{recipe_type.__name__}"
        )
    finally:
        ancestors.remove(identity)


def _encode_contour(contour: Any, path: str) -> dict[str, Any]:
    if type(contour) is SketchRectangle:
        _exact_dataclass(
            contour,
            SketchRectangle,
            {"operation", "x", "y", "width", "height"},
            path,
        )
        return {
            "type": "rectangle",
            "operation": _string(
                contour.operation,
                f"{path}.operation",
                error_type=ProjectV1EncodeError,
            ),
            "x": _number(contour.x, f"{path}.x", ProjectV1EncodeError),
            "y": _number(contour.y, f"{path}.y", ProjectV1EncodeError),
            "width": _number(
                contour.width, f"{path}.width", ProjectV1EncodeError
            ),
            "height": _number(
                contour.height, f"{path}.height", ProjectV1EncodeError
            ),
        }
    if type(contour) is SketchCircle:
        _exact_dataclass(
            contour, SketchCircle, {"operation", "x", "y", "radius"}, path
        )
        return {
            "type": "circle",
            "operation": _string(
                contour.operation,
                f"{path}.operation",
                error_type=ProjectV1EncodeError,
            ),
            "x": _number(contour.x, f"{path}.x", ProjectV1EncodeError),
            "y": _number(contour.y, f"{path}.y", ProjectV1EncodeError),
            "radius": _number(
                contour.radius, f"{path}.radius", ProjectV1EncodeError
            ),
        }
    raise ProjectV1EncodeError(
        f"{path} 的草图轮廓类型无法由 v1 无损编码：{type(contour).__name__}"
    )


def _encode_mesh_settings(settings: Any, path: str) -> dict[str, Any] | None:
    if settings is None:
        return None
    _exact_dataclass(
        settings,
        MeshSettings,
        {"size", "order", "cell_shape", "local_size", "local_controls"},
        path,
    )
    local_size = settings.local_size
    return {
        "size": _number(settings.size, f"{path}.size", ProjectV1EncodeError),
        "order": _integer(
            settings.order, f"{path}.order", error_type=ProjectV1EncodeError
        ),
        "cell_shape": _string(
            settings.cell_shape,
            f"{path}.cell_shape",
            error_type=ProjectV1EncodeError,
        ),
        "local_size": None
        if local_size is None
        else _number(local_size, f"{path}.local_size", ProjectV1EncodeError),
        "local_controls": [
            _encode_local_control(item, f"{path}.local_controls[{index}]")
            for index, item in enumerate(
                _runtime_sequence(settings.local_controls, f"{path}.local_controls")
            )
        ],
    }


def _encode_local_control(control: Any, path: str) -> dict[str, Any]:
    _exact_dataclass(
        control, LocalMeshControl, {"entity_kind", "entity_id", "size"}, path
    )
    return {
        "entity_kind": _string(
            control.entity_kind,
            f"{path}.entity_kind",
            error_type=ProjectV1EncodeError,
        ),
        "entity_id": _positive_integer(
            control.entity_id, f"{path}.entity_id", ProjectV1EncodeError
        ),
        "size": _number(control.size, f"{path}.size", ProjectV1EncodeError),
    }


def _encode_step(step: Any, path: str) -> dict[str, Any]:
    _exact_dataclass(
        step,
        AnalysisStep,
        {
            "name",
            "procedure",
            "boundaries",
            "cloads",
            "surface_loads",
            "outputs",
            "metadata",
            "edge_loads",
            "line_loads",
            "gravity_loads",
        },
        path,
    )
    return {
        "name": _string(step.name, f"{path}.name", error_type=ProjectV1EncodeError),
        "procedure": _string(
            step.procedure, f"{path}.procedure", error_type=ProjectV1EncodeError
        ),
        "metadata": _json_object(
            step.metadata, f"{path}.metadata", error_type=ProjectV1EncodeError
        ),
        "boundaries": [
            _encode_boundary(item, f"{path}.boundaries[{index}]")
            for index, item in enumerate(
                _runtime_sequence(step.boundaries, f"{path}.boundaries")
            )
        ],
        "cloads": [
            _encode_cload(item, f"{path}.cloads[{index}]")
            for index, item in enumerate(
                _runtime_sequence(step.cloads, f"{path}.cloads")
            )
        ],
        "edge_loads": [
            _encode_edge_load(item, f"{path}.edge_loads[{index}]")
            for index, item in enumerate(
                _runtime_sequence(step.edge_loads, f"{path}.edge_loads")
            )
        ],
        "surface_loads": [
            _encode_surface_load(item, f"{path}.surface_loads[{index}]")
            for index, item in enumerate(
                _runtime_sequence(step.surface_loads, f"{path}.surface_loads")
            )
        ],
        "line_loads": [
            _encode_line_load(item, f"{path}.line_loads[{index}]")
            for index, item in enumerate(
                _runtime_sequence(step.line_loads, f"{path}.line_loads")
            )
        ],
        "gravity_loads": [
            _encode_gravity_load(item, f"{path}.gravity_loads[{index}]")
            for index, item in enumerate(
                _runtime_sequence(step.gravity_loads, f"{path}.gravity_loads")
            )
        ],
        "outputs": [
            _encode_output(item, f"{path}.outputs[{index}]")
            for index, item in enumerate(
                _runtime_sequence(step.outputs, f"{path}.outputs")
            )
        ],
    }


def _encode_boundary(boundary: Any, path: str) -> dict[str, Any]:
    _exact_dataclass(
        boundary,
        DisplacementConstraint,
        {"target", "first_component", "last_component", "value"},
        path,
    )
    return {
        "target": _target(boundary.target, f"{path}.target", ProjectV1EncodeError),
        "first_component": _integer(
            boundary.first_component,
            f"{path}.first_component",
            error_type=ProjectV1EncodeError,
        ),
        "last_component": _integer(
            boundary.last_component,
            f"{path}.last_component",
            error_type=ProjectV1EncodeError,
        ),
        "value": _number(boundary.value, f"{path}.value", ProjectV1EncodeError),
    }


def _encode_cload(load: Any, path: str) -> dict[str, Any]:
    _exact_dataclass(load, NodalLoad, {"target", "component", "value"}, path)
    return {
        "target": _target(load.target, f"{path}.target", ProjectV1EncodeError),
        "component": _integer(
            load.component, f"{path}.component", error_type=ProjectV1EncodeError
        ),
        "value": _number(load.value, f"{path}.value", ProjectV1EncodeError),
    }


def _encode_edge_load(load: Any, path: str) -> dict[str, Any]:
    _exact_dataclass(
        load, EdgeLoad, {"edge", "vector", "magnitude", "load_type"}, path
    )
    return {
        "edge": _string(load.edge, f"{path}.edge", error_type=ProjectV1EncodeError),
        "vector": list(
            _number_array(load.vector, f"{path}.vector", ProjectV1EncodeError)
        ),
        "magnitude": None
        if load.magnitude is None
        else _number(load.magnitude, f"{path}.magnitude", ProjectV1EncodeError),
        "load_type": _string(
            load.load_type, f"{path}.load_type", error_type=ProjectV1EncodeError
        ),
    }


def _encode_surface_load(load: Any, path: str) -> dict[str, Any]:
    _exact_dataclass(
        load, SurfaceLoad, {"surface", "vector", "magnitude", "load_type"}, path
    )
    return {
        "surface": _string(
            load.surface, f"{path}.surface", error_type=ProjectV1EncodeError
        ),
        "vector": list(
            _number_array(load.vector, f"{path}.vector", ProjectV1EncodeError)
        ),
        "magnitude": None
        if load.magnitude is None
        else _number(load.magnitude, f"{path}.magnitude", ProjectV1EncodeError),
        "load_type": _string(
            load.load_type, f"{path}.load_type", error_type=ProjectV1EncodeError
        ),
    }


def _encode_line_load(load: Any, path: str) -> dict[str, Any]:
    _exact_dataclass(
        load, LineLoad, {"target", "vector", "coordinate_system"}, path
    )
    vector = _number_array(load.vector, f"{path}.vector", ProjectV1EncodeError)
    coordinate_system = _string(
        load.coordinate_system,
        f"{path}.coordinate_system",
        error_type=ProjectV1EncodeError,
    )
    if coordinate_system not in {"global", "local"}:
        raise ProjectV1EncodeError(
            f"{path}.coordinate_system 必须是 'global' 或 'local'"
        )
    return {
        "target": _target(load.target, f"{path}.target", ProjectV1EncodeError),
        "vector": list(vector),
        "coordinate_system": coordinate_system,
    }


def _encode_gravity_load(load: Any, path: str) -> dict[str, Any]:
    _exact_dataclass(load, GravityLoad, {"acceleration", "target"}, path)
    return {
        "acceleration": list(
            _number_array(
                load.acceleration, f"{path}.acceleration", ProjectV1EncodeError
            )
        ),
        "target": None
        if load.target is None
        else _target(load.target, f"{path}.target", ProjectV1EncodeError),
    }


def _encode_output(output: Any, path: str) -> dict[str, Any]:
    _exact_dataclass(
        output, OutputRequest, {"kind", "target", "variables", "metadata"}, path
    )
    return {
        "kind": _string(
            output.kind, f"{path}.kind", error_type=ProjectV1EncodeError
        ),
        "target": _string(
            output.target, f"{path}.target", error_type=ProjectV1EncodeError
        ),
        "variables": [
            _string(
                item,
                f"{path}.variables[{index}]",
                error_type=ProjectV1EncodeError,
            )
            for index, item in enumerate(
                _runtime_sequence(output.variables, f"{path}.variables")
            )
        ],
        "metadata": _json_object(
            output.metadata, f"{path}.metadata", error_type=ProjectV1EncodeError
        ),
    }


def _history_for_recipe(recipe: Any) -> list[FeatureRecord]:
    """Reconstruct the legacy shallow history when an old v1 key is absent."""

    if type(recipe) is SketchGeometry:
        return [FeatureRecord("Sketch-1", "sketch")]
    if type(recipe) is ExtrudedGeometry:
        return _history_for_recipe(recipe.base) + [
            FeatureRecord("Extrude-1", "extrude")
        ]
    if type(recipe) is MovedGeometry:
        return _history_for_recipe(recipe.base) + [FeatureRecord("Move-1", "move")]
    if type(recipe) is RotatedGeometry:
        return _history_for_recipe(recipe.base) + [
            FeatureRecord("Rotate-1", "rotate")
        ]
    if type(recipe) is BooleanGeometry:
        label = {
            "fuse": "Fuse-1",
            "cut": "Cut-1",
            "fragment": "Partition-1",
        }[recipe.operation]
        return _history_for_recipe(recipe.object_geometry) + [
            FeatureRecord(label, recipe.operation)
        ]
    return [FeatureRecord("Base-1", "base")]


def _unwrap_project_snapshot(snapshot: Any) -> ProjectSnapshot:
    try:
        from fem.application.session import ProjectSaveSnapshot, ProjectSnapshot
    except ImportError as exc:
        raise ProjectV1EncodeError("fem.application.session 尚未提供项目 snapshot") from exc

    if type(snapshot) is ProjectSaveSnapshot:
        project = snapshot.snapshot
    elif type(snapshot) is ProjectSnapshot:
        project = snapshot
    else:
        raise ProjectV1EncodeError(
            "save/encode 需要 ProjectSnapshot 或 ProjectSaveSnapshot"
        )
    if type(project) is not ProjectSnapshot:
        raise ProjectV1EncodeError("保存 snapshot 未包含有效的 ProjectSnapshot")
    if getattr(project, "model", None) is not None:
        raise ProjectV1EncodeError("v1 不持久化模型制品，当前 snapshot 无法无损保存")
    return project


def _snapshot_attr(snapshot: Any, name: str) -> Any:
    try:
        return getattr(snapshot, name)
    except AttributeError as exc:
        raise ProjectV1EncodeError(f"ProjectSnapshot 缺少字段 {name!r}") from exc


def _snapshot_sequence(
    snapshot: Any,
    name: str,
    *,
    mapping_values: bool = False,
) -> tuple[Any, ...]:
    value = _snapshot_attr(snapshot, name)
    if mapping_values and isinstance(value, Mapping):
        sequence = tuple(value.values())
    else:
        sequence = _runtime_sequence(value, f"snapshot.{name}")
    return tuple(sequence)


def _exact_dataclass(
    value: Any,
    expected_type: type[Any],
    expected_fields: set[str],
    path: str,
) -> None:
    if type(value) is not expected_type:
        raise ProjectV1EncodeError(
            f"{path} 必须是 {expected_type.__name__}，收到 {type(value).__name__}"
        )
    actual_fields = {item.name for item in fields(expected_type)}
    if actual_fields != expected_fields:
        unsupported = sorted(actual_fields ^ expected_fields)
        raise ProjectV1EncodeError(
            f"{expected_type.__name__} 与 v1 字段契约不一致，拒绝静默丢失："
            f"{unsupported}"
        )


def _construct_decode(
    constructor: type[Any],
    path: str,
    *args: Any,
    **kwargs: Any,
) -> Any:
    try:
        return constructor(*args, **kwargs)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ProjectV1DecodeError(f"{path} 无效：{exc}") from exc


def _require_unique_names(
    values: Sequence[Any],
    path: str,
    *,
    encode: bool = False,
) -> None:
    error_type = ProjectV1EncodeError if encode else ProjectV1DecodeError
    seen: set[str] = set()
    for index, item in enumerate(values):
        name = _string(
            getattr(item, "name", None),
            f"{path}[{index}].name",
            error_type=error_type,
        )
        if name in seen:
            raise error_type(f"{path} 包含重复名称：{name!r}")
        seen.add(name)


def _keys(
    data: Mapping[str, Any],
    path: str,
    *,
    required: set[str],
    optional: set[str],
    error_type: type[ProjectV1Error],
) -> None:
    actual = set(data)
    missing = sorted(required - actual)
    if missing:
        raise error_type(f"{path} 缺少必需字段：{', '.join(missing)}")
    unknown = sorted(actual - required - optional)
    if unknown:
        raise error_type(f"{path} 包含 v1 未知字段：{', '.join(unknown)}")


def _mapping(
    value: Any,
    path: str,
    *,
    error_type: type[ProjectV1Error],
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise error_type(f"{path} 必须是 JSON object")
    if any(not isinstance(key, str) for key in value):
        raise error_type(f"{path} 的所有键必须是字符串")
    return value


def _array(
    value: Any,
    path: str,
    *,
    error_type: type[ProjectV1Error],
) -> tuple[Any, ...]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise error_type(f"{path} 必须是 JSON array")
    return tuple(value)


def _runtime_sequence(value: Any, path: str) -> tuple[Any, ...]:
    if isinstance(value, (str, bytes, bytearray, Mapping)) or not isinstance(
        value, Sequence
    ):
        raise ProjectV1EncodeError(f"{path} 必须是有序序列")
    return tuple(value)


def _string(
    value: Any,
    path: str,
    *,
    error_type: type[ProjectV1Error],
) -> str:
    if not isinstance(value, str):
        raise error_type(f"{path} 必须是字符串")
    if not value.strip():
        raise error_type(f"{path} 不能为空")
    return value


def _integer(
    value: Any,
    path: str,
    *,
    error_type: type[ProjectV1Error],
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise error_type(f"{path} 必须是整数")
    return value


def _positive_integer(
    value: Any,
    path: str,
    error_type: type[ProjectV1Error],
) -> int:
    result = _integer(value, path, error_type=error_type)
    if result <= 0:
        raise error_type(f"{path} 必须大于零")
    return result


def _number(
    value: Any,
    path: str,
    error_type: type[ProjectV1Error],
) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise error_type(f"{path} 必须是数值")
    if isinstance(value, float) and not math.isfinite(value):
        raise error_type(f"{path} 必须是有限数值")
    return value


def _target(
    value: Any,
    path: str,
    error_type: type[ProjectV1Error],
) -> str | int:
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise error_type(f"{path} 必须是名称或整数编号")
    if isinstance(value, str) and not value.strip():
        raise error_type(f"{path} 不能为空")
    return value


def _number_array(
    value: Any,
    path: str,
    error_type: type[ProjectV1Error],
) -> tuple[int | float, ...]:
    if error_type is ProjectV1EncodeError:
        items = _runtime_sequence(value, path)
    else:
        items = _array(value, path, error_type=error_type)
    return tuple(
        _number(item, f"{path}[{index}]", error_type)
        for index, item in enumerate(items)
    )


def _json_object(
    value: Any,
    path: str,
    *,
    error_type: type[ProjectV1Error],
) -> dict[str, Any]:
    if type(value) is not dict:
        raise error_type(f"{path} 必须是普通 JSON object")
    return _json_value(value, path, error_type, set())


def _json_value(
    value: Any,
    path: str,
    error_type: type[ProjectV1Error],
    ancestors: set[int],
) -> Any:
    if value is None or isinstance(value, (bool, str, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise error_type(f"{path} 必须是有限数值")
        return value
    if type(value) is list:
        identity = id(value)
        if identity in ancestors:
            raise error_type(f"{path} 包含循环 JSON 引用")
        ancestors.add(identity)
        try:
            return [
                _json_value(item, f"{path}[{index}]", error_type, ancestors)
                for index, item in enumerate(value)
            ]
        finally:
            ancestors.remove(identity)
    if type(value) is dict:
        identity = id(value)
        if identity in ancestors:
            raise error_type(f"{path} 包含循环 JSON 引用")
        ancestors.add(identity)
        try:
            result: dict[str, Any] = {}
            for key, item in value.items():
                if not isinstance(key, str):
                    raise error_type(f"{path} 的 JSON object 键必须是字符串")
                result[key] = _json_value(
                    item, f"{path}.{key}", error_type, ancestors
                )
            return result
        finally:
            ancestors.remove(identity)
    raise error_type(
        f"{path} 的 {type(value).__name__} 值无法由 JSON 无损表示"
    )


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ProjectV1DecodeError(f"项目 JSON 包含重复键：{key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> NoReturn:
    raise ProjectV1DecodeError(f"项目 JSON 包含非有限数值：{value}")


__all__ = [
    "LOGICAL_TOPOLOGY_VERSION",
    "SCHEMA_VERSION",
    "ProjectV1DecodeError",
    "ProjectV1EncodeError",
    "ProjectV1Error",
    "decode_project_v1",
    "dumps_project_v1",
    "encode_project_v1",
    "load_project_v1",
    "load_native_project",
    "loads_project_v1",
    "read_project_v1",
    "save_project_v1",
    "save_native_project",
    "write_project_v1",
]
