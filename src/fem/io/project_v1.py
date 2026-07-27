"""Lossless, detached persistence for native ``.femproj`` v1 projects.

The codec deliberately has no knowledge of a live :class:`ModelSession`.  A
load fully decodes and validates a detached ``ProjectSnapshot``; installing
that snapshot and accepting a completed save remain application-layer
transactions.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import fields
from functools import partial
import math
import os
from pathlib import Path
from typing import Any, TYPE_CHECKING

from fem.application.definitions import (
    FeatureRecord,
    NamedRegion,
    NativePart,
    normalize_model_definitions,
)
from fem.application.feature_history import derive_feature_history
from fem.geometry.recipes import (
    BooleanGeometry,
    ExtrudedGeometry,
    MovedGeometry,
    RotatedGeometry,
    SketchGeometry,
)
from fem.geometry.measurements import (
    TargetRadiusResolutionError,
    resolve_legacy_hole_target,
)
from fem.geometry.recipe_topology import describe_recipe_topology
from fem.geometry.references import LogicalEntityRef
from fem.mesh.settings import (
    LocalMeshControl,
    MeshSettings,
    MeshSizeFalloff,
)

from ._project_codec import (
    ProjectFieldCodecPolicy,
    atomic_write_project,
    decode_assignment_field,
    decode_geometry_field,
    decode_material_field,
    decode_section_field,
    decode_step_field,
    dumps_json,
    encode_assignment_field,
    encode_geometry_field,
    encode_material_field,
    encode_section_field,
    encode_step_field,
    loads_json_strict,
    unwrap_project_snapshot,
)
from ._project_errors import (
    ProjectDecodeError,
    ProjectEncodeError,
    ProjectError,
)
from .project_migration import (
    LegacyLocalMeshControlV1,
    LegacyMeshSettingsV1,
    LegacyNamedRegionV1,
    LegacyProjectV1,
    ProjectMigrationNotice,
    ProjectV1MigrationError,
    _validate_current_native_authoring,
    migrate_project_v1,
)

if TYPE_CHECKING:
    from fem.application.session import ProjectSaveSnapshot, ProjectSnapshot


SCHEMA_VERSION = 1
LOGICAL_TOPOLOGY_VERSION = 1


class ProjectV1Error(ProjectError):
    """Base error for a project that cannot be represented losslessly."""


class ProjectV1DecodeError(ProjectV1Error, ProjectDecodeError):
    """The serialized project is malformed, incomplete, or unsupported."""


class ProjectV1EncodeError(ProjectV1Error, ProjectEncodeError):
    """The snapshot contains state that v1 cannot encode losslessly."""


_V1_FIELD_POLICY = ProjectFieldCodecPolicy(
    version_label="v1",
    decode_error=ProjectV1DecodeError,
    encode_error=ProjectV1EncodeError,
    require_current_fields=False,
    assignment_orientation=False,
)

_decode_material = partial(
    decode_material_field,
    policy=_V1_FIELD_POLICY,
)
_decode_section = partial(
    decode_section_field,
    policy=_V1_FIELD_POLICY,
)
_decode_assignment = partial(
    decode_assignment_field,
    policy=_V1_FIELD_POLICY,
)
_decode_geometry = partial(
    decode_geometry_field,
    policy=_V1_FIELD_POLICY,
)
_decode_step = partial(
    decode_step_field,
    policy=_V1_FIELD_POLICY,
)
_encode_material = partial(
    encode_material_field,
    policy=_V1_FIELD_POLICY,
)
_encode_section = partial(
    encode_section_field,
    policy=_V1_FIELD_POLICY,
)
_encode_assignment = partial(
    encode_assignment_field,
    policy=_V1_FIELD_POLICY,
)
_encode_geometry = partial(
    encode_geometry_field,
    policy=_V1_FIELD_POLICY,
)
_encode_step = partial(
    encode_step_field,
    policy=_V1_FIELD_POLICY,
)


def loads_project_v1(
    data: str | bytes | bytearray,
    *,
    source_path: str | Path | None = None,
) -> ProjectSnapshot:
    """Decode a complete JSON document into a detached project snapshot."""

    payload = loads_json_strict(data, error_type=ProjectV1DecodeError)
    snapshot, _notices = _decode_project_v1_loaded(
        payload,
        source_path=source_path,
    )
    return snapshot


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
    snapshot, _notices = _decode_project_v1_loaded(
        payload,
        source_path=source_path,
    )
    return snapshot


def _decode_project_v1_loaded(
    payload: Mapping[str, Any],
    source_path: str | Path | None = None,
) -> tuple[ProjectSnapshot, tuple[ProjectMigrationNotice, ...]]:
    """Decode and migrate one parsed v1 mapping while retaining notices."""

    legacy = _decode_legacy_v1_payload(
        payload,
        source_path=source_path,
    )
    try:
        return migrate_project_v1(legacy)
    except ProjectV1MigrationError as error:
        raise ProjectV1DecodeError(str(error)) from error


def _decode_legacy_v1_payload(
    payload: Mapping[str, Any],
    *,
    source_path: str | Path | None = None,
) -> LegacyProjectV1:
    """Decode a parsed v1 mapping into its private frozen wire DTO."""

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
    source_kind = _string(root["source"], "$.source", error_type=ProjectV1DecodeError)
    if source_kind != "native":
        raise ProjectV1DecodeError("v1 项目只支持 source='native'")

    geometry = _decode_geometry(root["geometry"], "$.geometry")
    mesh_settings = _decode_mesh_settings(root.get("mesh_settings"), "$.mesh_settings")
    parts = (
        None
        if "parts" not in root
        else tuple(
            _decode_part(item, f"$.parts[{index}]")
            for index, item in enumerate(
                _array(root["parts"], "$.parts", error_type=ProjectV1DecodeError)
            )
        )
    )

    feature_history_present = "feature_history" in root
    if feature_history_present:
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

    return LegacyProjectV1(
        source_path=None if source_path is None else Path(source_path),
        logical_topology_version=topology_version,
        parts=parts,
        geometry_recipe=geometry,
        mesh_settings=mesh_settings,
        feature_history=feature_history,
        feature_history_present=feature_history_present,
        named_regions=named_regions,
        material_definitions=materials,
        section_definitions=sections,
        region_assignments=assignments,
        analysis_definitions=steps,
    )


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

    # Losslessness guards intentionally precede contextual capability
    # validation so callers receive the stable v1-specific reason first.
    _guard_v1_orientations(assignments)
    _guard_v1_analysis_targets(steps)

    if len(parts) != 1:
        raise ProjectV1EncodeError(
            "v1 native 项目必须恰好包含一个 NativePart；"
            f"收到 {len(parts)} 个"
        )
    try:
        canonical_history = derive_feature_history(geometry)
    except (KeyError, TypeError, ValueError) as error:
        raise ProjectV1EncodeError(
            f"snapshot.geometry_recipe 无法推导 canonical feature history：{error}"
        ) from error
    if features != canonical_history:
        raise ProjectV1EncodeError(
            "snapshot.feature_history 必须等于 geometry recipe 的 "
            "current canonical derivation"
        )

    _require_unique_names(named_regions, "snapshot.named_regions", encode=True)
    _require_unique_names(materials, "snapshot.material_definitions", encode=True)
    _require_unique_names(sections, "snapshot.section_definitions", encode=True)
    _require_unique_names(steps, "snapshot.analysis_definitions", encode=True)

    encoded_mesh_settings = _encode_mesh_settings(
        _snapshot_attr(project, "mesh_settings"),
        "snapshot.mesh_settings",
        geometry,
    )
    encoded_named_regions = [
        _encode_named_region(
            item,
            f"snapshot.named_regions[{index}]",
            geometry,
        )
        for index, item in enumerate(named_regions)
    ]

    try:
        definitions = normalize_model_definitions(
            materials,
            sections,
            assignments,
            steps,
        )
        if (
            definitions.materials != materials
            or definitions.sections != sections
            or definitions.assignments != assignments
            or definitions.steps != steps
        ):
            raise ProjectV1EncodeError(
                "snapshot definitions 不是 current canonical authoring values"
            )
        _validate_current_native_authoring(
            geometry,
            _snapshot_attr(project, "mesh_settings"),
            named_regions,
            definitions.materials,
            definitions.sections,
            definitions.assignments,
            definitions.steps,
        )
    except ProjectV1EncodeError:
        raise
    except (KeyError, TypeError, ValueError) as error:
        raise ProjectV1EncodeError(
            f"snapshot native authoring context 无效：{error}"
        ) from error

    payload = {
        "schema": SCHEMA_VERSION,
        "logical_topology_version": LOGICAL_TOPOLOGY_VERSION,
        "source": "native",
        "parts": [
            _encode_part(item, f"snapshot.parts[{index}]")
            for index, item in enumerate(parts)
        ],
        "geometry": _encode_geometry(geometry, "snapshot.geometry_recipe", set()),
        "mesh_settings": encoded_mesh_settings,
        "feature_history": [
            _encode_feature(item, f"snapshot.feature_history[{index}]")
            for index, item in enumerate(canonical_history)
        ],
        "named_regions": encoded_named_regions,
        "materials": [
            _encode_material(item, f"snapshot.material_definitions[{index}]")
            for index, item in enumerate(definitions.materials)
        ],
        "sections": [
            _encode_section(item, f"snapshot.section_definitions[{index}]")
            for index, item in enumerate(definitions.sections)
        ],
        "assignments": [
            _encode_assignment(item, f"snapshot.region_assignments[{index}]")
            for index, item in enumerate(definitions.assignments)
        ],
        "steps": [
            _encode_step(item, f"snapshot.analysis_definitions[{index}]")
            for index, item in enumerate(definitions.steps)
        ],
    }
    # This catches non-finite numbers and excessive/cyclic JSON metadata before
    # any destination file is touched.
    dumps_json(
        payload,
        indent=None,
        error_type=ProjectV1EncodeError,
        error_message="项目包含无法无损编码的 JSON 值",
    )
    return payload


def dumps_project_v1(
    snapshot: ProjectSnapshot | ProjectSaveSnapshot,
    *,
    indent: int | None = 2,
) -> str:
    """Serialize a project snapshot as deterministic UTF-8 JSON text."""

    return dumps_json(
        encode_project_v1(snapshot),
        indent=indent,
        error_type=ProjectV1EncodeError,
        error_message="项目包含无法编码的值",
    )


def save_project_v1(
    path: str | Path,
    snapshot: ProjectSnapshot | ProjectSaveSnapshot,
) -> Path:
    """Atomically save an immutable project snapshot.

    The temporary file is created beside the target, flushed to disk, decoded
    again for complete validation, and only then installed with ``os.replace``.
    """

    expected = encode_project_v1(snapshot)
    serialized = dumps_json(
        expected,
        indent=2,
        error_type=ProjectV1EncodeError,
        error_message="项目包含无法编码的值",
    )
    return atomic_write_project(
        path,
        serialized + "\n",
        verifier=load_project_v1,
        semantic_encoder=encode_project_v1,
        expected_semantic=expected,
        error_type=ProjectV1EncodeError,
        mismatch_message="临时项目文件校验后与保存 snapshot 不一致",
        replace_func=os.replace,
    )


# Explicit file-oriented aliases make call sites read naturally while keeping
# the primary names symmetric with ``json.loads`` / ``json.dumps``.
read_project_v1 = load_project_v1
write_project_v1 = save_project_v1


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


def _decode_named_region(value: Any, path: str) -> LegacyNamedRegionV1:
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
    return LegacyNamedRegionV1(
        name=_string(data["name"], f"{path}.name", error_type=ProjectV1DecodeError),
        entity_kind=entity_kind,
        entity_ids=entity_ids,
    )


def _decode_mesh_settings(value: Any, path: str) -> LegacyMeshSettingsV1 | None:
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
    size = _number(data["size"], f"{path}.size", ProjectV1DecodeError)
    if size <= 0:
        raise ProjectV1DecodeError(f"{path}.size 必须大于零")
    order = _integer(
        data.get("order", 1),
        f"{path}.order",
        error_type=ProjectV1DecodeError,
    )
    if order not in {1, 2}:
        raise ProjectV1DecodeError(f"{path}.order 只能是一阶或二阶")
    cell_shape = _string(
        data.get("cell_shape", "triangle"),
        f"{path}.cell_shape",
        error_type=ProjectV1DecodeError,
    )
    if cell_shape not in {
        "triangle",
        "quadrilateral",
        "tetrahedron",
        "hexahedron",
    }:
        raise ProjectV1DecodeError(f"{path}.cell_shape 不是受支持的网格类型")

    local_size_value = data.get("local_size")
    local_size = (
        None
        if local_size_value is None
        else _number(local_size_value, f"{path}.local_size", ProjectV1DecodeError)
    )
    if local_size is not None and (
        local_size <= 0 or local_size >= size
    ):
        raise ProjectV1DecodeError(
            f"{path}.local_size 必须大于零且小于全局尺寸"
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
    for index, control in enumerate(controls):
        if control.size >= size:
            raise ProjectV1DecodeError(
                f"{path}.local_controls[{index}].size "
                "必须小于全局尺寸"
            )
    return LegacyMeshSettingsV1(
        size=float(size),
        order=order,
        cell_shape=cell_shape,
        local_size=None if local_size is None else float(local_size),
        local_controls=controls,
    )


def _decode_local_control(value: Any, path: str) -> LegacyLocalMeshControlV1:
    data = _mapping(value, path, error_type=ProjectV1DecodeError)
    _keys(
        data,
        path,
        required={"entity_kind", "entity_id", "size"},
        optional=set(),
        error_type=ProjectV1DecodeError,
    )
    entity_kind = _string(
        data["entity_kind"],
        f"{path}.entity_kind",
        error_type=ProjectV1DecodeError,
    )
    if entity_kind not in {"point", "edge", "face"}:
        raise ProjectV1DecodeError(
            f"{path}.entity_kind 只支持 point、edge 或 face"
        )
    size = _number(data["size"], f"{path}.size", ProjectV1DecodeError)
    if size <= 0:
        raise ProjectV1DecodeError(f"{path}.size 必须大于零")
    return LegacyLocalMeshControlV1(
        entity_kind=entity_kind,
        entity_id=_positive_integer(
            data["entity_id"],
            f"{path}.entity_id",
            ProjectV1DecodeError,
        ),
        size=float(size),
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


def _encode_named_region(
    region: Any,
    path: str,
    recipe: Any,
) -> dict[str, Any]:
    _exact_dataclass(
        region,
        NamedRegion,
        {"name", "references"},
        path,
    )
    references = _runtime_sequence(region.references, f"{path}.references")
    if not references:
        raise ProjectV1EncodeError(f"{path}.references 不能为空")
    if any(type(reference) is not LogicalEntityRef for reference in references):
        raise ProjectV1EncodeError(
            f"{path}.references 必须只包含 LogicalEntityRef"
        )
    entity_kinds = {reference.kind for reference in references}
    if len(entity_kinds) != 1:
        raise ProjectV1EncodeError(
            f"{path}.references 不能混合不同实体类型"
        )
    entity_kind = next(iter(entity_kinds))
    entity_ids = [
        _reference_to_ordinal(
            recipe,
            reference,
            path=f"{path}.references[{index}]",
        )
        for index, reference in enumerate(references)
    ]
    if len(set(entity_ids)) != len(entity_ids):
        raise ProjectV1EncodeError(f"{path}.references 包含重复实体引用")
    return {
        "name": _string(
            region.name, f"{path}.name", error_type=ProjectV1EncodeError
        ),
        "entity_kind": entity_kind,
        "entity_ids": entity_ids,
    }


def _encode_mesh_settings(
    settings: Any,
    path: str,
    recipe: Any,
) -> dict[str, Any] | None:
    if settings is None:
        return None
    _exact_dataclass(
        settings,
        MeshSettings,
        {
            "size",
            "order",
            "cell_shape",
            "local_controls",
            "line_element_type",
        },
        path,
    )
    if settings.line_element_type is not None:
        raise ProjectV1EncodeError(
            f"{path}.line_element_type 无法由 v1 无损表示"
        )
    size = _number(settings.size, f"{path}.size", ProjectV1EncodeError)
    if size <= 0:
        raise ProjectV1EncodeError(f"{path}.size 必须大于零")
    order = _integer(
        settings.order,
        f"{path}.order",
        error_type=ProjectV1EncodeError,
    )
    if order not in {1, 2}:
        raise ProjectV1EncodeError(f"{path}.order 只能是一阶或二阶")
    cell_shape = _string(
        settings.cell_shape,
        f"{path}.cell_shape",
        error_type=ProjectV1EncodeError,
    )
    if cell_shape not in {
        "triangle",
        "quadrilateral",
        "tetrahedron",
        "hexahedron",
    }:
        raise ProjectV1EncodeError(f"{path}.cell_shape 不是受支持的网格类型")

    controls = _canonicalize_v1_writer_controls(
        _runtime_sequence(settings.local_controls, f"{path}.local_controls"),
        path=path,
    )
    legacy_local_controls: list[dict[str, Any]] = []
    local_size: float | int | None = None
    legacy_hole_target: LogicalEntityRef | None = None
    target_radius_count = 0
    for index, control in enumerate(controls):
        control_path = f"{path}.local_controls[{index}]"
        if control.size >= size:
            raise ProjectV1EncodeError(
                f"{control_path}.size 必须小于全局尺寸"
            )
        if control.falloff == MeshSizeFalloff("global_size", 0.0, 2.0):
            legacy_local_controls.append(
                _encode_local_control(control, control_path, recipe)
            )
            continue
        if control.falloff == MeshSizeFalloff("target_radius", 0.25, 2.0):
            target_radius_count += 1
            if target_radius_count > 1:
                raise ProjectV1EncodeError(
                    f"{path}.local_controls 包含多个可见的 "
                    "target_radius profile，v1 只能表示一个 local_size"
                )
            if legacy_hole_target is None:
                try:
                    legacy_hole_target = resolve_legacy_hole_target(recipe)
                except (
                    KeyError,
                    TypeError,
                    ValueError,
                    TargetRadiusResolutionError,
                ) as error:
                    raise ProjectV1EncodeError(
                        f"{control_path} 无法证明 v1 legacy hole target"
                    ) from error
            if control.target != legacy_hole_target:
                raise ProjectV1EncodeError(
                    f"{control_path}.target 不是唯一可证明的 v1 "
                    "legacy hole target"
                )
            local_size = _number(
                control.size,
                f"{control_path}.size",
                ProjectV1EncodeError,
            )
            continue
        raise ProjectV1EncodeError(
            f"{control_path}.falloff 无法由 v1 无损表示；"
            "只支持 global_size(0.0, 2.0) 或 "
            "target_radius(0.25, 2.0)"
        )

    return {
        "size": size,
        "order": order,
        "cell_shape": cell_shape,
        "local_size": local_size,
        "local_controls": legacy_local_controls,
    }


def _encode_local_control(
    control: Any,
    path: str,
    recipe: Any,
) -> dict[str, Any]:
    _exact_dataclass(
        control,
        LocalMeshControl,
        {"target", "size", "falloff"},
        path,
    )
    return {
        "entity_kind": control.target.kind,
        "entity_id": _reference_to_ordinal(
            recipe,
            control.target,
            path=f"{path}.target",
        ),
        "size": _number(control.size, f"{path}.size", ProjectV1EncodeError),
    }


def _canonicalize_v1_writer_controls(
    controls: Sequence[Any],
    *,
    path: str,
) -> tuple[LocalMeshControl, ...]:
    unique: dict[
        tuple[LogicalEntityRef, MeshSizeFalloff],
        LocalMeshControl,
    ] = {}
    for index, control in enumerate(controls):
        control_path = f"{path}.local_controls[{index}]"
        _exact_dataclass(
            control,
            LocalMeshControl,
            {"target", "size", "falloff"},
            control_path,
        )
        if type(control.target) is not LogicalEntityRef:
            raise ProjectV1EncodeError(
                f"{control_path}.target 必须是 LogicalEntityRef"
            )
        _exact_dataclass(
            control.falloff,
            MeshSizeFalloff,
            {"reference", "start_factor", "end_factor"},
            f"{control_path}.falloff",
        )
        size = _number(
            control.size,
            f"{control_path}.size",
            ProjectV1EncodeError,
        )
        if size <= 0:
            raise ProjectV1EncodeError(
                f"{control_path}.size 必须大于零"
            )
        key = (control.target, control.falloff)
        previous = unique.get(key)
        if previous is None:
            unique[key] = control
        elif previous.size != control.size:
            raise ProjectV1EncodeError(
                f"{path}.local_controls 对 target "
                f"{control.target.logical_id!r} 和 falloff "
                f"{control.falloff.reference!r} 包含冲突 size"
            )
    return tuple(unique.values())


def _reference_to_ordinal(
    recipe: Any,
    reference: LogicalEntityRef,
    *,
    path: str,
) -> int:
    if type(reference) is not LogicalEntityRef:
        raise ProjectV1EncodeError(
            f"{path} 必须是 LogicalEntityRef"
        )
    try:
        topology = describe_recipe_topology(recipe)
    except (KeyError, TypeError, ValueError) as error:
        raise ProjectV1EncodeError(
            f"{path} 无法读取 geometry topology：{error}"
        ) from error
    if not topology.exact:
        raise ProjectV1EncodeError(
            f"{path} 无法反向编码：geometry topology 不是 exact"
        )
    entities = topology.entities_of(reference.kind)
    for ordinal, entity in enumerate(entities, start=1):
        if entity.logical_id != reference.logical_id:
            continue
        if not entity.selectable:
            raise ProjectV1EncodeError(
                f"{path} 指向不可选择实体 {reference.logical_id!r}"
            )
        return ordinal
    raise ProjectV1EncodeError(
        f"{path} 引用了 geometry catalog 中不存在的 logical ID "
        f"{reference.logical_id!r}"
    )


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


def _guard_v1_orientations(assignments: Sequence[Any]) -> None:
    for index, assignment in enumerate(assignments):
        if getattr(assignment, "beam_orientation", None) is not None:
            raise ProjectV1EncodeError(
                "snapshot.region_assignments"
                f"[{index}].beam_orientation 无法由 .femproj v1 无损表示；"
                "v1 不支持 Beam orientation"
            )


def _guard_v1_analysis_targets(steps: Sequence[Any]) -> None:
    for step_index, step in enumerate(steps):
        for collection_name in ("boundaries", "cloads", "line_loads"):
            values = getattr(step, collection_name, ())
            if isinstance(values, (str, bytes, bytearray, Mapping)):
                continue
            for index, item in enumerate(values):
                _stable_encode_target(
                    getattr(item, "target", None),
                    "snapshot.analysis_definitions"
                    f"[{step_index}].{collection_name}[{index}].target",
                )
        gravity_loads = getattr(step, "gravity_loads", ())
        if isinstance(
            gravity_loads,
            (str, bytes, bytearray, Mapping),
        ):
            continue
        for index, load in enumerate(gravity_loads):
            target = getattr(load, "target", None)
            if target is not None:
                _stable_encode_target(
                    target,
                    "snapshot.analysis_definitions"
                    f"[{step_index}].gravity_loads[{index}].target",
                )


def _unwrap_project_snapshot(snapshot: Any) -> ProjectSnapshot:
    project = unwrap_project_snapshot(
        snapshot,
        error_type=ProjectV1EncodeError,
    )
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


def _stable_encode_target(value: Any, path: str) -> str:
    if type(value) is not str or not value.strip():
        if isinstance(value, int) and not isinstance(value, bool):
            raise ProjectV1EncodeError(
                f"{path} 不能使用 mesh integer target；"
                "v1 writer 只接受 non-empty stable region name"
            )
        raise ProjectV1EncodeError(
            f"{path} 必须是 non-empty stable region name"
        )
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
    "loads_project_v1",
    "read_project_v1",
    "save_project_v1",
    "write_project_v1",
]
