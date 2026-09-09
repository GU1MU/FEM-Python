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
        raise ProjectV1DecodeError(f"unsupported project schema: {schema!r}")
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
        raise ProjectV1DecodeError("v1 projects only support source='native'")

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
        raise ProjectV1EncodeError("v1 projects only support saving Native Session")

    geometry = _snapshot_attr(project, "geometry_recipe")
    if geometry is None:
        raise ProjectV1EncodeError("create a sketch or geometry before saving the project")

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
            "v1 Native projects must contain exactly one NativePart; "
            f"received {len(parts)}"
        )
    try:
        canonical_history = derive_feature_history(geometry)
    except (KeyError, TypeError, ValueError) as error:
        raise ProjectV1EncodeError(
            f"snapshot.geometry_recipe cannot derive canonical feature history: {error}"
        ) from error
    if features != canonical_history:
        raise ProjectV1EncodeError(
            "snapshot.feature_history must equal the geometry recipe's "
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
                "snapshot definitions are not current canonical authoring values"
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
            f"invalid snapshot Native authoring context: {error}"
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
        error_message="project contains JSON values that cannot be encoded losslessly",
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
        error_message="project contains values that cannot be encoded",
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
        error_message="project contains values that cannot be encoded",
    )
    return atomic_write_project(
        path,
        serialized + "\n",
        verifier=load_project_v1,
        semantic_encoder=encode_project_v1,
        expected_semantic=expected,
        error_type=ProjectV1EncodeError,
        mismatch_message="validated temporary project file does not match the saved snapshot",
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
        raise ProjectV1DecodeError(f"{path}.entity_kind is not a supported entity type")
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
        raise ProjectV1DecodeError(f"{path}.entity_ids contains duplicate entity IDs")
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
        raise ProjectV1DecodeError(f"{path}.size must be greater than zero")
    order = _integer(
        data.get("order", 1),
        f"{path}.order",
        error_type=ProjectV1DecodeError,
    )
    if order not in {1, 2}:
        raise ProjectV1DecodeError(f"{path}.order must be first or second order")
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
        raise ProjectV1DecodeError(f"{path}.cell_shape is not a supported mesh type")

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
            f"{path}.local_size must be greater than zero and less than the global size"
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
                "must be less than the global size"
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
            f"{path}.entity_kind only supports point, edge or face"
        )
    size = _number(data["size"], f"{path}.size", ProjectV1DecodeError)
    if size <= 0:
        raise ProjectV1DecodeError(f"{path}.size must be greater than zero")
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
    if type(part) is not NativePart:
        raise ProjectV1EncodeError(f"{path} must be NativePart")
    if (
        part.geometry_recipe is not None
        or part.mesh_settings is not None
        or part.suppressed
        or part.provenance is not None
    ):
        raise ProjectV1EncodeError(
            f"{path} contains part ownership fields that v1 cannot represent"
        )
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
        raise ProjectV1EncodeError(f"{path}.references must not be empty")
    if any(type(reference) is not LogicalEntityRef for reference in references):
        raise ProjectV1EncodeError(
            f"{path}.references must contain only LogicalEntityRef"
        )
    entity_kinds = {reference.kind for reference in references}
    if len(entity_kinds) != 1:
        raise ProjectV1EncodeError(
            f"{path}.references must not mix entity types"
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
        raise ProjectV1EncodeError(f"{path}.references contains duplicate entity references")
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
            "auto_level",
            "strict_cell_shape",
        },
        path,
    )
    if settings.line_element_type is not None:
        raise ProjectV1EncodeError(
            f"{path}.line_element_type cannot be represented losslessly by v1"
        )
    if settings.auto_level is not None:
        raise ProjectV1EncodeError(
            f"{path}.auto_level cannot be represented losslessly by v1"
        )
    if settings.strict_cell_shape:
        raise ProjectV1EncodeError(
            f"{path}.strict_cell_shape cannot be represented losslessly by v1"
        )
    size = _number(settings.size, f"{path}.size", ProjectV1EncodeError)
    if size <= 0:
        raise ProjectV1EncodeError(f"{path}.size must be greater than zero")
    order = _integer(
        settings.order,
        f"{path}.order",
        error_type=ProjectV1EncodeError,
    )
    if order not in {1, 2}:
        raise ProjectV1EncodeError(f"{path}.order must be first or second order")
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
        raise ProjectV1EncodeError(f"{path}.cell_shape is not a supported mesh type")

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
                f"{control_path}.size must be less than the global size"
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
                    f"{path}.local_controls contains multiple visible "
                    "target_radius profiles; v1 can only represent one local_size"
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
                        f"{control_path} cannot prove a v1 legacy hole target"
                    ) from error
            if control.target != legacy_hole_target:
                raise ProjectV1EncodeError(
                    f"{control_path}.target is not a uniquely provable v1 "
                    "legacy hole target"
                )
            local_size = _number(
                control.size,
                f"{control_path}.size",
                ProjectV1EncodeError,
            )
            continue
        raise ProjectV1EncodeError(
            f"{control_path}.falloff cannot be represented losslessly by v1; "
            "only global_size(0.0, 2.0) or "
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
                f"{control_path}.target must be LogicalEntityRef"
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
                f"{control_path}.size must be greater than zero"
            )
        key = (control.target, control.falloff)
        previous = unique.get(key)
        if previous is None:
            unique[key] = control
        elif previous.size != control.size:
            raise ProjectV1EncodeError(
                f"{path}.local_controls for target "
                f"{control.target.logical_id!r} and falloff "
                f"{control.falloff.reference!r} contains conflicting sizes"
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
            f"{path} must be LogicalEntityRef"
        )
    try:
        topology = describe_recipe_topology(recipe)
    except (KeyError, TypeError, ValueError) as error:
        raise ProjectV1EncodeError(
            f"{path} cannot read geometry topology: {error}"
        ) from error
    if not topology.exact:
        raise ProjectV1EncodeError(
            f"{path} cannot be reverse-encoded: geometry topology is not exact"
        )
    entities = topology.entities_of(reference.kind)
    for ordinal, entity in enumerate(entities, start=1):
        if entity.logical_id != reference.logical_id:
            continue
        if not entity.selectable:
            raise ProjectV1EncodeError(
                f"{path} points to unselectable entity {reference.logical_id!r}"
            )
        return ordinal
    raise ProjectV1EncodeError(
        f"{path} references a logical ID absent from the geometry catalog: "
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
                f"[{index}].beam_orientation cannot be represented losslessly by .femproj v1; "
                "v1 does not support beam orientation"
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
        raise ProjectV1EncodeError("v1 does not persist model artifacts; the current snapshot cannot be saved losslessly")
    return project


def _snapshot_attr(snapshot: Any, name: str) -> Any:
    try:
        return getattr(snapshot, name)
    except AttributeError as exc:
        raise ProjectV1EncodeError(f"ProjectSnapshot is missing field {name!r}") from exc


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
            f"{path} must be {expected_type.__name__}, received {type(value).__name__}"
        )
    actual_fields = {item.name for item in fields(expected_type)}
    if actual_fields != expected_fields:
        unsupported = sorted(actual_fields ^ expected_fields)
        raise ProjectV1EncodeError(
            f"{expected_type.__name__} does not match the v1 field contract; refusing silent loss: "
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
        raise ProjectV1DecodeError(f"{path} is invalid: {exc}") from exc


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
            raise error_type(f"{path} contains duplicate name: {name!r}")
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
        raise error_type(f"{path} is missing required fields: {', '.join(missing)}")
    unknown = sorted(actual - required - optional)
    if unknown:
        raise error_type(f"{path} contains unknown v1 fields: {', '.join(unknown)}")


def _mapping(
    value: Any,
    path: str,
    *,
    error_type: type[ProjectV1Error],
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise error_type(f"{path} must be a JSON object")
    if any(not isinstance(key, str) for key in value):
        raise error_type(f"{path} keys must all be strings")
    return value


def _array(
    value: Any,
    path: str,
    *,
    error_type: type[ProjectV1Error],
) -> tuple[Any, ...]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise error_type(f"{path} must be a JSON array")
    return tuple(value)


def _runtime_sequence(value: Any, path: str) -> tuple[Any, ...]:
    if isinstance(value, (str, bytes, bytearray, Mapping)) or not isinstance(
        value, Sequence
    ):
        raise ProjectV1EncodeError(f"{path} must be an ordered sequence")
    return tuple(value)


def _string(
    value: Any,
    path: str,
    *,
    error_type: type[ProjectV1Error],
) -> str:
    if not isinstance(value, str):
        raise error_type(f"{path} must be a string")
    if not value.strip():
        raise error_type(f"{path} must not be empty")
    return value


def _integer(
    value: Any,
    path: str,
    *,
    error_type: type[ProjectV1Error],
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise error_type(f"{path} must be an integer")
    return value


def _positive_integer(
    value: Any,
    path: str,
    error_type: type[ProjectV1Error],
) -> int:
    result = _integer(value, path, error_type=error_type)
    if result <= 0:
        raise error_type(f"{path} must be greater than zero")
    return result


def _number(
    value: Any,
    path: str,
    error_type: type[ProjectV1Error],
) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise error_type(f"{path} must be a number")
    if isinstance(value, float) and not math.isfinite(value):
        raise error_type(f"{path} must be a finite number")
    return value


def _target(
    value: Any,
    path: str,
    error_type: type[ProjectV1Error],
) -> str | int:
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise error_type(f"{path} must be a name or integer ID")
    if isinstance(value, str) and not value.strip():
        raise error_type(f"{path} must not be empty")
    return value


def _stable_encode_target(value: Any, path: str) -> str:
    if type(value) is not str or not value.strip():
        if isinstance(value, int) and not isinstance(value, bool):
            raise ProjectV1EncodeError(
                f"{path} cannot use a mesh integer target; "
                "v1 writer only accepts a non-empty stable region name"
            )
        raise ProjectV1EncodeError(
            f"{path} must be a non-empty stable region name"
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
        raise error_type(f"{path} must be a plain JSON object")
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
            raise error_type(f"{path} must be a finite number")
        return value
    if type(value) is list:
        identity = id(value)
        if identity in ancestors:
            raise error_type(f"{path} contains a circular JSON reference")
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
            raise error_type(f"{path} contains a circular JSON reference")
        ancestors.add(identity)
        try:
            result: dict[str, Any] = {}
            for key, item in value.items():
                if not isinstance(key, str):
                    raise error_type(f"{path} JSON object keys must be strings")
                result[key] = _json_value(
                    item, f"{path}.{key}", error_type, ancestors
                )
            return result
        finally:
            ancestors.remove(identity)
    raise error_type(
        f"{path} has a {type(value).__name__} value that JSON cannot represent losslessly"
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
