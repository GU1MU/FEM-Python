"""Strict schema-v5 persistence for canonical multi-Body native geometry."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

from fem.application.definitions import normalize_model_definitions
from fem.application.definitions import NativePart
from fem.application.feature_history import derive_feature_history
from fem.application.project_validation import (
    NativeProjectValidationError,
    validate_native_project_inputs,
)
from fem.application.recipe_compiler import compile_recipe
from fem.application.session import (
    BooleanReferenceUndoRecord,
    ProjectSaveSnapshot,
    ProjectSnapshot,
)
from fem.geometry import model as geometry_model
from fem.geometry.recipe_analysis import legacy_sketches_to_strict
from fem.geometry.recipe_topology import (
    describe_recipe_topology,
    topology_fingerprint_for_recipe,
)
from fem.geometry.recipe_topology import canonicalize_multi_body_logical_id
from fem.geometry.references import LogicalEntityRef
from fem.geometry.recipes import (
    BooleanGeometry,
    ExtrudedGeometry,
    MovedGeometry,
    MultiBodyGeometry,
    NATIVE_GEOMETRY_TYPES,
    RotatedGeometry,
    geometry_dimension,
)

from . import project_v2 as _v2
from . import project_v4 as _v4
from ._project_codec import (
    ProjectFieldCodecPolicy,
    atomic_write_project,
    decode_assignment_field,
    decode_geometry_field,
    decode_material_field,
    decode_section_field,
    decode_step_field,
    dumps_canonical_json,
    encode_assignment_field,
    encode_geometry_field,
    encode_material_field,
    encode_section_field,
    encode_step_field,
    loads_json_strict,
    unwrap_project_snapshot,
)
from ._project_errors import ProjectDecodeError, ProjectEncodeError, ProjectError
from .project_migration import migrate_project_snapshot_to_v5


FORMAT_NAME = "fem-python-project"
SCHEMA_VERSION = 5


class ProjectV5Error(ProjectError):
    """Base error for schema-v5 project processing."""


class ProjectV5DecodeError(ProjectV5Error, ProjectDecodeError):
    """A schema-v5 payload is malformed or incompatible."""


class ProjectV5EncodeError(ProjectV5Error, ProjectEncodeError):
    """A snapshot cannot be represented losslessly by schema v5."""


_V5_FIELD_POLICY = ProjectFieldCodecPolicy(
    version_label="v5",
    decode_error=ProjectV5DecodeError,
    encode_error=ProjectV5EncodeError,
    require_current_fields=True,
    assignment_orientation=True,
    allow_wire_geometry=True,
    allow_strict_sketch=True,
    extrusion_source_faces=True,
    displacement_region_targets=True,
    body_force_loads=True,
    allow_multi_body=True,
    allow_planar_boolean=False,
)


def loads_project_v5(
    data: str | bytes | bytearray,
    *,
    source_path: str | Path | None = None,
) -> ProjectSnapshot:
    payload = loads_json_strict(
        data,
        error_type=ProjectV5DecodeError,
        document_label="v5 项目",
    )
    return decode_project_v5(payload, source_path=source_path)


def decode_project_v5(
    payload: Mapping[str, Any] | str | bytes | bytearray,
    *,
    source_path: str | Path | None = None,
    _field_policy: ProjectFieldCodecPolicy = _V5_FIELD_POLICY,
) -> ProjectSnapshot:
    """Decode one detached canonical schema-v5 project."""

    if isinstance(payload, (str, bytes, bytearray)):
        return loads_project_v5(payload, source_path=source_path)
    try:
        root = _v2._mapping(payload, "$", ProjectV5DecodeError)
        _v2._keys(
            root,
            "$",
            required={"format", "schema", "project"},
            optional=set(),
            error_type=ProjectV5DecodeError,
        )
        if _v2._string(
            root["format"],
            "$.format",
            ProjectV5DecodeError,
        ) != FORMAT_NAME:
            raise ProjectV5DecodeError(
                f"$.format 必须精确等于 {FORMAT_NAME!r}"
            )
        schema = _v2._integer(
            root["schema"],
            "$.schema",
            ProjectV5DecodeError,
        )
        if schema != SCHEMA_VERSION:
            raise ProjectV5DecodeError(
                f"v5 decoder 不能读取 schema {schema!r}"
            )
        project = _mapping_with_keys(
            root["project"],
            "$.project",
            {"kind", "authoring"},
        )
        if _v2._string(
            project["kind"],
            "$.project.kind",
            ProjectV5DecodeError,
        ) != "native":
            raise ProjectV5DecodeError("$.project.kind 只接受 'native'")
        authoring = _mapping_with_keys(
            project["authoring"],
            "$.project.authoring",
            {
                "part",
                "geometry",
                "logical_topology",
                "mesh_settings",
                "named_regions",
                "definitions",
                "boolean_undo_records",
            },
            optional={"model_name"},
        )
        model_name = _v2._string(
            authoring.get("model_name", "Model-1"),
            "$.project.authoring.model_name",
            ProjectV5DecodeError,
        )
        part = _decode_part_v5(
            authoring["part"],
            "$.project.authoring.part",
        )
        geometry = decode_geometry_field(
            authoring["geometry"],
            "$.project.authoring.geometry",
            policy=_field_policy,
        )
        if not isinstance(geometry, NATIVE_GEOMETRY_TYPES):
            raise ProjectV5DecodeError(
                "$.project.authoring.geometry 不是 native geometry recipe"
            )
        _require_v5_geometry(geometry, encode=False)
        stored_fingerprint = _v4._decode_topology_fingerprint_v4(
            authoring["logical_topology"],
            "$.project.authoring.logical_topology",
        )
        _v4._require_matching_topology_fingerprint_v4(
            stored_fingerprint,
            topology_fingerprint_for_recipe(geometry),
            authoring["logical_topology"],
            "$.project.authoring.logical_topology",
        )
        _authenticate_v5_boolean_proofs(geometry, encode=False)
        mesh_settings = _v4._decode_mesh_settings_v4(
            authoring["mesh_settings"],
            "$.project.authoring.mesh_settings",
            geometry,
        )
        named_regions = tuple(
            _v2._decode_named_region(
                item,
                f"$.project.authoring.named_regions[{index}]",
                geometry,
            )
            for index, item in enumerate(
                _v2._array(
                    authoring["named_regions"],
                    "$.project.authoring.named_regions",
                    ProjectV5DecodeError,
                )
            )
        )
        _v2._unique_names(
            named_regions,
            "$.project.authoring.named_regions",
            ProjectV5DecodeError,
        )
        definitions = _mapping_with_keys(
            authoring["definitions"],
            "$.project.authoring.definitions",
            {"materials", "sections", "assignments", "steps"},
        )
        materials = _decode_array(
            definitions["materials"],
            "$.project.authoring.definitions.materials",
            decode_material_field,
            policy=_field_policy,
        )
        sections = _decode_array(
            definitions["sections"],
            "$.project.authoring.definitions.sections",
            decode_section_field,
            policy=_field_policy,
        )
        assignments = _decode_array(
            definitions["assignments"],
            "$.project.authoring.definitions.assignments",
            decode_assignment_field,
            policy=_field_policy,
        )
        steps = _decode_array(
            definitions["steps"],
            "$.project.authoring.definitions.steps",
            decode_step_field,
            policy=_field_policy,
        )
        for values, path in (
            (materials, "$.project.authoring.definitions.materials"),
            (sections, "$.project.authoring.definitions.sections"),
            (steps, "$.project.authoring.definitions.steps"),
        ):
            _v2._unique_names(values, path, ProjectV5DecodeError)
        _v2._validate_definition_links_v2(
            materials,
            sections,
            assignments,
            encode=False,
        )
        normalized = normalize_model_definitions(
            materials,
            sections,
            assignments,
            steps,
        )
        boolean_undo_records = tuple(
            _decode_boolean_undo_record(
                item,
                f"$.project.authoring.boolean_undo_records[{index}]",
                policy=_field_policy,
            )
            for index, item in enumerate(
                _v2._array(
                    authoring["boolean_undo_records"],
                    "$.project.authoring.boolean_undo_records",
                    ProjectV5DecodeError,
                )
            )
        )
        _require_boolean_undo_record_coverage(
            geometry,
            boolean_undo_records,
            encode=False,
        )
        _require_canonical_v5_references(
            geometry,
            mesh_settings,
            named_regions,
            encode=False,
        )
        _validate_inputs(
            geometry,
            mesh_settings,
            named_regions,
            normalized.materials,
            normalized.sections,
            normalized.assignments,
            normalized.steps,
            encode=False,
        )
        return ProjectSnapshot(
            source_kind="native",
            source_path=None if source_path is None else Path(source_path),
            model_name=model_name,
            parts=(part,),
            geometry_recipe=geometry,
            mesh_settings=mesh_settings,
            feature_history=derive_feature_history(geometry),
            named_regions=named_regions,
            material_definitions=normalized.materials,
            section_definitions=normalized.sections,
            region_assignments=normalized.assignments,
            analysis_definitions=normalized.steps,
            model=None,
            boolean_reference_undo_records=boolean_undo_records,
        )
    except ProjectV5Error:
        raise
    except (_v2.ProjectV2Error, _v4.ProjectV4Error) as error:
        raise ProjectV5DecodeError(str(error)) from error
    except (KeyError, TypeError, ValueError) as error:
        raise ProjectV5DecodeError(
            f"$.project.authoring 无效：{error}"
        ) from error


def encode_project_v5(
    snapshot: ProjectSnapshot | ProjectSaveSnapshot,
    *,
    _field_policy: ProjectFieldCodecPolicy = _V5_FIELD_POLICY,
) -> dict[str, Any]:
    """Encode one detached snapshot as canonical schema v5."""

    try:
        project = unwrap_project_snapshot(
            snapshot,
            error_type=ProjectV5EncodeError,
        )
        if project.source_kind != "native":
            raise ProjectV5EncodeError(
                "snapshot.source_kind 只接受 'native'"
            )
        if project.model is not None:
            raise ProjectV5EncodeError(
                "v5 不持久化 compiled/runtime model"
            )
        project, _notices = migrate_project_snapshot_to_v5(project)
        geometry = project.geometry_recipe
        if geometry is None or not isinstance(
            geometry,
            NATIVE_GEOMETRY_TYPES,
        ):
            raise ProjectV5EncodeError(
                "snapshot.geometry_recipe 必须是 native geometry"
            )
        geometry = legacy_sketches_to_strict(geometry)
        project = replace(
            project,
            geometry_recipe=geometry,
            feature_history=derive_feature_history(geometry),
        )
        _require_v5_geometry(geometry, encode=True)
        _authenticate_v5_boolean_proofs(geometry, encode=True)
        parts = tuple(project.parts)
        if len(parts) != 1:
            raise ProjectV5EncodeError(
                "v5 native 项目必须且只能包含一个 part"
            )
        named_regions = tuple(project.named_regions)
        materials = tuple(project.material_definitions)
        sections = tuple(project.section_definitions)
        assignments = tuple(project.region_assignments)
        steps = tuple(project.analysis_definitions)
        boolean_undo_records = tuple(
            project.boolean_reference_undo_records
        )
        _require_boolean_undo_record_coverage(
            geometry,
            boolean_undo_records,
            encode=True,
        )
        _v2._validate_encode_contextual_references(
            geometry,
            named_regions,
            project.mesh_settings,
        )
        _v2._validate_definition_links_v2(
            materials,
            sections,
            assignments,
            encode=True,
        )
        normalized = normalize_model_definitions(
            materials,
            sections,
            assignments,
            steps,
        )
        if (
            materials != normalized.materials
            or sections != normalized.sections
            or assignments != normalized.assignments
            or steps != normalized.steps
        ):
            raise ProjectV5EncodeError(
                "snapshot definitions 不是 canonical 形式"
            )
        _require_canonical_v5_references(
            geometry,
            project.mesh_settings,
            named_regions,
            encode=True,
        )
        _validate_inputs(
            geometry,
            project.mesh_settings,
            named_regions,
            materials,
            sections,
            assignments,
            steps,
            encode=True,
        )
        payload = {
            "format": FORMAT_NAME,
            "schema": SCHEMA_VERSION,
            "project": {
                "kind": "native",
                "authoring": {
                    "model_name": _v2._string(
                        project.model_name,
                        "snapshot.model_name",
                        ProjectV5EncodeError,
                    ),
                    "part": _encode_part_v5(
                        parts[0],
                        "snapshot.parts[0]",
                    ),
                    "geometry": encode_geometry_field(
                        geometry,
                        "snapshot.geometry_recipe",
                        set(),
                        policy=_field_policy,
                    ),
                    "logical_topology": (
                        _v4._encode_topology_fingerprint_v4(
                            topology_fingerprint_for_recipe(geometry)
                        )
                    ),
                    "mesh_settings": _v4._encode_mesh_settings_v4(
                        project.mesh_settings,
                        "snapshot.mesh_settings",
                        geometry,
                    ),
                    "named_regions": [
                        _v2._encode_named_region(
                            region,
                            f"snapshot.named_regions[{index}]",
                        )
                        for index, region in enumerate(named_regions)
                    ],
                    "boolean_undo_records": [
                        _encode_boolean_undo_record(
                            record,
                            f"snapshot.boolean_reference_undo_records[{index}]",
                            policy=_field_policy,
                        )
                        for index, record in enumerate(
                            boolean_undo_records
                        )
                    ],
                    "definitions": {
                        "materials": _encode_array(
                            materials,
                            "snapshot.material_definitions",
                            encode_material_field,
                            policy=_field_policy,
                        ),
                        "sections": _encode_array(
                            sections,
                            "snapshot.section_definitions",
                            encode_section_field,
                            policy=_field_policy,
                        ),
                        "assignments": _encode_array(
                            assignments,
                            "snapshot.region_assignments",
                            encode_assignment_field,
                            policy=_field_policy,
                        ),
                        "steps": _encode_array(
                            steps,
                            "snapshot.analysis_definitions",
                            encode_step_field,
                            policy=_field_policy,
                        ),
                    },
                },
            },
        }
        dumps_canonical_json(payload, error_type=ProjectV5EncodeError)
        return payload
    except ProjectV5Error:
        raise
    except (_v2.ProjectV2Error, _v4.ProjectV4Error) as error:
        raise ProjectV5EncodeError(str(error)) from error
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise ProjectV5EncodeError(
            f"snapshot 无法由 v5 无损表示：{error}"
        ) from error


def dumps_project_v5(
    snapshot: ProjectSnapshot | ProjectSaveSnapshot,
) -> str:
    return dumps_canonical_json(
        encode_project_v5(snapshot),
        error_type=ProjectV5EncodeError,
    )


def load_project_v5(path: str | Path) -> ProjectSnapshot:
    source = Path(path)
    return loads_project_v5(source.read_bytes(), source_path=source)


def save_project_v5(
    path: str | Path,
    snapshot: ProjectSnapshot | ProjectSaveSnapshot,
    *,
    checkpoint: Callable[[], Any] | None = None,
) -> Path:
    if checkpoint is not None:
        checkpoint()
    payload = encode_project_v5(snapshot)
    if checkpoint is not None:
        checkpoint()
    serialized = dumps_canonical_json(
        payload,
        error_type=ProjectV5EncodeError,
    )
    return atomic_write_project(
        path,
        serialized,
        verifier=load_project_v5,
        semantic_encoder=encode_project_v5,
        expected_semantic=payload,
        error_type=ProjectV5EncodeError,
        mismatch_message=(
            "临时 v5 项目回读后的 canonical authoring 值与保存 snapshot 不一致"
        ),
        checkpoint=checkpoint,
    )


read_project_v5 = load_project_v5
write_project_v5 = save_project_v5


def _mapping_with_keys(
    value: Any,
    path: str,
    required: set[str],
    *,
    optional: set[str] | None = None,
) -> Mapping[str, Any]:
    data = _v2._mapping(value, path, ProjectV5DecodeError)
    _v2._keys(
        data,
        path,
        required=required,
        optional=set(optional or ()),
        error_type=ProjectV5DecodeError,
    )
    return data


def _decode_part_v5(value: Any, path: str) -> NativePart:
    data = _mapping_with_keys(value, path, {"name"})
    try:
        return NativePart(
            _v2._string(
                data["name"],
                f"{path}.name",
                ProjectV5DecodeError,
            )
        )
    except (TypeError, ValueError) as error:
        raise ProjectV5DecodeError(f"{path} 无效：{error}") from error


def _encode_part_v5(part: Any, path: str) -> dict[str, Any]:
    if type(part) is not NativePart:
        raise ProjectV5EncodeError(f"{path} 必须是 NativePart")
    return {
        "name": _v2._string(
            part.name,
            f"{path}.name",
            ProjectV5EncodeError,
        )
    }


def _decode_array(
    value: Any,
    path: str,
    decoder,
    *,
    policy: ProjectFieldCodecPolicy = _V5_FIELD_POLICY,
) -> tuple[Any, ...]:
    return tuple(
        decoder(item, f"{path}[{index}]", policy=policy)
        for index, item in enumerate(
            _v2._array(value, path, ProjectV5DecodeError)
        )
    )


def _encode_array(
    values: tuple[Any, ...],
    path: str,
    encoder,
    *,
    policy: ProjectFieldCodecPolicy = _V5_FIELD_POLICY,
) -> list[Any]:
    return [
        encoder(item, f"{path}[{index}]", policy=policy)
        for index, item in enumerate(values)
    ]


def _decode_boolean_undo_record(
    value: Any,
    path: str,
    *,
    policy: ProjectFieldCodecPolicy = _V5_FIELD_POLICY,
) -> BooleanReferenceUndoRecord:
    data = _mapping_with_keys(
        value,
        path,
        {
            "feature_id",
            "target_body_id",
            "before_geometry",
            "after_geometry",
            "before_named_regions",
            "after_named_regions",
            "before_mesh_settings",
            "after_mesh_settings",
            "before_materials",
            "after_materials",
            "before_sections",
            "after_sections",
            "before_assignments",
            "after_assignments",
            "before_steps",
            "after_steps",
        },
    )
    geometries = {
        phase: decode_geometry_field(
            data[f"{phase}_geometry"],
            f"{path}.{phase}_geometry",
            policy=policy,
        )
        for phase in ("before", "after")
    }
    states: dict[str, tuple[Any, ...] | Any | None] = {}
    for phase, geometry in geometries.items():
        if not isinstance(geometry, NATIVE_GEOMETRY_TYPES):
            raise ProjectV5DecodeError(
                f"{path}.{phase}_geometry must be native geometry"
            )
        _require_v5_geometry(geometry, encode=False)
        _authenticate_v5_boolean_proofs(geometry, encode=False)
        regions = tuple(
            _v2._decode_named_region(
                item,
                f"{path}.{phase}_named_regions[{index}]",
                geometry,
            )
            for index, item in enumerate(
                _v2._array(
                    data[f"{phase}_named_regions"],
                    f"{path}.{phase}_named_regions",
                    ProjectV5DecodeError,
                )
            )
        )
        _v2._unique_names(
            regions,
            f"{path}.{phase}_named_regions",
            ProjectV5DecodeError,
        )
        mesh_settings = _v4._decode_mesh_settings_v4(
            data[f"{phase}_mesh_settings"],
            f"{path}.{phase}_mesh_settings",
            geometry,
        )
        record_materials = _decode_array(
            data[f"{phase}_materials"],
            f"{path}.{phase}_materials",
            decode_material_field,
            policy=policy,
        )
        record_sections = _decode_array(
            data[f"{phase}_sections"],
            f"{path}.{phase}_sections",
            decode_section_field,
            policy=policy,
        )
        assignments = _decode_array(
            data[f"{phase}_assignments"],
            f"{path}.{phase}_assignments",
            decode_assignment_field,
            policy=policy,
        )
        steps = _decode_array(
            data[f"{phase}_steps"],
            f"{path}.{phase}_steps",
            decode_step_field,
            policy=policy,
        )
        normalized = normalize_model_definitions(
            record_materials,
            record_sections,
            assignments,
            steps,
        )
        if (
            normalized.materials != record_materials
            or normalized.sections != record_sections
            or normalized.assignments != assignments
            or normalized.steps != steps
        ):
            raise ProjectV5DecodeError(
                f"{path}.{phase} reference state is not canonical"
            )
        _require_canonical_v5_references(
            geometry,
            mesh_settings,
            regions,
            encode=False,
        )
        _validate_inputs(
            geometry,
            mesh_settings,
            regions,
            record_materials,
            record_sections,
            assignments,
            steps,
            encode=False,
        )
        states[f"{phase}_named_regions"] = regions
        states[f"{phase}_mesh_settings"] = mesh_settings
        states[f"{phase}_materials"] = record_materials
        states[f"{phase}_sections"] = record_sections
        states[f"{phase}_assignments"] = assignments
        states[f"{phase}_steps"] = steps
    try:
        return BooleanReferenceUndoRecord(
            _v2._string(
                data["feature_id"],
                f"{path}.feature_id",
                ProjectV5DecodeError,
            ),
            _v2._string(
                data["target_body_id"],
                f"{path}.target_body_id",
                ProjectV5DecodeError,
            ),
            geometries["before"],
            geometries["after"],
            states["before_named_regions"],
            states["after_named_regions"],
            states["before_mesh_settings"],
            states["after_mesh_settings"],
            states["before_materials"],
            states["after_materials"],
            states["before_sections"],
            states["after_sections"],
            states["before_assignments"],
            states["after_assignments"],
            states["before_steps"],
            states["after_steps"],
        )
    except (TypeError, ValueError) as error:
        raise ProjectV5DecodeError(f"{path} invalid: {error}") from error


def _encode_boolean_undo_record(
    record: BooleanReferenceUndoRecord,
    path: str,
    *,
    policy: ProjectFieldCodecPolicy = _V5_FIELD_POLICY,
) -> dict[str, Any]:
    if type(record) is not BooleanReferenceUndoRecord:
        raise ProjectV5EncodeError(
            f"{path} must be BooleanReferenceUndoRecord"
        )
    payload: dict[str, Any] = {
        "feature_id": record.feature_id,
        "target_body_id": record.target_body_id,
    }
    for phase in ("before", "after"):
        geometry = getattr(record, f"{phase}_geometry")
        regions = tuple(getattr(record, f"{phase}_named_regions"))
        mesh_settings = getattr(record, f"{phase}_mesh_settings")
        record_materials = tuple(getattr(record, f"{phase}_materials"))
        record_sections = tuple(getattr(record, f"{phase}_sections"))
        assignments = tuple(getattr(record, f"{phase}_assignments"))
        steps = tuple(getattr(record, f"{phase}_steps"))
        _require_v5_geometry(geometry, encode=True)
        _authenticate_v5_boolean_proofs(geometry, encode=True)
        _require_canonical_v5_references(
            geometry,
            mesh_settings,
            regions,
            encode=True,
        )
        _validate_inputs(
            geometry,
            mesh_settings,
            regions,
            record_materials,
            record_sections,
            assignments,
            steps,
            encode=True,
        )
        payload[f"{phase}_geometry"] = encode_geometry_field(
            geometry,
            f"{path}.{phase}_geometry",
            set(),
            policy=policy,
        )
        payload[f"{phase}_named_regions"] = [
            _v2._encode_named_region(
                region,
                f"{path}.{phase}_named_regions[{index}]",
            )
            for index, region in enumerate(regions)
        ]
        payload[f"{phase}_mesh_settings"] = (
            _v4._encode_mesh_settings_v4(
                mesh_settings,
                f"{path}.{phase}_mesh_settings",
                geometry,
            )
        )
        payload[f"{phase}_materials"] = _encode_array(
            record_materials,
            f"{path}.{phase}_materials",
            encode_material_field,
            policy=policy,
        )
        payload[f"{phase}_sections"] = _encode_array(
            record_sections,
            f"{path}.{phase}_sections",
            encode_section_field,
            policy=policy,
        )
        payload[f"{phase}_assignments"] = _encode_array(
            assignments,
            f"{path}.{phase}_assignments",
            encode_assignment_field,
            policy=policy,
        )
        payload[f"{phase}_steps"] = _encode_array(
            steps,
            f"{path}.{phase}_steps",
            encode_step_field,
            policy=policy,
        )
    return payload


def _require_boolean_undo_record_coverage(
    geometry: object,
    records: tuple[BooleanReferenceUndoRecord, ...],
    *,
    encode: bool,
) -> None:
    active_feature_ids = {
        context.feature_id
        for boolean in _all_boolean_recipes(geometry)
        for context in (boolean.body_context, boolean.planar_context)
        if context is not None
    }
    record_ids = tuple(record.feature_id for record in records)
    error_type = ProjectV5EncodeError if encode else ProjectV5DecodeError
    prefix = (
        "snapshot.boolean_reference_undo_records"
        if encode
        else "$.project.authoring.boolean_undo_records"
    )
    if len(record_ids) != len(set(record_ids)):
        raise error_type(f"{prefix} contains duplicate feature IDs")
    if set(record_ids) != active_feature_ids:
        raise error_type(
            f"{prefix} must cover every active strict Boolean feature"
        )


def _require_v5_geometry(recipe: object, *, encode: bool) -> None:
    if geometry_dimension(recipe) == 3 and not isinstance(
        recipe,
        MultiBodyGeometry,
    ):
        error_type = ProjectV5EncodeError if encode else ProjectV5DecodeError
        prefix = "snapshot.geometry_recipe" if encode else (
            "$.project.authoring.geometry"
        )
        raise error_type(
            f"{prefix} 的 3D geometry 必须使用 MultiBodyGeometry"
        )
    if isinstance(recipe, MultiBodyGeometry):
        active_ids = {body.id for body in recipe.bodies}
        error_type = ProjectV5EncodeError if encode else ProjectV5DecodeError
        prefix = (
            "snapshot.geometry_recipe"
            if encode
            else "$.project.authoring.geometry"
        )
        active_feature_paths: dict[str, str] = {}
        for index, body in enumerate(recipe.bodies):
            for suffix, boolean, expected_owner_id in _strict_boolean_recipes(
                body.recipe,
                body.id,
            ):
                context = boolean.body_context
                if context is None:
                    continue
                path = (
                    f"snapshot.geometry_recipe.bodies[{index}].recipe{suffix}"
                    ".body_context"
                    if encode
                    else (
                        "$.project.authoring.geometry.bodies"
                        f"[{index}].recipe{suffix}.body_context"
                    )
                )
                if context.target_body_id != expected_owner_id:
                    raise error_type(
                        f"{path}.target_body_id must equal owning Body ID "
                        f"{expected_owner_id!r}"
                    )
                if context.tool_body_id in active_ids:
                    raise error_type(
                        f"{path}.tool_body_id must name a consumed Body"
                    )
                if not context.proven:
                    raise error_type(
                        f"{path} must contain proven Boolean lineage"
                    )
                if context.tool_body_id not in recipe.retired_body_ids:
                    raise error_type(
                        f"{path}.tool_body_id must be present in "
                        "retired_body_ids"
                    )
                previous = active_feature_paths.get(context.feature_id)
                if previous is not None:
                    raise error_type(
                        f"{path}.feature_id duplicates active feature "
                        f"{context.feature_id!r} at {previous}"
                    )
                active_feature_paths[context.feature_id] = path
                source_ids = {
                    "target": {
                        entity.logical_id
                        for entity in describe_recipe_topology(
                            boolean.object_geometry
                        ).entities
                        if entity.selectable
                    },
                    "tool": {
                        entity.logical_id
                        for entity in describe_recipe_topology(
                            boolean.tool_geometry
                        ).entities
                        if entity.selectable
                    },
                }
                for mapping_index, mapping in enumerate(
                    context.topology_mappings
                ):
                    if (
                        mapping.source_logical_id
                        not in source_ids[mapping.source]
                    ):
                        raise error_type(
                            f"{path}.topology_mappings[{mapping_index}] "
                            f"references unknown {mapping.source} source "
                            f"{mapping.source_logical_id!r}"
                        )
        conflict = set(active_feature_paths) & set(
            recipe.retired_boolean_feature_ids
        )
        if conflict:
            feature_id = min(conflict, key=lambda value: int(value[2:]))
            raise error_type(
                f"{prefix}.retired_boolean_feature_ids conflicts with "
                f"active feature {feature_id!r}"
            )
    error_type = ProjectV5EncodeError if encode else ProjectV5DecodeError
    prefix = (
        "snapshot.geometry_recipe"
        if encode
        else "$.project.authoring.geometry"
    )
    for boolean in _all_boolean_recipes(recipe):
        context = boolean.planar_context
        if context is not None and not context.proven:
            raise error_type(
                f"{prefix}.planar_context must contain proven Boolean lineage"
            )


def _strict_boolean_recipes(
    recipe: Any,
    expected_owner_id: str,
    suffix: str = "",
) -> tuple[tuple[str, BooleanGeometry, str], ...]:
    if isinstance(recipe, (MovedGeometry, RotatedGeometry, ExtrudedGeometry)):
        return _strict_boolean_recipes(
            recipe.base,
            expected_owner_id,
            f"{suffix}.base",
        )
    if isinstance(recipe, BooleanGeometry):
        context = recipe.body_context
        object_owner_id = (
            expected_owner_id
            if context is None
            else context.target_body_id
        )
        tool_owner_id = (
            expected_owner_id
            if context is None
            else context.tool_body_id
        )
        return (
            (suffix, recipe, expected_owner_id),
            *_strict_boolean_recipes(
                recipe.object_geometry,
                object_owner_id,
                f"{suffix}.object",
            ),
            *_strict_boolean_recipes(
                recipe.tool_geometry,
                tool_owner_id,
                f"{suffix}.tool",
            ),
        )
    return ()


def _all_boolean_recipes(recipe: object) -> tuple[BooleanGeometry, ...]:
    if isinstance(recipe, MultiBodyGeometry):
        return tuple(
            boolean
            for body in recipe.bodies
            for boolean in _all_boolean_recipes(body.recipe)
        )
    if isinstance(recipe, (MovedGeometry, RotatedGeometry, ExtrudedGeometry)):
        return _all_boolean_recipes(recipe.base)
    if isinstance(recipe, BooleanGeometry):
        return (
            recipe,
            *_all_boolean_recipes(recipe.object_geometry),
            *_all_boolean_recipes(recipe.tool_geometry),
        )
    return ()


def _authenticate_v5_boolean_proofs(
    recipe: object,
    *,
    encode: bool,
) -> None:
    """Replay strict OCC proofs before a decoded snapshot can be installed."""

    if not any(
        boolean.body_context is not None or boolean.planar_context is not None
        for boolean in _all_boolean_recipes(recipe)
    ):
        return
    try:
        with geometry_model(
            "project-v5-boolean-proof-authentication",
            dimension=geometry_dimension(recipe),
        ) as cad:
            compile_recipe(cad, recipe)
    except Exception as error:
        error_type = ProjectV5EncodeError if encode else ProjectV5DecodeError
        prefix = (
            "snapshot.geometry_recipe"
            if encode
            else "$.project.authoring.geometry"
        )
        raise error_type(
            f"{prefix} Boolean OCC proof authentication failed: {error}"
        ) from error


def _require_canonical_v5_references(
    geometry: Any,
    mesh_settings: Any,
    named_regions: tuple[Any, ...],
    *,
    encode: bool,
) -> None:
    if not isinstance(geometry, MultiBodyGeometry):
        return
    error_type = ProjectV5EncodeError if encode else ProjectV5DecodeError
    prefix = "snapshot" if encode else "$.project.authoring"
    values: list[tuple[str, LogicalEntityRef]] = []
    for region_index, region in enumerate(named_regions):
        for reference_index, reference in enumerate(region.references):
            if type(reference) is LogicalEntityRef:
                values.append(
                    (
                        f"{prefix}.named_regions[{region_index}]"
                        f".references[{reference_index}]",
                        reference,
                    )
                )
    if mesh_settings is not None:
        for index, control in enumerate(mesh_settings.local_controls):
            values.append(
                (
                    f"{prefix}.mesh_settings.local_controls[{index}].target",
                    control.target,
                )
            )
    for path, reference in values:
        try:
            canonical = canonicalize_multi_body_logical_id(
                geometry,
                reference.logical_id,
            )
        except KeyError as error:
            raise error_type(
                f"{path} contains unknown logical reference "
                f"{reference.logical_id!r}"
            ) from error
        if canonical != reference.logical_id:
            raise error_type(
                f"{path} must use canonical Body namespace {canonical!r}"
            )


def _validate_inputs(
    recipe: Any,
    mesh_settings: Any,
    named_regions: tuple[Any, ...],
    materials: tuple[Any, ...],
    sections: tuple[Any, ...],
    assignments: tuple[Any, ...],
    steps: tuple[Any, ...],
    *,
    encode: bool,
) -> None:
    try:
        validate_native_project_inputs(
            recipe,
            mesh_settings,
            named_regions,
            materials,
            sections,
            assignments,
            steps,
        )
    except NativeProjectValidationError as error:
        error_type = ProjectV5EncodeError if encode else ProjectV5DecodeError
        prefix = "snapshot" if encode else "$.project.authoring"
        raise error_type(f"{prefix} 无效：{error}") from error


__all__ = [
    "FORMAT_NAME",
    "SCHEMA_VERSION",
    "ProjectV5DecodeError",
    "ProjectV5EncodeError",
    "ProjectV5Error",
    "decode_project_v5",
    "dumps_project_v5",
    "encode_project_v5",
    "load_project_v5",
    "loads_project_v5",
    "read_project_v5",
    "save_project_v5",
    "write_project_v5",
]
