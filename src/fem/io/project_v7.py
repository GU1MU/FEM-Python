"""Strict schema-v7 persistence for canonical native Part ownership."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

from fem.application.definitions import (
    MeshEntityRef,
    NamedRegion,
    mesh_entity_ref_sort_key,
    normalize_model_definitions,
)
from fem.application.feature_history import derive_feature_history
from fem.application.native_part import (
    NativePart,
    PartBooleanProvenance,
    part_boolean_feature_id_sort_key,
    validate_native_parts,
)
from fem.application.recipe_compiler import compile_recipe
from fem.application.session import (
    PartBooleanUndoRecord,
    ProjectSaveSnapshot,
    ProjectSnapshot,
)
from fem.geometry import model as geometry_model
from fem.geometry.part_namespace import (
    part_id_from_logical_id,
    strip_part_logical_id,
)
from fem.geometry.recipe_analysis import legacy_sketches_to_strict
from fem.geometry.recipe_topology import (
    describe_recipe_topology,
    topology_fingerprint_for_recipe,
)
from fem.geometry.references import LogicalEntityRef
from fem.geometry.recipes import (
    BooleanGeometry,
    ExtrudedGeometry,
    FaceSketchBooleanGeometry,
    MovedGeometry,
    NATIVE_GEOMETRY_TYPES,
    RotatedGeometry,
    geometry_dimension,
)
from fem.mesh.settings import LocalMeshControl, MeshSettings, MeshSizeFalloff

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


SCHEMA_VERSION = 7
FORMAT_NAME = "fem-python-project"


class ProjectV7Error(ProjectError):
    """Base error for schema-v7 project processing."""


class ProjectV7DecodeError(ProjectV7Error, ProjectDecodeError):
    """A schema-v7 payload is malformed or incompatible."""


class ProjectV7EncodeError(ProjectV7Error, ProjectEncodeError):
    """A snapshot cannot be represented losslessly by schema v7."""


_V7_FIELD_POLICY = ProjectFieldCodecPolicy(
    version_label="v7",
    decode_error=ProjectV7DecodeError,
    encode_error=ProjectV7EncodeError,
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
)


def loads_project_v7(
    data: str | bytes | bytearray,
    *,
    source_path: str | Path | None = None,
) -> ProjectSnapshot:
    payload = loads_json_strict(
        data,
        error_type=ProjectV7DecodeError,
        document_label="v7 项目",
    )
    return decode_project_v7(payload, source_path=source_path)


def decode_project_v7(
    payload: Mapping[str, Any] | str | bytes | bytearray,
    *,
    source_path: str | Path | None = None,
    _field_policy: ProjectFieldCodecPolicy = _V7_FIELD_POLICY,
) -> ProjectSnapshot:
    """Decode canonical v7 Parts and authenticate every strict proof."""

    if isinstance(payload, (str, bytes, bytearray)):
        return loads_project_v7(payload, source_path=source_path)
    try:
        root = _mapping(payload, "$")
        _keys(root, "$", {"format", "schema", "project"})
        if _string(root["format"], "$.format") != FORMAT_NAME:
            raise ProjectV7DecodeError(
                f"$.format 必须精确等于 {FORMAT_NAME!r}"
            )
        if _integer(root["schema"], "$.schema") != SCHEMA_VERSION:
            raise ProjectV7DecodeError(
                f"v7 decoder 不能读取 schema {root['schema']!r}"
            )
        project = _mapping(root["project"], "$.project")
        _keys(project, "$.project", {"kind", "authoring"})
        if _string(project["kind"], "$.project.kind") != "native":
            raise ProjectV7DecodeError("$.project.kind 只接受 'native'")
        authoring = _mapping(project["authoring"], "$.project.authoring")
        _keys(
            authoring,
            "$.project.authoring",
            {
                "model_name",
                "active_part_id",
                "parts",
                "retired_part_ids",
                "retired_part_boolean_feature_ids",
                "named_regions",
                "definitions",
                "part_boolean_undo_records",
            },
        )
        parts = tuple(
            _decode_part(
                item,
                f"$.project.authoring.parts[{index}]",
                field_policy=_field_policy,
            )
            for index, item in enumerate(
                _array(authoring["parts"], "$.project.authoring.parts")
            )
        )
        parts = validate_native_parts(parts)
        active_part_id = (
            None
            if authoring["active_part_id"] is None
            else _string(
                authoring["active_part_id"],
                "$.project.authoring.active_part_id",
            )
        )
        retired_part_ids = tuple(
            _string(
                item,
                f"$.project.authoring.retired_part_ids[{index}]",
            )
            for index, item in enumerate(
                _array(
                    authoring["retired_part_ids"],
                    "$.project.authoring.retired_part_ids",
                )
            )
        )
        retired_feature_ids = tuple(
            _string(
                item,
                "$.project.authoring."
                f"retired_part_boolean_feature_ids[{index}]",
            )
            for index, item in enumerate(
                _array(
                    authoring["retired_part_boolean_feature_ids"],
                    "$.project.authoring."
                    "retired_part_boolean_feature_ids",
                )
            )
        )
        regions = tuple(
            _decode_named_region(
                item,
                f"$.project.authoring.named_regions[{index}]",
            )
            for index, item in enumerate(
                _array(
                    authoring["named_regions"],
                    "$.project.authoring.named_regions",
                )
            )
        )
        _require_unique_names(
            regions,
            "$.project.authoring.named_regions",
        )
        definitions = _mapping(
            authoring["definitions"],
            "$.project.authoring.definitions",
        )
        _keys(
            definitions,
            "$.project.authoring.definitions",
            {"materials", "sections", "assignments", "steps"},
        )
        materials = _decode_array(
            definitions["materials"],
            "$.project.authoring.definitions.materials",
            decode_material_field,
            field_policy=_field_policy,
        )
        sections = _decode_array(
            definitions["sections"],
            "$.project.authoring.definitions.sections",
            decode_section_field,
            field_policy=_field_policy,
        )
        assignments = _decode_array(
            definitions["assignments"],
            "$.project.authoring.definitions.assignments",
            decode_assignment_field,
            field_policy=_field_policy,
        )
        steps = _decode_array(
            definitions["steps"],
            "$.project.authoring.definitions.steps",
            decode_step_field,
            field_policy=_field_policy,
        )
        normalized = normalize_model_definitions(
            materials,
            sections,
            assignments,
            steps,
        )
        records = tuple(
            _decode_part_boolean_undo_record(
                item,
                f"$.project.authoring.part_boolean_undo_records[{index}]",
                field_policy=_field_policy,
            )
            for index, item in enumerate(
                _array(
                    authoring["part_boolean_undo_records"],
                    "$.project.authoring.part_boolean_undo_records",
                )
            )
        )
        _validate_part_graph(parts, records)
        _validate_references(parts, regions)
        for part in parts:
            _authenticate_part(part, encode=False)
        active = (
            next(
                (part for part in parts if part.id == active_part_id),
                None,
            )
            if active_part_id is not None
            else next(
                (part for part in parts if not part.suppressed),
                None,
            )
        )
        return ProjectSnapshot(
            source_kind="native",
            source_path=None if source_path is None else Path(source_path),
            model_name=_string(
                authoring["model_name"],
                "$.project.authoring.model_name",
            ),
            parts=parts,
            geometry_recipe=(
                None if active is None else active.geometry_recipe
            ),
            mesh_settings=(
                None if active is None else active.mesh_settings
            ),
            feature_history=(
                ()
                if active is None
                else derive_feature_history(active.geometry_recipe)
            ),
            named_regions=regions,
            material_definitions=normalized.materials,
            section_definitions=normalized.sections,
            region_assignments=normalized.assignments,
            analysis_definitions=normalized.steps,
            part_boolean_undo_records=records,
            retired_part_ids=retired_part_ids,
            retired_part_boolean_feature_ids=retired_feature_ids,
            active_part_id=active_part_id,
        )
    except ProjectV7Error:
        raise
    except Exception as error:
        raise ProjectV7DecodeError(
            f"$.project.authoring 无效：{error}"
        ) from error


def encode_project_v7(
    snapshot: ProjectSnapshot | ProjectSaveSnapshot,
    *,
    _field_policy: ProjectFieldCodecPolicy = _V7_FIELD_POLICY,
) -> dict[str, Any]:
    """Encode only canonical Part-owned authoring state."""

    try:
        project = unwrap_project_snapshot(snapshot)
        parts = validate_native_parts(tuple(project.parts))
        for part in parts:
            _authenticate_part(part, encode=True)
        regions = tuple(project.named_regions)
        _validate_references(parts, regions)
        normalized = normalize_model_definitions(
            tuple(project.material_definitions),
            tuple(project.section_definitions),
            tuple(project.region_assignments),
            tuple(project.analysis_definitions),
        )
        records = tuple(project.part_boolean_undo_records)
        _validate_part_graph(parts, records)
        payload = {
            "format": FORMAT_NAME,
            "schema": SCHEMA_VERSION,
            "project": {
                "kind": "native",
                "authoring": {
                    "model_name": str(project.model_name),
                    "active_part_id": project.active_part_id,
                    "parts": [
                        _encode_part(
                            part,
                            f"snapshot.parts[{index}]",
                            field_policy=_field_policy,
                        )
                        for index, part in enumerate(parts)
                    ],
                    "retired_part_ids": list(project.retired_part_ids),
                    "retired_part_boolean_feature_ids": list(
                        project.retired_part_boolean_feature_ids
                    ),
                    "named_regions": [
                        _encode_named_region(
                            region,
                            f"snapshot.named_regions[{index}]",
                        )
                        for index, region in enumerate(regions)
                    ],
                    "definitions": {
                        "materials": _encode_array(
                            normalized.materials,
                            "snapshot.material_definitions",
                            encode_material_field,
                            field_policy=_field_policy,
                        ),
                        "sections": _encode_array(
                            normalized.sections,
                            "snapshot.section_definitions",
                            encode_section_field,
                            field_policy=_field_policy,
                        ),
                        "assignments": _encode_array(
                            normalized.assignments,
                            "snapshot.region_assignments",
                            encode_assignment_field,
                            field_policy=_field_policy,
                        ),
                        "steps": _encode_array(
                            normalized.steps,
                            "snapshot.analysis_definitions",
                            encode_step_field,
                            field_policy=_field_policy,
                        ),
                    },
                    "part_boolean_undo_records": [
                        _encode_part_boolean_undo_record(
                            record,
                            "snapshot.part_boolean_undo_records"
                            f"[{index}]",
                            field_policy=_field_policy,
                        )
                        for index, record in enumerate(records)
                    ],
                },
            },
        }
        dumps_canonical_json(payload, error_type=ProjectV7EncodeError)
        return payload
    except ProjectV7Error:
        raise
    except Exception as error:
        raise ProjectV7EncodeError(
            f"snapshot 无法由 v7 无损表示：{error}"
        ) from error


def _decode_part(
    value: Any,
    path: str,
    *,
    field_policy: ProjectFieldCodecPolicy = _V7_FIELD_POLICY,
) -> NativePart:
    data = _mapping(value, path)
    _keys(
        data,
        path,
        {
            "id",
            "name",
            "suppressed",
            "geometry",
            "logical_topology",
            "mesh_settings",
            "provenance",
        },
    )
    recipe = decode_geometry_field(
        data["geometry"],
        f"{path}.geometry",
        policy=field_policy,
    )
    if not isinstance(recipe, NATIVE_GEOMETRY_TYPES):
        raise ProjectV7DecodeError(
            f"{path}.geometry 必须是 native recipe"
        )
    stored = _v4._decode_topology_fingerprint_v4(
        data["logical_topology"],
        f"{path}.logical_topology",
    )
    _v4._require_matching_topology_fingerprint_v4(
        stored,
        topology_fingerprint_for_recipe(recipe),
        data["logical_topology"],
        f"{path}.logical_topology",
    )
    part_id = _string(data["id"], f"{path}.id")
    settings = _decode_mesh_settings(
        data["mesh_settings"],
        f"{path}.mesh_settings",
        part_id,
        recipe,
    )
    provenance = _decode_provenance(
        data["provenance"],
        f"{path}.provenance",
    )
    if type(data["suppressed"]) is not bool:
        raise ProjectV7DecodeError(f"{path}.suppressed 必须是 bool")
    return NativePart(
        id=part_id,
        name=_string(data["name"], f"{path}.name"),
        geometry_recipe=recipe,
        mesh_settings=settings,
        suppressed=data["suppressed"],
        provenance=provenance,
    )


def _encode_part(
    part: NativePart,
    path: str,
    *,
    field_policy: ProjectFieldCodecPolicy = _V7_FIELD_POLICY,
) -> dict[str, Any]:
    if type(part) is not NativePart or part.geometry_recipe is None:
        raise ProjectV7EncodeError(f"{path} 必须是完整 NativePart")
    recipe = legacy_sketches_to_strict(part.geometry_recipe)
    # Legacy and strict SketchGeometry intentionally compare by their common
    # authored shape.  Identity, rather than equality, is therefore required
    # here so the encoded geometry and its topology fingerprint are derived
    # from the same canonical recipe.
    if recipe is not part.geometry_recipe:
        part = replace(part, geometry_recipe=recipe)
    return {
        "id": part.id,
        "name": part.name,
        "suppressed": part.suppressed,
        "geometry": encode_geometry_field(
            part.geometry_recipe,
            f"{path}.geometry",
            set(),
            policy=field_policy,
        ),
        "logical_topology": _v4._encode_topology_fingerprint_v4(
            topology_fingerprint_for_recipe(part.geometry_recipe)
        ),
        "mesh_settings": _encode_mesh_settings(
            part.mesh_settings,
            f"{path}.mesh_settings",
        ),
        "provenance": _encode_provenance(
            part.provenance,
            f"{path}.provenance",
        ),
    }


def _decode_mesh_settings(
    value: Any,
    path: str,
    part_id: str,
    recipe: Any,
) -> MeshSettings | None:
    if value is None:
        return None
    data = _mapping(value, path)
    _keys(
        data,
        path,
        {"size", "order", "cell_shape", "local_controls", "line_element_type"},
    )
    controls: list[LocalMeshControl] = []
    for index, raw in enumerate(
        _array(data["local_controls"], f"{path}.local_controls")
    ):
        item_path = f"{path}.local_controls[{index}]"
        item = _mapping(raw, item_path)
        _keys(item, item_path, {"target", "size", "falloff"})
        target = LogicalEntityRef(
            _string(item["target"], f"{item_path}.target")
        )
        if part_id_from_logical_id(target.logical_id) != part_id:
            raise ProjectV7DecodeError(
                f"{item_path}.target 不属于 {part_id}"
            )
        local_id = strip_part_logical_id(part_id, target.logical_id)
        try:
            describe_recipe_topology(recipe).entity(local_id)
        except KeyError as error:
            raise ProjectV7DecodeError(
                f"{item_path}.target 引用了未知几何实体"
            ) from error
        falloff_data = _mapping(item["falloff"], f"{item_path}.falloff")
        _keys(
            falloff_data,
            f"{item_path}.falloff",
            {"reference", "start_factor", "end_factor"},
        )
        controls.append(
            LocalMeshControl(
                target,
                _number(item["size"], f"{item_path}.size"),
                MeshSizeFalloff(
                    _string(
                        falloff_data["reference"],
                        f"{item_path}.falloff.reference",
                    ),
                    _number(
                        falloff_data["start_factor"],
                        f"{item_path}.falloff.start_factor",
                    ),
                    _number(
                        falloff_data["end_factor"],
                        f"{item_path}.falloff.end_factor",
                    ),
                ),
            )
        )
    line_type = data["line_element_type"]
    if line_type is not None:
        line_type = _string(line_type, f"{path}.line_element_type")
    return MeshSettings(
        _number(data["size"], f"{path}.size"),
        order=_integer(data["order"], f"{path}.order"),
        cell_shape=_string(data["cell_shape"], f"{path}.cell_shape"),
        local_controls=tuple(controls),
        line_element_type=line_type,
    )


def _encode_mesh_settings(
    settings: MeshSettings | None,
    path: str,
) -> dict[str, Any] | None:
    if settings is None:
        return None
    if type(settings) is not MeshSettings:
        raise ProjectV7EncodeError(f"{path} 必须是 MeshSettings 或 null")
    return {
        "size": settings.size,
        "order": settings.order,
        "cell_shape": settings.cell_shape,
        "local_controls": [
            {
                "target": control.target.logical_id,
                "size": control.size,
                "falloff": {
                    "reference": control.falloff.reference,
                    "start_factor": control.falloff.start_factor,
                    "end_factor": control.falloff.end_factor,
                },
            }
            for control in settings.local_controls
        ],
        "line_element_type": settings.line_element_type,
    }


def _decode_provenance(
    value: Any,
    path: str,
) -> PartBooleanProvenance | None:
    if value is None:
        return None
    data = _mapping(value, path)
    _keys(
        data,
        path,
        {"feature_id", "target_part_id", "tool_part_id", "operation"},
    )
    return PartBooleanProvenance(
        *(
            _string(data[name], f"{path}.{name}")
            for name in (
                "feature_id",
                "target_part_id",
                "tool_part_id",
                "operation",
            )
        )
    )


def _encode_provenance(
    value: PartBooleanProvenance | None,
    path: str,
) -> dict[str, Any] | None:
    if value is None:
        return None
    if type(value) is not PartBooleanProvenance:
        raise ProjectV7EncodeError(
            f"{path} 必须是 PartBooleanProvenance 或 null"
        )
    return {
        "feature_id": value.feature_id,
        "target_part_id": value.target_part_id,
        "tool_part_id": value.tool_part_id,
        "operation": value.operation,
    }


def _decode_named_region(value: Any, path: str) -> NamedRegion:
    data = _mapping(value, path)
    _keys(data, path, {"name", "references"})
    references = []
    for index, raw in enumerate(
        _array(data["references"], f"{path}.references")
    ):
        item_path = f"{path}.references[{index}]"
        if isinstance(raw, Mapping):
            references.append(
                _decode_mesh_entity_reference(raw, item_path)
            )
        else:
            references.append(
                LogicalEntityRef(_string(raw, item_path))
            )
    return NamedRegion(
        _string(data["name"], f"{path}.name"),
        tuple(references),
    )


def _encode_named_region(value: NamedRegion, path: str) -> dict[str, Any]:
    if type(value) is not NamedRegion:
        raise ProjectV7EncodeError(f"{path} 必须是 NamedRegion")
    references = tuple(value.references)
    if all(type(reference) is MeshEntityRef for reference in references):
        canonical = tuple(
            sorted(references, key=mesh_entity_ref_sort_key)
        )
        encoded = [
            _encode_mesh_entity_reference(
                reference,
                f"{path}.references[{index}]",
            )
            for index, reference in enumerate(references)
        ]
    elif all(
        type(reference) is LogicalEntityRef for reference in references
    ):
        canonical = tuple(
            sorted(
                references,
                key=lambda reference: reference.logical_id,
            )
        )
        encoded = [reference.logical_id for reference in references]
    else:
        raise ProjectV7EncodeError(
            f"{path}.references 混用了网格与逻辑引用"
        )
    if references != canonical:
        raise ProjectV7EncodeError(
            f"{path}.references 不是 canonical 顺序"
        )
    return {"name": value.name, "references": encoded}


def _decode_mesh_entity_reference(
    value: Mapping[str, Any],
    path: str,
) -> MeshEntityRef:
    data = _mapping(value, path)
    kind = _string(data.get("kind"), f"{path}.kind")
    part_id = data.get("part_id")
    owner = (
        None
        if part_id is None
        else _string(part_id, f"{path}.part_id")
    )
    if kind == "node":
        _keys(data, path, {"kind", "node_id"}, {"part_id"})
        return MeshEntityRef.node(
            _integer(data["node_id"], f"{path}.node_id"),
            part_id=owner,
        )
    if kind == "element":
        _keys(data, path, {"kind", "element_id"}, {"part_id"})
        return MeshEntityRef.element(
            _integer(data["element_id"], f"{path}.element_id"),
            part_id=owner,
        )
    if kind not in {"edge", "face"}:
        raise ProjectV7DecodeError(
            f"{path}.kind 不支持网格实体类型 {kind!r}"
        )
    _keys(
        data,
        path,
        {"kind", "element_id", "local_index", "node_ids"},
        {"part_id"},
    )
    node_ids = tuple(
        _integer(item, f"{path}.node_ids[{index}]")
        for index, item in enumerate(
            _array(data["node_ids"], f"{path}.node_ids")
        )
    )
    factory = (
        MeshEntityRef.edge if kind == "edge" else MeshEntityRef.face
    )
    return factory(
        _integer(data["element_id"], f"{path}.element_id"),
        _integer(data["local_index"], f"{path}.local_index"),
        node_ids,
        part_id=owner,
    )


def _encode_mesh_entity_reference(
    reference: MeshEntityRef,
    path: str,
) -> dict[str, Any]:
    if type(reference) is not MeshEntityRef:
        raise ProjectV7EncodeError(f"{path} 必须是 MeshEntityRef")
    payload: dict[str, Any] = {"kind": reference.kind}
    if reference.part_id is not None:
        payload["part_id"] = reference.part_id
    if reference.kind == "node":
        payload["node_id"] = int(reference.node_id)
    elif reference.kind == "element":
        payload["element_id"] = int(reference.element_id)
    else:
        payload.update(
            {
                "element_id": int(reference.element_id),
                "local_index": int(reference.local_index),
                "node_ids": [
                    int(node_id) for node_id in reference.node_ids
                ],
            }
        )
    return payload


def _decode_part_boolean_undo_record(
    value: Any,
    path: str,
    *,
    field_policy: ProjectFieldCodecPolicy = _V7_FIELD_POLICY,
) -> PartBooleanUndoRecord:
    data = _mapping(value, path)
    _keys(
        data,
        path,
        {
            "feature_id",
            "result_part_id",
            "source_parts",
            "result_part",
            "before_named_regions",
            "after_named_regions",
            "before_assignments",
            "after_assignments",
            "before_steps",
            "after_steps",
        },
    )
    source_parts = tuple(
        _decode_part(
            item,
            f"{path}.source_parts[{index}]",
            field_policy=field_policy,
        )
        for index, item in enumerate(
            _array(data["source_parts"], f"{path}.source_parts")
        )
    )
    return PartBooleanUndoRecord(
        _string(data["feature_id"], f"{path}.feature_id"),
        _string(data["result_part_id"], f"{path}.result_part_id"),
        source_parts,
        _decode_part(
            data["result_part"],
            f"{path}.result_part",
            field_policy=field_policy,
        ),
        tuple(
            _decode_named_region(
                item,
                f"{path}.before_named_regions[{index}]",
            )
            for index, item in enumerate(
                _array(
                    data["before_named_regions"],
                    f"{path}.before_named_regions",
                )
            )
        ),
        tuple(
            _decode_named_region(
                item,
                f"{path}.after_named_regions[{index}]",
            )
            for index, item in enumerate(
                _array(
                    data["after_named_regions"],
                    f"{path}.after_named_regions",
                )
            )
        ),
        _decode_array(
            data["before_assignments"],
            f"{path}.before_assignments",
            decode_assignment_field,
            field_policy=field_policy,
        ),
        _decode_array(
            data["after_assignments"],
            f"{path}.after_assignments",
            decode_assignment_field,
            field_policy=field_policy,
        ),
        _decode_array(
            data["before_steps"],
            f"{path}.before_steps",
            decode_step_field,
            field_policy=field_policy,
        ),
        _decode_array(
            data["after_steps"],
            f"{path}.after_steps",
            decode_step_field,
            field_policy=field_policy,
        ),
    )


def _encode_part_boolean_undo_record(
    value: PartBooleanUndoRecord,
    path: str,
    *,
    field_policy: ProjectFieldCodecPolicy = _V7_FIELD_POLICY,
) -> dict[str, Any]:
    if type(value) is not PartBooleanUndoRecord:
        raise ProjectV7EncodeError(
            f"{path} 必须是 PartBooleanUndoRecord"
        )
    return {
        "feature_id": value.feature_id,
        "result_part_id": value.result_part_id,
        "source_parts": [
            _encode_part(
                part,
                f"{path}.source_parts[{index}]",
                field_policy=field_policy,
            )
            for index, part in enumerate(value.source_parts)
        ],
        "result_part": _encode_part(
            value.result_part,
            f"{path}.result_part",
            field_policy=field_policy,
        ),
        "before_named_regions": [
            _encode_named_region(
                region,
                f"{path}.before_named_regions[{index}]",
            )
            for index, region in enumerate(value.before_named_regions)
        ],
        "after_named_regions": [
            _encode_named_region(
                region,
                f"{path}.after_named_regions[{index}]",
            )
            for index, region in enumerate(value.after_named_regions)
        ],
        "before_assignments": _encode_array(
            value.before_assignments,
            f"{path}.before_assignments",
            encode_assignment_field,
            field_policy=field_policy,
        ),
        "after_assignments": _encode_array(
            value.after_assignments,
            f"{path}.after_assignments",
            encode_assignment_field,
            field_policy=field_policy,
        ),
        "before_steps": _encode_array(
            value.before_steps,
            f"{path}.before_steps",
            encode_step_field,
            field_policy=field_policy,
        ),
        "after_steps": _encode_array(
            value.after_steps,
            f"{path}.after_steps",
            encode_step_field,
            field_policy=field_policy,
        ),
    }


def _validate_references(
    parts: tuple[NativePart, ...],
    regions: tuple[NamedRegion, ...],
) -> None:
    by_id = {part.id: part for part in parts}
    topologies = {
        part.id: describe_recipe_topology(part.geometry_recipe)
        for part in parts
    }
    for region in regions:
        for reference in region.references:
            if type(reference) is MeshEntityRef:
                if reference.part_id is None and len(parts) != 1:
                    raise ValueError(
                        f"region {region.name!r} mesh reference is missing "
                        "its Part owner"
                    )
                if reference.part_id is not None and reference.part_id not in by_id:
                    raise ValueError(
                        f"region {region.name!r} mesh reference has no "
                        f"active Part: {reference.part_id!r}"
                    )
                continue
            if type(reference) is not LogicalEntityRef:
                continue
            owner = part_id_from_logical_id(reference.logical_id)
            if owner not in by_id:
                raise ValueError(
                    f"region {region.name!r} reference has no active Part: "
                    f"{reference.logical_id!r}"
                )
            topologies[owner].entity(
                strip_part_logical_id(owner, reference.logical_id)
            )


def _validate_part_graph(
    parts: tuple[NativePart, ...],
    records: tuple[PartBooleanUndoRecord, ...],
) -> None:
    by_id = {part.id: part for part in parts}
    by_feature = {record.feature_id: record for record in records}
    if len(by_feature) != len(records):
        raise ValueError("Part Boolean undo feature IDs must be unique")
    if records != tuple(
        sorted(
            records,
            key=lambda item: part_boolean_feature_id_sort_key(
                item.feature_id
            ),
        )
    ):
        raise ValueError("Part Boolean undo records are not canonical")
    for part in parts:
        provenance = part.provenance
        if provenance is None:
            continue
        if part.id not in by_id:
            raise ValueError("Boolean result Part is missing")
        if any(source not in by_id for source in provenance.source_part_ids):
            raise ValueError("Boolean source Part is missing")
        if any(not by_id[source].suppressed for source in provenance.source_part_ids):
            raise ValueError("Boolean source Parts must be suppressed")
        record = by_feature.get(provenance.feature_id)
        if record is None or record.result_part_id != part.id:
            raise ValueError("Boolean result lacks its exact undo record")
        if (
            provenance != record.result_part.provenance
            or not _recipe_contains_state(
                part.geometry_recipe,
                record.result_part.geometry_recipe,
            )
        ):
            raise ValueError(
                "Boolean live result is not derived from its exact undo state"
            )
        for expected_source in record.source_parts:
            live_source = by_id[expected_source.id]
            if (
                replace(
                    live_source,
                    suppressed=expected_source.suppressed,
                )
                != expected_source
            ):
                raise ValueError(
                    "Boolean live source does not match its exact undo state"
                )


def _recipe_contains_state(current: Any, expected: Any) -> bool:
    if current == expected:
        return True
    if isinstance(current, (MovedGeometry, RotatedGeometry, ExtrudedGeometry)):
        return _recipe_contains_state(current.base, expected)
    return False


def _authenticate_part(part: NativePart, *, encode: bool) -> None:
    strict = _contains_strict_boolean(part.geometry_recipe)
    if not strict:
        return
    try:
        with geometry_model(
            f"project-v7-{part.id}-proof-authentication",
            dimension=geometry_dimension(part.geometry_recipe),
        ) as cad:
            compile_recipe(cad, part.geometry_recipe)
    except Exception as error:
        error_type = ProjectV7EncodeError if encode else ProjectV7DecodeError
        prefix = (
            f"snapshot.parts[{part.id}]"
            if encode
            else f"$.project.authoring.parts[{part.id}]"
        )
        raise error_type(
            f"{prefix} OCC proof authentication failed: {error}"
        ) from error


def _contains_strict_boolean(recipe: object) -> bool:
    if isinstance(recipe, FaceSketchBooleanGeometry):
        return True
    if isinstance(recipe, BooleanGeometry):
        return (
            recipe.body_context is not None
            or recipe.planar_context is not None
            or recipe.part_context is not None
            or _contains_strict_boolean(recipe.object_geometry)
            or _contains_strict_boolean(recipe.tool_geometry)
        )
    base = getattr(recipe, "base", None)
    return base is not None and _contains_strict_boolean(base)


def _decode_array(
    value: Any,
    path: str,
    decoder,
    *,
    field_policy: ProjectFieldCodecPolicy = _V7_FIELD_POLICY,
) -> tuple[Any, ...]:
    return tuple(
        decoder(item, f"{path}[{index}]", policy=field_policy)
        for index, item in enumerate(_array(value, path))
    )


def _encode_array(
    values: tuple[Any, ...],
    path: str,
    encoder,
    *,
    field_policy: ProjectFieldCodecPolicy = _V7_FIELD_POLICY,
) -> list[Any]:
    return [
        encoder(item, f"{path}[{index}]", policy=field_policy)
        for index, item in enumerate(values)
    ]


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProjectV7DecodeError(f"{path} 必须是 object")
    return value


def _array(value: Any, path: str) -> tuple[Any, ...]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, list):
        raise ProjectV7DecodeError(f"{path} 必须是 array")
    return tuple(value)


def _keys(
    value: Mapping[str, Any],
    path: str,
    required: set[str],
    optional: set[str] | None = None,
) -> None:
    actual = set(value)
    missing = required - actual
    extra = actual - required - (set() if optional is None else optional)
    if missing:
        raise ProjectV7DecodeError(
            f"{path} 缺少字段 {sorted(missing)[0]!r}"
        )
    if extra:
        raise ProjectV7DecodeError(
            f"{path} 包含未知字段 {sorted(extra)[0]!r}"
        )


def _string(value: Any, path: str) -> str:
    if type(value) is not str or not value.strip():
        raise ProjectV7DecodeError(f"{path} 必须是非空 string")
    if value != value.strip():
        raise ProjectV7DecodeError(f"{path} 不能包含首尾空白")
    return value


def _integer(value: Any, path: str) -> int:
    if type(value) is not int:
        raise ProjectV7DecodeError(f"{path} 必须是 integer")
    return value


def _number(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProjectV7DecodeError(f"{path} 必须是 number")
    return float(value)


def _require_unique_names(values: tuple[Any, ...], path: str) -> None:
    names = tuple(str(value.name) for value in values)
    if len(names) != len(set(names)):
        raise ProjectV7DecodeError(f"{path} 名称必须唯一")


def dumps_project_v7(
    snapshot: ProjectSnapshot | ProjectSaveSnapshot,
) -> str:
    return dumps_canonical_json(
        encode_project_v7(snapshot),
        error_type=ProjectV7EncodeError,
    )


def load_project_v7(path: str | Path) -> ProjectSnapshot:
    source = Path(path)
    return loads_project_v7(source.read_bytes(), source_path=source)


def save_project_v7(
    path: str | Path,
    snapshot: ProjectSnapshot | ProjectSaveSnapshot,
    *,
    checkpoint: Callable[[], Any] | None = None,
) -> Path:
    if checkpoint is not None:
        checkpoint()
    payload = encode_project_v7(snapshot)
    if checkpoint is not None:
        checkpoint()
    serialized = dumps_canonical_json(
        payload,
        error_type=ProjectV7EncodeError,
    )
    return atomic_write_project(
        path,
        serialized,
        verifier=load_project_v7,
        semantic_encoder=encode_project_v7,
        expected_semantic=payload,
        error_type=ProjectV7EncodeError,
        mismatch_message=(
            "临时 v7 项目回读后的 canonical Part authoring 与 snapshot 不一致"
        ),
        checkpoint=checkpoint,
    )


read_project_v7 = load_project_v7
write_project_v7 = save_project_v7


__all__ = [
    "FORMAT_NAME",
    "ProjectV7DecodeError",
    "ProjectV7EncodeError",
    "ProjectV7Error",
    "SCHEMA_VERSION",
    "decode_project_v7",
    "dumps_project_v7",
    "encode_project_v7",
    "load_project_v7",
    "loads_project_v7",
    "read_project_v7",
    "save_project_v7",
    "write_project_v7",
]
