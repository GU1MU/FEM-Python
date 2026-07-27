"""Strict native-authoring ``.femproj`` schema v3 codec.

Schema v3 keeps the v2 envelope and definition field codecs, while widening
only the geometry and mesh-intent contracts required by native 1D authoring.
The v2 module remains the frozen compatibility codec.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from fem.application.definitions import (
    NamedRegion,
    RegionAssignment,
    normalize_model_definitions,
)
from fem.application.feature_history import derive_feature_history
from fem.application.project_validation import (
    NativeProjectValidationError,
    validate_native_project_inputs,
)
from fem.application.session import ProjectSaveSnapshot, ProjectSnapshot
from fem.geometry.recipe_topology import (
    TOPOLOGY_REFERENCE_CONTRACT,
    TopologyFingerprint,
    TopologyFingerprintEntity,
    topology_fingerprint_for_recipe,
)
from fem.geometry.recipes import NATIVE_GEOMETRY_TYPES
from fem.geometry.references import LogicalEntityRef, logical_ref_sort_key
from fem.mesh.settings import MeshSettings

from . import project_v2 as _v2
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


FORMAT_NAME = "fem-python-project"
SCHEMA_VERSION = 3


class ProjectV3Error(ProjectError):
    """Base error for schema-v3 project processing."""


class ProjectV3DecodeError(ProjectV3Error, ProjectDecodeError):
    """A schema-v3 payload is malformed or incompatible."""


class ProjectV3EncodeError(ProjectV3Error, ProjectEncodeError):
    """A snapshot cannot be represented losslessly by schema v3."""


_V3_FIELD_POLICY = ProjectFieldCodecPolicy(
    version_label="v3",
    decode_error=ProjectV3DecodeError,
    encode_error=ProjectV3EncodeError,
    require_current_fields=True,
    assignment_orientation=True,
    allow_wire_geometry=True,
)


def loads_project_v3(
    data: str | bytes | bytearray,
    *,
    source_path: str | Path | None = None,
) -> ProjectSnapshot:
    """Strictly parse and decode one schema-v3 document."""

    payload = loads_json_strict(
        data,
        error_type=ProjectV3DecodeError,
        document_label="v3 项目",
    )
    return decode_project_v3(payload, source_path=source_path)


def decode_project_v3(
    payload: Mapping[str, Any] | str | bytes | bytearray,
    *,
    source_path: str | Path | None = None,
) -> ProjectSnapshot:
    """Decode one detached schema-v3 native authoring snapshot."""

    if isinstance(payload, (str, bytes, bytearray)):
        return loads_project_v3(payload, source_path=source_path)
    try:
        root = _v2._mapping(payload, "$", ProjectV3DecodeError)
        _v2._keys(
            root,
            "$",
            required={"format", "schema", "project"},
            optional=set(),
            error_type=ProjectV3DecodeError,
        )
        if _v2._string(root["format"], "$.format", ProjectV3DecodeError) != FORMAT_NAME:
            raise ProjectV3DecodeError(
                f"$.format 必须精确等于 {FORMAT_NAME!r}"
            )
        schema = _v2._integer(root["schema"], "$.schema", ProjectV3DecodeError)
        if schema != SCHEMA_VERSION:
            raise ProjectV3DecodeError(f"v3 decoder 不能读取 schema {schema!r}")

        project = _v2._mapping(
            root["project"],
            "$.project",
            ProjectV3DecodeError,
        )
        _v2._keys(
            project,
            "$.project",
            required={"kind", "authoring"},
            optional=set(),
            error_type=ProjectV3DecodeError,
        )
        if _v2._string(
            project["kind"],
            "$.project.kind",
            ProjectV3DecodeError,
        ) != "native":
            raise ProjectV3DecodeError("$.project.kind 本阶段只接受 'native'")
        authoring = _v2._mapping(
            project["authoring"],
            "$.project.authoring",
            ProjectV3DecodeError,
        )
        _v2._keys(
            authoring,
            "$.project.authoring",
            required={
                "part",
                "geometry",
                "logical_topology",
                "mesh_settings",
                "named_regions",
                "definitions",
            },
            optional=set(),
            error_type=ProjectV3DecodeError,
        )

        part = _v2._decode_part(authoring["part"], "$.project.authoring.part")
        geometry = decode_geometry_field(
            authoring["geometry"],
            "$.project.authoring.geometry",
            policy=_V3_FIELD_POLICY,
        )
        if not isinstance(geometry, NATIVE_GEOMETRY_TYPES):
            raise ProjectV3DecodeError(
                "$.project.authoring.geometry 不是有效的 native geometry recipe"
            )
        stored_fingerprint = _decode_topology_fingerprint_v3(
            authoring["logical_topology"],
            "$.project.authoring.logical_topology",
        )
        recomputed_fingerprint = topology_fingerprint_for_recipe(geometry)
        _require_matching_topology_fingerprint_v3(
            stored_fingerprint,
            recomputed_fingerprint,
            authoring["logical_topology"],
            "$.project.authoring.logical_topology",
        )
        mesh_settings = _decode_mesh_settings_v3(
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
                    ProjectV3DecodeError,
                )
            )
        )
        _v2._unique_names(
            named_regions,
            "$.project.authoring.named_regions",
            ProjectV3DecodeError,
        )

        definitions = _v2._mapping(
            authoring["definitions"],
            "$.project.authoring.definitions",
            ProjectV3DecodeError,
        )
        _v2._keys(
            definitions,
            "$.project.authoring.definitions",
            required={"materials", "sections", "assignments", "steps"},
            optional=set(),
            error_type=ProjectV3DecodeError,
        )
        materials = tuple(
            decode_material_field(
                item,
                f"$.project.authoring.definitions.materials[{index}]",
                policy=_V3_FIELD_POLICY,
            )
            for index, item in enumerate(
                _v2._array(
                    definitions["materials"],
                    "$.project.authoring.definitions.materials",
                    ProjectV3DecodeError,
                )
            )
        )
        sections = tuple(
            decode_section_field(
                item,
                f"$.project.authoring.definitions.sections[{index}]",
                policy=_V3_FIELD_POLICY,
            )
            for index, item in enumerate(
                _v2._array(
                    definitions["sections"],
                    "$.project.authoring.definitions.sections",
                    ProjectV3DecodeError,
                )
            )
        )
        assignments = tuple(
            decode_assignment_field(
                item,
                f"$.project.authoring.definitions.assignments[{index}]",
                policy=_V3_FIELD_POLICY,
            )
            for index, item in enumerate(
                _v2._array(
                    definitions["assignments"],
                    "$.project.authoring.definitions.assignments",
                    ProjectV3DecodeError,
                )
            )
        )
        steps = tuple(
            decode_step_field(
                item,
                f"$.project.authoring.definitions.steps[{index}]",
                policy=_V3_FIELD_POLICY,
            )
            for index, item in enumerate(
                _v2._array(
                    definitions["steps"],
                    "$.project.authoring.definitions.steps",
                    ProjectV3DecodeError,
                )
            )
        )
        for values, path in (
            (materials, "$.project.authoring.definitions.materials"),
            (sections, "$.project.authoring.definitions.sections"),
            (steps, "$.project.authoring.definitions.steps"),
        ):
            _v2._unique_names(values, path, ProjectV3DecodeError)

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
        materials = normalized.materials
        sections = normalized.sections
        assignments = normalized.assignments
        steps = normalized.steps
        _validate_native_project_inputs_v3(
            geometry,
            mesh_settings,
            named_regions,
            materials,
            sections,
            assignments,
            steps,
            encode=False,
        )
        return ProjectSnapshot(
            source_kind="native",
            source_path=None if source_path is None else Path(source_path),
            parts=(part,),
            geometry_recipe=geometry,
            mesh_settings=mesh_settings,
            feature_history=derive_feature_history(geometry),
            named_regions=named_regions,
            material_definitions=materials,
            section_definitions=sections,
            region_assignments=assignments,
            analysis_definitions=steps,
            model=None,
        )
    except ProjectV3Error:
        raise
    except _v2.ProjectV2Error as error:
        raise ProjectV3DecodeError(str(error)) from error
    except (KeyError, TypeError, ValueError) as error:
        raise ProjectV3DecodeError(
            f"$.project.authoring 无效：{error}"
        ) from error


def load_project_v3(path: str | Path) -> ProjectSnapshot:
    """Read one schema-v3 project without touching a live Session."""

    source = Path(path)
    return loads_project_v3(source.read_bytes(), source_path=source)


def encode_project_v3(
    snapshot: ProjectSnapshot | ProjectSaveSnapshot,
) -> dict[str, Any]:
    """Encode native authoring inputs as a canonical v3 mapping."""

    try:
        project = unwrap_project_snapshot(
            snapshot,
            error_type=ProjectV3EncodeError,
        )
        if project.source_kind != "native":
            raise ProjectV3EncodeError("snapshot.source_kind 本阶段只接受 'native'")
        if project.model is not None:
            raise ProjectV3EncodeError("v3 不持久化 compiled/runtime model")
        geometry = project.geometry_recipe
        if geometry is None:
            raise ProjectV3EncodeError("snapshot.geometry_recipe 不能为空")
        if not isinstance(geometry, NATIVE_GEOMETRY_TYPES):
            raise ProjectV3EncodeError("snapshot.geometry_recipe 不是 native geometry")
        parts = tuple(project.parts)
        if len(parts) != 1:
            raise ProjectV3EncodeError("v3 native 项目必须且只能包含一个 part")
        expected_history = derive_feature_history(geometry)
        if tuple(project.feature_history) != tuple(expected_history):
            raise ProjectV3EncodeError(
                "snapshot.feature_history 不是 geometry recipe 的 canonical 派生投影"
            )

        named_regions = tuple(project.named_regions)
        materials = tuple(project.material_definitions)
        sections = tuple(project.section_definitions)
        assignments = tuple(project.region_assignments)
        steps = tuple(project.analysis_definitions)
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
            raise ProjectV3EncodeError(
                "snapshot definitions 不是 normalize_model_definitions 产生的 canonical 形式"
            )
        _validate_native_project_inputs_v3(
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
                    "part": _v2._encode_part(parts[0], "snapshot.parts[0]"),
                    "geometry": encode_geometry_field(
                        geometry,
                        "snapshot.geometry_recipe",
                        set(),
                        policy=_V3_FIELD_POLICY,
                    ),
                    "logical_topology": _encode_topology_fingerprint_v3(
                        topology_fingerprint_for_recipe(geometry)
                    ),
                    "mesh_settings": _encode_mesh_settings_v3(
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
                    "definitions": {
                        "materials": [
                            encode_material_field(
                                material,
                                f"snapshot.material_definitions[{index}]",
                                policy=_V3_FIELD_POLICY,
                            )
                            for index, material in enumerate(materials)
                        ],
                        "sections": [
                            encode_section_field(
                                section,
                                f"snapshot.section_definitions[{index}]",
                                policy=_V3_FIELD_POLICY,
                            )
                            for index, section in enumerate(sections)
                        ],
                        "assignments": [
                            encode_assignment_field(
                                assignment,
                                f"snapshot.region_assignments[{index}]",
                                policy=_V3_FIELD_POLICY,
                            )
                            for index, assignment in enumerate(assignments)
                        ],
                        "steps": [
                            encode_step_field(
                                step,
                                f"snapshot.analysis_definitions[{index}]",
                                policy=_V3_FIELD_POLICY,
                            )
                            for index, step in enumerate(steps)
                        ],
                    },
                },
            },
        }
        dumps_canonical_json(payload, error_type=ProjectV3EncodeError)
        return payload
    except ProjectV3Error:
        raise
    except _v2.ProjectV2Error as error:
        raise ProjectV3EncodeError(str(error)) from error
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise ProjectV3EncodeError(
            f"snapshot 无法由 v3 无损表示：{error}"
        ) from error


def dumps_project_v3(snapshot: ProjectSnapshot | ProjectSaveSnapshot) -> str:
    """Return deterministic UTF-8 schema-v3 JSON with one trailing LF."""

    return dumps_canonical_json(
        encode_project_v3(snapshot),
        error_type=ProjectV3EncodeError,
    )


def save_project_v3(
    path: str | Path,
    snapshot: ProjectSnapshot | ProjectSaveSnapshot,
) -> Path:
    """Atomically write and semantically read back one v3 project."""

    payload = encode_project_v3(snapshot)
    serialized = dumps_canonical_json(payload, error_type=ProjectV3EncodeError)
    return atomic_write_project(
        path,
        serialized,
        verifier=load_project_v3,
        semantic_encoder=encode_project_v3,
        expected_semantic=payload,
        error_type=ProjectV3EncodeError,
        mismatch_message=(
            "临时 v3 项目回读后的 canonical authoring 值与保存 snapshot 不一致"
        ),
    )


read_project_v3 = load_project_v3
write_project_v3 = save_project_v3


def _validate_native_project_inputs_v3(
    recipe: Any,
    mesh_settings: MeshSettings | None,
    named_regions: tuple[NamedRegion, ...],
    materials: tuple[Any, ...],
    sections: tuple[Any, ...],
    assignments: tuple[RegionAssignment, ...],
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
        message = str(error)
        if message.startswith("assignments["):
            prefix = (
                "snapshot.region_assignments"
                if encode
                else "$.project.authoring.definitions.assignments"
            )
            message = prefix + message[len("assignments") :]
        elif message.startswith("steps["):
            prefix = (
                "snapshot.analysis_definitions"
                if encode
                else "$.project.authoring.definitions.steps"
            )
            message = prefix + message[len("steps") :]
        elif encode:
            message = f"snapshot 无法由 v3 无损表示：{message}"
        else:
            message = f"$.project.authoring 无效：{message}"
        error_type = ProjectV3EncodeError if encode else ProjectV3DecodeError
        raise error_type(message) from error


def _decode_mesh_settings_v3(
    value: Any,
    path: str,
    recipe: Any,
) -> MeshSettings | None:
    if value is None:
        return None
    data = _v2._mapping(value, path, ProjectV3DecodeError)
    _v2._keys(
        data,
        path,
        required={"size", "order", "cell_shape", "local_controls", "line_element_type"},
        optional=set(),
        error_type=ProjectV3DecodeError,
    )
    controls = tuple(
        _v2._decode_local_control(
            item,
            f"{path}.local_controls[{index}]",
            recipe,
        )
        for index, item in enumerate(
            _v2._array(
                data["local_controls"],
                f"{path}.local_controls",
                ProjectV3DecodeError,
            )
        )
    )
    raw_line_type = data["line_element_type"]
    line_type = (
        None
        if raw_line_type is None
        else _v2._string(
            raw_line_type,
            f"{path}.line_element_type",
            ProjectV3DecodeError,
        )
    )
    try:
        return MeshSettings(
            _v2._number(data["size"], f"{path}.size", ProjectV3DecodeError),
            order=_v2._integer(
                data["order"],
                f"{path}.order",
                ProjectV3DecodeError,
            ),
            cell_shape=_v2._string(
                data["cell_shape"],
                f"{path}.cell_shape",
                ProjectV3DecodeError,
            ),
            local_controls=controls,
            line_element_type=line_type,
        )
    except (TypeError, ValueError) as error:
        raise ProjectV3DecodeError(f"{path} 无效：{error}") from error


def _encode_mesh_settings_v3(
    settings: Any,
    path: str,
    recipe: Any,
) -> dict[str, Any] | None:
    if settings is None:
        return None
    if type(settings) is not MeshSettings:
        raise ProjectV3EncodeError(f"{path} 必须是 MeshSettings 或 null")
    return {
        "size": _v2._number(settings.size, f"{path}.size", ProjectV3EncodeError),
        "order": _v2._integer(settings.order, f"{path}.order", ProjectV3EncodeError),
        "cell_shape": _v2._string(
            settings.cell_shape,
            f"{path}.cell_shape",
            ProjectV3EncodeError,
        ),
        "local_controls": [
            _v2._encode_local_control(
                control,
                f"{path}.local_controls[{index}]",
            )
            for index, control in enumerate(settings.local_controls)
        ],
        "line_element_type": settings.line_element_type,
    }


def _encode_topology_fingerprint_v3(
    fingerprint: TopologyFingerprint,
) -> dict[str, Any]:
    return {
        "contract": fingerprint.contract,
        "signature": {
            "dimension": fingerprint.dimension,
            "exact": fingerprint.exact,
            "entities": [
                {
                    "kind": entity.kind,
                    "logical_id": entity.logical_id,
                    "semantic_role": entity.semantic_role,
                    "selectable": entity.selectable,
                    "topology_links": list(entity.topology_links),
                }
                for entity in fingerprint.entities
            ],
        },
    }


def _decode_topology_fingerprint_v3(
    value: Any,
    path: str,
) -> TopologyFingerprint:
    data = _v2._mapping(value, path, ProjectV3DecodeError)
    _v2._keys(
        data,
        path,
        required={"contract", "signature"},
        optional=set(),
        error_type=ProjectV3DecodeError,
    )
    contract = _v2._integer(
        data["contract"],
        f"{path}.contract",
        ProjectV3DecodeError,
    )
    if contract != TOPOLOGY_REFERENCE_CONTRACT:
        raise ProjectV3DecodeError(f"{path}.contract 不支持：{contract!r}")
    signature = _v2._mapping(
        data["signature"],
        f"{path}.signature",
        ProjectV3DecodeError,
    )
    _v2._keys(
        signature,
        f"{path}.signature",
        required={"dimension", "exact", "entities"},
        optional=set(),
        error_type=ProjectV3DecodeError,
    )
    exact = signature["exact"]
    if type(exact) is not bool:
        raise ProjectV3DecodeError(f"{path}.signature.exact 必须是 boolean")
    records: list[TopologyFingerprintEntity] = []
    for index, item in enumerate(
        _v2._array(
            signature["entities"],
            f"{path}.signature.entities",
            ProjectV3DecodeError,
        )
    ):
        record_path = f"{path}.signature.entities[{index}]"
        record = _v2._mapping(item, record_path, ProjectV3DecodeError)
        _v2._keys(
            record,
            record_path,
            required={
                "kind",
                "logical_id",
                "semantic_role",
                "selectable",
                "topology_links",
            },
            optional=set(),
            error_type=ProjectV3DecodeError,
        )
        selectable = record["selectable"]
        if type(selectable) is not bool:
            raise ProjectV3DecodeError(f"{record_path}.selectable 必须是 boolean")
        links = tuple(
            _v2._string(
                link,
                f"{record_path}.topology_links[{link_index}]",
                ProjectV3DecodeError,
            )
            for link_index, link in enumerate(
                _v2._array(
                    record["topology_links"],
                    f"{record_path}.topology_links",
                    ProjectV3DecodeError,
                )
            )
        )
        logical_id = _v2._string(
            record["logical_id"],
            f"{record_path}.logical_id",
            ProjectV3DecodeError,
        )
        if any(item.logical_id == logical_id for item in records):
            raise ProjectV3DecodeError(
                f"{record_path}.logical_id duplicates another topology entity"
            )
        try:
            records.append(
                TopologyFingerprintEntity(
                    _v2._string(record["kind"], f"{record_path}.kind", ProjectV3DecodeError),
                    logical_id,
                    _v2._string(
                        record["semantic_role"],
                        f"{record_path}.semantic_role",
                        ProjectV3DecodeError,
                    ),
                    selectable,
                    links,
                )
            )
        except (TypeError, ValueError) as error:
            raise ProjectV3DecodeError(f"{record_path} 无效：{error}") from error
    try:
        return TopologyFingerprint(
            dimension=_v2._integer(
                signature["dimension"],
                f"{path}.signature.dimension",
                ProjectV3DecodeError,
            ),
            exact=exact,
            entities=tuple(records),
            contract=contract,
        )
    except (TypeError, ValueError) as error:
        raise ProjectV3DecodeError(f"{path} 无效：{error}") from error


def _require_matching_topology_fingerprint_v3(
    stored: TopologyFingerprint,
    expected: TopologyFingerprint,
    raw_value: Any,
    path: str,
) -> None:
    signature_path = f"{path}.signature"
    if stored.dimension != expected.dimension:
        raise ProjectV3DecodeError(
            f"{signature_path}.dimension 与 geometry 重新计算值 "
            f"{expected.dimension!r} 不一致（topology fingerprint）"
        )
    if stored.exact != expected.exact:
        raise ProjectV3DecodeError(
            f"{signature_path}.exact 与 geometry 重新计算值 "
            f"{expected.exact!r} 不一致（topology fingerprint）"
        )
    stored_ids = tuple(item.logical_id for item in stored.entities)
    if len(stored_ids) != len(set(stored_ids)):
        raise ProjectV3DecodeError(
            f"{signature_path}.entities contains duplicate logical IDs "
            "(topology fingerprint)"
        )
    stored_by_id = {item.logical_id: item for item in stored.entities}
    expected_by_id = {item.logical_id: item for item in expected.entities}
    raw_signature = _v2._mapping(
        _v2._mapping(raw_value, path, ProjectV3DecodeError)["signature"],
        f"{path}.signature",
        ProjectV3DecodeError,
    )
    raw_entities = _v2._array(
        raw_signature["entities"],
        f"{path}.signature.entities",
        ProjectV3DecodeError,
    )
    raw_indices = {
        item["logical_id"]: index
        for index, item in enumerate(raw_entities)
        if isinstance(item, Mapping) and "logical_id" in item
    }
    missing = sorted(
        set(expected_by_id) - set(stored_by_id),
        key=lambda logical_id: logical_ref_sort_key(LogicalEntityRef(logical_id)),
    )
    if missing:
        raise ProjectV3DecodeError(
            f"{signature_path}.entities 缺少 geometry topology entity "
            f"{missing[0]!r}（topology fingerprint）"
        )
    extra = sorted(
        set(stored_by_id) - set(expected_by_id),
        key=lambda logical_id: logical_ref_sort_key(LogicalEntityRef(logical_id)),
    )
    if extra:
        index = raw_indices.get(extra[0], 0)
        raise ProjectV3DecodeError(
            f"{signature_path}.entities[{index}].logical_id 包含 geometry 中不存在的 "
            f"topology entity {extra[0]!r}（topology fingerprint）"
        )
    for logical_id, expected_entity in expected_by_id.items():
        stored_entity = stored_by_id[logical_id]
        index = raw_indices[logical_id]
        entity_path = f"{signature_path}.entities[{index}]"
        for field_name in ("kind", "semantic_role", "selectable", "topology_links"):
            if getattr(stored_entity, field_name) != getattr(expected_entity, field_name):
                raise ProjectV3DecodeError(
                    f"{entity_path}.{field_name} 与 geometry 重新计算值不一致 "
                    "（topology fingerprint）"
                )


__all__ = [
    "FORMAT_NAME",
    "SCHEMA_VERSION",
    "ProjectV3DecodeError",
    "ProjectV3EncodeError",
    "ProjectV3Error",
    "decode_project_v3",
    "dumps_project_v3",
    "encode_project_v3",
    "load_project_v3",
    "loads_project_v3",
    "read_project_v3",
    "save_project_v3",
    "write_project_v3",
]
