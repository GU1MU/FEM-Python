"""Strict native-authoring ``.femproj`` schema v2 codec."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
import math
from typing import Any

from fem.application.definitions import (
    NamedRegion,
    NativePart,
    RegionAssignment,
    normalize_model_definitions,
)
from fem.application.feature_history import derive_feature_history
from fem.application.native_regions import validate_logical_reference
from fem.application.project_validation import (
    NativeProjectValidationError,
    validate_native_project_inputs,
)
from fem.application.session import ProjectSaveSnapshot, ProjectSnapshot
from fem.elements import BEAM_LOCAL_Y_REFERENCE_KEY
from fem.geometry.recipe_topology import (
    TOPOLOGY_REFERENCE_CONTRACT,
    TopologyFingerprint,
    TopologyFingerprintEntity,
    topology_fingerprint_for_recipe,
)
from fem.geometry.measurements import resolve_target_radius
from fem.geometry.references import (
    EntityKind,
    LogicalEntityRef,
    logical_ref_sort_key,
)
from fem.mesh.settings import LocalMeshControl, MeshSettings, MeshSizeFalloff

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
from ._project_errors import (
    ProjectDecodeError,
    ProjectEncodeError,
    ProjectError,
)


FORMAT_NAME = "fem-python-project"
SCHEMA_VERSION = 2


class ProjectV2Error(ProjectError):
    """Base error for schema-v2 project processing."""


class ProjectV2DecodeError(ProjectV2Error, ProjectDecodeError):
    """A schema-v2 payload is malformed or incompatible."""


class ProjectV2EncodeError(ProjectV2Error, ProjectEncodeError):
    """A snapshot cannot be represented losslessly by schema v2."""


_V2_FIELD_POLICY = ProjectFieldCodecPolicy(
    version_label="v2",
    decode_error=ProjectV2DecodeError,
    encode_error=ProjectV2EncodeError,
    require_current_fields=True,
    assignment_orientation=True,
)


def loads_project_v2(
    data: str | bytes | bytearray,
    *,
    source_path: str | Path | None = None,
) -> ProjectSnapshot:
    """Strictly parse and decode one complete schema-v2 document."""

    payload = loads_json_strict(
        data,
        error_type=ProjectV2DecodeError,
        document_label="v2 项目",
    )
    return decode_project_v2(payload, source_path=source_path)


def decode_project_v2(
    payload: Mapping[str, Any] | str | bytes | bytearray,
    *,
    source_path: str | Path | None = None,
) -> ProjectSnapshot:
    """Decode a detached current authoring snapshot."""

    if isinstance(payload, (str, bytes, bytearray)):
        return loads_project_v2(payload, source_path=source_path)
    root = _mapping(payload, "$", ProjectV2DecodeError)
    _keys(
        root,
        "$",
        required={"format", "schema", "project"},
        optional=set(),
        error_type=ProjectV2DecodeError,
    )
    if _string(root["format"], "$.format", ProjectV2DecodeError) != FORMAT_NAME:
        raise ProjectV2DecodeError(
            f"$.format 必须精确等于 {FORMAT_NAME!r}"
        )
    schema = _integer(root["schema"], "$.schema", ProjectV2DecodeError)
    if schema != SCHEMA_VERSION:
        raise ProjectV2DecodeError(
            f"v2 decoder 不能读取 schema {schema!r}"
        )

    project = _mapping(root["project"], "$.project", ProjectV2DecodeError)
    _keys(
        project,
        "$.project",
        required={"kind", "authoring"},
        optional=set(),
        error_type=ProjectV2DecodeError,
    )
    if _string(project["kind"], "$.project.kind", ProjectV2DecodeError) != "native":
        raise ProjectV2DecodeError(
            "$.project.kind 本阶段只接受 'native'"
        )
    authoring = _mapping(
        project["authoring"],
        "$.project.authoring",
        ProjectV2DecodeError,
    )
    _keys(
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
        error_type=ProjectV2DecodeError,
    )

    part = _decode_part(authoring["part"], "$.project.authoring.part")
    geometry = decode_geometry_field(
        authoring["geometry"],
        "$.project.authoring.geometry",
        policy=_V2_FIELD_POLICY,
    )
    stored_fingerprint = _decode_topology_fingerprint(
        authoring["logical_topology"],
        "$.project.authoring.logical_topology",
    )
    recomputed_fingerprint = topology_fingerprint_for_recipe(geometry)
    _require_matching_topology_fingerprint(
        stored_fingerprint,
        recomputed_fingerprint,
        authoring["logical_topology"],
        "$.project.authoring.logical_topology",
    )

    mesh_settings = _decode_mesh_settings(
        authoring["mesh_settings"],
        "$.project.authoring.mesh_settings",
        geometry,
    )
    named_regions = tuple(
        _decode_named_region(
            item,
            f"$.project.authoring.named_regions[{index}]",
            geometry,
        )
        for index, item in enumerate(
            _array(
                authoring["named_regions"],
                "$.project.authoring.named_regions",
                ProjectV2DecodeError,
            )
        )
    )
    _unique_names(
        named_regions,
        "$.project.authoring.named_regions",
        ProjectV2DecodeError,
    )

    definitions = _mapping(
        authoring["definitions"],
        "$.project.authoring.definitions",
        ProjectV2DecodeError,
    )
    _keys(
        definitions,
        "$.project.authoring.definitions",
        required={"materials", "sections", "assignments", "steps"},
        optional=set(),
        error_type=ProjectV2DecodeError,
    )
    materials = tuple(
        _decode_definition_v2(
            "material",
            item,
            f"$.project.authoring.definitions.materials[{index}]",
        )
        for index, item in enumerate(
            _array(
                definitions["materials"],
                "$.project.authoring.definitions.materials",
                ProjectV2DecodeError,
            )
        )
    )
    sections = tuple(
        _decode_definition_v2(
            "section",
            item,
            f"$.project.authoring.definitions.sections[{index}]",
        )
        for index, item in enumerate(
            _array(
                definitions["sections"],
                "$.project.authoring.definitions.sections",
                ProjectV2DecodeError,
            )
        )
    )
    assignments = tuple(
        decode_assignment_v2(
            item,
            path=f"$.project.authoring.definitions.assignments[{index}]",
        )
        for index, item in enumerate(
            _array(
                definitions["assignments"],
                "$.project.authoring.definitions.assignments",
                ProjectV2DecodeError,
            )
        )
    )
    steps = tuple(
        _decode_definition_v2(
            "step",
            item,
            f"$.project.authoring.definitions.steps[{index}]",
        )
        for index, item in enumerate(
            _array(
                definitions["steps"],
                "$.project.authoring.definitions.steps",
                ProjectV2DecodeError,
            )
        )
    )
    _unique_names(
        materials,
        "$.project.authoring.definitions.materials",
        ProjectV2DecodeError,
    )
    _unique_names(
        sections,
        "$.project.authoring.definitions.sections",
        ProjectV2DecodeError,
    )
    _unique_names(
        steps,
        "$.project.authoring.definitions.steps",
        ProjectV2DecodeError,
    )

    try:
        _validate_definition_links_v2(
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
        _validate_native_project_inputs_v2(
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
    except ProjectV2Error:
        raise
    except (KeyError, TypeError, ValueError) as error:
        raise ProjectV2DecodeError(
            f"$.project.authoring 无效：{error}"
        ) from error


def load_project_v2(path: str | Path) -> ProjectSnapshot:
    """Read a schema-v2 project without touching a live Session."""

    source = Path(path)
    return loads_project_v2(source.read_bytes(), source_path=source)


def encode_project_v2(
    snapshot: ProjectSnapshot | ProjectSaveSnapshot,
) -> dict[str, Any]:
    """Encode current native authoring inputs as a canonical v2 mapping."""

    try:
        project = unwrap_project_snapshot(
            snapshot,
            error_type=ProjectV2EncodeError,
        )
        if project.source_kind != "native":
            raise ProjectV2EncodeError(
                "snapshot.source_kind 本阶段只接受 'native'"
            )
        if project.model is not None:
            raise ProjectV2EncodeError(
                "v2 不持久化 compiled/runtime model"
            )
        geometry = project.geometry_recipe
        if geometry is None:
            raise ProjectV2EncodeError(
                "snapshot.geometry_recipe 不能为空"
            )
        parts = tuple(project.parts)
        if len(parts) != 1:
            raise ProjectV2EncodeError(
                "v2 native 项目必须且只能包含一个 part"
            )
        expected_history = derive_feature_history(geometry)
        if tuple(project.feature_history) != tuple(expected_history):
            raise ProjectV2EncodeError(
                "snapshot.feature_history 不是 geometry recipe 的 canonical 派生投影"
            )

        named_regions = tuple(project.named_regions)
        materials = tuple(project.material_definitions)
        sections = tuple(project.section_definitions)
        assignments = tuple(project.region_assignments)
        steps = tuple(project.analysis_definitions)
        _validate_encode_contextual_references(
            geometry,
            named_regions,
            project.mesh_settings,
        )
        _validate_definition_links_v2(
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
            raise ProjectV2EncodeError(
                "snapshot definitions 不是 normalize_model_definitions "
                "产生的 canonical 形式"
            )
        _validate_native_project_inputs_v2(
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
                    "part": _encode_part(
                        parts[0],
                        "snapshot.parts[0]",
                    ),
                    "geometry": encode_geometry_field(
                        geometry,
                        "snapshot.geometry_recipe",
                        set(),
                        policy=_V2_FIELD_POLICY,
                    ),
                    "logical_topology": _encode_topology_fingerprint(
                        topology_fingerprint_for_recipe(geometry)
                    ),
                    "mesh_settings": _encode_mesh_settings(
                        project.mesh_settings,
                        "snapshot.mesh_settings",
                    ),
                    "named_regions": [
                        _encode_named_region(
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
                                policy=_V2_FIELD_POLICY,
                            )
                            for index, material in enumerate(materials)
                        ],
                        "sections": [
                            encode_section_field(
                                section,
                                f"snapshot.section_definitions[{index}]",
                                policy=_V2_FIELD_POLICY,
                            )
                            for index, section in enumerate(sections)
                        ],
                        "assignments": [
                            encode_assignment_v2(
                                assignment,
                                path=(
                                    "snapshot.region_assignments"
                                    f"[{index}]"
                                ),
                            )
                            for index, assignment in enumerate(assignments)
                        ],
                        "steps": [
                            encode_step_field(
                                step,
                                f"snapshot.analysis_definitions[{index}]",
                                policy=_V2_FIELD_POLICY,
                            )
                            for index, step in enumerate(steps)
                        ],
                    },
                },
            },
        }
        # Validate every extension bag and non-finite value before a target
        # directory or temporary file can be created.
        dumps_canonical_json(
            payload,
            error_type=ProjectV2EncodeError,
        )
        return payload
    except ProjectV2Error:
        raise
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise ProjectV2EncodeError(
            f"snapshot 无法由 v2 无损表示：{error}"
        ) from error


def dumps_project_v2(
    snapshot: ProjectSnapshot | ProjectSaveSnapshot,
) -> str:
    """Return deterministic UTF-8 schema-v2 JSON with one trailing LF."""

    return dumps_canonical_json(
        encode_project_v2(snapshot),
        error_type=ProjectV2EncodeError,
    )


def save_project_v2(
    path: str | Path,
    snapshot: ProjectSnapshot | ProjectSaveSnapshot,
) -> Path:
    """Atomically write and read back one canonical schema-v2 project."""

    payload = encode_project_v2(snapshot)
    serialized = dumps_canonical_json(
        payload,
        error_type=ProjectV2EncodeError,
    )
    return atomic_write_project(
        path,
        serialized,
        verifier=load_project_v2,
        semantic_encoder=encode_project_v2,
        expected_semantic=payload,
        error_type=ProjectV2EncodeError,
        mismatch_message=(
            "临时 v2 项目回读后的 canonical authoring 值与保存 snapshot 不一致"
        ),
    )


read_project_v2 = load_project_v2
write_project_v2 = save_project_v2


def encode_assignment_v2(
    assignment: RegionAssignment,
    *,
    path: str = "assignment",
) -> dict[str, Any]:
    """Encode one assignment field object, including explicit orientation."""

    return encode_assignment_field(
        assignment,
        path,
        policy=_V2_FIELD_POLICY,
    )


def decode_assignment_v2(
    value: Any,
    *,
    path: str = "assignment",
) -> RegionAssignment:
    """Decode one required v2 assignment object."""

    return decode_assignment_field(
        value,
        path,
        policy=_V2_FIELD_POLICY,
    )


def _decode_part(value: Any, path: str) -> NativePart:
    data = _mapping(value, path, ProjectV2DecodeError)
    _keys(
        data,
        path,
        required={"name", "body_name"},
        optional=set(),
        error_type=ProjectV2DecodeError,
    )
    try:
        return NativePart(
            _string(data["name"], f"{path}.name", ProjectV2DecodeError),
            _string(
                data["body_name"],
                f"{path}.body_name",
                ProjectV2DecodeError,
            ),
        )
    except (TypeError, ValueError) as error:
        raise ProjectV2DecodeError(f"{path} 无效：{error}") from error


def _encode_part(part: Any, path: str) -> dict[str, Any]:
    if type(part) is not NativePart:
        raise ProjectV2EncodeError(f"{path} 必须是 NativePart")
    return {
        "name": _string(part.name, f"{path}.name", ProjectV2EncodeError),
        "body_name": _string(
            part.body_name,
            f"{path}.body_name",
            ProjectV2EncodeError,
        ),
    }


def _decode_named_region(
    value: Any,
    path: str,
    recipe: Any,
) -> NamedRegion:
    data = _mapping(value, path, ProjectV2DecodeError)
    _keys(
        data,
        path,
        required={"name", "references"},
        optional=set(),
        error_type=ProjectV2DecodeError,
    )
    references = tuple(
        _decode_contextual_reference(
            item,
            f"{path}.references[{index}]",
            recipe,
        )
        for index, item in enumerate(
            _array(
                data["references"],
                f"{path}.references",
                ProjectV2DecodeError,
            )
        )
    )
    try:
        return NamedRegion(
            _string(data["name"], f"{path}.name", ProjectV2DecodeError),
            references,
        )
    except (TypeError, ValueError) as error:
        raise ProjectV2DecodeError(f"{path} 无效：{error}") from error


def _decode_logical_reference(value: Any, path: str) -> LogicalEntityRef:
    try:
        return LogicalEntityRef(
            _string(value, path, ProjectV2DecodeError)
        )
    except ProjectV2Error:
        raise
    except (TypeError, ValueError) as error:
        raise ProjectV2DecodeError(f"{path} 无效：{error}") from error


def _decode_contextual_reference(
    value: Any,
    path: str,
    recipe: Any,
    *,
    allowed_kinds: tuple[EntityKind, ...] | None = None,
) -> LogicalEntityRef:
    reference = _decode_logical_reference(value, path)
    return _validate_contextual_reference(
        reference,
        path,
        recipe,
        ProjectV2DecodeError,
        allowed_kinds=allowed_kinds,
    )


def _validate_contextual_reference(
    reference: LogicalEntityRef,
    path: str,
    recipe: Any,
    error_type: type[ProjectV2DecodeError | ProjectV2EncodeError],
    *,
    allowed_kinds: tuple[EntityKind, ...] | None = None,
) -> LogicalEntityRef:
    try:
        validate_logical_reference(
            recipe,
            reference,
            allowed_kinds=allowed_kinds,
            require_exact=False,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise error_type(f"{path} 无效：{error}") from error
    return reference


def _validate_encode_contextual_references(
    recipe: Any,
    named_regions: tuple[NamedRegion, ...],
    mesh_settings: MeshSettings | None,
) -> None:
    for region_index, region in enumerate(named_regions):
        for reference_index, reference in enumerate(region.references):
            _validate_contextual_reference(
                reference,
                (
                    f"snapshot.named_regions[{region_index}]"
                    f".references[{reference_index}]"
                ),
                recipe,
                ProjectV2EncodeError,
            )
    if mesh_settings is None:
        return
    for index, control in enumerate(mesh_settings.local_controls):
        path = f"snapshot.mesh_settings.local_controls[{index}].target"
        target = _validate_contextual_reference(
            control.target,
            path,
            recipe,
            ProjectV2EncodeError,
            allowed_kinds=("point", "edge", "face"),
        )
        if control.falloff.reference != "target_radius":
            continue
        try:
            resolve_target_radius(recipe, target)
        except (KeyError, TypeError, ValueError) as error:
            raise ProjectV2EncodeError(f"{path} 无效：{error}") from error


def _validate_definition_links_v2(
    materials: tuple[Any, ...],
    sections: tuple[Any, ...],
    assignments: tuple[Any, ...],
    *,
    encode: bool,
) -> None:
    error_type = ProjectV2EncodeError if encode else ProjectV2DecodeError
    material_prefix = (
        "snapshot.material_definitions"
        if encode
        else "$.project.authoring.definitions.materials"
    )
    section_prefix = (
        "snapshot.section_definitions"
        if encode
        else "$.project.authoring.definitions.sections"
    )
    assignment_prefix = (
        "snapshot.region_assignments"
        if encode
        else "$.project.authoring.definitions.assignments"
    )
    material_names = {
        str(material.name).strip() for material in materials
    }
    section_names = {str(section.name).strip() for section in sections}
    for index, material in enumerate(materials):
        if BEAM_LOCAL_Y_REFERENCE_KEY in material.properties:
            _raise_v2_context_error(
                error_type,
                (
                    f"{material_prefix}[{index}].properties"
                    f".{BEAM_LOCAL_Y_REFERENCE_KEY}"
                ),
                "reserved Beam orientation property is not allowed",
            )
    for index, section in enumerate(sections):
        if BEAM_LOCAL_Y_REFERENCE_KEY in section.properties:
            _raise_v2_context_error(
                error_type,
                (
                    f"{section_prefix}[{index}].properties"
                    f".{BEAM_LOCAL_Y_REFERENCE_KEY}"
                ),
                "reserved Beam orientation property is not allowed",
            )
        material_name = str(section.material).strip()
        if material_name not in material_names:
            _raise_v2_context_error(
                error_type,
                f"{section_prefix}[{index}].material",
                f"references missing material {material_name!r}",
            )
    for index, assignment in enumerate(assignments):
        section_name = str(assignment.section_name).strip()
        if section_name not in section_names:
            _raise_v2_context_error(
                error_type,
                f"{assignment_prefix}[{index}].section_name",
                f"references missing section {section_name!r}",
            )


def _raise_v2_context_error(
    error_type: type[ProjectV2DecodeError | ProjectV2EncodeError],
    path: str,
    message: str,
) -> None:
    cause = ValueError(message)
    raise error_type(f"{path} 无效：{message}") from cause


def _validate_native_project_inputs_v2(
    recipe: Any,
    mesh_settings: MeshSettings | None,
    named_regions: tuple[NamedRegion, ...],
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
            message = f"snapshot 无法由 v2 无损表示：{message}"
        else:
            message = f"$.project.authoring 无效：{message}"
        error_type = ProjectV2EncodeError if encode else ProjectV2DecodeError
        raise error_type(message) from error


def _encode_named_region(region: Any, path: str) -> dict[str, Any]:
    if type(region) is not NamedRegion:
        raise ProjectV2EncodeError(f"{path} 必须是 NamedRegion")
    references = tuple(region.references)
    canonical = tuple(sorted(references, key=logical_ref_sort_key))
    if references != canonical:
        raise ProjectV2EncodeError(
            f"{path}.references 不是 canonical logical-reference 顺序"
        )
    return {
        "name": _string(
            region.name,
            f"{path}.name",
            ProjectV2EncodeError,
        ),
        "references": [reference.logical_id for reference in references],
    }


def _decode_mesh_settings(
    value: Any,
    path: str,
    recipe: Any,
) -> MeshSettings | None:
    if value is None:
        return None
    data = _mapping(value, path, ProjectV2DecodeError)
    _keys(
        data,
        path,
        required={"size", "order", "cell_shape", "local_controls"},
        optional=set(),
        error_type=ProjectV2DecodeError,
    )
    controls = tuple(
        _decode_local_control(
            item,
            f"{path}.local_controls[{index}]",
            recipe,
        )
        for index, item in enumerate(
            _array(
                data["local_controls"],
                f"{path}.local_controls",
                ProjectV2DecodeError,
            )
        )
    )
    try:
        return MeshSettings(
            _number(data["size"], f"{path}.size", ProjectV2DecodeError),
            order=_integer(
                data["order"],
                f"{path}.order",
                ProjectV2DecodeError,
            ),
            cell_shape=_string(
                data["cell_shape"],
                f"{path}.cell_shape",
                ProjectV2DecodeError,
            ),
            local_controls=controls,
        )
    except (TypeError, ValueError) as error:
        raise ProjectV2DecodeError(f"{path} 无效：{error}") from error


def _encode_mesh_settings(
    settings: Any,
    path: str,
) -> dict[str, Any] | None:
    if settings is None:
        return None
    if type(settings) is not MeshSettings:
        raise ProjectV2EncodeError(f"{path} 必须是 MeshSettings 或 null")
    return {
        "size": _number(
            settings.size,
            f"{path}.size",
            ProjectV2EncodeError,
        ),
        "order": _integer(
            settings.order,
            f"{path}.order",
            ProjectV2EncodeError,
        ),
        "cell_shape": _string(
            settings.cell_shape,
            f"{path}.cell_shape",
            ProjectV2EncodeError,
        ),
        "local_controls": [
            _encode_local_control(
                control,
                f"{path}.local_controls[{index}]",
            )
            for index, control in enumerate(settings.local_controls)
        ],
    }


def _decode_local_control(
    value: Any,
    path: str,
    recipe: Any,
) -> LocalMeshControl:
    data = _mapping(value, path, ProjectV2DecodeError)
    _keys(
        data,
        path,
        required={"target", "size", "falloff"},
        optional=set(),
        error_type=ProjectV2DecodeError,
    )
    falloff_data = _mapping(
        data["falloff"],
        f"{path}.falloff",
        ProjectV2DecodeError,
    )
    _keys(
        falloff_data,
        f"{path}.falloff",
        required={"reference", "start_factor", "end_factor"},
        optional=set(),
        error_type=ProjectV2DecodeError,
    )
    try:
        falloff = MeshSizeFalloff(
            _string(
                falloff_data["reference"],
                f"{path}.falloff.reference",
                ProjectV2DecodeError,
            ),
            _number(
                falloff_data["start_factor"],
                f"{path}.falloff.start_factor",
                ProjectV2DecodeError,
            ),
            _number(
                falloff_data["end_factor"],
                f"{path}.falloff.end_factor",
                ProjectV2DecodeError,
            ),
        )
    except ProjectV2Error:
        raise
    except (TypeError, ValueError) as error:
        raise ProjectV2DecodeError(
            f"{path}.falloff 无效：{error}"
        ) from error
    target = _decode_contextual_reference(
        data["target"],
        f"{path}.target",
        recipe,
        allowed_kinds=("point", "edge", "face"),
    )
    if falloff.reference == "target_radius":
        try:
            resolve_target_radius(recipe, target)
        except (KeyError, TypeError, ValueError) as error:
            raise ProjectV2DecodeError(
                f"{path}.target 无效：{error}"
            ) from error
    try:
        return LocalMeshControl(
            target,
            _number(
                data["size"],
                f"{path}.size",
                ProjectV2DecodeError,
            ),
            falloff,
        )
    except ProjectV2Error:
        raise
    except (TypeError, ValueError) as error:
        raise ProjectV2DecodeError(f"{path} 无效：{error}") from error


def _encode_local_control(control: Any, path: str) -> dict[str, Any]:
    if type(control) is not LocalMeshControl:
        raise ProjectV2EncodeError(f"{path} 必须是 LocalMeshControl")
    falloff = control.falloff
    if type(falloff) is not MeshSizeFalloff:
        raise ProjectV2EncodeError(
            f"{path}.falloff 必须是 MeshSizeFalloff"
        )
    return {
        "target": control.target.logical_id,
        "size": _number(
            control.size,
            f"{path}.size",
            ProjectV2EncodeError,
        ),
        "falloff": {
            "reference": falloff.reference,
            "start_factor": _number(
                falloff.start_factor,
                f"{path}.falloff.start_factor",
                ProjectV2EncodeError,
            ),
            "end_factor": _number(
                falloff.end_factor,
                f"{path}.falloff.end_factor",
                ProjectV2EncodeError,
            ),
        },
    }


def _encode_topology_fingerprint(
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
                }
                for entity in fingerprint.entities
            ],
        },
    }


def _decode_topology_fingerprint(
    value: Any,
    path: str,
) -> TopologyFingerprint:
    data = _mapping(value, path, ProjectV2DecodeError)
    _keys(
        data,
        path,
        required={"contract", "signature"},
        optional=set(),
        error_type=ProjectV2DecodeError,
    )
    contract = _integer(
        data["contract"],
        f"{path}.contract",
        ProjectV2DecodeError,
    )
    if contract != TOPOLOGY_REFERENCE_CONTRACT:
        raise ProjectV2DecodeError(
            f"{path}.contract 不支持：{contract!r}"
        )
    signature = _mapping(
        data["signature"],
        f"{path}.signature",
        ProjectV2DecodeError,
    )
    _keys(
        signature,
        f"{path}.signature",
        required={"dimension", "exact", "entities"},
        optional=set(),
        error_type=ProjectV2DecodeError,
    )
    exact = signature["exact"]
    if type(exact) is not bool:
        raise ProjectV2DecodeError(
            f"{path}.signature.exact 必须是 boolean"
        )
    records: list[TopologyFingerprintEntity] = []
    for index, item in enumerate(
        _array(
            signature["entities"],
            f"{path}.signature.entities",
            ProjectV2DecodeError,
        )
    ):
        record_path = f"{path}.signature.entities[{index}]"
        record = _mapping(item, record_path, ProjectV2DecodeError)
        _keys(
            record,
            record_path,
            required={"kind", "logical_id", "semantic_role", "selectable"},
            optional=set(),
            error_type=ProjectV2DecodeError,
        )
        selectable = record["selectable"]
        if type(selectable) is not bool:
            raise ProjectV2DecodeError(
                f"{record_path}.selectable 必须是 boolean"
            )
        try:
            records.append(
                TopologyFingerprintEntity(
                    _string(
                        record["kind"],
                        f"{record_path}.kind",
                        ProjectV2DecodeError,
                    ),
                    _string(
                        record["logical_id"],
                        f"{record_path}.logical_id",
                        ProjectV2DecodeError,
                    ),
                    _string(
                        record["semantic_role"],
                        f"{record_path}.semantic_role",
                        ProjectV2DecodeError,
                    ),
                    selectable,
                )
            )
        except (TypeError, ValueError) as error:
            raise ProjectV2DecodeError(
                f"{record_path} 无效：{error}"
            ) from error
    try:
        return TopologyFingerprint(
            dimension=_integer(
                signature["dimension"],
                f"{path}.signature.dimension",
                ProjectV2DecodeError,
            ),
            exact=exact,
            entities=tuple(records),
            contract=contract,
        )
    except (TypeError, ValueError) as error:
        raise ProjectV2DecodeError(f"{path} 无效：{error}") from error


def _require_matching_topology_fingerprint(
    stored: TopologyFingerprint,
    expected: TopologyFingerprint,
    raw_value: Any,
    path: str,
) -> None:
    """Reject stale evidence at the narrowest meaningful JSON path."""

    signature_path = f"{path}.signature"
    if stored.dimension != expected.dimension:
        raise ProjectV2DecodeError(
            f"{signature_path}.dimension 与 geometry 重新计算值 "
            f"{expected.dimension!r} 不一致（topology fingerprint）"
        )
    if stored.exact != expected.exact:
        raise ProjectV2DecodeError(
            f"{signature_path}.exact 与 geometry 重新计算值 "
            f"{expected.exact!r} 不一致（topology fingerprint）"
        )

    stored_by_id = {
        entity.logical_id: entity for entity in stored.entities
    }
    expected_by_id = {
        entity.logical_id: entity for entity in expected.entities
    }
    raw_indices = _topology_entity_indices(raw_value, path)
    missing = sorted(
        set(expected_by_id) - set(stored_by_id),
        key=lambda logical_id: logical_ref_sort_key(
            LogicalEntityRef(logical_id)
        ),
    )
    if missing:
        raise ProjectV2DecodeError(
            f"{signature_path}.entities 缺少 geometry topology entity "
            f"{missing[0]!r}（topology fingerprint）"
        )
    extra = sorted(
        set(stored_by_id) - set(expected_by_id),
        key=lambda logical_id: logical_ref_sort_key(
            LogicalEntityRef(logical_id)
        ),
    )
    if extra:
        logical_id = extra[0]
        index = raw_indices[logical_id]
        raise ProjectV2DecodeError(
            f"{signature_path}.entities[{index}].logical_id 包含 geometry "
            f"中不存在的 topology entity {logical_id!r}"
            "（topology fingerprint）"
        )

    for logical_id, expected_entity in expected_by_id.items():
        stored_entity = stored_by_id[logical_id]
        index = raw_indices[logical_id]
        entity_path = f"{signature_path}.entities[{index}]"
        if stored_entity.semantic_role != expected_entity.semantic_role:
            raise ProjectV2DecodeError(
                f"{entity_path}.semantic_role 与 geometry 重新计算值 "
                f"{expected_entity.semantic_role!r} 不一致"
                "（topology fingerprint）"
            )
        if stored_entity.selectable != expected_entity.selectable:
            raise ProjectV2DecodeError(
                f"{entity_path}.selectable 与 geometry 重新计算值 "
                f"{expected_entity.selectable!r} 不一致"
                "（topology fingerprint）"
            )


def _topology_entity_indices(
    raw_value: Any,
    path: str,
) -> dict[str, int]:
    data = _mapping(raw_value, path, ProjectV2DecodeError)
    signature = _mapping(
        data["signature"],
        f"{path}.signature",
        ProjectV2DecodeError,
    )
    entities = _array(
        signature["entities"],
        f"{path}.signature.entities",
        ProjectV2DecodeError,
    )
    return {
        _mapping(
            item,
            f"{path}.signature.entities[{index}]",
            ProjectV2DecodeError,
        )["logical_id"]: index
        for index, item in enumerate(entities)
    }


def _decode_definition_v2(kind: str, value: Any, path: str) -> Any:
    """Decode one strict current definition through the shared field codec."""

    decoders = {
        "material": decode_material_field,
        "section": decode_section_field,
        "step": decode_step_field,
    }
    try:
        decoder = decoders[kind]
    except KeyError as error:
        raise ProjectV2DecodeError(
            f"{path} 使用未知 definition kind {kind!r}"
        ) from error
    return decoder(value, path, policy=_V2_FIELD_POLICY)


def _unique_names(
    values: Sequence[Any],
    path: str,
    error_type: type[Exception],
) -> None:
    seen: dict[str, str] = {}
    for index, value in enumerate(values):
        name = getattr(value, "name", None)
        if type(name) is not str or not name.strip():
            raise error_type(f"{path}[{index}].name 必须是 non-empty string")
        folded = name.casefold()
        if folded in seen:
            raise error_type(
                f"{path} 包含忽略大小写后重复的名称 "
                f"{seen[folded]!r} 与 {name!r}"
            )
        seen[folded] = name


def _keys(
    data: Mapping[str, Any],
    path: str,
    *,
    required: set[str],
    optional: set[str],
    error_type: type[Exception],
) -> None:
    actual = set(data)
    missing = sorted(required - actual)
    if missing:
        raise error_type(f"{path} 缺少必需字段：{', '.join(missing)}")
    unknown = sorted(actual - required - optional)
    if unknown:
        raise error_type(f"{path} 包含 v2 未知字段：{', '.join(unknown)}")


def _mapping(
    value: Any,
    path: str,
    error_type: type[Exception],
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise error_type(f"{path} 必须是 JSON object")
    if any(type(key) is not str for key in value):
        raise error_type(f"{path} 的键必须是字符串")
    return value


def _array(
    value: Any,
    path: str,
    error_type: type[Exception],
) -> tuple[Any, ...]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(
        value,
        Sequence,
    ):
        raise error_type(f"{path} 必须是 JSON array")
    return tuple(value)


def _string(
    value: Any,
    path: str,
    error_type: type[Exception],
) -> str:
    if type(value) is not str:
        raise error_type(f"{path} 必须是字符串")
    if not value.strip():
        raise error_type(f"{path} 不能为空")
    return value


def _integer(
    value: Any,
    path: str,
    error_type: type[Exception],
) -> int:
    if type(value) is not int:
        raise error_type(f"{path} 必须是严格整数")
    return value


def _number(
    value: Any,
    path: str,
    error_type: type[Exception],
) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise error_type(f"{path} 必须是有限实数")
    try:
        finite = math.isfinite(float(value))
    except OverflowError as error:
        raise error_type(f"{path} 必须是有限实数") from error
    if not finite:
        raise error_type(f"{path} 必须是有限实数")
    return value


__all__ = [
    "FORMAT_NAME",
    "SCHEMA_VERSION",
    "ProjectV2DecodeError",
    "ProjectV2EncodeError",
    "ProjectV2Error",
    "decode_assignment_v2",
    "decode_project_v2",
    "dumps_project_v2",
    "encode_assignment_v2",
    "encode_project_v2",
    "load_project_v2",
    "loads_project_v2",
    "read_project_v2",
    "save_project_v2",
    "write_project_v2",
]
