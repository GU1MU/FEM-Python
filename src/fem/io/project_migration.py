"""Typed compatibility migration from frozen ``.femproj`` schema v1."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fem.application.definitions import (
    FeatureRecord,
    NamedRegion,
    NativePart,
    RegionAssignment,
    SectionDefinition,
    normalize_model_definitions,
)
from fem.application.feature_history import derive_feature_history
from fem.application.project_validation import (
    NativeProjectValidationError,
    validate_native_project_inputs,
)
from fem.application.session import ProjectSnapshot
from fem.application.session import PartBooleanUndoRecord
from fem.application.native_part import (
    PartBooleanProvenance,
    validate_native_parts,
)
from fem.core.model import (
    AnalysisStep,
    MaterialDefinition,
)
from fem.geometry.measurements import (
    TargetRadiusResolutionError,
    resolve_legacy_hole_target,
)
from fem.geometry.recipe_analysis import legacy_sketches_to_strict
from fem.geometry.recipe_topology import describe_recipe_topology
from fem.geometry.recipe_topology import canonicalize_multi_body_logical_id
from fem.geometry.body_operations import materialize_multi_body
from fem.geometry.recipes import (
    BooleanGeometry,
    ExtrudedGeometry,
    MovedGeometry,
    MultiBodyGeometry,
    RotatedGeometry,
    geometry_dimension,
)
from fem.geometry.part_boolean import namespace_part_boolean_context
from fem.geometry.part_namespace import namespace_part_logical_id
from fem.geometry.references import LogicalEntityRef
from fem.mesh.settings import (
    LocalMeshControl,
    MeshSettings,
    MeshSizeFalloff,
)


_ENTITY_KINDS = frozenset({"point", "edge", "face", "body"})
_GLOBAL_FALLOFF = MeshSizeFalloff("global_size", 0.0, 2.0)
_TARGET_RADIUS_FALLOFF = MeshSizeFalloff("target_radius", 0.25, 2.0)


@dataclass(frozen=True, slots=True)
class ProjectMigrationNotice:
    """One stable, non-authoritative compatibility migration notice."""

    code: str
    message: str
    path: str


@dataclass(frozen=True, slots=True)
class LegacyNamedRegionV1:
    """One v1 named region retaining ordinal entity references."""

    name: str
    entity_kind: str
    entity_ids: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class LegacyLocalMeshControlV1:
    """One v1 entity-local size retaining its compatibility ordinal."""

    entity_kind: str
    entity_id: int
    size: float


@dataclass(frozen=True, slots=True)
class LegacyMeshSettingsV1:
    """Frozen v1 mesh settings including the legacy hole-size entry."""

    size: float
    order: int
    cell_shape: str
    local_size: float | None
    local_controls: tuple[LegacyLocalMeshControlV1, ...]


@dataclass(frozen=True, slots=True)
class LegacyProjectV1:
    """Fully decoded private v1 wire DTO before current-domain migration."""

    source_path: Path | None
    logical_topology_version: int | None
    parts: tuple[NativePart, ...] | None
    geometry_recipe: Any
    mesh_settings: LegacyMeshSettingsV1 | None
    feature_history: tuple[FeatureRecord, ...]
    feature_history_present: bool
    named_regions: tuple[LegacyNamedRegionV1, ...]
    material_definitions: tuple[MaterialDefinition, ...]
    section_definitions: tuple[SectionDefinition, ...]
    region_assignments: tuple[RegionAssignment, ...]
    analysis_definitions: tuple[AnalysisStep, ...]


class ProjectV1MigrationError(ValueError):
    """A valid v1 wire value cannot be mapped to current authoring state."""


def migrate_project_v1(
    legacy: LegacyProjectV1,
) -> tuple[ProjectSnapshot, tuple[ProjectMigrationNotice, ...]]:
    """Migrate one detached legacy DTO into canonical current authoring state."""

    if type(legacy) is not LegacyProjectV1:
        raise TypeError("legacy must be a LegacyProjectV1")
    if legacy.logical_topology_version not in {None, 1}:
        raise ProjectV1MigrationError(
            "$.logical_topology_version 使用不支持的逻辑拓扑契约版本："
            f"{legacy.logical_topology_version!r}"
        )

    explicit_integer_references = bool(
        any(region.entity_ids for region in legacy.named_regions)
        or (
            legacy.mesh_settings is not None
            and legacy.mesh_settings.local_controls
        )
    )
    if (
        legacy.logical_topology_version is None
        and explicit_integer_references
    ):
        raise ProjectV1MigrationError(
            "旧项目缺少逻辑拓扑契约版本，无法安全恢复命名区域或局部网格控制；"
            "请移除这些实体引用后打开项目并重新选择"
        )

    if legacy.parts is None:
        parts = (NativePart(),)
    elif len(legacy.parts) != 1:
        raise ProjectV1MigrationError(
            "$.parts 必须显式包含且只包含一个 NativePart；"
            f"收到 {len(legacy.parts)} 个"
        )
    else:
        parts = legacy.parts

    source_geometry = legacy.geometry_recipe
    geometry_recipe = legacy_sketches_to_strict(source_geometry)

    named_regions = tuple(
        _migrate_named_region(
            source_geometry,
            region,
            index=index,
        )
        for index, region in enumerate(legacy.named_regions)
    )
    mesh_settings, mesh_notices = _migrate_mesh_settings(
        source_geometry,
        legacy.mesh_settings,
    )

    canonical_history = derive_feature_history(geometry_recipe)
    notices: list[ProjectMigrationNotice] = [
        ProjectMigrationNotice(
            code="project.schema.v1",
            message=(
                "项目已通过 schema 1 compatibility migration 打开；"
                "下次显式保存将升级为 schema 7（v7）当前项目格式"
            ),
            path="$.schema",
        )
    ]
    if geometry_recipe is not source_geometry:
        notices.append(
            ProjectMigrationNotice(
                code="project.v1.sketch_curve_graph",
                message=(
                    "legacy SketchGeometry contours 已迁移为严格 point/curve graph；"
                    "Profile 材料/孔关系由 loop containment 重新推导"
                ),
                path="$.geometry",
            )
        )
    notices.extend(mesh_notices)
    if (
        legacy.feature_history_present
        and legacy.feature_history != canonical_history
    ):
        notices.append(
            ProjectMigrationNotice(
                code="project.v1.feature_history_rederived",
                message=(
                    "legacy feature_history 是显示缓存，已根据 geometry "
                    "recipe 重新推导"
                ),
                path="$.feature_history",
            )
        )

    try:
        definitions = normalize_model_definitions(
            legacy.material_definitions,
            legacy.section_definitions,
            legacy.region_assignments,
            legacy.analysis_definitions,
        )
        _validate_current_native_authoring(
            geometry_recipe,
            mesh_settings,
            named_regions,
            definitions.materials,
            definitions.sections,
            definitions.assignments,
            definitions.steps,
        )
        snapshot = ProjectSnapshot(
            source_kind="native",
            source_path=legacy.source_path,
            parts=parts,
            geometry_recipe=geometry_recipe,
            mesh_settings=mesh_settings,
            feature_history=canonical_history,
            named_regions=named_regions,
            material_definitions=definitions.materials,
            section_definitions=definitions.sections,
            region_assignments=definitions.assignments,
            analysis_definitions=definitions.steps,
        )
    except ProjectV1MigrationError:
        raise
    except (KeyError, TypeError, ValueError) as error:
        raise ProjectV1MigrationError(
            f"v1 项目迁移后的 authoring context 无效：{error}"
        ) from error
    return snapshot, tuple(notices)


def _migrate_named_region(
    recipe: Any,
    legacy: LegacyNamedRegionV1,
    *,
    index: int,
) -> NamedRegion:
    path = f"$.named_regions[{index}]"
    if not legacy.entity_ids:
        raise ProjectV1MigrationError(
            f"{path}.entity_ids 不能为空；named region {legacy.name!r} "
            "无法迁移为有效的 current region"
        )
    references = tuple(
        _ordinal_to_reference(
            recipe,
            legacy.entity_kind,
            ordinal,
            path=f"{path}.entity_ids[{entity_index}]",
        )
        for entity_index, ordinal in enumerate(legacy.entity_ids)
    )
    try:
        return NamedRegion(legacy.name, references)
    except (TypeError, ValueError) as error:
        raise ProjectV1MigrationError(f"{path} 无法迁移：{error}") from error


def _migrate_mesh_settings(
    recipe: Any,
    legacy: LegacyMeshSettingsV1 | None,
) -> tuple[MeshSettings | None, tuple[ProjectMigrationNotice, ...]]:
    if legacy is None:
        return None, ()
    controls: list[LocalMeshControl] = []
    for index, control in enumerate(legacy.local_controls):
        path = f"$.mesh_settings.local_controls[{index}]"
        target = _ordinal_to_reference(
            recipe,
            control.entity_kind,
            control.entity_id,
            path=f"{path}.entity_id",
        )
        controls.append(
            LocalMeshControl(
                target,
                control.size,
                _GLOBAL_FALLOFF,
            )
        )

    notices: list[ProjectMigrationNotice] = []
    if legacy.local_size is not None:
        try:
            target = resolve_legacy_hole_target(recipe)
        except (KeyError, TypeError, ValueError, TargetRadiusResolutionError) as error:
            raise ProjectV1MigrationError(
                "$.mesh_settings.local_size 无法证明唯一圆孔 target；"
                "请在旧项目中移除该值或重新选择可证明的圆孔边界"
            ) from error
        controls.append(
            LocalMeshControl(
                target,
                legacy.local_size,
                _TARGET_RADIUS_FALLOFF,
            )
        )
        notices.append(
            ProjectMigrationNotice(
                code="project.v1.local_size_migrated",
                message=(
                    "legacy local_size 已迁移为 target_radius "
                    "local mesh control"
                ),
                path="$.mesh_settings.local_size",
            )
        )

    canonical_controls = _canonicalize_controls(
        controls,
        path="$.mesh_settings",
    )
    try:
        return (
            MeshSettings(
                legacy.size,
                order=legacy.order,
                cell_shape=legacy.cell_shape,
                local_controls=canonical_controls,
            ),
            tuple(notices),
        )
    except (TypeError, ValueError) as error:
        raise ProjectV1MigrationError(
            f"$.mesh_settings 无法迁移：{error}"
        ) from error


def _ordinal_to_reference(
    recipe: Any,
    entity_kind: str,
    ordinal: int,
    *,
    path: str,
) -> LogicalEntityRef:
    if entity_kind not in _ENTITY_KINDS:
        raise ProjectV1MigrationError(
            f"{path} 的 entity kind {entity_kind!r} 不受支持"
        )
    topology = describe_recipe_topology(recipe)
    if not topology.exact:
        raise ProjectV1MigrationError(
            f"{path} 无法迁移：geometry topology 不是 exact"
        )
    entities = topology.entities_of(entity_kind)
    if ordinal <= 0 or ordinal > len(entities):
        raise ProjectV1MigrationError(
            f"{path} ordinal {ordinal!r} 超出 {entity_kind!r} catalog 范围"
        )
    entity = entities[ordinal - 1]
    if not entity.selectable:
        raise ProjectV1MigrationError(
            f"{path} ordinal {ordinal!r} 映射到不可选择实体 "
            f"{entity.logical_id!r}"
        )
    return LogicalEntityRef(entity.logical_id)


def _canonicalize_controls(
    controls: Iterable[LocalMeshControl],
    *,
    path: str,
) -> tuple[LocalMeshControl, ...]:
    unique: dict[
        tuple[LogicalEntityRef, MeshSizeFalloff],
        LocalMeshControl,
    ] = {}
    for control in controls:
        key = (control.target, control.falloff)
        previous = unique.get(key)
        if previous is None:
            unique[key] = control
        elif previous.size != control.size:
            raise ProjectV1MigrationError(
                f"{path} 对 target {control.target.logical_id!r} 和 "
                f"falloff {control.falloff.reference!r} 包含冲突 size"
            )
    return tuple(unique.values())


def _validate_current_native_authoring(
    recipe: Any,
    mesh_settings: MeshSettings | None,
    named_regions: tuple[NamedRegion, ...],
    materials: tuple[MaterialDefinition, ...],
    sections: tuple[SectionDefinition, ...],
    assignments: tuple[RegionAssignment, ...],
    steps: tuple[AnalysisStep, ...],
) -> None:
    """Delegate to the sole detached current project validator."""

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
    except ProjectV1MigrationError:
        raise
    except NativeProjectValidationError as error:
        raise ProjectV1MigrationError(str(error)) from error


__all__ = [
    "ProjectMigrationNotice",
    "migrate_project_snapshot_to_v5",
    "migrate_project_snapshot_to_v7",
    "migrate_project_v1",
]


def migrate_project_snapshot_to_v5(
    snapshot: ProjectSnapshot,
) -> tuple[ProjectSnapshot, tuple[ProjectMigrationNotice, ...]]:
    """Promote legacy 3D authoring into the canonical schema-v5 Body model."""

    if type(snapshot) is not ProjectSnapshot:
        raise TypeError("snapshot must be a ProjectSnapshot")
    recipe = snapshot.geometry_recipe
    if (
        recipe is None
        or geometry_dimension(recipe) != 3
        or isinstance(recipe, MultiBodyGeometry)
    ):
        return snapshot, ()
    body_name = (
        snapshot.parts[0].body_name
        if snapshot.parts
        else "Body-1"
    )
    geometry = materialize_multi_body(
        recipe,
        geometry_name=f"{getattr(recipe, 'name', 'Part-1')} Geometry",
        first_body_name=body_name,
    )

    def canonical(reference: LogicalEntityRef) -> LogicalEntityRef:
        try:
            logical_id = canonicalize_multi_body_logical_id(
                geometry,
                reference.logical_id,
            )
        except KeyError:
            logical_id = _canonicalize_expanded_extrusion_reference(
                geometry,
                reference.logical_id,
            )
        return LogicalEntityRef(logical_id)

    named_regions = tuple(
        replace(
            region,
            references=tuple(
                canonical(reference)
                if type(reference) is LogicalEntityRef
                else reference
                for reference in region.references
            ),
        )
        for region in snapshot.named_regions
    )
    mesh_settings = snapshot.mesh_settings
    if mesh_settings is not None:
        mesh_settings = replace(
            mesh_settings,
            local_controls=tuple(
                replace(control, target=canonical(control.target))
                for control in mesh_settings.local_controls
            ),
        )
    migrated = replace(
        snapshot,
        geometry_recipe=geometry,
        mesh_settings=mesh_settings,
        feature_history=derive_feature_history(geometry),
        named_regions=named_regions,
    )
    return (
        migrated,
        (
            ProjectMigrationNotice(
                "project.schema.v5.multi_body",
                "3D native geometry was promoted to stable schema-v5 Bodies.",
                "$.project.authoring.geometry",
            ),
        ),
    )


def _canonicalize_expanded_extrusion_reference(
    geometry: MultiBodyGeometry,
    logical_id: str,
) -> str:
    reference = LogicalEntityRef(logical_id)
    if reference.kind == "body":
        raise ProjectV1MigrationError(
            "legacy aggregate body:domain is ambiguous after multi-Profile "
            "Body materialization"
        )
    _kind, semantic_name = logical_id.split(":", 1)
    candidates: list[str] = []
    for body in geometry.bodies:
        body_recipe = body.recipe
        if not isinstance(body_recipe, ExtrudedGeometry):
            continue
        source_ids = body_recipe.source_face_ids
        if len(source_ids) != 1:
            continue
        source_name = source_ids[0].split(":", 1)[1]
        if semantic_name == source_name:
            local_name = "domain"
        elif semantic_name.endswith(f"/{source_name}"):
            local_name = semantic_name[: -(len(source_name) + 1)]
        else:
            continue
        candidate = f"{reference.kind}:{body.id}/{local_name}"
        try:
            describe_recipe_topology(geometry).entity(candidate)
        except KeyError:
            continue
        candidates.append(candidate)
    if len(candidates) != 1:
        raise ProjectV1MigrationError(
            f"legacy logical reference {logical_id!r} cannot be uniquely "
            "mapped to one schema-v5 Body"
        )
    return candidates[0]


def migrate_project_snapshot_to_v7(
    snapshot: ProjectSnapshot,
) -> tuple[ProjectSnapshot, tuple[ProjectMigrationNotice, ...]]:
    """Move one v1-v6 global recipe into canonical Part-owned authoring."""

    if type(snapshot) is not ProjectSnapshot:
        raise TypeError("snapshot must be a ProjectSnapshot")
    if snapshot.parts and all(
        part.geometry_recipe is not None for part in snapshot.parts
    ):
        return snapshot, ()
    recipe = snapshot.geometry_recipe
    if recipe is None:
        return replace(
            snapshot,
            parts=(),
            geometry_recipe=None,
            mesh_settings=None,
            feature_history=(),
            active_part_id=None,
        ), ()
    if isinstance(recipe, MultiBodyGeometry):
        migrated = _migrate_multi_body_snapshot_to_v7(snapshot, recipe)
    else:
        part_name = (
            snapshot.parts[0].name if snapshot.parts else "部件-1"
        )
        part = NativePart(
            id="P1",
            name=part_name,
            geometry_recipe=recipe,
            mesh_settings=_migrate_part_mesh_settings(
                snapshot.mesh_settings,
                "P1",
            ),
            body_name=(
                snapshot.parts[0].body_name
                if snapshot.parts
                else "Body-1"
            ),
        )
        regions = tuple(
            _migrate_region_to_part(
                region,
                lambda logical_id: namespace_part_logical_id(
                    "P1",
                    logical_id,
                ),
            )
            for region in snapshot.named_regions
        )
        migrated = replace(
            snapshot,
            parts=(part,),
            geometry_recipe=part.geometry_recipe,
            mesh_settings=part.mesh_settings,
            feature_history=part.feature_history,
            named_regions=regions,
            boolean_reference_undo_records=(),
            active_part_id="P1",
        )
    return (
        migrated,
        (
            ProjectMigrationNotice(
                "project.schema.v7.native_parts",
                "旧项目几何已迁移为稳定的原生部件所有权。",
                "$.project.authoring.parts",
            ),
        ),
    )


def _migrate_multi_body_snapshot_to_v7(
    snapshot: ProjectSnapshot,
    geometry: MultiBodyGeometry,
) -> ProjectSnapshot:
    strict_results = tuple(
        body
        for body in geometry.bodies
        if isinstance(body.recipe, BooleanGeometry)
        and body.recipe.body_context is not None
    )
    if not strict_results:
        parts = tuple(
            NativePart(
                id=f"P{int(body.id[1:])}",
                name=body.name,
                geometry_recipe=body.recipe,
                mesh_settings=_migrate_body_mesh_settings(
                    snapshot.mesh_settings,
                    body.id,
                    f"P{int(body.id[1:])}",
                ),
            )
            for body in geometry.bodies
        )
        regions = tuple(
            _migrate_region_to_part(
                region,
                _legacy_body_ref_to_part_ref,
            )
            for region in snapshot.named_regions
        )
        active = parts[0]
        return replace(
            snapshot,
            parts=validate_native_parts(parts),
            geometry_recipe=active.geometry_recipe,
            mesh_settings=active.mesh_settings,
            feature_history=active.feature_history,
            named_regions=regions,
            boolean_reference_undo_records=(),
            active_part_id=active.id,
            retired_part_ids=tuple(
                f"P{int(value[1:])}"
                for value in geometry.retired_body_ids
            ),
        )
    if len(strict_results) != 1:
        raise ProjectV1MigrationError(
            "schema v7 migration currently requires a unique active strict "
            "Body Boolean result"
        )
    return _migrate_strict_body_boolean_graph(
        snapshot,
        geometry,
        strict_results[0],
    )


def _migrate_strict_body_boolean_graph(
    snapshot: ProjectSnapshot,
    geometry: MultiBodyGeometry,
    active_result_body: Any,
) -> ProjectSnapshot:
    """Promote every persisted strict Body Boolean record into one Part DAG."""

    legacy_records = tuple(
        record
        for record in snapshot.boolean_reference_undo_records
        if str(record.feature_id).startswith("BF")
    )
    if not legacy_records:
        raise ProjectV1MigrationError(
            "active Body Boolean lacks its exact undo source snapshot"
        )
    records_by_feature = {
        record.feature_id: record for record in legacy_records
    }
    if len(records_by_feature) != len(legacy_records):
        raise ProjectV1MigrationError(
            "Body Boolean undo feature IDs are not unique"
        )
    ordered = tuple(
        sorted(
            legacy_records,
            key=lambda record: int(record.feature_id[2:]),
        )
    )
    result_body_by_feature: dict[str, Any] = {}
    for record in ordered:
        if not isinstance(record.after_geometry, MultiBodyGeometry):
            raise ProjectV1MigrationError(
                f"{record.feature_id} after_geometry is not MultiBodyGeometry"
            )
        matches = tuple(
            body
            for body in record.after_geometry.bodies
            if isinstance(body.recipe, BooleanGeometry)
            and body.recipe.body_context is not None
            and body.recipe.body_context.feature_id == record.feature_id
        )
        if len(matches) != 1:
            raise ProjectV1MigrationError(
                f"{record.feature_id} lacks one exact result Body"
            )
        result_body_by_feature[record.feature_id] = matches[0]

    active_context = active_result_body.recipe.body_context
    if (
        active_context is None
        or active_context.feature_id not in result_body_by_feature
    ):
        raise ProjectV1MigrationError(
            "active Body Boolean has no matching undo record"
        )

    parts_by_id: dict[str, NativePart] = {}
    migrated_records: list[PartBooleanUndoRecord] = []
    result_part_by_body_id: dict[str, NativePart] = {}
    source_part_ids: set[str] = set()
    next_result_number = (
        max(
            (
                *(
                    int(body.id[1:])
                    for record in ordered
                    for state in (
                        record.before_geometry,
                        record.after_geometry,
                    )
                    if isinstance(state, MultiBodyGeometry)
                    for body in state.bodies
                ),
                *(int(value[1:]) for value in geometry.retired_body_ids),
            ),
            default=0,
        )
        + 1
    )

    def part_id(body_id: str) -> str:
        return f"P{int(body_id[1:])}"

    def source_part(body: Any, settings: MeshSettings | None) -> NativePart:
        prior = result_part_by_body_id.get(body.id)
        if prior is not None:
            provenance = prior.provenance
            if provenance is None:
                raise ProjectV1MigrationError(
                    f"legacy Boolean result {body.id} lacks provenance"
                )
            legacy_feature_id = f"BF{int(provenance.feature_id[3:])}"
            original_result = result_body_by_feature[legacy_feature_id]
            current = replace(
                prior,
                geometry_recipe=_promote_legacy_result_state(
                    body.recipe,
                    original_result.recipe,
                    prior.geometry_recipe,
                ),
                mesh_settings=_migrate_body_mesh_settings(
                    settings,
                    body.id,
                    prior.id,
                ),
            )
            parts_by_id[current.id] = current
            result_part_by_body_id[body.id] = current
            return current
        resolved_id = part_id(body.id)
        candidate = NativePart(
            id=resolved_id,
            name=body.name,
            geometry_recipe=body.recipe,
            mesh_settings=_migrate_body_mesh_settings(
                settings,
                body.id,
                resolved_id,
            ),
        )
        existing = parts_by_id.get(resolved_id)
        if existing is not None and existing != candidate:
            raise ProjectV1MigrationError(
                f"legacy Body {body.id} has conflicting exact states"
            )
        parts_by_id[resolved_id] = candidate
        return candidate

    for legacy_record in ordered:
        prior_result_part_by_body_id = dict(result_part_by_body_id)
        result_body = result_body_by_feature[legacy_record.feature_id]
        raw_context = result_body.recipe.body_context
        if raw_context is None or not raw_context.proven:
            raise ProjectV1MigrationError(
                f"{legacy_record.feature_id} lacks exact lineage proof"
            )
        if not isinstance(legacy_record.before_geometry, MultiBodyGeometry):
            raise ProjectV1MigrationError(
                f"{legacy_record.feature_id} lacks exact source geometry"
            )
        before_by_id = {
            body.id: body
            for body in legacy_record.before_geometry.bodies
        }
        try:
            target_body = before_by_id[raw_context.target_body_id]
            tool_body = before_by_id[raw_context.tool_body_id]
        except KeyError as error:
            raise ProjectV1MigrationError(
                f"{legacy_record.feature_id} source Body is missing"
            ) from error
        target_source = source_part(
            target_body,
            legacy_record.before_mesh_settings,
        )
        tool_source = source_part(
            tool_body,
            legacy_record.before_mesh_settings,
        )
        result_id = f"P{next_result_number}"
        next_result_number += 1
        feature_id = f"PBF{int(legacy_record.feature_id[2:])}"
        part_context = namespace_part_boolean_context(
            feature_id=feature_id,
            target_part_id=target_source.id,
            tool_part_id=tool_source.id,
            result_part_id=result_id,
            result_entities=raw_context.result_entities,
            topology_mappings=raw_context.topology_mappings,
        )
        result_recipe = replace(
            result_body.recipe,
            object_geometry=target_source.geometry_recipe,
            tool_geometry=tool_source.geometry_recipe,
            body_context=None,
            part_context=part_context,
        )
        result_name = (
            f"{'合并' if result_body.recipe.operation == 'fuse' else '切除'}"
            f"结果-{int(legacy_record.feature_id[2:])}"
        )
        result_part = NativePart(
            id=result_id,
            name=result_name,
            geometry_recipe=result_recipe,
            mesh_settings=_migrate_body_mesh_settings(
                legacy_record.after_mesh_settings or snapshot.mesh_settings,
                result_body.id,
                result_id,
            ),
            provenance=PartBooleanProvenance(
                feature_id,
                target_source.id,
                tool_source.id,
                result_body.recipe.operation,
            ),
        )
        existing_result = parts_by_id.get(result_id)
        if existing_result is not None and existing_result != result_part:
            raise ProjectV1MigrationError(
                f"legacy Boolean result {result_body.id} conflicts with a Part"
            )
        parts_by_id[result_id] = result_part
        result_part_by_body_id[result_body.id] = result_part
        source_part_ids.update((target_source.id, tool_source.id))

        before_owner_map = {
            body.id: (
                prior_result_part_by_body_id[body.id].id
                if body.id in prior_result_part_by_body_id
                else part_id(body.id)
            )
            for body in legacy_record.before_geometry.bodies
        }
        after_owner_map = {
            body.id: (
                result_id
                if body.id == result_body.id
                else part_id(body.id)
            )
            for body in legacy_record.after_geometry.bodies
        }
        after_owner_map[raw_context.target_body_id] = result_id
        before_regions = tuple(
            _migrate_region_to_part(
                region,
                lambda logical_id, owner_map=before_owner_map: (
                    _legacy_body_ref_to_part_ref(
                        logical_id,
                        owner_map=owner_map,
                    )
                ),
            )
            for region in legacy_record.before_named_regions
        )
        after_regions = tuple(
            _migrate_region_to_part(
                region,
                lambda logical_id, owner_map=after_owner_map: (
                    _legacy_body_ref_to_part_ref(
                        logical_id,
                        owner_map=owner_map,
                    )
                ),
            )
            for region in legacy_record.after_named_regions
        )
        migrated_records.append(
            PartBooleanUndoRecord(
                feature_id,
                result_id,
                (target_source, tool_source),
                result_part,
                before_regions,
                after_regions,
                legacy_record.before_assignments,
                legacy_record.after_assignments,
                legacy_record.before_steps,
                legacy_record.after_steps,
            )
        )

    for body in geometry.bodies:
        if body.id in result_part_by_body_id:
            continue
        source_part(body, snapshot.mesh_settings)

    active_result = result_part_by_body_id.get(active_result_body.id)
    if active_result is None:
        raise ProjectV1MigrationError(
            "active Body Boolean result was not promoted"
        )
    parts = validate_native_parts(
        tuple(
            sorted(
                (
                    replace(
                        part,
                        suppressed=part.id in source_part_ids,
                    )
                    for part in parts_by_id.values()
                ),
                key=lambda part: int(part.id[1:]),
            )
        )
    )
    active_result = next(part for part in parts if part.id == active_result.id)
    final_owner_map = {
        body.id: (
            result_part_by_body_id[body.id].id
            if body.id in result_part_by_body_id
            else part_id(body.id)
        )
        for body in geometry.bodies
    }
    final_context = active_result_body.recipe.body_context
    if final_context is not None:
        final_owner_map[final_context.target_body_id] = active_result.id
    regions = tuple(
        _migrate_region_to_part(
            region,
            lambda logical_id: _legacy_body_ref_to_part_ref(
                logical_id,
                owner_map=final_owner_map,
            ),
        )
        for region in snapshot.named_regions
    )
    active_ids = {part.id for part in parts}
    active_feature_ids = {
        record.feature_id for record in migrated_records
    }
    return replace(
        snapshot,
        parts=parts,
        geometry_recipe=active_result.geometry_recipe,
        mesh_settings=active_result.mesh_settings,
        feature_history=active_result.feature_history,
        named_regions=regions,
        boolean_reference_undo_records=(),
        part_boolean_undo_records=tuple(migrated_records),
        retired_part_ids=tuple(
            f"P{int(value[1:])}"
            for value in geometry.retired_body_ids
            if f"P{int(value[1:])}" not in active_ids
        ),
        retired_part_boolean_feature_ids=tuple(
            f"PBF{int(value[2:])}"
            for value in geometry.retired_boolean_feature_ids
            if f"PBF{int(value[2:])}" not in active_feature_ids
        ),
        active_part_id=active_result.id,
    )


def _promote_legacy_result_state(
    current: Any,
    legacy_result: Any,
    canonical_result: Any,
) -> Any:
    """Retain later local features while replacing legacy Boolean identity."""

    if current == legacy_result:
        return canonical_result
    if isinstance(current, (MovedGeometry, RotatedGeometry, ExtrudedGeometry)):
        return replace(
            current,
            base=_promote_legacy_result_state(
                current.base,
                legacy_result,
                canonical_result,
            ),
        )
    raise ProjectV1MigrationError(
        "legacy Boolean result state is not derived from its exact record"
    )


def _migrate_part_mesh_settings(
    settings: MeshSettings | None,
    part_id: str,
) -> MeshSettings | None:
    if settings is None:
        return None
    return replace(
        settings,
        local_controls=tuple(
            replace(
                control,
                target=LogicalEntityRef(
                    namespace_part_logical_id(
                        part_id,
                        control.target.logical_id,
                    )
                ),
            )
            for control in settings.local_controls
        ),
    )


def _migrate_body_mesh_settings(
    settings: MeshSettings | None,
    body_id: str,
    part_id: str,
) -> MeshSettings | None:
    if settings is None:
        return None
    controls = []
    for control in settings.local_controls:
        logical_id = control.target.logical_id
        reference = LogicalEntityRef(logical_id)
        semantic = logical_id.split(":", 1)[1]
        owner, separator, local_name = semantic.partition("/")
        if separator != "/" or owner != body_id:
            continue
        if reference.kind == "body":
            local_id = "body:domain"
        else:
            local_id = f"{reference.kind}:{local_name}"
        controls.append(
            replace(
                control,
                target=LogicalEntityRef(
                    namespace_part_logical_id(part_id, local_id)
                ),
            )
        )
    return replace(settings, local_controls=tuple(controls))


def _migrate_region_to_part(
    region: NamedRegion,
    mapper,
) -> NamedRegion:
    return replace(
        region,
        references=tuple(
            LogicalEntityRef(mapper(reference.logical_id))
            if type(reference) is LogicalEntityRef
            else reference
            for reference in region.references
        ),
    )


def _legacy_body_ref_to_part_ref(
    logical_id: str,
    *,
    owner_map: dict[str, str] | None = None,
) -> str:
    reference = LogicalEntityRef(logical_id)
    semantic = logical_id.split(":", 1)[1]
    body_id, separator, local_name = semantic.partition("/")
    if reference.kind == "body" and separator != "/":
        body_id = semantic
        local_name = "domain"
        separator = "/"
    if separator != "/" or not body_id.startswith("B"):
        raise ProjectV1MigrationError(
            f"legacy MultiBody reference lacks Body namespace: {logical_id!r}"
        )
    part_id = (
        owner_map[body_id]
        if owner_map is not None
        else f"P{int(body_id[1:])}"
    )
    local_id = (
        "body:domain"
        if reference.kind == "body"
        else f"{reference.kind}:{local_name}"
    )
    return namespace_part_logical_id(part_id, local_id)
