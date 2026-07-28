"""Typed compatibility migration from frozen ``.femproj`` schema v1."""

from __future__ import annotations

from collections.abc import Iterable
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
from fem.core.model import (
    AnalysisStep,
    MaterialDefinition,
)
from fem.geometry.measurements import (
    TargetRadiusResolutionError,
    resolve_legacy_hole_target,
)
from fem.geometry.recipe_analysis import legacy_sketch_to_strict
from fem.geometry.recipe_topology import describe_recipe_topology
from fem.geometry.references import LogicalEntityRef
from fem.geometry.recipes import SketchGeometry
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
    geometry_recipe = (
        legacy_sketch_to_strict(source_geometry)
        if type(source_geometry) is SketchGeometry and source_geometry.is_legacy
        else source_geometry
    )

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
                "下次显式保存将升级为 schema 3（v3）当前项目格式"
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
    "migrate_project_v1",
]
