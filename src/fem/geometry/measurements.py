"""Pure geometric measures proven from native recipe topology lineage."""

from __future__ import annotations

from .recipe_topology import (
    RecipeTopology,
    TopologyMapping,
    describe_recipe_topology,
)
from .recipe_analysis import analyze_sketch_profiles
from .references import LogicalEntityRef
from .recipes import (
    BooleanGeometry,
    ExtrudedGeometry,
    MovedGeometry,
    NATIVE_GEOMETRY_TYPES,
    NativeGeometry,
    PlateWithHoleGeometry,
    RotatedGeometry,
    SketchCircle,
    SketchGeometry,
    WireGeometry,
)


class TargetRadiusResolutionError(ValueError):
    """A target cannot be proven to represent one circular hole radius."""


def resolve_target_radius(
    recipe: NativeGeometry,
    target: LogicalEntityRef,
) -> float:
    """Return the radius proven for one circular hole edge or swept side."""

    topology = _validated_topology(recipe, target)
    entity = topology.entity(target.logical_id)

    if _contains_wire(recipe):
        raise TargetRadiusResolutionError(
            "target_radius falloff is not supported for wire point or member "
            "targets"
        )

    if isinstance(recipe, PlateWithHoleGeometry):
        if entity.semantic_role == "boundary.hole-loop" and entity.kind == "edge":
            return recipe.hole_radius
    elif isinstance(recipe, SketchGeometry):
        if (
            topology.transition.operation == "sketch.cut-contained-circle"
            and entity.semantic_role == "boundary.hole-loop"
            and entity.kind == "edge"
        ):
            circles = tuple(
                contour
                for contour in recipe.contours
                if contour.operation == "cut" and isinstance(contour, SketchCircle)
            )
            if len(circles) == 1:
                return circles[0].radius
        if recipe.is_strict and entity.kind == "edge":
            analysis = analyze_sketch_profiles(recipe)
            if entity.semantic_role == "boundary.hole-loop":
                circles = tuple(
                    curve
                    for profile in analysis.profiles
                    if profile.is_hole
                    for curve in recipe.curves
                    if isinstance(curve, SketchCircle)
                    and curve.id in {item.lstrip("-") for item in profile.curve_ids}
                )
                if len(circles) == 1:
                    return circles[0].radius
            curve_id = entity.logical_id.split(":", 1)[1]
            curve = recipe.curve(curve_id)
            if isinstance(curve, SketchCircle) and any(
                profile.is_hole
                and curve_id in {item.lstrip("-") for item in profile.curve_ids}
                for profile in analysis.profiles
            ):
                return curve.radius
    elif isinstance(recipe, BooleanGeometry):
        if topology.transition.operation == "boolean.cut-contained-circle":
            mapping = _unique_source_mapping(topology, target, "tool")
            if mapping is not None:
                return _resolve_circular_source_radius(
                    recipe.tool_geometry,
                    LogicalEntityRef(mapping.source_logical_id),
                )
    elif isinstance(recipe, (MovedGeometry, RotatedGeometry)):
        mapping = _unique_source_mapping(topology, target, "base")
        if mapping is not None:
            return resolve_target_radius(
                recipe.base,
                LogicalEntityRef(mapping.source_logical_id),
            )
    elif isinstance(recipe, ExtrudedGeometry):
        mapping = _unique_source_mapping(topology, target, "base")
        if mapping is not None:
            return resolve_target_radius(
                recipe.base,
                LogicalEntityRef(mapping.source_logical_id),
            )

    raise TargetRadiusResolutionError(
        f"logical target {target.logical_id!r} has no proven circular-hole radius"
    )


def resolve_legacy_hole_target(recipe: NativeGeometry) -> LogicalEntityRef:
    """Discover the unique target represented by the legacy hole local size."""

    if not isinstance(recipe, NATIVE_GEOMETRY_TYPES):
        raise TypeError(f"unsupported native geometry recipe: {type(recipe).__name__}")
    topology = describe_recipe_topology(recipe)
    if not topology.exact:
        raise TargetRadiusResolutionError(
            "legacy hole target requires an exact recipe topology"
        )

    target: LogicalEntityRef | None = None
    if isinstance(recipe, PlateWithHoleGeometry):
        target = _unique_entity_with_role(
            topology,
            kind="edge",
            semantic_role="boundary.hole-loop",
        )
    elif isinstance(recipe, SketchGeometry):
        if topology.transition.operation == "sketch.cut-contained-circle":
            target = _unique_entity_with_role(
                topology,
                kind="edge",
                semantic_role="boundary.hole-loop",
            )
        if recipe.is_strict:
            analysis = analyze_sketch_profiles(recipe)
            candidates = tuple(
                curve.id
                for profile in analysis.profiles
                if profile.is_hole
                for curve in recipe.curves
                if isinstance(curve, SketchCircle)
                and curve.id in {item.lstrip("-") for item in profile.curve_ids}
            )
            if len(candidates) == 1:
                target = LogicalEntityRef(f"edge:{candidates[0]}")
    elif isinstance(recipe, BooleanGeometry):
        if topology.transition.operation == "boolean.cut-contained-circle":
            candidates = tuple(
                LogicalEntityRef(mapping.target_logical_id)
                for mapping in topology.transition.mappings
                if mapping.source == "tool"
                and mapping.source_logical_id == "edge:outer"
                and mapping.relation == "derived"
            )
            if len(candidates) == 1:
                target = candidates[0]
    elif isinstance(recipe, (MovedGeometry, RotatedGeometry)):
        base_target = resolve_legacy_hole_target(recipe.base)
        target = _unique_target_mapping(
            topology,
            source="base",
            source_logical_id=base_target.logical_id,
            target_kind=base_target.kind,
        )
    elif isinstance(recipe, ExtrudedGeometry):
        base_target = resolve_legacy_hole_target(recipe.base)
        target = _unique_target_mapping(
            topology,
            source="base",
            source_logical_id=base_target.logical_id,
            target_kind="face",
        )

    if target is None:
        raise TargetRadiusResolutionError(
            f"{type(recipe).__name__} has no unique proven legacy circular hole"
        )
    resolve_target_radius(recipe, target)
    return target


def _validated_topology(
    recipe: NativeGeometry,
    target: LogicalEntityRef,
) -> RecipeTopology:
    if not isinstance(recipe, NATIVE_GEOMETRY_TYPES):
        raise TypeError(f"unsupported native geometry recipe: {type(recipe).__name__}")
    if type(target) is not LogicalEntityRef:
        raise TypeError("target must be a LogicalEntityRef")
    topology = describe_recipe_topology(recipe)
    if not topology.exact:
        raise TargetRadiusResolutionError(
            "target radius requires an exact recipe topology"
        )
    try:
        entity = topology.entity(target.logical_id)
    except KeyError as error:
        raise TargetRadiusResolutionError(
            f"unknown logical target {target.logical_id!r}"
        ) from error
    if entity.kind != target.kind:
        raise TargetRadiusResolutionError(
            f"logical target kind does not match {target.logical_id!r}"
        )
    if not entity.selectable:
        raise TargetRadiusResolutionError(
            f"logical target {target.logical_id!r} is not selectable"
        )
    return topology


def _resolve_circular_source_radius(
    recipe: NativeGeometry,
    target: LogicalEntityRef,
) -> float:
    """Resolve a circular Boolean source without making it a public target."""

    topology = _validated_topology(recipe, target)
    entity = topology.entity(target.logical_id)
    from .recipes import DiskGeometry

    if (
        isinstance(recipe, DiskGeometry)
        and topology.transition.operation == "primitive.disk"
        and entity.semantic_role == "boundary.outer"
        and entity.kind == "edge"
    ):
        return recipe.radius
    if isinstance(recipe, (MovedGeometry, RotatedGeometry)):
        mapping = _unique_source_mapping(topology, target, "base")
        if mapping is not None:
            return _resolve_circular_source_radius(
                recipe.base,
                LogicalEntityRef(mapping.source_logical_id),
            )
    raise TargetRadiusResolutionError(
        f"logical source {target.logical_id!r} is not a proven circle"
    )


def _unique_source_mapping(
    topology: RecipeTopology,
    target: LogicalEntityRef,
    source: str,
) -> TopologyMapping | None:
    mappings = tuple(
        mapping
        for mapping in topology.transition.mappings
        if mapping.source == source and mapping.target_logical_id == target.logical_id
    )
    return mappings[0] if len(mappings) == 1 else None


def _unique_target_mapping(
    topology: RecipeTopology,
    *,
    source: str,
    source_logical_id: str,
    target_kind: str,
) -> LogicalEntityRef | None:
    candidates = tuple(
        LogicalEntityRef(mapping.target_logical_id)
        for mapping in topology.transition.mappings
        if mapping.source == source
        and mapping.source_logical_id == source_logical_id
        and LogicalEntityRef(mapping.target_logical_id).kind == target_kind
    )
    return candidates[0] if len(candidates) == 1 else None


def _unique_entity_with_role(
    topology: RecipeTopology,
    *,
    kind: str,
    semantic_role: str,
) -> LogicalEntityRef | None:
    candidates = tuple(
        LogicalEntityRef(entity.logical_id)
        for entity in topology.entities
        if entity.kind == kind
        and entity.semantic_role == semantic_role
        and entity.selectable
    )
    return candidates[0] if len(candidates) == 1 else None


def _contains_wire(recipe: NativeGeometry) -> bool:
    if isinstance(recipe, WireGeometry):
        return True
    if isinstance(recipe, (MovedGeometry, RotatedGeometry)):
        return _contains_wire(recipe.base)
    return False


__all__ = [
    "TargetRadiusResolutionError",
    "resolve_legacy_hole_target",
    "resolve_target_radius",
]
