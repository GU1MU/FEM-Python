"""Detached OCC preparation for strict Boolean operations between Parts."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Literal

from fem.geometry.part_boolean import namespace_part_boolean_context
from fem.geometry.part_namespace import namespace_part_logical_id
from fem.geometry.recipes import (
    BooleanBodyContext,
    BooleanGeometry,
    ExtrudedGeometry,
    MovedGeometry,
    PartBooleanContext,
    RevolvedGeometry,
    RotatedGeometry,
)
from fem.geometry.solid_boolean_lineage import (
    BooleanLineageResolutionError,
    BooleanLineageProof,
    capture_boolean_operand_evidence,
    resolve_solid_boolean_lineage,
    validate_solid_boolean_input_map,
)
from fem.geometry.types import StrictBodyBooleanPreview

from .native_part import NativePart
from .recipe_compiler import compile_recipe


@dataclass(frozen=True, slots=True)
class StrictPartBooleanResult:
    """One proven detached result recipe and its true OCC preview."""

    recipe: BooleanGeometry
    context: PartBooleanContext
    proof: BooleanLineageProof
    preview: StrictBodyBooleanPreview


def prepare_part_boolean(
    cad: Any,
    target: NativePart,
    tool: NativePart,
    operation: Literal["fuse", "cut"],
    *,
    result_part_id: str,
    feature_id: str,
    result_name: str,
) -> StrictPartBooleanResult:
    """Execute and prove a cross-Part Boolean without mutating Session state."""

    if type(target) is not NativePart or type(tool) is not NativePart:
        raise TypeError("target and tool must be NativePart values")
    if target.id == tool.id:
        raise ValueError("target and tool Parts must differ")
    for role, part in (("target", target), ("tool", tool)):
        if part.suppressed:
            raise ValueError(f"{role} Part is suppressed")
        if part.dimension != 3 or part.geometry_recipe is None:
            raise ValueError(f"{role} Part must own exact 3D geometry")
    if operation not in {"fuse", "cut"}:
        raise ValueError("Part Boolean operation must be fuse or cut")

    local_context = BooleanBodyContext(
        f"BF{feature_id[3:]}",
        "B1",
        "B2",
        tool.name,
    )
    draft = BooleanGeometry(
        str(result_name).strip(),
        operation,
        target.geometry_recipe,
        tool.geometry_recipe,
        body_context=local_context,
    )
    target_compiled = compile_recipe(cad, draft.object_geometry)
    tool_compiled = compile_recipe(cad, draft.tool_geometry)
    target_evidence = capture_boolean_operand_evidence(cad, target_compiled)
    tool_evidence = capture_boolean_operand_evidence(cad, tool_compiled)
    boolean_operation = cad.fuse if operation == "fuse" else cad.cut
    result = boolean_operation(
        target_compiled.domain,
        tool_compiled.domain,
    )
    validate_solid_boolean_input_map(result)
    volumes = result.of_dimension(3)
    boundary = (
        ()
        if len(volumes) != 1
        else tuple(cad.boundary(volumes, combined=False))
    )
    proof = resolve_solid_boolean_lineage(
        cad,
        target_evidence,
        tool_evidence,
        result,
        boundary,
        local_context,
        operation=operation,
    )
    context = namespace_part_boolean_context(
        feature_id=feature_id,
        target_part_id=target.id,
        tool_part_id=tool.id,
        result_part_id=result_part_id,
        result_entities=proof.result_entities,
        topology_mappings=proof.topology_mappings,
    )
    recipe = replace(
        draft,
        body_context=None,
        part_context=context,
    )
    preview = _part_boolean_preview(
        cad,
        proof,
        result_part_id=result_part_id,
    )
    return StrictPartBooleanResult(recipe, context, proof, preview)


def _part_boolean_preview(
    cad: Any,
    proof: BooleanLineageProof,
    *,
    result_part_id: str,
) -> StrictBodyBooleanPreview:
    return _part_boolean_preview_from_entities(
        cad,
        proof.logical_entities,
        result_part_id=result_part_id,
    )


def prepare_strict_part_recipe_preview(
    cad: Any,
    result_part_id: str,
    recipe: Any,
) -> StrictBodyBooleanPreview:
    """Replay a persisted Part Boolean and detach its exact OCC tessellation."""

    context = _persisted_part_context(recipe)
    if context is None:
        raise TypeError("recipe must contain a strict Part Boolean")
    if not context.proven or context.result_part_id != result_part_id:
        raise BooleanLineageResolutionError(
            "boolean.part.preview.unproven: persisted Part proof is incomplete "
            "or belongs to another result Part"
        )
    compiled = compile_recipe(cad, recipe)
    return _part_boolean_preview_from_entities(
        cad,
        compiled.logical_entities,
        result_part_id=result_part_id,
    )


def _persisted_part_context(recipe: Any) -> PartBooleanContext | None:
    if isinstance(recipe, BooleanGeometry):
        if recipe.part_context is not None:
            return recipe.part_context
        return _persisted_part_context(
            recipe.object_geometry
        ) or _persisted_part_context(recipe.tool_geometry)
    if isinstance(
        recipe,
        (MovedGeometry, RotatedGeometry, ExtrudedGeometry, RevolvedGeometry),
    ):
        return _persisted_part_context(recipe.base)
    return None


def _part_boolean_preview_from_entities(
    cad: Any,
    logical_entities: Any,
    *,
    result_part_id: str,
) -> StrictBodyBooleanPreview:
    by_entity: dict[Any, str] = {}
    for local_id, entities in logical_entities.items():
        if local_id.startswith("body:"):
            continue
        logical_id = namespace_part_logical_id(result_part_id, local_id)
        for entity in entities:
            previous = by_entity.setdefault(entity, logical_id)
            if previous != logical_id:
                raise ValueError(
                    "boolean.part.preview-ambiguous: one OCC entity has "
                    "multiple logical identities"
                )
    faces = tuple(entity for entity in by_entity if entity.dimension == 2)
    edges = tuple(entity for entity in by_entity if entity.dimension == 1)
    points = tuple(entity for entity in by_entity if entity.dimension == 0)
    tessellation = cad.tessellate_surfaces(faces, edges, points)
    return StrictBodyBooleanPreview(
        result_part_id,
        tessellation.points,
        tessellation.faces,
        tessellation.edges,
        tuple(by_entity[item] for item in tessellation.face_entities),
        tuple(by_entity[item] for item in tessellation.edge_entities),
        tuple(
            None if item is None else by_entity[item]
            for item in tessellation.point_entities
        ),
    )


__all__ = [
    "StrictPartBooleanResult",
    "prepare_part_boolean",
    "prepare_strict_part_recipe_preview",
]
