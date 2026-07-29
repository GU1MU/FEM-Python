"""Atomic preparation of strict Booleans between committed solid Bodies."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Literal

from fem.geometry.body_operations import (
    install_proven_body_boolean,
    provisional_body_boolean,
)
from fem.geometry.recipes import (
    BooleanGeometry,
    ExtrudedGeometry,
    MovedGeometry,
    MultiBodyGeometry,
    RevolvedGeometry,
    RotatedGeometry,
)
from fem.geometry.solid_boolean_lineage import (
    BooleanLineageProof,
    BooleanLineageResolutionError,
    capture_boolean_operand_evidence,
    resolve_solid_boolean_lineage,
    validate_solid_boolean_input_map,
)
from fem.geometry.types import StrictBodyBooleanPreview

from .recipe_compiler import compile_recipe


@dataclass(frozen=True, slots=True)
class StrictBodyBooleanResult:
    """Proven recipe edit and one-session OCC lineage evidence."""

    geometry: MultiBodyGeometry
    recipe: BooleanGeometry
    proof: BooleanLineageProof
    preview: StrictBodyBooleanPreview


def prepare_solid_body_boolean(
    cad: Any,
    geometry: MultiBodyGeometry,
    target_body_id: str,
    tool_body_id: str,
    operation: Literal["fuse", "cut"],
) -> StrictBodyBooleanResult:
    """Execute and prove one Boolean without mutating the live Session."""

    if type(geometry) is not MultiBodyGeometry:
        raise TypeError("geometry must be a MultiBodyGeometry")
    draft = provisional_body_boolean(
        geometry,
        target_body_id,
        tool_body_id,
        operation,
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
    context = draft.body_context
    if context is None:
        raise RuntimeError("strict Body Boolean draft lost its context")
    proof = resolve_solid_boolean_lineage(
        cad,
        target_evidence,
        tool_evidence,
        result,
        boundary,
        context,
        operation=operation,
    )
    _require_no_unaffected_overlap(
        cad,
        geometry,
        proof.result_volume,
        excluded={target_body_id, tool_body_id},
    )
    preview = _tessellate_boolean_proof(
        cad,
        proof,
        target_body_id=target_body_id,
    )
    proven_context = replace(
        context,
        result_entities=proof.result_entities,
        topology_mappings=proof.topology_mappings,
    )
    proven_recipe = replace(draft, body_context=proven_context)
    updated = install_proven_body_boolean(geometry, proven_recipe)
    return StrictBodyBooleanResult(updated, proven_recipe, proof, preview)


def _tessellate_boolean_proof(
    cad: Any,
    proof: BooleanLineageProof,
    *,
    target_body_id: str,
) -> StrictBodyBooleanPreview:
    return _tessellate_logical_entities(
        cad,
        proof.logical_entities,
        target_body_id=target_body_id,
    )


def prepare_strict_body_recipe_preview(
    cad: Any,
    target_body_id: str,
    recipe: Any,
) -> StrictBodyBooleanPreview:
    """Replay persisted strict proof and detach its true OCC tessellation."""

    context = _persisted_strict_context(recipe)
    if context is None:
        raise TypeError("recipe must contain a strict BooleanGeometry")
    if not context.proven:
        raise BooleanLineageResolutionError(
            "boolean.lineage.unproven: persisted preview requires proof"
        )
    compiled = compile_recipe(cad, recipe)
    return _tessellate_logical_entities(
        cad,
        compiled.logical_entities,
        target_body_id=target_body_id,
    )


def _persisted_strict_context(recipe: Any):
    if isinstance(recipe, BooleanGeometry):
        if recipe.body_context is not None:
            return recipe.body_context
        if recipe.planar_context is not None:
            return recipe.planar_context
        return (
            _persisted_strict_context(recipe.object_geometry)
            or _persisted_strict_context(recipe.tool_geometry)
        )
    if isinstance(
        recipe,
        (MovedGeometry, RotatedGeometry, ExtrudedGeometry, RevolvedGeometry),
    ):
        return _persisted_strict_context(recipe.base)
    return None


def _tessellate_logical_entities(
    cad: Any,
    logical_entities: Any,
    *,
    target_body_id: str,
) -> StrictBodyBooleanPreview:
    by_entity: dict[Any, str] = {}
    for logical_id, entities in logical_entities.items():
        if logical_id.startswith("body:"):
            continue
        for entity in entities:
            previous = by_entity.setdefault(entity, logical_id)
            if previous != logical_id:
                raise BooleanLineageResolutionError(
                    "boolean.preview.entity-ambiguous: one OCC entity has "
                    "multiple logical identities"
                )
    faces = tuple(
        entity for entity in by_entity if entity.dimension == 2
    )
    edges = tuple(
        entity for entity in by_entity if entity.dimension == 1
    )
    points = tuple(
        entity for entity in by_entity if entity.dimension == 0
    )
    tessellation = cad.tessellate_surfaces(faces, edges, points)
    return StrictBodyBooleanPreview(
        target_body_id,
        tessellation.points,
        tessellation.faces,
        tessellation.edges,
        tuple(by_entity[entity] for entity in tessellation.face_entities),
        tuple(by_entity[entity] for entity in tessellation.edge_entities),
        tuple(
            None if entity is None else by_entity[entity]
            for entity in tessellation.point_entities
        ),
    )


def _require_no_unaffected_overlap(
    cad: Any,
    geometry: MultiBodyGeometry,
    result_volume: Any,
    *,
    excluded: set[str],
) -> None:
    tolerance = (
        float(cad.effective_bounding_box_tolerance(1.0e-9))
        if hasattr(cad, "effective_bounding_box_tolerance")
        else 1.0e-9
    )
    for body in geometry.bodies:
        if body.id in excluded:
            continue
        compiled = compile_recipe(cad, body.recipe)
        if len(compiled.domain) != 1:
            raise BooleanLineageResolutionError(
                f"boolean.unaffected.single-volume: Body {body.id} "
                "did not compile to one volume"
            )
        if float(cad.distance(result_volume, compiled.domain[0])) <= tolerance:
            raise BooleanLineageResolutionError(
                "boolean.unaffected.overlap: result touches or overlaps "
                f"unaffected Body {body.id}"
            )


__all__ = [
    "StrictBodyBooleanPreview",
    "StrictBodyBooleanResult",
    "prepare_solid_body_boolean",
    "prepare_strict_body_recipe_preview",
]
