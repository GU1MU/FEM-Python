"""Atomic preparation of strict planar Boolean features."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Literal

from fem.geometry.planar_boolean_lineage import (
    PlanarBooleanLineageProof,
    PlanarBooleanLineageResolutionError,
    capture_planar_operand_evidence,
    resolve_planar_boolean_lineage,
    validate_planar_boolean_input_map,
)
from fem.geometry.planar_boolean_selection import resolve_planar_boolean_faces
from fem.geometry.recipes import (
    BooleanGeometry,
    ExtrudedGeometry,
    MovedGeometry,
    PlanarBooleanContext,
    RotatedGeometry,
)
from fem.geometry.types import StrictPlanarBooleanPreview

from .recipe_compiler import (
    _planar_unaffected_logical_entities,
    compile_recipe,
)


@dataclass(frozen=True, slots=True)
class StrictPlanarBooleanResult:
    """Proven detached planar recipe and true OCC preview."""

    geometry: BooleanGeometry
    proof: PlanarBooleanLineageProof
    preview: StrictPlanarBooleanPreview


def prepare_planar_boolean(
    cad: Any,
    object_geometry: object,
    target_face_id: str,
    tool_geometry: object,
    tool_face_ids: tuple[str, ...],
    operation: Literal["fuse", "cut"],
) -> StrictPlanarBooleanResult:
    """Execute and prove one planar Boolean without mutating Session state."""

    selection = resolve_planar_boolean_faces(
        object_geometry,
        target_face_id,
        tool_geometry,
        tool_face_ids,
    )
    context = PlanarBooleanContext(
        next_planar_boolean_feature_id(object_geometry),
        selection.target_face_id,
        selection.tool_face_ids,
    )
    draft = BooleanGeometry(
        f"{getattr(object_geometry, 'name', 'Planar')}-Boolean",
        operation,
        object_geometry,
        tool_geometry,
        planar_context=context,
    )
    object_compiled = compile_recipe(cad, object_geometry)
    tool_compiled = compile_recipe(cad, tool_geometry)
    target_surfaces = tuple(object_compiled.logical_entities[selection.target_face_id])
    if len(target_surfaces) != 1:
        raise PlanarBooleanLineageResolutionError(
            "planar-boolean.target.face-not-unique: target must resolve "
            "to one OCC surface"
        )
    tool_surfaces = tuple(
        tool_compiled.logical_entities[face_id][0]
        for face_id in selection.tool_face_ids
    )
    target_evidence = capture_planar_operand_evidence(
        cad,
        object_compiled,
        (selection.target_face_id,),
    )
    tool_evidence = capture_planar_operand_evidence(
        cad,
        tool_compiled,
        selection.tool_face_ids,
    )
    unaffected = _planar_unaffected_logical_entities(
        object_compiled,
        selection,
    )
    boolean_operation = cad.fuse if operation == "fuse" else cad.cut
    result = boolean_operation(target_surfaces, tool_surfaces)
    validate_planar_boolean_input_map(
        result,
        tool_count=len(tool_surfaces),
        operation=operation,
    )
    proof = resolve_planar_boolean_lineage(
        cad,
        target_evidence,
        tool_evidence,
        result,
        context,
        operation=operation,
        unaffected_logical_entities=unaffected,
    )
    proven_context = replace(
        context,
        result_entities=proof.result_entities,
        topology_mappings=proof.topology_mappings,
    )
    geometry = replace(draft, planar_context=proven_context)
    preview = _tessellate_planar_entities(
        cad,
        proof.logical_entities,
        target_face_id=selection.target_face_id,
    )
    return StrictPlanarBooleanResult(geometry, proof, preview)


def prepare_strict_planar_recipe_preview(
    cad: Any,
    recipe: Any,
) -> StrictPlanarBooleanPreview:
    """Replay one persisted planar proof and detach true OCC tessellation."""

    context = _persisted_planar_context(recipe)
    if context is None:
        raise TypeError("recipe must contain a strict planar BooleanGeometry")
    if not context.proven:
        raise PlanarBooleanLineageResolutionError(
            "planar-boolean.lineage.unproven: persisted preview requires proof"
        )
    compiled = compile_recipe(cad, recipe)
    return _tessellate_planar_entities(
        cad,
        compiled.logical_entities,
        target_face_id=context.target_face_id,
    )


def next_planar_boolean_feature_id(recipe: object | None) -> str:
    """Allocate the next PB identity from committed feature history."""

    used: set[int] = set()

    def visit(item: object | None) -> None:
        if isinstance(item, BooleanGeometry):
            if item.planar_context is not None:
                used.add(int(item.planar_context.feature_id[2:]))
            visit(item.object_geometry)
            visit(item.tool_geometry)
        elif isinstance(item, (MovedGeometry, RotatedGeometry, ExtrudedGeometry)):
            visit(item.base)

    visit(recipe)
    number = 1
    while number in used:
        number += 1
    return f"PB{number}"


def _persisted_planar_context(recipe: Any) -> PlanarBooleanContext | None:
    if isinstance(recipe, BooleanGeometry):
        if recipe.planar_context is not None:
            return recipe.planar_context
        return _persisted_planar_context(
            recipe.object_geometry
        ) or _persisted_planar_context(recipe.tool_geometry)
    if isinstance(recipe, (MovedGeometry, RotatedGeometry, ExtrudedGeometry)):
        return _persisted_planar_context(recipe.base)
    return None


def _tessellate_planar_entities(
    cad: Any,
    logical_entities: Any,
    *,
    target_face_id: str,
) -> StrictPlanarBooleanPreview:
    by_entity: dict[Any, str] = {}
    for logical_id, entities in logical_entities.items():
        if logical_id.startswith("body:"):
            continue
        for entity in entities:
            previous = by_entity.setdefault(entity, logical_id)
            if previous != logical_id:
                raise PlanarBooleanLineageResolutionError(
                    "planar-boolean.preview.entity-ambiguous: one OCC entity "
                    "has multiple logical identities"
                )
    faces = tuple(entity for entity in by_entity if entity.dimension == 2)
    edges = tuple(entity for entity in by_entity if entity.dimension == 1)
    points = tuple(entity for entity in by_entity if entity.dimension == 0)
    tessellation = cad.tessellate_surfaces(faces, edges, points)
    return StrictPlanarBooleanPreview(
        target_face_id,
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


__all__ = [
    "StrictPlanarBooleanPreview",
    "StrictPlanarBooleanResult",
    "next_planar_boolean_feature_id",
    "prepare_planar_boolean",
    "prepare_strict_planar_recipe_preview",
]
