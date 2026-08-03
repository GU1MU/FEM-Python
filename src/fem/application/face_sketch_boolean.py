"""Detached exact preparation for planar-face sketch Boolean features."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from fem.geometry.recipes import (
    FaceSketchBooleanGeometry,
    FaceSketchBooleanStepProof,
)
from fem.geometry.solid_boolean_lineage import BooleanLineageResolutionError
from fem.geometry.types import StrictBodyBooleanPreview

from .recipe_compiler import _prepare_face_sketch_boolean_draft


@dataclass(frozen=True, slots=True)
class FaceSketchBooleanResult:
    """One proven recipe and detached exact preview; Session remains untouched."""

    geometry: FaceSketchBooleanGeometry
    step_proofs: tuple[FaceSketchBooleanStepProof, ...]
    preview: StrictBodyBooleanPreview


def prepare_face_sketch_boolean(
    cad: Any,
    recipe: FaceSketchBooleanGeometry,
) -> FaceSketchBooleanResult:
    """Execute the complete stable-profile chain without committing it."""

    if type(recipe) is not FaceSketchBooleanGeometry:
        raise TypeError("recipe must be a FaceSketchBooleanGeometry")
    prepared = _prepare_face_sketch_boolean_draft(cad, recipe)
    geometry = replace(recipe, step_proofs=prepared.step_proofs)
    preview = _tessellate_target(
        cad,
        prepared.target_logical_entities,
        target_body_id=prepared.target_body_id,
    )
    return FaceSketchBooleanResult(geometry, prepared.step_proofs, preview)


def prepare_face_sketch_boolean_preview(
    cad: Any,
    recipe: FaceSketchBooleanGeometry,
) -> StrictBodyBooleanPreview:
    """Strictly replay persisted step proofs and return the exact result mesh."""

    if type(recipe) is not FaceSketchBooleanGeometry:
        raise TypeError("recipe must be a FaceSketchBooleanGeometry")
    if len(recipe.step_proofs) != len(recipe.participating_profile_ids):
        raise BooleanLineageResolutionError(
            "face-sketch-boolean.lineage.unproven: persisted preview requires "
            "one proof for every participating profile"
        )
    prepared = _prepare_face_sketch_boolean_draft(cad, recipe)
    if prepared.step_proofs != recipe.step_proofs:
        raise BooleanLineageResolutionError(
            "face-sketch-boolean.lineage.catalog-mismatch: persisted proof "
            "does not match the current OCC result"
        )
    return _tessellate_target(
        cad,
        prepared.target_logical_entities,
        target_body_id=prepared.target_body_id,
    )


def _tessellate_target(
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
                    "face-sketch-boolean.preview.entity-ambiguous: one OCC "
                    "entity has multiple logical identities"
                )
    faces = tuple(entity for entity in by_entity if entity.dimension == 2)
    edges = tuple(entity for entity in by_entity if entity.dimension == 1)
    points = tuple(entity for entity in by_entity if entity.dimension == 0)
    tessellation = cad.tessellate_surfaces(faces, edges, points)
    return StrictBodyBooleanPreview(
        target_body_id,
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
    "FaceSketchBooleanResult",
    "prepare_face_sketch_boolean",
    "prepare_face_sketch_boolean_preview",
]
