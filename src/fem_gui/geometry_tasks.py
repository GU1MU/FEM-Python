"""Detached exact geometry preparation for interactive document edits."""

from __future__ import annotations

from dataclasses import dataclass

from fem import geometry as geometry_runtime
from fem.application import NativePart, prepare_face_sketch_boolean, prepare_planar_boolean
from fem.application.face_sketch_boolean import FaceSketchBooleanResult
from fem.application.planar_boolean import StrictPlanarBooleanResult
from fem.application.recipe_compiler import compile_recipe
from fem.application.session import authenticate_native_part_geometry
from fem.geometry import (
    ExtrudedGeometry,
    FaceSketchBooleanGeometry,
    ResolvedFaceWorkplane,
    RevolvedGeometry,
    SketchReferencePoint,
    SketchGeometry,
    provide_face_reference_points,
    resolve_face_workplane,
)
from fem.geometry.gmsh_coordinator import PROCESS_GMSH_COORDINATOR

from .geometry_preview import (
    GeometryPreview,
    build_geometry_preview,
    namespace_part_geometry_preview,
)
from .workers import TaskContext


@dataclass(frozen=True, slots=True)
class PreparedFaceSketchWorkplane:
    workplane: ResolvedFaceWorkplane
    reference_points: tuple[SketchReferencePoint, ...]


def prepare_face_boolean_preview(
    recipe: FaceSketchBooleanGeometry, generation: int, context: TaskContext,
) -> FaceSketchBooleanResult:
    context.report("Performing exact extrude boolean and validating topology...")
    with PROCESS_GMSH_COORDINATOR.acquire(
        "Exact Extrude Boolean Preview", cancelled=lambda: context.is_cancelled,
    ):
        context.checkpoint()
        with geometry_runtime.model(f"{recipe.name}-{generation}", dimension=3) as cad:
            result = prepare_face_sketch_boolean(cad, recipe)
        context.checkpoint()
    return result


def prepare_planar_boolean_preview(
    source: object, operation: str, target_face_id: str,
    tool: SketchGeometry, tool_face_ids: tuple[str, ...], context: TaskContext,
) -> StrictPlanarBooleanResult:
    context.report("Performing planar boolean and validating exact topology...")
    with PROCESS_GMSH_COORDINATOR.acquire(
        "2D Boolean Preview", cancelled=lambda: context.is_cancelled,
    ):
        context.checkpoint()
        with geometry_runtime.model(f"{getattr(source, 'name', 'planar')}-{operation}-preview", dimension=2) as cad:
            result = prepare_planar_boolean(cad, source, target_face_id, tool, tool_face_ids, operation)
        context.checkpoint()
    return result


def prepare_face_sketch_workplane(
    recipe: object, face_id: str, context: TaskContext,
) -> PreparedFaceSketchWorkplane:
    """Resolve a Face and its references while owning the CAD runtime."""

    context.report("Resolving sketch workplane...")
    with PROCESS_GMSH_COORDINATOR.acquire(
        "Resolve Face Sketch Workplane", cancelled=lambda: context.is_cancelled,
    ):
        context.checkpoint()
        with geometry_runtime.model(
            f"{getattr(recipe, 'name', 'solid')}-face-sketch", dimension=3,
        ) as cad:
            compiled = compile_recipe(cad, recipe)
            context.checkpoint()
            workplane = resolve_face_workplane(cad, compiled.logical_entities, face_id)
            reference_points = provide_face_reference_points(
                cad, compiled.logical_entities, workplane,
            )
        context.checkpoint()
    return PreparedFaceSketchWorkplane(workplane, tuple(reference_points))


def prepare_profile_solid_geometry(
    part_id: str,
    recipes: tuple[ExtrudedGeometry | RevolvedGeometry, ...],
    context: TaskContext,
) -> GeometryPreview:
    """Certify every candidate solid and prepare the visible primary Part."""

    context.report("Checking solid geometry...")
    with PROCESS_GMSH_COORDINATOR.acquire(
        "Check Profile Solid Geometry", cancelled=lambda: context.is_cancelled,
    ):
        context.checkpoint()
        for recipe in recipes:
            authenticate_native_part_geometry(
                NativePart(id=part_id, name=recipe.name, geometry_recipe=recipe),
            )
            context.checkpoint()
        primary_recipe = min(recipes, key=lambda recipe: tuple(recipe.source_face_ids))
        preview = namespace_part_geometry_preview(
            part_id, build_geometry_preview(primary_recipe),
        )
        context.checkpoint()
    return preview
