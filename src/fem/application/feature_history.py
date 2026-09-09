"""Headless feature-history projection derived solely from geometry recipes."""

from __future__ import annotations

from typing import Any

from fem.geometry.recipes import (
    BooleanGeometry,
    BoxGeometry,
    CylinderGeometry,
    DiskGeometry,
    ExtrudedGeometry,
    FaceSketchBooleanDirection,
    FaceSketchBooleanGeometry,
    FaceSketchBooleanOperation,
    MovedGeometry,
    MultiBodyGeometry,
    NATIVE_GEOMETRY_TYPES,
    NativeGeometry,
    PlateWithHoleGeometry,
    PathSweptGeometry,
    RectangleGeometry,
    RevolvedGeometry,
    RotatedGeometry,
    SketchGeometry,
    WireGeometry,
)
from fem.geometry.recipe_analysis import analyze_sketch_profiles
from fem.geometry.extrusion_selection import (
    ExtrusionSourceResolutionError,
    resolve_extrusion_source_faces,
)

from .definitions import FeatureRecord


def derive_feature_history(recipe: NativeGeometry) -> tuple[FeatureRecord, ...]:
    """Return the canonical shallow feature projection for one recipe chain."""

    _require_native_recipe(recipe)
    if isinstance(recipe, MultiBodyGeometry):
        return tuple(
            FeatureRecord(
                f"{body.id}/{record.name}",
                record.kind,
                {
                    **record.payload,
                    "body_id": body.id,
                    "body_name": body.name,
                },
            )
            for body in recipe.bodies
            for record in derive_feature_history(body.recipe)
        )
    records: list[FeatureRecord] = []
    counters: dict[str, int] = {}

    def add(kind: str, summary: str) -> None:
        counters[kind] = counters.get(kind, 0) + 1
        records.append(
            FeatureRecord(
                f"{kind}-{counters[kind]}",
                kind.casefold(),
                {"summary": summary},
            )
        )

    def visit(item: NativeGeometry) -> None:
        if isinstance(item, MultiBodyGeometry):
            records.extend(derive_feature_history(item))
        elif isinstance(item, WireGeometry):
            add("Wire", derive_geometry_feature_rows(item)[0])
        elif isinstance(item, SketchGeometry):
            add("Sketch", derive_geometry_feature_rows(item)[0])
        elif isinstance(item, MovedGeometry):
            visit(item.base)
            add("Move", derive_geometry_feature_rows(item)[-1])
        elif isinstance(item, RotatedGeometry):
            visit(item.base)
            add("Rotate", derive_geometry_feature_rows(item)[-1])
        elif isinstance(item, ExtrudedGeometry):
            visit(item.base)
            add("Extrude", derive_geometry_feature_rows(item)[-1])
        elif isinstance(item, RevolvedGeometry):
            visit(item.base)
            add("Sweep", derive_geometry_feature_rows(item)[-1])
        elif isinstance(item, PathSweptGeometry):
            visit(item.base)
            add("PathSweep", derive_geometry_feature_rows(item)[-1])
        elif isinstance(item, FaceSketchBooleanGeometry):
            visit(item.base)
            records.append(
                FeatureRecord(
                    item.name,
                    (
                        "face_sketch_boolean_fuse"
                        if item.operation is FaceSketchBooleanOperation.FUSE
                        else "face_sketch_boolean_cut"
                    ),
                    {
                        "summary": derive_geometry_feature_rows(item)[-1],
                        "feature_id": item.feature_id,
                        "support_face_id": item.support_face_id,
                        "direction": item.direction.display_name,
                        "distance": item.distance,
                        "profile_count": len(item.participating_profile_ids),
                        "association_count": len(item.external_references),
                    },
                )
            )
        elif isinstance(item, BooleanGeometry):
            visit(item.object_geometry)
            kind = {
                "fuse": "Fuse",
                "cut": "Cut",
                "fragment": "Partition",
            }[item.operation]
            add(kind, derive_geometry_feature_rows(item)[-1])
        else:
            add("Base", derive_geometry_feature_rows(item)[0])

    visit(recipe)
    return tuple(records)


def remove_terminal_feature(recipe: NativeGeometry) -> NativeGeometry:
    """Return the recipe that remains after removing one outer feature.

    This mirrors the native GUI feature manager: only the terminal feature in
    a single-body recipe chain is removable, so no descendant dependency can
    be orphaned.
    """

    _require_native_recipe(recipe)
    if isinstance(
        recipe,
        (
            MovedGeometry,
            RotatedGeometry,
            ExtrudedGeometry,
            RevolvedGeometry,
            PathSweptGeometry,
            FaceSketchBooleanGeometry,
        ),
    ):
        return recipe.base
    if isinstance(recipe, BooleanGeometry):
        return recipe.object_geometry
    raise ValueError("geometry recipe has no removable terminal feature")


def derive_geometry_feature_rows(
    recipe: NativeGeometry,
) -> tuple[str, ...]:
    """Return the pure user-facing summaries used by feature history."""

    _require_native_recipe(recipe)
    if isinstance(recipe, MultiBodyGeometry):
        return tuple(
            f"{body.name} [{body.id}]  {row}"
            for body in recipe.bodies
            for row in derive_geometry_feature_rows(body.recipe)
        )
    if isinstance(recipe, SketchGeometry):
        if recipe.is_strict:
            analysis = analyze_sketch_profiles(recipe)
            material_count = sum(
                profile.is_material for profile in analysis.profiles
            )
            hole_count = sum(profile.is_hole for profile in analysis.profiles)
            return (
                f"Sketch  Points={len(recipe.points)}, Curves={len(recipe.curves)}, "
                f"Profiles={material_count}, Holes={hole_count}",
            )
        material_count = sum(
            contour.operation == "material" for contour in recipe.contours
        )
        cut_count = len(recipe.contours) - material_count
        return (
            f"Sketch  Contours={len(recipe.contours)}, Material={material_count}, "
            f"Cuts={cut_count}",
        )
    if isinstance(recipe, WireGeometry):
        return (f"Wire  Nodes={len(recipe.points)}, Members={len(recipe.members)}",)
    if isinstance(recipe, MovedGeometry):
        return derive_geometry_feature_rows(recipe.base) + (
            f"Move  X={recipe.dx:g}, Y={recipe.dy:g}, Z={recipe.dz:g}",
        )
    if isinstance(recipe, RotatedGeometry):
        return derive_geometry_feature_rows(recipe.base) + (
            f"Rotate  {recipe.axis.upper()} axis, {recipe.angle_degrees:g}°",
        )
    if isinstance(recipe, ExtrudedGeometry):
        try:
            profile_count = len(
                resolve_extrusion_source_faces(
                    recipe.base,
                    recipe.source_face_ids,
                ).face_ids
            )
        except ExtrusionSourceResolutionError:
            profile_count = len(recipe.source_face_ids)
        profile_summary = (
            ""
            if profile_count <= 1
            else f", Profiles={profile_count}"
        )
        return derive_geometry_feature_rows(recipe.base) + (
            f"Extrude  Height={recipe.height:g}{profile_summary}",
        )
    if isinstance(recipe, RevolvedGeometry):
        profile_count = len(recipe.source_face_ids)
        profile_summary = (
            ""
            if profile_count <= 1
            else f", Profiles={profile_count}"
        )
        return derive_geometry_feature_rows(recipe.base) + (
            f"Sweep  {recipe.axis.upper()} axis, "
            f"{recipe.angle_degrees:g}°{profile_summary}",
        )
    if isinstance(recipe, PathSweptGeometry):
        return derive_geometry_feature_rows(recipe.base) + (
            f"Path sweep  Path segments={len(recipe.path.members)}, "
            f"frame={recipe.frame_strategy}",
        )
    if isinstance(recipe, FaceSketchBooleanGeometry):
        operation = (
            "Extrude Fuse"
            if recipe.operation is FaceSketchBooleanOperation.FUSE
            else "Extrude Cut"
        )
        direction = (
            "Outward"
            if recipe.direction is FaceSketchBooleanDirection.OUTWARD
            else "Inward"
        )
        return derive_geometry_feature_rows(recipe.base) + (
            f"{operation}  Workplane={recipe.support_face_id}, "
            f"Direction={direction}, Distance={recipe.distance:g}, "
            f"Profiles={len(recipe.participating_profile_ids)}, "
            f"External references={len(recipe.external_references)}",
        )
    if isinstance(recipe, BooleanGeometry):
        names = {"fuse": "Fuse", "cut": "Cut", "fragment": "Fragment"}
        if recipe.planar_context is not None:
            return derive_geometry_feature_rows(recipe.object_geometry) + (
                f"2D {names[recipe.operation]}  "
                f"Target={recipe.planar_context.target_face_id}, "
                f"Tool Profiles={len(recipe.planar_context.tool_face_ids)}",
            )
        return derive_geometry_feature_rows(recipe.object_geometry) + (
            f"{names[recipe.operation]}  Tool body={recipe.tool_geometry.name}",
        )
    if isinstance(recipe, RectangleGeometry):
        description = f"Rectangle  {recipe.width:g} × {recipe.height:g}"
    elif isinstance(recipe, DiskGeometry):
        description = f"Disk  Radius={recipe.radius:g}"
    elif isinstance(recipe, PlateWithHoleGeometry):
        description = (
            f"Plate with hole  {recipe.width:g} × {recipe.height:g}, "
            f"Hole radius={recipe.hole_radius:g}"
        )
    elif isinstance(recipe, BoxGeometry):
        description = f"Box  {recipe.width:g} × {recipe.depth:g} × {recipe.height:g}"
    elif isinstance(recipe, CylinderGeometry):
        description = f"Cylinder  Radius={recipe.radius:g}, Height={recipe.height:g}"
    else:  # pragma: no cover - _require_native_recipe owns supported types
        raise TypeError(f"unsupported native geometry recipe: {type(recipe).__name__}")
    return (f"Base geometry  {description}",)


def _require_native_recipe(recipe: Any) -> None:
    if not isinstance(recipe, NATIVE_GEOMETRY_TYPES):
        raise TypeError(f"unsupported native geometry recipe: {type(recipe).__name__}")


__all__ = [
    "derive_feature_history",
    "derive_geometry_feature_rows",
    "remove_terminal_feature",
]
