"""Headless feature-history projection derived solely from geometry recipes."""

from __future__ import annotations

from typing import Any

from fem.geometry.recipes import (
    BooleanGeometry,
    BoxGeometry,
    CylinderGeometry,
    DiskGeometry,
    ExtrudedGeometry,
    MovedGeometry,
    MultiBodyGeometry,
    NATIVE_GEOMETRY_TYPES,
    NativeGeometry,
    PlateWithHoleGeometry,
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
        if isinstance(item, WireGeometry):
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
                f"草图  点={len(recipe.points)}，曲线={len(recipe.curves)}，"
                f"Profile={material_count}，孔={hole_count}",
            )
        material_count = sum(
            contour.operation == "material" for contour in recipe.contours
        )
        cut_count = len(recipe.contours) - material_count
        return (
            f"草图  轮廓={len(recipe.contours)}，材料={material_count}，"
            f"切除={cut_count}",
        )
    if isinstance(recipe, WireGeometry):
        return (f"线框  节点={len(recipe.points)}，杆件={len(recipe.members)}",)
    if isinstance(recipe, MovedGeometry):
        return derive_geometry_feature_rows(recipe.base) + (
            f"移动  X={recipe.dx:g}，Y={recipe.dy:g}，Z={recipe.dz:g}",
        )
    if isinstance(recipe, RotatedGeometry):
        return derive_geometry_feature_rows(recipe.base) + (
            f"旋转  {recipe.axis.upper()} 轴，{recipe.angle_degrees:g}°",
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
            else f"，Profiles={profile_count}"
        )
        return derive_geometry_feature_rows(recipe.base) + (
            f"拉伸  高度={recipe.height:g}{profile_summary}",
        )
    if isinstance(recipe, RevolvedGeometry):
        profile_count = len(recipe.source_face_ids)
        profile_summary = (
            ""
            if profile_count <= 1
            else f"，Profiles={profile_count}"
        )
        return derive_geometry_feature_rows(recipe.base) + (
            f"扫掠  {recipe.axis.upper()} 轴，"
            f"{recipe.angle_degrees:g}°{profile_summary}",
        )
    if isinstance(recipe, BooleanGeometry):
        names = {"fuse": "合并", "cut": "切除", "fragment": "分割"}
        if recipe.planar_context is not None:
            return derive_geometry_feature_rows(recipe.object_geometry) + (
                f"二维{names[recipe.operation]}  "
                f"目标={recipe.planar_context.target_face_id}，"
                f"工具 Profiles={len(recipe.planar_context.tool_face_ids)}",
            )
        return derive_geometry_feature_rows(recipe.object_geometry) + (
            f"{names[recipe.operation]}  工具体={recipe.tool_geometry.name}",
        )
    if isinstance(recipe, RectangleGeometry):
        description = f"矩形  {recipe.width:g} × {recipe.height:g}"
    elif isinstance(recipe, DiskGeometry):
        description = f"圆盘  半径={recipe.radius:g}"
    elif isinstance(recipe, PlateWithHoleGeometry):
        description = (
            f"带孔板  {recipe.width:g} × {recipe.height:g}，"
            f"孔半径={recipe.hole_radius:g}"
        )
    elif isinstance(recipe, BoxGeometry):
        description = f"长方体  {recipe.width:g} × {recipe.depth:g} × {recipe.height:g}"
    elif isinstance(recipe, CylinderGeometry):
        description = f"圆柱  半径={recipe.radius:g}，高度={recipe.height:g}"
    else:  # pragma: no cover - _require_native_recipe owns supported types
        raise TypeError(f"unsupported native geometry recipe: {type(recipe).__name__}")
    return (f"基础体  {description}",)


def _require_native_recipe(recipe: Any) -> None:
    if not isinstance(recipe, NATIVE_GEOMETRY_TYPES):
        raise TypeError(f"unsupported native geometry recipe: {type(recipe).__name__}")


__all__ = ["derive_feature_history", "derive_geometry_feature_rows"]
