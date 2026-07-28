"""Canonical target/tool Profile selection for strict planar Booleans."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from .extrusion_selection import (
    ExtrusionSourceResolutionError,
    ExtrusionSourceSelection,
    resolve_extrusion_source_faces,
)
from .references import LogicalEntityRef


class PlanarBooleanSelectionError(ValueError):
    """A planar Boolean target or tool selection cannot be proven."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        logical_id: str | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.logical_id = logical_id


@dataclass(frozen=True, slots=True)
class PlanarBooleanSelection:
    """One canonical target face and one or more canonical tool Profiles."""

    target: ExtrusionSourceSelection
    tool: ExtrusionSourceSelection

    def __post_init__(self) -> None:
        if type(self.target) is not ExtrusionSourceSelection:
            raise TypeError("target must be ExtrusionSourceSelection")
        if type(self.tool) is not ExtrusionSourceSelection:
            raise TypeError("tool must be ExtrusionSourceSelection")
        if len(self.target.face_ids) != 1:
            raise ValueError("planar Boolean requires exactly one target face")
        if not self.tool.face_ids:
            raise ValueError("planar Boolean requires at least one tool Profile")

    @property
    def target_face_id(self) -> str:
        return self.target.face_ids[0]

    @property
    def tool_face_ids(self) -> tuple[str, ...]:
        return self.tool.face_ids


def resolve_planar_boolean_faces(
    object_geometry: object,
    target_face_id: str | LogicalEntityRef,
    tool_geometry: object,
    tool_face_ids: Iterable[str | LogicalEntityRef],
) -> PlanarBooleanSelection:
    """Resolve exact material faces for one strict planar Boolean."""

    from .recipes import SketchGeometry

    if type(tool_geometry) is not SketchGeometry or not tool_geometry.is_strict:
        raise PlanarBooleanSelectionError(
            "planar-boolean.tool.strict-sketch-required",
            "二维布尔工具必须是独立的严格草图",
        )
    if not _uses_global_xy_plane(object_geometry):
        raise PlanarBooleanSelectionError(
            "planar-boolean.target.plane-unsupported",
            "二维布尔目标必须位于全局 XY 平面",
        )
    if not _uses_global_xy_plane(tool_geometry):
        raise PlanarBooleanSelectionError(
            "planar-boolean.tool.plane-unsupported",
            "二维布尔工具草图必须位于全局 XY 平面",
        )
    try:
        requested_tools = tuple(tool_face_ids)
    except TypeError as error:
        raise TypeError("tool_face_ids must be iterable") from error
    if not requested_tools:
        raise PlanarBooleanSelectionError(
            "planar-boolean.tool.required",
            "二维布尔至少需要一个闭合 material Profile 作为工具轮廓",
        )
    target = _resolve(
        object_geometry,
        (target_face_id,),
        role="target",
    )
    if len(target.face_ids) != 1:
        raise PlanarBooleanSelectionError(
            "planar-boolean.target.required",
            "二维布尔必须恰好选择一个 material Face",
        )
    tool = _resolve(tool_geometry, requested_tools, role="tool")
    return PlanarBooleanSelection(target, tool)


def _resolve(
    geometry: object,
    requested: tuple[str | LogicalEntityRef, ...],
    *,
    role: str,
) -> ExtrusionSourceSelection:
    try:
        return resolve_extrusion_source_faces(geometry, requested)
    except ExtrusionSourceResolutionError as error:
        code_suffix = error.code.removeprefix("extrude.source-face.")
        raise PlanarBooleanSelectionError(
            f"planar-boolean.{role}.{code_suffix}",
            str(error),
            logical_id=error.logical_id,
        ) from error


def _uses_global_xy_plane(recipe: object) -> bool:
    from .recipes import (
        BooleanGeometry,
        DiskGeometry,
        MovedGeometry,
        PlateWithHoleGeometry,
        RectangleGeometry,
        RotatedGeometry,
        SketchGeometry,
        SketchPlane,
    )

    if isinstance(
        recipe,
        (RectangleGeometry, DiskGeometry, PlateWithHoleGeometry),
    ):
        return True
    if isinstance(recipe, SketchGeometry):
        return recipe.is_strict and recipe.plane == SketchPlane.xy()
    if isinstance(recipe, MovedGeometry):
        return recipe.dz == 0.0 and _uses_global_xy_plane(recipe.base)
    if isinstance(recipe, RotatedGeometry):
        return recipe.axis == "z" and _uses_global_xy_plane(recipe.base)
    if isinstance(recipe, BooleanGeometry):
        return _uses_global_xy_plane(recipe.object_geometry) and _uses_global_xy_plane(
            recipe.tool_geometry
        )
    return False


__all__ = [
    "PlanarBooleanSelection",
    "PlanarBooleanSelectionError",
    "resolve_planar_boolean_faces",
]
