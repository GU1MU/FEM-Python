"""Detached controller for strict planar Boolean authoring."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from fem.geometry import (
    LogicalEntityRef,
    NATIVE_GEOMETRY_TYPES,
    SketchGeometry,
    geometry_dimension,
    resolve_extrusion_source_faces,
    resolve_planar_boolean_faces,
)


PlanarBooleanOperation = Literal["fuse", "cut"]


@dataclass(slots=True)
class PlanarBooleanController:
    """Own target/tool draft state without changing the committed Session."""

    geometry: object
    base_session_revision: int
    operation: PlanarBooleanOperation
    target_face_id: str | None = None
    tool_geometry: SketchGeometry | None = None
    tool_face_ids: tuple[str, ...] = ()
    selecting_target: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.geometry, NATIVE_GEOMETRY_TYPES):
            raise TypeError("geometry must be a native geometry recipe")
        if geometry_dimension(self.geometry) != 2:
            raise ValueError("strict planar Boolean requires 2D geometry")
        if type(self.base_session_revision) is not int:
            raise TypeError("base_session_revision must be an integer")
        self.set_operation(self.operation)
        if self.target_face_id is not None:
            self.set_target(self.target_face_id)

    @property
    def ready(self) -> bool:
        return (
            self.target_face_id is not None
            and self.tool_geometry is not None
            and bool(self.tool_face_ids)
        )

    def set_operation(self, operation: str) -> None:
        if operation not in {"fuse", "cut"}:
            raise ValueError("planar Boolean operation must be fuse or cut")
        self.operation = operation

    def request_target_selection(self) -> None:
        self.selecting_target = True

    def assign_reference(self, reference: LogicalEntityRef) -> None:
        if not self.selecting_target:
            raise ValueError("no planar Boolean target selection is pending")
        if type(reference) is not LogicalEntityRef or reference.kind != "face":
            raise ValueError("planar Boolean target selection requires a Face")
        self.set_target(reference.logical_id)
        self.selecting_target = False

    def set_target(self, logical_id: str) -> None:
        target = resolve_extrusion_source_faces(
            self.geometry,
            (logical_id,),
        )
        if len(target.face_ids) != 1:
            raise ValueError("planar Boolean requires exactly one target Face")
        self.target_face_id = target.face_ids[0]
        if self.tool_geometry is not None:
            self._validate_complete_selection()

    def set_tool_recipe(self, recipe: SketchGeometry) -> None:
        if type(recipe) is not SketchGeometry or not recipe.is_strict:
            raise TypeError("planar Boolean tool must be a strict sketch")
        selection = resolve_extrusion_source_faces(recipe)
        self.tool_geometry = recipe
        self.tool_face_ids = selection.face_ids
        if self.target_face_id is not None:
            self._validate_complete_selection()

    def _validate_complete_selection(self) -> None:
        if self.target_face_id is None or self.tool_geometry is None:
            return
        selection = resolve_planar_boolean_faces(
            self.geometry,
            self.target_face_id,
            self.tool_geometry,
            self.tool_face_ids,
        )
        self.target_face_id = selection.target_face_id
        self.tool_face_ids = selection.tool_face_ids

    def target_label(self) -> str:
        return self.target_face_id or "未选择"

    def tool_label(self) -> str:
        if self.tool_geometry is None:
            return "未绘制"
        return f"{len(self.tool_face_ids)} 个 Profiles"


__all__ = ["PlanarBooleanController", "PlanarBooleanOperation"]
