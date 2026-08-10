"""Detached controller for strict planar Boolean authoring."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

from fem.geometry import (
    LogicalEntityRef,
    NATIVE_GEOMETRY_TYPES,
    SketchGeometry,
    SketchExternalCoincidence,
    SketchExternalReference,
    SketchExternalReferenceType,
    SketchPlane,
    SketchReferencePoint,
    geometry_dimension,
    resolve_extrusion_source_faces,
    resolve_planar_boolean_faces,
)

from .geometry_preview import GeometryPreview


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
    external_references: tuple[SketchExternalReference, ...] = ()
    external_coincidences: tuple[SketchExternalCoincidence, ...] = ()
    unresolved_reference_ids: tuple[str, ...] = ()

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

    def confirm_target_selection(self) -> None:
        if self.target_face_id is None:
            raise ValueError("请先选择目标面")
        self.selecting_target = False

    def clear_target(self) -> None:
        """Clear the detached target selection without changing the Session."""

        self.target_face_id = None
        self.selecting_target = False

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

    def set_tool_recipe(
        self,
        recipe: SketchGeometry,
        *,
        external_references: tuple[SketchExternalReference, ...] = (),
        external_coincidences: tuple[SketchExternalCoincidence, ...] = (),
        unresolved_reference_ids: tuple[str, ...] = (),
    ) -> None:
        if type(recipe) is not SketchGeometry or not recipe.is_strict:
            raise TypeError("planar Boolean tool must be a strict sketch")
        selection = resolve_extrusion_source_faces(recipe)
        self.tool_geometry = recipe
        self.tool_face_ids = selection.face_ids
        self.set_tool_associations(
            external_references,
            external_coincidences,
            unresolved_reference_ids,
        )
        if self.target_face_id is not None:
            self._validate_complete_selection()

    def clear_tool(self) -> None:
        """Delete the detached tool sketch without changing the Session."""

        self.tool_geometry = None
        self.tool_face_ids = ()
        self.external_references = ()
        self.external_coincidences = ()
        self.unresolved_reference_ids = ()

    def set_tool_associations(
        self,
        references: tuple[SketchExternalReference, ...],
        coincidences: tuple[SketchExternalCoincidence, ...],
        unresolved_reference_ids: tuple[str, ...] = (),
    ) -> None:
        values = tuple(references)
        relations = tuple(coincidences)
        unresolved = tuple(sorted(set(unresolved_reference_ids)))
        if any(type(item) is not SketchExternalReference for item in values):
            raise TypeError("二维布尔外部参考类型无效")
        if any(type(item) is not SketchExternalCoincidence for item in relations):
            raise TypeError("二维布尔外部重合关系类型无效")
        if len({item.id for item in values}) != len(values):
            raise ValueError("二维布尔外部参考 ID 不能重复")
        if len({item.point_id for item in relations}) != len(relations):
            raise ValueError("每个二维布尔草图点最多只能绑定一个外部参考")
        reference_ids = {item.id for item in values}
        point_ids = (
            set()
            if self.tool_geometry is None
            else {point.id for point in self.tool_geometry.points}
        )
        if any(
            item.reference_id not in reference_ids or item.point_id not in point_ids
            for item in relations
        ):
            raise ValueError("二维布尔外部关联引用了不存在的点或参考")
        if not set(unresolved).issubset(reference_ids):
            raise ValueError("二维布尔未解析状态引用了不存在的外部参考")
        self.external_references = values
        self.external_coincidences = relations
        self.unresolved_reference_ids = unresolved

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
        return "已选择" if self.target_face_id is not None else "未选择"

    def tool_label(self) -> str:
        if self.tool_geometry is None:
            return "未绘制"
        return f"{len(self.tool_face_ids)} 个闭合轮廓"


def planar_reference_points(
    preview: GeometryPreview,
    face_id: str,
    *,
    plane: SketchPlane | None = None,
) -> tuple[SketchReferencePoint, ...]:
    """Build the shared discrete reference set for a planar Boolean Face."""

    if type(preview) is not GeometryPreview:
        raise TypeError("preview must be a GeometryPreview")
    support = LogicalEntityRef(face_id)
    if support.kind != "face":
        raise ValueError("planar reference support must be a Face")
    frame = SketchPlane.xy() if plane is None else plane
    selected_faces = tuple(
        face
        for face, logical_id in zip(
            preview.faces,
            preview.face_logical_ids,
            strict=True,
        )
        if logical_id == support.logical_id
    )
    if not selected_faces:
        return ()
    selected_indices = {index for face in selected_faces for index in face}
    result: list[SketchReferencePoint] = []
    seen_points: set[str] = set()
    for index in sorted(selected_indices):
        logical_id = preview.point_logical_ids[index]
        if logical_id is None or logical_id in seen_points:
            continue
        reference = LogicalEntityRef(logical_id)
        if reference.kind != "point":
            continue
        seen_points.add(reference.logical_id)
        result.append(
            _planar_reference_point(
                frame,
                reference,
                SketchExternalReferenceType.TOPOLOGY_VERTEX,
                preview.points[index],
            )
        )
    edge_cells: dict[str, list[tuple[int, ...]]] = {}
    for edge, logical_id in zip(
        preview.edges,
        preview.edge_logical_ids,
        strict=True,
    ):
        if (
            logical_id is not None
            and set(edge).issubset(selected_indices)
            and LogicalEntityRef(logical_id).kind == "edge"
        ):
            edge_cells.setdefault(logical_id, []).append(edge)
    for logical_id, cells in sorted(edge_cells.items()):
        indices = tuple(dict.fromkeys(index for cell in cells for index in cell))
        local = tuple(frame.to_local(preview.points[index]) for index in indices)
        center = _fitted_circle_center(local)
        if center is None:
            endpoints = max(
                (
                    (left, right)
                    for left in local
                    for right in local
                ),
                key=lambda pair: math.dist(*pair),
            )
            u = 0.5 * (endpoints[0][0] + endpoints[1][0])
            v = 0.5 * (endpoints[0][1] + endpoints[1][1])
            derived = SketchExternalReferenceType.LINE_MIDPOINT
        else:
            u, v = center
            degrees: dict[int, int] = {}
            for cell in cells:
                for start, end in zip(cell, cell[1:]):
                    degrees[start] = degrees.get(start, 0) + 1
                    degrees[end] = degrees.get(end, 0) + 1
            derived = (
                SketchExternalReferenceType.CIRCLE_CENTER
                if degrees and all(value % 2 == 0 for value in degrees.values())
                else SketchExternalReferenceType.ARC_CENTER
            )
        result.append(
            _planar_reference_point(
                frame,
                LogicalEntityRef(logical_id),
                derived,
                frame.to_global(u, v),
            )
        )
    center = _face_area_center(preview, selected_faces, frame)
    result.append(
        _planar_reference_point(
            frame,
            support,
            SketchExternalReferenceType.FACE_CENTER,
            frame.to_global(*center),
        )
    )
    return tuple(result)


def _planar_reference_point(
    plane: SketchPlane,
    source: LogicalEntityRef,
    derived_type: SketchExternalReferenceType,
    position: tuple[float, float, float],
) -> SketchReferencePoint:
    reference = SketchExternalReference(
        f"reference:{derived_type.value}:{source.logical_id}",
        source,
        derived_type,
    )
    u, v = plane.to_local(position)
    return SketchReferencePoint(reference, position, u, v)


def _fitted_circle_center(
    points: tuple[tuple[float, float], ...],
) -> tuple[float, float] | None:
    if len(points) < 3:
        return None
    first = points[0]
    for middle_index in range(1, len(points) - 1):
        middle = points[middle_index]
        for last in points[middle_index + 1 :]:
            denominator = 2.0 * (
                first[0] * (middle[1] - last[1])
                + middle[0] * (last[1] - first[1])
                + last[0] * (first[1] - middle[1])
            )
            if abs(denominator) <= 1.0e-12:
                continue
            squares = tuple(x * x + y * y for x, y in (first, middle, last))
            center = (
                (
                    squares[0] * (middle[1] - last[1])
                    + squares[1] * (last[1] - first[1])
                    + squares[2] * (first[1] - middle[1])
                )
                / denominator,
                (
                    squares[0] * (last[0] - middle[0])
                    + squares[1] * (first[0] - last[0])
                    + squares[2] * (middle[0] - first[0])
                )
                / denominator,
            )
            radii = tuple(math.dist(center, point) for point in points)
            radius = sum(radii) / len(radii)
            tolerance = max(1.0e-8, radius * 1.0e-5)
            if radius > 0.0 and max(abs(value - radius) for value in radii) <= tolerance:
                return center
    return None


def _face_area_center(
    preview: GeometryPreview,
    faces: tuple[tuple[int, ...], ...],
    plane: SketchPlane,
) -> tuple[float, float]:
    weighted_u = 0.0
    weighted_v = 0.0
    total_area = 0.0
    for face in faces:
        vertices = tuple(plane.to_local(preview.points[index]) for index in face)
        for index in range(1, len(vertices) - 1):
            first, second, third = vertices[0], vertices[index], vertices[index + 1]
            area = abs(
                (second[0] - first[0]) * (third[1] - first[1])
                - (second[1] - first[1]) * (third[0] - first[0])
            ) * 0.5
            weighted_u += area * (first[0] + second[0] + third[0]) / 3.0
            weighted_v += area * (first[1] + second[1] + third[1]) / 3.0
            total_area += area
    if total_area <= 0.0:
        raise ValueError("目标面无法计算几何中心")
    return weighted_u / total_area, weighted_v / total_area


__all__ = [
    "PlanarBooleanController",
    "PlanarBooleanOperation",
    "planar_reference_points",
]
