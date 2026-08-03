"""Stable planar-face workplane preparation for associated sketches."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

from .recipes import (
    FaceSketchBooleanDirection,
    FaceSketchWorkplaneStrategy,
    SketchPlane,
)
from .references import LogicalEntityRef


_AXES: dict[str, tuple[float, float, float]] = {
    "x": (1.0, 0.0, 0.0),
    "y": (0.0, 1.0, 0.0),
    "z": (0.0, 0.0, 1.0),
}
_FRAME_TOLERANCE = 1.0e-9


class FaceWorkplaneResolutionError(ValueError):
    """A stable Chinese diagnostic for a workplane that cannot be resolved."""

    def __init__(self, code: str, message: str) -> None:
        if type(code) is not str or not code.strip():
            raise ValueError("工作面诊断代码不能为空")
        if type(message) is not str or not message.strip():
            raise ValueError("工作面诊断消息不能为空")
        self.code = code
        self.message = message
        super().__init__(f"{code}: {message}")


@dataclass(frozen=True, slots=True)
class ResolvedFaceWorkplane:
    """Current exact OCC entities and the deterministic right-handed frame."""

    support_face_id: str
    target_body_id: str
    surface: Any
    volume: Any
    plane: SketchPlane
    outward_normal: tuple[float, float, float]
    strategy: FaceSketchWorkplaneStrategy
    area: float

    def __post_init__(self) -> None:
        face = LogicalEntityRef(self.support_face_id)
        body = LogicalEntityRef(self.target_body_id)
        if face.kind != "face" or body.kind != "body":
            raise ValueError("工作面和目标 Body 逻辑 ID 类型无效")
        if getattr(self.surface, "dimension", None) != 2:
            raise ValueError("工作面 OCC 实体必须为二维")
        if getattr(self.volume, "dimension", None) != 3:
            raise ValueError("目标 Body OCC 实体必须为三维")
        if type(self.plane) is not SketchPlane:
            raise TypeError("工作面必须使用 SketchPlane")
        if type(self.strategy) is not FaceSketchWorkplaneStrategy:
            raise TypeError("工作面坐标策略无效")
        if not math.isfinite(self.area) or self.area <= 0.0:
            raise ValueError("工作面面积必须为有限正值")

    @property
    def origin(self) -> tuple[float, float, float]:
        return self.plane.origin

    @property
    def u_axis(self) -> tuple[float, float, float]:
        return self.plane.x_direction

    @property
    def v_axis(self) -> tuple[float, float, float]:
        return self.plane.y_direction

    def direction_vector(
        self,
        direction: FaceSketchBooleanDirection,
    ) -> tuple[float, float, float]:
        """Return the signed extrusion direction in global coordinates."""

        if type(direction) is not FaceSketchBooleanDirection:
            raise TypeError("拉伸方向无效")
        return direction.vector(self.outward_normal)


def resolve_face_workplane(
    cad: Any,
    logical_entities: Mapping[str, tuple[Any, ...]],
    support_face: str | LogicalEntityRef | None,
    strategy: FaceSketchWorkplaneStrategy | None = None,
) -> ResolvedFaceWorkplane:
    """Resolve one stable logical plane face to its unique owning solid frame."""

    if support_face is None or (
        type(support_face) is str and not support_face.strip()
    ):
        raise FaceWorkplaneResolutionError(
            "face-workplane.face-required",
            "未选择工作面",
        )
    reference = (
        support_face
        if type(support_face) is LogicalEntityRef
        else LogicalEntityRef(support_face)
    )
    if reference.kind != "face":
        raise FaceWorkplaneResolutionError(
            "face-workplane.face-required",
            "未选择工作面",
        )
    entity_mapping = getattr(logical_entities, "logical_entities", logical_entities)
    try:
        surfaces = tuple(entity_mapping.get(reference.logical_id, ()))
    except AttributeError as error:
        raise TypeError("logical_entities 必须是逻辑实体映射") from error
    if len(surfaces) != 1 or getattr(surfaces[0], "dimension", None) != 2:
        raise FaceWorkplaneResolutionError(
            "face-workplane.face-stale",
            "所选工作面已失效，无法由原逻辑面 ID 唯一恢复",
        )
    surface = surfaces[0]
    try:
        geometry_type = cad.geometry_type(surface)
    except Exception as error:
        raise FaceWorkplaneResolutionError(
            "face-workplane.face-stale",
            "所选工作面已失效，无法由原逻辑面 ID 唯一恢复",
        ) from error
    if geometry_type.casefold() != "plane":
        raise FaceWorkplaneResolutionError(
            "face-workplane.not-plane",
            "所选工作面不是解析平面",
        )

    try:
        adjacent_volumes = tuple(cad.adjacent(surface, dimension=3))
    except Exception as error:
        raise FaceWorkplaneResolutionError(
            "face-workplane.body-ambiguous",
            "工作面无法唯一确定目标 Body",
        ) from error
    if len(adjacent_volumes) != 1:
        raise FaceWorkplaneResolutionError(
            "face-workplane.body-ambiguous",
            "工作面无法唯一确定目标 Body",
        )
    volume = adjacent_volumes[0]
    body_id = _resolve_target_body_id(entity_mapping, volume)

    try:
        outward = tuple(cad.outward_surface_normal(surface, volume))
        origin = tuple(cad.center_of_mass(surface))
        area = float(cad.area(surface))
        normalized_strategy, u_axis = _resolve_u_axis(outward, strategy)
        v_axis = _normalized_cross(outward, u_axis)
        plane = SketchPlane(origin, u_axis, v_axis)
    except FaceWorkplaneResolutionError:
        raise
    except Exception as error:
        raise FaceWorkplaneResolutionError(
            "face-workplane.frame-unresolved",
            "工作面坐标系无法按保存策略恢复",
        ) from error
    if not math.isfinite(area) or area <= 0.0:
        raise FaceWorkplaneResolutionError(
            "face-workplane.frame-unresolved",
            "工作面坐标系无法按保存策略恢复",
        )
    return ResolvedFaceWorkplane(
        reference.logical_id,
        body_id,
        surface,
        volume,
        plane,
        outward,
        normalized_strategy,
        area,
    )


def _resolve_target_body_id(
    logical_entities: Mapping[str, tuple[Any, ...]],
    volume: Any,
) -> str:
    matches: list[str] = []
    for logical_id, entities in logical_entities.items():
        try:
            reference = LogicalEntityRef(logical_id)
        except (TypeError, ValueError):
            continue
        if reference.kind == "body" and volume in tuple(entities):
            matches.append(logical_id)
    specific = sorted(value for value in matches if value != "body:domain")
    if len(specific) == 1:
        return specific[0]
    if not specific and matches.count("body:domain") == 1:
        domain = tuple(logical_entities["body:domain"])
        if len(domain) == 1:
            return "body:domain"
    raise FaceWorkplaneResolutionError(
        "face-workplane.body-ambiguous",
        "工作面无法唯一确定目标 Body",
    )


def _resolve_u_axis(
    outward: tuple[float, ...],
    strategy: FaceSketchWorkplaneStrategy | None,
) -> tuple[FaceSketchWorkplaneStrategy, tuple[float, float, float]]:
    if len(outward) != 3:
        raise ValueError("外法向必须包含三个分量")
    magnitude = math.sqrt(sum(value * value for value in outward))
    if not math.isfinite(magnitude) or magnitude <= _FRAME_TOLERANCE:
        raise ValueError("外法向无效")
    normal = tuple(value / magnitude for value in outward)
    candidates = (
        tuple(_AXES)
        if strategy is None
        else (strategy.seed_axis,)
    )
    for axis_name in candidates:
        seed = _AXES[axis_name]
        dot = sum(left * right for left, right in zip(seed, normal, strict=True))
        projected = tuple(
            component - dot * normal_component
            for component, normal_component in zip(seed, normal, strict=True)
        )
        projected_magnitude = math.sqrt(sum(value * value for value in projected))
        if projected_magnitude <= _FRAME_TOLERANCE:
            continue
        sign = 1 if strategy is None else strategy.sign
        u_axis = tuple(sign * value / projected_magnitude for value in projected)
        return FaceSketchWorkplaneStrategy(axis_name, sign), u_axis
    raise FaceWorkplaneResolutionError(
        "face-workplane.frame-unresolved",
        "工作面坐标系无法按保存策略恢复",
    )


def _normalized_cross(
    left: tuple[float, ...],
    right: tuple[float, ...],
) -> tuple[float, float, float]:
    values = (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )
    magnitude = math.sqrt(sum(value * value for value in values))
    if not math.isfinite(magnitude) or magnitude <= _FRAME_TOLERANCE:
        raise ValueError("工作面 V 轴无法恢复")
    return tuple(value / magnitude for value in values)


__all__ = [
    "FaceWorkplaneResolutionError",
    "ResolvedFaceWorkplane",
    "resolve_face_workplane",
]
