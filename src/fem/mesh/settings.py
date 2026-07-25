"""Headless mesh settings for native model authoring."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class LocalMeshControl:
    """One local size attached to a stable logical preview entity."""

    entity_kind: Literal["point", "edge", "face"]
    entity_id: int
    size: float

    def __post_init__(self) -> None:
        if self.entity_kind not in {"point", "edge", "face"}:
            raise ValueError("局部网格控制只支持点、边或面")
        if int(self.entity_id) <= 0:
            raise ValueError("几何实体编号必须大于零")
        if float(self.size) <= 0.0:
            raise ValueError("局部网格尺寸必须大于零")
        object.__setattr__(self, "entity_id", int(self.entity_id))
        object.__setattr__(self, "size", float(self.size))


@dataclass(frozen=True, slots=True)
class MeshSettings:
    """Global settings for the first native geometry-to-mesh workflow."""

    size: float
    order: Literal[1, 2] = 1
    cell_shape: Literal[
        "triangle", "quadrilateral", "tetrahedron", "hexahedron"
    ] = "triangle"
    local_size: float | None = None
    local_controls: tuple[LocalMeshControl, ...] = ()

    def __post_init__(self) -> None:
        if float(self.size) <= 0.0:
            raise ValueError("全局网格尺寸必须大于零")
        if self.order not in (1, 2):
            raise ValueError("单元阶次只能是一阶或二阶")
        if self.cell_shape not in {
            "triangle",
            "quadrilateral",
            "tetrahedron",
            "hexahedron",
        }:
            raise ValueError("网格类型只能是三角形、四边形、四面体或六面体")
        if self.local_size is not None and float(self.local_size) <= 0.0:
            raise ValueError("局部网格尺寸必须大于零")
        if self.local_size is not None and float(self.local_size) >= float(self.size):
            raise ValueError("局部网格尺寸必须小于全局尺寸")
        controls = tuple(self.local_controls)
        if any(control.size >= float(self.size) for control in controls):
            raise ValueError("实体局部尺寸必须小于全局尺寸")
        keys = {(control.entity_kind, control.entity_id) for control in controls}
        if len(keys) != len(controls):
            raise ValueError("同一个几何实体不能重复设置局部尺寸")
        object.__setattr__(self, "size", float(self.size))
        object.__setattr__(self, "local_controls", controls)
        if self.local_size is not None:
            object.__setattr__(self, "local_size", float(self.local_size))


__all__ = ["LocalMeshControl", "MeshSettings"]
