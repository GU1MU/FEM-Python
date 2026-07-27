"""Headless mesh settings for native model authoring."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Literal

from fem.geometry import LogicalEntityRef, logical_ref_sort_key


@dataclass(frozen=True, slots=True)
class MeshSizeFalloff:
    """Typed distance range used by one local mesh-size control."""

    reference: Literal["global_size", "target_radius"] = "global_size"
    start_factor: float = 0.0
    end_factor: float = 2.0

    def __post_init__(self) -> None:
        if type(self.reference) is not str or self.reference not in {
            "global_size",
            "target_radius",
        }:
            raise ValueError(
                "mesh-size falloff reference must be 'global_size' or "
                "'target_radius'"
            )
        if (
            isinstance(self.start_factor, bool)
            or not isinstance(self.start_factor, (int, float))
            or not math.isfinite(float(self.start_factor))
        ):
            raise ValueError("mesh-size falloff start_factor must be finite")
        if (
            isinstance(self.end_factor, bool)
            or not isinstance(self.end_factor, (int, float))
            or not math.isfinite(float(self.end_factor))
        ):
            raise ValueError("mesh-size falloff end_factor must be finite")
        start = float(self.start_factor)
        end = float(self.end_factor)
        if start < 0.0 or start >= end:
            raise ValueError(
                "mesh-size falloff requires 0 <= start_factor < end_factor"
            )
        object.__setattr__(self, "start_factor", start)
        object.__setattr__(self, "end_factor", end)


@dataclass(frozen=True, slots=True)
class LocalMeshControl:
    """One local size attached to a stable logical preview entity."""

    target: LogicalEntityRef
    size: float
    falloff: MeshSizeFalloff = field(default_factory=MeshSizeFalloff)

    def __post_init__(self) -> None:
        if type(self.target) is not LogicalEntityRef:
            raise TypeError("local mesh target must be a LogicalEntityRef")
        if self.target.kind not in {"point", "edge", "face"}:
            raise ValueError("局部网格控制只支持点、边或面")
        if (
            isinstance(self.size, bool)
            or not isinstance(self.size, (int, float))
            or not math.isfinite(float(self.size))
            or float(self.size) <= 0.0
        ):
            raise ValueError("局部网格尺寸必须大于零")
        if type(self.falloff) is not MeshSizeFalloff:
            raise TypeError("local mesh falloff must be a MeshSizeFalloff")
        object.__setattr__(self, "size", float(self.size))

    @property
    def entity_kind(self) -> str:
        """Return the target kind for display-only consumers."""

        return self.target.kind


@dataclass(frozen=True, slots=True)
class MeshSettings:
    """Global settings for the first native geometry-to-mesh workflow."""

    size: float
    order: Literal[1, 2] = 1
    cell_shape: Literal[
        "line", "triangle", "quadrilateral", "tetrahedron", "hexahedron"
    ] = "triangle"
    local_controls: tuple[LocalMeshControl, ...] = ()
    line_element_type: Literal["Truss2", "Beam2"] | None = None

    def __post_init__(self) -> None:
        if (
            isinstance(self.size, bool)
            or not isinstance(self.size, (int, float))
            or not math.isfinite(float(self.size))
            or float(self.size) <= 0.0
        ):
            raise ValueError("全局网格尺寸必须大于零")
        if isinstance(self.order, bool) or self.order not in (1, 2):
            raise ValueError("单元阶次只能是一阶或二阶")
        if type(self.cell_shape) is not str or self.cell_shape not in {
            "line",
            "triangle",
            "quadrilateral",
            "tetrahedron",
            "hexahedron",
        }:
            raise ValueError(
                "网格类型只能是线、三角形、四边形、四面体或六面体"
            )
        if self.cell_shape == "line":
            if type(self.line_element_type) is not str or self.line_element_type not in {
                "Truss2",
                "Beam2",
            }:
                raise ValueError(
                    "线网格必须显式指定 Truss2 或 Beam2 单元类型"
                )
            if self.order != 1:
                raise ValueError("线网格只支持一阶两节点单元")
        elif self.line_element_type is not None:
            raise ValueError(
                "只有线网格可以指定 Truss2 或 Beam2 单元类型"
            )
        controls = tuple(self.local_controls)
        if any(type(control) is not LocalMeshControl for control in controls):
            raise TypeError(
                "local_controls must contain only LocalMeshControl values"
            )
        if any(control.size >= float(self.size) for control in controls):
            raise ValueError("实体局部尺寸必须小于全局尺寸")
        keys = {(control.target, control.falloff) for control in controls}
        if len(keys) != len(controls):
            raise ValueError("同一个几何实体和 falloff profile 不能重复设置局部尺寸")
        object.__setattr__(self, "size", float(self.size))
        falloff_order = {"global_size": 0, "target_radius": 1}
        object.__setattr__(
            self,
            "local_controls",
            tuple(
                sorted(
                    controls,
                    key=lambda control: (
                        *logical_ref_sort_key(control.target),
                        falloff_order[control.falloff.reference],
                        control.falloff.start_factor,
                        control.falloff.end_factor,
                    ),
                )
            ),
        )


__all__ = ["LocalMeshControl", "MeshSettings", "MeshSizeFalloff"]
