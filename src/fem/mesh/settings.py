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
            raise ValueError("Local mesh controls support only points, edges, or faces")
        if (
            isinstance(self.size, bool)
            or not isinstance(self.size, (int, float))
            or not math.isfinite(float(self.size))
            or float(self.size) <= 0.0
        ):
            raise ValueError("Local mesh size must be greater than zero")
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
    auto_level: Literal[1, 2, 3, 4, 5] | None = None
    strict_cell_shape: bool = False

    def __post_init__(self) -> None:
        if (
            isinstance(self.size, bool)
            or not isinstance(self.size, (int, float))
            or not math.isfinite(float(self.size))
            or float(self.size) <= 0.0
        ):
            raise ValueError("Global mesh size must be greater than zero")
        if isinstance(self.order, bool) or self.order not in (1, 2):
            raise ValueError("Element order must be 1 or 2")
        if self.auto_level is not None and (
            isinstance(self.auto_level, bool)
            or type(self.auto_level) is not int
            or self.auto_level not in {1, 2, 3, 4, 5}
        ):
            raise ValueError("AutoMesh level must be a strict integer from 1 to 5")
        if type(self.strict_cell_shape) is not bool:
            raise TypeError("strict_cell_shape must be a bool")
        if type(self.cell_shape) is not str or self.cell_shape not in {
            "line",
            "triangle",
            "quadrilateral",
            "tetrahedron",
            "hexahedron",
        }:
            raise ValueError(
                "Mesh type must be line, triangle, quadrilateral, tetrahedron, or hexahedron"
            )
        if self.cell_shape == "line":
            if type(self.line_element_type) is not str or self.line_element_type not in {
                "Truss2",
                "Beam2",
            }:
                raise ValueError(
                    "A line mesh must explicitly specify the Truss2 or Beam2 element type"
                )
            if self.order != 1:
                raise ValueError("Line meshes support only first-order, two-node elements")
        elif self.line_element_type is not None:
            raise ValueError(
                "Only line meshes can specify the Truss2 or Beam2 element type"
            )
        controls = tuple(self.local_controls)
        if any(type(control) is not LocalMeshControl for control in controls):
            raise TypeError(
                "local_controls must contain only LocalMeshControl values"
            )
        if any(control.size >= float(self.size) for control in controls):
            raise ValueError("Entity local size must be smaller than global size")
        keys = {(control.target, control.falloff) for control in controls}
        if len(keys) != len(controls):
            raise ValueError("Local size cannot be set more than once for the same geometric entity and falloff profile")
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
