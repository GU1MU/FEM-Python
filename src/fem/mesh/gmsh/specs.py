"""Immutable public mesh specifications for native Gmsh generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ._validation import (
    _positive_float,
    _validate_order,
    _validate_spec_level,
)


_CellShape = Literal["tri", "tri-quad", "quad", "tet", "hex"]
_CELL_SHAPES = frozenset({"tri", "tri-quad", "quad", "tet", "hex"})


@dataclass(frozen=True, slots=True)
class MeshSpec:
    """Immutable controls for explicit native Gmsh mesh generation."""

    size: float | None = None
    order: Literal[1, 2] = 1
    recombine: bool = False

    def __post_init__(self) -> None:
        normalized_size = (
            None if self.size is None else _positive_float(self.size, "size")
        )
        object.__setattr__(self, "size", normalized_size)
        _validate_order(self.order)
        if not isinstance(self.recombine, bool):
            raise TypeError(
                f"recombine must be a boolean, got {self.recombine!r}"
            )


@dataclass(frozen=True, slots=True)
class AutoMeshSpec:
    """Immutable controls for level-scaled strict-shape mesh generation."""

    level: Literal[1, 2, 3, 4, 5] = 3
    cell_shape: _CellShape | None = None
    order: Literal[1, 2] = 1

    def __post_init__(self) -> None:
        _validate_spec_level(self.level)
        if self.cell_shape is not None and (
            not isinstance(self.cell_shape, str)
            or self.cell_shape not in _CELL_SHAPES
        ):
            raise ValueError(
                "cell_shape must be exactly 'tri', 'tri-quad', 'quad', "
                f"'tet', 'hex', or None, got {self.cell_shape!r}"
            )
        _validate_order(self.order)


__all__ = ["AutoMeshSpec", "MeshSpec"]
