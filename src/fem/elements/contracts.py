"""Reference-element contracts independent of physics and materials."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class ElementDefinition(Protocol):
    """Topology, interpolation, quadrature, and DOF requirements of a cell."""

    canonical_type: str
    aliases: tuple[str, ...]
    node_count: int

    def gauss_points(self) -> tuple[tuple[float, ...], ...]:
        """Return parent coordinates followed by an integration weight."""

    @staticmethod
    def shape_functions(*coordinates: float) -> np.ndarray:
        """Evaluate interpolation functions."""

    @staticmethod
    def shape_gradients(*coordinates: float) -> np.ndarray:
        """Evaluate parent-coordinate interpolation gradients."""


__all__ = ["ElementDefinition"]
