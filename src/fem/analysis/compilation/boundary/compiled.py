"""Compile authored boundary data into constraints and loads."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fem.constraints import ConstraintSet
from fem.loads import LoadSet

from .condition import BoundaryCondition
from .step import boundary_for_step


@dataclass(frozen=True, slots=True)
class CompiledBoundary:
    """Separated equation constraints and external loads for one step."""

    constraints: ConstraintSet
    loads: LoadSet

    @classmethod
    def from_boundary(cls, boundary: BoundaryCondition) -> "CompiledBoundary":
        if type(boundary) is not BoundaryCondition:
            raise TypeError("boundary must be BoundaryCondition")
        return cls(
            constraints=ConstraintSet(boundary.prescribed_displacements),
            loads=LoadSet(
                nodal_forces=boundary.nodal_forces,
                body_forces=tuple(boundary.body_forces),
                surface_tractions=tuple(boundary.surface_tractions),
                edge_tractions=tuple(boundary.edge_tractions),
                gravity=boundary.gravity,
                line_loads=tuple(boundary.line_loads),
                element_gravities=tuple(boundary.element_gravities),
            ),
        )


def compile_boundary(model: Any, step: Any = None) -> CompiledBoundary:
    """Resolve one authored step without exposing its mutable DTO downstream."""

    return CompiledBoundary.from_boundary(boundary_for_step(model, step))


__all__ = ["CompiledBoundary", "compile_boundary"]
