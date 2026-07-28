from __future__ import annotations

from typing import Sequence

from ..core.model import AnalysisStep, DisplacementConstraint
from ._components import component_ranges


def displacement(
    step: AnalysisStep,
    target: str | int,
    components: int | Sequence[int],
    value: float = 0.0,
    *,
    target_kind: str = "node_set",
) -> tuple[DisplacementConstraint, ...]:
    """Add displacement constraints to a step using 1-based components."""
    constraints = tuple(
        DisplacementConstraint(
            target,
            first,
            last,
            value,
            target_kind=target_kind,
        )
        for first, last in component_ranges(components)
    )
    step.boundaries = tuple(step.boundaries) + constraints
    return constraints
