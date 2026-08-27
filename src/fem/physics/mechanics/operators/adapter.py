"""Adapter from mesh-aware linear mechanics services to physics evaluation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from fem.physics.contracts import LocalContribution
from fem.state import EvaluationContext, LocalFieldState, StateManager


@dataclass(frozen=True, slots=True)
class MechanicalOperatorAdapter:
    """Present an element service as a :class:`PhysicsOperator`."""

    mesh: Any
    operator: Any

    def initialize(
        self,
        element: Any,
        resources: Mapping[str, Any],
        state: StateManager | None,
        properties: Mapping[str, Any] | None = None,
        *,
        state_namespace: str,
    ) -> None:
        del element, resources, state, properties, state_namespace

    def evaluate(
        self,
        element: Any,
        reference_coordinates: np.ndarray,
        fields: Mapping[str, LocalFieldState],
        dofs: tuple[int, ...],
        resources: Mapping[str, Any],
        state: StateManager | None,
        properties: Mapping[str, Any] | None = None,
        *,
        context: EvaluationContext,
        state_namespace: str,
    ) -> LocalContribution:
        del resources, state, state_namespace
        try:
            displacement = fields["U"].values
        except KeyError as exc:
            raise ValueError("linear mechanics requires local field 'U'") from exc
        local_method = getattr(self.operator, "linear_contribution", None)
        if callable(local_method):
            return local_method(
                element,
                reference_coordinates,
                displacement,
                dofs,
                properties,
                time=context.time,
                temperature=context.parameters.get("temperature"),
            )
        tangent = np.asarray(
            self.operator.stiffness(self.mesh, element),
            dtype=float,
        )
        local = np.asarray(displacement, dtype=float).reshape(-1)
        return LocalContribution(
            dofs=dofs,
            residual=tangent @ local,
            tangent=tangent,
        )


__all__ = ["MechanicalOperatorAdapter"]
