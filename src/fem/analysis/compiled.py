"""Immutable output of the authoring-to-runtime analysis compiler."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fem.model import DofSpace
from fem.problem import Problem

from .contracts import AnalysisRequest


@dataclass(frozen=True, slots=True)
class CompiledSystem:
    """Detached executable equation graph owned by one analysis request."""

    problem: Problem

    def __post_init__(self) -> None:
        if not isinstance(self.problem, Problem):
            raise TypeError("compiled system must implement the Problem contract")

    @property
    def dof_space(self) -> DofSpace:
        return self.problem.dof_space

    @property
    def assembly(self) -> Any:
        return getattr(self.problem, "assembly")


@dataclass(frozen=True, slots=True)
class CompiledAnalysis:
    """Complete numerical system consumed by a procedure driver.

    Authoring metadata, region names, section assignments, and element-type
    dispatch have already been resolved before this object is created.
    """

    request: AnalysisRequest
    system: CompiledSystem
    result_model: Any | None = None

    def __post_init__(self) -> None:
        if type(self.request) is not AnalysisRequest:
            raise TypeError("request must be exactly AnalysisRequest")
        if not isinstance(self.system, CompiledSystem):
            raise TypeError("system must be exactly CompiledSystem")
        if self.result_model is not None and not hasattr(
            getattr(self.result_model, "mesh", None),
            "elements",
        ):
            raise TypeError("compiled result model must expose mesh elements")

    @property
    def problem(self) -> Problem:
        """Return the equation problem held by the compiled system."""

        return self.system.problem

    @property
    def dof_space(self) -> DofSpace:
        return self.system.dof_space

    @property
    def assembly(self) -> Any:
        """Return the compiled global assembly, never an authored mesh."""

        return self.system.assembly

    @property
    def bindings(self) -> tuple[Any, ...]:
        """Return immutable local operator bindings in the compiled graph."""

        return tuple(getattr(self.assembly, "bindings"))

    @property
    def constraints(self) -> Any:
        """Return compiled equation constraints."""

        return getattr(self.system.problem, "constraints")

    @property
    def reference_load(self) -> Any:
        """Return the owned reference load vector."""

        return self.system.problem.reference_load

    @property
    def solver_config(self) -> tuple[Any, Any]:
        """Return the typed procedure controls and options snapshot."""

        return self.request.controls, self.request.options

    def model_for_result(self) -> Any | None:
        """Return the detached effective-property model for post-processing."""

        return self.result_model


__all__ = ["CompiledAnalysis", "CompiledSystem"]
