"""Read-only queries over authored analysis-step definitions."""

from __future__ import annotations

from typing import Any

from .entities import AnalysisStep, AnalysisStepSnapshot


def resolve_analysis_step(
    model: Any,
    step: str | int | AnalysisStep | None = None,
) -> AnalysisStep | None:
    """Resolve a step selector without compiling numerical boundary data."""

    if step is None:
        for candidate in model.steps:
            if candidate.name.lower() != "initial":
                return candidate
        return model.steps[0] if model.steps else None
    if isinstance(step, AnalysisStep):
        return step
    if isinstance(step, int):
        return model.steps[step]
    for candidate in model.steps:
        if candidate.name == step:
            return candidate
    raise KeyError(f"analysis step {step} is not defined")


def effective_displacement_constraints(
    model: Any,
    step: str | int | AnalysisStep | AnalysisStepSnapshot | None = None,
) -> tuple[Any, ...]:
    """Return cumulative authored displacement constraints through one step."""

    if isinstance(step, AnalysisStepSnapshot):
        selected_index = next(
            (
                index
                for index, candidate in enumerate(model.steps)
                if str(candidate.name) == step.name
            ),
            None,
        )
        if selected_index is None:
            raise KeyError(f"analysis step {step.name} is not defined")
        return tuple(
            constraint
            for candidate in model.steps[:selected_index]
            for constraint in candidate.boundaries
        ) + tuple(step.boundaries)

    selected = (
        step
        if step is not None and any(candidate is step for candidate in model.steps)
        else resolve_analysis_step(model, step)
    )
    if selected is None:
        return ()
    selected_index = next(
        (
            index
            for index, candidate in enumerate(model.steps)
            if candidate is selected
        ),
        None,
    )
    if selected_index is None:
        selected_name = str(selected.name)
        selected_index = next(
            (
                index
                for index, candidate in enumerate(model.steps)
                if str(candidate.name) == selected_name
            ),
            None,
        )
    if selected_index is None:
        return (
            *(
                constraint
                for candidate in model.steps
                for constraint in candidate.boundaries
            ),
            *selected.boundaries,
        )
    return tuple(
        constraint
        for candidate in model.steps[: selected_index + 1]
        for constraint in candidate.boundaries
    )


__all__ = ["effective_displacement_constraints", "resolve_analysis_step"]
