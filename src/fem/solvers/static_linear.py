from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Literal, overload

from .. import materials
from ..assemble import assemble_global_stiffness_sparse
from ..boundary import step as _boundary_step
from ..boundary.constraints import apply_dirichlet
from ..boundary.loads import build_load_vector
from ..core.model import AnalysisStep
from ..core.result import ModelResult, ModelResults
from ..core.validation import validate_model
from . import linear


__all__ = ["solve"]


StepSelector = str | int | AnalysisStep


@dataclass(frozen=True)
class _ResolvedSelection:
    """Resolved static steps and the result mode chosen by the call shape."""

    steps: tuple[AnalysisStep | None, ...]
    plural: bool


@overload
def solve(
    model: Any,
    step: StepSelector | None = None,
    name: str | None = None,
    *,
    steps: None = None,
) -> ModelResult: ...


@overload
def solve(
    model: Any,
    step: None = None,
    name: str | None = None,
    *,
    steps: Literal["all"] | Iterable[StepSelector],
) -> ModelResults: ...


def solve(
    model: Any,
    step: StepSelector | None = None,
    name: str | None = None,
    *,
    steps: Literal["all"] | Iterable[StepSelector] | None = None,
) -> ModelResult | ModelResults:
    """Solve one or more independent linear static model steps."""
    selection = _resolve_selection(model, step, steps)
    _validate_selection(model, selection.steps)

    materials.apply_sections(model)
    boundaries = tuple(
        _boundary_step.boundary_for_step(model, selected_step)
        for selected_step in selection.steps
    )
    base_stiffness = assemble_global_stiffness_sparse(model.mesh)

    results = tuple(
        _solve_prepared_step(
            model,
            selected_step,
            boundary,
            base_stiffness,
            _result_name(model, selected_step, name, selection.plural),
        )
        for selected_step, boundary in zip(selection.steps, boundaries)
    )
    if selection.plural:
        return ModelResults(model, results)
    return results[0]


def _resolve_selection(
    model: Any,
    step: StepSelector | None,
    steps: Literal["all"] | Iterable[StepSelector] | None,
) -> _ResolvedSelection:
    """Resolve scalar or plural selectors before model preparation begins."""
    if steps is None:
        return _ResolvedSelection((_resolve_step(model, step),), False)
    if step is not None:
        raise ValueError("step and steps are mutually exclusive")

    selected_steps = _resolve_plural_steps(model, steps)
    _reject_duplicate_steps(selected_steps)
    return _ResolvedSelection(selected_steps, True)


def _resolve_plural_steps(
    model: Any,
    steps: Literal["all"] | Iterable[StepSelector],
) -> tuple[AnalysisStep | None, ...]:
    """Resolve a plural step selection while preserving its order."""
    if isinstance(steps, str):
        if steps != "all":
            raise TypeError("steps must be the exact string 'all' or an iterable of selectors")
        runnable = tuple(
            candidate
            for candidate in model.steps
            if candidate.name.lower() != "initial"
        )
        if runnable:
            return runnable
        if model.steps:
            return (model.steps[0],)
        return (None,)
    if isinstance(steps, (bytes, bytearray)):
        raise TypeError("steps must be the exact string 'all' or an iterable of selectors")

    try:
        selectors = tuple(steps)
    except TypeError as exc:
        raise TypeError("steps must be the exact string 'all' or an iterable of selectors") from exc
    if not selectors:
        raise ValueError("steps must contain at least one selector")

    resolved: list[AnalysisStep | None] = []
    for selector in selectors:
        if selector is None:
            raise TypeError("plural step selectors cannot be None")
        resolved.append(_resolve_step(model, selector))
    return tuple(resolved)


def _resolve_step(
    model: Any,
    selector: StepSelector | None,
) -> AnalysisStep | None:
    """Resolve one valid scalar selector through the canonical step resolver."""
    if selector is not None and (
        isinstance(selector, bool)
        or not isinstance(selector, (str, int, AnalysisStep))
    ):
        raise TypeError("step selector must be a step name, index, or AnalysisStep")
    return _boundary_step.get_step(model, selector)


def _reject_duplicate_steps(steps: tuple[AnalysisStep | None, ...]) -> None:
    """Reject repeated steps and result-name collisions in one selection."""
    seen: set[int] = set()
    seen_names: dict[str, str] = {}
    for selected_step in steps:
        identity = id(selected_step)
        if identity in seen:
            step_name = selected_step.name if selected_step is not None else "step"
            raise ValueError(f"analysis step {step_name} was selected more than once")
        seen.add(identity)
        step_name = selected_step.name if selected_step is not None else "step"
        name_key = str(step_name).casefold()
        if name_key in seen_names:
            raise ValueError(
                "selected analysis step names must be unique ignoring case; "
                f"got {seen_names[name_key]!r} and {step_name!r}"
            )
        seen_names[name_key] = str(step_name)


def _validate_selection(
    model: Any,
    steps: tuple[AnalysisStep | None, ...],
) -> None:
    """Validate every selected load case before preparing the shared system."""
    for selected_step in steps:
        validate_model(model, selected_step)
        _validate_static_step(selected_step)


def _solve_prepared_step(
    model: Any,
    selected_step: AnalysisStep | None,
    boundary: Any,
    base_stiffness: Any,
    name: str | None,
) -> ModelResult:
    """Solve one load case against an already prepared base stiffness."""
    load = build_load_vector(model.mesh, boundary)
    constrained_stiffness, constrained_load = apply_dirichlet(
        base_stiffness,
        load,
        boundary,
    )
    displacement = linear.solve(constrained_stiffness, constrained_load)
    reactions = base_stiffness @ displacement - load
    return ModelResult(
        model,
        selected_step,
        displacement,
        reactions,
        name=name,
    )


def _result_name(
    model: Any,
    step: AnalysisStep | None,
    name: str | None,
    plural: bool,
) -> str | None:
    """Return a scalar name unchanged or suffix a plural result name."""
    if not plural:
        return name
    base = name if name is not None else (model.name or "result")
    step_name = step.name if step is not None else "step"
    return f"{base}_{step_name}"


def _validate_static_step(step: AnalysisStep | None) -> None:
    """Reject procedures and geometric nonlinearity unsupported by this solver."""
    if step is None:
        return
    if str(step.procedure).strip().lower() != "static":
        raise ValueError(
            f"static_linear solver requires procedure 'static', got {step.procedure!r}"
        )
    nlgeom = next(
        (
            value
            for key, value in step.metadata.items()
            if str(key).strip().lower() == "nlgeom"
        ),
        None,
    )
    if _truthy_option(nlgeom):
        raise ValueError("static_linear solver does not support nlgeom")


def _truthy_option(value: Any) -> bool:
    """Return semantic truth for common bool-like analysis options."""
    if value is None:
        return False
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"", "0", "false", "no", "off"}:
            return False
        if normalized in {"1", "true", "yes", "on"}:
            return True
    return bool(value)
