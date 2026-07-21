from __future__ import annotations

from typing import Any

from .. import materials
from ..assemble import assemble_global_stiffness_sparse
from ..boundary import step as _boundary_step
from ..boundary.constraints import apply_dirichlet
from ..boundary.loads import build_load_vector
from ..core.model import AnalysisStep
from ..core.result import ModelResult, ModelResults
from ..core.validation import validate_model
from . import linear


__all__ = ["solve", "solve_all"]


def solve(
    model: Any,
    step: str | int | AnalysisStep | None = None,
    name: str | None = None,
) -> ModelResult:
    """Solve one linear static model step."""
    selected_step = _boundary_step.get_step(model, step)
    validate_model(model, selected_step)
    _validate_static_step(selected_step)
    materials.apply_sections(model)
    boundary = _boundary_step.boundary_for_step(model, selected_step)
    K = assemble_global_stiffness_sparse(model.mesh)
    F = build_load_vector(model.mesh, boundary)
    K_mod, F_mod = apply_dirichlet(K, F, boundary)
    U = linear.solve(K_mod, F_mod)
    reactions = K @ U - F
    return ModelResult(
        model,
        selected_step,
        U,
        reactions,
        name=name,
    )


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


def solve_all(
    model: Any,
    selected_steps: Any = None,
    name: str | None = None,
) -> ModelResults:
    """Solve multiple non-initial model steps."""
    steps = _solve_all_steps(model, selected_steps)
    multi_step = len(steps) > 1
    results = tuple(
        solve(
            model,
            step,
            name=_result_name(model, step, name, multi_step),
        )
        for step in steps
    )
    return ModelResults(model, results)


def _solve_all_steps(model: Any, steps: Any) -> tuple[AnalysisStep | None, ...]:
    """Resolve solve_all step selectors."""
    if steps is None:
        runnable = tuple(step for step in model.steps if step.name.lower() != "initial")
        if runnable:
            return runnable
        if model.steps:
            return (model.steps[0],)
        return (None,)
    if isinstance(steps, (str, int, AnalysisStep)):
        return (_boundary_step.get_step(model, steps),)
    return tuple(_boundary_step.get_step(model, step) for step in steps)


def _result_name(
    model: Any,
    step: AnalysisStep | None,
    name: str | None,
    multi_step: bool,
) -> str | None:
    """Return a non-conflicting result name for solve_all."""
    if not multi_step:
        return name
    base = name or model.name or "result"
    step_name = step.name if step is not None else "step"
    return f"{base}_{step_name}"
