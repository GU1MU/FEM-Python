from __future__ import annotations

from time import perf_counter
from typing import Any

from .. import materials
from ..assemble import assemble_global_stiffness_sparse
from ..boundary.constraints import apply_dirichlet
from ..boundary.loads import build_load_vector
from ..boundary.step import boundary_for_step, get_step
from ..core.model import AnalysisStep
from ..core.result import ModelResult, ModelResults
from ..core.validation import validate_model
from . import linear


def solve(
    model: Any,
    step: str | int | AnalysisStep | None = None,
    name: str | None = None,
    *,
    _validated_step: AnalysisStep | None | object = ...,
    timings: dict[str, float] | None = None,
) -> ModelResult:
    """Solve one linear static model step."""
    if _validated_step is ...:
        started = perf_counter()
        selected_step = validate_problem(model, step)
        _record_timing(timings, "模型验证", started)
    else:
        selected_step = _validated_step

    started = perf_counter()
    materials.apply_sections(model)
    boundary = boundary_for_step(model, selected_step)
    _record_timing(timings, "分析准备", started)

    started = perf_counter()
    K = assemble_global_stiffness_sparse(model.mesh)
    _record_timing(timings, "刚度矩阵装配", started)

    started = perf_counter()
    F = build_load_vector(model.mesh, boundary)
    K_mod, F_mod = apply_dirichlet(K, F, boundary)
    _record_timing(timings, "载荷与边界条件", started)

    started = perf_counter()
    U = linear.solve(K_mod, F_mod)
    _record_timing(timings, "线性方程求解", started)

    started = perf_counter()
    reactions = K @ U - F
    result = ModelResult(
        model,
        selected_step,
        U,
        reactions,
        name=name,
    )
    _record_timing(timings, "反力与结果封装", started)
    return result


def _record_timing(
    timings: dict[str, float] | None,
    name: str,
    started: float,
) -> None:
    """Record one optional solver stage without coupling the solver to the GUI."""
    if timings is not None:
        timings[name] = perf_counter() - started


def validate_problem(
    model: Any,
    step: str | int | AnalysisStep | None = None,
) -> AnalysisStep | None:
    """使用与线性静力求解完全一致的规则验证模型和分析步。"""
    selected_step = get_step(model, step)
    validate_model(model, selected_step)
    _validate_static_step(selected_step)
    return selected_step


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
        return (get_step(model, steps),)
    return tuple(get_step(model, step) for step in steps)


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
