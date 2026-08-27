"""Shared presentation labels for typed analysis formulations."""

from __future__ import annotations

from fem.analysis import (
    ExecutionStrategy,
    resolve_analysis_request,
    resolve_execution_plan,
    resolve_formulation,
)
from fem.model import GeometryMode, StaticFormulation, resolve_analysis_step
from fem.model import DynamicProcedureKind, DynamicStepControls


_FORMULATION_LABELS = {
    StaticFormulation.LINEAR: "线性静力",
    StaticFormulation.NONLINEAR: "非线性静力",
}

_MATERIAL_LABELS = {
    "linear_elastic": "线性弹性",
    "j2_plasticity": "J2 塑性",
}


def analysis_formulation_label(formulation: StaticFormulation) -> str:
    """Return the visible label for one typed static formulation."""

    if not isinstance(formulation, StaticFormulation):
        raise TypeError("formulation must be StaticFormulation")
    return _FORMULATION_LABELS[formulation]


def analysis_step_label(step: object) -> str:
    """Resolve one authored step and return its solver label."""

    if str(getattr(step, "procedure", "static")).strip().casefold() == "dynamic":
        controls = getattr(step, "controls", None)
        if not isinstance(controls, DynamicStepControls):
            controls = DynamicStepControls.from_metadata(
                getattr(step, "metadata", {})
            )
        return (
            "动力学-显式"
            if controls.procedure_kind is DynamicProcedureKind.EXPLICIT
            else "动力学-隐式"
        )
    return analysis_formulation_label(resolve_formulation(step))


def analysis_execution_label(model: object, step: object) -> str:
    """Describe the actual geometry/material execution plan for the GUI."""

    selected = resolve_analysis_step(model, step)
    if selected is None:
        raise ValueError("analysis step is required")
    request = resolve_analysis_request(selected)
    plan = resolve_execution_plan(model, request)
    if plan.strategy in {
        ExecutionStrategy.IMPLICIT_DYNAMIC_LINEAR,
        ExecutionStrategy.IMPLICIT_DYNAMIC_NONLINEAR,
        ExecutionStrategy.EXPLICIT_DYNAMIC_LINEAR,
        ExecutionStrategy.EXPLICIT_DYNAMIC_NONLINEAR,
    }:
        implicit = plan.strategy in {
            ExecutionStrategy.IMPLICIT_DYNAMIC_LINEAR,
            ExecutionStrategy.IMPLICIT_DYNAMIC_NONLINEAR,
        }
        nonlinear = plan.strategy in {
            ExecutionStrategy.IMPLICIT_DYNAMIC_NONLINEAR,
            ExecutionStrategy.EXPLICIT_DYNAMIC_NONLINEAR,
        }
        procedure = "动力学-隐式" if implicit else "动力学-显式"
        response = "非线性响应" if nonlinear else "线性响应"
        return f"{procedure}（{response}）"
    geometry = (
        "小应变" if plan.geometry_mode is GeometryMode.SMALL_STRAIN else "有限应变"
    )
    material = "、".join(
        _MATERIAL_LABELS.get(model, model)
        for model in plan.material_models
    )
    if plan.strategy.value == "direct_linear":
        # Keep the familiar label for the direct optimization. The
        # incremental path is deliberately explicit because it can occur
        # with NLGEOM off when the material owns history.
        return "线性静力"
    return f"增量 Newton（{geometry}·{material}）"


__all__ = [
    "analysis_execution_label",
    "analysis_formulation_label",
    "analysis_step_label",
]
