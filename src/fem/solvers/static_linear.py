from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Literal, overload

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import splu

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
    _validated_step: AnalysisStep | None | object = ...,
    timings: dict[str, float] | None = None,
) -> ModelResult: ...


@overload
def solve(
    model: Any,
    step: None = None,
    name: str | None = None,
    *,
    steps: Literal["all"] | Iterable[StepSelector],
    _validated_step: object = ...,
    timings: dict[str, float] | None = None,
) -> ModelResults: ...


def solve(
    model: Any,
    step: StepSelector | None = None,
    name: str | None = None,
    *,
    steps: Literal["all"] | Iterable[StepSelector] | None = None,
    _validated_step: AnalysisStep | None | object = ...,
    timings: dict[str, float] | None = None,
) -> ModelResult | ModelResults:
    """Solve one or more independent linear static model steps."""
    if _validated_step is not ...:
        if steps is not None:
            raise ValueError("_validated_step is only valid for a scalar solve")
        selection = _ResolvedSelection((_validated_step,), False)
    else:
        started = perf_counter()
        if steps is None:
            selection = _ResolvedSelection((validate_problem(model, step),), False)
        else:
            selection = _resolve_selection(model, step, steps)
            _validate_selection(model, selection.steps)
        _record_timing(timings, "模型验证", started)

    started = perf_counter()
    materials.apply_sections(model)
    boundaries = tuple(
        _boundary_step.boundary_for_step(model, selected_step)
        for selected_step in selection.steps
    )
    _record_timing(timings, "分析准备", started)

    started = perf_counter()
    base_stiffness = assemble_global_stiffness_sparse(model.mesh)
    _record_timing(timings, "刚度矩阵装配", started)

    results = tuple(
        _solve_prepared_step(
            model,
            selected_step,
            boundary,
            base_stiffness,
            _result_name(model, selected_step, name, selection.plural),
            timings,
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
            raise TypeError(
                "steps must be the exact string 'all' or an iterable of selectors"
            )
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
        raise TypeError(
            "steps must be the exact string 'all' or an iterable of selectors"
        )

    try:
        selectors = tuple(steps)
    except TypeError as exc:
        raise TypeError(
            "steps must be the exact string 'all' or an iterable of selectors"
        ) from exc
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
    timings: dict[str, float] | None,
) -> ModelResult:
    """Solve one load case against an already prepared base stiffness."""
    started = perf_counter()
    load = build_load_vector(model.mesh, boundary)
    constrained_stiffness, constrained_load = apply_dirichlet(
        base_stiffness,
        load,
        boundary,
    )
    _record_timing(timings, "载荷与边界条件", started)

    started = perf_counter()
    displacement = linear.solve(constrained_stiffness, constrained_load)
    _record_timing(timings, "线性方程求解", started)

    started = perf_counter()
    reactions = base_stiffness @ displacement - load
    result = ModelResult(
        model,
        selected_step,
        displacement,
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
        timings[name] = timings.get(name, 0.0) + perf_counter() - started


def validate_problem(
    model: Any,
    step: StepSelector | None = None,
) -> AnalysisStep | None:
    """使用与线性静力求解完全一致的规则验证模型和分析步。"""
    selected_step = _resolve_step(model, step)
    validate_model(model, selected_step)
    _validate_static_step(selected_step)
    return selected_step


def validate_stiffness(
    model: Any,
    step: StepSelector | None = None,
) -> AnalysisStep | None:
    """Verify that assigned sections and constraints produce a solvable matrix."""
    selected_step = validate_problem(model, step)
    materials.apply_sections(model)
    boundary = _boundary_step.boundary_for_step(model, selected_step)
    stiffness = assemble_global_stiffness_sparse(model.mesh)
    zero_load = np.zeros(model.mesh.num_dofs, dtype=float)
    constrained_stiffness, _ = apply_dirichlet(
        stiffness,
        zero_load,
        boundary,
    )
    try:
        _validate_nonsingular_stiffness(constrained_stiffness)
    except (RuntimeError, ValueError) as error:
        raise ValueError(
            "模型约束不足或刚度矩阵奇异；请检查刚体位移、材料、截面和单元连接"
        ) from error
    return selected_step


def validate_constraint_stability(model: Any, boundary: Any) -> None:
    """Reject unconstrained rigid-body modes without assembling stiffness."""
    mesh = model.mesh
    node_lookup = {int(node.id): node for node in mesh.nodes}
    components = _connected_node_components(mesh)
    constrained_dofs = set(boundary.prescribed_displacements)
    dofs_per_node = int(mesh.dofs_per_node)
    is_3d = any(hasattr(node, "z") for node in mesh.nodes)
    mode_count = 6 if is_3d else 3

    for node_ids in components:
        coordinates = np.asarray(
            [
                [
                    float(node_lookup[node_id].x),
                    float(node_lookup[node_id].y),
                    float(getattr(node_lookup[node_id], "z", 0.0)),
                ]
                for node_id in node_ids
            ],
            dtype=float,
        )
        center = np.mean(coordinates, axis=0)
        rows: list[np.ndarray] = []
        for node_id, coordinate in zip(node_ids, coordinates):
            relative = coordinate - center
            for component in range(dofs_per_node):
                dof_id = mesh.global_dof(node_id, component)
                if dof_id not in constrained_dofs:
                    continue
                row = _rigid_mode_row(relative, component, is_3d)
                if row is not None:
                    rows.append(row)
        rank = (
            int(np.linalg.matrix_rank(np.asarray(rows, dtype=float)))
            if rows
            else 0
        )
        if rank < mode_count:
            raise ValueError(
                "模型约束不足或刚度矩阵奇异；"
                "请检查刚体位移、材料、截面和单元连接"
            )


def _connected_node_components(mesh: Any) -> tuple[tuple[int, ...], ...]:
    """Return connectivity components using element-node incidence only."""
    parents = {int(node.id): int(node.id) for node in mesh.nodes}

    def root(node_id: int) -> int:
        while parents[node_id] != node_id:
            parents[node_id] = parents[parents[node_id]]
            node_id = parents[node_id]
        return node_id

    for element in mesh.elements:
        node_ids = tuple(int(node_id) for node_id in element.node_ids)
        if not node_ids:
            continue
        first_root = root(node_ids[0])
        for node_id in node_ids[1:]:
            other_root = root(node_id)
            if other_root != first_root:
                parents[other_root] = first_root

    grouped: dict[int, list[int]] = {}
    for node_id in parents:
        grouped.setdefault(root(node_id), []).append(node_id)
    return tuple(tuple(node_ids) for node_ids in grouped.values())


def _rigid_mode_row(
    relative: np.ndarray,
    component: int,
    is_3d: bool,
) -> np.ndarray | None:
    """Return one constrained-DOF row of the rigid-body mode matrix."""
    x, y, z = (float(value) for value in relative)
    if is_3d:
        rows = (
            (1.0, 0.0, 0.0, 0.0, z, -y),
            (0.0, 1.0, 0.0, -z, 0.0, x),
            (0.0, 0.0, 1.0, y, -x, 0.0),
            (0.0, 0.0, 0.0, 1.0, 0.0, 0.0),
            (0.0, 0.0, 0.0, 0.0, 1.0, 0.0),
            (0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        )
    else:
        rows = (
            (1.0, 0.0, -y),
            (0.0, 1.0, x),
            (0.0, 0.0, 1.0),
        )
    if component >= len(rows):
        return None
    return np.asarray(rows[component], dtype=float)


def _validate_nonsingular_stiffness(stiffness: Any) -> None:
    """Check scaled sparse-LU pivots without depending on the load vector."""
    diagonal = np.abs(np.asarray(stiffness.diagonal(), dtype=float))
    if (
        diagonal.size != stiffness.shape[0]
        or not np.all(np.isfinite(diagonal))
        or np.any(diagonal <= 0.0)
    ):
        raise ValueError("stiffness matrix has a zero or invalid diagonal")
    inverse_scale = 1.0 / np.sqrt(diagonal)
    scaling = diags(inverse_scale)
    scaled = (scaling @ stiffness @ scaling).tocsc()
    factor = splu(scaled)
    pivots = np.abs(np.asarray(factor.U.diagonal(), dtype=float))
    tolerance = (
        np.finfo(float).eps
        * max(stiffness.shape[0], 1)
        * max(float(np.max(pivots)), 1.0)
    )
    if (
        not np.all(np.isfinite(pivots))
        or float(np.min(pivots)) <= tolerance
    ):
        raise ValueError("stiffness matrix is numerically rank deficient")


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
