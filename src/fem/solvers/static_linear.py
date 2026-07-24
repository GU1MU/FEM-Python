from __future__ import annotations

from time import perf_counter
from typing import Any

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import splu

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


def validate_stiffness(
    model: Any,
    step: str | int | AnalysisStep | None = None,
) -> AnalysisStep | None:
    """Verify that assigned sections and constraints produce a solvable matrix."""
    selected_step = validate_problem(model, step)
    materials.apply_sections(model)
    boundary = boundary_for_step(model, selected_step)
    K = assemble_global_stiffness_sparse(model.mesh)
    zero_load = np.zeros(model.mesh.num_dofs, dtype=float)
    K_mod, _F_mod = apply_dirichlet(K, zero_load, boundary)
    try:
        _validate_nonsingular_stiffness(K_mod)
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


def _validate_nonsingular_stiffness(K: Any) -> None:
    """Check scaled sparse-LU pivots without depending on the load vector."""
    diagonal = np.abs(np.asarray(K.diagonal(), dtype=float))
    if (
        diagonal.size != K.shape[0]
        or not np.all(np.isfinite(diagonal))
        or np.any(diagonal <= 0.0)
    ):
        raise ValueError("stiffness matrix has a zero or invalid diagonal")
    inverse_scale = 1.0 / np.sqrt(diagonal)
    scaling = diags(inverse_scale)
    scaled = (scaling @ K @ scaling).tocsc()
    factor = splu(scaled)
    pivots = np.abs(np.asarray(factor.U.diagonal(), dtype=float))
    tolerance = (
        np.finfo(float).eps
        * max(K.shape[0], 1)
        * max(float(np.max(pivots)), 1.0)
    )
    if (
        not np.all(np.isfinite(pivots))
        or float(np.min(pivots)) <= tolerance
    ):
        raise ValueError("stiffness matrix is numerically rank deficient")


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
