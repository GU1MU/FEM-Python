"""Numerical MVP solver for strict planar sketch constraints."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Literal

import numpy as np
from scipy.optimize import least_squares

from .recipes import (
    SketchCircle,
    SketchConstraint,
    SketchCurve,
    SketchFixedConstraint,
    SketchGeometry,
    SketchPlane,
    SketchPoint,
    validate_sketch_constraints,
)
from .sketch_constraints import (
    SketchResidualBlock,
    duplicate_constraint_ids,
    evaluate_sketch_residuals,
    flatten_residual_blocks,
    sketch_characteristic_length,
)


SketchSolveStatus = Literal[
    "under_constrained",
    "fully_constrained",
    "redundant",
    "conflicting",
    "failed",
]


@dataclass(frozen=True, slots=True)
class SketchSolveResult:
    """Detached solver output and diagnostics; it never mutates its input."""

    points: tuple[SketchPoint, ...]
    curves: tuple[SketchCurve, ...]
    status: SketchSolveStatus
    remaining_dof: int
    max_residual: float
    redundant_constraint_ids: tuple[str, ...] = ()
    conflicting_constraint_ids: tuple[str, ...] = ()

    @property
    def succeeded(self) -> bool:
        return self.status not in {"conflicting", "failed"}

    @property
    def redundant_ids(self) -> tuple[str, ...]:
        return self.redundant_constraint_ids

    @property
    def conflict_candidate_ids(self) -> tuple[str, ...]:
        return self.conflicting_constraint_ids


def solve_sketch_constraints(
    sketch: SketchGeometry,
    *,
    fixed_point_ids: Iterable[str] = (),
    previous_solution: SketchSolveResult | SketchGeometry | None = None,
    new_constraint_ids: Iterable[str] = (),
    residual_tolerance: float = 1.0e-7,
    rank_tolerance: float | None = None,
    max_nfev: int = 500,
) -> SketchSolveResult:
    """Solve one validated strict sketch without changing it."""

    if type(sketch) is not SketchGeometry or not sketch.is_strict:
        raise TypeError("sketch must be a strict SketchGeometry")
    return solve_sketch_draft(
        sketch.points,
        sketch.curves,
        sketch.constraints,
        fixed_point_ids=fixed_point_ids,
        previous_solution=previous_solution,
        new_constraint_ids=new_constraint_ids,
        residual_tolerance=residual_tolerance,
        rank_tolerance=rank_tolerance,
        max_nfev=max_nfev,
    )


def solve_sketch_draft(
    points: tuple[SketchPoint, ...],
    curves: tuple[SketchCurve, ...],
    constraints: tuple[SketchConstraint, ...],
    *,
    fixed_point_ids: Iterable[str] = (),
    previous_solution: SketchSolveResult | SketchGeometry | None = None,
    new_constraint_ids: Iterable[str] = (),
    residual_tolerance: float = 1.0e-7,
    rank_tolerance: float | None = None,
    max_nfev: int = 500,
) -> SketchSolveResult:
    """Solve detached draft arrays, including temporarily invalid arc radii."""

    point_map = {point.id: point for point in points}
    if len(point_map) != len(points):
        raise ValueError("duplicate sketch point id")
    validate_sketch_constraints(constraints, point_map, curves)
    if not math.isfinite(residual_tolerance) or residual_tolerance <= 0.0:
        raise ValueError("residual_tolerance must be finite and positive")
    if isinstance(max_nfev, bool) or not isinstance(max_nfev, int) or max_nfev < 1:
        raise ValueError("max_nfev must be a positive integer")

    scale = sketch_characteristic_length(point_map, curves, constraints)
    externally_fixed = frozenset(fixed_point_ids)
    unknown_fixed = externally_fixed.difference(point_map)
    if unknown_fixed:
        raise ValueError(f"fixed point does not exist: {sorted(unknown_fixed)[0]}")
    fixed_targets: dict[str, SketchPoint] = {
        point_id: point_map[point_id] for point_id in externally_fixed
    }
    for constraint in constraints:
        if not isinstance(constraint, SketchFixedConstraint) or not constraint.enabled:
            continue
        fixed_targets.setdefault(
            constraint.point_id,
            SketchPoint(constraint.point_id, constraint.u, constraint.v),
        )

    point_variable_ids = tuple(point.id for point in points if point.id not in fixed_targets)
    circle_variable_ids = tuple(
        curve.id
        for curve in curves
        if isinstance(curve, SketchCircle) and curve.is_curve
    )
    previous_points, previous_radii = _previous_values(previous_solution)
    initial: list[float] = []
    for point_id in point_variable_ids:
        point = previous_points.get(point_id, point_map[point_id])
        initial.extend((point.u / scale, point.v / scale))
    curve_map = {curve.id: curve for curve in curves}
    for curve_id in circle_variable_ids:
        circle = curve_map[curve_id]
        assert isinstance(circle, SketchCircle)
        initial.append(previous_radii.get(curve_id, circle.radius) / scale)
    x0 = np.asarray(initial, dtype=float)
    lower = np.full(x0.shape, -np.inf)
    upper = np.full(x0.shape, np.inf)
    if circle_variable_ids:
        lower[-len(circle_variable_ids):] = 1.0e-12

    def unpack(values: np.ndarray) -> tuple[dict[str, SketchPoint], tuple[SketchCurve, ...]]:
        solved_points = dict(point_map)
        solved_points.update(fixed_targets)
        offset = 0
        for point_id in point_variable_ids:
            solved_points[point_id] = SketchPoint(
                point_id, values[offset] * scale, values[offset + 1] * scale
            )
            offset += 2
        solved_curves: list[SketchCurve] = []
        radii = {
            curve_id: values[offset + index] * scale
            for index, curve_id in enumerate(circle_variable_ids)
        }
        for curve in curves:
            if isinstance(curve, SketchCircle) and curve.is_curve:
                solved_curves.append(
                    SketchCircle(curve.id, curve.center_point_id, radii[curve.id])
                )
            else:
                solved_curves.append(curve)
        return solved_points, tuple(solved_curves)

    def residual(values: np.ndarray) -> np.ndarray:
        candidate_points, candidate_curves = unpack(values)
        blocks = evaluate_sketch_residuals(
            candidate_points,
            candidate_curves,
            constraints,
            characteristic_length=scale,
        )
        return np.asarray(flatten_residual_blocks(blocks), dtype=float)

    try:
        if x0.size == 0:
            optimum = x0
            jacobian = np.empty((residual(x0).size, 0), dtype=float)
            optimizer_succeeded = True
        elif residual(x0).size == 0:
            optimum = x0
            jacobian = np.empty((0, x0.size), dtype=float)
            optimizer_succeeded = True
        else:
            optimized = least_squares(
                residual,
                x0,
                bounds=(lower, upper),
                xtol=1.0e-12,
                ftol=1.0e-12,
                gtol=1.0e-12,
                max_nfev=max_nfev,
            )
            optimum = optimized.x
            jacobian = np.asarray(optimized.jac, dtype=float)
            optimizer_succeeded = bool(optimized.success)
        solved_point_map, solved_curves = unpack(optimum)
        blocks = evaluate_sketch_residuals(
            solved_point_map,
            solved_curves,
            constraints,
            characteristic_length=scale,
        )
        residual_values = np.asarray(flatten_residual_blocks(blocks), dtype=float)
    except Exception:
        return _failed_result(points, curves)

    solved_points = tuple(solved_point_map[point.id] for point in points)
    if (
        not optimizer_succeeded
        or not np.all(np.isfinite(optimum))
        or not np.all(np.isfinite(residual_values))
    ):
        return _failed_result(points, curves)
    try:
        SketchGeometry(
            "solver validation",
            SketchPlane.xy(),
            solved_points,
            solved_curves,
            constraints,
        )
    except (TypeError, ValueError):
        return _failed_result(points, curves)
    maximum = float(np.max(np.abs(residual_values))) if residual_values.size else 0.0
    rank = _matrix_rank(jacobian, rank_tolerance)
    remaining_dof = max(0, int(x0.size - rank))
    redundant = _redundant_constraint_ids(
        blocks,
        jacobian,
        constraints,
        rank_tolerance,
    )
    if maximum > residual_tolerance:
        conflicts = _conflict_candidates(
            blocks,
            new_constraint_ids,
            residual_tolerance,
        )
        status: SketchSolveStatus = "conflicting"
    elif redundant:
        conflicts = ()
        status = "redundant"
    elif remaining_dof:
        conflicts = ()
        status = "under_constrained"
    else:
        conflicts = ()
        status = "fully_constrained"
    return SketchSolveResult(
        solved_points,
        solved_curves,
        status,
        remaining_dof,
        maximum,
        redundant,
        conflicts,
    )


def _previous_values(
    previous: SketchSolveResult | SketchGeometry | None,
) -> tuple[dict[str, SketchPoint], dict[str, float]]:
    if previous is None:
        return {}, {}
    if isinstance(previous, SketchSolveResult):
        points, curves = previous.points, previous.curves
    elif type(previous) is SketchGeometry and previous.is_strict:
        points, curves = previous.points, previous.curves
    else:
        raise TypeError("previous_solution must be a solve result or strict sketch")
    return (
        {point.id: point for point in points},
        {
            curve.id: curve.radius
            for curve in curves
            if isinstance(curve, SketchCircle) and curve.is_curve
        },
    )


def _matrix_rank(matrix: np.ndarray, tolerance: float | None) -> int:
    if not matrix.size:
        return 0
    if tolerance is not None:
        if not math.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("rank_tolerance must be finite and positive")
        return int(np.linalg.matrix_rank(matrix, tol=tolerance))
    return int(np.linalg.matrix_rank(matrix))


def _redundant_constraint_ids(
    blocks: tuple[SketchResidualBlock, ...],
    jacobian: np.ndarray,
    constraints: tuple[SketchConstraint, ...],
    rank_tolerance: float | None,
) -> tuple[str, ...]:
    duplicates = list(duplicate_constraint_ids(constraints))
    duplicate_set = set(duplicates)
    row = 0
    accepted = np.empty((0, jacobian.shape[1]), dtype=float)
    for block in blocks:
        count = len(block.values)
        rows = jacobian[row:row + count]
        row += count
        if block.internal:
            accepted = np.vstack((accepted, rows))
            continue
        if block.owner_id in duplicate_set:
            continue
        constraint = next(item for item in constraints if item.id == block.owner_id)
        if isinstance(constraint, SketchFixedConstraint):
            continue
        standalone_rank = _matrix_rank(rows, rank_tolerance)
        if standalone_rank == 0:
            if max((abs(value) for value in block.values), default=0.0) <= 1.0e-7:
                duplicates.append(block.owner_id)
            continue
        before_rank = _matrix_rank(accepted, rank_tolerance)
        combined = np.vstack((accepted, rows))
        after_rank = _matrix_rank(combined, rank_tolerance)
        if standalone_rank and after_rank - before_rank < standalone_rank:
            duplicates.append(block.owner_id)
        else:
            accepted = combined
    return tuple(dict.fromkeys(duplicates))


def _conflict_candidates(
    blocks: tuple[SketchResidualBlock, ...],
    new_constraint_ids: Iterable[str],
    tolerance: float,
) -> tuple[str, ...]:
    magnitudes = {
        block.owner_id: max((abs(value) for value in block.values), default=0.0)
        for block in blocks
        if not block.internal
    }
    ordered: list[str] = []
    for constraint_id in new_constraint_ids:
        if constraint_id in magnitudes and constraint_id not in ordered:
            ordered.append(constraint_id)
    ordered.extend(
        constraint_id
        for constraint_id, magnitude in sorted(
            magnitudes.items(), key=lambda item: (-item[1], item[0])
        )
        if magnitude > tolerance and constraint_id not in ordered
    )
    return tuple(ordered)


def _failed_result(
    points: tuple[SketchPoint, ...], curves: tuple[SketchCurve, ...]
) -> SketchSolveResult:
    return SketchSolveResult(points, curves, "failed", 0, math.inf)


__all__ = [
    "SketchSolveResult",
    "SketchSolveStatus",
    "solve_sketch_constraints",
    "solve_sketch_draft",
]
