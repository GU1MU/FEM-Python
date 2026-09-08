from __future__ import annotations

import math

import pytest

from fem.geometry import (
    SketchArc,
    SketchCircle,
    SketchCoincidentConstraint,
    SketchDistanceDimension,
    SketchFixedConstraint,
    SketchGeometry,
    SketchHorizontalConstraint,
    SketchLine,
    SketchPlane,
    SketchPoint,
    SketchPointOnCurveConstraint,
    SketchRadiusDimension,
    SketchVerticalConstraint,
    evaluate_sketch_residuals,
    solve_sketch_constraints,
    solve_sketch_draft,
)


def _line_sketch(*constraints: object) -> SketchGeometry:
    return SketchGeometry(
        "solver",
        SketchPlane.xy(),
        (SketchPoint("P1", 0.0, 0.0), SketchPoint("P2", 2.0, 1.0)),
        (SketchLine("L1", "P1", "P2"),),
        constraints,
    )


@pytest.mark.parametrize(
    "constraint, expected_size",
    [
        (SketchCoincidentConstraint("G", "P1", "P2"), 2),
        (SketchPointOnCurveConstraint("G", "P2", "L1"), 1),
        (SketchHorizontalConstraint("G", "L1"), 1),
        (SketchVerticalConstraint("G", "L1"), 1),
        (SketchFixedConstraint("G", "P1", 0.0, 0.0), 2),
        (SketchDistanceDimension("G", "P1", "P2", math.sqrt(5.0)), 1),
    ],
)
def test_each_point_and_line_constraint_maps_to_stable_residual_block(
    constraint: object, expected_size: int
) -> None:
    sketch = _line_sketch(constraint)

    blocks = evaluate_sketch_residuals(
        {point.id: point for point in sketch.points},
        sketch.curves,
        sketch.constraints,
    )

    assert blocks[-1].owner_id == "G"
    assert len(blocks[-1].values) == expected_size


def test_circle_radius_is_a_variable_and_arc_has_internal_equal_radius_rule() -> None:
    circle_points = (SketchPoint("C", 0.0, 0.0),)
    circle = SketchCircle("Circle", "C", 2.0)
    result = solve_sketch_draft(
        circle_points,
        (circle,),
        (SketchRadiusDimension("R", "Circle", 4.0),),
        fixed_point_ids=("C",),
    )
    assert result.status == "fully_constrained"
    assert isinstance(result.curves[0], SketchCircle)
    assert result.curves[0].radius == pytest.approx(4.0)

    points = (
        SketchPoint("S", 2.0, 0.0),
        SketchPoint("C", 0.0, 0.0),
        SketchPoint("E", 0.0, 1.0),
    )
    arc = SketchArc("A", "S", "C", "E")
    solved = solve_sketch_draft(points, (arc,), (), fixed_point_ids=("S", "C"))
    solved_map = {point.id: point for point in solved.points}
    assert math.hypot(solved_map["E"].u, solved_map["E"].v) == pytest.approx(2.0)
    assert solved.max_residual < 1.0e-7


def test_point_on_arc_rejects_the_opposite_supporting_circle_branch() -> None:
    points = {
        point.id: point
        for point in (
            SketchPoint("S", 1.0, 0.0),
            SketchPoint("C", 0.0, 0.0),
            SketchPoint("E", 0.0, 1.0),
            SketchPoint("P", -1.0, 0.0),
        )
    }
    arc = SketchArc("A", "S", "C", "E", "ccw")
    blocks = evaluate_sketch_residuals(
        points,
        (arc,),
        (SketchPointOnCurveConstraint("On", "P", "A"),),
    )
    assert abs(blocks[-1].values[0]) > 0.5


def test_point_on_line_rejects_a_projection_outside_the_finite_segment() -> None:
    points = {
        point.id: point
        for point in (
            SketchPoint("S", 0.0, 0.0),
            SketchPoint("E", 1.0, 0.0),
            SketchPoint("P", 2.0, 0.0),
        )
    }
    blocks = evaluate_sketch_residuals(
        points,
        (SketchLine("L", "S", "E"),),
        (SketchPointOnCurveConstraint("On", "P", "L"),),
    )
    assert blocks[-1].values[0] >= 0.5


def test_combination_reports_fully_constrained_and_is_repeatable() -> None:
    sketch = _line_sketch(
        SketchFixedConstraint("F", "P1", 0.0, 0.0),
        SketchHorizontalConstraint("H", "L1"),
        SketchDistanceDimension("D", "P1", "P2", 3.0),
    )
    first = solve_sketch_constraints(sketch)
    second = solve_sketch_constraints(sketch, previous_solution=first)

    assert first.status == second.status == "fully_constrained"
    assert first.remaining_dof == second.remaining_dof == 0
    assert second.points == first.points


@pytest.mark.parametrize("scale", [1.0e-4, 1.0e6])
def test_characteristic_length_normalization_is_size_independent(scale: float) -> None:
    sketch = SketchGeometry(
        "scaled",
        SketchPlane.xy(),
        (SketchPoint("P1", 0.0, 0.0), SketchPoint("P2", 2.0 * scale, scale)),
        (SketchLine("L1", "P1", "P2"),),
        (
            SketchFixedConstraint("F", "P1", 0.0, 0.0),
            SketchHorizontalConstraint("H", "L1"),
            SketchDistanceDimension("D", "P1", "P2", 3.0 * scale),
        ),
    )

    result = solve_sketch_constraints(sketch)

    assert result.status == "fully_constrained"
    assert result.max_residual < 1.0e-7


def test_under_constrained_redundant_and_conflicting_diagnostics_use_constraint_ids() -> None:
    under = solve_sketch_constraints(
        _line_sketch(SketchHorizontalConstraint("H", "L1"))
    )
    assert under.status == "under_constrained"
    assert under.remaining_dof == 3

    redundant = solve_sketch_constraints(
        _line_sketch(
            SketchHorizontalConstraint("H1", "L1"),
            SketchHorizontalConstraint("H2", "L1"),
        )
    )
    assert redundant.status == "redundant"
    assert redundant.redundant_constraint_ids == ("H2",)

    conflict = solve_sketch_constraints(
        _line_sketch(
            SketchFixedConstraint("Old", "P1", 0.0, 0.0),
            SketchFixedConstraint("New", "P1", 1.0, 0.0),
        ),
        new_constraint_ids=("New",),
    )
    assert conflict.status == "conflicting"
    assert conflict.conflicting_constraint_ids[0] == "New"
    assert conflict.max_residual > 0.1

    known_geometry_redundancy = solve_sketch_constraints(
        _line_sketch(SketchHorizontalConstraint("KnownH", "L1")),
        fixed_point_ids=("P1", "P2"),
    )
    assert known_geometry_redundancy.status == "conflicting"

    already_horizontal = SketchGeometry(
        "known",
        SketchPlane.xy(),
        (SketchPoint("P1", 0.0, 0.0), SketchPoint("P2", 2.0, 0.0)),
        (SketchLine("L1", "P1", "P2"),),
        (SketchHorizontalConstraint("KnownH", "L1"),),
    )
    known_geometry_redundancy = solve_sketch_constraints(
        already_horizontal,
        fixed_point_ids=("P1", "P2"),
    )
    assert known_geometry_redundancy.status == "redundant"
    assert known_geometry_redundancy.redundant_constraint_ids == ("KnownH",)


def test_two_point_on_curve_constraints_keep_the_previous_intersection_branch() -> None:
    sketch = SketchGeometry(
        "branches",
        SketchPlane.xy(),
        (
            SketchPoint("C1", -1.0, 0.0),
            SketchPoint("C2", 1.0, 0.0),
            SketchPoint("P", 0.0, 1.7),
        ),
        (
            SketchCircle("A", "C1", 2.0),
            SketchCircle("B", "C2", 2.0),
        ),
        (
            SketchPointOnCurveConstraint("OnA", "P", "A"),
            SketchPointOnCurveConstraint("OnB", "P", "B"),
            SketchRadiusDimension("RA", "A", 2.0),
            SketchRadiusDimension("RB", "B", 2.0),
        ),
    )
    upper = solve_sketch_constraints(sketch, fixed_point_ids=("C1", "C2"))
    repeated = solve_sketch_constraints(
        sketch,
        fixed_point_ids=("C1", "C2"),
        previous_solution=upper,
    )

    upper_point = {point.id: point for point in upper.points}["P"]
    repeated_point = {point.id: point for point in repeated.points}["P"]
    assert upper_point.v > 0.0
    assert repeated_point.v > 0.0
    assert repeated_point == upper_point


def test_optimizer_failure_returns_failed_without_partial_values(monkeypatch) -> None:
    def fail(*args: object, **kwargs: object) -> object:
        raise ValueError("synthetic optimizer failure")

    monkeypatch.setattr("fem.geometry.sketch_solver.least_squares", fail)
    sketch = _line_sketch(SketchHorizontalConstraint("H", "L1"))
    result = solve_sketch_constraints(sketch)

    assert result.status == "failed"
    assert result.points == sketch.points
    assert result.curves == sketch.curves
