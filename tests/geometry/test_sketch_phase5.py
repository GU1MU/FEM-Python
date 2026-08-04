from __future__ import annotations

import math

import pytest

from fem.geometry import (
    SketchAngleDimension,
    SketchArc,
    SketchCircle,
    SketchConcentricConstraint,
    SketchEqualLengthConstraint,
    SketchEqualRadiusConstraint,
    SketchFixedConstraint,
    SketchGeometry,
    SketchHorizontalConstraint,
    SketchLine,
    SketchParallelConstraint,
    SketchPerpendicularConstraint,
    SketchPlane,
    SketchPoint,
    SketchRadiusDimension,
    SketchTangentConstraint,
    analyze_sketch_profiles,
    evaluate_sketch_residuals,
    intersect_sketch_curves,
    solve_sketch_constraints,
    split_curve_at,
)


def _residuals(sketch: SketchGeometry) -> tuple[float, ...]:
    blocks = evaluate_sketch_residuals(
        {point.id: point for point in sketch.points}, sketch.curves, sketch.constraints
    )
    return tuple(value for block in blocks if not block.internal for value in block.values)


def test_each_advanced_constraint_has_a_zero_residual_for_valid_geometry() -> None:
    points = (
        SketchPoint("P1", 0.0, 0.0), SketchPoint("P2", 2.0, 0.0),
        SketchPoint("P3", 0.0, 1.0), SketchPoint("P4", 2.0, 1.0),
        SketchPoint("P5", 0.0, 0.0), SketchPoint("P6", 0.0, 2.0),
        SketchPoint("C1", 0.0, 2.0), SketchPoint("C2", 2.0, 2.0),
        SketchPoint("C3", 0.0, 2.0),
    )
    curves = (
        SketchLine("L1", "P1", "P2"), SketchLine("L2", "P3", "P4"),
        SketchLine("L3", "P5", "P6"),
        SketchCircle("R1", "C1", 1.0), SketchCircle("R2", "C2", 1.0),
        SketchCircle("R3", "C3", 2.0),
    )
    constraints = (
        SketchParallelConstraint("Parallel", "L1", "L2"),
        SketchPerpendicularConstraint("Perpendicular", "L1", "L3"),
        SketchEqualLengthConstraint("Length", "L1", "L2"),
        SketchEqualRadiusConstraint("Radius", "R1", "R2"),
        SketchConcentricConstraint("Center", "R1", "R3"),
        SketchTangentConstraint("Tangent", "L2", "R1", 0),
        SketchAngleDimension("Angle", "L1", "L3", math.pi / 2.0),
    )
    sketch = SketchGeometry("advanced", SketchPlane.xy(), points, curves, constraints)

    assert max(abs(value) for value in _residuals(sketch)) < 1.0e-12


def test_advanced_constraints_report_redundancy_and_conflict_ids() -> None:
    points = (
        SketchPoint("P1", 0.0, 0.0), SketchPoint("P2", 1.0, 0.0),
        SketchPoint("P3", 0.0, 1.0), SketchPoint("P4", 1.0, 1.0),
    )
    curves = (SketchLine("L1", "P1", "P2"), SketchLine("L2", "P3", "P4"))
    duplicate = SketchGeometry(
        "redundant", SketchPlane.xy(), points, curves,
        (SketchParallelConstraint("A", "L1", "L2"), SketchParallelConstraint("B", "L1", "L2")),
    )
    redundant = solve_sketch_constraints(duplicate)
    assert redundant.status == "redundant"
    assert "B" in redundant.redundant_constraint_ids

    conflict_points = (*points[:3], SketchPoint("P4", 0.0, 2.0))
    fixed = tuple(
        SketchFixedConstraint(f"F{index}", point.id, point.u, point.v)
        for index, point in enumerate(conflict_points)
    )
    conflict = SketchGeometry(
        "conflict", SketchPlane.xy(), conflict_points, curves,
        (*fixed, SketchParallelConstraint("Conflict", "L1", "L2")),
    )
    result = solve_sketch_constraints(conflict, new_constraint_ids=("Conflict",))
    assert result.status == "conflicting"
    assert result.conflicting_constraint_ids[0] == "Conflict"


@pytest.mark.parametrize("kind", ["line_arc", "arc_arc"])
def test_tangent_support_solution_outside_finite_curve_domain_conflicts(kind: str) -> None:
    if kind == "line_arc":
        points = (
            SketchPoint("LS", 2.0, 1.0), SketchPoint("LE", 3.0, 1.0),
            SketchPoint("AS", -1.0, 0.0), SketchPoint("AC", 0.0, 0.0),
            SketchPoint("AE", 0.0, 1.0),
        )
        curves = (
            SketchLine("L", "LS", "LE"),
            SketchArc("A", "AS", "AC", "AE", "cw"),
        )
        tangent = SketchTangentConstraint("T", "L", "A", 1)
    else:
        points = (
            SketchPoint("A1S", 0.0, 1.0), SketchPoint("A1C", 0.0, 0.0),
            SketchPoint("A1E", -1.0, 0.0), SketchPoint("A2S", 2.0, 1.0),
            SketchPoint("A2C", 2.0, 0.0), SketchPoint("A2E", 3.0, 0.0),
        )
        curves = (
            SketchArc("A1", "A1S", "A1C", "A1E", "ccw"),
            SketchArc("A2", "A2S", "A2C", "A2E", "cw"),
        )
        tangent = SketchTangentConstraint("T", "A1", "A2", 0)
    fixed = tuple(
        SketchFixedConstraint(f"F{index}", point.id, point.u, point.v)
        for index, point in enumerate(points)
    )
    sketch = SketchGeometry("finite tangent", SketchPlane.xy(), points, curves, (*fixed, tangent))

    result = solve_sketch_constraints(sketch, new_constraint_ids=("T",))
    assert result.status == "conflicting"
    assert result.conflicting_constraint_ids[0] == "T"


def test_line_arc_circle_split_is_stable_directional_and_rejects_degenerate() -> None:
    line_sketch = SketchGeometry(
        "line", SketchPlane.xy(),
        (SketchPoint("P1", 0.0, 0.0), SketchPoint("P2", 2.0, 0.0)),
        (SketchLine("L", "P1", "P2"),),
        (SketchHorizontalConstraint("H", "L"),),
    )
    first = split_curve_at(line_sketch, "L", 0.5)
    replay = split_curve_at(line_sketch, "L", 0.5)
    assert first.derived_curve_ids == replay.derived_curve_ids
    assert first.split_point_ids == replay.split_point_ids
    assert all(isinstance(item, SketchLine) for item in first.curves)
    assert len(first.constraints) == 2
    with pytest.raises(ValueError, match="strictly inside"):
        split_curve_at(line_sketch, "L", 0.0)

    arc_sketch = SketchGeometry(
        "arc", SketchPlane.xy(),
        (SketchPoint("S", 1.0, 0.0), SketchPoint("C", 0.0, 0.0), SketchPoint("E", 0.0, -1.0)),
        (SketchArc("A", "S", "C", "E", "cw"),),
    )
    arc_result = split_curve_at(arc_sketch, "A", 0.5)
    assert [item.orientation for item in arc_result.curves] == ["cw", "cw"]

    circle_sketch = SketchGeometry(
        "circle", SketchPlane.xy(), (SketchPoint("C", 0.0, 0.0),),
        (SketchCircle("O", "C", 1.0),),
        (SketchRadiusDimension("R", "O", 1.0),),
    )
    circle_result = split_curve_at(circle_sketch, "O", 0.0)
    assert len(circle_result.curves) == 2
    assert all(isinstance(item, SketchArc) and item.orientation == "ccw" for item in circle_result.curves)
    assert len(set(circle_result.split_point_ids)) == 2
    assert len(circle_result.constraints) == 2


def test_split_order_reuses_coordinate_ids_and_reports_ambiguous_constraint_removal() -> None:
    sketch = SketchGeometry(
        "order",
        SketchPlane.xy(),
        (
            SketchPoint("C", 0.0, 0.0),
            SketchPoint("LS", -2.0, 0.0), SketchPoint("LE", 2.0, 0.0),
            SketchPoint("BS", -2.0, 2.0), SketchPoint("BE", 2.0, 2.0),
        ),
        (
            SketchCircle("O", "C", 1.0),
            SketchLine("L", "LS", "LE"),
            SketchLine("B", "BS", "BE"),
        ),
        (SketchEqualLengthConstraint("Ambiguous", "L", "B"),),
    )
    intersections = intersect_sketch_curves(
        sketch.curve("O"), sketch.curve("L"),
        {point.id: point for point in sketch.points},
    ).intersections
    circle_parameters = tuple(item.left_parameter for item in intersections)
    line_parameters = tuple(item.right_parameter for item in intersections)

    circle_first = split_curve_at(sketch, "O", circle_parameters)
    circle_then_line = split_curve_at(circle_first.sketch, "L", line_parameters)
    line_first = split_curve_at(sketch, "L", line_parameters)
    line_then_circle = split_curve_at(line_first.sketch, "O", circle_parameters)

    assert {item.id for item in circle_then_line.points} == {
        item.id for item in line_then_circle.points
    }
    assert {item.id for item in circle_then_line.curves} == {
        item.id for item in line_then_circle.curves
    }
    assert line_first.removed_constraint_ids == ("Ambiguous",)
    assert line_first.diagnostics


def test_mixed_line_arc_closed_profile_uses_exact_curve_graph() -> None:
    sketch = SketchGeometry(
        "semicircle", SketchPlane.xy(),
        (SketchPoint("S", -1.0, 0.0), SketchPoint("C", 0.0, 0.0), SketchPoint("E", 1.0, 0.0)),
        (SketchArc("A", "S", "C", "E", "cw"), SketchLine("L", "E", "S")),
    )

    analysis = analyze_sketch_profiles(sketch)
    assert len(analysis.profiles) == 1
    assert analysis.profiles[0].area == pytest.approx(math.pi / 2.0)
    assert not any(item.code in {"sketch.open-loop", "sketch.crossing"} for item in analysis.diagnostics)
