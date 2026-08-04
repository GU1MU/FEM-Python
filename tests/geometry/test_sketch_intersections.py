from __future__ import annotations

import math
import subprocess
import sys

import pytest

from fem.geometry.recipes import SketchArc, SketchCircle, SketchLine, SketchPoint
from fem.geometry.sketch_intersections import intersect_sketch_curves


def _points(**coordinates: tuple[float, float]) -> dict[str, SketchPoint]:
    return {
        point_id: SketchPoint(point_id, coordinate[0], coordinate[1])
        for point_id, coordinate in coordinates.items()
    }


def test_line_line_crossing_overlap_and_degenerate_are_analytic() -> None:
    points = _points(a=(-1.0, 0.0), b=(1.0, 0.0), c=(0.0, -1.0), d=(0.0, 1.0))
    result = intersect_sketch_curves(
        SketchLine("L1", "a", "b"), SketchLine("L2", "c", "d"), points
    )
    values = [
        (item.u, item.v, item.left_parameter, item.right_parameter)
        for item in result.intersections
    ]
    assert values == [
        pytest.approx((0.0, 0.0, 0.5, 0.5))
    ]
    assert result.intersections[0].kind == "crossing"

    overlap_points = _points(a=(0.0, 0.0), b=(2.0, 0.0), c=(1.0, 0.0), d=(3.0, 0.0))
    overlap = intersect_sketch_curves(
        SketchLine("L1", "a", "b"),
        SketchLine("L2", "c", "d"),
        overlap_points,
    )
    assert overlap.intersections == ()
    assert [item.kind for item in overlap.diagnostics] == ["overlap"]

    separate_points = _points(a=(0.0, 0.0), b=(1.0, 0.0), c=(0.0, 1.0), d=(1.0, 1.0))
    assert intersect_sketch_curves(
        SketchLine("L1", "a", "b"),
        SketchLine("L2", "c", "d"),
        separate_points,
    ).intersections == ()
    touching_points = _points(a=(0.0, 0.0), b=(1.0, 0.0), c=(1.0, 0.0), d=(2.0, 0.0))
    touching = intersect_sketch_curves(
        SketchLine("L1", "a", "b"),
        SketchLine("L2", "c", "d"),
        touching_points,
    )
    assert len(touching.intersections) == 1
    assert touching.intersections[0].kind == "tangent"

    degenerate_points = _points(a=(0.0, 0.0), b=(0.0, 0.0), c=(0.0, -1.0), d=(0.0, 1.0))
    degenerate = intersect_sketch_curves(
        SketchLine("L1", "a", "b"),
        SketchLine("L2", "c", "d"),
        degenerate_points,
    )
    assert degenerate.intersections == ()
    assert degenerate.diagnostics[0].kind == "degenerate"


def test_line_circle_has_stable_double_intersections_and_one_tangent() -> None:
    points = _points(
        a=(-2.0, 0.0),
        b=(2.0, 0.0),
        t0=(-2.0, 1.0),
        t1=(2.0, 1.0),
        n0=(-2.0, 2.0),
        n1=(2.0, 2.0),
        o=(0.0, 0.0),
    )
    circle = SketchCircle("C", "o", 1.0)
    crossing = intersect_sketch_curves(SketchLine("L", "a", "b"), circle, points)
    assert [(item.u, item.left_parameter, item.branch_hint) for item in crossing.intersections] == [
        pytest.approx((-1.0, 0.25, 0)),
        pytest.approx((1.0, 0.75, 1)),
    ]
    assert {item.kind for item in crossing.intersections} == {"crossing"}

    tangent = intersect_sketch_curves(SketchLine("T", "t0", "t1"), circle, points)
    assert len(tangent.intersections) == 1
    assert tangent.intersections[0].kind == "tangent"
    assert (tangent.intersections[0].u, tangent.intersections[0].v) == pytest.approx((0.0, 1.0))
    assert intersect_sketch_curves(
        SketchLine("N", "n0", "n1"), circle, points
    ).intersections == ()


def test_line_arc_filters_against_sweep_and_preserves_tangent_deduplication() -> None:
    points = _points(
        a=(-2.0, 0.5),
        b=(2.0, 0.5),
        n0=(-2.0, -0.5),
        n1=(2.0, -0.5),
        t0=(-2.0, 1.0),
        t1=(2.0, 1.0),
        start=(1.0, 0.0),
        center=(0.0, 0.0),
        end=(-1.0, 0.0),
    )
    arc = SketchArc("A", "start", "center", "end", "ccw")
    crossing = intersect_sketch_curves(SketchLine("L", "a", "b"), arc, points)
    assert len(crossing.intersections) == 2
    assert [item.v for item in crossing.intersections] == pytest.approx([0.5, 0.5])
    assert intersect_sketch_curves(SketchLine("N", "n0", "n1"), arc, points).intersections == ()
    tangent = intersect_sketch_curves(SketchLine("T", "t0", "t1"), arc, points)
    assert len(tangent.intersections) == 1
    assert tangent.intersections[0].kind == "tangent"


def test_circle_circle_crossing_tangent_none_and_coincident() -> None:
    points = _points(o0=(0.0, 0.0), o1=(1.0, 0.0), o2=(2.0, 0.0), far=(4.0, 0.0))
    left = SketchCircle("C0", "o0", 1.0)
    crossing = intersect_sketch_curves(left, SketchCircle("C1", "o1", 1.0), points)
    assert len(crossing.intersections) == 2
    assert {round(item.v, 8) for item in crossing.intersections} == {
        round(-math.sqrt(3.0) / 2.0, 8),
        round(math.sqrt(3.0) / 2.0, 8),
    }
    tangent = intersect_sketch_curves(left, SketchCircle("C2", "o2", 1.0), points)
    assert len(tangent.intersections) == 1
    assert tangent.intersections[0].kind == "tangent"
    assert intersect_sketch_curves(left, SketchCircle("C3", "far", 1.0), points).intersections == ()
    coincident = intersect_sketch_curves(left, SketchCircle("C4", "o0", 1.0), points)
    assert coincident.intersections == ()
    assert coincident.diagnostics[0].kind == "coincident"


def test_circle_arc_and_arc_arc_use_support_circle_then_sweep_filter() -> None:
    points = _points(
        o0=(0.0, 0.0),
        o1=(1.0, 0.0),
        o2=(2.0, 0.0),
        far=(4.0, 0.0),
        a0s=(2.0, 0.0),
        a0e=(0.0, 0.0),
        a1s=(1.0, 0.0),
        a1e=(-1.0, 0.0),
        fars=(5.0, 0.0),
        fare=(3.0, 0.0),
        tangent_start=(3.0, 0.0),
        tangent_end=(1.0, 0.0),
    )
    circle = SketchCircle("C", "o0", 1.0)
    right_upper = SketchArc("A0", "a0s", "o1", "a0e", "ccw")
    circle_arc = intersect_sketch_curves(circle, right_upper, points)
    assert len(circle_arc.intersections) == 1
    assert (circle_arc.intersections[0].u, circle_arc.intersections[0].v) == pytest.approx(
        (0.5, math.sqrt(3.0) / 2.0)
    )
    tangent_arc = SketchArc(
        "AT", "tangent_start", "o2", "tangent_end", "ccw"
    )
    circle_arc_tangent = intersect_sketch_curves(circle, tangent_arc, points)
    assert len(circle_arc_tangent.intersections) == 1
    assert circle_arc_tangent.intersections[0].kind == "tangent"

    left_upper = SketchArc("A1", "a1s", "o0", "a1e", "ccw")
    arc_arc = intersect_sketch_curves(left_upper, right_upper, points)
    assert len(arc_arc.intersections) == 1
    assert arc_arc.intersections[0].kind == "crossing"
    far_upper = SketchArc("AF", "fars", "far", "fare", "ccw")
    assert intersect_sketch_curves(left_upper, far_upper, points).intersections == ()
    arc_tangent = intersect_sketch_curves(left_upper, tangent_arc, points)
    assert len(arc_tangent.intersections) == 1
    assert arc_tangent.intersections[0].kind == "tangent"
    assert intersect_sketch_curves(circle, far_upper, points).intersections == ()

    coincident_circle_arc = intersect_sketch_curves(
        circle,
        SketchArc("AC", "a1s", "o0", "a1e", "ccw"),
        points,
    )
    assert coincident_circle_arc.intersections == ()
    assert coincident_circle_arc.diagnostics[0].kind == "overlap"
    overlapping_arcs = intersect_sketch_curves(
        left_upper,
        SketchArc("AO", "a1s", "o0", "a1e", "ccw"),
        points,
    )
    assert overlapping_arcs.intersections == ()
    assert overlapping_arcs.diagnostics[0].kind == "overlap"


def test_pure_intersection_module_does_not_load_gui_or_meshing_dependencies() -> None:
    check = subprocess.run(
        (
            sys.executable,
            "-c",
            "import sys; import fem.geometry.sketch_intersections; "
            "print(any(name == 'PySide6' or name.startswith('PySide6.') "
            "or name == 'pyvista' or name.startswith('pyvista.') "
            "or name == 'gmsh' or name.startswith('gmsh.') for name in sys.modules))",
        ),
        check=True,
        capture_output=True,
        text=True,
    )
    assert check.stdout.strip() == "False"
