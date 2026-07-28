from __future__ import annotations

import math

import pytest

from fem.geometry import (
    SketchArc,
    SketchCircle,
    SketchGeometry,
    SketchLine,
    SketchPlane,
    SketchPoint,
    analyze_sketch_profiles,
    legacy_sketch_to_strict,
)
from fem.geometry.recipe_topology import describe_recipe_topology
from fem.geometry.recipes import SketchRectangle


def _rectangle(
    name: str = "Plate",
    *,
    x: float = 0.0,
    y: float = 0.0,
    width: float = 4.0,
    height: float = 3.0,
) -> SketchGeometry:
    points = (
        SketchPoint("P1", x, y),
        SketchPoint("P2", x + width, y),
        SketchPoint("P3", x + width, y + height),
        SketchPoint("P4", x, y + height),
    )
    curves = (
        SketchLine("L1", "P1", "P2"),
        SketchLine("L2", "P2", "P3"),
        SketchLine("L3", "P3", "P4"),
        SketchLine("L4", "P4", "P1"),
    )
    return SketchGeometry(name, SketchPlane.xy(), points, curves)


def test_strict_rectangle_has_stable_profile_and_exact_topology() -> None:
    sketch = _rectangle()
    result = analyze_sketch_profiles(sketch)

    assert result.valid
    assert len(result.profiles) == 1
    assert result.profiles[0].curve_ids == ("L1", "L2", "L3", "L4")
    assert result.profiles[0].signed_area == pytest.approx(12.0)
    assert result.profiles[0].id == analyze_sketch_profiles(sketch).profiles[0].id

    topology = describe_recipe_topology(sketch)
    assert topology.exact
    assert topology.entity("point:P1").selectable
    assert topology.entity("edge:L1").selectable
    assert topology.entity(f"face:{result.profiles[0].id}").selectable
    assert topology.entity("body:domain").selectable


def test_hole_and_disjoint_profiles_are_deterministic() -> None:
    outer = _rectangle(width=10.0, height=8.0)
    points = outer.points + (
        SketchPoint("P5", 2.0, 2.0),
        SketchPoint("P6", 4.0, 2.0),
        SketchPoint("P7", 4.0, 4.0),
        SketchPoint("P8", 2.0, 4.0),
    )
    curves = outer.curves + (
        SketchLine("L5", "P5", "P6"),
        SketchLine("L6", "P6", "P7"),
        SketchLine("L7", "P7", "P8"),
        SketchLine("L8", "P8", "P5"),
    )
    result = analyze_sketch_profiles(
        SketchGeometry("Plate", SketchPlane.xy(), points, curves)
    )

    assert result.valid
    assert [profile.role for profile in result.profiles] == ["outer", "hole"]
    assert result.profiles[1].parent_profile_id == result.profiles[0].id


def test_circle_and_arc_profiles_use_geometric_area() -> None:
    circle = SketchGeometry(
        "Circle",
        SketchPlane.xy(),
        (SketchPoint("P1", 1.0, 2.0),),
        (SketchCircle("C1", "P1", 3.0),),
    )
    circle_profile = analyze_sketch_profiles(circle).profiles[0]
    assert circle_profile.signed_area == pytest.approx(9.0 * math.pi)

    arc_points = (
        SketchPoint("O", 0.0, 0.0),
        SketchPoint("P1", 1.0, 0.0),
        SketchPoint("P2", 0.0, 1.0),
        SketchPoint("P3", -1.0, 0.0),
        SketchPoint("P4", 0.0, -1.0),
    )
    arc_curves = (
        SketchArc("A1", "P1", "O", "P2"),
        SketchArc("A2", "P2", "O", "P3"),
        SketchArc("A3", "P3", "O", "P4"),
        SketchArc("A4", "P4", "O", "P1"),
    )
    arc_profile = analyze_sketch_profiles(
        SketchGeometry("Arc", SketchPlane.xy(), arc_points, arc_curves)
    ).profiles[0]
    assert arc_profile.signed_area == pytest.approx(math.pi)


def test_invalid_open_and_crossing_curves_are_blocking() -> None:
    open_sketch = SketchGeometry(
        "Open",
        SketchPlane.xy(),
        (SketchPoint("P1", 0.0, 0.0), SketchPoint("P2", 1.0, 0.0)),
        (SketchLine("L1", "P1", "P2"),),
    )
    result = analyze_sketch_profiles(open_sketch)
    assert not result.valid
    assert "sketch.open-loop" in {item.code for item in result.diagnostics}

    crossing = SketchGeometry(
        "Crossing",
        SketchPlane.xy(),
        (
            SketchPoint("P1", 0.0, 0.0),
            SketchPoint("P2", 2.0, 2.0),
            SketchPoint("P3", 0.0, 2.0),
            SketchPoint("P4", 2.0, 0.0),
        ),
        (SketchLine("L1", "P1", "P2"), SketchLine("L2", "P3", "P4")),
    )
    assert "sketch.crossing" in {
        item.code for item in analyze_sketch_profiles(crossing).diagnostics
    }


def test_legacy_contours_have_explicit_curve_migration() -> None:
    legacy = SketchGeometry(
        "Legacy",
        (
            SketchRectangle("material", 0.0, 0.0, 4.0, 3.0),
            SketchCircle("cut", 2.0, 1.5, 0.5),
        ),
    )
    strict = legacy_sketch_to_strict(legacy)

    assert strict.is_strict
    result = analyze_sketch_profiles(strict)
    assert result.valid
    assert [profile.role for profile in result.profiles] == ["outer", "hole"]
