from __future__ import annotations

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
    constraints_after_curve_split,
    copy_sketch_constraints,
)


def _geometry(*constraints: object) -> SketchGeometry:
    return SketchGeometry(
        "约束草图",
        SketchPlane.xy(),
        (
            SketchPoint("P1", 0.0, 0.0),
            SketchPoint("P2", 2.0, 0.0),
            SketchPoint("P3", 2.0, 2.0),
            SketchPoint("P4", 0.0, 2.0),
            SketchPoint("PC", 1.0, 1.0),
        ),
        (
            SketchLine("L1", "P1", "P2"),
            SketchLine("L2", "P2", "P3"),
            SketchArc("A1", "P3", "PC", "P4"),
            SketchCircle("C1", "PC", 0.5),
        ),
        constraints,
    )


def test_all_phase2_constraint_value_objects_validate_on_a_strict_sketch() -> None:
    constraints = (
        SketchCoincidentConstraint("G1", "P1", "P2", "inferred"),
        SketchPointOnCurveConstraint("G2", "P4", "C1"),
        SketchHorizontalConstraint("G3", "L1", enabled=False),
        SketchVerticalConstraint("G4", "L2"),
        SketchFixedConstraint("G5", "PC", 1.0, 1.0),
        SketchDistanceDimension("D1", "P1", "P2", 2.0, driving=False),
        SketchRadiusDimension("D2", "A1", 2**0.5),
    )

    sketch = _geometry(*constraints)

    assert sketch.constraints == constraints
    assert sketch.constraints[0].source == "inferred"
    assert sketch.constraints[-2].driving is False


@pytest.mark.parametrize(
    "constraints, message",
    [
        (
            (
                SketchFixedConstraint("Same", "P1", 0.0, 0.0),
                SketchFixedConstraint("same", "P2", 2.0, 0.0),
            ),
            "duplicate sketch constraint id",
        ),
        ((SketchFixedConstraint("G1", "missing", 0.0, 0.0),), "unknown point"),
        ((SketchHorizontalConstraint("G1", "C1"),), "wrong curve type"),
        ((SketchRadiusDimension("D1", "L1", 1.0),), "wrong curve type"),
    ],
)
def test_constraint_graph_rejects_duplicate_dangling_and_wrong_type_references(
    constraints: tuple[object, ...], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _geometry(*constraints)


@pytest.mark.parametrize("value", [0.0, -1.0, float("inf"), float("nan")])
def test_dimensions_require_finite_positive_values(value: float) -> None:
    with pytest.raises(ValueError, match="positive|finite"):
        SketchDistanceDimension("D1", "P1", "P2", value)


def test_copy_requires_complete_id_maps_and_split_drops_only_ambiguous_constraints() -> None:
    source = (
        SketchHorizontalConstraint("G1", "L1"),
        SketchFixedConstraint("G2", "P1", 0.0, 0.0),
    )
    copied = copy_sketch_constraints(
        source,
        {"L1": "L9", "P1": "P9"},
        {"G1": "G9", "G2": "G10"},
    )

    assert copied[0] == SketchHorizontalConstraint("G9", "L9")
    assert constraints_after_curve_split(source, "L1") == (source[1],)
    with pytest.raises(ValueError, match="incomplete"):
        copy_sketch_constraints(source, {}, {})
