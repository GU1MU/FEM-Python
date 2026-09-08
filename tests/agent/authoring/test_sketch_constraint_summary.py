from __future__ import annotations

from fem.geometry import (
    SketchAngleDimension,
    SketchFixedConstraint,
    SketchGeometry,
    SketchLine,
    SketchParallelConstraint,
    SketchPlane,
    SketchPoint,
)
from fem_agent.geometry_authoring import (
    geometry_recipe_from_payload,
    geometry_recipe_to_payload,
    planar_geometry_catalog,
    update_planar_point,
)


def _sketch() -> SketchGeometry:
    return SketchGeometry(
        "Agent 只读约束",
        SketchPlane.xy(),
        (
            SketchPoint("P1", 0.0, 0.0),
            SketchPoint("P2", 1.0, 0.0),
            SketchPoint("P3", 1.0, 1.0),
            SketchPoint("P4", 0.0, 1.0),
        ),
        (
            SketchLine("L1", "P1", "P2"),
            SketchLine("L2", "P2", "P3"),
            SketchLine("L3", "P3", "P4"),
            SketchLine("L4", "P4", "P1"),
        ),
        (SketchFixedConstraint("G1", "P1", 0.0, 0.0),),
    )


def test_point_edit_preserves_constraints_and_payload() -> None:
    sketch = _sketch()
    edited = update_planar_point(sketch, point_id="P2", x=2.0).recipe
    assert edited.point("P2").u == 2.0
    assert edited.constraints == sketch.constraints
    assert geometry_recipe_from_payload(geometry_recipe_to_payload(edited)) == edited


def test_agent_summary_names_advanced_constraints_and_angle_dimension() -> None:
    sketch = SketchGeometry(
        "Agent 高级约束摘要",
        SketchPlane.xy(),
        (
            SketchPoint("P1", 0.0, 0.0), SketchPoint("P2", 1.0, 0.0),
            SketchPoint("P3", 0.0, 1.0), SketchPoint("P4", 1.0, 1.0),
        ),
        (SketchLine("L1", "P1", "P2"), SketchLine("L2", "P3", "P4")),
        (
            SketchParallelConstraint("A1", "L1", "L2"),
            SketchAngleDimension("A2", "L1", "L2", 0.0, False),
        ),
    )

    summary = planar_geometry_catalog(sketch)["constraint_summary"]
    assert summary["types"] == [
        "SketchAngleDimension", "SketchParallelConstraint"
    ]
    assert summary["driving_dimension_count"] == 0
