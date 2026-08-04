from __future__ import annotations

import pytest

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
        (SketchPoint("P1", 0.0, 0.0), SketchPoint("P2", 1.0, 0.0)),
        (SketchLine("L1", "P1", "P2"),),
        (SketchFixedConstraint("G1", "P1", 0.0, 0.0),),
    )


def test_agent_catalog_recognizes_constraint_summary_and_old_edit_preserves_it() -> None:
    sketch = _sketch()
    catalog = planar_geometry_catalog(sketch)
    edited = update_planar_point(sketch, point_id="P2", x=2.0).recipe

    assert catalog["constraint_summary"]["count"] == 1
    assert catalog["constraint_summary"]["capability"] == {
        "read": True,
        "create": False,
        "edit": False,
    }
    assert edited.constraints == sketch.constraints


def test_agent_full_recipe_payload_exposes_only_summary_and_rejects_reauthoring() -> None:
    payload = geometry_recipe_to_payload(_sketch())

    assert payload["constraint_summary"]["count"] == 1
    with pytest.raises(ValueError, match="只读识别草图约束摘要"):
        geometry_recipe_from_payload(payload)


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
