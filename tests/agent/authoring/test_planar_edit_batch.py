from __future__ import annotations

from copy import deepcopy

import pytest

from fem.geometry import (
    SketchArc,
    SketchGeometry,
    SketchLine,
    SketchPlane,
    SketchPoint,
    SketchVerticalConstraint,
)
from fem_agent import geometry_authoring


def _sketch_with_arc_profile() -> SketchGeometry:
    return SketchGeometry(
        "Plate with arc cutout",
        SketchPlane.xy(),
        (
            SketchPoint("P1", 0.0, 0.0),
            SketchPoint("P2", 10.0, 0.0),
            SketchPoint("P3", 10.0, 10.0),
            SketchPoint("P4", 0.0, 10.0),
            SketchPoint("P5", 2.0, 2.0),
            SketchPoint("P6", 3.0, 2.0),
            SketchPoint("P7", 4.0, 2.0),
        ),
        (
            SketchLine("L1", "P1", "P2"),
            SketchLine("L2", "P2", "P3"),
            SketchLine("L3", "P3", "P4"),
            SketchLine("L4", "P4", "P1"),
            SketchArc("A1", "P5", "P6", "P7", "ccw"),
            SketchLine("L5", "P7", "P5"),
        ),
    )


def test_batch_applies_curve_and_constraint_edits_without_changing_inputs() -> None:
    sketch = _sketch_with_arc_profile()
    original_sketch = deepcopy(sketch)
    edits = [
        {
            "operation": "update_line",
            "line_id": "L1",
            "start": {"point_id": "P2"},
            "end": {"point_id": "P1"},
        },
        {"operation": "update_arc", "arc_id": "A1", "orientation": "cw"},
        {
            "operation": "add_constraint",
            "constraint": {"kind": "horizontal", "line_id": "L1", "enabled": False},
        },
        {
            "operation": "replace_constraint",
            "constraint_id": "K1",
            "constraint": {"kind": "vertical", "line_id": "L2", "enabled": False},
        },
    ]
    submitted_edits = deepcopy(edits)

    draft = geometry_authoring.apply_planar_edit_batch(sketch, edits=edits)

    assert draft.proof.exact is True
    assert draft.recipe.curve("L1").start_point_id == "P2"
    assert draft.recipe.curve("L1").end_point_id == "P1"
    assert draft.recipe.curve("A1").orientation == "cw"
    assert len(draft.recipe.constraints) == 1
    constraint = draft.recipe.constraints[0]
    assert isinstance(constraint, SketchVerticalConstraint)
    assert constraint.line_id == "L2"
    assert constraint.enabled is False
    assert sketch == original_sketch
    assert edits == submitted_edits


@pytest.mark.parametrize(
    ("edit", "message"),
    (
        (
            {
                "operation": "add_line",
                "start": {"point_id": "P1"},
                "end": {"point_id": "P2"},
                "unexpected": True,
            },
            "unsupported fields",
        ),
        ({"operation": "batch", "edits": []}, "unsupported operation"),
    ),
    ids=("unsupported-fields", "nested-batch"),
)
def test_batch_rejection_preserves_sketch_and_submitted_edits(
    edit: dict[str, object], message: str
) -> None:
    sketch = _sketch_with_arc_profile()
    original_sketch = deepcopy(sketch)
    edits = [{"operation": "update_point", "point_id": "P3", "x": 11.0}, edit]
    submitted_edits = deepcopy(edits)

    with pytest.raises(ValueError, match=message):
        geometry_authoring.apply_planar_edit_batch(sketch, edits=edits)

    assert sketch == original_sketch
    assert edits == submitted_edits
