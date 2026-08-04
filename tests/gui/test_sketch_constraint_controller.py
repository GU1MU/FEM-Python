from __future__ import annotations

import pytest

from fem.geometry import (
    SketchDistanceDimension,
    SketchFixedConstraint,
    SketchGeometry,
    SketchHorizontalConstraint,
    SketchLine,
    SketchPlane,
    SketchPoint,
)
from fem_gui.sketch_editor import SketchDraftController, SketchDraftSnapshot


def _sketch() -> SketchGeometry:
    return SketchGeometry(
        "矩形",
        SketchPlane.xy(),
        (
            SketchPoint("P1", 0.0, 0.0),
            SketchPoint("P2", 2.0, 0.0),
            SketchPoint("P3", 2.0, 1.0),
            SketchPoint("P4", 0.0, 1.0),
        ),
        (
            SketchLine("L1", "P1", "P2"),
            SketchLine("L2", "P2", "P3"),
            SketchLine("L3", "P3", "P4"),
            SketchLine("L4", "P4", "P1"),
        ),
        (
            SketchHorizontalConstraint("G1", "L1"),
            SketchFixedConstraint("G2", "P1", 0.0, 0.0),
            SketchDistanceDimension("D1", "P1", "P2", 2.0),
        ),
    )


def test_delete_cascades_constraints_and_one_undo_restores_everything() -> None:
    controller = SketchDraftController(root=_sketch())

    controller.delete_point("P1")

    assert {item.id for item in controller.constraints} == set()
    assert "L1" not in {item.id for item in controller.snapshot().curves}
    controller.undo()
    assert controller.to_sketch_geometry() == _sketch()
    controller.redo()
    assert controller.constraints == ()


def test_constraint_add_delete_and_restore_share_geometry_history() -> None:
    controller = SketchDraftController(root=_sketch())
    controller.delete_constraint("G1")
    controller.add_constraint(SketchHorizontalConstraint("G3", "L3", "inferred"))

    assert {item.id for item in controller.constraints} == {"G2", "D1", "G3"}
    controller.undo()
    assert {item.id for item in controller.constraints} == {"G2", "D1"}
    controller.undo()
    assert {item.id for item in controller.constraints} == {"G1", "G2", "D1"}


def test_snapshot_restore_and_history_never_drop_constraints() -> None:
    controller = SketchDraftController(root=_sketch())
    original = controller.snapshot()

    assert original.constraints == _sketch().constraints
    restored = SketchDraftController(snapshot=original)
    assert restored.snapshot().constraints == original.constraints

    controller.delete_constraint("G1")
    changed = controller.snapshot().constraints
    assert {item.id for item in changed} == {"G2", "D1"}
    controller.undo()
    assert controller.snapshot().constraints == original.constraints
    controller.redo()
    assert controller.snapshot().constraints == changed


def test_add_constraint_rejects_non_constraint_objects() -> None:
    controller = SketchDraftController(root=_sketch())

    with pytest.raises(TypeError, match="SketchConstraint"):
        controller.add_constraint(object())  # type: ignore[arg-type]


def test_snapshot_rejects_case_insensitive_duplicate_constraint_ids() -> None:
    sketch = _sketch()
    snapshot = SketchDraftSnapshot(
        sketch.name,
        sketch.plane,
        sketch.points,
        sketch.curves,
        constraints=(
            SketchFixedConstraint("Same", "P1", 0.0, 0.0),
            SketchFixedConstraint("same", "P2", 2.0, 0.0),
        ),
    )

    with pytest.raises(ValueError, match="duplicate sketch constraint id"):
        SketchDraftController(snapshot=snapshot)
