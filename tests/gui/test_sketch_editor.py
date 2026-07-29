from __future__ import annotations

import pytest

from fem_gui.sketch_editor import (
    SketchDraftController,
    SketchDraftValidationError,
)


def test_empty_draft_stays_detached_until_finish() -> None:
    controller = SketchDraftController("")

    assert controller.snapshot().points == ()
    assert not controller.can_finish
    assert {item.code for item in controller.finish_diagnostics} >= {
        "sketch.blank-name",
        "sketch.invalid-domain",
    }
    with pytest.raises(SketchDraftValidationError):
        controller.finish()


def test_rectangle_is_one_undo_command_and_has_profile() -> None:
    controller = SketchDraftController("Plate")
    controller.add_rectangle((0.0, 0.0), (4.0, 3.0))

    assert len(controller.snapshot().points) == 4
    assert len(controller.snapshot().curves) == 4
    assert controller.can_finish
    assert len(controller.profiles) == 1

    controller.undo()
    assert controller.snapshot().points == ()
    assert controller.snapshot().curves == ()
    controller.redo()
    assert len(controller.snapshot().curves) == 4


def test_id_generation_is_case_insensitive_and_redo_branch_clears() -> None:
    controller = SketchDraftController()
    controller.add_point(0.0, 0.0, point_id="p1")
    with pytest.raises(ValueError, match="已被占用"):
        controller.add_point(1.0, 0.0, point_id="P1")
    controller.add_point(1.0, 0.0)
    controller.undo()
    controller.add_point(2.0, 0.0)
    assert not controller.can_redo


def test_circle_and_three_point_arc_are_strict_entities() -> None:
    controller = SketchDraftController("Curves")
    circle = controller.add_circle((0.0, 0.0), 1.0)
    arc = controller.add_arc(
        (1.0, 0.0),
        (0.0, 1.0),
        (-1.0, 0.0),
    )

    assert circle.id.startswith("C")
    assert arc.id.startswith("A")
    assert controller.snapshot().curves[-1].id == arc.id
    assert not controller.can_finish


def test_finish_returns_detached_strict_geometry_and_restore_is_safe() -> None:
    controller = SketchDraftController("Plate")
    controller.add_rectangle((0.0, 0.0), (2.0, 1.0))
    sketch = controller.finish()
    snapshot = controller.snapshot()

    restored = SketchDraftController.from_sketch_geometry(sketch)
    assert restored.finish() == sketch
    restored.move_point("P1", -1.0, -1.0)
    assert sketch.point("P1").u == 0.0
    restored.restore_snapshot(snapshot)
    assert restored.finish() == sketch


def test_trim_splits_a_crossing_line_as_one_undo_command() -> None:
    controller = SketchDraftController("Trim")
    controller.add_point(-1.0, 0.5, point_id="P1")
    controller.add_point(3.0, 0.5, point_id="P2")
    controller.add_point(0.0, 0.0, point_id="P3")
    controller.add_point(0.0, 1.0, point_id="P4")
    controller.add_point(2.0, 0.0, point_id="P5")
    controller.add_point(2.0, 1.0, point_id="P6")
    target = controller.add_line("P1", "P2", curve_id="L1")
    controller.add_line("P3", "P4", curve_id="L2")
    controller.add_line("P5", "P6", curve_id="L3")
    before = controller.snapshot()

    replacements = controller.trim(target.id)

    assert len(replacements) == 2
    assert len(controller.snapshot().curves) == 4
    controller.undo()
    assert controller.snapshot() == before
    controller.redo()
    assert len(controller.snapshot().curves) == 4


def test_trim_with_one_intersection_removes_clicked_side() -> None:
    controller = SketchDraftController("Trim one side")
    controller.add_point(-1.0, 0.0, point_id="P1")
    controller.add_point(1.0, 0.0, point_id="P2")
    controller.add_point(0.0, -1.0, point_id="P3")
    controller.add_point(0.0, 1.0, point_id="P4")
    target = controller.add_line("P1", "P2", curve_id="L1")
    controller.add_line("P3", "P4", curve_id="L2")

    replacements = controller.trim_curve(target.id, (0.75, 0.4))

    assert len(replacements) == 1
    assert replacements[0].id == target.id
    snapshot = controller.snapshot()
    points = {point.id: point for point in snapshot.points}
    kept = replacements[0]
    assert (points[kept.start_point_id].u, points[kept.start_point_id].v) == (
        -1.0,
        0.0,
    )
    assert (points[kept.end_point_id].u, points[kept.end_point_id].v) == (
        0.0,
        0.0,
    )


def test_trim_deletes_whole_curve_when_overlap_has_no_unique_boundary() -> None:
    controller = SketchDraftController("Trim overlap")
    controller.add_point(0.0, 0.0, point_id="P1")
    controller.add_point(2.0, 0.0, point_id="P2")
    controller.add_point(1.0, 0.0, point_id="P3")
    controller.add_point(3.0, 0.0, point_id="P4")
    target = controller.add_line("P1", "P2", curve_id="L1")
    overlap = controller.add_line("P3", "P4", curve_id="L2")
    before = controller.snapshot()

    replacements = controller.trim_curve(target.id, (0.5, 0.2))

    assert replacements == ()
    assert tuple(curve.id for curve in controller.snapshot().curves) == (
        overlap.id,
    )
    assert {point.id for point in controller.snapshot().points} == {"P3", "P4"}
    controller.undo()
    assert controller.snapshot() == before
