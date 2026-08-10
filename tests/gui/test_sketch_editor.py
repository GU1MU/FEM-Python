from __future__ import annotations

import pytest

from fem.geometry import (
    LogicalEntityRef,
    SketchExternalReference,
    SketchExternalReferenceType,
    SketchHorizontalConstraint,
    SketchReferencePoint,
    SketchVerticalConstraint,
)
from fem_gui.sketch_editor import (
    SketchDraftController,
    SketchDraftValidationError,
)


def _reference_point(u: float, v: float) -> SketchReferencePoint:
    return SketchReferencePoint(
        SketchExternalReference(
            "R1",
            LogicalEntityRef("point:support/P1"),
            SketchExternalReferenceType.TOPOLOGY_VERTEX,
        ),
        (u, v, 0.0),
        u,
        v,
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
    assert tuple(type(item) for item in controller.constraints) == (
        SketchHorizontalConstraint,
        SketchVerticalConstraint,
        SketchHorizontalConstraint,
        SketchVerticalConstraint,
    )
    assert all(item.source == "inferred" for item in controller.constraints)
    assert controller.can_finish
    assert len(controller.profiles) == 1

    controller.undo()
    assert controller.snapshot().points == ()
    assert controller.snapshot().curves == ()
    assert controller.constraints == ()
    controller.redo()
    assert len(controller.snapshot().curves) == 4
    assert len(controller.constraints) == 4


def test_selection_is_transient_and_does_not_change_history() -> None:
    authored = SketchDraftController("Selection")
    authored.add_rectangle((0.0, 0.0), (2.0, 1.0))
    controller = SketchDraftController.from_sketch_geometry(authored.finish())
    point = controller.snapshot().points[0]
    revision = controller.snapshot().revision
    can_undo = controller.can_undo
    dirty = controller.dirty

    controller.select_point(point.id)
    controller.clear_selection()
    controller.select_point(point.id)

    assert controller.snapshot().revision == revision
    assert controller.can_undo is can_undo
    assert controller.dirty is dirty
    assert revision == 0
    assert not can_undo
    assert not dirty


def test_point_usage_and_multi_delete_are_one_stable_edit() -> None:
    controller = SketchDraftController("point-list")
    start = controller.add_point(0.0, 0.0, point_id="P1")
    end = controller.add_point(2.0, 0.0, point_id="P2")
    center = controller.add_point(1.0, 1.0, point_id="P3")
    controller.add_line(start.id, end.id, curve_id="L1")
    controller.add_circle((center.u, center.v), 0.5, point_id="P4", curve_id="C1")

    assert controller.point_usage(start.id) == ("端点",)
    assert controller.point_usage("P4") == ("圆心",)
    before = controller.snapshot()
    controller.select_many([start.id, end.id])
    controller.delete_many([start.id, end.id])

    assert controller.snapshot().revision == before.revision + 1
    assert "L1" not in {curve.id for curve in controller.snapshot().curves}
    controller.undo()
    assert controller.snapshot() == before


def test_profile_selection_does_not_repeat_profile_analysis(monkeypatch) -> None:
    controller = SketchDraftController("profile-selection")
    controller.add_rectangle((0.0, 0.0), (2.0, 1.0))
    profile_id = controller.profiles[0].id
    revision = controller.snapshot().revision
    monkeypatch.setattr(
        controller,
        "derive_profiles",
        lambda: (_ for _ in ()).throw(
            AssertionError("profile selection repeated contour analysis")
        ),
    )

    controller.select_profile(profile_id)

    assert controller.selected_ids == (profile_id,)
    assert controller.snapshot().revision == revision


def test_draw_segment_and_batch_move_are_single_atomic_edits() -> None:
    controller = SketchDraftController("Atomic edits")
    start = controller.add_point(0.0, 0.0)
    before_segment = controller.snapshot()

    line = controller.add_line_to_point(start.id, (2.0, 0.0))

    assert line.end_point_id in {point.id for point in controller.snapshot().points}
    controller.undo()
    assert controller.snapshot() == before_segment

    first = controller.add_point(1.0, 0.0)
    second = controller.add_point(2.0, 0.0)
    before_move = controller.snapshot()
    controller.move_points(
        {
            first.id: (1.0, 1.0),
            second.id: (2.0, 2.0),
        }
    )
    controller.undo()
    assert controller.snapshot() == before_move


def test_composite_edit_failure_rolls_back_entire_draft(monkeypatch) -> None:
    controller = SketchDraftController("Rollback")
    start = controller.add_point(0.0, 0.0)
    before = controller.snapshot()

    def fail_binding(_point_id, _reference_point) -> None:
        raise ValueError("binding failed")

    monkeypatch.setattr(controller, "_bind_external_reference", fail_binding)
    with pytest.raises(ValueError, match="binding failed"):
        controller.add_line_to_point(
            start.id,
            (1.0, 0.0),
            external_reference=_reference_point(1.0, 0.0),
        )

    assert controller.snapshot() == before


def test_point_cascade_delete_lists_dependencies_and_undo_restores_all() -> None:
    controller = SketchDraftController("Cascade")
    first = controller.add_point(0.0, 0.0)
    shared = controller.add_point(1.0, 0.0)
    last = controller.add_point(2.0, 0.0)
    first_line = controller.add_line(first.id, shared.id)
    second_line = controller.add_line(shared.id, last.id)
    before = controller.snapshot()

    assert controller.dependent_curve_ids(shared.id) == (
        first_line.id,
        second_line.id,
    )
    controller.delete_point(shared.id)
    assert controller.snapshot().curves == ()
    controller.undo()
    assert controller.snapshot() == before


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
