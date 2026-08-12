from __future__ import annotations

from fem.geometry import (
    LogicalEntityRef,
    SketchCoincidentConstraint,
    SketchDistanceDimension,
    SketchExternalReference,
    SketchExternalReferenceType,
    SketchFixedConstraint,
    SketchGeometry,
    SketchHorizontalConstraint,
    SketchLine,
    SketchPlane,
    SketchPoint,
    SketchReferencePoint,
)
from fem_gui.sketch_editor import SketchDraftController


def _controller() -> SketchDraftController:
    return SketchDraftController(
        root=SketchGeometry(
            "atomic",
            SketchPlane.xy(),
            (SketchPoint("P1", 0.0, 0.0), SketchPoint("P2", 2.0, 1.0)),
            (SketchLine("L1", "P1", "P2"),),
            (SketchFixedConstraint("F", "P1", 0.0, 0.0),),
        )
    )


def test_successful_constrained_edit_is_one_atomic_undo_step() -> None:
    controller = _controller()
    before = controller.snapshot()

    result = controller.commit_constrained_edit(
        add_constraints=(
            SketchHorizontalConstraint("H", "L1"),
            SketchDistanceDimension("D", "P1", "P2", 3.0),
        )
    )

    assert result.status == "fully_constrained"
    assert controller.snapshot().revision == before.revision + 1
    assert controller.can_undo
    controller.undo()
    assert controller.snapshot() == before


def test_replacing_constraint_preserves_its_original_order() -> None:
    controller = _controller()
    horizontal = SketchHorizontalConstraint("H", "L1")
    distance = SketchDistanceDimension("D", "P1", "P2", 3.0)
    assert controller.add_constraint_and_solve(horizontal).succeeded
    assert controller.add_constraint_and_solve(distance).succeeded
    before_ids = tuple(item.id for item in controller.constraints)

    result = controller.replace_constraint_and_solve(
        "D",
        SketchDistanceDimension("D", "P1", "P2", 4.0),
    )

    assert result.succeeded
    assert tuple(item.id for item in controller.constraints) == before_ids


def test_conflicting_atomic_edit_preserves_geometry_constraints_revision_and_history() -> None:
    controller = _controller()
    before = controller.snapshot()
    assert not controller.can_undo

    result = controller.add_constraint_and_solve(
        SketchFixedConstraint("Conflict", "P1", 4.0, 0.0)
    )

    assert result.status == "conflicting"
    assert result.conflicting_constraint_ids[0] == "Conflict"
    assert controller.snapshot() == before
    assert not controller.can_undo


def test_temporary_solve_never_mutates_and_constrained_move_commits_once() -> None:
    controller = _controller()
    controller.add_constraint(SketchHorizontalConstraint("H", "L1"))
    before = controller.snapshot()
    undo_was_available = controller.can_undo

    preview = controller.solve_constraints_temporary(
        point_coordinates={"P2": (4.0, 0.0)}
    )
    assert preview.succeeded
    assert controller.snapshot() == before
    assert controller.can_undo is undo_was_available

    committed = controller.move_points_constrained({"P2": (4.0, 0.0)})
    assert committed.succeeded
    assert controller.snapshot().revision == before.revision + 1
    assert controller.snapshot().points[1].u == 4.0


def test_external_coincidence_is_a_known_condition_and_rejection_adds_no_history() -> None:
    controller = _controller()
    reference = SketchReferencePoint(
        SketchExternalReference(
            "R1",
            LogicalEntityRef("point:support/P2"),
            SketchExternalReferenceType.TOPOLOGY_VERTEX,
        ),
        (2.0, 1.0, 0.0),
        2.0,
        1.0,
    )
    controller.associate_point("P2", reference)
    before = controller.snapshot()

    result = controller.add_constraint_and_solve(
        SketchHorizontalConstraint("H", "L1")
    )

    assert result.status == "conflicting"
    assert controller.snapshot() == before
    controller.undo()
    assert controller.snapshot().external_coincidences == ()


def test_constraint_that_collapses_an_existing_line_fails_atomically() -> None:
    controller = _controller()
    before = controller.snapshot()

    result = controller.add_constraint_and_solve(
        SketchCoincidentConstraint("Collapse", "P1", "P2")
    )

    assert result.status == "failed"
    assert controller.snapshot() == before
    assert not controller.can_undo
