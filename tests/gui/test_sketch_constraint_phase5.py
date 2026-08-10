from __future__ import annotations

import os

import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from fem.geometry import (
    LogicalEntityRef,
    SketchAngleDimension,
    SketchArc,
    SketchCircle,
    SketchFixedConstraint,
    SketchExternalReference,
    SketchExternalReferenceType,
    SketchGeometry,
    SketchLine,
    SketchParallelConstraint,
    SketchPlane,
    SketchPoint,
    SketchReferencePoint,
)
from fem_gui.sketch_editor import SketchDraftController
from fem_gui.widgets.sketch_editor_panel import SketchEditorPanel


def _line_controller() -> SketchDraftController:
    return SketchDraftController.from_geometry(
        SketchGeometry(
            "split",
            SketchPlane.xy(),
            (
                SketchPoint("P1", 0.0, 0.0), SketchPoint("P2", 2.0, 0.0),
                SketchPoint("P3", 0.0, 1.0), SketchPoint("P4", 2.0, 1.0),
            ),
            (SketchLine("L1", "P1", "P2"), SketchLine("L2", "P3", "P4")),
        )
    )


def _application() -> QApplication:
    return QApplication.instance() or QApplication([])


def test_controller_split_selection_and_constraints_are_one_undo_redo_step() -> None:
    controller = _line_controller()
    controller.add_constraint(SketchParallelConstraint("Parallel", "L1", "L2"))
    controller.select_curve("L1")
    before = controller.snapshot()

    result = controller.split_curve_at("L1", 0.5)
    after = controller.snapshot()
    assert after.selected_ids == result.derived_curve_ids
    assert len(after.constraints) == 2

    undone = controller.undo()
    assert undone.curves == before.curves
    assert undone.constraints == before.constraints
    assert undone.selected_ids == ("L1",)
    redone = controller.redo()
    assert redone.curves == after.curves
    assert redone.constraints == after.constraints
    assert redone.selected_ids == after.selected_ids


def test_circle_line_trim_splits_at_both_real_intersections_and_removes_clicked_arc() -> None:
    controller = SketchDraftController.from_geometry(
        SketchGeometry(
            "trim",
            SketchPlane.xy(),
            (
                SketchPoint("C", 0.0, 0.0),
                SketchPoint("LS", -2.0, 0.0), SketchPoint("LE", 2.0, 0.0),
            ),
            (SketchCircle("O", "C", 1.0), SketchLine("L", "LS", "LE")),
        )
    )

    remaining = controller.trim_at_intersection("O", "L", (0.0, 1.0))
    snapshot = controller.snapshot()
    assert len(remaining) == 1
    assert len([item for item in snapshot.curves if isinstance(item, SketchArc)]) == 1
    assert len([item for item in snapshot.curves if isinstance(item, SketchLine)]) == 3
    assert {round(point.u, 8) for point in snapshot.points if abs(point.v) < 1.0e-8} >= {-1.0, 1.0}
    controller.undo()
    assert [type(item).__name__ for item in controller.snapshot().curves] == [
        "SketchCircle", "SketchLine"
    ]


def test_trim_cascades_constraint_on_orphaned_endpoint_and_undo_restores_it() -> None:
    controller = SketchDraftController.from_geometry(
        SketchGeometry(
            "tangent trim",
            SketchPlane.xy(),
            (
                SketchPoint("C", 0.0, 0.0),
                SketchPoint("LS", -2.0, 1.0), SketchPoint("LE", 2.0, 1.0),
            ),
            (SketchLine("L", "LS", "LE"), SketchCircle("O", "C", 1.0)),
            (SketchFixedConstraint("F", "LS", -2.0, 1.0),),
        )
    )
    controller.associate_point(
        "LS",
        SketchReferencePoint(
            SketchExternalReference(
                "R1",
                LogicalEntityRef("point:support/P1"),
                SketchExternalReferenceType.TOPOLOGY_VERTEX,
            ),
            (-2.0, 1.0, 0.0),
            -2.0,
            1.0,
        ),
    )

    controller.trim_at_intersection("L", "O", (-1.5, 1.0))
    assert "LS" not in {point.id for point in controller.snapshot().points}
    assert controller.constraints == ()
    assert controller.snapshot().external_coincidences == ()
    assert controller.snapshot().external_references == ()
    restored = controller.undo()
    assert "LS" in {point.id for point in restored.points}
    assert tuple(item.id for item in restored.constraints) == ("F",)
    assert tuple(item.point_id for item in restored.external_coincidences) == ("LS",)


def test_panel_creates_displays_and_deletes_advanced_relation_and_angle() -> None:
    _application()
    controller = _line_controller()
    panel = SketchEditorPanel(controller)

    parallel = panel.create_constraint("parallel", ("L1", "L2"))
    angle = panel.create_constraint(
        "angle", ("L1", "L2"), value=0.0, driving=False
    )
    assert isinstance(parallel, SketchParallelConstraint)
    assert isinstance(angle, SketchAngleDimension)
    assert panel.constraints_table.rowCount() == 2
    assert panel.constraints_table.item(0, 0).text() == "平行"
    angle_row = next(
        row
        for row in range(2)
        if panel.constraints_table.item(row, 0).text() == "角度"
    )
    assert panel.constraints_table.item(angle_row, 1).text() == "0"

    panel.delete_constraint(parallel.id)
    assert tuple(item.id for item in controller.constraints) == (angle.id,)
