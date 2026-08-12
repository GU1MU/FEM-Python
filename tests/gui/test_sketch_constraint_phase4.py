from __future__ import annotations

import math
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QSettings, Qt
from PySide6.QtWidgets import QAbstractItemView, QApplication, QDialog

from fem.geometry import (
    SketchCoincidentConstraint,
    SketchDistanceDimension,
    SketchFixedConstraint,
    SketchHorizontalConstraint,
    SketchLine,
    SketchPoint,
    SketchPointOnCurveConstraint,
    SketchRadiusDimension,
    SketchVerticalConstraint,
)
from fem_gui.sketch_constraint_ui import (
    build_constraint_overlays,
    infer_line_preview,
    solve_status_text,
)
from fem_gui.sketch_editor import SketchDraftController
from fem_gui.sketch_preferences import SketchPreferences, load_sketch_preferences
import fem_gui.widgets.sketch_editor_panel as sketch_editor_panel_module
from fem_gui.widgets.sketch_editor_panel import SketchEditorPanel
from fem_gui.widgets.viewport import FEMViewport


def _line_controller() -> SketchDraftController:
    controller = SketchDraftController("约束界面")
    controller.add_point("P1", 0.0, 0.0)
    controller.add_point("P2", 2.0, 0.0)
    controller.add_line("L1", "P1", "P2")
    return controller


def _application() -> QApplication:
    return QApplication.instance() or QApplication([])


def _constraint_controller() -> SketchDraftController:
    controller = SketchDraftController("七类约束")
    for point in (
        SketchPoint("P1", 0.0, 0.0),
        SketchPoint("P2", 2.0, 0.0),
        SketchPoint("P3", 0.0, 0.0),
        SketchPoint("P4", 0.0, 2.0),
    ):
        controller.add_point(point.id, point.u, point.v)
    controller.add_line("L1", "P1", "P2")
    controller.add_line("L2", "P1", "P4")
    controller.add_circle((5.0, 5.0), 1.0, point_id="O", curve_id="C1")
    return controller


def test_overlay_and_chinese_solver_status_are_renderer_independent() -> None:
    controller = _line_controller()
    result = controller.add_constraint_and_solve(
        SketchHorizontalConstraint("C1", "L1")
    )
    snapshot = controller.snapshot()
    overlays = build_constraint_overlays(
        snapshot.points,
        snapshot.curves,
        snapshot.constraints,
        snapshot.plane,
    )

    assert [(item.kind, item.text, item.entity_ids) for item in overlays] == [
        ("horizontal", "水平", ("L1",))
    ]
    assert "剩余自由度" not in solve_status_text(result)
    assert solve_status_text(result) == "欠约束"


def test_radius_overlay_shows_value_away_from_circle_center() -> None:
    controller = _constraint_controller()
    controller.add_constraint_and_solve(
        SketchRadiusDimension("R1", "C1", 1.0, driving=False)
    )
    snapshot = controller.snapshot()

    overlay = build_constraint_overlays(
        snapshot.points,
        snapshot.curves,
        snapshot.constraints,
        snapshot.plane,
    )[0]

    assert overlay.kind == "radius"
    assert overlay.text == "R 1"
    assert overlay.position[:2] == pytest.approx(
        (5.0 + 0.72 / math.sqrt(2.0), 5.0 + 0.72 / math.sqrt(2.0))
    )


def test_panel_constraint_crud_filters_selection_and_driving_dimension() -> None:
    _application()
    controller = _line_controller()
    panel = SketchEditorPanel(controller)
    controller.select_curve("L1")
    panel._selection_changed_lightweight()

    horizontal = panel.create_constraint("horizontal", ("L1",))
    assert horizontal is not None
    dimension = panel.create_constraint(
        "distance", ("L1",), value=3.0, driving=True
    )
    assert isinstance(dimension, SketchDistanceDimension)
    points = {point.id: point for point in controller.snapshot().points}
    assert math.isclose(math.dist((points["P1"].u, points["P1"].v), (points["P2"].u, points["P2"].v)), 3.0)
    assert panel.constraints_table.rowCount() == 2
    assert panel.constraints_table.columnCount() == 2
    assert [
        panel.constraints_table.horizontalHeaderItem(column).text()
        for column in range(2)
    ] == ["类型", "值"]
    assert panel.constraints_table.verticalHeader().isHidden()
    assert (
        panel.constraints_table.verticalScrollMode()
        == QAbstractItemView.ScrollMode.ScrollPerPixel
    )
    constraint_scroll = panel.constraints_table.verticalScrollBar()
    assert constraint_scroll.singleStep() == 18
    assert "width: 10px" in constraint_scroll.styleSheet()
    assert "border-radius: 4px" in constraint_scroll.styleSheet()
    assert "agent_chat_scroll_up.svg" in constraint_scroll.styleSheet()
    assert panel.constraints_table.item(0, 0).text() == "水平"
    assert panel.constraints_table.item(0, 1).text() == ""
    assert panel.constraints_table.item(1, 0).text() == "距离"
    assert panel.constraints_table.item(1, 1).text() == "3"
    assert all(
        panel.constraints_table.item(row, column).textAlignment()
        == Qt.AlignmentFlag.AlignCenter
        for row in range(2)
        for column in range(2)
    )
    assert (
        panel.constraints_table.item(1, 0).data(Qt.ItemDataRole.UserRole)
        == dimension.id
    )

    panel.delete_constraint(horizontal.id)
    assert tuple(item.id for item in controller.constraints) == (dimension.id,)
    controller.undo()
    assert {item.id for item in controller.constraints} == {horizontal.id, dimension.id}


def test_reference_dimension_ignores_typed_target_and_reports_measurement() -> None:
    _application()
    controller = _line_controller()
    panel = SketchEditorPanel(controller)
    dimension = panel.create_constraint(
        "distance", ("P1", "P2"), value=99.0, driving=False
    )
    assert dimension.value == 2.0
    assert panel.edit_dimension(dimension.id, value=123.0, driving=False)
    restored = next(item for item in controller.constraints if item.id == dimension.id)
    assert restored.value == 2.0


def test_staged_selection_creates_point_on_curve_without_stable_ids() -> None:
    _application()
    controller = _constraint_controller()
    panel = SketchEditorPanel(controller)
    panel._start_constraint_command("point_on_curve")

    assert panel.constraint_command_prompt.text() == "请选择点"
    assert not panel.confirm_constraint_command_button.isEnabled()
    panel._select_point("P3")
    assert panel.constraint_command_prompt.text() == "请选择曲线"
    panel._select_curve("L1")
    assert "已选择" not in panel.constraint_command_prompt.text()
    assert "1/" not in panel.constraint_command_prompt.text()
    assert panel.confirm_constraint_command_button.isEnabled()
    panel.confirm_constraint_command_button.click()

    assert len(controller.constraints) == 1
    assert isinstance(controller.constraints[0], SketchPointOnCurveConstraint)
    assert (controller.constraints[0].point_id, controller.constraints[0].curve_id) == (
        "P3", "L1"
    )


def test_dimension_is_reference_until_edit_dialog_sets_driving_value(
    monkeypatch,
) -> None:
    _application()
    controller = _line_controller()
    panel = SketchEditorPanel(controller)
    panel._start_constraint_command("distance")
    panel._select_curve("L1")
    panel.confirm_constraint_command_button.click()
    dimension = controller.constraints[0]
    assert isinstance(dimension, SketchDistanceDimension)
    assert not dimension.driving
    assert dimension.value == 2.0

    def accept_edit(dialog) -> QDialog.DialogCode:
        dialog.value_spin.setValue(4.0)
        return QDialog.DialogCode.Accepted

    monkeypatch.setattr(
        sketch_editor_panel_module._DimensionEditorDialog,
        "exec",
        accept_edit,
    )
    panel.constraints_table.setCurrentCell(0, 0)
    panel.edit_constraint_button.click()
    driving = controller.constraints[0]
    assert driving.driving and driving.value == 4.0
    points = {point.id: point for point in controller.snapshot().points}
    assert math.isclose(
        math.hypot(points["P2"].u - points["P1"].u, points["P2"].v - points["P1"].v),
        4.0,
    )


def test_fixed_constraint_lists_coordinates_and_edit_moves_circle(
    monkeypatch,
) -> None:
    _application()
    controller = SketchDraftController("固定圆心")
    controller.add_circle((5.0, 4.0), 2.0, point_id="O", curve_id="C1")
    panel = SketchEditorPanel(controller)
    fixed = panel.create_constraint("fixed", ("O",))

    assert isinstance(fixed, SketchFixedConstraint)
    assert panel.constraints_table.item(0, 1).text() == "5.00, 4.00"
    panel.constraints_table.setCurrentCell(0, 0)
    assert panel.edit_constraint_button.isEnabled()

    def accept_edit(dialog) -> QDialog.DialogCode:
        dialog.u_spin.setValue(7.25)
        dialog.v_spin.setValue(-1.5)
        return QDialog.DialogCode.Accepted

    monkeypatch.setattr(
        sketch_editor_panel_module._FixedConstraintEditorDialog,
        "exec",
        accept_edit,
    )
    panel.edit_constraint_button.click()

    edited = controller.constraints[0]
    center = next(point for point in controller.snapshot().points if point.id == "O")
    circle = controller.snapshot().curves[0]
    assert isinstance(edited, SketchFixedConstraint)
    assert (edited.u, edited.v) == (7.25, -1.5)
    assert (center.u, center.v) == (7.25, -1.5)
    assert circle.radius == 2.0
    assert panel.constraints_table.item(0, 1).text() == "7.25, -1.50"

    controller.undo()
    restored = next(point for point in controller.snapshot().points if point.id == "O")
    assert (restored.u, restored.v) == (5.0, 4.0)


def test_constraint_table_selection_highlights_its_sketch_entities() -> None:
    _application()
    controller = _line_controller()
    controller.add_constraint(SketchFixedConstraint("F1", "P1", 0.0, 0.0))
    controller.add_constraint(SketchHorizontalConstraint("H1", "L1"))
    controller.add_constraint(SketchPointOnCurveConstraint("C1", "P1", "L1"))
    panel = SketchEditorPanel(controller)
    viewport = FEMViewport()
    panel.attach_viewport(viewport)

    panel.constraints_table.setCurrentCell(0, 0)
    assert viewport._sketch_constraint_selection == (("point", "P1"),)

    panel.constraints_table.setCurrentCell(1, 0)
    assert viewport._sketch_constraint_selection == (("curve", "L1"),)

    panel.constraints_table.setCurrentCell(2, 0)
    assert viewport._sketch_constraint_selection == (
        ("point", "P1"),
        ("curve", "L1"),
    )

    panel.constraints_table.clearSelection()
    assert viewport._sketch_constraint_selection == ()
    viewport.close()


def test_fixed_constraint_overlay_is_offset_from_the_constrained_point() -> None:
    controller = _line_controller()
    controller.add_constraint(SketchFixedConstraint("F1", "P1", 0.0, 0.0))
    snapshot = controller.snapshot()

    overlay = build_constraint_overlays(
        snapshot.points,
        snapshot.curves,
        snapshot.constraints,
        snapshot.plane,
    )[0]

    assert overlay.kind == "fixed"
    assert overlay.text == "固定"
    assert overlay.position[:2] == pytest.approx((0.07, 0.07))


def test_staged_fixed_selection_creates_one_constraint_per_point_atomically() -> None:
    _application()
    controller = SketchDraftController("批量固定")
    controller.add_point("P1", 0.0, 0.0)
    controller.add_point("P2", 2.0, 3.0)
    controller.add_line("L1", "P1", "P2")
    panel = SketchEditorPanel(controller)
    before = controller.snapshot()

    panel._start_constraint_command("fixed")
    panel._select_point("P1")
    panel._select_point("P2")
    assert panel.confirm_constraint_command_button.isEnabled()
    assert "2 个点" in panel.constraint_command_prompt.text()
    panel._confirm_constraint_command()

    assert len(controller.constraints) == 2
    assert all(
        isinstance(item, SketchFixedConstraint) for item in controller.constraints
    )
    assert controller.snapshot().revision == before.revision + 1
    controller.undo()
    assert controller.snapshot().constraints == ()


@pytest.mark.parametrize(
    ("kind", "expected_type"),
    (
        ("vertical", SketchVerticalConstraint),
        ("distance", SketchDistanceDimension),
    ),
)
def test_staged_multi_line_selection_creates_independent_constraints_atomically(
    kind, expected_type
) -> None:
    _application()
    controller = SketchDraftController("批量直线约束")
    for point in (
        SketchPoint("P1", 0.0, 0.0),
        SketchPoint("P2", 0.0, 1.0),
        SketchPoint("P3", 2.0, 0.0),
        SketchPoint("P4", 2.0, 3.0),
    ):
        controller.add_point(point.id, point.u, point.v)
    controller.add_line("L1", "P1", "P2")
    controller.add_line("L2", "P3", "P4")
    panel = SketchEditorPanel(controller)
    before = controller.snapshot()

    panel._start_constraint_command(kind)
    panel._select_curve("L1")
    panel._select_curve("L2")
    assert panel.confirm_constraint_command_button.isEnabled()
    assert "2 条直线" in panel.constraint_command_prompt.text()
    panel._confirm_constraint_command()

    assert len(controller.constraints) == 2
    assert all(isinstance(item, expected_type) for item in controller.constraints)
    if kind == "distance":
        assert all(not item.driving for item in controller.constraints)
    assert controller.snapshot().revision == before.revision + 1
    controller.undo()
    assert controller.snapshot().constraints == ()


def test_staged_multi_radius_selection_creates_dimensions_atomically() -> None:
    _application()
    controller = SketchDraftController("批量圆弧半径")
    controller.add_circle(
        (0.0, 0.0),
        2.0,
        point_id="O1",
        curve_id="C1",
    )
    controller.add_arc(
        (4.0, 0.0),
        (5.0, 1.0),
        (4.0, 2.0),
        start_point_id="P1",
        center_point_id="O2",
        end_point_id="P2",
        curve_id="A1",
    )
    panel = SketchEditorPanel(controller)
    before = controller.snapshot()

    panel._start_constraint_command("radius")
    panel._select_curve("C1")
    panel._select_curve("A1")
    assert panel.confirm_constraint_command_button.isEnabled()
    assert "2 个圆或圆弧" in panel.constraint_command_prompt.text()
    panel._confirm_constraint_command()

    dimensions = controller.constraints
    assert len(dimensions) == 2
    assert all(isinstance(item, SketchRadiusDimension) for item in dimensions)
    assert all(not item.driving for item in dimensions)
    assert tuple(item.curve_id for item in dimensions) == ("C1", "A1")
    assert tuple(item.value for item in dimensions) == pytest.approx((2.0, 1.0))
    assert controller.snapshot().revision == before.revision + 1
    controller.undo()
    assert controller.snapshot().constraints == ()


def test_rectangle_fixed_corner_and_two_edited_lengths_fully_constrain() -> None:
    _application()
    controller = SketchDraftController("矩形约束流程")
    controller.add_rectangle((1.0, 1.0), (5.0, 3.0))
    panel = SketchEditorPanel(controller)

    panel._start_constraint_command("fixed")
    panel._select_point("P1")
    panel._confirm_constraint_command()
    for line_id in ("L1", "L2"):
        panel._start_constraint_command("distance")
        panel._select_curve(line_id)
        panel._confirm_constraint_command()

    dimensions = tuple(
        item
        for item in controller.constraints
        if isinstance(item, SketchDistanceDimension)
    )
    assert len(dimensions) == 2
    assert all(not item.driving for item in dimensions)
    assert controller.current_solve_result().status == "under_constrained"

    assert panel.edit_dimension(dimensions[0].id, value=4.0, driving=True)
    assert panel.edit_dimension(dimensions[1].id, value=2.0, driving=True)

    assert controller.current_solve_result().status == "fully_constrained"
    assert panel.render_data().constraint_status == "fully_constrained"
    point = next(item for item in controller.snapshot().points if item.id == "P1")
    assert (point.u, point.v) == (1.0, 1.0)


@pytest.mark.parametrize(
    ("kind", "targets", "value", "expected_type"),
    (
        ("coincident", ("P1", "P3"), None, SketchCoincidentConstraint),
        ("point_on_curve", ("P3", "L1"), None, SketchPointOnCurveConstraint),
        ("horizontal", ("L1",), None, SketchHorizontalConstraint),
        ("vertical", ("L2",), None, SketchVerticalConstraint),
        ("fixed", ("P1",), None, SketchFixedConstraint),
        ("distance", ("P1", "P2"), 2.0, SketchDistanceDimension),
        ("radius", ("C1",), 1.0, SketchRadiusDimension),
    ),
)
def test_all_seven_create_api_entries_validate_targets(
    kind, targets, value, expected_type
) -> None:
    _application()
    panel = SketchEditorPanel(_constraint_controller())
    created = panel.create_constraint(kind, targets, value=value)
    assert isinstance(created, expected_type)


@pytest.mark.parametrize(
    ("kind", "targets"),
    (
        ("coincident", ()),
        ("coincident", ("P1",)),
        ("point_on_curve", ("L1", "P1")),
        ("horizontal", ("P1",)),
        ("vertical", ("C1",)),
        ("fixed", ("missing",)),
        ("distance", ("C1",)),
        ("radius", ("L1",)),
    ),
)
def test_invalid_constraint_targets_raise_chinese_value_error(kind, targets) -> None:
    _application()
    panel = SketchEditorPanel(_constraint_controller())
    with pytest.raises(ValueError, match="约束目标无效"):
        panel.create_constraint(kind, targets, value=1.0)


def test_staged_selection_ignores_wrong_entity_kind_without_crashing() -> None:
    _application()
    panel = SketchEditorPanel(_constraint_controller())
    panel._start_constraint_command("point_on_curve")
    panel._select_curve("L1")

    assert panel.controller.constraints == ()
    assert panel._constraint_command_targets == []
    assert panel.constraint_command_prompt.text() == "请选择点"
    assert not panel.confirm_constraint_command_button.isEnabled()


def test_inference_preview_switch_grid_exception_and_atomic_confirmation() -> None:
    assert infer_line_preview(
        (0.0, 0.0), (2.0, 0.001), auto_constraints=True
    ).kinds == ("horizontal",)
    assert infer_line_preview(
        (0.0, 0.0), (2.0, 0.0), auto_constraints=False,
        snap_kind="sketch_point", snapped_point_id="P9",
    ).kinds == ()
    grid = infer_line_preview(
        (0.0, 0.0), (1.0, 0.2), auto_constraints=True, snap_kind="grid"
    )
    assert "fixed" not in grid.kinds and "distance" not in grid.kinds

    controller = SketchDraftController("推断")
    controller.add_point("P1", 0.0, 0.0)
    before = controller.snapshot()
    line, result = controller.add_inferred_line(
        "P1", (2.0, 0.0), horizontal=True
    )
    assert result.succeeded and isinstance(line, SketchLine)
    assert any(isinstance(item, SketchHorizontalConstraint) for item in controller.constraints)
    controller.undo()
    assert controller.snapshot().points == before.points
    assert controller.snapshot().curves == before.curves
    assert controller.snapshot().constraints == before.constraints


def test_drag_preview_is_history_free_and_release_is_one_undo() -> None:
    _application()
    controller = _line_controller()
    panel = SketchEditorPanel(controller)
    before = controller.snapshot()
    preview = panel.preview_constrained_drag("P2", 4.0, 1.0)
    assert preview.succeeded
    assert controller.snapshot() == before
    committed = panel.commit_constrained_drag("P2", 4.0, 1.0)
    assert committed.succeeded
    controller.undo()
    assert controller.snapshot().points == before.points


def test_viewport_drag_signals_use_temporary_then_atomic_panel_paths() -> None:
    _application()
    controller = _line_controller()
    panel = SketchEditorPanel(controller)
    viewport = FEMViewport()
    panel.attach_viewport(viewport)
    before = controller.snapshot()
    target = controller.plane.to_global(4.0, 1.0)
    viewport.sketchPointDragPreviewRequested.emit("P2", target)
    assert controller.snapshot() == before
    viewport.sketchPointDragCommitRequested.emit("P2", target)
    assert controller.snapshot().revision == before.revision + 1
    controller.undo()
    assert controller.snapshot().points == before.points


def test_conflict_preview_has_chinese_reason_and_writes_no_history() -> None:
    controller = _line_controller()
    controller.add_constraint(SketchFixedConstraint("F1", "P1", 0.0, 0.0))
    controller.add_constraint(SketchFixedConstraint("F2", "P2", 2.0, 0.0))
    before = controller.snapshot()
    result = controller.commit_constrained_edit(
        add_constraints=(SketchDistanceDimension("D1", "P1", "P2", 3.0),)
    )
    assert result.status == "conflicting"
    assert "约束冲突" in solve_status_text(result)
    assert "冲突候选" in solve_status_text(result)
    assert controller.snapshot() == before


def test_intersection_confirmation_adds_two_point_on_curve_relations_atomically() -> None:
    controller = SketchDraftController("交点推断")
    for point in (
        SketchPoint("A", -1.0, 0.0), SketchPoint("B", 1.0, 0.0),
        SketchPoint("C", 0.0, -1.0), SketchPoint("D", 0.0, 1.0),
        SketchPoint("S", -2.0, -2.0),
    ):
        controller.add_point(point.id, point.u, point.v)
    controller.add_line("L1", "A", "B")
    controller.add_line("L2", "C", "D")
    before = controller.snapshot()
    line, result = controller.add_inferred_line(
        "S", (0.0, 0.0), intersection_curve_ids=("L1", "L2")
    )
    assert result.succeeded and line is not None
    assert [type(item).__name__ for item in controller.constraints] == [
        "SketchPointOnCurveConstraint", "SketchPointOnCurveConstraint"
    ]
    controller.undo()
    assert controller.snapshot().points == before.points
    assert controller.snapshot().curves == before.curves
    assert controller.snapshot().constraints == ()


def test_viewport_signal_path_confirms_intersection_auto_relations_once() -> None:
    _application()
    controller = SketchDraftController("信号交点")
    for point in (
        SketchPoint("A", -1.0, 0.0), SketchPoint("B", 1.0, 0.0),
        SketchPoint("C", 0.0, -1.0), SketchPoint("D", 0.0, 1.0),
        SketchPoint("S", -2.0, -2.0),
    ):
        controller.add_point(point.id, point.u, point.v)
    controller.add_line("L1", "A", "B")
    controller.add_line("L2", "C", "D")
    panel = SketchEditorPanel(controller)
    viewport = FEMViewport()
    panel.attach_viewport(viewport)
    viewport.sketchSnapConfirmed.emit(
        {"kind": "sketch_point", "point_id": "S", "curve_ids": ()}
    )
    viewport.sketchWorkPlanePointSelected.emit(controller.plane.to_global(-2.0, -2.0))
    revision = controller.snapshot().revision
    viewport.sketchSnapConfirmed.emit(
        {"kind": "intersection", "point_id": None, "curve_ids": ("L1", "L2")}
    )
    viewport.sketchWorkPlanePointSelected.emit(controller.plane.to_global(0.0, 0.0))
    assert controller.snapshot().revision == revision + 1
    assert [type(item).__name__ for item in controller.constraints] == [
        "SketchPointOnCurveConstraint", "SketchPointOnCurveConstraint"
    ]


def test_viewport_signal_path_auto_off_reuses_points_and_grid_adds_no_constraints() -> None:
    _application()
    controller = SketchDraftController("信号开关")
    controller.add_point("P1", 0.0, 0.0)
    controller.add_point("P2", 2.0, 0.1)
    panel = SketchEditorPanel(controller)
    viewport = FEMViewport()
    panel.attach_viewport(viewport)
    panel._preferences = SketchPreferences(auto_constraints=False)
    viewport.sketchSnapConfirmed.emit(
        {"kind": "sketch_point", "point_id": "P1", "curve_ids": ()}
    )
    viewport.sketchWorkPlanePointSelected.emit(controller.plane.to_global(0.0, 0.0))
    viewport.sketchSnapConfirmed.emit(
        {"kind": "sketch_point", "point_id": "P2", "curve_ids": ()}
    )
    viewport.sketchWorkPlanePointSelected.emit(controller.plane.to_global(2.0, 0.1))
    first_line = controller.snapshot().curves[-1]
    assert isinstance(first_line, SketchLine)
    assert (first_line.start_point_id, first_line.end_point_id) == ("P1", "P2")
    assert len(controller.snapshot().points) == 2
    assert controller.constraints == ()
    viewport.sketchSnapConfirmed.emit(
        {"kind": "grid", "point_id": None, "curve_ids": ()}
    )
    viewport.sketchWorkPlanePointSelected.emit(controller.plane.to_global(3.0, 0.4))
    assert controller.constraints == ()


def test_hover_preview_and_cancel_are_visible_but_history_free() -> None:
    _application()
    controller = SketchDraftController("预览取消")
    controller.add_point("P1", 0.0, 0.0)
    panel = SketchEditorPanel(controller)
    viewport = FEMViewport()
    panel.attach_viewport(viewport)
    viewport._sketch_draft_render_data = panel.render_data()
    viewport.sketchSnapConfirmed.emit(
        {"kind": "sketch_point", "point_id": "P1", "curve_ids": ()}
    )
    viewport.sketchWorkPlanePointSelected.emit(controller.plane.to_global(0.0, 0.0))
    before = controller.snapshot()
    viewport.sketchInferencePreviewChanged.emit(
        {
            "point": controller.plane.to_global(2.0, 0.0),
            "snap_kind": "grid",
            "point_id": None,
            "curve_ids": (),
        }
    )
    assert viewport._sketch_draft_render_data.inference_preview == ("horizontal",)
    assert controller.snapshot() == before
    viewport.sketchInferencePreviewChanged.emit({"point": None})
    assert viewport._sketch_draft_render_data.inference_preview == ()
    viewport.sketchInferencePreviewChanged.emit(
        {
            "point": controller.plane.to_global(2.0, 0.0),
            "snap_kind": "grid",
            "point_id": None,
            "curve_ids": (),
        }
    )
    assert viewport._sketch_draft_render_data.inference_preview == ("horizontal",)
    viewport.sketchInferencePreviewChanged.emit("invalid context")
    assert viewport._sketch_draft_render_data.inference_preview == ()
    viewport.sketchPendingInteractionCancelled.emit()
    assert viewport._sketch_draft_render_data.inference_preview == ()
    assert controller.snapshot() == before

def test_auto_constraint_uses_fixed_default_despite_existing_store(tmp_path) -> None:
    _application()
    store = QSettings(str(tmp_path / "sketch.ini"), QSettings.Format.IniFormat)
    store.setValue("sketch/auto_constraints", False)
    panel = SketchEditorPanel(SketchDraftController("偏好"), settings=store)
    assert not hasattr(panel, "auto_constraints_check")
    assert panel._preferences.auto_constraints
    panel.spacing_spin.setValue(0.25)
    store.sync()
    assert load_sketch_preferences(store).auto_constraints is True
