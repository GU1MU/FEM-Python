from __future__ import annotations

import os
import math

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import (
    QEvent,
    QPoint,
    QPointF,
    QItemSelectionModel,
    QSettings,
    Qt,
)
from PySide6.QtGui import QWheelEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QDialog,
    QGroupBox,
    QLabel,
    QLineEdit,
    QListWidget,
)

from fem.geometry import (
    LogicalEntityRef,
    RectangleGeometry,
    SketchCircle,
    SketchExternalReference,
    SketchExternalReferenceType,
    SketchGeometry,
    SketchLine,
    SketchPoint,
    SketchReferencePoint,
)
import fem_gui.main_window as main_window_module
import fem_gui.widgets.sketch_editor_panel as sketch_editor_panel_module
from fem_gui.geometry_preview import GeometryPreview
from fem_gui.main_window import FEMMainWindow
from fem_gui.sketch_preferences import load_sketch_preferences
from fem_gui.sketch_editor import SketchDraftController
from fem_gui.widgets.sketch_editor_panel import SketchEditorPanel
from fem_gui.widgets.viewport import (
    FEMViewport,
    SketchDraftRenderData,
    _sketch_axis_local_endpoints,
    _sketch_camera_bounds,
    _sketch_curve_sample_count,
    _sketch_intersection_points,
    _sketch_snap_label,
    _sketch_shape_preview_points,
)


def _application() -> QApplication:
    return QApplication.instance() or QApplication([])


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


def _wheel_event(delta: int = -120) -> QWheelEvent:
    return QWheelEvent(
        QPointF(20.0, 20.0),
        QPointF(20.0, 20.0),
        QPoint(),
        QPoint(0, delta),
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
        Qt.ScrollPhase.ScrollUpdate,
        False,
    )


def test_panel_uses_fine_default_grid_and_omits_curve_profile_lists() -> None:
    _application()
    panel = SketchEditorPanel(SketchDraftController("compact-panel"))

    assert panel.spacing_spin.value() == 0.1
    assert panel.spacing_spin.text() == "0.100"
    assert not hasattr(panel, "curves_list")
    assert not hasattr(panel, "profiles_list")
    assert panel.findChildren(QListWidget) == []
    labels = {label.text() for label in panel.findChildren(QLabel)}
    assert "曲线" not in labels
    assert "Profiles" not in labels


def test_panel_keeps_advanced_sketch_behavior_internal() -> None:
    _application()
    panel = SketchEditorPanel(SketchDraftController("fixed-defaults"))

    group_titles = {
        group.title() for group in panel.findChildren(QGroupBox)
    }
    assert "网格" in group_titles
    assert "约束" in group_titles
    assert group_titles.isdisjoint({"捕捉", "显示", "绘图行为"})
    assert "草图约束与尺寸" not in group_titles
    for object_name in ("sketchGridGroup", "sketchConstraintGroup"):
        group = panel.findChild(QGroupBox, object_name)
        assert group is not None
        assert "border: none" in group.styleSheet()
    assert "点坐标" not in {
        label.text() for label in panel.findChildren(QLabel)
    }
    assert panel.point_search_edit.isHidden()
    assert panel.point_filter_combo.isHidden()
    assert panel.points_table.isHidden()
    assert panel.constraint_type_combo.isHidden()
    assert panel.constraint_targets_edit.isHidden()
    assert panel.constraint_value_spin.isHidden()
    assert panel.constraint_driving_check.isHidden()
    assert panel.solve_status_label.isHidden()
    assert panel.diagnostic_scroll.isHidden()
    assert not hasattr(panel, "delete_button")
    assert not hasattr(panel, "release_association_button")
    for attribute in (
        "snap_check",
        "snap_sketch_points_check",
        "snap_external_points_check",
        "snap_midpoints_check",
        "snap_centers_check",
        "snap_intersections_check",
        "screen_snap_tolerance_spin",
        "auto_merge_tolerance_spin",
        "show_point_ids_check",
        "show_external_labels_check",
        "show_profile_fill_check",
        "show_work_plane_axes_check",
        "continuous_polyline_check",
        "end_polyline_on_close_check",
        "keep_tool_after_completion_check",
        "confirm_cascade_delete_check",
        "auto_constraints_check",
    ):
        assert not hasattr(panel, attribute)
    assert panel._preferences.grid_snap
    assert panel._preferences.snap_sketch_points
    assert panel._preferences.snap_external_points
    assert panel._preferences.snap_midpoints
    assert panel._preferences.snap_centers
    assert panel._preferences.snap_intersections
    assert panel._preferences.auto_constraints


def test_new_controller_discards_reference_points_from_previous_sketch() -> None:
    _application()
    panel = SketchEditorPanel(SketchDraftController("previous-sketch"))
    panel.set_reference_points((_reference_point(1.0, 2.0),))

    panel.set_controller(SketchDraftController("new-sketch"))

    assert panel._reference_points == ()
    panel.close()


def test_new_sketch_after_part_deletion_has_no_stale_reference_points(
    monkeypatch,
) -> None:
    _application()
    window = FEMMainWindow()
    window._create_native_model("模型-1")
    window._apply_session_delta(
        window.session.add_native_part(
            RectangleGeometry("旧草图", 2.0, 1.0),
            name="旧部件",
        )
    )
    stale_reference = _reference_point(1.0, 2.0)
    window.sketch_editor_panel.set_reference_points((stale_reference,))
    monkeypatch.setattr(
        main_window_module.QMessageBox,
        "question",
        lambda *_args, **_kwargs: (
            main_window_module.QMessageBox.StandardButton.Yes
        ),
    )
    monkeypatch.setattr(
        main_window_module.QInputDialog,
        "getText",
        lambda *_args, **_kwargs: ("新部件", True),
    )

    window.delete_geometry()
    window.start_sketch_geometry()

    assert window.document.parts == ()
    assert window.sketch_editor_panel._reference_points == ()
    assert window.viewport._sketch_reference_points == ()
    window.cancel_sketch_geometry()
    window.close()


def test_constraint_type_click_starts_choice_without_extra_confirmation() -> None:
    _application()
    dialog = sketch_editor_panel_module._ConstraintTypeDialog()
    item = next(
        dialog.type_list.item(index)
        for index in range(dialog.type_list.count())
        if dialog.type_list.item(index).data(Qt.ItemDataRole.UserRole) == "fixed"
    )

    dialog.type_list.itemClicked.emit(item)

    assert dialog.selected_kind == "fixed"
    assert dialog.result() == QDialog.DialogCode.Accepted
    dialog.close()


def test_constraint_type_dialog_uses_smooth_agent_chat_scrollbar() -> None:
    app = _application()
    dialog = sketch_editor_panel_module._ConstraintTypeDialog()
    dialog.type_list.setFixedHeight(120)
    dialog.show()
    app.processEvents()

    assert (
        dialog.type_list.verticalScrollMode()
        == QAbstractItemView.ScrollMode.ScrollPerPixel
    )
    assert dialog.type_list.horizontalScrollBarPolicy() == (
        Qt.ScrollBarPolicy.ScrollBarAlwaysOff
    )
    scroll_bar = dialog.type_list.verticalScrollBar()
    assert scroll_bar.maximum() > 0
    assert scroll_bar.singleStep() == 18
    assert "width: 10px" in scroll_bar.styleSheet()
    assert "border-radius: 4px" in scroll_bar.styleSheet()
    assert "agent_chat_scroll_up.svg" in scroll_bar.styleSheet()

    QApplication.sendEvent(dialog.type_list.viewport(), _wheel_event())
    app.processEvents()

    assert 0 < scroll_bar.value() < scroll_bar.pageStep()
    dialog.close()


def test_constraint_actions_share_one_row_and_success_prompt_is_removed() -> None:
    app = _application()
    controller = SketchDraftController("constraint-actions")
    controller.add_rectangle((0.0, 0.0), (2.0, 1.0))
    panel = SketchEditorPanel(controller)
    panel.resize(420, 650)
    panel.show()
    app.processEvents()

    button_y = {
        button.mapTo(panel, QPoint(0, 0)).y()
        for button in (
            panel.add_constraint_button,
            panel.delete_constraint_button,
            panel.edit_constraint_button,
        )
    }
    assert len(button_y) == 1
    assert panel.diagnostic_label.text() == ""
    assert panel.diagnostic_scroll.isHidden()
    assert panel.solve_status_label.isHidden()
    panel.close()


def test_constraint_command_bar_uses_staged_prompts_and_target_highlights() -> None:
    _application()
    controller = SketchDraftController("staged-command")
    controller.add_point("P1", 0.0, 0.0)
    controller.add_point("P2", 1.0, 0.0)
    panel = SketchEditorPanel(controller)
    viewport = FEMViewport()
    panel.begin(viewport)

    panel._start_constraint_command("coincident")
    assert panel.constraint_command_bar.parentWidget() is viewport
    assert "QLabel#sketchConstraintCommandPrompt" in (
        panel.constraint_command_bar.styleSheet()
    )
    assert "background: transparent" in panel.constraint_command_bar.styleSheet()
    assert panel.constraint_command_prompt.text() == "请选择第一个点"
    panel._select_point("P1")
    assert panel.constraint_command_prompt.text() == "请选择第二个点"
    assert viewport._sketch_constraint_selection == (("point", "P1"),)

    panel._select_point("P1")
    assert panel.constraint_command_prompt.text() == "请选择第一个点"
    assert viewport._sketch_constraint_selection == ()

    panel._select_point("P1")
    panel._select_point("P2")
    assert panel.constraint_command_prompt.text() == "重合：点击确定添加约束"
    assert "已选择" not in panel.constraint_command_prompt.text()
    panel.cancel_constraint_command_button.click()
    assert panel._constraint_command_kind is None
    assert not viewport._sketch_constraint_selection_active
    panel.end()
    viewport.close()


def test_removed_entity_buttons_remain_available_by_keyboard_and_context_menu() -> None:
    _application()
    controller = SketchDraftController("context-actions")
    point = controller.add_point(
        0.0,
        0.0,
        external_reference=_reference_point(0.0, 0.0),
    )
    controller.select_point(point.id)
    panel = SketchEditorPanel(controller)
    viewport = FEMViewport()
    panel.attach_viewport(viewport)

    menu = panel._create_sketch_context_menu("point", point.id)

    assert [action.text() for action in menu.actions()] == ["删除", "解除关联"]
    assert not hasattr(panel, "delete_button")
    assert not hasattr(panel, "release_association_button")
    viewport.sketchDeleteRequested.emit()
    assert controller.snapshot().points == ()
    menu.close()
    panel.close()
    viewport.close()


def test_rectangle_and_circle_second_click_previews_follow_cursor() -> None:
    rectangle = _sketch_shape_preview_points(
        "rectangle",
        ((1.0, 2.0, 0.0),),
        (4.0, 6.0, 0.0),
    )
    assert rectangle == (
        (1.0, 2.0, 0.0),
        (4.0, 2.0, 0.0),
        (4.0, 6.0, 0.0),
        (1.0, 6.0, 0.0),
        (1.0, 2.0, 0.0),
    )

    circle = _sketch_shape_preview_points(
        "circle",
        ((1.0, 2.0, 0.0),),
        (4.0, 6.0, 0.0),
    )
    assert len(circle) == 65
    assert circle[0] == pytest.approx(circle[-1])
    assert all(
        math.hypot(point[0] - 1.0, point[1] - 2.0)
        == pytest.approx(5.0)
        for point in circle
    )


def test_sketch_preview_cancel_clears_pending_shape_and_redraws() -> None:
    _application()
    viewport = FEMViewport()
    redraws: list[bool] = []
    viewport._sketch_authoring_active = True
    viewport._show_sketch_authoring_hover = (
        lambda *, render: redraws.append(render)
    )
    viewport.set_sketch_pending_points(((0.0, 0.0, 0.0),))

    assert viewport.cancel_pending_sketch_interaction()
    assert viewport._sketch_pending_points == ()
    assert redraws == [True, True]
    viewport.close()


def test_empty_sketch_camera_fit_uses_compact_default_work_area() -> None:
    bounds = _sketch_camera_bounds((), 0.1)

    assert bounds[:4] == pytest.approx((-2.0, 2.0, -2.0, 2.0))
    assert bounds[4] < 0.0 < bounds[5]

    _application()
    viewport = FEMViewport()
    viewport._sketch_authoring_active = True
    viewport._sketch_grid_spacing = 0.1
    viewport._sketch_draft_render_data = SketchDraftRenderData(
        (),
        (),
        (),
        (),
    )
    assert viewport._fit_bounds() == pytest.approx(bounds)
    viewport.close()


def test_sketch_display_size_does_not_change_camera_fit() -> None:
    expected = _sketch_camera_bounds((), 0.1)
    _application()
    viewport = FEMViewport()
    viewport._sketch_authoring_active = True
    viewport._sketch_grid_spacing = 0.1
    viewport._sketch_display_size = 50.0
    viewport._sketch_draft_render_data = SketchDraftRenderData(
        (),
        (),
        (),
        (),
    )

    assert viewport._fit_bounds() == pytest.approx(expected)
    viewport.close()


def test_selected_drawing_face_expands_sketch_space_outward() -> None:
    _application()
    viewport = FEMViewport()
    viewport._sketch_authoring_active = True
    viewport._sketch_grid_spacing = 0.1
    viewport._sketch_draft_render_data = SketchDraftRenderData(
        (),
        (),
        (),
        (),
    )
    preview = GeometryPreview(
        (
            (10.0, 20.0, 0.0),
            (110.0, 20.0, 0.0),
            (110.0, 70.0, 0.0),
            (10.0, 70.0, 0.0),
        ),
        ((0, 1, 2, 3),),
        (),
        ("face:target",),
    )

    viewport.show_sketch_reference_preview(
        preview,
        support_face_id="face:target",
    )

    assert viewport._sketch_support_points == preview.points
    assert viewport._fit_bounds() == pytest.approx(
        (0.0, 120.0, 10.0, 80.0, -6.0e-5, 6.0e-5)
    )
    viewport.close()


def test_sketch_uv_overlay_uses_two_decimals_and_bold_black_text() -> None:
    _application()
    viewport = FEMViewport()
    viewport._sketch_authoring_active = True
    viewport._sketch_draft_render_data = SketchDraftRenderData(
        (),
        (),
        (),
        (),
    )

    viewport._update_sketch_uv_label((1.234, -2.346, 0.0))

    label = viewport._sketch_uv_label
    assert label.text() == "U = 1.23\nV = -2.35"
    assert label.font().bold()
    assert "color: #000000" in label.styleSheet()
    assert "background: rgba(255, 255, 255, 230)" in label.styleSheet()
    assert not label.isHidden()
    viewport._update_sketch_uv_label(None)
    assert label.isHidden()
    viewport.close()


def test_sketch_axes_remain_anchored_when_rectangle_is_away_from_origin() -> None:
    class GridLayout:
        center = (1.2, -1.0, 0.0)
        plane_size = 4.0

    u_start, u_end = _sketch_axis_local_endpoints(GridLayout(), 0)
    v_start, v_end = _sketch_axis_local_endpoints(GridLayout(), 1)

    assert u_start == pytest.approx((-0.8, 0.0))
    assert u_end == pytest.approx((3.2, 0.0))
    assert v_start == pytest.approx((0.0, -3.0))
    assert v_end == pytest.approx((0.0, 1.0))
    assert u_start[1] == u_end[1] == 0.0
    assert v_start[0] == v_end[0] == 0.0

    controller = SketchDraftController("offset-rectangle")
    panel = SketchEditorPanel(controller)
    panel.set_mode("rectangle")
    panel._point_from_viewport((1.2, -1.0, 0.0))
    panel._point_from_viewport((2.4, -0.2, 0.0))
    coordinates = {(point.u, point.v) for point in controller.snapshot().points}
    assert coordinates == {
        (1.2, -1.0),
        (2.4, -1.0),
        (2.4, -0.2),
        (1.2, -0.2),
    }
    points = controller.snapshot().points
    assert sum(point.u for point in points) / len(points) == pytest.approx(1.8)
    assert sum(point.v for point in points) / len(points) == pytest.approx(-0.6)


def test_panel_polyline_closes_a_profile_and_emits_finish() -> None:
    _application()
    controller = SketchDraftController("panel-sketch")
    panel = SketchEditorPanel(controller)
    panel.set_mode("polyline")

    for point in (
        (0.0, 0.0, 0.0),
        (4.0, 0.0, 0.0),
        (4.0, 2.0, 0.0),
        (0.0, 2.0, 0.0),
        (0.0, 0.0, 0.0),
    ):
        panel._point_from_viewport(point)

    assert len(controller.snapshot().points) == 4
    assert len(controller.snapshot().curves) == 4
    assert controller.can_finish
    assert panel._pending_points == [(0.0, 0.0)]
    assert panel._polyline_start_id == panel._polyline_first_id
    assert panel.mode == "polyline"
    render_data = panel.render_data()
    assert render_data.faces
    assert set(render_data.curve_ids) == {
        curve.id for curve in controller.snapshot().curves
    }
    assert set(render_data.face_ids) == {
        profile.id for profile in controller.profiles
    }

    finished: list[bool] = []
    panel.finishRequested.connect(lambda: finished.append(True))
    panel.try_finish()
    assert finished == [True]


def test_panel_keeps_invalid_open_draft_detached() -> None:
    _application()
    controller = SketchDraftController("open-sketch")
    panel = SketchEditorPanel(controller)
    panel._point_from_viewport((0.0, 0.0, 0.0))
    panel._point_from_viewport((1.0, 0.0, 0.0))

    finished: list[bool] = []
    panel.finishRequested.connect(lambda: finished.append(True))
    panel.try_finish()

    assert not controller.can_finish
    assert finished == []
    assert not panel.finish_button.isEnabled()
    assert panel.render_data().curves == ((0, 1),)


def test_many_diagnostics_stay_scrollable_and_cancel_remains_available() -> None:
    app = _application()
    controller = SketchDraftController("many-diagnostics")
    for index in range(18):
        start = controller.add_point(float(index), 0.0)
        end = controller.add_point(float(index), 1.0)
        controller.add_line(start.id, end.id)
    panel = SketchEditorPanel(controller)
    panel.resize(420, 650)
    panel.show()
    app.processEvents()

    cancelled: list[bool] = []
    panel.cancelRequested.connect(lambda: cancelled.append(True))

    assert panel.diagnostic_label.text().count("\n") > 10
    assert panel.diagnostic_scroll.isHidden()
    assert panel.cancel_button.isVisible()
    assert (
        panel.cancel_button.mapTo(
            panel,
            panel.cancel_button.rect().bottomLeft(),
        ).y()
        <= panel.rect().bottom()
    )
    panel.cancel_button.click()
    assert cancelled == [True]
    panel.close()


def test_editor_content_scrolls_without_expanding_the_window_minimum_height() -> None:
    app = _application()
    panel = SketchEditorPanel(SketchDraftController("scrollable-panel"))
    panel.resize(420, 400)
    panel.show()
    app.processEvents()

    assert panel.minimumSizeHint().height() < 300
    assert panel.editor_scroll.verticalScrollBar().maximum() > 0
    assert panel.editor_scroll.horizontalScrollBar().maximum() == 0
    assert panel.editor_scroll.verticalScrollBar().isVisible()
    scroll_style = panel.editor_scroll.verticalScrollBar().styleSheet()
    assert "background: transparent" in scroll_style
    assert "width: 10px" in scroll_style
    assert "border-radius: 4px" in scroll_style
    assert "agent_chat_scroll_up.svg" in scroll_style
    QApplication.sendEvent(panel.editor_scroll.viewport(), _wheel_event())
    app.processEvents()
    assert panel.editor_scroll.verticalScrollBar().value() > 0
    assert panel.cancel_button.isVisible()
    assert (
        panel.cancel_button.mapTo(
            panel,
            panel.cancel_button.rect().bottomLeft(),
        ).y()
        <= panel.rect().bottom()
    )
    panel.close()


def test_dimension_dialog_accepts_precise_input_and_blocks_wheel() -> None:
    _application()
    dialog = sketch_editor_panel_module._DimensionEditorDialog(
        "半径",
        2.0,
    )
    value = dialog.value_spin.value()

    QApplication.sendEvent(dialog.value_spin, _wheel_event())

    assert dialog.value_spin.decimals() == 12
    assert dialog.value_spin.singleStep() == 0.1
    assert dialog.value_spin.text() == "2.0"
    assert dialog.value_spin.value() == value
    dialog.value_spin.selectAll()
    QTest.keyClicks(dialog.value_spin, "0.25")
    QTest.keyClick(dialog.value_spin, Qt.Key.Key_Return)
    assert dialog.value_spin.value() == 0.25
    assert dialog.value_spin.text() == "0.25"
    assert not hasattr(dialog, "driving_check")
    assert "驱动尺寸" not in {label.text() for label in dialog.findChildren(QLabel)}
    dialog.close()


def test_sketch_numeric_editors_share_full_precision() -> None:
    _application()
    panel = SketchEditorPanel(SketchDraftController("precise-inputs"))
    fixed_dialog = sketch_editor_panel_module._FixedConstraintEditorDialog(
        0.123456789012,
        -0.123456789012,
        use_xy_labels=True,
    )

    editors = (
        fixed_dialog.u_spin,
        fixed_dialog.v_spin,
        panel.line_length_spin,
        panel.circle_radius_spin,
        panel.arc_radius_spin,
        panel.arc_start_angle_spin,
        panel.arc_end_angle_spin,
        panel.constraint_value_spin,
    )
    assert all(editor.decimals() == 12 for editor in editors)
    assert fixed_dialog.u_spin.value() == 0.123456789012
    assert fixed_dialog.v_spin.value() == -0.123456789012

    fixed_dialog.close()
    panel.close()


def test_event_filter_tolerates_non_wheel_event_during_panel_construction() -> None:
    _application()
    panel = SketchEditorPanel(SketchDraftController("construction-event"))
    constraint_type_combo = panel.constraint_type_combo
    constraint_value_spin = panel.constraint_value_spin

    del panel.constraint_type_combo
    del panel.constraint_value_spin
    try:
        handled = panel.eventFilter(
            panel.points_table,
            QEvent(QEvent.Type.Hide),
        )
    finally:
        panel.constraint_type_combo = constraint_type_combo
        panel.constraint_value_spin = constraint_value_spin

    assert handled is False
    panel.close()


def test_sketch_entry_refits_after_splitter_layout_settles(monkeypatch) -> None:
    app = _application()
    window = FEMMainWindow()
    window.show()
    window.resize(1000, 700)
    window._create_native_model("模型-1")
    app.processEvents()
    original_width = window.viewport.width()
    fitted_sizes = []
    monkeypatch.setattr(
        window.viewport,
        "fit",
        lambda: fitted_sizes.append(window.viewport.size()),
    )

    window._begin_sketch_editor(
        None,
        original_recipe=None,
        part_name="部件-1",
    )

    assert fitted_sizes == []
    app.processEvents()
    assert len(fitted_sizes) == 1
    assert fitted_sizes[0] == window.viewport.size()
    assert fitted_sizes[0].width() < original_width
    assert window.minimumSizeHint().height() < 600
    window._exit_sketch_editor()
    window.close()


def test_invalid_dirty_sketch_can_be_cancelled_from_panel(
    monkeypatch,
) -> None:
    _application()
    window = FEMMainWindow()
    window._create_native_model("模型-1")
    window._begin_sketch_editor(
        None,
        original_recipe=None,
        part_name="未完成部件",
    )
    controller = window._sketch_editor_controller
    assert controller is not None
    start = controller.add_point(0.0, 0.0)
    end = controller.add_point(1.0, 0.0)
    controller.add_line(start.id, end.id)
    window.sketch_editor_panel._refresh()
    monkeypatch.setattr(
        window,
        "_confirm_sketch_editor_discard",
        lambda: True,
    )

    window.sketch_editor_panel.cancel_button.click()

    assert window._sketch_editor_controller is None
    assert window.sketch_editor_panel.isHidden()
    assert not window.viewport.sketch_authoring_active
    window.close()


def test_trim_click_uses_unsnapped_work_plane_position() -> None:
    _application()
    viewport = FEMViewport()
    viewport._sketch_authoring_mode = "trim"
    viewport._sketch_grid_snap = True
    viewport._sketch_grid_spacing = 1.0
    viewport._display_to_world = (
        lambda _x, _y, depth: (0.24, 0.37, float(depth))
    )
    viewport._sketch_curve_at = lambda _x, _y: "L7"
    requests: list[tuple[str, tuple[float, float, float]]] = []
    viewport.sketchTrimRequested.connect(
        lambda curve_id, point: requests.append((curve_id, tuple(point)))
    )

    viewport._sketch_authoring_click(10, 20)

    assert requests == [("L7", pytest.approx((0.24, 0.37, 0.0)))]
    viewport.close()


def test_panel_keeps_valid_profile_fill_with_blocking_open_curve() -> None:
    _application()
    controller = SketchDraftController("partial-profile")
    controller.add_rectangle((0.0, 0.0), (4.0, 2.0))
    first = controller.add_point(6.0, 0.0)
    second = controller.add_point(7.0, 1.0)
    controller.add_line(first.id, second.id)
    panel = SketchEditorPanel(controller)

    assert controller.profiles
    assert not controller.can_finish
    assert not panel.finish_button.isEnabled()
    render_data = panel.render_data()
    assert render_data.faces
    assert set(render_data.face_ids) == {
        profile.id for profile in controller.profiles
    }
    assert len(render_data.curve_ids) == 5


def test_panel_edits_line_endpoints_and_exact_length() -> None:
    _application()
    controller = SketchDraftController("line-parameters")
    start = controller.add_point(0.0, 0.0)
    end = controller.add_point(3.0, 4.0)
    alternate = controller.add_point(0.0, 2.0)
    line = controller.add_line(start.id, end.id)
    panel = SketchEditorPanel(controller)
    panel._select_curve(line.id)

    assert panel.line_parameter_group.isHidden() is False
    assert panel.line_length_spin.value() == 5.0
    panel.line_length_spin.setValue(10.0)
    panel._line_length_changed()
    points = {point.id: point for point in controller.snapshot().points}
    assert points[end.id].u == 6.0
    assert points[end.id].v == 8.0

    panel.line_end_combo.setCurrentIndex(
        panel.line_end_combo.findData(alternate.id)
    )
    panel._line_endpoints_changed()
    updated = next(
        curve
        for curve in controller.snapshot().curves
        if curve.id == line.id
    )
    assert updated.end_point_id == alternate.id
    assert panel.line_length_spin.value() == 2.0


def test_panel_edits_circle_and_arc_parameters() -> None:
    _application()
    controller = SketchDraftController("curve-parameters")
    circle = controller.add_circle((0.0, 0.0), 2.0)
    arc = controller.add_arc((3.0, 0.0), (4.0, 1.0), (3.0, 2.0))
    panel = SketchEditorPanel(controller)

    panel._select_curve(circle.id)
    assert panel.circle_parameter_group.isHidden() is False
    panel.circle_radius_spin.setValue(3.5)
    panel._circle_radius_changed()
    updated_circle = next(
        curve
        for curve in controller.snapshot().curves
        if curve.id == circle.id
    )
    assert updated_circle.radius == 3.5

    panel._select_curve(arc.id)
    assert panel.arc_parameter_group.isHidden() is False
    before_radius = controller.snapshot()
    panel.arc_radius_spin.setValue(2.0)
    panel._arc_radius_changed()
    snapshot = controller.snapshot()
    assert snapshot.revision == before_radius.revision + 1
    points = {point.id: point for point in snapshot.points}
    updated_arc = next(
        curve for curve in snapshot.curves if curve.id == arc.id
    )
    center = points[updated_arc.center_point_id]
    start = points[updated_arc.start_point_id]
    end = points[updated_arc.end_point_id]
    assert math.hypot(start.u - center.u, start.v - center.v) == 2.0
    assert math.hypot(end.u - center.u, end.v - center.v) == 2.0
    controller.undo()
    assert controller.snapshot() == before_radius
    controller.redo()

    panel.arc_start_angle_spin.setValue(180.0)
    panel._arc_start_angle_changed()
    points = {
        point.id: point for point in controller.snapshot().points
    }
    start = points[updated_arc.start_point_id]
    assert start.u == pytest.approx(center.u - 2.0)
    assert start.v == pytest.approx(center.v)


def test_point_table_edits_by_stable_id_after_sorting() -> None:
    _application()
    controller = SketchDraftController("stable-rows")
    controller.add_point(1.0, 0.0, point_id="P10")
    controller.add_point(2.0, 0.0, point_id="P2")
    panel = SketchEditorPanel(controller)
    panel.points_table.setSortingEnabled(True)
    panel.points_table.sortItems(0, Qt.SortOrder.DescendingOrder)
    target_row = next(
        row
        for row in range(panel.points_table.rowCount())
        if panel.points_table.item(row, 0).data(Qt.ItemDataRole.UserRole) == "P10"
    )

    panel.points_table.item(target_row, 1).setText("9.0")

    points = {point.id: point for point in controller.snapshot().points}
    assert points["P10"].u == 9.0
    assert points["P2"].u == 2.0


def test_phase2_point_search_filter_sort_edit_and_delete_stay_id_based() -> None:
    _application()
    controller = SketchDraftController("phase-2-list")
    controller.add_point(10.0, 0.0, point_id="P10")
    controller.add_point(2.0, 0.0, point_id="P2")
    controller.add_point(
        1.0,
        0.0,
        point_id="P1",
        external_reference=_reference_point(1.0, 0.0),
    )
    panel = SketchEditorPanel(controller)

    assert [panel.points_table.item(row, 0).text() for row in range(3)] == [
        "P1",
        "P2",
        "P10",
    ]
    panel.point_filter_combo.setCurrentIndex(
        panel.point_filter_combo.findData("free")
    )
    assert set(panel._visible_point_ids()) == {"P2", "P10"}
    panel.point_search_edit.setText("10")
    assert panel._visible_point_ids() == ("P10",)

    panel.points_table.item(0, 1).setText("19.0")
    assert {point.id: point.u for point in controller.snapshot().points}["P10"] == 19.0
    panel.points_table.selectRow(0)
    panel.delete_selected()
    assert {point.id for point in controller.snapshot().points} == {"P1", "P2"}

    panel.point_search_edit.clear()
    panel.point_filter_combo.setCurrentIndex(
        panel.point_filter_combo.findData("associated")
    )
    assert panel._visible_point_ids() == ("P1",)
    controller.refresh_external_references(())
    panel.point_filter_combo.setCurrentIndex(
        panel.point_filter_combo.findData("unresolved")
    )
    assert panel._visible_point_ids() == ("P1",)


def test_phase2_viewport_and_table_multi_selection_and_clear_are_symmetric() -> None:
    _application()
    controller = SketchDraftController("phase-2-selection")
    for point_id in ("P10", "P2", "P1"):
        controller.add_point(float(len(controller.snapshot().points)), 0.0, point_id=point_id)
    panel = SketchEditorPanel(controller)

    class _Viewport:
        def __init__(self) -> None:
            self.selections = []

        def update_sketch_selection(self, kind, ids) -> None:
            self.selections.append((kind, tuple(ids)))

    viewport = _Viewport()
    panel._viewport = viewport
    panel._select_point("P1")
    panel._select_point("P10", Qt.KeyboardModifier.ShiftModifier)

    assert controller.selected_ids == ("P1", "P2", "P10")
    assert set(
        panel._point_id_for_row(index.row())
        for index in panel.points_table.selectionModel().selectedRows()
    ) == {"P1", "P2", "P10"}
    assert viewport.selections[-1] == ("point", ("P1", "P2", "P10"))

    panel.points_table.clearSelection()
    assert controller.selected_ids == ()
    assert viewport.selections[-1] == (None, ())
    row = panel._row_for_point_id("P2")
    panel.points_table.selectionModel().select(
        panel.points_table.model().index(row, 0),
        QItemSelectionModel.SelectionFlag.Select
        | QItemSelectionModel.SelectionFlag.Rows,
    )
    assert controller.selected_ids == ("P2",)
    assert viewport.selections[-1] == ("point", ("P2",))


def test_phase2_selection_is_lightweight_and_does_not_analyze_or_rebuild(
    monkeypatch,
) -> None:
    _application()
    controller = SketchDraftController("phase-2-lightweight")
    controller.add_point(0.0, 0.0, point_id="P1")
    panel = SketchEditorPanel(controller)

    class _Viewport:
        def __init__(self) -> None:
            self.light_updates = []

        def update_sketch_selection(self, kind, ids) -> None:
            self.light_updates.append((kind, tuple(ids)))

        def update_sketch_draft(self, _data) -> None:
            raise AssertionError("selection rebuilt the full sketch preview")

    panel._viewport = _Viewport()
    monkeypatch.setattr(
        controller,
        "derive_profiles",
        lambda: (_ for _ in ()).throw(
            AssertionError("selection triggered profile analysis")
        ),
    )

    panel._select_point("P1")
    panel._authoring_missed("select")

    assert panel._viewport.light_updates == [("point", ("P1",)), (None, ())]
    assert controller.snapshot().revision == 1


def test_phase2_refresh_preserves_selection_scroll_and_edit_cell() -> None:
    app = _application()
    controller = SketchDraftController("phase-2-refresh")
    for index in range(40):
        controller.add_point(float(index), 0.0, point_id=f"P{index + 1}")
    panel = SketchEditorPanel(controller)
    panel.resize(360, 300)
    panel.show()
    app.processEvents()
    target_id = "P30"
    row = panel._row_for_point_id(target_id)
    target = panel.points_table.item(row, 1)
    panel.points_table.selectRow(row)
    panel.points_table.setCurrentItem(
        target,
        QItemSelectionModel.SelectionFlag.NoUpdate,
    )
    panel.points_table.verticalScrollBar().setValue(12)
    scroll = panel.points_table.verticalScrollBar().value()
    panel.points_table.editItem(target)
    app.processEvents()

    panel._refresh()
    app.processEvents()

    assert panel._point_id_for_row(panel.points_table.currentRow()) == target_id
    assert panel.points_table.currentColumn() == 1
    assert target_id in controller.selected_ids
    assert panel.points_table.verticalScrollBar().value() == scroll
    assert panel.points_table.state().name == "EditingState"
    panel.close()


def test_phase2_double_click_requests_point_focus() -> None:
    _application()
    controller = SketchDraftController("phase-2-focus")
    controller.add_point(0.0, 0.0, point_id="P7")
    panel = SketchEditorPanel(controller)
    focused = []
    panel.entityFocusRequested.connect(lambda kind, entity_id: focused.append((kind, entity_id)))

    panel._focus_point_item(panel.points_table.item(0, 0))

    assert focused == [("point", "P7")]


def test_phase2_viewport_selection_update_only_rebuilds_highlight(monkeypatch) -> None:
    _application()
    viewport = FEMViewport()
    viewport._sketch_authoring_active = True
    viewport._sketch_draft_render_data = SketchDraftRenderData(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
        ("P1", "P2"),
        (),
        (),
    )
    highlight_updates = []
    monkeypatch.setattr(
        viewport,
        "_show_sketch_selection",
        lambda *, render: highlight_updates.append(render),
    )
    monkeypatch.setattr(
        viewport,
        "_show_sketch_draft",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("selection rebuilt the complete viewport draft")
        ),
    )

    viewport.update_sketch_selection("point", ("P1", "P2"))
    data = viewport._sketch_draft_render_data

    assert data.selected_ids == ("P1", "P2")
    assert data.selected_id == "P1"
    assert highlight_updates == [True]
    viewport.close()


def test_phase2_point_editor_enter_commits_escape_cancels_and_idle_keys_are_safe() -> None:
    app = _application()
    controller = SketchDraftController("phase-2-edit-keys")
    controller.add_point(1.0, 2.0, point_id="P1")
    panel = SketchEditorPanel(controller)
    panel.resize(420, 900)
    panel.show()
    app.processEvents()
    item = panel.points_table.item(0, 1)

    panel.points_table.setCurrentItem(item)
    panel.points_table.editItem(item)
    app.processEvents()
    editor = panel.points_table.findChild(QLineEdit)
    assert editor is not None
    editor.selectAll()
    QTest.keyClicks(editor, "4.5")
    QTest.keyClick(editor, Qt.Key.Key_Return)
    app.processEvents()
    assert controller.snapshot().points[0].u == 4.5

    item = panel.points_table.item(0, 1)
    panel.points_table.setCurrentItem(item)
    panel.points_table.editItem(item)
    app.processEvents()
    editor = panel.points_table.findChild(QLineEdit)
    assert editor is not None
    editor.selectAll()
    QTest.keyClicks(editor, "9.0")
    QTest.keyClick(editor, Qt.Key.Key_Escape)
    app.processEvents()
    assert controller.snapshot().points[0].u == 4.5

    finished = []
    cancelled = []
    panel.finishRequested.connect(lambda: finished.append(True))
    panel.cancelRequested.connect(lambda: cancelled.append(True))
    panel.points_table.setFocus()
    QTest.keyClick(panel.points_table, Qt.Key.Key_Return)
    QTest.keyClick(panel.points_table, Qt.Key.Key_Escape)
    app.processEvents()
    assert finished == []
    assert cancelled == []
    panel.close()


def test_associated_point_status_is_read_only_until_released() -> None:
    _application()
    controller = SketchDraftController("associations")
    point = controller.add_point(
        1.0,
        2.0,
        external_reference=_reference_point(1.0, 2.0),
    )
    controller.select_point(point.id)
    panel = SketchEditorPanel(controller)

    assert panel.points_table.item(0, 3).text() == "已关联"
    assert not (
        panel.points_table.item(0, 1).flags() & Qt.ItemFlag.ItemIsEditable
    )
    controller.refresh_external_references(())
    panel._refresh()
    assert panel.points_table.item(0, 3).text() == "未解析"

    panel.release_selected_association()

    assert panel.points_table.item(0, 3).text() == "自由"
    assert panel.points_table.item(0, 1).flags() & Qt.ItemFlag.ItemIsEditable


def test_point_delete_prompt_lists_cascade_and_undo_restores_entities(
    monkeypatch,
) -> None:
    _application()
    controller = SketchDraftController("delete-cascade")
    first = controller.add_point(0.0, 0.0)
    shared = controller.add_point(1.0, 0.0)
    last = controller.add_point(2.0, 0.0)
    first_line = controller.add_line(first.id, shared.id)
    second_line = controller.add_line(shared.id, last.id)
    controller.select_point(shared.id)
    panel = SketchEditorPanel(controller)
    prompts: list[str] = []

    def confirm(_parent, _title, message, *_args):
        prompts.append(message)
        return sketch_editor_panel_module.QMessageBox.StandardButton.Yes

    monkeypatch.setattr(
        sketch_editor_panel_module.QMessageBox,
        "question",
        confirm,
    )
    panel.delete_selected()

    assert first_line.id in prompts[0]
    assert second_line.id in prompts[0]
    assert shared.id not in {point.id for point in controller.snapshot().points}
    assert controller.snapshot().curves == ()
    controller.undo()
    assert {point.id for point in controller.snapshot().points} >= {shared.id}
    assert {curve.id for curve in controller.snapshot().curves} >= {
        first_line.id,
        second_line.id,
    }


def test_only_grid_preferences_are_user_configurable(tmp_path) -> None:
    _application()
    path = tmp_path / "sketch.ini"
    store = QSettings(str(path), QSettings.Format.IniFormat)
    store.setValue("sketch/grid_visible", False)
    store.setValue("sketch/grid_spacing", 0.25)
    store.setValue("sketch/grid_snap", False)
    store.setValue("sketch/snap_midpoints", False)
    store.setValue("sketch/screen_snap_tolerance", 13.0)
    store.setValue("sketch/auto_merge_tolerance", 0.02)
    store.setValue("sketch/show_point_ids", False)
    store.setValue("sketch/show_external_labels", False)
    store.setValue("sketch/show_profile_fill", False)
    store.setValue("sketch/show_work_plane_axes", False)
    store.setValue("sketch/continuous_polyline", False)
    store.setValue("sketch/end_polyline_on_close", True)
    store.setValue("sketch/keep_tool_after_completion", False)
    store.setValue("sketch/confirm_cascade_delete", False)
    store.setValue("sketch/auto_constraints", False)
    controller = SketchDraftController("phase-3-preferences")
    point = controller.add_point(0.0, 0.0)
    controller.select_point(point.id)
    panel = SketchEditorPanel(controller, settings=store)
    revision = controller.snapshot().revision
    can_undo = controller.can_undo

    panel.grid_visible_check.setChecked(True)
    panel.grid_visible_check.setChecked(False)
    panel.set_mode("circle")
    panel.point_filter_combo.setCurrentIndex(
        panel.point_filter_combo.findData("free")
    )
    store.sync()

    assert controller.snapshot().revision == revision
    assert controller.can_undo == can_undo
    assert controller.selected_ids == (point.id,)
    restored = load_sketch_preferences(
        QSettings(str(path), QSettings.Format.IniFormat)
    )
    assert not restored.grid_visible
    assert restored.grid_snap
    assert restored.grid_spacing == 0.25
    assert restored.snap_midpoints
    assert restored.screen_snap_tolerance == 9.0
    assert restored.auto_merge_tolerance == 1.0e-6
    assert restored.show_point_ids
    assert restored.show_external_labels
    assert restored.show_profile_fill
    assert restored.show_work_plane_axes
    assert restored.continuous_polyline
    assert not restored.end_polyline_on_close
    assert restored.keep_tool_after_completion
    assert restored.confirm_cascade_delete
    assert restored.auto_constraints
    controller.undo()
    assert controller.snapshot().points == ()

    reopened = SketchEditorPanel(
        SketchDraftController("phase-3-reopened"),
        settings=QSettings(str(path), QSettings.Format.IniFormat),
    )
    assert not reopened.grid_visible_check.isChecked()
    assert reopened._preferences.grid_snap
    assert reopened.mode == "polyline"
    assert reopened.point_filter_combo.currentData() == "all"
    assert reopened.controller.selected_ids == ()


def test_phase3_grid_display_and_snap_are_independent() -> None:
    _application()
    viewport = FEMViewport()
    viewport._display_to_world = (
        lambda _x, _y, depth: (0.24, 0.24, float(depth))
    )
    viewport._world_points_to_display = lambda points: points
    viewport._device_pixel_ratio = lambda: 1.0

    viewport.set_sketch_grid(visible=False, snap=True, spacing=0.25)
    assert not viewport._sketch_grid_visible
    assert viewport._sketch_grid_snap
    assert viewport._sketch_grid_spacing == 0.25
    point, _reason = viewport._sketch_work_plane_point_at(0, 0)
    assert point == (0.25, 0.25, 0.0)
    assert viewport._sketch_authoring_snap_kind == "grid"

    viewport.set_sketch_grid(visible=True, snap=False)
    assert viewport._sketch_grid_visible
    assert not viewport._sketch_grid_snap
    assert viewport._sketch_grid_spacing == 0.25
    point, _reason = viewport._sketch_work_plane_point_at(0, 0)
    assert point == (0.24, 0.24, 0.0)
    assert viewport._sketch_authoring_snap_kind is None
    viewport.close()


def test_phase3_snap_categories_tolerance_priority_intersections_and_feedback() -> None:
    _application()
    viewport = FEMViewport()
    viewport._display_to_world = (
        lambda x, y, depth: (float(x), float(y), float(depth))
    )
    viewport._world_points_to_display = lambda points: points
    viewport._device_pixel_ratio = lambda: 1.0
    viewport._sketch_grid_snap = False
    viewport._sketch_draft_render_data = SketchDraftRenderData(
        ((0.0, 0.0, 0.0),),
        ("P1",),
        (),
        (),
    )
    viewport._sketch_reference_points = (_reference_point(0.0, 0.0),)

    viewport.set_sketch_preferences(
        snap_sketch_points=True,
        snap_external_points=True,
        snap_midpoints=False,
        snap_centers=False,
        snap_intersections=False,
        screen_snap_tolerance=9.0,
        show_point_ids=True,
        show_external_labels=True,
        show_profile_fill=True,
        show_work_plane_axes=True,
    )
    point, reason = viewport._sketch_work_plane_point_at(0, 0)
    assert reason is None
    assert point == (0.0, 0.0, 0.0)
    assert viewport._sketch_authoring_snap_kind == "sketch_point"

    viewport._sketch_snap_sketch_points = False
    viewport._sketch_work_plane_point_at(0, 0)
    assert viewport._sketch_authoring_snap_kind == "topology_vertex"
    viewport._sketch_snap_external_points = False
    viewport._sketch_work_plane_point_at(0, 0)
    assert viewport._sketch_authoring_snap_kind is None

    def derived_reference(
        reference_type: SketchExternalReferenceType,
    ) -> SketchReferencePoint:
        return SketchReferencePoint(
            SketchExternalReference(
                f"R-{reference_type.value}",
                LogicalEntityRef("edge:support/E1"),
                reference_type,
            ),
            (0.0, 0.0, 0.0),
            0.0,
            0.0,
        )

    viewport._sketch_reference_points = (
        derived_reference(SketchExternalReferenceType.LINE_MIDPOINT),
    )
    viewport._sketch_snap_midpoints = False
    viewport._sketch_work_plane_point_at(0, 0)
    assert viewport._sketch_authoring_snap_kind is None
    viewport._sketch_snap_midpoints = True
    viewport._sketch_work_plane_point_at(0, 0)
    assert viewport._sketch_authoring_snap_kind == "line_midpoint"

    viewport._sketch_reference_points = (
        derived_reference(SketchExternalReferenceType.CIRCLE_CENTER),
    )
    viewport._sketch_snap_midpoints = False
    viewport._sketch_snap_centers = False
    viewport._sketch_work_plane_point_at(0, 0)
    assert viewport._sketch_authoring_snap_kind is None
    viewport._sketch_snap_centers = True
    viewport._sketch_work_plane_point_at(0, 0)
    assert viewport._sketch_authoring_snap_kind == "circle_center"

    viewport._sketch_reference_points = ()
    viewport._sketch_draft_render_data = SketchDraftRenderData(
        (),
        (),
        (),
        (),
        snap_midpoints=((0.0, 0.0, 0.0),),
        snap_centers=((2.0, 0.0, 0.0),),
    )
    viewport._sketch_snap_midpoints = False
    viewport._sketch_snap_centers = False
    viewport._sketch_work_plane_point_at(0, 0)
    assert viewport._sketch_authoring_snap_kind is None
    viewport._sketch_snap_midpoints = True
    viewport._sketch_work_plane_point_at(0, 0)
    assert viewport._sketch_authoring_snap_kind == "line_midpoint"
    viewport._sketch_snap_midpoints = False
    viewport._sketch_snap_centers = True
    viewport._sketch_work_plane_point_at(2, 0)
    assert viewport._sketch_authoring_snap_kind == "circle_center"

    viewport._sketch_snap_sketch_points = True
    viewport._sketch_draft_render_data = SketchDraftRenderData(
        ((5.0, 0.0, 0.0),),
        ("P5",),
        (),
        (),
    )
    viewport._sketch_screen_snap_tolerance = 4.0
    point, _reason = viewport._sketch_work_plane_point_at(0, 0)
    assert point == (0.0, 0.0, 0.0)
    assert viewport._sketch_authoring_snap_kind is None
    viewport._sketch_screen_snap_tolerance = 6.0
    point, _reason = viewport._sketch_work_plane_point_at(0, 0)
    assert point == (5.0, 0.0, 0.0)
    assert viewport._sketch_authoring_snap_kind == "sketch_point"

    crossings = SketchDraftRenderData(
        (
            (-1.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, -1.0, 0.0),
            (0.0, 1.0, 0.0),
        ),
        (None, None, None, None),
        ((0, 1), (2, 3)),
        ("L1", "L2"),
    )
    assert _sketch_intersection_points(crossings) == ((0.0, 0.0, 0.0),)
    viewport._sketch_draft_render_data = crossings
    viewport._sketch_snap_sketch_points = False
    viewport._sketch_snap_intersections = False
    viewport._sketch_work_plane_point_at(0, 0)
    assert viewport._sketch_authoring_snap_kind is None
    viewport._sketch_snap_intersections = True
    point, _reason = viewport._sketch_work_plane_point_at(0, 0)
    assert point == (0.0, 0.0, 0.0)
    assert viewport._sketch_authoring_snap_kind == "intersection"

    assert _sketch_snap_label("sketch_point") == "草图点"
    assert _sketch_snap_label("topology_vertex") == "外部参考点"
    assert _sketch_snap_label("grid") == "网格点"
    assert _sketch_snap_label("line_midpoint") == "中点"
    assert _sketch_snap_label("circle_center") == "圆心"
    assert _sketch_snap_label("intersection") == "交点"
    viewport.close()


def test_intersection_candidates_use_analytic_curves_not_display_polylines() -> None:
    data = SketchDraftRenderData(
        (
            (-2.0, 0.5, 0.0),
            (2.0, 0.5, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (-1.0, 0.0, 0.0),
            (0.0, -1.0, 0.0),
        ),
        ("P0", "P1", None, None, None, None),
        ((0, 1), (2, 3, 4, 5, 2)),
        ("L", "C"),
        geometry_revision=7,
        analytic_points=(
            SketchPoint("P0", -2.0, 0.5),
            SketchPoint("P1", 2.0, 0.5),
            SketchPoint("O", 0.0, 0.0),
        ),
        analytic_curves=(
            SketchLine("L", "P0", "P1"),
            SketchCircle("C", "O", 1.0),
        ),
    )

    intersections = _sketch_intersection_points(data)
    assert intersections[0] == pytest.approx((-math.sqrt(3.0) / 2.0, 0.5, 0.0))
    assert intersections[1] == pytest.approx((math.sqrt(3.0) / 2.0, 0.5, 0.0))


def test_curved_display_sampling_tracks_screen_chord_error() -> None:
    coarse = _sketch_curve_sample_count(10.0, math.tau, 0.1)
    zoomed = _sketch_curve_sample_count(10.0, math.tau, 0.01)

    assert zoomed > coarse
    for count, world_per_pixel in ((coarse, 0.1), (zoomed, 0.01)):
        chord_error_pixels = (
            10.0 * (1.0 - math.cos(math.pi / count)) / world_per_pixel
        )
        assert chord_error_pixels <= 0.75


def test_fixed_auto_merge_and_drawing_behaviors() -> None:
    _application()
    controller = SketchDraftController("phase-3-behaviors")
    panel = SketchEditorPanel(controller)
    existing = controller.add_point(0.0, 0.0)
    assert panel._point_id_at(0.5e-6, 0.0) == existing.id
    assert panel._point_id_at(2.0e-6, 0.0) is None

    panel._point_from_viewport((0.0, 0.0, 0.0))
    panel._point_from_viewport((1.0, 0.0, 0.0))
    assert len(controller.snapshot().curves) == 1
    assert panel._pending_points == [(1.0, 0.0)]
    assert panel._polyline_start_id is not None
    assert panel.mode == "polyline"

    rectangle_controller = SketchDraftController("phase-3-tool")
    rectangle_panel = SketchEditorPanel(rectangle_controller)
    rectangle_panel.set_mode("rectangle")
    rectangle_panel._point_from_viewport((0.0, 0.0, 0.0))
    rectangle_panel._point_from_viewport((2.0, 1.0, 0.0))
    assert rectangle_panel.mode == "rectangle"

    close_controller = SketchDraftController("phase-3-close")
    close_panel = SketchEditorPanel(close_controller)
    for point in (
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (1.0, 1.0, 0.0),
        (0.0, 0.0, 0.0),
    ):
        close_panel._point_from_viewport(point)
    assert close_panel.mode == "polyline"
    assert close_panel._polyline_start_id == close_panel._polyline_first_id
    assert close_panel._pending_points == [(0.0, 0.0)]


def test_main_window_commits_strict_sketch_only_on_finish(monkeypatch) -> None:
    app = _application()
    window = FEMMainWindow()
    window._create_native_model("模型-1")
    prompts = []

    def get_text(_parent, title, prompt, **options):
        prompts.append((title, prompt, options.get("text")))
        return "支架部件", True

    monkeypatch.setattr(
        main_window_module.QInputDialog,
        "getText",
        get_text,
    )

    window.start_sketch_geometry()
    controller = window._sketch_editor_controller
    assert controller is not None
    assert prompts == [("新建二维草图", "部件名称：", "部件-1")]
    assert window.document.geometry_recipe is None
    assert window.document.parts == ()
    assert window.sketch_editor_panel.isHidden() is False
    assert window.viewport.sketch_authoring_active

    controller.add_rectangle((0.0, 0.0), (4.0, 2.0))
    preview_renders = []
    fit_calls = []
    original_preview = window.viewport.show_geometry_preview

    def counted_preview(preview, **kwargs):
        preview_renders.append(kwargs.get("render", True))
        return original_preview(preview, **kwargs)

    monkeypatch.setattr(
        window.viewport,
        "show_geometry_preview",
        counted_preview,
    )
    monkeypatch.setattr(
        window.viewport,
        "fit",
        lambda: fit_calls.append(True),
    )
    window.finish_sketch_geometry()

    recipe = window.document.geometry_recipe
    assert isinstance(recipe, SketchGeometry)
    assert recipe.is_strict
    assert window.document.parts[0].name == "支架部件"
    active_document_id = window.workspace.active_document_id
    assert active_document_id is not None
    assert (
        window.model_tree.roots[active_document_id].child(0).text(0)
        == "支架部件"
    )
    assert window._sketch_editor_controller is None
    assert window.sketch_editor_panel.isHidden()
    assert not window.viewport.sketch_authoring_active
    assert preview_renders == [False]
    assert fit_calls == []

    app.processEvents()

    assert fit_calls == [True]

    window.close_model(confirm=False)
    window.close()


def test_new_sketch_appends_part_without_replacing_existing(
    monkeypatch,
) -> None:
    _application()
    committed = SketchDraftController("committed")
    committed.add_rectangle((0.0, 0.0), (2.0, 1.0))
    original = committed.to_sketch_geometry()
    window = FEMMainWindow()
    window._apply_session_delta(window.session.new_native_project())
    window._set_native_geometry(original, "测试")
    monkeypatch.setattr(
        main_window_module.QInputDialog,
        "getText",
        lambda *_args, **_options: ("Part-2", True),
    )

    window.start_sketch_geometry()

    controller = window._sketch_editor_controller
    assert controller is not None
    controller.add_rectangle((0.0, 0.0), (4.0, 2.0))
    window.finish_sketch_geometry()

    assert tuple(part.name for part in window.document.parts) == (
        "部件-1",
        "Part-2",
    )
    assert window.document.parts[0].geometry_recipe == original
    assert window.document.parts[1].geometry_recipe != original
    assert window._sketch_editor_controller is None
    window.close_model(confirm=False)
    window.close()


def test_edit_root_commits_to_active_part() -> None:
    _application()
    committed = SketchDraftController("editable")
    committed.add_rectangle((0.0, 0.0), (4.0, 2.0))
    original = committed.to_sketch_geometry()
    window = FEMMainWindow()
    window._apply_session_delta(window.session.new_native_project())
    window._set_native_geometry(original, "测试")

    window.show_geometry_manager()

    controller = window._sketch_editor_controller
    assert controller is not None
    controller.add_circle((2.0, 1.0), 0.25)
    window.finish_sketch_geometry()

    assert len(window.document.parts) == 1
    assert window.document.geometry_recipe != original
    assert window._sketch_editor_controller is None
    window.close_model(confirm=False)
    window.close()


def test_unified_create_command_routes_2d_to_sketch_editor(
    monkeypatch,
) -> None:
    _application()

    class _CreationDialog:
        def __init__(self, _parent, *, default_part_name) -> None:
            assert default_part_name == "部件-1"

        def exec(self) -> bool:
            return True

        def creation_kind(self) -> str:
            return "2d"

        def part_name(self) -> str:
            return "Part-1"

        def sketch_size(self) -> float:
            return 75.0

    monkeypatch.setattr(
        main_window_module,
        "GeometryCreationDialog",
        _CreationDialog,
    )
    window = FEMMainWindow()
    window._apply_session_delta(window.session.new_native_project())

    window.create_geometry()

    assert window._sketch_editor_controller is not None
    assert window._sketch_editor_part_name == "Part-1"
    assert window.viewport._sketch_display_size == 75.0
    assert window._wire_editor_controller is None
    window._exit_sketch_editor()
    window.close_model(confirm=False)
    window.close()


def test_unified_create_command_routes_size_to_wire_editor(
    monkeypatch,
) -> None:
    _application()

    class _CreationDialog:
        def __init__(self, _parent, *, default_part_name) -> None:
            assert default_part_name == "部件-1"

        def exec(self) -> bool:
            return True

        def creation_kind(self) -> str:
            return "1d"

        def part_name(self) -> str:
            return "Wire-1"

        def sketch_size(self) -> float:
            return 60.0

    monkeypatch.setattr(
        main_window_module,
        "GeometryCreationDialog",
        _CreationDialog,
    )
    window = FEMMainWindow()
    window._apply_session_delta(window.session.new_native_project())

    window.create_geometry()

    assert window._wire_editor_controller is not None
    assert window._wire_editor_part_name == "Wire-1"
    assert window.viewport._wire_display_size == 60.0
    assert window._sketch_editor_controller is None
    window._exit_wire_editor()
    window.close_model(confirm=False)
    window.close()


def test_new_sketch_name_dialog_cancel_keeps_part_and_editor_unchanged(
    monkeypatch,
) -> None:
    _application()
    window = FEMMainWindow()
    window._create_native_model("模型-1")
    revision = window.document.session_revision
    monkeypatch.setattr(
        main_window_module.QInputDialog,
        "getText",
        lambda *_args, **_options: ("忽略", False),
    )

    window.start_sketch_geometry()

    assert window._sketch_editor_controller is None
    assert window.document.session_revision == revision
    assert window.document.parts == ()
    window.close_model(confirm=False)
    window.close()


def test_unified_create_command_opens_separate_3d_solid_chooser(
    monkeypatch,
) -> None:
    _application()
    events: list[str] = []

    class _CreationDialog:
        def __init__(self, _parent, *, default_part_name) -> None:
            events.append("dimension")
            assert default_part_name == "部件-1"

        def exec(self) -> bool:
            return True

        def creation_kind(self) -> str:
            return "3d"

        def part_name(self) -> str:
            return "圆柱部件"

    class _SolidDialog:
        def __init__(self, _parent) -> None:
            events.append("solid")

        def exec(self) -> bool:
            return True

        def solid_kind(self) -> str:
            return "cylinder"

    monkeypatch.setattr(
        main_window_module,
        "GeometryCreationDialog",
        _CreationDialog,
    )
    monkeypatch.setattr(
        main_window_module,
        "BasicSolidCreationDialog",
        _SolidDialog,
    )
    window = FEMMainWindow()
    window._apply_session_delta(window.session.new_native_project())
    window._create_basic_solid_part = (
        lambda kind, name: events.append(f"{kind}:{name}")
    )

    window.create_geometry()

    assert events == [
        "dimension",
        "solid",
        "3d_cylinder:圆柱部件",
    ]
    window.close_model(confirm=False)
    window.close()


def test_strict_sketch_cut_enters_detached_planar_boolean_workflow() -> None:
    _application()
    draft = SketchDraftController("cut-sketch")
    draft.add_rectangle((0.0, 0.0), (4.0, 2.0))
    recipe = draft.to_sketch_geometry()
    window = FEMMainWindow()
    window._apply_session_delta(window.session.new_native_project())
    window._set_native_geometry(recipe, "测试")

    window.cut_geometry()

    assert window._sketch_editor_controller is None
    assert window._planar_boolean_controller is not None
    assert window._planar_boolean_controller.geometry == recipe
    assert window.document.geometry_recipe == recipe
    assert window.planar_boolean_panel.isHidden()
    assert window.viewport_panel._active_bottom_overlay is (
        window.viewport_panel.planar_boolean_face_bar
    )
    window.cancel_planar_boolean()
    assert window.document.geometry_recipe == recipe
    window.close_model(confirm=False)
    window.close()
