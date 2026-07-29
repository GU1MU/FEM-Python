from __future__ import annotations

import os
import math

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication, QLabel, QListWidget

from fem.geometry import SketchGeometry
import fem_gui.main_window as main_window_module
from fem_gui.main_window import FEMMainWindow
from fem_gui.sketch_editor import SketchDraftController
from fem_gui.widgets.sketch_editor_panel import SketchEditorPanel
from fem_gui.widgets.viewport import (
    FEMViewport,
    SketchDraftRenderData,
    _sketch_camera_bounds,
    _sketch_shape_preview_points,
)


def _application() -> QApplication:
    return QApplication.instance() or QApplication([])


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
    assert panel._pending_points == []
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
    assert panel.diagnostic_scroll.height() <= 150
    assert panel.cancel_button.isVisible()
    assert panel.cancel_button.geometry().bottom() <= panel.rect().bottom()
    panel.cancel_button.click()
    assert cancelled == [True]
    panel.close()


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
    panel.arc_radius_spin.setValue(2.0)
    panel._arc_radius_changed()
    snapshot = controller.snapshot()
    points = {point.id: point for point in snapshot.points}
    updated_arc = next(
        curve for curve in snapshot.curves if curve.id == arc.id
    )
    center = points[updated_arc.center_point_id]
    start = points[updated_arc.start_point_id]
    end = points[updated_arc.end_point_id]
    assert math.hypot(start.u - center.u, start.v - center.v) == 2.0
    assert math.hypot(end.u - center.u, end.v - center.v) == 2.0

    panel.arc_start_angle_spin.setValue(180.0)
    panel._arc_start_angle_changed()
    points = {
        point.id: point for point in controller.snapshot().points
    }
    start = points[updated_arc.start_point_id]
    assert start.u == pytest.approx(center.u - 2.0)
    assert start.v == pytest.approx(center.v)


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
    assert (
        window.model_tree.topLevelItem(0).child(0).text(0)
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
    assert window._wire_editor_controller is None
    window._exit_sketch_editor()
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
    assert not window.planar_boolean_panel.isHidden()
    window.cancel_planar_boolean()
    assert window.document.geometry_recipe == recipe
    window.close_model(confirm=False)
    window.close()
