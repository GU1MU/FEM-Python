from __future__ import annotations

import os
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QLabel

from fem.application import StrictPlanarBooleanResult
from fem.geometry import (
    LogicalEntityRef,
    RectangleGeometry,
    SketchArc,
    SketchGeometry,
    SketchLine,
    SketchPlane,
    SketchPoint,
    WireGeometry,
    WireMember,
    WirePoint,
)
from fem_gui.geometry_preview import (
    GeometryPreview,
    build_geometry_preview,
    namespace_part_geometry_preview,
)
import fem_gui.main_window as main_window_module
from fem_gui.main_window import FEMMainWindow
from fem_gui.planar_boolean import PlanarBooleanController
from fem_gui.task_controller import TaskApplyStatus
from fem_gui.widgets.planar_boolean_panel import PlanarBooleanPanel


def _application() -> QApplication:
    return QApplication.instance() or QApplication([])


def _tool_sketch() -> SketchGeometry:
    return SketchGeometry(
        "Tool",
        SketchPlane.xy(),
        (
            SketchPoint("P1", 0.5, 0.5),
            SketchPoint("P2", 1.5, 0.5),
            SketchPoint("P3", 1.5, 1.5),
            SketchPoint("P4", 0.5, 1.5),
        ),
        (
            SketchLine("L1", "P1", "P2"),
            SketchLine("L2", "P2", "P3"),
            SketchLine("L3", "P3", "P4"),
            SketchLine("L4", "P4", "P1"),
        ),
    )


def test_controller_keeps_target_and_tool_state_detached() -> None:
    source = RectangleGeometry("Target", 3.0, 2.0)
    controller = PlanarBooleanController(source, 4, "cut")

    controller.request_target_selection()
    controller.assign_reference(LogicalEntityRef("face:domain"))
    controller.set_tool_recipe(_tool_sketch())

    assert controller.ready
    assert controller.target_face_id == "face:domain"
    assert len(controller.tool_face_ids) == 1
    assert controller.geometry == source
    assert controller.target_label() == "已选择"
    assert controller.tool_label() == "1 个闭合轮廓"


def test_controller_clears_target_and_tool_independently() -> None:
    controller = PlanarBooleanController(
        RectangleGeometry("Target", 3.0, 2.0),
        4,
        "cut",
        target_face_id="face:domain",
    )
    controller.set_tool_recipe(_tool_sketch())

    controller.clear_target()

    assert controller.target_face_id is None
    assert controller.tool_geometry is not None
    assert not controller.ready

    controller.set_target("face:domain")
    controller.clear_tool()

    assert controller.target_face_id == "face:domain"
    assert controller.tool_geometry is None
    assert controller.tool_face_ids == ()
    assert not controller.ready


def test_controller_requires_a_valid_face_target() -> None:
    controller = PlanarBooleanController(
        RectangleGeometry("Target", 3.0, 2.0),
        4,
        "cut",
    )
    assert not controller.ready
    controller.request_target_selection()

    try:
        controller.assign_reference(LogicalEntityRef("edge:bottom"))
    except ValueError as error:
        assert "Face" in str(error)
    else:
        raise AssertionError("an edge was accepted as a planar target")


def test_panel_reenables_inputs_after_running_preview_is_cancelled() -> None:
    _application()
    controller = PlanarBooleanController(
        RectangleGeometry("Target", 3.0, 2.0),
        4,
        "cut",
        target_face_id="face:domain",
    )
    controller.set_tool_recipe(_tool_sketch())
    panel = PlanarBooleanPanel()
    panel.begin(controller)
    assert panel.target_label.text() == "已选择"
    assert panel.tool_label.text() == "1 个闭合轮廓"
    assert panel.clear_target_button.isEnabled()
    assert panel.delete_tool_button.isEnabled()
    assert all(
        label.text() != "预览与诊断"
        for label in panel.findChildren(QLabel)
    )
    panel.set_preview_running(True)
    assert not panel.operation_combo.isEnabled()

    panel.end()
    panel.begin(controller)

    assert panel.operation_combo.isEnabled()
    assert panel.target_button.isEnabled()
    assert panel.tool_button.isEnabled()
    assert panel.clear_target_button.isEnabled()
    assert panel.delete_tool_button.isEnabled()
    panel.close()


def test_planar_boolean_panel_can_clear_target_and_delete_tool() -> None:
    _application()
    window = FEMMainWindow()
    source = RectangleGeometry("Target", 3.0, 2.0)
    window._set_native_geometry(source, "测试")
    window._selected_geometry_refs = {LogicalEntityRef("face:domain")}
    window.cut_geometry()
    controller = window._planar_boolean_controller
    assert controller is not None
    controller.set_tool_recipe(_tool_sketch())
    window.planar_boolean_panel.refresh()

    window.planar_boolean_panel.clear_target_button.click()

    assert controller.target_face_id is None
    assert controller.tool_geometry is not None
    assert window._selected_geometry_refs == set()
    assert not window.planar_boolean_panel.finish_button.isEnabled()

    controller.set_target("face:domain")
    window.planar_boolean_panel.refresh()
    window.planar_boolean_panel.delete_tool_button.click()

    assert controller.target_face_id == "face:domain"
    assert controller.tool_geometry is None
    assert controller.tool_face_ids == ()
    assert not window.planar_boolean_panel.finish_button.isEnabled()
    window.cancel_planar_boolean()
    window.close()


def test_planar_boolean_target_pick_is_persistently_highlighted(
    monkeypatch,
) -> None:
    _application()
    window = FEMMainWindow()
    window._set_native_geometry(RectangleGeometry("Target", 3.0, 2.0), "测试")
    highlighted = []
    monkeypatch.setattr(
        window.viewport,
        "highlight_geometry_entities",
        lambda references: highlighted.append(tuple(references)),
    )
    window.cut_geometry()
    bar = window.viewport_panel.planar_boolean_face_bar
    assert window.viewport_panel._active_bottom_overlay is bar
    assert bar.prompt_label.text() == "请选择目标面"
    assert not bar.confirm_button.isEnabled()
    target = LogicalEntityRef("face:domain")

    assert window._assign_planar_boolean_reference(target)

    assert window._selected_geometry_refs == {target}
    assert highlighted[-1] == (target,)
    assert window.viewport_panel._active_bottom_overlay is bar
    assert bar.confirm_button.isEnabled()
    assert window._sketch_editor_controller is None

    bar.confirm_button.click()

    assert bar.isHidden()
    assert window.viewport_panel._active_bottom_overlay is None
    assert window._sketch_editor_controller is not None
    window.cancel_planar_boolean()
    window.close()


def test_planar_boolean_face_prompt_cancel_restores_original_state() -> None:
    _application()
    window = FEMMainWindow()
    source = RectangleGeometry("Target", 3.0, 2.0)
    window._set_native_geometry(source, "测试")

    window.fuse_geometry()
    window.viewport_panel.planar_boolean_face_bar.cancel_button.click()

    assert window._planar_boolean_controller is None
    assert window._sketch_editor_controller is None
    assert window.viewport_panel._active_bottom_overlay is None
    assert window.document.geometry_recipe == source
    window.close()


def test_2d_boolean_with_preselected_face_waits_for_confirmation(
    monkeypatch,
) -> None:
    _application()
    window = FEMMainWindow()
    source = RectangleGeometry("Target", 3.0, 2.0)
    window._set_native_geometry(source, "测试")
    target = LogicalEntityRef("face:domain")
    window._selected_geometry_refs = {target}
    highlighted = []
    monkeypatch.setattr(
        window.viewport,
        "highlight_geometry_entities",
        lambda references: highlighted.append(tuple(references)),
    )

    window.cut_geometry()

    controller = window._planar_boolean_controller
    assert controller is not None
    assert window._body_boolean_controller is None
    assert controller.target_face_id == "face:domain"
    assert highlighted[-1] == (target,)
    assert window.planar_boolean_panel.isHidden()
    bar = window.viewport_panel.planar_boolean_face_bar
    assert window.viewport_panel._active_bottom_overlay is bar
    assert bar.confirm_button.isEnabled()
    assert window._sketch_editor_controller is None

    bar.confirm_button.click()

    assert bar.isHidden()
    assert window._sketch_editor_controller is not None
    assert window.document.geometry_recipe == source
    assert not window.actions["geometry_cut"].isEnabled()
    assert not window.actions["geometry_fuse"].isEnabled()

    window.cancel_planar_boolean()
    assert window.document.geometry_recipe == source
    assert window._selected_geometry_refs == {LogicalEntityRef("face:domain")}
    window.close()


def test_boolean_actions_are_disabled_for_one_dimensional_geometry() -> None:
    _application()
    window = FEMMainWindow()
    window._set_native_geometry(
        WireGeometry(
            "Wire",
            (
                WirePoint("P1", 0.0, 0.0, 0.0),
                WirePoint("P2", 1.0, 0.0, 0.0),
            ),
            (WireMember("M1", "P1", "P2"),),
        ),
        "测试",
    )

    assert not window.actions["geometry_cut"].isEnabled()
    assert not window.actions["geometry_fuse"].isEnabled()
    window.close()


def test_boolean_actions_are_disabled_while_sketch_editor_is_active() -> None:
    _application()
    window = FEMMainWindow()
    sketch = _tool_sketch()
    window._set_native_geometry(sketch, "测试")
    window._begin_sketch_editor(sketch, original_recipe=sketch)

    assert not window.actions["geometry_cut"].isEnabled()
    assert not window.actions["geometry_fuse"].isEnabled()
    window.cancel_sketch_geometry()
    window.close()


def test_tool_sketch_finish_requests_automatic_boolean_commit(
    monkeypatch,
) -> None:
    _application()
    window = FEMMainWindow()
    source = RectangleGeometry("Target", 3.0, 2.0)
    window._set_native_geometry(source, "测试")
    reference_calls = []
    monkeypatch.setattr(
        window.viewport,
        "show_sketch_reference_preview",
        lambda preview, **kwargs: reference_calls.append((preview, kwargs)),
    )
    refresh_calls = []
    monkeypatch.setattr(
        window,
        "_refresh_planar_boolean_preview",
        lambda **kwargs: refresh_calls.append(kwargs),
    )
    window._selected_geometry_refs = {LogicalEntityRef("face:domain")}
    window.cut_geometry()
    window.viewport_panel.planar_boolean_face_bar.confirm_button.click()

    draft = window._sketch_editor_controller
    assert draft is not None
    assert window.sketch_editor_panel.authoring_purpose == ("planar_boolean_tool")
    assert reference_calls
    assert "support_face_id" not in reference_calls[-1][1]
    draft.add_rectangle(0.5, 0.5, 1.5, 1.5)
    window.finish_sketch_geometry()

    controller = window._planar_boolean_controller
    assert controller is not None and controller.ready
    assert window._sketch_editor_controller is None
    assert window.planar_boolean_panel.isHidden()
    assert window.document.geometry_recipe == source
    assert refresh_calls == [{"auto_commit": True}]

    window.cancel_planar_boolean()
    window.close()


def test_successful_automatic_preview_immediately_requests_commit(
    monkeypatch,
) -> None:
    _application()
    window = FEMMainWindow()
    source = RectangleGeometry("Target", 3.0, 2.0)
    window._set_native_geometry(source, "测试")
    window._selected_geometry_refs = {LogicalEntityRef("face:domain")}
    window.cut_geometry()
    window.viewport_panel.planar_boolean_face_bar.confirm_button.click()
    controller = window._planar_boolean_controller
    assert controller is not None
    controller.set_tool_recipe(_tool_sketch())
    payload = object.__new__(StrictPlanarBooleanResult)
    object.__setattr__(payload, "geometry", source)
    object.__setattr__(payload, "proof", object())
    object.__setattr__(payload, "preview", object())
    monkeypatch.setattr(
        main_window_module,
        "build_strict_planar_boolean_preview",
        lambda _geometry, _preview: build_geometry_preview(source),
    )
    commit_calls = []
    monkeypatch.setattr(
        window,
        "finish_planar_boolean",
        lambda **kwargs: commit_calls.append(kwargs),
    )

    def complete_start(
        _workload,
        on_success,
        _label,
        _on_failure,
        **kwargs,
    ):
        assert kwargs["apply_result"](payload).status is TaskApplyStatus.ACCEPTED
        on_success(payload)
        return True

    monkeypatch.setattr(window, "_start_task", complete_start)

    window._refresh_planar_boolean_preview(auto_commit=True)

    assert window._planar_boolean_preview_result is payload
    assert len(commit_calls) == 1
    assert type(commit_calls[0]["preview"]) is GeometryPreview
    window.cancel_planar_boolean()
    window.close()


def test_automatic_finish_commits_without_reopening_a_panel(monkeypatch) -> None:
    _application()
    window = FEMMainWindow()
    source = RectangleGeometry("Target", 3.0, 2.0)
    window._set_native_geometry(source, "测试")
    window._selected_geometry_refs = {LogicalEntityRef("face:domain")}
    window.cut_geometry()
    controller = window._planar_boolean_controller
    assert controller is not None
    controller.set_tool_recipe(_tool_sketch())
    window._exit_sketch_editor()
    payload = object.__new__(StrictPlanarBooleanResult)
    object.__setattr__(payload, "geometry", source)
    object.__setattr__(
        payload,
        "proof",
        SimpleNamespace(result_entities=()),
    )
    object.__setattr__(payload, "preview", object())
    window._planar_boolean_preview_result = payload
    monkeypatch.setattr(
        main_window_module,
        "build_strict_planar_boolean_preview",
        lambda _geometry, _preview: build_geometry_preview(source),
    )
    commits = []

    def commit(recipe, label, **kwargs):
        commits.append((recipe, label, kwargs))
        return True

    monkeypatch.setattr(window, "_set_native_geometry", commit)

    window.finish_planar_boolean()

    assert len(commits) == 1
    recipe, label, kwargs = commits[0]
    assert recipe == source
    assert label == "二维布尔后的"
    assert kwargs["base_session_revision"] == controller.base_session_revision
    assert kwargs["preserve_editor"] is True
    assert kwargs["geometry_preview"].face_logical_ids == ("face:P1/domain",)
    assert window._planar_boolean_controller is None
    assert window.planar_boolean_panel.isHidden()
    window.close()


def test_committed_exact_preview_survives_full_projection_rebuild(
    monkeypatch,
) -> None:
    _application()
    window = FEMMainWindow()
    window._set_native_geometry(
        RectangleGeometry("Initial", 3.0, 2.0),
        "测试",
    )
    updated = RectangleGeometry("Updated", 4.0, 2.0)
    part_id = str(window.document.active_part_id)
    exact = namespace_part_geometry_preview(
        part_id,
        build_geometry_preview(updated),
    )
    shown = []
    monkeypatch.setattr(
        window.viewport,
        "show_geometry_preview",
        lambda preview, **_kwargs: shown.append(preview),
    )

    assert window._set_native_geometry(
        updated,
        "精确预览测试",
        geometry_preview=exact,
    )
    cache = window._geometry_preview_cache
    assert cache is not None
    assert cache[1] == window._native_part_preview_cache_key(window.document)
    assert cache[2] is exact

    def reject_fallback(_parts):
        raise AssertionError("当前精确预览不应回退到配方重建")

    monkeypatch.setattr(
        main_window_module,
        "build_multi_part_geometry_preview",
        reject_fallback,
    )
    exact_rebuilds = []
    monkeypatch.setattr(
        window,
        "_recipe_contains_strict_boolean",
        lambda _recipe: True,
    )
    monkeypatch.setattr(
        window,
        "_schedule_exact_boolean_preview",
        lambda *_args: exact_rebuilds.append(True),
    )

    window._rebuild_full_projection()

    assert shown == [exact, exact]
    assert exact_rebuilds == []
    window.close()


def test_fuse_tool_accepts_arc_closed_by_target_edge_and_tangent_rectangle(
    monkeypatch,
) -> None:
    _application()
    window = FEMMainWindow()
    source = RectangleGeometry("Target", 3.0, 2.0)
    window._set_native_geometry(source, "测试")
    window._selected_geometry_refs = {LogicalEntityRef("face:domain")}
    window.fuse_geometry()
    window.viewport_panel.planar_boolean_face_bar.confirm_button.click()
    monkeypatch.setattr(
        window,
        "_refresh_planar_boolean_preview",
        lambda **_kwargs: None,
    )

    draft = window._sketch_editor_controller
    assert draft is not None
    references = {
        item.reference.source.logical_id: item
        for item in window.sketch_editor_panel._reference_points
    }
    draft.add_arc(
        (3.0, 0.0),
        (4.0, 1.0),
        (3.0, 2.0),
        start_external_reference=references["point:bottom-right"],
        end_external_reference=references["point:top-right"],
    )
    draft.add_rectangle((1.0, -1.0), (3.0, 0.0))
    window.sketch_editor_panel._refresh()

    assert not draft.can_finish
    assert window.sketch_editor_panel.finish_button.isEnabled()
    window.sketch_editor_panel.try_finish()

    controller = window._planar_boolean_controller
    assert controller is not None and controller.ready
    assert controller.tool_geometry is not None
    assert len(controller.tool_face_ids) == 2
    assert sum(
        isinstance(curve, SketchArc) for curve in controller.tool_geometry.curves
    ) == 1
    assert sum(
        isinstance(curve, SketchLine) for curve in controller.tool_geometry.curves
    ) == 5
    window.cancel_planar_boolean()
    window.close()


def test_tool_sketch_cancel_aborts_the_guided_boolean_workflow() -> None:
    _application()
    window = FEMMainWindow()
    source = RectangleGeometry("Target", 3.0, 2.0)
    window._set_native_geometry(source, "测试")
    window._selected_geometry_refs = {LogicalEntityRef("face:domain")}
    window.cut_geometry()
    window.viewport_panel.planar_boolean_face_bar.confirm_button.click()

    assert window._sketch_editor_controller is not None
    window.cancel_sketch_geometry()

    assert window._sketch_editor_controller is None
    assert window._planar_boolean_controller is None
    assert window.planar_boolean_panel.isHidden()
    assert window.document.geometry_recipe == source
    window.close()


def test_stale_async_preview_is_discarded_when_operation_changes(
    monkeypatch,
) -> None:
    _application()
    window = FEMMainWindow()
    source = RectangleGeometry("Target", 3.0, 2.0)
    window._set_native_geometry(source, "测试")
    window._selected_geometry_refs = {LogicalEntityRef("face:domain")}
    window.cut_geometry()
    controller = window._planar_boolean_controller
    assert controller is not None
    controller.set_tool_recipe(_tool_sketch())
    captured = {}

    def capture_start(
        _workload,
        _on_success,
        _label,
        _on_failure,
        **kwargs,
    ):
        captured["apply_result"] = kwargs["apply_result"]
        return True

    monkeypatch.setattr(window, "_start_task", capture_start)
    window._refresh_planar_boolean_preview()
    controller.set_operation("fuse")
    payload = object.__new__(StrictPlanarBooleanResult)

    outcome = captured["apply_result"](payload)

    assert outcome.status is TaskApplyStatus.STALE
    assert window._planar_boolean_preview_result is None
    assert not window.planar_boolean_panel.finish_button.isEnabled()
    window.cancel_planar_boolean()
    window.close()


def test_occ_preview_failure_keeps_committed_geometry(monkeypatch) -> None:
    _application()
    window = FEMMainWindow()
    source = RectangleGeometry("Target", 3.0, 2.0)
    window._set_native_geometry(source, "测试")
    window._selected_geometry_refs = {LogicalEntityRef("face:domain")}
    window.cut_geometry()
    controller = window._planar_boolean_controller
    assert controller is not None
    controller.set_tool_recipe(_tool_sketch())

    def fail_start(
        _workload,
        _on_success,
        _label,
        on_failure,
        **_kwargs,
    ):
        on_failure("synthetic OCC failure")
        return True

    monkeypatch.setattr(window, "_start_task", fail_start)
    window._refresh_planar_boolean_preview()

    assert window.document.geometry_recipe == source
    assert window._planar_boolean_preview_result is None
    assert not window.planar_boolean_panel.finish_button.isEnabled()
    window.cancel_planar_boolean()
    window.close()


def test_automatic_preview_failure_reopens_the_tool_sketch(monkeypatch) -> None:
    application = _application()
    window = FEMMainWindow()
    source = RectangleGeometry("Target", 3.0, 2.0)
    window._set_native_geometry(source, "测试")
    window._selected_geometry_refs = {LogicalEntityRef("face:domain")}
    window.cut_geometry()
    window.viewport_panel.planar_boolean_face_bar.confirm_button.click()
    controller = window._planar_boolean_controller
    assert controller is not None
    controller.set_tool_recipe(_tool_sketch())
    window._exit_sketch_editor()

    def fail_start(
        _workload,
        _on_success,
        _label,
        on_failure,
        **_kwargs,
    ):
        on_failure("synthetic OCC failure")
        return True

    monkeypatch.setattr(window, "_start_task", fail_start)

    window._refresh_planar_boolean_preview(auto_commit=True)
    application.processEvents()

    assert window.document.geometry_recipe == source
    assert window._sketch_editor_controller is not None
    assert window.sketch_editor_panel.authoring_purpose == "planar_boolean_tool"
    window.cancel_sketch_geometry()
    window.close()


def test_tool_sketch_revision_conflict_keeps_latest_committed_geometry(
    monkeypatch,
) -> None:
    _application()
    window = FEMMainWindow()
    source = RectangleGeometry("Target", 3.0, 2.0)
    window._set_native_geometry(source, "测试")
    window._selected_geometry_refs = {LogicalEntityRef("face:domain")}
    window.cut_geometry()
    window.viewport_panel.planar_boolean_face_bar.confirm_button.click()
    draft = window._sketch_editor_controller
    assert draft is not None
    draft.add_rectangle(0.5, 0.5, 1.5, 1.5)

    latest = RectangleGeometry("Latest", 5.0, 2.0)
    assert window._set_native_geometry(latest, "外部更新")
    window.finish_sketch_geometry()

    assert window.document.geometry_recipe == latest
    assert window._sketch_editor_controller is draft
    assert window._planar_boolean_controller is not None

    monkeypatch.setattr(window, "_confirm_sketch_editor_discard", lambda: True)
    window.cancel_sketch_geometry()
    window.cancel_planar_boolean()
    window.close()
