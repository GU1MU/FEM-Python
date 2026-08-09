from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QLabel

from fem.application import StrictPlanarBooleanResult
from fem.geometry import (
    LogicalEntityRef,
    RectangleGeometry,
    SketchGeometry,
    SketchLine,
    SketchPlane,
    SketchPoint,
    WireGeometry,
    WireMember,
    WirePoint,
)
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


def test_2d_boolean_dispatches_to_non_modal_planar_workflow() -> None:
    _application()
    window = FEMMainWindow()
    source = RectangleGeometry("Target", 3.0, 2.0)
    window._set_native_geometry(source, "测试")
    window._selected_geometry_refs = {LogicalEntityRef("face:domain")}

    window.cut_geometry()

    controller = window._planar_boolean_controller
    assert controller is not None
    assert window._body_boolean_controller is None
    assert controller.target_face_id == "face:domain"
    assert not window.planar_boolean_panel.isHidden()
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


def test_tool_sketch_finish_returns_to_planar_panel_without_committing(
    monkeypatch,
) -> None:
    _application()
    window = FEMMainWindow()
    source = RectangleGeometry("Target", 3.0, 2.0)
    window._set_native_geometry(source, "测试")
    window._selected_geometry_refs = {LogicalEntityRef("face:domain")}
    window.cut_geometry()
    monkeypatch.setattr(
        window,
        "_refresh_planar_boolean_preview",
        lambda: None,
    )

    window._edit_planar_boolean_tool()
    draft = window._sketch_editor_controller
    assert draft is not None
    assert window.sketch_editor_panel.authoring_purpose == ("planar_boolean_tool")
    draft.add_rectangle(0.5, 0.5, 1.5, 1.5)
    window.finish_sketch_geometry()

    controller = window._planar_boolean_controller
    assert controller is not None and controller.ready
    assert window._sketch_editor_controller is None
    assert not window.planar_boolean_panel.isHidden()
    assert window.document.geometry_recipe == source

    window.cancel_planar_boolean()
    window.close()


def test_tool_sketch_cancel_returns_to_boolean_panel() -> None:
    _application()
    window = FEMMainWindow()
    source = RectangleGeometry("Target", 3.0, 2.0)
    window._set_native_geometry(source, "测试")
    window._selected_geometry_refs = {LogicalEntityRef("face:domain")}
    window.cut_geometry()
    controller = window._planar_boolean_controller

    window._edit_planar_boolean_tool()
    assert window._sketch_editor_controller is not None
    window.cancel_sketch_geometry()

    assert window._sketch_editor_controller is None
    assert window._planar_boolean_controller is controller
    assert not window.planar_boolean_panel.isHidden()
    assert window.document.geometry_recipe == source
    window.cancel_planar_boolean()
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


def test_tool_sketch_revision_conflict_keeps_latest_committed_geometry(
    monkeypatch,
) -> None:
    _application()
    window = FEMMainWindow()
    source = RectangleGeometry("Target", 3.0, 2.0)
    window._set_native_geometry(source, "测试")
    window._selected_geometry_refs = {LogicalEntityRef("face:domain")}
    window.cut_geometry()
    window._edit_planar_boolean_tool()
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
