from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication
import pytest

from fem.application import NativePart
from fem.geometry import (
    BoxGeometry,
    FaceSketchBooleanOperation,
    LogicalEntityRef,
    MovedGeometry,
)
from fem_gui.face_sketch_boolean_dialog import FaceSketchBooleanDialog
from fem_gui.main_window import FEMMainWindow
from fem_gui.part_boolean import PartBooleanController
from fem_gui.sketch_editor import SketchDraftController
from fem_gui.widgets.boolean_feature_panel import BooleanFeaturePanel


def _application() -> QApplication:
    return QApplication.instance() or QApplication([])


def _parts() -> tuple[NativePart, ...]:
    return (
        NativePart(
            id="P1",
            name="目标部件",
            geometry_recipe=BoxGeometry("目标", 2.0, 1.0, 1.0),
        ),
        NativePart(
            id="P2",
            name="工具部件",
            geometry_recipe=MovedGeometry(
                BoxGeometry("工具", 1.0, 1.0, 1.0),
                1.5,
                0.0,
                0.0,
            ),
        ),
    )


def test_controller_assigns_distinct_part_operands() -> None:
    controller = PartBooleanController(_parts(), 4, "cut")
    controller.request_selection("target")
    controller.assign_reference(LogicalEntityRef("part:P1"))
    controller.request_selection("tool")
    controller.assign_reference(LogicalEntityRef("part:P2"))

    assert controller.ready
    assert controller.target_part_id == "P1"
    assert controller.tool_part_id == "P2"
    assert controller.part_label("P1") == "目标部件 [P1]"


def test_controller_rejects_face_as_part_operand() -> None:
    controller = PartBooleanController(_parts(), 4, "cut")
    controller.request_selection("target")

    with pytest.raises(ValueError, match="稳定部件"):
        controller.assign_reference(LogicalEntityRef("face:P1/top"))


def test_panel_uses_part_terminology_and_result_name() -> None:
    _application()
    panel = BooleanFeaturePanel()
    controller = PartBooleanController(
        _parts(),
        4,
        "fuse",
        target_part_id="P1",
    )
    panel.begin(controller)

    assert panel.result_name() == "合并结果-1"
    texts = {
        widget.text()
        for widget in panel.findChildren(type(panel.status_label))
    }
    assert any("目标部件" in text for text in texts)
    assert any("工具部件" in text for text in texts)
    panel.close()


def test_3d_boolean_target_face_pick_does_not_change_session_revision(
    monkeypatch,
) -> None:
    _application()
    window = FEMMainWindow()
    window._create_native_model("模型-1")
    window._set_native_geometry(
        BoxGeometry("目标", 2.0, 1.0, 1.0),
        "测试",
    )
    monkeypatch.setattr(
        window,
        "_face_sketch_selection_is_valid",
        lambda: bool(window._selected_geometry_refs),
    )
    assert window.actions["geometry_cut"].isEnabled()
    window.cut_geometry()
    bar = window.viewport_panel.planar_boolean_face_bar
    assert window._body_boolean_controller is None
    assert window._solid_face_boolean_operation is (
        FaceSketchBooleanOperation.CUT
    )
    assert window.viewport_panel._active_bottom_overlay is bar
    assert not bar.confirm_button.isEnabled()
    base_revision = window.document.session_revision

    window._on_geometry_entity_pick(
        LogicalEntityRef("face:P1/top")
    )

    assert window._selected_geometry_refs == {
        LogicalEntityRef("face:P1/top")
    }
    assert bar.confirm_button.isEnabled()
    assert window.document.session_revision == base_revision
    launches = []
    monkeypatch.setattr(
        window,
        "start_face_sketch_boolean",
        lambda: launches.append(True),
    )

    bar.confirm_button.click()

    assert launches == [True]
    assert bar.isHidden()
    window._cancel_solid_face_boolean()
    window.close()


def test_3d_boolean_parameter_dialog_locks_requested_operation() -> None:
    _application()
    dialog = FaceSketchBooleanDialog()

    dialog.fix_operation(FaceSketchBooleanOperation.CUT)

    assert dialog.operation_combo.currentData() == "cut"
    assert not dialog.operation_combo.isEnabled()
    assert dialog.windowTitle() == "拉伸切除"
    assert dialog.distance_label.text() == "切除深度："
    dialog.close()


def test_3d_cut_parameters_default_toward_the_solid(monkeypatch) -> None:
    _application()
    window = FEMMainWindow()
    draft = SketchDraftController("工具草图")
    draft.add_rectangle(0.0, 0.0, 1.0, 1.0)
    window._face_sketch_controller = object()
    window._solid_face_boolean_operation = FaceSketchBooleanOperation.CUT
    changed = []
    monkeypatch.setattr(
        window,
        "_face_sketch_boolean_parameters_changed",
        lambda parameters: changed.append(parameters),
    )

    window._open_face_sketch_boolean_dialog(draft.to_sketch_geometry())

    dialog = window._face_sketch_dialog
    assert dialog is not None
    parameters = dialog.parameters()
    assert parameters is not None
    assert parameters.operation is FaceSketchBooleanOperation.CUT
    assert parameters.direction.value == "inward"
    assert not dialog.operation_combo.isEnabled()
    assert changed == [parameters]
    dialog.close_for_workflow()
    window._face_sketch_dialog = None
    window._face_sketch_controller = None
    window._solid_face_boolean_operation = None
    window.close()
