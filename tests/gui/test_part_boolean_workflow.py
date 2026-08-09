from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication
import pytest

from fem.application import NativePart
from fem.geometry import BoxGeometry, LogicalEntityRef, MovedGeometry
from fem_gui.main_window import FEMMainWindow
from fem_gui.part_boolean import PartBooleanController
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


def test_viewport_operand_pick_does_not_change_session_revision(
    monkeypatch,
) -> None:
    _application()
    window = FEMMainWindow()
    window._create_native_model("模型-1")
    window._set_native_geometry(
        BoxGeometry("目标", 2.0, 1.0, 1.0),
        "测试",
    )
    window._apply_session_delta(
        window.session.add_native_part(
            MovedGeometry(
                BoxGeometry("工具", 1.0, 1.0, 1.0),
                1.5,
                0.0,
                0.0,
            ),
            name="工具部件",
        )
    )
    window.cut_geometry()
    controller = window._body_boolean_controller
    assert controller is not None
    base_revision = window.document.session_revision
    monkeypatch.setattr(window, "_refresh_body_boolean_preview", lambda: None)

    window._request_body_boolean_selection("tool")
    window._on_geometry_entity_pick(
        LogicalEntityRef("body:P1/domain")
    )

    assert controller.tool_part_id == "P1"
    assert window.document.session_revision == base_revision
    window.cancel_body_boolean()
    window.close()
