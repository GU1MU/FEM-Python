from __future__ import annotations

import os
from time import monotonic

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication
import pytest

from fem import geometry
from fem.application import NativePart, prepare_part_boolean
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


def test_committed_part_boolean_projects_and_rebuilds_exact_preview(
    real_gmsh,
) -> None:
    del real_gmsh
    application = _application()
    window = FEMMainWindow()
    window._create_native_model("模型-1")
    for part in _parts():
        window._apply_session_delta(
            window.session.add_native_part(
                part.geometry_recipe,
                name=part.name,
            )
        )
    target, tool = window.document.parts
    with geometry.model("GUI 部件布尔", dimension=3) as cad:
        prepared = prepare_part_boolean(
            cad,
            target,
            tool,
            "fuse",
            result_part_id="P3",
            feature_id="PBF1",
            result_name="合并结果-1",
        )

    window._apply_session_delta(
        window.session.apply_part_boolean(
            "P1",
            "P2",
            "fuse",
            "合并结果-1",
            result=prepared,
        )
    )
    deadline = monotonic() + 10.0
    while monotonic() < deadline:
        application.processEvents()
        if not window.busy and window._pending_exact_boolean_preview_key is None:
            break

    cached = window._geometry_preview_cache
    assert window.document.active_part_id == "P3"
    assert cached is not None
    assert set(cached[2].face_part_ids) == {"P3"}
    assert all(
        logical_id is None or logical_id.startswith("face:P3/")
        for logical_id in cached[2].face_logical_ids
    )
    window.close()


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
