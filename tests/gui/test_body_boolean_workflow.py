from __future__ import annotations

import os
from time import monotonic

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication, QMessageBox

from fem import geometry as geometry_runtime
from fem.geometry import (
    BooleanLineageProof,
    BoxGeometry,
    CylinderGeometry,
    LogicalEntityRef,
    MovedGeometry,
    MultiBodyGeometry,
    SolidBody,
    provisional_body_boolean,
)
from fem.application import (
    StrictBodyBooleanPreview,
    StrictBodyBooleanResult,
    prepare_solid_body_boolean,
)
from fem_gui.body_boolean import BodyBooleanController
from fem_gui.main_window import FEMMainWindow
from fem_gui.preprocessing_dialogs import AddBodyGeometryDialog
from fem_gui.task_controller import TaskApplyStatus


def _application() -> QApplication:
    return QApplication.instance() or QApplication([])


def _two_bodies() -> MultiBodyGeometry:
    return MultiBodyGeometry(
        "Boolean Geometry",
        (
            SolidBody("B1", "Target", BoxGeometry("A", 2.0, 1.0, 1.0)),
            SolidBody(
                "B2",
                "Tool",
                MovedGeometry(
                    BoxGeometry("B", 2.0, 1.0, 1.0),
                    1.0,
                    0.0,
                    0.0,
                ),
            ),
        ),
    )


def test_controller_assigns_explicit_roles_without_set_ordering() -> None:
    controller = BodyBooleanController(_two_bodies(), 7, "cut")
    controller.request_selection("tool")
    assert controller.assign_reference(LogicalEntityRef("body:B2")) == "tool"
    controller.request_selection("target")
    assert (
        controller.assign_reference(LogicalEntityRef("body:B1"))
        == "target"
    )

    assert controller.ready
    assert controller.target_body_id == "B1"
    assert controller.tool_body_id == "B2"


def test_controller_rejects_same_body_for_both_roles() -> None:
    controller = BodyBooleanController(_two_bodies(), 7, "fuse")
    controller.request_selection("target")
    controller.assign_reference(LogicalEntityRef("body:B1"))
    controller.request_selection("tool")

    try:
        controller.assign_reference(LogicalEntityRef("body:B1"))
    except ValueError as error:
        assert "different Bodies" in str(error)
    else:
        raise AssertionError("same target/tool Body was accepted")


def test_non_modal_boolean_panel_cancel_preserves_committed_geometry() -> None:
    _application()
    window = FEMMainWindow()
    source = _two_bodies()
    window._set_native_geometry(source, "测试")

    window.cut_geometry()

    assert window._body_boolean_controller is not None
    assert not window.boolean_feature_panel.isHidden()
    assert window.document.geometry_recipe == source

    window.cancel_body_boolean()

    assert window._body_boolean_controller is None
    assert window.document.geometry_recipe == source
    window.close()


def test_boolean_guided_picks_fill_target_and_tool_slots(monkeypatch) -> None:
    _application()
    window = FEMMainWindow()
    window._set_native_geometry(_two_bodies(), "测试")
    window.fuse_geometry()
    monkeypatch.setattr(window, "_refresh_body_boolean_preview", lambda: None)

    window._request_body_boolean_selection("tool")
    window._on_geometry_entity_pick(LogicalEntityRef("body:B2"))
    window._request_body_boolean_selection("target")
    window._on_geometry_entity_pick(LogicalEntityRef("body:B1"))

    controller = window._body_boolean_controller
    assert controller is not None
    assert controller.target_body_id == "B1"
    assert controller.tool_body_id == "B2"
    assert window._selected_geometry_refs == set()
    window.cancel_body_boolean()
    window.close()


def test_last_body_delete_confirms_impact_and_never_reuses_id(
    monkeypatch,
) -> None:
    _application()
    window = FEMMainWindow()
    window._set_native_geometry(
        BoxGeometry("Original", 1.0, 1.0, 1.0),
        "测试",
    )
    window._selected_geometry_refs = {LogicalEntityRef("body:B1")}
    messages: list[str] = []

    def confirm(_parent, _title, message, *_buttons):
        messages.append(str(message))
        return QMessageBox.StandardButton.Yes

    monkeypatch.setattr(QMessageBox, "question", confirm)
    window.delete_geometry()

    assert window.document.geometry_recipe is None
    assert window.session.retired_body_ids == ("B1",)
    assert messages and "命名区域" in messages[0]

    window._set_native_geometry(
        BoxGeometry("Replacement", 1.0, 1.0, 1.0),
        "测试",
    )
    replacement = window.document.geometry_recipe
    assert isinstance(replacement, MultiBodyGeometry)
    assert tuple(body.id for body in replacement.bodies) == ("B2",)
    assert replacement.retired_body_ids == ("B1",)
    window.close()


def test_add_body_dialog_emits_placed_detached_preview_before_commit(
    monkeypatch,
) -> None:
    _application()
    window = FEMMainWindow()
    source = _two_bodies()
    window._set_native_geometry(source, "测试")
    previews = []
    original_show = window.viewport.show_geometry_preview

    def capture_preview(preview, *args, **kwargs):
        previews.append(preview)
        return original_show(preview, *args, **kwargs)

    def finish_dialog(dialog):
        assert isinstance(dialog, AddBodyGeometryDialog)
        assert window.document.geometry_recipe == source
        dialog.kind_combo.setCurrentIndex(1)
        dialog.radius_spin.setValue(0.25)
        dialog.height_spin.setValue(1.5)
        dialog.x_spin.setValue(4.0)
        assert window.document.geometry_recipe == source
        return 1

    monkeypatch.setattr(
        window.viewport,
        "show_geometry_preview",
        capture_preview,
    )
    monkeypatch.setattr(window, "_exec_dialog", finish_dialog)

    window.add_body_geometry()

    committed = window.document.geometry_recipe
    assert isinstance(committed, MultiBodyGeometry)
    added = committed.body("B3").recipe
    assert isinstance(added, MovedGeometry)
    assert isinstance(added.base, CylinderGeometry)
    assert (added.dx, added.dy, added.dz) == (4.0, 0.0, 0.0)
    assert any(
        set(preview.face_body_logical_ids)
        >= {"body:B1", "body:B2", "body:B3"}
        for preview in previews
    )
    window.close()


def test_add_body_dialog_cancel_restores_committed_projection(
    monkeypatch,
) -> None:
    _application()
    window = FEMMainWindow()
    source = _two_bodies()
    window._set_native_geometry(source, "测试")
    monkeypatch.setattr(window, "_exec_dialog", lambda _dialog: 0)

    window.add_body_geometry()

    assert window.document.geometry_recipe == source
    window.close()


def test_cancelled_boolean_preview_discards_late_worker_result(
    monkeypatch,
) -> None:
    _application()
    window = FEMMainWindow()
    source = _two_bodies()
    window._set_native_geometry(source, "测试")
    callbacks: dict[str, object] = {}

    def capture_start(
        _workload,
        _on_success,
        _error_title,
        _on_failure=None,
        **kwargs,
    ):
        callbacks["apply_result"] = kwargs["apply_result"]
        return True

    monkeypatch.setattr(window, "_start_task", capture_start)
    window.fuse_geometry()
    window._request_body_boolean_selection("target")
    window._on_geometry_entity_pick(LogicalEntityRef("body:B1"))
    window._request_body_boolean_selection("tool")
    window._on_geometry_entity_pick(LogicalEntityRef("body:B2"))

    assert not window.boolean_feature_panel.operation_combo.isEnabled()
    apply_result = callbacks["apply_result"]
    window.cancel_body_boolean()
    draft = provisional_body_boolean(source, "B1", "B2", "fuse")
    late = StrictBodyBooleanResult(
        source,
        draft,
        BooleanLineageProof(
            object(),
            {},
            (),
            (),
            (),
        ),
        StrictBodyBooleanPreview("B1", (), (), (), (), (), ()),
    )
    outcome = apply_result(late)

    assert outcome.status is TaskApplyStatus.STALE
    assert window._body_boolean_preview_result is None
    assert window.document.geometry_recipe == source
    window.close()


@pytest.mark.gmsh
def test_reopened_boolean_recipe_rebuilds_exact_occ_preview(
    real_gmsh,
) -> None:
    del real_gmsh
    application = _application()
    source = _two_bodies()
    with geometry_runtime.model(
        "gui-reopen-boolean-source",
        dimension=3,
    ) as cad:
        prepared = prepare_solid_body_boolean(
            cad,
            source,
            "B1",
            "B2",
            "fuse",
        )

    window = FEMMainWindow()
    window._set_native_geometry(prepared.geometry, "重开")
    deadline = monotonic() + 10.0
    while monotonic() < deadline:
        application.processEvents()
        if (
            window._pending_exact_boolean_preview_key is None
            and not window.task_controller.busy
            and window._geometry_preview_cache is not None
        ):
            break
    application.processEvents()

    cached = window._geometry_preview_cache
    assert cached is not None
    preview = cached[2]
    assert any(
        logical_id is not None and "/boolean/BF1/" in logical_id
        for logical_id in preview.face_logical_ids
    )
    window.close()


@pytest.mark.gmsh
def test_async_boolean_finish_consumes_tool_and_selects_target(
    real_gmsh,
) -> None:
    del real_gmsh
    application = _application()
    window = FEMMainWindow()
    window._set_native_geometry(_two_bodies(), "测试")
    window.fuse_geometry()
    window._request_body_boolean_selection("target")
    window._on_geometry_entity_pick(LogicalEntityRef("body:B1"))
    window._request_body_boolean_selection("tool")
    window._on_geometry_entity_pick(LogicalEntityRef("body:B2"))

    deadline = monotonic() + 10.0
    while window.task_controller.busy and monotonic() < deadline:
        application.processEvents()
    application.processEvents()

    assert window._body_boolean_preview_result is not None
    assert window.boolean_feature_panel.finish_button.isEnabled()
    window.finish_body_boolean()
    committed = window.document.geometry_recipe
    assert isinstance(committed, MultiBodyGeometry)
    assert tuple(body.id for body in committed.bodies) == ("B1",)
    assert window._selected_geometry_refs == {
        LogicalEntityRef("body:B1")
    }
    window.close()
