from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from fem.application import NativePart
from fem.geometry import (
    BoxGeometry,
    ExtrudedGeometry,
    RectangleGeometry,
    RotatedGeometry,
    SketchGeometry,
    SketchLine,
    SketchPlane,
    SketchPoint,
)
from fem_gui.action_state import ACTION_DESCRIPTORS
from fem_gui.part_geometry_preview import build_multi_part_geometry_preview
from fem_gui.main_window import FEMMainWindow
from fem_gui.preprocessing_dialogs import GeometryCreationDialog
from fem_gui.widgets.model_tree import ROLE_KEY, ROLE_KIND, ModelTree


def _application() -> QApplication:
    return QApplication.instance() or QApplication([])


def test_new_part_dialog_collects_name_and_dimension() -> None:
    _application()
    dialog = GeometryCreationDialog(default_part_name="部件-2")

    assert dialog.windowTitle() == "新建部件"
    assert dialog.part_name() == "部件-2"
    assert dialog.creation_kind() == "1d"
    dialog.dimension_list.setCurrentRow(2)
    assert dialog.creation_kind() == "3d"
    dialog.close()


def test_add_body_action_is_absent_from_production_catalog() -> None:
    assert all(
        descriptor.key.value != "geometry_add_body"
        and descriptor.text != "添加实体"
        for descriptor in ACTION_DESCRIPTORS
    )


def test_model_tree_uses_color_for_current_part_without_showing_ids() -> None:
    _application()
    tree = ModelTree()
    parts = (
        NativePart(
            id="P1",
            name="部件-1",
            geometry_recipe=RectangleGeometry("草图-1", 2.0, 1.0),
        ),
        NativePart(
            id="P2",
            name="工具部件",
            geometry_recipe=BoxGeometry("实体-1", 1.0, 1.0, 1.0),
            suppressed=True,
        ),
    )
    tree.set_geometry_preview(
        "模型-1",
        (),
        parts=parts,
        active_part_id="P1",
    )
    root = tree.topLevelItem(0)

    assert root.childCount() == 2
    assert root.child(0).data(0, ROLE_KIND) == "part"
    assert root.child(0).data(0, ROLE_KEY) == "P1"
    assert root.child(0).text(0) == "部件-1"
    assert root.child(0).background(0).color().name() == "#d9ecff"
    assert root.child(1).text(0) == "工具部件（已抑制）"
    assert tree.currentItem() is root.child(0)
    tree.close()


def test_multi_part_preview_carries_namespaced_refs_and_part_ids() -> None:
    parts = (
        NativePart(
            id="P1",
            name="部件-1",
            geometry_recipe=RectangleGeometry("草图-1", 2.0, 1.0),
        ),
        NativePart(
            id="P2",
            name="部件-2",
            geometry_recipe=RectangleGeometry("草图-2", 1.0, 1.0),
        ),
    )

    preview = build_multi_part_geometry_preview(parts)

    assert set(preview.face_part_ids) == {"P1", "P2"}
    assert set(preview.face_logical_ids) == {
        "face:P1/domain",
        "face:P2/domain",
    }
    assert set(preview.face_body_logical_ids) == {
        "body:P1/domain",
        "body:P2/domain",
    }


def test_switching_current_part_hides_other_geometry_previews() -> None:
    _application()
    window = FEMMainWindow()
    window._create_native_model("模型-1")
    window._apply_session_delta(
        window.session.add_native_part(
            RectangleGeometry("草图-1", 2.0, 1.0),
            name="部件-1",
        )
    )
    window._apply_session_delta(
        window.session.add_native_part(
            RectangleGeometry("草图-2", 1.0, 1.0),
            name="部件-2",
        )
    )

    preview = window.viewport._geometry_preview
    assert preview is not None
    assert set(preview.face_part_ids) == {"P2"}

    window._highlight_tree_entry("part", "P1")

    preview = window.viewport._geometry_preview
    assert window.document.active_part_id == "P1"
    assert preview is not None
    assert set(preview.face_part_ids) == {"P1"}
    root = window.model_tree.topLevelItem(0)
    assert root.child(0).text(0) == "部件-1"
    assert root.child(0).background(0).color().name() == "#d9ecff"
    assert root.child(1).text(0) == "部件-2"
    window.close()


def test_rotating_extruded_strict_part_commits_without_stale_occ_refs(
    monkeypatch,
) -> None:
    _application()
    sketch = SketchGeometry(
        "草图",
        SketchPlane.xy(),
        (
            SketchPoint("P1", 0.0, 0.0),
            SketchPoint("P2", 2.0, 0.0),
            SketchPoint("P3", 2.0, 1.0),
            SketchPoint("P4", 0.0, 1.0),
        ),
        (
            SketchLine("L1", "P1", "P2"),
            SketchLine("L2", "P2", "P3"),
            SketchLine("L3", "P3", "P4"),
            SketchLine("L4", "P4", "P1"),
        ),
    )
    source = ExtrudedGeometry(sketch, 0.75)
    window = FEMMainWindow()
    window._create_native_model("模型-1")
    window._apply_session_delta(
        window.session.add_native_part(source, name="拉伸部件")
    )
    errors: list[tuple[str, str]] = []

    class _Dialog:
        def __init__(self, base, *_args, **_kwargs) -> None:
            self._base = base

        def recipe(self):
            return RotatedGeometry(self._base, "x", 37.0)

    monkeypatch.setattr(
        "fem_gui.main_window.RotateGeometryDialog",
        _Dialog,
    )
    monkeypatch.setattr(window, "_exec_dialog", lambda _dialog: True)
    monkeypatch.setattr(
        window,
        "_show_error",
        lambda title, message: errors.append((title, message)),
    )

    window.rotate_geometry()

    assert errors == []
    assert window.document.geometry_recipe == RotatedGeometry(
        source,
        "x",
        37.0,
    )
    window.close()


def test_finishing_basic_solids_appends_parts(monkeypatch) -> None:
    _application()
    window = FEMMainWindow()
    window._create_native_model("模型-1")

    class _Dialog:
        def __init__(self, *args, **kwargs) -> None:
            del args, kwargs

        def recipe(self):
            return BoxGeometry("长方体-1", 1.0, 1.0, 1.0)

    monkeypatch.setattr("fem_gui.main_window.BoxGeometryDialog", _Dialog)
    monkeypatch.setattr(window, "_exec_dialog", lambda _dialog: True)

    window._create_basic_solid_part("3d_box", "部件-1")
    window._create_basic_solid_part("3d_box", "部件-2")

    assert tuple(part.id for part in window.document.parts) == ("P1", "P2")
    assert tuple(part.name for part in window.document.parts) == (
        "部件-1",
        "部件-2",
    )
    window.close()


def test_detached_sketch_and_wire_finish_append_while_cancel_adds_nothing() -> None:
    _application()
    window = FEMMainWindow()
    window._create_native_model("模型-1")

    window._begin_sketch_editor(
        None,
        original_recipe=None,
        part_name="二维部件",
    )
    sketch = window._sketch_editor_controller
    assert sketch is not None
    sketch.add_rectangle(0.0, 0.0, 2.0, 1.0)
    window.finish_sketch_geometry()

    window._begin_wire_editor(
        None,
        original_recipe=None,
        part_name="一维部件",
    )
    wire = window._wire_editor_controller
    assert wire is not None
    wire.add_point("P1", 0.0, 0.0, 0.0)
    wire.add_point("P2", 1.0, 0.0, 0.0)
    wire.add_member("M1", "P1", "P2")
    window.finish_wire_geometry()

    before_cancel = tuple(window.document.parts)
    window._begin_sketch_editor(
        None,
        original_recipe=None,
        part_name="不会创建的部件",
    )
    window.cancel_sketch_geometry()

    assert tuple(part.id for part in window.document.parts) == ("P1", "P2")
    assert tuple(part.name for part in window.document.parts) == (
        "二维部件",
        "一维部件",
    )
    assert tuple(part.dimension for part in window.document.parts) == (2, 1)
    assert tuple(window.document.parts) == before_cancel
    window.close()


def test_suppressed_tree_part_never_edits_or_deletes_active_part(
    monkeypatch,
) -> None:
    _application()
    window = FEMMainWindow()
    window._create_native_model("模型-1")
    window.session.add_native_part(
        RectangleGeometry("草图-1", 1.0, 1.0),
        name="源部件",
    )
    window.session.add_native_part(
        RectangleGeometry("草图-2", 1.0, 1.0),
        name="结果部件",
    )
    window._rebuild_full_projection()
    window._apply_session_delta(window.session.suppress_native_part("P1"))
    calls: list[str] = []
    monkeypatch.setattr(window, "_show_error", lambda *_args: None)
    monkeypatch.setattr(
        window,
        "show_geometry_manager",
        lambda: calls.append("edit"),
    )
    monkeypatch.setattr(
        window,
        "delete_geometry",
        lambda: calls.append("delete"),
    )

    window._edit_tree_entry("part", "P1")
    window._delete_tree_entry("part", "P1")

    assert calls == []
    assert window.document.active_part_id == "P2"
    assert tuple(part.id for part in window.document.parts) == ("P1", "P2")
    window.close()


def test_suppressed_source_parts_can_be_shown_as_unpickable_ghosts() -> None:
    _application()
    window = FEMMainWindow()
    window._create_native_model("模型-1")
    window._apply_session_delta(
        window.session.add_native_part(
            BoxGeometry("源实体", 1.0, 1.0, 1.0),
            name="源部件",
        )
    )
    window._apply_session_delta(
        window.session.add_native_part(
            BoxGeometry("结果实体", 1.0, 1.0, 1.0),
            name="结果部件",
        )
    )
    window._apply_session_delta(
        window.session.suppress_native_part("P1")
    )

    action = window.actions["suppressed_part_ghosts"]
    assert action.isEnabled()
    assert action.text() == "显示已抑制源部件"
    action.trigger()

    ghost = window.viewport._geometry_ghost_preview
    active = window.viewport._geometry_preview
    assert ghost is not None
    assert set(ghost.face_part_ids) == {"P1"}
    assert active is not None
    assert set(active.face_part_ids) == {"P2"}
    assert all(
        "P1" not in reference.logical_id
        for reference in window.viewport._geometry_pick_to_ref.values()
    )
    window.close()
