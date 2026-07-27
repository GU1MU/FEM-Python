from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QDialogButtonBox

from fem.geometry import WireGeometry, WireMember, WirePoint
from fem.mesh.settings import MeshSettings
from fem_gui.geometry_preview import build_geometry_preview
from fem_gui.main_window import FEMMainWindow
from fem_gui.preprocessing_dialogs import MeshControlsDialog, MeshSettingsDialog
from fem_gui.wire_editor import (
    WireDraftController,
    WireDraftValidationError,
    intersect_ray_with_work_plane,
    snap_work_plane_point,
)


def _application() -> QApplication:
    return QApplication.instance() or QApplication([])


def test_wire_draft_is_detached_and_preserves_named_graph_identity() -> None:
    controller = WireDraftController(name="Draft")
    first = controller.add_point()
    second = controller.add_point(x=1.0, y=0.0, z=2.0)
    member = controller.add_member(start=first.name, end=second.name)

    snapshot = controller.snapshot()
    controller.rename_point(first.name, "Joint-A")
    assert snapshot.points[0].name == first.name
    assert controller.snapshot().members[0].start == "Joint-A"
    assert controller.snapshot().members[0].name == member.name
    assert controller.to_geometry().points[1].z == 2.0


def test_wire_draft_finish_diagnostics_cover_incomplete_and_duplicate_edges() -> None:
    controller = WireDraftController(name="Wire")
    controller.add_point("P1")
    controller.add_point("P2", 1.0)
    controller.add_point("P3", 2.0)
    controller.add_member("M1", "P1", "P2")
    controller.add_member("M2", "P2", "P1")

    codes = {item.code for item in controller.finish_diagnostics()}
    assert "member.endpoint.duplicate" in codes
    assert "point.unused" in codes
    try:
        controller.to_geometry()
    except WireDraftValidationError as error:
        assert error.diagnostics
    else:
        raise AssertionError("invalid wire draft unexpectedly serialized")


def test_work_plane_intersection_and_snap_preserve_fixed_coordinate() -> None:
    assert intersect_ray_with_work_plane((0, 0, -2), (2, 4, 2), "XY", 0.0) == (
        1.0,
        2.0,
        0.0,
    )
    assert intersect_ray_with_work_plane((0, 0, 0), (1, 0, 0), "XY") is None
    assert snap_work_plane_point((-0.6, 1.4, 3.5), "XY", 1.0) == (
        -1.0,
        1.0,
        3.5,
    )


def test_wire_preview_uses_declared_order_and_face_free_dimension_one_topology() -> None:
    recipe = WireGeometry(
        "Wire",
        (
            WirePoint("P2", 2.0, 0.0, 4.0),
            WirePoint("P1", 0.0, 0.0, 1.0),
        ),
        (WireMember("M1", "P2", "P1"),),
    )
    preview = build_geometry_preview(recipe)
    assert preview.dimension == 1
    assert preview.points == ((2.0, 0.0, 4.0), (0.0, 0.0, 1.0))
    assert preview.edges == ((0, 1),)
    assert preview.point_logical_ids == ("point:P2", "point:P1")
    assert preview.edge_logical_ids == ("edge:M1",)
    assert preview.faces == ()
    assert preview.body_logical_id == "body:domain"


def test_line_mesh_dialog_requires_explicit_formulation_and_controls_preserve_it() -> None:
    _application()
    fresh = MeshSettingsDialog(None, mesh_dimension=1, suggested_size=0.25)
    assert fresh.formulation_combo.currentData() is None
    assert not fresh._buttons.button(QDialogButtonBox.StandardButton.Ok).isEnabled()
    fresh.formulation_combo.setCurrentIndex(
        fresh.formulation_combo.findData("Beam2")
    )
    assert fresh.settings().line_element_type == "Beam2"

    settings = MeshSettings(
        0.25,
        cell_shape="line",
        line_element_type="Truss2",
    )
    dialog = MeshControlsDialog(settings)
    assert dialog.settings().line_element_type == "Truss2"
    assert "Line mesh" in dialog.control_list.item(2).text()


def test_main_window_can_commit_a_wire_after_detached_edit() -> None:
    _application()
    window = FEMMainWindow()
    window.new_native_model()
    assert window.actions["geometry_wire"].isEnabled()
    window.start_wire_geometry()
    controller = window._wire_editor_controller
    assert controller is not None
    assert not window.actions["mesh_settings"].isEnabled()
    assert window.actions["top"].isEnabled()
    controller.add_point("P1", 0.0, 0.0, 0.0)
    controller.add_point("P2", 1.0, 2.0, 3.0)
    controller.add_member("M1", "P1", "P2")
    window.wire_editor_panel._refresh()
    window.finish_wire_geometry()
    assert window._wire_editor_controller is None
    assert isinstance(window.document.geometry_recipe, WireGeometry)
    assert window.document.geometry_recipe.members[0].start == "P1"
    assert window.viewport._geometry_preview.dimension == 1
    window.close()
