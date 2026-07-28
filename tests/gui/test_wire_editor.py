from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtWidgets import QApplication, QDialogButtonBox, QLabel

from fem.geometry import LogicalEntityRef, WireGeometry, WireMember, WirePoint
from fem.mesh.settings import LocalMeshControl, MeshSettings
from fem_gui.geometry_preview import build_geometry_preview
from fem_gui.main_window import FEMMainWindow
from fem_gui.preprocessing_dialogs import MeshControlsDialog, MeshSettingsDialog
from fem_gui.wire_editor import (
    WireDraftController,
    WireDraftValidationError,
    intersect_ray_with_work_plane,
    snap_work_plane_point,
)
from fem_gui.widgets.viewport import (
    FEMViewport,
    WireDraftRenderData,
    _capture_camera_state,
    _restore_camera_state,
    _wire_grid_layout,
)
from fem_gui.widgets.wire_editor_panel import WireEditorPanel


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
    decimal_offset = intersect_ray_with_work_plane(
        (0.0, 0.0, -0.3),
        (0.0, 0.0, 0.7),
        "XY",
        0.1,
    )
    assert decimal_offset is not None
    assert decimal_offset[2] == 0.1
    assert snap_work_plane_point(
        (0.049999999, -0.049999999, 0.0),
        "XY",
        0.1,
    ) == (0.0, 0.0, 0.0)


def test_wire_editor_panel_uses_clear_chinese_actions_without_bottom_explanations() -> None:
    _application()
    panel = WireEditorPanel(WireDraftController())

    assert panel.name_edit.text() == "线体-1"
    assert panel.point_mode_button.text() == "添加点"
    assert panel.member_mode_button.text() == "连接杆件"
    assert panel.select_mode_button.text() == "选择对象"
    assert panel.add_point_button.text() == "新增"
    assert panel.delete_point_button.text() == "删除"
    assert panel.add_member_button.text() == "新增"
    assert panel.delete_member_button.text() == "删除"
    assert panel.offset_spin.decimals() == 2
    assert panel.spacing_spin.decimals() == 2
    assert panel.spacing_spin.minimum() == 0.01
    assert panel.spacing_spin.value() == 0.1
    assert not hasattr(panel, "snap_check")
    form = panel.layout().itemAt(0).layout()
    assert form.labelForField(panel.spacing_spin).text() == "吸附间距"
    assert "工作平面" in panel.point_mode_button.toolTip()
    assert "两个已有点" in panel.member_mode_button.toolTip()
    assert "点或杆件" in panel.select_mode_button.toolTip()
    assert panel.points_table.horizontalHeaderItem(0).text() == "名称"
    assert panel.members_table.horizontalHeaderItem(1).text() == "起点"
    assert panel.members_table.horizontalHeaderItem(2).text() == "终点"
    assert panel.findChild(QLabel, "wireValidationLabel") is None
    assert not hasattr(panel, "hint_label")
    assert not hasattr(panel, "coincident_confirm")


def test_viewport_point_is_snapped_again_before_entering_the_draft() -> None:
    _application()
    controller = WireDraftController()
    panel = WireEditorPanel(controller)

    panel._point_from_viewport((0.049999999, -0.049999999, 1.0e-8))

    created = controller.snapshot().points[0]
    assert (created.x, created.y, created.z) == (0.0, 0.0, 0.0)


def test_single_wire_point_builds_a_snap_aligned_grid_without_an_exception() -> None:
    layout = _wire_grid_layout(
        np.asarray(((0.24, 0.74, 0.0),), dtype=float),
        "XY",
        0.0,
        0.1,
    )

    assert layout.plane_size >= 10.0
    assert np.isclose(
        layout.plane_size / layout.resolution,
        layout.visible_spacing,
    )
    assert np.isclose(layout.visible_spacing / 0.1, 1.0)
    assert np.allclose(layout.center, (0.2, 0.7, 0.0))
    fine_layout = _wire_grid_layout(
        np.asarray(((0.24, 0.74, 0.0),), dtype=float),
        "XY",
        0.0,
        0.01,
    )
    assert np.isclose(fine_layout.visible_spacing, 0.01)
    assert np.isclose(
        fine_layout.plane_size / fine_layout.resolution,
        0.01,
    )


def test_wire_viewport_click_applies_enabled_grid_snapping() -> None:
    _application()
    viewport = FEMViewport()
    viewport._wire_authoring_mode = "point"
    viewport._wire_work_plane = "XY"
    viewport._wire_plane_offset = 0.0
    viewport._wire_grid_snap = True
    viewport._wire_grid_spacing = 0.25
    viewport._display_to_world = lambda _x, _y, depth: (
        0.37,
        0.62,
        -1.0 if depth == 0.0 else 1.0,
    )
    viewport._wire_point_at = lambda _x, _y: None
    selected: list[tuple[float, float, float]] = []
    viewport.wireWorkPlanePointSelected.connect(selected.append)

    viewport._wire_authoring_click(10, 20)

    assert selected == [(0.25, 0.5, 0.0)]


def test_point_hover_preview_and_click_share_the_same_snapped_coordinate() -> None:
    _application()
    viewport = FEMViewport()
    viewport._wire_authoring_mode = "point"
    viewport._wire_work_plane = "XY"
    viewport._wire_plane_offset = 0.0
    viewport._wire_grid_snap = True
    viewport._wire_grid_spacing = 0.1
    viewport._wire_draft_render_data = WireDraftRenderData((), (), (), ())
    viewport._display_to_world = lambda _x, _y, depth: (
        0.049999999,
        -0.049999999,
        -1.0 if depth == 0.0 else 1.0,
    )
    viewport._wire_point_at = lambda _x, _y: None
    viewport._show_wire_authoring_hover = lambda **_options: None
    selected: list[tuple[float, float, float]] = []
    viewport.wireWorkPlanePointSelected.connect(selected.append)

    viewport._update_wire_authoring_hover(10, 20)
    preview = viewport._wire_authoring_preview_point
    viewport._wire_authoring_click(10, 20)

    assert preview == (0.0, 0.0, 0.0)
    assert selected == [preview]


def test_clicking_an_existing_wire_point_immediately_redraws_its_highlight() -> None:
    _application()
    viewport = FEMViewport()
    viewport._wire_authoring_mode = "point"
    viewport._wire_point_at = lambda _x, _y: "P1"
    viewport._display_to_world = lambda _x, _y, depth: (
        0.0,
        0.0,
        -1.0 if depth == 0.0 else 1.0,
    )
    redraws: list[dict[str, bool]] = []
    viewport._show_wire_draft = lambda **options: redraws.append(options)

    viewport._wire_authoring_click(10, 20)

    assert viewport._wire_authoring_selection == ("point", "P1")
    assert redraws == [{"render": True}]


def test_new_and_table_selected_points_request_viewport_highlighting() -> None:
    _application()
    panel = WireEditorPanel(WireDraftController())
    viewport = FEMViewport()
    panel.entityFocusRequested.connect(viewport.focus_wire_draft_entity)
    panel.begin(viewport)
    panel.spacing_spin.setValue(0.25)

    panel.add_point()
    assert viewport._wire_grid_snap
    assert viewport._wire_grid_spacing == 0.25
    assert viewport._wire_authoring_selection == ("point", "P1")
    panel.add_point()
    panel.points_table.clearSelection()
    panel.points_table.selectRow(0)
    assert viewport._wire_authoring_selection == ("point", "P1")

    panel.end()


def test_coincident_named_points_no_longer_require_a_panel_confirmation() -> None:
    _application()
    controller = WireDraftController()
    controller.add_point("P1", 0.0, 0.0)
    controller.add_point("P2", 1.0, 0.0)
    controller.add_point("P3", 0.0, 0.0)
    controller.add_member("M1", "P1", "P2")
    controller.add_member("M2", "P2", "P3")
    panel = WireEditorPanel(controller)

    assert controller.coincident_point_groups() == (("P1", "P3"),)
    assert controller.can_finish
    assert panel.finish_button.isEnabled()


class _WireCamera:
    def __init__(self) -> None:
        self.position = (3.0, 4.0, 5.0)
        self.focal_point = (1.0, 2.0, 0.0)
        self.view_up = (0.0, 1.0, 0.0)
        self.parallel_scale = 2.5
        self.view_angle = 30.0
        self.parallel_projection = 1

    def GetPosition(self):
        return self.position

    def GetFocalPoint(self):
        return self.focal_point

    def GetViewUp(self):
        return self.view_up

    def GetParallelScale(self):
        return self.parallel_scale

    def GetViewAngle(self):
        return self.view_angle

    def GetParallelProjection(self):
        return self.parallel_projection

    def SetPosition(self, *value) -> None:
        self.position = tuple(value)

    def SetFocalPoint(self, *value) -> None:
        self.focal_point = tuple(value)

    def SetViewUp(self, *value) -> None:
        self.view_up = tuple(value)

    def SetParallelScale(self, value) -> None:
        self.parallel_scale = value

    def SetViewAngle(self, value) -> None:
        self.view_angle = value

    def SetParallelProjection(self, value) -> None:
        self.parallel_projection = value

    def OrthogonalizeViewUp(self) -> None:
        pass


class _WirePlotter:
    def __init__(self) -> None:
        self.camera = _WireCamera()
        self.clipping_resets = 0

    def reset_camera_clipping_range(self) -> None:
        self.clipping_resets += 1


def test_wire_draft_redraw_restores_camera_framing_and_zoom() -> None:
    plotter = _WirePlotter()
    expected = _capture_camera_state(plotter)
    assert expected is not None
    plotter.camera.position = (90.0, 80.0, 70.0)
    plotter.camera.focal_point = (60.0, 50.0, 40.0)
    plotter.camera.view_up = (1.0, 0.0, 0.0)
    plotter.camera.parallel_scale = 0.01
    plotter.camera.view_angle = 5.0
    plotter.camera.parallel_projection = 0

    _restore_camera_state(plotter, expected)

    assert _capture_camera_state(plotter) == expected
    assert plotter.clipping_resets == 1


def test_wire_draft_data_updates_explicitly_preserve_the_current_camera() -> None:
    _application()
    viewport = FEMViewport()
    redraws: list[dict[str, bool]] = []
    viewport._wire_authoring_active = True
    viewport._show_wire_draft = lambda **options: redraws.append(options)

    viewport.update_wire_draft(WireDraftRenderData((), (), (), ()))

    assert redraws == [{"render": True, "reset_camera": False}]


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
    assert "线网格" in dialog.control_list.item(2).text()
    assert "单元形式" in dialog.control_list.item(3).text()


def test_line_mesh_dialog_uses_chinese_text_without_policy_description() -> None:
    _application()
    settings = MeshSettings(
        0.25,
        cell_shape="line",
        line_element_type="Truss2",
    )
    dialog = MeshSettingsDialog(
        settings,
        mesh_dimension=1,
        suggested_size=0.25,
    )

    assert not dialog.size_spin.isEnabled()
    assert dialog.method_combo.currentText() == "线网格"
    assert dialog.shape_combo.currentText() == "线网格"
    assert dialog.formulation_combo.currentText() == "Truss2"
    assert not hasattr(dialog, "line_policy_label")
    assert dialog._buttons.button(
        QDialogButtonBox.StandardButton.Ok
    ).isEnabled()

    beam_index = dialog.formulation_combo.findData("Beam2")
    dialog.formulation_combo.setCurrentIndex(beam_index)
    assert dialog.size_spin.isEnabled()
    assert dialog.formulation_combo.currentText() == "Beam2"


def test_truss_mesh_dialog_rejects_legacy_local_controls() -> None:
    _application()
    settings = MeshSettings(
        0.25,
        cell_shape="line",
        local_controls=(
            LocalMeshControl(LogicalEntityRef("edge:M1"), 0.1),
        ),
        line_element_type="Truss2",
    )
    dialog = MeshSettingsDialog(
        settings,
        mesh_dimension=1,
        suggested_size=0.25,
    )

    assert not dialog._buttons.button(
        QDialogButtonBox.StandardButton.Ok
    ).isEnabled()


def test_main_window_can_commit_a_wire_after_detached_edit() -> None:
    _application()
    window = FEMMainWindow()
    window._create_native_model("Model-1")
    assert window.actions["geometry_wire"].isEnabled()
    assert window.actions["geometry_wire"].text() == "新建线体"
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
