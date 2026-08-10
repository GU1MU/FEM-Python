from __future__ import annotations

import os
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from fem.application import NativePart
from fem.geometry import (
    BoxGeometry,
    CylinderGeometry,
    LogicalEntityRef,
    MovedGeometry,
    SketchCircle,
    SketchGeometry,
    SketchLine,
    SketchPlane,
    SketchPoint,
)
from fem_gui.geometry_preview import GeometryPreview, build_geometry_preview
from fem_gui.part_geometry_preview import build_multi_part_geometry_preview
from fem_gui.widgets import viewport as viewport_module
from fem_gui.widgets.viewport import (
    FEMViewport,
    PickHit,
    SketchDraftRenderData,
    WireDraftRenderData,
    _geometry_edge_polydata,
    _geometry_point_polydata,
    _geometry_surface_polydata,
    _line_only_polydata,
    _sketch_constraint_label_options,
    _sketch_geometry_color,
    _wire_coordinate_label,
)

pytestmark = pytest.mark.optional_runtime

pv = pytest.importorskip(
    "pyvista",
    reason="[optional-native-runtime] PyVista is unavailable",
)


def test_sketch_constraint_labels_use_high_contrast_colors() -> None:
    normal = _sketch_constraint_label_options(warning=False)
    warning = _sketch_constraint_label_options(warning=True)

    assert normal["shape_color"] == "#fff8e1"
    assert normal["text_color"] == "#263238"
    assert normal["font_size"] == 12
    assert normal["margin"] == 5
    assert normal["always_visible"] is True
    assert warning["shape_color"] == "#ffebee"
    assert warning["text_color"] == "#b71c1c"


def _application() -> QApplication:
    return QApplication.instance() or QApplication([])


def _rendered_viewport() -> tuple[FEMViewport, object]:
    _application()
    plotter = pv.Plotter(off_screen=True, window_size=(400, 400))
    plotter.camera_position = [(0.0, 0.0, 3.0), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)]
    plotter.render()
    viewport = FEMViewport()
    viewport._plotter = SimpleNamespace(
        renderer=plotter.renderer,
        height=lambda: 400,
        width=lambda: 400,
        _getPixelRatio=lambda: 1.0,
        devicePixelRatioF=lambda: 1.0,
    )
    return viewport, plotter


def test_isometric_view_keeps_z_up_with_real_vtk() -> None:
    _application()
    plotter = pv.Plotter(off_screen=True, window_size=(400, 400))
    viewport = FEMViewport()
    viewport._plotter = plotter
    viewport._grid = SimpleNamespace(
        points=np.asarray(((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0)))
    )

    viewport.set_view("iso")

    display = viewport._world_points_to_display(
        np.asarray(
            (
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, 0.0, 1.0),
            )
        )
    )
    assert display is not None
    origin, positive_x, positive_y, positive_z = display
    assert positive_x[0] < origin[0] < positive_y[0]
    assert positive_z[0] == pytest.approx(origin[0])
    assert positive_z[1] > origin[1]
    assert positive_x[1] < origin[1]
    assert positive_y[1] < origin[1]
    plotter.close()
    viewport.close()


def test_line_only_polydata_does_not_create_vertex_cells() -> None:
    points = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.0, 1.0, 0.0),
        )
    )

    dataset = _line_only_polydata(pv, points, (2, 0, 1, 2, 1, 2))

    assert dataset.n_points == 3
    assert dataset.n_verts == 0
    assert dataset.n_lines == 2
    assert dataset.n_cells == 2


def test_qt_to_vtk_position_has_no_high_dpi_one_pixel_offset() -> None:
    _application()
    viewport = FEMViewport()
    viewport._plotter = SimpleNamespace(
        height=lambda: 300,
        _getPixelRatio=lambda: 2.0,
    )

    assert viewport._qt_to_vtk_position(10.0, 20.0) == (20, 559)


def test_wire_hover_coordinate_label_has_visible_separators() -> None:
    assert _wire_coordinate_label((0.3, 0.3, 0.0)) == "(0.30, 0.30, 0.00)"


def test_sketch_draft_picking_resolves_point_curve_and_profile() -> None:
    _application()
    viewport = FEMViewport()
    viewport._sketch_draft_render_data = SketchDraftRenderData(
        (
            (0.0, 0.0, 0.0),
            (10.0, 0.0, 0.0),
            (10.0, 10.0, 0.0),
            (0.0, 10.0, 0.0),
        ),
        ("P1", "P2", "P3", "P4"),
        ((0, 1), (1, 2), (2, 3), (3, 0)),
        ("L1", "L2", "L3", "L4"),
        ((0, 1, 2), (0, 2, 3)),
        ("profile/outer", "profile/outer"),
    )
    viewport._world_points_to_display = lambda points: np.column_stack(
        (points[:, :2], np.full(len(points), 0.5))
    )
    viewport._device_pixel_ratio = lambda: 1.0

    assert viewport._sketch_point_at(0, 0) == "P1"
    assert viewport._sketch_curve_at(5, 0) == "L1"
    assert viewport._sketch_profile_at(5, 5) == "profile/outer"
    viewport.close()


def test_sketch_authoring_click_emits_stable_draft_ids() -> None:
    _application()
    viewport = FEMViewport()
    viewport._sketch_authoring_mode = "select"
    viewport._sketch_point_at = lambda _x, _y: None
    viewport._sketch_curve_at = lambda _x, _y: "L7"
    selected: list[str] = []
    viewport.sketchDraftCurveSelected.connect(selected.append)

    viewport._sketch_authoring_click(10, 20)

    assert selected == ["L7"]
    viewport.close()


def test_empty_sketch_shows_xy_axes_origin_and_cursor_coordinates(
    monkeypatch,
) -> None:
    _application()
    plotter = pv.Plotter(off_screen=True, window_size=(400, 400))
    viewport = FEMViewport()
    viewport._plotter = plotter
    viewport._ensure_plotter = lambda: True
    viewport._sketch_grid_spacing = 1.0
    viewport._sketch_draft_render_data = SketchDraftRenderData(
        (),
        (),
        (),
        (),
    )
    monkeypatch.setattr(viewport_module, "_pyvista", pv)
    monkeypatch.setattr(
        viewport_module,
        "is_offscreen_environment",
        lambda: False,
    )

    viewport._show_sketch_draft(render=True, reset_camera=False)
    viewport._set_sketch_authoring_preview_point((1.25, -2.5, 0.0))

    assert "sketch_work_plane_grid" in viewport._actors
    assert "sketch_work_plane_axis_0" in viewport._actors
    assert "sketch_work_plane_axis_1" in viewport._actors
    assert "sketch_work_plane_origin" in viewport._actors
    assert "sketch_work_plane_axis_labels" in viewport._actors
    assert "sketch_authoring_hover" in viewport._actors
    assert "sketch_authoring_hover_label" in viewport._actors
    hover_property = viewport._actors["sketch_authoring_hover"].GetProperty()
    assert hover_property.GetPointSize() == 9.0
    assert hover_property.GetColor() == pytest.approx((1.0, 0.8353, 0.3098), abs=1e-3)
    plotter.close()
    viewport.close()


def test_entering_xy_sketch_places_positive_x_right_and_positive_y_up(
    monkeypatch,
) -> None:
    _application()
    plotter = pv.Plotter(off_screen=True, window_size=(400, 400))
    viewport = FEMViewport()
    viewport._plotter = plotter
    viewport._ensure_plotter = lambda: True
    viewport._sketch_grid_spacing = 1.0
    viewport._sketch_draft_render_data = SketchDraftRenderData(
        (),
        (),
        (),
        (),
    )
    monkeypatch.setattr(viewport_module, "_pyvista", pv)
    monkeypatch.setattr(viewport_module, "is_offscreen_environment", lambda: False)

    viewport._show_sketch_draft(render=True, reset_camera=True)

    display = viewport._world_points_to_display(
        np.asarray(
            (
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
            )
        )
    )
    assert display is not None
    origin, positive_x, positive_y = display
    assert positive_x[0] > origin[0]
    assert positive_y[1] > origin[1]
    plotter.close()
    viewport.close()


def test_sketch_constraint_state_selection_and_hover_have_distinct_colors(
    monkeypatch,
) -> None:
    _application()
    plotter = pv.Plotter(off_screen=True, window_size=(400, 400))
    viewport = FEMViewport()
    viewport._plotter = plotter
    viewport._ensure_plotter = lambda: True
    viewport._sketch_authoring_active = True
    viewport._sketch_draft_render_data = SketchDraftRenderData(
        ((0.0, 0.0, 0.0), (2.0, 0.0, 0.0)),
        ("P1", "P2"),
        ((0, 1),),
        ("L1",),
        constraint_status="fully_constrained",
    )
    monkeypatch.setattr(viewport_module, "_pyvista", pv)
    monkeypatch.setattr(viewport_module, "is_offscreen_environment", lambda: False)

    viewport._show_sketch_draft(render=False)
    assert _sketch_geometry_color("under_constrained") == "#1976a8"
    assert _sketch_geometry_color("fully_constrained") == "#2e7d32"
    assert viewport._actors["sketch_draft_curves"].GetProperty().GetColor() == pytest.approx(
        (0.1804, 0.4902, 0.1961), abs=1e-3
    )

    viewport._set_sketch_entity_hover("curve", "L1")
    hover = viewport._actors["sketch_entity_hover"].GetProperty()
    assert hover.GetLineWidth() == 6.0
    assert hover.GetColor() == pytest.approx((1.0, 0.8353, 0.3098), abs=1e-3)

    viewport.update_sketch_selection("curve", ("L1",))
    selected = viewport._actors["sketch_authoring_selection"].GetProperty()
    assert selected.GetLineWidth() == 8.0
    assert selected.GetColor() == pytest.approx((1.0, 0.7020, 0.0), abs=1e-3)

    viewport.begin_sketch_constraint_selection()
    viewport.set_sketch_constraint_selection(
        (("point", "P1"), ("curve", "L1"))
    )
    assert "sketch_constraint_selection_points" in viewport._actors
    assert "sketch_constraint_selection_curves" in viewport._actors
    plotter.close()
    viewport.close()


def test_sketch_second_point_preview_actor_is_cleared_on_cancel(
    monkeypatch,
) -> None:
    _application()
    plotter = pv.Plotter(off_screen=True, window_size=(400, 400))
    viewport = FEMViewport()
    viewport._plotter = plotter
    viewport._ensure_plotter = lambda: True
    viewport._sketch_authoring_active = True
    viewport._sketch_authoring_mode = "rectangle"
    monkeypatch.setattr(viewport_module, "_pyvista", pv)
    monkeypatch.setattr(
        viewport_module,
        "is_offscreen_environment",
        lambda: False,
    )

    viewport.set_sketch_pending_points(((0.0, 0.0, 0.0),))
    viewport._set_sketch_authoring_preview_point((1.0, 0.5, 0.0))
    assert "sketch_authoring_shape_preview" in viewport._actors

    assert viewport.cancel_pending_sketch_interaction()
    assert "sketch_authoring_shape_preview" not in viewport._actors

    viewport._sketch_authoring_mode = "circle"
    viewport.set_sketch_pending_points(((0.0, 0.0, 0.0),))
    viewport._set_sketch_authoring_preview_point((0.0, 1.0, 0.0))
    assert "sketch_authoring_shape_preview" in viewport._actors

    viewport.stop_sketch_authoring()
    assert "sketch_authoring_shape_preview" not in viewport._actors
    plotter.close()
    viewport.close()


def test_single_wire_point_and_selection_render_as_a_highlight(monkeypatch) -> None:
    _application()
    plotter = pv.Plotter(off_screen=True, window_size=(400, 400))
    viewport = FEMViewport()
    viewport._plotter = plotter
    viewport._ensure_plotter = lambda: True
    viewport._wire_work_plane = "XY"
    viewport._wire_plane_offset = 0.0
    viewport._wire_grid_spacing = 0.1
    viewport._wire_draft_render_data = WireDraftRenderData(
        ((0.2, 0.7, 0.0), (1.2, 0.7, 0.0)),
        ("P1", "P2"),
        ((0, 1),),
        ("M1",),
    )
    viewport._wire_authoring_selection = ("point", "P1")
    monkeypatch.setattr(viewport_module, "_pyvista", pv)
    monkeypatch.setattr(viewport_module, "is_offscreen_environment", lambda: False)

    viewport._show_wire_draft(render=True, reset_camera=False)

    assert "wire_work_plane_grid" in viewport._actors
    assert "wire_work_plane_axis_0" in viewport._actors
    assert "wire_work_plane_axis_1" in viewport._actors
    assert "wire_work_plane_origin" in viewport._actors
    assert "wire_draft_members" in viewport._actors
    assert "wire_draft_points" in viewport._actors
    assert "wire_authoring_selection" in viewport._actors
    assert "wire_authoring_selection_label" in viewport._actors
    viewport._set_wire_authoring_hover(("point", "P2"))
    assert "wire_authoring_hover_outline" in viewport._actors
    assert "wire_authoring_hover" in viewport._actors
    assert "wire_authoring_hover_label" in viewport._actors
    viewport._set_wire_authoring_hover(
        None,
        preview_point=(0.0, 0.0, 0.0),
    )
    assert "wire_authoring_hover_outline" in viewport._actors
    assert "wire_authoring_hover" in viewport._actors
    assert "wire_authoring_hover_label" in viewport._actors
    plotter.close()


def test_triangulated_geometry_faces_preserve_logical_ids() -> None:
    preview = build_geometry_preview(BoxGeometry("box", 1.0, 1.0, 1.0))
    surface = _geometry_surface_polydata(
        pv,
        np.asarray(preview.points, dtype=float),
        preview,
        tuple(range(1, len(preview.faces) + 1)),
    )

    ids = np.asarray(surface.cell_data["geometry_pick_id"], dtype=np.int64)
    assert surface.n_cells == 12
    assert len(set(ids)) == 6
    assert all(np.count_nonzero(ids == pick_id) == 2 for pick_id in set(ids))


def test_multi_part_polydata_carries_part_identity() -> None:
    preview = build_multi_part_geometry_preview(
        (
            NativePart(
                id="P1",
                name="部件-1",
                geometry_recipe=BoxGeometry("实体-1", 1.0, 1.0, 1.0),
            ),
            NativePart(
                id="P2",
                name="部件-2",
                geometry_recipe=MovedGeometry(
                    BoxGeometry("实体-2", 1.0, 1.0, 1.0),
                    2.0,
                    0.0,
                    0.0,
                ),
            ),
        )
    )
    points = np.asarray(preview.points, dtype=float)
    surface = _geometry_surface_polydata(
        pv,
        points,
        preview,
        tuple(range(1, len(preview.faces) + 1)),
    )
    edges = _geometry_edge_polydata(
        pv,
        points,
        preview,
        tuple(range(1, len(preview.edges) + 1)),
    )
    vertices = _geometry_point_polydata(
        pv,
        points,
        preview,
        tuple(range(1, len(preview.points) + 1)),
    )

    assert set(surface.cell_data["geometry_part_id"]) == {"P1", "P2"}
    assert set(edges.cell_data["geometry_part_id"]) == {"P1", "P2"}
    assert set(vertices.point_data["geometry_part_id"]) == {"P1", "P2"}


def test_preview_cells_without_logical_ids_are_not_selectable() -> None:
    preview = GeometryPreview(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        ((0, 1, 2),),
        ((0, 1),),
    )
    points = np.asarray(preview.points, dtype=float)
    surface = _geometry_surface_polydata(pv, points, preview, (0,))
    edges = _geometry_edge_polydata(pv, points, preview, (0,))
    vertices = _geometry_point_polydata(
        pv,
        points,
        preview,
        (0,) * len(preview.points),
    )

    assert set(surface.cell_data["geometry_pick_id"]) == {0}
    assert set(edges.cell_data["geometry_pick_id"]) == {0}
    assert vertices.n_points == 0


def test_face_pick_returns_frontmost_visible_logical_face() -> None:
    viewport, plotter = _rendered_viewport()
    try:
        surface = pv.Cube().triangulate()
        centers = np.asarray(surface.cell_centers().points)
        logical_ids = np.full(surface.n_cells, 300, dtype=np.int64)
        logical_ids[centers[:, 2] > 0.49] = 101
        logical_ids[centers[:, 2] < -0.49] = 202
        surface.cell_data["geometry_pick_id"] = logical_ids
        plotter.add_mesh(surface)
        plotter.render()
        viewport._geometry_preview_surface = surface
        viewport._selection_mode = "geometry_face"

        hit = viewport._resolve_pick(200, 200)

        assert hit is not None
        assert hit.pick_id == 101
        assert hit.world_position[2] == pytest.approx(0.5)
    finally:
        plotter.close()
        viewport.close()


def test_node_and_element_picks_read_discontinuous_dataset_ids() -> None:
    viewport, plotter = _rendered_viewport()
    try:
        nodes = pv.PolyData(np.asarray(((0.0, 0.0, 0.0), (0.8, 0.0, 0.0))))
        nodes.point_data["node_id"] = np.asarray((50, 900), dtype=np.int64)
        viewport._selection_mode = "node"
        node_hit = viewport._pick_screen_point(
            200,
            200,
            nodes,
            "node_id",
            "nodes",
            None,
            8.0,
        )

        element = pv.PolyData(
            np.asarray(((-1.0, -1.0, 0.0), (1.0, -1.0, 0.0), (0.0, 1.0, 0.0))),
            faces=np.asarray((3, 0, 1, 2), dtype=np.int64),
        )
        element.cell_data["element_id"] = np.asarray((500,), dtype=np.int64)
        viewport._selection_mode = "element"
        element_hit = viewport._pick_cell(
            200,
            200,
            element,
            "element_id",
            "elements",
            "element",
        )

        assert node_hit is not None and node_hit.pick_id == 50
        assert element_hit is not None and element_hit.pick_id == 500
    finally:
        plotter.close()
        viewport.close()


def test_edge_pick_tolerance_is_stable_in_display_pixels_across_zoom() -> None:
    viewport, plotter = _rendered_viewport()
    try:
        preview = GeometryPreview(
            ((-1.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
            (),
            ((0, 1),),
            edge_logical_ids=("edge:test",),
        )
        points = np.asarray(preview.points, dtype=float)
        viewport._geometry_preview = preview
        viewport._install_geometry_pick_bindings(preview)
        pick_id = viewport._geometry_edge_pick_ids[0]
        viewport._geometry_preview_edges = _geometry_edge_polydata(
            pv,
            points,
            preview,
            viewport._geometry_edge_pick_ids,
        )
        viewport._geometry_preview_surface = None
        viewport._selection_mode = "geometry_edge"
        camera = plotter.renderer.GetActiveCamera()
        camera.ParallelProjectionOn()

        hits = []
        for scale in (2.0, 20.0):
            camera.SetParallelScale(scale)
            plotter.render()
            midpoint = viewport._world_to_display(np.zeros(3))
            hits.append(
                viewport._pick_screen_edge(
                    int(round(midpoint[0])),
                    int(round(midpoint[1] + 5.0)),
                    6.0,
                )
            )

        assert [hit.pick_id if hit else None for hit in hits] == [
            pick_id,
            pick_id,
        ]
    finally:
        plotter.close()
        viewport.close()


def test_click_reuses_current_preselection_candidate(monkeypatch) -> None:
    _application()
    viewport = FEMViewport()
    viewport._plotter = SimpleNamespace(
        height=lambda: 400,
        _getPixelRatio=lambda: 1.0,
        devicePixelRatioF=lambda: 1.0,
    )
    viewport._selection_mode = "geometry_face"
    monkeypatch.setattr(
        viewport,
        "_resolve_pick",
        lambda *_args: pytest.fail("click must reuse the hover candidate"),
    )
    preview = GeometryPreview(
        ((0.0, 0.0, 0.0),) * 3,
        ((0, 1, 2),),
        (),
        face_logical_ids=("face:front",),
    )
    viewport._install_geometry_pick_bindings(preview)
    viewport._hover_hit = PickHit(
        "geometry_face",
        viewport._geometry_face_pick_ids[0],
        "geometry_surface",
        (100.0, 100.0),
        (0.0, 0.0, 0.0),
        vtk_cell_id=4,
    )
    picked: list[LogicalEntityRef] = []
    fem_picks: list[tuple[str, int]] = []
    viewport.geometryEntityPicked.connect(picked.append)
    viewport.entityPicked.connect(
        lambda kind, key: fem_picks.append((kind, key))
    )

    viewport._pick_qt_position(100.0, 299.0)

    assert picked == [LogicalEntityRef("face:front")]
    assert fem_picks == []
    viewport.close()


@pytest.mark.parametrize(
    ("kind", "logical_id"),
    (
        ("point", "point:bottom-front-left"),
        ("edge", "edge:bottom-front"),
        ("face", "face:bottom"),
        ("body", "body:domain"),
    ),
)
def test_geometry_pick_signal_emits_logical_reference(
    kind,
    logical_id,
) -> None:
    _application()
    viewport = FEMViewport()
    preview = build_geometry_preview(
        BoxGeometry("box", 1.0, 1.0, 1.0)
    )
    viewport._install_geometry_pick_bindings(preview)
    reference = LogicalEntityRef(logical_id)
    pick_id = viewport._geometry_ref_to_pick_ids[reference][0]
    picked = []
    fem_picks = []
    viewport.geometryEntityPicked.connect(picked.append)
    viewport.entityPicked.connect(
        lambda entity_kind, key: fem_picks.append((entity_kind, key))
    )

    viewport._submit_pick(
        PickHit(
            f"geometry_{kind}",
            pick_id,
            "geometry_preview",
            (0.0, 0.0),
            (0.0, 0.0, 0.0),
        )
    )

    assert picked == [reference]
    assert fem_picks == []
    viewport.close()


def test_viewport_allocates_private_tokens_per_display_cell() -> None:
    _application()
    viewport = FEMViewport()
    preview = build_geometry_preview(
        CylinderGeometry("cylinder", 1.0, 2.0),
        segments=8,
    )

    viewport._install_geometry_pick_bindings(preview)

    outer = LogicalEntityRef("face:outer")
    outer_tokens = viewport._geometry_ref_to_pick_ids[outer]
    outer_display_cells = preview.face_logical_ids.count("face:outer")
    assert len(outer_tokens) == outer_display_cells
    assert len(set(outer_tokens)) == outer_display_cells
    assert all(
        viewport._geometry_pick_to_ref[token] == outer
        for token in outer_tokens
    )
    assert all(
        token == 0
        for token, logical_id in zip(
            viewport._geometry_point_pick_ids,
            preview.point_logical_ids,
            strict=True,
        )
        if logical_id is None
    )
    viewport.close()


def test_strict_sketch_preview_exposes_profile_hole_and_alias_pick_ids() -> None:
    _application()
    recipe = SketchGeometry(
        "strict-preview",
        SketchPlane.xy(),
        (
            SketchPoint("P1", 0.0, 0.0),
            SketchPoint("P2", 4.0, 0.0),
            SketchPoint("P3", 4.0, 3.0),
            SketchPoint("P4", 0.0, 3.0),
            SketchPoint("P5", 2.0, 1.5),
        ),
        (
            SketchLine("L1", "P1", "P2"),
            SketchLine("L2", "P2", "P3"),
            SketchLine("L3", "P3", "P4"),
            SketchLine("L4", "P4", "P1"),
            SketchCircle("C1", "P5", 0.5),
        ),
    )

    preview = build_geometry_preview(recipe, segments=24)
    viewport = FEMViewport()
    viewport._install_geometry_pick_bindings(preview)

    assert preview.faces
    assert "edge:C1" in preview.edge_logical_ids
    assert "edge:hole-loop" in preview.edge_logical_ids
    assert "edge:outer-loop" in preview.edge_logical_ids
    assert "face:domain" in preview.face_logical_ids
    assert any(
        logical_id is not None and logical_id.startswith("face:profile/")
        for logical_id in preview.face_logical_ids
    )
    assert LogicalEntityRef("point:P1") in viewport._geometry_ref_to_pick_ids
    assert LogicalEntityRef("edge:C1") in viewport._geometry_ref_to_pick_ids
    assert LogicalEntityRef("face:domain") in viewport._geometry_ref_to_pick_ids
    assert LogicalEntityRef("body:domain") in viewport._geometry_ref_to_pick_ids
    viewport.close()


def test_preview_install_rebuilds_private_pick_maps() -> None:
    _application()
    viewport = FEMViewport()
    first = build_geometry_preview(
        CylinderGeometry("cylinder", 1.0, 2.0),
        segments=8,
    )
    viewport._install_geometry_pick_bindings(first)
    assert LogicalEntityRef("face:outer") in viewport._geometry_ref_to_pick_ids

    second = GeometryPreview(
        points=((0.0, 0.0, 0.0),) * 3,
        faces=((0, 1, 2),),
        edges=(),
        face_logical_ids=("face:replacement",),
    )
    viewport._install_geometry_pick_bindings(second)

    replacement = LogicalEntityRef("face:replacement")
    assert set(viewport._geometry_ref_to_pick_ids) == {replacement}
    assert set(viewport._geometry_pick_to_ref.values()) == {replacement}
    viewport.close()


def test_fem_pick_signal_keeps_integer_node_and_element_ids() -> None:
    _application()
    viewport = FEMViewport()
    picked = []
    geometry_picks = []
    viewport.entityPicked.connect(
        lambda kind, key: picked.append((kind, key))
    )
    viewport.geometryEntityPicked.connect(geometry_picks.append)

    for kind, key in (("node", 50), ("element", 500)):
        viewport._submit_pick(
            PickHit(
                kind,
                key,
                "model",
                (0.0, 0.0),
                (0.0, 0.0, 0.0),
            )
        )

    assert picked == [("node", 50), ("element", 500)]
    assert geometry_picks == []
    viewport.close()


def test_logical_reference_highlight_covers_every_display_cell(
    monkeypatch,
) -> None:
    _application()
    preview = build_geometry_preview(
        CylinderGeometry("cylinder", 1.0, 2.0),
        segments=8,
    )
    viewport = FEMViewport()
    viewport._install_geometry_pick_bindings(preview)
    surface = _geometry_surface_polydata(
        pv,
        np.asarray(preview.points, dtype=float),
        preview,
        viewport._geometry_face_pick_ids,
    )
    highlighted = []

    class Plotter:
        def add_mesh(self, data, **_kwargs):
            highlighted.append(data)
            return object()

    viewport._plotter = Plotter()
    viewport._geometry_preview = preview
    viewport._geometry_preview_surface = surface
    monkeypatch.setattr(
        viewport,
        "_clear_beam_frame_preview",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        viewport,
        "_offset_highlight_actor",
        lambda _actor: None,
    )
    monkeypatch.setattr(
        viewport,
        "_update_pickable_actors",
        lambda: None,
    )
    monkeypatch.setattr(viewport, "_render", lambda: None)

    viewport.highlight_geometry(
        LogicalEntityRef("face:outer")
    )

    assert len(highlighted) == 1
    assert highlighted[0].n_cells == (
        2 * preview.face_logical_ids.count("face:outer")
    )
    viewport.close()


def test_auxiliary_actors_are_never_marked_pickable() -> None:
    class Actor:
        def __init__(self) -> None:
            self.pickable = None

        def SetPickable(self, value: bool) -> None:
            self.pickable = bool(value)

    _application()
    viewport = FEMViewport()
    viewport._selection_mode = "geometry_face"
    viewport._actors = {
        name: Actor()
        for name in (
            "geometry_surface",
            "result",
            "element_edges",
            "undeformed_overlay",
            "symbols",
            "selection",
            "preselection",
        )
    }

    viewport._update_pickable_actors()

    assert viewport._actors["geometry_surface"].pickable
    assert all(
        not actor.pickable
        for name, actor in viewport._actors.items()
        if name != "geometry_surface"
    )
    viewport.close()


@pytest.mark.parametrize("kind", ("geometry_face", "geometry_body"))
def test_surface_preselection_never_exposes_internal_triangulation(
    monkeypatch,
    kind,
) -> None:
    _application()
    preview = build_geometry_preview(BoxGeometry("box", 1.0, 1.0, 1.0))
    viewport = FEMViewport()
    viewport._install_geometry_pick_bindings(preview)
    surface = _geometry_surface_polydata(
        pv,
        np.asarray(preview.points, dtype=float),
        preview,
        viewport._geometry_face_pick_ids,
    )
    calls: list[dict[str, object]] = []

    class Actor:
        def SetPickable(self, _value):
            pass

    class Plotter:
        def add_mesh(self, _data, **kwargs):
            calls.append(kwargs)
            return Actor()

    viewport._plotter = Plotter()
    viewport._geometry_preview_surface = surface
    monkeypatch.setattr(viewport_module, "_pyvista", pv)
    monkeypatch.setattr(viewport, "_remove_actor", lambda _name: None)
    monkeypatch.setattr(viewport, "_offset_highlight_actor", lambda _actor: None)
    monkeypatch.setattr(viewport, "_render", lambda: None)
    pick_id = (
        viewport._geometry_face_pick_ids[0]
        if kind == "geometry_face"
        else viewport._geometry_body_pick_id
    )
    viewport._show_preselection(
        PickHit(
            kind,
            pick_id,
            "geometry_surface",
            (100.0, 100.0),
            (0.0, 0.0, 0.0),
            vtk_cell_id=0,
        )
    )

    assert calls[-1]["show_edges"] is False
    viewport.close()


def test_surface_preselection_highlights_every_cell_of_logical_face(
    monkeypatch,
) -> None:
    _application()
    preview = GeometryPreview(
        points=(
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.0, 1.0, 0.0),
            (0.0, 1.0, 0.0),
        ),
        faces=((0, 1, 2), (0, 2, 3)),
        edges=(),
        face_logical_ids=("face:domain", "face:domain"),
        topological_dimension=2,
    )
    viewport = FEMViewport()
    viewport._install_geometry_pick_bindings(preview)
    viewport._geometry_preview_surface = _geometry_surface_polydata(
        pv,
        np.asarray(preview.points, dtype=float),
        preview,
        viewport._geometry_face_pick_ids,
    )
    highlighted = []

    class Actor:
        def SetPickable(self, _value):
            pass

    class Plotter:
        def add_mesh(self, data, **_kwargs):
            highlighted.append(data)
            return Actor()

    viewport._plotter = Plotter()
    monkeypatch.setattr(viewport_module, "_pyvista", pv)
    monkeypatch.setattr(viewport, "_remove_actor", lambda _name: None)
    monkeypatch.setattr(
        viewport,
        "_offset_highlight_actor",
        lambda _actor: None,
    )
    monkeypatch.setattr(viewport, "_render", lambda: None)

    viewport._show_preselection(
        PickHit(
            "geometry_face",
            viewport._geometry_face_pick_ids[0],
            "geometry_surface",
            (100.0, 100.0),
            (0.0, 0.0, 0.0),
            vtk_cell_id=0,
        )
    )

    assert len(highlighted) == 1
    assert highlighted[0].n_cells == 2
    viewport.close()
