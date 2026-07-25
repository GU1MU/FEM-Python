from __future__ import annotations

import os
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

pv = pytest.importorskip("pyvista")

from fem_gui.preprocessing import BoxGeometry, GeometryPreview, build_geometry_preview
from fem_gui.widgets import viewport as viewport_module
from fem_gui.widgets.viewport import (
    FEMViewport,
    PickHit,
    _geometry_edge_polydata,
    _geometry_surface_polydata,
)


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


def test_triangulated_geometry_faces_preserve_logical_ids() -> None:
    preview = build_geometry_preview(BoxGeometry("box", 1.0, 1.0, 1.0))
    surface = _geometry_surface_polydata(
        pv,
        np.asarray(preview.points, dtype=float),
        preview,
    )

    ids = np.asarray(surface.cell_data["geometry_entity_id"], dtype=np.int64)
    assert surface.n_cells == 12
    assert set(ids) == set(range(1, 7))
    assert all(np.count_nonzero(ids == entity_id) == 2 for entity_id in range(1, 7))


def test_face_pick_returns_frontmost_visible_logical_face() -> None:
    viewport, plotter = _rendered_viewport()
    try:
        surface = pv.Cube().triangulate()
        centers = np.asarray(surface.cell_centers().points)
        logical_ids = np.full(surface.n_cells, 300, dtype=np.int64)
        logical_ids[centers[:, 2] > 0.49] = 101
        logical_ids[centers[:, 2] < -0.49] = 202
        surface.cell_data["geometry_entity_id"] = logical_ids
        plotter.add_mesh(surface)
        plotter.render()
        viewport._geometry_preview_surface = surface
        viewport._selection_mode = "geometry_face"

        hit = viewport._resolve_pick(200, 200)

        assert hit is not None
        assert hit.entity_id == 101
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

        assert node_hit is not None and node_hit.entity_id == 50
        assert element_hit is not None and element_hit.entity_id == 500
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
            (),
            (77,),
        )
        points = np.asarray(preview.points, dtype=float)
        viewport._geometry_preview = preview
        viewport._geometry_preview_edges = _geometry_edge_polydata(pv, points, preview)
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

        assert [hit.entity_id if hit else None for hit in hits] == [77, 77]
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
    viewport._hover_hit = PickHit(
        "geometry_face",
        12,
        "geometry_surface",
        (100.0, 100.0),
        (0.0, 0.0, 0.0),
        vtk_cell_id=4,
    )
    monkeypatch.setattr(
        viewport,
        "_resolve_pick",
        lambda *_args: pytest.fail("click must reuse the hover candidate"),
    )
    picked: list[tuple[str, int]] = []
    viewport.entityPicked.connect(lambda kind, key: picked.append((kind, key)))

    viewport._pick_qt_position(100.0, 299.0)

    assert picked == [("geometry_face", 12)]
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
    surface = _geometry_surface_polydata(
        pv,
        np.asarray(preview.points, dtype=float),
        preview,
    )
    calls: list[dict[str, object]] = []

    class Actor:
        def SetPickable(self, _value):
            pass

    class Plotter:
        def add_mesh(self, _data, **kwargs):
            calls.append(kwargs)
            return Actor()

    viewport = FEMViewport()
    viewport._plotter = Plotter()
    viewport._geometry_preview_surface = surface
    monkeypatch.setattr(viewport_module, "_pyvista", pv)
    monkeypatch.setattr(viewport, "_remove_actor", lambda _name: None)
    monkeypatch.setattr(viewport, "_offset_highlight_actor", lambda _actor: None)
    monkeypatch.setattr(viewport, "_render", lambda: None)
    viewport._show_preselection(
        PickHit(
            kind,
            1,
            "geometry_surface",
            (100.0, 100.0),
            (0.0, 0.0, 0.0),
            vtk_cell_id=0,
        )
    )

    assert calls[-1]["show_edges"] is False
    viewport.close()
