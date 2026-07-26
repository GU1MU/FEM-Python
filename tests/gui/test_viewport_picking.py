from __future__ import annotations

import os
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from fem.geometry import BoxGeometry, CylinderGeometry, LogicalEntityRef
from fem_gui.geometry_preview import GeometryPreview, build_geometry_preview
from fem_gui.widgets import viewport as viewport_module
from fem_gui.widgets.viewport import (
    FEMViewport,
    PickHit,
    _geometry_edge_polydata,
    _geometry_point_polydata,
    _geometry_surface_polydata,
)

pytestmark = pytest.mark.optional_runtime

pv = pytest.importorskip(
    "pyvista",
    reason="[optional-native-runtime] PyVista is unavailable",
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
        tuple(range(1, len(preview.faces) + 1)),
    )

    ids = np.asarray(surface.cell_data["geometry_pick_id"], dtype=np.int64)
    assert surface.n_cells == 12
    assert len(set(ids)) == 6
    assert all(np.count_nonzero(ids == pick_id) == 2 for pick_id in set(ids))


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
