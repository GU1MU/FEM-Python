from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pyvista as pv
import pytest
import vtk
from PySide6.QtWidgets import QApplication

from fem.application import MeshEntityRef
from fem.geometry import LogicalEntityRef
from fem_gui.widgets.viewport import (
    FEMViewport,
    PickHit,
    _SelectionRubberBand,
)


def _application() -> QApplication:
    return QApplication.instance() or QApplication([])


def test_selection_rubber_band_is_a_border_only_vtk_overlay() -> None:
    renderer = vtk.vtkRenderer()
    band = _SelectionRubberBand(renderer)

    band.set_rectangle((90, 10), (10, 70))

    assert band._polydata.GetNumberOfLines() == 4
    assert band._polydata.GetNumberOfPolys() == 0
    assert tuple(band._points.GetPoint(0)) == (10.0, 10.0, 0.0)
    assert tuple(band._points.GetPoint(2)) == (90.0, 70.0, 0.0)
    assert renderer.GetActors2D().GetNumberOfItems() == 2
    assert all(not actor.GetPickable() for actor in band._actors)
    assert all(
        actor.GetProperty().GetLineStipplePattern() == 0xF0F0
        for actor in band._actors
    )

    assert band.show()
    assert all(actor.GetVisibility() for actor in band._actors)
    assert band.hide()
    assert all(not actor.GetVisibility() for actor in band._actors)


def test_viewport_rubber_band_uses_vtk_display_coordinates() -> None:
    _application()

    class Plotter:
        def __init__(self) -> None:
            self.renderer = vtk.vtkRenderer()
            self.render_count = 0

        def _getPixelRatio(self):
            return 1.0

        def height(self):
            return 100

        def render(self):
            self.render_count += 1

    viewport = FEMViewport()
    plotter = Plotter()
    viewport._plotter = plotter

    viewport._show_selection_rubber_band((10.0, 20.0), (90.0, 70.0))

    band = viewport._selection_rubber_band
    assert band is not None
    assert tuple(band._points.GetPoint(0)) == (10.0, 29.0, 0.0)
    assert tuple(band._points.GetPoint(2)) == (90.0, 79.0, 0.0)
    assert all(actor.GetVisibility() for actor in band._actors)
    assert plotter.render_count == 1

    viewport._hide_selection_rubber_band()

    assert all(not actor.GetVisibility() for actor in band._actors)
    assert plotter.render_count == 2
    viewport.close()


def test_mesh_scope_pick_signal_emits_typed_mesh_references() -> None:
    _application()
    viewport = FEMViewport()
    picked = []
    viewport.meshEntityPicked.connect(picked.append)
    edge = MeshEntityRef.edge(10, 2, (4, 5))
    face = MeshEntityRef.face(20, 1, (6, 7, 8))
    viewport._mesh_scope_pick_to_ref[("edge", 1)] = edge
    viewport._mesh_scope_pick_to_ref[("face", 1)] = face

    for kind, pick_id in (
        ("mesh_node", 4),
        ("mesh_edge", 1),
        ("mesh_face", 1),
        ("mesh_element", 20),
        ("mesh_body", 30),
    ):
        viewport._submit_pick(
            PickHit(
                kind,
                pick_id,
                "model",
                (0.0, 0.0),
                (0.0, 0.0, 0.0),
            )
        )

    assert picked == [
        MeshEntityRef.node(4),
        edge,
        face,
        MeshEntityRef.element(20),
        MeshEntityRef.element(30),
    ]
    viewport.close()


def test_mesh_scope_boundary_topology_is_built_only_for_edge_or_face_mode(
    monkeypatch,
) -> None:
    _application()
    viewport = FEMViewport()
    builds = []

    def install() -> None:
        builds.append(viewport._selection_mode)
        viewport._mesh_scope_pick_bindings_ready = True

    monkeypatch.setattr(
        viewport,
        "_install_mesh_scope_pick_bindings",
        install,
    )

    viewport.set_selection_mode("mesh_node")
    viewport.set_selection_mode("mesh_element")
    assert builds == []

    viewport.set_selection_mode("mesh_edge")
    viewport.set_selection_mode("mesh_face")
    assert builds == ["mesh_edge"]
    viewport.close()


def test_display_projection_is_cached_until_camera_or_dataset_changes(
    monkeypatch,
) -> None:
    _application()

    class Plotter:
        renderer = vtk.vtkRenderer()

        @staticmethod
        def width():
            return 100

        @staticmethod
        def height():
            return 100

    viewport = FEMViewport()
    viewport._plotter = Plotter()
    dataset = pv.PolyData(np.asarray(((0.0, 0.0, 0.0),)))
    calls = 0

    def project(points):
        nonlocal calls
        calls += 1
        return np.asarray(points, dtype=float)

    monkeypatch.setattr(viewport, "_world_points_to_display", project)

    viewport._dataset_points_to_display(dataset)
    viewport._dataset_points_to_display(dataset)
    assert calls == 1

    dataset.points = np.asarray(((1.0, 0.0, 0.0),))
    viewport._dataset_points_to_display(dataset)
    assert calls == 2
    viewport.close()


@pytest.mark.parametrize(
    ("mode", "expected"),
    (
        ("geometry_point", LogicalEntityRef("point:inside")),
        ("geometry_edge", LogicalEntityRef("edge:crossing")),
        ("geometry_face", LogicalEntityRef("face:crossing")),
        ("geometry_body", LogicalEntityRef("body:crossing")),
    ),
)
@pytest.mark.parametrize(
    ("start", "end"),
    (
        ((0.0, 0.0), (10.0, 10.0)),
        ((10.0, 10.0), (0.0, 0.0)),
    ),
)
def test_geometry_box_selection_intersects_all_entity_kinds_in_both_directions(
    monkeypatch,
    mode,
    expected,
    start,
    end,
) -> None:
    _application()
    viewport = FEMViewport()
    point_data = pv.PolyData(
        np.asarray(((5.0, 5.0, 0.0), (20.0, 20.0, 0.0)))
    )
    point_data.point_data["geometry_pick_id"] = np.asarray((1, 0))
    edge_data = pv.PolyData()
    edge_data.points = np.asarray(((0.0, 5.0, 0.0), (10.0, 5.0, 0.0)))
    edge_data.lines = np.asarray((2, 0, 1))
    edge_data.cell_data["geometry_pick_id"] = np.asarray((2,))
    surface_data = pv.PolyData(
        np.asarray(
            (
                (0.0, 0.0, 0.0),
                (10.0, 0.0, 0.0),
                (10.0, 10.0, 0.0),
                (0.0, 10.0, 0.0),
            )
        ),
        faces=np.asarray((4, 0, 1, 2, 3)),
    )
    surface_data.cell_data["geometry_pick_id"] = np.asarray((3,))
    surface_data.cell_data["geometry_body_pick_id"] = np.asarray((4,))
    viewport._geometry_preview_points = point_data
    viewport._geometry_preview_edges = edge_data
    viewport._geometry_preview_surface = surface_data
    viewport._geometry_pick_to_ref = {
        1: LogicalEntityRef("point:inside"),
        2: LogicalEntityRef("edge:crossing"),
        3: LogicalEntityRef("face:crossing"),
        4: LogicalEntityRef("body:crossing"),
    }
    viewport._selection_mode = mode
    monkeypatch.setattr(
        viewport,
        "_world_points_to_display",
        lambda points: np.column_stack(
            (np.asarray(points)[:, :2], np.full(len(points), 0.5))
        ),
    )
    monkeypatch.setattr(
        viewport,
        "_vtk_rectangle",
        lambda *_args: (4.0, 6.0, 4.0, 6.0),
    )

    selected = (
        viewport._geometry_points_in_qt_rectangle(start, end)
        if mode == "geometry_point"
        else viewport._geometry_entities_in_qt_rectangle(start, end)
    )

    assert selected == (expected,)
    viewport.close()


@pytest.mark.parametrize("mode", ("mesh_edge", "mesh_face", "mesh_body"))
def test_mesh_box_selection_intersects_edge_face_and_body_without_cell_loops(
    monkeypatch,
    mode,
) -> None:
    _application()
    viewport = FEMViewport()
    dataset = pv.PolyData(
        np.asarray(
            (
                (0.0, 0.0, 0.0),
                (10.0, 0.0, 0.0),
                (10.0, 10.0, 0.0),
                (0.0, 10.0, 0.0),
            )
        ),
        faces=np.asarray((4, 0, 1, 2, 3)),
    )
    dataset.cell_data["mesh_scope_pick_id"] = np.asarray((1,))
    dataset.cell_data["element_id"] = np.asarray((10,))
    edge = MeshEntityRef.edge(10, 1, (1, 2))
    face = MeshEntityRef.face(10, 1, (1, 2, 3, 4))
    viewport._mesh_scope_edges = dataset
    viewport._mesh_scope_faces = dataset
    viewport._pick_grid = dataset
    viewport._mesh_scope_pick_to_ref = {
        ("edge", 1): edge,
        ("face", 1): face,
    }
    viewport._selection_mode = mode
    monkeypatch.setattr(
        viewport,
        "_world_points_to_display",
        lambda points: np.column_stack(
            (np.asarray(points)[:, :2], np.full(len(points), 0.5))
        ),
    )
    monkeypatch.setattr(
        viewport,
        "_vtk_rectangle",
        lambda *_args: (4.0, 6.0, 4.0, 6.0),
    )

    selected = viewport._mesh_entities_in_qt_rectangle(
        (0.0, 0.0),
        (10.0, 10.0),
    )

    assert selected == (
        edge
        if mode == "mesh_edge"
        else face
        if mode == "mesh_face"
        else MeshEntityRef.element(10),
    )
    viewport.close()
