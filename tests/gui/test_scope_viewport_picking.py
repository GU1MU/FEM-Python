from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import vtk
from PySide6.QtWidgets import QApplication

from fem.application import MeshEntityRef
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
    band.set_containment(False)

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
