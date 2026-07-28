from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtGui import QColor, QImage
from PySide6.QtWidgets import QApplication, QWidget

from fem.application import MeshEntityRef
from fem_gui.widgets.viewport import (
    FEMViewport,
    PickHit,
    _SelectionRubberBand,
)


def _application() -> QApplication:
    return QApplication.instance() or QApplication([])


def test_selection_rubber_band_has_no_interior_fill() -> None:
    application = _application()
    parent = QWidget()
    parent.resize(100, 80)
    band = _SelectionRubberBand(parent)
    band.setGeometry(10, 10, 80, 60)
    band.show()
    application.processEvents()
    image = QImage(
        band.size(),
        QImage.Format.Format_ARGB32_Premultiplied,
    )
    image.fill(QColor(0, 0, 0, 0))

    band.render(image)

    assert image.pixelColor(40, 30).alpha() == 0
    assert image.pixelColor(1, 30).alpha() > 0
    parent.close()


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
