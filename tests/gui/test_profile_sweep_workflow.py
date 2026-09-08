from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import QSize
from PySide6.QtWidgets import QApplication

from fem.geometry import (
    LogicalEntityRef,
    RectangleGeometry,
    RevolvedGeometry,
    RotatedGeometry,
)
from fem_gui.geometry_preview import build_geometry_preview
from fem_gui.icons import icon
from fem_gui.main_window import FEMMainWindow
from fem_gui.preprocessing_dialogs import SweepGeometryDialog
from tests.helpers.profile_sketches import (
    profile_face_id,
    two_profile_sketch,
)


def _application() -> QApplication:
    return QApplication.instance() or QApplication([])


def _rectangle() -> RectangleGeometry:
    return RectangleGeometry("扫掠草图", 2.0, 1.0)


def test_sweep_preview_is_three_dimensional_and_body_pickable() -> None:
    preview = build_geometry_preview(
        RevolvedGeometry(
            _rectangle(),
            "x",
            180.0,
            ("face:domain",),
        )
    )

    assert preview.dimension == 3
    assert preview.body_logical_id == "body:domain"
    assert preview.points
    assert preview.faces
    assert set(preview.face_logical_ids) == {
        "face:start",
        "face:end",
        "face:sides",
    }


def test_sweep_dialog_builds_axis_angle_recipe() -> None:
    _application()
    dialog = SweepGeometryDialog(
        _rectangle(),
        source_face_ids=("face:domain",),
    )
    dialog.axis_combo.setCurrentIndex(1)
    dialog.angle_spin.setValue(225.0)

    recipe = dialog.recipe()

    assert dialog.windowTitle() == "扫掠几何"
    assert tuple(
        dialog.axis_combo.itemData(index)
        for index in range(dialog.axis_combo.count())
    ) == ("x", "y", "z")
    assert recipe.axis == "y"
    assert recipe.angle_degrees == 225.0
    assert recipe.source_face_ids == ("face:domain",)
    dialog.close()


def test_sweep_action_commits_beside_extrusion(monkeypatch) -> None:
    _application()
    window = FEMMainWindow()
    base = _rectangle()
    window._set_native_geometry(base, "测试草图")
    window._update_action_states()

    assert window.actions["geometry_sweep"].isEnabled()
    assert window.actions["geometry_sweep"].text() == "扫掠"
    assert not window.actions["geometry_sweep"].icon().isNull()
    assert not icon("sweep").pixmap(QSize(32, 32)).isNull()

    def accept(dialog: SweepGeometryDialog) -> bool:
        dialog.axis_combo.setCurrentIndex(0)
        dialog.angle_spin.setValue(180.0)
        return True

    monkeypatch.setattr(window, "_exec_dialog", accept)
    window.sweep_geometry()

    recipe = window.document.geometry_recipe
    assert isinstance(recipe, RevolvedGeometry)
    assert recipe.axis == "x"
    assert recipe.angle_degrees == 180.0
    assert window._selected_geometry_refs == set()
    assert window._geometry_selection_mode == "body"
    window.close()


def test_rotating_swept_part_passes_occ_single_solid_authentication() -> None:
    _application()
    window = FEMMainWindow()
    sweep = RevolvedGeometry(
        _rectangle(),
        "x",
        180.0,
        ("face:domain",),
    )
    window._set_native_geometry(sweep, "测试扫掠体")

    accepted = window._set_native_geometry(
        RotatedGeometry(sweep, "z", 37.0),
        "旋转后的",
    )

    assert accepted
    assert window.document.geometry_recipe == RotatedGeometry(
        sweep,
        "z",
        37.0,
    )
    window.close()


def test_sweep_cancel_preserves_recipe(monkeypatch) -> None:
    _application()
    window = FEMMainWindow()
    base = _rectangle()
    window._set_native_geometry(base, "测试草图")
    monkeypatch.setattr(window, "_exec_dialog", lambda _dialog: False)

    window.sweep_geometry()

    assert window.document.geometry_recipe == base
    window.close()


def test_multi_profile_sweep_creates_independent_parts(monkeypatch) -> None:
    _application()
    window = FEMMainWindow()
    sketch = two_profile_sketch()
    face_ids = (
        profile_face_id(sketch, "L1"),
        profile_face_id(sketch, "L5"),
    )
    window._set_native_geometry(sketch, "测试草图")
    window._set_geometry_selection_mode("face")
    window._selected_geometry_refs = {
        LogicalEntityRef(face_id) for face_id in face_ids
    }

    def accept(dialog: SweepGeometryDialog) -> bool:
        dialog.axis_combo.setCurrentIndex(0)
        dialog.angle_spin.setValue(180.0)
        return True

    monkeypatch.setattr(window, "_exec_dialog", accept)
    window.sweep_geometry()

    assert tuple(part.id for part in window.document.parts) == ("P1", "P2")
    assert all(
        isinstance(part.geometry_recipe, RevolvedGeometry)
        for part in window.document.parts
    )
    assert {
        part.geometry_recipe.source_face_ids
        for part in window.document.parts
    } == {(face_ids[0],), (face_ids[1],)}
    window.close()
