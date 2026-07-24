from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QDialogButtonBox, QLabel

from fem_gui.document import NamedRegion
from fem_gui.preprocessing import (
    LocalMeshControl,
    MeshSettings,
    ExtrudedGeometry,
    SketchGeometry,
    SketchCircle,
    SketchRectangle,
)
from fem_gui.preprocessing_dialogs import (
    GeometryManagerDialog,
    LocalMeshControlDialog,
    MeshControlsDialog,
    MeshSettingsDialog,
    NamedRegionDialog,
    NamedRegionManagerDialog,
    SketchContourDialog,
)


def _application() -> QApplication:
    return QApplication.instance() or QApplication([])


def test_sketch_contour_dialog_only_shows_shape_specific_dimensions() -> None:
    _application()
    rectangle = SketchContourDialog(
        SketchRectangle("material", 0.0, 0.0, 10.0, 5.0)
    )
    assert rectangle.radius_spin.isHidden()
    assert not rectangle.width_spin.isHidden()
    assert not rectangle.height_spin.isHidden()
    assert rectangle.width_spin.text() == "10.00"
    rectangle.width_spin.setValue(0.0001)
    assert rectangle.width_spin.text() == "0.0001"
    assert rectangle.width_spin.value() == 0.0001
    assert rectangle.findChildren(QDialogButtonBox)[0].button(
        QDialogButtonBox.StandardButton.Ok
    ).text() == "确定"

    circle = SketchContourDialog(SketchCircle("cut", 1.0, 2.0, 3.0))
    assert circle.radius_spin.isHidden() is False
    assert circle.width_spin.isHidden()
    assert circle.height_spin.isHidden()


def test_mesh_settings_has_no_special_hole_size_field() -> None:
    _application()
    dialog = MeshSettingsDialog(MeshSettings(5.0))

    labels = {label.text() for label in dialog.findChildren(QLabel)}

    assert "孔边局部尺寸" not in labels
    assert dialog.settings().local_size is None


def test_mesh_settings_method_exposes_only_supported_element_shape() -> None:
    _application()
    dialog = MeshSettingsDialog(
        MeshSettings(5.0, cell_shape="quadrilateral"),
        mesh_dimension=2,
    )

    assert dialog.method_combo.currentData() == "recombine"
    assert dialog.shape_combo.currentData() == "quadrilateral"
    dialog.method_combo.setCurrentIndex(
        dialog.method_combo.findData("free")
    )
    assert dialog.shape_combo.count() == 1
    assert dialog.settings().cell_shape == "triangle"

    volume = MeshSettingsDialog(
        MeshSettings(5.0, cell_shape="hexahedron"),
        mesh_dimension=3,
        allow_hexahedron=False,
    )
    assert volume.method_combo.findData("structured") == -1
    assert volume.settings().cell_shape == "tetrahedron"


def test_local_mesh_dialog_records_the_viewport_selected_edge() -> None:
    _application()
    dialog = LocalMeshControlDialog(
        "edge",
        2,
        5.0,
    )

    assert dialog.control().entity_id == 2
    labels = {label.text() for label in dialog.findChildren(QLabel)}
    assert "已选择 1 个边" in labels
    assert "边 2" not in labels


def test_named_region_dialog_and_manager_support_multiple_entities() -> None:
    _application()
    create_dialog = NamedRegionDialog("edge", (1, 3, 4))
    assert create_dialog.name_edit.text() == "EdgeSet-1"
    assert "已选择 3 个边" in {
        label.text() for label in create_dialog.findChildren(QLabel)
    }

    manager = NamedRegionManagerDialog({
        "Fixed": NamedRegion("Fixed", "edge", (1, 3, 4)),
    })
    assert manager.table.item(0, 2).text() == "3 个"
    manager.name_edit.setText("Support")
    manager._rename()

    assert tuple(manager.values()) == ("Support",)
    assert manager.values()["Support"].entity_ids == (1, 3, 4)


def test_mesh_control_manager_deletes_only_the_selected_local_control() -> None:
    _application()
    dialog = MeshControlsDialog(
        MeshSettings(
            1.0,
            local_controls=(
                LocalMeshControl("edge", 1, 0.25),
                LocalMeshControl("edge", 3, 0.5),
            ),
        )
    )
    dialog.control_list.setCurrentRow(3)
    dialog._delete()

    controls = dialog.settings().local_controls
    assert len(controls) == 1
    assert controls[0].entity_id == 3
    assert controls[0].size == 0.5
    assert "边 3" not in dialog.control_list.item(3).text()


def test_feature_manager_only_edits_the_base_and_deletes_the_last_feature():
    _application()
    recipe = ExtrudedGeometry(
        SketchGeometry(
            "Sketch-1",
            (SketchRectangle("material", 0.0, 0.0, 2.0, 1.0),),
        ),
        1.0,
    )
    dialog = GeometryManagerDialog(
        recipe,
        can_edit_base=True,
    )

    dialog.feature_list.setCurrentRow(0)
    assert dialog.edit_button.isEnabled()
    assert not dialog.delete_button.isEnabled()
    dialog.feature_list.setCurrentRow(dialog.feature_list.count() - 1)
    assert not dialog.edit_button.isEnabled()
    assert dialog.delete_button.isEnabled()
