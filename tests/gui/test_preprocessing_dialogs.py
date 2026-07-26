from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QDialogButtonBox, QLabel

from fem.application import NamedRegion
from fem.geometry import (
    ExtrudedGeometry,
    LogicalEntityRef,
    SketchCircle,
    SketchGeometry,
    SketchRectangle,
)
from fem.mesh import settings as mesh_settings_api
from fem.mesh.settings import LocalMeshControl, MeshSettings
from fem_gui.preprocessing_dialogs import (
    GeometryManagerDialog,
    LocalMeshControlDialog,
    MeshControlsDialog,
    MeshSettingsDialog,
    NamedRegionDialog,
    NamedRegionManagerDialog,
    SketchContourDialog,
)
import fem_gui.preprocessing_dialogs as preprocessing_dialogs_module


def _application() -> QApplication:
    return QApplication.instance() or QApplication([])


def _falloff(
    reference: str = "global_size",
    start_factor: float = 0.0,
    end_factor: float = 2.0,
):
    return mesh_settings_api.MeshSizeFalloff(
        reference,
        start_factor,
        end_factor,
    )


def _control(
    logical_id: str,
    size: float,
    *,
    falloff=None,
) -> LocalMeshControl:
    return LocalMeshControl(
        LogicalEntityRef(logical_id),
        size,
        _falloff() if falloff is None else falloff,
    )


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
    assert not hasattr(dialog.settings(), "local_size")


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
        LogicalEntityRef("edge:right"),
        5.0,
    )

    assert dialog.control().target == LogicalEntityRef("edge:right")
    assert dialog.control().falloff == _falloff()
    labels = {label.text() for label in dialog.findChildren(QLabel)}
    assert "已选择 1 个边" in labels
    assert "边 2" not in labels


def test_named_region_dialog_and_manager_support_multiple_entities() -> None:
    _application()
    references = tuple(
        LogicalEntityRef(logical_id)
        for logical_id in (
            "edge:bottom",
            "edge:top",
            "edge:left",
        )
    )
    create_dialog = NamedRegionDialog(references)
    assert create_dialog.name_edit.text() == "EdgeSet-1"
    assert "已选择 3 个边" in {
        label.text() for label in create_dialog.findChildren(QLabel)
    }

    manager = NamedRegionManagerDialog({
        "Fixed": NamedRegion("Fixed", references),
    })
    assert manager.table.item(0, 2).text() == "3 个"
    manager.name_edit.setText("Support")
    manager._rename()

    assert tuple(manager.values()) == ("Support",)
    assert set(manager.values()["Support"].references) == set(references)


def test_mesh_control_manager_deletes_only_the_selected_local_control() -> None:
    _application()
    dialog = MeshControlsDialog(
        MeshSettings(
            1.0,
            local_controls=(
                _control("edge:bottom", 0.25),
                _control("edge:top", 0.5),
            ),
        )
    )
    dialog.control_list.setCurrentRow(3)
    dialog._delete()

    controls = dialog.settings().local_controls
    assert len(controls) == 1
    assert controls[0].target == LogicalEntityRef("edge:top")
    assert controls[0].size == 0.5
    assert "边 3" not in dialog.control_list.item(3).text()


def test_mesh_control_manager_edit_preserves_target_radius_falloff(
    monkeypatch,
) -> None:
    _application()
    falloff = _falloff("target_radius", 0.25, 2.0)
    current = _control(
        "edge:hole-loop",
        0.25,
        falloff=falloff,
    )
    dialog = MeshControlsDialog(
        MeshSettings(1.0, local_controls=(current,))
    )

    class AcceptedDialog:
        def __init__(
            self,
            target,
            _global_size,
            _parent,
            *,
            current_size,
            falloff,
        ) -> None:
            assert target == current.target
            assert current_size == current.size
            assert falloff == current.falloff
            self._control = LocalMeshControl(
                target,
                0.2,
                falloff,
            )

        def exec(self):
            return True

        def control(self):
            return self._control

    monkeypatch.setattr(
        preprocessing_dialogs_module,
        "LocalMeshControlDialog",
        AcceptedDialog,
    )
    dialog.control_list.setCurrentRow(3)

    dialog._edit()

    edited = dialog.settings().local_controls[0]
    assert edited.size == 0.2
    assert edited.falloff == falloff


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
