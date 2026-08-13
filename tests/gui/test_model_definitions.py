from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QDoubleSpinBox,
    QLabel,
)

from fem.application import SectionDefinition
from fem.core.model import MaterialDefinition
from fem_gui.dialogs import AdaptivePrecisionDoubleSpinBox
from fem_gui.model_dialogs import (
    DensityBehaviorDialog,
    ElasticBehaviorDialog,
    MaterialEditDialog,
    MaterialManagerDialog,
    SectionEditDialog,
    SectionManagerDialog,
)


def _application() -> QApplication:
    return QApplication.instance() or QApplication([])


def test_new_material_starts_empty_and_uses_modal_parameter_fields_only():
    _application()
    material_dialog = MaterialEditDialog()
    assert material_dialog.findChildren(QDoubleSpinBox) == []
    assert material_dialog.behavior_table.rowCount() == 0
    assert "双击材料行为，或选中后点击“编辑参数”。" not in {
        label.text() for label in material_dialog.findChildren(QLabel)
    }
    material_dialog.name_edit.setText("Aluminium")
    material = material_dialog.material()
    elastic_dialog = ElasticBehaviorDialog(material.properties)
    density_dialog = DensityBehaviorDialog(material.properties)
    assert elastic_dialog.elastic_spin.text() == ""
    assert elastic_dialog.poisson_spin.text() == ""
    assert not elastic_dialog.ok_button.isEnabled()
    elastic_dialog.elastic_spin.setValue(70000.0)
    elastic_dialog.poisson_spin.setValue(0.33)
    assert elastic_dialog.ok_button.isEnabled()
    configured_material = MaterialDefinition(
        material.name,
        elastic_dialog.values(),
    )
    section_dialog = SectionEditDialog(
        [configured_material],
        section_presets=(
            "solid_plane_stress",
            "solid_plane_strain",
        ),
    )
    section = section_dialog.section()

    assert material.name == "Aluminium"
    assert material.properties == {}
    assert elastic_dialog.values() == {"E": 70000.0, "nu": 0.33}
    assert density_dialog.value() > 0.0
    assert section.material == "Aluminium"
    assert section.properties["plane_type"] == "stress"
    assert section.properties["thickness"] > 0.0

    existing_elastic = ElasticBehaviorDialog({"E": 210000.0, "nu": 0.3})
    assert existing_elastic.ok_button.isEnabled()
    assert existing_elastic.values() == {"E": 210000.0, "nu": 0.3}


def test_material_inputs_use_adaptive_precision_consistently():
    _application()
    elastic = ElasticBehaviorDialog({"E": 210000.123456, "nu": 0.333333})
    density = DensityBehaviorDialog({"rho": 7850.123456})

    editors = (
        elastic.elastic_spin,
        elastic.poisson_spin,
        density.density_spin,
    )
    assert all(
        isinstance(editor, AdaptivePrecisionDoubleSpinBox)
        and editor.decimals() == 12
        for editor in editors
    )
    assert elastic.elastic_spin.text() == "210000.12"
    assert elastic.poisson_spin.text() == "0.33"
    assert density.density_spin.text() == "7850.12"
    assert elastic.values()["nu"] == 0.333333

    elastic.poisson_spin.selectAll()
    QTest.keyClicks(elastic.poisson_spin, "0.333333")
    QTest.keyClick(elastic.poisson_spin, Qt.Key.Key_Return)
    assert elastic.poisson_spin.text() == "0.333333"
    assert elastic.values()["nu"] == 0.333333


def test_section_dialog_uses_dimension_specific_supported_parameters():
    _application()
    material = MaterialDefinition(
        "Steel",
        {"E": 210000.0, "nu": 0.3},
    )
    plane = SectionEditDialog(
        [material],
        model_dimension=2,
        section_presets=(
            "solid_plane_stress",
            "solid_plane_strain",
        ),
    )
    plane.type_combo.setCurrentIndex(
        plane.type_combo.findData("solid_plane_strain")
    )
    plane_section = plane.section()

    assert plane.type_combo.itemText(0) == "平面应力"
    assert plane.type_combo.itemText(1) == "平面应变"
    assert plane_section.section_type == "solid"
    assert plane_section.properties["plane_type"] == "strain"
    assert plane.form.isRowVisible(plane.thickness_spin)

    solid = SectionEditDialog(
        [material],
        model_dimension=3,
        section_presets=("solid",),
    )
    solid_section = solid.section()

    assert solid.type_combo.itemText(0) == "三维实体"
    assert not solid.form.isRowVisible(solid.thickness_spin)
    assert solid_section.section_type == "solid"
    assert "plane_type" not in solid_section.properties
    assert "thickness" not in solid_section.properties

    imported = SectionDefinition(
        "Beam section",
        "Steel",
        "beam",
        {"area": 12.0, "I11": 3.0},
    )
    line = SectionEditDialog(
        [material],
        imported,
        model_dimension=1,
    )

    assert not line.type_combo.isEnabled()
    assert line.section() == imported

    imported_shell = SectionDefinition(
        "Imported shell",
        "Steel",
        "shell",
        {"thickness": 0.8},
    )
    shell = SectionEditDialog(
        [material],
        imported_shell,
        model_dimension=2,
    )
    assert not shell.type_combo.isEnabled()
    assert not shell.form.isRowVisible(shell.thickness_spin)
    assert shell.section() == imported_shell


def test_material_editor_preserves_unknown_inp_behaviors_read_only():
    _application()
    original = MaterialDefinition(
        "Imported",
        {
            "E": 1000.0,
            "nu": 0.25,
            "future_behavior": ((0.0, 1.0),),
        },
    )
    dialog = MaterialEditDialog(original)

    assert dialog.behavior_table.rowCount() == 2
    assert "来自 INP" in dialog.behavior_table.item(1, 0).text()
    dialog.behavior_table.selectRow(1)
    assert not dialog.edit_behavior_button.isEnabled()
    assert not dialog.delete_behavior_button.isEnabled()

    assert dialog.material().properties == original.properties


def test_definition_managers_edit_copies_and_use_read_only_tables():
    _application()
    materials = [
        MaterialDefinition(
            "Steel",
            {"E": 210000.0, "nu": 0.3, "rho": 7850.0},
        )
    ]
    sections = [SectionDefinition("Solid", "Steel")]
    material_dialog = MaterialManagerDialog(materials)
    section_dialog = SectionManagerDialog(materials, sections)

    assert (
        material_dialog.table.editTriggers()
        == QAbstractItemView.EditTrigger.NoEditTriggers
    )
    assert material_dialog.table.columnCount() == 2
    assert material_dialog.table.item(0, 1).text() == "线弹性、密度"
    material_dialog._delete()
    section_dialog._delete()

    assert [item.name for item in materials] == ["Steel"]
    assert [item.name for item in sections] == ["Solid"]
    assert material_dialog.values() == []
    assert section_dialog.values() == []


def test_material_manager_creates_mutable_copy_from_snapshot_tuple():
    _application()
    original = (
        MaterialDefinition("Steel", {"E": 210000.0, "nu": 0.3}),
    )
    dialog = MaterialManagerDialog(original)

    dialog._store(
        MaterialDefinition("Aluminum", {"E": 70000.0, "nu": 0.33})
    )

    assert [material.name for material in dialog.values()] == [
        "Steel",
        "Aluminum",
    ]
    assert [material.name for material in original] == ["Steel"]


def test_section_manager_creates_mutable_copy_from_snapshot_tuple():
    _application()
    materials = (MaterialDefinition("Steel", {"E": 210000.0, "nu": 0.3}),)
    original = (SectionDefinition("Solid", "Steel"),)
    dialog = SectionManagerDialog(materials, original)

    dialog._store(SectionDefinition("Shell", "Steel"))

    assert [section.name for section in dialog.values()] == ["Solid", "Shell"]
    assert [section.name for section in original] == ["Solid"]
