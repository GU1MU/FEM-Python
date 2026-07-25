from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QAbstractItemView, QApplication, QDoubleSpinBox

from fem.core.model import ElementSet, MaterialDefinition
from fem_gui.document import FEMDocument, RegionAssignment, SectionDefinition
from fem_gui.model_definitions import (
    apply_document_definitions,
    hydrate_document_definitions,
    section_assignment_issues,
)
from fem_gui.model_dialogs import (
    DensityBehaviorDialog,
    ElasticBehaviorDialog,
    MaterialEditDialog,
    MaterialManagerDialog,
    SectionEditDialog,
    SectionManagerDialog,
)
from fem_gui.preprocessing import MeshSettings, RectangleGeometry, generate_fem_model


def _application() -> QApplication:
    return QApplication.instance() or QApplication([])


def test_native_material_section_and_region_assignment_compile_to_fem_model():
    document = FEMDocument()
    model = generate_fem_model(RectangleGeometry("plate", 2.0, 1.0), MeshSettings(0.5))
    document.set_generated_model(model, geometry_recipe=RectangleGeometry("plate", 2.0, 1.0), mesh_settings=MeshSettings(0.5))
    document.material_definitions = [MaterialDefinition("Steel", {"E": 210000.0, "nu": 0.3})]
    document.section_definitions = [SectionDefinition("Section-1", "Steel", "solid", {"thickness": 2.0})]
    document.region_assignments = [RegionAssignment("Section-1", "DOMAIN")]

    apply_document_definitions(document)

    assert model.materials["Steel"].properties["E"] == 210000.0
    assert model.sections[0].element_set == "DOMAIN"
    assert model.sections[0].material == "Steel"
    assert model.sections[0].properties["thickness"] == 2.0


def test_inp_definitions_are_hydrated_for_the_same_management_dialogs(gui_inp_path):
    from fem.abaqus import read

    document = FEMDocument()
    document.set_model(gui_inp_path, read(gui_inp_path))
    hydrate_document_definitions(document)

    assert [material.name for material in document.material_definitions] == ["STEEL"]
    assert document.section_definitions[0].material == "STEEL"
    assert document.region_assignments[0].region_name == "SOLID"


def test_material_and_section_dialogs_use_modal_parameter_fields_only():
    _application()
    material_dialog = MaterialEditDialog()
    assert material_dialog.findChildren(QDoubleSpinBox) == []
    assert material_dialog.behavior_table.item(0, 0).text() == "线弹性"
    material_dialog.name_edit.setText("Aluminium")
    material = material_dialog.material()
    elastic_dialog = ElasticBehaviorDialog(material.properties)
    density_dialog = DensityBehaviorDialog(material.properties)
    section_dialog = SectionEditDialog([material])
    section = section_dialog.section()

    assert material.name == "Aluminium"
    assert material.properties["E"] > 0.0
    assert elastic_dialog.values()["nu"] == 0.3
    assert density_dialog.value() > 0.0
    assert section.material == "Aluminium"
    assert section.properties["plane_type"] == "stress"
    assert section.properties["thickness"] > 0.0


def test_section_dialog_uses_dimension_specific_supported_parameters():
    _application()
    material = MaterialDefinition(
        "Steel",
        {"E": 210000.0, "nu": 0.3},
    )
    plane = SectionEditDialog([material], model_dimension=2)
    plane.type_combo.setCurrentIndex(plane.type_combo.findData("strain"))
    plane_section = plane.section()

    assert plane.type_combo.itemText(0) == "平面应力"
    assert plane.type_combo.itemText(1) == "平面应变"
    assert plane_section.section_type == "solid"
    assert plane_section.properties["plane_type"] == "strain"
    assert plane.form.isRowVisible(plane.thickness_spin)

    solid = SectionEditDialog([material], model_dimension=3)
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


def test_readiness_reports_missing_material_and_section_without_faking_solver_support():
    document = FEMDocument()
    model = generate_fem_model(RectangleGeometry("plate", 2.0, 1.0), MeshSettings(0.5))
    document.set_generated_model(model, geometry_recipe=RectangleGeometry("plate", 2.0, 1.0), mesh_settings=MeshSettings(0.5))

    assert section_assignment_issues(document) == ("尚未定义材料",)


def test_readiness_reports_incomplete_elasticity_and_unassigned_elements():
    document = FEMDocument()
    model = generate_fem_model(
        RectangleGeometry("plate", 2.0, 1.0),
        MeshSettings(0.5),
    )
    document.set_generated_model(
        model,
        geometry_recipe=RectangleGeometry("plate", 2.0, 1.0),
        mesh_settings=MeshSettings(0.5),
    )
    first_element_id = model.mesh.elements[0].id
    model.element_sets["PARTIAL"] = ElementSet("PARTIAL", [first_element_id])
    document.material_definitions = [
        MaterialDefinition("Incomplete", {"E": 210000.0})
    ]
    document.section_definitions = [
        SectionDefinition("Section-1", "Incomplete")
    ]
    document.region_assignments = [
        RegionAssignment("Section-1", "PARTIAL")
    ]
    apply_document_definitions(document)

    issues = section_assignment_issues(document)

    assert any("缺少线弹性参数" in issue for issue in issues)
    assert any("尚未分配截面" in issue for issue in issues)


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
