from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QDialogButtonBox
import pytest

from fem.application import RegionAssignment, RegionRef, SectionDefinition
from fem.core.model import MaterialDefinition
from fem.materials import MaterialPropertyError, SectionPropertyError
from fem_gui.model_dialogs import (
    RegionAssignmentDialog,
    SectionEditDialog,
    SectionManagerDialog,
)


def _application() -> QApplication:
    return QApplication.instance() or QApplication([])


def _steel() -> MaterialDefinition:
    return MaterialDefinition("Steel", {"E": 210000.0, "nu": 0.3})


@pytest.mark.parametrize(
    ("preset", "values", "expected"),
    (
        ("truss", {"area_spin": 2.5}, {"area": 2.5}),
        (
            "rectangle",
            {"height_spin": 3.0, "width_spin": 2.0},
            {"height": 3.0, "width": 2.0},
        ),
        ("solid_circle", {"radius_spin": 1.25}, {"radius": 1.25}),
        (
            "hollow_circle",
            {"outer_radius_spin": 2.0, "inner_radius_spin": 1.5},
            {"outer_radius": 2.0, "inner_radius": 1.5},
        ),
    ),
)
def test_capability_presets_create_line_sections(
    preset,
    values,
    expected,
):
    _application()
    dialog = SectionEditDialog(
        [_steel()],
        model_dimension=1,
        section_presets=(preset,),
    )
    for name, value in values.items():
        getattr(dialog, name).setValue(value)

    section = dialog.section()

    assert section.section_type == preset
    assert section.properties == expected


def test_truss_requires_only_elastic_modulus_but_beam_requires_poisson_ratio():
    _application()
    axial_material = MaterialDefinition("Axial", {"E": 1000.0})
    truss = SectionEditDialog(
        [axial_material],
        model_dimension=1,
        section_presets=("truss",),
    )
    beam = SectionEditDialog(
        [axial_material],
        model_dimension=1,
        section_presets=("rectangle",),
    )

    assert truss.section().properties == {"area": 1.0}
    with pytest.raises(MaterialPropertyError, match="nu"):
        beam.section()


def test_hollow_circle_relation_is_shown_and_validated():
    _application()
    dialog = SectionEditDialog(
        [_steel()],
        model_dimension=1,
        section_presets=("hollow_circle",),
    )
    dialog.outer_radius_spin.setValue(1.0)
    dialog.inner_radius_spin.setValue(1.5)

    assert dialog.form.isRowVisible(dialog.validation_label)
    assert not dialog.buttons.button(
        QDialogButtonBox.StandardButton.Ok
    ).isEnabled()
    with pytest.raises(SectionPropertyError, match="outer_radius"):
        dialog.section()


def test_beam_shape_switch_clears_irrelevant_legacy_fields():
    _application()
    original = SectionDefinition(
        "Beam",
        "Steel",
        "rectangle",
        {
            "height": 3.0,
            "width": 2.0,
            "radius": 99.0,
            "area": 6.0,
            "Iyy": 4.5,
            "custom_metadata": "preserved",
        },
    )
    dialog = SectionEditDialog(
        [_steel()],
        original,
        model_dimension=1,
        section_presets=(
            "rectangle",
            "solid_circle",
            "hollow_circle",
        ),
    )

    assert dialog.form.isRowVisible(dialog.limitation_label)
    assert "beam.orientation.assumed" in dialog.limitation_label.text()
    dialog.type_combo.setCurrentIndex(
        dialog.type_combo.findData("solid_circle")
    )
    dialog.radius_spin.setValue(1.25)
    section = dialog.section()

    assert not dialog.form.isRowVisible(dialog.limitation_label)
    assert section.section_type == "solid_circle"
    assert section.properties == {
        "custom_metadata": "preserved",
        "radius": 1.25,
    }


def test_unknown_imported_section_is_exactly_preserved_read_only():
    _application()
    imported = SectionDefinition(
        "Future",
        "Steel",
        "future_profile",
        {"opaque": {"values": [1, 2, 3]}},
    )
    dialog = SectionEditDialog(
        [_steel()],
        imported,
        model_dimension=1,
        section_presets=("rectangle",),
    )
    dialog.name_edit.setText("Ignored")

    assert not dialog.name_edit.isEnabled()
    assert not dialog.material_combo.isEnabled()
    assert not dialog.type_combo.isEnabled()
    assert dialog.section() == imported
    assert dialog.section() is not imported


def test_line_section_manager_uses_presets_and_authoring_policy():
    _application()
    enabled = SectionManagerDialog(
        [_steel()],
        [],
        model_dimension=1,
        section_presets=("truss",),
    )
    no_presets = SectionManagerDialog(
        [_steel()],
        [],
        model_dimension=1,
    )
    policy_disabled = SectionManagerDialog(
        [_steel()],
        [],
        model_dimension=1,
        section_presets=("truss",),
        authoring_enabled=False,
    )

    assert enabled.add_button.isEnabled()
    assert not no_presets.add_button.isEnabled()
    assert not policy_disabled.add_button.isEnabled()


def test_region_assignment_uses_per_section_typed_compatible_targets():
    _application()
    sections = (
        SectionDefinition("Truss", "Steel", "truss", {"area": 1.0}),
        SectionDefinition(
            "Beam",
            "Steel",
            "rectangle",
            {"height": 1.0, "width": 1.0},
        ),
    )
    truss_target = RegionRef("element_set", "TRUSS_SET")
    beam_target = RegionRef("element_set", "BEAM_SET")
    dialog = RegionAssignmentDialog(
        sections,
        compatible_targets={
            "Truss": (truss_target,),
            "Beam": (beam_target,),
        },
    )

    assert dialog.region_combo.currentData() == truss_target
    assert dialog.assignment() == RegionAssignment("Truss", "TRUSS_SET")

    dialog.section_combo.setCurrentText("Beam")
    assert dialog.region_combo.currentData() == beam_target
    assert dialog.assignment() == RegionAssignment("Beam", "BEAM_SET")


def test_region_assignment_preserves_namespaces_and_legacy_strings():
    _application()
    section = SectionDefinition("Section", "Steel")
    node = RegionRef("node_set", "SAME")
    elements = RegionRef("element_set", "SAME")
    typed = RegionAssignmentDialog([section], [node, elements])

    assert typed.region_combo.itemText(0) == "SAME（节点集）"
    assert typed.region_combo.itemText(1) == "SAME（单元集）"
    with pytest.raises(ValueError, match="element_set"):
        typed.assignment()

    typed.region_combo.setCurrentIndex(1)
    assert typed.assignment() == RegionAssignment("Section", "SAME")

    legacy = RegionAssignmentDialog([section], ["DOMAIN"])
    assert legacy.assignment() == RegionAssignment("Section", "DOMAIN")
