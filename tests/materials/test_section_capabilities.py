from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from fem.abaqus import read
from fem import materials
from fem.core.model import (
    ElementSet,
    MaterialDefinition,
    SectionAssignment,
)


def _element(
    element_id: int,
    element_type: str,
    **properties,
) -> SimpleNamespace:
    return SimpleNamespace(
        id=element_id,
        type=element_type,
        props=dict(properties),
    )


def _model(*elements: SimpleNamespace) -> SimpleNamespace:
    return SimpleNamespace(
        mesh=SimpleNamespace(elements=list(elements)),
        materials={},
        sections=[],
        element_sets={},
        metadata={},
    )


def test_known_section_schemas_resolve_owned_effective_properties() -> None:
    continuum_material = {"E": 210.0, "nu": 0.3}

    plane = materials.resolve_section_properties(
        "CPE3",
        continuum_material,
        "solid",
        {},
    )
    solid = materials.resolve_section_properties(
        "C3D4",
        continuum_material,
        "solid",
        {},
    )
    truss = materials.resolve_section_properties(
        "Truss2",
        {"E": 100.0},
        "truss",
        {"area": 2.0},
    )

    assert plane.element_family == "plane_continuum"
    assert plane.section_family == "solid"
    assert plane.effective_properties["plane_type"] == "strain"
    assert plane.effective_properties["thickness"] == 1.0
    assert solid.element_family == "solid_continuum"
    assert "thickness" not in solid.effective_properties
    assert "plane_type" not in solid.effective_properties
    assert truss.element_family == "truss"
    assert truss.section_family == "truss"
    assert truss.effective_properties["E"] == 100.0
    assert truss.effective_properties["area"] == 2.0
    assert "nu" not in truss.effective_properties

    continuum_material["E"] = 1.0
    assert plane.effective_properties["E"] == 210.0


@pytest.mark.parametrize(
    ("section_type", "properties", "expected"),
    (
        (
            "rectangle",
            {"height": 4.0, "width": 2.0},
            {"height": 4.0, "width": 2.0},
        ),
        (
            "solid_circle",
            {"radius": 2.0},
            {"radius": 2.0},
        ),
        (
            "hollow_circle",
            {"outer_radius": 2.0, "inner_radius": 1.0},
            {"outer_radius": 2.0, "inner_radius": 1.0},
        ),
    ),
)
def test_beam_presets_use_one_validated_schema(
    section_type,
    properties,
    expected,
) -> None:
    resolved = materials.resolve_section_properties(
        "Beam2",
        {"E": 210.0, "nu": 0.3},
        section_type,
        properties,
    )

    assert resolved.element_family == "beam"
    assert resolved.section_family == "beam"
    assert resolved.section_type == section_type
    assert {
        name: resolved.effective_properties[name]
        for name in expected
    } == expected


def test_plane_defaults_inherit_explicit_and_imported_formulation_data() -> None:
    material = {"E": 210.0, "nu": 0.3}

    inherited = materials.resolve_section_properties(
        "Tri3",
        material,
        "solid",
        {},
        baseline_properties={
            "abaqus_type": "CPE3",
            "thickness": 2.5,
        },
    )
    overridden = materials.resolve_section_properties(
        "CPE3",
        material,
        "solid",
        {"plane_type": "stress", "thickness": 3.0},
    )

    assert inherited.effective_properties["plane_type"] == "strain"
    assert inherited.effective_properties["thickness"] == 2.5
    assert overridden.effective_properties["plane_type"] == "stress"
    assert overridden.effective_properties["thickness"] == 3.0


@pytest.mark.parametrize("value", (0.0, -1.0, float("nan"), float("inf")))
def test_all_families_require_a_positive_finite_elastic_modulus(
    value,
) -> None:
    with pytest.raises(materials.MaterialPropertyError, match="E"):
        materials.resolve_section_properties(
            "Truss2",
            {"E": value},
            "truss",
            {"area": 1.0},
        )


@pytest.mark.parametrize("element_type", ("Tri3", "Tet4", "Beam2"))
@pytest.mark.parametrize(
    "nu",
    (None, -1.0, 0.5, float("nan"), float("inf")),
)
def test_continuum_and_beam_require_core_range_poisson_ratio(
    element_type,
    nu,
) -> None:
    properties = {"E": 210.0}
    if nu is not None:
        properties["nu"] = nu
    section_type = "solid"
    section_properties = {}
    if element_type == "Beam2":
        section_type = "rectangle"
        section_properties = {"height": 1.0, "width": 1.0}

    with pytest.raises(materials.MaterialPropertyError, match="nu"):
        materials.resolve_section_properties(
            element_type,
            properties,
            section_type,
            section_properties,
        )


def test_truss_requires_only_E_and_ignores_irrelevant_nu() -> None:
    resolved = materials.resolve_section_properties(
        "Truss2",
        {"E": 210.0, "nu": "not-used"},
        "truss",
        {"area": 1.0},
    )

    assert resolved.effective_properties["nu"] == "not-used"


@pytest.mark.parametrize("rho", (-1.0, float("nan"), float("inf"), None))
def test_optional_density_is_finite_and_nonnegative_when_present(rho) -> None:
    with pytest.raises(materials.MaterialPropertyError, match="rho"):
        materials.resolve_section_properties(
            "Truss2",
            {"E": 210.0, "rho": rho},
            "truss",
            {"area": 1.0},
        )

    resolved = materials.resolve_section_properties(
        "Truss2",
        {"E": 210.0, "rho": 0.0},
        "truss",
        {"area": 1.0},
    )
    assert resolved.effective_properties["rho"] == 0.0


@pytest.mark.parametrize(
    ("element_type", "section_type", "properties"),
    (
        ("Tri3", "truss", {"area": 1.0}),
        ("Tet4", "rectangle", {"height": 1.0, "width": 1.0}),
        ("Truss2", "solid", {}),
        ("Beam2", "solid", {}),
    ),
)
def test_section_family_compatibility_fails_closed(
    element_type,
    section_type,
    properties,
) -> None:
    with pytest.raises(materials.SectionCompatibilityError):
        materials.resolve_section_properties(
            element_type,
            {"E": 210.0, "nu": 0.3},
            section_type,
            properties,
        )


@pytest.mark.parametrize(
    ("element_type", "section_type", "properties", "message"),
    (
        ("Tri3", "solid", {"thickness": 0.0}, "thickness"),
        ("Tet4", "solid", {"thickness": 1.0}, "does not use"),
        ("Truss2", "truss", {"area": 0.0}, "area"),
        (
            "Beam2",
            "hollow_circle",
            {"outer_radius": 1.0, "inner_radius": 1.0},
            "outer_radius",
        ),
        (
            "Beam2",
            "rectangle",
            {"height": 1.0, "width": 2.0, "radius": 3.0},
            "radius",
        ),
    ),
)
def test_section_property_validation_covers_known_invalid_shapes(
    element_type,
    section_type,
    properties,
    message,
) -> None:
    with pytest.raises(materials.SectionPropertyError, match=message):
        materials.resolve_section_properties(
            element_type,
            {"E": 210.0, "nu": 0.3},
            section_type,
            properties,
        )


def test_resolution_preserves_order_last_match_and_uncovered_facts() -> None:
    first = _element(1, "Truss2", custom="first")
    second = _element(2, "Truss2")
    uncovered = _element(3, "Truss2", E=9.0, area=9.0)
    model = _model(first, second, uncovered)
    model.materials = {
        "first": MaterialDefinition("first", {"E": 100.0}),
        "last": MaterialDefinition("last", {"E": 50.0}),
    }
    model.element_sets = {
        "BOTH": ElementSet("BOTH", (1, 2)),
        "SECOND": ElementSet("SECOND", (2,)),
    }
    model.sections = [
        SectionAssignment("BOTH", "first", "truss", {"area": 2.0}),
        SectionAssignment("SECOND", "last", "truss", {"area": 4.0}),
    ]
    before = deepcopy(model)

    resolution = materials.resolve_sections(model)

    assert resolution.passed
    assert resolution.assignment_order == (0, 1)
    assert [item.element_set for item in resolution.assignments] == [
        "BOTH",
        "SECOND",
    ]
    assert resolution.for_element(1).material == "first"
    assert resolution.for_element(1).effective_properties["area"] == 2.0
    assert resolution.for_element(2).material == "last"
    assert resolution.for_element(2).effective_properties["area"] == 4.0
    assert resolution.uncovered_element_ids == (3,)
    assert not resolution.fully_covered
    assert model.mesh.elements[0].props == before.mesh.elements[0].props
    assert model.mesh.elements[1].props == before.mesh.elements[1].props
    assert model.metadata == before.metadata


def test_resolution_aggregates_missing_and_incompatible_information() -> None:
    truss = _element(1, "Truss2")
    solid = _element(2, "Tet4")
    model = _model(truss, solid)
    model.materials["steel"] = MaterialDefinition(
        "steel",
        {"E": 210.0, "nu": 0.3},
    )
    model.element_sets = {
        "TRUSS": ElementSet("TRUSS", (1,)),
        "MISSING_ID": ElementSet("MISSING_ID", (99,)),
        "MIXED": ElementSet("MIXED", (1, 2)),
    }
    model.sections = [
        SectionAssignment("TRUSS", "missing", "truss", {"area": 1.0}),
        SectionAssignment("missing-set", "steel", "solid"),
        SectionAssignment("MISSING_ID", "steel", "solid"),
        SectionAssignment("MIXED", "steel", "truss", {"area": 1.0}),
    ]

    resolution = materials.resolve_sections(model)

    assert not resolution.passed
    assert resolution.missing_materials == ("missing",)
    assert resolution.missing_element_sets == ("missing-set",)
    assert resolution.missing_element_ids == (99,)
    assert resolution.incompatible_element_ids == (2,)
    assert resolution.for_element(1).assignment_index == 3
    assert resolution.uncovered_element_ids == ()
    assert [issue.code for issue in resolution.issues] == [
        "definition.material.missing",
        "definition.section.reference_missing",
        "definition.section.reference_missing",
        "definition.section.incompatible",
    ]


def test_resolution_supports_importer_internal_element_sets() -> None:
    element = _element(1, "Tet4")
    model = _model(element)
    model.materials["steel"] = MaterialDefinition(
        "steel",
        {"E": 210.0, "nu": 0.3},
    )
    model.metadata["_abaqus_internal_element_sets"] = {
        "_section_0_SOLID": ElementSet("_section_0_SOLID", (1,))
    }
    model.sections = [
        SectionAssignment("_section_0_SOLID", "steel", "solid")
    ]

    resolution = materials.resolve_sections(model)

    assert resolution.passed
    assert resolution.fully_covered
    assert resolution.uncovered_element_ids == ()
    assert resolution.for_element(1).element_set == "_section_0_SOLID"


def test_real_importer_internal_section_set_uses_the_same_resolution() -> None:
    fixture = (
        Path(__file__).parents[1]
        / "fixtures"
        / "inp"
        / "internal_section_set.inp"
    )
    model = read(fixture)

    resolution = materials.resolve_sections(model)
    materials.apply_sections(model)

    assert resolution.passed
    assert resolution.uncovered_element_ids == ()
    assert resolution.effective_assignments[0].element_set.startswith(
        "_section_"
    )
    assert model.mesh.elements[0].props["material"] == "STEEL"


def test_apply_sections_consumes_resolution_and_restores_baseline() -> None:
    element = _element(1, "CPE3", custom="base")
    model = _model(element)
    model.materials["steel"] = MaterialDefinition(
        "steel",
        {"E": 210.0, "nu": 0.3},
    )
    model.element_sets["DOMAIN"] = ElementSet("DOMAIN", (1,))
    model.sections = [
        SectionAssignment(
            "DOMAIN",
            "steel",
            "solid",
            {"thickness": 2.0},
        )
    ]
    resolved = materials.resolve_sections(model).for_element(1)

    materials.apply_sections(model)

    assert {
        name: element.props[name]
        for name in resolved.applied_properties
    } == resolved.applied_properties
    assert element.props["plane_type"] == "strain"
    assert element.props["custom"] == "base"

    model.sections.clear()
    materials.apply_sections(model)
    assert element.props == {"custom": "base"}


def test_legacy_truss_default_is_normalized_without_weakening_other_families() -> None:
    resolved = materials.resolve_section_properties(
        "Truss2",
        {"E": 210.0},
        "solid",
        {"area": 1.0},
    )

    assert resolved.section_type == "truss"
    assert resolved.section_family == "truss"

    plane = materials.resolve_section_properties(
        "Tri3",
        {"E": 210.0, "nu": 0.3},
        "plane",
        {"thickness": 1.0},
    )
    assert plane.section_type == "solid"


def test_later_invalid_assignment_cannot_leave_an_earlier_match_effective() -> None:
    element = _element(1, "Truss2")
    model = _model(element)
    model.materials["steel"] = MaterialDefinition("steel", {"E": 210.0})
    model.element_sets["BAR"] = ElementSet("BAR", (1,))
    model.sections = [
        SectionAssignment("BAR", "steel", "truss", {"area": 1.0}),
        SectionAssignment("BAR", "steel", "solid"),
    ]

    resolution = materials.resolve_sections(model)

    assert not resolution.passed
    assert resolution.for_element(1) is None
    assert resolution.uncovered_element_ids == ()
    assert resolution.incompatible_element_ids == (1,)
