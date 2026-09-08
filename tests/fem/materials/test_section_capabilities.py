from __future__ import annotations

import pytest

from fem import materials
from fem.elements import BEAM_LOCAL_Y_REFERENCE_KEY


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
    ("section_type", "properties"),
    (
        ("rectangle", {"height": 4.0, "width": 2.0}),
        ("solid_circle", {"radius": 2.0}),
        ("hollow_circle", {"outer_radius": 2.0, "inner_radius": 1.0}),
    ),
    ids=("rectangle", "solid-circle", "hollow-circle"),
)
def test_beam_presets_preserve_validated_dimensions(section_type, properties) -> None:
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
        for name in properties
    } == properties


def test_plane_defaults_use_explicit_formulation_not_source_provenance() -> None:
    material = {"E": 210.0, "nu": 0.3}

    inherited = materials.resolve_section_properties(
        "Tri3",
        material,
        "solid",
        {},
        baseline_properties={
            "plane_type": "strain",
            "abaqus_type": "CPS3",
            "thickness": 2.5,
        },
    )
    provenance_only = materials.resolve_section_properties(
        "Tri3",
        material,
        "solid",
        {},
        baseline_properties={
            "abaqus_type": "CPE3",
            "thickness": 1.5,
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
    assert provenance_only.effective_properties["plane_type"] == "stress"
    assert provenance_only.effective_properties["thickness"] == 1.5
    assert overridden.effective_properties["plane_type"] == "stress"
    assert overridden.effective_properties["thickness"] == 3.0


@pytest.mark.parametrize("properties", ({}, {"E": 0.0}), ids=("missing", "zero"))
def test_truss_rejects_missing_or_invalid_elastic_modulus(
    properties,
) -> None:
    with pytest.raises(materials.MaterialPropertyError, match="E"):
        materials.resolve_section_properties(
            "Truss2",
            properties,
            "truss",
            {"area": 1.0},
        )


@pytest.mark.parametrize(
    ("element_type", "nu"),
    (
        ("Tri3", None),
        ("Tri3", 0.5),
        ("Tet4", None),
        ("Tet4", 0.5),
        ("Beam2", None),
        ("Beam2", 0.5),
    ),
)
def test_continuum_and_beam_reject_missing_or_invalid_poisson_ratio(
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


@pytest.mark.parametrize("rho", (-1.0, None))
def test_section_rejects_invalid_density_when_present(rho) -> None:
    with pytest.raises(materials.MaterialPropertyError, match="rho"):
        materials.resolve_section_properties(
            "Truss2",
            {"E": 210.0, "rho": rho},
            "truss",
            {"area": 1.0},
        )


def test_zero_density_is_allowed() -> None:
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
def test_section_rejects_incompatible_element_family(
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


def test_beam_orientation_property_is_canonical_and_assignment_owned() -> None:
    reference = [0, 2, 0]

    resolved = materials.resolve_section_properties(
        "Beam2",
        {"E": 210.0, "nu": 0.3},
        "rectangle",
        {
            "height": 1.0,
            "width": 2.0,
            BEAM_LOCAL_Y_REFERENCE_KEY: reference,
        },
    )
    reference[1] = 9

    assert resolved.applied_properties[BEAM_LOCAL_Y_REFERENCE_KEY] == (
        0.0,
        2.0,
        0.0,
    )
    assert resolved.effective_properties[BEAM_LOCAL_Y_REFERENCE_KEY] == (
        0.0,
        2.0,
        0.0,
    )


def test_covered_automatic_beam_suppresses_direct_baseline_orientation() -> None:
    resolved = materials.resolve_section_properties(
        "Beam2",
        {"E": 210.0, "nu": 0.3},
        "rectangle",
        {"height": 1.0, "width": 2.0},
        baseline_properties={
            BEAM_LOCAL_Y_REFERENCE_KEY: (0.0, 1.0, 0.0),
        },
    )

    assert BEAM_LOCAL_Y_REFERENCE_KEY not in resolved.applied_properties
    assert BEAM_LOCAL_Y_REFERENCE_KEY not in resolved.effective_properties


@pytest.mark.parametrize("element_type", ("Tri3", "Tet4", "Truss2"))
def test_non_beam_section_rejects_beam_orientation_property(
    element_type,
) -> None:
    section_type = "truss" if element_type == "Truss2" else "solid"
    section_properties = (
        {"area": 1.0}
        if element_type == "Truss2"
        else {"thickness": 1.0}
        if element_type == "Tri3"
        else {}
    )
    section_properties[BEAM_LOCAL_Y_REFERENCE_KEY] = (0.0, 1.0, 0.0)

    with pytest.raises(
        materials.SectionPropertyError,
        match="only with Beam",
    ):
        materials.resolve_section_properties(
            element_type,
            {"E": 210.0, "nu": 0.3},
            section_type,
            section_properties,
        )


def test_unrepresentable_beam_orientation_keeps_typed_section_error() -> None:
    with pytest.raises(materials.SectionPropertyError) as caught:
        materials.resolve_section_properties(
            "Beam2",
            {"E": 210.0, "nu": 0.3},
            "rectangle",
            {
                "height": 1.0,
                "width": 2.0,
                BEAM_LOCAL_Y_REFERENCE_KEY: (10**10000, 0.0, 0.0),
            },
        )

    assert getattr(caught.value, "code", None) == "beam.orientation.invalid"
