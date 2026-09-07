from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from fem import materials
from fem.materials import assignment as material_assignment
from fem.io.inp import read
from fem.core.model import ElementSet, MaterialDefinition, SectionAssignment
from fem.core.mesh import Element3D, Mesh3D, Node3D
from fem.elements import BEAM_LOCAL_Y_REFERENCE_KEY, BeamOrientation, resolve_beam_frame
from tests.helpers.model_builders import make_truss_workflow_model


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
        SectionAssignment("BOTH", "first", "truss", {"area": 2.0, "first_only": True}),
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

    materials.apply_sections(model)
    assert first.props["E"] == 100.0
    assert first.props["area"] == 2.0
    assert first.props["custom"] == "first"
    assert second.props["E"] == 50.0
    assert second.props["area"] == 4.0
    assert "first_only" not in second.props
    assert uncovered.props == before.mesh.elements[2].props


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


def test_real_importer_internal_section_set_uses_the_same_resolution() -> None:
    fixture = (
        Path(__file__).parents[1]
        / "helpers" / "fixtures"
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


def test_apply_sections_matches_resolved_plane_formulation() -> None:
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


def test_resolution_preserves_stable_beam_orientation_issue_codes() -> None:
    beam = _element(1, "Beam2")
    solid = _element(2, "Tet4")
    model = _model(beam, solid)
    model.materials["steel"] = MaterialDefinition(
        "steel",
        {"E": 210.0, "nu": 0.3},
    )
    model.element_sets = {
        "BEAM": ElementSet("BEAM", (1,)),
        "SOLID": ElementSet("SOLID", (2,)),
    }
    model.sections = [
        SectionAssignment(
            "BEAM",
            "steel",
            "rectangle",
            {
                "height": 1.0,
                "width": 2.0,
                BEAM_LOCAL_Y_REFERENCE_KEY: (0.0, 0.0, 0.0),
            },
        ),
        SectionAssignment(
            "SOLID",
            "steel",
            "solid",
            {BEAM_LOCAL_Y_REFERENCE_KEY: (0.0, 1.0, 0.0)},
        ),
    ]

    resolution = materials.resolve_sections(model)

    assert [issue.code for issue in resolution.issues] == [
        "beam.orientation.invalid",
        "beam.orientation.unsupported_target",
    ]

    with pytest.raises(ValueError) as invalid:
        resolution.require_valid()
    assert getattr(invalid.value, "code", None) == "beam.orientation.invalid"

    before = [dict(element.props) for element in model.mesh.elements]
    with pytest.raises(ValueError) as applied:
        materials.apply_sections(model)
    assert getattr(applied.value, "code", None) == "beam.orientation.invalid"
    assert [element.props for element in model.mesh.elements] == before


@pytest.mark.parametrize("element_type", ("Truss2", "Tet4"))
def test_uncovered_non_beam_cannot_keep_direct_orientation(
    element_type,
) -> None:
    element = _element(
        1,
        element_type,
        **{BEAM_LOCAL_Y_REFERENCE_KEY: (0.0, 1.0, 0.0)},
    )
    model = _model(element)

    with pytest.raises(ValueError) as caught:
        materials.apply_sections(model)

    assert getattr(caught.value, "code", None) == (
        "beam.orientation.unsupported_target"
    )
    assert element.props[BEAM_LOCAL_Y_REFERENCE_KEY] == (0.0, 1.0, 0.0)


def _beam_orientation_ownership_model():
    direct_reference = (0.0, 1.0, 0.0)
    element = Element3D(
        1,
        [1, 2],
        "Beam2",
        {
            "E": 10.0,
            "nu": 0.25,
            "section_type": "rectangle",
            "height": 3.0,
            "width": 1.0,
            BEAM_LOCAL_Y_REFERENCE_KEY: direct_reference,
            "custom": "direct",
        },
    )
    model = _model(element)
    model.mesh = Mesh3D(
        nodes=[
            Node3D(1, 0.0, 0.0, 0.0),
            Node3D(2, 2.0, 0.0, 0.0),
        ],
        elements=[element],
        dofs_per_node=6,
    )
    model.materials["steel"] = MaterialDefinition(
        "steel",
        {"E": 210.0, "nu": 0.3},
    )
    model.element_sets["BEAM"] = ElementSet("BEAM", (1,))
    return model, element, direct_reference


def _beam_assignment(reference=None):
    properties = {"height": 2.0, "width": 1.0}
    if reference is not None:
        properties[BEAM_LOCAL_Y_REFERENCE_KEY] = reference
    return SectionAssignment(
        "BEAM",
        "steel",
        "rectangle",
        properties,
    )


def test_apply_sections_clears_and_restores_owned_beam_orientation() -> None:
    model, element, direct_reference = _beam_orientation_ownership_model()
    model.sections = [_beam_assignment()]

    materials.apply_sections(model)

    assert BEAM_LOCAL_Y_REFERENCE_KEY not in element.props
    assert resolve_beam_frame(model.mesh, element).source == "automatic"

    assignment_reference = (0.0, 0.0, 1.0)
    model.sections = [_beam_assignment(BeamOrientation(assignment_reference))]
    materials.apply_sections(model)
    assert element.props[BEAM_LOCAL_Y_REFERENCE_KEY] == assignment_reference
    assert resolve_beam_frame(model.mesh, element).source == "explicit"

    model.sections = [_beam_assignment()]
    materials.apply_sections(model)
    assert BEAM_LOCAL_Y_REFERENCE_KEY not in element.props
    assert resolve_beam_frame(model.mesh, element).source == "automatic"

    model.sections.clear()
    materials.apply_sections(model)
    assert element.props[BEAM_LOCAL_Y_REFERENCE_KEY] == direct_reference
    assert element.props["custom"] == "direct"
    assert resolve_beam_frame(model.mesh, element).source == "explicit"


def test_section_ownership_survives_model_deepcopy() -> None:
    model, _, direct_reference = _beam_orientation_ownership_model()
    assignment_reference = (0.0, 0.0, 1.0)
    model.sections = [_beam_assignment(assignment_reference)]
    materials.apply_sections(model)

    copied = deepcopy(model)
    copied_element = copied.mesh.elements[0]
    copied.sections = [_beam_assignment()]
    materials.apply_sections(copied)
    assert BEAM_LOCAL_Y_REFERENCE_KEY not in copied_element.props

    copied.sections.clear()
    materials.apply_sections(copied)

    assert copied_element.props[BEAM_LOCAL_Y_REFERENCE_KEY] == (
        direct_reference
    )
    assert resolve_beam_frame(copied.mesh, copied_element).orientation == (
        BeamOrientation(direct_reference)
    )


def test_last_automatic_assignment_cannot_leak_earlier_orientation() -> None:
    model, element, _ = _beam_orientation_ownership_model()
    model.sections = [
        _beam_assignment((0.0, 0.0, 1.0)),
        _beam_assignment(),
    ]

    resolution = materials.resolve_sections(model)
    materials.apply_sections(model)

    effective = resolution.for_element(1)
    assert effective.assignment_index == 1
    assert BEAM_LOCAL_Y_REFERENCE_KEY not in effective.effective_properties
    assert BEAM_LOCAL_Y_REFERENCE_KEY not in element.props


def test_apply_sections_rejects_shadowed_parallel_orientation() -> None:
    model, element, direct_reference = _beam_orientation_ownership_model()
    model.sections = [
        _beam_assignment((1.0, 0.0, 0.0)),
        _beam_assignment((0.0, 1.0, 0.0)),
    ]
    before = deepcopy(element.props)

    with pytest.raises(ValueError) as caught:
        materials.apply_sections(model)

    assert getattr(caught.value, "code", None) == "beam.orientation.parallel"
    assert element.props == before
    assert element.props[BEAM_LOCAL_Y_REFERENCE_KEY] == direct_reference


def test_apply_sections_restores_original_properties_after_change_and_removal():
    original_props = {"E": 10.0, "area": 2.0, "custom": "base"}
    model = make_truss_workflow_model(element_props=dict(original_props))
    materials.add(
        model,
        materials.linear_elastic.material("steel", E=100.0, nu=0.3),
    )
    materials.assign(model, "steel", "bar", area=3.0, first_only="old")

    materials.apply_sections(model)
    elem = model.mesh.elements[0]
    assert elem.props["E"] == 100.0
    assert elem.props["area"] == 3.0
    assert elem.props["first_only"] == "old"

    model.sections.clear()
    materials.add(
        model,
        materials.linear_elastic.material("aluminum", E=50.0, nu=0.25),
    )
    materials.assign(model, "aluminum", "bar", area=4.0)
    materials.apply_sections(model)

    assert elem.props["E"] == 50.0
    assert elem.props["area"] == 4.0
    assert elem.props["custom"] == "base"
    assert "first_only" not in elem.props

    model.sections.clear()
    materials.apply_sections(model)

    assert elem.props == original_props


def test_apply_sections_rejects_missing_effective_beam2_section():
    model = _model(_element(1, "Beam2"))

    with pytest.raises(ValueError, match=r"Element 1.*section_type"):
        materials.apply_sections(model)


def test_apply_sections_does_not_restore_old_baseline_into_replaced_element():
    model = make_truss_workflow_model(element_props={"E": 10.0, "area": 2.0})
    materials.add(
        model,
        materials.linear_elastic.material("steel", E=100.0, nu=0.3),
    )
    materials.assign(model, "steel", "bar")
    materials.apply_sections(model)

    replacement = Element3D(
        1,
        [1, 2],
        "Truss2",
        {"E": 9.0, "area": 9.0, "replacement": True},
    )
    model.mesh.elements[0] = replacement
    materials.apply_sections(model)
    model.sections.clear()
    materials.apply_sections(model)

    assert replacement.props == {
        "E": 9.0,
        "area": 9.0,
        "replacement": True,
    }


@pytest.mark.parametrize("failure", ["material", "set", "element", "section"])
def test_apply_sections_resolution_failure_preserves_previous_state(failure):
    model = make_truss_workflow_model(element_props={"E": 10.0, "area": 2.0})
    materials.add(
        model,
        materials.linear_elastic.material("steel", E=100.0, nu=0.3),
    )
    materials.assign(model, "steel", "bar", area=3.0)
    materials.apply_sections(model)
    elem = model.mesh.elements[0]
    props_before = deepcopy(elem.props)
    metadata_before = deepcopy(model.metadata)

    if failure == "material":
        model.sections[:] = [SectionAssignment("bar", "missing")]
        message = "material missing is not defined"
    elif failure == "set":
        model.sections[:] = [SectionAssignment("missing", "steel")]
        message = "element set missing is not defined"
    elif failure == "element":
        model.element_sets["bar"] = ElementSet("bar", (999,))
        message = "element 999 is not defined"
    else:
        model.sections[:] = [SectionAssignment("bar", "steel", "truss", {"area": -1.0})]
        message = "Element 1.*area"

    error = ValueError if failure == "section" else KeyError
    with pytest.raises(error, match=message):
        materials.apply_sections(model)

    assert elem.props == props_before
    assert model.metadata == metadata_before


def test_apply_sections_commit_failure_rolls_back_props_and_metadata(monkeypatch):
    model = make_truss_workflow_model(element_props={"E": 10.0, "area": 2.0})
    materials.add(
        model,
        materials.linear_elastic.material("steel", E=100.0, nu=0.3),
    )
    materials.assign(model, "steel", "bar", area=3.0)
    materials.apply_sections(model)
    elem = model.mesh.elements[0]
    props_before = deepcopy(elem.props)
    metadata_before = deepcopy(model.metadata)
    original_restore = material_assignment._restore_tracked_keys

    def restore_then_fail(elem, keys, baseline):
        original_restore(elem, keys, baseline)
        raise RuntimeError("commit failed")

    monkeypatch.setattr(
        material_assignment,
        "_restore_tracked_keys",
        restore_then_fail,
    )

    with pytest.raises(RuntimeError, match="commit failed"):
        materials.apply_sections(model)

    assert elem.props == props_before
    assert model.metadata == metadata_before


def test_applied_properties_are_independent_of_definitions_and_other_elements():
    first = _element(1, "Truss2")
    second = _element(2, "Truss2")
    model = _model(first, second)
    model.materials["steel"] = MaterialDefinition("steel", {"E": 100.0})
    model.element_sets["ALL"] = ElementSet("ALL", (1, 2))
    model.sections = [SectionAssignment("ALL", "steel", "truss", {"area": 2.0})]

    materials.apply_sections(model)
    first.props["area"] = 99.0
    first.props["E"] = 50.0

    assert second.props["area"] == 2.0
    assert second.props["E"] == 100.0
    assert model.sections[0].properties == {"area": 2.0}
    assert model.materials["steel"].properties == {"E": 100.0}


def test_apply_sections_assigns_beam_material_and_section_properties():
    model, element, _ = _beam_orientation_ownership_model()
    element.props.clear()
    aluminum = materials.linear_elastic.material("aluminum", E=70.0, nu=0.33, rho=2.7)
    materials.add(model, aluminum)
    materials.assign(
        model, aluminum, model.element_sets["BEAM"],
        section_type="solid_circle", radius=0.02,
    )

    materials.apply_sections(model)

    assert {key: element.props[key] for key in (
        "material", "E", "nu", "rho", "section_type", "radius"
    )} == {
        "material": "aluminum", "E": 70.0, "nu": 0.33, "rho": 2.7,
        "section_type": "solid_circle", "radius": 0.02,
    }
