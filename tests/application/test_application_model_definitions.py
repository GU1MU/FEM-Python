from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from fem.application import (
    DefinitionRejected,
    ModelDefinitions,
    ModelSession,
    NativePart,
    RegionAssignment,
    RegionRef,
    SectionDefinition,
    TokenStatus,
    compile_model_definitions,
    definitions_from_model,
    normalize_model_definitions,
)
from fem.core.model import (
    AnalysisStep,
    ElementSet,
    MaterialDefinition,
    SectionAssignment,
)
from fem.elements import BeamOrientation


def _base_model() -> SimpleNamespace:
    return SimpleNamespace(
        materials={},
        sections=[],
        steps=[],
        element_sets={"DOMAIN": ElementSet("DOMAIN", (1,))},
        metadata={},
        mesh=SimpleNamespace(
            nodes=[],
            elements=[
                SimpleNamespace(
                    id=1,
                    type="Tri3",
                    node_ids=(1, 2, 3),
                    props={},
                )
            ],
        ),
    )


def _definitions() -> ModelDefinitions:
    return normalize_model_definitions(
        (MaterialDefinition("Steel", {"E": 210_000.0, "nu": 0.3}),),
        (
            SectionDefinition(
                "Solid",
                "Steel",
                "solid",
                {"thickness": 2.0},
            ),
        ),
        (RegionAssignment("Solid", "DOMAIN"),),
        (AnalysisStep("Step-1", metadata={"nlgeom": False}),),
    )


def _beam_base_model(
    *elements: SimpleNamespace,
) -> SimpleNamespace:
    if not elements:
        elements = (
            SimpleNamespace(
                id=1,
                type="Beam2",
                node_ids=(1, 2),
                props={},
            ),
        )
    node_ids = tuple(
        dict.fromkeys(
            node_id
            for element in elements
            for node_id in element.node_ids
        )
    )
    nodes = {
        1: SimpleNamespace(id=1, x=0.0, y=0.0, z=0.0),
        2: SimpleNamespace(id=2, x=1.0, y=0.0, z=0.0),
        3: SimpleNamespace(id=3, x=0.0, y=1.0, z=0.0),
        4: SimpleNamespace(id=4, x=0.0, y=0.0, z=1.0),
    }
    return SimpleNamespace(
        materials={},
        sections=[],
        steps=[],
        element_sets={
            "BEAMS": ElementSet(
                "BEAMS",
                tuple(element.id for element in elements),
            )
        },
        metadata={},
        mesh=SimpleNamespace(
            nodes=[nodes[node_id] for node_id in node_ids],
            elements=list(elements),
        ),
    )


def _beam_definitions(
    orientation: BeamOrientation | None,
) -> ModelDefinitions:
    return normalize_model_definitions(
        (
            MaterialDefinition(
                "Steel",
                {"E": 210_000.0, "nu": 0.3},
            ),
        ),
        (
            SectionDefinition(
                "Rectangle",
                "Steel",
                "rectangle",
                {"height": 0.1, "width": 0.02},
            ),
        ),
        (
            RegionAssignment(
                "Rectangle",
                "BEAMS",
                orientation,
            ),
        ),
        (),
    )


def test_normalization_takes_deep_ownership_and_trims_names() -> None:
    properties = {"E": 10.0, "nested": {"value": 1}}
    metadata = {"nested": {"value": 2}}
    material = MaterialDefinition(" Steel ", properties)
    step = AnalysisStep(" Step-1 ", metadata=metadata)

    definitions = normalize_model_definitions(
        (material,),
        (SectionDefinition(" Solid ", "Steel"),),
        (RegionAssignment("Solid", " DOMAIN "),),
        (step,),
    )
    properties["nested"]["value"] = 9
    metadata["nested"]["value"] = 9
    step.metadata["nested"]["value"] = 8

    assert definitions.materials[0].name == "Steel"
    assert definitions.materials[0].properties["nested"]["value"] == 1
    assert definitions.sections[0].name == "Solid"
    assert definitions.assignments[0].region_name == "DOMAIN"
    assert definitions.steps[0].name == "Step-1"
    assert definitions.steps[0].metadata["nested"]["value"] == 2


def test_normalization_owns_typed_beam_orientation() -> None:
    orientation = BeamOrientation((0.0, 1.0, 0.0))

    definitions = _beam_definitions(orientation)

    assert definitions.assignments[0].beam_orientation == orientation
    assert definitions.assignments[0].beam_orientation.local_y_reference == (
        0.0,
        1.0,
        0.0,
    )


@pytest.mark.parametrize(
    "orientation",
    (
        {"local_y_reference": (0.0, 1.0, 0.0)},
        (0.0, 1.0, 0.0),
    ),
    ids=("mapping", "sequence"),
)
def test_normalization_rejects_untyped_beam_orientation(
    orientation: object,
) -> None:
    with pytest.raises(DefinitionRejected) as caught:
        normalize_model_definitions(
            (MaterialDefinition("Steel", {}),),
            (SectionDefinition("Rectangle", "Steel", "rectangle"),),
            (RegionAssignment("Rectangle", "BEAMS", orientation),),
            (),
        )

    diagnostic = caught.value.diagnostics[0]
    assert diagnostic.code == "beam.orientation.invalid"
    assert diagnostic.subject == RegionRef("element_set", "BEAMS")
    assert diagnostic.path == (
        "definitions",
        "assignments",
        "0",
        "beam_orientation",
    )


def test_normalization_rejects_reserved_section_orientation_property() -> None:
    with pytest.raises(DefinitionRejected) as caught:
        normalize_model_definitions(
            (MaterialDefinition("Steel", {}),),
            (
                SectionDefinition(
                    "Rectangle",
                    "Steel",
                    "rectangle",
                    {"beam_local_y_reference": (0.0, 1.0, 0.0)},
                ),
            ),
            (RegionAssignment("Rectangle", "BEAMS"),),
            (),
        )

    diagnostic = caught.value.diagnostics[0]
    assert diagnostic.code == "beam.orientation.invalid"
    assert diagnostic.path[-1] == "beam_local_y_reference"


def test_normalization_rejects_reserved_orientation_on_unused_material() -> None:
    with pytest.raises(DefinitionRejected) as caught:
        normalize_model_definitions(
            (
                MaterialDefinition(
                    "Unused",
                    {
                        "beam_local_y_reference": (
                            0.0,
                            1.0,
                            0.0,
                        )
                    },
                ),
            ),
            (),
            (),
            (),
        )

    diagnostic = caught.value.diagnostics[0]
    assert diagnostic.code == "beam.orientation.invalid"
    assert diagnostic.path == (
        "definitions",
        "materials",
        "0",
        "properties",
        "beam_local_y_reference",
    )


def test_compile_returns_a_detached_model_and_preserves_definition_order() -> None:
    base = _base_model()
    definitions = _definitions()

    result = compile_model_definitions(base, definitions)
    compiled = result.require_model()

    assert result.passed
    assert compiled is not base
    assert base.materials == {}
    assert base.sections == []
    assert base.steps == []
    assert tuple(compiled.materials) == ("Steel",)
    assert compiled.sections == [
        SectionAssignment(
            "DOMAIN",
            "Steel",
            "solid",
            {"thickness": 2.0},
        )
    ]
    assert [step.name for step in compiled.steps] == ["Step-1"]


def test_compile_supports_importer_internal_element_sets() -> None:
    base = _base_model()
    base.element_sets = {}
    base.metadata["_abaqus_internal_element_sets"] = {
        "_INTERNAL": ElementSet("_INTERNAL", (1,))
    }
    definitions = _definitions()
    internal = ModelDefinitions(
        definitions.materials,
        definitions.sections,
        (RegionAssignment("Solid", "_INTERNAL"),),
        definitions.steps,
    )

    compiled = compile_model_definitions(base, internal).require_model()

    assert compiled.sections[0].element_set == "_INTERNAL"


def test_beam_orientation_compiles_per_assignment_and_projects_back() -> None:
    orientation = BeamOrientation((0.0, 1.0, 0.0))
    result = compile_model_definitions(
        _beam_base_model(),
        _beam_definitions(orientation),
    )

    compiled = result.require_model()
    assert compiled.sections[0].properties[
        "beam_local_y_reference"
    ] == (0.0, 1.0, 0.0)
    assert (
        "beam_local_y_reference"
        not in compiled.mesh.elements[0].props
    )

    projected = definitions_from_model(compiled)

    assert projected.assignments[0].beam_orientation == orientation
    assert (
        "beam_local_y_reference"
        not in projected.sections[0].properties
    )


def test_same_beam_section_can_have_independent_region_orientations() -> None:
    first = SimpleNamespace(
        id=1,
        type="Beam2",
        node_ids=(1, 2),
        props={},
    )
    second = SimpleNamespace(
        id=2,
        type="Beam2",
        node_ids=(1, 2),
        props={},
    )
    model = _beam_base_model(first, second)
    model.element_sets = {
        "FIRST": ElementSet("FIRST", (1,)),
        "SECOND": ElementSet("SECOND", (2,)),
    }
    definitions = normalize_model_definitions(
        (
            MaterialDefinition(
                "Steel",
                {"E": 210_000.0, "nu": 0.3},
            ),
        ),
        (
            SectionDefinition(
                "Rectangle",
                "Steel",
                "rectangle",
                {"height": 0.1, "width": 0.02},
            ),
        ),
        (
            RegionAssignment(
                "Rectangle",
                "FIRST",
                BeamOrientation((0.0, 1.0, 0.0)),
            ),
            RegionAssignment(
                "Rectangle",
                "SECOND",
                BeamOrientation((0.0, 0.0, 1.0)),
            ),
        ),
        (),
    )

    compiled = compile_model_definitions(
        model,
        definitions,
    ).require_model()

    assert tuple(
        section.properties["beam_local_y_reference"]
        for section in compiled.sections
    ) == (
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    )


def test_parallel_beam_orientation_is_a_typed_compile_rejection() -> None:
    result = compile_model_definitions(
        _beam_base_model(),
        _beam_definitions(BeamOrientation((1.0, 0.0, 0.0))),
    )

    assert not result.passed
    assert result.model is None
    diagnostic = result.diagnostics[0]
    assert diagnostic.code == "beam.orientation.parallel"
    assert diagnostic.subject == RegionRef("element_set", "BEAMS")
    assert diagnostic.details_dict()["element_id"] == 1
    assert diagnostic.details_dict()["reference"] == (1.0, 0.0, 0.0)


def test_one_parallel_element_rejects_a_multi_element_assignment() -> None:
    model = _beam_base_model(
        SimpleNamespace(
            id=1,
            type="Beam2",
            node_ids=(1, 2),
            props={},
        ),
        SimpleNamespace(
            id=2,
            type="Beam2",
            node_ids=(1, 3),
            props={},
        ),
    )

    result = compile_model_definitions(
        model,
        _beam_definitions(BeamOrientation((0.0, 1.0, 0.0))),
    )

    assert not result.passed
    parallel = next(
        item
        for item in result.diagnostics
        if item.code == "beam.orientation.parallel"
    )
    assert parallel.details_dict()["element_id"] == 2


def test_shadowed_parallel_assignment_still_rejects_compile() -> None:
    model = _beam_base_model()
    definitions = _beam_definitions(
        BeamOrientation((1.0, 0.0, 0.0))
    )
    definitions = ModelDefinitions(
        definitions.materials,
        definitions.sections,
        (
            definitions.assignments[0],
            RegionAssignment(
                "Rectangle",
                "BEAMS",
                BeamOrientation((0.0, 1.0, 0.0)),
            ),
        ),
        definitions.steps,
    )

    result = compile_model_definitions(model, definitions)

    assert not result.passed
    diagnostic = result.diagnostics[0]
    assert diagnostic.code == "beam.orientation.parallel"
    assert diagnostic.details_dict()["assignment_index"] == 0
    assert diagnostic.details_dict()["element_id"] == 1


def test_beam_orientation_rejects_non_beam_assignment_target() -> None:
    model = _base_model()
    model.element_sets = {
        "BEAMS": ElementSet("BEAMS", (1,)),
    }
    result = compile_model_definitions(
        model,
        _beam_definitions(BeamOrientation((0.0, 1.0, 0.0))),
    )

    assert not result.passed
    assert result.model is None
    diagnostic = result.diagnostics[0]
    assert diagnostic.code == "beam.orientation.unsupported_target"
    assert diagnostic.subject == RegionRef("element_set", "BEAMS")
    assert diagnostic.details_dict()["element_id"] == 1


def test_beam_orientation_rejects_empty_assignment_target() -> None:
    model = _beam_base_model()
    model.element_sets["BEAMS"] = ElementSet("BEAMS", ())

    result = compile_model_definitions(
        model,
        _beam_definitions(BeamOrientation((0.0, 1.0, 0.0))),
    )

    assert not result.passed
    assert result.model is None
    diagnostic = result.diagnostics[0]
    assert diagnostic.code == "beam.orientation.unsupported_target"
    assert diagnostic.subject == RegionRef("element_set", "BEAMS")
    assert diagnostic.details_dict()["element_set"] == "BEAMS"
    assert "element_id" not in diagnostic.details_dict()


def test_compile_failure_has_diagnostics_and_no_installable_model() -> None:
    definitions = _definitions()
    invalid = ModelDefinitions(
        definitions.materials,
        definitions.sections,
        (RegionAssignment("Solid", "MISSING"),),
        definitions.steps,
    )

    result = compile_model_definitions(_base_model(), invalid)

    assert not result.passed
    assert result.model is None
    assert result.diagnostics[0].blocking
    try:
        result.require_model()
    except DefinitionRejected as error:
        assert error.diagnostics == result.diagnostics
    else:
        raise AssertionError("failed compile must reject model installation")


def test_projection_returns_owned_editable_definitions() -> None:
    model = _base_model()
    model.materials = {
        "Steel": MaterialDefinition("Steel", {"E": 1.0, "nu": 0.3})
    }
    model.sections = [
        SectionAssignment("DOMAIN", "Steel", "solid", {"thickness": 3.0})
    ]
    model.steps = [AnalysisStep("Step-1", metadata={"nested": {"value": 1}})]

    projected = definitions_from_model(model)
    model.materials["Steel"].properties["E"] = 99.0
    model.steps[0].metadata["nested"]["value"] = 99

    assert projected.materials[0].properties["E"] == 1.0
    assert projected.sections[0].properties["thickness"] == 3.0
    assert projected.assignments[0] == RegionAssignment(
        "Section-1",
        "DOMAIN",
    )
    assert projected.steps[0].metadata["nested"]["value"] == 1


def test_projection_rejects_invalid_historical_beam_orientation() -> None:
    model = _beam_base_model()
    model.materials = {
        "Steel": MaterialDefinition(
            "Steel",
            {"E": 210_000.0, "nu": 0.3},
        )
    }
    model.sections = [
        SectionAssignment(
            "BEAMS",
            "Steel",
            "rectangle",
            {
                "height": 0.1,
                "width": 0.02,
                "beam_local_y_reference": (0.0, 1.0),
            },
        )
    ]

    with pytest.raises(DefinitionRejected) as caught:
        definitions_from_model(model)

    diagnostic = caught.value.diagnostics[0]
    assert diagnostic.code == "beam.orientation.invalid"
    assert diagnostic.subject == RegionRef("element_set", "BEAMS")


def test_projection_rejects_parallel_orientation_on_any_target_element() -> None:
    model = _beam_base_model(
        SimpleNamespace(
            id=1,
            type="Beam2",
            node_ids=(1, 2),
            props={},
        ),
        SimpleNamespace(
            id=2,
            type="Beam2",
            node_ids=(1, 3),
            props={},
        ),
    )
    model.materials = {
        "Steel": MaterialDefinition(
            "Steel",
            {"E": 210_000.0, "nu": 0.3},
        )
    }
    model.sections = [
        SectionAssignment(
            "BEAMS",
            "Steel",
            "rectangle",
            {
                "height": 0.1,
                "width": 0.02,
                "beam_local_y_reference": (0.0, 1.0, 0.0),
            },
        )
    ]

    with pytest.raises(DefinitionRejected) as caught:
        definitions_from_model(model)

    diagnostic = caught.value.diagnostics[0]
    assert diagnostic.code == "beam.orientation.parallel"
    assert diagnostic.subject == RegionRef("element_set", "BEAMS")
    assert diagnostic.details_dict()["element_id"] == 2


def test_projection_rejects_orientation_on_non_beam_target() -> None:
    model = _base_model()
    model.materials = {
        "Steel": MaterialDefinition("Steel", {"E": 1.0, "nu": 0.3})
    }
    model.sections = [
        SectionAssignment(
            "DOMAIN",
            "Steel",
            "solid",
            {"beam_local_y_reference": (0.0, 1.0, 0.0)},
        )
    ]

    with pytest.raises(DefinitionRejected) as caught:
        definitions_from_model(model)

    diagnostic = caught.value.diagnostics[0]
    assert diagnostic.code == "beam.orientation.unsupported_target"
    assert diagnostic.subject == RegionRef("element_set", "DOMAIN")


def test_failed_session_definitions_command_is_atomic() -> None:
    session = ModelSession()
    session.new_native_project()
    session.replace_geometry((NativePart(),), {"kind": "box"})
    session.replace_model_definitions(
        _definitions().materials,
        _definitions().sections,
        _definitions().assignments,
        _definitions().steps,
    )
    before = session.snapshot()

    try:
        session.replace_model_definitions(
            before.material_definitions,
            before.section_definitions,
            (RegionAssignment("Missing", "DOMAIN"),),
            before.analysis_definitions,
        )
    except DefinitionRejected:
        pass
    else:
        raise AssertionError("invalid definition links must be rejected")

    after = session.snapshot()
    assert after.session_revision == before.session_revision
    assert after.project_revision == before.project_revision
    assert after.model_revision == before.model_revision
    assert after.material_definitions == before.material_definitions
    assert after.section_definitions == before.section_definitions
    assert after.region_assignments == before.region_assignments
    assert after.analysis_definitions == before.analysis_definitions


def test_failed_compile_preserves_artifact_validation_token_and_revisions() -> None:
    definitions = _definitions()
    imported_model = compile_model_definitions(
        _base_model(),
        definitions,
    ).require_model()
    session = ModelSession()
    task = session.prepare_import(Path("owned-model.inp"))
    session.accept_imported_model(task.token, imported_model)
    validation = session.prepare_validation("Step-1")
    before = session.snapshot()

    with pytest.raises(DefinitionRejected):
        session.replace_model_definitions(
            definitions.materials,
            definitions.sections,
            (RegionAssignment("Solid", "MISSING"),),
            definitions.steps,
        )

    after = session.snapshot()
    assert after.session_revision == before.session_revision
    assert after.project_revision == before.project_revision
    assert after.model_revision == before.model_revision
    assert after.artifact.artifact_id == before.artifact.artifact_id
    assert after.validations == before.validations
    assert after.runs == before.runs
    assert session.validate_task_token(validation.token) is (
        TokenStatus.CURRENT
    )


def test_parallel_orientation_session_command_is_atomic() -> None:
    imported_model = compile_model_definitions(
        _beam_base_model(),
        _beam_definitions(None),
    ).require_model()
    session = ModelSession()
    task = session.prepare_import(Path("beam.inp"))
    session.accept_imported_model(task.token, imported_model)
    before = session.snapshot()

    with pytest.raises(DefinitionRejected) as caught:
        session.replace_model_definitions(
            before.material_definitions,
            before.section_definitions,
            (
                RegionAssignment(
                    before.region_assignments[0].section_name,
                    "BEAMS",
                    BeamOrientation((1.0, 0.0, 0.0)),
                ),
            ),
            before.analysis_definitions,
        )

    assert caught.value.diagnostics[0].code == (
        "beam.orientation.parallel"
    )
    after = session.snapshot()
    assert after.session_revision == before.session_revision
    assert after.project_revision == before.project_revision
    assert after.model_revision == before.model_revision
    assert after.region_assignments == before.region_assignments
    assert after.artifact.artifact_id == before.artifact.artifact_id
