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
