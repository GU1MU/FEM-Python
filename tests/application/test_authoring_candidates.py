from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from pathlib import Path

from fem.abaqus import read
from fem.application import (
    AuthoringStatus,
    BeamOrientation,
    ModelDefinitions,
    RegionAssignment,
    SectionDefinition,
    definitions_from_model,
    evaluate_authoring_candidate,
)
from fem.core.model import LineLoad


_FIXTURES = Path(__file__).parents[1] / "fixtures" / "inp"


def _beam_model_and_definitions():
    model = read(_FIXTURES / "beam2_rectangle_uniform_load.inp")
    return model, definitions_from_model(model)


def _with_orientation(
    definitions: ModelDefinitions,
    reference: tuple[float, float, float] | None,
) -> ModelDefinitions:
    assignment = definitions.assignments[0]
    orientation = (
        None if reference is None else BeamOrientation(reference)
    )
    return ModelDefinitions(
        materials=definitions.materials,
        sections=definitions.sections,
        assignments=(
            RegionAssignment(
                assignment.section_name,
                assignment.region_name,
                orientation,
            ),
        ),
        steps=definitions.steps,
    )


def test_automatic_rectangle_and_local_load_candidates_are_limited() -> None:
    model, definitions = _beam_model_and_definitions()

    assignment = evaluate_authoring_candidate(
        model,
        definitions,
        operation="section.rectangle",
        candidate=definitions.assignments[0],
        candidate_index=0,
    )
    line_load = evaluate_authoring_candidate(
        model,
        definitions,
        operation="load.line.local",
        candidate=LineLoad(
            "BEAM",
            (0.0, -500.0, 0.0),
            "local",
        ),
        step_name="UniformLoad",
    )

    assert assignment.status is AuthoringStatus.LIMITED
    assert line_load.status is AuthoringStatus.LIMITED
    assert not assignment.can_submit
    assert not line_load.can_submit
    assert assignment.diagnostics[0].code == "beam.orientation.assumed"
    assert line_load.diagnostics[0].code == "beam.orientation.assumed"


def test_explicit_rectangle_and_local_load_candidates_are_enabled() -> None:
    model, definitions = _beam_model_and_definitions()
    explicit = _with_orientation(definitions, (0.0, 1.0, 0.0))

    assignment = evaluate_authoring_candidate(
        model,
        explicit,
        operation="section.rectangle",
        candidate=explicit.assignments[0],
        candidate_index=0,
    )
    line_load = evaluate_authoring_candidate(
        model,
        explicit,
        operation="load.line.local",
        candidate=LineLoad(
            "BEAM",
            (0.0, -500.0, 0.0),
            "local",
        ),
        step_name="UniformLoad",
    )

    assert assignment.status is AuthoringStatus.ENABLED
    assert line_load.status is AuthoringStatus.ENABLED
    assert assignment.can_submit
    assert line_load.can_submit
    assert assignment.diagnostics == ()
    assert line_load.diagnostics == ()


def test_parallel_and_nonbeam_assignment_candidates_fail_closed() -> None:
    model, definitions = _beam_model_and_definitions()
    parallel = _with_orientation(definitions, (1.0, 0.0, 0.0))

    parallel_decision = evaluate_authoring_candidate(
        model,
        definitions,
        operation="section.rectangle",
        candidate=parallel.assignments[0],
        candidate_index=0,
    )

    truss = read(_FIXTURES / "truss2_tension.inp")
    truss_definitions = definitions_from_model(truss)
    rectangle = SectionDefinition(
        "Rectangle",
        truss_definitions.materials[0].name,
        "rectangle",
        {"height": 0.1, "width": 0.02},
    )
    nonbeam_definitions = ModelDefinitions(
        materials=truss_definitions.materials,
        sections=(*truss_definitions.sections, rectangle),
        assignments=truss_definitions.assignments,
        steps=truss_definitions.steps,
    )
    nonbeam_decision = evaluate_authoring_candidate(
        truss,
        nonbeam_definitions,
        operation="section.rectangle",
        candidate=RegionAssignment(
            "Rectangle",
            "TRUSS",
            BeamOrientation((0.0, 1.0, 0.0)),
        ),
    )

    assert parallel_decision.status is AuthoringStatus.UNAVAILABLE
    assert not parallel_decision.can_submit
    assert {
        item.code for item in parallel_decision.diagnostics
    } == {"beam.orientation.parallel"}
    assert all(
        item.details_dict()["operation"] == "section.rectangle"
        for item in parallel_decision.diagnostics
    )
    assert nonbeam_decision.status is AuthoringStatus.UNAVAILABLE
    assert nonbeam_decision.diagnostics[0].blocking
    assert nonbeam_decision.diagnostics[0].code == (
        "beam.orientation.unsupported_target"
    )


def test_circle_and_global_line_load_candidates_do_not_require_orientation() -> None:
    model, definitions = _beam_model_and_definitions()
    circle = replace(
        definitions.sections[0],
        section_type="solid_circle",
        properties={"radius": 0.05},
    )
    circle_definitions = ModelDefinitions(
        materials=definitions.materials,
        sections=(circle,),
        assignments=(
            RegionAssignment(circle.name, "BEAM"),
        ),
        steps=definitions.steps,
    )

    assignment = evaluate_authoring_candidate(
        model,
        circle_definitions,
        operation="section.solid_circle",
        candidate=circle_definitions.assignments[0],
        candidate_index=0,
    )
    global_load = evaluate_authoring_candidate(
        model,
        circle_definitions,
        operation="load.line.global",
        candidate=LineLoad(
            "BEAM",
            (0.0, -500.0, 0.0),
            "global",
        ),
        step_name="UniformLoad",
    )

    assert assignment.status is AuthoringStatus.ENABLED
    assert assignment.can_submit
    assert global_load.status is AuthoringStatus.ENABLED
    assert global_load.can_submit
    assert assignment.diagnostics == ()
    assert global_load.diagnostics == ()


def test_assignment_candidate_index_preserves_last_assignment_wins() -> None:
    model, definitions = _beam_model_and_definitions()
    later_section = replace(
        definitions.sections[0],
        name="Later",
    )
    overlap = ModelDefinitions(
        materials=definitions.materials,
        sections=(*definitions.sections, later_section),
        assignments=(
            RegionAssignment(
                definitions.sections[0].name,
                "BEAM",
            ),
            RegionAssignment(
                later_section.name,
                "BEAM",
                BeamOrientation((0.0, 1.0, 0.0)),
            ),
        ),
        steps=definitions.steps,
    )
    candidate = overlap.assignments[0]

    edit = evaluate_authoring_candidate(
        model,
        overlap,
        operation="section.rectangle",
        candidate=candidate,
        candidate_index=0,
    )
    create = evaluate_authoring_candidate(
        model,
        overlap,
        operation="section.rectangle",
        candidate=candidate,
    )

    assert edit.status is AuthoringStatus.ENABLED
    assert create.status is AuthoringStatus.LIMITED


def test_candidate_evaluation_does_not_mutate_inputs() -> None:
    model, definitions = _beam_model_and_definitions()
    model_before = deepcopy(model)
    definitions_before = deepcopy(definitions)

    evaluate_authoring_candidate(
        model,
        definitions,
        operation="section.rectangle",
        candidate=RegionAssignment(
            definitions.assignments[0].section_name,
            "BEAM",
            BeamOrientation((0.0, 1.0, 0.0)),
        ),
        candidate_index=0,
    )

    assert model == model_before
    assert definitions == definitions_before
