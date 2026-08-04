from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

import fem.application.preflight as preflight_module
from fem.io.inp import read
from fem.application import (
    PreflightDiagnostic,
    PreflightFacts,
    PreflightReport,
    PreflightSeverity,
    PreflightStage,
    RegionRef,
    TaskToken,
    run_static_preflight,
)
from fem.core.model import (
    AnalysisStep,
    DisplacementConstraint,
    LineLoad,
    NodalLoad,
    OutputRequest,
    SectionAssignment,
)
from fem.elements import BEAM_LOCAL_Y_REFERENCE_KEY


_FIXTURES = (
    Path(__file__).parents[1]
    / "fixtures"
    / "inp"
    / "abaqus_standard"
)


def _read(name: str):
    return read(_FIXTURES / name)


def test_report_outcome_is_derived_only_from_error_diagnostics() -> None:
    warning = PreflightDiagnostic(
        code="test.warning",
        severity=PreflightSeverity.WARNING,
        stage=PreflightStage.OUTPUT,
        message="warning",
    )
    error = PreflightDiagnostic(
        code="test.error",
        severity=PreflightSeverity.ERROR,
        stage=PreflightStage.STIFFNESS,
        message="error",
    )

    assert PreflightReport("Step", (warning,)).passed
    assert not PreflightReport("Step", (warning, error)).passed
    with pytest.raises(TypeError):
        PreflightReport("Step", passed=True)


def test_valid_collinear_truss_uses_actual_stiffness_and_passes() -> None:
    report = run_static_preflight(
        _read("truss2_tension.inp"),
        "Tension",
    )

    assert report.passed
    assert report.numerical_stability_checked
    assert report.facts.step_name == "Tension"
    assert report.facts.dof_count == 6
    assert report.facts.displacement_count == 5
    assert report.diagnostics == ()


def test_quick_preflight_skips_stiffness_without_mutating_model(
    monkeypatch,
) -> None:
    model = _read("truss2_tension.inp")
    before = deepcopy(model.mesh.elements[0].props)
    calls = []
    monkeypatch.setattr(
        preflight_module.static_linear,
        "validate_stiffness",
        lambda *_args, **_kwargs: calls.append(True),
    )

    report = run_static_preflight(
        model,
        "Tension",
        check_numerical_stability=False,
        copy_model=False,
    )

    assert report.passed
    assert not report.numerical_stability_checked
    assert calls == []
    assert model.mesh.elements[0].props == before
    assert {
        item.code for item in report.warnings
    } == {"static.stiffness.skipped_large_model"}


def test_large_model_quick_check_avoids_per_element_resolution(
    monkeypatch,
) -> None:
    model = _read("truss2_tension.inp")

    def unexpected(*_args, **_kwargs):
        raise AssertionError("full-model operation must not run")

    monkeypatch.setattr(
        preflight_module,
        "describe_model_capabilities",
        unexpected,
    )
    monkeypatch.setattr(
        preflight_module.materials,
        "resolve_sections",
        unexpected,
    )
    monkeypatch.setattr(
        preflight_module.static_linear,
        "validate_stiffness",
        unexpected,
    )

    report = run_static_preflight(
        model,
        "Tension",
        check_numerical_stability=False,
        copy_model=False,
        quick_check=True,
    )

    assert report.passed
    assert not report.numerical_stability_checked
    assert {
        item.code for item in report.warnings
    } == {
        "model.capability.sampled_large_model",
        "static.stiffness.skipped_large_model",
    }


def test_large_model_quick_check_keeps_section_coverage_blocking() -> None:
    model = _read("truss2_tension.inp")
    model.sections = []

    report = run_static_preflight(
        model,
        "Tension",
        check_numerical_stability=False,
        copy_model=False,
        quick_check=True,
    )
    codes = {item.code for item in report.errors}

    assert not report.passed
    assert "definition.section.missing" in codes
    assert "definition.section.unassigned_elements" in codes


def test_valid_beam_reports_orientation_limitation_without_blocking() -> None:
    model = _read("beam2_rectangle_uniform_load.inp")
    model.sections[0].properties.pop(
        BEAM_LOCAL_Y_REFERENCE_KEY,
        None,
    )
    report = run_static_preflight(
        model,
        "UniformLoad",
    )

    assert report.passed
    assert report.numerical_stability_checked
    assert report.facts.line_load_count == 1
    assert {
        item.code for item in report.diagnostics
    } == {"beam.orientation.assumed"}


def test_circle_with_global_load_has_no_orientation_warning() -> None:
    model = _read("beam2_rectangle_uniform_load.inp")
    original = model.sections[0]
    model.sections = [
        SectionAssignment(
            original.element_set,
            original.material,
            "solid_circle",
            {"radius": 0.05},
        )
    ]
    selected = next(
        item for item in model.steps if item.name == "UniformLoad"
    )
    selected.line_loads = (
        LineLoad("BEAM", (0.0, -500.0, 0.0), "global"),
    )

    report = run_static_preflight(model, "UniformLoad")

    assert report.passed
    assert "beam.orientation.assumed" not in {
        item.code for item in report.diagnostics
    }


def test_only_selected_step_local_load_requires_orientation() -> None:
    model = _read("beam2_rectangle_uniform_load.inp")
    original = model.sections[0]
    model.sections = [
        SectionAssignment(
            original.element_set,
            original.material,
            "solid_circle",
            {"radius": 0.05},
        )
    ]
    model.steps.append(
        AnalysisStep(
            "GlobalOnly",
            line_loads=(
                LineLoad(
                    "BEAM",
                    (0.0, -500.0, 0.0),
                    "global",
                ),
            ),
        )
    )

    unrelated = run_static_preflight(model, "GlobalOnly")
    selected = run_static_preflight(model, "UniformLoad")

    assert "beam.orientation.assumed" not in {
        item.code for item in unrelated.diagnostics
    }
    assert "beam.orientation.assumed" in {
        item.code for item in selected.diagnostics
    }


def test_explicit_orientation_removes_installed_and_local_warnings() -> None:
    model = _read("beam2_rectangle_uniform_load.inp")
    model.sections[0].properties[
        BEAM_LOCAL_Y_REFERENCE_KEY
    ] = (0.0, 1.0, 0.0)

    report = run_static_preflight(model, "UniformLoad")

    assert report.passed
    assert report.numerical_stability_checked
    assert "beam.orientation.assumed" not in {
        item.code for item in report.diagnostics
    }


def test_parallel_orientation_blocks_before_stiffness_with_typed_subject() -> None:
    model = _read("beam2_rectangle_uniform_load.inp")
    model.sections[0].properties[
        BEAM_LOCAL_Y_REFERENCE_KEY
    ] = (1.0, 0.0, 0.0)

    report = run_static_preflight(model, "UniformLoad")
    diagnostic = next(
        item
        for item in report.diagnostics
        if item.code == "beam.orientation.parallel"
    )

    assert not report.passed
    assert not report.numerical_stability_checked
    assert diagnostic.subject == RegionRef("element_set", "BEAM")
    assert diagnostic.details_dict()["element_id"] == 1
    assert diagnostic.details_dict()["operation"] == (
        "section.assignment"
    )


def test_invalid_orientation_blocks_before_stiffness_with_stable_code() -> None:
    model = _read("beam2_rectangle_uniform_load.inp")
    model.sections[0].properties[
        BEAM_LOCAL_Y_REFERENCE_KEY
    ] = (0.0, 0.0, 0.0)

    report = run_static_preflight(model, "UniformLoad")
    diagnostic = next(
        item
        for item in report.diagnostics
        if (
            item.code == "beam.orientation.invalid"
            and item.details_dict().get("operation")
            == "section.assignment"
        )
    )

    assert not report.passed
    assert not report.numerical_stability_checked
    assert diagnostic.subject == RegionRef("element_set", "BEAM")
    assert diagnostic.details_dict()["element_id"] == 1
    assert diagnostic.details_dict()["reference"] == (0.0, 0.0, 0.0)


def test_shadowed_parallel_orientation_still_blocks_preflight() -> None:
    model = _read("beam2_rectangle_uniform_load.inp")
    valid = model.sections[0]
    model.sections = [
        SectionAssignment(
            valid.element_set,
            valid.material,
            valid.section_type,
            {
                **valid.properties,
                BEAM_LOCAL_Y_REFERENCE_KEY: (1.0, 0.0, 0.0),
            },
        ),
        SectionAssignment(
            valid.element_set,
            valid.material,
            valid.section_type,
            {
                **valid.properties,
                BEAM_LOCAL_Y_REFERENCE_KEY: (0.0, 1.0, 0.0),
            },
        ),
    ]

    report = run_static_preflight(model, "UniformLoad")

    diagnostic = next(
        item
        for item in report.diagnostics
        if item.code == "beam.orientation.parallel"
    )
    assert not report.passed
    assert not report.numerical_stability_checked
    assert diagnostic.details_dict()["assignment_index"] == 0


def test_zero_length_beam_is_reported_as_structure_error() -> None:
    model = _read("beam2_rectangle_uniform_load.inp")
    element = model.mesh.elements[0]
    nodes = {node.id: node for node in model.mesh.nodes}
    first, second = (
        nodes[element.node_ids[0]],
        nodes[element.node_ids[1]],
    )
    second.x = first.x
    second.y = first.y
    second.z = first.z

    report = run_static_preflight(model, "UniformLoad")

    diagnostic = next(
        item
        for item in report.diagnostics
        if item.code == "model.structure.invalid"
    )
    assert not report.passed
    assert not report.numerical_stability_checked
    assert diagnostic.details_dict()["element_id"] == 1
    assert "zero length" in diagnostic.message


def test_nonbeam_orientation_target_is_blocked_before_stiffness() -> None:
    model = _read("truss2_tension.inp")
    original = model.sections[0]
    model.sections = [
        SectionAssignment(
            "TRUSS",
            original.material,
            "rectangle",
            {
                "height": 0.1,
                "width": 0.1,
                BEAM_LOCAL_Y_REFERENCE_KEY: (0.0, 1.0, 0.0),
            },
        )
    ]

    report = run_static_preflight(model, "Tension")
    diagnostic = next(
        item
        for item in report.diagnostics
        if item.code == "beam.orientation.unsupported_target"
    )

    assert not report.passed
    assert not report.numerical_stability_checked
    assert diagnostic.subject == RegionRef("element_set", "TRUSS")
    assert diagnostic.details_dict()["operation"] == (
        "section.assignment"
    )


def test_integer_local_load_warning_preserves_element_subject() -> None:
    model = _read("beam2_rectangle_uniform_load.inp")
    original = model.sections[0]
    model.sections = [
        SectionAssignment(
            original.element_set,
            original.material,
            "solid_circle",
            {"radius": 0.05},
        )
    ]
    selected = next(
        item for item in model.steps if item.name == "UniformLoad"
    )
    selected.line_loads = (
        LineLoad(1, (0.0, -500.0, 0.0), "local"),
    )

    report = run_static_preflight(model, "UniformLoad")
    warning = next(
        item
        for item in report.diagnostics
        if item.code == "beam.orientation.assumed"
    )

    assert warning.subject == 1
    assert warning.details_dict()["operation"] == "load.line.local"


def test_underconstrained_truss_returns_stable_singular_code() -> None:
    model = _read("truss2_tension.inp")
    initial = next(step for step in model.steps if step.name == "Initial")
    initial.boundaries = (
        DisplacementConstraint("FIXED", 1, 1, 0.0),
    )

    report = run_static_preflight(model, "Tension")

    assert not report.passed
    assert report.numerical_stability_checked
    assert "static.stiffness.singular" in {
        item.code for item in report.diagnostics
    }


def test_selected_step_is_not_polluted_by_unrelated_invalid_step() -> None:
    model = _read("truss2_tension.inp")
    model.steps.append(
        AnalysisStep(
            "Broken",
            cloads=(NodalLoad("MISSING", 1, 1.0),),
        )
    )

    selected = run_static_preflight(model, "Tension")
    broken = run_static_preflight(model, "Broken")

    assert selected.passed
    assert not broken.passed
    assert "step.reference.invalid" in {
        item.code for item in broken.diagnostics
    }


def test_invalid_inherited_initial_boundary_fails_selected_step() -> None:
    model = _read("truss2_tension.inp")
    initial = next(step for step in model.steps if step.name == "Initial")
    initial.boundaries = (
        DisplacementConstraint("MISSING", 1, 3, 0.0),
    )

    report = run_static_preflight(model, "Tension")

    assert not report.passed
    diagnostic = next(
        item
        for item in report.diagnostics
        if item.code == "step.reference.invalid"
    )
    assert diagnostic.subject == "Tension"


def test_missing_section_coverage_is_blocking_before_stiffness() -> None:
    model = _read("truss2_tension.inp")
    model.sections = []

    report = run_static_preflight(model, "Tension")
    codes = {item.code for item in report.diagnostics}

    assert not report.passed
    assert not report.numerical_stability_checked
    assert "definition.section.missing" in codes
    assert "definition.section.unassigned_elements" in codes


def test_unsupported_output_warning_does_not_create_a_false_failure() -> None:
    model = _read("truss2_tension.inp")
    step = next(item for item in model.steps if item.name == "Tension")
    step.outputs = (OutputRequest("field", "node", ("Future",)),)
    report = run_static_preflight(model, "Tension")
    warning = next(
        item
        for item in report.diagnostics
        if item.stage is PreflightStage.OUTPUT
    )

    assert warning.code == "output.request.variable_unsupported"
    assert warning.severity is PreflightSeverity.WARNING
    assert report.passed


def test_preflight_emits_only_projected_unsupported_output_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _read("truss2_tension.inp")
    step = next(item for item in model.steps if item.name == "Tension")
    step.outputs = (
        OutputRequest("field", "node", ("U",)),
        OutputRequest("field", "node", ("Future",)),
    )
    observed = []
    original = preflight_module._evaluate_output_requests

    def evaluate(candidate_model, outputs):
        requests = tuple(outputs)
        evaluation = original(candidate_model, requests)
        observed.append((requests, evaluation))
        return evaluation

    monkeypatch.setattr(
        preflight_module,
        "_evaluate_output_requests",
        evaluate,
    )

    report = run_static_preflight(model, "Tension")
    output_diagnostics = tuple(
        diagnostic
        for diagnostic in report.diagnostics
        if diagnostic.stage is PreflightStage.OUTPUT
    )

    assert len(observed) == 1
    requests, evaluation = observed[0]
    assert requests == step.outputs
    assert evaluation.complete
    assert tuple(
        projection.executable
        for projection in evaluation.projections
    ) == (True, False)
    assert len(output_diagnostics) == 1
    assert (
        output_diagnostics[0].code
        == "output.request.variable_unsupported"
    )
    assert output_diagnostics[0].path == (
        "steps",
        "Tension",
        "outputs",
        "1",
        "variables",
        "0",
    )
    assert dict(output_diagnostics[0].details)["request_index"] == 1
    assert report.passed


def test_report_is_bound_to_validation_token_provenance() -> None:
    token = TaskToken(
        session_id="session-1",
        task_id="task-1",
        task_kind="validation",
        dependency_revisions=(("model_revision", 7),),
        artifact_id="artifact-1",
        step_name="Tension",
    )

    report = run_static_preflight(
        _read("truss2_tension.inp"),
        "Tension",
        token=token,
    )

    assert report.session_id == "session-1"
    assert report.artifact_id == "artifact-1"
    assert report.model_revision == 7
    assert report.step_name == "Tension"


def test_preflight_does_not_mutate_the_caller_model() -> None:
    model = _read("truss2_tension.inp")
    before = deepcopy(model.mesh.elements[0].props)

    run_static_preflight(model, "Tension")

    assert model.mesh.elements[0].props == before


def test_facts_reject_mutable_or_untyped_values() -> None:
    with pytest.raises(TypeError, match="PreflightFacts"):
        PreflightReport("Step", facts={"node_count": 1})

    facts = PreflightFacts(step_name="Step", node_count=2)
    report = PreflightReport("Step", facts=facts)
    assert report.facts.node_count == 2
