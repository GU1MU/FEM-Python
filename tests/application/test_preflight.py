from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from fem.abaqus import read
from fem.application import (
    PreflightDiagnostic,
    PreflightFacts,
    PreflightReport,
    PreflightSeverity,
    PreflightStage,
    TaskToken,
    run_static_preflight,
)
from fem.core.model import (
    AnalysisStep,
    DisplacementConstraint,
    NodalLoad,
)


_FIXTURES = Path(__file__).parents[1] / "fixtures" / "inp"


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
    assert {
        item.code for item in report.diagnostics
    } == {"output.request.not_executed"}


def test_valid_beam_reports_orientation_limitation_without_blocking() -> None:
    report = run_static_preflight(
        _read("beam2_rectangle_uniform_load.inp"),
        "UniformLoad",
    )

    assert report.passed
    assert report.numerical_stability_checked
    assert report.facts.line_load_count == 1
    assert {
        item.code for item in report.diagnostics
    } == {
        "beam.orientation.assumed",
        "output.request.not_executed",
    }


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


def test_output_request_warning_does_not_create_a_false_failure() -> None:
    model = _read("truss2_tension.inp")
    report = run_static_preflight(model, "Tension")
    warning = next(
        item
        for item in report.diagnostics
        if item.code == "output.request.not_executed"
    )

    assert warning.severity is PreflightSeverity.WARNING
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
