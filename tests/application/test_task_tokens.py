from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from fem.abaqus import read
from fem.application import (
    BeamOrientation,
    ModelSession,
    NativePart,
    PreflightReport,
    RegionAssignment,
    TokenStatus,
)
from fem.core.model import AnalysisStep
from fem.geometry.recipes import BoxGeometry
from tests.helpers.preflight_builders import passing_preflight_report


_FIXTURES = (
    Path(__file__).parents[1]
    / "fixtures"
    / "inp"
    / "abaqus_standard"
)


def _model(*step_names: str) -> SimpleNamespace:
    return SimpleNamespace(
        materials={},
        sections=[],
        steps=[AnalysisStep(name) for name in step_names],
        element_sets={},
        metadata={},
        mesh=SimpleNamespace(nodes=[], elements=[]),
    )


def _session() -> ModelSession:
    session = ModelSession()
    session.new_native_project()
    session.replace_geometry(
        (NativePart(),), BoxGeometry("Box", 1.0, 1.0, 1.0)
    )
    session.replace_model_definitions((), (), (), (AnalysisStep("Step-A"),))
    mesh = session.prepare_mesh_generation()
    session.accept_generated_model(mesh.token, _model("Step-A"))
    validation = session.prepare_validation("Step-A")
    session.accept_validation(
        validation.token,
        passing_preflight_report(validation.token),
    )
    return session


def _beam_session() -> ModelSession:
    session = ModelSession()
    task = session.prepare_import(
        _FIXTURES / "beam2_rectangle_uniform_load.inp"
    )
    session.accept_imported_model(
        task.token,
        read(_FIXTURES / "beam2_rectangle_uniform_load.inp"),
    )
    validation = session.prepare_validation("UniformLoad")
    session.accept_validation(
        validation.token,
        passing_preflight_report(validation.token),
    )
    return session


def test_import_token_is_stale_after_any_session_transition() -> None:
    session = ModelSession()
    task = session.prepare_import("old.inp")
    session.new_native_project()
    before = session.snapshot()

    delta = session.accept_imported_model(task.token, _model("Step-A"))

    assert not delta.accepted
    assert delta.token_status is TokenStatus.STALE_SESSION
    assert session.snapshot().session_id == before.session_id
    assert session.snapshot().session_revision == before.session_revision


def test_mesh_token_uses_mesh_input_revision() -> None:
    session = ModelSession()
    session.new_native_project()
    session.replace_geometry(
        (NativePart(),), BoxGeometry("Box", 1.0, 1.0, 1.0)
    )
    task = session.prepare_mesh_generation()
    session.replace_geometry(
        (NativePart(),), BoxGeometry("Box", 2.0, 1.0, 1.0)
    )

    delta = session.accept_generated_model(task.token, _model("Step-A"))

    assert not delta.accepted
    assert delta.token_status is TokenStatus.STALE_REVISION
    assert session.snapshot().artifact is None


def test_clear_generated_model_rejects_an_issued_mesh_token() -> None:
    session = ModelSession()
    session.new_native_project()
    session.replace_geometry(
        (NativePart(),), BoxGeometry("Box", 1.0, 1.0, 1.0)
    )
    task = session.prepare_mesh_generation()

    session.clear_generated_model()
    delta = session.accept_generated_model(task.token, _model("Step-A"))

    assert not delta.accepted
    assert delta.token_status is TokenStatus.STALE_REVISION
    assert session.snapshot().artifact is None


def test_token_tampering_reports_artifact_step_and_run_mismatch() -> None:
    session = _session()
    validation = session.prepare_validation("Step-A")
    wrong_artifact = replace(validation.token, artifact_id="other")
    wrong_step = replace(validation.token, step_name="Step-B")

    assert (
        session.validate_task_token(wrong_artifact)
        is TokenStatus.STALE_ARTIFACT
    )
    assert session.validate_task_token(wrong_step) is TokenStatus.STALE_STEP

    solve = session.prepare_solve("Step-A", "Job-1")
    wrong_run = replace(solve.token, run_id="other")
    assert session.validate_task_token(wrong_run) is TokenStatus.STALE_RUN


@pytest.mark.parametrize("terminal", ["failed", "cancelled"])
def test_failed_and_cancelled_callbacks_use_the_same_token_gate(
    terminal: str,
) -> None:
    session = _session()
    solve = session.prepare_solve("Step-A", "Job-1")
    session.begin_run(solve.token)
    session.replace_model_definitions((), (), (), (AnalysisStep("Step-A"),))
    before_revision = session.session_revision

    if terminal == "failed":
        delta = session.accept_run_failed(solve.token, "boom")
    else:
        delta = session.accept_run_cancelled(solve.token)

    assert not delta.accepted
    assert session.session_revision == before_revision
    assert not session.snapshot().runs


def test_repeated_completion_is_explicitly_rejected() -> None:
    session = _session()
    task = session.prepare_validation("Step-A")
    report = passing_preflight_report(task.token)
    first = session.accept_validation(task.token, report)
    revision = session.session_revision
    second = session.accept_validation(task.token, report)

    assert first.accepted
    assert not second.accepted
    assert second.token_status is TokenStatus.ALREADY_COMPLETED
    assert session.session_revision == revision


def test_result_projection_token_becomes_stale_with_its_model() -> None:
    session = _session()
    solve = session.prepare_solve("Step-A", "Job-1")
    session.begin_run(solve.token)
    session.accept_run_result(solve.token, {"value": 1})
    projection = session.prepare_result_projection(solve.run_id)
    assert (
        session.validate_task_token(projection.token)
        is TokenStatus.CURRENT
    )

    session.replace_model_definitions((), (), (), (AnalysisStep("Step-A"),))

    assert session.validate_task_token(projection.token) in {
        TokenStatus.STALE_REVISION,
        TokenStatus.STALE_ARTIFACT,
    }


def test_orientation_edit_rejects_old_validation_and_solve_callbacks() -> None:
    session = _beam_session()
    validation = session.prepare_validation("UniformLoad")
    solve = session.prepare_solve("UniformLoad", "Beam-Job")
    session.begin_run(solve.token)
    before = session.snapshot()
    assignment = before.assignments[0]

    session.replace_model_definitions(
        before.materials,
        before.sections,
        (
            RegionAssignment(
                assignment.section_name,
                assignment.region_name,
                BeamOrientation((0.0, 1.0, 0.0)),
            ),
        ),
        before.steps,
    )
    after_edit = session.snapshot()

    validation_delta = session.accept_validation(
        validation.token,
        passing_preflight_report(validation.token),
    )
    solve_delta = session.accept_run_result(
        solve.token,
        {"U": [1.0]},
    )

    assert after_edit.model_revision == before.model_revision + 1
    assert not validation_delta.accepted
    assert validation_delta.token_status in {
        TokenStatus.STALE_REVISION,
        TokenStatus.STALE_ARTIFACT,
    }
    assert not solve_delta.accepted
    assert solve_delta.token_status in {
        TokenStatus.STALE_REVISION,
        TokenStatus.STALE_ARTIFACT,
        TokenStatus.STALE_RUN,
    }
    assert session.session_revision == after_edit.session_revision
    assert not session.snapshot().validations
    assert not session.snapshot().runs


def test_close_and_load_identity_reject_old_validation_token() -> None:
    session = _session()
    token = session.prepare_validation("Step-A").token
    session.close()

    assert session.validate_task_token(token) is TokenStatus.STALE_SESSION


def test_generic_failure_and_cancellation_consume_current_tokens() -> None:
    failed_session = ModelSession()
    failed_session.new_native_project()
    failed_session.replace_geometry(
        (NativePart(),), BoxGeometry("Box", 1.0, 1.0, 1.0)
    )
    mesh = failed_session.prepare_mesh_generation()
    failed = failed_session.accept_task_failed(mesh.token, "mesher failed")
    repeated = failed_session.accept_task_failed(mesh.token, "again")

    assert failed.accepted
    assert not repeated.accepted
    assert repeated.token_status is TokenStatus.ALREADY_COMPLETED

    cancelled_session = ModelSession()
    imported = cancelled_session.prepare_import("cancelled.inp")
    cancelled = cancelled_session.accept_task_cancelled(imported.token)
    assert cancelled.accepted
    assert (
        cancelled_session.validate_task_token(imported.token)
        is TokenStatus.ALREADY_COMPLETED
    )


def test_generic_validation_failure_keeps_failed_step_record() -> None:
    session = _session()
    validation = session.prepare_validation("Step-A")

    delta = session.accept_task_failed(validation.token, ValueError("bad"))

    assert delta.accepted
    record = session.validation_for("Step-A")
    assert record is not None
    assert not record.passed
    assert isinstance(record.report, PreflightReport)
    assert record.report.errors[0].code == "preflight.internal_error"
    assert record.report.errors[0].message == "bad"
    assert not session.can_submit("Step-A")


def test_result_projection_acceptance_is_single_use() -> None:
    session = _session()
    solve = session.prepare_solve("Step-A", "Job-1")
    session.begin_run(solve.token)
    session.accept_run_result(solve.token, {"value": 1})
    projection = session.prepare_result_projection(solve.run_id)

    accepted = session.accept_result_projection(projection.token)
    repeated = session.accept_result_projection(projection.token)

    assert accepted.accepted
    assert not repeated.accepted
    assert repeated.token_status is TokenStatus.ALREADY_COMPLETED


def test_result_projection_is_stale_after_a_newer_session_transition() -> None:
    session = _session()
    solve = session.prepare_solve("Step-A", "Job-1")
    session.begin_run(solve.token)
    session.accept_run_result(solve.token, {"value": 1})
    projection = session.prepare_result_projection(solve.run_id)

    session.select_result(solve.run_id)
    before_revision = session.session_revision
    delta = session.accept_result_projection(projection.token)

    assert not delta.accepted
    assert delta.token_status is TokenStatus.STALE_REVISION
    assert session.session_revision == before_revision
