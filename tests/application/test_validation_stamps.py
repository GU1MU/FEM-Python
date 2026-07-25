from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from fem.application import ModelSession, NativePart, TokenStatus
from fem.core.model import AnalysisStep
from tests.helpers.preflight_builders import (
    failing_preflight_report,
    passing_preflight_report,
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
    session.replace_geometry((NativePart(),), {"kind": "box"})
    session.replace_model_definitions(
        (), (), (), (AnalysisStep("Step-A"), AnalysisStep("Step-B"))
    )
    mesh = session.prepare_mesh_generation()
    session.accept_generated_model(mesh.token, _model("Step-A", "Step-B"))
    return session


def test_validation_is_independent_for_each_step() -> None:
    session = _session()
    step_a = session.prepare_validation("Step-A")
    session.accept_validation(
        step_a.token,
        passing_preflight_report(step_a.token),
    )

    assert session.validation_for("Step-A").passed
    assert session.validation_for("Step-B") is None
    assert session.can_submit("Step-A")
    assert not session.can_submit("Step-B")

    step_b = session.prepare_validation("Step-B")
    session.accept_validation(
        step_b.token,
        passing_preflight_report(step_b.token),
    )

    assert session.can_submit("Step-A")
    assert session.can_submit("Step-B")


def test_failed_report_is_retained_but_does_not_create_valid_stamp() -> None:
    session = _session()
    task = session.prepare_validation("Step-A")
    delta = session.accept_validation(
        task.token,
        failing_preflight_report(
            task.token,
            code="preflight.boundary.missing",
            message="missing boundary",
        ),
    )

    record = session.validation_for("Step-A")
    assert delta.accepted
    assert record is not None
    assert not record.passed
    assert record.report.errors[0].code == "preflight.boundary.missing"
    assert record.report.errors[0].message == "missing boundary"
    assert not session.can_submit("Step-A")


def test_model_revision_change_removes_old_stamps() -> None:
    session = _session()
    task = session.prepare_validation("Step-A")
    session.accept_validation(
        task.token,
        passing_preflight_report(task.token),
    )
    previous_revision = session.model_revision

    session.replace_model_definitions(
        (), (), (), (AnalysisStep("Step-A"), AnalysisStep("Step-B"))
    )

    assert session.model_revision == previous_revision + 1
    assert session.validation_for("Step-A") is None
    assert not session.can_submit("Step-A")


def test_old_validation_callback_is_rejected_for_new_model_revision() -> None:
    session = _session()
    old_task = session.prepare_validation("Step-A")
    session.replace_model_definitions(
        (), (), (), (AnalysisStep("Step-A"), AnalysisStep("Step-B"))
    )
    before_revision = session.session_revision

    delta = session.accept_validation(
        old_task.token,
        passing_preflight_report(old_task.token),
    )

    assert not delta.accepted
    assert delta.token_status in {
        TokenStatus.STALE_REVISION,
        TokenStatus.STALE_ARTIFACT,
    }
    assert session.session_revision == before_revision
    assert session.validation_for("Step-A") is None


@pytest.mark.parametrize(
    "report",
    [
        {"passed": True},
        True,
        ValueError("worker failed"),
        SimpleNamespace(passed=True),
    ],
    ids=["dict", "bool", "exception", "duck-object"],
)
def test_validation_rejects_untyped_reports(report: object) -> None:
    session = _session()
    task = session.prepare_validation("Step-A")

    with pytest.raises(TypeError, match="PreflightReport"):
        session.accept_validation(task.token, report)

    assert session.validation_for("Step-A") is None
    assert session.validate_task_token(task.token) is TokenStatus.CURRENT


@pytest.mark.parametrize(
    "changes",
    [
        {"step_name": "Step-B"},
        {"session_id": "other-session"},
        {"artifact_id": "other-artifact"},
        {"model_revision": -1},
    ],
    ids=["step", "session", "artifact", "model-revision"],
)
def test_validation_rejects_report_provenance_mismatch(
    changes: dict[str, object],
) -> None:
    session = _session()
    task = session.prepare_validation("Step-A")
    report = replace(passing_preflight_report(task.token), **changes)

    with pytest.raises(ValueError, match="provenance"):
        session.accept_validation(task.token, report)

    assert session.validation_for("Step-A") is None
    assert session.validate_task_token(task.token) is TokenStatus.CURRENT


def test_failed_validation_callback_requires_error_diagnostic() -> None:
    session = _session()
    task = session.prepare_validation("Step-A")

    with pytest.raises(ValueError, match="error diagnostic"):
        session.accept_validation_failed(
            task.token,
            passing_preflight_report(task.token),
        )

    assert session.validation_for("Step-A") is None
    assert session.validate_task_token(task.token) is TokenStatus.CURRENT


def test_cancelled_validation_leaves_no_record() -> None:
    session = _session()
    task = session.prepare_validation("Step-A")

    delta = session.accept_task_cancelled(task.token)

    assert delta.accepted
    assert session.validation_for("Step-A") is None
    assert not session.can_submit("Step-A")
    assert (
        session.validate_task_token(task.token)
        is TokenStatus.ALREADY_COMPLETED
    )

