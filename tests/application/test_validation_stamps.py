from __future__ import annotations

from types import SimpleNamespace

from fem.application import ModelSession, NativePart, TokenStatus
from fem.core.model import AnalysisStep


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
    session.accept_validation(step_a.token, {"passed": True})

    assert session.validation_for("Step-A").passed
    assert session.validation_for("Step-B") is None
    assert session.can_submit("Step-A")
    assert not session.can_submit("Step-B")

    step_b = session.prepare_validation("Step-B")
    session.accept_validation(step_b.token, {"passed": True})

    assert session.can_submit("Step-A")
    assert session.can_submit("Step-B")


def test_failed_report_is_retained_but_does_not_create_valid_stamp() -> None:
    session = _session()
    task = session.prepare_validation("Step-A")
    delta = session.accept_validation(
        task.token, {"passed": False, "issues": ["missing boundary"]}
    )

    record = session.validation_for("Step-A")
    assert delta.accepted
    assert record is not None
    assert not record.passed
    assert record.report["issues"] == ["missing boundary"]
    assert not session.can_submit("Step-A")


def test_model_revision_change_removes_old_stamps() -> None:
    session = _session()
    task = session.prepare_validation("Step-A")
    session.accept_validation(task.token, {"passed": True})
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
        old_task.token, {"passed": True}
    )

    assert not delta.accepted
    assert delta.token_status in {
        TokenStatus.STALE_REVISION,
        TokenStatus.STALE_ARTIFACT,
    }
    assert session.session_revision == before_revision
    assert session.validation_for("Step-A") is None

