from __future__ import annotations

from types import SimpleNamespace

from fem.application import ModelSession, NativePart, RunStatus
from fem.core.model import AnalysisStep
from fem.geometry.recipes import BoxGeometry
from tests.helpers.preflight_builders import passing_preflight_report


def _model() -> SimpleNamespace:
    return SimpleNamespace(
        materials={},
        sections=[],
        steps=[AnalysisStep("Step-A")],
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
    session.accept_generated_model(mesh.token, _model())
    validation = session.prepare_validation("Step-A")
    session.accept_validation(
        validation.token,
        passing_preflight_report(validation.token),
    )
    return session


def test_pending_running_succeeded_lifecycle_and_provenance() -> None:
    session = _session()
    solve = session.prepare_solve("Step-A", "Job-1")
    pending = session.find_run(solve.run_id)
    assert pending.status is RunStatus.PENDING

    session.begin_run(solve.token)
    assert session.find_run(solve.run_id).status is RunStatus.RUNNING

    session.accept_run_result(
        solve.token,
        {"value": 42},
        timings={"solve": 0.25},
    )
    succeeded = session.find_run(solve.run_id)
    current = session.current_result()
    snapshot = session.snapshot()

    assert succeeded.status is RunStatus.SUCCEEDED
    assert succeeded.has_result
    assert succeeded.timings == {"solve": 0.25}
    assert current.result == {"value": 42}
    assert current.provenance.session_id == snapshot.session_id
    assert current.provenance.artifact_id == snapshot.artifact.artifact_id
    assert current.provenance.model_revision == snapshot.model_revision
    assert current.provenance.step_name == "Step-A"
    assert current.provenance.run_id == solve.run_id
    assert snapshot.displayed_result_run_id == solve.run_id


def test_failed_new_run_preserves_previous_successful_display() -> None:
    session = _session()
    first = session.prepare_solve("Step-A", "Job-1")
    session.begin_run(first.token)
    session.accept_run_result(first.token, {"value": 1})

    second = session.prepare_solve("Step-A", "Job-2")
    session.begin_run(second.token)
    assert session.current_result().result == {"value": 1}
    session.accept_run_failed(second.token, "solver failed")

    assert session.find_run(second.run_id).status is RunStatus.FAILED
    assert session.current_result().result == {"value": 1}
    assert session.snapshot().displayed_result_run_id == first.run_id


def test_cancelled_new_run_preserves_previous_successful_display() -> None:
    session = _session()
    first = session.prepare_solve("Step-A", "Job-1")
    session.begin_run(first.token)
    session.accept_run_result(first.token, {"value": 1})

    second = session.prepare_solve("Step-A", "Job-2")
    session.begin_run(second.token)
    session.request_cancel(second.run_id)
    session.accept_run_cancelled(second.token)

    cancelled = session.find_run(second.run_id)
    assert cancelled.status is RunStatus.CANCELLED
    assert cancelled.cancellation_requested
    assert session.current_result().result == {"value": 1}
    assert session.snapshot().displayed_result_run_id == first.run_id


def test_model_revision_change_clears_run_history_and_display() -> None:
    session = _session()
    solve = session.prepare_solve("Step-A", "Job-1")
    session.begin_run(solve.token)
    session.accept_run_result(solve.token, {"value": 1})

    session.replace_model_definitions((), (), (), (AnalysisStep("Step-A"),))
    snapshot = session.snapshot()

    assert snapshot.runs == ()
    assert snapshot.displayed_result_run_id is None
    assert snapshot.displayed_result is None
    assert session.current_result() is None
