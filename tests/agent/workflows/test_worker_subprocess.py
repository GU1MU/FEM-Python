import shutil
import threading

import numpy as np
import pytest

import fem_agent.worker as worker_module
from fem_agent.artifacts import atomic_write_json, read_json_file
from fem_agent.confirmation import ConfirmationStore
from fem_agent.diagnostics import DiagnosticCode, make_diagnostic
from fem_agent.schemas import ResultQuery, ResultQueryKind, RunStatus, UnitContext
from fem_agent.worker import (
    IsolatedFEMInspector,
    IsolatedFEMResultQuerier,
    IsolatedFEMWorker,
    ResultQueryRequest,
    ResultQueryWorkerError,
    WorkerResponse,
    WorkerResponseIntegrityError,
    execute_result_query_request,
)

from tests.helpers.agent_worker_fixtures import (
    _prepared_revision,
    _mark_persisted_worker_inactive,
)


@pytest.fixture(scope="module")
def solved_worker_workspace(tmp_path_factory):
    """One real child solve supplies independent corruption-test copies."""
    directory = tmp_path_factory.mktemp("solved-worker-baseline")
    workspace, _artifacts, _revisions, record = _prepared_revision(
        directory, confirmed=True,
    )
    response = IsolatedFEMWorker(workspace).run(
        record.session_id,
        revision=record.revision,
        revision_hash=record.revision_hash,
        idempotency_key="persisted-result",
        timeout_seconds=30,
    )
    assert response.status == RunStatus.SUCCEEDED
    return workspace, record, response


@pytest.fixture
def copied_worker_run(tmp_path, solved_worker_workspace):
    source, record, response = solved_worker_workspace
    workspace = tmp_path / "workspace"
    shutil.copytree(source, workspace)
    worker = IsolatedFEMWorker(workspace)
    return worker, worker.artifacts, record, response


def test_isolated_inspector_round_trips_unicode_unit_context(tmp_path):
    units = UnitContext(
        length="mm",
        force="N",
        stress="MPa",
        density="tonne/mm³",
        acceleration="mm/s²",
        convention="毫米-牛顿单位制",
    )
    workspace, _artifacts, _revisions, record = _prepared_revision(
        tmp_path,
        unit_context=units,
    )

    response = IsolatedFEMInspector(workspace).inspect(
        record.spec,
        record.revision_hash,
    )

    assert response.summary.unit_context == units
    assert not response.summary.has_blocking_diagnostics


def test_isolated_worker_solves_queries_exports_and_writes_manifest(tmp_path):
    workspace, artifacts, revisions, record = _prepared_revision(tmp_path)
    ConfirmationStore(workspace, revisions).confirm(
        record.session_id,
        revision=record.revision,
        revision_hash=record.revision_hash,
    )
    worker = IsolatedFEMWorker(workspace)

    response = worker.run(
        record.session_id,
        revision=record.revision,
        revision_hash=record.revision_hash,
        idempotency_key="confirmed_solve",
        timeout_seconds=30,
    )

    assert isinstance(response, WorkerResponse)
    assert response.status == RunStatus.SUCCEEDED
    assert response.result_summary is not None
    assert response.result_summary.finite_vectors is True
    assert len(response.result_summary.scalars) == 2
    kinds = {artifact.kind for artifact in response.artifacts}
    assert {
        "csv",
        "result_summary",
        "solution",
        "diagnostics",
        "manifest",
    } <= kinds
    run = artifacts.run_directory(record.session_id, response.run_id)
    assert (run.path / "manifest.json").is_file()
    assert (run.path / "diagnostics.json").is_file()
    assert (run.path / "result-summary.json").is_file()
    solution = np.load(run.path / "solution.npy", allow_pickle=False)
    assert solution.shape == (2, 12)
    assert np.all(np.isfinite(solution))
    manifest = read_json_file(run.path / "manifest.json")
    solution_model_sha256 = manifest["tool_parameters"][
        "solution_model_sha256"
    ]
    assert len(solution_model_sha256) == 64
    assert all(
        character in "0123456789abcdef"
        for character in solution_model_sha256
    )

    repeated = worker.run(
        record.session_id,
        revision=record.revision,
        revision_hash=record.revision_hash,
        idempotency_key="confirmed_solve",
        timeout_seconds=30,
    )
    assert repeated.run_id == response.run_id
    assert len(list((run.path.parent).iterdir())) == 1


def test_worker_solves_without_queries_and_supports_later_postprocessing(
    tmp_path,
    monkeypatch,
):
    workspace, artifacts, revisions, record = _prepared_revision(
        tmp_path,
        requested_queries=(),
        export_formats=(),
    )
    ConfirmationStore(workspace, revisions).confirm(
        record.session_id,
        revision=record.revision,
        revision_hash=record.revision_hash,
    )
    response = IsolatedFEMWorker(workspace).run(
        record.session_id,
        revision=record.revision,
        revision_hash=record.revision_hash,
        idempotency_key="solve_before_query",
        timeout_seconds=30,
    )

    assert response.status == RunStatus.SUCCEEDED
    assert response.result_summary is None
    assert any(item.kind == "solution" for item in response.artifacts)

    def unexpected_second_solve(*args, **kwargs):
        raise AssertionError("post-solve queries must not repeat the solve")

    monkeypatch.setattr(
        worker_module,
        "solve_analysis",
        unexpected_second_solve,
    )
    summary = execute_result_query_request(
        workspace,
        ResultQueryRequest(
            session_id=record.session_id,
            revision=record.revision,
            revision_hash=record.revision_hash,
            run_id=response.run_id,
            queries=(
                ResultQuery(
                    ResultQueryKind.MAX_DISPLACEMENT_MAGNITUDE,
                    edge="Surf-right",
                ),
                ResultQuery(
                    ResultQueryKind.STRESS_EXTREMA,
                    element_set="SOLID",
                    measure="von_mises",
                ),
            ),
        ),
    )

    assert summary.diagnostics == ()
    assert summary.scalars[0].region == "Surf-right"
    assert summary.scalars[0].unit == "mm"
    stress_scalars = tuple(
        item
        for item in summary.scalars
        if item.query_kind == ResultQueryKind.STRESS_EXTREMA
    )
    assert len(stress_scalars) == 2
    assert all(item.region == "SOLID" for item in stress_scalars)
    assert all(item.unit == "MPa" for item in stress_scalars)

    original_inspector = worker_module.inspect_abaqus

    def inspect_with_different_dof_order(*args, **kwargs):
        imported = original_inspector(*args, **kwargs)
        imported.model.mesh.nodes.reverse()
        imported.model.mesh.rebuild_dof_map()
        return imported

    monkeypatch.setattr(
        worker_module,
        "inspect_abaqus",
        inspect_with_different_dof_order,
    )
    with pytest.raises(ValueError, match="solution fingerprint"):
        execute_result_query_request(
            workspace,
            ResultQueryRequest(
                session_id=record.session_id,
                revision=record.revision,
                revision_hash=record.revision_hash,
                run_id=response.run_id,
                queries=(
                    ResultQuery(
                        ResultQueryKind.MAX_DISPLACEMENT_MAGNITUDE,
                        edge="Surf-right",
                    ),
                ),
            ),
        )


def test_isolated_postsolve_query_rejects_a_tampered_solution(copied_worker_run):
    worker, artifacts, record, response = copied_worker_run
    workspace = worker.artifacts.root
    solution = next(
        item for item in response.artifacts if item.kind == "solution"
    )
    path = artifacts.resolve_artifact(
        record.session_id,
        solution.artifact_id,
    )
    path.write_bytes(b"tampered")

    with pytest.raises(ResultQueryWorkerError):
        IsolatedFEMResultQuerier(workspace).query(
            response,
            (
                ResultQuery(
                    ResultQueryKind.MAX_DISPLACEMENT_MAGNITUDE,
                    edge="Surf-right",
                ),
            ),
            timeout_seconds=30,
        )

    assert response.status == RunStatus.SUCCEEDED


def test_worker_recovers_a_missing_response_from_a_verified_manifest(copied_worker_run):
    worker, artifacts, record, original = copied_worker_run
    run = artifacts.run_directory(record.session_id, original.run_id)
    response_path = run.path / "logs" / "worker-response.json"
    response_path.unlink()
    _mark_persisted_worker_inactive(run)
    original_manifest_artifact = next(
        item for item in original.artifacts if item.kind == "manifest"
    )
    manifest_metadata = (
        artifacts.session_path(record.session_id)
        / "artifacts"
        / f"{original_manifest_artifact.artifact_id}.json"
    )
    manifest_metadata.unlink()

    recovered = worker.run(
        record.session_id,
        revision=record.revision,
        revision_hash=record.revision_hash,
        idempotency_key="persisted-result",
        timeout_seconds=30,
    )

    assert recovered.run_id == original.run_id
    assert recovered.status == original.status
    assert recovered.result_summary == original.result_summary
    assert recovered.diagnostics == original.diagnostics
    assert recovered.elapsed_seconds == original.elapsed_seconds
    recovered_manifest_artifact = next(
        item for item in recovered.artifacts if item.kind == "manifest"
    )
    assert (
        recovered_manifest_artifact.artifact_id
        != original_manifest_artifact.artifact_id
    )
    assert response_path.is_file()
    assert len(list(run.path.parent.iterdir())) == 1


def test_worker_fails_closed_when_a_commit_manifest_is_corrupt(copied_worker_run):
    worker, artifacts, record, original = copied_worker_run
    run = artifacts.run_directory(record.session_id, original.run_id)
    response_path = run.path / "logs" / "worker-response.json"
    response_path.unlink()
    _mark_persisted_worker_inactive(run)
    manifest_path = run.path / "manifest.json"
    manifest = read_json_file(manifest_path)
    manifest["source_sha256"] = "0" * 64
    atomic_write_json(manifest_path, manifest, overwrite=True)

    with pytest.raises(WorkerResponseIntegrityError):
        worker.run(
            record.session_id,
            revision=record.revision,
            revision_hash=record.revision_hash,
            idempotency_key="persisted-result",
            timeout_seconds=30,
        )

    assert not response_path.exists()
    assert len(list(run.path.parent.iterdir())) == 1


def test_worker_fails_closed_when_a_manifest_artifact_hash_is_wrong(copied_worker_run):
    worker, artifacts, record, original = copied_worker_run
    run = artifacts.run_directory(record.session_id, original.run_id)
    response_path = run.path / "logs" / "worker-response.json"
    response_path.unlink()
    _mark_persisted_worker_inactive(run)
    diagnostics_path = run.path / "diagnostics.json"
    diagnostics = read_json_file(diagnostics_path)
    diagnostics["diagnostics"].append(
        make_diagnostic(
            DiagnosticCode.WORKER_CRASH,
            "Tampered diagnostic.",
            source="test.worker",
        ).to_dict()
    )
    atomic_write_json(diagnostics_path, diagnostics, overwrite=True)

    with pytest.raises(WorkerResponseIntegrityError):
        worker.run(
            record.session_id,
            revision=record.revision,
            revision_hash=record.revision_hash,
            idempotency_key="persisted-result",
            timeout_seconds=30,
        )

    assert not response_path.exists()
    assert len(list(run.path.parent.iterdir())) == 1


def test_worker_timeout_returns_a_failed_response_and_manifest(tmp_path):
    workspace, artifacts, revisions, record = _prepared_revision(tmp_path)
    ConfirmationStore(workspace, revisions).confirm(
        record.session_id,
        revision=record.revision,
        revision_hash=record.revision_hash,
    )

    worker = IsolatedFEMWorker(workspace)
    response = worker.run(
        record.session_id,
        revision=record.revision,
        revision_hash=record.revision_hash,
        idempotency_key="timed_out_solve",
        timeout_seconds=0.001,
    )

    assert response.status == RunStatus.FAILED
    assert response.diagnostics[0].code == "WORKER_TIMEOUT"
    run = artifacts.run_directory(record.session_id, response.run_id)
    assert (run.path / "manifest.json").is_file()


def test_worker_cancellation_returns_a_cancelled_response_and_manifest(tmp_path):
    workspace, artifacts, revisions, record = _prepared_revision(tmp_path)
    ConfirmationStore(workspace, revisions).confirm(
        record.session_id,
        revision=record.revision,
        revision_hash=record.revision_hash,
    )
    cancelled = threading.Event()
    cancelled.set()

    worker = IsolatedFEMWorker(workspace)
    response = worker.run(
        record.session_id,
        revision=record.revision,
        revision_hash=record.revision_hash,
        idempotency_key="cancelled_solve",
        timeout_seconds=30,
        cancel_event=cancelled,
    )

    assert response.status == RunStatus.CANCELLED
    assert response.diagnostics[0].code == "OPERATION_CANCELLED"
    run = artifacts.run_directory(record.session_id, response.run_id)
    assert (run.path / "manifest.json").is_file()

    retried = worker.run(
        record.session_id,
        revision=record.revision,
        revision_hash=record.revision_hash,
        idempotency_key="cancelled_solve",
        timeout_seconds=30,
    )

    assert retried.status == RunStatus.SUCCEEDED
    assert retried.run_id != response.run_id
    assert len(list(run.path.parent.iterdir())) == 2


def test_worker_crash_is_normalized_without_terminating_the_parent(tmp_path):
    workspace, artifacts, revisions, record = _prepared_revision(tmp_path)
    ConfirmationStore(workspace, revisions).confirm(
        record.session_id,
        revision=record.revision,
        revision_hash=record.revision_hash,
    )
    crash_executable = shutil.which("false") or shutil.which("where")
    if crash_executable is None:
        pytest.skip(
            "[platform-capability] no harmless always-failing executable "
            "is available"
        )

    response = IsolatedFEMWorker(
        workspace,
        python_executable=crash_executable,
    ).run(
        record.session_id,
        revision=record.revision,
        revision_hash=record.revision_hash,
        idempotency_key="crashed_solve",
        timeout_seconds=30,
    )

    assert response.status == RunStatus.FAILED
    assert response.diagnostics[0].code == "WORKER_CRASH"
    run = artifacts.run_directory(record.session_id, response.run_id)
    assert (run.path / "manifest.json").is_file()
