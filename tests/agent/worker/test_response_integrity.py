import shutil
from datetime import datetime, timedelta, timezone

import pytest

import fem_agent.worker as worker_module
from fem_agent.artifacts import atomic_write_json, read_json_file
from fem_agent.confirmation import ConfirmationStore
from fem_agent.diagnostics import DiagnosticCode
from fem_agent.worker import (
    IsolatedFEMWorker,
    WorkerRequest,
    WorkerResponseIntegrityError,
)

from tests.helpers.agent_worker_fixtures import (
    _prepared_revision,
    _persist_failure_response,
)


def test_worker_skips_a_damaged_older_response_for_a_valid_later_attempt(
    tmp_path,
):
    workspace, artifacts, revisions, record = _prepared_revision(tmp_path)
    ConfirmationStore(workspace, revisions).confirm(
        record.session_id,
        revision=record.revision,
        revision_hash=record.revision_hash,
    )
    damaged = artifacts.create_run(
        record.session_id,
        idempotency_key="damaged_response",
    )
    atomic_write_json(
        damaged.path / "logs" / "worker-response.json",
        {"schema_version": 999},
    )
    recovered = artifacts.create_run(
        record.session_id,
        idempotency_key="damaged_response_retry_1",
    )
    expected = _persist_failure_response(
        artifacts,
        record,
        recovered,
        DiagnosticCode.SOLVER_FAILED,
    )

    response = IsolatedFEMWorker(workspace).run(
        record.session_id,
        revision=record.revision,
        revision_hash=record.revision_hash,
        idempotency_key="damaged_response",
        timeout_seconds=30,
    )

    assert response == expected


def test_worker_does_not_mask_the_latest_damaged_response_with_a_retry(
    tmp_path,
):
    workspace, artifacts, revisions, record = _prepared_revision(tmp_path)
    ConfirmationStore(workspace, revisions).confirm(
        record.session_id,
        revision=record.revision,
        revision_hash=record.revision_hash,
    )
    damaged = artifacts.create_run(
        record.session_id,
        idempotency_key="latest_damaged_response",
    )
    atomic_write_json(
        damaged.path / "logs" / "worker-response.json",
        {"schema_version": 999},
    )

    with pytest.raises(WorkerResponseIntegrityError):
        IsolatedFEMWorker(workspace).run(
            record.session_id,
            revision=record.revision,
            revision_hash=record.revision_hash,
            idempotency_key="latest_damaged_response",
            timeout_seconds=30,
        )

    assert len(list(damaged.path.parent.iterdir())) == 1


def test_verified_response_loader_rejects_response_tampering(
    tmp_path,
):
    workspace, artifacts, revisions, record = _prepared_revision(tmp_path)
    ConfirmationStore(workspace, revisions).confirm(
        record.session_id,
        revision=record.revision,
        revision_hash=record.revision_hash,
    )
    run = artifacts.create_run(
        record.session_id,
        idempotency_key="tampered_verified_response",
    )
    _persist_failure_response(
        artifacts,
        record,
        run,
        DiagnosticCode.SOLVER_FAILED,
    )
    response_path = run.path / "logs" / "worker-response.json"
    payload = read_json_file(response_path)
    payload["elapsed_seconds"] += 1
    atomic_write_json(response_path, payload, overwrite=True)

    with pytest.raises(WorkerResponseIntegrityError):
        worker_module.load_verified_worker_response(
            artifacts,
            record,
            run.run_id,
        )
    with pytest.raises(WorkerResponseIntegrityError):
        IsolatedFEMWorker(workspace).run(
            record.session_id,
            revision=record.revision,
            revision_hash=record.revision_hash,
            idempotency_key="tampered_verified_response",
            timeout_seconds=30,
        )

    assert len(list(run.path.parent.iterdir())) == 1


def test_verified_response_loader_rejects_committed_artifact_tampering(
    tmp_path,
):
    _, artifacts, _, record = _prepared_revision(tmp_path)
    run = artifacts.create_run(
        record.session_id,
        idempotency_key="tampered_verified_artifact",
    )
    _persist_failure_response(
        artifacts,
        record,
        run,
        DiagnosticCode.SOLVER_FAILED,
    )
    diagnostics_path = run.path / "diagnostics.json"
    payload = read_json_file(diagnostics_path)
    payload["diagnostics"][0]["message"] = "Changed after commit."
    atomic_write_json(diagnostics_path, payload, overwrite=True)

    with pytest.raises(WorkerResponseIntegrityError):
        worker_module.load_verified_worker_response(
            artifacts,
            record,
            run.run_id,
        )


def test_verified_response_loader_rejects_a_response_copied_between_runs(
    tmp_path,
):
    workspace, artifacts, revisions, record = _prepared_revision(tmp_path)
    ConfirmationStore(workspace, revisions).confirm(
        record.session_id,
        revision=record.revision,
        revision_hash=record.revision_hash,
    )
    source = artifacts.create_run(
        record.session_id,
        idempotency_key="source_verified_response",
    )
    _persist_failure_response(
        artifacts,
        record,
        source,
        DiagnosticCode.SOLVER_FAILED,
    )
    copied = artifacts.create_run(
        record.session_id,
        idempotency_key="copied_verified_response",
    )
    shutil.copyfile(
        source.path / "logs" / "worker-response.json",
        copied.path / "logs" / "worker-response.json",
    )

    with pytest.raises(WorkerResponseIntegrityError):
        worker_module.load_verified_worker_response(
            artifacts,
            record,
            copied.run_id,
        )
    with pytest.raises(WorkerResponseIntegrityError):
        IsolatedFEMWorker(workspace).run(
            record.session_id,
            revision=record.revision,
            revision_hash=record.revision_hash,
            idempotency_key="copied_verified_response",
            timeout_seconds=30,
        )

    assert len(list(copied.path.parent.iterdir())) == 2


def test_orphaned_request_cannot_be_rebound_to_another_revision(tmp_path):
    workspace, artifacts, revisions, record = _prepared_revision(tmp_path)
    ConfirmationStore(workspace, revisions).confirm(
        record.session_id,
        revision=record.revision,
        revision_hash=record.revision_hash,
    )
    run = artifacts.create_run(
        record.session_id,
        idempotency_key="bound_request",
    )
    atomic_write_json(
        run.path / "logs" / "worker-request.json",
        WorkerRequest(
            session_id=record.session_id,
            revision=record.revision,
            revision_hash="b" * 64,
            run_id=run.run_id,
            idempotency_key="bound_request",
            deadline_at=(
                datetime.now(timezone.utc) + timedelta(seconds=30)
            ).isoformat(),
        ).to_dict(),
    )

    with pytest.raises(WorkerResponseIntegrityError):
        IsolatedFEMWorker(workspace).run(
            record.session_id,
            revision=record.revision,
            revision_hash=record.revision_hash,
            idempotency_key="bound_request",
            timeout_seconds=30,
        )

    assert len(list(run.path.parent.iterdir())) == 1
