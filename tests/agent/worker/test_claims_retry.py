import threading
from datetime import datetime, timedelta, timezone

import pytest

import fem_agent.worker as worker_module
from fem_agent.artifacts import atomic_write_json
from fem_agent.confirmation import ConfirmationRequiredError
from fem_agent.diagnostics import DiagnosticCode, make_diagnostic
from fem_agent.schemas import RunStatus
from fem_agent.worker import (
    IsolatedFEMWorker,
    WorkerResponse,
    WorkerRunInProgressError,
    _persisted_worker_is_active,
)

from tests.helpers.agent_worker_fixtures import (
    _prepared_revision,
    _persist_failure_response,
)


def test_persisted_prelaunch_claim_remains_active_until_its_deadline(tmp_path):
    workspace, artifacts, _, record = _prepared_revision(tmp_path)
    run = artifacts.create_run(
        record.session_id,
        idempotency_key="prelaunch_claim",
    )
    state_path = run.path / "logs" / "worker-process.json"
    future = (
        datetime.now(timezone.utc) + timedelta(seconds=30)
    ).isoformat()
    atomic_write_json(
        state_path,
        {
            "schema_version": 1,
            "session_id": record.session_id,
            "revision": record.revision,
            "revision_hash": record.revision_hash,
            "run_id": run.run_id,
            "supervisor_pid": 2_147_483_647,
            "pid": None,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "deadline_at": future,
        },
    )

    assert _persisted_worker_is_active(state_path, record, run)


def test_claim_unlock_failure_does_not_mask_the_body_result(
    tmp_path,
    monkeypatch,
):
    workspace, artifacts, _, record = _prepared_revision(tmp_path)

    def fail_unlock(_stream):
        raise OSError("unlock failed")

    monkeypatch.setattr(worker_module, "_unlock_claim_stream", fail_unlock)
    for _ in range(2):
        with worker_module._exclusive_worker_claim(
            artifacts,
            record.session_id,
            "unlock_failure",
        ):
            result = "completed"

    assert result == "completed"


def test_worker_rejects_an_unconfirmed_revision(tmp_path):
    workspace, _, _, record = _prepared_revision(tmp_path)
    worker = IsolatedFEMWorker(workspace)

    with pytest.raises(ConfirmationRequiredError):
        worker.run(
            record.session_id,
            revision=record.revision,
            revision_hash=record.revision_hash,
            idempotency_key="unconfirmed_solve",
        )


@pytest.mark.parametrize(
    "diagnostic_code",
    [
        DiagnosticCode.INVALID_MODEL,
        DiagnosticCode.SOLVER_FAILED,
        DiagnosticCode.RESULT_QUERY_FAILED,
        DiagnosticCode.EXPORT_FAILED,
    ],
)
def test_worker_reuses_a_deterministic_failure(
    tmp_path,
    diagnostic_code,
):
    workspace, artifacts, revisions, record = _prepared_revision(tmp_path, confirmed=True)
    run = artifacts.create_run(
        record.session_id,
        idempotency_key="deterministic_failure",
    )
    expected = _persist_failure_response(
        artifacts,
        record,
        run,
        diagnostic_code,
    )

    repeated = IsolatedFEMWorker(
        workspace,
        python_executable=tmp_path / "missing-python",
    ).run(
        record.session_id,
        revision=record.revision,
        revision_hash=record.revision_hash,
        idempotency_key="deterministic_failure",
        timeout_seconds=30,
    )

    assert repeated == expected
    assert len(list(run.path.parent.iterdir())) == 1


def test_worker_retry_budget_exhaustion_is_a_nontransient_terminal_response(
    tmp_path,
    monkeypatch,
):
    workspace, artifacts, revisions, record = _prepared_revision(tmp_path, confirmed=True)
    monkeypatch.setattr(worker_module, "_MAX_WORKER_ATTEMPTS", 2)
    for attempt, code in enumerate(
        (DiagnosticCode.WORKER_TIMEOUT, DiagnosticCode.WORKER_CRASH)
    ):
        key = (
            "exhausted_retries"
            if attempt == 0
            else f"exhausted_retries_retry_{attempt}"
        )
        run = artifacts.create_run(
            record.session_id,
            idempotency_key=key,
        )
        _persist_failure_response(
            artifacts,
            record,
            run,
            code,
            elapsed_seconds=0.25,
        )

    worker = IsolatedFEMWorker(
        workspace,
        python_executable=tmp_path / "must-not-launch",
    )
    exhausted = worker.run(
        record.session_id,
        revision=record.revision,
        revision_hash=record.revision_hash,
        idempotency_key="exhausted_retries",
        timeout_seconds=30,
    )

    assert exhausted.status == RunStatus.FAILED
    assert [item.code for item in exhausted.diagnostics] == [
        "WORKER_RETRY_EXHAUSTED"
    ]
    assert exhausted.diagnostics[0].entity == "worker-retry-budget"
    assert exhausted.elapsed_seconds == pytest.approx(0.5)
    assert not worker_module._is_transient_worker_failure(exhausted)
    exhausted_run = artifacts.run_directory(
        record.session_id,
        exhausted.run_id,
    )
    assert (exhausted_run.path / "manifest.json").is_file()

    repeated = worker.run(
        record.session_id,
        revision=record.revision,
        revision_hash=record.revision_hash,
        idempotency_key="exhausted_retries",
        timeout_seconds=30,
    )

    assert repeated == exhausted
    (exhausted_run.path / "logs" / "worker-response.json").unlink()
    recovered = worker.run(
        record.session_id,
        revision=record.revision,
        revision_hash=record.revision_hash,
        idempotency_key="exhausted_retries",
        timeout_seconds=30,
    )

    assert recovered == exhausted
    assert len(
        list(
            (
                artifacts.session_path(record.session_id)
                / "runs"
            ).iterdir()
        )
    ) == 3


def test_cross_instance_claim_prevents_a_duplicate_worker_launch(
    tmp_path,
    monkeypatch,
):
    workspace, artifacts, revisions, record = _prepared_revision(tmp_path, confirmed=True)
    first = IsolatedFEMWorker(workspace)
    second = IsolatedFEMWorker(workspace)
    entered_launch = threading.Event()
    release_launch = threading.Event()
    first_result = []
    first_errors = []

    def held_launch(
        request,
        _record,
        run,
        _request_path,
        _response_path,
        _timeout,
        _cancel_event,
    ):
        entered_launch.set()
        if not release_launch.wait(timeout=2):
            raise AssertionError("test did not release the held worker launch")
        return WorkerResponse(
            session_id=request.session_id,
            revision=request.revision,
            revision_hash=request.revision_hash,
            run_id=run.run_id,
            status=RunStatus.FAILED,
            result_summary=None,
            artifacts=(),
            diagnostics=(
                make_diagnostic(
                    DiagnosticCode.SOLVER_FAILED,
                    "The deterministic worker stage failed.",
                    source="test.worker",
                ),
            ),
            elapsed_seconds=0.01,
        )

    monkeypatch.setattr(first, "_launch", held_launch)

    def run_first():
        try:
            first_result.append(
                first.run(
                    record.session_id,
                    revision=record.revision,
                    revision_hash=record.revision_hash,
                    idempotency_key="claimed_solve",
                    timeout_seconds=30,
                )
            )
        except Exception as error:
            first_errors.append(error)

    thread = threading.Thread(target=run_first)
    thread.start()
    assert entered_launch.wait(timeout=2)
    try:
        with pytest.raises(WorkerRunInProgressError):
            second.run(
                record.session_id,
                revision=record.revision,
                revision_hash=record.revision_hash,
                idempotency_key="claimed_solve",
                timeout_seconds=30,
            )
    finally:
        release_launch.set()
        thread.join(timeout=2)

    assert not thread.is_alive()
    assert not first_errors
    assert len(first_result) == 1
    assert len(list((artifacts.session_path(record.session_id) / "runs").iterdir())) == 1
