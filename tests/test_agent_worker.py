import io
import json
import shutil
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

import fem_agent.worker as worker_module
from fem_agent.artifacts import ArtifactStore, atomic_write_json, read_json_file
from fem_agent.confirmation import ConfirmationRequiredError, ConfirmationStore
from fem_agent.diagnostics import DiagnosticCode, make_diagnostic
from fem_agent.schemas import (
    ExportFormat,
    ImportAnalysisSpec,
    ResourceLimits,
    ResultQuery,
    ResultQueryKind,
    RunStatus,
    UnitContext,
)
from fem_agent.state import RevisionStore
from fem_agent.worker import (
    InspectionWorkerError,
    IsolatedFEMInspector,
    IsolatedFEMResultQuerier,
    IsolatedFEMWorker,
    ResultQueryRequest,
    ResultQueryWorkerError,
    WorkerRequest,
    WorkerResponse,
    WorkerResponseIntegrityError,
    WorkerRunInProgressError,
    execute_result_query_request,
    _persisted_worker_is_active,
    _start_worker_deadline_watchdog,
    _terminate_process,
    scrub_worker_environment,
)
from tests.helpers.abaqus_builders import write_perforated_plate_style_inp


def _units():
    return UnitContext(
        length="mm",
        force="N",
        stress="MPa",
        density="tonne/mm^3",
        acceleration="mm/s^2",
    )


def _prepared_revision(
    tmp_path,
    *,
    unit_context=None,
    resource_limits=None,
    requested_queries=None,
    export_formats=None,
):
    source = write_perforated_plate_style_inp(
        tmp_path,
        "worker_model.inp",
        ("*Boundary", "Set-right, 1, 1, 0.05"),
    )
    workspace = tmp_path / "workspace"
    artifacts = ArtifactStore(workspace)
    session_id = artifacts.create_session("ses_worker")
    artifact = artifacts.copy_input(session_id, source)
    revisions = RevisionStore(workspace)
    revisions.create_session(session_id)
    record = revisions.initialize(
        ImportAnalysisSpec(
            session_id=session_id,
            revision=1,
            source_artifact_id=artifact.artifact_id,
            source_sha256=artifact.sha256,
            unit_context=unit_context or _units(),
            analysis_step="Step-1",
            requested_queries=(
                requested_queries
                if requested_queries is not None
                else (
                    ResultQuery(
                        ResultQueryKind.MAX_DISPLACEMENT_MAGNITUDE
                    ),
                    ResultQuery(
                        ResultQueryKind.REACTION_SUM,
                        component=1,
                        node_set="Set-left",
                    ),
                )
            ),
            export_formats=(
                export_formats
                if export_formats is not None
                else (ExportFormat.CSV,)
            ),
            resource_limits=resource_limits or ResourceLimits(),
        ),
        idempotency_key="initialize_worker",
    )
    return workspace, artifacts, revisions, record


def _mark_persisted_worker_inactive(run):
    state_path = run.path / "logs" / "worker-process.json"
    state = read_json_file(state_path)
    state["supervisor_pid"] = 2_147_483_647
    atomic_write_json(state_path, state, overwrite=True)


def _persist_failure_response(
    artifacts,
    record,
    run,
    diagnostic_code,
    *,
    elapsed_seconds=0.01,
):
    response_path = run.path / "logs" / "worker-response.json"
    diagnostic = make_diagnostic(
        diagnostic_code,
        "The persisted worker stage failed.",
        source="test.worker",
    )
    return worker_module._write_terminal_failure(
        artifacts,
        record,
        run,
        response_path,
        RunStatus.FAILED,
        diagnostic,
        elapsed_seconds=elapsed_seconds,
    )


class _InspectionProcess:
    def __init__(self, stdout: bytes = b"", stderr: bytes = b""):
        self.stdin = io.BytesIO()
        self.stdout = io.BytesIO(stdout)
        self.stderr = io.BytesIO(stderr)
        self.returncode = None
        self.terminated = False

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated = True
        self.returncode = -15

    def kill(self):
        self.terminated = True
        self.returncode = -9

    def wait(self, timeout=None):
        if self.returncode is None:
            raise worker_module.subprocess.TimeoutExpired("inspection", timeout)
        return self.returncode


def test_isolated_inspector_honors_cancellation_without_unbounded_capture(
    monkeypatch,
    tmp_path,
):
    workspace, _artifacts, _revisions, record = _prepared_revision(tmp_path)
    process = _InspectionProcess()
    monkeypatch.setattr(
        worker_module.subprocess,
        "Popen",
        lambda *args, **kwargs: process,
    )
    cancelled = threading.Event()
    cancelled.set()

    with pytest.raises(InspectionWorkerError, match="cancelled"):
        IsolatedFEMInspector(workspace).inspect(
            record.spec,
            record.revision_hash,
            cancel_event=cancelled,
        )

    assert process.terminated


def test_isolated_inspector_rejects_excessive_control_output(
    monkeypatch,
    tmp_path,
):
    workspace, _artifacts, _revisions, record = _prepared_revision(tmp_path)
    process = _InspectionProcess(
        stdout=b"x"
        * (worker_module._FIXED_INSPECTION_OUTPUT_LIMIT_BYTES + 1),
    )
    monkeypatch.setattr(
        worker_module.subprocess,
        "Popen",
        lambda *args, **kwargs: process,
    )

    with pytest.raises(InspectionWorkerError, match="control-output limit"):
        IsolatedFEMInspector(workspace).inspect(
            record.spec,
            record.revision_hash,
        )

    assert process.terminated


def test_control_json_uses_utf8_bytes_and_enforces_the_byte_limit():
    value = {
        "acceleration": "mm/s²",
        "density": "tonne/mm³",
        "region": "自由端",
    }

    payload = worker_module._encode_control_json(value)

    assert json.loads(payload.decode("utf-8")) == value
    assert worker_module._read_control_json(
        io.BytesIO(payload),
        description="inspection request",
    ) == value
    with pytest.raises(ValueError, match="control payload limit"):
        worker_module._read_control_json(
            io.BytesIO(
                b"x"
                * (worker_module._FIXED_CONTROL_PAYLOAD_LIMIT_BYTES + 1)
            ),
            description="inspection request",
        )


@pytest.mark.integration
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


def test_worker_contract_round_trips_through_json():
    request = WorkerRequest(
        "ses_worker",
        1,
        "a" * 64,
        "run_worker",
        "solve_worker",
        "2026-07-24T12:00:00Z",
    )

    assert WorkerRequest.from_dict(json.loads(json.dumps(request.to_dict()))) == request


def test_worker_deadline_watchdog_uses_the_persisted_deadline():
    expired = threading.Event()
    exit_codes = []
    deadline = (
        datetime.now(timezone.utc) + timedelta(seconds=0.05)
    ).isoformat()
    watchdog = _start_worker_deadline_watchdog(
        deadline,
        exit_process=lambda code: (exit_codes.append(code), expired.set()),
    )
    try:
        assert expired.wait(timeout=2)
    finally:
        assert watchdog is not None
        watchdog.cancel()

    assert exit_codes == [124]


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


def test_process_termination_failure_is_normalized():
    class UnterminableProcess:
        def poll(self):
            return None

        def terminate(self):
            raise PermissionError("termination denied")

    assert _terminate_process(UnterminableProcess()) is False


@pytest.mark.parametrize(
    ("spec_limit", "fixed_limit", "effective_limit"),
    [
        (32, 64, 32),
        (80, 64, 64),
    ],
)
def test_worker_terminates_when_combined_logs_exceed_the_effective_limit(
    tmp_path,
    monkeypatch,
    spec_limit,
    fixed_limit,
    effective_limit,
):
    workspace, artifacts, revisions, record = _prepared_revision(
        tmp_path,
        resource_limits=ResourceLimits(max_output_bytes=spec_limit),
    )
    ConfirmationStore(workspace, revisions).confirm(
        record.session_id,
        revision=record.revision,
        revision_hash=record.revision_hash,
    )
    monkeypatch.setattr(
        worker_module,
        "_FIXED_WORKER_LOG_LIMIT_BYTES",
        fixed_limit,
    )
    monkeypatch.setattr(worker_module, "_repository_commit", lambda: None)
    launched = []

    class FloodingProcess:
        def __init__(self):
            self.pid = 44_444
            self.returncode = None
            self.stdout = io.BytesIO(b"O" * effective_limit)
            self.stderr = io.BytesIO(b"E" * (effective_limit + 1))
            self.terminate_calls = 0

        def poll(self):
            return self.returncode

        def terminate(self):
            self.terminate_calls += 1
            self.returncode = -15

        def wait(self, timeout=None):
            return self.returncode

        def kill(self):
            self.returncode = -9

    def launch_flood(*_args, **_kwargs):
        process = FloodingProcess()
        launched.append(process)
        return process

    monkeypatch.setattr(worker_module.subprocess, "Popen", launch_flood)
    worker = IsolatedFEMWorker(workspace)
    response = worker.run(
        record.session_id,
        revision=record.revision,
        revision_hash=record.revision_hash,
        idempotency_key=f"log_limit_{spec_limit}",
        timeout_seconds=30,
    )

    assert response.status == RunStatus.FAILED
    assert [item.code for item in response.diagnostics] == ["RESOURCE_LIMIT"]
    diagnostic = response.diagnostics[0]
    assert diagnostic.entity == "worker-logs"
    assert f"spec.max_output_bytes={spec_limit}" in diagnostic.message
    assert f"fixed_log_limit={fixed_limit}" in diagnostic.message
    assert launched[0].terminate_calls == 1
    run = artifacts.run_directory(record.session_id, response.run_id)
    total_log_bytes = sum(
        (run.path / "logs" / name).stat().st_size
        for name in ("worker-stdout.log", "worker-stderr.log")
    )
    assert total_log_bytes == effective_limit
    assert (run.path / "manifest.json").is_file()

    repeated = worker.run(
        record.session_id,
        revision=record.revision,
        revision_hash=record.revision_hash,
        idempotency_key=f"log_limit_{spec_limit}",
        timeout_seconds=30,
    )

    assert repeated == response
    assert len(launched) == 1


@pytest.mark.parametrize(
    ("file_sizes", "file_limit", "byte_limit"),
    [
        ((1, 1), 1, 100),
        ((5,), 10, 4),
    ],
)
def test_worker_terminates_when_staged_exports_exceed_the_spec_quota(
    tmp_path,
    monkeypatch,
    file_sizes,
    file_limit,
    byte_limit,
):
    workspace, artifacts, revisions, record = _prepared_revision(
        tmp_path,
        resource_limits=ResourceLimits(
            max_output_files=file_limit,
            max_output_bytes=byte_limit,
        ),
    )
    ConfirmationStore(workspace, revisions).confirm(
        record.session_id,
        revision=record.revision,
        revision_hash=record.revision_hash,
    )
    monkeypatch.setattr(worker_module, "_repository_commit", lambda: None)
    launched = []

    class ExportingProcess:
        def __init__(self, command):
            self.pid = 55_555
            self.returncode = None
            self.stdout = io.BytesIO()
            self.stderr = io.BytesIO()
            self.terminate_calls = 0
            request_path = Path(
                command[command.index("--request") + 1]
            )
            self.run_path = request_path.parent.parent
            staging = (
                self.run_path
                / "exports"
                / ".fem-agent-export-test"
            )
            staging.mkdir()
            for index, size in enumerate(file_sizes):
                (staging / f"part-{index}.bin").write_bytes(b"X" * size)

        def poll(self):
            return self.returncode

        def terminate(self):
            self.terminate_calls += 1
            self.returncode = -15

        def wait(self, timeout=None):
            return self.returncode

        def kill(self):
            self.returncode = -9

    def launch_export(command, **_kwargs):
        process = ExportingProcess(command)
        launched.append(process)
        return process

    monkeypatch.setattr(worker_module.subprocess, "Popen", launch_export)
    response = IsolatedFEMWorker(workspace).run(
        record.session_id,
        revision=record.revision,
        revision_hash=record.revision_hash,
        idempotency_key=f"export_limit_{file_limit}_{byte_limit}",
        timeout_seconds=30,
    )

    assert response.status == RunStatus.FAILED
    assert [item.code for item in response.diagnostics] == ["RESOURCE_LIMIT"]
    diagnostic = response.diagnostics[0]
    assert diagnostic.entity == "worker-exports"
    assert f"max_output_files={file_limit}" in diagnostic.message
    assert f"max_output_bytes={byte_limit}" in diagnostic.message
    assert launched[0].terminate_calls == 1
    assert not (
        launched[0].run_path
        / "exports"
        / ".fem-agent-export-test"
    ).exists()
    terminal_run = artifacts.run_directory(
        record.session_id,
        response.run_id,
    )
    assert terminal_run.path == launched[0].run_path
    assert (terminal_run.path / "manifest.json").is_file()

    repeated = IsolatedFEMWorker(workspace).run(
        record.session_id,
        revision=record.revision,
        revision_hash=record.revision_hash,
        idempotency_key=f"export_limit_{file_limit}_{byte_limit}",
        timeout_seconds=30,
    )

    assert repeated == response
    assert len(launched) == 1


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


def test_worker_environment_scrubs_provider_credentials():
    clean = scrub_worker_environment(
        {
            "PATH": "bin",
            "DEEPSEEK_API_KEY": "deep-secret",
            "OPENAI_API_KEY": "open-secret",
            "GITHUB_TOKEN": "github-secret",
            "FEM_AUTH": "custom-secret",
            "PYTHONPATH": "malicious-module-path",
            "PYTHONHOME": "malicious-runtime-path",
            "ORDINARY_SETTING": "retained",
        }
    )

    assert clean["PATH"] == "bin"
    assert "ORDINARY_SETTING" not in clean
    assert "FEM_AUTH" not in clean
    assert "PYTHONPATH" not in clean
    assert "PYTHONHOME" not in clean
    assert clean["FEM_AGENT_WORKER"] == "1"
    assert all("secret" not in value for value in clean.values())


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
    workspace, artifacts, revisions, record = _prepared_revision(tmp_path)
    ConfirmationStore(workspace, revisions).confirm(
        record.session_id,
        revision=record.revision,
        revision_hash=record.revision_hash,
    )
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
    workspace, artifacts, revisions, record = _prepared_revision(tmp_path)
    ConfirmationStore(workspace, revisions).confirm(
        record.session_id,
        revision=record.revision,
        revision_hash=record.revision_hash,
    )
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


def test_cross_instance_claim_prevents_a_duplicate_worker_launch(
    tmp_path,
    monkeypatch,
):
    workspace, artifacts, revisions, record = _prepared_revision(tmp_path)
    ConfirmationStore(workspace, revisions).confirm(
        record.session_id,
        revision=record.revision,
        revision_hash=record.revision_hash,
    )
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
        if not release_launch.wait(timeout=5):
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
    assert entered_launch.wait(timeout=5)
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
        thread.join(timeout=5)

    assert not thread.is_alive()
    assert not first_errors
    assert len(first_result) == 1
    assert len(list((artifacts.session_path(record.session_id) / "runs").iterdir())) == 1


@pytest.mark.integration
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


@pytest.mark.integration
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


@pytest.mark.integration
def test_isolated_postsolve_query_rejects_a_tampered_solution(tmp_path):
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
        idempotency_key="tamper_solution",
        timeout_seconds=30,
    )
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


@pytest.mark.integration
def test_worker_recovers_a_missing_response_from_a_verified_manifest(tmp_path):
    workspace, artifacts, revisions, record = _prepared_revision(tmp_path)
    ConfirmationStore(workspace, revisions).confirm(
        record.session_id,
        revision=record.revision,
        revision_hash=record.revision_hash,
    )
    worker = IsolatedFEMWorker(workspace)
    original = worker.run(
        record.session_id,
        revision=record.revision,
        revision_hash=record.revision_hash,
        idempotency_key="recover_committed_solve",
        timeout_seconds=30,
    )
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
        idempotency_key="recover_committed_solve",
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


@pytest.mark.integration
def test_worker_fails_closed_when_a_commit_manifest_is_corrupt(tmp_path):
    workspace, artifacts, revisions, record = _prepared_revision(tmp_path)
    ConfirmationStore(workspace, revisions).confirm(
        record.session_id,
        revision=record.revision,
        revision_hash=record.revision_hash,
    )
    worker = IsolatedFEMWorker(workspace)
    original = worker.run(
        record.session_id,
        revision=record.revision,
        revision_hash=record.revision_hash,
        idempotency_key="corrupt_committed_manifest",
        timeout_seconds=30,
    )
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
            idempotency_key="corrupt_committed_manifest",
            timeout_seconds=30,
        )

    assert not response_path.exists()
    assert len(list(run.path.parent.iterdir())) == 1


@pytest.mark.integration
def test_worker_fails_closed_when_a_manifest_artifact_hash_is_wrong(tmp_path):
    workspace, artifacts, revisions, record = _prepared_revision(tmp_path)
    ConfirmationStore(workspace, revisions).confirm(
        record.session_id,
        revision=record.revision,
        revision_hash=record.revision_hash,
    )
    worker = IsolatedFEMWorker(workspace)
    original = worker.run(
        record.session_id,
        revision=record.revision,
        revision_hash=record.revision_hash,
        idempotency_key="corrupt_committed_artifact",
        timeout_seconds=30,
    )
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
            idempotency_key="corrupt_committed_artifact",
            timeout_seconds=30,
        )

    assert not response_path.exists()
    assert len(list(run.path.parent.iterdir())) == 1


@pytest.mark.integration
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


@pytest.mark.integration
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


@pytest.mark.integration
@pytest.mark.platform
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
