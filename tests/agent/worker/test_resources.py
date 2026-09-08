import io
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

import fem_agent.worker as worker_module
from fem_agent.confirmation import ConfirmationStore
from fem_agent.schemas import ResourceLimits, RunStatus
from fem_agent.worker import (
    IsolatedFEMWorker,
    _start_worker_deadline_watchdog,
    _terminate_process,
    scrub_worker_environment,
)

from tests.helpers.agent_worker_fixtures import (
    _prepared_revision,
)


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

    process_api = SimpleNamespace(
        Popen=launch_flood,
        CREATE_NO_WINDOW=getattr(
            worker_module.subprocess,
            "CREATE_NO_WINDOW",
            0,
        ),
        DEVNULL=worker_module.subprocess.DEVNULL,
        PIPE=worker_module.subprocess.PIPE,
        TimeoutExpired=worker_module.subprocess.TimeoutExpired,
    )
    monkeypatch.setattr(worker_module, "subprocess", process_api)
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
