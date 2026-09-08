import io
import json
import threading

import pytest

import fem_agent.worker as worker_module
from fem_agent.worker import InspectionWorkerError, IsolatedFEMInspector, WorkerRequest

from tests.helpers.agent_worker_fixtures import (
    _prepared_revision,
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


@pytest.mark.parametrize("cancelled", [True, False], ids=["cancel", "output-limit"])
def test_isolated_inspector_terminates_on_cancellation_or_output_limit(monkeypatch, tmp_path, cancelled):
    workspace, _, _, record = _prepared_revision(tmp_path)
    process = _InspectionProcess(
        stdout=b"" if cancelled else b"x" * (worker_module._FIXED_INSPECTION_OUTPUT_LIMIT_BYTES + 1),
    )
    monkeypatch.setattr(worker_module.subprocess, "Popen", lambda *args, **kwargs: process)
    cancellation = threading.Event()
    if cancelled:
        cancellation.set()
    with pytest.raises(InspectionWorkerError):
        IsolatedFEMInspector(workspace).inspect(
            record.spec, record.revision_hash, cancel_event=cancellation,
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
