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
