import io
import json
import threading
from types import SimpleNamespace

import pytest

import fem_agent.worker as worker_module
from fem_agent.schemas import ResultQuery, ResultQueryKind, ResultSummary, RunStatus
from fem_agent.worker import (
    IsolatedFEMInspector,
    IsolatedFEMResultQuerier,
    execute_inspection_request,
)
from fem_agent.worker_protocol import (
    InspectionRequest,
    InspectionWorkerError,
    ResultQueryRequest,
    ResultQueryWorkerError,
    WorkerResponse,
)

from tests.helpers.agent_worker_fixtures import _prepared_revision


class _RequestPipe(io.BytesIO):
    def __init__(self):
        super().__init__()
        self.payload = b""
        self.write_error = None

    def write(self, payload):
        if self.write_error is not None:
            raise self.write_error
        return super().write(payload)

    def close(self):
        if not self.closed:
            self.payload = self.getvalue()
        super().close()


class _ControlProcess:
    def __init__(self, stdout, stderr=b"", returncode=0):
        self.stdin = _RequestPipe()
        self.stdout = io.BytesIO(stdout)
        self.stderr = io.BytesIO(stderr)
        self.returncode = returncode
        self.terminate_calls = 0
        self.wait_timeouts = []
        self.wait_error = None

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminate_calls += 1
        self.returncode = -15

    def kill(self):
        self.returncode = -9

    def wait(self, timeout=None):
        self.wait_timeouts.append(timeout)
        if self.wait_error is not None:
            error, self.wait_error = self.wait_error, None
            raise error
        if self.returncode is None:
            raise worker_module.subprocess.TimeoutExpired("control process", timeout)
        return self.returncode


class _ControlReader:
    """Deterministic pipe readers also allow output to arrive during join."""

    def __init__(self, target, args, name, daemon, options):
        self.target = target
        self.args = args
        self.name = name
        self.daemon = daemon
        self.options = options
        self.join_timeouts = []

    def start(self):
        if self.options.start_error is not None:
            raise self.options.start_error
        if not self.options.deferred:
            self.target(*self.args)

    def join(self, timeout):
        assert not self.args[0].closed
        self.join_timeouts.append(timeout)
        if self.options.deferred:
            self.target(*self.args)

    def is_alive(self):
        return self.options.alive


@pytest.fixture
def control_readers(monkeypatch):
    options = SimpleNamespace(
        deferred=False, alive=False, start_error=None, threads=[]
    )

    def create_reader(**kwargs):
        reader = _ControlReader(**kwargs, options=options)
        options.threads.append(reader)
        return reader

    monkeypatch.setattr(
        worker_module,
        "threading",
        SimpleNamespace(Thread=create_reader, Lock=threading.Lock, Event=threading.Event),
    )
    return options


@pytest.fixture(params=["inspection", "result-query"])
def control_call(request, tmp_path, control_readers):
    workspace, _, _, record = _prepared_revision(tmp_path)
    if request.param == "inspection":
        client = IsolatedFEMInspector(workspace)
        control_request = InspectionRequest(record.spec, record.revision_hash, validate=True)
        expected = execute_inspection_request(workspace, control_request)

        def invoke(**kwargs):
            return client.inspect(record.spec, record.revision_hash, validate=True, **kwargs)

        error_type = InspectionWorkerError
        description = "The isolated Abaqus inspection"
    else:
        client = IsolatedFEMResultQuerier(workspace)
        response = WorkerResponse(
            session_id=record.session_id,
            revision=record.revision,
            revision_hash=record.revision_hash,
            run_id="run_control",
            status=RunStatus.SUCCEEDED,
            result_summary=None,
            artifacts=(),
            diagnostics=(),
            elapsed_seconds=0.01,
        )
        queries = (
            ResultQuery(ResultQueryKind.REACTION_SUM, component=1, node_set="自由端"),
        )
        control_request = ResultQueryRequest(
            record.session_id, record.revision, record.revision_hash,
            response.run_id, queries,
        )
        expected = ResultSummary(response.run_id, record.spec.analysis_step, True, ())

        def invoke(**kwargs):
            return client.query(response, queries, **kwargs)

        error_type = ResultQueryWorkerError
        description = "The isolated result query"
    return SimpleNamespace(
        invoke=invoke,
        operation=request.param,
        request=control_request,
        expected=expected,
        payload=json.dumps(expected.to_dict(), ensure_ascii=False).encode("utf-8"),
        error_type=error_type,
        description=description,
        workspace=workspace,
    )


@pytest.fixture
def control_process(monkeypatch, control_call):
    process = _ControlProcess(control_call.payload, stderr="子进程日志".encode("utf-8"))
    process.launches = []

    def launch(command, **kwargs):
        process.launches.append((command, kwargs))
        return process

    monkeypatch.setattr(worker_module.subprocess, "Popen", launch)
    return process


def _assert_pipes_closed(process):
    for name in ("stdin", "stdout", "stderr"):
        pipe = getattr(process, name)
        if pipe is not None:
            assert pipe.closed


def test_control_process_returns_response_and_closes_joined_pipes(
    monkeypatch, control_call, control_process, control_readers,
):
    monkeypatch.setenv("OPENAI_API_KEY", "test-secret")
    monkeypatch.setenv("PYTHONPATH", "test-injection")
    cancellation = threading.Event()
    cancellation.set()

    assert control_call.invoke(cancel_event=cancellation) == control_call.expected

    command, options = control_process.launches[0]
    assert command[1:5] == ["-I", "-B", "-m", "fem_agent.worker"]
    assert command[-1] == (
        "--inspect-stdin" if control_call.operation == "inspection" else "--query-stdin"
    )
    assert command[-2] == str(control_call.workspace)
    assert options["shell"] is False
    assert options["stdin"] == options["stdout"] == options["stderr"] == worker_module.subprocess.PIPE
    assert "OPENAI_API_KEY" not in options["env"]
    assert "PYTHONPATH" not in options["env"]
    assert json.loads(control_process.stdin.payload.decode("utf-8")) == control_call.request.to_dict()
    assert control_process.terminate_calls == 0
    assert control_process.wait_timeouts == [worker_module._WORKER_LOG_JOIN_SECONDS]
    assert [reader.name for reader in control_readers.threads] == [
        f"fem-agent-{control_call.operation}-stdout",
        f"fem-agent-{control_call.operation}-stderr",
    ]
    assert all(reader.daemon and len(reader.join_timeouts) == 1 for reader in control_readers.threads)
    _assert_pipes_closed(control_process)


@pytest.mark.parametrize(
    ("stdout", "stderr", "cancelled", "expired", "message"),
    [
        (b"", b"", True, False, "was cancelled."),
        (b"", b"", False, True, "exceeded its time limit."),
        (b"x" * 33, b"", False, False, "exceeded its bounded control-output limit."),
        (b"", b"x" * 33, False, False, "exceeded its bounded control-output limit."),
        (b"x" * 20, b"x" * 20, True, True, "exceeded its bounded control-output limit."),
        (b"", b"", True, True, "was cancelled."),
    ],
    ids=["cancel", "timeout", "stdout-limit", "stderr-limit", "combined-limit-priority", "cancel-priority"],
)
def test_control_process_terminates_for_limits_in_original_priority(
    monkeypatch, control_call, control_process, stdout, stderr, cancelled, expired, message,
):
    control_process.stdout = io.BytesIO(stdout)
    control_process.stderr = io.BytesIO(stderr)
    control_process.returncode = None
    monkeypatch.setattr(worker_module, "_FIXED_INSPECTION_OUTPUT_LIMIT_BYTES", 32)
    clock_values = iter([0.0, 2.0])
    monkeypatch.setattr(
        worker_module, "time",
        SimpleNamespace(monotonic=lambda: next(clock_values, 2.0 if expired else 0.0)),
    )
    cancellation = threading.Event()
    if cancelled:
        cancellation.set()

    with pytest.raises(control_call.error_type) as caught:
        control_call.invoke(timeout_seconds=1.0, cancel_event=cancellation)

    assert str(caught.value) == f"{control_call.description} {message}"
    assert control_process.terminate_calls == 1
    _assert_pipes_closed(control_process)


def test_control_process_checks_output_limit_after_exit_and_reader_join(
    monkeypatch, control_call, control_process, control_readers,
):
    monkeypatch.setattr(worker_module, "_FIXED_INSPECTION_OUTPUT_LIMIT_BYTES", 32)
    control_process.stdout = io.BytesIO(b"x" * 33)
    control_readers.deferred = True

    with pytest.raises(control_call.error_type) as caught:
        control_call.invoke()

    assert str(caught.value) == f"{control_call.description} exceeded its bounded control-output limit."
    assert control_process.terminate_calls == 0
    _assert_pipes_closed(control_process)


@pytest.mark.parametrize("capture_fault", ["read-error", "reader-alive"])
def test_control_process_capture_failure_precedes_cancellation_and_exit_status(
    monkeypatch, control_call, control_process, control_readers, capture_fault,
):
    if capture_fault == "read-error":
        def fail_read(_size):
            raise OSError("pipe read failed")

        monkeypatch.setattr(control_process.stdout, "read", fail_read)
    else:
        control_readers.alive = True
    control_process.returncode = None
    cancellation = threading.Event()
    cancellation.set()

    with pytest.raises(control_call.error_type) as caught:
        control_call.invoke(cancel_event=cancellation)

    assert str(caught.value) == f"{control_call.description} output could not be captured safely."
    assert control_process.terminate_calls == 1
    _assert_pipes_closed(control_process)


@pytest.mark.parametrize("missing_pipe", ["stdin", "stdout", "stderr"])
def test_control_process_rejects_missing_pipes_and_cleans_up(
    control_call, control_process, missing_pipe,
):
    getattr(control_process, missing_pipe).close()
    setattr(control_process, missing_pipe, None)
    control_process.returncode = None

    with pytest.raises(control_call.error_type) as caught:
        control_call.invoke()

    assert str(caught.value) == f"{control_call.description} process could not start."
    pipe_description = "inspection" if control_call.operation == "inspection" else "result query"
    assert str(caught.value.__cause__) == f"{pipe_description} process pipes were not created"
    assert control_process.terminate_calls == 1
    _assert_pipes_closed(control_process)


@pytest.mark.parametrize("error_stage", ["launch", "write", "wait", "wait-timeout"])
def test_control_process_normalizes_process_errors_after_cleanup(
    monkeypatch, control_call, control_process, error_stage,
):
    error = OSError("control process failed")
    if error_stage == "launch":
        def fail_launch(*_args, **_kwargs):
            raise error

        monkeypatch.setattr(worker_module.subprocess, "Popen", fail_launch)
    elif error_stage == "write":
        control_process.stdin.write_error = error
        control_process.returncode = None
    else:
        if error_stage == "wait-timeout":
            error = worker_module.subprocess.TimeoutExpired("control process", 3.0)
        control_process.wait_error = error

    with pytest.raises(control_call.error_type) as caught:
        control_call.invoke()

    message = "process did not stop cleanly." if error_stage == "wait-timeout" else "process could not start."
    assert str(caught.value) == f"{control_call.description} {message}"
    assert caught.value.__cause__ is error
    if error_stage != "launch":
        _assert_pipes_closed(control_process)
    if error_stage == "write":
        assert control_process.terminate_calls == 1


def test_control_process_does_not_wrap_cleanup_errors(
    monkeypatch, control_call, control_process,
):
    cleanup_error = OSError("pipe cleanup failed")
    close_pipes = worker_module._close_process_pipes

    def fail_cleanup(process):
        close_pipes(process)
        raise cleanup_error

    monkeypatch.setattr(worker_module, "_close_process_pipes", fail_cleanup)
    control_process.stdin.write_error = OSError("request write failed")

    with pytest.raises(OSError) as caught:
        control_call.invoke()

    assert caught.value is cleanup_error
    _assert_pipes_closed(control_process)


def test_control_process_cleans_up_when_reader_start_fails(
    control_call, control_process, control_readers,
):
    error = RuntimeError("reader could not start")
    control_readers.start_error = error
    control_process.returncode = None

    with pytest.raises(RuntimeError) as caught:
        control_call.invoke()

    assert caught.value is error
    assert control_process.terminate_calls == 1
    _assert_pipes_closed(control_process)


@pytest.mark.parametrize("returncode", [1, -9])
def test_control_process_requires_successful_exit_before_decoding_response(
    control_call, control_process, returncode,
):
    control_process.stdout = io.BytesIO(b"invalid response")
    control_process.returncode = returncode

    with pytest.raises(control_call.error_type) as caught:
        control_call.invoke()

    assert str(caught.value) == f"{control_call.description} process failed."
    _assert_pipes_closed(control_process)


@pytest.mark.parametrize("payload", [b"\xff", b"not-json", b"[]", b"{}"])
def test_control_process_rejects_invalid_business_response(
    control_call, control_process, payload,
):
    control_process.stdout = io.BytesIO(payload)

    with pytest.raises(control_call.error_type) as caught:
        control_call.invoke()

    assert str(caught.value) == f"{control_call.description} returned an invalid response."
    assert caught.value.__cause__ is not None
    _assert_pipes_closed(control_process)


@pytest.mark.parametrize("identity_part", ["primary", "context"])
def test_control_process_rejects_response_with_wrong_identity(
    control_call, control_process, identity_part,
):
    payload = control_call.expected.to_dict()
    if control_call.operation == "inspection":
        if identity_part == "primary":
            payload["summary"]["revision"] += 1
        else:
            payload["summary"]["source_sha256"] = "0" * 64
    elif identity_part == "primary":
        payload["run_id"] = "run_other"
    else:
        payload["step"] = "Another-step"
    control_process.stdout = io.BytesIO(json.dumps(payload).encode("utf-8"))

    with pytest.raises(control_call.error_type) as caught:
        control_call.invoke()

    assert str(caught.value) == f"{control_call.description} response identity did not match."
    _assert_pipes_closed(control_process)


@pytest.mark.parametrize("delay_stage", ["encoding", "launch"])
def test_control_process_timeout_starts_after_encoding_and_before_launch(
    monkeypatch, control_call, control_process, delay_stage,
):
    clock = {"now": 0.0}
    encode = worker_module._encode_control_json

    def delayed_encode(value):
        if delay_stage == "encoding":
            clock["now"] += 10.0
        return encode(value)

    def launch(*_args, **_kwargs):
        if delay_stage == "launch":
            clock["now"] += 10.0
        return control_process

    def complete_after_poll(_seconds):
        control_process.returncode = 0

    monkeypatch.setattr(worker_module, "_encode_control_json", delayed_encode)
    monkeypatch.setattr(worker_module.subprocess, "Popen", launch)
    monkeypatch.setattr(
        worker_module, "time",
        SimpleNamespace(monotonic=lambda: clock["now"], sleep=complete_after_poll),
    )
    control_process.returncode = None

    if delay_stage == "encoding":
        assert control_call.invoke(timeout_seconds=1.0) == control_call.expected
        assert control_process.terminate_calls == 0
    else:
        with pytest.raises(control_call.error_type) as caught:
            control_call.invoke(timeout_seconds=1.0)
        assert str(caught.value) == f"{control_call.description} exceeded its time limit."
        assert control_process.terminate_calls == 1
    _assert_pipes_closed(control_process)
