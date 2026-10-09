"""Isolated, revision-bound FEM worker process."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import platform
import shutil
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import fields, is_dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Literal, Mapping

import numpy as np
import scipy

from fem import materials
from fem.core.result import ModelResult

from .artifacts import (
    ArtifactStore,
    RunDirectory,
    atomic_write_bytes,
    atomic_write_json,
    read_json_file,
    safe_child,
    validate_identifier,
)
from .confirmation import ConfirmationStore
from .diagnostics import (
    DiagnosticCode,
    has_errors,
    make_diagnostic,
)
from .schemas import (
    ArtifactRecord,
    Diagnostic,
    DiagnosticSeverity,
    ImportAnalysisSpec,
    ResultQuery,
    ResultSummary,
    RunManifest,
    RunStatus,
)
from .state import RevisionRecord, RevisionStore
from .summaries import build_analysis_summary
from .tools.exports import export_results
from .tools.importing import inspect_abaqus
from .tools.results import query_results
from .tools.solving import solve_analysis
from .tools.validation import validate_analysis
from .worker_integrity import (
    WorkerResponseIntegrityError,
    _recover_response_from_manifest,
    _require_request_identity,
    load_verified_worker_response,
)
from .worker_protocol import (
    INSPECTION_SCHEMA_VERSION,
    RESULT_QUERY_SCHEMA_VERSION,
    WORKER_SCHEMA_VERSION,
    InspectionRequest,
    InspectionResponse,
    InspectionWorkerError,
    ResultQueryRequest,
    ResultQueryWorkerError,
    WorkerRequest,
    WorkerResponse,
    _encode_control_json,
    _mapping,
    _read_control_json,
    _sha256,
    _timestamp,
    _write_control_json,
)


_WORKER_SELF_TIMEOUT_EXIT_CODE = 124
_WORKER_DEADLINE_GRACE_SECONDS = 2.0
_MAX_WORKER_ATTEMPTS = 8
_FIXED_WORKER_LOG_LIMIT_BYTES = 8 * 1024 * 1024
_FIXED_INSPECTION_OUTPUT_LIMIT_BYTES = 2 * 1024 * 1024
_INSPECTION_POLL_SECONDS = 0.05
_WORKER_LOG_READ_BYTES = 64 * 1024
_WORKER_LOG_JOIN_SECONDS = 3.0
_RETRY_EXHAUSTED_DIAGNOSTIC_CODE = "WORKER_RETRY_EXHAUSTED"
_SENSITIVE_ENVIRONMENT_FRAGMENTS = (
    "API_KEY",
    "APIKEY",
    "TOKEN",
    "ACCESS_TOKEN",
    "AUTH_TOKEN",
    "BEARER_TOKEN",
    "PASSWORD",
    "SECRET",
    "CREDENTIAL",
    "DEEPSEEK",
    "OPENAI",
    "ANTHROPIC",
)
_ALLOWED_WORKER_ENVIRONMENT = frozenset(
    {
        "COMSPEC",
        "LANG",
        "LC_ALL",
        "NUMBER_OF_PROCESSORS",
        "OS",
        "PATH",
        "PATHEXT",
        "PROCESSOR_ARCHITECTURE",
        "SYSTEMDRIVE",
        "SYSTEMROOT",
        "TEMP",
        "TMP",
        "TMPDIR",
        "TZ",
        "WINDIR",
    }
)
_TRANSIENT_WORKER_DIAGNOSTIC_CODES = frozenset(
    {
        DiagnosticCode.OPERATION_CANCELLED.value,
        DiagnosticCode.WORKER_CRASH.value,
        DiagnosticCode.WORKER_TIMEOUT.value,
    }
)


class _BoundedWorkerLogCapture:
    """Copy two worker streams under one strict, shared byte budget."""

    def __init__(self, limit_bytes: int):
        if (
            isinstance(limit_bytes, bool)
            or not isinstance(limit_bytes, int)
            or limit_bytes <= 0
        ):
            raise ValueError("worker log limit must be a positive integer")
        self.limit_bytes = limit_bytes
        self._captured_bytes = 0
        self._lock = threading.Lock()
        self._exceeded = threading.Event()
        self._failure: Exception | None = None

    @property
    def captured_bytes(self) -> int:
        with self._lock:
            return self._captured_bytes

    @property
    def exceeded(self) -> bool:
        return self._exceeded.is_set()

    @property
    def failure(self) -> Exception | None:
        with self._lock:
            return self._failure

    def copy(self, source: Any, destination: Any) -> None:
        try:
            while True:
                chunk = source.read(_WORKER_LOG_READ_BYTES)
                if not chunk:
                    return
                with self._lock:
                    remaining = self.limit_bytes - self._captured_bytes
                    accepted = min(len(chunk), remaining)
                    if accepted:
                        written = destination.write(chunk[:accepted])
                        if written != accepted:
                            raise OSError("worker log write was incomplete")
                        self._captured_bytes += accepted
                    if accepted != len(chunk):
                        self._exceeded.set()
                        return
        except Exception as error:
            with self._lock:
                if self._failure is None:
                    self._failure = error


class WorkerRunInProgressError(RuntimeError):
    """The same persisted revision already has a live worker process."""


def _run_bounded_control_process(
    command: list[str],
    request_payload: bytes,
    *,
    operation: Literal["inspection", "result-query"],
    creationflags: int,
    timeout_seconds: float,
    cancel_event: threading.Event | None,
) -> tuple[bytes, str | None, OSError | subprocess.TimeoutExpired | None]:
    """Return stdout, a control failure reason, and any launch/stop error.

    Cleanup errors propagate directly; the caller translates launch/stop errors
    and control failures into its own business exception.
    """
    process: subprocess.Popen[bytes] | None = None
    capture = _BoundedWorkerLogCapture(
        _FIXED_INSPECTION_OUTPUT_LIMIT_BYTES
    )
    stdout_buffer = io.BytesIO()
    stderr_buffer = io.BytesIO()
    readers: list[threading.Thread] = []
    failure: str | None = None
    process_error: OSError | subprocess.TimeoutExpired | None = None
    started = time.monotonic()
    try:
        process = subprocess.Popen(
            command,
            cwd=Path(__file__).resolve().parents[2],
            env=scrub_worker_environment(os.environ),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=False,
            creationflags=creationflags,
        )
        if (
            process.stdin is None
            or process.stdout is None
            or process.stderr is None
        ):
            description = "inspection" if operation == "inspection" else "result query"
            raise OSError(f"{description} process pipes were not created")
        process.stdin.write(request_payload)
        process.stdin.close()
        for name, source, destination in (
            ("stdout", process.stdout, stdout_buffer),
            ("stderr", process.stderr, stderr_buffer),
        ):
            reader = threading.Thread(
                target=capture.copy,
                args=(source, destination),
                name=f"fem-agent-{operation}-{name}",
                daemon=True,
            )
            reader.start()
            readers.append(reader)

        while process.poll() is None:
            if capture.exceeded:
                failure = "output-limit"
                _terminate_process(process)
                break
            if cancel_event is not None and cancel_event.is_set():
                failure = "cancelled"
                _terminate_process(process)
                break
            if time.monotonic() - started >= timeout_seconds:
                failure = "timeout"
                _terminate_process(process)
                break
            time.sleep(_INSPECTION_POLL_SECONDS)
        if process.poll() is None:
            _terminate_process(process)
        process.wait(timeout=_WORKER_LOG_JOIN_SECONDS)
    except OSError as exc:
        process_error = exc
    except subprocess.TimeoutExpired as exc:
        if process is not None:
            _terminate_process(process)
        process_error = exc
    finally:
        if process is not None and process.poll() is None:
            _terminate_process(process)
        deadline = time.monotonic() + _WORKER_LOG_JOIN_SECONDS
        for reader in readers:
            reader.join(max(0.0, deadline - time.monotonic()))
        _close_process_pipes(process)
    if process_error is not None:
        return b"", None, process_error
    if any(reader.is_alive() for reader in readers) or capture.failure is not None:
        return b"", "capture", None
    if capture.exceeded and failure is None:
        failure = "output-limit"
    if failure is not None:
        return b"", failure, None
    if process is None or process.returncode != 0:
        return b"", "exit", None
    return stdout_buffer.getvalue(), None, None


class IsolatedFEMInspector:
    """Parse and summarize an attached input outside the Agent process."""

    def __init__(
        self,
        workspace: str | os.PathLike[str],
        *,
        python_executable: str | os.PathLike[str] | None = None,
    ):
        self.artifacts = ArtifactStore(workspace)
        self.python_executable = str(python_executable or sys.executable)

    def inspect(
        self,
        spec: ImportAnalysisSpec,
        revision_hash: str,
        *,
        validate: bool = False,
        timeout_seconds: float | None = None,
        cancel_event: threading.Event | None = None,
    ) -> InspectionResponse:
        request = InspectionRequest(
            spec=spec,
            revision_hash=revision_hash,
            validate=validate,
        )
        timeout = (
            min(spec.resource_limits.worker_timeout_seconds, 60.0)
            if timeout_seconds is None
            else float(timeout_seconds)
        )
        if timeout <= 0:
            raise ValueError("timeout_seconds must be greater than zero")
        command = [
            self.python_executable,
            "-I",
            "-B",
            "-m",
            "fem_agent.worker",
            "--workspace",
            str(self.artifacts.root),
            "--inspect-stdin",
        ]
        creationflags = (
            getattr(subprocess, "CREATE_NO_WINDOW", 0)
            if os.name == "nt"
            else 0
        )
        request_payload = _encode_control_json(request.to_dict())
        stdout_payload, failure, process_error = _run_bounded_control_process(
            command,
            request_payload,
            operation="inspection",
            creationflags=creationflags,
            timeout_seconds=timeout,
            cancel_event=cancel_event,
        )
        if isinstance(process_error, OSError):
            raise InspectionWorkerError(
                "The isolated Abaqus inspection process could not start."
            ) from process_error
        if isinstance(process_error, subprocess.TimeoutExpired):
            raise InspectionWorkerError(
                "The isolated Abaqus inspection process did not stop cleanly."
            ) from process_error
        if failure is not None:
            messages = {
                "capture": "The isolated Abaqus inspection output could not be captured safely.",
                "output-limit": "The isolated Abaqus inspection exceeded its bounded control-output limit.",
                "cancelled": "The isolated Abaqus inspection was cancelled.",
                "timeout": "The isolated Abaqus inspection exceeded its time limit.",
                "exit": "The isolated Abaqus inspection process failed.",
            }
            raise InspectionWorkerError(messages[failure])
        try:
            payload = json.loads(stdout_payload.decode("utf-8"))
            response = InspectionResponse.from_dict(
                _mapping(payload, "inspection response")
            )
        except (UnicodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
            raise InspectionWorkerError(
                "The isolated Abaqus inspection returned an invalid response."
            ) from exc
        if (
            response.summary.revision != spec.revision
            or response.summary.revision_hash != revision_hash
            or response.summary.source_artifact_id != spec.source_artifact_id
            or response.summary.source_sha256 != spec.source_sha256
        ):
            raise InspectionWorkerError(
                "The isolated Abaqus inspection response identity did not match."
            )
        return response


class IsolatedFEMResultQuerier:
    """Evaluate bounded result queries outside the Agent process."""

    def __init__(
        self,
        workspace: str | os.PathLike[str],
        *,
        python_executable: str | os.PathLike[str] | None = None,
    ):
        self.artifacts = ArtifactStore(workspace)
        self.revisions = RevisionStore(self.artifacts.root)
        self.python_executable = str(python_executable or sys.executable)

    def query(
        self,
        response: WorkerResponse,
        queries: tuple[ResultQuery, ...],
        *,
        timeout_seconds: float | None = None,
        cancel_event: threading.Event | None = None,
    ) -> ResultSummary:
        if not isinstance(response, WorkerResponse):
            raise TypeError("response must be a WorkerResponse")
        if response.status != RunStatus.SUCCEEDED:
            raise ValueError("result queries require a successful local run")
        record = self.revisions.require_current(
            response.session_id,
            expected_revision=response.revision,
            expected_hash=response.revision_hash,
        )
        request = ResultQueryRequest(
            session_id=response.session_id,
            revision=response.revision,
            revision_hash=response.revision_hash,
            run_id=response.run_id,
            queries=tuple(queries),
        )
        timeout = (
            record.spec.resource_limits.worker_timeout_seconds
            if timeout_seconds is None
            else float(timeout_seconds)
        )
        if timeout <= 0:
            raise ValueError("timeout_seconds must be greater than zero")
        command = [
            self.python_executable,
            "-I",
            "-B",
            "-m",
            "fem_agent.worker",
            "--workspace",
            str(self.artifacts.root),
            "--query-stdin",
        ]
        creationflags = (
            getattr(subprocess, "CREATE_NO_WINDOW", 0)
            if os.name == "nt"
            else 0
        )
        request_payload = _encode_control_json(request.to_dict())
        stdout_payload, failure, process_error = _run_bounded_control_process(
            command,
            request_payload,
            operation="result-query",
            creationflags=creationflags,
            timeout_seconds=timeout,
            cancel_event=cancel_event,
        )
        if isinstance(process_error, OSError):
            raise ResultQueryWorkerError(
                "The isolated result query process could not start."
            ) from process_error
        if isinstance(process_error, subprocess.TimeoutExpired):
            raise ResultQueryWorkerError(
                "The isolated result query process did not stop cleanly."
            ) from process_error
        if failure is not None:
            messages = {
                "capture": "The isolated result query output could not be captured safely.",
                "output-limit": "The isolated result query exceeded its bounded control-output limit.",
                "cancelled": "The isolated result query was cancelled.",
                "timeout": "The isolated result query exceeded its time limit.",
                "exit": "The isolated result query process failed.",
            }
            raise ResultQueryWorkerError(messages[failure])
        try:
            payload = json.loads(stdout_payload.decode("utf-8"))
            summary = ResultSummary.from_dict(
                _mapping(payload, "result summary")
            )
        except (UnicodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
            raise ResultQueryWorkerError(
                "The isolated result query returned an invalid response."
            ) from exc
        if (
            summary.run_id != response.run_id
            or summary.step != record.spec.analysis_step
        ):
            raise ResultQueryWorkerError(
                "The isolated result query response identity did not match."
            )
        return summary


class IsolatedFEMWorker:
    """Launch one scrubbed subprocess per confirmed analysis run."""

    def __init__(
        self,
        workspace: str | os.PathLike[str],
        *,
        python_executable: str | os.PathLike[str] | None = None,
    ):
        self.artifacts = ArtifactStore(workspace)
        self.revisions = RevisionStore(self.artifacts.root)
        self.confirmations = ConfirmationStore(
            self.artifacts.root,
            self.revisions,
        )
        self.python_executable = str(python_executable or sys.executable)

    def run(
        self,
        session_id: str,
        *,
        revision: int,
        revision_hash: str,
        idempotency_key: str,
        timeout_seconds: float | None = None,
        cancel_event: threading.Event | None = None,
    ) -> WorkerResponse:
        """Run, reuse, time out, or cancel one exact confirmed revision."""

        record = self.revisions.require_current(
            session_id,
            expected_revision=revision,
            expected_hash=revision_hash,
        )
        self.confirmations.require_confirmed(
            session_id,
            revision=revision,
            revision_hash=revision_hash,
        )
        with _exclusive_worker_claim(
            self.artifacts,
            session_id,
            idempotency_key,
        ):
            timeout = (
                float(timeout_seconds)
                if timeout_seconds is not None
                else record.spec.resource_limits.worker_timeout_seconds
            )
            if timeout <= 0:
                raise ValueError("timeout_seconds must be greater than zero")
            return self._run_claimed(
                record,
                idempotency_key=idempotency_key,
                timeout=timeout,
                deadline_at=_deadline_after(timeout),
                cancel_event=cancel_event,
            )

    def _run_claimed(
        self,
        record: RevisionRecord,
        *,
        idempotency_key: str,
        timeout: float,
        deadline_at: str,
        cancel_event: threading.Event | None,
    ) -> WorkerResponse:
        retry = 0
        latest_valid_attempt = -1
        latest_damaged_attempt = -1
        latest_response_error: Exception | None = None
        transient_elapsed_seconds = 0.0
        while True:
            effective_key = _worker_attempt_key(idempotency_key, retry)
            run = self.artifacts._find_run_by_idempotency_key(
                record.session_id,
                effective_key,
            )
            if run is None:
                if latest_damaged_attempt > latest_valid_attempt:
                    raise WorkerResponseIntegrityError(
                        "The latest persisted worker response failed "
                        "integrity validation."
                    ) from latest_response_error
                run = self.artifacts.create_run(
                    record.session_id,
                    idempotency_key=effective_key,
                )
            response_path = safe_child(
                run.path,
                "logs",
                "worker-response.json",
            )
            if response_path.exists():
                try:
                    existing = load_verified_worker_response(
                        self.artifacts,
                        record,
                        run.run_id,
                    )
                except Exception as error:
                    latest_damaged_attempt = retry
                    latest_response_error = error
                    retry += 1
                    continue
                latest_valid_attempt = retry
                if (
                    existing.status == RunStatus.SUCCEEDED
                    or not _is_transient_worker_failure(existing)
                ):
                    return existing
                transient_elapsed_seconds += existing.elapsed_seconds
                if retry + 1 >= _MAX_WORKER_ATTEMPTS:
                    return _retry_exhausted_response(
                        self.artifacts,
                        record,
                        idempotency_key,
                        elapsed_seconds=transient_elapsed_seconds,
                    )
                retry += 1
                continue
            request_path = safe_child(run.path, "logs", "worker-request.json")
            stdout_path = safe_child(run.path, "logs", "worker-stdout.log")
            stderr_path = safe_child(run.path, "logs", "worker-stderr.log")
            process_state = safe_child(
                run.path,
                "logs",
                "worker-process.json",
            )
            manifest_path = safe_child(run.path, "manifest.json")
            has_control_state = (
                request_path.exists()
                or stdout_path.exists()
                or stderr_path.exists()
                or process_state.exists()
            )
            if has_control_state:
                if not request_path.exists():
                    raise WorkerResponseIntegrityError(
                        "Persisted worker logs are missing their bound request."
                    )
                try:
                    persisted_request = WorkerRequest.from_dict(
                        read_json_file(request_path)
                    )
                    _require_request_identity(
                        persisted_request,
                        record,
                        run,
                        effective_key,
                    )
                except Exception as error:
                    raise WorkerResponseIntegrityError(
                        "The persisted worker request failed integrity validation."
                    ) from error
                if _persisted_worker_is_active(
                    process_state,
                    record,
                    run,
                ):
                    raise WorkerRunInProgressError(
                        "The confirmed revision already has an active worker."
                    )
                if manifest_path.exists():
                    if not process_state.exists():
                        raise WorkerResponseIntegrityError(
                            "A committed run is missing its process state."
                        )
                    return _recover_response_from_manifest(
                        self.artifacts,
                        record,
                        run,
                        response_path,
                    )
                if retry + 1 >= _MAX_WORKER_ATTEMPTS:
                    return _retry_exhausted_response(
                        self.artifacts,
                        record,
                        idempotency_key,
                        elapsed_seconds=transient_elapsed_seconds,
                    )
                retry += 1
                continue
            if manifest_path.exists():
                raise WorkerResponseIntegrityError(
                    "A committed run is missing its bound worker request."
                )
            break

        request = WorkerRequest(
            session_id=record.session_id,
            revision=record.revision,
            revision_hash=record.revision_hash,
            run_id=run.run_id,
            idempotency_key=effective_key,
            deadline_at=deadline_at,
        )
        if request_path.exists():
            persisted = WorkerRequest.from_dict(read_json_file(request_path))
            if persisted != request:
                raise ValueError(
                    "the run idempotency key is already bound to another request"
                )
        else:
            atomic_write_json(request_path, request.to_dict())

        return self._launch(
            request,
            record,
            run,
            request_path,
            response_path,
            timeout,
            cancel_event,
        )

    def _launch(
        self,
        request: WorkerRequest,
        record: RevisionRecord,
        run: RunDirectory,
        request_path: Path,
        response_path: Path,
        timeout: float,
        cancel_event: threading.Event | None,
    ) -> WorkerResponse:
        stdout_path = safe_child(run.path, "logs", "worker-stdout.log")
        stderr_path = safe_child(run.path, "logs", "worker-stderr.log")
        command = [
            self.python_executable,
            "-I",
            "-B",
            "-m",
            "fem_agent.worker",
            "--workspace",
            str(self.artifacts.root),
            "--request",
            str(request_path),
            "--response",
            str(response_path),
        ]
        creationflags = (
            getattr(subprocess, "CREATE_NO_WINDOW", 0)
            if os.name == "nt"
            else 0
        )
        started = time.monotonic()
        process_state_path = safe_child(
            run.path,
            "logs",
            "worker-process.json",
        )
        process_started_at = _utc_now()
        process: subprocess.Popen[bytes] | None = None
        reason: Diagnostic | None = None
        process_may_be_active = False
        # Logs and exported artifacts are supervised as independent budgets.
        log_limit_bytes = min(
            record.spec.resource_limits.max_output_bytes,
            _FIXED_WORKER_LOG_LIMIT_BYTES,
        )
        log_capture = _BoundedWorkerLogCapture(log_limit_bytes)
        log_threads: list[threading.Thread] = []
        try:
            atomic_write_json(
                process_state_path,
                {
                    "schema_version": WORKER_SCHEMA_VERSION,
                    "session_id": record.session_id,
                    "revision": record.revision,
                    "revision_hash": record.revision_hash,
                    "run_id": run.run_id,
                    "supervisor_pid": os.getpid(),
                    "pid": None,
                    "started_at": process_started_at,
                    "deadline_at": request.deadline_at,
                },
            )
            with (
                stdout_path.open("xb", buffering=0) as stdout,
                stderr_path.open("xb", buffering=0) as stderr,
            ):
                process = subprocess.Popen(
                    command,
                    cwd=Path(__file__).resolve().parents[2],
                    env=scrub_worker_environment(os.environ),
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    shell=False,
                    creationflags=creationflags,
                    bufsize=0,
                )
                if process.stdout is None or process.stderr is None:
                    raise RuntimeError("worker log pipes were not created")
                for name, source, destination in (
                    ("stdout", process.stdout, stdout),
                    ("stderr", process.stderr, stderr),
                ):
                    thread = threading.Thread(
                        target=log_capture.copy,
                        args=(source, destination),
                        name=f"fem-worker-{name}-{run.run_id}",
                        daemon=True,
                    )
                    thread.start()
                    log_threads.append(thread)
                atomic_write_json(
                    process_state_path,
                    {
                        "schema_version": WORKER_SCHEMA_VERSION,
                        "session_id": record.session_id,
                        "revision": record.revision,
                        "revision_hash": record.revision_hash,
                        "run_id": run.run_id,
                        "supervisor_pid": os.getpid(),
                        "pid": process.pid,
                        "started_at": process_started_at,
                        "deadline_at": request.deadline_at,
                    },
                    overwrite=True,
                )
                while process.poll() is None:
                    if log_capture.exceeded:
                        reason = _worker_log_limit_diagnostic(
                            record,
                            log_limit_bytes,
                        )
                        process_may_be_active = not _terminate_process(process)
                        break
                    if log_capture.failure is not None:
                        reason = make_diagnostic(
                            DiagnosticCode.WORKER_CRASH,
                            "The local FEM worker log capture failed.",
                            source="fem.worker",
                            entity="worker-logs",
                        )
                        process_may_be_active = not _terminate_process(process)
                        break
                    export_limit_reason = _worker_export_limit_violation(
                        record,
                        safe_child(run.path, "exports"),
                    )
                    if export_limit_reason is not None:
                        reason = export_limit_reason
                        process_may_be_active = not _terminate_process(process)
                        break
                    if cancel_event is not None and cancel_event.is_set():
                        reason = make_diagnostic(
                            DiagnosticCode.OPERATION_CANCELLED,
                            "The local FEM worker was cancelled.",
                            source="fem.worker",
                        )
                        process_may_be_active = not _terminate_process(process)
                        break
                    if time.monotonic() - started >= timeout:
                        reason = make_diagnostic(
                            DiagnosticCode.WORKER_TIMEOUT,
                            (
                                "The local FEM worker exceeded the "
                                f"{timeout:g}-second limit."
                            ),
                            source="fem.worker",
                        )
                        process_may_be_active = not _terminate_process(process)
                        break
                    time.sleep(0.05)
                join_deadline = time.monotonic() + _WORKER_LOG_JOIN_SECONDS
                for thread in log_threads:
                    thread.join(
                        timeout=max(0.0, join_deadline - time.monotonic())
                    )
                if log_capture.exceeded:
                    if process.poll() is None:
                        process_may_be_active = not _terminate_process(process)
                    reason = _worker_log_limit_diagnostic(
                        record,
                        log_limit_bytes,
                    )
                else:
                    export_limit_reason = _worker_export_limit_violation(
                        record,
                        safe_child(run.path, "exports"),
                    )
                    if export_limit_reason is not None:
                        if process.poll() is None:
                            process_may_be_active = not _terminate_process(
                                process
                            )
                        reason = export_limit_reason
                if reason is None and log_capture.failure is not None:
                    reason = make_diagnostic(
                        DiagnosticCode.WORKER_CRASH,
                        "The local FEM worker log capture failed.",
                        source="fem.worker",
                        entity="worker-logs",
                    )
                elif reason is None and any(
                    thread.is_alive() for thread in log_threads
                ):
                    if process.poll() is None:
                        process_may_be_active = not _terminate_process(process)
                    reason = make_diagnostic(
                        DiagnosticCode.WORKER_CRASH,
                        "The local FEM worker log capture did not finish.",
                        source="fem.worker",
                        entity="worker-logs",
                    )
        except KeyboardInterrupt:
            reason = make_diagnostic(
                DiagnosticCode.OPERATION_CANCELLED,
                "The local FEM worker was interrupted.",
                source="fem.worker",
            )
            if process is not None:
                process_may_be_active = not _terminate_process(process)
        except OSError:
            reason = make_diagnostic(
                DiagnosticCode.WORKER_CRASH,
                "The local FEM worker process could not be started.",
                source="fem.worker",
            )
            if process is not None and process.poll() is None:
                process_may_be_active = not _terminate_process(process)
        except Exception:
            reason = make_diagnostic(
                DiagnosticCode.WORKER_CRASH,
                "The local FEM worker launch protocol failed.",
                source="fem.worker",
            )
            if process is not None and process.poll() is None:
                process_may_be_active = not _terminate_process(process)
        finally:
            if process is not None and process.poll() is None:
                process_may_be_active = not _terminate_process(process)
            join_deadline = time.monotonic() + _WORKER_LOG_JOIN_SECONDS
            for thread in log_threads:
                thread.join(
                    timeout=max(0.0, join_deadline - time.monotonic())
                )
            _close_process_pipes(process)

        supervisor_resource_violation = (
            reason is not None
            and reason.code == DiagnosticCode.RESOURCE_LIMIT.value
        )
        if supervisor_resource_violation and not process_may_be_active:
            _remove_worker_export_staging(
                safe_child(run.path, "exports")
            )
        if (
            response_path.exists()
            and not log_capture.exceeded
            and not supervisor_resource_violation
        ):
            try:
                response = load_verified_worker_response(
                    self.artifacts,
                    record,
                    run.run_id,
                )
                return response
            except Exception as error:
                reason = make_diagnostic(
                    DiagnosticCode.WORKER_CRASH,
                    f"The worker response was invalid: {type(error).__name__}.",
                    source="fem.worker",
                )
        if reason is None:
            return_code = None if process is None else process.returncode
            if return_code == _WORKER_SELF_TIMEOUT_EXIT_CODE:
                reason = make_diagnostic(
                    DiagnosticCode.WORKER_TIMEOUT,
                    "The local FEM worker reached its persisted deadline.",
                    source="fem.worker",
                )
            else:
                reason = make_diagnostic(
                    DiagnosticCode.WORKER_CRASH,
                    f"The local FEM worker exited with code {return_code}.",
                    source="fem.worker",
                )
        status = (
            RunStatus.CANCELLED
            if reason.code == DiagnosticCode.OPERATION_CANCELLED.value
            else RunStatus.FAILED
        )
        failure_run, failure_response_path = _terminal_failure_run(
            self.artifacts,
            record.session_id,
            request.idempotency_key,
            run,
            response_path,
            force_recovery=process_may_be_active,
        )
        return _write_terminal_failure(
            self.artifacts,
            record,
            failure_run,
            failure_response_path,
            status,
            reason,
            elapsed_seconds=time.monotonic() - started,
        )


def execute_inspection_request(
    workspace: str | os.PathLike[str],
    request: InspectionRequest,
) -> InspectionResponse:
    """Run local import, summary, and optional validation in the child."""

    artifacts = ArtifactStore(workspace)
    artifact = artifacts.get_artifact(
        request.spec.session_id,
        request.spec.source_artifact_id,
    )
    if artifact.kind != "input" or artifact.sha256 != request.spec.source_sha256:
        raise ValueError("inspection input identity does not match the specification")
    path = artifacts.resolve_artifact(
        request.spec.session_id,
        request.spec.source_artifact_id,
        verify=True,
    )
    imported = inspect_abaqus(
        path,
        resource_limits=request.spec.resource_limits,
    )
    summary = build_analysis_summary(
        imported,
        request.spec,
        request.revision_hash,
    )
    diagnostics = list(summary.diagnostics)
    if (
        request.validate
        and imported.model is not None
        and imported.runnable_step is not None
        and not has_errors(diagnostics)
    ):
        diagnostics.extend(
            validate_analysis(imported.model, imported.runnable_step)
        )
    return InspectionResponse(
        summary=summary,
        diagnostics=_deduplicate_diagnostics(diagnostics),
    )


def execute_result_query_request(
    workspace: str | os.PathLike[str],
    request: ResultQueryRequest,
) -> ResultSummary:
    """Query one verified solved state without repeating the FEM solve."""

    artifacts = ArtifactStore(workspace)
    revisions = RevisionStore(artifacts.root)
    confirmations = ConfirmationStore(artifacts.root, revisions)
    record = revisions.require_current(
        request.session_id,
        expected_revision=request.revision,
        expected_hash=request.revision_hash,
    )
    confirmations.require_confirmed(
        request.session_id,
        revision=request.revision,
        revision_hash=request.revision_hash,
    )
    response = load_verified_worker_response(
        artifacts,
        record,
        request.run_id,
    )
    if response.status != RunStatus.SUCCEEDED:
        raise ValueError("result queries require a successful local run")

    solution_records = tuple(
        item for item in response.artifacts if item.kind == "solution"
    )
    if len(solution_records) != 1:
        raise ValueError(
            "this run has no reusable solution state; run the analysis again "
            "before requesting new results"
        )
    solution_record = solution_records[0]
    if (
        solution_record.size_bytes
        > record.spec.resource_limits.max_output_bytes
    ):
        raise ValueError("the reusable solution state exceeds max_output_bytes")
    solution_path = artifacts.resolve_artifact(
        request.session_id,
        solution_record.artifact_id,
        verify=True,
    )
    manifest_records = tuple(
        item for item in response.artifacts if item.kind == "manifest"
    )
    if len(manifest_records) != 1:
        raise ValueError("the completed run has no verified manifest")
    manifest_path = artifacts.resolve_artifact(
        request.session_id,
        manifest_records[0].artifact_id,
        verify=True,
    )
    manifest = RunManifest.from_dict(read_json_file(manifest_path))
    expected_model_sha256 = manifest.tool_parameters.get(
        "solution_model_sha256"
    )
    if not isinstance(expected_model_sha256, str):
        raise ValueError(
            "this run predates reusable model fingerprints; run the "
            "analysis again before requesting new results"
        )
    _sha256(expected_model_sha256, "solution_model_sha256")

    source = artifacts.resolve_artifact(
        request.session_id,
        record.spec.source_artifact_id,
        verify=True,
    )
    imported = inspect_abaqus(
        source,
        resource_limits=record.spec.resource_limits,
    )
    if (
        has_errors(imported.diagnostics)
        or imported.model is None
        or imported.runnable_step is None
    ):
        raise ValueError("the completed run model could not be reconstructed")
    if str(imported.runnable_step.name) != record.spec.analysis_step:
        raise ValueError("the completed run analysis step no longer matches")
    if record.spec.unit_context is None:
        raise ValueError("the completed run has no declared unit context")

    materials.apply_sections(imported.model)
    if _solution_model_sha256(imported.model) != expected_model_sha256:
        raise ValueError(
            "the reconstructed model layout does not match the saved "
            "solution fingerprint; run the analysis again"
        )
    result = _result_from_solution(
        imported.model,
        imported.runnable_step,
        solution_path,
    )
    return query_results(
        result,
        request.queries,
        run_id=request.run_id,
        unit_context=record.spec.unit_context,
    )


def execute_worker_request(
    workspace: str | os.PathLike[str],
    request: WorkerRequest,
) -> WorkerResponse:
    """Execute the complete deterministic pipeline inside the child process."""

    started_wall = _utc_now()
    started = time.monotonic()
    artifacts = ArtifactStore(workspace)
    revisions = RevisionStore(artifacts.root)
    confirmations = ConfirmationStore(artifacts.root, revisions)
    run = artifacts.run_directory(request.session_id, request.run_id)
    record = revisions.require_current(
        request.session_id,
        expected_revision=request.revision,
        expected_hash=request.revision_hash,
    )
    confirmations.require_confirmed(
        request.session_id,
        revision=request.revision,
        revision_hash=request.revision_hash,
    )

    diagnostics: list[Diagnostic] = []
    validation_diagnostics: tuple[Diagnostic, ...] = ()
    result_summary: ResultSummary | None = None
    registered: list[ArtifactRecord] = []
    durations: dict[str, float] = {}
    status = RunStatus.RUNNING
    source = artifacts.get_artifact(
        request.session_id,
        record.spec.source_artifact_id,
    )
    if source.sha256 != record.spec.source_sha256:
        diagnostics.append(
            make_diagnostic(
                DiagnosticCode.INVALID_INPUT,
                "The revision source hash does not match its artifact record.",
                source="fem.worker",
            )
        )
    input_path: Path | None = None
    if not diagnostics:
        try:
            input_path = artifacts.resolve_artifact(
                request.session_id,
                record.spec.source_artifact_id,
                verify=True,
            )
        except Exception:
            diagnostics.append(
                make_diagnostic(
                    DiagnosticCode.INVALID_INPUT,
                    "The immutable input artifact failed its integrity check.",
                    source="fem.worker",
                )
            )

    import_result = None
    if input_path is not None and not diagnostics:
        phase = time.monotonic()
        import_result = inspect_abaqus(
            input_path,
            resource_limits=record.spec.resource_limits,
        )
        durations["import"] = time.monotonic() - phase
        diagnostics.extend(import_result.diagnostics)

    model = getattr(import_result, "model", None)
    step = getattr(import_result, "runnable_step", None)
    if not has_errors(diagnostics) and model is not None and step is not None:
        diagnostics.extend(_resource_diagnostics(record, model, source.size_bytes))
        if str(step.name) != record.spec.analysis_step:
            diagnostics.append(
                make_diagnostic(
                    DiagnosticCode.STALE_REVISION,
                    "The confirmed analysis step does not match the imported model.",
                    source="fem.worker",
                    step=str(step.name),
                )
            )

    if not has_errors(diagnostics) and model is not None and step is not None:
        phase = time.monotonic()
        validation_diagnostics = validate_analysis(model, step)
        durations["validation"] = time.monotonic() - phase
        diagnostics.extend(validation_diagnostics)

    solve_outcome = None
    solution_model_sha256: str | None = None
    if not has_errors(diagnostics) and model is not None and step is not None:
        solve_outcome = solve_analysis(model, step)
        durations["solve"] = solve_outcome.elapsed_seconds
        diagnostics.extend(solve_outcome.diagnostics)

    if solve_outcome is not None and solve_outcome.result is not None:
        try:
            solution_path = safe_child(run.path, "solution.npy")
            solution_payload = _solution_payload(solve_outcome.result)
            model_sha256 = _solution_model_sha256(
                solve_outcome.result.model
            )
            if (
                len(solution_payload)
                > record.spec.resource_limits.max_output_bytes
            ):
                diagnostics.append(
                    make_diagnostic(
                        DiagnosticCode.RESOURCE_LIMIT,
                        (
                            "The reusable solution state exceeds "
                            "max_output_bytes."
                        ),
                        source="fem.worker",
                        entity="solution",
                    )
                )
            else:
                atomic_write_bytes(solution_path, solution_payload)
                solution_artifact = artifacts.register_run_artifact(
                    request.session_id,
                    run.run_id,
                    solution_path,
                    kind="solution",
                )
                registered.append(solution_artifact)
                solution_model_sha256 = model_sha256
        except Exception:
            diagnostics.append(
                make_diagnostic(
                    DiagnosticCode.WORKER_CRASH,
                    (
                        "The solved state could not be persisted for "
                        "post-solve result queries."
                    ),
                    source="fem.worker",
                    entity="solution",
                )
            )

    if (
        solve_outcome is not None
        and solve_outcome.result is not None
        and not has_errors(diagnostics)
        and record.spec.requested_queries
    ):
        phase = time.monotonic()
        result_summary = query_results(
            solve_outcome.result,
            record.spec.requested_queries,
            run_id=run.run_id,
            unit_context=record.spec.unit_context,
        )
        durations["queries"] = time.monotonic() - phase
        result_path = safe_child(run.path, "result-summary.json")
        atomic_write_json(result_path, result_summary.to_dict())
        registered.append(
            artifacts.register_run_artifact(
                request.session_id,
                run.run_id,
                result_path,
                kind="result_summary",
            )
        )

    if (
        solve_outcome is not None
        and solve_outcome.result is not None
        and not has_errors(diagnostics)
        and record.spec.export_formats
    ):
        phase = time.monotonic()
        export_outcome = export_results(
            solve_outcome.result,
            record.spec.export_formats,
            run_id=run.run_id,
            run_directory=run.path,
            exports_directory=safe_child(run.path, "exports"),
            resource_limits=record.spec.resource_limits,
        )
        durations["exports"] = time.monotonic() - phase
        diagnostics.extend(export_outcome.diagnostics)
        for exported in export_outcome.artifacts:
            exported_path = safe_child(
                run.path,
                *Path(exported.display_path).parts,
            )
            registered.append(
                artifacts.register_run_artifact(
                    request.session_id,
                    run.run_id,
                    exported_path,
                    kind=exported.kind,
                )
            )

    status = RunStatus.FAILED if has_errors(diagnostics) else RunStatus.SUCCEEDED
    durations["total"] = time.monotonic() - started
    completed_wall = _utc_now()
    registered.extend(
        _write_numerical_records(
            artifacts,
            record,
            run,
            status=status,
            diagnostics=tuple(diagnostics),
            validation_diagnostics=validation_diagnostics,
            result_summary=result_summary,
            artifact_records=tuple(registered),
            solution_model_sha256=solution_model_sha256,
            timestamps={
                "started_at": started_wall,
                "completed_at": completed_wall,
            },
            durations=durations,
        )
    )
    return WorkerResponse(
        session_id=record.session_id,
        revision=record.revision,
        revision_hash=record.revision_hash,
        run_id=run.run_id,
        status=status,
        result_summary=result_summary,
        artifacts=tuple(registered),
        diagnostics=tuple(diagnostics),
        elapsed_seconds=durations["total"],
    )


def _solution_payload(result: ModelResult) -> bytes:
    displacement = np.asarray(result.U, dtype=np.float64)
    reactions = np.asarray(result.reactions, dtype=np.float64)
    if (
        displacement.ndim != 1
        or reactions.ndim != 1
        or displacement.shape != reactions.shape
        or not np.all(np.isfinite(displacement))
        or not np.all(np.isfinite(reactions))
    ):
        raise ValueError("the solved vectors are not reusable")
    state = np.stack((displacement, reactions), axis=0)
    buffer = io.BytesIO()
    np.save(buffer, state, allow_pickle=False)
    return buffer.getvalue()


def _result_from_solution(
    model: Any,
    step: Any,
    solution_path: Path,
) -> ModelResult:
    state = np.load(solution_path, allow_pickle=False)
    expected_shape = (2, int(model.mesh.num_dofs))
    if not isinstance(state, np.ndarray) or state.shape != expected_shape:
        raise ValueError("the reusable solution state has an invalid shape")
    if not np.issubdtype(state.dtype, np.number):
        raise ValueError("the reusable solution state is not numeric")
    normalized = np.asarray(state, dtype=np.float64)
    if not np.all(np.isfinite(normalized)):
        raise ValueError("the reusable solution state is not finite")
    return ModelResult(
        model,
        step,
        normalized[0],
        normalized[1],
    )


def _solution_model_sha256(model: Any) -> str:
    """Bind saved vectors to the reconstructed DOF and result topology."""

    mesh = model.mesh
    payload = {
        "schema_version": RESULT_QUERY_SCHEMA_VERSION,
        "mesh": {
            "dofs_per_node": int(mesh.dofs_per_node),
            "nodes": _fingerprint_value(tuple(mesh.nodes)),
            "elements": _fingerprint_value(tuple(mesh.elements)),
        },
        "node_sets": _fingerprint_value(model.node_sets),
        "element_sets": _fingerprint_value(model.element_sets),
        "edges": _fingerprint_value(model.edges),
        "surfaces": _fingerprint_value(model.surfaces),
        "materials": _fingerprint_value(model.materials),
        "sections": _fingerprint_value(tuple(model.sections)),
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _fingerprint_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return _fingerprint_value(value.item())
    if value is None or isinstance(value, (bool, str, int)):
        return value
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError("model fingerprint values must be finite")
        return {"float_hex": value.hex()}
    if isinstance(value, np.ndarray):
        return {
            "array_dtype": str(value.dtype),
            "array_shape": list(value.shape),
            "array_values": _fingerprint_value(value.tolist()),
        }
    if is_dataclass(value) and not isinstance(value, type):
        return {
            "dataclass": (
                f"{type(value).__module__}.{type(value).__qualname__}"
            ),
            "fields": {
                item.name: _fingerprint_value(getattr(value, item.name))
                for item in fields(value)
            },
        }
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("model fingerprint mapping keys must be strings")
        return {
            key: _fingerprint_value(value[key])
            for key in sorted(value)
        }
    if isinstance(value, (list, tuple)):
        return [_fingerprint_value(item) for item in value]
    raise TypeError(
        "model fingerprint contains unsupported value type "
        f"{type(value).__module__}.{type(value).__qualname__}"
    )


def scrub_worker_environment(
    environment: Mapping[str, str],
) -> dict[str, str]:
    """Return an environment without recognizable provider credentials."""

    clean: dict[str, str] = {}
    for key, value in environment.items():
        normalized = str(key).upper()
        if normalized not in _ALLOWED_WORKER_ENVIRONMENT:
            continue
        if any(fragment in normalized for fragment in _SENSITIVE_ENVIRONMENT_FRAGMENTS):
            continue
        clean[str(key)] = str(value)
    clean["FEM_AGENT_WORKER"] = "1"
    return clean


def _write_numerical_records(
    store: ArtifactStore,
    record: RevisionRecord,
    run: RunDirectory,
    *,
    status: RunStatus,
    diagnostics: tuple[Diagnostic, ...],
    validation_diagnostics: tuple[Diagnostic, ...],
    result_summary: ResultSummary | None,
    artifact_records: tuple[ArtifactRecord, ...],
    timestamps: Mapping[str, str],
    durations: Mapping[str, float],
    solution_model_sha256: str | None = None,
) -> tuple[ArtifactRecord, ...]:
    diagnostics_path = safe_child(run.path, "diagnostics.json")
    atomic_write_json(
        diagnostics_path,
        {
            "schema_version": WORKER_SCHEMA_VERSION,
            "diagnostics": [item.to_dict() for item in diagnostics],
        },
    )
    diagnostics_artifact = store.register_run_artifact(
        record.session_id,
        run.run_id,
        diagnostics_path,
        kind="diagnostics",
    )
    result_artifact_id = next(
        (
            item.artifact_id
            for item in artifact_records
            if item.kind == "result_summary"
        ),
        None,
    )
    tool_parameters = {
        "requested_queries": [
            item.to_dict() for item in record.spec.requested_queries
        ],
        "export_formats": [
            item.value for item in record.spec.export_formats
        ],
        "resource_limits": record.spec.resource_limits.to_dict(),
    }
    if solution_model_sha256 is not None:
        _sha256(solution_model_sha256, "solution_model_sha256")
        tool_parameters["solution_model_sha256"] = (
            solution_model_sha256
        )
    manifest = RunManifest(
        session_id=record.session_id,
        revision=record.revision,
        revision_hash=record.revision_hash,
        run_id=run.run_id,
        status=status,
        source_sha256=record.spec.source_sha256,
        repository_commit=_repository_commit(),
        runtime_versions=_runtime_versions(),
        unit_context=record.spec.unit_context,
        analysis_step=record.spec.analysis_step,
        tool_parameters=tool_parameters,
        validation_diagnostics=validation_diagnostics,
        result_summary_artifact_id=result_artifact_id,
        artifacts=(*artifact_records, diagnostics_artifact),
        timestamps=dict(timestamps),
        durations_seconds=dict(durations),
        diagnostics=diagnostics,
    )
    manifest_path = safe_child(run.path, "manifest.json")
    atomic_write_json(manifest_path, manifest.to_dict())
    manifest_artifact = store.register_run_artifact(
        record.session_id,
        run.run_id,
        manifest_path,
        kind="manifest",
    )
    return diagnostics_artifact, manifest_artifact


def _retry_exhausted_response(
    artifacts: ArtifactStore,
    record: RevisionRecord,
    idempotency_key: str,
    *,
    elapsed_seconds: float,
) -> WorkerResponse:
    terminal_key = _worker_attempt_key(
        idempotency_key,
        _MAX_WORKER_ATTEMPTS,
    )
    run = artifacts.create_run(
        record.session_id,
        idempotency_key=terminal_key,
    )
    response_path = safe_child(
        run.path,
        "logs",
        "worker-response.json",
    )
    if response_path.exists():
        try:
            response = load_verified_worker_response(
                artifacts,
                record,
                run.run_id,
            )
            _require_retry_exhausted_semantics(response)
        except Exception as error:
            raise WorkerResponseIntegrityError(
                "The persisted retry-exhaustion response failed integrity "
                "validation."
            ) from error
        return response
    manifest_path = safe_child(run.path, "manifest.json")
    if manifest_path.exists():
        response = _recover_response_from_manifest(
            artifacts,
            record,
            run,
            response_path,
        )
        try:
            _require_retry_exhausted_semantics(response)
        except Exception as error:
            raise WorkerResponseIntegrityError(
                "The recovered retry-exhaustion response has invalid "
                "semantics."
            ) from error
        return response
    if _run_has_published_output(run, response_path):
        raise WorkerResponseIntegrityError(
            "The retry-exhaustion run contains incomplete published output."
        )
    diagnostic = make_diagnostic(
        _RETRY_EXHAUSTED_DIAGNOSTIC_CODE,
        (
            "The local FEM worker exhausted its "
            f"{_MAX_WORKER_ATTEMPTS}-attempt retry budget after repeated "
            "transient or incomplete failures."
        ),
        source="fem.worker",
        entity="worker-retry-budget",
        remediation=(
            "Inspect the persisted attempt diagnostics and start a new "
            "confirmed revision before running again."
        ),
    )
    return _write_terminal_failure(
        artifacts,
        record,
        run,
        response_path,
        RunStatus.FAILED,
        diagnostic,
        elapsed_seconds=elapsed_seconds,
    )


def _require_retry_exhausted_semantics(response: WorkerResponse) -> None:
    if (
        response.status != RunStatus.FAILED
        or len(response.diagnostics) != 1
        or response.diagnostics[0].code
        != _RETRY_EXHAUSTED_DIAGNOSTIC_CODE
    ):
        raise ValueError(
            "persisted retry-exhaustion response has invalid semantics"
        )


def _terminal_failure_run(
    artifacts: ArtifactStore,
    session_id: str,
    idempotency_key: str,
    original_run: RunDirectory,
    original_response_path: Path,
    *,
    force_recovery: bool = False,
) -> tuple[RunDirectory, Path]:
    if (
        not force_recovery
        and not _run_has_published_output(original_run, original_response_path)
    ):
        return original_run, original_response_path
    base = idempotency_key[:96]
    attempt = 1
    while True:
        recovery = artifacts.create_run(
            session_id,
            idempotency_key=f"{base}_terminal_{attempt}",
        )
        response_path = safe_child(
            recovery.path,
            "logs",
            "worker-response.json",
        )
        if not _run_has_published_output(recovery, response_path):
            return recovery, response_path
        attempt += 1


def _run_has_published_output(
    run: RunDirectory,
    response_path: Path,
) -> bool:
    for name in (
        "diagnostics.json",
        "manifest.json",
        "result-summary.json",
        "solution.npy",
    ):
        if safe_child(run.path, name).exists():
            return True
    if response_path.exists():
        return True
    exports = safe_child(run.path, "exports")
    return any(exports.iterdir())


def _write_terminal_failure(
    artifacts: ArtifactStore,
    record: RevisionRecord,
    run: RunDirectory,
    response_path: Path,
    status: RunStatus,
    diagnostic: Diagnostic,
    *,
    elapsed_seconds: float,
) -> WorkerResponse:
    started = _utc_now()
    written = _write_numerical_records(
        artifacts,
        record,
        run,
        status=status,
        diagnostics=(diagnostic,),
        validation_diagnostics=(),
        result_summary=None,
        artifact_records=(),
        timestamps={"started_at": started, "completed_at": _utc_now()},
        durations={"total": elapsed_seconds},
    )
    response = WorkerResponse(
        session_id=record.session_id,
        revision=record.revision,
        revision_hash=record.revision_hash,
        run_id=run.run_id,
        status=status,
        result_summary=None,
        artifacts=written,
        diagnostics=(diagnostic,),
        elapsed_seconds=elapsed_seconds,
    )
    atomic_write_json(response_path, response.to_dict())
    return response


def _resource_diagnostics(
    record: RevisionRecord,
    model: Any,
    source_size: int,
) -> tuple[Diagnostic, ...]:
    limits = record.spec.resource_limits
    values = {
        "input bytes": (source_size, limits.max_input_bytes),
        "nodes": (int(model.mesh.num_nodes), limits.max_nodes),
        "elements": (int(model.mesh.num_elements), limits.max_elements),
        "DOFs": (int(model.mesh.num_dofs), limits.max_dofs),
    }
    return tuple(
        make_diagnostic(
            DiagnosticCode.RESOURCE_LIMIT,
            f"The model has {actual} {name}; the configured limit is {limit}.",
            source="fem.worker",
        )
        for name, (actual, limit) in values.items()
        if actual > limit
    )


def _worker_log_limit_diagnostic(
    record: RevisionRecord,
    effective_limit_bytes: int,
) -> Diagnostic:
    configured_limit = record.spec.resource_limits.max_output_bytes
    return make_diagnostic(
        DiagnosticCode.RESOURCE_LIMIT,
        (
            "The local FEM worker exceeded its combined stdout/stderr log "
            f"limit of {effective_limit_bytes} bytes "
            f"(spec.max_output_bytes={configured_limit}; "
            f"fixed_log_limit={_FIXED_WORKER_LOG_LIMIT_BYTES})."
        ),
        source="fem.worker",
        entity="worker-logs",
        remediation=(
            "Reduce worker logging or raise spec.max_output_bytes without "
            "exceeding the fixed log safety limit."
        ),
    )


def _worker_export_limit_violation(
    record: RevisionRecord,
    exports_directory: Path,
) -> Diagnostic | None:
    limits = record.spec.resource_limits
    file_count = 0
    total_bytes = 0
    pending = [exports_directory]
    while pending:
        directory = pending.pop()
        try:
            with os.scandir(directory) as iterator:
                entries = tuple(iterator)
        except FileNotFoundError:
            continue
        for entry in entries:
            try:
                if entry.is_dir(follow_symlinks=False):
                    pending.append(Path(entry.path))
                    continue
                file_count += 1
                total_bytes += entry.stat(follow_symlinks=False).st_size
            except FileNotFoundError:
                continue
            if (
                file_count > limits.max_output_files
                or total_bytes > limits.max_output_bytes
            ):
                return make_diagnostic(
                    DiagnosticCode.RESOURCE_LIMIT,
                    (
                        "The local FEM worker exceeded its export quota while "
                        "generating output "
                        f"(files={file_count}, bytes={total_bytes}; "
                        f"max_output_files={limits.max_output_files}, "
                        f"max_output_bytes={limits.max_output_bytes})."
                    ),
                    source="fem.worker",
                    entity="worker-exports",
                    remediation=(
                        "Reduce requested export output or raise the revision "
                        "export limits before confirming a new run."
                    ),
                )
    return None


def _remove_worker_export_staging(exports_directory: Path) -> None:
    """Discard only exporter-owned temporary entries after a stopped worker."""

    try:
        with os.scandir(exports_directory) as iterator:
            entries = tuple(iterator)
    except FileNotFoundError:
        return
    for entry in entries:
        if not entry.name.startswith(".fem-agent-export-"):
            continue
        candidate = Path(entry.path)
        try:
            if entry.is_dir(follow_symlinks=False):
                shutil.rmtree(candidate)
            else:
                candidate.unlink()
        except FileNotFoundError:
            continue
        except OSError:
            # A preserved staging entry forces the terminal response into an
            # isolated recovery run instead of risking an in-place commit.
            continue


def _deduplicate_diagnostics(
    diagnostics: list[Diagnostic],
) -> tuple[Diagnostic, ...]:
    unique: list[Diagnostic] = []
    seen: set[tuple[str, str, str | None, str | None]] = set()
    for diagnostic in diagnostics:
        key = (
            diagnostic.code,
            diagnostic.message,
            diagnostic.entity,
            diagnostic.step,
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(diagnostic)
    return tuple(unique)


@contextmanager
def _exclusive_worker_claim(
    artifacts: ArtifactStore,
    session_id: str,
    idempotency_key: str,
) -> Iterator[None]:
    """Hold a cross-process claim for one idempotent worker family."""

    validate_identifier(session_id, "session_id")
    validate_identifier(idempotency_key, "idempotency_key")
    session = artifacts.session_path(session_id)
    digest = hashlib.sha256(idempotency_key.encode("utf-8")).hexdigest()
    claim_path = safe_child(session, f".worker-claim-{digest}.lock")
    with claim_path.open("a+b") as stream:
        stream.seek(0, os.SEEK_END)
        if stream.tell() == 0:
            stream.write(b"\0")
            stream.flush()
            os.fsync(stream.fileno())
        stream.seek(0)
        try:
            _lock_claim_stream(stream)
        except OSError as error:
            raise WorkerRunInProgressError(
                "The confirmed revision already has an active worker."
            ) from error
        try:
            yield
        finally:
            try:
                _unlock_claim_stream(stream)
            except OSError:
                # Closing the stream still releases the OS claim.  An unlock
                # failure must not replace the completed run result or its
                # original exception.
                pass


def _lock_claim_stream(stream: Any) -> None:
    if os.name == "nt":
        import msvcrt

        msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        return
    import fcntl

    fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def _unlock_claim_stream(stream: Any) -> None:
    stream.seek(0)
    if os.name == "nt":
        import msvcrt

        msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        return
    import fcntl

    fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _worker_attempt_key(idempotency_key: str, retry: int) -> str:
    base = validate_identifier(idempotency_key, "idempotency_key")
    if isinstance(retry, bool) or not isinstance(retry, int) or retry < 0:
        raise ValueError("retry must be a nonnegative integer")
    if retry == 0:
        return base
    suffix = f"_retry_{retry}"
    if len(base) + len(suffix) <= 128:
        return f"{base}{suffix}"
    digest = hashlib.sha256(base.encode("utf-8")).hexdigest()[:16]
    prefix_length = 128 - len(suffix) - len(digest) - 1
    return f"{base[:prefix_length]}_{digest}{suffix}"


def _deadline_after(timeout_seconds: float) -> str:
    deadline = datetime.now(timezone.utc) + timedelta(seconds=timeout_seconds)
    return deadline.isoformat().replace("+00:00", "Z")


def _is_transient_worker_failure(response: WorkerResponse) -> bool:
    if response.status not in {RunStatus.FAILED, RunStatus.CANCELLED}:
        return False
    error_codes = {
        diagnostic.code
        for diagnostic in response.diagnostics
        if diagnostic.severity == DiagnosticSeverity.ERROR
    }
    return bool(error_codes) and error_codes <= _TRANSIENT_WORKER_DIAGNOSTIC_CODES


def _persisted_worker_is_active(
    state_path: Path,
    record: RevisionRecord,
    run: RunDirectory,
) -> bool:
    if not state_path.exists():
        return False
    try:
        state = read_json_file(state_path)
        required = {
            "schema_version",
            "session_id",
            "revision",
            "revision_hash",
            "run_id",
            "supervisor_pid",
            "pid",
            "started_at",
            "deadline_at",
        }
        if set(state) != required:
            raise ValueError("worker process state has invalid fields")
        if (
            state["schema_version"] != WORKER_SCHEMA_VERSION
            or state["session_id"] != record.session_id
            or state["revision"] != record.revision
            or state["revision_hash"] != record.revision_hash
            or state["run_id"] != run.run_id
        ):
            raise ValueError("worker process state identity mismatch")
        supervisor_pid = state["supervisor_pid"]
        if (
            isinstance(supervisor_pid, bool)
            or not isinstance(supervisor_pid, int)
            or supervisor_pid <= 0
        ):
            raise ValueError("worker supervisor PID is invalid")
        pid = state["pid"]
        if pid is not None and (
            isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0
        ):
            raise ValueError("worker child PID is invalid")
        _timestamp(state["started_at"], "started_at")
        deadline = _timestamp(state["deadline_at"], "deadline_at")
        if pid is not None:
            return _process_is_active(pid) or _process_is_active(supervisor_pid)
        if _process_is_active(supervisor_pid):
            return True
        reserved_until = deadline + timedelta(
            seconds=_WORKER_DEADLINE_GRACE_SECONDS
        )
        return datetime.now(timezone.utc) <= reserved_until
    except WorkerResponseIntegrityError:
        raise
    except Exception as error:
        raise WorkerResponseIntegrityError(
            "The persisted worker process state failed integrity validation."
        ) from error


def _process_is_active(pid: int) -> bool:
    if os.name == "nt":
        try:
            import ctypes
            from ctypes import wintypes

            process_query_limited_information = 0x1000
            still_active = 259
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            handle = kernel32.OpenProcess(
                process_query_limited_information,
                False,
                pid,
            )
            if not handle:
                return False
            try:
                exit_code = wintypes.DWORD()
                if not kernel32.GetExitCodeProcess(
                    handle,
                    ctypes.byref(exit_code),
                ):
                    return False
                return exit_code.value == still_active
            finally:
                kernel32.CloseHandle(handle)
        except Exception:
            return False
    try:
        os.kill(pid, 0)
    except (OSError, ValueError):
        return False
    return True


def _repository_commit() -> str | None:
    project_root = Path(__file__).resolve().parents[2]
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
            shell=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = completed.stdout.strip()
    if completed.returncode == 0 and len(value) == 40:
        return value
    return None


def _runtime_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
    }


def _terminate_process(process: subprocess.Popen[bytes]) -> bool:
    if process.poll() is not None:
        return True
    try:
        process.terminate()
    except OSError:
        return process.poll() is not None
    try:
        process.wait(timeout=3)
        return True
    except (OSError, subprocess.TimeoutExpired):
        pass
    try:
        process.kill()
    except OSError:
        return process.poll() is not None
    try:
        process.wait(timeout=3)
    except (OSError, subprocess.TimeoutExpired):
        return process.poll() is not None
    return True


def _close_process_pipes(
    process: subprocess.Popen[bytes] | None,
) -> None:
    """Close parent-owned subprocess pipe objects after capture threads stop."""
    if process is None:
        return
    for name in ("stdin", "stdout", "stderr"):
        stream = getattr(process, name, None)
        if stream is None:
            continue
        try:
            stream.close()
        except OSError:
            pass


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _start_worker_deadline_watchdog(
    deadline_at: str | None,
    *,
    exit_process: Callable[[int], None] = os._exit,
) -> threading.Timer | None:
    if deadline_at is None:
        return None
    deadline = _timestamp(deadline_at, "deadline_at")
    delay = max(
        0.0,
        (deadline - datetime.now(timezone.utc)).total_seconds(),
    )
    watchdog = threading.Timer(
        delay,
        exit_process,
        args=(_WORKER_SELF_TIMEOUT_EXIT_CODE,),
    )
    watchdog.daemon = True
    watchdog.start()
    return watchdog


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--request")
    parser.add_argument("--response")
    parser.add_argument("--inspect-stdin", action="store_true")
    parser.add_argument("--query-stdin", action="store_true")
    args = parser.parse_args(argv)
    if args.inspect_stdin or args.query_stdin:
        if args.inspect_stdin and args.query_stdin:
            parser.error(
                "--inspect-stdin and --query-stdin are mutually exclusive"
            )
        if args.request is not None or args.response is not None:
            parser.error(
                "stdin control modes cannot be combined with control files"
            )
    elif args.request is None or args.response is None:
        parser.error("--request and --response are required for a solve worker")
    return args


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    workspace = Path(args.workspace).resolve(strict=True)
    if args.inspect_stdin:
        request = InspectionRequest.from_dict(
            _mapping(
                _read_control_json(
                    sys.stdin.buffer,
                    description="inspection request",
                ),
                "inspection request",
            )
        )
        response = execute_inspection_request(workspace, request)
        _write_control_json(
            sys.stdout.buffer,
            response.to_dict(),
        )
        return 0
    if args.query_stdin:
        request = ResultQueryRequest.from_dict(
            _mapping(
                _read_control_json(
                    sys.stdin.buffer,
                    description="result query request",
                ),
                "result query request",
            )
        )
        summary = execute_result_query_request(workspace, request)
        _write_control_json(
            sys.stdout.buffer,
            summary.to_dict(),
        )
        return 0
    request_path = Path(args.request).resolve(strict=True)
    response_path = Path(args.response).resolve(strict=False)
    if not request_path.is_relative_to(workspace):
        raise ValueError("worker request path must remain inside the workspace")
    if request_path.is_symlink():
        raise ValueError("worker request path must not be a symbolic link")
    request = WorkerRequest.from_dict(read_json_file(request_path))
    store = ArtifactStore(workspace)
    run = store.run_directory(request.session_id, request.run_id)
    expected_request = safe_child(run.path, "logs", "worker-request.json")
    expected_response = safe_child(run.path, "logs", "worker-response.json")
    if request_path != expected_request or response_path != expected_response:
        raise ValueError("worker control paths do not match the requested run")
    watchdog = _start_worker_deadline_watchdog(request.deadline_at)
    try:
        response = execute_worker_request(workspace, request)
        atomic_write_json(response_path, response.to_dict())
    finally:
        if watchdog is not None:
            watchdog.cancel()
    return 0 if response.status == RunStatus.SUCCEEDED else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "InspectionRequest",
    "InspectionResponse",
    "InspectionWorkerError",
    "IsolatedFEMResultQuerier",
    "IsolatedFEMInspector",
    "IsolatedFEMWorker",
    "ResultQueryRequest",
    "ResultQueryWorkerError",
    "WorkerRequest",
    "WorkerResponse",
    "WorkerResponseIntegrityError",
    "WorkerRunInProgressError",
    "execute_inspection_request",
    "execute_result_query_request",
    "execute_worker_request",
    "load_verified_worker_response",
    "scrub_worker_environment",
]
