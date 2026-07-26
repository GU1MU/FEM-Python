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
from dataclasses import dataclass, fields, is_dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Callable, Iterator, Mapping

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
    AnalysisSummary,
    ArtifactRecord,
    Diagnostic,
    DiagnosticSeverity,
    ImportAnalysisSpec,
    ResultQuery,
    ResultSummary,
    RunManifest,
    RunStatus,
)
from .state import RevisionRecord, RevisionStore, hash_revision_spec
from .summaries import build_analysis_summary
from .tools.exports import export_results
from .tools.importing import inspect_abaqus
from .tools.results import query_results
from .tools.solving import solve_analysis
from .tools.validation import validate_analysis


WORKER_SCHEMA_VERSION = 1
INSPECTION_SCHEMA_VERSION = 1
RESULT_QUERY_SCHEMA_VERSION = 1
_WORKER_SELF_TIMEOUT_EXIT_CODE = 124
_WORKER_DEADLINE_GRACE_SECONDS = 2.0
_MAX_WORKER_ATTEMPTS = 8
_FIXED_WORKER_LOG_LIMIT_BYTES = 8 * 1024 * 1024
_FIXED_INSPECTION_OUTPUT_LIMIT_BYTES = 2 * 1024 * 1024
_FIXED_CONTROL_PAYLOAD_LIMIT_BYTES = 1024 * 1024
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


def _encode_control_json(value: Mapping[str, Any]) -> bytes:
    """Encode a child-process control message independently of the OS locale."""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _read_control_json(stream: BinaryIO, *, description: str) -> Any:
    """Read one bounded UTF-8 JSON message from a binary control stream."""

    payload = stream.read(_FIXED_CONTROL_PAYLOAD_LIMIT_BYTES + 1)
    if len(payload) > _FIXED_CONTROL_PAYLOAD_LIMIT_BYTES:
        raise ValueError(f"{description} exceeds the control payload limit")
    try:
        return json.loads(payload.decode("utf-8"))
    except UnicodeDecodeError as exc:
        raise ValueError(f"{description} must be valid UTF-8") from exc


def _write_control_json(stream: BinaryIO, value: Mapping[str, Any]) -> None:
    """Write one UTF-8 JSON message to a binary control stream."""

    stream.write(_encode_control_json(value))
    stream.flush()


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


@dataclass(frozen=True)
class WorkerRequest:
    """Serializable launch request containing opaque identifiers only."""

    session_id: str
    revision: int
    revision_hash: str
    run_id: str
    idempotency_key: str
    deadline_at: str

    def __post_init__(self) -> None:
        validate_identifier(self.session_id, "session_id")
        if isinstance(self.revision, bool) or not isinstance(self.revision, int):
            raise ValueError("revision must be an integer")
        if self.revision <= 0:
            raise ValueError("revision must be positive")
        _sha256(self.revision_hash, "revision_hash")
        validate_identifier(self.run_id, "run_id")
        validate_identifier(self.idempotency_key, "idempotency_key")
        _timestamp(self.deadline_at, "deadline_at")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": WORKER_SCHEMA_VERSION,
            "session_id": self.session_id,
            "revision": self.revision,
            "revision_hash": self.revision_hash,
            "run_id": self.run_id,
            "idempotency_key": self.idempotency_key,
            "deadline_at": self.deadline_at,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "WorkerRequest":
        required = {
            "schema_version",
            "session_id",
            "revision",
            "revision_hash",
            "run_id",
            "idempotency_key",
            "deadline_at",
        }
        if set(value) != required:
            raise ValueError("worker request has invalid fields")
        if value["schema_version"] != WORKER_SCHEMA_VERSION:
            raise ValueError("worker request has an unsupported version")
        return cls(
            session_id=value["session_id"],
            revision=value["revision"],
            revision_hash=value["revision_hash"],
            run_id=value["run_id"],
            idempotency_key=value["idempotency_key"],
            deadline_at=value["deadline_at"],
        )


@dataclass(frozen=True)
class InspectionRequest:
    """Opaque, serializable request for isolated import and summary work."""

    spec: ImportAnalysisSpec
    revision_hash: str
    validate: bool = False

    def __post_init__(self) -> None:
        _sha256(self.revision_hash, "revision_hash")
        if hash_revision_spec(self.spec) != self.revision_hash:
            raise ValueError("inspection revision hash does not match its specification")
        if not isinstance(self.validate, bool):
            raise ValueError("validate must be a boolean")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": INSPECTION_SCHEMA_VERSION,
            "spec": self.spec.to_dict(),
            "revision_hash": self.revision_hash,
            "validate": self.validate,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "InspectionRequest":
        required = {
            "schema_version",
            "spec",
            "revision_hash",
            "validate",
        }
        if set(value) != required:
            raise ValueError("inspection request has invalid fields")
        if value["schema_version"] != INSPECTION_SCHEMA_VERSION:
            raise ValueError("inspection request has an unsupported version")
        return cls(
            spec=ImportAnalysisSpec.from_dict(_mapping(value["spec"], "spec")),
            revision_hash=value["revision_hash"],
            validate=value["validate"],
        )


@dataclass(frozen=True)
class InspectionResponse:
    """Provider-safe output from the isolated import process."""

    summary: AnalysisSummary
    diagnostics: tuple[Diagnostic, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "diagnostics", tuple(self.diagnostics))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": INSPECTION_SCHEMA_VERSION,
            "summary": self.summary.to_dict(),
            "diagnostics": [item.to_dict() for item in self.diagnostics],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "InspectionResponse":
        required = {"schema_version", "summary", "diagnostics"}
        if set(value) != required:
            raise ValueError("inspection response has invalid fields")
        if value["schema_version"] != INSPECTION_SCHEMA_VERSION:
            raise ValueError("inspection response has an unsupported version")
        return cls(
            summary=AnalysisSummary.from_dict(
                _mapping(value["summary"], "summary")
            ),
            diagnostics=tuple(
                Diagnostic.from_dict(_mapping(item, "diagnostic"))
                for item in _array(value["diagnostics"], "diagnostics")
            ),
        )


@dataclass(frozen=True)
class ResultQueryRequest:
    """Opaque request for bounded queries against one completed local run."""

    session_id: str
    revision: int
    revision_hash: str
    run_id: str
    queries: tuple[ResultQuery, ...]

    def __post_init__(self) -> None:
        validate_identifier(self.session_id, "session_id")
        if (
            isinstance(self.revision, bool)
            or not isinstance(self.revision, int)
            or self.revision <= 0
        ):
            raise ValueError("revision must be a positive integer")
        _sha256(self.revision_hash, "revision_hash")
        validate_identifier(self.run_id, "run_id")
        object.__setattr__(self, "queries", tuple(self.queries))
        if not self.queries:
            raise ValueError("at least one result query is required")
        if not all(isinstance(item, ResultQuery) for item in self.queries):
            raise TypeError("queries must contain only ResultQuery values")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": RESULT_QUERY_SCHEMA_VERSION,
            "session_id": self.session_id,
            "revision": self.revision,
            "revision_hash": self.revision_hash,
            "run_id": self.run_id,
            "queries": [item.to_dict() for item in self.queries],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ResultQueryRequest":
        required = {
            "schema_version",
            "session_id",
            "revision",
            "revision_hash",
            "run_id",
            "queries",
        }
        if set(value) != required:
            raise ValueError("result query request has invalid fields")
        if value["schema_version"] != RESULT_QUERY_SCHEMA_VERSION:
            raise ValueError("result query request has an unsupported version")
        return cls(
            session_id=value["session_id"],
            revision=value["revision"],
            revision_hash=value["revision_hash"],
            run_id=value["run_id"],
            queries=tuple(
                ResultQuery.from_dict(_mapping(item, "query"))
                for item in _array(value["queries"], "queries")
            ),
        )


class InspectionWorkerError(RuntimeError):
    """Safe parent-process failure from isolated inspection."""


class ResultQueryWorkerError(RuntimeError):
    """Safe parent-process failure from isolated result postprocessing."""


class WorkerRunInProgressError(RuntimeError):
    """The same persisted revision already has a live worker process."""


class WorkerResponseIntegrityError(RuntimeError):
    """A persisted response for the current attempt failed validation."""


@dataclass(frozen=True)
class WorkerResponse:
    """Bounded response read by the parent process."""

    session_id: str
    revision: int
    revision_hash: str
    run_id: str
    status: RunStatus
    result_summary: ResultSummary | None
    artifacts: tuple[ArtifactRecord, ...]
    diagnostics: tuple[Diagnostic, ...]
    elapsed_seconds: float

    def __post_init__(self) -> None:
        validate_identifier(self.session_id, "session_id")
        if (
            isinstance(self.revision, bool)
            or not isinstance(self.revision, int)
            or self.revision <= 0
        ):
            raise ValueError("revision must be a positive integer")
        _sha256(self.revision_hash, "revision_hash")
        validate_identifier(self.run_id, "run_id")
        object.__setattr__(self, "status", RunStatus(self.status))
        object.__setattr__(self, "artifacts", tuple(self.artifacts))
        object.__setattr__(self, "diagnostics", tuple(self.diagnostics))
        if (
            isinstance(self.elapsed_seconds, bool)
            or not isinstance(self.elapsed_seconds, (int, float))
            or self.elapsed_seconds < 0
        ):
            raise ValueError("elapsed_seconds must be nonnegative")
        object.__setattr__(self, "elapsed_seconds", float(self.elapsed_seconds))

    @property
    def ok(self) -> bool:
        return self.status == RunStatus.SUCCEEDED and not has_errors(
            self.diagnostics
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": WORKER_SCHEMA_VERSION,
            "session_id": self.session_id,
            "revision": self.revision,
            "revision_hash": self.revision_hash,
            "run_id": self.run_id,
            "status": self.status.value,
            "result_summary": (
                None
                if self.result_summary is None
                else self.result_summary.to_dict()
            ),
            "artifacts": [item.to_dict() for item in self.artifacts],
            "diagnostics": [item.to_dict() for item in self.diagnostics],
            "elapsed_seconds": self.elapsed_seconds,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "WorkerResponse":
        required = {
            "schema_version",
            "session_id",
            "revision",
            "revision_hash",
            "run_id",
            "status",
            "result_summary",
            "artifacts",
            "diagnostics",
            "elapsed_seconds",
        }
        if set(value) != required:
            raise ValueError("worker response has invalid fields")
        if value["schema_version"] != WORKER_SCHEMA_VERSION:
            raise ValueError("worker response has an unsupported version")
        raw_summary = value["result_summary"]
        return cls(
            session_id=value["session_id"],
            revision=value["revision"],
            revision_hash=value["revision_hash"],
            run_id=value["run_id"],
            status=RunStatus(value["status"]),
            result_summary=(
                None
                if raw_summary is None
                else ResultSummary.from_dict(_mapping(raw_summary, "result_summary"))
            ),
            artifacts=tuple(
                ArtifactRecord.from_dict(_mapping(item, "artifact"))
                for item in _array(value["artifacts"], "artifacts")
            ),
            diagnostics=tuple(
                Diagnostic.from_dict(_mapping(item, "diagnostic"))
                for item in _array(value["diagnostics"], "diagnostics")
            ),
            elapsed_seconds=_number(value["elapsed_seconds"], "elapsed_seconds"),
        )


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
        process: subprocess.Popen[bytes] | None = None
        capture = _BoundedWorkerLogCapture(
            _FIXED_INSPECTION_OUTPUT_LIMIT_BYTES
        )
        stdout_buffer = io.BytesIO()
        stderr_buffer = io.BytesIO()
        readers: list[threading.Thread] = []
        failure: str | None = None
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
                raise OSError("inspection process pipes were not created")
            process.stdin.write(request_payload)
            process.stdin.close()
            for name, source, destination in (
                ("stdout", process.stdout, stdout_buffer),
                ("stderr", process.stderr, stderr_buffer),
            ):
                reader = threading.Thread(
                    target=capture.copy,
                    args=(source, destination),
                    name=f"fem-agent-inspection-{name}",
                    daemon=True,
                )
                reader.start()
                readers.append(reader)

            while process.poll() is None:
                if capture.exceeded:
                    failure = (
                        "The isolated Abaqus inspection exceeded its "
                        "bounded control-output limit."
                    )
                    _terminate_process(process)
                    break
                if cancel_event is not None and cancel_event.is_set():
                    failure = "The isolated Abaqus inspection was cancelled."
                    _terminate_process(process)
                    break
                if time.monotonic() - started >= timeout:
                    failure = (
                        "The isolated Abaqus inspection exceeded its time limit."
                    )
                    _terminate_process(process)
                    break
                time.sleep(_INSPECTION_POLL_SECONDS)
            if process.poll() is None:
                _terminate_process(process)
            process.wait(timeout=_WORKER_LOG_JOIN_SECONDS)
        except OSError as exc:
            raise InspectionWorkerError(
                "The isolated Abaqus inspection process could not start."
            ) from exc
        except subprocess.TimeoutExpired as exc:
            if process is not None:
                _terminate_process(process)
            raise InspectionWorkerError(
                "The isolated Abaqus inspection process did not stop cleanly."
            ) from exc
        finally:
            if process is not None and process.poll() is None:
                _terminate_process(process)
            deadline = time.monotonic() + _WORKER_LOG_JOIN_SECONDS
            for reader in readers:
                reader.join(max(0.0, deadline - time.monotonic()))
            _close_process_pipes(process)
        if any(reader.is_alive() for reader in readers) or capture.failure is not None:
            raise InspectionWorkerError(
                "The isolated Abaqus inspection output could not be captured safely."
            )
        if capture.exceeded and failure is None:
            failure = (
                "The isolated Abaqus inspection exceeded its bounded "
                "control-output limit."
            )
        if failure is not None:
            raise InspectionWorkerError(failure)
        if process is None or process.returncode != 0:
            raise InspectionWorkerError(
                "The isolated Abaqus inspection process failed."
            )
        try:
            payload = json.loads(stdout_buffer.getvalue().decode("utf-8"))
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
        process: subprocess.Popen[bytes] | None = None
        capture = _BoundedWorkerLogCapture(
            _FIXED_INSPECTION_OUTPUT_LIMIT_BYTES
        )
        stdout_buffer = io.BytesIO()
        stderr_buffer = io.BytesIO()
        readers: list[threading.Thread] = []
        failure: str | None = None
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
                raise OSError("result query process pipes were not created")
            process.stdin.write(request_payload)
            process.stdin.close()
            for name, source, destination in (
                ("stdout", process.stdout, stdout_buffer),
                ("stderr", process.stderr, stderr_buffer),
            ):
                reader = threading.Thread(
                    target=capture.copy,
                    args=(source, destination),
                    name=f"fem-agent-result-query-{name}",
                    daemon=True,
                )
                reader.start()
                readers.append(reader)

            while process.poll() is None:
                if capture.exceeded:
                    failure = (
                        "The isolated result query exceeded its bounded "
                        "control-output limit."
                    )
                    _terminate_process(process)
                    break
                if cancel_event is not None and cancel_event.is_set():
                    failure = "The isolated result query was cancelled."
                    _terminate_process(process)
                    break
                if time.monotonic() - started >= timeout:
                    failure = "The isolated result query exceeded its time limit."
                    _terminate_process(process)
                    break
                time.sleep(_INSPECTION_POLL_SECONDS)
            if process.poll() is None:
                _terminate_process(process)
            process.wait(timeout=_WORKER_LOG_JOIN_SECONDS)
        except OSError as exc:
            raise ResultQueryWorkerError(
                "The isolated result query process could not start."
            ) from exc
        except subprocess.TimeoutExpired as exc:
            if process is not None:
                _terminate_process(process)
            raise ResultQueryWorkerError(
                "The isolated result query process did not stop cleanly."
            ) from exc
        finally:
            if process is not None and process.poll() is None:
                _terminate_process(process)
            deadline = time.monotonic() + _WORKER_LOG_JOIN_SECONDS
            for reader in readers:
                reader.join(max(0.0, deadline - time.monotonic()))
            _close_process_pipes(process)
        if any(reader.is_alive() for reader in readers) or capture.failure is not None:
            raise ResultQueryWorkerError(
                "The isolated result query output could not be captured safely."
            )
        if capture.exceeded and failure is None:
            failure = (
                "The isolated result query exceeded its bounded "
                "control-output limit."
            )
        if failure is not None:
            raise ResultQueryWorkerError(failure)
        if process is None or process.returncode != 0:
            raise ResultQueryWorkerError(
                "The isolated result query process failed."
            )
        try:
            payload = json.loads(stdout_buffer.getvalue().decode("utf-8"))
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


def _recover_response_from_manifest(
    store: ArtifactStore,
    record: RevisionRecord,
    run: RunDirectory,
    response_path: Path,
) -> WorkerResponse:
    """Rebuild a missing response only from a fully verified commit manifest."""

    try:
        response = _verified_worker_response_from_manifest(
            store,
            record,
            run,
            allow_manifest_registration=True,
        )
        atomic_write_json(response_path, response.to_dict())
        return response
    except WorkerResponseIntegrityError:
        raise
    except Exception as error:
        raise WorkerResponseIntegrityError(
            "The committed worker manifest failed integrity validation."
        ) from error


def load_verified_worker_response(
    store: ArtifactStore,
    record: RevisionRecord,
    run_id: str,
) -> WorkerResponse:
    """Load one persisted response after rebuilding its committed truth."""

    try:
        run = store.run_directory(record.session_id, run_id)
        response_path = safe_child(
            run.path,
            "logs",
            "worker-response.json",
        )
        persisted = WorkerResponse.from_dict(read_json_file(response_path))
        _require_response_identity(persisted, record, run)
        verified = _verified_worker_response_from_manifest(
            store,
            record,
            run,
            allow_manifest_registration=False,
        )
        if persisted != verified:
            raise ValueError(
                "persisted worker response does not match its committed manifest"
            )
        return persisted
    except WorkerResponseIntegrityError:
        raise
    except Exception as error:
        raise WorkerResponseIntegrityError(
            "The persisted worker response failed integrity validation."
        ) from error


def _verified_worker_response_from_manifest(
    store: ArtifactStore,
    record: RevisionRecord,
    run: RunDirectory,
    *,
    allow_manifest_registration: bool,
) -> WorkerResponse:
    manifest_path = safe_child(run.path, "manifest.json")
    manifest = RunManifest.from_dict(read_json_file(manifest_path))
    _require_manifest_identity(store, record, run, manifest)
    verified_paths = _verify_manifest_artifacts(
        store,
        record.session_id,
        run,
        manifest,
    )
    result_summary = _manifest_result_summary(
        run,
        manifest,
        verified_paths,
    )
    manifest_artifact = _verified_manifest_artifact(
        store,
        record.session_id,
        run,
        manifest,
        allow_registration=allow_manifest_registration,
    )
    response = WorkerResponse(
        session_id=manifest.session_id,
        revision=manifest.revision,
        revision_hash=manifest.revision_hash,
        run_id=manifest.run_id,
        status=manifest.status,
        result_summary=result_summary,
        artifacts=(*manifest.artifacts, manifest_artifact),
        diagnostics=manifest.diagnostics,
        elapsed_seconds=manifest.durations_seconds["total"],
    )
    _require_response_identity(response, record, run)
    return response


def _require_manifest_identity(
    store: ArtifactStore,
    record: RevisionRecord,
    run: RunDirectory,
    manifest: RunManifest,
) -> None:
    if (
        manifest.session_id != record.session_id
        or manifest.revision != record.revision
        or manifest.revision_hash != record.revision_hash
        or manifest.run_id != run.run_id
        or manifest.source_sha256 != record.spec.source_sha256
    ):
        raise ValueError("worker manifest identity does not match its run")
    if manifest.status not in {
        RunStatus.SUCCEEDED,
        RunStatus.FAILED,
        RunStatus.CANCELLED,
    }:
        raise ValueError("worker manifest does not contain a terminal status")
    if manifest.status == RunStatus.SUCCEEDED and has_errors(
        manifest.diagnostics
    ):
        raise ValueError("successful worker manifest contains error diagnostics")
    if (
        manifest.status in {RunStatus.FAILED, RunStatus.CANCELLED}
        and not has_errors(manifest.diagnostics)
    ):
        raise ValueError("failed worker manifest contains no error diagnostic")
    if (
        manifest.unit_context != record.spec.unit_context
        or manifest.analysis_step != record.spec.analysis_step
    ):
        raise ValueError("worker manifest analysis specification mismatch")
    expected_parameters = {
        "requested_queries": [
            item.to_dict() for item in record.spec.requested_queries
        ],
        "export_formats": [
            item.value for item in record.spec.export_formats
        ],
        "resource_limits": record.spec.resource_limits.to_dict(),
    }
    actual_parameters = dict(manifest.tool_parameters)
    solution_model_sha256 = actual_parameters.pop(
        "solution_model_sha256",
        None,
    )
    if solution_model_sha256 is not None:
        _sha256(solution_model_sha256, "solution_model_sha256")
    if actual_parameters != expected_parameters:
        raise ValueError("worker manifest tool parameters mismatch")
    has_solution = any(
        artifact.kind == "solution"
        for artifact in manifest.artifacts
    )
    if solution_model_sha256 is not None and not has_solution:
        raise ValueError(
            "worker manifest has a model fingerprint without a solution"
        )
    if any(
        diagnostic not in manifest.diagnostics
        for diagnostic in manifest.validation_diagnostics
    ):
        raise ValueError("worker manifest validation diagnostics mismatch")
    if set(manifest.timestamps) != {"started_at", "completed_at"}:
        raise ValueError("worker manifest timestamps are incomplete")
    started = _timestamp(manifest.timestamps["started_at"], "started_at")
    completed = _timestamp(manifest.timestamps["completed_at"], "completed_at")
    if completed < started:
        raise ValueError("worker manifest completion precedes its start")
    if "total" not in manifest.durations_seconds:
        raise ValueError("worker manifest is missing its total duration")

    source = store.get_artifact(
        record.session_id,
        record.spec.source_artifact_id,
    )
    if (
        source.kind != "input"
        or source.sha256 != record.spec.source_sha256
    ):
        raise ValueError("worker manifest source artifact mismatch")
    store.resolve_artifact(
        record.session_id,
        source.artifact_id,
        verify=True,
    )


def _verify_manifest_artifacts(
    store: ArtifactStore,
    session_id: str,
    run: RunDirectory,
    manifest: RunManifest,
) -> dict[str, Path]:
    records = manifest.artifacts
    identifiers = [item.artifact_id for item in records]
    display_paths = [item.display_path for item in records]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("worker manifest contains duplicate artifact identifiers")
    if len(set(display_paths)) != len(display_paths):
        raise ValueError("worker manifest contains duplicate artifact paths")

    verified: dict[str, Path] = {}
    export_kinds = set(manifest.tool_parameters["export_formats"])
    allowed_kinds = {
        "diagnostics",
        "result_summary",
        "solution",
        *export_kinds,
    }
    for expected in records:
        parts = PurePosixPath(expected.display_path).parts
        if len(parts) < 3 or parts[:2] != ("runs", run.run_id):
            raise ValueError("worker manifest artifact escapes its run")
        if expected.kind == "manifest":
            raise ValueError("worker manifest cannot list itself as an artifact")
        if expected.kind not in allowed_kinds:
            raise ValueError("worker manifest contains an unexpected artifact kind")
        if expected.kind in export_kinds and (
            len(parts) != 4 or parts[2] != "exports"
        ):
            raise ValueError("worker export artifact is stored at an invalid path")
        persisted = store.get_artifact(session_id, expected.artifact_id)
        if persisted != expected:
            raise ValueError("worker manifest artifact metadata mismatch")
        resolved = store.resolve_artifact(
            session_id,
            expected.artifact_id,
            verify=True,
        )
        if not resolved.is_relative_to(run.path):
            raise ValueError("worker manifest artifact resolves outside its run")
        verified[expected.artifact_id] = resolved

    diagnostics = [item for item in records if item.kind == "diagnostics"]
    expected_diagnostics_path = (
        PurePosixPath("runs") / run.run_id / "diagnostics.json"
    ).as_posix()
    if (
        len(diagnostics) != 1
        or diagnostics[0].display_path != expected_diagnostics_path
    ):
        raise ValueError("worker manifest diagnostics artifact mismatch")
    diagnostics_payload = read_json_file(
        verified[diagnostics[0].artifact_id]
    )
    if set(diagnostics_payload) != {"schema_version", "diagnostics"}:
        raise ValueError("worker diagnostics artifact has invalid fields")
    if diagnostics_payload["schema_version"] != WORKER_SCHEMA_VERSION:
        raise ValueError("worker diagnostics artifact has an unsupported version")
    persisted_diagnostics = tuple(
        Diagnostic.from_dict(_mapping(item, "diagnostic"))
        for item in _array(
            diagnostics_payload["diagnostics"],
            "diagnostics",
        )
    )
    if persisted_diagnostics != manifest.diagnostics:
        raise ValueError("worker diagnostics artifact does not match its manifest")

    solutions = [item for item in records if item.kind == "solution"]
    expected_solution_path = (
        PurePosixPath("runs") / run.run_id / "solution.npy"
    ).as_posix()
    if len(solutions) > 1:
        raise ValueError("worker manifest contains duplicate solution artifacts")
    if solutions and solutions[0].display_path != expected_solution_path:
        raise ValueError("worker solution is stored at an invalid path")
    solution_path = safe_child(run.path, "solution.npy")
    if solution_path.exists() != bool(solutions):
        raise ValueError("worker manifest solution artifact mismatch")

    listed_display_paths = set(display_paths)
    exports = safe_child(run.path, "exports")
    for candidate in exports.iterdir():
        candidate_display_path = (
            PurePosixPath("runs")
            / run.run_id
            / "exports"
            / candidate.name
        ).as_posix()
        if (
            candidate.is_symlink()
            or not candidate.is_file()
            or candidate_display_path not in listed_display_paths
        ):
            raise ValueError("worker export files do not match their manifest")
    return verified


def _manifest_result_summary(
    run: RunDirectory,
    manifest: RunManifest,
    verified_paths: Mapping[str, Path],
) -> ResultSummary | None:
    result_records = tuple(
        item for item in manifest.artifacts if item.kind == "result_summary"
    )
    result_path = safe_child(run.path, "result-summary.json")
    if manifest.result_summary_artifact_id is None:
        if result_records or result_path.exists():
            raise ValueError("worker manifest result-summary identity mismatch")
        return None
    if (
        len(result_records) != 1
        or result_records[0].artifact_id
        != manifest.result_summary_artifact_id
    ):
        raise ValueError("worker manifest result-summary artifact mismatch")
    expected_display_path = (
        PurePosixPath("runs") / run.run_id / "result-summary.json"
    ).as_posix()
    record = result_records[0]
    if record.display_path != expected_display_path:
        raise ValueError("worker result summary is stored at an invalid path")
    summary = ResultSummary.from_dict(
        read_json_file(verified_paths[record.artifact_id])
    )
    if summary.run_id != run.run_id or summary.step != manifest.analysis_step:
        raise ValueError("worker result summary identity mismatch")
    if any(
        scalar.run_id != run.run_id or scalar.step != manifest.analysis_step
        for scalar in summary.scalars
    ):
        raise ValueError("worker scalar result identity mismatch")
    return summary


def _verified_manifest_artifact(
    store: ArtifactStore,
    session_id: str,
    run: RunDirectory,
    manifest: RunManifest,
    *,
    allow_registration: bool,
) -> ArtifactRecord:
    expected_display_path = (
        PurePosixPath("runs") / run.run_id / "manifest.json"
    ).as_posix()
    run_records = tuple(
        item
        for item in store.list_artifacts(session_id)
        if _artifact_belongs_to_run(item, run.run_id)
    )
    manifest_records = tuple(
        item
        for item in run_records
        if item.display_path == expected_display_path
    )
    if len(manifest_records) > 1:
        raise ValueError("worker manifest has duplicate artifact metadata")
    listed_ids = {item.artifact_id for item in manifest.artifacts}
    allowed_ids = listed_ids | {
        item.artifact_id for item in manifest_records
    }
    if any(item.artifact_id not in allowed_ids for item in run_records):
        raise ValueError("worker run contains artifacts absent from its manifest")

    manifest_path = safe_child(run.path, "manifest.json")
    if manifest_records:
        manifest_artifact = manifest_records[0]
        if manifest_artifact.kind != "manifest":
            raise ValueError("worker manifest artifact has the wrong kind")
        resolved = store.resolve_artifact(
            session_id,
            manifest_artifact.artifact_id,
            verify=True,
        )
        if resolved != manifest_path:
            raise ValueError("worker manifest artifact path mismatch")
    elif allow_registration:
        manifest_artifact = store.register_run_artifact(
            session_id,
            run.run_id,
            manifest_path,
            kind="manifest",
        )
    else:
        raise ValueError("worker manifest artifact metadata is missing")
    reloaded = RunManifest.from_dict(read_json_file(manifest_path))
    if reloaded != manifest:
        raise ValueError("worker manifest changed during recovery")
    store.resolve_artifact(
        session_id,
        manifest_artifact.artifact_id,
        verify=True,
    )
    return manifest_artifact


def _artifact_belongs_to_run(
    artifact: ArtifactRecord,
    run_id: str,
) -> bool:
    parts = PurePosixPath(artifact.display_path).parts
    return len(parts) >= 3 and parts[:2] == ("runs", run_id)


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


def _require_response_identity(
    response: WorkerResponse,
    record: RevisionRecord,
    run: RunDirectory,
) -> None:
    if (
        response.session_id != record.session_id
        or response.revision != record.revision
        or response.revision_hash != record.revision_hash
        or response.run_id != run.run_id
    ):
        raise ValueError("persisted worker response identity does not match its run")
    if response.status == RunStatus.RUNNING:
        raise ValueError("persisted worker response is not terminal")
    if response.status == RunStatus.SUCCEEDED and has_errors(response.diagnostics):
        raise ValueError("successful worker response contains error diagnostics")
    if (
        response.status in {RunStatus.FAILED, RunStatus.CANCELLED}
        and not has_errors(response.diagnostics)
    ):
        raise ValueError("failed worker response contains no error diagnostic")


def _require_request_identity(
    request: WorkerRequest,
    record: RevisionRecord,
    run: RunDirectory,
    idempotency_key: str,
) -> None:
    if (
        request.session_id != record.session_id
        or request.revision != record.revision
        or request.revision_hash != record.revision_hash
        or request.run_id != run.run_id
        or request.idempotency_key != idempotency_key
    ):
        raise ValueError("persisted worker request identity does not match its run")


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


def _timestamp(value: Any, name: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as error:
        raise ValueError(f"{name} must be an ISO-8601 timestamp") from error
    if parsed.tzinfo is None:
        raise ValueError(f"{name} must include a timezone")
    return parsed.astimezone(timezone.utc)


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


def _sha256(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _array(value: Any, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be an array")
    return value


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    return float(value)


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
