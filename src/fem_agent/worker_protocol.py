"""Serializable requests, responses, and UTF-8 control messages for FEM workers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, BinaryIO, Mapping

from .artifacts import validate_identifier
from .diagnostics import has_errors
from .schemas import (
    AnalysisSummary,
    ArtifactRecord,
    Diagnostic,
    ImportAnalysisSpec,
    ResultQuery,
    ResultSummary,
    RunStatus,
)
from .state import hash_revision_spec


WORKER_SCHEMA_VERSION = 1
INSPECTION_SCHEMA_VERSION = 1
RESULT_QUERY_SCHEMA_VERSION = 1
_FIXED_CONTROL_PAYLOAD_LIMIT_BYTES = 1024 * 1024


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


__all__ = [
    "INSPECTION_SCHEMA_VERSION",
    "RESULT_QUERY_SCHEMA_VERSION",
    "WORKER_SCHEMA_VERSION",
    "InspectionRequest",
    "InspectionResponse",
    "InspectionWorkerError",
    "ResultQueryRequest",
    "ResultQueryWorkerError",
    "WorkerRequest",
    "WorkerResponse",
]
