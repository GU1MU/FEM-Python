"""Immutable, versioned analysis specifications for Agent sessions."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import threading
import uuid
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .artifacts import (
    ArtifactIntegrityError,
    SessionNotFoundError,
    atomic_write_json,
    ensure_session_directory,
    normalize_workspace,
    read_json_file,
    safe_child,
    validate_identifier,
)
from .schemas import ImportAnalysisSpec, SchemaValidationError


REVISION_RECORD_SCHEMA_VERSION = 1
_REVISION_FILENAME_PATTERN = re.compile(r"([0-9]{8})\.json\Z")


class RevisionStoreError(RuntimeError):
    """Base class for revision persistence errors."""


class RevisionNotFoundError(RevisionStoreError):
    """Raised when a requested revision does not exist."""


class StaleRevisionError(RevisionStoreError):
    """Raised when a mutation or confirmation targets an old revision."""


class IdempotencyConflictError(RevisionStoreError):
    """Raised when one idempotency key is reused for different input."""


class RevisionCorruptionError(RevisionStoreError):
    """Raised when a persisted immutable revision fails validation."""


@dataclass(frozen=True)
class RevisionRecord:
    """One immutable persisted specification and its provenance."""

    session_id: str
    revision: int
    revision_hash: str
    expected_revision: int
    idempotency_key: str
    operation: str
    request_fingerprint: str
    created_at: str
    spec: ImportAnalysisSpec

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": REVISION_RECORD_SCHEMA_VERSION,
            "session_id": self.session_id,
            "revision": self.revision,
            "revision_hash": self.revision_hash,
            "expected_revision": self.expected_revision,
            "idempotency_key": self.idempotency_key,
            "operation": self.operation,
            "request_fingerprint": self.request_fingerprint,
            "created_at": self.created_at,
            "spec": self.spec.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> RevisionRecord:
        required = {
            "schema_version",
            "session_id",
            "revision",
            "revision_hash",
            "expected_revision",
            "idempotency_key",
            "operation",
            "request_fingerprint",
            "created_at",
            "spec",
        }
        if set(payload) != required:
            raise RevisionCorruptionError("revision record has invalid fields")
        if payload["schema_version"] != REVISION_RECORD_SCHEMA_VERSION:
            raise RevisionCorruptionError("revision record has an unsupported version")
        try:
            session_id = validate_identifier(payload["session_id"], "session_id")
            revision = _positive_integer(payload["revision"], "revision")
            expected_revision = _nonnegative_integer(
                payload["expected_revision"],
                "expected_revision",
            )
            idempotency_key = validate_identifier(
                payload["idempotency_key"],
                "idempotency_key",
            )
            operation = validate_identifier(payload["operation"], "operation")
            revision_hash = _sha256(payload["revision_hash"], "revision_hash")
            request_fingerprint = _sha256(
                payload["request_fingerprint"],
                "request_fingerprint",
            )
            created_at = _timestamp(payload["created_at"])
            raw_spec = payload["spec"]
            if not isinstance(raw_spec, Mapping):
                raise RevisionCorruptionError("revision spec must be an object")
            spec = ImportAnalysisSpec.from_dict(raw_spec)
        except (
            ArtifactIntegrityError,
            SchemaValidationError,
            TypeError,
            ValueError,
        ) as exc:
            if isinstance(exc, RevisionCorruptionError):
                raise
            raise RevisionCorruptionError("revision record is invalid") from exc

        record = cls(
            session_id=session_id,
            revision=revision,
            revision_hash=revision_hash,
            expected_revision=expected_revision,
            idempotency_key=idempotency_key,
            operation=operation,
            request_fingerprint=request_fingerprint,
            created_at=created_at,
            spec=spec,
        )
        _verify_record(record)
        return record


class RevisionStore:
    """Append-only store for session analysis specifications."""

    def __init__(self, workspace: str | os.PathLike[str]) -> None:
        self._root = normalize_workspace(workspace)
        self._lock = threading.RLock()

    @property
    def root(self) -> Path:
        return self._root

    def create_session(self, session_id: str | None = None) -> str:
        """Create or idempotently reopen a session revision directory."""

        identifier = (
            validate_identifier(session_id, "session_id")
            if session_id is not None
            else f"ses_{uuid.uuid4().hex}"
        )
        with self._lock:
            session = ensure_session_directory(self._root, identifier, create=True)
            revisions = safe_child(session, "revisions")
            revisions.mkdir(mode=0o700, exist_ok=True)
            _ensure_plain_directory(revisions)
        return identifier

    def initialize(
        self,
        spec: ImportAnalysisSpec,
        *,
        idempotency_key: str,
        operation: str = "initialize",
    ) -> RevisionRecord:
        """Persist revision one for a session, creating the session if needed."""

        if spec.revision != 1:
            raise ValueError("initial specification must have revision 1")
        self.create_session(spec.session_id)
        return self.commit(
            spec,
            expected_revision=0,
            idempotency_key=idempotency_key,
            operation=operation,
        )

    def commit(
        self,
        spec: ImportAnalysisSpec,
        *,
        expected_revision: int,
        idempotency_key: str,
        operation: str = "mutation",
    ) -> RevisionRecord:
        """Append a specification using optimistic concurrency and idempotency."""

        expected = _nonnegative_integer(expected_revision, "expected_revision")
        key = validate_identifier(idempotency_key, "idempotency_key")
        operation_name = validate_identifier(operation, "operation")
        normalized_spec = _normalize_spec(spec)
        session_id = validate_identifier(normalized_spec.session_id, "session_id")
        if normalized_spec.revision != expected + 1:
            raise ValueError(
                "spec.revision must equal expected_revision + 1"
            )
        request_fingerprint = _request_fingerprint(
            normalized_spec,
            expected,
            operation_name,
        )

        with self._lock:
            revisions = self._revision_directory(session_id)
            prior = self._find_by_idempotency_key(session_id, key)
            if prior is not None:
                if prior.request_fingerprint != request_fingerprint:
                    raise IdempotencyConflictError(
                        "idempotency key was already used for a different mutation"
                    )
                return prior

            current = self.latest(session_id)
            actual_revision = 0 if current is None else current.revision
            if actual_revision != expected:
                raise StaleRevisionError(
                    f"expected revision {expected}, current revision is "
                    f"{actual_revision}"
                )

            record = RevisionRecord(
                session_id=session_id,
                revision=normalized_spec.revision,
                revision_hash=hash_revision_spec(normalized_spec),
                expected_revision=expected,
                idempotency_key=key,
                operation=operation_name,
                request_fingerprint=request_fingerprint,
                created_at=_utc_now(),
                spec=normalized_spec,
            )
            target = safe_child(
                revisions,
                _revision_filename(record.revision),
            )
            try:
                atomic_write_json(target, record.to_dict())
            except FileExistsError as exc:
                existing = self.get(session_id, record.revision)
                if (
                    existing.idempotency_key == key
                    and existing.request_fingerprint == request_fingerprint
                ):
                    return existing
                raise StaleRevisionError(
                    f"revision {record.revision} was committed concurrently"
                ) from exc
            return record

    def mutate(
        self,
        session_id: str,
        *,
        expected_revision: int,
        idempotency_key: str,
        changes: Mapping[str, Any],
        operation: str = "mutation",
    ) -> RevisionRecord:
        """Create the next revision by replacing selected specification fields."""

        identifier = validate_identifier(session_id, "session_id")
        expected = _positive_integer(expected_revision, "expected_revision")
        if not isinstance(changes, Mapping):
            raise TypeError("changes must be a mapping")
        forbidden = {"session_id", "revision"}
        unknown = set(changes) - {
            "source_artifact_id",
            "source_sha256",
            "unit_context",
            "analysis_step",
            "requested_queries",
            "export_formats",
            "resource_limits",
            "assumptions",
        }
        if forbidden & set(changes):
            raise ValueError("session_id and revision cannot be replaced")
        if unknown:
            raise ValueError(
                f"unknown specification fields: {', '.join(sorted(unknown))}"
            )
        try:
            base = self.get(identifier, expected)
        except RevisionNotFoundError as exc:
            current = self.latest(identifier)
            actual = 0 if current is None else current.revision
            raise StaleRevisionError(
                f"expected revision {expected}, current revision is {actual}"
            ) from exc
        try:
            candidate = replace(
                base.spec,
                revision=expected + 1,
                **dict(changes),
            )
        except TypeError as exc:
            raise ValueError("invalid specification mutation") from exc
        return self.commit(
            candidate,
            expected_revision=expected,
            idempotency_key=idempotency_key,
            operation=operation,
        )

    def latest(self, session_id: str) -> RevisionRecord | None:
        """Return the current revision, or ``None`` before initialization."""

        records = self.list_records(session_id)
        return records[-1] if records else None

    def get(self, session_id: str, revision: int) -> RevisionRecord:
        """Load and verify one exact immutable revision."""

        identifier = validate_identifier(session_id, "session_id")
        number = _positive_integer(revision, "revision")
        revisions = self._revision_directory(identifier)
        path = safe_child(revisions, _revision_filename(number))
        if not path.exists():
            raise RevisionNotFoundError(
                f"session {identifier} has no revision {number}"
            )
        try:
            payload = read_json_file(path)
            record = RevisionRecord.from_dict(payload)
        except ArtifactIntegrityError as exc:
            raise RevisionCorruptionError(
                f"revision {number} cannot be read safely"
            ) from exc
        if record.session_id != identifier or record.revision != number:
            raise RevisionCorruptionError("revision file identity mismatch")
        return record

    def list_records(self, session_id: str) -> tuple[RevisionRecord, ...]:
        """Load all revisions and verify that their sequence has no gaps."""

        identifier = validate_identifier(session_id, "session_id")
        revisions = self._revision_directory(identifier)
        numbered_paths: list[tuple[int, Path]] = []
        for path in revisions.iterdir():
            match = _REVISION_FILENAME_PATTERN.fullmatch(path.name)
            if match is None:
                if path.name.startswith(".") and path.name.endswith(".tmp"):
                    continue
                raise RevisionCorruptionError(
                    "revision directory contains an unexpected entry"
                )
            numbered_paths.append((int(match.group(1)), path))
        numbered_paths.sort(key=lambda item: item[0])
        for expected_number, (actual_number, _) in enumerate(
            numbered_paths,
            start=1,
        ):
            if actual_number != expected_number:
                raise RevisionCorruptionError("revision sequence contains a gap")
        return tuple(
            self.get(identifier, number) for number, _ in numbered_paths
        )

    def require_current(
        self,
        session_id: str,
        *,
        expected_revision: int | None = None,
        expected_hash: str | None = None,
    ) -> RevisionRecord:
        """Return the current record and reject stale number/hash expectations."""

        identifier = validate_identifier(session_id, "session_id")
        current = self.latest(identifier)
        if current is None:
            raise RevisionNotFoundError(
                f"session {identifier} has no analysis revision"
            )
        if expected_revision is not None:
            expected = _positive_integer(expected_revision, "expected_revision")
            if current.revision != expected:
                raise StaleRevisionError(
                    f"expected revision {expected}, current revision is "
                    f"{current.revision}"
                )
        if expected_hash is not None:
            digest = _sha256(expected_hash, "expected_hash")
            if current.revision_hash != digest:
                raise StaleRevisionError("revision hash does not match current state")
        return current

    def _revision_directory(self, session_id: str) -> Path:
        try:
            session = ensure_session_directory(
                self._root,
                session_id,
                create=False,
            )
        except SessionNotFoundError:
            raise
        revisions = safe_child(session, "revisions")
        if not revisions.exists():
            raise SessionNotFoundError(f"unknown session_id {session_id}")
        _ensure_plain_directory(revisions)
        return revisions

    def _find_by_idempotency_key(
        self,
        session_id: str,
        idempotency_key: str,
    ) -> RevisionRecord | None:
        for record in self.list_records(session_id):
            if record.idempotency_key == idempotency_key:
                return record
        return None


def hash_revision_spec(spec: ImportAnalysisSpec) -> str:
    """Return the deterministic SHA-256 of the canonical specification JSON."""

    normalized = _normalize_spec(spec)
    encoded = normalized.to_json().encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _normalize_spec(spec: ImportAnalysisSpec) -> ImportAnalysisSpec:
    if not isinstance(spec, ImportAnalysisSpec):
        raise TypeError("spec must be an ImportAnalysisSpec")
    try:
        return ImportAnalysisSpec.from_dict(spec.to_dict())
    except SchemaValidationError:
        raise


def _request_fingerprint(
    spec: ImportAnalysisSpec,
    expected_revision: int,
    operation: str,
) -> str:
    payload = {
        "expected_revision": expected_revision,
        "operation": operation,
        "spec": spec.to_dict(),
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _verify_record(record: RevisionRecord) -> None:
    if record.spec.session_id != record.session_id:
        raise RevisionCorruptionError("revision spec session mismatch")
    if record.spec.revision != record.revision:
        raise RevisionCorruptionError("revision spec number mismatch")
    if record.expected_revision + 1 != record.revision:
        raise RevisionCorruptionError("revision predecessor mismatch")
    if hash_revision_spec(record.spec) != record.revision_hash:
        raise RevisionCorruptionError("revision hash mismatch")
    expected_fingerprint = _request_fingerprint(
        record.spec,
        record.expected_revision,
        record.operation,
    )
    if expected_fingerprint != record.request_fingerprint:
        raise RevisionCorruptionError("revision request fingerprint mismatch")


def _revision_filename(revision: int) -> str:
    if revision > 99_999_999:
        raise ValueError("revision exceeds the persisted filename range")
    return f"{revision:08d}.json"


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonnegative_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _sha256(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _timestamp(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError("created_at must be a string")
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ValueError("created_at must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise ValueError("created_at must include a timezone")
    return value


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _ensure_plain_directory(path: Path) -> None:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise RevisionCorruptionError("revision directory cannot be inspected") from exc
    attributes = getattr(metadata, "st_file_attributes", 0)
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    if not path.is_dir() or bool(attributes & reparse_flag):
        raise RevisionCorruptionError("revision path is not a plain directory")


__all__ = [
    "IdempotencyConflictError",
    "RevisionCorruptionError",
    "RevisionNotFoundError",
    "RevisionRecord",
    "RevisionStore",
    "RevisionStoreError",
    "SessionNotFoundError",
    "StaleRevisionError",
    "hash_revision_spec",
]
