"""Revision-bound confirmation records for deterministic solve authorization."""

from __future__ import annotations

import os
from dataclasses import dataclass
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
from .state import (
    RevisionCorruptionError,
    RevisionNotFoundError,
    RevisionRecord,
    RevisionStore,
    StaleRevisionError,
)


CONFIRMATION_SCHEMA_VERSION = 1


class ConfirmationError(RuntimeError):
    """Base class for confirmation failures."""


class ConfirmationRequiredError(ConfirmationError):
    """Raised when a solve lacks confirmation for its exact revision."""


class ConfirmationNotReadyError(ConfirmationError):
    """Raised when required analysis fields are still incomplete."""


class ConfirmationCorruptionError(ConfirmationError):
    """Raised when persisted confirmation metadata fails validation."""


@dataclass(frozen=True)
class ConfirmationRecord:
    """Local proof that a user confirmed one exact immutable revision."""

    session_id: str
    revision: int
    revision_hash: str
    confirmed_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": CONFIRMATION_SCHEMA_VERSION,
            "session_id": self.session_id,
            "revision": self.revision,
            "revision_hash": self.revision_hash,
            "confirmed_at": self.confirmed_at,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ConfirmationRecord:
        required = {
            "schema_version",
            "session_id",
            "revision",
            "revision_hash",
            "confirmed_at",
        }
        if set(payload) != required:
            raise ConfirmationCorruptionError(
                "confirmation record has invalid fields"
            )
        if payload["schema_version"] != CONFIRMATION_SCHEMA_VERSION:
            raise ConfirmationCorruptionError(
                "confirmation record has an unsupported version"
            )
        try:
            session_id = validate_identifier(payload["session_id"], "session_id")
            revision = _positive_integer(payload["revision"], "revision")
            revision_hash = _sha256(payload["revision_hash"], "revision_hash")
            confirmed_at = _timestamp(payload["confirmed_at"])
        except (TypeError, ValueError) as exc:
            raise ConfirmationCorruptionError(
                "confirmation record is invalid"
            ) from exc
        return cls(
            session_id=session_id,
            revision=revision,
            revision_hash=revision_hash,
            confirmed_at=confirmed_at,
        )


class ConfirmationStore:
    """Persist and enforce confirmation for the current revision only."""

    def __init__(
        self,
        workspace: str | os.PathLike[str],
        revision_store: RevisionStore | None = None,
    ) -> None:
        self._root = normalize_workspace(workspace)
        if revision_store is not None and revision_store.root != self._root:
            raise ValueError("revision_store must use the same workspace")
        self._revisions = revision_store or RevisionStore(self._root)

    @property
    def root(self) -> Path:
        return self._root

    def confirm(
        self,
        session_id: str,
        *,
        revision: int,
        revision_hash: str,
    ) -> ConfirmationRecord:
        """Confirm an exact current revision; repeated confirmation is idempotent."""

        identifier = validate_identifier(session_id, "session_id")
        number = _positive_integer(revision, "revision")
        digest = _sha256(revision_hash, "revision_hash")
        current = self._revisions.require_current(
            identifier,
            expected_revision=number,
            expected_hash=digest,
        )
        if not current.spec.ready_for_confirmation:
            raise ConfirmationNotReadyError(
                "unit context, analysis step, and result requests are required"
            )

        directory = self._confirmation_directory(identifier, create=True)
        target = safe_child(directory, _confirmation_filename(number))
        if target.exists():
            existing = self._load_path(target)
            self._verify_against_revision(existing, current)
            return existing

        record = ConfirmationRecord(
            session_id=identifier,
            revision=number,
            revision_hash=digest,
            confirmed_at=_utc_now(),
        )
        try:
            atomic_write_json(target, record.to_dict())
        except FileExistsError:
            existing = self._load_path(target)
            self._verify_against_revision(existing, current)
            return existing
        return record

    def get(
        self,
        session_id: str,
        revision: int,
    ) -> ConfirmationRecord | None:
        """Load a historical confirmation and verify its revision hash."""

        identifier = validate_identifier(session_id, "session_id")
        number = _positive_integer(revision, "revision")
        directory = self._confirmation_directory(identifier, create=False)
        if directory is None:
            return None
        path = safe_child(directory, _confirmation_filename(number))
        if not path.exists():
            return None
        record = self._load_path(path)
        try:
            revision_record = self._revisions.get(identifier, number)
        except (RevisionNotFoundError, RevisionCorruptionError) as exc:
            raise ConfirmationCorruptionError(
                "confirmation references an invalid revision"
            ) from exc
        self._verify_against_revision(record, revision_record)
        return record

    def current(self, session_id: str) -> ConfirmationRecord | None:
        """Return confirmation only when it belongs to the current revision."""

        identifier = validate_identifier(session_id, "session_id")
        current = self._revisions.latest(identifier)
        if current is None:
            return None
        return self.get(identifier, current.revision)

    def is_confirmed(
        self,
        session_id: str,
        *,
        revision: int | None = None,
        revision_hash: str | None = None,
    ) -> bool:
        """Return whether the exact current revision is confirmed."""

        try:
            self.require_confirmed(
                session_id,
                revision=revision,
                revision_hash=revision_hash,
            )
        except (
            ConfirmationRequiredError,
            RevisionNotFoundError,
            SessionNotFoundError,
            StaleRevisionError,
        ):
            return False
        return True

    def require_confirmed(
        self,
        session_id: str,
        *,
        revision: int | None = None,
        revision_hash: str | None = None,
    ) -> ConfirmationRecord:
        """Require confirmation for the supplied, current number and hash."""

        identifier = validate_identifier(session_id, "session_id")
        current = self._revisions.require_current(
            identifier,
            expected_revision=revision,
            expected_hash=revision_hash,
        )
        confirmation = self.get(identifier, current.revision)
        if confirmation is None:
            raise ConfirmationRequiredError(
                f"revision {current.revision} has not been confirmed"
            )
        self._verify_against_revision(confirmation, current)
        return confirmation

    require_confirmation = require_confirmed

    def _confirmation_directory(
        self,
        session_id: str,
        *,
        create: bool,
    ) -> Path | None:
        session = ensure_session_directory(
            self._root,
            session_id,
            create=False,
        )
        directory = safe_child(session, "confirmations")
        if create:
            directory.mkdir(mode=0o700, exist_ok=True)
        if not directory.exists():
            return None
        if not directory.is_dir():
            raise ConfirmationCorruptionError(
                "confirmation path is not a directory"
            )
        return directory

    def _load_path(self, path: Path) -> ConfirmationRecord:
        try:
            payload = read_json_file(path)
            return ConfirmationRecord.from_dict(payload)
        except ArtifactIntegrityError as exc:
            raise ConfirmationCorruptionError(
                "confirmation record cannot be read safely"
            ) from exc

    @staticmethod
    def _verify_against_revision(
        confirmation: ConfirmationRecord,
        revision: RevisionRecord,
    ) -> None:
        if (
            confirmation.session_id != revision.session_id
            or confirmation.revision != revision.revision
            or confirmation.revision_hash != revision.revision_hash
        ):
            raise ConfirmationCorruptionError(
                "confirmation does not match its immutable revision"
            )


ConfirmationManager = ConfirmationStore


def _confirmation_filename(revision: int) -> str:
    if revision > 99_999_999:
        raise ValueError("revision exceeds the persisted filename range")
    return f"{revision:08d}.json"


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
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
        raise ValueError("confirmed_at must be a string")
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ValueError("confirmed_at must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise ValueError("confirmed_at must include a timezone")
    return value


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


__all__ = [
    "ConfirmationCorruptionError",
    "ConfirmationError",
    "ConfirmationManager",
    "ConfirmationNotReadyError",
    "ConfirmationRecord",
    "ConfirmationRequiredError",
    "ConfirmationStore",
]
