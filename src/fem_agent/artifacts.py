"""Local, path-confined storage for Agent inputs and run artifacts.

The store deliberately separates a user-selected source path from the paths
that are later visible to the Agent engine.  Once copied, inputs are addressed
only by opaque identifiers and relative display paths.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import threading
import unicodedata
import uuid
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from .schemas import ArtifactRecord


STORE_SCHEMA_VERSION = 1
_IDENTIFIER_PATTERN = re.compile(r"[A-Za-z0-9_-]{1,128}\Z")
_REVISION_METADATA_LIMIT = 1024 * 1024
_COPY_CHUNK_SIZE = 1024 * 1024
_WINDOWS_RESERVED_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{number}" for number in range(1, 10)),
    *(f"LPT{number}" for number in range(1, 10)),
}


class ArtifactStoreError(RuntimeError):
    """Base class for local artifact storage failures."""


class InvalidIdentifierError(ArtifactStoreError, ValueError):
    """Raised when an opaque identifier could affect path resolution."""


class UnsafePathError(ArtifactStoreError, ValueError):
    """Raised when a path escapes its allowed root or crosses a link."""


class InputRejectedError(ArtifactStoreError, ValueError):
    """Raised when a selected input is not an acceptable bounded ``.inp``."""


class SessionNotFoundError(ArtifactStoreError):
    """Raised when a session directory does not exist."""


class ArtifactNotFoundError(ArtifactStoreError):
    """Raised when an artifact identifier is unknown in a session."""


class ArtifactIntegrityError(ArtifactStoreError):
    """Raised when persisted metadata or bytes fail integrity checks."""


@dataclass(frozen=True)
class RunDirectory:
    """Opaque run identifier and its local, session-relative directory."""

    run_id: str
    path: Path
    display_path: str


class ArtifactStore:
    """Create isolated session inputs and run directories below one workspace."""

    def __init__(self, workspace: str | os.PathLike[str]) -> None:
        self._root = normalize_workspace(workspace)
        self._sessions_root = safe_child(self._root, "sessions")
        self._sessions_root.mkdir(mode=0o700, parents=True, exist_ok=True)
        _require_directory(self._sessions_root)
        self._lock = threading.RLock()

    @property
    def root(self) -> Path:
        return self._root

    def create_session(self, session_id: str | None = None) -> str:
        """Create, or idempotently reopen, a session directory."""

        with self._lock:
            identifier = (
                validate_identifier(session_id, "session_id")
                if session_id is not None
                else _new_identifier("ses")
            )
            session = ensure_session_directory(
                self._root,
                identifier,
                create=True,
            )
            for name in ("inputs", "artifacts", "runs"):
                directory = safe_child(session, name)
                directory.mkdir(mode=0o700, exist_ok=True)
                _require_directory(directory)
            return identifier

    def session_path(self, session_id: str) -> Path:
        """Return an existing, verified session directory."""

        return ensure_session_directory(self._root, session_id, create=False)

    def copy_input(
        self,
        session_id: str,
        source_path: str | os.PathLike[str],
        *,
        max_bytes: int = 50 * 1024 * 1024,
        source_encoding: str | None = None,
    ) -> ArtifactRecord:
        """Copy one regular ``.inp`` file into an immutable session location.

        The source may be outside the Agent workspace because it is selected by
        the local UI.  A declared source encoding transcodes the private copy
        to UTF-8 while leaving the user-owned source unchanged.  The returned
        record contains no source path.
        """

        if isinstance(max_bytes, bool) or not isinstance(max_bytes, int):
            raise InputRejectedError("max_bytes must be an integer")
        if max_bytes <= 0:
            raise InputRejectedError("max_bytes must be greater than zero")
        if source_encoding not in {
            None,
            "utf-8",
            "utf-8-sig",
            "utf-16",
            "gb18030",
        }:
            raise InputRejectedError("source_encoding is not supported")

        source = _validated_input_source(source_path)
        try:
            initial_stat = source.stat()
        except OSError as error:
            raise InputRejectedError(
                "input metadata could not be read"
            ) from error
        initial_size = initial_stat.st_size
        if initial_size > max_bytes:
            raise InputRejectedError(
                f"input exceeds the configured {max_bytes}-byte limit"
            )

        with self._lock:
            session = self.session_path(session_id)
            inputs = safe_child(session, "inputs")
            metadata_directory = safe_child(session, "artifacts")
            _require_directory(inputs)
            _require_directory(metadata_directory)

            artifact_id, artifact_directory = _create_unique_directory(
                inputs,
                "art",
            )
            filename = _sanitize_input_filename(source.name)
            destination = safe_child(artifact_directory, filename)
            temporary = safe_child(
                artifact_directory,
                f".{uuid.uuid4().hex}.copying",
            )

            digest = hashlib.sha256()
            size_bytes = 0
            try:
                with temporary.open("xb") as output:
                    if source_encoding is None:
                        with source.open("rb") as source_stream:
                            while True:
                                chunk = source_stream.read(_COPY_CHUNK_SIZE)
                                if not chunk:
                                    break
                                size_bytes += len(chunk)
                                if size_bytes > max_bytes:
                                    raise InputRejectedError(
                                        "input exceeds the configured "
                                        f"{max_bytes}-byte limit"
                                    )
                                digest.update(chunk)
                                output.write(chunk)
                    else:
                        with source.open(
                            "r",
                            encoding=source_encoding,
                            errors="strict",
                            newline="",
                        ) as source_stream:
                            while True:
                                text = source_stream.read(_COPY_CHUNK_SIZE)
                                if not text:
                                    break
                                chunk = text.encode("utf-8")
                                size_bytes += len(chunk)
                                if size_bytes > max_bytes:
                                    raise InputRejectedError(
                                        "transcoded input exceeds the configured "
                                        f"{max_bytes}-byte limit"
                                    )
                                digest.update(chunk)
                                output.write(chunk)
                    output.flush()
                    os.fsync(output.fileno())
                try:
                    final_stat = source.stat()
                except OSError as error:
                    raise InputRejectedError(
                        "input changed while it was being copied"
                    ) from error
                if (
                    final_stat.st_size != initial_stat.st_size
                    or final_stat.st_mtime_ns != initial_stat.st_mtime_ns
                    or final_stat.st_ctime_ns != initial_stat.st_ctime_ns
                    or final_stat.st_dev != initial_stat.st_dev
                    or final_stat.st_ino != initial_stat.st_ino
                ):
                    raise InputRejectedError(
                        "input changed while it was being copied"
                    )
                _publish_new_file(temporary, destination)
            finally:
                if temporary.exists():
                    temporary.unlink()

            record = ArtifactRecord(
                artifact_id=artifact_id,
                kind="input",
                display_path=(
                    PurePosixPath("inputs") / artifact_id / filename
                ).as_posix(),
                sha256=digest.hexdigest(),
                size_bytes=size_bytes,
            )
            self._write_artifact_metadata(session_id, record)
            return record

    def resolve_artifact(
        self,
        session_id: str,
        artifact_id: str,
        *,
        verify: bool = True,
    ) -> Path:
        """Resolve an opaque artifact ID without allowing workspace escape."""

        session = self.session_path(session_id)
        record = self.get_artifact(session_id, artifact_id)
        relative = PurePosixPath(record.display_path)
        path = safe_child(session, *relative.parts)
        _require_regular_file(path)
        if verify:
            actual_size, actual_hash = _hash_file(path)
            if actual_size != record.size_bytes or actual_hash != record.sha256:
                raise ArtifactIntegrityError(
                    f"artifact {record.artifact_id} no longer matches its metadata"
                )
        return path

    def get_artifact(self, session_id: str, artifact_id: str) -> ArtifactRecord:
        """Load and validate one persisted artifact record."""

        identifier = validate_identifier(artifact_id, "artifact_id")
        session = self.session_path(session_id)
        metadata_path = safe_child(session, "artifacts", f"{identifier}.json")
        if not metadata_path.exists():
            raise ArtifactNotFoundError(f"unknown artifact_id {identifier}")
        payload = read_json_file(metadata_path)
        record = _artifact_record_from_metadata(
            payload,
            expected_session_id=validate_identifier(session_id, "session_id"),
            expected_artifact_id=identifier,
        )
        return record

    def list_artifacts(
        self,
        session_id: str,
        *,
        kind: str | None = None,
    ) -> tuple[ArtifactRecord, ...]:
        """Return records in deterministic artifact-ID order."""

        session = self.session_path(session_id)
        metadata_directory = safe_child(session, "artifacts")
        records: list[ArtifactRecord] = []
        for path in sorted(metadata_directory.glob("*.json")):
            _require_regular_file(path)
            artifact_id = path.stem
            validate_identifier(artifact_id, "artifact_id")
            record = self.get_artifact(session_id, artifact_id)
            if kind is None or record.kind == kind:
                records.append(record)
        return tuple(records)

    def create_run(
        self,
        session_id: str,
        *,
        idempotency_key: str | None = None,
    ) -> RunDirectory:
        """Create one isolated run directory, idempotently when keyed."""

        normalized_key = (
            None
            if idempotency_key is None
            else validate_identifier(idempotency_key, "idempotency_key")
        )
        with self._lock:
            session = self.session_path(session_id)
            runs = safe_child(session, "runs")
            _require_directory(runs)
            if normalized_key is not None:
                existing = self._find_run_by_idempotency_key(
                    session_id,
                    normalized_key,
                )
                if existing is not None:
                    return existing

            run_id, run_path = _create_unique_directory(runs, "run")
            for name in ("exports", "logs"):
                safe_child(run_path, name).mkdir(mode=0o700)
            metadata = {
                "schema_version": STORE_SCHEMA_VERSION,
                "session_id": validate_identifier(session_id, "session_id"),
                "run_id": run_id,
                "idempotency_key": normalized_key,
            }
            atomic_write_json(safe_child(run_path, ".run.json"), metadata)
            return RunDirectory(
                run_id=run_id,
                path=run_path,
                display_path=(PurePosixPath("runs") / run_id).as_posix(),
            )

    def run_directory(self, session_id: str, run_id: str) -> RunDirectory:
        """Resolve and validate one run directory and its metadata."""

        identifier = validate_identifier(run_id, "run_id")
        session = self.session_path(session_id)
        path = safe_child(session, "runs", identifier)
        if not path.exists():
            raise ArtifactNotFoundError(f"unknown run_id {identifier}")
        _require_directory(path)
        metadata_path = safe_child(path, ".run.json")
        payload = read_json_file(metadata_path)
        expected = {
            "schema_version",
            "session_id",
            "run_id",
            "idempotency_key",
        }
        if set(payload) != expected:
            raise ArtifactIntegrityError("run metadata has invalid fields")
        if payload["schema_version"] != STORE_SCHEMA_VERSION:
            raise ArtifactIntegrityError("run metadata has an unsupported version")
        if payload["session_id"] != session_id or payload["run_id"] != identifier:
            raise ArtifactIntegrityError("run metadata identity mismatch")
        key = payload["idempotency_key"]
        if key is not None:
            validate_identifier(key, "idempotency_key")
        return RunDirectory(
            run_id=identifier,
            path=path,
            display_path=(PurePosixPath("runs") / identifier).as_posix(),
        )

    def register_run_artifact(
        self,
        session_id: str,
        run_id: str,
        path: str | os.PathLike[str],
        *,
        kind: str,
    ) -> ArtifactRecord:
        """Register an existing regular file confined to an active run."""

        if not isinstance(kind, str) or not kind.strip() or "\x00" in kind:
            raise ValueError("kind must be a non-blank string without NUL")
        run = self.run_directory(session_id, run_id)
        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = safe_child(run.path, candidate)
        else:
            candidate = candidate.resolve(strict=False)
            if not candidate.is_relative_to(run.path):
                raise UnsafePathError("run artifact path escapes the run directory")
        _require_regular_file(candidate)
        relative = candidate.relative_to(run.path)
        if relative == Path(".run.json"):
            raise UnsafePathError("run metadata cannot be registered as an artifact")
        size_bytes, digest = _hash_file(candidate)

        with self._lock:
            session = self.session_path(session_id)
            metadata_directory = safe_child(session, "artifacts")
            artifact_id = _unused_artifact_id(metadata_directory)
            record = ArtifactRecord(
                artifact_id=artifact_id,
                kind=kind.strip(),
                display_path=(
                    PurePosixPath("runs") / run.run_id / PurePosixPath(relative.as_posix())
                ).as_posix(),
                sha256=digest,
                size_bytes=size_bytes,
            )
            self._write_artifact_metadata(session_id, record)
            return record

    def _write_artifact_metadata(
        self,
        session_id: str,
        record: ArtifactRecord,
    ) -> None:
        session = self.session_path(session_id)
        metadata_path = safe_child(
            session,
            "artifacts",
            f"{record.artifact_id}.json",
        )
        record_payload = record.to_dict()
        record_payload.pop("schema_version", None)
        payload = {
            "schema_version": STORE_SCHEMA_VERSION,
            "session_id": validate_identifier(session_id, "session_id"),
            "record": record_payload,
        }
        atomic_write_json(metadata_path, payload)

    def _find_run_by_idempotency_key(
        self,
        session_id: str,
        idempotency_key: str,
    ) -> RunDirectory | None:
        session = self.session_path(session_id)
        runs = safe_child(session, "runs")
        for path in sorted(runs.iterdir()):
            if not path.is_dir():
                raise ArtifactIntegrityError("run storage contains a non-directory")
            run = self.run_directory(session_id, path.name)
            payload = read_json_file(safe_child(run.path, ".run.json"))
            if payload["idempotency_key"] == idempotency_key:
                return run
        return None


def normalize_workspace(workspace: str | os.PathLike[str]) -> Path:
    """Create and return the canonical workspace directory."""

    raw = Path(workspace).expanduser()
    raw.mkdir(mode=0o700, parents=True, exist_ok=True)
    resolved = raw.resolve(strict=True)
    _require_directory(resolved, allow_root_reparse=True)
    return resolved


def validate_identifier(value: Any, name: str) -> str:
    """Validate an opaque identifier before using it in a path."""

    if not isinstance(value, str) or _IDENTIFIER_PATTERN.fullmatch(value) is None:
        raise InvalidIdentifierError(
            f"{name} may contain only 1-128 letters, digits, underscores, or hyphens"
        )
    return value


def safe_child(root: Path, *parts: str | os.PathLike[str]) -> Path:
    """Resolve a child and reject absolute, traversal, and linked escapes."""

    canonical_root = root.resolve(strict=True)
    candidate = canonical_root
    for raw_part in parts:
        part = Path(raw_part)
        if part.is_absolute() or part.drive or part.anchor:
            raise UnsafePathError("absolute paths are not allowed inside the store")
        candidate = candidate / part
    resolved = candidate.resolve(strict=False)
    if not resolved.is_relative_to(canonical_root):
        raise UnsafePathError("path escapes the configured workspace")
    return resolved


def ensure_session_directory(
    workspace_root: Path,
    session_id: str,
    *,
    create: bool,
) -> Path:
    """Return a safe session directory, creating it only when requested."""

    identifier = validate_identifier(session_id, "session_id")
    sessions = safe_child(workspace_root, "sessions")
    if create:
        sessions.mkdir(mode=0o700, exist_ok=True)
    if not sessions.exists():
        raise SessionNotFoundError(f"unknown session_id {identifier}")
    _require_directory(sessions)
    session = safe_child(sessions, identifier)
    if create:
        session.mkdir(mode=0o700, exist_ok=True)
    if not session.exists():
        raise SessionNotFoundError(f"unknown session_id {identifier}")
    _require_directory(session)
    return session


def atomic_write_json(
    path: Path,
    payload: Mapping[str, Any],
    *,
    overwrite: bool = False,
) -> None:
    """Write UTF-8 JSON atomically, optionally replacing an owned file."""

    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ) + "\n"
    parent = path.parent.resolve(strict=True)
    target = safe_child(parent, path.name)
    temporary = safe_child(parent, f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(serialized)
            stream.flush()
            os.fsync(stream.fileno())
        if overwrite:
            os.replace(temporary, target)
        else:
            _publish_new_file(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_write_bytes(
    path: Path,
    payload: bytes,
    *,
    overwrite: bool = False,
) -> None:
    """Write binary data atomically, optionally replacing an owned file."""

    if not isinstance(payload, bytes):
        raise TypeError("payload must be bytes")
    parent = path.parent.resolve(strict=True)
    target = safe_child(parent, path.name)
    temporary = safe_child(parent, f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        if overwrite:
            os.replace(temporary, target)
        else:
            _publish_new_file(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


def read_json_file(
    path: Path,
    *,
    max_bytes: int = _REVISION_METADATA_LIMIT,
) -> dict[str, Any]:
    """Read a small regular UTF-8 JSON object from a confined path."""

    _require_regular_file(path)
    size = path.stat().st_size
    if size > max_bytes:
        raise ArtifactIntegrityError("metadata file exceeds the allowed size")
    try:
        text = path.read_text(encoding="utf-8")
        payload = json.loads(text)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ArtifactIntegrityError("metadata is not valid UTF-8 JSON") from exc
    if not isinstance(payload, dict):
        raise ArtifactIntegrityError("metadata JSON must contain an object")
    return payload


def _new_identifier(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def _unused_artifact_id(metadata_directory: Path) -> str:
    for _ in range(128):
        identifier = _new_identifier("art")
        path = safe_child(metadata_directory, f"{identifier}.json")
        if not path.exists():
            return identifier
    raise ArtifactStoreError("could not allocate a unique artifact identifier")


def _create_unique_directory(parent: Path, prefix: str) -> tuple[str, Path]:
    for _ in range(128):
        identifier = _new_identifier(prefix)
        path = safe_child(parent, identifier)
        try:
            path.mkdir(mode=0o700)
        except FileExistsError:
            continue
        _require_directory(path)
        return identifier, path
    raise ArtifactStoreError("could not allocate a unique directory")


def _validated_input_source(path: str | os.PathLike[str]) -> Path:
    raw = Path(path).expanduser()
    if not raw.exists():
        raise InputRejectedError("input file does not exist")
    if _is_reparse_point(raw):
        raise InputRejectedError("input file must not be a symbolic link")
    try:
        resolved = raw.resolve(strict=True)
    except OSError as exc:
        raise InputRejectedError("input file cannot be resolved") from exc
    try:
        _require_regular_file(resolved)
    except UnsafePathError as exc:
        raise InputRejectedError("input must be a regular file") from exc
    if resolved.suffix.lower() != ".inp":
        raise InputRejectedError("input file must use the .inp extension")
    return resolved


def _sanitize_input_filename(filename: str) -> str:
    name = Path(filename).name
    characters = []
    for character in name:
        if (
            unicodedata.category(character) in {"Cc", "Cf", "Cs", "Zl", "Zp"}
            or character in '<>:"/\\|?*'
        ):
            characters.append("_")
        else:
            characters.append(character)
    sanitized = "".join(characters).rstrip(" .")
    stem = Path(sanitized).stem.rstrip(" .")
    if not stem:
        stem = "input"
    if stem.upper() in _WINDOWS_RESERVED_NAMES:
        stem = f"_{stem}"
    stem = stem[:180]
    return f"{stem}.inp"


def _artifact_record_from_metadata(
    payload: Mapping[str, Any],
    *,
    expected_session_id: str,
    expected_artifact_id: str,
) -> ArtifactRecord:
    if set(payload) != {"schema_version", "session_id", "record"}:
        raise ArtifactIntegrityError("artifact metadata has invalid fields")
    if payload["schema_version"] != STORE_SCHEMA_VERSION:
        raise ArtifactIntegrityError("artifact metadata has an unsupported version")
    if payload["session_id"] != expected_session_id:
        raise ArtifactIntegrityError("artifact metadata session mismatch")
    raw_record = payload["record"]
    if not isinstance(raw_record, Mapping):
        raise ArtifactIntegrityError("artifact metadata record must be an object")
    normalized = dict(raw_record)
    normalized.pop("schema_version", None)
    try:
        record = ArtifactRecord.from_dict(normalized)
    except (TypeError, ValueError) as exc:
        raise ArtifactIntegrityError("artifact metadata record is invalid") from exc
    if record.artifact_id != expected_artifact_id:
        raise ArtifactIntegrityError("artifact metadata identifier mismatch")
    return record


def _publish_new_file(temporary: Path, target: Path) -> None:
    """Atomically publish a completed file without replacing an existing one."""

    try:
        os.link(temporary, target)
    except FileExistsError:
        raise
    except OSError as exc:
        raise ArtifactStoreError("filesystem cannot atomically publish a new file") from exc
    temporary.unlink()


def _hash_file(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(_COPY_CHUNK_SIZE)
            if not chunk:
                break
            size += len(chunk)
            digest.update(chunk)
    return size, digest.hexdigest()


def _require_directory(path: Path, *, allow_root_reparse: bool = False) -> None:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise UnsafePathError("required directory cannot be inspected") from exc
    if not stat.S_ISDIR(metadata.st_mode):
        raise UnsafePathError("required path is not a directory")
    if not allow_root_reparse and _is_reparse_metadata(metadata):
        raise UnsafePathError("linked directories are not allowed in the store")


def _require_regular_file(path: Path) -> None:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise UnsafePathError("required file cannot be inspected") from exc
    if not stat.S_ISREG(metadata.st_mode) or _is_reparse_metadata(metadata):
        raise UnsafePathError("required path is not a regular unlinked file")


def _is_reparse_point(path: Path) -> bool:
    try:
        return _is_reparse_metadata(path.lstat())
    except OSError:
        return False


def _is_reparse_metadata(metadata: os.stat_result) -> bool:
    attributes = getattr(metadata, "st_file_attributes", 0)
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    return stat.S_ISLNK(metadata.st_mode) or bool(attributes & reparse_flag)


__all__ = [
    "ArtifactIntegrityError",
    "ArtifactNotFoundError",
    "ArtifactRecord",
    "ArtifactStore",
    "ArtifactStoreError",
    "InputRejectedError",
    "InvalidIdentifierError",
    "RunDirectory",
    "SessionNotFoundError",
    "UnsafePathError",
]
