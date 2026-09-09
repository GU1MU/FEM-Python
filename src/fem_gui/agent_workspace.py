"""Local workspace boundary for the FEM Agent GUI.

Manage user workspace paths, a bounded file metadata index, and the local ``/workspace`` command.
This module does not read file contents, create Agent session directories, or depend on ``fem_agent``.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import uuid
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

from PySide6.QtCore import QStandardPaths
from PySide6.QtWidgets import QFileDialog, QWidget


MAX_WORKSPACE_SCAN_ENTRIES = 2_500
MAX_WORKSPACE_INDEX_FILES = 500
MAX_VISIBLE_WORKSPACE_CANDIDATES = 40
_WINDOWS_REPARSE_POINT_ATTRIBUTE = 0x0400


class WorkspacePathError(ValueError):
    """The workspace or candidate file violates the local path boundary."""


@dataclass(frozen=True, slots=True)
class UserWorkspace:
    """Normalized user workspace identity."""

    workspace_id: str
    root: Path


@dataclass(frozen=True, slots=True)
class WorkspaceFileReference:
    """Workspace file metadata and its reference position in the input text."""

    workspace_id: str
    workspace_root: str
    relative_path: str
    filename: str
    file_type: str
    size_bytes: int
    modified_time_ns: int
    metadata_version: str
    mention_start: int | None = None
    mention_end: int | None = None

    @property
    def mention_text(self) -> str:
        return f"@{self.relative_path}"

    def at_text_range(
        self,
        start: int,
        end: int,
    ) -> WorkspaceFileReference:
        return replace(
            self,
            mention_start=int(start),
            mention_end=int(end),
        )


@dataclass(frozen=True, slots=True)
class WorkspaceIndexSnapshot:
    """Immutable index from a bounded scan."""

    workspace: UserWorkspace
    files: tuple[WorkspaceFileReference, ...]
    scanned_entries: int
    skipped_unsafe_entries: int
    truncated: bool

    def matching_files(
        self,
        query: str,
        *,
        limit: int = MAX_VISIBLE_WORKSPACE_CANDIDATES,
    ) -> tuple[WorkspaceFileReference, ...]:
        needle = query.strip().replace("\\", "/").casefold()
        matches = (
            reference
            for reference in self.files
            if not needle
            or needle in reference.relative_path.casefold()
            or needle in reference.filename.casefold()
        )
        return tuple(list(matches)[: max(0, int(limit))])


@dataclass(frozen=True, slots=True)
class WorkspaceSelectionResult:
    """Local workspace command result; cancellation preserves the previous state."""

    cancelled: bool
    workspace: UserWorkspace | None
    index: WorkspaceIndexSnapshot | None
    error: str | None = None

    @property
    def succeeded(self) -> bool:
        return (
            not self.cancelled
            and self.error is None
            and self.workspace is not None
            and self.index is not None
        )


DirectoryChooser = Callable[[QWidget | None, str], str | Path | None]


def resolve_agent_data_root(
    explicit_root: str | os.PathLike[str] | None = None,
) -> Path:
    """Return the private Agent data root without creating directories."""
    if explicit_root is None:
        local_data = QStandardPaths.writableLocation(
            QStandardPaths.StandardLocation.AppLocalDataLocation
        )
        if not local_data:
            raise WorkspacePathError("Qt did not provide a private application data directory")
        explicit_root = Path(local_data) / "fem-agent"
    return Path(explicit_root).expanduser().resolve(strict=False)


def normalize_user_workspace(
    selected_path: str | os.PathLike[str],
) -> UserWorkspace:
    """Validate and normalize an existing user directory."""
    try:
        root = Path(selected_path).expanduser().resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise WorkspacePathError("Selected workspace does not exist or is inaccessible") from exc
    if not root.is_dir():
        raise WorkspacePathError("Selected workspace is not a directory")

    identity_source = os.path.normcase(str(root)).replace("\\", "/")
    workspace_id = hashlib.sha256(
        identity_source.encode("utf-8")
    ).hexdigest()[:20]
    return UserWorkspace(workspace_id=workspace_id, root=root)


def _has_reparse_attribute(file_stat: os.stat_result) -> bool:
    attributes = int(getattr(file_stat, "st_file_attributes", 0))
    return bool(attributes & _WINDOWS_REPARSE_POINT_ATTRIBUTE)


def _is_unsafe_link(file_stat: os.stat_result) -> bool:
    return stat.S_ISLNK(file_stat.st_mode) or _has_reparse_attribute(file_stat)


def _relative_path_inside(
    workspace_root: Path,
    candidate: Path,
) -> Path:
    absolute = Path(os.path.abspath(candidate))
    try:
        return absolute.relative_to(workspace_root)
    except ValueError as exc:
        raise WorkspacePathError("File path is outside the user workspace") from exc


def build_workspace_file_reference(
    workspace: UserWorkspace,
    candidate: str | os.PathLike[str],
) -> WorkspaceFileReference:
    """Build a reference from regular file metadata without reading its contents."""
    candidate_path = Path(candidate)
    if not candidate_path.is_absolute():
        candidate_path = workspace.root / candidate_path
    relative = _relative_path_inside(workspace.root, candidate_path)
    if not relative.parts:
        raise WorkspacePathError("Workspace root is not a referenceable file")

    current = workspace.root
    final_stat: os.stat_result | None = None
    for part in relative.parts:
        current = current / part
        try:
            final_stat = os.lstat(current)
        except OSError as exc:
            raise WorkspacePathError("File does not exist or is inaccessible") from exc
        if _is_unsafe_link(final_stat):
            raise WorkspacePathError("Cannot reference symbolic links or reparse points")

    try:
        resolved = current.resolve(strict=True)
        resolved.relative_to(workspace.root)
    except (OSError, RuntimeError, ValueError) as exc:
        raise WorkspacePathError("Resolved file is outside the user workspace") from exc
    if final_stat is None or not stat.S_ISREG(final_stat.st_mode):
        raise WorkspacePathError("Only regular files can be referenced")

    relative_text = relative.as_posix()
    file_type = current.suffix.casefold().lstrip(".") or "file"
    modified_time_ns = int(final_stat.st_mtime_ns)
    size_bytes = int(final_stat.st_size)
    version_source = (
        f"{size_bytes}:{modified_time_ns}:"
        f"{int(getattr(final_stat, 'st_dev', 0))}:"
        f"{int(getattr(final_stat, 'st_ino', 0))}"
    )
    metadata_version = hashlib.sha256(
        version_source.encode("ascii")
    ).hexdigest()[:20]
    return WorkspaceFileReference(
        workspace_id=workspace.workspace_id,
        workspace_root=str(workspace.root),
        relative_path=relative_text,
        filename=current.name,
        file_type=file_type,
        size_bytes=size_bytes,
        modified_time_ns=modified_time_ns,
        metadata_version=metadata_version,
    )


def _path_is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


# ---------------------------------------------------------------------------
# Export ledger
# ---------------------------------------------------------------------------

EXPORT_LEDGER_DIRECTORY_NAME = "export_ledgers"
EXPORT_LEDGER_SCHEMA_VERSION = "1.0"
_EXPORT_LEDGER_MAX_RECORDS = 500
_EXPORT_LEDGER_KINDS = {"csv", "png"}


class ExportLedgerRecordError(ValueError):
    """The export ledger record violates the bounded contract."""


def _bounded_ledger_text(value: object, name: str, maximum: int) -> str:
    if not isinstance(value, str) or len(value) > maximum:
        raise ExportLedgerRecordError(f"{name} must be a string of at most {maximum} characters")
    return value


def _bounded_ledger_integer(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ExportLedgerRecordError(f"{name} must be a non-negative integer")
    return value


@dataclass(frozen=True, slots=True)
class ExportLedgerRecord:
    """Ledger record for an Agent export written to disk."""

    display_path: str
    kind: str
    sha256: str
    size_bytes: int
    document_id: str
    session_id: str
    run_id: str
    materialization_generation: int
    overrides_summary: str
    exported_at: str
    tool: str

    def __post_init__(self) -> None:
        _bounded_ledger_text(self.display_path, "display_path", 512)
        if self.kind not in _EXPORT_LEDGER_KINDS:
            raise ExportLedgerRecordError("kind must be csv or png")
        digest = self.sha256
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ExportLedgerRecordError("sha256 must be 64 lowercase hexadecimal characters")
        _bounded_ledger_integer(self.size_bytes, "size_bytes")
        _bounded_ledger_text(self.document_id, "document_id", 128)
        _bounded_ledger_text(self.session_id, "session_id", 128)
        _bounded_ledger_text(self.run_id, "run_id", 128)
        _bounded_ledger_integer(
            self.materialization_generation,
            "materialization_generation",
        )
        _bounded_ledger_text(self.overrides_summary, "overrides_summary", 256)
        _bounded_ledger_text(self.exported_at, "exported_at", 40)
        _bounded_ledger_text(self.tool, "tool", 64)

    def to_dict(self) -> dict[str, object]:
        return {
            "display_path": self.display_path,
            "kind": self.kind,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "document_id": self.document_id,
            "session_id": self.session_id,
            "run_id": self.run_id,
            "materialization_generation": self.materialization_generation,
            "overrides_summary": self.overrides_summary,
            "exported_at": self.exported_at,
            "tool": self.tool,
        }

    @classmethod
    def from_dict(cls, payload: object) -> ExportLedgerRecord:
        if not isinstance(payload, dict):
            raise ExportLedgerRecordError("Ledger record must be an object")
        return cls(
            display_path=payload.get("display_path", ""),
            kind=payload.get("kind", ""),
            sha256=payload.get("sha256", ""),
            size_bytes=payload.get("size_bytes", -1),
            document_id=payload.get("document_id", ""),
            session_id=payload.get("session_id", ""),
            run_id=payload.get("run_id", ""),
            materialization_generation=payload.get(
                "materialization_generation", -1
            ),
            overrides_summary=payload.get("overrides_summary", ""),
            exported_at=payload.get("exported_at", ""),
            tool=payload.get("tool", ""),
        )


def _validate_workspace_identity(workspace_id: str) -> str:
    if (
        not isinstance(workspace_id, str)
        or not workspace_id
        or len(workspace_id) > 64
        or any(
            not (character.isascii() and (character.isalnum() or character in "-_"))
            for character in workspace_id
        )
    ):
        raise WorkspacePathError("workspace_id is not a valid ledger key")
    return workspace_id


def export_ledger_path(
    agent_data_root: str | os.PathLike[str],
    workspace_id: str,
) -> Path:
    """Return the workspace ledger path without creating directories."""
    identity = _validate_workspace_identity(workspace_id)
    return (
        Path(agent_data_root)
        / EXPORT_LEDGER_DIRECTORY_NAME
        / f"{identity}.json"
    )


def read_export_ledger(
    agent_data_root: str | os.PathLike[str],
    workspace_id: str,
) -> tuple[ExportLedgerRecord, ...]:
    """Read a workspace ledger; treat missing or corrupt data as empty."""
    path = export_ledger_path(agent_data_root, workspace_id)
    try:
        text = path.read_text(encoding="utf-8")
        payload = json.loads(text)
    except (OSError, UnicodeDecodeError, ValueError):
        return ()
    if not isinstance(payload, dict):
        return ()
    raw_records = payload.get("records")
    if not isinstance(raw_records, list):
        return ()
    records: list[ExportLedgerRecord] = []
    for raw_record in raw_records:
        try:
            records.append(ExportLedgerRecord.from_dict(raw_record))
        except ExportLedgerRecordError:
            continue
    return tuple(records[-_EXPORT_LEDGER_MAX_RECORDS:])


def append_export_ledger_record(
    agent_data_root: str | os.PathLike[str],
    workspace_id: str,
    record: ExportLedgerRecord,
) -> Path:
    """Append a ledger record by reading, merging, and writing via a temporary file and rename."""
    path = export_ledger_path(agent_data_root, workspace_id)
    existing = read_export_ledger(agent_data_root, workspace_id)
    records = (*existing, record)[-_EXPORT_LEDGER_MAX_RECORDS:]
    payload = {
        "schema_version": EXPORT_LEDGER_SCHEMA_VERSION,
        "workspace_id": workspace_id,
        "records": [item.to_dict() for item in records],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    serialized = json.dumps(payload, ensure_ascii=False, indent=1) + "\n"
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(serialized)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            try:
                temporary.unlink()
            except OSError:
                pass
    return path


class WorkspaceIndexer:
    """Regular file metadata indexer that scans a bounded number of entries."""

    def __init__(
        self,
        *,
        max_entries: int = MAX_WORKSPACE_SCAN_ENTRIES,
        max_files: int = MAX_WORKSPACE_INDEX_FILES,
    ) -> None:
        self.max_entries = max(1, int(max_entries))
        self.max_files = max(1, int(max_files))

    def scan(
        self,
        workspace: UserWorkspace,
        *,
        excluded_roots: Sequence[Path] = (),
    ) -> WorkspaceIndexSnapshot:
        excluded = tuple(
            Path(path).resolve(strict=False)
            for path in excluded_roots
        )
        pending: deque[Path] = deque((workspace.root,))
        files: list[WorkspaceFileReference] = []
        scanned_entries = 0
        skipped_unsafe_entries = 0
        truncated = False

        while pending and not truncated:
            directory = pending.popleft()
            if directory != workspace.root:
                try:
                    directory_stat = os.lstat(directory)
                except OSError:
                    continue
                if _is_unsafe_link(directory_stat):
                    skipped_unsafe_entries += 1
                    continue
            if any(_path_is_within(directory, root) for root in excluded):
                continue

            try:
                iterator = os.scandir(directory)
            except OSError:
                continue
            with iterator:
                for entry in iterator:
                    if scanned_entries >= self.max_entries:
                        truncated = True
                        break
                    scanned_entries += 1
                    try:
                        entry_stat = entry.stat(follow_symlinks=False)
                    except OSError:
                        continue
                    if _is_unsafe_link(entry_stat):
                        skipped_unsafe_entries += 1
                        continue

                    entry_path = Path(entry.path)
                    if any(
                        _path_is_within(
                            entry_path.resolve(strict=False),
                            root,
                        )
                        for root in excluded
                    ):
                        continue
                    if stat.S_ISDIR(entry_stat.st_mode):
                        pending.append(entry_path)
                        continue
                    if not stat.S_ISREG(entry_stat.st_mode):
                        continue
                    try:
                        reference = build_workspace_file_reference(
                            workspace,
                            entry_path,
                        )
                    except WorkspacePathError:
                        skipped_unsafe_entries += 1
                        continue
                    files.append(reference)
                    if len(files) >= self.max_files:
                        truncated = True
                        break

        files.sort(key=lambda item: item.relative_path.casefold())
        return WorkspaceIndexSnapshot(
            workspace=workspace,
            files=tuple(files),
            scanned_entries=scanned_entries,
            skipped_unsafe_entries=skipped_unsafe_entries,
            truncated=truncated,
        )


def choose_workspace_directory(
    parent: QWidget | None,
    initial_directory: str,
) -> str:
    """Qt directory picker boundary for injecting a headless implementation in tests."""
    return QFileDialog.getExistingDirectory(
        parent,
        "Select workspace",
        initial_directory,
        QFileDialog.Option.ShowDirsOnly,
    )


class WorkspaceCommandHandler:
    """Single workspace command handler shared by ``/workspace`` and the add menu."""

    def __init__(
        self,
        *,
        directory_chooser: DirectoryChooser | None = None,
        indexer: WorkspaceIndexer | None = None,
        agent_data_root: str | os.PathLike[str] | None = None,
    ) -> None:
        self.directory_chooser = (
            directory_chooser or choose_workspace_directory
        )
        self.indexer = indexer or WorkspaceIndexer()
        self.agent_data_root = resolve_agent_data_root(agent_data_root)
        self.user_workspace: UserWorkspace | None = None
        self.workspace_index: WorkspaceIndexSnapshot | None = None
        self.execution_count = 0

    def execute(
        self,
        command: str,
        *,
        parent: QWidget | None = None,
    ) -> WorkspaceSelectionResult:
        """Execute a local command; currently only ``/workspace`` is supported."""
        if command.strip().casefold() != "/workspace":
            raise ValueError(f"Unknown local command: {command}")
        self.execution_count += 1
        initial_directory = (
            str(self.user_workspace.root)
            if self.user_workspace is not None
            else ""
        )
        selected = self.directory_chooser(parent, initial_directory)
        if selected is None or not str(selected).strip():
            return WorkspaceSelectionResult(
                cancelled=True,
                workspace=self.user_workspace,
                index=self.workspace_index,
            )

        try:
            workspace = normalize_user_workspace(selected)
            if (
                _path_is_within(workspace.root, self.agent_data_root)
                or _path_is_within(self.agent_data_root, workspace.root)
            ):
                raise WorkspacePathError(
                    "User workspace cannot overlap the private Agent data directory"
                )
            snapshot = self.indexer.scan(
                workspace,
                excluded_roots=(self.agent_data_root,),
            )
        except WorkspacePathError as exc:
            return WorkspaceSelectionResult(
                cancelled=False,
                workspace=self.user_workspace,
                index=self.workspace_index,
                error=str(exc),
            )

        self.user_workspace = workspace
        self.workspace_index = snapshot
        return WorkspaceSelectionResult(
            cancelled=False,
            workspace=workspace,
            index=snapshot,
        )
