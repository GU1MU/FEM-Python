"""Landing foundation for Agent export files inside a user workspace.

All Agent-authored exports land flat in ``<workspace>/agent_exports``.  This
module owns the shared landing rules: filename sanitization (reusing the
artifact-store cleaning rules, Chinese characters included), Explorer-style
conflict incrementing that never overwrites an existing file, pre-write path
confinement with symlink rejection, and the fail-closed CSV byte limit.
A vanished or failing workspace raises ``ExportStorageError`` or ``OSError``
directly; callers only diagnose and never clean up or retry.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path

from .artifacts import _hash_file, _sanitize_filename_stem


EXPORT_DIRECTORY_NAME = "agent_exports"
MAX_EXPORT_CSV_BYTES = 64 * 1024 * 1024
# 视口图片上限：4 倍质量截图的保守字节预算，超限同样 fail-closed。
MAX_EXPORT_IMAGE_BYTES = 256 * 1024 * 1024
MAX_EXPORT_NAME_ATTEMPTS = 500
_FALLBACK_EXPORT_STEM = "export"
_WINDOWS_REPARSE_POINT_ATTRIBUTE = 0x0400


class ExportStorageError(ValueError):
    """Fail-closed violation of the export landing boundary."""


def sanitize_export_filename(name: str, extension: str) -> str:
    """Clean one requested export name into ``<stem><extension>``.

    Applies the artifact-store cleaning rules (control characters and Windows
    separators become ``_``, reserved device names gain a ``_`` prefix,
    Chinese characters stay untouched) and falls back to a neutral stem when
    nothing usable remains.
    """

    suffix = extension if extension.startswith(".") else f".{extension}"
    stem = _sanitize_filename_stem(name) or _FALLBACK_EXPORT_STEM
    return f"{stem}{suffix.casefold()}"


def _is_unsafe_link(path: Path) -> bool:
    file_stat = os.lstat(path)
    attributes = int(getattr(file_stat, "st_file_attributes", 0))
    return stat.S_ISLNK(file_stat.st_mode) or bool(
        attributes & _WINDOWS_REPARSE_POINT_ATTRIBUTE
    )


def ensure_export_directory(workspace_root: Path) -> Path:
    """Return the flat export directory, creating it on first use.

    The workspace root itself must be a real directory and must not be a
    symbolic link; the export directory is created only when missing.
    """

    root = Path(workspace_root)
    try:
        root_stat = os.lstat(root)
    except OSError as exc:
        raise ExportStorageError("workspace directory is not accessible") from exc
    if not stat.S_ISDIR(root_stat.st_mode):
        raise ExportStorageError("workspace root is not a directory")
    if _is_unsafe_link(root):
        raise ExportStorageError("workspace root must not be a symbolic link")
    exports = root / EXPORT_DIRECTORY_NAME
    try:
        exports.mkdir(exist_ok=True)
    except OSError as exc:
        raise ExportStorageError("export directory cannot be created") from exc
    if _is_unsafe_link(exports):
        raise ExportStorageError("export directory must not be a symbolic link")
    if not exports.is_dir():
        raise ExportStorageError("export directory path is not a directory")
    return exports


def _relative_path_inside(root: Path, candidate: Path) -> Path:
    absolute = Path(os.path.abspath(candidate))
    try:
        return absolute.relative_to(root)
    except ValueError as exc:
        raise ExportStorageError("export path escapes the user workspace") from exc


def validate_export_target(exports_root: Path, target: Path) -> Path:
    """Confine one export target to a flat regular file inside the directory.

    Mirrors the workspace ``_relative_path_inside`` pattern: the target must
    be a direct child of the export directory, must not be or traverse a
    symbolic link, and must resolve back inside the export directory.
    """

    exports = Path(exports_root).resolve(strict=True)
    relative = _relative_path_inside(exports, target)
    if len(relative.parts) != 1 or not relative.parts[0]:
        raise ExportStorageError("exports must stay flat in agent_exports")
    candidate = exports / relative.parts[0]
    if candidate.exists() or candidate.is_symlink():
        try:
            if _is_unsafe_link(candidate):
                raise ExportStorageError(
                    "export target must not be a symbolic link"
                )
        except OSError as exc:
            raise ExportStorageError("export target cannot be inspected") from exc
    try:
        resolved = candidate.resolve(strict=False)
        resolved.relative_to(exports)
    except (OSError, RuntimeError, ValueError) as exc:
        raise ExportStorageError("export path escapes the user workspace") from exc
    return candidate


def allocate_export_path(exports_root: Path, filename: str) -> Path:
    """Allocate a non-existing target name with Explorer-style incrementing.

    ``a.csv`` falls back to ``a(1).csv``, then ``a(2).csv``, and so on; an
    existing file is never overwritten and existing targets are never reused.
    """

    exports = Path(exports_root)
    name = Path(filename).name
    stem = Path(name).stem
    suffix = Path(name).suffix
    if not stem:
        raise ExportStorageError("export filename is empty")
    candidate = validate_export_target(exports, exports / name)
    if not candidate.exists() and not candidate.is_symlink():
        return candidate
    for index in range(1, MAX_EXPORT_NAME_ATTEMPTS):
        incremented = f"{stem}({index}){suffix}"
        candidate = validate_export_target(exports, exports / incremented)
        if not candidate.exists() and not candidate.is_symlink():
            return candidate
    raise ExportStorageError("could not allocate a unique export filename")


def verify_export_file_size(path: Path, *, maximum_bytes: int) -> tuple[int, str]:
    """Fail-closed size check plus sha256 for one landed export file.

    Files above the bound are removed before raising so a failed export never
    leaves an oversized artifact behind.
    """

    size, digest = _hash_file(Path(path))
    if size > maximum_bytes:
        try:
            Path(path).unlink()
        except OSError:
            pass
        raise ExportStorageError("exported file exceeds the allowed size")
    return size, digest


__all__ = [
    "EXPORT_DIRECTORY_NAME",
    "MAX_EXPORT_CSV_BYTES",
    "MAX_EXPORT_IMAGE_BYTES",
    "ExportStorageError",
    "allocate_export_path",
    "ensure_export_directory",
    "sanitize_export_filename",
    "validate_export_target",
    "verify_export_file_size",
]
