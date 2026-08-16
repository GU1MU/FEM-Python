"""Tests for the workspace export landing foundation (agent_exports)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from fem_agent.export_storage import (
    EXPORT_DIRECTORY_NAME,
    ExportStorageError,
    allocate_export_path,
    ensure_export_directory,
    sanitize_export_filename,
    validate_export_target,
    verify_export_file_size,
)


def _try_symlink(source: Path, link: Path) -> bool:
    try:
        os.symlink(source, link)
    except OSError:
        return False
    return link.is_symlink()


def test_sanitize_keeps_chinese_and_neutralizes_separators() -> None:
    assert sanitize_export_filename("位移结果 run-1", ".csv") == (
        "位移结果 run-1.csv"
    )
    assert sanitize_export_filename("a|b*c", ".csv") == "a_b_c.csv"
    assert sanitize_export_filename("con", ".csv") == "_con.csv"
    assert sanitize_export_filename("///", ".csv") == "export.csv"
    assert sanitize_export_filename("table", "CSV") == "table.csv"


def test_ensure_export_directory_creates_flat_dir_once(tmp_path: Path) -> None:
    exports = ensure_export_directory(tmp_path)
    assert exports == tmp_path / EXPORT_DIRECTORY_NAME
    assert exports.is_dir()
    assert ensure_export_directory(tmp_path) == exports
    with pytest.raises(ExportStorageError):
        ensure_export_directory(tmp_path / "missing")
    (tmp_path / "plain.txt").write_text("x", encoding="utf-8")
    with pytest.raises(ExportStorageError):
        ensure_export_directory(tmp_path / "plain.txt")


def test_ensure_export_directory_rejects_symlinked_root(
    tmp_path: Path,
) -> None:
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    if not _try_symlink(real, link):
        pytest.skip("symlinks are unavailable on this platform")
    with pytest.raises(ExportStorageError):
        ensure_export_directory(link)


def test_allocate_increments_like_explorer_and_never_overwrites(
    tmp_path: Path,
) -> None:
    exports = ensure_export_directory(tmp_path)
    first = allocate_export_path(exports, "a.csv")
    assert first == exports / "a.csv"
    first.write_text("x", encoding="utf-8")
    second = allocate_export_path(exports, "a.csv")
    assert second == exports / "a(1).csv"
    second.write_text("y", encoding="utf-8")
    third = allocate_export_path(exports, "a.csv")
    assert third == exports / "a(2).csv"
    assert first.read_text(encoding="utf-8") == "x"
    assert second.read_text(encoding="utf-8") == "y"
    assert not third.exists()
    with pytest.raises(ExportStorageError):
        allocate_export_path(exports, "")


def test_validate_target_rejects_escape_nesting_and_links(
    tmp_path: Path,
) -> None:
    exports = ensure_export_directory(tmp_path)
    outside = tmp_path.parent / "outside.csv"
    with pytest.raises(ExportStorageError):
        validate_export_target(exports, outside)
    with pytest.raises(ExportStorageError):
        validate_export_target(exports, exports / "nested" / "a.csv")
    with pytest.raises(ExportStorageError):
        validate_export_target(exports, exports)
    link = exports / "link.csv"
    target = tmp_path / "real.csv"
    target.write_text("z", encoding="utf-8")
    if _try_symlink(target, link):
        with pytest.raises(ExportStorageError):
            validate_export_target(exports, link)
        with pytest.raises(ExportStorageError):
            allocate_export_path(exports, "link.csv")


def test_verify_export_file_size_is_fail_closed(tmp_path: Path) -> None:
    file_path = tmp_path / "small.csv"
    file_path.write_bytes(b"column\n1\n")
    size, digest = verify_export_file_size(file_path, maximum_bytes=64)
    assert size == len(b"column\n1\n")
    assert len(digest) == 64
    big = tmp_path / "big.csv"
    big.write_bytes(b"\x00" * 8)
    with pytest.raises(ExportStorageError):
        verify_export_file_size(big, maximum_bytes=4)
    assert not big.exists()
