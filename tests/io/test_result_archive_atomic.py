from __future__ import annotations

from pathlib import Path

import pytest

from fem.io._atomic_binary import atomic_write_verified_binary


def _write(path: Path, data: bytes, *, verifier, semantic, expected, **kwargs):
    return atomic_write_verified_binary(
        path,
        data,
        verifier=verifier,
        semantic_encoder=semantic,
        expected_semantic=expected,
        **kwargs,
    )


def test_binary_atomic_write_keeps_existing_target_on_semantic_mismatch(tmp_path: Path):
    target = tmp_path / "result.femres"
    target.write_bytes(b"old")

    with pytest.raises(ValueError, match="semantic"):
        _write(
            target,
            b"new",
            verifier=lambda path: path.read_bytes(),
            semantic=lambda value: value,
            expected=b"not-new",
        )

    assert target.read_bytes() == b"old"
    assert tuple(tmp_path.glob(".*.tmp")) == ()


def test_binary_atomic_write_keeps_existing_target_when_replace_fails(tmp_path: Path):
    target = tmp_path / "result.femres"
    target.write_bytes(b"old")

    def fail_replace(_temporary, _target):
        raise OSError("replace failure")

    with pytest.raises(OSError, match="replace failure"):
        _write(
            target,
            b"new",
            verifier=lambda path: path.read_bytes(),
            semantic=lambda value: value,
            expected=b"new",
            replace_func=fail_replace,
        )

    assert target.read_bytes() == b"old"
    assert tuple(tmp_path.glob(".*.tmp")) == ()


def test_binary_atomic_write_cleans_temp_after_checkpoint_cancellation(tmp_path: Path):
    target = tmp_path / "result.femres"
    target.write_bytes(b"old")
    calls = 0

    def checkpoint():
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("cancel")

    with pytest.raises(RuntimeError, match="cancel"):
        _write(
            target,
            b"new",
            verifier=lambda path: path.read_bytes(),
            semantic=lambda value: value,
            expected=b"new",
            checkpoint=checkpoint,
        )

    assert target.read_bytes() == b"old"
    assert tuple(tmp_path.glob(".*.tmp")) == ()
