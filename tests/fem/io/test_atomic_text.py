from __future__ import annotations

import os
from pathlib import Path

import pytest

from fem.io import _atomic_text as atomic_text_module
from fem.io._atomic_text import (
    atomic_write_verified_text,
    atomic_write_verified_text_stream,
)


def _write(
    target: Path,
    serialized: str,
    *,
    checkpoint=None,
    before_replace=None,
) -> Path:
    return atomic_write_verified_text(
        target,
        serialized,
        verifier=lambda path: path.read_text(encoding="utf-8"),
        semantic_encoder=lambda value: value,
        expected_semantic=serialized,
        checkpoint=checkpoint,
        before_replace=before_replace,
    )


def test_before_replace_is_the_last_checkpoint_before_commit(
    tmp_path: Path,
) -> None:
    target = tmp_path / "result.txt"
    target.write_text("old", encoding="utf-8")
    observed: list[str] = []

    returned = _write(
        target,
        "new\n",
        before_replace=lambda: observed.append(
            target.read_text(encoding="utf-8")
        ),
    )

    assert returned == target
    assert observed == ["old"]
    assert target.read_text(encoding="utf-8") == "new\n"
    assert list(tmp_path.glob(f".{target.name}.*.tmp")) == []


def test_before_replace_failure_preserves_target_and_cleans_temp(
    tmp_path: Path,
) -> None:
    target = tmp_path / "result.txt"
    target.write_text("old", encoding="utf-8")

    def cancel() -> None:
        raise RuntimeError("cancelled before commit")

    with pytest.raises(RuntimeError, match="cancelled before commit"):
        _write(target, "new\n", before_replace=cancel)

    assert target.read_text(encoding="utf-8") == "old"
    assert list(tmp_path.glob(f".{target.name}.*.tmp")) == []


@pytest.mark.parametrize(
    ("cancel_on", "stage"),
    (
        (1, "write"),
        (2, "readback"),
        (3, "compare"),
        (4, "replace"),
    ),
)
def test_checkpoint_cancellation_at_each_transaction_stage(
    tmp_path: Path,
    cancel_on: int,
    stage: str,
) -> None:
    target = tmp_path / "result.txt"
    target.write_text("old", encoding="utf-8")
    calls = 0

    def cancel_at_stage() -> None:
        nonlocal calls
        calls += 1
        if calls == cancel_on:
            raise RuntimeError(f"cancelled before {stage}")

    with pytest.raises(RuntimeError, match=f"cancelled before {stage}"):
        _write(target, "new\n", checkpoint=cancel_at_stage)

    assert calls == cancel_on
    assert target.read_text(encoding="utf-8") == "old"
    assert list(tmp_path.glob(f".{target.name}.*.tmp")) == []


def test_success_calls_four_checkpoints_then_final_hook_once(
    tmp_path: Path,
) -> None:
    target = tmp_path / "result.txt"
    target.write_text("old", encoding="utf-8")
    events: list[str] = []
    checkpoint_calls = 0
    cancellation_pending = False

    def checkpoint() -> None:
        nonlocal checkpoint_calls
        checkpoint_calls += 1
        events.append(f"checkpoint-{checkpoint_calls}")
        if cancellation_pending:
            raise RuntimeError("late cancellation must not be observed")

    def verifier(path: Path) -> str:
        events.append("readback")
        return path.read_text(encoding="utf-8")

    def semantic_encoder(value: str) -> str:
        events.append("compare")
        return value

    def before_replace() -> None:
        events.append("before_replace")

    def replace_and_cancel(
        source: str | Path,
        destination: str | Path,
    ) -> None:
        nonlocal cancellation_pending
        events.append("replace")
        os.replace(source, destination)
        cancellation_pending = True

    returned = atomic_write_verified_text(
        target,
        "new\n",
        verifier=verifier,
        semantic_encoder=semantic_encoder,
        expected_semantic="new\n",
        checkpoint=checkpoint,
        before_replace=before_replace,
        replace_func=replace_and_cancel,
    )

    assert returned == target
    assert checkpoint_calls == 4
    assert events == [
        "checkpoint-1",
        "checkpoint-2",
        "readback",
        "checkpoint-3",
        "compare",
        "checkpoint-4",
        "before_replace",
        "replace",
    ]
    assert target.read_text(encoding="utf-8") == "new\n"
    assert list(tmp_path.glob(f".{target.name}.*.tmp")) == []


def test_invalid_utf8_is_typed_before_parent_creation(
    tmp_path: Path,
) -> None:
    class AtomicEncodingError(ValueError):
        pass

    parent = tmp_path / "not-created"
    target = parent / "result.txt"

    with pytest.raises(
        AtomicEncodingError,
        match="valid strict UTF-8",
    ):
        atomic_write_verified_text(
            target,
            "\ud800",
            verifier=lambda path: path,
            semantic_encoder=lambda value: value,
            expected_semantic=target,
            error_type=AtomicEncodingError,
        )

    assert not parent.exists()


def test_streaming_writer_verifies_and_atomically_installs_utf8(
    tmp_path: Path,
) -> None:
    target = tmp_path / "result.txt"
    target.write_text("old", encoding="utf-8")
    observed: list[str] = []

    returned = atomic_write_verified_text_stream(
        target,
        lambda stream: (
            stream.write("第一行\n"),
            stream.write("second,row\n"),
        ),
        before_replace=lambda: observed.append(
            target.read_text(encoding="utf-8")
        ),
    )

    assert returned == target
    assert observed == ["old"]
    assert target.read_text(encoding="utf-8") == "第一行\nsecond,row\n"
    assert list(tmp_path.glob(f".{target.name}.*.tmp")) == []


def test_streaming_verification_mismatch_preserves_old_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "result.txt"
    target.write_text("old", encoding="utf-8")
    monkeypatch.setattr(
        atomic_text_module,
        "_sha256_file",
        lambda _path: b"incorrect-digest",
    )

    with pytest.raises(ValueError, match="byte verification failed"):
        atomic_write_verified_text_stream(
            target,
            lambda stream: stream.write("new\n"),
        )

    assert target.read_text(encoding="utf-8") == "old"
    assert list(tmp_path.glob(f".{target.name}.*.tmp")) == []
