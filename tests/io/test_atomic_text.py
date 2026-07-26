from __future__ import annotations

from pathlib import Path

import pytest

from fem.io._atomic_text import atomic_write_verified_text


def _write(
    target: Path,
    serialized: str,
    *,
    before_replace=None,
) -> Path:
    return atomic_write_verified_text(
        target,
        serialized,
        verifier=lambda path: path.read_text(encoding="utf-8"),
        semantic_encoder=lambda value: value,
        expected_semantic=serialized,
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
