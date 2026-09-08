from __future__ import annotations

from collections.abc import Callable
import os
from pathlib import Path
from typing import Any, TextIO

import pytest

from fem.application.definitions import NativePart
from fem.application.feature_history import derive_feature_history
from fem.application.session import ProjectSnapshot
from fem.geometry.recipes import RectangleGeometry
from fem.io import _project_codec as project_codec
from fem.io import project_v2
from fem.io._project_codec import atomic_write_project, loads_json_strict
from fem.io._project_errors import ProjectEncodeError


_SERIALIZED = '{"format":"fem.project","schema":2}\n'
_SEMANTIC = {"format": "fem.project", "schema": 2}


def _target_with_old_content(tmp_path: Path) -> Path:
    target = tmp_path / "project.femproj"
    target.write_text("old-project", encoding="utf-8")
    return target


def _temporary_files(target: Path) -> list[Path]:
    return list(target.parent.glob(f".{target.name}.*.tmp"))


def _write_default(
    target: Path,
    *,
    verifier: Callable[[Path], Any] | None = None,
    semantic_encoder: Callable[[Any], Any] | None = None,
    expected_semantic: Any = _SEMANTIC,
    replace_func: Callable[[str | Path, str | Path], Any] | None = None,
    unlink_func: Callable[[Path], Any] | None = None,
) -> Path:
    return atomic_write_project(
        target,
        _SERIALIZED,
        verifier=(
            (lambda path: loads_json_strict(path.read_bytes()))
            if verifier is None
            else verifier
        ),
        semantic_encoder=(
            (lambda value: value)
            if semantic_encoder is None
            else semantic_encoder
        ),
        expected_semantic=expected_semantic,
        replace_func=replace_func,
        unlink_func=unlink_func,
    )


class _FaultingTextStream:
    def __init__(
        self,
        stream: TextIO,
        stage: str,
        error: BaseException | None,
    ) -> None:
        self._stream = stream
        self._stage = stage
        self._error = error

    def _raise_failure(self, stage: str) -> None:
        if self._stage != stage:
            return
        if self._error is not None:
            raise self._error
        raise OSError(f"injected {stage} failure")

    def write(self, value: str) -> int:
        self._raise_failure("write")
        return self._stream.write(value)

    def flush(self) -> None:
        self._raise_failure("flush")
        self._stream.flush()

    def fileno(self) -> int:
        return self._stream.fileno()

    def close(self) -> None:
        self._stream.close()
        self._raise_failure("close")


def _inject_stream_failure(
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
    *,
    error: BaseException | None = None,
) -> None:
    original_fdopen = project_codec.os.fdopen

    def faulting_fdopen(
        descriptor: int,
        *args: Any,
        **kwargs: Any,
    ) -> _FaultingTextStream:
        return _FaultingTextStream(
            original_fdopen(descriptor, *args, **kwargs),
            stage,
            error,
        )

    monkeypatch.setattr(project_codec.os, "fdopen", faulting_fdopen)


@pytest.mark.parametrize("stage", ["write", "flush", "close"])
def test_stream_stage_failure_preserves_old_target_and_cleans_temp(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
) -> None:
    target = _target_with_old_content(tmp_path)
    _inject_stream_failure(monkeypatch, stage)

    with pytest.raises(OSError, match=rf"injected {stage} failure"):
        _write_default(target)

    assert target.read_text(encoding="utf-8") == "old-project"
    assert _temporary_files(target) == []


def test_parent_creation_failure_happens_before_temp_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = _target_with_old_content(tmp_path)
    original_mkdir = Path.mkdir

    def fail_target_parent(
        path: Path,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        if path == target.parent:
            raise OSError("injected parent creation failure")
        original_mkdir(path, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", fail_target_parent)

    with pytest.raises(OSError, match="injected parent creation failure"):
        _write_default(target)

    assert target.read_text(encoding="utf-8") == "old-project"
    assert _temporary_files(target) == []


def test_temp_creation_failure_preserves_old_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = _target_with_old_content(tmp_path)

    def fail_mkstemp(*args: Any, **kwargs: Any) -> tuple[int, str]:
        del args, kwargs
        raise OSError("injected temp creation failure")

    monkeypatch.setattr(project_codec.tempfile, "mkstemp", fail_mkstemp)

    with pytest.raises(OSError, match="injected temp creation failure"):
        _write_default(target)

    assert target.read_text(encoding="utf-8") == "old-project"
    assert _temporary_files(target) == []


def test_fsync_failure_preserves_old_target_and_cleans_temp(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = _target_with_old_content(tmp_path)

    def fail_fsync(descriptor: int) -> None:
        del descriptor
        raise OSError("injected fsync failure")

    monkeypatch.setattr(project_codec.os, "fsync", fail_fsync)

    with pytest.raises(OSError, match="injected fsync failure"):
        _write_default(target)

    assert target.read_text(encoding="utf-8") == "old-project"
    assert _temporary_files(target) == []


def test_verifier_failure_preserves_old_target_and_cleans_temp(
    tmp_path: Path,
) -> None:
    target = _target_with_old_content(tmp_path)
    primary = RuntimeError("injected readback failure")

    def fail_verifier(path: Path) -> Any:
        assert path.read_text(encoding="utf-8") == _SERIALIZED
        raise primary

    with pytest.raises(RuntimeError, match="injected readback failure") as caught:
        _write_default(target, verifier=fail_verifier)

    assert caught.value is primary
    assert target.read_text(encoding="utf-8") == "old-project"
    assert _temporary_files(target) == []


def test_semantic_encoder_failure_preserves_old_target_and_cleans_temp(
    tmp_path: Path,
) -> None:
    target = _target_with_old_content(tmp_path)
    primary = RuntimeError("injected semantic encoder failure")

    def fail_semantic_encoder(value: Any) -> Any:
        assert value == _SEMANTIC
        raise primary

    with pytest.raises(
        RuntimeError,
        match="injected semantic encoder failure",
    ) as caught:
        _write_default(target, semantic_encoder=fail_semantic_encoder)

    assert caught.value is primary
    assert target.read_text(encoding="utf-8") == "old-project"
    assert _temporary_files(target) == []


def test_semantic_compare_mismatch_preserves_old_target_and_cleans_temp(
    tmp_path: Path,
) -> None:
    target = _target_with_old_content(tmp_path)

    with pytest.raises(ProjectEncodeError, match="snapshot 不一致"):
        _write_default(
            target,
            expected_semantic={"format": "fem.project", "schema": 1},
        )

    assert target.read_text(encoding="utf-8") == "old-project"
    assert _temporary_files(target) == []


def test_replace_failure_preserves_old_target_and_cleans_temp(
    tmp_path: Path,
) -> None:
    target = _target_with_old_content(tmp_path)
    primary = PermissionError("injected replace failure")

    def fail_replace(source: str | Path, destination: str | Path) -> None:
        assert Path(source) in _temporary_files(target)
        assert Path(destination) == target
        raise primary

    with pytest.raises(PermissionError, match="injected replace failure") as caught:
        _write_default(target, replace_func=fail_replace)

    assert caught.value is primary
    assert target.read_text(encoding="utf-8") == "old-project"
    assert _temporary_files(target) == []


@pytest.mark.parametrize(
    "stage",
    ["write", "flush", "fsync", "verify", "replace"],
)
def test_cleanup_failure_is_a_note_on_each_primary_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
) -> None:
    target = _target_with_old_content(tmp_path)
    primary = OSError(f"injected {stage} failure")

    if stage in {"write", "flush"}:
        _inject_stream_failure(monkeypatch, stage, error=primary)
    elif stage == "fsync":
        def fail_fsync(descriptor: int) -> None:
            del descriptor
            raise primary

        monkeypatch.setattr(project_codec.os, "fsync", fail_fsync)

    def verifier(path: Path) -> Any:
        if stage == "verify":
            raise primary
        return loads_json_strict(path.read_bytes())

    def replace(source: str | Path, destination: str | Path) -> None:
        if stage == "replace":
            raise primary
        os.replace(source, destination)

    def fail_cleanup(path: Path) -> None:
        raise PermissionError(f"injected cleanup failure for {path.name}")

    match = rf"injected {stage} failure"
    with pytest.raises(OSError, match=match) as caught:
        _write_default(
            target,
            verifier=verifier,
            replace_func=replace,
            unlink_func=fail_cleanup,
        )

    assert caught.value is primary
    notes = getattr(caught.value, "__notes__", ())
    remaining = _temporary_files(target)
    assert any("injected cleanup failure" in note for note in notes)
    assert len(remaining) == 1
    assert any(str(remaining[0]) in note for note in notes)
    assert target.read_text(encoding="utf-8") == "old-project"


def _minimal_v2_snapshot() -> ProjectSnapshot:
    geometry = RectangleGeometry("板", 2.0, 1.0)
    return ProjectSnapshot(
        parts=(NativePart("零件", "主体"),),
        geometry_recipe=geometry,
        feature_history=derive_feature_history(geometry),
    )


def test_save_project_v2_readback_failure_is_atomic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = _target_with_old_content(tmp_path)
    primary = project_v2.ProjectV2DecodeError(
        "injected v2 readback failure"
    )

    def fail_readback(path: str | Path) -> ProjectSnapshot:
        assert Path(path) in _temporary_files(target)
        raise primary

    monkeypatch.setattr(project_v2, "load_project_v2", fail_readback)

    with pytest.raises(
        project_v2.ProjectV2DecodeError,
        match="injected v2 readback failure",
    ) as caught:
        project_v2.save_project_v2(target, _minimal_v2_snapshot())

    assert caught.value is primary
    assert target.read_text(encoding="utf-8") == "old-project"
    assert _temporary_files(target) == []


def test_save_project_v2_replace_failure_is_atomic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = _target_with_old_content(tmp_path)
    primary = PermissionError("injected v2 replace failure")

    def fail_replace(source: str | Path, destination: str | Path) -> None:
        assert Path(source) in _temporary_files(target)
        assert Path(destination) == target
        raise primary

    monkeypatch.setattr(project_codec.os, "replace", fail_replace)

    with pytest.raises(
        PermissionError,
        match="injected v2 replace failure",
    ) as caught:
        project_v2.save_project_v2(target, _minimal_v2_snapshot())

    assert caught.value is primary
    assert target.read_text(encoding="utf-8") == "old-project"
    assert _temporary_files(target) == []
