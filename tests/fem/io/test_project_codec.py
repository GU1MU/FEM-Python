from __future__ import annotations

from pathlib import Path

import pytest

from fem.application.session import ProjectSnapshot
from fem.io import (
    ProjectDecodeError,
    ProjectEncodeError,
)
from fem.io import _project_codec as project_codec
from fem.io._project_codec import (
    atomic_write_project,
    dumps_canonical_json,
    loads_json_strict,
    unwrap_project_snapshot,
)


@pytest.mark.parametrize(
    ("document", "message"),
    [
        ('{"schema": 1, "schema": 2}', "重复键"),
        ('{"value": NaN}', "非有限数值"),
        ('{"value": Infinity}', "非有限数值"),
        ('{"value": 1e9999}', "非有限数值"),
    ],
)
def test_strict_json_rejects_ambiguous_or_non_finite_documents(
    document,
    message,
) -> None:
    with pytest.raises(ProjectDecodeError, match=message):
        loads_json_strict(document)


def test_strict_json_decodes_utf8_and_rejects_invalid_bytes() -> None:
    assert loads_json_strict('{"名称": "矩形板"}'.encode("utf-8")) == {
        "名称": "矩形板"
    }

    with pytest.raises(ProjectDecodeError, match="UTF-8"):
        loads_json_strict(b'{"name":"\xff"}')


def test_canonical_json_is_recursive_stable_utf8_with_one_final_lf() -> None:
    first = {
        "z": {"second": 2, "first": 1},
        "名称": "矩形板",
        "a": [{"z": 2, "a": 1}],
    }
    second = {
        "a": [{"a": 1, "z": 2}],
        "名称": "矩形板",
        "z": {"first": 1, "second": 2},
    }

    serialized = dumps_canonical_json(first)

    assert serialized == dumps_canonical_json(second)
    assert serialized.startswith('{\n  "a"')
    assert '"first"' in serialized
    assert "矩形板" in serialized
    assert serialized.endswith("\n")
    assert not serialized.endswith("\n\n")


def test_snapshot_unwrap_returns_a_deepcopy() -> None:
    original = ProjectSnapshot(
        geometry_recipe={"kind": "rectangle", "values": [1, 2]},
    )

    detached = unwrap_project_snapshot(original)
    detached.geometry_recipe["values"].append(3)

    assert original.geometry_recipe["values"] == [1, 2]
    assert detached is not original


def test_atomic_writer_verifies_then_installs_exact_utf8_bytes(tmp_path) -> None:
    target = tmp_path / "project.femproj"
    target.write_text("previous", encoding="utf-8")
    expected = {"format": "fem.project", "schema": 2, "名称": "矩形板"}
    serialized = dumps_canonical_json(expected)
    verified_paths: list[Path] = []

    def verify(path: Path):
        verified_paths.append(path)
        return loads_json_strict(path.read_bytes())

    returned = atomic_write_project(
        target,
        serialized,
        verifier=verify,
        semantic_encoder=lambda value: value,
        expected_semantic=expected,
    )

    assert returned == target
    assert target.read_bytes() == serialized.encode("utf-8")
    assert len(verified_paths) == 1
    assert verified_paths[0].parent == target.parent
    assert list(tmp_path.glob(f".{target.name}.*.tmp")) == []


def test_atomic_writer_fsync_failure_preserves_target_and_cleans_temp(
    tmp_path,
    monkeypatch,
) -> None:
    target = tmp_path / "project.femproj"
    target.write_text("previous", encoding="utf-8")

    def fail_fsync(_descriptor):
        raise OSError("fsync failed")

    monkeypatch.setattr(project_codec.os, "fsync", fail_fsync)

    with pytest.raises(OSError, match="fsync failed"):
        atomic_write_project(
            target,
            "{}\n",
            verifier=lambda path: loads_json_strict(path.read_bytes()),
            semantic_encoder=lambda value: value,
            expected_semantic={},
        )

    assert target.read_text(encoding="utf-8") == "previous"
    assert list(tmp_path.glob(f".{target.name}.*.tmp")) == []


def test_atomic_writer_compare_failure_preserves_target_and_cleans_temp(
    tmp_path,
) -> None:
    target = tmp_path / "project.femproj"
    target.write_text("previous", encoding="utf-8")

    with pytest.raises(ProjectEncodeError, match="snapshot 不一致"):
        atomic_write_project(
            target,
            '{"schema": 2}\n',
            verifier=lambda path: loads_json_strict(path.read_bytes()),
            semantic_encoder=lambda value: value,
            expected_semantic={"schema": 1},
        )

    assert target.read_text(encoding="utf-8") == "previous"
    assert list(tmp_path.glob(f".{target.name}.*.tmp")) == []


def test_cleanup_failure_is_noted_without_masking_primary_error(
    tmp_path,
) -> None:
    target = tmp_path / "project.femproj"
    target.write_text("previous", encoding="utf-8")
    primary = RuntimeError("verification failed")

    def fail_verify(_path: Path):
        raise primary

    def fail_cleanup(_path: Path):
        raise OSError("cleanup failed")

    with pytest.raises(RuntimeError, match="verification failed") as caught:
        atomic_write_project(
            target,
            "{}\n",
            verifier=fail_verify,
            semantic_encoder=lambda value: value,
            expected_semantic={},
            unlink_func=fail_cleanup,
        )

    assert caught.value is primary
    notes = getattr(caught.value, "__notes__", ())
    assert any("cleanup failed" in note for note in notes)
    assert any(f".{target.name}." in note and ".tmp" in note for note in notes)
    assert target.read_text(encoding="utf-8") == "previous"
    assert len(list(tmp_path.glob(f".{target.name}.*.tmp"))) == 1
