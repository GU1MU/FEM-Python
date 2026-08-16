from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from fem_agent.artifacts import (
    ArtifactIntegrityError,
    ArtifactStore,
    InputRejectedError,
    InvalidIdentifierError,
    UnsafePathError,
)


def test_input_copy_is_independent_and_reopenable(tmp_path: Path) -> None:
    workspace = tmp_path / "agent-workspace"
    source = tmp_path / "梁模型.inp"
    original = b"*Heading\n** local input\n*Node\n1, 0., 0., 0.\n"
    source.write_bytes(original)

    store = ArtifactStore(workspace)
    session_id = store.create_session("ses_copy")
    record = store.copy_input(session_id, source)
    copied = store.resolve_artifact(session_id, record.artifact_id)

    assert copied.read_bytes() == original
    assert record.sha256 == hashlib.sha256(original).hexdigest()
    assert record.size_bytes == len(original)
    assert record.display_path.startswith(f"inputs/{record.artifact_id}/")
    assert str(source.parent) not in record.display_path

    source.write_bytes(b"*Heading\nchanged original\n")
    source.unlink()

    reopened = ArtifactStore(workspace)
    assert reopened.resolve_artifact(
        session_id,
        record.artifact_id,
    ).read_bytes() == original


def test_repeated_input_copy_never_overwrites_existing_copy(tmp_path: Path) -> None:
    source = tmp_path / "model.inp"
    source.write_bytes(b"*Heading\nfirst\n")
    store = ArtifactStore(tmp_path / "workspace")
    session_id = store.create_session()

    first = store.copy_input(session_id, source)
    first_path = store.resolve_artifact(session_id, first.artifact_id)
    source.write_bytes(b"*Heading\nsecond\n")
    second = store.copy_input(session_id, source)
    second_path = store.resolve_artifact(session_id, second.artifact_id)

    assert first.artifact_id != second.artifact_id
    assert first_path != second_path
    assert first_path.read_bytes() == b"*Heading\nfirst\n"
    assert second_path.read_bytes() == b"*Heading\nsecond\n"


@pytest.mark.parametrize(
    ("name", "content", "max_bytes"),
    [
        ("model.txt", b"not an inp", 100),
        ("large.inp", b"123456", 5),
    ],
)
def test_input_copy_rejects_invalid_or_oversized_files(
    tmp_path: Path,
    name: str,
    content: bytes,
    max_bytes: int,
) -> None:
    source = tmp_path / name
    source.write_bytes(content)
    store = ArtifactStore(tmp_path / "workspace")
    session_id = store.create_session()

    with pytest.raises(InputRejectedError):
        store.copy_input(session_id, source, max_bytes=max_bytes)


def test_artifact_resolution_rejects_traversal_identifiers(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "workspace")
    session_id = store.create_session()

    with pytest.raises(InvalidIdentifierError):
        store.resolve_artifact(session_id, "../outside")
    with pytest.raises(InvalidIdentifierError):
        store.session_path("..")


def test_tampered_input_fails_hash_verification(tmp_path: Path) -> None:
    source = tmp_path / "model.inp"
    source.write_bytes(b"*Heading\n")
    store = ArtifactStore(tmp_path / "workspace")
    session_id = store.create_session()
    record = store.copy_input(session_id, source)
    copied = store.resolve_artifact(session_id, record.artifact_id)

    copied.write_bytes(b"tampered")

    with pytest.raises(ArtifactIntegrityError):
        store.resolve_artifact(session_id, record.artifact_id)


def test_artifact_metadata_cannot_redirect_outside_session(tmp_path: Path) -> None:
    source = tmp_path / "model.inp"
    source.write_bytes(b"*Heading\n")
    workspace = tmp_path / "workspace"
    store = ArtifactStore(workspace)
    session_id = store.create_session()
    record = store.copy_input(session_id, source)
    metadata_path = (
        workspace
        / "sessions"
        / session_id
        / "artifacts"
        / f"{record.artifact_id}.json"
    )
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    payload["record"]["display_path"] = "../outside.inp"
    metadata_path.write_text(
        json.dumps(payload, ensure_ascii=False),
        encoding="utf-8",
    )

    with pytest.raises(ArtifactIntegrityError):
        store.resolve_artifact(session_id, record.artifact_id)


@pytest.mark.platform
def test_stored_file_symlink_escape_is_rejected_when_supported(
    tmp_path: Path,
) -> None:
    source = tmp_path / "model.inp"
    source.write_bytes(b"*Heading\n")
    outside = tmp_path / "outside.inp"
    outside.write_bytes(b"outside")
    store = ArtifactStore(tmp_path / "workspace")
    session_id = store.create_session()
    record = store.copy_input(session_id, source)
    copied = store.resolve_artifact(session_id, record.artifact_id)
    copied.unlink()
    try:
        os.symlink(outside, copied)
    except OSError as exc:
        pytest.skip(
            f"[platform-capability] file symlinks are unavailable: {exc}"
        )

    with pytest.raises(UnsafePathError):
        store.resolve_artifact(session_id, record.artifact_id)


def test_run_creation_is_idempotent_and_outputs_are_confined(
    tmp_path: Path,
) -> None:
    store = ArtifactStore(tmp_path / "workspace")
    session_id = store.create_session()

    first = store.create_run(session_id, idempotency_key="solve_once")
    retry = store.create_run(session_id, idempotency_key="solve_once")
    assert retry.run_id == first.run_id
    assert retry.path == first.path

    result_path = first.path / "exports" / "summary.csv"
    result_path.write_text("node,value\n1,0.0\n", encoding="utf-8")
    record = store.register_run_artifact(
        session_id,
        first.run_id,
        result_path,
        kind="csv",
    )
    assert store.resolve_artifact(session_id, record.artifact_id) == result_path

    outside = tmp_path / "outside.csv"
    outside.write_text("sensitive", encoding="utf-8")
    with pytest.raises(UnsafePathError):
        store.register_run_artifact(
            session_id,
            first.run_id,
            outside,
            kind="csv",
        )


def test_artifact_metadata_is_atomic_and_has_no_temporary_files(
    tmp_path: Path,
) -> None:
    source = tmp_path / "model.inp"
    source.write_bytes(b"*Heading\n")
    workspace = tmp_path / "workspace"
    store = ArtifactStore(workspace)
    session_id = store.create_session()
    record = store.copy_input(session_id, source)
    metadata_directory = workspace / "sessions" / session_id / "artifacts"

    assert (metadata_directory / f"{record.artifact_id}.json").is_file()
    assert not list(metadata_directory.glob("*.tmp"))
