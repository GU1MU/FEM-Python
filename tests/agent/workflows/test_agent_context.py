from __future__ import annotations

import json

import pytest

from fem_agent.artifacts import ArtifactStore
from fem_gui.agent_context import (
    MAX_AGENT_INPUT_BYTES,
    MAX_CONTEXT_FILE_BYTES,
    WorkspaceContextError,
    prepare_workspace_context,
)
from fem_gui.agent_workspace import (
    build_workspace_file_reference,
    normalize_user_workspace,
)


def _reference(workspace_path, relative_path):
    workspace = normalize_user_workspace(workspace_path)
    return build_workspace_file_reference(
        workspace,
        relative_path,
    )


@pytest.mark.parametrize(
    ("encoding", "text"),
    (
        ("utf-8", "UTF-8 说明"),
        ("utf-8-sig", "带 BOM 的说明"),
        ("utf-16", "UTF-16 说明"),
        ("gb18030", "国标编码说明"),
    ),
)
def test_workspace_context_accepts_supported_text_encodings(
    tmp_path,
    encoding,
    text,
):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    source = workspace / f"notes-{encoding}.txt"
    source.write_text(text, encoding=encoding)

    prepared = prepare_workspace_context(
        (_reference(workspace, source.name),)
    )

    assert prepared.input_source is None
    assert prepared.request_context is not None
    document = json.loads(
        prepared.request_context.split("\n", 1)[1]
    )
    assert document["files"][0]["content"] == text
    assert str(workspace.resolve()) not in prepared.request_context


def test_six_megabyte_inp_is_kept_local_and_not_added_to_provider_context(
    tmp_path,
):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    source = workspace / "large-model.inp"
    line = b"** bounded local model comment\n"
    repetitions = (6 * 1024 * 1024) // len(line) + 1
    source.write_bytes((line * repetitions)[: 6 * 1024 * 1024])

    prepared = prepare_workspace_context(
        (_reference(workspace, source.name),)
    )

    assert prepared.input_source == source
    assert prepared.input_encoding == "utf-8"
    assert prepared.request_context is not None
    assert "local_fem_input" in prepared.request_context
    assert "bounded local model comment" not in prepared.request_context


def test_utf16_inp_can_be_transcoded_into_private_agent_storage(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    source = workspace / "model.inp"
    source.write_text("*Heading\n中文模型\n", encoding="utf-16")
    prepared = prepare_workspace_context(
        (_reference(workspace, source.name),)
    )
    store = ArtifactStore(tmp_path / "agent-private")
    session_id = store.create_session()

    record = store.copy_input(
        session_id,
        prepared.input_source,
        source_encoding=prepared.input_encoding,
    )
    private_copy = store.resolve_artifact(
        session_id,
        record.artifact_id,
    )

    assert private_copy.read_text(encoding="utf-8") == "*Heading\n中文模型\n"
    assert source.read_text(encoding="utf-16") == "*Heading\n中文模型\n"


def test_ordinary_context_rejects_oversize_and_binary_files(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    oversized = workspace / "oversized.md"
    oversized.write_bytes(b"a" * (MAX_CONTEXT_FILE_BYTES + 1))
    binary = workspace / "mesh.bin"
    binary.write_bytes(b"\x00\x01\x02\x03")

    with pytest.raises(WorkspaceContextError, match="1 MiB"):
        prepare_workspace_context(
            (_reference(workspace, oversized.name),)
        )
    with pytest.raises(WorkspaceContextError, match="二进制"):
        prepare_workspace_context(
            (_reference(workspace, binary.name),)
        )


def test_inp_over_fifty_megabytes_is_rejected_before_content_read(
    tmp_path,
):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    source = workspace / "too-large.inp"
    with source.open("wb") as stream:
        stream.truncate(MAX_AGENT_INPUT_BYTES + 1)

    with pytest.raises(WorkspaceContextError, match="50 MiB"):
        prepare_workspace_context(
            (_reference(workspace, source.name),)
        )


def test_combined_text_context_obeys_four_megabyte_turn_limit(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    references = []
    for index in range(4):
        source = workspace / f"context-{index}.txt"
        source.write_bytes(b"a" * MAX_CONTEXT_FILE_BYTES)
        references.append(_reference(workspace, source.name))

    with pytest.raises(WorkspaceContextError, match="4 MiB"):
        prepare_workspace_context(tuple(references))
