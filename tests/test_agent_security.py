import pytest

from fem import abaqus
from fem_agent.artifacts import (
    ArtifactStore,
    InputRejectedError,
    InvalidIdentifierError,
)
from fem_agent.engine import AgentSessionEngine
from fem_agent.providers.fake import FakeProvider
from fem_agent.tools.registry import AgentToolRegistry
from tests.helpers.abaqus_builders import write_perforated_plate_style_inp
from tests.helpers.file_builders import write_inp


def test_tool_catalog_has_no_path_or_code_execution_fields(tmp_path):
    registry = AgentToolRegistry(tmp_path / "workspace")

    serialized = repr(
        [definition.parameters for definition in registry.definitions]
    ).casefold()

    assert "path" not in serialized
    assert "python" not in serialized
    assert "shell" not in serialized
    assert "command" not in serialized


@pytest.mark.parametrize(
    "identifier",
    ["../escape", "C:\\escape", "/absolute", "with space"],
)
def test_artifact_identifiers_reject_traversal_and_absolute_paths(
    tmp_path,
    identifier,
):
    store = ArtifactStore(tmp_path / "workspace")
    session_id = store.create_session()

    with pytest.raises(InvalidIdentifierError):
        store.get_artifact(session_id, identifier)


def test_input_size_limit_is_checked_during_local_copy(tmp_path):
    source = write_inp(
        tmp_path,
        "too_large.inp",
        ["*Heading", "x" * 256],
    )
    store = ArtifactStore(tmp_path / "workspace")
    session_id = store.create_session()

    with pytest.raises(InputRejectedError, match="limit"):
        store.copy_input(session_id, source, max_bytes=32)


@pytest.mark.integration
def test_engine_import_is_process_isolated_and_ignores_workspace_modules(
    monkeypatch,
    tmp_path,
):
    source = write_perforated_plate_style_inp(
        tmp_path,
        "isolated_import.inp",
        ("*Boundary", "Set-right, 1, 1, 0.05"),
    )
    workspace = tmp_path / "workspace"
    hostile_package = workspace / "fem_agent"
    hostile_package.mkdir(parents=True)
    (hostile_package / "__init__.py").write_text("", encoding="utf-8")
    (hostile_package / "worker.py").write_text(
        "raise RuntimeError('workspace module hijack')\n",
        encoding="utf-8",
    )

    def fail_in_parent(*args, **kwargs):
        raise AssertionError("the parent process must not parse the Abaqus input")

    monkeypatch.setattr(abaqus, "parse_file", fail_in_parent)
    engine = AgentSessionEngine(workspace, FakeProvider())
    artifact = ArtifactStore(workspace).copy_input(engine.session_id, source)

    engine.attach_artifact(artifact.artifact_id)
    summary = engine.get_analysis_summary()

    assert summary.node_count == 6
    assert summary.element_count == 2
