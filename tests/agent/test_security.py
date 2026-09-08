
from fem.io import inp as abaqus
from fem_agent.artifacts import (
    ArtifactStore,
)
from fem_agent.engine import AgentSessionEngine
from fem_agent.providers.fake import FakeProvider
from fem_agent.tools.registry import AgentToolRegistry
from tests.helpers.abaqus_builders import write_perforated_plate_style_inp


def test_tool_catalog_has_no_path_or_code_execution_fields(tmp_path):
    registry = AgentToolRegistry(tmp_path / "workspace")

    serialized = repr(
        [definition.parameters for definition in registry.definitions]
    ).casefold()

    assert "path" not in serialized
    assert "python" not in serialized
    assert "shell" not in serialized
    assert "command" not in serialized


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

    monkeypatch.setattr(abaqus, "read_with_report", fail_in_parent)
    engine = AgentSessionEngine(workspace, FakeProvider())
    artifact = ArtifactStore(workspace).copy_input(engine.session_id, source)

    engine.attach_artifact(artifact.artifact_id)
    summary = engine.get_analysis_summary()

    assert summary.node_count == 6
    assert summary.element_count == 2
