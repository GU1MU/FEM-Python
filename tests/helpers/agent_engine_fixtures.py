from fem_agent.artifacts import ArtifactStore
from fem_agent.engine import AgentSessionEngine
from fem_agent.tools.registry import ToolExecutionContext

from tests.helpers.abaqus_builders import write_perforated_plate_style_inp


def _attached_engine(tmp_path, provider):
    source = write_perforated_plate_style_inp(
        tmp_path,
        "engine_model.inp",
        ("*Boundary", "Set-right, 1, 1, 0.05"),
    )
    workspace = tmp_path / "workspace"
    engine = AgentSessionEngine(
        workspace,
        provider,
        session_id="ses_engine",
    )
    artifact = ArtifactStore(workspace).copy_input(engine.session_id, source)
    engine.attach_artifact(artifact.artifact_id)
    return engine, source


def _ready_engine(tmp_path, provider):
    engine, _ = _attached_engine(tmp_path, provider)
    first = engine.revisions.require_current(engine.session_id)
    engine.registry.dispatch(
        "set_unit_context",
        {
            "length": "mm",
            "force": "N",
            "stress": "MPa",
            "density": "tonne/mm^3",
            "acceleration": "mm/s^2",
        },
        ToolExecutionContext(engine.session_id, first.revision, "ready_units"),
    )
    second = engine.revisions.require_current(engine.session_id)
    engine.registry.dispatch(
        "set_result_requests",
        {
            "queries": [{"kind": "max_displacement_magnitude"}],
            "export_formats": [],
        },
        ToolExecutionContext(
            engine.session_id,
            second.revision,
            "ready_results",
        ),
    )
    engine.get_analysis_summary()
    return engine
