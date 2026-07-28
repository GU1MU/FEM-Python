from fem_agent.artifacts import ArtifactStore
from fem_agent.engine import AgentSessionEngine
from fem_agent.providers.fake import FakeProvider
from fem_agent.tools.registry import AgentToolRegistry, ToolExecutionContext
from fem_agent.worker import InspectionWorkerError
from tests.helpers.abaqus_builders import write_perforated_plate_style_inp


def _attached_engine(tmp_path):
    source = write_perforated_plate_style_inp(
        tmp_path,
        "registry_model.inp",
        ("*Boundary", "Set-right, 1, 1, 0.05"),
    )
    workspace = tmp_path / "workspace"
    engine = AgentSessionEngine(
        workspace,
        FakeProvider(),
        session_id="ses_registry",
    )
    artifact = ArtifactStore(workspace).copy_input(engine.session_id, source)
    engine.attach_artifact(artifact.artifact_id)
    return engine, source


def test_registry_publishes_only_the_v0_whitelist(tmp_path):
    registry = AgentToolRegistry(tmp_path / "workspace")

    names = {definition.name for definition in registry.definitions}

    assert names == {
        "show_capabilities",
        "inspect_abaqus",
        "set_unit_context",
        "set_result_requests",
        "get_analysis_summary",
        "validate_analysis",
        "solve_confirmed_analysis",
        "query_results",
        "export_results",
        "list_artifacts",
    }
    assert all(
        definition.parameters["additionalProperties"] is False
        for definition in registry.definitions
    )


def test_result_request_tool_exposes_edge_and_surface_regions(tmp_path):
    registry = AgentToolRegistry(tmp_path / "workspace")
    definition = next(
        item
        for item in registry.definitions
        if item.name == "set_result_requests"
    )
    query_properties = (
        definition.parameters["properties"]["queries"]["items"]["properties"]
    )

    assert {"node_set", "edge", "surface"} <= set(query_properties)
    assert "2D edge" in query_properties["edge"]["description"]
    assert "3D face" in query_properties["surface"]["description"]


def test_postsolve_query_tool_requires_bounded_queries(tmp_path):
    registry = AgentToolRegistry(tmp_path / "workspace")
    definition = next(
        item
        for item in registry.definitions
        if item.name == "query_results"
    )

    assert definition.parameters["required"] == ["queries"]
    queries = definition.parameters["properties"]["queries"]
    assert queries["minItems"] == 1
    assert queries["maxItems"] == 64
    assert {"node_set", "edge", "surface"} <= set(
        queries["items"]["properties"]
    )


def test_result_configuration_allows_exports_without_precomputed_queries(
    tmp_path,
):
    engine, _ = _attached_engine(tmp_path)
    current = engine.revisions.require_current(engine.session_id)

    result = engine.registry.dispatch(
        "set_result_requests",
        {"queries": [], "export_formats": ["csv"]},
        ToolExecutionContext(
            engine.session_id,
            current.revision,
            "exports_without_queries",
        ),
    )

    updated = engine.revisions.require_current(engine.session_id)
    assert result.ok is True
    assert updated.spec.requested_queries == ()
    assert [item.value for item in updated.spec.export_formats] == ["csv"]


def test_malformed_tool_arguments_fail_before_revision_mutation(tmp_path):
    engine, _ = _attached_engine(tmp_path)
    registry = engine.registry
    before = engine.revisions.require_current(engine.session_id)
    context = ToolExecutionContext(
        engine.session_id,
        before.revision,
        "bad_units",
    )

    result = registry.dispatch(
        "set_unit_context",
        {
            "length": "mm",
            "force": "N",
            "stress": "MPa",
            "density": "tonne/mm^3",
            "acceleration": "mm/s^2",
            "source_path": "forbidden.inp",
        },
        context,
    )

    assert result.ok is False
    assert result.diagnostics[0].code == "INVALID_TOOL_ARGUMENTS"
    assert engine.revisions.require_current(engine.session_id) == before


def test_cloud_tool_cannot_authorize_a_solve(tmp_path):
    engine, _ = _attached_engine(tmp_path)
    current = engine.revisions.require_current(engine.session_id)

    result = engine.registry.dispatch(
        "solve_confirmed_analysis",
        {},
        ToolExecutionContext(
            engine.session_id,
            current.revision,
            "model_solve_attempt",
        ),
    )

    assert result.ok is False
    assert result.diagnostics[0].code == "CONFIRMATION_REQUIRED"


def test_unknown_tool_is_rejected_without_calling_fem(tmp_path):
    registry = AgentToolRegistry(tmp_path / "workspace")

    result = registry.dispatch(
        "run_python",
        {"code": "print('unsafe')"},
        ToolExecutionContext("ses_unknown", 0, "unknown_call"),
    )

    assert result.ok is False
    assert result.diagnostics[0].code == "UNKNOWN_TOOL"


def test_inspection_worker_failure_is_classified_as_infrastructure(
    monkeypatch,
    tmp_path,
):
    engine, _ = _attached_engine(tmp_path)
    current = engine.revisions.require_current(engine.session_id)

    def fail_inspection(*args, **kwargs):
        raise InspectionWorkerError("The inspection protocol failed.")

    monkeypatch.setattr(
        engine.registry.inspector,
        "inspect",
        fail_inspection,
    )

    result = engine.registry.dispatch(
        "get_analysis_summary",
        {},
        ToolExecutionContext(
            engine.session_id,
            current.revision,
            "failed_summary",
        ),
    )

    assert result.ok is False
    assert result.diagnostics[0].code == "WORKER_CRASH"
    assert result.diagnostics[0].code != "INVALID_MODEL"
