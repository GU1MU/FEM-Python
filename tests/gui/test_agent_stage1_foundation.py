from __future__ import annotations

from fem.application import ModelSession
from fem.application.results import build_solve_result_bundle
from fem.core.model import AnalysisStep
from fem.solvers import static_linear
from fem_agent.result_authoring import AgentResultQueryBridge
from fem_agent.tools.registry import ToolExecutionContext
from fem_agent.workspace_catalog import WorkspaceCatalogBridge
from fem_gui.agent_authoring import (
    AgentAuthoringBridge,
    SessionGeometryAuthoringPort,
    SessionResultQueryPort,
    authoring_context_from_snapshot,
    create_session_authoring_workflow_controller,
)
from fem_gui.agent_workspace_catalog import FEMWorkspaceCatalogPort
from fem_gui.workspace import FEMWorkspace
from tests.gui.test_agent_result_query_phase_a7 import (
    STEP_NAME,
    _query,
    _solved_session,
)
from tests.helpers.preflight_builders import (
    failing_preflight_report,
    passing_preflight_report,
)
from fem_agent.result_authoring import (
    AgentResultAggregation,
    AgentResultVariable,
)


def _second_success(session: ModelSession):
    task = session.prepare_solve(STEP_NAME, "作业-stage1-2")
    assert session.begin_run(task.token).accepted
    result = static_linear.solve(task.model, task.step_name, name="作业-stage1-2")
    assert session.accept_run_succeeded(
        task.token,
        build_solve_result_bundle(task, result),
    ).accepted
    return task


def test_two_runs_have_independent_result_catalogs_and_queries() -> None:
    session = _solved_session()
    first_run_id = session.snapshot().displayed_result_run_id
    first_request = _query(
        session,
        variable=AgentResultVariable.DISPLACEMENT,
        component="Magnitude",
        position="node",
        region="all_nodes",
        aggregation=AgentResultAggregation.MAXIMUM,
    )
    second = _second_success(session)
    port = SessionResultQueryPort(session)

    first_catalog = port.catalog(first_run_id)
    second_catalog = port.catalog(second.run_id)
    first_scalar = port.query(first_request)

    assert first_catalog.ok and second_catalog.ok and first_scalar.ok
    assert first_catalog.catalog.source.run_id == first_run_id
    assert second_catalog.catalog.source.run_id == second.run_id
    assert first_catalog.catalog.nodal_regions == ("all_nodes",)
    assert first_catalog.catalog.element_regions == ("all_elements",)
    assert first_scalar.scalar.source.run_id == first_run_id


def test_run_catalog_paginates_with_closed_document_identity() -> None:
    session = _solved_session()
    _second_success(session)
    authoring_bridge = AgentAuthoringBridge(
        SessionGeometryAuthoringPort(session, lambda: None)
    )
    authoring_bridge.bind_snapshot(session.snapshot(), document_id=41)
    controller = create_session_authoring_workflow_controller(
        session,
        authoring_bridge,
        AgentResultQueryBridge(SessionResultQueryPort(session)),
    )
    outcome = controller.dispatch(
        "read_analysis_run_catalog",
        {"cursor": 0, "limit": 1},
        ToolExecutionContext(session.session_id, session.session_revision, "runs"),
    )

    assert outcome.ok
    assert outcome.data["document_id"] == "41"
    assert outcome.data["session_id"] == session.session_id
    assert outcome.data["session_revision"] == session.session_revision
    assert outcome.data["next_cursor"] == 1
    assert outcome.data["truncated"] is True
    assert set(outcome.data["runs"][0]) == {
        "run_id",
        "name",
        "step_name",
        "status",
        "artifact_id",
        "model_revision",
        "source_run_id",
        "result_id",
        "materialization_generation",
    }


def test_context_counts_history_without_claiming_a_displayed_result() -> None:
    session = _solved_session()
    request = _query(
        session,
        variable=AgentResultVariable.DISPLACEMENT,
        component="Magnitude",
        position="node",
        region="all_nodes",
        aggregation=AgentResultAggregation.MAXIMUM,
    )
    snapshot = session.snapshot()
    session.replace_model_definitions(
        snapshot.materials,
        snapshot.sections,
        snapshot.assignments,
        snapshot.steps,
    )
    context = authoring_context_from_snapshot(session.snapshot(), document_id=7)

    assert context.run_count == 1
    assert context.result_count == 1
    assert context.result_available is False
    assert context.displayed_result_run_id is None
    assert SessionResultQueryPort(session).query(request).ok
    assert "query_accepted_result" in {
        item.operation for item in context.capabilities if item.enabled
    }
    authoring_bridge = AgentAuthoringBridge(
        SessionGeometryAuthoringPort(session, lambda: None)
    )
    authoring_bridge.bind_context(context)
    controller = create_session_authoring_workflow_controller(
        session,
        authoring_bridge,
        AgentResultQueryBridge(SessionResultQueryPort(session)),
    )
    names = {item.name for item in controller.definitions}
    assert controller.stage.value != "results_ready"
    assert {
        "read_analysis_run_catalog",
        "read_accepted_result_catalog",
        "query_accepted_result",
    }.issubset(names)


def test_workspace_catalog_contains_model_and_result_without_paths(tmp_path) -> None:
    workspace = FEMWorkspace()
    model = workspace.add_model(
        ModelSession(),
        display_name="模型目录项",
        source_path=tmp_path / "secret-model.fem.json",
    )
    result = workspace.add_result(
        ModelSession(),
        display_name="结果目录项",
        source_path=tmp_path / "secret-result.femres",
    )
    workspace.activate(result)

    payload = WorkspaceCatalogBridge(
        FEMWorkspaceCatalogPort(workspace)
    ).catalog().to_dict()

    assert payload["active_target"]["document_id"] == str(result.document_id)
    assert [item["document_kind"] for item in payload["documents"]] == [
        model.kind,
        result.kind,
    ]
    assert "path" not in str(payload).casefold()
    assert "secret-model" not in str(payload)
    assert "secret-result" not in str(payload)

    session = model.session
    authoring_bridge = AgentAuthoringBridge(
        SessionGeometryAuthoringPort(session, lambda: None)
    )
    authoring_bridge.bind_snapshot(session.snapshot(), document_id=model.document_id)
    controller = create_session_authoring_workflow_controller(
        session,
        authoring_bridge,
        AgentResultQueryBridge(SessionResultQueryPort(session)),
        workspace_catalog_bridge=WorkspaceCatalogBridge(
            FEMWorkspaceCatalogPort(workspace)
        ),
    )
    controller.invalidate_binding("document is read-only")
    stale_names = {item.name for item in controller.definitions}
    stale_catalog = controller.dispatch(
        "read_workspace_documents",
        {},
        ToolExecutionContext(session.session_id, session.session_revision, "stale"),
    )
    assert "read_workspace_documents" in stale_names
    assert stale_catalog.ok


def test_multistep_solve_requires_explicit_step_and_uses_job_supplier() -> None:
    session = _solved_session()
    session.replace_model_definitions(
        (),
        (),
        (),
        (AnalysisStep("Step-A"), AnalysisStep("Step-B")),
    )
    for step_name, report_builder in (
        ("Step-A", failing_preflight_report),
        ("Step-B", passing_preflight_report),
    ):
        validation = session.prepare_validation(step_name)
        assert session.accept_validation(
            validation.token,
            report_builder(validation.token),
        ).accepted
    authoring_bridge = AgentAuthoringBridge(
        SessionGeometryAuthoringPort(session, lambda: None)
    )
    authoring_bridge.bind_snapshot(session.snapshot())
    supplied: list[str] = []

    def next_job_name() -> str:
        supplied.append("作业-9")
        return supplied[-1]

    controller = create_session_authoring_workflow_controller(
        session,
        authoring_bridge,
        AgentResultQueryBridge(SessionResultQueryPort(session)),
        next_job_name=next_job_name,
    )
    assert controller.stage.value == "solve_ready"
    context = ToolExecutionContext(
        session.session_id,
        session.session_revision,
        "multistep",
    )

    ambiguous = controller.dispatch("prepare_solve_proposal", {}, context)
    explicit = controller.dispatch(
        "prepare_solve_proposal",
        {"step_name": "Step-B"},
        ToolExecutionContext(
            session.session_id,
            session.session_revision,
            "explicit-step",
        ),
    )

    assert not ambiguous.ok
    assert explicit.ok
    assert supplied == ["作业-9"]
    record = next(iter(authoring_bridge._records.values()))
    assert record.proposal.operations[0].parameters["step_name"] == "Step-B"
    assert (
        record.proposal.operations[0].parameters["job_name"]
        == "作业-9"
    )
    assert session.find_run("作业-9") is None


def test_solve_proposal_dispatch_targets_bound_numeric_document() -> None:
    session = _solved_session()
    authoring_bridge = AgentAuthoringBridge(
        SessionGeometryAuthoringPort(session, lambda: None)
    )
    authoring_bridge.bind_snapshot(session.snapshot(), document_id=2)
    controller = create_session_authoring_workflow_controller(
        session,
        authoring_bridge,
        AgentResultQueryBridge(SessionResultQueryPort(session)),
    )
    assert controller.stage.value == "results_ready"

    outcome = controller.dispatch(
        "prepare_solve_proposal",
        {"step_name": STEP_NAME},
        ToolExecutionContext(session.session_id, session.session_revision, "numeric"),
    )

    assert outcome.ok, outcome.summary
    record = next(iter(authoring_bridge._records.values()))
    assert record.proposal.target_document_id == "2"
    assert record.state.name == "PENDING_CONFIRMATION"


def test_results_ready_keeps_preflight_and_repeated_solve_tools() -> None:
    session = _solved_session()
    authoring_bridge = AgentAuthoringBridge(
        SessionGeometryAuthoringPort(session, lambda: None)
    )
    authoring_bridge.bind_snapshot(session.snapshot())
    controller = create_session_authoring_workflow_controller(
        session,
        authoring_bridge,
        AgentResultQueryBridge(SessionResultQueryPort(session)),
    )

    assert controller.stage.value == "results_ready"
    assert {"run_native_preflight", "prepare_solve_proposal"}.issubset(
        {item.name for item in controller.definitions}
    )


def test_multistep_preflight_requires_explicit_current_step() -> None:
    session = _solved_session()
    snapshot = session.snapshot()
    session.replace_model_definitions(
        snapshot.materials,
        snapshot.sections,
        snapshot.assignments,
        (AnalysisStep("Step-A"), AnalysisStep("Step-B")),
    )
    started: list[str] = []

    def start_preflight(request) -> bool:
        started.append(request.step_name)
        return True

    authoring_bridge = AgentAuthoringBridge(
        SessionGeometryAuthoringPort(
            session,
            lambda: None,
            start_preflight_task=start_preflight,
        )
    )
    authoring_bridge.bind_snapshot(session.snapshot())
    controller = create_session_authoring_workflow_controller(
        session,
        authoring_bridge,
        AgentResultQueryBridge(SessionResultQueryPort(session)),
    )

    ambiguous = controller.dispatch(
        "run_native_preflight",
        {},
        ToolExecutionContext(
            session.session_id,
            session.session_revision,
            "ambiguous-preflight",
        ),
    )
    explicit = controller.dispatch(
        "run_native_preflight",
        {"step_name": "Step-B"},
        ToolExecutionContext(
            session.session_id,
            session.session_revision,
            "explicit-preflight",
        ),
    )

    assert not ambiguous.ok
    assert explicit.ok
    assert started == ["Step-B"]
