from __future__ import annotations

from copy import deepcopy
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from fem.application import ModelSession, run_static_preflight
from fem.application.results import build_solve_result_bundle
from fem.io.project import load_project, save_project
from fem.mesh.settings import MeshSettings
from fem.solvers import static_linear
from fem_agent.authoring import ProposalState
from fem_agent.authoring_runtime import AuthoringWorkflowStage
from fem_agent.result_authoring import AgentResultQueryBridge
from fem_agent.tools.registry import ToolExecutionContext
from fem_gui.agent_authoring import (
    AgentAuthoringBridge,
    AgentPreflightState,
    SessionGeometryAuthoringPort,
    SessionResultQueryPort,
    create_session_authoring_workflow_controller,
)
from tests.test_agent_authoring_phase_a5 import _session as _a5_session


STEP_NAME = "分析步-静力"


def _production_controller(
    session: ModelSession,
) -> tuple[object, AgentAuthoringBridge]:
    state: dict[str, object] = {}

    def rebind() -> None:
        bridge = state["bridge"]
        controller = state["controller"]
        stale_ids = bridge.bind_snapshot(session.snapshot())
        controller.observe_binding(
            bridge.context,
            proposal_staled=bool(stale_ids),
        )

    def apply_definition_delta(_delta) -> None:
        rebind()

    def run_preflight(request) -> bool:
        task = session.prepare_validation(request.step_name)
        report = run_static_preflight(
            task.model,
            task.step_name,
            token=task.token,
        )
        assert report.passed
        assert session.accept_validation(task.token, report).accepted
        rebind()
        state["port"].complete_preflight(
            request.request_id,
            AgentPreflightState.PASSED,
            "passed",
        )
        return True

    def run_solve(request) -> bool:
        task = session.prepare_solve(request.step_name, request.job_name)
        assert session.begin_run(task.token).accepted
        result = static_linear.solve(
            task.model,
            task.step_name,
            name=request.job_name,
        )
        assert session.accept_run_succeeded(
            task.token,
            build_solve_result_bundle(task, result),
        ).accepted
        rebind()
        state["port"].complete_solve(
            request.proposal_id,
            ProposalState.SUCCEEDED,
            "succeeded",
        )
        return True

    port = SessionGeometryAuthoringPort(
        session,
        lambda: None,
        apply_definition_delta=apply_definition_delta,
        start_preflight_task=run_preflight,
        start_solve_task=run_solve,
    )
    bridge = AgentAuthoringBridge(port)
    bridge.bind_snapshot(session.snapshot())
    controller = create_session_authoring_workflow_controller(
        session,
        bridge,
        AgentResultQueryBridge(SessionResultQueryPort(session)),
    )
    state.update(port=port, bridge=bridge, controller=controller)
    return controller, bridge


def _dispatch(
    controller,
    session: ModelSession,
    name: str,
    arguments: dict[str, object],
    suffix: str,
):
    return controller.dispatch(
        name,
        arguments,
        ToolExecutionContext(
            session.snapshot().session_id,
            session.session_revision,
            suffix,
        ),
    )


def _apply_analysis_definitions(controller, session: ModelSession) -> None:
    actions = (
        ("create_static_step", {"name": STEP_NAME}),
        (
            "create_boundary_condition",
            {
                "name": "位移-固定端",
                "step_name": STEP_NAME,
                "target_scope": "边-固定端",
                "target_kind": "edge",
                "first_component": 1,
                "last_component": 2,
                "value": 0.0,
                "unit": "mm",
                "distribution": "uniform",
                "confirmed": True,
            },
        ),
        (
            "create_load",
            {
                "name": "载荷-拉伸",
                "step_name": STEP_NAME,
                "target_scope": "边-加载端",
                "entity_type": "edge",
                "load_type": "edge_traction",
                "component": None,
                "vector": [10.0, 0.0],
                "magnitude": None,
                "direction": "global_xy",
                "unit": "N/mm",
                "distribution": "uniform",
                "confirmed": True,
            },
        ),
        (
            "create_result_request",
            {
                "name": "结果请求-位移反力",
                "step_name": STEP_NAME,
                "target": "node",
                "variables": ["U", "RF"],
                "units": ["mm", "N"],
                "confirmed": True,
            },
        ),
    )
    for index, (action, parameters) in enumerate(actions):
        outcome = _dispatch(
            controller,
            session,
            "apply_model_definition",
            {"action": action, "parameters": parameters},
            f"definition-{index}",
        )
        assert outcome.ok, outcome.to_json()


def _solve_and_read_displacement(
    controller,
    bridge: AgentAuthoringBridge,
    session: ModelSession,
) -> dict[str, object]:
    preflight = _dispatch(
        controller,
        session,
        "run_native_preflight",
        {},
        "preflight",
    )
    assert preflight.ok, preflight.to_json()
    assert preflight.data["passed"] is True
    assert controller.stage is AuthoringWorkflowStage.SOLVE_READY

    proposal = _dispatch(
        controller,
        session,
        "prepare_solve_proposal",
        {},
        "solve-proposal",
    )
    assert proposal.ok, proposal.to_json()
    receipt = bridge.accept_from_gui_control(proposal.data["proposal_id"])
    controller.record_proposal_state("solve", receipt.state, receipt.message)
    assert receipt.state is ProposalState.SUCCEEDED
    assert controller.stage is AuthoringWorkflowStage.RESULTS_READY

    catalog_result = _dispatch(
        controller,
        session,
        "read_accepted_result_catalog",
        {},
        "result-catalog",
    )
    assert catalog_result.ok, catalog_result.to_json()
    catalog = catalog_result.data["catalog"]
    query = {
        "schema_version": "1.0",
        "variable": "U",
        "component": "Magnitude",
        "position": "node",
        "region": "all_nodes",
        "aggregation": "maximum",
        "expected_source": catalog["source"],
        "expected_materialization_generation": (
            catalog["materialization_generation"]
        ),
    }
    result = _dispatch(
        controller,
        session,
        "query_accepted_result",
        query,
        "result-query",
    )
    assert result.ok, result.to_json()
    scalar = result.data["scalar"]
    assert scalar["variable"] == "U"
    assert scalar["unit"] == "mm"
    assert scalar["source"] == catalog["source"]
    assert scalar["value"] >= 0.0
    return scalar


def test_a8_production_entry_solves_and_reads_one_accepted_result() -> None:
    session = _a5_session()
    controller, bridge = _production_controller(session)

    assert controller.stage is AuthoringWorkflowStage.DEFINITIONS_READY
    _apply_analysis_definitions(controller, session)
    scalar = _solve_and_read_displacement(controller, bridge, session)

    assert scalar["location"]["association"] == "node"


def test_a8_save_reopen_remesh_resumes_preflight_solve_and_result(
    tmp_path,
) -> None:
    session = _a5_session()
    controller, _bridge = _production_controller(session)
    _apply_analysis_definitions(controller, session)
    remeshed_candidate = deepcopy(session.snapshot().artifact.model)

    prepared = session.prepare_project_save()
    target = save_project(tmp_path / "accepted.femproj", prepared)
    assert session.accept_project_saved(prepared.token, target).accepted

    reopened = ModelSession()
    reopened.replace_from_snapshot(load_project(target).snapshot)
    task = reopened.prepare_agent_mesh_generation(
        "P1",
        MeshSettings(0.8),
        "b" * 64,
        expected_session_revision=reopened.session_revision,
    )
    assert reopened.accept_agent_generated_model(
        task.token,
        remeshed_candidate,
    ).accepted

    resumed, bridge = _production_controller(reopened)

    assert resumed.stage is AuthoringWorkflowStage.PREFLIGHT_READY
    assert reopened.snapshot().steps[0].name == STEP_NAME
    _solve_and_read_displacement(resumed, bridge, reopened)


def test_a8_direct_definitions_reject_nonconforming_visible_names() -> None:
    session = _a5_session()
    controller, _bridge = _production_controller(session)
    before = session.snapshot()

    rejected = _dispatch(
        controller,
        session,
        "apply_model_definition",
        {
            "action": "create_material",
            "parameters": {
                "name": "steel",
                "properties": {"E": 70000.0, "nu": 0.33},
            },
        },
        "bad-name",
    )

    assert not rejected.ok
    assert session.snapshot() == before
