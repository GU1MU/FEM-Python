from __future__ import annotations

from fem.application import ModelSession, run_static_preflight
from fem.application.preprocessing import generate_fem_model
from fem.application.results import build_solve_result_bundle
from fem.geometry import (
    PathSweptGeometry,
    RectangleGeometry,
    WireGeometry,
    WireMember,
    WirePoint,
)
from fem.solvers import static_linear
from fem_agent.authoring import ProposalState
from fem_agent.result_authoring import AgentResultQueryBridge
from fem_agent.tools.registry import ToolExecutionContext
from fem_gui.agent_authoring import (
    AgentAuthoringBridge,
    AgentPreflightState,
    SessionGeometryAuthoringPort,
    SessionResultQueryPort,
    create_session_authoring_workflow_controller,
)


def _path_sweep() -> PathSweptGeometry:
    return PathSweptGeometry(
        RectangleGeometry("Sweep profile", 0.4, 0.3),
        WireGeometry(
            "Ordered path",
            (
                WirePoint("A", 0.0, 0.0, 0.0),
                WirePoint("B", 0.0, 0.0, 1.0),
                WirePoint("C", 0.5, 0.0, 1.5),
            ),
            (
                WireMember("AB", "A", "B"),
                WireMember("BC", "B", "C"),
            ),
        ),
        ("face:domain",),
        "transport",
    )


def _dispatch(
    controller: object,
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


def _record_mesh_requirements(
    controller: object,
    session: ModelSession,
    *,
    cell_shape: str,
    order: int,
    global_size: float,
) -> None:
    recorded = _dispatch(
        controller,
        session,
        "set_authoring_requirements",
        {
            "turn_id": f"turn-mesh-{cell_shape}",
            "requirements": {
                "mesh_cell_shape": cell_shape,
                "mesh_order": order,
                "mesh_global_size": global_size,
            },
        },
        f"mesh-requirements-{cell_shape}",
    )
    assert recorded.ok, recorded.to_json()
    assert recorded.data["missing_requirements"] == []


def _production_controller(
    session: ModelSession,
    *,
    start_mesh_task=None,
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

    def run_mesh(request) -> bool:
        if start_mesh_task is not None:
            return bool(start_mesh_task(request))
        candidate = generate_fem_model(request.task)
        assert state["port"].accept_mesh_result(
            request.proposal_id,
            candidate,
        ).accepted
        rebind()
        return True

    def run_preflight(request) -> bool:
        task = session.prepare_validation(request.step_name)
        report = run_static_preflight(
            task.model,
            task.step_name,
            token=task.token,
        )
        assert report.passed, report.diagnostics
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
        start_mesh_task=run_mesh,
        apply_definition_delta=apply_definition_delta,
        start_solve_task=run_solve,
        start_preflight_task=run_preflight,
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
