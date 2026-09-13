from __future__ import annotations

from pathlib import Path
from fem.application import ModelSession, UnitContext
from fem.application.preprocessing import generate_fem_model
from fem.geometry.recipes import WireGeometry, WireMember, WirePoint
from fem_agent.authoring import ProposalState
from fem_agent.mesh_authoring import MeshIntent, create_mesh_proposal
from fem_agent.schemas import ExportFormat
from fem_agent.tools.exports import export_results
from fem_gui.agent_authoring import (
    AgentAuthoringBridge,
    AgentMeshTaskRequest,
    SessionGeometryAuthoringPort,
    authoring_context_from_snapshot,
)
from tests.helpers.agent_authoring_workflows import dispatch_authoring_tool


TRUSS_STEP_NAME = "分析步-拉伸"


def make_straight_wire() -> WireGeometry:
    return WireGeometry(
        "轴向杆",
        (
            WirePoint("Root", 0.0, 0.0, 0.0),
            WirePoint("Tip", 2.0, 0.0, 0.0),
        ),
        (WireMember("Bar", "Root", "Tip"),),
    )


def make_meshed_line_session(line_element_type: str = "Truss2") -> ModelSession:
    session = ModelSession()
    session.create_native_project_with_first_part(
        "Agent 轴向杆",
        UnitContext("mm", "N", "MPa"),
        make_straight_wire(),
        part_name="杆件",
    )
    requests: list[AgentMeshTaskRequest] = []
    port = SessionGeometryAuthoringPort(
        session,
        lambda: None,
        lambda request: requests.append(request) is None,
    )
    bridge = AgentAuthoringBridge(port)
    bridge.bind_snapshot(session.snapshot())
    proposal = create_mesh_proposal(
        proposal_id=f"proposal-line-mesh-{line_element_type}",
        agent_session_id="agent-line-mesh",
        turn_id="turn-mesh",
        source_tool_call_ids=("call-mesh",),
        context=authoring_context_from_snapshot(session.snapshot()),
        draft_revision=1,
        part_id="P1",
        mesh_intent=MeshIntent(
            "line",
            1,
            global_size=3.0,
            line_element_type=line_element_type,
        ),
    )
    bridge.register_proposal(proposal)
    receipt = bridge.accept_from_gui_control(proposal.proposal_id)
    assert receipt.state is ProposalState.RUNNING
    candidate = generate_fem_model(requests[0].task)
    assert port.accept_mesh_result(proposal.proposal_id, candidate).accepted
    return session


def apply_definition_action(
    controller: object,
    session: ModelSession,
    action: str,
    parameters: dict[str, object],
    suffix: str,
) -> None:
    outcome = dispatch_authoring_tool(
        controller,
        session,
        "apply_model_definition",
        {"action": action, "parameters": parameters},
        suffix,
    )
    assert outcome.ok, outcome.to_json()


def apply_truss_definitions(
    controller: object,
    session: ModelSession,
    *,
    include_tip_transverse_constraint: bool = True,
    include_section_assignment: bool = True,
) -> None:
    actions: list[tuple[str, dict[str, object], str]] = [
        (
            "create_named_region",
            {
                "name": "点-固定端",
                "part_id": "P1",
                "logical_ids": ["point:Root"],
                "mesh_kind": "node",
            },
            "root-region",
        ),
        (
            "create_named_region",
            {
                "name": "点-自由端",
                "part_id": "P1",
                "logical_ids": ["point:Tip"],
                "mesh_kind": "node",
            },
            "tip-region",
        ),
        (
            "create_named_region",
            {
                "name": "域-杆件",
                "part_id": "P1",
                "logical_ids": ["edge:Bar"],
                "mesh_kind": "element",
            },
            "bar-region",
        ),
        (
            "create_material",
            {
                "name": "材料-钢",
                "properties": {"E": 210000.0, "nu": 0.3},
            },
            "material",
        ),
    ]
    if include_section_assignment:
        actions.extend(
            [
                (
                    "create_section",
                    {
                        "name": "截面-拉杆",
                        "material": "材料-钢",
                        "section_type": "truss",
                        "properties": {"area": 10.0},
                    },
                    "section",
                ),
                (
                    "assign_section",
                    {
                        "section_name": "截面-拉杆",
                        "region_name": "域-杆件",
                    },
                    "assignment",
                ),
            ]
        )
    actions.extend(
        [
            ("create_static_step", {"name": TRUSS_STEP_NAME}, "step"),
            (
                "create_boundary_condition",
                {
                    "name": "位移-固定端",
                    "step_name": TRUSS_STEP_NAME,
                    "target_scope": "点-固定端",
                    "target_kind": "node_set",
                    "first_component": 1,
                    "last_component": 3,
                    "value": 0.0,
                    "unit": "mm",
                    "distribution": "uniform",
                    "confirmed": True,
                },
                "root-boundary",
            ),
        ]
    )
    if include_tip_transverse_constraint:
        actions.append(
            (
                "create_boundary_condition",
                {
                    "name": "位移-自由端横向",
                    "step_name": TRUSS_STEP_NAME,
                    "target_scope": "点-自由端",
                    "target_kind": "node_set",
                    "first_component": 2,
                    "last_component": 3,
                    "value": 0.0,
                    "unit": "mm",
                    "distribution": "uniform",
                    "confirmed": True,
                },
                "tip-boundary",
            )
        )
    actions.extend(
        [
            (
                "create_load",
                {
                    "name": "载荷-轴向力",
                    "step_name": TRUSS_STEP_NAME,
                    "target_scope": "点-自由端",
                    "entity_type": "node",
                    "load_type": "nodal",
                    "component": 1,
                    "vector": None,
                    "magnitude": 1000.0,
                    "direction": "global_x",
                    "unit": "N",
                    "distribution": "concentrated",
                    "confirmed": True,
                },
                "load",
            ),
            (
                "create_result_request",
                {
                    "name": "结果请求-节点",
                    "step_name": TRUSS_STEP_NAME,
                    "target": "node",
                    "variables": ["U", "RF"],
                    "units": ["mm", "N"],
                    "confirmed": True,
                },
                "node-output",
            ),
            (
                "create_result_request",
                {
                    "name": "结果请求-应力",
                    "step_name": TRUSS_STEP_NAME,
                    "target": "element",
                    "variables": ["S"],
                    "units": ["MPa"],
                    "confirmed": True,
                },
                "element-output",
            ),
        ]
    )
    for action, parameters, suffix in actions:
        apply_definition_action(controller, session, action, parameters, suffix)


def solve_authoring_session(controller: object, bridge: AgentAuthoringBridge, session: ModelSession):
    preflight = dispatch_authoring_tool(controller, session, "run_native_preflight", {}, "preflight")
    assert preflight.ok, preflight.to_json()
    assert preflight.data["passed"] is True
    proposed = dispatch_authoring_tool(
        controller,
        session,
        "prepare_solve_proposal",
        {},
        "solve-proposal",
    )
    assert proposed.ok, proposed.to_json()
    receipt = bridge.accept_from_gui_control(proposed.data["proposal_id"])
    controller.record_proposal_state("solve", receipt.state, receipt.message)
    assert receipt.state is ProposalState.SUCCEEDED
    record = session.current_result()
    assert record is not None
    return record.result


def export_line_result(result: object, root: Path, run_id: str, format_name: str):
    run = root / run_id
    exports = run / "exports"
    exports.mkdir(parents=True)
    outcome = export_results(
        result,
        (ExportFormat(format_name),),
        run_id=run_id,
        run_directory=run,
        exports_directory=exports,
    )
    assert outcome.ok, outcome.diagnostics
    return tuple(exports.iterdir())
