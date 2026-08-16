from __future__ import annotations

from pathlib import Path

import pytest

from fem.application import ModelSession, UnitContext, run_static_preflight
from fem.application.preprocessing import generate_fem_model
from fem.geometry.recipes import WireGeometry, WireMember, WirePoint
from fem.io.project import load_project, save_project
from fem_agent.authoring import ProposalState
from fem_agent.definition_action_authoring import create_definition_change
from fem_agent.mesh_authoring import MeshIntent, create_mesh_proposal
from fem_agent.schemas import ExportFormat
from fem_agent.tools.exports import export_results
from fem_gui.agent_authoring import (
    AgentAuthoringBridge,
    AgentMeshTaskRequest,
    SessionGeometryAuthoringPort,
    authoring_context_from_snapshot,
)
from tests.gui.test_agent_authoring_recovery_phase_a8 import (
    _dispatch,
    _production_controller,
)


STEP_NAME = "分析步-拉伸"


def _wire() -> WireGeometry:
    return WireGeometry(
        "轴向杆",
        (
            WirePoint("Root", 0.0, 0.0, 0.0),
            WirePoint("Tip", 2.0, 0.0, 0.0),
        ),
        (WireMember("Bar", "Root", "Tip"),),
    )


def _meshed_session(line_element_type: str = "Truss2") -> ModelSession:
    session = ModelSession()
    session.create_native_project_with_first_part(
        "Agent 轴向杆",
        UnitContext("mm", "N", "MPa"),
        _wire(),
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
        proposal_id=f"proposal-phase3-{line_element_type}",
        agent_session_id="agent-phase3",
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


def _apply(
    controller: object,
    session: ModelSession,
    action: str,
    parameters: dict[str, object],
    suffix: str,
) -> None:
    outcome = _dispatch(
        controller,
        session,
        "apply_model_definition",
        {"action": action, "parameters": parameters},
        suffix,
    )
    assert outcome.ok, outcome.to_json()


def _apply_truss_definitions(
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
            ("create_static_step", {"name": STEP_NAME}, "step"),
            (
                "create_boundary_condition",
                {
                    "name": "位移-固定端",
                    "step_name": STEP_NAME,
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
                    "step_name": STEP_NAME,
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
                    "step_name": STEP_NAME,
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
                    "step_name": STEP_NAME,
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
                    "step_name": STEP_NAME,
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
        _apply(controller, session, action, parameters, suffix)


def _solve(controller: object, bridge: AgentAuthoringBridge, session: ModelSession):
    preflight = _dispatch(controller, session, "run_native_preflight", {}, "preflight")
    assert preflight.ok, preflight.to_json()
    assert preflight.data["passed"] is True
    proposed = _dispatch(
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


def _export(result: object, root: Path, run_id: str, format_name: str):
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


def test_agent_truss2_full_loop_matches_oracle_exports_and_reopens(
    real_gmsh,
    tmp_path: Path,
) -> None:
    del real_gmsh
    session = _meshed_session()
    controller, bridge = _production_controller(session)
    _apply_truss_definitions(controller, session)
    result = _solve(controller, bridge, session)

    tip_id = result.model.node_sets["点-自由端"].node_ids[0]
    root_id = result.model.node_sets["点-固定端"].node_ids[0]
    expected = 1000.0 * 2.0 / (210000.0 * 10.0)
    assert result.nodal_displacement(tip_id, component=1) == pytest.approx(expected)
    assert result.nodal_reaction(root_id, component=1) == pytest.approx(-1000.0)

    before_proposal = session.snapshot()
    proposed = _dispatch(
        controller,
        session,
        "apply_model_definition",
        {
            "action": "create_material",
            "parameters": {
                "name": "材料-候选",
                "properties": {"E": 70000.0, "nu": 0.33},
            },
        },
        "result-invalidating-proposal",
    )
    assert proposed.ok, proposed.to_json()
    assert "proposal_id" in proposed.data
    pending = session.snapshot()
    assert pending.session_revision == before_proposal.session_revision
    assert pending.materials == before_proposal.materials
    assert pending.sections == before_proposal.sections
    assert pending.steps == before_proposal.steps
    assert tuple(run.run_id for run in pending.runs) == tuple(
        run.run_id for run in before_proposal.runs
    )
    rejected = bridge.reject_from_gui_control(proposed.data["proposal_id"])
    controller.record_proposal_state(
        "destructive_edit",
        rejected.state,
        rejected.message,
    )
    assert rejected.state is ProposalState.REJECTED
    after_rejection = session.snapshot()
    assert after_rejection.session_revision == before_proposal.session_revision
    assert after_rejection.materials == before_proposal.materials
    assert after_rejection.sections == before_proposal.sections
    assert after_rejection.steps == before_proposal.steps
    assert any(run.has_result for run in after_rejection.runs)

    csv_files = _export(result, tmp_path / "artifacts", "csv-run", "csv")
    vtk_files = _export(result, tmp_path / "artifacts", "vtk-run", "vtk")
    assert any(path.suffix == ".csv" for path in csv_files)
    vtk_path = next(path for path in vtk_files if path.suffix == ".vtk")
    vtk_text = vtk_path.read_text(encoding="utf-8")
    assert "CELL_TYPES 1\n3\n" in vtk_text

    prepared = session.prepare_project_save()
    target = save_project(tmp_path / "agent-truss.femproj", prepared)
    assert session.accept_project_saved(prepared.token, target).accepted
    reopened = ModelSession()
    assert reopened.replace_from_snapshot(load_project(target).snapshot).accepted
    task = reopened.prepare_mesh_generation()
    regenerated = generate_fem_model(task)
    assert reopened.accept_generated_model(task.token, regenerated).accepted
    resumed, resumed_bridge = _production_controller(reopened)
    reopened_result = _solve(resumed, resumed_bridge, reopened)
    reopened_tip = reopened_result.model.node_sets["点-自由端"].node_ids[0]
    assert reopened_result.nodal_displacement(
        reopened_tip,
        component=1,
    ) == pytest.approx(expected)
    assert {element.type for element in reopened_result.model.mesh.elements} == {
        "Truss2"
    }


def test_truss_definition_rejections_and_underconstraint_are_atomic(real_gmsh) -> None:
    del real_gmsh
    session = _meshed_session()
    controller, _bridge = _production_controller(session)
    _apply(
        controller,
        session,
        "create_material",
        {"name": "材料-钢", "properties": {"E": 210000.0, "nu": 0.3}},
        "material",
    )
    before = session.snapshot()
    rejected = _dispatch(
        controller,
        session,
        "apply_model_definition",
        {
            "action": "create_section",
            "parameters": {
                "name": "截面-零面积",
                "material": "材料-钢",
                "section_type": "truss",
                "properties": {"area": 0.0},
            },
        },
        "zero-area",
    )
    assert not rejected.ok
    assert session.snapshot() == before

    component_session = _meshed_session()
    component_controller, _ = _production_controller(component_session)
    _apply_truss_definitions(component_controller, component_session)
    component_before = component_session.snapshot()
    component_parameters = {
        "name": "载荷-非法分量",
        "step_name": STEP_NAME,
        "target_scope": "点-自由端",
        "entity_type": "node",
        "load_type": "nodal",
        "component": 4,
        "vector": None,
        "magnitude": 1.0,
        "direction": "global_x",
        "unit": "N",
        "distribution": "concentrated",
        "confirmed": True,
    }
    with pytest.raises(ValueError, match="integer from 1 to 3"):
        create_definition_change(
            patch_id="patch-component-four",
            proposal_id="proposal-component-four",
            agent_session_id="agent-phase3",
            turn_id="turn-component-four",
            source_tool_call_ids=("call-component-four",),
            context=authoring_context_from_snapshot(component_before),
            snapshot=component_before,
            draft_revision=1,
            action="create_load",
            parameters=component_parameters,
        )
    rejected_component = _dispatch(
        component_controller,
        component_session,
        "apply_model_definition",
        {
            "action": "create_load",
            "parameters": component_parameters,
        },
        "component-four",
    )
    assert not rejected_component.ok
    assert component_session.snapshot() == component_before

    rejected_target = _dispatch(
        component_controller,
        component_session,
        "apply_model_definition",
        {
            "action": "create_load",
            "parameters": {
                "name": "载荷-错误目标",
                "step_name": STEP_NAME,
                "target_scope": "点-不存在",
                "entity_type": "node",
                "load_type": "nodal",
                "component": 1,
                "vector": None,
                "magnitude": 1.0,
                "direction": "global_x",
                "unit": "N",
                "distribution": "concentrated",
                "confirmed": True,
            },
        },
        "missing-load-target",
    )
    assert not rejected_target.ok
    assert component_session.snapshot() == component_before

    underconstrained = _meshed_session()
    underconstrained_controller, _ = _production_controller(underconstrained)
    _apply_truss_definitions(
        underconstrained_controller,
        underconstrained,
        include_tip_transverse_constraint=False,
    )
    validation = underconstrained.prepare_validation(STEP_NAME)
    report = run_static_preflight(
        validation.model,
        validation.step_name,
        token=validation.token,
    )
    assert not report.passed
    assert "static.stiffness.singular" in {
        diagnostic.code for diagnostic in report.diagnostics
    }

    uncovered = _meshed_session()
    uncovered_controller, _ = _production_controller(uncovered)
    _apply_truss_definitions(
        uncovered_controller,
        uncovered,
        include_section_assignment=False,
    )
    uncovered_validation = uncovered.prepare_validation(STEP_NAME)
    uncovered_report = run_static_preflight(
        uncovered_validation.model,
        uncovered_validation.step_name,
        token=uncovered_validation.token,
    )
    uncovered_codes = {
        diagnostic.code for diagnostic in uncovered_report.diagnostics
    }
    assert not uncovered_report.passed
    assert "definition.section.missing" in uncovered_codes
    assert "definition.section.unassigned_elements" in uncovered_codes


def test_truss_section_assignment_rejects_beam2_region(real_gmsh) -> None:
    del real_gmsh
    session = _meshed_session("Beam2")
    controller, _bridge = _production_controller(session)
    for action, parameters, suffix in (
        (
            "create_named_region",
            {
                "name": "域-梁",
                "part_id": "P1",
                "logical_ids": ["edge:Bar"],
                "mesh_kind": "element",
            },
            "beam-region",
        ),
        (
            "create_material",
            {"name": "材料-钢", "properties": {"E": 210000.0, "nu": 0.3}},
            "material",
        ),
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
    ):
        _apply(controller, session, action, parameters, suffix)
    before = session.snapshot()
    rejected = _dispatch(
        controller,
        session,
        "apply_model_definition",
        {
            "action": "assign_section",
            "parameters": {
                "section_name": "截面-拉杆",
                "region_name": "域-梁",
            },
        },
        "wrong-region",
    )
    assert not rejected.ok
    assert session.snapshot() == before
