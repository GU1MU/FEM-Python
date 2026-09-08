from __future__ import annotations

from tests.helpers.agent_line_fixtures import (
    TRUSS_STEP_NAME,
    make_meshed_line_session,
    apply_definition_action,
    apply_truss_definitions,
    solve_authoring_session,
    export_line_result,
)

from pathlib import Path

import pytest

from fem.application import ModelSession, run_static_preflight
from fem.application.preprocessing import generate_fem_model
from fem.io.project import load_project, save_project
from fem_agent.authoring import ProposalState
from fem_agent.definition_action_authoring import create_definition_change
from fem_gui.agent_authoring import authoring_context_from_snapshot
from tests.helpers.agent_authoring_workflows import (
    dispatch_authoring_tool,
    make_authoring_controller,
)


pytestmark = pytest.mark.integration


def test_agent_truss2_full_loop_matches_oracle_exports_and_reopens(
    real_gmsh,
    tmp_path: Path,
) -> None:
    del real_gmsh
    session = make_meshed_line_session()
    controller, bridge = make_authoring_controller(session)
    apply_truss_definitions(controller, session)
    result = solve_authoring_session(controller, bridge, session)

    tip_id = result.model.node_sets["点-自由端"].node_ids[0]
    root_id = result.model.node_sets["点-固定端"].node_ids[0]
    expected = 1000.0 * 2.0 / (210000.0 * 10.0)
    assert result.nodal_displacement(tip_id, component=1) == pytest.approx(expected)
    assert result.nodal_reaction(root_id, component=1) == pytest.approx(-1000.0)

    before_edit = session.snapshot()
    applied = dispatch_authoring_tool(
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
        "create-additional-material",
    )
    assert applied.ok, applied.to_json()
    assert applied.data["state"] == "succeeded"
    assert applied.data["undo_available"] is True
    patch_id = applied.data["patch_id"]
    assert bridge.can_undo_patch(patch_id)
    after_edit = session.snapshot()
    assert after_edit.session_revision == before_edit.session_revision + 1
    created = next(item for item in after_edit.materials if item.name == "材料-候选")
    assert dict(created.properties) == {"E": 70000.0, "nu": 0.33}
    assert len(after_edit.materials) == len(before_edit.materials) + 1
    assert after_edit.sections == before_edit.sections
    assert after_edit.assignments == before_edit.assignments
    assert after_edit.steps == before_edit.steps
    assert tuple(run.run_id for run in after_edit.runs) == tuple(
        run.run_id for run in before_edit.runs
    )
    assert after_edit.displayed_result_run_id is None
    assert all(
        session.result_for(run.run_id) is not None
        for run in before_edit.runs if run.has_result
    )

    undone = bridge.undo_patch_from_gui_control(patch_id)
    assert undone.state.value == "undone"
    assert not undone.undo_available
    assert not bridge.can_undo_patch(patch_id)
    restored = session.snapshot()
    assert restored.materials == before_edit.materials
    assert restored.sections == before_edit.sections
    assert restored.assignments == before_edit.assignments
    assert restored.steps == before_edit.steps
    assert tuple(run.run_id for run in restored.runs) == tuple(
        run.run_id for run in before_edit.runs
    )
    assert all(
        session.result_for(run.run_id) is not None
        for run in before_edit.runs if run.has_result
    )

    csv_files = export_line_result(result, tmp_path / "artifacts", "csv-run", "csv")
    vtk_files = export_line_result(result, tmp_path / "artifacts", "vtk-run", "vtk")
    assert any(path.suffix == ".csv" for path in csv_files)
    vtk_path = next(path for path in vtk_files if path.suffix == ".vtk")
    vtk_text = vtk_path.read_text(encoding="utf-8")
    assert "CELL_TYPES 1\n3\n" in vtk_text

    prepared = session.prepare_project_save()
    target = save_project(tmp_path / "agent-truss.femproj", prepared)
    assert session.accept_project_saved(prepared.token, target).accepted
    reopened = ModelSession()
    assert reopened.replace_from_snapshot(load_project(target).snapshot).accepted
    part = reopened.snapshot().parts[0]
    assert part.mesh_settings is not None
    task = reopened.prepare_agent_mesh_generation(
        part.id,
        part.mesh_settings,
        "b" * 64,
        expected_session_revision=reopened.session_revision,
    )
    regenerated = generate_fem_model(task)
    assert reopened.accept_agent_generated_model(task.token, regenerated).accepted
    resumed, resumed_bridge = make_authoring_controller(reopened)
    reopened_result = solve_authoring_session(resumed, resumed_bridge, reopened)
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
    session = make_meshed_line_session()
    controller, _bridge = make_authoring_controller(session)
    apply_definition_action(
        controller,
        session,
        "create_material",
        {"name": "材料-钢", "properties": {"E": 210000.0, "nu": 0.3}},
        "material",
    )
    before = session.snapshot()
    rejected = dispatch_authoring_tool(
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

    component_session = make_meshed_line_session()
    component_controller, _ = make_authoring_controller(component_session)
    apply_truss_definitions(component_controller, component_session)
    component_before = component_session.snapshot()
    component_parameters = {
        "name": "载荷-非法分量",
        "step_name": TRUSS_STEP_NAME,
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
    rejected_component = dispatch_authoring_tool(
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

    rejected_target = dispatch_authoring_tool(
        component_controller,
        component_session,
        "apply_model_definition",
        {
            "action": "create_load",
            "parameters": {
                "name": "载荷-错误目标",
                "step_name": TRUSS_STEP_NAME,
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

    underconstrained = make_meshed_line_session()
    underconstrained_controller, _ = make_authoring_controller(underconstrained)
    apply_truss_definitions(
        underconstrained_controller,
        underconstrained,
        include_tip_transverse_constraint=False,
    )
    validation = underconstrained.prepare_validation(TRUSS_STEP_NAME)
    report = run_static_preflight(
        validation.model,
        validation.step_name,
        token=validation.token,
    )
    assert not report.passed
    assert "static.stiffness.singular" in {
        diagnostic.code for diagnostic in report.diagnostics
    }

    uncovered = make_meshed_line_session()
    uncovered_controller, _ = make_authoring_controller(uncovered)
    apply_truss_definitions(
        uncovered_controller,
        uncovered,
        include_section_assignment=False,
    )
    uncovered_validation = uncovered.prepare_validation(TRUSS_STEP_NAME)
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
    session = make_meshed_line_session("Beam2")
    controller, _bridge = make_authoring_controller(session)
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
        apply_definition_action(controller, session, action, parameters, suffix)
    before = session.snapshot()
    rejected = dispatch_authoring_tool(
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
