from __future__ import annotations

from math import isfinite
from pathlib import Path

import pytest

from fem.application import BeamOrientation, ModelSession
from fem.application.preprocessing import generate_fem_model
from fem.elements.beam_section import parse_beam2_section
from fem.io.project import load_project, save_project
from fem_agent.authoring import ModelPatch, ProposalState
from fem_agent.definition_action_authoring import create_definition_change
from fem_gui.agent_authoring import authoring_context_from_snapshot
from tests.integration.test_agent_truss2_authoring_phase3 import (
    _apply,
    _export,
    _meshed_session,
    _solve,
)
from tests.gui.test_agent_authoring_recovery_phase_a8 import (
    _dispatch,
    _production_controller,
)


STEP_NAME = "分析步-梁"


def _base_scopes_and_material(controller: object, session: ModelSession) -> None:
    for action, parameters, suffix in (
        (
            "create_named_region",
            {
                "name": "点-固定端",
                "part_id": "P1",
                "logical_ids": ["point:Root"],
                "mesh_kind": "node",
                "expected_count": 1,
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
                "expected_count": 1,
            },
            "tip-region",
        ),
        (
            "create_named_region",
            {
                "name": "域-梁",
                "part_id": "P1",
                "logical_ids": ["edge:Bar"],
                "mesh_kind": "element",
                "expected_count": 1,
            },
            "beam-region",
        ),
        (
            "create_material",
            {
                "name": "材料-钢",
                "properties": {"E": 210000.0, "nu": 0.3},
            },
            "material",
        ),
    ):
        _apply(controller, session, action, parameters, suffix)


def _beam_definition_actions(
    controller: object,
    session: ModelSession,
) -> None:
    _base_scopes_and_material(controller, session)
    for action, parameters, suffix in (
        (
            "create_section",
            {
                "name": "截面-矩形梁",
                "material": "材料-钢",
                "section_type": "rectangle",
                "properties": {"height": 20.0, "width": 10.0},
            },
            "section",
        ),
        (
            "assign_section",
            {
                "section_name": "截面-矩形梁",
                "region_name": "域-梁",
                "local_y_reference": [0.0, 1.0, 0.0],
            },
            "assignment",
        ),
        ("create_static_step", {"name": STEP_NAME}, "step"),
        (
            "create_boundary_condition",
            {
                "name": "位移-固定端",
                "step_name": STEP_NAME,
                "target_scope": "点-固定端",
                "target_kind": "node_set",
                "first_component": 1,
                "last_component": 6,
                "value": 0.0,
                "unit": "mm",
                "distribution": "uniform",
                "confirmed": True,
            },
            "fixed",
        ),
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
            "axial-force",
        ),
        (
            "create_load",
            {
                "name": "载荷-横向力",
                "step_name": STEP_NAME,
                "target_scope": "点-自由端",
                "entity_type": "node",
                "load_type": "nodal",
                "component": 2,
                "vector": None,
                "magnitude": -100.0,
                "direction": "global_y",
                "unit": "N",
                "distribution": "concentrated",
                "confirmed": True,
            },
            "bending-force",
        ),
        (
            "create_load",
            {
                "name": "载荷-扭矩",
                "step_name": STEP_NAME,
                "target_scope": "点-自由端",
                "entity_type": "node",
                "load_type": "nodal",
                "component": 4,
                "vector": None,
                "magnitude": 500.0,
                "direction": "global_rx",
                "unit": "N*mm",
                "distribution": "concentrated",
                "confirmed": True,
            },
            "torque",
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
    ):
        _apply(controller, session, action, parameters, suffix)


def test_agent_beam2_full_loop_matches_axial_bending_and_torsion_oracles(
    real_gmsh,
    tmp_path: Path,
) -> None:
    del real_gmsh
    session = _meshed_session("Beam2")
    controller, bridge = _production_controller(session)
    _beam_definition_actions(controller, session)

    assignment = session.snapshot().assignments[0]
    assert assignment.beam_orientation == BeamOrientation((0.0, 1.0, 0.0))
    result = _solve(controller, bridge, session)
    tip_id = result.model.node_sets["点-自由端"].node_ids[0]
    root_id = result.model.node_sets["点-固定端"].node_ids[0]

    length = 2.0
    young = 210000.0
    poisson = 0.3
    section = parse_beam2_section(
        {"section_type": "rectangle", "height": 20.0, "width": 10.0}
    )
    shear = young / (2.0 * (1.0 + poisson))
    shear_y, _ = section.abaqus_b31_shear_rigidities(
        shear,
        poisson,
        length,
    )
    expected_axial = 1000.0 * length / (young * section.area)
    expected_bending = -100.0 * (
        length**3 / (4.0 * young * section.Izz) + length / shear_y
    )
    expected_bending_rotation = -100.0 * length**2 / (
        2.0 * young * section.Izz
    )
    expected_twist = 500.0 * length / (shear * section.J)

    assert result.nodal_displacement(tip_id, 1) == pytest.approx(expected_axial)
    assert result.nodal_displacement(tip_id, 2) == pytest.approx(expected_bending)
    assert result.nodal_displacement(tip_id, 6) == pytest.approx(
        expected_bending_rotation
    )
    assert result.nodal_displacement(tip_id, 4) == pytest.approx(expected_twist)
    assert result.nodal_reaction(root_id, 1) == pytest.approx(-1000.0)
    assert result.nodal_reaction(root_id, 2) == pytest.approx(100.0)
    assert result.nodal_reaction(root_id, 4) == pytest.approx(-500.0)
    assert all(isfinite(value) for value in result.U)

    before_proposal = session.snapshot()
    proposed = _dispatch(
        controller,
        session,
        "apply_model_definition",
        {
            "action": "create_section",
            "parameters": {
                "name": "截面-候选圆梁",
                "material": "材料-钢",
                "section_type": "solid_circle",
                "properties": {"radius": 5.0},
            },
        },
        "result-invalidating-beam-section",
    )
    assert proposed.ok, proposed.to_json()
    assert "proposal_id" in proposed.data
    pending = session.snapshot()
    assert pending.session_revision == before_proposal.session_revision
    assert pending.materials == before_proposal.materials
    assert pending.sections == before_proposal.sections
    assert pending.assignments == before_proposal.assignments
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
    assert after_rejection.assignments == before_proposal.assignments
    assert after_rejection.steps == before_proposal.steps
    assert any(run.has_result for run in after_rejection.runs)

    csv_files = _export(result, tmp_path / "artifacts", "beam-csv", "csv")
    vtk_files = _export(result, tmp_path / "artifacts", "beam-vtk", "vtk")
    assert any(path.suffix == ".csv" for path in csv_files)
    vtk_path = next(path for path in vtk_files if path.suffix == ".vtk")
    assert "CELL_TYPES 1\n3\n" in vtk_path.read_text(encoding="utf-8")

    prepared = session.prepare_project_save()
    target = save_project(tmp_path / "agent-beam.femproj", prepared)
    assert session.accept_project_saved(prepared.token, target).accepted
    reopened = ModelSession()
    assert reopened.replace_from_snapshot(load_project(target).snapshot).accepted
    reopened_snapshot = reopened.snapshot()
    assert reopened_snapshot.assignments[0].beam_orientation == BeamOrientation(
        (0.0, 1.0, 0.0)
    )
    assert reopened_snapshot.steps[0].boundaries[0].last_component == 6
    assert {load.component for load in reopened_snapshot.steps[0].cloads} == {
        1,
        2,
        4,
    }
    task = reopened.prepare_mesh_generation()
    regenerated = generate_fem_model(task)
    assert reopened.accept_generated_model(task.token, regenerated).accepted
    reopened_controller, reopened_bridge = _production_controller(reopened)
    reopened_result = _solve(reopened_controller, reopened_bridge, reopened)
    reopened_tip = reopened_result.model.node_sets["点-自由端"].node_ids[0]
    assert reopened_result.nodal_displacement(reopened_tip, 4) == pytest.approx(
        expected_twist
    )
    assert {element.type for element in reopened_result.model.mesh.elements} == {
        "Beam2"
    }


@pytest.mark.parametrize(
    ("section_type", "properties"),
    [
        ("solid_circle", {"radius": 5.0}),
        ("hollow_circle", {"outer_radius": 5.0, "inner_radius": 3.0}),
    ],
)
def test_agent_supports_existing_circular_beam_sections(
    real_gmsh,
    section_type: str,
    properties: dict[str, float],
) -> None:
    del real_gmsh
    session = _meshed_session("Beam2")
    controller, _ = _production_controller(session)
    _apply(
        controller,
        session,
        "create_material",
        {"name": "材料-钢", "properties": {"E": 210000.0, "nu": 0.3}},
        "material",
    )
    _apply(
        controller,
        session,
        "create_section",
        {
            "name": "截面-圆梁",
            "material": "材料-钢",
            "section_type": section_type,
            "properties": properties,
        },
        section_type,
    )
    section = session.snapshot().sections[0]
    assert section.section_type == section_type
    assert section.properties == properties


def test_beam_definition_rejections_are_strict_and_atomic(real_gmsh) -> None:
    del real_gmsh
    session = _meshed_session("Beam2")
    controller, _ = _production_controller(session)
    _base_scopes_and_material(controller, session)

    for suffix, parameters in (
        (
            "zero-height",
            {
                "name": "截面-非法矩形",
                "material": "材料-钢",
                "section_type": "rectangle",
                "properties": {"height": 0.0, "width": 10.0},
            },
        ),
        (
            "invalid-annulus",
            {
                "name": "截面-非法圆管",
                "material": "材料-钢",
                "section_type": "hollow_circle",
                "properties": {"outer_radius": 3.0, "inner_radius": 4.0},
            },
        ),
    ):
        before = session.snapshot()
        rejected = _dispatch(
            controller,
            session,
            "apply_model_definition",
            {"action": "create_section", "parameters": parameters},
            suffix,
        )
        assert not rejected.ok
        assert session.snapshot() == before

    _apply(
        controller,
        session,
        "create_section",
        {
            "name": "截面-矩形梁",
            "material": "材料-钢",
            "section_type": "rectangle",
            "properties": {"height": 20.0, "width": 10.0},
        },
        "section",
    )
    for suffix, direction in (
        ("zero-orientation", [0.0, 0.0, 0.0]),
        ("parallel-orientation", [1.0, 0.0, 0.0]),
    ):
        before = session.snapshot()
        rejected = _dispatch(
            controller,
            session,
            "apply_model_definition",
            {
                "action": "assign_section",
                "parameters": {
                    "section_name": "截面-矩形梁",
                    "region_name": "域-梁",
                    "local_y_reference": direction,
                },
            },
            suffix,
        )
        assert not rejected.ok
        assert session.snapshot() == before

    truss_session = _meshed_session("Truss2")
    truss_controller, _ = _production_controller(truss_session)
    _base_scopes_and_material(truss_controller, truss_session)
    _apply(
        truss_controller,
        truss_session,
        "create_section",
        {
            "name": "截面-矩形梁",
            "material": "材料-钢",
            "section_type": "rectangle",
            "properties": {"height": 20.0, "width": 10.0},
        },
        "beam-section",
    )
    before = truss_session.snapshot()
    rejected = _dispatch(
        truss_controller,
        truss_session,
        "apply_model_definition",
        {
            "action": "assign_section",
            "parameters": {
                "section_name": "截面-矩形梁",
                "region_name": "域-梁",
                "local_y_reference": [0.0, 1.0, 0.0],
            },
        },
        "beam-on-truss",
    )
    assert not rejected.ok
    assert truss_session.snapshot() == before

    _apply(
        truss_controller,
        truss_session,
        "create_section",
        {
            "name": "截面-拉杆",
            "material": "材料-钢",
            "section_type": "truss",
            "properties": {"area": 10.0},
        },
        "truss-section",
    )
    before = truss_session.snapshot()
    rejected = _dispatch(
        truss_controller,
        truss_session,
        "apply_model_definition",
        {
            "action": "assign_section",
            "parameters": {
                "section_name": "截面-拉杆",
                "region_name": "域-梁",
                "local_y_reference": [0.0, 1.0, 0.0],
            },
        },
        "orientation-on-truss",
    )
    assert not rejected.ok
    assert truss_session.snapshot() == before


def test_beam_six_dof_units_and_summaries_are_explicit(real_gmsh) -> None:
    del real_gmsh
    session = _meshed_session("Beam2")
    controller, _ = _production_controller(session)
    _base_scopes_and_material(controller, session)
    _apply(controller, session, "create_static_step", {"name": STEP_NAME}, "step")

    snapshot = session.snapshot()
    boundary = create_definition_change(
        patch_id="patch-fixed",
        proposal_id="proposal-fixed",
        agent_session_id="agent-phase4",
        turn_id="turn-fixed",
        source_tool_call_ids=("call-fixed",),
        context=authoring_context_from_snapshot(snapshot),
        snapshot=snapshot,
        draft_revision=1,
        action="create_boundary_condition",
        parameters={
            "name": "位移-固定端",
            "step_name": STEP_NAME,
            "target_scope": "点-固定端",
            "target_kind": "node_set",
            "first_component": 1,
            "last_component": 6,
            "value": 0.0,
            "unit": "mm",
            "distribution": "uniform",
            "confirmed": True,
        },
    ).value
    assert isinstance(boundary, ModelPatch)
    boundary_details = boundary.expected_changes["details"]
    assert boundary_details["translation_components"] == [1, 2, 3]
    assert boundary_details["rotation_components"] == [4, 5, 6]
    assert boundary_details["rotation_unit"] == "rad"

    torque = create_definition_change(
        patch_id="patch-torque",
        proposal_id="proposal-torque",
        agent_session_id="agent-phase4",
        turn_id="turn-torque",
        source_tool_call_ids=("call-torque",),
        context=authoring_context_from_snapshot(snapshot),
        snapshot=snapshot,
        draft_revision=1,
        action="create_load",
        parameters={
            "name": "载荷-扭矩",
            "step_name": STEP_NAME,
            "target_scope": "点-自由端",
            "entity_type": "node",
            "load_type": "nodal",
            "component": 4,
            "vector": None,
            "magnitude": 10.0,
            "direction": "global_rx",
            "unit": "N*mm",
            "distribution": "concentrated",
            "confirmed": True,
        },
    ).value
    assert isinstance(torque, ModelPatch)
    assert torque.expected_changes["details"]["nodal_family"] == "moment"

    for suffix, action, parameters in (
        (
            "mixed-nonzero",
            "create_boundary_condition",
            {
                "name": "位移-非法混合",
                "step_name": STEP_NAME,
                "target_scope": "点-固定端",
                "target_kind": "node_set",
                "first_component": 1,
                "last_component": 6,
                "value": 1.0,
                "unit": "mm",
                "distribution": "uniform",
                "confirmed": True,
            },
        ),
        (
            "moment-force-unit",
            "create_load",
            {
                "name": "载荷-错误力矩单位",
                "step_name": STEP_NAME,
                "target_scope": "点-自由端",
                "entity_type": "node",
                "load_type": "nodal",
                "component": 4,
                "vector": None,
                "magnitude": 10.0,
                "direction": "global_rx",
                "unit": "N",
                "distribution": "concentrated",
                "confirmed": True,
            },
        ),
        (
            "moment-force-direction",
            "create_load",
            {
                "name": "载荷-错误力矩方向",
                "step_name": STEP_NAME,
                "target_scope": "点-自由端",
                "entity_type": "node",
                "load_type": "nodal",
                "component": 4,
                "vector": None,
                "magnitude": 10.0,
                "direction": "global_x",
                "unit": "N*mm",
                "distribution": "concentrated",
                "confirmed": True,
            },
        ),
    ):
        before = session.snapshot()
        rejected = _dispatch(
            controller,
            session,
            "apply_model_definition",
            {"action": action, "parameters": parameters},
            suffix,
        )
        assert not rejected.ok
        assert session.snapshot() == before

    _apply(
        controller,
        session,
        "create_boundary_condition",
        {
            "name": "位移-纯转动",
            "step_name": STEP_NAME,
            "target_scope": "点-固定端",
            "target_kind": "node_set",
            "first_component": 4,
            "last_component": 6,
            "value": 0.0,
            "unit": "rad",
            "distribution": "uniform",
            "confirmed": True,
        },
        "pure-rotation",
    )
