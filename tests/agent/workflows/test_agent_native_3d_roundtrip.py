from __future__ import annotations

import numpy as np
import pytest

from fem.application import ModelSession, NamedRegion, UnitContext
from fem.application.preprocessing import generate_fem_model
from fem.geometry import ExtrudedGeometry, LogicalEntityRef, RectangleGeometry, namespace_part_logical_id
from fem.io.project import dumps_project, loads_project
from fem.mesh.settings import MeshSettings
from fem_agent.authoring import ProposalState
from fem_agent.authoring_runtime import AuthoringWorkflowStage

from tests.helpers.agent_native_3d_workflows import _dispatch, _record_mesh_requirements, _production_controller


pytestmark = pytest.mark.integration


def _apply_agent_definition(
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


@pytest.mark.gmsh
def test_agent_controller_bridge_completes_real_3d_loop_and_reopens(
    real_gmsh,
) -> None:
    del real_gmsh
    recipe = ExtrudedGeometry(RectangleGeometry("Agent loop", 1.0, 1.0), 1.0)
    session = ModelSession()
    session.create_native_project_with_first_part(
        "Agent 3D loop",
        UnitContext("mm", "N", "MPa"),
        recipe,
        part_name="Continuum part",
    )
    controller, bridge = _production_controller(session)
    _record_mesh_requirements(
        controller,
        session,
        cell_shape="tetrahedron",
        order=1,
        global_size=0.35,
    )

    proposed_mesh = _dispatch(
        controller,
        session,
        "prepare_mesh_proposal",
        {},
        "agent-loop-mesh",
    )
    assert proposed_mesh.ok, proposed_mesh.to_json()
    assert not session.snapshot().mesh_current
    mesh_receipt = bridge.accept_from_gui_control(
        proposed_mesh.data["proposal_id"],
    )
    controller.record_proposal_state(
        "mesh",
        mesh_receipt.state,
        mesh_receipt.message,
    )

    assert mesh_receipt.state is ProposalState.SUCCEEDED
    assert controller.stage is AuthoringWorkflowStage.DEFINITIONS_READY
    assert {element.type for element in session.snapshot().model.mesh.elements} == {
        "Tet4"
    }
    definition_tool = next(
        item
        for item in controller.definitions
        if item.name == "apply_model_definition"
    )
    section_schema = next(
        item
        for item in definition_tool.parameters["oneOf"]
        if item["properties"]["action"]["const"] == "create_section"
    )
    assert any(
        variant["properties"].get("section_type", {}).get("const") == "solid"
        for variant in section_schema["properties"]["parameters"]["oneOf"]
    )

    topology = _dispatch(
        controller,
        session,
        "read_model_topology_context",
        {},
        "agent-loop-topology",
    )
    assert topology.ok, topology.to_json()

    def region_parameters(
        name: str,
        logical_id: str,
        mesh_kind: str,
    ) -> dict[str, object]:
        entry = next(
            item
            for item in topology.data["entries"]
            if item["logical_id"] == logical_id
            and item["mesh_kind"] == mesh_kind
        )
        assert entry["matched_count"] > 0
        return {
            "name": name,
            "part_id": entry["part_id"],
            "logical_ids": [entry["logical_id"]],
            "mesh_kind": entry["mesh_kind"],
        }

    actions = (
        (
            "create_named_region",
            region_parameters("域-实体", "body:domain", "element"),
            "agent-loop-body",
        ),
        (
            "create_named_region",
            region_parameters("面-固定端", "face:side/left", "face"),
            "agent-loop-fixed-face",
        ),
        (
            "create_named_region",
            region_parameters("面-加载端", "face:side/right", "face"),
            "agent-loop-loaded-face",
        ),
        (
            "create_material",
            {
                "name": "材料-线弹性",
                "properties": {"E": 1000.0, "nu": 0.0},
            },
            "agent-loop-material",
        ),
        (
            "create_section",
            {
                "name": "截面-实体",
                "material": "材料-线弹性",
                "section_type": "solid",
                "properties": {},
            },
            "agent-loop-section",
        ),
        (
            "assign_section",
            {
                "section_name": "截面-实体",
                "region_name": "域-实体",
            },
            "agent-loop-assignment",
        ),
        (
            "create_static_step",
            {"name": "分析步-静力"},
            "agent-loop-step",
        ),
        (
            "create_boundary_condition",
            {
                "name": "位移-固定端",
                "step_name": "分析步-静力",
                "target_scope": "面-固定端",
                "target_kind": "surface",
                "first_component": 1,
                "last_component": 3,
                "value": 0.0,
                "unit": "mm",
                "distribution": "uniform",
                "confirmed": True,
            },
            "agent-loop-boundary",
        ),
        (
            "create_load",
            {
                "name": "载荷-拉伸",
                "step_name": "分析步-静力",
                "target_scope": "面-加载端",
                "entity_type": "surface",
                "load_type": "surface_traction",
                "component": None,
                "vector": [10.0, 0.0, 0.0],
                "magnitude": None,
                "direction": "global_xyz",
                "unit": "MPa",
                "distribution": "uniform",
                "confirmed": True,
            },
            "agent-loop-load",
        ),
        (
            "create_result_request",
            {
                "name": "结果请求-节点",
                "step_name": "分析步-静力",
                "target": "node",
                "variables": ["U", "RF"],
                "units": ["mm", "N"],
                "confirmed": True,
            },
            "agent-loop-output",
        ),
    )
    for action, parameters, suffix in actions:
        _apply_agent_definition(
            controller,
            session,
            action,
            parameters,
            suffix,
        )

    preflight = _dispatch(
        controller,
        session,
        "run_native_preflight",
        {},
        "agent-loop-preflight",
    )
    assert preflight.ok, preflight.to_json()
    assert preflight.data["passed"] is True
    proposed_solve = _dispatch(
        controller,
        session,
        "prepare_solve_proposal",
        {},
        "agent-loop-solve",
    )
    assert proposed_solve.ok, proposed_solve.to_json()
    solve_receipt = bridge.accept_from_gui_control(
        proposed_solve.data["proposal_id"],
    )
    controller.record_proposal_state(
        "solve",
        solve_receipt.state,
        solve_receipt.message,
    )

    assert solve_receipt.state is ProposalState.SUCCEEDED
    assert controller.stage is AuthoringWorkflowStage.RESULTS_READY
    result_record = session.current_result()
    assert result_record is not None
    result = result_record.result
    loaded_ids = sorted(
        {
            node_id
            for face in result.model.surfaces["面-加载端"].faces
            for node_id in face.node_ids
        }
    )
    assert np.array(
        [result.nodal_displacement(node_id, 1) for node_id in loaded_ids],
    ) == pytest.approx(0.01, abs=1.0e-10)
    assert float(result.reactions[0::3].sum()) == pytest.approx(-10.0, abs=1.0e-9)

    save_request = _dispatch(
        controller,
        session,
        "request_project_save",
        {},
        "agent-loop-save",
    )
    assert save_request.ok, save_request.to_json()
    assert save_request.data["proposal_id"]
    reopened_snapshot = loads_project(
        dumps_project(session.prepare_project_save()),
    ).snapshot
    reopened = ModelSession()
    assert reopened.replace_from_snapshot(reopened_snapshot).accepted
    assert reopened.snapshot().sections[0].section_type == "solid"
    assert reopened.snapshot().named_regions == session.snapshot().named_regions
    reopened_controller, reopened_bridge = _production_controller(reopened)
    _record_mesh_requirements(
        reopened_controller,
        reopened,
        cell_shape="tetrahedron",
        order=1,
        global_size=0.35,
    )
    reopened_mesh_proposal = _dispatch(
        reopened_controller,
        reopened,
        "prepare_mesh_proposal",
        {},
        "agent-loop-reopened-mesh",
    )
    assert reopened_mesh_proposal.ok, reopened_mesh_proposal.to_json()
    reopened_mesh_receipt = reopened_bridge.accept_from_gui_control(
        reopened_mesh_proposal.data["proposal_id"],
    )
    reopened_controller.record_proposal_state(
        "mesh",
        reopened_mesh_receipt.state,
        reopened_mesh_receipt.message,
    )
    assert reopened_mesh_receipt.state is ProposalState.SUCCEEDED
    reopened_preflight = _dispatch(
        reopened_controller,
        reopened,
        "run_native_preflight",
        {},
        "agent-loop-reopened-preflight",
    )
    assert reopened_preflight.ok, reopened_preflight.to_json()
    reopened_proposal = _dispatch(
        reopened_controller,
        reopened,
        "prepare_solve_proposal",
        {},
        "agent-loop-reopened-solve",
    )
    assert reopened_proposal.ok, reopened_proposal.to_json()
    reopened_receipt = reopened_bridge.accept_from_gui_control(
        reopened_proposal.data["proposal_id"],
    )
    reopened_controller.record_proposal_state(
        "solve",
        reopened_receipt.state,
        reopened_receipt.message,
    )
    reopened_result = reopened.current_result()

    assert reopened_receipt.state is ProposalState.SUCCEEDED
    assert reopened_result is not None
    assert reopened_result.result.U == pytest.approx(result.U, abs=1.0e-12)


@pytest.mark.gmsh
def test_logical_face_region_survives_remesh_and_reopen(
    real_gmsh,
) -> None:
    del real_gmsh
    recipe = ExtrudedGeometry(RectangleGeometry("Stable faces", 1.0, 0.8), 0.6)
    face_reference = LogicalEntityRef(
        namespace_part_logical_id("P1", "face:side/right")
    )
    regions = (
        NamedRegion("LoadedFace", (face_reference,)),
    )
    session = ModelSession()
    session.create_native_project_with_first_part(
        "Stable face model",
        UnitContext("mm", "N", "MPa"),
        recipe,
        part_name="Extruded part",
    )
    session.replace_named_regions(regions)
    session.replace_part_mesh_settings(
        "P1",
        MeshSettings(0.4, cell_shape="tetrahedron", strict_cell_shape=True),
    )
    first_task = session.prepare_mesh_generation()
    assert session.accept_generated_model(
        first_task.token,
        generate_fem_model(first_task),
    ).accepted
    first_count = len(session.snapshot().model.surfaces["LoadedFace"].faces)

    reopened = loads_project(dumps_project(session.prepare_project_save())).snapshot
    restored = ModelSession()
    restored.replace_from_snapshot(reopened)
    restored.replace_part_mesh_settings(
        "P1",
        MeshSettings(0.25, cell_shape="tetrahedron", strict_cell_shape=True),
    )
    second_task = restored.prepare_mesh_generation()
    assert restored.accept_generated_model(
        second_task.token,
        generate_fem_model(second_task),
    ).accepted
    model = restored.snapshot().model
    loaded = model.surfaces["LoadedFace"]
    nodes = {int(node.id): node for node in model.mesh.nodes}

    assert first_count > 0
    assert loaded.faces
    assert len(loaded.faces) > first_count
    assert all(
        float(nodes[node_id].x) == pytest.approx(1.0, abs=1.0e-8)
        for face in loaded.faces
        for node_id in face.node_ids
    )
    assert restored.snapshot().named_regions["LoadedFace"].references == (
        face_reference,
    )
