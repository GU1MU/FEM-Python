from __future__ import annotations

import pytest

from fem.application import (
    MeshEntityRef,
    ModelSession,
    NamedRegion,
    ScopedDefinitionBatch,
    UnitContext,
)
from fem.application.native_scope_materialization import (
    NATIVE_PART_OWNERSHIP_KEY,
    mesh_references_for_logical_entities,
)
from fem.core.model import AnalysisStep, FEMModel, SurfaceLoad
from fem.geometry import (
    BoxGeometry,
    LogicalEntityRef,
    namespace_part_logical_id,
)
from fem.mesh.settings import MeshSettings
from fem.selection import faces as mesh_faces
from fem_agent.authoring import ProposalState
from fem_agent.authoring_runtime import AuthoringWorkflowStage
from fem_agent.editing_authoring import (
    apply_edit_operation,
    create_edit_patch,
)
from tests.gui.test_agent_authoring_recovery_phase_a8 import (
    STEP_NAME,
    _apply_analysis_definitions,
    _dispatch,
    _production_controller,
    _solve_and_read_displacement,
)
from tests.helpers.mesh_builders import make_selection_hex_mesh
from tests.test_agent_authoring_phase_a4 import _session as _a4_session
from tests.test_agent_authoring_phase_a5 import _session as _a5_session
from fem_gui.agent_authoring import authoring_context_from_snapshot


def _edit_patch(
    session: ModelSession,
    object_type: str,
    target_id: str,
    changes: dict[str, object],
    *,
    step_name: str | None = None,
):
    snapshot = session.snapshot()
    return create_edit_patch(
        patch_id=f"patch-edit-{object_type}",
        agent_session_id="agent-edit",
        turn_id="turn-edit",
        source_tool_call_ids=("call-edit",),
        context=authoring_context_from_snapshot(snapshot),
        snapshot=snapshot,
        draft_revision=1,
        object_type=object_type,
        target_id=target_id,
        step_name=step_name,
        changes=changes,
    )[0]


def _surface_session() -> ModelSession:
    session = ModelSession()
    session.create_native_project_with_first_part(
        "模型-三维块",
        UnitContext("mm", "N", "MPa"),
        BoxGeometry("实体-三维块", 2.0, 3.0, 4.0),
        part_name="部件-三维块",
    )
    mesh = make_selection_hex_mesh()
    model = FEMModel(
        mesh,
        name="模型-三维块",
        metadata={
            NATIVE_PART_OWNERSHIP_KEY: {
                "P1": {
                    "node_ids": tuple(node.id for node in mesh.nodes),
                    "element_ids": tuple(
                        element.id for element in mesh.elements
                    ),
                }
            }
        },
    )
    task = session.prepare_agent_mesh_generation(
        "P1",
        MeshSettings(1.0, cell_shape="hexahedron"),
        "c" * 64,
        expected_session_revision=session.session_revision,
    )
    assert session.accept_agent_generated_model(task.token, model).accepted
    face = mesh_faces.boundary(mesh)[0]
    snapshot = session.snapshot()
    session.apply_scoped_definition_batch(
        ScopedDefinitionBatch(
            snapshot.session_revision,
            (
                NamedRegion(
                    "面-加载",
                    (MeshEntityRef.face(*face, part_id="P1"),),
                ),
            ),
            (),
            (),
            (),
            (
                AnalysisStep(
                    "分析步-静力",
                    surface_loads=(
                        SurfaceLoad(
                            "面-加载",
                            (1.0, 0.0, 0.0),
                            None,
                            "traction",
                            "载荷-表面",
                        ),
                    ),
                    metadata={"nlgeom": False},
                ),
            ),
        )
    )
    return session


def test_strict_production_action_requires_unit_direction_and_confirmation() -> None:
    session = _a5_session()
    controller, _bridge = _production_controller(session)
    created = _dispatch(
        controller,
        session,
        "apply_model_definition",
        {
            "action": "create_static_step",
            "parameters": {"name": STEP_NAME},
        },
        "strict-step",
    )
    assert created.ok, created.to_json()
    before = session.snapshot()

    rejected = _dispatch(
        controller,
        session,
        "apply_model_definition",
        {
            "action": "create_load",
            "parameters": {
                "name": "载荷-不完整",
                "step_name": STEP_NAME,
                "target_scope": "边-加载端",
                "entity_type": "edge",
                "load_type": "edge_traction",
                "component": None,
                "vector": [10.0, 0.0],
                "magnitude": None,
                "direction": "global_xy",
                "distribution": "uniform",
                "confirmed": True,
            },
        },
        "strict-missing-unit",
    )

    assert not rejected.ok
    assert session.snapshot() == before


def test_strict_production_action_supports_nodal_loads() -> None:
    session = _a5_session()
    snapshot = session.snapshot()
    session.apply_scoped_definition_batch(
        ScopedDefinitionBatch(
            snapshot.session_revision,
            tuple(snapshot.named_regions.values())
            + (
                NamedRegion(
                    "点-加载",
                    (MeshEntityRef.node(2, part_id="P1"),),
                ),
            ),
            tuple(snapshot.materials),
            tuple(snapshot.sections),
            tuple(snapshot.assignments),
            tuple(snapshot.steps),
        )
    )
    controller, _bridge = _production_controller(session)
    step = _dispatch(
        controller,
        session,
        "apply_model_definition",
        {
            "action": "create_static_step",
            "parameters": {"name": STEP_NAME},
        },
        "nodal-step",
    )
    assert step.ok, step.to_json()

    load = _dispatch(
        controller,
        session,
        "apply_model_definition",
        {
            "action": "create_load",
            "parameters": {
                "name": "载荷-节点",
                "step_name": STEP_NAME,
                "target_scope": "点-加载",
                "entity_type": "node",
                "load_type": "nodal",
                "component": 2,
                "vector": None,
                "magnitude": -5.0,
                "direction": "global_y",
                "unit": "N",
                "distribution": "concentrated",
                "confirmed": True,
            },
        },
        "nodal-load",
    )

    assert load.ok, load.to_json()
    stored = session.snapshot().steps[0].cloads[0]
    assert stored.name == "载荷-节点"
    assert stored.component == 2
    assert stored.value == -5.0


def test_definition_iteration_applies_directly_and_retains_history() -> None:
    session = _a5_session()
    controller, bridge = _production_controller(session)
    _apply_analysis_definitions(controller, session)
    _solve_and_read_displacement(controller, bridge, session)
    before = session.snapshot()
    assert any(run.has_result for run in before.runs)

    applied = _dispatch(
        controller,
        session,
        "apply_model_definition",
        {
            "action": "create_material",
            "parameters": {
                "name": "材料-附加",
                "properties": {"E": 70000.0, "nu": 0.33},
            },
        },
        "result-invalidating-material",
    )

    assert applied.ok, applied.to_json()
    assert applied.data["state"] == "succeeded"
    assert "proposal_id" not in applied.data
    after = session.snapshot()
    assert controller.stage is AuthoringWorkflowStage.PREFLIGHT_READY
    assert "材料-附加" in {item.name for item in after.materials}
    assert tuple(run.run_id for run in after.runs) == tuple(
        run.run_id for run in before.runs
    )
    assert any(run.has_result for run in after.runs)
    assert after.displayed_result_run_id is None
    assert not after.validations
    for run in before.runs:
        if run.has_result:
            assert session.result_for(run.run_id) is not None
    assert {
        "read_analysis_run_catalog",
        "read_accepted_result_catalog",
        "query_accepted_result",
        "run_native_preflight",
    }.issubset({item.name for item in controller.definitions})

    preflight = _dispatch(
        controller,
        session,
        "run_native_preflight",
        {},
        "iteration-preflight",
    )
    assert preflight.ok, preflight.to_json()
    assert controller.stage is AuthoringWorkflowStage.SOLVE_READY
    solve = _dispatch(
        controller,
        session,
        "prepare_solve_proposal",
        {},
        "iteration-solve",
    )
    assert solve.ok, solve.to_json()
    receipt = bridge.accept_from_gui_control(solve.data["proposal_id"])
    controller.record_proposal_state("solve", receipt.state, receipt.message)
    assert receipt.state is ProposalState.SUCCEEDED
    rerun = session.snapshot()
    assert len(rerun.runs) == len(before.runs) + 1
    assert len({run.name for run in rerun.runs}) == len(rerun.runs)
    assert all(
        session.result_for(run.run_id) is not None
        for run in rerun.runs
        if run.has_result
    )


def test_definition_edit_applies_directly_and_retains_history() -> None:
    session = _a5_session()
    controller, bridge = _production_controller(session)
    _apply_analysis_definitions(controller, session)
    _solve_and_read_displacement(controller, bridge, session)
    before = session.snapshot()
    before_vector = before.steps[0].edge_loads[0].vector

    applied = _dispatch(
        controller,
        session,
        "edit_model_object",
        {
            "object_type": "load",
            "target_id": "载荷-拉伸",
            "step_name": STEP_NAME,
            "changes": {
                "vector": [25.0, 0.0],
                "entity_type": "edge",
                "load_type": "edge_traction",
                "direction": "global_xy",
                "unit": "N/mm",
                "distribution": "uniform",
                "confirmed": True,
            },
        },
        "result-invalidating-load-edit",
    )

    assert applied.ok, applied.to_json()
    assert applied.data["state"] == "succeeded"
    assert "proposal_id" not in applied.data
    after = session.snapshot()
    assert controller.stage is AuthoringWorkflowStage.PREFLIGHT_READY
    assert before_vector != (25.0, 0.0)
    assert after.steps[0].edge_loads[0].vector == (25.0, 0.0)
    assert tuple(run.run_id for run in after.runs) == tuple(
        run.run_id for run in before.runs
    )
    assert any(run.has_result for run in after.runs)
    assert after.displayed_result_run_id is None
    assert not after.validations

    undone = bridge.undo_patch_from_gui_control(applied.data["patch_id"])
    restored = session.snapshot()
    assert undone.inverse_patch.invalidation_impact["results"] is False
    assert restored.steps[0].edge_loads[0].vector == before_vector
    assert tuple(run.run_id for run in restored.runs) == tuple(
        run.run_id for run in before.runs
    )
    assert all(
        session.result_for(run.run_id) is not None
        for run in restored.runs
        if run.has_result
    )


def test_two_dimensional_boundary_and_load_edits_fail_closed() -> None:
    session = _a5_session()
    controller, _bridge = _production_controller(session)
    _apply_analysis_definitions(controller, session)
    revision = session.session_revision

    with pytest.raises(ValueError, match="component exceeds"):
        _edit_patch(
            session,
            "boundary_condition",
            "位移-固定端",
            {
                "last_component": 3,
                "unit": "mm",
                "distribution": "uniform",
                "confirmed": True,
            },
            step_name=STEP_NAME,
        )
    with pytest.raises(ValueError, match="vector dimension"):
        _edit_patch(
            session,
            "load",
            "载荷-拉伸",
            {
                "vector": [10.0, 0.0, 0.0],
                "entity_type": "edge",
                "load_type": "edge_traction",
                "direction": "global_xy",
                "unit": "N/mm",
                "distribution": "uniform",
                "confirmed": True,
            },
            step_name=STEP_NAME,
        )
    with pytest.raises(ValueError, match="unit does not match"):
        _edit_patch(
            session,
            "load",
            "载荷-拉伸",
            {
                "vector": [20.0, 0.0],
                "entity_type": "edge",
                "load_type": "edge_traction",
                "direction": "global_xy",
                "unit": "kN/mm",
                "distribution": "uniform",
                "confirmed": True,
            },
            step_name=STEP_NAME,
        )

    assert session.session_revision == revision
    assert session.snapshot().steps[0].boundaries[0].last_component == 2
    assert session.snapshot().steps[0].edge_loads[0].vector == (10.0, 0.0)


def test_three_dimensional_surface_edit_sign_fails_closed() -> None:
    session = _surface_session()
    revision = session.session_revision

    with pytest.raises(ValueError, match="positive inward"):
        _edit_patch(
            session,
            "load",
            "载荷-表面",
            {
                "vector": None,
                "magnitude": 2.0,
                "entity_type": "surface",
                "load_type": "surface_pressure",
                "direction": "outward_normal",
                "unit": "MPa",
                "distribution": "uniform",
                "confirmed": True,
            },
            step_name=STEP_NAME,
        )

    assert session.session_revision == revision
    stored = session.snapshot().steps[0].surface_loads[0]
    assert stored.load_type == "traction"
    assert stored.vector == (1.0, 0.0, 0.0)


def test_scope_redirect_uses_unreferenced_topology_catalog_edge() -> None:
    session = _a4_session()
    controller, _bridge = _production_controller(session)
    topology = _dispatch(
        controller,
        session,
        "read_model_topology_context",
        {},
        "scope-edit-topology",
    )
    assert topology.ok, topology.to_json()
    outer = next(
        item
        for item in topology.data["entries"]
        if item["semantic_role"] == "boundary.outer-loop"
        and item["mesh_kind"] == "edge"
    )
    hole = next(
        item
        for item in topology.data["entries"]
        if item["semantic_role"] == "boundary.hole-loop"
        and item["mesh_kind"] == "edge"
    )
    snapshot = session.snapshot()
    outer_references = mesh_references_for_logical_entities(
        snapshot.artifact.model,
        (
            LogicalEntityRef(
                namespace_part_logical_id(
                    outer["part_id"],
                    outer["logical_id"],
                )
            ),
        ),
        mesh_kind=outer["mesh_kind"],
    )
    session.apply_scoped_definition_batch(
        ScopedDefinitionBatch(
            snapshot.session_revision,
            (NamedRegion("边-可编辑", outer_references),),
            (),
            (),
            (),
            (),
        )
    )
    before = session.snapshot()
    with pytest.raises(ValueError, match="expected_count"):
        _edit_patch(
            session,
            "named_region",
            "边-可编辑",
            {
                "part_id": hole["part_id"],
                "logical_ids": [hole["logical_id"]],
                "mesh_kind": hole["mesh_kind"],
                "expected_count": hole["matched_count"] + 1,
            },
        )
    assert session.session_revision == before.session_revision
    patch = _edit_patch(
        session,
        "named_region",
        "边-可编辑",
        {
            "part_id": hole["part_id"],
            "logical_ids": [hole["logical_id"]],
            "mesh_kind": hole["mesh_kind"],
            "expected_count": hole["matched_count"],
        },
    )
    apply_edit_operation(
        session,
        patch.operations[0],
        base_session_revision=patch.base_session_revision,
    )

    after = session.snapshot()
    redirected = after.named_regions["边-可编辑"].references
    assert set(redirected).isdisjoint(outer_references)
    assert len(redirected) == hole["matched_count"]
    assert after.session_revision == before.session_revision + 1
