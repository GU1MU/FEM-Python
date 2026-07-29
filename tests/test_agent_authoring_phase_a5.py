from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import pytest

from fem.application import (
    AnalysisRun,
    MeshEntityRef,
    ModelSession,
    NamedRegion,
    RegionAssignment,
    RunStatus,
    ScopedDefinitionBatch,
    SectionDefinition,
    UnitContext,
)
from fem.application.native_scope_materialization import NATIVE_PART_OWNERSHIP_KEY
from fem.core.mesh import Element2D, Mesh2D, Node2D
from fem.core.model import FEMModel, MaterialDefinition
from fem.geometry import RectangleGeometry
from fem.io.project import dumps_project, loads_project
from fem.mesh.settings import MeshSettings
from fem_agent.analysis_authoring import (
    AnalysisAuthoringError,
    ConfirmedDisplacement,
    ConfirmedLoad,
    ConfirmedResultRequest,
    LinearStaticAnalysis,
    create_analysis_definition_change,
    require_non_destructive_a5_batch,
)
from fem_agent.authoring import (
    AgentProposal,
    ModelOperation,
    ModelPatch,
    OperationKind,
    ProposalState,
)
from fem_agent.definition_authoring import scoped_definition_batch_from_operations
from fem_gui.agent_authoring import (
    AgentAuthoringBridge,
    SessionGeometryAuthoringPort,
    authoring_context_from_snapshot,
)


def _session() -> ModelSession:
    session = ModelSession()
    session.create_native_project_with_first_part(
        "模型-板",
        UnitContext("mm", "N", "MPa"),
        RectangleGeometry("实体-板", 10.0, 4.0),
        part_name="部件-板",
    )
    model = FEMModel(
        Mesh2D(
            nodes=[
                Node2D(1, 0.0, 0.0),
                Node2D(2, 10.0, 0.0),
                Node2D(3, 10.0, 4.0),
                Node2D(4, 0.0, 4.0),
            ],
            elements=[
                Element2D(1, (1, 2, 3), "Tri3"),
                Element2D(2, (1, 3, 4), "Tri3"),
            ],
        ),
        name="模型-板",
        metadata={
            NATIVE_PART_OWNERSHIP_KEY: {
                "P1": {
                    "node_ids": (1, 2, 3, 4),
                    "element_ids": (1, 2),
                }
            }
        },
    )
    task = session.prepare_agent_mesh_generation(
        "P1",
        MeshSettings(1.0),
        "a" * 64,
        expected_session_revision=session.session_revision,
    )
    assert session.accept_agent_generated_model(task.token, model).accepted
    session.apply_scoped_definition_batch(
        ScopedDefinitionBatch(
            session.session_revision,
            (
                NamedRegion(
                    "边-固定端",
                    (MeshEntityRef.edge(2, 2, (4, 1), part_id="P1"),),
                ),
                NamedRegion(
                    "边-加载端",
                    (MeshEntityRef.edge(1, 1, (2, 3), part_id="P1"),),
                ),
                NamedRegion(
                    "域-板体",
                    (
                        MeshEntityRef.element(1, part_id="P1"),
                        MeshEntityRef.element(2, part_id="P1"),
                    ),
                ),
            ),
            (MaterialDefinition("材料-钢", {"E": 210000.0, "nu": 0.3}),),
            (
                SectionDefinition(
                    "截面-平面应力",
                    "材料-钢",
                    "solid",
                    {"plane_type": "stress", "thickness": 1.0},
                ),
            ),
            (RegionAssignment("截面-平面应力", "域-板体"),),
            (),
        )
    )
    return session


def _analysis(
    *,
    load_unit: str = "N/mm",
    pressure: bool = False,
) -> LinearStaticAnalysis:
    load = (
        ConfirmedLoad(
            "载荷-拉伸",
            "分析步-静力",
            "边-加载端",
            "edge",
            "edge_pressure",
            None,
            (),
            -12.0,
            "outward_normal",
            load_unit,
            "uniform",
            True,
        )
        if pressure
        else ConfirmedLoad(
            "载荷-拉伸",
            "分析步-静力",
            "边-加载端",
            "edge",
            "edge_traction",
            None,
            (10.0, 0.0),
            None,
            "global_xy",
            load_unit,
            "uniform",
            True,
        )
    )
    return LinearStaticAnalysis(
        "分析步-静力",
        2,
        "static",
        False,
        (
            ConfirmedDisplacement(
                "位移-固定端",
                "分析步-静力",
                "边-固定端",
                "edge",
                1,
                2,
                0.0,
                "mm",
                "uniform",
                True,
            ),
        ),
        (load,),
        (
            ConfirmedResultRequest(
                "结果请求-位移反力",
                "分析步-静力",
                "field",
                "node",
                ("U", "RF"),
                ("mm", "N"),
                True,
            ),
            ConfirmedResultRequest(
                "结果请求-应力",
                "分析步-静力",
                "field",
                "element",
                ("S",),
                ("MPa",),
                True,
            ),
        ),
        True,
    )


def _change(
    session: ModelSession,
    analysis: LinearStaticAnalysis | None = None,
    *,
    id_suffix: str = "",
):
    snapshot = session.snapshot()
    return create_analysis_definition_change(
        patch_id=f"patch-a5{id_suffix}",
        proposal_id=f"proposal-a5{id_suffix}",
        agent_session_id="agent-a5",
        turn_id=f"turn-a5{id_suffix}",
        source_tool_call_ids=(f"call-a5{id_suffix}",),
        context=authoring_context_from_snapshot(snapshot),
        snapshot=snapshot,
        draft_revision=5,
        analysis=_analysis() if analysis is None else analysis,
    )


def _renamed_analysis(suffix: str) -> LinearStaticAnalysis:
    original = _analysis()
    step_name = f"{original.step_name}-{suffix}"
    return replace(
        original,
        step_name=step_name,
        displacements=tuple(
            replace(
                item,
                name=f"{item.name}-{suffix}",
                step_name=step_name,
            )
            for item in original.displacements
        ),
        loads=tuple(
            replace(
                item,
                name=f"{item.name}-{suffix}",
                step_name=step_name,
            )
            for item in original.loads
        ),
        results=tuple(
            replace(
                item,
                name=f"{item.name}-{suffix}",
                step_name=step_name,
            )
            for item in original.results
        ),
    )


def test_a5_complete_static_definition_applies_atomically_and_undoes() -> None:
    session = _session()
    projections = []
    port = SessionGeometryAuthoringPort(
        session,
        lambda: None,
        apply_definition_delta=projections.append,
    )
    bridge = AgentAuthoringBridge(port)
    bridge.bind_snapshot(session.snapshot())
    patch = _change(session)

    assert type(patch) is ModelPatch
    applied = bridge.apply_automatic_patch(patch)
    after = session.snapshot()
    step = after.steps[0]

    assert applied.undo_available
    assert len(projections) == 1
    assert step.name == "分析步-静力"
    assert step.procedure == "static"
    assert step.metadata["nlgeom"] is False
    assert step.boundaries[0].name == "位移-固定端"
    assert step.edge_loads[0].name == "载荷-拉伸"
    assert [item.name for item in step.outputs] == [
        "结果请求-位移反力",
        "结果请求-应力",
    ]
    assert after.artifact.model.steps == [step]

    serialized = dumps_project(session.prepare_project_save())
    loaded = loads_project(serialized).snapshot
    reopened = ModelSession()
    reopened.replace_from_snapshot(loaded)
    reopened_task = reopened.prepare_agent_mesh_generation(
        "P1",
        MeshSettings(1.0),
        "b" * 64,
        expected_session_revision=reopened.session_revision,
    )
    assert reopened.accept_agent_generated_model(
        reopened_task.token,
        deepcopy(after.artifact.model),
    ).accepted
    recompiled = reopened.snapshot().artifact.model.steps[0]
    assert recompiled.boundaries[0].name == "位移-固定端"
    assert recompiled.edge_loads[0].name == "载荷-拉伸"
    assert recompiled.outputs[0].name == "结果请求-位移反力"

    bridge.bind_snapshot(after)
    undone = bridge.undo_patch_from_gui_control(patch.patch_id)
    assert not undone.undo_available
    assert session.snapshot().steps == ()
    assert len(projections) == 2


def test_a5_missing_confirmation_dimension_and_unit_fail_closed() -> None:
    session = _session()
    before = session.snapshot()
    with pytest.raises(AnalysisAuthoringError, match="unit must exactly match"):
        _change(session, _analysis(load_unit="kN/mm"))
    assert session.snapshot().session_revision == before.session_revision

    with pytest.raises(AnalysisAuthoringError, match="not confirmed"):
        replace(
            _analysis().displacements[0],
            confirmed=False,
        )

    with pytest.raises(AnalysisAuthoringError, match="exceeds"):
        replace(
            _analysis(),
            displacements=(
                replace(_analysis().displacements[0], last_component=3),
            ),
        )
    duplicate = replace(
        _analysis(),
        results=(
            _analysis().results[0],
            replace(
                _analysis().results[1],
                name=_analysis().results[0].name,
            ),
        ),
    )
    with pytest.raises(AnalysisAuthoringError, match="next|exists"):
        _change(session, duplicate)


def test_a5_pressure_direction_has_persistent_kernel_sign() -> None:
    step = _analysis(pressure=True).to_step()
    assert step.edge_loads[0].magnitude == -12.0
    assert step.edge_loads[0].load_type == "pressure"
    with pytest.raises(AnalysisAuthoringError, match="positive inward"):
        replace(
            _analysis(pressure=True).loads[0],
            magnitude=12.0,
        )


def test_a5_current_node_and_three_dimensional_face_loads_compile_to_kernel() -> None:
    analysis = LinearStaticAnalysis(
        "分析步-静力3D",
        3,
        "static",
        False,
        (
            ConfirmedDisplacement(
                "位移-支座",
                "分析步-静力3D",
                "点-支座",
                "node_set",
                1,
                3,
                0.0,
                "mm",
                "uniform",
                True,
            ),
        ),
        (
            ConfirmedLoad(
                "载荷-节点",
                "分析步-静力3D",
                "点-加载",
                "node",
                "nodal",
                2,
                (),
                -5.0,
                "global_y",
                "N",
                "concentrated",
                True,
            ),
            ConfirmedLoad(
                "载荷-端面",
                "分析步-静力3D",
                "面-端面",
                "surface",
                "surface_traction",
                None,
                (1.0, 0.0, 0.0),
                None,
                "global_xyz",
                "MPa",
                "uniform",
                True,
            ),
        ),
        (
            ConfirmedResultRequest(
                "结果请求-三维位移",
                "分析步-静力3D",
                "field",
                "node",
                ("U",),
                ("mm",),
                True,
            ),
        ),
        True,
    )

    step = analysis.to_step()
    assert step.cloads[0].name == "载荷-节点"
    assert step.cloads[0].component == 2
    assert step.surface_loads[0].name == "载荷-端面"
    assert step.surface_loads[0].vector == (1.0, 0.0, 0.0)


def test_a5_tampered_nlgeom_and_widened_json_types_are_rejected() -> None:
    session = _session()
    patch = _change(session)
    snapshot = session.snapshot()
    definitions = deepcopy(
        patch.operations[1].parameters["definitions"]
    )
    definitions["steps"][0]["metadata"]["nlgeom"] = "false"
    operations = (
        patch.operations[0],
        ModelOperation(
            OperationKind.UPSERT_MODEL_DEFINITIONS,
            {"definitions": definitions},
        ),
    )
    batch = scoped_definition_batch_from_operations(
        operations,
        snapshot,
        base_session_revision=snapshot.session_revision,
    )
    with pytest.raises(AnalysisAuthoringError, match="NLGEOM"):
        require_non_destructive_a5_batch(snapshot, batch)

    definitions["steps"][0]["boundaries"][0]["value"] = "0.0"
    with pytest.raises(TypeError, match="finite JSON number"):
        scoped_definition_batch_from_operations(
            (
                operations[0],
                ModelOperation(
                    OperationKind.UPSERT_MODEL_DEFINITIONS,
                    {"definitions": definitions},
                ),
            ),
            snapshot,
            base_session_revision=snapshot.session_revision,
        )


def test_a5_replays_legacy_a4_definition_operation_without_steps() -> None:
    session = _session()
    session.replace_model_definitions(
        session.snapshot().materials,
        session.snapshot().sections,
        session.snapshot().assignments,
        (_analysis().to_step(),),
    )
    snapshot = session.snapshot()
    legacy = deepcopy(
        _change(_session()).operations[1].parameters["definitions"]
    )
    del legacy["steps"]
    batch = scoped_definition_batch_from_operations(
        (
            ModelOperation(
                OperationKind.UPSERT_NAMED_REGIONS,
                {
                    "regions": deepcopy(
                        _change(_session()).operations[0].parameters["regions"]
                    )
                },
            ),
            ModelOperation(
                OperationKind.UPSERT_MODEL_DEFINITIONS,
                {"definitions": legacy},
            ),
        ),
        snapshot,
        base_session_revision=snapshot.session_revision,
    )
    assert batch.steps == snapshot.steps


def test_a5_overwrite_reject_and_stale_proposal_leave_gui_state_unchanged() -> None:
    session = _session()
    first = _change(session)
    batch = scoped_definition_batch_from_operations(
        first.operations,
        session.snapshot(),
        base_session_revision=session.session_revision,
    )
    session.apply_scoped_definition_batch(batch)
    port = SessionGeometryAuthoringPort(session, lambda: None)
    bridge = AgentAuthoringBridge(port)
    bridge.bind_snapshot(session.snapshot())
    proposal = _change(session, _renamed_analysis("2"))

    assert type(proposal) is AgentProposal
    assert proposal.invalidation_impact["results"] is False
    before = session.snapshot()
    bridge.register_proposal(proposal)
    receipt = bridge.reject_from_gui_control(proposal.proposal_id)
    assert receipt.state is ProposalState.REJECTED
    assert session.snapshot().session_revision == before.session_revision
    assert session.snapshot().steps == before.steps

    other = _change(
        session,
        _renamed_analysis("2"),
        id_suffix="-stale",
    )
    bridge.register_proposal(other)
    session.replace_model_definitions(
        (*before.materials, MaterialDefinition("材料-其他", {"E": 1.0})),
        before.sections,
        before.assignments,
        before.steps,
    )
    stale_ids = bridge.bind_snapshot(session.snapshot())
    assert other.proposal_id in stale_ids
    assert bridge.state(other.proposal_id) is ProposalState.STALE


def test_a5_invalid_scope_exception_is_atomic() -> None:
    session = _session()
    patch = _change(session)
    snapshot = session.snapshot()
    definitions = deepcopy(
        patch.operations[1].parameters["definitions"]
    )
    definitions["steps"][0]["boundaries"][0]["target"] = "边-不存在"
    batch = scoped_definition_batch_from_operations(
        (
            patch.operations[0],
            ModelOperation(
                OperationKind.UPSERT_MODEL_DEFINITIONS,
                {"definitions": definitions},
            ),
        ),
        snapshot,
        base_session_revision=snapshot.session_revision,
    )
    with pytest.raises(Exception, match="不存在|unknown|region|scope"):
        session.apply_scoped_definition_batch(batch)
    after = session.snapshot()
    assert after.session_revision == snapshot.session_revision
    assert after.steps == ()


def test_a5_valid_result_forces_gui_confirmation() -> None:
    session = _session()
    snapshot = session.snapshot()
    artifact = snapshot.artifact
    run = AnalysisRun(
        "run-a5",
        "作业-旧结果",
        "分析步-旧",
        artifact.artifact_id,
        artifact.model_revision,
        status=RunStatus.SUCCEEDED,
        result_id="result-a5",
    )
    result_snapshot = replace(snapshot, runs=(run,))
    change = create_analysis_definition_change(
        patch_id="patch-with-result",
        proposal_id="proposal-with-result",
        agent_session_id="agent-a5",
        turn_id="turn-with-result",
        source_tool_call_ids=("call-with-result",),
        context=authoring_context_from_snapshot(result_snapshot),
        snapshot=result_snapshot,
        draft_revision=5,
        analysis=_analysis(),
    )

    assert type(change) is AgentProposal
    assert change.invalidation_impact["results"] is True
    assert change.display_summary["confirm_label"] == "确认修改"
