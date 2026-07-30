from __future__ import annotations

import json

import pytest

from fem.application import (
    MeshEntityRef,
    ModelSession,
    NamedRegion,
    RegionAssignment,
    ScopedDefinitionBatch,
    SectionDefinition,
    UnitContext,
)
from fem.application.native_scope_materialization import (
    NATIVE_PART_OWNERSHIP_KEY,
)
from fem.core.mesh import Element2D, Mesh2D, Node2D
from fem.core.model import (
    AnalysisStep,
    DisplacementConstraint,
    EdgeLoad,
    FEMModel,
    MaterialDefinition,
    OutputRequest,
)
from fem.geometry import RectangleGeometry
from fem.mesh.settings import MeshSettings
from fem_agent.authoring import (
    AuthoringAuthorizationError,
    OperationKind,
    ProposalKind,
    ProposalState,
)
from fem_agent.authoring_runtime import AuthoringWorkflowStage
from fem_agent.deletion_authoring import (
    apply_delete_operation,
    create_delete_proposal,
    deletable_object_catalog,
)
from fem_agent.editing_authoring import (
    apply_edit_operation,
    create_edit_proposal,
    editable_object_catalog,
)
from fem_agent.result_authoring import AgentResultQueryBridge
from fem_agent.tools.registry import ToolExecutionContext
from fem_gui.agent_authoring import (
    AgentAuthoringBridge,
    SessionGeometryAuthoringPort,
    SessionResultQueryPort,
    authoring_context_from_snapshot,
    create_session_authoring_workflow_controller,
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
    session.accept_agent_generated_model(task.token, model)
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
            (
                AnalysisStep(
                    "分析步-静力",
                    boundaries=(
                        DisplacementConstraint(
                            "边-固定端",
                            1,
                            2,
                            name="位移-固定端",
                            target_kind="edge",
                        ),
                    ),
                    edge_loads=(
                        EdgeLoad(
                            "边-加载端",
                            (10.0, 0.0),
                            name="载荷-拉伸",
                        ),
                    ),
                    outputs=(
                        OutputRequest(
                            "field",
                            "node",
                            ("U", "RF"),
                            name="结果请求-位移反力",
                        ),
                        OutputRequest(
                            "field",
                            "element",
                            ("S",),
                            name="结果请求-应力",
                        ),
                        OutputRequest("field", "node", ("U",)),
                    ),
                ),
            ),
        )
    )
    return session


def _proposal(
    session: ModelSession,
    object_type: str,
    target_id: str,
    step_name: str | None = None,
):
    snapshot = session.snapshot()
    return create_delete_proposal(
        proposal_id=f"proposal-delete-{object_type}",
        agent_session_id="agent-delete",
        turn_id="turn-delete",
        source_tool_call_ids=("call-delete",),
        context=authoring_context_from_snapshot(snapshot),
        snapshot=snapshot,
        draft_revision=1,
        object_type=object_type,
        target_id=target_id,
        step_name=step_name,
    )[0]


def _edit_proposal(
    session: ModelSession,
    object_type: str,
    target_id: str,
    changes: dict[str, object],
    step_name: str | None = None,
):
    snapshot = session.snapshot()
    return create_edit_proposal(
        proposal_id=f"proposal-edit-{object_type}",
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


def test_deletable_catalog_is_bounded_stable_and_provider_safe() -> None:
    session = _session()
    snapshot = session.snapshot()
    catalog = deletable_object_catalog(snapshot)
    identities = {
        (item.object_type, item.target_id, item.step_name)
        for item in catalog
    }

    assert ("part", "P1", None) in identities
    assert ("generated_mesh", "current", None) in identities
    assert ("named_region", "边-固定端", None) in identities
    assert ("analysis_step", "分析步-静力", None) in identities
    assert (
        "boundary_condition",
        "位移-固定端",
        "分析步-静力",
    ) in identities
    assert ("load", "载荷-拉伸", "分析步-静力") in identities
    assert (
        "result_request",
        "结果请求-应力",
        "分析步-静力",
    ) in identities
    assert all(item.target_id for item in catalog)
    assert all(item.target_id != "None" for item in catalog)

    encoded = json.dumps(
        [item.to_provider_dict() for item in catalog],
        ensure_ascii=False,
    )
    assert "source_path" not in encoded
    assert "project_path" not in encoded
    assert len(deletable_object_catalog(snapshot, limit=3)) == 3


def test_delete_proposal_is_path_free_and_requires_unique_gui_authorization() -> None:
    session = _session()
    revision = session.session_revision
    proposal = _proposal(
        session,
        "boundary_condition",
        "位移-固定端",
        "分析步-静力",
    )
    refreshes: list[str] = []
    port = SessionGeometryAuthoringPort(
        session,
        lambda: refreshes.append("refresh"),
    )
    bridge = AgentAuthoringBridge(port)
    bridge.bind_snapshot(session.snapshot())

    assert proposal.proposal_kind is ProposalKind.DESTRUCTIVE_EDIT
    assert proposal.operations[0].kind is OperationKind.DELETE_MODEL_OBJECT
    assert session.session_revision == revision
    assert "确认删除" == proposal.display_summary["confirm_label"]
    assert "path" not in json.dumps(proposal.to_dict(), ensure_ascii=False)

    bridge.register_proposal(proposal)
    with pytest.raises(AuthoringAuthorizationError):
        bridge.accept_proposal(proposal.proposal_id)
    receipt = bridge.accept_from_gui_control(proposal.proposal_id)

    assert receipt.state is ProposalState.SUCCEEDED
    assert refreshes == ["refresh"]
    assert session.snapshot().steps[0].boundaries == ()
    with pytest.raises(AuthoringAuthorizationError):
        bridge.accept_from_gui_control(proposal.proposal_id)


@pytest.mark.parametrize(
    ("object_type", "target_id", "step_name"),
    [
        ("load", "载荷-拉伸", "分析步-静力"),
        ("result_request", "结果请求-应力", "分析步-静力"),
        ("analysis_step", "分析步-静力", None),
    ],
)
def test_step_and_named_children_are_deleted_atomically(
    object_type: str,
    target_id: str,
    step_name: str | None,
) -> None:
    session = _session()
    proposal = _proposal(session, object_type, target_id, step_name)
    delta = apply_delete_operation(
        session,
        proposal.operations[0],
        base_session_revision=proposal.base_session_revision,
    )
    snapshot = session.snapshot()

    assert delta.accepted
    if object_type == "analysis_step":
        assert snapshot.steps == ()
    elif object_type == "load":
        assert snapshot.steps[0].edge_loads == ()
        assert snapshot.steps[0].boundaries
    else:
        assert [
            item.name
            for item in snapshot.steps[0].outputs
            if item.name is not None
        ] == ["结果请求-位移反力"]


def test_scope_mesh_and_part_deletions_reuse_session_cascades() -> None:
    scope_session = _session()
    scope_proposal = _proposal(
        scope_session,
        "named_region",
        "边-固定端",
    )
    apply_delete_operation(
        scope_session,
        scope_proposal.operations[0],
        base_session_revision=scope_proposal.base_session_revision,
    )
    scoped = scope_session.snapshot()
    assert "边-固定端" not in scoped.named_regions
    assert scoped.steps[0].boundaries == ()
    assert scoped.steps[0].edge_loads

    assignment_proposal = _proposal(
        scope_session,
        "named_region",
        "域-板体",
    )
    apply_delete_operation(
        scope_session,
        assignment_proposal.operations[0],
        base_session_revision=assignment_proposal.base_session_revision,
    )
    assert scope_session.snapshot().assignments == ()

    mesh_session = _session()
    mesh_proposal = _proposal(mesh_session, "generated_mesh", "current")
    apply_delete_operation(
        mesh_session,
        mesh_proposal.operations[0],
        base_session_revision=mesh_proposal.base_session_revision,
    )
    mesh_snapshot = mesh_session.snapshot()
    assert mesh_snapshot.artifact is None
    assert mesh_snapshot.named_regions == {}
    assert mesh_snapshot.steps == ()

    part_session = _session()
    part_proposal = _proposal(part_session, "part", "P1")
    apply_delete_operation(
        part_session,
        part_proposal.operations[0],
        base_session_revision=part_proposal.base_session_revision,
    )
    assert part_session.snapshot().parts == ()


def test_controller_publishes_delete_tools_and_waits_for_gui_terminal() -> None:
    session = _session()
    holders: dict[str, object] = {}

    def refresh() -> None:
        bridge = holders["bridge"]
        controller = holders["controller"]
        assert isinstance(bridge, AgentAuthoringBridge)
        stale = bridge.bind_snapshot(session.snapshot())
        controller.observe_binding(  # type: ignore[union-attr]
            bridge.context,
            proposal_staled=bool(stale),
        )

    port = SessionGeometryAuthoringPort(session, refresh)
    bridge = AgentAuthoringBridge(port)
    bridge.bind_snapshot(session.snapshot())
    controller = create_session_authoring_workflow_controller(
        session,
        bridge,
        AgentResultQueryBridge(SessionResultQueryPort(session)),
    )
    holders.update(bridge=bridge, controller=controller)
    names = {item.name for item in controller.definitions}

    assert "read_deletable_objects" in names
    assert "prepare_delete_proposal" in names
    assert not any("accept" in name or "confirm" in name for name in names)

    context = ToolExecutionContext("agent-delete", 0, "delete-boundary")
    catalog = controller.dispatch("read_deletable_objects", {}, context)
    result = controller.dispatch(
        "prepare_delete_proposal",
        {
            "object_type": "boundary_condition",
            "target_id": "位移-固定端",
            "step_name": "分析步-静力",
        },
        context,
    )

    assert catalog.ok and result.ok
    assert controller.stage is AuthoringWorkflowStage.DESTRUCTIVE_EDIT_PENDING
    assert {
        item.name for item in controller.definitions
    } == {"read_authoring_context"}

    receipt = bridge.accept_from_gui_control(str(result.data["proposal_id"]))
    controller.record_proposal_state(
        "destructive_edit",
        receipt.state,
        receipt.message,
    )

    assert receipt.state is ProposalState.SUCCEEDED
    assert controller.stage is AuthoringWorkflowStage.ANALYSIS_DEFINITIONS_READY
    assert session.snapshot().steps[0].boundaries == ()


def test_editable_catalog_exposes_current_values_and_opaque_scope_references() -> None:
    catalog = editable_object_catalog(_session().snapshot())
    scope = next(
        item
        for item in catalog
        if item.object_type == "named_region"
        and item.target_id == "边-固定端"
    )
    boundary = next(
        item
        for item in catalog
        if item.object_type == "boundary_condition"
    )
    load = next(
        item for item in catalog if item.object_type == "load"
    )

    assert scope.details["entity_kind"] == "edge"
    assert all(
        str(key).startswith("scope-ref-")
        for key in scope.details["reference_keys"]
    )
    assert boundary.details["value"] == 0.0
    assert boundary.details["target_scope"] == "边-固定端"
    assert load.details["load_kind"] == "edge"
    assert load.details["vector"] == [10.0, 0.0]
    encoded = json.dumps(
        [item.to_provider_dict() for item in catalog],
        ensure_ascii=False,
    )
    assert "source_path" not in encoded
    assert "node_ids" not in encoded


def test_scope_edit_renames_dependencies_and_replaces_membership() -> None:
    session = _session()
    rename = _edit_proposal(
        session,
        "named_region",
        "边-固定端",
        {"new_name": "边-约束端"},
    )
    apply_edit_operation(
        session,
        rename.operations[0],
        base_session_revision=rename.base_session_revision,
    )
    renamed = session.snapshot()

    assert "边-固定端" not in renamed.named_regions
    assert renamed.steps[0].boundaries[0].target == "边-约束端"

    catalog = editable_object_catalog(renamed)
    loading = next(
        item
        for item in catalog
        if item.object_type == "named_region"
        and item.target_id == "边-加载端"
    )
    membership = _edit_proposal(
        session,
        "named_region",
        "边-约束端",
        {"reference_keys": loading.details["reference_keys"]},
    )
    apply_edit_operation(
        session,
        membership.operations[0],
        base_session_revision=membership.base_session_revision,
    )

    assert (
        session.snapshot().named_regions["边-约束端"].references
        == session.snapshot().named_regions["边-加载端"].references
    )

    domain = _edit_proposal(
        session,
        "named_region",
        "域-板体",
        {"new_name": "域-结构体"},
    )
    apply_edit_operation(
        session,
        domain.operations[0],
        base_session_revision=domain.base_session_revision,
    )
    assert session.snapshot().assignments[0].region_name == "域-结构体"


def test_boundary_and_load_edits_are_atomic_and_gui_authorized() -> None:
    session = _session()
    boundary = _edit_proposal(
        session,
        "boundary_condition",
        "位移-固定端",
        {
            "target_scope": "边-加载端",
                "first_component": 2,
                "last_component": 2,
                "value": 1.5,
                "unit": "mm",
                "distribution": "uniform",
                "confirmed": True,
        },
        "分析步-静力",
    )
    revision = session.session_revision
    refreshes: list[str] = []
    port = SessionGeometryAuthoringPort(
        session,
        lambda: refreshes.append("refresh"),
    )
    bridge = AgentAuthoringBridge(port)
    bridge.bind_snapshot(session.snapshot())
    bridge.register_proposal(boundary)

    assert session.session_revision == revision
    with pytest.raises(AuthoringAuthorizationError):
        bridge.accept_proposal(boundary.proposal_id)
    receipt = bridge.accept_from_gui_control(boundary.proposal_id)
    edited_boundary = session.snapshot().steps[0].boundaries[0]

    assert receipt.state is ProposalState.SUCCEEDED
    assert refreshes == ["refresh"]
    assert edited_boundary.target == "边-加载端"
    assert edited_boundary.target_kind == "edge"
    assert edited_boundary.first_component == 2
    assert edited_boundary.last_component == 2
    assert edited_boundary.value == 1.5

    load = _edit_proposal(
        session,
        "load",
        "载荷-拉伸",
        {
            "target_scope": "边-固定端",
            "vector": [20.0, 5.0],
            "entity_type": "edge",
            "load_type": "edge_traction",
            "direction": "global_xy",
            "unit": "N/mm",
            "distribution": "uniform",
            "confirmed": True,
        },
        "分析步-静力",
    )
    apply_edit_operation(
        session,
        load.operations[0],
        base_session_revision=load.base_session_revision,
    )
    edited_load = session.snapshot().steps[0].edge_loads[0]

    assert edited_load.edge == "边-固定端"
    assert edited_load.vector == (20.0, 5.0)
    assert edited_load.load_type == "traction"


def test_invalid_edit_is_rejected_without_changing_the_session() -> None:
    session = _session()
    revision = session.session_revision

    with pytest.raises(ValueError, match="unsupported fields"):
        _edit_proposal(
            session,
            "boundary_condition",
            "位移-固定端",
            {"vector": [1.0, 2.0]},
            "分析步-静力",
        )
    with pytest.raises(ValueError, match="unavailable identity"):
        _edit_proposal(
            session,
            "named_region",
            "边-固定端",
            {"reference_keys": ["scope-ref-" + "0" * 64]},
        )

    assert session.session_revision == revision


def test_controller_applies_supported_edit_directly_and_refreshes_binding() -> None:
    session = _session()
    holders: dict[str, object] = {}

    def refresh() -> None:
        bridge = holders["bridge"]
        controller = holders["controller"]
        assert isinstance(bridge, AgentAuthoringBridge)
        stale = bridge.bind_snapshot(session.snapshot())
        controller.observe_binding(  # type: ignore[union-attr]
            bridge.context,
            proposal_staled=bool(stale),
        )

    port = SessionGeometryAuthoringPort(
        session,
        refresh,
        apply_definition_delta=lambda _delta: refresh(),
    )
    bridge = AgentAuthoringBridge(port)
    bridge.bind_snapshot(session.snapshot())
    controller = create_session_authoring_workflow_controller(
        session,
        bridge,
        AgentResultQueryBridge(SessionResultQueryPort(session)),
    )
    holders.update(bridge=bridge, controller=controller)
    names = {item.name for item in controller.definitions}

    assert "read_editable_model_objects" in names
    assert "edit_model_object" in names
    assert "prepare_edit_model_object_proposal" not in names
    context = ToolExecutionContext("agent-edit", 0, "edit-load")
    catalog = controller.dispatch(
        "read_editable_model_objects",
        {},
        context,
    )
    result = controller.dispatch(
        "edit_model_object",
        {
            "object_type": "load",
            "target_id": "载荷-拉伸",
            "step_name": "分析步-静力",
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
        context,
    )

    assert catalog.ok and result.ok
    assert result.data["gui_synchronized"] is True
    assert "proposal_id" not in result.data
    assert controller.stage is AuthoringWorkflowStage.DEFINITIONS_READY
    assert session.snapshot().steps[0].edge_loads[0].vector == (25.0, 0.0)
