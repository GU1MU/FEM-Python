from __future__ import annotations

from dataclasses import replace

import pytest

from fem.application import (
    ModelSession,
    RevisionConflictError,
    RegionAssignment,
    AnalysisRun,
    RunStatus,
    ScopedDefinitionBatch,
    UnitContext,
)
from fem.application.native_scope_materialization import (
    NATIVE_PART_OWNERSHIP_KEY,
    NATIVE_SCOPE_CATALOG_KEY,
)
from fem.core.mesh import Element2D, Mesh2D, Node2D
from fem.core.model import FEMModel
from fem.geometry import (
    PlateWithHoleGeometry,
    SketchCircle,
    SketchRectangle,
)
from fem.mesh.settings import MeshSettings
from fem.io.project import dumps_project, loads_project
from fem.selection import edges as mesh_edges
from fem_agent.authoring import AgentProposal, ModelPatch, ProposalKind
from fem_agent.definition_authoring import (
    ScopeSelectionError,
    build_eccentric_plate_scopes,
    create_scope_definition_change,
    scoped_definition_batch_from_operations,
)
from fem_agent.geometry_authoring import planar_sketch_geometry
from fem_agent.tools.registry import AgentToolRegistry
from fem_gui.agent_authoring import authoring_context_from_snapshot


def _recipe() -> PlateWithHoleGeometry:
    return PlateWithHoleGeometry(
        "实体-偏心孔板",
        10.0,
        6.0,
        6.5,
        2.0,
        1.0,
    )


def _plate_model() -> FEMModel:
    mesh = Mesh2D(
        nodes=[
            Node2D(1, 0.0, 0.0),
            Node2D(2, 10.0, 0.0),
            Node2D(3, 10.0, 6.0),
            Node2D(4, 0.0, 6.0),
            Node2D(5, 7.5, 2.0),
            Node2D(6, 6.5, 3.0),
            Node2D(7, 5.5, 2.0),
            Node2D(8, 6.5, 1.0),
        ],
        elements=[
            Element2D(1, [1, 2, 8], "Tri3"),
            Element2D(2, [2, 5, 8], "Tri3"),
            Element2D(3, [2, 3, 5], "Tri3"),
            Element2D(4, [3, 6, 5], "Tri3"),
            Element2D(5, [3, 4, 6], "Tri3"),
            Element2D(6, [4, 7, 6], "Tri3"),
            Element2D(7, [4, 1, 7], "Tri3"),
            Element2D(8, [1, 8, 7], "Tri3"),
        ],
    )
    boundary = tuple(mesh_edges.boundary(mesh))
    outer_nodes = {1, 2, 3, 4}
    hole_nodes = {5, 6, 7, 8}

    def rows(node_ids: set[int]):
        return tuple(
            (
                element_id,
                local_index,
                tuple(edge_node_ids),
            )
            for element_id, local_index, edge_node_ids in boundary
            if set(edge_node_ids).issubset(node_ids)
        )

    catalog = {
        "edge:P1/outer-loop": {
            "kind": "edge",
            "node_ids": tuple(sorted(outer_nodes)),
            "element_ids": (),
            "edges": rows(outer_nodes),
            "faces": (),
        },
        "edge:P1/hole-loop": {
            "kind": "edge",
            "node_ids": tuple(sorted(hole_nodes)),
            "element_ids": (),
            "edges": rows(hole_nodes),
            "faces": (),
        },
        "face:P1/domain": {
            "kind": "face",
            "node_ids": tuple(range(1, 9)),
            "element_ids": tuple(range(1, 9)),
            "edges": (),
            "faces": (),
        },
    }
    return FEMModel(
        mesh,
        name="模型-偏心孔板",
        metadata={
            NATIVE_SCOPE_CATALOG_KEY: catalog,
            NATIVE_PART_OWNERSHIP_KEY: {
                "P1": {
                    "node_ids": tuple(range(1, 9)),
                    "element_ids": tuple(range(1, 9)),
                }
            },
        },
    )


def _session() -> ModelSession:
    session = ModelSession()
    session.create_native_project_with_first_part(
        "模型-偏心孔板",
        UnitContext("mm", "N", "MPa"),
        _recipe(),
        part_name="部件-偏心孔板",
    )
    task = session.prepare_agent_mesh_generation(
        "P1",
        MeshSettings(1.0),
        "a" * 64,
        expected_session_revision=session.session_revision,
    )
    assert session.accept_agent_generated_model(
        task.token,
        _plate_model(),
    ).accepted
    return session


def _change(session: ModelSession):
    snapshot = session.snapshot()
    return create_scope_definition_change(
        patch_id="patch-a4",
        proposal_id="proposal-a4",
        agent_session_id="agent-a4",
        turn_id="turn-a4",
        source_tool_call_ids=("call-a4",),
        context=authoring_context_from_snapshot(snapshot),
        snapshot=snapshot,
        draft_revision=4,
        material_function="结构钢",
        material_properties={"E": 210000.0, "nu": 0.3},
        section_function="平面应力",
        plane_type="stress",
        thickness=2.0,
    )


def test_a4_plate_scopes_have_four_semantic_aliases_and_exact_evidence() -> None:
    scopes = build_eccentric_plate_scopes(_session().snapshot())

    assert {region.name for region in scopes.regions} == {
        "边-固定端",
        "边-加载端",
        "边-孔边",
        "域-板体",
    }
    assert {region.entity_kind for region in scopes.regions} == {
        "edge",
        "element",
    }
    assert all(
        item.matched_count == item.expected_count > 0
        for item in scopes.evidence
    )
    assert all(
        reference.part_id == "P1"
        for region in scopes.regions
        for reference in region.references
    )


def test_a4_plate_scopes_accept_general_strict_sketch_recipe() -> None:
    draft = planar_sketch_geometry(
        "草图-孔板",
        contours=(
            SketchRectangle("material", 0.0, 0.0, 10.0, 6.0),
            SketchCircle("cut", 6.5, 2.0, 1.0),
        ),
    )
    circle = next(
        curve
        for curve in draft.recipe.curves
        if isinstance(curve, SketchCircle)
    )
    session = ModelSession()
    session.create_native_project_with_first_part(
        "模型-通用草图",
        UnitContext("mm", "N", "MPa"),
        draft.recipe,
        part_name="部件-孔板",
    )
    task = session.prepare_agent_mesh_generation(
        "P1",
        MeshSettings(1.0),
        "b" * 64,
        expected_session_revision=session.session_revision,
    )
    model = _plate_model()
    catalog = model.metadata[NATIVE_SCOPE_CATALOG_KEY]
    catalog[f"edge:P1/{circle.id}"] = catalog.pop("edge:P1/hole-loop")
    assert session.accept_agent_generated_model(task.token, model).accepted

    scopes = build_eccentric_plate_scopes(session.snapshot())

    assert {region.name for region in scopes.regions} == {
        "边-固定端",
        "边-加载端",
        "边-孔边",
        "域-板体",
    }


def test_a4_scope_selection_fails_closed_on_abnormal_catalog_identity() -> None:
    session = _session()
    snapshot = session.snapshot()
    snapshot.artifact.model.metadata[NATIVE_SCOPE_CATALOG_KEY][
        "edge:P1/hole-loop"
    ]["edges"] = ()

    with pytest.raises(
        (ScopeSelectionError, ValueError),
        match="mesh edge|scope",
    ):
        build_eccentric_plate_scopes(snapshot)


def test_a4_patch_decodes_to_one_atomic_scoped_definition_batch() -> None:
    session = _session()
    before = session.snapshot()
    patch = _change(session)
    assert type(patch) is ModelPatch
    batch = scoped_definition_batch_from_operations(
        patch.operations,
        before,
        base_session_revision=patch.base_session_revision,
    )

    delta = session.apply_scoped_definition_batch(batch)
    after = session.snapshot()

    assert delta.session_revision == before.session_revision + 1
    assert after.session_revision == before.session_revision + 1
    assert tuple(after.named_regions) == (
        "边-固定端",
        "边-加载端",
        "边-孔边",
        "域-板体",
    )
    assert [material.name for material in after.materials] == [
        "材料-结构钢"
    ]
    assert [section.name for section in after.sections] == [
        "截面-平面应力"
    ]
    assert after.assignments[0].region_name == "域-板体"
    assert "域-板体" in after.artifact.model.element_sets


def test_a4_atomic_failure_and_stale_batch_leave_state_unchanged() -> None:
    session = _session()
    patch = _change(session)
    snapshot = session.snapshot()
    batch = scoped_definition_batch_from_operations(
        patch.operations,
        snapshot,
        base_session_revision=patch.base_session_revision,
    )
    invalid = ScopedDefinitionBatch(
        batch.base_session_revision,
        batch.regions,
        batch.materials,
        batch.sections,
        (RegionAssignment("截面-不存在", "域-板体"),),
        batch.steps,
    )

    with pytest.raises(ValueError):
        session.apply_scoped_definition_batch(invalid)
    assert session.snapshot().session_revision == snapshot.session_revision
    assert not session.snapshot().named_regions

    session.replace_model_definitions((), (), (), ())
    with pytest.raises(RevisionConflictError):
        session.apply_scoped_definition_batch(batch)
    assert not session.snapshot().named_regions


def test_a4_existing_result_turns_change_into_confirmation_proposal() -> None:
    session = _session()
    snapshot = session.snapshot()
    artifact = snapshot.artifact
    run = AnalysisRun(
        "run-a4",
        "作业-旧结果",
        "分析步-旧",
        artifact.artifact_id,
        artifact.model_revision,
        status=RunStatus.SUCCEEDED,
        result_id="result-a4",
    )
    result_snapshot = replace(snapshot, runs=(run,))

    change = create_scope_definition_change(
        patch_id="patch-with-result",
        proposal_id="proposal-with-result",
        agent_session_id="agent-a4",
        turn_id="turn-a4",
        source_tool_call_ids=("call-a4",),
        context=authoring_context_from_snapshot(result_snapshot),
        snapshot=result_snapshot,
        draft_revision=4,
        material_function="结构钢",
        material_properties={"E": 210000.0, "nu": 0.3},
        section_function="平面应力",
        plane_type="stress",
        thickness=2.0,
    )

    assert type(change) is AgentProposal
    assert change.proposal_kind is ProposalKind.DESTRUCTIVE_EDIT
    assert change.invalidation_impact["results"] is True
    assert session.snapshot().session_revision == snapshot.session_revision


def test_a4_current_schema_round_trip_preserves_scopes_and_definitions() -> None:
    session = _session()
    patch = _change(session)
    before = session.snapshot()
    session.apply_scoped_definition_batch(
        scoped_definition_batch_from_operations(
            patch.operations,
            before,
            base_session_revision=before.session_revision,
        )
    )

    loaded = loads_project(
        dumps_project(session.prepare_project_save())
    ).snapshot

    assert {region.name for region in loaded.named_regions} == {
        "边-固定端",
        "边-加载端",
        "边-孔边",
        "域-板体",
    }
    assert [material.name for material in loaded.materials] == [
        "材料-结构钢"
    ]
    assert [section.name for section in loaded.sections] == [
        "截面-平面应力"
    ]
    assert loaded.assignments == (
        loaded.region_assignments[0],
    )
    assert loaded.assignments[0].region_name == "域-板体"


def test_a4_provider_catalog_exposes_no_confirmation_or_undo_tool(
    tmp_path,
) -> None:
    names = {
        definition.name
        for definition in AgentToolRegistry(
            tmp_path / "workspace"
        ).definitions
    }

    assert "accept_proposal" not in names
    assert "confirm_definition_change" not in names
    assert "undo_agent_patch" not in names
