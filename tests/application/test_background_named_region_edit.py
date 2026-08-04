from __future__ import annotations

from copy import deepcopy

import pytest

from fem.application import (
    DefinitionRejected,
    MeshEntityRef,
    ModelSession,
    NamedRegion,
    NamedRegionEditBatch,
    RegionAssignment,
    SectionDefinition,
    TokenStatus,
    compile_named_region_edit,
)
from fem.core.mesh import Element2D, Mesh2D, Mesh3D, Node2D, Node3D
from fem.core.model import FEMModel, MaterialDefinition


def _imported_session(tmp_path, *, node_count: int = 4) -> ModelSession:
    model = FEMModel(
        Mesh3D(
            [Node3D(index, float(index), 0.0, 0.0) for index in range(1, node_count + 1)],
            [],
        ),
        name="scope-background",
    )
    session = ModelSession()
    task = session.prepare_import(tmp_path / "scope.inp")
    session.accept_imported_model(task.token, model)
    return session


def _batch(session: ModelSession, name: str, *node_ids: int) -> NamedRegionEditBatch:
    return NamedRegionEditBatch(
        session.session_revision,
        (
            NamedRegion(
                name,
                tuple(MeshEntityRef.node(node_id) for node_id in node_ids),
            ),
        ),
    )


def test_background_and_synchronous_scope_commits_are_equivalent(tmp_path) -> None:
    synchronous = _imported_session(tmp_path / "sync")
    background = _imported_session(tmp_path / "background")
    sync_before = synchronous.projection_snapshot()
    background_before = background.projection_snapshot()

    sync_delta = synchronous.apply_named_region_edit(
        _batch(synchronous, "Picked", 1, 2, 3)
    )
    task = background.prepare_named_region_edit(
        _batch(background, "Picked", 1, 2, 3)
    )
    prepared = compile_named_region_edit(task)
    background_delta = background.accept_named_region_edit(task, prepared)

    sync_after = synchronous.projection_snapshot()
    background_after = background.projection_snapshot()
    assert sync_delta.changed == background_delta.changed
    assert sync_delta.invalidated == background_delta.invalidated
    assert sync_after.named_regions == background_after.named_regions
    assert sync_after.model.node_sets == background_after.model.node_sets
    assert sync_after.assignments == background_after.assignments
    assert sync_after.steps == background_after.steps
    assert sync_after.session_revision == sync_before.session_revision + 1
    assert background_after.session_revision == background_before.session_revision + 1
    assert background_after.model.mesh is background_before.model.mesh


def test_only_latest_scope_request_can_commit_when_a_finishes_after_b(tmp_path) -> None:
    session = _imported_session(tmp_path)
    task_a = session.prepare_named_region_edit(_batch(session, "A", 1))
    task_b = session.prepare_named_region_edit(_batch(session, "B", 2))
    prepared_a = compile_named_region_edit(task_a)
    prepared_b = compile_named_region_edit(task_b)

    accepted_b = session.accept_named_region_edit(task_b, prepared_b)
    stale_a = session.accept_named_region_edit(task_a, prepared_a)

    assert accepted_b.accepted
    assert not stale_a.accepted
    assert stale_a.token_status in {
        TokenStatus.STALE_REVISION,
        TokenStatus.ALREADY_COMPLETED,
    }
    assert tuple(session.projection_snapshot().named_regions) == ("B",)


@pytest.mark.parametrize("invalidation", ("document", "mesh", "undo"))
def test_scope_result_is_stale_after_document_mesh_or_undo_change(
    tmp_path,
    invalidation: str,
) -> None:
    session = _imported_session(tmp_path)
    task = session.prepare_named_region_edit(_batch(session, "Late", 1))
    prepared = compile_named_region_edit(task)
    if invalidation == "document":
        session.new_native_project()
    elif invalidation == "mesh":
        # Installing another imported mesh changes both document and mesh
        # identity, matching the remesh CAS boundary for imported fixtures.
        replacement = FEMModel(Mesh3D([Node3D(1, 0.0, 0.0, 0.0)], []))
        import_task = session.prepare_import(tmp_path / "replacement.inp")
        session.accept_imported_model(import_task.token, replacement)
    else:
        # A scope command followed by its inverse is the named-region undo
        # boundary: even an identical visible post-state has a newer revision.
        session.apply_named_region_edit(_batch(session, "Temporary", 2))

    stale = session.accept_named_region_edit(task, prepared)

    assert not stale.accepted
    assert "Late" not in session.projection_snapshot().named_regions


def test_background_compile_failure_is_atomic(tmp_path) -> None:
    session = _imported_session(tmp_path)
    before = session.snapshot()
    invalid = _batch(session, "Missing", 999)
    task = session.prepare_named_region_edit(invalid)

    with pytest.raises(ValueError, match="node reference is absent"):
        compile_named_region_edit(task)
    terminated = session.terminate_named_region_edit(task, "unknown node")
    after = session.snapshot()

    assert terminated.accepted
    assert after.session_revision == before.session_revision
    assert after.project_revision == before.project_revision
    assert after.model_revision == before.model_revision
    assert after.named_regions == before.named_regions
    assert after.model.node_sets == before.model.node_sets


def test_background_and_sync_rejections_have_the_same_diagnostic(tmp_path) -> None:
    synchronous = _imported_session(tmp_path / "sync")
    background = _imported_session(tmp_path / "background")
    sync_batch = _batch(synchronous, "Missing", 999)
    background_batch = _batch(background, "Missing", 999)

    with pytest.raises(ValueError) as sync_error:
        synchronous.apply_named_region_edit(sync_batch)
    task = background.prepare_named_region_edit(background_batch)
    with pytest.raises(ValueError) as background_error:
        compile_named_region_edit(task)

    assert str(background_error.value) == str(sync_error.value)


def test_sync_and_background_reject_incompatible_assigned_scope_edit(
    tmp_path,
) -> None:
    def assigned_session(path) -> ModelSession:
        model = FEMModel(
            Mesh2D(
                [
                    Node2D(1, 0.0, 0.0),
                    Node2D(2, 1.0, 0.0),
                    Node2D(3, 0.0, 1.0),
                    Node2D(4, 2.0, 0.0),
                ],
                [
                    Element2D(1, [1, 2, 3], "Tri3"),
                    Element2D(2, [2, 4], "Truss2"),
                ],
            )
        )
        session = ModelSession()
        imported = session.prepare_import(path / "assigned.inp")
        session.accept_imported_model(imported.token, model)
        session.apply_named_region_edit(
            NamedRegionEditBatch(
                session.session_revision,
                (NamedRegion("Assigned", (MeshEntityRef.element(1),)),),
            )
        )
        session.replace_model_definitions(
            (MaterialDefinition("Steel", {"E": 210_000.0, "nu": 0.3}),),
            (
                SectionDefinition(
                    "Solid",
                    "Steel",
                    "solid",
                    {"plane_type": "stress", "thickness": 1.0},
                ),
            ),
            (RegionAssignment("Solid", "Assigned"),),
            (),
        )
        return session

    synchronous = assigned_session(tmp_path / "sync")
    background = assigned_session(tmp_path / "background")
    sync_batch = NamedRegionEditBatch(
        synchronous.session_revision,
        (NamedRegion("Assigned", (MeshEntityRef.element(2),)),),
    )
    background_batch = NamedRegionEditBatch(
        background.session_revision,
        (NamedRegion("Assigned", (MeshEntityRef.element(2),)),),
    )

    with pytest.raises(DefinitionRejected) as sync_error:
        synchronous.apply_named_region_edit(sync_batch)
    task = background.prepare_named_region_edit(background_batch)
    with pytest.raises(DefinitionRejected) as background_error:
        compile_named_region_edit(task)

    assert str(background_error.value) == str(sync_error.value)


def test_task_snapshot_is_detached_from_later_batch_container_changes(tmp_path) -> None:
    session = _imported_session(tmp_path)
    batch = _batch(session, "Stable", 1, 2)
    task = session.prepare_named_region_edit(batch)
    copied = deepcopy(task.batch)

    assert task.token.session_id == session.session_id
    assert dict(task.token.dependency_revisions) == {
        "mesh_input_revision": session.mesh_input_revision,
        "model_revision": session.model_revision,
        "session_revision": session.session_revision,
    }
    assert task.request_sequence == 1
    assert copied == batch
