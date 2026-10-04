from __future__ import annotations

import pytest

from fem.application import (
    RevisionConflictError,
    RegionAssignment,
    ScopedDefinitionBatch,
)
from fem.io.project import dumps_project, loads_project
from fem_agent.authoring import ModelPatch
from fem_agent.definition_authoring import (
    scoped_definition_batch_from_operations,
)

from tests.helpers.agent_definition_fixtures import build_material_definition_patch
from tests.helpers.agent_session_fixtures import make_meshed_eccentric_plate_session


def test_patch_decodes_to_one_atomic_scoped_definition_batch() -> None:
    session = make_meshed_eccentric_plate_session()
    before = session.snapshot()
    patch = build_material_definition_patch(session)
    assert type(patch) is ModelPatch
    batch = scoped_definition_batch_from_operations(
        patch.operations,
        base_session_revision=patch.base_session_revision,
    )

    delta = session.apply_scoped_definition_batch(batch)
    after = session.snapshot()

    assert delta.session_revision == before.session_revision + 1
    assert after.session_revision == before.session_revision + 1
    assert not after.named_regions
    assert [material.name for material in after.materials] == ["Material-结构钢"]
    assert not after.sections
    assert not after.assignments


def test_atomic_failure_and_stale_batch_leave_state_unchanged() -> None:
    session = make_meshed_eccentric_plate_session()
    patch = build_material_definition_patch(session)
    snapshot = session.snapshot()
    batch = scoped_definition_batch_from_operations(
        patch.operations,
        base_session_revision=patch.base_session_revision,
    )
    invalid = ScopedDefinitionBatch(
        batch.base_session_revision,
        batch.regions,
        batch.materials,
        batch.sections,
        (RegionAssignment("Section-不存在", "Domain-Plate"),),
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


def test_current_schema_round_trip_preserves_scopes_and_definitions() -> None:
    session = make_meshed_eccentric_plate_session()
    patch = build_material_definition_patch(session)
    before = session.snapshot()
    session.apply_scoped_definition_batch(
        scoped_definition_batch_from_operations(
            patch.operations,
            base_session_revision=before.session_revision,
        )
    )

    loaded = loads_project(
        dumps_project(session.prepare_project_save())
    ).snapshot

    assert not loaded.named_regions
    assert [material.name for material in loaded.materials] == ["Material-结构钢"]
    assert not loaded.sections
    assert not loaded.assignments
