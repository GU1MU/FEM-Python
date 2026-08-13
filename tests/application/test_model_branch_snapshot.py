from __future__ import annotations

from dataclasses import replace

import pytest

from fem.application import ModelSession, RevisionConflictError, UnitContext
from fem.geometry import RectangleGeometry
from fem.mesh.settings import MeshSettings


def _native_session() -> ModelSession:
    session = ModelSession()
    session.create_native_project_with_first_part(
        "Source",
        UnitContext("mm", "N", "MPa"),
        RectangleGeometry("Plate", 10.0, 4.0),
    )
    session.replace_part_geometry(
        "P1",
        RectangleGeometry("Plate", 10.0, 4.0),
        mesh_settings=MeshSettings(1.0, cell_shape="quadrilateral"),
    )
    session.add_native_part(RectangleGeometry("Spare", 2.0, 1.0))
    session.delete_native_part("P2")
    return session


def test_branch_snapshot_reuses_save_inputs_without_path_model_or_token(
    tmp_path,
) -> None:
    session = _native_session()
    prepared = session.prepare_project_save()
    assert session.accept_project_saved(
        prepared.token, tmp_path / "source.fem.json"
    ).accepted
    tokens_before = dict(session._issued_tokens)

    branch = session.project_snapshot_for_branch(
        "Source-Iteration",
        expected_session_revision=session.session_revision,
    )

    assert session._issued_tokens == tokens_before
    saved = session.prepare_project_save().snapshot
    assert branch == replace(
        saved,
        source_path=None,
        model_name="Source-Iteration",
        model=None,
    )
    assert branch.source_path is None
    assert branch.project_path is None
    assert branch.model is None
    assert branch.parts is not saved.parts
    assert branch.parts[0] is not saved.parts[0]
    assert branch.retired_part_ids == ("P2",)


def test_branch_snapshot_rejects_stale_revision_without_issuing_token() -> None:
    session = _native_session()
    before = dict(session._issued_tokens)

    with pytest.raises(RevisionConflictError):
        session.project_snapshot_for_branch(
            expected_session_revision=session.session_revision - 1,
        )

    assert session._issued_tokens == before
