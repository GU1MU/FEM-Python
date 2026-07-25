from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from fem.application import (
    ModelSession,
    NativePart,
    ProjectSnapshot,
    RegionAssignment,
    RevisionConflictError,
    SectionDefinition,
)
from fem.core.model import AnalysisStep, MaterialDefinition
from tests.helpers.preflight_builders import passing_preflight_report


def _model(*step_names: str) -> SimpleNamespace:
    return SimpleNamespace(
        materials={},
        sections=[],
        steps=[AnalysisStep(name) for name in step_names],
        element_sets={},
        metadata={},
        mesh=SimpleNamespace(nodes=[], elements=[]),
    )


def _native_session(*step_names: str) -> ModelSession:
    session = ModelSession()
    session.new_native_project()
    session.replace_geometry((NativePart(),), {"kind": "box"})
    session.replace_model_definitions(
        (), (), (), tuple(AnalysisStep(name) for name in step_names)
    )
    task = session.prepare_mesh_generation()
    assert session.accept_generated_model(
        task.token, _model(*step_names)
    ).accepted
    return session


def test_domain_revisions_advance_only_for_their_semantics() -> None:
    session = ModelSession()
    first_id = session.session_id
    opened = session.new_native_project()
    opened_snapshot = session.snapshot()

    assert opened.session_revision == 1
    assert opened_snapshot.session_id != first_id
    assert not opened_snapshot.dirty

    geometry = session.replace_geometry((NativePart(),), {"kind": "box"})
    after_geometry = session.snapshot()
    assert geometry.session_revision == opened.session_revision + 1
    assert after_geometry.project_revision == opened_snapshot.project_revision + 1
    assert (
        after_geometry.mesh_input_revision
        == opened_snapshot.mesh_input_revision + 1
    )
    assert after_geometry.model_revision == opened_snapshot.model_revision + 1
    assert after_geometry.dirty

    before_validation = session.snapshot()
    session.replace_model_definitions((), (), (), (AnalysisStep("Step-A"),))
    mesh_task = session.prepare_mesh_generation()
    session.accept_generated_model(mesh_task.token, _model("Step-A"))
    validation_task = session.prepare_validation("Step-A")
    validation_delta = session.accept_validation(
        validation_task.token,
        passing_preflight_report(validation_task.token),
    )
    after_validation = session.snapshot()

    assert validation_delta.session_revision == after_validation.session_revision
    assert after_validation.project_revision > before_validation.project_revision
    validation_project_revision = after_validation.project_revision
    validation_mesh_revision = after_validation.mesh_input_revision
    validation_model_revision = after_validation.model_revision

    run_task = session.prepare_solve("Step-A", "Job-1")
    session.begin_run(run_task.token)
    session.accept_run_result(run_task.token, {"U": [1.0]})
    before_select = session.snapshot()
    session.select_result(run_task.run_id)
    after_select = session.snapshot()

    assert after_select.session_revision == before_select.session_revision + 1
    assert after_select.project_revision == validation_project_revision
    assert after_select.mesh_input_revision == validation_mesh_revision
    assert after_select.model_revision == validation_model_revision


def test_compare_and_swap_conflict_has_no_side_effects() -> None:
    session = _native_session("Step-A")
    before = session.snapshot()

    with pytest.raises(RevisionConflictError):
        session.replace_mesh_settings(
            {"size": 0.25},
            expected_session_revision=before.session_revision - 1,
        )

    after = session.snapshot()
    assert after.session_id == before.session_id
    assert after.session_revision == before.session_revision
    assert after.project_revision == before.project_revision
    assert after.mesh_input_revision == before.mesh_input_revision
    assert after.model_revision == before.model_revision
    assert after.mesh_settings == before.mesh_settings
    assert after.artifact.artifact_id == before.artifact.artifact_id


def test_new_close_and_snapshot_replace_change_session_identity() -> None:
    session = ModelSession()
    initial_id = session.session_id
    session.new_native_project()
    native_id = session.session_id

    session.close()
    closed_id = session.session_id
    session.new_native_project()
    replacement = session.snapshot()

    assert len({initial_id, native_id, closed_id, replacement.session_id}) == 4
    assert replacement.session_revision == 3
    assert not replacement.dirty


def test_successful_save_only_marks_its_project_revision_clean() -> None:
    session = ModelSession()
    session.new_native_project()
    session.replace_geometry((NativePart(),), {"kind": "box"})
    prepared = session.prepare_project_save()
    project_revision = session.project_revision
    before_session_revision = session.session_revision

    delta = session.accept_project_saved(
        prepared.token, Path("model.femproj")
    )

    snapshot = session.snapshot()
    assert delta.session_revision == before_session_revision + 1
    assert snapshot.project_revision == project_revision
    assert snapshot.saved_project_revision == project_revision
    assert snapshot.project_path == Path("model.femproj")
    assert not snapshot.dirty


def test_save_completion_is_stale_if_inputs_changed() -> None:
    session = ModelSession()
    session.new_native_project()
    session.replace_geometry((NativePart(),), {"kind": "box"})
    prepared = session.prepare_project_save()
    session.replace_mesh_settings({"size": 0.5})
    before = session.snapshot()

    delta = session.accept_project_saved(
        prepared.token, Path("old-inputs.femproj")
    )

    after = session.snapshot()
    assert not delta.accepted
    assert after.session_revision == before.session_revision
    assert after.saved_project_revision == before.saved_project_revision
    assert after.project_path == before.project_path
    assert after.dirty


def test_invalid_snapshot_install_is_atomic() -> None:
    session = _native_session("Step-A")
    before = session.snapshot()
    invalid = ProjectSnapshot(
        source_kind="native",
        source_path=Path("invalid.femproj"),
        parts=(NativePart(),),
        geometry_recipe={"kind": "box"},
        material_definitions=(MaterialDefinition("Steel", {}),),
        section_definitions=(SectionDefinition("Solid", "Missing"),),
        region_assignments=(RegionAssignment("Solid", "All"),),
        analysis_definitions=(AnalysisStep("Step-A"),),
        model=_model("Step-A"),
    )

    with pytest.raises(ValueError, match="missing material"):
        session.replace_from_snapshot(
            invalid,
            expected_session_revision=before.session_revision,
        )

    after = session.snapshot()
    assert after.session_id == before.session_id
    assert after.session_revision == before.session_revision
    assert after.project_revision == before.project_revision
    assert after.artifact.artifact_id == before.artifact.artifact_id
    assert after.project_path == before.project_path
