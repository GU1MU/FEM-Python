from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from fem.application import (
    ChangeKind,
    ModelSession,
    NativePart,
    ResultMaterializationTaskSnapshot,
    RunStatus,
    TokenStatus,
)
from fem.application.results import (
    FieldData,
    FieldPosition,
    FieldRequest,
    ResultFieldId,
    ResultMaterializationPatch,
    ResultVariable,
    restore_result_provider,
)
from fem.core.model import AnalysisStep, FEMModel
from fem.geometry.recipes import BoxGeometry
from tests.helpers.model_builders import make_simple_truss_mesh
from tests.helpers.preflight_builders import passing_preflight_report
from tests.helpers.result_builders import make_solve_result_bundle


def _model() -> FEMModel:
    return FEMModel(
        mesh=make_simple_truss_mesh(),
        steps=[AnalysisStep("Step-A")],
    )


def _session() -> ModelSession:
    session = ModelSession()
    session.new_native_project()
    session.replace_geometry(
        (NativePart(),),
        BoxGeometry("Box", 1.0, 1.0, 1.0),
    )
    session.replace_model_definitions(
        (),
        (),
        (),
        (AnalysisStep("Step-A"),),
    )
    mesh = session.prepare_mesh_generation()
    session.accept_generated_model(mesh.token, _model())
    validation = session.prepare_validation("Step-A")
    session.accept_validation(
        validation.token,
        passing_preflight_report(validation.token),
    )
    return session


def _succeed(
    session: ModelSession,
    run_name: str,
    *,
    marker: float,
):
    solve = session.prepare_solve("Step-A", run_name)
    session.begin_run(solve.token)
    session.accept_run_succeeded(
        solve.token,
        make_solve_result_bundle(solve, marker=marker),
    )
    return solve


def _key(
    record,
    position: FieldPosition,
    variable: ResultVariable = ResultVariable.S,
):
    provider = restore_result_provider(
        record.result,
        record.materialization,
    )
    return provider.resolve_request(
        FieldRequest(ResultFieldId(variable, position))
    )


def _materialize(task):
    provider = restore_result_provider(
        task.record.result,
        task.record.materialization,
    )
    return provider.materialize(task.field_keys)


def test_materialization_advances_only_runtime_result_generation() -> None:
    session = _session()
    solve = _succeed(session, "Job-1", marker=1.0)
    initial = session.current_result()
    key = _key(initial, FieldPosition.CENTROID)
    projection = session.prepare_result_projection(solve.run_id)
    task = session.prepare_result_materialization(
        solve.run_id,
        (key, key),
    )
    save = session.prepare_project_save()
    session.accept_project_saved(save.token, Path("saved.fem.json"))
    session.select_result(solve.run_id)

    assert type(task) is ResultMaterializationTaskSnapshot
    assert task.field_keys == (key,)
    assert task.record.materialization.generation == 0
    assert dict(task.token.dependency_revisions) == {
        "materialization_generation": 0,
        "model_revision": session.model_revision,
    }
    assert task.token.session_id == initial.materialization.source.session_id
    assert task.token.artifact_id == initial.materialization.source.artifact_id
    assert task.token.step_name == initial.materialization.source.step_name
    assert task.token.run_id == initial.materialization.source.run_id
    assert task.token.result_id == initial.materialization.source.result_id
    assert session.validate_task_token(task.token) is TokenStatus.CURRENT

    patch = _materialize(task)
    before_revision = session.session_revision
    before_project_revision = session.project_revision
    before_model_revision = session.model_revision
    before_dirty = session.dirty
    delta = session.accept_result_materialization(task.token, patch)
    current = session.current_result()

    assert delta.changed == {ChangeKind.RESULTS}
    assert delta.invalidated == frozenset()
    assert session.session_revision == before_revision + 1
    assert session.project_revision == before_project_revision
    assert session.model_revision == before_model_revision
    assert session.dirty is before_dirty
    assert current.materialization.generation == 1
    assert current.result_id == initial.result_id
    assert current.provenance == initial.provenance
    assert current.output_report == initial.output_report
    assert current.created_at == initial.created_at
    assert current.result.name == initial.result.name
    np.testing.assert_array_equal(current.result.U, initial.result.U)
    np.testing.assert_array_equal(
        current.result.reactions,
        initial.result.reactions,
    )
    assert key in {
        field_data.key for field_data in current.materialization.fields
    }
    assert {
        field_data.key for field_data in initial.materialization.fields
    }.issubset(
        {
            field_data.key
            for field_data in current.materialization.fields
        }
    )
    assert (
        session.validate_task_token(task.token)
        is TokenStatus.ALREADY_COMPLETED
    )
    assert (
        session.validate_task_token(projection.token)
        is TokenStatus.STALE_REVISION
    )
    assert not session.accept_result_projection(projection.token).accepted


def test_competing_workers_use_generation_cas_and_cache_hit_is_noop() -> None:
    session = _session()
    solve = _succeed(session, "Job-1", marker=2.0)
    key = _key(session.current_result(), FieldPosition.CENTROID)
    first = session.prepare_result_materialization(solve.run_id, (key,))
    second = session.prepare_result_materialization(solve.run_id, (key,))
    third = session.prepare_result_materialization(solve.run_id, (key,))
    first_patch = _materialize(first)
    second_patch = _materialize(second)

    session.accept_result_materialization(first.token, first_patch)
    after_first = session.session_revision
    rejected = session.accept_result_materialization(
        second.token,
        second_patch,
    )

    assert not rejected.accepted
    assert rejected.token_status is TokenStatus.STALE_REVISION
    assert session.session_revision == after_first
    assert session.current_result().materialization.generation == 1
    stale_failure = session.accept_task_failed(
        second.token,
        "late failure",
    )
    stale_cancel = session.accept_task_cancelled(third.token)
    assert not stale_failure.accepted
    assert not stale_cancel.accepted
    assert stale_failure.token_status is TokenStatus.STALE_REVISION
    assert stale_cancel.token_status is TokenStatus.STALE_REVISION
    assert session.session_revision == after_first
    assert session.current_result().materialization.generation == 1
    assert key in {
        field_data.key
        for field_data in session.current_result().materialization.fields
    }

    cache_hit = session.prepare_result_materialization(
        solve.run_id,
        (key,),
    )
    empty_patch = _materialize(cache_hit)
    assert empty_patch.fields == ()
    before_cache_hit = session.session_revision
    accepted = session.accept_result_materialization(
        cache_hit.token,
        empty_patch,
    )

    assert accepted.accepted
    assert accepted.changed == frozenset()
    assert session.session_revision == before_cache_hit
    assert session.current_result().materialization.generation == 1
    repeated = session.accept_result_materialization(
        cache_hit.token,
        empty_patch,
    )
    assert not repeated.accepted
    assert repeated.token_status is TokenStatus.ALREADY_COMPLETED


def test_materialization_accepts_only_the_exact_requested_patch() -> None:
    session = _session()
    solve = _succeed(session, "Job-1", marker=3.0)
    record = session.current_result()
    centroid = _key(record, FieldPosition.CENTROID)
    logarithmic_strain = _key(
        record,
        FieldPosition.CENTROID,
        ResultVariable.LE,
    )
    task = session.prepare_result_materialization(
        solve.run_id,
        (centroid,),
    )
    provider = restore_result_provider(
        task.record.result,
        task.record.materialization,
    )
    unrequested = provider.materialize((logarithmic_strain,))
    revision = session.session_revision

    with pytest.raises(ValueError, match="requested lazy keys"):
        session.accept_result_materialization(task.token, unrequested)
    with pytest.raises(TypeError, match="ResultMaterializationPatch"):
        session.accept_result_materialization(
            task.token,
            task.record.materialization,
        )
    valid = _materialize(task)
    field_data = valid.fields[0]
    wrong_descriptor = replace(
        field_data.descriptor,
        label_key="wrong.result.field",
    )
    invalid_field = FieldData(
        descriptor=wrong_descriptor,
        source=field_data.source,
        key=field_data.key,
        locations=field_data.locations,
        values=field_data.values,
    )
    invalid_patch = ResultMaterializationPatch(
        source=valid.source,
        fields=(invalid_field,),
    )
    with pytest.raises(ValueError, match="descriptor"):
        session.accept_result_materialization(
            task.token,
            invalid_patch,
        )

    assert session.session_revision == revision
    assert session.current_result().materialization.generation == 0
    assert session.find_run(solve.run_id).status is RunStatus.SUCCEEDED
    assert session.validate_task_token(task.token) is TokenStatus.CURRENT

    accepted = session.accept_result_materialization(
        task.token,
        valid,
    )
    assert accepted.accepted


def test_hidden_run_patch_updates_only_its_target_record() -> None:
    session = _session()
    run_a = _succeed(session, "Job-A", marker=4.0)
    record_a = session.current_result()
    key_a = _key(record_a, FieldPosition.CENTROID)
    task_a = session.prepare_result_materialization(
        run_a.run_id,
        (key_a,),
    )
    patch_a = _materialize(task_a)

    run_b = _succeed(session, "Job-B", marker=5.0)
    record_b = session.current_result()
    key_b = _key(record_b, FieldPosition.CENTROID)
    task_b = session.prepare_result_materialization(
        run_b.run_id,
        (key_b,),
    )
    patch_b = _materialize(task_b)
    revision = session.session_revision

    with pytest.raises(ValueError, match="target result"):
        session.accept_result_materialization(task_a.token, patch_b)
    assert session.session_revision == revision
    assert session.validate_task_token(task_a.token) is TokenStatus.CURRENT

    delta = session.accept_result_materialization(task_a.token, patch_a)

    assert delta.changed == {ChangeKind.RESULTS}
    assert session.snapshot().displayed_result_run_id == run_b.run_id
    assert session.current_result().provenance.run_id == run_b.run_id
    assert session.current_result().materialization.generation == 0
    projection_a = session.prepare_result_projection(run_a.run_id)
    assert projection_a.record.materialization.generation == 1
    assert key_a in {
        field_data.key
        for field_data in projection_a.record.materialization.fields
    }


@pytest.mark.parametrize("terminal", ("failed", "cancelled"))
def test_failed_or_cancelled_materialization_can_retry(
    terminal: str,
) -> None:
    session = _session()
    solve = _succeed(session, "Job-1", marker=6.0)
    key = _key(session.current_result(), FieldPosition.CENTROID)
    task = session.prepare_result_materialization(solve.run_id, (key,))
    before_revision = session.session_revision

    if terminal == "failed":
        receipt = session.accept_task_failed(task.token, "recovery failed")
    else:
        receipt = session.accept_task_cancelled(task.token)

    assert receipt.accepted
    assert receipt.changed == frozenset()
    assert session.session_revision == before_revision
    assert session.current_result().materialization.generation == 0
    assert (
        session.validate_task_token(task.token)
        is TokenStatus.ALREADY_COMPLETED
    )

    retry = session.prepare_result_materialization(solve.run_id, (key,))
    accepted = session.accept_result_materialization(
        retry.token,
        _materialize(retry),
    )
    assert accepted.accepted
    assert session.current_result().materialization.generation == 1


def test_model_edit_makes_materialization_task_stale() -> None:
    session = _session()
    solve = _succeed(session, "Job-1", marker=7.0)
    key = _key(session.current_result(), FieldPosition.CENTROID)
    task = session.prepare_result_materialization(solve.run_id, (key,))
    patch = _materialize(task)

    session.replace_model_definitions(
        (),
        (),
        (),
        (AnalysisStep("Step-A"),),
    )
    before_revision = session.session_revision
    rejected = session.accept_result_materialization(task.token, patch)

    assert not rejected.accepted
    assert rejected.token_status in {
        TokenStatus.STALE_ARTIFACT,
        TokenStatus.STALE_REVISION,
        TokenStatus.STALE_RUN,
    }
    assert session.session_revision == before_revision
    assert session.current_result() is None
