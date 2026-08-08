from __future__ import annotations

from dataclasses import replace
from collections.abc import Mapping
from pathlib import Path

import pytest

from fem.application import (
    ChangeKind,
    ResultArchiveSaveSnapshot,
    ResultFileState,
    RunStatus,
    SessionStateError,
    TokenStatus,
)
from fem.application.results import FieldPosition
from tests.application.test_result_materialization_session import (
    _key,
    _materialize,
    _session,
    _succeed,
)


def test_prepare_result_archive_save_binds_exact_source_and_generation() -> None:
    session = _session()
    solve = _succeed(session, "Job-1", marker=1.0)
    provider = session.current_result_provider()
    assert provider is not None

    payload = session.prepare_result_archive_save(solve.run_id)

    assert type(payload) is ResultArchiveSaveSnapshot
    assert payload.token.task_kind == "result_archive_save"
    assert payload.token.session_id == session.session_id
    assert payload.token.artifact_id == provider.source.artifact_id
    assert payload.token.run_id == solve.run_id
    assert payload.token.result_id == provider.source.result_id
    assert dict(payload.token.dependency_revisions) == {
        "materialization_generation": 0,
        "model_revision": session.model_revision,
    }
    assert payload.source == provider.source
    assert payload.archive.source == provider.source
    assert payload.archive.materialization is provider.snapshot
    assert payload.archive.materialization.topology._node_coordinates is (
        provider.snapshot.topology._node_coordinates
    )
    assert payload.archive.model_projection.topology is (
        payload.archive.materialization.topology
    )
    summaries = payload.archive.model_projection.summaries
    assert {
        "model",
        "mesh",
        "parts",
        "materials",
        "sections",
        "assignments",
    } <= set(summaries)
    assert summaries["mesh"]["node_count"] == len(
        payload.archive.materialization.topology.node_ids
    )

    def assert_json_safe(value):
        if isinstance(value, Mapping):
            for key, item in value.items():
                assert type(key) is str
                assert_json_safe(item)
            return
        if type(value) is tuple:
            for item in value:
                assert_json_safe(item)
            return
        assert value is None or type(value) in {bool, int, float, str}

    assert_json_safe(summaries)
    assert all(
        not any(separator in item for separator in ("/", "\\"))
        for item in str(summaries).split()
        if isinstance(item, str)
    )
    assert payload.archive.run.name == "Job-1"
    assert session.snapshot().has_unsaved_results


def test_prepare_does_not_call_public_array_copy_properties(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _session()
    solve = _succeed(session, "Job-1", marker=1.0)
    topology_type = type(session.current_result_provider().snapshot.topology)

    def fail_copy(_self):
        raise AssertionError("prepare must not copy public topology arrays")

    monkeypatch.setattr(topology_type, "node_coordinates", property(fail_copy))
    monkeypatch.setattr(topology_type, "nodal_displacements", property(fail_copy))

    payload = session.prepare_result_archive_save(solve.run_id)
    assert payload.archive.materialization.generation == 0


def test_prepare_reuses_acceptance_fingerprint_without_array_scan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _session()
    solve = _succeed(session, "Job-1", marker=1.0)
    import fem.application.results.archive as archive_module

    monkeypatch.setattr(
        archive_module,
        "result_model_fingerprint",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("prepare must reuse the cached model fingerprint")
        ),
    )
    payload = session.prepare_result_archive_save(solve.run_id)
    assert len(payload.archive.origin.model_fingerprint) == 64


def test_prepare_requires_cached_acceptance_fingerprint() -> None:
    session = _session()
    solve = _succeed(session, "Job-1", marker=1.0)
    session._result_model_fingerprints.pop(solve.run_id)

    with pytest.raises(SessionStateError, match="fingerprint"):
        session.prepare_result_archive_save(solve.run_id)


@pytest.mark.parametrize("terminal", ("failed", "cancelled"))
def test_non_successful_runs_cannot_prepare_archive_save(terminal: str) -> None:
    session = _session()
    solve = session.prepare_solve("Step-A", "Job-1")
    if terminal == "failed":
        session.accept_run_failed(solve.token, "solver failed")
    else:
        session.accept_run_cancelled(solve.token)

    with pytest.raises(SessionStateError):
        session.prepare_result_archive_save(solve.run_id)

    pending = session.prepare_solve("Step-A", "Job-2")
    with pytest.raises(SessionStateError):
        session.prepare_result_archive_save(pending.run_id)


def test_accept_save_updates_only_result_file_state_and_preserves_project_dirty() -> None:
    session = _session()
    solve = _succeed(session, "Job-1", marker=1.0)
    before = session.snapshot()
    payload = session.prepare_result_archive_save(solve.run_id)

    accepted = session.accept_result_archive_saved(payload.token, Path("job.femres"))
    after = session.snapshot()

    assert accepted.accepted
    assert accepted.changed == {ChangeKind.SAVED_STATE}
    assert type(after.result_file_states[solve.run_id]) is ResultFileState
    assert after.result_file_state.path == Path("job.femres")
    assert after.result_file_state.saved_generation == 0
    assert not after.has_unsaved_results
    assert after.project_revision == before.project_revision
    assert after.saved_project_revision == before.saved_project_revision
    assert after.model_revision == before.model_revision
    assert after.dirty == before.dirty
    assert after.runs[0].status is RunStatus.SUCCEEDED


def test_materialization_keeps_old_path_but_marks_result_unsaved() -> None:
    session = _session()
    solve = _succeed(session, "Job-1", marker=1.0)
    payload = session.prepare_result_archive_save(solve.run_id)
    session.accept_result_archive_saved(payload.token, "old.femres")
    key = _key(session.current_result(), FieldPosition.CENTROID)
    task = session.prepare_result_materialization(solve.run_id, (key,))

    session.accept_result_materialization(task.token, _materialize(task))
    state = session.result_file_state_for(solve.run_id)

    assert state is not None
    assert state.path == Path("old.femres")
    assert state.saved_generation == 0
    assert session.current_result_identity()[1] == 1
    assert session.has_unsaved_results
    assert session.unsaved_result_run_ids == (solve.run_id,)


def test_stale_failure_cancel_and_foreign_callbacks_do_not_mark_saved() -> None:
    session = _session()
    first = _succeed(session, "Job-1", marker=1.0)
    first_payload = session.prepare_result_archive_save(first.run_id)
    second_payload = session.prepare_result_archive_save(first.run_id)
    assert first_payload.token.task_id not in session._task_data

    superseded = session.accept_result_archive_saved(
        first_payload.token,
        "superseded.femres",
    )
    assert not superseded.accepted
    assert superseded.token_status is TokenStatus.STALE_REVISION
    assert session.result_file_state_for(first.run_id) is None

    failed = session.accept_result_archive_save_failed(
        second_payload.token,
        "worker failed",
    )
    assert failed.accepted
    assert session.result_file_state_for(first.run_id) is None

    retry = session.prepare_result_archive_save(first.run_id)
    foreign = replace(retry.token, session_id="foreign-session")
    rejected = session.accept_result_archive_saved(foreign, "foreign.femres")
    assert not rejected.accepted
    assert rejected.token_status is TokenStatus.STALE_SESSION
    assert session.result_file_state_for(first.run_id) is None

    cancelled = session.accept_result_archive_save_cancelled(retry.token)
    assert cancelled.accepted
    assert session.result_file_state_for(first.run_id) is None


def test_materialization_stales_old_save_and_retry_saves_new_generation() -> None:
    session = _session()
    solve = _succeed(session, "Job-1", marker=1.0)
    old_save = session.prepare_result_archive_save(solve.run_id)
    key = _key(session.current_result(), FieldPosition.CENTROID)
    materialization = session.prepare_result_materialization(solve.run_id, (key,))
    session.accept_result_materialization(
        materialization.token,
        _materialize(materialization),
    )

    assert old_save.token.task_id not in session._task_data
    stale = session.accept_result_archive_saved(old_save.token, "old.femres")
    assert not stale.accepted
    assert stale.token_status is TokenStatus.STALE_REVISION
    assert session.result_file_state_for(solve.run_id) is None
    assert session.has_unsaved_results

    new_save = session.prepare_result_archive_save(solve.run_id)
    assert new_save.materialization_generation == 1
    session.accept_result_archive_saved(new_save.token, "new.femres")
    state = session.result_file_state_for(solve.run_id)
    assert state is not None
    assert state.saved_generation == 1
    assert state.path == Path("new.femres")
    assert not session.has_unsaved_results


def test_multiple_successful_runs_keep_independent_file_state() -> None:
    session = _session()
    first = _succeed(session, "Job-A", marker=1.0)
    first_payload = session.prepare_result_archive_save(first.run_id)
    session.accept_result_archive_saved(first_payload.token, "a.femres")
    second = _succeed(session, "Job-B", marker=2.0)
    second_payload = session.prepare_result_archive_save(second.run_id)
    session.accept_result_archive_saved(second_payload.token, "b.femres")

    states = session.snapshot().result_file_states
    assert states[first.run_id].path == Path("a.femres")
    assert states[second.run_id].path == Path("b.femres")
    assert not session.has_unsaved_results

    session.select_result(first.run_id)
    assert session.snapshot().result_path == Path("a.femres")
    session.select_result(second.run_id)
    assert session.snapshot().result_path == Path("b.femres")


def test_model_invalidation_clears_result_file_states_and_save_token() -> None:
    session = _session()
    solve = _succeed(session, "Job-1", marker=1.0)
    payload = session.prepare_result_archive_save(solve.run_id)
    session.accept_result_archive_saved(payload.token, "job.femres")
    late_payload = session.prepare_result_archive_save(solve.run_id)

    session.replace_model_definitions((), (), (), session.snapshot().steps)

    assert session.snapshot().result_file_states == {}
    assert not session.snapshot().has_unsaved_results
    late = session.accept_result_archive_saved(late_payload.token, "late.femres")
    assert not late.accepted
    assert late.token_status in {
        TokenStatus.STALE_REVISION,
        TokenStatus.STALE_ARTIFACT,
        TokenStatus.STALE_RUN,
        TokenStatus.STALE_RESULT,
        TokenStatus.ALREADY_COMPLETED,
    }
