from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path
from threading import Event
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtWidgets import QApplication

import fem.application.definitions as definitions_module
import fem.application.session as session_module
import fem_gui.main_window as main_window_module
from fem.abaqus import read
from fem.application import RegionAssignment, describe_session_authoring
from fem.application.results import (
    FieldPosition,
    FieldRequest,
    ResultFieldId,
    ResultVariable,
    build_solve_result_bundle,
    restore_result_provider,
)
from fem.solvers.static_linear import solve
from fem_gui.main_window import FEMMainWindow
from fem_gui.model_dialogs import RegionAssignmentDialog
from fem_gui.task_controller import TaskApplyStatus
from fem_gui.visualization.model_adapter import build_model_geometry
from fem_gui.workers import TaskContext
from tests.helpers.model_builders import make_static_pull_truss_model
from tests.helpers.preflight_builders import passing_preflight_report


def _application() -> QApplication:
    return QApplication.instance() or QApplication([])


def _window_with_imported_model() -> FEMMainWindow:
    _application()
    window = FEMMainWindow()
    model = make_static_pull_truss_model()
    task = window.session.prepare_import(Path("projection.inp"))
    delta = window.session.accept_imported_model(task.token, model)

    assert window._apply_session_delta(
        delta,
        model_geometry=build_model_geometry(model),
        source_label="projection.inp",
    )
    return window


def _install_successful_result(
    window: FEMMainWindow,
    *,
    run_name: str = "Job-1",
) -> str:
    validation = window.session.prepare_validation("pull")
    assert window._apply_session_delta(
        window.session.accept_validation(
            validation.token,
            passing_preflight_report(validation.token),
        )
    )
    solve_task = window.session.prepare_solve("pull", run_name)
    assert solve_task.delta is not None
    assert window._apply_session_delta(solve_task.delta)
    assert window._apply_session_delta(
        window.session.begin_run(solve_task.token)
    )

    result = solve(solve_task.model, solve_task.step_name, name=run_name)
    assert window._apply_session_delta(
        window.session.accept_run_succeeded(
            solve_task.token,
            build_solve_result_bundle(solve_task, result),
        ),
    )
    return solve_task.run_id


def _centroid_stress_key(record):
    provider = restore_result_provider(
        record.result,
        record.materialization,
    )
    return provider.resolve_request(
        FieldRequest(
            ResultFieldId(
                ResultVariable.S,
                FieldPosition.CENTROID,
            )
        )
    )


def _materialize_task(task):
    provider = restore_result_provider(
        task.record.result,
        task.record.materialization,
    )
    return provider.materialize(task.field_keys)


def test_async_import_acceptance_and_projection_do_not_copy_on_gui_thread(
    monkeypatch,
) -> None:
    _application()
    window = FEMMainWindow()
    model = make_static_pull_truss_model()
    captured = {}

    monkeypatch.setattr(main_window_module, "parse_file", lambda _path: object())
    monkeypatch.setattr(
        main_window_module,
        "build_abaqus_model_with_report",
        lambda _deck: SimpleNamespace(model=model, notices=()),
    )

    def capture_start(workload, on_success, *_args, **kwargs):
        captured["workload"] = workload
        captured["on_success"] = on_success
        captured["apply_result"] = kwargs["apply_result"]
        return True

    monkeypatch.setattr(window, "_start_task", capture_start)
    assert window._begin_import(Path("worker-owned.inp"))
    payload = captured["workload"](
        TaskContext(1, Event(), lambda _task_id, _stage: None)
    )

    def unexpected_deepcopy(_value):
        raise AssertionError("GUI import acceptance must not deepcopy")

    def unexpected_snapshot():
        raise AssertionError("GUI projection must not use detached snapshot")

    monkeypatch.setattr(session_module, "deepcopy", unexpected_deepcopy)
    monkeypatch.setattr(definitions_module, "deepcopy", unexpected_deepcopy)
    monkeypatch.setattr(main_window_module, "deepcopy", unexpected_deepcopy)
    monkeypatch.setattr(window.session, "snapshot", unexpected_snapshot)

    outcome = captured["apply_result"](payload)
    assert outcome.status is TaskApplyStatus.ACCEPTED
    captured["on_success"](outcome.projection_value)

    assert window.document.model is not model
    assert window.document.source_path == Path("worker-owned.inp")
    assert window.geometry is not None
    window.close()


def test_result_and_run_status_deltas_reuse_trusted_gui_projection(
    monkeypatch,
) -> None:
    window = _window_with_imported_model()
    run_a = _install_successful_result(window, run_name="Job-A")
    _install_successful_result(window, run_name="Job-B")
    pending = window.session.prepare_solve("pull", "Job-Pending")
    assert pending.delta is not None
    assert window._apply_session_delta(pending.delta)

    projection_a = window.session.prepare_result_projection(run_a)
    key_a = _centroid_stress_key(projection_a.record)
    materialization = window.session.prepare_result_materialization(
        run_a,
        (key_a,),
    )
    patch = _materialize_task(materialization)
    model_before = window.document.model
    artifact_before = window.document.artifact
    select_delta = window.session.select_result(run_a)

    def unexpected_deepcopy(_value):
        raise AssertionError("GUI delta projection must not deepcopy the model")

    def unexpected_detach(_record):
        raise AssertionError("GUI delta projection must not detach a result")

    def unexpected_snapshot():
        raise AssertionError("GUI delta projection must not use snapshot()")

    monkeypatch.setattr(session_module, "deepcopy", unexpected_deepcopy)
    monkeypatch.setattr(
        session_module,
        "detached_result_record",
        unexpected_detach,
    )
    monkeypatch.setattr(window.session, "snapshot", unexpected_snapshot)

    assert window._apply_session_delta(select_delta)
    selected_provider = window.result_provider
    assert selected_provider is not None
    assert selected_provider.source.run_id == run_a

    result_delta = window.session.accept_result_materialization(
        materialization.token,
        patch,
    )
    assert window._apply_session_delta(result_delta)
    assert window.result_provider is not selected_provider
    assert window.result_provider.snapshot.generation == 1

    status_delta = window.session.request_cancel(pending.run_id)
    assert window._apply_session_delta(status_delta)
    assert window.document.model is model_before
    assert window.document.artifact is artifact_before
    assert next(
        run
        for run in window.document.runs
        if run.run_id == pending.run_id
    ).cancellation_requested
    window.close()


def test_one_delta_projects_result_to_every_gui_consumer() -> None:
    window = _window_with_imported_model()

    run_id = _install_successful_result(window)
    artifact_id = window.document.artifact.artifact_id
    record = window.session.current_result()
    provider = window.result_provider
    selection = window.result_selection
    payload = window.viewport._result_render_payload

    assert window.document.displayed_result_run_id == run_id
    assert record is not None
    assert provider is not None
    assert window.session.current_result_provider() is provider
    assert window.session.current_result_identity() == (
        provider.source,
        provider.snapshot.generation,
    )
    assert selection is not None
    assert payload is not None
    assert provider.source.artifact_id == artifact_id
    assert provider.source.run_id == run_id
    assert provider.source.result_id == record.result_id
    assert provider.snapshot.generation == record.materialization.generation == 0
    assert provider.catalog().default_selection == selection
    assert window.viewport.artifact_id == artifact_id
    assert payload.topology.source == provider.source
    assert payload.topology.selection == selection
    assert window.inspection_service is not None
    assert provider.source == record.materialization.source
    assert (
        window.inspection_service.result_provider
        is provider
    )
    assert not hasattr(window, "result_data")
    assert window.result_tree.topLevelItem(0).text(0) == "分析结果"
    assert window.result_tree.topLevelItem(0).child(0).text(0) == "pull"
    assert window.actions["query"].isEnabled()
    assert window.result_variable_combo.isEnabled()
    window.close()


def test_same_run_generation_rebuilds_projection_and_preserves_ready_fields() -> None:
    window = _window_with_imported_model()
    run_id = _install_successful_result(window)
    provider_before = window.result_provider
    selection_before = window.result_selection
    record = window.session.current_result()
    assert provider_before is not None
    assert window.session.current_result_provider() is provider_before
    assert selection_before is not None
    assert record is not None
    assert provider_before.snapshot.generation == 0
    ready_before = {
        field.key: field.values.copy()
        for field in provider_before.snapshot.fields
    }
    key = _centroid_stress_key(record)
    assert all(field.key != key for field in provider_before.snapshot.fields)
    task = window.session.prepare_result_materialization(run_id, (key,))

    delta = window.session.accept_result_materialization(
        task.token,
        _materialize_task(task),
    )

    assert window._apply_session_delta(delta)
    accepted = window.session.current_result()
    provider_after = window.result_provider
    assert accepted is not None
    assert provider_after is not None
    assert window.session.current_result_provider() is provider_after
    assert window.session.current_result_identity() == (
        provider_after.source,
        provider_after.snapshot.generation,
    )
    assert provider_after is not provider_before
    assert provider_after.source.run_id == run_id
    assert provider_after.source.result_id == accepted.result_id == record.result_id
    assert (
        provider_after.snapshot.generation
        == accepted.materialization.generation
        == 1
    )
    for field_key, expected in ready_before.items():
        np.testing.assert_array_equal(
            provider_after.field(field_key).values,
            expected,
        )
    assert provider_after.field(key).key == key
    assert window.inspection_service is not None
    assert window.result_selection == selection_before
    payload = window.viewport._result_render_payload
    assert payload is not None
    assert payload.topology.source == provider_after.source
    assert payload.topology.materialization_generation == 1
    assert payload.topology.selection == selection_before
    assert (
        window.inspection_service.result_provider
        is provider_after
    )
    window.close()


def test_hidden_run_materialization_does_not_replace_displayed_actor(
    monkeypatch,
) -> None:
    window = _window_with_imported_model()
    run_a = _install_successful_result(window, run_name="Job-A")
    record_a = window.session.current_result()
    assert record_a is not None
    key_a = _centroid_stress_key(record_a)
    task_a = window.session.prepare_result_materialization(run_a, (key_a,))
    patch_a = _materialize_task(task_a)
    provider_a = window.result_provider
    assert provider_a is not None

    run_b = _install_successful_result(window, run_name="Job-B")
    provider_b = window.result_provider
    selection_b = window.result_selection
    payload_b = window.viewport._result_render_payload
    assert provider_b is not None
    assert selection_b is not None
    assert payload_b is not None
    assert provider_b is not provider_a
    assert provider_b.source.run_id == run_b
    assert selection_b == provider_b.catalog().default_selection
    render_calls = []
    original_set_result_render_payload = (
        window.viewport.set_result_render_payload
    )

    def record_set_result_render_payload(payload):
        render_calls.append(payload)
        original_set_result_render_payload(payload)

    monkeypatch.setattr(
        window.viewport,
        "set_result_render_payload",
        record_set_result_render_payload,
    )

    delta = window.session.accept_result_materialization(
        task_a.token,
        patch_a,
    )

    assert window._apply_session_delta(delta)
    assert render_calls == []
    assert window.document.displayed_result_run_id == run_b
    assert window.result_provider is provider_b
    assert window.result_selection == selection_b
    assert window.viewport._result_render_payload is payload_b
    assert provider_b.snapshot.generation == 0

    projection_a = window.session.prepare_result_projection(run_a)
    provider_a = restore_result_provider(
        projection_a.record.result,
        projection_a.record.materialization,
    )
    assert provider_a.source.run_id == run_a
    assert provider_a.source.result_id == projection_a.record.result_id
    assert provider_a.snapshot.generation == 1
    assert provider_a.field(key_a).key == key_a
    window.close()


def test_artifact_and_run_mismatches_never_leave_stale_gui_caches() -> None:
    window = _window_with_imported_model()
    old_run_id = _install_successful_result(window)
    old_artifact_id = window.document.artifact.artifact_id

    stale_geometry = replace(window.geometry, artifact_id="stale-artifact")
    window.geometry = stale_geometry

    delta = window.session.replace_model_definitions(
        window.document.materials,
        window.document.sections,
        window.document.assignments,
        window.document.steps,
    )
    assert window._apply_session_delta(delta)

    current_artifact_id = window.document.artifact.artifact_id
    assert current_artifact_id != old_artifact_id
    assert window.geometry is not stale_geometry
    assert window.geometry.artifact_id == current_artifact_id
    assert window.viewport.artifact_id == current_artifact_id
    assert window.document.displayed_result_run_id is None
    assert window.session.find_run(old_run_id) is None
    assert window.inspection_service is not None
    assert window.inspection_service.result_provider is None
    assert window.result_provider is None
    assert window.result_selection is None
    assert not window.actions["query"].isEnabled()
    assert not window.result_variable_combo.isEnabled()
    window.close()


def test_action_gates_use_the_session_authoring_lifecycle_projection() -> None:
    window = _window_with_imported_model()
    window._update_action_states()
    initial = describe_session_authoring(window.document).step("pull")

    assert initial is not None
    assert initial.can_check
    assert not initial.can_submit
    assert window.actions["check_model"].isEnabled()
    assert not window.actions["submit_job"].isEnabled()

    validation = window.session.prepare_validation("pull")
    assert window._apply_session_delta(
        window.session.accept_validation(
            validation.token,
            passing_preflight_report(validation.token),
        )
    )
    window._update_action_states()
    validated = describe_session_authoring(window.document).step("pull")

    assert validated is not None
    assert validated.can_check
    assert validated.can_submit
    assert window.actions["check_model"].isEnabled()
    assert window.actions["submit_job"].isEnabled()
    window.close()


def test_assignment_edit_replaces_the_selected_index_through_session(
    monkeypatch,
    gui_inp_path,
) -> None:
    window = FEMMainWindow()
    model = read(gui_inp_path)
    window._model_loaded(
        gui_inp_path,
        (model, build_model_geometry(model)),
    )
    before = window.document
    original_section = before.sections[0]
    original_assignment = before.assignments[0]
    second_section = replace(original_section, name="Section-2")
    second_assignment = RegionAssignment(
        second_section.name,
        original_assignment.region_name,
    )
    assert window._apply_session_delta(
        window.session.replace_model_definitions(
            before.materials,
            (*before.sections, second_section),
            (*before.assignments, second_assignment),
            before.steps,
        )
    )
    revision = window.document.session_revision
    replacement = RegionAssignment(
        second_section.name,
        original_assignment.region_name,
    )
    errors = []
    monkeypatch.setattr(
        window,
        "_show_error",
        lambda title, message: errors.append((title, message)),
    )
    monkeypatch.setattr(
        RegionAssignmentDialog,
        "exec",
        lambda _dialog: True,
    )
    monkeypatch.setattr(
        RegionAssignmentDialog,
        "assignment",
        lambda _dialog: replacement,
    )

    window.edit_region_assignment(0)

    assert window.document.session_revision == revision + 1
    assert window.document.assignments == (
        replacement,
        second_assignment,
    )
    assert errors == []
    window.close_model(confirm=False)
    window.close()
