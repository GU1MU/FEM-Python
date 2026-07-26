from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtWidgets import QApplication

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
from fem_gui.visualization.model_adapter import build_model_geometry
from fem_gui.visualization.result_adapter import (
    build_result_data,
    build_result_data_from_provider,
)
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


def test_one_delta_projects_result_to_every_gui_consumer() -> None:
    window = _window_with_imported_model()

    run_id = _install_successful_result(window)
    artifact_id = window.document.artifact.artifact_id
    record = window.session.current_result()

    assert window.document.displayed_result_run_id == run_id
    assert record is not None
    assert window.result_data is not None
    assert window.result_data.artifact_id == artifact_id
    assert window.result_data.run_id == run_id
    assert window.result_data.result_id == record.result_id
    assert (
        window.result_data.materialization_generation
        == record.materialization.generation
        == 0
    )
    assert window.result_data.field_ready("U")
    assert window.viewport.artifact_id == artifact_id
    assert window.viewport.run_id == run_id
    assert window.inspection_service is not None
    assert window.inspection_service.result_data is window.result_data
    assert run_id in {
        window.document.displayed_result_run_id,
        window.result_data.run_id,
        window.viewport.run_id,
    }
    assert window.result_tree.topLevelItem(0).text(0) == "分析结果"
    assert window.result_tree.topLevelItem(0).child(0).text(0) == "pull"
    assert window.actions["query"].isEnabled()
    assert window.result_variable_combo.isEnabled()
    window.close()


def test_same_run_generation_rebuilds_projection_and_preserves_ready_fields() -> None:
    window = _window_with_imported_model()
    run_id = _install_successful_result(window)
    before = window.result_data
    record = window.session.current_result()
    assert before is not None
    assert record is not None
    assert before.materialization_generation == 0
    assert not before.field_ready("CENTROID:S11")
    ready_before = {
        key: scalar.values.copy()
        for key, scalar in before.fields.items()
        if scalar.ready
    }
    key = _centroid_stress_key(record)
    task = window.session.prepare_result_materialization(run_id, (key,))

    delta = window.session.accept_result_materialization(
        task.token,
        _materialize_task(task),
    )

    assert window._apply_session_delta(delta)
    after = window.result_data
    accepted = window.session.current_result()
    assert after is not None
    assert accepted is not None
    assert after is not before
    assert after.run_id == run_id
    assert after.result_id == accepted.result_id == record.result_id
    assert (
        after.materialization_generation
        == accepted.materialization.generation
        == 1
    )
    for field_key, expected in ready_before.items():
        assert after.field_ready(field_key)
        np.testing.assert_array_equal(
            after.fields[field_key].values,
            expected,
        )
    assert after.field_ready("CENTROID:S11")
    assert after.field_selections["CENTROID:S11"].field_key == key
    assert window.inspection_service is not None
    assert window.inspection_service.result_data is after
    assert window.viewport.run_id == run_id
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

    run_b = _install_successful_result(window, run_name="Job-B")
    displayed_b = window.result_data
    assert displayed_b is not None
    assert displayed_b.run_id == run_b
    set_result_calls = []
    original_set_result_data = window.viewport.set_result_data

    def record_set_result_data(data):
        set_result_calls.append(data)
        original_set_result_data(data)

    monkeypatch.setattr(
        window.viewport,
        "set_result_data",
        record_set_result_data,
    )

    delta = window.session.accept_result_materialization(
        task_a.token,
        patch_a,
    )

    assert window._apply_session_delta(delta)
    assert set_result_calls == []
    assert window.document.displayed_result_run_id == run_b
    assert window.result_data is displayed_b
    assert window.viewport.run_id == run_b
    assert window.result_data.run_id == run_b
    assert window.result_data.materialization_generation == 0

    projection_a = window.session.prepare_result_projection(run_a)
    provider_a = restore_result_provider(
        projection_a.record.result,
        projection_a.record.materialization,
    )
    projected_a = build_result_data_from_provider(
        provider_a,
        window.geometry,
        legacy_result=projection_a.record.result,
    )
    assert projected_a.run_id == run_a
    assert projected_a.result_id == projection_a.record.result_id
    assert projected_a.materialization_generation == 1
    assert projected_a.field_ready("CENTROID:S11")
    window.close()


def test_unprovenanced_result_projection_is_never_relabelled_as_current() -> None:
    window = _window_with_imported_model()
    validation = window.session.prepare_validation("pull")
    assert window._apply_session_delta(
        window.session.accept_validation(
            validation.token,
            passing_preflight_report(validation.token),
        )
    )
    solve_task = window.session.prepare_solve("pull", "Job-1")
    assert window._apply_session_delta(solve_task.delta)
    assert window._apply_session_delta(window.session.begin_run(solve_task.token))
    result = solve(solve_task.model, solve_task.step_name, name="Job-1")
    stale_projection = build_result_data(result, window.geometry)

    assert window._apply_session_delta(
        window.session.accept_run_succeeded(
            solve_task.token,
            build_solve_result_bundle(solve_task, result),
        ),
        result_projection=stale_projection,
    )

    assert window.result_data is not stale_projection
    assert stale_projection.artifact_id is None
    assert stale_projection.run_id is None
    assert window.result_data is not None
    assert window.result_data.artifact_id == solve_task.token.artifact_id
    assert window.result_data.run_id == solve_task.run_id
    window.close()


def test_artifact_and_run_mismatches_never_leave_stale_gui_caches() -> None:
    window = _window_with_imported_model()
    old_run_id = _install_successful_result(window)
    old_artifact_id = window.document.artifact.artifact_id

    stale_geometry = replace(window.geometry, artifact_id="stale-artifact")
    window.geometry = stale_geometry
    window.result_data = replace(
        window.result_data,
        artifact_id="stale-artifact",
        run_id="stale-run",
    )

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
    assert window.viewport.run_id is None
    assert window.result_data is None
    assert window.document.displayed_result_run_id is None
    assert window.session.find_run(old_run_id) is None
    assert window.inspection_service is not None
    assert window.inspection_service.result_data is None
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
