from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from fem.abaqus import read
from fem.application import RegionAssignment
from fem.solvers.static_linear import solve
from fem_gui.main_window import FEMMainWindow
from fem_gui.model_dialogs import RegionAssignmentDialog
from fem_gui.visualization.model_adapter import build_model_geometry
from fem_gui.visualization.result_adapter import build_result_data
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
    projection = replace(
        build_result_data(result, window.geometry),
        artifact_id=solve_task.token.artifact_id,
        run_id=solve_task.run_id,
    )
    assert window._apply_session_delta(
        window.session.accept_run_result(solve_task.token, result),
        result_projection=projection,
    )
    return solve_task.run_id


def test_one_delta_projects_result_to_every_gui_consumer() -> None:
    window = _window_with_imported_model()

    run_id = _install_successful_result(window)
    artifact_id = window.document.artifact.artifact_id

    assert window.document.displayed_result_run_id == run_id
    assert window.result_data is not None
    assert window.result_data.artifact_id == artifact_id
    assert window.result_data.run_id == run_id
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
        window.session.accept_run_result(solve_task.token, result),
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
        window.document.material_definitions,
        window.document.section_definitions,
        window.document.region_assignments,
        window.document.analysis_definitions,
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


def test_action_gates_delegate_validation_decisions_to_session_queries(
    monkeypatch,
) -> None:
    window = _window_with_imported_model()
    calls: list[tuple[str, str | None]] = []

    monkeypatch.setattr(
        window.session,
        "can_check",
        lambda step=None: calls.append(("check", step)) or False,
    )
    monkeypatch.setattr(
        window.session,
        "can_submit",
        lambda step=None: calls.append(("submit", step)) or False,
    )
    window._update_action_states()

    assert ("check", "pull") in calls
    assert ("submit", "pull") in calls
    assert not window.actions["check_model"].isEnabled()
    assert not window.actions["submit_job"].isEnabled()

    calls.clear()
    monkeypatch.setattr(
        window.session,
        "can_check",
        lambda step=None: calls.append(("check", step)) or True,
    )
    monkeypatch.setattr(
        window.session,
        "can_submit",
        lambda step=None: calls.append(("submit", step)) or True,
    )
    window._update_action_states()

    assert ("check", "pull") in calls
    assert ("submit", "pull") in calls
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
    original_section = before.section_definitions[0]
    original_assignment = before.region_assignments[0]
    second_section = replace(original_section, name="Section-2")
    second_assignment = RegionAssignment(
        second_section.name,
        original_assignment.region_name,
    )
    assert window._apply_session_delta(
        window.session.replace_model_definitions(
            before.material_definitions,
            (*before.section_definitions, second_section),
            (*before.region_assignments, second_assignment),
            before.analysis_definitions,
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
    assert window.document.region_assignments == (
        replacement,
        second_assignment,
    )
    assert errors == []
    window.close_model(confirm=False)
    window.close()
