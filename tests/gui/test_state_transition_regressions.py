from __future__ import annotations

import json
import os
from pathlib import Path
from time import monotonic

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import QThread
from PySide6.QtWidgets import QApplication

from fem.application import (
    ModelSession,
    NamedRegion,
    RegionAssignment,
    RunStatus,
    SectionDefinition,
    TokenStatus,
)
from fem.core.model import (
    AnalysisStep,
    DisplacementConstraint,
    MaterialDefinition,
    NodalLoad,
    OutputRequest,
)
from fem.application.results import build_solve_result_bundle
from fem.geometry import LogicalEntityRef
from fem.geometry.recipes import RectangleGeometry
from fem.io.project import save_project
from fem.mesh.settings import LocalMeshControl, MeshSettings
from fem.solvers.static_linear import solve
import fem_gui.main_window as main_window_module
from fem_gui.main_window import FEMMainWindow
from fem_gui.visualization.model_adapter import build_model_geometry
from tests.helpers.model_builders import (
    make_static_pull_truss_model,
    make_two_step_static_pull_truss_model,
)
from tests.helpers.preflight_builders import passing_preflight_report


_RESULT_ACTIONS = (
    "undeformed",
    "deformed",
    "contour",
    "overlay",
    "field",
    "display_settings",
    "scale",
    "contour_options",
    "query",
    "export_csv",
    "export_vtk",
    "screenshot",
)


def _wait_for_task(window: FEMMainWindow, timeout: float = 2.0) -> None:
    application = QApplication.instance() or QApplication([])
    deadline = monotonic() + timeout
    def busy() -> bool:
        controller = window.workspace.open_controller
        return window.busy or (controller is not None and controller.busy)

    while busy() and monotonic() < deadline:
        application.processEvents()
        QThread.msleep(1)
    application.processEvents()
    assert not busy()


def _application() -> QApplication:
    return QApplication.instance() or QApplication([])


def _new_window() -> FEMMainWindow:
    _application()
    return FEMMainWindow()


def _install_imported(
    window: FEMMainWindow,
    model=None,
    *,
    path: str = "regression.inp",
) -> None:
    installed_model = model or make_static_pull_truss_model()
    if model is None:
        installed_model.steps[0].outputs = (
            OutputRequest("field", "node", ("U",)),
        )
    task = window.session.prepare_import(Path(path))
    assert window._apply_session_delta(
        window.session.accept_imported_model(task.token, installed_model),
        model_geometry=build_model_geometry(installed_model),
        source_label=path,
    )


def _validate_step(window: FEMMainWindow, step_name: str) -> None:
    task = window.session.prepare_validation(step_name)
    assert window._apply_session_delta(
        window.session.accept_validation(
            task.token,
            passing_preflight_report(task.token),
        )
    )


def _succeed_run(
    window: FEMMainWindow,
    *,
    step_name: str = "pull",
    run_name: str = "Job-1",
) -> str:
    if not window.session.can_submit(step_name):
        _validate_step(window, step_name)
    task = window.session.prepare_solve(step_name, run_name)
    assert task.delta is not None
    assert window._apply_session_delta(task.delta)
    assert window._apply_session_delta(window.session.begin_run(task.token))
    result = solve(task.model, task.step_name, name=run_name)
    assert window._apply_session_delta(
        window.session.accept_run_succeeded(
            task.token,
            build_solve_result_bundle(task, result),
        )
    )
    return task.run_id


def _projection_signature(window: FEMMainWindow) -> tuple[object, ...]:
    artifact_id = (
        None
        if window.document.artifact is None
        else window.document.artifact.artifact_id
    )
    return (
        window.document.session_id,
        window.document.session_revision,
        artifact_id,
        window.document.displayed_result_run_id,
        window.geometry.artifact_id if window.geometry is not None else None,
        (
            window.result_provider.source.run_id
            if window.result_provider is not None
            else None
        ),
        window.viewport.artifact_id,
        (
            window.viewport._result_render_payload.topology.source.run_id
            if window.viewport._result_render_payload is not None
            else None
        ),
        window.model_tree.topLevelItem(0).text(0),
        window.result_tree.topLevelItem(0).text(0),
    )


def _assert_result_entries_disabled(window: FEMMainWindow) -> None:
    assert all(not window.actions[name].isEnabled() for name in _RESULT_ACTIONS)
    assert not window.result_variable_combo.isEnabled()
    assert not window.result_component_combo.isEnabled()
    assert not window.result_position_combo.isEnabled()
    assert not window.result_scale_combo.isEnabled()


def test_delete_and_recreate_geometry_remove_all_topology_references(
    monkeypatch,
) -> None:
    window = _new_window()
    monkeypatch.setattr(
        main_window_module.QMessageBox,
        "question",
        lambda *_args, **_kwargs: (
            main_window_module.QMessageBox.StandardButton.Yes
        ),
    )
    window._create_native_model("Model-1")
    window._set_native_geometry(
        RectangleGeometry("Plate", 2.0, 1.0),
        "矩形",
    )
    assert window._apply_session_delta(
        window.session.replace_named_regions(
            (
                NamedRegion(
                    "SolidDomain",
                    (LogicalEntityRef("body:P1/domain"),),
                ),
                NamedRegion(
                    "Fixed",
                    (LogicalEntityRef("edge:P1/left"),),
                ),
                NamedRegion(
                    "Loaded",
                    (LogicalEntityRef("edge:P1/right"),),
                ),
            )
        )
    )
    assert window._apply_session_delta(
        window.session.replace_mesh_settings(
            MeshSettings(
                0.5,
                local_controls=(
                    LocalMeshControl(
                        LogicalEntityRef("edge:P1/right"),
                        0.1,
                    ),
                ),
            )
        )
    )
    step = AnalysisStep(
        "Load",
        boundaries=(DisplacementConstraint("Fixed", 1, 2, 0.0),),
        cloads=(NodalLoad("Loaded", 1, 10.0),),
    )
    assert window._apply_session_delta(
        window.session.replace_model_definitions(
            (MaterialDefinition("Steel", {"E": 210000.0, "nu": 0.3}),),
            (SectionDefinition("Section-1", "Steel"),),
            (RegionAssignment("Section-1", "SolidDomain"),),
            (step,),
        )
    )

    window.delete_geometry()
    deleted = window.document

    assert deleted.geometry_recipe is None
    assert deleted.parts == ()
    assert deleted.feature_history == ()
    assert not deleted.named_regions
    assert deleted.assignments == ()
    assert deleted.steps == ()
    assert deleted.mesh_settings is None
    assert deleted.artifact is None
    assert window.geometry is None
    _assert_result_entries_disabled(window)

    window._set_native_geometry(
        RectangleGeometry("Replacement", 3.0, 1.5),
        "新矩形",
    )
    recreated = window.document
    assert recreated.geometry_recipe.name == "Replacement"
    assert not recreated.named_regions
    assert recreated.assignments == ()
    assert recreated.steps == ()
    window.close()


def test_named_region_change_retains_completed_result_history() -> None:
    window = _new_window()
    window._create_native_model("Model-1")
    window._set_native_geometry(
        RectangleGeometry("Plate", 2.0, 1.0),
        "矩形",
    )
    assert window._apply_session_delta(
        window.session.replace_named_regions(
            (
                NamedRegion(
                    "Old",
                    (LogicalEntityRef("edge:P1/bottom"),),
                ),
                NamedRegion(
                    "FIXED",
                    (LogicalEntityRef("edge:P1/left"),),
                ),
                NamedRegion(
                    "TIP",
                    (LogicalEntityRef("edge:P1/right"),),
                ),
            )
        )
    )
    step = AnalysisStep(
        "pull",
        boundaries=(
            DisplacementConstraint("FIXED", 1, 3, 0.0),
            DisplacementConstraint("TIP", 2, 3, 0.0),
        ),
        cloads=(NodalLoad("TIP", 1, 100.0),),
    )
    assert window._apply_session_delta(
        window.session.replace_model_definitions(
            (),
            (),
            (),
            (step,),
        )
    )
    task = window.session.prepare_mesh_generation()
    model = make_static_pull_truss_model()
    assert window._apply_session_delta(
        window.session.accept_generated_model(task.token, model),
        model_geometry=build_model_geometry(model),
    )
    run_id = _succeed_run(window)

    assert window._apply_session_delta(
        window.session.replace_named_regions(
            (
                NamedRegion(
                    "Changed",
                    (LogicalEntityRef("edge:P1/right"),),
                ),
                NamedRegion(
                    "FIXED",
                    (LogicalEntityRef("edge:P1/left"),),
                ),
                NamedRegion(
                    "TIP",
                    (LogicalEntityRef("edge:P1/right"),),
                ),
            )
        )
    )

    assert set(window.document.named_regions) == {
        "Changed",
        "FIXED",
        "TIP",
    }
    assert window.document.artifact is None
    assert not window.document.validations
    assert tuple(run.run_id for run in window.document.runs) == (run_id,)
    retained = window.session.find_run(run_id)
    assert retained is not None and retained.has_result
    assert window.result_provider is None
    assert window.result_selection is None
    assert window.viewport._result_render_payload is None
    _assert_result_entries_disabled(window)
    window._confirm_discard_changes = lambda: True
    window.close()


def test_native_geometry_module_restores_pre_mesh_geometry() -> None:
    window = _new_window()
    window._create_native_model("Model-1")
    window._set_native_geometry(
        RectangleGeometry("Plate", 2.0, 1.0),
        "矩形",
    )
    task = window.session.prepare_mesh_generation()
    model = make_static_pull_truss_model()
    assert window._apply_session_delta(
        window.session.accept_generated_model(task.token, model),
        model_geometry=build_model_geometry(model),
    )

    assert window.viewport._geometry_preview is not None
    window.ribbon.set_current("模型")
    assert window.viewport._geometry_preview is None
    window.ribbon.set_current("几何")
    preview = window.viewport._geometry_preview
    assert preview is not None
    assert preview.faces
    assert preview.topological_dimension == 2

    for module_name in ("网格", "模型", "分析"):
        window.ribbon.set_current(module_name)
        assert window.viewport._geometry_preview is None
        assert window.viewport._result_render_payload is None
        assert window.viewport.artifact_id == window.document.artifact.artifact_id

    window.ribbon.set_current("几何")
    assert window.viewport._geometry_preview == preview
    window.close()


def test_definition_change_replaces_artifact_and_retains_result_history() -> None:
    window = _new_window()
    _install_imported(window)
    _succeed_run(window)
    old_artifact_id = window.document.artifact.artifact_id

    materials = tuple(window.document.materials)
    assert window._apply_session_delta(
        window.session.replace_model_definitions(
            materials,
            window.document.sections,
            window.document.assignments,
            window.document.steps,
        )
    )

    assert window.document.artifact.artifact_id != old_artifact_id
    assert not window.document.validations
    assert len(window.document.runs) == 1
    assert window.document.runs[0].has_result
    assert window.session.current_result() is None
    assert window.result_provider is None
    assert window.result_selection is None
    assert not window.actions["submit_job"].isEnabled()
    _assert_result_entries_disabled(window)
    window._confirm_discard_changes = lambda: True
    window.close()


def test_step_switch_uses_independent_validation_stamp_for_action_gate() -> None:
    window = _new_window()
    _install_imported(window, make_two_step_static_pull_truss_model())
    _validate_step(window, "pull1")

    window._set_current_step("pull1")
    assert window.actions["submit_job"].isEnabled()

    window._set_current_step("pull2")
    assert window._current_step_name == "pull2"
    assert not window.actions["submit_job"].isEnabled()

    _validate_step(window, "pull2")
    assert window.actions["submit_job"].isEnabled()
    assert window.session.validation_for("pull1") is not None
    assert window.session.validation_for("pull2") is not None
    window.close()


@pytest.mark.parametrize(
    ("terminal", "expected_status"),
    [
        ("failed", RunStatus.FAILED),
        ("cancelled", RunStatus.CANCELLED),
    ],
)
def test_failed_or_cancelled_job_preserves_previous_displayed_result(
    terminal: str,
    expected_status: RunStatus,
) -> None:
    window = _new_window()
    _install_imported(window)
    first_run_id = _succeed_run(window, run_name="Job-1")
    first_provider = window.result_provider
    first_selection = window.result_selection
    first_payload = window.viewport._result_render_payload
    assert first_provider is not None
    assert first_selection is not None
    assert first_payload is not None

    task = window.session.prepare_solve("pull", "Job-2")
    assert task.delta is not None
    assert window._apply_session_delta(task.delta)
    assert window._apply_session_delta(window.session.begin_run(task.token))
    if terminal == "failed":
        terminal_delta = window.session.accept_run_failed(
            task.token,
            "solver failed",
        )
    else:
        assert window._apply_session_delta(
            window.session.request_cancel(task.run_id)
        )
        terminal_delta = window.session.accept_run_cancelled(task.token)
    assert window._apply_session_delta(terminal_delta)

    assert window.session.find_run(task.run_id).status is expected_status
    assert window.document.displayed_result_run_id == first_run_id
    assert window.session.current_result().provenance.run_id == first_run_id
    assert window.result_provider is first_provider
    assert window.result_selection == first_selection
    assert window.viewport._result_render_payload is first_payload
    assert window.actions["query"].isEnabled()
    window.close()


def test_stale_import_callback_cannot_change_projection() -> None:
    window = _new_window()
    _install_imported(window)
    stale = window.session.prepare_import("late.inp")
    assert window._apply_session_delta(
        window.session.replace_model_definitions(
            window.document.materials,
            window.document.sections,
            window.document.assignments,
            window.document.steps,
        )
    )
    before = _projection_signature(window)

    rejected = window.session.accept_imported_model(
        stale.token,
        make_static_pull_truss_model(load=999.0),
    )
    assert not rejected.accepted
    assert not window._apply_session_delta(rejected)
    assert _projection_signature(window) == before
    window.close()


def test_stale_mesh_callback_cannot_change_projection() -> None:
    window = _new_window()
    window._create_native_model("Model-1")
    window._set_native_geometry(
        RectangleGeometry("Plate", 2.0, 1.0),
        "矩形",
    )
    stale = window.session.prepare_mesh_generation()
    assert window._apply_session_delta(
        window.session.replace_mesh_settings(MeshSettings(0.25))
    )
    before = _projection_signature(window)

    rejected = window.session.accept_generated_model(
        stale.token,
        make_static_pull_truss_model(),
    )
    assert not rejected.accepted
    assert not window._apply_session_delta(rejected)
    assert _projection_signature(window) == before
    window.close()


def test_stale_validation_callback_cannot_change_projection() -> None:
    window = _new_window()
    _install_imported(window)
    stale = window.session.prepare_validation("pull")
    assert window._apply_session_delta(
        window.session.replace_model_definitions(
            window.document.materials,
            window.document.sections,
            window.document.assignments,
            window.document.steps,
        )
    )
    before = _projection_signature(window)

    rejected = window.session.accept_validation(
        stale.token,
        passing_preflight_report(stale.token),
    )
    assert not rejected.accepted
    assert not window._apply_session_delta(rejected)
    assert _projection_signature(window) == before
    window.close()


def test_stale_solve_callback_cannot_restore_invalidated_result() -> None:
    window = _new_window()
    _install_imported(window)
    _validate_step(window, "pull")
    stale = window.session.prepare_solve("pull", "Late-Job")
    assert stale.delta is not None
    assert window._apply_session_delta(stale.delta)
    assert window._apply_session_delta(window.session.begin_run(stale.token))
    result = solve(stale.model, stale.step_name, name=stale.run_name)
    bundle = build_solve_result_bundle(stale, result)
    assert window._apply_session_delta(
        window.session.replace_model_definitions(
            window.document.materials,
            window.document.sections,
            window.document.assignments,
            window.document.steps,
        )
    )
    before = _projection_signature(window)

    rejected = window.session.accept_run_succeeded(
        stale.token,
        bundle,
    )
    assert not rejected.accepted
    assert not window._apply_session_delta(rejected)
    assert _projection_signature(window) == before
    assert window.session.current_result() is None
    _assert_result_entries_disabled(window)
    window.close()


def test_revision_neutral_projection_receipt_preserves_current_cache() -> None:
    window = _new_window()
    _install_imported(window)
    run_id = _succeed_run(window)
    stale = window.session.prepare_result_projection(run_id)
    assert window._apply_session_delta(window.session.select_result(run_id))
    before = _projection_signature(window)

    receipt = window.session.accept_result_projection(stale.token)
    assert receipt.accepted
    assert window._apply_revision_neutral_task_receipt(receipt)
    assert _projection_signature(window) == before
    assert window.result_provider is not None
    assert window.result_provider.source.run_id == run_id
    window.close()


def test_hidden_run_projection_receipt_cannot_replace_current_cache() -> None:
    window = _new_window()
    _install_imported(window)
    run_a = _succeed_run(window, run_name="Job-A")
    projection = window.session.prepare_result_projection(run_a)
    provider_a = window.result_provider
    assert provider_a is not None and provider_a.source.run_id == run_a
    run_b = _succeed_run(window, run_name="Job-B")
    before = _projection_signature(window)

    receipt = window.session.accept_result_projection(projection.token)

    assert receipt.accepted
    assert window._apply_revision_neutral_task_receipt(receipt)
    assert _projection_signature(window) == before
    assert window.result_provider is not None
    assert window.result_provider.source.run_id == run_b
    window.close()


def test_projection_failure_and_cancel_receipts_are_revision_neutral(
    monkeypatch,
) -> None:
    window = _new_window()
    _install_imported(window)
    run_id = _succeed_run(window)
    failed = window.session.prepare_result_projection(run_id)
    cancelled = window.session.prepare_result_projection(run_id)
    shown: list[tuple[str, str]] = []
    monkeypatch.setattr(
        window,
        "_show_error",
        lambda title, message: shown.append((title, message)),
    )
    revision = window.session.session_revision

    window._session_task_failed(
        failed.token,
        "应力结果恢复失败",
        "recovery failed",
    )
    window._session_task_cancelled(cancelled.token)

    assert shown == [("应力结果恢复失败", "recovery failed")]
    assert window.session.session_revision == revision
    assert (
        window.session.validate_task_token(failed.token)
        is TokenStatus.ALREADY_COMPLETED
    )
    assert (
        window.session.validate_task_token(cancelled.token)
        is TokenStatus.ALREADY_COMPLETED
    )
    window.close()


@pytest.mark.parametrize(
    "failure_case",
    (
        "corrupt-v1-json",
        "tampered-v2-topology",
        "v1-migration",
    ),
)
def test_failed_project_open_preserves_session_tree_and_viewport(
    failure_case,
    tmp_path,
    monkeypatch,
) -> None:
    window = _new_window()
    _install_imported(window)
    _succeed_run(window)
    corrupt = tmp_path / f"{failure_case}.femproj"
    if failure_case == "corrupt-v1-json":
        corrupt.write_text(
            '{"schema": 1, "geometry": ',
            encoding="utf-8",
        )
    elif failure_case == "tampered-v2-topology":
        authoring = ModelSession()
        authoring.new_native_project()
        authoring.replace_geometry(
            authoring.snapshot().parts,
            RectangleGeometry("Plate", 2.0, 1.0),
        )
        save_project(corrupt, authoring.prepare_project_save())
        payload = json.loads(corrupt.read_text(encoding="utf-8"))
        payload["project"]["authoring"]["parts"][0][
            "logical_topology"
        ]["signature"]["entities"][0]["logical_id"] = "point:tampered"
        corrupt.write_text(
            json.dumps(payload, ensure_ascii=False),
            encoding="utf-8",
        )
    else:
        fixture = (
            Path(__file__).parents[1]
            / "fixtures"
            / "femproj"
            / "v1"
            / "line_load_unsupported.femproj"
        )
        corrupt.write_bytes(fixture.read_bytes())
    errors: list[tuple[str, str]] = []
    before_document = window.document
    before = _projection_signature(window)

    monkeypatch.setattr(
        main_window_module.QFileDialog,
        "getOpenFileName",
        lambda *_args, **_kwargs: (str(corrupt), ""),
    )
    monkeypatch.setattr(window, "_confirm_discard_changes", lambda: True)
    monkeypatch.setattr(
        window,
        "_show_error",
        lambda title, message: errors.append((title, message)),
    )

    window.open_native_project()
    _wait_for_task(window)

    assert errors
    assert window.document is before_document
    assert _projection_signature(window) == before
    assert "下次显式保存" not in window.status_panel.state_label.text()
    assert "schema 2" not in window.status_panel.state_label.text()
    assert "compatibility migration" not in (
        window.status_panel.state_label.text()
    )
    assert window.actions["query"].isEnabled()
    window.close()
