from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Event
from time import monotonic, sleep

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import QThread
from PySide6.QtWidgets import QApplication, QToolButton

from fem.application import AnalysisRun, ModelSession, RunStatus
from fem.abaqus import read
from fem.solvers import static_linear
from fem_gui.analysis_dialogs import JobManagerDialog
from fem_gui.main_window import FEMMainWindow
import fem_gui.main_window as main_window_module
from fem_gui.visualization.model_adapter import build_model_geometry
from tests.helpers.model_builders import make_static_pull_truss_model
from tests.helpers.preflight_builders import passing_preflight_report


def _application() -> QApplication:
    return QApplication.instance() or QApplication([])


def _wait_for_task(window: FEMMainWindow) -> None:
    controller = window.task_controller
    assert controller.busy
    deadline = monotonic() + 10.0
    application = QApplication.instance()
    while controller.busy and monotonic() < deadline:
        application.processEvents()
        QThread.msleep(1)
    application.processEvents()
    assert not controller.busy


def _validated_session() -> ModelSession:
    session = ModelSession()
    model = make_static_pull_truss_model()
    imported = session.prepare_import(Path("pull.inp"))
    session.accept_imported_model(imported.token, model)
    validation = session.prepare_validation("pull")
    session.accept_validation(
        validation.token,
        passing_preflight_report(validation.token),
    )
    return session


def _accept_validation(window: FEMMainWindow, step_name: str) -> None:
    task = window.session.prepare_validation(step_name)
    window._apply_session_delta(
        window.session.accept_validation(
            task.token,
            passing_preflight_report(task.token),
        )
    )


def test_analysis_job_timestamps_elapsed_and_result_state():
    started_at = datetime.now(timezone.utc) - timedelta(seconds=1.0)
    job = AnalysisRun(
        run_id="run-1",
        name="Job-1",
        step_name="Static-1",
        artifact_id="artifact-1",
        model_revision=1,
        status=RunStatus.RUNNING,
        started_at=started_at,
        messages=("开始线性静力分析",),
    )
    assert job.messages == ("开始线性静力分析",)
    assert job.elapsed_seconds is not None and job.elapsed_seconds >= 1.0
    assert not job.has_result

    completed = replace(
        job,
        status=RunStatus.SUCCEEDED,
        finished_at=started_at + timedelta(seconds=2.5),
        result_id="result-1",
    )
    assert completed.elapsed_seconds == pytest.approx(2.5)
    assert completed.has_result
    assert not replace(completed, status=RunStatus.FAILED).has_result


def test_session_runs_are_case_insensitive_and_cleared_by_model_transitions():
    session = _validated_session()
    first = session.prepare_solve("pull", "Job-1")
    assert session.next_run_name() == "Job-2"
    found = session.find_run("job-1")
    assert found is not None
    assert found.run_id == first.run_id
    with pytest.raises(ValueError):
        session.prepare_solve("pull", "JOB-1")

    replacement = session.prepare_import(Path("replacement.inp"))
    session.accept_imported_model(
        replacement.token,
        make_static_pull_truss_model(),
    )
    assert session.snapshot().runs == ()
    assert session.snapshot().active_job_name is None

    validation = session.prepare_validation("pull")
    session.accept_validation(
        validation.token,
        passing_preflight_report(validation.token),
    )
    session.prepare_solve("pull", "Job-1")
    session.close()
    assert session.snapshot().runs == ()
    assert session.snapshot().active_job_name is None


def test_job_actions_replace_direct_run(gui_inp_path):
    _application()
    window = FEMMainWindow()
    assert "run" not in window.actions
    assert "select_step" not in window.actions
    for name in ("step_info", "check_model", "submit_job", "resubmit_job", "job_manager"):
        assert name in window.actions
    ribbon_actions = {
        button.defaultAction() for button in window.ribbon.findChildren(QToolButton)
        if button.defaultAction() is not None
    }
    assert window.actions["submit_job"] in ribbon_actions
    model = read(gui_inp_path)
    window._model_loaded(gui_inp_path, (model, build_model_geometry(model)))
    assert window.actions["step_info"].isEnabled()
    assert window.actions["check_model"].isEnabled()
    assert not window.actions["submit_job"].isEnabled()
    assert window.check_current_model(show_success=False)
    assert window.actions["submit_job"].isEnabled()
    window.close()


def test_current_step_information_and_model_check_reuse_existing_services(monkeypatch, gui_inp_path):
    _application()
    window = FEMMainWindow()
    model = read(gui_inp_path)
    window._model_loaded(gui_inp_path, (model, build_model_geometry(model)))
    calls: list[tuple[str, object]] = []
    monkeypatch.setattr(
        window,
        "show_entity_information",
        lambda kind, key: calls.append((kind, key)),
    )
    window.show_current_step_information()
    expected_index = next(
        index for index, step in enumerate(model.steps) if step.name == "Static-1"
    )
    assert calls == [("step", expected_index)]
    reported: list[tuple[str, list[tuple[str, object]]]] = []
    monkeypatch.setattr(
        main_window_module,
        "show_information",
        lambda _parent, title, rows: reported.append((title, list(rows))),
    )
    assert window.check_current_model()
    assert reported[0][0] == "模型检查"
    assert ("检查结果", "通过") in reported[0][1]
    window.close()


def test_model_check_runs_the_shared_numerical_stiffness_preflight(
    monkeypatch,
    gui_inp_path,
):
    _application()
    window = FEMMainWindow()
    model = read(gui_inp_path)
    window._model_loaded(gui_inp_path, (model, build_model_geometry(model)))
    original = static_linear.validate_stiffness
    calls: list[str] = []

    def tracked(model, step):
        calls.append(str(step.name))
        return original(model, step)

    monkeypatch.setattr(
        static_linear,
        "validate_stiffness",
        tracked,
    )

    assert window.check_current_model(show_success=False)
    assert calls == ["Static-1"]
    window.close()


def test_submit_resubmit_open_history_and_reload_clear(gui_inp_path):
    _application()
    window = FEMMainWindow()
    model = read(gui_inp_path)
    geometry = build_model_geometry(model)
    window._model_loaded(gui_inp_path, (model, geometry))
    assert window.check_current_model(show_success=False)

    started1 = window._submit_job("Job-1", "Static-1")
    assert started1 is not None
    assert started1.status is RunStatus.RUNNING
    _wait_for_task(window)
    job1 = window.session.find_run(started1.run_id)
    assert job1 is not None and job1.has_result
    assert window.document.active_job_name is None
    assert window.document.displayed_result_run_id == job1.run_id
    assert window.result_tree.topLevelItem(0).child(0).text(0) == "Job-1 · Static-1"

    previous = window.session.current_result()
    assert previous is not None
    started2 = window._submit_job(
        "Job-2",
        "Static-1",
        source_job_name="Job-1",
    )
    assert started2 is not None
    _wait_for_task(window)
    job2 = window.session.find_run(started2.run_id)
    assert job2 is not None and job2.has_result

    window.open_job_result("Job-1")
    selected = window.session.current_result()
    assert selected is not None
    assert selected.provenance.run_id == job1.run_id
    assert window.document.displayed_result_run_id == job1.run_id
    window.reload_model()
    _wait_for_task(window)
    assert window.document.runs == ()
    assert window.document.active_job_name is None
    window.close()


def test_job_completes_with_primary_results_and_recovers_stress_on_demand(
    monkeypatch,
):
    _application()
    window = FEMMainWindow()
    errors: list[tuple[str, str]] = []
    monkeypatch.setattr(
        window,
        "_show_error",
        lambda title, message: errors.append((title, message)),
    )
    model = make_static_pull_truss_model()
    original_metadata = deepcopy(model.metadata)
    original_props = [
        deepcopy(element.props)
        for element in model.mesh.elements
    ]
    geometry = build_model_geometry(model)
    window._model_loaded(Path("pull.inp"), (model, geometry))
    _accept_validation(window, "pull")
    validations = []
    original_validate = static_linear.validate_problem

    def counted_validate(*args, **kwargs):
        validations.append(QThread.currentThread() is window.thread())
        return original_validate(*args, **kwargs)

    monkeypatch.setattr(static_linear, "validate_problem", counted_validate)

    started = window._submit_job("Job-1", "pull")
    assert started is not None
    _wait_for_task(window)

    job = window.session.find_run(started.run_id)
    record = window.session.current_result()
    data = window.result_data
    assert job is not None
    assert record is not None
    assert data is not None
    assert validations == [False]
    assert record.result.model is not model
    assert model.metadata == original_metadata
    assert [
        element.props
        for element in model.mesh.elements
    ] == original_props
    assert data.field_ready("U")
    assert not data.field_ready("CENTROID:Mises")
    assert "模型验证" in job.timings
    assert "线性方程求解" in job.timings
    assert "输出请求与初始结果" in job.timings

    revision = window.session.session_revision
    window._activate_result_field("CENTROID:Mises")
    _wait_for_task(window)
    assert window.session.session_revision == revision
    assert not data.field_ready("CENTROID:Mises")
    assert window.result_data is not data
    assert window.result_data is not None
    assert window.result_data.field_ready("CENTROID:Mises")
    assert window._display.field_key == "CENTROID:Mises"
    assert errors == []
    window.close()


def test_failed_job_keeps_previous_result(monkeypatch, gui_inp_path):
    _application()
    window = FEMMainWindow()
    model = read(gui_inp_path)
    geometry = build_model_geometry(model)
    window._model_loaded(gui_inp_path, (model, geometry))
    assert window.check_current_model(show_success=False)
    successful = window._submit_job("Job-1", "Static-1")
    assert successful is not None
    _wait_for_task(window)
    previous_result = window.session.current_result()
    assert previous_result is not None
    shown: list[tuple[str, str]] = []
    monkeypatch.setattr(window, "_show_error", lambda title, message: shown.append((title, message)))
    monkeypatch.setattr(
        static_linear, "solve",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("求解故障")),
    )

    failed_started = window._submit_job("Job-2", "Static-1")
    assert failed_started is not None
    _wait_for_task(window)
    failed = window.session.find_run(failed_started.run_id)
    assert failed is not None
    assert failed.status is RunStatus.FAILED
    assert failed.error == "求解故障"
    assert not failed.has_result
    current_result = window.session.current_result()
    assert current_result is not None
    assert current_result.provenance.run_id == previous_result.provenance.run_id
    assert (
        window.document.displayed_result_run_id
        == previous_result.provenance.run_id
    )
    assert shown == [("分析运行失败", "求解故障")]
    window.close()


def test_base_result_provider_failure_marks_run_failed_and_preserves_display(
    monkeypatch,
    gui_inp_path,
):
    _application()
    window = FEMMainWindow()
    model = read(gui_inp_path)
    window._model_loaded(
        gui_inp_path,
        (model, build_model_geometry(model)),
    )
    assert window.check_current_model(show_success=False)
    successful = window._submit_job("Job-1", "Static-1")
    assert successful is not None
    _wait_for_task(window)
    previous = window.session.current_result()
    shown: list[tuple[str, str]] = []
    monkeypatch.setattr(
        window,
        "_show_error",
        lambda title, message: shown.append((title, message)),
    )
    monkeypatch.setattr(
        main_window_module,
        "build_solve_result_bundle",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("结果基座故障")
        ),
    )

    started = window._submit_job("Job-2", "Static-1")
    assert started is not None
    _wait_for_task(window)

    failed = window.session.find_run(started.run_id)
    current = window.session.current_result()
    assert failed.status is RunStatus.FAILED
    assert failed.error == "结果基座故障"
    assert not failed.has_result
    assert current.provenance.run_id == previous.provenance.run_id
    assert window.document.displayed_result_run_id == previous.provenance.run_id
    assert shown == [("分析运行失败", "结果基座故障")]
    window.close()


def test_cancellation_during_output_execution_discards_bundle(
    monkeypatch,
):
    _application()
    window = FEMMainWindow()
    model = make_static_pull_truss_model()
    window._model_loaded(
        Path("pull.inp"),
        (model, build_model_geometry(model)),
    )
    _accept_validation(window, "pull")
    entered_output = Event()
    original_build = main_window_module.build_solve_result_bundle

    def wait_for_cancellation(
        task,
        result,
        *,
        cancellation,
    ):
        entered_output.set()
        while not cancellation.is_cancelled:
            sleep(0.001)
        cancellation.checkpoint()
        return original_build(
            task,
            result,
            cancellation=cancellation,
        )

    monkeypatch.setattr(
        main_window_module,
        "build_solve_result_bundle",
        wait_for_cancellation,
    )

    started = window._submit_job("Job-1", "pull")
    assert started is not None
    application = QApplication.instance()
    deadline = monotonic() + 10.0
    while not entered_output.is_set() and monotonic() < deadline:
        application.processEvents()
        QThread.msleep(1)
    assert entered_output.is_set()
    assert window.cancel_current_task()
    _wait_for_task(window)

    cancelled = window.session.find_run(started.run_id)
    assert cancelled.status is RunStatus.CANCELLED
    assert cancelled.cancellation_requested
    assert not cancelled.has_result
    assert window.session.current_result() is None
    window.close()


def test_solver_defensive_validation_failure_is_reported_by_job(monkeypatch):
    _application()
    window = FEMMainWindow()
    model = make_static_pull_truss_model()
    window._model_loaded(Path("pull.inp"), (model, build_model_geometry(model)))
    _accept_validation(window, "pull")
    monkeypatch.setattr(
        static_linear, "validate_problem",
        lambda *_args: (_ for _ in ()).throw(ValueError("模型引用错误")),
    )
    shown: list[tuple[str, str]] = []
    monkeypatch.setattr(window, "_show_error", lambda title, message: shown.append((title, message)))

    started = window._submit_job("Job-1", "pull")
    assert started is not None
    _wait_for_task(window)
    job = window.session.find_run(started.run_id)
    assert job is not None
    assert tuple(run.run_id for run in window.document.runs) == (job.run_id,)
    assert job.status is RunStatus.FAILED
    assert job.error == "模型引用错误"
    assert shown == [("分析运行失败", "模型引用错误")]
    window.close()


def test_submit_rejects_busy_empty_and_duplicate_names(monkeypatch, gui_inp_path):
    _application()
    window = FEMMainWindow()
    model = read(gui_inp_path)
    window._model_loaded(gui_inp_path, (model, build_model_geometry(model)))
    assert window.check_current_model(show_success=False)
    shown: list[tuple[str, str]] = []
    monkeypatch.setattr(window, "_show_error", lambda title, message: shown.append((title, message)))
    assert window._submit_job("   ", "Static-1") is None
    assert window.document.runs == ()

    first = window._submit_job("Job-1", "Static-1")
    assert first is not None
    assert window._submit_job("Job-2", "Static-1") is None
    _wait_for_task(window)
    assert window._submit_job("job-1", "Static-1") is None
    assert any("作业名称不能为空" in message for _title, message in shown)
    assert any("作业名称已存在" in message for _title, message in shown)
    window.close()


def test_job_workflow_creates_no_job_files(monkeypatch, tmp_path, gui_inp_path):
    _application()
    monkeypatch.chdir(tmp_path)
    window = FEMMainWindow()
    model = read(gui_inp_path)
    window._model_loaded(gui_inp_path, (model, build_model_geometry(model)))
    assert window.check_current_model(show_success=False)
    assert window._submit_job("Job-1", "Static-1") is not None
    _wait_for_task(window)
    for name in ("jobs", "job.json", "solver.log", "result.npz"):
        assert not (tmp_path / name).exists()
    window.close_model()
    assert window.document.runs == ()
    window.close()


def test_job_manager_shows_memory_log_and_history_actions(gui_inp_path):
    _application()
    window = FEMMainWindow()
    model = read(gui_inp_path)
    window._model_loaded(gui_inp_path, (model, build_model_geometry(model)))
    assert window.check_current_model(show_success=False)
    started = window._submit_job("Job-1", "Static-1")
    assert started is not None
    _wait_for_task(window)

    manager = window.show_job_manager()
    assert manager is not None
    assert manager.table.rowCount() == 1
    assert manager.table.item(0, 2).text() == "已完成"
    assert manager.resubmit_button.isEnabled()
    assert manager.open_result_button.isEnabled()
    assert window.show_job_manager() is manager
    manager.close()
    window.close()


def test_job_manager_refresh_preserves_manual_log_scroll_position():
    _application()
    job = AnalysisRun(
        run_id="run-1",
        name="Job-1",
        step_name="Static-1",
        artifact_id="artifact-1",
        model_revision=1,
        status=RunStatus.RUNNING,
        started_at=datetime.now(timezone.utc),
        messages=tuple(f"日志行 {index}" for index in range(100)),
    )
    manager = JobManagerDialog([job])
    manager.show()
    QApplication.processEvents()

    scroll_bar = manager.log_view.verticalScrollBar()
    assert scroll_bar.maximum() > 0
    scroll_bar.setValue(scroll_bar.maximum() // 2)
    manual_position = scroll_bar.value()

    manager.refresh()
    assert scroll_bar.value() == manual_position

    job = replace(job, messages=job.messages + ("新增日志",))
    manager.refresh([job])
    assert scroll_bar.value() == manual_position

    scroll_bar.setValue(scroll_bar.maximum())
    job = replace(job, messages=job.messages + ("继续新增日志",))
    manager.refresh([job])
    assert scroll_bar.value() == scroll_bar.maximum()
    manager.close()
