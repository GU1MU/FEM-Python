from __future__ import annotations

from copy import deepcopy
import os
from datetime import datetime, timedelta
from pathlib import Path
from time import monotonic

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import QThread
from PySide6.QtWidgets import QApplication, QToolButton

from fem.abaqus import read
from fem.solvers import static_linear
from fem_gui.analysis_dialogs import JobManagerDialog
from fem_gui.analysis_jobs import AnalysisJob, JobStatus
from fem_gui.document import FEMDocument
from fem_gui.main_window import FEMMainWindow
import fem_gui.main_window as main_window_module
from fem_gui.visualization.model_adapter import build_model_geometry
from tests.helpers.model_builders import make_static_pull_truss_model


def _application() -> QApplication:
    return QApplication.instance() or QApplication([])


def _wait_for_task(window: FEMMainWindow) -> None:
    assert window._thread is not None
    deadline = monotonic() + 10.0
    application = QApplication.instance()
    while window.busy and monotonic() < deadline:
        application.processEvents()
        QThread.msleep(1)
    application.processEvents()
    assert not window.busy


def test_analysis_job_timestamps_elapsed_and_result_state():
    job = AnalysisJob("Job-1", "Static-1", JobStatus.RUNNING)
    job.started_at = datetime.now() - timedelta(seconds=1.0)
    job.add_message("开始线性静力分析")
    assert job.messages[0].count(":") == 2
    assert job.elapsed_seconds is not None and job.elapsed_seconds >= 1.0
    assert not job.has_result

    job.status = JobStatus.COMPLETED
    job.finished_at = job.started_at + timedelta(seconds=2.5)
    job.model_result = object()
    job.result_data = object()
    assert job.elapsed_seconds == pytest.approx(2.5)
    assert job.has_result
    job.status = JobStatus.FAILED
    assert not job.has_result


def test_document_jobs_are_case_insensitive_and_cleared():
    document = FEMDocument()
    first = AnalysisJob("Job-1", "pull", JobStatus.COMPLETED)
    document.add_job(first)
    assert document.next_job_name() == "Job-2"
    assert document.find_job("job-1") is first
    with pytest.raises(ValueError):
        document.add_job(AnalysisJob("JOB-1", "pull", JobStatus.FAILED))

    document.set_model("model.inp", make_static_pull_truss_model())
    assert document.jobs == []
    assert document.active_job_name is None
    document.add_job(AnalysisJob("Job-1", "pull", JobStatus.COMPLETED))
    document.active_job_name = "Job-1"
    document.close()
    assert document.jobs == []
    assert document.active_job_name is None


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


def test_model_check_does_not_assemble_and_factor_the_stiffness(monkeypatch, gui_inp_path):
    _application()
    window = FEMMainWindow()
    model = read(gui_inp_path)
    window._model_loaded(gui_inp_path, (model, build_model_geometry(model)))
    monkeypatch.setattr(
        static_linear,
        "validate_stiffness",
        lambda *_args, **_kwargs: pytest.fail(
            "GUI model check must not perform the full numerical solve preflight"
        ),
    )

    assert window.check_current_model(show_success=False)
    window.close()


def test_submit_resubmit_open_history_and_reload_clear(gui_inp_path):
    _application()
    window = FEMMainWindow()
    model = read(gui_inp_path)
    geometry = build_model_geometry(model)
    window._model_loaded(gui_inp_path, (model, geometry))

    job1 = window._submit_job("Job-1", "Static-1")
    assert job1 is not None and job1.status is JobStatus.RUNNING
    _wait_for_task(window)
    assert job1.has_result
    assert window.document.active_job_name == "Job-1"
    assert window.result_tree.topLevelItem(0).child(0).text(0) == "Job-1 · Static-1"

    previous = job1.model_result
    job2 = window._submit_job("Job-2", "Static-1", source_job_name="Job-1")
    assert job2 is not None and job2.source_job_name == "Job-1"
    _wait_for_task(window)
    assert job2.has_result
    assert job1.model_result is previous

    window.open_job_result("Job-1")
    assert window.document.active_job_name == "Job-1"
    assert window.document.result is job1.model_result
    window.reload_model()
    _wait_for_task(window)
    assert window.document.jobs == []
    assert window.document.active_job_name is None
    window.close()


def test_job_completes_with_primary_results_and_recovers_stress_on_demand(
    monkeypatch,
):
    _application()
    window = FEMMainWindow()
    model = make_static_pull_truss_model()
    original_metadata = deepcopy(model.metadata)
    original_props = [
        deepcopy(element.props)
        for element in model.mesh.elements
    ]
    geometry = build_model_geometry(model)
    window._model_loaded(Path("pull.inp"), (model, geometry))
    validations = []
    original_validate = static_linear.validate_problem

    def counted_validate(*args, **kwargs):
        validations.append(QThread.currentThread() is window.thread())
        return original_validate(*args, **kwargs)

    monkeypatch.setattr(static_linear, "validate_problem", counted_validate)

    job = window._submit_job("Job-1", "pull")
    assert job is not None
    _wait_for_task(window)

    data = job.result_data
    assert validations == [False]
    assert job.model_result.model is not model
    assert model.metadata == original_metadata
    assert [
        element.props
        for element in model.mesh.elements
    ] == original_props
    assert data.field_ready("U")
    assert not data.field_ready("CENTROID:Mises")
    assert "模型验证" in job.timings
    assert "线性方程求解" in job.timings
    assert "位移与反力结果" in job.timings

    window._activate_result_field("CENTROID:Mises")
    _wait_for_task(window)
    assert not data.field_ready("CENTROID:Mises")
    assert job.result_data is window.result_data
    assert job.result_data.field_ready("CENTROID:Mises")
    assert window._display.field_key == "CENTROID:Mises"
    assert "应力恢复" in job.timings
    window.close()


def test_failed_job_keeps_previous_result(monkeypatch, gui_inp_path):
    _application()
    window = FEMMainWindow()
    model = read(gui_inp_path)
    geometry = build_model_geometry(model)
    window._model_loaded(gui_inp_path, (model, geometry))
    successful = window._submit_job("Job-1", "Static-1")
    assert successful is not None
    _wait_for_task(window)
    previous_result = window.document.result
    shown: list[tuple[str, str]] = []
    monkeypatch.setattr(window, "_show_error", lambda title, message: shown.append((title, message)))
    monkeypatch.setattr(
        static_linear, "solve",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("求解故障")),
    )

    failed = window._submit_job("Job-2", "Static-1")
    assert failed is not None
    _wait_for_task(window)
    assert failed.status is JobStatus.FAILED
    assert failed.error == "求解故障"
    assert not failed.has_result
    assert window.document.result is previous_result
    assert shown == [("分析运行失败", "求解故障")]
    window.close()


def test_check_failure_is_reported_by_background_job(monkeypatch):
    _application()
    window = FEMMainWindow()
    model = make_static_pull_truss_model()
    window._model_loaded(Path("pull.inp"), (model, build_model_geometry(model)))
    monkeypatch.setattr(
        static_linear, "validate_problem",
        lambda *_args: (_ for _ in ()).throw(ValueError("模型引用错误")),
    )
    shown: list[tuple[str, str]] = []
    monkeypatch.setattr(window, "_show_error", lambda title, message: shown.append((title, message)))

    job = window._submit_job("Job-1", "pull")
    assert job is not None
    _wait_for_task(window)
    assert window.document.jobs == [job]
    assert job.status is JobStatus.FAILED
    assert job.error == "模型引用错误"
    assert shown == [("模型检查失败", "模型引用错误")]
    window.close()


def test_submit_rejects_busy_empty_and_duplicate_names(monkeypatch, gui_inp_path):
    _application()
    window = FEMMainWindow()
    model = read(gui_inp_path)
    window._model_loaded(gui_inp_path, (model, build_model_geometry(model)))
    shown: list[tuple[str, str]] = []
    monkeypatch.setattr(window, "_show_error", lambda title, message: shown.append((title, message)))
    assert window._submit_job("   ", "Static-1") is None
    assert window.document.jobs == []

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
    assert window._submit_job("Job-1", "Static-1") is not None
    _wait_for_task(window)
    for name in ("jobs", "job.json", "solver.log", "result.npz"):
        assert not (tmp_path / name).exists()
    window.close_model()
    assert window.document.jobs == []
    window.close()


def test_job_manager_shows_memory_log_and_history_actions(gui_inp_path):
    _application()
    window = FEMMainWindow()
    model = read(gui_inp_path)
    window._model_loaded(gui_inp_path, (model, build_model_geometry(model)))
    job = window._submit_job("Job-1", "Static-1")
    assert job is not None
    _wait_for_task(window)

    manager = window.show_job_manager()
    assert manager is not None
    assert manager.table.rowCount() == 1
    assert "线性静力分析完成" in manager.log_view.toPlainText()
    assert manager.resubmit_button.isEnabled()
    assert manager.open_result_button.isEnabled()
    assert window.show_job_manager() is manager
    manager.close()
    window.close()


def test_job_manager_refresh_preserves_manual_log_scroll_position():
    _application()
    job = AnalysisJob("Job-1", "Static-1", JobStatus.RUNNING)
    job.messages = [f"日志行 {index}" for index in range(100)]
    manager = JobManagerDialog([job])
    manager.show()
    QApplication.processEvents()

    scroll_bar = manager.log_view.verticalScrollBar()
    assert scroll_bar.maximum() > 0
    scroll_bar.setValue(scroll_bar.maximum() // 2)
    manual_position = scroll_bar.value()

    manager.refresh()
    assert scroll_bar.value() == manual_position

    job.messages.append("新增日志")
    manager.refresh()
    assert scroll_bar.value() == manual_position

    scroll_bar.setValue(scroll_bar.maximum())
    job.messages.append("继续新增日志")
    manager.refresh()
    assert scroll_bar.value() == scroll_bar.maximum()
    manager.close()
