from __future__ import annotations

import os
from datetime import datetime, timezone

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from fem.application import AnalysisRun, RunDiagnostics, RunStatus
from fem_gui.analysis_dialogs import JobMonitorDialog


def _application() -> QApplication:
    return QApplication.instance() or QApplication([])


def test_job_monitor_uses_increment_table_and_abaqus_style_message_tabs():
    _application()
    job = AnalysisRun(
        run_id="run-1",
        name="Job-1",
        step_name="Step-1",
        artifact_id="artifact-1",
        model_revision=3,
        started_at=datetime.now(timezone.utc),
    )
    monitor = RunDiagnostics(
        job.run_id,
        job.name,
        job.step_name,
        analysis_type="完整分析",
        procedure="Static, General",
        nlgeom=True,
    )
    monitor.start()
    monitor.increment_started(1, 1, 0.5, 0.0)
    monitor.newton_iteration(1, 1, 1, 1.0, None)
    monitor.increment_converged(1, 1, 0.5, 1, 1.0e-10)
    dialog = JobMonitorDialog(job, monitor.snapshot())

    assert dialog.table.columnCount() == 10
    assert dialog.table.rowCount() == 1
    assert dialog.table.item(0, 1).text() == "1"
    assert dialog.table.item(0, 2).text() == "1"
    assert dialog.table.item(0, 8).text() == "已收敛"
    assert [dialog.tabs.tabText(index) for index in range(dialog.tabs.count())] == [
        "日志",
        "错误",
        "警告",
        "输出",
        "数据文件",
        "消息文件",
        "状态文件",
    ]
    assert "已收敛增量：1" in dialog._tab_views["output"].toPlainText()
    dialog.close()


def test_job_monitor_log_refresh_preserves_manual_scroll_position():
    _application()
    job = AnalysisRun(
        run_id="run-1",
        name="Job-1",
        step_name="Step-1",
        artifact_id="artifact-1",
        model_revision=1,
        started_at=datetime.now(timezone.utc),
    )
    monitor = RunDiagnostics(job.run_id, job.name, job.step_name)
    monitor.start()
    for index in range(40):
        monitor.stage_changed(f"求解阶段 {index}")
    dialog = JobMonitorDialog(job, monitor.snapshot())
    log_view = dialog._tab_views["log"]
    scroll_bar = log_view.verticalScrollBar()
    assert scroll_bar.maximum() > 0
    scroll_bar.setValue(scroll_bar.maximum() // 2)
    old_value = scroll_bar.value()

    monitor.stage_changed("新增求解消息")
    dialog.refresh(snapshot=monitor.snapshot())

    assert scroll_bar.value() == min(old_value, scroll_bar.maximum())
    dialog.close()


def test_job_monitor_shows_dynamic_time_step_and_energy_columns():
    _application()
    job = AnalysisRun(
        run_id="run-dynamic",
        name="Dynamic-1",
        step_name="Dynamic-Step",
        artifact_id="artifact-1",
        model_revision=1,
        started_at=datetime.now(timezone.utc),
    )
    monitor = RunDiagnostics(
        job.run_id,
        job.name,
        job.step_name,
        analysis_type="瞬态动力学",
        procedure="Dynamic, Explicit",
    )
    monitor.start()
    monitor.dynamic_increment_started(
        1,
        0.01,
        0.01,
        solver_kind="dynamic_explicit",
        stable_time_increment=0.02,
    )
    monitor.dynamic_increment_converged(
        1,
        0.01,
        0.01,
        0.0,
        solver_kind="dynamic_explicit",
        stable_time_increment=0.02,
        kinetic_energy=1.0,
        internal_energy=2.0,
        total_energy=3.0,
    )
    dialog = JobMonitorDialog(job, monitor.snapshot())

    assert dialog.table.columnCount() == 13
    assert dialog.table.rowCount() == 1
    assert dialog.table.item(0, 3).text() == "1.000e-02"
    assert dialog.table.item(0, 5).text() == "2.000e-02"
    assert dialog.table.item(0, 10).text() == "3.000e+00"
    assert dialog.table.item(0, 11).text() == "已完成"
    assert "稳定时间步" in dialog._tab_views["output"].toPlainText()
    dialog.close()


def test_job_monitor_exposes_lifecycle_and_increment_actions():
    _application()
    job = AnalysisRun(
        run_id="run-actions",
        name="Job-actions",
        step_name="Step-1",
        artifact_id="artifact-1",
        model_revision=1,
        status=RunStatus.SUCCEEDED,
        result_id="result-1",
        started_at=datetime.now(timezone.utc),
    )
    monitor = RunDiagnostics(job.run_id, job.name, job.step_name)
    monitor.start()
    monitor.increment_started(3, 1, 0.5, 0.0)
    monitor.increment_converged(3, 1, 0.5, 1, 1.0e-10)
    dialog = JobMonitorDialog(job, monitor.snapshot())
    emitted: list[tuple[str, int]] = []
    dialog.resultFrameRequested.connect(lambda name, increment: emitted.append((name, increment)))

    assert not dialog.cancel_button.isEnabled()
    assert dialog.rerun_button.isEnabled()
    assert dialog.open_result_button.isEnabled()
    dialog.table.selectRow(0)
    assert dialog.jump_result_button.isEnabled()
    dialog.jump_result_button.click()
    assert emitted == [("Job-actions", 3)]
    dialog.close()
