from __future__ import annotations

from copy import deepcopy
import os
from pathlib import Path
from threading import Event
from time import monotonic, sleep

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QEventLoop, QThread, QTimer
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import QApplication, QMessageBox

from fem.application import RunStatus
from fem.abaqus import read
from fem.solvers import static_linear
from fem_gui.main_window import FEMMainWindow
import fem_gui.main_window as main_window_module
from fem_gui.visualization.model_adapter import build_model_geometry
from tests.helpers.model_builders import make_static_pull_truss_model


def _application() -> QApplication:
    return QApplication.instance() or QApplication([])


def _wait_for_task(window: FEMMainWindow, timeout: float = 5.0) -> None:
    deadline = monotonic() + timeout
    application = _application()
    while window.busy and monotonic() < deadline:
        application.processEvents()
        QThread.msleep(1)
    application.processEvents()
    assert not window.busy


def _window_with_model() -> FEMMainWindow:
    window = FEMMainWindow()
    model = make_static_pull_truss_model()
    window._model_loaded(
        Path("pull.inp"),
        (model, build_model_geometry(model)),
    )
    validation = window.session.prepare_validation("pull")
    window._apply_session_delta(
        window.session.accept_validation(
            validation.token,
            {"passed": True},
        )
    )
    return window


def _bump_model_revision(window: FEMMainWindow) -> None:
    snapshot = window.document
    window._apply_session_delta(
        window.session.replace_model_definitions(
            snapshot.material_definitions,
            snapshot.section_definitions,
            snapshot.region_assignments,
            snapshot.analysis_definitions,
        )
    )


def test_task_callbacks_run_on_gui_thread_and_cleanup_once() -> None:
    _application()
    window = FEMMainWindow()
    workload_threads: list[bool] = []
    callback_threads: list[bool] = []
    heartbeat = [0]
    timer = QTimer()
    timer.setInterval(5)
    timer.timeout.connect(lambda: heartbeat.__setitem__(0, heartbeat[0] + 1))
    timer.start()

    def workload(context):
        workload_threads.append(QThread.currentThread() is window.thread())
        context.report("测试阶段")
        sleep(0.06)
        return 42

    assert window._start_task(
        workload,
        lambda value: callback_threads.append(
            QThread.currentThread() is window.thread() and value == 42
        ),
        "测试失败",
        task_name="线程测试",
    )
    _wait_for_task(window)
    timer.stop()

    assert workload_threads == [False]
    assert callback_threads == [True]
    assert heartbeat[0] > 0
    assert window._worker is None
    assert window._active_task_id is None
    assert not window.status_panel.cancel_button.isVisible()
    window.close()


def test_duplicate_task_is_rejected_and_callback_failure_recovers(
    monkeypatch,
) -> None:
    _application()
    window = FEMMainWindow()
    gate = Event()
    started = Event()
    shown: list[tuple[str, str]] = []
    monkeypatch.setattr(
        window,
        "_show_error",
        lambda title, message: shown.append((title, message)),
    )

    def workload(context):
        started.set()
        while not gate.is_set():
            context.checkpoint()
            sleep(0.001)
        return "done"

    assert window._start_task(
        workload,
        lambda _value: (_ for _ in ()).throw(RuntimeError("应用结果失败")),
        "任务失败",
        task_name="唯一任务",
    )
    deadline = monotonic() + 2.0
    while not started.is_set() and monotonic() < deadline:
        _application().processEvents()
    assert not window._start_task(
        lambda _context: None,
        lambda _value: None,
        "不应启动",
    )
    gate.set()
    _wait_for_task(window)

    assert shown == [("任务失败", "应用结果失败")]
    assert not window.busy
    window.close()


def test_nested_event_loop_does_not_finalize_active_callback() -> None:
    _application()
    window = FEMMainWindow()
    observations: list[tuple[bool, bool]] = []

    def succeeded(_value):
        nested_loop = QEventLoop()

        def inspect_while_callback_is_active():
            duplicate_started = window._start_task(
                lambda _context: None,
                lambda _result: None,
                "不应启动",
            )
            observations.append((window.busy, duplicate_started))
            nested_loop.quit()

        QTimer.singleShot(20, inspect_while_callback_is_active)
        nested_loop.exec()
        assert window.busy

    assert window._start_task(
        lambda _context: "done",
        succeeded,
        "不应失败",
        task_name="嵌套事件循环任务",
    )
    _wait_for_task(window)

    assert observations == [(True, False)]
    assert not window.busy
    window.close()


def test_model_and_mesh_checks_compute_off_the_gui_thread(
    monkeypatch,
    gui_inp_path,
) -> None:
    _application()
    window = FEMMainWindow()
    model = read(gui_inp_path)
    window._model_loaded(
        gui_inp_path,
        (model, build_model_geometry(model)),
    )
    model_threads: list[bool] = []
    mesh_threads: list[bool] = []
    dialogs: list[str] = []
    original_check = window._evaluate_model_check
    original_mesh = main_window_module.analyze_mesh

    def checked(model, step_name):
        model_threads.append(QThread.currentThread() is window.thread())
        return original_check(model, step_name)

    def analyzed(model):
        mesh_threads.append(QThread.currentThread() is window.thread())
        return original_mesh(model)

    monkeypatch.setattr(window, "_evaluate_model_check", checked)
    monkeypatch.setattr(main_window_module, "analyze_mesh", analyzed)
    monkeypatch.setattr(
        main_window_module,
        "show_information",
        lambda _parent, title, _rows: dialogs.append(title),
    )

    assert window.start_model_check()
    _wait_for_task(window)
    window.show_mesh_quality()
    _wait_for_task(window)

    assert model_threads == [False]
    assert mesh_threads == [False]
    assert dialogs == ["模型检查", "网格质量检查"]
    validation = window.session.validation_for("Static-1")
    assert validation is not None and validation.passed
    window.close()


def test_model_revision_discards_stale_check_results(
    monkeypatch,
    gui_inp_path,
) -> None:
    _application()
    window = FEMMainWindow()
    model = read(gui_inp_path)
    window._model_loaded(
        gui_inp_path,
        (model, build_model_geometry(model)),
    )
    model_check_entered = Event()
    model_check_release = Event()
    mesh_check_entered = Event()
    mesh_check_release = Event()
    dialogs: list[str] = []
    original_model_check = window._evaluate_model_check
    original_mesh_check = main_window_module.analyze_mesh

    def delayed_model_check(model_snapshot, step_name):
        model_check_entered.set()
        model_check_release.wait(2.0)
        return original_model_check(model_snapshot, step_name)

    def delayed_mesh_check(current_model):
        mesh_check_entered.set()
        mesh_check_release.wait(2.0)
        return original_mesh_check(current_model)

    monkeypatch.setattr(window, "_evaluate_model_check", delayed_model_check)
    monkeypatch.setattr(main_window_module, "analyze_mesh", delayed_mesh_check)
    monkeypatch.setattr(
        main_window_module,
        "show_information",
        lambda _parent, title, _rows: dialogs.append(title),
    )

    assert window.start_model_check()
    assert model_check_entered.wait(2.0)
    _bump_model_revision(window)
    model_check_release.set()
    _wait_for_task(window)
    assert window.session.validation_for("Static-1") is None

    window.show_mesh_quality()
    assert mesh_check_entered.wait(2.0)
    _bump_model_revision(window)
    mesh_check_release.set()
    _wait_for_task(window)

    assert dialogs == []
    window.close()


def test_real_mesh_analysis_keeps_qt_event_loop_responsive(
    monkeypatch,
    gui_inp_path,
) -> None:
    _application()
    window = FEMMainWindow()
    model = read(gui_inp_path)
    window._model_loaded(
        gui_inp_path,
        (model, build_model_geometry(model)),
    )
    heartbeat = [0]
    timer = QTimer()
    timer.setInterval(2)
    timer.timeout.connect(lambda: heartbeat.__setitem__(0, heartbeat[0] + 1))
    original_mesh_check = main_window_module.analyze_mesh

    def repeated_real_analysis(current_model):
        deadline = monotonic() + 0.08
        report = original_mesh_check(current_model)
        while monotonic() < deadline:
            report = original_mesh_check(current_model)
        return report

    monkeypatch.setattr(
        main_window_module,
        "analyze_mesh",
        repeated_real_analysis,
    )
    monkeypatch.setattr(
        main_window_module,
        "show_information",
        lambda *_args, **_kwargs: None,
    )
    timer.start()
    window.show_mesh_quality()
    _wait_for_task(window)
    timer.stop()

    assert heartbeat[0] > 0
    window.close()


def test_cancelled_task_discards_success_result() -> None:
    _application()
    window = FEMMainWindow()
    started = Event()
    succeeded: list[object] = []
    cancelled: list[bool] = []

    def workload(context):
        started.set()
        while True:
            context.checkpoint()
            sleep(0.001)

    assert window._start_task(
        workload,
        succeeded.append,
        "不应失败",
        task_name="可取消任务",
        on_cancelled=lambda: cancelled.append(True),
    )
    deadline = monotonic() + 2.0
    while not started.is_set() and monotonic() < deadline:
        _application().processEvents()
    assert window.cancel_current_task()
    _wait_for_task(window)

    assert succeeded == []
    assert cancelled == [True]
    assert not window.status_panel.cancel_button.isVisible()
    window.close()


def test_cancel_request_wins_over_already_queued_success() -> None:
    _application()
    window = FEMMainWindow()
    completed = Event()
    succeeded: list[object] = []
    cancelled: list[bool] = []

    def workload(_context):
        completed.set()
        return "stale result"

    assert window._start_task(
        workload,
        succeeded.append,
        "不应失败",
        task_name="竞态取消任务",
        on_cancelled=lambda: cancelled.append(True),
    )
    assert completed.wait(2.0)
    assert window.cancel_current_task()
    _wait_for_task(window)

    assert succeeded == []
    assert cancelled == [True]
    window.close()


def test_close_during_task_requests_cancel_and_waits(
    monkeypatch,
) -> None:
    _application()
    window = FEMMainWindow()
    started = Event()
    cancelled: list[bool] = []

    class CloseTaskMessageBox:
        Icon = QMessageBox.Icon
        ButtonRole = QMessageBox.ButtonRole

        def __init__(self, *_args, **_kwargs):
            self._cancel_button = None
            self._clicked_button = None

        def addButton(self, _text, role):
            button = object()
            if role is QMessageBox.ButtonRole.AcceptRole:
                self._cancel_button = button
            return button

        def exec(self):
            self._clicked_button = self._cancel_button

        def clickedButton(self):
            return self._clicked_button

    def workload(context):
        started.set()
        while True:
            context.checkpoint()
            sleep(0.001)

    monkeypatch.setattr(
        main_window_module,
        "QMessageBox",
        CloseTaskMessageBox,
    )
    assert window._start_task(
        workload,
        lambda _value: None,
        "不应失败",
        task_name="关闭等待任务",
        on_cancelled=lambda: cancelled.append(True),
    )
    assert started.wait(2.0)

    event = QCloseEvent()
    window.closeEvent(event)
    assert not event.isAccepted()
    assert window._close_after_task_cancel
    _wait_for_task(window)
    _application().processEvents()

    assert cancelled == [True]
    assert not window.busy
    assert not window._close_after_task_cancel


def test_cancelled_analysis_job_has_explicit_terminal_state(monkeypatch) -> None:
    _application()
    window = _window_with_model()
    model = window.document.model
    original_props = [
        deepcopy(element.props)
        for element in model.mesh.elements
    ]
    original_metadata = deepcopy(model.metadata)
    entered_solver = Event()
    release_solver = Event()
    original_solve = static_linear.solve

    def delayed_solve(*args, **kwargs):
        entered_solver.set()
        release_solver.wait(2.0)
        return original_solve(*args, **kwargs)

    monkeypatch.setattr(static_linear, "solve", delayed_solve)
    started = window._submit_job("Job-1", "pull")
    assert started is not None
    deadline = monotonic() + 2.0
    while not entered_solver.is_set() and monotonic() < deadline:
        _application().processEvents()
    assert window.cancel_current_task()
    release_solver.set()
    _wait_for_task(window)

    job = window.session.find_run(started.run_id)
    assert job is not None
    assert job.status is RunStatus.CANCELLED
    assert not job.has_result
    assert job.error is None
    assert window.actions["resubmit_job"].isEnabled()
    assert [
        element.props
        for element in model.mesh.elements
    ] == original_props
    assert model.metadata == original_metadata
    window.close()
