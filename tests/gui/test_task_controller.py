from __future__ import annotations

from collections.abc import Callable
import os
from time import monotonic

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QThread
from PySide6.QtWidgets import QApplication

from fem_gui.task_controller import (
    BackgroundTaskController,
    BackgroundTaskState,
    TaskApplyOutcome,
    TaskApplyStatus,
)


class _Signal:
    def __init__(self) -> None:
        self._slots: list[Callable[..., object]] = []

    def connect(self, slot: Callable[..., object]) -> None:
        self._slots.append(slot)

    def emit(self, *args: object) -> None:
        for slot in tuple(self._slots):
            slot(*args)


class _Thread:
    def __init__(self, _parent=None) -> None:
        self.started = _Signal()
        self.finished = _Signal()
        self.start_count = 0
        self.quit_count = 0
        self.delete_count = 0

    def start(self) -> None:
        self.start_count += 1

    def quit(self, *_args: object) -> None:
        self.quit_count += 1

    def deleteLater(self, *_args: object) -> None:
        self.delete_count += 1


class _Worker:
    def __init__(self, task_id: int, workload) -> None:
        self.task_id = task_id
        self.workload = workload
        self.progress = _Signal()
        self.succeeded = _Signal()
        self.failed = _Signal()
        self.cancelled = _Signal()
        self.finished = _Signal()
        self.thread = None
        self.run_count = 0
        self.cancel_count = 0
        self.delete_count = 0

    def moveToThread(self, thread: _Thread) -> None:
        self.thread = thread

    def run(self) -> None:
        self.run_count += 1

    def request_cancel(self) -> None:
        self.cancel_count += 1

    def deleteLater(self, *_args: object) -> None:
        self.delete_count += 1


class _Harness:
    def __init__(self) -> None:
        self.threads: list[_Thread] = []
        self.workers: list[_Worker] = []
        self.controller = BackgroundTaskController(
            thread_factory=self._thread_factory,
            worker_factory=self._worker_factory,
        )

    def _thread_factory(self, parent) -> _Thread:
        thread = _Thread(parent)
        self.threads.append(thread)
        return thread

    def _worker_factory(self, task_id: int, workload) -> _Worker:
        worker = _Worker(task_id, workload)
        self.workers.append(worker)
        return worker

    @property
    def thread(self) -> _Thread:
        return self.threads[-1]

    @property
    def worker(self) -> _Worker:
        return self.workers[-1]

    def finish_thread(self) -> None:
        self.worker.finished.emit(self.worker.task_id)
        self.thread.finished.emit()


def _start(
    harness: _Harness,
    *,
    apply_result=lambda value: TaskApplyOutcome.accepted(value),
    project_result=None,
    rebuild_projection=None,
    on_terminal=None,
    on_progress=None,
    on_projection_error=None,
) -> int:
    task_id = harness.controller.start(
        lambda _context: None,
        task_name="Focused task",
        apply_result=apply_result,
        project_result=project_result,
        rebuild_projection=rebuild_projection,
        on_terminal=on_terminal,
        on_progress=on_progress,
        on_projection_error=on_projection_error,
    )
    assert task_id is not None
    return task_id


def test_real_worker_boundary_runs_off_controller_thread() -> None:
    application = QApplication.instance() or QApplication([])
    controller = BackgroundTaskController()
    workload_threads: list[bool] = []
    projection_threads: list[bool] = []
    terminals = []

    def workload(_context) -> int:
        workload_threads.append(QThread.currentThread() is controller.thread())
        return 42

    task_id = controller.start(
        workload,
        task_name="Real QThread task",
        apply_result=TaskApplyOutcome.accepted,
        project_result=lambda value: projection_threads.append(
            QThread.currentThread() is controller.thread() and value == 42
        ),
        on_terminal=terminals.append,
    )
    assert task_id is not None
    deadline = monotonic() + 2.0
    while controller.busy and monotonic() < deadline:
        application.processEvents()
        QThread.msleep(1)
    application.processEvents()

    assert not controller.busy
    assert workload_threads == [False]
    assert projection_threads == [True]
    assert [completion.state for completion in terminals] == [
        BackgroundTaskState.SUCCEEDED
    ]


def test_apply_outcome_is_typed_and_nonaccepted_values_cannot_project() -> None:
    assert TaskApplyOutcome.accepted(42) == TaskApplyOutcome(
        TaskApplyStatus.ACCEPTED,
        42,
    )
    assert TaskApplyOutcome.stale("old").status is TaskApplyStatus.STALE
    assert TaskApplyOutcome.rejected("invalid").status is (TaskApplyStatus.REJECTED)

    with pytest.raises(TypeError):
        TaskApplyOutcome("accepted")
    with pytest.raises(ValueError, match="only an accepted"):
        TaskApplyOutcome(TaskApplyStatus.STALE, projection_value=object())


def test_success_applies_projects_and_emits_terminal_exactly_once() -> None:
    harness = _Harness()
    applied: list[object] = []
    projected: list[object] = []
    terminals = []
    progress: list[str] = []
    state_changes = []
    harness.controller.state_changed.connect(state_changes.append)
    task_id = _start(
        harness,
        apply_result=lambda value: (
            applied.append(value) or TaskApplyOutcome.accepted(value + 1)
        ),
        project_result=projected.append,
        on_terminal=terminals.append,
        on_progress=progress.append,
    )

    harness.worker.progress.emit(task_id, "working")
    harness.worker.succeeded.emit(task_id, 41)
    harness.worker.succeeded.emit(task_id, 99)
    harness.worker.failed.emit(task_id, "duplicate")
    harness.worker.cancelled.emit(task_id)

    assert applied == [41]
    assert projected == [42]
    assert progress == ["working"]
    assert len(terminals) == 1
    assert terminals[0].state is BackgroundTaskState.SUCCEEDED
    assert terminals[0].apply_status is TaskApplyStatus.ACCEPTED
    assert terminals[0].value == 42
    assert harness.controller.state is BackgroundTaskState.SUCCEEDED
    assert harness.controller.busy

    harness.finish_thread()

    assert harness.controller.state is BackgroundTaskState.IDLE
    assert not harness.controller.busy
    assert state_changes == [
        BackgroundTaskState.RUNNING,
        BackgroundTaskState.SUCCEEDED,
        BackgroundTaskState.IDLE,
    ]
    assert len(terminals) == 1


def test_worker_failure_never_calls_application_apply() -> None:
    harness = _Harness()
    applied: list[object] = []
    terminals = []
    task_id = _start(
        harness,
        apply_result=lambda value: applied.append(value),
        on_terminal=terminals.append,
    )

    harness.worker.failed.emit(task_id, "backend failed")

    assert not applied
    assert len(terminals) == 1
    assert terminals[0].state is BackgroundTaskState.FAILED
    assert terminals[0].message == "backend failed"
    harness.finish_thread()
    assert not harness.controller.busy


def test_cancel_before_workload_prevents_result_application() -> None:
    harness = _Harness()
    applied: list[object] = []
    terminals = []
    task_id = _start(
        harness,
        apply_result=lambda value: (
            applied.append(value) or TaskApplyOutcome.accepted(value)
        ),
        on_terminal=terminals.append,
    )

    assert harness.controller.request_cancel()
    assert harness.worker.run_count == 0
    assert harness.worker.cancel_count == 1
    harness.worker.cancelled.emit(task_id)

    assert not applied
    assert terminals[0].state is BackgroundTaskState.CANCELLED
    harness.finish_thread()


def test_cancel_during_workload_is_cooperative_and_exactly_once() -> None:
    harness = _Harness()
    terminals = []
    task_id = _start(harness, on_terminal=terminals.append)
    harness.thread.started.emit()
    assert harness.worker.run_count == 1

    assert harness.controller.request_cancel()
    harness.worker.cancelled.emit(task_id)
    harness.worker.cancelled.emit(task_id)
    harness.finish_thread()

    assert [item.state for item in terminals] == [BackgroundTaskState.CANCELLED]
    assert not harness.controller.busy


def test_success_arriving_after_cancel_is_cancelled_without_apply() -> None:
    harness = _Harness()
    applied: list[object] = []
    terminals = []
    task_id = _start(
        harness,
        apply_result=lambda value: (
            applied.append(value) or TaskApplyOutcome.accepted(value)
        ),
        on_terminal=terminals.append,
    )

    assert harness.controller.request_cancel()
    harness.worker.succeeded.emit(task_id, "late")

    assert not applied
    assert terminals[0].state is BackgroundTaskState.CANCELLED
    assert harness.controller.state is BackgroundTaskState.CANCELLED
    harness.finish_thread()


def test_application_exception_before_commit_becomes_failed() -> None:
    harness = _Harness()
    terminals = []
    task_id = _start(
        harness,
        apply_result=lambda _value: (_ for _ in ()).throw(RuntimeError("apply failed")),
        on_terminal=terminals.append,
    )

    harness.worker.succeeded.emit(task_id, object())

    assert terminals[0].state is BackgroundTaskState.FAILED
    assert terminals[0].apply_status is None
    assert terminals[0].message == "apply failed"
    harness.finish_thread()


def test_stale_apply_discards_without_gui_projection() -> None:
    harness = _Harness()
    projected: list[object] = []
    rebuilt: list[bool] = []
    terminals = []
    task_id = _start(
        harness,
        apply_result=lambda _value: TaskApplyOutcome.stale("revision changed"),
        project_result=projected.append,
        rebuild_projection=lambda: rebuilt.append(True),
        on_terminal=terminals.append,
    )

    harness.worker.succeeded.emit(task_id, object())

    assert not projected
    assert not rebuilt
    assert terminals[0].state is BackgroundTaskState.DISCARDED
    assert terminals[0].apply_status is TaskApplyStatus.STALE
    assert terminals[0].message == "revision changed"
    harness.finish_thread()


def test_rejected_apply_becomes_failed_without_gui_projection() -> None:
    harness = _Harness()
    projected: list[object] = []
    terminals = []
    task_id = _start(
        harness,
        apply_result=lambda _value: TaskApplyOutcome.rejected("invalid result"),
        project_result=projected.append,
        on_terminal=terminals.append,
    )

    harness.worker.succeeded.emit(task_id, object())

    assert not projected
    assert terminals[0].state is BackgroundTaskState.FAILED
    assert terminals[0].apply_status is TaskApplyStatus.REJECTED
    assert terminals[0].message == "invalid result"
    harness.finish_thread()


def test_projection_failure_keeps_success_and_rebuilds_without_reapply() -> None:
    harness = _Harness()
    applied: list[object] = []
    projected: list[object] = []
    rebuilt: list[bool] = []
    reported: list[str] = []
    terminals = []

    def project(value: object) -> None:
        projected.append(value)
        raise RuntimeError("projection failed")

    task_id = _start(
        harness,
        apply_result=lambda value: (
            applied.append(value) or TaskApplyOutcome.accepted("delta")
        ),
        project_result=project,
        rebuild_projection=lambda: rebuilt.append(True),
        on_terminal=terminals.append,
        on_projection_error=reported.append,
    )

    harness.worker.succeeded.emit(task_id, "payload")
    harness.worker.succeeded.emit(task_id, "duplicate")

    assert applied == ["payload"]
    assert projected == ["delta"]
    assert rebuilt == [True]
    assert reported == ["projection failed"]
    assert len(terminals) == 1
    completion = terminals[0]
    assert completion.state is BackgroundTaskState.SUCCEEDED
    assert completion.projection_error == "projection failed"
    assert completion.rebuild_error is None
    assert harness.controller.state is BackgroundTaskState.SUCCEEDED
    harness.finish_thread()


def test_thread_finished_before_terminal_waits_for_callback() -> None:
    harness = _Harness()
    terminals = []
    task_id = _start(harness, on_terminal=terminals.append)

    harness.thread.finished.emit()

    assert harness.controller.busy
    assert harness.controller.state is BackgroundTaskState.RUNNING
    assert not terminals

    harness.worker.succeeded.emit(task_id, "done")

    assert len(terminals) == 1
    assert not harness.controller.busy
    assert harness.controller.state is BackgroundTaskState.IDLE


def test_thread_finish_inside_projection_waits_until_callback_finalize() -> None:
    harness = _Harness()
    observations: list[tuple[bool, BackgroundTaskState]] = []
    terminals = []

    def project(_value: object) -> None:
        harness.thread.finished.emit()
        observations.append((harness.controller.busy, harness.controller.state))

    task_id = _start(
        harness,
        project_result=project,
        on_terminal=terminals.append,
    )
    harness.worker.succeeded.emit(task_id, "done")

    assert observations == [(True, BackgroundTaskState.SUCCEEDED)]
    assert len(terminals) == 1
    assert not harness.controller.busy
    assert harness.controller.state is BackgroundTaskState.IDLE


def test_duplicate_reentrant_signal_and_stale_task_id_do_not_reapply() -> None:
    harness = _Harness()
    applied: list[object] = []
    terminals = []

    def apply(value: object) -> TaskApplyOutcome:
        applied.append(value)
        harness.worker.succeeded.emit(harness.worker.task_id, "reentrant")
        return TaskApplyOutcome.accepted(value)

    task_id = _start(
        harness,
        apply_result=apply,
        on_terminal=terminals.append,
    )
    harness.worker.succeeded.emit(task_id + 99, "stale")
    harness.worker.failed.emit(task_id + 99, "stale")
    harness.worker.succeeded.emit(task_id, "current")

    assert applied == ["current"]
    assert len(terminals) == 1
    harness.finish_thread()


def test_close_after_cancel_runs_only_after_cleanup() -> None:
    harness = _Harness()
    closed: list[tuple[bool, BackgroundTaskState]] = []
    task_id = _start(harness)

    assert harness.controller.request_cancel(
        after_cleanup=lambda: closed.append(
            (harness.controller.busy, harness.controller.state)
        )
    )
    harness.worker.succeeded.emit(task_id, "late")
    assert not closed

    harness.finish_thread()
    harness.thread.finished.emit()

    assert closed == [(False, BackgroundTaskState.IDLE)]


def test_busy_is_the_single_start_gate_and_projects_state_changes() -> None:
    harness = _Harness()
    busy_changes: list[bool] = []
    harness.controller.busy_changed.connect(busy_changes.append)
    first_id = _start(harness)

    assert harness.controller.busy
    assert harness.controller.current_task_id == first_id
    assert harness.controller.current_task_name == "Focused task"
    assert (
        harness.controller.start(
            lambda _context: None,
            task_name="Duplicate",
            apply_result=TaskApplyOutcome.accepted,
        )
        is None
    )

    harness.worker.succeeded.emit(first_id, None)
    harness.finish_thread()
    second_id = _start(harness)

    assert second_id == first_id + 1
    assert busy_changes == [True, False, True]
