"""Single-owner orchestration for GUI background-task lifecycles."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
from typing import Any

from PySide6.QtCore import QObject, QThread, Signal, Slot

from .workers import TaskContext, TaskWorker


class BackgroundTaskState(str, Enum):
    """Finite states owned by one :class:`BackgroundTaskController`."""

    IDLE = "idle"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    DISCARDED = "discarded"

    @property
    def terminal(self) -> bool:
        return self in {
            BackgroundTaskState.SUCCEEDED,
            BackgroundTaskState.FAILED,
            BackgroundTaskState.CANCELLED,
            BackgroundTaskState.DISCARDED,
        }


class TaskApplyStatus(str, Enum):
    """Typed result of applying one worker payload at the application boundary."""

    ACCEPTED = "accepted"
    STALE = "stale"
    REJECTED = "rejected"


@dataclass(frozen=True, slots=True)
class TaskApplyOutcome:
    """Application acceptance result consumed exactly once by the controller."""

    status: TaskApplyStatus
    projection_value: Any = None
    message: str = ""

    def __post_init__(self) -> None:
        if type(self.status) is not TaskApplyStatus:
            raise TypeError("status must be a TaskApplyStatus")
        message = str(self.message).strip()
        if (
            self.status is not TaskApplyStatus.ACCEPTED
            and self.projection_value is not None
        ):
            raise ValueError(
                "only an accepted apply outcome may carry a projection value"
            )
        object.__setattr__(self, "message", message)

    @classmethod
    def accepted(cls, projection_value: Any = None) -> TaskApplyOutcome:
        return cls(TaskApplyStatus.ACCEPTED, projection_value)

    @classmethod
    def stale(cls, message: str = "") -> TaskApplyOutcome:
        return cls(TaskApplyStatus.STALE, message=message)

    @classmethod
    def rejected(cls, message: str) -> TaskApplyOutcome:
        return cls(TaskApplyStatus.REJECTED, message=message)


@dataclass(frozen=True, slots=True)
class TaskCompletion:
    """Immutable terminal record emitted at most once for one task."""

    task_id: int
    task_name: str
    state: BackgroundTaskState
    apply_status: TaskApplyStatus | None = None
    value: Any = None
    message: str = ""
    projection_error: str | None = None
    rebuild_error: str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.task_id, bool) or int(self.task_id) <= 0:
            raise ValueError("task_id must be a positive integer")
        if type(self.task_name) is not str or not self.task_name.strip():
            raise ValueError("task_name must be a non-empty string")
        if type(self.state) is not BackgroundTaskState or not self.state.terminal:
            raise ValueError("completion state must be terminal")
        if self.apply_status is not None and (
            type(self.apply_status) is not TaskApplyStatus
        ):
            raise TypeError("apply_status must be a TaskApplyStatus or None")
        if (
            self.state is BackgroundTaskState.SUCCEEDED
            and self.apply_status is not TaskApplyStatus.ACCEPTED
        ):
            raise ValueError("a succeeded task requires an accepted apply outcome")
        if (
            self.state is BackgroundTaskState.DISCARDED
            and self.apply_status is not TaskApplyStatus.STALE
        ):
            raise ValueError("a discarded task requires a stale apply outcome")
        object.__setattr__(self, "task_id", int(self.task_id))
        object.__setattr__(self, "task_name", self.task_name.strip())
        object.__setattr__(self, "message", str(self.message).strip())
        for field_name in ("projection_error", "rebuild_error"):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(self, field_name, str(value).strip())


ApplyCallback = Callable[[object], TaskApplyOutcome]
ProjectionCallback = Callable[[object], None]
TerminalCallback = Callable[[TaskCompletion], None]
ProgressCallback = Callable[[str], None]
ProjectionErrorCallback = Callable[[str], None]
Workload = Callable[[TaskContext], object]


@dataclass(slots=True)
class _ActiveTask:
    task_id: int
    task_name: str
    thread: Any
    worker: Any
    apply_result: ApplyCallback
    project_result: ProjectionCallback | None
    rebuild_projection: Callable[[], None] | None
    on_terminal: TerminalCallback | None
    on_progress: ProgressCallback | None
    on_projection_error: ProjectionErrorCallback | None
    state: BackgroundTaskState = BackgroundTaskState.RUNNING
    cancel_requested: bool = False
    callback_active: bool = False
    thread_finished: bool = False
    completion: TaskCompletion | None = None
    after_cleanup: list[Callable[[], None]] = field(default_factory=list)


class BackgroundTaskController(QObject):
    """Own one QThread/TaskWorker pair and its exactly-once terminal contract."""

    busy_changed = Signal(bool)
    state_changed = Signal(object)
    cancelling_changed = Signal(bool)
    progress = Signal(int, str)
    completed = Signal(object)
    projection_failed = Signal(str)

    def __init__(
        self,
        parent: QObject | None = None,
        *,
        thread_factory: Callable[[QObject | None], Any] | None = None,
        worker_factory: Callable[[int, Workload], Any] | None = None,
    ) -> None:
        super().__init__(parent)
        self._thread_factory = thread_factory or QThread
        self._worker_factory = worker_factory or TaskWorker
        self._task_counter = 0
        self._active: _ActiveTask | None = None
        self._last_completion: TaskCompletion | None = None

    @property
    def busy(self) -> bool:
        return self._active is not None

    @property
    def state(self) -> BackgroundTaskState:
        active = self._active
        return BackgroundTaskState.IDLE if active is None else active.state

    @property
    def current_task_id(self) -> int | None:
        return None if self._active is None else self._active.task_id

    @property
    def current_task_name(self) -> str:
        return "" if self._active is None else self._active.task_name

    @property
    def cancel_requested(self) -> bool:
        return bool(self._active is not None and self._active.cancel_requested)

    @property
    def last_completion(self) -> TaskCompletion | None:
        return self._last_completion

    def start(
        self,
        workload: Workload,
        *,
        task_name: str,
        apply_result: ApplyCallback,
        project_result: ProjectionCallback | None = None,
        rebuild_projection: Callable[[], None] | None = None,
        on_terminal: TerminalCallback | None = None,
        on_progress: ProgressCallback | None = None,
        on_projection_error: ProjectionErrorCallback | None = None,
    ) -> int | None:
        """Start one task, returning ``None`` while another task is busy."""

        if self.busy:
            return None
        if not callable(workload):
            raise TypeError("workload must be callable")
        if type(task_name) is not str or not task_name.strip():
            raise ValueError("task_name must be a non-empty string")
        if not callable(apply_result):
            raise TypeError("apply_result must be callable")
        for callback_name, callback in (
            ("project_result", project_result),
            ("rebuild_projection", rebuild_projection),
            ("on_terminal", on_terminal),
            ("on_progress", on_progress),
            ("on_projection_error", on_projection_error),
        ):
            if callback is not None and not callable(callback):
                raise TypeError(f"{callback_name} must be callable or None")

        self._task_counter += 1
        task_id = self._task_counter
        thread = self._thread_factory(self)
        worker = self._worker_factory(task_id, workload)
        active = _ActiveTask(
            task_id=task_id,
            task_name=task_name.strip(),
            thread=thread,
            worker=worker,
            apply_result=apply_result,
            project_result=project_result,
            rebuild_projection=rebuild_projection,
            on_terminal=on_terminal,
            on_progress=on_progress,
            on_projection_error=on_projection_error,
        )
        self._active = active
        self.busy_changed.emit(True)
        self.state_changed.emit(BackgroundTaskState.RUNNING)
        try:
            self._wire(active)
            thread.start()
        except Exception as error:
            logging.exception("GUI background task failed to start")
            active.thread_finished = True
            self._publish_simple_terminal(
                active,
                BackgroundTaskState.FAILED,
                message=_error_message(error),
            )
        return task_id

    def request_cancel(
        self,
        *,
        after_cleanup: Callable[[], None] | None = None,
    ) -> bool:
        """Request cooperative cancellation and optionally run after cleanup."""

        active = self._active
        if active is None or active.state is not BackgroundTaskState.RUNNING:
            return False
        if after_cleanup is not None:
            if not callable(after_cleanup):
                raise TypeError("after_cleanup must be callable or None")
            active.after_cleanup.append(after_cleanup)
        if active.cancel_requested:
            return False
        active.cancel_requested = True
        active.worker.request_cancel()
        self.cancelling_changed.emit(True)
        return True

    def _wire(self, active: _ActiveTask) -> None:
        thread = active.thread
        worker = active.worker
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(self._worker_progress)
        worker.succeeded.connect(self._worker_succeeded)
        worker.failed.connect(self._worker_failed)
        worker.cancelled.connect(self._worker_cancelled)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(self._thread_finished)
        thread.finished.connect(thread.deleteLater)

    @Slot(int, str)
    def _worker_progress(self, task_id: int, stage: str) -> None:
        active = self._matching_running(task_id)
        if active is None or active.cancel_requested:
            return
        text = str(stage)
        self.progress.emit(active.task_id, text)
        if active.on_progress is not None:
            try:
                active.on_progress(text)
            except Exception:
                logging.exception("GUI background task progress callback failed")

    @Slot(int, object)
    def _worker_succeeded(self, task_id: int, payload: object) -> None:
        active = self._matching_running(task_id)
        if active is None:
            return
        if active.cancel_requested:
            self._publish_simple_terminal(
                active,
                BackgroundTaskState.CANCELLED,
            )
            return

        active.callback_active = True
        try:
            try:
                outcome = active.apply_result(payload)
                if type(outcome) is not TaskApplyOutcome:
                    raise TypeError("apply_result must return a TaskApplyOutcome")
            except Exception as error:
                logging.exception("GUI background task application callback failed")
                self._publish_simple_terminal(
                    active,
                    BackgroundTaskState.FAILED,
                    message=_error_message(error),
                )
                return

            if outcome.status is TaskApplyStatus.STALE:
                self._publish_simple_terminal(
                    active,
                    BackgroundTaskState.DISCARDED,
                    apply_status=outcome.status,
                    message=outcome.message,
                )
                return
            if outcome.status is TaskApplyStatus.REJECTED:
                self._publish_simple_terminal(
                    active,
                    BackgroundTaskState.FAILED,
                    apply_status=outcome.status,
                    message=outcome.message or "task result was rejected",
                )
                return
            self._project_accepted(active, outcome)
        finally:
            active.callback_active = False
            self._maybe_finalize(active)

    def _project_accepted(
        self,
        active: _ActiveTask,
        outcome: TaskApplyOutcome,
    ) -> None:
        active.state = BackgroundTaskState.SUCCEEDED
        self.state_changed.emit(active.state)
        projection_error = None
        rebuild_error = None
        if active.project_result is not None:
            try:
                active.project_result(outcome.projection_value)
            except Exception as error:
                logging.exception("accepted Session transition GUI projection failed")
                projection_error = _error_message(error)
                if active.rebuild_projection is None:
                    rebuild_error = "full projection rebuild callback is not configured"
                else:
                    try:
                        active.rebuild_projection()
                    except Exception as rebuild_exception:
                        logging.exception("full GUI projection rebuild failed")
                        rebuild_error = _error_message(rebuild_exception)
                self._report_projection_error(
                    active,
                    projection_error,
                    rebuild_error,
                )
        completion = TaskCompletion(
            task_id=active.task_id,
            task_name=active.task_name,
            state=BackgroundTaskState.SUCCEEDED,
            apply_status=outcome.status,
            value=outcome.projection_value,
            projection_error=projection_error,
            rebuild_error=rebuild_error,
        )
        self._publish_completion(active, completion)

    @Slot(int, str)
    def _worker_failed(self, task_id: int, message: str) -> None:
        active = self._matching_running(task_id)
        if active is None:
            return
        active.callback_active = True
        try:
            if active.cancel_requested:
                self._publish_simple_terminal(
                    active,
                    BackgroundTaskState.CANCELLED,
                )
                return
            self._publish_simple_terminal(
                active,
                BackgroundTaskState.FAILED,
                message=str(message).strip() or "background task failed",
            )
        finally:
            active.callback_active = False
            self._maybe_finalize(active)

    @Slot(int)
    def _worker_cancelled(self, task_id: int) -> None:
        active = self._matching_running(task_id)
        if active is None:
            return
        active.callback_active = True
        try:
            self._publish_simple_terminal(
                active,
                BackgroundTaskState.CANCELLED,
            )
        finally:
            active.callback_active = False
            self._maybe_finalize(active)

    def _publish_simple_terminal(
        self,
        active: _ActiveTask,
        state: BackgroundTaskState,
        *,
        apply_status: TaskApplyStatus | None = None,
        message: str = "",
    ) -> None:
        if active.completion is not None or active.state.terminal:
            return
        active.state = state
        self.state_changed.emit(state)
        self._publish_completion(
            active,
            TaskCompletion(
                task_id=active.task_id,
                task_name=active.task_name,
                state=state,
                apply_status=apply_status,
                message=message,
            ),
        )

    def _publish_completion(
        self,
        active: _ActiveTask,
        completion: TaskCompletion,
    ) -> None:
        if active.completion is not None:
            return
        active.completion = completion
        self._last_completion = completion
        self.completed.emit(completion)
        if active.on_terminal is not None:
            try:
                active.on_terminal(completion)
            except Exception:
                logging.exception("GUI background task terminal callback failed")
        self._maybe_finalize(active)

    def _report_projection_error(
        self,
        active: _ActiveTask,
        projection_error: str,
        rebuild_error: str | None,
    ) -> None:
        message = projection_error
        if rebuild_error:
            message = f"{message}; rebuild failed: {rebuild_error}"
        self.projection_failed.emit(message)
        if active.on_projection_error is not None:
            try:
                active.on_projection_error(message)
            except Exception:
                logging.exception("GUI projection-error callback failed")

    @Slot()
    def _thread_finished(self) -> None:
        active = self._active
        if active is None:
            return
        sender = self.sender()
        # Manual test doubles do not participate in QObject.sender(). Their
        # signal is wired only to the currently owned thread.
        thread = active.thread if sender is None else sender
        if thread is not active.thread:
            return
        active.thread_finished = True
        self._maybe_finalize(active)

    def _matching_running(self, task_id: int) -> _ActiveTask | None:
        active = self._active
        if (
            active is None
            or int(task_id) != active.task_id
            or active.state is not BackgroundTaskState.RUNNING
            or active.completion is not None
            or active.callback_active
        ):
            return None
        return active

    def _maybe_finalize(self, active: _ActiveTask) -> None:
        if (
            self._active is not active
            or active.completion is None
            or not active.thread_finished
            or active.callback_active
        ):
            return
        callbacks = tuple(active.after_cleanup)
        was_cancelling = active.cancel_requested
        self._active = None
        if was_cancelling:
            self.cancelling_changed.emit(False)
        self.state_changed.emit(BackgroundTaskState.IDLE)
        self.busy_changed.emit(False)
        for callback in callbacks:
            try:
                callback()
            except Exception:
                logging.exception("GUI background task after-cleanup callback failed")


def _error_message(error: BaseException) -> str:
    return str(error).strip() or type(error).__name__


__all__ = [
    "BackgroundTaskController",
    "BackgroundTaskState",
    "TaskApplyOutcome",
    "TaskApplyStatus",
    "TaskCompletion",
]
