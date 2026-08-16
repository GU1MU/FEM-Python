"""Background task primitives that never update Qt widgets directly."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import logging
import time
from threading import Event, Thread
from typing import Any

from PySide6.QtCore import QObject, Signal, Slot


class TaskCancelled(RuntimeError):
    """Internal control-flow exception for cooperative task cancellation."""


class OwnedWorkerTimeout(RuntimeError):
    """One dedicated worker exceeded its bounded wall-clock budget."""


def run_owned_worker(
    workload: Callable[[], Any],
    *,
    name: str,
    timeout_seconds: float,
    cancel_event: Event | None = None,
    should_cancel: Callable[[], bool] | None = None,
    poll_seconds: float = 0.05,
) -> Any:
    """Run one workload on a fresh dedicated thread and return its plain result.

    Mirrors the ``inspect_abaqus`` isolation contract without a subprocess:
    the caller blocks in a bounded poll loop, only plain data crosses back,
    and budget exhaustion raises ``OwnedWorkerTimeout`` instead of leaking the
    worker thread.  Workload exceptions propagate unchanged so existing error
    contracts keep working.  Each invocation gets its own thread so one
    CAD-owning workload never inherits global state from a previous run; the
    workload owns every resource it opens (context managers close CAD models
    when cancellation or another exception unwinds the call).
    """

    outcome: dict[str, Any] = {}
    done = Event()

    def target() -> None:
        try:
            outcome["result"] = workload()
        except BaseException as error:
            outcome["error"] = error
        finally:
            done.set()

    thread = Thread(target=target, name=name, daemon=True)
    thread.start()
    deadline = time.monotonic() + timeout_seconds
    while not done.wait(timeout=poll_seconds):
        if should_cancel is not None and should_cancel():
            if cancel_event is not None:
                cancel_event.set()
        if time.monotonic() >= deadline:
            if cancel_event is not None:
                # Make the abandoned workload unwind at its next checkpoint so
                # it closes every resource it owns instead of running on
                # unsupervised (e.g. holding live CAD entities on its thread).
                cancel_event.set()
            raise OwnedWorkerTimeout(
                f"{name} exceeded its {timeout_seconds:.0f}s worker budget"
            )
    if "error" in outcome:
        raise outcome["error"]
    return outcome["result"]


@dataclass(slots=True)
class TaskContext:
    """Thread-safe progress and cancellation access passed to one workload."""

    task_id: int
    _cancelled: Event
    _report: Callable[[int, str], None]

    @property
    def is_cancelled(self) -> bool:
        return self._cancelled.is_set()

    def checkpoint(self) -> None:
        if self.is_cancelled:
            raise TaskCancelled()

    def report(self, stage: str) -> None:
        self.checkpoint()
        self._report(self.task_id, str(stage))


class TaskWorker(QObject):
    """Execute one non-GUI workload and report its complete lifecycle."""

    started = Signal(int)
    progress = Signal(int, str)
    succeeded = Signal(int, object)
    failed = Signal(int, str)
    cancelled = Signal(int)
    finished = Signal(int)

    def __init__(
        self,
        task_id: int,
        workload: Callable[[TaskContext], Any],
    ) -> None:
        super().__init__()
        self.task_id = int(task_id)
        self._workload = workload
        self._cancelled = Event()

    def request_cancel(self) -> None:
        """Request cancellation without queuing work on the busy worker loop."""
        self._cancelled.set()

    @Slot()
    def run(self) -> None:
        self.started.emit(self.task_id)
        context = TaskContext(
            self.task_id,
            self._cancelled,
            self.progress.emit,
        )
        try:
            context.checkpoint()
            result = self._workload(context)
            context.checkpoint()
        except TaskCancelled:
            self.cancelled.emit(self.task_id)
        except Exception as error:
            logging.exception("GUI background task failed")
            self.failed.emit(
                self.task_id,
                str(error).strip() or type(error).__name__,
            )
        else:
            self.succeeded.emit(self.task_id, result)
        finally:
            self.finished.emit(self.task_id)
