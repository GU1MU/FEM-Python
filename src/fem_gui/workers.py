"""Background task primitives that never update Qt widgets directly."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import logging
from threading import Event
from typing import Any

from PySide6.QtCore import QObject, Signal, Slot


class TaskCancelled(RuntimeError):
    """Internal control-flow exception for cooperative task cancellation."""


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
