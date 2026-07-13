"""不会阻塞 Qt 主线程的导入与求解任务。"""

from __future__ import annotations

from collections.abc import Callable
import logging
from typing import Any

from PySide6.QtCore import QObject, Signal, Slot


class TaskWorker(QObject):
    """执行一个无界面工作负载并报告四种生命周期信号。"""

    started = Signal()
    succeeded = Signal(object)
    failed = Signal(str)
    finished = Signal()

    def __init__(self, workload: Callable[[], Any]) -> None:
        super().__init__()
        self._workload = workload

    @Slot()
    def run(self) -> None:
        self.started.emit()
        try:
            result = self._workload()
        except Exception as error:
            logging.exception("GUI 后台任务失败")
            self.failed.emit(str(error).strip() or type(error).__name__)
        else:
            self.succeeded.emit(result)
        finally:
            self.finished.emit()
