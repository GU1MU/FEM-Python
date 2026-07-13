"""仅在当前 GUI 会话中存在的轻量分析作业数据。"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class JobStatus(str, Enum):
    """当前线性静力作业的状态。"""

    RUNNING = "运行中"
    COMPLETED = "已完成"
    FAILED = "失败"


@dataclass(slots=True)
class AnalysisJob:
    """纯内存作业记录，不包含任何文件或持久化信息。"""

    name: str
    step_name: str
    status: JobStatus
    created_at: datetime = field(default_factory=datetime.now)
    started_at: datetime | None = None
    finished_at: datetime | None = None
    source_job_name: str | None = None
    model_result: Any | None = None
    result_data: Any | None = None
    error: str | None = None
    messages: list[str] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)

    def add_message(self, message: str) -> None:
        """附加带本地时间前缀的会话日志。"""
        self.messages.append(f"{datetime.now():%H:%M:%S}  {str(message).strip()}")

    @property
    def elapsed_seconds(self) -> float | None:
        """返回已开始作业的动态或固定耗时。"""
        if self.started_at is None:
            return None
        end = datetime.now() if self.finished_at is None else self.finished_at
        return max(0.0, (end - self.started_at).total_seconds())

    @property
    def has_result(self) -> bool:
        """仅完整成功作业可以作为历史结果打开。"""
        return (
            self.status is JobStatus.COMPLETED
            and self.model_result is not None
            and self.result_data is not None
        )
