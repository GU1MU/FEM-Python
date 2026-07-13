"""GUI 文档状态，避免在窗口中复制模型状态。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .analysis_jobs import AnalysisJob, JobStatus


@dataclass(slots=True)
class FEMDocument:
    """当前打开的只读模型及其结果。"""

    path: Path | None = None
    model: Any | None = None
    result: Any | None = None
    step_name: str | None = None
    jobs: list[AnalysisJob] = field(default_factory=list)
    active_job_name: str | None = None

    @property
    def has_model(self) -> bool:
        return self.model is not None

    @property
    def has_result(self) -> bool:
        return self.result is not None

    def set_model(self, path: str | Path, model: Any) -> None:
        self.clear_jobs()
        self.path = Path(path)
        self.model = model
        self.result = None
        self.step_name = self.default_step_name()

    def close(self) -> None:
        self.clear_jobs()
        self.path = None
        self.model = None
        self.result = None
        self.step_name = None

    def add_job(self, job: AnalysisJob) -> None:
        """添加名称唯一的会话作业。"""
        job.name = job.name.strip()
        if not job.name:
            raise ValueError("作业名称不能为空")
        if self.find_job(job.name) is not None:
            raise ValueError(f"作业名称已存在：{job.name}")
        self.jobs.append(job)

    def find_job(self, name: str | None) -> AnalysisJob | None:
        """按忽略大小写的名称查找作业。"""
        normalized = str(name or "").strip().casefold()
        return next((job for job in self.jobs if job.name.casefold() == normalized), None)

    def next_job_name(self) -> str:
        """返回下一个不与现有名称冲突的 Job-N。"""
        number = 1
        while self.find_job(f"Job-{number}") is not None:
            number += 1
        return f"Job-{number}"

    def latest_resubmittable_job(self) -> AnalysisJob | None:
        """返回最近完成或失败的作业。"""
        for job in reversed(self.jobs):
            if job.status in {JobStatus.COMPLETED, JobStatus.FAILED}:
                return job
        return None

    def clear_jobs(self) -> None:
        """清除当前模型全部会话作业和活动作业指针。"""
        self.jobs.clear()
        self.active_job_name = None

    def default_step_name(self) -> str | None:
        if self.model is None:
            return None
        runnable = [step.name for step in self.model.steps if step.name.lower() != "initial"]
        if runnable:
            return runnable[0]
        return self.model.steps[0].name if self.model.steps else None

    def runnable_step_names(self) -> tuple[str, ...]:
        if self.model is None:
            return ()
        names = tuple(step.name for step in self.model.steps if step.name.lower() != "initial")
        return names or tuple(step.name for step in self.model.steps)
