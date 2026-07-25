"""Analysis run lifecycle and result provenance."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Mapping


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class RunStatus(str, Enum):
    """Lifecycle states for an in-session analysis run."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    COMPLETED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True)
class AnalysisRun:
    """One run tied to an exact model artifact and analysis step."""

    run_id: str
    name: str
    step_name: str
    artifact_id: str
    model_revision: int
    status: RunStatus = RunStatus.PENDING
    created_at: datetime = field(default_factory=utc_now)
    started_at: datetime | None = None
    finished_at: datetime | None = None
    source_run_id: str | None = None
    result_id: str | None = None
    error: str | None = None
    cancellation_requested: bool = False
    messages: tuple[str, ...] = ()
    timings: Mapping[str, float] = field(default_factory=dict)

    @property
    def has_result(self) -> bool:
        return self.status is RunStatus.SUCCEEDED and self.result_id is not None

    @property
    def elapsed_seconds(self) -> float | None:
        if self.started_at is None:
            return None
        end = utc_now() if self.finished_at is None else self.finished_at
        return max(0.0, (end - self.started_at).total_seconds())


@dataclass(frozen=True, slots=True)
class ResultProvenance:
    """Identity of the session inputs that produced a result."""

    session_id: str
    artifact_id: str
    model_revision: int
    step_name: str
    run_id: str


@dataclass(frozen=True, slots=True)
class ResultRecord:
    """One solver result plus its complete provenance."""

    result_id: str
    provenance: ResultProvenance
    result: Any
    created_at: datetime = field(default_factory=utc_now)

    @property
    def model_result(self) -> Any:
        return self.result

