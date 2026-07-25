"""Step-specific validation stamps and typed records."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone

from .diagnostics import PreflightReport


@dataclass(frozen=True, slots=True)
class ValidationStamp:
    """Exact model/step identity covered by a validation report."""

    session_id: str
    artifact_id: str
    model_revision: int
    step_name: str


@dataclass(frozen=True, slots=True)
class ValidationRecord:
    """A successful or failed validation attempt for one stamped step."""

    stamp: ValidationStamp
    report: PreflightReport
    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    def __post_init__(self) -> None:
        if not isinstance(self.report, PreflightReport):
            raise TypeError("validation report must be PreflightReport")
        object.__setattr__(self, "report", deepcopy(self.report))

    @property
    def passed(self) -> bool:
        """Derive the stamp outcome from its typed report."""

        return self.report.passed

    @property
    def is_valid(self) -> bool:
        return self.passed

