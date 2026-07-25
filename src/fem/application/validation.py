"""Step-specific validation records."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


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
    passed: bool
    report: Any = None
    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    @property
    def is_valid(self) -> bool:
        return self.passed

