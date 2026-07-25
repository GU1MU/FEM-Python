"""Stable, headless diagnostics shared by application services."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Mapping


class PreflightSeverity(str, Enum):
    """Severity used to derive a preflight outcome."""

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class PreflightStage(str, Enum):
    """Deterministic stage ordering for model-check diagnostics."""

    CAPABILITY = "capability"
    STRUCTURE = "structure"
    DEFINITIONS = "definitions"
    STEP = "step"
    BOUNDARY = "boundary"
    STIFFNESS = "stiffness"
    OUTPUT = "output"
    INTERNAL = "internal"


@dataclass(frozen=True, slots=True)
class PreflightDiagnostic:
    """One stable machine-readable model-check finding."""

    code: str
    severity: PreflightSeverity
    stage: PreflightStage
    message: str
    subject: Any = None
    path: tuple[str, ...] = ()
    remediation: str | None = None
    details: tuple[tuple[str, Any], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "code", str(self.code).strip())
        object.__setattr__(
            self,
            "severity",
            PreflightSeverity(self.severity),
        )
        object.__setattr__(self, "stage", PreflightStage(self.stage))
        object.__setattr__(self, "message", str(self.message))
        object.__setattr__(
            self,
            "path",
            tuple(str(part) for part in self.path),
        )
        object.__setattr__(self, "subject", deepcopy(self.subject))
        object.__setattr__(
            self,
            "details",
            _owned_details(self.details),
        )
        if not self.code:
            raise ValueError("diagnostic code must not be empty")
        if self.remediation is not None:
            object.__setattr__(
                self,
                "remediation",
                str(self.remediation),
            )

    @property
    def blocking(self) -> bool:
        """Return whether this finding prevents a passing report."""

        return self.severity is PreflightSeverity.ERROR

    def details_dict(self) -> dict[str, Any]:
        """Return a detached mapping for renderers and logging."""

        return deepcopy(dict(self.details))


def _owned_details(
    value: Mapping[str, Any] | Iterable[tuple[str, Any]],
) -> tuple[tuple[str, Any], ...]:
    items = value.items() if isinstance(value, Mapping) else value
    return tuple((str(key), deepcopy(item)) for key, item in items)


@dataclass(frozen=True, slots=True)
class PreflightFacts:
    """Small immutable summary displayed by front ends."""

    model_name: str | None = None
    step_name: str = ""
    procedure: str = ""
    node_count: int = 0
    element_count: int = 0
    dof_count: int = 0
    material_count: int = 0
    section_count: int = 0
    displacement_count: int = 0
    nodal_load_count: int = 0
    edge_load_count: int = 0
    surface_load_count: int = 0
    line_load_count: int = 0
    gravity_load_count: int = 0


@dataclass(frozen=True, slots=True)
class PreflightReport:
    """Typed static-preflight result whose outcome cannot be supplied."""

    step_name: str
    diagnostics: tuple[PreflightDiagnostic, ...] = ()
    facts: PreflightFacts = field(default_factory=PreflightFacts)
    numerical_stability_checked: bool = False
    session_id: str | None = None
    artifact_id: str | None = None
    model_revision: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "step_name", str(self.step_name))
        object.__setattr__(
            self,
            "diagnostics",
            tuple(deepcopy(tuple(self.diagnostics))),
        )
        object.__setattr__(self, "facts", deepcopy(self.facts))
        if not isinstance(self.facts, PreflightFacts):
            raise TypeError("preflight facts must be PreflightFacts")
        if any(
            not isinstance(item, PreflightDiagnostic)
            for item in self.diagnostics
        ):
            raise TypeError(
                "preflight diagnostics must be PreflightDiagnostic values"
            )
        if self.model_revision is not None:
            object.__setattr__(
                self,
                "model_revision",
                int(self.model_revision),
            )

    @property
    def passed(self) -> bool:
        """Derive success solely from the absence of error diagnostics."""

        return not any(item.blocking for item in self.diagnostics)

    @property
    def errors(self) -> tuple[PreflightDiagnostic, ...]:
        return tuple(item for item in self.diagnostics if item.blocking)

    @property
    def warnings(self) -> tuple[PreflightDiagnostic, ...]:
        return tuple(
            item
            for item in self.diagnostics
            if item.severity is PreflightSeverity.WARNING
        )


def internal_error_report(
    step_name: str,
    error: Any,
    *,
    session_id: str | None = None,
    artifact_id: str | None = None,
    model_revision: int | None = None,
) -> PreflightReport:
    """Return the canonical typed report for a validation worker failure."""

    diagnostic = PreflightDiagnostic(
        code="preflight.internal_error",
        severity=PreflightSeverity.ERROR,
        stage=PreflightStage.INTERNAL,
        message=str(error),
        subject=str(step_name),
        path=("steps", str(step_name)),
        remediation="请检查模型定义或查看应用日志。",
        details={"error_type": type(error).__name__},
    )
    return PreflightReport(
        step_name=str(step_name),
        diagnostics=(diagnostic,),
        facts=PreflightFacts(step_name=str(step_name)),
        numerical_stability_checked=False,
        session_id=session_id,
        artifact_id=artifact_id,
        model_revision=model_revision,
    )


__all__ = [
    "PreflightDiagnostic",
    "PreflightFacts",
    "PreflightReport",
    "PreflightSeverity",
    "PreflightStage",
    "internal_error_report",
]
