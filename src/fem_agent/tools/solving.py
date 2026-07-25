"""Deterministic wrapper around the existing linear-static FEM solver."""

from __future__ import annotations

import time
import unicodedata
import warnings
from dataclasses import dataclass
from typing import Any

from fem.solvers import static_linear

from ..diagnostics import DiagnosticCode, exception_diagnostic, make_diagnostic
from ..schemas import Diagnostic, DiagnosticSeverity


@dataclass(frozen=True)
class SolveOutcome:
    result: Any | None
    diagnostics: tuple[Diagnostic, ...]
    elapsed_seconds: float

    @property
    def ok(self) -> bool:
        return self.result is not None and not any(
            item.severity == DiagnosticSeverity.ERROR
            for item in self.diagnostics
        )


def solve_analysis(model: Any, step: Any) -> SolveOutcome:
    """Solve one explicit step and retain only bounded warning messages."""

    started = time.monotonic()
    captured: list[warnings.WarningMessage]
    try:
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            result = static_linear.solve(model, step=step)
    except Exception as error:
        return SolveOutcome(
            result=None,
            diagnostics=(
                exception_diagnostic(
                    DiagnosticCode.SOLVER_FAILED,
                    error,
                    source="fem.solver",
                    remediation=(
                        "Inspect constraints, material properties, loads, and "
                        "possible singularity without changing the model automatically."
                    ),
                ),
            ),
            elapsed_seconds=time.monotonic() - started,
        )

    diagnostics = tuple(
        make_diagnostic(
            "SOLVER_WARNING",
            _bounded_warning(item),
            source="fem.solver",
            severity=DiagnosticSeverity.WARNING,
        )
        for item in captured[:16]
    )
    return SolveOutcome(
        result=result,
        diagnostics=diagnostics,
        elapsed_seconds=time.monotonic() - started,
    )


def _bounded_warning(item: warnings.WarningMessage) -> str:
    message = str(item.message).strip() or item.category.__name__
    message = "".join(
        " "
        if unicodedata.category(character) in {"Cc", "Cf", "Cs", "Zl", "Zp"}
        else character
        for character in message
    ).strip()
    if len(message) > 1000:
        return message[:997] + "..."
    return message


__all__ = ["SolveOutcome", "solve_analysis"]
