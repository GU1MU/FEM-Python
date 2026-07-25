"""Deterministic FEM validation normalized into Agent diagnostics."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from fem.core import validate_model

from ..diagnostics import DiagnosticCode, exception_diagnostic, make_diagnostic
from ..schemas import Diagnostic


_SOURCE = "fem.validation"


def validate_analysis(
    model: Any,
    step: Any | str | None = None,
) -> tuple[Diagnostic, ...]:
    """Validate a model and its selected static step without mutating either.

    The FEM kernel remains the source of truth for structural validation.  This
    wrapper only resolves an optional step name, checks V0 solver semantics, and
    converts failures into provider-safe diagnostics.
    """

    try:
        selected_step = _resolve_step(model, step)
        validate_model(model, selected_step)
    except Exception as error:
        return (
            exception_diagnostic(
                DiagnosticCode.INVALID_MODEL,
                error,
                source=_SOURCE,
                remediation="Correct the imported model before confirmation.",
            ),
        )

    for candidate in _steps_to_check(model, selected_step):
        diagnostic = _unsupported_step_diagnostic(candidate)
        if diagnostic is not None:
            return (diagnostic,)
    return ()


def _resolve_step(model: Any, step: Any | str | None) -> Any | None:
    if step is None or not isinstance(step, str):
        return step
    matches = [
        candidate
        for candidate in getattr(model, "steps", ())
        if str(getattr(candidate, "name", "")) == step
    ]
    if not matches:
        raise KeyError(f"analysis step {step!r} is not defined")
    if len(matches) != 1:
        raise ValueError(f"analysis step {step!r} is not unique")
    return matches[0]


def _steps_to_check(model: Any, selected_step: Any | None) -> Iterable[Any]:
    if selected_step is not None:
        return (selected_step,)
    return tuple(getattr(model, "steps", ()))


def _unsupported_step_diagnostic(step: Any) -> Diagnostic | None:
    name = str(getattr(step, "name", "step"))
    procedure = str(getattr(step, "procedure", "")).strip().casefold()
    if procedure != "static":
        return make_diagnostic(
            DiagnosticCode.UNSUPPORTED_PROCEDURE,
            f"Analysis step {name!r} uses unsupported procedure "
            f"{getattr(step, 'procedure', None)!r}; V0 requires static.",
            source=_SOURCE,
            step=name,
            remediation="Provide exactly one linear static analysis step.",
        )

    metadata = getattr(step, "metadata", {})
    if not isinstance(metadata, Mapping):
        return make_diagnostic(
            DiagnosticCode.INVALID_MODEL,
            f"Analysis step {name!r} metadata must be a mapping.",
            source=_SOURCE,
            step=name,
            remediation="Correct the imported analysis-step metadata.",
        )
    nlgeom = next(
        (
            value
            for key, value in metadata.items()
            if str(key).strip().casefold() == "nlgeom"
        ),
        None,
    )
    if _truthy_option(nlgeom):
        return make_diagnostic(
            DiagnosticCode.UNSUPPORTED_PROCEDURE,
            f"Analysis step {name!r} enables geometric nonlinearity; "
            "V0 supports linear geometry only.",
            source=_SOURCE,
            step=name,
            remediation="Use a static step with NLGEOM disabled.",
        )
    return None


def _truthy_option(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized in {"", "0", "false", "no", "off"}:
            return False
        if normalized in {"1", "true", "yes", "on"}:
            return True
    return bool(value)


# A descriptive alias is useful to worker code that already imported the
# implementation under its earlier draft name.
validate_model_diagnostics = validate_analysis


__all__ = ["validate_analysis", "validate_model_diagnostics"]
