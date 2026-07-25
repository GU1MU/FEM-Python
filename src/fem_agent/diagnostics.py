"""Stable diagnostic codes and exception normalization."""

from __future__ import annotations

from enum import Enum
import re
import unicodedata
from typing import Any

from .schemas import Diagnostic, DiagnosticSeverity


class DiagnosticCode(str, Enum):
    INVALID_INPUT = "INVALID_INPUT"
    INPUT_OUTSIDE_WORKSPACE = "INPUT_OUTSIDE_WORKSPACE"
    INPUT_TOO_LARGE = "INPUT_TOO_LARGE"
    UNSUPPORTED_KEYWORD = "UNSUPPORTED_KEYWORD"
    UNSUPPORTED_KEYWORD_OPTION = "UNSUPPORTED_KEYWORD_OPTION"
    UNSUPPORTED_ELEMENT = "UNSUPPORTED_ELEMENT"
    IMPORT_FAILED = "IMPORT_FAILED"
    INVALID_MODEL = "INVALID_MODEL"
    UNSUPPORTED_PROCEDURE = "UNSUPPORTED_PROCEDURE"
    MULTI_STEP_HISTORY_UNSUPPORTED = "MULTI_STEP_HISTORY_UNSUPPORTED"
    UNIT_CONTEXT_REQUIRED = "UNIT_CONTEXT_REQUIRED"
    RESULT_REQUEST_REQUIRED = "RESULT_REQUEST_REQUIRED"
    STALE_REVISION = "STALE_REVISION"
    CONFIRMATION_REQUIRED = "CONFIRMATION_REQUIRED"
    BLOCKING_DIAGNOSTICS = "BLOCKING_DIAGNOSTICS"
    RESOURCE_LIMIT = "RESOURCE_LIMIT"
    SOLVER_FAILED = "SOLVER_FAILED"
    WORKER_CRASH = "WORKER_CRASH"
    WORKER_TIMEOUT = "WORKER_TIMEOUT"
    OPERATION_CANCELLED = "OPERATION_CANCELLED"
    OPERATION_IN_PROGRESS = "OPERATION_IN_PROGRESS"
    RESULT_QUERY_FAILED = "RESULT_QUERY_FAILED"
    EXPORT_FAILED = "EXPORT_FAILED"
    UNKNOWN_TOOL = "UNKNOWN_TOOL"
    INVALID_TOOL_ARGUMENTS = "INVALID_TOOL_ARGUMENTS"
    TOOL_LIMIT_EXCEEDED = "TOOL_LIMIT_EXCEEDED"
    PROVIDER_AUTHENTICATION_FAILED = "PROVIDER_AUTHENTICATION_FAILED"
    PROVIDER_PAYMENT_REQUIRED = "PROVIDER_PAYMENT_REQUIRED"
    PROVIDER_RATE_LIMITED = "PROVIDER_RATE_LIMITED"
    PROVIDER_TIMEOUT = "PROVIDER_TIMEOUT"
    PROVIDER_UNAVAILABLE = "PROVIDER_UNAVAILABLE"
    PROVIDER_MALFORMED_RESPONSE = "PROVIDER_MALFORMED_RESPONSE"
    IGNORED_METADATA = "IGNORED_METADATA"


class AgentDiagnosticError(RuntimeError):
    """An operation failure carrying a provider-safe diagnostic."""

    def __init__(self, diagnostic: Diagnostic):
        super().__init__(diagnostic.message)
        self.diagnostic = diagnostic


def make_diagnostic(
    code: DiagnosticCode | str,
    message: str,
    *,
    source: str,
    severity: DiagnosticSeverity = DiagnosticSeverity.ERROR,
    entity: str | None = None,
    step: str | None = None,
    remediation: str | None = None,
) -> Diagnostic:
    return Diagnostic(
        code=code.value if isinstance(code, DiagnosticCode) else str(code),
        severity=severity,
        message=message,
        source=source,
        entity=entity,
        step=step,
        remediation=remediation,
    )


def exception_diagnostic(
    code: DiagnosticCode | str,
    error: BaseException,
    *,
    source: str,
    remediation: str | None = None,
) -> Diagnostic:
    """Normalize an exception without leaking a traceback or local path."""

    message = _safe_exception_message(error)
    if len(message) > 1024:
        message = message[:1021] + "..."
    return make_diagnostic(
        code,
        message,
        source=source,
        remediation=remediation,
    )


def _safe_exception_message(error: BaseException) -> str:
    if not isinstance(error, (KeyError, TypeError, ValueError)):
        return f"{type(error).__name__}: the local operation failed."
    message = str(error).strip() or type(error).__name__
    normalized = message.casefold()
    sensitive_markers = (
        "api_key",
        "apikey",
        "password",
        "secret",
        "token=",
        "credential",
    )
    contains_path = (
        re.search(r"(?i)(?:^|[\s'\"])[a-z]:[\\/]", message) is not None
        or re.search(r"(?:^|[\s'\"])/(?:[^/\s]+/)+", message) is not None
        or "\\\\" in message
    )
    if contains_path or any(marker in normalized for marker in sensitive_markers):
        return f"{type(error).__name__}: invalid local operation input."
    return "".join(
        " "
        if unicodedata.category(character) in {"Cc", "Cf", "Cs", "Zl", "Zp"}
        else character
        for character in message
    ).strip()


def has_errors(diagnostics: Any) -> bool:
    return any(
        item.severity == DiagnosticSeverity.ERROR
        for item in diagnostics
        if isinstance(item, Diagnostic)
    )


__all__ = [
    "AgentDiagnosticError",
    "DiagnosticCode",
    "exception_diagnostic",
    "has_errors",
    "make_diagnostic",
]
