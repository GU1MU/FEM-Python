"""Stable diagnostic codes and exception normalization."""

from __future__ import annotations

from dataclasses import dataclass
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


# Profile transforms use a deliberately small, provider-visible vocabulary.
# Keep these values independent from the broader ``DiagnosticCode`` enum: the
# latter is persisted as an upper-case contract while transform diagnostics are
# consumed directly by the geometry authoring tools.
PROFILE_TRANSFORM_DIAGNOSTIC_CODES = (
    "profile-transform.part-not-found",
    "profile-transform.source-not-planar",
    "profile-transform.source-not-strict",
    "profile-transform.no-material-profile",
    "profile-transform.ambiguous-material-profiles",
    "profile-transform.invalid-source-id",
    "profile-transform.nonpositive-height",
    "profile-transform.invalid-path",
    "profile-transform.unsupported-frame",
    "profile-transform.topology-unproven",
    "profile-transform.unexpected-body-count",
    "profile-transform.stale-context",
    "profile-transform.preflight-failed",
)

_PROFILE_TRANSFORM_DEFAULTS: dict[str, tuple[bool, tuple[str, ...]]] = {
    "profile-transform.part-not-found": (True, ("part_id",)),
    "profile-transform.source-not-planar": (False, ()),
    "profile-transform.source-not-strict": (False, ()),
    "profile-transform.no-material-profile": (False, ()),
    "profile-transform.ambiguous-material-profiles": (
        True,
        ("profile_selection",),
    ),
    "profile-transform.invalid-source-id": (
        True,
        ("profile_selection", "context_revision"),
    ),
    "profile-transform.nonpositive-height": (True, ("height",)),
    "profile-transform.invalid-path": (True, ("path",)),
    "profile-transform.unsupported-frame": (True, ("frame_strategy",)),
    "profile-transform.topology-unproven": (False, ()),
    "profile-transform.unexpected-body-count": (False, ()),
    "profile-transform.stale-context": (
        True,
        ("part_id", "context_revision"),
    ),
    "profile-transform.preflight-failed": (True, ()),
}

_PROFILE_TRANSFORM_MESSAGE_MAX_BYTES = 512
_PROFILE_TRANSFORM_OPERATION_MAX_BYTES = 96
_PROFILE_TRANSFORM_DETAIL_MAX_BYTES = 320
_PROFILE_TRANSFORM_FIELD_MAX_BYTES = 64
_PROFILE_TRANSFORM_FIELD_MAX_COUNT = 8
_PROFILE_TRANSFORM_RECOVERY_MAX_BYTES = 240


@dataclass(frozen=True, slots=True)
class ProfileTransformDiagnostic:
    """Bounded, typed recovery data returned by Profile transform tools."""

    code: str
    message: str
    operation: str
    retryable: bool
    required_fields: tuple[str, ...]
    preserve_draft: bool = True
    candidates: tuple[str, ...] = ()
    first_failed_member: str | None = None

    def __post_init__(self) -> None:
        code = str(self.code).strip()
        if code not in PROFILE_TRANSFORM_DIAGNOSTIC_CODES:
            raise ValueError(f"unknown Profile transform diagnostic code: {code!r}")
        object.__setattr__(self, "code", code)
        object.__setattr__(self, "message", _bounded_profile_text(self.message))
        operation = _bounded_profile_text(
            self.operation,
            max_length=_PROFILE_TRANSFORM_OPERATION_MAX_BYTES,
        )
        object.__setattr__(self, "operation", operation or "Profile transform")
        if type(self.retryable) is not bool:
            raise TypeError("retryable must be a boolean")
        if type(self.preserve_draft) is not bool:
            raise TypeError("preserve_draft must be a boolean")
        fields = _normalize_profile_fields(tuple(self.required_fields))
        object.__setattr__(self, "required_fields", fields)
        candidates = tuple(
            _bounded_profile_text(item, max_length=192)
            for item in self.candidates
            if _bounded_profile_text(item, max_length=192)
        )
        object.__setattr__(self, "candidates", candidates[:32])
        if self.first_failed_member is not None:
            member = _bounded_profile_text(self.first_failed_member, max_length=128)
            object.__setattr__(self, "first_failed_member", member or None)

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "code": self.code,
            "message": self.message,
            "operation": self.operation,
            "retryable": self.retryable,
            "required_fields": list(self.required_fields),
            "preserve_draft": self.preserve_draft,
        }
        if self.candidates:
            payload["candidates"] = list(self.candidates)
        if self.first_failed_member is not None:
            payload["first_failed_member"] = self.first_failed_member
        return payload


def _bounded_profile_text(value: object, *, max_length: int = 512) -> str:
    """Collapse untrusted local exception text to a provider-safe message."""

    text = " ".join(str(value).split())
    if re.search(r"(?i)(?:[a-z]:[\\/]|\\\\|/(?:[^/\\s]+/){2,})", text):
        text = "local operation details were omitted"
    encoded = text.encode("utf-8")
    if len(encoded) <= max_length:
        return text
    return encoded[:max_length].decode("utf-8", errors="ignore")


def _normalize_profile_fields(
    values: tuple[str, ...] | list[str] | tuple[object, ...] | list[object],
) -> tuple[str, ...]:
    """Bound and de-duplicate required field names before composing recovery."""

    fields: list[str] = []
    seen: set[str] = set()
    for value in values:
        field = _bounded_profile_text(
            value,
            max_length=_PROFILE_TRANSFORM_FIELD_MAX_BYTES,
        )
        if field and field not in seen:
            fields.append(field)
            seen.add(field)
        if len(fields) >= _PROFILE_TRANSFORM_FIELD_MAX_COUNT:
            break
    return tuple(fields)


def _profile_transform_recovery_text(
    fields: tuple[str, ...],
    *,
    retryable: bool,
) -> str:
    """Build a complete, independently bounded recovery instruction."""

    if not fields:
        if retryable:
            return "Next action: reread the current context and retry."
        return "Next action: revise the request or geometry to a supported boundary."

    prefix = "Next input: "
    suffix = "."
    remaining = _PROFILE_TRANSFORM_RECOVERY_MAX_BYTES - len(
        (prefix + suffix).encode("utf-8")
    )
    selected: list[str] = []
    for field in fields:
        separator = ", " if selected else ""
        remaining -= len(separator.encode("utf-8"))
        if remaining <= 0:
            break
        bounded = _bounded_profile_text(
            field,
            max_length=min(_PROFILE_TRANSFORM_FIELD_MAX_BYTES, remaining),
        )
        if not bounded:
            break
        selected.append(bounded)
        remaining -= len(bounded.encode("utf-8"))
    if not selected:
        return "Next input: required fields."
    return prefix + ", ".join(selected) + suffix


def profile_transform_diagnostic(
    code: str,
    *,
    operation: str,
    detail: str = "",
    retryable: bool | None = None,
    required_fields: tuple[str, ...] | list[str] | None = None,
    preserve_draft: bool = True,
    candidates: tuple[str, ...] | list[str] = (),
    first_failed_member: str | None = None,
) -> ProfileTransformDiagnostic:
    """Create one stable transform diagnostic and its recovery message."""

    normalized = str(code).strip()
    if normalized not in PROFILE_TRANSFORM_DIAGNOSTIC_CODES:
        raise ValueError(f"unknown Profile transform diagnostic code: {normalized!r}")
    default_retryable, default_fields = _PROFILE_TRANSFORM_DEFAULTS[normalized]
    fields = _normalize_profile_fields(
        tuple(required_fields) if required_fields is not None else default_fields
    )
    effective_retryable = default_retryable if retryable is None else retryable
    message_detail = _bounded_profile_text(
        detail,
        max_length=_PROFILE_TRANSFORM_DETAIL_MAX_BYTES,
    ) or normalized.rsplit(".", 1)[-1]
    operation_text = _bounded_profile_text(
        operation,
        max_length=_PROFILE_TRANSFORM_OPERATION_MAX_BYTES,
    ) or "Profile transform"
    recovery_text = _profile_transform_recovery_text(
        fields,
        retryable=effective_retryable,
    )
    message_prefix = f"{operation_text}: "
    detail_separator = ". "
    detail_budget = _PROFILE_TRANSFORM_MESSAGE_MAX_BYTES - len(
        (message_prefix + detail_separator + recovery_text).encode("utf-8")
    )
    if detail_budget > 0:
        message_detail = _bounded_profile_text(
            message_detail,
            max_length=min(_PROFILE_TRANSFORM_DETAIL_MAX_BYTES, detail_budget),
        )
        message = f"{message_prefix}{message_detail}{detail_separator}{recovery_text}"
    else:
        message = f"{message_prefix}{recovery_text}"
    return ProfileTransformDiagnostic(
        normalized,
        message,
        operation_text,
        effective_retryable,
        fields,
        preserve_draft,
        tuple(candidates),
        first_failed_member,
    )


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
    "PROFILE_TRANSFORM_DIAGNOSTIC_CODES",
    "ProfileTransformDiagnostic",
    "exception_diagnostic",
    "has_errors",
    "make_diagnostic",
    "profile_transform_diagnostic",
]
