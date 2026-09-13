"""Structured FEM Agent event contract and in-memory projector for the GUI.

This module has no Qt, Provider, or ``fem_agent`` dependencies. It validates serializable events,
projects safe display state, and replays that state from a complete event log.
"""

from __future__ import annotations

import json
import math
import re
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from types import MappingProxyType
from typing import Any, Iterable, Mapping


AGENT_EVENT_SCHEMA_VERSION = "1.0"
MAX_EVENT_PAYLOAD_BYTES = 65_536
MAX_SUMMARY_CHARACTERS = 240

_IDENTIFIER_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}")
_REVISION_HASH_PATTERN = re.compile(r"[0-9a-fA-F]{64}")
_SENSITIVE_KEY_PATTERN = re.compile(
    r"(?:api[_-]?key|authorization|cookie|credential|password|secret|token)",
    re.IGNORECASE,
)
_FILE_URI_PATTERN = re.compile(
    r"\bfile:///[^\r\n,;]*",
    re.IGNORECASE,
)
_WINDOWS_ABSOLUTE_PATH_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])(?:[A-Za-z]:[\\/]|\\\\)[^,;\r\n]*"
)
_POSIX_ABSOLUTE_PATH_PATTERN = re.compile(
    r"(?<![A-Za-z0-9./:])/(?!/)[^,;\r\n]*"
)
_SECRET_VALUE_PATTERNS = (
    re.compile(r"\bsk-[A-Za-z0-9_-]{8,}\b"),
    re.compile(r"\bBearer\s+[A-Za-z0-9._~+/-]{8,}=*\b", re.IGNORECASE),
    re.compile(
        r"\b(?:api[_-]?key|password|secret|token)\s*[:=]\s*[^,;\r\n]+",
        re.IGNORECASE,
    ),
)


class AgentEventError(ValueError):
    """An event violates the contract or cannot be applied to the current session state."""


class EventType(str, Enum):
    TURN_STARTED = "turn_started"
    CONTINUATION_STARTED = "continuation_started"
    MESSAGE_START = "message_start"
    MESSAGE_DELTA = "message_delta"
    MESSAGE_COMPLETE = "message_complete"
    TOOL_REQUESTED = "tool_requested"
    TOOL_STARTED = "tool_started"
    TOOL_RESULT = "tool_result"
    TOOL_WARNING = "tool_warning"
    TOOL_FAILED = "tool_failed"
    DIAGNOSTIC = "diagnostic"
    CONFIRMATION_REQUESTED = "confirmation_requested"
    PROPOSAL_REQUESTED = "proposal_requested"
    PROPOSAL_ACCEPTED = "proposal_accepted"
    PROPOSAL_REJECTED = "proposal_rejected"
    PROPOSAL_STALE = "proposal_stale"
    PROPOSAL_STARTED = "proposal_started"
    PROPOSAL_PROGRESS = "proposal_progress"
    PROPOSAL_SUCCEEDED = "proposal_succeeded"
    PROPOSAL_FAILED = "proposal_failed"
    PROPOSAL_CANCELLED = "proposal_cancelled"
    TURN_CANCELLED = "turn_cancelled"
    TURN_COMPLETE = "turn_complete"
    TURN_FAILED = "turn_failed"


class TurnStatus(str, Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


class MessageStatus(str, Enum):
    STREAMING = "streaming"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    INTERRUPTED = "interrupted"


class MessagePresentationKind(str, Enum):
    PROCESS = "process"
    PROPOSAL_PREVIEW = "proposal_preview"
    PATCH_PREVIEW = "patch_preview"
    DECISION_REQUEST = "decision_request"
    RESULT_SUMMARY = "result_summary"


class ToolStatus(str, Enum):
    REQUESTED = "requested"
    RUNNING = "running"
    COMPLETED = "completed"
    WARNING = "warning"
    FAILED = "failed"
    CANCELLED = "cancelled"


class DiagnosticSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    BLOCKING = "blocking"


class TimelineKind(str, Enum):
    MESSAGE = "message"
    TOOL_GROUP = "tool_group"
    DIAGNOSTIC = "diagnostic"
    CONFIRMATION = "confirmation"
    PROPOSAL = "proposal"


class ProposalViewStatus(str, Enum):
    PENDING_CONFIRMATION = "pending_confirmation"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    STALE = "stale"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


_REQUIRED_PAYLOAD_FIELDS: dict[EventType, frozenset[str]] = {
    EventType.TURN_STARTED: frozenset({"user_message"}),
    EventType.CONTINUATION_STARTED: frozenset(
        {
            "proposal_id",
            "proposal_hash",
            "source_turn_id",
            "status",
        }
    ),
    EventType.MESSAGE_START: frozenset(
        {"message_id", "role", "format"}
    ),
    EventType.MESSAGE_DELTA: frozenset({"message_id", "delta"}),
    EventType.MESSAGE_COMPLETE: frozenset({"message_id"}),
    EventType.TOOL_REQUESTED: frozenset(
        {"call_id", "tool_name", "display_name", "request"}
    ),
    EventType.TOOL_STARTED: frozenset({"call_id"}),
    EventType.TOOL_RESULT: frozenset(
        {"call_id", "result", "duration_ms"}
    ),
    EventType.TOOL_WARNING: frozenset(
        {"call_id", "warning", "duration_ms"}
    ),
    EventType.TOOL_FAILED: frozenset(
        {"call_id", "error", "duration_ms"}
    ),
    EventType.DIAGNOSTIC: frozenset(
        {"diagnostic_id", "title", "message", "severity"}
    ),
    EventType.CONFIRMATION_REQUESTED: frozenset(
        {
            "confirmation_id",
            "title",
            "summary",
            "revision",
            "revision_hash",
        }
    ),
    EventType.PROPOSAL_REQUESTED: frozenset(
        {
            "proposal_id",
            "proposal_hash",
            "proposal_kind",
            "title",
            "summary",
            "impact",
            "confirm_label",
            "target_document_id",
            "target_session_id",
            "base_session_revision",
        }
    ),
    EventType.PROPOSAL_ACCEPTED: frozenset(
        {"proposal_id", "proposal_hash"}
    ),
    EventType.PROPOSAL_REJECTED: frozenset(
        {"proposal_id", "proposal_hash", "reason"}
    ),
    EventType.PROPOSAL_STALE: frozenset(
        {"proposal_id", "proposal_hash", "reason"}
    ),
    EventType.PROPOSAL_STARTED: frozenset(
        {"proposal_id", "proposal_hash"}
    ),
    EventType.PROPOSAL_PROGRESS: frozenset(
        {"proposal_id", "proposal_hash", "progress", "message"}
    ),
    EventType.PROPOSAL_SUCCEEDED: frozenset(
        {"proposal_id", "proposal_hash", "summary"}
    ),
    EventType.PROPOSAL_FAILED: frozenset(
        {"proposal_id", "proposal_hash", "reason"}
    ),
    EventType.PROPOSAL_CANCELLED: frozenset(
        {"proposal_id", "proposal_hash", "reason"}
    ),
    EventType.TURN_CANCELLED: frozenset({"reason"}),
    EventType.TURN_COMPLETE: frozenset(),
    EventType.TURN_FAILED: frozenset({"reason"}),
}

_OPTIONAL_PAYLOAD_FIELDS: dict[EventType, frozenset[str]] = {
    EventType.TURN_STARTED: frozenset(),
    EventType.CONTINUATION_STARTED: frozenset(),
    EventType.MESSAGE_START: frozenset({"presentation_kind"}),
    EventType.MESSAGE_DELTA: frozenset(),
    EventType.MESSAGE_COMPLETE: frozenset({"presentation_kind"}),
    EventType.TOOL_REQUESTED: frozenset(),
    EventType.TOOL_STARTED: frozenset(),
    EventType.TOOL_RESULT: frozenset(),
    EventType.TOOL_WARNING: frozenset({"result"}),
    EventType.TOOL_FAILED: frozenset({"diagnostic"}),
    EventType.DIAGNOSTIC: frozenset({"code"}),
    EventType.CONFIRMATION_REQUESTED: frozenset(),
    EventType.PROPOSAL_REQUESTED: frozenset(),
    EventType.PROPOSAL_ACCEPTED: frozenset(),
    EventType.PROPOSAL_REJECTED: frozenset(),
    EventType.PROPOSAL_STALE: frozenset(),
    EventType.PROPOSAL_STARTED: frozenset(),
    EventType.PROPOSAL_PROGRESS: frozenset(),
    EventType.PROPOSAL_SUCCEEDED: frozenset(),
    EventType.PROPOSAL_FAILED: frozenset(),
    EventType.PROPOSAL_CANCELLED: frozenset(),
    EventType.TURN_CANCELLED: frozenset(),
    EventType.TURN_COMPLETE: frozenset(),
    EventType.TURN_FAILED: frozenset(),
}


def _is_plain_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _require_identifier(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER_PATTERN.fullmatch(value):
        raise AgentEventError(f"{field_name} is not a valid identifier")
    return value


def _require_string(
    value: object,
    field_name: str,
    *,
    allow_empty: bool = False,
) -> str:
    if not isinstance(value, str):
        raise AgentEventError(f"{field_name} must be a string")
    if not allow_empty and not value.strip():
        raise AgentEventError(f"{field_name} must not be empty")
    return value


def _require_duration(value: object) -> float:
    if not _is_number(value) or float(value) < 0:
        raise AgentEventError("duration_ms must be a finite non-negative number")
    return float(value)


def _validate_timestamp(value: object) -> str:
    timestamp = _require_string(value, "timestamp")
    normalized = timestamp[:-1] + "+00:00" if timestamp.endswith("Z") else timestamp
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise AgentEventError("timestamp must be an ISO-8601 time") from exc
    if parsed.tzinfo is None:
        raise AgentEventError("timestamp must include a timezone")
    return timestamp


def _serialized_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    try:
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
    except (RecursionError, TypeError, ValueError) as exc:
        raise AgentEventError("payload must contain only JSON-serializable values") from exc
    if len(encoded.encode("utf-8")) > MAX_EVENT_PAYLOAD_BYTES:
        raise AgentEventError("payload exceeds the event size limit")
    decoded = json.loads(encoded)
    if not isinstance(decoded, dict):
        raise AgentEventError("payload must be an object")
    return decoded


def _freeze_json(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType(
            {key: _freeze_json(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _thaw_json(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_thaw_json(item) for item in value]
    return value


def _validate_payload(event_type: EventType, payload: Mapping[str, Any]) -> None:
    required = _REQUIRED_PAYLOAD_FIELDS[event_type]
    optional = _OPTIONAL_PAYLOAD_FIELDS[event_type]
    keys = frozenset(payload)
    missing = required - keys
    unknown = keys - required - optional
    if missing:
        raise AgentEventError(
            f"{event_type.value} is missing payload fields: {', '.join(sorted(missing))}"
        )
    if unknown:
        raise AgentEventError(
            f"{event_type.value} contains unknown payload fields: "
            f"{', '.join(sorted(unknown))}"
        )

    if event_type is EventType.TURN_STARTED:
        _require_string(payload["user_message"], "user_message")
        return
    if event_type is EventType.CONTINUATION_STARTED:
        _require_identifier(payload["proposal_id"], "proposal_id")
        _require_identifier(payload["source_turn_id"], "source_turn_id")
        proposal_hash = payload["proposal_hash"]
        if (
            not isinstance(proposal_hash, str)
            or not _REVISION_HASH_PATTERN.fullmatch(proposal_hash)
        ):
            raise AgentEventError(
                "proposal_hash must be a full hexadecimal SHA-256 value"
            )
        if payload["status"] not in {
            "succeeded",
            "rejected",
            "failed",
            "stale",
        }:
            raise AgentEventError("continuation status is not a resumable terminal state")
        return
    if event_type is EventType.MESSAGE_START:
        _require_identifier(payload["message_id"], "message_id")
        if payload["role"] != "assistant":
            raise AgentEventError("Phase 3 message role must be assistant")
        if payload["format"] != "restricted_markdown":
            raise AgentEventError("Message format must be restricted_markdown")
        try:
            MessagePresentationKind(
                payload.get(
                    "presentation_kind",
                    MessagePresentationKind.PROCESS.value,
                )
            )
        except (TypeError, ValueError) as error:
            raise AgentEventError("Invalid message presentation_kind") from error
        return
    if event_type is EventType.MESSAGE_DELTA:
        _require_identifier(payload["message_id"], "message_id")
        delta = payload["delta"]
        if not isinstance(delta, str):
            raise AgentEventError("delta must be a string")
        if delta == "":
            raise AgentEventError("delta must not be empty")
        return
    if event_type is EventType.MESSAGE_COMPLETE:
        _require_identifier(payload["message_id"], "message_id")
        try:
            MessagePresentationKind(
                payload.get(
                    "presentation_kind",
                    MessagePresentationKind.PROCESS.value,
                )
            )
        except (TypeError, ValueError) as error:
            raise AgentEventError("Invalid message presentation_kind") from error
        return
    if event_type is EventType.TOOL_REQUESTED:
        _require_identifier(payload["call_id"], "call_id")
        _require_identifier(payload["tool_name"], "tool_name")
        _require_string(payload["display_name"], "display_name")
        return
    if event_type is EventType.TOOL_STARTED:
        _require_identifier(payload["call_id"], "call_id")
        return
    if event_type in {
        EventType.TOOL_RESULT,
        EventType.TOOL_WARNING,
        EventType.TOOL_FAILED,
    }:
        _require_identifier(payload["call_id"], "call_id")
        _require_duration(payload["duration_ms"])
        if event_type is EventType.TOOL_WARNING:
            _require_string(payload["warning"], "warning")
        elif event_type is EventType.TOOL_FAILED:
            _require_string(payload["error"], "error")
            if "diagnostic" in payload:
                _require_string(payload["diagnostic"], "diagnostic")
        return
    if event_type is EventType.DIAGNOSTIC:
        _require_identifier(payload["diagnostic_id"], "diagnostic_id")
        _require_string(payload["title"], "title")
        _require_string(payload["message"], "message")
        try:
            DiagnosticSeverity(payload["severity"])
        except (TypeError, ValueError) as exc:
            raise AgentEventError("severity is not a known diagnostic level") from exc
        if "code" in payload:
            _require_identifier(payload["code"], "code")
        return
    if event_type is EventType.CONFIRMATION_REQUESTED:
        _require_identifier(payload["confirmation_id"], "confirmation_id")
        _require_string(payload["title"], "title")
        _require_string(payload["summary"], "summary")
        if not _is_plain_int(payload["revision"]) or payload["revision"] < 0:
            raise AgentEventError("revision must be a non-negative integer")
        revision_hash = payload["revision_hash"]
        if (
            not isinstance(revision_hash, str)
            or not _REVISION_HASH_PATTERN.fullmatch(revision_hash)
        ):
            raise AgentEventError("revision_hash must be a full hexadecimal SHA-256 value")
        return
    proposal_events = {
        EventType.PROPOSAL_REQUESTED,
        EventType.PROPOSAL_ACCEPTED,
        EventType.PROPOSAL_REJECTED,
        EventType.PROPOSAL_STALE,
        EventType.PROPOSAL_STARTED,
        EventType.PROPOSAL_PROGRESS,
        EventType.PROPOSAL_SUCCEEDED,
        EventType.PROPOSAL_FAILED,
        EventType.PROPOSAL_CANCELLED,
    }
    if event_type in proposal_events:
        _require_identifier(payload["proposal_id"], "proposal_id")
        proposal_hash = payload["proposal_hash"]
        if (
            not isinstance(proposal_hash, str)
            or not _REVISION_HASH_PATTERN.fullmatch(proposal_hash)
        ):
            raise AgentEventError(
                "proposal_hash must be a full hexadecimal SHA-256 value"
            )
        if event_type is EventType.PROPOSAL_REQUESTED:
            if payload["proposal_kind"] not in {
                "geometry",
                "mesh",
                "solve",
                "destructive_edit",
                "requirement_review",
                "project_save",
            }:
                raise AgentEventError("proposal_kind is not a known type")
            for field_name in (
                "title",
                "summary",
                "impact",
                "confirm_label",
            ):
                _require_string(payload[field_name], field_name)
            _require_identifier(
                payload["target_document_id"],
                "target_document_id",
            )
            _require_identifier(
                payload["target_session_id"],
                "target_session_id",
            )
            revision = payload["base_session_revision"]
            if not _is_plain_int(revision) or revision < 0:
                raise AgentEventError(
                    "base_session_revision must be a non-negative integer"
                )
        elif event_type is EventType.PROPOSAL_PROGRESS:
            progress = payload["progress"]
            if not _is_number(progress) or not 0.0 <= float(progress) <= 1.0:
                raise AgentEventError("proposal progress must be between 0 and 1")
            _require_string(payload["message"], "message")
        elif event_type is EventType.PROPOSAL_SUCCEEDED:
            _require_string(payload["summary"], "summary")
        elif event_type in {
            EventType.PROPOSAL_REJECTED,
            EventType.PROPOSAL_STALE,
            EventType.PROPOSAL_FAILED,
            EventType.PROPOSAL_CANCELLED,
        }:
            _require_string(payload["reason"], "reason")
        return
    if event_type in {
        EventType.TURN_CANCELLED,
        EventType.TURN_FAILED,
    }:
        _require_string(payload["reason"], "reason")


@dataclass(frozen=True)
class AgentEvent:
    """A serializable, self-validating GUI Agent event."""

    schema_version: str
    event_id: str
    session_id: str
    turn_id: str
    sequence: int
    event_type: EventType
    timestamp: str
    payload: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.schema_version != AGENT_EVENT_SCHEMA_VERSION:
            raise AgentEventError(
                f"Unsupported schema_version: {self.schema_version}"
            )
        _require_identifier(self.event_id, "event_id")
        _require_identifier(self.session_id, "session_id")
        _require_identifier(self.turn_id, "turn_id")
        if not _is_plain_int(self.sequence) or self.sequence < 1:
            raise AgentEventError("sequence must be an integer starting at 1")
        try:
            event_type = EventType(self.event_type)
        except (TypeError, ValueError) as exc:
            raise AgentEventError("Unknown event_type") from exc
        object.__setattr__(self, "event_type", event_type)
        object.__setattr__(self, "timestamp", _validate_timestamp(self.timestamp))
        if not isinstance(self.payload, Mapping):
            raise AgentEventError("payload must be an object")
        safe_payload = _serialized_payload(self.payload)
        _validate_payload(event_type, safe_payload)
        object.__setattr__(self, "payload", _freeze_json(safe_payload))

    @classmethod
    def create(
        cls,
        *,
        event_id: str,
        session_id: str,
        turn_id: str,
        sequence: int,
        event_type: EventType,
        payload: Mapping[str, Any],
        timestamp: str | None = None,
    ) -> AgentEvent:
        current_timestamp = timestamp or (
            datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        )
        return cls(
            schema_version=AGENT_EVENT_SCHEMA_VERSION,
            event_id=event_id,
            session_id=session_id,
            turn_id=turn_id,
            sequence=sequence,
            event_type=event_type,
            timestamp=current_timestamp,
            payload=payload,
        )

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> AgentEvent:
        if not isinstance(record, Mapping):
            raise AgentEventError("Event record must be an object")
        required = {
            "schema_version",
            "event_id",
            "session_id",
            "turn_id",
            "sequence",
            "event_type",
            "timestamp",
            "payload",
        }
        missing = required - set(record)
        unknown = set(record) - required
        if missing:
            raise AgentEventError(
                f"Event is missing fields: {', '.join(sorted(missing))}"
            )
        if unknown:
            raise AgentEventError(
                f"Event contains unknown fields: {', '.join(sorted(unknown))}"
            )
        return cls(
            schema_version=record["schema_version"],
            event_id=record["event_id"],
            session_id=record["session_id"],
            turn_id=record["turn_id"],
            sequence=record["sequence"],
            event_type=record["event_type"],
            timestamp=record["timestamp"],
            payload=record["payload"],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "event_id": self.event_id,
            "session_id": self.session_id,
            "turn_id": self.turn_id,
            "sequence": self.sequence,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp,
            "payload": _thaw_json(self.payload),
        }


def _redact_text(value: str) -> str:
    redacted = value
    for pattern in _SECRET_VALUE_PATTERNS:
        redacted = pattern.sub("<sensitive information hidden>", redacted)
    return redact_absolute_paths(redacted)


def redact_absolute_paths(value: str) -> str:
    """Hide local absolute paths while preserving other message text."""

    if not isinstance(value, str):
        raise TypeError("value must be a string")
    redacted = value
    redacted = _FILE_URI_PATTERN.sub(
        "<absolute path hidden>",
        redacted,
    )
    redacted = _WINDOWS_ABSOLUTE_PATH_PATTERN.sub(
        "<absolute path hidden>",
        redacted,
    )
    redacted = _POSIX_ABSOLUTE_PATH_PATTERN.sub(
        "<absolute path hidden>",
        redacted,
    )
    return redacted


def safe_tool_summary(
    value: object,
    *,
    max_characters: int = MAX_SUMMARY_CHARACTERS,
) -> str:
    """Summarize any tool value as bounded, redacted text without calling object ``repr``."""

    def summarize(item: object, depth: int) -> str:
        if item is None:
            return "null"
        if isinstance(item, bool):
            return "true" if item else "false"
        if isinstance(item, (int, float)):
            if isinstance(item, float) and not math.isfinite(item):
                return "<non-finite number>"
            return str(item)
        if isinstance(item, str):
            return _redact_text(item)
        if depth >= 3:
            return "…"
        if isinstance(item, Mapping):
            parts: list[str] = []
            for index, (key, child) in enumerate(item.items()):
                if index >= 8:
                    parts.append("…")
                    break
                safe_key = (
                    _redact_text(key)
                    if isinstance(key, str)
                    else "<non-string key>"
                )
                if isinstance(key, str) and _SENSITIVE_KEY_PATTERN.search(key):
                    child_text = "<sensitive information hidden>"
                else:
                    child_text = summarize(child, depth + 1)
                parts.append(f"{safe_key}={child_text}")
            return ", ".join(parts) if parts else "{}"
        if isinstance(item, (list, tuple)):
            values = [
                summarize(child, depth + 1)
                for child in item[:8]
            ]
            if len(item) > 8:
                values.append("…")
            return "[" + ", ".join(values) + "]"
        return "<unsupported value>"

    limit = max(24, int(max_characters))
    summary = summarize(value, 0).replace("\r", " ").replace("\n", " ")
    summary = " ".join(summary.split())
    if len(summary) > limit:
        return summary[: limit - 1] + "…"
    return summary


@dataclass
class MessageView:
    message_id: str
    role: str
    format: str
    presentation_kind: MessagePresentationKind = MessagePresentationKind.PROCESS
    text: str = ""
    status: MessageStatus = MessageStatus.STREAMING
    _text_chunks: list[str] = field(
        default_factory=list,
        repr=False,
        compare=False,
    )

    def append_delta(self, delta: str) -> None:
        self._text_chunks.append(delta)

    def materialize_text(self) -> str:
        if self._text_chunks:
            self.text = "".join((self.text, *self._text_chunks))
            self._text_chunks.clear()
        return self.text


@dataclass
class ToolActivityView:
    call_id: str
    tool_name: str
    display_name: str
    request_summary: str
    status: ToolStatus = ToolStatus.REQUESTED
    result_summary: str = ""
    diagnostics: list[str] = field(default_factory=list)
    duration_ms: float = 0.0


@dataclass
class ToolGroupView:
    group_id: str
    calls: list[ToolActivityView] = field(default_factory=list)
    expanded: bool = False

    @property
    def completed_count(self) -> int:
        return sum(
            call.status is ToolStatus.COMPLETED for call in self.calls
        )

    @property
    def warning_count(self) -> int:
        return sum(call.status is ToolStatus.WARNING for call in self.calls)

    @property
    def failed_count(self) -> int:
        return sum(call.status is ToolStatus.FAILED for call in self.calls)

    @property
    def cancelled_count(self) -> int:
        return sum(call.status is ToolStatus.CANCELLED for call in self.calls)

    @property
    def total_duration_ms(self) -> float:
        return sum(call.duration_ms for call in self.calls)


@dataclass
class DiagnosticView:
    diagnostic_id: str
    title: str
    message: str
    severity: DiagnosticSeverity
    code: str = ""


@dataclass
class ConfirmationView:
    confirmation_id: str
    title: str
    summary: str
    revision: int
    revision_hash: str
    authorized: bool = False


@dataclass
class ProposalView:
    proposal_id: str
    proposal_hash: str
    proposal_kind: str
    title: str
    summary: str
    impact: str
    confirm_label: str
    target_document_id: str
    target_session_id: str
    base_session_revision: int
    status: ProposalViewStatus = ProposalViewStatus.PENDING_CONFIRMATION
    progress: float = 0.0
    status_message: str = ""
    authorized: bool = False


@dataclass
class TimelineItem:
    kind: TimelineKind
    item_id: str


@dataclass
class TurnView:
    turn_id: str
    user_message: str
    status: TurnStatus = TurnStatus.RUNNING
    messages: list[MessageView] = field(default_factory=list)
    tool_groups: list[ToolGroupView] = field(default_factory=list)
    diagnostics: list[DiagnosticView] = field(default_factory=list)
    confirmations: list[ConfirmationView] = field(default_factory=list)
    proposals: list[ProposalView] = field(default_factory=list)
    timeline: list[TimelineItem] = field(default_factory=list)
    failure_reason: str = ""


@dataclass
class SessionPresentation:
    schema_version: str = AGENT_EVENT_SCHEMA_VERSION
    session_id: str = ""
    last_sequence: int = 0
    turns: list[TurnView] = field(default_factory=list)

    def to_snapshot(self) -> dict[str, Any]:
        """Return a serializable session snapshot containing only safe display fields."""

        def convert(value: object) -> object:
            if isinstance(value, Enum):
                return value.value
            if isinstance(value, dict):
                return {
                    str(key): convert(item)
                    for key, item in value.items()
                    if not str(key).startswith("_")
                }
            if isinstance(value, list):
                return [convert(item) for item in value]
            return value

        return convert(asdict(self))


class AgentEventProjector:
    """Project events into chat display state in strict session-wide sequence order."""

    def __init__(self) -> None:
        self._presentation = SessionPresentation()
        self._events: list[AgentEvent] = []
        self._event_ids: set[str] = set()
        self._active_turn_id: str | None = None
        self._last_timeline_kind: TimelineKind | None = None

    @property
    def presentation(self) -> SessionPresentation:
        self._materialize_message_text()
        return deepcopy(self._presentation)

    @property
    def presentation_view(self) -> SessionPresentation:
        """Return the internal projection for immediate reads by the GUI owner thread only."""
        self._materialize_message_text()
        return self._presentation

    @property
    def last_sequence(self) -> int:
        return self._presentation.last_sequence

    def export_event_log(self) -> tuple[dict[str, Any], ...]:
        return tuple(event.to_dict() for event in self._events)

    @classmethod
    def replay(cls, events: Iterable[AgentEvent]) -> AgentEventProjector:
        projector = cls()
        for event in events:
            projector.apply_in_place(event)
        return projector

    @classmethod
    def restore_event_log(
        cls,
        records: Iterable[Mapping[str, Any]],
    ) -> AgentEventProjector:
        return cls.replay(AgentEvent.from_dict(record) for record in records)

    def apply(self, event: AgentEvent) -> SessionPresentation:
        self.apply_in_place(event)
        return self.presentation

    def apply_in_place(self, event: AgentEvent) -> SessionPresentation:
        """Apply an event and return the internal projection without deep copies on the GUI hot path."""
        if not isinstance(event, AgentEvent):
            raise AgentEventError("projector only accepts AgentEvent")
        if event.event_id in self._event_ids:
            raise AgentEventError(f"Duplicate event_id: {event.event_id}")
        expected_sequence = self._presentation.last_sequence + 1
        if event.sequence != expected_sequence:
            raise AgentEventError(
                f"Expected sequence {expected_sequence}, received {event.sequence}"
            )
        if (
            self._presentation.session_id
            and event.session_id != self._presentation.session_id
        ):
            raise AgentEventError("Event crosses the current session")
        if event.event_type in {
            EventType.TURN_STARTED,
            EventType.CONTINUATION_STARTED,
        }:
            self._apply_turn_started(event)
        elif event.event_type in {
            EventType.PROPOSAL_ACCEPTED,
            EventType.PROPOSAL_REJECTED,
            EventType.PROPOSAL_STALE,
            EventType.PROPOSAL_STARTED,
            EventType.PROPOSAL_PROGRESS,
            EventType.PROPOSAL_SUCCEEDED,
            EventType.PROPOSAL_FAILED,
            EventType.PROPOSAL_CANCELLED,
        }:
            turn = self._find_turn(event.turn_id)
            self._apply_proposal_lifecycle(turn, event)
        else:
            turn = self._require_active_turn(event)
            self._apply_turn_event(turn, event)

        if not self._presentation.session_id:
            self._presentation.session_id = event.session_id
        self._presentation.last_sequence = event.sequence
        self._event_ids.add(event.event_id)
        self._events.append(event)
        return self._presentation

    def message_view(self, message_id: str) -> MessageView:
        """Return the internal message view for incremental widget updates by the owner thread."""
        for turn in reversed(self._presentation.turns):
            for message in reversed(turn.messages):
                if message.message_id == message_id:
                    message.materialize_text()
                    return message
        raise AgentEventError(f"Unknown message_id: {message_id}")

    def _materialize_message_text(self) -> None:
        for turn in self._presentation.turns:
            for message in turn.messages:
                message.materialize_text()

    def _apply_turn_started(self, event: AgentEvent) -> None:
        if self._active_turn_id is not None:
            raise AgentEventError("Previous turn has not ended")
        if any(turn.turn_id == event.turn_id for turn in self._presentation.turns):
            raise AgentEventError("turn_id already exists and cannot restart")
        turn = TurnView(
            turn_id=event.turn_id,
            user_message=(
                safe_tool_summary(
                    event.payload["user_message"],
                    max_characters=4_000,
                )
                if event.event_type is EventType.TURN_STARTED
                else ""
            ),
        )
        self._presentation.turns.append(turn)
        self._active_turn_id = event.turn_id
        self._last_timeline_kind = None

    def _require_active_turn(self, event: AgentEvent) -> TurnView:
        if self._active_turn_id is None:
            raise AgentEventError("No turn is currently running")
        if event.turn_id != self._active_turn_id:
            raise AgentEventError("Event crosses the current turn")
        return self._presentation.turns[-1]

    def _find_turn(self, turn_id: str) -> TurnView:
        for turn in self._presentation.turns:
            if turn.turn_id == turn_id:
                return turn
        raise AgentEventError("proposal event references an unknown turn")

    def _apply_turn_event(self, turn: TurnView, event: AgentEvent) -> None:
        event_type = event.event_type
        if event_type is EventType.MESSAGE_START:
            self._message_start(turn, event)
        elif event_type is EventType.MESSAGE_DELTA:
            self._message_delta(turn, event)
        elif event_type is EventType.MESSAGE_COMPLETE:
            self._message_complete(turn, event)
        elif event_type is EventType.TOOL_REQUESTED:
            self._tool_requested(turn, event)
        elif event_type is EventType.TOOL_STARTED:
            self._tool_started(turn, event)
        elif event_type in {
            EventType.TOOL_RESULT,
            EventType.TOOL_WARNING,
            EventType.TOOL_FAILED,
        }:
            self._tool_terminal(turn, event)
        elif event_type is EventType.DIAGNOSTIC:
            self._diagnostic(turn, event)
        elif event_type is EventType.CONFIRMATION_REQUESTED:
            self._confirmation(turn, event)
        elif event_type is EventType.PROPOSAL_REQUESTED:
            self._proposal_requested(turn, event)
        elif event_type is EventType.TURN_COMPLETE:
            self._turn_complete(turn)
        elif event_type is EventType.TURN_CANCELLED:
            self._turn_cancelled(turn, event)
        elif event_type is EventType.TURN_FAILED:
            self._turn_failed(turn, event)

    @staticmethod
    def _find_message(turn: TurnView, message_id: str) -> MessageView:
        for message in turn.messages:
            if message.message_id == message_id:
                return message
        raise AgentEventError(f"Unknown message_id: {message_id}")

    @staticmethod
    def _find_tool(turn: TurnView, call_id: str) -> ToolActivityView:
        for group in turn.tool_groups:
            for call in group.calls:
                if call.call_id == call_id:
                    return call
        raise AgentEventError(f"Unknown call_id: {call_id}")

    def _message_start(self, turn: TurnView, event: AgentEvent) -> None:
        message_id = event.payload["message_id"]
        if any(message.message_id == message_id for message in turn.messages):
            raise AgentEventError("message_id already exists")
        message = MessageView(
            message_id=message_id,
            role=event.payload["role"],
            format=event.payload["format"],
            presentation_kind=MessagePresentationKind(
                event.payload.get(
                    "presentation_kind",
                    MessagePresentationKind.PROCESS.value,
                )
            ),
        )
        turn.messages.append(message)
        turn.timeline.append(
            TimelineItem(TimelineKind.MESSAGE, message.message_id)
        )
        self._last_timeline_kind = TimelineKind.MESSAGE

    def _message_delta(self, turn: TurnView, event: AgentEvent) -> None:
        message = self._find_message(turn, event.payload["message_id"])
        if message.status is not MessageStatus.STREAMING:
            raise AgentEventError("Cannot append delta to a completed message")
        message.append_delta(event.payload["delta"])
        self._last_timeline_kind = TimelineKind.MESSAGE

    def _message_complete(self, turn: TurnView, event: AgentEvent) -> None:
        message = self._find_message(turn, event.payload["message_id"])
        if message.status is not MessageStatus.STREAMING:
            raise AgentEventError("Message has already ended")
        message.materialize_text()
        if "presentation_kind" in event.payload:
            message.presentation_kind = MessagePresentationKind(
                event.payload["presentation_kind"]
            )
        message.status = MessageStatus.COMPLETED
        self._last_timeline_kind = TimelineKind.MESSAGE

    def _tool_requested(self, turn: TurnView, event: AgentEvent) -> None:
        call_id = event.payload["call_id"]
        try:
            self._find_tool(turn, call_id)
        except AgentEventError:
            pass
        else:
            raise AgentEventError("call_id already exists")

        if (
            self._last_timeline_kind is TimelineKind.TOOL_GROUP
            and turn.tool_groups
        ):
            group = turn.tool_groups[-1]
        else:
            group = ToolGroupView(
                group_id=f"{turn.turn_id}:tools:{len(turn.tool_groups) + 1}"
            )
            turn.tool_groups.append(group)
            turn.timeline.append(
                TimelineItem(TimelineKind.TOOL_GROUP, group.group_id)
            )
        group.calls.append(
            ToolActivityView(
                call_id=call_id,
                tool_name=event.payload["tool_name"],
                display_name=safe_tool_summary(
                    event.payload["display_name"],
                    max_characters=80,
                ),
                request_summary=safe_tool_summary(event.payload["request"]),
            )
        )
        self._last_timeline_kind = TimelineKind.TOOL_GROUP

    def _tool_started(self, turn: TurnView, event: AgentEvent) -> None:
        call = self._find_tool(turn, event.payload["call_id"])
        if call.status is not ToolStatus.REQUESTED:
            raise AgentEventError("Tools can only enter running from requested")
        call.status = ToolStatus.RUNNING

    def _tool_terminal(self, turn: TurnView, event: AgentEvent) -> None:
        call = self._find_tool(turn, event.payload["call_id"])
        if call.status is not ToolStatus.RUNNING:
            raise AgentEventError("Terminal tool events require a running tool")
        call.duration_ms = _require_duration(event.payload["duration_ms"])
        if event.event_type is EventType.TOOL_RESULT:
            call.status = ToolStatus.COMPLETED
            call.result_summary = safe_tool_summary(event.payload["result"])
        elif event.event_type is EventType.TOOL_WARNING:
            call.status = ToolStatus.WARNING
            call.diagnostics.append(
                safe_tool_summary(event.payload["warning"])
            )
            if "result" in event.payload:
                call.result_summary = safe_tool_summary(
                    event.payload["result"]
                )
        else:
            call.status = ToolStatus.FAILED
            call.result_summary = safe_tool_summary(event.payload["error"])
            if "diagnostic" in event.payload:
                call.diagnostics.append(
                    safe_tool_summary(event.payload["diagnostic"])
                )

    def _diagnostic(self, turn: TurnView, event: AgentEvent) -> None:
        diagnostic_id = event.payload["diagnostic_id"]
        if any(
            diagnostic.diagnostic_id == diagnostic_id
            for diagnostic in turn.diagnostics
        ):
            raise AgentEventError("diagnostic_id already exists")
        diagnostic = DiagnosticView(
            diagnostic_id=diagnostic_id,
            title=safe_tool_summary(
                event.payload["title"],
                max_characters=100,
            ),
            message=safe_tool_summary(
                event.payload["message"],
                max_characters=1_000,
            ),
            severity=DiagnosticSeverity(event.payload["severity"]),
            code=event.payload.get("code", ""),
        )
        turn.diagnostics.append(diagnostic)
        turn.timeline.append(
            TimelineItem(TimelineKind.DIAGNOSTIC, diagnostic_id)
        )
        self._last_timeline_kind = TimelineKind.DIAGNOSTIC

    def _confirmation(self, turn: TurnView, event: AgentEvent) -> None:
        confirmation_id = event.payload["confirmation_id"]
        if any(
            confirmation.confirmation_id == confirmation_id
            for confirmation in turn.confirmations
        ):
            raise AgentEventError("confirmation_id already exists")
        confirmation = ConfirmationView(
            confirmation_id=confirmation_id,
            title=safe_tool_summary(
                event.payload["title"],
                max_characters=100,
            ),
            summary=safe_tool_summary(
                event.payload["summary"],
                max_characters=1_000,
            ),
            revision=event.payload["revision"],
            revision_hash=event.payload["revision_hash"],
        )
        turn.confirmations.append(confirmation)
        turn.timeline.append(
            TimelineItem(TimelineKind.CONFIRMATION, confirmation_id)
        )
        self._last_timeline_kind = TimelineKind.CONFIRMATION

    def _proposal_requested(
        self,
        turn: TurnView,
        event: AgentEvent,
    ) -> None:
        proposal_id = event.payload["proposal_id"]
        if any(item.proposal_id == proposal_id for item in turn.proposals):
            raise AgentEventError("proposal_id already exists")
        proposal = ProposalView(
            proposal_id=proposal_id,
            proposal_hash=event.payload["proposal_hash"],
            proposal_kind=event.payload["proposal_kind"],
            title=safe_tool_summary(
                event.payload["title"],
                max_characters=100,
            ),
            summary=safe_tool_summary(
                event.payload["summary"],
                max_characters=1_000,
            ),
            impact=safe_tool_summary(
                event.payload["impact"],
                max_characters=1_000,
            ),
            confirm_label=safe_tool_summary(
                event.payload["confirm_label"],
                max_characters=40,
            ),
            target_document_id=event.payload["target_document_id"],
            target_session_id=event.payload["target_session_id"],
            base_session_revision=event.payload["base_session_revision"],
        )
        turn.proposals.append(proposal)
        turn.timeline.append(
            TimelineItem(TimelineKind.PROPOSAL, proposal_id)
        )
        self._last_timeline_kind = TimelineKind.PROPOSAL

    def _apply_proposal_lifecycle(
        self,
        turn: TurnView,
        event: AgentEvent,
    ) -> None:
        proposal_id = event.payload["proposal_id"]
        proposal = next(
            (
                item
                for item in turn.proposals
                if item.proposal_id == proposal_id
            ),
            None,
        )
        if proposal is None:
            raise AgentEventError("proposal event references an unknown proposal_id")
        if proposal.proposal_hash != event.payload["proposal_hash"]:
            raise AgentEventError("proposal_hash does not match the request event")

        event_type = event.event_type
        current = proposal.status
        allowed: dict[EventType, frozenset[ProposalViewStatus]] = {
            EventType.PROPOSAL_ACCEPTED: frozenset(
                {ProposalViewStatus.PENDING_CONFIRMATION}
            ),
            EventType.PROPOSAL_REJECTED: frozenset(
                {ProposalViewStatus.PENDING_CONFIRMATION}
            ),
            EventType.PROPOSAL_STALE: frozenset(
                {
                    ProposalViewStatus.PENDING_CONFIRMATION,
                    ProposalViewStatus.ACCEPTED,
                    ProposalViewStatus.RUNNING,
                }
            ),
            EventType.PROPOSAL_STARTED: frozenset(
                {ProposalViewStatus.ACCEPTED}
            ),
            EventType.PROPOSAL_PROGRESS: frozenset(
                {ProposalViewStatus.RUNNING}
            ),
            EventType.PROPOSAL_SUCCEEDED: frozenset(
                {
                    ProposalViewStatus.ACCEPTED,
                    ProposalViewStatus.RUNNING,
                }
            ),
            EventType.PROPOSAL_FAILED: frozenset(
                {
                    ProposalViewStatus.PENDING_CONFIRMATION,
                    ProposalViewStatus.ACCEPTED,
                    ProposalViewStatus.RUNNING,
                }
            ),
            EventType.PROPOSAL_CANCELLED: frozenset(
                {
                    ProposalViewStatus.PENDING_CONFIRMATION,
                    ProposalViewStatus.ACCEPTED,
                    ProposalViewStatus.RUNNING,
                }
            ),
        }
        if current not in allowed[event_type]:
            raise AgentEventError(
                f"proposal cannot transition from {current.value} to {event_type.value}"
            )

        status_by_event = {
            EventType.PROPOSAL_ACCEPTED: ProposalViewStatus.ACCEPTED,
            EventType.PROPOSAL_REJECTED: ProposalViewStatus.REJECTED,
            EventType.PROPOSAL_STALE: ProposalViewStatus.STALE,
            EventType.PROPOSAL_STARTED: ProposalViewStatus.RUNNING,
            EventType.PROPOSAL_PROGRESS: ProposalViewStatus.RUNNING,
            EventType.PROPOSAL_SUCCEEDED: ProposalViewStatus.SUCCEEDED,
            EventType.PROPOSAL_FAILED: ProposalViewStatus.FAILED,
            EventType.PROPOSAL_CANCELLED: ProposalViewStatus.CANCELLED,
        }
        proposal.status = status_by_event[event_type]
        if event_type is EventType.PROPOSAL_ACCEPTED:
            proposal.authorized = True
            proposal.status_message = "Authorized by user"
        elif event_type is EventType.PROPOSAL_STARTED:
            proposal.progress = 0.0
            proposal.status_message = "Background task started"
        elif event_type is EventType.PROPOSAL_PROGRESS:
            proposal.progress = float(event.payload["progress"])
            proposal.status_message = safe_tool_summary(
                event.payload["message"],
                max_characters=300,
            )
        elif event_type is EventType.PROPOSAL_SUCCEEDED:
            proposal.progress = 1.0
            proposal.status_message = safe_tool_summary(
                event.payload["summary"],
                max_characters=500,
            )
        else:
            proposal.status_message = safe_tool_summary(
                event.payload["reason"],
                max_characters=500,
            )

    def _turn_complete(self, turn: TurnView) -> None:
        if any(
            message.status is MessageStatus.STREAMING
            for message in turn.messages
        ):
            raise AgentEventError("Cannot complete turn with unfinished messages")
        if any(
            call.status in {ToolStatus.REQUESTED, ToolStatus.RUNNING}
            for group in turn.tool_groups
            for call in group.calls
        ):
            raise AgentEventError("Cannot complete turn with unfinished tool calls")
        turn.status = TurnStatus.COMPLETED
        self._finish_turn()

    def _turn_cancelled(self, turn: TurnView, event: AgentEvent) -> None:
        reason = safe_tool_summary(
            event.payload["reason"],
            max_characters=500,
        )
        for message in turn.messages:
            if message.status is MessageStatus.STREAMING:
                message.status = MessageStatus.CANCELLED
        for group in turn.tool_groups:
            for call in group.calls:
                if call.status in {ToolStatus.REQUESTED, ToolStatus.RUNNING}:
                    call.status = ToolStatus.CANCELLED
                    call.diagnostics.append(reason)
        turn.status = TurnStatus.CANCELLED
        turn.failure_reason = reason
        self._finish_turn()

    def _turn_failed(self, turn: TurnView, event: AgentEvent) -> None:
        reason = safe_tool_summary(
            event.payload["reason"],
            max_characters=500,
        )
        for message in turn.messages:
            if message.status is MessageStatus.STREAMING:
                message.status = MessageStatus.INTERRUPTED
        for group in turn.tool_groups:
            for call in group.calls:
                if call.status in {ToolStatus.REQUESTED, ToolStatus.RUNNING}:
                    call.status = ToolStatus.FAILED
                    call.result_summary = reason
        turn.status = TurnStatus.FAILED
        turn.failure_reason = reason
        self._finish_turn()

    def _finish_turn(self) -> None:
        self._active_turn_id = None
        self._last_timeline_kind = None


class FakeAgentEventStream:
    """Generate deterministic in-memory events for phase 3 review without executing Agent capabilities."""

    def __init__(
        self,
        *,
        session_id: str = "agent-preview",
        start_sequence: int = 1,
        event_prefix: str = "phase3-event",
    ) -> None:
        _require_identifier(session_id, "session_id")
        _require_identifier(event_prefix, "event_prefix")
        if not _is_plain_int(start_sequence) or start_sequence < 1:
            raise AgentEventError("start_sequence must be a positive integer")
        self.session_id = session_id
        self._sequence = start_sequence
        self._event_prefix = event_prefix

    def _event(
        self,
        turn_id: str,
        event_type: EventType,
        payload: Mapping[str, Any],
    ) -> AgentEvent:
        sequence = self._sequence
        self._sequence += 1
        return AgentEvent.create(
            event_id=f"{self._event_prefix}-{sequence:04d}",
            session_id=self.session_id,
            turn_id=turn_id,
            sequence=sequence,
            event_type=event_type,
            payload=payload,
            timestamp=f"2026-07-29T08:00:{sequence % 60:02d}Z",
        )

    def review_preview(self) -> tuple[AgentEvent, ...]:
        turn_id = "phase3-turn-1"
        revision_hash = (
            "3b2c1af878b43b517931e25b3894d601"
            "55d2a2e1e4862d9135d84f471586a6cf"
        )
        specifications = (
            (
                "read_model_summary",
                "Read model summary",
                {"file": "frame.inp", "scope": "metadata"},
                {"nodes": 128, "elements": 96},
                200,
            ),
            (
                "check_material_sections",
                "Check materials and sections",
                {"model": "frame.inp"},
                {"materials": 1, "sections": 1},
                400,
            ),
            (
                "validate_boundary_conditions",
                "Validate boundary conditions",
                {"model": "frame.inp"},
                {"constraints": 2, "loads": 1},
                600,
            ),
        )
        events: list[AgentEvent] = [
            self._event(
                turn_id,
                EventType.TURN_STARTED,
                {
                    "user_message": (
                        "Use @design-notes.md to check the materials, constraints, and "
                        "loads in @frame.inp."
                    )
                },
            )
        ]
        for index, (
            tool_name,
            display_name,
            request,
            result,
            duration_ms,
        ) in enumerate(specifications, start=1):
            call_id = f"preview-call-{index}"
            events.extend(
                (
                    self._event(
                        turn_id,
                        EventType.TOOL_REQUESTED,
                        {
                            "call_id": call_id,
                            "tool_name": tool_name,
                            "display_name": display_name,
                            "request": request,
                        },
                    ),
                    self._event(
                        turn_id,
                        EventType.TOOL_STARTED,
                        {"call_id": call_id},
                    ),
                    self._event(
                        turn_id,
                        EventType.TOOL_RESULT,
                        {
                            "call_id": call_id,
                            "result": result,
                            "duration_ms": duration_ms,
                        },
                    ),
                )
            )
        events.extend(
            (
                self._event(
                    turn_id,
                    EventType.MESSAGE_START,
                    {
                        "message_id": "preview-message-1",
                        "role": "assistant",
                        "format": "restricted_markdown",
                    },
                ),
                self._event(
                    turn_id,
                    EventType.MESSAGE_DELTA,
                    {
                        "message_id": "preview-message-1",
                        "delta": (
                            "**Model precheck complete.** Material and section definitions are complete, "
                        ),
                    },
                ),
                self._event(
                    turn_id,
                    EventType.MESSAGE_DELTA,
                    {
                        "message_id": "preview-message-1",
                        "delta": (
                            "and boundary conditions suppress rigid body motion. Before analysis, reconfirm "
                            "the load units."
                        ),
                    },
                ),
                self._event(
                    turn_id,
                    EventType.MESSAGE_COMPLETE,
                    {"message_id": "preview-message-1"},
                ),
                self._event(
                    turn_id,
                    EventType.DIAGNOSTIC,
                    {
                        "diagnostic_id": "preview-analysis-summary",
                        "title": "Analysis summary",
                        "message": (
                            "The input uses N–mm units; confirm the concentrated "
                            "load units before solving."
                        ),
                        "severity": "warning",
                        "code": "UNIT-CHECK",
                    },
                ),
                self._event(
                    turn_id,
                    EventType.CONFIRMATION_REQUESTED,
                    {
                        "confirmation_id": "preview-confirmation-1",
                        "title": "Confirmation required",
                        "summary": (
                            "Confirm the load units for revision 12 before later stages can request "
                            "a solve."
                        ),
                        "revision": 12,
                        "revision_hash": revision_hash,
                    },
                ),
                self._event(
                    turn_id,
                    EventType.TURN_COMPLETE,
                    {},
                ),
            )
        )
        return tuple(events)
