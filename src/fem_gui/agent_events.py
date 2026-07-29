"""GUI 使用的 FEM Agent 结构化事件契约与纯内存投影器。

本模块没有 Qt、Provider 或 ``fem_agent`` 依赖。它只负责验证可序列化事件，
将事件归并为安全的展示状态，以及从完整事件日志重放同一状态。
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
    """事件不符合契约或不能应用到当前会话状态。"""


class EventType(str, Enum):
    TURN_STARTED = "turn_started"
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


_REQUIRED_PAYLOAD_FIELDS: dict[EventType, frozenset[str]] = {
    EventType.TURN_STARTED: frozenset({"user_message"}),
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
    EventType.TURN_CANCELLED: frozenset({"reason"}),
    EventType.TURN_COMPLETE: frozenset(),
    EventType.TURN_FAILED: frozenset({"reason"}),
}

_OPTIONAL_PAYLOAD_FIELDS: dict[EventType, frozenset[str]] = {
    EventType.TURN_STARTED: frozenset(),
    EventType.MESSAGE_START: frozenset(),
    EventType.MESSAGE_DELTA: frozenset(),
    EventType.MESSAGE_COMPLETE: frozenset(),
    EventType.TOOL_REQUESTED: frozenset(),
    EventType.TOOL_STARTED: frozenset(),
    EventType.TOOL_RESULT: frozenset(),
    EventType.TOOL_WARNING: frozenset({"result"}),
    EventType.TOOL_FAILED: frozenset({"diagnostic"}),
    EventType.DIAGNOSTIC: frozenset({"code"}),
    EventType.CONFIRMATION_REQUESTED: frozenset(),
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
        raise AgentEventError(f"{field_name} 不是有效标识符")
    return value


def _require_string(
    value: object,
    field_name: str,
    *,
    allow_empty: bool = False,
) -> str:
    if not isinstance(value, str):
        raise AgentEventError(f"{field_name} 必须是字符串")
    if not allow_empty and not value.strip():
        raise AgentEventError(f"{field_name} 不能为空")
    return value


def _require_duration(value: object) -> float:
    if not _is_number(value) or float(value) < 0:
        raise AgentEventError("duration_ms 必须是非负有限数值")
    return float(value)


def _validate_timestamp(value: object) -> str:
    timestamp = _require_string(value, "timestamp")
    normalized = timestamp[:-1] + "+00:00" if timestamp.endswith("Z") else timestamp
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise AgentEventError("timestamp 必须是 ISO-8601 时间") from exc
    if parsed.tzinfo is None:
        raise AgentEventError("timestamp 必须包含时区")
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
        raise AgentEventError("payload 必须只包含 JSON 可序列化值") from exc
    if len(encoded.encode("utf-8")) > MAX_EVENT_PAYLOAD_BYTES:
        raise AgentEventError("payload 超出事件大小上限")
    decoded = json.loads(encoded)
    if not isinstance(decoded, dict):
        raise AgentEventError("payload 必须是对象")
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
            f"{event_type.value} 缺少 payload 字段：{', '.join(sorted(missing))}"
        )
    if unknown:
        raise AgentEventError(
            f"{event_type.value} 包含未知 payload 字段："
            f"{', '.join(sorted(unknown))}"
        )

    if event_type is EventType.TURN_STARTED:
        _require_string(payload["user_message"], "user_message")
        return
    if event_type is EventType.MESSAGE_START:
        _require_identifier(payload["message_id"], "message_id")
        if payload["role"] != "assistant":
            raise AgentEventError("阶段 3 消息 role 只允许 assistant")
        if payload["format"] != "restricted_markdown":
            raise AgentEventError("消息 format 必须是 restricted_markdown")
        return
    if event_type is EventType.MESSAGE_DELTA:
        _require_identifier(payload["message_id"], "message_id")
        _require_string(payload["delta"], "delta", allow_empty=False)
        return
    if event_type is EventType.MESSAGE_COMPLETE:
        _require_identifier(payload["message_id"], "message_id")
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
            raise AgentEventError("severity 不是已知诊断级别") from exc
        if "code" in payload:
            _require_identifier(payload["code"], "code")
        return
    if event_type is EventType.CONFIRMATION_REQUESTED:
        _require_identifier(payload["confirmation_id"], "confirmation_id")
        _require_string(payload["title"], "title")
        _require_string(payload["summary"], "summary")
        if not _is_plain_int(payload["revision"]) or payload["revision"] < 0:
            raise AgentEventError("revision 必须是非负整数")
        revision_hash = payload["revision_hash"]
        if (
            not isinstance(revision_hash, str)
            or not _REVISION_HASH_PATTERN.fullmatch(revision_hash)
        ):
            raise AgentEventError("revision_hash 必须是完整 SHA-256 十六进制值")
        return
    if event_type in {
        EventType.TURN_CANCELLED,
        EventType.TURN_FAILED,
    }:
        _require_string(payload["reason"], "reason")


@dataclass(frozen=True)
class AgentEvent:
    """单个、可序列化且自校验的 GUI Agent 事件。"""

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
                f"不支持的 schema_version：{self.schema_version}"
            )
        _require_identifier(self.event_id, "event_id")
        _require_identifier(self.session_id, "session_id")
        _require_identifier(self.turn_id, "turn_id")
        if not _is_plain_int(self.sequence) or self.sequence < 1:
            raise AgentEventError("sequence 必须是从 1 开始的整数")
        try:
            event_type = EventType(self.event_type)
        except (TypeError, ValueError) as exc:
            raise AgentEventError("event_type 未知") from exc
        object.__setattr__(self, "event_type", event_type)
        object.__setattr__(self, "timestamp", _validate_timestamp(self.timestamp))
        if not isinstance(self.payload, Mapping):
            raise AgentEventError("payload 必须是对象")
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
            raise AgentEventError("事件记录必须是对象")
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
                f"事件缺少字段：{', '.join(sorted(missing))}"
            )
        if unknown:
            raise AgentEventError(
                f"事件包含未知字段：{', '.join(sorted(unknown))}"
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
        redacted = pattern.sub("<敏感信息已隐藏>", redacted)
    redacted = _FILE_URI_PATTERN.sub(
        "<绝对路径已隐藏>",
        redacted,
    )
    redacted = _WINDOWS_ABSOLUTE_PATH_PATTERN.sub(
        "<绝对路径已隐藏>",
        redacted,
    )
    redacted = _POSIX_ABSOLUTE_PATH_PATTERN.sub(
        "<绝对路径已隐藏>",
        redacted,
    )
    return redacted


def safe_tool_summary(
    value: object,
    *,
    max_characters: int = MAX_SUMMARY_CHARACTERS,
) -> str:
    """把任意工具值压缩为有界、脱敏且不调用对象 ``repr`` 的文字。"""

    def summarize(item: object, depth: int) -> str:
        if item is None:
            return "null"
        if isinstance(item, bool):
            return "true" if item else "false"
        if isinstance(item, (int, float)):
            if isinstance(item, float) and not math.isfinite(item):
                return "<非有限数值>"
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
                    else "<非字符串键>"
                )
                if isinstance(key, str) and _SENSITIVE_KEY_PATTERN.search(key):
                    child_text = "<敏感信息已隐藏>"
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
        return "<不支持的值>"

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
    text: str = ""
    status: MessageStatus = MessageStatus.STREAMING


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
    timeline: list[TimelineItem] = field(default_factory=list)
    failure_reason: str = ""


@dataclass
class SessionPresentation:
    schema_version: str = AGENT_EVENT_SCHEMA_VERSION
    session_id: str = ""
    last_sequence: int = 0
    turns: list[TurnView] = field(default_factory=list)

    def to_snapshot(self) -> dict[str, Any]:
        """返回只含安全展示字段的可序列化会话快照。"""

        def convert(value: object) -> object:
            if isinstance(value, Enum):
                return value.value
            if isinstance(value, dict):
                return {str(key): convert(item) for key, item in value.items()}
            if isinstance(value, list):
                return [convert(item) for item in value]
            return value

        return convert(asdict(self))


class AgentEventProjector:
    """严格按 session 全局 sequence 将事件投影为聊天展示状态。"""

    def __init__(self) -> None:
        self._presentation = SessionPresentation()
        self._events: list[AgentEvent] = []
        self._event_ids: set[str] = set()
        self._active_turn_id: str | None = None
        self._last_timeline_kind: TimelineKind | None = None

    @property
    def presentation(self) -> SessionPresentation:
        return deepcopy(self._presentation)

    @property
    def last_sequence(self) -> int:
        return self._presentation.last_sequence

    def export_event_log(self) -> tuple[dict[str, Any], ...]:
        return tuple(event.to_dict() for event in self._events)

    @classmethod
    def replay(cls, events: Iterable[AgentEvent]) -> AgentEventProjector:
        projector = cls()
        for event in events:
            projector.apply(event)
        return projector

    @classmethod
    def restore_event_log(
        cls,
        records: Iterable[Mapping[str, Any]],
    ) -> AgentEventProjector:
        return cls.replay(AgentEvent.from_dict(record) for record in records)

    def apply(self, event: AgentEvent) -> SessionPresentation:
        if not isinstance(event, AgentEvent):
            raise AgentEventError("projector 只接受 AgentEvent")
        if event.event_id in self._event_ids:
            raise AgentEventError(f"重复 event_id：{event.event_id}")
        expected_sequence = self._presentation.last_sequence + 1
        if event.sequence != expected_sequence:
            raise AgentEventError(
                f"sequence 应为 {expected_sequence}，收到 {event.sequence}"
            )
        if (
            self._presentation.session_id
            and event.session_id != self._presentation.session_id
        ):
            raise AgentEventError("事件跨越了当前 session")
        if event.event_type is EventType.TURN_STARTED:
            self._apply_turn_started(event)
        else:
            turn = self._require_active_turn(event)
            self._apply_turn_event(turn, event)

        if not self._presentation.session_id:
            self._presentation.session_id = event.session_id
        self._presentation.last_sequence = event.sequence
        self._event_ids.add(event.event_id)
        self._events.append(event)
        return self.presentation

    def _apply_turn_started(self, event: AgentEvent) -> None:
        if self._active_turn_id is not None:
            raise AgentEventError("上一 turn 尚未结束")
        if any(turn.turn_id == event.turn_id for turn in self._presentation.turns):
            raise AgentEventError("turn_id 已存在，不能重新开始")
        turn = TurnView(
            turn_id=event.turn_id,
            user_message=safe_tool_summary(
                event.payload["user_message"],
                max_characters=4_000,
            ),
        )
        self._presentation.turns.append(turn)
        self._active_turn_id = event.turn_id
        self._last_timeline_kind = None

    def _require_active_turn(self, event: AgentEvent) -> TurnView:
        if self._active_turn_id is None:
            raise AgentEventError("当前没有运行中的 turn")
        if event.turn_id != self._active_turn_id:
            raise AgentEventError("事件跨越了当前 turn")
        return self._presentation.turns[-1]

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
        raise AgentEventError(f"未知 message_id：{message_id}")

    @staticmethod
    def _find_tool(turn: TurnView, call_id: str) -> ToolActivityView:
        for group in turn.tool_groups:
            for call in group.calls:
                if call.call_id == call_id:
                    return call
        raise AgentEventError(f"未知 call_id：{call_id}")

    def _message_start(self, turn: TurnView, event: AgentEvent) -> None:
        message_id = event.payload["message_id"]
        if any(message.message_id == message_id for message in turn.messages):
            raise AgentEventError("message_id 已存在")
        message = MessageView(
            message_id=message_id,
            role=event.payload["role"],
            format=event.payload["format"],
        )
        turn.messages.append(message)
        turn.timeline.append(
            TimelineItem(TimelineKind.MESSAGE, message.message_id)
        )
        self._last_timeline_kind = TimelineKind.MESSAGE

    def _message_delta(self, turn: TurnView, event: AgentEvent) -> None:
        message = self._find_message(turn, event.payload["message_id"])
        if message.status is not MessageStatus.STREAMING:
            raise AgentEventError("已结束的消息不能继续追加 delta")
        message.text += event.payload["delta"]
        self._last_timeline_kind = TimelineKind.MESSAGE

    def _message_complete(self, turn: TurnView, event: AgentEvent) -> None:
        message = self._find_message(turn, event.payload["message_id"])
        if message.status is not MessageStatus.STREAMING:
            raise AgentEventError("消息已经结束")
        message.status = MessageStatus.COMPLETED
        self._last_timeline_kind = TimelineKind.MESSAGE

    def _tool_requested(self, turn: TurnView, event: AgentEvent) -> None:
        call_id = event.payload["call_id"]
        try:
            self._find_tool(turn, call_id)
        except AgentEventError:
            pass
        else:
            raise AgentEventError("call_id 已存在")

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
            raise AgentEventError("工具只能从 requested 进入 running")
        call.status = ToolStatus.RUNNING

    def _tool_terminal(self, turn: TurnView, event: AgentEvent) -> None:
        call = self._find_tool(turn, event.payload["call_id"])
        if call.status is not ToolStatus.RUNNING:
            raise AgentEventError("工具终态事件要求工具处于 running")
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
            raise AgentEventError("diagnostic_id 已存在")
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
            raise AgentEventError("confirmation_id 已存在")
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

    def _turn_complete(self, turn: TurnView) -> None:
        if any(
            message.status is MessageStatus.STREAMING
            for message in turn.messages
        ):
            raise AgentEventError("存在未完成消息，不能完成 turn")
        if any(
            call.status in {ToolStatus.REQUESTED, ToolStatus.RUNNING}
            for group in turn.tool_groups
            for call in group.calls
        ):
            raise AgentEventError("存在未完成工具调用，不能完成 turn")
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
    """为阶段 3 审核生成确定性的纯内存事件，不执行任何 Agent 能力。"""

    def __init__(
        self,
        *,
        session_id: str = "phase3-preview",
        start_sequence: int = 1,
        event_prefix: str = "phase3-event",
    ) -> None:
        _require_identifier(session_id, "session_id")
        _require_identifier(event_prefix, "event_prefix")
        if not _is_plain_int(start_sequence) or start_sequence < 1:
            raise AgentEventError("start_sequence 必须是正整数")
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
                "读取模型摘要",
                {"file": "frame.inp", "scope": "metadata"},
                {"nodes": 128, "elements": 96},
                200,
            ),
            (
                "check_material_sections",
                "检查材料与截面",
                {"model": "frame.inp"},
                {"materials": 1, "sections": 1},
                400,
            ),
            (
                "validate_boundary_conditions",
                "验证边界条件",
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
                        "请结合 @设计说明.md，检查 @frame.inp 的材料、约束和"
                        "载荷设置。"
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
                            "**模型预检查已完成。** 材料与截面定义完整，"
                        ),
                    },
                ),
                self._event(
                    turn_id,
                    EventType.MESSAGE_DELTA,
                    {
                        "message_id": "preview-message-1",
                        "delta": (
                            "边界条件能够抑制刚体位移。建议在正式分析前再次"
                            "确认载荷单位。"
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
                        "title": "分析摘要",
                        "message": (
                            "当前输入使用 N–mm 单位制；正式求解前应确认集中"
                            "载荷的单位。"
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
                        "title": "需要确认",
                        "summary": (
                            "确认 revision 12 的载荷单位后，后续阶段才可请求"
                            "执行求解。"
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
