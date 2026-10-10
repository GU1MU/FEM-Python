"""UI-neutral conversational state machine for FEM Agent."""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .artifacts import (
    ArtifactIntegrityError,
    ArtifactRecord,
    ArtifactStore,
    safe_child,
)
from .confirmation import ConfirmationStore
from .conversation_presentation import (
    AUTHORING_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    build_provider_messages,
    provider_snapshot_dict,
)
from .conversation_store import ConversationStorageLimit, ConversationStore
from .proposal_presentation import (
    geometry_proposal_preview_message,
    planar_retry_limit_message,
)
from .tool_audit import ToolAuditStore, build_round_audit_entry
from .diagnostics import DiagnosticCode, make_diagnostic
from .providers.base import (
    AssistantMessage,
    CloudModelProvider,
    ProviderAuthenticationError,
    ProviderCredentialMissingError,
    ProviderMalformedResponseError,
    ProviderPaymentRequiredError,
    ProviderRateLimitError,
    ProviderResponse,
    ProviderTimeoutError,
    ProviderUnavailableError,
    ToolCall,
    ToolDefinition,
)
from .schemas import (
    AnalysisSummary,
    Diagnostic,
    DiagnosticSeverity,
    ImportAnalysisSpec,
    ResourceLimits,
    RunStatus,
    SessionPhase,
    ToolResult,
)
from .state import RevisionRecord, RevisionStore, hash_revision_spec
from .tools.registry import (
    AgentToolRegistry,
    DynamicToolRegistry,
    ToolExecutionContext,
)
from .worker import (
    IsolatedFEMWorker,
    WorkerRunInProgressError,
)
from .worker_integrity import (
    WorkerResponseIntegrityError,
    load_verified_worker_response,
)
from .worker_protocol import WorkerResponse


_CREDENTIAL_ASSIGNMENT_PATTERN = re.compile(
    r"""(?ix)
    (?:^|[\s"'`])
    (?:
        (?:[a-z0-9]+[_-])*api[\s_-]*key
        |authorization
        |bearer
        |access[\s_-]*token
        |auth[\s_-]*token
        |password
        |secret
        |credential
    )
    \s*[:=]\s*
    [^\s,;]{8,}
    """
)
_BEARER_PATTERN = re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{8,}")
_COMMON_API_KEY_PATTERN = re.compile(r"(?i)\bsk-[A-Za-z0-9_-]{12,}\b")
_TRANSIENT_RUN_DIAGNOSTICS = frozenset(
    {
        DiagnosticCode.OPERATION_CANCELLED.value,
        DiagnosticCode.WORKER_CRASH.value,
        DiagnosticCode.WORKER_TIMEOUT.value,
    }
)


class EngineEventType(str, Enum):
    MESSAGE_STARTED = "message_started"
    MESSAGE_DELTA = "message_delta"
    TOOL_STARTED = "tool_started"
    TOOL_COMPLETED = "tool_completed"
    ANALYSIS_SUMMARY = "analysis_summary"
    STATE_CHANGED = "state_changed"
    CONFIRMATION_REQUIRED = "confirmation_required"
    RUN_PROGRESS = "run_progress"
    RUN_COMPLETED = "run_completed"
    DIAGNOSTIC = "diagnostic"
    OPERATION_CANCELLED = "operation_cancelled"
    ERROR = "error"


@dataclass(frozen=True)
class EngineEvent:
    event: EngineEventType
    session_id: str
    data: Mapping[str, Any]
    timestamp: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "event", EngineEventType(self.event))
        object.__setattr__(self, "data", dict(self.data))

    def to_dict(self) -> dict[str, Any]:
        return {
            "event": self.event.value,
            "session_id": self.session_id,
            "data": dict(self.data),
            "timestamp": self.timestamp,
        }


@dataclass(frozen=True)
class EngineSnapshot:
    session_id: str
    phase: SessionPhase
    revision: int
    revision_hash: str | None
    confirmed: bool
    active_run_id: str | None
    provider: str
    model: str
    cloud_enabled: bool
    workspace: str
    artifacts: tuple[ArtifactRecord, ...]


@dataclass(frozen=True)
class ContinuationCheckpoint:
    """One process-local, single-use proposal continuation identity."""

    session_id: str
    source_turn_id: str
    proposal_id: str
    proposal_hash: str
    model_revision: int
    proposal_kind: str = ""


@dataclass(frozen=True)
class _ProviderRequestResult:
    response: ProviderResponse
    streamed_text_parts: tuple[str, ...]
    streamed_reasoning_parts: tuple[str, ...]
    text_presented: bool
    reasoning_presented: bool


@dataclass(frozen=True)
class EngineConfig:
    max_cloud_turns: int = 12
    max_tool_calls: int = 48
    max_tool_payload_bytes: int = 64 * 1024
    max_conversation_messages: int = 64
    max_user_message_chars: int = 16_000
    # Must stay above the worst-case character count of the configured
    # provider output budget (65536 tokens * ~4 chars/token for English
    # reasoning); a smaller guard would re-truncate long thinking turns
    # with a misleading malformed-response diagnostic.
    max_provider_message_chars: int = 320_000
    max_request_context_bytes: int = 4 * 1024 * 1024
    max_tool_arguments_bytes: int = 64 * 1024
    max_conversation_storage_bytes: int = 4 * 1024 * 1024
    max_tool_audit_storage_bytes: int = 1024 * 1024

    def __post_init__(self) -> None:
        for name in (
            "max_cloud_turns",
            "max_tool_calls",
            "max_tool_payload_bytes",
            "max_conversation_messages",
            "max_user_message_chars",
            "max_provider_message_chars",
            "max_request_context_bytes",
            "max_tool_arguments_bytes",
            "max_conversation_storage_bytes",
            "max_tool_audit_storage_bytes",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in (
            "max_conversation_storage_bytes",
            "max_tool_audit_storage_bytes",
        ):
            if getattr(self, name) < 1024:
                raise ValueError(f"{name} must be at least 1024 bytes")
        minimum_tool_turn_messages = self.max_cloud_turns + self.max_tool_calls
        if self.max_conversation_messages < minimum_tool_turn_messages:
            raise ValueError(
                "max_conversation_messages must be at least "
                "max_cloud_turns + max_tool_calls so a complete tool turn "
                "can be sent back to the provider"
            )


class AgentSessionEngine:
    """Conversation, revision, confirmation, and worker orchestration."""

    def __init__(
        self,
        workspace: str | os.PathLike[str],
        provider: CloudModelProvider,
        *,
        session_id: str | None = None,
        config: EngineConfig | None = None,
        event_sink: Callable[[EngineEvent], None] | None = None,
        dynamic_tools: DynamicToolRegistry | None = None,
        defer_audit_persistence: bool = False,
    ):
        self.config = config or EngineConfig()
        self.provider = provider
        self.artifacts = ArtifactStore(workspace)
        self.revisions = RevisionStore(self.artifacts.root)
        self.confirmations = ConfirmationStore(
            self.artifacts.root,
            self.revisions,
        )
        self._cancel_event = threading.Event()
        self._operation_started_event = threading.Event()
        self.registry = AgentToolRegistry(
            self.artifacts.root,
            cancel_event=self._cancel_event,
            dynamic_tools=dynamic_tools,
        )
        self._system_prompt = (
            SYSTEM_PROMPT + AUTHORING_SYSTEM_PROMPT
            if dynamic_tools is not None
            else SYSTEM_PROMPT
        )
        self.worker = IsolatedFEMWorker(self.artifacts.root)
        self._state_lock = threading.RLock()
        self._operation_lock = threading.Lock()
        self._event_sinks: list[Callable[[EngineEvent], None]] = []
        if event_sink is not None:
            self._event_sinks.append(event_sink)
        self._running = False
        self._provider_active = False
        self._closed = False
        self._active_run: WorkerResponse | None = None
        self._summary_shown_revision: tuple[int, str] | None = None
        self._tool_result_cache: dict[str, ToolResult] = {}
        self._continuations: dict[str, ContinuationCheckpoint] = {}
        self._pending_round_audit: list[dict[str, Any]] | None = None
        self._defer_audit_persistence = bool(defer_audit_persistence)
        self._deferred_round_audit_batches: list[tuple[dict[str, Any], ...]] = []
        self._audit_write_lock = threading.Lock()
        self._conversation_store = ConversationStore(
            self.artifacts,
            max_messages=self.config.max_conversation_messages,
            max_storage_bytes=self.config.max_conversation_storage_bytes,
        )
        self._audit_store = ToolAuditStore(
            self.artifacts,
            max_storage_bytes=self.config.max_tool_audit_storage_bytes,
        )
        self.session_id = self.artifacts.create_session(session_id)
        self.revisions.create_session(self.session_id)
        self._history = self._load_conversation()
        self._active_run = self._load_latest_run()

    @property
    def workspace(self) -> Path:
        return self.artifacts.root

    def subscribe(
        self,
        sink: Callable[[EngineEvent], None],
    ) -> Callable[[], None]:
        """Subscribe to ordered events as operations produce them."""

        if not callable(sink):
            raise TypeError("event sink must be callable")
        with self._state_lock:
            self._event_sinks.append(sink)

        def unsubscribe() -> None:
            with self._state_lock:
                try:
                    self._event_sinks.remove(sink)
                except ValueError:
                    pass

        return unsubscribe

    def create_session(self) -> tuple[EngineEvent, ...]:
        """Switch this engine instance to a new empty local session."""

        if not self._operation_lock.acquire(blocking=False):
            self._operation_started_event.set()
            return self._operation_in_progress_events()
        self._begin_operation()
        try:
            return self._create_session()
        finally:
            self._operation_lock.release()

    def _create_session(self) -> tuple[EngineEvent, ...]:
        self._ensure_open()
        with self._state_lock:
            self.session_id = self.artifacts.create_session()
            self.revisions.create_session(self.session_id)
            self._history = []
            self._active_run = None
            self._summary_shown_revision = None
            self._tool_result_cache = {}
            self._continuations = {}
        return (self._state_event(),)

    new_session = create_session

    def attach_artifact(
        self,
        artifact_id: str,
        *,
        replace_existing: bool = False,
    ) -> tuple[EngineEvent, ...]:
        """Attach an already copied artifact without exposing its source path."""

        if not self._operation_lock.acquire(blocking=False):
            self._operation_started_event.set()
            return self._operation_in_progress_events()
        self._begin_operation()
        try:
            return self._attach_artifact(
                artifact_id,
                replace_existing=replace_existing,
            )
        finally:
            self._operation_lock.release()

    def _attach_artifact(
        self,
        artifact_id: str,
        *,
        replace_existing: bool = False,
    ) -> tuple[EngineEvent, ...]:
        self._ensure_open()
        artifact = self.artifacts.get_artifact(self.session_id, artifact_id)
        if artifact.kind != "input":
            raise ValueError("only an input artifact can be attached for analysis")
        current = self.revisions.latest(self.session_id)
        if (
            current is not None
            and current.spec.source_artifact_id != artifact.artifact_id
            and not replace_existing
        ):
            raise ValueError(
                "the session already has an input; explicit replacement is required"
            )
        if (
            current is not None
            and current.spec.source_artifact_id == artifact.artifact_id
        ):
            return (self._state_event(),)

        next_revision = 1 if current is None else current.revision + 1
        provisional = ImportAnalysisSpec(
            session_id=self.session_id,
            revision=next_revision,
            source_artifact_id=artifact.artifact_id,
            source_sha256=artifact.sha256,
            unit_context=None,
            analysis_step=None,
            requested_queries=(),
            export_formats=(),
            resource_limits=(
                ResourceLimits() if current is None else current.spec.resource_limits
            ),
        )
        try:
            inspection = self.registry.inspector.inspect(
                provisional,
                hash_revision_spec(provisional),
                cancel_event=self._cancel_event,
            )
            if self._cancel_event.is_set():
                return self._cancelled_operation_events("inspection")
            analysis_step = (
                None
                if inspection.summary.analysis_step is None
                else str(inspection.summary.analysis_step.get("name") or "") or None
            )
            import_diagnostics = tuple(
                item
                for item in inspection.summary.diagnostics
                if item.code
                not in {
                    DiagnosticCode.UNIT_CONTEXT_REQUIRED.value,
                    DiagnosticCode.RESULT_REQUEST_REQUIRED.value,
                }
                and not (
                    item.code == DiagnosticCode.INVALID_INPUT.value
                    and item.entity == "analysis_step"
                )
            )
        except Exception:
            if self._cancel_event.is_set():
                return self._cancelled_operation_events("inspection")
            analysis_step = None
            inspection_error = make_diagnostic(
                DiagnosticCode.WORKER_CRASH,
                "The isolated Abaqus inspection process failed.",
                source="agent.inspection_worker",
            )
            import_diagnostics = (inspection_error,)
        with self._state_lock:
            if self._cancel_event.is_set():
                return self._cancelled_operation_events("inspection")
            if current is None:
                record = self.revisions.initialize(
                    ImportAnalysisSpec(
                        session_id=self.session_id,
                        revision=1,
                        source_artifact_id=artifact.artifact_id,
                        source_sha256=artifact.sha256,
                        unit_context=None,
                        analysis_step=analysis_step,
                        requested_queries=(),
                        export_formats=(),
                        resource_limits=ResourceLimits(),
                    ),
                    idempotency_key=f"attach_{artifact.artifact_id}",
                    operation="attach",
                )
            else:
                record = self.revisions.mutate(
                    self.session_id,
                    expected_revision=current.revision,
                    idempotency_key=f"attach_{artifact.artifact_id}",
                    changes={
                        "source_artifact_id": artifact.artifact_id,
                        "source_sha256": artifact.sha256,
                        "unit_context": None,
                        "analysis_step": analysis_step,
                        "requested_queries": (),
                        "export_formats": (),
                        "assumptions": (),
                    },
                    operation="replace_input",
                )
            self._active_run = None
            self._summary_shown_revision = None
        events = [self._state_event(record)]
        events.extend(self._diagnostic_event(item) for item in import_diagnostics)
        return tuple(events)

    def send_message(
        self,
        text: str,
        *,
        request_context: str | None = None,
    ) -> tuple[EngineEvent, ...]:
        """Run a bounded cloud/tool loop and return structured UI events."""

        if not self._operation_lock.acquire(blocking=False):
            self._operation_started_event.set()
            return self._operation_in_progress_events()
        self._begin_operation()
        try:
            return self._send_message(
                text,
                request_context=request_context,
            )
        finally:
            self._operation_lock.release()

    def _send_message(
        self,
        text: str,
        *,
        request_context: str | None,
    ) -> tuple[EngineEvent, ...]:
        self._ensure_open()
        if not isinstance(text, str) or not text.strip():
            raise ValueError("message must be a non-blank string")
        if len(text) > self.config.max_user_message_chars:
            raise ValueError(
                f"message exceeds {self.config.max_user_message_chars} characters"
            )
        if request_context is not None:
            if not isinstance(request_context, str) or not request_context.strip():
                raise ValueError("request_context must be a non-blank string")
            try:
                context_bytes = request_context.encode("utf-8")
            except UnicodeEncodeError as error:
                raise ValueError("request_context must be valid UTF-8") from error
            if len(context_bytes) > self.config.max_request_context_bytes:
                raise ValueError(
                    "request_context exceeds "
                    f"{self.config.max_request_context_bytes} bytes"
                )
        if _contains_credential_material(
            text,
            self.provider,
        ) or (
            request_context is not None
            and _contains_credential_material(
                request_context,
                self.provider,
            )
        ):
            diagnostic = make_diagnostic(
                DiagnosticCode.INVALID_INPUT,
                (
                    "The message appears to contain credential material and "
                    "was not accepted or persisted."
                ),
                source="agent.engine",
            )
            return (
                self._diagnostic_event(diagnostic),
                self._event(
                    EngineEventType.ERROR,
                    {"diagnostic": diagnostic.to_dict()},
                ),
            )
        self._append_message(AssistantMessage("user", text))
        return self._run_provider_loop(
            request_context=request_context,
            allow_tools=True,
        )

    def _run_provider_loop(
        self,
        *,
        request_context: str | None,
        allow_tools: bool,
        trusted_terminal: tuple[str, str, str] | None = None,
    ) -> tuple[EngineEvent, ...]:
        """Run one provider user-turn and flush its round audit once."""

        batch: list[dict[str, Any]] = []
        with self._state_lock:
            if self._pending_round_audit is not None:
                raise RuntimeError("nested provider turns are not supported")
            self._pending_round_audit = batch
        try:
            return self._run_provider_loop_body(
                request_context=request_context,
                allow_tools=allow_tools,
                trusted_terminal=trusted_terminal,
            )
        finally:
            with self._state_lock:
                if self._pending_round_audit is batch:
                    self._pending_round_audit = None
            if self._defer_audit_persistence:
                self._queue_round_audit_batch(batch)
            else:
                self._persist_round_audit_entries(batch)

    def _run_provider_loop_body(
        self,
        *,
        request_context: str | None,
        allow_tools: bool,
        trusted_terminal: tuple[str, str, str] | None,
    ) -> tuple[EngineEvent, ...]:
        self._reset_provider_cancellation()
        events: list[EngineEvent] = []
        tool_calls_used = 0
        available_tools = (
            self.registry.available_definitions(self.session_id) if allow_tools else ()
        )
        initial = self.revisions.latest(self.session_id)
        turn_revision = 0 if initial is None else initial.revision
        turn_nonce = uuid.uuid4().hex
        # Hold terminal continuation text until tool publication is checked.
        defer_streamed_text = trusted_terminal is not None
        latest_user_request = self._latest_user_request()

        def local_assistant_recovery(
            recovery: str,
            *,
            storage_error: str,
            presentation_kind: str = "result_summary",
        ) -> tuple[EngineEvent, ...]:
            try:
                self._append_message(AssistantMessage("assistant", recovery))
            except ConversationStorageLimit:
                diagnostic = make_diagnostic(
                    DiagnosticCode.RESOURCE_LIMIT,
                    storage_error,
                    source="agent.engine",
                )
                events.append(self._diagnostic_event(diagnostic))
                events.append(
                    self._event(
                        EngineEventType.ERROR,
                        {"diagnostic": diagnostic.to_dict()},
                    )
                )
                return tuple(events)
            events.append(
                self._event(
                    EngineEventType.MESSAGE_STARTED,
                    {
                        "role": "assistant",
                        "presentation_kind": presentation_kind,
                    },
                )
            )
            events.append(
                self._event(
                    EngineEventType.MESSAGE_DELTA,
                    {"text": recovery},
                )
            )
            return tuple(events)

        def local_terminal_success_recovery() -> tuple[EngineEvent, ...]:
            summary = "" if trusted_terminal is None else trusted_terminal[1]
            return local_assistant_recovery(
                summary.strip() or "已完成。",
                storage_error=(
                    "The local proposal completion message could not fit in "
                    "the bounded conversation store."
                ),
            )

        def local_planar_retry_recovery() -> tuple[EngineEvent, ...]:
            recovery = planar_retry_limit_message(latest_user_request)
            return local_assistant_recovery(
                recovery,
                storage_error=(
                    "The local planar retry-limit message could not fit in "
                    "the bounded conversation store."
                ),
            )

        for _ in range(self.config.max_cloud_turns):
            if self._cancel_event.is_set():
                events.append(
                    self._event(
                        EngineEventType.OPERATION_CANCELLED,
                        {"scope": "provider"},
                    )
                )
                return tuple(events)
            try:
                request_result = self._request_provider_response(
                    request_context=request_context,
                    available_tools=available_tools,
                    defer_streamed_text=defer_streamed_text,
                    events=events,
                )
            except Exception as error:
                if self._cancel_event.is_set():
                    events.append(
                        self._event(
                            EngineEventType.OPERATION_CANCELLED,
                            {"scope": "provider"},
                        )
                    )
                    return tuple(events)
                diagnostic = _provider_diagnostic(error)
                events.append(self._diagnostic_event(diagnostic))
                events.append(
                    self._event(
                        EngineEventType.ERROR,
                        {"diagnostic": diagnostic.to_dict()},
                    )
                )
                return tuple(events)
            if self._cancel_event.is_set():
                events.append(
                    self._event(
                        EngineEventType.OPERATION_CANCELLED,
                        {"scope": "provider"},
                    )
                )
                return tuple(events)
            response = request_result.response
            response_error = self._provider_response_error(
                response.message,
                remaining_tool_calls=(self.config.max_tool_calls - tool_calls_used),
            )
            if response_error is None:
                response_error = self._provider_stream_error(request_result)
            if response_error is not None:
                events.append(self._diagnostic_event(response_error))
                events.append(
                    self._event(
                        EngineEventType.ERROR,
                        {"diagnostic": response_error.to_dict()},
                    )
                )
                return tuple(events)
            # Keep one privacy-safe record per provider round.  This is
            # intentionally written before dispatching any local tool so an
            # empty tool-call response remains auditable as "published but not
            # called".
            round_audit_entry = self._append_round_audit(
                available_tools,
                response.message.tool_calls,
            )

            if (
                trusted_terminal is not None
                and trusted_terminal[0] == "succeeded"
                and _message_calls_unpublished_tool(response.message, available_tools)
            ):
                return local_terminal_success_recovery()

            try:
                self._append_message(response.message)
            except ConversationStorageLimit:
                diagnostic = make_diagnostic(
                    DiagnosticCode.RESOURCE_LIMIT,
                    (
                        "The provider response could not fit in the bounded "
                        "local conversation store."
                    ),
                    source="agent.engine",
                )
                events.append(self._diagnostic_event(diagnostic))
                events.append(
                    self._event(
                        EngineEventType.ERROR,
                        {"diagnostic": diagnostic.to_dict()},
                    )
                )
                return tuple(events)
            for presentation_kind, content, already_streamed in (
                (
                    "process",
                    response.message.reasoning_content,
                    request_result.reasoning_presented,
                ),
                (
                    "result_summary",
                    response.message.content,
                    request_result.text_presented,
                ),
            ):
                if already_streamed or not content:
                    continue
                events.append(
                    self._event(
                        EngineEventType.MESSAGE_STARTED,
                        {
                            "role": "assistant",
                            "presentation_kind": presentation_kind,
                        },
                    )
                )
                events.append(
                    self._event(
                        EngineEventType.MESSAGE_DELTA,
                        {"text": content},
                    )
                )
            if not response.message.tool_calls:
                return tuple(events)

            for call in response.message.tool_calls:
                if self._cancel_event.is_set():
                    events.append(
                        self._event(
                            EngineEventType.OPERATION_CANCELLED,
                            {"scope": "provider"},
                        )
                    )
                    return tuple(events)
                tool_calls_used += 1
                if tool_calls_used > self.config.max_tool_calls:
                    diagnostic = make_diagnostic(
                        DiagnosticCode.TOOL_LIMIT_EXCEEDED,
                        (
                            "The provider exceeded the configured tool-call "
                            f"limit of {self.config.max_tool_calls}."
                        ),
                        source="agent.engine",
                    )
                    events.append(self._diagnostic_event(diagnostic))
                    return tuple(events)
                accepted_tool_names = round_audit_entry["tool_call_flags"].get(
                    "accepted_tool_names"
                )
                if isinstance(accepted_tool_names, list):
                    accepted_tool_names.append(call.name)
                current = self.revisions.latest(self.session_id)
                active_run = self._matching_active_run(current)
                context = ToolExecutionContext(
                    session_id=self.session_id,
                    expected_revision=0 if current is None else current.revision,
                    idempotency_key=_tool_idempotency_key(
                        call,
                        turn_revision=turn_revision,
                        turn_nonce=turn_nonce,
                    ),
                    completed_run=active_run,
                    turn_id=str(turn_nonce),
                )
                events.append(
                    self._event(
                        EngineEventType.TOOL_STARTED,
                        {
                            "tool": call.name,
                            "call_id": call.call_id,
                            "arguments": dict(call.arguments),
                        },
                    )
                )
                result = self._tool_result_cache.get(context.idempotency_key)
                if result is None:
                    result = self.registry.dispatch(
                        call.name,
                        call.arguments,
                        context,
                    )
                    self._tool_result_cache[context.idempotency_key] = result
                self._register_continuation_from_result(result)
                payload = result.to_json()
                if len(payload.encode("utf-8")) > self.config.max_tool_payload_bytes:
                    payload = _payload_limit_result(context).to_json()
                try:
                    self._append_message(
                        AssistantMessage(
                            "tool",
                            payload,
                            tool_call_id=call.call_id,
                        )
                    )
                except ConversationStorageLimit:
                    diagnostic = make_diagnostic(
                        DiagnosticCode.RESOURCE_LIMIT,
                        (
                            "A tool result could not fit in the bounded local "
                            "conversation store."
                        ),
                        source="agent.engine",
                    )
                    events.append(self._diagnostic_event(diagnostic))
                    events.append(
                        self._event(
                            EngineEventType.ERROR,
                            {"diagnostic": diagnostic.to_dict()},
                        )
                    )
                    return tuple(events)
                waits_for_confirmation = _tool_result_waits_for_confirmation(result)
                if waits_for_confirmation:
                    preview = geometry_proposal_preview_message(
                        call.name,
                        call.arguments,
                        result,
                        latest_user_request,
                    )
                    if preview is not None:
                        events.append(
                            self._event(
                                EngineEventType.MESSAGE_STARTED,
                                {
                                    "role": "assistant",
                                    "presentation_kind": "proposal_preview",
                                },
                            )
                        )
                        events.append(
                            self._event(
                                EngineEventType.MESSAGE_DELTA,
                                {"text": preview},
                            )
                        )
                events.append(
                    self._event(
                        EngineEventType.TOOL_COMPLETED,
                        {
                            "tool": call.name,
                            "call_id": call.call_id,
                            "result": result.to_dict(),
                        },
                    )
                )
                for diagnostic in result.diagnostics:
                    events.append(self._diagnostic_event(diagnostic))
                if (
                    call.name == "prepare_planar_construction_proposal"
                    and _planar_retry_limit_reached(result)
                ):
                    return local_planar_retry_recovery()
                if call.name in {
                    "inspect_abaqus",
                    "get_analysis_summary",
                    "validate_analysis",
                } and any(
                    item.code
                    in {
                        DiagnosticCode.WORKER_CRASH.value,
                        DiagnosticCode.WORKER_TIMEOUT.value,
                    }
                    for item in result.diagnostics
                ):
                    available_tools = ()
                presented_summary = self._summary_from_tool_result(result)
                if presented_summary is not None:
                    if not self._mark_summary_shown(presented_summary):
                        events.extend(self._cancelled_operation_events("inspection"))
                        return tuple(events)
                    events.append(
                        self._event(
                            EngineEventType.ANALYSIS_SUMMARY,
                            {"analysis_summary": presented_summary.to_dict()},
                        )
                    )
                if result.output_revision is not None:
                    self._active_run = None
                    if (
                        presented_summary is None
                        or presented_summary.revision != result.output_revision
                    ):
                        self._summary_shown_revision = None
                    events.append(self._state_event())
                if waits_for_confirmation:
                    return tuple(events)
                if available_tools:
                    available_tools = self.registry.available_definitions(
                        self.session_id
                    )
        diagnostic = make_diagnostic(
            DiagnosticCode.TOOL_LIMIT_EXCEEDED,
            (
                "The provider exceeded the configured cloud-turn limit of "
                f"{self.config.max_cloud_turns}."
            ),
            source="agent.engine",
        )
        events.append(self._diagnostic_event(diagnostic))
        return tuple(events)

    def _request_provider_response(
        self,
        *,
        request_context: str | None,
        available_tools: Sequence[ToolDefinition],
        defer_streamed_text: bool,
        events: list[EngineEvent],
    ) -> _ProviderRequestResult:
        """Receive one response, publishing accepted deltas as they arrive."""

        self._provider_active = True
        try:
            streamed_text_parts: list[str] = []
            streamed_reasoning_parts: list[str] = []
            stream_started = False
            reasoning_stream_started = False
            stream_presentation_kind: str | None = None

            def receive_delta(delta: str, *, reasoning: bool) -> None:
                nonlocal stream_started, reasoning_stream_started
                nonlocal stream_presentation_kind
                delta_kind = "reasoning" if reasoning else "text"
                if not isinstance(delta, str) or not delta:
                    raise ProviderMalformedResponseError(
                        f"provider stream emitted an invalid {delta_kind} delta"
                    )
                parts = streamed_reasoning_parts if reasoning else streamed_text_parts
                parts.append(delta)
                if defer_streamed_text:
                    return
                presentation_kind = "process" if reasoning else "result_summary"
                if stream_presentation_kind != presentation_kind:
                    events.append(
                        self._event(
                            EngineEventType.MESSAGE_STARTED,
                            {
                                "role": "assistant",
                                "presentation_kind": presentation_kind,
                            },
                        )
                    )
                    stream_presentation_kind = presentation_kind
                if reasoning:
                    reasoning_stream_started = True
                else:
                    stream_started = True
                events.append(
                    self._event(
                        EngineEventType.MESSAGE_DELTA,
                        {"text": delta},
                    )
                )

            def receive_text_delta(delta: str) -> None:
                receive_delta(delta, reasoning=False)

            def receive_reasoning_delta(delta: str) -> None:
                receive_delta(delta, reasoning=True)

            stream_completion = getattr(
                self.provider,
                "complete_stream",
                None,
            )
            if callable(stream_completion):
                provider_messages = self._provider_messages(
                    request_context=request_context,
                )
                if bool(
                    getattr(
                        self.provider,
                        "supports_reasoning_stream",
                        False,
                    )
                ):
                    response = stream_completion(
                        provider_messages,
                        available_tools,
                        receive_text_delta,
                        receive_reasoning_delta,
                    )
                else:
                    response = stream_completion(
                        provider_messages,
                        available_tools,
                        receive_text_delta,
                    )
            else:
                response = self.provider.complete(
                    self._provider_messages(
                        request_context=request_context,
                    ),
                    available_tools,
                )
        finally:
            self._provider_active = False
        return _ProviderRequestResult(
            response=response,
            streamed_text_parts=tuple(streamed_text_parts),
            streamed_reasoning_parts=tuple(streamed_reasoning_parts),
            text_presented=stream_started,
            reasoning_presented=reasoning_stream_started,
        )

    def _provider_stream_error(
        self,
        request_result: _ProviderRequestResult,
    ) -> Diagnostic | None:
        message = request_result.response.message
        if request_result.streamed_text_parts and "".join(
            request_result.streamed_text_parts
        ) != (message.content or ""):
            return make_diagnostic(
                DiagnosticCode.PROVIDER_MALFORMED_RESPONSE,
                "The provider stream did not match its final response.",
                source="agent.provider",
            )
        if request_result.streamed_reasoning_parts and "".join(
            request_result.streamed_reasoning_parts
        ) != (message.reasoning_content or ""):
            return make_diagnostic(
                DiagnosticCode.PROVIDER_MALFORMED_RESPONSE,
                "The provider reasoning stream did not match its final response.",
                source="agent.provider",
            )
        return None

    def continue_after_proposal(
        self,
        proposal_id: str,
        proposal_hash: str,
        source_turn_id: str,
        model_revision: int,
        status: str,
        summary: str = "",
    ) -> tuple[EngineEvent, ...]:
        """Consume one checkpoint and resume without synthesizing user input."""

        if not self._operation_lock.acquire(blocking=False):
            self._operation_started_event.set()
            return self._operation_in_progress_events()
        self._begin_operation()
        try:
            self._ensure_open()
            checkpoint = self._consume_continuation(proposal_id)
            normalized_status = str(status).strip().casefold()
            if checkpoint is None or normalized_status not in {
                "succeeded",
                "rejected",
                "failed",
                "stale",
                "cancelled",
            }:
                return ()
            if (
                checkpoint.session_id != self.session_id
                or checkpoint.proposal_hash != str(proposal_hash)
                or checkpoint.source_turn_id != str(source_turn_id)
                or checkpoint.model_revision != model_revision
            ):
                return ()
            bounded_summary = str(summary).strip()[:1_000]
            envelope = {
                "kind": "proposal_terminal",
                "session_id": checkpoint.session_id,
                "source_turn_id": checkpoint.source_turn_id,
                "proposal_id": checkpoint.proposal_id,
                "proposal_hash": checkpoint.proposal_hash,
                "model_revision": checkpoint.model_revision,
                "status": normalized_status,
                "summary": bounded_summary,
                "proposal_kind": checkpoint.proposal_kind,
            }
            self._append_message(
                AssistantMessage(
                    "system",
                    "Local GUI proposal terminal (trusted control result): "
                    + json.dumps(
                        envelope,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    + ". Treat this terminal status as authoritative. Do not "
                    "repeat the plan preview that was displayed immediately "
                    "before the proposal card, or repeat an earlier "
                    "pending-confirmation instruction or "
                    "ask the user to click, confirm, or accept this proposal "
                    "again. The local proposal card already presents the "
                    "terminal state, so do not add a standalone completion "
                    "acknowledgement. Evaluate the latest user request and "
                    "prior plan. A successful prerequisite does not complete "
                    "the overall request; continue only with the remaining "
                    "requested stage using current tools.",
                )
            )
            if normalized_status == "cancelled":
                return ()
            return self._run_provider_loop(
                request_context=None,
                allow_tools=normalized_status == "succeeded",
                trusted_terminal=(
                    normalized_status,
                    bounded_summary,
                    checkpoint.proposal_kind,
                ),
            )
        finally:
            self._operation_lock.release()

    def discard_continuation(self, proposal_id: str) -> bool:
        """Consume a checkpoint without invoking the provider."""

        return self._consume_continuation(proposal_id) is not None

    def _consume_continuation(
        self,
        proposal_id: str,
    ) -> ContinuationCheckpoint | None:
        with self._state_lock:
            return self._continuations.pop(str(proposal_id), None)

    def _register_continuation_from_result(self, result: ToolResult) -> None:
        if not result.ok or not isinstance(result.data, Mapping):
            return
        raw = result.data.get("continuation_checkpoint")
        if not isinstance(raw, Mapping):
            return
        try:
            checkpoint = ContinuationCheckpoint(
                session_id=str(raw["session_id"]),
                source_turn_id=str(raw["source_turn_id"]),
                proposal_id=str(raw["proposal_id"]),
                proposal_hash=str(raw["proposal_hash"]),
                model_revision=int(raw["model_revision"]),
                proposal_kind=str(raw.get("proposal_kind", "")),
            )
        except (KeyError, TypeError, ValueError):
            return
        if (
            checkpoint.session_id != self.session_id
            or not checkpoint.source_turn_id
            or not checkpoint.proposal_id
            or len(checkpoint.proposal_hash) != 64
            or checkpoint.model_revision < 0
        ):
            return
        with self._state_lock:
            existing = self._continuations.get(checkpoint.proposal_id)
            if existing is None:
                self._continuations[checkpoint.proposal_id] = checkpoint

    def confirm_revision(self) -> tuple[EngineEvent, ...]:
        """Confirm the exact current revision and run the complete pipeline."""

        if not self._operation_lock.acquire(blocking=False):
            self._operation_started_event.set()
            return self._operation_in_progress_events()
        self._begin_operation()
        try:
            return self._confirm_revision()
        finally:
            self._operation_lock.release()

    def _confirm_revision(self) -> tuple[EngineEvent, ...]:
        self._ensure_open()
        record = self.revisions.require_current(self.session_id)
        summary = self.registry.analysis_summary(record)
        if self._cancel_event.is_set():
            return self._cancelled_operation_events("confirmation")
        if summary.has_blocking_diagnostics:
            events = [self._diagnostic_event(item) for item in summary.diagnostics]
            events.append(
                self._event(
                    EngineEventType.CONFIRMATION_REQUIRED,
                    {
                        "accepted": False,
                        "revision": record.revision,
                        "revision_hash": record.revision_hash,
                    },
                )
            )
            return tuple(events)
        phase = self.get_snapshot().phase
        if phase != SessionPhase.AWAITING_CONFIRMATION:
            diagnostic = make_diagnostic(
                DiagnosticCode.CONFIRMATION_REQUIRED,
                (
                    "Confirmation is accepted only while the session is "
                    "awaiting confirmation; the current phase is "
                    f"{phase.value}."
                ),
                source="agent.engine",
            )
            return (
                self._diagnostic_event(diagnostic),
                self._event(
                    EngineEventType.CONFIRMATION_REQUIRED,
                    {
                        "accepted": False,
                        "reason": "invalid_session_phase",
                        "phase": phase.value,
                        "revision": record.revision,
                        "revision_hash": record.revision_hash,
                    },
                ),
            )
        if not self._summary_was_shown(record):
            if not self._mark_summary_shown(summary):
                return self._cancelled_operation_events("confirmation")
            return (
                self._event(
                    EngineEventType.ANALYSIS_SUMMARY,
                    {"analysis_summary": summary.to_dict()},
                ),
                self._event(
                    EngineEventType.CONFIRMATION_REQUIRED,
                    {
                        "accepted": False,
                        "reason": "summary_review_required",
                        "revision": record.revision,
                        "revision_hash": record.revision_hash,
                    },
                ),
            )
        self.confirmations.confirm(
            self.session_id,
            revision=record.revision,
            revision_hash=record.revision_hash,
        )
        events = [
            self._event(
                EngineEventType.STATE_CHANGED,
                {
                    "phase": SessionPhase.CONFIRMED.value,
                    "revision": record.revision,
                    "revision_hash": record.revision_hash,
                },
            ),
            self._event(
                EngineEventType.RUN_PROGRESS,
                {"stage": "worker_started", "revision": record.revision},
            ),
        ]
        return self._run_confirmed_record(record, events)

    def retry_transient_run(self) -> tuple[EngineEvent, ...]:
        """Retry one confirmed revision after an infrastructure failure."""

        if not self._operation_lock.acquire(blocking=False):
            self._operation_started_event.set()
            return self._operation_in_progress_events()
        self._begin_operation()
        try:
            return self._retry_transient_run()
        finally:
            self._operation_lock.release()

    def _retry_transient_run(self) -> tuple[EngineEvent, ...]:
        self._ensure_open()
        record = self.revisions.require_current(self.session_id)
        active_run = self._matching_active_run(record)
        if not self.confirmations.is_confirmed(
            self.session_id,
            revision=record.revision,
            revision_hash=record.revision_hash,
        ):
            diagnostic = make_diagnostic(
                DiagnosticCode.CONFIRMATION_REQUIRED,
                "The current revision is no longer confirmed.",
                source="agent.engine",
            )
            return (self._diagnostic_event(diagnostic),)
        if active_run is not None and not _is_transient_run(active_run):
            diagnostic = make_diagnostic(
                DiagnosticCode.INVALID_INPUT,
                (
                    "Retry is available only after a cancelled, timed-out, "
                    "or crashed worker run for the current revision."
                ),
                source="agent.engine",
            )
            return (
                self._diagnostic_event(diagnostic),
                self._event(
                    EngineEventType.ERROR,
                    {"diagnostic": diagnostic.to_dict()},
                ),
            )
        if self._cancel_event.is_set():
            return self._cancelled_operation_events("worker")
        events = [
            self._event(
                EngineEventType.RUN_PROGRESS,
                {
                    "stage": "worker_retry_started",
                    "revision": record.revision,
                    "previous_run_id": (
                        None if active_run is None else active_run.run_id
                    ),
                },
            )
        ]
        return self._run_confirmed_record(record, events)

    def _run_confirmed_record(
        self,
        record: RevisionRecord,
        events: list[EngineEvent],
    ) -> tuple[EngineEvent, ...]:
        self._running = True
        try:
            try:
                response = self.worker.run(
                    self.session_id,
                    revision=record.revision,
                    revision_hash=record.revision_hash,
                    idempotency_key=(
                        f"solve_{record.revision}_{record.revision_hash[:16]}"
                    ),
                    cancel_event=self._cancel_event,
                )
            except WorkerRunInProgressError:
                diagnostic = make_diagnostic(
                    DiagnosticCode.OPERATION_IN_PROGRESS,
                    "The confirmed revision already has an active worker.",
                    source="agent.worker",
                )
                events.append(self._diagnostic_event(diagnostic))
                self._running = False
                events.append(self._state_event())
                return tuple(events)
            except Exception:
                diagnostic = make_diagnostic(
                    DiagnosticCode.WORKER_CRASH,
                    "The local FEM worker could not complete its run protocol.",
                    source="agent.worker",
                )
                events.append(self._diagnostic_event(diagnostic))
                events.append(
                    self._event(
                        EngineEventType.ERROR,
                        {"diagnostic": diagnostic.to_dict()},
                    )
                )
                self._running = False
                events.append(self._state_event())
                return tuple(events)
        finally:
            self._running = False
        self._active_run = response
        for diagnostic in response.diagnostics:
            events.append(self._diagnostic_event(diagnostic))
        events.append(
            self._event(
                EngineEventType.RUN_COMPLETED,
                {
                    "run_id": response.run_id,
                    "status": response.status.value,
                    "result_summary": (
                        None
                        if response.result_summary is None
                        else response.result_summary.to_dict()
                    ),
                    "artifacts": [item.to_dict() for item in response.artifacts],
                },
            )
        )
        events.append(self._state_event())
        return tuple(events)

    def cancel_active_operation(self) -> tuple[EngineEvent, ...]:
        with self._state_lock:
            operation_active = self._operation_lock.locked()
            if operation_active:
                self._cancel_event.set()
            else:
                self._cancel_event.clear()
            if operation_active and self._provider_active:
                self._cancel_provider_request()
            scope = (
                "worker"
                if self._running
                else "provider"
                if self._provider_active
                else "operation"
                if operation_active
                else "idle"
            )
        return (
            self._event(
                EngineEventType.OPERATION_CANCELLED,
                {"scope": scope},
            ),
        )

    def get_snapshot(self) -> EngineSnapshot:
        current = self.revisions.latest(self.session_id)
        confirmed = (
            False
            if current is None
            else self.confirmations.is_confirmed(
                self.session_id,
                revision=current.revision,
                revision_hash=current.revision_hash,
            )
        )
        active_run = self._matching_active_run(current)
        phase = self._phase(current, confirmed, active_run)
        return EngineSnapshot(
            session_id=self.session_id,
            phase=phase,
            revision=0 if current is None else current.revision,
            revision_hash=None if current is None else current.revision_hash,
            confirmed=confirmed,
            active_run_id=(None if active_run is None else active_run.run_id),
            provider=self.provider.provider_name,
            model=self.provider.model_name,
            cloud_enabled=self.provider.provider_name != "fake",
            workspace=str(self.artifacts.root),
            artifacts=self.artifacts.list_artifacts(self.session_id),
        )

    def get_analysis_summary(self):
        """Return and mark the current deterministic summary."""

        if not self._operation_lock.acquire(blocking=False):
            raise RuntimeError("another provider or worker operation is active")
        self._begin_operation()
        try:
            summary = self._build_analysis_summary()
            if not self._mark_summary_shown(summary):
                raise RuntimeError("analysis summary was cancelled")
            return summary
        finally:
            self._operation_lock.release()

    def show_analysis_summary(self) -> tuple[EngineEvent, ...]:
        """Build the current summary as a cancellable UI event operation."""

        if not self._operation_lock.acquire(blocking=False):
            self._operation_started_event.set()
            return self._operation_in_progress_events()
        self._begin_operation()
        try:
            try:
                summary = self._build_analysis_summary()
            except Exception:
                if self._cancel_event.is_set():
                    return self._cancelled_operation_events("inspection")
                raise
            if not self._mark_summary_shown(summary):
                return self._cancelled_operation_events("inspection")
            return (
                self._event(
                    EngineEventType.ANALYSIS_SUMMARY,
                    {"analysis_summary": summary.to_dict()},
                ),
            )
        finally:
            self._operation_lock.release()

    def _build_analysis_summary(self) -> AnalysisSummary:
        record = self.revisions.require_current(self.session_id)
        return self.registry.analysis_summary(record)

    def list_artifacts(self) -> tuple[ArtifactRecord, ...]:
        return self.artifacts.list_artifacts(self.session_id)

    def close_session(self) -> tuple[EngineEvent, ...]:
        self.cancel_active_operation()
        with self._operation_lock:
            self._closed = True
            with self._state_lock:
                self._continuations.clear()
            self.flush_round_audit()
            return (self._state_event(),)

    def _provider_messages(
        self,
        *,
        request_context: str | None = None,
    ) -> tuple[AssistantMessage, ...]:
        state = self.get_snapshot()
        return build_provider_messages(
            self._system_prompt,
            self._history,
            {
                "session_id": state.session_id,
                "phase": state.phase.value,
                "revision": state.revision,
                "confirmed": state.confirmed,
                "active_run_id": state.active_run_id,
            },
            native_authoring=getattr(self.registry, "_dynamic_tools", None) is not None,
            authoring_snapshot=self.registry.provider_snapshot,
            max_messages=self.config.max_conversation_messages,
            request_context=request_context,
        )

    def _latest_user_request(self) -> str | None:
        return next(
            (
                message.content
                for message in reversed(self._history)
                if message.role == "user" and isinstance(message.content, str)
            ),
            None,
        )

    def _phase(
        self,
        current: RevisionRecord | None,
        confirmed: bool,
        active_run: WorkerResponse | None,
    ) -> SessionPhase:
        if self._running:
            return SessionPhase.RUNNING
        if active_run is not None:
            if active_run.status == RunStatus.SUCCEEDED:
                return SessionPhase.SOLVED
            if active_run.status == RunStatus.CANCELLED:
                return SessionPhase.CANCELLED
            return SessionPhase.FAILED
        if current is None:
            return SessionPhase.EMPTY
        if confirmed:
            return SessionPhase.CONFIRMED
        if current.spec.ready_for_confirmation:
            return SessionPhase.AWAITING_CONFIRMATION
        if current.spec.unit_context is not None or current.spec.requested_queries:
            return SessionPhase.DRAFT_READY
        return SessionPhase.INSPECTED

    def _state_event(
        self,
        record: RevisionRecord | None = None,
    ) -> EngineEvent:
        snapshot = self.get_snapshot()
        return self._event(
            EngineEventType.STATE_CHANGED,
            {
                "phase": snapshot.phase.value,
                "revision": snapshot.revision,
                "revision_hash": snapshot.revision_hash,
                "confirmed": snapshot.confirmed,
                "active_run_id": snapshot.active_run_id,
            },
        )

    def _diagnostic_event(self, diagnostic: Diagnostic) -> EngineEvent:
        return self._event(
            EngineEventType.DIAGNOSTIC,
            {"diagnostic": diagnostic.to_dict()},
        )

    def _operation_in_progress_events(self) -> tuple[EngineEvent, ...]:
        diagnostic = make_diagnostic(
            DiagnosticCode.OPERATION_IN_PROGRESS,
            "Another provider or worker operation is already active.",
            source="agent.engine",
        )
        return (
            self._diagnostic_event(diagnostic),
            self._event(
                EngineEventType.ERROR,
                {"diagnostic": diagnostic.to_dict()},
            ),
        )

    def _begin_operation(self) -> None:
        self._cancel_event.clear()
        self._operation_started_event.set()

    def reset_operation_start_signal(self) -> None:
        """Prepare the UI adapter to wait until a background call is armed."""

        self._operation_started_event.clear()

    def wait_for_operation_start(
        self,
        timeout_seconds: float | None = None,
    ) -> bool:
        """Wait until a public engine operation has armed cancellation."""

        return self._operation_started_event.wait(timeout_seconds)

    def _cancelled_operation_events(
        self,
        scope: str,
    ) -> tuple[EngineEvent, ...]:
        return (
            self._event(
                EngineEventType.OPERATION_CANCELLED,
                {"scope": scope},
            ),
        )

    def _mark_summary_shown(self, summary: AnalysisSummary) -> bool:
        with self._state_lock:
            if self._cancel_event.is_set():
                return False
            self._summary_shown_revision = (
                summary.revision,
                summary.revision_hash,
            )
            return True

    def _summary_was_shown(self, record: RevisionRecord) -> bool:
        return self._summary_shown_revision == (
            record.revision,
            record.revision_hash,
        )

    @staticmethod
    def _summary_from_tool_result(
        result: ToolResult,
    ) -> AnalysisSummary | None:
        if not result.ok or not isinstance(result.data, Mapping):
            return None
        raw_summary = result.data.get("analysis_summary")
        if not isinstance(raw_summary, Mapping):
            return None
        try:
            return AnalysisSummary.from_dict(raw_summary)
        except (TypeError, ValueError):
            return None

    def _event(
        self,
        event: EngineEventType,
        data: Mapping[str, Any],
    ) -> EngineEvent:
        created = EngineEvent(event, self.session_id, data, _utc_now())
        with self._state_lock:
            sinks = tuple(self._event_sinks)
        for sink in sinks:
            try:
                sink(created)
            except Exception:
                continue
        return created

    def _conversation_path(self) -> Path:
        return self._conversation_store.path(self.session_id)

    def _audit_path(self) -> Path:
        return self._audit_store.path(self.session_id)

    def _load_conversation(self) -> list[AssistantMessage]:
        return self._conversation_store.load(self.session_id)

    def _load_latest_run(self) -> WorkerResponse | None:
        current = self.revisions.latest(self.session_id)
        if current is None:
            return None
        session = self.artifacts.session_path(self.session_id)
        runs = safe_child(session, "runs")
        candidates: list[tuple[Path, WorkerResponse]] = []
        for directory in runs.iterdir():
            if not directory.is_dir():
                continue
            response = safe_child(
                directory,
                "logs",
                "worker-response.json",
            )
            if response.is_file():
                try:
                    loaded = load_verified_worker_response(
                        self.artifacts,
                        current,
                        directory.name,
                    )
                except (
                    ArtifactIntegrityError,
                    OSError,
                    TypeError,
                    ValueError,
                    WorkerResponseIntegrityError,
                ):
                    continue
                if self._matching_active_run(current, loaded) is not None:
                    candidates.append((response, loaded))
        if not candidates:
            return None
        _, latest = max(
            candidates,
            key=lambda item: item[0].stat().st_mtime_ns,
        )
        return latest

    def _matching_active_run(
        self,
        current: RevisionRecord | None,
        response: WorkerResponse | None = None,
    ) -> WorkerResponse | None:
        candidate = self._active_run if response is None else response
        if current is None or candidate is None:
            return None
        if (
            candidate.session_id != self.session_id
            or candidate.revision != current.revision
            or candidate.revision_hash != current.revision_hash
        ):
            return None
        return candidate

    def _append_message(self, message: AssistantMessage) -> None:
        self._history = self._conversation_store.append(
            self.session_id,
            self._history,
            message,
        )

    def _provider_response_error(
        self,
        message: AssistantMessage,
        *,
        remaining_tool_calls: int,
    ) -> Diagnostic | None:
        response_text_length = len(message.content or "") + len(
            message.reasoning_content or ""
        )
        if response_text_length > self.config.max_provider_message_chars:
            return make_diagnostic(
                DiagnosticCode.PROVIDER_MALFORMED_RESPONSE,
                "The provider response text exceeded the configured limit.",
                source="agent.provider",
            )
        if len(message.tool_calls) > remaining_tool_calls:
            return make_diagnostic(
                DiagnosticCode.TOOL_LIMIT_EXCEEDED,
                "The provider returned more tool calls than the remaining limit.",
                source="agent.provider",
            )
        for call in message.tool_calls:
            size = len(
                json.dumps(
                    call.arguments,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            )
            if size > self.config.max_tool_arguments_bytes:
                return make_diagnostic(
                    DiagnosticCode.PROVIDER_MALFORMED_RESPONSE,
                    "Provider tool arguments exceeded the configured limit.",
                    source="agent.provider",
                )
        return None

    def _append_round_audit(
        self,
        available_tools: Sequence[object],
        tool_calls: Sequence[ToolCall],
    ) -> dict[str, Any]:
        """Collect one bounded, privacy-safe record for a provider round."""

        snapshot_dict = provider_snapshot_dict(self.registry.provider_snapshot) or {}
        revision = snapshot_dict.get("session_revision")
        if type(revision) is not int:
            current = self.revisions.latest(self.session_id)
            revision = 0 if current is None else current.revision
        entry = build_round_audit_entry(
            self.session_id,
            snapshot_dict,
            revision,
            available_tools,
            tool_calls,
        )
        with self._state_lock:
            pending = self._pending_round_audit
            if pending is not None:
                pending.append(entry)
                return entry
        # Keep direct/internal callers durable too; provider turns install a
        # batch above so normal rounds never take this synchronous path.
        self._persist_round_audit_entries((entry,))
        return entry

    def _queue_round_audit_batch(
        self,
        entries: Sequence[Mapping[str, Any]],
    ) -> None:
        if not entries:
            return
        batch = tuple(dict(item) for item in entries)
        with self._state_lock:
            self._deferred_round_audit_batches.append(batch)

    def flush_round_audit(self) -> None:
        """Persist deferred round audits as one atomic user-turn write.

        GUI runtimes use deferred mode so the engine can deliver its terminal
        and final message-delta events before filesystem I/O.  Direct engine
        instances keep the default synchronous mode and therefore already
        have readable audit storage when ``send_message`` returns.
        """

        # Hold the write lock while claiming the queued batches.  A concurrent
        # close then waits for this attempt and can retry any batch requeued
        # after a failed write instead of observing a transiently empty queue.
        with self._audit_write_lock:
            with self._state_lock:
                batches = tuple(self._deferred_round_audit_batches)
                self._deferred_round_audit_batches.clear()
            if not batches:
                return
            entries = tuple(item for batch in batches for item in batch)
            try:
                self._persist_round_audit_entries_locked(entries)
            except Exception:
                with self._state_lock:
                    self._deferred_round_audit_batches = [
                        *batches,
                        *self._deferred_round_audit_batches,
                    ]
                raise

    def _persist_round_audit_entries(
        self,
        new_entries: Sequence[Mapping[str, Any]],
    ) -> None:
        """Atomically persist all collected entries for one provider turn."""

        with self._audit_write_lock:
            self._persist_round_audit_entries_locked(new_entries)

    def _persist_round_audit_entries_locked(
        self,
        new_entries: Sequence[Mapping[str, Any]],
    ) -> None:
        """Write one audit batch while the audit lock is held."""

        self._audit_store.persist(self.session_id, new_entries)

    def _reset_provider_cancellation(self) -> None:
        reset = getattr(self.provider, "reset_cancellation", None)
        if callable(reset):
            reset()

    def _cancel_provider_request(self) -> None:
        cancel = getattr(self.provider, "cancel_active_request", None)
        if callable(cancel):
            try:
                cancel()
            except Exception:
                pass

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("the Agent session is closed")


def _tool_idempotency_key(
    call: ToolCall,
    *,
    turn_revision: int,
    turn_nonce: str,
) -> str:
    encoded = json.dumps(
        {
            "turn_revision": turn_revision,
            "turn_nonce": turn_nonce,
            "call_id": call.call_id,
            "name": call.name,
            "arguments": call.arguments,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"tool_{hashlib.sha256(encoded).hexdigest()[:32]}"


def _planar_retry_limit_reached(result: ToolResult) -> bool:
    data = result.data
    if result.ok or not isinstance(data, Mapping):
        return False
    retry = data.get("retry")
    if not isinstance(retry, Mapping):
        return False
    attempt = retry.get("attempt")
    limit = retry.get("limit")
    return (
        isinstance(attempt, int)
        and not isinstance(attempt, bool)
        and isinstance(limit, int)
        and not isinstance(limit, bool)
        and attempt >= limit
        and retry.get("retryable") is False
        and isinstance(retry.get("blocker"), str)
        and bool(str(retry["blocker"]).strip())
    )


def _message_calls_unpublished_tool(
    message: AssistantMessage,
    available_tools: Sequence[object],
) -> bool:
    """Check calls against the tools published for this continuation."""

    published = {
        str(getattr(item, "name", ""))
        for item in available_tools
        if isinstance(getattr(item, "name", None), str)
    }
    return any(call.name not in published for call in message.tool_calls)


def _tool_result_waits_for_confirmation(result: ToolResult) -> bool:
    """Return true when a local proposal card is now the attention boundary."""

    if not result.ok or not isinstance(result.data, Mapping):
        return False
    return result.data.get("state") == "pending_confirmation" and isinstance(
        result.data.get("continuation_checkpoint"), Mapping
    )


def _payload_limit_result(context: ToolExecutionContext) -> ToolResult:
    diagnostic = make_diagnostic(
        DiagnosticCode.TOOL_LIMIT_EXCEEDED,
        "The local tool result exceeded the provider payload limit.",
        source="agent.engine",
    )
    return ToolResult(
        ok=False,
        session_id=context.session_id,
        input_revision=max(context.expected_revision, 0),
        idempotency_key=context.idempotency_key,
        summary=diagnostic.message,
        diagnostics=(diagnostic,),
    )


def _provider_diagnostic(error: Exception) -> Diagnostic:
    if isinstance(error, ProviderCredentialMissingError):
        code = DiagnosticCode.PROVIDER_AUTHENTICATION_FAILED
        message = (
            "Cloud provider credential is missing; set DEEPSEEK_API_KEY in "
            "the CLI environment."
        )
    elif isinstance(error, ProviderAuthenticationError):
        code = DiagnosticCode.PROVIDER_AUTHENTICATION_FAILED
        message = "Cloud provider authentication failed."
    elif isinstance(error, ProviderPaymentRequiredError):
        code = DiagnosticCode.PROVIDER_PAYMENT_REQUIRED
        message = "The cloud provider reported insufficient account balance."
    elif isinstance(error, ProviderRateLimitError):
        code = DiagnosticCode.PROVIDER_RATE_LIMITED
        message = "The cloud provider rate-limited the request."
    elif isinstance(error, ProviderTimeoutError):
        code = DiagnosticCode.PROVIDER_TIMEOUT
        message = "The cloud provider request timed out."
    elif isinstance(error, ProviderMalformedResponseError):
        code = DiagnosticCode.PROVIDER_MALFORMED_RESPONSE
        message = "The cloud provider returned a malformed response."
    elif isinstance(error, ProviderUnavailableError):
        code = DiagnosticCode.PROVIDER_UNAVAILABLE
        message = "The cloud provider is unavailable."
    else:
        code = DiagnosticCode.PROVIDER_UNAVAILABLE
        message = "The cloud provider request failed."
    return make_diagnostic(code, message, source="agent.provider")


def _is_transient_run(response: WorkerResponse) -> bool:
    if response.status not in {RunStatus.FAILED, RunStatus.CANCELLED}:
        return False
    error_codes = {
        diagnostic.code
        for diagnostic in response.diagnostics
        if diagnostic.severity == DiagnosticSeverity.ERROR
    }
    return bool(error_codes) and error_codes <= _TRANSIENT_RUN_DIAGNOSTICS


def _contains_credential_material(
    text: str,
    provider: CloudModelProvider,
) -> bool:
    if (
        _CREDENTIAL_ASSIGNMENT_PATTERN.search(text) is not None
        or _BEARER_PATTERN.search(text) is not None
        or _COMMON_API_KEY_PATTERN.search(text) is not None
    ):
        return True
    checker = getattr(provider, "contains_configured_credential", None)
    if not callable(checker):
        return False
    try:
        return bool(checker(text))
    except Exception:
        return True


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


__all__ = [
    "ContinuationCheckpoint",
    "AgentSessionEngine",
    "EngineConfig",
    "EngineEvent",
    "EngineEventType",
    "EngineSnapshot",
]
