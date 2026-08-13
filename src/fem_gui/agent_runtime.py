"""Qt-facing background runtime for the Phase 5 FEM Agent integration.

Engine construction, workspace context preparation, provider calls, tools,
confirmation runs, and shutdown stay outside the Qt main thread.  The GUI
receives validated ``AgentEvent`` objects through queued Qt signals.
"""

from __future__ import annotations

import os
import threading
import time
import weakref
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, wait as wait_for_futures
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from PySide6.QtCore import QObject, Qt, Signal

from fem_agent.artifacts import ArtifactStore, InputRejectedError
from fem_agent.authoring import ProposalState, RequirementReview
from fem_agent.authoring_runtime import (
    AuthoringTurnSnapshot,
    AuthoringWorkflowController,
)
from fem_agent.config import (
    ConfigError,
    LocalAgentConfig,
    find_main_config,
    resolve_local_config,
)
from fem_agent.engine import (
    AgentSessionEngine,
    EngineEvent,
    EngineEventType,
)
from fem_agent.providers.base import CloudModelProvider, ToolDefinition
from fem_agent.providers.deepseek import DeepSeekProvider
from fem_agent.providers.fake import FakeProvider
from fem_agent.schemas import ToolResult
from fem_agent.tools.registry import (
    DynamicToolRegistry,
    ToolExecutionContext,
)

from .agent_context import (
    PreparedWorkspaceContext,
    WorkspaceContextError,
    prepare_workspace_context,
)
from .agent_events import (
    AgentEvent,
    EventType,
    redact_absolute_paths,
    safe_tool_summary,
)
from .agent_workspace import WorkspaceFileReference


class AgentEnginePort(Protocol):
    """Small engine surface used by the GUI runtime."""

    session_id: str

    def reset_operation_start_signal(self) -> None: ...

    def wait_for_operation_start(
        self,
        timeout_seconds: float | None = None,
    ) -> bool: ...

    def send_message(
        self,
        text: str,
        *,
        request_context: str | None = None,
    ) -> tuple[EngineEvent, ...]: ...

    def continue_after_proposal(
        self,
        proposal_id: str,
        proposal_hash: str,
        source_turn_id: str,
        model_revision: int,
        status: str,
        summary: str = "",
    ) -> tuple[EngineEvent, ...]: ...

    def discard_continuation(self, proposal_id: str) -> bool: ...

    def create_session(self) -> tuple[EngineEvent, ...]: ...

    def attach_artifact(
        self,
        artifact_id: str,
        *,
        replace_existing: bool = False,
    ) -> tuple[EngineEvent, ...]: ...

    def confirm_revision(self) -> tuple[EngineEvent, ...]: ...

    def cancel_active_operation(self) -> tuple[EngineEvent, ...]: ...

    def close_session(self) -> tuple[EngineEvent, ...]: ...

    def flush_round_audit(self) -> None: ...

    def get_snapshot(self) -> Any: ...


ProviderFactory = Callable[[], CloudModelProvider]
EngineFactory = Callable[
    [Path, CloudModelProvider, Callable[[EngineEvent], None]],
    AgentEnginePort,
]

_MAX_MESSAGE_DELTA_CHARACTERS = 8_000
_MAX_ADAPTIVE_MESSAGE_DELTA_CHARACTERS = 64_000
_MESSAGE_DELTA_FRAME_SECONDS = 0.03
_AUTHORING_TOOL_OWNER_TIMEOUT_SECONDS = 30.0
_RUNTIME_SHUTDOWN_TIMEOUT_SECONDS = 2.0

_TOOL_DISPLAY_NAMES = {
    "show_capabilities": "检查 Agent 能力",
    "inspect_abaqus": "检查 Abaqus 模型",
    "set_unit_context": "设置单位上下文",
    "set_result_requests": "设置结果请求",
    "get_analysis_summary": "生成分析摘要",
    "validate_analysis": "验证分析模型",
    "solve_confirmed_analysis": "请求开始求解",
    "query_results": "查询求解结果",
    "export_results": "检查结果导出",
    "list_artifacts": "列出 Agent 工件",
}


def _diagnostic_identity(
    raw_diagnostic: object,
) -> tuple[str, str, str] | None:
    if not isinstance(raw_diagnostic, Mapping):
        return None
    return (
        str(raw_diagnostic.get("code", "")),
        str(raw_diagnostic.get("severity", "")),
        str(raw_diagnostic.get("message", "")),
    )


class AgentRuntimeConfigurationError(RuntimeError):
    """The configured GUI provider cannot be enabled safely."""


@dataclass
class _ToolActivity:
    name: str
    started_at: float
    terminal: bool = False


@dataclass
class _TurnContext:
    generation: int
    session_id: str
    turn_id: str
    message_counter: int = 0
    active_message_id: str | None = None
    pending_confirmation: dict[str, object] | None = None
    terminal: bool = False
    failure_reason: str | None = None
    diagnostic_count: int = 0
    operation: str = "message"
    tools: dict[str, _ToolActivity] = field(default_factory=dict)
    solve_call_id: str | None = None
    solve_succeeded: bool = False
    seen_engine_events: dict[int, EngineEvent] = field(default_factory=dict)
    pending_delta_chunks: list[str] = field(default_factory=list)
    pending_delta_characters: int = 0
    pending_delta_started_at: float | None = None
    max_pending_delta_characters: int = 0
    max_pending_delta_wait_seconds: float = 0.0
    delta_timer: threading.Timer | None = None
    embedded_tool_diagnostics: list[
        tuple[str, str, str]
    ] = field(default_factory=list)


@dataclass
class _ProposalLifecycle:
    session_id: str
    turn_id: str
    proposal_id: str
    proposal_hash: str
    proposal_kind: str
    source_turn_id: str
    model_revision: int
    state: ProposalState = ProposalState.PENDING_CONFIRMATION
    progress: float = 0.0


@dataclass(frozen=True)
class _ProposalContinuation:
    session_id: str
    proposal_id: str
    proposal_hash: str
    source_turn_id: str
    model_revision: int
    status: str
    summary: str


@dataclass
class _AuthoringToolInvocation:
    name: str
    arguments: Mapping[str, Any]
    context: ToolExecutionContext
    completed: threading.Event = field(default_factory=threading.Event)
    result: ToolResult | None = None
    error: BaseException | None = None
    cancelled: bool = False
    started: bool = False
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def claim(self) -> bool:
        with self._lock:
            if self.cancelled or self.completed.is_set():
                return False
            self.started = True
            return True

    def cancel(self, error: BaseException) -> bool:
        with self._lock:
            if self.completed.is_set():
                return False
            self.cancelled = True
            self.error = error
            self.completed.set()
            return True

    def finish(
        self,
        *,
        result: ToolResult | None = None,
        error: BaseException | None = None,
    ) -> None:
        with self._lock:
            if self.cancelled or self.completed.is_set():
                return
            self.result = result
            self.error = error
            self.completed.set()


class _QtAuthoringToolProxy(DynamicToolRegistry):
    """Expose definitions off-thread while dispatching on the Qt owner."""

    def __init__(
        self,
        runtime: "QtAgentRuntime",
        controller: AuthoringWorkflowController,
    ) -> None:
        self._runtime_ref = weakref.ref(runtime)
        self._controller = controller

    def _runtime_owner(self) -> "QtAgentRuntime":
        runtime = self._runtime_ref()
        if runtime is None:
            raise RuntimeError("Qt Agent runtime is no longer available")
        return runtime

    @property
    def definitions(self):
        # The provider worker only reads this immutable owner-thread cache.
        # Calling ``controller.definitions`` here would cross the Qt boundary
        # on every provider request.
        return self._runtime_owner()._authoring_tool_definitions()

    @property
    def provider_snapshot(self) -> AuthoringTurnSnapshot:
        return self._runtime_owner().authoring_turn_snapshot

    def refresh_turn_snapshot(
        self,
        published_tool_names: tuple[str, ...] = (),
    ) -> AuthoringTurnSnapshot:
        del published_tool_names
        return self.provider_snapshot

    def dispatch(
        self,
        name: str,
        arguments: Mapping[str, Any],
        context: ToolExecutionContext,
    ) -> ToolResult:
        return self._runtime_owner()._dispatch_authoring_tool(
            name,
            arguments,
            context,
        )


def _default_provider_factory() -> CloudModelProvider:
    config_path = find_main_config()
    if config_path is None:
        raise AgentRuntimeConfigurationError(
            "未找到 FEM Agent 配置文件。"
        )
    try:
        file_config = LocalAgentConfig.load(config_path)
        resolved = resolve_local_config(
            file_config,
            environ=os.environ,
        )
        if not resolved.enabled:
            raise AgentRuntimeConfigurationError(
                "FEM Agent 配置尚未启用；请将 enabled 设为 true。"
            )
        if resolved.provider.casefold() == "fake":
            return FakeProvider(model=resolved.model)
        if resolved.provider.casefold() != "deepseek":
            raise AgentRuntimeConfigurationError(
                "GUI 当前只支持 DeepSeek 或 Fake Provider。"
            )
        resolved.require_api_key()
        return DeepSeekProvider(
            resolved.provider_config(),
            environ=resolved.provider_environment(),
        )
    except AgentRuntimeConfigurationError:
        raise
    except ConfigError as exc:
        raise AgentRuntimeConfigurationError(
            "FEM Agent 配置无效或缺少凭据。"
        ) from exc


def _default_engine_factory(
    root: Path,
    provider: CloudModelProvider,
    event_sink: Callable[[EngineEvent], None],
) -> AgentEnginePort:
    return AgentSessionEngine(
        root,
        provider,
        event_sink=event_sink,
        defer_audit_persistence=True,
    )


class QtAgentRuntime(QObject):
    """Serialize one Agent session on background threads for a Qt consumer."""

    agentEventReady = Signal(object)
    sessionReset = Signal(str)
    busyChanged = Signal(bool)
    providerReady = Signal(str, str)
    solveFinished = Signal(int, str, bool)
    operationRejected = Signal(str)
    eventRejected = Signal(str)
    shutdownFinished = Signal()
    authoringToolRequested = Signal(object)

    def __init__(
        self,
        agent_data_root: str | Path,
        parent: QObject | None = None,
        *,
        provider_factory: ProviderFactory | None = None,
        engine_factory: EngineFactory | None = None,
        authoring_controller: AuthoringWorkflowController | None = None,
    ) -> None:
        super().__init__(parent)
        self.agent_data_root = Path(os.path.abspath(os.fspath(agent_data_root)))
        self._provider_factory = provider_factory or _default_provider_factory
        self._authoring_controller = authoring_controller
        self._owner_thread_id = threading.get_ident()
        self._dynamic_tools = (
            None
            if authoring_controller is None
            else _QtAuthoringToolProxy(self, authoring_controller)
        )
        if engine_factory is None:
            dynamic_tools = self._dynamic_tools

            def default_factory(
                root: Path,
                provider: CloudModelProvider,
                event_sink: Callable[[EngineEvent], None],
            ) -> AgentEnginePort:
                return AgentSessionEngine(
                    root,
                    provider,
                    event_sink=event_sink,
                    dynamic_tools=dynamic_tools,
                    defer_audit_persistence=True,
                )

            self._engine_factory = default_factory
        else:
            self._engine_factory = engine_factory
        self._session_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="fem-agent-session",
        )
        self._control_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="fem-agent-control",
        )
        self._lock = threading.RLock()
        self._engine_ready = threading.Event()
        self._engine: AgentEnginePort | None = None
        self._gui_session_id: str | None = None
        self._sequence = 0
        self._turn_counter = 0
        self._generation = 0
        self._active_turn: _TurnContext | None = None
        self._last_stream_backlog_metrics: dict[str, int | float] = {
            "pending_characters": 0,
            "max_pending_characters": 0,
            "max_wait_seconds": 0.0,
        }
        self._busy = False
        self._cancel_requested = False
        self._shutdown = False
        self._authoring_invocations: dict[int, _AuthoringToolInvocation] = {}
        self._authoring_snapshot = AuthoringTurnSnapshot.unavailable()
        self._authoring_definitions: tuple[ToolDefinition, ...] = ()
        self._authoring_snapshot_blocked = False
        self._proposal_lifecycles: dict[str, _ProposalLifecycle] = {}
        self._pending_continuation: _ProposalContinuation | None = None
        self._attached_input_key: tuple[str, str, str] | None = None
        self._target_document_id: str | None = None
        self._target_session_id: str | None = None
        self.sessionReset.connect(
            self._reset_authoring_controller_for_session,
            Qt.ConnectionType.QueuedConnection,
        )
        self.authoringToolRequested.connect(
            self._execute_authoring_tool,
            Qt.ConnectionType.QueuedConnection,
        )

    @property
    def authoring_controller(self) -> AuthoringWorkflowController | None:
        return self._authoring_controller

    @property
    def busy(self) -> bool:
        with self._lock:
            return self._busy

    @property
    def is_shutdown(self) -> bool:
        with self._lock:
            return self._shutdown

    @property
    def session_id(self) -> str | None:
        with self._lock:
            return self._gui_session_id

    @property
    def target_identity(self) -> tuple[str, str] | None:
        """Return the workspace document/session currently bound to Agent."""

        with self._lock:
            if (
                self._target_document_id is None
                or self._target_session_id is None
            ):
                return None
            return self._target_document_id, self._target_session_id

    def bind_target(
        self,
        document_id: str | int,
        session_id: str,
    ) -> None:
        """Bind the single runtime to an idle workspace document identity."""

        normalized_document = str(document_id)
        normalized_session = str(session_id)
        if not normalized_document or not normalized_session:
            raise ValueError("Agent target document/session identities are required")
        with self._lock:
            if self._shutdown:
                raise RuntimeError("Agent runtime is shut down")
            if self._busy:
                raise RuntimeError("Agent runtime must be idle before rebinding")
            self._target_document_id = normalized_document
            self._target_session_id = normalized_session

    @property
    def authoring_turn_snapshot(self) -> AuthoringTurnSnapshot:
        """Return the immutable owner-thread projection used by the provider."""

        with self._lock:
            return self._authoring_snapshot

    def _authoring_tool_definitions(self) -> tuple[ToolDefinition, ...]:
        with self._lock:
            return self._authoring_definitions

    def _invalidate_authoring_tool_cache_owner_thread(
        self,
        *,
        invalidate_controller: bool = True,
    ) -> AuthoringTurnSnapshot:
        """Atomically drop an invalid owner-thread projection."""

        self._require_owner_thread()
        previous = self.authoring_turn_snapshot
        controller = self._authoring_controller
        snapshot: AuthoringTurnSnapshot | None = None
        if controller is not None:
            if invalidate_controller:
                invalidate = getattr(controller, "invalidate_turn_snapshot", None)
                if callable(invalidate):
                    try:
                        candidate = invalidate()
                    except Exception:
                        candidate = None
                    if isinstance(candidate, AuthoringTurnSnapshot):
                        snapshot = candidate
            if snapshot is None:
                candidate = getattr(controller, "turn_snapshot", None)
                if isinstance(candidate, AuthoringTurnSnapshot):
                    snapshot = candidate
        generation = previous.snapshot_generation + 1
        if snapshot is None or snapshot.snapshot_generation < generation:
            snapshot = AuthoringTurnSnapshot.unavailable(generation=generation)
        global_read_definitions = (
            ()
            if controller is None
            else tuple(
                item
                for item in controller.definitions
                if item.name == "read_workspace_documents"
            )
        )
        with self._lock:
            self._authoring_snapshot = snapshot
            self._authoring_definitions = global_read_definitions
            self._authoring_snapshot_blocked = True
        return snapshot

    def refresh_authoring_turn_snapshot_from_gui(self) -> AuthoringTurnSnapshot:
        """Observe GUI authoring state and publish a new immutable cache.

        This method is owner-thread-only.  It is intentionally public so the
        main window can refresh the cache after a document transition without
        exposing a Session object to the provider worker.
        """

        self._require_owner_thread()
        controller = self._authoring_controller
        try:
            if controller is None:
                snapshot = AuthoringTurnSnapshot.unavailable()
                definitions: tuple[ToolDefinition, ...] = ()
            else:
                # ``create_session_authoring_workflow_controller`` and every
                # GUI document transition already observe the typed context on
                # this owner thread.  Reuse that observation for the normal
                # turn path; only an as-yet-unobserved controller needs a
                # context-reader call.
                if not controller.turn_snapshot.available:
                    controller.refresh_turn_snapshot()
                definitions = tuple(controller.definitions)
                snapshot = controller.set_published_tool_names(
                    tuple(item.name for item in definitions)
                )
        except Exception:
            self._invalidate_authoring_tool_cache_owner_thread()
            raise
        with self._lock:
            self._authoring_snapshot = snapshot
            self._authoring_definitions = definitions
            if snapshot.available:
                self._authoring_snapshot_blocked = False
        return snapshot

    def _try_refresh_authoring_turn_snapshot_from_gui(self) -> None:
        try:
            self.refresh_authoring_turn_snapshot_from_gui()
        except Exception:
            if self.authoring_turn_snapshot.available:
                self._invalidate_authoring_tool_cache_owner_thread()

    def _publish_authoring_tool_cache_owner_thread(self) -> None:
        """Refresh only stage/tool metadata after an already-observed call."""

        self._require_owner_thread()
        controller = self._authoring_controller
        if controller is None:
            return
        controller_snapshot = controller.turn_snapshot
        with self._lock:
            blocked = self._authoring_snapshot_blocked
        if blocked and not controller_snapshot.available:
            # Binding invalidation (or a failed owner-thread projection) must
            # not republish definitions derived from the old document.  A
            # later owner-thread refresh/observe call clears this block.
            with self._lock:
                self._authoring_snapshot = controller_snapshot
                self._authoring_definitions = tuple(
                    item
                    for item in controller.definitions
                    if item.name == "read_workspace_documents"
                )
            return
        try:
            definitions = tuple(controller.definitions)
            snapshot = controller.set_published_tool_names(
                tuple(item.name for item in definitions)
            )
        except Exception:
            self._invalidate_authoring_tool_cache_owner_thread()
            raise
        with self._lock:
            self._authoring_snapshot = snapshot
            self._authoring_definitions = definitions
            if snapshot.available:
                self._authoring_snapshot_blocked = False

    def _try_publish_authoring_tool_cache_owner_thread(self) -> None:
        try:
            self._publish_authoring_tool_cache_owner_thread()
        except Exception:
            if self.authoring_turn_snapshot.available:
                self._invalidate_authoring_tool_cache_owner_thread()

    @property
    def stream_backlog_metrics(self) -> dict[str, int | float]:
        """Return bounded coalescer backlog metrics for the active/last turn."""

        with self._lock:
            if self._active_turn is None:
                return dict(self._last_stream_backlog_metrics)
            return self._stream_backlog_metrics_locked(self._active_turn)

    def send_message(
        self,
        text: str,
        references: Sequence[WorkspaceFileReference] = (),
        *,
        workspace_root: str | os.PathLike[str] | None = None,
    ) -> bool:
        """Queue one user turn; file reads remain on the session worker."""
        if not isinstance(text, str) or not text.strip():
            self.operationRejected.emit("消息不能为空")
            return False
        provider_text = self._provider_safe_text(
            text,
            references,
            workspace_root=workspace_root,
        )
        # ``send_message`` is called by the Qt UI thread.  Refresh the
        # owner-thread projection before queuing the worker so the provider
        # never has to synchronously inspect GUI/Session state.
        self._publish_authoring_tool_cache_owner_thread()
        reference_snapshot = tuple(references)
        with self._lock:
            if self._shutdown:
                self.operationRejected.emit("Agent 后台已关闭")
                return False
            if self._busy:
                self.operationRejected.emit("当前会话已有操作正在运行")
                return False
            self._busy = True
            self._cancel_requested = False
            self._generation += 1
            generation = self._generation
            self._session_executor.submit(
                self._run_send,
                generation,
                provider_text,
                reference_snapshot,
            )
        self.busyChanged.emit(True)
        return True

    def confirm_solve(
        self,
        revision: int,
        revision_hash: str,
    ) -> bool:
        """Queue one revision-bound user confirmation and solve operation."""

        if (
            isinstance(revision, bool)
            or not isinstance(revision, int)
            or revision < 1
            or not isinstance(revision_hash, str)
            or len(revision_hash) != 64
            or any(
                character not in "0123456789abcdefABCDEF"
                for character in revision_hash
            )
        ):
            self.operationRejected.emit("求解确认信息无效")
            return False
        with self._lock:
            if self._shutdown:
                self.operationRejected.emit("Agent 后台已关闭")
                return False
            if self._busy:
                self.operationRejected.emit("当前会话已有操作正在运行")
                return False
            if self._engine is None:
                self.operationRejected.emit("当前没有可确认的 Agent 会话")
                return False
            self._busy = True
            self._cancel_requested = False
            self._generation += 1
            generation = self._generation
            self._session_executor.submit(
                self._run_confirm,
                generation,
                revision,
                revision_hash.casefold(),
            )
        self.busyChanged.emit(True)
        return True

    def new_session(self) -> bool:
        """Queue ``engine.create_session`` on the serialized worker."""
        with self._lock:
            if self._shutdown:
                self.operationRejected.emit("Agent 后台已关闭")
                return False
            if self._busy:
                self.operationRejected.emit("当前会话已有操作正在运行")
                return False
            self._busy = True
            self._cancel_requested = False
            self._generation += 1
            generation = self._generation
            self._session_executor.submit(
                self._run_new_session,
                generation,
            )
        self.busyChanged.emit(True)
        return True

    def cancel(self) -> bool:
        """Request cooperative cancellation without blocking the UI thread."""
        with self._lock:
            if self._shutdown or not self._busy:
                return False
            if self._cancel_requested:
                return True
            self._cancel_requested = True
            generation = self._generation
            self._control_executor.submit(
                self._run_cancel,
                generation,
            )
        controller = self._authoring_controller
        if controller is not None:
            controller.cancel_turn()
            self._try_publish_authoring_tool_cache_owner_thread()
        return True

    def resolve_requirement_review_from_gui(
        self,
        review: RequirementReview,
    ) -> None:
        self._require_owner_thread()
        controller = self._authoring_controller
        if controller is None:
            raise RuntimeError("authoring controller is not configured")
        controller.resolve_requirement_review(review)
        self._try_publish_authoring_tool_cache_owner_thread()

    def record_authoring_proposal_state_from_gui(
        self,
        operation: str,
        state: ProposalState | str,
        message: str = "",
    ) -> None:
        self._require_owner_thread()
        controller = self._authoring_controller
        if controller is None:
            raise RuntimeError("authoring controller is not configured")
        controller.record_proposal_state(operation, state, message)
        self._try_publish_authoring_tool_cache_owner_thread()

    def record_proposal_lifecycle_from_gui(
        self,
        proposal_id: str,
        proposal_hash: str,
        agent_session_id: str,
        turn_id: str,
        state: ProposalState | str,
        message: str = "",
        *,
        progress: float | None = None,
    ) -> bool:
        """Emit the event-only lifecycle for one GUI-controlled proposal."""

        self._require_owner_thread()
        normalized = ProposalState(state)
        emitted: list[AgentEvent] = []
        continuation: _ProposalContinuation | None = None
        discard_checkpoint = False
        with self._lock:
            lifecycle = self._proposal_lifecycles.get(str(proposal_id))
            if lifecycle is None:
                self.eventRejected.emit("提案生命周期尚未登记")
                return False
            identity_matches = (
                lifecycle.proposal_hash == str(proposal_hash)
                and lifecycle.session_id == str(agent_session_id)
                and str(turn_id) in {
                    lifecycle.turn_id,
                    lifecycle.source_turn_id,
                }
                and (
                    self._gui_session_id is None
                    or self._gui_session_id == lifecycle.session_id
                )
            )
            if not identity_matches:
                normalized = ProposalState.STALE
                message = "proposal hash、Agent session 或 turn identity 不匹配"
            if lifecycle.state in {
                ProposalState.REJECTED,
                ProposalState.STALE,
                ProposalState.SUCCEEDED,
                ProposalState.FAILED,
                ProposalState.CANCELLED,
            }:
                self.eventRejected.emit("已拒绝重复或迟到的提案终态")
                return False
            context = _TurnContext(
                generation=self._generation,
                session_id=lifecycle.session_id,
                turn_id=lifecycle.turn_id,
            )

            def append(event_type: EventType, payload: Mapping[str, object]) -> None:
                emitted.append(
                    self._new_event_locked(context, event_type, payload)
                )

            identity = {
                "proposal_id": lifecycle.proposal_id,
                "proposal_hash": lifecycle.proposal_hash,
            }
            if normalized in {
                ProposalState.ACCEPTED,
                ProposalState.RUNNING,
                ProposalState.SUCCEEDED,
            } and lifecycle.state is ProposalState.PENDING_CONFIRMATION:
                append(EventType.PROPOSAL_ACCEPTED, identity)
                lifecycle.state = ProposalState.ACCEPTED
            if normalized in {
                ProposalState.RUNNING,
                ProposalState.SUCCEEDED,
            } and lifecycle.state is ProposalState.ACCEPTED:
                append(EventType.PROPOSAL_STARTED, identity)
                lifecycle.state = ProposalState.RUNNING

            text = str(message).strip()
            terminal_text = text or (
                "已完成"
                if normalized is ProposalState.SUCCEEDED
                else normalized.value
            )
            if normalized is ProposalState.RUNNING:
                if progress is not None:
                    lifecycle.progress = float(progress)
                if text and lifecycle.state is ProposalState.RUNNING:
                    append(
                        EventType.PROPOSAL_PROGRESS,
                        {
                            **identity,
                            "progress": lifecycle.progress,
                            "message": text,
                        },
                    )
            elif normalized is ProposalState.SUCCEEDED:
                append(
                    EventType.PROPOSAL_SUCCEEDED,
                    {**identity, "summary": terminal_text},
                )
                lifecycle.state = normalized
                lifecycle.progress = 1.0
            elif normalized in {
                ProposalState.REJECTED,
                ProposalState.STALE,
                ProposalState.FAILED,
                ProposalState.CANCELLED,
            }:
                event_type = {
                    ProposalState.REJECTED: EventType.PROPOSAL_REJECTED,
                    ProposalState.STALE: EventType.PROPOSAL_STALE,
                    ProposalState.FAILED: EventType.PROPOSAL_FAILED,
                    ProposalState.CANCELLED: EventType.PROPOSAL_CANCELLED,
                }[normalized]
                append(
                    event_type,
                    {**identity, "reason": terminal_text},
                )
                lifecycle.state = normalized
            if normalized in {
                ProposalState.SUCCEEDED,
                ProposalState.REJECTED,
                ProposalState.STALE,
                ProposalState.FAILED,
                ProposalState.CANCELLED,
            }:
                controller = self._authoring_controller
                binding_identity = (
                    None
                    if controller is None
                    else controller.binding_identity
                )
                revision_changed = (
                    binding_identity is not None
                    and binding_identity[2] != lifecycle.model_revision
                )
                if normalized is ProposalState.CANCELLED or revision_changed:
                    discard_checkpoint = True
                else:
                    continuation = _ProposalContinuation(
                        session_id=lifecycle.session_id,
                        proposal_id=lifecycle.proposal_id,
                        proposal_hash=lifecycle.proposal_hash,
                        source_turn_id=lifecycle.source_turn_id,
                        model_revision=lifecycle.model_revision,
                        status=normalized.value,
                        summary=self._provider_safe_text(
                            terminal_text,
                            (),
                            workspace_root=None,
                        ),
                    )
        self._emit_events(emitted)
        self._try_publish_authoring_tool_cache_owner_thread()
        if discard_checkpoint:
            self._discard_proposal_continuation(str(proposal_id))
        elif continuation is not None:
            self._queue_proposal_continuation(continuation)
        return bool(emitted)

    def synchronize_event_projection_from_gui(self, presentation: object) -> None:
        """Restore proposal identities when the GUI replays a complete log."""

        self._require_owner_thread()
        session_id = str(getattr(presentation, "session_id", ""))
        last_sequence = int(getattr(presentation, "last_sequence", 0))
        lifecycles: dict[str, _ProposalLifecycle] = {}
        for turn in tuple(getattr(presentation, "turns", ())):
            turn_id = str(getattr(turn, "turn_id", ""))
            for proposal in tuple(getattr(turn, "proposals", ())):
                state = ProposalState(str(getattr(proposal, "status").value))
                lifecycles[str(proposal.proposal_id)] = _ProposalLifecycle(
                    session_id=session_id,
                    turn_id=turn_id,
                    proposal_id=str(proposal.proposal_id),
                    proposal_hash=str(proposal.proposal_hash),
                    proposal_kind=str(proposal.proposal_kind),
                    source_turn_id=turn_id,
                    model_revision=int(proposal.base_session_revision),
                    state=state,
                    progress=float(proposal.progress),
                )
        with self._lock:
            self._sequence = last_sequence
            self._proposal_lifecycles = lifecycles

    def proposal_source_turn_from_gui(
        self,
        proposal_id: str,
        proposal_hash: str,
        agent_session_id: str,
        display_turn_id: str,
    ) -> str | None:
        """Resolve a rendered proposal to its checkpoint-bound source turn."""

        self._require_owner_thread()
        with self._lock:
            lifecycle = self._proposal_lifecycles.get(str(proposal_id))
            if lifecycle is None or (
                lifecycle.proposal_hash != str(proposal_hash)
                or lifecycle.session_id != str(agent_session_id)
                or lifecycle.turn_id != str(display_turn_id)
            ):
                return None
            return lifecycle.source_turn_id

    def proposal_lifecycle_matches_from_gui(
        self,
        proposal_id: str,
        proposal_hash: str,
        agent_session_id: str,
        source_turn_id: str,
    ) -> bool:
        """Check the bridge identity before changing workflow stage."""

        self._require_owner_thread()
        with self._lock:
            lifecycle = self._proposal_lifecycles.get(str(proposal_id))
            return bool(
                lifecycle is not None
                and lifecycle.proposal_hash == str(proposal_hash)
                and lifecycle.session_id == str(agent_session_id)
                and lifecycle.source_turn_id == str(source_turn_id)
            )

    def invalidate_authoring_binding_from_gui(self, reason: str) -> None:
        self._require_owner_thread()
        controller = self._authoring_controller
        if controller is not None:
            controller.invalidate_binding(reason)
            # ``invalidate_binding`` has already advanced the controller
            # generation.  Drop the runtime cache without immediately
            # rebuilding it from the now-invalid observed context.
            self._invalidate_authoring_tool_cache_owner_thread(
                invalidate_controller=False,
            )

    def record_authoring_preflight_state_from_gui(
        self,
        state: str,
        message: str = "",
    ) -> None:
        self._require_owner_thread()
        controller = self._authoring_controller
        if controller is None:
            raise RuntimeError("authoring controller is not configured")
        controller.record_preflight_state(state, message)
        self._try_publish_authoring_tool_cache_owner_thread()

    def shutdown(self, *, wait: bool = True) -> None:
        """Cancel and close the engine without an unbounded executor join."""
        with self._lock:
            if self._shutdown:
                return
            self._shutdown = True
            self._cancel_requested = True
            if self._active_turn is not None:
                self._cancel_delta_timer_locked(self._active_turn)
            authoring_invocations = tuple(self._authoring_invocations.values())
            self._authoring_invocations.clear()
            generation = self._generation
            cancel_future = (
                self._control_executor.submit(
                    self._run_cancel,
                    generation,
                )
                if self._busy
                else None
            )
            should_close = self._engine is not None or self._busy
            close_future = (
                self._session_executor.submit(self._run_close)
                if should_close
                else None
            )
        for invocation in authoring_invocations:
            invocation.cancel(
                RuntimeError(
                    "Agent runtime closed during an authoring tool call"
                )
            )
        close_failed = False
        shutdown_timed_out = False
        futures = tuple(
            future
            for future in (cancel_future, close_future)
            if future is not None
        )
        if wait and futures:
            completed, pending = wait_for_futures(
                futures,
                timeout=_RUNTIME_SHUTDOWN_TIMEOUT_SECONDS,
            )
            shutdown_timed_out = bool(pending)
            if close_future in completed:
                try:
                    close_future.result()
                except Exception:
                    close_failed = True
        join_executors = wait and not shutdown_timed_out
        try:
            self._session_executor.shutdown(wait=join_executors)
        finally:
            self._control_executor.shutdown(wait=join_executors)
        if close_failed:
            self.operationRejected.emit("Agent 后台关闭时发生错误")
        elif shutdown_timed_out:
            self.operationRejected.emit("Agent 后台关闭超时")
        self.shutdownFinished.emit()

    def _provider_safe_text(
        self,
        text: str,
        references: Sequence[WorkspaceFileReference],
        *,
        workspace_root: str | os.PathLike[str] | None,
    ) -> str:
        safe_text = text
        safe_text = self._redact_known_path(
            safe_text,
            str(self.agent_data_root),
        )
        if workspace_root is not None:
            safe_text = self._redact_known_path(
                safe_text,
                os.fspath(workspace_root),
            )
        for reference in references:
            if not isinstance(reference, WorkspaceFileReference):
                continue
            safe_text = self._redact_known_path(
                safe_text,
                reference.workspace_root,
            )
        return redact_absolute_paths(safe_text)

    @staticmethod
    def _redact_known_path(text: str, path: str) -> str:
        needle = str(path).replace("\\", "/").casefold()
        if not needle:
            return text
        redacted = text
        while True:
            normalized = redacted.replace("\\", "/").casefold()
            index = normalized.find(needle)
            if index < 0:
                return redacted
            redacted = (
                redacted[:index]
                + "<本地路径已隐藏>"
                + redacted[index + len(needle) :]
            )

    def _ensure_engine(self) -> AgentEnginePort:
        with self._lock:
            existing = self._engine
        if existing is not None:
            return existing
        candidate = self._provider_factory()
        if not isinstance(candidate, (DeepSeekProvider, FakeProvider)):
            raise AgentRuntimeConfigurationError(
                "GUI 拒绝了未注册的 Provider 类型。"
            )
        engine = self._engine_factory(
            self.agent_data_root,
            candidate,
            self._receive_engine_event,
        )
        with self._lock:
            self._engine = engine
            self._engine_ready.set()
        self.providerReady.emit(
            candidate.provider_name,
            candidate.model_name,
        )
        return engine

    def _dispatch_authoring_tool(
        self,
        name: str,
        arguments: Mapping[str, Any],
        context: ToolExecutionContext,
    ) -> ToolResult:
        controller = self._authoring_controller
        if controller is None:
            raise RuntimeError("authoring controller is not configured")
        if threading.get_ident() == self._owner_thread_id:
            with self._lock:
                if self._shutdown:
                    raise RuntimeError("Agent runtime is closed")
            result = controller.dispatch(name, arguments, context)
            try:
                self._publish_authoring_tool_cache_owner_thread()
            except Exception:
                if self.authoring_turn_snapshot.available:
                    self._invalidate_authoring_tool_cache_owner_thread()
            return result
        invocation = _AuthoringToolInvocation(
            name,
            dict(arguments),
            context,
        )
        with self._lock:
            if self._shutdown:
                raise RuntimeError("Agent runtime is closed")
            self._authoring_invocations[id(invocation)] = invocation
        self.authoringToolRequested.emit(invocation)
        deadline = (
            time.monotonic() + _AUTHORING_TOOL_OWNER_TIMEOUT_SECONDS
        )
        while not invocation.completed.wait(timeout=0.05):
            with self._lock:
                if self._shutdown:
                    error = RuntimeError(
                        "Agent runtime closed during an authoring tool call"
                    )
                    invocation.cancel(error)
                    self._authoring_invocations.pop(id(invocation), None)
                    raise error
            if time.monotonic() >= deadline:
                error = TimeoutError(
                    "Agent authoring tool exceeded its owner-thread budget"
                )
                invocation.cancel(error)
                with self._lock:
                    self._authoring_invocations.pop(id(invocation), None)
                raise error
        with self._lock:
            self._authoring_invocations.pop(id(invocation), None)
        if invocation.cancelled:
            if invocation.error is None:
                raise RuntimeError("Agent authoring tool call was cancelled")
            raise invocation.error
        if invocation.error is not None:
            raise RuntimeError(
                "Agent authoring tool failed on the owner thread"
            ) from invocation.error
        if type(invocation.result) is not ToolResult:
            raise TypeError("Agent authoring tool returned an invalid result")
        return invocation.result

    def _execute_authoring_tool(
        self,
        invocation: _AuthoringToolInvocation,
    ) -> None:
        with self._lock:
            shutdown = self._shutdown
        if shutdown:
            invocation.cancel(
                RuntimeError(
                    "Agent runtime closed before an authoring tool call"
                )
            )
            return
        if not invocation.claim():
            return
        try:
            controller = self._authoring_controller
            if controller is None:
                raise RuntimeError("authoring controller is not configured")
            result = controller.dispatch(
                invocation.name,
                invocation.arguments,
                invocation.context,
            )
        except BaseException as error:
            invocation.finish(error=error)
        else:
            try:
                self._publish_authoring_tool_cache_owner_thread()
            except Exception:
                # The tool result remains authoritative, but a failed
                # projection must never leave a previous document cache live.
                if self.authoring_turn_snapshot.available:
                    self._invalidate_authoring_tool_cache_owner_thread()
            invocation.finish(result=result)
        finally:
            with self._lock:
                self._authoring_invocations.pop(id(invocation), None)

    def _reset_authoring_controller_for_session(
        self,
        _session_id: str,
    ) -> None:
        self._require_owner_thread()
        controller = self._authoring_controller
        if controller is None:
            return
        controller.invalidate_binding("Agent session changed")
        controller.reset_for_binding()
        self._try_publish_authoring_tool_cache_owner_thread()

    def _require_owner_thread(self) -> None:
        if threading.get_ident() != self._owner_thread_id:
            raise RuntimeError(
                "authoring GUI state must be changed on the runtime owner thread"
            )

    def _run_send(
        self,
        generation: int,
        text: str,
        references: tuple[WorkspaceFileReference, ...],
    ) -> None:
        context: _TurnContext | None = None
        engine: AgentEnginePort | None = None
        try:
            prepared = prepare_workspace_context(references)
            if self._cancellation_requested(generation):
                with self._lock:
                    existing = self._engine
                session_id = (
                    existing.session_id
                    if existing is not None
                    else f"runtime-session-{generation:08d}"
                )
                context, reset_session = self._start_turn(
                    generation,
                    session_id,
                )
                if reset_session:
                    self.sessionReset.emit(session_id)
                self.agentEventReady.emit(
                    self._new_event(
                        context,
                        EventType.TURN_STARTED,
                        {"user_message": text},
                    )
                )
                self._cancel_turn(context)
                return
            engine = self._ensure_engine()
            context, reset_session = self._start_turn(
                generation,
                engine.session_id,
            )
            if reset_session:
                self.sessionReset.emit(engine.session_id)
            self.agentEventReady.emit(
                self._new_event(
                    context,
                    EventType.TURN_STARTED,
                    {"user_message": text},
                )
            )
            self._attach_input_if_needed(
                engine,
                prepared,
            )
            if self._cancellation_requested(generation):
                self._cancel_turn(context)
                return
            engine.reset_operation_start_signal()
            engine.send_message(
                text,
                request_context=prepared.request_context,
            )
            self._finish_success(context)
        except Exception as error:
            if context is None:
                context, reset_session = self._start_fallback_turn(
                    generation,
                )
                if reset_session:
                    self.sessionReset.emit(context.session_id)
                self.agentEventReady.emit(
                    self._new_event(
                        context,
                        EventType.TURN_STARTED,
                        {"user_message": text},
                    )
                )
            if isinstance(error, WorkspaceContextError):
                self._fail_turn(
                    context,
                    title="工作区文件不可用",
                    message=str(error),
                    code="GUI-WORKSPACE-CONTEXT",
                )
            elif isinstance(error, AgentRuntimeConfigurationError):
                self._fail_turn(
                    context,
                    title="Agent 配置不可用",
                    message=str(error),
                    code="GUI-AGENT-CONFIG",
                )
            else:
                self._fail_turn(
                    context,
                    title="Agent 后台错误",
                    message="Agent 后台操作失败。",
                    code="GUI-RUNTIME-ERROR",
                )
        finally:
            self._flush_engine_round_audit(engine)
            self._finish_operation(generation)

    @staticmethod
    def _flush_engine_round_audit(engine: AgentEnginePort | None) -> None:
        if engine is None:
            return
        flush = getattr(engine, "flush_round_audit", None)
        if not callable(flush):
            return
        try:
            flush()
        except Exception:
            # The terminal UI event has already been emitted.  Keep the GUI
            # lifecycle responsive; close_session() retries the deferred batch
            # before releasing the engine when the runtime shuts down.
            return

    def _run_confirm(
        self,
        generation: int,
        revision: int,
        revision_hash: str,
    ) -> None:
        context: _TurnContext | None = None
        success = False
        try:
            with self._lock:
                engine = self._engine
            if engine is None:
                raise RuntimeError("Agent engine is unavailable")
            snapshot = engine.get_snapshot()
            if (
                snapshot.revision != revision
                or snapshot.revision_hash != revision_hash
            ):
                raise AgentRuntimeConfigurationError(
                    "模型状态已经变化，请重新检查分析摘要。"
                )
            context, reset_session = self._start_turn(
                generation,
                engine.session_id,
                operation="solve",
            )
            if reset_session:
                self.sessionReset.emit(engine.session_id)
            self.agentEventReady.emit(
                self._new_event(
                    context,
                    EventType.TURN_STARTED,
                    {"user_message": f"确认开始求解 revision {revision}"},
                )
            )
            call_id = f"{context.turn_id}-solve"
            context.solve_call_id = call_id
            context.tools[call_id] = _ToolActivity(
                "solve_confirmed_analysis",
                time.monotonic(),
            )
            self._emit_events(
                (
                    self._new_event(
                        context,
                        EventType.TOOL_REQUESTED,
                        {
                            "call_id": call_id,
                            "tool_name": "solve_confirmed_analysis",
                            "display_name": "开始有限元求解",
                            "request": (
                                f"revision={revision}, "
                                f"sha256={revision_hash[:16]}…"
                            ),
                        },
                    ),
                    self._new_event(
                        context,
                        EventType.TOOL_STARTED,
                        {"call_id": call_id},
                    ),
                )
            )
            engine.reset_operation_start_signal()
            if self._cancellation_requested(generation):
                self._cancel_turn(context)
                return
            engine.confirm_revision()
            success = context.solve_succeeded
            self._finish_success(context)
        except Exception as error:
            if context is None:
                session_id = (
                    self.session_id
                    or f"runtime-session-{generation:08d}"
                )
                context, reset_session = self._start_turn(
                    generation,
                    session_id,
                    operation="solve",
                )
                if reset_session:
                    self.sessionReset.emit(context.session_id)
                self.agentEventReady.emit(
                    self._new_event(
                        context,
                        EventType.TURN_STARTED,
                        {
                            "user_message": (
                                f"确认开始求解 revision {revision}"
                            )
                        },
                    )
                )
            message = (
                str(error)
                if isinstance(error, AgentRuntimeConfigurationError)
                else "求解确认未能完成。"
            )
            self._fail_turn(
                context,
                title="无法开始求解",
                message=message,
                code="GUI-SOLVE-CONFIRMATION",
            )
        finally:
            self.solveFinished.emit(
                revision,
                revision_hash,
                success,
            )
            self._finish_operation(generation)

    def _attach_input_if_needed(
        self,
        engine: AgentEnginePort,
        prepared: PreparedWorkspaceContext,
    ) -> None:
        if prepared.input_source is None:
            return
        reference = next(
            (
                item
                for item in prepared.references
                if item.file_type.casefold() == "inp"
            ),
            None,
        )
        if reference is None:
            raise WorkspaceContextError("Abaqus 输入引用缺少本地元数据")
        key = (
            reference.workspace_id,
            reference.relative_path,
            reference.metadata_version,
        )
        with self._lock:
            if self._attached_input_key == key:
                return
            replacing = self._attached_input_key is not None
        encoding = prepared.input_encoding
        try:
            artifact = ArtifactStore(self.agent_data_root).copy_input(
                engine.session_id,
                prepared.input_source,
                source_encoding=(
                    None if encoding == "utf-8" else encoding
                ),
            )
        except InputRejectedError as error:
            raise WorkspaceContextError(
                "Abaqus 输入文件无法安全复制，请重新选择"
            ) from error
        engine.attach_artifact(
            artifact.artifact_id,
            replace_existing=replacing,
        )
        with self._lock:
            self._attached_input_key = key

    def _run_new_session(self, generation: int) -> None:
        try:
            with self._lock:
                engine = self._engine
            if engine is None:
                engine = self._ensure_engine()
            else:
                engine.create_session()
            if generation != self._current_generation():
                return
            self._reset_gui_session(engine.session_id)
            self.sessionReset.emit(engine.session_id)
        except Exception:
            self.operationRejected.emit("无法创建新的 Agent 会话")
        finally:
            self._finish_operation(generation)

    def _queue_proposal_continuation(
        self,
        continuation: _ProposalContinuation,
    ) -> None:
        emit_busy = False
        discard = False
        with self._lock:
            continuation_entry = getattr(
                self._engine,
                "continue_after_proposal",
                None,
            )
            if not callable(continuation_entry):
                discard = True
            elif self._shutdown or continuation.session_id != self._gui_session_id:
                discard = True
            elif self._pending_continuation is not None:
                discard = True
            elif self._busy:
                self._pending_continuation = continuation
            else:
                self._busy = True
                self._cancel_requested = False
                self._generation += 1
                generation = self._generation
                self._session_executor.submit(
                    self._run_continuation,
                    generation,
                    continuation,
                )
                emit_busy = True
        if discard:
            self._discard_proposal_continuation(continuation.proposal_id)
        elif emit_busy:
            self.busyChanged.emit(True)

    def _discard_proposal_continuation(self, proposal_id: str) -> None:
        with self._lock:
            engine = self._engine
        discard = getattr(engine, "discard_continuation", None)
        if callable(discard):
            discard(proposal_id)

    def _run_continuation(
        self,
        generation: int,
        continuation: _ProposalContinuation,
    ) -> None:
        context: _TurnContext | None = None
        engine: AgentEnginePort | None = None
        try:
            with self._lock:
                engine = self._engine
                stopped = self._shutdown or generation != self._generation
            if stopped or engine is None:
                self._discard_proposal_continuation(
                    continuation.proposal_id
                )
                return
            if engine.session_id != continuation.session_id:
                self._discard_proposal_continuation(
                    continuation.proposal_id
                )
                return
            context, reset_session = self._start_turn(
                generation,
                continuation.session_id,
                operation="continuation",
            )
            if reset_session:
                self.sessionReset.emit(continuation.session_id)
            self.agentEventReady.emit(
                self._new_event(
                    context,
                    EventType.CONTINUATION_STARTED,
                    {
                        "proposal_id": continuation.proposal_id,
                        "proposal_hash": continuation.proposal_hash,
                        "source_turn_id": continuation.source_turn_id,
                        "status": continuation.status,
                    },
                )
            )
            continuation_entry = getattr(
                engine,
                "continue_after_proposal",
                None,
            )
            if not callable(continuation_entry):
                return
            engine.reset_operation_start_signal()
            continuation_entry(
                continuation.proposal_id,
                continuation.proposal_hash,
                continuation.source_turn_id,
                continuation.model_revision,
                continuation.status,
                continuation.summary,
            )
            self._finish_success(context)
        except Exception:
            if context is not None:
                self._fail_turn(
                    context,
                    title="Agent 续跑失败",
                    message="Agent 未能处理本地任务终态。",
                    code="GUI-CONTINUATION-ERROR",
                )
        finally:
            self._flush_engine_round_audit(engine)
            self._finish_operation(generation)

    def _run_cancel(self, generation: int) -> None:
        self._engine_ready.wait(timeout=5.0)
        with self._lock:
            if generation != self._generation:
                return
            engine = self._engine
        if engine is None:
            return
        engine.wait_for_operation_start(timeout_seconds=5.0)
        with self._lock:
            if generation != self._generation or not self._busy:
                return
        engine.cancel_active_operation()

    def _cancellation_requested(self, generation: int) -> bool:
        with self._lock:
            return (
                generation == self._generation
                and self._cancel_requested
            )

    def _cancel_turn(self, context: _TurnContext) -> None:
        emitted: list[AgentEvent] = []
        with self._lock:
            if self._active_turn is not context or context.terminal:
                return
            emitted.extend(self._flush_pending_delta_locked(context))
            emitted.append(
                self._new_event_locked(
                    context,
                    EventType.TURN_CANCELLED,
                    {"reason": "用户已取消本轮操作。"},
                )
            )
            context.terminal = True
        self._emit_events(emitted)

    def _run_close(self) -> None:
        with self._lock:
            engine = self._engine
        try:
            if engine is not None:
                engine.close_session()
        finally:
            with self._lock:
                if self._active_turn is not None:
                    self._cancel_delta_timer_locked(self._active_turn)
                self._engine = None
                self._engine_ready.clear()
                self._active_turn = None
                self._busy = False
                self._attached_input_key = None

    def _start_turn(
        self,
        generation: int,
        session_id: str,
        *,
        operation: str = "message",
    ) -> tuple[_TurnContext, bool]:
        with self._lock:
            reset_session = self._gui_session_id != session_id
            if reset_session:
                self._gui_session_id = session_id
                self._sequence = 0
                self._turn_counter = 0
                self._proposal_lifecycles.clear()
            self._turn_counter += 1
            turn_id = f"turn-{self._turn_counter:08d}"
            context = _TurnContext(
                generation=generation,
                session_id=session_id,
                turn_id=turn_id,
                operation=operation,
            )
            self._active_turn = context
            return context, reset_session

    def _start_fallback_turn(
        self,
        generation: int,
    ) -> tuple[_TurnContext, bool]:
        session_id = f"runtime-session-{generation:08d}"
        return self._start_turn(generation, session_id)

    def _reset_gui_session(self, session_id: str) -> None:
        with self._lock:
            if self._active_turn is not None:
                self._cancel_delta_timer_locked(self._active_turn)
            self._gui_session_id = session_id
            self._sequence = 0
            self._turn_counter = 0
            self._active_turn = None
            self._attached_input_key = None
            self._proposal_lifecycles.clear()

    def _receive_engine_event(self, event: EngineEvent) -> None:
        if event.event is EngineEventType.STATE_CHANGED:
            return
        emitted: list[AgentEvent] = []
        rejection: str | None = None
        with self._lock:
            context = self._active_turn
            if (
                context is None
                and self._shutdown
                and event.event is EngineEventType.OPERATION_CANCELLED
            ):
                return
            if context is None:
                rejection = "已拒绝无活动 turn 的 EngineEvent"
            elif (
                context.terminal and event.event is EngineEventType.OPERATION_CANCELLED
            ):
                return
            elif context.terminal:
                rejection = "已拒绝 turn 结束后的晚到 EngineEvent"
            elif event.session_id != context.session_id:
                rejection = "已拒绝跨 session 的 EngineEvent"
            elif context.seen_engine_events.get(id(event)) is event:
                rejection = "已拒绝重复 EngineEvent"
            else:
                context.seen_engine_events[id(event)] = event
                if event.event is not EngineEventType.MESSAGE_DELTA:
                    emitted.extend(
                        self._flush_pending_delta_locked(context)
                    )
                emitted.extend(self._map_engine_event_locked(context, event))
            if rejection is None:
                for mapped in emitted:
                    self.agentEventReady.emit(mapped)
        if rejection is not None:
            self.eventRejected.emit(rejection)
            return

    def _schedule_delta_flush_locked(self, context: _TurnContext) -> None:
        if context.delta_timer is not None:
            return
        timer: threading.Timer
        timer = threading.Timer(
            _MESSAGE_DELTA_FRAME_SECONDS,
            lambda: self._flush_delta_timer(context, timer),
        )
        timer.daemon = True
        context.delta_timer = timer
        timer.start()

    def _cancel_delta_timer_locked(self, context: _TurnContext) -> None:
        timer = context.delta_timer
        context.delta_timer = None
        if timer is not None:
            timer.cancel()

    def _flush_delta_timer(
        self,
        context: _TurnContext,
        timer: threading.Timer,
    ) -> None:
        with self._lock:
            if context.delta_timer is not timer:
                return
            context.delta_timer = None
            if (
                self._active_turn is context
                and not context.terminal
                and context.pending_delta_chunks
            ):
                event = self._flush_one_pending_delta_locked(
                    context,
                    adaptive=True,
                )
                if context.pending_delta_chunks:
                    self._schedule_delta_flush_locked(context)
                if event is not None:
                    self.agentEventReady.emit(event)

    def _flush_one_pending_delta_locked(
        self,
        context: _TurnContext,
        *,
        adaptive: bool = False,
    ) -> AgentEvent | None:
        if not context.pending_delta_chunks:
            return None
        now = time.monotonic()
        wait_seconds = 0.0
        if context.pending_delta_started_at is not None:
            wait_seconds = now - context.pending_delta_started_at
            context.max_pending_delta_wait_seconds = max(
                context.max_pending_delta_wait_seconds,
                wait_seconds,
            )
        batch_characters = _MAX_MESSAGE_DELTA_CHARACTERS
        if adaptive:
            backlog_batches = max(
                1,
                (
                    context.pending_delta_characters
                    + _MAX_MESSAGE_DELTA_CHARACTERS
                    - 1
                )
                // _MAX_MESSAGE_DELTA_CHARACTERS,
            )
            waited_batches = max(
                1,
                int(
                    wait_seconds
                    / _MESSAGE_DELTA_FRAME_SECONDS
                ),
            )
            batch_characters = min(
                _MAX_ADAPTIVE_MESSAGE_DELTA_CHARACTERS,
                _MAX_MESSAGE_DELTA_CHARACTERS
                * max(backlog_batches, waited_batches),
            )
        text = "".join(context.pending_delta_chunks)
        delta = text[:batch_characters]
        remainder = text[batch_characters:]
        context.pending_delta_chunks = [remainder] if remainder else []
        context.pending_delta_characters = len(remainder)
        if not remainder:
            context.pending_delta_started_at = None
        if not delta or context.active_message_id is None:
            return None
        return self._new_event_locked(
            context,
            EventType.MESSAGE_DELTA,
            {
                "message_id": context.active_message_id,
                "delta": delta,
            },
        )

    def _flush_pending_delta_locked(
        self,
        context: _TurnContext,
    ) -> list[AgentEvent]:
        self._cancel_delta_timer_locked(context)
        events: list[AgentEvent] = []
        while context.pending_delta_chunks:
            event = self._flush_one_pending_delta_locked(context)
            if event is not None:
                events.append(event)
        return events

    def _complete_active_message_locked(
        self,
        context: _TurnContext,
    ) -> list[AgentEvent]:
        message_id = context.active_message_id
        if message_id is None:
            return []
        context.active_message_id = None
        return [
            self._new_event_locked(
                context,
                EventType.MESSAGE_COMPLETE,
                {"message_id": message_id},
            )
        ]

    def _stream_backlog_metrics_locked(
        self,
        context: _TurnContext,
    ) -> dict[str, int | float]:
        max_wait = context.max_pending_delta_wait_seconds
        if context.pending_delta_started_at is not None:
            max_wait = max(
                max_wait,
                time.monotonic() - context.pending_delta_started_at,
            )
        return {
            "pending_characters": context.pending_delta_characters,
            "max_pending_characters": context.max_pending_delta_characters,
            "max_wait_seconds": max_wait,
        }

    def _map_engine_event_locked(
        self,
        context: _TurnContext,
        event: EngineEvent,
    ) -> list[AgentEvent]:
        if event.event is not EngineEventType.DIAGNOSTIC:
            context.embedded_tool_diagnostics.clear()
        if event.event is EngineEventType.MESSAGE_STARTED:
            events = self._complete_active_message_locked(context)
            # Provider tool loops often emit an empty assistant envelope before
            # tool calls.  Wait for text before creating a visible message.
            return events
        if event.event is EngineEventType.MESSAGE_DELTA:
            events: list[AgentEvent] = []
            if context.active_message_id is None:
                context.message_counter += 1
                context.active_message_id = (
                    f"{context.turn_id}-message-"
                    f"{context.message_counter:04d}"
                )
                events.append(
                    self._new_event_locked(
                        context,
                        EventType.MESSAGE_START,
                        {
                            "message_id": context.active_message_id,
                            "role": "assistant",
                            "format": "restricted_markdown",
                        },
                    )
                )
            text = event.data.get("text")
            if isinstance(text, str) and text:
                if context.pending_delta_characters == 0:
                    context.pending_delta_started_at = time.monotonic()
                context.pending_delta_chunks.append(text)
                context.pending_delta_characters += len(text)
                context.max_pending_delta_characters = max(
                    context.max_pending_delta_characters,
                    context.pending_delta_characters,
                )
                self._schedule_delta_flush_locked(context)
            return events
        if event.event is EngineEventType.DIAGNOSTIC:
            raw_diagnostic = event.data.get("diagnostic")
            identity = _diagnostic_identity(raw_diagnostic)
            if identity in context.embedded_tool_diagnostics:
                context.embedded_tool_diagnostics.remove(identity)
                return []
            context.embedded_tool_diagnostics.clear()
            return [
                self._diagnostic_event_locked(
                    context,
                    raw_diagnostic,
                )
            ]
        if event.event is EngineEventType.ERROR:
            diagnostic = event.data.get("diagnostic")
            if isinstance(diagnostic, Mapping):
                message = diagnostic.get("message")
                if isinstance(message, str) and message.strip():
                    context.failure_reason = self._safe_message(message)
            context.failure_reason = context.failure_reason or "Agent 后台操作失败。"
            return []
        if event.event is EngineEventType.OPERATION_CANCELLED:
            context.terminal = True
            return [
                self._new_event_locked(
                    context,
                    EventType.TURN_CANCELLED,
                    {"reason": "用户已取消本轮操作。"},
                )
            ]
        if event.event is EngineEventType.TOOL_STARTED:
            return self._tool_started_events_locked(context, event)
        if event.event is EngineEventType.TOOL_COMPLETED:
            return self._tool_completed_events_locked(context, event)
        if event.event is EngineEventType.ANALYSIS_SUMMARY:
            return self._analysis_summary_events_locked(context, event)
        if event.event is EngineEventType.CONFIRMATION_REQUIRED:
            context.failure_reason = "当前分析状态尚不能开始求解。"
            return [
                self._diagnostic_event_locked(
                    context,
                    {
                        "code": "CONFIRMATION-REQUIRED",
                        "severity": "warning",
                        "message": context.failure_reason,
                    },
                )
            ]
        if event.event is EngineEventType.RUN_PROGRESS:
            return []
        if event.event is EngineEventType.RUN_COMPLETED:
            return self._run_completed_events_locked(context, event)
        return []

    def _tool_started_events_locked(
        self,
        context: _TurnContext,
        event: EngineEvent,
    ) -> list[AgentEvent]:
        call_id = event.data.get("call_id")
        tool_name = event.data.get("tool")
        if not isinstance(call_id, str) or not isinstance(tool_name, str):
            context.failure_reason = "Agent 工具事件缺少标识信息。"
            return []
        if call_id in context.tools:
            self.eventRejected.emit("已拒绝重复的工具调用标识")
            return []
        events = self._complete_active_message_locked(context)
        context.tools[call_id] = _ToolActivity(
            tool_name,
            time.monotonic(),
        )
        arguments = event.data.get("arguments")
        return [
            *events,
            self._new_event_locked(
                context,
                EventType.TOOL_REQUESTED,
                {
                    "call_id": call_id,
                    "tool_name": tool_name,
                    "display_name": _TOOL_DISPLAY_NAMES.get(
                        tool_name,
                        tool_name,
                    ),
                    "request": safe_tool_summary(arguments),
                },
            ),
            self._new_event_locked(
                context,
                EventType.TOOL_STARTED,
                {"call_id": call_id},
            ),
        ]

    def _tool_completed_events_locked(
        self,
        context: _TurnContext,
        event: EngineEvent,
    ) -> list[AgentEvent]:
        call_id = event.data.get("call_id")
        if not isinstance(call_id, str):
            context.failure_reason = "Agent 工具返回缺少调用标识。"
            return []
        activity = context.tools.get(call_id)
        if activity is None:
            tool_name = event.data.get("tool")
            if not isinstance(tool_name, str):
                context.failure_reason = "Agent 工具返回缺少工具名称。"
                return []
            started = EngineEvent(
                EngineEventType.TOOL_STARTED,
                event.session_id,
                {
                    "call_id": call_id,
                    "tool": tool_name,
                    "arguments": {},
                },
                event.timestamp,
            )
            events = self._tool_started_events_locked(context, started)
            activity = context.tools[call_id]
        else:
            events = []
        if activity.terminal:
            self.eventRejected.emit("已拒绝重复的工具完成事件")
            return events
        activity.terminal = True
        duration_ms = max(
            0.0,
            (time.monotonic() - activity.started_at) * 1_000,
        )
        result = event.data.get("result")
        result_mapping = result if isinstance(result, Mapping) else {}
        summary = safe_tool_summary(
            result_mapping.get("summary", "工具调用已完成"),
        )
        diagnostics = result_mapping.get("diagnostics")
        diagnostic_items = (
            diagnostics
            if isinstance(diagnostics, (list, tuple))
            else ()
        )
        context.embedded_tool_diagnostics = [
            identity
            for item in diagnostic_items
            if (identity := _diagnostic_identity(item)) is not None
        ]
        messages = [
            safe_tool_summary(item.get("message"))
            for item in diagnostic_items
            if isinstance(item, Mapping)
            and isinstance(item.get("message"), str)
        ]
        severities = {
            str(item.get("severity", "")).casefold()
            for item in diagnostic_items
            if isinstance(item, Mapping)
        }
        if result_mapping.get("ok") is not True:
            return [
                *events,
                self._new_event_locked(
                    context,
                    EventType.TOOL_FAILED,
                    {
                        "call_id": call_id,
                        "error": summary,
                        "diagnostic": (
                            "；".join(messages)
                            if messages
                            else summary
                        ),
                        "duration_ms": duration_ms,
                    },
                ),
            ]
        if severities & {"warning", "error", "blocking"}:
            return [
                *events,
                self._new_event_locked(
                    context,
                    EventType.TOOL_WARNING,
                    {
                        "call_id": call_id,
                        "warning": (
                            "；".join(messages)
                            if messages
                            else summary
                        ),
                        "result": summary,
                        "duration_ms": duration_ms,
                    },
                ),
            ]
        completed_events = [
            *events,
            self._new_event_locked(
                context,
                EventType.TOOL_RESULT,
                {
                    "call_id": call_id,
                    "result": summary,
                    "duration_ms": duration_ms,
                },
            ),
        ]
        data = result_mapping.get("data")
        proposal_view = (
            data.get("proposal_view")
            if isinstance(data, Mapping)
            else None
        )
        if isinstance(proposal_view, Mapping):
            proposal_id = str(proposal_view.get("proposal_id", ""))
            proposal_hash = str(proposal_view.get("proposal_hash", ""))
            proposal_kind = str(proposal_view.get("proposal_kind", ""))
            checkpoint = (
                data.get("continuation_checkpoint")
                if isinstance(data, Mapping)
                else None
            )
            source_turn_id = context.turn_id
            model_revision = int(
                proposal_view.get("base_session_revision", 0)
            )
            if isinstance(checkpoint, Mapping):
                source_turn_id = str(
                    checkpoint.get("source_turn_id", source_turn_id)
                )
                raw_revision = checkpoint.get(
                    "model_revision",
                    model_revision,
                )
                if type(raw_revision) is int:
                    model_revision = raw_revision
            self._proposal_lifecycles[proposal_id] = _ProposalLifecycle(
                session_id=context.session_id,
                turn_id=context.turn_id,
                proposal_id=proposal_id,
                proposal_hash=proposal_hash,
                proposal_kind=proposal_kind,
                source_turn_id=source_turn_id,
                model_revision=model_revision,
            )
            completed_events.append(
                self._new_event_locked(
                    context,
                    EventType.PROPOSAL_REQUESTED,
                    dict(proposal_view),
                )
            )
        return completed_events

    def _analysis_summary_events_locked(
        self,
        context: _TurnContext,
        event: EngineEvent,
    ) -> list[AgentEvent]:
        summary = event.data.get("analysis_summary")
        if not isinstance(summary, Mapping):
            context.failure_reason = "分析摘要格式无效。"
            return []
        diagnostics = summary.get("diagnostics")
        if isinstance(diagnostics, (list, tuple)) and any(
            isinstance(item, Mapping)
            and str(item.get("severity", "")).casefold()
            in {"error", "blocking"}
            for item in diagnostics
        ):
            return []
        revision = summary.get("revision")
        revision_hash = summary.get("revision_hash")
        if (
            isinstance(revision, bool)
            or not isinstance(revision, int)
            or revision < 1
            or not isinstance(revision_hash, str)
            or len(revision_hash) != 64
        ):
            context.failure_reason = "分析摘要缺少有效 revision。"
            return []
        model_name = safe_tool_summary(
            summary.get("model_name", "模型"),
            max_characters=100,
        )
        step = summary.get("analysis_step")
        step_name = (
            safe_tool_summary(
                step.get("name", "未指定"),
                max_characters=80,
            )
            if isinstance(step, Mapping)
            else "未指定"
        )
        confirmation_id = (
            f"{context.turn_id}-confirmation-{revision}"
        )
        context.pending_confirmation = {
            "confirmation_id": confirmation_id,
            "title": "确认开始有限元求解",
            "summary": (
                f"{model_name} · {summary.get('node_count', 0)} 节点 · "
                f"{summary.get('element_count', 0)} 单元 · "
                f"分析步 {step_name}"
            ),
            "revision": revision,
            "revision_hash": revision_hash,
        }
        return []

    def _run_completed_events_locked(
        self,
        context: _TurnContext,
        event: EngineEvent,
    ) -> list[AgentEvent]:
        call_id = context.solve_call_id
        if call_id is None:
            context.failure_reason = "收到未关联确认操作的求解结果。"
            return []
        activity = context.tools.get(call_id)
        if activity is None:
            context.failure_reason = "求解操作缺少本地活动记录。"
            return []
        status = str(event.data.get("status", "")).casefold()
        duration_ms = max(
            0.0,
            (time.monotonic() - activity.started_at) * 1_000,
        )
        if status == "cancelled":
            context.terminal = True
            return [
                self._new_event_locked(
                    context,
                    EventType.TURN_CANCELLED,
                    {"reason": "用户已取消本轮求解。"},
                )
            ]
        activity.terminal = True
        if status != "succeeded":
            context.failure_reason = "有限元求解未成功完成。"
            return [
                self._new_event_locked(
                    context,
                    EventType.TOOL_FAILED,
                    {
                        "call_id": call_id,
                        "error": context.failure_reason,
                        "duration_ms": duration_ms,
                    },
                )
            ]
        context.solve_succeeded = True
        run_id = safe_tool_summary(
            event.data.get("run_id", "run"),
            max_characters=100,
        )
        artifacts = event.data.get("artifacts")
        artifact_count = (
            len(artifacts)
            if isinstance(artifacts, (list, tuple))
            else 0
        )
        return [
            self._new_event_locked(
                context,
                EventType.TOOL_RESULT,
                {
                    "call_id": call_id,
                    "result": (
                        f"求解完成 · run {run_id} · "
                        f"{artifact_count} 个 Agent 工件"
                    ),
                    "duration_ms": duration_ms,
                },
            )
        ]

    def _finish_success(self, context: _TurnContext) -> None:
        emitted: list[AgentEvent] = []
        with self._lock:
            if self._active_turn is not context or context.terminal:
                return
            emitted.extend(self._flush_pending_delta_locked(context))
            if context.failure_reason is not None:
                emitted.append(
                    self._new_event_locked(
                        context,
                        EventType.TURN_FAILED,
                        {"reason": context.failure_reason},
                    )
                )
            else:
                emitted.extend(
                    self._complete_active_message_locked(context)
                )
                if context.pending_confirmation is not None:
                    emitted.append(
                        self._new_event_locked(
                            context,
                            EventType.CONFIRMATION_REQUESTED,
                            context.pending_confirmation,
                        )
                    )
                emitted.append(
                    self._new_event_locked(
                        context,
                        EventType.TURN_COMPLETE,
                        {},
                    )
                )
            context.terminal = True
        for event in emitted:
            self.agentEventReady.emit(event)

    def _fail_turn(
        self,
        context: _TurnContext,
        *,
        title: str,
        message: str,
        code: str,
    ) -> None:
        emitted: list[AgentEvent] = []
        with self._lock:
            if self._active_turn is not context or context.terminal:
                return
            emitted.extend(self._flush_pending_delta_locked(context))
            emitted.append(
                self._diagnostic_event_locked(
                    context,
                    {
                        "code": code,
                        "severity": "error",
                        "message": message,
                        "title": title,
                    },
                )
            )
            emitted.append(
                self._new_event_locked(
                    context,
                    EventType.TURN_FAILED,
                    {"reason": message},
                )
            )
            context.terminal = True
        for event in emitted:
            self.agentEventReady.emit(event)

    def _diagnostic_event_locked(
        self,
        context: _TurnContext,
        raw_diagnostic: object,
    ) -> AgentEvent:
        diagnostic = raw_diagnostic if isinstance(raw_diagnostic, Mapping) else {}
        context.diagnostic_count += 1
        raw_severity = diagnostic.get("severity", "error")
        severity = (
            raw_severity
            if raw_severity in {"info", "warning", "error", "blocking"}
            else "error"
        )
        raw_code = diagnostic.get("code", "AGENT-DIAGNOSTIC")
        code = str(raw_code)[:100] if raw_code is not None else "AGENT-DIAGNOSTIC"
        raw_title = diagnostic.get("title")
        title = (
            self._safe_message(raw_title)
            if isinstance(raw_title, str) and raw_title.strip()
            else "Agent 诊断"
        )
        raw_message = diagnostic.get("message")
        message = (
            self._safe_message(raw_message)
            if isinstance(raw_message, str) and raw_message.strip()
            else "Agent 后台返回了诊断信息。"
        )
        return self._new_event_locked(
            context,
            EventType.DIAGNOSTIC,
            {
                "diagnostic_id": (
                    f"{context.turn_id}-diagnostic-{context.diagnostic_count:04d}"
                ),
                "title": title,
                "message": message,
                "severity": severity,
                "code": code,
            },
        )

    def _safe_message(self, value: str) -> str:
        return self._redact_known_path(
            str(value),
            str(self.agent_data_root),
        )[:1_000]

    def _new_event(
        self,
        context: _TurnContext,
        event_type: EventType,
        payload: Mapping[str, object],
    ) -> AgentEvent:
        with self._lock:
            return self._new_event_locked(
                context,
                event_type,
                payload,
            )

    def _emit_events(
        self,
        events: Sequence[AgentEvent],
    ) -> None:
        for event in events:
            self.agentEventReady.emit(event)

    def _new_event_locked(
        self,
        context: _TurnContext,
        event_type: EventType,
        payload: Mapping[str, object],
    ) -> AgentEvent:
        sequence = self._sequence + 1
        created = AgentEvent.create(
            event_id=f"event-{sequence:08d}",
            session_id=context.session_id,
            turn_id=context.turn_id,
            sequence=sequence,
            event_type=event_type,
            payload=payload,
        )
        self._sequence = sequence
        return created

    def _finish_operation(self, generation: int) -> None:
        continuation: _ProposalContinuation | None = None
        with self._lock:
            if generation != self._generation:
                return
            if self._active_turn is not None:
                self._last_stream_backlog_metrics = (
                    self._stream_backlog_metrics_locked(self._active_turn)
                )
            self._active_turn = None
            self._cancel_requested = False
            continuation = self._pending_continuation
            self._pending_continuation = None
            if continuation is None or self._shutdown:
                self._busy = False
            else:
                self._generation += 1
                next_generation = self._generation
                self._session_executor.submit(
                    self._run_continuation,
                    next_generation,
                    continuation,
                )
        if continuation is None or self._shutdown:
            self.busyChanged.emit(False)

    def _current_generation(self) -> int:
        with self._lock:
            return self._generation


__all__ = [
    "AgentEnginePort",
    "AgentRuntimeConfigurationError",
    "AuthoringTurnSnapshot",
    "EngineFactory",
    "ProviderFactory",
    "QtAgentRuntime",
]
