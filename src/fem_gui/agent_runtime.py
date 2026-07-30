"""Qt-facing background runtime for the Phase 5 FEM Agent integration.

Engine construction, workspace context preparation, provider calls, tools,
confirmation runs, and shutdown stay outside the Qt main thread.  The GUI
receives validated ``AgentEvent`` objects through queued Qt signals.
"""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from PySide6.QtCore import QObject, Qt, Signal

from fem_agent.artifacts import ArtifactStore, InputRejectedError
from fem_agent.authoring import ProposalState, RequirementReview
from fem_agent.authoring_runtime import AuthoringWorkflowController
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
from fem_agent.providers.base import CloudModelProvider
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

    def get_snapshot(self) -> Any: ...


ProviderFactory = Callable[[], CloudModelProvider]
EngineFactory = Callable[
    [Path, CloudModelProvider, Callable[[EngineEvent], None]],
    AgentEnginePort,
]

_MAX_MESSAGE_DELTA_CHARACTERS = 8_000
_AUTHORING_TOOL_OWNER_TIMEOUT_SECONDS = 30.0

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
    seen_engine_events: list[EngineEvent] = field(default_factory=list)


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
        self._runtime = runtime
        self._controller = controller

    @property
    def definitions(self):
        return self._controller.definitions

    def dispatch(
        self,
        name: str,
        arguments: Mapping[str, Any],
        context: ToolExecutionContext,
    ) -> ToolResult:
        return self._runtime._dispatch_authoring_tool(
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
        self._busy = False
        self._cancel_requested = False
        self._shutdown = False
        self._authoring_invocations: dict[int, _AuthoringToolInvocation] = {}
        self._attached_input_key: tuple[str, str, str] | None = None
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

    def invalidate_authoring_binding_from_gui(self, reason: str) -> None:
        self._require_owner_thread()
        controller = self._authoring_controller
        if controller is not None:
            controller.invalidate_binding(reason)

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

    def shutdown(self, *, wait: bool = True) -> None:
        """Cancel, close the engine off-thread, and join owned executors."""
        with self._lock:
            if self._shutdown:
                return
            self._shutdown = True
            self._cancel_requested = True
            authoring_invocations = tuple(self._authoring_invocations.values())
            self._authoring_invocations.clear()
            generation = self._generation
            if self._busy:
                self._control_executor.submit(
                    self._run_cancel,
                    generation,
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
        if wait and close_future is not None:
            try:
                close_future.result()
            except Exception:
                close_failed = True
        try:
            self._session_executor.shutdown(wait=wait)
        finally:
            self._control_executor.shutdown(wait=wait)
        if close_failed:
            self.operationRejected.emit("Agent 后台关闭时发生错误")
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
            return controller.dispatch(name, arguments, context)
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
            self._finish_operation(generation)

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
        with self._lock:
            if self._active_turn is not context or context.terminal:
                return
            event = self._new_event_locked(
                context,
                EventType.TURN_CANCELLED,
                {"reason": "用户已取消本轮操作。"},
            )
            context.terminal = True
        self.agentEventReady.emit(event)

    def _run_close(self) -> None:
        with self._lock:
            engine = self._engine
        try:
            if engine is not None:
                engine.close_session()
        finally:
            with self._lock:
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
            self._gui_session_id = session_id
            self._sequence = 0
            self._turn_counter = 0
            self._active_turn = None
            self._attached_input_key = None

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
            elif any(
                seen is event
                for seen in context.seen_engine_events
            ):
                rejection = "已拒绝重复 EngineEvent"
            else:
                context.seen_engine_events.append(event)
                emitted.extend(self._map_engine_event_locked(context, event))
            if rejection is None:
                for mapped in emitted:
                    self.agentEventReady.emit(mapped)
        if rejection is not None:
            self.eventRejected.emit(rejection)
            return

    def _map_engine_event_locked(
        self,
        context: _TurnContext,
        event: EngineEvent,
    ) -> list[AgentEvent]:
        if event.event is EngineEventType.MESSAGE_STARTED:
            events: list[AgentEvent] = []
            if context.active_message_id is not None:
                events.append(
                    self._new_event_locked(
                        context,
                        EventType.MESSAGE_COMPLETE,
                        {
                            "message_id": context.active_message_id,
                        },
                    )
                )
                context.active_message_id = None
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
                for start in range(
                    0,
                    len(text),
                    _MAX_MESSAGE_DELTA_CHARACTERS,
                ):
                    events.append(
                        self._new_event_locked(
                        context,
                        EventType.MESSAGE_DELTA,
                        {
                            "message_id": context.active_message_id,
                            "delta": text[
                                start : start
                                + _MAX_MESSAGE_DELTA_CHARACTERS
                                ],
                            },
                        )
                    )
            return events
        if event.event is EngineEventType.DIAGNOSTIC:
            return [
                self._diagnostic_event_locked(
                    context,
                    event.data.get("diagnostic"),
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
        context.tools[call_id] = _ToolActivity(
            tool_name,
            time.monotonic(),
        )
        arguments = event.data.get("arguments")
        return [
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
            if context.failure_reason is not None:
                emitted.append(
                    self._new_event_locked(
                        context,
                        EventType.TURN_FAILED,
                        {"reason": context.failure_reason},
                    )
                )
            else:
                if context.active_message_id is not None:
                    emitted.append(
                        self._new_event_locked(
                            context,
                            EventType.MESSAGE_COMPLETE,
                            {
                                "message_id": (
                                    context.active_message_id
                                )
                            },
                        )
                    )
                    context.active_message_id = None
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
        with self._lock:
            if generation != self._generation:
                return
            self._active_turn = None
            self._busy = False
            self._cancel_requested = False
        self.busyChanged.emit(False)

    def _current_generation(self) -> int:
        with self._lock:
            return self._generation


__all__ = [
    "AgentEnginePort",
    "AgentRuntimeConfigurationError",
    "EngineFactory",
    "ProviderFactory",
    "QtAgentRuntime",
]
