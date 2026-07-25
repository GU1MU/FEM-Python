"""Local terminal adapter for the UI-neutral FEM Agent engine."""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import unicodedata
from pathlib import Path
from typing import Callable, TextIO

from .artifacts import ArtifactStore
from .config import (
    ConfigError,
    LocalAgentConfig,
    ROOT_CONFIG_NAME,
    create_main_config_template,
    find_main_config,
    resolve_local_config,
)
from .engine import AgentSessionEngine, EngineEvent, EngineEventType
from .providers import DeepSeekProvider, FakeProvider, ProviderConfig
from .schemas import AnalysisSummary, ArtifactRecord


InputFunction = Callable[[str], str]
EngineOperation = Callable[[], tuple[EngineEvent, ...]]
_EXIT_COMMANDS = frozenset({"/exit", "/quit", "/q"})


class _OperationRunner:
    """Keep terminal input responsive while one engine operation is active."""

    def __init__(self, engine: AgentSessionEngine, output: TextIO):
        self._engine = engine
        self._output = output
        self._output_lock = threading.RLock()
        self._thread: threading.Thread | None = None
        self._announcement: threading.Event | None = None
        self._cancel_requested: threading.Event | None = None
        self._suppressed_live_events: frozenset[EngineEventType] = frozenset()
        self._agent_label_emitted = False
        self._unsubscribe = engine.subscribe(self._receive_live_event)

    @property
    def active(self) -> bool:
        thread = self._thread
        return thread is not None and thread.is_alive()

    def write(self, value: object) -> None:
        with self._output_lock:
            _write(self._output, value)

    def render(self, events: tuple[EngineEvent, ...]) -> None:
        with self._output_lock:
            self._agent_label_emitted = _render_events(
                events,
                self._output,
                agent_label_emitted=self._agent_label_emitted,
            )

    def _receive_live_event(self, event: EngineEvent) -> None:
        if threading.current_thread() is not self._thread:
            return
        if event.event in self._suppressed_live_events:
            return
        announcement = self._announcement
        if announcement is not None:
            announcement.wait()
        self.render((event,))

    def run_local(self, action: Callable[[], bool]) -> bool:
        with self._output_lock:
            return action()

    def start(
        self,
        label: str,
        operation: EngineOperation,
        *,
        suppress_live_diagnostics: bool = False,
        completion_message: str | None = None,
        completion_issue_message: str | None = None,
    ) -> bool:
        if self.active:
            self.write(
                "An operation is already active. Use /status or /cancel.\n"
            )
            return False

        announced = threading.Event()
        cancel_requested = threading.Event()
        suppressed_live_events = (
            frozenset({EngineEventType.DIAGNOSTIC})
            if suppress_live_diagnostics
            else frozenset()
        )

        def run() -> None:
            try:
                events = operation()
            except KeyboardInterrupt:
                announced.wait()
                self.write("\nCancellation requested.\n")
                self._engine.cancel_active_operation()
            except Exception as error:
                announced.wait()
                self.write(_prefixed_text("Error: ", error))
            else:
                announced.wait()
                cancelled = cancel_requested.is_set() or any(
                    event.event == EngineEventType.OPERATION_CANCELLED
                    for event in events
                )
                if not cancelled:
                    has_issues = any(
                        event.event == EngineEventType.ERROR
                        or (
                            event.event == EngineEventType.DIAGNOSTIC
                            and event.data.get("diagnostic", {}).get(
                                "severity"
                            )
                            == "error"
                        )
                        for event in events
                    )
                    message = (
                        completion_issue_message
                        if has_issues
                        else completion_message
                    )
                    if message is not None:
                        self.write(message)
            finally:
                if threading.current_thread() is self._thread:
                    self._suppressed_live_events = frozenset()

        thread = threading.Thread(
            target=run,
            name=f"fem-agent-{label.replace(' ', '-')}",
            daemon=True,
        )
        self._thread = thread
        self._announcement = announced
        self._cancel_requested = cancel_requested
        self._suppressed_live_events = suppressed_live_events
        self._agent_label_emitted = False
        try:
            self._engine.reset_operation_start_signal()
            thread.start()
            self._engine.wait_for_operation_start()
            if suppress_live_diagnostics:
                self.write(
                    f"[operation] {label} started; use /summary for inspection "
                    "details or /cancel to stop.\n"
                )
            else:
                self.write(
                    f"[operation] {label} started; /cancel remains available.\n"
                )
        finally:
            # A Ctrl+C can arrive after the worker starts and before the
            # operation banner is written. Never leave that worker (or a live
            # event callback) waiting forever for the banner gate.
            announced.set()
        return True

    def cancel(self) -> None:
        cancel_requested = self._cancel_requested
        if self.active and cancel_requested is not None:
            cancel_requested.set()
        self.render(self._engine.cancel_active_operation())

    def stop_for_shutdown(
        self,
        *,
        grace_seconds: float = 0.2,
        cancel_timeout_seconds: float = 10.0,
    ) -> bool:
        thread = self._thread
        if thread is None or not thread.is_alive():
            return True
        thread.join(grace_seconds)
        if not thread.is_alive():
            return True
        self.cancel()
        thread.join(cancel_timeout_seconds)
        return not thread.is_alive()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fem-agent")
    subparsers = parser.add_subparsers(dest="command", required=True)
    chat = subparsers.add_parser(
        "chat",
        prog="fem-agent",
        help="start a local Agent session",
        description="Start a local FEM Agent session (the chat word is optional).",
    )
    chat.add_argument(
        "--workspace",
        type=Path,
        default=None,
        help="local Agent workspace (default: value beside the config file)",
    )
    chat.add_argument(
        "--config",
        type=Path,
        help=f"local configuration file (default: discover {ROOT_CONFIG_NAME})",
    )
    chat.add_argument("--session", help="reopen an existing session ID")
    chat.add_argument(
        "--provider",
        choices=("deepseek", "fake"),
        default=None,
        help="cloud provider (default: FEM_AGENT_PROVIDER or deepseek)",
    )
    chat.add_argument("--model", help="provider model override")
    chat.add_argument(
        "--base-url",
        help="official DeepSeek OpenAI-compatible URL override",
    )
    chat.add_argument("--timeout", type=float, help="provider timeout in seconds")
    chat.add_argument("--max-retries", type=int, help="bounded provider retries")
    chat.add_argument(
        "--max-output-tokens",
        type=int,
        help="maximum provider output tokens",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    launched_without_arguments = not arguments
    if not arguments or arguments[0].startswith("-"):
        arguments.insert(0, "chat")
    args = build_parser().parse_args(arguments)
    if args.command != "chat":
        raise ValueError(f"unsupported command {args.command!r}")
    config_path: Path | None = None
    try:
        config_path = _main_config_path(args.config)
        if not config_path.exists():
            created = create_main_config_template(config_path)
            message = (
                _config_template_created_message(config_path)
                if created
                else _config_appeared_during_start_message(config_path)
            )
            _write(sys.stderr, message)
            _pause_after_config_setup(launched_without_arguments)
            return 2
        file_config = LocalAgentConfig.load(config_path)
        resolved = resolve_local_config(
            file_config,
            provider=args.provider,
            model=args.model,
            base_url=args.base_url,
            workspace=args.workspace,
            timeout_seconds=args.timeout,
            max_retries=args.max_retries,
            max_output_tokens=args.max_output_tokens,
            environ=os.environ,
        )
        if resolved.provider.casefold() not in {"deepseek", "fake"}:
            raise ConfigError("provider must be 'deepseek' or 'fake'")
        resolved.require_api_key()
    except ConfigError as error:
        _write(
            sys.stderr,
            _config_error_message(
                error,
                config_path=config_path,
                explicit_path=args.config,
            ),
        )
        return 2
    args.local_config = resolved
    args.config = config_path
    args.workspace = resolved.workspace
    return run_chat(args)


def run_chat(
    args: argparse.Namespace,
    *,
    input_func: InputFunction | None = None,
    output: TextIO | None = None,
) -> int:
    if input_func is None and output is None and _stdio_is_interactive():
        return _run_prompt_toolkit_chat(args)
    return _run_chat_loop(
        args,
        input_func=input if input_func is None else input_func,
        output=sys.stdout if output is None else output,
    )


def _run_prompt_toolkit_chat(args: argparse.Namespace) -> int:
    try:
        from prompt_toolkit import PromptSession
        from prompt_toolkit.patch_stdout import patch_stdout
    except ImportError as error:
        raise RuntimeError(
            "Interactive FEM Agent requires the 'agent' optional dependencies."
        ) from error

    session = PromptSession()
    with patch_stdout():
        return _run_chat_loop(
            args,
            input_func=session.prompt,
            output=sys.stdout,
        )


def _stdio_is_interactive() -> bool:
    try:
        return bool(sys.stdin.isatty() and sys.stdout.isatty())
    except (AttributeError, OSError):
        return False


def _run_chat_loop(
    args: argparse.Namespace,
    *,
    input_func: InputFunction,
    output: TextIO,
) -> int:
    provider = _provider_from_args(args)
    engine = AgentSessionEngine(
        args.workspace,
        provider,
        session_id=args.session,
    )
    local_store = ArtifactStore(engine.workspace)
    runner = _OperationRunner(engine, output)
    _render_startup(engine, output)

    while True:
        try:
            text = input_func("User> ")
        except EOFError:
            runner.write("\n")
            if not runner.stop_for_shutdown():
                runner.write(
                    "The active operation did not stop within the shutdown "
                    "timeout; the session remains open.\n"
                )
                return 1
            engine.close_session()
            return 0
        except KeyboardInterrupt:
            runner.write("\nCancellation requested.\n")
            runner.cancel()
            continue
        text = text.strip()
        if not text:
            continue
        try:
            if text.startswith("/"):
                normalized = text.partition(" ")[0].casefold()
                if normalized == "/cancel":
                    runner.cancel()
                    continue
                if normalized == "/confirm":
                    runner.start(
                        "confirmation run",
                        engine.confirm_revision,
                    )
                    continue
                if normalized == "/retry":
                    runner.start(
                        "worker retry",
                        engine.retry_transient_run,
                    )
                    continue
                if normalized in _EXIT_COMMANDS:
                    if not runner.stop_for_shutdown():
                        runner.write(
                            "The active operation is still stopping; the "
                            "session remains open.\n"
                        )
                        continue
                    engine.close_session()
                    runner.write("Session saved. Goodbye.\n")
                    return 0
                if (
                    runner.active
                    and normalized
                    not in {"/help", "/status", "/artifacts"}
                ):
                    runner.write(
                        "Wait for the active operation or enter /cancel.\n"
                    )
                    continue
                if normalized == "/attach":
                    prepared = _prepare_attachment(
                        text,
                        engine,
                        local_store,
                        input_func,
                        output,
                    )
                    if prepared is None:
                        continue
                    artifact, replace = prepared
                    runner.write(
                        f"Copied {Path(artifact.display_path).name} "
                        f"({artifact.sha256[:12]}…).\n"
                    )
                    runner.start(
                        "input inspection",
                        lambda: engine.attach_artifact(
                            artifact.artifact_id,
                            replace_existing=replace,
                        ),
                        suppress_live_diagnostics=True,
                        completion_message=(
                            "[operation] input inspection completed; "
                            "use /summary to review.\n"
                        ),
                        completion_issue_message=(
                            "[operation] input inspection completed with "
                            "issues; use /summary to review.\n"
                        ),
                    )
                    continue
                should_exit = runner.run_local(
                    lambda: _handle_command(
                        text,
                        engine,
                        local_store,
                        input_func,
                        output,
                    )
                )
                if should_exit:
                    return 0
                continue
            runner.start(
                "cloud turn",
                lambda message=text: engine.send_message(message),
            )
        except KeyboardInterrupt:
            runner.write("\nCancellation requested.\n")
            runner.cancel()
        except Exception as error:
            runner.write(_prefixed_text("Error: ", error))


def _provider_from_args(args: argparse.Namespace):
    local_config = getattr(args, "local_config", None)
    if isinstance(local_config, LocalAgentConfig):
        config = local_config.provider_config()
        if config.provider.casefold() == "fake":
            return FakeProvider()
        if config.provider.casefold() != "deepseek":
            raise ValueError(f"unsupported provider {config.provider!r}")
        return DeepSeekProvider(
            config,
            environ=local_config.provider_environment(),
        )
    config = ProviderConfig.from_env(
        provider=args.provider,
        model=args.model,
        base_url=args.base_url,
        timeout_seconds=args.timeout,
        max_retries=args.max_retries,
        max_output_tokens=args.max_output_tokens,
    )
    if config.provider.casefold() == "fake":
        return FakeProvider()
    if config.provider.casefold() != "deepseek":
        raise ValueError(f"unsupported provider {config.provider!r}")
    return DeepSeekProvider(config)


def _main_config_path(explicit_path: Path | None) -> Path:
    if explicit_path is not None:
        return explicit_path.expanduser().resolve()
    discovered = find_main_config()
    if discovered is None:
        return _suggested_main_config_path()
    return discovered


def _config_template_created_message(path: Path) -> str:
    return (
        f"Created a FEM Agent configuration template: {path}\n"
        'Open it, replace the empty "api_key" value with your DeepSeek '
        "API Key, then run fem-agent again.\n"
    )


def _config_appeared_during_start_message(path: Path) -> str:
    return (
        f"A FEM Agent configuration file appeared during startup: {path}\n"
        "For safety, this run did not open a workspace or contact the cloud. "
        "Review the file, then run fem-agent again.\n"
    )


def _pause_after_config_setup(launched_without_arguments: bool) -> None:
    """Keep a double-clicked console readable after first-run setup."""

    if not launched_without_arguments or not sys.stdin.isatty():
        return
    _write(
        sys.stderr,
        "Press Enter to close this window, then fill in api_key.\n",
    )
    try:
        sys.stdin.readline()
    except (EOFError, KeyboardInterrupt, OSError):
        pass


def _config_error_message(
    error: ConfigError,
    *,
    config_path: Path | None,
    explicit_path: Path | None,
) -> str:
    if config_path is not None:
        path = config_path
    elif explicit_path is not None:
        path = explicit_path.expanduser().resolve()
    else:
        path = _suggested_main_config_path()
    return (
        f"FEM Agent configuration is incomplete: {error}\n"
        f"Edit or create: {path}\n"
        "Set api_key and the official DeepSeek base_url, then run "
        "fem-agent again.\n"
    )


def _suggested_main_config_path() -> Path:
    for start in (Path.cwd().resolve(), Path(__file__).resolve().parent):
        for directory in (start, *start.parents):
            if (
                (directory / "pyproject.toml").is_file()
                and (directory / "src" / "fem_agent").is_dir()
            ):
                return directory / ROOT_CONFIG_NAME
    return Path.cwd().resolve() / ROOT_CONFIG_NAME


def _handle_command(
    command: str,
    engine: AgentSessionEngine,
    local_store: ArtifactStore,
    input_func: InputFunction,
    output: TextIO,
) -> bool:
    name = command.partition(" ")[0]
    normalized = name.casefold()
    if normalized == "/help":
        _write(output, _help_text())
        return False
    if normalized == "/attach":
        prepared = _prepare_attachment(
            command,
            engine,
            local_store,
            input_func,
            output,
        )
        if prepared is None:
            return False
        artifact, replace = prepared
        _render_events(
            engine.attach_artifact(
                artifact.artifact_id,
                replace_existing=replace,
            ),
            output,
        )
        _write(
            output,
            f"Attached {Path(artifact.display_path).name} "
            f"({artifact.sha256[:12]}…).\n"
        )
        return False
    if normalized == "/status":
        snapshot = engine.get_snapshot()
        _write(
            output,
            f"session={snapshot.session_id} phase={snapshot.phase.value} "
            f"revision={snapshot.revision} confirmed={snapshot.confirmed} "
            f"run={snapshot.active_run_id or '-'}\n"
        )
        return False
    if normalized == "/summary":
        _render_summary(engine.get_analysis_summary(), output)
        return False
    if normalized == "/confirm":
        _render_events(engine.confirm_revision(), output)
        return False
    if normalized == "/retry":
        _render_events(engine.retry_transient_run(), output)
        return False
    if normalized == "/cancel":
        _render_events(engine.cancel_active_operation(), output)
        return False
    if normalized == "/artifacts":
        records = engine.list_artifacts()
        if not records:
            _write(output, "No artifacts are registered.\n")
        for artifact in records:
            display_path = (
                Path(artifact.display_path).name
                if artifact.kind == "input"
                else artifact.display_path
            )
            _write(
                output,
                f"- [{artifact.kind}] "
                f"{display_path} {artifact.size_bytes} bytes "
                f"sha256={artifact.sha256[:12]}…\n"
            )
        return False
    if normalized == "/new":
        _render_events(engine.create_session(), output)
        _write(output, f"New session: {engine.session_id}\n")
        return False
    if normalized in _EXIT_COMMANDS:
        engine.close_session()
        _write(output, "Session saved. Goodbye.\n")
        return True
    _write(
        output,
        f"Unknown command {name!r}. Enter /help for available commands.\n",
    )
    return False


def _prepare_attachment(
    command: str,
    engine: AgentSessionEngine,
    local_store: ArtifactStore,
    input_func: InputFunction,
    output: TextIO,
) -> tuple[ArtifactRecord, bool] | None:
    _name, _, remainder = command.partition(" ")
    argument = _unquote_path(remainder.strip())
    if not argument:
        _write(output, "Usage: /attach <path-to-model.inp>\n")
        return None
    replace = False
    current = engine.revisions.latest(engine.session_id)
    if current is not None:
        answer = input_func(
            "This session already has an input. Replace it? [y/N] "
        )
        if answer.strip().casefold() not in {"y", "yes"}:
            _write(output, "Attachment cancelled.\n")
            return None
        replace = True
    artifact = local_store.copy_input(
        engine.session_id,
        Path(argument),
    )
    return artifact, replace


def _render_startup(engine: AgentSessionEngine, output: TextIO) -> None:
    snapshot = engine.get_snapshot()
    _write(
        output,
        "FEM Agent V0\n"
        f"provider: {snapshot.provider}\n"
        f"model: {snapshot.model}\n"
        f"session: {snapshot.session_id}\n"
        f"workspace: {snapshot.workspace}\n"
        f"cloud communication: {'enabled' if snapshot.cloud_enabled else 'disabled'}\n"
    )
    if snapshot.cloud_enabled:
        _write(
            output,
            "Privacy: conversation text, bounded model summaries, tool schemas, "
            "and bounded results are sent to the configured provider. Raw .inp "
            "content and full arrays remain local. DeepSeek documents disk context "
            "caching and does not document a request-level disable switch.\n"
        )
    _write(
        output,
        "V0 scope: one supported linear-static Abaqus .inp step; no model "
        "editing, nonlinear analysis, dynamics, contact, or unit conversion.\n"
    )
    _write(output, "Enter /help for commands; /exit to save and exit.\n")


def _render_events(
    events: tuple[EngineEvent, ...],
    output: TextIO,
    *,
    agent_label_emitted: bool = False,
) -> bool:
    for event in events:
        if event.event == EngineEventType.MESSAGE_DELTA:
            _write(
                output,
                _agent_text(
                    event.data["text"],
                    include_label=not agent_label_emitted,
                ),
            )
            agent_label_emitted = True
        elif event.event == EngineEventType.TOOL_STARTED:
            _write(output, f"[tool] {event.data['tool']}…\n")
        elif event.event == EngineEventType.TOOL_COMPLETED:
            result = event.data["result"]
            _write(
                output,
                f"[tool] {event.data['tool']}: {result['summary']}\n"
            )
        elif event.event == EngineEventType.ANALYSIS_SUMMARY:
            _render_summary(
                AnalysisSummary.from_dict(event.data["analysis_summary"]),
                output,
            )
        elif event.event == EngineEventType.DIAGNOSTIC:
            diagnostic = event.data["diagnostic"]
            _render_diagnostic(
                output,
                severity=diagnostic["severity"],
                code=diagnostic["code"],
                message=diagnostic["message"],
                entity=diagnostic.get("entity"),
                remediation=diagnostic.get("remediation"),
            )
        elif event.event == EngineEventType.RUN_PROGRESS:
            _write(output, f"[run] {event.data['stage']}\n")
        elif event.event == EngineEventType.RUN_COMPLETED:
            _write(
                output,
                f"[run] {event.data['run_id']} "
                f"{event.data['status']}\n"
            )
            if event.data["status"] == "succeeded":
                _write(
                    output,
                    "[run] solution saved; ask Agent what result "
                    "you want to analyze.\n",
                )
        elif event.event == EngineEventType.OPERATION_CANCELLED:
            _write(
                output,
                f"[cancelled] {event.data.get('scope', 'operation')}\n"
            )
    return agent_label_emitted


def _render_summary(summary: AnalysisSummary, output: TextIO) -> None:
    def render_items(label: str, items) -> None:
        _write(
            output,
            f"- {label}: "
            + (
                json.dumps(
                    list(items),
                    ensure_ascii=False,
                    sort_keys=True,
                )
                if items
                else "[]"
            )
            + "\n"
        )

    _write(
        output,
        f"Analysis summary revision {summary.revision}\n"
        f"- revision sha256: {summary.revision_hash}\n"
        f"- model: {summary.model_name}\n"
        f"- source input sha256: {summary.source_sha256}\n"
        f"- mesh: {summary.node_count} nodes, {summary.element_count} elements, "
        f"{summary.total_dofs} DOFs\n"
        f"- element types: {json.dumps(summary.element_types, ensure_ascii=False)}\n"
        f"- resource class: {summary.resource_class}\n"
    )
    render_items("node sets", summary.node_sets)
    render_items("element sets", summary.element_sets)
    render_items("edges", summary.edges)
    render_items("surfaces", summary.surfaces)
    render_items("materials", summary.materials)
    render_items("sections", summary.sections)
    _write(
        output,
        "- analysis step: "
        + (
            "-"
            if summary.analysis_step is None
            else json.dumps(
                summary.analysis_step,
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        + "\n"
    )
    render_items("constraints", summary.constraints)
    render_items("loads", summary.loads)
    if summary.unit_context is None:
        _write(output, "- units: missing\n")
    else:
        units = summary.unit_context
        _write(
            output,
            f"- units: length={units.length}, force={units.force}, "
            f"stress={units.stress}, density={units.density}, "
            f"acceleration={units.acceleration}\n"
        )
    _write(
        output,
        "- precomputed results (optional): "
        + ", ".join(query.kind.value for query in summary.requested_queries)
        + "\n"
        "- exports: "
        + ", ".join(item.value for item in summary.export_formats)
        + "\n"
    )
    render_items("keyword inventory", summary.keyword_inventory)
    if summary.collections_truncated:
        _write(
            output,
            "- note: one or more bounded summary collections were truncated\n"
        )
    for diagnostic in summary.diagnostics:
        _render_diagnostic(
            output,
            severity=diagnostic.severity.value,
            code=diagnostic.code,
            message=diagnostic.message,
            entity=diagnostic.entity,
            remediation=diagnostic.remediation,
        )
    if not summary.has_blocking_diagnostics:
        _write(
            output,
            "Enter /confirm to validate, solve, and save local results.\n",
        )


def _render_diagnostic(
    output: TextIO,
    *,
    severity: str,
    code: str,
    message: str,
    entity: str | None,
    remediation: str | None,
) -> None:
    context = "" if entity is None else f" [{entity}]"
    _write(output, f"[{severity}] {code}{context}: {message}\n")
    if remediation is not None:
        _write(output, f"  next: {remediation}\n")


def _write(output: TextIO, value: object) -> None:
    output.write(_terminal_text(value))


def _agent_text(value: object, *, include_label: bool = True) -> str:
    safe = _terminal_text(value)
    if not safe:
        return "Agent:\n" if include_label else "\n"
    lines = safe.split("\n")
    if safe.endswith("\n"):
        lines.pop()
    body = "".join(
        "│\n" if not line else f"│ {line}\n"
        for line in lines
    )
    return ("Agent:\n" if include_label else "") + body


def _prefixed_text(prefix: str, value: object) -> str:
    safe = _terminal_text(value)
    return "".join(
        f"{prefix}{line}\n"
        for line in safe.splitlines()
    ) or f"{prefix.rstrip()}\n"


def _terminal_text(value: object) -> str:
    text = str(value)
    chunks: list[str] = []
    for character in text:
        codepoint = ord(character)
        if character in {"\n", "\t"}:
            chunks.append(character)
        elif unicodedata.category(character) in {"Cc", "Cf", "Cs", "Zl", "Zp"}:
            chunks.append(f"\\u{codepoint:04x}")
        else:
            chunks.append(character)
    return "".join(chunks)


def _unquote_path(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]
    return value


def _help_text() -> str:
    return (
        "/help                 show this help\n"
        "/attach <path>        copy and attach a local Abaqus .inp\n"
        "/status               show session and revision state\n"
        "/summary              show the deterministic confirmation summary\n"
        "/confirm              confirm the exact revision and run automatically\n"
        "/retry                retry a transient worker failure for this revision\n"
        "/cancel               cancel the active provider or worker operation\n"
        "/artifacts            list local run artifacts\n"
        "/new                  create a new session\n"
        "/exit, /quit, /q      save and exit\n"
    )


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_parser", "main", "run_chat"]
