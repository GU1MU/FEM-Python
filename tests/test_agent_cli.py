from contextlib import contextmanager
import json
from io import StringIO
import os
import sys
import threading

import pytest

from fem_agent import cli
from fem_agent.artifacts import ArtifactStore
from fem_agent.engine import AgentSessionEngine
from fem_agent.providers.base import (
    AssistantMessage,
    ProviderResponse,
    ProviderUnavailableError,
)
from fem_agent.providers.fake import FakeProvider
from fem_agent.tools.registry import ToolExecutionContext
from tests.helpers.abaqus_builders import write_perforated_plate_style_inp


def _args(tmp_path, **overrides):
    parser = cli.build_parser()
    values = parser.parse_args(
        ["chat", "--workspace", str(tmp_path / "workspace"), "--provider", "fake"]
    )
    for name, value in overrides.items():
        setattr(values, name, value)
    return values


def _write_main_config(tmp_path, **overrides):
    document = {
        "provider": "fake",
        "model": "local-test-model",
        "base_url": "https://api.deepseek.com",
        "api_key": "",
        "workspace": "workspace",
        "timeout_seconds": 10,
        "max_retries": 0,
        "max_output_tokens": 128,
    }
    document.update(overrides)
    path = tmp_path / "fem-agent.config.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def test_main_without_subcommand_defaults_to_chat(monkeypatch, tmp_path):
    config_path = _write_main_config(tmp_path)
    captured = {}
    monkeypatch.setattr(cli, "find_main_config", lambda: config_path)

    def capture_run(args):
        captured["args"] = args
        return 0

    monkeypatch.setattr(cli, "run_chat", capture_run)

    assert cli.main([]) == 0
    assert captured["args"].command == "chat"
    assert captured["args"].config == config_path
    assert captured["args"].workspace == (tmp_path / "workspace").resolve()
    assert captured["args"].local_config.provider == "fake"


def test_no_argument_help_does_not_advertise_chat_as_required(capsys):
    with pytest.raises(SystemExit) as captured:
        cli.main(["--help"])

    output = capsys.readouterr().out
    assert captured.value.code == 0
    assert "usage: fem-agent [-h]" in output
    assert "usage: fem-agent chat" not in output
    assert "the chat word is optional" in output


def test_main_finds_root_config_from_venv_scripts_working_directory(
    monkeypatch,
    tmp_path,
):
    project = tmp_path / "project"
    scripts = project / ".venv" / "Scripts"
    scripts.mkdir(parents=True)
    (project / "src" / "fem_agent").mkdir(parents=True)
    (project / "pyproject.toml").write_text(
        "[project]\nname = \"fem-project\"\n",
        encoding="utf-8",
    )
    config_path = _write_main_config(project)
    captured = {}
    monkeypatch.chdir(scripts)
    monkeypatch.setattr(
        cli,
        "run_chat",
        lambda args: captured.setdefault("args", args) and 0,
    )

    assert cli.main([]) == 0
    assert captured["args"].config == config_path
    assert captured["args"].workspace == (project / "workspace").resolve()


def test_main_creates_missing_config_template_before_creating_workspace(
    monkeypatch,
    tmp_path,
    capsys,
):
    missing = tmp_path / "missing.json"
    workspace = tmp_path / "must-not-exist"
    monkeypatch.setattr(
        cli,
        "run_chat",
        lambda args: pytest.fail("run_chat must not start without config"),
    )

    code = cli.main(
        [
            "--config",
            str(missing),
            "--workspace",
            str(workspace),
        ]
    )

    captured = capsys.readouterr()
    assert code == 2
    assert str(missing) in captured.err
    assert "Created a FEM Agent configuration template" in captured.err
    assert '"api_key"' in captured.err
    document = json.loads(missing.read_text(encoding="utf-8"))
    assert document["api_key"] == ""
    assert document["base_url"] == "https://api.deepseek.com"
    assert not workspace.exists()


def test_main_exits_if_a_config_appears_during_template_creation(
    monkeypatch,
    tmp_path,
    capsys,
):
    config_path = tmp_path / "raced.json"
    raced_document = {
        "provider": "fake",
        "api_key": "other-process-value",
    }

    def simulate_racing_creator(path):
        path.write_text(json.dumps(raced_document), encoding="utf-8")
        return False

    monkeypatch.setattr(
        cli,
        "create_main_config_template",
        simulate_racing_creator,
    )
    monkeypatch.setattr(
        cli,
        "run_chat",
        lambda args: pytest.fail("raced config must not start chat"),
    )

    code = cli.main(["--config", str(config_path)])

    captured = capsys.readouterr()
    assert code == 2
    assert "appeared during startup" in captured.err
    assert "did not open a workspace or contact the cloud" in captured.err
    assert json.loads(config_path.read_text(encoding="utf-8")) == raced_document


def test_no_argument_first_run_creates_template_at_project_root_and_pauses(
    monkeypatch,
    tmp_path,
    capsys,
):
    project = tmp_path / "project"
    nested = project / "nested"
    nested.mkdir(parents=True)
    (project / "src" / "fem_agent").mkdir(parents=True)
    (project / "pyproject.toml").write_text(
        "[project]\nname = \"fem-project\"\n",
        encoding="utf-8",
    )
    terminal_input = StringIO("\n")
    terminal_input.isatty = lambda: True
    monkeypatch.chdir(nested)
    monkeypatch.setattr(cli, "find_main_config", lambda: None)
    monkeypatch.setattr(cli.sys, "stdin", terminal_input)
    monkeypatch.setattr(
        cli,
        "run_chat",
        lambda args: pytest.fail("first run must stop after setup"),
    )

    code = cli.main([])

    config_path = project / "fem-agent.config.json"
    captured = capsys.readouterr()
    assert code == 2
    assert config_path.exists()
    assert str(config_path) in captured.err
    assert "Press Enter to close this window" in captured.err
    assert terminal_input.tell() == 1


def test_main_does_not_create_directories_for_a_mistyped_config_path(
    monkeypatch,
    tmp_path,
    capsys,
):
    missing_parent = tmp_path / "missing-parent"
    config_path = missing_parent / "config.json"
    monkeypatch.setattr(
        cli,
        "run_chat",
        lambda args: pytest.fail("run_chat must not start without config"),
    )

    code = cli.main(["--config", str(config_path)])

    captured = capsys.readouterr()
    assert code == 2
    assert "parent directory" in captured.err
    assert str(config_path) in captured.err
    assert not missing_parent.exists()


def test_main_reports_empty_file_key_without_echoing_or_starting(
    monkeypatch,
    tmp_path,
    capsys,
):
    config_path = _write_main_config(
        tmp_path,
        provider="deepseek",
        api_key="",
    )
    original = config_path.read_bytes()
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setattr(cli, "find_main_config", lambda: config_path)
    monkeypatch.setattr(
        cli,
        "run_chat",
        lambda args: pytest.fail("run_chat must not start without a key"),
    )

    code = cli.main([])

    captured = capsys.readouterr()
    assert code == 2
    assert str(config_path) in captured.err
    assert "api_key is missing or empty" in captured.err
    assert "Created a FEM Agent configuration template" not in captured.err
    assert config_path.read_bytes() == original


def test_file_key_is_passed_privately_to_provider_without_mutating_environment(
    monkeypatch,
    tmp_path,
):
    secret = "private-config-key-123456"
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    args = _args(tmp_path, provider="deepseek")
    args.local_config = cli.LocalAgentConfig(
        api_key=secret,
        workspace=tmp_path / "workspace",
    )

    provider = cli._provider_from_args(args)

    assert provider.contains_configured_credential(f"value={secret}")
    assert "DEEPSEEK_API_KEY" not in os.environ
    assert secret not in repr(args.local_config)


@pytest.mark.parametrize("exit_command", ("/exit", "/quit", "/q"))
def test_cli_help_status_and_exit_are_local(
    monkeypatch,
    tmp_path,
    exit_command,
):
    provider = FakeProvider()
    monkeypatch.setattr(cli, "_provider_from_args", lambda args: provider)
    commands = iter(["/help", "/status", exit_command])
    output = StringIO()

    code = cli.run_chat(
        _args(tmp_path),
        input_func=lambda prompt: next(commands),
        output=output,
    )

    rendered = output.getvalue()
    assert code == 0
    assert "/attach <path>" in rendered
    assert "/exit, /quit, /q" in rendered
    assert "phase=empty" in rendered
    assert "Session saved" in rendered
    assert provider.requests == []


def test_cli_eof_closes_without_corrupting_session(monkeypatch, tmp_path):
    monkeypatch.setattr(
        cli,
        "_provider_from_args",
        lambda args: FakeProvider(),
    )
    output = StringIO()

    code = cli.run_chat(
        _args(tmp_path),
        input_func=lambda prompt: (_ for _ in ()).throw(EOFError()),
        output=output,
    )

    assert code == 0
    sessions = list((tmp_path / "workspace" / "sessions").iterdir())
    assert len(sessions) == 1


def test_run_chat_routes_default_tty_io_through_prompt_toolkit(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(cli, "_stdio_is_interactive", lambda: True)
    monkeypatch.setattr(
        cli,
        "_run_prompt_toolkit_chat",
        lambda args: 17,
    )

    assert cli.run_chat(_args(tmp_path)) == 17


@pytest.mark.optional_runtime
def test_prompt_toolkit_chat_passes_patched_stdout_to_chat_loop(
    monkeypatch,
    tmp_path,
):
    prompt_toolkit = pytest.importorskip(
        "prompt_toolkit",
        reason="[optional-native-runtime] prompt-toolkit is unavailable",
    )
    patch_stdout_module = pytest.importorskip(
        "prompt_toolkit.patch_stdout",
        reason="[optional-native-runtime] prompt-toolkit is unavailable",
    )

    captured = {}

    class FakePromptSession:
        def prompt(self, message):
            return message

    @contextmanager
    def fake_patch_stdout():
        original = sys.stdout
        proxy = StringIO()
        captured["proxy"] = proxy
        sys.stdout = proxy
        try:
            yield
        finally:
            sys.stdout = original

    def fake_chat_loop(args, *, input_func, output):
        captured["prompt_owner"] = input_func.__self__
        captured["output"] = output
        return 19

    monkeypatch.setattr(prompt_toolkit, "PromptSession", FakePromptSession)
    monkeypatch.setattr(
        patch_stdout_module,
        "patch_stdout",
        fake_patch_stdout,
    )
    monkeypatch.setattr(cli, "_run_chat_loop", fake_chat_loop)

    assert cli._run_prompt_toolkit_chat(_args(tmp_path)) == 19
    assert isinstance(captured["prompt_owner"], FakePromptSession)
    assert captured["output"] is captured["proxy"]


def test_cli_runs_attachment_inspection_as_a_background_operation(
    monkeypatch,
    tmp_path,
):
    class CompletionOutput(StringIO):
        def __init__(self):
            super().__init__()
            self.inspection_completed = threading.Event()

        def write(self, value):
            written = super().write(value)
            if "[operation] input inspection completed" in value:
                self.inspection_completed.set()
            return written

    source = write_perforated_plate_style_inp(
        tmp_path,
        "cli background.inp",
        ("*Boundary", "Set-right, 1, 1, 0.05"),
    )
    monkeypatch.setattr(cli, "_provider_from_args", lambda args: FakeProvider())
    inspection_entered = threading.Event()
    release_inspection = threading.Event()

    def inspect_in_background(self, artifact_id, *, replace_existing=False):
        inspection_entered.set()
        assert release_inspection.wait(2)
        return ()

    monkeypatch.setattr(
        AgentSessionEngine,
        "_attach_artifact",
        inspect_in_background,
    )
    output = CompletionOutput()
    command_index = 0

    def next_command(prompt):
        nonlocal command_index
        command_index += 1
        if command_index == 1:
            return f'/attach "{source}"'
        if command_index == 2:
            assert inspection_entered.wait(2)
            release_inspection.set()
            assert output.inspection_completed.wait(5)
            return "/status"
        return "/exit"

    code = cli.run_chat(
        _args(tmp_path),
        input_func=next_command,
        output=output,
    )

    rendered = output.getvalue()
    assert code == 0
    assert "Copied cli background.inp" in rendered
    copied_line = next(
        line for line in rendered.splitlines() if line.startswith("Copied ")
    )
    assert "starting inspection" not in copied_line
    assert "art_" not in rendered
    assert "[operation] input inspection started" in rendered
    assert (
        "[operation] input inspection completed; use /summary to review."
        in rendered
    )


def test_cli_does_not_render_attachment_diagnostic_inside_active_prompt(
    monkeypatch,
    tmp_path,
):
    source = write_perforated_plate_style_inp(
        tmp_path,
        "quiet attachment.inp",
        ("*Boundary", "Set-right, 1, 1, 0.05"),
    )
    source.write_text(
        "*Heading\nquiet attachment regression fixture\n"
        + source.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    prompt_active = threading.Event()
    inspection_finished = threading.Event()
    ignored_metadata_produced = threading.Event()

    def attach_after_next_prompt(self, artifact_id, *, replace_existing=False):
        assert prompt_active.wait(2)
        try:
            event = self._event(
                cli.EngineEventType.DIAGNOSTIC,
                {
                    "diagnostic": {
                        "severity": "info",
                        "code": "IGNORED_METADATA",
                        "message": "Ignored input metadata.",
                    }
                },
            )
            ignored_metadata_produced.set()
            return (event,)
        finally:
            inspection_finished.set()

    monkeypatch.setattr(
        AgentSessionEngine,
        "_attach_artifact",
        attach_after_next_prompt,
    )
    monkeypatch.setattr(cli, "_provider_from_args", lambda args: FakeProvider())
    output = StringIO()
    command_index = 0

    def next_command(prompt):
        nonlocal command_index
        output.write(prompt)
        command_index += 1
        if command_index == 1:
            return f'/attach "{source}"'
        assert command_index == 2
        prompt_active.set()
        assert inspection_finished.wait(2)
        return "/exit"

    code = cli.run_chat(
        _args(tmp_path),
        input_func=next_command,
        output=output,
    )

    rendered = output.getvalue()
    assert code == 0
    assert ignored_metadata_produced.is_set()
    assert "User> [info]" not in rendered
    assert "IGNORED_METADATA" not in rendered


def test_operation_runner_restores_live_diagnostics_after_quiet_operation():
    class CoordinatedEngine:
        def __init__(self):
            self.sink = None
            self.operation_started = threading.Event()

        def subscribe(self, sink):
            self.sink = sink
            return lambda: None

        def reset_operation_start_signal(self):
            self.operation_started.clear()

        def wait_for_operation_start(self):
            return self.operation_started.wait(2)

        def cancel_active_operation(self):
            return ()

    def diagnostic_event(code):
        return cli.EngineEvent(
            cli.EngineEventType.DIAGNOSTIC,
            "test-session",
            {
                "diagnostic": {
                    "severity": "info",
                    "code": code,
                    "message": f"{code} message",
                }
            },
            "2026-07-25T00:00:00Z",
        )

    engine = CoordinatedEngine()
    output = StringIO()
    runner = cli._OperationRunner(engine, output)
    quiet_finished = threading.Event()
    visible_finished = threading.Event()

    def quiet_operation():
        event = diagnostic_event("ATTACH_ONLY_INFO")
        engine.operation_started.set()
        engine.sink(event)
        quiet_finished.set()
        return (event,)

    assert runner.start(
        "input inspection",
        quiet_operation,
        suppress_live_diagnostics=True,
        completion_message="[operation] input inspection completed.\n",
    )
    assert quiet_finished.wait(2)
    assert runner._thread is not None
    runner._thread.join(2)
    assert not runner.active

    def visible_operation():
        event = diagnostic_event("NEXT_OPERATION_INFO")
        engine.operation_started.set()
        engine.sink(event)
        visible_finished.set()
        return (event,)

    assert runner.start("cloud turn", visible_operation)
    assert visible_finished.wait(2)
    assert runner._thread is not None
    runner._thread.join(2)
    assert not runner.active

    rendered = output.getvalue()
    assert "ATTACH_ONLY_INFO" not in rendered
    assert "[operation] input inspection completed." in rendered
    assert "[info] NEXT_OPERATION_INFO" in rendered


def test_operation_runner_skips_completion_message_after_cancellation():
    class CoordinatedEngine:
        def __init__(self):
            self.sink = None
            self.operation_started = threading.Event()

        def subscribe(self, sink):
            self.sink = sink
            return lambda: None

        def reset_operation_start_signal(self):
            self.operation_started.clear()

        def wait_for_operation_start(self):
            return self.operation_started.wait(2)

        def cancel_active_operation(self):
            return ()

    engine = CoordinatedEngine()
    output = StringIO()
    runner = cli._OperationRunner(engine, output)

    def cancelled_operation():
        event = cli.EngineEvent(
            cli.EngineEventType.OPERATION_CANCELLED,
            "test-session",
            {"scope": "inspection"},
            "2026-07-25T00:00:00Z",
        )
        engine.operation_started.set()
        engine.sink(event)
        return (event,)

    assert runner.start(
        "input inspection",
        cancelled_operation,
        completion_message="[operation] input inspection completed.\n",
    )
    assert runner._thread is not None
    runner._thread.join(2)

    rendered = output.getvalue()
    assert "[cancelled] inspection" in rendered
    assert "input inspection completed" not in rendered


def test_operation_runner_uses_issue_completion_for_error_diagnostic():
    class CoordinatedEngine:
        def __init__(self):
            self.sink = None
            self.operation_started = threading.Event()

        def subscribe(self, sink):
            self.sink = sink
            return lambda: None

        def reset_operation_start_signal(self):
            self.operation_started.clear()

        def wait_for_operation_start(self):
            return self.operation_started.wait(2)

        def cancel_active_operation(self):
            return ()

    engine = CoordinatedEngine()
    output = StringIO()
    runner = cli._OperationRunner(engine, output)

    def operation_with_issue():
        event = cli.EngineEvent(
            cli.EngineEventType.DIAGNOSTIC,
            "test-session",
            {
                "diagnostic": {
                    "severity": "error",
                    "code": "INVALID_INPUT",
                    "message": "Invalid model.",
                }
            },
            "2026-07-25T00:00:00Z",
        )
        engine.operation_started.set()
        engine.sink(event)
        return (event,)

    assert runner.start(
        "input inspection",
        operation_with_issue,
        suppress_live_diagnostics=True,
        completion_message="[operation] input inspection completed.\n",
        completion_issue_message=(
            "[operation] input inspection completed with issues.\n"
        ),
    )
    assert runner._thread is not None
    runner._thread.join(2)

    rendered = output.getvalue()
    assert "INVALID_INPUT" not in rendered
    assert "[operation] input inspection completed with issues." in rendered
    assert "[operation] input inspection completed.\n" not in rendered


def test_operation_runner_does_not_complete_after_late_cancel_request():
    release_operation = threading.Event()

    class CoordinatedEngine:
        def __init__(self):
            self.sink = None
            self.operation_started = threading.Event()

        def subscribe(self, sink):
            self.sink = sink
            return lambda: None

        def reset_operation_start_signal(self):
            self.operation_started.clear()

        def wait_for_operation_start(self):
            return self.operation_started.wait(2)

        def cancel_active_operation(self):
            release_operation.set()
            return (
                cli.EngineEvent(
                    cli.EngineEventType.OPERATION_CANCELLED,
                    "test-session",
                    {"scope": "inspection"},
                    "2026-07-25T00:00:00Z",
                ),
            )

    engine = CoordinatedEngine()
    output = StringIO()
    runner = cli._OperationRunner(engine, output)

    def operation_finishing_after_cancel():
        engine.operation_started.set()
        assert release_operation.wait(2)
        return ()

    assert runner.start(
        "input inspection",
        operation_finishing_after_cancel,
        completion_message="[operation] input inspection completed.\n",
    )
    runner.cancel()
    assert runner._thread is not None
    runner._thread.join(2)

    rendered = output.getvalue()
    assert "[cancelled] inspection" in rendered
    assert "input inspection completed" not in rendered


def test_operation_runner_labels_agent_once_per_cloud_turn():
    class CoordinatedEngine:
        def __init__(self):
            self.sink = None
            self.operation_started = threading.Event()

        def subscribe(self, sink):
            self.sink = sink
            return lambda: None

        def reset_operation_start_signal(self):
            self.operation_started.clear()

        def wait_for_operation_start(self):
            return self.operation_started.wait(2)

        def cancel_active_operation(self):
            return ()

    def message_event(text):
        return cli.EngineEvent(
            cli.EngineEventType.MESSAGE_DELTA,
            "test-session",
            {"text": text},
            "2026-07-25T00:00:00Z",
        )

    engine = CoordinatedEngine()
    output = StringIO()
    runner = cli._OperationRunner(engine, output)
    operation_finished = threading.Event()

    def operation():
        first = message_event("先检查模型。\n第二行说明。")
        second = message_event("检查完成。\n## 结果")
        engine.operation_started.set()
        engine.sink(first)
        engine.sink(second)
        operation_finished.set()
        return (first, second)

    assert runner.start("cloud turn", operation)
    assert operation_finished.wait(2)
    assert runner._thread is not None
    runner._thread.join(2)

    rendered = output.getvalue()
    assert rendered.count("Agent:") == 1
    assert "Agent:\n│ 先检查模型。\n│ 第二行说明。\n" in rendered
    assert "│ 检查完成。\n│ ## 结果\n" in rendered

    second_finished = threading.Event()

    def second_operation():
        event = message_event("第二轮回复。\n仍然只显示一次标签。")
        engine.operation_started.set()
        engine.sink(event)
        second_finished.set()
        return (event,)

    assert runner.start("cloud turn", second_operation)
    assert second_finished.wait(2)
    assert runner._thread is not None
    runner._thread.join(2)

    rendered = output.getvalue()
    assert rendered.count("Agent:") == 2
    assert "Agent:\n│ 第二轮回复。\n│ 仍然只显示一次标签。\n" in rendered


def test_operation_runner_releases_announcement_gate_when_start_is_interrupted():
    sink_entered = threading.Event()
    cancellation_released = threading.Event()
    operation_finished = threading.Event()

    class InterruptingOutput(StringIO):
        def write(self, value):
            if value.startswith("[operation] interrupted start started"):
                raise KeyboardInterrupt
            return super().write(value)

    class CoordinatedEngine:
        def __init__(self):
            self.sink = None
            self.cancel_calls = 0

        def subscribe(self, sink):
            self.sink = sink
            return lambda: None

        def reset_operation_start_signal(self):
            return None

        def wait_for_operation_start(self):
            assert sink_entered.wait(1)
            return True

        def cancel_active_operation(self):
            self.cancel_calls += 1
            cancellation_released.set()
            return ()

    engine = CoordinatedEngine()
    runner = cli._OperationRunner(engine, InterruptingOutput())

    def operation():
        sink_entered.set()
        engine.sink(
            cli.EngineEvent(
                cli.EngineEventType.MESSAGE_DELTA,
                "test-session",
                {"text": "worker event"},
                "2026-07-24T00:00:00Z",
            )
        )
        cancellation_released.wait()
        operation_finished.set()
        return ()

    try:
        with pytest.raises(KeyboardInterrupt):
            runner.start("interrupted start", operation)

        assert runner.stop_for_shutdown(
            grace_seconds=0,
            cancel_timeout_seconds=1,
        )
        assert operation_finished.is_set()
        assert engine.cancel_calls == 1
    finally:
        cancellation_released.set()
        if runner._announcement is not None:
            runner._announcement.set()
        if runner._thread is not None:
            runner._thread.join(1)


def test_cli_renders_provider_text_without_terminal_logic_in_engine(
    monkeypatch,
    tmp_path,
):
    provider = FakeProvider(
        [
            ProviderResponse(
                AssistantMessage("assistant", "请先使用 /attach。"),
                finish_reason="stop",
            )
        ]
    )
    monkeypatch.setattr(cli, "_provider_from_args", lambda args: provider)
    commands = iter(["你好", "/exit"])
    output = StringIO()

    code = cli.run_chat(
        _args(tmp_path),
        input_func=lambda prompt: next(commands),
        output=output,
    )

    rendered = output.getvalue()
    assert code == 0
    assert "Agent:\n│ 请先使用 /attach。" in rendered


def test_cli_run_completion_keeps_scalars_for_agent_analysis():
    output = StringIO()
    event = cli.EngineEvent(
        cli.EngineEventType.RUN_COMPLETED,
        "ses_cli_result",
        {
            "run_id": "run_cli_result",
            "status": "succeeded",
            "result_summary": {
                "scalars": [
                    {
                        "measure": "max_magnitude",
                        "value": 0.503840529776,
                        "unit": "mm",
                        "node_id": 4,
                    }
                ]
            },
            "artifacts": [],
        },
        "2026-07-25T00:00:00Z",
    )

    cli._render_events((event,), output)

    rendered = output.getvalue()
    assert "[run] run_cli_result succeeded" in rendered
    assert "[run] solution saved" in rendered
    assert "ask Agent what result you want to analyze" in rendered
    assert "local solution" not in rendered
    assert "0.503840529776" not in rendered
    assert "max_magnitude" not in rendered


def test_cli_escapes_terminal_control_sequences_from_provider(
    monkeypatch,
    tmp_path,
):
    provider = FakeProvider(
        [
            ProviderResponse(
                AssistantMessage(
                    "assistant",
                    (
                        "\x1b]52;c;clipboard-data\x07safe\u202etext"
                        "\u2028line\u061cmark\u200eltr\n[run] succeeded"
                    ),
                ),
                finish_reason="stop",
            )
        ]
    )
    monkeypatch.setattr(cli, "_provider_from_args", lambda args: provider)
    commands = iter(["你好", "/exit"])
    output = StringIO()

    cli.run_chat(
        _args(tmp_path),
        input_func=lambda prompt: next(commands),
        output=output,
    )

    rendered = output.getvalue()
    assert "\x1b" not in rendered
    assert "\x07" not in rendered
    assert "\u202e" not in rendered
    assert "\u2028" not in rendered
    assert "\u061c" not in rendered
    assert "\u200e" not in rendered
    assert "\\u001b" in rendered
    assert "\\u202e" in rendered
    assert "\\u2028" in rendered
    assert "\\u061c" in rendered
    assert "\\u200e" in rendered
    assert rendered.count("Agent:") == 1
    assert "\n│ [run] succeeded\n" in rendered
    assert "\n[run] succeeded\n" not in rendered


def test_cli_parser_exposes_deepseek_configuration():
    args = cli.build_parser().parse_args(
        [
            "chat",
            "--workspace",
            "local-workspace",
            "--provider",
            "deepseek",
            "--model",
            "deepseek-v4-pro",
            "--timeout",
            "45",
            "--max-retries",
            "1",
        ]
    )

    assert args.provider == "deepseek"
    assert args.model == "deepseek-v4-pro"
    assert args.timeout == 45
    assert args.max_retries == 1


def test_cli_cancel_remains_available_during_cloud_request(
    monkeypatch,
    tmp_path,
):
    class BlockingProvider:
        provider_name = "blocking"
        model_name = "blocking-test"

        def __init__(self):
            self.started = threading.Event()
            self.cancelled = threading.Event()

        def complete(self, messages, tools):
            self.started.set()
            self.cancelled.wait(5)
            raise ProviderUnavailableError("cancelled test request")

        def cancel_active_request(self):
            self.cancelled.set()

        def reset_cancellation(self):
            self.cancelled.clear()

    provider = BlockingProvider()
    monkeypatch.setattr(cli, "_provider_from_args", lambda args: provider)
    command_index = 0

    def next_command(prompt):
        nonlocal command_index
        command_index += 1
        if command_index == 1:
            return "等待云请求。"
        if command_index == 2:
            assert provider.started.wait(2)
            return "/cancel"
        assert provider.cancelled.wait(2)
        return "/exit"

    output = StringIO()
    code = cli.run_chat(
        _args(tmp_path),
        input_func=next_command,
        output=output,
    )

    assert code == 0
    assert provider.cancelled.is_set()
    assert "[cancelled] provider" in output.getvalue()


def test_cli_confirmation_summary_shows_engineering_review_content(tmp_path):
    source = write_perforated_plate_style_inp(
        tmp_path,
        "cli_summary.inp",
        ("*Dsload", "Surf-right, P, 2."),
    )
    workspace = tmp_path / "workspace"
    engine = AgentSessionEngine(workspace, FakeProvider())
    artifact = ArtifactStore(workspace).copy_input(engine.session_id, source)
    engine.attach_artifact(artifact.artifact_id)
    first = engine.revisions.require_current(engine.session_id)
    engine.registry.dispatch(
        "set_unit_context",
        {
            "length": "mm",
            "force": "N",
            "stress": "MPa",
            "density": "tonne/mm^3",
            "acceleration": "mm/s^2",
        },
        ToolExecutionContext(engine.session_id, first.revision, "cli_units"),
    )
    second = engine.revisions.require_current(engine.session_id)
    engine.registry.dispatch(
        "set_result_requests",
        {
            "queries": [{"kind": "max_displacement_magnitude"}],
            "export_formats": ["vtk"],
        },
        ToolExecutionContext(engine.session_id, second.revision, "cli_results"),
    )
    output = StringIO()

    cli._render_summary(engine.get_analysis_summary(), output)

    rendered = output.getvalue()
    assert artifact.artifact_id not in rendered
    for label in (
        "- materials:",
        "- sections:",
        "- constraints:",
        "- loads:",
        "- keyword inventory:",
        "- resource class:",
    ):
        assert label in rendered
    assert "Enter /confirm" in rendered


def test_cli_artifact_listing_hides_internal_artifact_ids(tmp_path):
    source = write_perforated_plate_style_inp(
        tmp_path,
        "visible-input.inp",
        ("*Cload", "Set-right, 1, 10."),
    )
    workspace = tmp_path / "workspace"
    engine = AgentSessionEngine(workspace, FakeProvider())
    store = ArtifactStore(workspace)
    artifact = store.copy_input(engine.session_id, source)
    output = StringIO()

    cli._handle_command(
        "/artifacts",
        engine,
        store,
        lambda prompt: "",
        output,
    )

    rendered = output.getvalue()
    assert "visible-input.inp" in rendered
    assert artifact.sha256[:12] in rendered
    assert artifact.artifact_id not in rendered
    assert "inputs/art_" not in rendered


def test_cli_diagnostic_rendering_includes_context_and_next_action():
    output = StringIO()

    cli._render_diagnostic(
        output,
        severity="error",
        code="EXAMPLE_ERROR",
        message="The example failed.",
        entity="solid section:data",
        remediation="Provide one positive thickness value.",
    )

    assert output.getvalue() == (
        "[error] EXAMPLE_ERROR [solid section:data]: The example failed.\n"
        "  next: Provide one positive thickness value.\n"
    )


def test_cli_help_includes_transient_worker_retry():
    assert "/retry" in cli._help_text()
