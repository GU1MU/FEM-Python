import json
from pathlib import Path

import pytest

import fem_agent.config as config_module
from fem_agent.config import (
    ConfigError,
    LocalAgentConfig,
    MAX_CONFIG_BYTES,
    ROOT_CONFIG_NAME,
    TEST_CONFIG_NAME,
    create_main_config_template,
    find_main_config,
    resolve_local_config,
)


def _write_config(path: Path, document: dict) -> Path:
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def _mark_project_root(path: Path) -> None:
    (path / "src" / "fem_agent").mkdir(parents=True, exist_ok=True)
    (path / "pyproject.toml").write_text(
        "[project]\nname = \"fem-project\"\n",
        encoding="utf-8",
    )


def test_config_names_are_stable():
    assert ROOT_CONFIG_NAME == "fem-agent.config.json"
    assert TEST_CONFIG_NAME == "fem-agent.test.config.json"


def test_create_main_config_template_is_loadable_and_has_an_empty_key(tmp_path):
    path = tmp_path / ROOT_CONFIG_NAME

    assert create_main_config_template(path)

    document = json.loads(path.read_text(encoding="utf-8"))
    config = LocalAgentConfig.load(path)
    assert document == {
        "provider": "deepseek",
        "model": "deepseek-v4-pro",
        "base_url": "https://api.deepseek.com",
        "api_key": "",
        "workspace": "agent-workspace",
        "timeout_seconds": 60,
        "max_retries": 2,
        "max_output_tokens": 8192,
    }
    assert not config.has_api_key
    assert config.workspace == (tmp_path / "agent-workspace").resolve()
    assert path.read_bytes().endswith(b"\n")


def test_create_main_config_template_never_overwrites_an_existing_file(
    tmp_path,
):
    path = _write_config(
        tmp_path / ROOT_CONFIG_NAME,
        {"api_key": "keep-this-value"},
    )
    original = path.read_bytes()

    assert not create_main_config_template(path)
    assert path.read_bytes() == original


def test_create_main_config_template_does_not_create_missing_parents(tmp_path):
    parent = tmp_path / "mistyped" / "directory"
    path = parent / ROOT_CONFIG_NAME

    with pytest.raises(ConfigError, match="parent directory"):
        create_main_config_template(path)

    assert not parent.exists()


def test_create_main_config_template_keeps_its_file_after_a_write_error(
    monkeypatch,
    tmp_path,
):
    path = tmp_path / ROOT_CONFIG_NAME

    def fail_to_sync(descriptor):
        raise OSError("simulated sync failure")

    monkeypatch.setattr(config_module.os, "fsync", fail_to_sync)
    with pytest.raises(ConfigError, match="could not be created"):
        create_main_config_template(path)

    assert path.exists()
    assert json.loads(path.read_text(encoding="utf-8"))["api_key"] == ""


def test_secret_bearing_config_paths_are_precisely_gitignored():
    project_root = Path(__file__).resolve().parents[1]
    patterns = set(
        (project_root / ".gitignore").read_text(encoding="utf-8").splitlines()
    )

    assert f"/{ROOT_CONFIG_NAME}" in patterns
    assert f"/tests/{TEST_CONFIG_NAME}" in patterns
    assert "*.config.json" not in patterns


def test_loads_config_and_resolves_workspace_relative_to_config(tmp_path):
    path = _write_config(
        tmp_path / ROOT_CONFIG_NAME,
        {
            "provider": "deepseek",
            "model": "deepseek-v4-flash",
            "base_url": "https://api.deepseek.com/v1",
            "api_key": "test-secret",
            "workspace": "private/workspace",
            "timeout_seconds": 12.5,
            "max_retries": 0,
            "max_output_tokens": 256,
            "enabled": True,
        },
    )

    config = LocalAgentConfig.load(path)

    assert config.provider == "deepseek"
    assert config.model == "deepseek-v4-flash"
    assert config.base_url == "https://api.deepseek.com/v1"
    assert config.workspace == (tmp_path / "private/workspace").resolve()
    assert config.timeout_seconds == 12.5
    assert config.max_retries == 0
    assert config.max_output_tokens == 256
    assert config.enabled is True
    assert config.has_api_key
    assert "test-secret" not in repr(config)


def test_missing_and_empty_api_keys_are_detectable_without_leaking_values(tmp_path):
    missing = LocalAgentConfig.load(
        _write_config(tmp_path / ROOT_CONFIG_NAME, {})
    )
    empty = LocalAgentConfig(api_key="  ")

    assert not missing.has_api_key
    assert not empty.has_api_key
    with pytest.raises(ConfigError, match="missing or empty") as captured:
        empty.require_api_key()
    assert "  " not in str(captured.value)


def test_provider_config_and_environment_keep_secret_separate():
    config = LocalAgentConfig(
        api_key="secret-value",
        timeout_seconds=7,
        max_retries=0,
        max_output_tokens=128,
    )

    provider = config.provider_config()
    environment = config.provider_environment({"OTHER": "kept"})

    assert not hasattr(provider, "api_key")
    assert provider.timeout_seconds == 7
    assert provider.max_retries == 0
    assert provider.max_output_tokens == 128
    assert environment == {
        "OTHER": "kept",
        "DEEPSEEK_API_KEY": "secret-value",
    }


@pytest.mark.parametrize(
    "raw",
    (
        "[]",
        '{"model":"a","model":"b"}',
        '{"unknown":1}',
        '{"timeout_seconds":true}',
        '{"max_retries":1.5}',
        '{"max_output_tokens":false}',
        '{"enabled":1}',
        '{"workspace":""}',
        '{"provider":null}',
        '{"timeout_seconds":NaN}',
    ),
)
def test_rejects_non_object_duplicate_unknown_or_wrong_typed_json(tmp_path, raw):
    path = tmp_path / ROOT_CONFIG_NAME
    path.write_text(raw, encoding="utf-8")

    with pytest.raises(ConfigError):
        LocalAgentConfig.load(path)


def test_rejects_invalid_utf8_and_oversized_files(tmp_path):
    invalid = tmp_path / "invalid.json"
    invalid.write_bytes(b'{"api_key":"\xff"}')
    with pytest.raises(ConfigError, match="UTF-8"):
        LocalAgentConfig.load(invalid)

    oversized = tmp_path / "oversized.json"
    oversized.write_bytes(b" " * (MAX_CONFIG_BYTES + 1))
    with pytest.raises(ConfigError, match="64 KiB"):
        LocalAgentConfig.load(oversized)


def test_validation_errors_do_not_include_api_key(tmp_path):
    secret = "super-secret-do-not-print"
    path = _write_config(
        tmp_path / ROOT_CONFIG_NAME,
        {
            "api_key": secret,
            "base_url": "http://example.invalid",
        },
    )

    with pytest.raises(ConfigError) as captured:
        LocalAgentConfig.load(path)

    assert secret not in str(captured.value)


def test_find_main_config_searches_cwd_ancestors_first(tmp_path):
    cwd_root = tmp_path / "cwd-root"
    nested = cwd_root / "one" / "two"
    nested.mkdir(parents=True)
    _mark_project_root(cwd_root)
    config = _write_config(cwd_root / ROOT_CONFIG_NAME, {})
    module_root = tmp_path / "module-root"
    module_path = module_root / "src" / "fem_agent" / "config.py"
    _mark_project_root(module_root)
    _write_config(module_root / ROOT_CONFIG_NAME, {})

    assert find_main_config(nested, module_path=module_path) == config


def test_find_main_config_uses_module_ancestor_for_double_click_layout(tmp_path):
    project = tmp_path / "project"
    script = project / ".venv" / "Scripts" / "fem-agent.exe"
    script.parent.mkdir(parents=True)
    _mark_project_root(project)
    config = _write_config(project / ROOT_CONFIG_NAME, {})
    unrelated_cwd = Path(tmp_path.anchor) / "fem-agent-unrelated-cwd-does-not-exist"

    assert find_main_config(unrelated_cwd, module_path=script) == config


def test_find_main_config_does_not_accept_unrelated_ancestor_config(
    monkeypatch,
    tmp_path,
):
    unrelated = tmp_path / "unrelated"
    working_directory = unrelated / "nested"
    working_directory.mkdir(parents=True)
    _write_config(unrelated / ROOT_CONFIG_NAME, {"api_key": "wrong-account"})

    project = tmp_path / "project"
    _mark_project_root(project)
    module_path = project / "src" / "fem_agent" / "config.py"
    expected = _write_config(
        project / ROOT_CONFIG_NAME,
        {"api_key": "expected-account"},
    )

    def identify_test_project(start):
        return project if start == project or project in start.parents else None

    monkeypatch.setattr(
        config_module,
        "_find_project_root",
        identify_test_project,
    )
    assert (
        find_main_config(working_directory, module_path=module_path)
        == expected
    )


def test_missing_anchored_project_config_does_not_fall_back_to_cwd_config(
    monkeypatch,
    tmp_path,
):
    unrelated_cwd = tmp_path / "outside"
    unrelated_cwd.mkdir()
    _write_config(
        unrelated_cwd / ROOT_CONFIG_NAME,
        {"api_key": "wrong-account"},
    )

    project = tmp_path / "project"
    _mark_project_root(project)
    module_path = project / "src" / "fem_agent" / "config.py"
    executable = project / ".venv" / "Scripts" / "python.exe"
    executable.parent.mkdir(parents=True)
    monkeypatch.setattr(config_module.sys, "argv", [str(executable)])
    monkeypatch.setattr(config_module.sys, "executable", str(executable))

    def identify_test_project(start):
        return project if start == project or project in start.parents else None

    monkeypatch.setattr(
        config_module,
        "_find_project_root",
        identify_test_project,
    )
    assert find_main_config(unrelated_cwd, module_path=module_path) is None


def test_resolve_precedence_is_cli_then_env_then_file_then_defaults(tmp_path):
    file_config = LocalAgentConfig(
        provider="file-provider",
        model="file-model",
        base_url="https://file.example",
        api_key="file-key",
        workspace=tmp_path / "file-workspace",
        timeout_seconds=10,
        max_retries=1,
        max_output_tokens=100,
        enabled=False,
    )
    env = {
        "FEM_AGENT_PROVIDER": "env-provider",
        "DEEPSEEK_MODEL": "env-model",
        "DEEPSEEK_BASE_URL": "https://env.example",
        "DEEPSEEK_API_KEY": "env-key",
        "FEM_AGENT_WORKSPACE": "env-workspace",
        "FEM_AGENT_PROVIDER_TIMEOUT": "20",
        "FEM_AGENT_PROVIDER_RETRIES": "2",
        "FEM_AGENT_MAX_OUTPUT_TOKENS": "200",
        "FEM_AGENT_ENABLED": "true",
    }

    from_env = resolve_local_config(file_config, environ=env)
    resolved = resolve_local_config(
        file_config,
        provider="cli-provider",
        model="cli-model",
        base_url="https://cli.example",
        api_key="cli-key",
        workspace=tmp_path / "cli-workspace",
        timeout_seconds=30,
        max_retries=3,
        max_output_tokens=300,
        enabled=False,
        environ=env,
    )

    assert from_env.provider == "env-provider"
    assert from_env.model == "env-model"
    assert from_env.api_key == "env-key"
    assert from_env.workspace == Path("env-workspace")
    assert from_env.timeout_seconds == 20
    assert from_env.max_retries == 2
    assert from_env.max_output_tokens == 200
    assert from_env.enabled is True
    assert resolved.provider == "cli-provider"
    assert resolved.model == "cli-model"
    assert resolved.base_url == "https://cli.example"
    assert resolved.api_key == "cli-key"
    assert resolved.workspace == tmp_path / "cli-workspace"
    assert resolved.timeout_seconds == 30
    assert resolved.max_retries == 3
    assert resolved.max_output_tokens == 300
    assert resolved.enabled is False


def test_resolve_uses_file_then_builtin_defaults(tmp_path):
    file_config = LocalAgentConfig(
        model="file-model",
        base_url="https://api.deepseek.com",
        workspace=tmp_path,
    )

    from_file = resolve_local_config(file_config, environ={})
    defaults = resolve_local_config(environ={})

    assert from_file.model == "file-model"
    assert from_file.workspace == tmp_path
    assert defaults.provider == "deepseek"
    assert defaults.model == "deepseek-v4-pro"
    assert defaults.base_url == "https://api.deepseek.com"
    assert defaults.workspace == Path("agent-workspace")
    assert defaults.max_output_tokens == 8192
    assert defaults.enabled is False


@pytest.mark.parametrize(
    ("name", "value"),
    (
        ("FEM_AGENT_PROVIDER_TIMEOUT", "nan"),
        ("FEM_AGENT_PROVIDER_RETRIES", "1.5"),
        ("FEM_AGENT_MAX_OUTPUT_TOKENS", "false"),
        ("FEM_AGENT_ENABLED", "sometimes"),
    ),
)
def test_rejects_invalid_environment_overrides_without_echoing_values(name, value):
    with pytest.raises(ConfigError) as captured:
        resolve_local_config(environ={name: value})

    assert value not in str(captured.value)
