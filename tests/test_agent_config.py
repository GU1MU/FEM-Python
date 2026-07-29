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


def test_secret_bearing_config_paths_are_precisely_gitignored():
    project_root = Path(__file__).resolve().parents[1]
    patterns = set(
        (project_root / ".gitignore").read_text(encoding="utf-8").splitlines()
    )

    assert f"/{ROOT_CONFIG_NAME}" in patterns
    assert f"/tests/{TEST_CONFIG_NAME}" in patterns
    assert "*.config.json" not in patterns


def test_loads_gui_agent_provider_config(tmp_path):
    path = _write_config(
        tmp_path / ROOT_CONFIG_NAME,
        {
            "provider": "deepseek",
            "model": "deepseek-v4-flash",
            "base_url": "https://api.deepseek.com/v1",
            "api_key": "test-secret",
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
        '{"workspace":"retired-cli-workspace"}',
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


def test_find_main_config_uses_module_ancestor_for_gui_entry_point(tmp_path):
    project = tmp_path / "project"
    script = project / ".venv" / "Scripts" / "fem-gui.exe"
    script.parent.mkdir(parents=True)
    _mark_project_root(project)
    config = _write_config(project / ROOT_CONFIG_NAME, {})
    unrelated_cwd = Path(tmp_path.anchor) / "fem-gui-unrelated-cwd-does-not-exist"

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


def test_resolve_precedence_is_environment_then_file_then_defaults():
    file_config = LocalAgentConfig(
        provider="file-provider",
        model="file-model",
        base_url="https://file.example",
        api_key="file-key",
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
        "FEM_AGENT_PROVIDER_TIMEOUT": "20",
        "FEM_AGENT_PROVIDER_RETRIES": "2",
        "FEM_AGENT_MAX_OUTPUT_TOKENS": "200",
        "FEM_AGENT_ENABLED": "true",
    }

    resolved = resolve_local_config(
        file_config,
        environ=env,
    )

    assert resolved.provider == "env-provider"
    assert resolved.model == "env-model"
    assert resolved.base_url == "https://env.example"
    assert resolved.api_key == "env-key"
    assert resolved.timeout_seconds == 20
    assert resolved.max_retries == 2
    assert resolved.max_output_tokens == 200
    assert resolved.enabled is True


def test_resolve_uses_file_then_builtin_defaults():
    file_config = LocalAgentConfig(
        model="file-model",
        base_url="https://api.deepseek.com",
    )

    from_file = resolve_local_config(file_config, environ={})
    defaults = resolve_local_config(environ={})

    assert from_file.model == "file-model"
    assert defaults.provider == "deepseek"
    assert defaults.model == "deepseek-v4-pro"
    assert defaults.base_url == "https://api.deepseek.com"
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
