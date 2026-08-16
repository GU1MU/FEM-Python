import os
from collections.abc import Mapping
from pathlib import Path

import pytest

from fem_agent.config import (
    ConfigError,
    LocalAgentConfig,
    TEST_CONFIG_NAME,
    resolve_local_config,
)
from fem_agent.providers.base import (
    AssistantMessage,
    ToolDefinition,
)
from fem_agent.providers.deepseek import DeepSeekProvider


from tests.helpers.agent_session_fixtures import (
    CLOUD_SMOKE_CONFIG_ENV,
    CLOUD_SMOKE_OPT_IN_ENV,
    _CLOUD_OPT_IN_REASON,
    _cloud_smoke_config,
)


@pytest.mark.parametrize(
    "environ",
    (
        {},
        {CLOUD_SMOKE_OPT_IN_ENV: "1"},
        {CLOUD_SMOKE_CONFIG_ENV: "ignored.json"},
    ),
)
def test_cloud_smoke_config_requires_both_explicit_gates_without_reading_config(
    monkeypatch,
    environ,
):
    def fail_if_loaded(_cls, _path):
        raise AssertionError("cloud config must not be read before both gates")

    monkeypatch.setattr(
        LocalAgentConfig,
        "load",
        classmethod(fail_if_loaded),
    )
    config, reason = _cloud_smoke_config(environ)

    assert config is None
    assert reason == _CLOUD_OPT_IN_REASON


def test_cloud_smoke_config_caps_cost_and_keeps_key_out_of_environment(
    tmp_path,
):
    original_environment_key = os.environ.get("DEEPSEEK_API_KEY")
    path = tmp_path / TEST_CONFIG_NAME
    path.write_text(
        """
{
  "enabled": true,
  "provider": "deepseek",
  "model": "deepseek-v4-flash",
  "base_url": "https://api.deepseek.com",
  "api_key": "local-test-secret",
  "timeout_seconds": 120,
  "max_retries": 10,
  "max_output_tokens": 4096
}
""".strip(),
        encoding="utf-8",
    )

    config, reason = _cloud_smoke_config({
        CLOUD_SMOKE_OPT_IN_ENV: "1",
        CLOUD_SMOKE_CONFIG_ENV: str(path),
    })

    assert reason is None
    assert config is not None
    assert config.model == "deepseek-v4-flash"
    assert config.timeout_seconds == 30
    assert config.max_retries == 0
    assert config.max_output_tokens == 256
    assert os.environ.get("DEEPSEEK_API_KEY") == original_environment_key
    assert "local-test-secret" not in repr(config)


def test_cloud_smoke_config_rejects_nonofficial_endpoint_before_network(
    tmp_path,
):
    path = tmp_path / TEST_CONFIG_NAME
    path.write_text(
        """
{
  "enabled": true,
  "base_url": "https://example.invalid",
  "api_key": "local-test-secret"
}
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="official HTTPS API endpoint"):
        _cloud_smoke_config({
            CLOUD_SMOKE_OPT_IN_ENV: "1",
            CLOUD_SMOKE_CONFIG_ENV: str(path),
        })


def test_cloud_smoke_config_requires_an_absolute_external_path():
    with pytest.raises(ConfigError, match="must be absolute"):
        _cloud_smoke_config({
            CLOUD_SMOKE_OPT_IN_ENV: "1",
            CLOUD_SMOKE_CONFIG_ENV: TEST_CONFIG_NAME,
        })


def test_cloud_smoke_diagnostics_do_not_echo_config_values(tmp_path):
    secret = "cloud-smoke-secret-value"
    path = tmp_path / TEST_CONFIG_NAME
    path.write_text(
        f"""
{{
  "provider": "{secret}",
  "api_key": "{secret}"
}}
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ConfigError) as captured:
        _cloud_smoke_config({
            CLOUD_SMOKE_OPT_IN_ENV: "1",
            CLOUD_SMOKE_CONFIG_ENV: str(path),
        })

    failure_output = f"invalid cloud smoke configuration: {captured.value}"
    assert secret not in failure_output
    assert secret not in _CLOUD_OPT_IN_REASON


def test_cloud_smoke_skip_reason_does_not_echo_config_values(tmp_path):
    secret = "cloud-smoke-model-secret"
    path = tmp_path / TEST_CONFIG_NAME
    path.write_text(
        f"""
{{
  "provider": "deepseek",
  "model": "{secret}"
}}
""".strip(),
        encoding="utf-8",
    )

    config, reason = _cloud_smoke_config({
        CLOUD_SMOKE_OPT_IN_ENV: "1",
        CLOUD_SMOKE_CONFIG_ENV: str(path),
    })

    assert config is None
    assert reason is not None
    assert reason.startswith("[cloud-opt-in]")
    assert secret not in reason


@pytest.mark.cloud
@pytest.mark.integration
def test_opt_in_deepseek_tool_call_smoke():
    try:
        config, reason = _cloud_smoke_config(os.environ)
    except ConfigError as error:
        pytest.fail(f"invalid cloud smoke configuration: {error}")
    if config is None:
        pytest.skip(reason)

    provider = DeepSeekProvider(
        config.provider_config(),
        environ=config.provider_environment({}),
    )
    tool = ToolDefinition(
        "show_capabilities",
        "Return the supported FEM Agent V0 capabilities.",
        {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
    )

    response = provider.complete(
        [
            AssistantMessage(
                "system",
                "Call the supplied show_capabilities tool exactly once.",
            ),
            AssistantMessage("user", "Show the available local capabilities."),
        ],
        [tool],
    )

    assert len(response.message.tool_calls) == 1
    call = response.message.tool_calls[0]
    assert call.name == "show_capabilities"
    assert call.arguments == {}
