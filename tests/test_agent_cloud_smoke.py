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


TEST_CONFIG_PATH = Path(__file__).with_name(TEST_CONFIG_NAME)


def _cloud_smoke_config(
    path: Path,
    environ: Mapping[str, str],
) -> tuple[LocalAgentConfig | None, str | None]:
    file_config = (
        LocalAgentConfig.load(path)
        if path.is_file()
        else LocalAgentConfig(
            model="deepseek-v4-flash",
            timeout_seconds=30,
            max_retries=0,
            max_output_tokens=256,
        )
    )
    opted_in = (
        environ.get("FEM_AGENT_CLOUD_SMOKE") == "1"
        or (path.is_file() and file_config.enabled)
    )
    if not opted_in:
        return None, (
            f"set enabled=true in {path.name} or "
            "FEM_AGENT_CLOUD_SMOKE=1 to enable the paid cloud smoke test"
        )

    resolved = resolve_local_config(file_config, environ=environ)
    if resolved.provider.casefold() != "deepseek":
        raise ConfigError("the cloud smoke test requires provider='deepseek'")
    if not resolved.has_api_key:
        return None, (
            f"set api_key in {path.name} or DEEPSEEK_API_KEY for the "
            "cloud smoke test"
        )
    return (
        LocalAgentConfig(
            provider="deepseek",
            model=resolved.model,
            base_url=resolved.base_url,
            api_key=resolved.api_key,
            workspace=resolved.workspace,
            timeout_seconds=min(resolved.timeout_seconds, 30),
            max_retries=0,
            max_output_tokens=min(resolved.max_output_tokens, 256),
            enabled=True,
        ),
        None,
    )


def test_cloud_smoke_config_is_disabled_without_explicit_opt_in(tmp_path):
    config, reason = _cloud_smoke_config(
        tmp_path / TEST_CONFIG_NAME,
        {},
    )

    assert config is None
    assert "enable the paid cloud smoke test" in reason


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

    config, reason = _cloud_smoke_config(path, {})

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
        _cloud_smoke_config(path, {})


@pytest.mark.integration
def test_opt_in_deepseek_tool_call_smoke():
    try:
        config, reason = _cloud_smoke_config(TEST_CONFIG_PATH, os.environ)
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
