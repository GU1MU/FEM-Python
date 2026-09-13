import os

import pytest

from fem_agent.config import ConfigError
from fem_agent.providers.base import AssistantMessage, ToolDefinition
from fem_agent.providers.deepseek import DeepSeekProvider

from tests.helpers.agent_cloud_config import load_cloud_smoke_config


def test_deepseek_tool_call_smoke():
    try:
        config, reason = load_cloud_smoke_config(os.environ)
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
