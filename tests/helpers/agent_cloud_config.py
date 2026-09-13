from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from fem_agent.config import ConfigError, LocalAgentConfig, ROOT_CONFIG_NAME, resolve_local_config


CLOUD_SMOKE_ENV = "FEM_AGENT_CLOUD_SMOKE"
CLOUD_SMOKE_CONFIG_ENV = "FEM_AGENT_CLOUD_SMOKE_CONFIG"
DEFAULT_CLOUD_CONFIG_PATH = Path(__file__).resolve().parents[2] / ROOT_CONFIG_NAME


def load_cloud_smoke_config(
    environ: Mapping[str, str],
) -> tuple[LocalAgentConfig | None, str | None]:
    if environ.get(CLOUD_SMOKE_ENV) == "0":
        return None, "[cloud-config] disabled by FEM_AGENT_CLOUD_SMOKE=0"
    raw_path = environ.get(CLOUD_SMOKE_CONFIG_ENV)
    if raw_path is None:
        path = DEFAULT_CLOUD_CONFIG_PATH
        if not path.is_file():
            return None, "[cloud-config] local Agent configuration is missing"
    else:
        path = Path(raw_path)
        if not path.is_absolute():
            raise ConfigError("the cloud smoke config path must be absolute")
    file_config = LocalAgentConfig.load(path)

    resolved = resolve_local_config(file_config, environ=environ)
    if resolved.provider.casefold() != "deepseek":
        raise ConfigError("the cloud smoke test requires provider='deepseek'")
    if not resolved.has_api_key:
        return None, (
            "[cloud-config] configure api_key in the local Agent "
            "config or set DEEPSEEK_API_KEY"
        )
    return (
        LocalAgentConfig(
            provider="deepseek",
            model=resolved.model,
            base_url=resolved.base_url,
            api_key=resolved.api_key,
            timeout_seconds=min(resolved.timeout_seconds, 30),
            max_retries=0,
            max_output_tokens=min(resolved.max_output_tokens, 256),
            enabled=True,
        ),
        None,
    )
