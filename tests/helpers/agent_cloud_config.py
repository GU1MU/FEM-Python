from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from fem_agent.config import ConfigError, LocalAgentConfig, resolve_local_config


CLOUD_SMOKE_OPT_IN_ENV = "FEM_AGENT_CLOUD_SMOKE"
CLOUD_SMOKE_CONFIG_ENV = "FEM_AGENT_CLOUD_SMOKE_CONFIG"
CLOUD_OPT_IN_REASON = (
    "[cloud-opt-in] set FEM_AGENT_CLOUD_SMOKE=1 and "
    "FEM_AGENT_CLOUD_SMOKE_CONFIG to an absolute external config path"
)


def load_cloud_smoke_config(
    environ: Mapping[str, str],
) -> tuple[LocalAgentConfig | None, str | None]:
    if environ.get(CLOUD_SMOKE_OPT_IN_ENV) != "1":
        return None, CLOUD_OPT_IN_REASON
    raw_path = environ.get(CLOUD_SMOKE_CONFIG_ENV)
    if not isinstance(raw_path, str) or not raw_path.strip():
        return None, CLOUD_OPT_IN_REASON

    path = Path(raw_path)
    if not path.is_absolute():
        raise ConfigError(
            "the cloud smoke config path must be absolute"
        )
    file_config = LocalAgentConfig.load(path)

    resolved = resolve_local_config(file_config, environ=environ)
    if resolved.provider.casefold() != "deepseek":
        raise ConfigError("the cloud smoke test requires provider='deepseek'")
    if not resolved.has_api_key:
        return None, (
            "[cloud-opt-in] configure api_key in the explicit cloud smoke "
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
