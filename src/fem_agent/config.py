"""Strict local configuration loading for the GUI Agent runtime."""

from __future__ import annotations

import json
import math
import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .providers.base import ProviderConfig


ROOT_CONFIG_NAME = "fem-agent.config.json"
TEST_CONFIG_NAME = "fem-agent.test.config.json"
MAX_CONFIG_BYTES = 64 * 1024

_CONFIG_FIELDS = frozenset(
    {
        "provider",
        "model",
        "base_url",
        "api_key",
        "timeout_seconds",
        "max_retries",
        "max_output_tokens",
        "enabled",
    }
)


class ConfigError(ValueError):
    """A local configuration file or override is invalid."""


@dataclass(frozen=True)
class LocalAgentConfig:
    """Resolved local settings.

    The literal credential deliberately remains outside ``ProviderConfig`` so
    its value cannot leak through provider validation errors or its repr.
    """

    provider: str = "deepseek"
    model: str = "deepseek-v4-pro"
    base_url: str = "https://api.deepseek.com"
    api_key: str | None = field(default=None, repr=False)
    timeout_seconds: float = 60.0
    max_retries: int = 2
    max_output_tokens: int = 8192
    enabled: bool = False

    def __post_init__(self) -> None:
        for name in ("provider", "model", "base_url"):
            if type(getattr(self, name)) is not str:
                raise ConfigError(f"{name} must be a string")
        if self.api_key is not None and type(self.api_key) is not str:
            raise ConfigError("api_key must be a string when provided")
        if (
            type(self.timeout_seconds) not in {int, float}
            or not math.isfinite(self.timeout_seconds)
        ):
            raise ConfigError("timeout_seconds must be a finite number")
        if type(self.max_retries) is not int:
            raise ConfigError("max_retries must be an integer")
        if type(self.max_output_tokens) is not int:
            raise ConfigError("max_output_tokens must be an integer")
        if type(self.enabled) is not bool:
            raise ConfigError("enabled must be a boolean")
        if self.api_key is not None:
            try:
                self.api_key.encode("utf-8")
            except UnicodeEncodeError as error:
                raise ConfigError("api_key must contain valid Unicode text") from error

        # Keep provider-specific validation in one place. The credential is
        # intentionally not passed to this constructor.
        try:
            ProviderConfig(
                provider=self.provider,
                model=self.model,
                base_url=self.base_url,
                timeout_seconds=self.timeout_seconds,
                max_retries=self.max_retries,
                max_output_tokens=self.max_output_tokens,
            )
        except (TypeError, ValueError) as error:
            raise ConfigError(str(error)) from error

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> "LocalAgentConfig":
        """Load one bounded, strict UTF-8 JSON object from *path*."""

        config_path = Path(path)
        try:
            with config_path.open("rb") as stream:
                raw = stream.read(MAX_CONFIG_BYTES + 1)
        except FileNotFoundError as error:
            raise ConfigError("configuration file was not found") from error
        except OSError as error:
            raise ConfigError("configuration file could not be read") from error
        if len(raw) > MAX_CONFIG_BYTES:
            raise ConfigError("configuration file exceeds the 64 KiB limit")
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ConfigError("configuration file must use strict UTF-8") from error
        try:
            document = json.loads(
                text,
                object_pairs_hook=_strict_object,
                parse_constant=_reject_nonfinite_constant,
            )
        except ConfigError:
            raise
        except (json.JSONDecodeError, RecursionError, ValueError) as error:
            raise ConfigError("configuration file contains invalid JSON") from error
        if type(document) is not dict:
            raise ConfigError("configuration file must contain a JSON object")
        if any(key not in _CONFIG_FIELDS for key in document):
            raise ConfigError("configuration file contains an unknown field")

        values = _validate_document_fields(document)
        return cls(**values)

    @property
    def has_api_key(self) -> bool:
        """Whether a non-empty credential is available, without exposing it."""

        return self.api_key is not None and bool(self.api_key.strip())

    def require_api_key(self) -> None:
        """Raise a secret-safe error when the selected provider needs a key."""

        if self.provider.casefold() == "deepseek" and not self.has_api_key:
            raise ConfigError(
                "api_key is missing or empty in the FEM Agent configuration"
            )

    def provider_config(self) -> ProviderConfig:
        """Build the credential-free provider settings."""

        return ProviderConfig(
            provider=self.provider,
            model=self.model,
            base_url=self.base_url,
            timeout_seconds=self.timeout_seconds,
            max_retries=self.max_retries,
            max_output_tokens=self.max_output_tokens,
        )

    def provider_environment(
        self,
        environ: Mapping[str, str] | None = None,
    ) -> dict[str, str]:
        """Return an environment overlay containing the resolved credential."""

        result = dict(os.environ if environ is None else environ)
        if self.api_key is None:
            result.pop("DEEPSEEK_API_KEY", None)
        else:
            result["DEEPSEEK_API_KEY"] = self.api_key
        return result


def find_main_config(
    start: str | os.PathLike[str] | None = None,
    *,
    module_path: str | os.PathLike[str] | None = None,
) -> Path | None:
    """Find the config at an anchored project root or the exact cwd.

    A project root must contain both ``pyproject.toml`` and ``src/fem_agent``.
    This keeps an unrelated ancestor config from silently selecting a
    different API account. Module and executable fallbacks keep an editable
    install usable when ``.venv/Scripts/fem-gui.exe`` is launched from an
    unrelated working directory.
    """

    start_directory = _resolved_directory(
        Path.cwd() if start is None else Path(start)
    )
    start_project = _find_project_root(start_directory)
    if start_project is not None:
        candidate = start_project / ROOT_CONFIG_NAME
        return candidate if candidate.is_file() else None

    search_roots: list[Path] = []
    module = Path(__file__) if module_path is None else Path(module_path)
    search_roots.append(module.parent)
    try:
        search_roots.append(Path(sys.argv[0]).resolve().parent)
    except (OSError, RuntimeError):
        pass
    try:
        search_roots.append(Path(sys.executable).resolve().parent)
    except (OSError, RuntimeError):
        pass

    visited: set[Path] = set()
    anchored_project_seen = False
    for root in search_roots:
        project_root = _find_project_root(_resolved_directory(root))
        if project_root is None or project_root in visited:
            continue
        anchored_project_seen = True
        visited.add(project_root)
        candidate = project_root / ROOT_CONFIG_NAME
        if candidate.is_file():
            return candidate
    if anchored_project_seen:
        return None

    # A non-project installation may still be configured from the exact
    # working directory. Do not scan its ancestors.
    candidate = start_directory / ROOT_CONFIG_NAME
    return candidate if candidate.is_file() else None


def _resolved_directory(path: Path) -> Path:
    try:
        return path.resolve()
    except (OSError, RuntimeError):
        return path.absolute()


def _find_project_root(start: Path) -> Path | None:
    for directory in (start, *start.parents):
        if (
            (directory / "pyproject.toml").is_file()
            and (directory / "src" / "fem_agent").is_dir()
        ):
            return directory
    return None


def resolve_local_config(
    file_config: LocalAgentConfig | None = None,
    *,
    environ: Mapping[str, str] | None = None,
) -> LocalAgentConfig:
    """Resolve environment, file, and built-in values in that order."""

    source = file_config or LocalAgentConfig()
    env = os.environ if environ is None else environ

    return LocalAgentConfig(
        provider=_environment_override(
            env,
            "FEM_AGENT_PROVIDER",
            source.provider,
        ),
        model=_environment_override(env, "DEEPSEEK_MODEL", source.model),
        base_url=_environment_override(
            env,
            "DEEPSEEK_BASE_URL",
            source.base_url,
        ),
        api_key=_environment_override(
            env,
            "DEEPSEEK_API_KEY",
            source.api_key,
        ),
        timeout_seconds=_numeric_override(
            env,
            "FEM_AGENT_PROVIDER_TIMEOUT",
            source.timeout_seconds,
            float,
        ),
        max_retries=_numeric_override(
            env,
            "FEM_AGENT_PROVIDER_RETRIES",
            source.max_retries,
            int,
        ),
        max_output_tokens=_numeric_override(
            env,
            "FEM_AGENT_MAX_OUTPUT_TOKENS",
            source.max_output_tokens,
            int,
        ),
        enabled=_boolean_override(
            env,
            "FEM_AGENT_ENABLED",
            source.enabled,
        ),
    )


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ConfigError("configuration file contains a duplicate field")
        result[key] = value
    return result


def _reject_nonfinite_constant(_value: str) -> None:
    raise ConfigError("configuration file contains a non-finite number")


def _validate_document_fields(document: dict[str, Any]) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for name in ("provider", "model", "base_url", "api_key"):
        if name not in document:
            continue
        value = document[name]
        if type(value) is not str:
            raise ConfigError(f"{name} must be a string")
        values[name] = value
    if "timeout_seconds" in document:
        value = document["timeout_seconds"]
        if type(value) not in {int, float}:
            raise ConfigError("timeout_seconds must be a number")
        values["timeout_seconds"] = value
    for name in ("max_retries", "max_output_tokens"):
        if name not in document:
            continue
        value = document[name]
        if type(value) is not int:
            raise ConfigError(f"{name} must be an integer")
        values[name] = value
    if "enabled" in document:
        value = document["enabled"]
        if type(value) is not bool:
            raise ConfigError("enabled must be a boolean")
        values["enabled"] = value
    return values


def _environment_override(
    environ: Mapping[str, str],
    env_name: str,
    file_value: str | None,
) -> str | None:
    if env_name in environ:
        value = environ[env_name]
        if type(value) is not str:
            raise ConfigError(f"{env_name} must contain text")
        return value
    return file_value


def _numeric_override(
    environ: Mapping[str, str],
    env_name: str,
    file_value: int | float,
    converter: type[int] | type[float],
) -> int | float:
    if env_name not in environ:
        return file_value
    value = environ[env_name]
    if type(value) is not str:
        raise ConfigError(f"{env_name} must contain text")
    try:
        return converter(value)
    except (TypeError, ValueError, OverflowError) as error:
        kind = "an integer" if converter is int else "a number"
        raise ConfigError(f"{env_name} must contain {kind}") from error


def _boolean_override(
    environ: Mapping[str, str],
    env_name: str,
    file_value: bool,
) -> bool:
    if env_name not in environ:
        return file_value
    value = environ[env_name]
    if type(value) is not str:
        raise ConfigError(f"{env_name} must contain text")
    normalized = value.strip().casefold()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ConfigError(f"{env_name} must contain a boolean")


__all__ = [
    "ConfigError",
    "LocalAgentConfig",
    "MAX_CONFIG_BYTES",
    "ROOT_CONFIG_NAME",
    "TEST_CONFIG_NAME",
    "find_main_config",
    "resolve_local_config",
]
