"""Provider-neutral messages, tool calls, and failure contracts."""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence
from urllib.parse import urlsplit


_ROLES = frozenset({"system", "user", "assistant", "tool"})
_TOOL_IDENTIFIER = re.compile(r"[A-Za-z0-9_-]{1,128}\Z")


@dataclass(frozen=True)
class ToolCall:
    call_id: str
    name: str
    arguments: Mapping[str, Any]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.call_id, str)
            or _TOOL_IDENTIFIER.fullmatch(self.call_id) is None
        ):
            raise ValueError("tool call_id contains unsupported characters")
        if (
            not isinstance(self.name, str)
            or len(self.name) > 64
            or _TOOL_IDENTIFIER.fullmatch(self.name) is None
        ):
            raise ValueError("tool name contains unsupported characters")
        if not isinstance(self.arguments, Mapping):
            raise TypeError("tool arguments must be an object")
        normalized_arguments = dict(self.arguments)
        _validate_json_value(normalized_arguments, "tool arguments")
        object.__setattr__(self, "arguments", normalized_arguments)


@dataclass(frozen=True)
class AssistantMessage:
    role: str
    content: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()
    tool_call_id: str | None = None
    reasoning_content: str | None = None

    def __post_init__(self) -> None:
        if self.role not in _ROLES:
            raise ValueError(f"unsupported message role: {self.role!r}")
        if self.content is not None and not isinstance(self.content, str):
            raise TypeError("message content must be a string or None")
        if self.content is not None:
            _require_utf8(self.content, "message content")
        if self.reasoning_content is not None and not isinstance(
            self.reasoning_content, str
        ):
            raise TypeError("message reasoning_content must be a string or None")
        if self.reasoning_content is not None:
            _require_utf8(self.reasoning_content, "message reasoning_content")
        object.__setattr__(self, "tool_calls", tuple(self.tool_calls))
        if self.role == "tool" and (
            not isinstance(self.tool_call_id, str)
            or _TOOL_IDENTIFIER.fullmatch(self.tool_call_id) is None
        ):
            raise ValueError("tool messages require a valid tool_call_id")
        if self.role != "tool" and self.tool_call_id is not None:
            raise ValueError("only tool messages may contain tool_call_id")
        if self.role != "assistant" and self.tool_calls:
            raise ValueError("only assistant messages may contain tool_calls")
        if self.role != "assistant" and self.reasoning_content is not None:
            raise ValueError("only assistant messages may contain reasoning_content")


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    parameters: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not self.name or len(self.name) > 64:
            raise ValueError("tool name must contain 1 through 64 characters")
        allowed = set(
            "abcdefghijklmnopqrstuvwxyz"
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            "0123456789_-"
        )
        if any(character not in allowed for character in self.name):
            raise ValueError("tool name contains an unsupported character")
        if not self.description or not isinstance(self.description, str):
            raise ValueError("tool description must be a non-empty string")
        _require_utf8(self.description, "tool description")
        if not isinstance(self.parameters, Mapping):
            raise TypeError("tool parameters must be a JSON schema object")
        normalized_parameters = dict(self.parameters)
        _validate_json_value(normalized_parameters, "tool parameters")
        object.__setattr__(self, "parameters", normalized_parameters)


@dataclass(frozen=True)
class ProviderResponse:
    message: AssistantMessage
    finish_reason: str
    usage: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.message.role != "assistant":
            raise ValueError("provider response message must use the assistant role")
        if not self.finish_reason:
            raise ValueError("finish_reason must be non-empty")
        normalized_usage: dict[str, int] = {}
        for key, value in self.usage.items():
            if value is None:
                continue
            if (
                not isinstance(key, str)
                or isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
            ):
                raise ValueError(
                    "provider usage must contain nonnegative integer values"
                )
            normalized_usage[key] = value
        object.__setattr__(self, "usage", normalized_usage)


@dataclass(frozen=True)
class ProviderConfig:
    provider: str = "deepseek"
    model: str = "deepseek-v4-pro"
    base_url: str = "https://api.deepseek.com"
    api_key_env: str = "DEEPSEEK_API_KEY"
    timeout_seconds: float = 60.0
    max_retries: int = 2
    retry_delay_seconds: float = 0.25
    max_output_tokens: int = 8192

    def __post_init__(self) -> None:
        for name in ("provider", "model", "base_url", "api_key_env"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        if self.provider.casefold() == "deepseek":
            parsed = urlsplit(self.base_url)
            if (
                parsed.scheme != "https"
                or parsed.hostname != "api.deepseek.com"
                or parsed.username is not None
                or parsed.password is not None
                or parsed.port not in {None, 443}
                or parsed.path not in {"", "/", "/v1", "/v1/"}
                or parsed.query
                or parsed.fragment
            ):
                raise ValueError(
                    "DeepSeek base_url must use the official HTTPS API endpoint"
                )
        if (
            isinstance(self.timeout_seconds, bool)
            or not isinstance(self.timeout_seconds, (int, float))
            or not math.isfinite(self.timeout_seconds)
            or self.timeout_seconds <= 0
            or self.timeout_seconds > 600
        ):
            raise ValueError(
                "timeout_seconds must be greater than zero and at most 600"
            )
        if (
            isinstance(self.max_retries, bool)
            or not isinstance(self.max_retries, int)
            or self.max_retries < 0
            or self.max_retries > 10
        ):
            raise ValueError(
                "max_retries must be a nonnegative integer at most 10"
            )
        if (
            isinstance(self.retry_delay_seconds, bool)
            or not isinstance(self.retry_delay_seconds, (int, float))
            or not math.isfinite(self.retry_delay_seconds)
            or self.retry_delay_seconds < 0
            or self.retry_delay_seconds > 60
        ):
            raise ValueError(
                "retry_delay_seconds must be between zero and 60"
            )
        if (
            isinstance(self.max_output_tokens, bool)
            or not isinstance(self.max_output_tokens, int)
            or self.max_output_tokens <= 0
        ):
            raise ValueError("max_output_tokens must be a positive integer")

    @classmethod
    def from_env(
        cls,
        *,
        provider: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        api_key_env: str | None = None,
        timeout_seconds: float | None = None,
        max_retries: int | None = None,
        max_output_tokens: int | None = None,
        environ: Mapping[str, str] | None = None,
    ) -> "ProviderConfig":
        values = os.environ if environ is None else environ
        return cls(
            provider=provider or values.get("FEM_AGENT_PROVIDER", "deepseek"),
            model=model or values.get("DEEPSEEK_MODEL", "deepseek-v4-pro"),
            base_url=base_url
            or values.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            api_key_env=api_key_env or "DEEPSEEK_API_KEY",
            timeout_seconds=(
                timeout_seconds
                if timeout_seconds is not None
                else float(values.get("FEM_AGENT_PROVIDER_TIMEOUT", "60"))
            ),
            max_retries=(
                max_retries
                if max_retries is not None
                else int(values.get("FEM_AGENT_PROVIDER_RETRIES", "2"))
            ),
            max_output_tokens=(
                max_output_tokens
                if max_output_tokens is not None
                else int(values.get("FEM_AGENT_MAX_OUTPUT_TOKENS", "8192"))
            ),
        )


class CloudModelProvider(Protocol):
    """Replaceable synchronous provider boundary used by the V0 engine."""

    @property
    def provider_name(self) -> str: ...

    @property
    def model_name(self) -> str: ...

    def complete(
        self,
        messages: Sequence[AssistantMessage],
        tools: Sequence[ToolDefinition],
    ) -> ProviderResponse: ...


class ProviderError(RuntimeError):
    """Base class for normalized provider failures."""


class ProviderAuthenticationError(ProviderError):
    pass


class ProviderCredentialMissingError(ProviderAuthenticationError):
    pass


class ProviderPaymentRequiredError(ProviderError):
    pass


class ProviderRateLimitError(ProviderError):
    pass


class ProviderTimeoutError(ProviderError):
    pass


class ProviderUnavailableError(ProviderError):
    pass


class ProviderMalformedResponseError(ProviderError):
    pass


def _validate_json_value(
    value: Any,
    name: str,
    *,
    depth: int = 0,
) -> None:
    if depth > 64:
        raise ValueError(f"{name} exceeds the maximum JSON nesting depth")
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{name} object keys must be strings")
            _require_utf8(key, f"{name} object key")
            _validate_json_value(item, name, depth=depth + 1)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _validate_json_value(item, name, depth=depth + 1)
        return
    if isinstance(value, str):
        _require_utf8(value, name)
        return
    if value is None or isinstance(value, (bool, int)):
        return
    if isinstance(value, float) and math.isfinite(value):
        return
    raise ValueError(f"{name} must contain finite JSON-compatible values")


def _require_utf8(value: str, name: str) -> None:
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as error:
        raise ValueError(f"{name} must contain valid Unicode text") from error


__all__ = [
    "AssistantMessage",
    "CloudModelProvider",
    "ProviderAuthenticationError",
    "ProviderConfig",
    "ProviderCredentialMissingError",
    "ProviderError",
    "ProviderMalformedResponseError",
    "ProviderPaymentRequiredError",
    "ProviderRateLimitError",
    "ProviderResponse",
    "ProviderTimeoutError",
    "ProviderUnavailableError",
    "ToolCall",
    "ToolDefinition",
]
