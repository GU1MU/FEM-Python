"""A privacy-safe Provider request capture for Phase 0 evidence.

Unlike ``FakeProvider``, this helper intentionally retains no messages or
tool objects.  Each call keeps only redacted system-context strings, the
published tool names, and stable hashes of the published schemas.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import hashlib
import json
import re
from typing import Iterable, Mapping, Sequence

from fem_agent.providers.base import (
    AssistantMessage,
    ProviderResponse,
    ToolDefinition,
)


_WINDOWS_PATH = re.compile(r"(?i)\b[A-Z]:\\[^\s\"']+")
_UNIX_PATH = re.compile(r"(?<![\w])/(?:[^\s\"']+/)+[^\s\"']*")
_CREDENTIAL = re.compile(
    r"(?ix)(?:api[_ -]?key|authorization|bearer|access[_ -]?token|"
    r"auth[_ -]?token|password|secret|credential)\s*[:=]\s*[^\s,;]+"
)
_UUID = re.compile(
    r"\b[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-"
    r"[89ab][0-9a-f]{3}-[0-9a-f]{12}\b",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class CapturedProviderRequest:
    """Bounded evidence retained for one Provider completion round."""

    system_context: tuple[str, ...]
    tool_names: tuple[str, ...]
    tool_schema_hashes: tuple[tuple[str, str], ...]

    @property
    def schema_hashes(self) -> dict[str, str]:
        """Return a convenient copy without exposing mutable capture state."""

        return dict(self.tool_schema_hashes)

    def to_dict(self) -> dict[str, object]:
        return {
            "system_context": list(self.system_context),
            "tool_names": list(self.tool_names),
            "tool_schema_hashes": dict(self.tool_schema_hashes),
        }


def tool_schema_hash(tool: ToolDefinition) -> str:
    """Hash one complete tool definition with stable UTF-8 JSON."""

    payload = {
        "description": tool.description,
        "name": tool.name,
        "parameters": tool.parameters,
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _redact_system_text(value: str) -> str:
    text = _CREDENTIAL.sub("<credential-redacted>", value)
    text = _WINDOWS_PATH.sub("<path-redacted>", text)
    text = _UNIX_PATH.sub("<path-redacted>", text)
    return _UUID.sub("<id-redacted>", text)


def _safe_system_context(messages: Sequence[AssistantMessage]) -> tuple[str, ...]:
    captured: list[str] = []
    for message in messages:
        if message.role != "system" or message.content is None:
            continue
        text = _redact_system_text(message.content)
        prefix = "Current local state (structured metadata only):"
        if text.startswith(prefix):
            raw = text[len(prefix) :].strip()
            try:
                state = json.loads(raw)
            except (TypeError, ValueError):
                pass
            else:
                if isinstance(state, Mapping):
                    state = dict(state)
                    state["session_id"] = "<session-redacted>"
                    state["active_run_id"] = None
                    text = prefix + " " + json.dumps(
                        state,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
        captured.append(text)
    return tuple(captured)


def capture_request(
    messages: Sequence[AssistantMessage],
    tools: Sequence[ToolDefinition],
) -> CapturedProviderRequest:
    """Project one Provider call into its privacy-safe baseline record."""

    normalized_tools = tuple(tools)
    return CapturedProviderRequest(
        system_context=_safe_system_context(messages),
        tool_names=tuple(tool.name for tool in normalized_tools),
        tool_schema_hashes=tuple(
            (tool.name, tool_schema_hash(tool)) for tool in normalized_tools
        ),
    )


class RequestCaptureProvider:
    """Deterministic Provider double retaining only bounded request evidence."""

    def __init__(
        self,
        responses: Iterable[ProviderResponse | BaseException] = (),
        *,
        model: str = "baseline-capture",
    ) -> None:
        self._responses = deque(responses)
        self._model = model
        self.requests: list[CapturedProviderRequest] = []

    @property
    def provider_name(self) -> str:
        return "capture"

    @property
    def model_name(self) -> str:
        return self._model

    def queue(self, *responses: ProviderResponse | BaseException) -> None:
        self._responses.extend(responses)

    def complete(
        self,
        messages: Sequence[AssistantMessage],
        tools: Sequence[ToolDefinition],
    ) -> ProviderResponse:
        self.requests.append(capture_request(messages, tools))
        if not self._responses:
            return ProviderResponse(
                AssistantMessage("assistant", content=""),
                finish_reason="stop",
            )
        response = self._responses.popleft()
        if isinstance(response, BaseException):
            raise response
        return response


__all__ = [
    "CapturedProviderRequest",
    "RequestCaptureProvider",
    "capture_request",
    "tool_schema_hash",
]
