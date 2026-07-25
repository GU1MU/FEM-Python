"""Deterministic scripted provider for tests and offline CLI evaluation."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

from .base import (
    AssistantMessage,
    ProviderResponse,
    ToolDefinition,
)


@dataclass(frozen=True)
class CapturedRequest:
    messages: tuple[AssistantMessage, ...]
    tools: tuple[ToolDefinition, ...]


ScriptItem = (
    ProviderResponse
    | BaseException
    | Callable[
        [tuple[AssistantMessage, ...], tuple[ToolDefinition, ...]],
        ProviderResponse,
    ]
)


class FakeProvider:
    """Return queued responses while retaining privacy-testable requests."""

    def __init__(
        self,
        script: Iterable[ScriptItem] = (),
        *,
        model: str = "fake-v0",
    ):
        self._script = deque(script)
        self._model = model
        self.requests: list[CapturedRequest] = []

    @property
    def provider_name(self) -> str:
        return "fake"

    @property
    def model_name(self) -> str:
        return self._model

    def queue(self, *items: ScriptItem) -> None:
        self._script.extend(items)

    def complete(
        self,
        messages: Sequence[AssistantMessage],
        tools: Sequence[ToolDefinition],
    ) -> ProviderResponse:
        normalized_messages = tuple(messages)
        normalized_tools = tuple(tools)
        self.requests.append(CapturedRequest(normalized_messages, normalized_tools))
        if not self._script:
            return ProviderResponse(
                AssistantMessage(
                    role="assistant",
                    content="No scripted provider response is available.",
                ),
                finish_reason="stop",
            )
        item = self._script.popleft()
        if isinstance(item, BaseException):
            raise item
        if callable(item):
            item = item(normalized_messages, normalized_tools)
        if not isinstance(item, ProviderResponse):
            raise TypeError("fake provider script items must return ProviderResponse")
        return item


__all__ = ["CapturedRequest", "FakeProvider", "ScriptItem"]
