"""Bounded conversation persistence and provider-history serialization."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .artifacts import ArtifactStore, atomic_write_json, read_json_file, safe_child
from .providers.base import AssistantMessage, ToolCall


_CONVERSATION_SCHEMA_VERSION = 1


class ConversationStorageLimit(ValueError):
    """Raised when one indivisible conversation turn exceeds its byte cap."""


class ConversationStore:
    """Persist histories while leaving active-turn state with the engine."""

    def __init__(
        self,
        artifacts: ArtifactStore,
        *,
        max_messages: int,
        max_storage_bytes: int,
    ) -> None:
        self.artifacts = artifacts
        self.max_messages = max_messages
        self.max_storage_bytes = max_storage_bytes

    def path(self, session_id: str) -> Path:
        session = self.artifacts.session_path(session_id)
        return safe_child(session, "conversation.json")

    def load(self, session_id: str) -> list[AssistantMessage]:
        path = self.path(session_id)
        if not path.exists():
            return []
        payload = read_json_file(
            path,
            max_bytes=self.max_storage_bytes,
        )
        if set(payload) != {"schema_version", "messages"}:
            raise ValueError("conversation storage has invalid fields")
        if payload["schema_version"] != _CONVERSATION_SCHEMA_VERSION:
            raise ValueError("conversation storage has an unsupported version")
        raw_messages = payload["messages"]
        if not isinstance(raw_messages, list):
            raise ValueError("conversation messages must be an array")
        return [_message_from_dict(item) for item in raw_messages]

    def append(
        self,
        session_id: str,
        history: Sequence[AssistantMessage],
        message: AssistantMessage,
    ) -> list[AssistantMessage]:
        """Return new history only after its complete turns are persisted."""

        retained = [*history, message]
        while len(retained) > self.max_messages * 2:
            trimmed = _drop_oldest_conversation_turn(retained)
            if trimmed is None:
                break
            retained = trimmed
        payload = _conversation_payload(retained)
        while json_storage_size_bytes(payload) > self.max_storage_bytes:
            trimmed = _drop_oldest_conversation_turn(retained)
            if trimmed is None:
                raise ConversationStorageLimit(
                    "the current conversation turn exceeds its storage limit"
                )
            retained = trimmed
            payload = _conversation_payload(retained)
        path = self.path(session_id)
        atomic_write_json(path, payload, overwrite=path.exists())
        return retained


def _conversation_payload(history: Sequence[AssistantMessage]) -> dict[str, Any]:
    return {
        "schema_version": _CONVERSATION_SCHEMA_VERSION,
        "messages": [_message_to_dict(item) for item in history],
    }


def _drop_oldest_conversation_turn(
    history: list[AssistantMessage],
) -> list[AssistantMessage] | None:
    if len(history) <= 1:
        return None
    next_user = next(
        (
            index
            for index, message in enumerate(history[1:], start=1)
            if message.role == "user"
        ),
        None,
    )
    return None if next_user is None else history[next_user:]


def _message_to_dict(message: AssistantMessage) -> dict[str, Any]:
    return {
        "role": message.role,
        "content": message.content,
        "reasoning_content": message.reasoning_content,
        "tool_call_id": message.tool_call_id,
        "tool_calls": [
            {
                "call_id": item.call_id,
                "name": item.name,
                "arguments": dict(item.arguments),
            }
            for item in message.tool_calls
        ],
    }


def _message_from_dict(value: Any) -> AssistantMessage:
    if not isinstance(value, Mapping):
        raise ValueError("conversation message must be an object")
    required = {"role", "content", "tool_call_id", "tool_calls", "reasoning_content"}
    field_names = frozenset(value)
    if field_names != required or not isinstance(
        value["tool_calls"], list
    ):
        raise ValueError("conversation message has invalid fields")
    return AssistantMessage(
        role=value["role"],
        content=value["content"],
        tool_call_id=value["tool_call_id"],
        reasoning_content=value["reasoning_content"],
        tool_calls=tuple(
            ToolCall(
                item["call_id"],
                item["name"],
                item["arguments"],
            )
            for item in value["tool_calls"]
            if isinstance(item, Mapping)
            and set(item) == {"call_id", "name", "arguments"}
        ),
    )


def complete_provider_history(
    messages: Sequence[AssistantMessage],
) -> list[AssistantMessage]:
    """Drop orphaned or incomplete tool-call groups after an interrupted turn."""

    complete: list[AssistantMessage] = []
    index = 0
    while index < len(messages):
        message = messages[index]
        if message.role == "tool":
            index += 1
            continue
        if message.role != "assistant" or not message.tool_calls:
            complete.append(message)
            index += 1
            continue
        expected_ids = {call.call_id for call in message.tool_calls}
        tool_messages: list[AssistantMessage] = []
        seen_ids: set[str] = set()
        cursor = index + 1
        while cursor < len(messages) and messages[cursor].role == "tool":
            tool_message = messages[cursor]
            if (
                tool_message.tool_call_id in expected_ids
                and tool_message.tool_call_id not in seen_ids
            ):
                tool_messages.append(tool_message)
                seen_ids.add(str(tool_message.tool_call_id))
            cursor += 1
        if seen_ids == expected_ids:
            complete.append(message)
            complete.extend(tool_messages)
        index = cursor
    return complete


def json_storage_size_bytes(payload: Mapping[str, Any]) -> int:
    return (
        len(
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        + 1
    )
