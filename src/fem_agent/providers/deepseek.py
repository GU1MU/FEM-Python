"""DeepSeek adapter using its official OpenAI-compatible API surface."""

from __future__ import annotations

import json
import os
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from types import SimpleNamespace
from typing import Any

from .base import (
    AssistantMessage,
    ProviderAuthenticationError,
    ProviderConfig,
    ProviderCredentialMissingError,
    ProviderMalformedResponseError,
    ProviderPaymentRequiredError,
    ProviderRateLimitError,
    ProviderResponse,
    ProviderTimeoutError,
    ProviderUnavailableError,
    ToolCall,
    ToolDefinition,
)

_MAX_TOOL_ARGUMENTS_BYTES = 64 * 1024


class DeepSeekProvider:
    """Normalize DeepSeek Chat Completions into the provider-neutral contract."""

    def __init__(
        self,
        config: ProviderConfig | None = None,
        *,
        client: Any | None = None,
        environ: Mapping[str, str] | None = None,
        sleep: Callable[[float], None] | None = None,
    ):
        self.config = config or ProviderConfig()
        if self.config.provider.casefold() != "deepseek":
            raise ValueError("DeepSeekProvider requires provider='deepseek'")
        self._environ = os.environ if environ is None else environ
        self._sleep = sleep
        self._client = client
        self._owns_client = client is None
        self._client_lock = threading.RLock()
        self._cancel_event = threading.Event()
        self._inflight_thread: threading.Thread | None = None

    @property
    def provider_name(self) -> str:
        return "deepseek"

    @property
    def model_name(self) -> str:
        return self.config.model

    def complete(
        self,
        messages: Sequence[AssistantMessage],
        tools: Sequence[ToolDefinition],
    ) -> ProviderResponse:
        if self._cancel_event.is_set():
            raise ProviderUnavailableError("The DeepSeek request was cancelled.")
        attempts = self.config.max_retries + 1
        last_error: Exception | None = None
        for attempt in range(attempts):
            if self._cancel_event.is_set():
                raise ProviderUnavailableError(
                    "The DeepSeek request was cancelled."
            )
            try:
                client = self._client or self._create_client()
                request: dict[str, Any] = {
                    "model": self.config.model,
                    "messages": [
                        _message_payload(message) for message in messages
                    ],
                    "stream": False,
                    "max_tokens": self.config.max_output_tokens,
                    "extra_body": {"thinking": {"type": "disabled"}},
                }
                if tools:
                    request["tools"] = [_tool_payload(tool) for tool in tools]
                    request["tool_choice"] = "auto"
                response = self._request_with_deadline(
                    client,
                    request,
                )
                if self._cancel_event.is_set():
                    raise ProviderUnavailableError(
                        "The DeepSeek request was cancelled."
                    )
                try:
                    return _normalize_response(response)
                except (
                    ProviderMalformedResponseError,
                    ProviderUnavailableError,
                ):
                    raise
                except (TypeError, ValueError, OverflowError) as error:
                    raise ProviderMalformedResponseError(
                        "DeepSeek returned a malformed response."
                    ) from error
            except (
                ProviderAuthenticationError,
                ProviderPaymentRequiredError,
                ProviderMalformedResponseError,
            ):
                raise
            except Exception as error:
                if self._cancel_event.is_set():
                    raise ProviderUnavailableError(
                        "The DeepSeek request was cancelled."
                    ) from error
                normalized, retryable = _normalize_sdk_error(error)
                last_error = normalized
                if not retryable or attempt + 1 >= attempts:
                    raise normalized from error
                delay = min(
                    self.config.retry_delay_seconds * (2**attempt),
                    60.0,
                )
                if delay:
                    self._wait_before_retry(delay)
        raise ProviderUnavailableError(str(last_error or "DeepSeek request failed"))

    def complete_stream(
        self,
        messages: Sequence[AssistantMessage],
        tools: Sequence[ToolDefinition],
        on_text_delta: Callable[[str], None],
    ) -> ProviderResponse:
        """Stream visible assistant text while retaining one normalized response."""

        if not callable(on_text_delta):
            raise TypeError("on_text_delta must be callable")
        if self._cancel_event.is_set():
            raise ProviderUnavailableError("The DeepSeek request was cancelled.")
        attempts = self.config.max_retries + 1
        last_error: Exception | None = None
        emitted_text = False

        def emit_text(delta: str) -> None:
            nonlocal emitted_text
            emitted_text = True
            on_text_delta(delta)

        for attempt in range(attempts):
            if self._cancel_event.is_set():
                raise ProviderUnavailableError(
                    "The DeepSeek request was cancelled."
                )
            try:
                client = self._client or self._create_client()
                request: dict[str, Any] = {
                    "model": self.config.model,
                    "messages": [
                        _message_payload(message) for message in messages
                    ],
                    "stream": True,
                    "max_tokens": self.config.max_output_tokens,
                    "extra_body": {"thinking": {"type": "disabled"}},
                }
                if tools:
                    request["tools"] = [_tool_payload(tool) for tool in tools]
                    request["tool_choice"] = "auto"
                return self._request_stream_with_deadline(
                    client,
                    request,
                    emit_text,
                )
            except (
                ProviderAuthenticationError,
                ProviderPaymentRequiredError,
                ProviderMalformedResponseError,
            ):
                raise
            except Exception as error:
                if self._cancel_event.is_set():
                    raise ProviderUnavailableError(
                        "The DeepSeek request was cancelled."
                    ) from error
                normalized, retryable = _normalize_sdk_error(error)
                last_error = normalized
                if (
                    emitted_text
                    or not retryable
                    or attempt + 1 >= attempts
                ):
                    raise normalized from error
                delay = min(
                    self.config.retry_delay_seconds * (2**attempt),
                    60.0,
                )
                if delay:
                    self._wait_before_retry(delay)
        raise ProviderUnavailableError(str(last_error or "DeepSeek request failed"))

    def _create_client(self) -> Any:
        with self._client_lock:
            if self._client is not None:
                return self._client
            key = self._environ.get(self.config.api_key_env)
            if not key:
                raise ProviderCredentialMissingError(
                    f"missing DeepSeek credential in {self.config.api_key_env}"
                )
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise ProviderUnavailableError(
                    "the 'openai' package is required; install the agent extra"
                ) from exc
            self._client = OpenAI(
                api_key=key,
                base_url=self.config.base_url,
                timeout=self.config.timeout_seconds,
                max_retries=0,
            )
            return self._client

    def reset_cancellation(self) -> None:
        """Prepare the adapter for a new engine-owned request."""

        self._cancel_event.clear()

    def cancel_active_request(self) -> None:
        """Best-effort interruption of the active synchronous HTTP request."""

        self._cancel_event.set()
        self._close_client(self._client)

    def _close_client(self, client: Any | None) -> None:
        if client is None:
            return
        with self._client_lock:
            close = getattr(client, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass
            if self._owns_client and self._client is client:
                self._client = None

    def _request_with_deadline(
        self,
        client: Any,
        request: Mapping[str, Any],
    ) -> Any:
        done = threading.Event()
        outcome: dict[str, Any] = {}

        def invoke() -> None:
            try:
                outcome["response"] = client.chat.completions.create(
                    **dict(request),
                )
            except Exception as error:
                outcome["error"] = error
            finally:
                done.set()

        thread = threading.Thread(
            target=invoke,
            name="fem-agent-deepseek-request",
            daemon=True,
        )
        with self._client_lock:
            previous = self._inflight_thread
            if previous is not None and previous.is_alive():
                raise ProviderUnavailableError(
                    "A previous DeepSeek request is still stopping."
                )
            self._inflight_thread = thread
        try:
            thread.start()
        except Exception:
            with self._client_lock:
                if self._inflight_thread is thread:
                    self._inflight_thread = None
            raise
        deadline = time.monotonic() + self.config.timeout_seconds
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self._close_client(client)
                if done.wait(0.1):
                    self._clear_inflight_thread(thread)
                raise ProviderTimeoutError(
                    "The DeepSeek request exceeded its wall-clock deadline."
                )
            if done.wait(min(remaining, 0.05)):
                break
            if self._cancel_event.is_set():
                self._close_client(client)
                if done.wait(0.1):
                    self._clear_inflight_thread(thread)
                raise ProviderUnavailableError(
                    "The DeepSeek request was cancelled."
                )
        self._clear_inflight_thread(thread)
        if self._cancel_event.is_set():
            self._close_client(client)
            raise ProviderUnavailableError(
                "The DeepSeek request was cancelled."
            )
        error = outcome.get("error")
        if isinstance(error, Exception):
            raise error
        if "response" not in outcome:
            raise ProviderUnavailableError(
                "The DeepSeek request ended without a response."
            )
        return outcome["response"]

    def _request_stream_with_deadline(
        self,
        client: Any,
        request: Mapping[str, Any],
        on_text_delta: Callable[[str], None],
    ) -> ProviderResponse:
        done = threading.Event()
        aborted = threading.Event()
        outcome: dict[str, Any] = {}

        def invoke() -> None:
            try:
                stream = client.chat.completions.create(**dict(request))
                outcome["response"] = _normalize_stream_response(
                    stream,
                    on_text_delta,
                    lambda: (
                        aborted.is_set()
                        or self._cancel_event.is_set()
                    ),
                )
            except Exception as error:
                outcome["error"] = error
            finally:
                done.set()

        thread = threading.Thread(
            target=invoke,
            name="fem-agent-deepseek-stream",
            daemon=True,
        )
        with self._client_lock:
            previous = self._inflight_thread
            if previous is not None and previous.is_alive():
                raise ProviderUnavailableError(
                    "A previous DeepSeek request is still stopping."
                )
            self._inflight_thread = thread
        try:
            thread.start()
        except Exception:
            with self._client_lock:
                if self._inflight_thread is thread:
                    self._inflight_thread = None
            raise
        deadline = time.monotonic() + self.config.timeout_seconds
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                aborted.set()
                self._close_client(client)
                if done.wait(0.1):
                    self._clear_inflight_thread(thread)
                raise ProviderTimeoutError(
                    "The DeepSeek stream exceeded its wall-clock deadline."
                )
            if done.wait(min(remaining, 0.05)):
                break
            if self._cancel_event.is_set():
                aborted.set()
                self._close_client(client)
                if done.wait(0.1):
                    self._clear_inflight_thread(thread)
                raise ProviderUnavailableError(
                    "The DeepSeek request was cancelled."
                )
        self._clear_inflight_thread(thread)
        if self._cancel_event.is_set():
            self._close_client(client)
            raise ProviderUnavailableError(
                "The DeepSeek request was cancelled."
            )
        error = outcome.get("error")
        if isinstance(error, Exception):
            raise error
        response = outcome.get("response")
        if not isinstance(response, ProviderResponse):
            raise ProviderUnavailableError(
                "The DeepSeek stream ended without a response."
            )
        return response

    def _clear_inflight_thread(self, thread: threading.Thread) -> None:
        with self._client_lock:
            if self._inflight_thread is thread and not thread.is_alive():
                self._inflight_thread = None

    def _wait_before_retry(self, delay: float) -> None:
        if self._sleep is None:
            cancelled = self._cancel_event.wait(delay)
        else:
            self._sleep(delay)
            cancelled = self._cancel_event.is_set()
        if cancelled:
            raise ProviderUnavailableError(
                "The DeepSeek request was cancelled during retry backoff."
            )

    def contains_configured_credential(self, text: str) -> bool:
        """Detect the configured key without exposing it to engine state."""

        key = self._environ.get(self.config.api_key_env)
        return bool(key and key.strip() and key in text)


def _message_payload(message: AssistantMessage) -> dict[str, Any]:
    payload: dict[str, Any] = {"role": message.role, "content": message.content}
    if message.role == "tool":
        payload["tool_call_id"] = message.tool_call_id
    if message.tool_calls:
        payload["tool_calls"] = [
            {
                "id": item.call_id,
                "type": "function",
                "function": {
                    "name": item.name,
                    "arguments": json.dumps(
                        item.arguments,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                },
            }
            for item in message.tool_calls
        ]
    return payload


def _tool_payload(tool: ToolDefinition) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
        },
    }


def _normalize_response(response: Any) -> ProviderResponse:
    choices = getattr(response, "choices", None)
    if not choices:
        raise ProviderMalformedResponseError("DeepSeek response contained no choices")
    choice = choices[0]
    finish_reason = str(getattr(choice, "finish_reason", "") or "")
    if finish_reason == "insufficient_system_resource":
        raise ProviderUnavailableError(
            "DeepSeek reported insufficient system resources"
        )
    if not finish_reason:
        raise ProviderMalformedResponseError(
            "DeepSeek response omitted finish_reason"
        )
    if finish_reason not in {"stop", "tool_calls"}:
        raise ProviderMalformedResponseError(
            f"DeepSeek response ended with unsupported reason {finish_reason!r}"
        )
    raw_message = getattr(choice, "message", None)
    if raw_message is None:
        raise ProviderMalformedResponseError("DeepSeek response omitted message")
    calls: list[ToolCall] = []
    seen_ids: set[str] = set()
    for raw_call in getattr(raw_message, "tool_calls", None) or ():
        call_id = str(getattr(raw_call, "id", "") or "")
        function = getattr(raw_call, "function", None)
        name = str(getattr(function, "name", "") or "")
        raw_arguments = getattr(function, "arguments", None)
        if not call_id or not name or not isinstance(raw_arguments, str):
            raise ProviderMalformedResponseError(
                "DeepSeek returned an incomplete tool call"
            )
        try:
            argument_size = len(raw_arguments.encode("utf-8"))
        except UnicodeEncodeError as exc:
            raise ProviderMalformedResponseError(
                "DeepSeek returned invalid Unicode tool arguments"
            ) from exc
        if argument_size > _MAX_TOOL_ARGUMENTS_BYTES:
            raise ProviderMalformedResponseError(
                "DeepSeek tool arguments exceeded the local size limit"
            )
        if call_id in seen_ids:
            raise ProviderMalformedResponseError(
                "DeepSeek returned a duplicate tool call identifier"
            )
        seen_ids.add(call_id)
        try:
            arguments = json.loads(
                raw_arguments,
                parse_constant=_reject_json_constant,
            )
        except (json.JSONDecodeError, RecursionError, ValueError) as exc:
            raise ProviderMalformedResponseError(
                "DeepSeek returned invalid JSON tool arguments"
            ) from exc
        if not isinstance(arguments, dict):
            raise ProviderMalformedResponseError(
                "DeepSeek tool arguments must be a JSON object"
            )
        calls.append(ToolCall(call_id, name, arguments))
    content = getattr(raw_message, "content", None)
    if content is not None and not isinstance(content, str):
        content = str(content)
    if content is not None:
        try:
            content.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise ProviderMalformedResponseError(
                "DeepSeek returned invalid Unicode text"
            ) from exc
    if content is None and not calls:
        raise ProviderMalformedResponseError(
            "DeepSeek response contained neither text nor tool calls"
        )
    if finish_reason == "tool_calls" and not calls:
        raise ProviderMalformedResponseError(
            "DeepSeek reported tool_calls without any tool calls"
        )
    if finish_reason == "stop" and calls:
        raise ProviderMalformedResponseError(
            "DeepSeek returned tool calls with an inconsistent finish reason"
        )
    usage = _usage_mapping(getattr(response, "usage", None))
    return ProviderResponse(
        AssistantMessage(
            role="assistant",
            content=content,
            tool_calls=tuple(calls),
        ),
        finish_reason=finish_reason,
        usage=usage,
    )


def _normalize_stream_response(
    stream: Any,
    on_text_delta: Callable[[str], None],
    cancelled: Callable[[], bool],
) -> ProviderResponse:
    content_parts: list[str] = []
    tool_parts: dict[int, dict[str, str]] = {}
    finish_reason = ""
    usage: Any = None
    saw_choice = False

    for chunk in stream:
        if cancelled():
            raise ProviderUnavailableError(
                "The DeepSeek request was cancelled."
            )
        chunk_usage = getattr(chunk, "usage", None)
        if chunk_usage is not None:
            usage = chunk_usage
        choices = getattr(chunk, "choices", None) or ()
        if not choices:
            continue
        saw_choice = True
        choice = choices[0]
        current_finish = str(
            getattr(choice, "finish_reason", "") or ""
        )
        if current_finish:
            finish_reason = current_finish
        delta = getattr(choice, "delta", None)
        if delta is None:
            continue
        content = getattr(delta, "content", None)
        if content is not None:
            if not isinstance(content, str):
                content = str(content)
            try:
                content.encode("utf-8")
            except UnicodeEncodeError as error:
                raise ProviderMalformedResponseError(
                    "DeepSeek returned invalid Unicode text"
                ) from error
            if content:
                content_parts.append(content)
                on_text_delta(content)
        for raw_call in getattr(delta, "tool_calls", None) or ():
            index = getattr(raw_call, "index", None)
            if isinstance(index, bool) or not isinstance(index, int) or index < 0:
                raise ProviderMalformedResponseError(
                    "DeepSeek streamed an invalid tool call index"
                )
            parts = tool_parts.setdefault(
                index,
                {"id": "", "name": "", "arguments": ""},
            )
            call_id = getattr(raw_call, "id", None)
            if call_id:
                parts["id"] += str(call_id)
            function = getattr(raw_call, "function", None)
            name = None if function is None else getattr(function, "name", None)
            arguments = (
                None
                if function is None
                else getattr(function, "arguments", None)
            )
            if name:
                parts["name"] += str(name)
            if arguments:
                parts["arguments"] += str(arguments)

    if not saw_choice:
        raise ProviderMalformedResponseError(
            "DeepSeek stream contained no choices"
        )
    raw_calls = [
        SimpleNamespace(
            id=parts["id"],
            function=SimpleNamespace(
                name=parts["name"],
                arguments=parts["arguments"],
            ),
        )
        for _index, parts in sorted(tool_parts.items())
    ]
    raw_response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason=finish_reason,
                message=SimpleNamespace(
                    content="".join(content_parts) or None,
                    tool_calls=raw_calls,
                ),
            )
        ],
        usage=usage,
    )
    return _normalize_response(raw_response)


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value!r}")


def _usage_mapping(usage: Any) -> dict[str, int]:
    if usage is None:
        return {}
    result: dict[str, int] = {}
    for name in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = getattr(usage, name, None)
        if value is not None:
            result[name] = int(value)
    return result


def _normalize_sdk_error(error: Exception) -> tuple[RuntimeError, bool]:
    if isinstance(
        error,
        (
            ProviderAuthenticationError,
            ProviderPaymentRequiredError,
            ProviderRateLimitError,
            ProviderTimeoutError,
            ProviderUnavailableError,
            ProviderMalformedResponseError,
        ),
    ):
        retryable = isinstance(
            error,
            (
                ProviderRateLimitError,
                ProviderUnavailableError,
            ),
        )
        return error, retryable
    try:
        import openai
    except ImportError:
        return ProviderUnavailableError(
            f"The DeepSeek request failed ({type(error).__name__})."
        ), False

    if isinstance(error, openai.AuthenticationError):
        return ProviderAuthenticationError(
            "DeepSeek authentication failed."
        ), False
    if isinstance(error, openai.RateLimitError):
        return ProviderRateLimitError(
            "DeepSeek rate-limited the request."
        ), True
    if isinstance(error, openai.APITimeoutError):
        return ProviderTimeoutError("The DeepSeek request timed out."), True
    if isinstance(error, openai.APIConnectionError):
        return ProviderUnavailableError(
            "The DeepSeek API could not be reached."
        ), True
    if isinstance(error, openai.APIStatusError):
        status = int(getattr(error, "status_code", 0) or 0)
        if status in {401, 403}:
            return ProviderAuthenticationError(
                "DeepSeek authentication failed."
            ), False
        if status == 402:
            return ProviderPaymentRequiredError(
                "DeepSeek reported insufficient account balance."
            ), False
        if status == 429:
            return ProviderRateLimitError(
                "DeepSeek rate-limited the request."
            ), True
        if status >= 500:
            return ProviderUnavailableError(
                f"DeepSeek is unavailable (HTTP {status})."
            ), True
        return ProviderMalformedResponseError(
            f"DeepSeek rejected the request (HTTP {status or 'unknown'})."
        ), False
    return ProviderUnavailableError(
        f"The DeepSeek request failed ({type(error).__name__})."
    ), False


__all__ = ["DeepSeekProvider"]
