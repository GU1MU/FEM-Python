import threading
import time
from types import SimpleNamespace

import pytest

from fem_agent.providers.base import (
    AssistantMessage,
    ProviderAuthenticationError,
    ProviderConfig,
    ProviderMalformedResponseError,
    ProviderTimeoutError,
    ProviderUnavailableError,
    ToolCall,
    ToolDefinition,
)
from fem_agent.providers.deepseek import DeepSeekProvider


class _CompletionsSpy:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


def _client(spy):
    return SimpleNamespace(chat=SimpleNamespace(completions=spy))


def _response(*, content="done", tool_calls=(), finish_reason="stop"):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason=finish_reason,
                message=SimpleNamespace(content=content, tool_calls=tool_calls),
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
        ),
    )


def _tool_call(arguments='{"session_id":"ses_test"}'):
    return SimpleNamespace(
        id="call_1",
        function=SimpleNamespace(name="get_analysis_summary", arguments=arguments),
    )


def _tool_definition():
    return ToolDefinition(
        name="get_analysis_summary",
        description="Return a bounded deterministic summary.",
        parameters={
            "type": "object",
            "properties": {"session_id": {"type": "string"}},
            "required": ["session_id"],
            "additionalProperties": False,
        },
    )


def test_deepseek_request_uses_openai_compatibility_and_disables_thinking():
    spy = _CompletionsSpy([_response()])
    provider = DeepSeekProvider(
        ProviderConfig(max_retries=0),
        client=_client(spy),
        environ={},
    )

    result = provider.complete(
        [AssistantMessage("user", "inspect the attached model")],
        [_tool_definition()],
    )

    assert result.message.content == "done"
    call = spy.calls[0]
    assert call["model"] == "deepseek-v4-pro"
    assert call["extra_body"] == {"thinking": {"type": "disabled"}}
    assert call["stream"] is False
    assert call["tools"][0]["function"]["name"] == "get_analysis_summary"


def test_provider_contract_rejects_non_json_tool_values_and_bad_tool_ids():
    with pytest.raises(ValueError, match="finite JSON"):
        ToolCall("call_1", "show_capabilities", {"value": float("nan")})

    with pytest.raises(ValueError, match="tool_call_id"):
        AssistantMessage("tool", "{}", tool_call_id="bad id")

    with pytest.raises(ValueError, match="valid Unicode"):
        ToolCall("call_1", "show_capabilities", {"value": "\ud800"})


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("timeout_seconds", float("nan")),
        ("timeout_seconds", float("inf")),
        ("retry_delay_seconds", float("nan")),
        ("retry_delay_seconds", float("inf")),
    ),
)
def test_provider_config_rejects_nonfinite_timing_values(field, value):
    with pytest.raises(ValueError):
        ProviderConfig(**{field: value})


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("timeout_seconds", 601),
        ("max_retries", 11),
        ("retry_delay_seconds", 61),
    ),
)
def test_provider_config_rejects_unbounded_retry_settings(field, value):
    with pytest.raises(ValueError):
        ProviderConfig(**{field: value})


def test_deepseek_normalizes_tool_calls_without_requiring_text():
    spy = _CompletionsSpy(
        [_response(content=None, tool_calls=[_tool_call()], finish_reason="tool_calls")]
    )
    provider = DeepSeekProvider(
        ProviderConfig(max_retries=0),
        client=_client(spy),
    )

    result = provider.complete([AssistantMessage("user", "status")], [_tool_definition()])

    assert result.message.content is None
    assert result.message.tool_calls[0].name == "get_analysis_summary"
    assert result.message.tool_calls[0].arguments == {"session_id": "ses_test"}


def test_deepseek_omits_tool_fields_when_no_tools_are_available():
    spy = _CompletionsSpy([_response()])
    provider = DeepSeekProvider(
        ProviderConfig(max_retries=0),
        client=_client(spy),
    )

    provider.complete([AssistantMessage("user", "hello")], [])

    assert "tools" not in spy.calls[0]
    assert "tool_choice" not in spy.calls[0]


def test_deepseek_rejects_malformed_tool_arguments():
    spy = _CompletionsSpy(
        [_response(content=None, tool_calls=[_tool_call("{broken")], finish_reason="tool_calls")]
    )
    provider = DeepSeekProvider(
        ProviderConfig(max_retries=0),
        client=_client(spy),
    )

    with pytest.raises(ProviderMalformedResponseError, match="invalid JSON"):
        provider.complete([AssistantMessage("user", "status")], [_tool_definition()])


@pytest.mark.parametrize(
    "arguments",
    (
        '{"value":NaN}',
        '{"value":"' + ("x" * (64 * 1024)) + '"}',
    ),
    ids=("nonfinite", "oversized"),
)
def test_deepseek_rejects_nonfinite_or_oversized_tool_arguments(arguments):
    spy = _CompletionsSpy(
        [
            _response(
                content=None,
                tool_calls=[_tool_call(arguments)],
                finish_reason="tool_calls",
            )
        ]
    )
    provider = DeepSeekProvider(
        ProviderConfig(max_retries=0),
        client=_client(spy),
    )

    with pytest.raises(ProviderMalformedResponseError):
        provider.complete(
            [AssistantMessage("user", "status")],
            [_tool_definition()],
        )


def test_deepseek_rejects_decoded_surrogate_tool_arguments():
    spy = _CompletionsSpy(
        [
            _response(
                content=None,
                tool_calls=[_tool_call('{"value":"\\ud800"}')],
                finish_reason="tool_calls",
            )
        ]
    )
    provider = DeepSeekProvider(
        ProviderConfig(max_retries=0),
        client=_client(spy),
    )

    with pytest.raises(ProviderMalformedResponseError):
        provider.complete(
            [AssistantMessage("user", "status")],
            [_tool_definition()],
        )


def test_deepseek_classifies_invalid_tool_identifiers_as_malformed():
    raw_call = _tool_call()
    raw_call.id = "call id with spaces"
    spy = _CompletionsSpy(
        [
            _response(
                content=None,
                tool_calls=[raw_call],
                finish_reason="tool_calls",
            )
        ]
    )
    provider = DeepSeekProvider(
        ProviderConfig(max_retries=0),
        client=_client(spy),
    )

    with pytest.raises(ProviderMalformedResponseError, match="malformed"):
        provider.complete(
            [AssistantMessage("user", "status")],
            [_tool_definition()],
        )


def test_deepseek_maps_insufficient_resource_finish_reason():
    spy = _CompletionsSpy(
        [_response(finish_reason="insufficient_system_resource")]
    )
    provider = DeepSeekProvider(
        ProviderConfig(max_retries=0),
        client=_client(spy),
    )

    with pytest.raises(ProviderUnavailableError, match="system resources"):
        provider.complete([AssistantMessage("user", "status")], [])


@pytest.mark.parametrize("finish_reason", ("length", "content_filter"))
def test_deepseek_rejects_partial_or_filtered_responses(finish_reason):
    spy = _CompletionsSpy([_response(finish_reason=finish_reason)])
    provider = DeepSeekProvider(
        ProviderConfig(max_retries=0),
        client=_client(spy),
    )

    with pytest.raises(ProviderMalformedResponseError, match="unsupported reason"):
        provider.complete([AssistantMessage("user", "status")], [])


@pytest.mark.parametrize(
    "response",
    (
        _response(content="text", tool_calls=[_tool_call()], finish_reason="stop"),
        _response(content="text", tool_calls=(), finish_reason="tool_calls"),
    ),
)
def test_deepseek_rejects_inconsistent_finish_reason_payload(response):
    provider = DeepSeekProvider(
        ProviderConfig(max_retries=0),
        client=_client(_CompletionsSpy([response])),
    )

    with pytest.raises(ProviderMalformedResponseError):
        provider.complete(
            [AssistantMessage("user", "status")],
            [_tool_definition()],
        )


def test_deepseek_enforces_an_outer_wall_clock_deadline():
    started = threading.Event()
    released = threading.Event()

    class BlockingCompletions:
        def create(self, **kwargs):
            started.set()
            released.wait(5.0)
            return _response()

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=BlockingCompletions()),
        close=released.set,
    )
    provider = DeepSeekProvider(
        ProviderConfig(timeout_seconds=0.05, max_retries=0),
        client=client,
    )
    before = time.monotonic()

    with pytest.raises(ProviderTimeoutError, match="wall-clock"):
        provider.complete([AssistantMessage("user", "status")], [])

    assert started.is_set()
    assert released.is_set()
    assert time.monotonic() - before < 1.0


def test_deepseek_does_not_stack_requests_when_timed_out_thread_survives():
    released = threading.Event()
    attempts = 0

    class StubbornCompletions:
        def create(self, **kwargs):
            nonlocal attempts
            attempts += 1
            released.wait(5.0)
            return _response()

    provider = DeepSeekProvider(
        ProviderConfig(timeout_seconds=0.05, max_retries=0),
        client=SimpleNamespace(
            chat=SimpleNamespace(completions=StubbornCompletions()),
            close=lambda: None,
        ),
    )
    try:
        with pytest.raises(ProviderTimeoutError):
            provider.complete([AssistantMessage("user", "status")], [])
        with pytest.raises(ProviderUnavailableError, match="still stopping"):
            provider.complete([AssistantMessage("user", "status")], [])
        assert attempts == 1
    finally:
        released.set()


def test_deepseek_retry_backoff_is_interruptible():
    attempted = threading.Event()

    class FailingCompletions:
        def create(self, **kwargs):
            attempted.set()
            raise ProviderUnavailableError("temporary")

    provider = DeepSeekProvider(
        ProviderConfig(
            timeout_seconds=1,
            max_retries=1,
            retry_delay_seconds=5,
        ),
        client=SimpleNamespace(
            chat=SimpleNamespace(completions=FailingCompletions()),
        ),
    )
    failures: list[Exception] = []
    thread = threading.Thread(
        target=lambda: _capture_provider_failure(provider, failures),
    )

    thread.start()
    assert attempted.wait(1.0)
    provider.cancel_active_request()
    thread.join(1.0)

    assert not thread.is_alive()
    assert isinstance(failures[0], ProviderUnavailableError)


def test_deepseek_does_not_forward_unknown_sdk_exception_text():
    spy = _CompletionsSpy([RuntimeError("credential=do-not-display")])
    provider = DeepSeekProvider(
        ProviderConfig(max_retries=0),
        client=_client(spy),
    )

    with pytest.raises(ProviderUnavailableError) as caught:
        provider.complete([AssistantMessage("user", "status")], [])

    assert "do-not-display" not in str(caught.value)


def test_missing_deepseek_credential_is_clear_and_never_persisted():
    provider = DeepSeekProvider(
        ProviderConfig(max_retries=0),
        environ={},
    )

    with pytest.raises(ProviderAuthenticationError, match="DEEPSEEK_API_KEY"):
        provider.complete([AssistantMessage("user", "hello")], [])


@pytest.mark.parametrize(
    "base_url",
    (
        "http://api.deepseek.com",
        "https://api.deepseek.com.evil.example",
        "https://example.com/v1",
        "https://user@api.deepseek.com",
        "https://api.deepseek.com/v1///",
    ),
)
def test_deepseek_rejects_non_official_or_insecure_base_urls(base_url):
    with pytest.raises(ValueError, match="official HTTPS"):
        ProviderConfig(base_url=base_url)


def _capture_provider_failure(provider, failures):
    try:
        provider.complete([AssistantMessage("user", "status")], [])
    except Exception as error:
        failures.append(error)
