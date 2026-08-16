import sys
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


def _response(
    *,
    content="done",
    reasoning_content=None,
    tool_calls=(),
    finish_reason="stop",
):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason=finish_reason,
                message=SimpleNamespace(
                    content=content,
                    reasoning_content=reasoning_content,
                    tool_calls=tool_calls,
                ),
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


def _stream_chunk(
    *,
    content=None,
    reasoning_content=None,
    tool_calls=(),
    finish_reason=None,
    usage=None,
):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason=finish_reason,
                delta=SimpleNamespace(
                    content=content,
                    reasoning_content=reasoning_content,
                    tool_calls=tool_calls,
                ),
            )
        ],
        usage=usage,
    )


def _stream_tool_call(
    *,
    index=0,
    call_id=None,
    name=None,
    arguments=None,
):
    return SimpleNamespace(
        index=index,
        id=call_id,
        function=SimpleNamespace(
            name=name,
            arguments=arguments,
        ),
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


def test_deepseek_request_uses_openai_compatibility_and_enables_thinking():
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
    assert call["extra_body"] == {"thinking": {"type": "enabled"}}
    assert call["stream"] is False
    assert call["tools"][0]["function"]["name"] == "get_analysis_summary"
    assert "tool_choice" not in call


def test_deepseek_streams_visible_text_and_returns_normalized_response():
    spy = _CompletionsSpy(
        [
            iter(
                (
                    _stream_chunk(content="正在"),
                    _stream_chunk(content="检查模型"),
                    _stream_chunk(finish_reason="stop"),
                )
            )
        ]
    )
    provider = DeepSeekProvider(
        ProviderConfig(max_retries=0),
        client=_client(spy),
        environ={},
    )
    deltas = []

    result = provider.complete_stream(
        [AssistantMessage("user", "检查模型")],
        [],
        deltas.append,
    )

    assert deltas == ["正在", "检查模型"]
    assert result.message.content == "正在检查模型"
    assert result.finish_reason == "stop"
    assert spy.calls[0]["stream"] is True
    assert spy.calls[0]["extra_body"] == {
        "thinking": {"type": "enabled"}
    }


def test_deepseek_normalizes_and_replays_reasoning_content():
    spy = _CompletionsSpy(
        [_response(content="完成。", reasoning_content="先检查工具结果。")]
    )
    provider = DeepSeekProvider(
        ProviderConfig(max_retries=0),
        client=_client(spy),
        environ={},
    )
    prior = AssistantMessage(
        "assistant",
        content=None,
        tool_calls=(
            ToolCall(
                "call_1",
                "get_analysis_summary",
                {"session_id": "ses_test"},
            ),
        ),
        reasoning_content="需要先读取分析摘要。",
    )

    result = provider.complete(
        [
            prior,
            AssistantMessage("tool", "{}", tool_call_id="call_1"),
        ],
        [_tool_definition()],
    )

    assert result.message.reasoning_content == "先检查工具结果。"
    assert spy.calls[0]["messages"][0]["reasoning_content"] == (
        "需要先读取分析摘要。"
    )
    assert spy.calls[0]["messages"][0]["content"] == ""


def test_deepseek_streams_reasoning_separately_from_formal_content():
    spy = _CompletionsSpy(
        [
            iter(
                (
                    _stream_chunk(reasoning_content="先读取"),
                    _stream_chunk(reasoning_content="上下文"),
                    _stream_chunk(content="已完成。"),
                    _stream_chunk(finish_reason="stop"),
                )
            )
        ]
    )
    provider = DeepSeekProvider(
        ProviderConfig(max_retries=0),
        client=_client(spy),
        environ={},
    )
    text_deltas = []
    reasoning_deltas = []

    result = provider.complete_stream(
        [AssistantMessage("user", "检查模型")],
        [],
        text_deltas.append,
        reasoning_deltas.append,
    )

    assert reasoning_deltas == ["先读取", "上下文"]
    assert text_deltas == ["已完成。"]
    assert result.message.reasoning_content == "先读取上下文"
    assert result.message.content == "已完成。"


def test_deepseek_stream_assembles_tool_call_fragments():
    spy = _CompletionsSpy(
        [
            iter(
                (
                    _stream_chunk(
                        tool_calls=(
                            _stream_tool_call(
                                call_id="call_1",
                                name="get_analysis_summary",
                                arguments='{"session_',
                            ),
                        )
                    ),
                    _stream_chunk(
                        tool_calls=(
                            _stream_tool_call(
                                arguments='id":"ses_test"}',
                            ),
                        )
                    ),
                    _stream_chunk(finish_reason="tool_calls"),
                )
            )
        ]
    )
    provider = DeepSeekProvider(
        ProviderConfig(max_retries=0),
        client=_client(spy),
    )

    result = provider.complete_stream(
        [AssistantMessage("user", "检查模型")],
        [_tool_definition()],
        lambda _delta: None,
    )

    assert result.message.content is None
    assert result.message.tool_calls == (
        ToolCall(
            "call_1",
            "get_analysis_summary",
            {"session_id": "ses_test"},
        ),
    )
    assert result.finish_reason == "tool_calls"


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
            released.wait(2.0)
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


def test_deepseek_stream_enforces_first_chunk_timeout():
    started = threading.Event()
    released = threading.Event()

    class BlockingStreamCompletions:
        def create(self, **kwargs):
            def chunks():
                started.set()
                released.wait(2.0)
                yield _stream_chunk(content="late")

            return chunks()

    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=BlockingStreamCompletions()
        ),
        close=released.set,
    )
    provider = DeepSeekProvider(
        ProviderConfig(timeout_seconds=0.05, max_retries=0),
        client=client,
    )
    before = time.monotonic()
    deltas = []

    with pytest.raises(ProviderTimeoutError, match="first-chunk"):
        provider.complete_stream(
            [AssistantMessage("user", "status")],
            [],
            deltas.append,
        )

    assert started.is_set()
    assert released.is_set()
    assert deltas == []
    assert time.monotonic() - before < 1.0


def test_deepseek_stream_refreshes_idle_timeout_for_active_long_stream():
    class ActiveStreamCompletions:
        def create(self, **kwargs):
            del kwargs

            def chunks():
                for index in range(5):
                    time.sleep(0.02)
                    yield _stream_chunk(
                        content=str(index),
                        finish_reason="stop" if index == 4 else None,
                    )

            return chunks()

    provider = DeepSeekProvider(
        ProviderConfig(timeout_seconds=0.05, max_retries=0),
        client=SimpleNamespace(
            chat=SimpleNamespace(
                completions=ActiveStreamCompletions()
            ),
        ),
    )
    before = time.monotonic()
    deltas: list[str] = []

    response = provider.complete_stream(
        [AssistantMessage("user", "status")],
        [],
        deltas.append,
    )

    assert time.monotonic() - before > provider.config.timeout_seconds
    assert deltas == ["0", "1", "2", "3", "4"]
    assert response.message.content == "01234"


def test_deepseek_stream_enforces_idle_timeout_after_first_chunk():
    released = threading.Event()

    class IdleStreamCompletions:
        def create(self, **kwargs):
            del kwargs

            def chunks():
                yield _stream_chunk(content="early")
                released.wait(2.0)
                yield _stream_chunk(content="late", finish_reason="stop")

            return chunks()

    provider = DeepSeekProvider(
        ProviderConfig(timeout_seconds=0.05, max_retries=0),
        client=SimpleNamespace(
            chat=SimpleNamespace(
                completions=IdleStreamCompletions()
            ),
            close=released.set,
        ),
    )
    deltas: list[str] = []

    with pytest.raises(ProviderTimeoutError, match="idle timeout"):
        provider.complete_stream(
            [AssistantMessage("user", "status")],
            [],
            deltas.append,
        )

    assert released.is_set()
    assert deltas == ["early"]


def test_deepseek_stream_enforces_longer_total_timeout():
    closed = threading.Event()

    class EndlessActiveStreamCompletions:
        def create(self, **kwargs):
            del kwargs

            def chunks():
                while not closed.is_set():
                    time.sleep(0.02)
                    yield _stream_chunk(content="active")

            return chunks()

    provider = DeepSeekProvider(
        ProviderConfig(timeout_seconds=0.05, max_retries=0),
        client=SimpleNamespace(
            chat=SimpleNamespace(
                completions=EndlessActiveStreamCompletions()
            ),
            close=closed.set,
        ),
    )
    before = time.monotonic()

    with pytest.raises(ProviderTimeoutError, match="total deadline"):
        provider.complete_stream(
            [AssistantMessage("user", "status")],
            [],
            lambda _delta: None,
        )

    elapsed = time.monotonic() - before
    assert closed.is_set()
    assert elapsed > provider.config.timeout_seconds
    assert elapsed < 1.0


def test_deepseek_does_not_stack_requests_when_timed_out_thread_survives():
    released = threading.Event()
    attempts = 0

    class StubbornCompletions:
        def create(self, **kwargs):
            nonlocal attempts
            attempts += 1
            released.wait(2.0)
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


def test_deepseek_does_not_forward_unknown_sdk_exception_text(monkeypatch):
    sdk_error = type("SDKError", (Exception,), {})
    monkeypatch.setitem(
        sys.modules,
        "openai",
        SimpleNamespace(
            AuthenticationError=sdk_error,
            RateLimitError=sdk_error,
            APITimeoutError=sdk_error,
            APIConnectionError=sdk_error,
            APIStatusError=sdk_error,
        ),
    )
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
