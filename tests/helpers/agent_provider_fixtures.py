from fem_agent.providers.base import AssistantMessage, ProviderResponse
from fem_agent.providers.fake import FakeProvider


def tool_response(*calls):
    return ProviderResponse(
        AssistantMessage("assistant", tool_calls=tuple(calls)),
        finish_reason="tool_calls",
    )


def text_response(text):
    return ProviderResponse(
        AssistantMessage("assistant", content=text),
        finish_reason="stop",
    )


class StreamingFakeProvider(FakeProvider):
    def complete_stream(self, messages, tools, on_text_delta):
        response = super().complete(messages, tools)
        content = response.message.content or ""
        split = max(1, len(content) // 2)
        for delta in (content[:split], content[split:]):
            if delta:
                on_text_delta(delta)
        return response


class ReasoningStreamingFakeProvider(FakeProvider):
    supports_reasoning_stream = True

    def complete_stream(
        self,
        messages,
        tools,
        on_text_delta,
        on_reasoning_delta,
    ):
        response = super().complete(messages, tools)
        reasoning = response.message.reasoning_content or ""
        reasoning_split = max(1, len(reasoning) // 2)
        for delta in (reasoning[:reasoning_split], reasoning[reasoning_split:]):
            if delta:
                on_reasoning_delta(delta)
        content = response.message.content or ""
        content_split = max(1, len(content) // 2)
        for delta in (content[:content_split], content[content_split:]):
            if delta:
                on_text_delta(delta)
        return response
