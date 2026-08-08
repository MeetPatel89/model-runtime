"""Tests for the typed Anthropic Messages API components."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator, Sequence
from dataclasses import dataclass
from types import TracebackType

import httpx
import pytest
from anthropic import RateLimitError as AnthropicSDKRateLimitError
from anthropic import types as anthropic_types
from anthropic.lib.streaming import ParsedMessageStopEvent
from anthropic.types import (
    RawContentBlockDeltaEvent,
    RawContentBlockStartEvent,
    StopReason,
)

from model_runtime import (
    AnthropicAdapter,
    AnthropicProviderOptions,
    AuthError,
    ContentFilterError,
    FinishReason,
    ImagePart,
    InvalidRequestError,
    Message,
    ModelRequest,
    ModelRuntimeError,
    ProviderUnavailableError,
    RateLimitError,
    StreamEnd,
    TextDelta,
    TextPart,
    ToolCall,
    ToolCallDelta,
    ToolDefinition,
    Usage,
)
from model_runtime.providers.anthropic import adapter as anthropic_adapter
from model_runtime.providers.anthropic.transport import AnthropicStreamEvent


def anthropic_message(
    content: Sequence[object] = (),
    *,
    stop_reason: StopReason = "end_turn",
    with_usage: bool = True,
) -> anthropic_types.Message:
    """Build an official SDK Message model for deterministic tests."""
    usage: dict[str, object]
    if with_usage:
        usage = {
            "input_tokens": 10,
            "output_tokens": 4,
            "cache_creation_input_tokens": 2,
            "cache_read_input_tokens": 3,
        }
    else:
        usage = {"input_tokens": 0, "output_tokens": 0}
    return anthropic_types.Message.model_validate(
        {
            "id": "msg-1",
            "type": "message",
            "role": "assistant",
            "model": "claude-test-2026",
            "content": list(content),
            "stop_reason": stop_reason,
            "stop_sequence": None,
            "usage": usage,
        }
    )


def response_with_text_and_tool() -> anthropic_types.Message:
    """Build a message containing text and one client tool call."""
    return anthropic_message(
        (
            {"type": "text", "text": "Let me check."},
            {
                "type": "tool_use",
                "id": "toolu-1",
                "name": "weather",
                "input": {"city": "Boston"},
            },
        ),
        stop_reason="tool_use",
    )


class FakeEventStream:
    """Async iterator over typed Anthropic streaming events."""

    def __init__(self, events: Sequence[AnthropicStreamEvent]) -> None:
        self._events: Iterator[AnthropicStreamEvent] = iter(events)

    def __aiter__(self) -> AsyncIterator[AnthropicStreamEvent]:
        """Return this event iterator."""
        return self

    async def __anext__(self) -> AnthropicStreamEvent:
        """Return the next configured event."""
        try:
            return next(self._events)
        except StopIteration:
            raise StopAsyncIteration from None


class FakeStreamManager:
    """SDK-shaped manager recording stream cleanup."""

    def __init__(self, events: Sequence[AnthropicStreamEvent]) -> None:
        self.stream = FakeEventStream(events)
        self.closed = False

    async def __aenter__(self) -> FakeEventStream:
        """Open the configured event stream."""
        return self.stream

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Record that the stream was closed."""
        self.closed = True


class FakeMessages:
    """Minimal typed Messages endpoint used by adapter tests."""

    def __init__(
        self,
        result: anthropic_types.Message,
        events: Sequence[AnthropicStreamEvent] = (),
    ) -> None:
        self.result = result
        self.complete_calls: list[dict[str, object]] = []
        self.stream_calls: list[dict[str, object]] = []
        self.stream_manager = FakeStreamManager(events)

    async def create(self, **kwargs: object) -> anthropic_types.Message:
        """Record request arguments and return the configured result."""
        self.complete_calls.append(kwargs)
        return self.result

    def stream(self, **kwargs: object) -> FakeStreamManager:
        """Record request arguments and return the configured stream manager."""
        self.stream_calls.append(kwargs)
        return self.stream_manager


class FakeClient:
    """Structurally compatible Anthropic client for offline tests."""

    def __init__(self, messages: FakeMessages) -> None:
        self.messages = messages


def fake_client(
    result: anthropic_types.Message,
    events: Sequence[AnthropicStreamEvent] = (),
) -> tuple[FakeClient, FakeMessages]:
    """Return a typed SDK-shaped client and its recording endpoint."""
    messages = FakeMessages(result, events)
    return FakeClient(messages), messages


def test_constructs_sdk_client_with_retries_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Constructed SDK clients disable their own retry policy."""
    created, _ = fake_client(response_with_text_and_tool())
    options: dict[str, object] = {}

    def make_client(**kwargs: object) -> FakeClient:
        """Capture SDK client options and return a typed fake client."""
        options.update(kwargs)
        return created

    monkeypatch.setattr(anthropic_adapter, "AsyncAnthropic", make_client)

    adapter = AnthropicAdapter(
        api_key="test-key",
        auth_token="test-token",
        base_url="https://api.anthropic.test",
    )

    assert adapter.client is created
    assert options == {
        "max_retries": 0,
        "api_key": "test-key",
        "auth_token": "test-token",
        "base_url": "https://api.anthropic.test",
    }


def test_anthropic_provider_options_are_typed_and_extensible() -> None:
    """Known Messages fields and explicit future fields form one mapping."""
    options = AnthropicProviderOptions(
        effort="high",
        service_tier="standard_only",
        tools=({"type": "web_search_20260318", "name": "web_search"},),
        extra={"future_option": {"enabled": True}},
    )

    assert dict(options) == {
        "output_config": {"effort": "high"},
        "service_tier": "standard_only",
        "tools": ({"type": "web_search_20260318", "name": "web_search"},),
        "future_option": {"enabled": True},
    }
    with pytest.raises(ValueError, match="both extra and a named field"):
        AnthropicProviderOptions(top_k=10, extra={"top_k": 20})
    with pytest.raises(ValueError, match="output_config.*effort.*conflict"):
        AnthropicProviderOptions(output_config={}, effort="low")


def test_rejects_nonpositive_default_token_limit() -> None:
    """The required Anthropic token default cannot represent invalid limits."""
    client, _ = fake_client(anthropic_message())

    with pytest.raises(ValueError, match="greater than zero"):
        AnthropicAdapter(client=client, default_max_output_tokens=0)


@pytest.mark.asyncio
async def test_complete_translates_messages_tools_images_and_options() -> None:
    """Messages calls translate normalized fields in both directions."""
    raw_response = response_with_text_and_tool()
    client, messages = fake_client(raw_response)
    adapter = AnthropicAdapter(client=client)
    request = ModelRequest(
        messages=(
            Message.system("Be concise."),
            Message.developer("Use tools when needed."),
            Message(
                "user",
                (
                    TextPart("What is here?"),
                    ImagePart("https://example.com/image.png"),
                ),
            ),
            Message.assistant(
                "Checking.",
                tool_calls=(ToolCall("toolu-prior", "weather", {"city": "New York"}),),
            ),
            Message.tool('{"temperature":20}', tool_call_id="toolu-prior"),
            Message.developer("Give temperatures in Celsius."),
            Message.user("Now check Boston."),
        ),
        tools=(
            ToolDefinition(
                "weather",
                "Get weather",
                {"type": "object", "properties": {"city": {"type": "string"}}},
                strict=True,
            ),
        ),
        temperature=0.2,
        max_output_tokens=200,
        stop=("END",),
        timeout=20,
        provider_options=AnthropicProviderOptions(
            effort="low",
            tools=({"type": "web_search_20260318", "name": "web_search"},),
        ),
    )

    response = await adapter.complete("claude-test", request)

    sent = messages.complete_calls[0]
    assert sent["model"] == "claude-test"
    assert sent["max_tokens"] == 200
    assert "stream" not in sent
    assert sent["system"] == [
        {"type": "text", "text": "Be concise."},
        {"type": "text", "text": "Use tools when needed."},
    ]
    assert sent["messages"] == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What is here?"},
                {
                    "type": "image",
                    "source": {
                        "type": "url",
                        "url": "https://example.com/image.png",
                    },
                },
            ],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Checking."},
                {
                    "type": "tool_use",
                    "id": "toolu-prior",
                    "name": "weather",
                    "input": {"city": "New York"},
                },
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu-prior",
                    "content": '{"temperature":20}',
                }
            ],
        },
        {
            "role": "system",
            "content": [{"type": "text", "text": "Give temperatures in Celsius."}],
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": "Now check Boston."}],
        },
    ]
    assert sent["tools"] == [
        {"type": "web_search_20260318", "name": "web_search"},
        {
            "name": "weather",
            "description": "Get weather",
            "input_schema": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
            },
            "strict": True,
        },
    ]
    assert sent["temperature"] == 0.2
    assert sent["stop_sequences"] == ["END"]
    assert sent["output_config"] == {"effort": "low"}
    assert sent["timeout"] == 20
    assert response.raw is raw_response
    assert response.finish_reason is FinishReason.TOOL_CALLS
    assert response.text == "Let me check."
    assert not isinstance(response.tool_calls[0].arguments, str)
    assert response.tool_calls[0].arguments["city"] == "Boston"
    assert response.usage == Usage(15, 4, 3)


@pytest.mark.asyncio
async def test_default_max_tokens_and_base64_images() -> None:
    """Anthropic gets a stable token default and translates supported data URLs."""
    client, messages = fake_client(anthropic_message())
    request = ModelRequest(
        messages=(
            Message(
                "user",
                (ImagePart("data:image/png;base64,aGVsbG8="),),
            ),
        )
    )

    await AnthropicAdapter(
        client=client,
        default_max_output_tokens=321,
    ).complete("claude-test", request)

    assert messages.complete_calls[0]["max_tokens"] == 321
    assert "output_config" not in messages.complete_calls[0]
    assert messages.complete_calls[0]["messages"] == [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": "aGVsbG8=",
                    },
                }
            ],
        }
    ]


@pytest.mark.asyncio
async def test_rejects_unrepresentable_or_adapter_owned_fields() -> None:
    """Unsupported or adapter-owned request fields fail instead of disappearing."""
    client, _ = fake_client(anthropic_message())
    adapter = AnthropicAdapter(client=client)

    with pytest.raises(InvalidRequestError, match="cannot override.*max_tokens"):
        await adapter.complete(
            "claude-test",
            ModelRequest.from_text("hi", provider_options={"max_tokens": 10}),
        )
    with pytest.raises(InvalidRequestError, match="does not support message names"):
        await adapter.complete(
            "claude-test",
            ModelRequest((Message("user", "hi", name="speaker"),)),
        )
    with pytest.raises(InvalidRequestError, match="detail levels"):
        await adapter.complete(
            "claude-test",
            ModelRequest((Message("user", (ImagePart("https://x", "high"),)),)),
        )
    with pytest.raises(InvalidRequestError, match="JSON object"):
        await adapter.complete(
            "claude-test",
            ModelRequest(
                (
                    Message.assistant(
                        tool_calls=(ToolCall("toolu-1", "weather", "not-json"),)
                    ),
                )
            ),
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("stop_reason", "expected"),
    [
        ("end_turn", FinishReason.STOP),
        ("stop_sequence", FinishReason.STOP),
        ("max_tokens", FinishReason.LENGTH),
        ("model_context_window_exceeded", FinishReason.LENGTH),
        ("tool_use", FinishReason.TOOL_CALLS),
        ("refusal", FinishReason.CONTENT_FILTER),
        ("pause_turn", FinishReason.UNKNOWN),
    ],
)
async def test_maps_anthropic_stop_reasons(
    stop_reason: StopReason,
    expected: FinishReason,
) -> None:
    """Messages stop reasons map to normalized finish reasons."""
    client, _ = fake_client(anthropic_message(stop_reason=stop_reason))
    result = await AnthropicAdapter(client=client).complete(
        "claude-test", ModelRequest.from_text("hi")
    )
    assert result.finish_reason is expected


def stream_events(
    final_message: anthropic_types.Message,
) -> list[AnthropicStreamEvent]:
    """Build typed text, tool, and terminal stream events."""
    return [
        RawContentBlockStartEvent.model_validate(
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            }
        ),
        RawContentBlockDeltaEvent.model_validate(
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "The answer"},
            }
        ),
        RawContentBlockStartEvent.model_validate(
            {
                "type": "content_block_start",
                "index": 2,
                "content_block": {
                    "type": "tool_use",
                    "id": "toolu-1",
                    "name": "weather",
                    "input": {},
                },
            }
        ),
        RawContentBlockDeltaEvent.model_validate(
            {
                "type": "content_block_delta",
                "index": 2,
                "delta": {
                    "type": "input_json_delta",
                    "partial_json": '{"city":"Boston"}',
                },
            }
        ),
        ParsedMessageStopEvent[object].model_validate(
            {
                "type": "message_stop",
                "message": final_message.model_dump(),
            }
        ),
    ]


@pytest.mark.asyncio
async def test_stream_consumes_typed_events_and_terminal_message() -> None:
    """Streams normalize typed deltas and use the SDK's final Message."""
    final_message = anthropic_message(
        (
            {"type": "text", "text": "The answer"},
            {
                "type": "tool_use",
                "id": "toolu-1",
                "name": "weather",
                "input": {"city": "Boston"},
            },
        ),
        stop_reason="tool_use",
    )
    client, messages = fake_client(final_message, stream_events(final_message))

    events = [
        event
        async for event in AnthropicAdapter(client=client).stream(
            "claude-test",
            ModelRequest.from_text("hi", max_output_tokens=50),
        )
    ]

    assert [event.delta for event in events if isinstance(event, TextDelta)] == [
        "The answer"
    ]
    tool_deltas = [event for event in events if isinstance(event, ToolCallDelta)]
    assert tool_deltas == [
        ToolCallDelta(0, "toolu-1", "weather", ""),
        ToolCallDelta(0, arguments_delta='{"city":"Boston"}'),
    ]
    end = events[-1]
    assert isinstance(end, StreamEnd)
    assert end.response.text == "The answer"
    assert isinstance(end.response.raw, anthropic_types.Message)
    assert end.response.tool_calls[0].id == "toolu-1"
    assert end.response.finish_reason is FinishReason.TOOL_CALLS
    assert end.usage == Usage(15, 4, 3)
    assert messages.stream_calls[0]["max_tokens"] == 50
    assert "stream" not in messages.stream_calls[0]
    assert messages.stream_manager.closed


@pytest.mark.asyncio
async def test_stream_requires_terminal_message_and_closes_manager() -> None:
    """Truncated Anthropic streams fail explicitly after releasing resources."""
    delta = RawContentBlockDeltaEvent.model_validate(
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "partial"},
        }
    )
    client, messages = fake_client(anthropic_message(), (delta,))

    with pytest.raises(ProviderUnavailableError, match="terminal message event"):
        _ = [
            event
            async for event in AnthropicAdapter(client=client).stream(
                "claude-test", ModelRequest.from_text("hi")
            )
        ]
    assert messages.stream_manager.closed


@dataclass(frozen=True, slots=True)
class FakeHTTPResponse:
    """HTTP response shape inspected by the error extractor."""

    headers: dict[str, str]


class StatusError(Exception):
    """SDK-shaped exception with an HTTP status and optional retry headers."""

    def __init__(
        self,
        message: str,
        status_code: int,
        *,
        body: object | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.body = body
        self.response = FakeHTTPResponse(headers or {})


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (StatusError("bad key", 401), AuthError),
        (StatusError("slow", 429, headers={"retry-after": "2.5"}), RateLimitError),
        (StatusError("bad request", 400), InvalidRequestError),
        (
            StatusError("blocked", 400, body={"error": {"type": "safety_policy"}}),
            ContentFilterError,
        ),
    ],
)
def test_error_mapping(
    source: StatusError,
    expected: type[ModelRuntimeError],
) -> None:
    """SDK-shaped errors map to the corresponding public exception type."""
    mapped = AnthropicAdapter.translate_error(source)
    assert isinstance(mapped, expected)
    assert mapped.__cause__ is source
    if isinstance(mapped, RateLimitError):
        assert mapped.retry_after == 2.5


def test_maps_constructed_anthropic_sdk_exception() -> None:
    """A real Anthropic SDK exception retains retry-after guidance."""
    request = httpx.Request("POST", "https://api.anthropic.test/v1/messages")
    response = httpx.Response(
        429,
        request=request,
        headers={"retry-after": "1.25"},
        json={"error": {"message": "slow down"}},
    )
    source = AnthropicSDKRateLimitError(
        "slow down",
        response=response,
        body={"error": {"message": "slow down"}},
    )

    mapped = AnthropicAdapter.translate_error(source)

    assert isinstance(mapped, RateLimitError)
    assert mapped.retry_after == 1.25
    assert mapped.__cause__ is source
