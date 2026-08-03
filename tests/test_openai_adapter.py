"""Tests for the typed OpenAI Chat Completions components."""

from __future__ import annotations

from dataclasses import dataclass

import httpx
import openai
import pytest
from openai.types.chat import ChatCompletion, ChatCompletionChunk

from model_runtime import (
    AuthError,
    ContentFilterError,
    FinishReason,
    ImagePart,
    InvalidRequestError,
    Message,
    ModelRequest,
    ModelRuntimeError,
    OpenAIAdapter,
    OpenAIProviderOptions,
    RateLimitError,
    StreamEnd,
    TextDelta,
    TextPart,
    ToolCallDelta,
    ToolDefinition,
    Usage,
)
from model_runtime.providers.openai.transport import (
    OpenAICreateResult,
)


class FakeCompletions:
    """Minimal typed completion endpoint used by adapter tests."""

    def __init__(self, result: OpenAICreateResult) -> None:
        self.result = result
        self.calls: list[dict[str, object]] = []

    async def create(self, **kwargs: object) -> OpenAICreateResult:
        """Record request arguments and return the configured result."""
        self.calls.append(kwargs)
        return self.result


class FakeChat:
    """Minimal typed chat resource."""

    def __init__(self, completions: FakeCompletions) -> None:
        self.completions = completions


class FakeClient:
    """Structurally compatible OpenAI client for offline tests."""

    def __init__(self, completions: FakeCompletions) -> None:
        self.chat = FakeChat(completions)


def fake_client(
    result: OpenAICreateResult,
) -> tuple[FakeClient, FakeCompletions]:
    """Return a typed SDK-shaped client and its recording endpoint."""
    completions = FakeCompletions(result)
    return FakeClient(completions), completions


def chat_completion() -> ChatCompletion:
    """Build a typed completion with text, a tool call, and token usage."""
    return ChatCompletion.model_validate(
        {
            "id": "completion-1",
            "object": "chat.completion",
            "created": 1,
            "model": "gpt-test-2026",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "content": "Let me check.",
                        "tool_calls": [
                            {
                                "id": "call-1",
                                "type": "function",
                                "function": {
                                    "name": "weather",
                                    "arguments": '{"city":"Boston"}',
                                },
                            }
                        ],
                    },
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 4,
                "total_tokens": 14,
                "prompt_tokens_details": {"cached_tokens": 3},
            },
        }
    )


def test_constructs_sdk_client_with_retries_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Constructed SDK clients disable their own retry policy."""
    created, _ = fake_client(chat_completion())
    options: dict[str, object] = {}

    def make_client(**kwargs: object) -> FakeClient:
        """Capture SDK client options and return a typed fake client."""
        options.update(kwargs)
        return created

    monkeypatch.setattr(openai, "AsyncOpenAI", make_client)

    adapter = OpenAIAdapter(
        api_key="test-key",
        base_url="https://api.openai.test/v1",
        organization="org-test",
        project="project-test",
    )

    assert adapter.client is created
    assert options == {
        "max_retries": 0,
        "api_key": "test-key",
        "base_url": "https://api.openai.test/v1",
        "organization": "org-test",
        "project": "project-test",
    }


def test_openai_provider_options_are_typed_and_forward_compatible() -> None:
    """Known keys and explicit future JSON keys share one read-only mapping."""
    options = OpenAIProviderOptions(
        seed=7,
        reasoning_effort="high",
        extra={"future_option": {"enabled": True}},
    )

    assert dict(options) == {
        "seed": 7,
        "reasoning_effort": "high",
        "future_option": {"enabled": True},
    }
    with pytest.raises(ValueError, match="both extra and a named field"):
        OpenAIProviderOptions(seed=7, extra={"seed": 8})


@pytest.mark.asyncio
async def test_complete_translates_request_response_tools_and_provider_options() -> (
    None
):
    """Completion calls translate normalized fields in both directions."""
    raw_response = chat_completion()
    client, completions = fake_client(raw_response)
    adapter = OpenAIAdapter(client=client)
    request = ModelRequest(
        messages=(
            Message.system("Be concise."),
            Message(
                "user",
                (TextPart("What is here?"), ImagePart("https://example.com/image.png")),
            ),
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
        max_output_tokens=100,
        provider_options=OpenAIProviderOptions(seed=7, reasoning_effort="low"),
    )

    response = await adapter.complete("gpt-test", request)

    sent = completions.calls[0]
    assert sent["model"] == "gpt-test"
    assert sent["stream"] is False
    assert sent["messages"] == (
        {"role": "system", "content": "Be concise."},
        {
            "role": "user",
            "content": (
                {"type": "text", "text": "What is here?"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://example.com/image.png",
                        "detail": "auto",
                    },
                },
            ),
        },
    )
    assert sent["max_completion_tokens"] == 100
    assert sent["seed"] == 7
    assert sent["reasoning_effort"] == "low"
    assert response.raw is raw_response
    assert response.finish_reason is FinishReason.TOOL_CALLS
    assert response.text == "Let me check."
    assert not isinstance(response.tool_calls[0].arguments, str)
    assert response.tool_calls[0].arguments["city"] == "Boston"
    assert response.usage == Usage(10, 4, 3)


class FakeStream:
    """Finite typed async iterable that records whether it was closed."""

    def __init__(self, chunks: list[ChatCompletionChunk]) -> None:
        self._chunks = iter(chunks)
        self.closed = False

    def __aiter__(self) -> FakeStream:
        """Return this asynchronous iterator."""
        return self

    async def __anext__(self) -> ChatCompletionChunk:
        """Return the next configured chunk or stop iteration."""
        try:
            return next(self._chunks)
        except StopIteration:
            raise StopAsyncIteration from None

    async def close(self) -> None:
        """Record that the stream has been closed."""
        self.closed = True


@pytest.mark.asyncio
async def test_stream_reconstructs_response_and_collects_usage() -> None:
    """Streams reconstruct the final response and collect terminal usage."""
    common: dict[str, object] = {
        "id": "chunk-1",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "gpt-test",
    }
    stream = FakeStream(
        [
            ChatCompletionChunk.model_validate(
                {
                    **common,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "content": "The ",
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "call-1",
                                        "function": {
                                            "name": "weather",
                                            "arguments": '{"city":',
                                        },
                                    }
                                ],
                            },
                            "finish_reason": None,
                        }
                    ],
                }
            ),
            ChatCompletionChunk.model_validate(
                {
                    **common,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "content": "answer",
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "function": {"arguments": '"Boston"}'},
                                    },
                                ],
                            },
                            "finish_reason": "tool_calls",
                        }
                    ],
                }
            ),
            ChatCompletionChunk.model_validate(
                {
                    **common,
                    "choices": [],
                    "usage": {
                        "prompt_tokens": 8,
                        "completion_tokens": 5,
                        "total_tokens": 13,
                    },
                }
            ),
        ]
    )
    client, completions = fake_client(stream)
    adapter = OpenAIAdapter(client=client)

    events = [
        event
        async for event in adapter.stream("gpt-test", ModelRequest.from_text("hi"))
    ]

    assert [event.delta for event in events if isinstance(event, TextDelta)] == [
        "The ",
        "answer",
    ]
    assert len([event for event in events if isinstance(event, ToolCallDelta)]) == 2
    end = events[-1]
    assert isinstance(end, StreamEnd)
    assert end.response.text == "The answer"
    assert not isinstance(end.response.tool_calls[0].arguments, str)
    assert end.response.tool_calls[0].arguments["city"] == "Boston"
    assert end.response.finish_reason is FinishReason.TOOL_CALLS
    assert end.usage == Usage(8, 5)
    assert completions.calls[0]["stream_options"] == {"include_usage": True}
    assert stream.closed


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
            StatusError("blocked", 400, body={"error": {"code": "content_filter"}}),
            ContentFilterError,
        ),
    ],
)
def test_error_mapping(
    source: StatusError,
    expected: type[ModelRuntimeError],
) -> None:
    """SDK-shaped errors map to the corresponding public exception type."""
    mapped = OpenAIAdapter.translate_error(source)
    assert isinstance(mapped, expected)
    assert mapped.__cause__ is source
    if isinstance(mapped, RateLimitError):
        assert mapped.retry_after == 2.5


def test_maps_constructed_openai_sdk_exception() -> None:
    """A real OpenAI SDK exception retains its retry-after guidance."""
    request = httpx.Request("POST", "https://api.openai.test/v1/chat/completions")
    response = httpx.Response(
        429,
        request=request,
        headers={"retry-after": "1.25"},
        json={"error": {"message": "slow down"}},
    )
    source = openai.RateLimitError(
        "slow down",
        response=response,
        body={"error": {"message": "slow down"}},
    )

    mapped = OpenAIAdapter.translate_error(source)

    assert isinstance(mapped, RateLimitError)
    assert mapped.retry_after == 1.25
    assert mapped.__cause__ is source
