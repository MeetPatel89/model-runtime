"""Tests for the typed OpenAI Responses API components."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import httpx
import pytest
from openai import RateLimitError as OpenAISDKRateLimitError
from openai.types.responses import (
    Response,
    ResponseCompletedEvent,
    ResponseFunctionCallArgumentsDeltaEvent,
    ResponseOutputItemAddedEvent,
    ResponseRefusalDeltaEvent,
    ResponseStatus,
    ResponseStreamEvent,
    ResponseTextDeltaEvent,
)

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
from model_runtime.providers.openai import adapter as openai_adapter
from model_runtime.providers.openai.transport import OpenAICreateResult


class FakeResponses:
    """Minimal typed Responses endpoint used by adapter tests."""

    def __init__(self, result: OpenAICreateResult) -> None:
        self.result = result
        self.calls: list[dict[str, object]] = []

    async def create(self, **kwargs: object) -> OpenAICreateResult:
        """Record request arguments and return the configured result."""
        self.calls.append(kwargs)
        return self.result


class FakeClient:
    """Structurally compatible OpenAI client for offline tests."""

    def __init__(self, responses: FakeResponses) -> None:
        self.responses = responses


def fake_client(result: OpenAICreateResult) -> tuple[FakeClient, FakeResponses]:
    """Return a typed SDK-shaped client and its recording endpoint."""
    responses = FakeResponses(result)
    return FakeClient(responses), responses


def openai_response(
    output: Sequence[object] = (),
    *,
    status: ResponseStatus = "completed",
    incomplete_reason: str | None = None,
    with_usage: bool = True,
) -> Response:
    """Build an official SDK Response model for deterministic tests."""
    body: dict[str, object] = {
        "id": "resp-1",
        "created_at": 1,
        "model": "gpt-test-2026",
        "object": "response",
        "output": list(output),
        "parallel_tool_calls": True,
        "status": status,
        "tool_choice": "auto",
        "tools": [],
    }
    if incomplete_reason is not None:
        body["incomplete_details"] = {"reason": incomplete_reason}
    if with_usage:
        body["usage"] = {
            "input_tokens": 10,
            "input_tokens_details": {
                "cached_tokens": 3,
                "cache_write_tokens": 0,
            },
            "output_tokens": 4,
            "output_tokens_details": {"reasoning_tokens": 1},
            "total_tokens": 14,
        }
    return Response.model_validate(body)


def response_with_text_and_function_call() -> Response:
    """Build a response with output text, a function call, and usage."""
    return openai_response(
        (
            output_message("Let me check."),
            {
                "type": "function_call",
                "id": "fc-1",
                "call_id": "call-1",
                "name": "weather",
                "arguments": '{"city":"Boston"}',
                "status": "completed",
            },
        )
    )


def output_message(text: str) -> dict[str, object]:
    """Build a JSON-shaped Responses output-message fixture."""
    annotations: list[object] = []
    return {
        "id": "msg-1",
        "type": "message",
        "status": "completed",
        "role": "assistant",
        "content": [
            {
                "type": "output_text",
                "text": text,
                "annotations": annotations,
            }
        ],
    }


def test_constructs_sdk_client_with_retries_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Constructed SDK clients disable their own retry policy."""
    created, _ = fake_client(response_with_text_and_function_call())
    options: dict[str, object] = {}

    def make_client(**kwargs: object) -> FakeClient:
        """Capture SDK client options and return a typed fake client."""
        options.update(kwargs)
        return created

    monkeypatch.setattr(openai_adapter, "AsyncOpenAI", make_client)

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


def test_openai_provider_options_match_responses_and_remain_extensible() -> None:
    """Known Responses fields and explicit future fields form one mapping."""
    options = OpenAIProviderOptions(
        store=False,
        reasoning_effort="high",
        verbosity="low",
        tools=({"type": "web_search"},),
        extra={"future_option": {"enabled": True}},
    )

    assert dict(options) == {
        "store": False,
        "reasoning": {"effort": "high"},
        "text": {"verbosity": "low"},
        "tools": ({"type": "web_search"},),
        "future_option": {"enabled": True},
    }
    with pytest.raises(ValueError, match="both extra and a named field"):
        OpenAIProviderOptions(store=False, extra={"store": True})
    with pytest.raises(ValueError, match="reasoning.*effort.*conflict"):
        OpenAIProviderOptions(reasoning={}, reasoning_effort="low")


@pytest.mark.asyncio
async def test_complete_translates_responses_items_tools_and_options() -> None:
    """Responses calls translate normalized fields in both directions."""
    raw_response = response_with_text_and_function_call()
    client, responses = fake_client(raw_response)
    adapter = OpenAIAdapter(client=client)
    request = ModelRequest(
        messages=(
            Message.system("Be concise."),
            Message(
                "user",
                (TextPart("What is here?"), ImagePart("https://example.com/image.png")),
            ),
            Message.assistant(
                "Checking.",
                tool_calls=(ToolCall("call-prior", "weather", {"city": "New York"}),),
            ),
            Message.tool('{"temperature":20}', tool_call_id="call-prior"),
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
        max_output_tokens=100,
        timeout=20,
        provider_options=OpenAIProviderOptions(
            store=False,
            reasoning_effort="low",
            tools=({"type": "web_search"},),
        ),
    )

    response = await adapter.complete("gpt-test", request)

    sent = responses.calls[0]
    assert sent["model"] == "gpt-test"
    assert sent["stream"] is False
    assert "messages" not in sent
    assert "max_completion_tokens" not in sent
    assert sent["input"] == [
        {"role": "system", "content": "Be concise."},
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "What is here?"},
                {
                    "type": "input_image",
                    "image_url": "https://example.com/image.png",
                    "detail": "auto",
                },
            ],
        },
        {"role": "assistant", "content": "Checking."},
        {
            "type": "function_call",
            "call_id": "call-prior",
            "name": "weather",
            "arguments": '{"city":"New York"}',
        },
        {
            "type": "function_call_output",
            "call_id": "call-prior",
            "output": '{"temperature":20}',
        },
        {"role": "user", "content": "Now check Boston."},
    ]
    assert sent["tools"] == [
        {"type": "web_search"},
        {
            "type": "function",
            "name": "weather",
            "description": "Get weather",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
            },
            "strict": True,
        },
    ]
    assert sent["max_output_tokens"] == 100
    assert sent["reasoning"] == {"effort": "low"}
    assert sent["store"] is False
    assert sent["timeout"] == 20
    assert response.raw is raw_response
    assert response.finish_reason is FinishReason.TOOL_CALLS
    assert response.text == "Let me check."
    assert response.tool_calls[0].id == "call-1"
    assert not isinstance(response.tool_calls[0].arguments, str)
    assert response.tool_calls[0].arguments["city"] == "Boston"
    assert response.usage == Usage(10, 4, 3)


@pytest.mark.asyncio
async def test_rejects_fields_responses_cannot_represent() -> None:
    """Unsupported or adapter-owned request fields fail instead of disappearing."""
    client, _ = fake_client(openai_response())
    adapter = OpenAIAdapter(client=client)

    with pytest.raises(InvalidRequestError, match="does not support stop"):
        await adapter.complete(
            "gpt-test",
            ModelRequest.from_text("hi", stop=("END",)),
        )
    with pytest.raises(InvalidRequestError, match="cannot override.*model"):
        await adapter.complete(
            "gpt-test",
            ModelRequest.from_text("hi", provider_options={"model": "other"}),
        )
    with pytest.raises(InvalidRequestError, match="does not support message names"):
        await adapter.complete(
            "gpt-test",
            ModelRequest((Message("user", "hi", name="speaker"),)),
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("response", "expected"),
    [
        (openai_response(), FinishReason.STOP),
        (
            openai_response(
                status="incomplete",
                incomplete_reason="max_output_tokens",
            ),
            FinishReason.LENGTH,
        ),
        (
            openai_response(
                status="incomplete",
                incomplete_reason="content_filter",
            ),
            FinishReason.CONTENT_FILTER,
        ),
        (openai_response(status="failed"), FinishReason.ERROR),
    ],
)
async def test_maps_responses_terminal_statuses(
    response: Response,
    expected: FinishReason,
) -> None:
    """Responses terminal statuses map to normalized finish reasons."""
    client, _ = fake_client(response)
    result = await OpenAIAdapter(client=client).complete(
        "gpt-test", ModelRequest.from_text("hi")
    )
    assert result.finish_reason is expected


@pytest.mark.asyncio
async def test_complete_preserves_refusal_text() -> None:
    """Responses refusal content remains visible through normalized text."""
    raw_response = openai_response(
        (
            {
                "id": "msg-refusal",
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "refusal", "refusal": "I cannot help."}],
            },
        )
    )
    client, _ = fake_client(raw_response)

    result = await OpenAIAdapter(client=client).complete(
        "gpt-test", ModelRequest.from_text("hi")
    )

    assert result.text == "I cannot help."


class FakeStream:
    """Finite typed async iterable that records whether it was closed."""

    def __init__(self, events: Sequence[ResponseStreamEvent]) -> None:
        self._events = iter(events)
        self.closed = False

    def __aiter__(self) -> FakeStream:
        """Return this asynchronous iterator."""
        return self

    async def __anext__(self) -> ResponseStreamEvent:
        """Return the next configured event or stop iteration."""
        try:
            return next(self._events)
        except StopIteration:
            raise StopAsyncIteration from None

    async def close(self) -> None:
        """Record that the stream has been closed."""
        self.closed = True


@pytest.mark.asyncio
async def test_stream_consumes_typed_events_and_terminal_response() -> None:
    """Streams normalize typed deltas and use the terminal Response for totals."""
    final_response = openai_response(
        (
            output_message("The answer"),
            {
                "type": "function_call",
                "id": "fc-1",
                "call_id": "call-1",
                "name": "weather",
                "arguments": '{"city":"Boston"}',
                "status": "completed",
            },
        )
    )
    native_events: list[ResponseStreamEvent] = [
        ResponseTextDeltaEvent.model_validate(
            {
                "type": "response.output_text.delta",
                "content_index": 0,
                "delta": "The ",
                "item_id": "msg-1",
                "logprobs": [],
                "output_index": 0,
                "sequence_number": 1,
            }
        ),
        ResponseOutputItemAddedEvent.model_validate(
            {
                "type": "response.output_item.added",
                "item": {
                    "type": "function_call",
                    "id": "fc-1",
                    "call_id": "call-1",
                    "name": "weather",
                    "arguments": "",
                    "status": "in_progress",
                },
                "output_index": 2,
                "sequence_number": 2,
            }
        ),
        ResponseFunctionCallArgumentsDeltaEvent.model_validate(
            {
                "type": "response.function_call_arguments.delta",
                "delta": '{"city":',
                "item_id": "fc-1",
                "output_index": 2,
                "sequence_number": 3,
            }
        ),
        ResponseFunctionCallArgumentsDeltaEvent.model_validate(
            {
                "type": "response.function_call_arguments.delta",
                "delta": '"Boston"}',
                "item_id": "fc-1",
                "output_index": 2,
                "sequence_number": 4,
            }
        ),
        ResponseTextDeltaEvent.model_validate(
            {
                "type": "response.output_text.delta",
                "content_index": 0,
                "delta": "answer",
                "item_id": "msg-1",
                "logprobs": [],
                "output_index": 0,
                "sequence_number": 5,
            }
        ),
        ResponseCompletedEvent.model_validate(
            {
                "type": "response.completed",
                "response": final_response,
                "sequence_number": 6,
            }
        ),
    ]
    stream = FakeStream(native_events)
    client, responses = fake_client(stream)
    adapter = OpenAIAdapter(client=client)

    events = [
        event
        async for event in adapter.stream(
            "gpt-test",
            ModelRequest.from_text(
                "hi",
                provider_options=OpenAIProviderOptions(store=False),
            ),
        )
    ]

    assert [event.delta for event in events if isinstance(event, TextDelta)] == [
        "The ",
        "answer",
    ]
    tool_deltas = [event for event in events if isinstance(event, ToolCallDelta)]
    assert len(tool_deltas) == 3
    assert tool_deltas[0] == ToolCallDelta(0, "call-1", "weather", "")
    assert "".join(event.arguments_delta for event in tool_deltas) == (
        '{"city":"Boston"}'
    )
    end = events[-1]
    assert isinstance(end, StreamEnd)
    assert end.response.text == "The answer"
    assert end.response.raw is final_response
    assert end.response.tool_calls[0].id == "call-1"
    assert end.response.finish_reason is FinishReason.TOOL_CALLS
    assert end.usage == Usage(10, 4, 3)
    assert responses.calls[0]["stream"] is True
    assert "stream_options" not in responses.calls[0]
    assert stream.closed


@pytest.mark.asyncio
async def test_stream_maps_refusal_delta_and_requires_terminal_event() -> None:
    """Refusal text remains visible and truncated streams fail explicitly."""
    refusal = ResponseRefusalDeltaEvent.model_validate(
        {
            "type": "response.refusal.delta",
            "content_index": 0,
            "delta": "I cannot help.",
            "item_id": "msg-1",
            "output_index": 0,
            "sequence_number": 1,
        }
    )
    stream = FakeStream((refusal,))
    client, _ = fake_client(stream)

    with pytest.raises(ProviderUnavailableError, match="terminal response event"):
        _ = [
            event
            async for event in OpenAIAdapter(client=client).stream(
                "gpt-test", ModelRequest.from_text("hi")
            )
        ]
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
    request = httpx.Request("POST", "https://api.openai.test/v1/responses")
    response = httpx.Response(
        429,
        request=request,
        headers={"retry-after": "1.25"},
        json={"error": {"message": "slow down"}},
    )
    source = OpenAISDKRateLimitError(
        "slow down",
        response=response,
        body={"error": {"message": "slow down"}},
    )

    mapped = OpenAIAdapter.translate_error(source)

    assert isinstance(mapped, RateLimitError)
    assert mapped.retry_after == 1.25
    assert mapped.__cause__ is source
