"""Contract tests for provider-reusable adapter orchestration."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass

import pytest

from model_runtime import (
    FinishReason,
    Message,
    ModelCapabilities,
    ModelRequest,
    ModelResponse,
    ModelRuntimeError,
    ProviderAdapter,
    ProviderUnavailableError,
    StreamEnd,
    StreamEvent,
    TextDelta,
    Usage,
)


@dataclass(frozen=True, slots=True)
class NativeRequest:
    """Fake provider request payload."""

    model: str
    prompt: str
    stream: bool


@dataclass(frozen=True, slots=True)
class NativeResponse:
    """Fake provider completion payload."""

    text: str


@dataclass(frozen=True, slots=True)
class NativeChunk:
    """Fake provider stream payload."""

    text: str


class NativeTransport:
    """Fake network boundary shared by completion and stream tests."""

    def __init__(self, *, failure: Exception | None = None) -> None:
        self.failure = failure
        self.requests: list[NativeRequest] = []

    async def complete(self, request: NativeRequest) -> NativeResponse:
        """Record a completion or raise the configured SDK-like failure."""
        self.requests.append(request)
        if self.failure is not None:
            raise self.failure
        return NativeResponse("native response")

    async def stream(self, request: NativeRequest) -> AsyncIterator[NativeChunk]:
        """Record a stream and emit two native chunks."""
        self.requests.append(request)
        yield NativeChunk("native ")
        yield NativeChunk("stream")


class NativeStreamDecoder:
    """Stateful decoder owned by one fake provider stream."""

    def __init__(self) -> None:
        self.parts: list[str] = []

    def feed(self, chunk: NativeChunk) -> tuple[TextDelta, ...]:
        """Normalize and accumulate one native chunk."""
        self.parts.append(chunk.text)
        return (TextDelta(chunk.text),)

    def finish(self) -> StreamEnd:
        """Return one terminal normalized response."""
        response = normalized_response("".join(self.parts))
        return StreamEnd(response)


class NativeCodec:
    """Fake provider translation boundary."""

    def encode_request(
        self,
        model_id: str,
        request: ModelRequest,
        *,
        stream: bool,
    ) -> NativeRequest:
        """Translate the normalized request into the native payload."""
        return NativeRequest(model_id, request.messages[-1].text, stream)

    def decode_response(
        self,
        response: NativeResponse,
        *,
        fallback_model: str,
    ) -> ModelResponse:
        """Translate the native completion into the public response."""
        return normalized_response(response.text, model=fallback_model, raw=response)

    def stream_decoder(self, *, fallback_model: str) -> NativeStreamDecoder:
        """Create isolated state for a single stream."""
        return NativeStreamDecoder()


class NativeErrorMapper:
    """Fake provider exception translator."""

    def translate(self, error: Exception) -> ModelRuntimeError:
        """Normalize a native failure and retain its cause."""
        return ProviderUnavailableError(
            f"native failure: {error}",
            retryable=True,
            cause=error,
            provider="native",
        )


def normalized_response(
    text: str,
    *,
    model: str | None = None,
    raw: object | None = None,
) -> ModelResponse:
    """Build a normalized fake-provider response."""
    return ModelResponse(
        message=Message.assistant(text),
        usage=Usage(1, 2),
        finish_reason=FinishReason.STOP,
        raw=raw,
        model=model,
    )


def adapter(
    transport: NativeTransport,
) -> ProviderAdapter[
    NativeRequest,
    NativeResponse,
    NativeChunk,
]:
    """Assemble a fake provider from the reusable production components."""
    return ProviderAdapter(
        transport=transport,
        codec=NativeCodec(),
        error_mapper=NativeErrorMapper(),
        capabilities=ModelCapabilities(streaming=True),
    )


@pytest.mark.asyncio
async def test_provider_adapter_reuses_completion_and_stream_lifecycle() -> None:
    """Provider components plug into common invocation and stream orchestration."""
    transport = NativeTransport()
    model = adapter(transport)
    request = ModelRequest.from_text("hello")

    completion = await model.complete("native-model", request)
    events: list[StreamEvent] = [
        event async for event in model.stream("native-model", request)
    ]

    assert completion.text == "native response"
    assert transport.requests == [
        NativeRequest("native-model", "hello", False),
        NativeRequest("native-model", "hello", True),
    ]
    assert [event.delta for event in events if isinstance(event, TextDelta)] == [
        "native ",
        "stream",
    ]
    assert isinstance(events[-1], StreamEnd)
    assert events[-1].response.text == "native stream"


@pytest.mark.asyncio
async def test_provider_adapter_normalizes_transport_errors_once() -> None:
    """The shared adapter applies an injected provider error mapper."""
    source = RuntimeError("offline")
    model = adapter(NativeTransport(failure=source))

    with pytest.raises(ProviderUnavailableError) as captured:
        await model.complete("native-model", ModelRequest.from_text("hello"))

    assert captured.value.provider == "native"
    assert captured.value.retryable
    assert captured.value.__cause__ is source
