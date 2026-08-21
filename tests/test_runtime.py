"""Tests for runtime routing, retrying, streaming, and usage accounting."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable
from typing import cast, override

import pytest

from model_runtime import (
    ChatModel,
    FinishReason,
    InvalidRequestError,
    Message,
    ModelCapabilities,
    ModelRequest,
    ModelResponse,
    ModelRouter,
    ModelRuntime,
    ModelRuntimeError,
    RateLimitError,
    RequestTimeout,
    RetryPolicy,
    StreamEnd,
    StreamEvent,
    TextDelta,
    TraceObserver,
    Usage,
)

DEFAULT_USAGE = Usage(2, 3, 1)


def response(text: str = "ok", usage: Usage = DEFAULT_USAGE) -> ModelResponse:
    """Build a successful assistant response for a fake model."""
    return ModelResponse(
        message=Message.assistant(text),
        usage=usage,
        finish_reason=FinishReason.STOP,
    )


class FakeModel:
    """Fake non-streaming model that returns configured outcomes in order."""

    capabilities = ModelCapabilities(tools=True)

    def __init__(self, outcomes: list[ModelResponse | Exception]) -> None:
        self.outcomes = outcomes
        self.calls = 0

    async def complete(self, model_id: str, request: ModelRequest) -> ModelResponse:
        """Return or raise the next configured outcome."""
        outcome = self.outcomes[self.calls]
        self.calls += 1
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    async def stream(
        self, model_id: str, request: ModelRequest
    ) -> AsyncIterator[StreamEvent]:
        """Reject streaming because this fake only serves completions."""
        raise NotImplementedError
        yield  # pragma: no cover


class Recorder:
    """Trace observer that records runtime lifecycle events."""

    def __init__(self) -> None:
        self.requests: list[str] = []
        self.responses: list[tuple[str, Usage]] = []
        self.errors: list[tuple[str, ModelRuntimeError]] = []
        self.retries: list[tuple[str, ModelRuntimeError, int, float]] = []

    def on_request(self, model_id: str, request: ModelRequest) -> None:
        """Record the requested model ID."""
        self.requests.append(model_id)

    async def on_response(
        self,
        model_id: str,
        response: ModelResponse,
        latency_seconds: float,
        usage: Usage,
    ) -> None:
        """Record a successful response and its usage."""
        assert latency_seconds >= 0
        self.responses.append((model_id, usage))

    def on_error(
        self, model_id: str, error: ModelRuntimeError, latency_seconds: float
    ) -> None:
        """Record a terminal error for its model ID."""
        self.errors.append((model_id, error))

    def on_retry(
        self,
        model_id: str,
        error: ModelRuntimeError,
        attempt: int,
        delay_seconds: float,
    ) -> None:
        """Record a retryable failed attempt and its selected delay."""
        self.retries.append((model_id, error, attempt, delay_seconds))


class LegacyRecorder:
    """Trace observer implementing the original lifecycle without retries."""

    def __init__(self) -> None:
        self.response_count = 0

    def on_request(self, model_id: str, request: ModelRequest) -> None:
        """Accept the request notification."""
        return None

    def on_response(
        self,
        model_id: str,
        response: ModelResponse,
        latency_seconds: float,
        usage: Usage,
    ) -> None:
        """Count a successful response."""
        self.response_count += 1

    def on_error(
        self,
        model_id: str,
        error: ModelRuntimeError,
        latency_seconds: float,
    ) -> None:
        """Accept a terminal error notification."""
        return None


def make_runtime(
    model: ChatModel,
    *,
    retry_policy: RetryPolicy | None = None,
    observer: TraceObserver | None = None,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> ModelRuntime:
    """Create a runtime with one fake chat route."""
    router = ModelRouter({"chat": (model, "provider-model")})
    return ModelRuntime(
        router,
        retry_policy=retry_policy,
        observer=observer,
        sleep=sleep,
    )


@pytest.mark.asyncio
async def test_complete_retries_traces_and_accounts_for_usage() -> None:
    """Completions retry, trace their lifecycle, and aggregate usage."""
    model = FakeModel([RateLimitError("retry", retry_after=0), response()])
    observer = Recorder()
    sleeps: list[float] = []

    async def sleep(delay: float) -> None:
        """Record retry delay without waiting."""
        sleeps.append(delay)

    runtime = make_runtime(
        model,
        retry_policy=RetryPolicy(max_attempts=2, jitter=0),
        observer=observer,
        sleep=sleep,
    )
    result = await runtime.complete("chat", ModelRequest.from_text("hello"))

    assert result.text == "ok"
    assert model.calls == 2
    assert sleeps == [0]
    assert observer.requests == ["provider-model"]
    assert observer.responses == [("provider-model", Usage(2, 3, 1))]
    assert observer.errors == []
    assert len(observer.retries) == 1
    retry_model, retry_error, retry_attempt, retry_delay = observer.retries[0]
    assert retry_model == "provider-model"
    assert isinstance(retry_error, RateLimitError)
    assert retry_attempt == 1
    assert retry_delay == 0
    assert runtime.total_usage == Usage(2, 3, 1)
    assert runtime.usage_for("provider-model") == Usage(2, 3, 1)


@pytest.mark.asyncio
async def test_retry_notifications_are_backward_compatible() -> None:
    """Observers without the optional retry capability continue to work."""
    observer = LegacyRecorder()
    runtime = make_runtime(
        FakeModel([RateLimitError("retry", retry_after=0), response()]),
        retry_policy=RetryPolicy(max_attempts=2, jitter=0),
        observer=observer,
    )

    result = await runtime.complete("chat", ModelRequest.from_text("hello"))

    assert result.text == "ok"
    assert observer.response_count == 1


@pytest.mark.asyncio
async def test_complete_normalizes_timeout_and_unknown_route() -> None:
    """Timeouts and route lookup failures use public runtime errors."""

    class SlowModel(FakeModel):
        """Fake model whose completion never returns."""

        @override
        async def complete(self, model_id: str, request: ModelRequest) -> ModelResponse:
            """Wait indefinitely so the runtime timeout fires."""
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

    runtime = make_runtime(
        SlowModel([]),
        retry_policy=RetryPolicy(max_attempts=1),
    )
    with pytest.raises(RequestTimeout) as captured:
        await runtime.complete("chat", ModelRequest.from_text("hi", timeout=0.001))
    assert isinstance(captured.value.__cause__, TimeoutError)

    with pytest.raises(InvalidRequestError):
        await runtime.complete("missing", ModelRequest.from_text("hi"))


class RetryingStreamModel:
    """Fake stream model that fails before emitting its first event once."""

    capabilities = ModelCapabilities()

    def __init__(self) -> None:
        self.calls = 0

    async def complete(self, model_id: str, request: ModelRequest) -> ModelResponse:
        """Reject completion because this fake is used only for streams."""
        raise NotImplementedError

    async def stream(
        self, model_id: str, request: ModelRequest
    ) -> AsyncIterator[StreamEvent]:
        """Fail once, then yield a delta and a final event."""
        self.calls += 1
        if self.calls == 1:
            raise RateLimitError("before first event", retry_after=0)
        yield TextDelta("done")
        yield StreamEnd(response("done", Usage(4, 1)))


@pytest.mark.asyncio
async def test_stream_retries_only_before_emitting_and_records_final_usage() -> None:
    """Streams retry before a delta and record final usage."""
    model = RetryingStreamModel()
    observer = Recorder()
    runtime = make_runtime(
        model,
        retry_policy=RetryPolicy(max_attempts=2, jitter=0),
        observer=observer,
    )

    events = [
        event async for event in runtime.stream("chat", ModelRequest.from_text("hi"))
    ]

    assert model.calls == 2
    assert events[0] == TextDelta("done")
    assert isinstance(events[-1], StreamEnd)
    assert runtime.total_usage == Usage(4, 1)
    assert len(observer.retries) == 1
    assert observer.retries[0][2:] == (1, 0)


@pytest.mark.asyncio
async def test_stream_does_not_retry_after_a_delta() -> None:
    """Streams do not retry a failure after yielding a partial response."""

    class BrokenStream(RetryingStreamModel):
        """Fake stream that fails after emitting a partial delta."""

        @override
        async def stream(
            self, model_id: str, request: ModelRequest
        ) -> AsyncIterator[StreamEvent]:
            """Yield a partial delta and then a retryable error."""
            self.calls += 1
            yield TextDelta("partial")
            raise RateLimitError("connection lost")

    model = BrokenStream()
    runtime = make_runtime(model, retry_policy=RetryPolicy(max_attempts=3, jitter=0))
    iterator = runtime.stream("chat", ModelRequest.from_text("hi"))

    assert await anext(iterator) == TextDelta("partial")
    with pytest.raises(RateLimitError):
        await anext(iterator)
    assert model.calls == 1


@pytest.mark.asyncio
async def test_stream_can_be_closed_early_without_reporting_an_error() -> None:
    """Caller cancellation closes a stream without emitting an error trace."""

    class OpenStream(RetryingStreamModel):
        """Fake stream that remains open after a partial delta."""

        @override
        async def stream(
            self, model_id: str, request: ModelRequest
        ) -> AsyncIterator[StreamEvent]:
            """Yield a delta and wait until the caller closes the stream."""
            self.calls += 1
            yield TextDelta("partial")
            await asyncio.Event().wait()

    model = OpenStream()
    observer = Recorder()
    runtime = make_runtime(model, observer=observer)
    iterator = cast(
        AsyncGenerator[StreamEvent, None],
        runtime.stream("chat", ModelRequest.from_text("hi")),
    )

    assert await anext(iterator) == TextDelta("partial")
    await iterator.aclose()

    assert model.calls == 1
    assert observer.errors == []
