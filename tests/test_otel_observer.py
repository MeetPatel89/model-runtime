"""Network-free tests for the optional OpenTelemetry observer."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterator

import pytest

pytest.importorskip(
    "opentelemetry.sdk",
    reason="the optional OpenTelemetry dependencies are not installed",
)

from opentelemetry.sdk.resources import Resource  # noqa: E402
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider  # noqa: E402
from opentelemetry.sdk.trace.export import SimpleSpanProcessor  # noqa: E402
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (  # noqa: E402
    InMemorySpanExporter,
)
from opentelemetry.sdk.trace.sampling import (  # noqa: E402
    ParentBased,
    TraceIdRatioBased,
)
from opentelemetry.trace import SpanKind, StatusCode, Tracer  # noqa: E402

from model_runtime import (  # noqa: E402
    FinishReason,
    ImagePart,
    Message,
    ModelCapabilities,
    ModelRequest,
    ModelResponse,
    ModelRouter,
    ModelRuntime,
    ProviderUnavailableError,
    RateLimitError,
    RetryPolicy,
    StreamEnd,
    StreamEvent,
    TextDelta,
    TextPart,
    Usage,
)
from model_runtime.observability import OTelTraceObserver  # noqa: E402


class OutcomeModel:
    """Fake model returning configured responses or errors in order."""

    capabilities = ModelCapabilities(streaming=False)

    def __init__(self, outcomes: list[ModelResponse | Exception]) -> None:
        self._outcomes = iter(outcomes)

    async def complete(self, model_id: str, request: ModelRequest) -> ModelResponse:
        """Return or raise the next configured outcome."""
        outcome = next(self._outcomes)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    async def stream(
        self, model_id: str, request: ModelRequest
    ) -> AsyncIterator[StreamEvent]:
        """Reject streaming, which is outside this fake's responsibility."""
        raise NotImplementedError
        yield


class StreamingModel:
    """Fake model yielding one successful normalized stream."""

    capabilities = ModelCapabilities(streaming=True)

    async def complete(self, model_id: str, request: ModelRequest) -> ModelResponse:
        """Reject completion, which is outside this fake's responsibility."""
        raise NotImplementedError

    async def stream(
        self, model_id: str, request: ModelRequest
    ) -> AsyncIterator[StreamEvent]:
        """Yield one delta followed by a terminal response and usage."""
        response = _response()
        yield TextDelta("public answer")
        yield StreamEnd(response=response, usage=response.usage)


@pytest.fixture
def otel() -> Iterator[tuple[Tracer, InMemorySpanExporter]]:
    """Provide an isolated tracer with an in-memory synchronous exporter."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    try:
        yield provider.get_tracer("model-runtime-tests"), exporter
    finally:
        provider.shutdown()


def _response() -> ModelResponse:
    return ModelResponse(
        message=Message.assistant("public answer"),
        usage=Usage(input_tokens=7, output_tokens=3, cached_tokens=2),
        finish_reason=FinishReason.STOP,
        model="returned-model",
    )


def _runtime(
    model: OutcomeModel,
    observer: OTelTraceObserver,
    timestamps: Iterator[float],
) -> ModelRuntime:
    return ModelRuntime(
        ModelRouter({"chat": (model, "configured-model")}),
        observer=observer,
        clock=lambda: next(timestamps),
    )


def _only_span(exporter: InMemorySpanExporter) -> ReadableSpan:
    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    return spans[0]


@pytest.mark.asyncio
async def test_success_records_genai_attributes_without_content(
    otel: tuple[Tracer, InMemorySpanExporter],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Successful calls emit model, provider, usage, and latency metadata."""
    monkeypatch.delenv(
        "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", raising=False
    )
    tracer, exporter = otel
    observer = OTelTraceObserver(tracer, provider_name="openai")
    runtime = _runtime(OutcomeModel([_response()]), observer, iter((10.0, 10.125)))

    await runtime.complete(
        "chat",
        ModelRequest.from_text(
            "private prompt", temperature=0.2, max_output_tokens=100
        ),
    )

    span = _only_span(exporter)
    attributes = span.attributes
    assert attributes is not None
    assert span.name == "chat configured-model"
    assert span.kind is SpanKind.CLIENT
    assert span.status.status_code is StatusCode.OK
    assert attributes["gen_ai.operation.name"] == "chat"
    assert attributes["gen_ai.provider.name"] == "openai"
    assert attributes["gen_ai.request.model"] == "configured-model"
    assert attributes["gen_ai.request.temperature"] == 0.2
    assert attributes["gen_ai.request.max_tokens"] == 100
    assert attributes["gen_ai.response.model"] == "returned-model"
    assert attributes["gen_ai.response.finish_reasons"] == ("stop",)
    assert attributes["gen_ai.usage.input_tokens"] == 7
    assert attributes["gen_ai.usage.output_tokens"] == 3
    assert attributes["gen_ai.usage.cache_read.input_tokens"] == 2
    assert attributes["model_runtime.latency_ms"] == 125.0
    assert "private prompt" not in str(attributes)
    assert "public answer" not in str(attributes)


@pytest.mark.asyncio
async def test_content_capture_is_explicit_and_text_only(
    otel: tuple[Tracer, InMemorySpanExporter],
) -> None:
    """Opted-in spans use GenAI JSON attributes without capturing image URLs."""
    tracer, exporter = otel
    observer = OTelTraceObserver(
        tracer,
        provider_name="openai",
        capture_message_content=True,
    )
    runtime = _runtime(OutcomeModel([_response()]), observer, iter((2.0, 2.1)))
    request = ModelRequest(
        messages=(
            Message.system("private system prompt"),
            Message(
                "user",
                (
                    ImagePart("https://private.example/image.png"),
                    TextPart("private prompt"),
                ),
            ),
        )
    )

    await runtime.complete("chat", request)

    attributes = _only_span(exporter).attributes
    assert attributes is not None
    assert attributes["gen_ai.input.messages"] == (
        '[{"role":"system","parts":[{"type":"text","content":'
        '"private system prompt"}]},{"role":"user","parts":'
        '[{"type":"text","content":"private prompt"}]}]'
    )
    assert attributes["gen_ai.output.messages"] == (
        '[{"role":"assistant","parts":[{"type":"text","content":'
        '"public answer"}],"finish_reason":"stop"}]'
    )
    assert "private.example" not in str(attributes)


@pytest.mark.asyncio
async def test_content_capture_environment_and_constructor_precedence(
    otel: tuple[Tracer, InMemorySpanExporter],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The standard GenAI env flag enables capture unless code disables it."""
    tracer, exporter = otel
    monkeypatch.setenv("OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "TrUe")

    enabled = OTelTraceObserver(tracer, provider_name="openai")
    enabled.on_request("enabled", ModelRequest.from_text("captured"))
    enabled.on_response("enabled", _response(), 0.1, Usage())

    disabled = OTelTraceObserver(
        tracer,
        provider_name="openai",
        capture_message_content=False,
    )
    disabled.on_request("disabled", ModelRequest.from_text("not captured"))
    disabled.on_response("disabled", _response(), 0.1, Usage())

    spans = {span.name: span for span in exporter.get_finished_spans()}
    assert spans["chat enabled"].attributes is not None
    assert spans["chat disabled"].attributes is not None
    assert "gen_ai.input.messages" in spans["chat enabled"].attributes
    assert "gen_ai.input.messages" not in spans["chat disabled"].attributes


@pytest.mark.asyncio
async def test_terminal_error_ends_span_with_error_status(
    otel: tuple[Tracer, InMemorySpanExporter],
) -> None:
    """A terminal runtime error produces one ended error span."""
    tracer, exporter = otel
    observer = OTelTraceObserver(tracer, provider_name="openai")
    error = ProviderUnavailableError(
        "temporarily unavailable", retryable=False, provider="openai"
    )
    runtime = _runtime(OutcomeModel([error]), observer, iter((5.0, 5.25)))

    with pytest.raises(ProviderUnavailableError):
        await runtime.complete("chat", ModelRequest.from_text("hello"))

    span = _only_span(exporter)
    attributes = span.attributes
    assert attributes is not None
    assert span.status.status_code is StatusCode.ERROR
    assert attributes["error.type"] == "ProviderUnavailableError"
    assert attributes["model_runtime.error.retryable"] is False
    assert attributes["model_runtime.error.provider"] == "openai"
    assert attributes["model_runtime.latency_ms"] == 250.0
    assert any(event.name == "exception" for event in span.events)


@pytest.mark.asyncio
async def test_retries_are_events_on_one_logical_span(
    otel: tuple[Tracer, InMemorySpanExporter],
) -> None:
    """Every scheduled retry records normalized taxonomy on the open span."""
    tracer, exporter = otel
    observer = OTelTraceObserver(tracer, provider_name="openai")
    retry_error = RateLimitError(
        "rate limited",
        retry_after=0,
        status_code=429,
        provider="openai",
    )
    timestamps = iter((3.0, 3.2))
    runtime = ModelRuntime(
        ModelRouter(
            {"chat": (OutcomeModel([retry_error, _response()]), "configured-model")}
        ),
        retry_policy=RetryPolicy(max_attempts=2, jitter=0),
        observer=observer,
        clock=lambda: next(timestamps),
    )

    await runtime.complete("chat", ModelRequest.from_text("hello"))

    span = _only_span(exporter)
    retry_events = [
        event for event in span.events if event.name == "model_runtime.retry"
    ]
    assert len(retry_events) == 1
    attributes = retry_events[0].attributes
    assert attributes is not None
    assert attributes["model_runtime.retry.attempt"] == 1
    assert attributes["model_runtime.retry.next_attempt"] == 2
    assert attributes["model_runtime.retry.delay_ms"] == 0.0
    assert attributes["error.type"] == "RateLimitError"
    assert attributes["model_runtime.error.retryable"] is True
    assert attributes["model_runtime.error.provider"] == "openai"
    assert attributes["model_runtime.error.status_code"] == 429
    assert span.status.status_code is StatusCode.OK


@pytest.mark.asyncio
async def test_completed_stream_ends_span_with_terminal_usage(
    otel: tuple[Tracer, InMemorySpanExporter],
) -> None:
    """A fully consumed stream produces the same logical-call span."""
    tracer, exporter = otel
    observer = OTelTraceObserver(tracer, provider_name="openai")
    timestamps = iter((1.0, 1.5))
    runtime = ModelRuntime(
        ModelRouter({"chat": (StreamingModel(), "configured-model")}),
        observer=observer,
        clock=lambda: next(timestamps),
    )

    events = [
        event
        async for event in runtime.stream("chat", ModelRequest.from_text("private"))
    ]

    assert len(events) == 2
    span = _only_span(exporter)
    assert span.attributes is not None
    assert span.attributes["gen_ai.usage.input_tokens"] == 7
    assert span.attributes["gen_ai.usage.output_tokens"] == 3


@pytest.mark.asyncio
async def test_concurrent_tasks_finish_their_own_spans(
    otel: tuple[Tracer, InMemorySpanExporter],
) -> None:
    """Context-local correlation keeps interleaved async calls separate."""
    tracer, exporter = otel
    observer = OTelTraceObserver(tracer, provider_name="openai")

    async def observe(model_id: str, usage: Usage) -> None:
        observer.on_request(model_id, ModelRequest.from_text("private"))
        await asyncio.sleep(0)
        observer.on_response(model_id, _response(), 0.1, usage)

    await asyncio.gather(
        observe("model-a", Usage(input_tokens=1, output_tokens=2)),
        observe("model-b", Usage(input_tokens=3, output_tokens=4)),
    )

    spans_by_model = {
        span.attributes["gen_ai.request.model"]: span
        for span in exporter.get_finished_spans()
        if span.attributes is not None
    }
    assert set(spans_by_model) == {"model-a", "model-b"}
    assert spans_by_model["model-a"].attributes is not None
    assert spans_by_model["model-b"].attributes is not None
    assert spans_by_model["model-a"].attributes["gen_ai.usage.input_tokens"] == 1
    assert spans_by_model["model-b"].attributes["gen_ai.usage.input_tokens"] == 3


@pytest.mark.asyncio
async def test_app_span_is_parent_of_async_runtime_spans(
    otel: tuple[Tracer, InMemorySpanExporter],
) -> None:
    """Current OTel context propagates into child asyncio tasks."""
    tracer, exporter = otel
    observer = OTelTraceObserver(tracer, provider_name="openai")

    with tracer.start_as_current_span("chat session turn") as parent:
        parent_context = parent.get_span_context()
        await asyncio.gather(
            _complete_observed_call(observer, "model-a"),
            _complete_observed_call(observer, "model-b"),
        )

    spans = exporter.get_finished_spans()
    children = [span for span in spans if span.name.startswith("chat model-")]
    assert len(children) == 2
    for child in children:
        assert child.context is not None
        assert child.context.trace_id == parent_context.trace_id
        assert child.parent is not None
        assert child.parent.span_id == parent_context.span_id


async def _complete_observed_call(observer: OTelTraceObserver, model_id: str) -> None:
    """Complete one runtime call in the current async context."""
    runtime = ModelRuntime(
        ModelRouter({"chat": (OutcomeModel([_response()]), model_id)}),
        observer=observer,
    )
    await runtime.complete("chat", ModelRequest.from_text("private"))


@pytest.mark.asyncio
async def test_resource_attributes_are_exported_on_runtime_spans() -> None:
    """Application-owned service metadata is present on runtime child spans."""
    sampled_exporter = InMemorySpanExporter()
    sampled_provider = TracerProvider(
        resource=Resource.create(
            {
                "service.name": "model-runtime-tests",
                "service.version": "test-version",
            }
        )
    )
    sampled_provider.add_span_processor(SimpleSpanProcessor(sampled_exporter))
    sampled_tracer = sampled_provider.get_tracer("sampled")
    try:
        with sampled_tracer.start_as_current_span("sampled parent"):
            await _complete_observed_call(
                OTelTraceObserver(sampled_tracer, provider_name="openai"),
                "model",
            )
        child = next(
            span
            for span in sampled_exporter.get_finished_spans()
            if span.name == "chat model"
        )
        assert child.resource.attributes["service.name"] == "model-runtime-tests"
        assert child.resource.attributes["service.version"] == "test-version"
    finally:
        sampled_provider.shutdown()


@pytest.mark.asyncio
async def test_parent_based_sampler_drops_the_whole_trace() -> None:
    """A dropped root sampling decision also drops runtime child spans."""
    dropped_exporter = InMemorySpanExporter()
    dropped_provider = TracerProvider(sampler=ParentBased(TraceIdRatioBased(0.0)))
    dropped_provider.add_span_processor(SimpleSpanProcessor(dropped_exporter))
    dropped_tracer = dropped_provider.get_tracer("dropped")
    try:
        with dropped_tracer.start_as_current_span("dropped parent"):
            await _complete_observed_call(
                OTelTraceObserver(dropped_tracer, provider_name="openai"),
                "model",
            )
        assert dropped_exporter.get_finished_spans() == ()
    finally:
        dropped_provider.shutdown()


def test_provider_name_cannot_be_blank(
    otel: tuple[Tracer, InMemorySpanExporter],
) -> None:
    """Invalid provider metadata fails before any request is observed."""
    tracer, _ = otel
    with pytest.raises(ValueError, match="provider_name must not be blank"):
        OTelTraceObserver(tracer, provider_name="  ")
