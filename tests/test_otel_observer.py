"""Network-free tests for the optional OpenTelemetry observer."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterator

import pytest

pytest.importorskip(
    "opentelemetry.sdk",
    reason="the optional OpenTelemetry dependencies are not installed",
)

from opentelemetry.sdk.trace import ReadableSpan, TracerProvider  # noqa: E402
from opentelemetry.sdk.trace.export import SimpleSpanProcessor  # noqa: E402
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (  # noqa: E402
    InMemorySpanExporter,
)
from opentelemetry.trace import SpanKind, StatusCode, Tracer  # noqa: E402

from model_runtime import (  # noqa: E402
    FinishReason,
    Message,
    ModelCapabilities,
    ModelRequest,
    ModelResponse,
    ModelRouter,
    ModelRuntime,
    ProviderUnavailableError,
    StreamEnd,
    StreamEvent,
    TextDelta,
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
) -> None:
    """Successful calls emit model, provider, usage, and latency metadata."""
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
    assert attributes["model_runtime.latency_ms"] == 250.0
    assert any(event.name == "exception" for event in span.events)


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


def test_provider_name_cannot_be_blank(
    otel: tuple[Tracer, InMemorySpanExporter],
) -> None:
    """Invalid provider metadata fails before any request is observed."""
    tracer, _ = otel
    with pytest.raises(ValueError, match="provider_name must not be blank"):
        OTelTraceObserver(tracer, provider_name="  ")
