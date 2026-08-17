"""OpenTelemetry tracing at the logical model-request boundary."""

from __future__ import annotations

from contextvars import ContextVar, Token
from dataclasses import dataclass

from opentelemetry.context import Context, attach, detach
from opentelemetry.trace import (
    Span,
    SpanKind,
    Status,
    StatusCode,
    Tracer,
    set_span_in_context,
)
from opentelemetry.util.types import AttributeValue

from ..errors import ModelRuntimeError
from ..types import ModelRequest, ModelResponse, Usage


@dataclass(frozen=True, slots=True)
class _ActiveSpan:
    """A model-call span and the token used to make it current."""

    span: Span
    context_token: Token[Context]


class OTelTraceObserver:
    """Record each logical runtime request as one OpenTelemetry client span.

    The observer owns no global tracer provider. Applications inject a tracer and
    retain responsibility for processor, exporter, and provider shutdown. Active
    spans are held per async context so concurrent tasks cannot finish each
    other's spans.
    """

    def __init__(self, tracer: Tracer, *, provider_name: str) -> None:
        if not provider_name.strip():
            raise ValueError("provider_name must not be blank")
        self._tracer = tracer
        self._provider_name = provider_name.strip()
        self._active_spans: ContextVar[tuple[_ActiveSpan, ...]] = ContextVar(
            "model_runtime_otel_active_spans", default=()
        )

    def on_request(self, model_id: str, request: ModelRequest) -> None:
        """Start and activate one span for a logical model request."""
        attributes: dict[str, AttributeValue] = {
            "gen_ai.operation.name": "chat",
            "gen_ai.provider.name": self._provider_name,
            "gen_ai.request.model": model_id,
        }
        if request.temperature is not None:
            attributes["gen_ai.request.temperature"] = request.temperature
        if request.max_output_tokens is not None:
            attributes["gen_ai.request.max_tokens"] = request.max_output_tokens

        span = self._tracer.start_span(
            f"chat {model_id}",
            kind=SpanKind.CLIENT,
            attributes=attributes,
        )
        try:
            context_token = attach(set_span_in_context(span))
        except Exception:
            span.end()
            raise
        active = _ActiveSpan(span, context_token)
        self._active_spans.set((*self._active_spans.get(), active))

    def on_response(
        self,
        model_id: str,
        response: ModelResponse,
        latency_seconds: float,
        usage: Usage,
    ) -> None:
        """Annotate and end the current model-request span successfully."""
        active = self._take_active_span()
        if active is None:
            return

        attributes: dict[str, AttributeValue] = {
            "gen_ai.request.model": model_id,
            "gen_ai.response.finish_reasons": (response.finish_reason.value,),
            "gen_ai.usage.input_tokens": usage.input_tokens,
            "gen_ai.usage.output_tokens": usage.output_tokens,
            "model_runtime.latency_ms": max(latency_seconds, 0.0) * 1000,
        }
        if response.model is not None:
            attributes["gen_ai.response.model"] = response.model
        if usage.cached_tokens:
            attributes["gen_ai.usage.cache_read.input_tokens"] = usage.cached_tokens

        try:
            active.span.set_attributes(attributes)
            active.span.set_status(Status(StatusCode.OK))
        finally:
            self._detach_and_end(active)

    def on_error(
        self,
        model_id: str,
        error: ModelRuntimeError,
        latency_seconds: float,
    ) -> None:
        """Record the terminal error and end the current model-request span."""
        active = self._take_active_span()
        if active is None:
            return

        attributes: dict[str, AttributeValue] = {
            "gen_ai.request.model": model_id,
            "error.type": type(error).__qualname__,
            "model_runtime.latency_ms": max(latency_seconds, 0.0) * 1000,
        }
        try:
            active.span.set_attributes(attributes)
            active.span.record_exception(error, escaped=True)
            active.span.set_status(
                Status(StatusCode.ERROR, description=type(error).__qualname__)
            )
        finally:
            self._detach_and_end(active)

    def _take_active_span(self) -> _ActiveSpan | None:
        active_spans = self._active_spans.get()
        if not active_spans:
            return None
        self._active_spans.set(active_spans[:-1])
        return active_spans[-1]

    @staticmethod
    def _detach_and_end(active: _ActiveSpan) -> None:
        try:
            detach(active.context_token)
        finally:
            active.span.end()
