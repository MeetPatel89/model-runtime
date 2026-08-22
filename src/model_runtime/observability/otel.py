"""OpenTelemetry tracing at the logical model-request boundary."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from json import dumps
from os import getenv
from types import MappingProxyType

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
from ..types import Message, ModelRequest, ModelResponse, TextPart, Usage

_CAPTURE_MESSAGE_CONTENT_ENV = "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"
_LANGFUSE_ATTRIBUTE_MAX_LENGTH = 200


@dataclass(frozen=True, slots=True)
class LangfuseTraceAttributes:
    """Trace-level Langfuse dimensions copied onto each runtime span.

    Langfuse queries observations directly, so session, user, tag, and trace
    metadata values must be present on every span where they should be
    filterable. The value object makes that propagation explicit while keeping
    request-specific context outside :class:`ModelRuntime`.
    """

    session_id: str | None = None
    user_id: str | None = None
    tags: Sequence[str] = ()
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Normalize collections and reject values Langfuse would discard."""
        self._validate_optional_identifier("session_id", self.session_id)
        self._validate_optional_identifier("user_id", self.user_id)

        if isinstance(self.tags, str):
            raise TypeError("tags must be a sequence of strings")
        normalized_tags = tuple(self.tags)
        for tag in normalized_tags:
            self._validate_text("tag", tag)

        if not isinstance(self.metadata, Mapping):
            raise TypeError("metadata must be a mapping of strings to strings")
        normalized_metadata = dict(self.metadata)
        for key, value in normalized_metadata.items():
            self._validate_text("metadata key", key, ascii_only=True)
            self._validate_text(f"metadata[{key!r}]", value)

        object.__setattr__(self, "tags", normalized_tags)
        object.__setattr__(
            self,
            "metadata",
            MappingProxyType(normalized_metadata),
        )

    def as_otel_attributes(self) -> dict[str, AttributeValue]:
        """Return Langfuse's raw-OTel trace attributes."""
        attributes: dict[str, AttributeValue] = {}
        if self.session_id is not None:
            attributes["langfuse.session.id"] = self.session_id
        if self.user_id is not None:
            attributes["langfuse.user.id"] = self.user_id
        if self.tags:
            attributes["langfuse.trace.tags"] = tuple(self.tags)
        attributes.update(
            {
                f"langfuse.trace.metadata.{key}": value
                for key, value in self.metadata.items()
            }
        )
        return attributes

    @staticmethod
    def _validate_optional_identifier(name: str, value: str | None) -> None:
        if value is not None:
            LangfuseTraceAttributes._validate_text(name, value, ascii_only=True)

    @staticmethod
    def _validate_text(name: str, value: str, *, ascii_only: bool = False) -> None:
        if not isinstance(value, str):
            raise TypeError(f"{name} must be a string")
        if not value:
            raise ValueError(f"{name} must not be empty")
        if ascii_only and not value.isascii():
            raise ValueError(f"{name} must contain only US-ASCII characters")
        if len(value) > _LANGFUSE_ATTRIBUTE_MAX_LENGTH:
            raise ValueError(
                f"{name} must contain at most "
                f"{_LANGFUSE_ATTRIBUTE_MAX_LENGTH} characters"
            )


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

    def __init__(
        self,
        tracer: Tracer,
        *,
        provider_name: str,
        capture_message_content: bool | None = None,
        langfuse_trace: LangfuseTraceAttributes | None = None,
    ) -> None:
        if not provider_name.strip():
            raise ValueError("provider_name must not be blank")
        if capture_message_content is not None and not isinstance(
            capture_message_content, bool
        ):
            raise TypeError("capture_message_content must be a bool or None")
        self._tracer = tracer
        self._provider_name = provider_name.strip()
        self._capture_message_content = (
            _capture_message_content_from_env()
            if capture_message_content is None
            else capture_message_content
        )
        if langfuse_trace is not None and not isinstance(
            langfuse_trace, LangfuseTraceAttributes
        ):
            raise TypeError("langfuse_trace must be LangfuseTraceAttributes or None")
        self._langfuse_trace = langfuse_trace
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
        if self._langfuse_trace is not None:
            attributes.update(self._langfuse_trace.as_otel_attributes())
            attributes["langfuse.observation.type"] = "generation"
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
            if self._capture_message_content and span.is_recording():
                span.set_attribute(
                    "gen_ai.input.messages",
                    _serialize_input_messages(request),
                )
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
        if self._capture_message_content and active.span.is_recording():
            attributes["gen_ai.output.messages"] = _serialize_output_message(response)

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
            "model_runtime.latency_ms": max(latency_seconds, 0.0) * 1000,
        }
        attributes.update(self._error_attributes(error))
        try:
            active.span.set_attributes(attributes)
            active.span.record_exception(error, escaped=True)
            active.span.set_status(
                Status(StatusCode.ERROR, description=type(error).__qualname__)
            )
        finally:
            self._detach_and_end(active)

    def on_retry(
        self,
        model_id: str,
        error: ModelRuntimeError,
        attempt: int,
        delay_seconds: float,
    ) -> None:
        """Add an event for a failed attempt without ending the logical span."""
        active = self._current_active_span()
        if active is None:
            return

        attributes: dict[str, AttributeValue] = {
            "gen_ai.request.model": model_id,
            "model_runtime.retry.attempt": attempt,
            "model_runtime.retry.next_attempt": attempt + 1,
            "model_runtime.retry.delay_ms": max(delay_seconds, 0.0) * 1000,
        }
        attributes.update(self._error_attributes(error))
        active.span.add_event("model_runtime.retry", attributes=attributes)

    def _current_active_span(self) -> _ActiveSpan | None:
        active_spans = self._active_spans.get()
        return active_spans[-1] if active_spans else None

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

    @staticmethod
    def _error_attributes(
        error: ModelRuntimeError,
    ) -> dict[str, AttributeValue]:
        attributes: dict[str, AttributeValue] = {
            "error.type": type(error).__qualname__,
            "model_runtime.error.retryable": error.retryable,
        }
        if error.provider is not None:
            attributes["model_runtime.error.provider"] = error.provider
        if error.status_code is not None:
            attributes["model_runtime.error.status_code"] = error.status_code
        return attributes


def _capture_message_content_from_env() -> bool:
    """Return true only for the OTel-specified case-insensitive ``true`` value."""
    return getenv(_CAPTURE_MESSAGE_CONTENT_ENV, "").casefold() == "true"


def _serialize_input_messages(request: ModelRequest) -> str:
    """Serialize normalized input text using the GenAI message JSON schema."""
    messages = [
        {
            "role": message.role.value,
            "parts": _text_parts(message),
        }
        for message in request.messages
    ]
    return dumps(messages, ensure_ascii=False, separators=(",", ":"))


def _serialize_output_message(response: ModelResponse) -> str:
    """Serialize normalized response text using the GenAI message JSON schema."""
    messages = [
        {
            "role": response.message.role.value,
            "parts": _text_parts(response.message),
            "finish_reason": response.finish_reason.value,
        }
    ]
    return dumps(messages, ensure_ascii=False, separators=(",", ":"))


def _text_parts(message: Message) -> list[dict[str, str]]:
    """Return normalized text while omitting images and tool-call structures."""
    return [
        {"type": "text", "content": part.text}
        for part in message.content
        if isinstance(part, TextPart)
    ]
