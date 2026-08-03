"""Reusable orchestration for strongly typed provider adapters."""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import Generic, Protocol, TypeVar

from ..errors import ModelRuntimeError
from ..types import (
    ModelCapabilities,
    ModelRequest,
    ModelResponse,
    StreamEnd,
    StreamEvent,
    TextDelta,
    ToolCallDelta,
)

type StreamDelta = TextDelta | ToolCallDelta

_RequestT_contra = TypeVar("_RequestT_contra", contravariant=True)
_ResponseT_co = TypeVar("_ResponseT_co", covariant=True)
_ChunkT_co = TypeVar("_ChunkT_co", covariant=True)
_ChunkT_contra = TypeVar("_ChunkT_contra", contravariant=True)
_RequestT_co = TypeVar("_RequestT_co", covariant=True)
_ResponseT_contra = TypeVar("_ResponseT_contra", contravariant=True)

RequestT = TypeVar("RequestT")
ResponseT = TypeVar("ResponseT")
ChunkT = TypeVar("ChunkT")


class ProviderTransport(Protocol[_RequestT_contra, _ResponseT_co, _ChunkT_co]):
    """Execute already encoded provider requests against an external SDK."""

    async def complete(self, request: _RequestT_contra) -> _ResponseT_co:
        """Execute a non-streaming request."""
        ...

    def stream(self, request: _RequestT_contra) -> AsyncIterator[_ChunkT_co]:
        """Yield provider-native chunks and own transport cleanup."""
        ...


class StreamDecoder(Protocol[_ChunkT_contra]):
    """Accumulate one provider stream while producing normalized deltas."""

    def feed(self, chunk: _ChunkT_contra) -> Sequence[StreamDelta]:
        """Consume one native chunk and return zero or more public deltas."""
        ...

    def finish(self) -> StreamEnd:
        """Build the single terminal event after the source stream ends."""
        ...


class ProviderCodec(Protocol[_RequestT_co, _ResponseT_contra, _ChunkT_contra]):
    """Translate between normalized values and one provider's native values."""

    def encode_request(
        self,
        model_id: str,
        request: ModelRequest,
        *,
        stream: bool,
    ) -> _RequestT_co:
        """Encode a normalized request for the provider transport."""
        ...

    def decode_response(
        self,
        response: _ResponseT_contra,
        *,
        fallback_model: str,
    ) -> ModelResponse:
        """Decode one provider-native completion."""
        ...

    def stream_decoder(self, *, fallback_model: str) -> StreamDecoder[_ChunkT_contra]:
        """Create isolated state for decoding one stream."""
        ...


class ProviderErrorMapper(Protocol):
    """Translate provider SDK exceptions into the public error taxonomy."""

    def translate(self, error: Exception) -> ModelRuntimeError:
        """Return the normalized equivalent of ``error``."""
        ...


class ProviderAdapter(Generic[RequestT, ResponseT, ChunkT]):
    """Implement ``ChatModel`` once using typed provider components.

    A provider integration supplies transport, translation, and error metadata
    behavior. Invocation flow, exception boundaries, stream finalization, and
    capabilities are kept here so new integrations do not duplicate them.
    """

    def __init__(
        self,
        *,
        transport: ProviderTransport[RequestT, ResponseT, ChunkT],
        codec: ProviderCodec[RequestT, ResponseT, ChunkT],
        error_mapper: ProviderErrorMapper,
        capabilities: ModelCapabilities,
    ) -> None:
        self._transport = transport
        self._codec = codec
        self._error_mapper = error_mapper
        self._capabilities = capabilities

    @property
    def capabilities(self) -> ModelCapabilities:
        """Features advertised by this adapter."""
        return self._capabilities

    async def complete(self, model_id: str, request: ModelRequest) -> ModelResponse:
        """Encode, execute, and decode one provider completion."""
        try:
            encoded = self._codec.encode_request(model_id, request, stream=False)
            response = await self._transport.complete(encoded)
            return self._codec.decode_response(response, fallback_model=model_id)
        except ModelRuntimeError:
            raise
        except Exception as exc:
            error = self._error_mapper.translate(exc)
            raise error from exc

    async def stream(
        self,
        model_id: str,
        request: ModelRequest,
    ) -> AsyncIterator[StreamEvent]:
        """Encode and decode one provider stream into normalized events."""
        try:
            encoded = self._codec.encode_request(model_id, request, stream=True)
            decoder = self._codec.stream_decoder(fallback_model=model_id)
            async for chunk in self._transport.stream(encoded):
                for event in decoder.feed(chunk):
                    yield event
            yield decoder.finish()
        except ModelRuntimeError:
            raise
        except Exception as exc:
            error = self._error_mapper.translate(exc)
            raise error from exc
