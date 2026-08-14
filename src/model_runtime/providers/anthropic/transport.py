"""Official Anthropic SDK transport with retries disabled by construction."""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable
from types import TracebackType
from typing import Protocol, cast

from anthropic import AsyncAnthropic
from anthropic.lib.streaming import ParsedMessageStreamEvent
from anthropic.types import Message

from ...errors import InvalidRequestError, ProviderUnavailableError
from ._types import AnthropicRequest

type AnthropicStreamEvent = ParsedMessageStreamEvent[object]


class AnthropicMessageStream(Protocol):
    """The typed event iterator returned by the SDK stream helper."""

    def __aiter__(self) -> AsyncIterator[AnthropicStreamEvent]:
        """Iterate over typed Messages API events."""
        ...


class AnthropicStreamManager(Protocol):
    """Async context manager owning an Anthropic streaming response."""

    async def __aenter__(self) -> AnthropicMessageStream:
        """Open and return the streaming event source."""
        ...

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close the streaming response."""
        ...


class AnthropicMessagesEndpoint(Protocol):
    """Injectable shape of ``client.messages``."""

    def create(self, **kwargs: object) -> Awaitable[Message]:
        """Create one complete message."""
        ...

    def stream(self, **kwargs: object) -> AnthropicStreamManager:
        """Create a managed typed message stream."""
        ...


class AnthropicClient(Protocol):
    """Narrow structural client accepted for deterministic tests."""

    @property
    def messages(self) -> AnthropicMessagesEndpoint:
        """Messages API resource."""
        ...


type AnthropicClientLike = AsyncAnthropic | AnthropicClient


class AnthropicTransport:
    """Invoke the Messages API and validate the requested response mode."""

    def __init__(self, client: AnthropicClientLike) -> None:
        self._client = client
        structural_client = cast(AnthropicClient, client)
        self._endpoint = structural_client.messages

    @property
    def client(self) -> AnthropicClientLike:
        """Underlying official or structurally compatible async client."""
        return self._client

    async def complete(self, request: AnthropicRequest) -> Message:
        """Execute one non-streaming Messages API call."""
        if request.stream:
            raise InvalidRequestError(
                "completion transport received a streaming Anthropic request",
                provider="anthropic",
            )
        result = await self._endpoint.create(**request.as_kwargs())
        if not isinstance(result, Message):
            raise ProviderUnavailableError(
                "Anthropic returned an invalid non-streaming response",
                retryable=False,
                provider="anthropic",
                details=result,
            )
        return result

    async def stream(
        self,
        request: AnthropicRequest,
    ) -> AsyncIterator[AnthropicStreamEvent]:
        """Execute one streaming call with SDK-managed HTTP cleanup."""
        if not request.stream:
            raise InvalidRequestError(
                "stream transport received a non-streaming Anthropic request",
                provider="anthropic",
            )
        manager = self._endpoint.stream(**request.as_kwargs())
        async with manager as stream:
            async for event in stream:
                yield event
