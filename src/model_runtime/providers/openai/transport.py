"""Official OpenAI SDK transport with retries disabled by construction."""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable
from typing import Protocol, cast

import openai
from openai.types.responses import Response, ResponseStreamEvent

from ...errors import InvalidRequestError, ProviderUnavailableError
from ._types import OpenAIRequest


class OpenAIResponseStream(Protocol):
    """The small portion of ``openai.AsyncStream`` used by the transport."""

    def __aiter__(self) -> AsyncIterator[ResponseStreamEvent]:
        """Iterate over typed Responses API events."""
        ...

    async def close(self) -> None:
        """Release the streaming HTTP response."""
        ...


type OpenAICreateResult = Response | OpenAIResponseStream


class OpenAIResponsesEndpoint(Protocol):
    """Injectable shape of ``client.responses``."""

    def create(self, **kwargs: object) -> Awaitable[OpenAICreateResult]:
        """Create a response or streaming event source."""
        ...


class OpenAIClient(Protocol):
    """Narrow structural client accepted for deterministic tests."""

    @property
    def responses(self) -> OpenAIResponsesEndpoint:
        """Responses API resource."""
        ...


type OpenAIClientLike = openai.AsyncOpenAI | OpenAIClient


class OpenAITransport:
    """Invoke the Responses API and validate the response mode."""

    def __init__(self, client: OpenAIClientLike) -> None:
        self._client = client
        structural_client = cast(OpenAIClient, client)
        self._endpoint = structural_client.responses

    @property
    def client(self) -> OpenAIClientLike:
        """Underlying official or structurally compatible async client."""
        return self._client

    async def complete(self, request: OpenAIRequest) -> Response:
        """Execute one non-streaming Responses API call."""
        if request.stream:
            raise InvalidRequestError(
                "completion transport received a streaming OpenAI request",
                provider="openai",
            )
        result = await self._endpoint.create(**request.as_kwargs())
        if not isinstance(result, Response):
            raise ProviderUnavailableError(
                "OpenAI returned a stream for a non-streaming request",
                retryable=False,
                provider="openai",
                details=result,
            )
        return result

    async def stream(
        self,
        request: OpenAIRequest,
    ) -> AsyncIterator[ResponseStreamEvent]:
        """Execute one streaming call and always close its HTTP response."""
        if not request.stream:
            raise InvalidRequestError(
                "stream transport received a non-streaming OpenAI request",
                provider="openai",
            )
        result = await self._endpoint.create(**request.as_kwargs())
        if isinstance(result, Response):
            raise ProviderUnavailableError(
                "OpenAI returned a response for a streaming request",
                retryable=False,
                provider="openai",
                details=result,
            )
        try:
            async for event in result:
                yield event
        finally:
            await result.close()
