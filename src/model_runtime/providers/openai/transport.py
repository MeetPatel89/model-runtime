"""Official OpenAI SDK transport with retries disabled by construction."""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable
from typing import Protocol, cast

import openai
from openai.types.chat import ChatCompletion, ChatCompletionChunk

from ...errors import InvalidRequestError, ProviderUnavailableError
from ._types import OpenAIRequest


class OpenAIChunkStream(Protocol):
    """The small portion of ``openai.AsyncStream`` used by the transport."""

    def __aiter__(self) -> AsyncIterator[ChatCompletionChunk]:
        """Iterate over Chat Completions chunks."""
        ...

    async def close(self) -> None:
        """Release the streaming HTTP response."""
        ...


type OpenAICreateResult = ChatCompletion | OpenAIChunkStream


class OpenAICompletionEndpoint(Protocol):
    """Injectable shape of ``client.chat.completions``."""

    def create(self, **kwargs: object) -> Awaitable[OpenAICreateResult]:
        """Create a completion or streaming response."""
        ...


class OpenAIChatResource(Protocol):
    """Injectable shape of ``client.chat``."""

    @property
    def completions(self) -> OpenAICompletionEndpoint:
        """Chat Completions endpoint."""
        ...


class OpenAIClient(Protocol):
    """Narrow structural client accepted for deterministic tests."""

    @property
    def chat(self) -> OpenAIChatResource:
        """Chat resource."""
        ...


type OpenAIClientLike = openai.AsyncOpenAI | OpenAIClient


class OpenAITransport:
    """Invoke Chat Completions and validate the response mode."""

    def __init__(self, client: OpenAIClientLike) -> None:
        self._client = client
        structural_client = cast(OpenAIClient, client)
        self._endpoint = structural_client.chat.completions

    @property
    def client(self) -> OpenAIClientLike:
        """Underlying official or structurally compatible async client."""
        return self._client

    async def complete(self, request: OpenAIRequest) -> ChatCompletion:
        """Execute one non-streaming SDK call."""
        if request.stream:
            raise InvalidRequestError(
                "completion transport received a streaming OpenAI request",
                provider="openai",
            )
        result = await self._endpoint.create(**request.as_kwargs())
        if not isinstance(result, ChatCompletion):
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
    ) -> AsyncIterator[ChatCompletionChunk]:
        """Execute one streaming SDK call and always close its HTTP response."""
        if not request.stream:
            raise InvalidRequestError(
                "stream transport received a non-streaming OpenAI request",
                provider="openai",
            )
        result = await self._endpoint.create(**request.as_kwargs())
        if isinstance(result, ChatCompletion):
            raise ProviderUnavailableError(
                "OpenAI returned a completion for a streaming request",
                retryable=False,
                provider="openai",
                details=result,
            )
        try:
            async for chunk in result:
                if not isinstance(chunk, ChatCompletionChunk):
                    raise ProviderUnavailableError(
                        "OpenAI yielded an invalid streaming chunk",
                        retryable=False,
                        provider="openai",
                        details=chunk,
                    )
                yield chunk
        finally:
            await result.close()
