"""Offline lifecycle smoke tests for the optional aiohttp transport."""

from __future__ import annotations

import pytest
from anthropic import AsyncAnthropic
from anthropic import DefaultAioHttpClient as AnthropicAioHttpClient
from openai import AsyncOpenAI
from openai import DefaultAioHttpClient as OpenAIAioHttpClient

from model_runtime import AnthropicAdapter, OpenAIAdapter

pytest.importorskip(
    "httpx_aiohttp",
    reason="the optional aiohttp transport is not installed",
)


@pytest.mark.asyncio
async def test_aiohttp_sdk_clients_close_their_http_clients() -> None:
    """Both SDK contexts close their injected aiohttp-based HTTP clients."""
    openai_http_client = OpenAIAioHttpClient()
    assert not openai_http_client.is_closed

    async with AsyncOpenAI(
        api_key="test-key",
        http_client=openai_http_client,
        max_retries=0,
    ) as openai_client:
        openai_adapter = OpenAIAdapter(client=openai_client)

        assert openai_adapter.client is openai_client
        assert not openai_http_client.is_closed

    assert openai_http_client.is_closed

    anthropic_http_client = AnthropicAioHttpClient()
    assert not anthropic_http_client.is_closed

    async with AsyncAnthropic(
        api_key="test-key",
        http_client=anthropic_http_client,
        max_retries=0,
    ) as anthropic_client:
        anthropic_adapter = AnthropicAdapter(client=anthropic_client)

        assert anthropic_adapter.client is anthropic_client
        assert not anthropic_http_client.is_closed

    assert anthropic_http_client.is_closed
