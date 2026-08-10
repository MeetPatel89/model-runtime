"""Public Anthropic adapter assembled from reusable provider components."""

from __future__ import annotations

import os

from anthropic import AsyncAnthropic

from ...errors import ModelRuntimeError
from ...types import ModelCapabilities
from ..base import ProviderAdapter
from ..errors import StandardProviderErrorMapper
from .codec import AnthropicCodec
from .errors import AnthropicErrorMetadataExtractor
from .transport import AnthropicClientLike, AnthropicTransport


class AnthropicAdapter(ProviderAdapter):
    """Anthropic Messages API integration using the official async SDK."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        auth_token: str | None = None,
        base_url: str | None = None,
        client: AnthropicClientLike | None = None,
        default_max_output_tokens: int = 1024,
        capabilities: ModelCapabilities | None = None,
    ) -> None:
        if default_max_output_tokens <= 0:
            raise ValueError("default_max_output_tokens must be greater than zero")
        if client is None:
            client = AsyncAnthropic(
                api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"),
                auth_token=auth_token,
                base_url=base_url,
                max_retries=0,
            )
        elif any(option is not None for option in (api_key, auth_token, base_url)):
            raise ValueError(
                "client cannot be combined with Anthropic credential or "
                "endpoint options"
            )

        self._anthropic_transport = AnthropicTransport(client)
        super().__init__(
            transport=self._anthropic_transport,
            codec=AnthropicCodec(
                default_max_output_tokens=default_max_output_tokens,
            ),
            error_mapper=StandardProviderErrorMapper(AnthropicErrorMetadataExtractor()),
            capabilities=capabilities or self._default_capabilities(),
        )

    @property
    def client(self) -> AnthropicClientLike:
        """Underlying official or structurally compatible async client."""
        return self._anthropic_transport.client

    @staticmethod
    def translate_error(error: Exception) -> ModelRuntimeError:
        """Map an Anthropic SDK error to the public runtime taxonomy."""
        mapper = StandardProviderErrorMapper(AnthropicErrorMetadataExtractor())
        return mapper.translate(error)

    _translate_error = translate_error

    @staticmethod
    def _default_capabilities() -> ModelCapabilities:
        return ModelCapabilities(
            tools=True,
            vision=True,
            structured_output=True,
            streaming=True,
            provider_features=frozenset(
                {
                    "built_in_tools",
                    "effort",
                    "mid_conversation_system",
                    "prompt_caching",
                    "service_tier",
                    "thinking",
                }
            ),
        )
