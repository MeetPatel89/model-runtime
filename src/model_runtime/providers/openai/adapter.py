"""Public OpenAI adapter assembled from reusable provider components."""

from __future__ import annotations

import openai
from openai.types.responses import Response, ResponseStreamEvent

from ...errors import ModelRuntimeError
from ...types import ModelCapabilities
from ..base import ProviderAdapter
from ..errors import StandardProviderErrorMapper
from ._types import OpenAIRequest
from .codec import OpenAICodec
from .errors import OpenAIErrorMetadataExtractor
from .transport import OpenAIClientLike, OpenAITransport


class OpenAIAdapter(ProviderAdapter[OpenAIRequest, Response, ResponseStreamEvent]):
    """OpenAI Responses API integration using the official async SDK."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        organization: str | None = None,
        project: str | None = None,
        client: OpenAIClientLike | None = None,
        capabilities: ModelCapabilities | None = None,
    ) -> None:
        if client is None:
            client = openai.AsyncOpenAI(
                api_key=api_key,
                base_url=base_url,
                organization=organization,
                project=project,
                max_retries=0,
            )
        elif any(
            option is not None for option in (api_key, base_url, organization, project)
        ):
            raise ValueError(
                "client cannot be combined with OpenAI credential or endpoint options"
            )

        self._openai_transport = OpenAITransport(client)
        super().__init__(
            transport=self._openai_transport,
            codec=OpenAICodec(),
            error_mapper=StandardProviderErrorMapper(OpenAIErrorMetadataExtractor()),
            capabilities=capabilities or self._default_capabilities(),
        )

    @property
    def client(self) -> OpenAIClientLike:
        """Underlying official or structurally compatible async client."""
        return self._openai_transport.client

    @staticmethod
    def translate_error(error: Exception) -> ModelRuntimeError:
        """Map an OpenAI SDK error to the public runtime taxonomy."""
        mapper = StandardProviderErrorMapper(OpenAIErrorMetadataExtractor())
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
                    "background",
                    "built_in_tools",
                    "conversation_state",
                    "logprobs",
                    "reasoning",
                    "service_tier",
                }
            ),
        )
