"""The small contract implemented by every provider adapter."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol, runtime_checkable

from .types import ModelCapabilities, ModelRequest, ModelResponse, StreamEvent


@runtime_checkable
class ChatModel(Protocol):
    """Contract implemented by a chat-model provider adapter."""

    @property
    def capabilities(self) -> ModelCapabilities:
        """Capabilities common enough for callers and routers to inspect."""
        ...

    async def complete(self, model_id: str, request: ModelRequest) -> ModelResponse:
        """Return one complete model response."""
        ...

    def stream(
        self, model_id: str, request: ModelRequest
    ) -> AsyncIterator[StreamEvent]:
        """Yield normalized deltas followed by exactly one ``StreamEnd``."""
        ...


@runtime_checkable
class ModelCatalog(Protocol):
    """Optional model-discovery behavior exposed by supporting adapters."""

    async def list_models(self) -> list[str]:
        """Return provider model identifiers available to the client."""
        ...
