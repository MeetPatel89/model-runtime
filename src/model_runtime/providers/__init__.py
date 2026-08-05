"""Reusable provider boundaries and built-in integrations."""

from .anthropic import AnthropicAdapter, AnthropicProviderOptions
from .base import (
    ProviderAdapter,
    ProviderCodec,
    ProviderErrorMapper,
    ProviderTransport,
    StreamDecoder,
    StreamDelta,
)
from .errors import (
    ErrorMetadataExtractor,
    ProviderErrorKind,
    ProviderErrorMetadata,
    StandardProviderErrorMapper,
)
from .openai import OpenAIAdapter, OpenAIProviderOptions

__all__ = [
    "AnthropicAdapter",
    "AnthropicProviderOptions",
    "ErrorMetadataExtractor",
    "OpenAIAdapter",
    "OpenAIProviderOptions",
    "ProviderAdapter",
    "ProviderCodec",
    "ProviderErrorKind",
    "ProviderErrorMapper",
    "ProviderErrorMetadata",
    "ProviderTransport",
    "StandardProviderErrorMapper",
    "StreamDecoder",
    "StreamDelta",
]
