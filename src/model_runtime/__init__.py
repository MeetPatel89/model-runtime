"""Public API for the provider-independent model runtime."""

from .errors import (
    AuthError,
    ContentFilterError,
    InvalidRequestError,
    ModelRuntimeError,
    ProviderUnavailableError,
    RateLimitError,
    RequestTimeout,
)
from .model import ChatModel
from .providers import OpenAIAdapter
from .retry import RetryPolicy
from .router import ModelRoute, ModelRouter, RouteSelector
from .runtime import ModelRuntime
from .tracing import NoOpTraceObserver, TraceObserver
from .types import (
    ContentPart,
    FinishReason,
    ImageContent,
    ImagePart,
    Message,
    MessageRole,
    ModelCapabilities,
    ModelRequest,
    ModelResponse,
    StreamEnd,
    StreamEvent,
    TextContent,
    TextDelta,
    TextPart,
    ToolCall,
    ToolCallDelta,
    ToolDefinition,
    Usage,
)

__all__ = [
    "AuthError",
    "ChatModel",
    "ContentFilterError",
    "ContentPart",
    "FinishReason",
    "ImageContent",
    "ImagePart",
    "InvalidRequestError",
    "Message",
    "MessageRole",
    "ModelCapabilities",
    "ModelRequest",
    "ModelResponse",
    "ModelRoute",
    "ModelRouter",
    "ModelRuntime",
    "ModelRuntimeError",
    "NoOpTraceObserver",
    "OpenAIAdapter",
    "ProviderUnavailableError",
    "RateLimitError",
    "RequestTimeout",
    "RetryPolicy",
    "RouteSelector",
    "StreamEnd",
    "StreamEvent",
    "TextContent",
    "TextDelta",
    "TextPart",
    "ToolCall",
    "ToolCallDelta",
    "ToolDefinition",
    "TraceObserver",
    "Usage",
]
