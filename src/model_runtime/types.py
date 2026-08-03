"""Provider-independent request, response, and streaming value objects."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Literal


def _immutable_mapping(value: Mapping[str, Any] | None) -> Mapping[str, Any]:
    """Return a shallow, immutable copy without changing nested provider values."""
    return MappingProxyType(dict(value or {}))


class MessageRole(str, Enum):
    """Roles supported by normalized chat messages."""

    SYSTEM = "system"
    DEVELOPER = "developer"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass(frozen=True, slots=True)
class TextPart:
    """A text content part."""

    text: str
    type: Literal["text"] = field(default="text", init=False)

    def __post_init__(self) -> None:
        """Validate the text content."""
        if not isinstance(self.text, str):
            raise TypeError("text content must be a string")


@dataclass(frozen=True, slots=True)
class ImagePart:
    """An image referenced by URL or a data URL."""

    url: str
    detail: Literal["auto", "low", "high"] = "auto"
    type: Literal["image"] = field(default="image", init=False)

    def __post_init__(self) -> None:
        """Validate the image URL and requested detail level."""
        if not isinstance(self.url, str) or not self.url:
            raise ValueError("image URL cannot be empty")
        if self.detail not in {"auto", "low", "high"}:
            raise ValueError("image detail must be 'auto', 'low', or 'high'")

    @property
    def image_url(self) -> str:
        """The image URL using OpenAI-compatible naming."""
        return self.url


# Friendly aliases for callers that prefer content-oriented names.
TextContent = TextPart
ImageContent = ImagePart
type ContentPart = TextPart | ImagePart


@dataclass(frozen=True, slots=True)
class ToolCall:
    """A complete tool invocation requested by a model."""

    id: str
    name: str
    arguments: Mapping[str, Any] | str = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Freeze mapping arguments while accepting JSON strings unchanged."""
        if isinstance(self.arguments, Mapping):
            object.__setattr__(self, "arguments", _immutable_mapping(self.arguments))
        elif not isinstance(self.arguments, str):
            raise TypeError("tool call arguments must be a mapping or JSON string")

    @property
    def arguments_json(self) -> str:
        """Tool arguments as a compact JSON string."""
        if isinstance(self.arguments, str):
            return self.arguments
        return json.dumps(
            dict(self.arguments), separators=(",", ":"), ensure_ascii=False
        )


@dataclass(frozen=True, slots=True)
class Message:
    """A chat message made up of typed content parts."""

    role: MessageRole | str
    content: tuple[ContentPart, ...] | Sequence[ContentPart] | str = ()
    tool_calls: tuple[ToolCall, ...] | Sequence[ToolCall] = ()
    tool_call_id: str | None = None
    name: str | None = None

    def __post_init__(self) -> None:
        """Normalize role, content, and tool calls to immutable value objects."""
        try:
            role = (
                self.role
                if isinstance(self.role, MessageRole)
                else MessageRole(self.role)
            )
        except ValueError as exc:
            allowed = ", ".join(role.value for role in MessageRole)
            raise ValueError(
                f"unsupported message role {self.role!r}; expected one of {allowed}"
            ) from exc

        if isinstance(self.content, str):
            content: tuple[ContentPart, ...] = (TextPart(self.content),)
        else:
            content = tuple(self.content)
        if not all(isinstance(part, (TextPart, ImagePart)) for part in content):
            raise TypeError("message content must contain TextPart or ImagePart values")

        object.__setattr__(self, "role", role)
        object.__setattr__(self, "content", content)
        tool_calls = tuple(self.tool_calls)
        if not all(isinstance(call, ToolCall) for call in tool_calls):
            raise TypeError("message tool_calls must contain ToolCall values")
        object.__setattr__(self, "tool_calls", tool_calls)

    @classmethod
    def system(cls, text: str) -> Message:
        """Create a system message containing ``text``."""
        return cls(MessageRole.SYSTEM, text)

    @classmethod
    def developer(cls, text: str) -> Message:
        """Create a developer message containing ``text``."""
        return cls(MessageRole.DEVELOPER, text)

    @classmethod
    def user(cls, text: str) -> Message:
        """Create a user message containing ``text``."""
        return cls(MessageRole.USER, text)

    @classmethod
    def assistant(
        cls,
        text: str = "",
        *,
        tool_calls: Sequence[ToolCall] = (),
    ) -> Message:
        """Create an assistant message with optional tool calls."""
        content: str | tuple[ContentPart, ...] = text if text else ()
        return cls(MessageRole.ASSISTANT, content, tuple(tool_calls))

    @classmethod
    def tool(cls, text: str, *, tool_call_id: str, name: str | None = None) -> Message:
        """Create a tool-result message associated with a tool call."""
        return cls(MessageRole.TOOL, text, tool_call_id=tool_call_id, name=name)

    @property
    def text(self) -> str:
        """All text parts joined without altering their contents."""
        return "".join(part.text for part in self.content if isinstance(part, TextPart))


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    """A JSON-schema-described function exposed to a model."""

    name: str
    description: str | None = None
    parameters: Mapping[str, Any] = field(default_factory=dict)
    strict: bool | None = None

    def __post_init__(self) -> None:
        """Validate the tool name and freeze its parameter schema."""
        if not self.name:
            raise ValueError("tool name cannot be empty")
        object.__setattr__(self, "parameters", _immutable_mapping(self.parameters))


@dataclass(frozen=True, slots=True)
class ModelRequest:
    """A normalized request accepted by every chat-model adapter.

    ``provider_options`` is intentionally open-ended. Its values are passed to the
    selected adapter unchanged, allowing provider features to be used immediately.
    """

    messages: tuple[Message, ...] | Sequence[Message]
    tools: tuple[ToolDefinition, ...] | Sequence[ToolDefinition] = ()
    temperature: float | None = None
    max_output_tokens: int | None = None
    stop: tuple[str, ...] | Sequence[str] | str = ()
    timeout: float | None = None
    provider_options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate and normalize request collections and provider options."""
        if self.timeout is not None and self.timeout <= 0:
            raise ValueError("timeout must be greater than zero")
        if self.max_output_tokens is not None and self.max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be greater than zero")
        messages = tuple(self.messages)
        tools = tuple(self.tools)
        if not all(isinstance(message, Message) for message in messages):
            raise TypeError("messages must contain Message values")
        if not all(isinstance(tool, ToolDefinition) for tool in tools):
            raise TypeError("tools must contain ToolDefinition values")
        stop = (self.stop,) if isinstance(self.stop, str) else tuple(self.stop)
        if not all(isinstance(item, str) for item in stop):
            raise TypeError("stop must contain strings")
        object.__setattr__(self, "messages", messages)
        object.__setattr__(self, "tools", tools)
        object.__setattr__(self, "stop", stop)
        object.__setattr__(
            self, "provider_options", _immutable_mapping(self.provider_options)
        )

    @classmethod
    def from_text(cls, text: str, **kwargs: Any) -> ModelRequest:
        """Create a request with one user message containing ``text``."""
        return cls(messages=(Message.user(text),), **kwargs)


@dataclass(frozen=True, slots=True)
class Usage:
    """Input, output, and cached token counts for a model invocation."""

    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0

    def __post_init__(self) -> None:
        """Reject negative token counts."""
        if min(self.input_tokens, self.output_tokens, self.cached_tokens) < 0:
            raise ValueError("token counts cannot be negative")

    @property
    def total_tokens(self) -> int:
        """Input and output tokens; cached tokens are an input subset."""
        # Cached tokens are a subset of input tokens for the supported providers.
        return self.input_tokens + self.output_tokens

    def __add__(self, other: Usage) -> Usage:
        """Return token usage aggregated with another usage value."""
        if not isinstance(other, Usage):
            return NotImplemented
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cached_tokens=self.cached_tokens + other.cached_tokens,
        )


class FinishReason(str, Enum):
    """Reasons a model response or stream finished."""

    STOP = "stop"
    LENGTH = "length"
    TOOL_CALLS = "tool_calls"
    CONTENT_FILTER = "content_filter"
    ERROR = "error"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class ModelResponse:
    """A normalized completed response returned by a chat model."""

    message: Message
    usage: Usage = field(default_factory=Usage)
    finish_reason: FinishReason = FinishReason.UNKNOWN
    raw: Any = None
    model: str | None = None

    def __post_init__(self) -> None:
        """Normalize unknown finish-reason strings to ``UNKNOWN``."""
        if not isinstance(self.finish_reason, FinishReason):
            try:
                object.__setattr__(
                    self, "finish_reason", FinishReason(self.finish_reason)
                )
            except ValueError:
                object.__setattr__(self, "finish_reason", FinishReason.UNKNOWN)

    @property
    def text(self) -> str:
        """The response message's text content."""
        return self.message.text

    @property
    def tool_calls(self) -> tuple[ToolCall, ...]:
        """Tool calls requested by the response message."""
        return self.message.tool_calls

    @property
    def content(self) -> tuple[ContentPart, ...]:
        """Typed content parts from the response message."""
        return self.message.content


@dataclass(frozen=True, slots=True)
class TextDelta:
    """An incremental text fragment emitted during streaming."""

    delta: str

    @property
    def text(self) -> str:
        """The emitted text fragment."""
        return self.delta


@dataclass(frozen=True, slots=True)
class ToolCallDelta:
    """An incremental update to one streaming tool call."""

    index: int
    id: str | None = None
    name: str | None = None
    arguments_delta: str = ""

    @property
    def arguments(self) -> str:
        """The incremental JSON arguments fragment."""
        return self.arguments_delta


@dataclass(frozen=True, slots=True)
class StreamEnd:
    """The final event of a stream, including its assembled response."""

    response: ModelResponse
    usage: Usage | None = None

    def __post_init__(self) -> None:
        """Use the response usage when the stream did not provide one separately."""
        if self.usage is None:
            object.__setattr__(self, "usage", self.response.usage)

    @property
    def final_response(self) -> ModelResponse:
        """The assembled final model response."""
        return self.response


type StreamEvent = TextDelta | ToolCallDelta | StreamEnd


@dataclass(frozen=True, slots=True)
class ModelCapabilities:
    """Features and limits advertised by a model adapter."""

    tools: bool = False
    vision: bool = False
    structured_output: bool = False
    streaming: bool = True
    max_context_tokens: int | None = None
    provider_features: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        """Validate the context limit and freeze provider feature names."""
        if self.max_context_tokens is not None and self.max_context_tokens <= 0:
            raise ValueError("max_context_tokens must be greater than zero")
        object.__setattr__(self, "provider_features", frozenset(self.provider_features))

    @property
    def supports_tools(self) -> bool:
        """Whether the model supports tool calls."""
        return self.tools

    @property
    def supports_vision(self) -> bool:
        """Whether the model supports image content."""
        return self.vision

    @property
    def supports_structured_output(self) -> bool:
        """Whether the model supports structured output."""
        return self.structured_output
