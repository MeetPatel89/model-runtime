"""Provider-independent request, response, and streaming value objects."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

from .json_types import JsonObject, immutable_json_object


def _empty_json_object() -> JsonObject:
    """Return a precisely typed empty JSON object for dataclass defaults."""
    return {}


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
    arguments: JsonObject | str = field(default_factory=_empty_json_object)

    def __post_init__(self) -> None:
        """Freeze mapping arguments while accepting JSON strings unchanged."""
        if isinstance(self.arguments, Mapping):
            object.__setattr__(self, "arguments", immutable_json_object(self.arguments))
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


@dataclass(frozen=True, slots=True, init=False)
class Message:
    """A chat message made up of typed content parts."""

    role: MessageRole
    content: tuple[ContentPart, ...]
    tool_calls: tuple[ToolCall, ...]
    tool_call_id: str | None = None
    name: str | None = None

    def __init__(
        self,
        role: MessageRole | str,
        content: Sequence[ContentPart] | str = (),
        tool_calls: Sequence[ToolCall] = (),
        tool_call_id: str | None = None,
        name: str | None = None,
    ) -> None:
        """Normalize flexible inputs into canonical immutable attributes."""
        try:
            normalized_role = (
                role if isinstance(role, MessageRole) else MessageRole(role)
            )
        except ValueError as exc:
            allowed = ", ".join(member.value for member in MessageRole)
            raise ValueError(
                f"unsupported message role {role!r}; expected one of {allowed}"
            ) from exc

        if isinstance(content, str):
            normalized_content: tuple[ContentPart, ...] = (TextPart(content),)
        else:
            normalized_content = tuple(content)
        if not all(
            isinstance(part, TextPart | ImagePart) for part in normalized_content
        ):
            raise TypeError("message content must contain TextPart or ImagePart values")

        normalized_tool_calls = tuple(tool_calls)
        if not all(isinstance(call, ToolCall) for call in normalized_tool_calls):
            raise TypeError("message tool_calls must contain ToolCall values")
        object.__setattr__(self, "role", normalized_role)
        object.__setattr__(self, "content", normalized_content)
        object.__setattr__(self, "tool_calls", normalized_tool_calls)
        object.__setattr__(self, "tool_call_id", tool_call_id)
        object.__setattr__(self, "name", name)

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
    parameters: JsonObject = field(default_factory=_empty_json_object)
    strict: bool | None = None

    def __post_init__(self) -> None:
        """Validate the tool name and freeze its parameter schema."""
        if not self.name:
            raise ValueError("tool name cannot be empty")
        object.__setattr__(self, "parameters", immutable_json_object(self.parameters))


@dataclass(frozen=True, slots=True, init=False)
class ModelRequest:
    """A normalized request accepted by every chat-model adapter.

    ``provider_options`` is intentionally open-ended. Its values are passed to the
    selected adapter unchanged, allowing provider features to be used immediately.
    """

    messages: tuple[Message, ...]
    tools: tuple[ToolDefinition, ...]
    temperature: float | None
    max_output_tokens: int | None
    stop: tuple[str, ...]
    timeout: float | None
    provider_options: JsonObject

    def __init__(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolDefinition] = (),
        temperature: float | None = None,
        max_output_tokens: int | None = None,
        stop: Sequence[str] | str = (),
        timeout: float | None = None,
        provider_options: JsonObject | None = None,
    ) -> None:
        """Validate inputs and store one canonical immutable representation."""
        if timeout is not None and timeout <= 0:
            raise ValueError("timeout must be greater than zero")
        if max_output_tokens is not None and max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be greater than zero")
        normalized_messages = tuple(messages)
        normalized_tools = tuple(tools)
        if not all(isinstance(message, Message) for message in normalized_messages):
            raise TypeError("messages must contain Message values")
        if not all(isinstance(tool, ToolDefinition) for tool in normalized_tools):
            raise TypeError("tools must contain ToolDefinition values")
        normalized_stop = (stop,) if isinstance(stop, str) else tuple(stop)
        if not all(isinstance(item, str) for item in normalized_stop):
            raise TypeError("stop must contain strings")
        object.__setattr__(self, "messages", normalized_messages)
        object.__setattr__(self, "tools", normalized_tools)
        object.__setattr__(self, "temperature", temperature)
        object.__setattr__(self, "max_output_tokens", max_output_tokens)
        object.__setattr__(self, "stop", normalized_stop)
        object.__setattr__(self, "timeout", timeout)
        object.__setattr__(
            self,
            "provider_options",
            immutable_json_object(provider_options),
        )

    @classmethod
    def from_text(
        cls,
        text: str,
        *,
        tools: Sequence[ToolDefinition] = (),
        temperature: float | None = None,
        max_output_tokens: int | None = None,
        stop: Sequence[str] | str = (),
        timeout: float | None = None,
        provider_options: JsonObject | None = None,
    ) -> ModelRequest:
        """Create a request with one user message containing ``text``."""
        return cls(
            messages=(Message.user(text),),
            tools=tools,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            stop=stop,
            timeout=timeout,
            provider_options=provider_options or {},
        )


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
    raw: object | None = None
    model: str | None = None

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


@dataclass(frozen=True, slots=True, init=False)
class StreamEnd:
    """The final event of a stream, including its assembled response."""

    response: ModelResponse
    usage: Usage

    def __init__(self, response: ModelResponse, usage: Usage | None = None) -> None:
        """Store explicit usage or the final response's canonical usage."""
        object.__setattr__(self, "response", response)
        object.__setattr__(self, "usage", response.usage if usage is None else usage)

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
