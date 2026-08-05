"""Typed translation between normalized values and Anthropic Messages."""

from __future__ import annotations

import base64
import binascii
from collections.abc import Mapping, Sequence
from typing import Literal, cast

import anthropic.types as anthropic_types
from anthropic.lib.streaming import ParsedMessageStopEvent
from anthropic.types import (
    Base64ImageSourceParam,
    ContentBlockParam,
    ImageBlockParam,
    InputJSONDelta,
    MessageParam,
    RawContentBlockDeltaEvent,
    RawContentBlockStartEvent,
    TextBlockParam,
    ToolParam,
    ToolResultBlockParam,
    ToolUseBlock,
    ToolUseBlockParam,
    URLImageSourceParam,
)

from ...errors import InvalidRequestError, ProviderUnavailableError
from ...json_types import (
    JsonObject,
    JsonValue,
    immutable_json_object,
    parse_json_object,
)
from ...types import (
    FinishReason,
    ImagePart,
    Message,
    MessageRole,
    ModelRequest,
    ModelResponse,
    StreamEnd,
    TextDelta,
    TextPart,
    ToolCall,
    ToolCallDelta,
    ToolDefinition,
    Usage,
)
from ..base import StreamDecoder, StreamDelta
from ._types import AnthropicRequest
from .transport import AnthropicStreamEvent

_ADAPTER_OWNED_OPTIONS = frozenset(
    {
        "max_tokens",
        "messages",
        "model",
        "stop_sequences",
        "stream",
        "system",
        "temperature",
        "timeout",
    }
)
type _ImageMediaType = Literal["image/jpeg", "image/png", "image/gif", "image/webp"]
_IMAGE_MEDIA_TYPES: frozenset[str] = frozenset(
    {"image/jpeg", "image/png", "image/gif", "image/webp"}
)


def _usage(raw: anthropic_types.Usage) -> Usage:
    cache_creation = raw.cache_creation_input_tokens or 0
    cache_read = raw.cache_read_input_tokens or 0
    return Usage(
        input_tokens=raw.input_tokens + cache_creation + cache_read,
        output_tokens=raw.output_tokens,
        cached_tokens=cache_read,
    )


def _finish_reason(response: anthropic_types.Message) -> FinishReason:
    if response.stop_reason in {"end_turn", "stop_sequence"}:
        return FinishReason.STOP
    if response.stop_reason in {"max_tokens", "model_context_window_exceeded"}:
        return FinishReason.LENGTH
    if response.stop_reason == "tool_use":
        return FinishReason.TOOL_CALLS
    if response.stop_reason == "refusal":
        return FinishReason.CONTENT_FILTER
    return FinishReason.UNKNOWN


def _tool_input(value: Mapping[str, object]) -> JsonObject:
    candidate = cast(JsonObject, value)
    return immutable_json_object(candidate)


def _decode_response(
    response: anthropic_types.Message,
    *,
    fallback_model: str,
) -> ModelResponse:
    content: list[TextPart] = []
    calls: list[ToolCall] = []
    for block in response.content:
        if isinstance(block, anthropic_types.TextBlock):
            content.append(TextPart(block.text))
        elif isinstance(block, ToolUseBlock):
            calls.append(
                ToolCall(
                    id=block.id,
                    name=block.name,
                    arguments=_tool_input(block.input),
                )
            )

    return ModelResponse(
        message=Message(
            MessageRole.ASSISTANT,
            tuple(content),
            tool_calls=tuple(calls),
        ),
        usage=_usage(response.usage),
        finish_reason=_finish_reason(response),
        raw=response,
        model=str(response.model) or fallback_model,
    )


class AnthropicStreamDecoder:
    """Decode typed Messages events and retain the terminal SDK message."""

    def __init__(self, *, fallback_model: str) -> None:
        self._fallback_model = fallback_model
        self._terminal_message: anthropic_types.Message | None = None
        self._tool_indexes: dict[int, int] = {}

    def feed(self, chunk: AnthropicStreamEvent) -> tuple[StreamDelta, ...]:
        """Consume one typed Messages event and return public deltas."""
        if isinstance(chunk, RawContentBlockStartEvent) and isinstance(
            chunk.content_block, ToolUseBlock
        ):
            return (
                ToolCallDelta(
                    index=self._tool_index(chunk.index),
                    id=chunk.content_block.id,
                    name=chunk.content_block.name,
                ),
            )

        if isinstance(chunk, RawContentBlockDeltaEvent):
            if isinstance(chunk.delta, anthropic_types.TextDelta):
                return (TextDelta(chunk.delta.text),)
            if isinstance(chunk.delta, InputJSONDelta):
                return (
                    ToolCallDelta(
                        index=self._tool_index(chunk.index),
                        arguments_delta=chunk.delta.partial_json,
                    ),
                )

        if isinstance(chunk, ParsedMessageStopEvent):
            self._terminal_message = chunk.message
        return ()

    def finish(self) -> StreamEnd:
        """Build the terminal event from the SDK's accumulated final message."""
        if self._terminal_message is None:
            raise ProviderUnavailableError(
                "Anthropic stream ended without a terminal message event",
                retryable=False,
                provider="anthropic",
            )
        response = _decode_response(
            self._terminal_message,
            fallback_model=self._fallback_model,
        )
        return StreamEnd(response=response)

    def _tool_index(self, content_index: int) -> int:
        try:
            return self._tool_indexes[content_index]
        except KeyError:
            index = len(self._tool_indexes)
            self._tool_indexes[content_index] = index
            return index


class AnthropicCodec:
    """Encode requests and decode Messages without owning network behavior."""

    def __init__(self, *, default_max_output_tokens: int = 1024) -> None:
        if default_max_output_tokens <= 0:
            raise ValueError("default_max_output_tokens must be greater than zero")
        self._default_max_output_tokens = default_max_output_tokens

    def encode_request(
        self,
        model_id: str,
        request: ModelRequest,
        *,
        stream: bool,
    ) -> AnthropicRequest:
        """Translate normalized fields to typed Messages API parameters."""
        provider_options, provider_tools = self._provider_options(
            request.provider_options
        )
        system: list[TextBlockParam] = []
        messages: list[MessageParam] = []
        conversation_started = False
        for message in request.messages:
            if message.role in {MessageRole.SYSTEM, MessageRole.DEVELOPER}:
                blocks = self._system_blocks(message)
                if conversation_started:
                    mid_conversation: MessageParam = {
                        "role": "system",
                        "content": list(blocks),
                    }
                    messages.append(mid_conversation)
                else:
                    system.extend(blocks)
                continue
            conversation_started = True
            messages.append(self._message(message))

        if not messages:
            raise InvalidRequestError(
                "Anthropic requires at least one non-system message",
                provider="anthropic",
            )

        return AnthropicRequest(
            model=model_id,
            messages=tuple(messages),
            max_tokens=request.max_output_tokens or self._default_max_output_tokens,
            stream=stream,
            system=tuple(system),
            function_tools=tuple(self._tool(tool) for tool in request.tools),
            provider_tools=provider_tools,
            temperature=request.temperature,
            stop_sequences=request.stop,
            timeout=request.timeout,
            provider_options=provider_options,
        )

    def decode_response(
        self,
        response: anthropic_types.Message,
        *,
        fallback_model: str,
    ) -> ModelResponse:
        """Translate typed content blocks into one assistant response."""
        return _decode_response(response, fallback_model=fallback_model)

    def stream_decoder(
        self,
        *,
        fallback_model: str,
    ) -> StreamDecoder[AnthropicStreamEvent]:
        """Create isolated state for one Messages API event stream."""
        return AnthropicStreamDecoder(fallback_model=fallback_model)

    @classmethod
    def _system_blocks(cls, message: Message) -> tuple[TextBlockParam, ...]:
        cls._validate_message_metadata(message)
        if message.tool_calls or message.tool_call_id is not None:
            raise InvalidRequestError(
                "Anthropic system messages cannot contain tool data",
                provider="anthropic",
            )
        cls._require_text_only(message)
        return tuple(
            cls._text_block(part.text)
            for part in message.content
            if isinstance(part, TextPart)
        )

    @classmethod
    def _message(cls, message: Message) -> MessageParam:
        cls._validate_message_metadata(message)
        if message.role is not MessageRole.TOOL and message.tool_call_id is not None:
            raise InvalidRequestError(
                "Anthropic tool_call_id is only valid on tool messages",
                provider="anthropic",
            )
        if message.role is not MessageRole.ASSISTANT and message.tool_calls:
            raise InvalidRequestError(
                "Anthropic tool calls may only be attached to assistant messages",
                provider="anthropic",
            )

        if message.role is MessageRole.TOOL:
            cls._require_text_only(message)
            if message.tool_call_id is None:
                raise InvalidRequestError(
                    "an Anthropic tool message requires tool_call_id",
                    provider="anthropic",
                )
            result: ToolResultBlockParam = {
                "type": "tool_result",
                "tool_use_id": message.tool_call_id,
                "content": message.text,
            }
            tool_message: MessageParam = {
                "role": "user",
                "content": [result],
            }
            return tool_message

        blocks: list[ContentBlockParam] = [
            cls._content_part(part) for part in message.content
        ]
        blocks.extend(cls._message_tool_call(call) for call in message.tool_calls)
        role: Literal["user", "assistant"] = (
            "assistant" if message.role is MessageRole.ASSISTANT else "user"
        )
        encoded: MessageParam = {"role": role, "content": blocks}
        return encoded

    @staticmethod
    def _validate_message_metadata(message: Message) -> None:
        if message.name is not None:
            raise InvalidRequestError(
                "the Anthropic Messages API does not support message names",
                provider="anthropic",
            )

    @staticmethod
    def _require_text_only(message: Message) -> None:
        if not all(isinstance(part, TextPart) for part in message.content):
            raise InvalidRequestError(
                f"Anthropic {message.role.value} messages support text content only",
                provider="anthropic",
            )

    @classmethod
    def _content_part(cls, part: TextPart | ImagePart) -> ContentBlockParam:
        if isinstance(part, TextPart):
            return cls._text_block(part.text)
        if part.detail != "auto":
            raise InvalidRequestError(
                "Anthropic images do not support normalized image detail levels",
                provider="anthropic",
            )
        image: ImageBlockParam = {
            "type": "image",
            "source": cls._image_source(part.url),
        }
        return image

    @staticmethod
    def _text_block(text: str) -> TextBlockParam:
        block: TextBlockParam = {"type": "text", "text": text}
        return block

    @staticmethod
    def _image_source(
        url: str,
    ) -> URLImageSourceParam | Base64ImageSourceParam:
        if not url.startswith("data:"):
            source: URLImageSourceParam = {"type": "url", "url": url}
            return source

        metadata, separator, data = url[5:].partition(",")
        media_type, marker_separator, encoding = metadata.partition(";")
        if (
            not separator
            or not marker_separator
            or encoding.lower() != "base64"
            or media_type not in _IMAGE_MEDIA_TYPES
        ):
            raise InvalidRequestError(
                "Anthropic data images require base64 JPEG, PNG, GIF, or WebP data",
                provider="anthropic",
            )
        try:
            base64.b64decode(data, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise InvalidRequestError(
                "Anthropic image data contains invalid base64",
                provider="anthropic",
            ) from exc
        typed_media_type = cast(_ImageMediaType, media_type)
        base64_source: Base64ImageSourceParam = {
            "type": "base64",
            "media_type": typed_media_type,
            "data": data,
        }
        return base64_source

    @staticmethod
    def _message_tool_call(call: ToolCall) -> ToolUseBlockParam:
        arguments = call.arguments
        if isinstance(arguments, str):
            parsed = parse_json_object(arguments)
            if parsed is None:
                raise InvalidRequestError(
                    "Anthropic tool call arguments must be a JSON object",
                    provider="anthropic",
                )
            arguments = parsed
        block: ToolUseBlockParam = {
            "type": "tool_use",
            "id": call.id,
            "name": call.name,
            "input": dict(arguments),
        }
        return block

    @staticmethod
    def _tool(tool: ToolDefinition) -> ToolParam:
        input_schema: dict[str, object] = dict(tool.parameters)
        result: ToolParam = {
            "name": tool.name,
            "input_schema": input_schema,
        }
        if tool.description is not None:
            result["description"] = tool.description
        if tool.strict is not None:
            result["strict"] = tool.strict
        return result

    @staticmethod
    def _provider_options(
        options: JsonObject,
    ) -> tuple[JsonObject, tuple[JsonObject, ...]]:
        conflicts = sorted(_ADAPTER_OWNED_OPTIONS.intersection(options))
        if conflicts:
            joined = ", ".join(conflicts)
            raise InvalidRequestError(
                "Anthropic provider_options cannot override normalized fields: "
                f"{joined}",
                provider="anthropic",
            )

        values: dict[str, JsonValue] = dict(options)
        if "tools" not in values:
            return immutable_json_object(values), ()

        raw_tools = values.pop("tools")
        if isinstance(raw_tools, str | bytes | bytearray) or not isinstance(
            raw_tools, Sequence
        ):
            raise InvalidRequestError(
                "Anthropic provider option 'tools' must be a sequence of JSON objects",
                provider="anthropic",
            )

        provider_tools: list[JsonObject] = []
        for raw_tool in raw_tools:
            if not isinstance(raw_tool, Mapping):
                raise InvalidRequestError(
                    "Anthropic provider option 'tools' must contain JSON objects",
                    provider="anthropic",
                )
            tool = cast(JsonObject, raw_tool)
            provider_tools.append(dict(immutable_json_object(tool)))
        return immutable_json_object(values), tuple(provider_tools)
