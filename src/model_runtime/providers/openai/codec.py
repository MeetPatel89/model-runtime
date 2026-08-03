"""Typed translation between normalized values and OpenAI Chat Completions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from openai.types.chat import (
    ChatCompletion,
    ChatCompletionAssistantMessageParam,
    ChatCompletionChunk,
    ChatCompletionContentPartImageParam,
    ChatCompletionContentPartParam,
    ChatCompletionContentPartTextParam,
    ChatCompletionDeveloperMessageParam,
    ChatCompletionFunctionToolParam,
    ChatCompletionMessageFunctionToolCall,
    ChatCompletionMessageFunctionToolCallParam,
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionToolMessageParam,
    ChatCompletionToolUnionParam,
    ChatCompletionUserMessageParam,
)
from openai.types.completion_usage import CompletionUsage
from openai.types.shared_params.function_definition import FunctionDefinition

from ...errors import InvalidRequestError, ProviderUnavailableError
from ...json_types import JsonObject, parse_json_object
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
from ..base import StreamDelta
from ._types import OpenAIRequest


def _finish_reason(
    value: Literal["stop", "length", "tool_calls", "content_filter", "function_call"]
    | None,
) -> FinishReason:
    if value is None:
        return FinishReason.UNKNOWN
    mapping = {
        "stop": FinishReason.STOP,
        "length": FinishReason.LENGTH,
        "tool_calls": FinishReason.TOOL_CALLS,
        "function_call": FinishReason.TOOL_CALLS,
        "content_filter": FinishReason.CONTENT_FILTER,
    }
    return mapping.get(value, FinishReason.UNKNOWN)


def _usage(raw: CompletionUsage | None) -> Usage:
    if raw is None:
        return Usage()
    details = raw.prompt_tokens_details
    cached_tokens = 0
    if details is not None and details.cached_tokens is not None:
        cached_tokens = details.cached_tokens
    return Usage(
        input_tokens=raw.prompt_tokens,
        output_tokens=raw.completion_tokens,
        cached_tokens=cached_tokens,
    )


def _arguments(value: str) -> JsonObject | str:
    return parse_json_object(value) or value


@dataclass(slots=True)
class _ToolCallBuffer:
    """Mutable assembly state for one streamed function call."""

    identifier_parts: list[str] = field(default_factory=list)
    name_parts: list[str] = field(default_factory=list)
    argument_parts: list[str] = field(default_factory=list)

    def append(
        self,
        *,
        identifier: str | None,
        name: str | None,
        arguments: str | None,
    ) -> None:
        """Append non-empty fragments from one chunk."""
        if identifier:
            self.identifier_parts.append(identifier)
        if name:
            self.name_parts.append(name)
        if arguments:
            self.argument_parts.append(arguments)

    def build(self, index: int) -> ToolCall:
        """Build the normalized final tool call."""
        arguments = "".join(self.argument_parts)
        return ToolCall(
            id="".join(self.identifier_parts) or f"tool_call_{index}",
            name="".join(self.name_parts),
            arguments=_arguments(arguments),
        )


class OpenAIStreamDecoder:
    """Decode and assemble the first choice in one OpenAI stream."""

    def __init__(self, *, fallback_model: str) -> None:
        self._fallback_model = fallback_model
        self._response_model = fallback_model
        self._chunks: list[ChatCompletionChunk] = []
        self._text_parts: list[str] = []
        self._tool_calls: dict[int, _ToolCallBuffer] = {}
        self._usage = Usage()
        self._finish_reason = FinishReason.UNKNOWN

    def feed(self, chunk: ChatCompletionChunk) -> tuple[StreamDelta, ...]:
        """Consume a native chunk and return normalized first-choice deltas."""
        self._chunks.append(chunk)
        self._response_model = chunk.model or self._response_model
        if chunk.usage is not None:
            self._usage = _usage(chunk.usage)

        events: list[StreamDelta] = []
        for choice in chunk.choices:
            if choice.index != 0:
                continue
            if choice.finish_reason is not None:
                self._finish_reason = _finish_reason(choice.finish_reason)
            if choice.delta.content:
                self._text_parts.append(choice.delta.content)
                events.append(TextDelta(choice.delta.content))
            for tool_delta in choice.delta.tool_calls or ():
                function = tool_delta.function
                name = function.name if function is not None else None
                arguments = function.arguments if function is not None else None
                aggregate = self._tool_calls.setdefault(
                    tool_delta.index,
                    _ToolCallBuffer(),
                )
                aggregate.append(
                    identifier=tool_delta.id,
                    name=name,
                    arguments=arguments,
                )
                events.append(
                    ToolCallDelta(
                        index=tool_delta.index,
                        id=tool_delta.id,
                        name=name,
                        arguments_delta=arguments or "",
                    )
                )
        return tuple(events)

    def finish(self) -> StreamEnd:
        """Build the terminal event from accumulated chunks."""
        calls = tuple(
            buffer.build(index) for index, buffer in sorted(self._tool_calls.items())
        )
        text = "".join(self._text_parts)
        response = ModelResponse(
            message=Message(
                MessageRole.ASSISTANT,
                (TextPart(text),) if text else (),
                tool_calls=calls,
            ),
            usage=self._usage,
            finish_reason=self._finish_reason,
            raw=tuple(self._chunks),
            model=self._response_model,
        )
        return StreamEnd(response=response, usage=self._usage)


class OpenAICodec:
    """Encode requests and decode responses without owning network behavior."""

    def encode_request(
        self,
        model_id: str,
        request: ModelRequest,
        *,
        stream: bool,
    ) -> OpenAIRequest:
        """Translate normalized request fields to typed SDK parameters."""
        return OpenAIRequest(
            model=model_id,
            messages=tuple(self._message(message) for message in request.messages),
            stream=stream,
            tools=tuple(self._tool(tool) for tool in request.tools),
            temperature=request.temperature,
            max_completion_tokens=request.max_output_tokens,
            stop=request.stop,
            timeout=request.timeout,
            provider_options=request.provider_options,
        )

    def decode_response(
        self,
        response: ChatCompletion,
        *,
        fallback_model: str,
    ) -> ModelResponse:
        """Translate the first Chat Completions choice to a normalized response."""
        if not response.choices:
            raise ProviderUnavailableError(
                "OpenAI returned a response with no choices",
                retryable=False,
                provider="openai",
                details=response,
            )
        choice = response.choices[0]
        message = choice.message
        calls = tuple(
            self._function_call(call)
            for call in message.tool_calls or ()
            if isinstance(call, ChatCompletionMessageFunctionToolCall)
        )
        return ModelResponse(
            message=Message(
                MessageRole.ASSISTANT,
                (TextPart(message.content),) if message.content else (),
                tool_calls=calls,
            ),
            usage=_usage(response.usage),
            finish_reason=_finish_reason(choice.finish_reason),
            raw=response,
            model=response.model or fallback_model,
        )

    def stream_decoder(
        self,
        *,
        fallback_model: str,
    ) -> OpenAIStreamDecoder:
        """Create isolated state for one Chat Completions stream."""
        return OpenAIStreamDecoder(fallback_model=fallback_model)

    @classmethod
    def _message(cls, message: Message) -> ChatCompletionMessageParam:
        if message.role is MessageRole.SYSTEM:
            cls._require_text_only(message)
            result: ChatCompletionSystemMessageParam = {
                "role": "system",
                "content": message.text,
            }
            if message.name is not None:
                result["name"] = message.name
            return result

        if message.role is MessageRole.DEVELOPER:
            cls._require_text_only(message)
            developer: ChatCompletionDeveloperMessageParam = {
                "role": "developer",
                "content": message.text,
            }
            if message.name is not None:
                developer["name"] = message.name
            return developer

        if message.role is MessageRole.USER:
            content: str | tuple[ChatCompletionContentPartParam, ...]
            if all(isinstance(part, TextPart) for part in message.content):
                content = message.text
            else:
                content = tuple(cls._content_part(part) for part in message.content)
            user: ChatCompletionUserMessageParam = {
                "role": "user",
                "content": content,
            }
            if message.name is not None:
                user["name"] = message.name
            return user

        if message.role is MessageRole.ASSISTANT:
            cls._require_text_only(message)
            assistant: ChatCompletionAssistantMessageParam = {
                "role": "assistant",
                "content": message.text or None,
            }
            if message.name is not None:
                assistant["name"] = message.name
            if message.tool_calls:
                assistant["tool_calls"] = tuple(
                    cls._message_tool_call(call) for call in message.tool_calls
                )
            return assistant

        cls._require_text_only(message)
        if message.tool_call_id is None:
            raise InvalidRequestError(
                "an OpenAI tool message requires tool_call_id",
                provider="openai",
            )
        tool_message: ChatCompletionToolMessageParam = {
            "role": "tool",
            "content": message.text,
            "tool_call_id": message.tool_call_id,
        }
        return tool_message

    @staticmethod
    def _require_text_only(message: Message) -> None:
        if not all(isinstance(part, TextPart) for part in message.content):
            raise InvalidRequestError(
                f"OpenAI {message.role.value} messages support text content only",
                provider="openai",
            )

    @staticmethod
    def _content_part(part: TextPart | ImagePart) -> ChatCompletionContentPartParam:
        if isinstance(part, TextPart):
            text: ChatCompletionContentPartTextParam = {
                "type": "text",
                "text": part.text,
            }
            return text
        image: ChatCompletionContentPartImageParam = {
            "type": "image_url",
            "image_url": {"url": part.url, "detail": part.detail},
        }
        return image

    @staticmethod
    def _message_tool_call(
        call: ToolCall,
    ) -> ChatCompletionMessageFunctionToolCallParam:
        return {
            "id": call.id,
            "type": "function",
            "function": {
                "name": call.name,
                "arguments": call.arguments_json,
            },
        }

    @staticmethod
    def _tool(tool: ToolDefinition) -> ChatCompletionToolUnionParam:
        function: FunctionDefinition = {
            "name": tool.name,
            "parameters": dict(tool.parameters),
        }
        if tool.description is not None:
            function["description"] = tool.description
        if tool.strict is not None:
            function["strict"] = tool.strict
        result: ChatCompletionFunctionToolParam = {
            "type": "function",
            "function": function,
        }
        return result

    @staticmethod
    def _function_call(call: ChatCompletionMessageFunctionToolCall) -> ToolCall:
        return ToolCall(
            id=call.id,
            name=call.function.name,
            arguments=_arguments(call.function.arguments),
        )
