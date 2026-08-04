"""Typed translation between normalized values and OpenAI Responses."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import cast

from openai.types.responses import (
    EasyInputMessageParam,
    FunctionToolParam,
    Response,
    ResponseCompletedEvent,
    ResponseErrorEvent,
    ResponseFailedEvent,
    ResponseFunctionCallArgumentsDeltaEvent,
    ResponseFunctionToolCall,
    ResponseFunctionToolCallParam,
    ResponseIncompleteEvent,
    ResponseInputImageParam,
    ResponseInputItemParam,
    ResponseInputMessageContentListParam,
    ResponseInputTextParam,
    ResponseOutputItemAddedEvent,
    ResponseOutputMessage,
    ResponseOutputRefusal,
    ResponseOutputText,
    ResponseRefusalDeltaEvent,
    ResponseStreamEvent,
    ResponseTextDeltaEvent,
    ResponseUsage,
)
from openai.types.responses.response_input_param import FunctionCallOutput

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
from ._types import OpenAIRequest

_ADAPTER_OWNED_OPTIONS = frozenset(
    {"input", "max_output_tokens", "model", "stream", "temperature", "timeout"}
)


def _usage(raw: ResponseUsage | None) -> Usage:
    if raw is None:
        return Usage()
    return Usage(
        input_tokens=raw.input_tokens,
        output_tokens=raw.output_tokens,
        cached_tokens=raw.input_tokens_details.cached_tokens,
    )


def _arguments(value: str) -> JsonObject | str:
    return parse_json_object(value) or value


def _finish_reason(response: Response) -> FinishReason:
    if response.status == "incomplete":
        details = response.incomplete_details
        if details is not None and details.reason == "max_output_tokens":
            return FinishReason.LENGTH
        if details is not None and details.reason == "content_filter":
            return FinishReason.CONTENT_FILTER
        return FinishReason.UNKNOWN
    if response.status in {"failed", "cancelled"}:
        return FinishReason.ERROR
    if any(isinstance(item, ResponseFunctionToolCall) for item in response.output):
        return FinishReason.TOOL_CALLS
    if response.status == "completed":
        return FinishReason.STOP
    return FinishReason.UNKNOWN


def _decode_response(response: Response, *, fallback_model: str) -> ModelResponse:
    content: list[TextPart] = []
    calls: list[ToolCall] = []
    for item in response.output:
        if isinstance(item, ResponseOutputMessage):
            for part in item.content:
                if isinstance(part, ResponseOutputText):
                    content.append(TextPart(part.text))
                elif isinstance(part, ResponseOutputRefusal):
                    content.append(TextPart(part.refusal))
        elif isinstance(item, ResponseFunctionToolCall):
            calls.append(
                ToolCall(
                    id=item.call_id,
                    name=item.name,
                    arguments=_arguments(item.arguments),
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


class OpenAIStreamDecoder:
    """Decode typed Responses stream events and retain the terminal response."""

    def __init__(self, *, fallback_model: str) -> None:
        self._fallback_model = fallback_model
        self._terminal_response: Response | None = None
        self._tool_indexes: dict[int, int] = {}

    def feed(self, chunk: ResponseStreamEvent) -> tuple[StreamDelta, ...]:
        """Consume one typed Responses event and return public deltas."""
        if isinstance(chunk, ResponseTextDeltaEvent | ResponseRefusalDeltaEvent):
            return (TextDelta(chunk.delta),)

        if isinstance(chunk, ResponseOutputItemAddedEvent) and isinstance(
            chunk.item, ResponseFunctionToolCall
        ):
            index = self._tool_index(chunk.output_index)
            return (
                ToolCallDelta(
                    index=index,
                    id=chunk.item.call_id,
                    name=chunk.item.name,
                    arguments_delta=chunk.item.arguments,
                ),
            )

        if isinstance(chunk, ResponseFunctionCallArgumentsDeltaEvent):
            return (
                ToolCallDelta(
                    index=self._tool_index(chunk.output_index),
                    arguments_delta=chunk.delta,
                ),
            )

        if isinstance(
            chunk,
            ResponseCompletedEvent | ResponseIncompleteEvent | ResponseFailedEvent,
        ):
            self._terminal_response = chunk.response
            return ()

        if isinstance(chunk, ResponseErrorEvent):
            raise ProviderUnavailableError(
                chunk.message,
                retryable=False,
                provider="openai",
                details=chunk,
            )
        return ()

    def finish(self) -> StreamEnd:
        """Build the terminal event from the SDK's final response event."""
        if self._terminal_response is None:
            raise ProviderUnavailableError(
                "OpenAI stream ended without a terminal response event",
                retryable=False,
                provider="openai",
            )
        response = _decode_response(
            self._terminal_response,
            fallback_model=self._fallback_model,
        )
        return StreamEnd(response=response)

    def _tool_index(self, output_index: int) -> int:
        try:
            return self._tool_indexes[output_index]
        except KeyError:
            index = len(self._tool_indexes)
            self._tool_indexes[output_index] = index
            return index


class OpenAICodec:
    """Encode requests and decode Responses without owning network behavior."""

    def encode_request(
        self,
        model_id: str,
        request: ModelRequest,
        *,
        stream: bool,
    ) -> OpenAIRequest:
        """Translate normalized fields to typed Responses API parameters."""
        if request.stop:
            raise InvalidRequestError(
                "the OpenAI Responses API does not support stop sequences",
                provider="openai",
            )
        provider_options, provider_tools = self._provider_options(
            request.provider_options
        )
        input_items: list[ResponseInputItemParam] = []
        for message in request.messages:
            input_items.extend(self._message_items(message))
        return OpenAIRequest(
            model=model_id,
            input=tuple(input_items),
            stream=stream,
            function_tools=tuple(self._tool(tool) for tool in request.tools),
            provider_tools=provider_tools,
            temperature=request.temperature,
            max_output_tokens=request.max_output_tokens,
            timeout=request.timeout,
            provider_options=provider_options,
        )

    def decode_response(
        self,
        response: Response,
        *,
        fallback_model: str,
    ) -> ModelResponse:
        """Translate typed output items into one normalized assistant response."""
        return _decode_response(response, fallback_model=fallback_model)

    def stream_decoder(
        self,
        *,
        fallback_model: str,
    ) -> StreamDecoder[ResponseStreamEvent]:
        """Create isolated state for one Responses API event stream."""
        return OpenAIStreamDecoder(fallback_model=fallback_model)

    @classmethod
    def _message_items(cls, message: Message) -> tuple[ResponseInputItemParam, ...]:
        if message.name is not None:
            raise InvalidRequestError(
                "the OpenAI Responses API does not support message names",
                provider="openai",
            )
        if message.role is not MessageRole.ASSISTANT and message.tool_calls:
            raise InvalidRequestError(
                "OpenAI tool calls may only be attached to assistant messages",
                provider="openai",
            )

        if message.role is MessageRole.TOOL:
            cls._require_text_only(message)
            if message.tool_call_id is None:
                raise InvalidRequestError(
                    "an OpenAI tool message requires tool_call_id",
                    provider="openai",
                )
            output: FunctionCallOutput = {
                "type": "function_call_output",
                "call_id": message.tool_call_id,
                "output": message.text,
            }
            return (output,)

        if message.role is MessageRole.USER:
            content: str | ResponseInputMessageContentListParam
            if all(isinstance(part, TextPart) for part in message.content):
                content = message.text
            else:
                content = [cls._content_part(part) for part in message.content]
            user: EasyInputMessageParam = {"role": "user", "content": content}
            return (user,)

        cls._require_text_only(message)
        if message.role is MessageRole.SYSTEM:
            system: EasyInputMessageParam = {
                "role": "system",
                "content": message.text,
            }
            return (system,)
        if message.role is MessageRole.DEVELOPER:
            developer: EasyInputMessageParam = {
                "role": "developer",
                "content": message.text,
            }
            return (developer,)

        items: list[ResponseInputItemParam] = []
        if message.content or not message.tool_calls:
            assistant: EasyInputMessageParam = {
                "role": "assistant",
                "content": message.text,
            }
            items.append(assistant)
        items.extend(cls._message_tool_call(call) for call in message.tool_calls)
        return tuple(items)

    @staticmethod
    def _require_text_only(message: Message) -> None:
        if not all(isinstance(part, TextPart) for part in message.content):
            raise InvalidRequestError(
                f"OpenAI {message.role.value} messages support text content only",
                provider="openai",
            )

    @staticmethod
    def _content_part(
        part: TextPart | ImagePart,
    ) -> ResponseInputTextParam | ResponseInputImageParam:
        if isinstance(part, TextPart):
            text: ResponseInputTextParam = {
                "type": "input_text",
                "text": part.text,
            }
            return text
        image: ResponseInputImageParam = {
            "type": "input_image",
            "image_url": part.url,
            "detail": part.detail,
        }
        return image

    @staticmethod
    def _message_tool_call(call: ToolCall) -> ResponseFunctionToolCallParam:
        return {
            "type": "function_call",
            "call_id": call.id,
            "name": call.name,
            "arguments": call.arguments_json,
        }

    @staticmethod
    def _tool(tool: ToolDefinition) -> FunctionToolParam:
        parameters: dict[str, object] = dict(tool.parameters)
        result: FunctionToolParam = {
            "type": "function",
            "name": tool.name,
            "parameters": parameters,
            "strict": tool.strict,
        }
        if tool.description is not None:
            result["description"] = tool.description
        return result

    @staticmethod
    def _provider_options(
        options: JsonObject,
    ) -> tuple[JsonObject, tuple[JsonObject, ...]]:
        conflicts = sorted(_ADAPTER_OWNED_OPTIONS.intersection(options))
        if conflicts:
            joined = ", ".join(conflicts)
            raise InvalidRequestError(
                f"OpenAI provider_options cannot override normalized fields: {joined}",
                provider="openai",
            )

        values: dict[str, JsonValue] = dict(options)
        if "tools" not in values:
            return immutable_json_object(values), ()

        raw_tools = values.pop("tools")
        if isinstance(raw_tools, str | bytes | bytearray) or not isinstance(
            raw_tools, Sequence
        ):
            raise InvalidRequestError(
                "OpenAI provider option 'tools' must be a sequence of JSON objects",
                provider="openai",
            )

        provider_tools: list[JsonObject] = []
        for raw_tool in raw_tools:
            if not isinstance(raw_tool, Mapping):
                raise InvalidRequestError(
                    "OpenAI provider option 'tools' must contain JSON objects",
                    provider="openai",
                )
            tool = cast(JsonObject, raw_tool)
            provider_tools.append(dict(immutable_json_object(tool)))
        return immutable_json_object(values), tuple(provider_tools)
