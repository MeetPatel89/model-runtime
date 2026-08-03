"""OpenAI Chat Completions adapter."""

from __future__ import annotations

import inspect
import json
from collections.abc import AsyncIterator, Mapping, Sequence
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any

import openai

from ..errors import (
    AuthError,
    ContentFilterError,
    InvalidRequestError,
    ModelRuntimeError,
    ProviderUnavailableError,
    RateLimitError,
    RequestTimeout,
)
from ..types import (
    FinishReason,
    ImagePart,
    Message,
    MessageRole,
    ModelCapabilities,
    ModelRequest,
    ModelResponse,
    StreamEnd,
    StreamEvent,
    TextDelta,
    TextPart,
    ToolCall,
    ToolCallDelta,
    ToolDefinition,
    Usage,
)


def _get(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    )


class OpenAIAdapter:
    """Translate normalized calls to the official async OpenAI SDK.

    SDK retries are disabled so retry behavior remains visible and centralized in
    :class:`model_runtime.ModelRuntime`. Supplying ``client`` is useful for tests;
    injected clients are expected to have SDK retries disabled by their owner.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        organization: str | None = None,
        project: str | None = None,
        client: Any | None = None,
        capabilities: ModelCapabilities | None = None,
    ) -> None:
        if client is None:
            client_options: dict[str, Any] = {"max_retries": 0}
            if api_key is not None:
                client_options["api_key"] = api_key
            if base_url is not None:
                client_options["base_url"] = base_url
            if organization is not None:
                client_options["organization"] = organization
            if project is not None:
                client_options["project"] = project
            client = openai.AsyncOpenAI(**client_options)
        self._client = client
        self._capabilities = capabilities or ModelCapabilities(
            tools=True,
            vision=True,
            structured_output=True,
            streaming=True,
            max_context_tokens=None,
            provider_features=frozenset(
                {"audio", "logprobs", "prediction", "reasoning_effort", "service_tier"}
            ),
        )

    @property
    def client(self) -> Any:
        """The underlying asynchronous OpenAI SDK client."""
        return self._client

    @property
    def capabilities(self) -> ModelCapabilities:
        """Features advertised by this adapter."""
        return self._capabilities

    async def complete(self, model_id: str, request: ModelRequest) -> ModelResponse:
        """Submit a non-streaming Chat Completions request."""
        try:
            result = self._client.chat.completions.create(
                **self._request_kwargs(model_id, request, stream=False)
            )
            if inspect.isawaitable(result):
                result = await result
            return self._translate_response(result, fallback_model=model_id)
        except ModelRuntimeError:
            raise
        except Exception as exc:
            error = self.translate_error(exc)
            raise error from exc

    async def stream(
        self, model_id: str, request: ModelRequest
    ) -> AsyncIterator[StreamEvent]:
        """Yield normalized events from a streaming Chat Completions request."""
        sdk_stream: Any = None
        try:
            result = self._client.chat.completions.create(
                **self._request_kwargs(model_id, request, stream=True)
            )
            sdk_stream = await result if inspect.isawaitable(result) else result

            chunks: list[Any] = []
            text_parts: list[str] = []
            tool_parts: dict[int, dict[str, str]] = {}
            finish_reason = FinishReason.UNKNOWN
            usage = Usage()
            response_model = model_id

            async for chunk in sdk_stream:
                chunks.append(chunk)
                response_model = _get(chunk, "model", response_model) or response_model
                chunk_usage = _get(chunk, "usage")
                if chunk_usage is not None:
                    usage = self._translate_usage(chunk_usage)

                choices = _get(chunk, "choices", ()) or ()
                for choice in choices:
                    raw_finish_reason = _get(choice, "finish_reason")
                    if raw_finish_reason is not None:
                        finish_reason = self._translate_finish_reason(raw_finish_reason)

                    delta = _get(choice, "delta", {}) or {}
                    content = _get(delta, "content")
                    for text in self._response_text_parts(content):
                        if text:
                            text_parts.append(text)
                            yield TextDelta(text)

                    for fallback_index, raw_tool_delta in enumerate(
                        _get(delta, "tool_calls", ()) or ()
                    ):
                        index = _get(raw_tool_delta, "index")
                        if index is None:
                            index = fallback_index
                        raw_id = _get(raw_tool_delta, "id") or ""
                        function = _get(raw_tool_delta, "function", {}) or {}
                        name = _get(function, "name") or ""
                        arguments = _get(function, "arguments") or ""

                        aggregate = tool_parts.setdefault(
                            index, {"id": "", "name": "", "arguments": ""}
                        )
                        aggregate["id"] += raw_id
                        aggregate["name"] += name
                        aggregate["arguments"] += arguments
                        yield ToolCallDelta(
                            index=index,
                            id=raw_id or None,
                            name=name or None,
                            arguments_delta=arguments,
                        )

            calls = tuple(
                ToolCall(
                    id=parts["id"] or f"tool_call_{index}",
                    name=parts["name"],
                    arguments=self._decode_arguments(parts["arguments"]),
                )
                for index, parts in sorted(tool_parts.items())
            )
            text = "".join(text_parts)
            message = Message(
                MessageRole.ASSISTANT,
                (TextPart(text),) if text else (),
                tool_calls=calls,
            )
            response = ModelResponse(
                message=message,
                usage=usage,
                finish_reason=finish_reason,
                raw=tuple(chunks),
                model=response_model,
            )
            yield StreamEnd(response=response, usage=usage)
        except ModelRuntimeError:
            raise
        except Exception as exc:
            error = self.translate_error(exc)
            raise error from exc
        finally:
            await self._close_stream(sdk_stream)

    def _request_kwargs(
        self,
        model_id: str,
        request: ModelRequest,
        *,
        stream: bool,
    ) -> dict[str, Any]:
        messages = [self._translate_message(message) for message in request.messages]
        kwargs: dict[str, Any] = {
            "model": model_id,
            "messages": messages,
            "stream": stream,
        }
        if request.tools:
            kwargs["tools"] = [self._translate_tool(tool) for tool in request.tools]
        if request.temperature is not None:
            kwargs["temperature"] = request.temperature
        if request.max_output_tokens is not None:
            kwargs["max_completion_tokens"] = request.max_output_tokens
        if request.stop:
            kwargs["stop"] = list(request.stop)
        if request.timeout is not None:
            kwargs["timeout"] = request.timeout
        if stream:
            kwargs["stream_options"] = {"include_usage": True}

        # Open-ended options intentionally flow through without schema filtering.
        kwargs.update(request.provider_options)
        # The adapter contract, rather than provider_options, owns call identity/mode.
        kwargs["model"] = model_id
        kwargs["messages"] = messages
        kwargs["stream"] = stream
        return kwargs

    @staticmethod
    def _translate_message(message: Message) -> dict[str, Any]:
        translated: dict[str, Any] = {"role": message.role.value}
        text_only = all(isinstance(part, TextPart) for part in message.content)
        if text_only:
            content: Any = "".join(part.text for part in message.content)
            if message.role is MessageRole.ASSISTANT and not content:
                content = None
        else:
            content = []
            for part in message.content:
                if isinstance(part, TextPart):
                    content.append({"type": "text", "text": part.text})
                elif isinstance(part, ImagePart):
                    content.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": part.url, "detail": part.detail},
                        }
                    )
        translated["content"] = content

        if message.name is not None:
            translated["name"] = message.name
        if message.tool_call_id is not None:
            translated["tool_call_id"] = message.tool_call_id
        if message.tool_calls:
            translated["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": call.arguments_json,
                    },
                }
                for call in message.tool_calls
            ]
        return translated

    @staticmethod
    def _translate_tool(tool: ToolDefinition) -> dict[str, Any]:
        function: dict[str, Any] = {
            "name": tool.name,
            "parameters": dict(tool.parameters),
        }
        if tool.description is not None:
            function["description"] = tool.description
        if tool.strict is not None:
            function["strict"] = tool.strict
        return {"type": "function", "function": function}

    def _translate_response(
        self, response: Any, *, fallback_model: str
    ) -> ModelResponse:
        choices = _get(response, "choices", ()) or ()
        if not choices:
            raise ProviderUnavailableError(
                "OpenAI returned a response with no choices",
                retryable=False,
                provider="openai",
                details=response,
            )
        choice = choices[0]
        raw_message = _get(choice, "message", {}) or {}
        content = _get(raw_message, "content")
        text = "".join(self._response_text_parts(content))

        calls: list[ToolCall] = []
        for index, raw_call in enumerate(_get(raw_message, "tool_calls", ()) or ()):
            function = _get(raw_call, "function", {}) or {}
            arguments = _get(function, "arguments", "") or ""
            calls.append(
                ToolCall(
                    id=_get(raw_call, "id") or f"tool_call_{index}",
                    name=_get(function, "name", "") or "",
                    arguments=self._decode_arguments(arguments),
                )
            )

        message = Message(
            MessageRole.ASSISTANT,
            (TextPart(text),) if text else (),
            tool_calls=tuple(calls),
        )
        return ModelResponse(
            message=message,
            usage=self._translate_usage(_get(response, "usage")),
            finish_reason=self._translate_finish_reason(_get(choice, "finish_reason")),
            raw=response,
            model=_get(response, "model", fallback_model) or fallback_model,
        )

    @staticmethod
    def _response_text_parts(content: Any) -> list[str]:
        if content is None:
            return []
        if isinstance(content, str):
            return [content]
        if not _is_sequence(content):
            text = _get(content, "text")
            return [text] if isinstance(text, str) else []
        result: list[str] = []
        for part in content:
            if isinstance(part, str):
                result.append(part)
                continue
            text = _get(part, "text")
            if isinstance(text, str):
                result.append(text)
        return result

    @staticmethod
    def _decode_arguments(value: Any) -> Mapping[str, Any] | str:
        if isinstance(value, Mapping):
            return value
        if not isinstance(value, str):
            return str(value)
        try:
            decoded = json.loads(value)
        except TypeError, ValueError:
            return value
        return decoded if isinstance(decoded, Mapping) else value

    @staticmethod
    def _translate_usage(raw_usage: Any) -> Usage:
        if raw_usage is None:
            return Usage()
        details = _get(raw_usage, "prompt_tokens_details", {}) or {}
        cached = _get(details, "cached_tokens", 0) or 0
        return Usage(
            input_tokens=_get(raw_usage, "prompt_tokens", 0) or 0,
            output_tokens=_get(raw_usage, "completion_tokens", 0) or 0,
            cached_tokens=cached,
        )

    @staticmethod
    def _translate_finish_reason(value: Any) -> FinishReason:
        mapping = {
            "stop": FinishReason.STOP,
            "length": FinishReason.LENGTH,
            "tool_calls": FinishReason.TOOL_CALLS,
            "function_call": FinishReason.TOOL_CALLS,
            "content_filter": FinishReason.CONTENT_FILTER,
            "error": FinishReason.ERROR,
        }
        return mapping.get(value, FinishReason.UNKNOWN)

    @staticmethod
    def _exception_is(exc: BaseException, *names: str) -> bool:
        classes = tuple(
            candidate
            for name in names
            if isinstance((candidate := getattr(openai, name, None)), type)
        )
        return bool(classes) and isinstance(exc, classes)

    @classmethod
    def translate_error(cls, exc: BaseException) -> ModelRuntimeError:
        """Map an OpenAI SDK error to the public runtime taxonomy."""
        if isinstance(exc, ModelRuntimeError):
            return exc

        status_code = _get(exc, "status_code")
        body = _get(exc, "body")
        retry_after = cls._retry_after(exc)
        message = str(exc) or exc.__class__.__name__
        common = {
            "cause": exc,
            "status_code": status_code,
            "provider": "openai",
            "details": body,
        }

        if cls._is_content_filter_error(exc):
            return ContentFilterError(message, retryable=False, **common)
        if cls._exception_is(
            exc, "AuthenticationError", "PermissionDeniedError"
        ) or status_code in {
            401,
            403,
        }:
            return AuthError(message, retryable=False, **common)
        if cls._exception_is(exc, "RateLimitError") or status_code == 429:
            return RateLimitError(message, retry_after=retry_after, **common)
        if cls._exception_is(exc, "APITimeoutError") or status_code == 408:
            return RequestTimeout(message, **common)
        if cls._exception_is(
            exc,
            "BadRequestError",
            "NotFoundError",
            "ConflictError",
            "UnprocessableEntityError",
        ) or status_code in {400, 404, 409, 422}:
            return InvalidRequestError(message, retryable=False, **common)
        if cls._exception_is(exc, "APIConnectionError", "InternalServerError"):
            return ProviderUnavailableError(
                message, retry_after=retry_after, retryable=True, **common
            )
        if isinstance(status_code, int) and status_code >= 500:
            return ProviderUnavailableError(
                message, retry_after=retry_after, retryable=True, **common
            )
        return ProviderUnavailableError(message, retryable=False, **common)

    # Backward-friendly private alias used by some adapter tests and integrations.
    _translate_error = translate_error

    @classmethod
    def _is_content_filter_error(cls, exc: BaseException) -> bool:
        values = [_get(exc, "code"), _get(exc, "body"), str(exc)]

        def contains_filter(value: Any) -> bool:
            if isinstance(value, Mapping):
                return any(contains_filter(item) for item in value.values())
            if _is_sequence(value):
                return any(contains_filter(item) for item in value)
            if value is None:
                return False
            normalized = str(value).lower().replace("-", "_").replace(" ", "_")
            return "content_filter" in normalized or "content_policy" in normalized

        return any(contains_filter(value) for value in values)

    @staticmethod
    def _retry_after(exc: BaseException) -> float | None:
        response = _get(exc, "response")
        headers = _get(response, "headers", {}) or {}
        lowered = {str(key).lower(): value for key, value in headers.items()}
        raw = lowered.get("retry-after")
        if raw is None and "retry-after-ms" in lowered:
            try:
                return max(0.0, float(lowered["retry-after-ms"]) / 1000.0)
            except TypeError, ValueError:
                return None
        if raw is None:
            return None
        try:
            return max(0.0, float(raw))
        except TypeError, ValueError:
            pass
        try:
            parsed = parsedate_to_datetime(str(raw))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=UTC)
            return max(0.0, (parsed - datetime.now(UTC)).total_seconds())
        except TypeError, ValueError, OverflowError:
            return None

    @staticmethod
    async def _close_stream(stream: Any) -> None:
        if stream is None:
            return
        close = getattr(stream, "aclose", None) or getattr(stream, "close", None)
        if close is None:
            return
        try:
            result = close()
            if inspect.isawaitable(result):
                await result
        except Exception:  # noqa: BLE001 - best-effort SDK stream cleanup
            return
