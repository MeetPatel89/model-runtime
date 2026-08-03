"""Typed values internal to the OpenAI integration."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, override

from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionToolUnionParam,
)

from ...json_types import JsonObject, JsonValue, immutable_json_object


class _Unset:
    """Distinguish an omitted SDK field from explicit ``None``."""

    __slots__ = ()


_UNSET = _Unset()
type _Option[T] = T | _Unset
type ReasoningEffort = Literal["none", "minimal", "low", "medium", "high", "xhigh"]
type ServiceTier = Literal["auto", "default", "flex", "scale", "priority", "fast"]


class OpenAIProviderOptions(Mapping[str, JsonValue]):
    """Immutable, key-checked options for OpenAI Chat Completions.

    ``extra`` is an explicit forward-compatibility path for JSON fields added by
    the SDK between versions of this package.
    """

    __slots__ = ("_values",)

    def __init__(
        self,
        *,
        audio: _Option[JsonObject | None] = _UNSET,
        frequency_penalty: _Option[float | None] = _UNSET,
        function_call: _Option[str | JsonObject] = _UNSET,
        functions: _Option[Sequence[JsonObject]] = _UNSET,
        logit_bias: _Option[Mapping[str, int] | None] = _UNSET,
        logprobs: _Option[bool | None] = _UNSET,
        max_tokens: _Option[int | None] = _UNSET,
        metadata: _Option[Mapping[str, str] | None] = _UNSET,
        modalities: _Option[Sequence[Literal["text", "audio"]] | None] = _UNSET,
        n: _Option[int | None] = _UNSET,
        parallel_tool_calls: _Option[bool] = _UNSET,
        prediction: _Option[JsonObject | None] = _UNSET,
        presence_penalty: _Option[float | None] = _UNSET,
        reasoning_effort: _Option[ReasoningEffort | None] = _UNSET,
        response_format: _Option[JsonObject] = _UNSET,
        seed: _Option[int | None] = _UNSET,
        service_tier: _Option[ServiceTier | None] = _UNSET,
        store: _Option[bool | None] = _UNSET,
        stream_options: _Option[JsonObject | None] = _UNSET,
        tool_choice: _Option[str | JsonObject] = _UNSET,
        top_logprobs: _Option[int | None] = _UNSET,
        top_p: _Option[float | None] = _UNSET,
        user: _Option[str] = _UNSET,
        verbosity: _Option[Literal["low", "medium", "high"] | None] = _UNSET,
        web_search_options: _Option[JsonObject] = _UNSET,
        extra_headers: _Option[Mapping[str, str]] = _UNSET,
        extra_query: _Option[JsonObject] = _UNSET,
        extra_body: _Option[JsonObject] = _UNSET,
        extra: JsonObject | None = None,
    ) -> None:
        values: dict[str, JsonValue] = dict(immutable_json_object(extra))
        self._put(values, "audio", audio)
        self._put(values, "frequency_penalty", frequency_penalty)
        self._put(values, "function_call", function_call)
        self._put(values, "functions", functions)
        self._put(values, "logit_bias", logit_bias)
        self._put(values, "logprobs", logprobs)
        self._put(values, "max_tokens", max_tokens)
        self._put(values, "metadata", metadata)
        self._put(values, "modalities", modalities)
        self._put(values, "n", n)
        self._put(values, "parallel_tool_calls", parallel_tool_calls)
        self._put(values, "prediction", prediction)
        self._put(values, "presence_penalty", presence_penalty)
        self._put(values, "reasoning_effort", reasoning_effort)
        self._put(values, "response_format", response_format)
        self._put(values, "seed", seed)
        self._put(values, "service_tier", service_tier)
        self._put(values, "store", store)
        self._put(values, "stream_options", stream_options)
        self._put(values, "tool_choice", tool_choice)
        self._put(values, "top_logprobs", top_logprobs)
        self._put(values, "top_p", top_p)
        self._put(values, "user", user)
        self._put(values, "verbosity", verbosity)
        self._put(values, "web_search_options", web_search_options)
        self._put(values, "extra_headers", extra_headers)
        self._put(values, "extra_query", extra_query)
        self._put(values, "extra_body", extra_body)
        self._values = immutable_json_object(values)

    @override
    def __getitem__(self, key: str) -> JsonValue:
        """Return one encoded SDK option."""
        return self._values[key]

    @override
    def __iter__(self) -> Iterator[str]:
        """Iterate over explicitly supplied option names."""
        return iter(self._values)

    @override
    def __len__(self) -> int:
        """Return the number of explicitly supplied options."""
        return len(self._values)

    @staticmethod
    def _put(
        values: dict[str, JsonValue],
        key: str,
        value: _Option[JsonValue],
    ) -> None:
        if isinstance(value, _Unset):
            return
        if key in values:
            raise ValueError(
                f"OpenAI option {key!r} is present in both extra and a named field"
            )
        values[key] = value


@dataclass(frozen=True, slots=True)
class OpenAIRequest:
    """A fully translated Chat Completions request."""

    model: str
    messages: tuple[ChatCompletionMessageParam, ...]
    stream: bool
    tools: tuple[ChatCompletionToolUnionParam, ...] = ()
    temperature: float | None = None
    max_completion_tokens: int | None = None
    stop: tuple[str, ...] = ()
    timeout: float | None = None
    provider_options: JsonObject | None = None

    def as_kwargs(self) -> dict[str, object]:
        """Build SDK keyword arguments while protecting adapter-owned fields."""
        parameters: dict[str, object] = {
            "model": self.model,
            "messages": self.messages,
            "stream": self.stream,
        }
        if self.tools:
            parameters["tools"] = self.tools
        if self.temperature is not None:
            parameters["temperature"] = self.temperature
        if self.max_completion_tokens is not None:
            parameters["max_completion_tokens"] = self.max_completion_tokens
        if self.stop:
            parameters["stop"] = self.stop
        if self.timeout is not None:
            parameters["timeout"] = self.timeout
        if self.stream:
            parameters["stream_options"] = {"include_usage": True}

        parameters.update(self.provider_options or {})
        parameters["model"] = self.model
        parameters["messages"] = self.messages
        parameters["stream"] = self.stream
        return parameters
