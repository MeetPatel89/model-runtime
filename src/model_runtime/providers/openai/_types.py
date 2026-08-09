"""Typed values internal to the OpenAI Responses integration."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, override

from openai.types.responses import FunctionToolParam, ResponseInputItemParam

from ...json_types import JsonObject, JsonValue, immutable_json_object


class _Unset:
    """Distinguish an omitted SDK field from explicit ``None``."""

    __slots__ = ()


_UNSET = _Unset()
type _Option[T] = T | _Unset
type ReasoningEffort = Literal["none", "minimal", "low", "medium", "high", "xhigh"]
type ServiceTier = Literal["auto", "default", "flex", "scale", "priority", "fast"]
type Truncation = Literal["auto", "disabled"]
type PromptCacheRetention = Literal["in_memory", "24h"]
type Verbosity = Literal["low", "medium", "high"]


class OpenAIProviderOptions(Mapping[str, JsonValue]):
    """Immutable, key-checked options for the OpenAI Responses API.

    Provider-native tools are supplied through ``tools`` and are combined with
    normalized function tools. ``extra`` is an explicit forward-compatibility
    path for JSON fields added by the SDK between versions of this package.
    """

    __slots__ = ("_values",)

    def __init__(
        self,
        *,
        background: _Option[bool | None] = _UNSET,
        context_management: _Option[Sequence[JsonObject] | None] = _UNSET,
        conversation: _Option[str | JsonObject | None] = _UNSET,
        include: _Option[Sequence[str] | None] = _UNSET,
        instructions: _Option[str | None] = _UNSET,
        max_tool_calls: _Option[int | None] = _UNSET,
        metadata: _Option[Mapping[str, str] | None] = _UNSET,
        moderation: _Option[JsonObject | None] = _UNSET,
        parallel_tool_calls: _Option[bool | None] = _UNSET,
        previous_response_id: _Option[str | None] = _UNSET,
        prompt: _Option[JsonObject | None] = _UNSET,
        prompt_cache_key: _Option[str | None] = _UNSET,
        prompt_cache_options: _Option[JsonObject | None] = _UNSET,
        prompt_cache_retention: _Option[PromptCacheRetention | None] = _UNSET,
        reasoning: _Option[JsonObject | None] = _UNSET,
        reasoning_effort: _Option[ReasoningEffort | None] = _UNSET,
        safety_identifier: _Option[str | None] = _UNSET,
        service_tier: _Option[ServiceTier | None] = _UNSET,
        store: _Option[bool | None] = _UNSET,
        stream_options: _Option[JsonObject | None] = _UNSET,
        text: _Option[JsonObject | None] = _UNSET,
        tool_choice: _Option[str | JsonObject] = _UNSET,
        tools: _Option[Sequence[JsonObject]] = _UNSET,
        top_logprobs: _Option[int | None] = _UNSET,
        top_p: _Option[float | None] = _UNSET,
        truncation: _Option[Truncation | None] = _UNSET,
        user: _Option[str] = _UNSET,
        verbosity: _Option[Verbosity | None] = _UNSET,
        extra_headers: _Option[Mapping[str, str]] = _UNSET,
        extra_query: _Option[JsonObject] = _UNSET,
        extra_body: _Option[JsonObject] = _UNSET,
        extra: JsonObject | None = None,
    ) -> None:
        values: dict[str, JsonValue] = dict(immutable_json_object(extra))
        self._put(values, "background", background)
        self._put(values, "context_management", context_management)
        self._put(values, "conversation", conversation)
        self._put(values, "include", include)
        self._put(values, "instructions", instructions)
        self._put(values, "max_tool_calls", max_tool_calls)
        self._put(values, "metadata", metadata)
        self._put(values, "moderation", moderation)
        self._put(values, "parallel_tool_calls", parallel_tool_calls)
        self._put(values, "previous_response_id", previous_response_id)
        self._put(values, "prompt", prompt)
        self._put(values, "prompt_cache_key", prompt_cache_key)
        self._put(values, "prompt_cache_options", prompt_cache_options)
        self._put(values, "prompt_cache_retention", prompt_cache_retention)
        self._put_nested(
            values,
            key="reasoning",
            value=reasoning,
            shorthand_key="effort",
            shorthand=reasoning_effort,
        )
        self._put(values, "safety_identifier", safety_identifier)
        self._put(values, "service_tier", service_tier)
        self._put(values, "store", store)
        self._put(values, "stream_options", stream_options)
        self._put_nested(
            values,
            key="text",
            value=text,
            shorthand_key="verbosity",
            shorthand=verbosity,
        )
        self._put(values, "tool_choice", tool_choice)
        self._put(values, "tools", tools)
        self._put(values, "top_logprobs", top_logprobs)
        self._put(values, "top_p", top_p)
        self._put(values, "truncation", truncation)
        self._put(values, "user", user)
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

    @classmethod
    def _put_nested(
        cls,
        values: dict[str, JsonValue],
        *,
        key: str,
        value: _Option[JsonObject | None],
        shorthand_key: str,
        shorthand: _Option[JsonValue],
    ) -> None:
        if not isinstance(value, _Unset) and not isinstance(shorthand, _Unset):
            raise ValueError(f"OpenAI options {key!r} and {shorthand_key!r} conflict")
        if not isinstance(value, _Unset):
            cls._put(values, key, value)
        elif not isinstance(shorthand, _Unset):
            cls._put(values, key, {shorthand_key: shorthand})

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
    """A fully translated Responses API request."""

    model: str
    input: tuple[ResponseInputItemParam, ...]
    stream: bool
    function_tools: tuple[FunctionToolParam, ...] = ()
    provider_tools: tuple[JsonObject, ...] = ()
    temperature: float | None = None
    max_output_tokens: int | None = None
    timeout: float | None = None
    provider_options: JsonObject | None = None

    def as_kwargs(self) -> dict[str, object]:
        """Build SDK keyword arguments from normalized and provider fields."""
        parameters: dict[str, object] = dict(self.provider_options or {})
        parameters["model"] = self.model
        parameters["input"] = list(self.input)
        parameters["stream"] = self.stream

        tools: list[object] = [*self.provider_tools, *self.function_tools]
        if tools:
            parameters["tools"] = tools
        if self.temperature is not None:
            parameters["temperature"] = self.temperature
        if self.max_output_tokens is not None:
            parameters["max_output_tokens"] = self.max_output_tokens
        if self.timeout is not None:
            parameters["timeout"] = self.timeout
        return parameters
