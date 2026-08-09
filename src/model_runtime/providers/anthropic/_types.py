"""Typed values internal to the Anthropic Messages integration."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, override

from anthropic.types import MessageParam, TextBlockParam, ToolParam

from ...json_types import JsonObject, JsonValue, immutable_json_object


class _Unset:
    """Distinguish an omitted SDK field from explicit ``None``."""

    __slots__ = ()


_UNSET = _Unset()
type _Option[T] = T | _Unset
type Effort = Literal["low", "medium", "high", "xhigh", "max"]
type ServiceTier = Literal["auto", "standard_only"]


class AnthropicProviderOptions(Mapping[str, JsonValue]):
    """Immutable, key-checked options for the Anthropic Messages API.

    Provider-native tools are supplied through ``tools`` and are combined with
    normalized function tools. ``extra`` is an explicit JSON-typed path for
    fields added by the SDK between releases of this package.
    """

    __slots__ = ("_values",)

    def __init__(
        self,
        *,
        cache_control: _Option[JsonObject | None] = _UNSET,
        container: _Option[str | None] = _UNSET,
        inference_geo: _Option[str | None] = _UNSET,
        metadata: _Option[JsonObject] = _UNSET,
        output_config: _Option[JsonObject] = _UNSET,
        effort: _Option[Effort | None] = _UNSET,
        service_tier: _Option[ServiceTier] = _UNSET,
        thinking: _Option[JsonObject] = _UNSET,
        tool_choice: _Option[JsonObject] = _UNSET,
        tools: _Option[Sequence[JsonObject]] = _UNSET,
        top_k: _Option[int] = _UNSET,
        top_p: _Option[float] = _UNSET,
        user_profile_id: _Option[str] = _UNSET,
        extra_headers: _Option[Mapping[str, str]] = _UNSET,
        extra_query: _Option[JsonObject] = _UNSET,
        extra_body: _Option[JsonObject] = _UNSET,
        extra: JsonObject | None = None,
    ) -> None:
        values: dict[str, JsonValue] = dict(immutable_json_object(extra))
        self._put(values, "cache_control", cache_control)
        self._put(values, "container", container)
        self._put(values, "inference_geo", inference_geo)
        self._put(values, "metadata", metadata)
        self._put_nested(
            values,
            key="output_config",
            value=output_config,
            shorthand_key="effort",
            shorthand=effort,
        )
        self._put(values, "service_tier", service_tier)
        self._put(values, "thinking", thinking)
        self._put(values, "tool_choice", tool_choice)
        self._put(values, "tools", tools)
        self._put(values, "top_k", top_k)
        self._put(values, "top_p", top_p)
        self._put(values, "user_profile_id", user_profile_id)
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
        value: _Option[JsonObject],
        shorthand_key: str,
        shorthand: _Option[JsonValue],
    ) -> None:
        if not isinstance(value, _Unset) and not isinstance(shorthand, _Unset):
            raise ValueError(
                f"Anthropic options {key!r} and {shorthand_key!r} conflict"
            )
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
                f"Anthropic option {key!r} is present in both extra and a named field"
            )
        values[key] = value


@dataclass(frozen=True, slots=True)
class AnthropicRequest:
    """A fully translated Anthropic Messages API request."""

    model: str
    messages: tuple[MessageParam, ...]
    max_tokens: int
    stream: bool
    system: tuple[TextBlockParam, ...] = ()
    function_tools: tuple[ToolParam, ...] = ()
    provider_tools: tuple[JsonObject, ...] = ()
    temperature: float | None = None
    stop_sequences: tuple[str, ...] = ()
    timeout: float | None = None
    provider_options: JsonObject | None = None

    def as_kwargs(self) -> dict[str, object]:
        """Build SDK keyword arguments from normalized and provider fields."""
        parameters: dict[str, object] = dict(self.provider_options or {})
        parameters["model"] = self.model
        parameters["messages"] = list(self.messages)
        parameters["max_tokens"] = self.max_tokens

        if self.system:
            parameters["system"] = list(self.system)
        tools: list[object] = [*self.provider_tools, *self.function_tools]
        if tools:
            parameters["tools"] = tools
        if self.temperature is not None:
            parameters["temperature"] = self.temperature
        if self.stop_sequences:
            parameters["stop_sequences"] = list(self.stop_sequences)
        if self.timeout is not None:
            parameters["timeout"] = self.timeout
        return parameters
