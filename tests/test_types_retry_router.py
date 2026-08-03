"""Tests for value objects, retry policy, and model routing."""

from __future__ import annotations

from collections.abc import AsyncIterator, MutableMapping
from dataclasses import FrozenInstanceError
from typing import cast

import pytest

from model_runtime import (
    JsonObject,
    JsonValue,
    Message,
    ModelCapabilities,
    ModelRequest,
    ModelResponse,
    ModelRouter,
    RateLimitError,
    RetryPolicy,
    StreamEvent,
    TextPart,
    ToolCall,
    ToolDefinition,
    Usage,
)


class StubModel:
    """Minimal model exposing capabilities for router tests."""

    capabilities = ModelCapabilities(tools=True, max_context_tokens=1234)

    async def complete(self, model_id: str, request: ModelRequest) -> ModelResponse:
        """Reject calls because router tests only inspect identity and metadata."""
        raise NotImplementedError

    async def stream(
        self,
        model_id: str,
        request: ModelRequest,
    ) -> AsyncIterator[StreamEvent]:
        """Reject calls because router tests only inspect identity and metadata."""
        raise NotImplementedError
        yield  # pragma: no cover


def test_value_objects_normalize_sequences_and_are_immutable() -> None:
    """Request values normalize collections and prevent mutation."""
    options = {"reasoning_effort": "high"}
    request = ModelRequest(
        messages=[Message.user("hello")],
        stop=["done"],
        provider_options=options,
    )
    options["reasoning_effort"] = "low"

    assert isinstance(request.messages, tuple)
    assert request.messages[0].content == (TextPart("hello"),)
    assert request.messages[0].text == "hello"
    assert request.provider_options["reasoning_effort"] == "high"
    mutable_options = cast(
        MutableMapping[str, JsonValue],
        request.provider_options,
    )
    with pytest.raises(TypeError):
        mutable_options["seed"] = 42
    with pytest.raises(FrozenInstanceError):
        setattr(request, "timeout", 1)


def test_tool_arguments_and_usage_helpers() -> None:
    """Tool arguments and usage arithmetic expose convenient helpers."""
    call = ToolCall("call-1", "weather", {"city": "Boston"})
    assert call.arguments_json == '{"city":"Boston"}'
    assert Usage(2, 3, 1).total_tokens == 5
    assert Usage(2, 3, 1) + Usage(4, 5, 2) == Usage(6, 8, 3)


def test_provider_options_reject_values_that_are_not_json() -> None:
    """Provider payload validation prevents opaque Python objects from leaking in."""
    invalid_options = cast(JsonObject, {"callback": object()})

    with pytest.raises(TypeError, match="JSON-compatible"):
        ModelRequest.from_text("hello", provider_options=invalid_options)
    with pytest.raises(ValueError, match="non-finite"):
        ModelRequest.from_text("hello", provider_options={"temperature": float("nan")})


def test_retry_policy_uses_retry_after_then_exponential_backoff() -> None:
    """Provider retry hints take precedence over exponential backoff."""
    policy = RetryPolicy(
        max_attempts=3,
        initial_delay=0.25,
        max_delay=1,
        multiplier=2,
        jitter=0,
    )
    ordinary = RateLimitError("slow down")
    instructed = RateLimitError("slow down", retry_after=4.5)

    assert policy.should_retry(ordinary, 1)
    assert not policy.should_retry(ordinary, 3)
    assert policy.delay_for(1, ordinary) == 0.25
    assert policy.delay_for(2, ordinary) == 0.5
    assert policy.delay_for(1, instructed) == 4.5


def test_router_registry_selector_and_capabilities() -> None:
    """Router registration, selection, and capability lookup compose correctly."""
    small = StubModel()
    large = StubModel()
    router = ModelRouter(
        selector=lambda name, request: "large" if request.tools else "small"
    )
    router.register("small", small, "provider-small")
    router.register("large", large, "provider-large")

    assert (
        router.resolve("default", ModelRequest.from_text("hi")).model_id
        == "provider-small"
    )
    with_tool = ModelRequest.from_text(
        "hi",
        tools=(ToolDefinition("lookup"),),
        provider_options={"test": True},
    )
    assert router.resolve("default", with_tool).adapter is large
    assert router.capabilities("large", with_tool).tools
    with pytest.raises(ValueError, match="already registered"):
        router.register("small", small, "other")
