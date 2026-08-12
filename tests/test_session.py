"""Tests for app-agnostic conversation sessions and their sync bridge."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable

import pytest

from model_runtime import (
    ChatSession,
    Message,
    ModelCapabilities,
    ModelRequest,
    ModelResponse,
    ModelRouter,
    ModelRuntime,
    StreamEvent,
    Usage,
)


class RecordingModel:
    """Deterministic chat adapter that records normalized requests."""

    capabilities = ModelCapabilities(streaming=False)

    def __init__(self, response: ModelResponse) -> None:
        self.response = response
        self.requests: list[tuple[str, ModelRequest]] = []

    async def complete(self, model_id: str, request: ModelRequest) -> ModelResponse:
        """Record and return the configured response."""
        self.requests.append((model_id, request))
        return self.response

    async def stream(
        self, model_id: str, request: ModelRequest
    ) -> AsyncIterator[StreamEvent]:
        """Reject streaming, which is outside these session tests."""
        raise NotImplementedError
        yield


def make_session(
    model: RecordingModel,
    *,
    history: tuple[Message, ...] = (),
    system_prompt: str | None = None,
    clock: Callable[[], float] | None = None,
) -> ChatSession:
    """Build a single-route session around ``model``."""
    runtime = ModelRuntime(ModelRouter({"primary": (model, "configured-model")}))
    if clock is None:
        return ChatSession(
            runtime,
            "primary",
            history=history,
            system_prompt=system_prompt,
        )
    return ChatSession(
        runtime,
        "primary",
        history=history,
        system_prompt=system_prompt,
        clock=clock,
    )


@pytest.mark.asyncio
async def test_complete_turn_records_history_response_and_telemetry() -> None:
    """A successful turn records normalized state only after completion."""
    raw = {"id": "response-123", "provider_value": [1, 2]}
    response = ModelResponse(
        message=Message.assistant("Hello back"),
        usage=Usage(input_tokens=8, output_tokens=3),
        raw=raw,
        model="returned-model",
    )
    model = RecordingModel(response)
    ticks = iter((10.0, 10.0125))
    session = make_session(
        model,
        history=(Message.user("Earlier"), Message.assistant("Earlier answer")),
        system_prompt="Be concise.",
        clock=lambda: next(ticks),
    )

    record = await session.complete_turn("Hello")

    assert record.text == "Hello back"
    assert record.response is response
    assert record.response.raw is raw
    assert record.route == "primary"
    assert record.provider == "primary"
    assert record.model == "returned-model"
    assert record.response_id == "response-123"
    assert record.latency_ms == pytest.approx(12.5)
    assert session.history == (
        Message.user("Earlier"),
        Message.assistant("Earlier answer"),
        Message.user("Hello"),
        response.message,
    )
    assert session.generation_log == (record,)
    assert session.last_record is record
    assert model.requests == [
        (
            "configured-model",
            ModelRequest(
                (
                    Message.system("Be concise."),
                    Message.user("Earlier"),
                    Message.assistant("Earlier answer"),
                    Message.user("Hello"),
                )
            ),
        )
    ]


@pytest.mark.asyncio
async def test_transient_generation_message_stays_out_of_visible_history() -> None:
    """Retrieved context can affect generation without replacing the user turn."""
    model = RecordingModel(ModelResponse(message=Message.assistant("Grounded")))
    session = make_session(model)
    transient = Message.user("Question\n\nRetrieved private context")

    record = await session.complete_turn(
        Message.user("Question"),
        generation_message=transient,
    )

    assert record.model == "configured-model"
    assert model.requests[0][1].messages == (transient,)
    assert session.history == (
        Message.user("Question"),
        Message.assistant("Grounded"),
    )


def test_sync_bridge_runs_chat_and_preserves_session_controls() -> None:
    """Synchronous callers can chat and manage independent history state."""
    model = RecordingModel(ModelResponse(message=Message.assistant("Done")))
    session = make_session(model, system_prompt="Original")

    assert session.chat_sync("Start") == "Done"
    session.clear_history()
    session.set_system_prompt("Updated")

    assert session.history == ()
    assert session.system_prompt == "Updated"
    assert len(session.generation_log) == 1


@pytest.mark.asyncio
async def test_sync_bridge_rejects_calls_from_a_running_event_loop() -> None:
    """Sync wrappers direct async callers to await the native API."""
    model = RecordingModel(ModelResponse(message=Message.assistant("unused")))
    session = make_session(model)

    with pytest.raises(RuntimeError, match=r"await the async API"):
        session.chat_sync("Hello")
    with pytest.raises(RuntimeError, match=r"await the async API"):
        session.complete_turn_sync("Hello")

    assert model.requests == []


def test_session_rejects_non_conversation_roles() -> None:
    """Visible history and new turns accept only user/assistant roles."""
    model = RecordingModel(ModelResponse(message=Message.assistant("unused")))

    with pytest.raises(ValueError, match="history messages"):
        make_session(model, history=(Message.system("hidden"),))

    session = make_session(model)
    with pytest.raises(ValueError, match="message must have the user role"):
        session.complete_turn_sync(Message.assistant("injected"))
