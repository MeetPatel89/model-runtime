"""Provisional app-agnostic conversation state built on :class:`ModelRuntime`.

Conversation memory lives here so applications do not need to duplicate turn
handling, telemetry, or the synchronous bridge. This module is intentionally a
small in-process session layer until a dedicated memory library owns richer
persistence and multi-participant conversation concerns.
"""

from __future__ import annotations

import asyncio
import math
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TypeVar

from .runtime import ModelRuntime
from .types import Message, MessageRole, ModelRequest, ModelResponse

Clock = Callable[[], float]
T = TypeVar("T")


def run_sync(
    operation: Callable[[], Awaitable[T]],
    *,
    operation_name: str = "async operation",
) -> T:
    """Run an awaitable factory when no event loop is active in this thread.

    A factory is accepted instead of an already-created coroutine so rejecting a
    call from async code cannot leak an unawaited coroutine.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(operation())
    raise RuntimeError(
        f"{operation_name} cannot run synchronously while an event loop is running; "
        "await the async API instead"
    )


@dataclass(frozen=True, slots=True)
class GenerationRecord:
    """One completed response and the route telemetry captured for it."""

    response: ModelResponse
    route: str
    model: str
    provider: str
    latency_ms: float
    response_id: str | None = None

    def __post_init__(self) -> None:
        """Reject incomplete identifiers and invalid elapsed times."""
        for name, value in (
            ("route", self.route),
            ("model", self.model),
            ("provider", self.provider),
        ):
            if not value.strip():
                raise ValueError(f"{name} must not be blank")
        if self.latency_ms < 0 or not math.isfinite(self.latency_ms):
            raise ValueError("latency_ms must be finite and non-negative")

    @property
    def text(self) -> str:
        """The normalized text returned by the model."""
        return self.response.text


class ChatSession:
    """Own system context, turn history, and generation telemetry for one route."""

    def __init__(
        self,
        runtime: ModelRuntime,
        route: str,
        *,
        system_prompt: str | None = None,
        history: Sequence[Message] = (),
        clock: Clock = time.perf_counter,
    ) -> None:
        if not route.strip():
            raise ValueError("route must not be blank")
        normalized_history = tuple(history)
        for message in normalized_history:
            self._validate_history_message(message)

        self._runtime = runtime
        self._route = route
        self._system_prompt = system_prompt
        self._history = list(normalized_history)
        self._clock = clock
        self._generation_log: list[GenerationRecord] = []

    @property
    def runtime(self) -> ModelRuntime:
        """The model runtime used for generation."""
        return self._runtime

    @property
    def route(self) -> str:
        """The logical runtime route used by this session."""
        return self._route

    @property
    def system_prompt(self) -> str | None:
        """The system prompt prepended to each request, when configured."""
        return self._system_prompt

    @property
    def history(self) -> tuple[Message, ...]:
        """An immutable snapshot of visible user and assistant turns."""
        return tuple(self._history)

    @property
    def generation_log(self) -> tuple[GenerationRecord, ...]:
        """An immutable snapshot of successful generations."""
        return tuple(self._generation_log)

    @property
    def last_record(self) -> GenerationRecord | None:
        """The most recent successful generation, when one exists."""
        return self._generation_log[-1] if self._generation_log else None

    def clear_history(self) -> None:
        """Clear visible turns without changing system context or telemetry."""
        self._history.clear()

    def set_system_prompt(self, system_prompt: str | None) -> None:
        """Replace the system prompt without changing conversation turns."""
        self._system_prompt = system_prompt

    async def complete_turn(
        self,
        message: str | Message,
        *,
        generation_message: Message | None = None,
    ) -> GenerationRecord:
        """Generate and atomically record one user/assistant conversation turn.

        ``generation_message`` supplies transient user context to the model while
        ``message`` remains the canonical user turn stored in visible history.
        """
        user_message = self._as_user_message(message)
        outbound_message = generation_message or user_message
        self._validate_user_message(
            outbound_message,
            argument_name="generation_message",
        )

        request_messages: list[Message] = []
        if self._system_prompt is not None:
            request_messages.append(Message.system(self._system_prompt))
        request_messages.extend(self._history)
        request_messages.append(outbound_message)
        request = ModelRequest(messages=request_messages)

        started = self._clock()
        response = await self._runtime.complete(self._route, request)
        latency_ms = (self._clock() - started) * 1000
        self._validate_assistant_message(response.message)

        model = response.model
        if model is None:
            model = self._runtime.router.resolve(self._route, request).model_id

        record = GenerationRecord(
            response=response,
            route=self._route,
            model=model,
            provider=self._route,
            latency_ms=latency_ms,
            response_id=self._response_id(response.raw),
        )
        self._history.extend((user_message, response.message))
        self._generation_log.append(record)
        return record

    async def chat(self, message: str | Message) -> str:
        """Generate and record one turn, returning only its response text."""
        return (await self.complete_turn(message)).text

    def complete_turn_sync(
        self,
        message: str | Message,
        *,
        generation_message: Message | None = None,
    ) -> GenerationRecord:
        """Run :meth:`complete_turn` from synchronous code."""
        return run_sync(
            lambda: self.complete_turn(
                message,
                generation_message=generation_message,
            ),
            operation_name="complete_turn_sync()",
        )

    def chat_sync(self, message: str | Message) -> str:
        """Run :meth:`chat` from synchronous code."""
        return run_sync(
            lambda: self.chat(message),
            operation_name="chat_sync()",
        )

    @staticmethod
    def _as_user_message(message: str | Message) -> Message:
        if isinstance(message, Message):
            ChatSession._validate_user_message(message, argument_name="message")
            return message
        if not isinstance(message, str):
            raise TypeError("message must be a string or Message")
        return Message.user(message)

    @staticmethod
    def _validate_history_message(message: Message) -> None:
        if not isinstance(message, Message):
            raise TypeError("history must contain Message values")
        if message.role not in {MessageRole.USER, MessageRole.ASSISTANT}:
            raise ValueError("history messages must have the user or assistant role")

    @staticmethod
    def _validate_user_message(message: Message, *, argument_name: str) -> None:
        if not isinstance(message, Message):
            raise TypeError(f"{argument_name} must be a Message")
        if message.role is not MessageRole.USER:
            raise ValueError(f"{argument_name} must have the user role")

    @staticmethod
    def _validate_assistant_message(message: Message) -> None:
        if message.role is not MessageRole.ASSISTANT:
            raise ValueError("model response message must have the assistant role")

    @staticmethod
    def _response_id(raw: object | None) -> str | None:
        value = raw.get("id") if isinstance(raw, Mapping) else getattr(raw, "id", None)
        return value if isinstance(value, str) else None
