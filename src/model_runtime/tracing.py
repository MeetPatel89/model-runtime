"""Dependency-free tracing hooks for model invocations."""

from __future__ import annotations

from collections.abc import Awaitable
from typing import Protocol

from .errors import ModelRuntimeError
from .types import ModelRequest, ModelResponse, Usage

type ObserverResult = None | Awaitable[None]


class TraceObserver(Protocol):
    """Observe logical requests without coupling the runtime to a tracing SDK.

    Implementations may use regular or async methods. Observer failures are
    isolated from model calls by ``ModelRuntime``.
    """

    def on_request(self, model_id: str, request: ModelRequest) -> ObserverResult:
        """Observe the start of a request."""
        ...

    def on_response(
        self,
        model_id: str,
        response: ModelResponse,
        latency_seconds: float,
        usage: Usage,
    ) -> ObserverResult:
        """Observe a successful response."""
        ...

    def on_error(
        self,
        model_id: str,
        error: ModelRuntimeError,
        latency_seconds: float,
    ) -> ObserverResult:
        """Observe a terminal request failure."""
        ...


class NoOpTraceObserver:
    """Default observer that deliberately does nothing."""

    def on_request(self, model_id: str, request: ModelRequest) -> None:
        """Ignore the start of a request."""
        return None

    def on_response(
        self,
        model_id: str,
        response: ModelResponse,
        latency_seconds: float,
        usage: Usage,
    ) -> None:
        """Ignore a successful response."""
        return None

    def on_error(
        self,
        model_id: str,
        error: ModelRuntimeError,
        latency_seconds: float,
    ) -> None:
        """Ignore a terminal request failure."""
        return None
