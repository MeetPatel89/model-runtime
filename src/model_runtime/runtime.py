"""The provider-independent model runtime facade."""

from __future__ import annotations

import asyncio
import inspect
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from types import MappingProxyType
from typing import Any

from .errors import (
    InvalidRequestError,
    ModelRuntimeError,
    ProviderUnavailableError,
    RequestTimeout,
)
from .retry import RetryPolicy
from .router import ModelRoute, ModelRouter
from .tracing import NoOpTraceObserver, TraceObserver
from .types import (
    ModelRequest,
    ModelResponse,
    StreamEnd,
    StreamEvent,
    TextDelta,
    ToolCallDelta,
    Usage,
)

Sleep = Callable[[float], Awaitable[None]]
Clock = Callable[[], float]


class ModelRuntime:
    """Own retries, timeouts, tracing, routing, and token accounting."""

    def __init__(
        self,
        router: ModelRouter,
        *,
        retry_policy: RetryPolicy | None = None,
        observer: TraceObserver | None = None,
        sleep: Sleep = asyncio.sleep,
        clock: Clock = time.monotonic,
    ) -> None:
        self.router = router
        self.retry_policy = retry_policy or RetryPolicy()
        self.observer = observer or NoOpTraceObserver()
        self._sleep = sleep
        self._clock = clock
        self._total_usage = Usage()
        self._usage_by_model: dict[str, Usage] = {}

    @property
    def total_usage(self) -> Usage:
        """Cumulative usage recorded by this runtime."""
        return self._total_usage

    @property
    def usage_by_model(self) -> Mapping[str, Usage]:
        """An immutable snapshot of cumulative usage by model ID."""
        return MappingProxyType(dict(self._usage_by_model))

    def usage_for(self, model_id: str) -> Usage:
        """Return cumulative usage for a model, or zero usage when absent."""
        return self._usage_by_model.get(model_id, Usage())

    def reset_usage(self) -> None:
        """Clear all usage totals tracked by this runtime."""
        self._total_usage = Usage()
        self._usage_by_model.clear()

    async def complete(self, model: str, request: ModelRequest) -> ModelResponse:
        """Complete a request with routing, retries, tracing, and accounting."""
        route = self._resolve(model, request)
        started = self._clock()
        await self._observe("on_request", route.model_id, request)

        attempt = 0
        while True:
            attempt += 1
            try:
                response = await self._with_timeout(
                    route.adapter.complete(route.model_id, request),
                    request.timeout,
                    route.model_id,
                )
                if not isinstance(response, ModelResponse):
                    raise ProviderUnavailableError(
                        "adapter returned an invalid completion response",
                        retryable=False,
                        provider=route.model_id,
                        details=response,
                    )
            except BaseException as exc:
                if isinstance(
                    exc,
                    (
                        KeyboardInterrupt,
                        SystemExit,
                        GeneratorExit,
                        asyncio.CancelledError,
                    ),
                ):
                    raise
                error = self._normalize_error(exc, route.model_id)
                if self.retry_policy.should_retry(error, attempt):
                    await self._sleep(self.retry_policy.delay_for(attempt, error))
                    continue
                await self._observe(
                    "on_error", route.model_id, error, self._clock() - started
                )
                raise error

            self._record_usage(route.model_id, response.usage)
            await self._observe(
                "on_response",
                route.model_id,
                response,
                self._clock() - started,
                response.usage,
            )
            return response

    async def stream(
        self, model: str, request: ModelRequest
    ) -> AsyncIterator[StreamEvent]:
        """Stream a request with retries before the first delta and final accounting."""
        route = self._resolve(model, request)
        started = self._clock()
        await self._observe("on_request", route.model_id, request)

        attempt = 0
        emitted_delta = False
        while True:
            attempt += 1
            iterator: AsyncIterator[StreamEvent] | None = None
            try:
                stream = route.adapter.stream(route.model_id, request)
                if inspect.isawaitable(stream):
                    stream = await self._with_timeout(
                        stream, request.timeout, route.model_id
                    )
                iterator = stream.__aiter__()

                while True:
                    try:
                        event = await self._with_timeout(
                            anext(iterator), request.timeout, route.model_id
                        )
                    except StopAsyncIteration:
                        raise ProviderUnavailableError(
                            "provider stream ended without a final StreamEnd event",
                            retryable=not emitted_delta,
                            provider=route.model_id,
                        ) from None

                    if isinstance(event, (TextDelta, ToolCallDelta)):
                        emitted_delta = True
                        yield event
                        continue
                    if not isinstance(event, StreamEnd):
                        raise ProviderUnavailableError(
                            "adapter yielded an invalid stream event",
                            retryable=False,
                            provider=route.model_id,
                            details=event,
                        )

                    usage = event.usage or event.response.usage
                    self._record_usage(route.model_id, usage)
                    await self._observe(
                        "on_response",
                        route.model_id,
                        event.response,
                        self._clock() - started,
                        usage,
                    )
                    yield event
                    return
            except BaseException as exc:
                if isinstance(
                    exc,
                    (
                        KeyboardInterrupt,
                        SystemExit,
                        GeneratorExit,
                        asyncio.CancelledError,
                    ),
                ):
                    raise
                error = self._normalize_error(exc, route.model_id)
                can_retry = not emitted_delta and self.retry_policy.should_retry(
                    error, attempt
                )
                if can_retry:
                    await self._close(iterator)
                    await self._sleep(self.retry_policy.delay_for(attempt, error))
                    continue
                await self._observe(
                    "on_error", route.model_id, error, self._clock() - started
                )
                raise error
            finally:
                await self._close(iterator)

    def _resolve(self, model: str, request: ModelRequest) -> ModelRoute:
        try:
            return self.router.resolve(model, request)
        except (KeyError, ValueError) as exc:
            raise InvalidRequestError(
                str(exc), retryable=False, cause=exc, provider=model
            ) from exc

    async def _with_timeout(
        self,
        value: Awaitable[Any],
        timeout: float | None,
        model_id: str,
    ) -> Any:
        try:
            if timeout is None:
                return await value
            return await asyncio.wait_for(value, timeout=timeout)
        except TimeoutError as exc:
            if timeout is None:
                message = f"request to {model_id!r} timed out"
            else:
                message = f"request to {model_id!r} timed out after {timeout:g} seconds"
            raise RequestTimeout(
                message,
                cause=exc,
                provider=model_id,
            ) from exc

    @staticmethod
    def _normalize_error(exc: BaseException, model_id: str) -> ModelRuntimeError:
        if isinstance(exc, ModelRuntimeError):
            return exc
        return ProviderUnavailableError(
            f"unexpected error from model {model_id!r}: {exc}",
            retryable=False,
            cause=exc,
            provider=model_id,
        )

    async def _observe(self, method_name: str, *args: Any) -> None:
        try:
            result = getattr(self.observer, method_name)(*args)
            if inspect.isawaitable(result):
                await result
        except Exception:  # noqa: BLE001 - observers are an isolated extension point
            # Instrumentation must not turn a successful provider call into a failure.
            return

    def _record_usage(self, model_id: str, usage: Usage) -> None:
        self._total_usage = self._total_usage + usage
        self._usage_by_model[model_id] = (
            self._usage_by_model.get(model_id, Usage()) + usage
        )

    @staticmethod
    async def _close(iterator: AsyncIterator[StreamEvent] | None) -> None:
        if iterator is None:
            return
        close = getattr(iterator, "aclose", None)
        if close is not None:
            try:
                await close()
            except Exception:  # noqa: BLE001 - best-effort cleanup must preserve the call error
                return
