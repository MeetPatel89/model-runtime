"""Retry policy used by :class:`model_runtime.ModelRuntime`."""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass

from .errors import ModelRuntimeError


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """Exponential backoff with proportional jitter.

    ``attempt`` values accepted by :meth:`should_retry` and :meth:`delay_for`
    are one-based and identify the attempt that just failed.
    """

    max_attempts: int = 3
    initial_delay: float = 0.5
    max_delay: float = 8.0
    multiplier: float = 2.0
    jitter: float = 0.2

    def __post_init__(self) -> None:
        """Validate the retry-policy configuration."""
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least one")
        if self.initial_delay < 0 or self.max_delay < 0:
            raise ValueError("retry delays cannot be negative")
        if self.max_delay < self.initial_delay:
            raise ValueError("max_delay cannot be less than initial_delay")
        if self.multiplier < 1:
            raise ValueError("multiplier must be at least one")
        if not 0 <= self.jitter <= 1:
            raise ValueError("jitter must be between zero and one")

    def should_retry(self, error: ModelRuntimeError, attempt: int) -> bool:
        """Return whether the error permits another attempt."""
        return error.retryable and attempt < self.max_attempts

    def delay_for(
        self,
        attempt: int,
        error: ModelRuntimeError,
        *,
        random_fn: Callable[[], float] = random.random,
    ) -> float:
        """Calculate the delay after a failed one-based ``attempt``.

        A provider's ``retry_after`` value takes precedence and is not jittered.
        """
        if attempt < 1:
            raise ValueError("attempt must be at least one")
        if error.retry_after is not None:
            return max(0.0, error.retry_after)

        delay = min(
            self.max_delay, self.initial_delay * self.multiplier ** (attempt - 1)
        )
        if not delay or not self.jitter:
            return delay
        offset = (random_fn() * 2.0 - 1.0) * delay * self.jitter
        return max(0.0, delay + offset)

    # A concise alias that reads naturally in custom runtimes.
    backoff = delay_for
