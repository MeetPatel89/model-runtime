"""Shared structural inspection of HTTP-shaped provider exceptions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Protocol, cast, runtime_checkable


@runtime_checkable
class _HasStatusCode(Protocol):
    @property
    def status_code(self) -> object:
        """Potential HTTP status code."""
        ...


@runtime_checkable
class _HasBody(Protocol):
    @property
    def body(self) -> object:
        """Potential error body."""
        ...


@runtime_checkable
class _HasCode(Protocol):
    @property
    def code(self) -> object:
        """Potential provider error code."""
        ...


@runtime_checkable
class _HasResponse(Protocol):
    @property
    def response(self) -> object:
        """Potential HTTP response."""
        ...


@runtime_checkable
class _HasHeaders(Protocol):
    @property
    def headers(self) -> object:
        """Potential HTTP headers."""
        ...


def error_status_code(error: Exception) -> int | None:
    """Return an integer status code from an SDK-shaped exception."""
    if not isinstance(error, _HasStatusCode):
        return None
    value = error.status_code
    return value if isinstance(value, int) else None


def error_body(error: Exception) -> object | None:
    """Return a provider error body when the exception exposes one."""
    return error.body if isinstance(error, _HasBody) else None


def error_code(error: Exception) -> object | None:
    """Return a provider error code when the exception exposes one."""
    return error.code if isinstance(error, _HasCode) else None


def contains_error_marker(value: object, markers: Sequence[str]) -> bool:
    """Search nested error data for one normalized semantic marker."""
    if isinstance(value, Mapping):
        mapping = cast(Mapping[object, object], value)
        return any(contains_error_marker(item, markers) for item in mapping.values())
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        sequence = cast(Sequence[object], value)
        return any(contains_error_marker(item, markers) for item in sequence)
    if value is None:
        return False
    normalized = str(value).lower().replace("-", "_").replace(" ", "_")
    return any(marker in normalized for marker in markers)


def error_retry_after(error: Exception) -> float | None:
    """Parse Retry-After seconds, milliseconds, or an HTTP date."""
    if not isinstance(error, _HasResponse):
        return None
    response = error.response
    if not isinstance(response, _HasHeaders):
        return None
    headers = response.headers
    if not isinstance(headers, Mapping):
        return None
    raw_headers = cast(Mapping[object, object], headers)
    lowered = {str(key).lower(): value for key, value in raw_headers.items()}
    raw = lowered.get("retry-after")
    if raw is None:
        milliseconds = lowered.get("retry-after-ms")
        if milliseconds is None:
            return None
        return _nonnegative_number(milliseconds, divisor=1000.0)

    seconds = _nonnegative_number(raw)
    if seconds is not None:
        return seconds
    try:
        parsed = parsedate_to_datetime(str(raw))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        return max(0.0, (parsed - datetime.now(UTC)).total_seconds())
    except TypeError, ValueError, OverflowError:
        return None


def _nonnegative_number(value: object, *, divisor: float = 1.0) -> float | None:
    if not isinstance(value, str | bytes | bytearray | int | float):
        return None
    try:
        return max(0.0, float(value) / divisor)
    except TypeError, ValueError, OverflowError:
        return None
