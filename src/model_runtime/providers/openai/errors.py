"""Extraction of typed error metadata from OpenAI SDK exceptions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Protocol, cast, runtime_checkable

import openai

from ..errors import ProviderErrorKind, ProviderErrorMetadata


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


class OpenAIErrorMetadataExtractor:
    """Inspect official SDK errors plus SDK-shaped test doubles."""

    provider_name = "openai"

    def extract(self, error: Exception) -> ProviderErrorMetadata:
        """Extract status, retry guidance, details, and semantic kind."""
        status_code = self._status_code(error)
        details = error.body if isinstance(error, _HasBody) else None
        kind = self._kind(error, status_code, details)
        return ProviderErrorMetadata(
            message=str(error) or type(error).__name__,
            kind=kind,
            status_code=status_code,
            retry_after=self._retry_after(error),
            details=details,
        )

    @classmethod
    def _kind(
        cls,
        error: Exception,
        status_code: int | None,
        details: object | None,
    ) -> ProviderErrorKind:
        code = error.code if isinstance(error, _HasCode) else None
        if cls._contains_content_filter(code) or cls._contains_content_filter(details):
            return ProviderErrorKind.CONTENT_FILTER
        if cls._contains_content_filter(str(error)):
            return ProviderErrorKind.CONTENT_FILTER
        if isinstance(error, openai.AuthenticationError | openai.PermissionDeniedError):
            return ProviderErrorKind.AUTH
        if isinstance(error, openai.RateLimitError):
            return ProviderErrorKind.RATE_LIMIT
        if isinstance(error, openai.APITimeoutError):
            return ProviderErrorKind.TIMEOUT
        if isinstance(
            error,
            openai.BadRequestError
            | openai.NotFoundError
            | openai.ConflictError
            | openai.UnprocessableEntityError,
        ):
            return ProviderErrorKind.INVALID_REQUEST
        if isinstance(error, openai.APIConnectionError | openai.InternalServerError):
            return ProviderErrorKind.UNAVAILABLE
        if status_code in {401, 403}:
            return ProviderErrorKind.AUTH
        if status_code == 429:
            return ProviderErrorKind.RATE_LIMIT
        if status_code == 408:
            return ProviderErrorKind.TIMEOUT
        if status_code in {400, 404, 409, 422}:
            return ProviderErrorKind.INVALID_REQUEST
        if status_code is not None and status_code >= 500:
            return ProviderErrorKind.UNAVAILABLE
        return ProviderErrorKind.UNKNOWN

    @staticmethod
    def _status_code(error: Exception) -> int | None:
        if not isinstance(error, _HasStatusCode):
            return None
        value = error.status_code
        return value if isinstance(value, int) else None

    @classmethod
    def _contains_content_filter(cls, value: object) -> bool:
        if isinstance(value, Mapping):
            mapping = cast(Mapping[object, object], value)
            return any(cls._contains_content_filter(item) for item in mapping.values())
        if isinstance(value, Sequence) and not isinstance(
            value, str | bytes | bytearray
        ):
            sequence = cast(Sequence[object], value)
            return any(cls._contains_content_filter(item) for item in sequence)
        if value is None:
            return False
        normalized = str(value).lower().replace("-", "_").replace(" ", "_")
        return "content_filter" in normalized or "content_policy" in normalized

    @classmethod
    def _retry_after(cls, error: Exception) -> float | None:
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
            return cls._nonnegative_number(milliseconds, divisor=1000.0)

        seconds = cls._nonnegative_number(raw)
        if seconds is not None:
            return seconds
        try:
            parsed = parsedate_to_datetime(str(raw))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=UTC)
            return max(0.0, (parsed - datetime.now(UTC)).total_seconds())
        except TypeError, ValueError, OverflowError:
            return None

    @staticmethod
    def _nonnegative_number(value: object, *, divisor: float = 1.0) -> float | None:
        if not isinstance(value, str | bytes | bytearray | int | float):
            return None
        try:
            return max(0.0, float(value) / divisor)
        except TypeError, ValueError, OverflowError:
            return None
