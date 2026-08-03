"""Reusable normalization of provider error metadata."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol

from ..errors import (
    AuthError,
    ContentFilterError,
    InvalidRequestError,
    ModelRuntimeError,
    ProviderUnavailableError,
    RateLimitError,
    RequestTimeout,
)


class ProviderErrorKind(str, Enum):
    """Provider-neutral classifications discovered at an SDK boundary."""

    AUTH = "auth"
    RATE_LIMIT = "rate_limit"
    TIMEOUT = "timeout"
    INVALID_REQUEST = "invalid_request"
    CONTENT_FILTER = "content_filter"
    UNAVAILABLE = "unavailable"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class ProviderErrorMetadata:
    """Typed facts extracted from an otherwise provider-native exception."""

    message: str
    kind: ProviderErrorKind = ProviderErrorKind.UNKNOWN
    status_code: int | None = None
    retry_after: float | None = None
    details: object | None = None


class ErrorMetadataExtractor(Protocol):
    """Extract normalized facts without deciding the public exception class."""

    @property
    def provider_name(self) -> str:
        """Stable provider name included in normalized errors."""
        ...

    def extract(self, error: Exception) -> ProviderErrorMetadata:
        """Inspect one provider-native exception."""
        ...


class StandardProviderErrorMapper:
    """Apply the shared runtime taxonomy to provider-specific metadata."""

    def __init__(self, extractor: ErrorMetadataExtractor) -> None:
        self._extractor = extractor

    def translate(self, error: Exception) -> ModelRuntimeError:
        """Translate an SDK exception using explicit kind then HTTP status."""
        if isinstance(error, ModelRuntimeError):
            return error

        metadata = self._extractor.extract(error)
        kind = self._resolved_kind(metadata)

        if kind is ProviderErrorKind.CONTENT_FILTER:
            return ContentFilterError(
                metadata.message,
                retryable=False,
                cause=error,
                status_code=metadata.status_code,
                provider=self._extractor.provider_name,
                details=metadata.details,
            )
        if kind is ProviderErrorKind.AUTH:
            return AuthError(
                metadata.message,
                retryable=False,
                cause=error,
                status_code=metadata.status_code,
                provider=self._extractor.provider_name,
                details=metadata.details,
            )
        if kind is ProviderErrorKind.RATE_LIMIT:
            return RateLimitError(
                metadata.message,
                retry_after=metadata.retry_after,
                cause=error,
                status_code=metadata.status_code,
                provider=self._extractor.provider_name,
                details=metadata.details,
            )
        if kind is ProviderErrorKind.TIMEOUT:
            return RequestTimeout(
                metadata.message,
                cause=error,
                status_code=metadata.status_code,
                provider=self._extractor.provider_name,
                details=metadata.details,
            )
        if kind is ProviderErrorKind.INVALID_REQUEST:
            return InvalidRequestError(
                metadata.message,
                retryable=False,
                cause=error,
                status_code=metadata.status_code,
                provider=self._extractor.provider_name,
                details=metadata.details,
            )
        if kind is ProviderErrorKind.UNAVAILABLE:
            return ProviderUnavailableError(
                metadata.message,
                retryable=True,
                retry_after=metadata.retry_after,
                cause=error,
                status_code=metadata.status_code,
                provider=self._extractor.provider_name,
                details=metadata.details,
            )
        return ProviderUnavailableError(
            metadata.message,
            retryable=False,
            cause=error,
            status_code=metadata.status_code,
            provider=self._extractor.provider_name,
            details=metadata.details,
        )

    @staticmethod
    def _resolved_kind(metadata: ProviderErrorMetadata) -> ProviderErrorKind:
        if metadata.kind is not ProviderErrorKind.UNKNOWN:
            return metadata.kind
        if metadata.status_code in {401, 403}:
            return ProviderErrorKind.AUTH
        if metadata.status_code == 429:
            return ProviderErrorKind.RATE_LIMIT
        if metadata.status_code == 408:
            return ProviderErrorKind.TIMEOUT
        if metadata.status_code in {400, 404, 409, 422}:
            return ProviderErrorKind.INVALID_REQUEST
        if metadata.status_code is not None and metadata.status_code >= 500:
            return ProviderErrorKind.UNAVAILABLE
        return ProviderErrorKind.UNKNOWN
