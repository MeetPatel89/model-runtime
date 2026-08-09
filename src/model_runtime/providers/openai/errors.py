"""Extraction of typed error metadata from OpenAI SDK exceptions."""

from __future__ import annotations

from openai import (
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    ConflictError,
    InternalServerError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitError,
    UnprocessableEntityError,
)

from .._http import (
    contains_error_marker,
    error_body,
    error_code,
    error_retry_after,
    error_status_code,
)
from ..errors import ProviderErrorKind, ProviderErrorMetadata


class OpenAIErrorMetadataExtractor:
    """Inspect official SDK errors plus SDK-shaped test doubles."""

    provider_name = "openai"

    def extract(self, error: Exception) -> ProviderErrorMetadata:
        """Extract status, retry guidance, details, and semantic kind."""
        status_code = error_status_code(error)
        details = error_body(error)
        kind = self._kind(error, status_code, details)
        return ProviderErrorMetadata(
            message=str(error) or type(error).__name__,
            kind=kind,
            status_code=status_code,
            retry_after=error_retry_after(error),
            details=details,
        )

    @classmethod
    def _kind(
        cls,
        error: Exception,
        status_code: int | None,
        details: object | None,
    ) -> ProviderErrorKind:
        code = error_code(error)
        if cls._contains_content_filter(code) or cls._contains_content_filter(details):
            return ProviderErrorKind.CONTENT_FILTER
        if cls._contains_content_filter(str(error)):
            return ProviderErrorKind.CONTENT_FILTER
        if isinstance(error, AuthenticationError | PermissionDeniedError):
            return ProviderErrorKind.AUTH
        if isinstance(error, RateLimitError):
            return ProviderErrorKind.RATE_LIMIT
        if isinstance(error, APITimeoutError):
            return ProviderErrorKind.TIMEOUT
        if isinstance(
            error,
            BadRequestError | NotFoundError | ConflictError | UnprocessableEntityError,
        ):
            return ProviderErrorKind.INVALID_REQUEST
        if isinstance(error, APIConnectionError | InternalServerError):
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
    def _contains_content_filter(value: object) -> bool:
        return contains_error_marker(value, ("content_filter", "content_policy"))
