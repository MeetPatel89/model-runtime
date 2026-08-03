"""Provider-neutral exceptions exposed by the runtime boundary."""

from __future__ import annotations


class ModelRuntimeError(Exception):
    """Base class for all model invocation failures."""

    default_retryable = False

    def __init__(
        self,
        message: str,
        *,
        retryable: bool | None = None,
        retry_after: float | None = None,
        cause: BaseException | None = None,
        status_code: int | None = None,
        provider: str | None = None,
        details: object | None = None,
    ) -> None:
        super().__init__(message)
        self.retryable = self.default_retryable if retryable is None else retryable
        self.retry_after = retry_after
        self.status_code = status_code
        self.provider = provider
        self.details = details
        self.original_error = cause
        if cause is not None:
            self.__cause__ = cause


class AuthError(ModelRuntimeError):
    """Credentials are absent or not authorized for the request."""


class RateLimitError(ModelRuntimeError):
    """The provider throttled the request."""

    default_retryable = True


class RequestTimeout(ModelRuntimeError):  # noqa: N818 - public taxonomy name
    """The provider call exceeded its configured timeout."""

    default_retryable = True


class InvalidRequestError(ModelRuntimeError):
    """The normalized or provider-specific request is invalid."""


class ContentFilterError(ModelRuntimeError):
    """The provider rejected content under its safety policy."""


class ProviderUnavailableError(ModelRuntimeError):
    """The provider cannot currently serve the request."""

    default_retryable = True
