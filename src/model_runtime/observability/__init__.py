"""Optional observability integrations for model-runtime."""

try:
    from .otel import LangfuseTraceAttributes, OTelTraceObserver
except ModuleNotFoundError as exc:
    if exc.name is None or not exc.name.startswith("opentelemetry"):
        raise
    raise ImportError(
        "OpenTelemetry support is not installed; install model-runtime[otel]"
    ) from exc

__all__ = ["LangfuseTraceAttributes", "OTelTraceObserver"]
