# pyright: reportMissingImports=false, reportMissingModuleSource=false

"""Send one hand-made OTLP span directly to the local Langfuse backend."""

# /// script
# requires-python = ">=3.14"
# dependencies = [
#   "opentelemetry-exporter-otlp-proto-http>=1.44.0,<2.0.0",
#   "opentelemetry-sdk>=1.44.0,<2.0.0",
#   "python-dotenv>=1.2.2,<2.0.0",
# ]
# ///

from __future__ import annotations

import os
from base64 import b64encode
from pathlib import Path

from dotenv import load_dotenv

# ty: ignore[unresolved-import]
from opentelemetry.exporter.otlp.proto.http.trace_exporter import (  # type: ignore[import-not-found]
    OTLPSpanExporter,
)

# ty: ignore[unresolved-import]
from opentelemetry.sdk.resources import Resource  # type: ignore[import-not-found]

# ty: ignore[unresolved-import]
from opentelemetry.sdk.trace import TracerProvider  # type: ignore[import-not-found]

# ty: ignore[unresolved-import]
from opentelemetry.sdk.trace.export import (  # type: ignore[import-not-found]
    SimpleSpanProcessor,
    SpanExportResult,
)

# ty: ignore[unresolved-import]
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (  # type: ignore[import-not-found]
    InMemorySpanExporter,
)


def _exporter() -> OTLPSpanExporter:
    """Build a direct, authenticated Langfuse OTLP/HTTP exporter."""
    public_key = os.environ["LANGFUSE_PUBLIC_KEY"]
    secret_key = os.environ["LANGFUSE_SECRET_KEY"]
    credentials = b64encode(f"{public_key}:{secret_key}".encode()).decode()
    base_url = os.getenv("LANGFUSE_BASE_URL", "http://localhost:3000")
    return OTLPSpanExporter(
        endpoint=f"{base_url.rstrip('/')}/api/public/otel/v1/traces",
        headers={
            "Authorization": f"Basic {credentials}",
            "x-langfuse-ingestion-version": "4",
        },
    )


def main() -> None:
    """Create, end, flush, and identify one test span."""
    load_dotenv(Path(__file__).with_name(".env"))
    capture_exporter = InMemorySpanExporter()
    provider = TracerProvider(
        resource=Resource.create(
            {
                "service.name": "model-runtime-otel-smoke-test",
                "service.version": "manual",
            }
        )
    )
    provider.add_span_processor(SimpleSpanProcessor(capture_exporter))
    tracer = provider.get_tracer("model_runtime.observability.smoke_test")

    with tracer.start_as_current_span(
        "manual OTLP smoke test",
        attributes={
            "smoke_test.phase": 0,
            "smoke_test.purpose": "verify Langfuse OTLP ingestion",
        },
    ) as span:
        trace_id = span.get_span_context().trace_id

    result = SpanExportResult.FAILURE
    try:
        otlp_exporter = _exporter()
        try:
            result = otlp_exporter.export(capture_exporter.get_finished_spans())
        finally:
            otlp_exporter.shutdown()
    finally:
        provider.shutdown()
    if result is not SpanExportResult.SUCCESS:
        raise RuntimeError("Langfuse rejected or could not receive the smoke-test span")
    print(f"exported smoke-test trace {trace_id:032x}; inspect it in Langfuse")


if __name__ == "__main__":
    main()
