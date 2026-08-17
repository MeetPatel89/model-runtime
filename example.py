"""Example of OpenAI calls traced to a local Langfuse backend."""

import asyncio
import os
from base64 import b64encode
from pathlib import Path

from dotenv import load_dotenv
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from model_runtime import (
    Message,
    ModelRequest,
    ModelRouter,
    ModelRuntime,
    OpenAIAdapter,
    OpenAIProviderOptions,
    StreamEnd,
    TextDelta,
)
from model_runtime.observability import OTelTraceObserver


def _langfuse_exporter() -> OTLPSpanExporter:
    """Build an authenticated OTLP/HTTP exporter for local Langfuse."""
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


async def main() -> None:
    """Run a completion and a stream against the configured OpenAI model."""
    load_dotenv()
    load_dotenv(Path(__file__).with_name("observability") / ".env")

    tracer_provider = TracerProvider()
    try:
        tracer_provider.add_span_processor(BatchSpanProcessor(_langfuse_exporter()))
        tracer = tracer_provider.get_tracer("model_runtime.example")

        adapter = OpenAIAdapter()
        router = ModelRouter().register("chat", adapter, os.environ["OPENAI_MODEL"])
        runtime = ModelRuntime(
            router,
            observer=OTelTraceObserver(tracer, provider_name="openai"),
        )
        openai_options = OpenAIProviderOptions(store=False, reasoning_effort="low")
        request = ModelRequest(
            messages=(
                Message.system("Answer clearly and briefly."),
                Message.user("Why is the sky blue?"),
            ),
            timeout=30,
            # OpenAI-specific options are translated only inside the OpenAI codec.
            provider_options=openai_options,
        )

        response = await runtime.complete("chat", request)
        print(response.text)
        print(response.usage)

        async for event in runtime.stream("chat", request):
            if isinstance(event, TextDelta):
                print(event.delta, end="", flush=True)
            elif isinstance(event, StreamEnd):
                print(f"\nstream usage: {event.usage}")
    finally:
        tracer_provider.shutdown()


asyncio.run(main())
