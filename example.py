"""Example of concurrent OpenAI and Anthropic calls traced to local Langfuse."""

import asyncio
import os
from base64 import b64encode
from pathlib import Path

from dotenv import load_dotenv
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from model_runtime import (
    AnthropicAdapter,
    AnthropicProviderOptions,
    Message,
    ModelRequest,
    ModelRouter,
    ModelRuntime,
    OpenAIAdapter,
    OpenAIProviderOptions,
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
    """Run OpenAI and Anthropic completions as concurrent asyncio tasks."""
    load_dotenv()
    load_dotenv(Path(__file__).with_name("observability") / ".env")

    tracer_provider = TracerProvider(
        resource=Resource.create({"service.name": "model-runtime-example"})
    )
    try:
        tracer_provider.add_span_processor(BatchSpanProcessor(_langfuse_exporter()))
        tracer = tracer_provider.get_tracer("model_runtime.example")

        router = (
            ModelRouter()
            .register("gpt", OpenAIAdapter(), os.environ["OPENAI_MODEL"])
            .register("claude", AnthropicAdapter(), os.environ["ANTHROPIC_MODEL"])
        )
        openai_runtime = ModelRuntime(
            router,
            observer=OTelTraceObserver(tracer, provider_name="openai"),
        )
        anthropic_runtime = ModelRuntime(
            router,
            observer=OTelTraceObserver(tracer, provider_name="anthropic"),
        )
        messages = (
            Message.system("Answer clearly and briefly."),
            Message.user("What are great textbooks for Linear Algebra?"),
        )
        openai_request = ModelRequest(
            messages=messages,
            timeout=30,
            # OpenAI-specific options are translated only inside the OpenAI codec.
            provider_options=OpenAIProviderOptions(),
        )
        anthropic_request = ModelRequest(
            messages=messages,
            timeout=30,
            # Anthropic-specific options are translated only inside the Anthropic codec.
            provider_options=AnthropicProviderOptions(),
        )

        with tracer.start_as_current_span("example concurrent chat"):
            async with asyncio.TaskGroup() as tasks:
                openai_task = tasks.create_task(
                    openai_runtime.complete("gpt", openai_request)
                )
                anthropic_task = tasks.create_task(
                    anthropic_runtime.complete("claude", anthropic_request)
                )

        for label, response in (
            ("openai", openai_task.result()),
            ("anthropic", anthropic_task.result()),
        ):
            print(f"=== {label} ===")
            print(response.text)
            print(response.usage)
            print("\n\n")
    finally:
        tracer_provider.shutdown()


asyncio.run(main())
