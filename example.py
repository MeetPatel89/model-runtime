"""Example of using model_runtime with the OpenAI Responses API."""

import asyncio
import os

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

adapter = OpenAIAdapter()
router = ModelRouter().register("chat", adapter, os.environ["OPENAI_MODEL"])
runtime = ModelRuntime(router)


async def main() -> None:
    """Run a completion and a stream against the configured OpenAI model."""
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


asyncio.run(main())
