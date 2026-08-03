"""Example of using the model_runtime library to interact with the OpenAI API."""

import asyncio
import os

from model_runtime import (
    Message,
    ModelRequest,
    ModelRouter,
    ModelRuntime,
    OpenAIAdapter,
    StreamEnd,
    TextDelta,
)

adapter = OpenAIAdapter()
router = ModelRouter().register("chat", adapter, os.environ["OPENAI_MODEL"])
runtime = ModelRuntime(router)


async def main() -> None:
    """Run a completion and a stream against the configured OpenAI model."""
    request = ModelRequest(
        messages=(
            Message.system("Answer clearly and briefly."),
            Message.user("Why is the sky blue?"),
        ),
        timeout=30,
        # OpenAI-specific options pass through to the SDK unchanged.
        provider_options={"seed": 7},
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
