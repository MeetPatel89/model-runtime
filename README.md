# model-runtime

`model-runtime` is an initial, async-first invocation layer for applications that need to call
LLMs without spreading provider SDK details through application code. It provides normalized
messages, responses, streaming events, errors, routing, retries, timeouts, tracing hooks, and
token accounting. Provider-specific request options and raw responses remain available, so the
abstraction does not hide new provider capabilities.

The first built-in adapter uses OpenAI Chat Completions. The adapter contract is intentionally
small enough for Anthropic, Google, or local inference adapters to be added independently.

## Setup

Python 3.14 and [`uv`](https://docs.astral.sh/uv/) are required.

```bash
uv sync
export OPENAI_API_KEY="your-key"
export OPENAI_MODEL="your-openai-model-id"
```

The OpenAI SDK reads `OPENAI_API_KEY` when no explicit key is supplied. Credentials and a custom
base URL can instead be passed directly to `OpenAIAdapter`; the package does not use global
configuration.

## Complete and streaming calls

Save this as `example.py` and run it with `uv run python example.py`:

```python
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
```

`runtime.total_usage` and `runtime.usage_by_model` contain process-local totals for successfully
completed calls and streams. A stream is counted when its final `StreamEnd` is received.

## Architecture

The application calls `ModelRuntime` with a logical model name. `ModelRouter` resolves that name
to a `ChatModel` adapter and provider model ID. The runtime applies the request timeout, retries
only normalized errors marked as retryable, notifies an optional `TraceObserver`, and records
usage. The adapter translates the request and response at the SDK boundary.

```text
application -> ModelRuntime -> ModelRouter -> ChatModel adapter -> provider SDK
                 |                |
                 |                +-- logical name -> provider model ID
                 +-- retry / timeout / tracing / usage
```

OpenAI SDK retries are disabled by `OpenAIAdapter`, leaving one predictable retry owner. The
default `RetryPolicy` uses exponential backoff with jitter and honors a provider `Retry-After`
header. Streaming calls are retried only before the first delta; retrying after output has been
delivered could duplicate content or tool calls.

The public error boundary consists of:

- `AuthError`
- `RateLimitError`
- `RequestTimeout`
- `InvalidRequestError`
- `ContentFilterError`
- `ProviderUnavailableError`

All inherit `ModelRuntimeError`, expose `retryable` and optional `retry_after`, and preserve the
provider exception as `__cause__`.

## Routing and capabilities

Register any number of logical names:

```python
router = ModelRouter()
router.register("fast", adapter, "provider-fast-model")
router.register("careful", adapter, "provider-careful-model")

if router.capabilities("fast").vision:
    ...
```

An optional selector can map an application-level name to a registered route based on the full
request:

```python
router = ModelRouter(selector=lambda name, request: "careful" if request.tools else "fast")
```

Policy belongs in that application-supplied callable; the router only stores and resolves routes.

## Adding an adapter

Implement the `ChatModel` protocol:

```python
from collections.abc import AsyncIterator

from model_runtime import (
    ModelCapabilities,
    ModelRequest,
    ModelResponse,
    StreamEvent,
)


class MyAdapter:
    capabilities = ModelCapabilities(tools=True, streaming=True)

    async def complete(self, model_id: str, request: ModelRequest) -> ModelResponse: ...

    async def stream(self, model_id: str, request: ModelRequest) -> AsyncIterator[StreamEvent]:
        # Yield TextDelta/ToolCallDelta values and exactly one final StreamEnd.
        ...
```

An adapter is responsible only for translating normalized values, extracting usage, retaining the
raw provider response, and mapping SDK failures to `ModelRuntimeError` subclasses. It should not
perform its own retries. Unknown provider options are adapter-specific and pass through without
filtering. `model`, `messages`, and `stream` are reserved because the adapter owns call identity
and invocation mode.

## Development

Tests use fake adapters and SDK responses; they make no network calls.

```bash
uv sync
uv run pytest
uvx ruff check .
uvx ruff format --check .
```

## Data handling and failure behavior

Requests are sent directly to the provider selected by the router; this package does not persist
prompt, response, credential, or trace data. An injected `TraceObserver` controls its own data
handling. `ModelResponse.raw` and tracing callbacks may contain provider payloads, so applications
should apply their own redaction and retention policies. Observer failures are isolated from model
calls, while provider and adapter failures are surfaced through the normalized error taxonomy.

## Project structure

```text
src/model_runtime/        shared types, runtime, router, retry, and tracing contracts
src/model_runtime/providers/openai.py
tests/                    network-free runtime and adapter tests
```

## Current limitations

- OpenAI is the only included provider, using the Chat Completions API.
- Usage totals are in memory and reset when the process exits.
- The normalized image part accepts URLs and data URLs; file loading is left to the application.
- Provider-specific response fields are available through `ModelResponse.raw`, not normalized.
- An injected OpenAI client is assumed to have its own SDK retries disabled.
