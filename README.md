# model-runtime

`model-runtime` is an initial, async-first and strictly typed invocation layer for applications
that need to call LLMs without spreading provider SDK details through application code. It
provides normalized messages, responses, streams, errors, routing, retries, timeouts, tracing
hooks, and token accounting while retaining provider-specific JSON options and raw SDK payloads.

The [OpenAI Responses API](https://developers.openai.com/api/docs/guides/migrate-to-responses) is
the first built-in integration and is the only OpenAI generation endpoint used. The provider
lifecycle is implemented once through reusable transport, codec, stream-decoder, and error-mapping
boundaries so another provider does not need to duplicate adapter orchestration.

## Setup

Python 3.14 and [`uv`](https://docs.astral.sh/uv/) are required.

```bash
uv sync
export OPENAI_API_KEY="your-key"
export OPENAI_MODEL="your-openai-model-id"
```

The OpenAI SDK reads `OPENAI_API_KEY` when no explicit key is supplied. Credentials, organization,
project, and a custom base URL can instead be passed to `OpenAIAdapter`. This package does not use
global configuration of its own.

## Complete and streaming calls

The repository's [`example.py`](example.py) runs with `uv run python example.py`:

```python
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
    openai_options = OpenAIProviderOptions(store=False, reasoning_effort="low")
    request = ModelRequest(
        messages=(
            Message.system("Answer clearly and briefly."),
            Message.user("Why is the sky blue?"),
        ),
        timeout=30,
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
```

`runtime.total_usage` and `runtime.usage_by_model` contain process-local totals for successfully
completed calls and streams. A stream is counted after its final `StreamEnd` is received.

## Architecture

`ModelRuntime` receives a logical model name. `ModelRouter` resolves it to a `ChatModel` and the
provider's model ID. The runtime owns request timeouts, retry policy, tracing notifications, and
usage totals. Provider SDK retries stay disabled so there is one visible retry owner.

```text
application -> ModelRuntime -> ModelRouter -> ChatModel / ProviderAdapter
                 |                |                    |
                 |                |                    +-- ProviderCodec + StreamDecoder
                 |                |                    +-- ProviderErrorMapper
                 |                |                    +-- ProviderTransport -> provider SDK
                 |                +-- logical name -> provider model ID
                 +-- retry / timeout / tracing / usage
```

The reusable provider components have focused responsibilities:

| Component | Responsibility |
| --- | --- |
| `ProviderAdapter` | Common completion/stream lifecycle and normalized exception boundary |
| `ProviderTransport` | Typed SDK invocation and transport stream cleanup |
| `ProviderCodec` | Native request/response translation and per-stream decoder creation |
| `StreamDecoder` | Stateful chunk normalization and construction of one terminal `StreamEnd` |
| `ProviderErrorMapper` | Provider exception to `ModelRuntimeError` translation |

`OpenAIAdapter` assembles `OpenAITransport`, `OpenAICodec`, `OpenAIStreamDecoder`, and the standard
error mapper with OpenAI-specific metadata extraction. OpenAI request translation, networking,
stream state, and exception inspection therefore have separate reasons to change.

The default `RetryPolicy` uses exponential backoff with jitter and honors a provider
`Retry-After` header. Streaming calls retry only before the first delta; retrying after output has
been delivered could duplicate text or tool calls.

## Strict payload typing

The normalized library code does not expose `Any`:

- Tool arguments, tool JSON Schemas, and provider options use recursive `JsonValue` and
  `JsonObject` aliases.
- `OpenAIProviderOptions` checks common Responses API keys. Newly released fields can still pass
  through its explicit JSON-typed `extra` mapping before the class is updated.
- `ModelResponse.raw` and error details are `object | None` because the concrete value belongs to
  the selected SDK. Unlike `Any`, `object` requires a caller to narrow or cast before access.
- Flexible constructor sequences are normalized to canonical tuple attributes. Reading
  `Message.content`, `ModelRequest.messages`, `ModelRequest.stop`, or `StreamEnd.usage` does not
  retain a constructor-only union.

Provider option mappings are copied into an immutable top-level view. Nested values remain
unchanged so an adapter can pass provider payloads through without silently rewriting their
representation. Values are validated as JSON-compatible when a request is constructed.

Strict checking is enforced with BasedPyright in addition to Ruff's annotation rules. Explicit
`Any`, inferred `Any`, unknown types, and unnecessary type-ignore comments fail the type check.

## OpenAI Responses behavior

The OpenAI transport calls `AsyncOpenAI.responses.create` for complete and streaming requests.
The codec maps normalized values to Responses concepts rather than Chat Completions shapes:

- normalized messages become typed `input` items;
- assistant tool calls become `function_call` items and tool results become matching
  `function_call_output` items using the same `call_id`;
- function definitions use the Responses API's internally tagged tool shape;
- complete calls read typed `response.output` items; and
- streams consume typed text, refusal, function-argument, and terminal response events.

Responses-native hosted tools are combined with normalized function tools:

```python
options = OpenAIProviderOptions(
    store=False,
    tools=({"type": "web_search"},),
    text={"verbosity": "low"},
)
```

`reasoning_effort="high"` is an ergonomic shorthand for
`reasoning={"effort": "high"}`. Likewise, `verbosity="low"` maps to
`text={"verbosity": "low"}`. Pass structured-output configuration through `text`, including its
`format` member, as required by Responses.

## Errors

The public taxonomy consists of `AuthError`, `RateLimitError`, `RequestTimeout`,
`InvalidRequestError`, `ContentFilterError`, and `ProviderUnavailableError`. All inherit
`ModelRuntimeError`, expose `retryable` and optional `retry_after`, and preserve the provider
exception as `__cause__`.

Providers with HTTP-like failures can reuse `StandardProviderErrorMapper`. They supply only an
`ErrorMetadataExtractor` that reports typed status, retry, provider-code, and detail metadata.

## Routing and capabilities

Register any number of logical names:

```python
router = ModelRouter()
router.register("fast", adapter, "provider-fast-model")
router.register("careful", adapter, "provider-careful-model")

if router.capabilities("fast").vision:
    ...
```

An optional selector can map an application-level name to a route using the complete request:

```python
router = ModelRouter(
    selector=lambda name, request: "careful" if request.tools else "fast"
)
```

Selection policy belongs in that application callable; the router only stores and resolves
routes.

## Adding a provider

Implement the three small structural protocols and assemble `ProviderAdapter`; component classes
do not need to inherit framework base classes:

```python
adapter = ProviderAdapter[NativeRequest, NativeResponse, NativeChunk](
    transport=MyTransport(),
    codec=MyCodec(),
    error_mapper=MyErrorMapper(),
    capabilities=ModelCapabilities(tools=True, streaming=True),
)
```

`MyTransport.complete` accepts `NativeRequest` and returns `NativeResponse`; `stream` yields
`NativeChunk` values and closes the SDK stream. `MyCodec` implements `encode_request`,
`decode_response`, and `stream_decoder`. The decoder's `feed` method returns normalized deltas and
its `finish` method returns one `StreamEnd`. `MyErrorMapper.translate` returns a normalized runtime
error. The network-free [provider contract tests](tests/test_provider_adapter.py) are a concrete,
executable example.

Provider transports should not retry. Unknown options are interpreted only by the selected codec.
For OpenAI, `model`, `input`, `stream`, `temperature`, `max_output_tokens`, and `timeout` are owned
by normalized request fields and cannot be overridden through `provider_options`. Responses-native
`tools` are handled specially and combined with normalized function tools.

## Development

Tests use fake transports and official SDK response models; they make no network calls.

```bash
uv sync
uv run pytest
uv run basedpyright
uvx ruff check .
uvx ruff format --check .
```

## Data handling and failure behavior

Requests are sent directly to the provider selected by the router. This package does not persist
prompts, responses, credentials, or traces itself. OpenAI Responses are stored by the provider by
default; pass `OpenAIProviderOptions(store=False)` when provider-side storage is not desired. An
injected `TraceObserver` controls its own data handling. `ModelResponse.raw`, error details, and
tracing callbacks may contain provider payloads, so applications should apply their own redaction
and retention policies. Observer failures are isolated from model calls, while provider failures
cross the normalized error boundary.

## Project structure

```text
src/model_runtime/                     domain types, runtime, router, retry, and tracing
src/model_runtime/providers/base.py    reusable typed provider orchestration and protocols
src/model_runtime/providers/errors.py  shared provider error normalization
src/model_runtime/providers/openai/    OpenAI adapter, transport, codec, stream, and errors
tests/                                 network-free runtime and provider contract tests
AGENTS.md                              Codex-native repository instructions
.cursor/rules/                         Cursor-native repository instructions
```

## Current limitations

- OpenAI is the only included provider and uses the Responses API.
- OpenAI normalization captures output text, refusal text, and function calls. Reasoning, hosted
  tool, citation, and other provider-specific output items remain available in `ModelResponse.raw`.
- Replaying `ModelResponse.message` manually does not preserve OpenAI reasoning or hosted-tool
  items. Use `previous_response_id` with provider-side state when those items must carry into a
  later call.
- The Responses API does not accept stop sequences or message names; the OpenAI codec rejects
  normalized requests containing either instead of silently dropping them.
- Usage totals are in memory and reset when the process exits.
- The normalized image part accepts URLs and data URLs; file loading is left to the application.
- Provider-specific response fields are available through `ModelResponse.raw`, not normalized.
- An injected OpenAI-compatible client is assumed to have its own SDK retries disabled.
