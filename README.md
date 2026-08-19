# model-runtime

`model-runtime` is an initial, async-first and strictly typed invocation layer for applications
that need to call LLMs without spreading provider SDK details through application code. It
provides normalized messages, responses, streams, errors, routing, retries, timeouts, tracing
hooks, token accounting, and in-process conversation sessions while retaining provider-specific
JSON options and raw SDK payloads.

Built-in integrations use the [OpenAI Responses
API](https://developers.openai.com/api/docs/guides/migrate-to-responses) and the [Anthropic
Messages API](https://platform.claude.com/docs/en/api/messages/create). The provider lifecycle is
implemented once through reusable transport, codec, stream-decoder, and error-mapping boundaries,
so each integration contains provider translation rather than duplicated adapter orchestration.

## Setup

Python 3.14 and [`uv`](https://docs.astral.sh/uv/) are required.

```bash
uv sync
export OPENAI_API_KEY="your-key"
export OPENAI_MODEL="your-openai-model-id"
# Or, when using Anthropic:
export ANTHROPIC_API_KEY="your-key"
export ANTHROPIC_MODEL="your-anthropic-model-id"
```

The OpenAI SDK reads `OPENAI_API_KEY` when no explicit key is supplied. Credentials, organization,
project, and a custom base URL can instead be passed to `OpenAIAdapter`. This package does not use
global configuration of its own. Likewise, the Anthropic SDK reads `ANTHROPIC_API_KEY`; an API key,
auth token, and custom base URL can instead be passed to `AnthropicAdapter`.

### Local Langfuse backend

[`observability/README.md`](observability/README.md) contains the local Langfuse v4 Docker Compose
stack, a standalone OTLP ingestion smoke test, and the Phase 1 runtime instrumentation guide.
OpenTelemetry remains optional:

```bash
uv sync --extra otel
```

Import `OTelTraceObserver` from `model_runtime.observability`, inject an application-owned OTel
`Tracer`, and pass the observer to `ModelRuntime`. The base installation and `import model_runtime`
do not require OpenTelemetry.

### Optional aiohttp transport

Both provider SDKs use `httpx` asynchronously by default. The SDK and adapter APIs remain async
when `DefaultAioHttpClient` replaces only their low-level HTTP transport. Applications with high
request concurrency can opt into this `aiohttp` support:

```bash
uv sync --extra aiohttp
```

Construct the HTTP client supplied by the relevant SDK, pass it to that SDK's async client, and
inject the context-managed SDK client into the adapter:

```python
from anthropic import AsyncAnthropic, DefaultAioHttpClient as AnthropicAioHttpClient
from openai import AsyncOpenAI, DefaultAioHttpClient as OpenAIAioHttpClient

from model_runtime import AnthropicAdapter, OpenAIAdapter


async def use_openai_aiohttp() -> None:
    http_client = OpenAIAioHttpClient()
    async with AsyncOpenAI(
        http_client=http_client,
        max_retries=0,
    ) as client:
        adapter = OpenAIAdapter(client=client)
        # Register and use adapter while the SDK client context is open.


async def use_anthropic_aiohttp() -> None:
    http_client = AnthropicAioHttpClient()
    async with AsyncAnthropic(
        http_client=http_client,
        max_retries=0,
    ) as client:
        adapter = AnthropicAdapter(client=client)
        # Register and use adapter while the SDK client context is open.
```

Injected SDK clients are caller-owned: adapters do not close them, so keep each adapter's use
inside the client context or otherwise close the client explicitly. `max_retries=0` keeps
`ModelRuntime` as the sole retry owner.

The optional backend can improve HTTP throughput under heavy concurrency; it does not make model
inference faster or guarantee lower latency for an individual request. `httpx` remains the default
because workload concurrency is application-specific and the compatibility layer does not support
every HTTPX feature, including write timeouts, event hooks, SOCKS or HTTPS proxies, and some HTTPX
extensions. Enable `aiohttp` only after measuring a representative application workload. See the
[OpenAI SDK guidance](https://github.com/openai/openai-python#with-aiohttp), [Anthropic SDK
guidance](https://platform.claude.com/docs/en/cli-sdks-libraries/sdks/python#using-aiohttp-for-better-concurrency),
and [`httpx-aiohttp` compatibility notes](https://karpetrosyan.github.io/httpx-aiohttp/compatibility/)
for details.

## Complete and streaming calls

The repository's [`example.py`](example.py) runs OpenAI and Anthropic completions as concurrent
asyncio tasks and exports one span for each finished logical call to the local Langfuse backend.
After starting Langfuse and configuring the environment as described in the observability guide,
run:

```bash
uv run --extra otel python example.py
```

The application owns OTel setup and shutdown; `ModelRuntime` receives only the observer:

```python
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from model_runtime import ModelRuntime
from model_runtime.observability import OTelTraceObserver


tracer_provider = TracerProvider(
    resource=Resource.create({"service.name": "my_application"}),
)
tracer_provider.add_span_processor(BatchSpanProcessor(langfuse_exporter))
tracer = tracer_provider.get_tracer("my_application")
runtime = ModelRuntime(
    router,
    observer=OTelTraceObserver(tracer, provider_name="openai"),
)

try:
    response = await runtime.complete("chat", request)
finally:
    tracer_provider.shutdown()
```

Here `router`, `request`, and `langfuse_exporter` are application values; the executable example
shows their complete construction, including Basic-auth OTLP/HTTP exporter configuration.
`provider_name` is explicit because a provider cannot be inferred reliably from its model ID.

`runtime.total_usage` and `runtime.usage_by_model` contain process-local totals for successfully
completed calls and streams. A stream is counted after its final `StreamEnd` is received.

To route the same normalized request through Anthropic, only the adapter, model ID, and provider
options change:

```python
from model_runtime import AnthropicAdapter

anthropic_adapter = AnthropicAdapter(default_max_output_tokens=1024)
router.register("claude", anthropic_adapter, os.environ["ANTHROPIC_MODEL"])
request = ModelRequest.from_text(
    "Why is the sky blue?",
)
response = await runtime.complete("claude", request)
```

## Stateful conversation sessions

`ChatSession` adds app-agnostic system context, visible turn history, and generation telemetry on
top of one logical runtime route. It is provisional infrastructure for simple in-process
conversations; durable or shared memory belongs in a dedicated memory library.

```python
from model_runtime import ChatSession, Message

session = ChatSession(
    runtime,
    "chat",
    system_prompt="Answer clearly and briefly.",
)


async def ask() -> None:
    record = await session.complete_turn("Why is the sky blue?")
    print(record.text)
    print(record.response.usage)


# Synchronous applications can use the guarded bridge when no event loop is active.
answer = session.chat_sync("What changes at sunset?")
```

`complete_turn` and `chat` are async-native. `complete_turn_sync` and `chat_sync` use
`asyncio.run` and raise a clear `RuntimeError` if called from a thread that already has a running
event loop; async callers should await the native methods instead. Successful turns append the
canonical user message and full normalized assistant message atomically. A failed call leaves
history and the generation log unchanged.

Pass a separate user-role `generation_message` to add transient context while keeping the
canonical question in visible history:

```python
record = await session.complete_turn(
    Message.user("What is the escalation threshold?"),
    generation_message=Message.user(
        "What is the escalation threshold?\n\nRetrieved context: Severity two."
    ),
)
```

`session.history` and `session.generation_log` return tuple snapshots. Each frozen
`GenerationRecord` exposes the response, text, logical route, resolved model, route-derived
provider name, elapsed milliseconds, and provider response ID when present. The record retains
`ModelResponse.raw` as-is; callers own any validation, redaction, serialization, or copying of
provider-native values.

## Architecture

`ChatSession` optionally owns conversation state around `ModelRuntime`. `ModelRuntime` receives a
logical model name, and `ModelRouter` resolves it to a `ChatModel` and the provider's model ID. The
runtime owns request timeouts, retry policy, tracing notifications, and usage totals. Provider SDK
retries stay disabled so there is one visible retry owner.

For a step-by-step walkthrough of one complete and streaming call through both OpenAI and
Anthropic — including encode/decode, tool rounds, retries, and sequence diagrams — see
[`docs/end-to-end-request-flow.md`](docs/end-to-end-request-flow.md).

```text
application -> ChatSession -> ModelRuntime -> ModelRouter -> ChatModel / ProviderAdapter
                    |            |                |                    |
                    |            |                |                    +-- ProviderCodec + StreamDecoder
                    |            |                |                    +-- ProviderErrorMapper
                    |            |                |                    +-- ProviderTransport -> provider SDK
                    |            |                +-- logical name -> provider model ID
                    |            +-- retry / timeout / tracing / usage
                    +-- system prompt / turn history / generation log / sync bridge
```

The reusable provider components have focused responsibilities:

| Component | Responsibility |
| --- | --- |
| `ChatSession` | In-process system context, atomic conversation turns, generation records, and sync bridge |
| `ProviderAdapter` | Common completion/stream lifecycle and normalized exception boundary |
| `ProviderTransport` | Typed SDK invocation and transport stream cleanup |
| `ProviderCodec` | Native request/response translation and per-stream decoder creation |
| `StreamDecoder` | Stateful chunk normalization and construction of one terminal `StreamEnd` |
| `ProviderErrorMapper` | Provider exception to `ModelRuntimeError` translation |

`OpenAIAdapter` and `AnthropicAdapter` each assemble provider-specific transports, codecs, stream
decoders, and metadata extractors around the same `ProviderAdapter` and standard error mapper.
Request translation, networking, stream state, and exception inspection therefore remain focused
components without duplicating invocation orchestration.

The default `RetryPolicy` uses exponential backoff with jitter and honors a provider
`Retry-After` header. Streaming calls retry only before the first delta; retrying after output has
been delivered could duplicate text or tool calls.

## Strict payload typing

The normalized library code does not expose `Any`:

- Tool arguments, tool JSON Schemas, and provider options use recursive `JsonValue` and
  `JsonObject` aliases.
- `OpenAIProviderOptions` checks common Responses API keys. Newly released fields can still pass
  through its explicit JSON-typed `extra` mapping before the class is updated.
- `AnthropicProviderOptions` does the same for Messages API options and combines provider-native
  tools with normalized function tools.
- `ModelResponse.raw` and error details are `object | None` because the concrete value belongs to
  the selected SDK. Unlike `Any`, `object` requires a caller to narrow or cast before access.
- `GenerationRecord` retains the complete `ModelResponse`, including its provider-native `raw`
  value, without a Pydantic or JSON round trip.
- Flexible constructor sequences are normalized to canonical tuple attributes. Reading
  `Message.content`, `ModelRequest.messages`, `ModelRequest.stop`, or `StreamEnd.usage` does not
  retain a constructor-only union.

Provider option mappings are copied into an immutable top-level view. Nested values remain
unchanged so an adapter can pass provider payloads through without silently rewriting their
representation. Values are validated as JSON-compatible when a request is constructed.

Strict checking is enforced with [`ty`](https://docs.astral.sh/ty/) and mypy, complemented by
Ruff's annotation and stub rules. Mypy loads Pydantic's official plugin with constructor settings
that preserve Pydantic's coercive runtime behavior while warning about required dynamic aliases.
The checked-in VS Code configuration uses ty as the Python language server, runs mypy from the
project environment, and disables Pylance/Pyright and BasedPyright type diagnostics.

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

## Anthropic Messages behavior

The Anthropic transport uses `AsyncAnthropic.messages.create` for complete calls and the SDK's
typed `messages.stream` helper for streams. The codec maps normalized values to Messages concepts:

- leading system and developer messages become top-level `system` text blocks;
- later system and developer messages retain their position as Anthropic mid-conversation
  `system` messages;
- assistant tool calls become `tool_use` blocks and normalized tool messages become user
  `tool_result` blocks with the matching `tool_use_id`;
- HTTP image URLs and base64 JPEG, PNG, GIF, and WebP data URLs become Anthropic image sources;
- complete calls read typed text and `tool_use` content blocks; and
- streams consume typed text, partial tool-input JSON, and terminal message events.

The Messages API requires `max_tokens`. `AnthropicAdapter` uses
`default_max_output_tokens=1024` when `ModelRequest.max_output_tokens` is omitted; pass a different
positive constructor default when appropriate. Anthropic usage reports regular, cache-creation,
and cache-read input tokens separately. Normalized `Usage.input_tokens` contains their sum, while
`Usage.cached_tokens` contains the cache-read subset.

Provider-native tools and structured output remain available without changing normalized types:

```python
options = AnthropicProviderOptions(
    tools=({"type": "web_search_20260318", "name": "web_search"},),
    output_config={
        "effort": "high",
        "format": {
            "type": "json_schema",
            "schema": {"type": "object"},
        },
    },
)
```

`effort="high"` is shorthand for `output_config={"effort": "high"}` and cannot be combined with
an explicit `output_config`. Use `thinking`, `tool_choice`, `cache_control`, `service_tier`, and
other named `AnthropicProviderOptions` fields for their corresponding Messages features. These
features are model-specific: only send `effort` to a model listed as compatible in Anthropic's
[effort documentation](https://platform.claude.com/docs/en/build-with-claude/effort), or check the
model's reported capabilities first. Unsupported options remain visible as `InvalidRequestError`
rather than being silently removed by the adapter.

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

Capabilities describe what an adapter can represent. Providers can expose different feature sets
for different model IDs, so model-specific options such as Anthropic `effort` must also be checked
against the provider's current model capabilities.

An optional selector can map an application-level name to a route using the complete request:

```python
router = ModelRouter(
    selector=lambda name, request: "careful" if request.tools else "fast"
)
```

Selection policy belongs in that application callable; the router only stores and resolves
routes.

`ModelCatalog` is a separate optional protocol, so custom `ChatModel` adapters and test fakes are
not required to support discovery. The built-in adapters implement it asynchronously through the
same injected SDK client used for generation:

```python
openai_models = await OpenAIAdapter().list_models()
anthropic_models = await AnthropicAdapter().list_models()
```

Discovery failures use the same `ModelRuntimeError` taxonomy and provider error mapping as model
requests.

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

For Anthropic, `model`, `messages`, `stream`, `system`, `temperature`, `max_tokens`,
`stop_sequences`, and `timeout` are adapter-owned. Anthropic-native `tools` are likewise combined
with normalized function tools.

## Development

Most tests use fake transports and official SDK response models. The optional lifecycle smoke test
constructs real SDK and HTTP clients but sends no requests; no test makes a network call.

```bash
uv sync --locked --all-extras
uv run --no-sync ruff format --check .
uv run --no-sync ruff check .
uv run --no-sync ty check
uv run --no-sync mypy
uv run --no-sync pytest
```

`--all-extras` installs and exercises the optional `aiohttp` transport and OpenTelemetry observer.
Without an optional extra, its corresponding tests skip automatically. GitHub Actions runs the
same locked quality suite for pull requests, pushes to `main`, and release tags, then builds and
smoke-installs both distribution formats.

## Releases

Releases use stable `vMAJOR.MINOR.PATCH` Git tags and follow Semantic Versioning. The version in
`pyproject.toml` must match the tag without its `v` prefix; tag CI rejects malformed or mismatched
versions. Notable consumer-facing changes are collected under `Unreleased` in the
[changelog](CHANGELOG.md), then moved into a dated version section when a release is tagged.

CI validates release distributions but does not publish them or create a GitHub Release. On a tag,
the wheel, source distribution, and their SHA-256 checksums are retained as workflow artifacts for
30 days.

## Data handling and failure behavior

Requests are sent directly to the provider selected by the router. This package does not durably
persist prompts, responses, credentials, or traces. A `ChatSession` retains normalized visible
messages and generation records, including raw provider responses, in process until the session is
discarded; `clear_history()` does not clear its generation log. OpenAI Responses are stored by the
provider by default; pass `OpenAIProviderOptions(store=False)` when provider-side storage is not
desired. An injected `TraceObserver` controls its own data handling. `OTelTraceObserver` records
model identifiers, selected request parameters, finish reasons, token usage, latency, and terminal
exception details, but does not record prompt or response content. Its application-owned exporter
determines where those spans are transmitted and retained. Anthropic documents the standard
Messages API's current retention behavior in its [API data retention
guide](https://platform.claude.com/docs/en/manage-claude/api-and-data-retention); provider-native
tools can have different policies. `ModelResponse.raw`, error details, and tracing callbacks may
contain provider payloads, so applications should apply their own redaction and retention policies.
Observer failures are isolated from model calls, while provider failures cross the normalized
error boundary.

## Project structure

```text
src/model_runtime/                     domain types, runtime, router, session, retry, and tracing
src/model_runtime/observability/       optional OpenTelemetry TraceObserver integration
src/model_runtime/providers/base.py    reusable typed provider orchestration and protocols
src/model_runtime/providers/errors.py  shared provider error normalization
src/model_runtime/providers/anthropic/ Anthropic adapter, transport, codec, stream, and errors
src/model_runtime/providers/openai/    OpenAI adapter, transport, codec, stream, and errors
docs/end-to-end-request-flow.md        detailed OpenAI and Anthropic request flow walkthrough
observability/                         local Langfuse stack, OTLP smoke test, and usage guide
tests/                                 network-free runtime and provider contract tests
CHANGELOG.md                           notable changes organized by release
.github/workflows/ci.yml               locked quality and distribution validation
AGENTS.md                              Codex-native repository instructions
.cursor/rules/                         Cursor-native repository instructions
```

## Current limitations

- OpenAI uses the Responses API exclusively; Anthropic uses the Messages API exclusively.
- OpenAI normalization captures output text, refusal text, and function calls. Reasoning, hosted
  tool, citation, and other provider-specific output items remain available in `ModelResponse.raw`.
- Replaying `ModelResponse.message` manually does not preserve OpenAI reasoning or hosted-tool
  items. Use `previous_response_id` with provider-side state when those items must carry into a
  later call.
- The Responses API does not accept stop sequences or message names; the OpenAI codec rejects
  normalized requests containing either instead of silently dropping them.
- Anthropic normalization captures output text and client `tool_use` blocks. Thinking, citations,
  server-tool output, containers, and other provider-specific blocks remain in `ModelResponse.raw`.
- Anthropic has no equivalent of normalized image detail levels; requests using `low` or `high`
  are rejected rather than silently downgraded. Supported data URLs must be base64 JPEG, PNG, GIF,
  or WebP images.
- Normalized developer messages map to Anthropic system instructions because Messages has no
  separate developer role. Mid-conversation system messages remain subject to model support and
  Anthropic's placement rules.
- Usage totals are in memory and reset when the process exits.
- Phase 1 OTel spans contain no prompt or response content, session/user attributes, retry events,
  resource attributes, or explicit sampling configuration; those are deferred to later phases.
- OTel span correlation assumes a request's observer hooks run in the same async context. Fully
  consumed completion and streaming calls end spans; abandoning a stream before its terminal event
  can leave its span open until later lifecycle support is added.
- `ChatSession` memory and telemetry are in process only; persistence, concurrency control, and
  multi-participant conversation semantics are outside this provisional layer.
- The normalized image part accepts URLs and data URLs; file loading is left to the application.
- Provider-specific response fields are available through `ModelResponse.raw`, not normalized.
- An injected OpenAI-compatible client is assumed to have its own SDK retries disabled.
- An injected Anthropic-compatible client is likewise assumed to have SDK retries disabled.
