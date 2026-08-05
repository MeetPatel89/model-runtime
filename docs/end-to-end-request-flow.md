# End-to-End Request Flow

This document walks through one complete LLM call — from your application code to
the provider API and back — for both **OpenAI** (Responses API) and **Anthropic**
(Messages API). The shared layers are the same; only the provider stack
(adapter → codec → transport → SDK) differs.

## Mental model

```text
application
    │
    ▼
ModelRuntime          ← routing, retries, timeouts, tracing, usage
    │
    ▼
ModelRouter           ← logical name → (ChatModel, provider model ID)
    │
    ▼
ProviderAdapter       ← shared encode → call → decode lifecycle
    │                     (OpenAIAdapter / AnthropicAdapter)
    ├── ProviderCodec       translate ModelRequest ↔ native request/response
    ├── ProviderTransport   call the official async SDK
    └── ProviderErrorMapper SDK exceptions → ModelRuntimeError
```

Your application never talks to `openai` or `anthropic` directly. It builds a
normalized `ModelRequest`, picks a logical model name (`"chat"`, `"claude"`, …),
and calls `runtime.complete` or `runtime.stream`. Everything below that boundary
is owned by this package.

---

## 1. Setup: wiring the stack

Before any request runs, you assemble three objects once:

```python
from model_runtime import (
    AnthropicAdapter,
    ModelRouter,
    ModelRuntime,
    OpenAIAdapter,
)

openai_adapter = OpenAIAdapter()  # AsyncOpenAI, max_retries=0
anthropic_adapter = AnthropicAdapter()  # AsyncAnthropic, max_retries=0

router = (
    ModelRouter()
    .register("chat", openai_adapter, "gpt-5")
    .register("claude", anthropic_adapter, "claude-sonnet-4-0")
)
runtime = ModelRuntime(router)
```

What each piece owns:

| Object | Responsibility |
| --- | --- |
| `OpenAIAdapter` / `AnthropicAdapter` | Build SDK client, wire transport + codec + error mapper, advertise capabilities |
| `ModelRouter` | Map logical names to `(adapter, provider_model_id)` |
| `ModelRuntime` | Resolve routes, enforce timeout/retry, notify observers, accumulate usage |

SDK-level retries are disabled (`max_retries=0`) so `ModelRuntime` is the only
retry owner.

---

## 2. Building a normalized request

Both providers accept the same domain types from `model_runtime.types`:

```python
from model_runtime import Message, ModelRequest, OpenAIProviderOptions

request = ModelRequest(
    messages=(
        Message.system("Answer clearly and briefly."),
        Message.user("Why is the sky blue?"),
    ),
    temperature=0.2,
    max_output_tokens=256,
    timeout=30,
    provider_options=OpenAIProviderOptions(store=False, reasoning_effort="low"),
)
```

Key fields on `ModelRequest`:

| Field | Meaning |
| --- | --- |
| `messages` | Conversation turns (`system` / `developer` / `user` / `assistant` / `tool`) |
| `tools` | Normalized function tools (`ToolDefinition`) |
| `temperature`, `max_output_tokens`, `stop`, `timeout` | Cross-provider controls |
| `provider_options` | Provider-native JSON (or typed `OpenAIProviderOptions` / `AnthropicProviderOptions`) |

Messages are immutable value objects. Content is a tuple of `TextPart` /
`ImagePart`. Assistants may carry `tool_calls`; tool results use
`Message.tool(..., tool_call_id=...)`.

Provider-specific knobs (reasoning, thinking, hosted tools, caching, …) go in
`provider_options`. The selected codec interprets them; the other provider never
sees them.

---

## 3. Shared path: `ModelRuntime.complete`

Call site:

```python
response = await runtime.complete("chat", request)
print(response.text)
print(response.usage)
print(response.finish_reason)
```

### Step-by-step inside `ModelRuntime`

1. **Resolve the route** — `router.resolve("chat", request)` returns a
   `ModelRoute(adapter=OpenAIAdapter, model_id="gpt-5")`. Missing names become
   `InvalidRequestError`. An optional router `selector` can rewrite the name
   based on the request (for example, tools → a more careful model).

2. **Trace request** — `observer.on_request(model_id, request)`. Observer
   failures are swallowed so instrumentation cannot break a successful call.

3. **Invoke the adapter under timeout** — if `request.timeout` is set,
   `asyncio.wait_for` wraps `adapter.complete(model_id, request)`. Expiry
   becomes `RequestTimeout`.

4. **Retry loop** — on failure, exceptions already in the public taxonomy pass
   through; anything else becomes non-retryable `ProviderUnavailableError`.
   `RetryPolicy.should_retry` checks `error.retryable` and attempt count
   (default max 3). Delay is exponential backoff with jitter, or the provider's
   `Retry-After` when present.

5. **Record success** — usage is added to `runtime.total_usage` and
   `runtime.usage_by_model[model_id]`. `observer.on_response` fires. The
   `ModelResponse` is returned to the caller.

Streaming (`runtime.stream`) follows the same resolve → observe → call shape,
with two important differences:

- Timeout applies **per event** (`anext`), not to the whole stream.
- Retries are allowed **only before the first** `TextDelta` / `ToolCallDelta`.
  After any delta has been yielded, a failure is terminal (retrying would
  duplicate output).

---

## 4. Shared path: `ProviderAdapter`

Both adapters subclass the same orchestration class. For a completion:

```text
encode_request(model_id, request, stream=False)
        │
        ▼
transport.complete(encoded_native_request)
        │
        ▼
decode_response(native_response) → ModelResponse
```

For a stream:

```text
encode_request(..., stream=True)
        │
        ▼
stream_decoder = codec.stream_decoder(...)
        │
        ▼
async for native_chunk in transport.stream(encoded):
    yield deltas from decoder.feed(chunk)
yield decoder.finish()   → StreamEnd(response=...)
```

Any non-`ModelRuntimeError` raised by codec or transport is passed through
`ProviderErrorMapper.translate` before leaving the adapter. The runtime then
applies retry policy to that normalized error.

---

## 5. OpenAI end-to-end (Responses API)

### 5.1 Adapter construction

`OpenAIAdapter` builds:

- `AsyncOpenAI(..., max_retries=0)` (or an injected test client)
- `OpenAITransport` over `client.responses`
- `OpenAICodec`
- `StandardProviderErrorMapper(OpenAIErrorMetadataExtractor)`
- Default capabilities: tools, vision, structured output, streaming, plus
  features such as `reasoning` and `built_in_tools`

### 5.2 Encode (`OpenAICodec.encode_request`)

Normalized → Responses shapes:

| Normalized input | Responses wire form |
| --- | --- |
| `system` / `developer` / text `user` | `EasyInputMessageParam` with matching role |
| Multimodal `user` | `input_text` + `input_image` (detail preserved) |
| `assistant` + `tool_calls` | assistant text item + `function_call` items (`call_id`) |
| `tool` message | `function_call_output` with the same `call_id` |
| `ToolDefinition` | `FunctionToolParam` (`type: "function"`) |
| Hosted tools in options | Prepended via `provider_options.tools` |

Hard rejects (no silent drop):

- `request.stop` — Responses has no stop sequences
- Message `name` — not representable
- Overrides of owned keys in `provider_options`: `model`, `input`, `stream`,
  `temperature`, `max_output_tokens`, `timeout`

Result: an `OpenAIRequest` whose `as_kwargs()` produces the arguments for
`responses.create`. Function tools and provider-native tools are concatenated.

### 5.3 Transport (`OpenAITransport`)

| Mode | Call | Result |
| --- | --- | --- |
| Complete | `client.responses.create(**kwargs)` with `stream=False` | Must be a `Response` |
| Stream | Same endpoint with `stream=True` | Iterate typed `ResponseStreamEvent`s; always `close()` |

### 5.4 Decode complete response

`OpenAICodec` walks `response.output`:

- `ResponseOutputMessage` text / refusal → `TextPart`
- `ResponseFunctionToolCall` → `ToolCall` (arguments parsed as JSON object when
  possible)
- Usage from `ResponseUsage` (cached tokens from `input_tokens_details`)
- Finish reason from status / incomplete details / presence of function calls

Everything else (reasoning items, hosted-tool output, citations, …) stays on
`ModelResponse.raw` as the SDK `Response` object.

### 5.5 Decode stream

`OpenAIStreamDecoder.feed`:

| SDK event | Public event |
| --- | --- |
| text / refusal delta | `TextDelta` |
| function-call item added | `ToolCallDelta` (id, name, initial args) |
| function-call arguments delta | `ToolCallDelta` (args fragment) |
| completed / incomplete / failed | Stash terminal `Response` (no yield yet) |
| error event | Raise `ProviderUnavailableError` |

`finish()` decodes the stashed terminal response into `StreamEnd`.

### 5.6 Full OpenAI sequence diagram

```text
app
 └─ runtime.complete("chat", request)
     ├─ router.resolve → OpenAIAdapter + "gpt-5"
     ├─ observer.on_request
     └─ OpenAIAdapter.complete
         ├─ OpenAICodec.encode_request
         │     Message* → Responses input items
         │     ToolDefinition* → function tools
         │     OpenAIProviderOptions → extra kwargs
         ├─ OpenAITransport.complete
         │     AsyncOpenAI.responses.create(...)
         │           ──HTTP──► OpenAI Responses API
         │           ◄─Response──
         ├─ OpenAICodec.decode_response → ModelResponse
         └─ (errors) OpenAIErrorMetadataExtractor → taxonomy
     ├─ record usage / observer.on_response
     └─ return ModelResponse to app
```

### 5.7 Minimal OpenAI example

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
)


async def main() -> None:
    adapter = OpenAIAdapter()
    router = ModelRouter().register("chat", adapter, os.environ["OPENAI_MODEL"])
    runtime = ModelRuntime(router)

    request = ModelRequest(
        messages=(
            Message.system("Be brief."),
            Message.user("Why is the sky blue?"),
        ),
        timeout=30,
        provider_options=OpenAIProviderOptions(store=False),
    )
    response = await runtime.complete("chat", request)
    print(response.text)


asyncio.run(main())
```

---

## 6. Anthropic end-to-end (Messages API)

### 6.1 Adapter construction

`AnthropicAdapter` builds:

- `AsyncAnthropic(..., max_retries=0)` (or an injected test client)
- `AnthropicTransport` over `client.messages`
- `AnthropicCodec(default_max_output_tokens=1024)` — Messages always needs
  `max_tokens`
- `StandardProviderErrorMapper(AnthropicErrorMetadataExtractor)`
- Default capabilities: tools, vision, structured output, streaming, plus
  features such as `thinking`, `prompt_caching`, `effort`

### 6.2 Encode (`AnthropicCodec.encode_request`)

Normalized → Messages shapes:

| Normalized input | Messages wire form |
| --- | --- |
| Leading `system` / `developer` | Top-level `system` text blocks |
| Mid-conversation `system` / `developer` | Anthropic `role: "system"` messages |
| `user` / `assistant` content | Text / image content blocks |
| `assistant` + `tool_calls` | `tool_use` blocks (`id`, `name`, `input` object) |
| `tool` message | User message with `tool_result` (`tool_use_id`) |
| `ToolDefinition` | `ToolParam` (`name`, `input_schema`, optional `strict`) |
| Images | URL source, or base64 data URL (jpeg/png/gif/webp); **detail must be `auto`** |
| `request.stop` | `stop_sequences` |
| `max_output_tokens` | `max_tokens` (falls back to adapter default 1024) |

Hard rejects:

- Empty conversation (only system messages) — Anthropic needs ≥1 non-system message
- Message `name`
- Image `detail` of `low` / `high`
- Overrides of owned keys: `model`, `messages`, `system`, `stream`,
  `temperature`, `max_tokens`, `stop_sequences`, `timeout`

Result: an `AnthropicRequest`. Note that `stream` is stored for transport
validation but **not** passed as a create kwarg — the transport chooses
`messages.create` vs `messages.stream`.

### 6.3 Transport (`AnthropicTransport`)

| Mode | Call | Result |
| --- | --- | --- |
| Complete | `client.messages.create(**kwargs)` | Must be a `Message` |
| Stream | `async with client.messages.stream(**kwargs)` | Yield typed stream events; SDK owns cleanup |

### 6.4 Decode complete response

`AnthropicCodec` walks `response.content`:

- `TextBlock` → `TextPart`
- `ToolUseBlock` → `ToolCall` with JSON-object `input`
- Usage: `input_tokens + cache_creation + cache_read`; `cached_tokens` =
  cache-read only
- Finish reason from `stop_reason` (`end_turn`, `max_tokens`, `tool_use`,
  `refusal`, …)

Thinking, citations, server tools, and other blocks remain on
`ModelResponse.raw`.

### 6.5 Decode stream

`AnthropicStreamDecoder.feed`:

| SDK event | Public event |
| --- | --- |
| content_block_start + `ToolUseBlock` | `ToolCallDelta` (id, name) |
| text delta | `TextDelta` |
| `InputJSONDelta` | `ToolCallDelta` (partial JSON) |
| `ParsedMessageStopEvent` | Stash final `Message` |

`finish()` decodes the stashed message into `StreamEnd`.

### 6.6 Full Anthropic sequence diagram

```text
app
 └─ runtime.complete("claude", request)
     ├─ router.resolve → AnthropicAdapter + "claude-sonnet-4-0"
     ├─ observer.on_request
     └─ AnthropicAdapter.complete
         ├─ AnthropicCodec.encode_request
         │     leading system → top-level system
         │     Message* → messages[]
         │     ToolDefinition* → tools
         │     AnthropicProviderOptions → extra kwargs
         ├─ AnthropicTransport.complete
         │     AsyncAnthropic.messages.create(...)
         │           ──HTTP──► Anthropic Messages API
         │           ◄─Message──
         ├─ AnthropicCodec.decode_response → ModelResponse
         └─ (errors) AnthropicErrorMetadataExtractor → taxonomy
     ├─ record usage / observer.on_response
     └─ return ModelResponse to app
```

### 6.7 Minimal Anthropic example

```python
import asyncio
import os
from model_runtime import (
    AnthropicAdapter,
    ModelRequest,
    ModelRouter,
    ModelRuntime,
)


async def main() -> None:
    adapter = AnthropicAdapter(default_max_output_tokens=1024)
    router = ModelRouter().register("claude", adapter, os.environ["ANTHROPIC_MODEL"])
    runtime = ModelRuntime(router)

    request = ModelRequest.from_text(
        "Why is the sky blue?",
        timeout=30,
    )
    response = await runtime.complete("claude", request)
    print(response.text)


asyncio.run(main())
```

---

## 7. Streaming path (both providers)

```python
from model_runtime import StreamEnd, TextDelta, ToolCallDelta

async for event in runtime.stream("chat", request):
    if isinstance(event, TextDelta):
        print(event.delta, end="", flush=True)
    elif isinstance(event, ToolCallDelta):
        ...  # assemble tool call by event.index
    elif isinstance(event, StreamEnd):
        print(event.response.text)
        print(event.usage)
```

Event contract:

1. Zero or more `TextDelta` / `ToolCallDelta`
2. Exactly one terminal `StreamEnd` carrying the full `ModelResponse` and usage
3. Usage is recorded only when `StreamEnd` arrives

Internally the adapter still does encode → transport stream → decoder; only the
native chunk types differ (OpenAI `ResponseStreamEvent` vs Anthropic parsed
message events).

---

## 8. Tool-calling round trip

The normalized loop is provider-agnostic:

```text
1. Request with ToolDefinition(s)
2. ModelResponse.finish_reason == TOOL_CALLS
3. App executes tools using response.message.tool_calls
4. New request appending:
     - the assistant message (with tool_calls)
     - Message.tool(result, tool_call_id=call.id) for each call
5. complete/stream again
```

Codec mapping for that replay:

| Concept | OpenAI | Anthropic |
| --- | --- | --- |
| Model invokes tool | `function_call` item | `tool_use` block |
| App returns result | `function_call_output` | user `tool_result` |
| Correlation id | `call_id` | `tool_use_id` |
| Arguments on wire | JSON **string** | JSON **object** |

Hosted / server tools (web search, etc.) are **not** normalized `ToolDefinition`s.
Pass them through `provider_options.tools`; they are prepended to function tools
on the wire. Their outputs typically remain in `ModelResponse.raw`.

---

## 9. Errors and retries

Public taxonomy (all subclass `ModelRuntimeError`):

| Error | Typical cause |
| --- | --- |
| `AuthError` | Bad / missing credentials |
| `RateLimitError` | 429 / rate limits (`retry_after` when available) |
| `RequestTimeout` | Runtime `asyncio.wait_for` expiry |
| `InvalidRequestError` | Bad route, illegal options, codec rejection |
| `ContentFilterError` | Safety / refusal paths |
| `ProviderUnavailableError` | Outages, unexpected payloads, unknown exceptions |

Flow:

```text
SDK exception
  → ProviderErrorMapper.translate  (status / type / headers)
  → ModelRuntimeError (retryable flag set)
  → RetryPolicy.should_retry / delay_for
  → retry or observer.on_error + raise
```

Default policy: up to 3 attempts, exponential backoff (0.5s → capped at 8s) with
±20% jitter. A provider `retry_after` wins and is not jittered.

---

## 10. Side-by-side differences that affect call code

| Concern | OpenAI | Anthropic |
| --- | --- | --- |
| Native API | Responses (`responses.create`) | Messages (`messages.create` / `.stream`) |
| System prompt | Role items in `input` | Top-level `system` (+ mid-conversation system) |
| Stop sequences | Rejected | `stop_sequences` |
| `max_tokens` | Optional `max_output_tokens` | Always set (default 1024) |
| Developer role | Native `developer` | Folded into system text |
| Image detail | `low` / `high` / `auto` | Must be `auto` |
| Stream entry | `create(stream=True)` | `messages.stream()` context manager |
| Reasoning / thinking | `reasoning` / `reasoning_effort` | `thinking` / `effort` via options |
| Continuity of special items | Prefer `previous_response_id` | Prompt caching / containers via options |

What stays identical for callers: `Message`, `ModelRequest`, `ModelResponse`,
`runtime.complete` / `runtime.stream`, stream events, and the public error types.

---

## 11. Where to read the code

| Layer | Path |
| --- | --- |
| Runtime orchestration | `src/model_runtime/runtime.py` |
| Routing | `src/model_runtime/router.py` |
| Domain types | `src/model_runtime/types.py` |
| Shared adapter lifecycle | `src/model_runtime/providers/base.py` |
| Retry policy | `src/model_runtime/retry.py` |
| Tracing hooks | `src/model_runtime/tracing.py` |
| OpenAI stack | `src/model_runtime/providers/openai/` (`adapter`, `codec`, `transport`, `_types`, `errors`) |
| Anthropic stack | `src/model_runtime/providers/anthropic/` (same layout) |
| Runnable OpenAI sample | `example.py` |
| Contract tests | `tests/test_runtime.py`, `tests/test_provider_adapter.py`, `tests/test_openai_adapter.py`, `tests/test_anthropic_adapter.py` |

---

## 12. Quick checklist when debugging a call

1. Confirm the logical name is registered and resolves to the adapter you expect
   (`router.resolve(name, request)`).
2. Inspect the normalized `ModelRequest` (messages, tools, timeout, options).
3. Remember codecs reject some combinations early (`stop` on OpenAI, missing
   non-system message on Anthropic, owned option overrides).
4. Check whether failure happened before the first stream delta (retryable) or
   after (terminal).
5. Use `ModelResponse.raw` / error `details` / `__cause__` for provider-native
   fields the normalized layer does not surface.
6. Verify credentials via env (`OPENAI_API_KEY` / `ANTHROPIC_API_KEY`) or
   adapter constructor kwargs — this package has no global config of its own.
