# Local Langfuse backend and Phase 2 OTel instrumentation

This directory implements Phases 0 through 2 of the observability plan: a local
Langfuse backend, one standalone OTLP ingestion smoke test, and optional rich
OpenTelemetry instrumentation for `ModelRuntime`. Text content capture is
privacy-off by default. This phase does not run evaluations, add Langfuse-native
session/user attributes, or introduce an OpenTelemetry Collector.

The original plan was written for Langfuse v3. The official self-hosting stack
now uses Langfuse v4, so this compose file follows the current v4 images and
sends the v4 ingestion header. The OTLP/HTTP endpoint remains
`/api/public/otel`.

## What runs

| Container | Responsibility |
| --- | --- |
| `langfuse-web` | Serves the UI, public API, authentication, and OTLP ingestion endpoint on `http://localhost:3000`. |
| `langfuse-worker` | Processes ingestion and other background jobs outside the web request path. |
| `postgres` | Stores transactional application state such as users, organizations, projects, and API-key metadata. |
| `clickhouse` | Stores and queries high-volume trace and observation data. |
| `redis` | Backs queues and short-lived coordination/cache state shared by web and worker. |
| `minio` | Provides local S3-compatible object storage; its startup command creates the `langfuse` bucket. |

Named Docker volumes retain each datastore across normal container restarts.
All published ports are bound to `127.0.0.1`; this is a local learning setup,
not production infrastructure.

## Start the backend

From the repository root, copy the template:

```bash
cp observability/.env.example observability/.env
```

Replace every `replace-with-...` value in `observability/.env`. Generate
independent values with these commands; `ENCRYPTION_KEY` must be the 64-character
hex output:

```bash
openssl rand -base64 32
openssl rand -hex 32
```

The populated `.env` is ignored by the repository and must not be committed.
Start the stack and inspect its status:

```bash
docker compose --env-file observability/.env \
  -f observability/compose.yaml up -d
docker compose --env-file observability/.env \
  -f observability/compose.yaml ps
```

Open `http://localhost:3000`, create the first user, organization, and project,
then create project API keys. Put the resulting `pk-lf-*` and `sk-lf-*` values
in `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` in `observability/.env`.
These keys authenticate ingestion into that one project; the secret key should
be handled like any other credential.

## Verify OTLP ingestion

The smoke test has inline `uv` dependencies and remains isolated from the
project environment. It constructs one plain OTel span and exports it directly
to Langfuse using OTLP/HTTP with Basic auth:

```bash
uv run observability/smoke_test.py
```

The OpenTelemetry imports are supplied by that isolated script environment,
not the project's `.venv` unless the `otel` extra is installed there. The
script therefore includes analyzer directives that suppress only missing-import
diagnostics; checks run inside the script environment still analyze the
installed packages normally.

The command prints the 32-character trace ID after export. Open the Tracing
view in Langfuse and locate the trace named `manual OTLP smoke test`. The span
has two deliberately non-GenAI attributes (`smoke_test.phase` and
`smoke_test.purpose`); runtime spans use GenAI semantic conventions as described
below.

## Trace model-runtime calls

Install the optional dependencies from the repository root:

```bash
uv sync --extra otel
```

The executable [`example.py`](../example.py) loads provider settings from the
root `.env` when present and Langfuse settings from `observability/.env`. It
requires `OPENAI_API_KEY`, `OPENAI_MODEL`, `ANTHROPIC_API_KEY`,
`ANTHROPIC_MODEL`, `LANGFUSE_PUBLIC_KEY`, and `LANGFUSE_SECRET_KEY`;
`LANGFUSE_BASE_URL` defaults to `http://localhost:3000`. The two provider calls
run as concurrent asyncio tasks under one parent application span. OTel context
propagates into both tasks, producing this trace hierarchy:

```text
example concurrent chat
├── chat <OpenAI model ID>
└── chat <Anthropic model ID>
```

Run it after the local stack is healthy:

```bash
uv run --extra otel python example.py
```

The application creates a `TracerProvider` with resource `service.name`
`model-runtime-example` and `service.version` equal to the installed package
version. It adds a `BatchSpanProcessor` and OTLP/HTTP exporter, then injects the
provider's tracer into `OTelTraceObserver`. The library does not install or
mutate a global tracer provider. Without an explicit `service.name`, the SDK
reports `unknown_service`. The example sends traces to
`/api/public/otel/v1/traces` using Basic auth and the
`x-langfuse-ingestion-version: 4` header, and shuts its provider down so the
batch processor flushes before process exit.

Each completed runtime call produces one client span named `chat <model-id>`.
The span covers the runtime's full logical call, including every retry, and
records these values:

| Attribute | Meaning |
| --- | --- |
| `gen_ai.operation.name` | The normalized `chat` operation. |
| `gen_ai.provider.name` | The provider explicitly configured on the observer. |
| `gen_ai.request.model` | The provider model selected by the router. |
| `gen_ai.request.temperature` | Requested temperature, when set. |
| `gen_ai.request.max_tokens` | Requested maximum output tokens, when set. |
| `gen_ai.response.model` | Provider-reported response model, when available. |
| `gen_ai.response.finish_reasons` | Normalized terminal finish reason. |
| `gen_ai.usage.input_tokens` | Total normalized input tokens, including cached input. |
| `gen_ai.usage.output_tokens` | Total normalized output tokens. |
| `gen_ai.usage.cache_read.input_tokens` | Cached input tokens, when nonzero. |
| `model_runtime.latency_ms` | End-to-end logical runtime latency, including retries. |
| `gen_ai.input.messages` | Opt-in input text encoded with the GenAI message schema. |
| `gen_ai.output.messages` | Opt-in response text and finish reason encoded with the GenAI message schema. |

Successful spans have OTel `OK` status. Terminal failures have `ERROR` status,
an `error.type` attribute, a standard exception event, and normalized
`model_runtime.error.retryable`, `model_runtime.error.provider`, and
`model_runtime.error.status_code` attributes when available. The values of
`error.type` are the public low-cardinality taxonomy from `errors.py`, such as
`RateLimitError`, `RequestTimeout`, and `ProviderUnavailableError`.

When an attempt will be retried, the same open span receives one
`model_runtime.retry` event before backoff. Its attributes include the failed
one-based `model_runtime.retry.attempt`, `model_runtime.retry.next_attempt`,
`model_runtime.retry.delay_ms`, and the normalized error attributes. The base
runtime exposes this through the optional `RetryTraceObserver` capability;
existing `TraceObserver` implementations without `on_retry` remain valid.

Observer failures remain isolated by `ModelRuntime` and cannot turn a
successful model call into a failure. Open spans are correlated through a
per-observer `ContextVar`, so interleaved calls in separate async tasks do not
finish one another's spans.

## Content capture and privacy

Prompt and response text is not recorded by default. To opt in for a controlled
environment, set this in `observability/.env` before constructing the observer:

```dotenv
OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=true
```

Only the case-insensitive value `true` enables the flag. Code can make the
choice explicit, and an explicit constructor value takes precedence over the
environment:

```python
observer = OTelTraceObserver(
    tracer,
    provider_name="openai",
    capture_message_content=True,
)
```

The observer serializes normalized `TextPart` values and roles to the current
GenAI input/output message JSON shape because Python span attributes do not
support nested objects. Textual tool-result messages are therefore included.
It deliberately omits image URLs/data, tool-call structures and arguments,
tool definitions, and provider-native raw payloads. There is no content
redaction or truncation in this phase. Treat enabled content as potentially
sensitive and high-volume, and confirm exporter access and backend retention
before enabling it. Independently, terminal exception events include the
normalized error message and stack trace; a provider-supplied error can contain
request details, so applications should review that failure path too.

## Sampling

The Python SDK selects its sampler when `TracerProvider` is constructed. The
safe learning default in `.env.example` records complete traces while making
the parent-based relationship explicit:

```dotenv
OTEL_TRACES_SAMPLER=parentbased_traceidratio
OTEL_TRACES_SAMPLER_ARG=1.0
```

To experiment with 25% head sampling, change the argument to `0.25` and restart
the example. `ParentBased` applies the root decision to the runtime children,
so Langfuse receives either the parent and its generation spans or none of that
trace. Keep `1.0` while learning or debugging; lower ratios trade visibility
for reduced ingestion and storage volume.

The observer is imported from `model_runtime.observability`. Importing the base
`model_runtime` package still works without the `otel` extra; importing its OTel
integration without the extra raises an installation hint.

If export returns `401`, confirm the public and secret keys belong to the same
project. If it cannot connect, check `docker compose ... ps` and inspect logs:

```bash
docker compose --env-file observability/.env \
  -f observability/compose.yaml logs --tail=100 langfuse-web langfuse-worker
```

Stop containers while retaining data:

```bash
docker compose --env-file observability/.env \
  -f observability/compose.yaml down
```

Adding `--volumes` to `down` permanently deletes the local Langfuse databases
and object storage.
