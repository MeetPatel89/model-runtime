# Local Langfuse backend and Phase 1 OTel instrumentation

This directory implements Phase 0 and Phase 1 of the observability plan: a
local Langfuse backend, one standalone OTLP ingestion smoke test, and optional
OpenTelemetry instrumentation for `ModelRuntime`. It does not capture prompts
or responses, run evaluations, add Langfuse-native session/user attributes, or
introduce an OpenTelemetry Collector.

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
  -f observability/docker-compose.yml up -d
docker compose --env-file observability/.env \
  -f observability/docker-compose.yml ps
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
requires `OPENAI_API_KEY`, `OPENAI_MODEL`, `LANGFUSE_PUBLIC_KEY`, and
`LANGFUSE_SECRET_KEY`; `LANGFUSE_BASE_URL` defaults to
`http://localhost:3000`. Run it after the local stack is healthy:

```bash
uv run --extra otel python example.py
```

The application creates a `TracerProvider`, `BatchSpanProcessor`, and OTLP/HTTP
exporter, then injects the provider's tracer into `OTelTraceObserver`. The
library does not install or mutate a global tracer provider. The example sends
traces to `/api/public/otel/v1/traces` using Basic auth and the
`x-langfuse-ingestion-version: 4` header, and shuts its provider down so the
batch processor flushes before process exit.

Each completed runtime call produces one client span named `chat <model-id>`.
The span covers the runtime's full logical call, including any retries, and
records these values without sending request or response content:

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

Successful spans have OTel `OK` status. Terminal failures have `ERROR` status,
an `error.type` attribute, and a standard exception event. Observer failures
remain isolated by `ModelRuntime` and cannot turn a successful model call into
a failure. Open spans are correlated through a per-observer `ContextVar`, so
interleaved calls in separate async tasks do not finish one another's spans.

The observer is imported from `model_runtime.observability`. Importing the base
`model_runtime` package still works without the `otel` extra; importing its OTel
integration without the extra raises an installation hint.

If export returns `401`, confirm the public and secret keys belong to the same
project. If it cannot connect, check `docker compose ... ps` and inspect logs:

```bash
docker compose --env-file observability/.env \
  -f observability/docker-compose.yml logs --tail=100 langfuse-web langfuse-worker
```

Stop containers while retaining data:

```bash
docker compose --env-file observability/.env \
  -f observability/docker-compose.yml down
```

Adding `--volumes` to `down` permanently deletes the local Langfuse databases
and object storage.
