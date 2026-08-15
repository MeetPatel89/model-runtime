# Phase 0: local Langfuse backend

This directory implements only Phase 0 of the observability plan: a local
Langfuse backend and one standalone OTLP ingestion smoke test. It does not
instrument `ModelRuntime`, add package dependencies, capture prompts or
responses, run evaluations, or introduce an OpenTelemetry Collector.

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

The smoke test has inline `uv` dependencies, so it does not change
`model-runtime` or add an `otel` package extra. It constructs one plain OTel
span and exports it directly to Langfuse using OTLP/HTTP with Basic auth:

```bash
uv run observability/smoke_test.py
```

The OpenTelemetry imports are supplied by that isolated script environment,
not the project's `.venv`. The script therefore includes analyzer directives
that suppress only missing-import diagnostics; checks run inside the script
environment still analyze the installed packages normally.

The command prints the 32-character trace ID after export. Open the Tracing
view in Langfuse and locate the trace named `manual OTLP smoke test`. The span
has two deliberately non-GenAI attributes (`smoke_test.phase` and
`smoke_test.purpose`); GenAI semantic conventions begin in Phase 1.

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
