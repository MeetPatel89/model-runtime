# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Add the Phase 0 local Langfuse v4 Docker Compose backend, environment template,
  component guide, and standalone direct-OTLP ingestion smoke test.
- Add optional OpenTelemetry instrumentation with GenAI semantic-convention
  attributes, direct Langfuse export in the example, and network-free tests.
- Add Phase 2 OpenTelemetry hierarchy propagation, opt-in text content capture,
  normalized retry events and error taxonomy, service name/version resource
  metadata, and parent-based trace sampling configuration.

### Fixed

- Remove a leftover debug `print` of Anthropic request kwargs from completion
  transport.

## [0.2.0] - 2026-08-12

### Added

- Add the async-first `ChatSession` conversation layer with system context,
  immutable history snapshots, atomic turn recording, transient generation
  context, and guarded synchronous wrappers.
- Add frozen `GenerationRecord` telemetry values that retain the normalized and
  raw model response, resolved model, logical route/provider name, response ID,
  and latency.
- Add the optional `ModelCatalog` protocol and asynchronous model discovery to
  the OpenAI and Anthropic adapters through their existing SDK clients and error
  mappers.

## [0.1.0] - 2026-08-10

### Added

- Add an async, provider-independent model runtime with normalized completion and
  streaming responses, routing, retries, timeouts, tracing, and token accounting.
- Add OpenAI Responses and Anthropic Messages integrations with normalized messages,
  tool calls, images, provider errors, and streaming events.
- Add typed provider options for provider-native tools, structured output, and other
  JSON-compatible request features.
- Add an optional `aiohttp` transport for both provider SDKs.
- Add network-free runtime and provider contract tests with strict ty, mypy, and Ruff
  checks.
