# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
