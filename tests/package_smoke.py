"""Smoke-test an installed model-runtime distribution."""

from __future__ import annotations

from importlib.metadata import version
from sys import argv

from model_runtime import (
    AnthropicAdapter,
    ChatSession,
    GenerationRecord,
    ModelRuntime,
    OpenAIAdapter,
)


def main() -> None:
    """Verify package metadata and representative public imports."""
    if len(argv) != 2:
        raise SystemExit("usage: package_smoke.py EXPECTED_VERSION")

    expected_version = argv[1]
    installed_version = version("model-runtime")
    if installed_version != expected_version:
        raise RuntimeError(
            f"installed version {installed_version!r} does not match "
            f"expected version {expected_version!r}"
        )

    public_classes = (
        AnthropicAdapter,
        ChatSession,
        GenerationRecord,
        ModelRuntime,
        OpenAIAdapter,
    )
    if not all(isinstance(public_class, type) for public_class in public_classes):
        raise RuntimeError("representative public exports are not classes")

    print(f"model-runtime {installed_version} distribution smoke test passed")


if __name__ == "__main__":
    main()
