"""Strict JSON-compatible types used at provider extension boundaries."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import cast

type JsonPrimitive = str | int | float | bool | None
type JsonValue = JsonPrimitive | Sequence["JsonValue"] | Mapping[str, "JsonValue"]
type JsonObject = Mapping[str, JsonValue]


def immutable_json_object(value: JsonObject | None = None) -> JsonObject:
    """Validate and shallow-copy a JSON object into an immutable mapping.

    Nested values deliberately retain their original representation. Provider
    option payloads are pass-through values, so an adapter must not silently
    rewrite a provider-specific list into a tuple or otherwise change it.
    """
    source = value or {}
    _validate_json(source, path="$", active_containers=set())
    return MappingProxyType(dict(source))


def parse_json_object(value: str) -> JsonObject | None:
    """Parse ``value`` only when it contains a valid JSON object."""
    try:
        decoded = cast(object, json.loads(value))
    except json.JSONDecodeError:
        return None
    if not isinstance(decoded, Mapping):
        return None
    candidate = cast(Mapping[object, object], decoded)
    if not all(isinstance(key, str) for key in candidate):
        return None
    typed_candidate = cast(JsonObject, candidate)
    try:
        return immutable_json_object(typed_candidate)
    except TypeError, ValueError:
        return None


def _validate_json(
    value: JsonValue,
    *,
    path: str,
    active_containers: set[int],
) -> None:
    """Reject non-JSON values and cyclic containers with a useful path."""
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a non-finite number")
        return
    if value is None or isinstance(value, (str, int, bool)):
        return
    if isinstance(value, bytes | bytearray):
        raise TypeError(f"{path} must contain JSON-compatible values, not bytes")

    if isinstance(value, Mapping):
        identity = id(value)
        if identity in active_containers:
            raise ValueError(f"{path} contains a cyclic mapping")
        active_containers.add(identity)
        try:
            for key, item in value.items():
                if not isinstance(key, str):
                    raise TypeError(f"{path} contains a non-string object key")
                _validate_json(
                    item,
                    path=f"{path}.{key}",
                    active_containers=active_containers,
                )
        finally:
            active_containers.remove(identity)
        return

    if isinstance(value, Sequence):
        identity = id(value)
        if identity in active_containers:
            raise ValueError(f"{path} contains a cyclic sequence")
        active_containers.add(identity)
        try:
            for index, item in enumerate(value):
                _validate_json(
                    item,
                    path=f"{path}[{index}]",
                    active_containers=active_containers,
                )
        finally:
            active_containers.remove(identity)
        return

    raise TypeError(
        f"{path} contains {type(value).__name__}; expected a JSON-compatible value"
    )
