"""Logical-name routing for chat model adapters."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

from .model import ChatModel
from .types import ModelCapabilities, ModelRequest


@dataclass(frozen=True, slots=True)
class ModelRoute:
    """Bind an adapter to its provider-specific model identifier."""

    adapter: ChatModel
    model_id: str


type RouteSelector = Callable[[str, ModelRequest], str | ModelRoute | None]


class ModelRouter:
    """Resolve application-facing names to an adapter and provider model ID.

    A selector may map a dynamic name (for example, ``"default"``) to a
    registered name based on the request. Selection policy remains application
    code; this class only stores and resolves routes.
    """

    def __init__(
        self,
        routes: Mapping[str, ModelRoute | tuple[ChatModel, str]] | None = None,
        *,
        selector: RouteSelector | None = None,
    ) -> None:
        self._routes: dict[str, ModelRoute] = {}
        self._selector = selector
        for name, route in (routes or {}).items():
            if isinstance(route, ModelRoute):
                self.register(name, route.adapter, route.model_id)
            else:
                adapter, model_id = route
                self.register(name, adapter, model_id)

    @property
    def routes(self) -> Mapping[str, ModelRoute]:
        """An immutable snapshot of the registered routes."""
        return MappingProxyType(dict(self._routes))

    def register(
        self,
        name: str,
        adapter: ChatModel,
        model_id: str,
        *,
        replace: bool = False,
    ) -> ModelRouter:
        """Register a route and return this router for fluent configuration."""
        if not name:
            raise ValueError("route name cannot be empty")
        if not model_id:
            raise ValueError("provider model ID cannot be empty")
        if name in self._routes and not replace:
            raise ValueError(f"a route named {name!r} is already registered")
        self._routes[name] = ModelRoute(adapter=adapter, model_id=model_id)
        return self

    def unregister(self, name: str) -> ModelRoute:
        """Remove and return the route registered under ``name``."""
        try:
            return self._routes.pop(name)
        except KeyError as exc:
            raise KeyError(f"no model route is registered for {name!r}") from exc

    def resolve(self, name: str, request: ModelRequest | None = None) -> ModelRoute:
        """Resolve a logical name, optionally passing it through the selector."""
        selected: str | ModelRoute | None = name
        if self._selector is not None:
            if request is None:
                raise ValueError("a request is required when the router has a selector")
            selected = self._selector(name, request)
        if selected is None:
            selected = name
        if isinstance(selected, ModelRoute):
            return selected
        try:
            return self._routes[selected]
        except KeyError as exc:
            raise KeyError(f"no model route is registered for {selected!r}") from exc

    def capabilities(
        self, name: str, request: ModelRequest | None = None
    ) -> ModelCapabilities:
        """Return capabilities for the route selected by ``name`` and ``request``."""
        return self.resolve(name, request).adapter.capabilities
