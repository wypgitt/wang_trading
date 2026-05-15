"""Small in-process LRU+TTL cache used by services and route helpers.

Standalone — no Redis, no diskcache. ``LRUTTLCache`` is thread-safe via
a single :class:`threading.Lock`; the access pattern is a sub-microsecond
dict/OrderedDict update so contention is negligible at expected load.

Use the :func:`cached` decorator for stateless service methods::

    @cached(ttl=30.0, cache_name="trade_ideas_top")
    def top_actionable(self) -> list[Idea]: ...

The cache key is built from ``(qualname, args, kwargs)`` so methods on
different instances of the same class share a key. Callers that need
per-instance scoping should include ``id(self)`` in the args or maintain
their own :class:`LRUTTLCache` instance.
"""

from __future__ import annotations

import functools
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from typing import Any, TypeVar


T = TypeVar("T")

_MISSING = object()


class LRUTTLCache:
    """Bounded LRU cache where each entry has its own TTL.

    - ``maxsize``: hard cap on entries (oldest evicted on insert).
    - Default ``ttl`` applies to ``set()`` calls that omit ``ttl_seconds``.
    - Expired entries are removed lazily on access; an optional
      :meth:`purge_expired` helper exists for callers that want to free
      memory eagerly.
    """

    def __init__(self, *, maxsize: int = 128, ttl: float = 30.0) -> None:
        if maxsize <= 0:
            raise ValueError("maxsize must be > 0")
        if ttl < 0:
            raise ValueError("ttl must be >= 0")
        self._maxsize = maxsize
        self._default_ttl = float(ttl)
        self._store: OrderedDict[Any, tuple[float, Any]] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)

    def __contains__(self, key: Any) -> bool:
        return self.get(key, default=_MISSING) is not _MISSING

    def get(self, key: Any, *, default: Any = None) -> Any:
        """Return the cached value or ``default`` if absent/expired."""

        now = time.monotonic()
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                self.misses += 1
                return default
            expires_at, value = entry
            if expires_at <= now:
                # Expired: drop it.
                self._store.pop(key, None)
                self.misses += 1
                return default
            # LRU bump.
            self._store.move_to_end(key)
            self.hits += 1
            return value

    def set(self, key: Any, value: Any, ttl_seconds: float | None = None) -> None:
        """Insert/update ``key`` with optional per-entry TTL."""

        ttl = self._default_ttl if ttl_seconds is None else float(ttl_seconds)
        expires_at = time.monotonic() + ttl
        with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
            self._store[key] = (expires_at, value)
            while len(self._store) > self._maxsize:
                self._store.popitem(last=False)

    def delete(self, key: Any) -> bool:
        """Remove ``key`` if present; returns whether anything was removed."""

        with self._lock:
            return self._store.pop(key, None) is not None

    def clear(self) -> None:
        with self._lock:
            self._store.clear()
            self.hits = 0
            self.misses = 0

    def purge_expired(self) -> int:
        """Drop all expired entries; returns the number removed."""

        now = time.monotonic()
        removed = 0
        with self._lock:
            for key in list(self._store):
                expires_at, _ = self._store[key]
                if expires_at <= now:
                    self._store.pop(key, None)
                    removed += 1
        return removed


# ---- Decorator ---------------------------------------------------------

# Global registry so independently-decorated functions can share a cache
# by name when desired. Empty-name decorators get their own per-function
# cache instance.
_named_caches: dict[str, LRUTTLCache] = {}
_named_caches_lock = threading.Lock()


def _resolve_cache(name: str | None, *, maxsize: int, ttl: float) -> LRUTTLCache:
    if not name:
        return LRUTTLCache(maxsize=maxsize, ttl=ttl)
    with _named_caches_lock:
        existing = _named_caches.get(name)
        if existing is None:
            existing = LRUTTLCache(maxsize=maxsize, ttl=ttl)
            _named_caches[name] = existing
        return existing


def _make_key(func: Callable[..., Any], args: tuple, kwargs: dict[str, Any]) -> tuple:
    # ``functools._make_key`` exists but is private and Py3-version-sensitive,
    # so build our own. Falls back to ``repr`` for unhashable kwargs values.
    try:
        kw_items = tuple(sorted(kwargs.items()))
        hash(kw_items)
        return (func.__qualname__, args, kw_items)
    except TypeError:
        return (func.__qualname__, repr(args), repr(sorted(kwargs.items())))


def cached(
    *,
    ttl: float = 30.0,
    maxsize: int = 128,
    cache_name: str | None = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Memoise a function with an LRU+TTL cache.

    ``cache_name`` opts into a shared, module-global cache instance
    (useful for invalidating multiple wrappers at once). Without a name,
    each decorated function gets its own cache.

    The decorated wrapper exposes ``.cache`` (the :class:`LRUTTLCache`
    instance) and ``.cache_clear()`` for invalidation in tests.
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        cache = _resolve_cache(cache_name, maxsize=maxsize, ttl=ttl)

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            key = _make_key(func, args, kwargs)
            value = cache.get(key, default=_MISSING)
            if value is _MISSING:
                value = func(*args, **kwargs)
                cache.set(key, value, ttl_seconds=ttl)
            return value  # type: ignore[return-value]

        wrapper.cache = cache  # type: ignore[attr-defined]
        wrapper.cache_clear = cache.clear  # type: ignore[attr-defined]
        return wrapper

    return decorator


__all__ = ["LRUTTLCache", "cached"]
