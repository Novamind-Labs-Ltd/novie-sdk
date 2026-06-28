"""Lazy Langfuse v2 client. Construction is non-network; any failure = None = instant fallback."""
from __future__ import annotations

from typing import Any, Protocol

from . import config


class PromptClient(Protocol):
    """Structural type of the client seam — both the Langfuse v2 client and the
    test FakeClient satisfy this. Keeps `registry` type-sound under mypy."""

    def get_prompt(self, name: str, **kwargs: Any) -> Any: ...


_UNSET: Any = object()

_cached: PromptClient | None = _UNSET
_override: PromptClient | None = _UNSET


def set_client_for_test(client: PromptClient | None) -> None:
    global _override
    _override = client


def reset_client() -> None:
    global _cached, _override
    _cached = _UNSET
    _override = _UNSET


def invalidate_cache() -> None:
    """Clear the lazily-built real client (e.g. on re-configure), but NOT the test
    override — configure() must not stomp a test's installed fake."""
    global _cached
    _cached = _UNSET


def _build_client(conn: "config.Connection") -> PromptClient:
    from langfuse import Langfuse  # imported lazily so the package loads without it at rest

    return Langfuse(host=conn.host, public_key=conn.public_key, secret_key=conn.secret_key)


def get_client() -> PromptClient | None:
    if _override is not _UNSET:
        return _override
    global _cached
    if _cached is not _UNSET:
        return _cached
    # ponytail: a concurrent double-build under a race is harmless — Langfuse
    # construction is non-network and idempotent; the discarded client is GC'd.
    # No lock (would be over-engineering for an idempotent build).
    conn = config.get_connection()
    if conn is None:
        _cached = None
        return None
    try:
        _cached = _build_client(conn)
    except Exception:
        _cached = None
    return _cached
