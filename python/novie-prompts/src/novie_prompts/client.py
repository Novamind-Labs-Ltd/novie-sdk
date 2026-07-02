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


def _patch_slash_encoding_bug(client: Any) -> None:
    """Work around a real langfuse SDK bug (verified against a live deployment,
    2026-07-02, langfuse 2.60.10): ``Langfuse._url_encode`` calls
    ``urllib.parse.quote(url)``, whose default ``safe='/'`` leaves ``/``
    un-escaped. Every real prompt id in this ecosystem uses the
    ``namespace/descriptor`` convention (ADR-080 §10 4A in the platform repo),
    so an un-encoded ``/`` splits one path segment into two — the request
    misses the API route entirely and falls through to the Langfuse web app's
    catch-all 404 page (HTML, not the JSON 404 the SDK expects to parse).
    ``get_managed_prompt`` still fails soft (returns the fallback), but the
    reason is misclassified as ``"exception"`` instead of ``"missing"``, and —
    the real cost — a genuinely seeded prompt can never be read back: every
    fetch of a slash-containing id was silently a no-op until this patch.

    langfuse already fixed this in a later major (its ``_url_encode`` gained
    an ``is_url_param`` kwarg + an httpx-version check we must not clobber),
    so this does NOT blindly override the method — it first *probes* the
    live client's actual behavior on a ``/`` and only replaces it when the
    bug is really there. No-op if the method is absent, already correct, or
    the probe itself errors (leave an unknown implementation alone)."""
    if not hasattr(client, "_url_encode"):
        return
    try:
        if client._url_encode("/") != "/":
            return  # already encodes slashes correctly — nothing to fix
    except Exception:
        return  # unknown signature/behavior — don't guess, leave it alone

    import types
    import urllib.parse

    def _fixed_url_encode(self: Any, url: str) -> str:
        return urllib.parse.quote(url, safe="")

    client._url_encode = types.MethodType(_fixed_url_encode, client)


def _build_client(conn: "config.Connection") -> PromptClient:
    from langfuse import Langfuse  # imported lazily so the package loads without it at rest

    client = Langfuse(host=conn.host, public_key=conn.public_key, secret_key=conn.secret_key)
    _patch_slash_encoding_bug(client)
    return client


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
