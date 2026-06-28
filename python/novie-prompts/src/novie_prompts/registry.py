"""The load-bearing fail-soft fetch contract (ADR-075 D6). NEVER raises; latency-bounded."""
from __future__ import annotations

import httpx

from . import config
from .client import get_client
from .telemetry import record_fallback, record_live

try:  # langfuse 404 → reason="missing"; tolerate its absence at import time
    from langfuse.api import NotFoundError as _NotFoundError  # type: ignore
except Exception:  # pragma: no cover - only if langfuse internals move
    _NotFoundError = ()  # isinstance(exc, ()) is always False → never matches


def get_managed_prompt(name: str, *, fallback: str, label: str = "production") -> str:
    """Langfuse-managed prompt with in-repo fallback. Returns `fallback` when
    disabled / unreachable / slow / missing / chat-type. NEVER raises."""
    if not config.is_enabled():
        record_fallback(name, "disabled")
        return fallback
    client = get_client()
    if client is None:
        record_fallback(name, "disabled")
        return fallback
    try:
        prompt = client.get_prompt(
            name,
            label=label,
            cache_ttl_seconds=config.CACHE_TTL_SECONDS,
            max_retries=1,  # NOT 0 — backoff treats 0 as retry-forever
            fetch_timeout_seconds=config.FETCH_TIMEOUT_SECONDS,  # seconds, not ms
            # NOTE: do NOT pass fallback= — it makes the SDK swallow the error
        )
        text = prompt.prompt
        if not isinstance(text, str):  # chat-type prompt → list, not str
            record_fallback(name, "chat_type")
            return fallback
        if not text.strip():  # blank body = unusable (almost always a mis-edit) → use the constant
            record_fallback(name, "empty")
            return fallback
        record_live(name)
        return text
    except Exception as exc:
        record_fallback(name, _classify(exc))
        return fallback


def _classify(exc: BaseException) -> str:
    # ponytail: only httpx.TimeoutException → "timeout". If a future SDK/transport
    # wraps timeouts differently (e.g. socket.timeout), they fall to "exception";
    # widen this if the timeout alerting bucket ever needs that transport.
    if isinstance(exc, httpx.TimeoutException):
        return "timeout"
    if isinstance(exc, _NotFoundError):
        return "missing"
    return "exception"
