"""Public prompt resolution: resolve_prompt (tier+env dispatch) and
get_managed_prompt (the content fetch arm). NEVER raises; fail-soft (ADR-040)."""
from __future__ import annotations

from . import client
from .config import cache_ttl_seconds, fetch_timeout_seconds
from .telemetry import record_fallback, record_live


def _classify(e: Exception) -> str:
    import httpx
    from langfuse.api import NotFoundError
    if isinstance(e, httpx.TimeoutException):   # Connect/ReadTimeout subclasses
        return "timeout"
    if isinstance(e, NotFoundError):            # 404 missing prompt/label
        return "missing"
    return "exception"


def get_managed_prompt(name: str, *, fallback: str, label: str = "production") -> str:
    """Langfuse-managed CONTENT prompt with in-repo fallback (ADR-040 fail-soft).

    NEVER raises; latency-bounded (max_retries=1). Every exit emits one counter.
    Do NOT pass fallback= to the SDK: it swallows the error and hides timeout/missing.
    """
    lf = client.get_client()
    if lf is None:
        record_fallback(name, reason="disabled")
        return fallback
    try:
        prompt = lf.get_prompt(
            name, label=label,
            cache_ttl_seconds=cache_ttl_seconds(),
            max_retries=1,                                  # NOT 0 — 0 retries forever
            fetch_timeout_seconds=fetch_timeout_seconds(),  # per-attempt ceiling (seconds)
        )
        text = prompt.prompt
        if not isinstance(text, str):                       # chat-typed prompt
            record_fallback(name, reason="chat_type")
            return fallback
        record_live(name)
        return text
    except Exception as e:                                  # noqa: BLE001 — fail-soft
        record_fallback(name, reason=_classify(e))
        return fallback


def resolve_prompt(
    name: str, *, fallback: str, tier: str,
    is_prod: bool, control_plane_fetch_enabled: bool,
) -> str:
    """Single dispatch by tier + INJECTED environment (ADR-075 D4/D7).

    `fallback` IS the in-code constant. env/flag are injected — this package
    MUST NOT import novie_platform.
    """
    if tier == "content":
        return get_managed_prompt(name, fallback=fallback)
    if tier != "control_plane":
        return fallback                                  # fail-safe (ADR-075 D4): unknown/typo'd tier → constant, never fetched
    # tier == "control_plane"
    if is_prod:
        return fallback                                  # prod: constant only
    if control_plane_fetch_enabled:                      # T2, non-prod only
        return get_managed_prompt(name, fallback=fallback)
    return fallback
