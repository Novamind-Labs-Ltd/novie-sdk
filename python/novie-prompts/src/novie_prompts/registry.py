"""Public prompt resolution: resolve_prompt (tier+env dispatch) and
get_managed_prompt (the content fetch arm). NEVER raises; fail-soft (ADR-040)."""
from __future__ import annotations


def _classify(e: Exception) -> str:
    import httpx
    from langfuse.api import NotFoundError
    if isinstance(e, httpx.TimeoutException):   # Connect/ReadTimeout subclasses
        return "timeout"
    if isinstance(e, NotFoundError):            # 404 missing prompt/label
        return "missing"
    return "exception"
