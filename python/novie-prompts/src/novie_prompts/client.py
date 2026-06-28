"""Lazy singleton Langfuse v2 client. Construction is non-network; any init
failure is swallowed → get_client() returns None → callers fall back instantly
(ADR-040 fail-soft)."""
from __future__ import annotations
import logging
from typing import Any

from langfuse import Langfuse as _Langfuse  # noqa: N814 (aliased for monkeypatch seam)
from . import config

_log = logging.getLogger(__name__)
_client: Any | None = None
_built = False


def reset_client() -> None:
    global _client, _built
    _client, _built = None, False


def get_client() -> Any | None:
    global _client, _built
    if _built:
        return _client
    _built = True
    if not config.is_enabled() or not config.host():
        _client = None
        return None
    cur = config.current()
    try:
        _client = _Langfuse(
            host=cur.host, public_key=cur.public_key, secret_key=cur.secret_key,
        )
    except Exception as e:  # noqa: BLE001 — any init failure = disabled mode
        _log.warning("novie-prompts: Langfuse client init failed, disabling: %s", e)
        _client = None
    return _client
