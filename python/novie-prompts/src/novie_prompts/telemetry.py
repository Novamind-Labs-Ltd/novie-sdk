"""Fail-soft, fail-LOUD telemetry. The recorder is INJECTED by the consumer
(its rcp_metrics sink) — NEVER Langfuse (circular during an outage, §7). Labels
are flattened into composite counter keys because rcp_metrics is a flat dict."""
from __future__ import annotations
from collections.abc import Callable

_record: Callable[[str], None] | None = None


def set_recorder(record: Callable[[str], None] | None) -> None:
    global _record
    _record = record


def record_fallback(name: str, reason: str) -> None:
    if _record is not None:
        _record(f"prompt_fallback_total__{name}__{reason}")


def record_live(name: str) -> None:
    if _record is not None:
        _record(f"prompt_served_live_total__{name}")
