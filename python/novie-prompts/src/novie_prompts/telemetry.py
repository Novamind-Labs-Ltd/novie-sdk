"""Fail-soft-not-fail-silent telemetry. Routes to an injected recorder, never to Langfuse."""
from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Recorder(Protocol):
    def record_fallback(self, name: str, reason: str) -> None: ...
    def record_live(self, name: str) -> None: ...


_recorder: Recorder | None = None


def set_recorder(recorder: Recorder | None) -> None:
    global _recorder
    _recorder = recorder


def has_recorder() -> bool:
    return _recorder is not None


def record_fallback(name: str, reason: str) -> None:
    if _recorder is not None:
        _recorder.record_fallback(name, reason)


def record_live(name: str) -> None:
    if _recorder is not None:
        _recorder.record_live(name)
