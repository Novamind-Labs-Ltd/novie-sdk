"""Typed SDK timeout and liveness defaults."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SdkTimeoutPolicy:
    capability_request_seconds: float = 8.0
    llm_request_seconds: float = 120.0
    llm_stream_read_idle_seconds: float = 60.0
    state_request_seconds: float = 30.0
    artifact_request_seconds: float = 60.0
    stream_keepalive_seconds: float = 25.0
    invocation_lease_seconds: int = 300
    subtask_idle_seconds: float = 120.0


DEFAULT_SDK_TIMEOUTS = SdkTimeoutPolicy()


def invocation_lease_renewal_seconds(lease_seconds: int) -> float:
    """Renew by elapsed time, never event count, at least three times per lease."""
    return min(60.0, max(float(lease_seconds), 1.0) / 3.0)


__all__ = [
    "DEFAULT_SDK_TIMEOUTS",
    "SdkTimeoutPolicy",
    "invocation_lease_renewal_seconds",
]
