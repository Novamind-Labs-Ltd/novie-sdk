"""Boot-time connection + per-call kill switch. No new env vars (Hard Discipline #7)."""
from __future__ import annotations

import os
from dataclasses import dataclass

CACHE_TTL_SECONDS: int = 60
FETCH_TIMEOUT_SECONDS: int = 2

_TRUE = {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Connection:
    host: str
    public_key: str
    secret_key: str

    def __repr__(self) -> str:
        # Redact creds: this object holds the Langfuse secret and is a module
        # global — its repr can land in logs or a crash reporter's frame locals.
        return f"Connection(host={self.host!r}, public_key=<redacted>, secret_key=<redacted>)"


_connection: Connection | None = None


def configure(*, host: str, public_key: str, secret_key: str) -> None:
    """Set the Langfuse connection once at boot (after the consumer resolves its secrets)."""
    global _connection
    _connection = Connection(host=host, public_key=public_key, secret_key=secret_key)
    # Invalidate any cached client (None or stale) so a fetch that ran before
    # configure(), or a re-configure with new creds, actually takes effect.
    from . import client  # late import avoids the client→config cycle at module load

    client.invalidate_cache()  # clears the real cached client, preserves any test override


def get_connection() -> Connection | None:
    return _connection


def is_enabled() -> bool:
    """Re-read the kill switch per call so flip + restart pins the fleet to fallback."""
    return os.environ.get("NOVIE_OBSERVABILITY_LANGFUSE_ENABLED", "false").strip().lower() in _TRUE


def resolve_label() -> str:
    """Resolve the Langfuse prompt label from the runtime environment, re-read
    per call (same as :func:`is_enabled`) so a flip + restart takes effect.

    Mirrors ``novie_platform.infra.runtime_mode.is_production_mode``'s env-var
    precedence without importing it (the SDK must not depend on a consumer
    repo): ``NOVIE_RUNTIME_MODE`` (preferred) wins over legacy ``NOVIE_ENV``,
    and an explicit ``NOVIE_RUNTIME_MODE=dev`` never escalates via a stray
    ``NOVIE_ENV=production``. Unset/unrecognized → ``"development"`` (the safe
    default — never accidentally serve an unreviewed dev experiment from a
    ``production``-labelled version by mistake).
    """
    runtime_mode = os.environ.get("NOVIE_RUNTIME_MODE", "").strip().lower()
    if runtime_mode == "production":
        return "production"
    if runtime_mode == "uat":
        return "uat"
    if runtime_mode == "dev":
        return "development"  # explicit override — never escalate via legacy NOVIE_ENV
    legacy = os.environ.get("NOVIE_ENV", "").strip().lower()
    if legacy == "production":
        return "production"
    if legacy == "uat":
        return "uat"
    return "development"


def reset() -> None:
    """Test seam: clear the configured connection."""
    global _connection
    _connection = None
