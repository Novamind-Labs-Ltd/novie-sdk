# ruff: noqa: RUF002, RUF003
"""ADR-026 — SDK middleware for cross-tenant pollution defense.

Agent processes are shared across tenant calls (cost-efficient) but
must not retain cross-tenant state in-process. RUNTIME_INVARIANTS
includes ``agent_process_no_cross_tenant_state_pollution`` as a P0
invariant; this module is the SDK-side enforcement / tooling layer.

Two surfaces:

1. **Precondition check** (``validate_tenant_context``) — fail-fast
   when a request arrives without ``tenant_id``. Gated by env /
   production so dev / unit tests keep working unchanged. Mirrors
   the same opt-in pattern as ``_requires_signed_agent_headers``.

2. **Helper for safe per-tenant caching** (``TenantScopedCache``) —
   agents that legitimately cache derived data per tenant can use
   this generic instead of building their own. It refuses to return
   one tenant's value to another and exposes a ``drop_other_tenants``
   sweep so a handler can scope its caches to the current request.

The agent author opts into the helper; the precondition check fires
automatically on every endpoint call once the production gate is on.
"""
from __future__ import annotations

import os
import threading
from typing import Generic, TypeVar

try:  # FastAPI may be optional in non-server consumers of the SDK.
    from fastapi import HTTPException
except Exception:  # pragma: no cover — bare-bones runtime
    HTTPException = None  # type: ignore[assignment]


T = TypeVar("T")


def _requires_tenant_context_validation() -> bool:
    """When to enforce the precondition.

    The check is opt-in via env / production-mode for the same reason
    signed-header verification is — local dev and unit tests
    legitimately exercise endpoints without setting ``tenant_id``.
    Production deployments and any explicit
    ``NOVIE_AGENT_REQUIRE_TENANT_CONTEXT=1`` opt-in get the strict
    behaviour.
    """
    if os.getenv("NOVIE_AGENT_REQUIRE_TENANT_CONTEXT") == "1":
        return True
    if os.getenv("NOVIE_RUNTIME_MODE", "").strip().lower() == "production":
        return True
    return os.getenv("NOVIE_ENV", "").strip().lower() == "production"


def validate_tenant_context(
    headers: object,
    *,
    required: bool | None = None,
) -> None:
    """Reject requests without a tenant_id when enforcement is on.

    ``headers`` is duck-typed: anything with a ``tenant_id`` attribute
    works (``RequestHeaders`` from this SDK, test fakes, etc.).
    Returns silently when enforcement is off (the default in dev /
    tests) — caller code stays the same.

    ``required=None`` (default) consults the env / production gate;
    callers that want unconditional enforcement pass ``required=True``,
    callers that want unconditional skip pass ``required=False``.
    """
    if required is None:
        required = _requires_tenant_context_validation()
    if not required:
        return
    tenant_id = str(getattr(headers, "tenant_id", "") or "").strip()
    if tenant_id:
        return
    if HTTPException is None:
        raise RuntimeError(
            "ADR-026: tenant_id header is required for this invocation "
            "but was empty; install novie-agent-sdk[server] for the "
            "FastAPI-shaped HTTPException response."
        )
    raise HTTPException(
        status_code=400,
        detail={
            "error": "tenant_context_required",
            "reason": (
                "ADR-026 invariant agent_process_no_cross_tenant_state_pollution: "
                "every agent invocation must carry x-novie-tenant-id. "
                "Production / NOVIE_AGENT_REQUIRE_TENANT_CONTEXT=1 enforces it."
            ),
        },
    )


class TenantScopedCache(Generic[T]):
    """Per-tenant cache helper for agent-side derived state.

    Pre-ADR-026 it was easy to write::

        _project_brief_cache: dict[str, ProjectBrief] = {}

        @agent.invoke
        async def handle(ctx):
            brief = _project_brief_cache.get(ctx.headers.session_id)
            ...

    That looks innocent but the cache key is wrong (session, not
    tenant) and a long-running agent process serving multiple tenants
    eventually leaks. ``TenantScopedCache`` is the safe shape::

        _project_brief_cache: TenantScopedCache[ProjectBrief] = (
            TenantScopedCache()
        )

        @agent.invoke
        async def handle(ctx):
            tenant_id = ctx.headers.tenant_id
            brief = _project_brief_cache.get(tenant_id)
            if brief is None:
                brief = await fetch_brief(tenant_id)
                _project_brief_cache.set(tenant_id, brief)
            # later, on a handler boundary:
            _project_brief_cache.drop_other_tenants(active_tenant_id=tenant_id)

    The class refuses to return one tenant's value under another
    tenant's key, and ``drop_other_tenants`` is the explicit-reset
    hook the per-call middleware can invoke.
    """

    def __init__(self) -> None:
        self._values: dict[str, T] = {}
        self._lock = threading.Lock()

    def get(self, tenant_id: str) -> T | None:
        key = (tenant_id or "").strip()
        if not key:
            return None
        with self._lock:
            return self._values.get(key)

    def set(self, tenant_id: str, value: T) -> None:
        key = (tenant_id or "").strip()
        if not key:
            raise ValueError(
                "TenantScopedCache.set requires a non-empty tenant_id; "
                "see ADR-026 — refusing to write a tenant-agnostic cache row."
            )
        with self._lock:
            self._values[key] = value

    def drop(self, tenant_id: str) -> T | None:
        """Remove a single tenant's entry; returns the dropped value."""
        key = (tenant_id or "").strip()
        if not key:
            return None
        with self._lock:
            return self._values.pop(key, None)

    def drop_other_tenants(self, *, active_tenant_id: str) -> tuple[str, ...]:
        """Remove every tenant's entry except ``active_tenant_id``.

        Returns the tuple of tenant ids that were dropped — useful for
        diagnostic logging when a request switches tenants mid-process.
        Calling with an unknown ``active_tenant_id`` clears the whole
        cache; that's intentional (no cached state survives across an
        unknown-tenant transition).
        """
        keep = (active_tenant_id or "").strip()
        with self._lock:
            dropped = tuple(
                tenant_id for tenant_id in self._values if tenant_id != keep
            )
            for tenant_id in dropped:
                self._values.pop(tenant_id, None)
        return dropped

    def clear(self) -> None:
        """Drop every cached entry."""
        with self._lock:
            self._values.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._values)

    def tenant_ids(self) -> tuple[str, ...]:
        """Snapshot of currently-cached tenant ids — diagnostics only."""
        with self._lock:
            return tuple(self._values.keys())


__all__ = [
    "TenantScopedCache",
    "validate_tenant_context",
]
