# ruff: noqa: RUF002, RUF003
"""ADR-026 — SDK tenant scoping middleware + cache helper tests."""
from __future__ import annotations

import pytest
from fastapi import HTTPException

from novie_agent_sdk import (
    RequestHeaders,
    TenantScopedCache,
    validate_tenant_context,
)


# ─── validate_tenant_context ─────────────────────────────────────────────────


def test_validate_skips_when_enforcement_off(monkeypatch) -> None:
    """Default (no env / dev mode) — validator returns silently even
    on empty tenant_id. Preserves existing tests / local-dev shape."""
    monkeypatch.delenv("NOVIE_AGENT_REQUIRE_TENANT_CONTEXT", raising=False)
    monkeypatch.delenv("NOVIE_RUNTIME_MODE", raising=False)
    monkeypatch.delenv("NOVIE_ENV", raising=False)

    validate_tenant_context(RequestHeaders(tenant_id=""))  # no raise


def test_validate_rejects_empty_when_enforced(monkeypatch) -> None:
    """Production / explicit env var → empty tenant_id is rejected
    with a 400 HTTPException carrying the ADR-026 invariant string."""
    monkeypatch.setenv("NOVIE_AGENT_REQUIRE_TENANT_CONTEXT", "1")

    with pytest.raises(HTTPException) as exc_info:
        validate_tenant_context(RequestHeaders(tenant_id=""))

    assert exc_info.value.status_code == 400
    detail = exc_info.value.detail
    assert detail["error"] == "tenant_context_required"
    assert "agent_process_no_cross_tenant_state_pollution" in detail["reason"]


def test_validate_rejects_whitespace_tenant_id(monkeypatch) -> None:
    """Whitespace-only tenant_id is treated as empty — defends against
    a misconfigured platform sending '   '."""
    monkeypatch.setenv("NOVIE_AGENT_REQUIRE_TENANT_CONTEXT", "1")

    with pytest.raises(HTTPException):
        validate_tenant_context(RequestHeaders(tenant_id="   "))


def test_validate_passes_when_tenant_id_present(monkeypatch) -> None:
    monkeypatch.setenv("NOVIE_AGENT_REQUIRE_TENANT_CONTEXT", "1")

    validate_tenant_context(RequestHeaders(tenant_id="tenant-novamind"))  # no raise


def test_required_kwarg_overrides_env_off(monkeypatch) -> None:
    """``required=True`` forces enforcement regardless of env. Lets
    callers (e.g., tests) exercise the strict path without setting
    env vars."""
    monkeypatch.delenv("NOVIE_AGENT_REQUIRE_TENANT_CONTEXT", raising=False)

    with pytest.raises(HTTPException):
        validate_tenant_context(RequestHeaders(tenant_id=""), required=True)


def test_required_kwarg_overrides_env_on(monkeypatch) -> None:
    """``required=False`` forces skip — useful for internal callbacks
    that legitimately run without tenant context."""
    monkeypatch.setenv("NOVIE_AGENT_REQUIRE_TENANT_CONTEXT", "1")

    validate_tenant_context(
        RequestHeaders(tenant_id=""), required=False,
    )  # no raise


def test_production_runtime_mode_triggers_enforcement(monkeypatch) -> None:
    monkeypatch.delenv("NOVIE_AGENT_REQUIRE_TENANT_CONTEXT", raising=False)
    monkeypatch.setenv("NOVIE_RUNTIME_MODE", "production")

    with pytest.raises(HTTPException):
        validate_tenant_context(RequestHeaders(tenant_id=""))


def test_production_env_triggers_enforcement(monkeypatch) -> None:
    monkeypatch.delenv("NOVIE_AGENT_REQUIRE_TENANT_CONTEXT", raising=False)
    monkeypatch.delenv("NOVIE_RUNTIME_MODE", raising=False)
    monkeypatch.setenv("NOVIE_ENV", "production")

    with pytest.raises(HTTPException):
        validate_tenant_context(RequestHeaders(tenant_id=""))


def test_validate_accepts_duck_typed_headers(monkeypatch) -> None:
    """The validator reads ``.tenant_id`` from any object — not
    coupled to ``RequestHeaders``. Lets test fakes work."""
    monkeypatch.setenv("NOVIE_AGENT_REQUIRE_TENANT_CONTEXT", "1")

    class _Fake:
        tenant_id = "t-1"

    validate_tenant_context(_Fake())  # no raise


# ─── TenantScopedCache ────────────────────────────────────────────────────────


def test_cache_get_and_set_per_tenant() -> None:
    cache: TenantScopedCache[str] = TenantScopedCache()
    cache.set("t1", "value-1")
    cache.set("t2", "value-2")

    assert cache.get("t1") == "value-1"
    assert cache.get("t2") == "value-2"
    assert cache.get("t3") is None


def test_cache_set_rejects_empty_tenant_id() -> None:
    """ADR-026: refuse to write a tenant-agnostic cache row. Stops
    "let's just key it by session" mistakes at the cache surface."""
    cache: TenantScopedCache[str] = TenantScopedCache()

    with pytest.raises(ValueError, match="ADR-026"):
        cache.set("", "value")
    with pytest.raises(ValueError, match="ADR-026"):
        cache.set("   ", "value")


def test_cache_get_normalizes_whitespace() -> None:
    cache: TenantScopedCache[str] = TenantScopedCache()
    cache.set("t1", "value-1")

    assert cache.get("t1") == "value-1"
    assert cache.get(" t1 ") == "value-1"  # whitespace stripped


def test_cache_get_with_empty_key_returns_none() -> None:
    """Defensive: ``get("")`` doesn't crash, just reports miss."""
    cache: TenantScopedCache[str] = TenantScopedCache()
    cache.set("t1", "value-1")

    assert cache.get("") is None
    assert cache.get(None) is None  # type: ignore[arg-type]


def test_cache_drop_returns_value() -> None:
    cache: TenantScopedCache[str] = TenantScopedCache()
    cache.set("t1", "value-1")

    dropped = cache.drop("t1")

    assert dropped == "value-1"
    assert cache.get("t1") is None
    assert cache.drop("t1") is None  # idempotent


def test_drop_other_tenants_keeps_active_only() -> None:
    """The headline ADR-026 hook: handler middleware can call this on
    every request boundary to ensure no other tenant's data lingers."""
    cache: TenantScopedCache[str] = TenantScopedCache()
    cache.set("t1", "v1")
    cache.set("t2", "v2")
    cache.set("t3", "v3")

    dropped = cache.drop_other_tenants(active_tenant_id="t2")

    assert set(dropped) == {"t1", "t3"}
    assert cache.get("t2") == "v2"
    assert cache.get("t1") is None
    assert cache.get("t3") is None


def test_drop_other_tenants_with_unknown_active_clears_all() -> None:
    """Unknown ``active_tenant_id`` means "we're switching to a tenant
    we've never seen" — caching is invalidated, no exception."""
    cache: TenantScopedCache[str] = TenantScopedCache()
    cache.set("t1", "v1")
    cache.set("t2", "v2")

    dropped = cache.drop_other_tenants(active_tenant_id="t-unknown")

    assert set(dropped) == {"t1", "t2"}
    assert len(cache) == 0


def test_drop_other_tenants_empty_active_clears_all() -> None:
    cache: TenantScopedCache[str] = TenantScopedCache()
    cache.set("t1", "v1")

    dropped = cache.drop_other_tenants(active_tenant_id="")

    assert dropped == ("t1",)
    assert len(cache) == 0


def test_clear_drops_everything() -> None:
    cache: TenantScopedCache[str] = TenantScopedCache()
    cache.set("t1", "v1")
    cache.set("t2", "v2")

    cache.clear()

    assert len(cache) == 0
    assert cache.tenant_ids() == ()


def test_len_reports_current_size() -> None:
    cache: TenantScopedCache[str] = TenantScopedCache()
    assert len(cache) == 0
    cache.set("t1", "v1")
    cache.set("t2", "v2")
    assert len(cache) == 2
    cache.drop("t1")
    assert len(cache) == 1


def test_tenant_ids_snapshot_is_stable() -> None:
    cache: TenantScopedCache[str] = TenantScopedCache()
    cache.set("t1", "v1")
    cache.set("t2", "v2")

    snapshot = cache.tenant_ids()

    assert set(snapshot) == {"t1", "t2"}
    # Snapshot is a tuple — mutating the cache after doesn't affect the
    # captured tuple.
    cache.drop("t1")
    assert set(snapshot) == {"t1", "t2"}


def test_cache_is_generic_over_value_type() -> None:
    """Type-system sanity: TenantScopedCache parameterises over the
    stored type so agents get the right .get() return type."""
    int_cache: TenantScopedCache[int] = TenantScopedCache()
    int_cache.set("t1", 42)
    value = int_cache.get("t1")
    assert value == 42

    dict_cache: TenantScopedCache[dict[str, str]] = TenantScopedCache()
    dict_cache.set("t1", {"role": "admin"})
    assert dict_cache.get("t1") == {"role": "admin"}
