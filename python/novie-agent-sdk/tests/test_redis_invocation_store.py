"""Cross-replica idempotency tests for :class:`RedisOneShotInvocationStore`.

Uses ``fakeredis`` (an in-process Redis mock) so the atomic Lua scripts
are exercised without a real Redis. The point is to prove that the
same ``(tenant_id, mode, idempotency_key)`` yields exactly one
``is_new=True`` even under concurrent / cross-replica-style access,
and that lease recycling, tenant isolation, terminal transition
idempotency, and ``get_by_invocation_id`` all behave.
"""
from __future__ import annotations

import asyncio

import pytest

pytest.importorskip("redis")
fakeredis = pytest.importorskip("fakeredis")

from novie_agent_sdk.redis_invocation_store import RedisOneShotInvocationStore  # noqa: E402


def _store(client=None, **kwargs):
    if client is None:
        client = fakeredis.aioredis.FakeRedis()
    return RedisOneShotInvocationStore(client=client, key_prefix="test:inv", **kwargs)


def test_start_or_get_new_then_duplicate():
    async def run():
        store = _store()
        is_new, rec = await store.start_or_get("k1", "invoke", tenant_id="t1")
        assert is_new is True
        assert rec.status == "in_progress"
        is_new2, rec2 = await store.start_or_get("k1", "invoke", tenant_id="t1")
        assert is_new2 is False
        assert rec2.invocation_id == rec.invocation_id
        assert rec2.status == "in_progress"

    asyncio.run(run())


def test_complete_then_replay():
    async def run():
        store = _store()
        await store.start_or_get("k", "invoke", tenant_id="t")
        await store.complete(
            "k", "invoke", tenant_id="t", response={"x": 1}, events=[{"a": 1}]
        )
        is_new, rec = await store.start_or_get("k", "invoke", tenant_id="t")
        assert is_new is False
        assert rec.status == "completed"
        assert rec.response == {"x": 1}
        assert rec.events == [{"a": 1}]

    asyncio.run(run())


def test_tenant_isolation():
    async def run():
        store = _store()
        await store.start_or_get("same", "invoke", tenant_id="tenantA")
        is_new, _ = await store.start_or_get("same", "invoke", tenant_id="tenantB")
        assert is_new is True

    asyncio.run(run())


def test_tenant_isolation_via_contextvar():
    # Endpoint-style usage: tenant set on the request contextvar, store
    # methods called without an explicit tenant_id.
    from novie_agent_sdk.runtime import _invocation_tenant_var

    async def run():
        store = _store()
        token = _invocation_tenant_var.set("ctxTenantA")
        try:
            await store.start_or_get("same", "invoke")
        finally:
            _invocation_tenant_var.reset(token)
        token = _invocation_tenant_var.set("ctxTenantB")
        try:
            is_new, _ = await store.start_or_get("same", "invoke")
        finally:
            _invocation_tenant_var.reset(token)
        assert is_new is True

    asyncio.run(run())


def test_stale_lease_recycles_to_new_invocation():
    async def run():
        store = _store()
        _, rec = await store.start_or_get("k", "invoke", tenant_id="t")
        first_id = rec.invocation_id
        # Force the lease to look long-expired.
        key = store._record_key("t", "invoke", "k")
        await store._redis.hset(key, "updated_at", "1000000000000")
        await store._redis.hset(key, "lease_seconds", "1")
        is_new, rec2 = await store.start_or_get("k", "invoke", tenant_id="t")
        assert is_new is True
        assert rec2.invocation_id != first_id

    asyncio.run(run())


def test_complete_idempotent_after_terminal():
    async def run():
        store = _store()
        await store.start_or_get("k", "invoke", tenant_id="t")
        await store.complete("k", "invoke", tenant_id="t", response={"x": 1})
        # A concurrent/retried complete must not overwrite the terminal state.
        await store.complete("k", "invoke", tenant_id="t", response={"x": 2})
        _, rec = await store.start_or_get("k", "invoke", tenant_id="t")
        assert rec.response == {"x": 1}

    asyncio.run(run())


def test_get_by_invocation_id():
    async def run():
        store = _store()
        _, rec = await store.start_or_get("k", "invoke", tenant_id="t")
        fetched = await store.get_by_invocation_id(rec.invocation_id)
        assert fetched is not None
        assert fetched.invocation_id == rec.invocation_id
        assert fetched.status == "in_progress"
        missing = await store.get_by_invocation_id("does-not-exist")
        assert missing is None

    asyncio.run(run())


def test_fail_open_on_redis_error():
    class BoomClient:
        async def eval(self, *args, **kwargs):
            raise RuntimeError("redis unavailable")

    async def run():
        store = RedisOneShotInvocationStore(client=BoomClient(), fail_open=True)
        is_new, _ = await store.start_or_get("k", "invoke", tenant_id="t")
        assert is_new is True  # degrades to "always new" rather than 5xx

    asyncio.run(run())
