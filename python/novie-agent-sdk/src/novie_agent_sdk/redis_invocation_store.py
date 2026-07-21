"""Shared Redis-backed one-shot idempotency / result store.

The default :class:`InMemoryOneShotInvocationStore` lives in a single
process and :class:`SqliteOneShotInvocationStore` lives in a single pod's
local file. Neither survives a retried request that the platform's
service mesh routes to a *different* agent replica, so the same
``Idempotency-Key`` can start a second execution and produce duplicate
artifacts / double LLM cost.

:class:`RedisOneShotInvocationStore` fixes this for multi-replica SaaS
deployments: every mutation is a single Lua script, so concurrent
``start_or_get`` calls across pods with the same ``(tenant_id, mode,
idempotency_key)`` yield exactly one ``is_new=True``. Records are
cache-only dedup state and carry a TTL, so Redis never grows unbounded.

Tenant isolation is enforced by key prefix (per AGENTS.md: shared
``NOVIE_REDIS_URL``, isolation by key prefix, no separate Redis
endpoint). The ``redis`` dependency is imported lazily so the package
still imports when Redis is unused.
"""
from __future__ import annotations

import json
import time
import uuid
from typing import Any

from .runtime import OneShotInvocationRecord, _invocation_tenant_var

_DEFAULT_KEY_PREFIX = "novie:agent:inv"
_DEFAULT_LEASE_SECONDS = 300
_DEFAULT_TTL_BUFFER_SECONDS = 3600

_START_OR_GET_LUA = """
local exists = redis.call('EXISTS', KEYS[1])
if exists == 0 then
  redis.call('HSET', KEYS[1],
    'status', 'in_progress',
    'invocation_id', ARGV[4],
    'created_at', ARGV[1],
    'updated_at', ARGV[1],
    'response_json', '',
    'events_json', '',
    'error', '',
    'lease_seconds', ARGV[2])
  redis.call('EXPIRE', KEYS[1], ARGV[3])
  redis.call('SET', KEYS[2], KEYS[1], 'EX', ARGV[3])
  return {1, 'in_progress', ARGV[4], ARGV[1], ARGV[2], '', '', ''}
end
local status = redis.call('HGET', KEYS[1], 'status')
if status ~= 'in_progress' then
  return {0, status,
    redis.call('HGET', KEYS[1], 'invocation_id'),
    redis.call('HGET', KEYS[1], 'updated_at'),
    redis.call('HGET', KEYS[1], 'lease_seconds'),
    redis.call('HGET', KEYS[1], 'error'),
    redis.call('HGET', KEYS[1], 'response_json'),
    redis.call('HGET', KEYS[1], 'events_json')}
end
local updated = redis.call('HGET', KEYS[1], 'updated_at')
local age = (ARGV[1] - tonumber(updated)) / 1000
if age > tonumber(ARGV[2]) then
  redis.call('HSET', KEYS[1],
    'status', 'in_progress',
    'invocation_id', ARGV[4],
    'created_at', ARGV[1],
    'updated_at', ARGV[1],
    'response_json', '',
    'events_json', '',
    'error', '')
  redis.call('EXPIRE', KEYS[1], ARGV[3])
  redis.call('SET', KEYS[2], KEYS[1], 'EX', ARGV[3])
  return {1, 'in_progress', ARGV[4], ARGV[1], ARGV[2], '', '', ''}
end
return {0, status,
  redis.call('HGET', KEYS[1], 'invocation_id'),
  updated,
  redis.call('HGET', KEYS[1], 'lease_seconds'),
  redis.call('HGET', KEYS[1], 'error'),
  redis.call('HGET', KEYS[1], 'response_json'),
  redis.call('HGET', KEYS[1], 'events_json')}
"""

_TRANSITION_LUA = """
if redis.call('HGET', KEYS[1], 'status') ~= 'in_progress' then
  return 0
end
redis.call('HSET', KEYS[1],
  'status', ARGV[3],
  'updated_at', ARGV[1],
  'response_json', ARGV[4],
  'events_json', ARGV[5],
  'error', ARGV[6])
redis.call('EXPIRE', KEYS[1], ARGV[2])
return 1
"""

_TOUCH_LUA = """
if redis.call('HGET', KEYS[1], 'status') == 'in_progress' then
  redis.call('HSET', KEYS[1], 'updated_at', ARGV[1])
  redis.call('EXPIRE', KEYS[1], ARGV[2])
end
return 1
"""

_GET_BY_INVOCATION_ID_LUA = """
local key = redis.call('GET', KEYS[1])
if not key then return {} end
return redis.call('HMGET', key,
  'status', 'invocation_id', 'created_at', 'updated_at',
  'response_json', 'events_json', 'error', 'lease_seconds')
"""


def _decode(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


class RedisOneShotInvocationStore:
    """Cross-replica-safe one-shot idempotency store backed by Redis.

    Construct from a URL, or inject a pre-built ``redis.asyncio`` client
    (handy for tests with ``fakeredis``).
    """

    def __init__(
        self,
        redis_url: str | None = None,
        *,
        client: Any = None,
        key_prefix: str | None = None,
        default_lease_seconds: int = _DEFAULT_LEASE_SECONDS,
        ttl_buffer_seconds: int = _DEFAULT_TTL_BUFFER_SECONDS,
        fail_open: bool = True,
    ) -> None:
        if client is None:
            if not redis_url:
                raise ValueError(
                    "RedisOneShotInvocationStore requires redis_url or client"
                )
            import redis.asyncio as _redis_asyncio

            client = _redis_asyncio.from_url(redis_url)
        self._redis = client
        self._key_prefix = key_prefix or _DEFAULT_KEY_PREFIX
        self._lease_seconds = int(default_lease_seconds)
        self._ttl_buffer = int(ttl_buffer_seconds)
        self._fail_open = fail_open
        # Store raw Lua and call via EVAL so the same code path works against
        # both real Redis and in-memory mocks (e.g. fakeredis) that may not
        # implement EVALSHA / script caching.
        self._scripts = {
            "start_or_get": _START_OR_GET_LUA,
            "transition": _TRANSITION_LUA,
            "touch": _TOUCH_LUA,
            "get_by_invocation_id": _GET_BY_INVOCATION_ID_LUA,
        }

    def _record_key(self, tenant_id: str, mode: str, idempotency_key: str) -> str:
        return f"{self._key_prefix}:{tenant_id}:{mode}:{idempotency_key}"

    def _index_key(self, invocation_id: str) -> str:
        return f"{self._key_prefix}:idx:{invocation_id}"

    def _ttl(self) -> int:
        return max(self._lease_seconds, self._ttl_buffer)

    async def start_or_get(
        self,
        idempotency_key: str,
        mode: str,
        *,
        tenant_id: str = "",
    ) -> tuple[bool, OneShotInvocationRecord]:
        tenant = tenant_id or _invocation_tenant_var.get()
        key = self._record_key(tenant, mode, idempotency_key)
        now = int(time.time() * 1000)
        lease = self._lease_seconds
        ttl = self._ttl()
        new_id = f"inv-{uuid.uuid4().hex}"
        idx = self._index_key(new_id)
        try:
            res = await self._redis.eval(
                self._scripts["start_or_get"], 2, key, idx, now, lease, ttl, new_id
            )
        except Exception:  # noqa: BLE001
            if not self._fail_open:
                raise
            import logging

            logging.getLogger(__name__).warning(
                "Redis start_or_get failed; failing open (no dedup)",
                exc_info=True,
            )
            return True, OneShotInvocationRecord(
                idempotency_key=idempotency_key,
                mode=mode,
                invocation_id=new_id,
            )
        is_new = bool(int(res[0]))
        status = _decode(res[1])
        inv = _decode(res[2])
        updated = _decode(res[3])
        lease_s = int(res[4]) if res[4] not in (None, b"") else lease
        err = _decode(res[5])
        response_json = _decode(res[6]) or None
        events_json = _decode(res[7]) or None
        return is_new, OneShotInvocationRecord(
            idempotency_key=idempotency_key,
            mode=mode,
            invocation_id=inv or new_id,
            status=status,
            created_at=updated,
            updated_at=updated,
            response=json.loads(response_json) if response_json else None,
            events=json.loads(events_json) if events_json else None,
            error=err or None,
            lease_seconds=int(lease_s),
        )

    async def complete(
        self,
        idempotency_key: str,
        mode: str,
        *,
        response: dict[str, Any] | None = None,
        events: list[dict[str, Any]] | None = None,
        tenant_id: str = "",
    ) -> None:
        tenant = tenant_id or _invocation_tenant_var.get()
        await self._transition(
            idempotency_key, mode, "completed", "", response, events, tenant
        )

    async def fail(
        self,
        idempotency_key: str,
        mode: str,
        error: str,
        *,
        tenant_id: str = "",
    ) -> None:
        tenant = tenant_id or _invocation_tenant_var.get()
        await self._transition(
            idempotency_key, mode, "failed", error, None, None, tenant
        )

    async def _transition(
        self,
        idempotency_key: str,
        mode: str,
        status: str,
        error: str,
        response: dict[str, Any] | None,
        events: list[dict[str, Any]] | None,
        tenant_id: str,
    ) -> None:
        key = self._record_key(tenant_id, mode, idempotency_key)
        now = int(time.time() * 1000)
        ttl = self._ttl()
        response_json = json.dumps(response) if response is not None else ""
        events_json = json.dumps(list(events or []))
        try:
            await self._redis.eval(
                self._scripts["transition"],
                1,
                key,
                now,
                ttl,
                status,
                response_json,
                events_json,
                error,
            )
        except Exception:  # noqa: BLE001
            if not self._fail_open:
                raise
            import logging

            logging.getLogger(__name__).warning(
                "Redis transition(%s) failed; failing open", status, exc_info=True
            )

    async def touch(
        self,
        idempotency_key: str,
        mode: str,
        *,
        tenant_id: str = "",
    ) -> None:
        tenant = tenant_id or _invocation_tenant_var.get()
        key = self._record_key(tenant, mode, idempotency_key)
        now = int(time.time() * 1000)
        ttl = self._ttl()
        try:
            await self._redis.eval(self._scripts["touch"], 1, key, now, ttl)
        except Exception:  # noqa: BLE001
            if not self._fail_open:
                raise
            import logging

            logging.getLogger(__name__).warning(
                "Redis touch failed; failing open", exc_info=True
            )

    async def get_by_invocation_id(
        self,
        invocation_id: str,
    ) -> OneShotInvocationRecord | None:
        idx = self._index_key(invocation_id)
        try:
            res = await self._redis.eval(
                self._scripts["get_by_invocation_id"], 1, idx
            )
        except Exception:  # noqa: BLE001
            if not self._fail_open:
                raise
            import logging

            logging.getLogger(__name__).warning(
                "Redis get_by_invocation_id failed; failing open",
                exc_info=True,
            )
            return None
        if not res:
            return None
        status = _decode(res[0])
        inv = _decode(res[1])
        created = _decode(res[2])
        updated = _decode(res[3])
        response_json = _decode(res[4]) or None
        events_json = _decode(res[5]) or None
        error = _decode(res[6]) or None
        lease_s = int(res[7]) if res[7] not in (None, b"") else self._lease_seconds
        return OneShotInvocationRecord(
            idempotency_key="",
            mode="",
            invocation_id=inv,
            status=status,
            created_at=created,
            updated_at=updated,
            response=json.loads(response_json) if response_json else None,
            events=json.loads(events_json) if events_json else None,
            error=error or None,
            lease_seconds=int(lease_s),
        )


__all__ = ["RedisOneShotInvocationStore"]
