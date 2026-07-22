"""Redis-backed one-shot invocation store for multi-pod agent runtimes."""
from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Any

from .runtime import (
    OneShotInvocationRecord,
    _current_invocation_tenant_id,
    _DEFAULT_INVOCATION_LEASE_SECONDS,
    _now_iso,
)


_DEFAULT_REDIS_PREFIX = "novie:agent-sdk:one-shot-invocations"
_DEFAULT_TTL_SECONDS = 7 * 24 * 60 * 60

_START_OR_GET_SCRIPT = """
local raw = redis.call('GET', KEYS[1])
if raw then
  local existing = cjson.decode(raw)
  local status = tostring(existing['status'] or '')
  local updated_epoch = tonumber(existing['updated_epoch'] or '0')
  local lease_seconds = tonumber(existing['lease_seconds'] or ARGV[4])
  local is_stale = status == 'in_progress'
    and updated_epoch > 0
    and (tonumber(ARGV[3]) - updated_epoch) > math.max(lease_seconds, 1)
  if not is_stale then
    return {0, raw}
  end
end
redis.call('SET', KEYS[1], ARGV[1], 'EX', tonumber(ARGV[2]))
redis.call('SET', KEYS[2], KEYS[1], 'EX', tonumber(ARGV[2]))
return {1, ARGV[1]}
"""

_COMPLETE_SCRIPT = """
local raw = redis.call('GET', KEYS[1])
if not raw then
  return 0
end
local record = cjson.decode(raw)
record['status'] = 'completed'
record['updated_at'] = ARGV[1]
record['updated_epoch'] = tonumber(ARGV[2])
record['response'] = cjson.decode(ARGV[3])
record['events'] = cjson.decode(ARGV[4])
record['error'] = cjson.null
local encoded = cjson.encode(record)
redis.call('SET', KEYS[1], encoded, 'EX', tonumber(ARGV[5]))
if record['invocation_id'] then
  redis.call('SET', ARGV[6] .. record['invocation_id'], KEYS[1], 'EX', tonumber(ARGV[5]))
end
return 1
"""

_FAIL_SCRIPT = """
local raw = redis.call('GET', KEYS[1])
if not raw then
  return 0
end
local record = cjson.decode(raw)
record['status'] = 'failed'
record['updated_at'] = ARGV[1]
record['updated_epoch'] = tonumber(ARGV[2])
record['error'] = ARGV[3]
local encoded = cjson.encode(record)
redis.call('SET', KEYS[1], encoded, 'EX', tonumber(ARGV[4]))
if record['invocation_id'] then
  redis.call('SET', ARGV[5] .. record['invocation_id'], KEYS[1], 'EX', tonumber(ARGV[4]))
end
return 1
"""

_TOUCH_SCRIPT = """
local raw = redis.call('GET', KEYS[1])
if not raw then
  return 0
end
local record = cjson.decode(raw)
if record['status'] ~= 'in_progress' then
  return 0
end
record['updated_at'] = ARGV[1]
record['updated_epoch'] = tonumber(ARGV[2])
local encoded = cjson.encode(record)
redis.call('SET', KEYS[1], encoded, 'EX', tonumber(ARGV[3]))
if record['invocation_id'] then
  redis.call('SET', ARGV[4] .. record['invocation_id'], KEYS[1], 'EX', tonumber(ARGV[3]))
end
return 1
"""


class RedisOneShotInvocationStore:
    """Shared Redis result cache for `/invoke` and `/stream` idempotency.

    Every write path is a single Lua script so multiple agent pods racing on the
    same tenant/idempotency key observe one atomic state transition.
    """

    def __init__(
        self,
        redis: Any,
        *,
        prefix: str | None = None,
        ttl_seconds: int | None = None,
    ) -> None:
        self._redis = redis
        self._prefix = (
            prefix
            or os.getenv("NOVIE_AGENT_INVOCATION_REDIS_PREFIX")
            or _DEFAULT_REDIS_PREFIX
        ).strip(":")
        self._ttl_seconds = max(
            int(
                ttl_seconds
                or _env_int(
                    "NOVIE_AGENT_INVOCATION_STORE_TTL_SECONDS",
                    _DEFAULT_TTL_SECONDS,
                )
            ),
            _DEFAULT_INVOCATION_LEASE_SECONDS * 2,
        )

    async def start_or_get(
        self,
        idempotency_key: str,
        mode: str,
    ) -> tuple[bool, OneShotInvocationRecord]:
        tenant_id = _current_invocation_tenant_id()
        record_key = self._record_key(tenant_id, mode, idempotency_key)
        record = OneShotInvocationRecord(
            idempotency_key=idempotency_key,
            mode=mode,
            tenant_id=tenant_id,
        )
        encoded = _record_to_json(record)
        result = await self._redis.eval(
            _START_OR_GET_SCRIPT,
            2,
            record_key,
            self._invocation_key(tenant_id, record.invocation_id),
            encoded,
            str(self._ttl_seconds),
            str(time.time()),
            str(record.lease_seconds),
        )
        started, raw = _decode_lua_pair(result)
        return started, _record_from_json(raw)

    async def complete(
        self,
        idempotency_key: str,
        mode: str,
        *,
        response: dict[str, Any] | None = None,
        events: list[dict[str, Any]] | None = None,
    ) -> None:
        tenant_id = _current_invocation_tenant_id()
        await self._redis.eval(
            _COMPLETE_SCRIPT,
            1,
            self._record_key(tenant_id, mode, idempotency_key),
            _now_iso(),
            str(time.time()),
            json.dumps(response),
            json.dumps(list(events or [])),
            str(self._ttl_seconds),
            self._invocation_prefix(tenant_id),
        )

    async def fail(self, idempotency_key: str, mode: str, error: str) -> None:
        tenant_id = _current_invocation_tenant_id()
        await self._redis.eval(
            _FAIL_SCRIPT,
            1,
            self._record_key(tenant_id, mode, idempotency_key),
            _now_iso(),
            str(time.time()),
            error,
            str(self._ttl_seconds),
            self._invocation_prefix(tenant_id),
        )

    async def get_by_invocation_id(
        self,
        invocation_id: str,
    ) -> OneShotInvocationRecord | None:
        tenant_id = _current_invocation_tenant_id()
        record_key = await self._redis.get(self._invocation_key(tenant_id, invocation_id))
        if not record_key:
            return None
        raw = await self._redis.get(_decode_text(record_key))
        if not raw:
            return None
        record = _record_from_json(raw)
        if record.tenant_id != tenant_id or record.invocation_id != invocation_id:
            return None
        return record

    async def touch(self, idempotency_key: str, mode: str) -> None:
        tenant_id = _current_invocation_tenant_id()
        await self._redis.eval(
            _TOUCH_SCRIPT,
            1,
            self._record_key(tenant_id, mode, idempotency_key),
            _now_iso(),
            str(time.time()),
            str(self._ttl_seconds),
            self._invocation_prefix(tenant_id),
        )

    def _record_key(self, tenant_id: str, mode: str, idempotency_key: str) -> str:
        tenant_hash = _hash_part(tenant_id)
        idem_hash = _hash_part(idempotency_key)
        return f"{self._prefix}:tenant:{tenant_hash}:mode:{mode}:idem:{idem_hash}"

    def _invocation_key(self, tenant_id: str, invocation_id: str) -> str:
        return f"{self._invocation_prefix(tenant_id)}{invocation_id}"

    def _invocation_prefix(self, tenant_id: str) -> str:
        return f"{self._prefix}:tenant:{_hash_part(tenant_id)}:invocation:"


def _record_to_json(record: OneShotInvocationRecord) -> str:
    return json.dumps(
        {
            "tenant_id": record.tenant_id,
            "idempotency_key": record.idempotency_key,
            "mode": record.mode,
            "invocation_id": record.invocation_id,
            "status": record.status,
            "created_at": record.created_at,
            "updated_at": record.updated_at,
            "updated_epoch": time.time(),
            "response": record.response,
            "events": record.events,
            "error": record.error,
            "lease_seconds": record.lease_seconds,
        },
        separators=(",", ":"),
    )


def _record_from_json(raw: Any) -> OneShotInvocationRecord:
    data = json.loads(_decode_text(raw))
    return OneShotInvocationRecord(
        tenant_id=str(data.get("tenant_id") or "_tenantless"),
        idempotency_key=str(data.get("idempotency_key") or ""),
        mode=str(data.get("mode") or ""),
        invocation_id=str(data.get("invocation_id") or ""),
        status=str(data.get("status") or "in_progress"),
        created_at=str(data.get("created_at") or _now_iso()),
        updated_at=str(data.get("updated_at") or _now_iso()),
        response=data.get("response"),
        events=data.get("events"),
        error=data.get("error"),
        lease_seconds=int(data.get("lease_seconds") or _DEFAULT_INVOCATION_LEASE_SECONDS),
    )


def _decode_lua_pair(result: Any) -> tuple[bool, Any]:
    if isinstance(result, (list, tuple)) and len(result) >= 2:
        return bool(int(result[0])), result[1]
    raise RuntimeError("Redis invocation store returned malformed Lua result")


def _decode_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _hash_part(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:32]


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, "").strip() or default)
    except ValueError:
        return default


__all__ = ["RedisOneShotInvocationStore"]
