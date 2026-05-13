"""A2A Agent Runtime —— Python SDK v2 核心实现。

提供完整的 A2A HTTP 运行时，托管：
  GET  /healthz
  GET  /.well-known/agent.json
  POST /invoke              (simple mode)
  POST /stream              (stream mode)
  POST /tasks               (tasks mode)
  GET  /tasks/{task_id}
  GET  /tasks/{task_id}/events
  GET  /tasks/{task_id}/result
  POST /tasks/{task_id}/cancel

开发者体验：
    from novie_agent_sdk.runtime import Agent, TaskContext
    import asyncio

    agent = Agent.from_manifest(".well-known/agent.json")

    @agent.task
    async def handle(ctx: TaskContext) -> dict:
        await ctx.emit_message("Processing...")
        return {"result": "done"}

    if __name__ == "__main__":
        asyncio.run(agent.serve())
"""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import os
import re
import sqlite3
import threading
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from typing import Protocol

_log = logging.getLogger(__name__)
_SAFE_AGENT_ID_CHARS = re.compile(r"[^A-Za-z0-9_.-]+")

# FastAPI is an optional dependency. Import at module level so that FastAPI's
# type-annotation resolver (get_type_hints) can find Request/BackgroundTasks
# in the module globals — if imported only inside build_app(), FastAPI would
# mis-parse ``request: Request`` as a query parameter instead of the HTTP request.
try:
    from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
    from fastapi.responses import JSONResponse, StreamingResponse

    _FASTAPI_AVAILABLE = True
except ImportError:
    BackgroundTasks = None  # type: ignore[assignment,misc]
    FastAPI = None  # type: ignore[assignment,misc]
    HTTPException = None  # type: ignore[assignment,misc]
    Request = None  # type: ignore[assignment,misc]
    JSONResponse = None  # type: ignore[assignment,misc]
    StreamingResponse = None  # type: ignore[assignment,misc]
    _FASTAPI_AVAILABLE = False

# ── Task lifecycle (re-uses protocol contract) ────────────────────────────────

from novie_protocol.contracts.agent_sdk_v2 import (
    AgentManifestV2,
    TERMINAL_STATUSES,
    is_valid_transition,
)

from .observability import AgentObservability, ObservabilitySink, build_default_sinks


# ── Contexts ──────────────────────────────────────────────────────────────────

@dataclass
class RequestHeaders:
    """A2A 请求头，由 Platform 注入。"""
    tenant_id: str = ""
    session_id: str = ""
    step_id: str = ""
    trace_id: str = ""
    workspace_id: str = ""
    project_id: str = ""
    user_id: str = ""
    service_principal: str = ""
    auth_source: str = ""
    request_id: str = ""
    timestamp: str = ""
    signature: str = ""
    idempotency_key: str = ""
    auth_token: str = ""
    raw: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_request(cls, headers: Any) -> "RequestHeaders":
        """从 FastAPI Request.headers 提取 Novie 标准 headers。"""
        h = dict(headers)
        return cls(
            tenant_id=h.get("x-novie-tenant-id", ""),
            session_id=h.get("x-novie-session-id", ""),
            step_id=h.get("x-novie-step-id", ""),
            trace_id=h.get("x-novie-trace-id", ""),
            workspace_id=h.get("x-novie-workspace-id", ""),
            project_id=h.get("x-novie-project-id", ""),
            user_id=h.get("x-novie-user-id", ""),
            service_principal=h.get("x-novie-service-principal", ""),
            auth_source=h.get("x-novie-auth-source", ""),
            request_id=h.get("x-novie-request-id", ""),
            timestamp=h.get("x-novie-timestamp", ""),
            signature=h.get("x-novie-sig", ""),
            idempotency_key=h.get("idempotency-key", ""),
            auth_token=h.get("authorization", "").removeprefix("Bearer ").strip(),
            raw=h,
        )

    def verify_signature(self, secret: str, *, ttl_seconds: int = 300) -> None:
        """Verify platform-signed A2A identity headers."""
        if not self.timestamp or not self.signature:
            raise HTTPException(
                401,
                detail={
                    "error": "a2a_signature_required",
                    "reason": "x-novie-timestamp and x-novie-sig are required",
                },
            )
        try:
            issued_at = int(self.timestamp)
        except ValueError as exc:
            raise HTTPException(
                401,
                detail={"error": "invalid_a2a_signature_timestamp"},
            ) from exc
        if abs(int(time.time()) - issued_at) > max(1, ttl_seconds):
            raise HTTPException(401, detail={"error": "stale_a2a_signature"})
        expected = _a2a_header_signature(self, secret)
        provided = self.signature.removeprefix("sha256=").strip()
        if not hmac.compare_digest(expected, provided):
            raise HTTPException(401, detail={"error": "invalid_a2a_signature"})


def _requires_signed_agent_headers() -> bool:
    if os.getenv("NOVIE_AGENT_REQUIRE_SIGNED_HEADERS") == "1":
        return True
    if os.getenv("NOVIE_RUNTIME_MODE", "").strip().lower() == "production":
        return True
    return os.getenv("NOVIE_ENV", "").strip().lower() == "production"


def _verify_agent_request_headers(headers: RequestHeaders) -> None:
    if not _requires_signed_agent_headers():
        return
    secret = os.getenv("NOVIE_A2A_SHARED_SECRET", "").strip()
    if not secret:
        raise HTTPException(
            401,
            detail={
                "error": "a2a_signature_required",
                "reason": "NOVIE_A2A_SHARED_SECRET is not configured",
            },
        )
    ttl = int(os.getenv("NOVIE_A2A_SIGNATURE_TTL_SECONDS", "300"))
    headers.verify_signature(secret, ttl_seconds=ttl)


def _a2a_header_signature(headers: RequestHeaders, secret: str) -> str:
    canonical = "\n".join(
        [
            headers.tenant_id,
            headers.workspace_id,
            headers.project_id,
            headers.user_id,
            headers.service_principal,
            headers.session_id,
            headers.step_id,
            headers.idempotency_key,
            headers.timestamp,
        ]
    )
    return hmac.new(
        secret.encode("utf-8"),
        canonical.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


@dataclass
class InvokeContext:
    """simple protocol handler 上下文。"""
    input: dict[str, Any]
    headers: RequestHeaders
    agent_manifest: AgentManifestV2
    observability: AgentObservability

    @property
    def brief(self) -> dict[str, Any]:
        return self.input.get("brief", {})


@dataclass
class StreamContext:
    """stream protocol handler 上下文。"""
    input: dict[str, Any]
    headers: RequestHeaders
    agent_manifest: AgentManifestV2
    observability: AgentObservability

    @property
    def brief(self) -> dict[str, Any]:
        return self.input.get("brief", {})


@dataclass
class TaskContext:
    """tasks protocol handler 上下文。

    用于在 task handler 中发布 events、检查 cancel token 等。
    """
    task_id: str
    input: dict[str, Any]
    headers: RequestHeaders
    agent_manifest: AgentManifestV2
    observability: AgentObservability
    _store: "TaskStore"
    _cancelled: asyncio.Event = field(default_factory=asyncio.Event)

    @property
    def brief(self) -> dict[str, Any]:
        return self.input.get("brief", {})

    @property
    def is_cancelled(self) -> bool:
        return self._cancelled.is_set()

    async def emit_event(self, kind: str, payload: dict[str, Any] | None = None, summary: str = "") -> None:
        """发布一条 task event。"""
        event = _make_event(self.task_id, kind, payload or {}, summary)
        await self._store.append_event(self.task_id, event)

    async def emit_message(self, text: str) -> None:
        await self.emit_event("message", {"text": text}, summary=text[:200])

    async def emit_artifact(self, artifact: dict[str, Any]) -> None:
        await self.emit_event("artifact_created", artifact)

    async def report_llm_usage(
        self,
        *,
        provider: str,
        model: str,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        total_tokens: int | None = None,
        latency_ms: float | None = None,
        phase: str | None = None,
        turn_id: str | None = None,
        span_name: str | None = None,
        idempotency_key: str | None = None,
        raw_usage_metadata: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Report one LLM usage event in the platform A2A event format."""
        report = await self.observability.report_llm_usage(
            provider=provider,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            latency_ms=latency_ms,
            phase=phase,
            turn_id=turn_id,
            span_name=span_name,
            idempotency_key=idempotency_key,
            raw_usage_metadata=raw_usage_metadata,
            metadata=metadata,
        )
        return report.to_dict()

    async def set_status(self, status: str) -> None:
        await self._store.update_task_status(self.task_id, status)
        await self.emit_event("status_changed", {"status": status})


# ── Task / Event stores ───────────────────────────────────────────────────────

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _env_int(name: str, *, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, *, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _make_event(task_id: str, kind: str, payload: dict[str, Any], summary: str = "") -> dict[str, Any]:
    return {
        "event_id": f"evt-{uuid.uuid4().hex[:16]}",
        "task_id": task_id,
        "kind": kind,
        "timestamp": _now_iso(),
        "summary": summary,
        **payload,
    }


@dataclass
class TaskRecord:
    task_id: str
    status: str = "pending"
    created_at: str = field(default_factory=_now_iso)
    updated_at: str = field(default_factory=_now_iso)
    input: dict[str, Any] = field(default_factory=dict)
    idempotency_key: str = ""
    result: dict[str, Any] | None = None
    error: str | None = None


@dataclass
class OneShotInvocationRecord:
    idempotency_key: str
    mode: str
    status: str = "in_progress"
    created_at: str = field(default_factory=_now_iso)
    updated_at: str = field(default_factory=_now_iso)
    response: dict[str, Any] | None = None
    events: list[dict[str, Any]] | None = None
    error: str | None = None


class OneShotInvocationStore(Protocol):
    async def start_or_get(
        self,
        idempotency_key: str,
        mode: str,
    ) -> tuple[bool, OneShotInvocationRecord]: ...

    async def complete(
        self,
        idempotency_key: str,
        mode: str,
        *,
        response: dict[str, Any] | None = None,
        events: list[dict[str, Any]] | None = None,
    ) -> None: ...

    async def fail(self, idempotency_key: str, mode: str, error: str) -> None: ...


class InMemoryOneShotInvocationStore:
    """In-memory one-shot idempotency cache for invoke/stream endpoints."""

    def __init__(self) -> None:
        self._records: dict[tuple[str, str], OneShotInvocationRecord] = {}
        self._lock = asyncio.Lock()

    async def start_or_get(
        self,
        idempotency_key: str,
        mode: str,
    ) -> tuple[bool, OneShotInvocationRecord]:
        async with self._lock:
            key = (mode, idempotency_key)
            existing = self._records.get(key)
            if existing is not None:
                return False, existing
            record = OneShotInvocationRecord(
                idempotency_key=idempotency_key,
                mode=mode,
            )
            self._records[key] = record
            return True, record

    async def complete(
        self,
        idempotency_key: str,
        mode: str,
        *,
        response: dict[str, Any] | None = None,
        events: list[dict[str, Any]] | None = None,
    ) -> None:
        async with self._lock:
            record = self._records.get((mode, idempotency_key))
            if record is None:
                return
            record.status = "completed"
            record.updated_at = _now_iso()
            record.response = response
            record.events = list(events or [])
            record.error = None

    async def fail(self, idempotency_key: str, mode: str, error: str) -> None:
        async with self._lock:
            record = self._records.get((mode, idempotency_key))
            if record is None:
                return
            record.status = "failed"
            record.updated_at = _now_iso()
            record.error = error


class SqliteOneShotInvocationStore:
    """Durable SQLite-backed one-shot idempotency/result cache."""

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._lock = threading.Lock()
        self._ensure_schema()

    async def start_or_get(
        self,
        idempotency_key: str,
        mode: str,
    ) -> tuple[bool, OneShotInvocationRecord]:
        return await asyncio.to_thread(
            self._start_or_get_sync, idempotency_key, mode,
        )

    def _start_or_get_sync(
        self,
        idempotency_key: str,
        mode: str,
    ) -> tuple[bool, OneShotInvocationRecord]:
        with self._lock, sqlite3.connect(self._db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                """
                SELECT * FROM sdk_one_shot_invocations
                WHERE mode = ? AND idempotency_key = ?
                """,
                (mode, idempotency_key),
            ).fetchone()
            if row is not None:
                return False, self._row_to_record(row)
            record = OneShotInvocationRecord(
                idempotency_key=idempotency_key,
                mode=mode,
            )
            conn.execute(
                """
                INSERT INTO sdk_one_shot_invocations (
                    mode, idempotency_key, status, created_at, updated_at,
                    response_json, events_json, error
                ) VALUES (?, ?, ?, ?, ?, NULL, NULL, NULL)
                """,
                (
                    mode,
                    idempotency_key,
                    record.status,
                    record.created_at,
                    record.updated_at,
                ),
            )
            conn.commit()
            return True, record

    async def complete(
        self,
        idempotency_key: str,
        mode: str,
        *,
        response: dict[str, Any] | None = None,
        events: list[dict[str, Any]] | None = None,
    ) -> None:
        await asyncio.to_thread(
            self._complete_sync,
            idempotency_key,
            mode,
            response,
            list(events or []),
        )

    def _complete_sync(
        self,
        idempotency_key: str,
        mode: str,
        response: dict[str, Any] | None,
        events: list[dict[str, Any]],
    ) -> None:
        with self._lock, sqlite3.connect(self._db_path) as conn:
            conn.execute(
                """
                UPDATE sdk_one_shot_invocations
                SET status = 'completed',
                    updated_at = ?,
                    response_json = ?,
                    events_json = ?,
                    error = NULL
                WHERE mode = ? AND idempotency_key = ?
                """,
                (
                    _now_iso(),
                    json.dumps(response) if response is not None else None,
                    json.dumps(events),
                    mode,
                    idempotency_key,
                ),
            )
            conn.commit()

    async def fail(self, idempotency_key: str, mode: str, error: str) -> None:
        await asyncio.to_thread(self._fail_sync, idempotency_key, mode, error)

    def _fail_sync(self, idempotency_key: str, mode: str, error: str) -> None:
        with self._lock, sqlite3.connect(self._db_path) as conn:
            conn.execute(
                """
                UPDATE sdk_one_shot_invocations
                SET status = 'failed', updated_at = ?, error = ?
                WHERE mode = ? AND idempotency_key = ?
                """,
                (_now_iso(), error, mode, idempotency_key),
            )
            conn.commit()

    def _row_to_record(self, row: sqlite3.Row) -> OneShotInvocationRecord:
        response_raw = row["response_json"]
        events_raw = row["events_json"]
        return OneShotInvocationRecord(
            idempotency_key=str(row["idempotency_key"]),
            mode=str(row["mode"]),
            status=str(row["status"]),
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
            response=json.loads(str(response_raw)) if response_raw else None,
            events=json.loads(str(events_raw)) if events_raw else None,
            error=str(row["error"]) if row["error"] is not None else None,
        )

    def _ensure_schema(self) -> None:
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sdk_one_shot_invocations (
                    mode TEXT NOT NULL,
                    idempotency_key TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    response_json TEXT,
                    events_json TEXT,
                    error TEXT,
                    PRIMARY KEY (mode, idempotency_key)
                )
                """,
            )
            conn.commit()


def _duplicate_one_shot_response(record: OneShotInvocationRecord) -> Any:
    return JSONResponse(
        status_code=409,
        content={
            "error": {
                "code": "retry_in_progress",
                "message": (
                    f"{record.mode} invocation for this Idempotency-Key is "
                    f"{record.status}; retry after it reaches a terminal state"
                ),
                "retryable": True,
                "classification": "state",
                "details": {
                    "idempotency_key": record.idempotency_key,
                    "mode": record.mode,
                    "status": record.status,
                },
            }
        },
    )


_INVOKE_RESPONSE_STATUSES = {
    "completed",
    "failed",
    "cancelled",
    "needs_confirmation",
}


def _coerce_invoke_response(result: dict[str, Any]) -> dict[str, Any]:
    """Wrap legacy handler output unless it already is a response envelope."""
    if isinstance(result, dict):
        status = result.get("status")
        if (
            isinstance(status, str)
            and status in _INVOKE_RESPONSE_STATUSES
            and (
                "output" in result
                or "error" in result
                or "confirmation" in result
            )
        ):
            return dict(result)
    return {"output": result, "status": "completed"}


def _default_invocation_store(manifest: AgentManifestV2) -> OneShotInvocationStore:
    if getattr(manifest.execution, "durability", "none") == "result_cache":
        return SqliteOneShotInvocationStore(
            str(_default_invocation_sqlite_path(manifest.agent_id)),
        )
    return InMemoryOneShotInvocationStore()


def _default_invocation_sqlite_path(agent_id: str) -> Path:
    explicit = os.getenv("NOVIE_AGENT_INVOCATION_STORE_PATH", "").strip()
    if explicit:
        path = Path(explicit)
    else:
        base = Path(os.getenv("NOVIE_AGENT_STATE_DIR", "").strip() or ".novie")
        safe_agent_id = _SAFE_AGENT_ID_CHARS.sub("-", agent_id).strip(".-")
        path = base / f"{safe_agent_id or 'agent'}.invocations.sqlite3"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


class TaskStore(Protocol):
    async def create_task(
        self,
        task_id: str,
        input_data: dict[str, Any],
        idempotency_key: str = "",
    ) -> TaskRecord: ...

    async def get_task(self, task_id: str) -> TaskRecord | None: ...
    async def update_task_status(self, task_id: str, new_status: str) -> None: ...
    async def set_task_result(
        self,
        task_id: str,
        result: dict[str, Any],
        status: str = "completed",
    ) -> None: ...
    async def set_task_error(self, task_id: str, error: str) -> None: ...
    async def cancel_task(self, task_id: str) -> bool: ...
    async def append_event(self, task_id: str, event: dict[str, Any]) -> None: ...
    async def get_events(self, task_id: str) -> list[dict[str, Any]]: ...
    async def get_result(self, task_id: str) -> dict[str, Any] | None: ...


class InMemoryTaskStore:
    """线程安全的 in-memory task + event 存储。

    TD #35 (2026-05-11) — bounded capacity + TTL eviction:

    - ``max_tasks``: when set, evicts the oldest task (FIFO insertion
      order) on insert once the limit is exceeded. ``None`` means
      unbounded (legacy behaviour, suitable only for short-lived test
      runs).
    - ``ttl_seconds``: when set, any task older than this is evicted
      lazily on each ``create_task`` call. ``None`` means no TTL.
    - Env overrides (`NOVIE_SDK_TASK_STORE_MAX_TASKS` /
      `NOVIE_SDK_TASK_STORE_TTL_SECONDS`) let deployers tune without
      touching code; explicit constructor args win over env.

    Eviction is cooperative — it happens at ``create_task`` time, so a
    completely idle store keeps holding tasks. Production deployments
    should call :meth:`evict_expired` from a background task if they
    need strict TTL enforcement without new traffic.
    """

    def __init__(
        self,
        *,
        max_tasks: int | None = None,
        ttl_seconds: float | None = None,
    ) -> None:
        self._tasks: dict[str, TaskRecord] = {}
        self._events: dict[str, list[dict[str, Any]]] = {}
        self._lock = asyncio.Lock()
        self._status_waiters: dict[str, asyncio.Condition] = {}
        self._max_tasks = max_tasks if max_tasks is not None else _env_int(
            "NOVIE_SDK_TASK_STORE_MAX_TASKS", default=0
        ) or None
        self._ttl_seconds = ttl_seconds if ttl_seconds is not None else _env_float(
            "NOVIE_SDK_TASK_STORE_TTL_SECONDS", default=0.0
        ) or None

    def _evict_expired_locked(self) -> None:
        """Drop tasks whose ``created_at`` is older than the TTL.

        Caller must hold ``self._lock``. ISO timestamp parsing is
        defensive — anything we can't parse is left alone (better than
        crashing the producer path on malformed legacy rows).
        """
        if self._ttl_seconds is None:
            return
        now = time.time()
        expired: list[str] = []
        for task_id, record in self._tasks.items():
            try:
                created_ts = datetime.fromisoformat(
                    record.created_at.replace("Z", "+00:00")
                ).timestamp()
            except (ValueError, AttributeError):
                continue
            if now - created_ts > self._ttl_seconds:
                expired.append(task_id)
        for task_id in expired:
            self._tasks.pop(task_id, None)
            self._events.pop(task_id, None)
            self._status_waiters.pop(task_id, None)

    def _evict_oldest_locked(self) -> None:
        """Drop the single oldest task (FIFO insertion order).

        Caller must hold ``self._lock``. Python dicts preserve
        insertion order, so the first key IS the oldest.
        """
        if not self._tasks:
            return
        oldest = next(iter(self._tasks))
        self._tasks.pop(oldest, None)
        self._events.pop(oldest, None)
        self._status_waiters.pop(oldest, None)

    async def evict_expired(self) -> int:
        """Public TTL sweep — returns the count of evicted tasks.

        Background sweeper hook: call this on a timer if you need
        strict TTL without producer traffic.
        """
        before = len(self._tasks)
        async with self._lock:
            self._evict_expired_locked()
        return before - len(self._tasks)

    async def create_task(
        self,
        task_id: str,
        input_data: dict[str, Any],
        idempotency_key: str = "",
    ) -> TaskRecord:
        async with self._lock:
            # TD #35 — evict expired tasks before checking idempotency
            # so a stale (expired) idempotency key doesn't shadow a new
            # caller.
            self._evict_expired_locked()
            if idempotency_key and any(
                t.idempotency_key == idempotency_key
                for t in self._tasks.values()
            ):
                # Idempotent: return existing task
                existing = next(
                    t for t in self._tasks.values()
                    if t.idempotency_key == idempotency_key
                )
                return existing
            # TD #35 — bound the capacity. Evict-on-insert FIFO so the
            # oldest task is dropped if we'd exceed ``max_tasks``.
            if self._max_tasks is not None and len(self._tasks) >= self._max_tasks:
                self._evict_oldest_locked()
            record = TaskRecord(
                task_id=task_id,
                input=input_data,
                idempotency_key=idempotency_key,
            )
            self._tasks[task_id] = record
            self._events[task_id] = []
            self._status_waiters[task_id] = asyncio.Condition()
            return record

    async def get_task(self, task_id: str) -> TaskRecord | None:
        return self._tasks.get(task_id)

    async def update_task_status(self, task_id: str, new_status: str) -> None:
        async with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return
            old_status = task.status
            if old_status == new_status:
                return
            if old_status in TERMINAL_STATUSES:
                return
            if not is_valid_transition(old_status, new_status):
                _log.warning(
                    "Invalid task status transition %s→%s task_id=%s",
                    old_status, new_status, task_id,
                )
                return
            task.status = new_status
            task.updated_at = _now_iso()
            cond = self._status_waiters.get(task_id)
        if cond:
            async with cond:
                cond.notify_all()

    async def set_task_result(
        self,
        task_id: str,
        result: dict[str, Any],
        status: str = "completed",
    ) -> None:
        async with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return
            task.result = result
            task.status = status
            task.updated_at = _now_iso()
            cond = self._status_waiters.get(task_id)
        if cond:
            async with cond:
                cond.notify_all()
        await self.append_event(
            task_id,
            _make_event(task_id, "task_completed" if status == "completed" else "task_failed", {}, ""),
        )

    async def set_task_error(self, task_id: str, error: str) -> None:
        async with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return
            task.error = error
            task.status = "failed"
            task.updated_at = _now_iso()
            cond = self._status_waiters.get(task_id)
        if cond:
            async with cond:
                cond.notify_all()
        await self.append_event(
            task_id,
            _make_event(task_id, "task_failed", {"error": error}, error[:200]),
        )

    async def cancel_task(self, task_id: str) -> bool:
        """取消 task。返回 True 如果成功取消。"""
        async with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return False
            if task.status in TERMINAL_STATUSES:
                return False
            task.status = "cancelled"
            task.updated_at = _now_iso()
            cond = self._status_waiters.get(task_id)
        if cond:
            async with cond:
                cond.notify_all()
        return True

    async def append_event(self, task_id: str, event: dict[str, Any]) -> None:
        async with self._lock:
            if task_id in self._events:
                self._events[task_id].append(event)

    async def get_events(self, task_id: str) -> list[dict[str, Any]]:
        return list(self._events.get(task_id, []))

    async def get_result(self, task_id: str) -> dict[str, Any] | None:
        task = self._tasks.get(task_id)
        if task is None or task.result is None:
            return None
        return task.result


class SqliteTaskStore:
    """Durable SQLite-backed task store."""

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._lock = threading.Lock()
        self._ensure_schema()

    async def create_task(
        self,
        task_id: str,
        input_data: dict[str, Any],
        idempotency_key: str = "",
    ) -> TaskRecord:
        return await asyncio.to_thread(self._create_task_sync, task_id, input_data, idempotency_key)

    def _create_task_sync(self, task_id: str, input_data: dict[str, Any], idempotency_key: str) -> TaskRecord:
        with self._lock, sqlite3.connect(self._db_path) as conn:
            conn.row_factory = sqlite3.Row
            if idempotency_key:
                row = conn.execute(
                    "SELECT * FROM sdk_tasks WHERE idempotency_key = ? ORDER BY created_at ASC LIMIT 1",
                    (idempotency_key,),
                ).fetchone()
                if row is not None:
                    return self._row_to_task_record(row)
            record = TaskRecord(
                task_id=task_id,
                input=input_data,
                idempotency_key=idempotency_key,
            )
            conn.execute(
                """
                INSERT INTO sdk_tasks (
                    task_id, status, created_at, updated_at, input_json, idempotency_key, result_json, error
                ) VALUES (?, ?, ?, ?, ?, ?, NULL, NULL)
                """,
                (
                    record.task_id,
                    record.status,
                    record.created_at,
                    record.updated_at,
                    json.dumps(record.input),
                    record.idempotency_key,
                ),
            )
            conn.commit()
            return record

    async def get_task(self, task_id: str) -> TaskRecord | None:
        return await asyncio.to_thread(self._get_task_sync, task_id)

    def _get_task_sync(self, task_id: str) -> TaskRecord | None:
        with sqlite3.connect(self._db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute("SELECT * FROM sdk_tasks WHERE task_id = ?", (task_id,)).fetchone()
            if row is None:
                return None
            return self._row_to_task_record(row)

    async def update_task_status(self, task_id: str, new_status: str) -> None:
        await asyncio.to_thread(self._update_task_status_sync, task_id, new_status)

    def _update_task_status_sync(self, task_id: str, new_status: str) -> None:
        with self._lock, sqlite3.connect(self._db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute("SELECT status FROM sdk_tasks WHERE task_id = ?", (task_id,)).fetchone()
            if row is None:
                return
            old_status = str(row["status"])
            if old_status == new_status or old_status in TERMINAL_STATUSES:
                return
            if not is_valid_transition(old_status, new_status):
                return
            conn.execute(
                "UPDATE sdk_tasks SET status = ?, updated_at = ? WHERE task_id = ?",
                (new_status, _now_iso(), task_id),
            )
            conn.commit()

    async def set_task_result(
        self,
        task_id: str,
        result: dict[str, Any],
        status: str = "completed",
    ) -> None:
        await asyncio.to_thread(self._set_task_result_sync, task_id, result, status)
        await self.append_event(
            task_id,
            _make_event(task_id, "task_completed" if status == "completed" else "task_failed", {}, ""),
        )

    def _set_task_result_sync(self, task_id: str, result: dict[str, Any], status: str) -> None:
        with self._lock, sqlite3.connect(self._db_path) as conn:
            conn.execute(
                "UPDATE sdk_tasks SET status = ?, result_json = ?, updated_at = ? WHERE task_id = ?",
                (status, json.dumps(result), _now_iso(), task_id),
            )
            conn.commit()

    async def set_task_error(self, task_id: str, error: str) -> None:
        await asyncio.to_thread(self._set_task_error_sync, task_id, error)
        await self.append_event(
            task_id,
            _make_event(task_id, "task_failed", {"error": error}, error[:200]),
        )

    def _set_task_error_sync(self, task_id: str, error: str) -> None:
        with self._lock, sqlite3.connect(self._db_path) as conn:
            conn.execute(
                "UPDATE sdk_tasks SET status = 'failed', error = ?, updated_at = ? WHERE task_id = ?",
                (error, _now_iso(), task_id),
            )
            conn.commit()

    async def cancel_task(self, task_id: str) -> bool:
        return await asyncio.to_thread(self._cancel_task_sync, task_id)

    def _cancel_task_sync(self, task_id: str) -> bool:
        with self._lock, sqlite3.connect(self._db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute("SELECT status FROM sdk_tasks WHERE task_id = ?", (task_id,)).fetchone()
            if row is None:
                return False
            status = str(row["status"])
            if status in TERMINAL_STATUSES:
                return False
            conn.execute(
                "UPDATE sdk_tasks SET status = 'cancelled', updated_at = ? WHERE task_id = ?",
                (_now_iso(), task_id),
            )
            conn.commit()
            return True

    async def append_event(self, task_id: str, event: dict[str, Any]) -> None:
        await asyncio.to_thread(self._append_event_sync, task_id, event)

    def _append_event_sync(self, task_id: str, event: dict[str, Any]) -> None:
        with self._lock, sqlite3.connect(self._db_path) as conn:
            conn.execute(
                "INSERT INTO sdk_task_events (task_id, event_json, created_at) VALUES (?, ?, ?)",
                (task_id, json.dumps(event), _now_iso()),
            )
            conn.commit()

    async def get_events(self, task_id: str) -> list[dict[str, Any]]:
        return await asyncio.to_thread(self._get_events_sync, task_id)

    def _get_events_sync(self, task_id: str) -> list[dict[str, Any]]:
        with sqlite3.connect(self._db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT event_json FROM sdk_task_events WHERE task_id = ? ORDER BY id ASC",
                (task_id,),
            ).fetchall()
        out: list[dict[str, Any]] = []
        for row in rows:
            out.append(json.loads(str(row["event_json"])))
        return out

    async def get_result(self, task_id: str) -> dict[str, Any] | None:
        return await asyncio.to_thread(self._get_result_sync, task_id)

    def _get_result_sync(self, task_id: str) -> dict[str, Any] | None:
        with sqlite3.connect(self._db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute("SELECT result_json FROM sdk_tasks WHERE task_id = ?", (task_id,)).fetchone()
            if row is None or row["result_json"] is None:
                return None
            return json.loads(str(row["result_json"]))

    def _row_to_task_record(self, row: sqlite3.Row) -> TaskRecord:
        return TaskRecord(
            task_id=str(row["task_id"]),
            status=str(row["status"]),
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
            input=json.loads(str(row["input_json"] or "{}")),
            idempotency_key=str(row["idempotency_key"] or ""),
            result=json.loads(str(row["result_json"])) if row["result_json"] else None,
            error=str(row["error"]) if row["error"] is not None else None,
        )

    def _ensure_schema(self) -> None:
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sdk_tasks (
                    task_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    input_json TEXT NOT NULL,
                    idempotency_key TEXT NOT NULL DEFAULT '',
                    result_json TEXT,
                    error TEXT
                )
                """,
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_sdk_tasks_idempotency ON sdk_tasks(idempotency_key)",
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sdk_task_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id TEXT NOT NULL,
                    event_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """,
            )
            conn.commit()


# ── Registration client ───────────────────────────────────────────────────────

class RegistrationClient:
    """向 Platform ManifestRegistry 注册/心跳/注销的客户端。"""

    def __init__(
        self,
        platform_url: str,
        manifest: AgentManifestV2,
        heartbeat_interval: float = 30.0,
        *,
        required: bool = False,
        registration_token: str = "",
        register_max_attempts: int = 5,
        register_backoff_seconds: float = 2.0,
    ) -> None:
        self._platform_url = platform_url.rstrip("/")
        self._manifest = manifest
        self._heartbeat_interval = heartbeat_interval
        self._required = required
        self._registration_token = registration_token.strip()
        self._register_max_attempts = max(1, int(register_max_attempts))
        self._register_backoff_seconds = max(0.0, float(register_backoff_seconds))
        self._heartbeat_task: asyncio.Task | None = None

    def _auth_headers(self) -> dict[str, str]:
        if not self._registration_token:
            return {}
        return {
            "Agent-Secret": self._registration_token,
            "Authorization": f"Bearer {self._registration_token}",
        }

    async def register(self) -> None:
        import httpx
        last_exc: Exception | None = None
        for attempt in range(1, self._register_max_attempts + 1):
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    resp = await client.post(
                        f"{self._platform_url}/agents/register",
                        json=self._manifest.to_dict(),
                        headers=self._auth_headers(),
                    )
                    resp.raise_for_status()
                    _log.info(
                        "Registered with Platform agent_id=%s attempt=%s",
                        self._manifest.agent_id,
                        attempt,
                    )
                    return
            except Exception as exc:
                last_exc = exc
                if attempt >= self._register_max_attempts:
                    break
                delay = self._register_backoff_seconds * attempt
                _log.warning(
                    "Platform registration attempt failed agent_id=%s attempt=%s/%s retry_in=%.1fs",
                    self._manifest.agent_id,
                    attempt,
                    self._register_max_attempts,
                    delay,
                    exc_info=True,
                )
                await asyncio.sleep(delay)

        if self._required:
            raise RuntimeError(
                f"failed to register with platform for agent {self._manifest.agent_id!r}"
            ) from last_exc
        _log.warning(
            "Failed to register with Platform after %s attempts; continuing without registration",
            self._register_max_attempts,
            exc_info=last_exc,
        )

    async def deregister(self) -> None:
        import httpx
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                await client.delete(
                    f"{self._platform_url}/agents/{self._manifest.agent_id}",
                    headers=self._auth_headers(),
                )
                _log.info("Deregistered from Platform agent_id=%s", self._manifest.agent_id)
        except Exception:
            _log.debug("Deregister failed (non-fatal)", exc_info=True)

    async def start_heartbeat(self) -> None:
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

    async def stop_heartbeat(self) -> None:
        if self._heartbeat_task is not None:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass

    async def _heartbeat_loop(self) -> None:
        import httpx
        while True:
            await asyncio.sleep(self._heartbeat_interval)
            try:
                async with httpx.AsyncClient(timeout=5.0) as client:
                    resp = await client.post(
                        f"{self._platform_url}/agents/{self._manifest.agent_id}/heartbeat",
                        headers=self._auth_headers(),
                    )
                    if resp.status_code not in (200, 204):
                        _log.debug("heartbeat returned %s", resp.status_code)
            except asyncio.CancelledError:
                raise
            except Exception:
                _log.debug("Heartbeat failed (non-fatal)", exc_info=True)


# ── Agent class ───────────────────────────────────────────────────────────────

InvokeHandlerFn = Callable[[InvokeContext], Awaitable[dict[str, Any]]]
StreamHandlerFn = Callable[[StreamContext], AsyncIterator[dict[str, Any]]]
TaskHandlerFn = Callable[[TaskContext], Awaitable[dict[str, Any]]]


class Agent:
    """A2A Agent Runtime v2。

    用法（tasks 模式）：
        agent = Agent.from_manifest(".well-known/agent.json")

        @agent.task
        async def run(ctx: TaskContext) -> dict:
            await ctx.emit_message("Working...")
            return {"result": "done"}

        agent.serve()
    """

    def __init__(
        self,
        manifest: AgentManifestV2,
        *,
        task_store: TaskStore | None = None,
        invocation_store: OneShotInvocationStore | None = None,
        observability_sinks: tuple[ObservabilitySink, ...] | None = None,
    ) -> None:
        self._manifest = manifest
        self._invoke_handler: InvokeHandlerFn | None = None
        self._stream_handler: StreamHandlerFn | None = None
        self._task_handler: TaskHandlerFn | None = None
        self._store: TaskStore = task_store or InMemoryTaskStore()
        self._invocation_store: OneShotInvocationStore = (
            invocation_store or _default_invocation_store(manifest)
        )
        self._cancel_events: dict[str, asyncio.Event] = {}
        self._registration_client: RegistrationClient | None = None
        self._observability = AgentObservability(
            agent_id=manifest.agent_id,
            sinks=observability_sinks if observability_sinks is not None else build_default_sinks(),
        )

    # ── Factory ────────────────────────────────────────────────────────────────

    @classmethod
    def from_manifest(cls, path: str | Path) -> "Agent":
        """从 .well-known/agent.json 文件创建 Agent。"""
        manifest = AgentManifestV2.from_file(path)
        errors = manifest.validate()
        if errors:
            raise ValueError(f"Invalid manifest at {path}: {errors}")
        return cls(manifest)

    @classmethod
    def from_manifest_dict(cls, d: dict[str, Any]) -> "Agent":
        manifest = AgentManifestV2.from_dict(d)
        errors = manifest.validate()
        if errors:
            raise ValueError(f"Invalid manifest: {errors}")
        return cls(manifest)

    # ── Handler registration ───────────────────────────────────────────────────

    def invoke(self, fn: InvokeHandlerFn) -> InvokeHandlerFn:
        """Register simple protocol handler (decorator or direct call)."""
        self._invoke_handler = fn
        return fn

    def stream(self, fn: StreamHandlerFn) -> StreamHandlerFn:
        """Register stream protocol handler (decorator or direct call)."""
        self._stream_handler = fn
        return fn

    def task(self, fn: TaskHandlerFn) -> TaskHandlerFn:
        """Register tasks protocol handler (decorator or direct call)."""
        self._task_handler = fn
        return fn

    # ── Build FastAPI app ──────────────────────────────────────────────────────

    def build_app(self) -> Any:
        """构建并返回 FastAPI app。不启动 server。"""
        if not _FASTAPI_AVAILABLE:
            raise RuntimeError(
                "novie-agent-sdk requires FastAPI. "
                "Install with: pip install 'novie-agent-sdk[server]'"
            )

        m = self._manifest
        if (
            m.protocol_mode == "tasks"
            and os.getenv("NOVIE_ENV", "").lower() == "production"
            and isinstance(self._store, InMemoryTaskStore)
        ):
            raise RuntimeError(
                "tasks mode requires a durable TaskStore in production; "
                "configure Agent(..., task_store=SqliteTaskStore(...) or a custom durable store)"
            )

        @asynccontextmanager
        async def _lifespan(app: FastAPI):
            if self._registration_client is not None:
                await self._registration_client.register()
                await self._registration_client.start_heartbeat()
            yield
            if self._registration_client is not None:
                await self._registration_client.stop_heartbeat()
                await self._registration_client.deregister()

        app = FastAPI(
            title=m.name,
            version=m.version,
            description=f"Novie A2A agent: {m.agent_id}",
            lifespan=_lifespan,
        )

        @app.get("/healthz")
        async def healthz():
            durable_store = not isinstance(self._store, InMemoryTaskStore)
            return {
                "status": "ok",
                "agent_id": m.agent_id,
                "version": m.version,
                "protocol_mode": m.protocol_mode,
                "supports_cancel": m.execution.supports_cancel,
                "supports_streaming": m.supports_streaming,
                "task_store_backend": self._store.__class__.__name__,
                "durable": durable_store,
                "durability": m.execution.durability,
                "invocation_store_backend": self._invocation_store.__class__.__name__,
            }

        @app.get("/.well-known/agent.json")
        async def agent_json():
            return JSONResponse(m.to_dict())

        # simple mode
        if m.protocol_mode == "simple" or self._invoke_handler is not None:
            @app.post("/invoke")
            async def invoke_endpoint(request: Request):
                if self._invoke_handler is None:
                    raise HTTPException(503, "No invoke handler registered")
                body = await _parse_json(request, HTTPException)
                hdrs = RequestHeaders.from_request(request.headers)
                _verify_agent_request_headers(hdrs)
                started_invocation = False
                if hdrs.idempotency_key:
                    started_invocation, invocation = await self._invocation_store.start_or_get(
                        hdrs.idempotency_key,
                        "invoke",
                    )
                    if not started_invocation:
                        if invocation.status == "completed" and invocation.response is not None:
                            return invocation.response
                        return _duplicate_one_shot_response(invocation)
                ctx = InvokeContext(
                    input=_extract_inputs(body),
                    headers=hdrs,
                    agent_manifest=m,
                    observability=self._observability.scoped(
                        session_id=hdrs.session_id,
                        step_id=hdrs.step_id,
                        trace_id=hdrs.trace_id,
                    ),
                )
                try:
                    result = await self._invoke_handler(ctx)
                except Exception as exc:
                    if started_invocation and hdrs.idempotency_key:
                        await self._invocation_store.fail(
                            hdrs.idempotency_key,
                            "invoke",
                            str(exc),
                        )
                    raise
                response = _coerce_invoke_response(result)
                if started_invocation and hdrs.idempotency_key:
                    await self._invocation_store.complete(
                        hdrs.idempotency_key,
                        "invoke",
                        response=response,
                    )
                return response

        # stream mode
        if m.protocol_mode == "stream" or self._stream_handler is not None:
            @app.post("/stream")
            async def stream_endpoint(request: Request):
                if self._stream_handler is None:
                    raise HTTPException(503, "No stream handler registered")
                body = await _parse_json(request, HTTPException)
                hdrs = RequestHeaders.from_request(request.headers)
                _verify_agent_request_headers(hdrs)
                started_invocation = False
                if hdrs.idempotency_key:
                    started_invocation, invocation = await self._invocation_store.start_or_get(
                        hdrs.idempotency_key,
                        "stream",
                    )
                    if not started_invocation:
                        if invocation.status == "completed" and invocation.events is not None:
                            async def _replay() -> AsyncIterator[bytes]:
                                for event in invocation.events or []:
                                    yield (json.dumps(event) + "\n").encode()

                            return StreamingResponse(
                                _replay(),
                                media_type="application/x-ndjson",
                            )
                        return _duplicate_one_shot_response(invocation)

                # Observability events (e.g. token_usage from
                # ``observability.report_llm_usage``) need to land in the same
                # NDJSON stream as handler events so the platform's
                # ``_call_stream`` can mirror them to SessionTimeline + write
                # ``UsageRecord`` (parity with ``tasks`` mode). 用一个 asyncio
                # queue 把 handler 与 langchain-callback 两路事件 multiplex 到
                # 单一 stream。
                event_queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()

                async def _emit_observability_event(event: dict[str, Any]) -> None:
                    await event_queue.put(("obs", event))

                ctx = StreamContext(
                    input=_extract_inputs(body),
                    headers=hdrs,
                    agent_manifest=m,
                    observability=self._observability.scoped(
                        session_id=hdrs.session_id,
                        step_id=hdrs.step_id,
                        trace_id=hdrs.trace_id,
                        task_event_emitter=_emit_observability_event,
                    ),
                )

                async def _gen() -> AsyncIterator[bytes]:
                    emitted_events: list[dict[str, Any]] = []

                    async def _runner() -> None:
                        try:
                            async for event in self._stream_handler(ctx):
                                await event_queue.put(("handler", event))
                        except Exception as exc:  # noqa: BLE001
                            await event_queue.put(("error", exc))
                        finally:
                            await event_queue.put(("done", None))

                    runner_task = asyncio.create_task(_runner())
                    try:
                        while True:
                            kind, payload = await event_queue.get()
                            if kind == "done":
                                break
                            if kind == "error":
                                if started_invocation and hdrs.idempotency_key:
                                    await self._invocation_store.fail(
                                        hdrs.idempotency_key,
                                        "stream",
                                        str(payload),
                                    )
                                raise payload  # type: ignore[misc]
                            if kind == "obs":
                                # Already a fully-formed dict (UsageReport.
                                # to_platform_task_event); forward verbatim.
                                emitted_events.append(payload)
                                yield (json.dumps(payload) + "\n").encode()
                                continue
                            # handler event
                            if isinstance(payload, dict):
                                event = payload
                            else:
                                event = {"kind": "content", "text": str(payload)}
                            emitted_events.append(event)
                            yield (json.dumps(event) + "\n").encode()
                        done_event = {"kind": "done", "output": {}}
                        emitted_events.append(done_event)
                        if started_invocation and hdrs.idempotency_key:
                            await self._invocation_store.complete(
                                hdrs.idempotency_key,
                                "stream",
                                events=emitted_events,
                            )
                        yield (json.dumps(done_event) + "\n").encode()
                    finally:
                        if not runner_task.done():
                            runner_task.cancel()
                        try:
                            await runner_task
                        except (asyncio.CancelledError, Exception):
                            pass

                return StreamingResponse(_gen(), media_type="application/x-ndjson")

        # tasks mode
        if m.protocol_mode == "tasks" or self._task_handler is not None:
            @app.post("/tasks", status_code=202)
            async def create_task(request: Request, background: BackgroundTasks):
                if self._task_handler is None:
                    raise HTTPException(503, "No task handler registered")
                body = await _parse_json(request, HTTPException)
                hdrs = RequestHeaders.from_request(request.headers)
                _verify_agent_request_headers(hdrs)
                task_id = f"task-{uuid.uuid4().hex}"
                record = await self._store.create_task(
                    task_id,
                    _extract_inputs(body),
                    idempotency_key=hdrs.idempotency_key,
                )
                if record.task_id != task_id:
                    # Idempotent hit
                    return {"task_id": record.task_id, "status": record.status}

                cancel_event = asyncio.Event()
                self._cancel_events[task_id] = cancel_event
                async def _emit_usage_event(event: dict[str, Any]) -> None:
                    await self._store.append_event(task_id, event)

                task_observability = self._observability.scoped(
                    session_id=hdrs.session_id,
                    step_id=hdrs.step_id,
                    trace_id=hdrs.trace_id,
                    task_id=record.task_id,
                    task_event_emitter=_emit_usage_event,
                )
                ctx = TaskContext(
                    task_id=task_id,
                    input=record.input,
                    headers=hdrs,
                    agent_manifest=m,
                    observability=task_observability,
                    _store=self._store,
                    _cancelled=cancel_event,
                )
                background.add_task(self._run_task, ctx, task_id)
                return {"task_id": task_id, "status": "pending"}

            @app.get("/tasks/{task_id}")
            async def get_task_status(task_id: str, request: Request):
                _verify_agent_request_headers(RequestHeaders.from_request(request.headers))
                record = await self._store.get_task(task_id)
                if record is None:
                    raise HTTPException(404, f"Task {task_id!r} not found")
                return {
                    "task_id": record.task_id,
                    "status": record.status,
                    "created_at": record.created_at,
                    "updated_at": record.updated_at,
                    "error": record.error,
                }

            @app.get("/tasks/{task_id}/events")
            async def get_task_events(task_id: str, request: Request):
                _verify_agent_request_headers(RequestHeaders.from_request(request.headers))
                record = await self._store.get_task(task_id)
                if record is None:
                    raise HTTPException(404, f"Task {task_id!r} not found")
                events = await self._store.get_events(task_id)
                return {"task_id": task_id, "events": events}

            @app.get("/tasks/{task_id}/result")
            async def get_task_result(task_id: str, request: Request):
                _verify_agent_request_headers(RequestHeaders.from_request(request.headers))
                record = await self._store.get_task(task_id)
                if record is None:
                    raise HTTPException(404, f"Task {task_id!r} not found")
                if record.status not in TERMINAL_STATUSES:
                    raise HTTPException(
                        409,
                        detail={
                            "error": "task_not_terminal",
                            "task_id": task_id,
                            "status": record.status,
                        },
                    )
                if record.result is None:
                    raise HTTPException(
                        422,
                        detail={"error": "no_result", "task_id": task_id, "status": record.status},
                    )
                return {"task_id": task_id, "status": record.status, "output": record.result}

            if m.execution.supports_cancel:
                @app.post("/tasks/{task_id}/cancel", status_code=202)
                async def cancel_task(task_id: str, request: Request):
                    _verify_agent_request_headers(RequestHeaders.from_request(request.headers))
                    record = await self._store.get_task(task_id)
                    if record is None:
                        raise HTTPException(404, f"Task {task_id!r} not found")
                    if record.status in TERMINAL_STATUSES:
                        raise HTTPException(
                            409,
                            detail={
                                "error": "task_already_terminal",
                                "task_id": task_id,
                                "status": record.status,
                            },
                        )
                    cancel_ev = self._cancel_events.get(task_id)
                    if cancel_ev:
                        cancel_ev.set()
                    await self._store.cancel_task(task_id)
                    return {"task_id": task_id, "status": "cancelled"}

        return app

    async def _run_task(self, ctx: TaskContext, task_id: str) -> None:
        """在 background task 中运行 task handler。"""
        await self._store.update_task_status(task_id, "running")
        await ctx.emit_event("status_changed", {"status": "running"})
        try:
            result = await self._task_handler(ctx)  # type: ignore[misc]
            if ctx.is_cancelled:
                return
            if not isinstance(result, dict):
                result = {"result": result}
            await self._store.set_task_result(task_id, result, status="completed")
        except asyncio.CancelledError:
            await self._store.cancel_task(task_id)
        except Exception as exc:
            _log.exception("Task handler raised exception task_id=%s", task_id)
            await self._store.set_task_error(task_id, str(exc))
        finally:
            self._cancel_events.pop(task_id, None)

    # ── Serve ──────────────────────────────────────────────────────────────────

    def configure_registration(
        self,
        platform_url: str | None = None,
        heartbeat_interval: float = 30.0,
        *,
        required: bool | None = None,
        registration_token: str | None = None,
        register_max_attempts: int | None = None,
        register_backoff_seconds: float | None = None,
    ) -> "Agent":
        """配置向 Platform 自动注册（启动时 register，运行时 heartbeat，关闭时 deregister）。"""
        url = platform_url or os.environ.get("NOVIE_PLATFORM_BASE_URL", "")
        if url:
            if required is None:
                required_raw = os.environ.get("NOVIE_AGENT_REGISTRATION_REQUIRED", "").strip().lower()
                required = required_raw in {"1", "true", "yes", "on"}
                if not required:
                    runtime_mode = os.getenv("NOVIE_RUNTIME_MODE", "").strip().lower()
                    runtime_env = os.getenv("NOVIE_ENV", "").strip().lower()
                    required = runtime_mode == "production" or runtime_env == "production"
            token = registration_token
            if token is None:
                token = (
                    os.getenv("NOVIE_AGENT_REGISTRATION_TOKEN", "").strip()
                    or os.getenv("NOVIE_AGENT_SECRET", "").strip()
                )
            if register_max_attempts is None:
                register_max_attempts = int(
                    os.getenv("NOVIE_AGENT_REGISTER_MAX_ATTEMPTS", "5")
                )
            if register_backoff_seconds is None:
                register_backoff_seconds = float(
                    os.getenv("NOVIE_AGENT_REGISTER_BACKOFF_SECONDS", "2")
                )
            self._registration_client = RegistrationClient(
                url,
                self._manifest,
                heartbeat_interval,
                required=bool(required),
                registration_token=token,
                register_max_attempts=register_max_attempts,
                register_backoff_seconds=register_backoff_seconds,
            )
        return self

    def serve(
        self,
        host: str = "0.0.0.0",
        port: int | None = None,
        **uvicorn_kwargs: Any,
    ) -> None:
        """启动 uvicorn server。阻塞直到 SIGINT/SIGTERM。"""
        import uvicorn

        if port is None:
            port = int(os.environ.get("PORT", "8000"))

        app = self.build_app()
        uvicorn.run(app, host=host, port=port, **uvicorn_kwargs)


# ── Internal helpers ───────────────────────────────────────────────────────────

async def _parse_json(request: Any, http_exc: type) -> dict[str, Any]:
    raw = await request.body()
    if not raw:
        raise http_exc(status_code=400, detail="empty request body")
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise http_exc(status_code=400, detail=f"invalid JSON: {exc.msg}") from exc
    if not isinstance(parsed, dict):
        raise http_exc(status_code=400, detail="request body must be a JSON object")
    return parsed


def _extract_inputs(body: dict[str, Any]) -> dict[str, Any]:
    """Pull the agent's input payload out of the A2A wire body.

    TechDebt #7 (2026-05-11) — the canonical wire shape (matched by the
    platform's ``_build_agent_payload`` and the Rust SDK) is::

        {
          "context": {...},
          "inputs": {...},          # ← canonical key
          "runtime_context_snapshot_ref": "...",
          "capability_grants": [...],
          "credential_leases": {...}
        }

    Older agents may still receive ``"input"`` (singular) from
    legacy callers — accept both with ``inputs`` winning. Falling
    back to the whole body (the pre-fix behaviour) leaked
    ``context`` / ``capability_grants`` / ``credential_leases``
    into the agent's ``input``, so we no longer do that — empty
    dict is the safe default for malformed payloads.

    The return value is always a dict; non-dict ``inputs`` values
    coerce to ``{}`` so the agent handler never crashes on a
    type-mismatched payload.
    """
    candidate = body.get("inputs")
    if isinstance(candidate, dict):
        return candidate
    legacy = body.get("input")
    if isinstance(legacy, dict):
        return legacy
    return {}
