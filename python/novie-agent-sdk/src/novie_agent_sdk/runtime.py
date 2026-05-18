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
  POST /tasks/{task_id}/asks/{gate_id}/resolve
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
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
from typing import Protocol

_log = logging.getLogger(__name__)
_SAFE_AGENT_ID_CHARS = re.compile(r"[^A-Za-z0-9_.-]+")
_MidRunAskTimeoutAction = Literal["skip", "auto_recommended", "fail_implementation"]

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
from novie_protocol.contracts.decision_gate import (
    DecisionGateEnvelope,
    DecisionGateOption,
)

from .observability import AgentObservability, ObservabilitySink, build_default_sinks
from .tenant_scoping import validate_tenant_context


def _build_ctx_platform_and_llm(
    hdrs: "RequestHeaders", agent_id: str
) -> tuple[Any, Any]:
    """Build (platform_ns, llm_facade) for an incoming request.

    Returns (None, None) on import error so the context is still
    created; the properties raise RuntimeError on access if None.
    """
    try:
        from .platform_namespace import build_platform_namespace
        from .llm_facade import build_llm_facade

        platform_ns = build_platform_namespace(hdrs, agent_id=agent_id)
        llm = build_llm_facade(platform_ns, agent_id=agent_id)
        return platform_ns, llm
    except Exception as exc:  # noqa: BLE001 — best-effort
        _log.debug("ctx platform/llm build failed: %s", exc)
        return None, None


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


class AskTimedOut(TimeoutError):
    """Raised when ``ctx.ask`` times out with a fail action."""

    def __init__(self, gate_id: str, *, default_action: str) -> None:
        super().__init__(
            f"mid-run ask {gate_id!r} timed out with default_action={default_action!r}"
        )
        self.gate_id = gate_id
        self.default_action = default_action


class AskBudgetExceeded(RuntimeError):
    """Raised when a task exceeds its configured mid-run ask budget."""

    def __init__(self, *, cap: int) -> None:
        super().__init__(f"mid-run ask budget exceeded: max_mid_run_asks={cap}")
        self.cap = cap


@dataclass
class InvokeContext:
    """simple protocol handler 上下文。"""
    input: dict[str, Any]
    headers: RequestHeaders
    agent_manifest: AgentManifestV2
    observability: AgentObservability
    # Injected by the SDK runtime; None when build failed (e.g. missing dep).
    _platform: Any = field(default=None, repr=False)
    _llm: Any = field(default=None, repr=False)

    @property
    def brief(self) -> dict[str, Any]:
        return self.input.get("brief", {})

    @property
    def platform(self) -> Any:
        """Live ``PlatformNamespace`` or ``_UnavailablePlatformNamespace``."""
        if self._platform is None:
            raise RuntimeError(
                "ctx.platform is not available — the SDK runtime failed to build "
                "the platform namespace for this request."
            )
        return self._platform

    @property
    def llm(self) -> Any:
        """``LlmFacade`` for this request (platform or BYOK)."""
        if self._llm is None:
            raise RuntimeError(
                "ctx.llm is not available — the SDK runtime failed to build "
                "the LLM facade for this request."
            )
        return self._llm


@dataclass
class StreamContext:
    """stream protocol handler 上下文。"""
    input: dict[str, Any]
    headers: RequestHeaders
    agent_manifest: AgentManifestV2
    observability: AgentObservability
    _platform: Any = field(default=None, repr=False)
    _llm: Any = field(default=None, repr=False)

    @property
    def brief(self) -> dict[str, Any]:
        return self.input.get("brief", {})

    @property
    def platform(self) -> Any:
        """Live ``PlatformNamespace`` or ``_UnavailablePlatformNamespace``."""
        if self._platform is None:
            raise RuntimeError(
                "ctx.platform is not available — the SDK runtime failed to build "
                "the platform namespace for this request."
            )
        return self._platform

    @property
    def llm(self) -> Any:
        """``LlmFacade`` for this request (platform or BYOK)."""
        if self._llm is None:
            raise RuntimeError(
                "ctx.llm is not available — the SDK runtime failed to build "
                "the LLM facade for this request."
            )
        return self._llm


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
    _platform: Any = field(default=None, repr=False)
    _llm: Any = field(default=None, repr=False)
    _mid_run_ask_count: int = 0

    @property
    def brief(self) -> dict[str, Any]:
        return self.input.get("brief", {})

    @property
    def platform(self) -> Any:
        """Live ``PlatformNamespace`` or ``_UnavailablePlatformNamespace``."""
        if self._platform is None:
            raise RuntimeError(
                "ctx.platform is not available — the SDK runtime failed to build "
                "the platform namespace for this request."
            )
        return self._platform

    @property
    def llm(self) -> Any:
        """``LlmFacade`` for this request (platform or BYOK)."""
        if self._llm is None:
            raise RuntimeError(
                "ctx.llm is not available — the SDK runtime failed to build "
                "the LLM facade for this request."
            )
        return self._llm

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

    async def heartbeat(
        self,
        *,
        phase: str = "",
        message: str = "",
        interval_seconds: float | None = None,
    ) -> None:
        payload: dict[str, Any] = {
            "agent_event_kind": "heartbeat",
        }
        if phase:
            payload["phase"] = phase
        if message:
            payload["message"] = message
        if interval_seconds is not None:
            payload["interval_seconds"] = interval_seconds
        await self.emit_event("heartbeat", payload, summary=message or phase or "heartbeat")

    async def ask(
        self,
        question: str,
        *,
        timeout: float | None = None,
        default_action: str = "skip",
        options: tuple[DecisionGateOption, ...] | list[DecisionGateOption] = (),
        recommended_option: str | None = None,
        gate_type: str = "missing_inputs",
        title: str = "Need input",
        allow_freeform: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Raise a mid-run user question and resume via timeout fallback.

        This is the SDK-side ADR-017 primitive. Until platform resume wiring
        can feed a human answer back into the running task, the local contract
        is timeout-driven: emit the decision gate, wait ``timeout`` seconds,
        then apply ``default_action``.
        """
        timeout_action = _normalize_ask_default_action(default_action)
        cap = _max_mid_run_asks(self.input)
        if self._mid_run_ask_count >= cap:
            await self.emit_event(
                "mid_run_ask_rejected",
                {
                    "agent_event_kind": "mid_run_ask_rejected",
                    "reason_code": "max_mid_run_asks_exceeded",
                    "max_mid_run_asks": cap,
                    "current_count": self._mid_run_ask_count,
                    "question": question,
                },
                summary=f"mid-run ask budget exceeded ({cap})",
            )
            raise AskBudgetExceeded(cap=cap)
        self._mid_run_ask_count += 1
        gate_id = f"ask-{uuid.uuid4().hex[:20]}"
        envelope = DecisionGateEnvelope(
            gate_id=gate_id,
            gate_type=gate_type,
            title=title,
            question=question,
            options=tuple(options),
            recommended_option=recommended_option,
            allow_freeform=allow_freeform,
            raised_by_agent_id=self.agent_manifest.agent_id,
            agent_metadata=dict(metadata or {}),
        )
        envelope_payload = asdict(envelope)
        envelope_payload["timeout_seconds"] = timeout
        envelope_payload["default_action_on_timeout"] = timeout_action
        await self.emit_event(
            "mid_run_ask",
            {
                "agent_event_kind": "mid_run_ask",
                "gate_id": gate_id,
                "gate_class": "decision_gate",
                "question": question,
                "timeout_seconds": timeout,
                "default_action_on_timeout": timeout_action,
                "envelope": envelope_payload,
            },
            summary=question[:200],
        )
        if hasattr(self._platform, "set_mid_run_ask_active"):
            self._platform.set_mid_run_ask_active(True)
        await self.set_status("waiting_for_human")
        wait_task = asyncio.create_task(
            self._store.wait_for_ask_resolution(self.task_id, gate_id, timeout)
        )
        cancel_task = asyncio.create_task(self._cancelled.wait())
        try:
            done, pending = await asyncio.wait(
                {wait_task, cancel_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in pending:
                task.cancel()
            if cancel_task in done and self.is_cancelled:
                raise asyncio.CancelledError()
            resolved = await wait_task if wait_task in done else None
        finally:
            if hasattr(self._platform, "set_mid_run_ask_active"):
                self._platform.set_mid_run_ask_active(False)
            for task in (wait_task, cancel_task):
                if not task.done():
                    task.cancel()

        await self.set_status("running")
        if resolved is not None:
            resolved_payload = {
                "gate_id": gate_id,
                "timed_out": False,
                **dict(resolved),
            }
            await self.emit_event(
                "mid_run_ask_resumed",
                {
                    "agent_event_kind": "mid_run_ask_resumed",
                    **resolved_payload,
                },
                summary=f"mid-run ask resolved: {gate_id}",
            )
            return resolved_payload

        resolution = _timeout_resolution(
            gate_id,
            timeout_action,
            recommended_option=recommended_option,
        )
        await self.emit_event(
            "mid_run_ask_timeout",
            {
                "agent_event_kind": "mid_run_ask_timeout",
                **resolution,
            },
            summary=f"mid-run ask timed out: {gate_id}",
        )
        if timeout_action == "fail_implementation":
            raise AskTimedOut(gate_id, default_action=timeout_action)
        return resolution

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


def _normalize_ask_default_action(value: str) -> _MidRunAskTimeoutAction:
    action = (value or "").strip().lower()
    if action == "continue":
        action = "skip"
    if action not in {"skip", "auto_recommended", "fail_implementation"}:
        raise ValueError(
            "ctx.ask default_action must be one of: "
            "skip, continue, auto_recommended, fail_implementation"
        )
    return action  # type: ignore[return-value]


def _timeout_resolution(
    gate_id: str,
    default_action: _MidRunAskTimeoutAction,
    *,
    recommended_option: str | None,
) -> dict[str, Any]:
    resolution_type = "auto_policy" if default_action == "auto_recommended" else "skipped"
    return {
        "gate_id": gate_id,
        "resolution_type": resolution_type,
        "selected_option_id": (
            recommended_option if default_action == "auto_recommended" else None
        ),
        "freeform_answer": "",
        "timed_out": True,
        "default_action": default_action,
    }


def _max_mid_run_asks(inputs: dict[str, Any]) -> int:
    lifecycle = inputs.get("lifecycle")
    value = lifecycle.get("max_mid_run_asks") if isinstance(lifecycle, dict) else None
    if value in (None, ""):
        value = os.getenv("NOVIE_SDK_MAX_MID_RUN_ASKS", "3")
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = 3
    return max(0, parsed)


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


# ── One-shot invocation store: lease + heartbeat defense ──────────────────────
#
# The /stream and /invoke endpoints register an idempotency record in
# ``in_progress`` and only transition it on a terminal event. If the platform
# disconnects mid-stream — or the agent process dies between ``start_or_get``
# and the terminal call — the record is stuck at ``in_progress``. Subsequent
# retries with the same Idempotency-Key would deadlock on 409
# ``retry_in_progress`` forever. Two defenses:
#
#   1. Tier A — stream/invoke handlers wrap their work in try/finally; if
#      neither ``complete`` nor ``fail`` was called, the finally calls
#      ``fail("...aborted_before_terminal")`` so the record reaches a terminal
#      state even on CancelledError / GeneratorExit / client disconnect.
#   2. Tier B — every ``in_progress`` record carries a lease
#      (``lease_seconds``). ``start_or_get`` checks ``_is_lease_stale``; if a
#      record is ``in_progress`` past its lease, it's recycled into a fresh
#      record so a new request can proceed (covers the agent-process-crash
#      case where Tier A's finally never ran). The stream handler renews the
#      lease via ``touch`` every ``NOVIE_AGENT_INVOCATION_HEARTBEAT_EVENTS``.
_DEFAULT_INVOCATION_LEASE_SECONDS = _env_int(
    "NOVIE_AGENT_INVOCATION_LEASE_SECONDS", default=300,
)
_DEFAULT_INVOCATION_HEARTBEAT_EVENTS = _env_int(
    "NOVIE_AGENT_INVOCATION_HEARTBEAT_EVENTS", default=10,
)


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
    lease_seconds: int = field(default_factory=lambda: _DEFAULT_INVOCATION_LEASE_SECONDS)


def _is_lease_stale(record: OneShotInvocationRecord) -> bool:
    """True if record is ``in_progress`` and its lease has lapsed."""
    if record.status != "in_progress":
        return False
    try:
        updated = datetime.fromisoformat(record.updated_at)
    except ValueError:
        return False
    if updated.tzinfo is None:
        updated = updated.replace(tzinfo=timezone.utc)
    age_seconds = (datetime.now(timezone.utc) - updated).total_seconds()
    return age_seconds > max(record.lease_seconds, 1)


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

    async def touch(self, idempotency_key: str, mode: str) -> None:
        """Renew the lease on an ``in_progress`` record. No-op if record is
        absent or already in a terminal state. Adopters can implement this as
        a no-op if they don't need lease-based recovery."""
        ...


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
                if _is_lease_stale(existing):
                    # Tier B: previous handler died without reaching a terminal
                    # state and its lease has elapsed. Mark the stale record
                    # ``expired`` (for any debug surface that still reads it)
                    # and let a fresh invocation proceed.
                    _log.warning(
                        "recycling stale in_progress invocation record "
                        "mode=%s idempotency_key=%s updated_at=%s lease=%ds",
                        mode,
                        idempotency_key,
                        existing.updated_at,
                        existing.lease_seconds,
                    )
                    existing.status = "expired"
                    existing.error = "lease_expired_no_activity"
                    existing.updated_at = _now_iso()
                else:
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

    async def touch(self, idempotency_key: str, mode: str) -> None:
        """Renew the lease on an ``in_progress`` record. Silent no-op for
        absent records or those already in a terminal state."""
        async with self._lock:
            record = self._records.get((mode, idempotency_key))
            if record is None or record.status != "in_progress":
                return
            record.updated_at = _now_iso()


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
                existing = self._row_to_record(row)
                if _is_lease_stale(existing):
                    # Tier B: stale ``in_progress`` row — agent died between
                    # ``start_or_get`` and the terminal call. Overwrite the
                    # row in-place with a fresh ``in_progress`` so a new
                    # request can proceed.
                    _log.warning(
                        "recycling stale in_progress invocation row "
                        "mode=%s idempotency_key=%s updated_at=%s lease=%ds",
                        mode,
                        idempotency_key,
                        existing.updated_at,
                        existing.lease_seconds,
                    )
                    now = _now_iso()
                    conn.execute(
                        """
                        UPDATE sdk_one_shot_invocations
                        SET status = 'in_progress',
                            created_at = ?,
                            updated_at = ?,
                            response_json = NULL,
                            events_json = NULL,
                            error = NULL
                        WHERE mode = ? AND idempotency_key = ?
                        """,
                        (now, now, mode, idempotency_key),
                    )
                    conn.commit()
                    return True, OneShotInvocationRecord(
                        idempotency_key=idempotency_key,
                        mode=mode,
                        status="in_progress",
                        created_at=now,
                        updated_at=now,
                    )
                return False, existing
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

    async def touch(self, idempotency_key: str, mode: str) -> None:
        await asyncio.to_thread(self._touch_sync, idempotency_key, mode)

    def _touch_sync(self, idempotency_key: str, mode: str) -> None:
        with self._lock, sqlite3.connect(self._db_path) as conn:
            conn.execute(
                """
                UPDATE sdk_one_shot_invocations
                SET updated_at = ?
                WHERE mode = ? AND idempotency_key = ?
                  AND status = 'in_progress'
                """,
                (_now_iso(), mode, idempotency_key),
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
    async def resolve_ask(
        self,
        task_id: str,
        gate_id: str,
        resolution: dict[str, Any],
    ) -> bool: ...
    async def wait_for_ask_resolution(
        self,
        task_id: str,
        gate_id: str,
        timeout: float | None,
    ) -> dict[str, Any] | None: ...


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
        self._ask_resolutions: dict[tuple[str, str], dict[str, Any]] = {}
        self._lock = asyncio.Lock()
        self._status_waiters: dict[str, asyncio.Condition] = {}
        self._ask_waiters: dict[tuple[str, str], asyncio.Condition] = {}
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
            for key in tuple(self._ask_resolutions):
                if key[0] == task_id:
                    self._ask_resolutions.pop(key, None)
                    self._ask_waiters.pop(key, None)

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
        for key in tuple(self._ask_resolutions):
            if key[0] == oldest:
                self._ask_resolutions.pop(key, None)
                self._ask_waiters.pop(key, None)

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

    async def resolve_ask(
        self,
        task_id: str,
        gate_id: str,
        resolution: dict[str, Any],
    ) -> bool:
        key = (task_id, gate_id)
        async with self._lock:
            if task_id not in self._tasks:
                return False
            payload = {
                "gate_id": gate_id,
                **dict(resolution),
                "resolved_at": _now_iso(),
            }
            self._ask_resolutions[key] = payload
            cond = self._ask_waiters.setdefault(key, asyncio.Condition())
        async with cond:
            cond.notify_all()
        await self.append_event(
            task_id,
            _make_event(
                task_id,
                "mid_run_ask_resolved",
                {
                    "agent_event_kind": "mid_run_ask_resolved",
                    **payload,
                },
                f"mid-run ask resolved: {gate_id}",
            ),
        )
        return True

    async def wait_for_ask_resolution(
        self,
        task_id: str,
        gate_id: str,
        timeout: float | None,
    ) -> dict[str, Any] | None:
        key = (task_id, gate_id)
        async with self._lock:
            existing = self._ask_resolutions.get(key)
            if existing is not None:
                return dict(existing)
            cond = self._ask_waiters.setdefault(key, asyncio.Condition())

        async def _wait() -> dict[str, Any] | None:
            while True:
                async with self._lock:
                    resolved = self._ask_resolutions.get(key)
                    if resolved is not None:
                        return dict(resolved)
                    task = self._tasks.get(task_id)
                    if task is None or task.status in TERMINAL_STATUSES:
                        return None
                async with cond:
                    await cond.wait()

        if timeout is None:
            return await _wait()
        try:
            return await asyncio.wait_for(_wait(), timeout=max(0.0, float(timeout)))
        except TimeoutError:
            return None


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

    async def resolve_ask(
        self,
        task_id: str,
        gate_id: str,
        resolution: dict[str, Any],
    ) -> bool:
        resolved = await asyncio.to_thread(
            self._resolve_ask_sync,
            task_id,
            gate_id,
            dict(resolution),
        )
        if resolved:
            await self.append_event(
                task_id,
                _make_event(
                    task_id,
                    "mid_run_ask_resolved",
                    {
                        "agent_event_kind": "mid_run_ask_resolved",
                        "gate_id": gate_id,
                        **dict(resolution),
                    },
                    f"mid-run ask resolved: {gate_id}",
                ),
            )
        return resolved

    def _resolve_ask_sync(
        self,
        task_id: str,
        gate_id: str,
        resolution: dict[str, Any],
    ) -> bool:
        with self._lock, sqlite3.connect(self._db_path) as conn:
            row = conn.execute(
                "SELECT task_id FROM sdk_tasks WHERE task_id = ?",
                (task_id,),
            ).fetchone()
            if row is None:
                return False
            payload = {
                "gate_id": gate_id,
                **dict(resolution),
                "resolved_at": _now_iso(),
            }
            conn.execute(
                """
                INSERT INTO sdk_task_ask_resolutions (
                    task_id, gate_id, resolution_json, resolved_at
                ) VALUES (?, ?, ?, ?)
                ON CONFLICT(task_id, gate_id)
                DO UPDATE SET
                    resolution_json = excluded.resolution_json,
                    resolved_at = excluded.resolved_at
                """,
                (task_id, gate_id, json.dumps(payload), payload["resolved_at"]),
            )
            conn.commit()
            return True

    async def wait_for_ask_resolution(
        self,
        task_id: str,
        gate_id: str,
        timeout: float | None,
    ) -> dict[str, Any] | None:
        deadline = None if timeout is None else time.monotonic() + max(0.0, float(timeout))
        while True:
            resolution = await asyncio.to_thread(
                self._get_ask_resolution_sync,
                task_id,
                gate_id,
            )
            if resolution is not None:
                return resolution
            task = await self.get_task(task_id)
            if task is None or task.status in TERMINAL_STATUSES:
                return None
            if deadline is not None and time.monotonic() >= deadline:
                return None
            await asyncio.sleep(0.05)

    def _get_ask_resolution_sync(
        self,
        task_id: str,
        gate_id: str,
    ) -> dict[str, Any] | None:
        with sqlite3.connect(self._db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                """
                SELECT resolution_json FROM sdk_task_ask_resolutions
                WHERE task_id = ? AND gate_id = ?
                """,
                (task_id, gate_id),
            ).fetchone()
        if row is None:
            return None
        return json.loads(str(row["resolution_json"]))

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
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sdk_task_ask_resolutions (
                    task_id TEXT NOT NULL,
                    gate_id TEXT NOT NULL,
                    resolution_json TEXT NOT NULL,
                    resolved_at TEXT NOT NULL,
                    PRIMARY KEY (task_id, gate_id)
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
                    if resp.status_code == 404:
                        _log.info(
                            "heartbeat returned 404; re-registering with Platform agent_id=%s",
                            self._manifest.agent_id,
                        )
                        await self.register()
                        continue
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
                validate_tenant_context(hdrs)
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
                    **dict(zip(("_platform", "_llm"), _build_ctx_platform_and_llm(hdrs, m.agent_id))),
                )
                invocation_resolved = False
                try:
                    try:
                        result = await self._invoke_handler(ctx)
                    except Exception as exc:
                        if started_invocation and hdrs.idempotency_key:
                            await self._invocation_store.fail(
                                hdrs.idempotency_key,
                                "invoke",
                                str(exc),
                            )
                            invocation_resolved = True
                        raise
                    response = _coerce_invoke_response(result)
                    if started_invocation and hdrs.idempotency_key:
                        await self._invocation_store.complete(
                            hdrs.idempotency_key,
                            "invoke",
                            response=response,
                        )
                        invocation_resolved = True
                    return response
                finally:
                    # Tier A safety net: ``CancelledError`` / ``GeneratorExit``
                    # / other BaseException bypass the inner ``except Exception``
                    # handler. Without this, the idempotency record gets
                    # stranded at ``in_progress`` and subsequent retries hit
                    # 409 ``retry_in_progress`` forever.
                    if (
                        started_invocation
                        and hdrs.idempotency_key
                        and not invocation_resolved
                    ):
                        try:
                            await self._invocation_store.fail(
                                hdrs.idempotency_key,
                                "invoke",
                                "invoke_aborted_before_terminal",
                            )
                        except Exception:  # noqa: BLE001
                            _log.exception(
                                "invocation_store.fail in finally raised; "
                                "record may stay in_progress until lease expires "
                                "(idempotency_key=%s)",
                                hdrs.idempotency_key,
                            )

        # stream mode
        if m.protocol_mode == "stream" or self._stream_handler is not None:
            @app.post("/stream")
            async def stream_endpoint(request: Request):
                if self._stream_handler is None:
                    raise HTTPException(503, "No stream handler registered")
                body = await _parse_json(request, HTTPException)
                hdrs = RequestHeaders.from_request(request.headers)
                _verify_agent_request_headers(hdrs)
                validate_tenant_context(hdrs)
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
                    **dict(zip(("_platform", "_llm"), _build_ctx_platform_and_llm(hdrs, m.agent_id))),
                )

                async def _gen() -> AsyncIterator[bytes]:
                    emitted_events: list[dict[str, Any]] = []
                    terminal_error_emitted = False
                    invocation_resolved = False
                    events_since_touch = 0

                    async def _maybe_heartbeat() -> None:
                        # Tier B: renew the in_progress lease so a long-running
                        # handler isn't recycled by ``_is_lease_stale`` while
                        # it's still actively producing.
                        nonlocal events_since_touch
                        events_since_touch += 1
                        if events_since_touch < _DEFAULT_INVOCATION_HEARTBEAT_EVENTS:
                            return
                        events_since_touch = 0
                        if not (started_invocation and hdrs.idempotency_key):
                            return
                        try:
                            await self._invocation_store.touch(
                                hdrs.idempotency_key, "stream",
                            )
                        except Exception:  # noqa: BLE001
                            # Heartbeat failure is non-fatal; the lease will
                            # just expire normally if downstream really did die.
                            _log.debug("invocation_store.touch raised", exc_info=True)

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
                                terminal_error_emitted = True
                                error_event = {
                                    "kind": "terminal_error",
                                    "error": str(payload),
                                    "output": {},
                                    "metadata": {
                                        "terminal_source": "sdk_exception_guard",
                                        "agent_id": m.agent_id,
                                    },
                                }
                                emitted_events.append(error_event)
                                if started_invocation and hdrs.idempotency_key:
                                    await self._invocation_store.complete(
                                        hdrs.idempotency_key,
                                        "stream",
                                        events=emitted_events,
                                    )
                                    invocation_resolved = True
                                yield (json.dumps(error_event) + "\n").encode()
                                break
                            if kind == "obs":
                                # Already a fully-formed dict (UsageReport.
                                # to_platform_task_event); forward verbatim.
                                emitted_events.append(payload)
                                yield (json.dumps(payload) + "\n").encode()
                                await _maybe_heartbeat()
                                continue
                            # handler event
                            if isinstance(payload, dict):
                                event = payload
                            else:
                                event = {"kind": "content", "text": str(payload)}
                            emitted_events.append(event)
                            yield (json.dumps(event) + "\n").encode()
                            await _maybe_heartbeat()
                        if not terminal_error_emitted:
                            done_event = {
                                "kind": "done",
                                "output": {},
                                "metadata": {"terminal_source": "sdk_sentinel"},
                            }
                            emitted_events.append(done_event)
                            if started_invocation and hdrs.idempotency_key:
                                await self._invocation_store.complete(
                                    hdrs.idempotency_key,
                                    "stream",
                                    events=emitted_events,
                                )
                                invocation_resolved = True
                            yield (json.dumps(done_event) + "\n").encode()
                    finally:
                        if not runner_task.done():
                            runner_task.cancel()
                        try:
                            await runner_task
                        except (asyncio.CancelledError, Exception):
                            pass
                        # Tier A safety net: client disconnect /
                        # GeneratorExit / CancelledError cut the loop before
                        # the terminal complete() call. Without this, the
                        # record sits in ``in_progress`` forever and retries
                        # deadlock on 409 retry_in_progress.
                        if (
                            started_invocation
                            and hdrs.idempotency_key
                            and not invocation_resolved
                        ):
                            try:
                                await self._invocation_store.fail(
                                    hdrs.idempotency_key,
                                    "stream",
                                    "stream_aborted_before_terminal",
                                )
                            except Exception:  # noqa: BLE001
                                _log.exception(
                                    "invocation_store.fail in finally raised; "
                                    "record may stay in_progress until lease "
                                    "expires (idempotency_key=%s)",
                                    hdrs.idempotency_key,
                                )

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
                validate_tenant_context(hdrs)
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
                _platform, _llm = _build_ctx_platform_and_llm(hdrs, m.agent_id)
                ctx = TaskContext(
                    task_id=task_id,
                    input=record.input,
                    headers=hdrs,
                    agent_manifest=m,
                    observability=task_observability,
                    _store=self._store,
                    _cancelled=cancel_event,
                    _platform=_platform,
                    _llm=_llm,
                )
                background.add_task(self._run_task, ctx, task_id)
                return {"task_id": task_id, "status": "pending"}

            @app.get("/tasks/{task_id}")
            async def get_task_status(task_id: str, request: Request):
                _hdrs_local = RequestHeaders.from_request(request.headers)
                _verify_agent_request_headers(_hdrs_local)
                validate_tenant_context(_hdrs_local)
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
                _hdrs_local = RequestHeaders.from_request(request.headers)
                _verify_agent_request_headers(_hdrs_local)
                validate_tenant_context(_hdrs_local)
                record = await self._store.get_task(task_id)
                if record is None:
                    raise HTTPException(404, f"Task {task_id!r} not found")
                events = await self._store.get_events(task_id)
                return {"task_id": task_id, "events": events}

            @app.post("/tasks/{task_id}/asks/{gate_id}/resolve", status_code=202)
            async def resolve_task_ask(task_id: str, gate_id: str, request: Request):
                _hdrs_local = RequestHeaders.from_request(request.headers)
                _verify_agent_request_headers(_hdrs_local)
                validate_tenant_context(_hdrs_local)
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
                body = await _parse_json(request, HTTPException)
                resolution = body.get("resolution") if isinstance(body.get("resolution"), dict) else body
                accepted = await self._store.resolve_ask(
                    task_id,
                    gate_id,
                    dict(resolution),
                )
                if not accepted:
                    raise HTTPException(404, f"Task {task_id!r} not found")
                return {
                    "task_id": task_id,
                    "gate_id": gate_id,
                    "status": "accepted",
                }

            @app.get("/tasks/{task_id}/result")
            async def get_task_result(task_id: str, request: Request):
                _hdrs_local = RequestHeaders.from_request(request.headers)
                _verify_agent_request_headers(_hdrs_local)
                validate_tenant_context(_hdrs_local)
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
                    _hdrs_local = RequestHeaders.from_request(request.headers)
                    _verify_agent_request_headers(_hdrs_local)
                    validate_tenant_context(_hdrs_local)
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
