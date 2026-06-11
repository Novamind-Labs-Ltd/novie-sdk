"""Protocol ``PlatformServices`` adapters backed by platform capabilities."""
from __future__ import annotations

import asyncio
import json
import logging
import os
from collections.abc import AsyncIterator, Mapping
from datetime import UTC, datetime
from typing import Any
from urllib import error, request

from novie_protocol.contracts import (
    AuditEvent,
    ExecutionContext,
    ExternalAgentCheckpointRecord,
    QuotaDecision,
    UsageRecord,
    UsageSummary,
)
from novie_protocol.services import PlatformServices

from .platform_callback import build_platform_callback_headers, sign_platform_callback_headers
from .platform_namespace import PlatformNamespace, build_platform_namespace

_log = logging.getLogger(__name__)
_DEFAULT_TIMEOUT_SECONDS = 30.0


class PlatformGatewayClient:
    """Read-only gateway client plus capability invoke shim."""

    def __init__(
        self,
        base_url: str,
        forward_headers: Mapping[str, str],
        *,
        agent_id: str,
        timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._headers = dict(forward_headers)
        self._agent_id = agent_id
        self._timeout = timeout_seconds
        self._namespace = build_platform_namespace(
            self._headers,
            agent_id=agent_id,
            base_url=self._base_url,
            timeout_seconds=timeout_seconds,
        )

    async def get_json(self, path: str) -> dict[str, Any] | None:
        if not path.startswith("/"):
            path = "/" + path
        url = f"{self._base_url}{path}"
        req = request.Request(url, method="GET")
        headers = sign_platform_callback_headers(self._headers, method="GET", path=path)
        for key, value in headers.items():
            if value:
                req.add_header(key, value)

        def _do_request() -> dict[str, Any] | None:
            try:
                with request.urlopen(req, timeout=self._timeout) as resp:
                    raw = resp.read().decode("utf-8")
            except error.HTTPError as exc:
                detail = ""
                try:
                    detail = exc.read().decode("utf-8")[:300]
                except Exception:
                    pass
                _log.warning(
                    "gateway GET failed path=%s status=%s detail=%s",
                    path,
                    exc.code,
                    detail,
                )
                return None
            except error.URLError as exc:
                _log.warning("gateway GET transport error path=%s reason=%s", path, exc.reason)
                return None
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                _log.warning("gateway GET returned non-JSON path=%s", path)
                return None
            return parsed if isinstance(parsed, dict) else None

        return await asyncio.to_thread(_do_request)

    async def invoke(self, capability_id: str, arguments: dict[str, Any]) -> dict[str, Any] | None:
        diagnostics = await self._namespace.invoke_capability(capability_id, arguments)
        return diagnostics.result if diagnostics.ok and isinstance(diagnostics.result, dict) else None

    def last_diagnostics(self) -> tuple[Any, ...]:
        return self._namespace.last_diagnostics()


class HttpPlatformKnowledgeService:
    def __init__(self, namespace: PlatformNamespace) -> None:
        self._namespace = namespace

    async def search(
        self,
        ctx: ExecutionContext,
        query: str,
        top_k: int = 5,
        *,
        project_id: str | None = None,
    ) -> list[dict[str, Any]]:
        scope = (
            project_id
            if project_id is not None and str(project_id).strip()
            else (ctx.tenant.project_id or ctx.tenant.workspace_id)
        )
        return await self._namespace.knowledge.search(
            query,
            top_k=top_k,
            project_id=str(scope) if scope else None,
        )


class HttpExternalAgentCheckpointService:
    def __init__(self, namespace: PlatformNamespace) -> None:
        self._namespace = namespace

    async def put(
        self,
        ctx: ExecutionContext,
        *,
        owner_agent_id: str,
        thread_id: str,
        payload: dict[str, Any],
        checkpoint_id: str | None = None,
        session_id: str | None = None,
        workflow_id: str | None = None,
        step_id: str | None = None,
        checkpoint_format: str = "langgraph",
        checkpoint_version: str = "1",
        summary: str | None = None,
        parent_checkpoint_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ExternalAgentCheckpointRecord:
        record = await self._namespace.checkpoints.put(
            owner_agent_id=owner_agent_id,
            thread_id=thread_id,
            payload=payload,
            checkpoint_id=checkpoint_id,
            session_id=session_id,
            workflow_id=workflow_id,
            step_id=step_id,
            checkpoint_format=checkpoint_format,
            checkpoint_version=checkpoint_version,
            summary=summary,
            parent_checkpoint_id=parent_checkpoint_id,
            metadata=metadata,
        )
        return _record_from_dict(
            record or {},
            ctx=ctx,
            owner_agent_id=owner_agent_id,
            thread_id=thread_id,
            payload=payload,
            summary=summary,
            metadata=metadata,
        )

    async def get(
        self,
        ctx: ExecutionContext,
        *,
        owner_agent_id: str,
        thread_id: str,
        checkpoint_id: str | None = None,
        step_id: str | None = None,
    ) -> ExternalAgentCheckpointRecord | None:
        record = await self._namespace.checkpoints.get(
            owner_agent_id=owner_agent_id,
            thread_id=thread_id,
            checkpoint_id=checkpoint_id,
        )
        if record is None:
            return None
        return _record_from_dict(record, ctx=ctx, owner_agent_id=owner_agent_id, thread_id=thread_id)

    async def list_history(
        self,
        ctx: ExecutionContext,
        *,
        owner_agent_id: str,
        thread_id: str,
        limit: int = 20,
    ) -> list[ExternalAgentCheckpointRecord]:
        records = await self._namespace.checkpoints.list(
            owner_agent_id=owner_agent_id,
            thread_id=thread_id,
            limit=limit,
        )
        return [
            _record_from_dict(record, ctx=ctx, owner_agent_id=owner_agent_id, thread_id=thread_id)
            for record in records
            if isinstance(record, dict)
        ]


def build_gateway_client(
    incoming_headers: Mapping[str, str],
    *,
    agent_id: str,
    base_url: str | None = None,
    timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
) -> PlatformGatewayClient | None:
    base = (base_url or os.getenv("NOVIE_PLATFORM_BASE_URL", "")).strip()
    if not base:
        return None
    forward_headers = build_platform_callback_headers(incoming_headers, agent_id=agent_id)
    if not forward_headers["x-novie-org-id"] or not forward_headers["x-novie-project-id"]:
        return None
    return PlatformGatewayClient(
        base,
        forward_headers,
        agent_id=agent_id,
        timeout_seconds=timeout_seconds,
    )


def build_http_platform_services(
    incoming_headers: Mapping[str, str],
    *,
    agent_id: str,
    base_url: str | None = None,
    timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
) -> PlatformServices | None:
    namespace = build_platform_namespace(
        incoming_headers,
        agent_id=agent_id,
        base_url=base_url,
        timeout_seconds=timeout_seconds,
    )
    if not getattr(namespace, "is_available", False):
        return None
    return PlatformServices(
        wiki=HttpPlatformKnowledgeService(namespace),
        policy=_NoopPolicyService(),
        review=_NoopReviewService(),
        checkpoint=_NoopCheckpointService(),
        time_travel=_NoopTimeTravelService(),
        events=_NoopEventBus(),
        audit=_NoopAuditService(),
        usage=_NoopUsageLedgerService(),
        quota=_NoopQuotaService(),
        external_agent_checkpoints=HttpExternalAgentCheckpointService(namespace),
    )


def _record_from_dict(
    data: dict[str, Any],
    *,
    ctx: ExecutionContext,
    owner_agent_id: str,
    thread_id: str,
    payload: dict[str, Any] | None = None,
    summary: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> ExternalAgentCheckpointRecord:
    def _str_or(default: str, key: str) -> str:
        value = data.get(key)
        return str(value) if value not in (None, "") else default

    def _opt_str(key: str) -> str | None:
        value = data.get(key)
        return str(value) if value not in (None, "") else None

    def _parse_created_at() -> datetime:
        raw = data.get("created_at")
        if isinstance(raw, str):
            try:
                return datetime.fromisoformat(raw.replace("Z", "+00:00"))
            except ValueError:
                pass
        return datetime.now(UTC)

    record_payload = data.get("payload") if isinstance(data.get("payload"), dict) else payload
    record_metadata = data.get("metadata") if isinstance(data.get("metadata"), dict) else metadata
    return ExternalAgentCheckpointRecord(
        checkpoint_id=_str_or("", "checkpoint_id"),
        tenant_id=_str_or(ctx.tenant.tenant_id, "tenant_id"),
        workspace_id=_str_or(ctx.tenant.workspace_id, "workspace_id"),
        owner_agent_id=_str_or(owner_agent_id, "owner_agent_id"),
        thread_id=_str_or(thread_id, "thread_id"),
        session_id=_opt_str("session_id"),
        workflow_id=_opt_str("workflow_id"),
        step_id=_opt_str("step_id"),
        checkpoint_format=_str_or("langgraph", "checkpoint_format"),
        checkpoint_version=_str_or("1", "checkpoint_version"),
        payload=dict(record_payload or {}),
        summary=_opt_str("summary") if "summary" in data else summary,
        parent_checkpoint_id=_opt_str("parent_checkpoint_id"),
        created_at=_parse_created_at(),
        metadata=dict(record_metadata or {}),
    )


class _NoopPolicyService:
    async def evaluate(self, request):  # pragma: no cover
        raise NotImplementedError


class _NoopReviewService:
    async def open_gate(self, ctx, gate_payload):  # pragma: no cover
        raise NotImplementedError

    async def wait_for_resolution(self, gate_id):  # pragma: no cover
        raise NotImplementedError


class _NoopCheckpointService:
    async def get(self, ctx, thread_id, checkpoint_id=None):  # pragma: no cover
        return None

    async def put(self, ctx, thread_id, payload, checkpoint_id=None):  # pragma: no cover
        return checkpoint_id or ""

    async def list_history(self, ctx, thread_id, limit=20):  # pragma: no cover
        return []


class _NoopTimeTravelService:
    async def list_history(self, ctx, thread_id, limit=20):  # pragma: no cover
        return []

    async def fork_from(self, ctx, thread_id, checkpoint_id, reason):  # pragma: no cover
        return thread_id


class _NoopEventBus:
    async def publish(self, topic, payload):  # pragma: no cover
        return None

    def subscribe(self, topic) -> AsyncIterator[Any]:  # pragma: no cover
        raise NotImplementedError


class _NoopAuditService:
    async def record(self, event: AuditEvent):  # pragma: no cover
        return None

    async def query(self, ctx, *, kinds=(), thread_id=None, limit=100):  # pragma: no cover
        return []


class _NoopUsageLedgerService:
    async def record(self, rec: UsageRecord):  # pragma: no cover
        return None

    async def get_summary(self, ctx, *, scope="session", scope_value=None, breakdown_by=None):  # pragma: no cover
        return UsageSummary(
            scope=scope,
            scope_value=scope_value or "",
            input_tokens=0,
            output_tokens=0,
            total_tokens=0,
            cost_usd=0.0,
            request_count=0,
            tool_call_count=0,
            record_count=0,
        )

    async def list_records(self, ctx, *, session_id=None, thread_id=None, agent_id=None, limit=200):  # pragma: no cover
        return []


class _NoopQuotaService:
    async def check_session_token_quota(self, ctx):  # pragma: no cover
        return QuotaDecision(allow=True, policy=None, current_value=0, limit=None)


__all__ = [
    "HttpExternalAgentCheckpointService",
    "HttpPlatformKnowledgeService",
    "PlatformGatewayClient",
    "build_gateway_client",
    "build_http_platform_services",
]
