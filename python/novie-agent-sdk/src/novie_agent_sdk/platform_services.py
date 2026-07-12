"""HTTP-backed ``novie_protocol.services.PlatformServices`` adapter."""
from __future__ import annotations

import json
import logging
import os
from collections.abc import AsyncIterator
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

from .degradation import (
    CapabilityCallDiagnostics,
    DegradationTracker,
    classify_envelope_error,
)
from .platform_callback import build_platform_callback_headers, sign_platform_callback_headers

_log = logging.getLogger(__name__)

KNOWLEDGE_SEARCH_CAP = "platform.knowledge.search"
CHECKPOINT_PUT_CAP = "platform.external_agent_checkpoint.put"
CHECKPOINT_GET_CAP = "platform.external_agent_checkpoint.get"
CHECKPOINT_LIST_CAP = "platform.external_agent_checkpoint.list"
DEFAULT_TIMEOUT_SECONDS = 30.0


class CapabilityClient:
    """Small platform capability client used by ``PlatformServices`` adapters."""

    def __init__(
        self,
        base_url: str,
        forward_headers: dict[str, str],
        *,
        agent_id: str,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._headers = dict(forward_headers)
        self._agent_id = agent_id
        self._timeout = timeout_seconds

    async def get_json(self, path: str) -> dict[str, Any] | None:
        if not path.startswith("/"):
            path = "/" + path
        url = f"{self._base_url}{path}"
        req = request.Request(url, method="GET")
        headers = sign_platform_callback_headers(self._headers, method="GET", path=path)
        for key, value in headers.items():
            if value:
                req.add_header(key, value)

        import asyncio

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
                _log.warning("gateway GET failed path=%s status=%s detail=%s", path, exc.code, detail)
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

    async def invoke(
        self,
        capability_id: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any] | None:
        diagnostics = await self.invoke_with_diagnostics(capability_id, arguments)
        return diagnostics.result if diagnostics.ok else None

    async def invoke_with_diagnostics(
        self,
        capability_id: str,
        arguments: dict[str, Any],
    ) -> CapabilityCallDiagnostics:
        path = "/invocations"
        url = f"{self._base_url}{path}"
        body = json.dumps(
            {
                "capability_id": capability_id,
                "provider_id": capability_id.rsplit(".", 1)[0],
                "mode": "execute",
                "inputs": dict(arguments),
            }
        ).encode("utf-8")
        req = request.Request(url, data=body, method="POST")
        headers = sign_platform_callback_headers(self._headers, method="POST", path=path)
        for key, value in headers.items():
            if value:
                req.add_header(key, value)

        import asyncio

        def _do_request() -> CapabilityCallDiagnostics:
            try:
                with request.urlopen(req, timeout=self._timeout) as resp:
                    raw = resp.read().decode("utf-8")
            except error.HTTPError as exc:
                detail = ""
                try:
                    detail = exc.read().decode("utf-8")[:300]
                except Exception:
                    pass
                envelope_code: str | None = None
                if detail:
                    try:
                        body_json = json.loads(detail)
                        if isinstance(body_json, dict):
                            envelope_code = (
                                str(body_json.get("error_code") or "")
                                or str(
                                    (body_json.get("detail") or {}).get("error_code")
                                    if isinstance(body_json.get("detail"), dict)
                                    else ""
                                )
                            ) or None
                    except json.JSONDecodeError:
                        envelope_code = None
                kind = classify_envelope_error(envelope_code, exc.code)
                return CapabilityCallDiagnostics(
                    ok=False,
                    result=None,
                    kind=kind,
                    error_code=envelope_code or "",
                    detail=detail,
                    capability_id=capability_id,
                )
            except error.URLError as exc:
                return CapabilityCallDiagnostics(
                    ok=False,
                    result=None,
                    kind="transport_error",
                    detail=str(exc.reason),
                    capability_id=capability_id,
                )
            try:
                envelope = json.loads(raw)
            except json.JSONDecodeError:
                return CapabilityCallDiagnostics(
                    ok=False,
                    result=None,
                    kind="platform_unavailable",
                    error_code="non_json_response",
                    capability_id=capability_id,
                )
            if not isinstance(envelope, dict):
                return CapabilityCallDiagnostics(
                    ok=False,
                    result=None,
                    kind="platform_unavailable",
                    error_code="non_object_envelope",
                    capability_id=capability_id,
                )
            if str(envelope.get("status")) != "ok":
                envelope_code = str(envelope.get("error_code") or "") or None
                kind = classify_envelope_error(envelope_code, http_status=None)
                detail = str(
                    envelope.get("error_message") or envelope.get("explanation") or ""
                )
                return CapabilityCallDiagnostics(
                    ok=False,
                    result=None,
                    kind=kind,
                    error_code=envelope_code or "",
                    detail=detail,
                    capability_id=capability_id,
                )
            result = envelope.get("output")
            return CapabilityCallDiagnostics(
                ok=True,
                result=result if isinstance(result, dict) else None,
                capability_id=capability_id,
            )

        return await asyncio.to_thread(_do_request)


class HttpWikiService:
    def __init__(
        self,
        client: CapabilityClient,
        *,
        tracker: DegradationTracker | None = None,
    ) -> None:
        self._client = client
        self._tracker = tracker

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
            else (getattr(ctx.tenant, "project_id", None) or ctx.tenant.workspace_id)
        )
        payload: dict[str, Any] = {"query": query, "top_k": int(top_k)}
        if scope:
            payload["project_id"] = str(scope)
        diagnostics = await self._client.invoke_with_diagnostics(KNOWLEDGE_SEARCH_CAP, payload)
        if not diagnostics.ok:
            if self._tracker is not None:
                self._tracker.mark_diagnostics("platform_knowledge_search", diagnostics)
            return []
        result = diagnostics.result or {}
        results_raw = result.get("results") or []
        if not isinstance(results_raw, list):
            return []
        out = [dict(item) for item in results_raw if isinstance(item, dict)]
        if not out and self._tracker is not None:
            self._tracker.mark("platform_knowledge_search", "no_results")
        return out


class HttpExternalAgentCheckpointService:
    def __init__(
        self,
        client: CapabilityClient,
        *,
        tracker: DegradationTracker | None = None,
    ) -> None:
        self._client = client
        self._tracker = tracker

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
        args: dict[str, Any] = {
            "thread_id": thread_id,
            "owner_agent_id": owner_agent_id,
            "payload": dict(payload),
            "checkpoint_format": checkpoint_format,
            "checkpoint_version": checkpoint_version,
            "metadata": dict(metadata or {}),
        }
        for key, value in (
            ("checkpoint_id", checkpoint_id),
            ("session_id", session_id),
            ("workflow_id", workflow_id),
            ("step_id", step_id),
            ("summary", summary),
            ("parent_checkpoint_id", parent_checkpoint_id),
        ):
            if value:
                args[key] = value
        diagnostics = await self._client.invoke_with_diagnostics(CHECKPOINT_PUT_CAP, args)
        if not diagnostics.ok and self._tracker is not None:
            self._tracker.mark_diagnostics("platform_external_agent_checkpoint", diagnostics)
        record_data: dict[str, Any] = {}
        if diagnostics.ok and isinstance(diagnostics.result, dict):
            checkpoint_block = diagnostics.result.get("checkpoint")
            if isinstance(checkpoint_block, dict):
                record_data = checkpoint_block
        return record_from_dict(
            record_data,
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
        args: dict[str, Any] = {
            "thread_id": thread_id,
            "owner_agent_id": owner_agent_id,
        }
        if checkpoint_id:
            args["checkpoint_id"] = checkpoint_id
        if step_id:
            args["step_id"] = step_id
        diagnostics = await self._client.invoke_with_diagnostics(CHECKPOINT_GET_CAP, args)
        if not diagnostics.ok:
            if self._tracker is not None:
                self._tracker.mark_diagnostics("platform_external_agent_checkpoint", diagnostics)
            return None
        result = diagnostics.result or {}
        checkpoint_block = result.get("checkpoint")
        if not isinstance(checkpoint_block, dict):
            return None
        return record_from_dict(
            checkpoint_block,
            ctx=ctx,
            owner_agent_id=owner_agent_id,
            thread_id=thread_id,
        )

    async def list_history(
        self,
        ctx: ExecutionContext,
        *,
        owner_agent_id: str,
        thread_id: str,
        limit: int = 20,
    ) -> list[ExternalAgentCheckpointRecord]:
        args = {
            "thread_id": thread_id,
            "owner_agent_id": owner_agent_id,
            "limit": int(limit),
        }
        result = await self._client.invoke(CHECKPOINT_LIST_CAP, args)
        if not isinstance(result, dict):
            return []
        items = result.get("checkpoints") or result.get("items") or []
        if not isinstance(items, list):
            return []
        return [
            record_from_dict(entry, ctx=ctx, owner_agent_id=owner_agent_id, thread_id=thread_id)
            for entry in items
            if isinstance(entry, dict)
        ]


def record_from_dict(
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
        if value in (None, ""):
            return None
        return str(value)

    def _parse_created_at() -> datetime:
        raw = data.get("created_at")
        if isinstance(raw, str):
            try:
                return datetime.fromisoformat(raw.replace("Z", "+00:00"))
            except ValueError:
                pass
        return datetime.now(UTC)

    record_payload = (
        dict(data["payload"])
        if isinstance(data.get("payload"), dict)
        else dict(payload or {})
    )
    record_metadata = (
        dict(data["metadata"])
        if isinstance(data.get("metadata"), dict)
        else dict(metadata or {})
    )
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
        payload=record_payload,
        summary=_opt_str("summary") if "summary" in data else summary,
        parent_checkpoint_id=_opt_str("parent_checkpoint_id"),
        created_at=_parse_created_at(),
        metadata=record_metadata,
    )


class _NoopPolicyService:
    async def evaluate(self, request: Any) -> Any:
        raise NotImplementedError("policy service is not wired for platform callbacks")


class _NoopReviewService:
    async def open_gate(self, ctx: ExecutionContext, gate_payload: dict[str, Any]) -> str:
        raise NotImplementedError("review service is not wired for platform callbacks")

    async def wait_for_resolution(self, gate_id: str) -> dict[str, Any]:
        raise NotImplementedError("review service is not wired for platform callbacks")


class _NoopCheckpointService:
    async def get(self, ctx: ExecutionContext, thread_id: str, checkpoint_id: str | None = None) -> None:
        return None

    async def list_history(self, ctx: ExecutionContext, thread_id: str, limit: int = 20) -> list[Any]:
        return []


class _NoopTimeTravelService:
    async def list_history(self, ctx: ExecutionContext, thread_id: str, limit: int = 20) -> list[Any]:
        return []

    async def fork_from(self, ctx: ExecutionContext, thread_id: str, checkpoint_id: str, reason: str) -> str:
        raise NotImplementedError("time travel is not wired for platform callbacks")


class _NoopEventBus:
    async def publish(self, topic: str, payload: dict[str, Any]) -> None:
        return None

    def subscribe(self, topic: str) -> AsyncIterator[dict[str, Any]]:
        raise NotImplementedError("event bus subscribe is not wired for platform callbacks")


class _NoopAuditService:
    async def record(self, event: AuditEvent) -> None:
        return None

    async def query(
        self,
        ctx: ExecutionContext,
        *,
        kinds: tuple[Any, ...] = (),
        thread_id: str | None = None,
        limit: int = 100,
    ) -> list[AuditEvent]:
        return []


class _NoopUsageLedgerService:
    async def record(self, record: UsageRecord) -> None:
        return None

    async def get_summary(
        self,
        ctx: ExecutionContext,
        *,
        scope: Any = "session",
        scope_value: str | None = None,
        breakdown_by: Any = None,
    ) -> UsageSummary:
        return UsageSummary(
            scope=scope,
            scope_value=scope_value or ctx.session_id,
            input_tokens=0,
            output_tokens=0,
            total_tokens=0,
            cost_usd=0.0,
            request_count=0,
            tool_call_count=0,
            record_count=0,
        )

    async def list_records(
        self,
        ctx: ExecutionContext,
        *,
        session_id: str | None = None,
        thread_id: str | None = None,
        agent_id: str | None = None,
        limit: int = 200,
    ) -> list[UsageRecord]:
        return []


class _NoopQuotaService:
    async def check_session_token_quota(self, ctx: ExecutionContext) -> QuotaDecision:
        return QuotaDecision(allow=True, policy=None, current_value=0, limit=None)

    def configure(self, policy: Any) -> None:
        return None


def build_forward_headers(incoming: dict[str, str], *, agent_id: str) -> dict[str, str]:
    return build_platform_callback_headers(incoming, agent_id=agent_id)


def build_gateway_client(
    incoming_headers: dict[str, str],
    *,
    agent_id: str,
    base_url: str | None = None,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> CapabilityClient | None:
    base = (base_url or os.getenv("NOVIE_PLATFORM_BASE_URL", "")).strip()
    if not base:
        return None
    headers = build_forward_headers(incoming_headers, agent_id=agent_id)
    if not headers["x-novie-org-id"] or not headers["x-novie-project-id"]:
        return None
    return CapabilityClient(base, headers, agent_id=agent_id, timeout_seconds=timeout_seconds)


def build_http_platform_services(
    incoming_headers: dict[str, str],
    *,
    agent_id: str,
    base_url: str | None = None,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    tracker: DegradationTracker | None = None,
) -> PlatformServices | None:
    base = (base_url or os.getenv("NOVIE_PLATFORM_BASE_URL", "")).strip()
    if not base:
        return None
    headers = build_forward_headers(incoming_headers, agent_id=agent_id)
    if not headers["x-novie-org-id"] or not headers["x-novie-project-id"]:
        _log.info("platform services not constructed: missing tenant/project headers")
        return None
    client = CapabilityClient(base, headers, agent_id=agent_id, timeout_seconds=timeout_seconds)
    return PlatformServices(
        wiki=HttpWikiService(client, tracker=tracker),
        policy=_NoopPolicyService(),
        review=_NoopReviewService(),
        checkpoint=_NoopCheckpointService(),
        time_travel=_NoopTimeTravelService(),
        events=_NoopEventBus(),
        audit=_NoopAuditService(),
        usage=_NoopUsageLedgerService(),
        quota=_NoopQuotaService(),
        external_agent_checkpoints=HttpExternalAgentCheckpointService(client, tracker=tracker),
    )


__all__ = [
    "CHECKPOINT_GET_CAP",
    "CHECKPOINT_LIST_CAP",
    "CHECKPOINT_PUT_CAP",
    "DEFAULT_TIMEOUT_SECONDS",
    "KNOWLEDGE_SEARCH_CAP",
    "CapabilityClient",
    "HttpExternalAgentCheckpointService",
    "HttpWikiService",
    "build_forward_headers",
    "build_gateway_client",
    "build_http_platform_services",
    "record_from_dict",
]
