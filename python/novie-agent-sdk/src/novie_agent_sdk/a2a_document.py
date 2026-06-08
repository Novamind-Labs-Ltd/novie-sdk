"""A2A adapter helpers for document-style agents."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from novie_protocol.agents import AgentStreamEvent
from novie_protocol.contracts import ExecutionContext, IdentityContext, TenantScope

from .runtime import InvokeContext, RequestHeaders, StreamContext


def load_runtime_manifest(
    card_path: str | Path,
    *,
    public_endpoint_env: str = "NOVIE_AGENT_PUBLIC_ENDPOINT",
) -> dict[str, Any]:
    manifest = json.loads(Path(card_path).read_text(encoding="utf-8"))
    public_endpoint = os.getenv(public_endpoint_env, "").strip()
    if public_endpoint:
        manifest["endpoint"] = public_endpoint.rstrip("/")
    return manifest


def resolve_capability_id(inputs: dict[str, Any] | None) -> str:
    source = inputs if isinstance(inputs, dict) else {}
    direct = source.get("capability_id")
    if isinstance(direct, str) and direct.strip():
        return direct.strip()
    grants = source.get("capability_grants")
    if isinstance(grants, list):
        for item in grants:
            if not isinstance(item, dict):
                continue
            capability_id = item.get("capability_id")
            if isinstance(capability_id, str) and capability_id.strip():
                return capability_id.strip()
    return ""


def context_block_from_request(
    ctx: InvokeContext | StreamContext | Any,
    *,
    default_request_id: str,
    default_session_id: str,
    default_thread_id: str,
) -> dict[str, Any]:
    inline = ctx.input.get("context") if isinstance(getattr(ctx, "input", None), dict) else None
    inline = inline if isinstance(inline, dict) else {}
    headers: RequestHeaders = ctx.headers
    raw = headers.raw if isinstance(headers.raw, dict) else {}
    block: dict[str, Any] = dict(inline)
    block.setdefault("request_id", headers.request_id or default_request_id)
    block.setdefault("session_id", headers.session_id or default_session_id)
    block.setdefault(
        "thread_id",
        raw.get("x-novie-thread-id") or headers.session_id or default_thread_id,
    )
    tenant = dict(block.get("tenant") or {})
    tenant.setdefault("tenant_id", headers.tenant_id or "t")
    tenant.setdefault("workspace_id", headers.workspace_id or "w")
    project_id = raw.get("x-novie-project-id") or getattr(headers, "project_id", None)
    if project_id:
        tenant.setdefault("project_id", project_id)
    block["tenant"] = tenant

    identity = dict(block.get("identity") or {})
    service_principal = str(getattr(headers, "service_principal", "") or "").strip()
    user_id = str(getattr(headers, "user_id", "") or "").strip()
    if "principal_id" not in identity:
        identity["principal_id"] = service_principal or user_id or "u"
    if "principal_type" not in identity:
        if service_principal.startswith("agent:"):
            identity["principal_type"] = "agent"
        elif service_principal:
            identity["principal_type"] = "service"
        else:
            identity["principal_type"] = "user"
    on_behalf_of_user_id = str(
        identity.get("on_behalf_of_user_id")
        or raw.get("x-novie-on-behalf-of-user-id")
        or ""
    ).strip()
    if not on_behalf_of_user_id and isinstance(getattr(ctx, "input", None), dict):
        on_behalf_of_user_id = str(ctx.input.get("on_behalf_of_user_id") or "").strip()
    if on_behalf_of_user_id:
        identity["on_behalf_of_user_id"] = on_behalf_of_user_id
    block["identity"] = identity
    block.setdefault("workflow_id", raw.get("x-novie-workflow-id"))
    block.setdefault("parent_step_id", headers.step_id or None)
    return block


def execution_context_from_block(ctx_data: dict[str, Any]) -> ExecutionContext:
    tenant_raw = ctx_data.get("tenant") or {}
    identity_raw = ctx_data.get("identity") or {}
    return ExecutionContext(
        request_id=ctx_data.get("request_id", "req-agent-local"),
        session_id=ctx_data.get("session_id", "sess-agent-local"),
        thread_id=ctx_data.get("thread_id", "thread-agent-local"),
        tenant=TenantScope(
            tenant_id=tenant_raw.get("tenant_id", "t"),
            workspace_id=tenant_raw.get("workspace_id", "w"),
        ),
        identity=IdentityContext(
            principal_id=identity_raw.get("principal_id", "u"),
            principal_type=identity_raw.get("principal_type", "user"),
            roles=tuple(identity_raw.get("roles") or ()),
        ),
        workflow_id=ctx_data.get("workflow_id"),
        parent_step_id=ctx_data.get("parent_step_id"),
        metadata=dict(ctx_data.get("metadata") or {}),
    )


def execution_context_from_request(
    ctx: InvokeContext | StreamContext | Any,
    *,
    default_request_id: str,
    default_session_id: str,
    default_thread_id: str,
) -> ExecutionContext:
    return execution_context_from_block(
        context_block_from_request(
            ctx,
            default_request_id=default_request_id,
            default_session_id=default_session_id,
            default_thread_id=default_thread_id,
        )
    )


def is_internal_stream_visibility(metadata: dict[str, Any]) -> bool:
    values = {
        str(metadata.get("visibility") or "").strip().lower(),
        str(metadata.get("content_visibility") or "").strip().lower(),
        str(metadata.get("tool_result_visibility") or "").strip().lower(),
        str(metadata.get("output_visibility") or "").strip().lower(),
    }
    return "internal" in values or metadata.get("internal") is True


def stream_event_to_wire(
    evt: AgentStreamEvent,
    *,
    suppress_internal: bool = True,
) -> dict[str, Any]:
    raw_event = evt.metadata.get("_wire_event") if isinstance(evt.metadata, dict) else None
    if isinstance(raw_event, dict):
        return dict(raw_event)
    metadata = dict(evt.metadata or {})
    internal_visibility = suppress_internal and is_internal_stream_visibility(metadata)
    if internal_visibility:
        suppressed_chars = len(str(evt.content or "")) + len(str(evt.tool_result or ""))
        metadata.setdefault("content_suppressed", True)
        metadata.setdefault("suppressed_chars", suppressed_chars)
    body: dict[str, Any] = {"kind": evt.kind}
    if evt.content and not internal_visibility:
        body["content"] = evt.content
    if evt.output is not None:
        body["output"] = evt.output
    if metadata:
        body["metadata"] = metadata
    if evt.tool_name is not None:
        body["tool_name"] = evt.tool_name
    if evt.tool_args is not None:
        body["tool_args"] = evt.tool_args
    if evt.tool_result is not None and not internal_visibility:
        body["tool_result"] = evt.tool_result
    if evt.tool_call_id is not None:
        body["tool_call_id"] = evt.tool_call_id
    return body


__all__ = [
    "context_block_from_request",
    "execution_context_from_block",
    "execution_context_from_request",
    "is_internal_stream_visibility",
    "load_runtime_manifest",
    "resolve_capability_id",
    "stream_event_to_wire",
]
