"""Small adapters for legacy external-agent runtimes.

The SDK runtime already owns HTTP protocol handling and exposes
``InvokeContext`` / ``StreamContext``. Some existing agents still have an
older internal runtime shape that expects a protocol ``ExecutionContext``,
a raw ``inputs`` dict, and a small LLM context object. This module projects
SDK request contexts into that shape without adding agent-specific business
semantics.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

from novie_protocol.agents import AgentStreamEvent
from novie_protocol.contracts import ExecutionContext, IdentityContext, TenantScope

from .platform_namespace import build_platform_namespace
from .runtime import InvokeContext, StreamContext


@dataclass(frozen=True, slots=True)
class LegacyAgentRequest:
    """Projected request data for pre-facade agent runtimes."""

    execution_context: ExecutionContext
    inputs: dict[str, Any]
    capability_id: str
    llm_context: Any


def resolve_capability_id(inputs: dict[str, Any]) -> str:
    """Resolve the requested capability from canonical A2A inputs.

    The platform may pass it either as ``capability_id`` or inside
    ``capability_grants``. This is protocol glue only; it does not infer
    task intent or map capability ids to business workflows.
    """
    direct = inputs.get("capability_id")
    if isinstance(direct, str) and direct.strip():
        return direct.strip()
    grants = inputs.get("capability_grants")
    if isinstance(grants, list):
        for item in grants:
            if not isinstance(item, dict):
                continue
            capability_id = item.get("capability_id")
            if isinstance(capability_id, str) and capability_id.strip():
                return capability_id.strip()
    return ""


def context_block_from_sdk_request(
    ctx: InvokeContext | StreamContext,
    *,
    agent_id: str,
) -> dict[str, Any]:
    """Build the legacy ``context`` block from SDK headers plus inputs."""
    inline = ctx.input.get("context") if isinstance(ctx.input, dict) else None
    inline = inline if isinstance(inline, dict) else {}
    headers = ctx.headers
    raw = headers.raw if isinstance(headers.raw, dict) else {}
    block: dict[str, Any] = dict(inline)
    block.setdefault("request_id", headers.request_id or f"req-{agent_id}-local")
    block.setdefault("session_id", headers.session_id or f"sess-{agent_id}-local")
    block.setdefault(
        "thread_id",
        raw.get("x-novie-thread-id")
        or headers.session_id
        or f"thread-{agent_id}-local",
    )
    tenant = dict(block.get("tenant") or {})
    tenant.setdefault("tenant_id", headers.tenant_id or "t")
    tenant.setdefault("workspace_id", headers.workspace_id or "w")
    tenant.setdefault("project_id", raw.get("x-novie-project-id") or headers.project_id or None)
    block["tenant"] = tenant
    identity = dict(block.get("identity") or {})
    identity.setdefault("principal_id", headers.user_id or "u")
    identity.setdefault("principal_type", "user")
    block["identity"] = identity
    block.setdefault("workflow_id", raw.get("x-novie-workflow-id"))
    block.setdefault("parent_step_id", headers.step_id or None)
    return block


def execution_context_from_block(
    ctx_data: dict[str, Any],
    *,
    agent_id: str,
) -> ExecutionContext:
    """Convert a legacy context block into ``ExecutionContext``."""
    tenant_raw = ctx_data.get("tenant") or {}
    identity_raw = ctx_data.get("identity") or {}
    return ExecutionContext(
        request_id=ctx_data.get("request_id", f"req-{agent_id}-local"),
        session_id=ctx_data.get("session_id", f"sess-{agent_id}-local"),
        thread_id=ctx_data.get("thread_id", f"thread-{agent_id}-local"),
        tenant=TenantScope(
            tenant_id=tenant_raw.get("tenant_id", "t"),
            workspace_id=tenant_raw.get("workspace_id", "w"),
            project_id=tenant_raw.get("project_id"),
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


def execution_context_from_sdk_request(
    ctx: InvokeContext | StreamContext,
    *,
    agent_id: str,
) -> ExecutionContext:
    return execution_context_from_block(
        context_block_from_sdk_request(ctx, agent_id=agent_id),
        agent_id=agent_id,
    )


def env_float(
    name: str,
    *,
    default: float,
    minimum: float | None = None,
) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        value = default
    else:
        try:
            value = float(raw)
        except ValueError:
            value = default
    if minimum is not None:
        value = max(value, minimum)
    return value


def llm_context_from_sdk_request(
    ctx: InvokeContext | StreamContext,
    *,
    agent_id: str,
    capability_timeout_seconds: float | None = None,
    llm_timeout_seconds: float | None = None,
) -> Any:
    """Return the small object expected by legacy LangChain model builders."""
    kwargs: dict[str, Any] = {}
    if capability_timeout_seconds is not None:
        kwargs["timeout_seconds"] = capability_timeout_seconds
    if llm_timeout_seconds is not None:
        kwargs["llm_timeout_seconds"] = llm_timeout_seconds
    platform_ns = build_platform_namespace(ctx.headers, agent_id=agent_id, **kwargs)
    return SimpleNamespace(headers=ctx.headers, platform_ns=platform_ns)


def legacy_request_from_sdk_context(
    ctx: InvokeContext | StreamContext,
    *,
    agent_id: str,
    capability_timeout_seconds: float | None = None,
    llm_timeout_seconds: float | None = None,
) -> LegacyAgentRequest:
    inputs = ctx.input if isinstance(ctx.input, dict) else {}
    return LegacyAgentRequest(
        execution_context=execution_context_from_sdk_request(ctx, agent_id=agent_id),
        inputs=inputs,
        capability_id=resolve_capability_id(inputs),
        llm_context=llm_context_from_sdk_request(
            ctx,
            agent_id=agent_id,
            capability_timeout_seconds=capability_timeout_seconds,
            llm_timeout_seconds=llm_timeout_seconds,
        ),
    )


def format_stream_event(evt: AgentStreamEvent) -> dict[str, Any]:
    """Render a protocol ``AgentStreamEvent`` as SDK stream payload."""
    body: dict[str, Any] = {"kind": evt.kind}
    if evt.content:
        body["content"] = evt.content
    if evt.output is not None:
        body["output"] = evt.output
    if evt.metadata:
        body["metadata"] = evt.metadata
    if evt.tool_name is not None:
        body["tool_name"] = evt.tool_name
    if evt.tool_args is not None:
        body["tool_args"] = evt.tool_args
    if evt.tool_result is not None:
        body["tool_result"] = evt.tool_result
    if evt.tool_call_id is not None:
        body["tool_call_id"] = evt.tool_call_id
    return body


__all__ = [
    "LegacyAgentRequest",
    "context_block_from_sdk_request",
    "env_float",
    "execution_context_from_block",
    "execution_context_from_sdk_request",
    "format_stream_event",
    "legacy_request_from_sdk_context",
    "llm_context_from_sdk_request",
    "resolve_capability_id",
]
