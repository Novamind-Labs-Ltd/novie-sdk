"""Checkpoint identity helpers for external agents."""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any


def checkpoint_step_id(ctx: Any) -> str:
    metadata = getattr(ctx, "metadata", None)
    metadata = metadata if isinstance(metadata, Mapping) else {}
    return str(getattr(ctx, "parent_step_id", None) or metadata.get("step_id") or "").strip()


def checkpoint_input_digest(parts: Mapping[str, Any] | None = None, **kwargs: Any) -> str:
    payload = dict(parts or {})
    payload.update(kwargs)
    rendered = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def checkpoint_matches_invocation(
    record: Any,
    *,
    payload_metadata: Mapping[str, Any] | None = None,
    ctx: Any,
    capability_id: str | None,
    input_digest: str,
) -> bool:
    record_metadata = getattr(record, "metadata", None)
    metadata: dict[str, Any] = dict(record_metadata) if isinstance(record_metadata, Mapping) else {}
    metadata.update(dict(payload_metadata or {}))

    expected_step_id = checkpoint_step_id(ctx)
    record_step_id = str(getattr(record, "step_id", "") or metadata.get("step_id") or "").strip()
    if expected_step_id and record_step_id and record_step_id != expected_step_id:
        return False

    expected_workflow_id = str(getattr(ctx, "workflow_id", "") or "").strip()
    record_workflow_id = str(
        getattr(record, "workflow_id", "") or metadata.get("workflow_id") or ""
    ).strip()
    if expected_workflow_id and record_workflow_id and record_workflow_id != expected_workflow_id:
        return False

    expected_capability_id = capability_id or ""
    record_capability_id = str(metadata.get("capability_id") or "").strip()
    if expected_capability_id and record_capability_id and record_capability_id != expected_capability_id:
        return False

    record_digest = str(metadata.get("input_digest") or "").strip()
    if record_digest and record_digest != input_digest:
        return False
    return True


def external_agent_checkpoint_service(services: Any | None) -> Any | None:
    """Return the platform checkpoint adapter exposed to external agents."""
    if services is None:
        return None
    return (
        getattr(services, "external_agent_checkpoints", None)
        or getattr(services, "checkpoint", None)
    )


async def put_external_agent_checkpoint(
    checkpoint_service: Any,
    ctx: Any,
    *,
    owner_agent_id: str,
    payload: dict[str, Any],
    workflow_id: str | None,
    step_id: str | None,
    parent_checkpoint_id: str | None = None,
    summary: str,
    metadata: dict[str, Any],
) -> Any:
    """Write an external-agent checkpoint across old and new service shapes."""
    metadata = dict(metadata)
    phase_outputs = payload.get("phase_outputs")
    finalize_output = phase_outputs.get("finalize") if isinstance(phase_outputs, dict) else None
    if isinstance(finalize_output, dict) and "checkpoint_version" in finalize_output:
        metadata.setdefault("checkpoint_version", finalize_output["checkpoint_version"])
    try:
        return await checkpoint_service.put(
            ctx,
            owner_agent_id=owner_agent_id,
            thread_id=ctx.thread_id,
            payload=payload,
            workflow_id=workflow_id,
            step_id=step_id,
            parent_checkpoint_id=parent_checkpoint_id,
            summary=summary,
            metadata=metadata,
        )
    except TypeError:
        return await checkpoint_service.put(ctx, ctx.thread_id, payload)


__all__ = [
    "checkpoint_input_digest",
    "checkpoint_matches_invocation",
    "checkpoint_step_id",
    "external_agent_checkpoint_service",
    "put_external_agent_checkpoint",
]
