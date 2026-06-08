"""Checkpoint resume helpers for document-style external agents."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from novie_protocol.agents import AgentStreamEvent

from .external_checkpoint import checkpoint_matches_invocation


@dataclass(frozen=True)
class DocumentResumeCandidate:
    payload: Any
    checkpoint_id: str
    record: Any


def _record_payload(record: Any) -> Any:
    if record is None:
        return None
    if hasattr(record, "payload"):
        return getattr(record, "payload")
    if isinstance(record, dict):
        return record.get("payload", record)
    return None


def _validate_payload(payload_model: Any, payload: Any) -> Any:
    if payload_model is None:
        return payload
    if hasattr(payload_model, "model_validate"):
        return payload_model.model_validate(payload)
    return payload_model(**payload)


async def get_matching_document_checkpoint(
    checkpoint_service: Any,
    ctx: Any,
    *,
    payload_model: Any,
    owner_agent_id: str | None = None,
    capability_id: str | None,
    input_digest: str,
    step_id: str | None = None,
    required_phase: str = "finalize",
) -> DocumentResumeCandidate | None:
    """Read and validate the latest matching checkpoint for this invocation."""
    if checkpoint_service is None:
        return None
    get_kwargs = {
        "owner_agent_id": owner_agent_id,
        "thread_id": getattr(ctx, "thread_id", None),
        "step_id": step_id or None,
    }
    try:
        record = await checkpoint_service.get(ctx, **get_kwargs)
    except TypeError:
        record = await checkpoint_service.get(ctx, getattr(ctx, "thread_id", None))
    if record is None:
        return None

    raw_payload = _record_payload(record)
    if raw_payload is None:
        return None
    payload = _validate_payload(payload_model, raw_payload)
    metadata = getattr(payload, "metadata", None)
    phase = str(getattr(payload, "current_phase", "") or "").strip()
    narrative = str(getattr(payload, "narrative", "") or "").strip()
    if required_phase and phase != required_phase:
        return None
    if not narrative:
        return None
    if not checkpoint_matches_invocation(
        record,
        payload_metadata=metadata if isinstance(metadata, dict) else None,
        ctx=ctx,
        capability_id=capability_id,
        input_digest=input_digest,
    ):
        return None
    return DocumentResumeCandidate(
        payload=payload,
        checkpoint_id=str(getattr(record, "checkpoint_id", "") or ""),
        record=record,
    )


def skipped_phase_events(
    *,
    skipped_phases: list[str] | tuple[str, ...],
    phase_metadata: Any,
    mode: str,
    phase: str,
    capability_id: str,
    checkpoint_id: str = "",
    event_name: str = "phase_skipped",
    reason: str = "resumed_from_checkpoint",
) -> list[AgentStreamEvent]:
    """Build trace events for phases skipped because a checkpoint was reused."""
    events: list[AgentStreamEvent] = []
    for runtime_phase in skipped_phases:
        metadata = phase_metadata(
            runtime_phase=runtime_phase,
            mode=mode,
            phase=phase,
            capability_id=capability_id,
        )
        metadata.update(
            {
                "event": event_name,
                "reason": reason,
                "resumed_from_checkpoint": True,
            }
        )
        if checkpoint_id:
            metadata["checkpoint_id"] = checkpoint_id
        events.append(AgentStreamEvent(kind="trace", metadata=metadata))
    return events


__all__ = [
    "DocumentResumeCandidate",
    "get_matching_document_checkpoint",
    "skipped_phase_events",
]
