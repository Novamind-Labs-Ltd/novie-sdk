"""Final-output helpers for document-style external agents."""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from novie_protocol.agents import AgentCard, AgentStreamEvent


def capability_provides_artifacts(
    *,
    card: AgentCard | None = None,
    capability_manifest: Iterable[Any] | None = None,
    capability_id: str,
    artifact_type: str = "",
    structured_output: Mapping[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    """Return a stable provides_artifacts map for a capability final event."""
    entries = list(capability_manifest or getattr(card, "capability_manifest", ()) or ())
    names: list[str] = []
    for entry in entries:
        if getattr(entry, "capability_id", None) == capability_id:
            names.extend(str(item) for item in (getattr(entry, "provides", ()) or ()))
            break
    if artifact_type:
        names.append(str(artifact_type))

    structured = dict(structured_output or {})
    provided: dict[str, dict[str, Any]] = {}
    for name in names:
        artifact_name = str(name or "").strip()
        if artifact_name:
            provided.setdefault(artifact_name, {"structured_output": structured})
    return provided


def recovery_metadata(
    *,
    finalize_attempts: int = 1,
    checkpoint_id: str = "",
    resumed_from_checkpoint: bool = False,
    fallback_used: bool = False,
    fallback_reason: str = "",
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the common recovery payload used by document agents."""
    return {
        "fallback_used": bool(fallback_used),
        "fallback_reason": str(fallback_reason or ""),
        "resumed_from_checkpoint": bool(resumed_from_checkpoint),
        "checkpoint_id": str(checkpoint_id or ""),
        "finalize_attempts": max(1, int(finalize_attempts or 1)),
        "metadata": dict(metadata or {}),
    }


def document_final_output(
    *,
    artifact_type: str,
    artifact_family: str,
    capability_id: str,
    analysis: str,
    narrative: str,
    structured_output: Mapping[str, Any] | None = None,
    final_payload: Mapping[str, Any] | None = None,
    card: AgentCard | None = None,
    capability_manifest: Iterable[Any] | None = None,
    mode_key: str | None = None,
    mode: str | None = None,
    phase_key: str | None = None,
    phase: str | None = None,
    checkpoint_id: str = "",
    budget_summary: Mapping[str, Any] | None = None,
    degraded_flags: Iterable[str] | None = None,
    quality: Mapping[str, Any] | None = None,
    resumed_from_checkpoint: bool = False,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a platform-friendly final output without hiding agent-specific fields."""
    structured = dict(structured_output or {})
    output: dict[str, Any] = {
        "artifact_type": artifact_type,
        "artifact_family": artifact_family,
        "capability_id": capability_id,
        "analysis": analysis,
        "content": analysis,
        "narrative": narrative,
        "structured_output": structured,
        "provides_artifacts": capability_provides_artifacts(
            card=card,
            capability_manifest=capability_manifest,
            capability_id=capability_id,
            artifact_type=artifact_type,
            structured_output=structured,
        ),
    }
    if mode_key and mode is not None:
        output[mode_key] = mode
    if phase_key and phase is not None:
        output[phase_key] = phase
    if final_payload is not None:
        output["final_payload"] = dict(final_payload)
    if checkpoint_id:
        output["checkpoint_id"] = checkpoint_id
    if budget_summary:
        output["budget_summary"] = dict(budget_summary)
    flags = [str(flag) for flag in degraded_flags or () if str(flag)]
    if flags:
        output["degraded_flags"] = flags
    if quality:
        output.update(dict(quality))
    if resumed_from_checkpoint:
        output["resumed_from_checkpoint"] = True
    if extra:
        output.update(dict(extra))
    return output


def document_final_event(
    *,
    output: Mapping[str, Any],
    metadata: Mapping[str, Any] | None = None,
) -> AgentStreamEvent:
    """Return the standard final AgentStreamEvent for document agents."""
    return AgentStreamEvent(
        kind="final",
        output=dict(output),
        metadata=dict(metadata or {}),
    )


__all__ = [
    "capability_provides_artifacts",
    "document_final_event",
    "document_final_output",
    "recovery_metadata",
]
