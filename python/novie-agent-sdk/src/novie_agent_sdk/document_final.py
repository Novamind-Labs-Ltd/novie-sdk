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


def _dump_model(value: Any) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        dumped = value.model_dump()
        return dict(dumped) if isinstance(dumped, Mapping) else {}
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def build_document_deliverable_event(
    *,
    card: AgentCard | None,
    structured: Any,
    artifact_type: str,
    artifact_family: str,
    capability_id: str | None,
    analysis: str,
    narrative: str,
    final_payload_type: type[Any],
    recovery_type: type[Any],
    mode_key: str | None = None,
    mode: str | None = None,
    phase_key: str | None = None,
    phase: str | None = None,
    plan_id: str | None = None,
    finalize_strategy: str = "native",
    finalize_attempts: int = 1,
    degraded_flags: Iterable[str] | None = None,
    checkpoint_id: str = "",
    resumed_from_checkpoint: bool = False,
    fallback_used: bool = False,
    fallback_reason: str = "",
    quality: Mapping[str, Any] | None = None,
    budget_summary: Mapping[str, Any] | None = None,
    authoring_ledger: Mapping[str, Any] | None = None,
    skill_contract: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
    payload_metadata: Mapping[str, Any] | None = None,
    recovery_metadata_extra: Mapping[str, Any] | None = None,
    output_extra: Mapping[str, Any] | None = None,
    title: str | None = None,
    authoring_strategy: str = "sectioned_longform",
) -> AgentStreamEvent:
    """Build the standard final deliverable event for document agents.

    Agent code still owns analysis rendering and domain payload classes. This
    helper owns the repeated envelope: recovery, final_payload,
    provides_artifacts, metadata, and the final ``AgentStreamEvent``.
    """
    structured_dump = _dump_model(structured)
    flags = [str(flag) for flag in degraded_flags or () if str(flag)]
    common_metadata: dict[str, Any] = {
        "artifact_type": artifact_type,
        "artifact_family": artifact_family,
        **({mode_key: mode} if mode_key and mode is not None else {}),
        **({phase_key: phase} if phase_key and phase is not None else {}),
        **({"capability_id": capability_id} if capability_id else {}),
        "finalize_strategy": finalize_strategy,
        "finalize_attempts": max(1, int(finalize_attempts or 1)),
    }
    event_metadata = {**common_metadata, **dict(metadata or {})}
    if fallback_used:
        event_metadata["delivery_mode"] = "non_stream_fallback"
    if checkpoint_id:
        event_metadata["checkpoint_id"] = checkpoint_id
    if resumed_from_checkpoint:
        event_metadata["resumed_from_checkpoint"] = True
    if budget_summary:
        event_metadata["budget_summary"] = dict(budget_summary)
    if flags:
        event_metadata["degraded_flags"] = list(flags)
    if quality:
        event_metadata["quality"] = dict(quality)
    if authoring_ledger:
        event_metadata["authoring_strategy"] = authoring_strategy
        event_metadata["authoring_ledger"] = dict(authoring_ledger)
    if skill_contract:
        event_metadata["skill_contract"] = dict(skill_contract)

    recovery_metadata = {
        "finalize_strategy": finalize_strategy,
        **(
            {"authoring_ledger": dict(authoring_ledger)}
            if authoring_ledger
            else {}
        ),
        **({"quality": dict(quality)} if quality else {}),
        **({"budget_summary": dict(budget_summary)} if budget_summary else {}),
        **dict(recovery_metadata_extra or {}),
    }
    recovery = recovery_type(
        fallback_used=fallback_used,
        fallback_reason=fallback_reason,
        resumed_from_checkpoint=resumed_from_checkpoint,
        checkpoint_id=checkpoint_id,
        finalize_attempts=max(1, int(finalize_attempts or 1)),
        metadata=recovery_metadata,
    )

    final_payload_metadata = {
        "narrative_preview": narrative[:500] if narrative else "",
        "artifact_type": artifact_type,
        "artifact_family": artifact_family,
        **({mode_key: mode} if mode_key and mode is not None else {}),
        **({phase_key: phase} if phase_key and phase is not None else {}),
        **({"quality": dict(quality)} if quality else {}),
        **(
            {
                "authoring_strategy": authoring_strategy,
                "authoring_ledger": dict(authoring_ledger),
            }
            if authoring_ledger
            else {}
        ),
        **({"checkpoint_id": checkpoint_id} if checkpoint_id else {}),
        **({"budget_summary": dict(budget_summary)} if budget_summary else {}),
        **dict(payload_metadata or {}),
    }
    final_payload = final_payload_type(
        plan_id=plan_id or capability_id or artifact_type,
        final_markdown=analysis,
        structured_output=structured_dump,
        degraded_flags=list(flags),
        recovery=recovery,
        metadata=final_payload_metadata,
    )

    output = document_final_output(
        artifact_type=artifact_type,
        artifact_family=artifact_family,
        capability_id=capability_id or "",
        analysis=analysis,
        narrative=narrative,
        structured_output=structured_dump,
        final_payload=_dump_model(final_payload),
        card=card,
        mode_key=mode_key,
        mode=mode,
        phase_key=phase_key,
        phase=phase,
        checkpoint_id=checkpoint_id,
        budget_summary=budget_summary,
        degraded_flags=flags,
        quality=quality,
        resumed_from_checkpoint=resumed_from_checkpoint,
        extra=output_extra,
    )
    output.update(
        {
            "kind": "document_deliverable",
            "title": str(
                title
                or structured_dump.get("summary")
                or capability_id
                or "Final Deliverable"
            )[:120],
            "final_markdown": analysis,
            "content": analysis,
            "authoring_strategy": authoring_strategy,
        }
    )
    if authoring_ledger:
        output["authoring_ledger"] = dict(authoring_ledger)
    return document_final_event(output=output, metadata=event_metadata)


__all__ = [
    "build_document_deliverable_event",
    "capability_provides_artifacts",
    "document_final_event",
    "document_final_output",
    "recovery_metadata",
]
