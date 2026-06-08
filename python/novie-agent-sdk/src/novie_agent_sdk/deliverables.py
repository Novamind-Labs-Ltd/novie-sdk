from __future__ import annotations

from typing import Any, Mapping


def markdown_deliverable_output(
    *,
    title: str,
    markdown: str,
    artifact_type: str,
    artifact_family: str = "",
    metadata: Mapping[str, Any] | None = None,
    provides_artifacts: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a standard final-output block for document-producing agents."""
    body = str(markdown or "")
    out: dict[str, Any] = {
        "kind": "document_deliverable",
        "title": str(title or "").strip(),
        "artifact_type": str(artifact_type or ""),
        "analysis": body,
        "final_markdown": body,
        "content": body,
        "metadata": dict(metadata or {}),
    }
    if artifact_family:
        out["artifact_family"] = str(artifact_family)
    if provides_artifacts:
        out["provides_artifacts"] = dict(provides_artifacts)
    return out


def bounded_handoff_output(
    *,
    handoff_markdown: str,
    artifact_type: str,
    artifact_family: str = "",
    capability_id: str | None = None,
    step_role: str = "upstream_handoff",
    output_contract: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
    provides_artifacts: Mapping[str, Any] | None = None,
    degraded_flags: list[str] | None = None,
    budget_summary: Mapping[str, Any] | None = None,
    recovery: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a standard internal handoff output for intermediate DAG steps.

    This is intentionally plain-dict based so document agents, planning agents,
    and future external agents can use the same platform envelope without
    depending on another agent's pydantic contracts.
    """
    handoff = str(handoff_markdown or "").strip() or "# Evidence Handoff\n\nNo evidence produced."
    output_contract_data = dict(output_contract or {})
    metadata_data = {
        **dict(metadata or {}),
        "artifact_type": str(artifact_type or ""),
        "step_role": step_role,
        "output_contract": output_contract_data,
        "output_visibility": "internal",
        "finalize_strategy": "bounded_handoff_commit",
    }
    if artifact_family:
        metadata_data["artifact_family"] = str(artifact_family)
    if capability_id:
        metadata_data["capability_id"] = capability_id
    if budget_summary:
        metadata_data["budget_summary"] = dict(budget_summary)

    structured_output = {
        "kind": "bounded_handoff",
        "handoff_markdown": handoff,
        "summary": handoff[:1000],
        "metadata": dict(metadata_data),
    }
    final_payload = {
        "plan_id": capability_id or artifact_type,
        "final_markdown": handoff,
        "structured_output": structured_output,
        "degraded_flags": list(degraded_flags or []),
        "recovery": {
            "fallback_used": False,
            "fallback_reason": "",
            "resumed_from_checkpoint": False,
            "checkpoint_id": "",
            "finalize_attempts": 1,
            "metadata": {"finalize_strategy": "bounded_handoff_commit"},
            **dict(recovery or {}),
        },
        "metadata": dict(metadata_data),
    }
    out: dict[str, Any] = {
        "kind": "bounded_handoff",
        "artifact_type": str(artifact_type or ""),
        "analysis": handoff,
        "content": handoff,
        "final_markdown": handoff,
        "step_role": step_role,
        "output_contract": output_contract_data,
        "output_visibility": "internal",
        "metadata": dict(metadata_data),
        "final_payload": final_payload,
    }
    if artifact_family:
        out["artifact_family"] = str(artifact_family)
    if capability_id:
        out["capability_id"] = capability_id
    if provides_artifacts:
        out["provides_artifacts"] = dict(provides_artifacts)
    if budget_summary:
        out["budget_summary"] = dict(budget_summary)
    return out


__all__ = ["bounded_handoff_output", "markdown_deliverable_output"]
