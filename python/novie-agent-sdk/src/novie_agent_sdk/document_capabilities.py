"""Capability contracts for document-style external agents."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class DocumentCapabilitySpec:
    """Runtime-facing capability contract for document agents.

    Agent packages own their registries and domain-specific values. The SDK
    owns the shared shape so PM, analyst, architect, and future document agents
    interpret platform manifest metadata consistently.
    """

    capability_id: str
    skill_sources: list[str]
    mode: str
    phase: str
    artifact_type: str
    artifact_family: str
    package_root: Path
    consumes: tuple[str, ...] = ("task_brief",)
    consumes_strict: tuple[str, ...] = ()
    optional_consumes: tuple[str, ...] = ()
    provides: tuple[str, ...] = ()
    artifact_access: str = "summary_then_fetch"
    synthesis_path: bool = False
    research_track: str | None = None
    side_effect_policy: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DocumentAgentInput:
    """Prompt-facing input envelope resolved from a capability contract."""

    brief: dict[str, Any]
    upstream: dict[str, Any]
    artifact_access: str
    uses_upstream_summary: bool


def resolve_document_agent_input(
    spec: DocumentCapabilitySpec | Any | None,
    *,
    brief: dict[str, Any],
    upstream: dict[str, Any],
) -> DocumentAgentInput:
    """Resolve artifact-access semantics into prompt input.

    Dependency strictness remains platform-enforced. This helper only decides
    what summary context the agent receives before it fetches exact artifacts.
    """
    artifact_access = str(getattr(spec, "artifact_access", "summary_then_fetch") or "")
    if artifact_access == "none":
        return DocumentAgentInput(
            brief=brief,
            upstream={},
            artifact_access=artifact_access,
            uses_upstream_summary=False,
        )
    return DocumentAgentInput(
        brief=brief,
        upstream=upstream,
        artifact_access=artifact_access or "summary_then_fetch",
        uses_upstream_summary=bool(upstream),
    )


__all__ = [
    "DocumentAgentInput",
    "DocumentCapabilitySpec",
    "resolve_document_agent_input",
]
