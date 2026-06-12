"""Capability contracts for document-style external agents."""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .skill_contracts import SkillContractResolver, SkillRuntimeContract


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


@dataclass(frozen=True)
class DocumentRuntimeProfile:
    """Resolved runtime profile for a skill-driven document agent.

    This intentionally stays coarse-grained: agent packages own their domain
    registry and artifact schemas, while the SDK owns the shared capability
    fail-closed behavior and optional skill runtime contract resolution.
    """

    capability_id: str
    spec: Any
    mode: str
    phase: str
    artifact_type: str
    artifact_family: str
    skill_contract: SkillRuntimeContract | None = None


def _capability_id_from_inputs(
    inputs: Mapping[str, Any] | None,
    explicit: str | None,
) -> str | None:
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip()
    if not isinstance(inputs, Mapping):
        return None
    direct = inputs.get("capability_id")
    if isinstance(direct, str) and direct.strip():
        return direct.strip()
    grants = inputs.get("capability_grants")
    if isinstance(grants, list):
        for item in grants:
            if not isinstance(item, Mapping):
                continue
            candidate = item.get("capability_id")
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()
    return None


def resolve_document_runtime_profile(
    *,
    agent_name: str,
    inputs: Mapping[str, Any] | None = None,
    capability_id: str | None = None,
    spec: Any | None = None,
    resolve_capability: Callable[[str | None], Any | None] | None = None,
    skill_sources: Sequence[str | Path] | None = None,
    require_skill_contract: bool = False,
) -> DocumentRuntimeProfile:
    """Resolve a document agent's capability profile and skill contract.

    ``resolve_capability`` remains agent-owned so domain packages keep control
    of their registries and aliases. The SDK centralizes the shared runtime
    contract: ``capability_id`` is required, unknown IDs fail closed, and skill
    runtime contracts are resolved relative to the capability package root.
    """
    label = str(agent_name or "document_agent")
    resolved_id = _capability_id_from_inputs(inputs, capability_id)
    if spec is None:
        if resolved_id is None:
            raise RuntimeError(
                f"{label} capability_id is required for skill-driven runtime"
            )
        if resolve_capability is None:
            raise RuntimeError(f"{label} resolve_capability is required")
        spec = resolve_capability(resolved_id)
        if spec is None:
            raise RuntimeError(f"unknown {label} capability_id: {resolved_id}")
    else:
        resolved_id = str(resolved_id or getattr(spec, "capability_id", "") or "")
        if not resolved_id:
            raise RuntimeError(
                f"{label} capability_id is required for skill-driven runtime"
            )

    sources = list(
        skill_sources
        if skill_sources is not None
        else (getattr(spec, "skill_sources", None) or ())
    )
    skill_contract: SkillRuntimeContract | None = None
    if sources or require_skill_contract:
        resolver = SkillContractResolver(root_dir=getattr(spec, "package_root", None))
        skill_contract = resolver.resolve(sources, required=require_skill_contract)

    return DocumentRuntimeProfile(
        capability_id=resolved_id,
        spec=spec,
        mode=str(getattr(spec, "mode", "") or ""),
        phase=str(getattr(spec, "phase", "") or ""),
        artifact_type=str(getattr(spec, "artifact_type", "") or ""),
        artifact_family=str(getattr(spec, "artifact_family", "") or ""),
        skill_contract=skill_contract,
    )


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
    "DocumentRuntimeProfile",
    "resolve_document_agent_input",
    "resolve_document_runtime_profile",
]
