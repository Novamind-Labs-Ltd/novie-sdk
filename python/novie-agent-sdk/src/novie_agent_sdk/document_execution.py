"""Shared execution planning for skill-driven document agents."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from novie_protocol.agents import AgentStreamEvent

from .deliverables import bounded_handoff_output
from .output_contract import fit_text_to_utf8_bytes
from .output_contract import StepRunPolicy
from .skill_contracts import SkillRuntimeContract

DIRECT_AUTHORING = "direct"
EVIDENCE_GRAPH = "evidence_graph"
GRAPH_HANDOFF = "graph_handoff"


@dataclass(frozen=True, slots=True)
class DocumentExecutionPlan:
    """One business-authoring pass for one platform execution step."""

    step_role: str
    preparation: str
    run_graph: bool
    graph_output: str
    run_sectioned_authoring: bool

    @property
    def direct_sectioned_authoring(self) -> bool:
        return self.run_sectioned_authoring and not self.run_graph


def resolve_document_execution_plan(
    *,
    run_policy: StepRunPolicy,
    skill_contract: SkillRuntimeContract | None,
) -> DocumentExecutionPlan:
    """Resolve graph and sectioned-authoring phases from role and skill policy.

    Upstream steps author once through their graph and return a platform-bounded
    handoff. Terminal steps use sectioned authoring exactly once. Skills that
    need tools before authoring can request an evidence-only graph preparation.
    """
    if run_policy.is_upstream_handoff:
        return DocumentExecutionPlan(
            step_role=run_policy.step_role,
            preparation=GRAPH_HANDOFF,
            run_graph=True,
            graph_output="artifact",
            run_sectioned_authoring=False,
        )

    preparation = (
        str(getattr(getattr(skill_contract, "runtime", None), "preparation", "") or "direct")
        .strip()
        .lower()
    )
    if preparation not in {DIRECT_AUTHORING, EVIDENCE_GRAPH}:
        raise RuntimeError(
            "invalid_document_runtime_preparation:"
            f"{preparation}; expected direct or evidence_graph"
        )
    return DocumentExecutionPlan(
        step_role=run_policy.step_role,
        preparation=preparation,
        run_graph=preparation == EVIDENCE_GRAPH,
        graph_output="evidence_dossier" if preparation == EVIDENCE_GRAPH else "",
        run_sectioned_authoring=True,
    )


def build_document_handoff_event(
    *,
    handoff_markdown: str,
    artifact_type: str,
    artifact_family: str,
    capability_id: str,
    run_policy: StepRunPolicy,
    provided_artifact_names: Sequence[str],
    metadata: Mapping[str, Any] | None = None,
    degraded_flags: Sequence[str] = (),
    budget_summary: Mapping[str, Any] | None = None,
) -> AgentStreamEvent:
    """Build the uniform terminal envelope for a graph-authored upstream step."""
    handoff = fit_text_to_utf8_bytes(
        handoff_markdown,
        max_bytes=run_policy.max_output_bytes,
    )
    structured = {
        "kind": "bounded_handoff",
        "handoff_markdown": handoff,
    }
    names = [str(name).strip() for name in provided_artifact_names]
    names.append(str(artifact_type or "").strip())
    provides_artifacts = {
        name: {"structured_output": structured}
        for name in names
        if name
    }
    output = bounded_handoff_output(
        handoff_markdown=handoff,
        artifact_type=artifact_type,
        artifact_family=artifact_family,
        capability_id=capability_id,
        step_role=run_policy.step_role,
        output_contract=run_policy.output_contract,
        metadata=metadata,
        provides_artifacts=provides_artifacts,
        degraded_flags=list(degraded_flags),
        budget_summary=budget_summary,
    )
    return AgentStreamEvent(kind="final", output=output)


__all__ = [
    "DIRECT_AUTHORING",
    "EVIDENCE_GRAPH",
    "GRAPH_HANDOFF",
    "DocumentExecutionPlan",
    "build_document_handoff_event",
    "resolve_document_execution_plan",
]
