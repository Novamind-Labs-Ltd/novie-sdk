"""Adaptive document-planning primitives for document-style agents.

This module owns the shared schema and parse/render helpers for agents that
plan a long-form document before drafting sections. Agent packages still own
their prompt wording, fallback policy, and domain-specific defaults.
"""
from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError

DocumentDepth = Literal["brief", "standard", "comprehensive", "deep"]
EvidenceRichness = Literal["low", "medium", "high"]
SectionImportance = Literal["low", "medium", "high", "critical"]
SectionDepth = Literal["short", "normal", "deep"]
CoverageStatus = Literal["pass", "needs_expansion", "blocked_by_evidence"]


class ExpansionPolicy(BaseModel):
    """How aggressively a document writer may expand weak sections."""

    max_rounds: int = Field(default=1, ge=0, le=3)
    max_sections_per_round: int = Field(default=4, ge=1, le=12)
    stop_when: list[str] = Field(default_factory=list)


class SectionContract(BaseModel):
    index: int = Field(ge=1)
    heading: str = Field(min_length=2)
    role: str = ""
    importance: SectionImportance = "medium"
    required_questions: list[str] = Field(default_factory=list)
    required_evidence_types: list[str] = Field(default_factory=list)
    source_refs: list[str] = Field(default_factory=list)
    expected_depth: SectionDepth = "normal"
    stop_conditions: list[str] = Field(default_factory=list)


class DocumentPlan(BaseModel):
    artifact_type: str = "document"
    audience: list[str] = Field(default_factory=list)
    depth: DocumentDepth = "standard"
    evidence_richness: EvidenceRichness = "medium"
    section_contracts: list[SectionContract] = Field(default_factory=list)
    completion_criteria: list[str] = Field(default_factory=list)
    expansion_policy: ExpansionPolicy = Field(default_factory=ExpansionPolicy)


class SectionCoverageScore(BaseModel):
    section_index: int = Field(ge=1)
    status: CoverageStatus = "pass"
    coverage_score: float = Field(default=1.0, ge=0.0, le=1.0)
    missing_questions: list[str] = Field(default_factory=list)
    missing_evidence: list[str] = Field(default_factory=list)
    notes: str = ""


class ExpansionTask(BaseModel):
    section_index: int = Field(ge=1)
    reason: str = ""
    missing_questions: list[str] = Field(default_factory=list)
    missing_evidence: list[str] = Field(default_factory=list)
    suggested_ref_reads: list[str] = Field(default_factory=list)
    expansion_instruction: str = ""


class CoverageReview(BaseModel):
    overall_status: CoverageStatus = "pass"
    section_scores: list[SectionCoverageScore] = Field(default_factory=list)
    expansion_tasks: list[ExpansionTask] = Field(default_factory=list)


def json_object_from_text(text: str) -> dict[str, Any]:
    """Parse a JSON object, tolerating surrounding prose around the object."""
    stripped = str(text or "").strip()
    if not stripped:
        return {}
    try:
        parsed = json.loads(stripped)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start < 0 or end <= start:
            return {}
        try:
            parsed = json.loads(stripped[start : end + 1])
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}


def normalise_plan_indexes(plan: DocumentPlan) -> DocumentPlan:
    """Rewrite section indexes to match their current order."""
    contracts: list[SectionContract] = []
    for index, contract in enumerate(plan.section_contracts, start=1):
        contracts.append(contract.model_copy(update={"index": index}))
    return plan.model_copy(update={"section_contracts": contracts})


def parse_document_plan(
    text: str,
    *,
    artifact_type: str = "document",
    min_sections: int = 3,
) -> DocumentPlan | None:
    """Parse a structured document plan from model output.

    Accepts either ``section_contracts`` or the older alias ``sections``.
    Returns ``None`` for invalid or too-small plans so callers can fall back.
    """
    payload = json_object_from_text(text)
    if not payload:
        return None
    if "section_contracts" not in payload and isinstance(payload.get("sections"), list):
        payload["section_contracts"] = payload.pop("sections")
    payload.setdefault("artifact_type", artifact_type)
    try:
        plan = DocumentPlan.model_validate(payload)
    except ValidationError:
        return None
    if len(plan.section_contracts) < min_sections:
        return None
    return normalise_plan_indexes(plan)


def render_document_plan(plan: DocumentPlan) -> str:
    lines = [
        "# DocumentPlan",
        f"- artifact_type: {plan.artifact_type}",
        f"- audience: {', '.join(plan.audience) if plan.audience else '(unspecified)'}",
        f"- depth: {plan.depth}",
        f"- evidence_richness: {plan.evidence_richness}",
        "",
        "## Section Contracts",
    ]
    for contract in plan.section_contracts:
        lines.append(
            f"{contract.index}. {contract.heading} "
            f"[importance={contract.importance}, depth={contract.expected_depth}]"
        )
        if contract.role:
            lines.append(f"   role: {contract.role}")
        if contract.required_questions:
            lines.append("   questions: " + "; ".join(contract.required_questions))
        if contract.required_evidence_types:
            lines.append("   evidence: " + "; ".join(contract.required_evidence_types))
        if contract.source_refs:
            lines.append("   refs: " + ", ".join(contract.source_refs))
    if plan.completion_criteria:
        lines.extend(["", "## Completion Criteria"])
        lines.extend(f"- {item}" for item in plan.completion_criteria)
    return "\n".join(lines).strip()


def render_section_contract(contract: SectionContract | None) -> str:
    if contract is None:
        return ""
    lines = [
        f"Section role: {contract.role or '(derive from heading and outline)'}",
        f"Importance: {contract.importance}",
        f"Expected depth: {contract.expected_depth}",
    ]
    if contract.required_questions:
        lines.append("Required questions:\n" + "\n".join(f"- {q}" for q in contract.required_questions))
    if contract.required_evidence_types:
        lines.append(
            "Required evidence types:\n"
            + "\n".join(f"- {item}" for item in contract.required_evidence_types)
        )
    if contract.stop_conditions:
        lines.append("Stop conditions:\n" + "\n".join(f"- {item}" for item in contract.stop_conditions))
    if contract.source_refs:
        lines.append("Candidate source refs:\n" + "\n".join(f"- {ref}" for ref in contract.source_refs))
    return "\n".join(lines)


def parse_coverage_review(text: str) -> CoverageReview:
    payload = json_object_from_text(text)
    if not payload:
        return CoverageReview(overall_status="pass")
    try:
        review = CoverageReview.model_validate(payload)
    except ValidationError:
        return CoverageReview(overall_status="pass")
    if review.overall_status == "pass":
        return review.model_copy(update={"expansion_tasks": []})
    tasks = [task for task in review.expansion_tasks if task.section_index >= 1]
    return review.model_copy(update={"expansion_tasks": tasks})


def render_coverage_review(review: CoverageReview) -> str:
    lines = ["# CoverageReview", f"- overall_status: {review.overall_status}"]
    if review.section_scores:
        lines.append("\n## Section Scores")
        for score in review.section_scores:
            lines.append(
                f"- s{score.section_index}: {score.status} "
                f"score={score.coverage_score:.2f}"
            )
            details = score.missing_questions + score.missing_evidence
            if details:
                lines.append("  gaps: " + "; ".join(details))
    if review.expansion_tasks:
        lines.append("\n## Expansion Tasks")
        for task in review.expansion_tasks:
            lines.append(f"- s{task.section_index}: {task.reason or task.expansion_instruction}")
    return "\n".join(lines).strip()


def render_expansion_task(task: ExpansionTask) -> str:
    lines = [f"# ExpansionTask s{task.section_index}"]
    if task.reason:
        lines.append(f"Reason: {task.reason}")
    if task.missing_questions:
        lines.append("Missing questions:\n" + "\n".join(f"- {q}" for q in task.missing_questions))
    if task.missing_evidence:
        lines.append("Missing evidence:\n" + "\n".join(f"- {item}" for item in task.missing_evidence))
    if task.suggested_ref_reads:
        lines.append("Suggested refs:\n" + "\n".join(f"- {ref}" for ref in task.suggested_ref_reads))
    if task.expansion_instruction:
        lines.append(f"Instruction: {task.expansion_instruction}")
    return "\n".join(lines).strip()


def source_ref_count(plan: DocumentPlan) -> int:
    refs: set[str] = set()
    for contract in plan.section_contracts:
        refs.update(ref for ref in contract.source_refs if ref)
    return len(refs)


def section_indexes_from_tasks(tasks: list[ExpansionTask]) -> list[int]:
    seen: set[int] = set()
    out: list[int] = []
    for task in tasks:
        if task.section_index in seen:
            continue
        seen.add(task.section_index)
        out.append(task.section_index)
    return out


__all__ = [
    "CoverageReview",
    "CoverageStatus",
    "DocumentDepth",
    "DocumentPlan",
    "EvidenceRichness",
    "ExpansionPolicy",
    "ExpansionTask",
    "SectionContract",
    "SectionCoverageScore",
    "SectionDepth",
    "SectionImportance",
    "json_object_from_text",
    "normalise_plan_indexes",
    "parse_coverage_review",
    "parse_document_plan",
    "render_coverage_review",
    "render_document_plan",
    "render_expansion_task",
    "render_section_contract",
    "section_indexes_from_tasks",
    "source_ref_count",
]
