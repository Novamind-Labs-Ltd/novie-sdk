"""Bounded execution-plan revisions for ADR-115 recovery parts."""
from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from typing import Any, Mapping, Sequence

from .document_authoring_identity import part_identity
from .document_authoring_plan import AuthoringExecutionPlan, PlannedPart


def _digest(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def append_recovery_part(
    plan: AuthoringExecutionPlan,
    *,
    section_index: int,
    issue_code: str,
    reason: str,
    scope: Mapping[str, Any],
) -> AuthoringExecutionPlan:
    """Append one bounded missing-objective part without rewriting accepted parts."""
    section = plan.sections[section_index]
    revision = plan.revision + 1
    total = len(section.parts) + 1
    existing = tuple(
        replace(
            part,
            total=total,
            plan_revisions=tuple(dict.fromkeys((*part.plan_revisions, revision))),
        )
        for part in section.parts
    )
    objective = (
        f"Resolve the remaining completeness issue `{issue_code}` for "
        f"{section.title}: {reason}"
    )
    objective_digest = _digest(
        {
            "section_id": section.section_id,
            "ordinal": total,
            "objective": objective,
        }
    )
    evidence_digest = _digest({"pending_evidence_scope": section.section_id})
    recovery = PlannedPart(
        section_id=section.section_id,
        ordinal=total,
        total=total,
        objective=objective,
        objective_digest=objective_digest,
        evidence_digest=evidence_digest,
        part_identity=part_identity(
            scope=scope,
            section_id=section.section_id,
            objective_digest=objective_digest,
            evidence_digest=evidence_digest,
        ),
        target_information_units=max(
            1,
            min(80, max(10, section.target_information_units // 3)),
        ),
        max_information_units=max(
            20,
            min(120, section.max_information_units),
        ),
        plan_revisions=(revision,),
    )
    revised_section = replace(section, parts=(*existing, recovery))
    sections = list(plan.sections)
    sections[section_index] = revised_section
    return replace(plan, revision=revision, sections=tuple(sections))


def merge_executed_section_parts(
    plan: AuthoringExecutionPlan,
    *,
    section_index: int,
    parts: Sequence[PlannedPart],
    revision: int,
) -> AuthoringExecutionPlan:
    """Carry runtime repartitions forward before appending semantic recovery."""
    if section_index < 0 or section_index >= len(plan.sections):
        raise IndexError("section_index is outside the authoring execution plan")
    if not parts:
        raise ValueError("executed section must contain at least one planned part")
    section = plan.sections[section_index]
    if any(part.section_id != section.section_id for part in parts):
        raise ValueError("executed section parts do not match the target section")
    sections = list(plan.sections)
    sections[section_index] = replace(section, parts=tuple(parts))
    return replace(
        plan,
        revision=max(plan.revision, revision),
        sections=tuple(sections),
    )


__all__ = ["append_recovery_part", "merge_executed_section_parts"]
