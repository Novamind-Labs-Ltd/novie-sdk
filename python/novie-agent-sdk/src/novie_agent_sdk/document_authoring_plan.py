"""ADR-115 immutable execution plans for completion-oriented authoring."""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, replace
from typing import Any, Mapping, Sequence

from .document_authoring_identity import part_identity


def _digest(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _positive_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


@dataclass(frozen=True, slots=True)
class PlannedPart:
    section_id: str
    ordinal: int
    total: int
    objective: str
    objective_digest: str
    evidence_digest: str
    part_identity: str
    target_information_units: int
    max_information_units: int
    plan_revisions: tuple[int, ...] = (1,)

    def with_evidence(self, evidence: Any, *, scope: Mapping[str, Any]) -> "PlannedPart":
        evidence_digest = _digest(evidence)
        identity = part_identity(
            scope=scope,
            section_id=self.section_id,
            objective_digest=self.objective_digest,
            evidence_digest=evidence_digest,
        )
        return replace(
            self,
            evidence_digest=evidence_digest,
            part_identity=identity,
        )


@dataclass(frozen=True, slots=True)
class SectionExecutionPlan:
    section_id: str
    title: str
    objective: str
    target_information_units: int
    max_information_units: int
    parts: tuple[PlannedPart, ...]


@dataclass(frozen=True, slots=True)
class AuthoringExecutionPlan:
    schema: str
    revision: int
    outline_digest: str
    sections: tuple[SectionExecutionPlan, ...]
    max_authoring_llm_calls: int
    max_authoring_compaction_calls: int
    max_section_review_calls: int
    provider_output_ceiling: int | None
    estimated_authoring_seconds: float
    deadline_feasible: bool
    input_digest: str

    @property
    def part_count(self) -> int:
        return sum(len(section.parts) for section in self.sections)

    @property
    def mandatory_call_count(self) -> int:
        return self.part_count + len(self.sections)

    def to_mapping(self) -> dict[str, Any]:
        return asdict(self)


def build_authoring_execution_plan(
    outline: Sequence[Any],
    *,
    scope: Mapping[str, Any],
    context_budget: Mapping[str, Any],
    revision: int = 1,
) -> AuthoringExecutionPlan:
    """Convert the accepted outline into bounded, immutable Planned Parts."""
    max_part_units = _positive_int(
        context_budget.get("max_part_information_units"),
        800,
    )
    ceiling = _positive_int(context_budget.get("max_output_tokens"), 0) or None
    sections: list[SectionExecutionPlan] = []
    outline_data: list[dict[str, Any]] = []
    for section in outline:
        section_id = str(getattr(section, "section_id", "") or "")
        title = str(getattr(section, "title", "") or "")
        objective = str(getattr(section, "objective", "") or title)
        target = _positive_int(
            getattr(section, "target_information_units", None)
            or getattr(section, "min_words", None),
            1,
        )
        declared_max = _positive_int(
            getattr(section, "max_information_units", None),
            0,
        )
        section_max = declared_max if declared_max > target else max(
            target + 1,
            math.ceil(target * 1.25),
        )
        part_total = max(1, math.ceil(section_max / max_part_units))
        part_target = max(1, math.ceil(target / part_total))
        part_max = max(part_target + 1, math.ceil(section_max / part_total))
        parts: list[PlannedPart] = []
        for ordinal in range(1, part_total + 1):
            focused = objective if part_total == 1 else (
                f"{objective} Complete bounded part {ordinal} of {part_total}; "
                "finish its local structures without covering later parts."
            )
            objective_digest = _digest(
                {"section_id": section_id, "ordinal": ordinal, "objective": focused}
            )
            evidence_digest = _digest({"pending_evidence_scope": section_id})
            parts.append(
                PlannedPart(
                    section_id=section_id,
                    ordinal=ordinal,
                    total=part_total,
                    objective=focused,
                    objective_digest=objective_digest,
                    evidence_digest=evidence_digest,
                    part_identity=part_identity(
                        scope=scope,
                        section_id=section_id,
                        objective_digest=objective_digest,
                        evidence_digest=evidence_digest,
                    ),
                    target_information_units=part_target,
                    max_information_units=part_max,
                    plan_revisions=(revision,),
                )
            )
        sections.append(
            SectionExecutionPlan(
                section_id=section_id,
                title=title,
                objective=objective,
                target_information_units=target,
                max_information_units=section_max,
                parts=tuple(parts),
            )
        )
        outline_data.append(
            {
                "section_id": section_id,
                "title": title,
                "objective": objective,
                "target_information_units": target,
                "max_information_units": section_max,
            }
        )
    part_count = sum(len(item.parts) for item in sections)
    section_count = len(sections)
    compaction_calls = _positive_int(
        context_budget.get("max_authoring_compaction_calls"),
        max(1, section_count // 2),
    )
    review_calls = _positive_int(
        context_budget.get("max_section_review_calls"),
        max(1, section_count * 2),
    )
    # Base parts + one bounded recovery part per section + up to two semantic
    # reviews per section + pressure-triggered compactions.
    default_max_calls = part_count + (section_count * 3) + compaction_calls
    max_calls = _positive_int(
        context_budget.get("max_authoring_llm_calls"),
        default_max_calls,
    )
    if max_calls < part_count + section_count:
        max_calls = part_count + section_count
    call_p95_seconds = _positive_int(
        context_budget.get("authoring_call_p95_seconds"),
        30,
    )
    evidence_p95_seconds = _positive_int(
        context_budget.get("authoring_evidence_p95_seconds"),
        5,
    )
    estimated_seconds = float(
        (part_count + section_count) * call_p95_seconds
        + section_count * evidence_p95_seconds
    )
    wall_clock_seconds = _positive_int(
        context_budget.get("max_wall_clock_seconds"),
        0,
    )
    deadline_feasible = (
        wall_clock_seconds <= 0 or estimated_seconds <= wall_clock_seconds
    )
    outline_digest = _digest(outline_data)
    input_digest = _digest(
        {
            "outline_digest": outline_digest,
            "scope": dict(scope),
            "provider_output_ceiling": ceiling,
            "max_part_units": max_part_units,
        }
    )
    return AuthoringExecutionPlan(
        schema="authoring-execution-plan.v1",
        revision=revision,
        outline_digest=outline_digest,
        sections=tuple(sections),
        max_authoring_llm_calls=max_calls,
        max_authoring_compaction_calls=compaction_calls,
        max_section_review_calls=review_calls,
        provider_output_ceiling=ceiling,
        estimated_authoring_seconds=estimated_seconds,
        deadline_feasible=deadline_feasible,
        input_digest=input_digest,
    )


__all__ = [
    "AuthoringExecutionPlan",
    "PlannedPart",
    "SectionExecutionPlan",
    "build_authoring_execution_plan",
]
