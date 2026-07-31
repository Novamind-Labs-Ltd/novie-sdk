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


def _non_negative_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed >= 0 else default


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
    required_points: tuple[str, ...] = ()


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
        required_points = tuple(
            dict.fromkeys(
                str(item).strip()
                for item in getattr(section, "required_points", ())
                if str(item).strip()
            )
        ) or (objective,)
        target = _positive_int(
            getattr(section, "target_information_units", None)
            or getattr(section, "min_words", None),
            1,
        )
        # The requested section length is the model target. Allow a narrow
        # 25% generation tolerance before asking for a bounded rewrite.
        section_max = max(
            target + 1,
            math.ceil(target * 1.25),
        )
        # A section is the retry and acceptance unit.  Keep one planned part per
        # section so "at most three retries" cannot multiply across hidden
        # sub-parts. Provider/output limits are handled by the same whole-section
        # rewrite loop rather than by repartitioning the user's section.
        part_total = 1
        part_target = max(1, math.ceil(target / part_total))
        part_max = max(part_target + 1, math.ceil(section_max / part_total))
        parts: list[PlannedPart] = []
        for ordinal in range(1, part_total + 1):
            assigned = required_points[
                (ordinal - 1) * len(required_points) // part_total:
                ordinal * len(required_points) // part_total
            ]
            if not assigned:
                assigned = (
                    required_points[min(ordinal - 1, len(required_points) - 1)],
                )
            focused = (
                (
                    objective
                    if assigned == (objective,)
                    else objective + "\nRequired coverage: " + "; ".join(assigned)
                )
                if part_total == 1
                else (
                    f"Complete required coverage point(s) {ordinal} of "
                    f"{part_total}: "
                    + "; ".join(assigned)
                    + ". Finish local structures without covering later points."
                )
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
                required_points=required_points,
            )
        )
        outline_data.append(
            {
                "section_id": section_id,
                "title": title,
                "objective": objective,
                "target_information_units": target,
                "max_information_units": section_max,
                "required_points": list(required_points),
            }
        )
    part_count = sum(len(item.parts) for item in sections)
    section_count = len(sections)
    compaction_calls = _positive_int(
        context_budget.get("max_authoring_compaction_calls"),
        max(1, section_count // 2),
    )
    recovery_rounds = 0
    seam_review_calls = (
        max(0, section_count - 1)
        if bool(context_budget.get("enable_document_seam_review"))
        else 0
    )
    final_tail_review_calls = (
        1 if bool(context_budget.get("enable_document_seam_review")) else 0
    )
    review_calls = _positive_int(
        context_budget.get("max_section_review_calls"),
        max(
            1,
            section_count * (1 + recovery_rounds)
            + seam_review_calls
            + final_tail_review_calls,
        ),
    )
    continuation_calls = 0
    compression_calls = _positive_int(
        context_budget.get("max_part_compressions"),
        3,
    )
    empty_retry_calls = _non_negative_int(
        context_budget.get("max_empty_part_retries"),
        1,
    )
    # Each planned part may need bounded provider-limit continuation and a
    # bounded closure/compression rewrite. Add one recovery draft plus up to two
    # semantic reviews per section and pressure-triggered compactions. This is a
    # call-count ceiling, not a token reservation; happy-path parts still use
    # one call.
    default_max_calls = (
        part_count
        * (1 + continuation_calls + compression_calls + empty_retry_calls)
        + (section_count * (1 + recovery_rounds))
        + seam_review_calls
        + final_tail_review_calls
        + (
            section_count
            * recovery_rounds
            * (1 + continuation_calls + compression_calls + empty_retry_calls)
        )
        + compaction_calls
    )
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
