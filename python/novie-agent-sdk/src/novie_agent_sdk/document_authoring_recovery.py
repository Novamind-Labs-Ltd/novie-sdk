"""Coverage cursor and bounded repartitioning for ADR-115 recovery."""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

from .document_authoring_identity import part_identity
from .document_authoring_plan import PlannedPart


def _digest(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str).encode()
    return hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True, slots=True)
class SectionCoverageCursor:
    section_id: str
    accepted_part_identities: tuple[str, ...] = ()
    covered_objective_digests: tuple[str, ...] = ()
    remaining_objective_digests: tuple[str, ...] = ()
    next_part_ordinal: int = 1
    blocking_issue_code: str = ""
    plan_revision: int = 1

    def accept(self, part: PlannedPart) -> "SectionCoverageCursor":
        accepted = tuple(dict.fromkeys((*self.accepted_part_identities, part.part_identity)))
        covered = tuple(dict.fromkeys((*self.covered_objective_digests, part.objective_digest)))
        remaining = tuple(
            item for item in self.remaining_objective_digests if item != part.objective_digest
        )
        return replace(
            self,
            accepted_part_identities=accepted,
            covered_objective_digests=covered,
            remaining_objective_digests=remaining,
            next_part_ordinal=part.ordinal + 1,
            blocking_issue_code="",
        )


def repartition_remaining_parts(
    parts: Sequence[PlannedPart],
    *,
    failed_index: int,
    revision: int,
    scope: Mapping[str, Any],
    min_part_information_units: int,
) -> tuple[PlannedPart, ...]:
    """Split one failed part and deterministically renumber only remaining work."""
    if failed_index < 0 or failed_index >= len(parts):
        raise IndexError("failed_index is outside the Planned Part sequence")
    failed = parts[failed_index]
    maximum = failed.max_information_units
    if maximum <= 2:
        raise ValueError("document_part_completion_exhausted")
    # The configured minimum is a planning preference, not proof that a small
    # final allowance is unsplittable. Adapt it for the bounded recovery tail
    # so a 3..(2 * configured minimum) unit remainder can still converge
    # without exceeding the document-wide hard maximum.
    effective_minimum = min(
        max(1, min_part_information_units),
        max(1, maximum // 2),
    )
    left_max = max(effective_minimum, maximum // 2)
    right_max = maximum - left_max
    if right_max < effective_minimum:
        right_max = effective_minimum
        left_max = maximum - right_max
    objectives = (
        f"{failed.objective} Complete the first remaining bounded portion.",
        f"{failed.objective} Complete the second remaining bounded portion.",
    )
    rebuilt: list[PlannedPart] = list(parts[:failed_index])
    remaining_specs: list[tuple[str, int, str]] = [
        (objectives[0], left_max, failed.evidence_digest),
        (objectives[1], right_max, failed.evidence_digest),
    ]
    remaining_specs.extend(
        (item.objective, item.max_information_units, item.evidence_digest)
        for item in parts[failed_index + 1 :]
    )
    total = len(rebuilt) + len(remaining_specs)
    for ordinal, (objective, maximum, evidence_digest) in enumerate(
        remaining_specs,
        start=len(rebuilt) + 1,
    ):
        objective_digest = _digest(
            {
                "section_id": failed.section_id,
                "ordinal": ordinal,
                "objective": objective,
            }
        )
        target = max(1, min(maximum - 1, math.ceil(maximum * 0.8)))
        rebuilt.append(
            PlannedPart(
                section_id=failed.section_id,
                ordinal=ordinal,
                total=total,
                objective=objective,
                objective_digest=objective_digest,
                evidence_digest=evidence_digest,
                part_identity=part_identity(
                    scope=scope,
                    section_id=failed.section_id,
                    objective_digest=objective_digest,
                    evidence_digest=evidence_digest,
                ),
                target_information_units=target,
                max_information_units=maximum,
                plan_revisions=(revision,),
            )
        )
    return tuple(replace(item, total=total) for item in rebuilt)


def coalesce_remaining_parts(
    parts: Sequence[PlannedPart],
    *,
    failed_index: int,
    revision: int,
    scope: Mapping[str, Any],
) -> tuple[PlannedPart, ...]:
    """Merge unfinished objectives when smaller parts worsen overlong output."""
    if failed_index < 0 or failed_index >= len(parts):
        raise IndexError("failed_index is outside the Planned Part sequence")
    unfinished = tuple(parts[failed_index:])
    if len(unfinished) < 2:
        raise ValueError("document_part_coalescing_not_applicable")
    prefix = list(parts[:failed_index])
    failed = unfinished[0]
    objective = (
        "Complete all of these remaining objectives together in one concise, "
        "bounded part. Cover each objective once and do not add background:\n- "
        + "\n- ".join(item.objective for item in unfinished)
    )
    ordinal = len(prefix) + 1
    total = ordinal
    objective_digest = _digest(
        {
            "section_id": failed.section_id,
            "ordinal": ordinal,
            "objective": objective,
            "coalesced_objective_digests": [
                item.objective_digest for item in unfinished
            ],
        }
    )
    maximum = sum(item.max_information_units for item in unfinished)
    target = max(
        1,
        min(
            maximum - 1,
            sum(item.target_information_units for item in unfinished),
        ),
    )
    merged = PlannedPart(
        section_id=failed.section_id,
        ordinal=ordinal,
        total=total,
        objective=objective,
        objective_digest=objective_digest,
        evidence_digest=failed.evidence_digest,
        part_identity=part_identity(
            scope=scope,
            section_id=failed.section_id,
            objective_digest=objective_digest,
            evidence_digest=failed.evidence_digest,
        ),
        target_information_units=target,
        max_information_units=maximum,
        plan_revisions=(revision,),
    )
    rebuilt = [replace(item, total=total) for item in prefix]
    rebuilt.append(merged)
    return tuple(rebuilt)


__all__ = [
    "SectionCoverageCursor",
    "coalesce_remaining_parts",
    "repartition_remaining_parts",
]
