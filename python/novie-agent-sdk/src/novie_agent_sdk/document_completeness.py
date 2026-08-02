"""Semantic completeness review with fail-closed mechanical document checks."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


_ISSUE_CODES = frozenset({
    "none",
    "cut_off",
    "unfinished_structure",
    "missing_planned_content",
})

def section_completeness_schema(point_ids: Sequence[str]) -> dict[str, Any]:
    """Build a provider schema whose semantic verdict can be derived in code."""
    return {
        "title": "SectionCompletenessReview",
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "structure_complete": {
                "type": "boolean",
                "description": (
                    "False when a sentence, list, table, code block, or other "
                    "local structure is unfinished."
                ),
            },
            "issue_code": {
                "type": "string",
                "enum": sorted(_ISSUE_CODES),
            },
            "coverage": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "point_id": {
                            "type": "string",
                            "enum": list(point_ids),
                        },
                        "covered": {"type": "boolean"},
                        "evidence": {
                            "type": "string",
                            "description": (
                                "A concise candidate-grounded explanation of "
                                "what is present or missing for this point."
                            ),
                        },
                    },
                    "required": ["point_id", "covered", "evidence"],
                },
            },
            "reason": {
                "type": "string",
                "description": "A concise explanation grounded in the candidate section.",
            },
        },
        "required": ["structure_complete", "issue_code", "coverage", "reason"],
    }


# Public compatibility constant. Runtime calls use the point-aware schema above.
SECTION_COMPLETENESS_SCHEMA: dict[str, Any] = section_completeness_schema(
    ("objective",)
)


class DocumentCompletenessReviewError(RuntimeError):
    """Raised when completeness cannot be proven through structured review."""

    code = "document_completeness_review_unavailable"
    retryable = True


@dataclass(frozen=True, slots=True)
class SectionCompletenessReview:
    complete: bool
    issue_code: str
    reason: str
    coverage: tuple[Mapping[str, Any], ...] = ()
    structure_complete: bool = True
    reliable: bool = True

    def to_metadata(self) -> dict[str, Any]:
        return {
            "complete": self.complete,
            "issue_code": self.issue_code,
            "reason": self.reason,
            "coverage": [dict(item) for item in self.coverage],
            "structure_complete": self.structure_complete,
            "reliable": self.reliable,
        }


def _review_output_token_budget(point_count: int) -> int:
    """Leave enough visible output for one evidence row per coverage point."""
    return min(4096, max(1024, 256 + max(1, point_count) * 160))


def _unavailable_review(
    point_ids: Sequence[str],
    *,
    reason: str,
) -> SectionCompletenessReview:
    return SectionCompletenessReview(
        complete=False,
        issue_code="missing_planned_content",
        reason=reason,
        coverage=tuple(
            {"point_id": point_id, "covered": False, "evidence": ""}
            for point_id in point_ids
        ),
        structure_complete=True,
        reliable=False,
    )


async def review_section_completeness(
    llm_facade: Any,
    *,
    section_title: str,
    section_objective: str,
    required_points: Sequence[Mapping[str, str]] = (),
    markdown: str,
) -> SectionCompletenessReview:
    """Use provider-enforced structured output for semantic completeness."""
    structured = getattr(llm_facade, "structured", None)
    if not callable(structured):
        raise DocumentCompletenessReviewError(
            "document_completeness_review_unavailable: structured LLM facade missing"
        )
    points = tuple(
        {
            "point_id": str(item.get("point_id") or "").strip(),
            "requirement": str(item.get("requirement") or "").strip(),
        }
        for item in required_points
        if str(item.get("point_id") or "").strip()
        and str(item.get("requirement") or "").strip()
    )
    if not points:
        points = ({"point_id": "objective", "requirement": section_objective},)
    point_ids = tuple(item["point_id"] for item in points)
    if len(set(point_ids)) != len(point_ids):
        raise DocumentCompletenessReviewError(
            "document_completeness_review_unavailable: duplicate coverage point ids"
        )
    try:
        result = await structured(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Review one document section against an explicit coverage "
                        "checklist. Return exactly one coverage entry for every supplied "
                        "point_id. Judge each point independently from the candidate text. "
                        "Set structure_complete false when the text ends mid-thought or "
                        "leaves a list, option, table, or code block unfinished. Do not "
                        "return an overall complete verdict; the runtime derives it from "
                        "the checklist. Do not judge writing style or evidence strength."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Section title: {section_title}\n"
                        f"Section objective: {section_objective}\n\n"
                        f"Required coverage points: {list(points)}\n\n"
                        "Candidate Markdown:\n"
                        f"{markdown}"
                    ),
                },
            ],
            output_schema=section_completeness_schema(point_ids),
            temperature=0,
            method="json_schema",
            strict=True,
            max_output_tokens=_review_output_token_budget(len(point_ids)),
            reasoning_mode="disabled",
            reasoning_workload="review",
        )
    except Exception as exc:
        return _unavailable_review(
            point_ids,
            reason=(
                "Structured completeness review was unavailable; deterministic "
                f"document checks remain active ({type(exc).__name__})."
            ),
        )

    payload = result.get("structured") if isinstance(result, Mapping) else None
    if not isinstance(payload, Mapping):
        return _unavailable_review(
            point_ids,
            reason=(
                "Structured completeness review returned no usable result; "
                "deterministic document checks remain active."
            ),
        )
    # Compatibility for callers whose single-objective test/provider adapter
    # still returns the ADR-114 shape. Production sectioned authoring always
    # supplies the point-aware schema above; multi-point coverage can never use
    # this path.
    if "complete" in payload and "coverage" not in payload and len(points) == 1:
        complete = payload.get("complete")
        issue_code = str(payload.get("issue_code") or "").strip()
        reason = str(payload.get("reason") or "").strip()
        if (
            not isinstance(complete, bool)
            or issue_code not in _ISSUE_CODES
            or not reason
            or (complete and issue_code != "none")
            or (not complete and issue_code == "none")
        ):
            raise DocumentCompletenessReviewError(
                "document_completeness_review_unavailable: invalid structured result"
            )
        coverage = (
            {
                "point_id": point_ids[0],
                "covered": complete,
                "evidence": reason,
            },
        )
        return SectionCompletenessReview(
            complete,
            issue_code,
            reason,
            coverage,
            complete or issue_code == "missing_planned_content",
        )
    raw_structure_complete = payload.get("structure_complete")
    structure_complete = (
        raw_structure_complete
        if isinstance(raw_structure_complete, bool)
        else False
    )
    issue_code = str(payload.get("issue_code") or "").strip()
    reason = str(payload.get("reason") or "").strip()
    raw_coverage = payload.get("coverage")
    review_shape_degraded = (
        not isinstance(raw_structure_complete, bool)
        or issue_code not in _ISSUE_CODES
        or not isinstance(raw_coverage, list)
    )
    coverage_by_id: dict[str, Mapping[str, Any]] = {}
    duplicate_ids: set[str] = set()
    for item in raw_coverage if isinstance(raw_coverage, list) else ():
        if not isinstance(item, Mapping):
            review_shape_degraded = True
            continue
        point_id = str(item.get("point_id") or "").strip()
        covered = item.get("covered")
        evidence = str(item.get("evidence") or "").strip()
        if point_id not in point_ids:
            review_shape_degraded = True
            continue
        if point_id in coverage_by_id:
            duplicate_ids.add(point_id)
            review_shape_degraded = True
            continue
        if not isinstance(covered, bool):
            covered = False
            review_shape_degraded = True
        # A negative verdict without candidate-grounded evidence is not a
        # usable semantic finding. Treat it as reviewer degradation rather than
        # repeatedly rewriting a section that may already contain the point.
        if covered is False and not evidence:
            review_shape_degraded = True
        coverage_by_id[point_id] = {
            "point_id": point_id,
            "covered": covered,
            "evidence": evidence,
        }
    coverage = [
        (
            {
                "point_id": point_id,
                "covered": False,
                "evidence": "",
            }
            if point_id not in coverage_by_id or point_id in duplicate_ids
            else coverage_by_id[point_id]
        )
        for point_id in point_ids
    ]
    if len(coverage_by_id) != len(point_ids):
        review_shape_degraded = True

    all_covered = all(bool(item["covered"]) for item in coverage)
    complete = structure_complete and all_covered and issue_code == "none"
    if review_shape_degraded:
        complete = False
        issue_code = (
            "unfinished_structure"
            if not structure_complete
            else "missing_planned_content"
        )
    elif not structure_complete and issue_code == "none":
        issue_code = "unfinished_structure"
    elif structure_complete and not all_covered:
        issue_code = "missing_planned_content"
    elif complete:
        issue_code = "none"
    return SectionCompletenessReview(
        complete,
        issue_code,
        reason
        or (
            "Coverage checklist response was incomplete; missing points require recovery."
            if review_shape_degraded
            else "Coverage checklist reviewed."
        ),
        tuple(coverage),
        structure_complete,
        not review_shape_degraded,
    )


def markdown_structure_violation(markdown: str) -> str | None:
    """Return a deterministic blocking defect for unclosed fenced blocks."""
    active_marker = ""
    active_length = 0
    for line in str(markdown or "").splitlines():
        stripped = line.lstrip()
        if active_marker:
            marker_length = len(stripped) - len(stripped.lstrip(active_marker))
            if (
                marker_length >= active_length
                and not stripped[marker_length:].strip()
            ):
                active_marker = ""
                active_length = 0
            continue
        marker = stripped[:1]
        if marker not in {"`", "~"}:
            continue
        marker_length = len(stripped) - len(stripped.lstrip(marker))
        if marker_length >= 3:
            active_marker = marker
            active_length = marker_length
    return "markdown_structure_unclosed" if active_marker else None


__all__ = [
    "DocumentCompletenessReviewError",
    "SECTION_COMPLETENESS_SCHEMA",
    "section_completeness_schema",
    "SectionCompletenessReview",
    "markdown_structure_violation",
    "review_section_completeness",
]
