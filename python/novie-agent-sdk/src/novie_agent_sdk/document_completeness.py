"""Fail-closed semantic and mechanical completeness checks for documents."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


_ISSUE_CODES = frozenset({
    "none",
    "cut_off",
    "unfinished_structure",
    "missing_planned_content",
})

SECTION_COMPLETENESS_SCHEMA: dict[str, Any] = {
    "title": "SectionCompletenessReview",
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "complete": {
            "type": "boolean",
            "description": "True only when the section is a complete work product.",
        },
        "issue_code": {
            "type": "string",
            "enum": sorted(_ISSUE_CODES),
        },
        "reason": {
            "type": "string",
            "description": "A concise explanation grounded in the candidate section.",
        },
    },
    "required": ["complete", "issue_code", "reason"],
}


class DocumentCompletenessReviewError(RuntimeError):
    """Raised when completeness cannot be proven through structured review."""

    code = "document_completeness_review_unavailable"
    retryable = True


@dataclass(frozen=True, slots=True)
class SectionCompletenessReview:
    complete: bool
    issue_code: str
    reason: str

    def to_metadata(self) -> dict[str, Any]:
        return {
            "complete": self.complete,
            "issue_code": self.issue_code,
            "reason": self.reason,
        }


async def review_section_completeness(
    llm_facade: Any,
    *,
    section_title: str,
    section_objective: str,
    markdown: str,
) -> SectionCompletenessReview:
    """Use provider-enforced structured output for semantic completeness."""
    structured = getattr(llm_facade, "structured", None)
    if not callable(structured):
        raise DocumentCompletenessReviewError(
            "document_completeness_review_unavailable: structured LLM facade missing"
        )
    try:
        result = await structured(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Review one document section for completeness. Mark it incomplete "
                        "when it ends mid-thought, leaves a list/option/structure unfinished, "
                        "or does not fulfill the supplied section objective. Do not judge "
                        "writing style, evidence strength, or final punctuation alone."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Section title: {section_title}\n"
                        f"Section objective: {section_objective}\n\n"
                        "Candidate Markdown:\n"
                        f"{markdown}"
                    ),
                },
            ],
            output_schema=SECTION_COMPLETENESS_SCHEMA,
            temperature=0,
            method="json_schema",
            strict=True,
            max_output_tokens=512,
        )
    except Exception as exc:
        raise DocumentCompletenessReviewError(
            "document_completeness_review_unavailable: structured review failed"
        ) from exc

    payload = result.get("structured") if isinstance(result, Mapping) else None
    if not isinstance(payload, Mapping):
        raise DocumentCompletenessReviewError(
            "document_completeness_review_unavailable: structured result missing"
        )
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
    return SectionCompletenessReview(complete, issue_code, reason)


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
    "SectionCompletenessReview",
    "markdown_structure_violation",
    "review_section_completeness",
]
