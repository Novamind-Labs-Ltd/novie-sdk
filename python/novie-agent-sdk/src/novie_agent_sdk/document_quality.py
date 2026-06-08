"""Reusable quality-loop result helpers for document-style agents."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class DocumentQualityOutcome:
    status: str = "not_run"
    checks_passed: bool = False
    revision_rounds: int = 0
    final_review_passed: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_metadata(self) -> dict[str, Any]:
        return {
            "quality_status": self.status,
            "quality_checks_passed": self.checks_passed,
            "revision_rounds": self.revision_rounds,
            "quality_final_review_passed": self.final_review_passed,
            **dict(self.metadata),
        }


@dataclass(frozen=True)
class DocumentQualityLoopResult:
    narrative: str
    outcome: DocumentQualityOutcome
    degraded_codes: list[str] = field(default_factory=list)


def skipped_quality_result(
    narrative: str,
    *,
    reason: str,
    metadata: dict[str, Any] | None = None,
) -> DocumentQualityLoopResult:
    """Represent an intentionally skipped quality loop in a standard shape."""
    outcome = DocumentQualityOutcome(
        status="skipped",
        checks_passed=True,
        revision_rounds=0,
        final_review_passed=True,
        metadata={"quality_loop_skipped": True, "quality_skip_reason": reason, **dict(metadata or {})},
    )
    return DocumentQualityLoopResult(narrative=narrative, outcome=outcome)


__all__ = [
    "DocumentQualityLoopResult",
    "DocumentQualityOutcome",
    "skipped_quality_result",
]
