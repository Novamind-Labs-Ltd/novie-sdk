"""Per-call output and wall-clock guards for sectioned document authoring."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


class DocumentAuthoringBudgetExceeded(RuntimeError):
    """Raised when a call cannot be allocated any output tokens.

    Kept for call-site compatibility. Sectioned authoring no longer maintains a
    document-wide cumulative token pool, so this is unused on the happy path.
    """

    code = "document_authoring_output_budget_exceeded"


class DocumentAuthoringDeadlineExceeded(TimeoutError):
    """Raised when an authoring run reaches its absolute wall-clock deadline."""

    code = "document_authoring_deadline_exceeded"


class DocumentAuthoringCallBudgetExceeded(RuntimeError):
    """Raised before an unplanned authoring LLM call can begin."""

    code = "document_authoring_call_budget_exceeded"


@dataclass(slots=True)
class AuthoringCallBudget:
    """Bound content, review, and compaction calls independently and in total."""

    total_limit: int
    compaction_limit: int
    review_limit: int
    used: int = 0
    compactions_used: int = 0
    reviews_used: int = 0

    def reserve(self, kind: str) -> None:
        if self.used >= self.total_limit:
            raise DocumentAuthoringCallBudgetExceeded(
                "document_authoring_call_budget_exceeded: total"
            )
        if kind == "compaction" and self.compactions_used >= self.compaction_limit:
            raise DocumentAuthoringCallBudgetExceeded(
                "document_authoring_call_budget_exceeded: compaction"
            )
        if kind == "review" and self.reviews_used >= self.review_limit:
            raise DocumentAuthoringCallBudgetExceeded(
                "document_authoring_call_budget_exceeded: review"
            )
        self.used += 1
        if kind == "compaction":
            self.compactions_used += 1
        elif kind == "review":
            self.reviews_used += 1

    def metadata(self) -> dict[str, int]:
        return {
            "authoring_llm_calls_used": self.used,
            "authoring_llm_calls_limit": self.total_limit,
            "authoring_compaction_calls_used": self.compactions_used,
            "authoring_compaction_calls_limit": self.compaction_limit,
            "authoring_review_calls_used": self.reviews_used,
            "authoring_review_calls_limit": self.review_limit,
        }


@dataclass(slots=True)
class DocumentOutputBudget:
    """Resolve the per-call completion ceiling for sectioned authoring.

    Each LLM call should receive the current model's output top
    (``max_output_tokens``). Skill ``max_document_output_tokens`` is a profile
    length hint for prompts/units — not a run-wide token pool to fair-share.
    Hard delivery size stays on ``max_document_output_bytes``.
    """

    total_tokens: int | None
    per_call_tokens: int | None
    remaining_tokens: int | None

    @classmethod
    def from_limits(
        cls,
        context_budget: Mapping[str, Any],
        *,
        contract_limit: int = 0,
    ) -> "DocumentOutputBudget":
        per_call = _positive_int(context_budget.get("max_output_tokens")) or None
        # ponytail: ignore profile/contract token caps for reservation. They used
        # to become a cumulative fair-share pool (medium=10k across outline +
        # N×draft/revise/summary), which starved later sections. Upgrade path if
        # a true document-wide ceiling returns: track actual usage, not reserved
        # max_tokens, and enforce only at finalize.
        _ = contract_limit
        _ = _positive_int(context_budget.get("max_document_output_tokens"))
        return cls(
            total_tokens=None,
            per_call_tokens=per_call,
            remaining_tokens=None,
        )

    @property
    def enabled(self) -> bool:
        return self.per_call_tokens is not None

    def reserve(
        self,
        requested_tokens: int | None,
        *,
        slots_remaining: int = 1,
    ) -> int | None:
        """Return the completion allowance for one provider call.

        ``slots_remaining`` is retained for call-site compatibility and ignored:
        later sections must not be starved to save room for planned siblings.
        """
        _ = slots_remaining
        if requested_tokens is not None:
            requested = _positive_int(requested_tokens)
            if requested <= 0:
                raise DocumentAuthoringBudgetExceeded(
                    "document_authoring_output_budget_exceeded: requested "
                    "output-token allowance is empty"
                )
            if self.per_call_tokens is None:
                return requested
            return min(requested, self.per_call_tokens)
        return self.per_call_tokens

    def metadata(self) -> dict[str, int | None]:
        return {
            "document_output_tokens_total": self.total_tokens,
            "document_output_tokens_remaining": self.remaining_tokens,
            "per_call_output_tokens": self.per_call_tokens,
        }


def _positive_int(value: Any) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


__all__ = [
    "AuthoringCallBudget",
    "DocumentAuthoringCallBudgetExceeded",
    "DocumentAuthoringBudgetExceeded",
    "DocumentAuthoringDeadlineExceeded",
    "DocumentOutputBudget",
]
