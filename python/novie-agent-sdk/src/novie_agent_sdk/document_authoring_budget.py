"""Run-level output and wall-clock guards for sectioned document authoring."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


class DocumentAuthoringBudgetExceeded(RuntimeError):
    """Raised before a document would exceed its cumulative output allocation."""

    code = "document_authoring_output_budget_exceeded"


class DocumentAuthoringDeadlineExceeded(TimeoutError):
    """Raised when an authoring run reaches its absolute wall-clock deadline."""

    code = "document_authoring_deadline_exceeded"


@dataclass(slots=True)
class DocumentOutputBudget:
    """Allocate a document's output ceiling across its LLM calls.

    ``max_output_tokens`` remains a provider call cap. This object makes the
    effective document cap cumulative and uses the caller's remaining planned
    slots to avoid giving the first section the whole document allowance.
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
        per_call = _positive_int(context_budget.get("max_output_tokens"))
        context_limit = _positive_int(context_budget.get("max_document_output_tokens"))
        # ``max_output_tokens`` is the provider limit for one call.  It must not
        # become the cumulative document limit: sectioned authoring deliberately
        # makes several calls (outline, sections, summaries, and finalization).
        # Only an explicit document limit participates in the run-wide ceiling.
        limits = [value for value in (context_limit, contract_limit) if value]
        total = min(limits) if limits else None
        return cls(
            total_tokens=total,
            per_call_tokens=per_call,
            remaining_tokens=total,
        )

    @property
    def enabled(self) -> bool:
        return self.remaining_tokens is not None

    def reserve(
        self,
        requested_tokens: int | None,
        *,
        slots_remaining: int = 1,
    ) -> int | None:
        """Reserve a bounded completion allowance for one provider call."""
        if self.remaining_tokens is None:
            return requested_tokens or self.per_call_tokens or None
        if self.remaining_tokens <= 0:
            raise DocumentAuthoringBudgetExceeded(
                "document_authoring_output_budget_exceeded: sectioned authoring "
                "exhausted its cumulative output-token budget"
            )
        requested = requested_tokens or self.per_call_tokens or self.remaining_tokens
        fair_share = max(1, self.remaining_tokens // max(1, slots_remaining))
        allocated = min(requested, fair_share, self.remaining_tokens)
        self.remaining_tokens -= allocated
        return allocated

    def metadata(self) -> dict[str, int | None]:
        return {
            "document_output_tokens_total": self.total_tokens,
            "document_output_tokens_remaining": self.remaining_tokens,
        }


def _positive_int(value: Any) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


__all__ = [
    "DocumentAuthoringBudgetExceeded",
    "DocumentAuthoringDeadlineExceeded",
    "DocumentOutputBudget",
]
