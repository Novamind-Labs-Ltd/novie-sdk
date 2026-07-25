from __future__ import annotations

import pytest

from novie_agent_sdk.document_authoring_budget import (
    DocumentAuthoringBudgetExceeded,
    DocumentOutputBudget,
)


def test_each_call_uses_model_top_not_document_fair_share() -> None:
    budget = DocumentOutputBudget.from_limits(
        {"max_output_tokens": 64000, "max_document_output_tokens": 10000}
    )

    assert budget.total_tokens is None
    assert budget.remaining_tokens is None
    assert budget.per_call_tokens == 64000
    # Profile document cap must not shrink or fair-share per-call ceilings.
    assert budget.reserve(64000, slots_remaining=9) == 64000
    assert budget.reserve(64000, slots_remaining=1) == 64000
    assert budget.reserve(None, slots_remaining=3) == 64000


def test_provider_call_limit_does_not_create_a_cumulative_document_budget() -> None:
    budget = DocumentOutputBudget.from_limits({"max_output_tokens": 900})

    assert budget.total_tokens is None
    assert budget.remaining_tokens is None
    assert budget.reserve(800) == 800
    assert budget.reserve(800) == 800


def test_requested_allowance_is_capped_by_model_top() -> None:
    budget = DocumentOutputBudget.from_limits(
        {"max_output_tokens": 1200, "max_document_output_tokens": 1000},
        contract_limit=1500,
    )

    assert budget.total_tokens is None
    assert budget.reserve(None) == 1200
    assert budget.reserve(200) == 200
    assert budget.reserve(5000) == 1200


def test_empty_requested_allowance_fails_closed() -> None:
    budget = DocumentOutputBudget.from_limits({"max_output_tokens": 1200})

    with pytest.raises(DocumentAuthoringBudgetExceeded) as exc_info:
        budget.reserve(0)

    assert exc_info.value.code == "document_authoring_output_budget_exceeded"
