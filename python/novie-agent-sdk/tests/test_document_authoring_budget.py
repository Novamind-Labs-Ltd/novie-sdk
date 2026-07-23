from __future__ import annotations

import pytest

from novie_agent_sdk.document_authoring_budget import (
    DocumentAuthoringBudgetExceeded,
    DocumentOutputBudget,
)


def test_document_output_budget_shares_one_limit_across_planned_calls() -> None:
    budget = DocumentOutputBudget.from_limits(
        {"max_output_tokens": 1200, "max_document_output_tokens": 900}
    )

    assert budget.reserve(800, slots_remaining=2) == 450
    assert budget.reserve(800, slots_remaining=1) == 450
    assert budget.remaining_tokens == 0
    with pytest.raises(DocumentAuthoringBudgetExceeded) as exc_info:
        budget.reserve(1)

    assert exc_info.value.code == "document_authoring_output_budget_exceeded"
    assert "document_authoring_output_budget_exceeded" in str(exc_info.value)


def test_provider_call_limit_does_not_create_a_cumulative_document_budget() -> None:
    budget = DocumentOutputBudget.from_limits({"max_output_tokens": 900})

    assert budget.total_tokens is None
    assert budget.remaining_tokens is None
    assert budget.reserve(800) == 800
    assert budget.reserve(800) == 800


def test_profile_document_limit_cannot_raise_the_run_budget() -> None:
    budget = DocumentOutputBudget.from_limits(
        {"max_output_tokens": 1200, "max_document_output_tokens": 1000},
        contract_limit=1500,
    )

    assert budget.total_tokens == 1000
    assert budget.reserve(None) == 1000
