from __future__ import annotations

import pytest

from novie_agent_sdk.document_authoring_budget import (
    DocumentAuthoringBudgetExceeded,
    DocumentInformationBudget,
    DocumentInformationBudgetExceeded,
    DocumentOutputBudget,
)
from novie_agent_sdk.document_section_parts import (
    _maximum_acceptable_information_units,
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


def test_information_budget_reserves_later_sections_and_tracks_actual_units() -> None:
    outline = [
        type("_Section", (), {"min_words": 100})(),
        type("_Section", (), {"min_words": 100})(),
    ]
    budget = DocumentInformationBudget.from_outline(
        outline,
        requested_minimum=200,
        requested_maximum=260,
    )

    budget.begin_section(resumed_units=0, reserved_future_units=100)
    assert budget.current_section_allowance == 160
    budget.accept_part(140)
    budget.commit_section(142)
    budget.begin_section(resumed_units=0, reserved_future_units=0)

    assert budget.remaining_units == 118
    with pytest.raises(DocumentInformationBudgetExceeded):
        budget.accept_part(119)


def test_information_budget_reserves_section_wrapper_before_parts() -> None:
    outline = [
        type("_Section", (), {"min_words": 100})(),
        type("_Section", (), {"min_words": 100})(),
    ]
    budget = DocumentInformationBudget.from_outline(
        outline,
        requested_minimum=200,
        requested_maximum=260,
    )

    budget.begin_section(
        resumed_units=0,
        reserved_future_units=100,
        section_overhead_units=2,
    )
    assert budget.current_section_allowance == 158
    budget.accept_part(158)
    budget.commit_section(160)

    assert budget.used_units == 160
    assert budget.pending_section_units == 0


def test_information_budget_discards_only_unused_temporary_capacity() -> None:
    budget = DocumentInformationBudget(
        minimum_units=100,
        maximum_units=300,
        used_units=140,
    )
    budget.discard_unused_capacity(80)
    assert budget.maximum_units == 220
    budget.discard_unused_capacity(500)
    assert budget.maximum_units == 140


def test_document_fair_share_never_expands_planned_part_maximum() -> None:
    part = type("_Part", (), {"max_information_units": 80})()
    budget = type("_Budget", (), {"current_section_allowance": 400})()

    assert _maximum_acceptable_information_units(part, budget) == 80


def test_aggregate_budget_can_only_clip_planned_part_maximum() -> None:
    part = type("_Part", (), {"max_information_units": 80})()
    budget = type("_Budget", (), {"current_section_allowance": 40})()

    assert _maximum_acceptable_information_units(part, budget) == 40
