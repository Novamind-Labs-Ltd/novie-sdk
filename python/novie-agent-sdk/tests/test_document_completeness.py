from __future__ import annotations

from typing import Any

import pytest

from novie_agent_sdk.document_completeness import (
    DocumentCompletenessReviewError,
    markdown_structure_violation,
    review_section_completeness,
)


class _StructuredLlm:
    def __init__(self, payload: Any) -> None:
        self.payload = payload
        self.calls: list[dict[str, Any]] = []

    async def structured(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        return {"structured": self.payload}


class _FailingStructuredLlm:
    async def structured(self, **_kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("provider returned no structured output")


@pytest.mark.asyncio
async def test_review_disables_reasoning_and_scales_output_for_coverage() -> None:
    point_ids = [f"point-{index}" for index in range(10)]
    llm = _StructuredLlm(
        {
            "structure_complete": True,
            "issue_code": "none",
            "coverage": [
                {"point_id": point_id, "covered": True, "evidence": "Present."}
                for point_id in point_ids
            ],
            "reason": "Complete.",
        }
    )

    await review_section_completeness(
        llm,
        section_title="Review",
        section_objective="Cover every point.",
        required_points=tuple(
            {"point_id": point_id, "requirement": f"Cover {point_id}."}
            for point_id in point_ids
        ),
        markdown="## Review\n\nComplete content.",
    )

    assert llm.calls[0]["reasoning_mode"] == "disabled"
    assert llm.calls[0]["reasoning_workload"] == "review"
    assert llm.calls[0]["max_output_tokens"] == 1856


@pytest.mark.asyncio
async def test_review_provider_failure_degrades_without_aborting_document() -> None:
    review = await review_section_completeness(
        _FailingStructuredLlm(),
        section_title="Review",
        section_objective="Cover the objective.",
        markdown="## Review\n\nMechanically complete content.",
    )

    assert review.complete is False
    assert review.reliable is False
    assert review.structure_complete is True
    assert "unavailable" in review.reason


@pytest.mark.asyncio
async def test_review_section_completeness_accepts_valid_strict_shape() -> None:
    review = await review_section_completeness(
        _StructuredLlm(
            {
                "complete": False,
                "issue_code": "cut_off",
                "reason": "The option ends mid-thought.",
            }
        ),
        section_title="Open questions",
        section_objective="Document all unresolved choices.",
        markdown="## Open questions\n\nOption B requires provider support",
    )

    assert review.complete is False
    assert review.issue_code == "cut_off"


@pytest.mark.asyncio
async def test_review_section_completeness_rejects_contradictory_shape() -> None:
    with pytest.raises(DocumentCompletenessReviewError):
        await review_section_completeness(
            _StructuredLlm(
                {
                    "complete": True,
                    "issue_code": "cut_off",
                    "reason": "Contradictory.",
                }
            ),
            section_title="Open questions",
            section_objective="Document all unresolved choices.",
            markdown="## Open questions\n\nBody.",
        )


@pytest.mark.asyncio
async def test_unfinished_structure_remains_fail_closed_at_sentence_boundary() -> None:
    semantic_gap = await review_section_completeness(
        _StructuredLlm(
            {
                "complete": False,
                "issue_code": "unfinished_structure",
                "reason": "The section may leave a structure open.",
            }
        ),
        section_title="Recommendation",
        section_objective="State the complete recommendation.",
        markdown="## Recommendation\n\n三项建议均已完整说明。",
    )
    still_open = await review_section_completeness(
        _StructuredLlm(
            {
                "complete": False,
                "issue_code": "unfinished_structure",
                "reason": "A fenced block remains open.",
            }
        ),
        section_title="Recommendation",
        section_objective="State the complete recommendation.",
        markdown="## Recommendation\n\n```text\nunfinished",
    )

    assert semantic_gap.complete is False
    assert semantic_gap.issue_code == "unfinished_structure"
    assert still_open.complete is False


@pytest.mark.asyncio
async def test_coverage_checklist_derives_incomplete_without_trusting_overall_reason() -> None:
    review = await review_section_completeness(
        _StructuredLlm(
            {
                "structure_complete": True,
                "issue_code": "none",
                "coverage": [
                    {
                        "point_id": "impact.reliability",
                        "covered": True,
                        "evidence": "Delivery reliability is discussed.",
                    },
                    {
                        "point_id": "impact.efficiency",
                        "covered": False,
                        "evidence": "Team efficiency is absent.",
                    },
                ],
                "reason": "The prose is syntactically closed.",
            }
        ),
        section_title="Business impact",
        section_objective="Assess reliability and efficiency.",
        required_points=(
            {
                "point_id": "impact.reliability",
                "requirement": "Assess delivery reliability.",
            },
            {
                "point_id": "impact.efficiency",
                "requirement": "Assess team efficiency.",
            },
        ),
        markdown="## Business impact\n\nDelivery is less predictable.",
    )

    assert review.complete is False
    assert review.issue_code == "missing_planned_content"


@pytest.mark.asyncio
async def test_coverage_checklist_treats_missing_point_as_local_gap() -> None:
    review = await review_section_completeness(
        _StructuredLlm(
            {
                "structure_complete": True,
                "issue_code": "none",
                "coverage": [
                    {
                        "point_id": "impact.reliability",
                        "covered": True,
                        "evidence": "Present.",
                    }
                ],
                "reason": "Only one result was returned.",
            }
        ),
        section_title="Business impact",
        section_objective="Assess reliability and efficiency.",
        required_points=(
            {
                "point_id": "impact.reliability",
                "requirement": "Assess delivery reliability.",
            },
            {
                "point_id": "impact.efficiency",
                "requirement": "Assess team efficiency.",
            },
        ),
        markdown="## Business impact\n\nDelivery is less predictable.",
    )

    assert review.complete is False
    assert review.issue_code == "missing_planned_content"
    assert review.coverage[1] == {
        "point_id": "impact.efficiency",
        "covered": False,
        "evidence": "",
    }
    assert review.reliable is False


@pytest.mark.asyncio
async def test_coverage_checklist_treats_invalid_entries_as_local_gaps() -> None:
    review = await review_section_completeness(
        _StructuredLlm(
            {
                "structure_complete": True,
                "issue_code": "none",
                "coverage": [
                    {
                        "point_id": "impact.reliability",
                        "covered": "yes",
                        "evidence": "Malformed verdict.",
                    },
                    {
                        "point_id": "unexpected",
                        "covered": True,
                        "evidence": "Unknown point.",
                    },
                ],
                "reason": "",
            }
        ),
        section_title="Business impact",
        section_objective="Assess reliability.",
        required_points=(
            {
                "point_id": "impact.reliability",
                "requirement": "Assess delivery reliability.",
            },
        ),
        markdown="## Business impact\n\nDelivery is less predictable.",
    )

    assert review.complete is False
    assert review.issue_code == "missing_planned_content"
    assert review.coverage[0]["covered"] is False
    assert review.reliable is False


@pytest.mark.asyncio
async def test_coverage_checklist_marks_unsupported_negative_as_unreliable() -> None:
    review = await review_section_completeness(
        _StructuredLlm(
            {
                "structure_complete": True,
                "issue_code": "missing_planned_content",
                "coverage": [
                    {
                        "point_id": "summary.goal",
                        "covered": False,
                        "evidence": "",
                    }
                ],
                "reason": "The section otherwise appears complete.",
            }
        ),
        section_title="Summary",
        section_objective="State the goal.",
        required_points=(
            {"point_id": "summary.goal", "requirement": "State the goal."},
        ),
        markdown="## Summary\n\nThe goal is stable delivery.",
    )

    assert review.complete is False
    assert review.issue_code == "missing_planned_content"
    assert review.reliable is False


@pytest.mark.asyncio
async def test_coverage_checklist_allows_empty_non_authoritative_explanations() -> None:
    review = await review_section_completeness(
        _StructuredLlm(
            {
                "structure_complete": True,
                "issue_code": "none",
                "coverage": [
                    {
                        "point_id": "summary.goal",
                        "covered": True,
                        "evidence": "",
                    }
                ],
                "reason": "",
            }
        ),
        section_title="Summary",
        section_objective="State the goal.",
        required_points=(
            {"point_id": "summary.goal", "requirement": "State the goal."},
        ),
        markdown="## Summary\n\nThe goal is stable delivery.",
    )

    assert review.complete is True
    assert review.reason == "Coverage checklist reviewed."


def test_markdown_structure_violation_detects_unclosed_fence() -> None:
    assert markdown_structure_violation("## Example\n\n```python\nprint('x')") == (
        "markdown_structure_unclosed"
    )
    assert markdown_structure_violation("## Example\n\n```python\nprint('x')\n```") is None
    assert markdown_structure_violation("## Example\n\n```python\n~~~\n```") is None
    assert markdown_structure_violation("## Example\n\n```python\n``") == (
        "markdown_structure_unclosed"
    )
