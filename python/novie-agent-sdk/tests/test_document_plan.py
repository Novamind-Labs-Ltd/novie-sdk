from __future__ import annotations

from novie_agent_sdk.document_plan import (
    ExpansionTask,
    parse_coverage_review,
    parse_document_plan,
    render_document_plan,
    section_indexes_from_tasks,
    source_ref_count,
)


def test_parse_document_plan_accepts_json_with_surrounding_text() -> None:
    plan = parse_document_plan(
        """
        Here is the plan:
        {
          "artifact_type": "management_report",
          "audience": ["executive"],
          "depth": "deep",
          "evidence_richness": "high",
          "section_contracts": [
            {"index": 10, "heading": "Executive Summary", "importance": "critical", "source_refs": ["artifact://a1"]},
            {"index": 20, "heading": "Market Context", "importance": "high", "source_refs": ["artifact://a2"]},
            {"index": 30, "heading": "Risks", "importance": "medium", "source_refs": ["artifact://a1"]}
          ],
          "completion_criteria": ["critical sections pass"]
        }
        """
    )

    assert plan is not None
    assert [section.index for section in plan.section_contracts] == [1, 2, 3]
    assert plan.section_contracts[0].importance == "critical"
    assert source_ref_count(plan) == 2
    assert "Executive Summary" in render_document_plan(plan)


def test_parse_document_plan_accepts_legacy_sections_alias() -> None:
    plan = parse_document_plan(
        """
        {
          "sections": [
            {"index": 1, "heading": "Alpha"},
            {"index": 2, "heading": "Beta"},
            {"index": 3, "heading": "Gamma"}
          ]
        }
        """,
        artifact_type="research_report",
    )

    assert plan is not None
    assert plan.artifact_type == "research_report"
    assert [section.heading for section in plan.section_contracts] == ["Alpha", "Beta", "Gamma"]


def test_parse_document_plan_rejects_too_small_or_invalid_plans() -> None:
    assert parse_document_plan("not json") is None
    assert (
        parse_document_plan(
            '{"section_contracts": [{"index": 1, "heading": "Only one"}]}'
        )
        is None
    )


def test_parse_coverage_review_drops_expansion_tasks_on_pass() -> None:
    review = parse_coverage_review(
        """
        {
          "overall_status": "pass",
          "section_scores": [{"section_index": 1, "status": "pass", "coverage_score": 1}],
          "expansion_tasks": [{"section_index": 1, "reason": "ignored"}]
        }
        """
    )

    assert review.overall_status == "pass"
    assert review.expansion_tasks == []


def test_section_indexes_from_tasks_preserves_first_seen_order() -> None:
    tasks = [
        ExpansionTask(section_index=2),
        ExpansionTask(section_index=1),
        ExpansionTask(section_index=2),
    ]

    assert section_indexes_from_tasks(tasks) == [2, 1]
