from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from novie_agent_sdk.document_authoring_compaction import (
    AUTHORING_CONTEXT_SUMMARY_SCHEMA,
    compact_authoring_context,
)
from novie_agent_sdk.document_authoring_budget import (
    AuthoringCallBudget,
    DocumentAuthoringCallBudgetExceeded,
)
from novie_agent_sdk.document_authoring_context import (
    AuthoringContextOverflow,
    build_authoring_context_envelope,
)
from novie_agent_sdk.document_authoring_plan import (
    build_authoring_execution_plan,
)
from novie_agent_sdk.document_authoring_plan_codec import (
    authoring_execution_plan_from_mapping,
)
from novie_agent_sdk.document_authoring_plan_recovery import (
    append_recovery_part,
    merge_executed_section_parts,
)
from novie_agent_sdk.document_authoring_recovery import (
    SectionCoverageCursor,
    repartition_remaining_parts,
)
from novie_agent_sdk.document_part_assembly import (
    PlannedPartRejected,
    accept_part,
    assemble_section,
)
from novie_agent_sdk.document_part_execution import execute_planned_parts


@dataclass
class _Section:
    section_id: str = "context"
    title: str = "Context"
    objective: str = "Explain the problem and constraints."
    min_words: int = 300


def _scope() -> dict[str, str]:
    return {
        "tenant_id": "tenant-1",
        "workspace_id": "workspace-1",
        "workflow_id": "workflow-1",
        "step_id": "step-1",
        "capability_id": "agent.pm.prd",
    }


def test_execution_plan_subdivides_and_keeps_identity_across_revision() -> None:
    first = build_authoring_execution_plan(
        [_Section()],
        scope=_scope(),
        context_budget={"max_part_information_units": 120},
        revision=1,
    )
    second = build_authoring_execution_plan(
        [_Section()],
        scope=_scope(),
        context_budget={"max_part_information_units": 120},
        revision=2,
    )

    assert first.part_count > 1
    assert first.mandatory_call_count == first.part_count + 1
    assert first.sections[0].target_information_units == 300
    assert first.sections[0].max_information_units > 300
    assert [
        item.part_identity for item in first.sections[0].parts
    ] == [
        item.part_identity for item in second.sections[0].parts
    ]


def test_execution_plan_marks_estimated_deadline_pressure_without_false_rejection() -> None:
    plan = build_authoring_execution_plan(
        [_Section(min_words=300)],
        scope=_scope(),
        context_budget={
            "max_part_information_units": 120,
            "max_wall_clock_seconds": 10,
            "authoring_call_p95_seconds": 30,
        },
    )

    assert plan.estimated_authoring_seconds > 10
    assert plan.deadline_feasible is False


def test_default_plan_keeps_normal_long_section_semantically_atomic() -> None:
    plan = build_authoring_execution_plan(
        [_Section(min_words=300)],
        scope=_scope(),
        context_budget={},
    )

    assert len(plan.sections[0].parts) == 1
    assert plan.sections[0].parts[0].objective == plan.sections[0].objective


def test_part_identity_changes_with_tenant_and_evidence_not_plan_revision() -> None:
    plan = build_authoring_execution_plan(
        [_Section(min_words=10)],
        scope=_scope(),
        context_budget={},
    )
    part = plan.sections[0].parts[0]
    with_evidence = part.with_evidence({"items": ["a"]}, scope=_scope())
    other_scope = {**_scope(), "tenant_id": "tenant-2"}
    other_tenant = part.with_evidence({"items": ["a"]}, scope=other_scope)

    assert with_evidence.part_identity != part.part_identity
    assert other_tenant.part_identity != with_evidence.part_identity


def test_recovery_appends_only_missing_objective_and_preserves_accepted_identity() -> None:
    plan = build_authoring_execution_plan(
        [_Section(min_words=10)],
        scope=_scope(),
        context_budget={},
    )
    revised = append_recovery_part(
        plan,
        section_index=0,
        issue_code="missing_planned_content",
        reason="The risk response is absent.",
        scope=_scope(),
    )

    assert revised.revision == plan.revision + 1
    assert len(revised.sections[0].parts) == 2
    assert (
        revised.sections[0].parts[0].part_identity
        == plan.sections[0].parts[0].part_identity
    )
    assert "risk response" in revised.sections[0].parts[1].objective

    restored = authoring_execution_plan_from_mapping(revised.to_mapping())
    assert restored == revised


def test_semantic_recovery_keeps_runtime_repartition_identities() -> None:
    plan = build_authoring_execution_plan(
        [_Section(min_words=80)],
        scope=_scope(),
        context_budget={"max_part_information_units": 100},
    )
    repartitioned = repartition_remaining_parts(
        plan.sections[0].parts,
        failed_index=0,
        revision=2,
        scope=_scope(),
        min_part_information_units=20,
    )
    executed = merge_executed_section_parts(
        plan,
        section_index=0,
        parts=repartitioned,
        revision=2,
    )
    recovered = append_recovery_part(
        executed,
        section_index=0,
        issue_code="missing_planned_content",
        reason="Finish the final trade-off.",
        scope=_scope(),
    )

    assert recovered.revision == 3
    assert [
        part.part_identity for part in recovered.sections[0].parts[:-1]
    ] == [
        part.part_identity for part in repartitioned
    ]


def test_authoring_call_budget_rejects_unplanned_review() -> None:
    budget = AuthoringCallBudget(
        total_limit=2,
        compaction_limit=1,
        review_limit=1,
    )
    budget.reserve("draft")
    budget.reserve("review")

    with pytest.raises(DocumentAuthoringCallBudgetExceeded, match="total"):
        budget.reserve("review")


def test_coverage_cursor_advances_only_after_accepted_part() -> None:
    plan = build_authoring_execution_plan(
        [_Section(min_words=10)],
        scope=_scope(),
        context_budget={},
    )
    part = plan.sections[0].parts[0]
    cursor = SectionCoverageCursor(
        section_id="context",
        remaining_objective_digests=(part.objective_digest,),
    ).accept(part)

    assert cursor.accepted_part_identities == (part.part_identity,)
    assert cursor.remaining_objective_digests == ()
    assert cursor.next_part_ordinal == 2


def test_context_envelope_drops_optional_content_before_required_slots() -> None:
    envelope = build_authoring_context_envelope(
        context_budget={
            "max_input_tokens": 1500,
            "max_output_tokens": 400,
            "structured_output_headroom_tokens": 100,
            "context_safety_margin_tokens": 100,
        },
        authoring_contract="contract",
        objective="objective",
        required_evidence="required evidence",
        coverage_cursor={"remaining": ["one"]},
        previous_part_handoff="handoff",
        authoring_summary="summary",
        optional_evidence="x" * 6000,
    )

    assert "drop_optional_evidence" in envelope.decisions
    assert "required evidence" in envelope.render()
    assert envelope.estimated_input_tokens <= envelope.maximum_input_tokens


def test_context_envelope_fails_when_noncompressible_minimum_cannot_fit() -> None:
    with pytest.raises(AuthoringContextOverflow):
        build_authoring_context_envelope(
            context_budget={
                "max_input_tokens": 300,
                "max_output_tokens": 150,
                "structured_output_headroom_tokens": 100,
                "context_safety_margin_tokens": 100,
            },
            authoring_contract="contract",
            objective="objective",
            required_evidence="required",
            coverage_cursor="cursor",
        )


class _SummaryLlm:
    async def structured(self, **kwargs: Any) -> dict[str, Any]:
        assert kwargs["output_schema"] == AUTHORING_CONTEXT_SUMMARY_SCHEMA
        return {
            "structured": {
                "covered_objectives": ["objective-a"],
                "unresolved_objectives": ["objective-b"],
                "established_facts": ["fact"],
                "terminology": ["term"],
                "decisions": ["decision"],
                "unresolved_questions": [],
                "continuity_notes": ["continue naturally"],
                "previous_part_handoff": "The next part addresses objective-b.",
            }
        }


@pytest.mark.asyncio
async def test_semantic_authoring_compaction_preserves_refs() -> None:
    result = await compact_authoring_context(
        _SummaryLlm(),
        prior_summary=None,
        accepted_part_markdown="Accepted canonical prose.",
        covered_objectives=["objective-a"],
        unresolved_objectives=["objective-b"],
        evidence_refs=[{"artifact_ref": "artifact://evidence-1"}],
    )

    assert result.mode == "semantic_llm"
    assert result.summary["profile"] == "authoring_context_v1"
    assert result.summary["evidence_refs"] == [
        {"artifact_ref": "artifact://evidence-1"}
    ]
    assert "Accepted canonical prose." not in str(result.summary)


class _BrokenSummaryLlm:
    async def structured(self, **_kwargs: Any) -> dict[str, Any]:
        raise TimeoutError


@pytest.mark.asyncio
async def test_semantic_compaction_failure_returns_deterministic_navigation() -> None:
    result = await compact_authoring_context(
        _BrokenSummaryLlm(),
        prior_summary=None,
        accepted_part_markdown="Accepted canonical prose.",
        covered_objectives=["objective-a"],
        unresolved_objectives=["objective-b"],
        evidence_refs=[{"artifact_ref": "artifact://evidence-1"}],
    )

    assert result.mode == "deterministic_fallback"
    assert result.error == "TimeoutError"
    assert result.summary["covered_objectives"] == ["objective-a"]
    assert result.summary["evidence_refs"][0]["artifact_ref"] == "artifact://evidence-1"


def test_part_assembly_normalizes_headings_and_uses_one_blank_seam() -> None:
    plan = build_authoring_execution_plan(
        [_Section(min_words=40)],
        scope=_scope(),
        context_budget={"max_part_information_units": 25},
    )
    parts = plan.sections[0].parts
    accepted = [
        accept_part(
            parts[0],
            "## Context\n\n### First\n\none two three",
            section_title="Context",
            truncated=False,
        ),
        accept_part(
            parts[1],
            "# Foreign\n\nfour five six",
            section_title="Context",
            truncated=False,
        ),
    ]

    section = assemble_section(section_title="Context", parts=accepted)

    assert section.count("## Context") == 1
    assert "### Foreign" in section
    assert "three\n\n### Foreign" in section


def test_part_normalization_repairs_welded_local_heading() -> None:
    part = build_authoring_execution_plan(
        [_Section(min_words=20)],
        scope=_scope(),
        context_budget={},
    ).sections[0].parts[0]

    accepted = accept_part(
        part,
        "alpha beta gamma.### Local Finding\n\ndelta epsilon zeta",
        section_title="Context",
        truncated=False,
    )

    assert "gamma.\n\n### Local Finding" in accepted.markdown


def test_truncated_part_is_never_accepted() -> None:
    part = build_authoring_execution_plan(
        [_Section(min_words=10)],
        scope=_scope(),
        context_budget={},
    ).sections[0].parts[0]

    with pytest.raises(PlannedPartRejected, match="output_truncated"):
        accept_part(
            part,
            "partial text",
            section_title="Context",
            truncated=True,
        )


def test_part_acceptance_allows_completion_headroom_but_keeps_hard_ceiling() -> None:
    part = build_authoring_execution_plan(
        [_Section(min_words=80)],
        scope=_scope(),
        context_budget={"max_part_information_units": 100},
    ).sections[0].parts[0]
    hard_maximum = 1200
    tolerated = " ".join(f"word-{index}" for index in range(hard_maximum))
    excessive = f"{tolerated} overflow"

    accepted = accept_part(
        part,
        tolerated,
        section_title="Context",
        truncated=False,
    )

    assert accepted.information_units == hard_maximum
    with pytest.raises(PlannedPartRejected, match="part_over_maximum"):
        accept_part(
            part,
            excessive,
            section_title="Context",
            truncated=False,
        )


@pytest.mark.asyncio
async def test_overlong_part_is_repartitioned_and_completed() -> None:
    section = build_authoring_execution_plan(
        [_Section(min_words=40)],
        scope=_scope(),
        context_budget={"max_part_information_units": 80},
    ).sections[0]
    generated_maxima: list[int] = []
    events: list[tuple[str, dict[str, Any]]] = []

    async def generate(part, _cursor, _handoff):  # type: ignore[no-untyped-def]
        generated_maxima.append(part.max_information_units)
        units = 700 if len(generated_maxima) == 1 else part.max_information_units - 1
        return " ".join(f"word-{index}" for index in range(units)), False

    async def persist(part):  # type: ignore[no-untyped-def]
        return {"artifact_ref": f"artifact://{part.plan.part_identity}"}

    async def checkpoint(_accepted, _cursor, _revision):  # type: ignore[no-untyped-def]
        return None

    async def emit(event, metadata):  # type: ignore[no-untyped-def]
        events.append((event, dict(metadata)))

    result = await execute_planned_parts(
        section,
        scope=_scope(),
        evidence={"artifact_ref": "artifact://evidence"},
        generate=generate,
        persist=persist,
        checkpoint=checkpoint,
        emit=emit,
        min_part_information_units=20,
    )

    assert generated_maxima[0] > generated_maxima[1]
    assert len(result.parts) == 2
    assert result.plan_revision == 2
    assert result.cursor.remaining_objective_digests == ()
    repartition = next(
        metadata
        for event, metadata in events
        if event == "document.part.repartitioned"
    )
    assert repartition["reason"] == "part_over_maximum"
