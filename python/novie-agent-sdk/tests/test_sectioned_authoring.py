from __future__ import annotations

from typing import Any

import pytest

from novie_agent_sdk import (
    PlatformLlmCallError,
    SectionDraft,
    SectionedLongformAuthor,
    SectionPlan,
    SkillContractResolver,
    run_sectioned_document_finalization,
    sectioned_authoring_contract_from_skill,
)
from novie_agent_sdk.sectioned_authoring import (
    _llm_stream_event_delta,
    _llm_stream_event_result,
)


class _FakeLlm:
    def __init__(self) -> None:
        self.force_wrong_heading = False
        self.chat_kwargs: list[dict[str, Any]] = []
        self.chat_prompts: list[str] = []

    async def structured(
        self,
        *,
        messages: list[dict[str, str]],
        output_schema: dict[str, Any],
        temperature: float,
    ) -> dict[str, Any]:
        return {
            "structured": {
                "length_profile": "short",
                "sections": [
                    {
                        "section_id": "context",
                        "title": "Context",
                        "objective": "Explain the problem context.",
                        "evidence_query": "context",
                        "min_words": 5,
                    },
                    {
                        "section_id": "recommendation",
                        "title": "Recommendation",
                        "objective": "Recommend the next step.",
                        "evidence_query": "recommendation",
                        "min_words": 5,
                    },
                ],
            }
        }

    async def chat(
        self,
        *,
        messages: list[dict[str, str]],
        temperature: float,
        **kwargs: Any,
    ) -> dict[str, str]:
        self.chat_kwargs.append(dict(kwargs))
        prompt = messages[0]["content"]
        self.chat_prompts.append(prompt)
        if "Polish the concatenated sections" in prompt:
            return {"content": "too short"}
        if "upstream-s1-evidence" in prompt:
            heading = (
                "Recommendation"
                if '"section_id": "recommendation"' in prompt
                else "Context"
            )
            return {
                "content": (
                    f"## {heading}\n\n"
                    "alpha beta gamma delta epsilon zeta "
                    "[artifact://upstream-s1-evidence]"
                )
            }
        if '"section_id": "recommendation"' in prompt:
            return {"content": "## Recommendation\n\neta theta iota kappa lambda mu"}
        if self.force_wrong_heading:
            return {"content": "## Wrong Heading\n\nalpha beta gamma delta epsilon zeta"}
        return {"content": "## Context\n\nalpha beta gamma delta epsilon zeta"}


class _FlakyStreamLlm:
    def __init__(self) -> None:
        self.calls = 0

    async def stream_text(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float,
        max_output_tokens: int,
        model: str | None = None,
    ):
        self.calls += 1
        if self.calls == 1:
            yield {"type": "chunk", "delta": {"content": "partial"}}
            raise PlatformLlmCallError(
                capability_id="platform.llm.chat",
                kind="platform_unavailable",
                error_code="internal_error",
                detail=(
                    "peer closed connection without sending complete message body "
                    "(incomplete chunked read)"
                ),
            )
        yield {"type": "chunk", "delta": {"content": "complete"}}
        yield {
            "type": "completed",
            "result": {
                "content": "complete",
                "usage_metadata": {"total_tokens": 7},
            },
        }


class _LongFakeLlm(_FakeLlm):
    async def structured(
        self,
        *,
        messages: list[dict[str, str]],
        output_schema: dict[str, Any],
        temperature: float,
    ) -> dict[str, Any]:
        return {
            "structured": {
                "length_profile": "long",
                "sections": [
                    {
                        "section_id": f"section-{index}",
                        "title": f"Section {index}",
                        "objective": f"Explain section {index}.",
                        "evidence_query": f"section {index}",
                        "min_words": 5,
                    }
                    for index in range(1, 6)
                ],
            }
        }

    async def chat(
        self,
        *,
        messages: list[dict[str, str]],
        temperature: float,
        **kwargs: Any,
    ) -> dict[str, str]:
        self.chat_kwargs.append(dict(kwargs))
        prompt = messages[0]["content"]
        self.chat_prompts.append(prompt)
        if "Merge this cluster" in prompt or "Polish the concatenated sections" in prompt:
            return {"content": "too short"}
        for index in range(1, 6):
            if f'"section_id": "section-{index}"' in prompt:
                return {
                    "content": (
                        f"## Section {index}\n\n"
                        "alpha beta gamma delta epsilon zeta"
                    )
                }
        return {"content": "## Section 1\n\nalpha beta gamma delta epsilon zeta"}


def test_llm_stream_event_delta_extracts_content_blocks() -> None:
    assert (
        _llm_stream_event_delta(
            {
                "type": "chunk",
                "delta": {
                    "content": [
                        {"type": "text", "text": "hello "},
                        {"type": "text", "text": "world"},
                    ]
                },
            }
        )
        == "hello world"
    )


def test_llm_stream_event_result_extracts_provider_text_shapes() -> None:
    result = _llm_stream_event_result(
        {
            "type": "completed",
            "result": {
                "message": {
                    "content": [
                        {"type": "text", "text": "section draft"},
                    ]
                },
                "usage_metadata": {"total_tokens": 12},
            },
        }
    )

    assert result is not None
    assert result["content"] == "section draft"
    assert result["usage_metadata"]["total_tokens"] == 12


@pytest.mark.asyncio
async def test_stream_llm_text_retries_transient_stream_failure_same_path() -> None:
    phase_events: list[dict[str, Any]] = []
    llm = _FlakyStreamLlm()
    author = SectionedLongformAuthor(
        llm_facade=llm,
        platform=_FakePlatform(),
        artifact_type="example_document",
        step_id="s2",
        capability_id="agent.example.write_document",
        context_budget={
            "llm_stream_max_attempts": 2,
            "llm_stream_retry_backoff_seconds": 0,
        },
        phase_event_sink=phase_events.append,
    )

    streamed = await author._stream_llm_text(
        purpose="revise_section",
        messages=[{"role": "user", "content": "rewrite"}],
        temperature=0.25,
        max_output_tokens=1200,
    )

    assert streamed.text == "complete"
    assert streamed.truncated is False
    assert llm.calls == 2
    assert [event["event"] for event in phase_events] == [
        "agent.llm_call.started",
        "agent.llm_call.delta",
        "agent.llm_call.retrying",
        "agent.llm_call.delta",
        "agent.llm_call.completed",
    ]
    retry_event = phase_events[2]
    assert retry_event["attempt"] == 1
    assert retry_event["next_attempt"] == 2
    assert retry_event["chars_total"] == len("partial")
    completed_event = phase_events[-1]
    assert completed_event["attempt"] == 2
    assert completed_event["chars_total"] == len("complete")


class _EmptyDraftLlm(_FakeLlm):
    async def chat(
        self,
        *,
        messages: list[dict[str, str]],
        temperature: float,
        **kwargs: Any,
    ) -> dict[str, str]:
        self.chat_kwargs.append(dict(kwargs))
        self.chat_prompts.append(messages[0]["content"])
        return {"content": ""}


class _PlaceholderDraftLlm(_FakeLlm):
    async def chat(
        self,
        *,
        messages: list[dict[str, str]],
        temperature: float,
        **kwargs: Any,
    ) -> dict[str, str]:
        self.chat_kwargs.append(dict(kwargs))
        self.chat_prompts.append(messages[0]["content"])
        return {"content": "## Context\n\nNo section draft was returned."}


class _FakeArtifacts:
    def __init__(self) -> None:
        self.created: list[dict[str, Any]] = []
        self.read_artifact_ids: list[str] = []

    async def create(self, **kwargs: Any) -> dict[str, Any]:
        artifact_id = f"artifact-{len(self.created) + 1}"
        result = {
            "artifact_id": artifact_id,
            "artifact_ref": f"artifact://{artifact_id}",
            "artifact_type": kwargs["artifact_type"],
            "bytes": len(str(kwargs.get("content") or "")),
        }
        self.created.append({**kwargs, **result})
        return result

    async def read_chunks(
        self,
        artifact_id: str,
        *,
        purpose: str,
        offset: int,
        max_bytes: int,
    ) -> dict[str, Any]:
        self.read_artifact_ids.append(artifact_id)
        return {
            "available": True,
            "content": f"evidence from {artifact_id}",
            "metadata": {"bytes": len(artifact_id)},
        }


class _FakeWorkpads:
    def __init__(self) -> None:
        self.entries: list[dict[str, Any]] = []
        self.snapshot_entries: list[dict[str, Any]] = []
        self.final_refs: list[str] = []
        self.record_error = ""

    async def snapshot(self, *, workflow_id: str | None, limit: int) -> dict[str, Any]:
        return {"entries": self.snapshot_entries[:limit]}

    async def record_entry(self, **kwargs: Any) -> dict[str, Any]:
        self.entries.append(dict(kwargs))
        if self.record_error:
            return {"available": False, "error": self.record_error}
        return {"entry_id": f"entry-{len(self.entries)}"}

    async def set_final_deliverable(
        self,
        artifact_ref: str,
        *,
        workflow_id: str | None,
        step_id: str | None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self.final_refs.append(artifact_ref)
        return {"ok": True}


class _FakePlatform:
    def __init__(self) -> None:
        self.artifacts = _FakeArtifacts()
        self.workpads = _FakeWorkpads()


@pytest.mark.asyncio
async def test_sectioned_author_records_outline_sections_and_final_ref() -> None:
    platform = _FakePlatform()
    phase_events: list[dict[str, Any]] = []
    author = SectionedLongformAuthor(
        llm_facade=_FakeLlm(),
        platform=platform,
        artifact_type="example_document",
        step_id="s2",
        capability_id="agent.example.write_document",
        authoring_contract={
            "coverage_model": "example_document",
            "min_outline_sections": 2,
            "max_outline_sections": 2,
            "min_section_words": 5,
            "default_section_words": 5,
            "max_section_words": 20,
            "final_retention_ratio": 0.8,
            "outline_artifact_type": "example_document.outline",
            "section_artifact_type": "example_document.section",
            "final_artifact_type": "example_document",
        },
        phase_event_sink=phase_events.append,
    )

    result = await author.author(
        brief={"title": "Example document"},
        upstream={},
        workflow_id="workflow-1",
        thread_id="thread-1",
        agent_id="writer",
    )

    assert "## Context" in result.markdown
    assert "## Recommendation" in result.markdown
    assert result.ledger["created_count"] == 4
    assert [item["role"] for item in result.ledger["artifact_refs"]] == [
        "outline",
        "section_draft",
        "section_draft",
        "final_deliverable",
    ]
    assert [item["artifact_type"] for item in platform.artifacts.created] == [
        "example_document.outline",
        "example_document.section",
        "example_document.section",
        "example_document",
    ]
    assert platform.workpads.final_refs == ["artifact://artifact-4"]
    event_names = [event["event"] for event in phase_events]
    assert [name for name in event_names if name.startswith("document.")] == [
        "document.profile.selected",
        "document.outline.started",
        "document.outline.completed",
        "document.section.started",
        "document.section.evidence_pack_built",
        "document.section.gap_detected",
        "document.section.quality_checked",
        "document.section.completed",
        "document.section.started",
        "document.section.evidence_pack_built",
        "document.section.gap_detected",
        "document.section.quality_checked",
        "document.section.completed",
        "document.final.polish_started",
        "document.final.created",
    ]
    assert "agent.llm_call.started" in event_names
    assert "agent.llm_call.delta" in event_names
    assert "agent.llm_call.completed" in event_names
    assert any(
        event["event"] == "agent.tool_call" and event.get("tool_name") == "evidence.build"
        for event in phase_events
    )
    assert any(
        event["event"] == "agent.tool_result" and event.get("tool_name") == "artifact.write"
        for event in phase_events
    )


@pytest.mark.asyncio
async def test_sectioned_author_degrades_workpad_record_failure_after_artifact_write() -> None:
    platform = _FakePlatform()
    platform.workpads.record_error = "platform_workpads_record_entry_timeout"
    phase_events: list[dict[str, Any]] = []
    author = SectionedLongformAuthor(
        llm_facade=_FakeLlm(),
        platform=platform,
        artifact_type="example_document",
        step_id="s2",
        capability_id="agent.example.write_document",
        authoring_contract={
            "coverage_model": "example_document",
            "min_outline_sections": 2,
            "max_outline_sections": 2,
            "min_section_words": 5,
            "default_section_words": 5,
            "max_section_words": 20,
            "final_retention_ratio": 0.8,
            "outline_artifact_type": "example_document.outline",
            "section_artifact_type": "example_document.section",
            "final_artifact_type": "example_document",
        },
        phase_event_sink=phase_events.append,
    )

    result = await author.author(
        brief={"title": "Example document"},
        upstream={},
        workflow_id="workflow-1",
        thread_id="thread-1",
        agent_id="writer",
    )

    assert result.ledger["created_count"] == 4
    assert not any(event["event"] == "agent.tool_error" for event in phase_events)
    assert any(
        event["event"] == "artifact.write.workpad_degraded"
        and event.get("error") == "platform_workpads_record_entry_timeout"
        for event in phase_events
    )
    assert any(
        event["event"] == "agent.tool_result" and event.get("tool_name") == "artifact.write"
        for event in phase_events
    )


@pytest.mark.asyncio
async def test_run_sectioned_document_finalization_returns_trace_and_quality(
    tmp_path,
) -> None:  # type: ignore[no-untyped-def]
    skill = tmp_path / "skills" / "report"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text(
        """---
name: report
metadata:
  novie:
    runtime_contract:
      version: 1
      runtime:
        strategy: sectioned_longform
      document:
        outline:
          min_sections: 2
          max_sections: 2
        section:
          min_units: 5
          default_units: 5
          max_units: 20
        final:
          min_retention_ratio: 0.8
---

# Report
""",
        encoding="utf-8",
    )
    contract = SkillContractResolver(root_dir=tmp_path).resolve(
        ["skills/report"],
        required=True,
    )

    llm = _FakeLlm()
    llm.platform_ns = _FakePlatform()

    result = await run_sectioned_document_finalization(
        llm_facade=llm,
        skill_contract=contract,
        artifact_type="example_document",
        step_id="s2",
        capability_id="agent.example.write_document",
        context_budget={},
        brief={"title": "Example document"},
        upstream={},
        workflow_id="workflow-1",
        thread_id="thread-1",
        agent_id="writer",
        mode_metadata={"example_mode": "write", "example_phase": "default"},
        draft_narrative="Draft narrative.",
        draft_narrative_key="_draft_narrative",
        draft_narrative_artifact_type="draft_narrative",
        draft_narrative_summary="Draft before final authoring.",
        document_input={"artifact_access": "summary_then_fetch"},
    )

    assert "## Context" in result.authoring_result.markdown
    assert result.finalize_strategy == "sectioned_longform"
    assert result.finalize_attempts == 1
    assert result.quality_result.outcome.status == "skipped"
    assert result.started_event.metadata["event"] == "sectioned_authoring_started"
    assert result.started_event.metadata["example_mode"] == "write"
    assert result.completed_event.metadata["event"] == "sectioned_authoring_completed"
    assert result.completed_event.metadata["section_count"] == 2
    assert result.authoring_ledger["section_count"] == 2


@pytest.mark.asyncio
async def test_run_sectioned_document_finalization_requires_sectioned_contract() -> None:
    with pytest.raises(RuntimeError, match="skill runtime contract"):
        await run_sectioned_document_finalization(
            llm_facade=_FakeLlm(),
            skill_contract=None,
            artifact_type="example_document",
            step_id="s2",
            capability_id="agent.example.write_document",
            context_budget={},
            brief={"title": "Example document"},
            upstream={},
        )


@pytest.mark.asyncio
async def test_sectioned_author_passes_budget_ceiling_to_content_calls() -> None:
    platform = _FakePlatform()
    llm = _FakeLlm()
    author = SectionedLongformAuthor(
        llm_facade=llm,
        platform=platform,
        artifact_type="example_document",
        step_id="s2",
        capability_id="agent.example.write_document",
        context_budget={"max_output_tokens": 64000},
        authoring_contract={
            "coverage_model": "example_document",
            "min_outline_sections": 2,
            "max_outline_sections": 2,
            "min_section_words": 5,
            "default_section_words": 5,
            "max_section_words": 20,
            "final_retention_ratio": 0.8,
        },
    )

    await author.author(
        brief={"title": "Example document"},
        upstream={},
        workflow_id="workflow-1",
        thread_id="thread-1",
        agent_id="writer",
    )

    # Two section drafts + the final polish all get the run's budget-contract
    # ceiling — content length is governed by prompt targets and the quality
    # gate, not by per-call heuristics.
    assert [item["max_output_tokens"] for item in llm.chat_kwargs] == [
        64000,
        64000,
        64000,
    ]


@pytest.mark.asyncio
async def test_sectioned_author_defers_output_cap_to_platform_without_budget() -> None:
    platform = _FakePlatform()
    llm = _FakeLlm()
    author = SectionedLongformAuthor(
        llm_facade=llm,
        platform=platform,
        artifact_type="example_document",
        step_id="s2",
        capability_id="agent.example.write_document",
        authoring_contract={
            "coverage_model": "example_document",
            "min_outline_sections": 2,
            "max_outline_sections": 2,
            "min_section_words": 5,
            "default_section_words": 5,
            "max_section_words": 20,
        },
    )

    await author.author(
        brief={"title": "Example document"},
        upstream={},
        workflow_id="workflow-1",
        thread_id="thread-1",
        agent_id="writer",
    )

    # No budget contract → None, so the platform applies its own default.
    assert [item["max_output_tokens"] for item in llm.chat_kwargs] == [
        None,
        None,
        None,
    ]


def test_sectioned_contract_applies_active_length_profile(tmp_path) -> None:  # type: ignore[no-untyped-def]
    skill = tmp_path / "skills" / "report"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text(
        """---
name: report
metadata:
  novie:
    runtime_contract:
      version: 1
      name: report
      runtime:
        strategy: sectioned_longform
        context_policy: evidence_pack_v1
      document:
        outline:
          min_sections: 2
          max_sections: 9
        section:
          min_units: 90
          default_units: 180
          max_units: 280
          max_revision_rounds: 1
        final:
          min_retention_ratio: 0.8
        length_profiles:
          long:
            min_sections: 8
            max_sections: 16
            min_units: 260
            default_units: 520
            max_units: 900
            max_revision_rounds: 2
            finalization: progressive_section_merge
            evidence_depth: deep
---

# Report
""",
        encoding="utf-8",
    )

    contract = SkillContractResolver(root_dir=tmp_path).resolve(["skills/report"], required=True)
    authoring = sectioned_authoring_contract_from_skill(
        contract,
        artifact_type="management_report",
        length_profile="long",
        profile_source="user_input",
        profile_confidence="confirmed",
    )

    assert authoring["length_profile"] == "long"
    assert authoring["profile_source"] == "user_input"
    assert authoring["min_outline_sections"] == 8
    assert authoring["max_outline_sections"] == 16
    assert authoring["default_section_words"] == 520
    assert authoring["max_section_revision_rounds"] == 2
    assert authoring["finalization"] == "progressive_section_merge"
    assert authoring["evidence_depth"] == "deep"


@pytest.mark.asyncio
async def test_long_profile_uses_progressive_section_merge() -> None:
    platform = _FakePlatform()
    phase_events: list[dict[str, Any]] = []
    llm = _LongFakeLlm()
    author = SectionedLongformAuthor(
        llm_facade=llm,
        platform=platform,
        artifact_type="example_document",
        step_id="s2",
        capability_id="agent.example.write_document",
        authoring_contract={
            "coverage_model": "example_document",
            "length_profile": "long",
            "profile_source": "user_input",
            "profile_confidence": "confirmed",
            "min_outline_sections": 5,
            "max_outline_sections": 5,
            "min_section_words": 5,
            "default_section_words": 5,
            "max_section_words": 20,
            "finalization": "progressive_section_merge",
            "final_retention_ratio": 0.8,
        },
        phase_event_sink=phase_events.append,
    )

    result = await author.author(
        brief={"title": "Example long document"},
        upstream={},
        workflow_id="workflow-1",
        thread_id="thread-1",
        agent_id="writer",
    )

    assert len(result.drafts) == 5
    assert result.ledger["length_profile"] == "long"
    assert result.ledger["finalization"] == "progressive_section_merge"
    assert any("Merge this cluster" in prompt for prompt in llm.chat_prompts)
    assert [event["event"] for event in phase_events].count(
        "document.final.merge_cluster_started"
    ) == 2
    assert [event["event"] for event in phase_events].count(
        "document.final.merge_cluster_completed"
    ) == 2


def test_outline_schema_carries_function_title() -> None:
    """LangChain rejects title-less dict schemas in with_structured_output
    (incident 2026-06-11: 'Unsupported function ... must have a top-level
    title key'). The outline schema must always ship a title."""
    from novie_agent_sdk.sectioned_authoring import (
        SectionedAuthoringContract,
        _outline_schema,
    )

    schema = _outline_schema(SectionedAuthoringContract())
    assert schema.get("title"), "outline schema lost its top-level title"
    assert "maxItems" not in schema["properties"]["sections"]


@pytest.mark.asyncio
async def test_sectioned_author_repairs_missing_planned_heading_before_quality_gate() -> None:
    platform = _FakePlatform()
    llm = _FakeLlm()
    llm.force_wrong_heading = True
    author = SectionedLongformAuthor(
        llm_facade=llm,
        platform=platform,
        artifact_type="example_document",
        step_id="s2",
        capability_id="agent.example.write_document",
        authoring_contract={
            "coverage_model": "example_document",
            "min_outline_sections": 2,
            "max_outline_sections": 2,
            "min_section_words": 5,
            "default_section_words": 5,
            "max_section_words": 20,
            "final_retention_ratio": 0.8,
            "outline_artifact_type": "example_document.outline",
            "section_artifact_type": "example_document.section",
            "final_artifact_type": "example_document",
        },
    )

    result = await author.author(
        brief={"title": "Example document"},
        upstream={},
        workflow_id="workflow-1",
        thread_id="thread-1",
        agent_id="writer",
    )

    assert result.drafts[0].markdown.startswith("## Context\n\n## Wrong Heading")
    assert result.drafts[0].quality["passed"] is True
    assert result.drafts[0].quality["failures"] == []


def test_unique_sources_gate_caps_requirement_by_available_evidence() -> None:
    """A thin evidence pack must not make the gate unsatisfiable: the required
    unique-source count is capped by what the pack actually offers, and grades
    sources *cited in the section* rather than merely present in the pack."""
    from novie_agent_sdk.sectioned_authoring import (
        SectionedAuthoringContract,
        SectionPlan,
        _evaluate_section_quality,
    )

    contract = SectionedAuthoringContract(
        min_section_words=3,
        require_evidence_refs=True,
        min_unique_sources_per_core_section=2,
    )
    plan = SectionPlan(section_id="context", title="Context", min_words=3)

    # Only one source available, and it is cited → no insufficient_unique_sources
    single = _evaluate_section_quality(
        plan=plan,
        markdown="## Context\n\nalpha beta gamma delta https://a.example",
        evidence_pack={"items": [{"url": "https://a.example", "title": "A"}]},
        contract=contract,
        revision_rounds=0,
    )
    assert "insufficient_unique_sources" not in single.failures
    assert single.unique_sources_available == 1
    assert single.unique_sources_cited == 1


def test_unique_sources_gate_fails_when_fewer_sources_cited_than_available() -> None:
    from novie_agent_sdk.sectioned_authoring import (
        SectionedAuthoringContract,
        SectionPlan,
        _evaluate_section_quality,
    )

    contract = SectionedAuthoringContract(
        min_section_words=3,
        require_evidence_refs=True,
        min_unique_sources_per_core_section=2,
    )
    plan = SectionPlan(section_id="context", title="Context", min_words=3)

    # Two sources available but only one cited → gate fails.
    result = _evaluate_section_quality(
        plan=plan,
        markdown="## Context\n\nalpha beta gamma delta https://a.example",
        evidence_pack={
            "items": [
                {"url": "https://a.example", "title": "A"},
                {"url": "https://b.example", "title": "B"},
            ]
        },
        contract=contract,
        revision_rounds=0,
    )
    assert "insufficient_unique_sources" in result.failures
    assert result.unique_sources_available == 2
    assert result.unique_sources_cited == 1


@pytest.mark.asyncio
async def test_sectioned_author_degrades_soft_gate_failure_instead_of_raising() -> None:
    """Default ``degrade`` enforcement records a best-effort section with a gap
    marker for soft failures rather than dead-ending the plan."""
    platform = _FakePlatform()
    phase_events: list[dict[str, Any]] = []
    author = SectionedLongformAuthor(
        llm_facade=_FakeLlm(),
        platform=platform,
        artifact_type="example_document",
        step_id="s2",
        capability_id="agent.example.write_document",
        authoring_contract={
            "coverage_model": "example_document",
            "min_outline_sections": 2,
            "max_outline_sections": 2,
            "min_section_words": 5,
            "default_section_words": 5,
            "max_section_words": 20,
            "final_retention_ratio": 0.8,
            # Soft gate the FakeLlm output never satisfies.
            "require_confidence_layer": True,
        },
        phase_event_sink=phase_events.append,
    )

    result = await author.author(
        brief={"title": "Example document"},
        upstream={},
        workflow_id="workflow-1",
        thread_id="thread-1",
        agent_id="writer",
    )

    assert result.ledger["degraded"] is True
    assert result.ledger["degraded_sections"]
    assert all(draft.quality["degraded"] is True for draft in result.drafts)
    assert "Evidence gap (auto-flagged)" in result.drafts[0].markdown
    assert "document.section.quality_degraded" in [e["event"] for e in phase_events]


@pytest.mark.asyncio
async def test_sectioned_author_strict_mode_raises_on_soft_gate_failure() -> None:
    platform = _FakePlatform()
    author = SectionedLongformAuthor(
        llm_facade=_FakeLlm(),
        platform=platform,
        artifact_type="example_document",
        step_id="s2",
        capability_id="agent.example.write_document",
        authoring_contract={
            "coverage_model": "example_document",
            "min_outline_sections": 2,
            "max_outline_sections": 2,
            "min_section_words": 5,
            "default_section_words": 5,
            "max_section_words": 20,
            "final_retention_ratio": 0.8,
            "require_confidence_layer": True,
            "gate_enforcement": "strict",
        },
    )

    with pytest.raises(RuntimeError, match="section_quality_gate_failed"):
        await author.author(
            brief={"title": "Example document"},
            upstream={},
            workflow_id="workflow-1",
            thread_id="thread-1",
            agent_id="writer",
        )


@pytest.mark.asyncio
async def test_sectioned_author_rejects_empty_section_drafts() -> None:
    platform = _FakePlatform()
    author = SectionedLongformAuthor(
        llm_facade=_EmptyDraftLlm(),
        platform=platform,
        artifact_type="example_document",
        step_id="s2",
        capability_id="agent.example.write_document",
        authoring_contract={
            "coverage_model": "example_document",
            "min_outline_sections": 2,
            "max_outline_sections": 2,
            "min_section_words": 5,
            "default_section_words": 5,
            "max_section_words": 20,
            "final_retention_ratio": 0.8,
        },
    )

    with pytest.raises(RuntimeError, match="section_quality_gate_failed:context"):
        await author.author(
            brief={"title": "Example document"},
            upstream={},
            workflow_id="workflow-1",
            thread_id="thread-1",
            agent_id="writer",
        )

    assert [item["artifact_type"] for item in platform.artifacts.created] == [
        "example_document.outline",
    ]
    assert platform.workpads.final_refs == []


@pytest.mark.asyncio
async def test_sectioned_author_rejects_placeholder_section_drafts() -> None:
    platform = _FakePlatform()
    author = SectionedLongformAuthor(
        llm_facade=_PlaceholderDraftLlm(),
        platform=platform,
        artifact_type="example_document",
        step_id="s2",
        capability_id="agent.example.write_document",
        authoring_contract={
            "coverage_model": "example_document",
            "min_outline_sections": 2,
            "max_outline_sections": 2,
            "min_section_words": 3,
            "default_section_words": 3,
            "max_section_words": 20,
            "final_retention_ratio": 0.8,
        },
    )

    with pytest.raises(RuntimeError, match="placeholder_section"):
        await author.author(
            brief={"title": "Example document"},
            upstream={},
            workflow_id="workflow-1",
            thread_id="thread-1",
            agent_id="writer",
        )

    assert [item["artifact_type"] for item in platform.artifacts.created] == [
        "example_document.outline",
    ]
    assert platform.workpads.final_refs == []


@pytest.mark.asyncio
async def test_deferred_intermediate_artifacts_do_not_persist_on_failure() -> None:
    platform = _FakePlatform()
    author = SectionedLongformAuthor(
        llm_facade=_EmptyDraftLlm(),
        platform=platform,
        artifact_type="example_document",
        step_id="s2",
        capability_id="agent.example.write_document",
        authoring_contract={
            "coverage_model": "example_document",
            "min_outline_sections": 2,
            "max_outline_sections": 2,
            "min_section_words": 5,
            "default_section_words": 5,
            "max_section_words": 20,
        },
        defer_intermediate_artifacts=True,
    )

    with pytest.raises(RuntimeError, match="section_quality_gate_failed:context"):
        await author.author(
            brief={"title": "Example document"},
            upstream={},
            workflow_id="workflow-1",
            thread_id="thread-1",
            agent_id="writer",
        )

    assert platform.artifacts.created == []
    assert platform.workpads.entries == []
    assert platform.workpads.final_refs == []


@pytest.mark.asyncio
async def test_sectioned_author_excludes_current_step_workpad_refs_from_evidence() -> None:
    platform = _FakePlatform()
    platform.workpads.snapshot_entries = [
        {
            "step_id": "s2",
            "title": "stale current-step section",
            "artifact_refs": [
                {
                    "artifact_id": "stale-s2-section",
                    "artifact_type": "example_document.section",
                    "ref": "artifact://stale-s2-section",
                }
            ],
        },
        {
            "step_id": "s1",
            "title": "upstream research",
            "artifact_refs": [
                {
                    "artifact_id": "upstream-s1-evidence",
                    "artifact_type": "market_analysis",
                    "ref": "artifact://upstream-s1-evidence",
                }
            ],
        },
    ]
    llm = _FakeLlm()
    author = SectionedLongformAuthor(
        llm_facade=llm,
        platform=platform,
        artifact_type="example_document",
        step_id="s2",
        capability_id="agent.example.write_document",
        authoring_contract={
            "coverage_model": "example_document",
            "min_outline_sections": 2,
            "max_outline_sections": 2,
            "min_section_words": 5,
            "default_section_words": 5,
            "max_section_words": 20,
            "final_retention_ratio": 0.8,
        },
    )

    await author.author(
        brief={"title": "Example document"},
        upstream={},
        workflow_id="workflow-1",
        thread_id="thread-1",
        agent_id="writer",
    )

    assert "upstream-s1-evidence" in platform.artifacts.read_artifact_ids
    assert "stale-s2-section" not in platform.artifacts.read_artifact_ids
    draft_prompts = [
        prompt
        for prompt in llm.chat_prompts
        if "Write exactly this document section" in prompt
    ]
    assert draft_prompts
    assert "evidence from upstream-s1-evidence" in draft_prompts[0]
    assert "stale-s2-section" not in draft_prompts[0]


class _BridgeLlm:
    """Minimal LLM double for the boundary-stitch finalize path."""

    def __init__(self, bridge: str = "With that established, the next theme follows.") -> None:
        self.bridge = bridge
        self.chat_kwargs: list[dict[str, Any]] = []
        self.chat_prompts: list[str] = []

    async def chat(
        self,
        *,
        messages: list[dict[str, str]],
        temperature: float,
        **kwargs: Any,
    ) -> dict[str, str]:
        self.chat_kwargs.append(dict(kwargs))
        self.chat_prompts.append(messages[0]["content"])
        return {"content": self.bridge}


def _stitch_author(
    llm: Any,
    *,
    phase_events: list[dict[str, Any]],
    contract_extra: dict[str, Any] | None = None,
) -> SectionedLongformAuthor:
    contract: dict[str, Any] = {
        "coverage_model": "example_document",
        "finalization": "boundary_stitch",
    }
    if contract_extra:
        contract.update(contract_extra)
    return SectionedLongformAuthor(
        llm_facade=llm,
        platform=_FakePlatform(),
        artifact_type="example_document",
        step_id="s1",
        capability_id="agent.example.write_document",
        authoring_contract=contract,
        phase_event_sink=phase_events.append,
    )


@pytest.mark.asyncio
async def test_boundary_stitch_preserves_bodies_and_inserts_bridges() -> None:
    phase_events: list[dict[str, Any]] = []
    llm = _BridgeLlm(bridge="With that established, attention turns to the next theme.")
    author = _stitch_author(llm, phase_events=phase_events)
    drafts = [
        SectionDraft(
            plan=SectionPlan(section_id="s1", title="Context"),
            markdown="## Context\n\nalpha beta gamma.",
        ),
        SectionDraft(
            plan=SectionPlan(section_id="s2", title="Findings"),
            markdown="## Findings\n\ndelta epsilon zeta.",
        ),
        SectionDraft(
            plan=SectionPlan(section_id="s3", title="Recommendation"),
            markdown="## Recommendation\n\neta theta iota.",
        ),
    ]
    combined = "\n\n".join(draft.markdown for draft in drafts)

    result = await author._boundary_stitch_final(
        brief={"title": "Doc"}, drafts=drafts, combined=combined
    )

    # Every section body is preserved verbatim — the seam pass cannot drop content.
    for draft in drafts:
        assert draft.markdown in result
    # A transition bridge was inserted between sections.
    assert "With that established" in result
    # One LLM call per seam (3 sections -> 2 seams), all seam-stitch prompts.
    assert len(llm.chat_prompts) == 2
    assert all("smoothing the seam" in prompt for prompt in llm.chat_prompts)
    events = [event["event"] for event in phase_events]
    assert events.count("document.final.seam_stitch_started") == 2
    assert events.count("document.final.seam_stitch_completed") == 2
    # Normal-size document must not trigger a truncation warning.
    assert "document.final.truncation_warning" not in events


@pytest.mark.asyncio
async def test_boundary_stitch_single_section_skips_llm() -> None:
    phase_events: list[dict[str, Any]] = []
    llm = _BridgeLlm()
    author = _stitch_author(llm, phase_events=phase_events)
    drafts = [
        SectionDraft(
            plan=SectionPlan(section_id="s1", title="Only"),
            markdown="## Only\n\nalpha beta gamma.",
        )
    ]

    result = await author._boundary_stitch_final(
        brief={}, drafts=drafts, combined=drafts[0].markdown
    )

    assert result == drafts[0].markdown
    assert llm.chat_prompts == []


@pytest.mark.asyncio
async def test_boundary_stitch_forwards_finalize_model_override() -> None:
    phase_events: list[dict[str, Any]] = []
    llm = _BridgeLlm(bridge="Next.")
    author = _stitch_author(
        llm,
        phase_events=phase_events,
        contract_extra={"finalize_model": "custom-finalize-model"},
    )
    drafts = [
        SectionDraft(plan=SectionPlan(section_id="s1", title="A"), markdown="## A\n\nalpha beta."),
        SectionDraft(plan=SectionPlan(section_id="s2", title="B"), markdown="## B\n\ngamma delta."),
    ]

    await author._boundary_stitch_final(
        brief={}, drafts=drafts, combined="\n\n".join(d.markdown for d in drafts)
    )

    assert llm.chat_kwargs
    assert all(kw.get("model") == "custom-finalize-model" for kw in llm.chat_kwargs)


@pytest.mark.asyncio
async def test_bounded_for_final_prompt_emits_truncation_warning() -> None:
    phase_events: list[dict[str, Any]] = []
    author = _stitch_author(_BridgeLlm(), phase_events=phase_events)

    short = "x" * 100
    assert (
        await author._bounded_for_final_prompt(short, limit=200, phase="single_polish")
        == short
    )
    assert all(
        event["event"] != "document.final.truncation_warning" for event in phase_events
    )

    long = "y" * 500
    bounded = await author._bounded_for_final_prompt(
        long, limit=200, phase="single_polish"
    )
    assert bounded == "y" * 200
    warnings = [
        event
        for event in phase_events
        if event["event"] == "document.final.truncation_warning"
    ]
    assert len(warnings) == 1
    assert warnings[0]["input_chars"] == 500
    assert warnings[0]["dropped_chars"] == 300
    assert warnings[0]["phase"] == "single_polish"


def _running_context_author(
    llm: Any,
    *,
    phase_events: list[dict[str, Any]],
    contract_extra: dict[str, Any] | None = None,
) -> SectionedLongformAuthor:
    contract: dict[str, Any] = {
        "coverage_model": "example_document",
        "length_profile": "long",
        "min_outline_sections": 5,
        "max_outline_sections": 5,
        "min_section_words": 5,
        "default_section_words": 5,
        "max_section_words": 20,
        "running_context_window_k": 2,
    }
    if contract_extra:
        contract.update(contract_extra)
    return SectionedLongformAuthor(
        llm_facade=llm,
        platform=_FakePlatform(),
        artifact_type="example_document",
        step_id="s2",
        capability_id="agent.example.write_document",
        authoring_contract=contract,
        phase_event_sink=phase_events.append,
    )


def test_compose_running_context_windows_bodies_and_summarizes_older() -> None:
    author = _stitch_author(
        _BridgeLlm(), phase_events=[], contract_extra={"running_context_window_k": 2}
    )
    drafts = [
        SectionDraft(
            plan=SectionPlan(section_id=f"s{i}", title=f"T{i}"),
            markdown=f"## T{i}\n\nbody-{i}",
        )
        for i in range(1, 5)
    ]

    ctx = author._compose_running_context(drafts, running_summary="EARLIER-SUMMARY")

    # Older sections are represented by the summary, not their verbatim bodies.
    assert "EARLIER-SUMMARY" in ctx
    assert "body-1" not in ctx
    assert "body-2" not in ctx
    # The last k=2 sections are included verbatim.
    assert "body-3" in ctx
    assert "body-4" in ctx


@pytest.mark.asyncio
async def test_fold_running_summary_forwards_model_and_emits_event() -> None:
    phase_events: list[dict[str, Any]] = []
    llm = _BridgeLlm(bridge="FOLDED SUMMARY")
    author = _stitch_author(
        llm,
        phase_events=phase_events,
        contract_extra={
            "running_summary_model": "sum-model",
            "running_summary_max_tokens": 400,
        },
    )
    dropped = SectionDraft(
        plan=SectionPlan(section_id="s1", title="T1"),
        markdown="## T1\n\nalpha beta gamma.",
    )

    out = await author._fold_running_summary("prior summary", dropped)

    assert out == "FOLDED SUMMARY"
    assert llm.chat_kwargs
    assert all(kw.get("model") == "sum-model" for kw in llm.chat_kwargs)
    assert any("Maintain a running summary" in prompt for prompt in llm.chat_prompts)
    assert any(
        event["event"] == "document.running_summary.updated" for event in phase_events
    )


@pytest.mark.asyncio
async def test_running_context_feeds_prior_context_and_folds_incrementally() -> None:
    phase_events: list[dict[str, Any]] = []
    llm = _LongFakeLlm()
    author = _running_context_author(llm, phase_events=phase_events)

    await author.author(
        brief={"title": "Doc"},
        upstream={},
        workflow_id="w",
        thread_id="t",
        agent_id="a",
    )

    # 5 sections, window k=2 -> one fold per section that leaves the window = 3.
    fold_prompts = [p for p in llm.chat_prompts if "Maintain a running summary" in p]
    assert len(fold_prompts) == 3
    updates = [
        event
        for event in phase_events
        if event["event"] == "document.running_summary.updated"
    ]
    assert len(updates) == 3
    # Later section drafts carry the running-context block.
    draft_prompts = [
        p for p in llm.chat_prompts if "Write exactly this document section" in p
    ]
    assert any("Document so far" in p for p in draft_prompts)


@pytest.mark.asyncio
async def test_running_context_disabled_uses_refs_only() -> None:
    phase_events: list[dict[str, Any]] = []
    llm = _LongFakeLlm()
    author = _running_context_author(
        llm, phase_events=phase_events, contract_extra={"running_context": False}
    )

    await author.author(
        brief={"title": "Doc"},
        upstream={},
        workflow_id="w",
        thread_id="t",
        agent_id="a",
    )

    assert all("Document so far" not in prompt for prompt in llm.chat_prompts)
    assert all("Maintain a running summary" not in prompt for prompt in llm.chat_prompts)


# --- finalization contract validation ---------------------------------------


def test_contract_accepts_known_finalization_modes() -> None:
    from novie_agent_sdk.sectioned_authoring import (
        KNOWN_FINALIZATION_MODES,
        SectionedAuthoringContract,
    )

    for mode in KNOWN_FINALIZATION_MODES:
        contract = SectionedAuthoringContract.from_mapping({"finalization": mode})
        assert contract.finalization == mode
    # Absent / empty values fall back to the default mode.
    assert SectionedAuthoringContract.from_mapping({}).finalization == "single_polish"
    assert SectionedAuthoringContract.from_mapping(None).finalization == "single_polish"


def test_contract_rejects_unknown_finalization_mode() -> None:
    from novie_agent_sdk.sectioned_authoring import SectionedAuthoringContract

    with pytest.raises(ValueError) as excinfo:
        SectionedAuthoringContract.from_mapping(
            {"finalization": "section_ledger_polish"}
        )
    message = str(excinfo.value)
    # The offending value and the valid modes are both named so the skill
    # author can fix SKILL.md without reading SDK source.
    assert "section_ledger_polish" in message
    assert "boundary_stitch" in message
    assert "progressive_section_merge" in message
    assert "single_polish" in message


@pytest.mark.asyncio
async def test_polish_final_emits_fallback_event_for_unvalidated_contract() -> None:
    from novie_agent_sdk.sectioned_authoring import SectionedAuthoringContract

    phase_events: list[dict[str, Any]] = []
    polished = "## Context\n\nalpha beta gamma.\n\n## Findings\n\ndelta epsilon zeta."
    llm = _BridgeLlm(bridge=polished)
    author = SectionedLongformAuthor(
        llm_facade=llm,
        platform=_FakePlatform(),
        artifact_type="example_document",
        step_id="s1",
        capability_id="agent.example.write_document",
        # Direct dataclass construction bypasses from_mapping validation.
        authoring_contract=SectionedAuthoringContract(
            finalization="section_ledger_polish"
        ),
        phase_event_sink=phase_events.append,
    )
    drafts = [
        SectionDraft(
            plan=SectionPlan(section_id="s1", title="Context"),
            markdown="## Context\n\nalpha beta gamma.",
        ),
        SectionDraft(
            plan=SectionPlan(section_id="s2", title="Findings"),
            markdown="## Findings\n\ndelta epsilon zeta.",
        ),
    ]

    result = await author._polish_final(brief={"title": "Doc"}, drafts=drafts)

    fallback_events = [
        event for event in phase_events
        if event["event"] == "document.finalize.mode_fallback"
    ]
    assert len(fallback_events) == 1
    assert fallback_events[0]["requested_mode"] == "section_ledger_polish"
    assert fallback_events[0]["effective_mode"] == "single_polish"
    # The single_polish path actually ran.
    assert any(
        "Polish the concatenated sections" in prompt for prompt in llm.chat_prompts
    )
    assert result == polished


# --- output truncation detection (stop_reason) --------------------------------


def test_finish_reason_normalisation() -> None:
    from novie_agent_sdk.sectioned_authoring import _finish_reason_of

    # OpenAI-compatible shape.
    assert _finish_reason_of({"response_metadata": {"finish_reason": "length"}}) == "length"
    # Anthropic-style shape, case-insensitive.
    assert _finish_reason_of({"response_metadata": {"stop_reason": "MAX_TOKENS"}}) == "max_tokens"
    assert _finish_reason_of({"response_metadata": {"finish_reason": "stop"}}) == "stop"
    # Unreadable shapes degrade to "not truncated", never to an error.
    assert _finish_reason_of({"response_metadata": {}}) == ""
    assert _finish_reason_of({"response_metadata": "bogus"}) == ""
    assert _finish_reason_of({}) == ""
    assert _finish_reason_of(None) == ""


class _TruncatingDraftLlm(_FakeLlm):
    """Every section draft reports finish_reason=length (cut at the token cap)."""

    async def chat(
        self,
        *,
        messages: list[dict[str, str]],
        temperature: float,
        **kwargs: Any,
    ) -> dict[str, Any]:
        result = await super().chat(
            messages=messages, temperature=temperature, **kwargs
        )
        if "Write exactly this document section" in messages[0]["content"]:
            return {**result, "response_metadata": {"finish_reason": "length"}}
        return dict(result)


@pytest.mark.asyncio
async def test_truncated_section_draft_fails_gate_retries_then_degrades() -> None:
    platform = _FakePlatform()
    phase_events: list[dict[str, Any]] = []
    llm = _TruncatingDraftLlm()
    author = SectionedLongformAuthor(
        llm_facade=llm,
        platform=platform,
        artifact_type="example_document",
        step_id="s2",
        capability_id="agent.example.write_document",
        authoring_contract={
            "coverage_model": "example_document",
            "min_outline_sections": 2,
            "max_outline_sections": 2,
            "min_section_words": 5,
            "default_section_words": 5,
            "max_section_words": 20,
            "max_section_revision_rounds": 1,
        },
        phase_event_sink=phase_events.append,
    )

    result = await author.author(
        brief={"title": "Example document"},
        upstream={},
        workflow_id="workflow-1",
        thread_id="thread-1",
        agent_id="writer",
    )

    event_names = [event["event"] for event in phase_events]
    # Detected on the first draft AND on the revision of each section.
    assert event_names.count("document.section.truncation_detected") == 4
    # The gate records the truncation as a deterministic failure.
    quality_events = [
        event for event in phase_events
        if event["event"] == "document.section.quality_checked"
    ]
    assert all(
        "output_truncated" in event["quality"]["failures"]
        for event in quality_events
    )
    # The revision prompt tells the model to finish within the budget.
    revision_prompts = [
        prompt for prompt in llm.chat_prompts
        if "Section quality gate failed" in prompt
    ]
    assert revision_prompts
    assert all(
        "cut off at the output length limit" in prompt
        for prompt in revision_prompts
    )
    # Degrade enforcement ships the section, but the gap note names the cut.
    assert "output_truncated" in result.markdown


@pytest.mark.asyncio
async def test_single_polish_truncated_output_returns_combined() -> None:
    class _TruncatedPolishLlm(_BridgeLlm):
        async def chat(
            self,
            *,
            messages: list[dict[str, str]],
            temperature: float,
            **kwargs: Any,
        ) -> dict[str, Any]:
            result = await super().chat(
                messages=messages, temperature=temperature, **kwargs
            )
            return {**result, "response_metadata": {"stop_reason": "max_tokens"}}

    phase_events: list[dict[str, Any]] = []
    # Longer than the combined drafts: the shrinkage-only retention guard
    # would have accepted this cut-off rewrite.
    polished = (
        "## Context\n\nalpha beta gamma delta epsilon zeta eta theta.\n\n"
        "## Findings\n\niota kappa lambda mu nu xi omicron pi rho sigma"
    )
    llm = _TruncatedPolishLlm(bridge=polished)
    author = SectionedLongformAuthor(
        llm_facade=llm,
        platform=_FakePlatform(),
        artifact_type="example_document",
        step_id="s1",
        capability_id="agent.example.write_document",
        authoring_contract={"finalization": "single_polish"},
        phase_event_sink=phase_events.append,
    )
    drafts = [
        SectionDraft(
            plan=SectionPlan(section_id="s1", title="Context"),
            markdown="## Context\n\nalpha beta gamma.",
        ),
        SectionDraft(
            plan=SectionPlan(section_id="s2", title="Findings"),
            markdown="## Findings\n\ndelta epsilon zeta.",
        ),
    ]

    result = await author._polish_final(brief={"title": "Doc"}, drafts=drafts)

    # Complete originals beat the truncated rewrite.
    assert polished not in result
    for draft in drafts:
        assert draft.markdown in result
    warnings = [
        event for event in phase_events
        if event["event"] == "document.final.truncation_warning"
    ]
    assert len(warnings) == 1
    assert warnings[0]["phase"] == "single_polish_output"
    assert warnings[0]["finish_reason"] == "max_tokens"


@pytest.mark.asyncio
async def test_boundary_stitch_drops_truncated_bridge() -> None:
    class _TruncatedBridgeLlm(_BridgeLlm):
        async def chat(
            self,
            *,
            messages: list[dict[str, str]],
            temperature: float,
            **kwargs: Any,
        ) -> dict[str, Any]:
            result = await super().chat(
                messages=messages, temperature=temperature, **kwargs
            )
            return {**result, "response_metadata": {"finish_reason": "length"}}

    phase_events: list[dict[str, Any]] = []
    llm = _TruncatedBridgeLlm(bridge="A transition that was cut mid-")
    author = _stitch_author(llm, phase_events=phase_events)
    drafts = [
        SectionDraft(
            plan=SectionPlan(section_id="s1", title="Context"),
            markdown="## Context\n\nalpha beta gamma.",
        ),
        SectionDraft(
            plan=SectionPlan(section_id="s2", title="Findings"),
            markdown="## Findings\n\ndelta epsilon zeta.",
        ),
    ]
    combined = "\n\n".join(draft.markdown for draft in drafts)

    result = await author._boundary_stitch_final(
        brief={"title": "Doc"}, drafts=drafts, combined=combined
    )

    # Bodies are intact; the cut-off bridge is dropped, not stitched in.
    for draft in drafts:
        assert draft.markdown in result
    assert "A transition that was cut mid-" not in result
    seam_events = [
        event for event in phase_events
        if event["event"] == "document.final.seam_stitch_completed"
    ]
    assert len(seam_events) == 1
    assert seam_events[0]["truncated"] is True
    assert seam_events[0]["bridged"] is False
    assert seam_events[0]["status"] == "degraded"
