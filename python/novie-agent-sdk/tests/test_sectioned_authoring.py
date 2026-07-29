from __future__ import annotations

import asyncio
from typing import Any

import pytest

from novie_agent_sdk import (
    DocumentAuthoringDeadlineExceeded,
    DocumentAuthoringRequest,
    PlatformLlmCallError,
    SectionDraft,
    SectionedLongformAuthor,
    SectionPlan,
    SkillContractResolver,
    run_sectioned_document_finalization,
    sectioned_authoring_contract_from_skill,
)
from novie_agent_sdk import astream_sectioned_document_finalization
from novie_agent_sdk.sectioned_authoring import (
    _HEADING_RE,
    _isolate_requested_section,
    _llm_stream_event_delta,
    _llm_stream_event_result,
    project_sectioned_phase_event,
    _section_structure_violation,
)


def test_isolate_requested_section_retitles_foreign_h2_without_preserving_it() -> None:
    isolated = _isolate_requested_section(
        "## Market Evidence\n\nDomain-specific body.\n\n"
        "## Recommendations\n\nUnrelated sibling.",
        "Domain Evidence",
    )

    assert isolated == "## Domain Evidence\n\nDomain-specific body."
    assert "## Market Evidence" not in isolated
    assert "## Recommendations" not in isolated


def test_sectioned_llm_deltas_do_not_cross_the_a2a_transport_boundary() -> None:
    assert project_sectioned_phase_event(
        {"event": "agent.llm_call.delta", "text_delta": "accepted section body"}
    ) == []


class _FakeLlm:
    def __init__(self) -> None:
        self.force_wrong_heading = False
        self.chat_kwargs: list[dict[str, Any]] = []
        self.chat_prompts: list[str] = []
        self.structured_prompts: list[str] = []

    async def structured(
        self,
        *,
        messages: list[dict[str, str]],
        output_schema: dict[str, Any],
        temperature: float,
        **_kwargs: Any,
    ) -> dict[str, Any]:
        self.structured_prompts.append("\n".join(item["content"] for item in messages))
        if output_schema.get("title") == "SectionCompletenessReview":
            return {
                "structured": {
                    "complete": True,
                    "issue_code": "none",
                    "reason": "The candidate section fulfills its objective.",
                }
            }
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
        **_kwargs: Any,
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
        **_kwargs: Any,
    ) -> dict[str, Any]:
        if output_schema.get("title") == "SectionCompletenessReview":
            return await super().structured(
                messages=messages,
                output_schema=output_schema,
                temperature=temperature,
            )
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


class _WholeDocumentDraftLlm(_FakeLlm):
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
        return {
            "content": (
                "# Example Document\n\n"
                "## Context\n\n"
                "alpha beta gamma delta epsilon zeta\n\n"
                "## Recommendation\n\n"
                "eta theta iota kappa lambda mu"
            )
        }


class _HangingLlm(_FakeLlm):
    async def chat(
        self,
        *,
        messages: list[dict[str, str]],
        temperature: float,
        **kwargs: Any,
    ) -> dict[str, str]:
        await asyncio.Event().wait()
        raise AssertionError("unreachable")


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

    async def read_text(self, artifact_id: str, **_kwargs: Any) -> str:
        self.read_artifact_ids.append(artifact_id)
        for artifact in self.created:
            if artifact["artifact_id"] == artifact_id:
                return str(artifact.get("content") or "")
        return ""


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
        "example_document.section_part",
        "example_document.section",
        "example_document.section_part",
        "example_document.section",
        "example_document",
    ]
    assert platform.workpads.final_refs == ["artifact://artifact-6"]
    assert all(item["reasoning_mode"] == "disabled" for item in llm.chat_kwargs)
    event_names = [event["event"] for event in phase_events]
    assert event_names.count("document.part.completed") == 2
    assert event_names.count("document.section.completed") == 2
    assert "document.final.deterministic_assembly_completed" in event_names
    assert "document.final.created" in event_names
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
async def test_sectioned_author_passes_skill_instructions_to_outline_and_writer() -> None:
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
            "final_retention_ratio": 0.8,
        },
        authoring_instructions="Use a decision log and state unresolved risks.",
    )

    await author.author(
        brief={"title": "Example document"},
        upstream={},
    )

    assert "Use a decision log" in llm.structured_prompts[0]
    assert any("Use a decision log" in prompt for prompt in llm.chat_prompts)


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
        request=DocumentAuthoringRequest(
            llm_facade=llm,
            skill_contract=contract,
            artifact_type="example_document",
            step_id="s2",
            capability_id="agent.example.write_document",
            context_budget={},
            brief={"title": "Example document"},
            upstream={},
            authoring_instructions="Follow the selected report skill.",
            workflow_id="workflow-1",
            thread_id="thread-1",
            agent_id="writer",
            mode_metadata={"example_mode": "write", "example_phase": "default"},
            draft_narrative="Draft narrative.",
            draft_narrative_key="_draft_narrative",
            draft_narrative_artifact_type="draft_narrative",
            draft_narrative_summary="Draft before final authoring.",
            document_input={"artifact_access": "summary_then_fetch"},
        ),
    )

    assert "## Context" in result.authoring_result.markdown
    assert result.finalize_strategy == "sectioned_longform"
    assert result.finalize_attempts == 1
    assert result.quality_result.outcome.status == "passed"
    assert result.quality_result.outcome.final_review_passed is True
    assert (
        result.quality_result.outcome.metadata["quality_publication_eligible"]
        is True
    )
    assert result.started_event.metadata["event"] == "sectioned_authoring_started"
    assert result.started_event.metadata["example_mode"] == "write"
    assert result.completed_event.metadata["event"] == "sectioned_authoring_completed"
    assert result.completed_event.metadata["section_count"] == 2
    assert result.authoring_ledger["section_count"] == 2


@pytest.mark.asyncio
async def test_run_sectioned_document_finalization_requires_sectioned_contract() -> None:
    with pytest.raises(RuntimeError, match="skill runtime contract"):
        await run_sectioned_document_finalization(
            request=DocumentAuthoringRequest(
                llm_facade=_FakeLlm(),
                skill_contract=None,
                artifact_type="example_document",
                step_id="s2",
                capability_id="agent.example.write_document",
                context_budget={},
                brief={"title": "Example document"},
                upstream={},
                authoring_instructions="Follow the selected report skill.",
            ),
        )


@pytest.mark.asyncio
async def test_streaming_finalization_surfaces_absolute_deadline(tmp_path) -> None:  # type: ignore[no-untyped-def]
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
---

# Report
""",
        encoding="utf-8",
    )
    contract = SkillContractResolver(root_dir=tmp_path).resolve(
        ["skills/report"],
        required=True,
    )
    llm = _HangingLlm()
    llm.platform_ns = _FakePlatform()
    events = []

    with pytest.raises(DocumentAuthoringDeadlineExceeded) as exc_info:
        async for event in astream_sectioned_document_finalization(
            request=DocumentAuthoringRequest(
                llm_facade=llm,
                skill_contract=contract,
                artifact_type="example_document",
                step_id="s2",
                capability_id="agent.example.write_document",
                context_budget={},
                brief={"title": "Example document"},
                upstream={},
                authoring_instructions="Follow the selected report skill.",
                wall_clock_deadline=asyncio.get_running_loop().time() + 0.01,
            ),
        ):
            events.append(event)

    assert exc_info.value.code == "document_authoring_deadline_exceeded"
    assert any(
        event.metadata.get("event") == "sectioned_authoring_deadline_exceeded"
        for event in events
        if hasattr(event, "metadata")
    )


def test_document_authoring_request_freezes_skill_resolved_inputs() -> None:
    brief = {"title": "Original title"}
    request = DocumentAuthoringRequest(
        llm_facade=None,
        skill_contract=None,
        artifact_type="example_document",
        step_id="s2",
        capability_id="agent.example.write_document",
        context_budget={"max_output_tokens": 1000},
        brief=brief,
        upstream={},
        authoring_instructions="Use the selected skill instructions.",
    )

    brief["title"] = "Mutated title"

    assert request.brief["title"] == "Original title"
    assert request.to_metadata()["document_authoring_request"][
        "skill_instructions_loaded"
    ] is True
    with pytest.raises(TypeError):
        request.brief["title"] = "Blocked mutation"  # type: ignore[index]


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
        context_budget={
            "max_output_tokens": 64000,
            # Profile document caps must not fair-share or shrink per-call tops.
            "max_document_output_tokens": 10000,
        },
        authoring_contract={
            "coverage_model": "example_document",
            "min_outline_sections": 2,
            "max_outline_sections": 2,
            "min_section_words": 5,
            "default_section_words": 5,
            "max_section_words": 20,
            "final_retention_ratio": 0.8,
            "max_document_output_tokens": 10000,
        },
    )

    await author.author(
        brief={"title": "Example document"},
        upstream={},
        workflow_id="workflow-1",
        thread_id="thread-1",
        agent_id="writer",
    )

    # Planned Parts receive a bounded per-call allowance below the provider top;
    # deterministic final assembly makes no third LLM call.
    assert [item["max_output_tokens"] for item in llm.chat_kwargs] == [256, 256]


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

    assert [item["max_output_tokens"] for item in llm.chat_kwargs] == [256, 256]


@pytest.mark.asyncio
async def test_large_planned_part_uses_tight_provider_allowance() -> None:
    platform = _FakePlatform()
    llm = _LongFakeLlm()
    author = SectionedLongformAuthor(
        llm_facade=llm,
        platform=platform,
        artifact_type="example_document",
        step_id="s2",
        capability_id="agent.example.write_document",
        authoring_contract={
            "coverage_model": "example_document",
            "min_outline_sections": 1,
            "max_outline_sections": 1,
            "min_section_words": 300,
            "default_section_words": 300,
            "max_section_words": 320,
        },
    )

    await author.author(
        brief={"title": "Example document"},
        upstream={},
        workflow_id="workflow-1",
        thread_id="thread-1",
        agent_id="writer",
    )

    allowances = [item["max_output_tokens"] for item in llm.chat_kwargs]
    assert allowances
    assert all(allowance <= 3200 for allowance in allowances)


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
            max_document_output_tokens: 12000
            finalization: deterministic_assembly
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
    assert authoring["max_document_output_tokens"] == 12000
    assert authoring["finalization"] == "deterministic_assembly"
    assert authoring["evidence_depth"] == "deep"


@pytest.mark.asyncio
async def test_long_profile_uses_deterministic_assembly() -> None:
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
            "finalization": "deterministic_assembly",
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
    assert result.ledger["finalization"] == "deterministic_assembly"
    assert not any("Merge this cluster" in prompt for prompt in llm.chat_prompts)
    assert any(
        event["event"] == "document.final.deterministic_assembly_completed"
        for event in phase_events
    )
    assert "document.final.merge_cluster_started" not in [
        event["event"] for event in phase_events
    ]


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

    assert result.drafts[0].markdown.startswith(
        "## Context\n\nalpha beta gamma delta epsilon zeta"
    )
    assert "## Wrong Heading" not in result.drafts[0].markdown
    assert result.drafts[0].quality["passed"] is True
    assert result.drafts[0].quality["failures"] == []


@pytest.mark.asyncio
async def test_sectioned_author_isolates_the_requested_section_from_a_whole_document_draft() -> None:
    llm = _WholeDocumentDraftLlm()
    author = SectionedLongformAuthor(
        llm_facade=llm,
        platform=_FakePlatform(),
        artifact_type="example_document",
        step_id="s1",
        capability_id="agent.example.write_document",
        context_budget={"max_output_tokens": 2000},
        authoring_contract={
            "min_outline_sections": 2,
            "max_outline_sections": 2,
            "min_section_words": 5,
            "default_section_words": 5,
            "max_section_words": 20,
            "require_evidence_refs": False,
        },
    )

    result = await author.author(
        brief={"title": "Example document"},
        upstream={},
        workflow_id="workflow-1",
        thread_id="thread-1",
        agent_id="writer",
    )

    assert result.drafts[0].markdown == "## Context\n\nalpha beta gamma delta epsilon zeta"
    assert result.drafts[1].markdown == "## Recommendation\n\neta theta iota kappa lambda mu"
    assert result.markdown.count("## Context") == 1
    assert result.markdown.count("## Recommendation") == 1
    draft_prompts = [
        prompt
        for prompt in llm.chat_prompts
        if "Write one complete bounded part" in prompt
    ]
    assert len(draft_prompts) == 2
    assert all("Cover only the current objective" in prompt for prompt in draft_prompts)


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
    """Default ``degrade`` enforcement records a best-effort section for soft
    failures rather than dead-ending the plan, and reports the degradation
    through structured metadata only — never in the reader-facing markdown."""
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
    # The deliverable stays clean: internal gate vocabulary is reviewer-facing,
    # not reader-facing.
    for draft in result.drafts:
        assert "Evidence gap (auto-flagged)" not in draft.markdown
        assert "missing_confidence_layer" not in draft.markdown

    # ...but the reason is not lost — it must still reach consumers structurally.
    assert any(
        "missing_confidence_layer" in section["failures"]
        for section in result.ledger["degraded_sections"]
    )
    assert "document.section.quality_degraded" in [e["event"] for e in phase_events]
    quality_events = [
        event
        for event in phase_events
        if event["event"] == "document.section.quality_checked"
    ]
    assert quality_events
    assert all(event["status"] == "gate_failed" for event in quality_events)


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

    with pytest.raises(RuntimeError, match="empty_part"):
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
async def test_deferred_intermediate_artifacts_keep_successful_final_in_memory() -> None:
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
        },
        defer_intermediate_artifacts=True,
    )

    result = await author.author(
        brief={"title": "Example document"},
        upstream={},
        workflow_id="workflow-1",
        thread_id="thread-1",
        agent_id="writer",
    )

    assert result.markdown
    assert result.ledger["final_ref"] == {}
    assert result.ledger["artifact_refs"] == []
    assert [
        item["artifact_type"] for item in platform.artifacts.created
    ] == [
        "example_document.section_part",
        "example_document.section_part",
    ]
    assert platform.workpads.entries == []
    assert platform.workpads.final_refs == []


@pytest.mark.asyncio
async def test_deferred_final_keeps_resume_sections_without_committing_final() -> None:
    """Persist resumable drafts while leaving final rendering to the agent."""
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
        },
        defer_final_artifact=True,
    )

    result = await author.author(
        brief={"title": "Example document"},
        upstream={},
        workflow_id="workflow-1",
        thread_id="thread-1",
        agent_id="writer",
    )

    assert [item["artifact_type"] for item in platform.artifacts.created] == [
        "example_document.outline",
        "example_document.section_part",
        "example_document.section",
        "example_document.section_part",
        "example_document.section",
    ]
    assert result.ledger["final_ref"] == {}
    assert platform.workpads.final_refs == []
    assert not any(
        "Polish the concatenated sections" in prompt
        for prompt in author._llm.chat_prompts
    )


@pytest.mark.asyncio
async def test_legacy_resumed_draft_is_revalidated_for_completeness() -> None:
    author = SectionedLongformAuthor(
        llm_facade=_FakeLlm(),
        platform=_FakePlatform(),
        artifact_type="example_document",
        step_id="s2",
        capability_id="agent.example.write_document",
    )
    legacy = SectionDraft(
        plan=SectionPlan(
            section_id="overview",
            title="Overview",
            objective="Provide a complete overview.",
        ),
        markdown="## Overview\n\nComplete overview body.",
        quality={},
    )

    admitted = await author._revalidate_resumed_drafts([legacy])

    assert admitted[0].quality["completeness_review"]["complete"] is True


@pytest.mark.asyncio
async def test_deferred_final_rejects_glued_heading_in_accepted_section() -> None:
    author = SectionedLongformAuthor(
        llm_facade=_FakeLlm(),
        platform=_FakePlatform(),
        artifact_type="example_document",
        step_id="s2",
        capability_id="agent.example.write_document",
        defer_final_artifact=True,
    )
    drafts = [
        SectionDraft(
            plan=SectionPlan(section_id="overview", title="Overview"),
            markdown="## Overview\n\nBody text.### Glued heading",
        )
    ]

    with pytest.raises(
        RuntimeError,
        match="document_final_integrity_invalid:glued_heading",
    ):
        await author._polish_final(brief={"title": "Doc"}, drafts=drafts)


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

    with pytest.raises(RuntimeError, match="empty_part"):
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
        if "Write one complete bounded part" in prompt
    ]
    assert draft_prompts
    assert "evidence from upstream-s1-evidence" in draft_prompts[0]
    assert "stale-s2-section" not in draft_prompts[0]


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
    assert SectionedAuthoringContract.from_mapping({}).finalization == "deterministic_assembly"
    assert SectionedAuthoringContract.from_mapping(None).finalization == "deterministic_assembly"


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
    assert "deterministic_assembly" in message


@pytest.mark.asyncio
async def test_deterministic_finalization_preserves_accepted_sections_without_llm() -> None:
    llm = _FakeLlm()
    events: list[dict[str, Any]] = []
    author = SectionedLongformAuthor(
        llm_facade=llm,
        platform=_FakePlatform(),
        artifact_type="example_document",
        step_id="s1",
        capability_id="agent.example.write_document",
        phase_event_sink=events.append,
    )
    complete = {
        "completeness_review": {
            "complete": True,
            "issue_code": "none",
            "reason": "complete",
        },
        "hard_failures": [],
    }
    drafts = [
        SectionDraft(
            plan=SectionPlan(section_id="context", title="Context"),
            markdown="## Context\n\nalpha beta gamma.",
            quality=complete,
        ),
        SectionDraft(
            plan=SectionPlan(section_id="findings", title="Findings"),
            markdown="## Findings\n\ndelta epsilon zeta.",
            quality=complete,
        ),
    ]

    result = await author._polish_final(brief={"title": "Doc"}, drafts=drafts)

    assert result == "\n\n".join(draft.markdown for draft in drafts)
    assert llm.chat_prompts == []
    assert events[-1]["event"] == "document.final.deterministic_assembly_completed"


# --- output truncation detection (stop_reason) --------------------------------


def test_finish_reason_normalisation() -> None:
    from novie_agent_sdk.sectioned_authoring import _finish_reason_of

    # OpenAI-compatible shape.
    assert _finish_reason_of({"response_metadata": {"finish_reason": "length"}}) == "length"
    # Anthropic-style shape, case-insensitive.
    assert _finish_reason_of({"response_metadata": {"stop_reason": "MAX_TOKENS"}}) == "length"
    assert (
        _finish_reason_of(
            {
                "response_metadata": {
                    "finish_reason": "stop",
                    "stop_reason": "MAX_TOKENS",
                }
            }
        )
        == "length"
    )
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
        if (
            "Write one complete bounded part" in messages[0]["content"]
            or "Continue exactly where it ended" in messages[0]["content"]
        ):
            return {**result, "response_metadata": {"finish_reason": "length"}}
        return dict(result)


class _ContinuingDraftLlm(_FakeLlm):
    def __init__(self) -> None:
        super().__init__()
        self.interrupted = False

    async def chat(
        self,
        *,
        messages: list[dict[str, str]],
        temperature: float,
        **kwargs: Any,
    ) -> dict[str, Any]:
        prompt = messages[0]["content"]
        self.chat_kwargs.append(dict(kwargs))
        self.chat_prompts.append(prompt)
        if "Continue exactly where it ended" in prompt:
            return {
                "content": "ma delta epsilon zeta",
                "response_metadata": {"finish_reason": "stop"},
            }
        if "Write one complete bounded part" in prompt and not self.interrupted:
            self.interrupted = True
            return {
                "content": "## Context\n\nalpha beta gam",
                "response_metadata": {"finish_reason": "length"},
            }
        return await super().chat(
            messages=messages,
            temperature=temperature,
            **kwargs,
        )


class _IncompleteThenCompleteLlm(_FakeLlm):
    def __init__(self) -> None:
        super().__init__()
        self.completeness_calls = 0

    async def structured(
        self,
        *,
        messages: list[dict[str, str]],
        output_schema: dict[str, Any],
        temperature: float,
        **kwargs: Any,
    ) -> dict[str, Any]:
        if output_schema.get("title") != "SectionCompletenessReview":
            return await super().structured(
                messages=messages,
                output_schema=output_schema,
                temperature=temperature,
                **kwargs,
            )
        self.completeness_calls += 1
        if self.completeness_calls == 1:
            return {
                "structured": {
                    "complete": False,
                    "issue_code": "cut_off",
                    "reason": "The second option ends mid-thought.",
                }
            }
        return {
            "structured": {
                "complete": True,
                "issue_code": "none",
                "reason": "The revised section completes its objective.",
            }
        }


@pytest.mark.asyncio
async def test_semantically_incomplete_section_is_revised_before_acceptance() -> None:
    phase_events: list[dict[str, Any]] = []
    llm = _IncompleteThenCompleteLlm()
    author = SectionedLongformAuthor(
        llm_facade=llm,
        platform=_FakePlatform(),
        artifact_type="example_document",
        step_id="s2",
        capability_id="agent.example.write_document",
        authoring_contract={
            "min_outline_sections": 2,
            "max_outline_sections": 2,
            "min_section_words": 5,
            "default_section_words": 5,
            "max_section_words": 20,
            "max_section_revision_rounds": 1,
        },
        defer_final_artifact=True,
        phase_event_sink=phase_events.append,
    )

    result = await author.author(
        brief={"title": "Example document"},
        upstream={},
    )

    reviews = [
        event for event in phase_events
        if event["event"] == "document.section.completeness_reviewed"
    ]
    assert reviews[0]["status"] == "gate_failed"
    assert reviews[0]["issue_code"] == "cut_off"
    assert any(
        "Resolve the remaining completeness issue" in prompt
        for prompt in llm.chat_prompts
    )
    assert any(
        event["event"] == "document.section.recovery_part_planned"
        for event in phase_events
    )
    assert len(result.drafts) == 2
    assert all(
        draft.quality["completeness_review"]["complete"] is True
        for draft in result.drafts
    )


@pytest.mark.asyncio
async def test_truncated_part_continues_before_repartitioning() -> None:
    events: list[dict[str, Any]] = []
    author = SectionedLongformAuthor(
        llm_facade=_ContinuingDraftLlm(),
        platform=_FakePlatform(),
        artifact_type="example_document",
        step_id="s2",
        capability_id="agent.example.write_document",
        authoring_contract={
            "min_outline_sections": 2,
            "max_outline_sections": 2,
            "min_section_words": 5,
            "default_section_words": 5,
            "max_section_words": 20,
        },
        defer_final_artifact=True,
        phase_event_sink=events.append,
    )

    result = await author.author(
        brief={"title": "Example document"},
        upstream={},
    )

    assert "alpha beta gamma delta epsilon zeta" in result.markdown
    event_names = [event["event"] for event in events]
    assert "document.part.continuation_requested" in event_names
    assert "document.part.continuation_completed" in event_names
    assert "document.part.repartitioned" not in event_names


@pytest.mark.asyncio
async def test_truncated_planned_part_fails_closed_when_it_cannot_be_split() -> None:
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

    with pytest.raises(RuntimeError, match="document_part_completion_exhausted"):
        await author.author(
            brief={"title": "Example document"},
            upstream={},
            workflow_id="workflow-1",
            thread_id="thread-1",
            agent_id="writer",
        )

    event_names = [event["event"] for event in phase_events]
    assert event_names.count("document.part.started") == 1
    assert "document.part.completed" not in event_names


@pytest.mark.asyncio
async def test_fixed_shape_outline_uses_canonical_section_titles_without_llm() -> None:
    phase_events: list[dict[str, Any]] = []
    llm = _FakeLlm()
    author = SectionedLongformAuthor(
        llm_facade=llm,
        platform=_FakePlatform(),
        artifact_type="requirements_analysis",
        step_id="s1",
        capability_id="agent.analyst.requirements_analysis",
        authoring_contract={
            "length_profile": "short",
            "default_section_words": 20,
            "canonical_section_titles": (
                "Summary",
                "Personas",
                "Goals",
            ),
        },
        phase_event_sink=phase_events.append,
    )

    _profile, outline = await author._build_outline(
        brief={"title": "Support triage"},
        upstream={},
    )

    assert [plan.title for plan in outline] == ["Summary", "Personas", "Goals"]
    assert llm.structured_prompts == []
    assert any(event["event"] == "document.outline.fixed_shape" for event in phase_events)


@pytest.mark.asyncio
async def test_terminal_output_byte_contract_does_not_compact_canonical_document() -> None:
    phase_events: list[dict[str, Any]] = []
    author = SectionedLongformAuthor(
        llm_facade=_FakeLlm(),
        platform=_FakePlatform(),
        artifact_type="example_document",
        step_id="s1",
        capability_id="agent.example.write_document",
        context_budget={
            "max_output_tokens": 1000,
            "max_document_output_bytes": 128,
        },
        authoring_contract={"finalization": "deterministic_assembly"},
        phase_event_sink=phase_events.append,
    )
    drafts = [
        SectionDraft(
            plan=SectionPlan(section_id="context", title="Context"),
            markdown="## Context\n\n" + "Evidence detail. " * 20,
            quality={
                "completeness_review": {
                    "complete": True,
                    "issue_code": "none",
                    "reason": "complete",
                },
                "hard_failures": [],
            },
        ),
        SectionDraft(
            plan=SectionPlan(section_id="findings", title="Findings"),
            markdown="## Findings\n\n" + "Recommendation detail. " * 20,
            quality={
                "completeness_review": {
                    "complete": True,
                    "issue_code": "none",
                    "reason": "complete",
                },
                "hard_failures": [],
            },
        ),
    ]

    result = await author._polish_final(brief={"title": "Doc"}, drafts=drafts)

    assert len(result.encode("utf-8")) > 128
    assert "Evidence detail." in result
    assert "Recommendation detail." in result
    assert not any(
        event["event"].startswith("document.final.output_byte")
        for event in phase_events
    )


class _ScaffoldLeakingLlm(_FakeLlm):
    """A model that copies its artifact-read tool observation into the draft."""

    def __init__(self) -> None:
        super().__init__()
        self.leaked: list[str] = []

    async def chat(
        self,
        *,
        messages: list[dict[str, str]],
        temperature: float,
        **kwargs: Any,
    ) -> dict[str, str]:
        result = await super().chat(
            messages=messages, temperature=temperature, **kwargs
        )
        content = result.get("content") or ""
        if content.startswith("## "):
            heading, _, body = content.partition("\n\n")
            content = (
                f"{heading}\n\n"
                "[artifact art-906227eab30943de] mode=chunks\n\n"
                f"{body}"
            )
            self.leaked.append(content)
        return {**result, "content": content}


@pytest.mark.asyncio
async def test_recorded_sections_never_carry_artifact_read_scaffolding() -> None:
    """End-to-end: the egress seam holds through the real authoring path.

    Prompt wording cannot guarantee a model won't echo its tool observation, so
    what actually protects the deliverable is the scrub on the write path. This
    drives the full author -> record flow with a model that always leaks.
    """
    platform = _FakePlatform()
    llm = _ScaffoldLeakingLlm()
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

    # The model really did leak, so this test would pass vacuously otherwise.
    assert llm.leaked
    assert all("[artifact art-906227eab30943de]" in text for text in llm.leaked)

    # Scrubbed at acceptance, so the in-memory drafts are clean as well — the
    # quality gate and the final merge never see the scaffolding.
    for draft in result.drafts:
        assert "[artifact art-906227eab30943de]" not in draft.markdown
    assert "[artifact art-906227eab30943de]" not in result.markdown

    markdown_artifacts = [
        entry
        for entry in platform.artifacts.created
        if str(entry.get("content_type") or "").startswith("text/")
    ]
    assert markdown_artifacts
    for entry in markdown_artifacts:
        assert "[artifact art-906227eab30943de]" not in str(entry.get("content") or "")
        assert "mode=chunks" not in str(entry.get("content") or "")
        assert "[artifact art-906227eab30943de]" not in str(entry.get("summary") or "")


def _structure_drafts() -> list[Any]:
    from types import SimpleNamespace

    return [
        SimpleNamespace(plan=SimpleNamespace(title="Context"), markdown=""),
        SimpleNamespace(plan=SimpleNamespace(title="Findings"), markdown=""),
    ]


def test_structure_guard_rejects_a_glued_level_two_heading() -> None:
    assert _section_structure_violation(
        "## Context\n\nBody text.## Findings\n\nMore.",
        _structure_drafts(),
    ) == "glued_heading"


def test_structure_guard_rejects_a_glued_subheading() -> None:
    """The outline check alone misses this: the level-two list is unchanged.

    A glued `### Detail` renders as body text, which is exactly the "heading
    lost its line break" defect readers report.
    """
    glued = "## Context\n\nBody text.### Detail\n\n## Findings\n\nMore."

    # The level-two outline still looks perfect, which is why the outline
    # comparison on its own cannot catch this.
    assert [
        m.group(1)
        for m in _HEADING_RE.finditer(glued)
        if len(m.group(0)) - len(m.group(0).lstrip("#")) == 2
    ] == ["Context", "Findings"]

    assert _section_structure_violation(glued, _structure_drafts()) == "glued_heading"


def test_structure_guard_accepts_well_formed_markdown() -> None:
    assert _section_structure_violation(
        "## Context\n\nBody text.\n\n### Detail\n\nMore.\n\n## Findings\n\nEnd.",
        _structure_drafts(),
    ) is None


def test_structure_guard_tolerates_prose_that_mentions_hashes() -> None:
    """A false positive costs the polish, so the pattern stays conservative."""
    for body in (
        "## Context\n\nWe use C# and F# heavily.\n\n## Findings\n\nThe ## marker is syntax.",
        "## Context\n\n| col | ## odd |\n| --- | --- |\n\n## Findings\n\nSee issue # 42.",
    ):
        assert _section_structure_violation(body, _structure_drafts()) is None, body
