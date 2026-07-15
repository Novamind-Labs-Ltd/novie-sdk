from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

from pydantic import BaseModel, Field

from novie_agent_sdk import (
    bounded_handoff_output,
    bounded_subtask_evidence_text,
    bounded_workpad_text,
    capability_provides_artifacts,
    content_delta_event,
    document_final_event,
    document_final_output,
    execution_workpad_entries,
    execution_workpad_context,
    get_matching_document_checkpoint,
    latest_workpad_entry,
    markdown_deliverable_output,
    progress_event,
    skipped_phase_events,
    skipped_quality_result,
    tool_call_event,
    tool_result_event,
    upstream_context,
    workpad_entries_by_kind,
    workpad_checkpoint_event,
    workpad_entry_event,
)


def test_workpad_entry_event_shape() -> None:
    event = workpad_entry_event(
        kind="outline",
        title="Outline",
        content="# Outline",
        base_metadata={"run_id": "run-1"},
        metadata={"step": "s1"},
    )

    assert event.kind == "trace"
    assert event.metadata["run_id"] == "run-1"
    assert event.metadata["event"] == "execution_workpad_entry"
    assert event.metadata["workpad_entry"] == {
        "kind": "outline",
        "title": "Outline",
        "content": "# Outline",
        "content_type": "text/markdown",
        "artifact_type": "execution_workpad.outline",
        "metadata": {"step": "s1"},
    }

    checkpoint = workpad_checkpoint_event(
        kind="final_report",
        title="Final",
        content="# Final",
    )
    assert checkpoint.metadata["workpad_entry"]["kind"] == "final_report"


def test_execution_workpad_context_reads_direct_or_nested_payload() -> None:
    compact = {
        "version": "execution_workpad.compact.v1",
        "entries": [{"kind": "outline"}],
    }

    assert execution_workpad_context({"execution_workpad": compact}) == compact
    assert execution_workpad_context({"platform_context": {"execution_workpad": compact}}) == compact
    assert execution_workpad_context({"platform_context": {}}) == {}


def test_workpad_entry_index_helpers() -> None:
    workpad = {
        "entries": [
            {"kind": "outline", "content": "a"},
            {"kind": "draft", "content": "b"},
            {"kind": "draft", "content": "c"},
            "ignored",
        ]
    }

    assert len(execution_workpad_entries(workpad)) == 3
    assert [entry["content"] for entry in workpad_entries_by_kind(workpad, "draft")] == ["b", "c"]
    assert latest_workpad_entry(workpad, kind="draft") == {"kind": "draft", "content": "c"}
    assert bounded_workpad_text("abcdef", limit=3).startswith("abc")


def test_subtask_evidence_text_preserves_long_body_by_default() -> None:
    evidence = "status: complete\n\n## Evidence\n" + ("A" * 9000)

    content, truncated = bounded_subtask_evidence_text(evidence)

    assert truncated is False
    assert content == evidence


def test_subtask_evidence_text_reports_explicit_truncation() -> None:
    content, truncated = bounded_subtask_evidence_text("A" * 30, limit=20)

    assert truncated is True
    assert content == ("A" * 20) + "\n\n[truncated]"


def test_upstream_context_reads_direct_or_nested_payload() -> None:
    context = {
        "version": "upstream_context.v1",
        "direct_dependencies": [{"step_id": "s0"}],
    }

    assert upstream_context({"upstream_context": context}) == context
    assert upstream_context({"platform_context": {"upstream_context": context}}) == context
    assert upstream_context({"platform_context": {}}) == {}


def test_markdown_deliverable_output_shape() -> None:
    out = markdown_deliverable_output(
        title="Final Report",
        markdown="# Final Report",
        artifact_type="management_report",
        artifact_family="document",
        metadata={"phase": "final"},
        provides_artifacts={"final": {"artifact_type": "management_report"}},
    )

    assert out["kind"] == "document_deliverable"
    assert out["title"] == "Final Report"
    assert out["analysis"] == "# Final Report"
    assert out["final_markdown"] == "# Final Report"
    assert out["content"] == "# Final Report"
    assert out["artifact_family"] == "document"
    assert out["metadata"] == {"phase": "final"}
    assert out["provides_artifacts"]["final"]["artifact_type"] == "management_report"


def test_bounded_handoff_output_shape() -> None:
    out = bounded_handoff_output(
        handoff_markdown="# Evidence Handoff\n\nFacts.",
        artifact_type="research_dossier",
        artifact_family="document",
        capability_id="agent.analyst.market_research",
        output_contract={"kind": "bounded_handoff", "max_bytes": 4096},
        metadata={"analysis_phase": "market_map"},
        provides_artifacts={"research_dossier": {"structured_output": {"kind": "bounded_handoff"}}},
        degraded_flags=["web.no_results"],
        budget_summary={"estimated_input_tokens": 100},
        recovery={"checkpoint_id": "ckpt-1"},
    )

    assert out["kind"] == "bounded_handoff"
    assert out["analysis"].startswith("# Evidence Handoff")
    assert out["output_visibility"] == "internal"
    assert out["step_role"] == "upstream_handoff"
    assert out["metadata"]["analysis_phase"] == "market_map"
    assert out["metadata"]["output_visibility"] == "internal"
    assert out["final_payload"]["final_markdown"].startswith("# Evidence Handoff")
    assert out["final_payload"]["structured_output"]["kind"] == "bounded_handoff"
    assert out["final_payload"]["degraded_flags"] == ["web.no_results"]
    assert out["final_payload"]["recovery"]["checkpoint_id"] == "ckpt-1"
    assert out["provides_artifacts"]["research_dossier"]["structured_output"]["kind"] == "bounded_handoff"


def test_document_final_helpers_shape() -> None:
    entry = SimpleNamespace(
        capability_id="agent.demo.write",
        provides=("demo_report",),
    )
    provided = capability_provides_artifacts(
        capability_manifest=(entry,),
        capability_id="agent.demo.write",
        artifact_type="demo_markdown",
        structured_output={"summary": "done"},
    )
    assert set(provided) == {"demo_report", "demo_markdown"}

    output = document_final_output(
        artifact_type="demo_markdown",
        artifact_family="document",
        capability_id="agent.demo.write",
        analysis="# Done",
        narrative="Draft notes",
        structured_output={"summary": "done"},
        final_payload={"final_markdown": "# Done"},
        capability_manifest=(entry,),
        mode_key="demo_mode",
        mode="write",
        phase_key="demo_phase",
        phase="final",
        checkpoint_id="ckpt-1",
        quality={"quality_status": "skipped"},
        resumed_from_checkpoint=True,
    )
    event = document_final_event(output=output, metadata={"event": "done"})

    assert event.kind == "final"
    assert event.output["content"] == "# Done"
    assert event.output["demo_mode"] == "write"
    assert event.output["resumed_from_checkpoint"] is True
    assert event.output["quality_status"] == "skipped"
    assert event.metadata["event"] == "done"


def test_stream_event_helpers_shape() -> None:
    progress = progress_event(
        "Writing section",
        runtime_phase="writing",
        capability_id="agent.analyst.report_synthesis",
        phase="final",
    )
    call = tool_call_event("fetch_artifact", args={"artifact_id": "a1"})
    result = tool_result_event("fetch_artifact", result="done", ok=True)
    content = content_delta_event("hello")

    assert progress.kind == "trace"
    assert progress.metadata["runtime_phase"] == "writing"
    assert progress.metadata["progress_label"] == "Writing section"
    assert call.metadata["event"] == "tool_call"
    assert call.metadata["tool_args"] == {"artifact_id": "a1"}
    assert result.content == "done"
    assert result.metadata["tool_ok"] is True
    assert result.metadata["tool_result_visibility"] == "internal"
    assert result.metadata["tool_result_chars"] == len("done")
    assert content.kind == "content"
    assert content.content == "hello"


def test_document_resume_and_quality_helpers() -> None:
    class _Payload(BaseModel):
        current_phase: str
        narrative: str
        phase_outputs: dict[str, Any] = Field(default_factory=dict)
        metadata: dict[str, Any] = Field(default_factory=dict)

    class _Service:
        async def get(self, _ctx, **_kwargs):
            return SimpleNamespace(
                checkpoint_id="ckpt-1",
                payload={
                    "current_phase": "finalize",
                    "narrative": "Recovered draft",
                    "phase_outputs": {
                        "finalize": {"checkpoint_version": 2, "narrative_chars": 15}
                    },
                    "metadata": {
                        "input_digest": "digest",
                        "capability_id": "agent.demo.write",
                        "checkpoint_version": 2,
                    },
                },
            )

    ctx = SimpleNamespace(thread_id="thread-1", workflow_id="")
    candidate = asyncio.run(
        get_matching_document_checkpoint(
            _Service(),
            ctx,
            payload_model=_Payload,
            owner_agent_id="demo",
            capability_id="agent.demo.write",
            input_digest="digest",
            step_id="",
        )
    )

    assert candidate is not None
    assert candidate.payload.narrative == "Recovered draft"

    class _LegacyService:
        async def get(self, _ctx, **_kwargs):
            return SimpleNamespace(
                checkpoint_id="legacy-ckpt",
                payload={
                    "current_phase": "finalize",
                    "narrative": "RAW_SECRET_USER_PROMPT",
                    "phase_outputs": {"draft": {"narrative_chars": 24}},
                    "metadata": {
                        "input_digest": "digest",
                        "capability_id": "agent.demo.write",
                    },
                },
            )

    assert asyncio.run(
        get_matching_document_checkpoint(
            _LegacyService(),
            ctx,
            payload_model=_Payload,
            owner_agent_id="demo",
            capability_id="agent.demo.write",
            input_digest="digest",
            step_id="",
        )
    ) is None

    class _PreFixFinalService:
        async def get(self, _ctx, **_kwargs):
            return SimpleNamespace(
                checkpoint_id="pre-fix-final-ckpt",
                payload={
                    "current_phase": "finalize",
                    "narrative": "RAW_SECRET_USER_OR_SKILL_PROMPT",
                    "phase_outputs": {
                        "finalize": {"checkpoint_version": 1, "narrative_chars": 31}
                    },
                    "metadata": {
                        "input_digest": "digest",
                        "capability_id": "agent.demo.write",
                    },
                },
            )

    assert asyncio.run(
        get_matching_document_checkpoint(
            _PreFixFinalService(),
            ctx,
            payload_model=_Payload,
            owner_agent_id="demo",
            capability_id="agent.demo.write",
            input_digest="digest",
            step_id="",
        )
    ) is None

    def _phase_metadata(**kwargs):
        return dict(kwargs)

    events = skipped_phase_events(
        skipped_phases=("draft",),
        phase_metadata=_phase_metadata,
        mode="write",
        phase="final",
        capability_id="agent.demo.write",
        checkpoint_id="ckpt-1",
    )
    quality = skipped_quality_result("Recovered draft", reason="checkpoint_resume")

    assert events[0].metadata["resumed_from_checkpoint"] is True
    assert quality.outcome.as_metadata()["quality_loop_skipped"] is True
