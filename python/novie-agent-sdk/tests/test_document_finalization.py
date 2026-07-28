from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel, Field

from novie_agent_sdk import (
    PublicAgentError,
    authoring_ledger_from_checkpoint,
    build_document_deliverable_event,
    build_document_finalization_manifest,
    task_brief_document_identity,
)


class _Recovery(BaseModel):
    fallback_used: bool = False
    fallback_reason: str = ""
    resumed_from_checkpoint: bool = False
    checkpoint_id: str = ""
    finalize_attempts: int = 1
    metadata: dict[str, Any] = Field(default_factory=dict)


class _FinalPayload(BaseModel):
    plan_id: str
    final_markdown: str
    structured_output: dict[str, Any] = Field(default_factory=dict)
    degraded_flags: list[str] = Field(default_factory=list)
    recovery: _Recovery = Field(default_factory=_Recovery)
    metadata: dict[str, Any] = Field(default_factory=dict)


def _ledger() -> dict[str, Any]:
    return {
        "outline_ref": {
            "artifact_ref": "artifact://outline",
            "artifact_type": "example_document.outline",
            "content_sha256": "outline-sha",
            "bytes": 100,
        },
        "artifact_refs": [
            {
                "role": "section_draft",
                "section_id": "overview",
                "artifact_ref": "artifact://section-1",
                "artifact_type": "example_document.section",
                "content_sha256": "section-1-sha",
                "bytes": 70000,
                "artifact": {
                    "metadata": {
                        "section_id": "overview",
                        "section_index": 1,
                        "section_title": "Overview",
                        "quality": {"passed": True},
                    },
                },
            },
            {
                "role": "section_draft",
                "section_id": "recommendation",
                "artifact_ref": "artifact://section-2",
                "artifact_type": "example_document.section",
                "content_sha256": "section-2-sha",
                "bytes": 80000,
                "metadata": {
                    "section_id": "recommendation",
                    "section_index": 2,
                    "section_title": "Recommendation",
                    "quality": {"passed": True},
                },
            },
        ],
        "degraded_sections": [],
    }


def test_manifest_orders_durable_section_refs_without_bodies() -> None:
    manifest = build_document_finalization_manifest(
        artifact_type="example_document",
        authoring_ledger=_ledger(),
    )

    assert manifest["version"] == "document-finalization.v1"
    assert [item["section_id"] for item in manifest["sections"]] == [
        "overview",
        "recommendation",
    ]
    assert manifest["sections"][0]["content_sha256"] == "section-1-sha"
    assert all(
        "content" not in section and "markdown" not in section
        for section in manifest["sections"]
    )


def test_document_event_transports_manifest_instead_of_large_body() -> None:
    large_body = "# Report\n\n" + ("complete evidence " * 10000)
    event = build_document_deliverable_event(
        card=None,
        structured={"summary": "Complete report", "body": large_body},
        document_title="Example report",
        artifact_type="example_document",
        artifact_family="document",
        capability_id="agent.example.write",
        analysis=large_body,
        narrative=large_body,
        final_payload_type=_FinalPayload,
        recovery_type=_Recovery,
        authoring_ledger=_ledger(),
    )

    assert event.output["finalization_manifest"]["kind"] == (
        "document_finalization_manifest"
    )
    assert large_body not in str(event.output)
    assert len(str(event.output)) < 10000


def test_checkpoint_rebuilds_manifest_ready_authoring_ledger() -> None:
    checkpoint = {
        "outline_ref": _ledger()["outline_ref"],
        "drafts": [
            {
                "plan": {"section_id": "overview"},
                "artifact_ref": _ledger()["artifact_refs"][0],
            },
            {
                "plan": {"section_id": "recommendation"},
                "artifact_ref": _ledger()["artifact_refs"][1],
            },
        ],
    }

    restored = authoring_ledger_from_checkpoint(checkpoint)
    manifest = build_document_finalization_manifest(
        artifact_type="example_document",
        authoring_ledger=restored,
    )

    assert [item["section_id"] for item in manifest["sections"]] == [
        "overview",
        "recommendation",
    ]


def test_document_event_maps_invalid_manifest_to_repairable_public_failure() -> None:
    with pytest.raises(PublicAgentError) as raised:
        build_document_deliverable_event(
            card=None,
            structured={},
            document_title="Example report",
            artifact_type="example_document",
            artifact_family="document",
            capability_id="agent.example.write",
            analysis="",
            narrative="",
            final_payload_type=_FinalPayload,
            recovery_type=_Recovery,
            authoring_ledger={"enabled": True},
        )

    assert raised.value.error_code == "document_finalization_manifest_invalid"
    assert raised.value.repair_eligible is True


def test_document_event_uses_canonical_title_without_summary_fallback_or_truncation() -> None:
    document_title = "香港高端健身房会员管理系统产品需求文档" * 8
    long_summary = "这是一段用于验证摘要不会覆盖标题的详细产品需求说明。" * 40

    event = build_document_deliverable_event(
        card=None,
        structured={"summary": long_summary},
        document_title=f"  {document_title}  ",
        title_source="task_brief",
        source_brief_id="brief-114",
        artifact_type="prd_document",
        artifact_family="document",
        capability_id="agent.pm.prd_create",
        analysis="# PRD",
        narrative="PRD narrative",
        final_payload_type=_FinalPayload,
        recovery_type=_Recovery,
    )

    assert event.output["document_title"] == document_title
    assert event.output["title"] == document_title
    assert event.output["summary"] == long_summary
    assert event.output["title_source"] == "task_brief"
    assert event.output["source_brief_id"] == "brief-114"
    assert len(event.output["title"]) > 120
    assert event.metadata["document_title"] == document_title
    assert event.metadata["title_source"] == "task_brief"
    assert event.metadata["source_brief_id"] == "brief-114"


@pytest.mark.parametrize("document_title", ["", "   ", "\n\t"])
def test_document_event_rejects_blank_document_title(document_title: str) -> None:
    with pytest.raises(ValueError, match="document_title is required"):
        build_document_deliverable_event(
            card=None,
            structured={"summary": "Must not become the title"},
            document_title=document_title,
            artifact_type="example_document",
            artifact_family="document",
            capability_id="agent.example.write",
            analysis="# Report",
            narrative="Report",
            final_payload_type=_FinalPayload,
            recovery_type=_Recovery,
        )


def test_task_brief_document_identity_trims_title_and_preserves_id() -> None:
    assert task_brief_document_identity(
        {"title": "  香港健身房 PRD  ", "brief_id": " brief-114 "},
        capability_id="agent.pm.prd_create",
    ) == ("香港健身房 PRD", "brief-114")


def test_task_brief_document_identity_reports_lineage_when_title_is_missing() -> None:
    with pytest.raises(ValueError) as raised:
        task_brief_document_identity(
            {"brief_id": "brief-114", "summary": "Long generated summary"},
            capability_id="agent.pm.prd_create",
            workflow_id="workflow-114",
            step_id="step-final",
        )

    message = str(raised.value)
    assert "task_brief_document_title_missing" in message
    assert "workflow_id=workflow-114" in message
    assert "step_id=step-final" in message
    assert "brief_id=brief-114" in message
    assert "capability_id=agent.pm.prd_create" in message
