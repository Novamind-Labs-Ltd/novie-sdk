from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel, Field

from novie_agent_sdk import (
    PublicAgentError,
    authoring_ledger_from_checkpoint,
    build_document_deliverable_event,
    build_document_finalization_manifest,
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
